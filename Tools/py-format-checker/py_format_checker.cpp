// py_format_checker.cpp
//
// Clang plugin that extracts every call to CPython's custom-format functions,
// parses the format string, and type-checks each argument against its spec.
// Both standard specs (%s, %d, %lu …) and CPython-specific specs
// (%R, %S, %A, %U, %T, %N, %V) are understood.
//
// Output per call-site (blocks separated by [py-fmt] sentinel lines):
//   [py-fmt] func:     FunctionName
//   [py-fmt] loc:      file:line
//   [py-fmt] fmt:      "format string"
//   [py-fmt] arg[0] %R     ok      PyObject *
//   [py-fmt] arg[1] %.200s MISMATCH  got=PyObject *  want=<char-ptr>
//
//   [py-fmt]
//   [py-fmt] hint:  "fixed format string"
//
// Mismatches are also emitted as real Clang compiler warnings so they appear
// inline in a normal build.
//
// Build:
//   cd Tools/py-format-checker && mkdir -p build && cd build
//   cmake .. && make -j$(nproc)
//
// Use (whole codebase via compile_commands.json, no rebuild needed):
//   python3 Tools/py-format-checker/run_checker.py

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace clang;

// --------------------------------------------------------------------------
// Integer type-checking strictness (PY_FMT_INTEGRAL_CHECK_MODE):
//   off      – accept any integer for integer specs; no width/sign check.
//   standard – enforce bit-width only (C99: integer promotions mask sign;
//              useful when the codebase freely mixes int/unsigned).
//   full     – enforce both bit-width and signedness.
// Default: off.
// --------------------------------------------------------------------------
enum class IntegralCheckMode { Off, Standard, Full };

// --------------------------------------------------------------------------
// Table: function-name -> {0-based format-string arg index, optional
//   filename substring filter}.  The filter (if non-empty) is matched
//   against the source-file path of the call-site; calls in other files
//   are silently ignored.  Use this for static/file-local helpers that
//   share a common name with unrelated functions in other translation units.
// --------------------------------------------------------------------------
struct FmtFuncInfo {
  unsigned fmtArgIdx;
  const char *fileFilter; // nullptr or "" means "match any file"
};

static const std::map<std::string, FmtFuncInfo> kFormatFuncs = {
    // format arg index 0 — no file constraint needed (unique public names)
    {"_PyErr_FormatNote", {0, nullptr}},
    {"PyUnicode_FromFormat", {0, nullptr}},
    {"PySys_FormatStdout", {0, nullptr}},
    {"PySys_FormatStderr", {0, nullptr}},
    // format arg index 1
    {"PyErr_Format", {1, nullptr}},
    {"_PyErr_FormatFromCause", {1, nullptr}},
    {"_Py_FatalErrorFormat", {1, nullptr}},
    {"PyUnicodeWriter_Format", {1, nullptr}},            // (writer, fmt, ...)
    {"PyBytesWriter_Format", {1, nullptr}},              // (writer, fmt, ...)
    {"_PyXIData_FormatNotShareableError", {1, nullptr}}, // (tstate, fmt, ...)
    {"_abiinfo_raise", {1, "modsupport.c"}},    // static in modsupport.c
    {"_PyTokenizer_syntaxerror", {1, nullptr}}, // (tok, fmt, ...)
    // format arg index 2
    {"_PyErr_Format", {2, nullptr}},
    {"_PyErr_FormatFromCauseTstate", {2, nullptr}},
    {"PyErr_WarnFormat", {2, nullptr}},         // (category, level, fmt, ...)
    {"PyErr_ResourceWarning", {2, nullptr}},    // (source, level, fmt, ...)
    {"_PyCompile_Error", {2, nullptr}},         // (compiler, loc, fmt, ...)
    {"_PyCompile_Warn", {2, nullptr}},          // (compiler, loc, fmt, ...)
    {"_PyTokenizer_parser_warn", {2, nullptr}}, // (tok, category, fmt, ...)
    // format arg index 3
    {"task_set_error_soon",
     {3, "_asynciomodule.c"}}, // static in _asynciomodule.c
    {"format_notshareableerror",
     {3, "crossinterp"}}, // static in crossinterp*.h/.c
    {"_PyTokenizer_syntaxerror_known_range",
     {3, nullptr}}, // (tok, col, end_col, fmt, ...)
    // format arg index 5
    {"PyErr_WarnExplicitFormat",
     {5, nullptr}}, // (cat, file, line, mod, reg, fmt, ...)
};

// ==========================================================================
// Format-string parser
// ==========================================================================

// One spec entry: the textual token (e.g. "%.200s") and the sequence of
// expected-type sentinels for the C arguments it consumes.  Most specs
// consume one argument; %V consumes two; %% consumes none.
//
// Sentinels:
//   <any-int>              any integer type (for %c only)
//   <int> / <uint>         signed / unsigned int        (%d %i / %u %x %X %o)
//   <long> / <ulong>       signed / unsigned long       (%ld / %lu %lx %lX %lo)
//   <longlong> / <ulonglong>  …long long               (%lld / %llu %llx …)
//   <ssize_t> / <size_t>   Py_ssize_t / size_t          (%zd / %zu)
//   <intmax_t> / <uintmax_t>  intmax_t / uintmax_t      (%jd / %ju)
//   <ptrdiff_t>            ptrdiff_t                    (%td)
//   <char-ptr>             char * or const char *       (%s)
//   <wchar-ptr>            wchar_t *                    (%ls, second arg of
//   %lV) <any-ptr>              any pointer                  (%p) <PyObject*>
//   PyObject * or any Py*-typed pointer (%R %S %A %U %T %#T) <PyTypeObject*>
//   PyTypeObject * specifically  (%N %#N) <unknown>              unrecognised
//   spec – arg is counted but not checked
struct FmtArgSpec {
  std::string text;                      // textual token, e.g. "%.200s"
  std::vector<std::string> expectedArgs; // one sentinel per consumed arg
  size_t fmtOffset =
      std::string::npos; // position of '%' in fmtStr; npos for synthetic specs
};

static std::vector<FmtArgSpec> parseFmtString(const std::string &fmt) {
  std::vector<FmtArgSpec> result;
  size_t i = 0;
  while (i < fmt.size()) {
    if (fmt[i] != '%') {
      ++i;
      continue;
    }
    size_t start = i++;
    if (i >= fmt.size())
      break;

    // Flags: -, 0, #  (CPython only recognises these three; '+' and ' '
    // are not consumed and would make the spec char unrecognised/invalid).
    // '#' is also the modifier in %#T and %#N (Python 3.13+); consuming
    // it here is correct because it precedes the conversion specifier.
    while (i < fmt.size() &&
           std::string("-#0").find(fmt[i]) != std::string::npos)
      ++i;

    // Width: decimal digits, or '*' (next arg is int)
    if (i < fmt.size() && fmt[i] == '*') {
      FmtArgSpec wfa;
      wfa.text = "%*";
      wfa.expectedArgs = {"<int>"};
      result.push_back(std::move(wfa));
      ++i;
    } else {
      while (i < fmt.size() && isdigit(fmt[i]))
        ++i;
    }

    // Precision: decimal digits, or '*' (next arg is int)
    if (i < fmt.size() && fmt[i] == '.') {
      ++i;
      if (i < fmt.size() && fmt[i] == '*') {
        FmtArgSpec pfa;
        pfa.text = "%.*";
        pfa.expectedArgs = {"<int>"};
        result.push_back(std::move(pfa));
        ++i;
      } else {
        while (i < fmt.size() && isdigit(fmt[i]))
          ++i;
      }
    }

    // Length modifier: l, ll, z, j, t.
    // Note: 'h' is NOT supported by CPython's format parser — %hd etc. are
    // invalid and would raise SystemError at runtime.  Likewise, 'L' (long
    // double) is not supported. We intentionally omit both so that %h<x> or
    // %L<x> hit the default/UNKNOWN_SPEC path rather than being silently
    // accepted.
    enum class Len { None, l, ll, z, j, t } lm = Len::None;
    if (i < fmt.size()) {
      if (fmt[i] == 'l') {
        if (i + 1 < fmt.size() && fmt[i + 1] == 'l') {
          lm = Len::ll;
          i += 2;
        } else {
          lm = Len::l;
          ++i;
        }
      } else if (fmt[i] == 'z') {
        lm = Len::z;
        ++i;
      } else if (fmt[i] == 'j') {
        lm = Len::j;
        ++i;
      } else if (fmt[i] == 't') {
        lm = Len::t;
        ++i;
      }
    }

    if (i >= fmt.size())
      break;
    char spec = fmt[i++];
    FmtArgSpec fa;
    fa.text = fmt.substr(start, i - start);
    fa.fmtOffset = start;

    // Build width-specific integer sentinels using the length modifier.
    // Sentinels encode both the expected C type and its size so that
    // e.g. %u (→ <uint32>) rejects an unsigned long long argument.
    auto intSentinel = [&](bool isSigned) -> std::string {
      switch (lm) {
      case Len::l:
        return isSigned ? "<long>" : "<ulong>";
      case Len::ll:
        return isSigned ? "<longlong>" : "<ulonglong>";
      case Len::z:
        return isSigned ? "<ssize_t>" : "<size_t>";
      case Len::j:
        return isSigned ? "<intmax_t>" : "<uintmax_t>";
      case Len::t:
        return "<ptrdiff_t>";
      default:
        return isSigned ? "<int>" : "<uint>";
      }
    };

    switch (spec) {
    case '%':
      break; // literal %, no argument consumed

    case 'c':
      fa.expectedArgs = {"<any-int>"};
      break;

    case 'd':
    case 'i':
      fa.expectedArgs = {intSentinel(true)};
      break;

    case 'u':
      fa.expectedArgs = {intSentinel(false)};
      break;

    case 'x':
      fa.expectedArgs = {intSentinel(false)};
      break;

    case 'X':
    case 'o':
      fa.expectedArgs = {intSentinel(false)};
      break;

    case 's':
      fa.expectedArgs = {lm == Len::l ? "<wchar-ptr>" : "<char-ptr>"};
      break;

    case 'p':
      fa.expectedArgs = {"<any-ptr>"};
      break;

    // CPython-specific specs (%R %S %A %U %T %#T): expect PyObject *
    case 'R':
    case 'S':
    case 'A':
    case 'U':
    case 'T':
      fa.expectedArgs = {"<PyObject*>"};
      break;

    case 'N':
      fa.expectedArgs = {"<PyTypeObject*>"};
      break;

    // %V: TWO arguments – PyObject * (nullable) then char*/wchar_t* (fallback)
    // %lV uses const wchar_t* as the fallback instead of const char*.
    case 'V':
      fa.expectedArgs = {"<PyObject*>",
                         lm == Len::l ? "<wchar-ptr>" : "<char-ptr>"};
      break;

    default:
      fa.expectedArgs = {"<unknown>"};
      break;
    }

    result.push_back(std::move(fa));
  }
  return result;
}

// ==========================================================================
// Type-compatibility helpers
// ==========================================================================

// True if qt is a pointer to char (const or mutable).
static bool isCharPtr(QualType qt) {
  QualType can = qt.getCanonicalType();
  if (!can->isPointerType())
    return false;
  return can->getPointeeType().getUnqualifiedType()->isCharType();
}

// Return the typedef name of the pointee of a pointer type, before any
// canonical unwrapping.  Returns "" if the type is not a pointer-to-typedef.
static llvm::StringRef pointeeTypedefName(QualType qt) {
  if (!qt->isPointerType())
    return "";
  QualType inner = qt->getPointeeType().getUnqualifiedType();
  if (const auto *TDT = inner->getAs<TypedefType>())
    return TDT->getDecl()->getName();
  return "";
}

// True if qt is a pointer to PyObject or any Py*-named struct/typedef.
// Accepts subtype patterns common in the CPython C API (PyListObject *,
// PyTypeObject *, etc.).
//
// IMPORTANT: getCanonicalType() strips typedefs, so PyObject * becomes
// struct _object *, PyTypeObject * becomes struct _typeobject *, etc.
// We therefore check the pre-canonical typedef name first.
static bool isPyObjectCompatible(QualType qt) {
  // Fast path: typedef name visible before canonicalization (most common)
  llvm::StringRef tdName = pointeeTypedefName(qt);
  if (tdName.starts_with("Py") || tdName.starts_with("_Py"))
    return true;

  // Canonical fallback for explicit casts like (PyObject *)ptr
  QualType can = qt.getCanonicalType();
  if (!can->isPointerType())
    return false;
  QualType pt = can->getPointeeType().getUnqualifiedType();
  if (const auto *RT = pt->getAs<RecordType>()) {
    // Structural check: any C struct whose first field is named "ob_base"
    // embeds PyObject_HEAD (or PyObject_VAR_HEAD) and is therefore a valid
    // PyObject subtype.  This covers PyObject itself (via PyVarObject),
    // public types (PyListObject, PyTypeObject…) and internal types
    // (TaskObj, FutureObj, buffered, ElementObject…) without any name list.
    const RecordDecl *RD = RT->getDecl()->getDefinition();
    if (RD) {
      auto it = RD->field_begin();
      if (it != RD->field_end() && it->getName() == "ob_base")
        return true;
    }
  }
  return false;
}

// True if qt is a pointer to PyTypeObject specifically.
static bool isPyTypeObjectPtr(QualType qt) {
  // Check typedef name before canonicalization
  if (pointeeTypedefName(qt) == "PyTypeObject")
    return true;

  // Canonical: struct _typeobject
  QualType can = qt.getCanonicalType();
  if (!can->isPointerType())
    return false;
  QualType pt = can->getPointeeType().getUnqualifiedType();
  if (const auto *RT = pt->getAs<RecordType>())
    return RT->getDecl()->getName() == "_typeobject";
  return false;
}

// Check whether an actual call-site type satisfies an expected sentinel.
// Width-specific sentinels (<int>, <uint>, <long>, etc.) are checked against
// the actual type according to 'checkMode':
//   Off      – any integer type is accepted.
//   Standard – bit-width must match; signedness is ignored.
//   Full     – both bit-width and signedness must match.
static bool typeMatches(QualType actual, const std::string &sentinel,
                        IntegralCheckMode checkMode, ASTContext &Ctx) {
  QualType can = actual.getCanonicalType();

  if (sentinel == "<any-ptr>")
    return can->isPointerType();

  if (sentinel == "<char-ptr>")
    return isCharPtr(actual);

  if (sentinel == "<PyObject*>")
    return isPyObjectCompatible(actual);

  if (sentinel == "<PyTypeObject*>")
    return isPyTypeObjectPtr(actual);

  if (sentinel == "<any-int>")
    return can->isIntegerType();

  if (sentinel == "<wchar-ptr>") {
    if (!actual->isPointerType())
      return false;
    QualType pointee = actual->getPointeeType().getUnqualifiedType();
    if (const auto *TDT = pointee->getAs<TypedefType>())
      return TDT->getDecl()->getName() == "wchar_t";
    return pointee.getCanonicalType()->isWideCharType();
  }

  // Enum types: resolve to the compiler-chosen underlying integer type before
  // doing width/signedness checks.  In C the underlying type is implementation-
  // defined, so checking the enum type directly would give unreliable results.
  // If the enum is incomplete (no underlying type yet), accept to avoid false
  // positives.
  if (const auto *ET = can->getAs<EnumType>()) {
    QualType underlying = ET->getDecl()->getIntegerType();
    if (!underlying.isNull())
      return typeMatches(underlying, sentinel, checkMode, Ctx);
    return true; // incomplete enum – accept to avoid false positives
  }

  // Width-specific integer sentinels.
  if (!can->isIntegerType())
    return false;
  if (checkMode == IntegralCheckMode::Off)
    return true; // accept any integer regardless of width/sign
  uint64_t actualBits = Ctx.getTypeSize(actual);
  bool actualUnsigned = actual.getCanonicalType()->isUnsignedIntegerType();

  auto matchWidthAndSign = [&](uint64_t expectBits, bool expectUnsigned) {
    if (actualBits != expectBits)
      return false;
    if (checkMode == IntegralCheckMode::Standard)
      return true;                           // width matches; ignore signedness
    return actualUnsigned == expectUnsigned; // Full: also check sign
  };

  if (sentinel == "<int>")
    return matchWidthAndSign(Ctx.getTypeSize(Ctx.IntTy), false);
  if (sentinel == "<uint>")
    return matchWidthAndSign(Ctx.getTypeSize(Ctx.IntTy), true);
  if (sentinel == "<long>")
    return matchWidthAndSign(Ctx.getTypeSize(Ctx.LongTy), false);
  if (sentinel == "<ulong>")
    return matchWidthAndSign(Ctx.getTypeSize(Ctx.LongTy), true);
  if (sentinel == "<longlong>")
    return matchWidthAndSign(Ctx.getTypeSize(Ctx.LongLongTy), false);
  if (sentinel == "<ulonglong>")
    return matchWidthAndSign(Ctx.getTypeSize(Ctx.LongLongTy), true);
  if (sentinel == "<ssize_t>")
    return matchWidthAndSign(Ctx.getTypeSize(Ctx.getSizeType()), false);
  if (sentinel == "<size_t>")
    return matchWidthAndSign(Ctx.getTypeSize(Ctx.getSizeType()), true);
  if (sentinel == "<intmax_t>")
    return matchWidthAndSign(Ctx.getTypeSize(Ctx.getIntMaxType()), false);
  if (sentinel == "<uintmax_t>")
    return matchWidthAndSign(Ctx.getTypeSize(Ctx.getIntMaxType()), true);
  if (sentinel == "<ptrdiff_t>")
    return matchWidthAndSign(Ctx.getTypeSize(Ctx.getPointerDiffType()), false);

  return false;
}

// Return the length modifier string that correctly describes 'actual' for use
// in an integer conversion spec.  Prefers semantic typedef names so that
// Py_ssize_t → "z" rather than "l" on 64-bit platforms.
static std::string suggestLenMod(QualType actual, ASTContext &Ctx) {
  if (const auto *TDT = actual->getAs<TypedefType>()) {
    llvm::StringRef name = TDT->getDecl()->getName();
    if (name == "Py_ssize_t" || name == "ssize_t")
      return "z";
    if (name == "size_t")
      return "z";
    if (name == "ptrdiff_t")
      return "t";
    if (name == "intmax_t" || name == "uintmax_t")
      return "j";
  }
  // Resolve enum types to their underlying integer type so that e.g. an enum
  // backed by `long` produces "l" rather than the empty modifier.
  QualType canon = actual.getCanonicalType();
  if (const auto *ET = canon->getAs<EnumType>()) {
    QualType underlying = ET->getDecl()->getIntegerType();
    if (!underlying.isNull())
      return suggestLenMod(underlying, Ctx);
    return ""; // incomplete enum – fall back to no modifier
  }
  if (const auto *BT = canon->getAs<BuiltinType>()) {
    switch (BT->getKind()) {
    case BuiltinType::Int:
    case BuiltinType::UInt:
      return "";
    case BuiltinType::Long:
    case BuiltinType::ULong:
      return "l";
    case BuiltinType::LongLong:
    case BuiltinType::ULongLong:
      return "ll";
    default:
      break;
    }
  }
  return "";
}

// Given the original spec token (e.g. "%lu") and the actual argument type,
// return a corrected token (e.g. "%llu") for integer conversion specs.
// Returns the original text unchanged for non-integer specs, non-integer
// actual types, or cases where we cannot reliably fix (e.g. the length
// modifier is followed by literal text that looks like more modifiers –
// the classic "%ull" pitfall where "%u" + "ll" is parsed but fixing "%u"
// to "%llu" would produce "%llull").
static std::string fixedSpec(const std::string &specText, QualType actual,
                             IntegralCheckMode checkMode, ASTContext &Ctx) {
  if (specText.size() < 2 || specText[0] != '%')
    return specText;
  char specChar = specText.back();
  // Only rewrite integer conversion specs.
  static const char intSpecs[] = "diouxX";
  if (!std::strchr(intSpecs, specChar))
    return specText;
  if (!actual.getCanonicalType()->isIntegerType())
    return specText;

  // Strip the old length modifier to extract the pure prefix
  // (flags + width + precision, without the leading '%').
  size_t specIdx = specText.size() - 1;
  size_t lenStart = specIdx; // default: no length modifier
  if (specIdx >= 3 && specText[specIdx - 2] == 'l' &&
      specText[specIdx - 1] == 'l') {
    lenStart = specIdx - 2;
  } else if (specIdx >= 2) {
    char c = specText[specIdx - 1];
    if (c == 'l' || c == 'z' || c == 'j' || c == 't')
      lenStart = specIdx - 1;
  }
  std::string purePrefix = specText.substr(1, lenStart - 1);
  std::string newMod = suggestLenMod(actual, Ctx);

  // In Full mode, also correct the signedness of the conversion character:
  // unsigned actual → 'u'; signed actual → 'd'/'i'.  Leave 'x'/'X'/'o'
  // alone (hex/octal are conventionally acceptable for either signedness).
  // In Standard mode only the length modifier is corrected.
  char newSpecChar = specChar;
  if (checkMode == IntegralCheckMode::Full) {
    bool actualUnsigned = actual.getCanonicalType()->isUnsignedIntegerType();
    if (actualUnsigned && (specChar == 'd' || specChar == 'i'))
      newSpecChar = 'u';
    else if (!actualUnsigned && specChar == 'u')
      newSpecChar = 'd';
  }

  return "%" + purePrefix + newMod + newSpecChar;
}

// --------------------------------------------------------------------------
// Visitor
// --------------------------------------------------------------------------
class PyFmtVisitor : public RecursiveASTVisitor<PyFmtVisitor> {
  ASTContext &Ctx;
  PrintingPolicy PP;
  unsigned DiagMismatch; // cached Clang warning ID
  bool ErrorOnly;        // suppress all-ok call-sites (PY_FMT_ERROR_ONLY)
  IntegralCheckMode
      IntegralMode; // integer width/sign checking (PY_FMT_INTEGRAL_CHECK_MODE)
  // Deduplication: when a format call lives inside a macro body it is
  // visited once per expansion.  We keep the spelling location (inside the
  // macro definition) + function name as a key and skip all but the first.
  std::set<std::string> seenMacroCalls_;

public:
  explicit PyFmtVisitor(ASTContext &Ctx) : Ctx(Ctx), PP(Ctx.getLangOpts()) {
    PP.SuppressTagKeyword = true;
    PP.SuppressScope = false;
    DiagMismatch = Ctx.getDiagnostics().getCustomDiagID(
        DiagnosticsEngine::Warning,
        "[py-fmt] format spec '%0': expected %1 but argument has type '%2'");
    // PY_FMT_ERROR_ONLY=0 → verbose (all call-sites).
    // Anything else, including unset → error-only (default).
    const char *env = std::getenv("PY_FMT_ERROR_ONLY");
    ErrorOnly = (env == nullptr || llvm::StringRef(env) != "0");

    // PY_FMT_INTEGRAL_CHECK_MODE controls integer width/sign checking.
    //   off                – accept any integer; no width/sign check.
    //   standard (default) – width must match; signedness ignored.
    //   full               – both width and signedness must match.
    const char *ienv = std::getenv("PY_FMT_INTEGRAL_CHECK_MODE");
    llvm::StringRef imode(ienv ? ienv : "");
    IntegralMode = IntegralCheckMode::Standard;
    if (imode == "full")
      IntegralMode = IntegralCheckMode::Full;
    else if (imode == "off")
      IntegralMode = IntegralCheckMode::Off;
  }

  bool VisitCallExpr(CallExpr *CE) {
    const FunctionDecl *FD = CE->getDirectCallee();
    if (!FD)
      return true;

    auto it = kFormatFuncs.find(FD->getNameAsString());
    if (it == kFormatFuncs.end())
      return true;

    const FmtFuncInfo &info = it->second;
    unsigned fmtIdx = info.fmtArgIdx;
    if (CE->getNumArgs() <= fmtIdx)
      return true;

    // ---- source location ------------------------------------------------
    // Resolve this early so we can apply the filename filter before
    // doing any further work.
    SourceManager &SM = Ctx.getSourceManager();
    SourceLocation rawLoc = CE->getExprLoc();

    // Deduplicate macro expansions: if this call lives inside a macro body
    // multiple expansions produce identical AST nodes differing only in the
    // expansion site.  Record the spelling location (inside the macro) and
    // skip every expansion after the first.
    if (rawLoc.isMacroID()) {
      SourceLocation spellLoc = SM.getSpellingLoc(rawLoc);
      std::string key = SM.getFilename(spellLoc).str() + ":" +
                        std::to_string(SM.getSpellingLineNumber(spellLoc)) +
                        ":" + FD->getNameAsString();
      if (!seenMacroCalls_.insert(key).second)
        return true; // already reported this macro call
    }

    // Walk through any macro expansions so we report the call-site location
    // rather than the location inside the macro definition body.
    SourceLocation loc = SM.getExpansionLoc(rawLoc);
    llvm::StringRef file = SM.getFilename(loc);

    unsigned line = SM.getExpansionLineNumber(loc);

    if (info.fileFilter && info.fileFilter[0] != '\0') {
      if (!file.contains(info.fileFilter))
        return true;
    }

    // ---- format string --------------------------------------------------
    const Expr *fmtExpr = CE->getArg(fmtIdx)->IgnoreParenImpCasts();
    std::string fmtStr = "<non-literal>";
    bool isLiteral = false;
    if (const auto *SL = dyn_cast<StringLiteral>(fmtExpr)) {
      fmtStr = SL->getString().str();
      isLiteral = true;
    }

    // Buffer all output for this call-site; only flush to stdout when there
    // is at least one mismatch (or when MismatchOnly is disabled).
    std::string outBuf;
    llvm::raw_string_ostream out(outBuf);
    bool hasMismatch = false;

    out << "[py-fmt] func:     " << FD->getNameAsString() << "\n"
        << "[py-fmt] loc:      " << file << ":" << line << "\n"
        << "[py-fmt] fmt:      \"" << fmtStr << "\"\n";

    // ---- non-literal: fall back to listing raw types --------------------
    if (!isLiteral) {
      if (!ErrorOnly) {
        out << "[py-fmt] args:     (";
        for (unsigned i = fmtIdx + 1; i < CE->getNumArgs(); ++i) {
          if (i > fmtIdx + 1)
            out << ", ";
          CE->getArg(i)->getType().print(out, PP);
        }
        out << ")\n";
        llvm::outs() << out.str();
      }
      return true;
    }

    // ---- parse format string and type-check each argument ---------------
    auto specs = parseFmtString(fmtStr);
    size_t maxSpecLen = 0;
    for (const auto &s : specs)
      if (s.text.size() > maxSpecLen)
        maxSpecLen = s.text.size();
    auto padSpec = [&](const std::string &t) {
      return t + std::string(maxSpecLen - t.size(), ' ');
    };
    unsigned argIdx = fmtIdx + 1; // index into CE->getArg(...)
    unsigned argNum = 0;          // 0-based counter for display

    struct SpecFix {
      size_t offset;
      size_t len;
      std::string replacement;
    };
    std::vector<SpecFix> specFixes;

    for (const auto &fa : specs) {
      for (const auto &sentinel : fa.expectedArgs) {
        if (argIdx >= CE->getNumArgs()) {
          out << "[py-fmt] arg[" << argNum << "] " << padSpec(fa.text)
              << "  MISSING_ARG  want=" << sentinel << "\n";
          hasMismatch = true;
          ++argNum;
          continue;
        }

        const Expr *argExpr = CE->getArg(argIdx);
        QualType actual = argExpr->getType();
        std::string actualStr = actual.getAsString(PP);

        if (sentinel == "<unknown>") {
          out << "[py-fmt] arg[" << argNum << "] " << padSpec(fa.text)
              << "  UNKNOWN_SPEC  got=" << actualStr << "\n";
          hasMismatch = true;
          Ctx.getDiagnostics().Report(argExpr->getExprLoc(), DiagMismatch)
              << fa.text << "<unknown spec>" << actualStr;
          ++argIdx;
          ++argNum;
          continue;
        }

        bool ok = typeMatches(actual, sentinel, IntegralMode, Ctx);

        if (ok) {
          out << "[py-fmt] arg[" << argNum << "] " << padSpec(fa.text)
              << "  ok      " << actualStr << "\n";
        } else {
          out << "[py-fmt] arg[" << argNum << "] " << padSpec(fa.text)
              << "  MISMATCH  got=" << actualStr << "  want=" << sentinel
              << "\n";
          hasMismatch = true;

          // Emit a real Clang compiler warning for the mismatch so
          // it appears inline during normal compilation.
          Ctx.getDiagnostics().Report(argExpr->getExprLoc(), DiagMismatch)
              << fa.text << sentinel << actualStr;
          // Record a fix for integer specs (non-integer mismatches like
          // %s with wrong pointer type cannot be auto-corrected here).
          // Guard: skip if the character immediately after this spec token
          // in the format string is alphanumeric — that means the original
          // spec was already malformed (e.g. "%ull" where "%u" is parsed as
          // the spec but "ll" is literal text); fixing "%u" → "%llu" inside
          // "%ull" would produce "%llull".
          if (fa.fmtOffset != std::string::npos) {
            size_t afterSpec = fa.fmtOffset + fa.text.size();
            bool malformed = afterSpec < fmtStr.size() &&
                             std::isalnum((unsigned char)fmtStr[afterSpec]);
            if (!malformed) {
              std::string fixed = fixedSpec(fa.text, actual, IntegralMode, Ctx);
              if (fixed != fa.text)
                specFixes.push_back({fa.fmtOffset, fa.text.size(), fixed});
            }
          }
        }

        ++argIdx;
        ++argNum;
      }
    }

    // ---- surplus arguments (more supplied than specs) -------------------
    if (argIdx < CE->getNumArgs()) {
      out << "[py-fmt] SURPLUS " << (CE->getNumArgs() - argIdx) << " arg(s)\n";
      hasMismatch = true;
    }

    // ---- fixed format string suggestion ---------------------------------
    if (!specFixes.empty()) {
      // Apply substitutions right-to-left so earlier offsets stay valid.
      std::sort(specFixes.begin(), specFixes.end(),
                [](const SpecFix &a, const SpecFix &b) {
                  return a.offset > b.offset;
                });
      std::string fixedFmt = fmtStr;
      for (const auto &fix : specFixes)
        fixedFmt.replace(fix.offset, fix.len, fix.replacement);
      out << "[py-fmt]\n"
          << "[py-fmt] hint:  \"" << fixedFmt << "\"\n";
    } else if (hasMismatch) {
      out << "[py-fmt]\n"
          << "[py-fmt] no fix available\n";
    }

    if (!ErrorOnly || hasMismatch)
      llvm::outs() << out.str();

    return true;
  }
};

// --------------------------------------------------------------------------
// Consumer / Action boilerplate
// --------------------------------------------------------------------------
class PyFmtConsumer : public ASTConsumer {
  PyFmtVisitor Visitor;

public:
  explicit PyFmtConsumer(ASTContext &Ctx) : Visitor(Ctx) {}
  void HandleTranslationUnit(ASTContext &Ctx) override {
    Visitor.TraverseDecl(Ctx.getTranslationUnitDecl());
  }
};

class PyFmtAction : public PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef) override {
    return std::make_unique<PyFmtConsumer>(CI.getASTContext());
  }

  bool ParseArgs(const CompilerInstance &,
                 const std::vector<std::string> &) override {
    return true;
  }

  // Run after the main compiler action so compilation still proceeds
  // normally and real errors/warnings are unaffected.
  ActionType getActionType() override { return AddAfterMainAction; }
};

static FrontendPluginRegistry::Add<PyFmtAction>
    X("py-format-extract", "Extract CPython custom-format-function call info");
