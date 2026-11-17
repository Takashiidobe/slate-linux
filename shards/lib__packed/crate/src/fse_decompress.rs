
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0+ OR BSD-3-Clause
//!
//! FSE : Finite State Entropy decoder
//! Copyright (c) Meta Platforms, Inc. and affiliates.
//!
//!  You can contact the author at :
//!  - FSE source repository : https://github.com/Cyan4973/FiniteStateEntropy
//!  - Public forum : https://groups.google.com/forum/#!forum/lz4c
//!
//! This source code is licensed under both the BSD-style license (found in the
//! LICENSE file in the root directory of this source tree) and the GPLv2 (found
//! in the COPYING file in the root directory of this source tree).
//! You may select, at your option, one of the above-listed licenses.
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! UNUSED_ATTR tells the compiler it is okay if the function is unused.
//! Compiler-specific
// This header exists to force full rebuild when the compiler is upgraded.
//
// When fixdep scans this, it will find this string "CONFIG_CC_VERSION_TEXT"
// and add dependency on include/config/CC_VERSION_TEXT, which is touched
// by Kconfig when the version string from the compiler changes.
// Additional tree-wide dependencies start here.
// If any of the GCC plugins change, we need to rebuild everything that
// was built with them, as they may have changed their behavior and those
// behaviors may need to be synchronized across all translation units.
// If the randstruct seed itself changes (whether for GCC plugins or
// Clang), the entire tree needs to be rebuilt since the randomization of
// structures may change between compilation units if not.
// If any external changes affect Clang's integer wrapping sanitizer
// behavior, a full rebuild is needed as the coverage for wrapping types
// may have changed, which may impact the expected behaviors that should
// not differ between compilation units.
// The use of "&&" / "||" is limited in certain expressions.
// The following enable to calculate "and" / "or" with macro expansion only.
// Helper macros to use CONFIG_ options in C/CPP expressions. Note that
// these only work with boolean and tristate options.
// Getting something that works in C and CPP for an arg that may or may
// not be defined is tricky.  Here, if we have "#define CONFIG_BOOGER 1"
// we match on the placeholder define, insert the "0," for arg1 and generate
// the triplet (0, 1, 0).  Then the last step cherry picks the 2nd arg (a one).
// When CONFIG_BOOGER is not defined, we generate a (... 1, 0) pair, and when
// the last step cherry picks the 2nd arg, we get a zero.
// IS_BUILTIN(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y', 0
// otherwise. For boolean options, this is equivalent to
// IS_ENABLED(CONFIG_FOO).
// IS_MODULE(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'm', 0
// otherwise.  CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1" in
// autoconf.h.
// IS_REACHABLE(CONFIG_FOO) evaluates to 1 if the currently compiled
// code can call a function defined in code compiled based on CONFIG_FOO.
// This is similar to IS_ENABLED(), but returns false when invoked from
// built-in code when CONFIG_FOO is set to 'm'.
// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
// __has_builtin is supported on gcc >= 10, clang >= 3 and icc >= 21.
// In the meantime, to support gcc < 10, we implement __has_builtin
// by hand.
// Indirect macros required for expanded argument pasting, eg. __LINE__.
// C23 introduces "auto" as a standard way to define type-inferred
// variables, but "auto" has been a (useless) keyword even since K&R C,
// so it has always been "namespace reserved."
//
// Until at some future time we require C23 support, we need the gcc
// extension __auto_type, but there is no reason to put that elsewhere
// in the source code.
// nothing
// sparse defines __CHECKER__; see Documentation/dev-tools/sparse.rst
//
// address spaces
// other
// Attributes
// Note: users of __always_inline currently do not write "inline" themselves,
// which seems to be required by gcc to apply the attribute according
// to its docs (and also "warning: always_inline function might not be
// inlinable [-Wattributes]" is emitted).
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-always_005finline-function-attribute
// clang: mentioned
// Note the long name.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-const-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-packed-type-attribute
// clang: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-packed-variable-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-cold-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Label-Attributes.html#index-cold-label-attribute
//
// When -falign-functions=N is in use, we must avoid the cold attribute as
// GCC drops the alignment for cold functions. Worse, GCC can implicitly mark
// callees of cold functions as cold themselves, so it's not sufficient to add
// __function_aligned here as that will not ensure that callees are correctly
// aligned.
//
// See:
//
//   https://lore.kernel.org/lkml/Y77%2FqVgvaJidFpYt@FVFF77S0Q05N
//   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88345#c9
// On x86-64 and arm64 targets, __preserve_most changes the calling convention
// of a function to make the code in the caller as unintrusive as possible. This
// convention behaves identically to the C calling convention on how arguments
// and return values are passed, but uses a different set of caller- and callee-
// saved registers.
//
// The purpose is to alleviates the burden of saving and recovering a large
// register set before and after the call in the caller.  This is beneficial for
// rarely taken slow paths, such as error-reporting functions that may be called
// from hot paths.
//
// Note: This may conflict with instrumentation inserted on function entry which
// does not use __preserve_most or equivalent convention (if in assembly). Since
// function tracing assumes the normal C calling convention, where the attribute
// is supported, __preserve_most implies notrace.  It is recommended to restrict
// use of the attribute to functions that should or already disable tracing.
//
// Optional: not supported by gcc.
//
// clang: https://clang.llvm.org/docs/AttributeReference.html#preserve-most
// Compiler specific macros.
// Some architectures need to provide custom definitions of macros provided
// by linux/compiler-*.h, and can do so using asm/compiler.h. We include that
// conditionally rather than using an asm-generic wrapper in order to avoid
// build failures if any C compilation, which will include this file via an
// -include argument in c_flags, occurs prior to the asm-generic wrappers being
// generated.
// it doesn't make sense on ARM (currently the only user of __naked)
// to trace naked functions because then mcount is called without
// stack and frame pointer being set up and there is no chance to
// restore the lr register to the value before mcount was called.
// Prefer gnu_inline, so that extern inline functions do not emit an
// externally visible function. This makes extern inline behave as per gnu89
// semantics rather than c99. This prevents multiple symbol definition errors
// of extern inline functions at link time.
// A lot of inline functions can cause havoc with function tracing.
// gcc provides both __inline__ and __inline as alternate spellings of
// the inline keyword, though the latter is undocumented. New kernel
// code should only use the inline spelling, but some existing code
// uses __inline__. Since we #define inline above, to ensure
// __inline__ has the same semantics, we need this #define.
//
// However, the spelling __inline is strictly reserved for referring
// to the bare keyword.
// GCC does not warn about unused static inline functions for -Wunused-function.
// Suppress the warning in clang as well by using __maybe_unused, but enable it
// for W=2 build. This will allow clang to find unused functions.
// Rather then using noinline to prevent stack consumption, use
// noinline_for_stack instead.  For documentation reasons.
// Use noinline_for_tracing for functions that should not be inlined.
// For tracing reasons.
// Sanitizer helper attributes: Because using __always_inline and
// __no_sanitize_* conflict, provide helper attributes that will either expand
// to __no_sanitize_* in compilation units where instrumentation is enabled
// (__SANITIZE_*__), or __always_inline in compilation units without
// instrumentation (__SANITIZE_*__ undefined).
//
// The assume attribute is used to indicate that a certain condition is
// assumed to be true. If this condition is violated at runtime, the behavior
// is undefined. Compilers may or may not use this indication to generate
// optimized code.
//
// Note that the clang documentation states that optimizers may react
// differently to this attribute, and this may even have a negative
// performance impact. Therefore this attribute should be used with care.
//
// Optional: only supported since gcc >= 13
// Optional: only supported since clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Statement-Attributes.html#index-assume-statement-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#id13
// Optional: only supported since gcc >= 15
// Optional: only supported since clang >= 18
//
//   gcc: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=108896
// clang: https://clang.llvm.org/docs/AttributeReference.html#counted-by-counted-by-or-null-sized-by-sized-by-or-null
//
// __bdos on clang < 19.1.2 can erroneously return 0:
// https://github.com/llvm/llvm-project/pull/110497
//
// __bdos on clang < 19.1.3 can be off by 4:
// https://github.com/llvm/llvm-project/pull/112636
// Runtime track number of objects pointed to by a pointer member for use by
// CONFIG_FORTIFY_SOURCE and CONFIG_UBSAN_BOUNDS.
//
// Optional: only supported since gcc >= 16
// Optional: only supported since clang >= 22
//
//   gcc: https://gcc.gnu.org/pipermail/gcc-patches/2025-April/681727.html
// clang: https://clang.llvm.org/docs/AttributeReference.html#counted-by-counted-by-or-null-sized-by-sized-by-or-null
// Optional: only supported since gcc >= 15
// Optional: not supported by Clang
//
// gcc: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=117178
// Apply __counted_by() when the Endianness matches to increase test coverage.
// This designates the minimum number of elements a passed array parameter must
// have. For example:
//
//     void some_function(u8 param[at_least 7]);
//
// If a caller passes an array with fewer than 7 elements, the compiler will
// emit a warning.
// Section for code which can't be instrumented at all
// The __cpuidle section is used twofold:
//
//  1) the original use -- identifying if a CPU is 'stuck' in idle state based
//     on it's instruction pointer. See cpu_in_idle().
//
//  2) supressing instrumentation around where cpuidle disables RCU; where the
//     function isn't strictly required for #1, this is interchangeable with
//     noinstr.
// The below symbols may be defined for one or more, but not ALL, of the above
// compilers. We don't consider that to be an error, so set them to nothing.
// For example, some of them are for compiler specific plugins.
// Any place that could be marked with the "alloc_size" attribute is also
// a place to be marked with the "malloc" attribute, except those that may
// be performing a _reallocation_, as that may alias the existing pointer.
// For these, use __realloc_size().
// When the size of an allocated object is needed, use the best available
// mechanism to find it. (For cases where sizeof() cannot be used.)
//
// Optional: only supported since gcc >= 12
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Object-Size-Checking.html
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#evaluating-object-size
// Determine if an attribute has been applied to a variable.
// Using __annotated needs to check for __annotated being available,
// or negative tests may fail when annotation cannot be checked. For
// example, see the definition of __is_cstr().
// __flex_counter() - Get pointer to counter member for the given
//                    flexible array, if it was annotated with __counted_by()
// @FAM: Pointer to flexible array member of an addressable struct instance
//
// For example, with:
//
// 	struct foo {
// 		int counter;
// 		short array[] __counted_by(counter);
// 	} *p;
//
// __flex_counter(p->array) will resolve to &p->counter.
//
// Note that Clang may not allow this to be assigned to a separate
// variable; it must be used directly.
//
// If p->array is unannotated, this returns (void *)NULL.
// Some versions of gcc do not mark 'asm goto' volatile:
//
//  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=103979
//
// We do it here by hand, because it doesn't hurt.
// Clang has trouble with constraints with multiple
// alternative behaviors ("g" , "rm" and "=rm").
// Use __typeof_unqual__() when available.
// Are two types/vars the same type (ignoring qualifiers)?
// __unqual_scalar_typeof(x) - Declare an unqualified scalar type, leaving
// 			       non-scalar types unchanged.
// __signed_scalar_typeof(x) - Declare a signed scalar type, leaving
// 			       non-scalar types unchanged.
// Is this type a native word size -- useful for atomic operations
// #ifdef __OPTIMIZE__ is only a good approximation; for instance "make
// CFLAGS_foo.o=-Og" defines __OPTIMIZE__, does not elide the conditional code
// and can break compilation with wrong error message(s). Combine with
// -U__OPTIMIZE__ when needed.
// 									 * __noreturn is needed to give the compiler enough			 * information to avoid certain possibly-uninitialized			 * warnings (regardless of the build failing).
// compiletime_assert - break build and emit msg if condition is false
// @condition: a compile-time constant condition to check
// @msg:       a message to emit if condition is false
//
// In tradition of POSIX assert, this macro will break the build if the
// supplied condition is *false*, emitting the supplied error message if the
// compiler has support to do so.
// Helpers for emitting diagnostics in pragmas.
// disable assert (default)
// disabled
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// FORCE_INLINE_TEMPLATE is used to define C "templates", which take constant
// parameters. They must be inlined for the compiler to eliminate the constant
// branches.
// target attribute
// Target attribute for BMI2 dynamic dispatch.
// Enable lzcnt, bmi, and bmi2.
// We test for bmi1 & bmi2. lzcnt is included in bmi1.
// Define ZSTD_FALLTHROUGH macro for annotating switch case with the 'fallthrough' attribute.
// - C23: https://en.cppreference.com/w/c/language/attributes/fallthrough
// - CPP17: https://en.cppreference.com/w/cpp/language/attributes/fallthrough
// - Else: __attribute__((__fallthrough__))
// Zstd relies on pointer overflow in its decompressor.
// We add this attribute to functions that rely on pointer overflow.
//
// older versions of clang [3.7, 5.0) will warn that pointer-overflow is ignored.
// Skipped when running bindgen due to a libclang issue;
// see https://github.com/rust-lang/rust-bindgen/issues/2244.
// Annotating a function/variable with __retain tells the compiler to place
// the object in its own section and set the flag SHF_GNU_RETAIN. This flag
// instructs the linker to retain the object during garbage-cleanup or LTO
// phases.
//
// Note that the __used macro is also used to prevent functions or data
// being optimized out, but operates at the compiler/IR-level and may still
// allow unintended removal of objects during linking.
//
// Optional: only supported since gcc >= 11, clang >= 13
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-retain-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#retain
macro_rules! CHECK_F {
($f:expr) => {
{
loop {
let _var_err__: u64 = $f;
loop {
if ERR_isError(_var_err__) != 0 {
return _var_err__;
}
if true {
break;
}
}
if true {
break;
}
}
}
};
}

/// static assert is triggered at compile time, leaving no runtime artefact.
/// static assert only works with compile-time constants.
/// Also, this variant can only be used inside a function.
macro_rules! DEBUG_STATIC_ASSERT {
() => {
};
}

macro_rules! ERROR__15 {
() => {
ZSTD_ERROR__17!()
};
}

macro_rules! ERROR__6 {
() => {
ZSTD_ERROR__7!()
};
}

macro_rules! ERROR {
() => {
ZSTD_ERROR__2!()
};
}

macro_rules! ERROR__2 {
() => {
ZSTD_ERROR__3!()
};
}

macro_rules! ERROR__8 {
() => {
ZSTD_ERROR!()
};
}

macro_rules! ERROR__3 {
() => {
ZSTD_ERROR__5!()
};
}

macro_rules! ERROR__5 {
() => {
ZSTD_ERROR__4!()
};
}

macro_rules! FSE_BUILD_DTABLE_WKSP_SIZE_U32 {
($maxTableLog:expr, $maxSymbolValue:expr) => {
FSE_BUILD_DTABLE_WKSP_SIZE!($maxTableLog, $maxSymbolValue).wrapping_add(core::mem::size_of::<u32>() as u64).wrapping_sub(1) / (core::mem::size_of::<u32>() as u64)
};
}

macro_rules! FSE_BUILD_DTABLE_WKSP_SIZE {
($maxTableLog:expr, $maxSymbolValue:expr) => {
(core::mem::size_of::<i16>() as u64).wrapping_mul($maxSymbolValue.wrapping_add(1) as u64).wrapping_add(1u64 << $maxTableLog).wrapping_add(8)
};
}

macro_rules! FSE_DECOMPRESS_WKSP_SIZE_U32 {
($maxTableLog:expr, $maxSymbolValue:expr) => {
(FSE_DTABLE_SIZE_U32!($maxTableLog).wrapping_add(1) as u64).wrapping_add(FSE_BUILD_DTABLE_WKSP_SIZE_U32!($maxTableLog, $maxSymbolValue)).wrapping_add(128u64).wrapping_add(1)
};
}

macro_rules! FSE_DECOMPRESS_WKSP_SIZE {
($maxTableLog:expr, $maxSymbolValue:expr) => {
FSE_DECOMPRESS_WKSP_SIZE_U32!($maxTableLog, $maxSymbolValue).wrapping_mul(core::mem::size_of::<u32>() as u64)
};
}

macro_rules! FSE_DTABLE_SIZE_U32 {
($maxTableLog:expr) => {
1i32.wrapping_add(1i32 << $maxTableLog)
};
}

macro_rules! FSE_DTABLE_SIZE {
($maxTableLog:expr) => {
(FSE_DTABLE_SIZE_U32!($maxTableLog) as u64).wrapping_mul(core::mem::size_of::<u32>() as u64)
};
}

macro_rules! FSE_GETSYMBOL {
($statePtr:expr, $fast:expr, $bitD:expr) => {
{
let __v: i32 = if ($fast) != 0 {
FSE_decodeSymbolFast($statePtr, core::ptr::addr_of_mut!($bitD)) as i32
} else {
FSE_decodeSymbol($statePtr, core::ptr::addr_of_mut!($bitD)) as i32
};
__v
}
};
}

/// use only *after* variable declarations
macro_rules! FSE_STATIC_ASSERT {
() => {
DEBUG_STATIC_ASSERT!();
};
}

macro_rules! FSE_TABLESTEP {
($tableSize:expr) => {
(($tableSize) >> 1i32).wrapping_add(($tableSize) >> 3i32).wrapping_add(3)
};
}

macro_rules! UNLIKELY {
($x:expr) => {
($x) as i64
};
}

macro_rules! ZSTD_ERROR__2 {
() => {
20i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__7 {
() => {
44i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__4 {
() => {
1i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__17 {
() => {
46i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__3 {
() => {
72i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__5 {
() => {
70i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR {
() => {
120i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_memcpy {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memcpy(($d) as *mut (), ($s) as *const (), $n) };
};
}

macro_rules! ZSTD_memset {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memset(($d) as *mut (), $s, $n) };
};
}

/// __get_unaligned_t - read an unaligned value from memory.
/// @type:	the type to load from the pointer.
/// @ptr:	the pointer to load from.
///
/// Use memcpy to affect an unaligned type sized load avoiding undefined behavior
/// from approaches like type punning that require -fno-strict-aliasing in order
/// to be correct. As type may be const, use __unqual_scalar_typeof to map to a
/// non-const type - you can't memcpy into a const type. The
/// __get_unaligned_ctrl_type gives __unqual_scalar_typeof its required
/// expression rather than type, a pointer is used to avoid warnings about mixing
/// the use of 0 and NULL. The void* cast silences ubsan warnings.
macro_rules! __get_unaligned_t {
($type:ty, $ptr:expr) => {
{
let mut __v: $type;
core::ptr::null_mut::<u32>();
let mut __v3: $type = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), (($ptr) as *mut ()) as *const (), core::mem::size_of::<$type>() as u64) };
__v = __v3;
__v
}
};
}

/// __get_unaligned_t - read an unaligned value from memory.
/// @type:	the type to load from the pointer.
/// @ptr:	the pointer to load from.
///
/// Use memcpy to affect an unaligned type sized load avoiding undefined behavior
/// from approaches like type punning that require -fno-strict-aliasing in order
/// to be correct. As type may be const, use __unqual_scalar_typeof to map to a
/// non-const type - you can't memcpy into a const type. The
/// __get_unaligned_ctrl_type gives __unqual_scalar_typeof its required
/// expression rather than type, a pointer is used to avoid warnings about mixing
/// the use of 0 and NULL. The void* cast silences ubsan warnings.
macro_rules! __get_unaligned_t__5 {
($type:ty, $ptr:expr) => {
{
let mut __v: $type;
core::ptr::null_mut::<u64>();
let mut __v3: $type = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), (($ptr) as *mut ()) as *const (), core::mem::size_of::<$type>() as u64) };
__v = __v3;
__v
}
};
}

/// __put_unaligned_t - write an unaligned value to memory.
/// @type:	the type of the value to store.
/// @val:	the value to store.
/// @ptr:	the pointer to store to.
///
/// Use memcpy to affect an unaligned type sized store avoiding undefined
/// behavior from approaches like type punning that require -fno-strict-aliasing
/// in order to be correct. The void* cast silences ubsan warnings.
macro_rules! __put_unaligned_t {
($val:expr, $ptr:expr) => {
loop {
let mut __put_unaligned_val: u64 = $val;
unsafe { __builtin_memcpy(($ptr) as *mut (), core::ptr::addr_of_mut!(__put_unaligned_val) as *const (), core::mem::size_of::<u64>() as u64) };
if true {
break;
}
}
};
}

macro_rules! put_unaligned {
($val:expr, $ptr:expr) => {
__put_unaligned_t!($val, $ptr);
};
}

unsafe extern "C" {
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    #[link_name = "memset"]
    fn __builtin_memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn FSE_readNCount_bmi2(arg0: *mut i16, arg1: *mut u32, arg2: *mut u32, arg3: *const (), arg4: u64, arg5: i32) -> u64;
}

// Optional: only supported since gcc >= 15, clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
fn get_unaligned_le32(p: *const ()) -> u32 {
__get_unaligned_t!(u32, p)
}

fn get_unaligned_le64(p: *const ()) -> u64 {
__get_unaligned_t__5!(u64, p)
}

fn MEM_32bits() -> u32 {
((core::mem::size_of::<u64>() as u64) == 4) as u32
}

fn MEM_write64(memPtr: *mut (), value: u64) {
put_unaligned!(value, memPtr as *mut u64);
}

fn MEM_readLE32(memPtr: *const ()) -> u32 {
get_unaligned_le32(memPtr)
}

fn MEM_readLE64(memPtr: *const ()) -> u64 {
get_unaligned_le64(memPtr)
}

fn MEM_readLEST(memPtr: *const ()) -> u64 {
if MEM_32bits() != 0 {
return MEM_readLE32(memPtr) as u64;
} else {
return MEM_readLE64(memPtr);
}
}

fn ERR_isError(code: u64) -> u32 {
(code > ERROR__8!()) as u32
}

// check and forward error code
unsafe extern "C" fn _force_has_format_string(format: *const u8, __va_args: ...) {
}

// Ignore: this is an internal helper.
//
// We want to force this function invocation to be syntactically correct, but
// we don't want to force runtime evaluation of its arguments.
// Return the specified error if the condition evaluates to true.
//
// In debug modes, prints additional information.
// In order to do that (particularly, printing the conditional that failed),
// this can't just wrap RETURN_ERROR().
fn ZSTD_countLeadingZeros32(val: u32) -> u32 {
(val.leading_zeros() as i32) as u32
}

fn ZSTD_highbit32(val: u32) -> u32 {
31u32.wrapping_sub(ZSTD_countLeadingZeros32(val))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bitstream_h_90 {
    bitContainer: u64,
    bitsConsumed: u32,
    ptr: *const u8,
    start: *const u8,
    limitPtr: *const u8,
}


fn BIT_initDStream(bitD: *mut __SlateRecord_bitstream_h_90, srcBuffer: *const (), srcSize: u64) -> u64 {
if srcSize < 1 {
ZSTD_memset!(bitD, 0i32, core::mem::size_of::<__SlateRecord_bitstream_h_90>() as u64);
return ERROR__2!();
}
unsafe {
(*bitD).start = srcBuffer as *const u8;
}
unsafe {
(*bitD).limitPtr = unsafe { (*bitD).start }.wrapping_offset(core::mem::size_of::<u64>() as isize);
}
if srcSize >= (core::mem::size_of::<u64>() as u64) {
// normal case
unsafe {
(*bitD).ptr = (srcBuffer as *const u8).wrapping_offset(srcSize as isize).wrapping_offset(-(core::mem::size_of::<u64>() as isize));
}
unsafe {
(*bitD).bitContainer = MEM_readLEST((*bitD).ptr as *const ());
}
let lastByte: u8 = unsafe { *(srcBuffer as *const u8).wrapping_offset(srcSize.wrapping_sub(1) as isize) };
let __v0: u32 = if lastByte != 0 {
8u32.wrapping_sub(ZSTD_highbit32(lastByte as u32))
} else {
0
};
unsafe {
(*bitD).bitsConsumed = __v0;
}
// ensures bitsConsumed is always set
if (lastByte as i32) == 0 {
return ERROR__5!();
}
// endMark not present
} else {
unsafe {
(*bitD).ptr = (*bitD).start;
}
unsafe {
(*bitD).bitContainer = *((*bitD).start as *const u8) as u64;
}
let mut __t0: i64 = match srcSize { 7 => 0, 6 => 1, 5 => 2, 4 => 3, 3 => 4, 2 => 5, _ => 6 };
loop {
match __t0 {
0 => {
let __v3: u64 = unsafe { (*bitD).bitContainer };
unsafe {
(*bitD).bitContainer = __v3.wrapping_add((*(srcBuffer as *const u8).wrapping_offset(6 as isize) as u64) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(16));
}
__t0 = 1;
continue;
}
1 => {
let __v6: u64 = unsafe { (*bitD).bitContainer };
unsafe {
(*bitD).bitContainer = __v6.wrapping_add((*(srcBuffer as *const u8).wrapping_offset(5 as isize) as u64) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(24));
}
__t0 = 2;
continue;
}
2 => {
let __v9: u64 = unsafe { (*bitD).bitContainer };
unsafe {
(*bitD).bitContainer = __v9.wrapping_add((*(srcBuffer as *const u8).wrapping_offset(4 as isize) as u64) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(32));
}
__t0 = 3;
continue;
}
3 => {
let __v12: u64 = unsafe { (*bitD).bitContainer };
unsafe {
(*bitD).bitContainer = __v12.wrapping_add((*(srcBuffer as *const u8).wrapping_offset(3 as isize) as u64) << 24i32);
}
__t0 = 4;
continue;
}
4 => {
let __v15: u64 = unsafe { (*bitD).bitContainer };
unsafe {
(*bitD).bitContainer = __v15.wrapping_add((*(srcBuffer as *const u8).wrapping_offset(2 as isize) as u64) << 16i32);
}
__t0 = 5;
continue;
}
5 => {
let __v18: u64 = unsafe { (*bitD).bitContainer };
unsafe {
(*bitD).bitContainer = __v18.wrapping_add((*(srcBuffer as *const u8).wrapping_offset(1 as isize) as u64) << 8i32);
}
__t0 = 6;
continue;
}
6 => {
break;
}
_ => {
break;
}
}
}
let lastByte: u8 = unsafe { *(srcBuffer as *const u8).wrapping_offset(srcSize.wrapping_sub(1) as isize) };
let __v20: u32 = if lastByte != 0 {
8u32.wrapping_sub(ZSTD_highbit32(lastByte as u32))
} else {
0
};
unsafe {
(*bitD).bitsConsumed = __v20;
}
if (lastByte as i32) == 0 {
return ERROR!();
}
// endMark not present
let __v22: u32 = unsafe { (*bitD).bitsConsumed };
unsafe {
(*bitD).bitsConsumed = __v22.wrapping_add(((core::mem::size_of::<u64>() as u64).wrapping_sub(srcSize) as u32).wrapping_mul(8));
}
}
srcSize
}

#[inline(always)]
fn BIT_reloadDStream(bitD: *mut __SlateRecord_bitstream_h_90) -> u32 {
// note : once in overflow mode, a bitstream remains in this mode until it's reset
if UNLIKELY!(((unsafe { (*bitD).bitsConsumed }) as u64) > (core::mem::size_of::<u64>() as u64).wrapping_mul(8)) != 0 {
unsafe {
(*bitD).ptr = core::ptr::addr_of!(zeroFilled) as *const u8;
}
// aliasing is allowed for char
// overflow detected, erroneous scenario or end of stream: no update
return 3;
}
if (unsafe { (*bitD).ptr }) >= unsafe { (*bitD).limitPtr } {
return BIT_reloadDStream_internal(bitD);
}
if (unsafe { (*bitD).ptr }) == unsafe { (*bitD).start } {
// reached end of bitStream => no update
if ((unsafe { (*bitD).bitsConsumed }) as u64) < (core::mem::size_of::<u64>() as u64).wrapping_mul(8) {
return 1;
}
return 2;
}
// start < ptr < limitPtr => cautious update
let mut nbBytes: u32 = (unsafe { (*bitD).bitsConsumed }) >> 3i32;
let mut result: u32 = 0;
if unsafe { (*bitD).ptr }.wrapping_offset(-(nbBytes as isize)) < unsafe { (*bitD).start } {
nbBytes = ((unsafe { unsafe { (*bitD).ptr }.offset_from((*bitD).start as *const u8) }) as i64) as u32; // ptr > start
result = 1;
}
let __v1: *const u8 = unsafe { (*bitD).ptr };
unsafe {
(*bitD).ptr = __v1.wrapping_offset(-(nbBytes as isize));
}
let __v4: u32 = unsafe { (*bitD).bitsConsumed };
unsafe {
(*bitD).bitsConsumed = __v4.wrapping_sub(nbBytes.wrapping_mul(8));
}
unsafe {
(*bitD).bitContainer = MEM_readLEST((*bitD).ptr as *const ());
}
// reminder : srcSize > sizeof(bitD->bitContainer), otherwise bitD->ptr == bitD->start
result
}

#[inline(always)]
fn BIT_getMiddleBits(bitContainer: u64, start: u32, nbBits: u32) -> u64 {
let regMask: u32 = (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1) as u32;
// if start > regMask, bitstream is corrupted, and result is undefined
// x86 transform & ((1 << nbBits) - 1) to bzhi instruction, it is better
// than accessing memory. When bmi2 instruction is not present, we consider
// such cpus old (pre-Haswell, 2013) and their performance is not of that
// importance.
bitContainer >> (start & regMask) & (1u64 << nbBits).wrapping_sub(1)
}

#[inline(always)]
fn BIT_lookBits(bitD: *const __SlateRecord_bitstream_h_90, nbBits: u32) -> u64 {
// arbitrate between double-shift and shift+mask
//
// if bitD->bitsConsumed + nbBits > sizeof(bitD->bitContainer)*8,
// bitstream is likely corrupted, and result is undefined
BIT_getMiddleBits(unsafe { (*bitD).bitContainer }, (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub((unsafe { (*bitD).bitsConsumed }) as u64).wrapping_sub(nbBits as u64) as u32, nbBits)
}

fn BIT_lookBitsFast(bitD: *const __SlateRecord_bitstream_h_90, nbBits: u32) -> u64 {
let regMask: u32 = (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1) as u32;
(unsafe { (*bitD).bitContainer }) << ((unsafe { (*bitD).bitsConsumed }) & regMask) >> (regMask.wrapping_add(1).wrapping_sub(nbBits) & regMask)
}

#[inline(always)]
fn BIT_skipBits(__v0: *mut __SlateRecord_bitstream_h_90, nbBits: u32) {
let __v1: u32 = unsafe { (*__v0).bitsConsumed };
unsafe {
(*__v0).bitsConsumed = __v1.wrapping_add(nbBits);
}
}

#[inline(always)]
fn BIT_readBits(bitD: *mut __SlateRecord_bitstream_h_90, nbBits: u32) -> u64 {
let value: u64 = BIT_lookBits(bitD as *const __SlateRecord_bitstream_h_90, nbBits);
BIT_skipBits(bitD, nbBits);
value
}

fn BIT_readBitsFast(bitD: *mut __SlateRecord_bitstream_h_90, nbBits: u32) -> u64 {
let value: u64 = BIT_lookBitsFast(bitD as *const __SlateRecord_bitstream_h_90, nbBits);
BIT_skipBits(bitD, nbBits);
value
}

fn BIT_reloadDStream_internal(bitD: *mut __SlateRecord_bitstream_h_90) -> u32 {
let __v1: *const u8 = unsafe { (*bitD).ptr };
unsafe {
(*bitD).ptr = __v1.wrapping_offset(-(((*bitD).bitsConsumed >> 3i32) as isize));
}
unsafe {
(*bitD).bitsConsumed = (*bitD).bitsConsumed & 7;
}
unsafe {
(*bitD).bitContainer = MEM_readLEST((*bitD).ptr as *const ());
}
0
}

static zeroFilled: u64 = 0;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_fse_h_355 {
    state: u64,
    table: *const (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_fse_h_509 {
    tableLog: u16,
    fastMode: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_fse_h_516 {
    newState: u16,
    symbol: u8,
    nbBits: u8,
}


fn FSE_initDState(DStatePtr: &mut __SlateRecord_fse_h_355, bitD: *mut __SlateRecord_bitstream_h_90, dt: *const u32) {
let ptr: *const () = dt as *const ();
unsafe {
(*(DStatePtr as *mut __SlateRecord_fse_h_355)).state = BIT_readBits(bitD, (*(ptr as *const __SlateRecord_fse_h_509)).tableLog as u32);
}
BIT_reloadDStream(bitD);
unsafe {
(*(DStatePtr as *mut __SlateRecord_fse_h_355)).table = dt.wrapping_offset(1 as isize) as *const ();
}
}

fn FSE_decodeSymbol(DStatePtr: *mut __SlateRecord_fse_h_355, bitD: *mut __SlateRecord_bitstream_h_90) -> u8 {
let DInfo: __SlateRecord_fse_h_516 = unsafe { *((*DStatePtr).table as *const __SlateRecord_fse_h_516).wrapping_offset((*DStatePtr).state as isize) };
let nbBits: u32 = DInfo.nbBits as u32;
let symbol: u8 = DInfo.symbol;
let lowBits: u64 = BIT_readBits(bitD, nbBits);
unsafe {
(*DStatePtr).state = ((DInfo.newState as i32) as u64).wrapping_add(lowBits);
}
symbol
}

fn FSE_decodeSymbolFast(DStatePtr: *mut __SlateRecord_fse_h_355, bitD: *mut __SlateRecord_bitstream_h_90) -> u8 {
let DInfo: __SlateRecord_fse_h_516 = unsafe { *((*DStatePtr).table as *const __SlateRecord_fse_h_516).wrapping_offset((*DStatePtr).state as isize) };
let nbBits: u32 = DInfo.nbBits as u32;
let symbol: u8 = DInfo.symbol;
let lowBits: u64 = BIT_readBitsFast(bitD, nbBits);
unsafe {
(*DStatePtr).state = ((DInfo.newState as i32) as u64).wrapping_add(lowBits);
}
symbol
}

// Includes
// assert
// template functions type & suffix
// Constants
// ZSTD_memcpy
// ZSTD_highbit32
// Error Management
// Templates
//
// designed to be included
// for type-specific functions (template emulation in C)
// Objective is to write these functions only once, for improved maintenance
// safety checks
// Function names
fn FSE_buildDTable_internal(dt: *mut u32, normalizedCounter: *const i16, maxSymbolValue: u32, tableLog: u32, workSpace: *mut (), wkspSize: u64) -> u64 {
let tdPtr: *mut () = dt.wrapping_offset(1 as isize) as *mut (); // because *dt is unsigned, 32-bits aligned on 32-bits
let tableDecode: *mut __SlateRecord_fse_h_516 = tdPtr as *mut __SlateRecord_fse_h_516;
let symbolNext: *mut u16 = workSpace as *mut u16;
let spread: *mut u8 = symbolNext.wrapping_offset(maxSymbolValue as isize).wrapping_offset(1 as isize) as *mut u8;
let maxSV1: u32 = maxSymbolValue.wrapping_add(1);
let tableSize: u32 = (1i32 << tableLog) as u32;
let mut highThreshold: u32 = tableSize.wrapping_sub(1);
// Sanity Checks
if FSE_BUILD_DTABLE_WKSP_SIZE!(tableLog, maxSymbolValue) > wkspSize {
return ERROR__15!();
}
if maxSymbolValue > 255 {
return ERROR__15!();
}
if tableLog > 12u32 {
return ERROR__6!();
}
// Init, lay down lowprob symbols
let mut DTableH: __SlateRecord_fse_h_509 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(DTableH) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
DTableH.tableLog = tableLog as u16;
DTableH.fastMode = 1;
let largeLimit: i16 = (1i32 << tableLog.wrapping_sub(1)) as i16;
for s in 0..maxSV1 {
if ((unsafe { *normalizedCounter.wrapping_offset(s as isize) }) as i32) == 1i32.wrapping_neg() {
let __v0: u32 = highThreshold;
highThreshold = __v0.wrapping_sub(1);
unsafe {
(*tableDecode.wrapping_offset(__v0 as isize)).symbol = s as u8;
}
unsafe {
*symbolNext.wrapping_offset(s as isize) = 1;
}
} else {
if ((unsafe { *normalizedCounter.wrapping_offset(s as isize) }) as i32) >= (largeLimit as i32) {
DTableH.fastMode = 0;
}
unsafe {
*symbolNext.wrapping_offset(s as isize) = *normalizedCounter.wrapping_offset(s as isize) as u16;
}
}
}
ZSTD_memcpy!(dt, core::ptr::addr_of_mut!(DTableH), core::mem::size_of::<__SlateRecord_fse_h_509>() as u64);
// Spread symbols
if highThreshold == tableSize.wrapping_sub(1) {
let tableMask: u64 = tableSize.wrapping_sub(1) as u64;
let step: u64 = FSE_TABLESTEP!(tableSize) as u64;
// First lay down the symbols in order.
// We use a uint64_t to lay down 8 bytes at a time. This reduces branch
// misses since small blocks generally have small table logs, so nearly
// all symbols have counts <= 8. We ensure we have 8 bytes at the end of
// our buffer to handle the over-write.
{
let add: u64 = 72340172838076673;
let mut pos: u64 = 0;
let mut sv: u64 = 0;
let mut s: u32 = 0;
while s < maxSV1 {
let mut i: i32 = 8;
let n: i32 = (unsafe { *normalizedCounter.wrapping_offset(s as isize) }) as i32;
MEM_write64(spread.wrapping_offset(pos as isize) as *mut (), sv);
while i < n {
MEM_write64(spread.wrapping_offset(pos as isize).wrapping_offset(i as isize) as *mut (), sv);
i = i.wrapping_add(8);
}
pos = pos.wrapping_add(n as u64);
s = s.wrapping_add(1);
sv = sv.wrapping_add(add);
}
}
// Now we spread those positions across the table.
// The benefit of doing it in two stages is that we avoid the
// variable size inner loop, which caused lots of branch misses.
// Now we can run through all the positions without any branch misses.
// We unroll the loop twice, since that is what empirically worked best.
let mut position: u64 = 0;
// FSE_MIN_TABLELOG is 5
let mut s: u64 = 0;
let unroll: u64 = 2;
while s < (tableSize as u64) {
for u in 0..unroll {
let uPosition: u64 = position.wrapping_add(u.wrapping_mul(step)) & tableMask;
unsafe {
(*tableDecode.wrapping_offset(uPosition as isize)).symbol = *spread.wrapping_offset(s.wrapping_add(u) as isize);
}
}
position = position.wrapping_add(unroll.wrapping_mul(step)) & tableMask;
s = s.wrapping_add(unroll);
}
} else {
let tableMask: u32 = tableSize.wrapping_sub(1);
let step: u32 = FSE_TABLESTEP!(tableSize);
let mut position: u32 = 0;
for s in 0..maxSV1 {
let mut i: i32 = 0;
while i < ((unsafe { *normalizedCounter.wrapping_offset(s as isize) }) as i32) {
unsafe {
(*tableDecode.wrapping_offset(position as isize)).symbol = s as u8;
}
position = position.wrapping_add(step) & tableMask;
while position > highThreshold {
position = position.wrapping_add(step) & tableMask;
}
// lowprob area
i = i.wrapping_add(1);
}
}
if position != 0 {
return ERROR__5!();
}
// position must reach all cells once, otherwise normalizedCounter is incorrect
}
// Build Decoding table
for u in 0..tableSize {
let symbol: u8 = unsafe { (*tableDecode.wrapping_offset(u as isize)).symbol };
let mut nextState: u32 = 0;
let __v28: *mut u16 = symbolNext.wrapping_offset((symbol as i32) as isize);
let __v29: u16 = unsafe { *__v28 };
unsafe {
*__v28 = (__v29 as i32).wrapping_add(1) as u16;
}
nextState = __v29 as u32;
unsafe {
(*tableDecode.wrapping_offset(u as isize)).nbBits = tableLog.wrapping_sub(ZSTD_highbit32(nextState)) as u8;
}
unsafe {
(*tableDecode.wrapping_offset(u as isize)).newState = (nextState << ((*tableDecode.wrapping_offset(u as isize)).nbBits as i32)).wrapping_sub(tableSize) as u16;
}
}
0
}

#[unsafe(no_mangle)]
extern "C" fn FSE_buildDTable_wksp(dt: *mut u32, normalizedCounter: *const i16, maxSymbolValue: u32, tableLog: u32, workSpace: *mut (), wkspSize: u64) -> u64 {
FSE_buildDTable_internal(dt, normalizedCounter, maxSymbolValue, tableLog, workSpace, wkspSize)
}

// Decompression (Byte symbols)
#[inline(always)]
fn FSE_decompress_usingDTable_generic(dst: *mut (), maxDstSize: u64, cSrc: *const (), cSrcSize: u64, dt: *const u32, fast: u32) -> u64 {
let ostart: *mut u8 = dst as *mut u8;
let mut op: *mut u8 = ostart;
let omax: *mut u8 = op.wrapping_offset(maxDstSize as isize);
let olimit: *mut u8 = omax.wrapping_offset(-(3 as isize));
let mut bitD: __SlateRecord_bitstream_h_90 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bitD) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
let mut state1: __SlateRecord_fse_h_355 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(state1) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut state2: __SlateRecord_fse_h_355 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(state2) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
// Init
CHECK_F!(BIT_initDStream(core::ptr::addr_of_mut!(bitD), cSrc, cSrcSize));
FSE_initDState(unsafe { &mut (*core::ptr::addr_of_mut!(state1)) }, core::ptr::addr_of_mut!(bitD), dt);
FSE_initDState(unsafe { &mut (*core::ptr::addr_of_mut!(state2)) }, core::ptr::addr_of_mut!(bitD), dt);
loop {
if (BIT_reloadDStream(core::ptr::addr_of_mut!(bitD)) as u32) == 3 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR!();
}
if true {
break;
}
}
// 4 symbols per loop
while (((BIT_reloadDStream(core::ptr::addr_of_mut!(bitD)) as u32) == 0) as i32) & ((op < olimit) as i32) != 0 {
unsafe {
*op.wrapping_offset(0 as isize) = FSE_GETSYMBOL!(core::ptr::addr_of_mut!(state1), fast, bitD) as u8;
}
if 31u64 > ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8) {
BIT_reloadDStream(core::ptr::addr_of_mut!(bitD));
}
// This test must be static
unsafe {
*op.wrapping_offset(1 as isize) = FSE_GETSYMBOL!(core::ptr::addr_of_mut!(state2), fast, bitD) as u8;
}
if 55u64 > ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8) {
if (BIT_reloadDStream(core::ptr::addr_of_mut!(bitD)) as u32) > 0 {
let __v9: *mut u8 = op;
op = __v9.wrapping_offset(2 as isize);
break;
}
}
// This test must be static
unsafe {
*op.wrapping_offset(2 as isize) = FSE_GETSYMBOL!(core::ptr::addr_of_mut!(state1), fast, bitD) as u8;
}
if 31u64 > ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8) {
BIT_reloadDStream(core::ptr::addr_of_mut!(bitD));
}
// This test must be static
unsafe {
*op.wrapping_offset(3 as isize) = FSE_GETSYMBOL!(core::ptr::addr_of_mut!(state2), fast, bitD) as u8;
}
let __v14: *mut u8 = op;
op = __v14.wrapping_offset(4 as isize);
}
// tail
//
// note : BIT_reloadDStream(&bitD) >= FSE_DStream_partiallyFilled; Ends at exactly BIT_DStream_completed
while true {
if op > omax.wrapping_offset(-(2 as isize)) {
return ERROR__3!();
}
let __v16: *mut u8 = op;
op = __v16.wrapping_offset(1 as isize);
unsafe {
*__v16 = FSE_GETSYMBOL!(core::ptr::addr_of_mut!(state1), fast, bitD) as u8;
}
if (BIT_reloadDStream(core::ptr::addr_of_mut!(bitD)) as u32) == 3 {
let __v19: *mut u8 = op;
op = __v19.wrapping_offset(1 as isize);
unsafe {
*__v19 = FSE_GETSYMBOL!(core::ptr::addr_of_mut!(state2), fast, bitD) as u8;
}
break;
}
if op > omax.wrapping_offset(-(2 as isize)) {
return ERROR__3!();
}
let __v23: *mut u8 = op;
op = __v23.wrapping_offset(1 as isize);
unsafe {
*__v23 = FSE_GETSYMBOL!(core::ptr::addr_of_mut!(state2), fast, bitD) as u8;
}
if (BIT_reloadDStream(core::ptr::addr_of_mut!(bitD)) as u32) == 3 {
let __v26: *mut u8 = op;
op = __v26.wrapping_offset(1 as isize);
unsafe {
*__v26 = FSE_GETSYMBOL!(core::ptr::addr_of_mut!(state1), fast, bitD) as u8;
}
break;
}
}
((unsafe { op.offset_from(ostart as *mut u8) }) as i64) as u64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_fse_decompress_c_241 {
    ncount: [i16; 256],
}


#[inline(always)]
fn FSE_decompress_wksp_body(dst: *mut (), dstCapacity: u64, cSrc: *const (), mut cSrcSize: u64, maxLog: u32, mut workSpace: *mut (), mut wkspSize: u64, bmi2: i32) -> u64 {
let istart: *const u8 = cSrc as *const u8;
let mut ip: *const u8 = istart;
let mut tableLog: u32 = 0;
let mut maxSymbolValue: u32 = 255;
let wksp: *mut __SlateRecord_fse_decompress_c_241 = workSpace as *mut __SlateRecord_fse_decompress_c_241;
let dtablePos: u64 = (core::mem::size_of::<__SlateRecord_fse_decompress_c_241>() as u64) / (core::mem::size_of::<u32>() as u64);
let dtable: *mut u32 = (workSpace as *mut u32).wrapping_offset(dtablePos as isize);
FSE_STATIC_ASSERT!();
if wkspSize < (core::mem::size_of::<__SlateRecord_fse_decompress_c_241>() as u64) {
return ERROR__5!();
}
// correct offset to dtable depends on this property
FSE_STATIC_ASSERT!();
// normal FSE decoding mode
let NCountLength: u64 = unsafe { FSE_readNCount_bmi2((*wksp).ncount.as_mut_ptr() as *mut i16, core::ptr::addr_of_mut!(maxSymbolValue), core::ptr::addr_of_mut!(tableLog), istart as *const (), cSrcSize, bmi2) };
if ERR_isError(NCountLength) != 0 {
return NCountLength;
}
if tableLog > maxLog {
return ERROR__6!();
}
let __v0: *const u8 = ip;
ip = __v0.wrapping_offset(NCountLength as isize);
cSrcSize = cSrcSize.wrapping_sub(NCountLength);
if FSE_DECOMPRESS_WKSP_SIZE!(tableLog, maxSymbolValue) > wkspSize {
return ERROR__6!();
}
workSpace = (workSpace as *mut u8).wrapping_offset(core::mem::size_of::<__SlateRecord_fse_decompress_c_241>() as isize).wrapping_offset(FSE_DTABLE_SIZE!(tableLog) as isize) as *mut ();
let __v4: u64 = wkspSize;
wkspSize = __v4.wrapping_sub((core::mem::size_of::<__SlateRecord_fse_decompress_c_241>() as u64).wrapping_add(FSE_DTABLE_SIZE!(tableLog)));
CHECK_F!(FSE_buildDTable_internal(dtable, (unsafe { (*wksp).ncount.as_mut_ptr() as *mut i16 }) as *const i16, maxSymbolValue, tableLog, workSpace, wkspSize));
let ptr: *const () = dtable as *const ();
let fastMode: u32 = (unsafe { (*(ptr as *const __SlateRecord_fse_h_509)).fastMode }) as u32;
// select fast mode (static)
if fastMode != 0 {
return FSE_decompress_usingDTable_generic(dst, dstCapacity, ip as *const (), cSrcSize, dtable as *const u32, 1u32);
}
FSE_decompress_usingDTable_generic(dst, dstCapacity, ip as *const (), cSrcSize, dtable as *const u32, 0u32)
}

/// Avoids the FORCE_INLINE of the _body() function.
fn FSE_decompress_wksp_body_default(dst: *mut (), dstCapacity: u64, cSrc: *const (), cSrcSize: u64, maxLog: u32, workSpace: *mut (), wkspSize: u64) -> u64 {
FSE_decompress_wksp_body(dst, dstCapacity, cSrc, cSrcSize, maxLog, workSpace, wkspSize, 0i32)
}

#[target_feature(enable = "lzcnt,bmi1,bmi2")]
unsafe fn FSE_decompress_wksp_body_bmi2(dst: *mut (), dstCapacity: u64, cSrc: *const (), cSrcSize: u64, maxLog: u32, workSpace: *mut (), wkspSize: u64) -> u64 {
FSE_decompress_wksp_body(dst, dstCapacity, cSrc, cSrcSize, maxLog, workSpace, wkspSize, 1i32)
}

#[unsafe(no_mangle)]
extern "C" fn FSE_decompress_wksp_bmi2(dst: *mut (), dstCapacity: u64, cSrc: *const (), cSrcSize: u64, maxLog: u32, workSpace: *mut (), wkspSize: u64, bmi2: i32) -> u64 {
if bmi2 != 0 {
return unsafe { FSE_decompress_wksp_body_bmi2(dst, dstCapacity, cSrc, cSrcSize, maxLog, workSpace, wkspSize) };
}
FSE_decompress_wksp_body_default(dst, dstCapacity, cSrc, cSrcSize, maxLog, workSpace, wkspSize)
}
