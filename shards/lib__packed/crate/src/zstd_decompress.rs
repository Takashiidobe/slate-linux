
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0+ OR BSD-3-Clause
//!
//! Copyright (c) Meta Platforms, Inc. and affiliates.
//! All rights reserved.
//!
//! This source code is licensed under both the BSD-style license (found in the
//! LICENSE file in the root directory of this source tree) and the GPLv2 (found
//! in the COPYING file in the root directory of this source tree).
//! You may select, at your option, one of the above-listed licenses.
//! UNUSED_ATTR tells the compiler it is okay if the function is unused.
//! All magic numbers are supposed read/written to/from files/memory using little-endian convention
//! Compiler-specific
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! The minimum workspace size for the `workSpace` used in
//! HUF_readDTableX1_wksp() and HUF_readDTableX2_wksp().
//!
//! The space used depends on HUF_TABLELOG_MAX, ranging from ~1500 bytes when
//! HUF_TABLE_LOG_MAX=12 to ~1850 bytes when HUF_TABLE_LOG_MAX=15.
//! Buffer overflow errors may potentially occur if code modifications result in
//! a required workspace size greater than that specified in the following
//! macro.
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
// FORCE_INLINE_TEMPLATE is used to define C "templates", which take constant
// parameters. They must be inlined for the compiler to eliminate the constant
// branches.
// Define ZSTD_FALLTHROUGH macro for annotating switch case with the 'fallthrough' attribute.
// - C23: https://en.cppreference.com/w/c/language/attributes/fallthrough
// - CPP17: https://en.cppreference.com/w/cpp/language/attributes/fallthrough
// - Else: __attribute__((__fallthrough__))
// Zstd relies on pointer overflow in its decompressor.
// We add this attribute to functions that rely on pointer overflow.
//
// older versions of clang [3.7, 5.0) will warn that pointer-overflow is ignored.
// disable assert (default)
// disabled
// disabled
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
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
// Optional: only supported since gcc >= 15, clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
macro_rules! BOUNDED__2 {
($min:expr, $val:expr, $max:expr) => {
MAX__3!($min, if ($val) < $max { $val } else { $max })
};
}

macro_rules! CHECK_DBOUNDS {
($p:expr, $v:expr) => {
{
loop {
if ZSTD_dParam_withinBounds(($p) as u32, $v) == 0 {
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
return ERROR__13!();
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

macro_rules! ERROR__16 {
() => {
ZSTD_ERROR__20!()
};
}

macro_rules! ERROR {
() => {
ZSTD_ERROR__2!()
};
}

macro_rules! ERROR__10 {
() => {
ZSTD_ERROR__21!()
};
}

macro_rules! ERROR__12 {
() => {
ZSTD_ERROR__16!()
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

macro_rules! ERROR__19 {
() => {
ZSTD_ERROR__11!()
};
}

macro_rules! ERROR__20 {
() => {
ZSTD_ERROR__23!()
};
}

macro_rules! ERROR__18 {
() => {
ZSTD_ERROR__12!()
};
}

macro_rules! ERROR__3 {
() => {
ZSTD_ERROR__5!()
};
}

macro_rules! ERROR__11 {
() => {
ZSTD_ERROR__19!()
};
}

macro_rules! ERROR__22 {
() => {
ZSTD_ERROR__18!()
};
}

macro_rules! ERROR__5 {
() => {
ZSTD_ERROR__4!()
};
}

macro_rules! ERROR__4 {
() => {
ZSTD_ERROR__6!()
};
}

macro_rules! ERROR__7 {
() => {
ZSTD_ERROR__8!()
};
}

macro_rules! ERROR__23 {
() => {
ZSTD_ERROR__13!()
};
}

macro_rules! ERROR__13 {
() => {
ZSTD_ERROR__9!()
};
}

macro_rules! ERROR__9 {
() => {
ZSTD_ERROR__10!()
};
}

macro_rules! ERROR__14 {
() => {
ZSTD_ERROR__22!()
};
}

macro_rules! MAX__2 {
($a:expr, $b:expr) => {
if ($a) > (($b) as u32) { $a } else { ($b) as u32 }
};
}

macro_rules! MAX__3 {
($a:expr, $b:expr) => {
if (($a) as u64) > $b { ($a) as u64 } else { $b }
};
}

macro_rules! MAX {
($a:expr, $b:expr) => {
if ($a) > $b { $a } else { $b }
};
}

macro_rules! MAX__5 {
($a:expr, $b:expr) => {
if ($a) > (($b) as u64) { $a } else { ($b) as u64 }
};
}

macro_rules! MIN__5 {
($a:expr, $b:expr) => {
if (($a) as u64) < $b { ($a) as u64 } else { $b }
};
}

macro_rules! MIN__2 {
($a:expr, $b:expr) => {
if ($a) < (($b) as u64) { $a } else { ($b) as u64 }
};
}

macro_rules! MIN {
($a:expr, $b:expr) => {
if ($a) < $b { $a } else { $b }
};
}

macro_rules! ZSTD_ERROR__9 {
() => {
42i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__10 {
() => {
60i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__11 {
() => {
82i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__12 {
() => {
10i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__6 {
() => {
30i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__13 {
() => {
80i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__2 {
() => {
20i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__8 {
() => {
64i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__16 {
() => {
22i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__4 {
() => {
1i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__18 {
() => {
104i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__19 {
() => {
16i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__3 {
() => {
72i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__20 {
() => {
74i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__21 {
() => {
40i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__5 {
() => {
70i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__22 {
() => {
14i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR {
() => {
120i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__23 {
() => {
32i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_FRAMEHEADERSIZE_MIN {
($format:expr) => {
if (($format) as u32) == 0 { 6i32 } else { 2i32 }
};
}

/// minimum input size required to query frame header size
macro_rules! ZSTD_FRAMEHEADERSIZE_PREFIX {
($format:expr) => {
if (($format) as u32) == 0 { 5i32 } else { 1i32 }
};
}

/// ---- static assert (debug) ---
macro_rules! ZSTD_STATIC_ASSERT {
() => {
DEBUG_STATIC_ASSERT!();
};
}

macro_rules! ZSTD_calloc {
($n:expr, $s:expr) => {
{
let mut __v: *mut ();
$n;
$s;
__v = core::ptr::null_mut::<()>();
__v
}
};
}

macro_rules! ZSTD_free {
($p:expr) => {
$p;
};
}

macro_rules! ZSTD_malloc {
($s:expr) => {
{
let mut __v: *mut ();
$s;
__v = core::ptr::null_mut::<()>();
__v
}
};
}

macro_rules! ZSTD_memcpy__2 {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memcpy(($d) as *mut (), $s, $n) };
};
}

macro_rules! ZSTD_memcpy {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memcpy(($d) as *mut (), ($s) as *const (), $n) };
};
}

macro_rules! ZSTD_memcpy__3 {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memcpy($d, ($s) as *const (), $n) };
};
}

macro_rules! ZSTD_memcpy__5 {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memcpy($d, $s, $n) };
};
}

macro_rules! ZSTD_memmove__2 {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memmove($d, $s, $n) };
};
}

macro_rules! ZSTD_memset__4 {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memset($d, $s, $n) };
};
}

macro_rules! ZSTD_memset {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memset(($d) as *mut (), $s, $n) };
};
}

macro_rules! ZSTD_memset__2 {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memset($d, ($s) as i32, $n) };
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
macro_rules! __get_unaligned_t__3 {
($type:ty, $ptr:expr) => {
{
let mut __v: $type;
core::ptr::null_mut::<u16>();
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
macro_rules! __put_unaligned_t__2 {
($type:ty, $val:expr, $ptr:expr) => {
loop {
let mut __put_unaligned_val: $type = $val;
unsafe { __builtin_memcpy($ptr, core::ptr::addr_of_mut!(__put_unaligned_val) as *const (), core::mem::size_of::<$type>() as u64) };
if true {
break;
}
}
};
}

unsafe extern "C" {
    fn ZSTD_getErrorCode(arg0: u64) -> u32;
    fn ZSTD_freeDDict(arg0: *mut ZSTD_DDict_s) -> u64;
    fn ZSTD_getDictID_fromDDict(arg0: *const ZSTD_DDict_s) -> u32;
    fn ZSTD_sizeof_DDict(arg0: *const ZSTD_DDict_s) -> u64;
    fn ZSTD_createDDict_advanced(arg0: *const (), arg1: u64, arg2: u32, arg3: u32, arg4: __SlateRecord_zstd_lib_h_1837) -> *mut ZSTD_DDict_s;
    #[link_name = "memset"]
    fn __builtin_memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn FSE_readNCount(arg0: *mut i16, arg1: *mut u32, arg2: *mut u32, arg3: *const (), arg4: u64) -> u64;
    fn HUF_readDTableX2_wksp(arg0: *mut u32, arg1: *const (), arg2: u64, arg3: *mut (), arg4: u64, arg5: i32) -> u64;
    fn xxh64(arg0: *const (), arg1: u64, arg2: u64) -> u64;
    fn xxh64_reset(arg0: *mut xxh64_state, arg1: u64);
    fn xxh64_update(arg0: *mut xxh64_state, arg1: *const (), arg2: u64) -> i32;
    fn xxh64_digest(arg0: *const xxh64_state) -> u64;
    fn ZSTD_getcBlockSize(arg0: *const (), arg1: u64, arg2: *mut __SlateRecord_zstd_internal_h_295) -> u64;
    fn ZSTD_checkContinuity(arg0: *mut ZSTD_DCtx_s, arg1: *const (), arg2: u64);
    fn ZSTD_DDict_dictContent(arg0: *const ZSTD_DDict_s) -> *const ();
    fn ZSTD_DDict_dictSize(arg0: *const ZSTD_DDict_s) -> u64;
    fn ZSTD_copyDDictParameters(arg0: *mut ZSTD_DCtx_s, arg1: *const ZSTD_DDict_s);
    fn ZSTD_decompressBlock_internal(arg0: *mut ZSTD_DCtx_s, arg1: *mut (), arg2: u64, arg3: *const (), arg4: u64, arg5: u32) -> u64;
    fn ZSTD_buildFSETable(arg0: *mut __SlateRecord_zstd_decompress_internal_h_73, arg1: *const i16, arg2: u32, arg3: *const u32, arg4: *const u8, arg5: u32, arg6: *mut (), arg7: u64, arg8: i32);
    #[link_name = "memmove"]
    fn __builtin_memmove(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
}

// valid since v0.8.0
// valid since v0.7.0
// all 16 values, from 0x184D2A50 to 0x184D2A5F, signal the beginning of a skippable frame
// ZSTD_getFrameContentSize() : requires v1.3.0+
// `src` should point to the start of a ZSTD encoded frame.
// `srcSize` must be at least as large as the frame header.
//           hint : any size >= `ZSTD_frameHeaderSize_max` is large enough.
// @return : - decompressed size of `src` frame content, if known
//           - ZSTD_CONTENTSIZE_UNKNOWN if the size cannot be determined
//           - ZSTD_CONTENTSIZE_ERROR if an error occurred (e.g. invalid magic number, srcSize too small)
//  note 1 : a 0 return value means the frame is valid but "empty".
//           When invoking this method on a skippable frame, it will return 0.
//  note 2 : decompressed size is an optional field, it may not be present (typically in streaming mode).
//           When `return==ZSTD_CONTENTSIZE_UNKNOWN`, data to decompress could be any size.
//           In which case, it's necessary to use streaming mode to decompress data.
//           Optionally, application can rely on some implicit limit,
//           as ZSTD_decompress() only needs an upper bound of decompressed size.
//           (For example, data could be necessarily cut into blocks <= 16 KB).
//  note 3 : decompressed size is always present when compression is completed using single-pass functions,
//           such as ZSTD_compress(), ZSTD_compressCCtx() ZSTD_compress_usingDict() or ZSTD_compress_usingCDict().
//  note 4 : decompressed size can be very large (64-bits value),
//           potentially larger than what local system can handle as a single memory segment.
//           In which case, it's necessary to use streaming mode to decompress data.
//  note 5 : If source is untrusted, decompressed size could be wrong or intentionally modified.
//           Always ensure return value fits within application's authorized limits.
//           Each application can set its own limits.
//  note 6 : This function replaces ZSTD_getDecompressedSize()
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_lib_h_527 {
    error: u64,
    lowerBound: i32,
    upperBound: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ZSTD_inBuffer_s {
    src: *const (),
    size: u64,
    pos: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ZSTD_outBuffer_s {
    dst: *mut (),
    size: u64,
    pos: u64,
}


// can be useful for static allocation
// The minimum valid max blocksize. Maximum blocksizes smaller than this make compressBound() inaccurate.
// by default, the streaming decoder will refuse any frame
// requiring larger than (1<<ZSTD_WINDOWLOG_LIMIT_DEFAULT) window size,
// to preserve host's memory from unreasonable requirements.
// This limit can be overridden using ZSTD_DCtx_setParameter(,ZSTD_d_windowLogMax,).
// The limit does not apply for one-pass decoders (such as ZSTD_decompress()), since no additional memory is allocated
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_lib_h_1489 {
    frameContentSize: u64,
    windowSize: u64,
    blockSizeMax: u32,
    frameType: u32,
    headerSize: u32,
    dictID: u32,
    checksumFlag: u32,
    _reserved1: u32,
    _reserved2: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ZSTD_DDict_s {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_lib_h_1837 {
    customAlloc: Option<unsafe extern "C" fn(*mut (), u64) -> *mut ()>,
    customFree: Option<unsafe extern "C" fn(*mut (), *mut ())>,
    opaque: *mut (),
}


static ZSTD_defaultCMem: __SlateConst<__SlateRecord_zstd_lib_h_1837> = __SlateConst(__SlateRecord_zstd_lib_h_1837 { customAlloc: None, customFree: None, opaque: core::ptr::null_mut::<()>() });

// ZSTD_d_format
// experimental parameter,
// allowing selection between ZSTD_format_e input compression formats
// ZSTD_d_stableOutBuffer
// Experimental parameter.
// Default is 0 == disabled. Set to 1 to enable.
//
// Tells the decompressor that the ZSTD_outBuffer will ALWAYS be the same
// between calls, except for the modifications that zstd makes to pos (the
// caller must not modify pos). This is checked by the decompressor, and
// decompression will fail if it ever changes. Therefore the ZSTD_outBuffer
// MUST be large enough to fit the entire decompressed frame. This will be
// checked when the frame content size is known. The data in the ZSTD_outBuffer
// in the range [dst, dst + pos) MUST not be modified during decompression
// or you will get data corruption.
//
// When this flag is enabled zstd won't allocate an output buffer, because
// it can write directly to the ZSTD_outBuffer, but it will still allocate
// an input buffer large enough to fit any compressed block. This will also
// avoid the memcpy() from the internal output buffer to the ZSTD_outBuffer.
// If you need to avoid the input buffer allocation use the buffer-less
// streaming API.
//
// NOTE: So long as the ZSTD_outBuffer always points to valid memory, using
// this flag is ALWAYS memory safe, and will never access out-of-bounds
// memory. However, decompression WILL fail if you violate the preconditions.
//
// WARNING: The data in the ZSTD_outBuffer in the range [dst, dst + pos) MUST
// not be modified during decompression or you will get data corruption. This
// is because zstd needs to reference data in the ZSTD_outBuffer to regenerate
// matches. Normally zstd maintains its own buffer for this purpose, but passing
// this flag tells zstd to use the user provided buffer.
// ZSTD_d_forceIgnoreChecksum
// Experimental parameter.
// Default is 0 == disabled. Set to 1 to enable
//
// Tells the decompressor to skip checksum validation during decompression, regardless
// of whether checksumming was specified during compression. This offers some
// slight performance benefits, and may be useful for debugging.
// Param has values of type ZSTD_forceIgnoreChecksum_e
// ZSTD_d_refMultipleDDicts
// Experimental parameter.
// Default is 0 == disabled. Set to 1 to enable
//
// If enabled and dctx is allocated on the heap, then additional memory will be allocated
// to store references to multiple ZSTD_DDict. That is, multiple calls of ZSTD_refDDict()
// using a given ZSTD_DCtx, rather than overwriting the previous DDict reference, will instead
// store all references. At decompression time, the appropriate dictID is selected
// from the set of DDicts based on the dictID in the frame.
//
// Usage is simply calling ZSTD_refDDict() on multiple dict buffers.
//
// Param has values of byte ZSTD_refMultipleDDicts_e
//
// WARNING: Enabling this parameter and calling ZSTD_DCtx_refDDict(), will trigger memory
// allocation for the hash table. ZSTD_freeDCtx() also frees this memory.
// Memory is allocated as per ZSTD_DCtx::customMem.
//
// Although this function allocates memory for the table, the user is still responsible for
// memory management of the underlying ZSTD_DDict* themselves.
// ZSTD_d_disableHuffmanAssembly
// Set to 1 to disable the Huffman assembly implementation.
// The default value is 0, which allows zstd to use the Huffman assembly
// implementation if available.
//
// This parameter can be used to disable Huffman assembly at runtime.
// If you want to disable it at compile time you can define the macro
// ZSTD_DISABLE_ASM.
// ZSTD_d_maxBlockSize
// Allowed values are between 1KB and ZSTD_BLOCKSIZE_MAX (128KB).
// The default is ZSTD_BLOCKSIZE_MAX, and setting to 0 will set to the default.
//
// Forces the decompressor to reject blocks whose content size is
// larger than the configured maxBlockSize. When maxBlockSize is
// larger than the windowSize, the windowSize is used instead.
// This saves memory on the decoder when you know all blocks are small.
//
// This option is typically used in conjunction with ZSTD_c_maxBlockSize.
//
// WARNING: This causes the decoder to reject otherwise valid frames
// that have block sizes larger than the configured maxBlockSize.
fn ZSTD_customMalloc(size: u64, mut customMem: __SlateRecord_zstd_lib_h_1837) -> *mut () {
if customMem.customAlloc != None {
return unsafe { customMem.customAlloc.unwrap()(customMem.opaque, size) };
}
ZSTD_malloc!(size)
}

fn ZSTD_customCalloc(size: u64, mut customMem: __SlateRecord_zstd_lib_h_1837) -> *mut () {
if customMem.customAlloc != None {
// calloc implemented as malloc+memset;
// not as efficient as calloc, but next best guess for custom malloc
let ptr: *mut () = unsafe { customMem.customAlloc.unwrap()(customMem.opaque, size) };
ZSTD_memset__4!(ptr, 0i32, size);
return ptr;
}
ZSTD_calloc!(1i32, size)
}

fn ZSTD_customFree(ptr: *mut (), mut customMem: __SlateRecord_zstd_lib_h_1837) {
if !ptr.is_null() {
if customMem.customFree != None {
unsafe { customMem.customFree.unwrap()(customMem.opaque, ptr) };
} else {
ZSTD_free!(ptr);
}
}
}

fn ERR_isError(code: u64) -> u32 {
(code > ERROR__8!()) as u32
}

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
// Unconditionally return the specified error.
//
// In debug modes, prints additional information.
// If the provided expression evaluates to an error code, returns that error code.
//
// In debug modes, prints additional information.
fn get_unaligned_le16(p: *const ()) -> u16 {
__get_unaligned_t__3!(u16, p)
}

fn get_unaligned_le32(p: *const ()) -> u32 {
__get_unaligned_t!(u32, p)
}

#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[repr(transparent)]
struct __SlateConst<T>(T);

unsafe impl<T> Sync for __SlateConst<T> {
}

fn get_unaligned_le64(p: *const ()) -> u64 {
__get_unaligned_t__5!(u64, p)
}

fn put_unaligned_le32(val: u32, p: *mut ()) {
__put_unaligned_t__2!(u32, val, p);
}

fn MEM_readLE16(memPtr: *const ()) -> u16 {
get_unaligned_le16(memPtr)
}

fn MEM_readLE32(memPtr: *const ()) -> u32 {
get_unaligned_le32(memPtr)
}

fn MEM_writeLE32(memPtr: *mut (), val32: u32) {
put_unaligned_le32(val32, memPtr);
}

fn MEM_readLE64(memPtr: *const ()) -> u64 {
get_unaligned_le64(memPtr)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cpu_h_28 {
    f1c: u32,
    f1d: u32,
    f7b: u32,
    f7c: u32,
}


fn ZSTD_cpuid() -> __SlateRecord_cpu_h_28 {
let mut f1c: u32 = 0;
let mut f1d: u32 = 0;
let mut f7b: u32 = 0;
let mut f7c: u32 = 0;
let mut n: u32 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("mov %rbx, {0:r}\ncpuid\nmov {0:r}, %rbx", out(reg) _, lateout("ax") __t0, in("ax") (0 as i32) as u32, out("cx") _, out("dx") _, options(pure, nomem, nostack, att_syntax));
n = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
if n >= 1 {
let mut f1a: u32 = 0;
unsafe {
let __t1: u32;
let __t2: u32;
let __t3: u32;
core::arch::asm!("mov %rbx, {0:r}\ncpuid\nmov {0:r}, %rbx", out(reg) _, lateout("ax") __t1, lateout("cx") __t2, lateout("dx") __t3, in("ax") (1 as i32) as u32, options(pure, nomem, nostack, att_syntax));
f1a = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
f1c = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
f1d = unsafe { core::mem::transmute::<u32, u32>(__t3 as u32) };
}
}
if n >= 7 {
let mut f7a: u32 = 0;
unsafe {
let __t4: u32;
let __t5: u32;
let __t6: u32;
core::arch::asm!("xchg {0:r}, %rbx\ncpuid\nxchg {0:r}, %rbx", out(reg) __t5, lateout("ax") __t4, lateout("cx") __t6, in("ax") (7 as i32) as u32, in("cx") (0 as i32) as u32, out("dx") _, options(pure, nomem, nostack, att_syntax));
f7a = unsafe { core::mem::transmute::<u32, u32>(__t4 as u32) };
f7b = unsafe { core::mem::transmute::<u32, u32>(__t5 as u32) };
f7c = unsafe { core::mem::transmute::<u32, u32>(__t6 as u32) };
}
}
let mut cpuid: __SlateRecord_cpu_h_28 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(cpuid) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
cpuid.f1c = f1c;
cpuid.f1d = f1d;
cpuid.f7b = f7b;
cpuid.f7c = f7c;
return cpuid;
}

// cpuid(1): Processor Info and Feature Bits.
// cpuid(7): Extended Features.
fn ZSTD_cpuid_bmi1(cpuid: __SlateRecord_cpu_h_28) -> i32 {
(cpuid.f7b & 8u32 != 0) as i32
}

fn ZSTD_cpuid_bmi2(cpuid: __SlateRecord_cpu_h_28) -> i32 {
(cpuid.f7b & 256u32 != 0) as i32
}

fn ZSTD_countLeadingZeros32(val: u32) -> u32 {
(val.leading_zeros() as i32) as u32
}

fn ZSTD_highbit32(val: u32) -> u32 {
31u32.wrapping_sub(ZSTD_countLeadingZeros32(val))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xxh64_state {
    total_len: u64,
    v1: u64,
    v2: u64,
    v3: u64,
    v4: u64,
    mem64: [u64; 4],
    memsize: u32,
}


// for inlining
// number of repcodes
static repStartValue: [u32; 3] = [1, 4, 8];

static ZSTD_fcs_fieldSize: __SlateAlign16<[u64; 4]> = __SlateAlign16([0u64, 2u64, 4u64, 8u64]);

static ZSTD_did_fieldSize: __SlateAlign16<[u64; 4]> = __SlateAlign16([0u64, 1u64, 2u64, 4u64]);

// magic number size
// C standard doesn't allow `static const` variable to be init using another `static const` variable
static ZSTD_blockHeaderSize: u64 = 3;

static LL_bits: __SlateAlign16<[u8; 36]> = __SlateAlign16([0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 1u8, 1u8, 1u8, 2u8, 2u8, 3u8, 3u8, 4u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8, 12u8, 13u8, 14u8, 15u8, 16u8]);

// for static allocation
static ML_bits: __SlateAlign16<[u8; 53]> = __SlateAlign16([0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 1u8, 1u8, 1u8, 2u8, 2u8, 3u8, 3u8, 4u8, 4u8, 5u8, 7u8, 8u8, 9u8, 10u8, 11u8, 12u8, 13u8, 14u8, 15u8, 16u8]);

// for static allocation
// for static allocation
fn ZSTD_limitCopy(dst: *mut (), dstCapacity: u64, src: *const (), srcSize: u64) -> u64 {
let length: u64 = MIN!(dstCapacity, srcSize);
if length > 0 {
ZSTD_memcpy__5!(dst, src, length);
}
length
}

// define "workspace is too large" as this number of times larger than needed
// when workspace is continuously too large
// during at least this number of times,
// context's memory usage is considered wasteful,
// because it's sized to handle a worst case scenario which rarely happens.
// In which case, resize it down to free some memory
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_internal_h_282 {
    nbBlocks: u64,
    compressedSize: u64,
    decompressedBound: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_internal_h_295 {
    blockType: u32,
    lastBlock: u32,
    origSize: u32,
}


fn ZSTD_cpuSupportsBmi2() -> i32 {
let cpuid: __SlateRecord_cpu_h_28 = ZSTD_cpuid();
let __v0: bool = ZSTD_cpuid_bmi1(cpuid) != 0 && ZSTD_cpuid_bmi2(cpuid) != 0;
__v0 as i32
}

static LL_base: __SlateAlign16<[u32; 36]> = __SlateAlign16([0u32, 1u32, 2u32, 3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32, 10u32, 11u32, 12u32, 13u32, 14u32, 15u32, 16u32, 18u32, 20u32, 22u32, 24u32, 28u32, 32u32, 40u32, 48u32, 64u32, 128u32, 256u32, 512u32, 1024u32, 2048u32, 4096u32, 8192u32, 16384u32, 32768u32, 65536u32]);

static OF_base: __SlateAlign16<[u32; 32]> = __SlateAlign16([0u32, 1u32, 1u32, 5u32, 13u32, 29u32, 61u32, 125u32, 253u32, 509u32, 1021u32, 2045u32, 4093u32, 8189u32, 16381u32, 32765u32, 65533u32, 131069u32, 262141u32, 524285u32, 1048573u32, 2097149u32, 4194301u32, 8388605u32, 16777213u32, 33554429u32, 67108861u32, 134217725u32, 268435453u32, 536870909u32, 1073741821u32, 2147483645u32]);

static OF_bits: __SlateAlign16<[u8; 32]> = __SlateAlign16([0u8, 1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8, 12u8, 13u8, 14u8, 15u8, 16u8, 17u8, 18u8, 19u8, 20u8, 21u8, 22u8, 23u8, 24u8, 25u8, 26u8, 27u8, 28u8, 29u8, 30u8, 31u8]);

static ML_base: __SlateAlign16<[u32; 53]> = __SlateAlign16([3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32, 10u32, 11u32, 12u32, 13u32, 14u32, 15u32, 16u32, 17u32, 18u32, 19u32, 20u32, 21u32, 22u32, 23u32, 24u32, 25u32, 26u32, 27u32, 28u32, 29u32, 30u32, 31u32, 32u32, 33u32, 34u32, 35u32, 37u32, 39u32, 41u32, 43u32, 47u32, 51u32, 59u32, 67u32, 83u32, 99u32, 131u32, 259u32, 515u32, 1027u32, 2051u32, 4099u32, 8195u32, 16387u32, 32771u32, 65539u32]);

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_decompress_internal_h_73 {
    nextState: u16,
    nbAdditionalBits: u8,
    nbBits: u8,
    baseValue: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_decompress_internal_h_88 {
    LLTable: [__SlateRecord_zstd_decompress_internal_h_73; 513],
    OFTable: [__SlateRecord_zstd_decompress_internal_h_73; 257],
    MLTable: [__SlateRecord_zstd_decompress_internal_h_73; 513],
    hufTable: [u32; 4097],
    rep: [u32; 3],
    workspace: [u32; 157],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_decompress_internal_h_109 {
    ddictPtrTable: *mut *const ZSTD_DDict_s,
    ddictPtrTableSize: u64,
    ddictPtrCount: u64,
}


// extra buffer, compensates when dst is not large enough to store litBuffer
#[repr(C)]
#[derive(Clone, Copy)]
struct ZSTD_DCtx_s {
    LLTptr: *const __SlateRecord_zstd_decompress_internal_h_73,
    MLTptr: *const __SlateRecord_zstd_decompress_internal_h_73,
    OFTptr: *const __SlateRecord_zstd_decompress_internal_h_73,
    HUFptr: *const u32,
    entropy: __SlateRecord_zstd_decompress_internal_h_88,
    workspace: [u32; 640],
    previousDstEnd: *const (),
    prefixStart: *const (),
    virtualStart: *const (),
    dictEnd: *const (),
    expected: u64,
    fParams: __SlateRecord_zstd_lib_h_1489,
    processedCSize: u64,
    decodedSize: u64,
    bType: u32,
    stage: u32,
    litEntropy: u32,
    fseEntropy: u32,
    xxhState: xxh64_state,
    headerSize: u64,
    format: u32,
    forceIgnoreChecksum: u32,
    validateChecksum: u32,
    litPtr: *const u8,
    customMem: __SlateRecord_zstd_lib_h_1837,
    litSize: u64,
    rleSize: u64,
    staticSize: u64,
    isFrameDecompression: i32,
    bmi2: i32,
    ddictLocal: *mut ZSTD_DDict_s,
    ddict: *const ZSTD_DDict_s,
    dictID: u32,
    ddictIsCold: i32,
    dictUses: i32,
    ddictSet: *mut __SlateRecord_zstd_decompress_internal_h_109,
    refMultipleDDicts: u32,
    disableHufAsm: i32,
    maxBlockSizeParam: i32,
    streamStage: u32,
    inBuff: *mut u8,
    inBuffSize: u64,
    inPos: u64,
    maxWindowSize: u64,
    outBuff: *mut u8,
    outBuffSize: u64,
    outStart: u64,
    outEnd: u64,
    lhSize: u64,
    hostageByte: u32,
    noForwardProgress: i32,
    outBufferMode: u32,
    expectedOutBuffer: ZSTD_outBuffer_s,
    litBuffer: *mut u8,
    litBufferEnd: *const u8,
    litBufferLocation: u32,
    litExtraBuffer: [u8; 65568],
    headerBuffer: [u8; 18],
    oversizedDuration: u64,
}


// Tuning parameters
//
// HEAPMODE :
// Select how default decompression function ZSTD_decompress() allocates its context,
// on stack (0), or into heap (1, default; requires malloc()).
// Note that functions with explicit context such as ZSTD_decompressDCtx() are unaffected.
// LEGACY_SUPPORT :
// if set to 1+, ZSTD_decompress() can decode older formats (v0.1+)
// MAXWINDOWSIZE_DEFAULT :
// maximum window size accepted by DStream __by default__.
// Frames requiring more memory will be rejected.
// It's possible to set a different limit using ZSTD_DCtx_setMaxWindowSize().
// NO_FORWARD_PROGRESS_MAX :
// maximum allowed nb of calls to ZSTD_decompressStream()
// without any forward progress
// (defined as: no byte read from input, and no byte flushed to output)
// before triggering an error.
// Dependencies
// ZSTD_memcpy, ZSTD_memmove, ZSTD_memset
// ZSTD_customMalloc, ZSTD_customCalloc, ZSTD_customFree
// blockProperties_t
// low level memory routines
// ZSTD_highbit32
// xxh64_reset, xxh64_update, xxh64_digest, XXH64
// ZSTD_DCtx
// ZSTD_DDictDictContent
// ZSTD_decompressBlock_internal
// Multiple DDicts Hashset internals *
// These two constants represent SIZE_MULT/COUNT_MULT load factor without using a float.
// Currently, that means a 0.75 load factor.
// So, if count * COUNT_MULT / size * SIZE_MULT != 0, then we've exceeded
// the load factor of the ddict hash set.
/// Hash function to determine starting position of dict insertion within the table
/// Returns an index between [0, hashSet->ddictPtrTableSize]
fn ZSTD_DDictHashSet_getIndex(hashSet: *const __SlateRecord_zstd_decompress_internal_h_109, mut dictID: u32) -> u64 {
let hash: u64 = unsafe { xxh64(core::ptr::addr_of_mut!(dictID) as *const (), core::mem::size_of::<u32>() as u64, 0u64) };
// DDict ptr table size is a multiple of 2, use size - 1 as mask to get index within [0, hashSet->ddictPtrTableSize)
hash & unsafe { (*hashSet).ddictPtrTableSize }.wrapping_sub(1)
}

/// Adds DDict to a hashset without resizing it.
/// If inserting a DDict with a dictID that already exists in the set, replaces the one in the set.
/// Returns 0 if successful, or a zstd error code if something went wrong.
fn ZSTD_DDictHashSet_emplaceDDict(hashSet: *mut __SlateRecord_zstd_decompress_internal_h_109, ddict: *const ZSTD_DDict_s) -> u64 {
let dictID: u32 = unsafe { ZSTD_getDictID_fromDDict(ddict) };
let mut idx: u64 = ZSTD_DDictHashSet_getIndex(hashSet as *const __SlateRecord_zstd_decompress_internal_h_109, dictID);
let idxRangeMask: u64 = unsafe { (*hashSet).ddictPtrTableSize }.wrapping_sub(1);
loop {
if (unsafe { (*hashSet).ddictPtrCount }) == unsafe { (*hashSet).ddictPtrTableSize } {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"Hash set is full!".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__5!();
}
if true {
break;
}
}
while false {
}
while !unsafe { *unsafe { (*hashSet).ddictPtrTable }.wrapping_offset(idx as isize) }.is_null() {
// Replace existing ddict if inserting ddict with same dictID
if (unsafe { ZSTD_getDictID_fromDDict(*unsafe { (*hashSet).ddictPtrTable }.wrapping_offset(idx as isize)) }) == dictID {
while false {
}
unsafe {
*unsafe { (*hashSet).ddictPtrTable }.wrapping_offset(idx as isize) = ddict;
}
return 0;
}
idx &= idxRangeMask;
idx = idx.wrapping_add(1);
}
while false {
}
unsafe {
*unsafe { (*hashSet).ddictPtrTable }.wrapping_offset(idx as isize) = ddict;
}
let __v14: u64 = unsafe { (*hashSet).ddictPtrCount };
unsafe {
(*hashSet).ddictPtrCount = __v14.wrapping_add(1);
}
0
}

/// Expands hash table by factor of DDICT_HASHSET_RESIZE_FACTOR and
/// rehashes all values, allocates new table, frees old table.
/// Returns 0 on success, otherwise a zstd error code.
fn ZSTD_DDictHashSet_expand(hashSet: *mut __SlateRecord_zstd_decompress_internal_h_109, customMem: __SlateRecord_zstd_lib_h_1837) -> u64 {
let newTableSize: u64 = unsafe { (*hashSet).ddictPtrTableSize }.wrapping_mul(2);
let newTable: *mut *const ZSTD_DDict_s = ZSTD_customCalloc((core::mem::size_of::<*mut ZSTD_DDict_s>() as u64).wrapping_mul(newTableSize), customMem) as *mut *const ZSTD_DDict_s;
let oldTable: *mut *const ZSTD_DDict_s = unsafe { (*hashSet).ddictPtrTable };
let oldTableSize: u64 = unsafe { (*hashSet).ddictPtrTableSize };
while false {
}
loop {
if newTable.is_null() {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"Expanded hashset allocation failed!".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__7!();
}
if true {
break;
}
}
unsafe {
(*hashSet).ddictPtrTable = newTable;
}
unsafe {
(*hashSet).ddictPtrTableSize = newTableSize;
}
unsafe {
(*hashSet).ddictPtrCount = 0;
}
for i in 0..oldTableSize {
if !unsafe { *oldTable.wrapping_offset(i as isize) }.is_null() {
loop {
let err_code: u64 = ZSTD_DDictHashSet_emplaceDDict(hashSet, unsafe { *oldTable.wrapping_offset(i as isize) });
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
}
}
ZSTD_customFree(oldTable as *mut (), customMem);
while false {
}
0
}

/// Fetches a DDict with the given dictID
/// Returns the ZSTD_DDict* with the requested dictID. If it doesn't exist, then returns NULL.
fn ZSTD_DDictHashSet_getDDict(hashSet: *mut __SlateRecord_zstd_decompress_internal_h_109, dictID: u32) -> *const ZSTD_DDict_s {
let mut idx: u64 = ZSTD_DDictHashSet_getIndex(hashSet as *const __SlateRecord_zstd_decompress_internal_h_109, dictID);
let idxRangeMask: u64 = unsafe { (*hashSet).ddictPtrTableSize }.wrapping_sub(1);
while false {
}
loop {
let currDictID: u64 = (unsafe { ZSTD_getDictID_fromDDict(*unsafe { (*hashSet).ddictPtrTable }.wrapping_offset(idx as isize)) }) as u64;
if currDictID == (dictID as u64) || currDictID == 0 {
// currDictID == 0 implies a NULL ddict entry
break;
} else {
idx &= idxRangeMask; // Goes to start of table when we reach the end
idx = idx.wrapping_add(1);
}
}
while false {
}
unsafe { *unsafe { (*hashSet).ddictPtrTable }.wrapping_offset(idx as isize) }
}

/// Allocates space for and returns a ddict hash set
/// The hash set's ZSTD_DDict* table has all values automatically set to NULL to begin with.
/// Returns NULL if allocation failed.
fn ZSTD_createDDictHashSet(customMem: __SlateRecord_zstd_lib_h_1837) -> *mut __SlateRecord_zstd_decompress_internal_h_109 {
let ret: *mut __SlateRecord_zstd_decompress_internal_h_109 = ZSTD_customMalloc(core::mem::size_of::<__SlateRecord_zstd_decompress_internal_h_109>() as u64, customMem) as *mut __SlateRecord_zstd_decompress_internal_h_109;
while false {
}
if ret.is_null() {
return core::ptr::null_mut::<__SlateRecord_zstd_decompress_internal_h_109>();
}
unsafe {
(*ret).ddictPtrTable = ZSTD_customCalloc(64u64.wrapping_mul(core::mem::size_of::<*mut ZSTD_DDict_s>() as u64), customMem) as *mut *const ZSTD_DDict_s;
}
if unsafe { (*ret).ddictPtrTable }.is_null() {
ZSTD_customFree(ret as *mut (), customMem);
return core::ptr::null_mut::<__SlateRecord_zstd_decompress_internal_h_109>();
}
unsafe {
(*ret).ddictPtrTableSize = 64;
}
unsafe {
(*ret).ddictPtrCount = 0;
}
ret
}

/// Frees the table of ZSTD_DDict* within a hashset, then frees the hashset itself.
/// Note: The ZSTD_DDict* within the table are NOT freed.
fn ZSTD_freeDDictHashSet(hashSet: *mut __SlateRecord_zstd_decompress_internal_h_109, customMem: __SlateRecord_zstd_lib_h_1837) {
while false {
}
if !hashSet.is_null() && !unsafe { (*hashSet).ddictPtrTable }.is_null() {
ZSTD_customFree((unsafe { (*hashSet).ddictPtrTable }) as *mut (), customMem);
}
if !hashSet.is_null() {
ZSTD_customFree(hashSet as *mut (), customMem);
}
}

/// Public function: Adds a DDict into the ZSTD_DDictHashSet, possibly triggering a resize of the hash set.
/// Returns 0 on success, or a ZSTD error.
fn ZSTD_DDictHashSet_addDDict(hashSet: *mut __SlateRecord_zstd_decompress_internal_h_109, ddict: *const ZSTD_DDict_s, customMem: __SlateRecord_zstd_lib_h_1837) -> u64 {
while false {
}
if (unsafe { (*hashSet).ddictPtrCount }.wrapping_mul(4) / unsafe { (*hashSet).ddictPtrTableSize }).wrapping_mul(3) != 0 {
loop {
let err_code: u64 = ZSTD_DDictHashSet_expand(hashSet, customMem);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
}
loop {
let err_code: u64 = ZSTD_DDictHashSet_emplaceDDict(hashSet, ddict);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
0
}

/// Context management
#[unsafe(no_mangle)]
extern "C" fn ZSTD_sizeof_DCtx(dctx: *const ZSTD_DCtx_s) -> u64 {
if dctx.is_null() {
return 0;
}
// support sizeof NULL
(core::mem::size_of::<ZSTD_DCtx_s>() as u64).wrapping_add(unsafe { ZSTD_sizeof_DDict((*dctx).ddictLocal as *const ZSTD_DDict_s) }).wrapping_add(unsafe { (*dctx).inBuffSize }).wrapping_add(unsafe { (*dctx).outBuffSize })
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_estimateDCtxSize() -> u64 {
core::mem::size_of::<ZSTD_DCtx_s>() as u64
}

fn ZSTD_startingInputLength(format: u32) -> u64 {
let startingInputLength: u64 = ZSTD_FRAMEHEADERSIZE_PREFIX!(format) as u64;
// only supports formats ZSTD_f_zstd1 and ZSTD_f_zstd1_magicless
startingInputLength
}

fn ZSTD_DCtx_resetParameters(dctx: *mut ZSTD_DCtx_s) {
unsafe {
(*dctx).format = 0;
}
unsafe {
(*dctx).maxWindowSize = 134217729u64;
}
unsafe {
(*dctx).outBufferMode = 0;
}
unsafe {
(*dctx).forceIgnoreChecksum = 0;
}
unsafe {
(*dctx).refMultipleDDicts = 0;
}
unsafe {
(*dctx).disableHufAsm = 0;
}
unsafe {
(*dctx).maxBlockSizeParam = 0;
}
}

fn ZSTD_initDCtx_internal(dctx: *mut ZSTD_DCtx_s) {
unsafe {
(*dctx).staticSize = 0;
}
unsafe {
(*dctx).ddict = core::ptr::null::<ZSTD_DDict_s>();
}
unsafe {
(*dctx).ddictLocal = core::ptr::null_mut::<ZSTD_DDict_s>();
}
unsafe {
(*dctx).dictEnd = core::ptr::null::<()>();
}
unsafe {
(*dctx).ddictIsCold = 0;
}
unsafe {
(*dctx).dictUses = 0;
}
unsafe {
(*dctx).inBuff = core::ptr::null_mut::<u8>();
}
unsafe {
(*dctx).inBuffSize = 0;
}
unsafe {
(*dctx).outBuffSize = 0;
}
unsafe {
(*dctx).streamStage = 0;
}
unsafe {
(*dctx).noForwardProgress = 0;
}
unsafe {
(*dctx).oversizedDuration = 0;
}
unsafe {
(*dctx).isFrameDecompression = 1;
}
unsafe {
(*dctx).bmi2 = ZSTD_cpuSupportsBmi2();
}
unsafe {
(*dctx).ddictSet = core::ptr::null_mut::<__SlateRecord_zstd_decompress_internal_h_109>();
}
ZSTD_DCtx_resetParameters(dctx);
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_initStaticDCtx(workspace: *mut (), workspaceSize: u64) -> *mut ZSTD_DCtx_s {
let dctx: *mut ZSTD_DCtx_s = workspace as *mut ZSTD_DCtx_s;
if (workspace as u64) & 7 != 0 {
return core::ptr::null_mut::<ZSTD_DCtx_s>();
}
// 8-aligned
if workspaceSize < (core::mem::size_of::<ZSTD_DCtx_s>() as u64) {
return core::ptr::null_mut::<ZSTD_DCtx_s>();
}
// minimum size
ZSTD_initDCtx_internal(dctx);
unsafe {
(*dctx).staticSize = workspaceSize;
}
unsafe {
(*dctx).inBuff = dctx.wrapping_offset(1 as isize) as *mut u8;
}
dctx
}

fn ZSTD_createDCtx_internal(customMem: __SlateRecord_zstd_lib_h_1837) -> *mut ZSTD_DCtx_s {
if ((customMem.customAlloc == None) as i32) ^ ((customMem.customFree == None) as i32) != 0 {
return core::ptr::null_mut::<ZSTD_DCtx_s>();
}
let dctx: *mut ZSTD_DCtx_s = ZSTD_customMalloc(core::mem::size_of::<ZSTD_DCtx_s>() as u64, customMem) as *mut ZSTD_DCtx_s;
if dctx.is_null() {
return core::ptr::null_mut::<ZSTD_DCtx_s>();
}
unsafe {
(*dctx).customMem = customMem;
}
ZSTD_initDCtx_internal(dctx);
dctx
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_createDCtx_advanced(customMem: __SlateRecord_zstd_lib_h_1837) -> *mut ZSTD_DCtx_s {
ZSTD_createDCtx_internal(customMem)
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_createDCtx() -> *mut ZSTD_DCtx_s {
while false {
}
ZSTD_createDCtx_internal(unsafe { ZSTD_defaultCMem.0 })
}

fn ZSTD_clearDict(dctx: *mut ZSTD_DCtx_s) {
unsafe { ZSTD_freeDDict((*dctx).ddictLocal) };
unsafe {
(*dctx).ddictLocal = core::ptr::null_mut::<ZSTD_DDict_s>();
}
unsafe {
(*dctx).ddict = core::ptr::null::<ZSTD_DDict_s>();
}
unsafe {
(*dctx).dictUses = 0;
}
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_freeDCtx(dctx: *mut ZSTD_DCtx_s) -> u64 {
if dctx.is_null() {
return 0;
}
// support free on NULL
loop {
if (unsafe { (*dctx).staticSize }) != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"not compatible with static DCtx".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__7!();
}
if true {
break;
}
}
let cMem: __SlateRecord_zstd_lib_h_1837 = unsafe { (*dctx).customMem };
ZSTD_clearDict(dctx);
ZSTD_customFree((unsafe { (*dctx).inBuff }) as *mut (), cMem);
unsafe {
(*dctx).inBuff = core::ptr::null_mut::<u8>();
}
if !unsafe { (*dctx).ddictSet }.is_null() {
ZSTD_freeDDictHashSet(unsafe { (*dctx).ddictSet }, cMem);
unsafe {
(*dctx).ddictSet = core::ptr::null_mut::<__SlateRecord_zstd_decompress_internal_h_109>();
}
}
ZSTD_customFree(dctx as *mut (), cMem);
0
}

/// no longer useful
#[unsafe(no_mangle)]
extern "C" fn ZSTD_copyDCtx(dstDCtx: *mut ZSTD_DCtx_s, srcDCtx: *const ZSTD_DCtx_s) {
let toCopy: u64 = ((unsafe { (core::ptr::addr_of_mut!((*dstDCtx).inBuff) as *mut u8).offset_from(dstDCtx as *mut u8) }) as i64) as u64;
ZSTD_memcpy!(dstDCtx, srcDCtx, toCopy); // no need to copy workspace
}

/// Given a dctx with a digested frame params, re-selects the correct ZSTD_DDict based on
/// the requested dict ID from the frame. If there exists a reference to the correct ZSTD_DDict, then
/// accordingly sets the ddict to be used to decompress the frame.
///
/// If no DDict is found, then no action is taken, and the ZSTD_DCtx::ddict remains as-is.
///
/// ZSTD_d_refMultipleDDicts must be enabled for this function to be called.
fn ZSTD_DCtx_selectFrameDDict(dctx: *mut ZSTD_DCtx_s) {
while false {
}
if !unsafe { (*dctx).ddict }.is_null() {
let frameDDict: *const ZSTD_DDict_s = ZSTD_DDictHashSet_getDDict(unsafe { (*dctx).ddictSet }, unsafe { (*dctx).fParams.dictID });
if !frameDDict.is_null() {
while false {
}
ZSTD_clearDict(dctx);
unsafe {
(*dctx).dictID = (*dctx).fParams.dictID;
}
unsafe {
(*dctx).ddict = frameDDict;
}
unsafe {
(*dctx).dictUses = -1;
}
}
}
}

// Frame header decoding
/// ZSTD_isFrame() :
/// Tells if the content of `buffer` starts with a valid Frame Identifier.
/// Note : Frame Identifier is 4 bytes. If `size < 4`, @return will always be 0.
/// Note 2 : Legacy Frame Identifiers are considered valid only if Legacy Support is enabled.
/// Note 3 : Skippable Frame Identifiers are considered valid.
#[unsafe(no_mangle)]
extern "C" fn ZSTD_isFrame(buffer: *const (), size: u64) -> u32 {
if size < 4 {
return 0;
}
let magic: u32 = MEM_readLE32(buffer);
if magic == 4247762216 {
return 1;
}
if magic & 4294967280 == 407710288 {
return 1;
}
0
}

/// ZSTD_isSkippableFrame() :
/// Tells if the content of `buffer` starts with a valid Frame Identifier for a skippable frame.
/// Note : Frame Identifier is 4 bytes. If `size < 4`, @return will always be 0.
#[unsafe(no_mangle)]
extern "C" fn ZSTD_isSkippableFrame(buffer: *const (), size: u64) -> u32 {
if size < 4 {
return 0;
}
let magic: u32 = MEM_readLE32(buffer);
if magic & 4294967280 == 407710288 {
return 1;
}
0
}

/// ZSTD_frameHeaderSize_internal() :
///  srcSize must be large enough to reach header size fields.
///  note : only works for formats ZSTD_f_zstd1 and ZSTD_f_zstd1_magicless.
/// @return : size of the Frame Header
///           or an error code, which can be tested with ZSTD_isError()
fn ZSTD_frameHeaderSize_internal(src: *const (), srcSize: u64, format: u32) -> u64 {
let minInputSize: u64 = ZSTD_startingInputLength(format);
loop {
if srcSize < minInputSize {
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
return ERROR__2!();
}
if true {
break;
}
}
let fhd: u8 = unsafe { *(src as *const u8).wrapping_offset(minInputSize.wrapping_sub(1) as isize) };
let dictID: u32 = ((fhd as i32) & 3) as u32;
let singleSegment: u32 = ((fhd as i32) >> 5i32 & 1) as u32;
let fcsId: u32 = ((fhd as i32) >> 6i32) as u32;
minInputSize.wrapping_add(((singleSegment == 0) as i32) as u64).wrapping_add(unsafe { ZSTD_did_fieldSize.0[dictID as usize] }).wrapping_add(unsafe { ZSTD_fcs_fieldSize.0[fcsId as usize] }).wrapping_add(((singleSegment != 0 && fcsId == 0) as i32) as u64)
}

/// ZSTD_frameHeaderSize() :
///  srcSize must be >= ZSTD_frameHeaderSize_prefix.
/// @return : size of the Frame Header,
///           or an error code (if srcSize is too small)
#[unsafe(no_mangle)]
extern "C" fn ZSTD_frameHeaderSize(src: *const (), srcSize: u64) -> u64 {
ZSTD_frameHeaderSize_internal(src, srcSize, 0u32)
}

/// ZSTD_getFrameHeader_advanced() :
///  decode Frame Header, or require larger `srcSize`.
///  note : only works for formats ZSTD_f_zstd1 and ZSTD_f_zstd1_magicless
/// @return : 0, `zfhPtr` is correctly filled,
///          >0, `srcSize` is too small, value is wanted `srcSize` amount,
///           or an error code, which can be tested using ZSTD_isError()
#[unsafe(no_mangle)]
extern "C" fn ZSTD_getFrameHeader_advanced(zfhPtr: *mut __SlateRecord_zstd_lib_h_1489, src: *const (), srcSize: u64, format: u32) -> u64 {
let ip: *const u8 = src as *const u8;
let minInputSize: u64 = ZSTD_startingInputLength(format);
while false {
}
if srcSize > 0 {
// note : technically could be considered an assert(), since it's an invalid entry
loop {
if src.is_null() {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"invalid parameter : src==NULL, but srcSize>0".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__5!();
}
if true {
break;
}
}
}
if srcSize < minInputSize {
if srcSize > 0 && (format as u32) != 1 {
// when receiving less than @minInputSize bytes,
// control these bytes at least correspond to a supported magic number
// in order to error out early if they don't.
let toCopy: u64 = MIN__5!(4i32, srcSize);
let mut hbuf: [u8; 4] = [0u8; 4];
MEM_writeLE32(hbuf.as_mut_ptr() as *mut (), 4247762216u32);
ZSTD_memcpy__2!(hbuf.as_mut_ptr() as *mut u8, src, toCopy);
if MEM_readLE32(hbuf.as_mut_ptr() as *const ()) != 4247762216 {
// not a zstd frame : let's check if it's a skippable frame
MEM_writeLE32(hbuf.as_mut_ptr() as *mut (), 407710288u32);
ZSTD_memcpy__2!(hbuf.as_mut_ptr() as *mut u8, src, toCopy);
if MEM_readLE32(hbuf.as_mut_ptr() as *const ()) & 4294967280 != 407710288 {
loop {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"first bytes don't correspond to any supported magic number".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__18!();
}
}
}
}
return minInputSize;
}
ZSTD_memset!(zfhPtr, 0i32, core::mem::size_of::<__SlateRecord_zstd_lib_h_1489>() as u64); // not strictly necessary, but static analyzers may not understand that zfhPtr will be read only if return value is zero, since they are 2 different signals
let __v11: bool = (format as u32) != 1 && MEM_readLE32(src) != 4247762216;
if __v11 {
if MEM_readLE32(src) & 4294967280 == 407710288 {
// skippable frame
if srcSize < 8 {
return 8;
}
// magic number + frame length
ZSTD_memset!(zfhPtr, 0i32, core::mem::size_of::<__SlateRecord_zstd_lib_h_1489>() as u64);
unsafe {
(*zfhPtr).frameType = 1;
}
unsafe {
(*zfhPtr).dictID = MEM_readLE32(src).wrapping_sub(407710288);
}
unsafe {
(*zfhPtr).headerSize = 8;
}
unsafe {
(*zfhPtr).frameContentSize = MEM_readLE32((src as *const u8).wrapping_offset(4 as isize) as *const ()) as u64;
}
return 0;
}
loop {
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
return ERROR__18!();
}
}
// ensure there is enough `srcSize` to fully read/decode frame header
let fhsize: u64 = ZSTD_frameHeaderSize_internal(src, srcSize, format);
if srcSize < fhsize {
return fhsize;
}
unsafe {
(*zfhPtr).headerSize = fhsize as u32;
}
let fhdByte: u8 = unsafe { *ip.wrapping_offset(minInputSize.wrapping_sub(1) as isize) };
let mut pos: u64 = minInputSize;
let dictIDSizeCode: u32 = ((fhdByte as i32) & 3) as u32;
let checksumFlag: u32 = ((fhdByte as i32) >> 2i32 & 1) as u32;
let singleSegment: u32 = ((fhdByte as i32) >> 5i32 & 1) as u32;
let fcsID: u32 = ((fhdByte as i32) >> 6i32) as u32;
let mut windowSize: u64 = 0;
let mut dictID: u32 = 0;
let mut frameContentSize: u64 = 18446744073709551615u64;
loop {
if (fhdByte as i32) & 8 != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"reserved bits, must be zero".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__14!();
}
if true {
break;
}
}
if singleSegment == 0 {
let mut wlByte: u8 = 0;
let __v22: u64 = pos;
pos = __v22.wrapping_add(1);
wlByte = unsafe { *ip.wrapping_offset(__v22 as isize) };
let windowLog: u32 = ((wlByte as i32) >> 3i32).wrapping_add(10) as u32;
loop {
if windowLog > ((if (core::mem::size_of::<u64>() as u64) == 4 { 30i32 } else { 31i32 }) as u32) {
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
return ERROR__11!();
}
if true {
break;
}
}
windowSize = 1u64 << windowLog;
windowSize = windowSize.wrapping_add((windowSize >> 3i32).wrapping_mul(((wlByte as i32) & 7) as u64));
}
// impossible
match dictIDSizeCode {
0 => {
}
1 => {
dictID = (unsafe { *ip.wrapping_offset(pos as isize) }) as u32;
pos = pos.wrapping_add(1);
}
2 => {
dictID = MEM_readLE16(ip.wrapping_offset(pos as isize) as *const ()) as u32;
pos = pos.wrapping_add(2);
}
3 => {
dictID = MEM_readLE32(ip.wrapping_offset(pos as isize) as *const ());
pos = pos.wrapping_add(4);
}
_ => {
// impossible
}
}
// impossible
match fcsID {
0 => {
if singleSegment != 0 {
frameContentSize = (unsafe { *ip.wrapping_offset(pos as isize) }) as u64;
}
}
1 => {
frameContentSize = (MEM_readLE16(ip.wrapping_offset(pos as isize) as *const ()) as i32).wrapping_add(256) as u64;
}
2 => {
frameContentSize = MEM_readLE32(ip.wrapping_offset(pos as isize) as *const ()) as u64;
}
3 => {
frameContentSize = MEM_readLE64(ip.wrapping_offset(pos as isize) as *const ());
}
_ => {
// impossible
if singleSegment != 0 {
frameContentSize = (unsafe { *ip.wrapping_offset(pos as isize) }) as u64;
}
}
}
if singleSegment != 0 {
windowSize = frameContentSize;
}
unsafe {
(*zfhPtr).frameType = 0;
}
unsafe {
(*zfhPtr).frameContentSize = frameContentSize;
}
unsafe {
(*zfhPtr).windowSize = windowSize;
}
unsafe {
(*zfhPtr).blockSizeMax = MIN__2!(windowSize, 131072i32) as u32;
}
unsafe {
(*zfhPtr).dictID = dictID;
}
unsafe {
(*zfhPtr).checksumFlag = checksumFlag;
}
0
}

/// ZSTD_getFrameHeader() :
///  decode Frame Header, or require larger `srcSize`.
///  note : this function does not consume input, it only reads it.
/// @return : 0, `zfhPtr` is correctly filled,
///          >0, `srcSize` is too small, value is wanted `srcSize` amount,
///           or an error code, which can be tested using ZSTD_isError()
#[unsafe(no_mangle)]
extern "C" fn ZSTD_getFrameHeader(zfhPtr: *mut __SlateRecord_zstd_lib_h_1489, src: *const (), srcSize: u64) -> u64 {
ZSTD_getFrameHeader_advanced(zfhPtr, src, srcSize, 0u32)
}

/// ZSTD_getFrameContentSize() :
///  compatible with legacy mode
/// @return : decompressed size of the single frame pointed to be `src` if known, otherwise
///         - ZSTD_CONTENTSIZE_UNKNOWN if the size cannot be determined
///         - ZSTD_CONTENTSIZE_ERROR if an error occurred (e.g. invalid magic number, srcSize too small)
#[unsafe(no_mangle)]
extern "C" fn ZSTD_getFrameContentSize(src: *const (), srcSize: u64) -> u64 {
let mut zfh: __SlateRecord_zstd_lib_h_1489 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(zfh) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
if ZSTD_getFrameHeader(core::ptr::addr_of_mut!(zfh), src, srcSize) != 0 {
return 18446744073709551614u64;
}
if (zfh.frameType as u32) == 1 {
return 0;
} else {
return zfh.frameContentSize;
}
}

fn readSkippableFrameSize(src: *const (), srcSize: u64) -> u64 {
let skippableHeaderSize: u64 = 8;
let mut sizeU32: u32 = 0;
loop {
if srcSize < 8 {
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
return ERROR__2!();
}
if true {
break;
}
}
sizeU32 = MEM_readLE32((src as *const u8).wrapping_offset(4 as isize) as *const ());
loop {
if sizeU32.wrapping_add(8) < sizeU32 {
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
return ERROR__14!();
}
if true {
break;
}
}
let skippableSize: u64 = skippableHeaderSize.wrapping_add(sizeU32 as u64);
loop {
if skippableSize > srcSize {
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
return ERROR__2!();
}
if true {
break;
}
}
skippableSize
}

/// ZSTD_readSkippableFrame() :
/// Retrieves content of a skippable frame, and writes it to dst buffer.
///
/// The parameter magicVariant will receive the magicVariant that was supplied when the frame was written,
/// i.e. magicNumber - ZSTD_MAGIC_SKIPPABLE_START.  This can be NULL if the caller is not interested
/// in the magicVariant.
///
/// Returns an error if destination buffer is not large enough, or if this is not a valid skippable frame.
///
/// @return : number of bytes written or a ZSTD error.
///
/// # Arguments
///
/// * `magicVariant` - optional, can be NULL
#[unsafe(no_mangle)]
extern "C" fn ZSTD_readSkippableFrame(dst: *mut (), dstCapacity: u64, magicVariant: *mut u32, src: *const (), srcSize: u64) -> u64 {
loop {
if srcSize < 8 {
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
return ERROR__2!();
}
if true {
break;
}
}
let magicNumber: u32 = MEM_readLE32(src);
let skippableFrameSize: u64 = readSkippableFrameSize(src, srcSize);
let skippableContentSize: u64 = skippableFrameSize.wrapping_sub(8);
// check input validity
loop {
if ZSTD_isSkippableFrame(src, srcSize) == 0 {
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
return ERROR__14!();
}
if true {
break;
}
}
loop {
if skippableFrameSize < 8 || skippableFrameSize > srcSize {
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
return ERROR__2!();
}
if true {
break;
}
}
loop {
if skippableContentSize > dstCapacity {
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
return ERROR__3!();
}
if true {
break;
}
}
// deliver payload
if skippableContentSize > 0 && !dst.is_null() {
ZSTD_memcpy__3!(dst, (src as *const u8).wrapping_offset(8 as isize), skippableContentSize);
}
if !magicVariant.is_null() {
unsafe {
*magicVariant = magicNumber.wrapping_sub(407710288);
}
}
skippableContentSize
}

/// ZSTD_findDecompressedSize() :
///  `srcSize` must be the exact length of some number of ZSTD compressed and/or
///      skippable frames
///  note: compatible with legacy mode
/// @return : decompressed size of the frames contained
#[unsafe(no_mangle)]
extern "C" fn ZSTD_findDecompressedSize(mut src: *const (), mut srcSize: u64) -> u64 {
let mut totalDstSize: u64 = 0;
while srcSize >= ZSTD_startingInputLength(0u32) {
let magicNumber: u32 = MEM_readLE32(src);
if magicNumber & 4294967280 == 407710288 {
let skippableSize: u64 = readSkippableFrameSize(src, srcSize);
if ERR_isError(skippableSize) != 0 {
return 18446744073709551614u64;
}
src = (src as *const u8).wrapping_offset(skippableSize as isize) as *const ();
srcSize = srcSize.wrapping_sub(skippableSize);
} else {
let fcs: u64 = ZSTD_getFrameContentSize(src, srcSize);
if fcs >= 18446744073709551614u64 {
return fcs;
}
if totalDstSize.wrapping_add(fcs) < totalDstSize {
return 18446744073709551614u64;
}
// check for overflow
totalDstSize = totalDstSize.wrapping_add(fcs);
// skip to next frame
let frameSrcSize: u64 = ZSTD_findFrameCompressedSize(src, srcSize);
if ERR_isError(frameSrcSize) != 0 {
return 18446744073709551614u64;
}
src = (src as *const u8).wrapping_offset(frameSrcSize as isize) as *const ();
srcSize = srcSize.wrapping_sub(frameSrcSize);
}
}
// while (srcSize >= ZSTD_frameHeaderSize_prefix)
if srcSize != 0 {
return 18446744073709551614u64;
}
totalDstSize
}

/// ZSTD_getDecompressedSize() :
/// *  compatible with legacy mode
/// * @return : decompressed size if known, 0 otherwise
///             note : 0 can mean any of the following :
///                   - frame content is empty
///                   - decompressed size field is not present in frame header
///                   - frame header unknown / not supported
///                   - frame header not complete (`srcSize` too small)
#[unsafe(no_mangle)]
extern "C" fn ZSTD_getDecompressedSize(src: *const (), srcSize: u64) -> u64 {
let ret: u64 = ZSTD_getFrameContentSize(src, srcSize);
ZSTD_STATIC_ASSERT!();
if ret >= 18446744073709551614u64 { 0 } else { ret }
}

/// ZSTD_decodeFrameHeader() :
/// `headerSize` must be the size provided by ZSTD_frameHeaderSize().
/// If multiple DDict references are enabled, also will choose the correct DDict to use.
/// @return : 0 if success, or an error code, which can be tested using ZSTD_isError()
fn ZSTD_decodeFrameHeader(dctx: *mut ZSTD_DCtx_s, src: *const (), headerSize: u64) -> u64 {
let result: u64 = ZSTD_getFrameHeader_advanced(unsafe { core::ptr::addr_of_mut!((*dctx).fParams) }, src, headerSize, unsafe { (*dctx).format });
if ERR_isError(result) != 0 {
return result;
}
// invalid header
loop {
if result > 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"headerSize too small".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__2!();
}
if true {
break;
}
}
// Reference DDict requested by frame if dctx references multiple ddicts
if ((unsafe { (*dctx).refMultipleDDicts }) as u32) == 1 && !unsafe { (*dctx).ddictSet }.is_null() {
ZSTD_DCtx_selectFrameDDict(dctx);
}
// Skip the dictID check in fuzzing mode, because it makes the search
// harder.
loop {
if (unsafe { (*dctx).fParams.dictID }) != 0 && (unsafe { (*dctx).dictID }) != unsafe { (*dctx).fParams.dictID } {
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
return ERROR__20!();
}
if true {
break;
}
}
unsafe {
(*dctx).validateChecksum = (if (*dctx).fParams.checksumFlag != 0 && ((*dctx).forceIgnoreChecksum as u32) == 0 { 1i32 } else { 0i32 }) as u32;
}
if (unsafe { (*dctx).validateChecksum }) != 0 {
unsafe { xxh64_reset(core::ptr::addr_of_mut!((*dctx).xxhState), 0u64) };
}
let __v11: u64 = unsafe { (*dctx).processedCSize };
unsafe {
(*dctx).processedCSize = __v11.wrapping_add(headerSize);
}
0
}

fn ZSTD_errorFrameSizeInfo(ret: u64) -> __SlateRecord_zstd_internal_h_282 {
let mut frameSizeInfo: __SlateRecord_zstd_internal_h_282 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(frameSizeInfo) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
frameSizeInfo.compressedSize = ret;
frameSizeInfo.decompressedBound = 18446744073709551614u64;
return frameSizeInfo;
}

fn ZSTD_findFrameSizeInfo(src: *const (), srcSize: u64, format: u32) -> __SlateRecord_zstd_internal_h_282 {
let mut frameSizeInfo: __SlateRecord_zstd_internal_h_282 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(frameSizeInfo) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
ZSTD_memset!(core::ptr::addr_of_mut!(frameSizeInfo), 0i32, core::mem::size_of::<__SlateRecord_zstd_internal_h_282>() as u64);
let __v0: bool = (format as u32) == 0 && srcSize >= 8 && MEM_readLE32(src) & 4294967280 == 407710288;
if __v0 {
frameSizeInfo.compressedSize = readSkippableFrameSize(src, srcSize);
return frameSizeInfo;
} else {
let mut ip: *const u8 = src as *const u8;
let ipstart: *const u8 = ip;
let mut remainingSize: u64 = srcSize;
let mut nbBlocks: u64 = 0;
let mut zfh: __SlateRecord_zstd_lib_h_1489 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(zfh) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
// Extract Frame Header
let ret: u64 = ZSTD_getFrameHeader_advanced(core::ptr::addr_of_mut!(zfh), src, srcSize, format);
if ERR_isError(ret) != 0 {
return ZSTD_errorFrameSizeInfo(ret);
}
if ret > 0 {
return ZSTD_errorFrameSizeInfo(ERROR__2!());
}
let __v1: *const u8 = ip;
ip = __v1.wrapping_offset(zfh.headerSize as isize);
remainingSize = remainingSize.wrapping_sub(zfh.headerSize as u64);
// Iterate over each block
while true {
let mut blockProperties: __SlateRecord_zstd_internal_h_295 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(blockProperties) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 12usize);
}
let cBlockSize: u64 = unsafe { ZSTD_getcBlockSize(ip as *const (), remainingSize, core::ptr::addr_of_mut!(blockProperties)) };
if ERR_isError(cBlockSize) != 0 {
return ZSTD_errorFrameSizeInfo(cBlockSize);
}
if unsafe { ZSTD_blockHeaderSize }.wrapping_add(cBlockSize) > remainingSize {
return ZSTD_errorFrameSizeInfo(ERROR__2!());
}
let __v5: *const u8 = ip;
ip = __v5.wrapping_offset(unsafe { ZSTD_blockHeaderSize }.wrapping_add(cBlockSize) as isize);
remainingSize = remainingSize.wrapping_sub(unsafe { ZSTD_blockHeaderSize }.wrapping_add(cBlockSize));
nbBlocks = nbBlocks.wrapping_add(1);
if blockProperties.lastBlock != 0 {
break;
}
}
// Final frame content checksum
if zfh.checksumFlag != 0 {
if remainingSize < 4 {
return ZSTD_errorFrameSizeInfo(ERROR__2!());
}
let __v12: *const u8 = ip;
ip = __v12.wrapping_offset(4 as isize);
}
frameSizeInfo.nbBlocks = nbBlocks;
frameSizeInfo.compressedSize = ((unsafe { ip.offset_from(ipstart as *const u8) }) as i64) as u64;
frameSizeInfo.decompressedBound = if zfh.frameContentSize != 18446744073709551615u64 { zfh.frameContentSize } else { nbBlocks.wrapping_mul(zfh.blockSizeMax as u64) };
return frameSizeInfo;
}
}

fn ZSTD_findFrameCompressedSize_advanced(src: *const (), srcSize: u64, format: u32) -> u64 {
let frameSizeInfo: __SlateRecord_zstd_internal_h_282 = ZSTD_findFrameSizeInfo(src, srcSize, format);
frameSizeInfo.compressedSize
}

/// ZSTD_findFrameCompressedSize() :
/// See docs in zstd.h
/// Note: compatible with legacy mode
#[unsafe(no_mangle)]
extern "C" fn ZSTD_findFrameCompressedSize(src: *const (), srcSize: u64) -> u64 {
ZSTD_findFrameCompressedSize_advanced(src, srcSize, 0u32)
}

/// ZSTD_decompressBound() :
/// compatible with legacy mode
/// `src` must point to the start of a ZSTD frame or a skippable frame
/// `srcSize` must be at least as large as the frame contained
/// @return : the maximum decompressed size of the compressed source
#[unsafe(no_mangle)]
extern "C" fn ZSTD_decompressBound(mut src: *const (), mut srcSize: u64) -> u64 {
let mut bound: u64 = 0;
// Iterate over each frame
while srcSize > 0 {
let frameSizeInfo: __SlateRecord_zstd_internal_h_282 = ZSTD_findFrameSizeInfo(src, srcSize, 0u32);
let compressedSize: u64 = frameSizeInfo.compressedSize;
let decompressedBound: u64 = frameSizeInfo.decompressedBound;
if ERR_isError(compressedSize) != 0 || decompressedBound == 18446744073709551614u64 {
return 18446744073709551614u64;
}
src = (src as *const u8).wrapping_offset(compressedSize as isize) as *const ();
srcSize = srcSize.wrapping_sub(compressedSize);
bound = bound.wrapping_add(decompressedBound);
}
bound
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_decompressionMargin(mut src: *const (), mut srcSize: u64) -> u64 {
let mut margin: u64 = 0;
let mut maxBlockSize: u32 = 0;
// Iterate over each frame
while srcSize > 0 {
let frameSizeInfo: __SlateRecord_zstd_internal_h_282 = ZSTD_findFrameSizeInfo(src, srcSize, 0u32);
let compressedSize: u64 = frameSizeInfo.compressedSize;
let decompressedBound: u64 = frameSizeInfo.decompressedBound;
let mut zfh: __SlateRecord_zstd_lib_h_1489 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(zfh) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
loop {
let err_code: u64 = ZSTD_getFrameHeader(core::ptr::addr_of_mut!(zfh), src, srcSize);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
if ERR_isError(compressedSize) != 0 || decompressedBound == 18446744073709551614u64 {
return ERROR!();
}
if (zfh.frameType as u32) == 0 {
// Add the frame header to our margin
margin = margin.wrapping_add(zfh.headerSize as u64);
// Add the checksum to our margin
margin = margin.wrapping_add((if zfh.checksumFlag != 0 { 4i32 } else { 0i32 }) as u64);
// Add 3 bytes per block
margin = margin.wrapping_add(3u64.wrapping_mul(frameSizeInfo.nbBlocks));
// Compute the max block size
maxBlockSize = MAX!(maxBlockSize, zfh.blockSizeMax);
} else {
// Add the entire skippable frame size to our margin.
margin = margin.wrapping_add(compressedSize);
}
src = (src as *const u8).wrapping_offset(compressedSize as isize) as *const ();
srcSize = srcSize.wrapping_sub(compressedSize);
}
// Add the max block size back to the margin.
margin = margin.wrapping_add(maxBlockSize as u64);
margin
}

// Frame decoding
/// ZSTD_insertBlock() :
/// insert `src` block into `dctx` history. Useful to track uncompressed blocks.
#[unsafe(no_mangle)]
extern "C" fn ZSTD_insertBlock(dctx: *mut ZSTD_DCtx_s, blockStart: *const (), blockSize: u64) -> u64 {
while false {
}
unsafe { ZSTD_checkContinuity(dctx, blockStart, blockSize) };
unsafe {
(*dctx).previousDstEnd = (blockStart as *const u8).wrapping_offset(blockSize as isize) as *const ();
}
blockSize
}

fn ZSTD_copyRawBlock(dst: *mut (), dstCapacity: u64, src: *const (), srcSize: u64) -> u64 {
while false {
}
loop {
if srcSize > dstCapacity {
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
return ERROR__3!();
}
if true {
break;
}
}
if dst.is_null() {
if srcSize == 0 {
return 0;
}
loop {
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
return ERROR__16!();
}
}
ZSTD_memmove__2!(dst, src, srcSize);
srcSize
}

fn ZSTD_setRleBlock(dst: *mut (), dstCapacity: u64, b: u8, regenSize: u64) -> u64 {
loop {
if regenSize > dstCapacity {
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
return ERROR__3!();
}
if true {
break;
}
}
if dst.is_null() {
if regenSize == 0 {
return 0;
}
loop {
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
return ERROR__16!();
}
}
ZSTD_memset__2!(dst, b, regenSize);
regenSize
}

fn ZSTD_DCtx_trace_end(dctx: *const ZSTD_DCtx_s, uncompressedSize: u64, compressedSize: u64, streaming: i32) {
}

/// ZSTD_decompressFrame() :
/// @dctx must be properly initialized
///  will update *srcPtr and *srcSizePtr,
///  to make *srcPtr progress by one frame.
fn ZSTD_decompressFrame(dctx: *mut ZSTD_DCtx_s, dst: *mut (), dstCapacity: u64, srcPtr: &mut *const (), srcSizePtr: &mut u64) -> u64 {
let istart: *const u8 = (unsafe { *(srcPtr as *mut *const ()) }) as *const u8;
let mut ip: *const u8 = istart;
let ostart: *mut u8 = dst as *mut u8;
let oend: *mut u8 = if dstCapacity != 0 { ostart.wrapping_offset(dstCapacity as isize) } else { ostart };
let mut op: *mut u8 = ostart;
let mut remainingSrcSize: u64 = unsafe { *(srcSizePtr as *mut u64) };
while false {
}
// check
loop {
if remainingSrcSize < (ZSTD_FRAMEHEADERSIZE_MIN!(unsafe { (*dctx).format }) as u64).wrapping_add(unsafe { ZSTD_blockHeaderSize }) {
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
return ERROR__2!();
}
if true {
break;
}
}
// Frame Header
let frameHeaderSize: u64 = ZSTD_frameHeaderSize_internal(ip as *const (), ZSTD_FRAMEHEADERSIZE_PREFIX!(unsafe { (*dctx).format }) as u64, unsafe { (*dctx).format });
if ERR_isError(frameHeaderSize) != 0 {
return frameHeaderSize;
}
loop {
if remainingSrcSize < frameHeaderSize.wrapping_add(unsafe { ZSTD_blockHeaderSize }) {
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
return ERROR__2!();
}
if true {
break;
}
}
loop {
let err_code: u64 = ZSTD_decodeFrameHeader(dctx, ip as *const (), frameHeaderSize);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
let __v16: *const u8 = ip;
ip = __v16.wrapping_offset(frameHeaderSize as isize);
remainingSrcSize = remainingSrcSize.wrapping_sub(frameHeaderSize);
// Shrink the blockSizeMax if enabled
if (unsafe { (*dctx).maxBlockSizeParam }) != 0 {
unsafe {
(*dctx).fParams.blockSizeMax = MIN!(unsafe { (*dctx).fParams.blockSizeMax }, (*dctx).maxBlockSizeParam as u32);
}
}
// Loop on each block
while true {
let mut oBlockEnd: *mut u8 = oend;
let mut decodedSize: u64 = 0;
let mut blockProperties: __SlateRecord_zstd_internal_h_295 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(blockProperties) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 12usize);
}
let cBlockSize: u64 = unsafe { ZSTD_getcBlockSize(ip as *const (), remainingSrcSize, core::ptr::addr_of_mut!(blockProperties)) };
if ERR_isError(cBlockSize) != 0 {
return cBlockSize;
}
let __v20: *const u8 = ip;
ip = __v20.wrapping_offset((unsafe { ZSTD_blockHeaderSize }) as isize);
remainingSrcSize = remainingSrcSize.wrapping_sub(unsafe { ZSTD_blockHeaderSize });
loop {
if cBlockSize > remainingSrcSize {
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
return ERROR__2!();
}
if true {
break;
}
}
if ip >= (op as *const u8) && ip < (oBlockEnd as *const u8) {
// We are decompressing in-place. Limit the output pointer so that we
// don't overwrite the block that we are currently reading. This will
// fail decompression if the input & output pointers aren't spaced
// far enough apart.
//
// This is important to set, even when the pointers are far enough
// apart, because ZSTD_decompressBlock_internal() can decide to store
// literals in the output buffer, after the block it is decompressing.
// Since we don't want anything to overwrite our input, we have to tell
// ZSTD_decompressBlock_internal to never write past ip.
//
// See ZSTD_allocateLiteralsBuffer() for reference.
oBlockEnd = op.wrapping_offset(((unsafe { ip.offset_from(op as *const u8) }) as i64) as isize);
}
match blockProperties.blockType as u32 {
2 => {
decodedSize = unsafe { ZSTD_decompressBlock_internal(dctx, op as *mut (), (oBlockEnd.offset_from(op as *mut u8) as i64) as u64, ip as *const (), cBlockSize, 0u32) };
}
0 => {
decodedSize = ZSTD_copyRawBlock(op as *mut (), ((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64, ip as *const (), cBlockSize); // Use oend instead of oBlockEnd because this function is safe to overlap. It uses memmove.
}
1 => {
decodedSize = ZSTD_setRleBlock(op as *mut (), ((unsafe { oBlockEnd.offset_from(op as *mut u8) }) as i64) as u64, unsafe { *ip }, blockProperties.origSize as u64);
}
_ => {
loop {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"invalid block type".as_ptr() as *const u8) };
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
}
}
loop {
let err_code: u64 = decodedSize;
if ERR_isError(err_code) != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"Block decompression failure".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return err_code;
}
if true {
break;
}
}
while false {
}
if (unsafe { (*dctx).validateChecksum }) != 0 {
unsafe { xxh64_update(core::ptr::addr_of_mut!((*dctx).xxhState), op as *const (), decodedSize) };
}
if decodedSize != 0 {
// support dst = NULL,0
let __v41: *mut u8 = op;
op = __v41.wrapping_offset(decodedSize as isize);
}
let __v43: *const u8 = ip;
ip = __v43.wrapping_offset(cBlockSize as isize);
remainingSrcSize = remainingSrcSize.wrapping_sub(cBlockSize);
if blockProperties.lastBlock != 0 {
break;
}
}
if (unsafe { (*dctx).fParams.frameContentSize }) != 18446744073709551615u64 {
loop {
if (((unsafe { op.offset_from(ostart as *mut u8) }) as i64) as u64) != unsafe { (*dctx).fParams.frameContentSize } {
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
}
if (unsafe { (*dctx).fParams.checksumFlag }) != 0 {
// Frame content checksum verification
loop {
if remainingSrcSize < 4 {
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
return ERROR__12!();
}
if true {
break;
}
}
if ((unsafe { (*dctx).forceIgnoreChecksum }) as u32) == 0 {
let checkCalc: u32 = (unsafe { xxh64_digest(core::ptr::addr_of_mut!((*dctx).xxhState) as *const xxh64_state) }) as u32;
let checkRead: u32 = MEM_readLE32(ip as *const ());
loop {
if checkRead != checkCalc {
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
return ERROR__12!();
}
if true {
break;
}
}
}
let __v63: *const u8 = ip;
ip = __v63.wrapping_offset(4 as isize);
remainingSrcSize = remainingSrcSize.wrapping_sub(4);
}
ZSTD_DCtx_trace_end(dctx as *const ZSTD_DCtx_s, ((unsafe { op.offset_from(ostart as *mut u8) }) as i64) as u64, ((unsafe { ip.offset_from(istart as *const u8) }) as i64) as u64, 0i32); // streaming
// Allow caller to get size read
while false {
}
unsafe {
*(srcPtr as *mut *const ()) = ip as *const ();
}
unsafe {
*(srcSizePtr as *mut u64) = remainingSrcSize;
}
((unsafe { op.offset_from(ostart as *mut u8) }) as i64) as u64
}

fn ZSTD_decompressMultiFrame(dctx: *mut ZSTD_DCtx_s, mut dst: *mut (), mut dstCapacity: u64, mut src: *const (), mut srcSize: u64, mut dict: *const (), mut dictSize: u64, ddict: *const ZSTD_DDict_s) -> u64 {
let dststart: *mut () = dst;
let mut moreThan1Frame: i32 = 0;
while false {
}
// either dict or ddict set, not both
if !ddict.is_null() {
dict = unsafe { ZSTD_DDict_dictContent(ddict) };
dictSize = unsafe { ZSTD_DDict_dictSize(ddict) };
}
while srcSize >= ZSTD_startingInputLength(unsafe { (*dctx).format }) {
'__slate_continue_9: {
if ((unsafe { (*dctx).format }) as u32) == 0 && srcSize >= 4 {
let magicNumber: u32 = MEM_readLE32(src);
while false {
}
if magicNumber & 4294967280 == 407710288 {
// skippable frame detected : skip it
let skippableSize: u64 = readSkippableFrameSize(src, srcSize);
loop {
let err_code: u64 = skippableSize;
if ERR_isError(err_code) != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"invalid skippable frame".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return err_code;
}
if true {
break;
}
}
src = (src as *const u8).wrapping_offset(skippableSize as isize) as *const ();
srcSize = srcSize.wrapping_sub(skippableSize);
break '__slate_continue_9; // check next frame
}
}
if !ddict.is_null() {
// we were called from ZSTD_decompress_usingDDict
loop {
let err_code: u64 = ZSTD_decompressBegin_usingDDict(dctx, ddict);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
} else {
// this will initialize correctly with no dict if dict == NULL, so
// use this in all cases but ddict
loop {
let err_code: u64 = ZSTD_decompressBegin_usingDict(dctx, dict, dictSize);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
}
unsafe { ZSTD_checkContinuity(dctx, dst as *const (), dstCapacity) };
let res: u64 = ZSTD_decompressFrame(dctx, dst, dstCapacity, unsafe { &mut (*core::ptr::addr_of_mut!(src)) }, unsafe { &mut (*core::ptr::addr_of_mut!(srcSize)) });
loop {
if ((unsafe { ZSTD_getErrorCode(res) }) as u32) == 10 && moreThan1Frame == 1 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"At least one frame successfully completed, but following bytes are garbage: it's more likely to be a srcSize error, specifying more input bytes than size of frame(s). Note: one could be unlucky, it might be a corruption error instead, happening right at the place where we expect zstd magic bytes. But this is _much_ less likely than a srcSize field error.".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__2!();
}
if true {
break;
}
}
if ERR_isError(res) != 0 {
return res;
}
if res != 0 {
dst = (dst as *mut u8).wrapping_offset(res as isize) as *mut ();
}
dstCapacity = dstCapacity.wrapping_sub(res);
moreThan1Frame = 1;
}
}
// while (srcSize >= ZSTD_frameHeaderSize_prefix)
loop {
if srcSize != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"input not entirely consumed".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__2!();
}
if true {
break;
}
}
((unsafe { (dst as *mut u8).offset_from(dststart as *mut u8) }) as i64) as u64
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_decompress_usingDict(dctx: *mut ZSTD_DCtx_s, dst: *mut (), dstCapacity: u64, src: *const (), srcSize: u64, dict: *const (), dictSize: u64) -> u64 {
ZSTD_decompressMultiFrame(dctx, dst, dstCapacity, src, srcSize, dict, dictSize, core::ptr::null::<ZSTD_DDict_s>())
}

fn ZSTD_getDDict(dctx: *mut ZSTD_DCtx_s) -> *const ZSTD_DDict_s {
// Impossible
match (unsafe { (*dctx).dictUses }) as i32 {
0 => {
ZSTD_clearDict(dctx);
return core::ptr::null::<ZSTD_DDict_s>();
}
-1 => {
return unsafe { (*dctx).ddict };
}
1 => {
unsafe {
(*dctx).dictUses = 0;
}
return unsafe { (*dctx).ddict };
}
_ => {
// Impossible
ZSTD_clearDict(dctx);
return core::ptr::null::<ZSTD_DDict_s>();
}
}
unsafe { core::mem::zeroed() }
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_decompressDCtx(dctx: *mut ZSTD_DCtx_s, dst: *mut (), dstCapacity: u64, src: *const (), srcSize: u64) -> u64 {
ZSTD_decompress_usingDDict(dctx, dst, dstCapacity, src, srcSize, ZSTD_getDDict(dctx))
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_decompress(dst: *mut (), dstCapacity: u64, src: *const (), srcSize: u64) -> u64 {
let mut regenSize: u64 = 0;
let dctx: *mut ZSTD_DCtx_s = ZSTD_createDCtx_internal(unsafe { ZSTD_defaultCMem.0 });
loop {
if dctx.is_null() {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"NULL pointer!".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__7!();
}
if true {
break;
}
}
regenSize = ZSTD_decompressDCtx(dctx, dst, dstCapacity, src, srcSize);
ZSTD_freeDCtx(dctx);
regenSize
}

/// Advanced Streaming Decompression API
/// Bufferless and synchronous
#[unsafe(no_mangle)]
extern "C" fn ZSTD_nextSrcSizeToDecompress(dctx: *mut ZSTD_DCtx_s) -> u64 {
unsafe { (*dctx).expected }
}

/// Similar to ZSTD_nextSrcSizeToDecompress(), but when a block input can be streamed, we
/// allow taking a partial block as the input. Currently only raw uncompressed blocks can
/// be streamed.
///
/// For blocks that can be streamed, this allows us to reduce the latency until we produce
/// output, and avoid copying the input.
///
/// @param inputSize - The total amount of input that the caller currently has.
fn ZSTD_nextSrcSizeToDecompressWithInputSize(dctx: *mut ZSTD_DCtx_s, inputSize: u64) -> u64 {
if !(((unsafe { (*dctx).stage }) as u32) == 3 || ((unsafe { (*dctx).stage }) as u32) == 4) {
return unsafe { (*dctx).expected };
}
if ((unsafe { (*dctx).bType }) as u32) != 0 {
return unsafe { (*dctx).expected };
}
BOUNDED__2!(1i32, inputSize, unsafe { (*dctx).expected })
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_nextInputType(dctx: *mut ZSTD_DCtx_s) -> u32 {
// should not happen
match (unsafe { (*dctx).stage }) as u32 {
0 | 1 => {
return 0;
}
2 => {
return 1;
}
3 => {
return 2;
}
4 => {
return 3;
}
5 => {
return 4;
}
6 | 7 => {
return 5;
}
_ => {
// should not happen
return 0;
}
}
unsafe { core::mem::zeroed() }
}

fn ZSTD_isSkipFrame(dctx: *mut ZSTD_DCtx_s) -> i32 {
(((unsafe { (*dctx).stage }) as u32) == 7) as i32
}

/// ZSTD_decompressContinue() :
/// srcSize : must be the exact nb of bytes expected (see ZSTD_nextSrcSizeToDecompress())
/// @return : nb of bytes generated into `dst` (necessarily <= `dstCapacity)
///           or an error code, which can be tested using ZSTD_isError()
#[unsafe(no_mangle)]
extern "C" fn ZSTD_decompressContinue(dctx: *mut ZSTD_DCtx_s, dst: *mut (), dstCapacity: u64, src: *const (), srcSize: u64) -> u64 {
while false {
}
// Sanity check
loop {
if srcSize != ZSTD_nextSrcSizeToDecompressWithInputSize(dctx, srcSize) {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"not allowed".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__2!();
}
if true {
break;
}
}
unsafe { ZSTD_checkContinuity(dctx, dst as *const (), dstCapacity) };
let __v7: u64 = unsafe { (*dctx).processedCSize };
unsafe {
(*dctx).processedCSize = __v7.wrapping_add(srcSize);
}
match (unsafe { (*dctx).stage }) as u32 {
0 => {
if ((unsafe { (*dctx).format }) as u32) == 0 {
// allows header
// to read skippable magic number
if MEM_readLE32(src) & 4294967280 == 407710288 {
// skippable frame
ZSTD_memcpy__2!(unsafe { (*dctx).headerBuffer.as_mut_ptr() as *mut u8 }, src, srcSize);
unsafe {
(*dctx).expected = 8u64.wrapping_sub(srcSize);
}
// remaining to load to get full skippable frame header
unsafe {
(*dctx).stage = 6;
}
return 0;
}
}
unsafe {
(*dctx).headerSize = ZSTD_frameHeaderSize_internal(src, srcSize, (*dctx).format);
}
if ERR_isError(unsafe { (*dctx).headerSize }) != 0 {
return unsafe { (*dctx).headerSize };
}
ZSTD_memcpy__2!(unsafe { (*dctx).headerBuffer.as_mut_ptr() as *mut u8 }, src, srcSize);
unsafe {
(*dctx).expected = unsafe { (*dctx).headerSize }.wrapping_sub(srcSize);
}
unsafe {
(*dctx).stage = 1;
}
return 0;
}
1 => {
ZSTD_memcpy__2!(unsafe { (*dctx).headerBuffer.as_mut_ptr() as *mut u8 }.wrapping_offset(unsafe { (*dctx).headerSize }.wrapping_sub(srcSize) as isize), src, srcSize);
loop {
let err_code: u64 = ZSTD_decodeFrameHeader(dctx, (unsafe { (*dctx).headerBuffer.as_mut_ptr() as *mut u8 }) as *const (), unsafe { (*dctx).headerSize });
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
unsafe {
(*dctx).expected = ZSTD_blockHeaderSize;
}
unsafe {
(*dctx).stage = 2;
}
return 0;
}
2 => {
let mut bp: __SlateRecord_zstd_internal_h_295 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bp) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 12usize);
}
let cBlockSize: u64 = unsafe { ZSTD_getcBlockSize(src, ZSTD_blockHeaderSize, core::ptr::addr_of_mut!(bp)) };
if ERR_isError(cBlockSize) != 0 {
return cBlockSize;
}
loop {
if cBlockSize > ((unsafe { (*dctx).fParams.blockSizeMax }) as u64) {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"Block Size Exceeds Maximum".as_ptr() as *const u8) };
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
unsafe {
(*dctx).expected = cBlockSize;
}
unsafe {
(*dctx).bType = bp.blockType;
}
unsafe {
(*dctx).rleSize = bp.origSize as u64;
}
if cBlockSize != 0 {
unsafe {
(*dctx).stage = (if bp.lastBlock != 0 { 4i32 } else { 3i32 }) as u32;
}
return 0;
}
// empty block
if bp.lastBlock != 0 {
if (unsafe { (*dctx).fParams.checksumFlag }) != 0 {
unsafe {
(*dctx).expected = 4;
}
unsafe {
(*dctx).stage = 5;
}
} else {
unsafe {
(*dctx).expected = 0;
}
// end of frame
unsafe {
(*dctx).stage = 0;
}
}
} else {
unsafe {
(*dctx).expected = ZSTD_blockHeaderSize;
}
// jump to next header
unsafe {
(*dctx).stage = 2;
}
}
return 0;
}
3 | 4 => {
while false {
}
let mut rSize: u64 = 0;
match (unsafe { (*dctx).bType }) as u32 {
2 => {
while false {
}
rSize = unsafe { ZSTD_decompressBlock_internal(dctx, dst, dstCapacity, src, srcSize, 1u32) };
unsafe {
(*dctx).expected = 0;
}
// Streaming not supported
}
0 => {
rSize = ZSTD_copyRawBlock(dst, dstCapacity, src, srcSize);
loop {
let err_code: u64 = rSize;
if ERR_isError(err_code) != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"ZSTD_copyRawBlock failed".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return err_code;
}
if true {
break;
}
}
let __v28: u64 = unsafe { (*dctx).expected };
unsafe {
(*dctx).expected = __v28.wrapping_sub(rSize);
}
}
1 => {
rSize = ZSTD_setRleBlock(dst, dstCapacity, unsafe { *(src as *const u8) }, unsafe { (*dctx).rleSize });
unsafe {
(*dctx).expected = 0;
}
// Streaming not supported
}
_ => {
loop {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"invalid block type".as_ptr() as *const u8) };
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
// should never happen
}
}
loop {
let err_code: u64 = rSize;
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
loop {
if rSize > ((unsafe { (*dctx).fParams.blockSizeMax }) as u64) {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"Decompressed Block Size Exceeds Maximum".as_ptr() as *const u8) };
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
while false {
}
let __v47: u64 = unsafe { (*dctx).decodedSize };
unsafe {
(*dctx).decodedSize = __v47.wrapping_add(rSize);
}
if (unsafe { (*dctx).validateChecksum }) != 0 {
unsafe { xxh64_update(core::ptr::addr_of_mut!((*dctx).xxhState), dst as *const (), rSize) };
}
unsafe {
(*dctx).previousDstEnd = (dst as *mut u8).wrapping_offset(rSize as isize) as *const ();
}
// Stay on the same stage until we are finished streaming the block.
if (unsafe { (*dctx).expected }) > 0 {
return rSize;
}
if ((unsafe { (*dctx).stage }) as u32) == 4 {
// end of frame
while false {
}
loop {
if (unsafe { (*dctx).fParams.frameContentSize }) != 18446744073709551615u64 && (unsafe { (*dctx).decodedSize }) != unsafe { (*dctx).fParams.frameContentSize } {
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
if (unsafe { (*dctx).fParams.checksumFlag }) != 0 {
// another round for frame checksum
unsafe {
(*dctx).expected = 4;
}
unsafe {
(*dctx).stage = 5;
}
} else {
ZSTD_DCtx_trace_end(dctx as *const ZSTD_DCtx_s, unsafe { (*dctx).decodedSize }, unsafe { (*dctx).processedCSize }, 1i32); // streaming
unsafe {
(*dctx).expected = 0;
}
// ends here
unsafe {
(*dctx).stage = 0;
}
}
} else {
unsafe {
(*dctx).stage = 2;
}
unsafe {
(*dctx).expected = ZSTD_blockHeaderSize;
}
}
return rSize;
}
5 => {
// guaranteed by dctx->expected
if (unsafe { (*dctx).validateChecksum }) != 0 {
let h32: u32 = (unsafe { xxh64_digest(core::ptr::addr_of_mut!((*dctx).xxhState) as *const xxh64_state) }) as u32;
let check32: u32 = MEM_readLE32(src);
while false {
}
loop {
if check32 != h32 {
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
return ERROR__12!();
}
if true {
break;
}
}
}
ZSTD_DCtx_trace_end(dctx as *const ZSTD_DCtx_s, unsafe { (*dctx).decodedSize }, unsafe { (*dctx).processedCSize }, 1i32); // streaming
unsafe {
(*dctx).expected = 0;
}
unsafe {
(*dctx).stage = 0;
}
return 0;
}
6 => {
ZSTD_memcpy__2!(unsafe { (*dctx).headerBuffer.as_mut_ptr() as *mut u8 }.wrapping_offset(8u64.wrapping_sub(srcSize) as isize), src, srcSize); // complete skippable header
unsafe {
(*dctx).expected = MEM_readLE32(((*dctx).headerBuffer.as_mut_ptr() as *mut u8).wrapping_offset(4 as isize) as *const ()) as u64;
}
// note : dctx->expected can grow seriously large, beyond local buffer size
unsafe {
(*dctx).stage = 7;
}
return 0;
}
7 => {
unsafe {
(*dctx).expected = 0;
}
unsafe {
(*dctx).stage = 0;
}
return 0;
}
_ => {
// impossible
loop {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"impossible to reach".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__5!();
}
// some compilers require default to do something
}
}
unsafe { core::mem::zeroed() }
}

fn ZSTD_refDictContent(dctx: *mut ZSTD_DCtx_s, dict: *const (), dictSize: u64) -> u64 {
unsafe {
(*dctx).dictEnd = (*dctx).previousDstEnd;
}
unsafe {
(*dctx).virtualStart = (dict as *const u8).wrapping_offset(-((((*dctx).previousDstEnd as *const u8).offset_from((*dctx).prefixStart as *const u8) as i64) as isize)) as *const ();
}
unsafe {
(*dctx).prefixStart = dict;
}
unsafe {
(*dctx).previousDstEnd = (dict as *const u8).wrapping_offset(dictSize as isize) as *const ();
}
0
}

/// ZSTD_loadDEntropy() :
///  dict : must point at beginning of a valid zstd dictionary.
/// @return : size of entropy tables read
#[unsafe(no_mangle)]
extern "C" fn ZSTD_loadDEntropy(entropy: *mut __SlateRecord_zstd_decompress_internal_h_88, dict: *const (), dictSize: u64) -> u64 {
let mut dictPtr: *const u8 = dict as *const u8;
let dictEnd: *const u8 = dictPtr.wrapping_offset(dictSize as isize);
loop {
if dictSize <= 8 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"dict is too small".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__4!();
}
if true {
break;
}
}
// dict must be valid
let __v5: *const u8 = dictPtr;
dictPtr = __v5.wrapping_offset(8 as isize); // skip header = magic + dictID
ZSTD_STATIC_ASSERT!();
ZSTD_STATIC_ASSERT!();
ZSTD_STATIC_ASSERT!();
let workspace: *mut () = (unsafe { core::ptr::addr_of_mut!((*entropy).LLTable) }) as *mut (); // use fse tables as temporary workspace; implies fse tables are grouped together
let workspaceSize: u64 = (core::mem::size_of::<[__SlateRecord_zstd_decompress_internal_h_73; 513]>() as u64).wrapping_add(core::mem::size_of::<[__SlateRecord_zstd_decompress_internal_h_73; 257]>() as u64).wrapping_add(core::mem::size_of::<[__SlateRecord_zstd_decompress_internal_h_73; 513]>() as u64);
let hSize: u64 = unsafe { HUF_readDTableX2_wksp((*entropy).hufTable.as_mut_ptr() as *mut u32, dictPtr as *const (), (dictEnd.offset_from(dictPtr as *const u8) as i64) as u64, workspace, workspaceSize, 0i32) }; // flags
loop {
if ERR_isError(hSize) != 0 {
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
return ERROR__4!();
}
if true {
break;
}
}
let __v12: *const u8 = dictPtr;
dictPtr = __v12.wrapping_offset(hSize as isize);
let mut offcodeNCount: __SlateAlign16<[i16; 32]> = __SlateAlign16([0i16; 32]);
let mut offcodeMaxValue: u32 = 31;
let mut offcodeLog: u32 = 0;
let offcodeHeaderSize: u64 = unsafe { FSE_readNCount(offcodeNCount.0.as_mut_ptr() as *mut i16, core::ptr::addr_of_mut!(offcodeMaxValue), core::ptr::addr_of_mut!(offcodeLog), dictPtr as *const (), (dictEnd.offset_from(dictPtr as *const u8) as i64) as u64) };
loop {
if ERR_isError(offcodeHeaderSize) != 0 {
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
return ERROR__4!();
}
if true {
break;
}
}
loop {
if offcodeMaxValue > 31 {
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
return ERROR__4!();
}
if true {
break;
}
}
loop {
if offcodeLog > 8 {
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
return ERROR__4!();
}
if true {
break;
}
}
unsafe { ZSTD_buildFSETable((*entropy).OFTable.as_mut_ptr() as *mut __SlateRecord_zstd_decompress_internal_h_73, offcodeNCount.0.as_mut_ptr() as *const i16, offcodeMaxValue, core::ptr::addr_of!(OF_base.0) as *const u32, core::ptr::addr_of!(OF_bits.0) as *const u8, offcodeLog, ((*entropy).workspace.as_mut_ptr() as *mut u32) as *mut (), core::mem::size_of::<[u32; 157]>() as u64, 0i32) }; // bmi2
let __v29: *const u8 = dictPtr;
dictPtr = __v29.wrapping_offset(offcodeHeaderSize as isize);
let mut matchlengthNCount: __SlateAlign16<[i16; 53]> = __SlateAlign16([0i16; 53]);
let mut matchlengthMaxValue: u32 = 52;
let mut matchlengthLog: u32 = 0;
let matchlengthHeaderSize: u64 = unsafe { FSE_readNCount(matchlengthNCount.0.as_mut_ptr() as *mut i16, core::ptr::addr_of_mut!(matchlengthMaxValue), core::ptr::addr_of_mut!(matchlengthLog), dictPtr as *const (), (dictEnd.offset_from(dictPtr as *const u8) as i64) as u64) };
loop {
if ERR_isError(matchlengthHeaderSize) != 0 {
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
return ERROR__4!();
}
if true {
break;
}
}
loop {
if matchlengthMaxValue > 52 {
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
return ERROR__4!();
}
if true {
break;
}
}
loop {
if matchlengthLog > 9 {
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
return ERROR__4!();
}
if true {
break;
}
}
unsafe { ZSTD_buildFSETable((*entropy).MLTable.as_mut_ptr() as *mut __SlateRecord_zstd_decompress_internal_h_73, matchlengthNCount.0.as_mut_ptr() as *const i16, matchlengthMaxValue, core::ptr::addr_of!(ML_base.0) as *const u32, core::ptr::addr_of!(ML_bits.0) as *const u8, matchlengthLog, ((*entropy).workspace.as_mut_ptr() as *mut u32) as *mut (), core::mem::size_of::<[u32; 157]>() as u64, 0i32) }; // bmi2
let __v46: *const u8 = dictPtr;
dictPtr = __v46.wrapping_offset(matchlengthHeaderSize as isize);
let mut litlengthNCount: __SlateAlign16<[i16; 36]> = __SlateAlign16([0i16; 36]);
let mut litlengthMaxValue: u32 = 35;
let mut litlengthLog: u32 = 0;
let litlengthHeaderSize: u64 = unsafe { FSE_readNCount(litlengthNCount.0.as_mut_ptr() as *mut i16, core::ptr::addr_of_mut!(litlengthMaxValue), core::ptr::addr_of_mut!(litlengthLog), dictPtr as *const (), (dictEnd.offset_from(dictPtr as *const u8) as i64) as u64) };
loop {
if ERR_isError(litlengthHeaderSize) != 0 {
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
return ERROR__4!();
}
if true {
break;
}
}
loop {
if litlengthMaxValue > 35 {
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
return ERROR__4!();
}
if true {
break;
}
}
loop {
if litlengthLog > 9 {
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
return ERROR__4!();
}
if true {
break;
}
}
unsafe { ZSTD_buildFSETable((*entropy).LLTable.as_mut_ptr() as *mut __SlateRecord_zstd_decompress_internal_h_73, litlengthNCount.0.as_mut_ptr() as *const i16, litlengthMaxValue, core::ptr::addr_of!(LL_base.0) as *const u32, core::ptr::addr_of!(LL_bits.0) as *const u8, litlengthLog, ((*entropy).workspace.as_mut_ptr() as *mut u32) as *mut (), core::mem::size_of::<[u32; 157]>() as u64, 0i32) }; // bmi2
let __v63: *const u8 = dictPtr;
dictPtr = __v63.wrapping_offset(litlengthHeaderSize as isize);
loop {
if dictPtr.wrapping_offset(12 as isize) > dictEnd {
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
return ERROR__4!();
}
if true {
break;
}
}
let dictContentSize: u64 = ((unsafe { dictEnd.offset_from(dictPtr.wrapping_offset(12 as isize) as *const u8) }) as i64) as u64;
for i in 0i32..3i32 {
let rep: u32 = MEM_readLE32(dictPtr as *const ());
let __v70: *const u8 = dictPtr;
dictPtr = __v70.wrapping_offset(4 as isize);
loop {
if rep == 0 || (rep as u64) > dictContentSize {
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
return ERROR__4!();
}
if true {
break;
}
}
unsafe {
*((*entropy).rep.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) = rep;
}
}
((unsafe { dictPtr.offset_from(dict as *const u8) }) as i64) as u64
}

fn ZSTD_decompress_insertDictionary(dctx: *mut ZSTD_DCtx_s, mut dict: *const (), mut dictSize: u64) -> u64 {
if dictSize < 8 {
return ZSTD_refDictContent(dctx, dict, dictSize);
}
let magic: u32 = MEM_readLE32(dict);
if magic != 3962610743 {
return ZSTD_refDictContent(dctx, dict, dictSize); // pure content mode
}
unsafe {
(*dctx).dictID = MEM_readLE32((dict as *const u8).wrapping_offset(4 as isize) as *const ());
}
// load entropy tables
let eSize: u64 = ZSTD_loadDEntropy(unsafe { core::ptr::addr_of_mut!((*dctx).entropy) }, dict, dictSize);
loop {
if ERR_isError(eSize) != 0 {
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
return ERROR__4!();
}
if true {
break;
}
}
dict = (dict as *const u8).wrapping_offset(eSize as isize) as *const ();
dictSize = dictSize.wrapping_sub(eSize);
unsafe {
(*dctx).fseEntropy = 1;
}
unsafe {
(*dctx).litEntropy = 1;
}
// reference dictionary content
ZSTD_refDictContent(dctx, dict, dictSize)
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_decompressBegin(dctx: *mut ZSTD_DCtx_s) -> u64 {
unsafe {
(*dctx).expected = ZSTD_startingInputLength((*dctx).format);
}
// dctx->format must be properly set
unsafe {
(*dctx).stage = 0;
}
unsafe {
(*dctx).processedCSize = 0;
}
unsafe {
(*dctx).decodedSize = 0;
}
unsafe {
(*dctx).previousDstEnd = core::ptr::null::<()>();
}
unsafe {
(*dctx).prefixStart = core::ptr::null::<()>();
}
unsafe {
(*dctx).virtualStart = core::ptr::null::<()>();
}
unsafe {
(*dctx).dictEnd = core::ptr::null::<()>();
}
unsafe {
*((*dctx).entropy.hufTable.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) = 201326604u32;
}
// cover both little and big endian
unsafe {
(*dctx).fseEntropy = 0;
}
unsafe {
(*dctx).litEntropy = 0;
}
unsafe {
(*dctx).dictID = 0;
}
unsafe {
(*dctx).bType = 3;
}
unsafe {
(*dctx).isFrameDecompression = 1;
}
ZSTD_STATIC_ASSERT!();
ZSTD_memcpy!(unsafe { (*dctx).entropy.rep.as_mut_ptr() as *mut u32 }, unsafe { core::ptr::addr_of!(repStartValue) as *const u32 }, core::mem::size_of::<[u32; 3]>() as u64); // initial repcodes
unsafe {
(*dctx).LLTptr = ((*dctx).entropy.LLTable.as_mut_ptr() as *mut __SlateRecord_zstd_decompress_internal_h_73) as *const __SlateRecord_zstd_decompress_internal_h_73;
}
unsafe {
(*dctx).MLTptr = ((*dctx).entropy.MLTable.as_mut_ptr() as *mut __SlateRecord_zstd_decompress_internal_h_73) as *const __SlateRecord_zstd_decompress_internal_h_73;
}
unsafe {
(*dctx).OFTptr = ((*dctx).entropy.OFTable.as_mut_ptr() as *mut __SlateRecord_zstd_decompress_internal_h_73) as *const __SlateRecord_zstd_decompress_internal_h_73;
}
unsafe {
(*dctx).HUFptr = ((*dctx).entropy.hufTable.as_mut_ptr() as *mut u32) as *const u32;
}
0
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_decompressBegin_usingDict(dctx: *mut ZSTD_DCtx_s, dict: *const (), dictSize: u64) -> u64 {
loop {
let err_code: u64 = ZSTD_decompressBegin(dctx);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
if !dict.is_null() && dictSize != 0 {
loop {
if ERR_isError(ZSTD_decompress_insertDictionary(dctx, dict, dictSize)) != 0 {
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
return ERROR__4!();
}
if true {
break;
}
}
}
0
}

// ======   ZSTD_DDict   ======
#[unsafe(no_mangle)]
extern "C" fn ZSTD_decompressBegin_usingDDict(dctx: *mut ZSTD_DCtx_s, ddict: *const ZSTD_DDict_s) -> u64 {
while false {
}
if !ddict.is_null() {
let dictStart: *const u8 = (unsafe { ZSTD_DDict_dictContent(ddict) }) as *const u8;
let dictSize: u64 = unsafe { ZSTD_DDict_dictSize(ddict) };
let dictEnd: *const () = dictStart.wrapping_offset(dictSize as isize) as *const ();
unsafe {
(*dctx).ddictIsCold = ((*dctx).dictEnd != dictEnd) as i32;
}
while false {
}
}
loop {
let err_code: u64 = ZSTD_decompressBegin(dctx);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
if !ddict.is_null() {
// NULL ddict is equivalent to no dictionary
unsafe { ZSTD_copyDDictParameters(dctx, ddict) };
}
0
}

/// ZSTD_getDictID_fromDict() :
/// Provides the dictID stored within dictionary.
/// if @return == 0, the dictionary is not conformant with Zstandard specification.
/// It can still be loaded, but as a content-only dictionary.
#[unsafe(no_mangle)]
extern "C" fn ZSTD_getDictID_fromDict(dict: *const (), dictSize: u64) -> u32 {
if dictSize < 8 {
return 0;
}
if MEM_readLE32(dict) != 3962610743 {
return 0;
}
MEM_readLE32((dict as *const u8).wrapping_offset(4 as isize) as *const ())
}

/// ZSTD_getDictID_fromFrame() :
/// Provides the dictID required to decompress frame stored within `src`.
/// If @return == 0, the dictID could not be decoded.
/// This could for one of the following reasons :
/// - The frame does not require a dictionary (most common case).
/// - The frame was built with dictID intentionally removed.
///   Needed dictionary is a hidden piece of information.
///   Note : this use case also happens when using a non-conformant dictionary.
/// - `srcSize` is too small, and as a result, frame header could not be decoded.
///   Note : possible if `srcSize < ZSTD_FRAMEHEADERSIZE_MAX`.
/// - This is not a Zstandard frame.
/// When identifying the exact failure cause, it's possible to use
/// ZSTD_getFrameHeader(), which will provide a more precise error code.
#[unsafe(no_mangle)]
extern "C" fn ZSTD_getDictID_fromFrame(src: *const (), srcSize: u64) -> u32 {
let mut zfp: __SlateRecord_zstd_lib_h_1489 = {
let mut __t0: __SlateRecord_zstd_lib_h_1489 = unsafe { core::mem::zeroed() };
__t0.frameContentSize = 0u64;
__t0.windowSize = 0u64;
__t0.blockSizeMax = 0u32;
__t0.frameType = 0u32;
__t0.headerSize = 0u32;
__t0.dictID = 0u32;
__t0.checksumFlag = 0u32;
__t0._reserved1 = 0u32;
__t0._reserved2 = 0u32;
__t0
};
let hError: u64 = ZSTD_getFrameHeader(core::ptr::addr_of_mut!(zfp), src, srcSize);
if ERR_isError(hError) != 0 {
return 0;
}
zfp.dictID
}

/// ZSTD_decompress_usingDDict() :
/// Decompression using a pre-digested Dictionary
/// Use dictionary without significant overhead.
#[unsafe(no_mangle)]
extern "C" fn ZSTD_decompress_usingDDict(dctx: *mut ZSTD_DCtx_s, dst: *mut (), dstCapacity: u64, src: *const (), srcSize: u64, ddict: *const ZSTD_DDict_s) -> u64 {
// pass content and size in case legacy frames are encountered
ZSTD_decompressMultiFrame(dctx, dst, dstCapacity, src, srcSize, core::ptr::null::<()>(), 0u64, ddict)
}

// Streaming decompression
#[unsafe(no_mangle)]
extern "C" fn ZSTD_createDStream() -> *mut ZSTD_DCtx_s {
while false {
}
ZSTD_createDCtx_internal(unsafe { ZSTD_defaultCMem.0 })
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_initStaticDStream(workspace: *mut (), workspaceSize: u64) -> *mut ZSTD_DCtx_s {
ZSTD_initStaticDCtx(workspace, workspaceSize)
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_createDStream_advanced(customMem: __SlateRecord_zstd_lib_h_1837) -> *mut ZSTD_DCtx_s {
ZSTD_createDCtx_internal(customMem)
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_freeDStream(zds: *mut ZSTD_DCtx_s) -> u64 {
ZSTD_freeDCtx(zds)
}

// ***  Initialization  ***
#[unsafe(no_mangle)]
extern "C" fn ZSTD_DStreamInSize() -> u64 {
131072u64.wrapping_add(unsafe { ZSTD_blockHeaderSize })
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_DStreamOutSize() -> u64 {
131072u64
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_DCtx_loadDictionary_advanced(dctx: *mut ZSTD_DCtx_s, dict: *const (), dictSize: u64, dictLoadMethod: u32, dictContentType: u32) -> u64 {
loop {
if ((unsafe { (*dctx).streamStage }) as u32) != 0 {
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
return ERROR__9!();
}
if true {
break;
}
}
ZSTD_clearDict(dctx);
if !dict.is_null() && dictSize != 0 {
unsafe {
(*dctx).ddictLocal = ZSTD_createDDict_advanced(dict, dictSize, dictLoadMethod, dictContentType, (*dctx).customMem);
}
loop {
if unsafe { (*dctx).ddictLocal }.is_null() {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"NULL pointer!".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__7!();
}
if true {
break;
}
}
unsafe {
(*dctx).ddict = (*dctx).ddictLocal as *const ZSTD_DDict_s;
}
unsafe {
(*dctx).dictUses = -1;
}
}
0
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_DCtx_loadDictionary_byReference(dctx: *mut ZSTD_DCtx_s, dict: *const (), dictSize: u64) -> u64 {
ZSTD_DCtx_loadDictionary_advanced(dctx, dict, dictSize, 1u32, 0u32)
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_DCtx_loadDictionary(dctx: *mut ZSTD_DCtx_s, dict: *const (), dictSize: u64) -> u64 {
ZSTD_DCtx_loadDictionary_advanced(dctx, dict, dictSize, 0u32, 0u32)
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_DCtx_refPrefix_advanced(dctx: *mut ZSTD_DCtx_s, prefix: *const (), prefixSize: u64, dictContentType: u32) -> u64 {
loop {
let err_code: u64 = ZSTD_DCtx_loadDictionary_advanced(dctx, prefix, prefixSize, 1u32, dictContentType);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
unsafe {
(*dctx).dictUses = 1;
}
0
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_DCtx_refPrefix(dctx: *mut ZSTD_DCtx_s, prefix: *const (), prefixSize: u64) -> u64 {
ZSTD_DCtx_refPrefix_advanced(dctx, prefix, prefixSize, 1u32)
}

/// ZSTD_initDStream_usingDict() :
/// return : expected size, aka ZSTD_startingInputLength().
/// this function cannot fail
#[unsafe(no_mangle)]
extern "C" fn ZSTD_initDStream_usingDict(zds: *mut ZSTD_DCtx_s, dict: *const (), dictSize: u64) -> u64 {
while false {
}
loop {
let err_code: u64 = ZSTD_DCtx_reset(zds, 1u32);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
loop {
let err_code: u64 = ZSTD_DCtx_loadDictionary(zds, dict, dictSize);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
ZSTD_startingInputLength(unsafe { (*zds).format })
}

/// note : this variant can't fail
#[unsafe(no_mangle)]
extern "C" fn ZSTD_initDStream(zds: *mut ZSTD_DCtx_s) -> u64 {
while false {
}
loop {
let err_code: u64 = ZSTD_DCtx_reset(zds, 1u32);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
loop {
let err_code: u64 = ZSTD_DCtx_refDDict(zds, core::ptr::null::<ZSTD_DDict_s>());
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
ZSTD_startingInputLength(unsafe { (*zds).format })
}

/// ZSTD_initDStream_usingDDict() :
/// ddict will just be referenced, and must outlive decompression session
/// this function cannot fail
#[unsafe(no_mangle)]
extern "C" fn ZSTD_initDStream_usingDDict(dctx: *mut ZSTD_DCtx_s, ddict: *const ZSTD_DDict_s) -> u64 {
while false {
}
loop {
let err_code: u64 = ZSTD_DCtx_reset(dctx, 1u32);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
loop {
let err_code: u64 = ZSTD_DCtx_refDDict(dctx, ddict);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
ZSTD_startingInputLength(unsafe { (*dctx).format })
}

/// ZSTD_resetDStream() :
/// return : expected size, aka ZSTD_startingInputLength().
/// this function cannot fail
#[unsafe(no_mangle)]
extern "C" fn ZSTD_resetDStream(dctx: *mut ZSTD_DCtx_s) -> u64 {
while false {
}
loop {
let err_code: u64 = ZSTD_DCtx_reset(dctx, 1u32);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
ZSTD_startingInputLength(unsafe { (*dctx).format })
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_DCtx_refDDict(dctx: *mut ZSTD_DCtx_s, ddict: *const ZSTD_DDict_s) -> u64 {
loop {
if ((unsafe { (*dctx).streamStage }) as u32) != 0 {
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
return ERROR__9!();
}
if true {
break;
}
}
ZSTD_clearDict(dctx);
if !ddict.is_null() {
unsafe {
(*dctx).ddict = ddict;
}
unsafe {
(*dctx).dictUses = -1;
}
if ((unsafe { (*dctx).refMultipleDDicts }) as u32) == 1 {
if unsafe { (*dctx).ddictSet }.is_null() {
unsafe {
(*dctx).ddictSet = ZSTD_createDDictHashSet((*dctx).customMem);
}
if unsafe { (*dctx).ddictSet }.is_null() {
loop {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"Failed to allocate memory for hash set!".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__7!();
}
}
}
// Impossible: ddictSet cannot have been allocated if static dctx
loop {
let err_code: u64 = ZSTD_DDictHashSet_addDDict(unsafe { (*dctx).ddictSet }, ddict, unsafe { (*dctx).customMem });
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
}
}
0
}

/// ZSTD_DCtx_setMaxWindowSize() :
/// note : no direct equivalence in ZSTD_DCtx_setParameter,
/// since this version sets windowSize, and the other sets windowLog
#[unsafe(no_mangle)]
extern "C" fn ZSTD_DCtx_setMaxWindowSize(dctx: *mut ZSTD_DCtx_s, maxWindowSize: u64) -> u64 {
let bounds: __SlateRecord_zstd_lib_h_527 = ZSTD_dParam_getBounds(100u32);
let min: u64 = 1u64 << bounds.lowerBound;
let max: u64 = 1u64 << bounds.upperBound;
loop {
if ((unsafe { (*dctx).streamStage }) as u32) != 0 {
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
return ERROR__9!();
}
if true {
break;
}
}
loop {
if maxWindowSize < min {
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
return ERROR__13!();
}
if true {
break;
}
}
loop {
if maxWindowSize > max {
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
return ERROR__13!();
}
if true {
break;
}
}
unsafe {
(*dctx).maxWindowSize = maxWindowSize;
}
0
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_DCtx_setFormat(dctx: *mut ZSTD_DCtx_s, format: u32) -> u64 {
ZSTD_DCtx_setParameter(dctx, 1000u32, (format as u32) as i32)
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_dParam_getBounds(dParam: u32) -> __SlateRecord_zstd_lib_h_527 {
let mut bounds: __SlateRecord_zstd_lib_h_527 = __SlateRecord_zstd_lib_h_527 { error: 0, lowerBound: 0, upperBound: 0 };
match dParam as u32 {
100 => {
bounds.lowerBound = 10;
bounds.upperBound = if (core::mem::size_of::<u64>() as u64) == 4 { 30 } else { 31 };
return bounds;
}
1000 => {
bounds.lowerBound = 0;
bounds.upperBound = 1;
ZSTD_STATIC_ASSERT!();
return bounds;
}
1001 => {
bounds.lowerBound = 0;
bounds.upperBound = 1;
return bounds;
}
1002 => {
bounds.lowerBound = 0;
bounds.upperBound = 1;
return bounds;
}
1003 => {
bounds.lowerBound = 0;
bounds.upperBound = 1;
return bounds;
}
1004 => {
bounds.lowerBound = 0;
bounds.upperBound = 1;
return bounds;
}
1005 => {
bounds.lowerBound = 1024i32;
bounds.upperBound = 131072i32;
return bounds;
}
_ => {
}
}
bounds.error = ERROR__10!();
return bounds;
}

/// ZSTD_dParam_withinBounds:
/// @return 1 if value is within dParam bounds,
/// 0 otherwise
fn ZSTD_dParam_withinBounds(dParam: u32, value: i32) -> i32 {
let bounds: __SlateRecord_zstd_lib_h_527 = ZSTD_dParam_getBounds(dParam);
if ERR_isError(bounds.error) != 0 {
return 0;
}
if value < bounds.lowerBound {
return 0;
}
if value > bounds.upperBound {
return 0;
}
1
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_DCtx_getParameter(dctx: *mut ZSTD_DCtx_s, param: u32, value: *mut i32) -> u64 {
match param as u32 {
100 => {
unsafe {
*value = ZSTD_highbit32((*dctx).maxWindowSize as u32) as i32;
}
return 0;
}
1000 => {
unsafe {
*value = ((*dctx).format as u32) as i32;
}
return 0;
}
1001 => {
unsafe {
*value = ((*dctx).outBufferMode as u32) as i32;
}
return 0;
}
1002 => {
unsafe {
*value = ((*dctx).forceIgnoreChecksum as u32) as i32;
}
return 0;
}
1003 => {
unsafe {
*value = ((*dctx).refMultipleDDicts as u32) as i32;
}
return 0;
}
1004 => {
unsafe {
*value = (*dctx).disableHufAsm;
}
return 0;
}
1005 => {
unsafe {
*value = (*dctx).maxBlockSizeParam;
}
return 0;
}
_ => {
}
}
loop {
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
return ERROR__10!();
}
unsafe { core::mem::zeroed() }
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_DCtx_setParameter(dctx: *mut ZSTD_DCtx_s, dParam: u32, mut value: i32) -> u64 {
loop {
if ((unsafe { (*dctx).streamStage }) as u32) != 0 {
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
return ERROR__9!();
}
if true {
break;
}
}
match dParam as u32 {
100 => {
if value == 0 {
value = 27;
}
CHECK_DBOUNDS!(100i32, value);
unsafe {
(*dctx).maxWindowSize = 1u64 << value;
}
return 0;
}
1000 => {
CHECK_DBOUNDS!(1000i32, value);
unsafe {
(*dctx).format = value as u32;
}
return 0;
}
1001 => {
CHECK_DBOUNDS!(1001i32, value);
unsafe {
(*dctx).outBufferMode = value as u32;
}
return 0;
}
1002 => {
CHECK_DBOUNDS!(1002i32, value);
unsafe {
(*dctx).forceIgnoreChecksum = value as u32;
}
return 0;
}
1003 => {
CHECK_DBOUNDS!(1003i32, value);
if (unsafe { (*dctx).staticSize }) != 0 {
loop {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"Static dctx does not support multiple DDicts!".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__10!();
}
}
unsafe {
(*dctx).refMultipleDDicts = value as u32;
}
return 0;
}
1004 => {
CHECK_DBOUNDS!(1004i32, value);
unsafe {
(*dctx).disableHufAsm = (value != 0) as i32;
}
return 0;
}
1005 => {
if value != 0 {
CHECK_DBOUNDS!(1005i32, value);
}
unsafe {
(*dctx).maxBlockSizeParam = value;
}
return 0;
}
_ => {
}
}
loop {
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
return ERROR__10!();
}
unsafe { core::mem::zeroed() }
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_DCtx_reset(dctx: *mut ZSTD_DCtx_s, reset: u32) -> u64 {
if (reset as u32) == 1 || (reset as u32) == 3 {
unsafe {
(*dctx).streamStage = 0;
}
unsafe {
(*dctx).noForwardProgress = 0;
}
unsafe {
(*dctx).isFrameDecompression = 1;
}
}
if (reset as u32) == 2 || (reset as u32) == 3 {
loop {
if ((unsafe { (*dctx).streamStage }) as u32) != 0 {
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
return ERROR__9!();
}
if true {
break;
}
}
ZSTD_clearDict(dctx);
ZSTD_DCtx_resetParameters(dctx);
}
0
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_sizeof_DStream(dctx: *const ZSTD_DCtx_s) -> u64 {
ZSTD_sizeof_DCtx(dctx)
}

fn ZSTD_decodingBufferSize_internal(windowSize: u64, frameContentSize: u64, blockSizeMax: u64) -> u64 {
let blockSize: u64 = MIN!(if windowSize < 131072u64 { windowSize } else { 131072u64 }, blockSizeMax);
// We need blockSize + WILDCOPY_OVERLENGTH worth of buffer so that if a block
// ends at windowSize + WILDCOPY_OVERLENGTH + 1 bytes, we can start writing
// the block at the beginning of the output buffer, and maintain a full window.
//
// We need another blockSize worth of buffer so that we can store split
// literals at the end of the block without overwriting the extDict window.
let neededRBSize: u64 = windowSize.wrapping_add(blockSize.wrapping_mul(2)).wrapping_add(64u64);
let neededSize: u64 = MIN!(frameContentSize, neededRBSize);
let minRBSize: u64 = neededSize;
loop {
if minRBSize != neededSize {
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
return ERROR__11!();
}
if true {
break;
}
}
minRBSize
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_decodingBufferSize_min(windowSize: u64, frameContentSize: u64) -> u64 {
ZSTD_decodingBufferSize_internal(windowSize, frameContentSize, 131072u64)
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_estimateDStreamSize(windowSize: u64) -> u64 {
let blockSize: u64 = MIN__2!(windowSize, 131072i32);
let inBuffSize: u64 = blockSize; // no block can be larger
let outBuffSize: u64 = ZSTD_decodingBufferSize_min(windowSize, 18446744073709551615u64);
ZSTD_estimateDCtxSize().wrapping_add(inBuffSize).wrapping_add(outBuffSize)
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_estimateDStreamSize_fromFrame(src: *const (), srcSize: u64) -> u64 {
let windowSizeMax: u32 = 1u32 << if (core::mem::size_of::<u64>() as u64) == 4 { 30i32 } else { 31i32 }; // note : should be user-selectable, but requires an additional parameter (or a dctx)
let mut zfh: __SlateRecord_zstd_lib_h_1489 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(zfh) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
let err: u64 = ZSTD_getFrameHeader(core::ptr::addr_of_mut!(zfh), src, srcSize);
if ERR_isError(err) != 0 {
return err;
}
loop {
if err > 0 {
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
return ERROR__2!();
}
if true {
break;
}
}
loop {
if zfh.windowSize > (windowSizeMax as u64) {
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
return ERROR__11!();
}
if true {
break;
}
}
ZSTD_estimateDStreamSize(zfh.windowSize)
}

// *****   Decompression   *****
fn ZSTD_DCtx_isOverflow(zds: *mut ZSTD_DCtx_s, neededInBuffSize: u64, neededOutBuffSize: u64) -> i32 {
(unsafe { (*zds).inBuffSize }.wrapping_add(unsafe { (*zds).outBuffSize }) >= neededInBuffSize.wrapping_add(neededOutBuffSize).wrapping_mul(3)) as i32
}

fn ZSTD_DCtx_updateOversizedDuration(zds: *mut ZSTD_DCtx_s, neededInBuffSize: u64, neededOutBuffSize: u64) {
if ZSTD_DCtx_isOverflow(zds, neededInBuffSize, neededOutBuffSize) != 0 {
let __v1: u64 = unsafe { (*zds).oversizedDuration };
unsafe {
(*zds).oversizedDuration = __v1.wrapping_add(1);
}
} else {
unsafe {
(*zds).oversizedDuration = 0;
}
}
}

fn ZSTD_DCtx_isOversizedTooLong(zds: *mut ZSTD_DCtx_s) -> i32 {
((unsafe { (*zds).oversizedDuration }) >= 128) as i32
}

/// Checks that the output buffer hasn't changed if ZSTD_obm_stable is used.
fn ZSTD_checkOutBuffer(zds: *const ZSTD_DCtx_s, output: *const ZSTD_outBuffer_s) -> u64 {
let expect: ZSTD_outBuffer_s = unsafe { (*zds).expectedOutBuffer };
// No requirement when ZSTD_obm_stable is not enabled.
if ((unsafe { (*zds).outBufferMode }) as u32) != 1 {
return 0;
}
// Any buffer is allowed in zdss_init, this must be the same for every other call until
// the context is reset.
if ((unsafe { (*zds).streamStage }) as u32) == 0 {
return 0;
}
// The buffer must match our expectation exactly.
if expect.dst == unsafe { (*output).dst } && expect.pos == unsafe { (*output).pos } && expect.size == unsafe { (*output).size } {
return 0;
}
loop {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"ZSTD_d_stableOutBuffer enabled but output differs!".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__22!();
}
unsafe { core::mem::zeroed() }
}

/// Calls ZSTD_decompressContinue() with the right parameters for ZSTD_decompressStream()
/// and updates the stage and the output buffer state. This call is extracted so it can be
/// used both when reading directly from the ZSTD_inBuffer, and in buffered input mode.
/// NOTE: You must break after calling this function since the streamStage is modified.
fn ZSTD_decompressContinueStream(zds: *mut ZSTD_DCtx_s, op: &mut *mut u8, oend: *mut u8, src: *const (), srcSize: u64) -> u64 {
let isSkipFrame: i32 = ZSTD_isSkipFrame(zds);
if ((unsafe { (*zds).outBufferMode }) as u32) == 0 {
let dstSize: u64 = if isSkipFrame != 0 { 0 } else { unsafe { (*zds).outBuffSize }.wrapping_sub(unsafe { (*zds).outStart }) };
let decodedSize: u64 = ZSTD_decompressContinue(zds, unsafe { (*zds).outBuff }.wrapping_offset((unsafe { (*zds).outStart }) as isize) as *mut (), dstSize, src, srcSize);
loop {
let err_code: u64 = decodedSize;
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
if decodedSize == 0 && isSkipFrame == 0 {
unsafe {
(*zds).streamStage = 2;
}
} else {
unsafe {
(*zds).outEnd = unsafe { (*zds).outStart }.wrapping_add(decodedSize);
}
unsafe {
(*zds).streamStage = 4;
}
}
} else {
// Write directly into the output buffer
let dstSize: u64 = if isSkipFrame != 0 { 0 } else { ((unsafe { oend.offset_from(*(op as *mut *mut u8) as *mut u8) }) as i64) as u64 };
let decodedSize: u64 = ZSTD_decompressContinue(zds, (unsafe { *(op as *mut *mut u8) }) as *mut (), dstSize, src, srcSize);
loop {
let err_code: u64 = decodedSize;
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
let __v11: *mut u8 = unsafe { *op };
unsafe {
*op = __v11.wrapping_offset(decodedSize as isize);
}
// Flushing is not needed.
unsafe {
(*zds).streamStage = 2;
}
}
0
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_decompressStream(zds: *mut ZSTD_DCtx_s, output: *mut ZSTD_outBuffer_s, input: *mut ZSTD_inBuffer_s) -> u64 {
let src: *const u8 = (unsafe { (*input).src }) as *const u8;
let istart: *const u8 = if (unsafe { (*input).pos }) != 0 { src.wrapping_offset((unsafe { (*input).pos }) as isize) } else { src };
let iend: *const u8 = if (unsafe { (*input).size }) != 0 { src.wrapping_offset((unsafe { (*input).size }) as isize) } else { src };
let mut ip: *const u8 = istart;
let dst: *mut u8 = (unsafe { (*output).dst }) as *mut u8;
let ostart: *mut u8 = if (unsafe { (*output).pos }) != 0 { dst.wrapping_offset((unsafe { (*output).pos }) as isize) } else { dst };
let oend: *mut u8 = if (unsafe { (*output).size }) != 0 { dst.wrapping_offset((unsafe { (*output).size }) as isize) } else { dst };
let mut op: *mut u8 = ostart;
let mut someMoreWork: u32 = 1;
while false {
}
loop {
if (unsafe { (*input).pos }) > unsafe { (*input).size } {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"forbidden. in: pos: %u   vs size: %u".as_ptr() as *const u8, (*input).pos as u32, (*input).size as u32) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__2!();
}
if true {
break;
}
}
loop {
if (unsafe { (*output).pos }) > unsafe { (*output).size } {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"forbidden. out: pos: %u   vs size: %u".as_ptr() as *const u8, (*output).pos as u32, (*output).size as u32) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__3!();
}
if true {
break;
}
}
while false {
}
loop {
let err_code: u64 = ZSTD_checkOutBuffer(zds as *const ZSTD_DCtx_s, output as *const ZSTD_outBuffer_s);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
while someMoreWork != 0 {
let mut __t0: i64 = match (unsafe { (*zds).streamStage }) as u32 { 0 => 0, 1 => 1, 2 => 2, 3 => 3, 4 => 4, _ => 5 };
loop {
match __t0 {
0 => {
while false {
}
unsafe {
(*zds).streamStage = 1;
}
unsafe {
(*zds).outEnd = 0;
}
unsafe {
(*zds).outStart = 0;
}
unsafe {
(*zds).inPos = 0;
}
unsafe {
(*zds).lhSize = 0;
}
unsafe {
(*zds).hostageByte = 0;
}
unsafe {
(*zds).expectedOutBuffer = *output;
}
__t0 = 1;
continue;
}
1 => {
while false {
}
let hSize: u64 = ZSTD_getFrameHeader_advanced(unsafe { core::ptr::addr_of_mut!((*zds).fParams) }, (unsafe { (*zds).headerBuffer.as_mut_ptr() as *mut u8 }) as *const (), unsafe { (*zds).lhSize }, unsafe { (*zds).format });
if ((unsafe { (*zds).refMultipleDDicts }) as u32) != 0 && !unsafe { (*zds).ddictSet }.is_null() {
ZSTD_DCtx_selectFrameDDict(zds);
}
if ERR_isError(hSize) != 0 {
return hSize; // error
}
if hSize != 0 {
// need more input
let toLoad: u64 = hSize.wrapping_sub(unsafe { (*zds).lhSize }); // if hSize!=0, hSize > zds->lhSize
let remainingInput: u64 = ((unsafe { iend.offset_from(ip as *const u8) }) as i64) as u64;
if toLoad > remainingInput {
// not enough input to load full header
if remainingInput > 0 {
ZSTD_memcpy!(unsafe { (*zds).headerBuffer.as_mut_ptr() as *mut u8 }.wrapping_offset((unsafe { (*zds).lhSize }) as isize), ip, remainingInput);
let __v21: u64 = unsafe { (*zds).lhSize };
unsafe {
(*zds).lhSize = __v21.wrapping_add(remainingInput);
}
}
unsafe {
(*input).pos = (*input).size;
}
// check first few bytes
loop {
let err_code: u64 = ZSTD_getFrameHeader_advanced(unsafe { core::ptr::addr_of_mut!((*zds).fParams) }, (unsafe { (*zds).headerBuffer.as_mut_ptr() as *mut u8 }) as *const (), unsafe { (*zds).lhSize }, unsafe { (*zds).format });
if ERR_isError(err_code) != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"First few bytes detected incorrect".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return err_code;
}
if true {
break;
}
}
// return hint input size
return MAX!((if ((unsafe { (*zds).format }) as u32) == 0 { 6i32 } else { 2i32 }) as u64, hSize).wrapping_sub(unsafe { (*zds).lhSize }).wrapping_add(unsafe { ZSTD_blockHeaderSize }); // remaining header bytes + next block header
}
ZSTD_memcpy!(unsafe { (*zds).headerBuffer.as_mut_ptr() as *mut u8 }.wrapping_offset((unsafe { (*zds).lhSize }) as isize), ip, toLoad);
unsafe {
(*zds).lhSize = hSize;
}
let __v28: *const u8 = ip;
ip = __v28.wrapping_offset(toLoad as isize);
break;
}
// check for single-pass mode opportunity
if (unsafe { (*zds).fParams.frameContentSize }) != 18446744073709551615u64 && ((unsafe { (*zds).fParams.frameType }) as u32) != 1 && (((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64) >= unsafe { (*zds).fParams.frameContentSize } {
let cSize: u64 = ZSTD_findFrameCompressedSize_advanced(istart as *const (), ((unsafe { iend.offset_from(istart as *const u8) }) as i64) as u64, unsafe { (*zds).format });
if cSize <= (((unsafe { iend.offset_from(istart as *const u8) }) as i64) as u64) {
// shortcut : using single-pass mode
let decompressedSize: u64 = ZSTD_decompress_usingDDict(zds, op as *mut (), ((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64, istart as *const (), cSize, ZSTD_getDDict(zds));
if ERR_isError(decompressedSize) != 0 {
return decompressedSize;
}
while false {
}
ip = istart.wrapping_offset(cSize as isize);
op = if !op.is_null() { op.wrapping_offset(decompressedSize as isize) } else { op }; // can occur if frameContentSize = 0 (empty frame)
unsafe {
(*zds).expected = 0;
}
unsafe {
(*zds).streamStage = 0;
}
someMoreWork = 0;
break;
}
}
// Check output buffer is large enough for ZSTD_odm_stable.
if ((unsafe { (*zds).outBufferMode }) as u32) == 1 && ((unsafe { (*zds).fParams.frameType }) as u32) != 1 && (unsafe { (*zds).fParams.frameContentSize }) != 18446744073709551615u64 && (((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64) < unsafe { (*zds).fParams.frameContentSize } {
loop {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"ZSTD_obm_stable passed but ZSTD_outBuffer is too small".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__3!();
}
}
// Consume header (see ZSTDds_decodeFrameHeader)
while false {
}
loop {
let err_code: u64 = ZSTD_decompressBegin_usingDDict(zds, ZSTD_getDDict(zds));
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
let __v42: bool = ((unsafe { (*zds).format }) as u32) == 0 && MEM_readLE32((unsafe { (*zds).headerBuffer.as_mut_ptr() as *mut u8 }) as *const ()) & 4294967280 == 407710288;
if __v42 {
// skippable frame
unsafe {
(*zds).expected = MEM_readLE32(((*zds).headerBuffer.as_mut_ptr() as *mut u8).wrapping_offset(4 as isize) as *const ()) as u64;
}
unsafe {
(*zds).stage = 7;
}
} else {
loop {
let err_code: u64 = ZSTD_decodeFrameHeader(zds, (unsafe { (*zds).headerBuffer.as_mut_ptr() as *mut u8 }) as *const (), unsafe { (*zds).lhSize });
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
unsafe {
(*zds).expected = ZSTD_blockHeaderSize;
}
unsafe {
(*zds).stage = 2;
}
}
// control buffer memory usage
while false {
}
unsafe {
(*zds).fParams.windowSize = MAX__5!(unsafe { (*zds).fParams.windowSize }, 1024u32);
}
loop {
if (unsafe { (*zds).fParams.windowSize }) > unsafe { (*zds).maxWindowSize } {
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
return ERROR__11!();
}
if true {
break;
}
}
if (unsafe { (*zds).maxBlockSizeParam }) != 0 {
unsafe {
(*zds).fParams.blockSizeMax = MIN!(unsafe { (*zds).fParams.blockSizeMax }, (*zds).maxBlockSizeParam as u32);
}
}
// Adapt buffer sizes to frame header instructions
let neededInBuffSize: u64 = MAX__2!(unsafe { (*zds).fParams.blockSizeMax }, 4i32) as u64; // frame checksum
let mut neededOutBuffSize: u64 = 0;
let __v54: u64 = if ((unsafe { (*zds).outBufferMode }) as u32) == 0 {
ZSTD_decodingBufferSize_internal(unsafe { (*zds).fParams.windowSize }, unsafe { (*zds).fParams.frameContentSize }, (unsafe { (*zds).fParams.blockSizeMax }) as u64)
} else {
0
};
neededOutBuffSize = __v54;
ZSTD_DCtx_updateOversizedDuration(zds, neededInBuffSize, neededOutBuffSize);
let tooSmall: i32 = ((unsafe { (*zds).inBuffSize }) < neededInBuffSize || (unsafe { (*zds).outBuffSize }) < neededOutBuffSize) as i32;
let tooLarge: i32 = ZSTD_DCtx_isOversizedTooLong(zds);
if tooSmall != 0 || tooLarge != 0 {
let bufferSize: u64 = neededInBuffSize.wrapping_add(neededOutBuffSize);
while false {
}
while false {
}
if (unsafe { (*zds).staticSize }) != 0 {
// static DCtx
while false {
}
// controlled at init
loop {
if bufferSize > unsafe { (*zds).staticSize }.wrapping_sub(core::mem::size_of::<ZSTD_DCtx_s>() as u64) {
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
return ERROR__7!();
}
if true {
break;
}
}
} else {
ZSTD_customFree((unsafe { (*zds).inBuff }) as *mut (), unsafe { (*zds).customMem });
unsafe {
(*zds).inBuffSize = 0;
}
unsafe {
(*zds).outBuffSize = 0;
}
unsafe {
(*zds).inBuff = ZSTD_customMalloc(bufferSize, (*zds).customMem) as *mut u8;
}
loop {
if unsafe { (*zds).inBuff }.is_null() {
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
return ERROR__7!();
}
if true {
break;
}
}
}
unsafe {
(*zds).inBuffSize = neededInBuffSize;
}
unsafe {
(*zds).outBuff = unsafe { (*zds).inBuff }.wrapping_offset((*zds).inBuffSize as isize);
}
unsafe {
(*zds).outBuffSize = neededOutBuffSize;
}
}
unsafe {
(*zds).streamStage = 2;
}
__t0 = 2;
continue;
}
2 => {
while false {
}
let neededInSize: u64 = ZSTD_nextSrcSizeToDecompressWithInputSize(zds, ((unsafe { iend.offset_from(ip as *const u8) }) as i64) as u64);
while false {
}
if neededInSize == 0 {
// end of frame
unsafe {
(*zds).streamStage = 0;
}
someMoreWork = 0;
break;
}
if (((unsafe { iend.offset_from(ip as *const u8) }) as i64) as u64) >= neededInSize {
// decode directly from src
loop {
let err_code: u64 = ZSTD_decompressContinueStream(zds, unsafe { &mut (*core::ptr::addr_of_mut!(op)) }, oend, ip as *const (), neededInSize);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
let __v75: *const u8 = ip;
ip = __v75.wrapping_offset(neededInSize as isize);
// Function modifies the stage so we must break
break;
}
if ip == iend {
someMoreWork = 0;
break;
}
// no more input
unsafe {
(*zds).streamStage = 3;
}
__t0 = 3;
continue;
}
3 => {
let neededInSize: u64 = ZSTD_nextSrcSizeToDecompress(zds);
let toLoad: u64 = neededInSize.wrapping_sub(unsafe { (*zds).inPos });
let isSkipFrame: i32 = ZSTD_isSkipFrame(zds);
let mut loadedSize: u64 = 0;
// At this point we shouldn't be decompressing a block that we can stream.
if isSkipFrame != 0 {
loadedSize = MIN!(toLoad, ((unsafe { iend.offset_from(ip as *const u8) }) as i64) as u64);
} else {
loop {
if toLoad > unsafe { (*zds).inBuffSize }.wrapping_sub(unsafe { (*zds).inPos }) {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"should never happen".as_ptr() as *const u8) };
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
loadedSize = ZSTD_limitCopy(unsafe { (*zds).inBuff }.wrapping_offset((unsafe { (*zds).inPos }) as isize) as *mut (), toLoad, ip as *const (), ((unsafe { iend.offset_from(ip as *const u8) }) as i64) as u64);
}
if loadedSize != 0 {
// ip may be NULL
let __v82: *const u8 = ip;
ip = __v82.wrapping_offset(loadedSize as isize);
let __v85: u64 = unsafe { (*zds).inPos };
unsafe {
(*zds).inPos = __v85.wrapping_add(loadedSize);
}
}
if loadedSize < toLoad {
someMoreWork = 0;
break;
}
// not enough input, wait for more
// decode loaded input
unsafe {
(*zds).inPos = 0;
}
// input is consumed
loop {
let err_code: u64 = ZSTD_decompressContinueStream(zds, unsafe { &mut (*core::ptr::addr_of_mut!(op)) }, oend, (unsafe { (*zds).inBuff }) as *const (), neededInSize);
if ERR_isError(err_code) != 0 {
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
return err_code;
}
if true {
break;
}
}
// Function modifies the stage so we must break
break;
}
4 => {
let toFlushSize: u64 = unsafe { (*zds).outEnd }.wrapping_sub(unsafe { (*zds).outStart });
let flushedSize: u64 = ZSTD_limitCopy(op as *mut (), ((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64, unsafe { (*zds).outBuff }.wrapping_offset((unsafe { (*zds).outStart }) as isize) as *const (), toFlushSize);
op = if !op.is_null() { op.wrapping_offset(flushedSize as isize) } else { op };
let __v93: u64 = unsafe { (*zds).outStart };
unsafe {
(*zds).outStart = __v93.wrapping_add(flushedSize);
}
if flushedSize == toFlushSize {
// flush completed
unsafe {
(*zds).streamStage = 2;
}
if (unsafe { (*zds).outBuffSize }) < unsafe { (*zds).fParams.frameContentSize } && unsafe { (*zds).outStart }.wrapping_add((unsafe { (*zds).fParams.blockSizeMax }) as u64) > unsafe { (*zds).outBuffSize } {
while false {
}
unsafe {
(*zds).outEnd = 0;
}
unsafe {
(*zds).outStart = 0;
}
}
break;
}
// cannot complete flush
someMoreWork = 0;
break;
}
5 => {
// impossible
loop {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"impossible to reach".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__5!();
}
// some compilers require default to do something
break;
}
_ => {
break;
}
}
}
}
// result
unsafe {
(*input).pos = (ip.offset_from((*input).src as *const u8) as i64) as u64;
}
unsafe {
(*output).pos = (op.offset_from((*output).dst as *mut u8) as i64) as u64;
}
// Update the expected output buffer for ZSTD_obm_stable.
unsafe {
(*zds).expectedOutBuffer = *output;
}
if ip == istart && op == ostart {
// no forward progress
let __v103: i32 = unsafe { (*zds).noForwardProgress };
unsafe {
(*zds).noForwardProgress = __v103.wrapping_add(1);
}
if (unsafe { (*zds).noForwardProgress }) >= 16 {
loop {
if op == oend {
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
return ERROR__23!();
}
if true {
break;
}
}
loop {
if ip == iend {
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
return ERROR__19!();
}
if true {
break;
}
}
}
} else {
unsafe {
(*zds).noForwardProgress = 0;
}
}
let mut nextSrcSizeHint: u64 = ZSTD_nextSrcSizeToDecompress(zds);
if nextSrcSizeHint == 0 {
// frame fully decoded
if (unsafe { (*zds).outEnd }) == unsafe { (*zds).outStart } {
// output fully flushed
if (unsafe { (*zds).hostageByte }) != 0 {
if (unsafe { (*input).pos }) >= unsafe { (*input).size } {
// can't release hostage (not present)
unsafe {
(*zds).streamStage = 2;
}
return 1;
}
let __v116: u64 = unsafe { (*input).pos };
unsafe {
(*input).pos = __v116.wrapping_add(1);
}
// release hostage
}
// zds->hostageByte
return 0;
}
// zds->outEnd == zds->outStart
if (unsafe { (*zds).hostageByte }) == 0 {
// output not fully flushed; keep last byte as hostage; will be released when all output is flushed
let __v119: u64 = unsafe { (*input).pos };
unsafe {
(*input).pos = __v119.wrapping_sub(1);
}
// note : pos > 0, otherwise, impossible to finish reading last block
unsafe {
(*zds).hostageByte = 1;
}
}
return 1;
}
// nextSrcSizeHint==0
let __v121: u64 = unsafe { ZSTD_blockHeaderSize }.wrapping_mul((((ZSTD_nextInputType(zds) as u32) == 2) as i32) as u64);
nextSrcSizeHint = nextSrcSizeHint.wrapping_add(__v121); // preload header of next block
nextSrcSizeHint = nextSrcSizeHint.wrapping_sub(unsafe { (*zds).inPos }); // part already loaded
nextSrcSizeHint
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_decompressStream_simpleArgs(dctx: *mut ZSTD_DCtx_s, dst: *mut (), dstCapacity: u64, dstPos: *mut u64, src: *const (), srcSize: u64, srcPos: *mut u64) -> u64 {
let mut output: ZSTD_outBuffer_s = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(output) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
let mut input: ZSTD_inBuffer_s = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(input) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
output.dst = dst;
output.size = dstCapacity;
output.pos = unsafe { *dstPos };
input.src = src;
input.size = srcSize;
input.pos = unsafe { *srcPos };
let cErr: u64 = ZSTD_decompressStream(dctx, core::ptr::addr_of_mut!(output), core::ptr::addr_of_mut!(input));
unsafe {
*dstPos = output.pos;
}
unsafe {
*srcPos = input.pos;
}
cErr
}
