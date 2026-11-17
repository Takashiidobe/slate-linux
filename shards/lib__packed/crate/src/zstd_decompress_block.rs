
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
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! cpuid(1): Processor Info and Feature Bits.
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// disable assert (default)
// disabled
// disabled
// cpuid(7): Extended Features.
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
macro_rules! BOUNDED {
($min:expr, $val:expr, $max:expr) => {
MAX!($min, if ($val) < $max { $val } else { $max })
};
}

macro_rules! COPY16 {
($d:expr, $s:expr) => {
loop {
ZSTD_copy16(($d) as *mut (), ($s) as *const ());
let __v: *mut u8 = $d;
let __v2: *mut u8 = __v.wrapping_offset(16 as isize);
$d = __v2;
let __v3: *const u8 = $s;
let __v4: *const u8 = __v3.wrapping_offset(16 as isize);
$s = __v4;
if true {
break;
}
}
};
}

macro_rules! COPY8__2 {
($d:expr, $s:expr) => {
loop {
ZSTD_copy8(($d) as *mut (), ($s) as *const ());
let __v: *mut u8 = $d;
let __v2: *mut u8 = __v.wrapping_offset(8 as isize);
$d = __v2;
let __v3: *const u8 = $s;
let __v4: *const u8 = __v3.wrapping_offset(8 as isize);
$s = __v4;
if true {
break;
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

macro_rules! ERROR__17 {
() => {
ZSTD_ERROR__14!()
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

macro_rules! ERROR__4 {
() => {
ZSTD_ERROR__6!()
};
}

macro_rules! FSE_TABLESTEP {
($tableSize:expr) => {
(($tableSize) >> 1i32).wrapping_add(($tableSize) >> 3i32).wrapping_add(3)
};
}

/// Tell the compiler that a branch is likely or unlikely.
/// Only use these macros if it causes the compiler to generate better code.
/// If you can remove a LIKELY/UNLIKELY annotation without speed changes in gcc
/// and clang, please do.
macro_rules! LIKELY {
($x:expr) => {
($x) as i64
};
}

macro_rules! MAX {
($a:expr, $b:expr) => {
if ($a) > $b { $a } else { $b }
};
}

macro_rules! MAX__4 {
($a:expr, $b:expr) => {
if ($a) > ((($b) as i32) as u32) { $a } else { (($b) as i32) as u32 }
};
}

macro_rules! MIN__3 {
($a:expr, $b:expr) => {
{
let __v: u64 = if ($a) < $b {
$a
} else {
$b
};
__v
}
};
}

macro_rules! MIN {
($a:expr, $b:expr) => {
if ($a) < $b { $a } else { $b }
};
}

macro_rules! PREFETCH_AREA {
($p:expr, $s:expr) => {
loop {
let _ptr: *const u8 = ($p) as *const u8;
let _size: u64 = ($s) as u64;
let mut _pos: u64 = 0;
while _pos < _size {
PREFETCH_L2!(_ptr.wrapping_offset(_pos as isize));
let __v: u64 = _pos;
_pos = __v.wrapping_add(64);
}
if true {
break;
}
}
};
}

/// rw==read
/// locality
macro_rules! PREFETCH_L1 {
($ptr:expr) => {
{
let _ = ($ptr) as *const ();
};
};
}

/// rw==read
/// locality
macro_rules! PREFETCH_L2 {
($ptr:expr) => {
{
let _ = ($ptr) as *const ();
};
};
}

macro_rules! SEQSYMBOL_TABLE_SIZE {
($log:expr) => {
1i32.wrapping_add(1i32 << $log)
};
}

macro_rules! UNLIKELY {
($x:expr) => {
($x) as i64
};
}

macro_rules! ZSTD_ERROR__6 {
() => {
30i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__2 {
() => {
20i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__14 {
() => {
24i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__4 {
() => {
1i32.wrapping_neg() as u64
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

/// ---- static assert (debug) ---
macro_rules! ZSTD_STATIC_ASSERT {
() => {
DEBUG_STATIC_ASSERT!();
};
}

macro_rules! ZSTD_memcpy__4 {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memcpy(($d) as *mut (), ($s) as *const (), ($n) as u64) };
};
}

macro_rules! ZSTD_memcpy {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memcpy(($d) as *mut (), ($s) as *const (), $n) };
};
}

macro_rules! ZSTD_memcpy__6 {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memcpy($d, $s, ($n) as u64) };
};
}

macro_rules! ZSTD_memmove__3 {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memmove($d, $s, ($n) as u64) };
};
}

macro_rules! ZSTD_memmove {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memmove(($d) as *mut (), ($s) as *const (), $n) };
};
}

macro_rules! ZSTD_memset {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memset(($d) as *mut (), $s, $n) };
};
}

macro_rules! ZSTD_memset__5 {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memset(($d) as *mut (), ($s) as i32, ($n) as u64) };
};
}

macro_rules! ZSTD_memset__3 {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memset(($d) as *mut (), ($s) as i32, $n) };
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
    fn FSE_readNCount(arg0: *mut i16, arg1: *mut u32, arg2: *mut u32, arg3: *const (), arg4: u64) -> u64;
    #[link_name = "memset"]
    fn __builtin_memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn HUF_decompress1X_usingDTable(arg0: *mut (), arg1: u64, arg2: *const (), arg3: u64, arg4: *const u32, arg5: i32) -> u64;
    fn HUF_decompress1X1_DCtx_wksp(arg0: *mut u32, arg1: *mut (), arg2: u64, arg3: *const (), arg4: u64, arg5: *mut (), arg6: u64, arg7: i32) -> u64;
    fn HUF_decompress4X_usingDTable(arg0: *mut (), arg1: u64, arg2: *const (), arg3: u64, arg4: *const u32, arg5: i32) -> u64;
    fn HUF_decompress4X_hufOnly_wksp(arg0: *mut u32, arg1: *mut (), arg2: u64, arg3: *const (), arg4: u64, arg5: *mut (), arg6: u64, arg7: i32) -> u64;
    #[link_name = "memmove"]
    fn __builtin_memmove(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn __builtin_prefetch(arg0: *const (), ...);
}

// FORCE_INLINE_TEMPLATE is used to define C "templates", which take constant
// parameters. They must be inlined for the compiler to eliminate the constant
// branches.
// force no inlining
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
/// Optional: only supported since gcc >= 15, clang >= 19
///
///   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
/// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
fn ZSTD_wrappedPtrAdd(ptr: *const u8, add: i64) -> *const u8 {
ptr.wrapping_offset(add as isize)
}

fn ZSTD_wrappedPtrSub(ptr: *const u8, sub: i64) -> *const u8 {
ptr.wrapping_offset(-(sub as isize))
}

fn ZSTD_maybeNullPtrAdd(ptr: *mut u8, add: i64) -> *mut u8 {
if add > 0 { ptr.wrapping_offset(add as isize) } else { ptr }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ZSTD_DDict_s {
}


fn get_unaligned_le16(p: *const ()) -> u16 {
__get_unaligned_t__3!(u16, p)
}

fn get_unaligned_le32(p: *const ()) -> u32 {
__get_unaligned_t!(u32, p)
}

fn get_unaligned_le64(p: *const ()) -> u64 {
__get_unaligned_t__5!(u64, p)
}

fn MEM_32bits() -> u32 {
((core::mem::size_of::<u64>() as u64) == 4) as u32
}

fn MEM_64bits() -> u32 {
((core::mem::size_of::<u64>() as u64) == 8) as u32
}

fn MEM_write64(memPtr: *mut (), value: u64) {
put_unaligned!(value, memPtr as *mut u64);
}

fn MEM_readLE16(memPtr: *const ()) -> u16 {
get_unaligned_le16(memPtr)
}

fn MEM_readLE24(memPtr: *const ()) -> u32 {
(MEM_readLE16(memPtr) as i32).wrapping_add(((unsafe { *(memPtr as *const u8).wrapping_offset(2 as isize) }) as i32) << 16i32) as u32
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

#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
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

fn BIT_endOfDStream(DStream: &__SlateRecord_bitstream_h_90) -> u32 {
((unsafe { (*(DStream as *const __SlateRecord_bitstream_h_90)).ptr }) == unsafe { (*(DStream as *const __SlateRecord_bitstream_h_90)).start } && ((unsafe { (*(DStream as *const __SlateRecord_bitstream_h_90)).bitsConsumed }) as u64) == (core::mem::size_of::<u64>() as u64).wrapping_mul(8)) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ZSTD_outBuffer_s {
    dst: *mut (),
    size: u64,
    pos: u64,
}


// can be useful for static allocation
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
struct __SlateRecord_zstd_lib_h_1837 {
    customAlloc: Option<unsafe extern "C" fn(*mut (), u64) -> *mut ()>,
    customFree: Option<unsafe extern "C" fn(*mut (), *mut ())>,
    opaque: *mut (),
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
// C standard doesn't allow `static const` variable to be init using another `static const` variable
static ZSTD_blockHeaderSize: u64 = 3;

// nbSeq==0
// litCSize
// RLE or RAW
// for a non-null block
// Assumption : MaxOff < MaxLL,MaxML
static LL_bits: __SlateAlign16<[u8; 36]> = __SlateAlign16([0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 1u8, 1u8, 1u8, 2u8, 2u8, 3u8, 3u8, 4u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8, 12u8, 13u8, 14u8, 15u8, 16u8]);

// for static allocation
static ML_bits: __SlateAlign16<[u8; 53]> = __SlateAlign16([0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 1u8, 1u8, 1u8, 2u8, 2u8, 3u8, 3u8, 4u8, 4u8, 5u8, 7u8, 8u8, 9u8, 10u8, 11u8, 12u8, 13u8, 14u8, 15u8, 16u8]);

// for static allocation
// for static allocation
fn ZSTD_copy8(dst: *mut (), src: *const ()) {
ZSTD_memcpy__6!(dst, src, 8i32);
}

fn ZSTD_copy16(dst: *mut (), src: *const ()) {
ZSTD_memmove__3!(dst, src, 16i32);
}

#[inline(always)]
fn ZSTD_wildcopy(dst: *mut (), src: *const (), length: i64, ovtype: u32) {
let diff: i64 = (unsafe { (dst as *mut u8).offset_from(src as *mut u8) }) as i64;
let mut ip: *const u8 = src as *const u8;
let mut op: *mut u8 = dst as *mut u8;
let oend: *mut u8 = op.wrapping_offset(length as isize);
if (ovtype as u32) == 1 && diff < 16 {
// Handle short offset copies.
loop {
COPY8__2!(op, ip);
if !(op < oend) {
break;
}
}
} else {
// Separate out the first COPY16() call because the copy length is
// almost certain to be short, so the branches have different
// probabilities. Since it is almost certain to be short, only do
// one COPY16() in the first call. Then, do two calls per loop since
// at that point it is more likely to have a high trip count.
ZSTD_copy16(op as *mut (), ip as *const ());
if 16 >= length {
return;
}
let __v6: *mut u8 = op;
op = __v6.wrapping_offset(16 as isize);
let __v8: *const u8 = ip;
ip = __v8.wrapping_offset(16 as isize);
loop {
COPY16!(op, ip);
COPY16!(op, ip);
if !(op < oend) {
break;
}
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_internal_h_295 {
    blockType: u32,
    lastBlock: u32,
    origSize: u32,
}


static LL_base: __SlateAlign16<[u32; 36]> = __SlateAlign16([0u32, 1u32, 2u32, 3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32, 10u32, 11u32, 12u32, 13u32, 14u32, 15u32, 16u32, 18u32, 20u32, 22u32, 24u32, 28u32, 32u32, 40u32, 48u32, 64u32, 128u32, 256u32, 512u32, 1024u32, 2048u32, 4096u32, 8192u32, 16384u32, 32768u32, 65536u32]);

static OF_base: __SlateAlign16<[u32; 32]> = __SlateAlign16([0u32, 1u32, 1u32, 5u32, 13u32, 29u32, 61u32, 125u32, 253u32, 509u32, 1021u32, 2045u32, 4093u32, 8189u32, 16381u32, 32765u32, 65533u32, 131069u32, 262141u32, 524285u32, 1048573u32, 2097149u32, 4194301u32, 8388605u32, 16777213u32, 33554429u32, 67108861u32, 134217725u32, 268435453u32, 536870909u32, 1073741821u32, 2147483645u32]);

static OF_bits: __SlateAlign16<[u8; 32]> = __SlateAlign16([0u8, 1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8, 12u8, 13u8, 14u8, 15u8, 16u8, 17u8, 18u8, 19u8, 20u8, 21u8, 22u8, 23u8, 24u8, 25u8, 26u8, 27u8, 28u8, 29u8, 30u8, 31u8]);

static ML_base: __SlateAlign16<[u32; 53]> = __SlateAlign16([3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32, 10u32, 11u32, 12u32, 13u32, 14u32, 15u32, 16u32, 17u32, 18u32, 19u32, 20u32, 21u32, 22u32, 23u32, 24u32, 25u32, 26u32, 27u32, 28u32, 29u32, 30u32, 31u32, 32u32, 33u32, 34u32, 35u32, 37u32, 39u32, 41u32, 43u32, 47u32, 51u32, 59u32, 67u32, 83u32, 99u32, 131u32, 259u32, 515u32, 1027u32, 2051u32, 4099u32, 8195u32, 16387u32, 32771u32, 65539u32]);

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_decompress_internal_h_66 {
    fastMode: u32,
    tableLog: u32,
}


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


fn ZSTD_DCtx_get_bmi2(dctx: *const ZSTD_DCtx_s) -> i32 {
unsafe { (*dctx).bmi2 }
}

// zstd_decompress_block :
// this module takes care of decompressing _compressed_ block
// Dependencies
// ZSTD_memcpy, ZSTD_memmove, ZSTD_memset
// prefetch
// bmi2
// low level memory routines
// ZSTD_DCtx
// ZSTD_DDictDictContent
// ZSTD_highbit32
// Macros
/// These two optional macros force the use one way or another of the two
/// ZSTD_decompressSequences implementations. You can't force in both directions
/// at the same time.
/// Memory operations
fn ZSTD_copy4(dst: *mut (), src: *const ()) {
ZSTD_memcpy__6!(dst, src, 4i32);
}

// Block decoding
fn ZSTD_blockSizeMax(dctx: *const ZSTD_DCtx_s) -> u64 {
let blockSizeMax: u64 = (if (unsafe { (*dctx).isFrameDecompression }) != 0 { unsafe { (*dctx).fParams.blockSizeMax } } else { 131072u32 }) as u64;
blockSizeMax
}

/// ZSTD_getcBlockSize() :
/// Provides the size of compressed block from block header `src`
#[unsafe(no_mangle)]
extern "C" fn ZSTD_getcBlockSize(src: *const (), srcSize: u64, bpPtr: *mut __SlateRecord_zstd_internal_h_295) -> u64 {
loop {
if srcSize < unsafe { ZSTD_blockHeaderSize } {
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
let cBlockHeader: u32 = MEM_readLE24(src);
let cSize: u32 = cBlockHeader >> 3i32;
unsafe {
(*bpPtr).lastBlock = cBlockHeader & 1;
}
unsafe {
(*bpPtr).blockType = (cBlockHeader >> 1i32 & 3) as u32;
}
unsafe {
(*bpPtr).origSize = cSize;
}
// only useful for RLE
if ((unsafe { (*bpPtr).blockType }) as u32) == 1 {
return 1;
}
loop {
if ((unsafe { (*bpPtr).blockType }) as u32) == 3 {
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
cSize as u64
}

/// Allocate buffer for literals, either overlapping current dst, or split between dst and litExtraBuffer, or stored entirely within litExtraBuffer
fn ZSTD_allocateLiteralsBuffer(dctx: *mut ZSTD_DCtx_s, dst: *mut (), dstCapacity: u64, litSize: u64, streaming: u32, expectedWriteSize: u64, splitImmediately: u32) {
let blockSizeMax: u64 = ZSTD_blockSizeMax(dctx as *const ZSTD_DCtx_s);
if (streaming as u32) == 0 && dstCapacity > blockSizeMax.wrapping_add(32).wrapping_add(litSize).wrapping_add(32) {
// If we aren't streaming, we can just put the literals after the output
// of the current block. We don't need to worry about overwriting the
// extDict of our window, because it doesn't exist.
// So if we have space after the end of the block, just put it there.
unsafe {
(*dctx).litBuffer = (dst as *mut u8).wrapping_offset(blockSizeMax as isize).wrapping_offset(32 as isize);
}
unsafe {
(*dctx).litBufferEnd = unsafe { (*dctx).litBuffer }.wrapping_offset(litSize as isize) as *const u8;
}
unsafe {
(*dctx).litBufferLocation = 1;
}
} else {
if litSize <= (BOUNDED!(64i32, 65536i32, 131072i32) as u64) {
// Literals fit entirely within the extra buffer, put them there to avoid
// having to split the literals.
unsafe {
(*dctx).litBuffer = (*dctx).litExtraBuffer.as_mut_ptr() as *mut u8;
}
unsafe {
(*dctx).litBufferEnd = unsafe { (*dctx).litBuffer }.wrapping_offset(litSize as isize) as *const u8;
}
unsafe {
(*dctx).litBufferLocation = 0;
}
} else {
// Literals must be split between the output block and the extra lit
// buffer. We fill the extra lit buffer with the tail of the literals,
// and put the rest of the literals at the end of the block, with
// WILDCOPY_OVERLENGTH of buffer room to allow for overreads.
// This MUST not write more than our maxBlockSize beyond dst, because in
// streaming mode, that could overwrite part of our extDict window.
if splitImmediately != 0 {
// won't fit in litExtraBuffer, so it will be split between end of dst and extra buffer
unsafe {
(*dctx).litBuffer = (dst as *mut u8).wrapping_offset(expectedWriteSize as isize).wrapping_offset(-(litSize as isize)).wrapping_offset(BOUNDED!(64i32, 65536i32, 131072i32) as isize).wrapping_offset(-(32 as isize));
}
unsafe {
(*dctx).litBufferEnd = unsafe { (*dctx).litBuffer }.wrapping_offset(litSize as isize).wrapping_offset(-(BOUNDED!(64i32, 65536i32, 131072i32) as isize)) as *const u8;
}
} else {
// initially this will be stored entirely in dst during huffman decoding, it will partially be shifted to litExtraBuffer after
unsafe {
(*dctx).litBuffer = (dst as *mut u8).wrapping_offset(expectedWriteSize as isize).wrapping_offset(-(litSize as isize));
}
unsafe {
(*dctx).litBufferEnd = (dst as *mut u8).wrapping_offset(expectedWriteSize as isize) as *const u8;
}
}
unsafe {
(*dctx).litBufferLocation = 2;
}
}
}
}

// ZSTD_decodeLiteralsBlock() :
// Where it is possible to do so without being stomped by the output during decompression, the literals block will be stored
// in the dstBuffer.  If there is room to do so, it will be stored in full in the excess dst space after where the current
// block will be output.  Otherwise it will be stored at the end of the current dst blockspace, with a small portion being
// stored in dctx->litExtraBuffer to help keep it "ahead" of the current output write.
//
// @return : nb of bytes read from src (< srcSize )
//  note : symbol not declared but exposed for fullbench
/// # Arguments
///
/// * `srcSize` - note : srcSize < BLOCKSIZE
fn ZSTD_decodeLiteralsBlock(dctx: *mut ZSTD_DCtx_s, src: *const (), srcSize: u64, dst: *mut (), dstCapacity: u64, streaming: u32) -> u64 {
while false {
}
loop {
if srcSize < 2u64 {
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
let istart: *const u8 = src as *const u8;
let litEncType: u32 = (((unsafe { *istart.wrapping_offset(0 as isize) }) as i32) & 3) as u32;
let blockSizeMax: u64 = ZSTD_blockSizeMax(dctx as *const ZSTD_DCtx_s);
// 2 - 2 - 14 - 14
// 2 - 2 - 18 - 18
// note : default is impossible, since lhlCode into [0..3]
// 2 - 2 - 10 - 10
// prefetch huffman table if cold
// heuristic
// note : default is impossible, since lhlCode into [0..3]
// risk reading beyond src buffer with wildcopy
// direct reference into compressed stream
// note : default is impossible, since lhlCode into [0..3]
match litEncType as u32 {
3 => {
while false {
}
loop {
if (unsafe { (*dctx).litEntropy }) == 0 {
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
if srcSize < 5 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"srcSize >= MIN_CBLOCK_SIZE == 2; here we need up to 5 for case 3".as_ptr() as *const u8) };
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
let mut lhSize: u64 = 0;
let mut litSize: u64 = 0;
let mut litCSize: u64 = 0;
let mut singleStream: u32 = 0;
let mut lhlCode: u32 = (((unsafe { *istart.wrapping_offset(0 as isize) }) as i32) >> 2i32 & 3) as u32;
let mut lhc: u32 = MEM_readLE32(istart as *const ());
let mut hufSuccess: u64 = 0;
let mut expectedWriteSize: u64 = MIN!(blockSizeMax, dstCapacity);
let mut flags: i32 = 0 | if ZSTD_DCtx_get_bmi2(dctx as *const ZSTD_DCtx_s) != 0 { 1 } else { 0 } | if (unsafe { (*dctx).disableHufAsm }) != 0 { 16 } else { 0 };
match lhlCode {
2 => {
lhSize = 4; // 2 - 2 - 14 - 14
litSize = (lhc >> 4i32 & 16383) as u64;
litCSize = (lhc >> 18i32) as u64;
}
3 => {
lhSize = 5; // 2 - 2 - 18 - 18
litSize = (lhc >> 4i32 & 262143) as u64;
litCSize = ((lhc >> 22i32) as u64).wrapping_add(((unsafe { *istart.wrapping_offset(4 as isize) }) as u64) << 10i32);
}
_ => {
singleStream = (lhlCode == 0) as u32; // note : default is impossible, since lhlCode into [0..3]
// 2 - 2 - 10 - 10
lhSize = 3;
litSize = (lhc >> 4i32 & 1023) as u64;
litCSize = (lhc >> 14i32 & 1023) as u64;
}
}
loop {
if litSize > 0 && dst.is_null() {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"NULL not handled".as_ptr() as *const u8) };
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
loop {
if litSize > blockSizeMax {
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
if singleStream == 0 {
loop {
if litSize < 6 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"Not enough literals (%zu) for the 4-streams mode (min %u)".as_ptr() as *const u8, litSize, 6i32) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__17!();
}
if true {
break;
}
}
}
loop {
if litCSize.wrapping_add(lhSize) > srcSize {
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
loop {
if expectedWriteSize < litSize {
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
ZSTD_allocateLiteralsBuffer(dctx, dst, dstCapacity, litSize, streaming, expectedWriteSize, 0u32);
// prefetch huffman table if cold
if (unsafe { (*dctx).ddictIsCold }) != 0 && litSize > 768 {
// heuristic
PREFETCH_AREA!(unsafe { (*dctx).HUFptr }, core::mem::size_of::<[u32; 4097]>() as u64);
}
if (litEncType as u32) == 3 {
if singleStream != 0 {
hufSuccess = unsafe { HUF_decompress1X_usingDTable((*dctx).litBuffer as *mut (), litSize, istart.wrapping_offset(lhSize as isize) as *const (), litCSize, (*dctx).HUFptr, flags) };
} else {
hufSuccess = unsafe { HUF_decompress4X_usingDTable((*dctx).litBuffer as *mut (), litSize, istart.wrapping_offset(lhSize as isize) as *const (), litCSize, (*dctx).HUFptr, flags) };
}
} else {
if singleStream != 0 {
hufSuccess = unsafe { HUF_decompress1X1_DCtx_wksp((*dctx).entropy.hufTable.as_mut_ptr() as *mut u32, (*dctx).litBuffer as *mut (), litSize, istart.wrapping_offset(lhSize as isize) as *const (), litCSize, ((*dctx).workspace.as_mut_ptr() as *mut u32) as *mut (), core::mem::size_of::<[u32; 640]>() as u64, flags) };
} else {
hufSuccess = unsafe { HUF_decompress4X_hufOnly_wksp((*dctx).entropy.hufTable.as_mut_ptr() as *mut u32, (*dctx).litBuffer as *mut (), litSize, istart.wrapping_offset(lhSize as isize) as *const (), litCSize, ((*dctx).workspace.as_mut_ptr() as *mut u32) as *mut (), core::mem::size_of::<[u32; 640]>() as u64, flags) };
}
}
if ((unsafe { (*dctx).litBufferLocation }) as u32) == 2 {
ZSTD_memcpy__4!(unsafe { (*dctx).litExtraBuffer.as_mut_ptr() as *mut u8 }, unsafe { (*dctx).litBufferEnd }.wrapping_offset(-(BOUNDED!(64i32, 65536i32, 131072i32) as isize)), BOUNDED!(64i32, 65536i32, 131072i32));
ZSTD_memmove!(unsafe { (*dctx).litBuffer }.wrapping_offset(BOUNDED!(64i32, 65536i32, 131072i32) as isize).wrapping_offset(-(32 as isize)), unsafe { (*dctx).litBuffer }, litSize.wrapping_sub(BOUNDED!(64i32, 65536i32, 131072i32) as u64));
let _v55: *mut u8 = unsafe { (*dctx).litBuffer };
let _v56: *mut u8 = _v55.wrapping_offset(BOUNDED!(64i32, 65536i32, 131072i32).wrapping_sub(32) as isize);
unsafe {
(*dctx).litBuffer = _v56;
}
let _v58: *const u8 = unsafe { (*dctx).litBufferEnd };
let _v59: *const u8 = _v58.wrapping_offset(-(32 as isize));
unsafe {
(*dctx).litBufferEnd = _v59;
}
}
loop {
if ERR_isError(hufSuccess) != 0 {
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
unsafe {
(*dctx).litPtr = (*dctx).litBuffer as *const u8;
}
unsafe {
(*dctx).litSize = litSize;
}
unsafe {
(*dctx).litEntropy = 1;
}
if (litEncType as u32) == 2 {
unsafe {
(*dctx).HUFptr = ((*dctx).entropy.hufTable.as_mut_ptr() as *mut u32) as *const u32;
}
}
return litCSize.wrapping_add(lhSize);
}
2 => {
loop {
if srcSize < 5 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"srcSize >= MIN_CBLOCK_SIZE == 2; here we need up to 5 for case 3".as_ptr() as *const u8) };
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
let mut lhSize: u64 = 0;
let mut litSize: u64 = 0;
let mut litCSize: u64 = 0;
let mut singleStream: u32 = 0;
let lhlCode: u32 = (((unsafe { *istart.wrapping_offset(0 as isize) }) as i32) >> 2i32 & 3) as u32;
let lhc: u32 = MEM_readLE32(istart as *const ());
let mut hufSuccess: u64 = 0;
let expectedWriteSize: u64 = MIN!(blockSizeMax, dstCapacity);
let flags: i32 = 0 | if ZSTD_DCtx_get_bmi2(dctx as *const ZSTD_DCtx_s) != 0 { 1 } else { 0 } | if (unsafe { (*dctx).disableHufAsm }) != 0 { 16 } else { 0 };
match lhlCode {
2 => {
lhSize = 4; // 2 - 2 - 14 - 14
litSize = (lhc >> 4i32 & 16383) as u64;
litCSize = (lhc >> 18i32) as u64;
}
3 => {
lhSize = 5; // 2 - 2 - 18 - 18
litSize = (lhc >> 4i32 & 262143) as u64;
litCSize = ((lhc >> 22i32) as u64).wrapping_add(((unsafe { *istart.wrapping_offset(4 as isize) }) as u64) << 10i32);
}
_ => {
singleStream = (lhlCode == 0) as u32; // note : default is impossible, since lhlCode into [0..3]
// 2 - 2 - 10 - 10
lhSize = 3;
litSize = (lhc >> 4i32 & 1023) as u64;
litCSize = (lhc >> 14i32 & 1023) as u64;
}
}
loop {
if litSize > 0 && dst.is_null() {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"NULL not handled".as_ptr() as *const u8) };
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
loop {
if litSize > blockSizeMax {
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
if singleStream == 0 {
loop {
if litSize < 6 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"Not enough literals (%zu) for the 4-streams mode (min %u)".as_ptr() as *const u8, litSize, 6i32) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__17!();
}
if true {
break;
}
}
}
loop {
if litCSize.wrapping_add(lhSize) > srcSize {
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
loop {
if expectedWriteSize < litSize {
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
ZSTD_allocateLiteralsBuffer(dctx, dst, dstCapacity, litSize, streaming, expectedWriteSize, 0u32);
// prefetch huffman table if cold
if (unsafe { (*dctx).ddictIsCold }) != 0 && litSize > 768 {
// heuristic
PREFETCH_AREA!(unsafe { (*dctx).HUFptr }, core::mem::size_of::<[u32; 4097]>() as u64);
}
if (litEncType as u32) == 3 {
if singleStream != 0 {
hufSuccess = unsafe { HUF_decompress1X_usingDTable((*dctx).litBuffer as *mut (), litSize, istart.wrapping_offset(lhSize as isize) as *const (), litCSize, (*dctx).HUFptr, flags) };
} else {
hufSuccess = unsafe { HUF_decompress4X_usingDTable((*dctx).litBuffer as *mut (), litSize, istart.wrapping_offset(lhSize as isize) as *const (), litCSize, (*dctx).HUFptr, flags) };
}
} else {
if singleStream != 0 {
hufSuccess = unsafe { HUF_decompress1X1_DCtx_wksp((*dctx).entropy.hufTable.as_mut_ptr() as *mut u32, (*dctx).litBuffer as *mut (), litSize, istart.wrapping_offset(lhSize as isize) as *const (), litCSize, ((*dctx).workspace.as_mut_ptr() as *mut u32) as *mut (), core::mem::size_of::<[u32; 640]>() as u64, flags) };
} else {
hufSuccess = unsafe { HUF_decompress4X_hufOnly_wksp((*dctx).entropy.hufTable.as_mut_ptr() as *mut u32, (*dctx).litBuffer as *mut (), litSize, istart.wrapping_offset(lhSize as isize) as *const (), litCSize, ((*dctx).workspace.as_mut_ptr() as *mut u32) as *mut (), core::mem::size_of::<[u32; 640]>() as u64, flags) };
}
}
if ((unsafe { (*dctx).litBufferLocation }) as u32) == 2 {
ZSTD_memcpy__4!(unsafe { (*dctx).litExtraBuffer.as_mut_ptr() as *mut u8 }, unsafe { (*dctx).litBufferEnd }.wrapping_offset(-(BOUNDED!(64i32, 65536i32, 131072i32) as isize)), BOUNDED!(64i32, 65536i32, 131072i32));
ZSTD_memmove!(unsafe { (*dctx).litBuffer }.wrapping_offset(BOUNDED!(64i32, 65536i32, 131072i32) as isize).wrapping_offset(-(32 as isize)), unsafe { (*dctx).litBuffer }, litSize.wrapping_sub(BOUNDED!(64i32, 65536i32, 131072i32) as u64));
let __v49: *mut u8 = unsafe { (*dctx).litBuffer };
let __v50: *mut u8 = __v49.wrapping_offset(BOUNDED!(64i32, 65536i32, 131072i32).wrapping_sub(32) as isize);
unsafe {
(*dctx).litBuffer = __v50;
}
let __v52: *const u8 = unsafe { (*dctx).litBufferEnd };
unsafe {
(*dctx).litBufferEnd = __v52.wrapping_offset(-(32 as isize));
}
}
loop {
if ERR_isError(hufSuccess) != 0 {
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
unsafe {
(*dctx).litPtr = (*dctx).litBuffer as *const u8;
}
unsafe {
(*dctx).litSize = litSize;
}
unsafe {
(*dctx).litEntropy = 1;
}
if (litEncType as u32) == 2 {
unsafe {
(*dctx).HUFptr = ((*dctx).entropy.hufTable.as_mut_ptr() as *mut u32) as *const u32;
}
}
return litCSize.wrapping_add(lhSize);
}
0 => {
let mut litSize: u64 = 0;
let mut lhSize: u64 = 0;
let lhlCode: u32 = (((unsafe { *istart.wrapping_offset(0 as isize) }) as i32) >> 2i32 & 3) as u32;
let expectedWriteSize: u64 = MIN!(blockSizeMax, dstCapacity);
match lhlCode {
1 => {
lhSize = 2;
litSize = ((MEM_readLE16(istart as *const ()) as i32) >> 4i32) as u64;
}
3 => {
lhSize = 3;
loop {
if srcSize < 3 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"srcSize >= MIN_CBLOCK_SIZE == 2; here we need lhSize = 3".as_ptr() as *const u8) };
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
litSize = (MEM_readLE24(istart as *const ()) >> 4i32) as u64;
}
_ => {
lhSize = 1; // note : default is impossible, since lhlCode into [0..3]
litSize = (((unsafe { *istart.wrapping_offset(0 as isize) }) as i32) >> 3i32) as u64;
}
}
loop {
if litSize > 0 && dst.is_null() {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"NULL not handled".as_ptr() as *const u8) };
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
loop {
if litSize > blockSizeMax {
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
loop {
if expectedWriteSize < litSize {
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
ZSTD_allocateLiteralsBuffer(dctx, dst, dstCapacity, litSize, streaming, expectedWriteSize, 1u32);
if lhSize.wrapping_add(litSize).wrapping_add(32) > srcSize {
// risk reading beyond src buffer with wildcopy
loop {
if litSize.wrapping_add(lhSize) > srcSize {
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
if ((unsafe { (*dctx).litBufferLocation }) as u32) == 2 {
ZSTD_memcpy!(unsafe { (*dctx).litBuffer }, istart.wrapping_offset(lhSize as isize), litSize.wrapping_sub(BOUNDED!(64i32, 65536i32, 131072i32) as u64));
ZSTD_memcpy__4!(unsafe { (*dctx).litExtraBuffer.as_mut_ptr() as *mut u8 }, istart.wrapping_offset(lhSize as isize).wrapping_offset(litSize as isize).wrapping_offset(-(BOUNDED!(64i32, 65536i32, 131072i32) as isize)), BOUNDED!(64i32, 65536i32, 131072i32));
} else {
ZSTD_memcpy!(unsafe { (*dctx).litBuffer }, istart.wrapping_offset(lhSize as isize), litSize);
}
unsafe {
(*dctx).litPtr = (*dctx).litBuffer as *const u8;
}
unsafe {
(*dctx).litSize = litSize;
}
return lhSize.wrapping_add(litSize);
}
// direct reference into compressed stream
unsafe {
(*dctx).litPtr = istart.wrapping_offset(lhSize as isize);
}
unsafe {
(*dctx).litSize = litSize;
}
unsafe {
(*dctx).litBufferEnd = unsafe { (*dctx).litPtr }.wrapping_offset(litSize as isize);
}
unsafe {
(*dctx).litBufferLocation = 0;
}
return lhSize.wrapping_add(litSize);
}
1 => {
let lhlCode: u32 = (((unsafe { *istart.wrapping_offset(0 as isize) }) as i32) >> 2i32 & 3) as u32;
let mut litSize: u64 = 0;
let mut lhSize: u64 = 0;
let expectedWriteSize: u64 = MIN!(blockSizeMax, dstCapacity);
match lhlCode {
1 => {
lhSize = 2;
loop {
if srcSize < 3 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"srcSize >= MIN_CBLOCK_SIZE == 2; here we need lhSize+1 = 3".as_ptr() as *const u8) };
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
litSize = ((MEM_readLE16(istart as *const ()) as i32) >> 4i32) as u64;
}
3 => {
lhSize = 3;
loop {
if srcSize < 4 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"srcSize >= MIN_CBLOCK_SIZE == 2; here we need lhSize+1 = 4".as_ptr() as *const u8) };
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
litSize = (MEM_readLE24(istart as *const ()) >> 4i32) as u64;
}
_ => {
lhSize = 1; // note : default is impossible, since lhlCode into [0..3]
litSize = (((unsafe { *istart.wrapping_offset(0 as isize) }) as i32) >> 3i32) as u64;
}
}
loop {
if litSize > 0 && dst.is_null() {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"NULL not handled".as_ptr() as *const u8) };
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
loop {
if litSize > blockSizeMax {
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
loop {
if expectedWriteSize < litSize {
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
ZSTD_allocateLiteralsBuffer(dctx, dst, dstCapacity, litSize, streaming, expectedWriteSize, 1u32);
if ((unsafe { (*dctx).litBufferLocation }) as u32) == 2 {
ZSTD_memset__3!(unsafe { (*dctx).litBuffer }, unsafe { *istart.wrapping_offset(lhSize as isize) }, litSize.wrapping_sub(BOUNDED!(64i32, 65536i32, 131072i32) as u64));
ZSTD_memset__5!(unsafe { (*dctx).litExtraBuffer.as_mut_ptr() as *mut u8 }, unsafe { *istart.wrapping_offset(lhSize as isize) }, BOUNDED!(64i32, 65536i32, 131072i32));
} else {
ZSTD_memset__3!(unsafe { (*dctx).litBuffer }, unsafe { *istart.wrapping_offset(lhSize as isize) }, litSize);
}
unsafe {
(*dctx).litPtr = (*dctx).litBuffer as *const u8;
}
unsafe {
(*dctx).litSize = litSize;
}
return lhSize.wrapping_add(1);
}
_ => {
loop {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"impossible".as_ptr() as *const u8) };
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
unsafe { core::mem::zeroed() }
}

/// Hidden declaration for fullbench
#[unsafe(no_mangle)]
extern "C" fn ZSTD_decodeLiteralsBlock_wrapper(dctx: *mut ZSTD_DCtx_s, src: *const (), srcSize: u64, dst: *mut (), dstCapacity: u64) -> u64 {
unsafe {
(*dctx).isFrameDecompression = 0;
}
ZSTD_decodeLiteralsBlock(dctx, src, srcSize, dst, dstCapacity, 0u32)
}

// Default FSE distribution tables.
// These are pre-calculated FSE decoding tables using default distributions as defined in specification :
// https://github.com/facebook/zstd/blob/release/doc/zstd_compression_format.md#default-distributions
// They were generated programmatically with following method :
// - start from default distributions, present in /lib/common/zstd_internal.h
// - generate tables normally, using ZSTD_buildFSETable()
// - printout the content of tables
// - prettify output, report below, test with fuzzer to ensure it's correct
/// Default FSE distribution table for Literal Lengths
/// header : fastMode, tableLog
/// nextState, nbAddBits, nbBits, baseVal
/// LL_defaultDTable
static LL_defaultDTable: __SlateAlign16<[__SlateRecord_zstd_decompress_internal_h_73; 65]> = __SlateAlign16([__SlateRecord_zstd_decompress_internal_h_73 { nextState: 1u16, nbAdditionalBits: 1u8, nbBits: 1u8, baseValue: 6u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 4u8, baseValue: 0u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 16u16, nbAdditionalBits: 0u8, nbBits: 4u8, baseValue: 0u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 1u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 3u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 4u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 6u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 7u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 9u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 10u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 12u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 14u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 1u8, nbBits: 5u8, baseValue: 16u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 1u8, nbBits: 5u8, baseValue: 20u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 1u8, nbBits: 5u8, baseValue: 22u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 2u8, nbBits: 5u8, baseValue: 28u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 3u8, nbBits: 5u8, baseValue: 32u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 4u8, nbBits: 5u8, baseValue: 48u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 6u8, nbBits: 5u8, baseValue: 64u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 7u8, nbBits: 5u8, baseValue: 128u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 8u8, nbBits: 6u8, baseValue: 256u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 10u8, nbBits: 6u8, baseValue: 1024u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 12u8, nbBits: 6u8, baseValue: 4096u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 4u8, baseValue: 0u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 4u8, baseValue: 1u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 2u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 4u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 5u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 7u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 8u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 10u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 11u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 13u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 1u8, nbBits: 5u8, baseValue: 16u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 1u8, nbBits: 5u8, baseValue: 18u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 1u8, nbBits: 5u8, baseValue: 22u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 2u8, nbBits: 5u8, baseValue: 24u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 3u8, nbBits: 5u8, baseValue: 32u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 3u8, nbBits: 5u8, baseValue: 40u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 6u8, nbBits: 4u8, baseValue: 64u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 16u16, nbAdditionalBits: 6u8, nbBits: 4u8, baseValue: 64u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 7u8, nbBits: 5u8, baseValue: 128u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 9u8, nbBits: 6u8, baseValue: 512u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 11u8, nbBits: 6u8, baseValue: 2048u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 48u16, nbAdditionalBits: 0u8, nbBits: 4u8, baseValue: 0u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 16u16, nbAdditionalBits: 0u8, nbBits: 4u8, baseValue: 1u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 2u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 3u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 5u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 6u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 8u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 9u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 11u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 12u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 15u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 1u8, nbBits: 5u8, baseValue: 18u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 1u8, nbBits: 5u8, baseValue: 20u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 2u8, nbBits: 5u8, baseValue: 24u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 2u8, nbBits: 5u8, baseValue: 28u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 3u8, nbBits: 5u8, baseValue: 40u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 4u8, nbBits: 5u8, baseValue: 48u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 16u8, nbBits: 6u8, baseValue: 65536u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 15u8, nbBits: 6u8, baseValue: 32768u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 14u8, nbBits: 6u8, baseValue: 16384u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 13u8, nbBits: 6u8, baseValue: 8192u32 }]);

/// Default FSE distribution table for Offset Codes
/// header : fastMode, tableLog
/// nextState, nbAddBits, nbBits, baseVal
/// OF_defaultDTable
static OF_defaultDTable: __SlateAlign16<[__SlateRecord_zstd_decompress_internal_h_73; 33]> = __SlateAlign16([__SlateRecord_zstd_decompress_internal_h_73 { nextState: 1u16, nbAdditionalBits: 1u8, nbBits: 1u8, baseValue: 5u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 0u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 6u8, nbBits: 4u8, baseValue: 61u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 9u8, nbBits: 5u8, baseValue: 509u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 15u8, nbBits: 5u8, baseValue: 32765u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 21u8, nbBits: 5u8, baseValue: 2097149u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 3u8, nbBits: 5u8, baseValue: 5u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 7u8, nbBits: 4u8, baseValue: 125u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 12u8, nbBits: 5u8, baseValue: 4093u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 18u8, nbBits: 5u8, baseValue: 262141u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 23u8, nbBits: 5u8, baseValue: 8388605u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 5u8, nbBits: 5u8, baseValue: 29u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 8u8, nbBits: 4u8, baseValue: 253u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 14u8, nbBits: 5u8, baseValue: 16381u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 20u8, nbBits: 5u8, baseValue: 1048573u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 2u8, nbBits: 5u8, baseValue: 1u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 16u16, nbAdditionalBits: 7u8, nbBits: 4u8, baseValue: 125u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 11u8, nbBits: 5u8, baseValue: 2045u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 17u8, nbBits: 5u8, baseValue: 131069u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 22u8, nbBits: 5u8, baseValue: 4194301u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 4u8, nbBits: 5u8, baseValue: 13u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 16u16, nbAdditionalBits: 8u8, nbBits: 4u8, baseValue: 253u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 13u8, nbBits: 5u8, baseValue: 8189u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 19u8, nbBits: 5u8, baseValue: 524285u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 1u8, nbBits: 5u8, baseValue: 1u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 16u16, nbAdditionalBits: 6u8, nbBits: 4u8, baseValue: 61u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 10u8, nbBits: 5u8, baseValue: 1021u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 16u8, nbBits: 5u8, baseValue: 65533u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 28u8, nbBits: 5u8, baseValue: 268435453u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 27u8, nbBits: 5u8, baseValue: 134217725u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 26u8, nbBits: 5u8, baseValue: 67108861u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 25u8, nbBits: 5u8, baseValue: 33554429u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 24u8, nbBits: 5u8, baseValue: 16777213u32 }]);

/// Default FSE distribution table for Match Lengths
/// header : fastMode, tableLog
/// nextState, nbAddBits, nbBits, baseVal
/// ML_defaultDTable
static ML_defaultDTable: __SlateAlign16<[__SlateRecord_zstd_decompress_internal_h_73; 65]> = __SlateAlign16([__SlateRecord_zstd_decompress_internal_h_73 { nextState: 1u16, nbAdditionalBits: 1u8, nbBits: 1u8, baseValue: 6u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 3u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 4u8, baseValue: 4u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 5u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 6u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 8u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 9u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 11u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 13u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 16u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 19u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 22u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 25u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 28u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 31u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 34u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 1u8, nbBits: 6u8, baseValue: 37u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 1u8, nbBits: 6u8, baseValue: 41u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 2u8, nbBits: 6u8, baseValue: 47u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 3u8, nbBits: 6u8, baseValue: 59u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 4u8, nbBits: 6u8, baseValue: 83u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 7u8, nbBits: 6u8, baseValue: 131u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 9u8, nbBits: 6u8, baseValue: 515u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 16u16, nbAdditionalBits: 0u8, nbBits: 4u8, baseValue: 4u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 4u8, baseValue: 5u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 6u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 7u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 9u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 10u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 12u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 15u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 18u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 21u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 24u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 27u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 30u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 33u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 1u8, nbBits: 6u8, baseValue: 35u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 1u8, nbBits: 6u8, baseValue: 39u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 2u8, nbBits: 6u8, baseValue: 43u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 3u8, nbBits: 6u8, baseValue: 51u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 4u8, nbBits: 6u8, baseValue: 67u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 5u8, nbBits: 6u8, baseValue: 99u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 8u8, nbBits: 6u8, baseValue: 259u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 4u8, baseValue: 4u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 48u16, nbAdditionalBits: 0u8, nbBits: 4u8, baseValue: 4u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 16u16, nbAdditionalBits: 0u8, nbBits: 4u8, baseValue: 5u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 7u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 8u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 10u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 32u16, nbAdditionalBits: 0u8, nbBits: 5u8, baseValue: 11u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 14u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 17u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 20u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 23u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 26u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 29u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 0u8, nbBits: 6u8, baseValue: 32u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 16u8, nbBits: 6u8, baseValue: 65539u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 15u8, nbBits: 6u8, baseValue: 32771u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 14u8, nbBits: 6u8, baseValue: 16387u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 13u8, nbBits: 6u8, baseValue: 8195u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 12u8, nbBits: 6u8, baseValue: 4099u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 11u8, nbBits: 6u8, baseValue: 2051u32 }, __SlateRecord_zstd_decompress_internal_h_73 { nextState: 0u16, nbAdditionalBits: 10u8, nbBits: 6u8, baseValue: 1027u32 }]);

fn ZSTD_buildSeqTable_rle(dt: *mut __SlateRecord_zstd_decompress_internal_h_73, baseValue: u32, nbAddBits: u8) {
let ptr: *mut () = dt as *mut ();
let cell: *mut __SlateRecord_zstd_decompress_internal_h_73 = dt.wrapping_offset(1 as isize);
unsafe {
(*(ptr as *mut __SlateRecord_zstd_decompress_internal_h_66)).tableLog = 0;
}
unsafe {
(*(ptr as *mut __SlateRecord_zstd_decompress_internal_h_66)).fastMode = 0;
}
unsafe {
(*cell).nbBits = 0;
}
unsafe {
(*cell).nextState = 0;
}
unsafe {
(*cell).nbAdditionalBits = nbAddBits;
}
unsafe {
(*cell).baseValue = baseValue;
}
}

/// ZSTD_buildFSETable() :
/// generate FSE decoding table for one symbol (ll, ml or off)
/// cannot fail if input is valid =>
/// all inputs are presumed validated at this stage
#[inline(always)]
fn ZSTD_buildFSETable_body(dt: *mut __SlateRecord_zstd_decompress_internal_h_73, normalizedCounter: *const i16, maxSymbolValue: u32, baseValue: *const u32, nbAdditionalBits: *const u8, tableLog: u32, wksp: *mut (), wkspSize: u64) {
let tableDecode: *mut __SlateRecord_zstd_decompress_internal_h_73 = dt.wrapping_offset(1 as isize);
let maxSV1: u32 = maxSymbolValue.wrapping_add(1);
let tableSize: u32 = (1i32 << tableLog) as u32;
let symbolNext: *mut u16 = wksp as *mut u16;
let spread: *mut u8 = symbolNext.wrapping_offset(MAX!(35i32, 52i32) as isize).wrapping_offset(1 as isize) as *mut u8;
let mut highThreshold: u32 = tableSize.wrapping_sub(1);
// Sanity Checks
// Init, lay down lowprob symbols
let mut DTableH: __SlateRecord_zstd_decompress_internal_h_66 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(DTableH) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
DTableH.tableLog = tableLog;
DTableH.fastMode = 1;
let largeLimit: i16 = (1i32 << tableLog.wrapping_sub(1)) as i16;
for s in 0..maxSV1 {
if ((unsafe { *normalizedCounter.wrapping_offset(s as isize) }) as i32) == 1i32.wrapping_neg() {
let __v0: u32 = highThreshold;
highThreshold = __v0.wrapping_sub(1);
unsafe {
(*tableDecode.wrapping_offset(__v0 as isize)).baseValue = s;
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
ZSTD_memcpy!(dt, core::ptr::addr_of_mut!(DTableH), core::mem::size_of::<__SlateRecord_zstd_decompress_internal_h_66>() as u64);
// Spread symbols
// Specialized symbol spreading for the case when there are
// no low probability (-1 count) symbols. When compressing
// small blocks we avoid low probability symbols to hit this
// case, since header decoding speed matters more.
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
(*tableDecode.wrapping_offset(uPosition as isize)).baseValue = *spread.wrapping_offset(s.wrapping_add(u) as isize) as u32;
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
let n: i32 = (unsafe { *normalizedCounter.wrapping_offset(s as isize) }) as i32;
for i in 0..n {
unsafe {
(*tableDecode.wrapping_offset(position as isize)).baseValue = s;
}
position = position.wrapping_add(step) & tableMask;
while UNLIKELY!(position > highThreshold) != 0 {
position = position.wrapping_add(step) & tableMask;
}
// lowprob area
}
}
// position must reach all cells once, otherwise normalizedCounter is incorrect
}
// Build Decoding table
for u in 0..tableSize {
let symbol: u32 = unsafe { (*tableDecode.wrapping_offset(u as isize)).baseValue };
let mut nextState: u32 = 0;
let __v28: *mut u16 = symbolNext.wrapping_offset(symbol as isize);
let __v29: u16 = unsafe { *__v28 };
unsafe {
*__v28 = (__v29 as i32).wrapping_add(1) as u16;
}
nextState = __v29 as u32;
unsafe {
(*tableDecode.wrapping_offset(u as isize)).nbBits = tableLog.wrapping_sub(ZSTD_highbit32(nextState)) as u8;
}
unsafe {
(*tableDecode.wrapping_offset(u as isize)).nextState = (nextState << ((*tableDecode.wrapping_offset(u as isize)).nbBits as i32)).wrapping_sub(tableSize) as u16;
}
unsafe {
(*tableDecode.wrapping_offset(u as isize)).nbAdditionalBits = *nbAdditionalBits.wrapping_offset(symbol as isize);
}
unsafe {
(*tableDecode.wrapping_offset(u as isize)).baseValue = *baseValue.wrapping_offset(symbol as isize);
}
}
}

/// Avoids the FORCE_INLINE of the _body() function.
fn ZSTD_buildFSETable_body_default(dt: *mut __SlateRecord_zstd_decompress_internal_h_73, normalizedCounter: *const i16, maxSymbolValue: u32, baseValue: *const u32, nbAdditionalBits: *const u8, tableLog: u32, wksp: *mut (), wkspSize: u64) {
ZSTD_buildFSETable_body(dt, normalizedCounter, maxSymbolValue, baseValue, nbAdditionalBits, tableLog, wksp, wkspSize);
}

#[target_feature(enable = "lzcnt,bmi1,bmi2")]
unsafe fn ZSTD_buildFSETable_body_bmi2(dt: *mut __SlateRecord_zstd_decompress_internal_h_73, normalizedCounter: *const i16, maxSymbolValue: u32, baseValue: *const u32, nbAdditionalBits: *const u8, tableLog: u32, wksp: *mut (), wkspSize: u64) {
ZSTD_buildFSETable_body(dt, normalizedCounter, maxSymbolValue, baseValue, nbAdditionalBits, tableLog, wksp, wkspSize);
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_buildFSETable(dt: *mut __SlateRecord_zstd_decompress_internal_h_73, normalizedCounter: *const i16, maxSymbolValue: u32, baseValue: *const u32, nbAdditionalBits: *const u8, tableLog: u32, wksp: *mut (), wkspSize: u64, bmi2: i32) {
if bmi2 != 0 {
unsafe { ZSTD_buildFSETable_body_bmi2(dt, normalizedCounter, maxSymbolValue, baseValue, nbAdditionalBits, tableLog, wksp, wkspSize) };
return;
}
ZSTD_buildFSETable_body_default(dt, normalizedCounter, maxSymbolValue, baseValue, nbAdditionalBits, tableLog, wksp, wkspSize);
}

/// ZSTD_buildSeqTable() :
/// @return : nb bytes read from src,
///           or an error code if it fails
fn ZSTD_buildSeqTable(DTableSpace: *mut __SlateRecord_zstd_decompress_internal_h_73, DTablePtr: *mut *const __SlateRecord_zstd_decompress_internal_h_73, r#type: u32, mut max: u32, maxLog: u32, src: *const (), srcSize: u64, baseValue: *const u32, nbAdditionalBits: *const u8, defaultTable: *const __SlateRecord_zstd_decompress_internal_h_73, flagRepeatTable: u32, ddictIsCold: i32, nbSeq: i32, wksp: *mut u32, wkspSize: u64, bmi2: i32) -> u64 {
match r#type as u32 {
1 => {
loop {
if srcSize == 0 {
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
if (((unsafe { *(src as *const u8) }) as i32) as u32) > max {
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
let symbol: u32 = (unsafe { *(src as *const u8) }) as u32;
let baseline: u32 = unsafe { *baseValue.wrapping_offset(symbol as isize) };
let nbBits: u8 = unsafe { *nbAdditionalBits.wrapping_offset(symbol as isize) };
ZSTD_buildSeqTable_rle(DTableSpace, baseline, nbBits);
unsafe {
*DTablePtr = DTableSpace as *const __SlateRecord_zstd_decompress_internal_h_73;
}
return 1;
}
0 => {
unsafe {
*DTablePtr = defaultTable;
}
return 0;
}
3 => {
loop {
if flagRepeatTable == 0 {
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
// prefetch FSE table if used
if ddictIsCold != 0 && nbSeq > 24 {
// heuristic
let pStart: *const () = (unsafe { *DTablePtr }) as *const ();
let pSize: u64 = (core::mem::size_of::<__SlateRecord_zstd_decompress_internal_h_73>() as u64).wrapping_mul(SEQSYMBOL_TABLE_SIZE!(maxLog) as u64);
PREFETCH_AREA!(pStart, pSize);
}
return 0;
}
2 => {
let mut tableLog: u32 = 0;
let mut norm: __SlateAlign16<[i16; 53]> = __SlateAlign16([0i16; 53]);
let headerSize: u64 = unsafe { FSE_readNCount(norm.0.as_mut_ptr() as *mut i16, core::ptr::addr_of_mut!(max), core::ptr::addr_of_mut!(tableLog), src, srcSize) };
loop {
if ERR_isError(headerSize) != 0 {
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
loop {
if tableLog > maxLog {
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
ZSTD_buildFSETable(DTableSpace, norm.0.as_mut_ptr() as *const i16, max, baseValue, nbAdditionalBits, tableLog, wksp as *mut (), wkspSize, bmi2);
unsafe {
*DTablePtr = DTableSpace as *const __SlateRecord_zstd_decompress_internal_h_73;
}
return headerSize;
}
_ => {
loop {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"impossible".as_ptr() as *const u8) };
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
}
}
unsafe { core::mem::zeroed() }
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_decodeSeqHeaders(dctx: *mut ZSTD_DCtx_s, nbSeqPtr: *mut i32, src: *const (), srcSize: u64) -> u64 {
let istart: *const u8 = src as *const u8;
let iend: *const u8 = istart.wrapping_offset(srcSize as isize);
let mut ip: *const u8 = istart;
let mut nbSeq: i32 = 0;
while false {
}
// check
loop {
if srcSize < 1 {
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
// SeqHead
let __v6: *const u8 = ip;
ip = __v6.wrapping_offset(1 as isize);
nbSeq = (unsafe { *__v6 }) as i32;
if nbSeq > 127 {
if nbSeq == 255 {
loop {
if ip.wrapping_offset(2 as isize) > iend {
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
nbSeq = (MEM_readLE16(ip as *const ()) as i32).wrapping_add(32512);
let __v13: *const u8 = ip;
ip = __v13.wrapping_offset(2 as isize);
} else {
loop {
if ip >= iend {
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
let __v20: *const u8 = ip;
ip = __v20.wrapping_offset(1 as isize);
nbSeq = (nbSeq.wrapping_sub(128) << 8i32).wrapping_add((unsafe { *__v20 }) as i32);
}
}
unsafe {
*nbSeqPtr = nbSeq;
}
if nbSeq == 0 {
// No sequence : section ends immediately
loop {
if ip != iend {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"extraneous data present in the Sequences section".as_ptr() as *const u8) };
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
return ((unsafe { ip.offset_from(istart as *const u8) }) as i64) as u64;
}
// FSE table descriptors
loop {
if ip.wrapping_offset(1 as isize) > iend {
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
// minimum possible size: 1 byte for symbol encoding types
loop {
if ((unsafe { *ip }) as i32) & 3 != 0 {
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
// The last field, Reserved, must be all-zeroes.
let LLtype: u32 = (((unsafe { *ip }) as i32) >> 6i32) as u32;
let OFtype: u32 = (((unsafe { *ip }) as i32) >> 4i32 & 3) as u32;
let MLtype: u32 = (((unsafe { *ip }) as i32) >> 2i32 & 3) as u32;
let __v37: *const u8 = ip;
ip = __v37.wrapping_offset(1 as isize);
// Build DTables
let llhSize: u64 = ZSTD_buildSeqTable(unsafe { (*dctx).entropy.LLTable.as_mut_ptr() as *mut __SlateRecord_zstd_decompress_internal_h_73 }, unsafe { core::ptr::addr_of_mut!((*dctx).LLTptr) }, LLtype, 35u32, 9u32, ip as *const (), ((unsafe { iend.offset_from(ip as *const u8) }) as i64) as u64, unsafe { core::ptr::addr_of!(LL_base.0) as *const u32 }, unsafe { core::ptr::addr_of!(LL_bits.0) as *const u8 }, unsafe { core::ptr::addr_of!(LL_defaultDTable.0) as *const __SlateRecord_zstd_decompress_internal_h_73 }, unsafe { (*dctx).fseEntropy }, unsafe { (*dctx).ddictIsCold }, nbSeq, unsafe { (*dctx).workspace.as_mut_ptr() as *mut u32 }, core::mem::size_of::<[u32; 640]>() as u64, ZSTD_DCtx_get_bmi2(dctx as *const ZSTD_DCtx_s));
loop {
if ERR_isError(llhSize) != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"ZSTD_buildSeqTable failed".as_ptr() as *const u8) };
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
let __v44: *const u8 = ip;
ip = __v44.wrapping_offset(llhSize as isize);
let ofhSize: u64 = ZSTD_buildSeqTable(unsafe { (*dctx).entropy.OFTable.as_mut_ptr() as *mut __SlateRecord_zstd_decompress_internal_h_73 }, unsafe { core::ptr::addr_of_mut!((*dctx).OFTptr) }, OFtype, 31u32, 8u32, ip as *const (), ((unsafe { iend.offset_from(ip as *const u8) }) as i64) as u64, unsafe { core::ptr::addr_of!(OF_base.0) as *const u32 }, unsafe { core::ptr::addr_of!(OF_bits.0) as *const u8 }, unsafe { core::ptr::addr_of!(OF_defaultDTable.0) as *const __SlateRecord_zstd_decompress_internal_h_73 }, unsafe { (*dctx).fseEntropy }, unsafe { (*dctx).ddictIsCold }, nbSeq, unsafe { (*dctx).workspace.as_mut_ptr() as *mut u32 }, core::mem::size_of::<[u32; 640]>() as u64, ZSTD_DCtx_get_bmi2(dctx as *const ZSTD_DCtx_s));
loop {
if ERR_isError(ofhSize) != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"ZSTD_buildSeqTable failed".as_ptr() as *const u8) };
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
let __v51: *const u8 = ip;
ip = __v51.wrapping_offset(ofhSize as isize);
let mlhSize: u64 = ZSTD_buildSeqTable(unsafe { (*dctx).entropy.MLTable.as_mut_ptr() as *mut __SlateRecord_zstd_decompress_internal_h_73 }, unsafe { core::ptr::addr_of_mut!((*dctx).MLTptr) }, MLtype, 52u32, 9u32, ip as *const (), ((unsafe { iend.offset_from(ip as *const u8) }) as i64) as u64, unsafe { core::ptr::addr_of!(ML_base.0) as *const u32 }, unsafe { core::ptr::addr_of!(ML_bits.0) as *const u8 }, unsafe { core::ptr::addr_of!(ML_defaultDTable.0) as *const __SlateRecord_zstd_decompress_internal_h_73 }, unsafe { (*dctx).fseEntropy }, unsafe { (*dctx).ddictIsCold }, nbSeq, unsafe { (*dctx).workspace.as_mut_ptr() as *mut u32 }, core::mem::size_of::<[u32; 640]>() as u64, ZSTD_DCtx_get_bmi2(dctx as *const ZSTD_DCtx_s));
loop {
if ERR_isError(mlhSize) != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"ZSTD_buildSeqTable failed".as_ptr() as *const u8) };
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
let __v58: *const u8 = ip;
ip = __v58.wrapping_offset(mlhSize as isize);
((unsafe { ip.offset_from(istart as *const u8) }) as i64) as u64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_decompress_block_c_783 {
    litLength: u64,
    matchLength: u64,
    offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_decompress_block_c_788 {
    state: u64,
    table: *const __SlateRecord_zstd_decompress_internal_h_73,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_decompress_block_c_796 {
    DStream: __SlateRecord_bitstream_h_90,
    stateLL: __SlateRecord_zstd_decompress_block_c_788,
    stateOffb: __SlateRecord_zstd_decompress_block_c_788,
    stateML: __SlateRecord_zstd_decompress_block_c_788,
    prevOffset: [u64; 3],
}


/// ZSTD_overlapCopy8() :
/// Copies 8 bytes from ip to op and updates op and ip where ip <= op.
/// If the offset is < 8 then the offset is spread to at least 8 bytes.
///
/// Precondition: *ip <= *op
/// Postcondition: *op - *op >= 8
#[inline(always)]
fn ZSTD_overlapCopy8(op: &mut *mut u8, ip: &mut *const u8, offset: u64) {
if offset < 8 {
// close range match, overlap
// added
// subtracted
let sub2: i32 = unsafe { dec64table.0[offset as usize] };
unsafe {
*unsafe { *(op as *mut *mut u8) }.wrapping_offset(0 as isize) = *unsafe { *(ip as *mut *const u8) }.wrapping_offset(0 as isize);
}
unsafe {
*unsafe { *(op as *mut *mut u8) }.wrapping_offset(1 as isize) = *unsafe { *(ip as *mut *const u8) }.wrapping_offset(1 as isize);
}
unsafe {
*unsafe { *(op as *mut *mut u8) }.wrapping_offset(2 as isize) = *unsafe { *(ip as *mut *const u8) }.wrapping_offset(2 as isize);
}
unsafe {
*unsafe { *(op as *mut *mut u8) }.wrapping_offset(3 as isize) = *unsafe { *(ip as *mut *const u8) }.wrapping_offset(3 as isize);
}
let __v1: *const u8 = unsafe { *ip };
unsafe {
*ip = __v1.wrapping_offset(dec32table.0[offset as usize] as isize);
}
ZSTD_copy4(unsafe { *(op as *mut *mut u8) }.wrapping_offset(4 as isize) as *mut (), (unsafe { *(ip as *mut *const u8) }) as *const ());
let __v4: *const u8 = unsafe { *ip };
unsafe {
*ip = __v4.wrapping_offset(-(sub2 as isize));
}
} else {
ZSTD_copy8((unsafe { *(op as *mut *mut u8) }) as *mut (), (unsafe { *(ip as *mut *const u8) }) as *const ());
}
let __v7: *const u8 = unsafe { *ip };
unsafe {
*ip = __v7.wrapping_offset(8 as isize);
}
let __v10: *mut u8 = unsafe { *op };
unsafe {
*op = __v10.wrapping_offset(8 as isize);
}
}

static dec32table: __SlateAlign16<[u32; 8]> = __SlateAlign16([0u32, 1u32, 2u32, 1u32, 4u32, 4u32, 4u32, 4u32]);

static dec64table: __SlateAlign16<[i32; 8]> = __SlateAlign16([8i32, 8i32, 8i32, 7i32, 8i32, 9i32, 10i32, 11i32]);

/// ZSTD_safecopy() :
/// Specialized version of memcpy() that is allowed to READ up to WILDCOPY_OVERLENGTH past the input buffer
/// and write up to 16 bytes past oend_w (op >= oend_w is allowed).
/// This function is only called in the uncommon case where the sequence is near the end of the block. It
/// should be fast for a single long sequence, but can be slow for several short sequences.
///
/// @param ovtype controls the overlap detection
///        - ZSTD_no_overlap: The source and destination are guaranteed to be at least WILDCOPY_VECLEN bytes apart.
///        - ZSTD_overlap_src_before_dst: The src and dst may overlap and may be any distance apart.
///          The src buffer must be before the dst buffer.
fn ZSTD_safecopy(mut op: *mut u8, oend_w: *const u8, mut ip: *const u8, mut length: i64, ovtype: u32) {
let diff: i64 = (unsafe { op.offset_from(ip as *mut u8) }) as i64;
let oend: *mut u8 = op.wrapping_offset(length as isize);
if length < 8 {
// Handle short lengths.
while op < oend {
let __v0: *const u8 = ip;
ip = __v0.wrapping_offset(1 as isize);
let __v2: *mut u8 = op;
op = __v2.wrapping_offset(1 as isize);
unsafe {
*__v2 = *__v0;
}
}
return;
}
if (ovtype as u32) == 1 {
// Copy 8 bytes and ensure the offset >= 8 when there can be overlap.
ZSTD_overlapCopy8(unsafe { &mut (*core::ptr::addr_of_mut!(op)) }, unsafe { &mut (*core::ptr::addr_of_mut!(ip)) }, diff as u64);
length = length.wrapping_sub(8);
}
if oend <= (oend_w as *mut u8) {
// No risk of overwrite.
ZSTD_wildcopy(op as *mut (), ip as *const (), length, ovtype);
return;
}
if op <= (oend_w as *mut u8) {
// Wildcopy until we get close to the end.
ZSTD_wildcopy(op as *mut (), ip as *const (), (unsafe { oend_w.offset_from(op as *const u8) }) as i64, ovtype);
let __v7: *const u8 = ip;
ip = __v7.wrapping_offset(((unsafe { oend_w.offset_from(op as *const u8) }) as i64) as isize);
let __v9: *mut u8 = op;
op = __v9.wrapping_offset(((unsafe { oend_w.offset_from(op as *const u8) }) as i64) as isize);
}
// Handle the leftovers.
while op < oend {
let __v11: *const u8 = ip;
ip = __v11.wrapping_offset(1 as isize);
let __v13: *mut u8 = op;
op = __v13.wrapping_offset(1 as isize);
unsafe {
*__v13 = *__v11;
}
}
}

/// ZSTD_safecopyDstBeforeSrc():
/// This version allows overlap with dst before src, or handles the non-overlap case with dst after src
/// Kept separate from more common ZSTD_safecopy case to avoid performance impact to the safecopy common case
fn ZSTD_safecopyDstBeforeSrc(mut op: *mut u8, mut ip: *const u8, length: i64) {
let diff: i64 = (unsafe { op.offset_from(ip as *mut u8) }) as i64;
let oend: *mut u8 = op.wrapping_offset(length as isize);
if length < 8 || diff > (8i32.wrapping_neg() as i64) {
// Handle short lengths, close overlaps, and dst not before src.
while op < oend {
let __v0: *const u8 = ip;
ip = __v0.wrapping_offset(1 as isize);
let __v2: *mut u8 = op;
op = __v2.wrapping_offset(1 as isize);
unsafe {
*__v2 = *__v0;
}
}
return;
}
if op <= oend.wrapping_offset(-(32 as isize)) && diff < (16i32.wrapping_neg() as i64) {
ZSTD_wildcopy(op as *mut (), ip as *const (), (unsafe { oend.wrapping_offset(-(32 as isize)).offset_from(op as *mut u8) }) as i64, 0u32);
let __v5: *const u8 = ip;
ip = __v5.wrapping_offset(((unsafe { oend.wrapping_offset(-(32 as isize)).offset_from(op as *mut u8) }) as i64) as isize);
let __v7: *mut u8 = op;
op = __v7.wrapping_offset(((unsafe { oend.wrapping_offset(-(32 as isize)).offset_from(op as *mut u8) }) as i64) as isize);
}
// Handle the leftovers.
while op < oend {
let __v9: *const u8 = ip;
ip = __v9.wrapping_offset(1 as isize);
let __v11: *mut u8 = op;
op = __v11.wrapping_offset(1 as isize);
unsafe {
*__v11 = *__v9;
}
}
}

/// ZSTD_execSequenceEnd():
/// This version handles cases that are near the end of the output buffer. It requires
/// more careful checks to make sure there is no overflow. By separating out these hard
/// and unlikely cases, we can speed up the common cases.
///
/// NOTE: This function needs to be fast for a single long sequence, but doesn't need
/// to be optimized for many small sequences, since those fall into ZSTD_execSequence().
#[inline(never)]
fn ZSTD_execSequenceEnd(mut op: *mut u8, oend: *mut u8, mut sequence: __SlateRecord_zstd_decompress_block_c_783, litPtr: &mut *const u8, litLimit: *const u8, prefixStart: *const u8, virtualStart: *const u8, dictEnd: *const u8) -> u64 {
let oLitEnd: *mut u8 = op.wrapping_offset(sequence.litLength as isize);
let sequenceLength: u64 = sequence.litLength.wrapping_add(sequence.matchLength);
let iLitEnd: *const u8 = unsafe { *(litPtr as *mut *const u8) }.wrapping_offset(sequence.litLength as isize);
let mut r#match: *const u8 = oLitEnd.wrapping_offset(-(sequence.offset as isize)) as *const u8;
let oend_w: *mut u8 = oend.wrapping_offset(-(32 as isize));
// bounds checks : careful of address space overflow in 32-bit mode
loop {
if sequenceLength > (((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64) {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"last match must fit within dstBuffer".as_ptr() as *const u8) };
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
loop {
if sequence.litLength > (((unsafe { litLimit.offset_from(*(litPtr as *mut *const u8) as *const u8) }) as i64) as u64) {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"try to read beyond literal buffer".as_ptr() as *const u8) };
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
// copy literals
ZSTD_safecopy(op, oend_w as *const u8, unsafe { *(litPtr as *mut *const u8) }, sequence.litLength as i64, 0u32);
op = oLitEnd;
unsafe {
*(litPtr as *mut *const u8) = iLitEnd;
}
// copy Match
if sequence.offset > (((unsafe { oLitEnd.offset_from(prefixStart as *mut u8) }) as i64) as u64) {
// offset beyond prefix
loop {
if sequence.offset > (((unsafe { oLitEnd.offset_from(virtualStart as *mut u8) }) as i64) as u64) {
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
r#match = dictEnd.wrapping_offset(-(((unsafe { prefixStart.offset_from(r#match as *const u8) }) as i64) as isize));
if r#match.wrapping_offset(sequence.matchLength as isize) <= dictEnd {
ZSTD_memmove!(oLitEnd, r#match, sequence.matchLength);
return sequenceLength;
}
// span extDict & currentPrefixSegment
let length1: u64 = ((unsafe { dictEnd.offset_from(r#match as *const u8) }) as i64) as u64;
ZSTD_memmove!(oLitEnd, r#match, length1);
op = oLitEnd.wrapping_offset(length1 as isize);
let __v15: u64 = sequence.matchLength;
sequence.matchLength = __v15.wrapping_sub(length1);
r#match = prefixStart;
}
ZSTD_safecopy(op, oend_w as *const u8, r#match, sequence.matchLength as i64, 1u32);
sequenceLength
}

/// ZSTD_execSequenceEndSplitLitBuffer():
/// This version is intended to be used during instances where the litBuffer is still split.  It is kept separate to avoid performance impact for the good case.
#[inline(never)]
fn ZSTD_execSequenceEndSplitLitBuffer(mut op: *mut u8, oend: *mut u8, oend_w: *const u8, mut sequence: __SlateRecord_zstd_decompress_block_c_783, litPtr: *mut *const u8, litLimit: *const u8, prefixStart: *const u8, virtualStart: *const u8, dictEnd: *const u8) -> u64 {
let oLitEnd: *mut u8 = op.wrapping_offset(sequence.litLength as isize);
let sequenceLength: u64 = sequence.litLength.wrapping_add(sequence.matchLength);
let iLitEnd: *const u8 = unsafe { *litPtr }.wrapping_offset(sequence.litLength as isize);
let mut r#match: *const u8 = oLitEnd.wrapping_offset(-(sequence.offset as isize)) as *const u8;
// bounds checks : careful of address space overflow in 32-bit mode
loop {
if sequenceLength > (((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64) {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"last match must fit within dstBuffer".as_ptr() as *const u8) };
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
loop {
if sequence.litLength > (((unsafe { litLimit.offset_from(*litPtr as *const u8) }) as i64) as u64) {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"try to read beyond literal buffer".as_ptr() as *const u8) };
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
// copy literals
loop {
if op > ((unsafe { *litPtr }) as *mut u8) && op < (unsafe { *litPtr }.wrapping_offset(sequence.litLength as isize) as *mut u8) {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"output should not catch up to and overwrite literal buffer".as_ptr() as *const u8) };
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
ZSTD_safecopyDstBeforeSrc(op, unsafe { *litPtr }, sequence.litLength as i64);
op = oLitEnd;
unsafe {
*litPtr = iLitEnd;
}
// copy Match
if sequence.offset > (((unsafe { oLitEnd.offset_from(prefixStart as *mut u8) }) as i64) as u64) {
// offset beyond prefix
loop {
if sequence.offset > (((unsafe { oLitEnd.offset_from(virtualStart as *mut u8) }) as i64) as u64) {
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
r#match = dictEnd.wrapping_offset(-(((unsafe { prefixStart.offset_from(r#match as *const u8) }) as i64) as isize));
if r#match.wrapping_offset(sequence.matchLength as isize) <= dictEnd {
ZSTD_memmove!(oLitEnd, r#match, sequence.matchLength);
return sequenceLength;
}
// span extDict & currentPrefixSegment
let length1: u64 = ((unsafe { dictEnd.offset_from(r#match as *const u8) }) as i64) as u64;
ZSTD_memmove!(oLitEnd, r#match, length1);
op = oLitEnd.wrapping_offset(length1 as isize);
let __v20: u64 = sequence.matchLength;
sequence.matchLength = __v20.wrapping_sub(length1);
r#match = prefixStart;
}
ZSTD_safecopy(op, oend_w, r#match, sequence.matchLength as i64, 1u32);
sequenceLength
}

#[inline(always)]
fn ZSTD_execSequence(mut op: *mut u8, oend: *mut u8, mut sequence: __SlateRecord_zstd_decompress_block_c_783, litPtr: &mut *const u8, litLimit: *const u8, prefixStart: *const u8, virtualStart: *const u8, dictEnd: *const u8) -> u64 {
let oLitEnd: *mut u8 = op.wrapping_offset(sequence.litLength as isize);
let sequenceLength: u64 = sequence.litLength.wrapping_add(sequence.matchLength);
let oMatchEnd: *mut u8 = op.wrapping_offset(sequenceLength as isize); // risk : address space overflow (32-bits)
let oend_w: *mut u8 = oend.wrapping_offset(-(32 as isize)); // risk : address space underflow on oend=NULL
let iLitEnd: *const u8 = unsafe { *(litPtr as *mut *const u8) }.wrapping_offset(sequence.litLength as isize);
let mut r#match: *const u8 = oLitEnd.wrapping_offset(-(sequence.offset as isize)) as *const u8;
// Precondition
// No underflow
// Handle edge cases in a slow path:
// - Read beyond end of literals
// - Match end is within WILDCOPY_OVERLIMIT of oend
// - 32-bit mode and the match length overflows
let __v0: bool = iLitEnd > litLimit || oMatchEnd > oend_w || MEM_32bits() != 0 && (((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64) < sequenceLength.wrapping_add(32);
if __v0 {
return ZSTD_execSequenceEnd(op, oend, sequence, litPtr, litLimit, prefixStart, virtualStart, dictEnd);
}
// Assumptions (everything else goes into ZSTD_execSequenceEnd())
// No overflow
// Non-zero match & no overflow
// No underflow
// Literal length is in bounds
// Can wildcopy literals
// Can wildcopy matches
// Copy Literals:
// Split out litLength <= 16 since it is nearly always true. +1.6% on gcc-9.
// We likely don't need the full 32-byte wildcopy.
ZSTD_copy16(op as *mut (), (unsafe { *(litPtr as *mut *const u8) }) as *const ());
if UNLIKELY!(sequence.litLength > 16) != 0 {
ZSTD_wildcopy(op.wrapping_offset(16 as isize) as *mut (), unsafe { *(litPtr as *mut *const u8) }.wrapping_offset(16 as isize) as *const (), sequence.litLength.wrapping_sub(16) as i64, 0u32);
}
op = oLitEnd;
unsafe {
*(litPtr as *mut *const u8) = iLitEnd;
}
// update for next sequence
// Copy Match
if sequence.offset > (((unsafe { oLitEnd.offset_from(prefixStart as *mut u8) }) as i64) as u64) {
// offset beyond prefix -> go into extDict
loop {
if UNLIKELY!(sequence.offset > (((unsafe { oLitEnd.offset_from(virtualStart as *mut u8) }) as i64) as u64)) != 0 {
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
r#match = dictEnd.wrapping_offset(((unsafe { r#match.offset_from(prefixStart as *const u8) }) as i64) as isize);
if r#match.wrapping_offset(sequence.matchLength as isize) <= dictEnd {
ZSTD_memmove!(oLitEnd, r#match, sequence.matchLength);
return sequenceLength;
}
// span extDict & currentPrefixSegment
let length1: u64 = ((unsafe { dictEnd.offset_from(r#match as *const u8) }) as i64) as u64;
ZSTD_memmove!(oLitEnd, r#match, length1);
op = oLitEnd.wrapping_offset(length1 as isize);
let __v6: u64 = sequence.matchLength;
sequence.matchLength = __v6.wrapping_sub(length1);
r#match = prefixStart;
}
// Match within prefix of 1 or more bytes
// Nearly all offsets are >= WILDCOPY_VECLEN bytes, which means we can use wildcopy
// without overlap checking.
if LIKELY!(sequence.offset >= 16) != 0 {
// We bet on a full wildcopy for matches, since we expect matches to be
// longer than literals (in general). In silesia, ~10% of matches are longer
// than 16 bytes.
ZSTD_wildcopy(op as *mut (), r#match as *const (), sequence.matchLength as i64, 0u32);
return sequenceLength;
}
// Copy 8 bytes and spread the offset to be >= 8.
ZSTD_overlapCopy8(unsafe { &mut (*core::ptr::addr_of_mut!(op)) }, unsafe { &mut (*core::ptr::addr_of_mut!(r#match)) }, sequence.offset);
// If the match length is > 8 bytes, then continue with the wildcopy.
if sequence.matchLength > 8 {
ZSTD_wildcopy(op as *mut (), r#match as *const (), (sequence.matchLength as i64).wrapping_sub(8), 1u32);
}
sequenceLength
}

#[inline(always)]
fn ZSTD_execSequenceSplitLitBuffer(mut op: *mut u8, oend: *mut u8, oend_w: *const u8, mut sequence: __SlateRecord_zstd_decompress_block_c_783, litPtr: *mut *const u8, litLimit: *const u8, prefixStart: *const u8, virtualStart: *const u8, dictEnd: *const u8) -> u64 {
let oLitEnd: *mut u8 = op.wrapping_offset(sequence.litLength as isize);
let sequenceLength: u64 = sequence.litLength.wrapping_add(sequence.matchLength);
let oMatchEnd: *mut u8 = op.wrapping_offset(sequenceLength as isize); // risk : address space overflow (32-bits)
let iLitEnd: *const u8 = unsafe { *litPtr }.wrapping_offset(sequence.litLength as isize);
let mut r#match: *const u8 = oLitEnd.wrapping_offset(-(sequence.offset as isize)) as *const u8;
// Precondition
// No underflow
// Handle edge cases in a slow path:
// - Read beyond end of literals
// - Match end is within WILDCOPY_OVERLIMIT of oend
// - 32-bit mode and the match length overflows
let __v0: bool = iLitEnd > litLimit || oMatchEnd > (oend_w as *mut u8) || MEM_32bits() != 0 && (((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64) < sequenceLength.wrapping_add(32);
if __v0 {
return ZSTD_execSequenceEndSplitLitBuffer(op, oend, oend_w, sequence, litPtr, litLimit, prefixStart, virtualStart, dictEnd);
}
// Assumptions (everything else goes into ZSTD_execSequenceEnd())
// No overflow
// Non-zero match & no overflow
// No underflow
// Literal length is in bounds
// Can wildcopy literals
// Can wildcopy matches
// Copy Literals:
// Split out litLength <= 16 since it is nearly always true. +1.6% on gcc-9.
// We likely don't need the full 32-byte wildcopy.
ZSTD_copy16(op as *mut (), (unsafe { *litPtr }) as *const ());
if UNLIKELY!(sequence.litLength > 16) != 0 {
ZSTD_wildcopy(op.wrapping_offset(16 as isize) as *mut (), unsafe { *litPtr }.wrapping_offset(16 as isize) as *const (), sequence.litLength.wrapping_sub(16) as i64, 0u32);
}
op = oLitEnd;
unsafe {
*litPtr = iLitEnd;
}
// update for next sequence
// Copy Match
if sequence.offset > (((unsafe { oLitEnd.offset_from(prefixStart as *mut u8) }) as i64) as u64) {
// offset beyond prefix -> go into extDict
loop {
if UNLIKELY!(sequence.offset > (((unsafe { oLitEnd.offset_from(virtualStart as *mut u8) }) as i64) as u64)) != 0 {
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
r#match = dictEnd.wrapping_offset(((unsafe { r#match.offset_from(prefixStart as *const u8) }) as i64) as isize);
if r#match.wrapping_offset(sequence.matchLength as isize) <= dictEnd {
ZSTD_memmove!(oLitEnd, r#match, sequence.matchLength);
return sequenceLength;
}
// span extDict & currentPrefixSegment
let length1: u64 = ((unsafe { dictEnd.offset_from(r#match as *const u8) }) as i64) as u64;
ZSTD_memmove!(oLitEnd, r#match, length1);
op = oLitEnd.wrapping_offset(length1 as isize);
let __v6: u64 = sequence.matchLength;
sequence.matchLength = __v6.wrapping_sub(length1);
r#match = prefixStart;
}
// Match within prefix of 1 or more bytes
// Nearly all offsets are >= WILDCOPY_VECLEN bytes, which means we can use wildcopy
// without overlap checking.
if LIKELY!(sequence.offset >= 16) != 0 {
// We bet on a full wildcopy for matches, since we expect matches to be
// longer than literals (in general). In silesia, ~10% of matches are longer
// than 16 bytes.
ZSTD_wildcopy(op as *mut (), r#match as *const (), sequence.matchLength as i64, 0u32);
return sequenceLength;
}
// Copy 8 bytes and spread the offset to be >= 8.
ZSTD_overlapCopy8(unsafe { &mut (*core::ptr::addr_of_mut!(op)) }, unsafe { &mut (*core::ptr::addr_of_mut!(r#match)) }, sequence.offset);
// If the match length is > 8 bytes, then continue with the wildcopy.
if sequence.matchLength > 8 {
ZSTD_wildcopy(op as *mut (), r#match as *const (), (sequence.matchLength as i64).wrapping_sub(8), 1u32);
}
sequenceLength
}

fn ZSTD_initFseState(DStatePtr: *mut __SlateRecord_zstd_decompress_block_c_788, bitD: *mut __SlateRecord_bitstream_h_90, dt: *const __SlateRecord_zstd_decompress_internal_h_73) {
let ptr: *const () = dt as *const ();
unsafe {
(*DStatePtr).state = BIT_readBits(bitD, (*(ptr as *const __SlateRecord_zstd_decompress_internal_h_66)).tableLog);
}
while false {
}
BIT_reloadDStream(bitD);
unsafe {
(*DStatePtr).table = dt.wrapping_offset(1 as isize);
}
}

#[inline(always)]
fn ZSTD_updateFseStateWithDInfo(DStatePtr: *mut __SlateRecord_zstd_decompress_block_c_788, bitD: *mut __SlateRecord_bitstream_h_90, nextState: u16, nbBits: u32) {
let lowBits: u64 = BIT_readBits(bitD, nbBits);
unsafe {
(*DStatePtr).state = ((nextState as i32) as u64).wrapping_add(lowBits);
}
}

// We need to add at most (ZSTD_WINDOWLOG_MAX_32 - 1) bits to read the maximum
// offset bits. But we can only read at most STREAM_ACCUMULATOR_MIN_32
// bits before reloading. This value is the maximum number of bytes we read
// after reloading when we are decoding long offsets.
/// ZSTD_decodeSequence():
/// @p longOffsets : tells the decoder to reload more bit while decoding large offsets
///                  only used in 32-bit mode
/// @return : Sequence (litL + matchL + offset)
#[inline(always)]
fn ZSTD_decodeSequence(seqState: &mut __SlateRecord_zstd_decompress_block_c_796, longOffsets: u32, isLastSeq: i32) -> __SlateRecord_zstd_decompress_block_c_783 {
let mut seq: __SlateRecord_zstd_decompress_block_c_783 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(seq) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
// ZSTD_seqSymbol is a 64 bits wide structure.
// It can be loaded in one operation
// and its fields extracted by simply shifting or bit-extracting on aarch64.
// GCC doesn't recognize this and generates more unnecessary ldr/ldrb/ldrh
// operations that cause performance drop. This can be avoided by using this
// ZSTD_memcpy hack.
let llDInfo: *const __SlateRecord_zstd_decompress_internal_h_73 = unsafe { (*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).stateLL.table }.wrapping_offset((unsafe { (*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).stateLL.state }) as isize);
let mlDInfo: *const __SlateRecord_zstd_decompress_internal_h_73 = unsafe { (*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).stateML.table }.wrapping_offset((unsafe { (*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).stateML.state }) as isize);
let ofDInfo: *const __SlateRecord_zstd_decompress_internal_h_73 = unsafe { (*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).stateOffb.table }.wrapping_offset((unsafe { (*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).stateOffb.state }) as isize);
seq.matchLength = (unsafe { (*mlDInfo).baseValue }) as u64;
seq.litLength = (unsafe { (*llDInfo).baseValue }) as u64;
let ofBase: u32 = unsafe { (*ofDInfo).baseValue };
let llBits: u8 = unsafe { (*llDInfo).nbAdditionalBits };
let mlBits: u8 = unsafe { (*mlDInfo).nbAdditionalBits };
let ofBits: u8 = unsafe { (*ofDInfo).nbAdditionalBits };
let totalBits: u8 = (llBits as i32).wrapping_add(mlBits as i32).wrapping_add(ofBits as i32) as u8;
let llNext: u16 = unsafe { (*llDInfo).nextState };
let mlNext: u16 = unsafe { (*mlDInfo).nextState };
let ofNext: u16 = unsafe { (*ofDInfo).nextState };
let llnbBits: u32 = (unsafe { (*llDInfo).nbBits }) as u32;
let mlnbBits: u32 = (unsafe { (*mlDInfo).nbBits }) as u32;
let ofnbBits: u32 = (unsafe { (*ofDInfo).nbBits }) as u32;
// As gcc has better branch and block analyzers, sometimes it is only
// valuable to mark likeliness for clang, it gives around 3-4% of
// performance.
// sequence
let mut offset: u64 = 0;
if (ofBits as i32) > 1 {
ZSTD_STATIC_ASSERT!();
ZSTD_STATIC_ASSERT!();
ZSTD_STATIC_ASSERT!();
ZSTD_STATIC_ASSERT!();
if MEM_32bits() != 0 && (longOffsets as u32) != 0 && (ofBits as i32) >= 25 {
// Always read extra bits, this keeps the logic simple,
// avoids branches, and avoids accidentally reading 0 bits.
let extraBits: u32 = (if true { 5i32 } else { 0i32 }) as u32;
offset = (ofBase as u64).wrapping_add(BIT_readBitsFast(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) }, ((ofBits as i32) as u32).wrapping_sub(extraBits)) << extraBits);
BIT_reloadDStream(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) });
let __v0: u64 = BIT_readBitsFast(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) }, extraBits);
offset = offset.wrapping_add(__v0);
} else {
offset = (ofBase as u64).wrapping_add(BIT_readBitsFast(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) }, ofBits as u32)); // >0
// <=  (ZSTD_WINDOWLOG_MAX-1) bits
if MEM_32bits() != 0 {
BIT_reloadDStream(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) });
}
}
unsafe {
*((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(2 as isize) = *((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(1 as isize);
}
unsafe {
*((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(1 as isize) = *((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize);
}
unsafe {
*((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) = offset;
}
} else {
let ll0: u32 = ((unsafe { (*llDInfo).baseValue }) == 0) as u32;
if LIKELY!((ofBits as i32) == 0) != 0 {
offset = unsafe { *((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(ll0 as isize) };
unsafe {
*((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(1 as isize) = *((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(((ll0 == 0) as i32) as isize);
}
unsafe {
*((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) = offset;
}
} else {
offset = (ofBase.wrapping_add(ll0) as u64).wrapping_add(BIT_readBitsFast(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) }, 1u32));
let mut temp: u64 = if offset == 3 { unsafe { *((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) }.wrapping_sub(1) } else { unsafe { *((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(offset as isize) } };
temp = temp.wrapping_sub(((temp == 0) as i32) as u64); // 0 is not valid: input corrupted => force offset to -1 => corruption detected at execSequence
if offset != 1 {
unsafe {
*((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(2 as isize) = *((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(1 as isize);
}
}
unsafe {
*((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(1 as isize) = *((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize);
}
let __v5: u64 = temp;
offset = __v5;
unsafe {
*((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).prevOffset.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) = __v5;
}
}
}
seq.offset = offset;
if (mlBits as i32) > 0 {
let __v6: u64 = BIT_readBitsFast(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) }, mlBits as u32);
let __v7: u64 = seq.matchLength;
seq.matchLength = __v7.wrapping_add(__v6);
}
// >0
if MEM_32bits() != 0 && (mlBits as i32).wrapping_add(llBits as i32) >= 25i32.wrapping_sub(if true { 5i32 } else { 0i32 }) {
BIT_reloadDStream(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) });
}
if MEM_64bits() != 0 && UNLIKELY!((totalBits as i32) >= 31i32) != 0 {
BIT_reloadDStream(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) });
}
// Ensure there are enough bits to read the rest of data in 64-bit mode.
ZSTD_STATIC_ASSERT!();
if (llBits as i32) > 0 {
let __v9: u64 = BIT_readBitsFast(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) }, llBits as u32);
let __v10: u64 = seq.litLength;
seq.litLength = __v10.wrapping_add(__v9);
}
// >0
if MEM_32bits() != 0 {
BIT_reloadDStream(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) });
}
while false {
}
if isLastSeq == 0 {
// don't update FSE state for last Sequence
ZSTD_updateFseStateWithDInfo(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).stateLL) }, unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) }, llNext, llnbBits); // <=  9 bits
ZSTD_updateFseStateWithDInfo(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).stateML) }, unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) }, mlNext, mlnbBits); // <=  9 bits
if MEM_32bits() != 0 {
BIT_reloadDStream(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) });
}
// <= 18 bits
ZSTD_updateFseStateWithDInfo(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).stateOffb) }, unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) }, ofNext, ofnbBits); // <=  8 bits
BIT_reloadDStream(unsafe { core::ptr::addr_of_mut!((*(seqState as *mut __SlateRecord_zstd_decompress_block_c_796)).DStream) });
}
return seq;
}

#[inline(always)]
fn ZSTD_decompressSequences_bodySplitLitBuffer(dctx: *mut ZSTD_DCtx_s, dst: *mut (), maxDstSize: u64, seqStart: *const (), seqSize: u64, mut nbSeq: i32, isLongOffset: u32) -> u64 {
let ip: *const u8 = seqStart as *const u8;
let iend: *const u8 = ip.wrapping_offset(seqSize as isize);
let ostart: *mut u8 = dst as *mut u8;
let oend: *mut u8 = ZSTD_maybeNullPtrAdd(ostart, maxDstSize as i64);
let mut op: *mut u8 = ostart;
let mut litPtr: *const u8 = unsafe { (*dctx).litPtr };
let mut litBufferEnd: *const u8 = unsafe { (*dctx).litBufferEnd };
let prefixStart: *const u8 = (unsafe { (*dctx).prefixStart }) as *const u8;
let vBase: *const u8 = (unsafe { (*dctx).virtualStart }) as *const u8;
let dictEnd: *const u8 = (unsafe { (*dctx).dictEnd }) as *const u8;
while false {
}
// Literals are split between internal buffer & output buffer
if nbSeq != 0 {
let mut seqState: __SlateRecord_zstd_decompress_block_c_796 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(seqState) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 112usize);
}
unsafe {
(*dctx).fseEntropy = 1;
}
for i in 0u32..3u32 {
seqState.prevOffset[i as usize] = (unsafe { *((*dctx).entropy.rep.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) }) as u64;
}
loop {
if ERR_isError(BIT_initDStream(core::ptr::addr_of_mut!(seqState.DStream), ip as *const (), ((unsafe { iend.offset_from(ip as *const u8) }) as i64) as u64)) != 0 {
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
ZSTD_initFseState(core::ptr::addr_of_mut!(seqState.stateLL), core::ptr::addr_of_mut!(seqState.DStream), unsafe { (*dctx).LLTptr });
ZSTD_initFseState(core::ptr::addr_of_mut!(seqState.stateOffb), core::ptr::addr_of_mut!(seqState.DStream), unsafe { (*dctx).OFTptr });
ZSTD_initFseState(core::ptr::addr_of_mut!(seqState.stateML), core::ptr::addr_of_mut!(seqState.DStream), unsafe { (*dctx).MLTptr });
ZSTD_STATIC_ASSERT!();
// decompress without overrunning litPtr begins
{
let mut sequence: __SlateRecord_zstd_decompress_block_c_783 = __SlateRecord_zstd_decompress_block_c_783 { litLength: 0, matchLength: 0, offset: 0 }; // some static analyzer believe that @sequence is not initialized (it necessarily is, since for(;;) loop as at least one iteration)
// Align the decompression loop to 32 + 16 bytes.
//
// zstd compiled with gcc-9 on an Intel i9-9900k shows 10% decompression
// speed swings based on the alignment of the decompression loop. This
// performance swing is caused by parts of the decompression loop falling
// out of the DSB. The entire decompression loop should fit in the DSB,
// when it can't we get much worse performance. You can measure if you've
// hit the good case or the bad case with this perf command for some
// compressed file test.zst:
//
//   perf stat -e cycles -e instructions -e idq.all_dsb_cycles_any_uops                 *             -e idq.all_mite_cycles_any_uops -- ./zstd -tq test.zst
//
// If you see most cycles served out of the MITE you've hit the bad case.
// If you see most cycles served out of the DSB you've hit the good case.
// If it is pretty even then you may be in an okay case.
//
// This issue has been reproduced on the following CPUs:
//   - Kabylake: Macbook Pro (15-inch, 2019) 2.4 GHz Intel Core i9
//               Use Instruments->Counters to get DSB/MITE cycles.
//               I never got performance swings, but I was able to
//               go from the good case of mostly DSB to half of the
//               cycles served from MITE.
//   - Coffeelake: Intel i9-9900k
//   - Coffeelake: Intel i7-9700k
//
// I haven't been able to reproduce the instability or DSB misses on any
// of the following CPUS:
//   - Haswell
//   - Broadwell: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GH
//   - Skylake
//
// Alignment is done for each of the three major decompression loops:
//   - ZSTD_decompressSequences_bodySplitLitBuffer - presplit section of the literal buffer
//   - ZSTD_decompressSequences_bodySplitLitBuffer - postsplit section of the literal buffer
//   - ZSTD_decompressSequences_body
// Alignment choices are made to minimize large swings on bad cases and influence on performance
// from changes external to this code, rather than to overoptimize on the current commit.
//
// If you are seeing performance stability this script can help test.
// It tests on 4 commits in zstd where I saw performance change.
//
//   https://gist.github.com/terrelln/9889fc06a423fd5ca6e99351564473f4
unsafe {
core::arch::asm!(".p2align 6", options(att_syntax));
}
// Handle the initial state where litBuffer is currently split between dst and litExtraBuffer
while nbSeq != 0 {
sequence = ZSTD_decodeSequence(unsafe { &mut (*core::ptr::addr_of_mut!(seqState)) }, isLongOffset, (nbSeq == 1) as i32);
if litPtr.wrapping_offset(sequence.litLength as isize) > unsafe { (*dctx).litBufferEnd } {
break;
}
let oneSeqSize: u64 = ZSTD_execSequenceSplitLitBuffer(op, oend, litPtr.wrapping_offset(sequence.litLength as isize).wrapping_offset(-(32 as isize)), sequence, core::ptr::addr_of_mut!(litPtr), litBufferEnd, prefixStart, vBase, dictEnd);
if UNLIKELY!(ERR_isError(oneSeqSize)) != 0 {
return oneSeqSize;
}
while false {
}
let __v11: *mut u8 = op;
op = __v11.wrapping_offset(oneSeqSize as isize);
nbSeq = nbSeq.wrapping_sub(1);
}
while false {
}
// If there are more sequences, they will need to read literals from litExtraBuffer; copy over the remainder from dst and update litPtr and litEnd
if nbSeq > 0 {
let leftoverLit: u64 = ((unsafe { unsafe { (*dctx).litBufferEnd }.offset_from(litPtr as *const u8) }) as i64) as u64;
while false {
}
if leftoverLit != 0 {
loop {
if leftoverLit > (((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64) {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"remaining lit must fit within dstBuffer".as_ptr() as *const u8) };
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
ZSTD_safecopyDstBeforeSrc(op, litPtr, leftoverLit as i64);
let __v22: u64 = sequence.litLength;
sequence.litLength = __v22.wrapping_sub(leftoverLit);
let __v24: *mut u8 = op;
op = __v24.wrapping_offset(leftoverLit as isize);
}
litPtr = (unsafe { (*dctx).litExtraBuffer.as_mut_ptr() as *mut u8 }) as *const u8;
litBufferEnd = unsafe { (*dctx).litExtraBuffer.as_mut_ptr() as *mut u8 }.wrapping_offset(BOUNDED!(64i32, 65536i32, 131072i32) as isize) as *const u8;
unsafe {
(*dctx).litBufferLocation = 0;
}
let oneSeqSize: u64 = ZSTD_execSequence(op, oend, sequence, unsafe { &mut (*core::ptr::addr_of_mut!(litPtr)) }, litBufferEnd, prefixStart, vBase, dictEnd);
if UNLIKELY!(ERR_isError(oneSeqSize)) != 0 {
return oneSeqSize;
}
while false {
}
let __v27: *mut u8 = op;
op = __v27.wrapping_offset(oneSeqSize as isize);
nbSeq = nbSeq.wrapping_sub(1);
}
}
if nbSeq > 0 {
// there is remaining lit from extra buffer
unsafe {
core::arch::asm!(".p2align 6", options(att_syntax));
}
unsafe {
core::arch::asm!("nop", options(att_syntax));
}
// worse for gcc-7 better for gcc-8, gcc-9, and gcc-10 and clang
unsafe {
core::arch::asm!(".p2align 4", options(att_syntax));
}
unsafe {
core::arch::asm!("nop", options(att_syntax));
}
unsafe {
core::arch::asm!(".p2align 3", options(att_syntax));
}
while nbSeq != 0 {
let sequence: __SlateRecord_zstd_decompress_block_c_783 = ZSTD_decodeSequence(unsafe { &mut (*core::ptr::addr_of_mut!(seqState)) }, isLongOffset, (nbSeq == 1) as i32);
let oneSeqSize: u64 = ZSTD_execSequence(op, oend, sequence, unsafe { &mut (*core::ptr::addr_of_mut!(litPtr)) }, litBufferEnd, prefixStart, vBase, dictEnd);
if UNLIKELY!(ERR_isError(oneSeqSize)) != 0 {
return oneSeqSize;
}
while false {
}
let __v32: *mut u8 = op;
op = __v32.wrapping_offset(oneSeqSize as isize);
nbSeq = nbSeq.wrapping_sub(1);
}
}
// check if reached exact end
while false {
}
loop {
if nbSeq != 0 {
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
while false {
}
loop {
if BIT_endOfDStream(unsafe { &(*(core::ptr::addr_of_mut!(seqState.DStream) as *const __SlateRecord_bitstream_h_90)) }) == 0 {
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
// save reps for next block
for i in 0u32..3u32 {
unsafe {
*((*dctx).entropy.rep.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) = seqState.prevOffset[i as usize] as u32;
}
}
}
// last literal segment
if ((unsafe { (*dctx).litBufferLocation }) as u32) == 2 {
// split hasn't been reached yet, first get dst then copy litExtraBuffer
let lastLLSize: u64 = ((unsafe { litBufferEnd.offset_from(litPtr as *const u8) }) as i64) as u64;
while false {
}
loop {
if lastLLSize > (((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64) {
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
if !op.is_null() {
ZSTD_memmove!(op, litPtr, lastLLSize);
let __v58: *mut u8 = op;
op = __v58.wrapping_offset(lastLLSize as isize);
}
litPtr = (unsafe { (*dctx).litExtraBuffer.as_mut_ptr() as *mut u8 }) as *const u8;
litBufferEnd = unsafe { (*dctx).litExtraBuffer.as_mut_ptr() as *mut u8 }.wrapping_offset(BOUNDED!(64i32, 65536i32, 131072i32) as isize) as *const u8;
unsafe {
(*dctx).litBufferLocation = 0;
}
}
// copy last literals from internal buffer
let lastLLSize: u64 = ((unsafe { litBufferEnd.offset_from(litPtr as *const u8) }) as i64) as u64;
while false {
}
loop {
if lastLLSize > (((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64) {
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
if !op.is_null() {
ZSTD_memcpy!(op, litPtr, lastLLSize);
let __v66: *mut u8 = op;
op = __v66.wrapping_offset(lastLLSize as isize);
}
while false {
}
((unsafe { op.offset_from(ostart as *mut u8) }) as i64) as u64
}

#[inline(always)]
fn ZSTD_decompressSequences_body(dctx: *mut ZSTD_DCtx_s, dst: *mut (), maxDstSize: u64, seqStart: *const (), seqSize: u64, mut nbSeq: i32, isLongOffset: u32) -> u64 {
let ip: *const u8 = seqStart as *const u8;
let iend: *const u8 = ip.wrapping_offset(seqSize as isize);
let ostart: *mut u8 = dst as *mut u8;
let __v0: *mut u8 = if ((unsafe { (*dctx).litBufferLocation }) as u32) == 0 {
ZSTD_maybeNullPtrAdd(ostart, maxDstSize as i64)
} else {
unsafe { (*dctx).litBuffer }
};
let oend: *mut u8 = __v0;
let mut op: *mut u8 = ostart;
let mut litPtr: *const u8 = unsafe { (*dctx).litPtr };
let litEnd: *const u8 = litPtr.wrapping_offset((unsafe { (*dctx).litSize }) as isize);
let prefixStart: *const u8 = (unsafe { (*dctx).prefixStart }) as *const u8;
let vBase: *const u8 = (unsafe { (*dctx).virtualStart }) as *const u8;
let dictEnd: *const u8 = (unsafe { (*dctx).dictEnd }) as *const u8;
while false {
}
// Regen sequences
if nbSeq != 0 {
let mut seqState: __SlateRecord_zstd_decompress_block_c_796 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(seqState) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 112usize);
}
unsafe {
(*dctx).fseEntropy = 1;
}
for i in 0u32..3u32 {
seqState.prevOffset[i as usize] = (unsafe { *((*dctx).entropy.rep.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) }) as u64;
}
loop {
if ERR_isError(BIT_initDStream(core::ptr::addr_of_mut!(seqState.DStream), ip as *const (), ((unsafe { iend.offset_from(ip as *const u8) }) as i64) as u64)) != 0 {
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
ZSTD_initFseState(core::ptr::addr_of_mut!(seqState.stateLL), core::ptr::addr_of_mut!(seqState.DStream), unsafe { (*dctx).LLTptr });
ZSTD_initFseState(core::ptr::addr_of_mut!(seqState.stateOffb), core::ptr::addr_of_mut!(seqState.DStream), unsafe { (*dctx).OFTptr });
ZSTD_initFseState(core::ptr::addr_of_mut!(seqState.stateML), core::ptr::addr_of_mut!(seqState.DStream), unsafe { (*dctx).MLTptr });
unsafe {
core::arch::asm!(".p2align 6", options(att_syntax));
}
unsafe {
core::arch::asm!("nop", options(att_syntax));
}
unsafe {
core::arch::asm!(".p2align 4", options(att_syntax));
}
unsafe {
core::arch::asm!("nop", options(att_syntax));
}
unsafe {
core::arch::asm!(".p2align 3", options(att_syntax));
}
while nbSeq != 0 {
let sequence: __SlateRecord_zstd_decompress_block_c_783 = ZSTD_decodeSequence(unsafe { &mut (*core::ptr::addr_of_mut!(seqState)) }, isLongOffset, (nbSeq == 1) as i32);
let oneSeqSize: u64 = ZSTD_execSequence(op, oend, sequence, unsafe { &mut (*core::ptr::addr_of_mut!(litPtr)) }, litEnd, prefixStart, vBase, dictEnd);
if UNLIKELY!(ERR_isError(oneSeqSize)) != 0 {
return oneSeqSize;
}
while false {
}
let __v11: *mut u8 = op;
op = __v11.wrapping_offset(oneSeqSize as isize);
nbSeq = nbSeq.wrapping_sub(1);
}
// check if reached exact end
loop {
if BIT_endOfDStream(unsafe { &(*(core::ptr::addr_of_mut!(seqState.DStream) as *const __SlateRecord_bitstream_h_90)) }) == 0 {
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
// save reps for next block
for i in 0u32..3u32 {
unsafe {
*((*dctx).entropy.rep.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) = seqState.prevOffset[i as usize] as u32;
}
}
}
// last literal segment
let lastLLSize: u64 = ((unsafe { litEnd.offset_from(litPtr as *const u8) }) as i64) as u64;
while false {
}
loop {
if lastLLSize > (((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64) {
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
if !op.is_null() {
ZSTD_memcpy!(op, litPtr, lastLLSize);
let __v30: *mut u8 = op;
op = __v30.wrapping_offset(lastLLSize as isize);
}
while false {
}
((unsafe { op.offset_from(ostart as *mut u8) }) as i64) as u64
}

fn ZSTD_decompressSequences_default(dctx: *mut ZSTD_DCtx_s, dst: *mut (), maxDstSize: u64, seqStart: *const (), seqSize: u64, nbSeq: i32, isLongOffset: u32) -> u64 {
ZSTD_decompressSequences_body(dctx, dst, maxDstSize, seqStart, seqSize, nbSeq, isLongOffset)
}

fn ZSTD_decompressSequencesSplitLitBuffer_default(dctx: *mut ZSTD_DCtx_s, dst: *mut (), maxDstSize: u64, seqStart: *const (), seqSize: u64, nbSeq: i32, isLongOffset: u32) -> u64 {
ZSTD_decompressSequences_bodySplitLitBuffer(dctx, dst, maxDstSize, seqStart, seqSize, nbSeq, isLongOffset)
}

#[inline(always)]
fn ZSTD_prefetchMatch(mut prefetchPos: u64, sequence: __SlateRecord_zstd_decompress_block_c_783, prefixStart: *const u8, dictEnd: *const u8) -> u64 {
prefetchPos = prefetchPos.wrapping_add(sequence.litLength);
let matchBase: *const u8 = if sequence.offset > prefetchPos { dictEnd } else { prefixStart };
// note : this operation can overflow when seq.offset is really too large, which can only happen when input is corrupted.
// No consequence though : memory address is only used for prefetching, not for dereferencing
let r#match: *const u8 = ZSTD_wrappedPtrSub(ZSTD_wrappedPtrAdd(matchBase, prefetchPos as i64), sequence.offset as i64);
PREFETCH_L1!(r#match);
PREFETCH_L1!(r#match.wrapping_offset(64 as isize)); // note : it's safe to invoke PREFETCH() on any memory address, including invalid ones
prefetchPos.wrapping_add(sequence.matchLength)
}

/// This decoding function employs prefetching
/// to reduce latency impact of cache misses.
/// It's generally employed when block contains a significant portion of long-distance matches
/// or when coupled with a "cold" dictionary
#[inline(always)]
fn ZSTD_decompressSequencesLong_body(dctx: *mut ZSTD_DCtx_s, dst: *mut (), maxDstSize: u64, seqStart: *const (), seqSize: u64, nbSeq: i32, isLongOffset: u32) -> u64 {
let ip: *const u8 = seqStart as *const u8;
let iend: *const u8 = ip.wrapping_offset(seqSize as isize);
let ostart: *mut u8 = dst as *mut u8;
let __v0: *mut u8 = if ((unsafe { (*dctx).litBufferLocation }) as u32) == 1 {
unsafe { (*dctx).litBuffer }
} else {
ZSTD_maybeNullPtrAdd(ostart, maxDstSize as i64)
};
let oend: *mut u8 = __v0;
let mut op: *mut u8 = ostart;
let mut litPtr: *const u8 = unsafe { (*dctx).litPtr };
let mut litBufferEnd: *const u8 = unsafe { (*dctx).litBufferEnd };
let prefixStart: *const u8 = (unsafe { (*dctx).prefixStart }) as *const u8;
let dictStart: *const u8 = (unsafe { (*dctx).virtualStart }) as *const u8;
let dictEnd: *const u8 = (unsafe { (*dctx).dictEnd }) as *const u8;
// Regen sequences
if nbSeq != 0 {
let mut sequences: __SlateAlign16<[__SlateRecord_zstd_decompress_block_c_783; 8]> = __SlateAlign16(unsafe { core::mem::zeroed() });
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sequences.0) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 192usize);
}
let seqAdvance: i32 = MIN!(nbSeq, 8i32);
let mut seqState: __SlateRecord_zstd_decompress_block_c_796 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(seqState) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 112usize);
}
// prepare in advance
let mut seqNb: i32 = 0;
let mut prefetchPos: u64 = ((unsafe { op.offset_from(prefixStart as *mut u8) }) as i64) as u64; // track position relative to prefixStart
unsafe {
(*dctx).fseEntropy = 1;
}
for i in 0i32..3i32 {
seqState.prevOffset[i as usize] = (unsafe { *((*dctx).entropy.rep.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) }) as u64;
}
loop {
if ERR_isError(BIT_initDStream(core::ptr::addr_of_mut!(seqState.DStream), ip as *const (), ((unsafe { iend.offset_from(ip as *const u8) }) as i64) as u64)) != 0 {
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
ZSTD_initFseState(core::ptr::addr_of_mut!(seqState.stateLL), core::ptr::addr_of_mut!(seqState.DStream), unsafe { (*dctx).LLTptr });
ZSTD_initFseState(core::ptr::addr_of_mut!(seqState.stateOffb), core::ptr::addr_of_mut!(seqState.DStream), unsafe { (*dctx).OFTptr });
ZSTD_initFseState(core::ptr::addr_of_mut!(seqState.stateML), core::ptr::addr_of_mut!(seqState.DStream), unsafe { (*dctx).MLTptr });
while seqNb < seqAdvance {
let sequence: __SlateRecord_zstd_decompress_block_c_783 = ZSTD_decodeSequence(unsafe { &mut (*core::ptr::addr_of_mut!(seqState)) }, isLongOffset, (seqNb == nbSeq.wrapping_sub(1)) as i32);
prefetchPos = ZSTD_prefetchMatch(prefetchPos, sequence, prefixStart, dictEnd);
sequences.0[seqNb as usize] = sequence;
seqNb = seqNb.wrapping_add(1);
}
// decompress without stomping litBuffer
while seqNb < nbSeq {
let sequence: __SlateRecord_zstd_decompress_block_c_783 = ZSTD_decodeSequence(unsafe { &mut (*core::ptr::addr_of_mut!(seqState)) }, isLongOffset, (seqNb == nbSeq.wrapping_sub(1)) as i32);
if ((unsafe { (*dctx).litBufferLocation }) as u32) == 2 && litPtr.wrapping_offset(sequences.0[(seqNb.wrapping_sub(8) & 7i32) as usize].litLength as isize) > unsafe { (*dctx).litBufferEnd } {
// lit buffer is reaching split point, empty out the first buffer and transition to litExtraBuffer
let leftoverLit: u64 = ((unsafe { unsafe { (*dctx).litBufferEnd }.offset_from(litPtr as *const u8) }) as i64) as u64;
if leftoverLit != 0 {
loop {
if leftoverLit > (((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64) {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"remaining lit must fit within dstBuffer".as_ptr() as *const u8) };
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
ZSTD_safecopyDstBeforeSrc(op, litPtr, leftoverLit as i64);
let __v17: *mut __SlateRecord_zstd_decompress_block_c_783 = (sequences.0.as_mut_ptr() as *mut __SlateRecord_zstd_decompress_block_c_783).wrapping_offset((seqNb.wrapping_sub(8) & 7i32) as isize);
let __v18: u64 = unsafe { (*__v17).litLength };
unsafe {
(*__v17).litLength = __v18.wrapping_sub(leftoverLit);
}
let __v20: *mut u8 = op;
op = __v20.wrapping_offset(leftoverLit as isize);
}
litPtr = (unsafe { (*dctx).litExtraBuffer.as_mut_ptr() as *mut u8 }) as *const u8;
litBufferEnd = unsafe { (*dctx).litExtraBuffer.as_mut_ptr() as *mut u8 }.wrapping_offset(BOUNDED!(64i32, 65536i32, 131072i32) as isize) as *const u8;
unsafe {
(*dctx).litBufferLocation = 0;
}
let oneSeqSize: u64 = ZSTD_execSequence(op, oend, sequences.0[(seqNb.wrapping_sub(8) & 7i32) as usize], unsafe { &mut (*core::ptr::addr_of_mut!(litPtr)) }, litBufferEnd, prefixStart, dictStart, dictEnd);
if ERR_isError(oneSeqSize) != 0 {
return oneSeqSize;
}
prefetchPos = ZSTD_prefetchMatch(prefetchPos, sequence, prefixStart, dictEnd);
sequences.0[(seqNb & 7i32) as usize] = sequence;
let __v22: *mut u8 = op;
op = __v22.wrapping_offset(oneSeqSize as isize);
} else {
// lit buffer is either wholly contained in first or second split, or not split at all
let mut oneSeqSize: u64 = 0;
let __v24: u64 = if ((unsafe { (*dctx).litBufferLocation }) as u32) == 2 {
ZSTD_execSequenceSplitLitBuffer(op, oend, litPtr.wrapping_offset(sequences.0[(seqNb.wrapping_sub(8) & 7i32) as usize].litLength as isize).wrapping_offset(-(32 as isize)), sequences.0[(seqNb.wrapping_sub(8) & 7i32) as usize], core::ptr::addr_of_mut!(litPtr), litBufferEnd, prefixStart, dictStart, dictEnd)
} else {
ZSTD_execSequence(op, oend, sequences.0[(seqNb.wrapping_sub(8) & 7i32) as usize], unsafe { &mut (*core::ptr::addr_of_mut!(litPtr)) }, litBufferEnd, prefixStart, dictStart, dictEnd)
};
oneSeqSize = __v24;
if ERR_isError(oneSeqSize) != 0 {
return oneSeqSize;
}
prefetchPos = ZSTD_prefetchMatch(prefetchPos, sequence, prefixStart, dictEnd);
sequences.0[(seqNb & 7i32) as usize] = sequence;
let __v25: *mut u8 = op;
op = __v25.wrapping_offset(oneSeqSize as isize);
}
seqNb = seqNb.wrapping_add(1);
}
loop {
if BIT_endOfDStream(unsafe { &(*(core::ptr::addr_of_mut!(seqState.DStream) as *const __SlateRecord_bitstream_h_90)) }) == 0 {
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
// finish queue
seqNb = seqNb.wrapping_sub(seqAdvance);
while seqNb < nbSeq {
let sequence: *mut __SlateRecord_zstd_decompress_block_c_783 = (sequences.0.as_mut_ptr() as *mut __SlateRecord_zstd_decompress_block_c_783).wrapping_offset((seqNb & 7i32) as isize);
if ((unsafe { (*dctx).litBufferLocation }) as u32) == 2 && litPtr.wrapping_offset((unsafe { (*sequence).litLength }) as isize) > unsafe { (*dctx).litBufferEnd } {
let leftoverLit: u64 = ((unsafe { unsafe { (*dctx).litBufferEnd }.offset_from(litPtr as *const u8) }) as i64) as u64;
if leftoverLit != 0 {
loop {
if leftoverLit > (((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64) {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"remaining lit must fit within dstBuffer".as_ptr() as *const u8) };
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
ZSTD_safecopyDstBeforeSrc(op, litPtr, leftoverLit as i64);
let __v43: u64 = unsafe { (*sequence).litLength };
unsafe {
(*sequence).litLength = __v43.wrapping_sub(leftoverLit);
}
let __v45: *mut u8 = op;
op = __v45.wrapping_offset(leftoverLit as isize);
}
litPtr = (unsafe { (*dctx).litExtraBuffer.as_mut_ptr() as *mut u8 }) as *const u8;
litBufferEnd = unsafe { (*dctx).litExtraBuffer.as_mut_ptr() as *mut u8 }.wrapping_offset(BOUNDED!(64i32, 65536i32, 131072i32) as isize) as *const u8;
unsafe {
(*dctx).litBufferLocation = 0;
}
let oneSeqSize: u64 = ZSTD_execSequence(op, oend, unsafe { *sequence }, unsafe { &mut (*core::ptr::addr_of_mut!(litPtr)) }, litBufferEnd, prefixStart, dictStart, dictEnd);
if ERR_isError(oneSeqSize) != 0 {
return oneSeqSize;
}
let __v47: *mut u8 = op;
op = __v47.wrapping_offset(oneSeqSize as isize);
} else {
let mut oneSeqSize: u64 = 0;
let __v49: u64 = if ((unsafe { (*dctx).litBufferLocation }) as u32) == 2 {
ZSTD_execSequenceSplitLitBuffer(op, oend, litPtr.wrapping_offset((unsafe { (*sequence).litLength }) as isize).wrapping_offset(-(32 as isize)), unsafe { *sequence }, core::ptr::addr_of_mut!(litPtr), litBufferEnd, prefixStart, dictStart, dictEnd)
} else {
ZSTD_execSequence(op, oend, unsafe { *sequence }, unsafe { &mut (*core::ptr::addr_of_mut!(litPtr)) }, litBufferEnd, prefixStart, dictStart, dictEnd)
};
oneSeqSize = __v49;
if ERR_isError(oneSeqSize) != 0 {
return oneSeqSize;
}
let __v50: *mut u8 = op;
op = __v50.wrapping_offset(oneSeqSize as isize);
}
seqNb = seqNb.wrapping_add(1);
}
// save reps for next block
for i in 0u32..3u32 {
unsafe {
*((*dctx).entropy.rep.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) = seqState.prevOffset[i as usize] as u32;
}
}
}
// last literal segment
if ((unsafe { (*dctx).litBufferLocation }) as u32) == 2 {
// first deplete literal buffer in dst, then copy litExtraBuffer
let lastLLSize: u64 = ((unsafe { litBufferEnd.offset_from(litPtr as *const u8) }) as i64) as u64;
loop {
if lastLLSize > (((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64) {
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
if !op.is_null() {
ZSTD_memmove!(op, litPtr, lastLLSize);
let __v63: *mut u8 = op;
op = __v63.wrapping_offset(lastLLSize as isize);
}
litPtr = (unsafe { (*dctx).litExtraBuffer.as_mut_ptr() as *mut u8 }) as *const u8;
litBufferEnd = unsafe { (*dctx).litExtraBuffer.as_mut_ptr() as *mut u8 }.wrapping_offset(BOUNDED!(64i32, 65536i32, 131072i32) as isize) as *const u8;
}
let lastLLSize: u64 = ((unsafe { litBufferEnd.offset_from(litPtr as *const u8) }) as i64) as u64;
loop {
if lastLLSize > (((unsafe { oend.offset_from(op as *mut u8) }) as i64) as u64) {
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
if !op.is_null() {
ZSTD_memmove!(op, litPtr, lastLLSize);
let __v70: *mut u8 = op;
op = __v70.wrapping_offset(lastLLSize as isize);
}
((unsafe { op.offset_from(ostart as *mut u8) }) as i64) as u64
}

fn ZSTD_decompressSequencesLong_default(dctx: *mut ZSTD_DCtx_s, dst: *mut (), maxDstSize: u64, seqStart: *const (), seqSize: u64, nbSeq: i32, isLongOffset: u32) -> u64 {
ZSTD_decompressSequencesLong_body(dctx, dst, maxDstSize, seqStart, seqSize, nbSeq, isLongOffset)
}

#[target_feature(enable = "lzcnt,bmi1,bmi2")]
unsafe fn ZSTD_decompressSequences_bmi2(dctx: *mut ZSTD_DCtx_s, dst: *mut (), maxDstSize: u64, seqStart: *const (), seqSize: u64, nbSeq: i32, isLongOffset: u32) -> u64 {
ZSTD_decompressSequences_body(dctx, dst, maxDstSize, seqStart, seqSize, nbSeq, isLongOffset)
}

#[target_feature(enable = "lzcnt,bmi1,bmi2")]
unsafe fn ZSTD_decompressSequencesSplitLitBuffer_bmi2(dctx: *mut ZSTD_DCtx_s, dst: *mut (), maxDstSize: u64, seqStart: *const (), seqSize: u64, nbSeq: i32, isLongOffset: u32) -> u64 {
ZSTD_decompressSequences_bodySplitLitBuffer(dctx, dst, maxDstSize, seqStart, seqSize, nbSeq, isLongOffset)
}

#[target_feature(enable = "lzcnt,bmi1,bmi2")]
unsafe fn ZSTD_decompressSequencesLong_bmi2(dctx: *mut ZSTD_DCtx_s, dst: *mut (), maxDstSize: u64, seqStart: *const (), seqSize: u64, nbSeq: i32, isLongOffset: u32) -> u64 {
ZSTD_decompressSequencesLong_body(dctx, dst, maxDstSize, seqStart, seqSize, nbSeq, isLongOffset)
}

fn ZSTD_decompressSequences(dctx: *mut ZSTD_DCtx_s, dst: *mut (), maxDstSize: u64, seqStart: *const (), seqSize: u64, nbSeq: i32, isLongOffset: u32) -> u64 {
while false {
}
if ZSTD_DCtx_get_bmi2(dctx as *const ZSTD_DCtx_s) != 0 {
return unsafe { ZSTD_decompressSequences_bmi2(dctx, dst, maxDstSize, seqStart, seqSize, nbSeq, isLongOffset) };
}
ZSTD_decompressSequences_default(dctx, dst, maxDstSize, seqStart, seqSize, nbSeq, isLongOffset)
}

fn ZSTD_decompressSequencesSplitLitBuffer(dctx: *mut ZSTD_DCtx_s, dst: *mut (), maxDstSize: u64, seqStart: *const (), seqSize: u64, nbSeq: i32, isLongOffset: u32) -> u64 {
while false {
}
if ZSTD_DCtx_get_bmi2(dctx as *const ZSTD_DCtx_s) != 0 {
return unsafe { ZSTD_decompressSequencesSplitLitBuffer_bmi2(dctx, dst, maxDstSize, seqStart, seqSize, nbSeq, isLongOffset) };
}
ZSTD_decompressSequencesSplitLitBuffer_default(dctx, dst, maxDstSize, seqStart, seqSize, nbSeq, isLongOffset)
}

/// ZSTD_decompressSequencesLong() :
/// decompression function triggered when a minimum share of offsets is considered "long",
/// aka out of cache.
/// note : "long" definition seems overloaded here, sometimes meaning "wider than bitstream register", and sometimes meaning "farther than memory cache distance".
/// This function will try to mitigate main memory latency through the use of prefetching
fn ZSTD_decompressSequencesLong(dctx: *mut ZSTD_DCtx_s, dst: *mut (), maxDstSize: u64, seqStart: *const (), seqSize: u64, nbSeq: i32, isLongOffset: u32) -> u64 {
while false {
}
if ZSTD_DCtx_get_bmi2(dctx as *const ZSTD_DCtx_s) != 0 {
return unsafe { ZSTD_decompressSequencesLong_bmi2(dctx, dst, maxDstSize, seqStart, seqSize, nbSeq, isLongOffset) };
}
ZSTD_decompressSequencesLong_default(dctx, dst, maxDstSize, seqStart, seqSize, nbSeq, isLongOffset)
}

/// @returns The total size of the history referenceable by zstd, including
/// both the prefix and the extDict. At @p op any offset larger than this
/// is invalid.
fn ZSTD_totalHistorySize(op: *mut u8, virtualStart: *const u8) -> u64 {
((unsafe { op.offset_from(virtualStart as *mut u8) }) as i64) as u64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_decompress_block_c_2005 {
    longOffsetShare: u32,
    maxNbAdditionalBits: u32,
}


/// ZSTD_getOffsetInfo() :
/// condition : offTable must be valid
/// @return : "share" of long offsets (arbitrarily defined as > (1<<23))
///           compared to maximum possible of (1<<OffFSELog),
///           as well as the maximum number additional bits required.
fn ZSTD_getOffsetInfo(offTable: *const __SlateRecord_zstd_decompress_internal_h_73, nbSeq: i32) -> __SlateRecord_zstd_decompress_block_c_2005 {
let mut info: __SlateRecord_zstd_decompress_block_c_2005 = __SlateRecord_zstd_decompress_block_c_2005 { longOffsetShare: 0, maxNbAdditionalBits: 0 };
// If nbSeq == 0, then the offTable is uninitialized, but we have
// no sequences, so both values should be 0.
if nbSeq != 0 {
let ptr: *const () = offTable as *const ();
let tableLog: u32 = unsafe { (*(ptr as *const __SlateRecord_zstd_decompress_internal_h_66).wrapping_offset(0 as isize)).tableLog };
let table: *const __SlateRecord_zstd_decompress_internal_h_73 = offTable.wrapping_offset(1 as isize);
let max: u32 = (1i32 << tableLog) as u32;
while false {
}
// max not too large
for u in 0..max {
info.maxNbAdditionalBits = MAX__4!(info.maxNbAdditionalBits, unsafe { (*table.wrapping_offset(u as isize)).nbAdditionalBits });
if ((unsafe { (*table.wrapping_offset(u as isize)).nbAdditionalBits }) as i32) > 22 {
let __v1: u32 = info.longOffsetShare;
info.longOffsetShare = __v1.wrapping_add(1);
}
}
info.longOffsetShare <<= 8u32.wrapping_sub(tableLog); // scale to OffFSELog
}
return info;
}

/// @returns The maximum offset we can decode in one read of our bitstream, without
/// reloading more bits in the middle of the offset bits read. Any offsets larger
/// than this must use the long offset decoder.
fn ZSTD_maxShortOffset() -> u64 {
if MEM_64bits() != 0 {
// We can decode any offset without reloading bits.
// This might change if the max window size grows.
ZSTD_STATIC_ASSERT!();
return 1i32.wrapping_neg() as u64;
} else {
// The maximum offBase is (1 << (STREAM_ACCUMULATOR_MIN + 1)) - 1.
// This offBase would require STREAM_ACCUMULATOR_MIN extra bits.
// Then we have to subtract ZSTD_REP_NUM to get the maximum possible offset.
let maxOffbase: u64 = (1u64 << ((if MEM_32bits() != 0 { 25i32 } else { 57i32 }) as u32).wrapping_add(1)).wrapping_sub(1);
let maxOffset: u64 = maxOffbase.wrapping_sub(3);
return maxOffset;
}
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_decompressBlock_internal(dctx: *mut ZSTD_DCtx_s, dst: *mut (), dstCapacity: u64, src: *const (), mut srcSize: u64, streaming: u32) -> u64 {
// blockType == blockCompressed
let mut ip: *const u8 = src as *const u8;
while false {
}
// Note : the wording of the specification
// allows compressed block to be sized exactly ZSTD_blockSizeMax(dctx).
// This generally does not happen, as it makes little sense,
// since an uncompressed block would feature same size and have no decompression cost.
// Also, note that decoder from reference libzstd before < v1.5.4
// would consider this edge case as an error.
// As a consequence, avoid generating compressed blocks of size ZSTD_blockSizeMax(dctx)
// for broader compatibility with the deployed ecosystem of zstd decoders
loop {
if srcSize > ZSTD_blockSizeMax(dctx as *const ZSTD_DCtx_s) {
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
// Decode literals section
let litCSize: u64 = ZSTD_decodeLiteralsBlock(dctx, src, srcSize, dst, dstCapacity, streaming);
while false {
}
if ERR_isError(litCSize) != 0 {
return litCSize;
}
let __v7: *const u8 = ip;
ip = __v7.wrapping_offset(litCSize as isize);
srcSize = srcSize.wrapping_sub(litCSize);
// Build Decoding Tables
// Compute the maximum block size, which must also work when !frame and fParams are unset.
// Additionally, take the min with dstCapacity to ensure that the totalHistorySize fits in a size_t.
let blockSizeMax: u64 = MIN__3!(dstCapacity, ZSTD_blockSizeMax(dctx as *const ZSTD_DCtx_s));
let totalHistorySize: u64 = ZSTD_totalHistorySize(ZSTD_maybeNullPtrAdd(dst as *mut u8, blockSizeMax as i64), (unsafe { (*dctx).virtualStart }) as *const u8);
// isLongOffset must be true if there are long offsets.
// Offsets are long if they are larger than ZSTD_maxShortOffset().
// We don't expect that to be the case in 64-bit mode.
//
// We check here to see if our history is large enough to allow long offsets.
// If it isn't, then we can't possible have (valid) long offsets. If the offset
// is invalid, then it is okay to read it incorrectly.
//
// If isLongOffsets is true, then we will later check our decoding table to see
// if it is even possible to generate long offsets.
let mut isLongOffset: u32 = 0;
let __v12: bool = MEM_32bits() != 0 && totalHistorySize > ZSTD_maxShortOffset();
isLongOffset = __v12 as u32;
// These macros control at build-time which decompressor implementation
// we use. If neither is defined, we do some inspection and dispatch at
// runtime.
let mut usePrefetchDecoder: i32 = unsafe { (*dctx).ddictIsCold };
let mut nbSeq: i32 = 0;
let seqHSize: u64 = ZSTD_decodeSeqHeaders(dctx, core::ptr::addr_of_mut!(nbSeq), ip as *const (), srcSize);
if ERR_isError(seqHSize) != 0 {
return seqHSize;
}
let __v13: *const u8 = ip;
ip = __v13.wrapping_offset(seqHSize as isize);
srcSize = srcSize.wrapping_sub(seqHSize);
loop {
if (dst.is_null() || dstCapacity == 0) && nbSeq > 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"NULL not handled".as_ptr() as *const u8) };
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
loop {
if MEM_64bits() != 0 && (core::mem::size_of::<u64>() as u64) == (core::mem::size_of::<*mut ()>() as u64) && (1i32.wrapping_neg() as u64).wrapping_sub(dst as u64) < 1048576u64 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"invalid dst".as_ptr() as *const u8) };
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
// If we could potentially have long offsets, or we might want to use the prefetch decoder,
// compute information about the share of long offsets, and the maximum nbAdditionalBits.
// NOTE: could probably use a larger nbSeq limit
if (isLongOffset as u32) != 0 || usePrefetchDecoder == 0 && totalHistorySize > 16777216u64 && nbSeq > 8 {
let info: __SlateRecord_zstd_decompress_block_c_2005 = ZSTD_getOffsetInfo(unsafe { (*dctx).OFTptr }, nbSeq);
let __v27: bool = (isLongOffset as u32) != 0 && info.maxNbAdditionalBits <= ((if MEM_32bits() != 0 { 25i32 } else { 57i32 }) as u32);
if __v27 {
// If isLongOffset, but the maximum number of additional bits that we see in our table is small
// enough, then we know it is impossible to have too long an offset in this block, so we can
// use the regular offset decoder.
isLongOffset = 0;
}
if usePrefetchDecoder == 0 {
let minShare: u32 = (if MEM_64bits() != 0 { 7i32 } else { 20i32 }) as u32; // heuristic values, correspond to 2.73% and 7.81%
usePrefetchDecoder = (info.longOffsetShare >= minShare) as i32;
}
}
unsafe {
(*dctx).ddictIsCold = 0;
}
if usePrefetchDecoder != 0 {
return ZSTD_decompressSequencesLong(dctx, dst, dstCapacity, ip as *const (), srcSize, nbSeq, isLongOffset);
}
// else
if ((unsafe { (*dctx).litBufferLocation }) as u32) == 2 {
return ZSTD_decompressSequencesSplitLitBuffer(dctx, dst, dstCapacity, ip as *const (), srcSize, nbSeq, isLongOffset);
} else {
return ZSTD_decompressSequences(dctx, dst, dstCapacity, ip as *const (), srcSize, nbSeq, isLongOffset);
}
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_checkContinuity(dctx: *mut ZSTD_DCtx_s, dst: *const (), dstSize: u64) {
if dst != unsafe { (*dctx).previousDstEnd } && dstSize > 0 {
// not contiguous
unsafe {
(*dctx).dictEnd = (*dctx).previousDstEnd;
}
unsafe {
(*dctx).virtualStart = (dst as *const u8).wrapping_offset(-((((*dctx).previousDstEnd as *const u8).offset_from((*dctx).prefixStart as *const u8) as i64) as isize)) as *const ();
}
unsafe {
(*dctx).prefixStart = dst;
}
unsafe {
(*dctx).previousDstEnd = dst;
}
}
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_decompressBlock_deprecated(dctx: *mut ZSTD_DCtx_s, dst: *mut (), dstCapacity: u64, src: *const (), srcSize: u64) -> u64 {
let mut dSize: u64 = 0;
unsafe {
(*dctx).isFrameDecompression = 0;
}
ZSTD_checkContinuity(dctx, dst as *const (), dstCapacity);
dSize = ZSTD_decompressBlock_internal(dctx, dst, dstCapacity, src, srcSize, 0u32);
loop {
let err_code: u64 = dSize;
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
(*dctx).previousDstEnd = (dst as *mut u8).wrapping_offset(dSize as isize) as *const ();
}
dSize
}

/// NOTE: Must just wrap ZSTD_decompressBlock_deprecated()
#[unsafe(no_mangle)]
extern "C" fn ZSTD_decompressBlock(dctx: *mut ZSTD_DCtx_s, dst: *mut (), dstCapacity: u64, src: *const (), srcSize: u64) -> u64 {
ZSTD_decompressBlock_deprecated(dctx, dst, dstCapacity, src, srcSize)
}
