
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0+ OR BSD-3-Clause
//!
//! Common functions of New Generation Entropy library
//! Copyright (c) Meta Platforms, Inc. and affiliates.
//!
//!  You can contact the author at :
//!  - FSE+HUF source repository : https://github.com/Cyan4973/FiniteStateEntropy
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
//! static assert is triggered at compile time, leaving no runtime artefact.
//! static assert only works with compile-time constants.
//! Also, this variant can only be used inside a function.
//! UNUSED_ATTR tells the compiler it is okay if the function is unused.
//! Compiler-specific
//! *** Constants ***
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
// __put_unaligned_t - write an unaligned value to memory.
// @type:	the type of the value to store.
// @val:	the value to store.
// @ptr:	the pointer to store to.
//
// Use memcpy to affect an unaligned type sized store avoiding undefined
// behavior from approaches like type punning that require -fno-strict-aliasing
// in order to be correct. The void* cast silences ubsan warnings.
// disable assert (default)
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
// max runtime value of tableLog (due to static allocation); can be modified up to HUF_TABLELOG_ABSOLUTEMAX
// HUF_readStats_wksp() :
// Same as HUF_readStats() but takes an external workspace which must be
// 4-byte aligned and its size must be >= HUF_READ_STATS_WORKSPACE_SIZE.
// If the CPU has BMI2 support, pass bmi2=1, otherwise pass bmi2=0.
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

macro_rules! ERROR__21 {
() => {
ZSTD_ERROR__15!()
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

macro_rules! ZSTD_ERROR__15 {
() => {
48i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__3 {
() => {
72i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR {
() => {
120i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_memcpy__2 {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memcpy(($d) as *mut (), $s, $n) };
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

unsafe extern "C" {
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn ERR_getErrorString(arg0: u32) -> *const u8;
    fn FSE_decompress_wksp_bmi2(arg0: *mut (), arg1: u64, arg2: *const (), arg3: u64, arg4: u32, arg5: *mut (), arg6: u64, arg7: i32) -> u64;
    #[link_name = "memset"]
    fn __builtin_memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
}

#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

// Optional: only supported since gcc >= 15, clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
fn get_unaligned_le32(p: *const ()) -> u32 {
__get_unaligned_t!(u32, p)
}

fn MEM_readLE32(memPtr: *const ()) -> u32 {
get_unaligned_le32(memPtr)
}

fn ERR_isError(code: u64) -> u32 {
(code > ERROR__8!()) as u32
}

fn ERR_getErrorCode(code: u64) -> u32 {
if ERR_isError(code) == 0 {
return 0;
}
0u64.wrapping_sub(code) as u32
}

fn ERR_getErrorName(code: u64) -> *const u8 {
unsafe { ERR_getErrorString(ERR_getErrorCode(code)) }
}

fn ZSTD_countTrailingZeros32(val: u32) -> u32 {
(val.trailing_zeros() as i32) as u32
}

fn ZSTD_countLeadingZeros32(val: u32) -> u32 {
(val.leading_zeros() as i32) as u32
}

fn ZSTD_highbit32(val: u32) -> u32 {
31u32.wrapping_sub(ZSTD_countLeadingZeros32(val))
}

// Dependencies
// ERR_*, ERROR
// FSE_MIN_TABLELOG
// ZSDT_highbit32, ZSTD_countTrailingZeros32
/// ===   Version   ===
#[unsafe(no_mangle)]
extern "C" fn FSE_versionNumber() -> u32 {
900u32
}

/// ===   Error Management   ===
#[unsafe(no_mangle)]
extern "C" fn FSE_isError(code: u64) -> u32 {
ERR_isError(code)
}

#[unsafe(no_mangle)]
extern "C" fn FSE_getErrorName(code: u64) -> *const u8 {
ERR_getErrorName(code)
}

#[unsafe(no_mangle)]
extern "C" fn HUF_isError(code: u64) -> u32 {
ERR_isError(code)
}

#[unsafe(no_mangle)]
extern "C" fn HUF_getErrorName(code: u64) -> *const u8 {
ERR_getErrorName(code)
}

// FSE NCount encoding-decoding
#[inline(always)]
fn FSE_readNCount_body(normalizedCounter: *mut i16, maxSVPtr: *mut u32, tableLogPtr: *mut u32, headerBuffer: *const (), hbSize: u64) -> u64 {
let istart: *const u8 = headerBuffer as *const u8;
let iend: *const u8 = istart.wrapping_offset(hbSize as isize);
let mut ip: *const u8 = istart;
let mut nbBits: i32 = 0;
let mut remaining: i32 = 0;
let mut threshold: i32 = 0;
let mut bitStream: u32 = 0;
let mut bitCount: i32 = 4;
let mut charnum: u32 = 0;
let maxSV1: u32 = unsafe { *maxSVPtr }.wrapping_add(1);
let mut previous0: i32 = 0;
if hbSize < 8 {
// This function only works when hbSize >= 8
let mut buffer: [u8; 8] = {
let mut __t0: [u8; 8] = unsafe { core::mem::zeroed() };
__t0[0] = 0u8;
__t0
};
ZSTD_memcpy__2!(buffer.as_mut_ptr() as *mut u8, headerBuffer, hbSize);
let countSize: u64 = FSE_readNCount(normalizedCounter, maxSVPtr, tableLogPtr, buffer.as_mut_ptr() as *const (), core::mem::size_of::<[u8; 8]>() as u64);
if FSE_isError(countSize) != 0 {
return countSize;
}
if countSize > hbSize {
return ERROR!();
}
return countSize;
}
// init
ZSTD_memset!(normalizedCounter, 0i32, (unsafe { *maxSVPtr }.wrapping_add(1) as u64).wrapping_mul(core::mem::size_of::<i16>() as u64)); // all symbols not present in NCount have a frequency of 0
bitStream = MEM_readLE32(ip as *const ());
nbBits = (bitStream & 15).wrapping_add(5) as i32; // extract tableLog
if nbBits > 15 {
return ERROR__6!();
}
bitStream >>= 4i32;
unsafe {
*tableLogPtr = nbBits as u32;
}
remaining = (1i32 << nbBits).wrapping_add(1);
threshold = 1i32 << nbBits;
nbBits = nbBits.wrapping_add(1);
loop {
if previous0 != 0 {
// Count the number of repeats. Each time the
// 2-bit repeat code is 0b11 there is another
// repeat.
// Avoid UB by setting the high bit to 1.
let mut repeats: i32 = (ZSTD_countTrailingZeros32(!bitStream | 2147483648) >> 1i32) as i32;
while repeats >= 12 {
charnum = charnum.wrapping_add(36u32);
if LIKELY!(ip <= iend.wrapping_offset(-(7 as isize))) != 0 {
let __v6: *const u8 = ip;
ip = __v6.wrapping_offset(3 as isize);
} else {
let __v8: i32 = bitCount;
bitCount = __v8.wrapping_sub(8i64.wrapping_mul((unsafe { iend.wrapping_offset(-(7 as isize)).offset_from(ip as *const u8) }) as i64) as i32);
bitCount &= 31;
ip = iend.wrapping_offset(-(4 as isize));
}
bitStream = MEM_readLE32(ip as *const ()) >> bitCount;
repeats = (ZSTD_countTrailingZeros32(!bitStream | 2147483648) >> 1i32) as i32;
}
charnum = charnum.wrapping_add(3i32.wrapping_mul(repeats) as u32);
bitStream >>= 2i32.wrapping_mul(repeats);
bitCount = bitCount.wrapping_add(2i32.wrapping_mul(repeats));
// Add the final repeat which isn't 0b11.
charnum = charnum.wrapping_add(bitStream & 3);
bitCount = bitCount.wrapping_add(2);
// This is an error, but break and return an error
// at the end, because returning out of a loop makes
// it harder for the compiler to optimize.
if charnum >= maxSV1 {
break;
}
// We don't need to set the normalized count to 0
// because we already memset the whole buffer to 0.
if LIKELY!(ip <= iend.wrapping_offset(-(7 as isize))) != 0 || ip.wrapping_offset((bitCount >> 3i32) as isize) <= iend.wrapping_offset(-(4 as isize)) {
// For first condition to work
let __v24: *const u8 = ip;
ip = __v24.wrapping_offset((bitCount >> 3i32) as isize);
bitCount &= 7;
} else {
let __v28: i32 = bitCount;
bitCount = __v28.wrapping_sub(8i64.wrapping_mul((unsafe { iend.wrapping_offset(-(4 as isize)).offset_from(ip as *const u8) }) as i64) as i32);
bitCount &= 31;
ip = iend.wrapping_offset(-(4 as isize));
}
bitStream = MEM_readLE32(ip as *const ()) >> bitCount;
}
let max: i32 = 2i32.wrapping_mul(threshold).wrapping_sub(1).wrapping_sub(remaining);
let mut count: i32 = 0;
if bitStream & (threshold.wrapping_sub(1) as u32) < (max as u32) {
count = (bitStream & (threshold.wrapping_sub(1) as u32)) as i32;
bitCount = bitCount.wrapping_add(nbBits.wrapping_sub(1));
} else {
count = (bitStream & (2i32.wrapping_mul(threshold).wrapping_sub(1) as u32)) as i32;
if count >= threshold {
count = count.wrapping_sub(max);
}
bitCount = bitCount.wrapping_add(nbBits);
}
count = count.wrapping_sub(1); // extra accuracy
// When it matters (small blocks), this is a
// predictable branch, because we don't use -1.
if count >= 0 {
remaining = remaining.wrapping_sub(count);
} else {
remaining = remaining.wrapping_add(count);
}
let __v44: u32 = charnum;
charnum = __v44.wrapping_add(1);
unsafe {
*normalizedCounter.wrapping_offset(__v44 as isize) = count as i16;
}
previous0 = (count == 0) as i32;
if remaining < threshold {
// This branch can be folded into the
// threshold update condition because we
// know that threshold > 1.
if remaining <= 1 {
break;
}
nbBits = ZSTD_highbit32(remaining as u32).wrapping_add(1) as i32;
threshold = 1i32 << nbBits.wrapping_sub(1);
}
if charnum >= maxSV1 {
break;
}
if LIKELY!(ip <= iend.wrapping_offset(-(7 as isize))) != 0 || ip.wrapping_offset((bitCount >> 3i32) as isize) <= iend.wrapping_offset(-(4 as isize)) {
let __v46: *const u8 = ip;
ip = __v46.wrapping_offset((bitCount >> 3i32) as isize);
bitCount &= 7;
} else {
let __v50: i32 = bitCount;
bitCount = __v50.wrapping_sub(8i64.wrapping_mul((unsafe { iend.wrapping_offset(-(4 as isize)).offset_from(ip as *const u8) }) as i64) as i32);
bitCount &= 31;
ip = iend.wrapping_offset(-(4 as isize));
}
bitStream = MEM_readLE32(ip as *const ()) >> bitCount;
}
if remaining != 1 {
return ERROR!();
}
// Only possible when there are too many zeros.
if charnum > maxSV1 {
return ERROR__21!();
}
if bitCount > 32 {
return ERROR!();
}
unsafe {
*maxSVPtr = charnum.wrapping_sub(1);
}
let __v54: *const u8 = ip;
ip = __v54.wrapping_offset((bitCount.wrapping_add(7) >> 3i32) as isize);
((unsafe { ip.offset_from(istart as *const u8) }) as i64) as u64
}

/// Avoids the FORCE_INLINE of the _body() function.
fn FSE_readNCount_body_default(normalizedCounter: *mut i16, maxSVPtr: *mut u32, tableLogPtr: *mut u32, headerBuffer: *const (), hbSize: u64) -> u64 {
FSE_readNCount_body(normalizedCounter, maxSVPtr, tableLogPtr, headerBuffer, hbSize)
}

#[target_feature(enable = "lzcnt,bmi1,bmi2")]
unsafe fn FSE_readNCount_body_bmi2(normalizedCounter: *mut i16, maxSVPtr: *mut u32, tableLogPtr: *mut u32, headerBuffer: *const (), hbSize: u64) -> u64 {
FSE_readNCount_body(normalizedCounter, maxSVPtr, tableLogPtr, headerBuffer, hbSize)
}

#[unsafe(no_mangle)]
extern "C" fn FSE_readNCount_bmi2(normalizedCounter: *mut i16, maxSVPtr: *mut u32, tableLogPtr: *mut u32, headerBuffer: *const (), hbSize: u64, bmi2: i32) -> u64 {
if bmi2 != 0 {
return unsafe { FSE_readNCount_body_bmi2(normalizedCounter, maxSVPtr, tableLogPtr, headerBuffer, hbSize) };
}
FSE_readNCount_body_default(normalizedCounter, maxSVPtr, tableLogPtr, headerBuffer, hbSize)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.entropy_common.FSE_readNCount")]
extern "C" fn FSE_readNCount(normalizedCounter: *mut i16, maxSVPtr: *mut u32, tableLogPtr: *mut u32, headerBuffer: *const (), hbSize: u64) -> u64 {
return FSE_readNCount_bmi2(normalizedCounter, maxSVPtr, tableLogPtr, headerBuffer, hbSize, 0i32); // bmi2
}

/// HUF_readStats() :
/// Read compact Huffman tree, saved by HUF_writeCTable().
/// `huffWeight` is destination buffer.
/// `rankStats` is assumed to be a table of at least HUF_TABLELOG_MAX U32.
/// @return : size read from `src` , or an error Code .
/// Note : Needed by HUF_readCTable() and HUF_readDTableX?() .
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.entropy_common.HUF_readStats")]
extern "C" fn HUF_readStats(huffWeight: *mut u8, hwSize: u64, rankStats: *mut u32, nbSymbolsPtr: *mut u32, tableLogPtr: *mut u32, src: *const (), srcSize: u64) -> u64 {
let mut wksp: __SlateAlign16<[u32; 219]> = __SlateAlign16([0u32; 219]);
return HUF_readStats_wksp(huffWeight, hwSize, rankStats, nbSymbolsPtr, tableLogPtr, src, srcSize, wksp.0.as_mut_ptr() as *mut (), core::mem::size_of::<[u32; 219]>() as u64, 0i32); // flags
}

#[inline(always)]
fn HUF_readStats_body(huffWeight: *mut u8, hwSize: u64, rankStats: *mut u32, nbSymbolsPtr: *mut u32, tableLogPtr: *mut u32, src: *const (), srcSize: u64, workSpace: *mut (), wkspSize: u64, bmi2: i32) -> u64 {
let mut weightTotal: u32 = 0;
let mut ip: *const u8 = src as *const u8;
let mut iSize: u64 = 0;
let mut oSize: u64 = 0;
if srcSize == 0 {
return ERROR__2!();
}
iSize = (unsafe { *ip.wrapping_offset(0 as isize) }) as u64;
// ZSTD_memset(huffWeight, 0, hwSize);
//
// is not necessary, even though some analyzer complain ...
if iSize >= 128 {
// special header
oSize = iSize.wrapping_sub(127);
iSize = oSize.wrapping_add(1) / 2;
if iSize.wrapping_add(1) > srcSize {
return ERROR__2!();
}
if oSize >= hwSize {
return ERROR!();
}
let __v0: *const u8 = ip;
ip = __v0.wrapping_offset(1 as isize);
let mut n: u32 = 0;
while (n as u64) < oSize {
unsafe {
*huffWeight.wrapping_offset(n as isize) = ((*ip.wrapping_offset((n / 2) as isize) as i32) >> 4i32) as u8;
}
unsafe {
*huffWeight.wrapping_offset(n.wrapping_add(1) as isize) = ((*ip.wrapping_offset((n / 2) as isize) as i32) & 15) as u8;
}
n = n.wrapping_add(2);
}
} else {
// header compressed with FSE (normal case)
if iSize.wrapping_add(1) > srcSize {
return ERROR__2!();
}
// max (hwSize-1) values decoded, as last one is implied
oSize = unsafe { FSE_decompress_wksp_bmi2(huffWeight as *mut (), hwSize.wrapping_sub(1), ip.wrapping_offset(1 as isize) as *const (), iSize, 6u32, workSpace, wkspSize, bmi2) };
if FSE_isError(oSize) != 0 {
return oSize;
}
}
// collect weight stats
ZSTD_memset!(rankStats, 0i32, 13u64.wrapping_mul(core::mem::size_of::<u32>() as u64));
let mut n: u32 = 0;
while (n as u64) < oSize {
if ((unsafe { *huffWeight.wrapping_offset(n as isize) }) as i32) > 12 {
return ERROR!();
}
let __v5: *mut u32 = rankStats.wrapping_offset(((unsafe { *huffWeight.wrapping_offset(n as isize) }) as i32) as isize);
let __v6: u32 = unsafe { *__v5 };
unsafe {
*__v5 = __v6.wrapping_add(1);
}
weightTotal = weightTotal.wrapping_add((1i32 << ((unsafe { *huffWeight.wrapping_offset(n as isize) }) as i32) >> 1i32) as u32);
n = n.wrapping_add(1);
}
if weightTotal == 0 {
return ERROR!();
}
// get last non-null symbol weight (implied, total must be 2^n)
let tableLog: u32 = ZSTD_highbit32(weightTotal).wrapping_add(1);
if tableLog > 12 {
return ERROR!();
}
unsafe {
*tableLogPtr = tableLog;
}
// determine last weight
let total: u32 = (1i32 << tableLog) as u32;
let rest: u32 = total.wrapping_sub(weightTotal);
let verif: u32 = (1i32 << ZSTD_highbit32(rest)) as u32;
let lastWeight: u32 = ZSTD_highbit32(rest).wrapping_add(1);
if verif != rest {
return ERROR!();
}
// last value must be a clean power of 2
unsafe {
*huffWeight.wrapping_offset(oSize as isize) = lastWeight as u8;
}
let __v13: *mut u32 = rankStats.wrapping_offset(lastWeight as isize);
let __v14: u32 = unsafe { *__v13 };
unsafe {
*__v13 = __v14.wrapping_add(1);
}
// check tree construction validity
if (unsafe { *rankStats.wrapping_offset(1 as isize) }) < 2 || (unsafe { *rankStats.wrapping_offset(1 as isize) }) & 1 != 0 {
return ERROR!();
}
// by construction : at least 2 elts of rank 1, must be even
// results
unsafe {
*nbSymbolsPtr = oSize.wrapping_add(1) as u32;
}
iSize.wrapping_add(1)
}

/// Avoids the FORCE_INLINE of the _body() function.
fn HUF_readStats_body_default(huffWeight: *mut u8, hwSize: u64, rankStats: *mut u32, nbSymbolsPtr: *mut u32, tableLogPtr: *mut u32, src: *const (), srcSize: u64, workSpace: *mut (), wkspSize: u64) -> u64 {
HUF_readStats_body(huffWeight, hwSize, rankStats, nbSymbolsPtr, tableLogPtr, src, srcSize, workSpace, wkspSize, 0i32)
}

#[target_feature(enable = "lzcnt,bmi1,bmi2")]
unsafe fn HUF_readStats_body_bmi2(huffWeight: *mut u8, hwSize: u64, rankStats: *mut u32, nbSymbolsPtr: *mut u32, tableLogPtr: *mut u32, src: *const (), srcSize: u64, workSpace: *mut (), wkspSize: u64) -> u64 {
HUF_readStats_body(huffWeight, hwSize, rankStats, nbSymbolsPtr, tableLogPtr, src, srcSize, workSpace, wkspSize, 1i32)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.entropy_common.HUF_readStats_wksp")]
extern "C" fn HUF_readStats_wksp(huffWeight: *mut u8, hwSize: u64, rankStats: *mut u32, nbSymbolsPtr: *mut u32, tableLogPtr: *mut u32, src: *const (), srcSize: u64, workSpace: *mut (), wkspSize: u64, flags: i32) -> u64 {
if flags & 1 != 0 {
return unsafe { HUF_readStats_body_bmi2(huffWeight, hwSize, rankStats, nbSymbolsPtr, tableLogPtr, src, srcSize, workSpace, wkspSize) };
}
HUF_readStats_body_default(huffWeight, hwSize, rankStats, nbSymbolsPtr, tableLogPtr, src, srcSize, workSpace, wkspSize)
}
