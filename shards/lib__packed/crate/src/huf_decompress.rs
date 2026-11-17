
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0+ OR BSD-3-Clause
//!
//! huff0 huffman decoder,
//! part of Finite State Entropy library
//! Copyright (c) Meta Platforms, Inc. and affiliates.
//!
//!  You can contact the author at :
//!  - FSE+HUF source repository : https://github.com/Cyan4973/FiniteStateEntropy
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
//! Compiler-specific
//! *** Constants ***
//! cpuid(1): Processor Info and Feature Bits.
//! number of repcodes
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
// max runtime value of tableLog (due to static allocation); can be modified up to HUF_TABLELOG_ABSOLUTEMAX
// absolute limit of HUF_MAX_TABLELOG. Beyond that value, code does not work
// HUF_readStats_wksp() :
// Same as HUF_readStats() but takes an external workspace which must be
// 4-byte aligned and its size must be >= HUF_READ_STATS_WORKSPACE_SIZE.
// If the CPU has BMI2 support, pass bmi2=1, otherwise pass bmi2=0.
// The minimum workspace size for the `workSpace` used in
// HUF_readDTableX1_wksp() and HUF_readDTableX2_wksp().
//
// The space used depends on HUF_TABLELOG_MAX, ranging from ~1500 bytes when
// HUF_TABLE_LOG_MAX=12 to ~1850 bytes when HUF_TABLE_LOG_MAX=15.
// Buffer overflow errors may potentially occur if code modifications result in
// a required workspace size greater than that specified in the following
// macro.
// cpuid(7): Extended Features.
// C standard doesn't allow `static const` variable to be init using another `static const` variable
// for static allocation
// for static allocation
// for static allocation
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

macro_rules! HUF_4X2_DECODE_SYMBOL {
($_stream:expr, $_decode3:expr, $bits:expr, $dtable:expr, $op:expr) => {
loop {
if ($_decode3) || ($_stream) != 3 {
let index: i32 = ($bits[($_stream) as usize] >> 53i32) as i32;
let entry: __SlateRecord_huf_decompress_c_950 = unsafe { *$dtable.wrapping_offset(index as isize) };
MEM_write16($op[($_stream) as usize] as *mut (), entry.sequence);
let __v: *mut u64 = ($bits.as_mut_ptr() as *mut u64).wrapping_offset(($_stream) as isize);
unsafe {
*__v = *__v << ((entry.nbBits as i32) & 63);
}
let __v4: *mut *mut u8 = ($op.as_mut_ptr() as *mut *mut u8).wrapping_offset(($_stream) as isize);
let __v5: *mut u8 = unsafe { *__v4 };
unsafe {
*__v4 = __v5.wrapping_offset((entry.length as i32) as isize);
}
}
if true {
break;
}
}
};
}

macro_rules! HUF_DECODE_SYMBOLX1_0 {
($ptr:expr, $DStreamPtr:expr, $dt:expr, $dtLog:expr) => {
loop {
let __v: *mut u8 = $ptr;
let __v2: *mut u8 = __v.wrapping_offset(1 as isize);
$ptr = __v2;
unsafe {
*__v = HUF_decodeSymbolX1($DStreamPtr, $dt, $dtLog);
}
if true {
break;
}
}
};
}

macro_rules! HUF_DECODE_SYMBOLX1_1 {
($ptr:expr, $DStreamPtr:expr, $dt:expr, $dtLog:expr) => {
loop {
if MEM_64bits() != 0 || true {
HUF_DECODE_SYMBOLX1_0!($ptr, $DStreamPtr, $dt, $dtLog);
}
if true {
break;
}
}
};
}

macro_rules! HUF_DECODE_SYMBOLX1_2 {
($ptr:expr, $DStreamPtr:expr, $dt:expr, $dtLog:expr) => {
loop {
if MEM_64bits() != 0 {
HUF_DECODE_SYMBOLX1_0!($ptr, $DStreamPtr, $dt, $dtLog);
}
if true {
break;
}
}
};
}

macro_rules! HUF_DECODE_SYMBOLX2_0 {
($ptr:expr, $DStreamPtr:expr, $dt:expr, $dtLog:expr) => {
loop {
let __v: u32 = HUF_decodeSymbolX2(($ptr) as *mut (), $DStreamPtr, $dt, $dtLog);
let __v2: *mut u8 = $ptr;
let __v3: *mut u8 = __v2.wrapping_offset(__v as isize);
$ptr = __v3;
if true {
break;
}
}
};
}

macro_rules! HUF_DECODE_SYMBOLX2_1 {
($ptr:expr, $DStreamPtr:expr, $dt:expr, $dtLog:expr) => {
loop {
if MEM_64bits() != 0 || true {
let __v: u32 = HUF_decodeSymbolX2(($ptr) as *mut (), $DStreamPtr, $dt, $dtLog);
let __v2: *mut u8 = $ptr;
let __v3: *mut u8 = __v2.wrapping_offset(__v as isize);
$ptr = __v3;
}
if true {
break;
}
}
};
}

macro_rules! HUF_DECODE_SYMBOLX2_2 {
($ptr:expr, $DStreamPtr:expr, $dt:expr, $dtLog:expr) => {
loop {
if MEM_64bits() != 0 {
let __v: u32 = HUF_decodeSymbolX2(($ptr) as *mut (), $DStreamPtr, $dt, $dtLog);
let __v2: *mut u8 = $ptr;
let __v3: *mut u8 = __v2.wrapping_offset(__v as isize);
$ptr = __v3;
}
if true {
break;
}
}
};
}

macro_rules! MIN__4 {
($a:expr, $b:expr) => {
if ($a) < (($b) as u32) { $a } else { ($b) as u32 }
};
}

macro_rules! MIN {
($a:expr, $b:expr) => {
if ($a) < $b { $a } else { $b }
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

macro_rules! ZSTD_memcpy__7 {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memcpy($d, ($s) as *const (), ($n) as u64) };
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
macro_rules! __get_unaligned_t__2 {
($ptr:expr) => {
{
let mut __v: u64;
core::ptr::null::<u64>();
let mut __v3: u64 = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), (($ptr) as *mut ()) as *const (), core::mem::size_of::<u64>() as u64) };
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

/// __put_unaligned_t - write an unaligned value to memory.
/// @type:	the type of the value to store.
/// @val:	the value to store.
/// @ptr:	the pointer to store to.
///
/// Use memcpy to affect an unaligned type sized store avoiding undefined
/// behavior from approaches like type punning that require -fno-strict-aliasing
/// in order to be correct. The void* cast silences ubsan warnings.
macro_rules! __put_unaligned_t__4 {
($val:expr, $ptr:expr) => {
loop {
let mut __put_unaligned_val: u16 = $val;
unsafe { __builtin_memcpy(($ptr) as *mut (), core::ptr::addr_of_mut!(__put_unaligned_val) as *const (), core::mem::size_of::<u16>() as u64) };
if true {
break;
}
}
};
}

macro_rules! get_unaligned {
($ptr:expr) => {
__get_unaligned_t__2!($ptr)
};
}

macro_rules! put_unaligned {
($val:expr, $ptr:expr) => {
__put_unaligned_t!($val, $ptr);
};
}

macro_rules! put_unaligned__3 {
($val:expr, $ptr:expr) => {
__put_unaligned_t__4!($val, $ptr);
};
}

unsafe extern "C" {
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    #[link_name = "memset"]
    fn __builtin_memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn HUF_readStats_wksp(arg0: *mut u8, arg1: u64, arg2: *mut u32, arg3: *mut u32, arg4: *mut u32, arg5: *const (), arg6: u64, arg7: *mut (), arg8: u64, arg9: i32) -> u64;
}

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
/// Optional: only supported since gcc >= 15, clang >= 19
///
///   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
/// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
fn ZSTD_maybeNullPtrAdd(ptr: *mut u8, add: i64) -> *mut u8 {
if add > 0 { ptr.wrapping_offset(add as isize) } else { ptr }
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

fn MEM_isLittleEndian() -> u32 {
1
}

fn MEM_read64(memPtr: *const ()) -> u64 {
get_unaligned!(memPtr as *const u64)
}

fn MEM_write16(memPtr: *mut (), value: u16) {
put_unaligned__3!(value, memPtr as *mut u16);
}

#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

fn MEM_write64(memPtr: *mut (), value: u64) {
put_unaligned!(value, memPtr as *mut u64);
}

fn MEM_readLE16(memPtr: *const ()) -> u16 {
get_unaligned_le16(memPtr)
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
// If the provided expression evaluates to an error code, returns that error code.
//
// In debug modes, prints additional information.
fn ZSTD_countLeadingZeros32(val: u32) -> u32 {
(val.leading_zeros() as i32) as u32
}

fn ZSTD_countTrailingZeros64(val: u64) -> u32 {
(val.trailing_zeros() as i32) as u32
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

fn BIT_reloadDStreamFast(bitD: *mut __SlateRecord_bitstream_h_90) -> u32 {
if UNLIKELY!((unsafe { (*bitD).ptr }) < unsafe { (*bitD).limitPtr }) != 0 {
return 3;
}
BIT_reloadDStream_internal(bitD)
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

static zeroFilled: u64 = 0;

fn BIT_endOfDStream(DStream: &__SlateRecord_bitstream_h_90) -> u32 {
((unsafe { (*(DStream as *const __SlateRecord_bitstream_h_90)).ptr }) == unsafe { (*(DStream as *const __SlateRecord_bitstream_h_90)).start } && ((unsafe { (*(DStream as *const __SlateRecord_bitstream_h_90)).bitsConsumed }) as u64) == (core::mem::size_of::<u64>() as u64).wrapping_mul(8)) as u32
}

// Dependencies
// ZSTD_memcpy, ZSTD_memset
// BIT_*
// to compress headers
// ZSTD_highbit32, ZSTD_countTrailingZeros64
// Constants
// Macros
// When DYNAMIC_BMI2 is enabled, fast decoders are only called when bmi2 is
// supported at runtime, so we can add the BMI2 target attribute.
// When it is disabled, we will still get BMI2 if it is enabled statically.
// Error Management
// Byte alignment for workSpace management
/// These two optional macros force the use one way or another of the two
/// Huffman decompression implementations. You can't force in both directions
/// at the same time.
/// BMI2 Variant Wrappers
/// -
///
///  generic DTableDesc
///
/// -
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_huf_decompress_c_138 {
    maxTableLog: u8,
    tableType: u8,
    tableLog: u8,
    reserved: u8,
}


fn HUF_getDTableDesc(table: *const u32) -> __SlateRecord_huf_decompress_c_138 {
let mut dtd: __SlateRecord_huf_decompress_c_138 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(dtd) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
ZSTD_memcpy!(core::ptr::addr_of_mut!(dtd), table, core::mem::size_of::<__SlateRecord_huf_decompress_c_138>() as u64);
return dtd;
}

fn HUF_initFastDStream(ip: *const u8) -> u64 {
let lastByte: u8 = unsafe { *ip.wrapping_offset(7 as isize) };
let mut bitsConsumed: u64 = 0;
let __v0: u32 = if lastByte != 0 {
8u32.wrapping_sub(ZSTD_highbit32(lastByte as u32))
} else {
0
};
bitsConsumed = __v0 as u64;
let value: u64 = MEM_readLEST(ip as *const ()) | 1;
value << bitsConsumed
}

/// The input/output arguments to the Huffman fast decoding loop:
///
/// ip [in/out] - The input pointers, must be updated to reflect what is consumed.
/// op [in/out] - The output pointers, must be updated to reflect what is written.
/// bits [in/out] - The bitstream containers, must be updated to reflect the current state.
/// dt [in] - The decoding table.
/// ilowest [in] - The beginning of the valid range of the input. Decoders may read
///                down to this pointer. It may be below iend[0].
/// oend [in] - The end of the output stream. op[3] must not cross oend.
/// iend [in] - The end of each input stream. ip[i] may cross iend[i],
///             as long as it is above ilowest, but that indicates corruption.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_huf_decompress_c_178 {
    ip: [*const u8; 4],
    op: [*mut u8; 4],
    bits: [u64; 4],
    dt: *const (),
    ilowest: *const u8,
    oend: *mut u8,
    iend: [*const u8; 4],
}


/// Initializes args for the fast decoding loop.
/// @returns 1 on success
///          0 if the fallback implementation should be used.
///          Or an error code on failure.
fn HUF_DecompressFastArgs_init(args: &mut __SlateRecord_huf_decompress_c_178, dst: *mut (), dstSize: u64, src: *const (), srcSize: u64, DTable: *const u32) -> u64 {
let mut __slate_compound_0: __SlateRecord_huf_decompress_c_138 = unsafe { core::mem::zeroed() };
let dt: *const () = DTable.wrapping_offset(1 as isize) as *const ();
let dtLog: u32 = (unsafe { (*{
__slate_compound_0 = HUF_getDTableDesc(DTable);
core::ptr::addr_of_mut!(__slate_compound_0)
}).tableLog }) as u32;
let istart: *const u8 = src as *const u8;
let oend: *mut u8 = ZSTD_maybeNullPtrAdd(dst as *mut u8, dstSize as i64);
// The fast decoding loop assumes 64-bit little-endian.
// This condition is false on x32.
let __v1: bool = MEM_isLittleEndian() == 0 || MEM_32bits() != 0;
if __v1 {
return 0;
}
// Avoid nullptr addition
if dstSize == 0 {
return 0;
}
// strict minimum : jump table + 1 byte per stream
if srcSize < 10 {
return ERROR!();
}
// Must have at least 8 bytes per stream because we don't handle initializing smaller bit containers.
// If table log is not correct at this point, fallback to the old decoder.
// On small inputs we don't have enough data to trigger the fast loop, so use the old decoder.
if dtLog != 11 {
return 0;
}
// Read the jump table.
let length1: u64 = MEM_readLE16(istart as *const ()) as u64;
let length2: u64 = MEM_readLE16(istart.wrapping_offset(2 as isize) as *const ()) as u64;
let length3: u64 = MEM_readLE16(istart.wrapping_offset(4 as isize) as *const ()) as u64;
let length4: u64 = srcSize.wrapping_sub(length1.wrapping_add(length2).wrapping_add(length3).wrapping_add(6));
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).iend.as_mut_ptr() as *mut *const u8).wrapping_offset(0 as isize) = istart.wrapping_offset(6 as isize);
}
// jumpTable
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).iend.as_mut_ptr() as *mut *const u8).wrapping_offset(1 as isize) = unsafe { *((*(args as *mut __SlateRecord_huf_decompress_c_178)).iend.as_mut_ptr() as *mut *const u8).wrapping_offset(0 as isize) }.wrapping_offset(length1 as isize);
}
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).iend.as_mut_ptr() as *mut *const u8).wrapping_offset(2 as isize) = unsafe { *((*(args as *mut __SlateRecord_huf_decompress_c_178)).iend.as_mut_ptr() as *mut *const u8).wrapping_offset(1 as isize) }.wrapping_offset(length2 as isize);
}
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).iend.as_mut_ptr() as *mut *const u8).wrapping_offset(3 as isize) = unsafe { *((*(args as *mut __SlateRecord_huf_decompress_c_178)).iend.as_mut_ptr() as *mut *const u8).wrapping_offset(2 as isize) }.wrapping_offset(length3 as isize);
}
// HUF_initFastDStream() requires this, and this small of an input
// won't benefit from the ASM loop anyways.
if length1 < 8 || length2 < 8 || length3 < 8 || length4 < 8 {
return 0;
}
if length4 > srcSize {
return ERROR!();
}
// overflow
// ip[] contains the position that is currently loaded into bits[].
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).ip.as_mut_ptr() as *mut *const u8).wrapping_offset(0 as isize) = unsafe { *((*(args as *mut __SlateRecord_huf_decompress_c_178)).iend.as_mut_ptr() as *mut *const u8).wrapping_offset(1 as isize) }.wrapping_offset(-(core::mem::size_of::<u64>() as isize));
}
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).ip.as_mut_ptr() as *mut *const u8).wrapping_offset(1 as isize) = unsafe { *((*(args as *mut __SlateRecord_huf_decompress_c_178)).iend.as_mut_ptr() as *mut *const u8).wrapping_offset(2 as isize) }.wrapping_offset(-(core::mem::size_of::<u64>() as isize));
}
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).ip.as_mut_ptr() as *mut *const u8).wrapping_offset(2 as isize) = unsafe { *((*(args as *mut __SlateRecord_huf_decompress_c_178)).iend.as_mut_ptr() as *mut *const u8).wrapping_offset(3 as isize) }.wrapping_offset(-(core::mem::size_of::<u64>() as isize));
}
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).ip.as_mut_ptr() as *mut *const u8).wrapping_offset(3 as isize) = (src as *const u8).wrapping_offset(srcSize as isize).wrapping_offset(-(core::mem::size_of::<u64>() as isize));
}
// op[] contains the output pointers.
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).op.as_mut_ptr() as *mut *mut u8).wrapping_offset(0 as isize) = dst as *mut u8;
}
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).op.as_mut_ptr() as *mut *mut u8).wrapping_offset(1 as isize) = unsafe { *((*(args as *mut __SlateRecord_huf_decompress_c_178)).op.as_mut_ptr() as *mut *mut u8).wrapping_offset(0 as isize) }.wrapping_offset((dstSize.wrapping_add(3) / 4) as isize);
}
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).op.as_mut_ptr() as *mut *mut u8).wrapping_offset(2 as isize) = unsafe { *((*(args as *mut __SlateRecord_huf_decompress_c_178)).op.as_mut_ptr() as *mut *mut u8).wrapping_offset(1 as isize) }.wrapping_offset((dstSize.wrapping_add(3) / 4) as isize);
}
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).op.as_mut_ptr() as *mut *mut u8).wrapping_offset(3 as isize) = unsafe { *((*(args as *mut __SlateRecord_huf_decompress_c_178)).op.as_mut_ptr() as *mut *mut u8).wrapping_offset(2 as isize) }.wrapping_offset((dstSize.wrapping_add(3) / 4) as isize);
}
// No point to call the ASM loop for tiny outputs.
if (unsafe { *((*(args as *mut __SlateRecord_huf_decompress_c_178)).op.as_mut_ptr() as *mut *mut u8).wrapping_offset(3 as isize) }) >= oend {
return 0;
}
// bits[] is the bit container.
// It is read from the MSB down to the LSB.
// It is shifted left as it is read, and zeros are
// shifted in. After the lowest valid bit a 1 is
// set, so that CountTrailingZeros(bits[]) can be used
// to count how many bits we've consumed.
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).bits.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) = HUF_initFastDStream(*((*(args as *mut __SlateRecord_huf_decompress_c_178)).ip.as_mut_ptr() as *mut *const u8).wrapping_offset(0 as isize));
}
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).bits.as_mut_ptr() as *mut u64).wrapping_offset(1 as isize) = HUF_initFastDStream(*((*(args as *mut __SlateRecord_huf_decompress_c_178)).ip.as_mut_ptr() as *mut *const u8).wrapping_offset(1 as isize));
}
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).bits.as_mut_ptr() as *mut u64).wrapping_offset(2 as isize) = HUF_initFastDStream(*((*(args as *mut __SlateRecord_huf_decompress_c_178)).ip.as_mut_ptr() as *mut *const u8).wrapping_offset(2 as isize));
}
unsafe {
*((*(args as *mut __SlateRecord_huf_decompress_c_178)).bits.as_mut_ptr() as *mut u64).wrapping_offset(3 as isize) = HUF_initFastDStream(*((*(args as *mut __SlateRecord_huf_decompress_c_178)).ip.as_mut_ptr() as *mut *const u8).wrapping_offset(3 as isize));
}
// The decoders must be sure to never read beyond ilowest.
// This is lower than iend[0], but allowing decoders to read
// down to ilowest can allow an extra iteration or two in the
// fast loop.
unsafe {
(*(args as *mut __SlateRecord_huf_decompress_c_178)).ilowest = istart;
}
unsafe {
(*(args as *mut __SlateRecord_huf_decompress_c_178)).oend = oend;
}
unsafe {
(*(args as *mut __SlateRecord_huf_decompress_c_178)).dt = dt;
}
1
}

fn HUF_initRemainingDStream(bit: &mut __SlateRecord_bitstream_h_90, args: &__SlateRecord_huf_decompress_c_178, stream: i32, segmentEnd: *mut u8) -> u64 {
// Validate that we haven't overwritten.
if (unsafe { *((*(args as *const __SlateRecord_huf_decompress_c_178)).op.as_ptr() as *const *mut u8).wrapping_offset(stream as isize) }) > segmentEnd {
return ERROR!();
}
// Validate that we haven't read beyond iend[].
// Note that ip[] may be < iend[] because the MSB is
// the next bit to read, and we may have consumed 100%
// of the stream, so down to iend[i] - 8 is valid.
if (unsafe { *((*(args as *const __SlateRecord_huf_decompress_c_178)).ip.as_ptr() as *const *const u8).wrapping_offset(stream as isize) }) < unsafe { *((*(args as *const __SlateRecord_huf_decompress_c_178)).iend.as_ptr() as *const *const u8).wrapping_offset(stream as isize) }.wrapping_offset(-(8 as isize)) {
return ERROR!();
}
// Construct the BIT_DStream_t.
unsafe {
(*(bit as *mut __SlateRecord_bitstream_h_90)).bitContainer = MEM_readLEST(*((*(args as *const __SlateRecord_huf_decompress_c_178)).ip.as_ptr() as *const *const u8).wrapping_offset(stream as isize) as *const ());
}
unsafe {
(*(bit as *mut __SlateRecord_bitstream_h_90)).bitsConsumed = ZSTD_countTrailingZeros64(*((*(args as *const __SlateRecord_huf_decompress_c_178)).bits.as_ptr() as *const u64).wrapping_offset(stream as isize));
}
unsafe {
(*(bit as *mut __SlateRecord_bitstream_h_90)).start = (*(args as *const __SlateRecord_huf_decompress_c_178)).ilowest as *const u8;
}
unsafe {
(*(bit as *mut __SlateRecord_bitstream_h_90)).limitPtr = unsafe { (*(bit as *mut __SlateRecord_bitstream_h_90)).start }.wrapping_offset(core::mem::size_of::<u64>() as isize);
}
unsafe {
(*(bit as *mut __SlateRecord_bitstream_h_90)).ptr = *((*(args as *const __SlateRecord_huf_decompress_c_178)).ip.as_ptr() as *const *const u8).wrapping_offset(stream as isize) as *const u8;
}
0
}

// Calls X(N) for each stream 0, 1, 2, 3.
// Calls X(N, var) for each stream 0, 1, 2, 3.
/// -
///
///  single-symbol decoding
///
/// -
/// single-symbol decoding
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_huf_decompress_c_326 {
    nbBits: u8,
    byte: u8,
}


/// Packs 4 HUF_DEltX1 structs into a U64. This is used to lay down 4 entries at
/// a time.
fn HUF_DEltX1_set4(symbol: u8, nbBits: u8) -> u64 {
let mut D4: u64 = if MEM_isLittleEndian() != 0 {
((symbol as i32) << 8i32).wrapping_add(nbBits as i32) as u64
} else {
(symbol as i32).wrapping_add((nbBits as i32) << 8i32) as u64
};
D4 = D4.wrapping_mul(281479271743489);
D4
}

/// Increase the tableLog to targetTableLog and rescales the stats.
/// If tableLog > targetTableLog this is a no-op.
/// @returns New tableLog
fn HUF_rescaleStats(huffWeight: *mut u8, rankVal: *mut u32, nbSymbols: u32, tableLog: u32, targetTableLog: u32) -> u32 {
if tableLog > targetTableLog {
return tableLog;
}
if tableLog < targetTableLog {
let scale: u32 = targetTableLog.wrapping_sub(tableLog);
// Increase the weight for all non-zero probability symbols by scale.
let mut s: u32 = 0;
while s < nbSymbols {
let __v0: *mut u8 = huffWeight.wrapping_offset(s as isize);
unsafe {
*__v0 = (*__v0 as i32).wrapping_add(((if (*huffWeight.wrapping_offset(s as isize) as i32) == 0 { 0u32 } else { scale }) as u8) as i32) as u8;
}
s = s.wrapping_add(1);
}
// Update rankVal to reflect the new weights.
// All weights except 0 get moved to weight + scale.
// Weights [1, scale] are empty.
s = targetTableLog;
while s > scale {
unsafe {
*rankVal.wrapping_offset(s as isize) = *rankVal.wrapping_offset(s.wrapping_sub(scale) as isize);
}
s = s.wrapping_sub(1);
}
s = scale;
while s > 0 {
unsafe {
*rankVal.wrapping_offset(s as isize) = 0;
}
s = s.wrapping_sub(1);
}
}
targetTableLog
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_huf_decompress_c_380 {
    rankVal: [u32; 13],
    rankStart: [u32; 13],
    statsWksp: [u32; 219],
    symbols: [u8; 256],
    huffWeight: [u8; 256],
}


#[unsafe(no_mangle)]
extern "C" fn HUF_readDTableX1_wksp(DTable: *mut u32, src: *const (), srcSize: u64, workSpace: *mut (), wkspSize: u64, flags: i32) -> u64 {
let mut tableLog: u32 = 0;
let mut nbSymbols: u32 = 0;
let mut iSize: u64 = 0;
let dtPtr: *mut () = DTable.wrapping_offset(1 as isize) as *mut ();
let dt: *mut __SlateRecord_huf_decompress_c_326 = dtPtr as *mut __SlateRecord_huf_decompress_c_326;
DEBUG_STATIC_ASSERT!();
if (core::mem::size_of::<__SlateRecord_huf_decompress_c_380>() as u64) > wkspSize {
return ERROR__6!();
}
DEBUG_STATIC_ASSERT!();
// ZSTD_memset(huffWeight, 0, sizeof(huffWeight));
//
// is not necessary, even though some analyzer complain ...
iSize = unsafe { HUF_readStats_wksp((*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).huffWeight.as_mut_ptr() as *mut u8, 256u64, (*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).rankVal.as_mut_ptr() as *mut u32, core::ptr::addr_of_mut!(nbSymbols), core::ptr::addr_of_mut!(tableLog), src, srcSize, ((*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).statsWksp.as_mut_ptr() as *mut u32) as *mut (), core::mem::size_of::<[u32; 219]>() as u64, flags) };
if ERR_isError(iSize) != 0 {
return iSize;
}
// Table header
let mut dtd: __SlateRecord_huf_decompress_c_138 = HUF_getDTableDesc(DTable as *const u32);
let maxTableLog: u32 = (dtd.maxTableLog as i32).wrapping_add(1) as u32;
let targetTableLog: u32 = MIN__4!(maxTableLog, 11i32);
tableLog = HUF_rescaleStats(unsafe { (*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).huffWeight.as_mut_ptr() as *mut u8 }, unsafe { (*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).rankVal.as_mut_ptr() as *mut u32 }, nbSymbols, tableLog, targetTableLog);
if tableLog > ((dtd.maxTableLog as i32).wrapping_add(1) as u32) {
return ERROR__6!();
}
// DTable too small, Huffman tree cannot fit in
dtd.tableType = 0;
dtd.tableLog = tableLog as u8;
ZSTD_memcpy!(DTable, core::ptr::addr_of_mut!(dtd), core::mem::size_of::<__SlateRecord_huf_decompress_c_138>() as u64);
// Compute symbols and rankStart given rankVal:
//
// rankVal already contains the number of values of each weight.
//
// symbols contains the symbols ordered by weight. First are the rankVal[0]
// weight 0 symbols, followed by the rankVal[1] weight 1 symbols, and so on.
// symbols[0] is filled (but unused) to avoid a branch.
//
// rankStart contains the offset where each rank belongs in the DTable.
// rankStart[0] is not filled because there are no entries in the table for
// weight 0.
let mut n: i32 = 0;
let mut nextRankStart: u32 = 0;
let unroll: i32 = 4;
let nLimit: i32 = (nbSymbols as i32).wrapping_sub(unroll).wrapping_add(1);
while n < (tableLog as i32).wrapping_add(1) {
let curr: u32 = nextRankStart;
let __v0: u32 = nextRankStart;
nextRankStart = __v0.wrapping_add(unsafe { *((*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).rankVal.as_mut_ptr() as *mut u32).wrapping_offset(n as isize) });
unsafe {
*((*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).rankStart.as_mut_ptr() as *mut u32).wrapping_offset(n as isize) = curr;
}
n = n.wrapping_add(1);
}
n = 0;
while n < nLimit {
for u in 0..unroll {
let w: u64 = (unsafe { *((*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).huffWeight.as_mut_ptr() as *mut u8).wrapping_offset(n.wrapping_add(u) as isize) }) as u64;
let __v5: *mut u32 = unsafe { (*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).rankStart.as_mut_ptr() as *mut u32 }.wrapping_offset(w as isize);
let __v6: u32 = unsafe { *__v5 };
unsafe {
*__v5 = __v6.wrapping_add(1);
}
unsafe {
*((*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).symbols.as_mut_ptr() as *mut u8).wrapping_offset(__v6 as isize) = n.wrapping_add(u) as u8;
}
}
n = n.wrapping_add(unroll);
}
while n < (nbSymbols as i32) {
let w: u64 = (unsafe { *((*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).huffWeight.as_mut_ptr() as *mut u8).wrapping_offset(n as isize) }) as u64;
let __v14: *mut u32 = unsafe { (*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).rankStart.as_mut_ptr() as *mut u32 }.wrapping_offset(w as isize);
let __v15: u32 = unsafe { *__v14 };
unsafe {
*__v14 = __v15.wrapping_add(1);
}
unsafe {
*((*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).symbols.as_mut_ptr() as *mut u8).wrapping_offset(__v15 as isize) = n as u8;
}
n = n.wrapping_add(1);
}
// fill DTable
// We fill all entries of each weight in order.
// That way length is a constant for each iteration of the outer loop.
// We can switch based on the length to a different inner loop which is
// optimized for that particular case.
let mut w: u32 = 1;
let mut symbol: i32 = (unsafe { *((*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).rankVal.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) }) as i32;
let mut rankStart: i32 = 0;
while w < tableLog.wrapping_add(1) {
let symbolCount: i32 = (unsafe { *((*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).rankVal.as_mut_ptr() as *mut u32).wrapping_offset(w as isize) }) as i32;
let length: i32 = 1i32 << w >> 1i32;
let mut uStart: i32 = rankStart;
let nbBits: u8 = tableLog.wrapping_add(1).wrapping_sub(w) as u8;
let mut s: i32 = 0;
let mut u: i32 = 0;
match length {
1 => {
s = 0;
while s < symbolCount {
let mut D: __SlateRecord_huf_decompress_c_326 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(D) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
}
D.byte = unsafe { *((*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).symbols.as_mut_ptr() as *mut u8).wrapping_offset(symbol.wrapping_add(s) as isize) };
D.nbBits = nbBits;
unsafe {
*dt.wrapping_offset(uStart as isize) = D;
}
uStart = uStart.wrapping_add(1);
s = s.wrapping_add(1);
}
}
2 => {
s = 0;
while s < symbolCount {
let mut D: __SlateRecord_huf_decompress_c_326 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(D) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
}
D.byte = unsafe { *((*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).symbols.as_mut_ptr() as *mut u8).wrapping_offset(symbol.wrapping_add(s) as isize) };
D.nbBits = nbBits;
unsafe {
*dt.wrapping_offset(uStart.wrapping_add(0) as isize) = D;
}
unsafe {
*dt.wrapping_offset(uStart.wrapping_add(1) as isize) = D;
}
uStart = uStart.wrapping_add(2);
s = s.wrapping_add(1);
}
}
4 => {
s = 0;
while s < symbolCount {
let D4: u64 = HUF_DEltX1_set4(unsafe { *((*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).symbols.as_mut_ptr() as *mut u8).wrapping_offset(symbol.wrapping_add(s) as isize) }, nbBits);
MEM_write64(dt.wrapping_offset(uStart as isize) as *mut (), D4);
uStart = uStart.wrapping_add(4);
s = s.wrapping_add(1);
}
}
8 => {
s = 0;
while s < symbolCount {
let D4: u64 = HUF_DEltX1_set4(unsafe { *((*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).symbols.as_mut_ptr() as *mut u8).wrapping_offset(symbol.wrapping_add(s) as isize) }, nbBits);
MEM_write64(dt.wrapping_offset(uStart as isize) as *mut (), D4);
MEM_write64(dt.wrapping_offset(uStart as isize).wrapping_offset(4 as isize) as *mut (), D4);
uStart = uStart.wrapping_add(8);
s = s.wrapping_add(1);
}
}
_ => {
s = 0;
while s < symbolCount {
let D4: u64 = HUF_DEltX1_set4(unsafe { *((*(workSpace as *mut __SlateRecord_huf_decompress_c_380)).symbols.as_mut_ptr() as *mut u8).wrapping_offset(symbol.wrapping_add(s) as isize) }, nbBits);
u = 0;
while u < length {
MEM_write64(dt.wrapping_offset(uStart as isize).wrapping_offset(u as isize).wrapping_offset(0 as isize) as *mut (), D4);
MEM_write64(dt.wrapping_offset(uStart as isize).wrapping_offset(u as isize).wrapping_offset(4 as isize) as *mut (), D4);
MEM_write64(dt.wrapping_offset(uStart as isize).wrapping_offset(u as isize).wrapping_offset(8 as isize) as *mut (), D4);
MEM_write64(dt.wrapping_offset(uStart as isize).wrapping_offset(u as isize).wrapping_offset(12 as isize) as *mut (), D4);
u = u.wrapping_add(16);
}
uStart = uStart.wrapping_add(length);
s = s.wrapping_add(1);
}
}
}
symbol = symbol.wrapping_add(symbolCount);
rankStart = rankStart.wrapping_add(symbolCount.wrapping_mul(length));
w = w.wrapping_add(1);
}
iSize
}

#[inline(always)]
fn HUF_decodeSymbolX1(Dstream: *mut __SlateRecord_bitstream_h_90, dt: *const __SlateRecord_huf_decompress_c_326, dtLog: u32) -> u8 {
let val: u64 = BIT_lookBitsFast(Dstream as *const __SlateRecord_bitstream_h_90, dtLog); // note : dtLog >= 1
let c: u8 = unsafe { (*dt.wrapping_offset(val as isize)).byte };
BIT_skipBits(Dstream, (unsafe { (*dt.wrapping_offset(val as isize)).nbBits }) as u32);
c
}

#[inline(always)]
fn HUF_decodeStreamX1(mut p: *mut u8, bitDPtr: *mut __SlateRecord_bitstream_h_90, pEnd: *mut u8, dt: *const __SlateRecord_huf_decompress_c_326, dtLog: u32) -> u64 {
let pStart: *mut u8 = p;
// up to 4 symbols at a time
if ((unsafe { pEnd.offset_from(p as *mut u8) }) as i64) > 3 {
while (((BIT_reloadDStream(bitDPtr) as u32) == 0) as i32) & ((p < pEnd.wrapping_offset(-(3 as isize))) as i32) != 0 {
HUF_DECODE_SYMBOLX1_2!(p, bitDPtr, dt, dtLog);
HUF_DECODE_SYMBOLX1_1!(p, bitDPtr, dt, dtLog);
HUF_DECODE_SYMBOLX1_2!(p, bitDPtr, dt, dtLog);
HUF_DECODE_SYMBOLX1_0!(p, bitDPtr, dt, dtLog);
}
} else {
BIT_reloadDStream(bitDPtr);
}
// [0-3] symbols remaining
if MEM_32bits() != 0 {
while (((BIT_reloadDStream(bitDPtr) as u32) == 0) as i32) & ((p < pEnd) as i32) != 0 {
HUF_DECODE_SYMBOLX1_0!(p, bitDPtr, dt, dtLog);
}
}
// no more data to retrieve from bitstream, no need to reload
while p < pEnd {
HUF_DECODE_SYMBOLX1_0!(p, bitDPtr, dt, dtLog);
}
((unsafe { pEnd.offset_from(pStart as *mut u8) }) as i64) as u64
}

#[inline(always)]
fn HUF_decompress1X1_usingDTable_internal_body(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32) -> u64 {
let op: *mut u8 = dst as *mut u8;
let oend: *mut u8 = ZSTD_maybeNullPtrAdd(op, dstSize as i64);
let dtPtr: *const () = DTable.wrapping_offset(1 as isize) as *const ();
let dt: *const __SlateRecord_huf_decompress_c_326 = dtPtr as *const __SlateRecord_huf_decompress_c_326;
let mut bitD: __SlateRecord_bitstream_h_90 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bitD) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
let dtd: __SlateRecord_huf_decompress_c_138 = HUF_getDTableDesc(DTable);
let dtLog: u32 = dtd.tableLog as u32;
CHECK_F!(BIT_initDStream(core::ptr::addr_of_mut!(bitD), cSrc, cSrcSize));
HUF_decodeStreamX1(op, core::ptr::addr_of_mut!(bitD), oend, dt, dtLog);
if BIT_endOfDStream(unsafe { &(*(core::ptr::addr_of_mut!(bitD) as *const __SlateRecord_bitstream_h_90)) }) == 0 {
return ERROR!();
}
dstSize
}

/// HUF_decompress4X1_usingDTable_internal_body():
/// Conditions :
/// @dstSize >= 6
#[inline(always)]
fn HUF_decompress4X1_usingDTable_internal_body(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32) -> u64 {
// Check
if cSrcSize < 10 {
return ERROR!();
}
// strict minimum : jump table + 1 byte per stream
if dstSize < 6 {
return ERROR!();
}
// stream 4-split doesn't work
let istart: *const u8 = cSrc as *const u8;
let ostart: *mut u8 = dst as *mut u8;
let oend: *mut u8 = ostart.wrapping_offset(dstSize as isize);
let olimit: *mut u8 = oend.wrapping_offset(-(3 as isize));
let dtPtr: *const () = DTable.wrapping_offset(1 as isize) as *const ();
let dt: *const __SlateRecord_huf_decompress_c_326 = dtPtr as *const __SlateRecord_huf_decompress_c_326;
// Init
let mut bitD1: __SlateRecord_bitstream_h_90 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bitD1) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
let mut bitD2: __SlateRecord_bitstream_h_90 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bitD2) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
let mut bitD3: __SlateRecord_bitstream_h_90 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bitD3) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
let mut bitD4: __SlateRecord_bitstream_h_90 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bitD4) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
let length1: u64 = MEM_readLE16(istart as *const ()) as u64;
let length2: u64 = MEM_readLE16(istart.wrapping_offset(2 as isize) as *const ()) as u64;
let length3: u64 = MEM_readLE16(istart.wrapping_offset(4 as isize) as *const ()) as u64;
let length4: u64 = cSrcSize.wrapping_sub(length1.wrapping_add(length2).wrapping_add(length3).wrapping_add(6));
let istart1: *const u8 = istart.wrapping_offset(6 as isize); // jumpTable
let istart2: *const u8 = istart1.wrapping_offset(length1 as isize);
let istart3: *const u8 = istart2.wrapping_offset(length2 as isize);
let istart4: *const u8 = istart3.wrapping_offset(length3 as isize);
let segmentSize: u64 = dstSize.wrapping_add(3) / 4;
let opStart2: *mut u8 = ostart.wrapping_offset(segmentSize as isize);
let opStart3: *mut u8 = opStart2.wrapping_offset(segmentSize as isize);
let opStart4: *mut u8 = opStart3.wrapping_offset(segmentSize as isize);
let mut op1: *mut u8 = ostart;
let mut op2: *mut u8 = opStart2;
let mut op3: *mut u8 = opStart3;
let mut op4: *mut u8 = opStart4;
let dtd: __SlateRecord_huf_decompress_c_138 = HUF_getDTableDesc(DTable);
let dtLog: u32 = dtd.tableLog as u32;
let mut endSignal: u32 = 1;
if length4 > cSrcSize {
return ERROR!();
}
// overflow
if opStart4 > oend {
return ERROR!();
}
// overflow
// validated above
CHECK_F!(BIT_initDStream(core::ptr::addr_of_mut!(bitD1), istart1 as *const (), length1));
CHECK_F!(BIT_initDStream(core::ptr::addr_of_mut!(bitD2), istart2 as *const (), length2));
CHECK_F!(BIT_initDStream(core::ptr::addr_of_mut!(bitD3), istart3 as *const (), length3));
CHECK_F!(BIT_initDStream(core::ptr::addr_of_mut!(bitD4), istart4 as *const (), length4));
// up to 16 symbols per loop (4 symbols per stream) in 64-bit mode
if (((unsafe { oend.offset_from(op4 as *mut u8) }) as i64) as u64) >= (core::mem::size_of::<u64>() as u64) {
while endSignal & (((op4 < olimit) as i32) as u32) != 0 {
HUF_DECODE_SYMBOLX1_2!(op1, core::ptr::addr_of_mut!(bitD1), dt, dtLog);
HUF_DECODE_SYMBOLX1_2!(op2, core::ptr::addr_of_mut!(bitD2), dt, dtLog);
HUF_DECODE_SYMBOLX1_2!(op3, core::ptr::addr_of_mut!(bitD3), dt, dtLog);
HUF_DECODE_SYMBOLX1_2!(op4, core::ptr::addr_of_mut!(bitD4), dt, dtLog);
HUF_DECODE_SYMBOLX1_1!(op1, core::ptr::addr_of_mut!(bitD1), dt, dtLog);
HUF_DECODE_SYMBOLX1_1!(op2, core::ptr::addr_of_mut!(bitD2), dt, dtLog);
HUF_DECODE_SYMBOLX1_1!(op3, core::ptr::addr_of_mut!(bitD3), dt, dtLog);
HUF_DECODE_SYMBOLX1_1!(op4, core::ptr::addr_of_mut!(bitD4), dt, dtLog);
HUF_DECODE_SYMBOLX1_2!(op1, core::ptr::addr_of_mut!(bitD1), dt, dtLog);
HUF_DECODE_SYMBOLX1_2!(op2, core::ptr::addr_of_mut!(bitD2), dt, dtLog);
HUF_DECODE_SYMBOLX1_2!(op3, core::ptr::addr_of_mut!(bitD3), dt, dtLog);
HUF_DECODE_SYMBOLX1_2!(op4, core::ptr::addr_of_mut!(bitD4), dt, dtLog);
HUF_DECODE_SYMBOLX1_0!(op1, core::ptr::addr_of_mut!(bitD1), dt, dtLog);
HUF_DECODE_SYMBOLX1_0!(op2, core::ptr::addr_of_mut!(bitD2), dt, dtLog);
HUF_DECODE_SYMBOLX1_0!(op3, core::ptr::addr_of_mut!(bitD3), dt, dtLog);
HUF_DECODE_SYMBOLX1_0!(op4, core::ptr::addr_of_mut!(bitD4), dt, dtLog);
let __v68: i32 = ((BIT_reloadDStreamFast(core::ptr::addr_of_mut!(bitD1)) as u32) == 0) as i32;
endSignal &= __v68 as u32;
let __v71: i32 = ((BIT_reloadDStreamFast(core::ptr::addr_of_mut!(bitD2)) as u32) == 0) as i32;
endSignal &= __v71 as u32;
let __v74: i32 = ((BIT_reloadDStreamFast(core::ptr::addr_of_mut!(bitD3)) as u32) == 0) as i32;
endSignal &= __v74 as u32;
let __v77: i32 = ((BIT_reloadDStreamFast(core::ptr::addr_of_mut!(bitD4)) as u32) == 0) as i32;
endSignal &= __v77 as u32;
}
}
// check corruption
//
// note : should not be necessary : op# advance in lock step, and we control op4.
// but curiously, binary generated by gcc 7.2 & 7.3 with -mbmi2 runs faster when >=1 test is present
if op1 > opStart2 {
return ERROR!();
}
if op2 > opStart3 {
return ERROR!();
}
if op3 > opStart4 {
return ERROR!();
}
// note : op4 supposed already verified within main loop
// finish bitStreams one by one
HUF_decodeStreamX1(op1, core::ptr::addr_of_mut!(bitD1), opStart2, dt, dtLog);
HUF_decodeStreamX1(op2, core::ptr::addr_of_mut!(bitD2), opStart3, dt, dtLog);
HUF_decodeStreamX1(op3, core::ptr::addr_of_mut!(bitD3), opStart4, dt, dtLog);
HUF_decodeStreamX1(op4, core::ptr::addr_of_mut!(bitD4), oend, dt, dtLog);
// check
let endCheck: u32 = BIT_endOfDStream(unsafe { &(*(core::ptr::addr_of_mut!(bitD1) as *const __SlateRecord_bitstream_h_90)) }) & BIT_endOfDStream(unsafe { &(*(core::ptr::addr_of_mut!(bitD2) as *const __SlateRecord_bitstream_h_90)) }) & BIT_endOfDStream(unsafe { &(*(core::ptr::addr_of_mut!(bitD3) as *const __SlateRecord_bitstream_h_90)) }) & BIT_endOfDStream(unsafe { &(*(core::ptr::addr_of_mut!(bitD4) as *const __SlateRecord_bitstream_h_90)) });
if endCheck == 0 {
return ERROR!();
}
// decoded size
dstSize
}

#[target_feature(enable = "lzcnt,bmi1,bmi2")]
#[unsafe(link_section = ".text.slate_distinct.huf_decompress.HUF_decompress4X1_usingDTable_internal_bmi2")]
unsafe extern "C" fn HUF_decompress4X1_usingDTable_internal_bmi2(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32) -> u64 {
HUF_decompress4X1_usingDTable_internal_body(dst, dstSize, cSrc, cSrcSize, DTable)
}

#[unsafe(link_section = ".text.slate_distinct.huf_decompress.HUF_decompress4X1_usingDTable_internal_default")]
extern "C" fn HUF_decompress4X1_usingDTable_internal_default(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32) -> u64 {
HUF_decompress4X1_usingDTable_internal_body(dst, dstSize, cSrc, cSrcSize, DTable)
}

#[target_feature(enable = "lzcnt,bmi1,bmi2")]
#[unsafe(link_section = ".text.slate_distinct.huf_decompress.HUF_decompress4X1_usingDTable_internal_fast_c_loop")]
unsafe extern "C" fn HUF_decompress4X1_usingDTable_internal_fast_c_loop(args: *mut __SlateRecord_huf_decompress_c_178) {
let mut stream: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_158: *mut i32 = core::ptr::addr_of_mut!(stream) as *mut i32;
let mut olimit: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_159: *mut *mut u8 = core::ptr::addr_of_mut!(olimit) as *mut *mut u8;
let mut ilowest: *const u8 = unsafe { core::mem::zeroed() };
let mut oend: *mut u8 = unsafe { core::mem::zeroed() };
let mut dtable: *const u16 = unsafe { core::mem::zeroed() };
let mut op: core::mem::MaybeUninit<__SlateAlign16<[*mut u8; 4]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_163: *mut [*mut u8; 4] = core::ptr::addr_of_mut!(op) as *mut [*mut u8; 4];
let mut ip: core::mem::MaybeUninit<__SlateAlign16<[*const u8; 4]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_164: *mut [*const u8; 4] = core::ptr::addr_of_mut!(ip) as *mut [*const u8; 4];
let mut bits: core::mem::MaybeUninit<__SlateAlign16<[u64; 4]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_165: *mut [u64; 4] = core::ptr::addr_of_mut!(bits) as *mut [u64; 4];
unsafe {
let __slate_auto_bytes_1 = __slate_slot_165 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 32usize);
let __slate_auto_bytes_2 = __slate_slot_164 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 32usize);
let __slate_auto_bytes_3 = __slate_slot_163 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 32usize);
dtable = (*args).dt as *const u16;
oend = (*args).oend;
ilowest = (*args).ilowest;
// Copy the arguments to local variables
ZSTD_memcpy!(core::ptr::addr_of_mut!(*__slate_slot_165), core::ptr::addr_of_mut!((*args).bits), core::mem::size_of::<[u64; 4]>() as u64);
ZSTD_memcpy__3!(core::ptr::addr_of_mut!(*__slate_slot_164) as *mut (), core::ptr::addr_of_mut!((*args).ip), core::mem::size_of::<[*const u8; 4]>() as u64);
ZSTD_memcpy!(core::ptr::addr_of_mut!(*__slate_slot_163), core::ptr::addr_of_mut!((*args).op), core::mem::size_of::<[*mut u8; 4]>() as u64);
'__loop_45: loop {
let __slate_auto_bytes = __slate_slot_159 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_158 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// Assert loop preconditions
*__slate_slot_158 = 0;
while *__slate_slot_158 < 4 {
let __v157: i32 = *__slate_slot_158;
*__slate_slot_158 = __v157.wrapping_add(1);
}
// Compute olimit
// Each iteration produces 5 output symbols per stream
let oiters: u64 = ((oend.offset_from((*__slate_slot_163)[3] as *mut u8) as i64) as u64) / 5;
// Each iteration consumes up to 11 bits * 5 = 55 bits < 7 bytes
// per stream.
let iiters: u64 = (((*__slate_slot_164)[0].offset_from(ilowest as *const u8) as i64) as u64) / 7;
// We can safely run iters iterations before running bounds checks
let iters: u64 = MIN!(oiters, iiters);
let symbols: u64 = iters.wrapping_mul(5);
// We can simply check that op[3] < olimit, instead of checking all
// of our bounds, since we can't hit the other bounds until we've run
// iters iterations, which only happens when op[3] == olimit.
*__slate_slot_159 = (*__slate_slot_163)[3].wrapping_offset(symbols as isize);
// Exit fast decoding loop once we reach the end.
if (*__slate_slot_163)[3] == *__slate_slot_159 {
break;
} else {
// Exit the decoding loop if any input pointer has crossed the
// previous one. This indicates corruption, and a precondition
// to our loop is that ip[i] >= ip[0].
*__slate_slot_158 = 1;
while *__slate_slot_158 < 4 {
if (*__slate_slot_164)[*__slate_slot_158 as usize] < (*__slate_slot_164)[(*__slate_slot_158).wrapping_sub(1) as usize] {
break '__loop_45;
} else {
let __v151: i32 = *__slate_slot_158;
*__slate_slot_158 = __v151.wrapping_add(1);
}
}
*__slate_slot_158 = 1;
while *__slate_slot_158 < 4 {
let __v149: i32 = *__slate_slot_158;
*__slate_slot_158 = __v149.wrapping_add(1);
}
// Manually unroll the loop because compilers don't consistently
// unroll the inner loops, which destroys performance.
loop {
// Decode 5 symbols in each of the 4 streams
loop {
let index_5: i32 = ((*__slate_slot_165)[0] >> 53i32) as i32;
let entry_5: i32 = *dtable.wrapping_offset(index_5 as isize) as i32;
let __v145: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(0 as isize);
*__v145 = *__v145 << (entry_5 & 63);
*(*__slate_slot_163)[0].wrapping_offset(0 as isize) = (entry_5 >> 8i32 & 255) as u8;
if true {
loop {
let index: i32 = ((*__slate_slot_165)[1] >> 53i32) as i32;
let entry: i32 = *dtable.wrapping_offset(index as isize) as i32;
let __v140: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(1 as isize);
*__v140 = *__v140 << (entry & 63);
*(*__slate_slot_163)[1].wrapping_offset(0 as isize) = (entry >> 8i32 & 255) as u8;
if true {
break;
}
}
loop {
let index: i32 = ((*__slate_slot_165)[2] >> 53i32) as i32;
let entry: i32 = *dtable.wrapping_offset(index as isize) as i32;
let __v135: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(2 as isize);
*__v135 = *__v135 << (entry & 63);
*(*__slate_slot_163)[2].wrapping_offset(0 as isize) = (entry >> 8i32 & 255) as u8;
if true {
break;
}
}
loop {
let index: i32 = ((*__slate_slot_165)[3] >> 53i32) as i32;
let entry: i32 = *dtable.wrapping_offset(index as isize) as i32;
let __v130: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(3 as isize);
*__v130 = *__v130 << (entry & 63);
*(*__slate_slot_163)[3].wrapping_offset(0 as isize) = (entry >> 8i32 & 255) as u8;
if true {
break;
}
}
if true {
break;
}
}
}
loop {
let index_4: i32 = ((*__slate_slot_165)[0] >> 53i32) as i32;
let entry_4: i32 = *dtable.wrapping_offset(index_4 as isize) as i32;
let __v125: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(0 as isize);
*__v125 = *__v125 << (entry_4 & 63);
*(*__slate_slot_163)[0].wrapping_offset(1 as isize) = (entry_4 >> 8i32 & 255) as u8;
if true {
loop {
let index: i32 = ((*__slate_slot_165)[1] >> 53i32) as i32;
let entry: i32 = *dtable.wrapping_offset(index as isize) as i32;
let __v120: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(1 as isize);
*__v120 = *__v120 << (entry & 63);
*(*__slate_slot_163)[1].wrapping_offset(1 as isize) = (entry >> 8i32 & 255) as u8;
if true {
break;
}
}
loop {
let index: i32 = ((*__slate_slot_165)[2] >> 53i32) as i32;
let entry: i32 = *dtable.wrapping_offset(index as isize) as i32;
let __v115: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(2 as isize);
*__v115 = *__v115 << (entry & 63);
*(*__slate_slot_163)[2].wrapping_offset(1 as isize) = (entry >> 8i32 & 255) as u8;
if true {
break;
}
}
loop {
let index: i32 = ((*__slate_slot_165)[3] >> 53i32) as i32;
let entry: i32 = *dtable.wrapping_offset(index as isize) as i32;
let __v110: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(3 as isize);
*__v110 = *__v110 << (entry & 63);
*(*__slate_slot_163)[3].wrapping_offset(1 as isize) = (entry >> 8i32 & 255) as u8;
if true {
break;
}
}
if true {
break;
}
}
}
loop {
let index_3: i32 = ((*__slate_slot_165)[0] >> 53i32) as i32;
let entry_3: i32 = *dtable.wrapping_offset(index_3 as isize) as i32;
let __v105: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(0 as isize);
*__v105 = *__v105 << (entry_3 & 63);
*(*__slate_slot_163)[0].wrapping_offset(2 as isize) = (entry_3 >> 8i32 & 255) as u8;
if true {
loop {
let index: i32 = ((*__slate_slot_165)[1] >> 53i32) as i32;
let entry: i32 = *dtable.wrapping_offset(index as isize) as i32;
let __v100: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(1 as isize);
*__v100 = *__v100 << (entry & 63);
*(*__slate_slot_163)[1].wrapping_offset(2 as isize) = (entry >> 8i32 & 255) as u8;
if true {
break;
}
}
loop {
let index: i32 = ((*__slate_slot_165)[2] >> 53i32) as i32;
let entry: i32 = *dtable.wrapping_offset(index as isize) as i32;
let __v95: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(2 as isize);
*__v95 = *__v95 << (entry & 63);
*(*__slate_slot_163)[2].wrapping_offset(2 as isize) = (entry >> 8i32 & 255) as u8;
if true {
break;
}
}
loop {
let index: i32 = ((*__slate_slot_165)[3] >> 53i32) as i32;
let entry: i32 = *dtable.wrapping_offset(index as isize) as i32;
let __v90: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(3 as isize);
*__v90 = *__v90 << (entry & 63);
*(*__slate_slot_163)[3].wrapping_offset(2 as isize) = (entry >> 8i32 & 255) as u8;
if true {
break;
}
}
if true {
break;
}
}
}
loop {
let index_2: i32 = ((*__slate_slot_165)[0] >> 53i32) as i32;
let entry_2: i32 = *dtable.wrapping_offset(index_2 as isize) as i32;
let __v85: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(0 as isize);
*__v85 = *__v85 << (entry_2 & 63);
*(*__slate_slot_163)[0].wrapping_offset(3 as isize) = (entry_2 >> 8i32 & 255) as u8;
if true {
loop {
let index: i32 = ((*__slate_slot_165)[1] >> 53i32) as i32;
let entry: i32 = *dtable.wrapping_offset(index as isize) as i32;
let __v80: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(1 as isize);
*__v80 = *__v80 << (entry & 63);
*(*__slate_slot_163)[1].wrapping_offset(3 as isize) = (entry >> 8i32 & 255) as u8;
if true {
break;
}
}
loop {
let index: i32 = ((*__slate_slot_165)[2] >> 53i32) as i32;
let entry: i32 = *dtable.wrapping_offset(index as isize) as i32;
let __v75: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(2 as isize);
*__v75 = *__v75 << (entry & 63);
*(*__slate_slot_163)[2].wrapping_offset(3 as isize) = (entry >> 8i32 & 255) as u8;
if true {
break;
}
}
loop {
let index: i32 = ((*__slate_slot_165)[3] >> 53i32) as i32;
let entry: i32 = *dtable.wrapping_offset(index as isize) as i32;
let __v70: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(3 as isize);
*__v70 = *__v70 << (entry & 63);
*(*__slate_slot_163)[3].wrapping_offset(3 as isize) = (entry >> 8i32 & 255) as u8;
if true {
break;
}
}
if true {
break;
}
}
}
loop {
let index_1: i32 = ((*__slate_slot_165)[0] >> 53i32) as i32;
let entry_1: i32 = *dtable.wrapping_offset(index_1 as isize) as i32;
let __v65: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(0 as isize);
*__v65 = *__v65 << (entry_1 & 63);
*(*__slate_slot_163)[0].wrapping_offset(4 as isize) = (entry_1 >> 8i32 & 255) as u8;
if true {
loop {
let index: i32 = ((*__slate_slot_165)[1] >> 53i32) as i32;
let entry: i32 = *dtable.wrapping_offset(index as isize) as i32;
let __v60: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(1 as isize);
*__v60 = *__v60 << (entry & 63);
*(*__slate_slot_163)[1].wrapping_offset(4 as isize) = (entry >> 8i32 & 255) as u8;
if true {
break;
}
}
loop {
let index: i32 = ((*__slate_slot_165)[2] >> 53i32) as i32;
let entry: i32 = *dtable.wrapping_offset(index as isize) as i32;
let __v55: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(2 as isize);
*__v55 = *__v55 << (entry & 63);
*(*__slate_slot_163)[2].wrapping_offset(4 as isize) = (entry >> 8i32 & 255) as u8;
if true {
break;
}
}
loop {
let index: i32 = ((*__slate_slot_165)[3] >> 53i32) as i32;
let entry: i32 = *dtable.wrapping_offset(index as isize) as i32;
let __v50: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(3 as isize);
*__v50 = *__v50 << (entry & 63);
*(*__slate_slot_163)[3].wrapping_offset(4 as isize) = (entry >> 8i32 & 255) as u8;
if true {
break;
}
}
if true {
break;
}
}
}
// Reload each of the 4 the bitstreams
loop {
let ctz_1: i32 = ZSTD_countTrailingZeros64((*__slate_slot_165)[0]) as i32;
let nbBits_1: i32 = ctz_1 & 7;
let nbBytes_1: i32 = ctz_1 >> 3i32;
let __v44: *mut *mut u8 = ((*__slate_slot_163).as_mut_ptr() as *mut *mut u8).wrapping_offset(0 as isize);
let __v43: *mut u8 = *__v44;
*__v44 = __v43.wrapping_offset(5 as isize);
let __v41: *mut *const u8 = ((*__slate_slot_164).as_mut_ptr() as *mut *const u8).wrapping_offset(0 as isize);
let __v40: *const u8 = *__v41;
*__v41 = __v40.wrapping_offset(-(nbBytes_1 as isize));
(*__slate_slot_165)[0] = MEM_read64((*__slate_slot_164)[0] as *const ()) | 1;
let __v38: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(0 as isize);
*__v38 = *__v38 << nbBits_1;
if true {
loop {
let ctz: i32 = ZSTD_countTrailingZeros64((*__slate_slot_165)[1]) as i32;
let nbBits: i32 = ctz & 7;
let nbBytes: i32 = ctz >> 3i32;
let __v32: *mut *mut u8 = ((*__slate_slot_163).as_mut_ptr() as *mut *mut u8).wrapping_offset(1 as isize);
let __v31: *mut u8 = *__v32;
*__v32 = __v31.wrapping_offset(5 as isize);
let __v29: *mut *const u8 = ((*__slate_slot_164).as_mut_ptr() as *mut *const u8).wrapping_offset(1 as isize);
let __v28: *const u8 = *__v29;
*__v29 = __v28.wrapping_offset(-(nbBytes as isize));
(*__slate_slot_165)[1] = MEM_read64((*__slate_slot_164)[1] as *const ()) | 1;
let __v26: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(1 as isize);
*__v26 = *__v26 << nbBits;
if true {
break;
}
}
loop {
let ctz: i32 = ZSTD_countTrailingZeros64((*__slate_slot_165)[2]) as i32;
let nbBits: i32 = ctz & 7;
let nbBytes: i32 = ctz >> 3i32;
let __v20: *mut *mut u8 = ((*__slate_slot_163).as_mut_ptr() as *mut *mut u8).wrapping_offset(2 as isize);
let __v19: *mut u8 = *__v20;
*__v20 = __v19.wrapping_offset(5 as isize);
let __v17: *mut *const u8 = ((*__slate_slot_164).as_mut_ptr() as *mut *const u8).wrapping_offset(2 as isize);
let __v16: *const u8 = *__v17;
*__v17 = __v16.wrapping_offset(-(nbBytes as isize));
(*__slate_slot_165)[2] = MEM_read64((*__slate_slot_164)[2] as *const ()) | 1;
let __v14: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(2 as isize);
*__v14 = *__v14 << nbBits;
if true {
break;
}
}
loop {
let ctz: i32 = ZSTD_countTrailingZeros64((*__slate_slot_165)[3]) as i32;
let nbBits: i32 = ctz & 7;
let nbBytes: i32 = ctz >> 3i32;
let __v8: *mut *mut u8 = ((*__slate_slot_163).as_mut_ptr() as *mut *mut u8).wrapping_offset(3 as isize);
let __v7: *mut u8 = *__v8;
*__v8 = __v7.wrapping_offset(5 as isize);
let __v5: *mut *const u8 = ((*__slate_slot_164).as_mut_ptr() as *mut *const u8).wrapping_offset(3 as isize);
let __v4: *const u8 = *__v5;
*__v5 = __v4.wrapping_offset(-(nbBytes as isize));
(*__slate_slot_165)[3] = MEM_read64((*__slate_slot_164)[3] as *const ()) | 1;
let __v2: *mut u64 = ((*__slate_slot_165).as_mut_ptr() as *mut u64).wrapping_offset(3 as isize);
*__v2 = *__v2 << nbBits;
if true {
break;
}
}
if true {
break;
}
}
}
if !((*__slate_slot_163)[3] < *__slate_slot_159) {
continue '__loop_45;
}
}
}
}
ZSTD_memcpy!(core::ptr::addr_of_mut!((*args).bits), core::ptr::addr_of_mut!(*__slate_slot_165), core::mem::size_of::<[u64; 4]>() as u64); // Save the final values of each of the state variables back to args.
ZSTD_memcpy__3!(core::ptr::addr_of_mut!((*args).ip) as *mut (), core::ptr::addr_of_mut!(*__slate_slot_164), core::mem::size_of::<[*const u8; 4]>() as u64);
ZSTD_memcpy!(core::ptr::addr_of_mut!((*args).op), core::ptr::addr_of_mut!(*__slate_slot_163), core::mem::size_of::<[*mut u8; 4]>() as u64);
}
}

/// @returns @p dstSize on success (>= 6)
///          0 if the fallback implementation should be used
///          An error if an error occurred
#[target_feature(enable = "lzcnt,bmi1,bmi2")]
unsafe fn HUF_decompress4X1_usingDTable_internal_fast(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32, loopFn: Option<unsafe extern "C" fn(*mut __SlateRecord_huf_decompress_c_178)>) -> u64 {
let dt: *const () = DTable.wrapping_offset(1 as isize) as *const ();
let oend: *mut u8 = ZSTD_maybeNullPtrAdd(dst as *mut u8, dstSize as i64);
let mut args: __SlateRecord_huf_decompress_c_178 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(args) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 152usize);
}
let ret: u64 = HUF_DecompressFastArgs_init(unsafe { &mut (*core::ptr::addr_of_mut!(args)) }, dst, dstSize, cSrc, cSrcSize, DTable);
loop {
let err_code: u64 = ret;
if ERR_isError(err_code) != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"Failed to init fast loop args".as_ptr() as *const u8) };
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
if ret == 0 {
return 0;
}
unsafe { loopFn.unwrap()(core::ptr::addr_of_mut!(args)) };
// Our loop guarantees that ip[] >= ilowest and that we haven't
// overwritten any op[].
// finish bit streams one by one.
let segmentSize: u64 = dstSize.wrapping_add(3) / 4;
let mut segmentEnd: *mut u8 = dst as *mut u8;
for i in 0i32..4i32 {
let mut bit: __SlateRecord_bitstream_h_90 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bit) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
if segmentSize <= (((unsafe { oend.offset_from(segmentEnd as *mut u8) }) as i64) as u64) {
let __v5: *mut u8 = segmentEnd;
segmentEnd = __v5.wrapping_offset(segmentSize as isize);
} else {
segmentEnd = oend;
}
loop {
let err_code: u64 = HUF_initRemainingDStream(unsafe { &mut (*core::ptr::addr_of_mut!(bit)) }, unsafe { &(*(core::ptr::addr_of_mut!(args) as *const __SlateRecord_huf_decompress_c_178)) }, i, segmentEnd);
if ERR_isError(err_code) != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"corruption".as_ptr() as *const u8) };
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
// Decompress and validate that we've produced exactly the expected length.
let __v12: u64 = HUF_decodeStreamX1(args.op[i as usize], core::ptr::addr_of_mut!(bit), segmentEnd, dt as *const __SlateRecord_huf_decompress_c_326, 11u32);
let __v13: *mut *mut u8 = (args.op.as_mut_ptr() as *mut *mut u8).wrapping_offset(i as isize);
let __v14: *mut u8 = unsafe { *__v13 };
unsafe {
*__v13 = __v14.wrapping_offset(__v12 as isize);
}
if args.op[i as usize] != segmentEnd {
return ERROR!();
}
}
// decoded size
dstSize
}

fn HUF_decompress1X1_usingDTable_internal_default(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32) -> u64 {
HUF_decompress1X1_usingDTable_internal_body(dst, dstSize, cSrc, cSrcSize, DTable)
}

#[target_feature(enable = "lzcnt,bmi1,bmi2")]
unsafe fn HUF_decompress1X1_usingDTable_internal_bmi2(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32) -> u64 {
HUF_decompress1X1_usingDTable_internal_body(dst, dstSize, cSrc, cSrcSize, DTable)
}

fn HUF_decompress1X1_usingDTable_internal(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32, flags: i32) -> u64 {
if flags & 1 != 0 {
return unsafe { HUF_decompress1X1_usingDTable_internal_bmi2(dst, dstSize, cSrc, cSrcSize, DTable) };
}
HUF_decompress1X1_usingDTable_internal_default(dst, dstSize, cSrc, cSrcSize, DTable)
}

fn HUF_decompress4X1_usingDTable_internal(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32, flags: i32) -> u64 {
let mut fallbackFn: Option<unsafe extern "C" fn(*mut (), u64, *const (), u64, *const u32) -> u64> = Some(HUF_decompress4X1_usingDTable_internal_default);
let loopFn: Option<unsafe extern "C" fn(*mut __SlateRecord_huf_decompress_c_178)> = Some(HUF_decompress4X1_usingDTable_internal_fast_c_loop);
if flags & 1 != 0 {
fallbackFn = Some(HUF_decompress4X1_usingDTable_internal_bmi2);
} else {
return unsafe { fallbackFn.unwrap()(dst, dstSize, cSrc, cSrcSize, DTable) };
}
if true && flags & 32 == 0 {
let ret: u64 = unsafe { HUF_decompress4X1_usingDTable_internal_fast(dst, dstSize, cSrc, cSrcSize, DTable, loopFn) };
if ret != 0 {
return ret;
}
}
unsafe { fallbackFn.unwrap()(dst, dstSize, cSrc, cSrcSize, DTable) }
}

fn HUF_decompress4X1_DCtx_wksp(dctx: *mut u32, dst: *mut (), dstSize: u64, cSrc: *const (), mut cSrcSize: u64, workSpace: *mut (), wkspSize: u64, flags: i32) -> u64 {
let mut ip: *const u8 = cSrc as *const u8;
let hSize: u64 = HUF_readDTableX1_wksp(dctx, cSrc, cSrcSize, workSpace, wkspSize, flags);
if ERR_isError(hSize) != 0 {
return hSize;
}
if hSize >= cSrcSize {
return ERROR__2!();
}
let __v0: *const u8 = ip;
ip = __v0.wrapping_offset(hSize as isize);
cSrcSize = cSrcSize.wrapping_sub(hSize);
HUF_decompress4X1_usingDTable_internal(dst, dstSize, ip as *const (), cSrcSize, dctx as *const u32, flags)
}

// double-symbols decoding
/// double-symbols decoding
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_huf_decompress_c_950 {
    sequence: u16,
    nbBits: u8,
    length: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_huf_decompress_c_951 {
    symbol: u8,
}


/// Constructs a HUF_DEltX2 in a U32.
fn HUF_buildDEltX2U32(symbol: u32, nbBits: u32, baseSeq: u32, level: i32) -> u32 {
let mut seq: u32 = 0;
DEBUG_STATIC_ASSERT!();
DEBUG_STATIC_ASSERT!();
DEBUG_STATIC_ASSERT!();
DEBUG_STATIC_ASSERT!();
if MEM_isLittleEndian() != 0 {
seq = if level == 1 { symbol } else { baseSeq.wrapping_add(symbol << 8i32) };
return seq.wrapping_add(nbBits << 16i32).wrapping_add((level as u32) << 24i32);
} else {
seq = if level == 1 { symbol << 8i32 } else { (baseSeq << 8i32).wrapping_add(symbol) };
return (seq << 16i32).wrapping_add(nbBits << 8i32).wrapping_add(level as u32);
}
}

/// Constructs a HUF_DEltX2.
fn HUF_buildDEltX2(symbol: u32, nbBits: u32, baseSeq: u32, level: i32) -> __SlateRecord_huf_decompress_c_950 {
let mut DElt: __SlateRecord_huf_decompress_c_950 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(DElt) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let val: u32 = HUF_buildDEltX2U32(symbol, nbBits, baseSeq, level);
DEBUG_STATIC_ASSERT!();
ZSTD_memcpy!(core::ptr::addr_of_mut!(DElt), core::ptr::addr_of!(val), core::mem::size_of::<u32>() as u64);
return DElt;
}

/// Constructs 2 HUF_DEltX2s and packs them into a U64.
fn HUF_buildDEltX2U64(symbol: u32, nbBits: u32, baseSeq: u16, level: i32) -> u64 {
let DElt: u32 = HUF_buildDEltX2U32(symbol, nbBits, baseSeq as u32, level);
(DElt as u64).wrapping_add((DElt as u64) << 32i32)
}

/// Fills the DTable rank with all the symbols from [begin, end) that are each
/// nbBits long.
///
/// @param DTableRank The start of the rank in the DTable.
/// @param begin The first symbol to fill (inclusive).
/// @param end The last symbol to fill (exclusive).
/// @param nbBits Each symbol is nbBits long.
/// @param tableLog The table log.
/// @param baseSeq If level == 1 { 0 } else { the first level symbol }
/// @param level The level in the table. Must be 1 or 2.
fn HUF_fillDTableX2ForWeight(mut DTableRank: *mut __SlateRecord_huf_decompress_c_950, begin: *const __SlateRecord_huf_decompress_c_951, end: *const __SlateRecord_huf_decompress_c_951, nbBits: u32, tableLog: u32, baseSeq: u16, level: i32) {
let length: u32 = 1u32 << (tableLog.wrapping_sub(nbBits) & 31); // quiet static-analyzer
let mut ptr: *const __SlateRecord_huf_decompress_c_951 = unsafe { core::mem::zeroed() };
match length {
1 => {
ptr = begin;
while ptr != end {
let DElt: __SlateRecord_huf_decompress_c_950 = HUF_buildDEltX2((unsafe { (*ptr).symbol }) as u32, nbBits, baseSeq as u32, level);
let __v0: *mut __SlateRecord_huf_decompress_c_950 = DTableRank;
DTableRank = __v0.wrapping_offset(1 as isize);
unsafe {
*__v0 = DElt;
}
let __v3: *const __SlateRecord_huf_decompress_c_951 = ptr;
ptr = __v3.wrapping_offset(1 as isize);
}
}
2 => {
ptr = begin;
while ptr != end {
let DElt: __SlateRecord_huf_decompress_c_950 = HUF_buildDEltX2((unsafe { (*ptr).symbol }) as u32, nbBits, baseSeq as u32, level);
unsafe {
*DTableRank.wrapping_offset(0 as isize) = DElt;
}
unsafe {
*DTableRank.wrapping_offset(1 as isize) = DElt;
}
let __v6: *mut __SlateRecord_huf_decompress_c_950 = DTableRank;
DTableRank = __v6.wrapping_offset(2 as isize);
let __v9: *const __SlateRecord_huf_decompress_c_951 = ptr;
ptr = __v9.wrapping_offset(1 as isize);
}
}
4 => {
ptr = begin;
while ptr != end {
let DEltX2: u64 = HUF_buildDEltX2U64((unsafe { (*ptr).symbol }) as u32, nbBits, baseSeq, level);
ZSTD_memcpy!(DTableRank.wrapping_offset(0 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
ZSTD_memcpy!(DTableRank.wrapping_offset(2 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
let __v11: *mut __SlateRecord_huf_decompress_c_950 = DTableRank;
DTableRank = __v11.wrapping_offset(4 as isize);
let __v14: *const __SlateRecord_huf_decompress_c_951 = ptr;
ptr = __v14.wrapping_offset(1 as isize);
}
}
8 => {
ptr = begin;
while ptr != end {
let DEltX2: u64 = HUF_buildDEltX2U64((unsafe { (*ptr).symbol }) as u32, nbBits, baseSeq, level);
ZSTD_memcpy!(DTableRank.wrapping_offset(0 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
ZSTD_memcpy!(DTableRank.wrapping_offset(2 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
ZSTD_memcpy!(DTableRank.wrapping_offset(4 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
ZSTD_memcpy!(DTableRank.wrapping_offset(6 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
let __v16: *mut __SlateRecord_huf_decompress_c_950 = DTableRank;
DTableRank = __v16.wrapping_offset(8 as isize);
let __v19: *const __SlateRecord_huf_decompress_c_951 = ptr;
ptr = __v19.wrapping_offset(1 as isize);
}
}
_ => {
ptr = begin;
while ptr != end {
let DEltX2: u64 = HUF_buildDEltX2U64((unsafe { (*ptr).symbol }) as u32, nbBits, baseSeq, level);
let DTableRankEnd: *mut __SlateRecord_huf_decompress_c_950 = DTableRank.wrapping_offset(length as isize);
while DTableRank != DTableRankEnd {
ZSTD_memcpy!(DTableRank.wrapping_offset(0 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
ZSTD_memcpy!(DTableRank.wrapping_offset(2 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
ZSTD_memcpy!(DTableRank.wrapping_offset(4 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
ZSTD_memcpy!(DTableRank.wrapping_offset(6 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
let __v22: *mut __SlateRecord_huf_decompress_c_950 = DTableRank;
DTableRank = __v22.wrapping_offset(8 as isize);
}
let __v25: *const __SlateRecord_huf_decompress_c_951 = ptr;
ptr = __v25.wrapping_offset(1 as isize);
}
}
}
}

/// HUF_fillDTableX2Level2() :
/// `rankValOrigin` must be a table of at least (HUF_TABLELOG_MAX + 1) U32
fn HUF_fillDTableX2Level2(DTable: *mut __SlateRecord_huf_decompress_c_950, targetLog: u32, consumedBits: u32, rankVal: *const u32, minWeight: i32, maxWeight1: i32, sortedSymbols: *const __SlateRecord_huf_decompress_c_951, rankStart: *const u32, nbBitsBaseline: u32, baseSeq: u16) {
// Fill skipped values (all positions up to rankVal[minWeight]).
// These are positions only get a single symbol because the combined weight
// is too large.
if minWeight > 1 {
let length: u32 = 1u32 << (targetLog.wrapping_sub(consumedBits) & 31); // quiet static-analyzer
let DEltX2: u64 = HUF_buildDEltX2U64(baseSeq as u32, consumedBits, 0u16, 1i32); // baseSeq
// level
let skipSize: i32 = (unsafe { *rankVal.wrapping_offset(minWeight as isize) }) as i32;
match length {
2 => {
ZSTD_memcpy!(DTable, core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
}
4 => {
ZSTD_memcpy!(DTable.wrapping_offset(0 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
ZSTD_memcpy!(DTable.wrapping_offset(2 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
}
_ => {
let mut i: i32 = 0;
while i < skipSize {
ZSTD_memcpy!(DTable.wrapping_offset(i as isize).wrapping_offset(0 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
ZSTD_memcpy!(DTable.wrapping_offset(i as isize).wrapping_offset(2 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
ZSTD_memcpy!(DTable.wrapping_offset(i as isize).wrapping_offset(4 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
ZSTD_memcpy!(DTable.wrapping_offset(i as isize).wrapping_offset(6 as isize), core::ptr::addr_of!(DEltX2), core::mem::size_of::<u64>() as u64);
i = i.wrapping_add(8);
}
}
}
}
// Fill each of the second level symbols by weight.
for w in minWeight..maxWeight1 {
let begin: i32 = (unsafe { *rankStart.wrapping_offset(w as isize) }) as i32;
let end: i32 = (unsafe { *rankStart.wrapping_offset(w.wrapping_add(1) as isize) }) as i32;
let nbBits: u32 = nbBitsBaseline.wrapping_sub(w as u32);
let totalBits: u32 = nbBits.wrapping_add(consumedBits);
HUF_fillDTableX2ForWeight(DTable.wrapping_offset((unsafe { *rankVal.wrapping_offset(w as isize) }) as isize), sortedSymbols.wrapping_offset(begin as isize), sortedSymbols.wrapping_offset(end as isize), totalBits, targetLog, baseSeq, 2i32); // level
}
}

fn HUF_fillDTableX2(DTable: *mut __SlateRecord_huf_decompress_c_950, targetLog: u32, sortedList: *const __SlateRecord_huf_decompress_c_951, rankStart: *const u32, rankValOrigin: *mut [u32; 13], maxWeight: u32, nbBitsBaseline: u32) {
let rankVal: *mut u32 = unsafe { (*rankValOrigin.wrapping_offset(0 as isize)).as_mut_ptr() as *mut u32 };
let scaleLog: i32 = nbBitsBaseline.wrapping_sub(targetLog) as i32; // note : targetLog >= srcLog, hence scaleLog <= 1
let minBits: u32 = nbBitsBaseline.wrapping_sub(maxWeight);
let wEnd: i32 = (maxWeight as i32).wrapping_add(1);
// Fill DTable in order of weight.
for w in 1..wEnd {
let begin: i32 = (unsafe { *rankStart.wrapping_offset(w as isize) }) as i32;
let end: i32 = (unsafe { *rankStart.wrapping_offset(w.wrapping_add(1) as isize) }) as i32;
let nbBits: u32 = nbBitsBaseline.wrapping_sub(w as u32);
if targetLog.wrapping_sub(nbBits) >= minBits {
// Enough room for a second symbol.
let mut start: i32 = (unsafe { *rankVal.wrapping_offset(w as isize) }) as i32;
let length: u32 = 1u32 << (targetLog.wrapping_sub(nbBits) & 31); // quiet static-analyzer
let mut minWeight: i32 = nbBits.wrapping_add(scaleLog as u32) as i32;
let mut s: i32 = 0;
if minWeight < 1 {
minWeight = 1;
}
// Fill the DTable for every symbol of weight w.
// These symbols get at least 1 second symbol.
s = begin;
while s != end {
HUF_fillDTableX2Level2(DTable.wrapping_offset(start as isize), targetLog, nbBits, (unsafe { (*rankValOrigin.wrapping_offset(nbBits as isize)).as_mut_ptr() as *mut u32 }) as *const u32, minWeight, wEnd, sortedList, rankStart, nbBitsBaseline, (unsafe { (*sortedList.wrapping_offset(s as isize)).symbol }) as u16);
start = (start as u32).wrapping_add(length) as i32;
s = s.wrapping_add(1);
}
} else {
// Only a single symbol.
HUF_fillDTableX2ForWeight(DTable.wrapping_offset((unsafe { *rankVal.wrapping_offset(w as isize) }) as isize), sortedList.wrapping_offset(begin as isize), sortedList.wrapping_offset(end as isize), nbBits, targetLog, 0u16, 1i32); // baseSeq
// level
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_huf_decompress_c_1174 {
    rankVal: [[u32; 13]; 12],
    rankStats: [u32; 13],
    rankStart0: [u32; 15],
    sortedSymbol: [__SlateRecord_huf_decompress_c_951; 256],
    weightList: [u8; 256],
    calleeWksp: [u32; 219],
}


#[unsafe(no_mangle)]
extern "C" fn HUF_readDTableX2_wksp(DTable: *mut u32, src: *const (), srcSize: u64, workSpace: *mut (), wkspSize: u64, flags: i32) -> u64 {
let mut tableLog: u32 = 0;
let mut maxW: u32 = 0;
let mut nbSymbols: u32 = 0;
let mut dtd: __SlateRecord_huf_decompress_c_138 = HUF_getDTableDesc(DTable as *const u32);
let mut maxTableLog: u32 = dtd.maxTableLog as u32;
let mut iSize: u64 = 0;
let dtPtr: *mut () = DTable.wrapping_offset(1 as isize) as *mut (); // force compiler to avoid strict-aliasing
let dt: *mut __SlateRecord_huf_decompress_c_950 = dtPtr as *mut __SlateRecord_huf_decompress_c_950;
let wksp: *mut __SlateRecord_huf_decompress_c_1174 = workSpace as *mut __SlateRecord_huf_decompress_c_1174;
if (core::mem::size_of::<__SlateRecord_huf_decompress_c_1174>() as u64) > wkspSize {
return ERROR__5!();
}
let rankStart: *mut u32 = unsafe { (*wksp).rankStart0.as_mut_ptr() as *mut u32 }.wrapping_offset(1 as isize);
ZSTD_memset!(unsafe { (*wksp).rankStats.as_mut_ptr() as *mut u32 }, 0i32, core::mem::size_of::<[u32; 13]>() as u64);
ZSTD_memset!(unsafe { (*wksp).rankStart0.as_mut_ptr() as *mut u32 }, 0i32, core::mem::size_of::<[u32; 15]>() as u64);
DEBUG_STATIC_ASSERT!(); // if compiler fails here, assertion is wrong
if maxTableLog > 12 {
return ERROR__6!();
}
// ZSTD_memset(weightList, 0, sizeof(weightList));
//
// is not necessary, even though some analyzer complain ...
iSize = unsafe { HUF_readStats_wksp((*wksp).weightList.as_mut_ptr() as *mut u8, 256u64, (*wksp).rankStats.as_mut_ptr() as *mut u32, core::ptr::addr_of_mut!(nbSymbols), core::ptr::addr_of_mut!(tableLog), src, srcSize, ((*wksp).calleeWksp.as_mut_ptr() as *mut u32) as *mut (), core::mem::size_of::<[u32; 219]>() as u64, flags) };
if ERR_isError(iSize) != 0 {
return iSize;
}
// check result
if tableLog > maxTableLog {
return ERROR__6!();
}
// DTable can't fit code depth
if tableLog <= 11 && maxTableLog > 11 {
maxTableLog = 11;
}
// find maxWeight
maxW = tableLog;
while (unsafe { *((*wksp).rankStats.as_mut_ptr() as *mut u32).wrapping_offset(maxW as isize) }) == 0 {
maxW = maxW.wrapping_sub(1);
}
// necessarily finds a solution before 0
// Get start index of each weight
{
let mut w: u32 = 1;
let mut nextRankStart: u32 = 0;
while w < maxW.wrapping_add(1) {
let curr: u32 = nextRankStart;
let __v3: u32 = nextRankStart;
nextRankStart = __v3.wrapping_add(unsafe { *((*wksp).rankStats.as_mut_ptr() as *mut u32).wrapping_offset(w as isize) });
unsafe {
*rankStart.wrapping_offset(w as isize) = curr;
}
w = w.wrapping_add(1);
}
unsafe {
*rankStart.wrapping_offset(0 as isize) = nextRankStart;
}
// put all 0w symbols at the end of sorted list
unsafe {
*rankStart.wrapping_offset(maxW.wrapping_add(1) as isize) = nextRankStart;
}
}
// sort symbols by weight
let mut s: u32 = 0;
while s < nbSymbols {
let w: u32 = (unsafe { *((*wksp).weightList.as_mut_ptr() as *mut u8).wrapping_offset(s as isize) }) as u32;
let mut r: u32 = 0;
let __v8: *mut u32 = rankStart.wrapping_offset(w as isize);
let __v9: u32 = unsafe { *__v8 };
unsafe {
*__v8 = __v9.wrapping_add(1);
}
r = __v9;
unsafe {
(*((*wksp).sortedSymbol.as_mut_ptr() as *mut __SlateRecord_huf_decompress_c_951).wrapping_offset(r as isize)).symbol = s as u8;
}
s = s.wrapping_add(1);
}
unsafe {
*rankStart.wrapping_offset(0 as isize) = 0;
}
// forget 0w symbols; this is beginning of weight(1)
// Build rankVal
let rankVal0: *mut u32 = unsafe { (*((*wksp).rankVal.as_mut_ptr() as *mut [u32; 13]).wrapping_offset(0 as isize)).as_mut_ptr() as *mut u32 };
{
let rescale: i32 = maxTableLog.wrapping_sub(tableLog).wrapping_sub(1) as i32; // tableLog <= maxTableLog
let mut nextRankVal: u32 = 0;
let mut w: u32 = 1;
while w < maxW.wrapping_add(1) {
let curr: u32 = nextRankVal;
let __v14: u32 = nextRankVal;
nextRankVal = __v14.wrapping_add((unsafe { *((*wksp).rankStats.as_mut_ptr() as *mut u32).wrapping_offset(w as isize) }) << w.wrapping_add(rescale as u32));
unsafe {
*rankVal0.wrapping_offset(w as isize) = curr;
}
w = w.wrapping_add(1);
}
}
let minBits: u32 = tableLog.wrapping_add(1).wrapping_sub(maxW);
let mut consumed: u32 = minBits;
while consumed < maxTableLog.wrapping_sub(minBits).wrapping_add(1) {
let rankValPtr: *mut u32 = unsafe { (*((*wksp).rankVal.as_mut_ptr() as *mut [u32; 13]).wrapping_offset(consumed as isize)).as_mut_ptr() as *mut u32 };
let mut w: u32 = 1;
while w < maxW.wrapping_add(1) {
unsafe {
*rankValPtr.wrapping_offset(w as isize) = *rankVal0.wrapping_offset(w as isize) >> consumed;
}
w = w.wrapping_add(1);
}
consumed = consumed.wrapping_add(1);
}
HUF_fillDTableX2(dt, maxTableLog, (unsafe { (*wksp).sortedSymbol.as_mut_ptr() as *mut __SlateRecord_huf_decompress_c_951 }) as *const __SlateRecord_huf_decompress_c_951, (unsafe { (*wksp).rankStart0.as_mut_ptr() as *mut u32 }) as *const u32, unsafe { (*wksp).rankVal.as_mut_ptr() as *mut [u32; 13] }, maxW, tableLog.wrapping_add(1));
dtd.tableLog = maxTableLog as u8;
dtd.tableType = 1;
ZSTD_memcpy!(DTable, core::ptr::addr_of_mut!(dtd), core::mem::size_of::<__SlateRecord_huf_decompress_c_138>() as u64);
iSize
}

#[inline(always)]
fn HUF_decodeSymbolX2(op: *mut (), DStream: *mut __SlateRecord_bitstream_h_90, dt: *const __SlateRecord_huf_decompress_c_950, dtLog: u32) -> u32 {
let val: u64 = BIT_lookBitsFast(DStream as *const __SlateRecord_bitstream_h_90, dtLog); // note : dtLog >= 1
ZSTD_memcpy__7!(op, unsafe { core::ptr::addr_of!((*dt.wrapping_offset(val as isize)).sequence) }, 2i32);
BIT_skipBits(DStream, (unsafe { (*dt.wrapping_offset(val as isize)).nbBits }) as u32);
(unsafe { (*dt.wrapping_offset(val as isize)).length }) as u32
}

#[inline(always)]
fn HUF_decodeLastSymbolX2(op: *mut (), DStream: *mut __SlateRecord_bitstream_h_90, dt: *const __SlateRecord_huf_decompress_c_950, dtLog: u32) -> u32 {
let val: u64 = BIT_lookBitsFast(DStream as *const __SlateRecord_bitstream_h_90, dtLog); // note : dtLog >= 1
ZSTD_memcpy__7!(op, unsafe { core::ptr::addr_of!((*dt.wrapping_offset(val as isize)).sequence) }, 1i32);
if ((unsafe { (*dt.wrapping_offset(val as isize)).length }) as i32) == 1 {
BIT_skipBits(DStream, (unsafe { (*dt.wrapping_offset(val as isize)).nbBits }) as u32);
} else {
if ((unsafe { (*DStream).bitsConsumed }) as u64) < (core::mem::size_of::<u64>() as u64).wrapping_mul(8) {
BIT_skipBits(DStream, (unsafe { (*dt.wrapping_offset(val as isize)).nbBits }) as u32);
if ((unsafe { (*DStream).bitsConsumed }) as u64) > (core::mem::size_of::<u64>() as u64).wrapping_mul(8) {
unsafe {
(*DStream).bitsConsumed = (core::mem::size_of::<u64>() as u64).wrapping_mul(8) as u32;
}
}
// ugly hack; works only because it's the last symbol. Note : can't easily extract nbBits from just this symbol
}
}
1
}

#[inline(always)]
fn HUF_decodeStreamX2(mut p: *mut u8, bitDPtr: *mut __SlateRecord_bitstream_h_90, pEnd: *mut u8, dt: *const __SlateRecord_huf_decompress_c_950, dtLog: u32) -> u64 {
let pStart: *mut u8 = p;
// up to 8 symbols at a time
if (((unsafe { pEnd.offset_from(p as *mut u8) }) as i64) as u64) >= (core::mem::size_of::<u64>() as u64) {
let __v0: bool = dtLog <= 11 && MEM_64bits() != 0;
if __v0 {
// up to 10 symbols at a time
while (((BIT_reloadDStream(bitDPtr) as u32) == 0) as i32) & ((p < pEnd.wrapping_offset(-(9 as isize))) as i32) != 0 {
HUF_DECODE_SYMBOLX2_0!(p, bitDPtr, dt, dtLog);
HUF_DECODE_SYMBOLX2_0!(p, bitDPtr, dt, dtLog);
HUF_DECODE_SYMBOLX2_0!(p, bitDPtr, dt, dtLog);
HUF_DECODE_SYMBOLX2_0!(p, bitDPtr, dt, dtLog);
HUF_DECODE_SYMBOLX2_0!(p, bitDPtr, dt, dtLog);
}
} else {
// up to 8 symbols at a time
while (((BIT_reloadDStream(bitDPtr) as u32) == 0) as i32) & ((p < pEnd.wrapping_offset(-((core::mem::size_of::<u64>() as u64).wrapping_sub(1) as isize))) as i32) != 0 {
HUF_DECODE_SYMBOLX2_2!(p, bitDPtr, dt, dtLog);
HUF_DECODE_SYMBOLX2_1!(p, bitDPtr, dt, dtLog);
HUF_DECODE_SYMBOLX2_2!(p, bitDPtr, dt, dtLog);
HUF_DECODE_SYMBOLX2_0!(p, bitDPtr, dt, dtLog);
}
}
} else {
BIT_reloadDStream(bitDPtr);
}
// closer to end : up to 2 symbols at a time
if (((unsafe { pEnd.offset_from(p as *mut u8) }) as i64) as u64) >= 2 {
while (((BIT_reloadDStream(bitDPtr) as u32) == 0) as i32) & ((p <= pEnd.wrapping_offset(-(2 as isize))) as i32) != 0 {
HUF_DECODE_SYMBOLX2_0!(p, bitDPtr, dt, dtLog);
}
while p <= pEnd.wrapping_offset(-(2 as isize)) {
HUF_DECODE_SYMBOLX2_0!(p, bitDPtr, dt, dtLog);
}
// no need to reload : reached the end of DStream
}
if p < pEnd {
let __v49: u32 = HUF_decodeLastSymbolX2(p as *mut (), bitDPtr, dt, dtLog);
let __v50: *mut u8 = p;
p = __v50.wrapping_offset(__v49 as isize);
}
((unsafe { p.offset_from(pStart as *mut u8) }) as i64) as u64
}

#[inline(always)]
fn HUF_decompress1X2_usingDTable_internal_body(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32) -> u64 {
let mut bitD: __SlateRecord_bitstream_h_90 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bitD) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
// Init
CHECK_F!(BIT_initDStream(core::ptr::addr_of_mut!(bitD), cSrc, cSrcSize));
// decode
let ostart: *mut u8 = dst as *mut u8;
let oend: *mut u8 = ZSTD_maybeNullPtrAdd(ostart, dstSize as i64);
let dtPtr: *const () = DTable.wrapping_offset(1 as isize) as *const (); // force compiler to not use strict-aliasing
let dt: *const __SlateRecord_huf_decompress_c_950 = dtPtr as *const __SlateRecord_huf_decompress_c_950;
let dtd: __SlateRecord_huf_decompress_c_138 = HUF_getDTableDesc(DTable);
HUF_decodeStreamX2(ostart, core::ptr::addr_of_mut!(bitD), oend, dt, dtd.tableLog as u32);
// check
if BIT_endOfDStream(unsafe { &(*(core::ptr::addr_of_mut!(bitD) as *const __SlateRecord_bitstream_h_90)) }) == 0 {
return ERROR!();
}
// decoded size
dstSize
}

/// HUF_decompress4X2_usingDTable_internal_body():
/// Conditions:
/// @dstSize >= 6
#[inline(always)]
fn HUF_decompress4X2_usingDTable_internal_body(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32) -> u64 {
if cSrcSize < 10 {
return ERROR!();
}
// strict minimum : jump table + 1 byte per stream
if dstSize < 6 {
return ERROR!();
}
// stream 4-split doesn't work
let istart: *const u8 = cSrc as *const u8;
let ostart: *mut u8 = dst as *mut u8;
let oend: *mut u8 = ostart.wrapping_offset(dstSize as isize);
let olimit: *mut u8 = oend.wrapping_offset(-((core::mem::size_of::<u64>() as u64).wrapping_sub(1) as isize));
let dtPtr: *const () = DTable.wrapping_offset(1 as isize) as *const ();
let dt: *const __SlateRecord_huf_decompress_c_950 = dtPtr as *const __SlateRecord_huf_decompress_c_950;
// Init
let mut bitD1: __SlateRecord_bitstream_h_90 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bitD1) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
let mut bitD2: __SlateRecord_bitstream_h_90 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bitD2) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
let mut bitD3: __SlateRecord_bitstream_h_90 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bitD3) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
let mut bitD4: __SlateRecord_bitstream_h_90 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bitD4) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
let length1: u64 = MEM_readLE16(istart as *const ()) as u64;
let length2: u64 = MEM_readLE16(istart.wrapping_offset(2 as isize) as *const ()) as u64;
let length3: u64 = MEM_readLE16(istart.wrapping_offset(4 as isize) as *const ()) as u64;
let length4: u64 = cSrcSize.wrapping_sub(length1.wrapping_add(length2).wrapping_add(length3).wrapping_add(6));
let istart1: *const u8 = istart.wrapping_offset(6 as isize); // jumpTable
let istart2: *const u8 = istart1.wrapping_offset(length1 as isize);
let istart3: *const u8 = istart2.wrapping_offset(length2 as isize);
let istart4: *const u8 = istart3.wrapping_offset(length3 as isize);
let segmentSize: u64 = dstSize.wrapping_add(3) / 4;
let opStart2: *mut u8 = ostart.wrapping_offset(segmentSize as isize);
let opStart3: *mut u8 = opStart2.wrapping_offset(segmentSize as isize);
let opStart4: *mut u8 = opStart3.wrapping_offset(segmentSize as isize);
let mut op1: *mut u8 = ostart;
let mut op2: *mut u8 = opStart2;
let mut op3: *mut u8 = opStart3;
let mut op4: *mut u8 = opStart4;
let mut endSignal: u32 = 1;
let dtd: __SlateRecord_huf_decompress_c_138 = HUF_getDTableDesc(DTable);
let dtLog: u32 = dtd.tableLog as u32;
if length4 > cSrcSize {
return ERROR!();
}
// overflow
if opStart4 > oend {
return ERROR!();
}
// overflow
// validated above
CHECK_F!(BIT_initDStream(core::ptr::addr_of_mut!(bitD1), istart1 as *const (), length1));
CHECK_F!(BIT_initDStream(core::ptr::addr_of_mut!(bitD2), istart2 as *const (), length2));
CHECK_F!(BIT_initDStream(core::ptr::addr_of_mut!(bitD3), istart3 as *const (), length3));
CHECK_F!(BIT_initDStream(core::ptr::addr_of_mut!(bitD4), istart4 as *const (), length4));
// 16-32 symbols per loop (4-8 symbols per stream)
if (((unsafe { oend.offset_from(op4 as *mut u8) }) as i64) as u64) >= (core::mem::size_of::<u64>() as u64) {
while endSignal & (((op4 < olimit) as i32) as u32) != 0 {
HUF_DECODE_SYMBOLX2_2!(op1, core::ptr::addr_of_mut!(bitD1), dt, dtLog);
HUF_DECODE_SYMBOLX2_1!(op1, core::ptr::addr_of_mut!(bitD1), dt, dtLog);
HUF_DECODE_SYMBOLX2_2!(op1, core::ptr::addr_of_mut!(bitD1), dt, dtLog);
HUF_DECODE_SYMBOLX2_0!(op1, core::ptr::addr_of_mut!(bitD1), dt, dtLog);
HUF_DECODE_SYMBOLX2_2!(op2, core::ptr::addr_of_mut!(bitD2), dt, dtLog);
HUF_DECODE_SYMBOLX2_1!(op2, core::ptr::addr_of_mut!(bitD2), dt, dtLog);
HUF_DECODE_SYMBOLX2_2!(op2, core::ptr::addr_of_mut!(bitD2), dt, dtLog);
HUF_DECODE_SYMBOLX2_0!(op2, core::ptr::addr_of_mut!(bitD2), dt, dtLog);
let __v40: i32 = ((BIT_reloadDStreamFast(core::ptr::addr_of_mut!(bitD1)) as u32) == 0) as i32;
endSignal &= __v40 as u32;
let __v43: i32 = ((BIT_reloadDStreamFast(core::ptr::addr_of_mut!(bitD2)) as u32) == 0) as i32;
endSignal &= __v43 as u32;
HUF_DECODE_SYMBOLX2_2!(op3, core::ptr::addr_of_mut!(bitD3), dt, dtLog);
HUF_DECODE_SYMBOLX2_1!(op3, core::ptr::addr_of_mut!(bitD3), dt, dtLog);
HUF_DECODE_SYMBOLX2_2!(op3, core::ptr::addr_of_mut!(bitD3), dt, dtLog);
HUF_DECODE_SYMBOLX2_0!(op3, core::ptr::addr_of_mut!(bitD3), dt, dtLog);
HUF_DECODE_SYMBOLX2_2!(op4, core::ptr::addr_of_mut!(bitD4), dt, dtLog);
HUF_DECODE_SYMBOLX2_1!(op4, core::ptr::addr_of_mut!(bitD4), dt, dtLog);
HUF_DECODE_SYMBOLX2_2!(op4, core::ptr::addr_of_mut!(bitD4), dt, dtLog);
HUF_DECODE_SYMBOLX2_0!(op4, core::ptr::addr_of_mut!(bitD4), dt, dtLog);
let __v78: i32 = ((BIT_reloadDStreamFast(core::ptr::addr_of_mut!(bitD3)) as u32) == 0) as i32;
endSignal &= __v78 as u32;
let __v81: i32 = ((BIT_reloadDStreamFast(core::ptr::addr_of_mut!(bitD4)) as u32) == 0) as i32;
endSignal &= __v81 as u32;
}
}
// check corruption
if op1 > opStart2 {
return ERROR!();
}
if op2 > opStart3 {
return ERROR!();
}
if op3 > opStart4 {
return ERROR!();
}
// note : op4 already verified within main loop
// finish bitStreams one by one
HUF_decodeStreamX2(op1, core::ptr::addr_of_mut!(bitD1), opStart2, dt, dtLog);
HUF_decodeStreamX2(op2, core::ptr::addr_of_mut!(bitD2), opStart3, dt, dtLog);
HUF_decodeStreamX2(op3, core::ptr::addr_of_mut!(bitD3), opStart4, dt, dtLog);
HUF_decodeStreamX2(op4, core::ptr::addr_of_mut!(bitD4), oend, dt, dtLog);
// check
let endCheck: u32 = BIT_endOfDStream(unsafe { &(*(core::ptr::addr_of_mut!(bitD1) as *const __SlateRecord_bitstream_h_90)) }) & BIT_endOfDStream(unsafe { &(*(core::ptr::addr_of_mut!(bitD2) as *const __SlateRecord_bitstream_h_90)) }) & BIT_endOfDStream(unsafe { &(*(core::ptr::addr_of_mut!(bitD3) as *const __SlateRecord_bitstream_h_90)) }) & BIT_endOfDStream(unsafe { &(*(core::ptr::addr_of_mut!(bitD4) as *const __SlateRecord_bitstream_h_90)) });
if endCheck == 0 {
return ERROR!();
}
// decoded size
dstSize
}

#[target_feature(enable = "lzcnt,bmi1,bmi2")]
#[unsafe(link_section = ".text.slate_distinct.huf_decompress.HUF_decompress4X2_usingDTable_internal_bmi2")]
unsafe extern "C" fn HUF_decompress4X2_usingDTable_internal_bmi2(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32) -> u64 {
HUF_decompress4X2_usingDTable_internal_body(dst, dstSize, cSrc, cSrcSize, DTable)
}

#[unsafe(link_section = ".text.slate_distinct.huf_decompress.HUF_decompress4X2_usingDTable_internal_default")]
extern "C" fn HUF_decompress4X2_usingDTable_internal_default(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32) -> u64 {
HUF_decompress4X2_usingDTable_internal_body(dst, dstSize, cSrc, cSrcSize, DTable)
}

#[target_feature(enable = "lzcnt,bmi1,bmi2")]
#[unsafe(link_section = ".text.slate_distinct.huf_decompress.HUF_decompress4X2_usingDTable_internal_fast_c_loop")]
unsafe extern "C" fn HUF_decompress4X2_usingDTable_internal_fast_c_loop(args: *mut __SlateRecord_huf_decompress_c_178) {
let mut entry_1: core::mem::MaybeUninit<__SlateRecord_huf_decompress_c_950> = core::mem::MaybeUninit::uninit();
let __slate_slot_50: *mut __SlateRecord_huf_decompress_c_950 = core::ptr::addr_of_mut!(entry_1) as *mut __SlateRecord_huf_decompress_c_950;
let mut entry_2: core::mem::MaybeUninit<__SlateRecord_huf_decompress_c_950> = core::mem::MaybeUninit::uninit();
let __slate_slot_58: *mut __SlateRecord_huf_decompress_c_950 = core::ptr::addr_of_mut!(entry_2) as *mut __SlateRecord_huf_decompress_c_950;
let mut entry_3: core::mem::MaybeUninit<__SlateRecord_huf_decompress_c_950> = core::mem::MaybeUninit::uninit();
let __slate_slot_66: *mut __SlateRecord_huf_decompress_c_950 = core::ptr::addr_of_mut!(entry_3) as *mut __SlateRecord_huf_decompress_c_950;
let mut entry_5: core::mem::MaybeUninit<__SlateRecord_huf_decompress_c_950> = core::mem::MaybeUninit::uninit();
let __slate_slot_82: *mut __SlateRecord_huf_decompress_c_950 = core::ptr::addr_of_mut!(entry_5) as *mut __SlateRecord_huf_decompress_c_950;
let mut entry_6: core::mem::MaybeUninit<__SlateRecord_huf_decompress_c_950> = core::mem::MaybeUninit::uninit();
let __slate_slot_90: *mut __SlateRecord_huf_decompress_c_950 = core::ptr::addr_of_mut!(entry_6) as *mut __SlateRecord_huf_decompress_c_950;
let mut entry_7: core::mem::MaybeUninit<__SlateRecord_huf_decompress_c_950> = core::mem::MaybeUninit::uninit();
let __slate_slot_98: *mut __SlateRecord_huf_decompress_c_950 = core::ptr::addr_of_mut!(entry_7) as *mut __SlateRecord_huf_decompress_c_950;
let mut entry_9: core::mem::MaybeUninit<__SlateRecord_huf_decompress_c_950> = core::mem::MaybeUninit::uninit();
let __slate_slot_114: *mut __SlateRecord_huf_decompress_c_950 = core::ptr::addr_of_mut!(entry_9) as *mut __SlateRecord_huf_decompress_c_950;
let mut entry_10: core::mem::MaybeUninit<__SlateRecord_huf_decompress_c_950> = core::mem::MaybeUninit::uninit();
let __slate_slot_122: *mut __SlateRecord_huf_decompress_c_950 = core::ptr::addr_of_mut!(entry_10) as *mut __SlateRecord_huf_decompress_c_950;
let mut entry_11: core::mem::MaybeUninit<__SlateRecord_huf_decompress_c_950> = core::mem::MaybeUninit::uninit();
let __slate_slot_130: *mut __SlateRecord_huf_decompress_c_950 = core::ptr::addr_of_mut!(entry_11) as *mut __SlateRecord_huf_decompress_c_950;
let mut entry_13: core::mem::MaybeUninit<__SlateRecord_huf_decompress_c_950> = core::mem::MaybeUninit::uninit();
let __slate_slot_146: *mut __SlateRecord_huf_decompress_c_950 = core::ptr::addr_of_mut!(entry_13) as *mut __SlateRecord_huf_decompress_c_950;
let mut entry_14: core::mem::MaybeUninit<__SlateRecord_huf_decompress_c_950> = core::mem::MaybeUninit::uninit();
let __slate_slot_154: *mut __SlateRecord_huf_decompress_c_950 = core::ptr::addr_of_mut!(entry_14) as *mut __SlateRecord_huf_decompress_c_950;
let mut entry_15: core::mem::MaybeUninit<__SlateRecord_huf_decompress_c_950> = core::mem::MaybeUninit::uninit();
let __slate_slot_162: *mut __SlateRecord_huf_decompress_c_950 = core::ptr::addr_of_mut!(entry_15) as *mut __SlateRecord_huf_decompress_c_950;
let mut entry_17: core::mem::MaybeUninit<__SlateRecord_huf_decompress_c_950> = core::mem::MaybeUninit::uninit();
let __slate_slot_178: *mut __SlateRecord_huf_decompress_c_950 = core::ptr::addr_of_mut!(entry_17) as *mut __SlateRecord_huf_decompress_c_950;
let mut entry_18: core::mem::MaybeUninit<__SlateRecord_huf_decompress_c_950> = core::mem::MaybeUninit::uninit();
let __slate_slot_186: *mut __SlateRecord_huf_decompress_c_950 = core::ptr::addr_of_mut!(entry_18) as *mut __SlateRecord_huf_decompress_c_950;
let mut entry: core::mem::MaybeUninit<__SlateRecord_huf_decompress_c_950> = core::mem::MaybeUninit::uninit();
let __slate_slot_194: *mut __SlateRecord_huf_decompress_c_950 = core::ptr::addr_of_mut!(entry) as *mut __SlateRecord_huf_decompress_c_950;
// Compute olimit
// Each loop does 5 table lookups for each of the 4 streams.
// Each table lookup consumes up to 11 bits of input, and produces
// up to 2 bytes of output.
//
// We can consume up to 7 bytes of input per iteration per stream.
// We also know that each input pointer is >= ip[0]. So we can run
// iters loops before running out of input.
let mut iters: u64 = unsafe { core::mem::zeroed() };
let mut stream: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_206: *mut i32 = core::ptr::addr_of_mut!(stream) as *mut i32;
let mut olimit: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_207: *mut *mut u8 = core::ptr::addr_of_mut!(olimit) as *mut *mut u8;
let mut ilowest: *const u8 = unsafe { core::mem::zeroed() };
let mut dtable: *const __SlateRecord_huf_decompress_c_950 = unsafe { core::mem::zeroed() };
let mut oend: core::mem::MaybeUninit<__SlateAlign16<[*mut u8; 4]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_210: *mut [*mut u8; 4] = core::ptr::addr_of_mut!(oend) as *mut [*mut u8; 4];
let mut op: core::mem::MaybeUninit<__SlateAlign16<[*mut u8; 4]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_211: *mut [*mut u8; 4] = core::ptr::addr_of_mut!(op) as *mut [*mut u8; 4];
let mut ip: core::mem::MaybeUninit<__SlateAlign16<[*const u8; 4]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_212: *mut [*const u8; 4] = core::ptr::addr_of_mut!(ip) as *mut [*const u8; 4];
let mut bits: core::mem::MaybeUninit<__SlateAlign16<[u64; 4]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_213: *mut [u64; 4] = core::ptr::addr_of_mut!(bits) as *mut [u64; 4];
unsafe {
let __slate_auto_bytes_1 = __slate_slot_213 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 32usize);
let __slate_auto_bytes_2 = __slate_slot_212 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 32usize);
let __slate_auto_bytes_3 = __slate_slot_211 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 32usize);
let __slate_auto_bytes_4 = __slate_slot_210 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 32usize);
dtable = (*args).dt as *const __SlateRecord_huf_decompress_c_950;
ilowest = (*args).ilowest;
// Copy the arguments to local registers.
ZSTD_memcpy!(core::ptr::addr_of_mut!(*__slate_slot_213), core::ptr::addr_of_mut!((*args).bits), core::mem::size_of::<[u64; 4]>() as u64);
ZSTD_memcpy__3!(core::ptr::addr_of_mut!(*__slate_slot_212) as *mut (), core::ptr::addr_of_mut!((*args).ip), core::mem::size_of::<[*const u8; 4]>() as u64);
ZSTD_memcpy!(core::ptr::addr_of_mut!(*__slate_slot_211), core::ptr::addr_of_mut!((*args).op), core::mem::size_of::<[*mut u8; 4]>() as u64);
(*__slate_slot_210)[0] = (*__slate_slot_211)[1];
(*__slate_slot_210)[1] = (*__slate_slot_211)[2];
(*__slate_slot_210)[2] = (*__slate_slot_211)[3];
(*__slate_slot_210)[3] = (*args).oend;
'__loop_48: loop {
let __slate_auto_bytes = __slate_slot_207 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_206 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// Assert loop preconditions
*__slate_slot_206 = 0;
while *__slate_slot_206 < 4 {
let __v205: i32 = *__slate_slot_206;
*__slate_slot_206 = __v205.wrapping_add(1);
}
iters = (((*__slate_slot_212)[0].offset_from(ilowest as *const u8) as i64) as u64) / 7;
// Each iteration can produce up to 10 bytes of output per stream.
// Each output stream my advance at different rates. So take the
// minimum number of safe iterations among all the output streams.
*__slate_slot_206 = 0;
while *__slate_slot_206 < 4 {
let oiters: u64 = (((*__slate_slot_210)[*__slate_slot_206 as usize].offset_from((*__slate_slot_211)[*__slate_slot_206 as usize] as *mut u8) as i64) as u64) / 10;
iters = MIN!(iters, oiters);
let __v201: i32 = *__slate_slot_206;
*__slate_slot_206 = __v201.wrapping_add(1);
}
// Each iteration produces at least 5 output symbols. So until
// op[3] crosses olimit, we know we haven't executed iters
// iterations yet. This saves us maintaining an iters counter,
// at the expense of computing the remaining # of iterations
// more frequently.
*__slate_slot_207 = (*__slate_slot_211)[3].wrapping_offset(iters.wrapping_mul(5) as isize);
// Exit the fast decoding loop once we reach the end.
if (*__slate_slot_211)[3] == *__slate_slot_207 {
break;
} else {
// Exit the decoding loop if any input pointer has crossed the
// previous one. This indicates corruption, and a precondition
// to our loop is that ip[i] >= ip[0].
*__slate_slot_206 = 1;
while *__slate_slot_206 < 4 {
if (*__slate_slot_212)[*__slate_slot_206 as usize] < (*__slate_slot_212)[(*__slate_slot_206).wrapping_sub(1) as usize] {
break '__loop_48;
} else {
let __v199: i32 = *__slate_slot_206;
*__slate_slot_206 = __v199.wrapping_add(1);
}
}
*__slate_slot_206 = 1;
while *__slate_slot_206 < 4 {
let __v197: i32 = *__slate_slot_206;
*__slate_slot_206 = __v197.wrapping_add(1);
}
// Manually unroll the loop because compilers don't consistently
// unroll the inner loops, which destroys performance.
loop {
// Decode 5 symbols from each of the first 3 streams.
// The final stream will be decoded during the reload phase
// to reduce register pressure.
loop {
let index_9: i32 = ((*__slate_slot_213)[0] >> 53i32) as i32;
core::ptr::write(__slate_slot_194, *dtable.wrapping_offset(index_9 as isize));
MEM_write16((*__slate_slot_211)[0] as *mut (), (*__slate_slot_194).sequence);
let __v193: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(0 as isize);
*__v193 = *__v193 << (((*__slate_slot_194).nbBits as i32) & 63);
let __v190: *mut *mut u8 = ((*__slate_slot_211).as_mut_ptr() as *mut *mut u8).wrapping_offset(0 as isize);
let __v189: *mut u8 = *__v190;
*__v190 = __v189.wrapping_offset(((*__slate_slot_194).length as i32) as isize);
if true {
loop {
let index: i32 = ((*__slate_slot_213)[1] >> 53i32) as i32;
core::ptr::write(__slate_slot_186, *dtable.wrapping_offset(index as isize));
MEM_write16((*__slate_slot_211)[1] as *mut (), (*__slate_slot_186).sequence);
let __v185: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(1 as isize);
*__v185 = *__v185 << (((*__slate_slot_186).nbBits as i32) & 63);
let __v182: *mut *mut u8 = ((*__slate_slot_211).as_mut_ptr() as *mut *mut u8).wrapping_offset(1 as isize);
let __v181: *mut u8 = *__v182;
*__v182 = __v181.wrapping_offset(((*__slate_slot_186).length as i32) as isize);
if true {
break;
}
}
loop {
let index: i32 = ((*__slate_slot_213)[2] >> 53i32) as i32;
core::ptr::write(__slate_slot_178, *dtable.wrapping_offset(index as isize));
MEM_write16((*__slate_slot_211)[2] as *mut (), (*__slate_slot_178).sequence);
let __v177: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(2 as isize);
*__v177 = *__v177 << (((*__slate_slot_178).nbBits as i32) & 63);
let __v174: *mut *mut u8 = ((*__slate_slot_211).as_mut_ptr() as *mut *mut u8).wrapping_offset(2 as isize);
let __v173: *mut u8 = *__v174;
*__v174 = __v173.wrapping_offset(((*__slate_slot_178).length as i32) as isize);
if true {
break;
}
}
while false {
}
if true {
break;
}
}
}
loop {
let index_7: i32 = ((*__slate_slot_213)[0] >> 53i32) as i32;
core::ptr::write(__slate_slot_162, *dtable.wrapping_offset(index_7 as isize));
MEM_write16((*__slate_slot_211)[0] as *mut (), (*__slate_slot_162).sequence);
let __v161: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(0 as isize);
*__v161 = *__v161 << (((*__slate_slot_162).nbBits as i32) & 63);
let __v158: *mut *mut u8 = ((*__slate_slot_211).as_mut_ptr() as *mut *mut u8).wrapping_offset(0 as isize);
let __v157: *mut u8 = *__v158;
*__v158 = __v157.wrapping_offset(((*__slate_slot_162).length as i32) as isize);
if true {
loop {
let index: i32 = ((*__slate_slot_213)[1] >> 53i32) as i32;
core::ptr::write(__slate_slot_154, *dtable.wrapping_offset(index as isize));
MEM_write16((*__slate_slot_211)[1] as *mut (), (*__slate_slot_154).sequence);
let __v153: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(1 as isize);
*__v153 = *__v153 << (((*__slate_slot_154).nbBits as i32) & 63);
let __v150: *mut *mut u8 = ((*__slate_slot_211).as_mut_ptr() as *mut *mut u8).wrapping_offset(1 as isize);
let __v149: *mut u8 = *__v150;
*__v150 = __v149.wrapping_offset(((*__slate_slot_154).length as i32) as isize);
if true {
break;
}
}
loop {
let index: i32 = ((*__slate_slot_213)[2] >> 53i32) as i32;
core::ptr::write(__slate_slot_146, *dtable.wrapping_offset(index as isize));
MEM_write16((*__slate_slot_211)[2] as *mut (), (*__slate_slot_146).sequence);
let __v145: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(2 as isize);
*__v145 = *__v145 << (((*__slate_slot_146).nbBits as i32) & 63);
let __v142: *mut *mut u8 = ((*__slate_slot_211).as_mut_ptr() as *mut *mut u8).wrapping_offset(2 as isize);
let __v141: *mut u8 = *__v142;
*__v142 = __v141.wrapping_offset(((*__slate_slot_146).length as i32) as isize);
if true {
break;
}
}
while false {
}
if true {
break;
}
}
}
loop {
let index_5: i32 = ((*__slate_slot_213)[0] >> 53i32) as i32;
core::ptr::write(__slate_slot_130, *dtable.wrapping_offset(index_5 as isize));
MEM_write16((*__slate_slot_211)[0] as *mut (), (*__slate_slot_130).sequence);
let __v129: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(0 as isize);
*__v129 = *__v129 << (((*__slate_slot_130).nbBits as i32) & 63);
let __v126: *mut *mut u8 = ((*__slate_slot_211).as_mut_ptr() as *mut *mut u8).wrapping_offset(0 as isize);
let __v125: *mut u8 = *__v126;
*__v126 = __v125.wrapping_offset(((*__slate_slot_130).length as i32) as isize);
if true {
loop {
let index: i32 = ((*__slate_slot_213)[1] >> 53i32) as i32;
core::ptr::write(__slate_slot_122, *dtable.wrapping_offset(index as isize));
MEM_write16((*__slate_slot_211)[1] as *mut (), (*__slate_slot_122).sequence);
let __v121: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(1 as isize);
*__v121 = *__v121 << (((*__slate_slot_122).nbBits as i32) & 63);
let __v118: *mut *mut u8 = ((*__slate_slot_211).as_mut_ptr() as *mut *mut u8).wrapping_offset(1 as isize);
let __v117: *mut u8 = *__v118;
*__v118 = __v117.wrapping_offset(((*__slate_slot_122).length as i32) as isize);
if true {
break;
}
}
loop {
let index: i32 = ((*__slate_slot_213)[2] >> 53i32) as i32;
core::ptr::write(__slate_slot_114, *dtable.wrapping_offset(index as isize));
MEM_write16((*__slate_slot_211)[2] as *mut (), (*__slate_slot_114).sequence);
let __v113: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(2 as isize);
*__v113 = *__v113 << (((*__slate_slot_114).nbBits as i32) & 63);
let __v110: *mut *mut u8 = ((*__slate_slot_211).as_mut_ptr() as *mut *mut u8).wrapping_offset(2 as isize);
let __v109: *mut u8 = *__v110;
*__v110 = __v109.wrapping_offset(((*__slate_slot_114).length as i32) as isize);
if true {
break;
}
}
while false {
}
if true {
break;
}
}
}
loop {
let index_3: i32 = ((*__slate_slot_213)[0] >> 53i32) as i32;
core::ptr::write(__slate_slot_98, *dtable.wrapping_offset(index_3 as isize));
MEM_write16((*__slate_slot_211)[0] as *mut (), (*__slate_slot_98).sequence);
let __v97: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(0 as isize);
*__v97 = *__v97 << (((*__slate_slot_98).nbBits as i32) & 63);
let __v94: *mut *mut u8 = ((*__slate_slot_211).as_mut_ptr() as *mut *mut u8).wrapping_offset(0 as isize);
let __v93: *mut u8 = *__v94;
*__v94 = __v93.wrapping_offset(((*__slate_slot_98).length as i32) as isize);
if true {
loop {
let index: i32 = ((*__slate_slot_213)[1] >> 53i32) as i32;
core::ptr::write(__slate_slot_90, *dtable.wrapping_offset(index as isize));
MEM_write16((*__slate_slot_211)[1] as *mut (), (*__slate_slot_90).sequence);
let __v89: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(1 as isize);
*__v89 = *__v89 << (((*__slate_slot_90).nbBits as i32) & 63);
let __v86: *mut *mut u8 = ((*__slate_slot_211).as_mut_ptr() as *mut *mut u8).wrapping_offset(1 as isize);
let __v85: *mut u8 = *__v86;
*__v86 = __v85.wrapping_offset(((*__slate_slot_90).length as i32) as isize);
if true {
break;
}
}
loop {
let index: i32 = ((*__slate_slot_213)[2] >> 53i32) as i32;
core::ptr::write(__slate_slot_82, *dtable.wrapping_offset(index as isize));
MEM_write16((*__slate_slot_211)[2] as *mut (), (*__slate_slot_82).sequence);
let __v81: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(2 as isize);
*__v81 = *__v81 << (((*__slate_slot_82).nbBits as i32) & 63);
let __v78: *mut *mut u8 = ((*__slate_slot_211).as_mut_ptr() as *mut *mut u8).wrapping_offset(2 as isize);
let __v77: *mut u8 = *__v78;
*__v78 = __v77.wrapping_offset(((*__slate_slot_82).length as i32) as isize);
if true {
break;
}
}
while false {
}
if true {
break;
}
}
}
loop {
let index_1: i32 = ((*__slate_slot_213)[0] >> 53i32) as i32;
core::ptr::write(__slate_slot_66, *dtable.wrapping_offset(index_1 as isize));
MEM_write16((*__slate_slot_211)[0] as *mut (), (*__slate_slot_66).sequence);
let __v65: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(0 as isize);
*__v65 = *__v65 << (((*__slate_slot_66).nbBits as i32) & 63);
let __v62: *mut *mut u8 = ((*__slate_slot_211).as_mut_ptr() as *mut *mut u8).wrapping_offset(0 as isize);
let __v61: *mut u8 = *__v62;
*__v62 = __v61.wrapping_offset(((*__slate_slot_66).length as i32) as isize);
if true {
loop {
let index: i32 = ((*__slate_slot_213)[1] >> 53i32) as i32;
core::ptr::write(__slate_slot_58, *dtable.wrapping_offset(index as isize));
MEM_write16((*__slate_slot_211)[1] as *mut (), (*__slate_slot_58).sequence);
let __v57: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(1 as isize);
*__v57 = *__v57 << (((*__slate_slot_58).nbBits as i32) & 63);
let __v54: *mut *mut u8 = ((*__slate_slot_211).as_mut_ptr() as *mut *mut u8).wrapping_offset(1 as isize);
let __v53: *mut u8 = *__v54;
*__v54 = __v53.wrapping_offset(((*__slate_slot_58).length as i32) as isize);
if true {
break;
}
}
loop {
let index: i32 = ((*__slate_slot_213)[2] >> 53i32) as i32;
core::ptr::write(__slate_slot_50, *dtable.wrapping_offset(index as isize));
MEM_write16((*__slate_slot_211)[2] as *mut (), (*__slate_slot_50).sequence);
let __v49: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(2 as isize);
*__v49 = *__v49 << (((*__slate_slot_50).nbBits as i32) & 63);
let __v46: *mut *mut u8 = ((*__slate_slot_211).as_mut_ptr() as *mut *mut u8).wrapping_offset(2 as isize);
let __v45: *mut u8 = *__v46;
*__v46 = __v45.wrapping_offset(((*__slate_slot_50).length as i32) as isize);
if true {
break;
}
}
while false {
}
if true {
break;
}
}
}
// Decode one symbol from the final stream
HUF_4X2_DECODE_SYMBOL!(3i32, true, *__slate_slot_213, dtable, *__slate_slot_211);
// Decode 4 symbols from the final stream & reload bitstreams.
// The final stream is reloaded last, meaning that all 5 symbols
// are decoded from the final stream before it is reloaded.
loop {
HUF_4X2_DECODE_SYMBOL!(3i32, true, *__slate_slot_213, dtable, *__slate_slot_211);
let ctz_1: i32 = ZSTD_countTrailingZeros64((*__slate_slot_213)[0]) as i32;
let nbBits_1: i32 = ctz_1 & 7;
let nbBytes_1: i32 = ctz_1 >> 3i32;
let __v32: *mut *const u8 = ((*__slate_slot_212).as_mut_ptr() as *mut *const u8).wrapping_offset(0 as isize);
let __v31: *const u8 = *__v32;
*__v32 = __v31.wrapping_offset(-(nbBytes_1 as isize));
(*__slate_slot_213)[0] = MEM_read64((*__slate_slot_212)[0] as *const ()) | 1;
let __v29: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(0 as isize);
*__v29 = *__v29 << nbBits_1;
if true {
loop {
HUF_4X2_DECODE_SYMBOL!(3i32, true, *__slate_slot_213, dtable, *__slate_slot_211);
let ctz: i32 = ZSTD_countTrailingZeros64((*__slate_slot_213)[1]) as i32;
let nbBits: i32 = ctz & 7;
let nbBytes: i32 = ctz >> 3i32;
let __v23: *mut *const u8 = ((*__slate_slot_212).as_mut_ptr() as *mut *const u8).wrapping_offset(1 as isize);
let __v22: *const u8 = *__v23;
*__v23 = __v22.wrapping_offset(-(nbBytes as isize));
(*__slate_slot_213)[1] = MEM_read64((*__slate_slot_212)[1] as *const ()) | 1;
let __v20: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(1 as isize);
*__v20 = *__v20 << nbBits;
if true {
break;
}
}
loop {
HUF_4X2_DECODE_SYMBOL!(3i32, true, *__slate_slot_213, dtable, *__slate_slot_211);
let ctz: i32 = ZSTD_countTrailingZeros64((*__slate_slot_213)[2]) as i32;
let nbBits: i32 = ctz & 7;
let nbBytes: i32 = ctz >> 3i32;
let __v14: *mut *const u8 = ((*__slate_slot_212).as_mut_ptr() as *mut *const u8).wrapping_offset(2 as isize);
let __v13: *const u8 = *__v14;
*__v14 = __v13.wrapping_offset(-(nbBytes as isize));
(*__slate_slot_213)[2] = MEM_read64((*__slate_slot_212)[2] as *const ()) | 1;
let __v11: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(2 as isize);
*__v11 = *__v11 << nbBits;
if true {
break;
}
}
loop {
HUF_4X2_DECODE_SYMBOL!(3i32, true, *__slate_slot_213, dtable, *__slate_slot_211);
let ctz: i32 = ZSTD_countTrailingZeros64((*__slate_slot_213)[3]) as i32;
let nbBits: i32 = ctz & 7;
let nbBytes: i32 = ctz >> 3i32;
let __v5: *mut *const u8 = ((*__slate_slot_212).as_mut_ptr() as *mut *const u8).wrapping_offset(3 as isize);
let __v4: *const u8 = *__v5;
*__v5 = __v4.wrapping_offset(-(nbBytes as isize));
(*__slate_slot_213)[3] = MEM_read64((*__slate_slot_212)[3] as *const ()) | 1;
let __v2: *mut u64 = ((*__slate_slot_213).as_mut_ptr() as *mut u64).wrapping_offset(3 as isize);
*__v2 = *__v2 << nbBits;
if true {
break;
}
}
if true {
break;
}
}
}
if !((*__slate_slot_211)[3] < *__slate_slot_207) {
continue '__loop_48;
}
}
}
}
ZSTD_memcpy!(core::ptr::addr_of_mut!((*args).bits), core::ptr::addr_of_mut!(*__slate_slot_213), core::mem::size_of::<[u64; 4]>() as u64); // Save the final values of each of the state variables back to args.
ZSTD_memcpy__3!(core::ptr::addr_of_mut!((*args).ip) as *mut (), core::ptr::addr_of_mut!(*__slate_slot_212), core::mem::size_of::<[*const u8; 4]>() as u64);
ZSTD_memcpy!(core::ptr::addr_of_mut!((*args).op), core::ptr::addr_of_mut!(*__slate_slot_211), core::mem::size_of::<[*mut u8; 4]>() as u64);
}
}

#[target_feature(enable = "lzcnt,bmi1,bmi2")]
unsafe fn HUF_decompress4X2_usingDTable_internal_fast(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32, loopFn: Option<unsafe extern "C" fn(*mut __SlateRecord_huf_decompress_c_178)>) -> u64 {
let dt: *const () = DTable.wrapping_offset(1 as isize) as *const ();
let oend: *mut u8 = ZSTD_maybeNullPtrAdd(dst as *mut u8, dstSize as i64);
let mut args: __SlateRecord_huf_decompress_c_178 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(args) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 152usize);
}
let ret: u64 = HUF_DecompressFastArgs_init(unsafe { &mut (*core::ptr::addr_of_mut!(args)) }, dst, dstSize, cSrc, cSrcSize, DTable);
loop {
let err_code: u64 = ret;
if ERR_isError(err_code) != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"Failed to init asm args".as_ptr() as *const u8) };
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
if ret == 0 {
return 0;
}
unsafe { loopFn.unwrap()(core::ptr::addr_of_mut!(args)) };
// note : op4 already verified within main loop
// finish bitStreams one by one
let segmentSize: u64 = dstSize.wrapping_add(3) / 4;
let mut segmentEnd: *mut u8 = dst as *mut u8;
for i in 0i32..4i32 {
let mut bit: __SlateRecord_bitstream_h_90 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bit) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
if segmentSize <= (((unsafe { oend.offset_from(segmentEnd as *mut u8) }) as i64) as u64) {
let __v5: *mut u8 = segmentEnd;
segmentEnd = __v5.wrapping_offset(segmentSize as isize);
} else {
segmentEnd = oend;
}
loop {
let err_code: u64 = HUF_initRemainingDStream(unsafe { &mut (*core::ptr::addr_of_mut!(bit)) }, unsafe { &(*(core::ptr::addr_of_mut!(args) as *const __SlateRecord_huf_decompress_c_178)) }, i, segmentEnd);
if ERR_isError(err_code) != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"corruption".as_ptr() as *const u8) };
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
let __v12: u64 = HUF_decodeStreamX2(args.op[i as usize], core::ptr::addr_of_mut!(bit), segmentEnd, dt as *const __SlateRecord_huf_decompress_c_950, 11u32);
let __v13: *mut *mut u8 = (args.op.as_mut_ptr() as *mut *mut u8).wrapping_offset(i as isize);
let __v14: *mut u8 = unsafe { *__v13 };
unsafe {
*__v13 = __v14.wrapping_offset(__v12 as isize);
}
if args.op[i as usize] != segmentEnd {
return ERROR!();
}
}
// decoded size
dstSize
}

fn HUF_decompress4X2_usingDTable_internal(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32, flags: i32) -> u64 {
let mut fallbackFn: Option<unsafe extern "C" fn(*mut (), u64, *const (), u64, *const u32) -> u64> = Some(HUF_decompress4X2_usingDTable_internal_default);
let loopFn: Option<unsafe extern "C" fn(*mut __SlateRecord_huf_decompress_c_178)> = Some(HUF_decompress4X2_usingDTable_internal_fast_c_loop);
if flags & 1 != 0 {
fallbackFn = Some(HUF_decompress4X2_usingDTable_internal_bmi2);
} else {
return unsafe { fallbackFn.unwrap()(dst, dstSize, cSrc, cSrcSize, DTable) };
}
if true && flags & 32 == 0 {
let ret: u64 = unsafe { HUF_decompress4X2_usingDTable_internal_fast(dst, dstSize, cSrc, cSrcSize, DTable, loopFn) };
if ret != 0 {
return ret;
}
}
unsafe { fallbackFn.unwrap()(dst, dstSize, cSrc, cSrcSize, DTable) }
}

fn HUF_decompress1X2_usingDTable_internal_default(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32) -> u64 {
HUF_decompress1X2_usingDTable_internal_body(dst, dstSize, cSrc, cSrcSize, DTable)
}

#[target_feature(enable = "lzcnt,bmi1,bmi2")]
unsafe fn HUF_decompress1X2_usingDTable_internal_bmi2(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32) -> u64 {
HUF_decompress1X2_usingDTable_internal_body(dst, dstSize, cSrc, cSrcSize, DTable)
}

fn HUF_decompress1X2_usingDTable_internal(dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32, flags: i32) -> u64 {
if flags & 1 != 0 {
return unsafe { HUF_decompress1X2_usingDTable_internal_bmi2(dst, dstSize, cSrc, cSrcSize, DTable) };
}
HUF_decompress1X2_usingDTable_internal_default(dst, dstSize, cSrc, cSrcSize, DTable)
}

#[unsafe(no_mangle)]
extern "C" fn HUF_decompress1X2_DCtx_wksp(DCtx: *mut u32, dst: *mut (), dstSize: u64, cSrc: *const (), mut cSrcSize: u64, workSpace: *mut (), wkspSize: u64, flags: i32) -> u64 {
let mut ip: *const u8 = cSrc as *const u8;
let hSize: u64 = HUF_readDTableX2_wksp(DCtx, cSrc, cSrcSize, workSpace, wkspSize, flags);
if ERR_isError(hSize) != 0 {
return hSize;
}
if hSize >= cSrcSize {
return ERROR__2!();
}
let __v0: *const u8 = ip;
ip = __v0.wrapping_offset(hSize as isize);
cSrcSize = cSrcSize.wrapping_sub(hSize);
HUF_decompress1X2_usingDTable_internal(dst, dstSize, ip as *const (), cSrcSize, DCtx as *const u32, flags)
}

fn HUF_decompress4X2_DCtx_wksp(dctx: *mut u32, dst: *mut (), dstSize: u64, cSrc: *const (), mut cSrcSize: u64, workSpace: *mut (), wkspSize: u64, flags: i32) -> u64 {
let mut ip: *const u8 = cSrc as *const u8;
let hSize: u64 = HUF_readDTableX2_wksp(dctx, cSrc, cSrcSize, workSpace, wkspSize, flags);
if ERR_isError(hSize) != 0 {
return hSize;
}
if hSize >= cSrcSize {
return ERROR__2!();
}
let __v0: *const u8 = ip;
ip = __v0.wrapping_offset(hSize as isize);
cSrcSize = cSrcSize.wrapping_sub(hSize);
HUF_decompress4X2_usingDTable_internal(dst, dstSize, ip as *const (), cSrcSize, dctx as *const u32, flags)
}

// Universal decompression selectors
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_huf_decompress_c_1790 {
    tableTime: u32,
    decode256Time: u32,
}


/// Quantization
/// single, double
/// single, double, quad
/// Q==0 : impossible
/// Q==1 : impossible
/// Q == 2 : 12-18%
/// Q == 3 : 18-25%
/// Q == 4 : 25-32%
/// Q == 5 : 32-38%
/// Q == 6 : 38-44%
/// Q == 7 : 44-50%
/// Q == 8 : 50-56%
/// Q == 9 : 56-62%
/// Q ==10 : 62-69%
/// Q ==11 : 69-75%
/// Q ==12 : 75-81%
/// Q ==13 : 81-87%
/// Q ==14 : 87-93%
/// Q ==15 : 93-99%
static algoTime: __SlateAlign16<[[__SlateRecord_huf_decompress_c_1790; 2]; 16]> = __SlateAlign16([[__SlateRecord_huf_decompress_c_1790 { tableTime: 0u32, decode256Time: 0u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 1u32, decode256Time: 1u32 }], [__SlateRecord_huf_decompress_c_1790 { tableTime: 0u32, decode256Time: 0u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 1u32, decode256Time: 1u32 }], [__SlateRecord_huf_decompress_c_1790 { tableTime: 150u32, decode256Time: 216u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 381u32, decode256Time: 119u32 }], [__SlateRecord_huf_decompress_c_1790 { tableTime: 170u32, decode256Time: 205u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 514u32, decode256Time: 112u32 }], [__SlateRecord_huf_decompress_c_1790 { tableTime: 177u32, decode256Time: 199u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 539u32, decode256Time: 110u32 }], [__SlateRecord_huf_decompress_c_1790 { tableTime: 197u32, decode256Time: 194u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 644u32, decode256Time: 107u32 }], [__SlateRecord_huf_decompress_c_1790 { tableTime: 221u32, decode256Time: 192u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 735u32, decode256Time: 107u32 }], [__SlateRecord_huf_decompress_c_1790 { tableTime: 256u32, decode256Time: 189u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 881u32, decode256Time: 106u32 }], [__SlateRecord_huf_decompress_c_1790 { tableTime: 359u32, decode256Time: 188u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 1167u32, decode256Time: 109u32 }], [__SlateRecord_huf_decompress_c_1790 { tableTime: 582u32, decode256Time: 187u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 1570u32, decode256Time: 114u32 }], [__SlateRecord_huf_decompress_c_1790 { tableTime: 688u32, decode256Time: 187u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 1712u32, decode256Time: 122u32 }], [__SlateRecord_huf_decompress_c_1790 { tableTime: 825u32, decode256Time: 186u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 1965u32, decode256Time: 136u32 }], [__SlateRecord_huf_decompress_c_1790 { tableTime: 976u32, decode256Time: 185u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 2131u32, decode256Time: 150u32 }], [__SlateRecord_huf_decompress_c_1790 { tableTime: 1180u32, decode256Time: 186u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 2070u32, decode256Time: 175u32 }], [__SlateRecord_huf_decompress_c_1790 { tableTime: 1377u32, decode256Time: 185u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 1731u32, decode256Time: 202u32 }], [__SlateRecord_huf_decompress_c_1790 { tableTime: 1412u32, decode256Time: 185u32 }, __SlateRecord_huf_decompress_c_1790 { tableTime: 1695u32, decode256Time: 202u32 }]]);

/// HUF_selectDecoder() :
///  Tells which decoder is likely to decode faster,
///  based on a set of pre-computed metrics.
/// @return : 0==HUF_decompress4X1, 1==HUF_decompress4X2 .
///  Assumption : 0 < dstSize <= 128 KB
#[unsafe(no_mangle)]
extern "C" fn HUF_selectDecoder(dstSize: u64, cSrcSize: u64) -> u32 {
// decoder timing evaluation
let Q: u32 = if cSrcSize >= dstSize { 15 } else { (cSrcSize.wrapping_mul(16) / dstSize) as u32 }; // Q < 16
let D256: u32 = (dstSize >> 8i32) as u32;
let DTime0: u32 = unsafe { algoTime.0[Q as usize][0].tableTime }.wrapping_add(unsafe { algoTime.0[Q as usize][0].decode256Time }.wrapping_mul(D256));
let mut DTime1: u32 = unsafe { algoTime.0[Q as usize][1].tableTime }.wrapping_add(unsafe { algoTime.0[Q as usize][1].decode256Time }.wrapping_mul(D256));
DTime1 = DTime1.wrapping_add(DTime1 >> 5i32); // small advantage to algorithm using less memory, to reduce cache eviction
(DTime1 < DTime0) as u32
}

#[unsafe(no_mangle)]
extern "C" fn HUF_decompress1X_DCtx_wksp(dctx: *mut u32, dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, workSpace: *mut (), wkspSize: u64, flags: i32) -> u64 {
// validation checks
if dstSize == 0 {
return ERROR__3!();
}
if cSrcSize > dstSize {
return ERROR!();
}
// invalid
if cSrcSize == dstSize {
ZSTD_memcpy__5!(dst, cSrc, dstSize);
return dstSize;
}
// not compressed
if cSrcSize == 1 {
ZSTD_memset__2!(dst, unsafe { *(cSrc as *const u8) }, dstSize);
return dstSize;
}
// RLE
let algoNb: u32 = HUF_selectDecoder(dstSize, cSrcSize);
let __v0: u64 = if algoNb != 0 {
HUF_decompress1X2_DCtx_wksp(dctx, dst, dstSize, cSrc, cSrcSize, workSpace, wkspSize, flags)
} else {
HUF_decompress1X1_DCtx_wksp(dctx, dst, dstSize, cSrc, cSrcSize, workSpace, wkspSize, flags)
};
__v0
}

#[unsafe(no_mangle)]
extern "C" fn HUF_decompress1X_usingDTable(dst: *mut (), maxDstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32, flags: i32) -> u64 {
let dtd: __SlateRecord_huf_decompress_c_138 = HUF_getDTableDesc(DTable);
let __v0: u64 = if dtd.tableType != 0 {
HUF_decompress1X2_usingDTable_internal(dst, maxDstSize, cSrc, cSrcSize, DTable, flags)
} else {
HUF_decompress1X1_usingDTable_internal(dst, maxDstSize, cSrc, cSrcSize, DTable, flags)
};
__v0
}

#[unsafe(no_mangle)]
extern "C" fn HUF_decompress1X1_DCtx_wksp(dctx: *mut u32, dst: *mut (), dstSize: u64, cSrc: *const (), mut cSrcSize: u64, workSpace: *mut (), wkspSize: u64, flags: i32) -> u64 {
let mut ip: *const u8 = cSrc as *const u8;
let hSize: u64 = HUF_readDTableX1_wksp(dctx, cSrc, cSrcSize, workSpace, wkspSize, flags);
if ERR_isError(hSize) != 0 {
return hSize;
}
if hSize >= cSrcSize {
return ERROR__2!();
}
let __v0: *const u8 = ip;
ip = __v0.wrapping_offset(hSize as isize);
cSrcSize = cSrcSize.wrapping_sub(hSize);
HUF_decompress1X1_usingDTable_internal(dst, dstSize, ip as *const (), cSrcSize, dctx as *const u32, flags)
}

#[unsafe(no_mangle)]
extern "C" fn HUF_decompress4X_usingDTable(dst: *mut (), maxDstSize: u64, cSrc: *const (), cSrcSize: u64, DTable: *const u32, flags: i32) -> u64 {
let dtd: __SlateRecord_huf_decompress_c_138 = HUF_getDTableDesc(DTable);
let __v0: u64 = if dtd.tableType != 0 {
HUF_decompress4X2_usingDTable_internal(dst, maxDstSize, cSrc, cSrcSize, DTable, flags)
} else {
HUF_decompress4X1_usingDTable_internal(dst, maxDstSize, cSrc, cSrcSize, DTable, flags)
};
__v0
}

#[unsafe(no_mangle)]
extern "C" fn HUF_decompress4X_hufOnly_wksp(dctx: *mut u32, dst: *mut (), dstSize: u64, cSrc: *const (), cSrcSize: u64, workSpace: *mut (), wkspSize: u64, flags: i32) -> u64 {
// validation checks
if dstSize == 0 {
return ERROR__3!();
}
if cSrcSize == 0 {
return ERROR!();
}
let algoNb: u32 = HUF_selectDecoder(dstSize, cSrcSize);
let __v0: u64 = if algoNb != 0 {
HUF_decompress4X2_DCtx_wksp(dctx, dst, dstSize, cSrc, cSrcSize, workSpace, wkspSize, flags)
} else {
HUF_decompress4X1_DCtx_wksp(dctx, dst, dstSize, cSrc, cSrcSize, workSpace, wkspSize, flags)
};
__v0
}
