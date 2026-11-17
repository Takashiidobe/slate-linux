
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0-only
//!
//! -*- linux-c -*- ------------------------------------------------------- *
//!
//! Copyright (C) 1991, 1992 Linus Torvalds
//! Copyright 2007 rPath, Inc. - All Rights Reserved
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! Invalid argument
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
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-pure-function-attribute
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
// Math result not representable
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
/// Works only for digits and letters, but small and fast
macro_rules! TOLOWER {
($x:expr) => {
(($x) as i32) | 32
};
}

macro_rules! unlikely {
($x:expr) => {
(($x) != 0) as i32
};
}

unsafe extern "C" {
    fn __builtin_expect(arg0: i32, arg1: i32) -> i32;
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_string_c_205 {
    v64: u64,
    v32: [u32; 2],
}


fn isdigit(ch: i32) -> i32 {
(ch >= 48 && ch <= 57) as i32
}

fn isxdigit(ch: i32) -> i32 {
if isdigit(ch) != 0 {
return 1;
}
if ch >= 97 && ch <= 102 {
return 1;
}
(ch >= 65 && ch <= 70) as i32
}

#[unsafe(export_name = "memcmp")]
unsafe extern "C" fn __slate_export_memcmp(s1: *const core::ffi::c_void, s2: *const core::ffi::c_void, len: usize) -> core::ffi::c_int {
return memcmp(s1 as *const (), s2 as *const (), len as u32) as core::ffi::c_int;
}

// Very basic string functions
// Undef these macros so that the functions that we provide
// here will have the correct names regardless of how string.h
// may have chosen to #define them.
fn memcmp(mut s1: *const (), mut s2: *const (), mut len: u32) -> i32 {
let mut diff: bool = false;
// Make sure ZF is properly set in the len==0 case because in it,
// RCX==0 and the REPE; CMPSB won't get executed.
unsafe {
let __t0: u8;
let __t1: u32;
let __t2: u32;
let __t3: u32;
core::arch::asm!("mov %esi, {2:e}\nmov {1:e}, %esi\ntest %ecx, %ecx\n\trepe cmpsb\nsetnz {0}\nmov %esi, {1:e}\nmov {2:e}, %esi", lateout(reg_byte) __t0, inout(reg) s2 as u32 => __t2, out(reg) _, inlateout("di") s1 as u32 => __t1, inlateout("cx") len as u32 => __t3, options(nostack, att_syntax));
diff = (__t0 as u8) != 0;
s1 = unsafe { core::mem::transmute::<u32, *const ()>(__t1 as u32) };
s2 = unsafe { core::mem::transmute::<u32, *const ()>(__t2 as u32) };
len = unsafe { core::mem::transmute::<u32, u32>(__t3 as u32) };
}
diff as i32
}

#[unsafe(export_name = "bcmp")]
unsafe extern "C" fn __slate_export_bcmp(s1: *const core::ffi::c_void, s2: *const core::ffi::c_void, len: usize) -> core::ffi::c_int {
return bcmp(s1 as *const (), s2 as *const (), len as u32) as core::ffi::c_int;
}

/// Clang may lower `memcmp == 0` to `bcmp == 0`.
fn bcmp(s1: *const (), s2: *const (), len: u32) -> i32 {
memcmp(s1, s2, len)
}

#[unsafe(no_mangle)]
extern "C" fn strcmp(str1: *const i8, str2: *const i8) -> i32 {
let mut s1: *const u8 = str1 as *const u8;
let mut s2: *const u8 = str2 as *const u8;
let mut delta: i32 = 0;
while (unsafe { *s1 }) != 0 || (unsafe { *s2 }) != 0 {
delta = ((unsafe { *s1 }) as i32) - ((unsafe { *s2 }) as i32);
if delta != 0 {
return delta;
}
let __v0: *const u8 = s1;
s1 = unsafe { __v0.add(1) };
let __v2: *const u8 = s2;
s2 = unsafe { __v2.add(1) };
}
0
}

#[unsafe(no_mangle)]
extern "C" fn strncmp(mut cs: *const i8, mut ct: *const i8, mut count: u32) -> i32 {
let mut c1: u8 = 0;
let mut c2: u8 = 0;
while count != 0 {
let __v0: *const i8 = cs;
cs = unsafe { __v0.add(1) };
c1 = (unsafe { *__v0 }) as u8;
let __v2: *const i8 = ct;
ct = unsafe { __v2.add(1) };
c2 = (unsafe { *__v2 }) as u8;
if (c1 as i32) != (c2 as i32) {
return if (c1 as i32) < (c2 as i32) { -1 } else { 1 };
}
if c1 == 0 {
break;
}
count = count.wrapping_sub(1);
}
0
}

#[unsafe(export_name = "strlen")]
unsafe extern "C" fn __slate_export_strlen(s: *const core::ffi::c_char) -> usize {
strlen(s as *const i8) as usize
}

#[unsafe(no_mangle)]
extern "C" fn strnlen(s: *const i8, mut maxlen: u32) -> u32 {
let mut es: *const i8 = s;
while (unsafe { *es }) != 0 && maxlen != 0 {
let __v0: *const i8 = es;
es = unsafe { __v0.add(1) };
maxlen = maxlen.wrapping_sub(1);
}
((unsafe { es.offset_from(s as *const i8) }) as i32) as u32
}

fn simple_guess_base(cp: *const i8) -> u32 {
if ((unsafe { *cp.add(0) }) as i32) == 48 {
let __v0: bool = TOLOWER!(unsafe { *cp.add(1) }) == 120 && isxdigit((unsafe { *cp.add(2) }) as i32) != 0;
if __v0 {
return 16;
} else {
return 8;
}
} else {
return 10;
}
}

/// simple_strtoull - convert a string to an unsigned long long
/// @cp: The start of the string
/// @endp: A pointer to the end of the parsed string will be placed here
/// @base: The number base to use
#[unsafe(no_mangle)]
extern "C" fn simple_strtoull(mut cp: *const i8, endp: *mut *mut i8, mut base: u32) -> u64 {
let mut result: u64 = 0;
if base == 0 {
base = simple_guess_base(cp);
}
if base == 16 && ((unsafe { *cp.add(0) }) as i32) == 48 && TOLOWER!(unsafe { *cp.add(1) }) == 120 {
let __v0: *const i8 = cp;
cp = unsafe { __v0.add(2) };
}
while isxdigit((unsafe { *cp }) as i32) != 0 {
let value: u32 = (if isdigit((unsafe { *cp }) as i32) != 0 { ((unsafe { *cp }) as i32) - 48 } else { TOLOWER!(unsafe { *cp }) - 97 + 10 }) as u32;
if value >= base {
break;
}
result = result.wrapping_mul(base as u64).wrapping_add(value as u64);
let __v3: *const i8 = cp;
cp = unsafe { __v3.add(1) };
}
if !endp.is_null() {
unsafe {
*endp = cp as *mut i8;
}
}
result
}

#[unsafe(no_mangle)]
extern "C" fn simple_strtol(cp: *const i8, endp: *mut *mut i8, base: u32) -> i32 {
if ((unsafe { *cp }) as i32) == 45 {
return simple_strtoull(unsafe { cp.add(1) }, endp, base).wrapping_neg() as i32;
}
simple_strtoull(cp, endp, base) as i32
}

/// strlen - Find the length of a string
/// @s: The string to be sized
fn strlen(s: *const i8) -> u32 {
let mut sc: *const i8 = unsafe { core::mem::zeroed() };
sc = s;
while ((unsafe { *sc }) as i32) != 0 {
let __v1: *const i8 = sc;
sc = unsafe { __v1.add(1) };
}
// nothing
((unsafe { sc.offset_from(s as *const i8) }) as i32) as u32
}

/// strstr - Find the first substring in a %NUL terminated string
/// @s1: The string to be searched
/// @s2: The string to search for
#[unsafe(no_mangle)]
extern "C" fn strstr(mut s1: *const i8, s2: *const i8) -> *mut i8 {
let mut l1: u32 = 0;
let l2: u32 = strlen(s2);
if l2 == 0 {
return s1 as *mut i8;
}
l1 = strlen(s1);
while l1 >= l2 {
l1 = l1.wrapping_sub(1);
if memcmp(s1 as *const (), s2 as *const (), l2) == 0 {
return s1 as *mut i8;
}
let __v2: *const i8 = s1;
s1 = unsafe { __v2.add(1) };
}
core::ptr::null_mut::<i8>()
}

/// strchr - Find the first occurrence of the character c in the string s.
/// @s: the string to be searched
/// @c: the character to search for
#[unsafe(no_mangle)]
extern "C" fn strchr(mut s: *const i8, c: i32) -> *mut i8 {
while ((unsafe { *s }) as i32) != ((c as i8) as i32) {
let __v0: *const i8 = s;
s = unsafe { __v0.add(1) };
if ((unsafe { *__v0 }) as i32) == 0 {
return core::ptr::null_mut::<i8>();
}
}
s as *mut i8
}

fn __div_u64_rem(dividend: u64, mut divisor: u32, remainder: &mut u32) -> u64 {
let mut d: __SlateRecord_string_c_205 = __SlateRecord_string_c_205 { v64: dividend };
let mut upper: u32 = unsafe { d.v32[1] };
unsafe {
d.v32[1] = 0;
}
if upper >= divisor {
unsafe {
d.v32[1] = upper / divisor;
}
upper %= divisor;
}
unsafe {
let __t0: u32;
let __t1: u32;
core::arch::asm!("divl {0:e}", in(reg) divisor as u32, inlateout("ax") d.v32[0] as u32 => __t0, inlateout("dx") upper as u32 => __t1, options(pure, nomem, nostack, att_syntax));
d.v32[0] = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
*(remainder as *mut u32) = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
unsafe { d.v64 }
}

fn __div_u64(dividend: u64, divisor: u32) -> u64 {
let mut remainder: u32 = 0;
__div_u64_rem(dividend, divisor, unsafe { &mut (*core::ptr::addr_of_mut!(remainder)) })
}

fn _tolower(c: i8) -> i8 {
((c as i32) | 32) as i8
}

fn _parse_integer_fixup_radix(mut s: *const i8, base: &mut u32) -> *const i8 {
if (unsafe { *(base as *mut u32) }) == 0 {
if ((unsafe { *s.add(0) }) as i32) == 48 {
let __v0: bool = (_tolower(unsafe { *s.add(1) }) as i32) == 120 && isxdigit((unsafe { *s.add(2) }) as i32) != 0;
if __v0 {
unsafe {
*(base as *mut u32) = 16;
}
} else {
unsafe {
*(base as *mut u32) = 8;
}
}
} else {
unsafe {
*(base as *mut u32) = 10;
}
}
}
let __v1: bool = (unsafe { *(base as *mut u32) }) == 16 && ((unsafe { *s.add(0) }) as i32) == 48 && (_tolower(unsafe { *s.add(1) }) as i32) == 120;
if __v1 {
let __v2: *const i8 = s;
s = unsafe { __v2.add(2) };
}
s
}

// Convert non-negative integer string representation in explicitly given radix
// to an integer.
// Return number of characters consumed maybe or-ed with overflow bit.
// If overflow occurs, result integer (incorrect) is still returned.
//
// Don't you dare use this function.
fn _parse_integer(mut s: *const i8, base: u32, p: &mut u64) -> u32 {
let mut res: u64 = 0;
let mut rv: u32 = 0;
while true {
let c: u32 = (unsafe { *s }) as u32;
let lc: u32 = c | 32; // don't tolower() this line
let mut val: u32 = 0;
if 48 <= c && c <= 57 {
val = c.wrapping_sub(48);
} else {
if 97 <= lc && lc <= 102 {
val = lc.wrapping_sub(97).wrapping_add(10);
} else {
break;
}
}
if val >= base {
break;
}
// Check for overflow only if we are within range of
// it in the max base we support (16)
if unlikely!(res & 17293822569102704640u64) != 0 {
if res > __div_u64(18446744073709551615u64.wrapping_sub(val as u64), base) {
rv |= 2147483648u32;
}
}
res = res.wrapping_mul(base as u64).wrapping_add(val as u64);
rv = rv.wrapping_add(1);
let __v5: *const i8 = s;
s = unsafe { __v5.add(1) };
}
unsafe {
*(p as *mut u64) = res;
}
rv
}

fn _kstrtoull(mut s: *const i8, mut base: u32, res: *mut u64) -> i32 {
let mut _res: u64 = 0;
let mut rv: u32 = 0;
if ((unsafe { *s.add(0) }) as i32) == 43 {
let __v0: *const i8 = s;
s = unsafe { __v0.add(1) };
}
s = _parse_integer_fixup_radix(s, unsafe { &mut (*core::ptr::addr_of_mut!(base)) });
rv = _parse_integer(s, base, unsafe { &mut (*core::ptr::addr_of_mut!(_res)) });
if rv & 2147483648u32 != 0 {
return -34;
}
if rv == 0 {
return -22;
}
let __v2: *const i8 = s;
s = unsafe { __v2.add(rv as usize) };
if ((unsafe { *s }) as i32) == 10 {
let __v4: *const i8 = s;
s = unsafe { __v4.add(1) };
}
if (unsafe { *s }) != 0 {
return -22;
}
unsafe {
*res = _res;
}
0
}

fn _kstrtoul(s: *const i8, base: u32, res: *mut u32) -> i32 {
let mut tmp: u64 = 0;
let rv: i32 = _kstrtoull(s, base, core::ptr::addr_of_mut!(tmp));
if rv < 0 {
return rv;
}
if tmp != ((tmp as u32) as u64) {
return -34;
}
unsafe {
*res = tmp as u32;
}
0
}

/// boot_kstrtoul - convert a string to an unsigned long
/// @s: The start of the string. The string must be null-terminated, and may also
///  include a single newline before its terminating null. The first character
///  may also be a plus sign, but not a minus sign.
/// @base: The number base to use. The maximum supported base is 16. If base is
///  given as 0, then the base of the string is automatically detected with the
///  conventional semantics - If it begins with 0x the number will be parsed as a
///  hexadecimal (case insensitive), if it otherwise begins with 0, it will be
///  parsed as an octal number. Otherwise it will be parsed as a decimal.
/// @res: Where to write the result of the conversion on success.
///
/// Returns 0 on success, -ERANGE on overflow and -EINVAL on parsing error.
/// Used as a replacement for the simple_strtoull.
#[unsafe(no_mangle)]
extern "C" fn boot_kstrtoul(s: *const i8, base: u32, res: *mut u32) -> i32 {
// We want to shortcut function call, but
// __builtin_types_compatible_p(unsigned long, unsigned long long) = 0.
if ((const { core::mem::size_of::<u32>() }) as u32) == ((const { core::mem::size_of::<u64>() }) as u32) && ((const { core::mem::align_of::<u32>() }) as u32) == 8 {
return _kstrtoull(s, base, res as *mut u64);
} else {
return _kstrtoul(s, base, res);
}
}
