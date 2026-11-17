
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0
//!
//! linux/lib/string.c
//!
//! Copyright (C) 1991, 1992  Linus Torvalds
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! static_assert - check integer constant expression at build time
//! @expr: expression to be checked
//!
//! static_assert() is a wrapper for the C11 _Static_assert, with a
//! little macro magic to make the message optional (defaulting to the
//! stringification of the tested expression).
//!
//! Contrary to BUILD_BUG_ON(), static_assert() can be used at global
//! scope, but requires the expression to be an integer constant
//! expression (i.e., it is not enough that __builtin_constant_p() is
//! true for expr).
//!
//! Also note that BUILD_BUG_ON() fails the build if the condition is
//! true, while static_assert() fails the build if the expression is
//! false.
//! check_add_overflow() - Calculate addition with overflow checking
//! @a: first addend
//! @b: second addend
//! @d: pointer to store sum
//!
//! Returns true on wrap-around, false otherwise.
//!
//! *@d holds the results of the attempted addition, regardless of whether
//! wrap-around occurred.
//! See linux/instrumentation.h
//! Begin/end of an instrumentation safe region
//! 64 bit
//! longword load with zeropad on fault
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! strlen("ratelimit") + 1
//! Argument list too long
//! Value too large for defined data type
//! This counts to 15. Any more, it will return 16th argument.
//! IS_ERR_VALUE - Detect an error pointer.
//! @x: The pointer to check.
//!
//! Like IS_ERR(), but does not generate a compiler warning if result is unused.
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! "rep_good" REP microcode works well
//! label
//! const_*() definitions provide good compile-time optimizations when
//! the passed arguments can be resolved at compile time.
//! Many architecture-specific non-atomic bitops contain inline asm code and due
//! to that the compiler can't optimize them to compile-time expressions or
//! constants. In contrary, generic_*() helpers are defined in pure C and
//! compilers optimize them just well.
//! Therefore, to make `unsigned long foo = 0; __set_bit(BAR, &foo)` effectively
//! equal to `unsigned long foo = BIT(BAR)`, pick the generic C alternative when
//! the arguments can be resolved at compile time. That expression itself is a
//! constant and doesn't bring any functional changes to the rest of cases.
//! The casts to `uintptr_t` are needed to mitigate `-Waddress` warnings when
//! passing a bitmap from .bss or .data (-> `!!addr` is always true).
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! Compile time versions of __arch_hweightN()
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
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
// Note: do not use this directly. Instead, use __alloc_size() since it is conditionally
// available and includes other attributes. For GCC < 9.1, __alloc_size__ gets undefined
// in compiler-gcc.h, due to misbehaviors.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-alloc_005fsize-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#alloc-size
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
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-format-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#format
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-malloc-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#malloc
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-noreturn-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#noreturn
// clang: https://clang.llvm.org/docs/AttributeReference.html#id1
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-packed-type-attribute
// clang: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-packed-variable-attribute
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-pure-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-section-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-section-variable-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#section-declspec-allocate
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-warn_005funused_005fresult-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#nodiscard-warn-unused-result
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
// This version is i.e. to prevent dead stores elimination on @ptr
// where gcc and llvm may behave differently when otherwise using
// normal barrier(): while gcc behavior gets along with a normal
// barrier(), llvm needs an explicit input variable to be assumed
// clobbered. The issue is as follows: while the inline asm might
// access any memory it wants, the compiler could have fit all of
// @ptr into memory registers instead, and since @ptr never escaped
// from that, it proved that the inline asm wasn't touching any of
// it. This version works well with both compilers, i.e. we're telling
// the compiler that the inline asm absolutely may see the contents
// of @ptr. See also: https://llvm.org/bugs/show_bug.cgi?id=15495
// Format: __UNIQUE_ID_<name>_<__COUNTER__>
// Force the compiler to emit 'sym' as a symbol, so that we can reference
// it from inline assembler. Necessary in case 'sym' could be inlined
// otherwise, or eliminated entirely due to lack of references that are
// visible to the compiler.
// kcsan_check_write - check regular write access for races
//
// @ptr: address of access
// @size: size of access
// kcsan_check_read_write - check regular read-write access for races
//
// @ptr: address of access
// @size: size of access
// check_sub_overflow() - Calculate subtraction with overflow checking
// @a: minuend; value to subtract from
// @b: subtrahend; value to subtract from @a
// @d: pointer to store difference
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted subtraction, regardless of whether
// wrap-around occurred.
// check_mul_overflow() - Calculate multiplication with overflow checking
// @a: first factor
// @b: second factor
// @d: pointer to store product
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted multiplication, regardless of whether
// wrap-around occurred.
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Because instrumentation_{begin,end}() can nest, objtool validation considers
// _begin() a +1 and _end() a -1 and computes a sum over the instructions.
// When the value is greater than 0, we consider instrumentation allowed.
//
// There is a problem with code like:
//
// noinstr void foo()
// {
// 	instrumentation_begin();
// 	...
// 	if (cond) {
// 		instrumentation_begin();
// 		...
// 		instrumentation_end();
// 	}
// 	bar();
// 	instrumentation_end();
// }
//
// If instrumentation_end() would be an empty label, like all the other
// annotations, the inner _end(), which is at the end of a conditional block,
// would land on the instruction after the block.
//
// If we then consider the sum of the !cond path, we'll see that the call to
// bar() is with a 0-value, even though, we meant it to happen with a positive
// value.
//
// To avoid this, have _end() be a NOP instruction, this ensures it will be
// part of the condition block and does not escape.
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// Concatenate two parameters, but allow them to be expanded beforehand.
// "popcnt" POPCNT instruction
// "erms" Enhanced REP MOVSB/STOSB instructions
// "la57" 5-level page tables
// new instruction
// feature + flags
// source len
// replacement len
// replacement
// alternative assembly primitive:
// Like alternative_input, but with a single output argument
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
// order_base_2 - calculate the (rounded up) base 2 order of the argument
// @n: parameter
//
// The first few values calculated by this routine:
//  ob2(0) = 0
//  ob2(1) = 0
//  ob2(2) = 1
//  ob2(3) = 2
//  ob2(4) = 2
//  ob2(5) = 3
//  ... and so on.
// __put_unaligned_t - write an unaligned value to memory.
// @type:	the type of the value to store.
// @val:	the value to store.
// @ptr:	the pointer to store to.
//
// Use memcpy to affect an unaligned type sized store avoiding undefined
// behavior from approaches like type punning that require -fno-strict-aliasing
// in order to be correct. The void* cast silences ubsan warnings.
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
macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

/// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
/// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
/// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
macro_rules! IS_ENABLED {
() => {
__is_defined!()
};
}

/// REPEAT_BYTE - repeat the value @x multiple times as an unsigned long value
/// @x: value to repeat
///
/// NOTE: @x is not checked for > 0xff; larger values produce odd results.
macro_rules! REPEAT_BYTE {
($x:expr) => {
72340172838076673u64.wrapping_mul(($x) as u64)
};
}

macro_rules! ____is_defined {
() => {
0i32
};
}

/// __ffs - find first set bit in word
/// @word: The word to search
///
/// Undefined if no bit exists, so code should check against 0 first.
macro_rules! __ffs {
($word:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($word) } { ($word.trailing_zeros() as i32) as u64 } else { variable__ffs($word) }
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

/// Getting something that works in C and CPP for an arg that may or may
/// not be defined is tricky.  Here, if we have "#define CONFIG_BOOGER 1"
/// we match on the placeholder define, insert the "0," for arg1 and generate
/// the triplet (0, 1, 0).  Then the last step cherry picks the 2nd arg (a one).
/// When CONFIG_BOOGER is not defined, we generate a (... 1, 0) pair, and when
/// the last step cherry picks the 2nd arg, we get a zero.
macro_rules! __is_defined {
() => {
____is_defined!()
};
}

macro_rules! __ismask {
($x:expr) => {
unsafe { *(core::ptr::addr_of!(_ctype) as *const u8).wrapping_offset(((($x) as u8) as i32) as isize) }
};
}

/// Keep the initial has_zero() value for both bitmask and size calc
macro_rules! create_zero_mask {
($bits:expr) => {
$bits
};
}

macro_rules! find_zero {
($bits:expr) => {
__ffs!($bits) >> 3i32
};
}

macro_rules! get_unaligned__2 {
($ptr:expr) => {
__get_unaligned_t__2!($ptr)
};
}

macro_rules! isupper {
($c:expr) => {
(__ismask!($c) as i32) & 1 != 0
};
}

macro_rules! tolower {
($c:expr) => {
__tolower($c)
};
}

macro_rules! unlikely__2 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely {
($x:expr) => {
!(!$x) as i64
};
}

unsafe extern "C" {
    static mut _ctype: [u8; 0];
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __builtin_unreachable();
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_f217c4720f04f6db(arg0: u64) -> bool;
}

fn __tolower(mut c: u8) -> u8 {
if isupper!(c) {
c = (c as i32).wrapping_sub(-32i32) as u8;
}
c
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.sized_strscpy")]
extern "C" fn sized_strscpy(dest: *mut u8, src: *const u8, mut count: u64) -> i64 {
let constants: word_at_a_time = word_at_a_time { one_bits: REPEAT_BYTE!(1i32), high_bits: REPEAT_BYTE!(128i32) };
let mut max: u64 = count;
let mut res: i64 = 0;
let __v0: bool = if count == 0 {
true
} else {
let mut __v1: i64;
let __ret_warn_on: i32 = (count > 2147483647u64) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("7: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 7b - ., 3; .popsection\n# {0}", const 7 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_sized_strscpy_2, sym __slate_asm_string_sized_strscpy_3, const 104 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("8: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 8b - ., 4; .popsection\n# {0}", const 8 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
__v1 != 0
};
if __v0 {
return 7i32.wrapping_neg() as i64;
}
// load_unaligned_zeropad() or read_word_at_a_time() below may read
// uninitialized bytes after the trailing zero and use them in
// comparisons. Disable this optimization under KMSAN to prevent
// false positive reports.
if IS_ENABLED!() != 0 {
max = 0;
}
while max >= (core::mem::size_of::<u64>() as u64) {
let mut c: u64 = load_unaligned_zeropad(src.wrapping_offset(res as isize) as *const ());
let mut data: u64 = 0;
if has_zero(c, unsafe { &mut (*core::ptr::addr_of_mut!(data)) }, unsafe { &(*core::ptr::addr_of!(constants)) }) != 0 {
data = prep_zero_mask(c, data, unsafe { &(*core::ptr::addr_of!(constants)) });
data = create_zero_mask!(data);
unsafe {
*(dest.wrapping_offset(res as isize) as *mut u64) = c & zero_bytemask(data);
}
return (res as u64).wrapping_add(find_zero!(data)) as i64;
}
count = count.wrapping_sub(core::mem::size_of::<u64>() as u64);
if unlikely!(count == 0) != 0 {
let __v9: u64 = c & 72057594037927935u64;
c = __v9;
unsafe {
*(dest.wrapping_offset(res as isize) as *mut u64) = c;
}
return 7i32.wrapping_neg() as i64;
}
unsafe {
*(dest.wrapping_offset(res as isize) as *mut u64) = c;
}
res = (res as u64).wrapping_add(core::mem::size_of::<u64>() as u64) as i64;
max = max.wrapping_sub(core::mem::size_of::<u64>() as u64);
}
while count > 1 {
let c: u8 = unsafe { *src.wrapping_offset(res as isize) };
unsafe {
*dest.wrapping_offset(res as isize) = c;
}
if c == 0 {
return res;
}
res = res.wrapping_add(1);
count = count.wrapping_sub(1);
}
// Force NUL-termination.
unsafe {
*dest.wrapping_offset(res as isize) = 0;
}
// Return E2BIG if the source didn't stop
if (unsafe { *src.wrapping_offset(res as isize) }) != 0 { 7i32.wrapping_neg() as i64 } else { res }
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strlcat")]
extern "C" fn strlcat(mut dest: *mut u8, src: *const u8, mut count: u64) -> u64 {
let dsize: u64 = strlen(dest as *const u8);
let mut len: u64 = strlen(src);
let res: u64 = dsize.wrapping_add(len);
// This would be a bug
loop {
if unlikely!(dsize >= count) != 0 {
loop {
unsafe {
core::arch::asm!("13: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 13b - ., 3; .popsection\n# {0}", const 13 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_strlcat_0, sym __slate_asm_string_strlcat_1, const 241 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { core::hint::unreachable_unchecked() };
if true {
break;
}
}
}
if true {
break;
}
}
let __v5: *mut u8 = dest;
dest = __v5.wrapping_offset(dsize as isize);
count = count.wrapping_sub(dsize);
if len >= count {
len = count.wrapping_sub(1);
}
unsafe { __builtin_memcpy(dest as *mut (), src as *const (), len) };
unsafe {
*dest.wrapping_offset(len as isize) = 0;
}
res
}

#[inline(always)]
fn variable__ffs(mut word: u64) -> u64 {
unsafe {
let __t0: u64;
core::arch::asm!("tzcnt {1:r},{0:r}", lateout(reg) __t0, in(reg) word as u64, options(pure, nomem, nostack, att_syntax));
word = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
word
}

#[repr(C)]
#[derive(Clone, Copy)]
struct word_at_a_time {
    one_bits: u64,
    high_bits: u64,
}


fn has_zero(a: u64, bits: &mut u64, c: &word_at_a_time) -> u64 {
let mask: u64 = a.wrapping_sub(unsafe { (*(c as *const word_at_a_time)).one_bits }) & !a & unsafe { (*(c as *const word_at_a_time)).high_bits };
unsafe {
*(bits as *mut u64) = mask;
}
mask
}

fn prep_zero_mask(a: u64, bits: u64, c: &word_at_a_time) -> u64 {
bits
}

fn zero_bytemask(mut bits: u64) -> u64 {
bits = bits.wrapping_sub(1) & !bits;
bits >> 7i32
}

fn load_unaligned_zeropad(mut addr: *const ()) -> u64 {
let mut ret: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("1:\tmov ({1:r}), {0:r}\n2:\n .pushsection __ex_table, \"aM\", @progbits, 12\n .balign 4\n .long (1b) - .\n .long (2b) - .\n .long 20 \n .popsection\n", lateout(reg) __t0, in(reg) addr as *mut u64, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
ret
}

// This file should be used only for "library" routines that may have
// alternative implementations on specific architectures (generally
// found in <asm-xx/string.h>), or get overloaded by FORTIFY_SOURCE.
// (Specifically, this file is built with __NO_FORTIFY.)
//
// Other helper functions should live in string_helpers.c.
/// strncasecmp - Case insensitive, length-limited string comparison
/// @s1: One string
/// @s2: The other string
/// @len: the maximum number of characters to compare
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strncasecmp")]
extern "C" fn strncasecmp(mut s1: *const u8, mut s2: *const u8, mut len: u64) -> i32 {
// Yes, Virginia, it had better be unsigned
let mut c1: u8 = 0;
let mut c2: u8 = 0;
if len == 0 {
return 0;
}
loop {
let __v0: *const u8 = s1;
s1 = __v0.wrapping_offset(1 as isize);
c1 = unsafe { *__v0 };
let __v2: *const u8 = s2;
s2 = __v2.wrapping_offset(1 as isize);
c2 = unsafe { *__v2 };
if c1 == 0 || c2 == 0 {
break;
}
if (c1 as i32) != (c2 as i32) {
c1 = tolower!(c1);
c2 = tolower!(c2);
if (c1 as i32) != (c2 as i32) {
break;
}
}
let __v6: u64 = len.wrapping_sub(1);
len = __v6;
if __v6 == 0 {
break;
}
}
(c1 as i32).wrapping_sub(c2 as i32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strncasecmp_4: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8, u64) -> i32>, *mut ()>(Some(strncasecmp)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strcasecmp")]
extern "C" fn strcasecmp(mut s1: *const u8, mut s2: *const u8) -> i32 {
let mut c1: i32 = 0;
let mut c2: i32 = 0;
loop {
let __v0: *const u8 = s1;
s1 = __v0.wrapping_offset(1 as isize);
c1 = __tolower(unsafe { *__v0 }) as i32;
let __v2: *const u8 = s2;
s2 = __v2.wrapping_offset(1 as isize);
c2 = __tolower(unsafe { *__v2 }) as i32;
if !(c1 == c2 && c1 != 0) {
break;
}
}
c1.wrapping_sub(c2)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strcasecmp_5: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8) -> i32>, *mut ()>(Some(strcasecmp)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strcpy")]
extern "C" fn strcpy(mut dest: *mut u8, mut src: *const u8) -> *mut u8 {
let tmp: *mut u8 = dest;
loop {
let __v0: *const u8 = src;
src = __v0.wrapping_offset(1 as isize);
let __v2: *mut u8 = dest;
dest = __v2.wrapping_offset(1 as isize);
let __v4: u8 = unsafe { *__v0 };
unsafe {
*__v2 = __v4;
}
if (__v4 as i32) == 0 {
break;
}
}
// nothing
tmp
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strcpy_6: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u8, *const u8) -> *mut u8>, *mut ()>(Some(strcpy)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_sized_strscpy_9: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u8, *const u8, u64) -> i64>, *mut ()>(Some(sized_strscpy)) };

/// stpcpy - copy a string from src to dest returning a pointer to the new end
///          of dest, including src's %NUL-terminator. May overrun dest.
/// @dest: pointer to end of string being copied into. Must be large enough
///        to receive copy.
/// @src: pointer to the beginning of string being copied from. Must not overlap
///       dest.
///
/// stpcpy differs from strcpy in a key way: the return value is a pointer
/// to the new %NUL-terminating character in @dest. (For strcpy, the return
/// value is a pointer to the start of @dest). This interface is considered
/// unsafe as it doesn't perform bounds checking of the inputs. As such it's
/// not recommended for usage. Instead, its definition is provided in case
/// the compiler lowers other libcalls to stpcpy.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.stpcpy")]
extern "C" fn stpcpy(mut dest: *mut u8, mut src: *const u8) -> *mut u8 {
loop {
let __v0: *const u8 = src;
src = __v0.wrapping_offset(1 as isize);
let __v2: *mut u8 = dest;
dest = __v2.wrapping_offset(1 as isize);
let __v4: u8 = unsafe { *__v0 };
unsafe {
*__v2 = __v4;
}
if (__v4 as i32) == 0 {
break;
}
}
// nothing
let __v6: *mut u8 = dest;
let __v7: *mut u8 = __v6.wrapping_offset(-(1 as isize));
dest = __v7;
__v7
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_stpcpy_10: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u8, *const u8) -> *mut u8>, *mut ()>(Some(stpcpy)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strcat")]
extern "C" fn strcat(mut dest: *mut u8, mut src: *const u8) -> *mut u8 {
let tmp: *mut u8 = dest;
while (unsafe { *dest }) != 0 {
let __v0: *mut u8 = dest;
dest = __v0.wrapping_offset(1 as isize);
}
loop {
let __v3: *const u8 = src;
src = __v3.wrapping_offset(1 as isize);
let __v5: *mut u8 = dest;
dest = __v5.wrapping_offset(1 as isize);
let __v7: u8 = unsafe { *__v3 };
unsafe {
*__v5 = __v7;
}
if (__v7 as i32) == 0 {
break;
}
}
tmp
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strcat_11: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u8, *const u8) -> *mut u8>, *mut ()>(Some(strcat)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strncat")]
extern "C" fn strncat(mut dest: *mut u8, mut src: *const u8, mut count: u64) -> *mut u8 {
let tmp: *mut u8 = dest;
if count != 0 {
while (unsafe { *dest }) != 0 {
let __v0: *mut u8 = dest;
dest = __v0.wrapping_offset(1 as isize);
}
loop {
let __v6: *const u8 = src;
src = __v6.wrapping_offset(1 as isize);
let __v8: *mut u8 = dest;
dest = __v8.wrapping_offset(1 as isize);
let __v10: u8 = unsafe { *__v6 };
unsafe {
*__v8 = __v10;
}
if (__v10 as i32) == 0 {
break;
}
let __v4: u64 = count.wrapping_sub(1);
count = __v4;
if __v4 == 0 {
unsafe {
*dest = 0;
}
break;
}
}
}
tmp
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strncat_12: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u8, *const u8, u64) -> *mut u8>, *mut ()>(Some(strncat)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strlcat_14: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u8, *const u8, u64) -> u64>, *mut ()>(Some(strlcat)) };

/// strcmp - Compare two strings
/// @cs: One string
/// @ct: Another string
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strcmp")]
extern "C" fn strcmp(mut cs: *const u8, mut ct: *const u8) -> i32 {
let mut c1: u8 = 0;
let mut c2: u8 = 0;
while true {
let __v0: *const u8 = cs;
cs = __v0.wrapping_offset(1 as isize);
c1 = unsafe { *__v0 };
let __v2: *const u8 = ct;
ct = __v2.wrapping_offset(1 as isize);
c2 = unsafe { *__v2 };
if (c1 as i32) != (c2 as i32) {
return if (c1 as i32) < (c2 as i32) { 1i32.wrapping_neg() } else { 1 };
}
if c1 == 0 {
break;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strcmp_15: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8) -> i32>, *mut ()>(Some(strcmp)) };

/// strncmp - Compare two length-limited strings
/// @cs: One string
/// @ct: Another string
/// @count: The maximum number of bytes to compare
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strncmp")]
extern "C" fn strncmp(mut cs: *const u8, mut ct: *const u8, mut count: u64) -> i32 {
let mut c1: u8 = 0;
let mut c2: u8 = 0;
while count != 0 {
let __v0: *const u8 = cs;
cs = __v0.wrapping_offset(1 as isize);
c1 = unsafe { *__v0 };
let __v2: *const u8 = ct;
ct = __v2.wrapping_offset(1 as isize);
c2 = unsafe { *__v2 };
if (c1 as i32) != (c2 as i32) {
return if (c1 as i32) < (c2 as i32) { 1i32.wrapping_neg() } else { 1 };
}
if c1 == 0 {
break;
}
count = count.wrapping_sub(1);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strncmp_16: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8, u64) -> i32>, *mut ()>(Some(strncmp)) };

/// strchr - Find the first occurrence of a character in a string
/// @s: The string to be searched
/// @c: The character to search for
///
/// Note that the %NUL-terminator is considered part of the string, and can
/// be searched for.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strchr")]
extern "C" fn strchr(mut s: *const u8, c: i32) -> *mut u8 {
while ((unsafe { *s }) as i32) != ((c as u8) as i32) {
if ((unsafe { *s }) as i32) == 0 {
return core::ptr::null_mut::<u8>();
}
let __v1: *const u8 = s;
s = __v1.wrapping_offset(1 as isize);
}
s as *mut u8
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_strncasecmp: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strcasecmp: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strcpy: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_sized_strscpy: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_stpcpy: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strcat: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strncat: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strlcat: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strcmp: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strncmp: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strchr: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strchrnul: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strrchr: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strnchr: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strlen: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad strlen ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strnlen: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strspn: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strcspn: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strpbrk: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strsep: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_memcmp: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad memcmp ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bcmp: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad bcmp ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_memscan: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strstr: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_strnstr: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_memchr: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_memchr_inv: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous", sym strncasecmp, sym strcasecmp, sym strcpy, sym sized_strscpy, sym stpcpy, sym strcat, sym strncat, sym strlcat, sym strcmp, sym strncmp, sym strchr, sym strchrnul, sym strrchr, sym strnchr, sym strnlen, sym strspn, sym strcspn, sym strpbrk, sym strsep, sym memscan, sym strstr, sym strnstr, sym memchr, sym memchr_inv, options(att_syntax));

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strchr_17: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, i32) -> *mut u8>, *mut ()>(Some(strchr)) };

#[unsafe(export_name = "memcmp")]
unsafe extern "C" fn __slate_export_memcmp(cs: *const core::ffi::c_void, ct: *const core::ffi::c_void, count: usize) -> core::ffi::c_int {
return memcmp(cs as *const (), ct as *const (), count as u64) as core::ffi::c_int;
}

/// strchrnul - Find and return a character in a string, or end of string
/// @s: The string to be searched
/// @c: The character to search for
///
/// Returns pointer to first occurrence of 'c' in s. If c is not found, then
/// return a pointer to the null byte at the end of s.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strchrnul")]
extern "C" fn strchrnul(mut s: *const u8, c: i32) -> *mut u8 {
while (unsafe { *s }) != 0 && ((unsafe { *s }) as i32) != ((c as u8) as i32) {
let __v0: *const u8 = s;
s = __v0.wrapping_offset(1 as isize);
}
s as *mut u8
}

#[unsafe(export_name = "strlen")]
unsafe extern "C" fn __slate_export_strlen(s: *const core::ffi::c_char) -> usize {
strlen(s as *const u8) as usize
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strchrnul_18: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, i32) -> *mut u8>, *mut ()>(Some(strchrnul)) };

/// strnchrnul - Find and return a character in a length limited string,
/// or end of string
/// @s: The string to be searched
/// @count: The number of characters to be searched
/// @c: The character to search for
///
/// Returns pointer to the first occurrence of 'c' in s. If c is not found,
/// then return a pointer to the last character of the string.
#[unsafe(no_mangle)]
extern "C" fn strnchrnul(mut s: *const u8, mut count: u64, c: i32) -> *mut u8 {
loop {
let __v2: u64 = count;
count = __v2.wrapping_sub(1);
if !(__v2 != 0 && (unsafe { *s }) != 0 && ((unsafe { *s }) as i32) != ((c as u8) as i32)) {
break;
}
let __v0: *const u8 = s;
s = __v0.wrapping_offset(1 as isize);
}
s as *mut u8
}

/// strrchr - Find the last occurrence of a character in a string
/// @s: The string to be searched
/// @c: The character to search for
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strrchr")]
extern "C" fn strrchr(mut s: *const u8, c: i32) -> *mut u8 {
let mut last: *const u8 = core::ptr::null::<u8>();
loop {
if ((unsafe { *s }) as i32) == ((c as u8) as i32) {
last = s;
}
let __v1: *const u8 = s;
s = __v1.wrapping_offset(1 as isize);
if (unsafe { *__v1 }) == 0 {
break;
}
}
last as *mut u8
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strrchr_19: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, i32) -> *mut u8>, *mut ()>(Some(strrchr)) };

/// strnchr - Find a character in a length limited string
/// @s: The string to be searched
/// @count: The number of characters to be searched
/// @c: The character to search for
///
/// Note that the %NUL-terminator is considered part of the string, and can
/// be searched for.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strnchr")]
extern "C" fn strnchr(mut s: *const u8, mut count: u64, c: i32) -> *mut u8 {
loop {
let __v3: u64 = count;
count = __v3.wrapping_sub(1);
if __v3 == 0 {
break;
}
if ((unsafe { *s }) as i32) == ((c as u8) as i32) {
return s as *mut u8;
}
let __v0: *const u8 = s;
s = __v0.wrapping_offset(1 as isize);
if ((unsafe { *__v0 }) as i32) == 0 {
break;
}
}
core::ptr::null_mut::<u8>()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strnchr_20: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u64, i32) -> *mut u8>, *mut ()>(Some(strnchr)) };

#[unsafe(link_section = ".text.slate_distinct.string.strlen")]
extern "C" fn strlen(s: *const u8) -> u64 {
let mut sc: *const u8 = unsafe { core::mem::zeroed() };
sc = s;
while ((unsafe { *sc }) as i32) != 0 {
let __v1: *const u8 = sc;
sc = __v1.wrapping_offset(1 as isize);
}
// nothing
((unsafe { sc.offset_from(s as *const u8) }) as i64) as u64
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strlen_21: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> u64>, *mut ()>(Some(strlen)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strnlen")]
extern "C" fn strnlen(s: *const u8, mut count: u64) -> u64 {
let mut sc: *const u8 = unsafe { core::mem::zeroed() };
sc = s;
loop {
let __v0: u64 = count;
count = __v0.wrapping_sub(1);
if !(__v0 != 0 && ((unsafe { *sc }) as i32) != 0) {
break;
}
let __v3: *const u8 = sc;
sc = __v3.wrapping_offset(1 as isize);
}
// nothing
((unsafe { sc.offset_from(s as *const u8) }) as i64) as u64
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strnlen_22: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, u64) -> u64>, *mut ()>(Some(strnlen)) };

/// strspn - Calculate the length of the initial substring of @s which only contain letters in @accept
/// @s: The string to be searched
/// @accept: The string to search for
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strspn")]
extern "C" fn strspn(s: *const u8, accept: *const u8) -> u64 {
let mut p: *const u8 = unsafe { core::mem::zeroed() };
p = s;
while ((unsafe { *p }) as i32) != 0 {
if strchr(accept, (unsafe { *p }) as i32).is_null() {
break;
}
let __v1: *const u8 = p;
p = __v1.wrapping_offset(1 as isize);
}
((unsafe { p.offset_from(s as *const u8) }) as i64) as u64
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strspn_23: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8) -> u64>, *mut ()>(Some(strspn)) };

/// strcspn - Calculate the length of the initial substring of @s which does not contain letters in @reject
/// @s: The string to be searched
/// @reject: The string to avoid
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strcspn")]
extern "C" fn strcspn(s: *const u8, reject: *const u8) -> u64 {
let mut p: *const u8 = unsafe { core::mem::zeroed() };
p = s;
while ((unsafe { *p }) as i32) != 0 {
if !strchr(reject, (unsafe { *p }) as i32).is_null() {
break;
}
let __v1: *const u8 = p;
p = __v1.wrapping_offset(1 as isize);
}
((unsafe { p.offset_from(s as *const u8) }) as i64) as u64
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strcspn_24: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8) -> u64>, *mut ()>(Some(strcspn)) };

/// strpbrk - Find the first occurrence of a set of characters
/// @cs: The string to be searched
/// @ct: The characters to search for
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strpbrk")]
extern "C" fn strpbrk(cs: *const u8, ct: *const u8) -> *mut u8 {
let mut sc: *const u8 = unsafe { core::mem::zeroed() };
sc = cs;
while ((unsafe { *sc }) as i32) != 0 {
if !strchr(ct, (unsafe { *sc }) as i32).is_null() {
return sc as *mut u8;
}
let __v1: *const u8 = sc;
sc = __v1.wrapping_offset(1 as isize);
}
core::ptr::null_mut::<u8>()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strpbrk_25: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8) -> *mut u8>, *mut ()>(Some(strpbrk)) };

/// strsep - Split a string into tokens
/// @s: The string to be searched
/// @ct: The characters to search for
///
/// strsep() updates @s to point after the token, ready for the next call.
///
/// It returns empty tokens, too, behaving exactly like the libc function
/// of that name. In fact, it was stolen from glibc2 and de-fancy-fied.
/// Same semantics, slimmer shape. ;)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strsep")]
extern "C" fn strsep(s: *mut *mut u8, ct: *const u8) -> *mut u8 {
let sbegin: *mut u8 = unsafe { *s };
let mut end: *mut u8 = unsafe { core::mem::zeroed() };
if sbegin.is_null() {
return core::ptr::null_mut::<u8>();
}
end = strpbrk(sbegin as *const u8, ct);
if !end.is_null() {
let __v0: *mut u8 = end;
end = __v0.wrapping_offset(1 as isize);
unsafe {
*__v0 = 0;
}
}
unsafe {
*s = end;
}
sbegin
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strsep_26: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut *mut u8, *const u8) -> *mut u8>, *mut ()>(Some(strsep)) };

// memcmp - Compare two areas of memory
// @cs: One area of memory
// @ct: Another area of memory
// @count: The size of the area.
#[unsafe(link_section = ".text.slate_distinct.string.memcmp")]
extern "C" fn memcmp(mut cs: *const (), mut ct: *const (), mut count: u64) -> i32 {
let mut su1: *const u8 = unsafe { core::mem::zeroed() };
let mut su2: *const u8 = unsafe { core::mem::zeroed() };
let mut res: i32 = 0;
if count >= (core::mem::size_of::<u64>() as u64) {
let mut u1: *const u64 = cs as *const u64;
let mut u2: *const u64 = ct as *const u64;
while get_unaligned__2!(u1) == get_unaligned__2!(u2) {
let __v7: *const u64 = u1;
u1 = __v7.wrapping_offset(1 as isize);
let __v9: *const u64 = u2;
u2 = __v9.wrapping_offset(1 as isize);
count = count.wrapping_sub(core::mem::size_of::<u64>() as u64);
if count < (core::mem::size_of::<u64>() as u64) {
break;
}
}
cs = u1 as *const ();
ct = u2 as *const ();
}
su1 = cs as *const u8;
su2 = ct as *const u8;
while 0 < count {
let __v14: i32 = ((unsafe { *su1 }) as i32).wrapping_sub((unsafe { *su2 }) as i32);
res = __v14;
if __v14 != 0 {
break;
}
let __v16: *const u8 = su1;
su1 = __v16.wrapping_offset(1 as isize);
let __v18: *const u8 = su2;
su2 = __v18.wrapping_offset(1 as isize);
count = count.wrapping_sub(1);
}
res
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_memcmp_27: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const (), *const (), u64) -> i32>, *mut ()>(Some(memcmp)) };

/// bcmp - returns 0 if and only if the buffers have identical contents.
/// @a: pointer to first buffer.
/// @b: pointer to second buffer.
/// @len: size of buffers.
///
/// The sign or magnitude of a non-zero return value has no particular
/// meaning, and architectures may implement their own more efficient bcmp(). So
/// while this particular implementation is a simple (tail) call to memcmp, do
/// not rely on anything but whether the return value is zero or non-zero.
#[unsafe(link_section = ".text.slate_distinct.string.bcmp")]
extern "C" fn bcmp(a: *const (), b: *const (), len: u64) -> i32 {
memcmp(a, b, len)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bcmp_28: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const (), *const (), u64) -> i32>, *mut ()>(Some(bcmp)) };

/// memscan - Find a character in an area of memory.
/// @addr: The memory area
/// @c: The byte to search for
/// @size: The size of the area.
///
/// returns the address of the first occurrence of @c, or 1 byte past
/// the area if @c is not found
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.memscan")]
extern "C" fn memscan(addr: *mut (), c: i32, mut size: u64) -> *mut () {
let mut p: *mut u8 = addr as *mut u8;
while size != 0 {
if ((unsafe { *p }) as i32) == ((c as u8) as i32) {
return p as *mut ();
}
let __v0: *mut u8 = p;
p = __v0.wrapping_offset(1 as isize);
size = size.wrapping_sub(1);
}
p as *mut ()
}

#[unsafe(export_name = "bcmp")]
unsafe extern "C" fn __slate_export_bcmp(a: *const core::ffi::c_void, b: *const core::ffi::c_void, len: usize) -> core::ffi::c_int {
return bcmp(a as *const (), b as *const (), len as u64) as core::ffi::c_int;
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_memscan_29: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32, u64) -> *mut ()>, *mut ()>(Some(memscan)) };

/// strstr - Find the first substring in a %NUL terminated string
/// @s1: The string to be searched
/// @s2: The string to search for
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strstr")]
extern "C" fn strstr(mut s1: *const u8, s2: *const u8) -> *mut u8 {
let mut l1: u64 = 0;
let l2: u64 = strlen(s2);
if l2 == 0 {
return s1 as *mut u8;
}
l1 = strlen(s1);
while l1 >= l2 {
l1 = l1.wrapping_sub(1);
if memcmp(s1 as *const (), s2 as *const (), l2) == 0 {
return s1 as *mut u8;
}
let __v2: *const u8 = s1;
s1 = __v2.wrapping_offset(1 as isize);
}
core::ptr::null_mut::<u8>()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strstr_30: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8) -> *mut u8>, *mut ()>(Some(strstr)) };

/// strnstr - Find the first substring in a length-limited string
/// @s1: The string to be searched
/// @s2: The string to search for
/// @len: the maximum number of characters to search
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.strnstr")]
extern "C" fn strnstr(mut s1: *const u8, s2: *const u8, mut len: u64) -> *mut u8 {
let l2: u64 = strlen(s2);
if l2 == 0 {
return s1 as *mut u8;
}
while len >= l2 {
len = len.wrapping_sub(1);
if memcmp(s1 as *const (), s2 as *const (), l2) == 0 {
return s1 as *mut u8;
}
let __v2: *const u8 = s1;
s1 = __v2.wrapping_offset(1 as isize);
}
core::ptr::null_mut::<u8>()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_strnstr_31: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8, u64) -> *mut u8>, *mut ()>(Some(strnstr)) };

/// memchr - Find a character in an area of memory.
/// @s: The memory area
/// @c: The byte to search for
/// @n: The size of the area.
///
/// returns the address of the first occurrence of @c, or %NULL
/// if @c is not found
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.memchr")]
extern "C" fn memchr(s: *const (), c: i32, mut n: u64) -> *mut () {
let mut p: *const u8 = s as *const u8;
loop {
let __v2: u64 = n;
n = __v2.wrapping_sub(1);
if __v2 == 0 {
break;
}
let __v0: *const u8 = p;
p = __v0.wrapping_offset(1 as isize);
if ((c as u8) as i32) == ((unsafe { *__v0 }) as i32) {
return p.wrapping_offset(-(1 as isize)) as *mut ();
}
}
core::ptr::null_mut::<()>()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_memchr_32: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const (), i32, u64) -> *mut ()>, *mut ()>(Some(memchr)) };

fn check_bytes8(mut start: *const u8, value: u8, mut bytes: u32) -> *mut () {
while bytes != 0 {
if ((unsafe { *start }) as i32) != (value as i32) {
return start as *mut ();
}
let __v0: *const u8 = start;
start = __v0.wrapping_offset(1 as isize);
bytes = bytes.wrapping_sub(1);
}
core::ptr::null_mut::<()>()
}

/// memchr_inv - Find an unmatching character in an area of memory.
/// @start: The memory area
/// @c: Find a character other than c
/// @bytes: The size of the area.
///
/// returns the address of the first character other than @c, or %NULL
/// if the whole buffer contains just @c.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.string.memchr_inv")]
extern "C" fn memchr_inv(mut start: *const (), c: i32, mut bytes: u64) -> *mut () {
let value: u8 = c as u8;
let mut value64: u64 = 0;
let mut words: u64 = 0;
let mut prefix: u32 = 0;
if bytes <= 16 {
return check_bytes8(start as *const u8, value, bytes as u32);
}
value64 = value as u64;
value64 = value64.wrapping_mul(72340172838076673);
prefix = ((start as u64) % 8) as u32;
if prefix != 0 {
prefix = 8u32.wrapping_sub(prefix);
let r: *mut u8 = check_bytes8(start as *const u8, value, prefix) as *mut u8;
if !r.is_null() {
return r as *mut ();
}
start = (unsafe { (start as *const u8).wrapping_offset(prefix as isize) }) as *const ();
bytes = bytes.wrapping_sub(prefix as u64);
}
words = bytes / 8;
while words != 0 {
if (unsafe { *(start as *mut u64) }) != value64 {
return check_bytes8(start as *const u8, value, 8u32);
}
start = (unsafe { (start as *const u8).wrapping_offset(8 as isize) }) as *const ();
words = words.wrapping_sub(1);
}
check_bytes8(start as *const u8, value, (bytes % 8) as u32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_memchr_inv_33: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const (), i32, u64) -> *mut ()>, *mut ()>(Some(memchr_inv)) };

static __slate_asm_string_sized_strscpy_2: [u8; 1] = *b"\0";

static __slate_asm_string_sized_strscpy_3: [u8; 42] = *b"/home/takashi/c-corpus/linux/lib/string.c\0";

static __slate_asm_string_strlcat_0: [u8; 1] = *b"\0";

static __slate_asm_string_strlcat_1: [u8; 42] = *b"/home/takashi/c-corpus/linux/lib/string.c\0";
