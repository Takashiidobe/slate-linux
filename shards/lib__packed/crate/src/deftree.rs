
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! +++ trees.c
//!
//! trees.c -- output deflated data using Huffman coding
//! Copyright (C) 1995-1996 Jean-loup Gailly
//! For conditions of distribution and use, see copyright notice in zlib.h
//! This counts to 15. Any more, it will return 16th argument.
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
//! Value too large for defined data type
//! IS_ERR_VALUE - Detect an error pointer.
//! @x: The pointer to check.
//!
//! Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
//! 64 bit
//! BUILD_BUG_ON_MSG - break compile if a condition is true & emit supplied
//! 		      error message.
//! @cond: the condition which the compiler should know is false.
//! @msg: build-time error message
//!
//! See BUILD_BUG_ON for description.
//! Begin/end of an instrumentation safe region
//! __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
//! the symbol table so that objtool can reference it when it generates the
//! .static_call_sites section.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! strlen("ratelimit") + 1
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
//! "popcnt" POPCNT instruction
//! Compile time versions of __arch_hweightN()
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
//! largest prime smaller than 65536
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
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// static_assert - check integer constant expression at build time
// @expr: expression to be checked
//
// static_assert() is a wrapper for the C11 _Static_assert, with a
// little macro magic to make the message optional (defaulting to the
// stringification of the tested expression).
//
// Contrary to BUILD_BUG_ON(), static_assert() can be used at global
// scope, but requires the expression to be an integer constant
// expression (i.e., it is not enough that __builtin_constant_p() is
// true for expr).
//
// Also note that BUILD_BUG_ON() fails the build if the condition is
// true, while static_assert() fails the build if the expression is
// false.
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
// new instruction
// feature + flags
// source len
// replacement len
// replacement
// alternative assembly primitive:
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
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
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// Optional: only supported since gcc >= 15, clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
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
macro_rules! __constant_bitrev32 {
($x:expr) => {
{
let mut __v: u32;
let mut __v2: u32 = $x;
__v2 = __v2 >> 16i32 | __v2 << 16i32;
__v2 = (__v2 & 4278255360) >> 8i32 | (__v2 & 16711935) << 8i32;
__v2 = (__v2 & 4042322160) >> 4i32 | (__v2 & 252645135) << 4i32;
__v2 = (__v2 & 3435973836) >> 2i32 | (__v2 & 858993459) << 2i32;
__v2 = (__v2 & 2863311530) >> 1i32 | (__v2 & 1431655765) << 1i32;
__v = __v2;
__v
}
};
}

macro_rules! __is_nonneg {
($ux:expr) => {
statically_true!((($ux) as i64) >= 0)
};
}

/// __sign_use for integer expressions:
///   bit #0 set if ok for unsigned comparisons
///   bit #1 set if ok for signed comparisons
///
/// In particular, statically non-negative signed integer expressions
/// are ok for both.
///
/// NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
/// in expressions, and are accepted for signed conversions for now.
/// This is debatable.
///
/// Note that 'x' is the original expression, and 'ux' is the unique variable
/// that contains the value.
///
/// We use 'ux' for pure type checking, and 'x' for when we need to look at the
/// value (but without evaluating it for side effects!
/// Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
///
/// Pointers end up being checked by the normal C type rules at the actual
/// comparison, and these expressions only need to be careful to not cause
/// warnings for pointer use.
macro_rules! __sign_use__3 {
($ux:expr) => {
if is_signed_type__3!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u8>() }) as u64) < 4) as i32)) }
};
}

macro_rules! __types_ok__3 {
($ux:expr, $uy:expr) => {
__sign_use__3!($ux) & __sign_use__3!($uy)
};
}

macro_rules! bitrev32 {
($x:expr) => {
{
let mut __v: u32;
let __v2: u32 = $x;
let __v3: u32 = if unsafe { __slate_intrinsic_1c5364fcd9da51fe(__v2) } {
__constant_bitrev32!(__v2)
} else {
generic___bitrev32(__v2)
};
__v = __v3;
__v
}
};
}

macro_rules! d_code {
($dist:expr) => {
if ($dist) < 256 { (unsafe { dist_code.0[($dist) as usize] }) as i32 } else { (unsafe { dist_code.0[256u32.wrapping_add(($dist) >> 7i32) as usize] }) as i32 }
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__3 {
() => {
((1i32.wrapping_neg() as u8) as i32) < 1
};
}

/// Remove the smallest element from the heap and recreate the heap with
/// one less element. Updates heap and heap_len.
macro_rules! pqremove {
($s:expr, $tree:expr, $top:expr) => {
$top = unsafe { *((*$s).heap.as_mut_ptr() as *mut i32).wrapping_offset(1 as isize) };
let __v: *mut deflate_state = $s;
let __v2: i32 = unsafe { (*__v).heap_len };
unsafe {
(*__v).heap_len = __v2.wrapping_sub(1);
}
unsafe {
*((*$s).heap.as_mut_ptr() as *mut i32).wrapping_offset(1 as isize) = *((*$s).heap.as_mut_ptr() as *mut i32).wrapping_offset(__v2 as isize);
}
pqdownheap($s, $tree, 1i32);
};
}

/// Output a byte on the stream.
/// IN assertion: there is enough room in pending_buf.
macro_rules! put_byte {
($s:expr, $c:expr) => {
let __v: *mut deflate_state = $s;
let __v2: i32 = unsafe { (*__v).pending };
unsafe {
(*__v).pending = __v2.wrapping_add(1);
}
unsafe {
*unsafe { (*$s).pending_buf }.wrapping_offset(__v2 as isize) = $c;
}
};
}

/// Output a short LSB first on the stream.
/// IN assertion: there is enough room in pendingBuf.
macro_rules! put_short {
($s:expr, $w:expr) => {
put_byte!($s, ((($w) as i32) & 255) as u8);
put_byte!($s, (((($w) as u16) as i32) >> 8i32) as u8);
};
}

macro_rules! send_bits__2 {
($s:expr, $value:expr, $length:expr) => {
let len: i32 = ($length) as i32;
if (unsafe { (*$s).bi_valid }) > (16u64.wrapping_mul(core::mem::size_of::<u8>() as u64) as i32).wrapping_sub(len) {
let val: i32 = ($value) as i32;
let __v: *mut deflate_state = $s;
let __v2: u16 = unsafe { (*__v).bi_buf };
let __v3: u16 = ((__v2 as i32) | val << unsafe { (*$s).bi_valid }) as u16;
unsafe {
(*__v).bi_buf = __v3;
}
put_short!($s, unsafe { (*$s).bi_buf });
unsafe {
(*$s).bi_buf = (((val as u16) as i32) >> 16u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub((*$s).bi_valid as u64)) as u16;
}
let __v4: *mut deflate_state = $s;
unsafe {
(*__v4).bi_valid = ((*__v4).bi_valid as u64).wrapping_add((len as u64).wrapping_sub(16u64.wrapping_mul(core::mem::size_of::<u8>() as u64))) as i32;
}
} else {
let __v7: *mut deflate_state = $s;
let __v8: u16 = unsafe { (*__v7).bi_buf };
let __v9: u16 = ((__v8 as i32) | (($value) as i32) << unsafe { (*$s).bi_valid }) as u16;
unsafe {
(*__v7).bi_buf = __v9;
}
let __v10: *mut deflate_state = $s;
let __v11: i32 = unsafe { (*__v10).bi_valid };
unsafe {
(*__v10).bi_valid = __v11.wrapping_add(len);
}
}
};
}

macro_rules! send_bits__3 {
($s:expr, $value:expr, $length:expr) => {
let len: i32 = $length;
if (unsafe { (*$s).bi_valid }) > (16u64.wrapping_mul(core::mem::size_of::<u8>() as u64) as i32).wrapping_sub(len) {
let val: i32 = ($value) as i32;
let __v: *mut deflate_state = $s;
let __v2: u16 = unsafe { (*__v).bi_buf };
let __v3: u16 = ((__v2 as i32) | val << unsafe { (*$s).bi_valid }) as u16;
unsafe {
(*__v).bi_buf = __v3;
}
put_short!($s, unsafe { (*$s).bi_buf });
unsafe {
(*$s).bi_buf = (((val as u16) as i32) >> 16u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub((*$s).bi_valid as u64)) as u16;
}
let __v4: *mut deflate_state = $s;
unsafe {
(*__v4).bi_valid = ((*__v4).bi_valid as u64).wrapping_add((len as u64).wrapping_sub(16u64.wrapping_mul(core::mem::size_of::<u8>() as u64))) as i32;
}
} else {
let __v7: *mut deflate_state = $s;
let __v8: u16 = unsafe { (*__v7).bi_buf };
let __v9: u16 = (((__v8 as i32) as u32) | ($value) << unsafe { (*$s).bi_valid }) as u16;
unsafe {
(*__v7).bi_buf = __v9;
}
let __v10: *mut deflate_state = $s;
let __v11: i32 = unsafe { (*__v10).bi_valid };
unsafe {
(*__v10).bi_valid = __v11.wrapping_add(len);
}
}
};
}

macro_rules! send_bits {
($s:expr, $value:expr, $length:expr) => {
let len: i32 = $length;
if (unsafe { (*$s).bi_valid }) > (16u64.wrapping_mul(core::mem::size_of::<u8>() as u64) as i32).wrapping_sub(len) {
let val: i32 = $value;
let __v: *mut deflate_state = $s;
let __v2: u16 = unsafe { (*__v).bi_buf };
let __v3: u16 = ((__v2 as i32) | val << unsafe { (*$s).bi_valid }) as u16;
unsafe {
(*__v).bi_buf = __v3;
}
put_short!($s, unsafe { (*$s).bi_buf });
unsafe {
(*$s).bi_buf = (((val as u16) as i32) >> 16u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub((*$s).bi_valid as u64)) as u16;
}
let __v4: *mut deflate_state = $s;
unsafe {
(*__v4).bi_valid = ((*__v4).bi_valid as u64).wrapping_add((len as u64).wrapping_sub(16u64.wrapping_mul(core::mem::size_of::<u8>() as u64))) as i32;
}
} else {
let __v7: *mut deflate_state = $s;
let __v8: u16 = unsafe { (*__v7).bi_buf };
let __v9: u16 = ((__v8 as i32) | ($value) << unsafe { (*$s).bi_valid }) as u16;
unsafe {
(*__v7).bi_buf = __v9;
}
let __v10: *mut deflate_state = $s;
let __v11: i32 = unsafe { (*__v10).bi_valid };
unsafe {
(*__v10).bi_valid = __v11.wrapping_add(len);
}
}
};
}

macro_rules! send_bits__4 {
($s:expr, $value:expr, $length:expr) => {
let len: i32 = $length;
if (unsafe { (*$s).bi_valid }) > (16u64.wrapping_mul(core::mem::size_of::<u8>() as u64) as i32).wrapping_sub(len) {
let val: i32 = ($value) as i32;
let __v: *mut deflate_state = $s;
let __v2: u16 = unsafe { (*__v).bi_buf };
let __v3: u16 = ((__v2 as i32) | val << unsafe { (*$s).bi_valid }) as u16;
unsafe {
(*__v).bi_buf = __v3;
}
put_short!($s, unsafe { (*$s).bi_buf });
unsafe {
(*$s).bi_buf = (((val as u16) as i32) >> 16u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub((*$s).bi_valid as u64)) as u16;
}
let __v4: *mut deflate_state = $s;
unsafe {
(*__v4).bi_valid = ((*__v4).bi_valid as u64).wrapping_add((len as u64).wrapping_sub(16u64.wrapping_mul(core::mem::size_of::<u8>() as u64))) as i32;
}
} else {
let __v7: *mut deflate_state = $s;
let __v8: u16 = unsafe { (*__v7).bi_buf };
let __v9: u16 = ((__v8 as i32) | (($value) as i32) << unsafe { (*$s).bi_valid }) as u16;
unsafe {
(*__v7).bi_buf = __v9;
}
let __v10: *mut deflate_state = $s;
let __v11: i32 = unsafe { (*__v10).bi_valid };
unsafe {
(*__v10).bi_valid = __v11.wrapping_add(len);
}
}
};
}

macro_rules! send_code__2 {
($s:expr, $c:expr, $tree:expr) => {
send_bits__2!($s, unsafe { (*$tree.wrapping_offset(($c) as isize)).fc.code }, unsafe { (*$tree.wrapping_offset(($c) as isize)).dl.len });
};
}

macro_rules! send_code {
($s:expr, $c:expr, $tree:expr) => {
send_bits__2!($s, unsafe { (*($tree.as_mut_ptr() as *mut ct_data_s).wrapping_offset(($c) as isize)).fc.code }, unsafe { (*($tree.as_mut_ptr() as *mut ct_data_s).wrapping_offset(($c) as isize)).dl.len });
};
}

macro_rules! send_code__3 {
($s:expr, $c:expr, $tree:expr) => {
send_bits__2!($s, unsafe { $tree[($c) as usize].fc.code }, unsafe { $tree[($c) as usize].dl.len });
};
}

/// Compares to subtrees, using the tree depth as tie breaker when
/// the subtrees have equal frequency. This minimizes the worst case length.
macro_rules! smaller {
($tree:expr, $n:expr, $m:expr, $depth:expr) => {
((unsafe { (*$tree.wrapping_offset(($n) as isize)).fc.freq }) as i32) < ((unsafe { (*$tree.wrapping_offset(($m) as isize)).fc.freq }) as i32) || ((unsafe { (*$tree.wrapping_offset(($n) as isize)).fc.freq }) as i32) == ((unsafe { (*$tree.wrapping_offset(($m) as isize)).fc.freq }) as i32) && ((unsafe { *($depth.as_mut_ptr() as *mut u8).wrapping_offset(($n) as isize) }) as i32) <= ((unsafe { *($depth.as_mut_ptr() as *mut u8).wrapping_offset(($m) as isize) }) as i32)
};
}

/// Useful shorthand for "is this condition known at compile-time?"
///
/// Note that the condition may involve non-constant values,
/// but the compiler may know enough about the details of the
/// values to determine that the condition is statically true.
macro_rules! statically_true {
($x:expr) => {
(unsafe { __slate_intrinsic_1cfe0293d0a2af8d($x) }) && $x
};
}

unsafe extern "C" {
    static mut byte_rev_table: [u8; 256];
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn __compiletime_assert_7();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct internal_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct z_stream_s {
    next_in: *const u8,
    avail_in: u64,
    total_in: u64,
    next_out: *mut u8,
    avail_out: u64,
    total_out: u64,
    msg: *mut u8,
    state: *mut internal_state,
    workspace: *mut (),
    data_type: i32,
    adler: u64,
    reserved: u64,
}


#[inline(always)]
fn generic___bitrev8(byte: u8) -> u8 {
unsafe { byte_rev_table[(byte as i32) as usize] }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_defutil_h_51 {
    freq: u16,
    code: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_defutil_h_51_2 {
    dad: u16,
    len: u16,
}


#[inline(always)]
fn generic___bitrev16(x: u16) -> u16 {
((generic___bitrev8(((x as i32) & 255) as u8) as i32) << 8i32 | (generic___bitrev8(((x as i32) >> 8i32) as u8) as i32)) as u16
}

#[inline(always)]
fn generic___bitrev32(x: u32) -> u32 {
((generic___bitrev16((x & 65535) as u16) as i32) << 16i32 | (generic___bitrev16((x >> 16i32) as u16) as i32)) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ct_data_s {
    fc: __SlateRecord_defutil_h_51,
    dl: __SlateRecord_defutil_h_51_2,
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct tree_desc_s {
    dyn_tree: *mut ct_data_s,
    max_code: i32,
    stat_desc: *mut static_tree_desc_s,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct deflate_state {
    strm: *mut z_stream_s,
    status: i32,
    pending_buf: *mut u8,
    pending_buf_size: u64,
    pending_out: *mut u8,
    pending: i32,
    noheader: i32,
    data_type: u8,
    method: u8,
    last_flush: i32,
    w_size: u32,
    w_bits: u32,
    w_mask: u32,
    window: *mut u8,
    window_size: u64,
    prev: *mut u16,
    head: *mut u16,
    ins_h: u32,
    hash_size: u32,
    hash_bits: u32,
    hash_mask: u32,
    hash_shift: u32,
    block_start: i64,
    match_length: u32,
    prev_match: u32,
    match_available: i32,
    strstart: u32,
    match_start: u32,
    lookahead: u32,
    prev_length: u32,
    max_chain_length: u32,
    max_lazy_match: u32,
    level: i32,
    strategy: i32,
    good_match: u32,
    nice_match: i32,
    dyn_ltree: [ct_data_s; 573],
    dyn_dtree: [ct_data_s; 61],
    bl_tree: [ct_data_s; 39],
    l_desc: tree_desc_s,
    d_desc: tree_desc_s,
    bl_desc: tree_desc_s,
    bl_count: [u16; 16],
    heap: [i32; 573],
    heap_len: i32,
    heap_max: i32,
    depth: [u8; 573],
    l_buf: *mut u8,
    lit_bufsize: u32,
    last_lit: u32,
    d_buf: *mut u16,
    opt_len: u64,
    static_len: u64,
    compressed_len: u64,
    matches: u32,
    last_eob_len: i32,
    bi_buf: u16,
    bi_valid: i32,
}


fn bi_flush(s: *mut deflate_state) {
if (unsafe { (*s).bi_valid }) == 16 {
put_short!(s, unsafe { (*s).bi_buf });
unsafe {
(*s).bi_buf = 0;
}
unsafe {
(*s).bi_valid = 0;
}
} else {
if (unsafe { (*s).bi_valid }) >= 8 {
put_byte!(s, (unsafe { (*s).bi_buf }) as u8);
unsafe {
(*s).bi_buf = (((*s).bi_buf as i32) >> 8i32) as u16;
}
let __v13: i32 = unsafe { (*s).bi_valid };
unsafe {
(*s).bi_valid = __v13.wrapping_sub(8);
}
}
}
}

fn bi_windup(s: *mut deflate_state) {
if (unsafe { (*s).bi_valid }) > 8 {
put_short!(s, unsafe { (*s).bi_buf });
} else {
if (unsafe { (*s).bi_valid }) > 0 {
put_byte!(s, (unsafe { (*s).bi_buf }) as u8);
}
}
unsafe {
(*s).bi_buf = 0;
}
unsafe {
(*s).bi_valid = 0;
}
}

//  ALGORITHM
//
//      The "deflation" process uses several Huffman trees. The more
//      common source values are represented by shorter bit sequences.
//
//      Each code tree is stored in a compressed form which is itself
// a Huffman encoding of the lengths of all the code strings (in
// ascending order by source values).  The actual code strings are
// reconstructed from the lengths in the inflate process, as described
// in the deflate specification.
//
//  REFERENCES
//
//      Deutsch, L.P.,"'Deflate' Compressed Data Format Specification".
//      Available in ftp.uu.net:/pub/archiving/zip/doc/deflate-1.1.doc
//
//      Storer, James A.
//          Data Compression:  Methods and Theory, pp. 49-50.
//          Computer Science Press, 1988.  ISBN 0-7167-8156-5.
//
//      Sedgewick, R.
//          Algorithms, p290.
//          Addison-Wesley, 1983. ISBN 0-201-06672-6.
// From: trees.c,v 1.11 1996/07/24 13:41:06 me Exp $
// #include "deflate.h"
// Constants
// Bit length codes must not exceed MAX_BL_BITS bits
// end of block literal code
// repeat previous bit length 3-6 times (2 bits of repeat count)
// repeat a zero length 3-10 times  (3 bits of repeat count)
// repeat a zero length 11-138 times  (7 bits of repeat count)
/// extra bits for each length code
static extra_lbits: __SlateAlign16<[i32; 29]> = __SlateAlign16([0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 1i32, 1i32, 1i32, 1i32, 2i32, 2i32, 2i32, 2i32, 3i32, 3i32, 3i32, 3i32, 4i32, 4i32, 4i32, 4i32, 5i32, 5i32, 5i32, 5i32, 0i32]);

/// extra bits for each distance code
static extra_dbits: __SlateAlign16<[i32; 30]> = __SlateAlign16([0i32, 0i32, 0i32, 0i32, 1i32, 1i32, 2i32, 2i32, 3i32, 3i32, 4i32, 4i32, 5i32, 5i32, 6i32, 6i32, 7i32, 7i32, 8i32, 8i32, 9i32, 9i32, 10i32, 10i32, 11i32, 11i32, 12i32, 12i32, 13i32, 13i32]);

/// extra bits for each bit length code
static extra_blbits: __SlateAlign16<[i32; 19]> = __SlateAlign16([0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 2i32, 3i32, 7i32]);

static bl_order: __SlateAlign16<[u8; 19]> = __SlateAlign16([16u8, 17u8, 18u8, 0u8, 8u8, 7u8, 9u8, 6u8, 10u8, 5u8, 11u8, 4u8, 12u8, 3u8, 13u8, 2u8, 14u8, 1u8, 15u8]);

// The lengths of the bit length codes are sent in order of decreasing
// probability, to avoid transmitting the lengths for unused bit length codes.
// Local data. These are initialized only once.
static mut static_ltree: __SlateAlign16<[ct_data_s; 288]> = __SlateAlign16(unsafe { core::mem::zeroed() });

// The static literal tree. Since the bit lengths are imposed, there is no
// need for the L_CODES extra codes used during heap construction. However
// The codes 286 and 287 are needed to build a canonical tree (see zlib_tr_init
// below).
static mut static_dtree: __SlateAlign16<[ct_data_s; 30]> = __SlateAlign16(unsafe { core::mem::zeroed() });

// The static distance tree. (Actually a trivial tree since all codes use
// 5 bits.)
static mut dist_code: __SlateAlign16<[u8; 512]> = __SlateAlign16(unsafe { core::mem::zeroed() });

// distance codes. The first 256 values correspond to the distances
// 3 .. 258, the last 256 values correspond to the top 8 bits of
// the 15 bit distances.
static mut length_code: __SlateAlign16<[u8; 256]> = __SlateAlign16(unsafe { core::mem::zeroed() });

// length code for each normalized match length (0 == MIN_MATCH)
static mut base_length: __SlateAlign16<[i32; 29]> = __SlateAlign16(unsafe { core::mem::zeroed() });

// First normalized length for each code (0 = MIN_MATCH)
static mut base_dist: __SlateAlign16<[i32; 30]> = __SlateAlign16(unsafe { core::mem::zeroed() });

// First normalized distance for each code (0 = distance of 1)
#[repr(C)]
#[derive(Clone, Copy)]
struct static_tree_desc_s {
    /// static tree or NULL
    static_tree: *const ct_data_s,
    /// extra bits for each code or NULL
    extra_bits: *const i32,
    /// base index for extra_bits
    extra_base: i32,
    /// max number of elements in the tree
    elems: i32,
    /// max bit length for the codes
    max_length: i32,
}


static mut static_l_desc: static_tree_desc_s = {
let mut __t0: static_tree_desc_s = unsafe { core::mem::zeroed() };
__t0.static_tree = (unsafe { core::ptr::addr_of_mut!(static_ltree.0) as *mut ct_data_s }) as *const ct_data_s;
__t0.extra_bits = unsafe { core::ptr::addr_of!(extra_lbits.0) as *const i32 };
__t0.extra_base = 257i32;
__t0.elems = 286i32;
__t0.max_length = 15i32;
__t0
};

static mut static_d_desc: static_tree_desc_s = {
let mut __t0: static_tree_desc_s = unsafe { core::mem::zeroed() };
__t0.static_tree = (unsafe { core::ptr::addr_of_mut!(static_dtree.0) as *mut ct_data_s }) as *const ct_data_s;
__t0.extra_bits = unsafe { core::ptr::addr_of!(extra_dbits.0) as *const i32 };
__t0.extra_base = 0i32;
__t0.elems = 30i32;
__t0.max_length = 15i32;
__t0
};

static mut static_bl_desc: static_tree_desc_s = {
let mut __t0: static_tree_desc_s = unsafe { core::mem::zeroed() };
__t0.static_tree = core::ptr::null::<ct_data_s>();
__t0.extra_bits = unsafe { core::ptr::addr_of!(extra_blbits.0) as *const i32 };
__t0.extra_base = 0i32;
__t0.elems = 19i32;
__t0.max_length = 7i32;
__t0
};

// Local (static) routines in this file.
/// # Arguments
///
/// * `desc` - the tree descriptor
fn build_tree(s: *mut deflate_state, desc: *mut tree_desc_s) {
let tree: *mut ct_data_s = unsafe { (*desc).dyn_tree };
let stree: *const ct_data_s = unsafe { (*(*desc).stat_desc).static_tree };
let elems: i32 = unsafe { (*(*desc).stat_desc).elems };
let mut n: i32 = 0;
let mut m: i32 = 0; // iterate over heap elements
let mut max_code: i32 = 1i32.wrapping_neg(); // largest code with non zero frequency
let mut node: i32 = 0; // new node being created
// Construct the initial heap, with least frequent element in
// heap[SMALLEST]. The sons of heap[n] are heap[2*n] and heap[2*n+1].
// heap[0] is not used.
unsafe {
(*s).heap_len = 0;
}
unsafe {
(*s).heap_max = 573i32;
}
while n < elems {
if ((unsafe { (*tree.wrapping_offset(n as isize)).fc.freq }) as i32) != 0 {
let __v1: i32 = n;
max_code = __v1;
let __v3: i32 = unsafe { (*s).heap_len };
let __v4: i32 = __v3.wrapping_add(1);
unsafe {
(*s).heap_len = __v4;
}
unsafe {
*((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(__v4 as isize) = __v1;
}
unsafe {
*((*s).depth.as_mut_ptr() as *mut u8).wrapping_offset(n as isize) = 0;
}
} else {
unsafe {
(*tree.wrapping_offset(n as isize)).dl.len = 0;
}
}
n = n.wrapping_add(1);
}
// The pkzip format requires that at least one distance code exists,
// and that at least one bit should be sent even if there is only one
// possible code. So to avoid special checks later on we force at least
// two codes of non zero frequency.
while (unsafe { (*s).heap_len }) < 2 {
let __v8: i32 = if max_code < 2 {
let __v10: i32 = max_code.wrapping_add(1);
max_code = __v10;
__v10
} else {
0
};
let __v12: i32 = unsafe { (*s).heap_len };
let __v13: i32 = __v12.wrapping_add(1);
unsafe {
(*s).heap_len = __v13;
}
let __v14: i32 = __v8;
unsafe {
*((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(__v13 as isize) = __v14;
}
node = __v14;
unsafe {
(*tree.wrapping_offset(node as isize)).fc.freq = 1;
}
unsafe {
*((*s).depth.as_mut_ptr() as *mut u8).wrapping_offset(node as isize) = 0;
}
let __v16: u64 = unsafe { (*s).opt_len };
unsafe {
(*s).opt_len = __v16.wrapping_sub(1);
}
if !stree.is_null() {
let __v19: u64 = unsafe { (*s).static_len };
unsafe {
(*s).static_len = __v19.wrapping_sub(((*stree.wrapping_offset(node as isize)).dl.len as i32) as u64);
}
}
// node is 0 or 1 so it does not have extra bits
}
unsafe {
(*desc).max_code = max_code;
}
// The elements heap[heap_len/2+1 .. heap_len] are leaves of the tree,
// establish sub-heaps of increasing lengths:
n = (unsafe { (*s).heap_len }) / 2;
while n >= 1 {
pqdownheap(s, tree, n);
n = n.wrapping_sub(1);
}
// Construct the Huffman tree by repeatedly combining the least two
// frequent nodes.
node = elems; // next internal node of the tree
loop {
pqremove!(s, tree, n);
// n = node of least frequency
m = unsafe { *((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(1 as isize) }; // m = node of next least frequency
let __v29: i32 = unsafe { (*s).heap_max };
let __v30: i32 = __v29.wrapping_sub(1);
unsafe {
(*s).heap_max = __v30;
}
unsafe {
*((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(__v30 as isize) = n;
}
// keep the nodes sorted by frequency
let __v32: i32 = unsafe { (*s).heap_max };
let __v33: i32 = __v32.wrapping_sub(1);
unsafe {
(*s).heap_max = __v33;
}
unsafe {
*((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(__v33 as isize) = m;
}
// Create a new node father of n and m
unsafe {
(*tree.wrapping_offset(node as isize)).fc.freq = ((*tree.wrapping_offset(n as isize)).fc.freq as i32).wrapping_add((*tree.wrapping_offset(m as isize)).fc.freq as i32) as u16;
}
let mut __v34: i32;
let __UNIQUE_ID_x__5: u8 = unsafe { *((*s).depth.as_mut_ptr() as *mut u8).wrapping_offset(n as isize) };
let __UNIQUE_ID_y__6: u8 = unsafe { *((*s).depth.as_mut_ptr() as *mut u8).wrapping_offset(m as isize) };
loop {
if __types_ok__3!(__UNIQUE_ID_x__5, __UNIQUE_ID_y__6) == 0 {
unsafe { __compiletime_assert_7() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v34 = if (__UNIQUE_ID_x__5 as i32) > (__UNIQUE_ID_y__6 as i32) { __UNIQUE_ID_x__5 as i32 } else { __UNIQUE_ID_y__6 as i32 };
unsafe {
*((*s).depth.as_mut_ptr() as *mut u8).wrapping_offset(node as isize) = __v34.wrapping_add(1) as u8;
}
let __v36: u16 = node as u16;
unsafe {
(*tree.wrapping_offset(m as isize)).dl.dad = __v36;
}
unsafe {
(*tree.wrapping_offset(n as isize)).dl.dad = __v36;
}
// and insert the new node in the heap
let __v37: i32 = node;
node = __v37.wrapping_add(1);
unsafe {
*((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(1 as isize) = __v37;
}
pqdownheap(s, tree, 1i32);
if (unsafe { (*s).heap_len }) < 2 {
break;
}
}
let __v41: i32 = unsafe { (*s).heap_max };
let __v42: i32 = __v41.wrapping_sub(1);
unsafe {
(*s).heap_max = __v42;
}
unsafe {
*((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(__v42 as isize) = *((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(1 as isize);
}
// At this point, the fields freq and dad are set. We can now
// generate the bit lengths.
gen_bitlen(s, desc);
// The field len is now set, we can generate the bit codes
gen_codes(tree, max_code, unsafe { (*s).bl_count.as_mut_ptr() as *mut u16 });
}

// Send a code of the given tree. c and tree must not have side effects
// Mapping from a distance to a distance code. dist is the distance - 1 and
// must not have side effects. dist_code[256] and dist_code[257] are never
// used.
/// Initialize the various 'constant' tables. In a multi-threaded environment,
/// this function may be called by two threads concurrently, but this is
/// harmless since both invocations do exactly the same thing.
fn tr_static_init() {
let mut n: i32 = 0; // iterates over tree elements
// Construct the codes of the static literal tree
let mut bits: i32 = 0; // bit counter
// Initialize the mapping length (0..255) -> length code (0..28)
let mut length: i32 = 0; // length value
let mut code: i32 = 0; // code value
// Initialize the mapping dist (0..32K) -> dist code (0..29)
let mut dist: i32 = 0; // distance index
let mut bl_count: __SlateAlign16<[u16; 16]> = __SlateAlign16([0u16; 16]);
// number of codes at each bit length for an optimal tree
if (unsafe { static_init_done }) != 0 {
return;
}
while code < 28i32 {
unsafe {
base_length.0[code as usize] = length;
}
n = 0;
while n < 1i32 << unsafe { extra_lbits.0[code as usize] } {
let __v0: i32 = length;
length = __v0.wrapping_add(1);
unsafe {
length_code.0[__v0 as usize] = code as u8;
}
n = n.wrapping_add(1);
}
code = code.wrapping_add(1);
}
// Note that the length 255 (match length 258) can be represented
// in two different ways: code 284 + 5 bits or code 285, so we
// overwrite length_code[255] to use the best encoding:
unsafe {
length_code.0[length.wrapping_sub(1) as usize] = code as u8;
}
code = 0;
while code < 16 {
unsafe {
base_dist.0[code as usize] = dist;
}
n = 0;
while n < 1i32 << unsafe { extra_dbits.0[code as usize] } {
let __v8: i32 = dist;
dist = __v8.wrapping_add(1);
unsafe {
dist_code.0[__v8 as usize] = code as u8;
}
n = n.wrapping_add(1);
}
code = code.wrapping_add(1);
}
dist >>= 7i32; // from now on, all distances are divided by 128
while code < 30 {
unsafe {
base_dist.0[code as usize] = dist << 7i32;
}
n = 0;
while n < 1i32 << unsafe { extra_dbits.0[code as usize] }.wrapping_sub(7) {
let __v18: i32 = dist;
dist = __v18.wrapping_add(1);
unsafe {
dist_code.0[256i32.wrapping_add(__v18) as usize] = code as u8;
}
n = n.wrapping_add(1);
}
code = code.wrapping_add(1);
}
while bits <= 15 {
bl_count.0[bits as usize] = 0;
bits = bits.wrapping_add(1);
}
n = 0;
while n <= 143 {
let __v29: i32 = n;
n = __v29.wrapping_add(1);
unsafe {
static_ltree.0[__v29 as usize].dl.len = 8;
}
let __v31: *mut u16 = (bl_count.0.as_mut_ptr() as *mut u16).wrapping_offset(8 as isize);
unsafe {
*__v31 = (*__v31 as i32).wrapping_add(1) as u16;
}
}
while n <= 255 {
let __v35: i32 = n;
n = __v35.wrapping_add(1);
unsafe {
static_ltree.0[__v35 as usize].dl.len = 9;
}
let __v37: *mut u16 = (bl_count.0.as_mut_ptr() as *mut u16).wrapping_offset(9 as isize);
unsafe {
*__v37 = (*__v37 as i32).wrapping_add(1) as u16;
}
}
while n <= 279 {
let __v41: i32 = n;
n = __v41.wrapping_add(1);
unsafe {
static_ltree.0[__v41 as usize].dl.len = 7;
}
let __v43: *mut u16 = (bl_count.0.as_mut_ptr() as *mut u16).wrapping_offset(7 as isize);
unsafe {
*__v43 = (*__v43 as i32).wrapping_add(1) as u16;
}
}
while n <= 287 {
let __v47: i32 = n;
n = __v47.wrapping_add(1);
unsafe {
static_ltree.0[__v47 as usize].dl.len = 8;
}
let __v49: *mut u16 = (bl_count.0.as_mut_ptr() as *mut u16).wrapping_offset(8 as isize);
unsafe {
*__v49 = (*__v49 as i32).wrapping_add(1) as u16;
}
}
// Codes 286 and 287 do not exist, but we must include them in the
// tree construction to get a canonical Huffman tree (longest code
// all ones)
gen_codes(unsafe { core::ptr::addr_of_mut!(static_ltree.0) as *mut ct_data_s }, 287i32, bl_count.0.as_mut_ptr() as *mut u16);
// The static distance tree is trivial:
n = 0;
while n < 30 {
unsafe {
static_dtree.0[n as usize].dl.len = 5;
}
unsafe {
static_dtree.0[n as usize].fc.code = (bitrev32!(n as u32) >> 27i32) as u16;
}
n = n.wrapping_add(1);
}
unsafe {
static_init_done = 1;
}
}

static mut static_init_done: i32 = unsafe { core::mem::zeroed() };

/// Initialize the tree data structures for a new zlib stream.
#[unsafe(no_mangle)]
extern "C" fn zlib_tr_init(s: *mut deflate_state) {
tr_static_init();
unsafe {
(*s).compressed_len = 0;
}
unsafe {
(*s).l_desc.dyn_tree = (*s).dyn_ltree.as_mut_ptr() as *mut ct_data_s;
}
unsafe {
(*s).l_desc.stat_desc = core::ptr::addr_of_mut!(static_l_desc);
}
unsafe {
(*s).d_desc.dyn_tree = (*s).dyn_dtree.as_mut_ptr() as *mut ct_data_s;
}
unsafe {
(*s).d_desc.stat_desc = core::ptr::addr_of_mut!(static_d_desc);
}
unsafe {
(*s).bl_desc.dyn_tree = (*s).bl_tree.as_mut_ptr() as *mut ct_data_s;
}
unsafe {
(*s).bl_desc.stat_desc = core::ptr::addr_of_mut!(static_bl_desc);
}
unsafe {
(*s).bi_buf = 0;
}
unsafe {
(*s).bi_valid = 0;
}
unsafe {
(*s).last_eob_len = 8;
}
// enough lookahead for inflate
// Initialize the first block of the first file:
init_block(s);
}

/// Initialize a new block.
fn init_block(s: *mut deflate_state) {
// Initialize the trees.
let mut n: i32 = 0; // iterates over tree elements
while n < 286i32 {
unsafe {
(*((*s).dyn_ltree.as_mut_ptr() as *mut ct_data_s).wrapping_offset(n as isize)).fc.freq = 0;
}
n = n.wrapping_add(1);
}
n = 0;
while n < 30 {
unsafe {
(*((*s).dyn_dtree.as_mut_ptr() as *mut ct_data_s).wrapping_offset(n as isize)).fc.freq = 0;
}
n = n.wrapping_add(1);
}
n = 0;
while n < 19 {
unsafe {
(*((*s).bl_tree.as_mut_ptr() as *mut ct_data_s).wrapping_offset(n as isize)).fc.freq = 0;
}
n = n.wrapping_add(1);
}
unsafe {
(*((*s).dyn_ltree.as_mut_ptr() as *mut ct_data_s).wrapping_offset(256 as isize)).fc.freq = 1;
}
unsafe {
(*s).static_len = 0;
}
unsafe {
(*s).opt_len = 0;
}
unsafe {
(*s).matches = 0;
}
unsafe {
(*s).last_lit = 0;
}
}

// Index within the heap array of least frequent node in the Huffman tree
/// Restore the heap property by moving down the tree starting at node k,
/// exchanging a node with the smallest of its two sons if necessary, stopping
/// when the heap property is re-established (each father smaller than its
/// two sons).
///
/// # Arguments
///
/// * `tree` - the tree to restore
/// * `k` - node to move down
fn pqdownheap(s: *mut deflate_state, mut tree: *mut ct_data_s, mut k: i32) {
let v: i32 = unsafe { *((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(k as isize) };
let mut j: i32 = k << 1i32; // left son of k
while j <= unsafe { (*s).heap_len } {
// Set j to the smallest of the two sons:
if j < unsafe { (*s).heap_len } && smaller!(tree, unsafe { *((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(j.wrapping_add(1) as isize) }, unsafe { *((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(j as isize) }, (*s).depth) {
j = j.wrapping_add(1);
}
// Exit if v is smaller than both sons
if smaller!(tree, v, unsafe { *((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(j as isize) }, (*s).depth) {
break;
}
// Exchange v with the smallest son
unsafe {
*((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(k as isize) = *((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(j as isize);
}
k = j;
// And continue down the tree, setting j to the left son of k
j <<= 1i32;
}
unsafe {
*((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(k as isize) = v;
}
}

/// Compute the optimal bit lengths for a tree and update the total bit length
/// for the current block.
/// IN assertion: the fields freq and dad are set, heap[heap_max] and
///    above are the tree nodes sorted by increasing frequency.
/// OUT assertions: the field len is set to the optimal bit length, the
///     array bl_count contains the frequencies for each bit length.
///     The length opt_len is updated; static_len is also updated if stree is
///     not null.
///
/// # Arguments
///
/// * `desc` - the tree descriptor
fn gen_bitlen(s: *mut deflate_state, desc: *mut tree_desc_s) {
let tree: *mut ct_data_s = unsafe { (*desc).dyn_tree };
let max_code: i32 = unsafe { (*desc).max_code };
let stree: *const ct_data_s = unsafe { (*(*desc).stat_desc).static_tree };
let extra: *const i32 = unsafe { (*(*desc).stat_desc).extra_bits };
let base: i32 = unsafe { (*(*desc).stat_desc).extra_base };
let max_length: i32 = unsafe { (*(*desc).stat_desc).max_length };
let mut h: i32 = 0; // heap index
let mut n: i32 = 0;
let mut m: i32 = 0; // iterate over the tree elements
let mut bits: i32 = 0; // bit length
let mut xbits: i32 = 0; // extra bits
let mut f: u16 = 0; // frequency
let mut overflow: i32 = 0; // number of elements with bit length too large
while bits <= 15 {
unsafe {
*((*s).bl_count.as_mut_ptr() as *mut u16).wrapping_offset(bits as isize) = 0;
}
bits = bits.wrapping_add(1);
}
// In a first pass, compute the optimal bit lengths (which may
// overflow in the case of the bit length tree).
unsafe {
(*tree.wrapping_offset(*((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset((*s).heap_max as isize) as isize)).dl.len = 0;
}
// root of the heap
h = unsafe { (*s).heap_max }.wrapping_add(1);
while h < 573i32 {
n = unsafe { *((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(h as isize) };
bits = ((unsafe { (*tree.wrapping_offset(((*tree.wrapping_offset(n as isize)).dl.dad as i32) as isize)).dl.len }) as i32).wrapping_add(1);
if bits > max_length {
bits = max_length;
overflow = overflow.wrapping_add(1);
}
unsafe {
(*tree.wrapping_offset(n as isize)).dl.len = bits as u16;
}
// We overwrite tree[n].Dad which is no longer needed
if n <= max_code {
// not a leaf node
let __v6: *mut u16 = unsafe { (*s).bl_count.as_mut_ptr() as *mut u16 }.wrapping_offset(bits as isize);
unsafe {
*__v6 = (*__v6 as i32).wrapping_add(1) as u16;
}
xbits = 0;
if n >= base {
xbits = unsafe { *extra.wrapping_offset(n.wrapping_sub(base) as isize) };
}
f = unsafe { (*tree.wrapping_offset(n as isize)).fc.freq };
let __v10: u64 = unsafe { (*s).opt_len };
unsafe {
(*s).opt_len = __v10.wrapping_add((f as u64).wrapping_mul(bits.wrapping_add(xbits) as u64));
}
if !stree.is_null() {
let __v13: u64 = unsafe { (*s).static_len };
unsafe {
(*s).static_len = __v13.wrapping_add((f as u64).wrapping_mul(((*stree.wrapping_offset(n as isize)).dl.len as i32).wrapping_add(xbits) as u64));
}
}
}
h = h.wrapping_add(1);
}
if overflow == 0 {
return;
}
// This happens for example on obj2 and pic of the Calgary corpus
// Find the first bit length which could increase:
loop {
bits = max_length.wrapping_sub(1);
while ((unsafe { *((*s).bl_count.as_mut_ptr() as *mut u16).wrapping_offset(bits as isize) }) as i32) == 0 {
bits = bits.wrapping_sub(1);
}
let __v20: *mut u16 = unsafe { (*s).bl_count.as_mut_ptr() as *mut u16 }.wrapping_offset(bits as isize);
unsafe {
*__v20 = (*__v20 as i32).wrapping_sub(1) as u16;
}
// move one leaf down the tree
let __v23: *mut u16 = unsafe { (*s).bl_count.as_mut_ptr() as *mut u16 }.wrapping_offset(bits.wrapping_add(1) as isize);
unsafe {
*__v23 = (*__v23 as i32).wrapping_add(2) as u16;
}
// move one overflow item as its brother
let __v26: *mut u16 = unsafe { (*s).bl_count.as_mut_ptr() as *mut u16 }.wrapping_offset(max_length as isize);
unsafe {
*__v26 = (*__v26 as i32).wrapping_sub(1) as u16;
}
// The brother of the overflow item also moves one step up,
// but this does not affect bl_count[max_length]
overflow = overflow.wrapping_sub(2);
if overflow <= 0 {
break;
}
}
// Now recompute all bit lengths, scanning in increasing frequency.
// h is still equal to HEAP_SIZE. (It is simpler to reconstruct all
// lengths instead of fixing only the wrong ones. This idea is taken
// from 'ar' written by Haruhiko Okumura.)
bits = max_length;
while bits != 0 {
n = (unsafe { *((*s).bl_count.as_mut_ptr() as *mut u16).wrapping_offset(bits as isize) }) as i32;
while n != 0 {
let __v33: i32 = h.wrapping_sub(1);
h = __v33;
m = unsafe { *((*s).heap.as_mut_ptr() as *mut i32).wrapping_offset(__v33 as isize) };
if m <= max_code {
if (((unsafe { (*tree.wrapping_offset(m as isize)).dl.len }) as i32) as u32) != (bits as u32) {
let __v36: u64 = unsafe { (*s).opt_len };
unsafe {
(*s).opt_len = __v36.wrapping_add((bits as i64).wrapping_sub((*tree.wrapping_offset(m as isize)).dl.len as i64).wrapping_mul((*tree.wrapping_offset(m as isize)).fc.freq as i64) as u64);
}
unsafe {
(*tree.wrapping_offset(m as isize)).dl.len = bits as u16;
}
}
n = n.wrapping_sub(1);
}
}
bits = bits.wrapping_sub(1);
}
}

/// Generate the codes for a given tree and bit counts (which need not be
/// optimal).
/// IN assertion: the array bl_count contains the bit length statistics for
/// the given tree and the field len is set for all tree elements.
/// OUT assertion: the field code is set for all tree elements of non
///     zero code length.
///
/// # Arguments
///
/// * `tree` - the tree to decorate
/// * `max_code` - largest code with non zero frequency
/// * `bl_count` - number of codes at each bit length
fn gen_codes(tree: *mut ct_data_s, max_code: i32, bl_count: *mut u16) {
let mut next_code: __SlateAlign16<[u16; 16]> = __SlateAlign16([0u16; 16]); // next code value for each bit length
let mut code: u16 = 0; // running code value
// The distribution counts are first used to generate the code values
// without bit reversal.
let mut bits: i32 = 1; // bit index
let mut n: i32 = 0; // code index
while bits <= 15 {
let __v0: u16 = ((code as i32).wrapping_add((unsafe { *bl_count.wrapping_offset(bits.wrapping_sub(1) as isize) }) as i32) << 1i32) as u16;
code = __v0;
next_code.0[bits as usize] = __v0;
bits = bits.wrapping_add(1);
}
// Check that the bit counts in bl_count are consistent. The last code
// must be all ones.
while n <= max_code {
let len: i32 = (unsafe { (*tree.wrapping_offset(n as isize)).dl.len }) as i32;
if len != 0 {
// Now reverse the bits
let mut __v5: u32;
let mut __x: u32 = 0;
let __v6: *mut u16 = (next_code.0.as_mut_ptr() as *mut u16).wrapping_offset(len as isize);
let __v7: u16 = unsafe { *__v6 };
unsafe {
*__v6 = (__v7 as i32).wrapping_add(1) as u16;
}
__x = __v7 as u32;
let __v9: u32 = if unsafe { __slate_intrinsic_1c5364fcd9da51fe(__x) } {
__constant_bitrev32!(__x)
} else {
generic___bitrev32(__x)
};
__v5 = __v9;
unsafe {
(*tree.wrapping_offset(n as isize)).fc.code = (__v5 >> 32i32.wrapping_sub(len)) as u16;
}
}
n = n.wrapping_add(1);
}
}

// Construct one Huffman tree and assigns the code bit strings and lengths.
// Update the total bit length for the current block.
// IN assertion: the field freq is set for all tree elements.
// OUT assertions: the fields len and code are set to the optimal bit length
//     and corresponding code. The length opt_len is updated; static_len is
//     also updated if stree is not null. The field max_code is set.
/// Scan a literal or distance tree to determine the frequencies of the codes
/// in the bit length tree.
///
/// # Arguments
///
/// * `tree` - the tree to be scanned
/// * `max_code` - and its largest code of non zero frequency
fn scan_tree(s: *mut deflate_state, tree: *mut ct_data_s, max_code: i32) {
let mut n: i32 = 0; // iterates over all tree elements
let mut prevlen: i32 = 1i32.wrapping_neg(); // last emitted length
let mut curlen: i32 = 0; // length of current code
let mut nextlen: i32 = (unsafe { (*tree.wrapping_offset(0 as isize)).dl.len }) as i32; // length of next code
let mut count: i32 = 0; // repeat count of the current code
let mut max_count: i32 = 7; // max repeat count
let mut min_count: i32 = 4; // min repeat count
if nextlen == 0 {
max_count = 138;
min_count = 3;
}
unsafe {
(*tree.wrapping_offset(max_code.wrapping_add(1) as isize)).dl.len = 65535;
}
// guard
while n <= max_code {
'__slate_continue_2: {
curlen = nextlen;
nextlen = (unsafe { (*tree.wrapping_offset(n.wrapping_add(1) as isize)).dl.len }) as i32;
let __v1: i32 = count.wrapping_add(1);
count = __v1;
if __v1 < max_count && curlen == nextlen {
break '__slate_continue_2;
} else {
if count < min_count {
let __v3: *mut ct_data_s = unsafe { (*s).bl_tree.as_mut_ptr() as *mut ct_data_s }.wrapping_offset(curlen as isize);
unsafe {
(*__v3).fc.freq = ((*__v3).fc.freq as i32).wrapping_add(count) as u16;
}
} else {
if curlen != 0 {
if curlen != prevlen {
let __v6: *mut ct_data_s = unsafe { (*s).bl_tree.as_mut_ptr() as *mut ct_data_s }.wrapping_offset(curlen as isize);
unsafe {
(*__v6).fc.freq = ((*__v6).fc.freq as i32).wrapping_add(1) as u16;
}
}
let __v9: *mut ct_data_s = unsafe { (*s).bl_tree.as_mut_ptr() as *mut ct_data_s }.wrapping_offset(16 as isize);
unsafe {
(*__v9).fc.freq = ((*__v9).fc.freq as i32).wrapping_add(1) as u16;
}
} else {
if count <= 10 {
let __v12: *mut ct_data_s = unsafe { (*s).bl_tree.as_mut_ptr() as *mut ct_data_s }.wrapping_offset(17 as isize);
unsafe {
(*__v12).fc.freq = ((*__v12).fc.freq as i32).wrapping_add(1) as u16;
}
} else {
let __v15: *mut ct_data_s = unsafe { (*s).bl_tree.as_mut_ptr() as *mut ct_data_s }.wrapping_offset(18 as isize);
unsafe {
(*__v15).fc.freq = ((*__v15).fc.freq as i32).wrapping_add(1) as u16;
}
}
}
}
}
count = 0;
prevlen = curlen;
if nextlen == 0 {
max_count = 138;
min_count = 3;
} else {
if curlen == nextlen {
max_count = 6;
min_count = 3;
} else {
max_count = 7;
min_count = 4;
}
}
}
n = n.wrapping_add(1);
}
}

/// Send a literal or distance tree in compressed form, using the codes in
/// bl_tree.
///
/// # Arguments
///
/// * `tree` - the tree to be scanned
/// * `max_code` - and its largest code of non zero frequency
fn send_tree(s: *mut deflate_state, tree: *mut ct_data_s, max_code: i32) {
let mut n: i32 = 0; // iterates over all tree elements
let mut prevlen: i32 = 1i32.wrapping_neg(); // last emitted length
let mut curlen: i32 = 0; // length of current code
let mut nextlen: i32 = (unsafe { (*tree.wrapping_offset(0 as isize)).dl.len }) as i32; // length of next code
let mut count: i32 = 0; // repeat count of the current code
let mut max_count: i32 = 7; // max repeat count
let mut min_count: i32 = 4; // min repeat count
// tree[max_code+1].Len = -1;
//
// guard already set
if nextlen == 0 {
max_count = 138;
min_count = 3;
}
while n <= max_code {
'__slate_continue_2: {
curlen = nextlen;
nextlen = (unsafe { (*tree.wrapping_offset(n.wrapping_add(1) as isize)).dl.len }) as i32;
let __v1: i32 = count.wrapping_add(1);
count = __v1;
if __v1 < max_count && curlen == nextlen {
break '__slate_continue_2;
} else {
if count < min_count {
loop {
send_code!(s, curlen, (*s).bl_tree);
let __v23: i32 = count.wrapping_sub(1);
count = __v23;
if __v23 == 0 {
break;
}
}
} else {
if curlen != 0 {
if curlen != prevlen {
send_code!(s, curlen, (*s).bl_tree);
count = count.wrapping_sub(1);
}
send_code!(s, 16i32, (*s).bl_tree);
send_bits!(s, count.wrapping_sub(3), 2i32);
} else {
if count <= 10 {
send_code!(s, 17i32, (*s).bl_tree);
send_bits!(s, count.wrapping_sub(3), 3i32);
} else {
send_code!(s, 18i32, (*s).bl_tree);
send_bits!(s, count.wrapping_sub(11), 7i32);
}
}
}
}
count = 0;
prevlen = curlen;
if nextlen == 0 {
max_count = 138;
min_count = 3;
} else {
if curlen == nextlen {
max_count = 6;
min_count = 3;
} else {
max_count = 7;
min_count = 4;
}
}
}
n = n.wrapping_add(1);
}
}

/// Construct the Huffman tree for the bit lengths and return the index in
/// bl_order of the last bit length code to send.
fn build_bl_tree(s: *mut deflate_state) -> i32 {
let mut max_blindex: i32 = 0; // index of last bit length code of non zero freq
// Determine the bit length frequencies for literal and distance trees
scan_tree(s, unsafe { (*s).dyn_ltree.as_mut_ptr() as *mut ct_data_s }, unsafe { (*s).l_desc.max_code });
scan_tree(s, unsafe { (*s).dyn_dtree.as_mut_ptr() as *mut ct_data_s }, unsafe { (*s).d_desc.max_code });
// Build the bit length tree:
build_tree(s, unsafe { core::ptr::addr_of_mut!((*s).bl_desc) });
// opt_len now includes the length of the tree representations, except
// the lengths of the bit lengths codes and the 5+5+4 bits for the counts.
// Determine the number of bit length codes to send. The pkzip format
// requires that at least 4 bit length codes be sent. (appnote.txt says
// 3 but the actual value used is 4.)
max_blindex = 18i32;
while max_blindex >= 3 {
if ((unsafe { (*((*s).bl_tree.as_mut_ptr() as *mut ct_data_s).wrapping_offset((bl_order.0[max_blindex as usize] as i32) as isize)).dl.len }) as i32) != 0 {
break;
}
max_blindex = max_blindex.wrapping_sub(1);
}
// Update opt_len to include the bit length tree and counts
let __v4: u64 = unsafe { (*s).opt_len };
unsafe {
(*s).opt_len = __v4.wrapping_add(3i32.wrapping_mul(max_blindex.wrapping_add(1)).wrapping_add(5).wrapping_add(5).wrapping_add(4) as u64);
}
max_blindex
}

/// Send the header for a block using dynamic Huffman trees: the counts, the
/// lengths of the bit length codes, the literal tree and the distance tree.
/// IN assertion: lcodes >= 257, dcodes >= 1, blcodes >= 4.
///
/// # Arguments
///
/// * `lcodes` - number of codes for each tree
/// * `dcodes` - number of codes for each tree
/// * `blcodes` - number of codes for each tree
fn send_all_trees(s: *mut deflate_state, lcodes: i32, dcodes: i32, blcodes: i32) {
send_bits!(s, lcodes.wrapping_sub(257), 5i32);
// not +255 as stated in appnote.txt
send_bits!(s, dcodes.wrapping_sub(1), 5i32);
send_bits!(s, blcodes.wrapping_sub(4), 4i32);
// not -3 as stated in appnote.txt
// index in bl_order
for rank in 0..blcodes {
send_bits__4!(s, unsafe { (*((*s).bl_tree.as_mut_ptr() as *mut ct_data_s).wrapping_offset((bl_order.0[rank as usize] as i32) as isize)).dl.len }, 3i32);
}
send_tree(s, unsafe { (*s).dyn_ltree.as_mut_ptr() as *mut ct_data_s }, lcodes.wrapping_sub(1)); // literal tree
send_tree(s, unsafe { (*s).dyn_dtree.as_mut_ptr() as *mut ct_data_s }, dcodes.wrapping_sub(1)); // distance tree
}

/// Send a stored block
///
/// # Arguments
///
/// * `buf` - input block
/// * `stored_len` - length of input block
/// * `eof` - true if this is the last block for a file
#[unsafe(no_mangle)]
extern "C" fn zlib_tr_stored_block(s: *mut deflate_state, buf: *mut u8, stored_len: u64, eof: i32) {
send_bits!(s, 0i32.wrapping_add(eof), 3i32);
// send block type
unsafe {
(*s).compressed_len = unsafe { (*s).compressed_len }.wrapping_add(3).wrapping_add(7) & 18446744073709551608u64;
}
let __v19: u64 = unsafe { (*s).compressed_len };
unsafe {
(*s).compressed_len = __v19.wrapping_add(stored_len.wrapping_add(4) << 3i32);
}
copy_block(s, buf, stored_len as u32, 1i32); // with header
}

/// Send just the `stored block' type code without any length bytes or data.
#[unsafe(no_mangle)]
extern "C" fn zlib_tr_stored_type_only(s: *mut deflate_state) {
send_bits!(s, 0i32, 3i32);
bi_windup(s);
unsafe {
(*s).compressed_len = unsafe { (*s).compressed_len }.wrapping_add(3) & 18446744073709551608u64;
}
}

/// Send one empty static block to give enough lookahead for inflate.
/// This takes 10 bits, of which 7 may remain in the bit buffer.
/// The current inflate code requires 9 bits of lookahead. If the
/// last two codes for the previous block (real code plus EOB) were coded
/// on 5 bits or less, inflate may have only 5+3 bits of lookahead to decode
/// the last real code. In this case we send two empty static blocks instead
/// of one. (There are no problems if the previous block is stored or fixed.)
/// To simplify the code, we assume the worst case of last real code encoded
/// on one bit only.
#[unsafe(no_mangle)]
extern "C" fn zlib_tr_align(s: *mut deflate_state) {
send_bits!(s, 2i32, 3i32);
send_code__3!(s, 256i32, static_ltree.0);
let __v37: u64 = unsafe { (*s).compressed_len };
unsafe {
(*s).compressed_len = __v37.wrapping_add(10);
}
// 3 for block type, 7 for EOB
bi_flush(s);
// Of the 10 bits for the empty block, we have already sent
// (10 - bi_valid) bits. The lookahead for the last real code (before
// the EOB of the previous block) was thus at least one plus the length
// of the EOB plus what we have just sent of the empty static block.
if 1i32.wrapping_add(unsafe { (*s).last_eob_len }).wrapping_add(10).wrapping_sub(unsafe { (*s).bi_valid }) < 9 {
send_bits!(s, 2i32, 3i32);
send_code__3!(s, 256i32, static_ltree.0);
let __v76: u64 = unsafe { (*s).compressed_len };
unsafe {
(*s).compressed_len = __v76.wrapping_add(10);
}
bi_flush(s);
}
unsafe {
(*s).last_eob_len = 7;
}
}

/// Determine the best encoding for the current block: dynamic trees, static
/// trees or store, and output the encoded block to the zip file. This function
/// returns the total compressed length for the file so far.
///
/// # Arguments
///
/// * `buf` - input block, or NULL if too old
/// * `stored_len` - length of input block
/// * `eof` - true if this is the last block for a file
#[unsafe(no_mangle)]
extern "C" fn zlib_tr_flush_block(s: *mut deflate_state, buf: *mut u8, stored_len: u64, eof: i32) -> u64 {
let mut opt_lenb: u64 = 0;
let mut static_lenb: u64 = 0; // opt_len and static_len in bytes
let mut max_blindex: i32 = 0; // index of last bit length code of non zero freq
// Build the Huffman trees unless a stored block is forced
if (unsafe { (*s).level }) > 0 {
// Check if the file is ascii or binary
if ((unsafe { (*s).data_type }) as i32) == 2 {
set_data_type(s);
}
// Construct the literal and distance trees
build_tree(s, unsafe { core::ptr::addr_of_mut!((*s).l_desc) });
build_tree(s, unsafe { core::ptr::addr_of_mut!((*s).d_desc) });
// At this point, opt_len and static_len are the total bit lengths of
// the compressed block data, excluding the tree representations.
// Build the bit length tree for the above two trees, and get the index
// in bl_order of the last bit length code to send.
max_blindex = build_bl_tree(s);
// Determine the best encoding. Compute first the block length in bytes
opt_lenb = unsafe { (*s).opt_len }.wrapping_add(3).wrapping_add(7) >> 3i32;
static_lenb = unsafe { (*s).static_len }.wrapping_add(3).wrapping_add(7) >> 3i32;
if static_lenb <= opt_lenb {
opt_lenb = static_lenb;
}
} else {
let __v0: u64 = stored_len.wrapping_add(5);
static_lenb = __v0;
opt_lenb = __v0; // force a stored block
}
// If compression failed and this is the first and last block,
// and if the .zip file can be seeked (to rewrite the local header),
// the whole file is transformed into a stored file:
if stored_len.wrapping_add(4) <= opt_lenb && !buf.is_null() {
// 4: two words for the lengths
//
// The test buf != NULL is only necessary if LIT_BUFSIZE > WSIZE.
// Otherwise we can't have processed more than WSIZE input bytes since
// the last block flush, because compression would have been
// successful. If LIT_BUFSIZE <= WSIZE, it is never too late to
// transform a block into a stored block.
zlib_tr_stored_block(s, buf, stored_len, eof);
} else {
if static_lenb == opt_lenb {
send_bits!(s, 2i32.wrapping_add(eof), 3i32);
compress_block(s, unsafe { core::ptr::addr_of_mut!(static_ltree.0) as *mut ct_data_s }, unsafe { core::ptr::addr_of_mut!(static_dtree.0) as *mut ct_data_s });
let __v20: u64 = unsafe { (*s).compressed_len };
unsafe {
(*s).compressed_len = __v20.wrapping_add(3u64.wrapping_add((*s).static_len));
}
} else {
send_bits!(s, 4i32.wrapping_add(eof), 3i32);
send_all_trees(s, unsafe { (*s).l_desc.max_code }.wrapping_add(1), unsafe { (*s).d_desc.max_code }.wrapping_add(1), max_blindex.wrapping_add(1));
compress_block(s, unsafe { (*s).dyn_ltree.as_mut_ptr() as *mut ct_data_s }, unsafe { (*s).dyn_dtree.as_mut_ptr() as *mut ct_data_s });
let __v41: u64 = unsafe { (*s).compressed_len };
unsafe {
(*s).compressed_len = __v41.wrapping_add(3u64.wrapping_add((*s).opt_len));
}
}
}
init_block(s);
if eof != 0 {
bi_windup(s);
let __v44: u64 = unsafe { (*s).compressed_len };
unsafe {
(*s).compressed_len = __v44.wrapping_add(7);
}
// align on byte boundary
}
(unsafe { (*s).compressed_len }) >> 3i32
}

/// Save the match info and tally the frequency counts. Return true if
/// the current block must be flushed.
///
/// # Arguments
///
/// * `dist` - distance of matched string
/// * `lc` - match length-MIN_MATCH or unmatched char (if dist==0)
#[unsafe(no_mangle)]
extern "C" fn zlib_tr_tally(s: *mut deflate_state, mut dist: u32, lc: u32) -> i32 {
unsafe {
*unsafe { (*s).d_buf }.wrapping_offset((*s).last_lit as isize) = dist as u16;
}
let __v1: u32 = unsafe { (*s).last_lit };
unsafe {
(*s).last_lit = __v1.wrapping_add(1);
}
unsafe {
*unsafe { (*s).l_buf }.wrapping_offset(__v1 as isize) = lc as u8;
}
if dist == 0 {
// lc is the unmatched char
let __v3: *mut ct_data_s = unsafe { (*s).dyn_ltree.as_mut_ptr() as *mut ct_data_s }.wrapping_offset(lc as isize);
unsafe {
(*__v3).fc.freq = ((*__v3).fc.freq as i32).wrapping_add(1) as u16;
}
} else {
let __v7: u32 = unsafe { (*s).matches };
unsafe {
(*s).matches = __v7.wrapping_add(1);
}
// Here, lc is the match length - MIN_MATCH
dist = dist.wrapping_sub(1); // dist = match distance - 1
let __v11: *mut ct_data_s = unsafe { (*s).dyn_ltree.as_mut_ptr() as *mut ct_data_s }.wrapping_offset(((unsafe { length_code.0[lc as usize] }) as i32).wrapping_add(256).wrapping_add(1) as isize);
unsafe {
(*__v11).fc.freq = ((*__v11).fc.freq as i32).wrapping_add(1) as u16;
}
let __v14: *mut ct_data_s = unsafe { (*s).dyn_dtree.as_mut_ptr() as *mut ct_data_s }.wrapping_offset(d_code!(dist) as isize);
unsafe {
(*__v14).fc.freq = ((*__v14).fc.freq as i32).wrapping_add(1) as u16;
}
}
// Try to guess if it is profitable to stop the current block here
if (unsafe { (*s).last_lit }) & 4095 == 0 && (unsafe { (*s).level }) > 2 {
// Compute an upper bound for the compressed length
let mut out_length: u64 = ((unsafe { (*s).last_lit }) as u64).wrapping_mul(8);
let in_length: u64 = ((unsafe { (*s).strstart }) as i64).wrapping_sub(unsafe { (*s).block_start }) as u64;
for dcode in 0i32..30i32 {
let __v17: u64 = out_length;
out_length = __v17.wrapping_add(((unsafe { (*((*s).dyn_dtree.as_mut_ptr() as *mut ct_data_s).wrapping_offset(dcode as isize)).fc.freq }) as u64).wrapping_mul(5i64.wrapping_add((unsafe { extra_dbits.0[dcode as usize] }) as i64) as u64));
}
out_length >>= 3i32;
if (unsafe { (*s).matches }) < (unsafe { (*s).last_lit }) / 2 && out_length < in_length / 2 {
return 1;
}
}
return ((unsafe { (*s).last_lit }) == unsafe { (*s).lit_bufsize }.wrapping_sub(1)) as i32;
// We avoid equality with lit_bufsize because of wraparound at 64K
// on 16 bit machines and because stored blocks are restricted to
// 64K-1 bytes.
}

/// Send the block data compressed using the given Huffman trees
///
/// # Arguments
///
/// * `ltree` - literal tree
/// * `dtree` - distance tree
fn compress_block(s: *mut deflate_state, mut ltree: *mut ct_data_s, mut dtree: *mut ct_data_s) {
let mut dist: u32 = 0; // distance of matched string
let mut lc: i32 = 0; // match length or unmatched char (if dist == 0)
let mut lx: u32 = 0; // running index in l_buf
let mut code: u32 = 0; // the code to send
let mut extra: i32 = 0; // number of extra bits to send
if (unsafe { (*s).last_lit }) != 0 {
loop {
dist = (unsafe { *unsafe { (*s).d_buf }.wrapping_offset(lx as isize) }) as u32;
let __v0: u32 = lx;
lx = __v0.wrapping_add(1);
lc = (unsafe { *unsafe { (*s).l_buf }.wrapping_offset(__v0 as isize) }) as i32;
if dist == 0 {
send_code__2!(s, lc, ltree);
// send a literal byte
} else {
// Here, lc is the match length - MIN_MATCH
code = (unsafe { length_code.0[lc as usize] }) as u32;
send_code__2!(s, code.wrapping_add(256).wrapping_add(1), ltree);
// send the length code
extra = unsafe { extra_lbits.0[code as usize] };
if extra != 0 {
lc = lc.wrapping_sub(unsafe { base_length.0[code as usize] });
send_bits!(s, lc, extra);
// send the extra length bits
}
dist = dist.wrapping_sub(1); // dist is now the match distance - 1
code = d_code!(dist) as u32;
send_code__2!(s, code, dtree);
// send the distance code
extra = unsafe { extra_dbits.0[code as usize] };
if extra != 0 {
dist = dist.wrapping_sub((unsafe { base_dist.0[code as usize] }) as u32);
send_bits__3!(s, dist, extra);
// send the extra distance bits
}
}
// literal or match pair ?
// Check that the overlay between pending_buf and d_buf+l_buf is ok:
if lx >= unsafe { (*s).last_lit } {
break;
}
}
}
send_code__2!(s, 256i32, ltree);
unsafe {
(*s).last_eob_len = (*ltree.wrapping_offset(256 as isize)).dl.len as i32;
}
}

/// Set the data type to ASCII or BINARY, using a crude approximation:
/// binary if more than 20% of the bytes are <= 6 or >= 128, ascii otherwise.
/// IN assertion: the fields freq of dyn_ltree are set and the total of all
/// frequencies does not exceed 64K (to fit in an int on 16 bit machines).
fn set_data_type(s: *mut deflate_state) {
let mut n: i32 = 0;
let mut ascii_freq: u32 = 0;
let mut bin_freq: u32 = 0;
while n < 7 {
let __v0: i32 = n;
n = __v0.wrapping_add(1);
let __v2: u16 = unsafe { (*((*s).dyn_ltree.as_mut_ptr() as *mut ct_data_s).wrapping_offset(__v0 as isize)).fc.freq };
bin_freq = bin_freq.wrapping_add((__v2 as i32) as u32);
}
while n < 128 {
let __v6: i32 = n;
n = __v6.wrapping_add(1);
let __v8: u16 = unsafe { (*((*s).dyn_ltree.as_mut_ptr() as *mut ct_data_s).wrapping_offset(__v6 as isize)).fc.freq };
ascii_freq = ascii_freq.wrapping_add((__v8 as i32) as u32);
}
while n < 256 {
let __v12: i32 = n;
n = __v12.wrapping_add(1);
let __v14: u16 = unsafe { (*((*s).dyn_ltree.as_mut_ptr() as *mut ct_data_s).wrapping_offset(__v12 as isize)).fc.freq };
bin_freq = bin_freq.wrapping_add((__v14 as i32) as u32);
}
unsafe {
(*s).data_type = (if bin_freq > ascii_freq >> 2i32 { 0i32 } else { 1i32 }) as u8;
}
}

/// Copy a stored block, storing first the length and its
/// one's complement if requested.
///
/// # Arguments
///
/// * `buf` - the input data
/// * `len` - its length
/// * `header` - true if block header must be written
fn copy_block(s: *mut deflate_state, buf: *mut u8, len: u32, header: i32) {
bi_windup(s); // align on byte boundary
unsafe {
(*s).last_eob_len = 8;
}
// enough lookahead for inflate
if header != 0 {
put_short!(s, len as u16);
put_short!(s, !len as u16);
}
// bundle up the put_byte(s, *buf++) calls
unsafe { memcpy(unsafe { (*s).pending_buf }.wrapping_offset((*s).pending as isize) as *mut (), buf as *const (), len as u64) };
unsafe {
(*s).pending = ((*s).pending as u32).wrapping_add(len) as i32;
}
}
