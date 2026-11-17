
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! inflate.c -- zlib decompression
//! Copyright (C) 1995-2005 Mark Adler
//! For conditions of distribution and use, see copyright notice in zlib.h
//!
//! Based on zlib 1.2.3 but modified for the Linux Kernel by
//! Richard Purdie <richard@openedhand.com>
//!
//! Changes mainly for static instead of dynamic memory allocation
//! 32K LZ77 window
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
// Useful shorthand for "is this condition known at compile-time?"
//
// Note that the condition may involve non-constant values,
// but the compiler may know enough about the details of the
// values to determine that the condition is statically true.
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
/// Return the low n bits of the bit accumulator (n < 16)
macro_rules! BITS {
($n:expr, $hold:expr) => {
(($hold) as u32) & (1u32 << $n).wrapping_sub(1)
};
}

/// Remove zero to seven bits as needed to go to a byte boundary
macro_rules! BYTEBITS {
($hold:expr, $bits:expr) => {
loop {
let __v: u64 = $hold;
let __v2: u64 = __v >> (($bits) & 7);
$hold = __v2;
let __v3: u32 = $bits;
let __v4: u32 = __v3.wrapping_sub(($bits) & 7);
$bits = __v4;
if true {
break;
}
}
};
}

macro_rules! DO16 {
($buf:expr, $s1:expr, $s2:expr) => {
DO8!($buf, 0i32, $s1, $s2);
DO8!($buf, 8i32, $s1, $s2);
};
}

macro_rules! DO1 {
($buf:expr, $i:expr, $s1:expr, $s2:expr) => {
let __v: u64 = $s1;
let __v2: u64 = __v.wrapping_add(((unsafe { *$buf.wrapping_offset(($i) as isize) }) as i32) as u64);
$s1 = __v2;
let __v3: u64 = $s2;
let __v4: u64 = __v3.wrapping_add($s1);
$s2 = __v4;
};
}

macro_rules! DO2 {
($buf:expr, $i:expr, $s1:expr, $s2:expr) => {
DO1!($buf, $i, $s1, $s2);
DO1!($buf, $i.wrapping_add(1), $s1, $s2);
};
}

macro_rules! DO4 {
($buf:expr, $i:expr, $s1:expr, $s2:expr) => {
DO2!($buf, $i, $s1, $s2);
DO2!($buf, $i.wrapping_add(2), $s1, $s2);
};
}

macro_rules! DO8 {
($buf:expr, $i:expr, $s1:expr, $s2:expr) => {
DO4!($buf, $i, $s1, $s2);
DO4!($buf, $i.wrapping_add(4), $s1, $s2);
};
}

/// Remove n bits from the bit accumulator
macro_rules! DROPBITS__2 {
($n:expr, $hold:expr, $bits:expr) => {
loop {
let __v: u64 = $hold;
let __v2: u64 = __v >> (($n) as i32);
$hold = __v2;
let __v3: u32 = $bits;
let __v4: u32 = __v3.wrapping_sub(($n) as u32);
$bits = __v4;
if true {
break;
}
}
};
}

/// Remove n bits from the bit accumulator
macro_rules! DROPBITS {
($n:expr, $hold:expr, $bits:expr) => {
loop {
let __v: u64 = $hold;
let __v2: u64 = __v >> $n;
$hold = __v2;
let __v3: u32 = $bits;
let __v4: u32 = __v3.wrapping_sub(($n) as u32);
$bits = __v4;
if true {
break;
}
}
};
}

macro_rules! INFLATE_NEED_CHECKSUM {
() => {
1i32
};
}

macro_rules! INFLATE_NEED_UPDATEWINDOW {
() => {
1i32
};
}

macro_rules! INFLATE_RESET_HOOK {
() => {
while false {
}
};
}

macro_rules! INFLATE_TYPEDO_HOOK {
() => {
while false {
}
};
}

/// Clear the input bit accumulator
macro_rules! INITBITS {
($hold:expr, $bits:expr) => {
loop {
$hold = 0;
$bits = 0;
if true {
break;
}
}
};
}

/// Load registers with state in inflate() for speed
macro_rules! LOAD {
($put:expr, $strm:expr, $left:expr, $next:expr, $have:expr, $hold:expr, $state:expr, $bits:expr) => {
loop {
$put = unsafe { (*$strm).next_out };
$left = (unsafe { (*$strm).avail_out }) as u32;
$next = unsafe { (*$strm).next_in };
$have = (unsafe { (*$strm).avail_in }) as u32;
$hold = unsafe { (*$state).hold };
$bits = unsafe { (*$state).bits };
if true {
break;
}
}
};
}

/// Restore state from registers in inflate()
macro_rules! RESTORE {
($strm:expr, $put:expr, $left:expr, $next:expr, $have:expr, $state:expr, $hold:expr, $bits:expr) => {
loop {
unsafe {
(*$strm).next_out = $put;
}
unsafe {
(*$strm).avail_out = ($left) as u64;
}
unsafe {
(*$strm).next_in = $next;
}
unsafe {
(*$strm).avail_in = ($have) as u64;
}
unsafe {
(*$state).hold = $hold;
}
unsafe {
(*$state).bits = $bits;
}
if true {
break;
}
}
};
}

/// Reverse the bytes in a 32-bit value
macro_rules! REVERSE {
($q:expr) => {
(($q) >> 24i32 & 255).wrapping_add(($q) >> 8i32 & 65280).wrapping_add((($q) & 65280) << 8i32).wrapping_add((($q) & 255) << 24i32)
};
}

/// check function to use adler32() for zlib or crc32() for gzip
macro_rules! UPDATE__2 {
($check:expr, $buf:expr, $len:expr) => {
zlib_adler32($check, $buf, ($len) as u32)
};
}

/// check function to use adler32() for zlib or crc32() for gzip
macro_rules! UPDATE {
($check:expr, $buf:expr, $len:expr) => {
zlib_adler32($check, ($buf) as *const u8, $len)
};
}

macro_rules! WS {
($strm:expr) => {
(unsafe { (*$strm).workspace }) as *mut inflate_workspace
};
}

unsafe extern "C" {
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn zlib_inflate_table(arg0: u32, arg1: *mut u16, arg2: u32, arg3: *mut *mut __SlateRecord_inftrees_h_31, arg4: *mut u32, arg5: *mut u16) -> i32;
    fn inflate_fast(arg0: *mut z_stream_s, arg1: u32);
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


// Only for inflate at present
fn zlib_adler32(adler: u64, mut buf: *const u8, mut len: u32) -> u64 {
let mut s1: u64 = adler & 65535;
let mut s2: u64 = adler >> 16i32 & 65535;
let mut k: i32 = 0;
if buf.is_null() {
return 1;
}
while len > 0 {
k = (if len < 5552 { len } else { 5552u32 }) as i32;
len = len.wrapping_sub(k as u32);
while k >= 16 {
DO16!(buf, s1, s2);
let __v66: *const u8 = buf;
buf = __v66.wrapping_offset(16 as isize);
k = k.wrapping_sub(16);
}
if k != 0 {
loop {
let __v71: *const u8 = buf;
buf = __v71.wrapping_offset(1 as isize);
let __v73: u8 = unsafe { *__v71 };
s1 = s1.wrapping_add((__v73 as i32) as u64);
s2 = s2.wrapping_add(s1);
let __v80: i32 = k.wrapping_sub(1);
k = __v80;
if __v80 == 0 {
break;
}
}
}
s1 %= 65521;
s2 %= 65521;
}
s2 << 16i32 | s1
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_inftrees_h_31 {
    op: u8,
    bits: u8,
    val: u16,
}


// Maximum size of dynamic tree.  The maximum found in a long but non-
// exhaustive search was 1444 code structures (852 for length/literals
// and 592 for distances, the latter actually the result of an
// exhaustive search).  The true maximum is not known, but the value
// below is more than safe.
#[repr(C)]
#[derive(Clone, Copy)]
struct inflate_state {
    mode: u32,
    last: i32,
    wrap: i32,
    havedict: i32,
    flags: i32,
    dmax: u32,
    check: u64,
    total: u64,
    wbits: u32,
    wsize: u32,
    whave: u32,
    write: u32,
    window: *mut u8,
    hold: u64,
    bits: u32,
    length: u32,
    offset: u32,
    extra: u32,
    lencode: *const __SlateRecord_inftrees_h_31,
    distcode: *const __SlateRecord_inftrees_h_31,
    lenbits: u32,
    distbits: u32,
    ncode: u32,
    nlen: u32,
    ndist: u32,
    have: u32,
    next: *mut __SlateRecord_inftrees_h_31,
    lens: [u16; 320],
    work: [u16; 288],
    codes: [__SlateRecord_inftrees_h_31; 2048],
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct inflate_workspace {
    inflate_state: inflate_state,
    working_window: [u8; 32768],
}


// architecture-specific bits
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.inflate.zlib_inflate_workspacesize")]
extern "C" fn zlib_inflate_workspacesize() -> i32 {
core::mem::size_of::<inflate_workspace>() as i32
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.inflate.zlib_inflateReset")]
extern "C" fn zlib_inflateReset(strm: *mut z_stream_s) -> i32 {
if strm.is_null() || unsafe { (*strm).state }.is_null() {
return 2i32.wrapping_neg();
}
let state: *mut inflate_state = (unsafe { (*strm).state }) as *mut inflate_state;
unsafe {
(*state).total = 0;
}
unsafe {
(*strm).total_out = 0;
}
unsafe {
(*strm).total_in = 0;
}
unsafe {
(*strm).msg = core::ptr::null_mut::<u8>();
}
unsafe {
(*strm).adler = 1;
}
// to support ill-conceived Java test suite
unsafe {
(*state).mode = 0;
}
unsafe {
(*state).last = 0;
}
unsafe {
(*state).havedict = 0;
}
unsafe {
(*state).dmax = 32768;
}
unsafe {
(*state).hold = 0;
}
unsafe {
(*state).bits = 0;
}
let __v0: *mut __SlateRecord_inftrees_h_31 = unsafe { (*state).codes.as_mut_ptr() as *mut __SlateRecord_inftrees_h_31 };
unsafe {
(*state).next = __v0;
}
let __v1: *const __SlateRecord_inftrees_h_31 = __v0 as *const __SlateRecord_inftrees_h_31;
unsafe {
(*state).distcode = __v1;
}
unsafe {
(*state).lencode = __v1;
}
// Initialise Window
unsafe {
(*state).wsize = 1u32 << (*state).wbits;
}
unsafe {
(*state).write = 0;
}
unsafe {
(*state).whave = 0;
}
INFLATE_RESET_HOOK!();
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.inflate.zlib_inflateInit2")]
extern "C" fn zlib_inflateInit2(strm: *mut z_stream_s, mut windowBits: i32) -> i32 {
if strm.is_null() {
return 2i32.wrapping_neg();
}
unsafe {
(*strm).msg = core::ptr::null_mut::<u8>();
}
// in case we return an error
let state: *mut inflate_state = unsafe { core::ptr::addr_of_mut!((*WS!(strm)).inflate_state) };
unsafe {
(*strm).state = state as *mut internal_state;
}
if windowBits < 0 {
unsafe {
(*state).wrap = 0;
}
windowBits = windowBits.wrapping_neg();
} else {
unsafe {
(*state).wrap = (windowBits >> 4i32).wrapping_add(1);
}
}
if windowBits < 8 || windowBits > 15 {
return 2i32.wrapping_neg();
}
unsafe {
(*state).wbits = windowBits as u32;
}
unsafe {
(*state).window = ((*WS!(strm)).working_window.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize);
}
zlib_inflateReset(strm)
}

/// Return state with length and distance decoding tables and index sizes set to
/// fixed code decoding.  This returns fixed tables from inffixed.h.
fn zlib_fixedtables(state: *mut inflate_state) {
// inffixed.h -- table for decoding fixed codes
// Generated automatically by makefixed().
// WARNING: this file should *not* be used by applications. It
// is part of the implementation of the compression library and
// is subject to change. Applications should only use zlib.h.
unsafe {
(*state).lencode = core::ptr::addr_of!(lenfix.0) as *const __SlateRecord_inftrees_h_31;
}
unsafe {
(*state).lenbits = 9;
}
unsafe {
(*state).distcode = core::ptr::addr_of!(distfix.0) as *const __SlateRecord_inftrees_h_31;
}
unsafe {
(*state).distbits = 5;
}
}

static lenfix: __SlateAlign16<[__SlateRecord_inftrees_h_31; 512]> = __SlateAlign16([__SlateRecord_inftrees_h_31 { op: 96u8, bits: 7u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 80u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 16u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 8u8, val: 115u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 31u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 112u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 48u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 192u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 10u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 96u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 32u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 160u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 128u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 64u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 224u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 6u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 88u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 24u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 144u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 59u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 120u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 56u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 208u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 17u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 104u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 40u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 176u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 8u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 136u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 72u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 240u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 4u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 84u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 20u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 227u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 43u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 116u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 52u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 200u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 13u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 100u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 36u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 168u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 4u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 132u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 68u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 232u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 8u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 92u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 28u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 152u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 83u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 124u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 60u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 216u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 23u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 108u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 44u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 184u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 12u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 140u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 76u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 248u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 3u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 82u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 18u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 163u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 35u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 114u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 50u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 196u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 11u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 98u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 34u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 164u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 2u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 130u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 66u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 228u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 7u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 90u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 26u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 148u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 67u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 122u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 58u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 212u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 19u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 106u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 42u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 180u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 10u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 138u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 74u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 244u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 5u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 86u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 22u16 }, __SlateRecord_inftrees_h_31 { op: 64u8, bits: 8u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 51u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 118u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 54u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 204u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 15u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 102u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 38u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 172u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 6u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 134u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 70u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 236u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 9u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 94u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 30u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 156u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 99u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 126u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 62u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 220u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 27u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 110u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 46u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 188u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 14u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 142u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 78u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 252u16 }, __SlateRecord_inftrees_h_31 { op: 96u8, bits: 7u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 81u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 17u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 131u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 31u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 113u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 49u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 194u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 10u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 97u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 33u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 162u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 1u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 129u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 65u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 226u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 6u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 89u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 25u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 146u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 59u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 121u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 57u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 210u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 17u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 105u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 41u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 178u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 9u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 137u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 73u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 242u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 4u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 85u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 21u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 8u8, val: 258u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 43u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 117u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 53u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 202u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 13u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 101u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 37u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 170u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 5u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 133u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 69u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 234u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 8u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 93u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 29u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 154u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 83u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 125u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 61u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 218u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 23u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 109u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 45u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 186u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 13u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 141u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 77u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 250u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 3u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 83u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 19u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 195u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 35u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 115u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 51u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 198u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 11u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 99u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 35u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 166u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 3u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 131u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 67u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 230u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 7u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 91u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 27u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 150u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 67u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 123u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 59u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 214u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 19u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 107u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 43u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 182u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 11u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 139u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 75u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 246u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 5u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 87u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 23u16 }, __SlateRecord_inftrees_h_31 { op: 64u8, bits: 8u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 51u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 119u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 55u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 206u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 15u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 103u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 39u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 174u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 7u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 135u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 71u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 238u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 9u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 95u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 31u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 158u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 99u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 127u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 63u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 222u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 27u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 111u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 47u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 190u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 15u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 143u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 79u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 254u16 }, __SlateRecord_inftrees_h_31 { op: 96u8, bits: 7u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 80u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 16u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 8u8, val: 115u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 31u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 112u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 48u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 193u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 10u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 96u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 32u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 161u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 128u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 64u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 225u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 6u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 88u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 24u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 145u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 59u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 120u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 56u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 209u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 17u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 104u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 40u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 177u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 8u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 136u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 72u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 241u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 4u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 84u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 20u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 227u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 43u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 116u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 52u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 201u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 13u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 100u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 36u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 169u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 4u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 132u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 68u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 233u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 8u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 92u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 28u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 153u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 83u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 124u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 60u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 217u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 23u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 108u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 44u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 185u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 12u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 140u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 76u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 249u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 3u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 82u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 18u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 163u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 35u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 114u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 50u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 197u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 11u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 98u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 34u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 165u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 2u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 130u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 66u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 229u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 7u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 90u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 26u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 149u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 67u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 122u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 58u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 213u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 19u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 106u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 42u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 181u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 10u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 138u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 74u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 245u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 5u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 86u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 22u16 }, __SlateRecord_inftrees_h_31 { op: 64u8, bits: 8u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 51u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 118u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 54u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 205u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 15u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 102u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 38u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 173u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 6u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 134u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 70u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 237u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 9u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 94u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 30u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 157u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 99u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 126u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 62u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 221u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 27u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 110u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 46u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 189u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 14u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 142u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 78u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 253u16 }, __SlateRecord_inftrees_h_31 { op: 96u8, bits: 7u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 81u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 17u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 131u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 31u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 113u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 49u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 195u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 10u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 97u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 33u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 163u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 1u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 129u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 65u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 227u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 6u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 89u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 25u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 147u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 59u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 121u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 57u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 211u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 17u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 105u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 41u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 179u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 9u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 137u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 73u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 243u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 4u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 85u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 21u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 8u8, val: 258u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 43u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 117u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 53u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 203u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 13u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 101u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 37u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 171u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 5u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 133u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 69u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 235u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 8u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 93u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 29u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 155u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 83u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 125u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 61u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 219u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 23u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 109u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 45u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 187u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 13u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 141u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 77u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 251u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 3u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 83u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 19u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 195u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 35u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 115u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 51u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 199u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 11u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 99u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 35u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 167u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 3u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 131u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 67u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 231u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 7u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 91u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 27u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 151u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 67u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 123u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 59u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 215u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 19u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 107u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 43u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 183u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 11u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 139u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 75u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 247u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 5u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 87u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 23u16 }, __SlateRecord_inftrees_h_31 { op: 64u8, bits: 8u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 51u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 119u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 55u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 207u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 15u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 103u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 39u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 175u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 7u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 135u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 71u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 239u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 9u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 95u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 31u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 159u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 99u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 127u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 63u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 223u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 27u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 111u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 47u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 191u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 15u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 143u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 79u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 255u16 }]);

static distfix: __SlateAlign16<[__SlateRecord_inftrees_h_31; 32]> = __SlateAlign16([__SlateRecord_inftrees_h_31 { op: 16u8, bits: 5u8, val: 1u16 }, __SlateRecord_inftrees_h_31 { op: 23u8, bits: 5u8, val: 257u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 5u8, val: 17u16 }, __SlateRecord_inftrees_h_31 { op: 27u8, bits: 5u8, val: 4097u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 5u8, val: 5u16 }, __SlateRecord_inftrees_h_31 { op: 25u8, bits: 5u8, val: 1025u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 5u8, val: 65u16 }, __SlateRecord_inftrees_h_31 { op: 29u8, bits: 5u8, val: 16385u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 5u8, val: 3u16 }, __SlateRecord_inftrees_h_31 { op: 24u8, bits: 5u8, val: 513u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 5u8, val: 33u16 }, __SlateRecord_inftrees_h_31 { op: 28u8, bits: 5u8, val: 8193u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 5u8, val: 9u16 }, __SlateRecord_inftrees_h_31 { op: 26u8, bits: 5u8, val: 2049u16 }, __SlateRecord_inftrees_h_31 { op: 22u8, bits: 5u8, val: 129u16 }, __SlateRecord_inftrees_h_31 { op: 64u8, bits: 5u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 5u8, val: 2u16 }, __SlateRecord_inftrees_h_31 { op: 23u8, bits: 5u8, val: 385u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 5u8, val: 25u16 }, __SlateRecord_inftrees_h_31 { op: 27u8, bits: 5u8, val: 6145u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 5u8, val: 7u16 }, __SlateRecord_inftrees_h_31 { op: 25u8, bits: 5u8, val: 1537u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 5u8, val: 97u16 }, __SlateRecord_inftrees_h_31 { op: 29u8, bits: 5u8, val: 24577u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 5u8, val: 4u16 }, __SlateRecord_inftrees_h_31 { op: 24u8, bits: 5u8, val: 769u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 5u8, val: 49u16 }, __SlateRecord_inftrees_h_31 { op: 28u8, bits: 5u8, val: 12289u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 5u8, val: 13u16 }, __SlateRecord_inftrees_h_31 { op: 26u8, bits: 5u8, val: 3073u16 }, __SlateRecord_inftrees_h_31 { op: 22u8, bits: 5u8, val: 193u16 }, __SlateRecord_inftrees_h_31 { op: 64u8, bits: 5u8, val: 0u16 }]);

/// Update the window with the last wsize (normally 32K) bytes written before
/// returning. This is only called when a window is already in use, or when
/// output has been written during this inflate call, but the end of the deflate
/// stream has not been reached yet. It is also called to window dictionary data
/// when a dictionary is loaded.
///
/// Providing output buffers larger than 32K to inflate() should provide a speed
/// advantage, since only the last 32K of output is copied to the sliding window
/// upon return from inflate(), and since all distances after the first 32K of
/// output will fall in the output data, making match copies simpler and faster.
/// The advantage may be dependent on the size of the processor's data caches.
fn zlib_updatewindow(strm: *mut z_stream_s, out: u32) {
let mut copy: u32 = 0;
let mut dist: u32 = 0;
let state: *mut inflate_state = (unsafe { (*strm).state }) as *mut inflate_state;
// copy state->wsize or less output bytes into the circular window
copy = (out as u64).wrapping_sub(unsafe { (*strm).avail_out }) as u32;
if copy >= unsafe { (*state).wsize } {
unsafe { memcpy((*state).window as *mut (), unsafe { (*strm).next_out }.wrapping_offset(-((*state).wsize as isize)) as *const (), (*state).wsize as u64) };
unsafe {
(*state).write = 0;
}
unsafe {
(*state).whave = (*state).wsize;
}
} else {
dist = unsafe { (*state).wsize }.wrapping_sub(unsafe { (*state).write });
if dist > copy {
dist = copy;
}
unsafe { memcpy(unsafe { (*state).window }.wrapping_offset((*state).write as isize) as *mut (), unsafe { (*strm).next_out }.wrapping_offset(-(copy as isize)) as *const (), dist as u64) };
copy = copy.wrapping_sub(dist);
if copy != 0 {
unsafe { memcpy((*state).window as *mut (), unsafe { (*strm).next_out }.wrapping_offset(-(copy as isize)) as *const (), copy as u64) };
unsafe {
(*state).write = copy;
}
unsafe {
(*state).whave = (*state).wsize;
}
} else {
let __v3: u32 = unsafe { (*state).write };
unsafe {
(*state).write = __v3.wrapping_add(dist);
}
if (unsafe { (*state).write }) == unsafe { (*state).wsize } {
unsafe {
(*state).write = 0;
}
}
if (unsafe { (*state).whave }) < unsafe { (*state).wsize } {
let __v6: u32 = unsafe { (*state).whave };
unsafe {
(*state).whave = __v6.wrapping_add(dist);
}
}
}
}
}

/// At the end of a Deflate-compressed PPP packet, we expect to have seen
/// a `stored' block type value but not the (zero) length bytes.
///
/// Returns true if inflate is currently at the end of a block generated by
/// Z_SYNC_FLUSH or Z_FULL_FLUSH. This function is used by one PPP
/// implementation to provide an additional safety check. PPP uses
/// Z_SYNC_FLUSH but removes the length bytes of the resulting empty stored
/// block. When decompressing, PPP checks that at the end of input packet,
/// inflate is waiting for these length bytes.
fn zlib_inflateSyncPacket(strm: *mut z_stream_s) -> i32 {
if strm.is_null() || unsafe { (*strm).state }.is_null() {
return 2i32.wrapping_neg();
}
let state: *mut inflate_state = (unsafe { (*strm).state }) as *mut inflate_state;
if ((unsafe { (*state).mode }) as u32) == 13 && (unsafe { (*state).bits }) == 0 {
unsafe {
(*state).mode = 11;
}
return 0;
}
3i32.wrapping_neg()
}

// Macros for inflate():
// Get a byte of input into the bit accumulator, or return from inflate()
// if there is no input available.
// Assure that there are at least n bits in the bit accumulator.  If there is
// not enough available input to do that, then return from inflate().
// inflate() uses a state machine to process as much input data and generate as
// much output data as possible before returning.  The state machine is
// structured roughly as follows:
//
//  for (;;) switch (state) {
//  ...
//  case STATEn:
//      if (not enough input data or output space to make progress)
//          return;
//      ... make progress ...
//      state = STATEm;
//      break;
//  ...
//  }
//
// so when inflate() is called again, the same case is attempted again, and
// if the appropriate resources are provided, the machine proceeds to the
// next state.  The NEEDBITS() macro is usually the way the state evaluates
// whether it can proceed or should return.  NEEDBITS() does the return if
// the requested bits are not available.  The typical use of the BITS macros
// is:
//
//      NEEDBITS(n);
//      ... do something with BITS(n) ...
//      DROPBITS(n);
//
// where NEEDBITS(n) either returns from inflate() if there isn't enough
// input left to load n bits into the accumulator, or it continues.  BITS(n)
// gives the low n bits in the accumulator.  When done, DROPBITS(n) drops
// the low n bits off the accumulator.  INITBITS() clears the accumulator
// and sets the number of available bits to zero.  BYTEBITS() discards just
// enough bits to put the accumulator on a byte boundary.  After BYTEBITS()
// and a NEEDBITS(8), then BITS(8) would return the next byte in the stream.
//
// NEEDBITS(n) uses PULLBYTE() to get an available byte of input, or to return
// if there is no input available.  The decoding of variable length codes uses
// PULLBYTE() directly in order to pull just enough bytes to decode the next
// code, and no more.
//
// Some states loop until they get enough input, making sure that enough
// state information is maintained to continue the loop where it left off
// if NEEDBITS() returns in the loop.  For example, want, need, and keep
// would all have to actually be part of the saved state in case NEEDBITS()
// returns:
//
//  case STATEw:
//      while (want < need) {
//          NEEDBITS(n);
//          keep[want++] = BITS(n);
//          DROPBITS(n);
//      }
//      state = STATEx;
//  case STATEx:
//
// As shown above, if the next state is also the next case, then the break
// is omitted.
//
// A state may also return if there is not enough output space available to
// complete that state.  Those states are copying stored data, writing a
// literal byte, and copying a matching string.
//
// When returning, a "goto inf_leave" is used to update the total counters,
// update the check value, and determine whether any progress has been made
// during that inflate() call in order to return the proper return code.
// Progress is defined as a change in either strm->avail_in or strm->avail_out.
// When there is a window, goto inf_leave will update the window with the last
// output written.  If a goto inf_leave occurs in the middle of decompression
// and there is no window currently, goto inf_leave will create one and copy
// output to the window for the next call of inflate().
//
// In this implementation, the flush parameter of inflate() only affects the
// return code (per zlib.h).  inflate() always writes as much as possible to
// strm->next_out, given the space available and the provided input--the effect
// documented in zlib.h of Z_SYNC_FLUSH.  Furthermore, inflate() always defers
// the allocation of and copying into a sliding window until necessary, which
// provides the effect documented in zlib.h for Z_FINISH when the entire input
// stream available.  So the only thing the flush parameter actually does is:
// when flush is set to Z_FINISH, inflate() cannot return Z_OK.  Instead it
// will return Z_BUF_ERROR if it has not reached the end of the stream.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.inflate.zlib_inflate")]
extern "C" fn zlib_inflate(strm: *mut z_stream_s, flush: i32) -> i32 {
// return code
// permutation of code lengths
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_236: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32; // length to copy for repeats, bits to drop
let mut len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_237: *mut u32 = core::ptr::addr_of_mut!(len) as *mut u32; // parent table entry
let mut last: core::mem::MaybeUninit<__SlateRecord_inftrees_h_31> = core::mem::MaybeUninit::uninit();
let __slate_slot_238: *mut __SlateRecord_inftrees_h_31 = core::ptr::addr_of_mut!(last) as *mut __SlateRecord_inftrees_h_31; // current decoding table entry
let mut this: core::mem::MaybeUninit<__SlateRecord_inftrees_h_31> = core::mem::MaybeUninit::uninit();
let __slate_slot_239: *mut __SlateRecord_inftrees_h_31 = core::ptr::addr_of_mut!(this) as *mut __SlateRecord_inftrees_h_31; // where to copy match bytes from
let mut from: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_240: *mut *mut u8 = core::ptr::addr_of_mut!(from) as *mut *mut u8; // number of stored or match bytes to copy
let mut copy: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_241: *mut u32 = core::ptr::addr_of_mut!(copy) as *mut u32; // save starting available input and output
let mut out: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_242: *mut u32 = core::ptr::addr_of_mut!(out) as *mut u32;
let mut r#in: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_243: *mut u32 = core::ptr::addr_of_mut!(r#in) as *mut u32; // bits in bit buffer
let mut bits: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_244: *mut u32 = core::ptr::addr_of_mut!(bits) as *mut u32; // bit buffer
let mut hold: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_245: *mut u64 = core::ptr::addr_of_mut!(hold) as *mut u64; // available input and output
let mut left: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_246: *mut u32 = core::ptr::addr_of_mut!(left) as *mut u32;
let mut have: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_247: *mut u32 = core::ptr::addr_of_mut!(have) as *mut u32; // next output
let mut put: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_248: *mut *mut u8 = core::ptr::addr_of_mut!(put) as *mut *mut u8; // next input
let mut next: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_249: *mut *const u8 = core::ptr::addr_of_mut!(next) as *mut *const u8;
let mut state: core::mem::MaybeUninit<*mut inflate_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_250: *mut *mut inflate_state = core::ptr::addr_of_mut!(state) as *mut *mut inflate_state;
unsafe {
let __slate_auto_bytes = __slate_slot_250 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_249 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_248 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_247 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_246 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_245 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_244 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_243 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_242 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_241 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_10 = __slate_slot_240 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_11 = __slate_slot_239 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_12 = __slate_slot_238 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_13 = __slate_slot_237 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_13.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_236 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// Do not check for strm->next_out == NULL here as ppc zImage
// inflates to strm->next_out = 0
if strm.is_null() || unsafe { (*strm).state }.is_null() || unsafe { (*strm).next_in }.is_null() && (*strm).avail_in != 0 {
return 2i32.wrapping_neg();
} else {
*__slate_slot_250 = (*strm).state as *mut inflate_state;
if ((*(*__slate_slot_250)).mode as u32) == 11 {
(*(*__slate_slot_250)).mode = 12;
}
// skip check
LOAD!(*__slate_slot_248, strm, *__slate_slot_246, *__slate_slot_249, *__slate_slot_247, *__slate_slot_245, *__slate_slot_250, *__slate_slot_244);
*__slate_slot_243 = *__slate_slot_247;
*__slate_slot_242 = *__slate_slot_246;
*__slate_slot_236 = 0;
'__join_8: {
'__join_12: {
'__join_13: {
'__join_14: {
'__join_11: {
'__loop_194: loop {
'__join_170: {
'__join_172: {
'__join_175: {
'__join_150: {
'__join_39: {
'__join_48: {
'__join_60: {
'__join_67: {
'__join_85: {
'__join_92: {
let __t1: u32 = (*(*__slate_slot_250)).mode as u32;
if __t1 == 0 {
if (*(*__slate_slot_250)).wrap == 0 {
(*(*__slate_slot_250)).mode = 12;
continue '__loop_194;
} else {
'__loop_189: loop {
if *__slate_slot_244 < 16 {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v235: u32 = *__slate_slot_247;
*__slate_slot_247 = __v235.wrapping_sub(1);
let __v233: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v233.wrapping_offset(1 as isize);
let __v230: u64 = *__slate_slot_245;
*__slate_slot_245 = __v230.wrapping_add((*__v233 as u64) << *__slate_slot_244);
let __v228: u32 = *__slate_slot_244;
*__slate_slot_244 = __v228.wrapping_add(8);
if true {
continue '__loop_189;
}
}
}
} else {
if true {
break;
}
}
}
if ((BITS!(8i32, *__slate_slot_245) << 8i32) as u64).wrapping_add(*__slate_slot_245 >> 8i32) % 31 != 0 {
(*strm).msg = c"incorrect header check".as_ptr() as *mut u8;
(*(*__slate_slot_250)).mode = 27;
continue '__loop_194;
} else {
if BITS!(4i32, *__slate_slot_245) != 8 {
(*strm).msg = c"unknown compression method".as_ptr() as *mut u8;
(*(*__slate_slot_250)).mode = 27;
continue '__loop_194;
} else {
DROPBITS!(4i32, *__slate_slot_245, *__slate_slot_244);
*__slate_slot_237 = BITS!(4i32, *__slate_slot_245).wrapping_add(8);
if *__slate_slot_237 > (*(*__slate_slot_250)).wbits {
(*strm).msg = c"invalid window size".as_ptr() as *mut u8;
(*(*__slate_slot_250)).mode = 27;
continue '__loop_194;
} else {
(*(*__slate_slot_250)).dmax = 1u32 << *__slate_slot_237;
let __v226: u64 = zlib_adler32(0u64, core::ptr::null::<u8>(), 0u32);
(*(*__slate_slot_250)).check = __v226;
(*strm).adler = __v226;
(*(*__slate_slot_250)).mode = (if *__slate_slot_245 & 512 != 0 { 9i32 } else { 11i32 }) as u32;
INITBITS!(*__slate_slot_245, *__slate_slot_244);
continue '__loop_194;
}
}
}
}
} else {
if __t1 == 9 {
'__loop_178: loop {
if *__slate_slot_244 < 32 {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v225: u32 = *__slate_slot_247;
*__slate_slot_247 = __v225.wrapping_sub(1);
let __v223: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v223.wrapping_offset(1 as isize);
let __v220: u64 = *__slate_slot_245;
*__slate_slot_245 = __v220.wrapping_add((*__v223 as u64) << *__slate_slot_244);
let __v218: u32 = *__slate_slot_244;
*__slate_slot_244 = __v218.wrapping_add(8);
if true {
continue '__loop_178;
}
}
}
} else {
if true {
break;
}
}
}
let __v216: u64 = REVERSE!(*__slate_slot_245);
(*(*__slate_slot_250)).check = __v216;
(*strm).adler = __v216;
INITBITS!(*__slate_slot_245, *__slate_slot_244);
(*(*__slate_slot_250)).mode = 10;
break '__join_175;
} else {
if __t1 == 10 {
break '__join_175;
} else {
if __t1 == 11 {
break '__join_172;
} else {
if __t1 == 12 {
break '__join_170;
} else {
if __t1 == 13 {
BYTEBITS!(*__slate_slot_245, *__slate_slot_244); // go to byte boundary
'__loop_155: loop {
if *__slate_slot_244 < 32 {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v205: u32 = *__slate_slot_247;
*__slate_slot_247 = __v205.wrapping_sub(1);
let __v203: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v203.wrapping_offset(1 as isize);
let __v200: u64 = *__slate_slot_245;
*__slate_slot_245 = __v200.wrapping_add((*__v203 as u64) << *__slate_slot_244);
let __v198: u32 = *__slate_slot_244;
*__slate_slot_244 = __v198.wrapping_add(8);
if true {
continue '__loop_155;
}
}
}
} else {
if true {
break;
}
}
}
if *__slate_slot_245 & 65535 != *__slate_slot_245 >> 16i32 ^ 65535 {
(*strm).msg = c"invalid stored block lengths".as_ptr() as *mut u8;
(*(*__slate_slot_250)).mode = 27;
continue '__loop_194;
} else {
(*(*__slate_slot_250)).length = (*__slate_slot_245 as u32) & 65535;
INITBITS!(*__slate_slot_245, *__slate_slot_244);
(*(*__slate_slot_250)).mode = 14;
break '__join_150;
}
} else {
if __t1 == 14 {
break '__join_150;
} else {
if __t1 == 15 {
'__loop_140: loop {
if *__slate_slot_244 < 14 {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v185: u32 = *__slate_slot_247;
*__slate_slot_247 = __v185.wrapping_sub(1);
let __v183: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v183.wrapping_offset(1 as isize);
let __v180: u64 = *__slate_slot_245;
*__slate_slot_245 = __v180.wrapping_add((*__v183 as u64) << *__slate_slot_244);
let __v178: u32 = *__slate_slot_244;
*__slate_slot_244 = __v178.wrapping_add(8);
if true {
continue '__loop_140;
}
}
}
} else {
if true {
break;
}
}
}
(*(*__slate_slot_250)).nlen = BITS!(5i32, *__slate_slot_245).wrapping_add(257);
DROPBITS!(5i32, *__slate_slot_245, *__slate_slot_244);
(*(*__slate_slot_250)).ndist = BITS!(5i32, *__slate_slot_245).wrapping_add(1);
DROPBITS!(5i32, *__slate_slot_245, *__slate_slot_244);
(*(*__slate_slot_250)).ncode = BITS!(4i32, *__slate_slot_245).wrapping_add(4);
DROPBITS!(4i32, *__slate_slot_245, *__slate_slot_244);
if (*(*__slate_slot_250)).nlen > 286 || (*(*__slate_slot_250)).ndist > 30 {
(*strm).msg = c"too many length or distance symbols".as_ptr() as *mut u8;
(*(*__slate_slot_250)).mode = 27;
continue '__loop_194;
} else {
(*(*__slate_slot_250)).have = 0;
(*(*__slate_slot_250)).mode = 16;
}
} else {
if __t1 != 16 {
if __t1 == 17 {
break '__join_92;
} else {
if __t1 == 18 {
break '__join_85;
} else {
if __t1 == 19 {
break '__join_67;
} else {
if __t1 == 20 {
break '__join_60;
} else {
if __t1 == 21 {
break '__join_48;
} else {
if __t1 == 22 {
break '__join_39;
} else {
if __t1 == 23 {
if *__slate_slot_246 == 0 {
break '__join_8;
} else {
let __v35: *mut u8 = *__slate_slot_248;
*__slate_slot_248 = __v35.wrapping_offset(1 as isize);
*__v35 = (*(*__slate_slot_250)).length as u8;
let __v33: u32 = *__slate_slot_246;
*__slate_slot_246 = __v33.wrapping_sub(1);
(*(*__slate_slot_250)).mode = 18;
continue '__loop_194;
}
} else {
if __t1 == 24 {
if (*(*__slate_slot_250)).wrap != 0 {
'__loop_20: loop {
if *__slate_slot_244 < 32 {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v31: u32 = *__slate_slot_247;
*__slate_slot_247 = __v31.wrapping_sub(1);
let __v29: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v29.wrapping_offset(1 as isize);
let __v26: u64 = *__slate_slot_245;
*__slate_slot_245 = __v26.wrapping_add((*__v29 as u64) << *__slate_slot_244);
let __v24: u32 = *__slate_slot_244;
*__slate_slot_244 = __v24.wrapping_add(8);
if true {
continue '__loop_20;
}
}
}
} else {
if true {
break;
}
}
}
let __v22: u32 = *__slate_slot_242;
*__slate_slot_242 = __v22.wrapping_sub(*__slate_slot_246);
let __v19: u64 = (*strm).total_out;
(*strm).total_out = __v19.wrapping_add(*__slate_slot_242 as u64);
let __v17: *mut inflate_state = *__slate_slot_250;
let __v16: u64 = (*__v17).total;
(*__v17).total = __v16.wrapping_add(*__slate_slot_242 as u64);
if INFLATE_NEED_CHECKSUM!() != 0 && *__slate_slot_242 != 0 {
let __v14: u64 = UPDATE!(unsafe { (*(*__slate_slot_250)).check }, (*__slate_slot_248).wrapping_offset(-(*__slate_slot_242 as isize)), *__slate_slot_242);
(*(*__slate_slot_250)).check = __v14;
(*strm).adler = __v14;
}
*__slate_slot_242 = *__slate_slot_246;
if REVERSE!(*__slate_slot_245) != (*(*__slate_slot_250)).check {
(*strm).msg = c"incorrect data check".as_ptr() as *mut u8;
(*(*__slate_slot_250)).mode = 27;
continue '__loop_194;
} else {
break '__join_14;
}
} else {
break '__join_13;
}
} else {
if __t1 == 26 {
break '__join_12;
} else {
if __t1 == 27 {
break '__join_11;
} else {
if __t1 == 28 {
return 4i32.wrapping_neg();
} else {
if __t1 == 29 {
break '__loop_194;
} else {
break '__loop_194;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
while (*(*__slate_slot_250)).have < (*(*__slate_slot_250)).ncode {
'__loop_133: loop {
if *__slate_slot_244 < 3 {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v176: u32 = *__slate_slot_247;
*__slate_slot_247 = __v176.wrapping_sub(1);
let __v174: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v174.wrapping_offset(1 as isize);
let __v171: u64 = *__slate_slot_245;
*__slate_slot_245 = __v171.wrapping_add((*__v174 as u64) << *__slate_slot_244);
let __v169: u32 = *__slate_slot_244;
*__slate_slot_244 = __v169.wrapping_add(8);
if true {
continue '__loop_133;
}
}
}
} else {
if true {
break;
}
}
}
let __v167: *mut inflate_state = *__slate_slot_250;
let __v166: u32 = (*__v167).have;
(*__v167).have = __v166.wrapping_add(1);
*((*(*__slate_slot_250)).lens.as_mut_ptr() as *mut u16).wrapping_offset((order.0[__v166 as usize] as i32) as isize) = BITS!(3i32, *__slate_slot_245) as u16;
DROPBITS!(3i32, *__slate_slot_245, *__slate_slot_244);
}
while (*(*__slate_slot_250)).have < 19 {
let __v164: *mut inflate_state = *__slate_slot_250;
let __v163: u32 = (*__v164).have;
(*__v164).have = __v163.wrapping_add(1);
*((*(*__slate_slot_250)).lens.as_mut_ptr() as *mut u16).wrapping_offset((order.0[__v163 as usize] as i32) as isize) = 0;
}
(*(*__slate_slot_250)).next = (*(*__slate_slot_250)).codes.as_mut_ptr() as *mut __SlateRecord_inftrees_h_31;
(*(*__slate_slot_250)).lencode = (*(*__slate_slot_250)).next as *const __SlateRecord_inftrees_h_31;
(*(*__slate_slot_250)).lenbits = 7;
*__slate_slot_236 = zlib_inflate_table(0u32, (*(*__slate_slot_250)).lens.as_mut_ptr() as *mut u16, 19u32, core::ptr::addr_of_mut!((*(*__slate_slot_250)).next), core::ptr::addr_of_mut!((*(*__slate_slot_250)).lenbits), (*(*__slate_slot_250)).work.as_mut_ptr() as *mut u16);
if *__slate_slot_236 != 0 {
(*strm).msg = c"invalid code lengths set".as_ptr() as *mut u8;
(*(*__slate_slot_250)).mode = 27;
continue '__loop_194;
} else {
(*(*__slate_slot_250)).have = 0;
(*(*__slate_slot_250)).mode = 17;
}
}
'__join_91: {
'__join_103: {
'__loop_92: loop {
if (*(*__slate_slot_250)).have < unsafe { (*(*__slate_slot_250)).nlen }.wrapping_add((*(*__slate_slot_250)).ndist) {
'__loop_124: loop {
*__slate_slot_239 = *unsafe { (*(*__slate_slot_250)).lencode }.wrapping_offset(BITS!(unsafe { (*(*__slate_slot_250)).lenbits }, *__slate_slot_245) as isize);
if ((*__slate_slot_239).bits as u32) <= *__slate_slot_244 {
break;
} else {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v161: u32 = *__slate_slot_247;
*__slate_slot_247 = __v161.wrapping_sub(1);
let __v159: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v159.wrapping_offset(1 as isize);
let __v156: u64 = *__slate_slot_245;
*__slate_slot_245 = __v156.wrapping_add((*__v159 as u64) << *__slate_slot_244);
let __v154: u32 = *__slate_slot_244;
*__slate_slot_244 = __v154.wrapping_add(8);
if true {
continue '__loop_124;
}
}
}
}
}
if ((*__slate_slot_239).val as i32) < 16 {
'__loop_95: loop {
if *__slate_slot_244 < ((*__slate_slot_239).bits as u32) {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v120: u32 = *__slate_slot_247;
*__slate_slot_247 = __v120.wrapping_sub(1);
let __v118: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v118.wrapping_offset(1 as isize);
let __v115: u64 = *__slate_slot_245;
*__slate_slot_245 = __v115.wrapping_add((*__v118 as u64) << *__slate_slot_244);
let __v113: u32 = *__slate_slot_244;
*__slate_slot_244 = __v113.wrapping_add(8);
if true {
continue '__loop_95;
}
}
}
} else {
if true {
break;
}
}
}
DROPBITS__2!((*__slate_slot_239).bits, *__slate_slot_245, *__slate_slot_244);
let __v111: *mut inflate_state = *__slate_slot_250;
let __v110: u32 = (*__v111).have;
(*__v111).have = __v110.wrapping_add(1);
*((*(*__slate_slot_250)).lens.as_mut_ptr() as *mut u16).wrapping_offset(__v110 as isize) = (*__slate_slot_239).val;
} else {
if ((*__slate_slot_239).val as i32) == 16 {
'__loop_106: loop {
if *__slate_slot_244 < (((*__slate_slot_239).bits as i32).wrapping_add(2) as u32) {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v134: u32 = *__slate_slot_247;
*__slate_slot_247 = __v134.wrapping_sub(1);
let __v132: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v132.wrapping_offset(1 as isize);
let __v129: u64 = *__slate_slot_245;
*__slate_slot_245 = __v129.wrapping_add((*__v132 as u64) << *__slate_slot_244);
let __v127: u32 = *__slate_slot_244;
*__slate_slot_244 = __v127.wrapping_add(8);
if true {
continue '__loop_106;
}
}
}
} else {
if true {
break;
}
}
}
DROPBITS__2!((*__slate_slot_239).bits, *__slate_slot_245, *__slate_slot_244);
if (*(*__slate_slot_250)).have == 0 {
break '__join_103;
} else {
*__slate_slot_237 = *((*(*__slate_slot_250)).lens.as_mut_ptr() as *mut u16).wrapping_offset(unsafe { (*(*__slate_slot_250)).have }.wrapping_sub(1) as isize) as u32;
*__slate_slot_241 = 3u32.wrapping_add(BITS!(2i32, *__slate_slot_245));
DROPBITS!(2i32, *__slate_slot_245, *__slate_slot_244);
}
} else {
if ((*__slate_slot_239).val as i32) == 17 {
'__loop_111: loop {
if *__slate_slot_244 < (((*__slate_slot_239).bits as i32).wrapping_add(3) as u32) {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v143: u32 = *__slate_slot_247;
*__slate_slot_247 = __v143.wrapping_sub(1);
let __v141: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v141.wrapping_offset(1 as isize);
let __v138: u64 = *__slate_slot_245;
*__slate_slot_245 = __v138.wrapping_add((*__v141 as u64) << *__slate_slot_244);
let __v136: u32 = *__slate_slot_244;
*__slate_slot_244 = __v136.wrapping_add(8);
if true {
continue '__loop_111;
}
}
}
} else {
if true {
break;
}
}
}
DROPBITS__2!((*__slate_slot_239).bits, *__slate_slot_245, *__slate_slot_244);
*__slate_slot_237 = 0;
*__slate_slot_241 = 3u32.wrapping_add(BITS!(3i32, *__slate_slot_245));
DROPBITS!(3i32, *__slate_slot_245, *__slate_slot_244);
} else {
'__loop_116: loop {
if *__slate_slot_244 < (((*__slate_slot_239).bits as i32).wrapping_add(7) as u32) {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v152: u32 = *__slate_slot_247;
*__slate_slot_247 = __v152.wrapping_sub(1);
let __v150: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v150.wrapping_offset(1 as isize);
let __v147: u64 = *__slate_slot_245;
*__slate_slot_245 = __v147.wrapping_add((*__v150 as u64) << *__slate_slot_244);
let __v145: u32 = *__slate_slot_244;
*__slate_slot_244 = __v145.wrapping_add(8);
if true {
continue '__loop_116;
}
}
}
} else {
if true {
break;
}
}
}
DROPBITS__2!((*__slate_slot_239).bits, *__slate_slot_245, *__slate_slot_244);
*__slate_slot_237 = 0;
*__slate_slot_241 = 11u32.wrapping_add(BITS!(7i32, *__slate_slot_245));
DROPBITS!(7i32, *__slate_slot_245, *__slate_slot_244);
}
}
if unsafe { (*(*__slate_slot_250)).have }.wrapping_add(*__slate_slot_241) > unsafe { (*(*__slate_slot_250)).nlen }.wrapping_add((*(*__slate_slot_250)).ndist) {
break;
} else {
loop {
let __v122: u32 = *__slate_slot_241;
*__slate_slot_241 = __v122.wrapping_sub(1);
if __v122 != 0 {
let __v125: *mut inflate_state = *__slate_slot_250;
let __v124: u32 = (*__v125).have;
(*__v125).have = __v124.wrapping_add(1);
*((*(*__slate_slot_250)).lens.as_mut_ptr() as *mut u16).wrapping_offset(__v124 as isize) = *__slate_slot_237 as u16;
} else {
continue '__loop_92;
}
}
}
}
} else {
break '__join_91;
}
}
(*strm).msg = c"invalid bit length repeat".as_ptr() as *mut u8;
(*(*__slate_slot_250)).mode = 27;
break '__join_91;
}
(*strm).msg = c"invalid bit length repeat".as_ptr() as *mut u8;
(*(*__slate_slot_250)).mode = 27;
}
// handle error breaks in while
if ((*(*__slate_slot_250)).mode as u32) == 27 {
continue '__loop_194;
} else {
// build code tables
(*(*__slate_slot_250)).next = (*(*__slate_slot_250)).codes.as_mut_ptr() as *mut __SlateRecord_inftrees_h_31;
(*(*__slate_slot_250)).lencode = (*(*__slate_slot_250)).next as *const __SlateRecord_inftrees_h_31;
(*(*__slate_slot_250)).lenbits = 9;
*__slate_slot_236 = zlib_inflate_table(1u32, (*(*__slate_slot_250)).lens.as_mut_ptr() as *mut u16, (*(*__slate_slot_250)).nlen, core::ptr::addr_of_mut!((*(*__slate_slot_250)).next), core::ptr::addr_of_mut!((*(*__slate_slot_250)).lenbits), (*(*__slate_slot_250)).work.as_mut_ptr() as *mut u16);
if *__slate_slot_236 != 0 {
(*strm).msg = c"invalid literal/lengths set".as_ptr() as *mut u8;
(*(*__slate_slot_250)).mode = 27;
continue '__loop_194;
} else {
(*(*__slate_slot_250)).distcode = (*(*__slate_slot_250)).next as *const __SlateRecord_inftrees_h_31;
(*(*__slate_slot_250)).distbits = 6;
*__slate_slot_236 = zlib_inflate_table(2u32, ((*(*__slate_slot_250)).lens.as_mut_ptr() as *mut u16).wrapping_offset((*(*__slate_slot_250)).nlen as isize), (*(*__slate_slot_250)).ndist, core::ptr::addr_of_mut!((*(*__slate_slot_250)).next), core::ptr::addr_of_mut!((*(*__slate_slot_250)).distbits), (*(*__slate_slot_250)).work.as_mut_ptr() as *mut u16);
if *__slate_slot_236 != 0 {
(*strm).msg = c"invalid distances set".as_ptr() as *mut u8;
(*(*__slate_slot_250)).mode = 27;
continue '__loop_194;
} else {
(*(*__slate_slot_250)).mode = 18;
}
}
}
}
if *__slate_slot_247 >= 6 && *__slate_slot_246 >= 258 {
RESTORE!(strm, *__slate_slot_248, *__slate_slot_246, *__slate_slot_249, *__slate_slot_247, *__slate_slot_250, *__slate_slot_245, *__slate_slot_244);
inflate_fast(strm, *__slate_slot_242);
LOAD!(*__slate_slot_248, strm, *__slate_slot_246, *__slate_slot_249, *__slate_slot_247, *__slate_slot_245, *__slate_slot_250, *__slate_slot_244);
continue '__loop_194;
} else {
'__loop_83: loop {
*__slate_slot_239 = *unsafe { (*(*__slate_slot_250)).lencode }.wrapping_offset(BITS!(unsafe { (*(*__slate_slot_250)).lenbits }, *__slate_slot_245) as isize);
if ((*__slate_slot_239).bits as u32) <= *__slate_slot_244 {
break;
} else {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v108: u32 = *__slate_slot_247;
*__slate_slot_247 = __v108.wrapping_sub(1);
let __v106: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v106.wrapping_offset(1 as isize);
let __v103: u64 = *__slate_slot_245;
*__slate_slot_245 = __v103.wrapping_add((*__v106 as u64) << *__slate_slot_244);
let __v101: u32 = *__slate_slot_244;
*__slate_slot_244 = __v101.wrapping_add(8);
if true {
continue '__loop_83;
}
}
}
}
}
if (*__slate_slot_239).op != 0 && ((*__slate_slot_239).op as i32) & 240 == 0 {
*__slate_slot_238 = *__slate_slot_239;
'__loop_78: loop {
*__slate_slot_239 = *unsafe { (*(*__slate_slot_250)).lencode }.wrapping_offset((((*__slate_slot_238).val as i32) as u32).wrapping_add(BITS!(((*__slate_slot_238).bits as i32).wrapping_add((*__slate_slot_238).op as i32), *__slate_slot_245) >> ((*__slate_slot_238).bits as i32)) as isize);
if (((*__slate_slot_238).bits as i32).wrapping_add((*__slate_slot_239).bits as i32) as u32) <= *__slate_slot_244 {
break;
} else {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v99: u32 = *__slate_slot_247;
*__slate_slot_247 = __v99.wrapping_sub(1);
let __v97: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v97.wrapping_offset(1 as isize);
let __v94: u64 = *__slate_slot_245;
*__slate_slot_245 = __v94.wrapping_add((*__v97 as u64) << *__slate_slot_244);
let __v92: u32 = *__slate_slot_244;
*__slate_slot_244 = __v92.wrapping_add(8);
if true {
continue '__loop_78;
}
}
}
}
}
DROPBITS__2!((*__slate_slot_238).bits, *__slate_slot_245, *__slate_slot_244);
}
DROPBITS__2!((*__slate_slot_239).bits, *__slate_slot_245, *__slate_slot_244);
(*(*__slate_slot_250)).length = (*__slate_slot_239).val as u32;
if ((*__slate_slot_239).op as i32) == 0 {
(*(*__slate_slot_250)).mode = 23;
continue '__loop_194;
} else {
if ((*__slate_slot_239).op as i32) & 32 != 0 {
(*(*__slate_slot_250)).mode = 11;
continue '__loop_194;
} else {
if ((*__slate_slot_239).op as i32) & 64 != 0 {
(*strm).msg = c"invalid literal/length code".as_ptr() as *mut u8;
(*(*__slate_slot_250)).mode = 27;
continue '__loop_194;
} else {
(*(*__slate_slot_250)).extra = ((*__slate_slot_239).op as u32) & 15;
(*(*__slate_slot_250)).mode = 19;
}
}
}
}
}
if (*(*__slate_slot_250)).extra != 0 {
'__loop_64: loop {
if *__slate_slot_244 < ((*(*__slate_slot_250)).extra as u32) {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v90: u32 = *__slate_slot_247;
*__slate_slot_247 = __v90.wrapping_sub(1);
let __v88: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v88.wrapping_offset(1 as isize);
let __v85: u64 = *__slate_slot_245;
*__slate_slot_245 = __v85.wrapping_add((*__v88 as u64) << *__slate_slot_244);
let __v83: u32 = *__slate_slot_244;
*__slate_slot_244 = __v83.wrapping_add(8);
if true {
continue '__loop_64;
}
}
}
} else {
if true {
break;
}
}
}
let __v81: *mut inflate_state = *__slate_slot_250;
let __v80: u32 = (*__v81).length;
let __v79: u32 = __v80.wrapping_add(BITS!(unsafe { (*(*__slate_slot_250)).extra }, *__slate_slot_245));
(*__v81).length = __v79;
DROPBITS!(unsafe { (*(*__slate_slot_250)).extra }, *__slate_slot_245, *__slate_slot_244);
}
(*(*__slate_slot_250)).mode = 20;
}
'__loop_60: loop {
*__slate_slot_239 = *unsafe { (*(*__slate_slot_250)).distcode }.wrapping_offset(BITS!(unsafe { (*(*__slate_slot_250)).distbits }, *__slate_slot_245) as isize);
if ((*__slate_slot_239).bits as u32) <= *__slate_slot_244 {
break;
} else {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v78: u32 = *__slate_slot_247;
*__slate_slot_247 = __v78.wrapping_sub(1);
let __v76: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v76.wrapping_offset(1 as isize);
let __v73: u64 = *__slate_slot_245;
*__slate_slot_245 = __v73.wrapping_add((*__v76 as u64) << *__slate_slot_244);
let __v71: u32 = *__slate_slot_244;
*__slate_slot_244 = __v71.wrapping_add(8);
if true {
continue '__loop_60;
}
}
}
}
}
if ((*__slate_slot_239).op as i32) & 240 == 0 {
*__slate_slot_238 = *__slate_slot_239;
'__loop_55: loop {
*__slate_slot_239 = *unsafe { (*(*__slate_slot_250)).distcode }.wrapping_offset((((*__slate_slot_238).val as i32) as u32).wrapping_add(BITS!(((*__slate_slot_238).bits as i32).wrapping_add((*__slate_slot_238).op as i32), *__slate_slot_245) >> ((*__slate_slot_238).bits as i32)) as isize);
if (((*__slate_slot_238).bits as i32).wrapping_add((*__slate_slot_239).bits as i32) as u32) <= *__slate_slot_244 {
break;
} else {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v69: u32 = *__slate_slot_247;
*__slate_slot_247 = __v69.wrapping_sub(1);
let __v67: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v67.wrapping_offset(1 as isize);
let __v64: u64 = *__slate_slot_245;
*__slate_slot_245 = __v64.wrapping_add((*__v67 as u64) << *__slate_slot_244);
let __v62: u32 = *__slate_slot_244;
*__slate_slot_244 = __v62.wrapping_add(8);
if true {
continue '__loop_55;
}
}
}
}
}
DROPBITS__2!((*__slate_slot_238).bits, *__slate_slot_245, *__slate_slot_244);
}
DROPBITS__2!((*__slate_slot_239).bits, *__slate_slot_245, *__slate_slot_244);
if ((*__slate_slot_239).op as i32) & 64 != 0 {
(*strm).msg = c"invalid distance code".as_ptr() as *mut u8;
(*(*__slate_slot_250)).mode = 27;
continue '__loop_194;
} else {
(*(*__slate_slot_250)).offset = (*__slate_slot_239).val as u32;
(*(*__slate_slot_250)).extra = ((*__slate_slot_239).op as u32) & 15;
(*(*__slate_slot_250)).mode = 21;
}
}
if (*(*__slate_slot_250)).extra != 0 {
'__loop_45: loop {
if *__slate_slot_244 < ((*(*__slate_slot_250)).extra as u32) {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v60: u32 = *__slate_slot_247;
*__slate_slot_247 = __v60.wrapping_sub(1);
let __v58: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v58.wrapping_offset(1 as isize);
let __v55: u64 = *__slate_slot_245;
*__slate_slot_245 = __v55.wrapping_add((*__v58 as u64) << *__slate_slot_244);
let __v53: u32 = *__slate_slot_244;
*__slate_slot_244 = __v53.wrapping_add(8);
if true {
continue '__loop_45;
}
}
}
} else {
if true {
break;
}
}
}
let __v51: *mut inflate_state = *__slate_slot_250;
let __v50: u32 = (*__v51).offset;
let __v49: u32 = __v50.wrapping_add(BITS!(unsafe { (*(*__slate_slot_250)).extra }, *__slate_slot_245));
(*__v51).offset = __v49;
DROPBITS!(unsafe { (*(*__slate_slot_250)).extra }, *__slate_slot_245, *__slate_slot_244);
}
if (*(*__slate_slot_250)).offset > unsafe { (*(*__slate_slot_250)).whave }.wrapping_add(*__slate_slot_242).wrapping_sub(*__slate_slot_246) {
(*strm).msg = c"invalid distance too far back".as_ptr() as *mut u8;
(*(*__slate_slot_250)).mode = 27;
continue '__loop_194;
} else {
(*(*__slate_slot_250)).mode = 22;
}
}
if *__slate_slot_246 == 0 {
break '__join_8;
} else {
*__slate_slot_241 = (*__slate_slot_242).wrapping_sub(*__slate_slot_246);
if (*(*__slate_slot_250)).offset > *__slate_slot_241 {
// copy from window
*__slate_slot_241 = unsafe { (*(*__slate_slot_250)).offset }.wrapping_sub(*__slate_slot_241);
if *__slate_slot_241 > (*(*__slate_slot_250)).write {
let __v48: u32 = *__slate_slot_241;
*__slate_slot_241 = __v48.wrapping_sub((*(*__slate_slot_250)).write);
*__slate_slot_240 = unsafe { (*(*__slate_slot_250)).window }.wrapping_offset(unsafe { (*(*__slate_slot_250)).wsize }.wrapping_sub(*__slate_slot_241) as isize);
} else {
*__slate_slot_240 = unsafe { (*(*__slate_slot_250)).window }.wrapping_offset(unsafe { (*(*__slate_slot_250)).write }.wrapping_sub(*__slate_slot_241) as isize);
}
if *__slate_slot_241 > (*(*__slate_slot_250)).length {
*__slate_slot_241 = (*(*__slate_slot_250)).length;
}
} else {
// copy from output
*__slate_slot_240 = (*__slate_slot_248).wrapping_offset(-((*(*__slate_slot_250)).offset as isize));
*__slate_slot_241 = (*(*__slate_slot_250)).length;
}
if *__slate_slot_241 > *__slate_slot_246 {
*__slate_slot_241 = *__slate_slot_246;
}
let __v46: u32 = *__slate_slot_246;
*__slate_slot_246 = __v46.wrapping_sub(*__slate_slot_241);
let __v44: *mut inflate_state = *__slate_slot_250;
let __v43: u32 = (*__v44).length;
(*__v44).length = __v43.wrapping_sub(*__slate_slot_241);
loop {
let __v41: *mut u8 = *__slate_slot_240;
*__slate_slot_240 = __v41.wrapping_offset(1 as isize);
let __v39: *mut u8 = *__slate_slot_248;
*__slate_slot_248 = __v39.wrapping_offset(1 as isize);
*__v39 = *__v41;
let __v37: u32 = *__slate_slot_241;
let __v36: u32 = __v37.wrapping_sub(1);
*__slate_slot_241 = __v36;
if __v36 == 0 {
break;
}
}
if (*(*__slate_slot_250)).length == 0 {
(*(*__slate_slot_250)).mode = 18;
continue '__loop_194;
} else {
continue '__loop_194;
}
}
}
*__slate_slot_241 = (*(*__slate_slot_250)).length;
if *__slate_slot_241 != 0 {
if *__slate_slot_241 > *__slate_slot_247 {
*__slate_slot_241 = *__slate_slot_247;
}
if *__slate_slot_241 > *__slate_slot_246 {
*__slate_slot_241 = *__slate_slot_246;
}
if *__slate_slot_241 == 0 {
break '__join_8;
} else {
memcpy(*__slate_slot_248 as *mut (), *__slate_slot_249 as *const (), *__slate_slot_241 as u64);
let __v196: u32 = *__slate_slot_247;
*__slate_slot_247 = __v196.wrapping_sub(*__slate_slot_241);
let __v194: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v194.wrapping_offset(*__slate_slot_241 as isize);
let __v192: u32 = *__slate_slot_246;
*__slate_slot_246 = __v192.wrapping_sub(*__slate_slot_241);
let __v190: *mut u8 = *__slate_slot_248;
*__slate_slot_248 = __v190.wrapping_offset(*__slate_slot_241 as isize);
let __v188: *mut inflate_state = *__slate_slot_250;
let __v187: u32 = (*__v188).length;
(*__v188).length = __v187.wrapping_sub(*__slate_slot_241);
continue '__loop_194;
}
} else {
(*(*__slate_slot_250)).mode = 11;
continue '__loop_194;
}
}
if (*(*__slate_slot_250)).havedict == 0 {
RESTORE!(strm, *__slate_slot_248, *__slate_slot_246, *__slate_slot_249, *__slate_slot_247, *__slate_slot_250, *__slate_slot_245, *__slate_slot_244);
return 2;
} else {
let __v215: u64 = zlib_adler32(0u64, core::ptr::null::<u8>(), 0u32);
(*(*__slate_slot_250)).check = __v215;
(*strm).adler = __v215;
(*(*__slate_slot_250)).mode = 11;
}
}
if flush == 6 {
break '__join_8;
}
}
INFLATE_TYPEDO_HOOK!();
if (*(*__slate_slot_250)).last != 0 {
BYTEBITS!(*__slate_slot_245, *__slate_slot_244);
(*(*__slate_slot_250)).mode = 24;
} else {
'__loop_166: loop {
if *__slate_slot_244 < 3 {
loop {
if *__slate_slot_247 == 0 {
break '__join_8;
} else {
let __v214: u32 = *__slate_slot_247;
*__slate_slot_247 = __v214.wrapping_sub(1);
let __v212: *const u8 = *__slate_slot_249;
*__slate_slot_249 = __v212.wrapping_offset(1 as isize);
let __v209: u64 = *__slate_slot_245;
*__slate_slot_245 = __v209.wrapping_add((*__v212 as u64) << *__slate_slot_244);
let __v207: u32 = *__slate_slot_244;
*__slate_slot_244 = __v207.wrapping_add(8);
if true {
continue '__loop_166;
}
}
}
} else {
if true {
break;
}
}
}
(*(*__slate_slot_250)).last = BITS!(1i32, *__slate_slot_245) as i32;
DROPBITS!(1i32, *__slate_slot_245, *__slate_slot_244);
let __t0: u32 = BITS!(2i32, *__slate_slot_245);
if __t0 == 0 {
(*(*__slate_slot_250)).mode = 13;
// stored block
} else {
if __t0 == 1 {
zlib_fixedtables(*__slate_slot_250); // fixed block
(*(*__slate_slot_250)).mode = 18;
// decode codes
} else {
if __t0 == 2 {
(*(*__slate_slot_250)).mode = 15;
// dynamic block
} else {
if __t0 == 3 {
(*strm).msg = c"invalid block type".as_ptr() as *mut u8;
(*(*__slate_slot_250)).mode = 27;
}
}
}
}
DROPBITS!(2i32, *__slate_slot_245, *__slate_slot_244);
}
}
return 2i32.wrapping_neg();
}
*__slate_slot_236 = 3i32.wrapping_neg();
break '__join_8;
}
INITBITS!(*__slate_slot_245, *__slate_slot_244);
}
(*(*__slate_slot_250)).mode = 26;
}
*__slate_slot_236 = 1;
}
RESTORE!(strm, *__slate_slot_248, *__slate_slot_246, *__slate_slot_249, *__slate_slot_247, *__slate_slot_250, *__slate_slot_245, *__slate_slot_244);
if INFLATE_NEED_UPDATEWINDOW!() != 0 && ((*(*__slate_slot_250)).wsize != 0 || ((*(*__slate_slot_250)).mode as u32) < 24 && (*__slate_slot_242 as u64) != (*strm).avail_out) {
zlib_updatewindow(strm, *__slate_slot_242);
}
*__slate_slot_243 = (*__slate_slot_243 as u64).wrapping_sub((*strm).avail_in) as u32;
*__slate_slot_242 = (*__slate_slot_242 as u64).wrapping_sub((*strm).avail_out) as u32;
let __v8: u64 = (*strm).total_in;
(*strm).total_in = __v8.wrapping_add(*__slate_slot_243 as u64);
let __v5: u64 = (*strm).total_out;
(*strm).total_out = __v5.wrapping_add(*__slate_slot_242 as u64);
let __v3: *mut inflate_state = *__slate_slot_250;
let __v2: u64 = (*__v3).total;
(*__v3).total = __v2.wrapping_add(*__slate_slot_242 as u64);
if INFLATE_NEED_CHECKSUM!() != 0 && (*(*__slate_slot_250)).wrap != 0 && *__slate_slot_242 != 0 {
let __v0: u64 = UPDATE!(unsafe { (*(*__slate_slot_250)).check }, unsafe { (*strm).next_out }.wrapping_offset(-(*__slate_slot_242 as isize)), *__slate_slot_242);
(*(*__slate_slot_250)).check = __v0;
(*strm).adler = __v0;
}
(*strm).data_type = unsafe { (*(*__slate_slot_250)).bits }.wrapping_add((if (*(*__slate_slot_250)).last != 0 { 64i32 } else { 0i32 }) as u32).wrapping_add((if ((*(*__slate_slot_250)).mode as u32) == 11 { 128i32 } else { 0i32 }) as u32) as i32;
if flush == 2 && *__slate_slot_236 == 0 && (*strm).avail_out != 0 && (*strm).avail_in == 0 {
return zlib_inflateSyncPacket(strm);
} else {
if (*__slate_slot_243 == 0 && *__slate_slot_242 == 0 || flush == 5) && *__slate_slot_236 == 0 {
*__slate_slot_236 = 5i32.wrapping_neg();
}
return *__slate_slot_236;
}
}
}
// Return from inflate(), updating the total counts and the check value.
// If there was no progress during the inflate() call, return a buffer
// error.  Call zlib_updatewindow() to create and/or update the window state.
}

static order: __SlateAlign16<[u16; 19]> = __SlateAlign16([16u16, 17u16, 18u16, 0u16, 8u16, 7u16, 9u16, 6u16, 10u16, 5u16, 11u16, 4u16, 12u16, 3u16, 13u16, 2u16, 14u16, 1u16, 15u16]);

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.inflate.zlib_inflateEnd")]
extern "C" fn zlib_inflateEnd(strm: *mut z_stream_s) -> i32 {
if strm.is_null() || unsafe { (*strm).state }.is_null() {
return 2i32.wrapping_neg();
}
0
}

/// This subroutine adds the data at next_in/avail_in to the output history
/// without performing any output.  The output buffer must be "caught up";
/// i.e. no pending output but this should always be the case. The state must
/// be waiting on the start of a block (i.e. mode == TYPE or HEAD).  On exit,
/// the output will also be caught up, and the checksum will have been updated
/// if need be.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.inflate.zlib_inflateIncomp")]
extern "C" fn zlib_inflateIncomp(z: *mut z_stream_s) -> i32 {
let state: *mut inflate_state = (unsafe { (*z).state }) as *mut inflate_state;
let saved_no: *mut u8 = unsafe { (*z).next_out };
let saved_ao: u32 = (unsafe { (*z).avail_out }) as u32;
if ((unsafe { (*state).mode }) as u32) != 11 && ((unsafe { (*state).mode }) as u32) != 0 {
return 3i32.wrapping_neg();
}
// Setup some variables to allow misuse of updateWindow
unsafe {
(*z).avail_out = 0;
}
unsafe {
(*z).next_out = ((*z).next_in as *mut u8).wrapping_offset((*z).avail_in as isize);
}
zlib_updatewindow(z, (unsafe { (*z).avail_in }) as u32);
// Restore saved variables
unsafe {
(*z).avail_out = saved_ao as u64;
}
unsafe {
(*z).next_out = saved_no;
}
let __v0: u64 = UPDATE__2!(unsafe { (*state).check }, unsafe { (*z).next_in }, unsafe { (*z).avail_in });
unsafe {
(*state).check = __v0;
}
unsafe {
(*z).adler = __v0;
}
let __v2: u64 = unsafe { (*z).total_out };
unsafe {
(*z).total_out = __v2.wrapping_add((*z).avail_in);
}
let __v5: u64 = unsafe { (*z).total_in };
unsafe {
(*z).total_in = __v5.wrapping_add((*z).avail_in);
}
let __v8: *const u8 = unsafe { (*z).next_in };
unsafe {
(*z).next_in = __v8.wrapping_offset((*z).avail_in as isize);
}
let __v11: u64 = unsafe { (*state).total };
unsafe {
(*state).total = __v11.wrapping_add((*z).avail_in);
}
unsafe {
(*z).avail_in = 0;
}
0
}
