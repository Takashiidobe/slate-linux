
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0-or-later
//!
//! x86 instruction analysis
//!
//! Copyright (C) IBM Corporation, 2002, 2004, 2009
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
//! BUILD_BUG_ON_MSG - break compile if a condition is true & emit supplied
//! 		      error message.
//! @cond: the condition which the compiler should know is false.
//! @msg: build-time error message
//!
//! See BUILD_BUG_ON for description.
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
//! This counts to 15. Any more, it will return 16th argument.
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
//! Invalid argument
//! No data available
//! IS_ERR_VALUE - Detect an error pointer.
//! @x: The pointer to check.
//!
//! Like IS_ERR(), but does not generate a compiler warning if result is unused.
//! Legacy last prefixes
//! ud2 ; .ascii "xen"
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
// BUILD_BUG - break compile if used.
//
// If you have some code that you expect the compiler to eliminate at
// build time, you should use BUILD_BUG to detect if it is
// unexpectedly used.
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
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// Value too large for defined data type
// __put_unaligned_t - write an unaligned value to memory.
// @type:	the type of the value to store.
// @val:	the value to store.
// @ptr:	the pointer to store to.
//
// Use memcpy to affect an unaligned type sized store avoiding undefined
// behavior from approaches like type punning that require -fno-strict-aliasing
// in order to be correct. The void* cast silences ubsan warnings.
// ud2 ; .ascii "kvm"
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
/// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
/// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
/// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
macro_rules! IS_ENABLED__2 {
() => {
__or!()
};
}

/// VEX bit fields
/// EVEX Byte1
macro_rules! X86_EVEX_M {
($vex:expr) => {
(($vex) as i32) & 7
};
}

macro_rules! X86_MODRM_MOD__2 {
($modrm:expr) => {
((($modrm) as i32) & 192) >> 6i32
};
}

macro_rules! X86_MODRM_MOD {
($modrm:expr) => {
(($modrm) & 192) >> 6i32
};
}

macro_rules! X86_MODRM_REG {
($modrm:expr) => {
((($modrm) as i32) & 56) >> 3i32
};
}

macro_rules! X86_MODRM_RM {
($modrm:expr) => {
($modrm) & 7
};
}

macro_rules! X86_MODRM_RM__2 {
($modrm:expr) => {
(($modrm) as i32) & 7
};
}

/// REX2 M0
macro_rules! X86_REX2_M {
($rex:expr) => {
(($rex) as i32) & 128
};
}

/// REX or REX2 W
macro_rules! X86_REX_W {
($rex:expr) => {
(($rex) as i32) & 8
};
}

macro_rules! X86_SIB_BASE {
($sib:expr) => {
($sib) & 7
};
}

/// VEX3 Byte1
macro_rules! X86_VEX3_M {
($vex:expr) => {
(($vex) as i32) & 31
};
}

/// VEX3 Byte2, VEX2 Byte1
macro_rules! X86_VEX_P {
($vex:expr) => {
(($vex) as i32) & 3
};
}

/// VEX bit flags
/// VEX3 Byte2
macro_rules! X86_VEX_W {
($vex:expr) => {
(($vex) as i32) & 128
};
}

/// XOP Byte2
macro_rules! X86_XOP_M {
($xop:expr) => {
(($xop) as i32) & 31
};
}

/// XOP Byte3
macro_rules! X86_XOP_P {
($xop:expr) => {
(($xop) as i32) & 3
};
}

macro_rules! ____or {
() => {
1i32
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
($ptr:expr) => {
{
let mut __v: i32;
core::ptr::null_mut::<i32>();
let mut __v3: i32 = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), (($ptr) as *mut ()) as *const (), core::mem::size_of::<i32>() as u64) };
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
macro_rules! __get_unaligned_t__2 {
($ptr:expr) => {
{
let mut __v: i16;
core::ptr::null_mut::<i16>();
let mut __v3: i16 = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), (($ptr) as *mut ()) as *const (), core::mem::size_of::<i16>() as u64) };
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
($ptr:expr) => {
{
let mut __v: u16;
core::ptr::null_mut::<u16>();
let mut __v3: u16 = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), (($ptr) as *mut ()) as *const (), core::mem::size_of::<u16>() as u64) };
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
macro_rules! __get_unaligned_t__4 {
($ptr:expr) => {
{
let mut __v: i8;
core::ptr::null_mut::<i8>();
let mut __v3: i8 = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), (($ptr) as *mut ()) as *const (), core::mem::size_of::<i8>() as u64) };
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
($ptr:expr) => {
{
let mut __v: u8;
core::ptr::null_mut::<u8>();
let mut __v3: u8 = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), (($ptr) as *mut ()) as *const (), core::mem::size_of::<u8>() as u64) };
__v = __v3;
__v
}
};
}

macro_rules! __or {
() => {
____or!()
};
}

macro_rules! get_unaligned__4 {
($ptr:expr) => {
__get_unaligned_t__4!($ptr)
};
}

macro_rules! get_unaligned__3 {
($ptr:expr) => {
__get_unaligned_t__2!($ptr)
};
}

macro_rules! get_unaligned__5 {
($ptr:expr) => {
__get_unaligned_t__3!($ptr)
};
}

macro_rules! get_unaligned__2 {
($ptr:expr) => {
__get_unaligned_t!($ptr)
};
}

macro_rules! get_unaligned {
($ptr:expr) => {
__get_unaligned_t__5!($ptr)
};
}

macro_rules! unlikely {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely__2 {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! unlikely__9 {
($x:expr) => {
(($x) != 0) as i64
};
}

/// Verify next sizeof(t) bytes can be on the same instruction
macro_rules! validate_next {
($t:ty, $insn:expr, $n:expr) => {
unsafe { (*$insn).next_byte }.wrapping_offset(core::mem::size_of::<$t>() as isize).wrapping_offset(($n) as isize) <= unsafe { (*$insn).end_kaddr }
};
}

unsafe extern "C" {
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn inat_get_opcode_attribute(arg0: u8) -> u32;
    fn inat_get_last_prefix_id(arg0: u8) -> i32;
    fn inat_get_escape_attribute(arg0: u8, arg1: i32, arg2: u32) -> u32;
    fn inat_get_group_attribute(arg0: u8, arg1: i32, arg2: u32) -> u32;
    fn inat_get_avx_attribute(arg0: u8, arg1: u8, arg2: u8) -> u32;
    fn inat_get_xop_attribute(arg0: u8, arg1: u8) -> u32;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn __compiletime_assert_5();
    fn __compiletime_assert_6();
    fn __compiletime_assert_7();
    fn __compiletime_assert_8();
    fn __compiletime_assert_9();
    fn __compiletime_assert_10();
    fn __compiletime_assert_11();
    fn __compiletime_assert_12();
    fn __compiletime_assert_13();
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn __compiletime_assert_16();
    fn __compiletime_assert_17();
    fn __compiletime_assert_18();
    fn __compiletime_assert_19();
    fn __compiletime_assert_20();
    fn __compiletime_assert_21();
    fn __compiletime_assert_22();
    fn __compiletime_assert_23();
    fn __compiletime_assert_24();
    fn __compiletime_assert_25();
    fn __compiletime_assert_26();
    fn __compiletime_assert_27();
    fn __compiletime_assert_28();
    fn __compiletime_assert_29();
    fn __compiletime_assert_30();
    fn __compiletime_assert_31();
    fn __compiletime_assert_32();
    fn __compiletime_assert_33();
    fn __compiletime_assert_34();
    fn __compiletime_assert_35();
    fn __compiletime_assert_36();
    fn __compiletime_assert_37();
    fn __compiletime_assert_38();
    fn __compiletime_assert_39();
    fn __compiletime_assert_40();
}

// 0x66
//
// LPFX1
// 0x67
// x86-64 REX prefix
// 0x4X
// AVX VEX prefixes
// 2-bytes VEX prefix
// 3-bytes VEX prefix
// EVEX prefix
// x86-64 REX2 prefix
// 0xD5
// AMD XOP prefix
// 0x8F
// Immediate size
// Escape opcodes
// Group opcodes (1-16)
// Immediates
fn inat_is_legacy_prefix(mut attr: u32) -> i32 {
attr &= 31u32;
(attr != 0 && attr <= 11) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_insn_h_17 {
    value: i32,
    bytes: [u8; 4],
}


fn inat_is_address_size_prefix(attr: u32) -> i32 {
(attr & 31u32 == 11) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_insn_h_74 {
    vex_prefix: insn_field,
    xop_prefix: insn_field,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_insn_h_86 {
    immediate: insn_field,
    moffset1: insn_field,
    immediate1: insn_field,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_insn_h_91 {
    moffset2: insn_field,
    immediate2: insn_field,
}


fn inat_is_operand_size_prefix(attr: u32) -> i32 {
(attr & 31u32 == 1) as i32
}

fn inat_is_rex_prefix(attr: u32) -> i32 {
(attr & 31u32 == 12) as i32
}

fn inat_is_rex2_prefix(attr: u32) -> i32 {
(attr & 31u32 == 16) as i32
}

fn inat_is_vex_prefix(mut attr: u32) -> i32 {
attr &= 31u32;
(attr == 13 || attr == 14 || attr == 15) as i32
}

fn inat_is_evex_prefix(attr: u32) -> i32 {
(attr & 31u32 == 15) as i32
}

fn inat_is_vex3_prefix(attr: u32) -> i32 {
(attr & 31u32 == 14) as i32
}

fn inat_is_xop_prefix(attr: u32) -> i32 {
(attr & 31u32 == 17) as i32
}

fn inat_is_escape(attr: u32) -> i32 {
(attr & 96u32) as i32
}

fn inat_is_group(attr: u32) -> i32 {
(attr & 3968u32) as i32
}

fn inat_has_immediate(attr: u32) -> i32 {
(attr & 28672u32) as i32
}

fn inat_immediate_size(attr: u32) -> i32 {
((attr & 28672u32) >> 12i32) as i32
}

fn inat_has_modrm(attr: u32) -> i32 {
(attr & 32768u32) as i32
}

fn inat_is_force64(attr: u32) -> i32 {
(attr & 65536u32) as i32
}

fn inat_has_second_immediate(attr: u32) -> i32 {
(attr & 131072u32) as i32
}

fn inat_has_moffset(attr: u32) -> i32 {
(attr & 262144u32) as i32
}

fn inat_accept_vex(attr: u32) -> i32 {
(attr & 1048576u32) as i32
}

fn inat_accept_xop(attr: u32) -> i32 {
(attr & 1048576u32) as i32
}

fn inat_must_vex(attr: u32) -> i32 {
(attr & 6291456u32) as i32
}

fn inat_must_evex(attr: u32) -> i32 {
(attr & 4194304u32) as i32
}

fn inat_evex_scalable(attr: u32) -> i32 {
(attr & 33554432u32) as i32
}

fn inat_is_invalid64(attr: u32) -> i32 {
(attr & 67108864u32) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct insn_field {
    __slate_anon_0: __SlateRecord_insn_h_17,
    got: u8,
    nbytes: u8,
}


fn insn_field_set(p: *mut insn_field, v: i32, n: u8) {
unsafe {
(*p).__slate_anon_0.value = v;
}
unsafe {
(*p).nbytes = n;
}
}

fn insn_set_byte(p: *mut insn_field, n: u8, v: u8) {
unsafe {
*((*p).__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset((n as i32) as isize) = v;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct insn {
    prefixes: insn_field,
    rex_prefix: insn_field,
    __slate_anon_2: __SlateRecord_insn_h_74,
    opcode: insn_field,
    modrm: insn_field,
    sib: insn_field,
    displacement: insn_field,
    __slate_anon_7: __SlateRecord_insn_h_86,
    __slate_anon_8: __SlateRecord_insn_h_91,
    emulate_prefix_size: i32,
    attr: u32,
    opnd_bytes: u8,
    addr_bytes: u8,
    length: u8,
    x86_64: u8,
    kaddr: *const u8,
    end_kaddr: *const u8,
    next_byte: *const u8,
}


// VEX2.M always 1
#[unsafe(no_mangle)]
extern "C" fn insn_get_prefixes(insn: *mut insn) -> i32 {
let mut v: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u8 = core::ptr::addr_of_mut!(v) as *mut u8;
let mut v_1: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u8 = core::ptr::addr_of_mut!(v_1) as *mut u8;
let mut v_2: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u8 = core::ptr::addr_of_mut!(v_2) as *mut u8;
let mut __v24: bool = unsafe { core::mem::zeroed() };
let mut v_3: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut u8 = core::ptr::addr_of_mut!(v_3) as *mut u8;
let mut b2: u8 = unsafe { core::mem::zeroed() };
let mut __v31: bool = unsafe { core::mem::zeroed() };
let mut v_4: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut u8 = core::ptr::addr_of_mut!(v_4) as *mut u8;
let mut v_5: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_43: *mut u8 = core::ptr::addr_of_mut!(v_5) as *mut u8;
let mut v_6: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_48: *mut u8 = core::ptr::addr_of_mut!(v_6) as *mut u8;
let mut v_7: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_55: *mut u8 = core::ptr::addr_of_mut!(v_7) as *mut u8;
let mut v: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_79: *mut u8 = core::ptr::addr_of_mut!(v) as *mut u8;
let mut nb: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_84: *mut i32 = core::ptr::addr_of_mut!(nb) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_85: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut lb: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_86: *mut u8 = core::ptr::addr_of_mut!(lb) as *mut u8;
let mut b: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_87: *mut u8 = core::ptr::addr_of_mut!(b) as *mut u8;
let mut attr: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_88: *mut u32 = core::ptr::addr_of_mut!(attr) as *mut u32;
let mut prefixes: *mut insn_field = unsafe { core::mem::zeroed() };
unsafe {
prefixes = core::ptr::addr_of_mut!((*insn).prefixes);
let __slate_auto_bytes_5 = __slate_slot_88 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_87 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_7 = __slate_slot_86 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_8 = __slate_slot_85 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_84 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
if (*prefixes).got != 0 {
return 0;
} else {
'__join_0: {
insn_get_emulate_prefix(insn);
*__slate_slot_84 = 0;
*__slate_slot_86 = 0;
if unlikely__2!(!validate_next!(u8, insn, 0i32)) == 0 {
let r_4: u8 = get_unaligned!(((*insn).next_byte as *mut u8).wrapping_offset(0 as isize));
let __slate_auto_bytes_4 = __slate_slot_79 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 1usize);
*__slate_slot_79 = r_4;
*__slate_slot_87 = *__slate_slot_79;
*__slate_slot_88 = inat_get_opcode_attribute(*__slate_slot_87);
'__loop_46: while inat_is_legacy_prefix(*__slate_slot_88) != 0 {
// Skip if same prefix
*__slate_slot_85 = 0;
'__join_48: {
while *__slate_slot_85 < *__slate_slot_84 {
if (*((*prefixes).__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(*__slate_slot_85 as isize) as i32) == (*__slate_slot_87 as i32) {
break '__join_48;
} else {
let __v78: i32 = *__slate_slot_85;
*__slate_slot_85 = __v78.wrapping_add(1);
}
}
if *__slate_slot_84 == 4 {
break '__loop_46;
} else {
// Invalid instruction
let __v76: i32 = *__slate_slot_84;
*__slate_slot_84 = __v76.wrapping_add(1);
*((*prefixes).__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(__v76 as isize) = *__slate_slot_87;
if inat_is_address_size_prefix(*__slate_slot_88) != 0 {
// address size switches 2/4 or 4/8
if (*insn).x86_64 != 0 {
(*insn).addr_bytes = (((*insn).addr_bytes as i32) ^ 12) as u8;
} else {
(*insn).addr_bytes = (((*insn).addr_bytes as i32) ^ 6) as u8;
}
} else {
if inat_is_operand_size_prefix(*__slate_slot_88) != 0 {
// oprand size switches 2/4
(*insn).opnd_bytes = (((*insn).opnd_bytes as i32) ^ 6) as u8;
}
}
}
}
(*prefixes).nbytes = ((*prefixes).nbytes as i32).wrapping_add(1) as u8;
let __v61: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v61.wrapping_offset(1 as isize);
*__slate_slot_86 = *__slate_slot_87;
if unlikely__2!(!validate_next!(u8, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: u8 = get_unaligned!(((*insn).next_byte as *mut u8).wrapping_offset(0 as isize));
let __slate_auto_bytes = __slate_slot_55 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_55 = r;
*__slate_slot_87 = *__slate_slot_55;
*__slate_slot_88 = inat_get_opcode_attribute(*__slate_slot_87);
}
}
// Set the last prefix
if *__slate_slot_86 != 0 && (*__slate_slot_86 as i32) != (*((*insn).prefixes.__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) as i32) {
if unlikely__9!(unsafe { *((*insn).prefixes.__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) }) != 0 {
// Swap the last prefix
*__slate_slot_87 = *((*insn).prefixes.__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize);
*__slate_slot_85 = 0;
while *__slate_slot_85 < *__slate_slot_84 {
if (*((*prefixes).__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(*__slate_slot_85 as isize) as i32) == (*__slate_slot_86 as i32) {
insn_set_byte(prefixes, *__slate_slot_85 as u8, *__slate_slot_87);
}
let __v54: i32 = *__slate_slot_85;
*__slate_slot_85 = __v54.wrapping_add(1);
}
}
insn_set_byte(core::ptr::addr_of_mut!((*insn).prefixes), 3u8, *__slate_slot_86);
}
'__join_1: {
// Decode REX prefix
if (*insn).x86_64 != 0 {
if unlikely__2!(!validate_next!(u8, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r_3: u8 = get_unaligned!(((*insn).next_byte as *mut u8).wrapping_offset(0 as isize));
let __slate_auto_bytes_3 = __slate_slot_48 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 1usize);
*__slate_slot_48 = r_3;
*__slate_slot_87 = *__slate_slot_48;
*__slate_slot_88 = inat_get_opcode_attribute(*__slate_slot_87);
if inat_is_rex_prefix(*__slate_slot_88) != 0 {
insn_field_set(core::ptr::addr_of_mut!((*insn).rex_prefix), *__slate_slot_87 as i32, 1u8);
let __v38: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v38.wrapping_offset(1 as isize);
if X86_REX_W!(*__slate_slot_87) != 0 {
(*insn).opnd_bytes = 8;
}
// REX.W overrides opnd_size
} else {
if inat_is_rex2_prefix(*__slate_slot_88) != 0 {
insn_set_byte(core::ptr::addr_of_mut!((*insn).rex_prefix), 0u8, *__slate_slot_87);
if unlikely__2!(!validate_next!(u8, insn, 1i32)) != 0 {
break '__join_0;
} else {
let r: u8 = get_unaligned!(((*insn).next_byte as *mut u8).wrapping_offset(1 as isize));
let __slate_auto_bytes = __slate_slot_43 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_43 = r;
*__slate_slot_87 = *__slate_slot_43;
insn_set_byte(core::ptr::addr_of_mut!((*insn).rex_prefix), 1u8, *__slate_slot_87);
(*insn).rex_prefix.nbytes = 2;
let __v41: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v41.wrapping_offset(2 as isize);
if X86_REX_W!(*__slate_slot_87) != 0 {
(*insn).opnd_bytes = 8;
}
// REX.W overrides opnd_size
(*insn).rex_prefix.got = 1;
break '__join_1;
}
}
}
}
}
(*insn).rex_prefix.got = 1;
if unlikely__2!(!validate_next!(u8, insn, 0i32)) != 0 {
break '__join_0;
} else {
{
let r: u8 = get_unaligned!(((*insn).next_byte as *mut u8).wrapping_offset(0 as isize));
let __slate_auto_bytes = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_32 = r;
let __v35: u8 = *__slate_slot_32;
// Decode VEX/XOP prefix
*__slate_slot_87 = __v35;
if inat_is_vex_prefix(*__slate_slot_88) != 0 {
__v31 = true;
} else {
__v31 = inat_is_xop_prefix(*__slate_slot_88) != 0;
}
}
if __v31 {
if unlikely__2!(!validate_next!(u8, insn, 1i32)) != 0 {
break '__join_0;
} else {
let r_2: u8 = get_unaligned!(((*insn).next_byte as *mut u8).wrapping_offset(1 as isize));
let __slate_auto_bytes_2 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
*__slate_slot_25 = r_2;
b2 = *__slate_slot_25;
if inat_is_xop_prefix(*__slate_slot_88) != 0 && X86_MODRM_REG!(b2) == 0 {
// Grp1A.0 is always POP Ev
} else {
if (*insn).x86_64 == 0 {
// In 32-bits mode, if the [7:6] bits (mod bits of
// ModRM) on the second byte are not 11b, it is
// LDS or LES or BOUND.
if X86_MODRM_MOD__2!(b2) != 3 {
break '__join_1;
}
}
insn_set_byte(core::ptr::addr_of_mut!((*insn).__slate_anon_2.vex_prefix), 0u8, *__slate_slot_87);
insn_set_byte(core::ptr::addr_of_mut!((*insn).__slate_anon_2.vex_prefix), 1u8, b2);
if inat_is_evex_prefix(*__slate_slot_88) != 0 {
if unlikely__2!(!validate_next!(u8, insn, 2i32)) != 0 {
break '__join_0;
} else {
let r_1: u8 = get_unaligned!(((*insn).next_byte as *mut u8).wrapping_offset(2 as isize));
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
*__slate_slot_8 = r_1;
b2 = *__slate_slot_8;
insn_set_byte(core::ptr::addr_of_mut!((*insn).__slate_anon_2.vex_prefix), 2u8, b2);
if unlikely__2!(!validate_next!(u8, insn, 3i32)) != 0 {
break '__join_0;
} else {
let r: u8 = get_unaligned!(((*insn).next_byte as *mut u8).wrapping_offset(3 as isize));
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_3 = r;
b2 = *__slate_slot_3;
insn_set_byte(core::ptr::addr_of_mut!((*insn).__slate_anon_2.vex_prefix), 3u8, b2);
(*insn).__slate_anon_2.vex_prefix.nbytes = 4;
let __v1: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v1.wrapping_offset(4 as isize);
if (*insn).x86_64 != 0 && X86_VEX_W!(b2) != 0 {
(*insn).opnd_bytes = 8;
}
// VEX.W overrides opnd_size
}
}
} else {
if inat_is_vex3_prefix(*__slate_slot_88) != 0 {
__v24 = true;
} else {
__v24 = inat_is_xop_prefix(*__slate_slot_88) != 0;
}
if __v24 {
if unlikely__2!(!validate_next!(u8, insn, 2i32)) != 0 {
break '__join_0;
} else {
let r: u8 = get_unaligned!(((*insn).next_byte as *mut u8).wrapping_offset(2 as isize));
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_16 = r;
b2 = *__slate_slot_16;
insn_set_byte(core::ptr::addr_of_mut!((*insn).__slate_anon_2.vex_prefix), 2u8, b2);
(*insn).__slate_anon_2.vex_prefix.nbytes = 3;
let __v14: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v14.wrapping_offset(3 as isize);
if (*insn).x86_64 != 0 && X86_VEX_W!(b2) != 0 {
(*insn).opnd_bytes = 8;
}
// VEX.W/XOP.W overrides opnd_size
}
} else {
// For VEX2, fake VEX3-like byte#2.
// Makes it easier to decode vex.W, vex.vvvv,
// vex.L and vex.pp. Masking with 0x7f sets vex.W == 0.
insn_set_byte(core::ptr::addr_of_mut!((*insn).__slate_anon_2.vex_prefix), 2u8, ((b2 as i32) & 127) as u8);
(*insn).__slate_anon_2.vex_prefix.nbytes = 2;
let __v22: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v22.wrapping_offset(2 as isize);
}
}
}
}
}
}
}
(*insn).__slate_anon_2.vex_prefix.got = 1;
(*prefixes).got = 1;
return 0;
}
}
return 61i32.wrapping_neg();
}
}
}

#[unsafe(no_mangle)]
extern "C" fn insn_get_opcode(insn: *mut insn) -> i32 {
let mut v: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u8 = core::ptr::addr_of_mut!(v) as *mut u8;
let mut __v14: bool = unsafe { core::mem::zeroed() };
let mut __v16: bool = unsafe { core::mem::zeroed() };
let mut __v17: bool = unsafe { core::mem::zeroed() };
let mut __v18: bool = unsafe { core::mem::zeroed() };
// SCALABLE EVEX uses p bits to encode operand size
let mut __v19: bool = unsafe { core::mem::zeroed() };
let mut p: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u8 = core::ptr::addr_of_mut!(p) as *mut u8;
let mut m: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u8 = core::ptr::addr_of_mut!(m) as *mut u8;
let mut v: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u8 = core::ptr::addr_of_mut!(v) as *mut u8;
let mut op: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut u8 = core::ptr::addr_of_mut!(op) as *mut u8;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut pfx_id: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut i32 = core::ptr::addr_of_mut!(pfx_id) as *mut i32;
let mut opcode: *mut insn_field = unsafe { core::mem::zeroed() };
unsafe {
opcode = core::ptr::addr_of_mut!((*insn).opcode);
let __slate_auto_bytes_3 = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 1usize);
if (*opcode).got != 0 {
return 0;
} else {
*__slate_slot_31 = insn_get_prefixes(insn);
if *__slate_slot_31 != 0 {
return *__slate_slot_31;
} else {
'__join_0: {
if unlikely__2!(!validate_next!(u8, insn, 0i32)) == 0 {
let r_1: u8 = get_unaligned!((*insn).next_byte as *mut u8);
let __v25: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v25.wrapping_offset(core::mem::size_of::<u8>() as isize);
let __slate_auto_bytes_2 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
*__slate_slot_22 = r_1;
let __v28: u8 = *__slate_slot_22;
// Get first opcode
*__slate_slot_30 = __v28;
insn_set_byte(opcode, 0u8, *__slate_slot_30);
(*opcode).nbytes = 1;
// Check if there is VEX/XOP prefix or not
if insn_is_avx_or_xop(insn) != 0 {
let __slate_auto_bytes_1 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
// XOP prefix has different encoding
if unlikely!(avx_insn_is_xop(insn)) != 0 {
*__slate_slot_21 = insn_xop_map_bits(insn);
(*insn).attr = inat_get_xop_attribute(*__slate_slot_30, *__slate_slot_21);
if inat_accept_xop((*insn).attr) == 0 {
(*insn).attr = 0;
return 22i32.wrapping_neg();
}
// XOP has only 1 byte for opcode
} else {
*__slate_slot_21 = insn_vex_m_bits(insn);
*__slate_slot_20 = insn_vex_p_bits(insn);
(*insn).attr = inat_get_avx_attribute(*__slate_slot_30, *__slate_slot_21, *__slate_slot_20);
if inat_evex_scalable((*insn).attr) != 0 {
__v19 = insn_vex_w_bit(insn) == 0;
} else {
__v19 = false;
}
if __v19 && (*__slate_slot_20 as i32) == 1 {
(*insn).opnd_bytes = 2;
}
if inat_must_evex((*insn).attr) != 0 {
__v18 = insn_is_evex(insn) == 0;
} else {
__v18 = false;
}
if __v18 {
__v17 = true;
} else {
if inat_accept_vex((*insn).attr) == 0 {
__v16 = inat_is_group((*insn).attr) == 0;
} else {
__v16 = false;
}
__v17 = __v16;
}
if __v17 {
// This instruction is bad
(*insn).attr = 0;
return 22i32.wrapping_neg();
}
// VEX has only 1 byte for opcode
}
} else {
// Check if there is REX2 prefix or not
if insn_is_rex2(insn) != 0 {
if insn_rex2_m_bit(insn) != 0 {
// map 1 is escape 0x0f
let esc_attr: u32 = inat_get_opcode_attribute(15u8);
*__slate_slot_32 = insn_last_prefix_id(insn);
(*insn).attr = inat_get_escape_attribute(*__slate_slot_30, *__slate_slot_32, esc_attr);
} else {
(*insn).attr = inat_get_opcode_attribute(*__slate_slot_30);
}
} else {
(*insn).attr = inat_get_opcode_attribute(*__slate_slot_30);
if (*insn).x86_64 != 0 {
__v14 = inat_is_invalid64((*insn).attr) != 0;
} else {
__v14 = false;
}
if __v14 {
// This instruction is invalid, like UD2. Stop decoding.
(*insn).attr = (*insn).attr & 67108864u32;
}
while inat_is_escape((*insn).attr) != 0 {
if unlikely__2!(!validate_next!(u8, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: u8 = get_unaligned!((*insn).next_byte as *mut u8);
let __v6: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v6.wrapping_offset(core::mem::size_of::<u8>() as isize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_3 = r;
let __v9: u8 = *__slate_slot_3;
// Get escaped opcode
*__slate_slot_30 = __v9;
let __v1: u8 = (*opcode).nbytes;
(*opcode).nbytes = (__v1 as i32).wrapping_add(1) as u8;
*((*opcode).__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset((__v1 as i32) as isize) = *__slate_slot_30;
*__slate_slot_32 = insn_last_prefix_id(insn);
(*insn).attr = inat_get_escape_attribute(*__slate_slot_30, *__slate_slot_32, (*insn).attr);
}
}
if inat_must_vex((*insn).attr) != 0 {
// This instruction is bad
(*insn).attr = 0;
return 22i32.wrapping_neg();
}
}
}
(*opcode).got = 1;
return 0;
}
}
return 61i32.wrapping_neg();
}
}
}
}

#[unsafe(no_mangle)]
extern "C" fn insn_get_modrm(insn: *mut insn) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut v: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u8 = core::ptr::addr_of_mut!(v) as *mut u8;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut r#mod: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u8 = core::ptr::addr_of_mut!(r#mod) as *mut u8;
let mut pfx_id: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u8 = core::ptr::addr_of_mut!(pfx_id) as *mut u8;
let mut modrm: *mut insn_field = unsafe { core::mem::zeroed() };
unsafe {
modrm = core::ptr::addr_of_mut!((*insn).modrm);
let __slate_auto_bytes_1 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_3 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
if (*modrm).got != 0 {
return 0;
} else {
*__slate_slot_11 = insn_get_opcode(insn);
if *__slate_slot_11 != 0 {
return *__slate_slot_11;
} else {
if inat_has_modrm((*insn).attr) != 0 {
if unlikely__2!(!validate_next!(u8, insn, 0i32)) != 0 {
return 61i32.wrapping_neg();
} else {
let r: u8 = get_unaligned!((*insn).next_byte as *mut u8);
let __v6: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v6.wrapping_offset(core::mem::size_of::<u8>() as isize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_3 = r;
*__slate_slot_12 = *__slate_slot_3;
insn_field_set(modrm, *__slate_slot_12 as i32, 1u8);
if inat_is_group((*insn).attr) != 0 {
*__slate_slot_13 = insn_last_prefix_id(insn) as u8;
(*insn).attr = inat_get_group_attribute(*__slate_slot_12, *__slate_slot_13 as i32, (*insn).attr);
if insn_is_avx_or_xop(insn) != 0 {
__v2 = inat_accept_vex((*insn).attr) == 0;
} else {
__v2 = false;
}
if __v2 {
__v1 = inat_accept_xop((*insn).attr) == 0;
} else {
__v1 = false;
}
if __v1 {
// Bad insn
(*insn).attr = 0;
return 22i32.wrapping_neg();
}
}
}
}
if (*insn).x86_64 != 0 {
__v0 = inat_is_force64((*insn).attr) != 0;
} else {
__v0 = false;
}
if __v0 {
(*insn).opnd_bytes = 8;
}
(*modrm).got = 1;
return 0;
}
}
}
}

#[unsafe(no_mangle)]
extern "C" fn insn_get_sib(insn: *mut insn) -> i32 {
let mut v: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u8 = core::ptr::addr_of_mut!(v) as *mut u8;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut modrm: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u8 = core::ptr::addr_of_mut!(modrm) as *mut u8;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
if (*insn).sib.got != 0 {
return 0;
} else {
*__slate_slot_8 = insn_get_modrm(insn);
if *__slate_slot_8 != 0 {
return *__slate_slot_8;
} else {
if (*insn).modrm.nbytes != 0 {
*__slate_slot_9 = *((*insn).modrm.__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize);
if ((*insn).addr_bytes as i32) != 2 && X86_MODRM_MOD__2!(*__slate_slot_9) != 3 && X86_MODRM_RM__2!(*__slate_slot_9) == 4 {
if unlikely__2!(!validate_next!(u8, insn, 0i32)) != 0 {
return 61i32.wrapping_neg();
} else {
let r: u8 = get_unaligned!((*insn).next_byte as *mut u8);
let __v3: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v3.wrapping_offset(core::mem::size_of::<u8>() as isize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_0 = r;
insn_field_set(core::ptr::addr_of_mut!((*insn).sib), *__slate_slot_0 as i32, 1u8);
}
}
}
(*insn).sib.got = 1;
return 0;
}
}
}
}

#[unsafe(no_mangle)]
extern "C" fn insn_get_displacement(insn: *mut insn) -> i32 {
let mut v: core::mem::MaybeUninit<i8> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i8 = core::ptr::addr_of_mut!(v) as *mut i8;
let mut v_1: core::mem::MaybeUninit<i16> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i16 = core::ptr::addr_of_mut!(v_1) as *mut i16;
let mut v: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(v) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut base: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut u8 = core::ptr::addr_of_mut!(base) as *mut u8;
let mut rm: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u8 = core::ptr::addr_of_mut!(rm) as *mut u8;
let mut r#mod: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u8 = core::ptr::addr_of_mut!(r#mod) as *mut u8;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_3 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_4 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
if (*insn).displacement.got != 0 {
return 0;
} else {
*__slate_slot_24 = insn_get_sib(insn);
if *__slate_slot_24 != 0 {
return *__slate_slot_24;
} else {
'__join_1: {
if (*insn).modrm.nbytes != 0 {
// Interpreting the modrm byte:
// mod = 00 - no displacement fields (exceptions below)
// mod = 01 - 1-byte displacement field
// mod = 10 - displacement field is 4 bytes, or 2 bytes if
// 	address size = 2 (0x67 prefix in 32-bit mode)
// mod = 11 - no memory operand
//
// If address size = 2...
// mod = 00, r/m = 110 - displacement field is 2 bytes
//
// If address size != 2...
// mod != 11, r/m = 100 - SIB byte exists
// mod = 00, SIB base = 101 - displacement field is 4 bytes
// mod = 00, r/m = 101 - rip-relative addressing, displacement
// 	field is 4 bytes
*__slate_slot_27 = X86_MODRM_MOD!(unsafe { (*insn).modrm.__slate_anon_0.value }) as u8;
*__slate_slot_26 = X86_MODRM_RM!(unsafe { (*insn).modrm.__slate_anon_0.value }) as u8;
*__slate_slot_25 = X86_SIB_BASE!(unsafe { (*insn).sib.__slate_anon_0.value }) as u8;
if (*__slate_slot_27 as i32) != 3 {
if (*__slate_slot_27 as i32) == 1 {
if unlikely__2!(!validate_next!(i8, insn, 0i32)) == 0 {
let r: i8 = get_unaligned__4!((*insn).next_byte as *mut i8);
let __v3: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v3.wrapping_offset(core::mem::size_of::<i8>() as isize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_0 = r;
insn_field_set(core::ptr::addr_of_mut!((*insn).displacement), *__slate_slot_0 as i32, 1u8);
break '__join_1;
}
} else {
if ((*insn).addr_bytes as i32) == 2 {
if (*__slate_slot_27 as i32) == 0 && (*__slate_slot_26 as i32) == 6 || (*__slate_slot_27 as i32) == 2 {
if unlikely__2!(!validate_next!(i16, insn, 0i32)) == 0 {
let r: i16 = get_unaligned__3!((*insn).next_byte as *mut i16);
let __v11: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v11.wrapping_offset(core::mem::size_of::<i16>() as isize);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
*__slate_slot_8 = (r as u16) as i16;
insn_field_set(core::ptr::addr_of_mut!((*insn).displacement), *__slate_slot_8 as i32, 2u8);
break '__join_1;
}
} else {
break '__join_1;
}
} else {
if (*__slate_slot_27 as i32) == 0 && (*__slate_slot_26 as i32) == 5 || (*__slate_slot_27 as i32) == 2 || (*__slate_slot_27 as i32) == 0 && (*__slate_slot_25 as i32) == 5 {
if unlikely__2!(!validate_next!(i32, insn, 0i32)) == 0 {
let r: i32 = get_unaligned__2!((*insn).next_byte as *mut i32);
let __v19: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v19.wrapping_offset(core::mem::size_of::<i32>() as isize);
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_16 = (r as u32) as i32;
insn_field_set(core::ptr::addr_of_mut!((*insn).displacement), *__slate_slot_16, 4u8);
break '__join_1;
}
} else {
break '__join_1;
}
}
}
return 61i32.wrapping_neg();
}
}
}
(*insn).displacement.got = 1;
return 0;
}
}
}
}

#[unsafe(no_mangle)]
extern "C" fn insn_get_immediate(insn: *mut insn) -> i32 {
let mut v: core::mem::MaybeUninit<i8> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i8 = core::ptr::addr_of_mut!(v) as *mut i8;
let mut v_1: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(v_1) as *mut i32;
let mut v_2: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(v_2) as *mut i32;
let mut v_3: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut i32 = core::ptr::addr_of_mut!(v_3) as *mut i32;
let mut v_4: core::mem::MaybeUninit<i16> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut i16 = core::ptr::addr_of_mut!(v_4) as *mut i16;
let mut v: core::mem::MaybeUninit<i8> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut i8 = core::ptr::addr_of_mut!(v) as *mut i8;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_48: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_48 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
if (*insn).__slate_anon_7.immediate.got != 0 {
return 0;
} else {
*__slate_slot_48 = insn_get_displacement(insn);
if *__slate_slot_48 != 0 {
return *__slate_slot_48;
} else {
'__join_0: {
if inat_has_moffset((*insn).attr) != 0 {
if __get_moffset(insn) == 0 {
break '__join_0;
}
} else {
if inat_has_immediate((*insn).attr) != 0 {
let __t0: i32 = inat_immediate_size((*insn).attr);
if __t0 == 1 {
if unlikely__2!(!validate_next!(i8, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: i8 = get_unaligned__4!((*insn).next_byte as *mut i8);
let __v43: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v43.wrapping_offset(core::mem::size_of::<i8>() as isize);
let __slate_auto_bytes = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_40 = r;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_7.immediate), *__slate_slot_40 as i32, 1u8);
}
} else {
if __t0 == 2 {
if unlikely__2!(!validate_next!(i16, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: i16 = get_unaligned__3!((*insn).next_byte as *mut i16);
let __v35: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v35.wrapping_offset(core::mem::size_of::<i16>() as isize);
let __slate_auto_bytes = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
*__slate_slot_32 = (r as u16) as i16;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_7.immediate), *__slate_slot_32 as i32, 2u8);
}
} else {
if __t0 == 3 {
if unlikely__2!(!validate_next!(i32, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: i32 = get_unaligned__2!((*insn).next_byte as *mut i32);
let __v27: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v27.wrapping_offset(core::mem::size_of::<i32>() as isize);
let __slate_auto_bytes = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_24 = (r as u32) as i32;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_7.immediate), *__slate_slot_24, 4u8);
}
} else {
if __t0 == 4 {
if unlikely__2!(!validate_next!(i32, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r_1: i32 = get_unaligned__2!((*insn).next_byte as *mut i32);
let __v19: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v19.wrapping_offset(core::mem::size_of::<i32>() as isize);
let __slate_auto_bytes_1 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
*__slate_slot_16 = (r_1 as u32) as i32;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_7.immediate1), *__slate_slot_16, 4u8);
if unlikely__2!(!validate_next!(i32, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: i32 = get_unaligned__2!((*insn).next_byte as *mut i32);
let __v11: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v11.wrapping_offset(core::mem::size_of::<i32>() as isize);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_8 = (r as u32) as i32;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_8.immediate2), *__slate_slot_8, 4u8);
}
}
} else {
if __t0 == 5 {
if __get_immptr(insn) == 0 {
break '__join_0;
}
} else {
if __t0 == 6 {
if __get_immv32(insn) == 0 {
break '__join_0;
}
} else {
if __t0 == 7 {
if __get_immv(insn) == 0 {
break '__join_0;
}
} else {
break '__join_0;
}
}
}
}
}
}
}
if inat_has_second_immediate((*insn).attr) != 0 {
if unlikely__2!(!validate_next!(i8, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: i8 = get_unaligned__4!((*insn).next_byte as *mut i8);
let __v3: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v3.wrapping_offset(core::mem::size_of::<i8>() as isize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_0 = r;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_8.immediate2), *__slate_slot_0 as i32, 1u8);
}
}
}
}
(*insn).__slate_anon_7.immediate.got = 1;
return 0;
}
return 61i32.wrapping_neg();
}
}
}
// Here, insn must have an immediate, but failed
}

fn insn_is_rex2(insn: *mut insn) -> i32 {
if (unsafe { (*insn).prefixes.got }) == 0 {
insn_get_prefixes(insn);
}
(((unsafe { (*insn).rex_prefix.nbytes }) as i32) == 2) as i32
}

fn insn_rex2_m_bit(insn: *mut insn) -> u8 {
X86_REX2_M!(unsafe { *((*insn).rex_prefix.__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) }) as u8
}

fn insn_is_avx_or_xop(insn: *mut insn) -> i32 {
if (unsafe { (*insn).prefixes.got }) == 0 {
insn_get_prefixes(insn);
}
((unsafe { (*insn).__slate_anon_2.vex_prefix.__slate_anon_0.value }) != 0) as i32
}

fn insn_is_evex(insn: *mut insn) -> i32 {
if (unsafe { (*insn).prefixes.got }) == 0 {
insn_get_prefixes(insn);
}
(((unsafe { (*insn).__slate_anon_2.vex_prefix.nbytes }) as i32) == 4) as i32
}

fn avx_insn_is_xop(insn: *mut insn) -> i32 {
let attr: u32 = unsafe { inat_get_opcode_attribute(*((*insn).__slate_anon_2.vex_prefix.__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize)) };
inat_is_xop_prefix(attr)
}

fn insn_vex_m_bits(insn: *mut insn) -> u8 {
if ((unsafe { (*insn).__slate_anon_2.vex_prefix.nbytes }) as i32) == 2 {
return 1;
} else {
if ((unsafe { (*insn).__slate_anon_2.vex_prefix.nbytes }) as i32) == 3 {
return X86_VEX3_M!(unsafe { *((*insn).__slate_anon_2.vex_prefix.__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) }) as u8;
} else {
return X86_EVEX_M!(unsafe { *((*insn).__slate_anon_2.vex_prefix.__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) }) as u8;
}
}
// 2 bytes VEX
// 3 bytes VEX
// EVEX
}

fn insn_vex_p_bits(insn: *mut insn) -> u8 {
if ((unsafe { (*insn).__slate_anon_2.vex_prefix.nbytes }) as i32) == 2 {
return X86_VEX_P!(unsafe { *((*insn).__slate_anon_2.vex_prefix.__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) }) as u8;
} else {
return X86_VEX_P!(unsafe { *((*insn).__slate_anon_2.vex_prefix.__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) }) as u8;
}
// 2 bytes VEX
}

fn insn_vex_w_bit(insn: *mut insn) -> u8 {
if ((unsafe { (*insn).__slate_anon_2.vex_prefix.nbytes }) as i32) < 3 {
return 0;
}
X86_VEX_W!(unsafe { *((*insn).__slate_anon_2.vex_prefix.__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) }) as u8
}

fn insn_xop_map_bits(insn: *mut insn) -> u8 {
if ((unsafe { (*insn).__slate_anon_2.xop_prefix.nbytes }) as i32) < 3 {
return 0;
}
// XOP is 3 bytes
X86_XOP_M!(unsafe { *((*insn).__slate_anon_2.xop_prefix.__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) }) as u8
}

fn insn_xop_p_bits(insn: *mut insn) -> u8 {
X86_XOP_P!(unsafe { *((*insn).__slate_anon_2.vex_prefix.__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) }) as u8
}

fn insn_last_prefix_id(insn: *mut insn) -> i32 {
if insn_is_avx_or_xop(insn) != 0 {
if avx_insn_is_xop(insn) != 0 {
return insn_xop_p_bits(insn) as i32;
}
return insn_vex_p_bits(insn) as i32; // VEX_p is a SIMD prefix id
}
if (unsafe { *((*insn).prefixes.__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) }) != 0 {
return unsafe { inat_get_last_prefix_id(*((*insn).prefixes.__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize)) };
}
0
}

// __ignore_sync_check__
// __ignore_sync_check__
// __ignore_sync_check__
// __ignore_sync_check__
/// insn_init() - initialize struct insn
/// @insn:	&struct insn to be initialized
/// @kaddr:	address (in kernel memory) of instruction (or copy thereof)
/// @buf_len:	length of the insn buffer at @kaddr
/// @x86_64:	!0 for 64-bit kernel or 64-bit app
#[unsafe(no_mangle)]
extern "C" fn insn_init(insn: *mut insn, kaddr: *const (), mut buf_len: i32, x86_64: i32) {
// Instructions longer than MAX_INSN_SIZE (15 bytes) are invalid
// even if the input buffer is long enough to hold them.
if buf_len > 15 {
buf_len = 15;
}
unsafe { memset(insn as *mut (), 0i32, core::mem::size_of::<insn>() as u64) };
unsafe {
(*insn).kaddr = kaddr as *const u8;
}
unsafe {
(*insn).end_kaddr = ((kaddr as *const u8).wrapping_offset(buf_len as isize) as *const ()) as *const u8;
}
unsafe {
(*insn).next_byte = kaddr as *const u8;
}
unsafe {
(*insn).x86_64 = x86_64 as u8;
}
unsafe {
(*insn).opnd_bytes = 4;
}
if x86_64 != 0 {
unsafe {
(*insn).addr_bytes = 8;
}
} else {
unsafe {
(*insn).addr_bytes = 4;
}
}
}

static xen_prefix: [u8; 5] = [15, 11, 120, 101, 110];

static kvm_prefix: [u8; 5] = [15, 11, 107, 118, 109];

fn __insn_get_emulate_prefix(insn: *mut insn, prefix: *const u8, len: u64) -> i32 {
let mut v: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u8 = core::ptr::addr_of_mut!(v) as *mut u8;
let mut i: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(i) as *mut u64;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
*__slate_slot_10 = 0;
'__join_0: {
while *__slate_slot_10 < len {
if unlikely__2!(!validate_next!(u8, insn, *__slate_slot_10)) != 0 {
break '__join_0;
} else {
let r: u8 = get_unaligned!(((*insn).next_byte as *mut u8).wrapping_offset(*__slate_slot_10 as isize));
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_5 = r;
if (*__slate_slot_5 as i32) != (*prefix.wrapping_offset(*__slate_slot_10 as isize) as i32) {
break '__join_0;
} else {
let __v4: u64 = *__slate_slot_10;
*__slate_slot_10 = __v4.wrapping_add(1);
}
}
}
(*insn).emulate_prefix_size = len as i32;
let __v1: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v1.wrapping_offset(len as isize);
return 1;
}
return 0;
}
}

fn insn_get_emulate_prefix(insn: *mut insn) {
if __insn_get_emulate_prefix(insn, unsafe { core::ptr::addr_of!(xen_prefix) as *const u8 }, core::mem::size_of::<[u8; 5]>() as u64) != 0 {
return;
}
__insn_get_emulate_prefix(insn, unsafe { core::ptr::addr_of!(kvm_prefix) as *const u8 }, core::mem::size_of::<[u8; 5]>() as u64);
}

// insn_get_prefixes - scan x86 instruction prefix bytes
// @insn:	&struct insn containing instruction
//
// Populates the @insn->prefixes bitmap, and updates @insn->next_byte
// to point to the (first) opcode.  No effect if @insn->prefixes.got
// is already set.
//
// * Returns:
// 0:  on success
// < 0: on error
// insn_get_opcode - collect opcode(s)
// @insn:	&struct insn containing instruction
//
// Populates @insn->opcode, updates @insn->next_byte to point past the
// opcode byte(s), and set @insn->attr (except for groups).
// If necessary, first collects any preceding (prefix) bytes.
// Sets @insn->opcode.value = opcode1.  No effect if @insn->opcode.got
// is already 1.
//
// Returns:
// 0:  on success
// < 0: on error
// insn_get_modrm - collect ModRM byte, if any
// @insn:	&struct insn containing instruction
//
// Populates @insn->modrm and updates @insn->next_byte to point past the
// ModRM byte, if any.  If necessary, first collects the preceding bytes
// (prefixes and opcode(s)).  No effect if @insn->modrm.got is already 1.
//
// Returns:
// 0:  on success
// < 0: on error
/// insn_rip_relative() - Does instruction use RIP-relative addressing mode?
/// @insn:	&struct insn containing instruction
///
/// If necessary, first collects the instruction up to and including the
/// ModRM byte.  No effect if @insn->x86_64 is 0.
#[unsafe(no_mangle)]
extern "C" fn insn_rip_relative(insn: *mut insn) -> i32 {
let modrm: *mut insn_field = unsafe { core::ptr::addr_of_mut!((*insn).modrm) };
let mut ret: i32 = 0;
if (unsafe { (*insn).x86_64 }) == 0 {
return 0;
}
ret = insn_get_modrm(insn);
if ret != 0 {
return 0;
}
// For rip-relative instructions, the mod field (top 2 bits)
// is zero and the r/m field (bottom 3 bits) is 0x5.
((unsafe { (*modrm).nbytes }) != 0 && ((unsafe { *((*modrm).__slate_anon_0.bytes.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) & 199 == 5) as i32
}

// insn_get_sib() - Get the SIB byte of instruction
// @insn:	&struct insn containing instruction
//
// If necessary, first collects the instruction up to and including the
// ModRM byte.
//
// Returns:
// 0: if decoding succeeded
// < 0: otherwise.
// insn_get_displacement() - Get the displacement of instruction
// @insn:	&struct insn containing instruction
//
// If necessary, first collects the instruction up to and including the
// SIB byte.
// Displacement value is sign-expanded.
//
// * Returns:
// 0: if decoding succeeded
// < 0: otherwise.
// Decode moffset16/32/64. Return 0 if failed
fn __get_moffset(insn: *mut insn) -> i32 {
let mut v: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(v) as *mut i32;
let mut v_1: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(v_1) as *mut i32;
let mut v_2: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(v_2) as *mut i32;
let mut v: core::mem::MaybeUninit<i16> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut i16 = core::ptr::addr_of_mut!(v) as *mut i16;
unsafe {
'__join_0: {
let __t0: i32 = (*insn).addr_bytes as i32;
if __t0 == 2 {
if unlikely__2!(!validate_next!(i16, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: i16 = get_unaligned__3!((*insn).next_byte as *mut i16);
let __v27: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v27.wrapping_offset(core::mem::size_of::<i16>() as isize);
let __slate_auto_bytes = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
*__slate_slot_24 = (r as u16) as i16;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_7.moffset1), *__slate_slot_24 as i32, 2u8);
}
} else {
if __t0 == 4 {
if unlikely__2!(!validate_next!(i32, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: i32 = get_unaligned__2!((*insn).next_byte as *mut i32);
let __v19: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v19.wrapping_offset(core::mem::size_of::<i32>() as isize);
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_16 = (r as u32) as i32;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_7.moffset1), *__slate_slot_16, 4u8);
}
} else {
if __t0 == 8 {
if unlikely__2!(!validate_next!(i32, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r_1: i32 = get_unaligned__2!((*insn).next_byte as *mut i32);
let __v11: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v11.wrapping_offset(core::mem::size_of::<i32>() as isize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
*__slate_slot_8 = (r_1 as u32) as i32;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_7.moffset1), *__slate_slot_8, 4u8);
if unlikely__2!(!validate_next!(i32, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: i32 = get_unaligned__2!((*insn).next_byte as *mut i32);
let __v3: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v3.wrapping_offset(core::mem::size_of::<i32>() as isize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_0 = (r as u32) as i32;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_8.moffset2), *__slate_slot_0, 4u8);
}
}
} else {
break '__join_0;
}
}
}
(*insn).__slate_anon_8.moffset2.got = 1;
(*insn).__slate_anon_7.moffset1.got = 1;
return 1;
}
return 0;
}
// opnd_bytes must be modified manually
}

// Decode imm v32(Iz). Return 0 if failed
fn __get_immv32(insn: *mut insn) -> i32 {
let mut v: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(v) as *mut i32;
let mut v: core::mem::MaybeUninit<i16> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i16 = core::ptr::addr_of_mut!(v) as *mut i16;
unsafe {
'__join_0: {
'__join_1: {
let __t0: i32 = (*insn).opnd_bytes as i32;
if __t0 == 2 {
if unlikely__2!(!validate_next!(i16, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: i16 = get_unaligned__3!((*insn).next_byte as *mut i16);
let __v11: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v11.wrapping_offset(core::mem::size_of::<i16>() as isize);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
*__slate_slot_8 = (r as u16) as i16;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_7.immediate), *__slate_slot_8 as i32, 2u8);
break '__join_1;
}
} else {
if __t0 != 4 {
if __t0 != 8 {
break '__join_0;
}
}
}
if unlikely__2!(!validate_next!(i32, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: i32 = get_unaligned__2!((*insn).next_byte as *mut i32);
let __v3: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v3.wrapping_offset(core::mem::size_of::<i32>() as isize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_0 = (r as u32) as i32;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_7.immediate), *__slate_slot_0, 4u8);
}
}
return 1;
}
return 0;
}
// opnd_bytes must be modified manually
}

// Decode imm v64(Iv/Ov), Return 0 if failed
fn __get_immv(insn: *mut insn) -> i32 {
let mut v: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(v) as *mut i32;
let mut v_1: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(v_1) as *mut i32;
let mut v_2: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(v_2) as *mut i32;
let mut v: core::mem::MaybeUninit<i16> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut i16 = core::ptr::addr_of_mut!(v) as *mut i16;
unsafe {
'__join_0: {
let __t0: i32 = (*insn).opnd_bytes as i32;
if __t0 == 2 {
if unlikely__2!(!validate_next!(i16, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: i16 = get_unaligned__3!((*insn).next_byte as *mut i16);
let __v27: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v27.wrapping_offset(core::mem::size_of::<i16>() as isize);
let __slate_auto_bytes = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
*__slate_slot_24 = (r as u16) as i16;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_7.immediate1), *__slate_slot_24 as i32, 2u8);
}
} else {
if __t0 == 4 {
if unlikely__2!(!validate_next!(i32, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: i32 = get_unaligned__2!((*insn).next_byte as *mut i32);
let __v19: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v19.wrapping_offset(core::mem::size_of::<i32>() as isize);
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_16 = (r as u32) as i32;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_7.immediate1), *__slate_slot_16, 4u8);
(*insn).__slate_anon_7.immediate1.nbytes = 4;
}
} else {
if __t0 == 8 {
if unlikely__2!(!validate_next!(i32, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r_1: i32 = get_unaligned__2!((*insn).next_byte as *mut i32);
let __v11: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v11.wrapping_offset(core::mem::size_of::<i32>() as isize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
*__slate_slot_8 = (r_1 as u32) as i32;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_7.immediate1), *__slate_slot_8, 4u8);
if unlikely__2!(!validate_next!(i32, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: i32 = get_unaligned__2!((*insn).next_byte as *mut i32);
let __v3: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v3.wrapping_offset(core::mem::size_of::<i32>() as isize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_0 = (r as u32) as i32;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_8.immediate2), *__slate_slot_0, 4u8);
}
}
} else {
break '__join_0;
}
}
}
(*insn).__slate_anon_8.immediate2.got = 1;
(*insn).__slate_anon_7.immediate1.got = 1;
return 1;
}
return 0;
}
// opnd_bytes must be modified manually
}

// Decode ptr16:16/32(Ap)
fn __get_immptr(insn: *mut insn) -> i32 {
let mut v: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u16 = core::ptr::addr_of_mut!(v) as *mut u16;
let mut v_1: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(v_1) as *mut i32;
let mut v: core::mem::MaybeUninit<i16> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i16 = core::ptr::addr_of_mut!(v) as *mut i16;
unsafe {
'__join_0: {
let __t0: i32 = (*insn).opnd_bytes as i32;
if __t0 == 2 {
if unlikely__2!(!validate_next!(i16, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: i16 = get_unaligned__3!((*insn).next_byte as *mut i16);
let __v19: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v19.wrapping_offset(core::mem::size_of::<i16>() as isize);
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
*__slate_slot_16 = (r as u16) as i16;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_7.immediate1), *__slate_slot_16 as i32, 2u8);
}
} else {
if __t0 == 4 {
if unlikely__2!(!validate_next!(i32, insn, 0i32)) != 0 {
break '__join_0;
} else {
let r: i32 = get_unaligned__2!((*insn).next_byte as *mut i32);
let __v11: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v11.wrapping_offset(core::mem::size_of::<i32>() as isize);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_8 = (r as u32) as i32;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_7.immediate1), *__slate_slot_8, 4u8);
}
} else {
if __t0 == 8 {
return 0;
} else {
break '__join_0;
}
}
}
if unlikely__2!(!validate_next!(u16, insn, 0i32)) == 0 {
let r: u16 = get_unaligned__5!((*insn).next_byte as *mut u16);
let __v3: *const u8 = (*insn).next_byte;
(*insn).next_byte = __v3.wrapping_offset(core::mem::size_of::<u16>() as isize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
*__slate_slot_0 = r;
insn_field_set(core::ptr::addr_of_mut!((*insn).__slate_anon_8.immediate2), *__slate_slot_0 as i32, 2u8);
(*insn).__slate_anon_8.immediate2.got = 1;
(*insn).__slate_anon_7.immediate1.got = 1;
return 1;
}
}
return 0;
}
// ptr16:64 is not exist (no segment)
// opnd_bytes must be modified manually
}

// insn_get_immediate() - Get the immediate in an instruction
// @insn:	&struct insn containing instruction
//
// If necessary, first collects the instruction up to and including the
// displacement bytes.
// Basically, most of immediates are sign-expanded. Unsigned-value can be
// computed by bit masking with ((1 << (nbytes * 8)) - 1)
//
// Returns:
// 0:  on success
// < 0: on error
/// insn_get_length() - Get the length of instruction
/// @insn:	&struct insn containing instruction
///
/// If necessary, first collects the instruction up to and including the
/// immediates bytes.
///
/// Returns:
///  - 0 on success
///  - < 0 on error
#[unsafe(no_mangle)]
extern "C" fn insn_get_length(insn: *mut insn) -> i32 {
let mut ret: i32 = 0;
if (unsafe { (*insn).length }) != 0 {
return 0;
}
ret = insn_get_immediate(insn);
if ret != 0 {
return ret;
}
unsafe {
(*insn).length = ((*insn).next_byte as u64).wrapping_sub((*insn).kaddr as u64) as u8;
}
0
}

/// Ensure this instruction is decoded completely
fn insn_complete(insn: *mut insn) -> i32 {
((unsafe { (*insn).opcode.got }) != 0 && (unsafe { (*insn).modrm.got }) != 0 && (unsafe { (*insn).sib.got }) != 0 && (unsafe { (*insn).displacement.got }) != 0 && (unsafe { (*insn).__slate_anon_7.immediate.got }) != 0) as i32
}

/// insn_decode() - Decode an x86 instruction
/// @insn:	&struct insn to be initialized
/// @kaddr:	address (in kernel memory) of instruction (or copy thereof)
/// @buf_len:	length of the insn buffer at @kaddr
/// @m:		insn mode, see enum insn_mode
///
/// Returns:
/// 0: if decoding succeeded
/// < 0: otherwise.
#[unsafe(no_mangle)]
extern "C" fn insn_decode(insn: *mut insn, kaddr: *const (), buf_len: i32, m: u32) -> i32 {
let mut ret: i32 = 0;
// #define INSN_MODE_KERN	-1 __ignore_sync_check__ mode is only valid in the kernel
if (m as u32) == 2 {
insn_init(insn, kaddr, buf_len, IS_ENABLED__2!());
} else {
insn_init(insn, kaddr, buf_len, ((m as u32) == 1) as i32);
}
ret = insn_get_length(insn);
if ret != 0 {
return ret;
}
if insn_complete(insn) != 0 {
return 0;
}
22i32.wrapping_neg()
}
