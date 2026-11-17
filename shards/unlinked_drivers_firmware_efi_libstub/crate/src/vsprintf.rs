
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0
//!
//! When building position independent code with GCC using the -fPIC option,
//! (or even the -fPIE one on older versions), it will assume that we are
//! building a dynamic object (either a shared library or an executable) that
//! may have symbol references that can only be resolved at load time. For a
//! variety of reasons (ELF symbol preemption, the CoW footprint of the section
//! that is modified by the loader), this results in all references to symbols
//! with external linkage to go via entries in the Global Offset Table (GOT),
//! which carries absolute addresses which need to be fixed up when the
//! executable image is loaded at an offset which is different from its link
//! time offset.
//!
//! Fortunately, there is a way to inform the compiler that such symbol
//! references will be satisfied at link time rather than at load time, by
//! giving them 'hidden' visibility.
//! SPDX-License-Identifier: GPL-2.0-only
//!
//! -*- linux-c -*- ------------------------------------------------------- *
//!
//! Copyright (C) 1991, 1992 Linus Torvalds
//! Copyright 2007 rPath, Inc. - All Rights Reserved
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
//! 64 bit
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
//! __sign_use for integer expressions:
//!   bit #0 set if ok for unsigned comparisons
//!   bit #1 set if ok for signed comparisons
//!
//! In particular, statically non-negative signed integer expressions
//! are ok for both.
//!
//! NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
//! in expressions, and are accepted for signed conversions for now.
//! This is debatable.
//!
//! Note that 'x' is the original expression, and 'ux' is the unique variable
//! that contains the value.
//!
//! We use 'ux' for pure type checking, and 'x' for when we need to look at the
//! value (but without evaluating it for side effects!
//! Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
//!
//! Pointers end up being checked by the normal C type rules at the actual
//! comparison, and these expressions only need to be careful to not cause
//! warnings for pointer use.
//! Value too large for defined data type
//! IS_ERR_VALUE - Detect an error pointer.
//! @x: The pointer to check.
//!
//! Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// Mark a position in code as unreachable.  This can be used to
// suppress control flow warnings after asm blocks that transfer
// control elsewhere.
// Format: __UNIQUE_ID_<name>_<__COUNTER__>
// &a[0] degrades to a pointer: a different type from an array
// Force the compiler to emit 'sym' as a symbol, so that we can reference
// it from inline assembler. Necessary in case 'sym' could be inlined
// otherwise, or eliminated entirely due to lack of references that are
// visible to the compiler.
// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
// bool and also pointer types.
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
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
/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__10 {
() => {
((core::mem::size_of::<[i8; 22]>() as u64) / (core::mem::size_of::<i8>() as u64)).wrapping_add(__must_be_array__2!() as u64)
};
}

macro_rules! PUTC {
($c:expr, $pos:expr, $size:expr, $buf:expr) => {
loop {
if ($pos) < $size {
unsafe {
*$buf.offset(($pos) as isize) = ($c) as i8;
}
}
let __v: u64 = $pos;
let __v2: u64 = __v.wrapping_add(1);
$pos = __v2;
if true {
break;
}
}
};
}

macro_rules! PUTC__2 {
($c:expr, $pos:expr, $size:expr, $buf:expr) => {
loop {
if ($pos) < $size {
unsafe {
*$buf.offset(($pos) as isize) = $c;
}
}
let __v: u64 = $pos;
let __v2: u64 = __v.wrapping_add(1);
$pos = __v2;
if true {
break;
}
}
};
}

macro_rules! __must_be_array__2 {
() => {
core::mem::size_of::<__SlateRecord_vsprintf_c_305>() as i32
};
}

macro_rules! barrier_before_unreachable {
() => {
while false {
}
};
}

macro_rules! va_arg__2 {
($v:expr) => {
unsafe { $v.next_arg::<*mut ()>() }
};
}

macro_rules! va_arg__5 {
($v:expr) => {
unsafe { (*core::ptr::addr_of_mut!($v)).next_arg::<u32>() }
};
}

macro_rules! va_arg__6 {
($v:expr) => {
unsafe { $v.next_arg::<u32>() }
};
}

macro_rules! va_arg__3 {
($v:expr) => {
unsafe { (*core::ptr::addr_of_mut!($v)).next_arg::<i64>() }
};
}

macro_rules! va_arg__4 {
($v:expr) => {
unsafe { (*core::ptr::addr_of_mut!($v)).next_arg::<u64>() }
};
}

macro_rules! va_arg {
($v:expr) => {
unsafe { (*core::ptr::addr_of_mut!($v)).next_arg::<i32>() }
};
}

macro_rules! va_arg__7 {
($v:expr) => {
unsafe { $v.next_arg::<i32>() }
};
}

macro_rules! va_copy {
($d:expr, $s:expr) => {
$d = $s.clone();
};
}

macro_rules! va_end {
() => {
};
}

unsafe extern "C" {
    fn strnlen(arg0: *const i8, arg1: u64) -> u64;
    fn __builtin_unreachable();
    fn __compiletime_assert_5();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_vsprintf_c_305 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

/// Optional: only supported since gcc >= 15, clang >= 19
///
///   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
/// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
fn isdigit(c: i32) -> i32 {
(48 <= c && c <= 57) as i32
}

#[unsafe(no_mangle)]
extern "C" fn vsnprintf(mut buf: *mut i8, size: u64, mut fmt: *const i8, mut ap: core::ffi::VaList<'static>) -> i32 {
let mut __UNIQUE_ID_y__4: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__3: u64 = unsafe { core::mem::zeroed() };
let mut clen: u64 = unsafe { core::mem::zeroed() };
let mut s8: *mut u8 = unsafe { core::mem::zeroed() };
let mut c32: u32 = unsafe { core::mem::zeroed() };
let mut ws: *const u16 = unsafe { core::mem::zeroed() };
let mut __v78: bool = unsafe { core::mem::zeroed() };
let mut args: core::mem::MaybeUninit<core::ffi::VaList<'static>> = core::mem::MaybeUninit::uninit();
let __slate_slot_79: *mut core::ffi::VaList<'static> = core::ptr::addr_of_mut!(args) as *mut core::ffi::VaList<'static>; // 'h', 'hh', 'l' or 'll' for integer fields
let mut qualifier: i32 = unsafe { core::mem::zeroed() };
// min. # of digits for integers; max
// number of chars for from string
let mut precision: i32 = unsafe { core::mem::zeroed() };
// width of output field
let mut field_width: i32 = unsafe { core::mem::zeroed() };
// flags to number()
let mut flags: i32 = unsafe { core::mem::zeroed() };
let mut sign: i8 = unsafe { core::mem::zeroed() };
let mut pos: u64 = unsafe { core::mem::zeroed() };
let mut len: u64 = unsafe { core::mem::zeroed() };
let mut s: *const i8 = unsafe { core::mem::zeroed() };
let mut base: i32 = unsafe { core::mem::zeroed() };
let mut num: i64 = unsafe { core::mem::zeroed() };
let mut tmp_end: *mut i8 = unsafe { core::mem::zeroed() };
// The maximum space required is to print a 64-bit number in octal
let mut tmp: core::mem::MaybeUninit<__SlateAlign16<[i8; 22]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_91: *mut [i8; 22] = core::ptr::addr_of_mut!(tmp) as *mut [i8; 22];
unsafe {
tmp_end = ((*__slate_slot_91).as_mut_ptr() as *mut i8).offset(ARRAY_SIZE__10!() as isize);
// We want to pass our input va_list to helper functions by reference,
// but there's an annoying edge case. If va_list was originally passed
// to us by value, we could just pass &ap down to the helpers. This is
// the case on, for example, X86_32.
// However, on X86_64 (and possibly others), va_list is actually a
// size-1 array containing a structure. Our function parameter ap has
// decayed from T[1] to T*, and &ap has type T** rather than T(*)[1],
// which is what will be expected by a function taking a va_list *
// parameter.
// One standard way to solve this mess is by creating a copy in a local
// variable of type va_list and then passing a pointer to that local
// copy instead, which is what we do here.
va_copy!(*__slate_slot_79, ap);
pos = 0;
'__loop_5: while *fmt != 0 {
if (*fmt as i32) != 37 {
__v78 = true;
} else {
let __v77: *const i8 = fmt;
let __v76: *const i8 = __v77.add(1);
fmt = __v76;
__v78 = (*__v76 as i32) == 37;
}
if __v78 {
PUTC__2!(unsafe { *fmt }, pos, size, buf);
} else {
// process flags
flags = get_flags(&mut (*core::ptr::addr_of_mut!(fmt)));
// get field width
field_width = get_int(&mut (*core::ptr::addr_of_mut!(fmt)), core::ptr::addr_of_mut!(*__slate_slot_79));
if field_width < 0 {
field_width = -field_width;
flags |= 16;
}
if flags & 16 != 0 {
let __v72: i32 = flags & !1;
flags = __v72;
}
// get the precision
precision = -1;
if (*fmt as i32) == 46 {
let __v71: *const i8 = fmt;
fmt = __v71.add(1);
precision = get_int(&mut (*core::ptr::addr_of_mut!(fmt)), core::ptr::addr_of_mut!(*__slate_slot_79));
if precision >= 0 {
let __v68: i32 = flags & !1;
flags = __v68;
}
}
// get the conversion qualifier
qualifier = -1;
if (*fmt as i32) == 104 || (*fmt as i32) == 108 {
qualifier = *fmt as i32;
let __v67: *const i8 = fmt;
fmt = __v67.add(1);
if qualifier == (*fmt as i32) {
qualifier -= 97 - 65;
let __v63: *const i8 = fmt;
fmt = __v63.add(1);
}
}
'__join_35: {
'__join_63: {
'__join_52: {
'__join_55: {
'__join_56: {
'__join_53: {
sign = 0;
let __t0: i32 = *fmt as i32;
if __t0 == 99 {
flags &= 16;
s = (*__slate_slot_91).as_mut_ptr() as *const i8;
if qualifier == 108 {
*((*__slate_slot_91).as_mut_ptr() as *mut u16).add(0) = va_arg__6!(*__slate_slot_79) as u16;
*((*__slate_slot_91).as_mut_ptr() as *mut u16).add(1) = 0;
precision = 2147483647i32;
break '__join_63;
} else {
(*__slate_slot_91)[0] = (va_arg__7!(*__slate_slot_79) as u8) as i8;
len = 1;
precision = 1;
break '__join_35;
}
} else {
if __t0 == 115 {
flags &= 16;
if precision < 0 {
precision = 2147483647i32;
}
s = va_arg__2!(*__slate_slot_79) as *const i8;
if s.is_null() {
s = (if precision < 6 { c"".as_ptr() as *mut i8 } else { c"(null)".as_ptr() as *mut i8 }) as *const i8;
} else {
if qualifier == 108 {
break '__join_63;
}
}
let __v54: u64 = strnlen(s, precision as u64);
len = __v54;
precision = __v54 as i32;
break '__join_35;
} else {
if __t0 == 111 {
base = 8;
break '__join_52;
} else {
if __t0 == 112 {
if precision < 0 {
precision = 2u64.wrapping_mul(core::mem::size_of::<*mut ()>() as u64) as i32;
}
break '__join_56;
} else {
if __t0 == 120 {
break '__join_56;
} else {
if __t0 == 88 {
break '__join_55;
} else {
if __t0 != 100 {
if __t0 != 105 {
if __t0 == 117 {
break '__join_53;
} else {
break '__loop_5;
}
}
}
}
}
}
}
}
}
flags |= 2;
}
let __v48: i32 = flags & !64;
flags = __v48;
base = 10;
break '__join_52;
}
flags |= 32;
}
base = 16;
}
if (*fmt as i32) == 112 {
num = (va_arg__2!(*__slate_slot_79) as u64) as i64;
} else {
num = get_number(flags & 2, qualifier, core::ptr::addr_of_mut!(*__slate_slot_79)) as i64;
}
sign = get_sign(&mut (*core::ptr::addr_of_mut!(num)), flags);
if sign != 0 {
field_width -= 1;
}
s = number(tmp_end, num as u64, base, (flags & 32) as i8) as *const i8;
len = (tmp_end.offset_from(s as *mut i8) as i64) as u64;
// default precision is 1
if precision < 0 {
precision = 1;
}
// precision is minimum number of digits to print
if (precision as u64) < len {
precision = len as i32;
}
if flags & 64 != 0 {
// For octal, a leading 0 is printed only if necessary,
// i.e. if it's not already there because of the
// precision.
if base == 8 && (precision as u64) == len {
precision += 1;
}
// For hexadecimal, the leading 0x is skipped if the
// output is empty, i.e. both the number and the
// precision are 0.
if base == 16 && precision > 0 {
field_width -= 2;
} else {
let __v42: i32 = flags & !64;
flags = __v42;
}
}
// For zero padding, increase the precision to fill the field
// width.
if flags & 1 != 0 && field_width > precision {
precision = field_width;
break '__join_35;
} else {
break '__join_35;
}
}
flags |= 128;
let __v55: u64 = utf16s_utf8nlen(s as *const u16, precision as u64);
len = __v55;
precision = __v55 as i32;
}
field_width -= precision; // Calculate the padding necessary
// Leading padding with ' '
if flags & 16 == 0 {
loop {
let __v37: i32 = field_width;
field_width = __v37 - 1;
if __v37 > 0 {
PUTC!(32i32, pos, size, buf);
} else {
break;
}
}
}
// sign
if sign != 0 {
PUTC__2!(sign, pos, size, buf);
}
// 0x/0X for hexadecimal
if flags & 64 != 0 {
PUTC!(48i32, pos, size, buf);
PUTC!(88 | flags & 32, pos, size, buf);
}
// Zero padding and excess precision
loop {
let __v35: i32 = precision;
precision = __v35 - 1;
if (__v35 as u64) > len {
PUTC!(48i32, pos, size, buf);
} else {
break;
}
}
// Actual output
if flags & 128 != 0 {
ws = s as *const u16;
loop {
let __v8: u64 = len;
len = __v8.wrapping_sub(1);
if __v8 > 0 {
c32 = utf16_to_utf32(&mut (*core::ptr::addr_of_mut!(ws)));
if c32 < 128 {
PUTC!(c32, pos, size, buf);
} else {
// Number of trailing octets
clen = (1 + ((c32 >= 2048) as i32) + ((c32 >= 65536) as i32)) as u64;
len = len.wrapping_sub(clen);
s8 = buf.add(pos as usize) as *mut u8;
// Avoid writing partial character
PUTC!(0i32, pos, size, buf);
pos = pos.wrapping_add(clen);
if pos < size {
// Set high bits of leading octet
*s8 = (1920i32 >> clen) as u8;
// Write trailing octets in reverse order
let __v19: *mut u8 = s8;
s8 = __v19.add(clen as usize);
while clen != 0 {
let __v17: *mut u8 = s8;
s8 = __v17.sub(1);
*__v17 = (128 | c32 & 63) as u8;
clen = clen.wrapping_sub(1);
c32 >>= 6i32;
}
// Set low bits of leading octet
*s8 = (((*s8 as i32) as u32) | c32) as u8;
}
}
} else {
break;
}
}
} else {
'__loop_21: loop {
let __v29: u64 = len;
len = __v29.wrapping_sub(1);
if __v29 > 0 {
loop {
if pos < size {
let __v33: *const i8 = s;
s = __v33.add(1);
*buf.add(pos as usize) = *__v33;
}
pos = pos.wrapping_add(1);
if true {
continue '__loop_21;
}
}
} else {
break;
}
}
}
// Trailing padding with ' '
loop {
let __v6: i32 = field_width;
field_width = __v6 - 1;
if __v6 > 0 {
PUTC!(32i32, pos, size, buf);
} else {
break;
}
}
}
let __v4: *const i8 = fmt;
fmt = __v4.add(1);
}
va_end!();
if size != 0 {
__UNIQUE_ID_x__3 = pos;
__UNIQUE_ID_y__4 = size.wrapping_sub(1);
while false {
}
*buf.add((if __UNIQUE_ID_x__3 < __UNIQUE_ID_y__4 { __UNIQUE_ID_x__3 } else { __UNIQUE_ID_y__4 }) as usize) = 0;
}
return pos as i32;
}
// integer number formats - set up the flags and "break"
// Bail out if the conversion specifier is invalid.
// There's probably a typo in the format string and the
// remaining specifiers are unlikely to match up with
// the arguments.
}

// Oh, it's a waste of space, but oh-so-yummy for debugging.
fn skip_atoi(s: &mut *const i8) -> i32 {
let mut i: i32 = 0;
while isdigit((unsafe { *(*(s as *mut *const i8)) }) as i32) != 0 {
let __v1: *const i8 = unsafe { *s };
unsafe {
*s = __v1.add(1);
}
i = i * 10 + ((unsafe { *__v1 }) as i32) - 48;
}
i
}

/// put_dec_full4 handles numbers in the range 0 <= r < 10000.
/// The multiplier 0xccd is round(2^15/10), and the approximation
/// r/10 == (r * 0xccd) >> 15 is exact for all r < 16389.
fn put_dec_full4(mut end: *mut i8, mut r: u32) {
for i in 0i32..3i32 {
let q: u32 = r.wrapping_mul(3277) >> 15i32;
let __v0: *mut i8 = end;
let __v1: *mut i8 = unsafe { __v0.sub(1) };
end = __v1;
unsafe {
*__v1 = 48u32.wrapping_add(r.wrapping_sub(q.wrapping_mul(10))) as i8;
}
r = q;
}
let __v5: *mut i8 = end;
let __v6: *mut i8 = unsafe { __v5.sub(1) };
end = __v6;
unsafe {
*__v6 = 48u32.wrapping_add(r) as i8;
}
}

// put_dec is copied from lib/vsprintf.c with small modifications
/// Call put_dec_full4 on x % 10000, return x / 10000.
/// The approximation x/10000 == (x * 0x346DC5D7) >> 43
/// holds for all x < 1,128,869,999.  The largest value this
/// helper will ever be asked to convert is 1,125,520,955.
/// (second call in the put_dec code, assuming n is all-ones).
fn put_dec_helper4(end: *mut i8, x: u32) -> u32 {
let q: u32 = ((x as u64).wrapping_mul(879609303) >> 43i32) as u32;
put_dec_full4(end, x.wrapping_sub(q.wrapping_mul(10000)));
q
}

/// Based on code by Douglas W. Jones found at
/// <http://www.cs.uiowa.edu/~jones/bcd/decimal.html#sixtyfour>
/// (with permission from the author).
/// Performs no 64-bit division and hence should be fast on 32-bit machines.
fn put_dec(end: *mut i8, n: u64) -> *mut i8 {
let mut d3: u32 = 0;
let mut d2: u32 = 0;
let mut d1: u32 = 0;
let mut q: u32 = 0;
let mut h: u32 = 0;
let mut p: *mut i8 = end;
d1 = (n as u32) >> 16i32; // implicit "& 0xffff"
h = (n >> 32i32) as u32;
d2 = h & 65535;
d3 = h >> 16i32; // implicit "& 0xffff"
// n = 2^48 d3 + 2^32 d2 + 2^16 d1 + d0
// = 281_4749_7671_0656 d3 + 42_9496_7296 d2 + 6_5536 d1 + d0
q = 656u32.wrapping_mul(d3).wrapping_add(7296u32.wrapping_mul(d2)).wrapping_add(5536u32.wrapping_mul(d1)).wrapping_add((n as u32) & 65535);
q = put_dec_helper4(p, q);
let __v0: *mut i8 = p;
p = unsafe { __v0.sub(4) };
q = q.wrapping_add(7671u32.wrapping_mul(d3).wrapping_add(9496u32.wrapping_mul(d2)).wrapping_add(6u32.wrapping_mul(d1)));
q = put_dec_helper4(p, q);
let __v4: *mut i8 = p;
p = unsafe { __v4.sub(4) };
q = q.wrapping_add(4749u32.wrapping_mul(d3).wrapping_add(42u32.wrapping_mul(d2)));
q = put_dec_helper4(p, q);
let __v8: *mut i8 = p;
p = unsafe { __v8.sub(4) };
q = q.wrapping_add(281u32.wrapping_mul(d3));
q = put_dec_helper4(p, q);
let __v12: *mut i8 = p;
p = unsafe { __v12.sub(4) };
put_dec_full4(p, q);
let __v14: *mut i8 = p;
p = unsafe { __v14.sub(4) };
// strip off the extra 0's we printed
while p < end && ((unsafe { *p }) as i32) == 48 {
let __v16: *mut i8 = p;
p = unsafe { __v16.add(1) };
}
p
}

fn number(mut end: *mut i8, mut num: u64, base: i32, locase: i8) -> *mut i8 {
// locase = 0 or 0x20. ORing digits or letters with 'locase'
// produces same digits or (maybe lowercased) letters
// we are called with base 8, 10 or 16, only, thus don't need "G..."
// "GHIJKLMNOPQRSTUVWXYZ";
match base {
10 => {
if num != 0 {
end = put_dec(end, num);
}
}
8 => {
while num != 0 {
let __v1: *mut i8 = end;
let __v2: *mut i8 = unsafe { __v1.sub(1) };
end = __v2;
unsafe {
*__v2 = 48u64.wrapping_add(num & 7) as i8;
}
num >>= 3i32;
}
}
16 => {
while num != 0 {
let __v6: *mut i8 = end;
let __v7: *mut i8 = unsafe { __v6.sub(1) };
end = __v7;
unsafe {
*__v7 = ((digits.0[(num & 15) as usize] as i32) | (locase as i32)) as i8;
}
num >>= 4i32;
}
}
_ => {
loop {
barrier_before_unreachable!();
unsafe { core::hint::unreachable_unchecked() };
if true {
break;
}
}
}
}
end
}

static digits: __SlateAlign16<[i8; 16]> = __SlateAlign16([48i8, 49i8, 50i8, 51i8, 52i8, 53i8, 54i8, 55i8, 56i8, 57i8, 65i8, 66i8, 67i8, 68i8, 69i8, 70i8]);

// pad with zero
// unsigned/signed long
// show plus
// space if plus
// left justified
// Must be 32 == 0x20
// 0x
// UTF-16 string
fn get_flags(fmt: &mut *const i8) -> i32 {
let mut flags: i32 = 0;
loop {
match (unsafe { *(*(fmt as *mut *const i8)) }) as i32 {
45 => {
flags |= 16;
}
43 => {
flags |= 4;
}
32 => {
flags |= 8;
}
35 => {
flags |= 64;
}
48 => {
flags |= 1;
}
_ => {
return flags;
}
}
let __v12: *const i8 = unsafe { *fmt };
unsafe {
*fmt = __v12.add(1);
}
if false {
break;
}
}
unsafe { core::mem::zeroed() }
}

fn get_int(fmt: &mut *const i8, ap: *mut core::ffi::VaList<'static>) -> i32 {
if isdigit((unsafe { *(*(fmt as *mut *const i8)) }) as i32) != 0 {
return skip_atoi(fmt);
}
if ((unsafe { *(*(fmt as *mut *const i8)) }) as i32) == 42 {
let __v1: *const i8 = unsafe { *fmt };
unsafe {
*fmt = __v1.add(1);
}
// it's the next argument
return va_arg!(*ap);
}
0
}

fn get_number(sign: i32, qualifier: i32, ap: *mut core::ffi::VaList<'static>) -> u64 {
if sign != 0 {
match qualifier {
76 => {
return va_arg__3!(*ap) as u64;
}
108 => {
return va_arg__3!(*ap) as u64;
}
104 => {
return (va_arg!(*ap) as i16) as u64;
}
72 => {
return (va_arg!(*ap) as i8) as u64;
}
_ => {
return va_arg!(*ap) as u64;
}
}
} else {
match qualifier {
76 => {
return va_arg__4!(*ap);
}
108 => {
return va_arg__4!(*ap);
}
104 => {
return (va_arg!(*ap) as u16) as u64;
}
72 => {
return (va_arg!(*ap) as u8) as u64;
}
_ => {
return va_arg__5!(*ap) as u64;
}
}
}
unsafe { core::mem::zeroed() }
}

fn get_sign(num: &mut i64, flags: i32) -> i8 {
if flags & 2 == 0 {
return 0;
}
if (unsafe { *(num as *mut i64) }) < 0 {
unsafe {
*(num as *mut i64) = -(*(num as *mut i64));
}
return 45;
}
if flags & 4 != 0 {
return 43;
}
if flags & 8 != 0 {
return 32;
}
0
}

fn utf16s_utf8nlen(mut s16: *const u16, maxlen: u64) -> u64 {
let mut len: u64 = 0;
let mut clen: u64 = 0;
while len < maxlen && (unsafe { *s16 }) != 0 {
let mut c0: u16 = 0;
let __v0: *const u16 = s16;
s16 = unsafe { __v0.add(1) };
c0 = unsafe { *__v0 };
// First, get the length for a BMP character
clen = (1 + (((c0 as i32) >= 128) as i32) + (((c0 as i32) >= 2048) as i32)) as u64;
if len.wrapping_add(clen) > maxlen {
break;
}
// If this is a high surrogate, and we're already at maxlen, we
// can't include the character if it's a valid surrogate pair.
// Avoid accessing one extra word just to check if it's valid
// or not.
if (c0 as i32) & 64512 == 55296 {
if len.wrapping_add(clen) == maxlen {
break;
}
if ((unsafe { *s16 }) as i32) & 64512 == 56320 {
let __v3: *const u16 = s16;
s16 = unsafe { __v3.add(1) };
clen = clen.wrapping_add(1);
}
}
len = len.wrapping_add(clen);
}
len
}

fn utf16_to_utf32(s16: &mut *const u16) -> u32 {
let mut c0: u16 = 0;
let mut c1: u16 = 0;
let __v1: *const u16 = unsafe { *s16 };
unsafe {
*s16 = __v1.add(1);
}
c0 = unsafe { *__v1 };
// not a surrogate
if (c0 as i32) & 63488 != 55296 {
return c0 as u32;
}
// invalid: low surrogate instead of high
if (c0 as i32) & 1024 != 0 {
return 65533;
}
c1 = unsafe { *(*(s16 as *mut *const u16)) };
// invalid: missing low surrogate
if (c1 as i32) & 64512 != 56320 {
return 65533;
}
// valid surrogate pair
let __v4: *const u16 = unsafe { *s16 };
unsafe {
*s16 = __v4.add(1);
}
(-56613888i32 + ((c0 as i32) << 10i32) + (c1 as i32)) as u32
}

#[unsafe(no_mangle)]
unsafe extern "C" fn snprintf(buf: *mut i8, size: u64, fmt: *const i8, mut __va_args: ...) -> i32 {
let mut args: core::ffi::VaList<'static> = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
args = unsafe { core::mem::transmute::<core::ffi::VaList<'_>, core::ffi::VaList<'static>>(__va_args.clone()) };
i = vsnprintf(buf, size, fmt, args.clone());
va_end!();
i
}
