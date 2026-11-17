
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0-only
//!
//! linux/lib/cmdline.c
//! Helper functions generally used for parsing kernel command line
//! and module options.
//!
//! Code and copyrights come from init/main.c and arch/i386/kernel/setup.c.
//!
//! GNU Indent formatting options for this file: -kr -i8 -npsl -pcs
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
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
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
macro_rules! __ismask {
($x:expr) => {
unsafe { *(core::ptr::addr_of!(_ctype) as *const u8).wrapping_offset(((($x) as u8) as i32) as isize) }
};
}

/// check_shl_overflow() - Calculate a left-shifted value and check overflow
/// @a: Value to be shifted
/// @s: How many bits left to shift
/// @d: Pointer to where to store the result
///
/// Computes *@d = (@a << @s)
///
/// Returns true if '*@d' cannot hold the result or when '@a << @s' doesn't
/// make sense. Example conditions:
///
/// - '@a << @s' causes bits to be lost when stored in *@d.
/// - '@s' is garbage (e.g. negative) or so large that the result of
///   '@a << @s' is guaranteed to be 0.
/// - '@a' is negative.
/// - '@a << @s' sets the sign bit, if any, in '*@d'.
///
/// '*@d' will hold the results of the attempted shift, but is not
/// considered "safe for use" if true is returned.
macro_rules! check_shl_overflow {
($a:expr, $s:expr, $d:expr) => {
{
let mut __v: bool;
let __v2: u64 = $a;
let __v3: u32 = $s;
let __v4: *mut u64 = $d;
let __v5: u64 = __v2;
let __v6: u32 = if __v3 >= 0 && (__v3 as u64) < 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64) { __v3 } else { 0 };
unsafe {
*__v4 = __v5 << __v6;
}
__v = __v6 != __v3 || (unsafe { *__v4 }) < 0 || __v2 < 0 || (unsafe { *__v4 }) >> __v6 != __v2;
__must_check_overflow(__v)
}
};
}

/// Note: isspace() must return false for %NUL-terminator
macro_rules! isspace {
($c:expr) => {
(__ismask!($c) as i32) & 32 != 0
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
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
    fn simple_strtol(arg0: *const u8, arg1: *mut *mut u8, arg2: u32) -> i64;
    fn simple_strtoull(arg0: *const u8, arg1: *mut *mut u8, arg2: u32) -> u64;
    fn strlen(arg0: *const u8) -> u64;
    fn strncmp(arg0: *const u8, arg1: *const u8, arg2: u64) -> i32;
    fn skip_spaces(arg0: *const u8) -> *mut u8;
}

#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
}

// check_add_overflow() - Calculate addition with overflow checking
// @a: first addend
// @b: second addend
// @d: pointer to store sum
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted addition, regardless of whether
// wrap-around occurred.
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
// If a hyphen was found in get_option, this will handle the
// range of numbers, M-N.  This will expand the range and insert
// the values[M, M+1, ..., N] into the ints array in get_options.
fn get_range(str: &mut *mut u8, mut pint: *mut i32, mut n: i32) -> i32 {
let mut x: i32 = 0;
let mut inc_counter: i32 = 0;
let mut upper_range: i32 = 0;
let __v1: *mut u8 = unsafe { *str };
unsafe {
*str = __v1.wrapping_offset(1 as isize);
}
upper_range = (unsafe { simple_strtol(*(str as *mut *mut u8) as *const u8, core::ptr::null_mut::<*mut u8>(), 0u32) }) as i32;
inc_counter = upper_range.wrapping_sub(unsafe { *pint });
x = unsafe { *pint };
while n != 0 && x < upper_range {
let __v3: *mut i32 = pint;
pint = __v3.wrapping_offset(1 as isize);
unsafe {
*__v3 = x;
}
x = x.wrapping_add(1);
n = n.wrapping_sub(1);
}
inc_counter
}

// get_option - Parse integer from an option string
// @str: option string
// @pint: (optional output) integer value parsed from @str
//
// Read an int from an option string; if available accept a subsequent
// comma as well.
//
// When @pint is NULL the function can be used as a validator of
// the current option in the string.
//
// Return:
// 0 - no int in string
// 1 - int found, no subsequent comma
// 2 - int found including a subsequent comma
// 3 - hyphen found to denote a range
//
// Leading hyphen without integer is no integer case, but we consume it
// for the sake of simplification.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.cmdline.get_option")]
extern "C" fn get_option(str: *mut *mut u8, pint: *mut i32) -> i32 {
let mut cur: *mut u8 = unsafe { *str };
let mut value: i32 = 0;
if cur.is_null() || (unsafe { *cur }) == 0 {
return 0;
}
if ((unsafe { *cur }) as i32) == 45 {
let __v0: *mut u8 = cur;
let __v1: *mut u8 = __v0.wrapping_offset(1 as isize);
cur = __v1;
value = unsafe { simple_strtoull(__v1 as *const u8, str, 0u32) }.wrapping_neg() as i32;
} else {
value = (unsafe { simple_strtoull(cur as *const u8, str, 0u32) }) as i32;
}
if !pint.is_null() {
unsafe {
*pint = value;
}
}
if cur == unsafe { *str } {
return 0;
}
if ((unsafe { *(*str) }) as i32) == 44 {
let __v3: *mut u8 = unsafe { *str };
unsafe {
*str = __v3.wrapping_offset(1 as isize);
}
return 2;
}
if ((unsafe { *(*str) }) as i32) == 45 {
return 3;
}
1
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_get_option_5: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut *mut u8, *mut i32) -> i32>, *mut ()>(Some(get_option)) };

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_get_option: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_get_options: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_memparse: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_next_arg: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous", sym get_option, sym get_options, sym memparse, sym next_arg, options(att_syntax));

// get_options - Parse a string into a list of integers
// @str: String to be parsed
// @nints: size of integer array
// @ints: integer array (must have room for at least one element)
//
// This function parses a string containing a comma-separated
// list of integers, a hyphen-separated range of _positive_ integers,
// or a combination of both.  The parse halts when the array is
// full, or when no more numbers can be retrieved from the
// string.
//
// When @nints is 0, the function just validates the given @str and
// returns the amount of parseable integers as described below.
//
// Returns:
//
// The first element is filled by the number of collected integers
// in the range. The rest is what was parsed from the @str.
//
// Return value is the character in the string which caused
// the parse to end (typically a null terminator, if @str is
// completely parseable).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.cmdline.get_options")]
extern "C" fn get_options(mut str: *const u8, nints: i32, ints: *mut i32) -> *mut u8 {
let validate: bool = nints == 0;
let mut res: i32 = 0;
let mut i: i32 = 1;
while i < nints || validate {
let pint: *mut i32 = if validate { ints } else { ints.wrapping_offset(i as isize) };
res = get_option(core::ptr::addr_of_mut!(str) as *mut *mut u8, pint);
if res == 0 {
break;
}
if res == 3 {
let n: i32 = if validate { 0 } else { nints.wrapping_sub(i) };
let range_nums: i32 = get_range(unsafe { &mut (*(core::ptr::addr_of_mut!(str) as *mut *mut u8)) }, pint, n);
if range_nums < 0 {
break;
}
// Decrement the result by one to leave out the
// last number in the range.  The next iteration
// will handle the upper number in the range
i = i.wrapping_add(range_nums.wrapping_sub(1));
}
i = i.wrapping_add(1);
if res == 1 {
break;
}
}
unsafe {
*ints.wrapping_offset(0 as isize) = i.wrapping_sub(1);
}
str as *mut u8
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_get_options_6: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, i32, *mut i32) -> *mut u8>, *mut ()>(Some(get_options)) };

// memparse - parse a string with mem suffixes into a number
// @ptr: Where parse begins
// @retptr: (output) Optional pointer to next char after parse completes
//
// Parses a string into a number.  The number stored at @ptr is
// potentially suffixed with K, M, G, T, P, E.
//
// Return: The value as recognized by simple_strtoull() multiplied
// by the value as specified by suffix, if any.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.cmdline.memparse")]
extern "C" fn memparse(ptr: *const u8, retptr: *mut *mut u8) -> u64 {
let mut endptr: *mut u8 = unsafe { core::mem::zeroed() }; // local pointer to end of parsed string
let mut ret: u64 = unsafe { simple_strtoull(ptr, core::ptr::addr_of_mut!(endptr), 0u32) };
let mut shl: u32 = 0;
// Consume valid suffix even in case of overflow.
let mut __t0: i64 = match (unsafe { *endptr }) as i32 { 69 | 101 => 0, 80 | 112 => 1, 84 | 116 => 2, 71 | 103 => 3, 77 | 109 => 4, 75 | 107 => 5, _ => 6 };
loop {
match __t0 {
0 => {
shl = shl.wrapping_add(10);
__t0 = 1;
continue;
}
1 => {
shl = shl.wrapping_add(10);
__t0 = 2;
continue;
}
2 => {
shl = shl.wrapping_add(10);
__t0 = 3;
continue;
}
3 => {
shl = shl.wrapping_add(10);
__t0 = 4;
continue;
}
4 => {
shl = shl.wrapping_add(10);
__t0 = 5;
continue;
}
5 => {
shl = shl.wrapping_add(10);
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
if shl != 0 && likely!(ptr != (endptr as *const u8)) != 0 {
// Have valid suffix with preceding number.
if unlikely!(check_shl_overflow!(ret, shl, core::ptr::addr_of_mut!(ret))) != 0 {
ret = !0;
}
let __v19: *mut u8 = endptr;
endptr = __v19.wrapping_offset(1 as isize);
}
if !retptr.is_null() {
unsafe {
*retptr = endptr;
}
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_memparse_7: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *mut *mut u8) -> u64>, *mut ()>(Some(memparse)) };

/// parse_option_str - Parse a string and check an option is set or not
/// @str: String to be parsed
/// @option: option name
///
/// This function parses a string containing a comma-separated list of
/// strings like a=b,c.
///
/// Return: True if there's such option in the string or false otherwise.
#[unsafe(no_mangle)]
extern "C" fn parse_option_str(mut str: *const u8, option: *const u8) -> bool {
while (unsafe { *str }) != 0 {
if (unsafe { strncmp(str, option, strlen(option)) }) == 0 {
let __v0: u64 = unsafe { strlen(option) };
let __v1: *const u8 = str;
str = __v1.wrapping_offset(__v0 as isize);
if (unsafe { *str }) == 0 || ((unsafe { *str }) as i32) == 44 {
return true;
}
}
while (unsafe { *str }) != 0 && ((unsafe { *str }) as i32) != 44 {
let __v3: *const u8 = str;
str = __v3.wrapping_offset(1 as isize);
}
if ((unsafe { *str }) as i32) == 44 {
let __v6: *const u8 = str;
str = __v6.wrapping_offset(1 as isize);
}
}
false
}

/// Parse a string to get a param value pair.
/// You can use " around spaces, but can't escape ".
/// Hyphens and underscores equivalent in parameter names.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.cmdline.next_arg")]
extern "C" fn next_arg(mut args: *mut u8, param: *mut *mut u8, val: *mut *mut u8) -> *mut u8 {
let mut i: u32 = 0;
let mut equals: u32 = 0;
let mut in_quote: i32 = 0;
let mut quoted: i32 = 0;
if ((unsafe { *args }) as i32) == 34 {
let __v0: *mut u8 = args;
args = __v0.wrapping_offset(1 as isize);
in_quote = 1;
quoted = 1;
}
while (unsafe { *args.wrapping_offset(i as isize) }) != 0 {
if isspace!(unsafe { *args.wrapping_offset(i as isize) }) && in_quote == 0 {
break;
}
if equals == 0 {
if ((unsafe { *args.wrapping_offset(i as isize) }) as i32) == 61 {
equals = i;
}
}
if ((unsafe { *args.wrapping_offset(i as isize) }) as i32) == 34 {
in_quote = (in_quote == 0) as i32;
}
i = i.wrapping_add(1);
}
unsafe {
*param = args;
}
if equals == 0 {
unsafe {
*val = core::ptr::null_mut::<u8>();
}
} else {
unsafe {
*args.wrapping_offset(equals as isize) = 0;
}
unsafe {
*val = args.wrapping_offset(equals as isize).wrapping_offset(1 as isize);
}
// Don't include quotes in value.
if ((unsafe { *(*val) }) as i32) == 34 {
let __v6: *mut u8 = unsafe { *val };
unsafe {
*val = __v6.wrapping_offset(1 as isize);
}
if ((unsafe { *args.wrapping_offset(i.wrapping_sub(1) as isize) }) as i32) == 34 {
unsafe {
*args.wrapping_offset(i.wrapping_sub(1) as isize) = 0;
}
}
}
}
if quoted != 0 && i > 0 && ((unsafe { *args.wrapping_offset(i.wrapping_sub(1) as isize) }) as i32) == 34 {
unsafe {
*args.wrapping_offset(i.wrapping_sub(1) as isize) = 0;
}
}
if (unsafe { *args.wrapping_offset(i as isize) }) != 0 {
unsafe {
*args.wrapping_offset(i as isize) = 0;
}
let __v8: *mut u8 = args;
args = __v8.wrapping_offset(i.wrapping_add(1) as isize);
} else {
let __v10: *mut u8 = args;
args = __v10.wrapping_offset(i as isize);
}
// Chew up trailing spaces.
unsafe { skip_spaces(args as *const u8) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_next_arg_8: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u8, *mut *mut u8, *mut *mut u8) -> *mut u8>, *mut ()>(Some(next_arg)) };
