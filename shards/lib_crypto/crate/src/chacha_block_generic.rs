
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0-or-later
//!
//! The "hash function" used as the core of the ChaCha stream cipher (RFC7539)
//!
//! Copyright (C) 2015 Martin Willi
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
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! __get_unaligned_t - read an unaligned value from memory.
//! @type:	the type to load from the pointer.
//! @ptr:	the pointer to load from.
//!
//! Use memcpy to affect an unaligned type sized load avoiding undefined behavior
//! from approaches like type punning that require -fno-strict-aliasing in order
//! to be correct. As type may be const, use __unqual_scalar_typeof to map to a
//! non-const type - you can't memcpy into a const type. The
//! __get_unaligned_ctrl_type gives __unqual_scalar_typeof its required
//! expression rather than type, a pointer is used to avoid warnings about mixing
//! the use of 0 and NULL. The void* cast silences ubsan warnings.
//! This counts to 15. Any more, it will return 16th argument.
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
//! 32-bit stream position, then 96-bit nonce (RFC7539 convention)
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
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
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
// &a[0] degrades to a pointer: a different type from an array
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// new instruction
// feature + flags
// source len
// replacement len
// replacement
// alternative assembly primitive:
// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
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
/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__9 {
() => {
((core::mem::size_of::<[u32; 16]>() as u64) / (core::mem::size_of::<u32>() as u64)).wrapping_add(__must_be_array__2!() as u64)
};
}

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

macro_rules! __must_be_array__2 {
() => {
core::mem::size_of::<__SlateRecord_chacha_block_generic_c_85>() as i32
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
($type:ty, $val:expr, $ptr:expr) => {
loop {
let mut __put_unaligned_val: $type = $val;
unsafe { __builtin_memcpy($ptr, core::ptr::addr_of_mut!(__put_unaligned_val) as *const (), core::mem::size_of::<$type>() as u64) };
if true {
break;
}
}
};
}

macro_rules! unlikely {
($x:expr) => {
(($x) != 0) as i64
};
}

unsafe extern "C" {
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
}

// Optional: only supported since gcc >= 15, clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
fn put_unaligned_le32(val: u32, p: *mut ()) {
__put_unaligned_t!(u32, val, p);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_chacha_block_generic_c_85 {
}


fn memzero_explicit(mut s: *mut (), count: u64) {
unsafe { memset(s, 0i32, count) };
unsafe {
core::arch::asm!("\n# {0:r}", in(reg) s as u64, options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct chacha_state {
    x: [u32; 16],
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_chacha_block_generic: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_hchacha_block_generic: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous", sym chacha_block_generic, sym hchacha_block_generic, options(att_syntax));

fn chacha_zeroize_state(state: *mut chacha_state) {
memzero_explicit(state as *mut (), core::mem::size_of::<chacha_state>() as u64);
}

// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
fn rol32(word: u32, shift: u32) -> u32 {
word << (shift & 31) | word >> (shift.wrapping_neg() & 31)
}

fn chacha_permute(state: &mut chacha_state, nrounds: i32) {
let x: *mut u32 = unsafe { (*(state as *mut chacha_state)).x.as_mut_ptr() as *mut u32 };
let mut i: i32 = 0;
// whitelist the allowed round counts
let mut __v0: i64;
let __ret_warn_on: i32 = (nrounds != 20 && nrounds != 12) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("5: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 5b - ., 3; .popsection\n# {0}", const 5 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_chacha_permute_1, sym __slate_asm_string_chacha_permute_2, const 22 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("6: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 6b - ., 4; .popsection\n# {0}", const 6 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
while i < nrounds {
let __v5: *mut u32 = x.wrapping_offset(0 as isize);
let __v6: u32 = unsafe { *__v5 };
unsafe {
*__v5 = __v6.wrapping_add(*x.wrapping_offset(4 as isize));
}
unsafe {
*x.wrapping_offset(12 as isize) = rol32(*x.wrapping_offset(12 as isize) ^ *x.wrapping_offset(0 as isize), 16u32);
}
let __v8: *mut u32 = x.wrapping_offset(1 as isize);
let __v9: u32 = unsafe { *__v8 };
unsafe {
*__v8 = __v9.wrapping_add(*x.wrapping_offset(5 as isize));
}
unsafe {
*x.wrapping_offset(13 as isize) = rol32(*x.wrapping_offset(13 as isize) ^ *x.wrapping_offset(1 as isize), 16u32);
}
let __v11: *mut u32 = x.wrapping_offset(2 as isize);
let __v12: u32 = unsafe { *__v11 };
unsafe {
*__v11 = __v12.wrapping_add(*x.wrapping_offset(6 as isize));
}
unsafe {
*x.wrapping_offset(14 as isize) = rol32(*x.wrapping_offset(14 as isize) ^ *x.wrapping_offset(2 as isize), 16u32);
}
let __v14: *mut u32 = x.wrapping_offset(3 as isize);
let __v15: u32 = unsafe { *__v14 };
unsafe {
*__v14 = __v15.wrapping_add(*x.wrapping_offset(7 as isize));
}
unsafe {
*x.wrapping_offset(15 as isize) = rol32(*x.wrapping_offset(15 as isize) ^ *x.wrapping_offset(3 as isize), 16u32);
}
let __v17: *mut u32 = x.wrapping_offset(8 as isize);
let __v18: u32 = unsafe { *__v17 };
unsafe {
*__v17 = __v18.wrapping_add(*x.wrapping_offset(12 as isize));
}
unsafe {
*x.wrapping_offset(4 as isize) = rol32(*x.wrapping_offset(4 as isize) ^ *x.wrapping_offset(8 as isize), 12u32);
}
let __v20: *mut u32 = x.wrapping_offset(9 as isize);
let __v21: u32 = unsafe { *__v20 };
unsafe {
*__v20 = __v21.wrapping_add(*x.wrapping_offset(13 as isize));
}
unsafe {
*x.wrapping_offset(5 as isize) = rol32(*x.wrapping_offset(5 as isize) ^ *x.wrapping_offset(9 as isize), 12u32);
}
let __v23: *mut u32 = x.wrapping_offset(10 as isize);
let __v24: u32 = unsafe { *__v23 };
unsafe {
*__v23 = __v24.wrapping_add(*x.wrapping_offset(14 as isize));
}
unsafe {
*x.wrapping_offset(6 as isize) = rol32(*x.wrapping_offset(6 as isize) ^ *x.wrapping_offset(10 as isize), 12u32);
}
let __v26: *mut u32 = x.wrapping_offset(11 as isize);
let __v27: u32 = unsafe { *__v26 };
unsafe {
*__v26 = __v27.wrapping_add(*x.wrapping_offset(15 as isize));
}
unsafe {
*x.wrapping_offset(7 as isize) = rol32(*x.wrapping_offset(7 as isize) ^ *x.wrapping_offset(11 as isize), 12u32);
}
let __v29: *mut u32 = x.wrapping_offset(0 as isize);
let __v30: u32 = unsafe { *__v29 };
unsafe {
*__v29 = __v30.wrapping_add(*x.wrapping_offset(4 as isize));
}
unsafe {
*x.wrapping_offset(12 as isize) = rol32(*x.wrapping_offset(12 as isize) ^ *x.wrapping_offset(0 as isize), 8u32);
}
let __v32: *mut u32 = x.wrapping_offset(1 as isize);
let __v33: u32 = unsafe { *__v32 };
unsafe {
*__v32 = __v33.wrapping_add(*x.wrapping_offset(5 as isize));
}
unsafe {
*x.wrapping_offset(13 as isize) = rol32(*x.wrapping_offset(13 as isize) ^ *x.wrapping_offset(1 as isize), 8u32);
}
let __v35: *mut u32 = x.wrapping_offset(2 as isize);
let __v36: u32 = unsafe { *__v35 };
unsafe {
*__v35 = __v36.wrapping_add(*x.wrapping_offset(6 as isize));
}
unsafe {
*x.wrapping_offset(14 as isize) = rol32(*x.wrapping_offset(14 as isize) ^ *x.wrapping_offset(2 as isize), 8u32);
}
let __v38: *mut u32 = x.wrapping_offset(3 as isize);
let __v39: u32 = unsafe { *__v38 };
unsafe {
*__v38 = __v39.wrapping_add(*x.wrapping_offset(7 as isize));
}
unsafe {
*x.wrapping_offset(15 as isize) = rol32(*x.wrapping_offset(15 as isize) ^ *x.wrapping_offset(3 as isize), 8u32);
}
let __v41: *mut u32 = x.wrapping_offset(8 as isize);
let __v42: u32 = unsafe { *__v41 };
unsafe {
*__v41 = __v42.wrapping_add(*x.wrapping_offset(12 as isize));
}
unsafe {
*x.wrapping_offset(4 as isize) = rol32(*x.wrapping_offset(4 as isize) ^ *x.wrapping_offset(8 as isize), 7u32);
}
let __v44: *mut u32 = x.wrapping_offset(9 as isize);
let __v45: u32 = unsafe { *__v44 };
unsafe {
*__v44 = __v45.wrapping_add(*x.wrapping_offset(13 as isize));
}
unsafe {
*x.wrapping_offset(5 as isize) = rol32(*x.wrapping_offset(5 as isize) ^ *x.wrapping_offset(9 as isize), 7u32);
}
let __v47: *mut u32 = x.wrapping_offset(10 as isize);
let __v48: u32 = unsafe { *__v47 };
unsafe {
*__v47 = __v48.wrapping_add(*x.wrapping_offset(14 as isize));
}
unsafe {
*x.wrapping_offset(6 as isize) = rol32(*x.wrapping_offset(6 as isize) ^ *x.wrapping_offset(10 as isize), 7u32);
}
let __v50: *mut u32 = x.wrapping_offset(11 as isize);
let __v51: u32 = unsafe { *__v50 };
unsafe {
*__v50 = __v51.wrapping_add(*x.wrapping_offset(15 as isize));
}
unsafe {
*x.wrapping_offset(7 as isize) = rol32(*x.wrapping_offset(7 as isize) ^ *x.wrapping_offset(11 as isize), 7u32);
}
let __v53: *mut u32 = x.wrapping_offset(0 as isize);
let __v54: u32 = unsafe { *__v53 };
unsafe {
*__v53 = __v54.wrapping_add(*x.wrapping_offset(5 as isize));
}
unsafe {
*x.wrapping_offset(15 as isize) = rol32(*x.wrapping_offset(15 as isize) ^ *x.wrapping_offset(0 as isize), 16u32);
}
let __v56: *mut u32 = x.wrapping_offset(1 as isize);
let __v57: u32 = unsafe { *__v56 };
unsafe {
*__v56 = __v57.wrapping_add(*x.wrapping_offset(6 as isize));
}
unsafe {
*x.wrapping_offset(12 as isize) = rol32(*x.wrapping_offset(12 as isize) ^ *x.wrapping_offset(1 as isize), 16u32);
}
let __v59: *mut u32 = x.wrapping_offset(2 as isize);
let __v60: u32 = unsafe { *__v59 };
unsafe {
*__v59 = __v60.wrapping_add(*x.wrapping_offset(7 as isize));
}
unsafe {
*x.wrapping_offset(13 as isize) = rol32(*x.wrapping_offset(13 as isize) ^ *x.wrapping_offset(2 as isize), 16u32);
}
let __v62: *mut u32 = x.wrapping_offset(3 as isize);
let __v63: u32 = unsafe { *__v62 };
unsafe {
*__v62 = __v63.wrapping_add(*x.wrapping_offset(4 as isize));
}
unsafe {
*x.wrapping_offset(14 as isize) = rol32(*x.wrapping_offset(14 as isize) ^ *x.wrapping_offset(3 as isize), 16u32);
}
let __v65: *mut u32 = x.wrapping_offset(10 as isize);
let __v66: u32 = unsafe { *__v65 };
unsafe {
*__v65 = __v66.wrapping_add(*x.wrapping_offset(15 as isize));
}
unsafe {
*x.wrapping_offset(5 as isize) = rol32(*x.wrapping_offset(5 as isize) ^ *x.wrapping_offset(10 as isize), 12u32);
}
let __v68: *mut u32 = x.wrapping_offset(11 as isize);
let __v69: u32 = unsafe { *__v68 };
unsafe {
*__v68 = __v69.wrapping_add(*x.wrapping_offset(12 as isize));
}
unsafe {
*x.wrapping_offset(6 as isize) = rol32(*x.wrapping_offset(6 as isize) ^ *x.wrapping_offset(11 as isize), 12u32);
}
let __v71: *mut u32 = x.wrapping_offset(8 as isize);
let __v72: u32 = unsafe { *__v71 };
unsafe {
*__v71 = __v72.wrapping_add(*x.wrapping_offset(13 as isize));
}
unsafe {
*x.wrapping_offset(7 as isize) = rol32(*x.wrapping_offset(7 as isize) ^ *x.wrapping_offset(8 as isize), 12u32);
}
let __v74: *mut u32 = x.wrapping_offset(9 as isize);
let __v75: u32 = unsafe { *__v74 };
unsafe {
*__v74 = __v75.wrapping_add(*x.wrapping_offset(14 as isize));
}
unsafe {
*x.wrapping_offset(4 as isize) = rol32(*x.wrapping_offset(4 as isize) ^ *x.wrapping_offset(9 as isize), 12u32);
}
let __v77: *mut u32 = x.wrapping_offset(0 as isize);
let __v78: u32 = unsafe { *__v77 };
unsafe {
*__v77 = __v78.wrapping_add(*x.wrapping_offset(5 as isize));
}
unsafe {
*x.wrapping_offset(15 as isize) = rol32(*x.wrapping_offset(15 as isize) ^ *x.wrapping_offset(0 as isize), 8u32);
}
let __v80: *mut u32 = x.wrapping_offset(1 as isize);
let __v81: u32 = unsafe { *__v80 };
unsafe {
*__v80 = __v81.wrapping_add(*x.wrapping_offset(6 as isize));
}
unsafe {
*x.wrapping_offset(12 as isize) = rol32(*x.wrapping_offset(12 as isize) ^ *x.wrapping_offset(1 as isize), 8u32);
}
let __v83: *mut u32 = x.wrapping_offset(2 as isize);
let __v84: u32 = unsafe { *__v83 };
unsafe {
*__v83 = __v84.wrapping_add(*x.wrapping_offset(7 as isize));
}
unsafe {
*x.wrapping_offset(13 as isize) = rol32(*x.wrapping_offset(13 as isize) ^ *x.wrapping_offset(2 as isize), 8u32);
}
let __v86: *mut u32 = x.wrapping_offset(3 as isize);
let __v87: u32 = unsafe { *__v86 };
unsafe {
*__v86 = __v87.wrapping_add(*x.wrapping_offset(4 as isize));
}
unsafe {
*x.wrapping_offset(14 as isize) = rol32(*x.wrapping_offset(14 as isize) ^ *x.wrapping_offset(3 as isize), 8u32);
}
let __v89: *mut u32 = x.wrapping_offset(10 as isize);
let __v90: u32 = unsafe { *__v89 };
unsafe {
*__v89 = __v90.wrapping_add(*x.wrapping_offset(15 as isize));
}
unsafe {
*x.wrapping_offset(5 as isize) = rol32(*x.wrapping_offset(5 as isize) ^ *x.wrapping_offset(10 as isize), 7u32);
}
let __v92: *mut u32 = x.wrapping_offset(11 as isize);
let __v93: u32 = unsafe { *__v92 };
unsafe {
*__v92 = __v93.wrapping_add(*x.wrapping_offset(12 as isize));
}
unsafe {
*x.wrapping_offset(6 as isize) = rol32(*x.wrapping_offset(6 as isize) ^ *x.wrapping_offset(11 as isize), 7u32);
}
let __v95: *mut u32 = x.wrapping_offset(8 as isize);
let __v96: u32 = unsafe { *__v95 };
unsafe {
*__v95 = __v96.wrapping_add(*x.wrapping_offset(13 as isize));
}
unsafe {
*x.wrapping_offset(7 as isize) = rol32(*x.wrapping_offset(7 as isize) ^ *x.wrapping_offset(8 as isize), 7u32);
}
let __v98: *mut u32 = x.wrapping_offset(9 as isize);
let __v99: u32 = unsafe { *__v98 };
unsafe {
*__v98 = __v99.wrapping_add(*x.wrapping_offset(14 as isize));
}
unsafe {
*x.wrapping_offset(4 as isize) = rol32(*x.wrapping_offset(4 as isize) ^ *x.wrapping_offset(9 as isize), 7u32);
}
i = i.wrapping_add(2);
}
}

/// chacha_block_generic - generate one keystream block and increment block counter
/// @state: input state matrix
/// @out: output keystream block
/// @nrounds: number of rounds (20 or 12; 20 is recommended)
///
/// This is the ChaCha core, a function from 64-byte strings to 64-byte strings.
/// The caller has already converted the endianness of the input.  This function
/// also handles incrementing the block counter in the input matrix.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.chacha_block_generic.chacha_block_generic")]
extern "C" fn chacha_block_generic(state: *mut chacha_state, out: *mut u8, nrounds: i32) {
let mut permuted_state: chacha_state = unsafe { *state };
let mut i: i32 = 0;
chacha_permute(unsafe { &mut (*core::ptr::addr_of_mut!(permuted_state)) }, nrounds);
while (i as u64) < ARRAY_SIZE__9!() {
put_unaligned_le32(permuted_state.x[i as usize].wrapping_add(unsafe { *((*state).x.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) }), out.wrapping_offset((i as u64).wrapping_mul(core::mem::size_of::<u32>() as u64) as isize) as *mut ());
i = i.wrapping_add(1);
}
let __v3: *mut u32 = unsafe { (*state).x.as_mut_ptr() as *mut u32 }.wrapping_offset(12 as isize);
let __v4: u32 = unsafe { *__v3 };
unsafe {
*__v3 = __v4.wrapping_add(1);
}
chacha_zeroize_state(core::ptr::addr_of_mut!(permuted_state));
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_chacha_block_generic_7: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut chacha_state, *mut u8, i32)>, *mut ()>(Some(chacha_block_generic)) };

/// hchacha_block_generic - abbreviated ChaCha core, for XChaCha
/// @state: input state matrix
/// @out: the output words
/// @nrounds: number of rounds (20 or 12; 20 is recommended)
///
/// HChaCha is the ChaCha equivalent of HSalsa and is an intermediate step
/// towards XChaCha (see https://cr.yp.to/snuffle/xsalsa-20081128.pdf).  HChaCha
/// skips the final addition of the initial state, and outputs only certain words
/// of the state.  It should not be used for streaming directly.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.chacha_block_generic.hchacha_block_generic")]
extern "C" fn hchacha_block_generic(state: *const chacha_state, out: *mut u32, nrounds: i32) {
let mut permuted_state: chacha_state = unsafe { *state };
chacha_permute(unsafe { &mut (*core::ptr::addr_of_mut!(permuted_state)) }, nrounds);
unsafe { memcpy(out.wrapping_offset(0 as isize) as *mut (), (permuted_state.x.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) as *const (), 16u64) };
unsafe { memcpy(out.wrapping_offset(4 as isize) as *mut (), (permuted_state.x.as_mut_ptr() as *mut u32).wrapping_offset(12 as isize) as *const (), 16u64) };
chacha_zeroize_state(core::ptr::addr_of_mut!(permuted_state));
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hchacha_block_generic_8: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const chacha_state, *mut u32, i32)>, *mut ()>(Some(hchacha_block_generic)) };

static __slate_asm_string_chacha_permute_1: [u8; 1] = *b"\0";

static __slate_asm_string_chacha_permute_2: [u8; 63] = *b"/home/takashi/c-corpus/linux/lib/crypto/chacha-block-generic.c\0";
