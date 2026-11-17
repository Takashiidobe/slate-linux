
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0
//!
//! base64.c - Base64 with support for multiple variants
//!
//! Copyright (c) 2020 Hannes Reinecke, SUSE
//!
//! Based on the base64url routines from fs/crypto/fname.c
//! (which are using the URL-safe Base64 encoding),
//! modified to support multiple Base64 variants.
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
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.base64.base64_decode")]
extern "C" fn base64_decode(src: *const u8, mut srclen: i32, dst: *mut u8, mut padding: bool, variant: u32) -> i32 {
let mut bp: *mut u8 = dst;
let mut input: [i8; 4] = [0i8; 4];
let mut val: i32 = 0;
let mut s: *const u8 = src as *const u8;
let base64_rev_tables: *const i8 = unsafe { core::ptr::addr_of!(base64_rev_maps.0[(variant as u32) as usize]) as *const i8 };
while srclen >= 4 {
input[0] = unsafe { *base64_rev_tables.wrapping_offset((*s.wrapping_offset(0 as isize) as i32) as isize) };
input[1] = unsafe { *base64_rev_tables.wrapping_offset((*s.wrapping_offset(1 as isize) as i32) as isize) };
input[2] = unsafe { *base64_rev_tables.wrapping_offset((*s.wrapping_offset(2 as isize) as i32) as isize) };
input[3] = unsafe { *base64_rev_tables.wrapping_offset((*s.wrapping_offset(3 as isize) as i32) as isize) };
val = (input[0] as i32) << 18i32 | (input[1] as i32) << 12i32 | (input[2] as i32) << 6i32 | (input[3] as i32);
if unlikely!(val < 0) != 0 {
if !padding || srclen != 4 || ((unsafe { *s.wrapping_offset(3 as isize) }) as i32) != 61 {
return 1i32.wrapping_neg();
}
padding = false;
srclen = if ((unsafe { *s.wrapping_offset(2 as isize) }) as i32) == 61 { 2 } else { 3 };
break;
}
let __v1: *mut u8 = bp;
bp = __v1.wrapping_offset(1 as isize);
unsafe {
*__v1 = (val >> 16i32) as u8;
}
let __v3: *mut u8 = bp;
bp = __v3.wrapping_offset(1 as isize);
unsafe {
*__v3 = (val >> 8i32) as u8;
}
let __v5: *mut u8 = bp;
bp = __v5.wrapping_offset(1 as isize);
unsafe {
*__v5 = val as u8;
}
let __v7: *const u8 = s;
s = __v7.wrapping_offset(4 as isize);
srclen = srclen.wrapping_sub(4);
}
if likely!(srclen == 0) != 0 {
return ((unsafe { bp.offset_from(dst as *mut u8) }) as i64) as i32;
}
if padding || srclen == 1 {
return 1i32.wrapping_neg();
}
val = ((unsafe { *base64_rev_tables.wrapping_offset((*s.wrapping_offset(0 as isize) as i32) as isize) }) as i32) << 12i32 | ((unsafe { *base64_rev_tables.wrapping_offset((*s.wrapping_offset(1 as isize) as i32) as isize) }) as i32) << 6i32;
if srclen == 2 {
if (val as u32) & 2147484671 != 0 {
return 1i32.wrapping_neg();
}
let __v11: *mut u8 = bp;
bp = __v11.wrapping_offset(1 as isize);
unsafe {
*__v11 = (val >> 10i32) as u8;
}
} else {
val |= (unsafe { *base64_rev_tables.wrapping_offset((*s.wrapping_offset(2 as isize) as i32) as isize) }) as i32;
if (val as u32) & 2147483651 != 0 {
return 1i32.wrapping_neg();
}
let __v15: *mut u8 = bp;
bp = __v15.wrapping_offset(1 as isize);
unsafe {
*__v15 = (val >> 10i32) as u8;
}
let __v17: *mut u8 = bp;
bp = __v17.wrapping_offset(1 as isize);
unsafe {
*__v17 = (val >> 2i32) as u8;
}
}
((unsafe { bp.offset_from(dst as *mut u8) }) as i64) as i32
}

static base64_tables: __SlateAlign16<[[u8; 65]; 3]> = __SlateAlign16([[65u8, 66u8, 67u8, 68u8, 69u8, 70u8, 71u8, 72u8, 73u8, 74u8, 75u8, 76u8, 77u8, 78u8, 79u8, 80u8, 81u8, 82u8, 83u8, 84u8, 85u8, 86u8, 87u8, 88u8, 89u8, 90u8, 97u8, 98u8, 99u8, 100u8, 101u8, 102u8, 103u8, 104u8, 105u8, 106u8, 107u8, 108u8, 109u8, 110u8, 111u8, 112u8, 113u8, 114u8, 115u8, 116u8, 117u8, 118u8, 119u8, 120u8, 121u8, 122u8, 48u8, 49u8, 50u8, 51u8, 52u8, 53u8, 54u8, 55u8, 56u8, 57u8, 43u8, 47u8, 0u8], [65u8, 66u8, 67u8, 68u8, 69u8, 70u8, 71u8, 72u8, 73u8, 74u8, 75u8, 76u8, 77u8, 78u8, 79u8, 80u8, 81u8, 82u8, 83u8, 84u8, 85u8, 86u8, 87u8, 88u8, 89u8, 90u8, 97u8, 98u8, 99u8, 100u8, 101u8, 102u8, 103u8, 104u8, 105u8, 106u8, 107u8, 108u8, 109u8, 110u8, 111u8, 112u8, 113u8, 114u8, 115u8, 116u8, 117u8, 118u8, 119u8, 120u8, 121u8, 122u8, 48u8, 49u8, 50u8, 51u8, 52u8, 53u8, 54u8, 55u8, 56u8, 57u8, 45u8, 95u8, 0u8], [65u8, 66u8, 67u8, 68u8, 69u8, 70u8, 71u8, 72u8, 73u8, 74u8, 75u8, 76u8, 77u8, 78u8, 79u8, 80u8, 81u8, 82u8, 83u8, 84u8, 85u8, 86u8, 87u8, 88u8, 89u8, 90u8, 97u8, 98u8, 99u8, 100u8, 101u8, 102u8, 103u8, 104u8, 105u8, 106u8, 107u8, 108u8, 109u8, 110u8, 111u8, 112u8, 113u8, 114u8, 115u8, 116u8, 117u8, 118u8, 119u8, 120u8, 121u8, 122u8, 48u8, 49u8, 50u8, 51u8, 52u8, 53u8, 54u8, 55u8, 56u8, 57u8, 43u8, 44u8, 0u8]]);

// Initialize the base64 reverse mapping for a single character
// This macro maps a character to its corresponding base64 value,
// returning -1 if the character is invalid.
// char 'A'-'Z' maps to 0-25, 'a'-'z' maps to 26-51, '0'-'9' maps to 52-61,
// ch_62 maps to 62, ch_63 maps to 63, and other characters return -1
// Recursive macros to generate multiple Base64 reverse mapping table entries.
// Each macro generates a sequence of entries in the lookup table:
// INIT_2 generates 2 entries, INIT_4 generates 4, INIT_8 generates 8, and so on up to INIT_32.
static base64_rev_maps: __SlateAlign16<[[i8; 256]; 3]> = __SlateAlign16([{
let mut __t0: [i8; 256] = unsafe { core::mem::zeroed() };
let __t1 = 1i32.wrapping_neg() as i8;
let mut __t2: usize = 0;
while __t2 <= 31 {
__t0[__t2] = __t1;
__t2 += 1;
}
__t0[32] = 1i32.wrapping_neg() as i8;
__t0[33] = 1i32.wrapping_neg() as i8;
__t0[34] = 1i32.wrapping_neg() as i8;
__t0[35] = 1i32.wrapping_neg() as i8;
__t0[36] = 1i32.wrapping_neg() as i8;
__t0[37] = 1i32.wrapping_neg() as i8;
__t0[38] = 1i32.wrapping_neg() as i8;
__t0[39] = 1i32.wrapping_neg() as i8;
__t0[40] = 1i32.wrapping_neg() as i8;
__t0[41] = 1i32.wrapping_neg() as i8;
__t0[42] = 1i32.wrapping_neg() as i8;
__t0[43] = 62i8;
__t0[44] = 1i32.wrapping_neg() as i8;
__t0[45] = 1i32.wrapping_neg() as i8;
__t0[46] = 1i32.wrapping_neg() as i8;
__t0[47] = 63i8;
__t0[48] = 52i8;
__t0[49] = 53i8;
__t0[50] = 54i8;
__t0[51] = 55i8;
__t0[52] = 56i8;
__t0[53] = 57i8;
__t0[54] = 58i8;
__t0[55] = 59i8;
__t0[56] = 60i8;
__t0[57] = 61i8;
__t0[58] = 1i32.wrapping_neg() as i8;
__t0[59] = 1i32.wrapping_neg() as i8;
__t0[60] = 1i32.wrapping_neg() as i8;
__t0[61] = 1i32.wrapping_neg() as i8;
__t0[62] = 1i32.wrapping_neg() as i8;
__t0[63] = 1i32.wrapping_neg() as i8;
__t0[64] = 1i32.wrapping_neg() as i8;
__t0[65] = 0i8;
__t0[66] = 1i8;
__t0[67] = 2i8;
__t0[68] = 3i8;
__t0[69] = 4i8;
__t0[70] = 5i8;
__t0[71] = 6i8;
__t0[72] = 7i8;
__t0[73] = 8i8;
__t0[74] = 9i8;
__t0[75] = 10i8;
__t0[76] = 11i8;
__t0[77] = 12i8;
__t0[78] = 13i8;
__t0[79] = 14i8;
__t0[80] = 15i8;
__t0[81] = 16i8;
__t0[82] = 17i8;
__t0[83] = 18i8;
__t0[84] = 19i8;
__t0[85] = 20i8;
__t0[86] = 21i8;
__t0[87] = 22i8;
__t0[88] = 23i8;
__t0[89] = 24i8;
__t0[90] = 25i8;
__t0[91] = 1i32.wrapping_neg() as i8;
__t0[92] = 1i32.wrapping_neg() as i8;
__t0[93] = 1i32.wrapping_neg() as i8;
__t0[94] = 1i32.wrapping_neg() as i8;
__t0[95] = 1i32.wrapping_neg() as i8;
__t0[96] = 1i32.wrapping_neg() as i8;
__t0[97] = 26i8;
__t0[98] = 27i8;
__t0[99] = 28i8;
__t0[100] = 29i8;
__t0[101] = 30i8;
__t0[102] = 31i8;
__t0[103] = 32i8;
__t0[104] = 33i8;
__t0[105] = 34i8;
__t0[106] = 35i8;
__t0[107] = 36i8;
__t0[108] = 37i8;
__t0[109] = 38i8;
__t0[110] = 39i8;
__t0[111] = 40i8;
__t0[112] = 41i8;
__t0[113] = 42i8;
__t0[114] = 43i8;
__t0[115] = 44i8;
__t0[116] = 45i8;
__t0[117] = 46i8;
__t0[118] = 47i8;
__t0[119] = 48i8;
__t0[120] = 49i8;
__t0[121] = 50i8;
__t0[122] = 51i8;
__t0[123] = 1i32.wrapping_neg() as i8;
__t0[124] = 1i32.wrapping_neg() as i8;
__t0[125] = 1i32.wrapping_neg() as i8;
__t0[126] = 1i32.wrapping_neg() as i8;
__t0[127] = 1i32.wrapping_neg() as i8;
let __t3 = 1i32.wrapping_neg() as i8;
let mut __t4: usize = 128;
while __t4 <= 255 {
__t0[__t4] = __t3;
__t4 += 1;
}
__t0
}, {
let mut __t5: [i8; 256] = unsafe { core::mem::zeroed() };
let __t6 = 1i32.wrapping_neg() as i8;
let mut __t7: usize = 0;
while __t7 <= 31 {
__t5[__t7] = __t6;
__t7 += 1;
}
__t5[32] = 1i32.wrapping_neg() as i8;
__t5[33] = 1i32.wrapping_neg() as i8;
__t5[34] = 1i32.wrapping_neg() as i8;
__t5[35] = 1i32.wrapping_neg() as i8;
__t5[36] = 1i32.wrapping_neg() as i8;
__t5[37] = 1i32.wrapping_neg() as i8;
__t5[38] = 1i32.wrapping_neg() as i8;
__t5[39] = 1i32.wrapping_neg() as i8;
__t5[40] = 1i32.wrapping_neg() as i8;
__t5[41] = 1i32.wrapping_neg() as i8;
__t5[42] = 1i32.wrapping_neg() as i8;
__t5[43] = 1i32.wrapping_neg() as i8;
__t5[44] = 1i32.wrapping_neg() as i8;
__t5[45] = 62i8;
__t5[46] = 1i32.wrapping_neg() as i8;
__t5[47] = 1i32.wrapping_neg() as i8;
__t5[48] = 52i8;
__t5[49] = 53i8;
__t5[50] = 54i8;
__t5[51] = 55i8;
__t5[52] = 56i8;
__t5[53] = 57i8;
__t5[54] = 58i8;
__t5[55] = 59i8;
__t5[56] = 60i8;
__t5[57] = 61i8;
__t5[58] = 1i32.wrapping_neg() as i8;
__t5[59] = 1i32.wrapping_neg() as i8;
__t5[60] = 1i32.wrapping_neg() as i8;
__t5[61] = 1i32.wrapping_neg() as i8;
__t5[62] = 1i32.wrapping_neg() as i8;
__t5[63] = 1i32.wrapping_neg() as i8;
__t5[64] = 1i32.wrapping_neg() as i8;
__t5[65] = 0i8;
__t5[66] = 1i8;
__t5[67] = 2i8;
__t5[68] = 3i8;
__t5[69] = 4i8;
__t5[70] = 5i8;
__t5[71] = 6i8;
__t5[72] = 7i8;
__t5[73] = 8i8;
__t5[74] = 9i8;
__t5[75] = 10i8;
__t5[76] = 11i8;
__t5[77] = 12i8;
__t5[78] = 13i8;
__t5[79] = 14i8;
__t5[80] = 15i8;
__t5[81] = 16i8;
__t5[82] = 17i8;
__t5[83] = 18i8;
__t5[84] = 19i8;
__t5[85] = 20i8;
__t5[86] = 21i8;
__t5[87] = 22i8;
__t5[88] = 23i8;
__t5[89] = 24i8;
__t5[90] = 25i8;
__t5[91] = 1i32.wrapping_neg() as i8;
__t5[92] = 1i32.wrapping_neg() as i8;
__t5[93] = 1i32.wrapping_neg() as i8;
__t5[94] = 1i32.wrapping_neg() as i8;
__t5[95] = 63i8;
__t5[96] = 1i32.wrapping_neg() as i8;
__t5[97] = 26i8;
__t5[98] = 27i8;
__t5[99] = 28i8;
__t5[100] = 29i8;
__t5[101] = 30i8;
__t5[102] = 31i8;
__t5[103] = 32i8;
__t5[104] = 33i8;
__t5[105] = 34i8;
__t5[106] = 35i8;
__t5[107] = 36i8;
__t5[108] = 37i8;
__t5[109] = 38i8;
__t5[110] = 39i8;
__t5[111] = 40i8;
__t5[112] = 41i8;
__t5[113] = 42i8;
__t5[114] = 43i8;
__t5[115] = 44i8;
__t5[116] = 45i8;
__t5[117] = 46i8;
__t5[118] = 47i8;
__t5[119] = 48i8;
__t5[120] = 49i8;
__t5[121] = 50i8;
__t5[122] = 51i8;
__t5[123] = 1i32.wrapping_neg() as i8;
__t5[124] = 1i32.wrapping_neg() as i8;
__t5[125] = 1i32.wrapping_neg() as i8;
__t5[126] = 1i32.wrapping_neg() as i8;
__t5[127] = 1i32.wrapping_neg() as i8;
let __t8 = 1i32.wrapping_neg() as i8;
let mut __t9: usize = 128;
while __t9 <= 255 {
__t5[__t9] = __t8;
__t9 += 1;
}
__t5
}, {
let mut __t10: [i8; 256] = unsafe { core::mem::zeroed() };
let __t11 = 1i32.wrapping_neg() as i8;
let mut __t12: usize = 0;
while __t12 <= 31 {
__t10[__t12] = __t11;
__t12 += 1;
}
__t10[32] = 1i32.wrapping_neg() as i8;
__t10[33] = 1i32.wrapping_neg() as i8;
__t10[34] = 1i32.wrapping_neg() as i8;
__t10[35] = 1i32.wrapping_neg() as i8;
__t10[36] = 1i32.wrapping_neg() as i8;
__t10[37] = 1i32.wrapping_neg() as i8;
__t10[38] = 1i32.wrapping_neg() as i8;
__t10[39] = 1i32.wrapping_neg() as i8;
__t10[40] = 1i32.wrapping_neg() as i8;
__t10[41] = 1i32.wrapping_neg() as i8;
__t10[42] = 1i32.wrapping_neg() as i8;
__t10[43] = 62i8;
__t10[44] = 63i8;
__t10[45] = 1i32.wrapping_neg() as i8;
__t10[46] = 1i32.wrapping_neg() as i8;
__t10[47] = 1i32.wrapping_neg() as i8;
__t10[48] = 52i8;
__t10[49] = 53i8;
__t10[50] = 54i8;
__t10[51] = 55i8;
__t10[52] = 56i8;
__t10[53] = 57i8;
__t10[54] = 58i8;
__t10[55] = 59i8;
__t10[56] = 60i8;
__t10[57] = 61i8;
__t10[58] = 1i32.wrapping_neg() as i8;
__t10[59] = 1i32.wrapping_neg() as i8;
__t10[60] = 1i32.wrapping_neg() as i8;
__t10[61] = 1i32.wrapping_neg() as i8;
__t10[62] = 1i32.wrapping_neg() as i8;
__t10[63] = 1i32.wrapping_neg() as i8;
__t10[64] = 1i32.wrapping_neg() as i8;
__t10[65] = 0i8;
__t10[66] = 1i8;
__t10[67] = 2i8;
__t10[68] = 3i8;
__t10[69] = 4i8;
__t10[70] = 5i8;
__t10[71] = 6i8;
__t10[72] = 7i8;
__t10[73] = 8i8;
__t10[74] = 9i8;
__t10[75] = 10i8;
__t10[76] = 11i8;
__t10[77] = 12i8;
__t10[78] = 13i8;
__t10[79] = 14i8;
__t10[80] = 15i8;
__t10[81] = 16i8;
__t10[82] = 17i8;
__t10[83] = 18i8;
__t10[84] = 19i8;
__t10[85] = 20i8;
__t10[86] = 21i8;
__t10[87] = 22i8;
__t10[88] = 23i8;
__t10[89] = 24i8;
__t10[90] = 25i8;
__t10[91] = 1i32.wrapping_neg() as i8;
__t10[92] = 1i32.wrapping_neg() as i8;
__t10[93] = 1i32.wrapping_neg() as i8;
__t10[94] = 1i32.wrapping_neg() as i8;
__t10[95] = 1i32.wrapping_neg() as i8;
__t10[96] = 1i32.wrapping_neg() as i8;
__t10[97] = 26i8;
__t10[98] = 27i8;
__t10[99] = 28i8;
__t10[100] = 29i8;
__t10[101] = 30i8;
__t10[102] = 31i8;
__t10[103] = 32i8;
__t10[104] = 33i8;
__t10[105] = 34i8;
__t10[106] = 35i8;
__t10[107] = 36i8;
__t10[108] = 37i8;
__t10[109] = 38i8;
__t10[110] = 39i8;
__t10[111] = 40i8;
__t10[112] = 41i8;
__t10[113] = 42i8;
__t10[114] = 43i8;
__t10[115] = 44i8;
__t10[116] = 45i8;
__t10[117] = 46i8;
__t10[118] = 47i8;
__t10[119] = 48i8;
__t10[120] = 49i8;
__t10[121] = 50i8;
__t10[122] = 51i8;
__t10[123] = 1i32.wrapping_neg() as i8;
__t10[124] = 1i32.wrapping_neg() as i8;
__t10[125] = 1i32.wrapping_neg() as i8;
__t10[126] = 1i32.wrapping_neg() as i8;
__t10[127] = 1i32.wrapping_neg() as i8;
let __t13 = 1i32.wrapping_neg() as i8;
let mut __t14: usize = 128;
while __t14 <= 255 {
__t10[__t14] = __t13;
__t14 += 1;
}
__t10
}]);

/// base64_encode() - Base64-encode some binary data
/// @src: the binary data to encode
/// @srclen: the length of @src in bytes
/// @dst: (output) the Base64-encoded string.  Not NUL-terminated.
/// @padding: whether to append '=' padding characters
/// @variant: which base64 variant to use
///
/// Encodes data using the selected Base64 variant.
///
/// Return: the length of the resulting Base64-encoded string in bytes.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.base64.base64_encode")]
extern "C" fn base64_encode(mut src: *const u8, mut srclen: i32, dst: *mut u8, padding: bool, variant: u32) -> i32 {
let mut ac: u32 = 0;
let mut cp: *mut u8 = dst;
let base64_table: *const u8 = unsafe { core::ptr::addr_of!(base64_tables.0[(variant as u32) as usize]) as *const u8 };
while srclen >= 3 {
ac = (((unsafe { *src.wrapping_offset(0 as isize) }) as i32) << 16i32 | ((unsafe { *src.wrapping_offset(1 as isize) }) as i32) << 8i32 | ((unsafe { *src.wrapping_offset(2 as isize) }) as i32)) as u32;
let __v0: *mut u8 = cp;
cp = __v0.wrapping_offset(1 as isize);
unsafe {
*__v0 = *base64_table.wrapping_offset((ac >> 18i32) as isize);
}
let __v2: *mut u8 = cp;
cp = __v2.wrapping_offset(1 as isize);
unsafe {
*__v2 = *base64_table.wrapping_offset((ac >> 12i32 & 63) as isize);
}
let __v4: *mut u8 = cp;
cp = __v4.wrapping_offset(1 as isize);
unsafe {
*__v4 = *base64_table.wrapping_offset((ac >> 6i32 & 63) as isize);
}
let __v6: *mut u8 = cp;
cp = __v6.wrapping_offset(1 as isize);
unsafe {
*__v6 = *base64_table.wrapping_offset((ac & 63) as isize);
}
let __v8: *const u8 = src;
src = __v8.wrapping_offset(3 as isize);
srclen = srclen.wrapping_sub(3);
}
match srclen {
2 => {
ac = (((unsafe { *src.wrapping_offset(0 as isize) }) as i32) << 16i32 | ((unsafe { *src.wrapping_offset(1 as isize) }) as i32) << 8i32) as u32;
let __v13: *mut u8 = cp;
cp = __v13.wrapping_offset(1 as isize);
unsafe {
*__v13 = *base64_table.wrapping_offset((ac >> 18i32) as isize);
}
let __v15: *mut u8 = cp;
cp = __v15.wrapping_offset(1 as isize);
unsafe {
*__v15 = *base64_table.wrapping_offset((ac >> 12i32 & 63) as isize);
}
let __v17: *mut u8 = cp;
cp = __v17.wrapping_offset(1 as isize);
unsafe {
*__v17 = *base64_table.wrapping_offset((ac >> 6i32 & 63) as isize);
}
if padding {
let __v19: *mut u8 = cp;
cp = __v19.wrapping_offset(1 as isize);
unsafe {
*__v19 = 61;
}
}
}
1 => {
ac = (((unsafe { *src.wrapping_offset(0 as isize) }) as i32) << 16i32) as u32;
let __v22: *mut u8 = cp;
cp = __v22.wrapping_offset(1 as isize);
unsafe {
*__v22 = *base64_table.wrapping_offset((ac >> 18i32) as isize);
}
let __v24: *mut u8 = cp;
cp = __v24.wrapping_offset(1 as isize);
unsafe {
*__v24 = *base64_table.wrapping_offset((ac >> 12i32 & 63) as isize);
}
if padding {
let __v26: *mut u8 = cp;
cp = __v26.wrapping_offset(1 as isize);
unsafe {
*__v26 = 61;
}
let __v28: *mut u8 = cp;
cp = __v28.wrapping_offset(1 as isize);
unsafe {
*__v28 = 61;
}
}
}
_ => {
}
}
((unsafe { cp.offset_from(dst as *mut u8) }) as i64) as i32
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_base64_encode: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_base64_decode: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous", sym base64_encode, sym base64_decode, options(att_syntax));

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_base64_encode_5: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, i32, *mut u8, bool, u32) -> i32>, *mut ()>(Some(base64_encode)) };

// base64_decode() - Base64-decode a string
// @src: the string to decode.  Doesn't need to be NUL-terminated.
// @srclen: the length of @src in bytes
// @dst: (output) the decoded binary data
// @padding: whether the input is expected to include '=' padding characters
// @variant: which base64 variant to use
//
// Decodes a string using the selected Base64 variant.
//
// Return: the length of the resulting decoded binary data in bytes,
// 	   or -1 if the string isn't a valid Base64 string.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_base64_decode_6: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, i32, *mut u8, bool, u32) -> i32>, *mut ()>(Some(base64_decode)) };
