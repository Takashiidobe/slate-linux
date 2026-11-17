
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0-only
//!
//! ----------------------------------------------------------------------- *
//!
//! Copyright 2012 Intel Corporation; author H. Peter Anvin
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
// printk - print a kernel message
// @fmt: format string
//
// This is printk(). It can be called from any context. We want it to work.
//
// If printk indexing is enabled, _printk() is called from printk_index_wrap.
// Otherwise, printk is simply #defined to _printk.
//
// We try to grab the console_lock. If we succeed, it's easy - we log the
// output and call the console drivers.  If we fail to get the semaphore, we
// place the output into the log buffer and return. The current holder of
// the console_sem will notice the new output in console_unlock(); and will
// send it to the consoles before releasing the lock.
//
// One effect of this deferred printing is that code which calls printk() and
// then changes console_loglevel may break. This is because console_loglevel
// is inspected when the actual printing occurs.
//
// See also:
// printf(3)
//
// See the vsnprintf() documentation for format string extensions over C99.
// pr_warn - Print a warning-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_WARNING loglevel. It uses pr_fmt()
// to generate the format string.
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
// strscpy - Copy a C-string into a sized buffer
// @dst: Where to copy the string to
// @src: Where to copy the string from
// @...: Size of destination buffer (optional)
//
// Copy the source string @src, or as much of it as fits, into the
// destination @dst buffer. The behavior is undefined if the string
// buffers overlap. The destination @dst buffer is always NUL terminated,
// unless it's zero-sized.
//
// The size argument @... is only required when @dst is not an array, or
// when the copy needs to be smaller than sizeof(@dst).
//
// Preferred to strncpy() since it always returns a valid string, and
// doesn't unnecessarily force the tail of the destination buffer to be
// zero padded. If padding is desired please use strscpy_pad().
//
// Returns the number of characters copied in @dst (not including the
// trailing %NUL) or -E2BIG if @size is 0 or the copy from @src was
// truncated.
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
/// @a is a power of 2 value
macro_rules! ALIGN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL__2!($x, $a)
};
}

macro_rules! PTR_ALIGN__2 {
($p:expr, $a:expr) => {
ALIGN!(($p) as u64, $a) as *const u8
};
}

macro_rules! __ALIGN_KERNEL_MASK {
($x:expr, $mask:expr) => {
$x.wrapping_add($mask) & !$mask
};
}

macro_rules! __ALIGN_KERNEL__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as u64).wrapping_sub(1))
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr {
() => {
core::mem::size_of::<__SlateRecord_earlycpio_c_129>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__2 {
() => {
core::mem::size_of::<__SlateRecord_earlycpio_c_129_2>() as i32
};
}

unsafe extern "C" {
    fn _printk(arg0: *const u8, ...) -> i32;
    fn memcmp(arg0: *const (), arg1: *const (), arg2: u64) -> i32;
    fn strlen(arg0: *const u8) -> u64;
    fn sized_strscpy(arg0: *mut u8, arg1: *const u8, arg2: u64) -> i64;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cpio_data {
    data: *mut (),
    size: u64,
    name: [u8; 18],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_earlycpio_c_129 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_earlycpio_c_129_2 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

// earlycpio.c
//
// Find a specific cpio member; must precede any compressed content.
// This is used to locate data items in the initramfs used by the
// kernel itself during early boot (before the main initramfs is
// decompressed.)  It is the responsibility of the initramfs creator
// to ensure that these items are uncompressed at the head of the
// blob.  Depending on the boot loader or package tool that may be a
// separate file or part of the same file.
// find_cpio_data - Search for files in an uncompressed cpio
// @path:       The directory to search for, including a slash at the end
// @data:       Pointer to the cpio archive or a header inside
// @len:        Remaining length of the cpio based on data pointer
// @nextoff:    When a matching file is found, this is the offset from the
//              beginning of the cpio to the beginning of the next file, not the
//              matching file itself. It can be used to iterate through the cpio
//              to find all files inside of a directory path.
//
// Return:      &struct cpio_data containing the address, length and
//              filename (with the directory path cut off) of the found file.
//              If you search for a filename and not for files in a directory,
//              pass the absolute path of the filename in the cpio and make sure
//              the match returned an empty filename string.
#[unsafe(no_mangle)]
extern "C" fn find_cpio_data(path: *const u8, data: *mut (), mut len: u64, nextoff: *mut i64) -> cpio_data {
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut j: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut i32 = core::ptr::addr_of_mut!(j) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut mypathsize: u64 = unsafe { core::mem::zeroed() };
let mut x: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u8 = core::ptr::addr_of_mut!(x) as *mut u8;
let mut c: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut u8 = core::ptr::addr_of_mut!(c) as *mut u8;
let mut v: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut u32 = core::ptr::addr_of_mut!(v) as *mut u32;
let mut chp: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut *mut u32 = core::ptr::addr_of_mut!(chp) as *mut *mut u32;
let mut ch: core::mem::MaybeUninit<__SlateAlign16<[u32; 14]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut [u32; 14] = core::ptr::addr_of_mut!(ch) as *mut [u32; 14];
let mut nptr: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut *const u8 = core::ptr::addr_of_mut!(nptr) as *mut *const u8;
let mut dptr: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut *const u8 = core::ptr::addr_of_mut!(dptr) as *mut *const u8;
let mut p: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut *const u8 = core::ptr::addr_of_mut!(p) as *mut *const u8;
let mut cd: core::mem::MaybeUninit<cpio_data> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut cpio_data = core::ptr::addr_of_mut!(cd) as *mut cpio_data;
let mut cpio_header_len: u64 = unsafe { core::mem::zeroed() };
unsafe {
cpio_header_len = 110u64;
core::ptr::write(__slate_slot_35, {
let mut __t0: cpio_data = core::mem::zeroed();
__t0.data = core::ptr::null_mut::<()>();
__t0.size = 0u64;
__t0.name = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t0
});
let __slate_auto_bytes = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 56usize);
let __slate_auto_bytes_4 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_7 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 1usize);
mypathsize = strlen(path);
let __slate_auto_bytes_8 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_34 = data as *const u8;
'__join_0: {
loop {
if len > cpio_header_len {
if *(*__slate_slot_34) == 0 {
// All cpio headers need to be 4-byte aligned
let __v23: *const u8 = *__slate_slot_34;
*__slate_slot_34 = __v23.wrapping_offset(4 as isize);
len = len.wrapping_sub(4);
} else {
*__slate_slot_24 = 6; // The magic field is only 6 characters
*__slate_slot_30 = (*__slate_slot_31).as_mut_ptr() as *mut u32;
*__slate_slot_25 = 14;
while *__slate_slot_25 != 0 {
*__slate_slot_29 = 0;
loop {
let __v11: i32 = *__slate_slot_24;
*__slate_slot_24 = __v11.wrapping_sub(1);
if __v11 != 0 {
*__slate_slot_29 <<= 4i32;
let __v17: *const u8 = *__slate_slot_34;
*__slate_slot_34 = __v17.wrapping_offset(1 as isize);
*__slate_slot_28 = *__v17;
*__slate_slot_27 = (*__slate_slot_28 as i32).wrapping_sub(48) as u8;
if (*__slate_slot_27 as i32) < 10 {
let __v15: u32 = *__slate_slot_29;
*__slate_slot_29 = __v15.wrapping_add((*__slate_slot_27 as i32) as u32);
} else {
*__slate_slot_27 = ((*__slate_slot_28 as i32) | 32).wrapping_sub(97) as u8;
if (*__slate_slot_27 as i32) < 6 {
let __v13: u32 = *__slate_slot_29;
*__slate_slot_29 = __v13.wrapping_add((*__slate_slot_27 as i32).wrapping_add(10) as u32);
} else {
break '__join_0;
}
}
} else {
break;
}
}
let __v9: *mut u32 = *__slate_slot_30;
*__slate_slot_30 = __v9.wrapping_offset(1 as isize);
*__v9 = *__slate_slot_29;
*__slate_slot_24 = 8; // All other fields are 8 characters
let __v7: i32 = *__slate_slot_25;
*__slate_slot_25 = __v7.wrapping_sub(1);
}
if (*__slate_slot_31)[0].wrapping_sub(460545) > 1 {
break '__join_0;
} else {
// Invalid magic
len = len.wrapping_sub(cpio_header_len);
*__slate_slot_33 = PTR_ALIGN__2!((*__slate_slot_34).wrapping_offset((*__slate_slot_31)[12] as isize), 4i32);
*__slate_slot_32 = PTR_ALIGN__2!((*__slate_slot_33).wrapping_offset((*__slate_slot_31)[7] as isize), 4i32);
if *__slate_slot_32 > (*__slate_slot_34).wrapping_offset(len as isize) || *__slate_slot_33 < *__slate_slot_34 || *__slate_slot_32 < *__slate_slot_33 {
break '__join_0;
} else {
// Buffer overrun
if (*__slate_slot_31)[2] & 61440 == 32768 && ((*__slate_slot_31)[12] as u64) >= mypathsize {
__v3 = memcmp(*__slate_slot_34 as *const (), path as *const (), mypathsize) == 0;
} else {
__v3 = false;
}
if __v3 {
break;
} else {
let __v1: u64 = len;
len = __v1.wrapping_sub(((*__slate_slot_32).offset_from(*__slate_slot_34 as *const u8) as i64) as u64);
*__slate_slot_34 = *__slate_slot_32;
}
}
}
}
} else {
break '__join_0;
}
}
if !nextoff.is_null() {
*nextoff = (*__slate_slot_32 as i64).wrapping_sub(data as i64);
}
if ((*__slate_slot_31)[12] as u64).wrapping_sub(mypathsize) >= 18 {
while false {
}
_printk(c"\x014File %s exceeding MAX_CPIO_FILE_NAME [%d]\n".as_ptr() as *const u8, *__slate_slot_34, 18i32);
}
sized_strscpy((*__slate_slot_35).name.as_mut_ptr() as *mut u8, (*__slate_slot_34).wrapping_offset(mypathsize as isize), 18i32.wrapping_add(__must_be_cstr!()).wrapping_add(__must_be_cstr__2!()) as u64);
(*__slate_slot_35).data = *__slate_slot_33 as *mut ();
(*__slate_slot_35).size = (*__slate_slot_31)[7] as u64;
return *__slate_slot_35;
}
return *__slate_slot_35;
}
// Invalid hexadecimal
// Found it!
}
