
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0-only
//!
//! lib/hexdump.c
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
//! Invalid argument
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
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
let mut __v: u16;
core::ptr::null::<u16>();
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
macro_rules! __get_unaligned_t__3 {
($ptr:expr) => {
{
let mut __v: u32;
core::ptr::null::<u32>();
let mut __v3: u32 = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), (($ptr) as *mut ()) as *const (), core::mem::size_of::<u32>() as u64) };
__v = __v3;
__v
}
};
}

macro_rules! __is_nonneg {
($ux:expr) => {
statically_true!((($ux) as i64) >= 0)
};
}

macro_rules! __ismask {
($x:expr) => {
unsafe { *(core::ptr::addr_of!(_ctype) as *const u8).wrapping_offset(((($x) as u8) as i32) as isize) }
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
if is_signed_type__3!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
};
}

macro_rules! __types_ok__3 {
($ux:expr, $uy:expr) => {
__sign_use__3!($ux) & __sign_use__3!($uy)
};
}

macro_rules! get_unaligned__3 {
($ptr:expr) => {
__get_unaligned_t__3!($ptr)
};
}

macro_rules! get_unaligned__4 {
($ptr:expr) => {
__get_unaligned_t__5!($ptr)
};
}

macro_rules! get_unaligned__2 {
($ptr:expr) => {
__get_unaligned_t__2!($ptr)
};
}

macro_rules! hex_asc_hi__2 {
($x:expr) => {
unsafe { *(core::ptr::addr_of!(hex_asc.0) as *const u8).wrapping_offset((((($x) as i32) & 240) >> 4i32) as isize) }
};
}

macro_rules! hex_asc_hi__3 {
($x:expr) => {
unsafe { hex_asc.0[(((($x) as i32) & 240) >> 4i32) as usize] }
};
}

macro_rules! hex_asc_lo__2 {
($x:expr) => {
unsafe { hex_asc.0[((($x) as i32) & 15) as usize] }
};
}

macro_rules! hex_asc_lo__3 {
($x:expr) => {
unsafe { *(core::ptr::addr_of!(hex_asc.0) as *const u8).wrapping_offset(((($x) as i32) & 15) as isize) }
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__3 {
() => {
1i32.wrapping_neg() < 1
};
}

macro_rules! isascii {
($c:expr) => {
((($c) as u8) as i32) <= 127
};
}

macro_rules! isprint {
($c:expr) => {
(__ismask!($c) as i32) & (16 | 1 | 2 | 4 | 128) != 0
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

macro_rules! unlikely {
($x:expr) => {
!(!$x) as i64
};
}

unsafe extern "C" {
    static mut _ctype: [u8; 0];
    fn _printk(arg0: *const u8, ...) -> i32;
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn __compiletime_assert_13();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
}

#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

fn hex_byte_pack(mut buf: *mut u8, byte: u8) -> *mut u8 {
let __v0: *mut u8 = buf;
buf = __v0.wrapping_offset(1 as isize);
unsafe {
*__v0 = hex_asc_hi__2!(byte);
}
let __v2: *mut u8 = buf;
buf = __v2.wrapping_offset(1 as isize);
unsafe {
*__v2 = hex_asc_lo__3!(byte);
}
buf
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hexdump.hex2bin")]
extern "C" fn hex2bin(mut dst: *mut u8, mut src: *const u8, mut count: u64) -> i32 {
loop {
let __v6: u64 = count;
count = __v6.wrapping_sub(1);
if __v6 == 0 {
break;
}
let mut hi: i32 = 0;
let mut lo: i32 = 0;
let __v0: *const u8 = src;
src = __v0.wrapping_offset(1 as isize);
hi = hex_to_bin(unsafe { *__v0 });
if unlikely!(hi < 0) != 0 {
return 22i32.wrapping_neg();
}
let __v2: *const u8 = src;
src = __v2.wrapping_offset(1 as isize);
lo = hex_to_bin(unsafe { *__v2 });
if unlikely!(lo < 0) != 0 {
return 22i32.wrapping_neg();
}
let __v4: *mut u8 = dst;
dst = __v4.wrapping_offset(1 as isize);
unsafe {
*__v4 = (hi << 4i32 | lo) as u8;
}
}
0
}

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
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hexdump.hex_dump_to_buffer")]
extern "C" fn hex_dump_to_buffer(buf: *const (), mut len: u64, mut rowsize: i32, mut groupsize: i32, linebuf: *mut u8, linebuflen: u64, ascii: bool) -> i32 {
let mut ptr8: *const u64 = unsafe { core::mem::zeroed() };
let mut ptr4: *const u32 = unsafe { core::mem::zeroed() };
let mut ptr2: *const u16 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut ascii_column: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut i32 = core::ptr::addr_of_mut!(ascii_column) as *mut i32;
let mut lx: i32 = unsafe { core::mem::zeroed() };
let mut j: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut i32 = core::ptr::addr_of_mut!(j) as *mut i32;
let mut ch: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_37: *mut u8 = core::ptr::addr_of_mut!(ch) as *mut u8;
let mut ngroups: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut i32 = core::ptr::addr_of_mut!(ngroups) as *mut i32;
let mut ptr: *const u8 = unsafe { core::mem::zeroed() };
unsafe {
ptr = buf as *const u8;
let __slate_auto_bytes = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_37 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
lx = 0;
let __slate_auto_bytes_3 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if rowsize != 16 && rowsize != 32 {
rowsize = 16;
}
if len > (rowsize as u64) {
len = rowsize as u64;
}
// limit to one line at a time
if !is_power_of_2(groupsize as u64) || groupsize > 8 {
groupsize = 1;
}
if len % (groupsize as u64) != 0 {
groupsize = 1;
}
'__join_0: {
// no mixed size output
*__slate_slot_38 = (len / (groupsize as u64)) as i32;
*__slate_slot_34 = rowsize.wrapping_mul(2).wrapping_add(rowsize / groupsize).wrapping_add(1);
if linebuflen != 0 {
'__join_2: {
if len != 0 {
'__join_1: {
if groupsize == 8 {
ptr8 = buf as *const u64;
*__slate_slot_36 = 0;
while *__slate_slot_36 < *__slate_slot_38 {
*__slate_slot_33 = snprintf(linebuf.wrapping_offset(lx as isize), linebuflen.wrapping_sub(lx as u64), c"%s%16.16llx".as_ptr() as *const u8, if *__slate_slot_36 != 0 { c" ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, get_unaligned__2!(ptr8.wrapping_offset(*__slate_slot_36 as isize)));
if (*__slate_slot_33 as u64) >= linebuflen.wrapping_sub(lx as u64) {
break '__join_0;
} else {
lx = lx.wrapping_add(*__slate_slot_33);
let __v9: i32 = *__slate_slot_36;
*__slate_slot_36 = __v9.wrapping_add(1);
}
}
} else {
if groupsize == 4 {
ptr4 = buf as *const u32;
*__slate_slot_36 = 0;
while *__slate_slot_36 < *__slate_slot_38 {
*__slate_slot_33 = snprintf(linebuf.wrapping_offset(lx as isize), linebuflen.wrapping_sub(lx as u64), c"%s%8.8x".as_ptr() as *const u8, if *__slate_slot_36 != 0 { c" ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, get_unaligned__3!(ptr4.wrapping_offset(*__slate_slot_36 as isize)));
if (*__slate_slot_33 as u64) >= linebuflen.wrapping_sub(lx as u64) {
break '__join_0;
} else {
lx = lx.wrapping_add(*__slate_slot_33);
let __v14: i32 = *__slate_slot_36;
*__slate_slot_36 = __v14.wrapping_add(1);
}
}
} else {
if groupsize == 2 {
ptr2 = buf as *const u16;
*__slate_slot_36 = 0;
while *__slate_slot_36 < *__slate_slot_38 {
*__slate_slot_33 = snprintf(linebuf.wrapping_offset(lx as isize), linebuflen.wrapping_sub(lx as u64), c"%s%4.4x".as_ptr() as *const u8, if *__slate_slot_36 != 0 { c" ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, get_unaligned__4!(ptr2.wrapping_offset(*__slate_slot_36 as isize)) as i32);
if (*__slate_slot_33 as u64) >= linebuflen.wrapping_sub(lx as u64) {
break '__join_0;
} else {
lx = lx.wrapping_add(*__slate_slot_33);
let __v19: i32 = *__slate_slot_36;
*__slate_slot_36 = __v19.wrapping_add(1);
}
}
} else {
*__slate_slot_36 = 0;
while (*__slate_slot_36 as u64) < len {
if linebuflen < (lx.wrapping_add(2) as u64) {
break '__join_1;
} else {
*__slate_slot_37 = *ptr.wrapping_offset(*__slate_slot_36 as isize);
let __v32: i32 = lx;
lx = __v32.wrapping_add(1);
*linebuf.wrapping_offset(__v32 as isize) = hex_asc_hi__3!(*__slate_slot_37);
if linebuflen < (lx.wrapping_add(2) as u64) {
break '__join_1;
} else {
let __v30: i32 = lx;
lx = __v30.wrapping_add(1);
*linebuf.wrapping_offset(__v30 as isize) = hex_asc_lo__2!(*__slate_slot_37);
if linebuflen < (lx.wrapping_add(2) as u64) {
break '__join_1;
} else {
let __v28: i32 = lx;
lx = __v28.wrapping_add(1);
*linebuf.wrapping_offset(__v28 as isize) = 32;
let __v26: i32 = *__slate_slot_36;
*__slate_slot_36 = __v26.wrapping_add(1);
}
}
}
}
if *__slate_slot_36 != 0 {
lx = lx.wrapping_sub(1);
}
}
}
}
if !ascii {
break '__join_2;
} else {
while lx < *__slate_slot_34 {
if linebuflen < (lx.wrapping_add(2) as u64) {
break '__join_1;
} else {
let __v7: i32 = lx;
lx = __v7.wrapping_add(1);
*linebuf.wrapping_offset(__v7 as isize) = 32;
}
}
*__slate_slot_36 = 0;
loop {
if (*__slate_slot_36 as u64) < len {
if linebuflen < (lx.wrapping_add(2) as u64) {
break;
} else {
*__slate_slot_37 = *ptr.wrapping_offset(*__slate_slot_36 as isize);
let __v5: i32 = lx;
lx = __v5.wrapping_add(1);
*linebuf.wrapping_offset(__v5 as isize) = (if isascii!(*__slate_slot_37) && isprint!(*__slate_slot_37) { *__slate_slot_37 as i32 } else { 46i32 }) as u8;
let __v3: i32 = *__slate_slot_36;
*__slate_slot_36 = __v3.wrapping_add(1);
}
} else {
break '__join_2;
}
}
}
}
let __v1: i32 = lx;
lx = __v1.wrapping_add(1);
*linebuf.wrapping_offset(__v1 as isize) = 0;
break '__join_0;
}
}
*linebuf.wrapping_offset(lx as isize) = 0;
return lx;
}
}
return (if ascii { (*__slate_slot_34 as u64).wrapping_add(len) } else { groupsize.wrapping_mul(2).wrapping_add(1).wrapping_mul(*__slate_slot_38).wrapping_sub(1) as u64 }) as i32;
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hexdump.print_hex_dump")]
extern "C" fn print_hex_dump(level: *const u8, prefix_str: *const u8, prefix_type: i32, mut rowsize: i32, groupsize: i32, buf: *const (), len: u64, ascii: bool) {
let ptr: *const u8 = buf as *const u8;
let mut i: i32 = 0;
let mut linelen: i32 = 0;
let mut remaining: i32 = len as i32;
let mut linebuf: __SlateAlign16<[u8; 131]> = __SlateAlign16([0u8; 131]);
if rowsize != 16 && rowsize != 32 {
rowsize = 16;
}
while (i as u64) < len {
let mut __v0: i32;
let __UNIQUE_ID_x__11: i32 = remaining;
let __UNIQUE_ID_y__12: i32 = rowsize;
loop {
if __types_ok__3!(__UNIQUE_ID_x__11, __UNIQUE_ID_y__12) == 0 {
unsafe { __compiletime_assert_13() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__11 < __UNIQUE_ID_y__12 { __UNIQUE_ID_x__11 } else { __UNIQUE_ID_y__12 };
linelen = __v0;
remaining = remaining.wrapping_sub(rowsize);
hex_dump_to_buffer(ptr.wrapping_offset(i as isize) as *const (), linelen as u64, rowsize, groupsize, linebuf.0.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 131]>() as u64, ascii);
match prefix_type {
1 => {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"%s%s%p: %s\n".as_ptr() as *const u8, level, prefix_str, ptr.wrapping_offset(i as isize), linebuf.0.as_mut_ptr() as *mut u8) };
}
2 => {
let mut __v7: i32;
while false {
}
__v7 = unsafe { _printk(c"%s%s%.8x: %s\n".as_ptr() as *const u8, level, prefix_str, i, linebuf.0.as_mut_ptr() as *mut u8) };
}
_ => {
let mut __v9: i32;
while false {
}
__v9 = unsafe { _printk(c"%s%s%s\n".as_ptr() as *const u8, level, prefix_str, linebuf.0.as_mut_ptr() as *mut u8) };
}
}
i = i.wrapping_add(rowsize);
}
}

#[inline(always)]
fn is_power_of_2(n: u64) -> bool {
n.wrapping_sub(1) < n ^ n.wrapping_sub(1)
}

// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
// @n: parameter
//
// constant-capable log of base 2 calculation
// - this can be used to initialise global variables from constant data, hence
// the massive ternary operator construction
//
// selects the appropriately-sized optimised version depending on sizeof(n)
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
#[unsafe(no_mangle)]
static hex_asc: __SlateAlign16<[u8; 17]> = __SlateAlign16([48u8, 49u8, 50u8, 51u8, 52u8, 53u8, 54u8, 55u8, 56u8, 57u8, 97u8, 98u8, 99u8, 100u8, 101u8, 102u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hex_asc_5: *mut () = (unsafe { core::ptr::addr_of!(hex_asc.0) }) as *mut ();

#[unsafe(no_mangle)]
static hex_asc_upper: __SlateAlign16<[u8; 17]> = __SlateAlign16([48u8, 49u8, 50u8, 51u8, 52u8, 53u8, 54u8, 55u8, 56u8, 57u8, 65u8, 66u8, 67u8, 68u8, 69u8, 70u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hex_asc_upper_6: *mut () = (unsafe { core::ptr::addr_of!(hex_asc_upper.0) }) as *mut ();

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_hex_asc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_hex_asc_upper: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_hex_to_bin: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_hex2bin: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bin2hex: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_hex_dump_to_buffer: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_print_hex_dump: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous", sym hex_asc, sym hex_asc_upper, sym hex_to_bin, sym hex2bin, sym bin2hex, sym hex_dump_to_buffer, sym print_hex_dump, options(att_syntax));

/// hex_to_bin - convert a hex digit to its real value
/// @ch: ascii character represents hex digit
///
/// hex_to_bin() converts one hex digit to its actual value or -1 in case of bad
/// input.
///
/// This function is used to load cryptographic keys, so it is coded in such a
/// way that there are no conditions or memory accesses that depend on data.
///
/// Explanation of the logic:
/// (ch - '9' - 1) is negative if ch <= '9'
/// ('0' - 1 - ch) is negative if ch >= '0'
/// we "and" these two values, so the result is negative if ch is in the range
/// 	'0' ... '9'
/// we are only interested in the sign, so we do a shift ">> 8"; note that right
/// 	shift of a negative value is implementation-defined, so we cast the
/// 	value to (unsigned) before the shift --- we have 0xffffff if ch is in
/// 	the range '0' ... '9', 0 otherwise
/// we "and" this value with (ch - '0' + 1) --- we have a value 1 ... 10 if ch is
/// 	in the range '0' ... '9', 0 otherwise
/// we add this value to -1 --- we have a value 0 ... 9 if ch is in the range '0'
/// 	... '9', -1 otherwise
/// the next line is similar to the previous one, but we need to decode both
/// 	uppercase and lowercase letters, so we use (ch & 0xdf), which converts
/// 	lowercase to uppercase
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hexdump.hex_to_bin")]
extern "C" fn hex_to_bin(ch: u8) -> i32 {
let cu: u8 = ((ch as i32) & 223) as u8;
(1i32.wrapping_neg() as u32).wrapping_add(((ch as i32).wrapping_sub(48).wrapping_add(1) as u32) & (((ch as i32).wrapping_sub(57).wrapping_sub(1) & 47i32.wrapping_sub(ch as i32)) as u32) >> 8i32).wrapping_add(((cu as i32).wrapping_sub(65).wrapping_add(11) as u32) & (((cu as i32).wrapping_sub(70).wrapping_sub(1) & 64i32.wrapping_sub(cu as i32)) as u32) >> 8i32) as i32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hex_to_bin_7: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u8) -> i32>, *mut ()>(Some(hex_to_bin)) };

// hex2bin - convert an ascii hexadecimal string to its binary representation
// @dst: binary result
// @src: ascii hexadecimal string
// @count: result length
//
// Return 0 on success, -EINVAL in case of bad input.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hex2bin_8: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u8, *const u8, u64) -> i32>, *mut ()>(Some(hex2bin)) };

/// bin2hex - convert binary data to an ascii hexadecimal string
/// @dst: ascii hexadecimal result
/// @src: binary data
/// @count: binary data length
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hexdump.bin2hex")]
extern "C" fn bin2hex(mut dst: *mut u8, src: *const (), mut count: u64) -> *mut u8 {
let mut _src: *const u8 = src as *const u8;
loop {
let __v2: u64 = count;
count = __v2.wrapping_sub(1);
if __v2 == 0 {
break;
}
let __v0: *const u8 = _src;
_src = __v0.wrapping_offset(1 as isize);
dst = hex_byte_pack(dst, unsafe { *__v0 });
}
dst
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bin2hex_9: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u8, *const (), u64) -> *mut u8>, *mut ()>(Some(bin2hex)) };

// hex_dump_to_buffer - convert a blob of data to "hex ASCII" in memory
// @buf: data blob to dump
// @len: number of bytes in the @buf
// @rowsize: number of bytes to print per line; must be 16 or 32
// @groupsize: number of bytes to print at a time (1, 2, 4, 8; default = 1)
// @linebuf: where to put the converted data
// @linebuflen: total size of @linebuf, including space for terminating NUL
// @ascii: include ASCII after the hex output
//
// hex_dump_to_buffer() works on one "line" of output at a time, i.e.,
// 16 or 32 bytes of input data converted to hex + ASCII output.
//
// Given a buffer of u8 data, hex_dump_to_buffer() converts the input data
// to a hex + ASCII dump at the supplied memory location.
// The converted output is always NUL-terminated.
//
// E.g.:
//   hex_dump_to_buffer(frame->data, frame->len, 16, 1,
// 			linebuf, sizeof(linebuf), true);
//
// example output buffer:
// 40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f  @ABCDEFGHIJKLMNO
//
// Return:
// The amount of bytes placed in the buffer without terminating NUL. If the
// output was truncated, then the return value is the number of bytes
// (excluding the terminating NUL) which would have been written to the final
// string if enough space had been available.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hex_dump_to_buffer_10: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const (), u64, i32, i32, *mut u8, u64, bool) -> i32>, *mut ()>(Some(hex_dump_to_buffer)) };

// print_hex_dump - print a text hex dump to syslog for a binary blob of data
// @level: kernel log level (e.g. KERN_DEBUG)
// @prefix_str: string to prefix each line with;
//  caller supplies trailing spaces for alignment if desired
// @prefix_type: controls whether prefix of an offset, address, or none
//  is printed (%DUMP_PREFIX_OFFSET, %DUMP_PREFIX_ADDRESS, %DUMP_PREFIX_NONE)
// @rowsize: number of bytes to print per line; must be 16 or 32
// @groupsize: number of bytes to print at a time (1, 2, 4, 8; default = 1)
// @buf: data blob to dump
// @len: number of bytes in the @buf
// @ascii: include ASCII after the hex output
//
// Given a buffer of u8 data, print_hex_dump() prints a hex + ASCII dump
// to the kernel log at the specified kernel log level, with an optional
// leading prefix.
//
// print_hex_dump() works on one "line" of output at a time, i.e.,
// 16 or 32 bytes of input data converted to hex + ASCII output.
// print_hex_dump() iterates over the entire input @buf, breaking it into
// "line size" chunks to format and print.
//
// E.g.:
//   print_hex_dump(KERN_DEBUG, "raw data: ", DUMP_PREFIX_ADDRESS,
// 		    16, 1, frame->data, frame->len, true);
//
// Example output using %DUMP_PREFIX_OFFSET and 1-byte mode:
// 0009ab42: 40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f  @ABCDEFGHIJKLMNO
// Example output using %DUMP_PREFIX_ADDRESS and 4-byte mode:
// ffffffff88089af0: 73727170 77767574 7b7a7978 7f7e7d7c  pqrstuvwxyz{|}~.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_print_hex_dump_14: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8, i32, i32, i32, *const (), u64, bool)>, *mut ()>(Some(print_hex_dump)) };
