
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0
//! This version is i.e. to prevent dead stores elimination on @ptr
//! where gcc and llvm may behave differently when otherwise using
//! normal barrier(): while gcc behavior gets along with a normal
//! barrier(), llvm needs an explicit input variable to be assumed
//! clobbered. The issue is as follows: while the inline asm might
//! access any memory it wants, the compiler could have fit all of
//! @ptr into memory registers instead, and since @ptr never escaped
//! from that, it proved that the inline asm wasn't touching any of
//! it. This version works well with both compilers, i.e. we're telling
//! the compiler that the inline asm absolutely may see the contents
//! of @ptr. See also: https://llvm.org/bugs/show_bug.cgi?id=15495
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! We need to compute the minimum and maximum values representable in a given
//! type. These macros may also be useful elsewhere. It would seem more obvious
//! to do something like:
//!
//! #define type_min(T) (T)(is_signed_type(T) ? (T)1 << (8*sizeof(T)-1) : 0)
//! #define type_max(T) (T)(is_signed_type(T) ? ((T)1 << (8*sizeof(T)-1)) - 1 : ~(T)0)
//!
//! Unfortunately, the middle expressions, strictly speaking, have
//! undefined behaviour, and at least some versions of gcc warn about
//! the type_max expression (but not if -fsanitize=undefined is in
//! effect; in that case, the warning is deferred to runtime...).
//!
//! The slightly excessive casting in type_min is to make sure the
//! macros also produce sensible values for the exotic type _Bool. [The
//! overflow checkers only almost work for _Bool, but that's
//! a-feature-not-a-bug, since people shouldn't be doing arithmetic on
//! _Bools. Besides, the gcc builtins don't allow _Bool* as third
//! argument.]
//!
//! Idea stolen from
//! https://mail-index.netbsd.org/tech-misc/2007/02/05/0000.html -
//! credit to Christian Biere.
//! Generate a mask for the specified type @t. Additional checks are made to
//! guarantee the value returned fits in that type, relying on
//! -Wshift-count-overflow compiler check to detect incompatible arguments.
//! For example, all these create build errors or warnings:
//!
//! - GENMASK(15, 20): wrong argument order
//! - GENMASK(72, 15): doesn't fit unsigned long
//! - GENMASK_U32(33, 15): doesn't fit in a u32
//! Invalid argument
//! Value too large for defined data type
//! IS_ERR_VALUE - Detect an error pointer.
//! @x: The pointer to check.
//!
//! Like IS_ERR(), but does not generate a compiler warning if result is unused.
//! 64 bit
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
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-malloc-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#malloc
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
// This returns a constant expression while determining if an argument is
// a constant expression, most importantly without evaluating the argument.
// Glory to Martin Uecker <Martin.Uecker@med.uni-goettingen.de>
//
// Details:
// - sizeof() return an integer constant expression, and does not evaluate
//   the value of its operand; it only examines the type of its operand.
// - The results of comparing two integer constant expressions is also
//   an integer constant expression.
// - The first literal "8" isn't important. It could be any literal value.
// - The second literal "8" is to avoid warnings about unaligned pointers;
//   this could otherwise just be "1".
// - (long)(x) is used to avoid warnings about 64-bit types on 32-bit
//   architectures.
// - The C Standard defines "null pointer constant", "(void *)0", as
//   distinct from other void pointers.
// - If (x) is an integer constant expression, then the "* 0l" resolves
//   it into an integer constant expression of value 0. Since it is cast to
//   "void *", this makes the second operand a null pointer constant.
// - If (x) is not an integer constant expression, then the second operand
//   resolves to a void pointer (but not a null pointer constant: the value
//   is not an integer constant 0).
// - The conditional operator's third operand, "(int *)8", is an object
//   pointer (to type "int").
// - The behavior (including the return type) of the conditional operator
//   ("operand1 ? operand2 : operand3") depends on the kind of expressions
//   given for the second and third operands. This is the central mechanism
//   of the macro:
//   - When one operand is a null pointer constant (i.e. when x is an integer
//     constant expression) and the other is an object pointer (i.e. our
//     third operand), the conditional operator returns the type of the
//     object pointer operand (i.e. "int *"). Here, within the sizeof(), we
//     would then get:
//       sizeof(*((int *)(...))  == sizeof(int)  == 4
//   - When one operand is a void pointer (i.e. when x is not an integer
//     constant expression) and the other is an object pointer (i.e. our
//     third operand), the conditional operator returns a "void *" type.
//     Here, within the sizeof(), we would then get:
//       sizeof(*((void *)(...)) == sizeof(void) == 1
// - The equality comparison to "sizeof(int)" therefore depends on (x):
//     sizeof(int) == sizeof(int)     (x) was a constant expression
//     sizeof(int) != sizeof(void)    (x) was not a constant expression
// Similar to statically_true() but produces a constant expression
//
// To be used in conjunction with macros, such as BUILD_BUG_ON_ZERO(),
// which require their input to be a constant expression and for which
// statically_true() would otherwise fail.
//
// This is a trade-off: const_true() requires all its operands to be
// compile time constants. Else, it would always returns false even on
// the most trivial cases like:
//
//   true || non_const_var
//
// On the opposite, statically_true() is able to fold more complex
// tautologies and will return true on expressions such as:
//
//   !(non_const_var * 8 % 4)
//
// For the general case, statically_true() is better.
// BUILD_BUG_ON_MSG - break compile if a condition is true & emit supplied
// 		      error message.
// @cond: the condition which the compiler should know is false.
// @msg: build-time error message
//
// See BUILD_BUG_ON for description.
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// max3 - return maximum of three values
// @x: first value
// @y: second value
// @z: third value
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
macro_rules! __cmp_once {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = ($y) as $type;
__v = if __v2 > __v3 { __v2 } else { __v3 };
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
macro_rules! __sign_use {
($ux:expr) => {
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
};
}

macro_rules! __types_ok3 {
($ux:expr, $uy:expr, $uz:expr) => {
__sign_use!($ux) & __sign_use!($uy) & __sign_use!($uz)
};
}

macro_rules! __types_ok__2 {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__2 {
() => {
1i32.wrapping_neg() < 1
};
}

/// max_t - return maximum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! max_t {
($type:ty, $x:expr, $y:expr) => {
__cmp_once!($type, $x, $y)
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
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __compiletime_assert_16();
    fn __compiletime_assert_19();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct resource {
    start: u64,
    end: u64,
    name: *const u8,
    flags: u64,
    desc: u64,
    parent: *mut resource,
    sibling: *mut resource,
    child: *mut resource,
}


// Resource type
// PCI/ISA I/O ports
// Resource extended types
// No address assigned yet
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct screen_info {
    orig_x: u8,
    orig_y: u8,
    ext_mem_k: u16,
    orig_video_page: u16,
    orig_video_mode: u8,
    orig_video_cols: u8,
    flags: u8,
    unused2: u8,
    orig_video_ega_bx: u16,
    unused3: u16,
    orig_video_lines: u8,
    orig_video_isVGA: u8,
    orig_video_points: u16,
    lfb_width: u16,
    lfb_height: u16,
    lfb_depth: u16,
    lfb_base: u32,
    lfb_size: u32,
    cl_magic: u16,
    cl_offset: u16,
    lfb_linelength: u16,
    red_size: u8,
    red_pos: u8,
    green_size: u8,
    green_pos: u8,
    blue_size: u8,
    blue_pos: u8,
    rsvd_size: u8,
    rsvd_pos: u8,
    vesapm_seg: u16,
    vesapm_off: u16,
    pages: u16,
    vesa_attributes: u16,
    capabilities: u32,
    ext_lfb_base: u32,
    _reserved: [u8; 2],
}


// Monochrome Text Display
// CGA Display
// EGA/VGA in Monochrome Mode
// EGA in Color Mode
// VGA+ in Color Mode
// VESA VGA in graphic mode
// ACER PICA-61 local S3 video
// MIPS Magnum 4000 G364 video
// Various SGI graphics hardware
// DEC TGA
// Sun frame buffer.
// Sun PCI based frame buffer.
// PowerMacintosh frame buffer.
// EFI graphic mode
// Frame buffer base is 64-bit
fn __screen_info_lfb_base(si: *const screen_info) -> u64 {
let mut lfb_base: u64 = (unsafe { (*si).lfb_base }) as u64;
if (unsafe { (*si).capabilities }) & 2u32 != 0 {
lfb_base |= ((unsafe { (*si).ext_lfb_base }) as u64) << 32i32;
}
lfb_base
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_pixel_format_h_9 {
    __slate_anon_0: __SlateRecord_pixel_format_h_10,
    index: __SlateRecord_pixel_format_h_16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pixel_format_h_10 {
    alpha: __SlateRecord_pixel_format_h_11,
    red: __SlateRecord_pixel_format_h_11,
    green: __SlateRecord_pixel_format_h_11,
    blue: __SlateRecord_pixel_format_h_11,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pixel_format_h_11 {
    offset: u8,
    length: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pixel_format_h_16 {
    offset: u8,
    length: u8,
}


fn __screen_info_lfb_size(si: *const screen_info, r#type: u32) -> u64 {
let mut lfb_size: u64 = (unsafe { (*si).lfb_size }) as u64;
if r#type == 35 {
lfb_size <<= 16i32;
}
lfb_size
}

fn __screen_info_video_type(r#type: u32) -> u32 {
match r#type {
16 | 17 | 32..=35 | 48 | 49 | 51 | 64 | 80 | 81 | 96 | 112 => {
return r#type;
}
_ => {
return 0;
}
}
unsafe { core::mem::zeroed() }
}

fn screen_info_video_type(si: *const screen_info) -> u32 {
let mut r#type: u32 = 0;
// check if display output is on
if (unsafe { (*si).orig_video_isVGA }) == 0 {
return 0;
}
// check for a known VIDEO_TYPE_ constant
r#type = __screen_info_video_type((unsafe { (*si).orig_video_isVGA }) as u32);
if r#type != 0 {
return (unsafe { (*si).orig_video_isVGA }) as u32;
}
// check if text mode has been initialized
if (unsafe { (*si).orig_video_lines }) == 0 || (unsafe { (*si).orig_video_cols }) == 0 {
return 0;
}
// 80x25 text, mono
if ((unsafe { (*si).orig_video_mode }) as i32) == 7 {
if ((unsafe { (*si).orig_video_ega_bx }) as i32) & 255 != 16 {
return 32;
} else {
return 16;
}
}
// EGA/VGA, 16 colors
if ((unsafe { (*si).orig_video_ega_bx }) as i32) & 255 != 16 {
if (unsafe { (*si).orig_video_isVGA }) != 0 {
return 34;
} else {
return 33;
}
}
// the rest...
17
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_screen_info_resources: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___screen_info_lfb_bits_per_pixel: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_screen_info_pixel_format: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous", sym screen_info_resources, sym __screen_info_lfb_bits_per_pixel, sym screen_info_pixel_format, options(att_syntax));

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.screen_info_generic.__screen_info_lfb_bits_per_pixel")]
extern "C" fn __screen_info_lfb_bits_per_pixel(si: *const screen_info) -> u32 {
let mut bits_per_pixel: u32 = (unsafe { (*si).lfb_depth }) as u32;
if bits_per_pixel > 8 {
let mut __v0: i32;
let mut __UNIQUE_ID_x__17: i32 = 0;
let mut __v1: i32;
let __UNIQUE_ID_x__13: i32 = ((unsafe { (*si).red_size }) as i32).wrapping_add((unsafe { (*si).red_pos }) as i32);
let __UNIQUE_ID_y__14: i32 = ((unsafe { (*si).green_size }) as i32).wrapping_add((unsafe { (*si).green_pos }) as i32);
let __UNIQUE_ID_z__15: i32 = ((unsafe { (*si).blue_size }) as i32).wrapping_add((unsafe { (*si).blue_pos }) as i32);
loop {
if __types_ok3!(__UNIQUE_ID_x__13, __UNIQUE_ID_y__14, __UNIQUE_ID_z__15) == 0 {
unsafe { __compiletime_assert_16() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = if __UNIQUE_ID_x__13 > if __UNIQUE_ID_y__14 > __UNIQUE_ID_z__15 { __UNIQUE_ID_y__14 } else { __UNIQUE_ID_z__15 } { __UNIQUE_ID_x__13 } else { if __UNIQUE_ID_y__14 > __UNIQUE_ID_z__15 { __UNIQUE_ID_y__14 } else { __UNIQUE_ID_z__15 } };
__UNIQUE_ID_x__17 = __v1;
let __UNIQUE_ID_y__18: i32 = ((unsafe { (*si).rsvd_size }) as i32).wrapping_add((unsafe { (*si).rsvd_pos }) as i32);
loop {
if __types_ok__2!(__UNIQUE_ID_x__17, __UNIQUE_ID_y__18) == 0 {
unsafe { __compiletime_assert_19() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__17 > __UNIQUE_ID_y__18 { __UNIQUE_ID_x__17 } else { __UNIQUE_ID_y__18 };
bits_per_pixel = __v0 as u32;
bits_per_pixel = max_t!(u32, bits_per_pixel, unsafe { (*si).lfb_depth });
}
bits_per_pixel
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pixel_format {
    bits_per_pixel: u8,
    indexed: bool,
    __slate_anon_2: __SlateRecord_pixel_format_h_9,
}


fn resource_init_named(r: *mut resource, start: u64, size: u64, name: *const u8, flags: u32) {
unsafe { memset(r as *mut (), 0i32, core::mem::size_of::<resource>() as u64) };
unsafe {
(*r).start = start;
}
unsafe {
(*r).end = start.wrapping_add(size).wrapping_sub(1);
}
unsafe {
(*r).name = name;
}
unsafe {
(*r).flags = flags as u64;
}
}

fn resource_init_io_named(r: *mut resource, start: u64, size: u64, name: *const u8) {
resource_init_named(r, start, size, name, 256u32);
}

fn resource_init_mem_named(r: *mut resource, start: u64, size: u64, name: *const u8) {
resource_init_named(r, start, size, name, 512u32);
}

fn __screen_info_has_ega_gfx(mode: u32) -> bool {
match mode {
13..=16 => {
return true; // 320x200-4
// 640x200-4
// 640x350-1
// 640x350-4
}
_ => {
return false;
}
}
unsafe { core::mem::zeroed() }
}

fn __screen_info_has_vga_gfx(mode: u32) -> bool {
match mode {
16 | 18 | 19 | 106 => {
return true; // 640x480-1
// 640x480-4
// 320-200-8
// 800x600-4 (VESA)
}
_ => {
return __screen_info_has_ega_gfx(mode);
}
}
unsafe { core::mem::zeroed() }
}

/// screen_info_resources() - Get resources from screen_info structure
/// @si: the screen_info
/// @r: pointer to an array of resource structures
/// @num: number of elements in @r:
///
/// Returns:
/// The number of resources stored in @r on success, or a negative errno code otherwise.
///
/// A call to screen_info_resources() returns the resources consumed by the
/// screen_info's device or framebuffer. The result is stored in the caller-supplied
/// array @r with up to @num elements. The function returns the number of
/// initialized elements.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.screen_info_generic.screen_info_resources")]
extern "C" fn screen_info_resources(si: *const screen_info, r: *mut resource, num: u64) -> i64 {
let mut pos: *mut resource = r;
let r#type: u32 = screen_info_video_type(si);
let mut base: u64 = 0;
let mut size: u64 = 0;
match r#type {
16 => {
if num > 0 {
let __v0: *mut resource = pos;
pos = __v0.wrapping_offset(1 as isize);
resource_init_io_named(__v0, 944u64, 12u64, c"mda".as_ptr() as *const u8);
}
if num > 1 {
let __v2: *mut resource = pos;
pos = __v2.wrapping_offset(1 as isize);
resource_init_io_named(__v2, 959u64, 1u64, c"mda".as_ptr() as *const u8);
}
if num > 2 {
let __v4: *mut resource = pos;
pos = __v4.wrapping_offset(1 as isize);
resource_init_mem_named(__v4, 720896u64, 8192u64, c"mda".as_ptr() as *const u8);
}
}
17 => {
if num > 0 {
let __v7: *mut resource = pos;
pos = __v7.wrapping_offset(1 as isize);
resource_init_io_named(__v7, 980u64, 2u64, c"cga".as_ptr() as *const u8);
}
if num > 1 {
let __v9: *mut resource = pos;
pos = __v9.wrapping_offset(1 as isize);
resource_init_mem_named(__v9, 753664u64, 8192u64, c"cga".as_ptr() as *const u8);
}
}
32 => {
if num > 0 {
let __v11: *mut resource = pos;
pos = __v11.wrapping_offset(1 as isize);
resource_init_io_named(__v11, 959u64, 16u64, c"ega".as_ptr() as *const u8);
}
if num > 1 {
let __v13: *mut resource = pos;
pos = __v13.wrapping_offset(1 as isize);
resource_init_mem_named(__v13, 720896u64, 32768u64, c"ega".as_ptr() as *const u8);
}
}
33 => {
if num > 0 {
let __v15: *mut resource = pos;
pos = __v15.wrapping_offset(1 as isize);
resource_init_io_named(__v15, 960u64, 32u64, c"ega".as_ptr() as *const u8);
}
if num > 1 {
if __screen_info_has_ega_gfx((unsafe { (*si).orig_video_mode }) as u32) {
let __v17: *mut resource = pos;
pos = __v17.wrapping_offset(1 as isize);
resource_init_mem_named(__v17, 655360u64, 65536u64, c"ega".as_ptr() as *const u8);
} else {
let __v19: *mut resource = pos;
pos = __v19.wrapping_offset(1 as isize);
resource_init_mem_named(__v19, 753664u64, 32768u64, c"ega".as_ptr() as *const u8);
}
}
}
34 => {
if num > 0 {
let __v21: *mut resource = pos;
pos = __v21.wrapping_offset(1 as isize);
resource_init_io_named(__v21, 960u64, 32u64, c"vga+".as_ptr() as *const u8);
}
if num > 1 {
if __screen_info_has_vga_gfx((unsafe { (*si).orig_video_mode }) as u32) {
let __v23: *mut resource = pos;
pos = __v23.wrapping_offset(1 as isize);
resource_init_mem_named(__v23, 655360u64, 65536u64, c"vga+".as_ptr() as *const u8);
} else {
let __v25: *mut resource = pos;
pos = __v25.wrapping_offset(1 as isize);
resource_init_mem_named(__v25, 753664u64, 32768u64, c"vga+".as_ptr() as *const u8);
}
}
}
35 | 112 => {
base = __screen_info_lfb_base(si);
if base != 0 {
size = __screen_info_lfb_size(si, r#type);
if size != 0 {
if num > 0 {
let __v27: *mut resource = pos;
pos = __v27.wrapping_offset(1 as isize);
resource_init_mem_named(__v27, base, size, c"lfb".as_ptr() as *const u8);
}
}
}
}
_ => {
return 22i32.wrapping_neg() as i64; // not supported
}
}
(unsafe { pos.offset_from(r as *mut resource) }) as i64
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_screen_info_resources_12: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const screen_info, *mut resource, u64) -> i64>, *mut ()>(Some(screen_info_resources)) };

// The meaning of depth and bpp for direct-color formats is
// inconsistent:
//
//  - DRM format info specifies depth as the number of color
//    bits; including alpha, but not including filler bits.
//  - Linux' EFI platform code computes lfb_depth from the
//    individual color channels, including the reserved bits.
//  - VBE 1.1 defines lfb_depth for XRGB1555 as 16, but later
//    versions use 15.
//  - On the kernel command line, 'bpp' of 32 is usually
//    XRGB8888 including the filler bits, but 15 is XRGB1555
//    not including the filler bit.
//
// It is not easily possible to fix this in struct screen_info,
// as this could break UAPI. The best solution is to compute
// bits_per_pixel from the color bits, reserved bits and
// reported lfb_depth, whichever is highest.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___screen_info_lfb_bits_per_pixel_22: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const screen_info) -> u32>, *mut ()>(Some(__screen_info_lfb_bits_per_pixel)) };

fn __screen_info_lfb_pixel_format(si: *const screen_info, f: *mut pixel_format) -> i32 {
let bits_per_pixel: u32 = __screen_info_lfb_bits_per_pixel(si);
if bits_per_pixel > 255u32 {
return 22i32.wrapping_neg();
}
unsafe {
(*f).bits_per_pixel = bits_per_pixel as u8;
}
if ((unsafe { (*si).lfb_depth }) as i32) > 8 {
unsafe {
(*f).indexed = false;
}
unsafe {
(*f).__slate_anon_2.__slate_anon_0.alpha.offset = 0;
}
unsafe {
(*f).__slate_anon_2.__slate_anon_0.alpha.length = 0;
}
unsafe {
(*f).__slate_anon_2.__slate_anon_0.red.offset = (*si).red_pos;
}
unsafe {
(*f).__slate_anon_2.__slate_anon_0.red.length = (*si).red_size;
}
unsafe {
(*f).__slate_anon_2.__slate_anon_0.green.offset = (*si).green_pos;
}
unsafe {
(*f).__slate_anon_2.__slate_anon_0.green.length = (*si).green_size;
}
unsafe {
(*f).__slate_anon_2.__slate_anon_0.blue.offset = (*si).blue_pos;
}
unsafe {
(*f).__slate_anon_2.__slate_anon_0.blue.length = (*si).blue_size;
}
} else {
unsafe {
(*f).indexed = true;
}
unsafe {
(*f).__slate_anon_2.index.offset = 0;
}
unsafe {
(*f).__slate_anon_2.index.length = (*si).lfb_depth as u8;
}
}
0
}

/// screen_info_pixel_format - Returns the screen-info format as pixel-format description
///
/// @si: the screen_info
/// @f: pointer to return pixel-format description
///
/// Returns:
/// 0 on success, or a negative errno code otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.screen_info_generic.screen_info_pixel_format")]
extern "C" fn screen_info_pixel_format(si: *const screen_info, f: *mut pixel_format) -> i32 {
let r#type: u32 = screen_info_video_type(si);
// TODO: Add support for additional types as needed.
match r#type {
35 | 112 => {
return __screen_info_lfb_pixel_format(si, f);
}
_ => {
}
}
// not supported
22i32.wrapping_neg()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_screen_info_pixel_format_23: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const screen_info, *mut pixel_format) -> i32>, *mut ()>(Some(screen_info_pixel_format)) };
