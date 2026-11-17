
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0-only
//!
//! -*- linux-c -*- ------------------------------------------------------- *
//!
//! Copyright (C) 1991, 1992 Linus Torvalds
//! Copyright 2007 rPath, Inc. - All Rights Reserved
//! number of edd_info structs starting at EDDBUF
//! This returns a constant expression while determining if an argument is
//! a constant expression, most importantly without evaluating the argument.
//! Glory to Martin Uecker <Martin.Uecker@med.uni-goettingen.de>
//!
//! Details:
//! - sizeof() return an integer constant expression, and does not evaluate
//!   the value of its operand; it only examines the type of its operand.
//! - The results of comparing two integer constant expressions is also
//!   an integer constant expression.
//! - The first literal "8" isn't important. It could be any literal value.
//! - The second literal "8" is to avoid warnings about unaligned pointers;
//!   this could otherwise just be "1".
//! - (long)(x) is used to avoid warnings about 64-bit types on 32-bit
//!   architectures.
//! - The C Standard defines "null pointer constant", "(void *)0", as
//!   distinct from other void pointers.
//! - If (x) is an integer constant expression, then the "* 0l" resolves
//!   it into an integer constant expression of value 0. Since it is cast to
//!   "void *", this makes the second operand a null pointer constant.
//! - If (x) is not an integer constant expression, then the second operand
//!   resolves to a void pointer (but not a null pointer constant: the value
//!   is not an integer constant 0).
//! - The conditional operator's third operand, "(int *)8", is an object
//!   pointer (to type "int").
//! - The behavior (including the return type) of the conditional operator
//!   ("operand1 ? operand2 : operand3") depends on the kind of expressions
//!   given for the second and third operands. This is the central mechanism
//!   of the macro:
//!   - When one operand is a null pointer constant (i.e. when x is an integer
//!     constant expression) and the other is an object pointer (i.e. our
//!     third operand), the conditional operator returns the type of the
//!     object pointer operand (i.e. "int *"). Here, within the sizeof(), we
//!     would then get:
//!       sizeof(*((int *)(...))  == sizeof(int)  == 4
//!   - When one operand is a void pointer (i.e. when x is not an integer
//!     constant expression) and the other is an object pointer (i.e. our
//!     third operand), the conditional operator returns a "void *" type.
//!     Here, within the sizeof(), we would then get:
//!       sizeof(*((void *)(...)) == sizeof(void) == 1
//! - The equality comparison to "sizeof(int)" therefore depends on (x):
//!     sizeof(int) == sizeof(int)     (x) was a constant expression
//!     sizeof(int) != sizeof(void)    (x) was not a constant expression
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! Monochrome Text Display
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
//! This is the maximum number of entries in struct boot_params::e820_table
//! (the zeropage), which is part of the x86 boot protocol ABI:
//! Defines x86 CPU feature bits
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
// max number of signatures to store
// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
// bool and also pointer types.
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
// N 32-bit words worth of info
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
macro_rules! __do_div {
($n:expr, $base:expr) => {
{
let mut __v: i32;
let __v2: i32 = ((($n) as u32) % (($base) as u32)) as i32;
$n = ((($n) as u32) / (($base) as u32)) as i32;
__v = __v2;
__v
}
};
}

macro_rules! va_arg__4 {
($v:expr) => {
unsafe { __slate_i386_va::arg::<_, *mut ()>(core::ptr::addr_of_mut!($v)) }
};
}

macro_rules! va_arg__2 {
($v:expr) => {
unsafe { __slate_i386_va::arg::<_, *mut i32>(core::ptr::addr_of_mut!($v)) }
};
}

macro_rules! va_arg__3 {
($v:expr) => {
unsafe { __slate_i386_va::arg::<_, u32>(core::ptr::addr_of_mut!($v)) }
};
}

macro_rules! va_arg__5 {
($v:expr) => {
unsafe { __slate_i386_va::arg::<_, *mut i8>(core::ptr::addr_of_mut!($v)) }
};
}

macro_rules! va_arg {
($v:expr) => {
unsafe { __slate_i386_va::arg::<_, i32>(core::ptr::addr_of_mut!($v)) }
};
}

macro_rules! va_end {
() => {
};
}

mod __slate_i386_va {
    pub unsafe fn arg<L, T>(list: *mut L) -> T {
        unsafe {
            let cursor = &mut *(list as *mut *const u8);
            if core::mem::align_of::<T>() >= 16 {
                *cursor = ((*cursor as usize + 15) & !15) as *const u8;
            }
            let size = core::mem::size_of::<T>();
            let value = core::ptr::read_unaligned(*cursor as *const T);
            *cursor = cursor.add((size + 3) & !3);
            value
        }
    }
}


unsafe extern "C" {
    fn strnlen(arg0: *const i8, arg1: u32) -> u32;
    fn puts(arg0: *const i8);
}

/// Optional: only supported since gcc >= 15, clang >= 19
///
///   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
/// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
fn isdigit(ch: i32) -> i32 {
(ch >= 48 && ch <= 57) as i32
}

// Oh, it's a waste of space, but oh-so-yummy for debugging.  This
// version of printf() does not include 64-bit support.  "Live with
// it."
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

// pad with zero
// unsigned/signed long
// show plus
// space if plus
// left justified
// Must be 32 == 0x20
// 0x
fn number(mut str: *mut i8, mut num: i32, base: i32, mut size: i32, mut precision: i32, mut r#type: i32) -> *mut i8 {
// we are called with base 8, 10 or 16, only, thus don't need "G..."
// "GHIJKLMNOPQRSTUVWXYZ";
let mut tmp: [i8; 66] = [0i8; 66];
let mut c: i8 = 0;
let mut sign: i8 = 0;
let mut locase: i8 = 0;
let mut i: i32 = 0;
// locase = 0 or 0x20. ORing digits or letters with 'locase'
// produces same digits or (maybe lowercased) letters
locase = (r#type & 32) as i8;
if r#type & 16 != 0 {
let __v1: i32 = r#type & !1;
r#type = __v1;
}
if base < 2 || base > 16 {
return core::ptr::null_mut::<i8>();
}
c = (if r#type & 1 != 0 { 48i32 } else { 32i32 }) as i8;
if r#type & 2 != 0 {
if num < 0 {
sign = 45;
num = -num;
size -= 1;
} else {
if r#type & 4 != 0 {
sign = 43;
size -= 1;
} else {
if r#type & 8 != 0 {
sign = 32;
size -= 1;
}
}
}
}
if r#type & 64 != 0 {
if base == 16 {
size -= 2;
} else {
if base == 8 {
size -= 1;
}
}
}
if num == 0 {
let __v12: i32 = i;
i = __v12 + 1;
tmp[__v12 as usize] = 48;
} else {
while num != 0 {
let __v14: i32 = i;
i = __v14 + 1;
tmp[__v14 as usize] = (((unsafe { digits[__do_div!(num, base) as usize] }) as i32) | (locase as i32)) as i8;
}
}
if i > precision {
precision = i;
}
size -= precision;
if r#type & 1 + 16 == 0 {
loop {
let __v23: i32 = size;
size = __v23 - 1;
if __v23 <= 0 {
break;
}
let __v21: *mut i8 = str;
str = unsafe { __v21.add(1) };
unsafe {
*__v21 = 32;
}
}
}
if sign != 0 {
let __v26: *mut i8 = str;
str = unsafe { __v26.add(1) };
unsafe {
*__v26 = sign;
}
}
if r#type & 64 != 0 {
if base == 8 {
let __v28: *mut i8 = str;
str = unsafe { __v28.add(1) };
unsafe {
*__v28 = 48;
}
} else {
if base == 16 {
let __v30: *mut i8 = str;
str = unsafe { __v30.add(1) };
unsafe {
*__v30 = 48;
}
let __v32: *mut i8 = str;
str = unsafe { __v32.add(1) };
unsafe {
*__v32 = (88 | (locase as i32)) as i8;
}
}
}
}
if r#type & 16 == 0 {
loop {
let __v36: i32 = size;
size = __v36 - 1;
if __v36 <= 0 {
break;
}
let __v34: *mut i8 = str;
str = unsafe { __v34.add(1) };
unsafe {
*__v34 = c;
}
}
}
loop {
let __v41: i32 = precision;
precision = __v41 - 1;
if i >= __v41 {
break;
}
let __v39: *mut i8 = str;
str = unsafe { __v39.add(1) };
unsafe {
*__v39 = 48;
}
}
loop {
let __v46: i32 = i;
i = __v46 - 1;
if __v46 <= 0 {
break;
}
let __v44: *mut i8 = str;
str = unsafe { __v44.add(1) };
unsafe {
*__v44 = tmp[i as usize];
}
}
loop {
let __v51: i32 = size;
size = __v51 - 1;
if __v51 <= 0 {
break;
}
let __v49: *mut i8 = str;
str = unsafe { __v49.add(1) };
unsafe {
*__v49 = 32;
}
}
str
}

static digits: [i8; 16] = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70];

#[unsafe(no_mangle)]
extern "C" fn vsprintf(buf: *mut i8, mut fmt: *const i8, mut args: *mut i8) -> i32 {
// 'h', 'l', or 'L' for integer fields
let mut qualifier: i32 = unsafe { core::mem::zeroed() };
// min. # of digits for integers; max
// number of chars for from string
let mut precision: i32 = unsafe { core::mem::zeroed() };
// width of output field
let mut field_width: i32 = unsafe { core::mem::zeroed() };
// flags to number()
let mut flags: i32 = unsafe { core::mem::zeroed() };
let mut s: *const i8 = unsafe { core::mem::zeroed() };
let mut str: *mut i8 = unsafe { core::mem::zeroed() };
let mut base: i32 = unsafe { core::mem::zeroed() };
let mut i: i32 = unsafe { core::mem::zeroed() };
let mut num: u32 = unsafe { core::mem::zeroed() };
let mut len: i32 = unsafe { core::mem::zeroed() };
unsafe {
str = buf;
while *fmt != 0 {
'__join_2: {
if (*fmt as i32) != 37 {
let __v65: *mut i8 = str;
str = __v65.add(1);
*__v65 = *fmt;
} else {
// process flags
flags = 0;
loop {
let __v63: *const i8 = fmt;
fmt = __v63.add(1); // this also skips first '%'
let __t1: i32 = *fmt as i32;
if __t1 == 45 {
flags |= 16;
} else {
if __t1 == 43 {
flags |= 4;
} else {
if __t1 == 32 {
flags |= 8;
} else {
if __t1 == 35 {
flags |= 64;
} else {
if __t1 == 48 {
flags |= 1;
} else {
break;
}
}
}
}
}
}
// get field width
field_width = -1;
if isdigit(*fmt as i32) != 0 {
field_width = skip_atoi(&mut (*core::ptr::addr_of_mut!(fmt)));
} else {
if (*fmt as i32) == 42 {
let __v51: *const i8 = fmt;
fmt = __v51.add(1);
// it's the next argument
field_width = va_arg!(args);
if field_width < 0 {
field_width = -field_width;
flags |= 16;
}
}
}
// get the precision
precision = -1;
if (*fmt as i32) == 46 {
let __v47: *const i8 = fmt;
fmt = __v47.add(1);
if isdigit(*fmt as i32) != 0 {
precision = skip_atoi(&mut (*core::ptr::addr_of_mut!(fmt)));
} else {
if (*fmt as i32) == 42 {
let __v45: *const i8 = fmt;
fmt = __v45.add(1);
// it's the next argument
precision = va_arg!(args);
}
}
if precision < 0 {
precision = 0;
}
}
// get the conversion qualifier
qualifier = -1;
if (*fmt as i32) == 104 || (*fmt as i32) == 108 || (*fmt as i32) == 76 {
qualifier = *fmt as i32;
let __v43: *const i8 = fmt;
fmt = __v43.add(1);
}
'__join_11: {
'__join_16: {
// default base
base = 10;
let __t0: i32 = *fmt as i32;
if __t0 == 99 {
if flags & 16 == 0 {
loop {
let __v38: i32 = field_width - 1;
field_width = __v38;
if __v38 > 0 {
let __v41: *mut i8 = str;
str = __v41.add(1);
*__v41 = 32;
} else {
break;
}
}
}
let __v37: *mut i8 = str;
str = __v37.add(1);
*__v37 = (va_arg!(args) as u8) as i8;
loop {
let __v32: i32 = field_width - 1;
field_width = __v32;
if __v32 > 0 {
let __v35: *mut i8 = str;
str = __v35.add(1);
*__v35 = 32;
} else {
break '__join_2;
}
}
} else {
if __t0 == 115 {
s = va_arg__5!(args) as *const i8;
len = strnlen(s, precision as u32) as i32;
if flags & 16 == 0 {
loop {
let __v29: i32 = field_width;
field_width = __v29 - 1;
if len < __v29 {
let __v31: *mut i8 = str;
str = __v31.add(1);
*__v31 = 32;
} else {
break;
}
}
}
i = 0;
while i < len {
let __v27: *const i8 = s;
s = __v27.add(1);
let __v25: *mut i8 = str;
str = __v25.add(1);
*__v25 = *__v27;
i += 1;
}
loop {
let __v19: i32 = field_width;
field_width = __v19 - 1;
if len < __v19 {
let __v21: *mut i8 = str;
str = __v21.add(1);
*__v21 = 32;
} else {
break '__join_2;
}
}
} else {
if __t0 == 112 {
if field_width == -1 {
field_width = 2u32.wrapping_mul(core::mem::size_of::<*mut ()>() as u32) as i32;
flags |= 1;
}
str = number(str, (va_arg__4!(args) as u32) as i32, 16i32, field_width, precision, flags);
break '__join_2;
} else {
if __t0 == 110 {
if qualifier == 108 {
let ip: *mut i32 = va_arg__2!(args);
*ip = str.offset_from(buf as *mut i8) as i32;
break '__join_2;
} else {
let ip: *mut i32 = va_arg__2!(args);
*ip = str.offset_from(buf as *mut i8) as i32;
break '__join_2;
}
} else {
if __t0 == 37 {
let __v13: *mut i8 = str;
str = __v13.add(1);
*__v13 = 37;
break '__join_2;
} else {
if __t0 == 111 {
base = 8;
break '__join_11;
} else {
if __t0 == 120 {
flags |= 32;
break '__join_16;
} else {
if __t0 == 88 {
break '__join_16;
} else {
if __t0 != 100 {
if __t0 != 105 {
if __t0 == 117 {
break '__join_11;
} else {
let __v7: *mut i8 = str;
str = __v7.add(1);
*__v7 = 37;
if *fmt != 0 {
let __v3: *mut i8 = str;
str = __v3.add(1);
*__v3 = *fmt;
break '__join_2;
} else {
let __v5: *const i8 = fmt;
fmt = __v5.sub(1);
break '__join_2;
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
flags |= 2;
break '__join_11;
}
base = 16;
}
if qualifier == 108 {
num = va_arg__3!(args);
} else {
if qualifier == 104 {
num = (va_arg!(args) as u16) as u32;
if flags & 2 != 0 {
num = (num as i16) as u32;
}
} else {
if flags & 2 != 0 {
num = va_arg!(args) as u32;
} else {
num = va_arg__3!(args);
}
}
}
str = number(str, num as i32, base, field_width, precision, flags);
}
}
let __v1: *const i8 = fmt;
fmt = __v1.add(1);
}
*str = 0;
return str.offset_from(buf as *mut i8) as i32;
}
// integer number formats - set up the flags and "break"
}

#[unsafe(no_mangle)]
unsafe extern "C" fn sprintf(buf: *mut i8, fmt: *const i8, mut __va_args: ...) -> i32 {
let mut i: i32 = 0;
let args: *mut i8 = unsafe { core::mem::transmute::<core::ffi::VaList<'_>, *mut i8>(__va_args.clone()) };
i = vsprintf(buf, fmt, args);
va_end!();
i
}

#[unsafe(no_mangle)]
unsafe extern "C" fn printf(fmt: *const i8, mut __va_args: ...) -> i32 {
let mut printf_buf: [i8; 1024] = [0i8; 1024];
let mut printed: i32 = 0;
let args: *mut i8 = unsafe { core::mem::transmute::<core::ffi::VaList<'_>, *mut i8>(__va_args.clone()) };
printed = vsprintf(printf_buf.as_mut_ptr() as *mut i8, fmt, args);
va_end!();
unsafe { puts(printf_buf.as_mut_ptr() as *const i8) };
printed
}
