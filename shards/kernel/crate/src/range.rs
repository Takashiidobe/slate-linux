
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0
//!
//! Range add and subtract
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
//! BUILD_BUG_ON_MSG - break compile if a condition is true & emit supplied
//! 		      error message.
//! @cond: the condition which the compiler should know is false.
//! @msg: build-time error message
//!
//! See BUILD_BUG_ON for description.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! check_add_overflow() - Calculate addition with overflow checking
//! @a: first addend
//! @b: second addend
//! @d: pointer to store sum
//!
//! Returns true on wrap-around, false otherwise.
//!
//! *@d holds the results of the attempted addition, regardless of whether
//! wrap-around occurred.
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! strlen("ratelimit") + 1
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
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-format-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#format
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
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
// pr_err - Print an error-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_ERR loglevel. It uses pr_fmt() to
// generate the format string.
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
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u64>() }) as u64) < 4) as i32)) }
};
}

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
(1i32.wrapping_neg() as u64) < 1
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
    fn _printk(arg0: *const u8, ...) -> i32;
    fn sort(arg0: *mut (), arg1: u64, arg2: u64, arg3: Option<unsafe extern "C" fn(*const (), *const ()) -> i32>, arg4: Option<unsafe extern "C" fn(*mut (), *mut (), i32)>);
    fn memmove(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn __compiletime_assert_2();
    fn __compiletime_assert_5();
    fn __compiletime_assert_8();
    fn __compiletime_assert_11();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[unsafe(no_mangle)]
extern "C" fn add_range_with_merge(range: *mut range, az: i32, mut nr_range: i32, mut start: u64, mut end: u64) -> i32 {
// get new start/end:
let mut i: i32 = 0;
if start >= end {
return nr_range;
}
while i < nr_range {
let mut common_start: u64 = 0;
let mut common_end: u64 = 0;
if (unsafe { (*range.wrapping_offset(i as isize)).end }) != 0 {
let mut __v1: u64;
let __UNIQUE_ID_x__0: u64 = unsafe { (*range.wrapping_offset(i as isize)).start };
let __UNIQUE_ID_y__1: u64 = start;
loop {
if __types_ok!(__UNIQUE_ID_x__0, __UNIQUE_ID_y__1) == 0 {
unsafe { __compiletime_assert_2() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = if __UNIQUE_ID_x__0 > __UNIQUE_ID_y__1 { __UNIQUE_ID_x__0 } else { __UNIQUE_ID_y__1 };
common_start = __v1;
let mut __v3: u64;
let __UNIQUE_ID_x__3: u64 = unsafe { (*range.wrapping_offset(i as isize)).end };
let __UNIQUE_ID_y__4: u64 = end;
loop {
if __types_ok!(__UNIQUE_ID_x__3, __UNIQUE_ID_y__4) == 0 {
unsafe { __compiletime_assert_5() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__3 < __UNIQUE_ID_y__4 { __UNIQUE_ID_x__3 } else { __UNIQUE_ID_y__4 };
common_end = __v3;
if common_start <= common_end {
// new start/end, will add it back at last
let mut __v5: u64;
let __UNIQUE_ID_x__6: u64 = unsafe { (*range.wrapping_offset(i as isize)).start };
let __UNIQUE_ID_y__7: u64 = start;
loop {
if __types_ok!(__UNIQUE_ID_x__6, __UNIQUE_ID_y__7) == 0 {
unsafe { __compiletime_assert_8() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = if __UNIQUE_ID_x__6 < __UNIQUE_ID_y__7 { __UNIQUE_ID_x__6 } else { __UNIQUE_ID_y__7 };
start = __v5;
let mut __v7: u64;
let __UNIQUE_ID_x__9: u64 = unsafe { (*range.wrapping_offset(i as isize)).end };
let __UNIQUE_ID_y__10: u64 = end;
loop {
if __types_ok!(__UNIQUE_ID_x__9, __UNIQUE_ID_y__10) == 0 {
unsafe { __compiletime_assert_11() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v7 = if __UNIQUE_ID_x__9 > __UNIQUE_ID_y__10 { __UNIQUE_ID_x__9 } else { __UNIQUE_ID_y__10 };
end = __v7;
unsafe { memmove(range.wrapping_offset(i as isize) as *mut (), range.wrapping_offset(i.wrapping_add(1) as isize) as *const (), (nr_range.wrapping_sub(i.wrapping_add(1)) as u64).wrapping_mul(core::mem::size_of::<range>() as u64)) };
unsafe {
(*range.wrapping_offset(nr_range.wrapping_sub(1) as isize)).start = 0;
}
unsafe {
(*range.wrapping_offset(nr_range.wrapping_sub(1) as isize)).end = 0;
}
nr_range = nr_range.wrapping_sub(1);
i = i.wrapping_sub(1);
}
}
i = i.wrapping_add(1);
}
// Need to add it:
add_range(range, az, nr_range, start, end)
}

#[unsafe(no_mangle)]
extern "C" fn add_range(range: *mut range, az: i32, mut nr_range: i32, start: u64, end: u64) -> i32 {
if start >= end {
return nr_range;
}
// Out of slots:
if nr_range >= az {
return nr_range;
}
unsafe {
(*range.wrapping_offset(nr_range as isize)).start = start;
}
unsafe {
(*range.wrapping_offset(nr_range as isize)).end = end;
}
nr_range = nr_range.wrapping_add(1);
nr_range
}

#[unsafe(no_mangle)]
extern "C" fn subtract_range(range: *mut range, az: i32, start: u64, end: u64) {
let mut i: i32 = 0;
if start >= end {
return;
}
for j in 0..az {
if (unsafe { (*range.wrapping_offset(j as isize)).end }) != 0 {
if start <= unsafe { (*range.wrapping_offset(j as isize)).start } && end >= unsafe { (*range.wrapping_offset(j as isize)).end } {
unsafe {
(*range.wrapping_offset(j as isize)).start = 0;
}
unsafe {
(*range.wrapping_offset(j as isize)).end = 0;
}
} else {
if start <= unsafe { (*range.wrapping_offset(j as isize)).start } && end < unsafe { (*range.wrapping_offset(j as isize)).end } && (unsafe { (*range.wrapping_offset(j as isize)).start }) < end {
unsafe {
(*range.wrapping_offset(j as isize)).start = end;
}
} else {
if start > unsafe { (*range.wrapping_offset(j as isize)).start } && end >= unsafe { (*range.wrapping_offset(j as isize)).end } && (unsafe { (*range.wrapping_offset(j as isize)).end }) > start {
unsafe {
(*range.wrapping_offset(j as isize)).end = start;
}
} else {
if start > unsafe { (*range.wrapping_offset(j as isize)).start } && end < unsafe { (*range.wrapping_offset(j as isize)).end } {
// Find the new spare:
i = 0;
while i < az {
if (unsafe { (*range.wrapping_offset(i as isize)).end }) == 0 {
break;
}
i = i.wrapping_add(1);
}
if i < az {
unsafe {
(*range.wrapping_offset(i as isize)).end = (*range.wrapping_offset(j as isize)).end;
}
unsafe {
(*range.wrapping_offset(i as isize)).start = end;
}
} else {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x013%s: run out of slot in ranges\n".as_ptr() as *const u8, c"subtract_range".as_ptr() as *mut u8) };
}
unsafe {
(*range.wrapping_offset(j as isize)).end = start;
}
}
}
}
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.range.cmp_range")]
extern "C" fn cmp_range(x1: *const (), x2: *const ()) -> i32 {
if (unsafe { (*(x1 as *const range)).start }) < unsafe { (*(x2 as *const range)).start } {
return 1i32.wrapping_neg();
}
if (unsafe { (*(x1 as *const range)).start }) > unsafe { (*(x2 as *const range)).start } {
return 1;
}
0
}

#[unsafe(no_mangle)]
extern "C" fn clean_sort_range(range: *mut range, az: i32) -> i32 {
let mut i: i32 = 0;
let mut j: i32 = 0;
let mut k: i32 = az.wrapping_sub(1);
let mut nr_range: i32 = az;
while i < k {
if (unsafe { (*range.wrapping_offset(i as isize)).end }) == 0 {
j = k;
while j > i {
if (unsafe { (*range.wrapping_offset(j as isize)).end }) != 0 {
k = j;
break;
}
j = j.wrapping_sub(1);
}
if j == i {
break;
}
unsafe {
(*range.wrapping_offset(i as isize)).start = (*range.wrapping_offset(k as isize)).start;
}
unsafe {
(*range.wrapping_offset(i as isize)).end = (*range.wrapping_offset(k as isize)).end;
}
unsafe {
(*range.wrapping_offset(k as isize)).start = 0;
}
unsafe {
(*range.wrapping_offset(k as isize)).end = 0;
}
k = k.wrapping_sub(1);
}
i = i.wrapping_add(1);
}
// count it
i = 0;
while i < az {
if (unsafe { (*range.wrapping_offset(i as isize)).end }) == 0 {
nr_range = i;
break;
}
i = i.wrapping_add(1);
}
// sort them
unsafe { sort(range as *mut (), nr_range as u64, core::mem::size_of::<range>() as u64, Some(cmp_range), None) };
nr_range
}

#[unsafe(no_mangle)]
extern "C" fn sort_range(range: *mut range, nr_range: i32) {
// sort them
unsafe { sort(range as *mut (), nr_range as u64, core::mem::size_of::<range>() as u64, Some(cmp_range), None) };
}
