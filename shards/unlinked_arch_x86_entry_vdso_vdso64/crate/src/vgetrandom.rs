
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0-only
//!
//! Copyright (C) 2022-2024 Jason A. Donenfeld <Jason@zx2c4.com>. All Rights Reserved.
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
//! @a is a power of 2 value
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
//! Bad address
//! page can be read
//! Zero memory under memory pressure.
//! Flags for getrandom(2)
//!
//! GRND_NONBLOCK	Don't block and return EAGAIN instead
//! GRND_RANDOM		No effect
//! GRND_INSECURE	Return non-cryptographic random bytes
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
// &a[0] degrades to a pointer: a different type from an array
// BUILD_BUG_ON - break compile if a condition is true.
// @condition: the condition which the compiler should know is false.
//
// If you have some code which relies on certain constants being equal, or
// some other compile-time-evaluated condition, you should use BUILD_BUG_ON to
// detect if someone changes it.
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
// page can be written
// don't use a file
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
/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE {
() => {
((core::mem::size_of::<[u32; 13]>() as u64) / (core::mem::size_of::<u32>() as u64)).wrapping_add(__must_be_array!() as u64)
};
}

/// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
/// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
/// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
macro_rules! IS_ENABLED {
() => {
__or!()
};
}

macro_rules! MEMCPY_AND_ZERO_SRC {
($type:ty, $dst:expr, $src:expr, $len:expr) => {
loop {
while ($len) >= (core::mem::size_of::<$type>() as u64) {
__put_unaligned_t!($type, __get_unaligned_t__3!($type, $src), $dst);
__put_unaligned_t__2!($type, 0i32, $src);
let __v: *mut () = $dst;
let __v2: *mut () = (unsafe { (__v as *const u8).wrapping_offset(core::mem::size_of::<$type>() as isize) }) as *mut ();
$dst = __v2;
let __v3: *mut () = $src;
let __v4: *mut () = (unsafe { (__v3 as *const u8).wrapping_offset(core::mem::size_of::<$type>() as isize) }) as *mut ();
$src = __v4;
let mut __v5: $type = $len;
let __v6: $type = __v5.wrapping_sub(core::mem::size_of::<$type>() as u64);
$len = __v6;
}
if true {
break;
}
}
};
}

macro_rules! MEMCPY_AND_ZERO_SRC__2 {
($type:ty, $dst:expr, $src:expr, $len:expr) => {
loop {
while ($len) >= (core::mem::size_of::<$type>() as u64) {
__put_unaligned_t!($type, __get_unaligned_t__4!($type, $src), $dst);
__put_unaligned_t__2!($type, 0i32, $src);
let __v: *mut () = $dst;
let __v2: *mut () = (unsafe { (__v as *const u8).wrapping_offset(core::mem::size_of::<$type>() as isize) }) as *mut ();
$dst = __v2;
let __v3: *mut () = $src;
let __v4: *mut () = (unsafe { (__v3 as *const u8).wrapping_offset(core::mem::size_of::<$type>() as isize) }) as *mut ();
$src = __v4;
let __v5: u64 = $len;
let __v6: u64 = __v5.wrapping_sub(core::mem::size_of::<$type>() as u64);
$len = __v6;
}
if true {
break;
}
}
};
}

macro_rules! MEMCPY_AND_ZERO_SRC__3 {
($type:ty, $dst:expr, $src:expr, $len:expr) => {
loop {
while ($len) >= (core::mem::size_of::<$type>() as u64) {
__put_unaligned_t!($type, __get_unaligned_t!($type, $src), $dst);
__put_unaligned_t__2!($type, 0i32, $src);
let __v: *mut () = $dst;
let __v2: *mut () = (unsafe { (__v as *const $type).wrapping_offset(core::mem::size_of::<$type>() as isize) }) as *mut ();
$dst = __v2;
let __v3: *mut () = $src;
let __v4: *mut () = (unsafe { (__v3 as *const $type).wrapping_offset(core::mem::size_of::<$type>() as isize) }) as *mut ();
$src = __v4;
let __v5: u64 = $len;
let __v6: u64 = __v5.wrapping_sub(core::mem::size_of::<$type>() as u64);
$len = __v6;
}
if true {
break;
}
}
};
}

macro_rules! MEMCPY_AND_ZERO_SRC__4 {
($type:ty, $dst:expr, $src:expr, $len:expr) => {
loop {
while ($len) >= (core::mem::size_of::<$type>() as u64) {
__put_unaligned_t!($type, __get_unaligned_t__2!($type, $src), $dst);
__put_unaligned_t__2!($type, 0i32, $src);
let __v: *mut () = $dst;
let __v2: *mut () = (unsafe { (__v as *const u8).wrapping_offset(core::mem::size_of::<$type>() as isize) }) as *mut ();
$dst = __v2;
let __v3: *mut () = $src;
let __v4: *mut () = (unsafe { (__v3 as *const u8).wrapping_offset(core::mem::size_of::<$type>() as isize) }) as *mut ();
$src = __v4;
let __v5: u64 = $len;
let __v6: u64 = __v5.wrapping_sub(core::mem::size_of::<$type>() as u64);
$len = __v6;
}
if true {
break;
}
}
};
}

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! __WRITE_ONCE__2 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u64, ($val) as u64) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut bool, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__3 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u64, $val) };
if true {
break;
}
}
};
}

macro_rules! ____or {
() => {
1i32
};
}

macro_rules! __cmp_once {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = $y;
__v = if __v2 < __v3 { __v2 } else { __v3 };
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
macro_rules! __get_unaligned_t {
($type:ty, $ptr:expr) => {
{
let mut __v: $type;
core::ptr::null_mut::<u8>();
let mut __v3: $type = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), ($ptr) as *const (), core::mem::size_of::<$type>() as u64) };
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
($type:ty, $ptr:expr) => {
{
let mut __v: $type;
core::ptr::null_mut::<u16>();
let mut __v3: $type = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), ($ptr) as *const (), core::mem::size_of::<$type>() as u64) };
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
($type:ty, $ptr:expr) => {
{
let mut __v: $type;
core::ptr::null_mut::<u64>();
let mut __v3: $type = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), ($ptr) as *const (), core::mem::size_of::<$type>() as u64) };
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
($type:ty, $ptr:expr) => {
{
let mut __v: $type;
core::ptr::null_mut::<u32>();
let mut __v3: $type = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), ($ptr) as *const (), core::mem::size_of::<$type>() as u64) };
__v = __v3;
__v
}
};
}

macro_rules! __must_be_array {
() => {
core::mem::size_of::<__SlateRecord_getrandom_c_86>() as i32
};
}

macro_rules! __or {
() => {
____or!()
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

/// __put_unaligned_t - write an unaligned value to memory.
/// @type:	the type of the value to store.
/// @val:	the value to store.
/// @ptr:	the pointer to store to.
///
/// Use memcpy to affect an unaligned type sized store avoiding undefined
/// behavior from approaches like type punning that require -fno-strict-aliasing
/// in order to be correct. The void* cast silences ubsan warnings.
macro_rules! __put_unaligned_t__2 {
($type:ty, $val:expr, $ptr:expr) => {
loop {
let mut __put_unaligned_val: $type = ($val) as $type;
unsafe { __builtin_memcpy($ptr, core::ptr::addr_of_mut!(__put_unaligned_val) as *const (), core::mem::size_of::<$type>() as u64) };
if true {
break;
}
}
};
}

macro_rules! kcsan_rmb {
() => {
while false {
}
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t {
($type:ty, $x:expr, $y:expr) => {
__cmp_once!($type, $x, $y)
};
}

macro_rules! unlikely__2 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely {
($x:expr) => {
!(!$x) as i64
};
}

unsafe extern "C" {
    static mut vdso_u_rng_data: vdso_rng_data;
    fn __arch_chacha20_blocks_nostack(arg0: *mut u8, arg1: *const u32, arg2: *mut u32, arg3: u64);
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __compiletime_assert_10();
    fn __compiletime_assert_11();
    fn __compiletime_assert_12();
    fn __compiletime_assert_13();
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn __compiletime_assert_16();
    fn __compiletime_assert_19();
    fn __compiletime_assert_20();
    fn __compiletime_assert_21();
    fn __compiletime_assert_22();
    fn __compiletime_assert_23();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vdso_rng_data {
    generation: u64,
    is_ready: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vgetrandom_state {
    __slate_anon_0: __SlateRecord_getrandom_h_34,
    generation: u64,
    pos: u8,
    in_use: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_getrandom_h_34 {
    __slate_anon_0: __SlateRecord_getrandom_h_35,
    batch_key: [u8; 128],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_getrandom_h_35 {
    batch: [u8; 96],
    key: [u32; 8],
}


#[inline(always)]
fn getrandom_syscall(mut buffer: *mut (), mut len: u64, mut flags: u32) -> i64 {
let mut ret: i64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("syscall", inlateout("ax") ((318 as i32) as u32) as u64 => __t0, in("di") buffer as u64, in("si") len as u64, in("dx") flags as u32, out("cx") _, out("r11") _, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u64, i64>(__t0 as u64) };
}
ret
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_getrandom_c_86 {
}


core::arch::global_asm!(".weak getrandom\n.set getrandom, {}", sym __vdso_getrandom);

#[repr(C)]
#[derive(Clone, Copy)]
struct vgetrandom_opaque_params {
    size_of_opaque_state: u32,
    mmap_prot: u32,
    mmap_flags: u32,
    reserved: [u32; 13],
}


#[inline(always)]
fn __arch_get_vdso_u_rng_data() -> *const vdso_rng_data {
(unsafe { core::ptr::addr_of_mut!(vdso_u_rng_data) }) as *const vdso_rng_data
}

fn memcpy_and_zero_src(mut dst: *mut (), mut src: *mut (), mut len: u64) {
if IS_ENABLED!() != 0 {
if IS_ENABLED!() != 0 {
MEMCPY_AND_ZERO_SRC!(u64, dst, src, len);
}
MEMCPY_AND_ZERO_SRC__2!(u32, dst, src, len);
MEMCPY_AND_ZERO_SRC__4!(u16, dst, src, len);
}
MEMCPY_AND_ZERO_SRC__3!(u8, dst, src, len);
}

#[inline(always)]
fn __cvdso_getrandom_data(rng_info: *const vdso_rng_data, mut buffer: *mut (), mut len: u64, flags: u32, opaque_state: *mut (), opaque_len: u64) -> i64 {
// Since @rng_info->generation will never be 0, re-read @state->generation, rather
// than using the local current_generation variable, to learn whether a fork
// occurred or if @state was zeroed due to memory pressure. Primarily, though, this
// indicates whether the kernel's RNG has reseeded, in which case generate a new key
// and start over.
let mut __v5: u64 = unsafe { core::mem::zeroed() };
let mut i: u64 = unsafe { core::mem::zeroed() };
let mut params: *mut vgetrandom_opaque_params = unsafe { core::mem::zeroed() };
let mut counter: [u32; 2] = unsafe { core::mem::zeroed() };
let mut current_generation: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u64 = core::ptr::addr_of_mut!(current_generation) as *mut u64;
let mut orig_buffer: *mut () = unsafe { core::mem::zeroed() };
let mut have_retried: bool = unsafe { core::mem::zeroed() };
let mut in_use: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut bool = core::ptr::addr_of_mut!(in_use) as *mut bool;
let mut orig_len: u64 = unsafe { core::mem::zeroed() };
let mut nblocks: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u64 = core::ptr::addr_of_mut!(nblocks) as *mut u64;
let mut batch_len: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u64 = core::ptr::addr_of_mut!(batch_len) as *mut u64;
let mut state: *mut vgetrandom_state = unsafe { core::mem::zeroed() };
// = MAX_RW_COUNT
let mut ret: i64 = unsafe { core::mem::zeroed() };
unsafe {
ret = min_t!(u64, 2147483647u64 & !(_AC!() << 12i32).wrapping_sub(1), len) as i64;
state = opaque_state as *mut vgetrandom_state;
let __slate_auto_bytes = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
orig_len = len;
let __slate_auto_bytes_2 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
have_retried = false;
orig_buffer = buffer;
let __slate_auto_bytes = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
counter = {
let mut __t0: [u32; 2] = core::mem::zeroed();
__t0[0] = 0u32;
__t0
};
if unlikely!(opaque_len == !0 && buffer.is_null() && len == 0 && flags == 0) != 0 {
params = opaque_state as *mut vgetrandom_opaque_params;
(*params).size_of_opaque_state = core::mem::size_of::<vgetrandom_state>() as u32;
(*params).mmap_prot = 3u32;
(*params).mmap_flags = 40u32;
i = 0;
while i < ARRAY_SIZE!() {
*((*params).reserved.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) = 0;
i = i.wrapping_add(1);
}
return 0;
} else {
// The state must not straddle a page, since pages can be zeroed at any time.
if unlikely!(((opaque_state as u64) & !(!(_AC!() << 12i32).wrapping_sub(1))).wrapping_add(core::mem::size_of::<vgetrandom_state>() as u64) > _AC!() << 12i32) != 0 {
return 14i32.wrapping_neg() as i64;
} else {
'__join_0: {
// Handle unexpected flags by falling back to the kernel.
if unlikely__2!(flags & 4294967288u32) == 0 {
// If the caller passes the wrong size, which might happen due to CRIU, fallback.
if unlikely!(opaque_len != (core::mem::size_of::<vgetrandom_state>() as u64)) == 0 {
while false {
}
// If the kernel's RNG is not yet ready, then it's not possible to provide random bytes from
// userspace, because A) the various @flags require this to block, or not, depending on
// various factors unavailable to userspace, and B) the kernel's behavior before the RNG is
// ready is to reseed from the entropy pool at every invocation.
let __v15: u8 = core::ptr::read_volatile(core::ptr::addr_of!((*rng_info).is_ready) as *const u8);
if __v15 != 0 {
// This condition is checked after @rng_info->is_ready, because before the kernel's RNG is
// initialized, the @flags parameter may require this to block or return an error, even when
// len is zero.
if unlikely!(len == 0) != 0 {
return 0;
} else {
while false {
}
// @state->in_use is basic reentrancy protection against this running in a signal handler
// with the same @opaque_state, but obviously not atomic wrt multiple CPUs or more than one
// level of reentrancy. If a signal interrupts this after reading @state->in_use, but before
// writing @state->in_use, there is still no race, because the signal handler will run to
// its completion before returning execution.
let __v14: bool = core::ptr::read_volatile(core::ptr::addr_of_mut!((*state).in_use) as *const bool);
*__slate_slot_24 = __v14;
if unlikely!(*__slate_slot_24) == 0 {
// The syscall simply fills the buffer and does not touch @state, so fallback.
loop {
if true {
__WRITE_ONCE!((*state).in_use, true);
if true {
break;
}
}
}
'__join_28: {
'__join_11: {
loop {
if true {
let __v13: u64 = core::ptr::read_volatile(core::ptr::addr_of!((*rng_info).generation) as *const u64);
*__slate_slot_21 = __v13;
// @rng_info->generation must always be read here, as it serializes @state->key with the
// kernel's RNG reseeding schedule.
// If @state->generation doesn't match the kernel RNG's generation, then it means the
// kernel's RNG has reseeded, and so @state->key is reseeded as well.
if unlikely!((*state).generation != *__slate_slot_21) != 0 {
// Write the generation before filling the key, in case of fork. If there is a fork
// just after this line, the parent and child will get different random bytes from
// the syscall, which is good. However, were this line to occur after the getrandom
// syscall, then both child and parent could have the same bytes and the same
// generation counter, so the fork would not be detected. Therefore, write
// @state->generation before the call to the getrandom syscall.
loop {
if true {
__WRITE_ONCE__3!((*state).generation, *__slate_slot_21);
if true {
break;
}
}
}
// Prevent the syscall from being reordered wrt current_generation. Pairs with the
// smp_store_release(&vdso_k_rng_data->generation) in random.c.
loop {
kcsan_rmb!();
loop {
kcsan_rmb!();
core::arch::asm!("", options(nostack, att_syntax));
if true {
break;
}
}
if true {
break;
}
}
// Reseed @state->key using fresh bytes from the kernel.
if (getrandom_syscall(((*state).__slate_anon_0.__slate_anon_0.key.as_mut_ptr() as *mut u32) as *mut (), core::mem::size_of::<[u32; 8]>() as u64, 0u32) as u64) != (core::mem::size_of::<[u32; 8]>() as u64) {
break '__join_28;
} else {
// Set @state->pos to beyond the end of the batch, so that the batch is refilled
// using the new key.
(*state).pos = core::mem::size_of::<[u8; 96]>() as u8;
}
}
// Set len to the total amount of bytes that this function is allowed to read, ret.
len = ret as u64;
loop {
*__slate_slot_27 = min_t!(u64, (core::mem::size_of::<[u8; 96]>() as u64).wrapping_sub(((*state).pos as i32) as u64), len);
// First use bytes out of @state->batch, which may have been filled by the last call to this
// function.
if *__slate_slot_27 != 0 {
// Zeroing at the same time as memcpying helps preserve forward secrecy.
memcpy_and_zero_src(buffer, ((*state).__slate_anon_0.__slate_anon_0.batch.as_mut_ptr() as *mut u8).wrapping_offset(((*state).pos as i32) as isize) as *mut (), *__slate_slot_27);
(*state).pos = (((*state).pos as i32) as u64).wrapping_add(*__slate_slot_27) as u8;
buffer = (buffer as *const u8).wrapping_offset(*__slate_slot_27 as isize) as *mut ();
len = len.wrapping_sub(*__slate_slot_27);
}
if len == 0 {
break;
} else {
// Generate blocks of RNG output directly into @buffer while there's enough room left.
*__slate_slot_26 = len / 64;
if *__slate_slot_26 != 0 {
__arch_chacha20_blocks_nostack(buffer as *mut u8, ((*state).__slate_anon_0.__slate_anon_0.key.as_mut_ptr() as *mut u32) as *const u32, counter.as_mut_ptr() as *mut u32, *__slate_slot_26);
buffer = (buffer as *const u8).wrapping_offset((*__slate_slot_26).wrapping_mul(64) as isize) as *mut ();
len = len.wrapping_sub((*__slate_slot_26).wrapping_mul(64));
}
while false {
}
// Refill the batch and overwrite the key, in order to preserve forward secrecy.
__arch_chacha20_blocks_nostack((*state).__slate_anon_0.batch_key.as_mut_ptr() as *mut u8, ((*state).__slate_anon_0.__slate_anon_0.key.as_mut_ptr() as *mut u32) as *const u32, counter.as_mut_ptr() as *mut u32, (core::mem::size_of::<[u8; 128]>() as u64) / 64);
// Since the batch was just refilled, set the position back to 0 to indicate a full batch.
(*state).pos = 0;
}
}
// Prevent the loop from being reordered wrt ->generation.
core::arch::asm!("", options(nostack, att_syntax));
while false {
}
__v5 = core::ptr::read_volatile(core::ptr::addr_of_mut!((*state).generation) as *const u64);
while false {
}
let __v4: u64 = core::ptr::read_volatile(core::ptr::addr_of!((*rng_info).generation) as *const u64);
if __v5 != __v4 {
// Prevent this from looping forever in case of low memory or racing with a
// user force-reseeding the kernel's RNG using the ioctl.
if have_retried {
break '__join_11;
} else {
have_retried = true;
buffer = orig_buffer;
}
} else {
break;
}
}
}
// Set @state->in_use to false only when there will be no more reads or writes of
// @state.
loop {
if true {
__WRITE_ONCE!((*state).in_use, false);
if true {
break;
}
}
}
return ret;
}
loop {
if true {
__WRITE_ONCE!((*state).in_use, false);
if true {
break '__join_0;
}
}
}
}
// If the syscall failed to refresh the key, then @state->key is now
// invalid, so invalidate the generation so that it is not used again, and
// fallback to using the syscall entirely.
loop {
if true {
__WRITE_ONCE__2!((*state).generation, 0i32);
if true {
break;
}
}
}
// Set @state->in_use to false only after the last write to @state in the
// line above.
loop {
if true {
__WRITE_ONCE!((*state).in_use, false);
if true {
break;
}
}
}
}
}
}
}
}
}
return getrandom_syscall(orig_buffer, orig_len, flags);
}
}
}
}

#[inline(always)]
fn __cvdso_getrandom(buffer: *mut (), len: u64, flags: u32, opaque_state: *mut (), opaque_len: u64) -> i64 {
__cvdso_getrandom_data(__arch_get_vdso_u_rng_data(), buffer, len, flags, opaque_state, opaque_len)
}

#[unsafe(no_mangle)]
extern "C" fn __vdso_getrandom(buffer: *mut (), len: u64, flags: u32, opaque_state: *mut (), opaque_len: u64) -> i64 {
__cvdso_getrandom(buffer, len, flags, opaque_state, opaque_len)
}
