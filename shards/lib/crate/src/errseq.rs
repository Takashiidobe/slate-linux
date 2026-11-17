
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
//! See linux/instrumentation.h
//! Begin/end of an instrumentation safe region
//! 64 bit
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
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
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
//! label
//! "popcnt" POPCNT instruction
//! This counts to 15. Any more, it will return 16th argument.
//! Atomic operations are already serializing on x86
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
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// An exchange-type operation, which takes a value and a pointer, and
// returns the old value.
// Note: no "lock" prefix even on SMP: xchg always implies lock anyway.
// Since this is generally used to protect other memory information, we
// use "asm volatile" and "memory" clobbers to prevent gcc from moving
// information around.
// Atomic compare and exchange.  Compare OLD with MEM, if identical,
// store NEW in MEM.  Return the initial value in MEM.  Success is
// indicated by comparing RETURN with OLD.
// xadd() adds "inc" to "*ptr" and atomically returns the previous
// value of "*ptr".
//
// xadd() is locked when multiple CPUs are online
// Concatenate two parameters, but allow them to be expanded beforehand.
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
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
macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

/// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
/// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
/// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
macro_rules! IS_ENABLED {
() => {
__is_defined!()
};
}

macro_rules! ____is_defined {
() => {
0i32
};
}

/// Getting something that works in C and CPP for an arg that may or may
/// not be defined is tricky.  Here, if we have "#define CONFIG_BOOGER 1"
/// we match on the placeholder define, insert the "0," for arg1 and generate
/// the triplet (0, 1, 0).  Then the last step cherry picks the 2nd arg (a one).
/// When CONFIG_BOOGER is not defined, we generate a (... 1, 0) pair, and when
/// the last step cherry picks the 2nd arg, we get a zero.
macro_rules! __is_defined {
() => {
____is_defined!()
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
() => {
((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
/// @n: parameter
///
/// constant-capable log of base 2 calculation
/// - this can be used to initialise global variables from constant data, hence
/// the massive ternary operator construction
///
/// selects the appropriately-sized optimised version depending on sizeof(n)
macro_rules! ilog2 {
($n:expr) => {
if true { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { if (core::mem::size_of::<i32>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64(($n) as u64) } }
};
}

macro_rules! kcsan_check_atomic_read_write {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 7i32);
};
}

macro_rules! kcsan_mb {
() => {
while false {
}
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
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
    static mut __SCK__WARN_trap: static_call_key;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __cmpxchg_wrong_size();
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __compiletime_assert_12();
    fn __compiletime_assert_15();
    fn __compiletime_assert_17();
    fn __compiletime_assert_19();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_f217c4720f04f6db(arg0: u64) -> bool;
}

fn kasan_check_write(p: *const (), size: u32) -> bool {
true
}

fn kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

// kcsan_check_read - check regular read access for races
//
// @ptr: address of access
// @size: size of access
// kcsan_check_write - check regular write access for races
//
// @ptr: address of access
// @size: size of access
// kcsan_check_read_write - check regular read-write access for races
//
// @ptr: address of access
// @size: size of access
#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


// __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
// the symbol table so that objtool can reference it when it generates the
// .static_call_sites section.
#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_key {
    func: *mut (),
    __slate_anon_1: __SlateRecord_static_call_types_h_65,
}


#[inline(always)]
unsafe extern "C" fn __WARN_validate_printf(fmt: *const u8, __va_args: ...) {
}

// inhibit tail-call optimization
#[repr(C)]
#[derive(Clone, Copy)]
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


#[inline(always)]
fn instrument_atomic_check_alignment(v: *const (), size: u64) {
if IS_ENABLED!() != 0 {
let mut mask: u32 = size.wrapping_sub(1) as u32;
if IS_ENABLED!() != 0 {
mask = ((mask as u64) & (core::mem::size_of::<__SlateRecord_instrumented_h_66>() as u64).wrapping_sub(1)) as u32;
}
let mut __v2: i64;
let __ret_warn_on: i32 = ((v as u64) & (mask as u64) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("8: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 8b - ., 3; .popsection\n# {0}", const 8 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_instrument_atomic_check_alignment_3, sym __slate_asm_string_instrument_atomic_check_alignment_4, const 67 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("9: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 9b - ., 4; .popsection\n# {0}", const 9 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
}

#[inline(always)]
fn instrument_atomic_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_read_write!(v, size);
instrument_atomic_check_alignment(v, size);
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_errseq_set: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_errseq_sample: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_errseq_check: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_errseq_check_and_advance: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous", sym errseq_set, sym errseq_sample, sym errseq_check, sym errseq_check_and_advance, options(att_syntax));

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.errseq.errseq_set")]
extern "C" fn errseq_set(eseq: *mut u32, err: i32) -> u32 {
let mut cur: u32 = 0;
let mut old: u32 = 0;
// Ensure the error code actually fits where we want it to go. If it
// doesn't then just throw a warning and don't record anything. We
// also don't accept zero here as that would effectively clear a
// previous error.
let mut __v0: u32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_12() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(eseq as *const u32) };
old = __v0;
let mut __v2: i64;
let __ret_warn_on: i32 = (unlikely!(err == 0 || (err.wrapping_neg() as u32) > 4095) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"err = %d\n".as_ptr() as *const u8, err) };
let mut __v3: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v3 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v4: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_errseq_set_5, sym __slate_asm_string_errseq_set_6, const 76 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v4 = bug;
unsafe { __v3.unwrap()(__v4, err) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
return old;
}
loop {
let mut new: u32 = 0;
// Clear out error bits and set new error
new = old & (!((1i32 << ilog2!(4095i32).wrapping_add(1)).wrapping_sub(1) | 1i32 << ilog2!(4095i32).wrapping_add(1)) as u32) | (err.wrapping_neg() as u32);
// Only increment if someone has looked at it
if old & ((1i32 << ilog2!(4095i32).wrapping_add(1)) as u32) != 0 {
let __v8: u32 = new;
new = __v8.wrapping_add((1i32 << ilog2!(4095i32).wrapping_add(1).wrapping_add(1)) as u32);
}
// If there would be no change, then call it done
if new == old {
cur = new;
break;
}
// Try to swap the new value into place
let mut __v11: u32;
let __ai_ptr: *mut u32 = eseq;
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<u32>() as u64);
let mut __v13: u32;
let mut __ret: u32 = 0;
let mut __old: u32 = old;
let mut __new: u32 = new;
match (const { core::mem::size_of::<u32>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = __ai_ptr as *mut u8;
unsafe {
let __t1: u32;
core::arch::asm!("lock cmpxchgb {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t1, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
}
2 => {
let mut __ptr: *mut u16 = __ai_ptr as *mut u16;
unsafe {
let __t2: u32;
core::arch::asm!("lock cmpxchgw {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t2, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
}
4 => {
let mut __ptr: *mut u32 = __ai_ptr as *mut u32;
unsafe {
let __t3: u32;
core::arch::asm!("lock cmpxchgl {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t3, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t3 as u32) };
}
}
8 => {
let mut __ptr: *mut u64 = __ai_ptr as *mut u64;
unsafe {
let __t4: u32;
core::arch::asm!("lock cmpxchgq {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t4, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t4 as u32) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
__v13 = __ret;
__v11 = __v13;
cur = __v11;
// Call it success if we did the swap or someone else beat us
// to it for the same value.
if likely!(cur == old || cur == new) != 0 {
break;
}
// Raced with an update, try again
old = cur;
}
cur
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.errseq.errseq_sample")]
extern "C" fn errseq_sample(eseq: *mut u32) -> u32 {
let mut old: u32 = 0;
let mut __v0: u32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_15() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(eseq as *const u32) };
old = __v0;
// If nobody has seen this error yet, then we can be the first.
if old & ((1i32 << ilog2!(4095i32).wrapping_add(1)) as u32) == 0 {
old = 0;
}
old
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.errseq.errseq_check")]
extern "C" fn errseq_check(eseq: *mut u32, since: u32) -> i32 {
let mut cur: u32 = 0;
let mut __v0: u32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_17() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(eseq as *const u32) };
cur = __v0;
if likely!(cur == since) != 0 {
return 0;
}
(cur & ((1i32 << ilog2!(4095i32).wrapping_add(1)).wrapping_sub(1) as u32)).wrapping_neg() as i32
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.errseq.errseq_check_and_advance")]
extern "C" fn errseq_check_and_advance(eseq: *mut u32, since: *mut u32) -> i32 {
let mut err: i32 = 0;
let mut old: u32 = 0;
let mut new: u32 = 0;
// Most callers will want to use the inline wrapper to check this,
// so that the common case of no error is handled without needing
// to take the lock that protects the "since" value.
let mut __v0: u32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_19() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(eseq as *const u32) };
old = __v0;
if old != unsafe { *since } {
// Set the flag and try to swap it into place if it has
// changed.
//
// We don't care about the outcome of the swap here. If the
// swap doesn't occur, then it has either been updated by a
// writer who is altering the value in some way (updating
// counter or resetting the error), or another reader who is
// just setting the "seen" flag. Either outcome is OK, and we
// can advance "since" and return an error based on what we
// have.
new = old | ((1i32 << ilog2!(4095i32).wrapping_add(1)) as u32);
if new != old {
let mut __v2: u32;
let __ai_ptr: *mut u32 = eseq;
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<u32>() as u64);
let mut __v4: u32;
let mut __ret: u32 = 0;
let mut __old: u32 = old;
let mut __new: u32 = new;
match (const { core::mem::size_of::<u32>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = __ai_ptr as *mut u8;
unsafe {
let __t0: u32;
core::arch::asm!("lock cmpxchgb {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t0, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
}
2 => {
let mut __ptr: *mut u16 = __ai_ptr as *mut u16;
unsafe {
let __t1: u32;
core::arch::asm!("lock cmpxchgw {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t1, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
}
4 => {
let mut __ptr: *mut u32 = __ai_ptr as *mut u32;
unsafe {
let __t2: u32;
core::arch::asm!("lock cmpxchgl {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t2, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
}
8 => {
let mut __ptr: *mut u64 = __ai_ptr as *mut u64;
unsafe {
let __t3: u32;
core::arch::asm!("lock cmpxchgq {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t3, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t3 as u32) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
__v4 = __ret;
__v2 = __v4;
}
unsafe {
*since = new;
}
err = (new & ((1i32 << ilog2!(4095i32).wrapping_add(1)).wrapping_sub(1) as u32)).wrapping_neg() as i32;
}
err
}

// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
#[inline(always)]
fn fls(mut x: u32) -> i32 {
let mut r: i32 = 0;
if unsafe { __slate_intrinsic_1c5364fcd9da51fe(x) } {
return if x != 0 { 32i32.wrapping_sub(x.leading_zeros() as i32) } else { 0 };
}
// AMD64 says BSRL won't clobber the dest reg if x==0; Intel64 says the
// dest reg is undefined if x==0, but their CPU architect says its
// value is written to set it to the same as before, except that the
// top 32 bits will be cleared.
//
// We cannot do this on 32 bits because at the very least some
// 486 CPUs did not behave this way.
unsafe {
let __t0: u32;
core::arch::asm!("bsrl {1:e},{0:e}", inlateout(reg) (1 as i32).wrapping_neg() as u32 => __t0, in(reg) x as u32, options(pure, nomem, nostack, att_syntax));
r = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
r.wrapping_add(1)
}

#[inline(always)]
fn fls64(mut x: u64) -> i32 {
let mut bitpos: i32 = 1i32.wrapping_neg();
if unsafe { __slate_intrinsic_f217c4720f04f6db(x) } {
return if x != 0 { 64i32.wrapping_sub(x.leading_zeros() as i32) } else { 0 };
}
// AMD64 says BSRQ won't clobber the dest reg if x==0; Intel64 says the
// dest reg is undefined if x==0, but their CPU architect says its
// value is written to set it to the same as before.
unsafe {
let __t0: u32;
core::arch::asm!("bsrq {1:r},{0:r}", inlateout(reg) bitpos as u32 => __t0, in(reg) x as u64, options(pure, nomem, nostack, att_syntax));
bitpos = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
bitpos.wrapping_add(1)
}

#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
}

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
// An errseq_t is a way of recording errors in one place, and allowing any
// number of "subscribers" to tell whether it has changed since a previous
// point where it was sampled.
//
// It's implemented as an unsigned 32-bit value. The low order bits are
// designated to hold an error code (between 0 and -MAX_ERRNO). The upper bits
// are used as a counter. This is done with atomics instead of locking so that
// these functions can be called from any context.
//
// The general idea is for consumers to sample an errseq_t value. That value
// can later be used to tell whether any new errors have occurred since that
// sampling was done.
//
// Note that there is a risk of collisions if new errors are being recorded
// frequently, since we have so few bits to use as a counter.
//
// To mitigate this, one bit is used as a flag to tell whether the value has
// been sampled since a new value was recorded. That allows us to avoid bumping
// the counter if no one has sampled it since the last time an error was
// recorded.
//
// A new errseq_t should always be zeroed out.  A errseq_t value of all zeroes
// is the special (but common) case where there has never been an error. An all
// zero value thus serves as the "epoch" if one wishes to know whether there
// has ever been an error set since it was first initialized.
// The low bits are designated for error code (max of MAX_ERRNO)
// This bit is used as a flag to indicate whether the value has been seen
// Leverage macro ERRSEQ_SEEN to define errno mask macro here
// The lowest bit of the counter
// errseq_set - set a errseq_t for later reporting
// @eseq: errseq_t field that should be set
// @err: error to set (must be between -1 and -MAX_ERRNO)
//
// This function sets the error in @eseq, and increments the sequence counter
// if the last sequence was sampled at some point in the past.
//
// Any error set will always overwrite an existing error.
//
// Return: The previous value, primarily for debugging purposes. The
// return value should not be used as a previously sampled value in later
// calls as it will not have the SEEN flag set.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_13: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_errseq_set_14: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u32, i32) -> u32>, *mut ()>(Some(errseq_set)) };

// errseq_sample() - Grab current errseq_t value.
// @eseq: Pointer to errseq_t to be sampled.
//
// This function allows callers to initialise their errseq_t variable.
// If the error has been "seen", new callers will not see an old error.
// If there is an unseen error in @eseq, the caller of this function will
// see it the next time it checks for an error.
//
// Context: Any context.
// Return: The current errseq value.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_errseq_sample_16: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u32) -> u32>, *mut ()>(Some(errseq_sample)) };

// errseq_check() - Has an error occurred since a particular sample point?
// @eseq: Pointer to errseq_t value to be checked.
// @since: Previously-sampled errseq_t from which to check.
//
// Grab the value that eseq points to, and see if it has changed @since
// the given value was sampled. The @since value is not advanced, so there
// is no need to mark the value as seen.
//
// Return: The latest error set in the errseq_t or 0 if it hasn't changed.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_errseq_check_18: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u32, u32) -> i32>, *mut ()>(Some(errseq_check)) };

// errseq_check_and_advance() - Check an errseq_t and advance to current value.
// @eseq: Pointer to value being checked and reported.
// @since: Pointer to previously-sampled errseq_t to check against and advance.
//
// Grab the eseq value, and see whether it matches the value that @since
// points to. If it does, then just return 0.
//
// If it doesn't, then the value has changed. Set the "seen" flag, and try to
// swap it into place as the new eseq value. Then, set that value as the new
// "since" value, and return whatever the error portion is set to.
//
// Note that no locking is provided here for concurrent updates to the "since"
// value. The caller must provide that if necessary. Because of this, callers
// may want to do a lockless errseq_check before taking the lock and calling
// this.
//
// Return: Negative errno if one has been stored, or 0 if no new error has
// occurred.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_errseq_check_and_advance_20: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u32, *mut u32) -> i32>, *mut ()>(Some(errseq_check_and_advance)) };

static __slate_asm_string_errseq_set_5: [u8; 10] = *b"err = %d\n\0";

static __slate_asm_string_errseq_set_6: [u8; 42] = *b"/home/takashi/c-corpus/linux/lib/errseq.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";
