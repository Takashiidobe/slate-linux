
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! inffast.c -- fast decoding
//! Copyright (C) 1995-2004 Mark Adler
//! For conditions of distribution and use, see copyright notice in zlib.h
//! This counts to 15. Any more, it will return 16th argument.
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
//! largest prime smaller than 65536
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
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// Optional: only supported since gcc >= 15, clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
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
#[repr(C)]
#[derive(Clone, Copy)]
struct internal_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct z_stream_s {
    next_in: *const u8,
    avail_in: u64,
    total_in: u64,
    next_out: *mut u8,
    avail_out: u64,
    total_out: u64,
    msg: *mut u8,
    state: *mut internal_state,
    workspace: *mut (),
    data_type: i32,
    adler: u64,
    reserved: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_inftrees_h_31 {
    op: u8,
    bits: u8,
    val: u16,
}


// Maximum size of dynamic tree.  The maximum found in a long but non-
// exhaustive search was 1444 code structures (852 for length/literals
// and 592 for distances, the latter actually the result of an
// exhaustive search).  The true maximum is not known, but the value
// below is more than safe.
#[repr(C)]
#[derive(Clone, Copy)]
struct inflate_state {
    mode: u32,
    last: i32,
    wrap: i32,
    havedict: i32,
    flags: i32,
    dmax: u32,
    check: u64,
    total: u64,
    wbits: u32,
    wsize: u32,
    whave: u32,
    write: u32,
    window: *mut u8,
    hold: u64,
    bits: u32,
    length: u32,
    offset: u32,
    extra: u32,
    lencode: *const __SlateRecord_inftrees_h_31,
    distcode: *const __SlateRecord_inftrees_h_31,
    lenbits: u32,
    distbits: u32,
    ncode: u32,
    nlen: u32,
    ndist: u32,
    have: u32,
    next: *mut __SlateRecord_inftrees_h_31,
    lens: [u16; 320],
    work: [u16; 288],
    codes: [__SlateRecord_inftrees_h_31; 2048],
}


#[repr(C)]
#[derive(Clone, Copy)]
union uu {
    us: u16,
    b: [u8; 2],
}


/// Endian independent version
fn get_unaligned16(p: *const u16) -> u16 {
let mut mm: uu = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(mm) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
}
let b: *mut u8 = p as *mut u8;
unsafe {
mm.b[0] = *b.wrapping_offset(0 as isize);
}
unsafe {
mm.b[1] = *b.wrapping_offset(1 as isize);
}
unsafe { mm.us }
}

/// Decode literal, length, and distance codes and write out the resulting
/// literal and match bytes until either not enough input or output is
/// available, an end-of-block is encountered, or a data error is encountered.
/// When large enough input and output buffers are supplied to inflate(), for
/// example, a 16K input buffer and a 64K output buffer, more than 95% of the
/// inflate execution time is spent in this routine.
///
/// Entry assumptions:
///
///      state->mode == LEN
///      strm->avail_in >= 6
///      strm->avail_out >= 258
///      start >= strm->avail_out
///      state->bits < 8
///
/// On return, state->mode is one of:
///
///      LEN -- ran out of enough output space or enough available input
///      TYPE -- reached end of block code, inflate() to interpret next block
///      BAD -- error in block data
///
/// Notes:
///
///  - The maximum input bits used by a length/distance pair is 15 bits for the
///    length code, 5 bits for the length extra, 15 bits for the distance code,
///    and 13 bits for the distance extra.  This totals 48 bits, or six bytes.
///    Therefore if strm->avail_in >= 6, then there is enough input to avoid
///    checking for available input while decoding.
///
///  - The maximum bytes that a single length/distance pair can output is 258
///    bytes, which is the maximum length that can be coded.  inflate_fast()
///    requires strm->avail_out >= 258 for each loop to avoid checking for
///    output space.
///
///  - @start:	inflate()'s starting value for strm->avail_out
#[unsafe(no_mangle)]
extern "C" fn inflate_fast(strm: *mut z_stream_s, start: u32) {
let mut sfrom: core::mem::MaybeUninit<*mut u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_84: *mut *mut u16 = core::ptr::addr_of_mut!(sfrom) as *mut *mut u16;
let mut mm: core::mem::MaybeUninit<uu> = core::mem::MaybeUninit::uninit();
let __slate_slot_89: *mut uu = core::ptr::addr_of_mut!(mm) as *mut uu;
let mut pat16: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_90: *mut u16 = core::ptr::addr_of_mut!(pat16) as *mut u16;
let mut loops: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_97: *mut u64 = core::ptr::addr_of_mut!(loops) as *mut u64;
let mut sout: core::mem::MaybeUninit<*mut u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_98: *mut *mut u16 = core::ptr::addr_of_mut!(sout) as *mut *mut u16;
// where to copy match from
let mut from: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_172: *mut *mut u8 = core::ptr::addr_of_mut!(from) as *mut *mut u8; // match distance
let mut dist: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_173: *mut u32 = core::ptr::addr_of_mut!(dist) as *mut u32; // match length, unused bytes
let mut len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_174: *mut u32 = core::ptr::addr_of_mut!(len) as *mut u32;
// code bits, operation, extra bits, or
//
//  window position, window bytes to copy
let mut op: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_175: *mut u32 = core::ptr::addr_of_mut!(op) as *mut u32; // retrieved table entry
let mut this: core::mem::MaybeUninit<__SlateRecord_inftrees_h_31> = core::mem::MaybeUninit::uninit();
let __slate_slot_176: *mut __SlateRecord_inftrees_h_31 = core::ptr::addr_of_mut!(this) as *mut __SlateRecord_inftrees_h_31; // mask for first level of distance codes
let mut dmask: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_177: *mut u32 = core::ptr::addr_of_mut!(dmask) as *mut u32; // mask for first level of length codes
let mut lmask: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_178: *mut u32 = core::ptr::addr_of_mut!(lmask) as *mut u32; // local strm->distcode
let mut dcode: core::mem::MaybeUninit<*const __SlateRecord_inftrees_h_31> = core::mem::MaybeUninit::uninit();
let __slate_slot_179: *mut *const __SlateRecord_inftrees_h_31 = core::ptr::addr_of_mut!(dcode) as *mut *const __SlateRecord_inftrees_h_31; // local strm->lencode
let mut lcode: core::mem::MaybeUninit<*const __SlateRecord_inftrees_h_31> = core::mem::MaybeUninit::uninit();
let __slate_slot_180: *mut *const __SlateRecord_inftrees_h_31 = core::ptr::addr_of_mut!(lcode) as *mut *const __SlateRecord_inftrees_h_31; // local strm->bits
let mut bits: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_181: *mut u32 = core::ptr::addr_of_mut!(bits) as *mut u32; // local strm->hold
let mut hold: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_182: *mut u64 = core::ptr::addr_of_mut!(hold) as *mut u64; // allocated sliding window, if wsize != 0
let mut window: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_183: *mut *mut u8 = core::ptr::addr_of_mut!(window) as *mut *mut u8; // window write index
let mut write: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_184: *mut u32 = core::ptr::addr_of_mut!(write) as *mut u32; // valid bytes in the window
let mut whave: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_185: *mut u32 = core::ptr::addr_of_mut!(whave) as *mut u32; // window size or zero if not using window
let mut wsize: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_186: *mut u32 = core::ptr::addr_of_mut!(wsize) as *mut u32; // while out < end, enough space available
let mut end: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_187: *mut *mut u8 = core::ptr::addr_of_mut!(end) as *mut *mut u8; // inflate()'s initial strm->next_out
let mut beg: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_188: *mut *mut u8 = core::ptr::addr_of_mut!(beg) as *mut *mut u8; // local strm->next_out
let mut out: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_189: *mut *mut u8 = core::ptr::addr_of_mut!(out) as *mut *mut u8; // while in < last, enough input available
let mut last: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_190: *mut *const u8 = core::ptr::addr_of_mut!(last) as *mut *const u8; // local strm->next_in
let mut r#in: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_191: *mut *const u8 = core::ptr::addr_of_mut!(r#in) as *mut *const u8;
let mut state: core::mem::MaybeUninit<*mut inflate_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_192: *mut *mut inflate_state = core::ptr::addr_of_mut!(state) as *mut *mut inflate_state;
unsafe {
let __slate_auto_bytes_3 = __slate_slot_192 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_191 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_190 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_189 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_188 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_187 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_9 = __slate_slot_186 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_10 = __slate_slot_185 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_11 = __slate_slot_184 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_12 = __slate_slot_183 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_13 = __slate_slot_182 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_13.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_14 = __slate_slot_181 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_14.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_15 = __slate_slot_180 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_15.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_16 = __slate_slot_179 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_16.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_17 = __slate_slot_178 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_17.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_18 = __slate_slot_177 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_18.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_19 = __slate_slot_176 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_19.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_20 = __slate_slot_175 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_20.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_21 = __slate_slot_174 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_21.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_22 = __slate_slot_173 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_22.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_23 = __slate_slot_172 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_23.add(0usize), 0u8, 8usize);
// copy state to local variables
*__slate_slot_192 = (*strm).state as *mut inflate_state;
*__slate_slot_191 = (*strm).next_in;
*__slate_slot_190 = (*__slate_slot_191).wrapping_offset(unsafe { (*strm).avail_in }.wrapping_sub(5) as isize);
*__slate_slot_189 = (*strm).next_out;
*__slate_slot_188 = (*__slate_slot_189).wrapping_offset(-((start as u64).wrapping_sub((*strm).avail_out) as isize));
*__slate_slot_187 = (*__slate_slot_189).wrapping_offset(unsafe { (*strm).avail_out }.wrapping_sub(257) as isize);
*__slate_slot_186 = (*(*__slate_slot_192)).wsize;
*__slate_slot_185 = (*(*__slate_slot_192)).whave;
*__slate_slot_184 = (*(*__slate_slot_192)).write;
*__slate_slot_183 = (*(*__slate_slot_192)).window;
*__slate_slot_182 = (*(*__slate_slot_192)).hold;
*__slate_slot_181 = (*(*__slate_slot_192)).bits;
*__slate_slot_180 = (*(*__slate_slot_192)).lencode;
*__slate_slot_179 = (*(*__slate_slot_192)).distcode;
*__slate_slot_178 = (1u32 << (*(*__slate_slot_192)).lenbits).wrapping_sub(1);
*__slate_slot_177 = (1u32 << (*(*__slate_slot_192)).distbits).wrapping_sub(1);
// decode literals and length/distances until end-of-block or not enough
// input data or output space
'__join_0: {
'__join_25: {
'__join_45: {
'__loop_64: loop {
if *__slate_slot_181 < 15 {
let __v171: *const u8 = *__slate_slot_191;
*__slate_slot_191 = __v171.wrapping_offset(1 as isize);
let __v168: u64 = *__slate_slot_182;
*__slate_slot_182 = __v168.wrapping_add((*__v171 as u64) << *__slate_slot_181);
let __v166: u32 = *__slate_slot_181;
*__slate_slot_181 = __v166.wrapping_add(8);
let __v164: *const u8 = *__slate_slot_191;
*__slate_slot_191 = __v164.wrapping_offset(1 as isize);
let __v161: u64 = *__slate_slot_182;
*__slate_slot_182 = __v161.wrapping_add((*__v164 as u64) << *__slate_slot_181);
let __v159: u32 = *__slate_slot_181;
*__slate_slot_181 = __v159.wrapping_add(8);
}
*__slate_slot_176 = *(*__slate_slot_180).wrapping_offset((*__slate_slot_182 & (*__slate_slot_178 as u64)) as isize);
'__join_1: {
'__join_2: {
loop {
*__slate_slot_175 = (*__slate_slot_176).bits as u32;
*__slate_slot_182 >>= *__slate_slot_175;
let __v155: u32 = *__slate_slot_181;
*__slate_slot_181 = __v155.wrapping_sub(*__slate_slot_175);
*__slate_slot_175 = (*__slate_slot_176).op as u32;
if *__slate_slot_175 == 0 {
break '__join_2;
} else {
if *__slate_slot_175 & 16 != 0 {
break;
} else {
if *__slate_slot_175 & 64 == 0 {
// 2nd level length code
*__slate_slot_176 = *(*__slate_slot_180).wrapping_offset((((*__slate_slot_176).val as i32) as u64).wrapping_add(*__slate_slot_182 & ((1u32 << *__slate_slot_175).wrapping_sub(1) as u64)) as isize);
} else {
break '__loop_64;
}
}
}
}
// length base
*__slate_slot_174 = (*__slate_slot_176).val as u32;
*__slate_slot_175 &= 15; // number of extra bits
if *__slate_slot_175 != 0 {
if *__slate_slot_181 < *__slate_slot_175 {
let __v151: *const u8 = *__slate_slot_191;
*__slate_slot_191 = __v151.wrapping_offset(1 as isize);
let __v148: u64 = *__slate_slot_182;
*__slate_slot_182 = __v148.wrapping_add((*__v151 as u64) << *__slate_slot_181);
let __v146: u32 = *__slate_slot_181;
*__slate_slot_181 = __v146.wrapping_add(8);
}
let __v144: u32 = *__slate_slot_174;
*__slate_slot_174 = __v144.wrapping_add((*__slate_slot_182 as u32) & (1u32 << *__slate_slot_175).wrapping_sub(1));
*__slate_slot_182 >>= *__slate_slot_175;
let __v140: u32 = *__slate_slot_181;
*__slate_slot_181 = __v140.wrapping_sub(*__slate_slot_175);
}
if *__slate_slot_181 < 15 {
let __v138: *const u8 = *__slate_slot_191;
*__slate_slot_191 = __v138.wrapping_offset(1 as isize);
let __v135: u64 = *__slate_slot_182;
*__slate_slot_182 = __v135.wrapping_add((*__v138 as u64) << *__slate_slot_181);
let __v133: u32 = *__slate_slot_181;
*__slate_slot_181 = __v133.wrapping_add(8);
let __v131: *const u8 = *__slate_slot_191;
*__slate_slot_191 = __v131.wrapping_offset(1 as isize);
let __v128: u64 = *__slate_slot_182;
*__slate_slot_182 = __v128.wrapping_add((*__v131 as u64) << *__slate_slot_181);
let __v126: u32 = *__slate_slot_181;
*__slate_slot_181 = __v126.wrapping_add(8);
}
*__slate_slot_176 = *(*__slate_slot_179).wrapping_offset((*__slate_slot_182 & (*__slate_slot_177 as u64)) as isize);
loop {
*__slate_slot_175 = (*__slate_slot_176).bits as u32;
*__slate_slot_182 >>= *__slate_slot_175;
let __v122: u32 = *__slate_slot_181;
*__slate_slot_181 = __v122.wrapping_sub(*__slate_slot_175);
*__slate_slot_175 = (*__slate_slot_176).op as u32;
if *__slate_slot_175 & 16 != 0 {
break;
} else {
if *__slate_slot_175 & 64 == 0 {
// 2nd level distance code
*__slate_slot_176 = *(*__slate_slot_179).wrapping_offset((((*__slate_slot_176).val as i32) as u64).wrapping_add(*__slate_slot_182 & ((1u32 << *__slate_slot_175).wrapping_sub(1) as u64)) as isize);
} else {
break '__join_45;
}
}
}
// distance base
*__slate_slot_173 = (*__slate_slot_176).val as u32;
*__slate_slot_175 &= 15; // number of extra bits
if *__slate_slot_181 < *__slate_slot_175 {
let __v118: *const u8 = *__slate_slot_191;
*__slate_slot_191 = __v118.wrapping_offset(1 as isize);
let __v115: u64 = *__slate_slot_182;
*__slate_slot_182 = __v115.wrapping_add((*__v118 as u64) << *__slate_slot_181);
let __v113: u32 = *__slate_slot_181;
*__slate_slot_181 = __v113.wrapping_add(8);
if *__slate_slot_181 < *__slate_slot_175 {
let __v111: *const u8 = *__slate_slot_191;
*__slate_slot_191 = __v111.wrapping_offset(1 as isize);
let __v108: u64 = *__slate_slot_182;
*__slate_slot_182 = __v108.wrapping_add((*__v111 as u64) << *__slate_slot_181);
let __v106: u32 = *__slate_slot_181;
*__slate_slot_181 = __v106.wrapping_add(8);
}
}
let __v104: u32 = *__slate_slot_173;
*__slate_slot_173 = __v104.wrapping_add((*__slate_slot_182 as u32) & (1u32 << *__slate_slot_175).wrapping_sub(1));
*__slate_slot_182 >>= *__slate_slot_175;
let __v100: u32 = *__slate_slot_181;
*__slate_slot_181 = __v100.wrapping_sub(*__slate_slot_175);
*__slate_slot_175 = ((*__slate_slot_189).offset_from(*__slate_slot_188 as *mut u8) as i64) as u32; // max distance in output
if *__slate_slot_173 > *__slate_slot_175 {
// see if copy from window
*__slate_slot_175 = (*__slate_slot_173).wrapping_sub(*__slate_slot_175); // distance back in window
if *__slate_slot_175 > *__slate_slot_185 {
break '__join_25;
} else {
*__slate_slot_172 = *__slate_slot_183;
if *__slate_slot_184 == 0 {
// very common case
let __v39: *mut u8 = *__slate_slot_172;
*__slate_slot_172 = __v39.wrapping_offset((*__slate_slot_186).wrapping_sub(*__slate_slot_175) as isize);
if *__slate_slot_175 < *__slate_slot_174 {
// some from window
let __v37: u32 = *__slate_slot_174;
*__slate_slot_174 = __v37.wrapping_sub(*__slate_slot_175);
loop {
let __v35: *mut u8 = *__slate_slot_172;
*__slate_slot_172 = __v35.wrapping_offset(1 as isize);
let __v33: *mut u8 = *__slate_slot_189;
*__slate_slot_189 = __v33.wrapping_offset(1 as isize);
*__v33 = *__v35;
let __v31: u32 = *__slate_slot_175;
let __v30: u32 = __v31.wrapping_sub(1);
*__slate_slot_175 = __v30;
if __v30 == 0 {
break;
}
}
*__slate_slot_172 = (*__slate_slot_189).wrapping_offset(-(*__slate_slot_173 as isize)); // rest from output
}
} else {
if *__slate_slot_184 < *__slate_slot_175 {
// wrap around window
let __v59: *mut u8 = *__slate_slot_172;
*__slate_slot_172 = __v59.wrapping_offset((*__slate_slot_186).wrapping_add(*__slate_slot_184).wrapping_sub(*__slate_slot_175) as isize);
let __v57: u32 = *__slate_slot_175;
*__slate_slot_175 = __v57.wrapping_sub(*__slate_slot_184);
if *__slate_slot_175 < *__slate_slot_174 {
// some from end of window
let __v55: u32 = *__slate_slot_174;
*__slate_slot_174 = __v55.wrapping_sub(*__slate_slot_175);
loop {
let __v53: *mut u8 = *__slate_slot_172;
*__slate_slot_172 = __v53.wrapping_offset(1 as isize);
let __v51: *mut u8 = *__slate_slot_189;
*__slate_slot_189 = __v51.wrapping_offset(1 as isize);
*__v51 = *__v53;
let __v49: u32 = *__slate_slot_175;
let __v48: u32 = __v49.wrapping_sub(1);
*__slate_slot_175 = __v48;
if __v48 == 0 {
break;
}
}
*__slate_slot_172 = *__slate_slot_183;
if *__slate_slot_184 < *__slate_slot_174 {
// some from start of window
*__slate_slot_175 = *__slate_slot_184;
let __v47: u32 = *__slate_slot_174;
*__slate_slot_174 = __v47.wrapping_sub(*__slate_slot_175);
loop {
let __v45: *mut u8 = *__slate_slot_172;
*__slate_slot_172 = __v45.wrapping_offset(1 as isize);
let __v43: *mut u8 = *__slate_slot_189;
*__slate_slot_189 = __v43.wrapping_offset(1 as isize);
*__v43 = *__v45;
let __v41: u32 = *__slate_slot_175;
let __v40: u32 = __v41.wrapping_sub(1);
*__slate_slot_175 = __v40;
if __v40 == 0 {
break;
}
}
*__slate_slot_172 = (*__slate_slot_189).wrapping_offset(-(*__slate_slot_173 as isize)); // rest from output
}
}
} else {
// contiguous in window
let __v69: *mut u8 = *__slate_slot_172;
*__slate_slot_172 = __v69.wrapping_offset((*__slate_slot_184).wrapping_sub(*__slate_slot_175) as isize);
if *__slate_slot_175 < *__slate_slot_174 {
// some from window
let __v67: u32 = *__slate_slot_174;
*__slate_slot_174 = __v67.wrapping_sub(*__slate_slot_175);
loop {
let __v65: *mut u8 = *__slate_slot_172;
*__slate_slot_172 = __v65.wrapping_offset(1 as isize);
let __v63: *mut u8 = *__slate_slot_189;
*__slate_slot_189 = __v63.wrapping_offset(1 as isize);
*__v63 = *__v65;
let __v61: u32 = *__slate_slot_175;
let __v60: u32 = __v61.wrapping_sub(1);
*__slate_slot_175 = __v60;
if __v60 == 0 {
break;
}
}
*__slate_slot_172 = (*__slate_slot_189).wrapping_offset(-(*__slate_slot_173 as isize)); // rest from output
}
}
}
while *__slate_slot_174 > 2 {
let __v29: *mut u8 = *__slate_slot_172;
*__slate_slot_172 = __v29.wrapping_offset(1 as isize);
let __v27: *mut u8 = *__slate_slot_189;
*__slate_slot_189 = __v27.wrapping_offset(1 as isize);
*__v27 = *__v29;
let __v25: *mut u8 = *__slate_slot_172;
*__slate_slot_172 = __v25.wrapping_offset(1 as isize);
let __v23: *mut u8 = *__slate_slot_189;
*__slate_slot_189 = __v23.wrapping_offset(1 as isize);
*__v23 = *__v25;
let __v21: *mut u8 = *__slate_slot_172;
*__slate_slot_172 = __v21.wrapping_offset(1 as isize);
let __v19: *mut u8 = *__slate_slot_189;
*__slate_slot_189 = __v19.wrapping_offset(1 as isize);
*__v19 = *__v21;
let __v17: u32 = *__slate_slot_174;
*__slate_slot_174 = __v17.wrapping_sub(3);
}
if *__slate_slot_174 != 0 {
let __v15: *mut u8 = *__slate_slot_172;
*__slate_slot_172 = __v15.wrapping_offset(1 as isize);
let __v13: *mut u8 = *__slate_slot_189;
*__slate_slot_189 = __v13.wrapping_offset(1 as isize);
*__v13 = *__v15;
if *__slate_slot_174 > 1 {
let __v11: *mut u8 = *__slate_slot_172;
*__slate_slot_172 = __v11.wrapping_offset(1 as isize);
let __v9: *mut u8 = *__slate_slot_189;
*__slate_slot_189 = __v9.wrapping_offset(1 as isize);
*__v9 = *__v11;
break '__join_1;
} else {
break '__join_1;
}
} else {
break '__join_1;
}
}
} else {
{
let __slate_auto_bytes_2 = __slate_slot_98 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_97 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_172 = (*__slate_slot_189).wrapping_offset(-(*__slate_slot_173 as isize)); // copy direct from output
// minimum length is three
//
// Align out addr
if ((*__slate_slot_189).wrapping_offset(-(1 as isize)) as i64) & 1 == 0 {
let __v96: *mut u8 = *__slate_slot_172;
*__slate_slot_172 = __v96.wrapping_offset(1 as isize);
let __v94: *mut u8 = *__slate_slot_189;
*__slate_slot_189 = __v94.wrapping_offset(1 as isize);
*__v94 = *__v96;
let __v92: u32 = *__slate_slot_174;
*__slate_slot_174 = __v92.wrapping_sub(1);
}
}
*__slate_slot_98 = *__slate_slot_189 as *mut u16;
if *__slate_slot_173 > 2 {
let __slate_auto_bytes = __slate_slot_84 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_84 = *__slate_slot_172 as *mut u16;
*__slate_slot_97 = (*__slate_slot_174 >> 1i32) as u64;
loop {
let __v79: *mut u16 = *__slate_slot_84;
*__slate_slot_84 = __v79.wrapping_offset(1 as isize);
let __v77: *mut u16 = *__slate_slot_98;
*__slate_slot_98 = __v77.wrapping_offset(1 as isize);
*__v77 = *__v79;
let __v75: u64 = *__slate_slot_97;
let __v74: u64 = __v75.wrapping_sub(1);
*__slate_slot_97 = __v74;
if __v74 == 0 {
break;
}
}
*__slate_slot_189 = *__slate_slot_98 as *mut u8;
*__slate_slot_172 = *__slate_slot_84 as *mut u8;
} else {
// dist == 1 or dist == 2
let __slate_auto_bytes_1 = __slate_slot_90 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 2usize);
*__slate_slot_90 = *(*__slate_slot_98).wrapping_offset(-(1 as isize));
if *__slate_slot_173 == 1 {
let __slate_auto_bytes = __slate_slot_89 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
// copy one char pattern to both bytes
(*__slate_slot_89).us = *__slate_slot_90;
(*__slate_slot_89).b[0] = (*__slate_slot_89).b[1];
*__slate_slot_90 = (*__slate_slot_89).us;
}
*__slate_slot_97 = (*__slate_slot_174 >> 1i32) as u64;
loop {
let __v88: *mut u16 = *__slate_slot_98;
*__slate_slot_98 = __v88.wrapping_offset(1 as isize);
*__v88 = *__slate_slot_90;
let __v86: u64 = *__slate_slot_97;
let __v85: u64 = __v86.wrapping_sub(1);
*__slate_slot_97 = __v85;
if __v85 == 0 {
break;
}
}
*__slate_slot_189 = *__slate_slot_98 as *mut u8;
}
if *__slate_slot_174 & 1 != 0 {
let __v73: *mut u8 = *__slate_slot_172;
*__slate_slot_172 = __v73.wrapping_offset(1 as isize);
let __v71: *mut u8 = *__slate_slot_189;
*__slate_slot_189 = __v71.wrapping_offset(1 as isize);
*__v71 = *__v73;
break '__join_1;
} else {
break '__join_1;
}
}
}
// literal
let __v7: *mut u8 = *__slate_slot_189;
*__slate_slot_189 = __v7.wrapping_offset(1 as isize);
*__v7 = (*__slate_slot_176).val as u8;
}
if !(*__slate_slot_191 < *__slate_slot_190 && *__slate_slot_189 < *__slate_slot_187) {
break '__join_0;
}
}
if *__slate_slot_175 & 32 != 0 {
// end-of-block
(*(*__slate_slot_192)).mode = 11;
break '__join_0;
} else {
(*strm).msg = c"invalid literal/length code".as_ptr() as *mut u8;
(*(*__slate_slot_192)).mode = 27;
break '__join_0;
}
}
(*strm).msg = c"invalid distance code".as_ptr() as *mut u8;
(*(*__slate_slot_192)).mode = 27;
break '__join_0;
}
(*strm).msg = c"invalid distance too far back".as_ptr() as *mut u8;
(*(*__slate_slot_192)).mode = 27;
}
// return unused bytes (on entry, bits < 8, so in won't go too far back)
*__slate_slot_174 = *__slate_slot_181 >> 3i32;
let __v5: *const u8 = *__slate_slot_191;
*__slate_slot_191 = __v5.wrapping_offset(-(*__slate_slot_174 as isize));
let __v3: u32 = *__slate_slot_181;
*__slate_slot_181 = __v3.wrapping_sub(*__slate_slot_174 << 3i32);
*__slate_slot_182 &= (1u32 << *__slate_slot_181).wrapping_sub(1) as u64;
// update state and return
(*strm).next_in = *__slate_slot_191;
(*strm).next_out = *__slate_slot_189;
(*strm).avail_in = ((if *__slate_slot_191 < *__slate_slot_190 { 5i64.wrapping_add((*__slate_slot_190).offset_from(*__slate_slot_191 as *const u8) as i64) } else { 5i64.wrapping_sub((*__slate_slot_191).offset_from(*__slate_slot_190 as *const u8) as i64) }) as u32) as u64;
(*strm).avail_out = ((if *__slate_slot_189 < *__slate_slot_187 { 257i64.wrapping_add((*__slate_slot_187).offset_from(*__slate_slot_189 as *mut u8) as i64) } else { 257i64.wrapping_sub((*__slate_slot_189).offset_from(*__slate_slot_187 as *mut u8) as i64) }) as u32) as u64;
(*(*__slate_slot_192)).hold = *__slate_slot_182;
(*(*__slate_slot_192)).bits = *__slate_slot_181;
return;
}
}

// inflate_fast() speedups that turned out slower (on a PowerPC G3 750CXe):
// - Using bit fields for code structure
// - Different op definition to avoid & for extra bits (do & for table bits)
// - Three separate decoding do-loops for direct, window, and write == 0
// - Special case for distance > 1 copies to do overlapped load and store copy
// - Explicit branch predictions (based on measured branch probabilities)
// - Deferring match copy and interspersed it with decoding subsequent codes
// - Swapping literal/length else
// - Swapping window/direct else
// - Larger unrolled copy loops (three is about right)
// - Moving len -= 3 statement into middle of loop
