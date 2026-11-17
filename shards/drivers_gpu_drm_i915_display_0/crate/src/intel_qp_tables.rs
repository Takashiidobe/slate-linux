
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: MIT
//!
//! Copyright © 2021 Intel Corporation
//! VESA Display Stream Compression DSC 1.2 constants
//! Format: __UNIQUE_ID_<name>_<__COUNTER__>
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
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
// Force the compiler to emit 'sym' as a symbol, so that we can reference
// it from inline assembler. Necessary in case 'sym' could be inlined
// otherwise, or eliminated entirely due to lack of references that are
// visible to the compiler.
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
macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

macro_rules! unlikely {
($x:expr) => {
(($x) != 0) as i64
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
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
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

/// Optional: only supported since gcc >= 15, clang >= 19
///
///   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
/// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
#[repr(C)]
#[derive(Clone, Copy)]
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


#[unsafe(no_mangle)]
extern "C" fn intel_lookup_range_min_qp(bpc: i32, buf_i: i32, bpp_i: i32, is_420: bool) -> u8 {
loop {
if bpc == 8 {
if is_420 {
return unsafe { rc_range_minqp420_8bpc.0[buf_i as usize][bpp_i as usize] };
} else {
return unsafe { rc_range_minqp444_8bpc.0[buf_i as usize][bpp_i as usize] };
}
}
if true {
break;
}
}
loop {
if bpc == 10 {
if is_420 {
return unsafe { rc_range_minqp420_10bpc.0[buf_i as usize][bpp_i as usize] };
} else {
return unsafe { rc_range_minqp444_10bpc.0[buf_i as usize][bpp_i as usize] };
}
}
if true {
break;
}
}
loop {
if bpc == 12 {
if is_420 {
return unsafe { rc_range_minqp420_12bpc.0[buf_i as usize][bpp_i as usize] };
} else {
return unsafe { rc_range_minqp444_12bpc.0[buf_i as usize][bpp_i as usize] };
}
}
if true {
break;
}
}
let mut __v3: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"bpc".as_ptr() as *mut u8, bpc as i64) };
let mut __v4: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v4 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v5: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_lookup_range_min_qp_6, sym __slate_asm_string_intel_lookup_range_min_qp_7, const 463 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v5 = bug;
unsafe { __v4.unwrap()(__v5, c"bpc".as_ptr() as *mut u8, bpc as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
0
}

// from BPP 6 to 24 in steps of 0.5
// from BPP 6 to 30 in steps of 0.5
// from BPP 6 to 36 in steps of 0.5
// For YCbCr420 the bits_per_pixel sent in PPS params
// is double the target bpp. The below values represent
// the target bpp.
//
// from BPP 4 to 12 in steps of 0.5
// from BPP 4 to 15 in steps of 0.5
// from BPP 4 to 18 in steps of 0.5
/// These qp tables are as per the C model
/// and it has the rows pointing to bpps which increment
/// in steps of 0.5
static rc_range_minqp444_8bpc: __SlateAlign16<[[u8; 37]; 15]> = __SlateAlign16([[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [3u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8], [5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8], [6u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8], [6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8], [6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 0u8], [6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 0u8], [9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8], [14u8, 14u8, 13u8, 13u8, 12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8]]);

static rc_range_maxqp444_8bpc: __SlateAlign16<[[u8; 37]; 15]> = __SlateAlign16([[4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8], [8u8, 7u8, 7u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8], [8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8], [9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 4u8, 4u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8], [9u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8], [9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8], [10u8, 10u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8], [11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8], [12u8, 11u8, 11u8, 10u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8], [12u8, 12u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8], [12u8, 12u8, 12u8, 11u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8], [12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8], [13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8], [15u8, 15u8, 14u8, 14u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 10u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8]]);

static rc_range_minqp444_10bpc: __SlateAlign16<[[u8; 49]; 15]> = __SlateAlign16([[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [7u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [9u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8], [9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8], [9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8], [10u8, 9u8, 9u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 6u8, 6u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 0u8], [10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8], [10u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8], [10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8], [12u8, 12u8, 12u8, 12u8, 12u8, 12u8, 12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8], [18u8, 18u8, 17u8, 17u8, 16u8, 16u8, 16u8, 16u8, 15u8, 15u8, 14u8, 14u8, 14u8, 14u8, 13u8, 13u8, 13u8, 12u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8]]);

static rc_range_maxqp444_10bpc: __SlateAlign16<[[u8; 49]; 15]> = __SlateAlign16([[8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 6u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [10u8, 10u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [12u8, 11u8, 11u8, 10u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [12u8, 12u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 9u8, 8u8, 7u8, 7u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 8u8, 8u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8], [13u8, 12u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 10u8, 9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8], [13u8, 13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8], [14u8, 14u8, 13u8, 13u8, 12u8, 12u8, 12u8, 12u8, 12u8, 12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 10u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8], [15u8, 15u8, 14u8, 14u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 12u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8], [16u8, 15u8, 15u8, 14u8, 14u8, 14u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8], [16u8, 16u8, 15u8, 15u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 13u8, 13u8, 13u8, 12u8, 12u8, 11u8, 11u8, 10u8, 10u8, 10u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8], [16u8, 16u8, 16u8, 15u8, 15u8, 15u8, 14u8, 14u8, 14u8, 14u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 12u8, 11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8], [16u8, 16u8, 16u8, 16u8, 15u8, 15u8, 15u8, 15u8, 15u8, 14u8, 14u8, 13u8, 13u8, 13u8, 12u8, 12u8, 12u8, 11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8], [17u8, 17u8, 17u8, 17u8, 16u8, 16u8, 15u8, 15u8, 15u8, 15u8, 14u8, 14u8, 14u8, 14u8, 13u8, 13u8, 12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8], [19u8, 19u8, 18u8, 18u8, 17u8, 17u8, 17u8, 17u8, 16u8, 16u8, 15u8, 15u8, 15u8, 15u8, 14u8, 14u8, 14u8, 13u8, 13u8, 13u8, 12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 4u8]]);

static rc_range_minqp444_12bpc: __SlateAlign16<[[u8; 61]; 15]> = __SlateAlign16([[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [11u8, 10u8, 10u8, 9u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 6u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 7u8, 6u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 8u8, 7u8, 7u8, 7u8, 7u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [13u8, 12u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [13u8, 13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8], [13u8, 13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8], [13u8, 13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8], [14u8, 13u8, 13u8, 12u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 0u8], [14u8, 14u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8], [14u8, 14u8, 14u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8], [14u8, 14u8, 14u8, 14u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8], [17u8, 17u8, 17u8, 17u8, 16u8, 16u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 14u8, 14u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8], [22u8, 22u8, 21u8, 21u8, 20u8, 20u8, 20u8, 20u8, 19u8, 19u8, 18u8, 18u8, 18u8, 18u8, 17u8, 17u8, 17u8, 16u8, 16u8, 16u8, 15u8, 15u8, 15u8, 15u8, 14u8, 14u8, 13u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 3u8]]);

static rc_range_maxqp444_12bpc: __SlateAlign16<[[u8; 61]; 15]> = __SlateAlign16([[12u8, 12u8, 12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 11u8, 10u8, 9u8, 9u8, 6u8, 6u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [14u8, 14u8, 13u8, 13u8, 12u8, 12u8, 12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 9u8, 9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [16u8, 15u8, 15u8, 14u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 12u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [16u8, 16u8, 15u8, 15u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 13u8, 13u8, 13u8, 12u8, 11u8, 11u8, 10u8, 10u8, 10u8, 8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [17u8, 16u8, 16u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 14u8, 14u8, 13u8, 12u8, 12u8, 11u8, 10u8, 10u8, 10u8, 10u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8], [17u8, 16u8, 16u8, 16u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 14u8, 14u8, 14u8, 13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 7u8, 6u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8], [17u8, 17u8, 16u8, 16u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 14u8, 14u8, 13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 10u8, 9u8, 9u8, 9u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 0u8], [18u8, 18u8, 17u8, 17u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 15u8, 15u8, 14u8, 13u8, 13u8, 12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 10u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8], [19u8, 19u8, 18u8, 18u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 16u8, 16u8, 16u8, 15u8, 15u8, 14u8, 14u8, 13u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 10u8, 9u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8], [20u8, 19u8, 19u8, 18u8, 18u8, 18u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 16u8, 16u8, 15u8, 14u8, 14u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8], [20u8, 20u8, 19u8, 19u8, 18u8, 18u8, 18u8, 18u8, 18u8, 18u8, 17u8, 17u8, 17u8, 16u8, 16u8, 15u8, 15u8, 14u8, 14u8, 14u8, 13u8, 13u8, 12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8], [20u8, 20u8, 20u8, 19u8, 19u8, 19u8, 18u8, 18u8, 18u8, 18u8, 17u8, 17u8, 17u8, 17u8, 16u8, 16u8, 16u8, 15u8, 15u8, 15u8, 14u8, 14u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 6u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8], [20u8, 20u8, 20u8, 20u8, 19u8, 19u8, 19u8, 19u8, 19u8, 18u8, 18u8, 17u8, 17u8, 17u8, 16u8, 16u8, 16u8, 15u8, 15u8, 15u8, 14u8, 14u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 6u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8], [21u8, 21u8, 21u8, 21u8, 20u8, 20u8, 19u8, 19u8, 19u8, 19u8, 18u8, 18u8, 18u8, 18u8, 17u8, 17u8, 16u8, 16u8, 16u8, 16u8, 15u8, 15u8, 14u8, 14u8, 14u8, 14u8, 13u8, 13u8, 12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8], [23u8, 23u8, 22u8, 22u8, 21u8, 21u8, 21u8, 21u8, 20u8, 20u8, 19u8, 19u8, 19u8, 19u8, 18u8, 18u8, 18u8, 17u8, 17u8, 17u8, 16u8, 16u8, 16u8, 16u8, 15u8, 15u8, 14u8, 14u8, 14u8, 14u8, 14u8, 13u8, 13u8, 12u8, 12u8, 12u8, 12u8, 12u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 4u8]]);

static rc_range_minqp420_8bpc: __SlateAlign16<[[u8; 17]; 15]> = __SlateAlign16([[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8], [3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8], [3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8], [3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 0u8], [3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 0u8], [3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8], [5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 1u8, 1u8], [5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 2u8, 2u8, 1u8], [5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 2u8, 1u8], [9u8, 8u8, 8u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 5u8, 5u8, 4u8, 3u8, 3u8, 3u8, 2u8], [13u8, 12u8, 12u8, 11u8, 10u8, 10u8, 9u8, 8u8, 8u8, 7u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 3u8]]);

static rc_range_maxqp420_8bpc: __SlateAlign16<[[u8; 17]; 15]> = __SlateAlign16([[4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8], [5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 3u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8], [6u8, 6u8, 6u8, 6u8, 6u8, 5u8, 4u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8], [7u8, 7u8, 7u8, 7u8, 7u8, 5u8, 4u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 0u8], [7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 5u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 0u8], [7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 5u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8], [8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 6u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8], [9u8, 9u8, 9u8, 8u8, 8u8, 7u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 1u8], [10u8, 10u8, 9u8, 9u8, 9u8, 8u8, 7u8, 6u8, 5u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8], [10u8, 10u8, 10u8, 9u8, 9u8, 8u8, 8u8, 7u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 3u8, 2u8, 2u8], [11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 8u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 4u8, 3u8, 3u8, 2u8], [11u8, 11u8, 11u8, 10u8, 9u8, 9u8, 9u8, 8u8, 7u8, 7u8, 6u8, 5u8, 5u8, 4u8, 4u8, 3u8, 2u8], [13u8, 12u8, 12u8, 11u8, 10u8, 10u8, 9u8, 8u8, 8u8, 7u8, 6u8, 6u8, 5u8, 4u8, 4u8, 4u8, 3u8], [14u8, 13u8, 13u8, 12u8, 11u8, 11u8, 10u8, 9u8, 9u8, 8u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 4u8]]);

static rc_range_minqp420_10bpc: __SlateAlign16<[[u8; 23]; 15]> = __SlateAlign16([[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [4u8, 4u8, 4u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8], [7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 5u8, 5u8, 5u8, 5u8, 5u8, 4u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8], [7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 0u8], [7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 6u8, 5u8, 4u8, 4u8, 4u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 0u8], [7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8], [7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8], [9u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 1u8], [9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 7u8, 6u8, 6u8, 5u8, 4u8, 4u8, 3u8, 3u8, 2u8, 1u8], [9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 6u8, 5u8, 4u8, 4u8, 3u8, 3u8, 2u8, 1u8], [13u8, 12u8, 12u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 9u8, 9u8, 8u8, 7u8, 7u8, 6u8, 5u8, 5u8, 4u8, 3u8, 3u8, 2u8, 2u8], [17u8, 16u8, 16u8, 15u8, 14u8, 14u8, 13u8, 12u8, 12u8, 11u8, 10u8, 10u8, 10u8, 9u8, 8u8, 8u8, 7u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8]]);

static rc_range_maxqp420_10bpc: __SlateAlign16<[[u8; 23]; 15]> = __SlateAlign16([[8u8, 8u8, 7u8, 6u8, 4u8, 4u8, 3u8, 3u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [8u8, 8u8, 8u8, 7u8, 6u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [9u8, 9u8, 9u8, 8u8, 8u8, 7u8, 6u8, 5u8, 4u8, 3u8, 3u8, 3u8, 3u8, 3u8, 2u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8], [10u8, 10u8, 10u8, 9u8, 9u8, 8u8, 7u8, 6u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8], [11u8, 11u8, 11u8, 10u8, 10u8, 8u8, 7u8, 6u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 0u8], [11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 8u8, 7u8, 6u8, 6u8, 6u8, 5u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 1u8], [11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 9u8, 8u8, 7u8, 7u8, 7u8, 6u8, 5u8, 5u8, 4u8, 3u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8], [12u8, 12u8, 12u8, 12u8, 12u8, 11u8, 10u8, 9u8, 8u8, 8u8, 8u8, 7u8, 6u8, 5u8, 5u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8], [13u8, 13u8, 13u8, 12u8, 12u8, 11u8, 10u8, 10u8, 9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 6u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8], [14u8, 14u8, 13u8, 13u8, 13u8, 12u8, 11u8, 10u8, 9u8, 9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 6u8, 5u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8], [14u8, 14u8, 14u8, 13u8, 13u8, 12u8, 12u8, 11u8, 10u8, 10u8, 9u8, 9u8, 8u8, 8u8, 7u8, 6u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 2u8], [15u8, 15u8, 14u8, 14u8, 13u8, 13u8, 12u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 8u8, 7u8, 7u8, 6u8, 5u8, 5u8, 4u8, 4u8, 3u8, 2u8], [15u8, 15u8, 15u8, 14u8, 13u8, 13u8, 13u8, 12u8, 11u8, 11u8, 10u8, 9u8, 9u8, 8u8, 8u8, 7u8, 6u8, 5u8, 5u8, 4u8, 4u8, 3u8, 2u8], [17u8, 16u8, 16u8, 15u8, 14u8, 14u8, 13u8, 12u8, 12u8, 11u8, 10u8, 10u8, 9u8, 8u8, 8u8, 7u8, 6u8, 6u8, 5u8, 4u8, 4u8, 3u8, 3u8], [18u8, 17u8, 17u8, 16u8, 15u8, 15u8, 14u8, 13u8, 13u8, 12u8, 11u8, 11u8, 11u8, 10u8, 9u8, 9u8, 8u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8]]);

static rc_range_minqp420_12bpc: __SlateAlign16<[[u8; 29]; 15]> = __SlateAlign16([[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [9u8, 8u8, 8u8, 7u8, 7u8, 6u8, 5u8, 5u8, 4u8, 4u8, 4u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [10u8, 9u8, 9u8, 8u8, 8u8, 8u8, 7u8, 7u8, 6u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [11u8, 10u8, 10u8, 10u8, 10u8, 9u8, 9u8, 8u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 5u8, 4u8, 3u8, 3u8, 3u8, 2u8, 2u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 9u8, 8u8, 7u8, 6u8, 5u8, 5u8, 4u8, 4u8, 3u8, 3u8, 3u8, 2u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8], [11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 9u8, 8u8, 8u8, 7u8, 6u8, 5u8, 5u8, 5u8, 5u8, 4u8, 3u8, 3u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8], [11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 10u8, 10u8, 9u8, 8u8, 8u8, 8u8, 7u8, 6u8, 6u8, 5u8, 4u8, 4u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8], [11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 6u8, 5u8, 5u8, 4u8, 4u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8], [11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 6u8, 5u8, 4u8, 4u8, 3u8, 2u8, 2u8, 1u8, 1u8, 1u8], [13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 12u8, 11u8, 11u8, 10u8, 10u8, 9u8, 9u8, 8u8, 8u8, 7u8, 7u8, 6u8, 5u8, 4u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8], [13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 12u8, 12u8, 11u8, 10u8, 10u8, 9u8, 8u8, 8u8, 7u8, 7u8, 6u8, 5u8, 4u8, 3u8, 3u8, 2u8, 2u8, 1u8], [13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 13u8, 12u8, 12u8, 11u8, 11u8, 10u8, 9u8, 8u8, 8u8, 7u8, 7u8, 6u8, 5u8, 4u8, 4u8, 3u8, 2u8, 2u8, 1u8], [15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 14u8, 13u8, 13u8, 12u8, 11u8, 11u8, 10u8, 9u8, 9u8, 8u8, 8u8, 7u8, 6u8, 6u8, 5u8, 4u8, 4u8, 3u8, 3u8, 2u8], [21u8, 20u8, 20u8, 19u8, 18u8, 18u8, 17u8, 16u8, 16u8, 15u8, 14u8, 14u8, 14u8, 13u8, 12u8, 12u8, 11u8, 10u8, 10u8, 10u8, 9u8, 8u8, 8u8, 7u8, 6u8, 6u8, 5u8, 5u8, 4u8]]);

static rc_range_maxqp420_12bpc: __SlateAlign16<[[u8; 29]; 15]> = __SlateAlign16([[11u8, 10u8, 9u8, 8u8, 6u8, 6u8, 5u8, 5u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [12u8, 11u8, 11u8, 10u8, 9u8, 8u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 4u8, 3u8, 3u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [13u8, 12u8, 12u8, 11u8, 11u8, 10u8, 9u8, 8u8, 7u8, 6u8, 6u8, 6u8, 5u8, 5u8, 4u8, 3u8, 3u8, 2u8, 1u8, 1u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8], [14u8, 13u8, 13u8, 12u8, 12u8, 11u8, 10u8, 9u8, 8u8, 7u8, 7u8, 6u8, 6u8, 5u8, 5u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8], [15u8, 14u8, 14u8, 13u8, 13u8, 11u8, 10u8, 9u8, 8u8, 7u8, 7u8, 7u8, 7u8, 6u8, 6u8, 5u8, 4u8, 4u8, 4u8, 3u8, 3u8, 2u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8], [15u8, 15u8, 15u8, 14u8, 14u8, 13u8, 12u8, 11u8, 10u8, 10u8, 10u8, 9u8, 8u8, 7u8, 6u8, 6u8, 5u8, 5u8, 4u8, 4u8, 4u8, 3u8, 2u8, 2u8, 1u8, 1u8, 0u8, 0u8, 0u8], [15u8, 15u8, 15u8, 15u8, 15u8, 14u8, 13u8, 12u8, 11u8, 11u8, 11u8, 10u8, 9u8, 8u8, 7u8, 6u8, 6u8, 6u8, 6u8, 5u8, 4u8, 4u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8, 0u8], [16u8, 16u8, 16u8, 16u8, 16u8, 15u8, 14u8, 13u8, 12u8, 12u8, 12u8, 11u8, 10u8, 9u8, 9u8, 8u8, 7u8, 7u8, 6u8, 5u8, 5u8, 4u8, 3u8, 3u8, 2u8, 2u8, 2u8, 1u8, 1u8], [17u8, 17u8, 17u8, 16u8, 16u8, 15u8, 14u8, 14u8, 13u8, 13u8, 12u8, 12u8, 11u8, 11u8, 10u8, 9u8, 8u8, 8u8, 7u8, 6u8, 6u8, 5u8, 5u8, 3u8, 3u8, 2u8, 2u8, 1u8, 1u8], [18u8, 18u8, 17u8, 17u8, 17u8, 16u8, 15u8, 14u8, 13u8, 13u8, 13u8, 12u8, 12u8, 11u8, 11u8, 10u8, 9u8, 8u8, 8u8, 7u8, 6u8, 5u8, 5u8, 4u8, 3u8, 3u8, 2u8, 2u8, 1u8], [18u8, 18u8, 18u8, 17u8, 17u8, 16u8, 16u8, 15u8, 14u8, 14u8, 13u8, 13u8, 12u8, 12u8, 11u8, 10u8, 9u8, 9u8, 8u8, 8u8, 7u8, 6u8, 5u8, 4u8, 4u8, 3u8, 3u8, 2u8, 2u8], [19u8, 19u8, 18u8, 18u8, 17u8, 17u8, 16u8, 15u8, 15u8, 14u8, 14u8, 13u8, 13u8, 12u8, 11u8, 11u8, 10u8, 9u8, 9u8, 8u8, 8u8, 7u8, 6u8, 5u8, 4u8, 4u8, 3u8, 3u8, 2u8], [19u8, 19u8, 19u8, 18u8, 17u8, 17u8, 17u8, 16u8, 15u8, 15u8, 14u8, 13u8, 13u8, 12u8, 12u8, 11u8, 10u8, 9u8, 9u8, 8u8, 8u8, 7u8, 6u8, 5u8, 5u8, 4u8, 3u8, 3u8, 2u8], [21u8, 20u8, 20u8, 19u8, 18u8, 18u8, 17u8, 16u8, 16u8, 15u8, 14u8, 14u8, 13u8, 12u8, 12u8, 11u8, 10u8, 10u8, 9u8, 9u8, 8u8, 7u8, 7u8, 6u8, 5u8, 5u8, 4u8, 4u8, 3u8], [22u8, 21u8, 21u8, 20u8, 19u8, 19u8, 18u8, 17u8, 17u8, 16u8, 15u8, 15u8, 15u8, 14u8, 13u8, 13u8, 12u8, 11u8, 11u8, 11u8, 10u8, 9u8, 9u8, 8u8, 7u8, 7u8, 6u8, 6u8, 5u8]]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_0: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn intel_lookup_range_max_qp(bpc: i32, buf_i: i32, bpp_i: i32, is_420: bool) -> u8 {
loop {
if bpc == 8 {
if is_420 {
return unsafe { rc_range_maxqp420_8bpc.0[buf_i as usize][bpp_i as usize] };
} else {
return unsafe { rc_range_maxqp444_8bpc.0[buf_i as usize][bpp_i as usize] };
}
}
if true {
break;
}
}
loop {
if bpc == 10 {
if is_420 {
return unsafe { rc_range_maxqp420_10bpc.0[buf_i as usize][bpp_i as usize] };
} else {
return unsafe { rc_range_maxqp444_10bpc.0[buf_i as usize][bpp_i as usize] };
}
}
if true {
break;
}
}
loop {
if bpc == 12 {
if is_420 {
return unsafe { rc_range_maxqp420_12bpc.0[buf_i as usize][bpp_i as usize] };
} else {
return unsafe { rc_range_maxqp444_12bpc.0[buf_i as usize][bpp_i as usize] };
}
}
if true {
break;
}
}
let mut __v3: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"bpc".as_ptr() as *mut u8, bpc as i64) };
let mut __v4: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v4 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v5: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_lookup_range_max_qp_6, sym __slate_asm_string_intel_lookup_range_max_qp_7, const 473 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v5 = bug;
unsafe { __v4.unwrap()(__v5, c"bpc".as_ptr() as *mut u8, bpc as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_1: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

static __slate_asm_string_intel_lookup_range_max_qp_6: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_intel_lookup_range_max_qp_7: [u8; 76] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_qp_tables.c\0";

static __slate_asm_string_intel_lookup_range_min_qp_6: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_intel_lookup_range_min_qp_7: [u8; 76] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_qp_tables.c\0";
