
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0+ OR BSD-3-Clause
//!
//! Copyright (c) Meta Platforms, Inc. and affiliates.
//! All rights reserved.
//!
//! This source code is licensed under both the BSD-style license (found in the
//! LICENSE file in the root directory of this source tree) and the GPLv2 (found
//! in the COPYING file in the root directory of this source tree).
//! You may select, at your option, one of the above-listed licenses.
//! UNUSED_ATTR tells the compiler it is okay if the function is unused.
//! valid since v0.7.0
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! static assert is triggered at compile time, leaving no runtime artefact.
//! static assert only works with compile-time constants.
//! Also, this variant can only be used inside a function.
//! cpuid(1): Processor Info and Feature Bits.
//! Compiler-specific
//! for inlining
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
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-packed-type-attribute
// clang: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-packed-variable-attribute
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
// FORCE_INLINE_TEMPLATE is used to define C "templates", which take constant
// parameters. They must be inlined for the compiler to eliminate the constant
// branches.
// Define ZSTD_FALLTHROUGH macro for annotating switch case with the 'fallthrough' attribute.
// - C23: https://en.cppreference.com/w/c/language/attributes/fallthrough
// - CPP17: https://en.cppreference.com/w/cpp/language/attributes/fallthrough
// - Else: __attribute__((__fallthrough__))
// Zstd relies on pointer overflow in its decompressor.
// We add this attribute to functions that rely on pointer overflow.
//
// older versions of clang [3.7, 5.0) will warn that pointer-overflow is ignored.
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// __put_unaligned_t - write an unaligned value to memory.
// @type:	the type of the value to store.
// @val:	the value to store.
// @ptr:	the pointer to store to.
//
// Use memcpy to affect an unaligned type sized store avoiding undefined
// behavior from approaches like type punning that require -fno-strict-aliasing
// in order to be correct. The void* cast silences ubsan warnings.
// disable assert (default)
// disabled
// disabled
// cpuid(7): Extended Features.
// number of repcodes
// magic number size
// C standard doesn't allow `static const` variable to be init using another `static const` variable
// for static allocation
// for static allocation
// for static allocation
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
macro_rules! ERROR__8 {
() => {
ZSTD_ERROR!()
};
}

macro_rules! ERROR__4 {
() => {
ZSTD_ERROR__6!()
};
}

macro_rules! ERROR__7 {
() => {
ZSTD_ERROR__8!()
};
}

macro_rules! ZSTD_ERROR__6 {
() => {
30i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR__8 {
() => {
64i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_ERROR {
() => {
120i32.wrapping_neg() as u64
};
}

macro_rules! ZSTD_free {
($p:expr) => {
$p;
};
}

macro_rules! ZSTD_malloc {
($s:expr) => {
{
let mut __v: *mut ();
$s;
__v = core::ptr::null_mut::<()>();
__v
}
};
}

macro_rules! ZSTD_memcpy__2 {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memcpy(($d) as *mut (), $s, $n) };
};
}

macro_rules! ZSTD_memcpy__5 {
($d:expr, $s:expr, $n:expr) => {
unsafe { __builtin_memcpy($d, $s, $n) };
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
core::ptr::null_mut::<u32>();
let mut __v3: $type = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), (($ptr) as *mut ()) as *const (), core::mem::size_of::<$type>() as u64) };
__v = __v3;
__v
}
};
}

unsafe extern "C" {
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn ZSTD_loadDEntropy(arg0: *mut __SlateRecord_zstd_decompress_internal_h_88, arg1: *const (), arg2: u64) -> u64;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ZSTD_outBuffer_s {
    dst: *mut (),
    size: u64,
    pos: u64,
}


// can be useful for static allocation
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_lib_h_1489 {
    frameContentSize: u64,
    windowSize: u64,
    blockSizeMax: u32,
    frameType: u32,
    headerSize: u32,
    dictID: u32,
    checksumFlag: u32,
    _reserved1: u32,
    _reserved2: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_lib_h_1837 {
    customAlloc: Option<unsafe extern "C" fn(*mut (), u64) -> *mut ()>,
    customFree: Option<unsafe extern "C" fn(*mut (), *mut ())>,
    opaque: *mut (),
}


fn ZSTD_customMalloc(size: u64, mut customMem: __SlateRecord_zstd_lib_h_1837) -> *mut () {
if customMem.customAlloc != None {
return unsafe { customMem.customAlloc.unwrap()(customMem.opaque, size) };
}
ZSTD_malloc!(size)
}

fn ZSTD_customFree(ptr: *mut (), mut customMem: __SlateRecord_zstd_lib_h_1837) {
if !ptr.is_null() {
if customMem.customFree != None {
unsafe { customMem.customFree.unwrap()(customMem.opaque, ptr) };
} else {
ZSTD_free!(ptr);
}
}
}

fn get_unaligned_le32(p: *const ()) -> u32 {
__get_unaligned_t!(u32, p)
}

fn MEM_readLE32(memPtr: *const ()) -> u32 {
get_unaligned_le32(memPtr)
}

fn ERR_isError(code: u64) -> u32 {
(code > ERROR__8!()) as u32
}

unsafe extern "C" fn _force_has_format_string(format: *const u8, __va_args: ...) {
}

// Ignore: this is an internal helper.
//
// We want to force this function invocation to be syntactically correct, but
// we don't want to force runtime evaluation of its arguments.
// Return the specified error if the condition evaluates to true.
//
// In debug modes, prints additional information.
// In order to do that (particularly, printing the conditional that failed),
// this can't just wrap RETURN_ERROR().
// If the provided expression evaluates to an error code, returns that error code.
//
// In debug modes, prints additional information.
#[repr(C)]
#[derive(Clone, Copy)]
struct xxh64_state {
    total_len: u64,
    v1: u64,
    v2: u64,
    v3: u64,
    v4: u64,
    mem64: [u64; 4],
    memsize: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_decompress_internal_h_73 {
    nextState: u16,
    nbAdditionalBits: u8,
    nbBits: u8,
    baseValue: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_decompress_internal_h_88 {
    LLTable: [__SlateRecord_zstd_decompress_internal_h_73; 513],
    OFTable: [__SlateRecord_zstd_decompress_internal_h_73; 257],
    MLTable: [__SlateRecord_zstd_decompress_internal_h_73; 513],
    hufTable: [u32; 4097],
    rep: [u32; 3],
    workspace: [u32; 157],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_zstd_decompress_internal_h_109 {
    ddictPtrTable: *mut *const ZSTD_DDict_s,
    ddictPtrTableSize: u64,
    ddictPtrCount: u64,
}


// extra buffer, compensates when dst is not large enough to store litBuffer
#[repr(C)]
#[derive(Clone, Copy)]
struct ZSTD_DCtx_s {
    LLTptr: *const __SlateRecord_zstd_decompress_internal_h_73,
    MLTptr: *const __SlateRecord_zstd_decompress_internal_h_73,
    OFTptr: *const __SlateRecord_zstd_decompress_internal_h_73,
    HUFptr: *const u32,
    entropy: __SlateRecord_zstd_decompress_internal_h_88,
    workspace: [u32; 640],
    previousDstEnd: *const (),
    prefixStart: *const (),
    virtualStart: *const (),
    dictEnd: *const (),
    expected: u64,
    fParams: __SlateRecord_zstd_lib_h_1489,
    processedCSize: u64,
    decodedSize: u64,
    bType: u32,
    stage: u32,
    litEntropy: u32,
    fseEntropy: u32,
    xxhState: xxh64_state,
    headerSize: u64,
    format: u32,
    forceIgnoreChecksum: u32,
    validateChecksum: u32,
    litPtr: *const u8,
    customMem: __SlateRecord_zstd_lib_h_1837,
    litSize: u64,
    rleSize: u64,
    staticSize: u64,
    isFrameDecompression: i32,
    bmi2: i32,
    ddictLocal: *mut ZSTD_DDict_s,
    ddict: *const ZSTD_DDict_s,
    dictID: u32,
    ddictIsCold: i32,
    dictUses: i32,
    ddictSet: *mut __SlateRecord_zstd_decompress_internal_h_109,
    refMultipleDDicts: u32,
    disableHufAsm: i32,
    maxBlockSizeParam: i32,
    streamStage: u32,
    inBuff: *mut u8,
    inBuffSize: u64,
    inPos: u64,
    maxWindowSize: u64,
    outBuff: *mut u8,
    outBuffSize: u64,
    outStart: u64,
    outEnd: u64,
    lhSize: u64,
    hostageByte: u32,
    noForwardProgress: i32,
    outBufferMode: u32,
    expectedOutBuffer: ZSTD_outBuffer_s,
    litBuffer: *mut u8,
    litBufferEnd: *const u8,
    litBufferLocation: u32,
    litExtraBuffer: [u8; 65568],
    headerBuffer: [u8; 18],
    oversizedDuration: u64,
}


// zstd_ddict.c :
// concentrates all logic that needs to know the internals of ZSTD_DDict object
// Dependencies
// ZSTD_customMalloc, ZSTD_customFree
// ZSTD_memcpy, ZSTD_memmove, ZSTD_memset
// bmi2
// low level memory routines
/// Types
/// typedef'd to ZSTD_DDict within "zstd.h"
#[repr(C)]
#[derive(Clone, Copy)]
struct ZSTD_DDict_s {
    dictBuffer: *mut (),
    dictContent: *const (),
    dictSize: u64,
    entropy: __SlateRecord_zstd_decompress_internal_h_88,
    dictID: u32,
    entropyPresent: u32,
    cMem: __SlateRecord_zstd_lib_h_1837,
}


#[unsafe(no_mangle)]
extern "C" fn ZSTD_DDict_dictContent(ddict: *const ZSTD_DDict_s) -> *const () {
unsafe { (*ddict).dictContent }
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_DDict_dictSize(ddict: *const ZSTD_DDict_s) -> u64 {
unsafe { (*ddict).dictSize }
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_copyDDictParameters(dctx: *mut ZSTD_DCtx_s, ddict: *const ZSTD_DDict_s) {
while false {
}
unsafe {
(*dctx).dictID = (*ddict).dictID;
}
unsafe {
(*dctx).prefixStart = (*ddict).dictContent;
}
unsafe {
(*dctx).virtualStart = (*ddict).dictContent;
}
unsafe {
(*dctx).dictEnd = ((*ddict).dictContent as *const u8).wrapping_offset((*ddict).dictSize as isize) as *const ();
}
unsafe {
(*dctx).previousDstEnd = (*dctx).dictEnd;
}
if (unsafe { (*ddict).entropyPresent }) != 0 {
unsafe {
(*dctx).litEntropy = 1;
}
unsafe {
(*dctx).fseEntropy = 1;
}
unsafe {
(*dctx).LLTptr = (*ddict).entropy.LLTable.as_ptr() as *const __SlateRecord_zstd_decompress_internal_h_73;
}
unsafe {
(*dctx).MLTptr = (*ddict).entropy.MLTable.as_ptr() as *const __SlateRecord_zstd_decompress_internal_h_73;
}
unsafe {
(*dctx).OFTptr = (*ddict).entropy.OFTable.as_ptr() as *const __SlateRecord_zstd_decompress_internal_h_73;
}
unsafe {
(*dctx).HUFptr = (*ddict).entropy.hufTable.as_ptr() as *const u32;
}
unsafe {
*((*dctx).entropy.rep.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) = *((*ddict).entropy.rep.as_ptr() as *const u32).wrapping_offset(0 as isize);
}
unsafe {
*((*dctx).entropy.rep.as_mut_ptr() as *mut u32).wrapping_offset(1 as isize) = *((*ddict).entropy.rep.as_ptr() as *const u32).wrapping_offset(1 as isize);
}
unsafe {
*((*dctx).entropy.rep.as_mut_ptr() as *mut u32).wrapping_offset(2 as isize) = *((*ddict).entropy.rep.as_ptr() as *const u32).wrapping_offset(2 as isize);
}
} else {
unsafe {
(*dctx).litEntropy = 0;
}
unsafe {
(*dctx).fseEntropy = 0;
}
}
}

fn ZSTD_loadEntropy_intoDDict(ddict: *mut ZSTD_DDict_s, dictContentType: u32) -> u64 {
unsafe {
(*ddict).dictID = 0;
}
unsafe {
(*ddict).entropyPresent = 0;
}
if (dictContentType as u32) == 1 {
return 0;
}
if (unsafe { (*ddict).dictSize }) < 8 {
if (dictContentType as u32) == 2 {
return ERROR__4!();
}
// only accept specified dictionaries
return 0; // pure content mode
}
let magic: u32 = MEM_readLE32(unsafe { (*ddict).dictContent });
if magic != 3962610743 {
if (dictContentType as u32) == 2 {
return ERROR__4!();
}
// only accept specified dictionaries
return 0; // pure content mode
}
unsafe {
(*ddict).dictID = MEM_readLE32(((*ddict).dictContent as *const u8).wrapping_offset(4 as isize) as *const ());
}
// load entropy tables
loop {
if ERR_isError(unsafe { ZSTD_loadDEntropy(core::ptr::addr_of_mut!((*ddict).entropy), (*ddict).dictContent, (*ddict).dictSize) }) != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return ERROR__4!();
}
if true {
break;
}
}
unsafe {
(*ddict).entropyPresent = 1;
}
0
}

fn ZSTD_initDDict_internal(ddict: *mut ZSTD_DDict_s, dict: *const (), mut dictSize: u64, dictLoadMethod: u32, dictContentType: u32) -> u64 {
if (dictLoadMethod as u32) == 1 || dict.is_null() || dictSize == 0 {
unsafe {
(*ddict).dictBuffer = core::ptr::null_mut::<()>();
}
unsafe {
(*ddict).dictContent = dict;
}
if dict.is_null() {
dictSize = 0;
}
} else {
let internalBuffer: *mut () = ZSTD_customMalloc(dictSize, unsafe { (*ddict).cMem });
unsafe {
(*ddict).dictBuffer = internalBuffer;
}
unsafe {
(*ddict).dictContent = internalBuffer as *const ();
}
if internalBuffer.is_null() {
return ERROR__7!();
}
ZSTD_memcpy__5!(internalBuffer, dict, dictSize);
}
unsafe {
(*ddict).dictSize = dictSize;
}
unsafe {
*((*ddict).entropy.hufTable.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) = 201326604u32;
}
// cover both little and big endian
// parse dictionary content
loop {
let err_code: u64 = ZSTD_loadEntropy_intoDDict(ddict, dictContentType);
if ERR_isError(err_code) != 0 {
while false {
}
loop {
if false {
unsafe { _force_has_format_string(c"".as_ptr() as *const u8) };
}
if true {
break;
}
}
while false {
}
while false {
}
return err_code;
}
if true {
break;
}
}
0
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_createDDict_advanced(dict: *const (), dictSize: u64, dictLoadMethod: u32, dictContentType: u32, customMem: __SlateRecord_zstd_lib_h_1837) -> *mut ZSTD_DDict_s {
if ((customMem.customAlloc == None) as i32) ^ ((customMem.customFree == None) as i32) != 0 {
return core::ptr::null_mut::<ZSTD_DDict_s>();
}
let ddict: *mut ZSTD_DDict_s = ZSTD_customMalloc(core::mem::size_of::<ZSTD_DDict_s>() as u64, customMem) as *mut ZSTD_DDict_s;
if ddict.is_null() {
return core::ptr::null_mut::<ZSTD_DDict_s>();
}
unsafe {
(*ddict).cMem = customMem;
}
let initResult: u64 = ZSTD_initDDict_internal(ddict, dict, dictSize, dictLoadMethod, dictContentType);
if ERR_isError(initResult) != 0 {
ZSTD_freeDDict(ddict);
return core::ptr::null_mut::<ZSTD_DDict_s>();
}
ddict
}

/// ZSTD_createDDict() :
/// Create a digested dictionary, to start decompression without startup delay.
/// `dict` content is copied inside DDict.
/// Consequently, `dict` can be released after `ZSTD_DDict` creation
#[unsafe(no_mangle)]
extern "C" fn ZSTD_createDDict(dict: *const (), dictSize: u64) -> *mut ZSTD_DDict_s {
let allocator: __SlateRecord_zstd_lib_h_1837 = __SlateRecord_zstd_lib_h_1837 { customAlloc: None, customFree: None, opaque: core::ptr::null_mut::<()>() };
ZSTD_createDDict_advanced(dict, dictSize, 0u32, 0u32, allocator)
}

/// ZSTD_createDDict_byReference() :
/// Create a digested dictionary, to start decompression without startup delay.
/// Dictionary content is simply referenced, it will be accessed during decompression.
/// Warning : dictBuffer must outlive DDict (DDict must be freed before dictBuffer)
#[unsafe(no_mangle)]
extern "C" fn ZSTD_createDDict_byReference(dictBuffer: *const (), dictSize: u64) -> *mut ZSTD_DDict_s {
let allocator: __SlateRecord_zstd_lib_h_1837 = __SlateRecord_zstd_lib_h_1837 { customAlloc: None, customFree: None, opaque: core::ptr::null_mut::<()>() };
ZSTD_createDDict_advanced(dictBuffer, dictSize, 1u32, 0u32, allocator)
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_initStaticDDict(sBuffer: *mut (), sBufferSize: u64, mut dict: *const (), dictSize: u64, dictLoadMethod: u32, dictContentType: u32) -> *const ZSTD_DDict_s {
let neededSpace: u64 = (core::mem::size_of::<ZSTD_DDict_s>() as u64).wrapping_add(if (dictLoadMethod as u32) == 1 { 0 } else { dictSize });
let ddict: *mut ZSTD_DDict_s = sBuffer as *mut ZSTD_DDict_s;
if (sBuffer as u64) & 7 != 0 {
return core::ptr::null::<ZSTD_DDict_s>();
}
// 8-aligned
if sBufferSize < neededSpace {
return core::ptr::null::<ZSTD_DDict_s>();
}
if (dictLoadMethod as u32) == 0 {
ZSTD_memcpy__2!(ddict.wrapping_offset(1 as isize), dict, dictSize); // local copy
dict = ddict.wrapping_offset(1 as isize) as *const ();
}
if ERR_isError(ZSTD_initDDict_internal(ddict, dict, dictSize, 1u32, dictContentType)) != 0 {
return core::ptr::null::<ZSTD_DDict_s>();
}
ddict as *const ZSTD_DDict_s
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_freeDDict(ddict: *mut ZSTD_DDict_s) -> u64 {
if ddict.is_null() {
return 0;
}
// support free on NULL
let cMem: __SlateRecord_zstd_lib_h_1837 = unsafe { (*ddict).cMem };
ZSTD_customFree(unsafe { (*ddict).dictBuffer }, cMem);
ZSTD_customFree(ddict as *mut (), cMem);
0
}

/// ZSTD_estimateDDictSize() :
/// Estimate amount of memory that will be needed to create a dictionary for decompression.
/// Note : dictionary created by reference using ZSTD_dlm_byRef are smaller
#[unsafe(no_mangle)]
extern "C" fn ZSTD_estimateDDictSize(dictSize: u64, dictLoadMethod: u32) -> u64 {
(core::mem::size_of::<ZSTD_DDict_s>() as u64).wrapping_add(if (dictLoadMethod as u32) == 1 { 0 } else { dictSize })
}

#[unsafe(no_mangle)]
extern "C" fn ZSTD_sizeof_DDict(ddict: *const ZSTD_DDict_s) -> u64 {
if ddict.is_null() {
return 0;
}
// support sizeof on NULL
(core::mem::size_of::<ZSTD_DDict_s>() as u64).wrapping_add(if !unsafe { (*ddict).dictBuffer }.is_null() { unsafe { (*ddict).dictSize } } else { 0 })
}

/// ZSTD_getDictID_fromDDict() :
/// Provides the dictID of the dictionary loaded into `ddict`.
/// If @return == 0, the dictionary is not conformant to Zstandard specification, or empty.
/// Non-conformant dictionaries can still be loaded, but as content-only dictionaries.
#[unsafe(no_mangle)]
extern "C" fn ZSTD_getDictID_fromDDict(ddict: *const ZSTD_DDict_s) -> u32 {
if ddict.is_null() {
return 0;
}
unsafe { (*ddict).dictID }
}
