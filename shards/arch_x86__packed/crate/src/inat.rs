
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0-or-later
//!
//! x86 instruction attribute tables
//!
//! Written by Masami Hiramatsu <mhiramat@redhat.com>
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! Legacy last prefixes
//! REX2 M0
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// VEX bit flags
// VEX3 Byte2
// VEX bit fields
// EVEX Byte1
// VEX3 Byte1
// VEX2.M always 1
// VEX3 Byte2, VEX2 Byte1
// VEX3.M Maximum value
// XOP Byte2
// XOP Byte3
// Min of XOP.M
// Max of XOP.M
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
macro_rules! X86_MODRM_REG {
($modrm:expr) => {
((($modrm) as i32) & 56) >> 3i32
};
}

#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

// 0x66
//
// LPFX1
// 0xF3
//
// LPFX2
// 0xF2
//
// LPFX3
// Other Legacy prefixes
// 0xF0
// 0x2E
// 0x3E
// 0x26
// 0x64
// 0x65
// 0x36
// 0x67
// x86-64 REX prefix
// 0x4X
// AVX VEX prefixes
// 2-bytes VEX prefix
// 3-bytes VEX prefix
// EVEX prefix
// x86-64 REX2 prefix
// 0xD5
// AMD XOP prefix
// 0x8F
// Immediate size
// Escape opcodes
// Group opcodes (1-16)
// Immediates
// Attribute making macros for attribute tables
fn inat_last_prefix_id(attr: u32) -> i32 {
if attr & 31u32 > 3 {
return 0;
} else {
return (attr & 31u32) as i32;
}
}

fn inat_escape_id(attr: u32) -> i32 {
((attr & 96u32) >> 5i32) as i32
}

fn inat_is_group(attr: u32) -> i32 {
(attr & 3968u32) as i32
}

fn inat_group_id(attr: u32) -> i32 {
((attr & 3968u32) >> 7i32) as i32
}

fn inat_group_common_attribute(attr: u32) -> i32 {
(attr & 4294963327u32) as i32
}

fn inat_has_variant(attr: u32) -> i32 {
(attr & 524288u32) as i32
}

#[unsafe(no_mangle)]
static inat_primary_table: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[0] = 32768u32;
__t0[1] = 32768u32;
__t0[2] = 32768u32;
__t0[3] = 32768u32;
__t0[4] = 4096u32;
__t0[5] = 24576u32;
__t0[6] = 67108864u32;
__t0[7] = 67108864u32;
__t0[8] = 32768u32;
__t0[9] = 32768u32;
__t0[10] = 32768u32;
__t0[11] = 32768u32;
__t0[12] = 4096u32;
__t0[13] = 24576u32;
__t0[14] = 67108864u32;
__t0[15] = 32u32;
__t0[16] = 32768u32;
__t0[17] = 32768u32;
__t0[18] = 32768u32;
__t0[19] = 32768u32;
__t0[20] = 4096u32;
__t0[21] = 24576u32;
__t0[22] = 67108864u32;
__t0[23] = 67108864u32;
__t0[24] = 32768u32;
__t0[25] = 32768u32;
__t0[26] = 32768u32;
__t0[27] = 32768u32;
__t0[28] = 4096u32;
__t0[29] = 24576u32;
__t0[30] = 67108864u32;
__t0[31] = 67108864u32;
__t0[32] = 32768u32;
__t0[33] = 32768u32;
__t0[34] = 32768u32;
__t0[35] = 32768u32;
__t0[36] = 4096u32;
__t0[37] = 24576u32;
__t0[38] = 7u32;
__t0[39] = 67108864u32;
__t0[40] = 32768u32;
__t0[41] = 32768u32;
__t0[42] = 32768u32;
__t0[43] = 32768u32;
__t0[44] = 4096u32;
__t0[45] = 24576u32;
__t0[46] = 5u32;
__t0[47] = 67108864u32;
__t0[48] = 32768u32;
__t0[49] = 32768u32;
__t0[50] = 32768u32;
__t0[51] = 32768u32;
__t0[52] = 4096u32;
__t0[53] = 24576u32;
__t0[54] = 10u32;
__t0[55] = 67108864u32;
__t0[56] = 32768u32;
__t0[57] = 32768u32;
__t0[58] = 32768u32;
__t0[59] = 32768u32;
__t0[60] = 4096u32;
__t0[61] = 24576u32;
__t0[62] = 6u32;
__t0[63] = 67108864u32;
__t0[64] = 12u32;
__t0[65] = 12u32;
__t0[66] = 12u32;
__t0[67] = 12u32;
__t0[68] = 12u32;
__t0[69] = 12u32;
__t0[70] = 12u32;
__t0[71] = 12u32;
__t0[72] = 12u32;
__t0[73] = 12u32;
__t0[74] = 12u32;
__t0[75] = 12u32;
__t0[76] = 12u32;
__t0[77] = 12u32;
__t0[78] = 12u32;
__t0[79] = 12u32;
__t0[80] = 65536u32;
__t0[81] = 65536u32;
__t0[82] = 65536u32;
__t0[83] = 65536u32;
__t0[84] = 65536u32;
__t0[85] = 65536u32;
__t0[86] = 65536u32;
__t0[87] = 65536u32;
__t0[88] = 65536u32;
__t0[89] = 65536u32;
__t0[90] = 65536u32;
__t0[91] = 65536u32;
__t0[92] = 65536u32;
__t0[93] = 65536u32;
__t0[94] = 65536u32;
__t0[95] = 65536u32;
__t0[96] = 67108864u32;
__t0[97] = 67108864u32;
__t0[98] = 32783u32;
__t0[99] = 32768u32;
__t0[100] = 8u32;
__t0[101] = 9u32;
__t0[102] = 1u32;
__t0[103] = 11u32;
__t0[104] = 24576u32;
__t0[105] = 57344u32;
__t0[106] = 69632u32;
__t0[107] = 36864u32;
__t0[112] = 8392704u32;
__t0[113] = 8392704u32;
__t0[114] = 8392704u32;
__t0[115] = 8392704u32;
__t0[116] = 8392704u32;
__t0[117] = 8392704u32;
__t0[118] = 8392704u32;
__t0[119] = 8392704u32;
__t0[120] = 8392704u32;
__t0[121] = 8392704u32;
__t0[122] = 8392704u32;
__t0[123] = 8392704u32;
__t0[124] = 8392704u32;
__t0[125] = 8392704u32;
__t0[126] = 8392704u32;
__t0[127] = 8392704u32;
__t0[128] = 36992u32;
__t0[129] = 57472u32;
__t0[130] = 67145856u32;
__t0[131] = 36992u32;
__t0[132] = 32768u32;
__t0[133] = 32768u32;
__t0[134] = 32768u32;
__t0[135] = 32768u32;
__t0[136] = 32768u32;
__t0[137] = 32768u32;
__t0[138] = 32768u32;
__t0[139] = 32768u32;
__t0[140] = 32768u32;
__t0[141] = 32768u32;
__t0[142] = 32768u32;
__t0[143] = 98577u32;
__t0[154] = 67129344u32;
__t0[156] = 65536u32;
__t0[157] = 65536u32;
__t0[160] = 8650752u32;
__t0[161] = 8650752u32;
__t0[162] = 8650752u32;
__t0[163] = 8650752u32;
__t0[164] = 8388608u32;
__t0[165] = 8388608u32;
__t0[166] = 8388608u32;
__t0[167] = 8388608u32;
__t0[168] = 8392704u32;
__t0[169] = 8413184u32;
__t0[170] = 8388608u32;
__t0[171] = 8388608u32;
__t0[172] = 8388608u32;
__t0[173] = 8388608u32;
__t0[174] = 8388608u32;
__t0[175] = 8388608u32;
__t0[176] = 4096u32;
__t0[177] = 4096u32;
__t0[178] = 4096u32;
__t0[179] = 4096u32;
__t0[180] = 4096u32;
__t0[181] = 4096u32;
__t0[182] = 4096u32;
__t0[183] = 4096u32;
__t0[184] = 28672u32;
__t0[185] = 28672u32;
__t0[186] = 28672u32;
__t0[187] = 28672u32;
__t0[188] = 28672u32;
__t0[189] = 28672u32;
__t0[190] = 28672u32;
__t0[191] = 28672u32;
__t0[192] = 37248u32;
__t0[193] = 37248u32;
__t0[194] = 73728u32;
__t0[196] = 32782u32;
__t0[197] = 32781u32;
__t0[198] = 37376u32;
__t0[199] = 57984u32;
__t0[200] = 139264u32;
__t0[201] = 65536u32;
__t0[202] = 8192u32;
__t0[205] = 4096u32;
__t0[206] = 67108864u32;
__t0[208] = 33152u32;
__t0[209] = 33152u32;
__t0[210] = 33152u32;
__t0[211] = 33152u32;
__t0[212] = 67112960u32;
__t0[213] = 4112u32;
__t0[216] = 32768u32;
__t0[217] = 32768u32;
__t0[218] = 32768u32;
__t0[219] = 32768u32;
__t0[220] = 32768u32;
__t0[221] = 32768u32;
__t0[222] = 32768u32;
__t0[223] = 32768u32;
__t0[224] = 8458240u32;
__t0[225] = 8458240u32;
__t0[226] = 8458240u32;
__t0[227] = 8458240u32;
__t0[228] = 8392704u32;
__t0[229] = 8392704u32;
__t0[230] = 8392704u32;
__t0[231] = 8392704u32;
__t0[232] = 8478720u32;
__t0[233] = 8478720u32;
__t0[234] = 75517952u32;
__t0[235] = 8458240u32;
__t0[236] = 8388608u32;
__t0[237] = 8388608u32;
__t0[238] = 8388608u32;
__t0[239] = 8388608u32;
__t0[240] = 4u32;
__t0[242] = 3u32;
__t0[243] = 2u32;
__t0[246] = 33536u32;
__t0[247] = 33664u32;
__t0[254] = 33792u32;
__t0[255] = 33920u32;
__t0
});

#[unsafe(no_mangle)]
static inat_escape_table_1: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[0] = 34048u32;
__t0[1] = 34176u32;
__t0[2] = 32768u32;
__t0[3] = 32768u32;
__t0[13] = 34304u32;
__t0[15] = 36864u32;
__t0[16] = 1605632u32;
__t0[17] = 1605632u32;
__t0[18] = 1605632u32;
__t0[19] = 1605632u32;
__t0[20] = 1605632u32;
__t0[21] = 1605632u32;
__t0[22] = 1605632u32;
__t0[23] = 1605632u32;
__t0[24] = 34432u32;
__t0[26] = 557056u32;
__t0[27] = 557056u32;
__t0[28] = 34560u32;
__t0[30] = 34688u32;
__t0[31] = 32768u32;
__t0[32] = 32768u32;
__t0[33] = 32768u32;
__t0[34] = 32768u32;
__t0[35] = 32768u32;
__t0[40] = 1605632u32;
__t0[41] = 1605632u32;
__t0[42] = 557056u32;
__t0[43] = 1605632u32;
__t0[44] = 557056u32;
__t0[45] = 557056u32;
__t0[46] = 1605632u32;
__t0[47] = 1605632u32;
__t0[48] = 8388608u32;
__t0[49] = 8388608u32;
__t0[50] = 8388608u32;
__t0[51] = 8388608u32;
__t0[52] = 8388608u32;
__t0[53] = 8388608u32;
__t0[55] = 8388608u32;
__t0[56] = 64u32;
__t0[58] = 96u32;
__t0[64] = 32768u32;
__t0[65] = 1605632u32;
__t0[66] = 1605632u32;
__t0[67] = 32768u32;
__t0[68] = 1605632u32;
__t0[69] = 1605632u32;
__t0[70] = 1605632u32;
__t0[71] = 1605632u32;
__t0[72] = 32768u32;
__t0[73] = 32768u32;
__t0[74] = 1605632u32;
__t0[75] = 1605632u32;
__t0[76] = 32768u32;
__t0[77] = 32768u32;
__t0[78] = 32768u32;
__t0[79] = 32768u32;
__t0[80] = 1605632u32;
__t0[81] = 1605632u32;
__t0[82] = 1605632u32;
__t0[83] = 1605632u32;
__t0[84] = 1605632u32;
__t0[85] = 1605632u32;
__t0[86] = 1605632u32;
__t0[87] = 1605632u32;
__t0[88] = 1605632u32;
__t0[89] = 1605632u32;
__t0[90] = 1605632u32;
__t0[91] = 1605632u32;
__t0[92] = 1605632u32;
__t0[93] = 1605632u32;
__t0[94] = 1605632u32;
__t0[95] = 1605632u32;
__t0[96] = 557056u32;
__t0[97] = 557056u32;
__t0[98] = 557056u32;
__t0[99] = 557056u32;
__t0[100] = 557056u32;
__t0[101] = 557056u32;
__t0[102] = 557056u32;
__t0[103] = 557056u32;
__t0[104] = 557056u32;
__t0[105] = 557056u32;
__t0[106] = 557056u32;
__t0[107] = 557056u32;
__t0[108] = 524288u32;
__t0[109] = 524288u32;
__t0[110] = 557056u32;
__t0[111] = 557056u32;
__t0[112] = 561152u32;
__t0[113] = 34816u32;
__t0[114] = 34944u32;
__t0[115] = 35072u32;
__t0[116] = 557056u32;
__t0[117] = 557056u32;
__t0[118] = 557056u32;
__t0[119] = 1048576u32;
__t0[120] = 1605632u32;
__t0[121] = 1605632u32;
__t0[122] = 524288u32;
__t0[123] = 524288u32;
__t0[124] = 524288u32;
__t0[125] = 524288u32;
__t0[126] = 557056u32;
__t0[127] = 557056u32;
__t0[128] = 8478720u32;
__t0[129] = 8478720u32;
__t0[130] = 8478720u32;
__t0[131] = 8478720u32;
__t0[132] = 8478720u32;
__t0[133] = 8478720u32;
__t0[134] = 8478720u32;
__t0[135] = 8478720u32;
__t0[136] = 8478720u32;
__t0[137] = 8478720u32;
__t0[138] = 8478720u32;
__t0[139] = 8478720u32;
__t0[140] = 8478720u32;
__t0[141] = 8478720u32;
__t0[142] = 8478720u32;
__t0[143] = 8478720u32;
__t0[144] = 1605632u32;
__t0[145] = 1605632u32;
__t0[146] = 1605632u32;
__t0[147] = 1605632u32;
__t0[148] = 32768u32;
__t0[149] = 32768u32;
__t0[150] = 32768u32;
__t0[151] = 32768u32;
__t0[152] = 1605632u32;
__t0[153] = 1605632u32;
__t0[154] = 32768u32;
__t0[155] = 32768u32;
__t0[156] = 32768u32;
__t0[157] = 32768u32;
__t0[158] = 32768u32;
__t0[159] = 32768u32;
__t0[160] = 65536u32;
__t0[161] = 65536u32;
__t0[163] = 32768u32;
__t0[164] = 36864u32;
__t0[165] = 32768u32;
__t0[166] = 35200u32;
__t0[167] = 35328u32;
__t0[168] = 65536u32;
__t0[169] = 65536u32;
__t0[171] = 32768u32;
__t0[172] = 36864u32;
__t0[173] = 32768u32;
__t0[174] = 35456u32;
__t0[175] = 32768u32;
__t0[176] = 32768u32;
__t0[177] = 32768u32;
__t0[178] = 32768u32;
__t0[179] = 32768u32;
__t0[180] = 32768u32;
__t0[181] = 32768u32;
__t0[182] = 32768u32;
__t0[183] = 32768u32;
__t0[184] = 524288u32;
__t0[185] = 35584u32;
__t0[186] = 39808u32;
__t0[187] = 32768u32;
__t0[188] = 557056u32;
__t0[189] = 557056u32;
__t0[190] = 32768u32;
__t0[191] = 32768u32;
__t0[192] = 32768u32;
__t0[193] = 32768u32;
__t0[194] = 1609728u32;
__t0[195] = 32768u32;
__t0[196] = 561152u32;
__t0[197] = 561152u32;
__t0[198] = 1609728u32;
__t0[199] = 35840u32;
__t0[208] = 524288u32;
__t0[209] = 557056u32;
__t0[210] = 557056u32;
__t0[211] = 557056u32;
__t0[212] = 557056u32;
__t0[213] = 557056u32;
__t0[214] = 524288u32;
__t0[215] = 557056u32;
__t0[216] = 557056u32;
__t0[217] = 557056u32;
__t0[218] = 557056u32;
__t0[219] = 557056u32;
__t0[220] = 557056u32;
__t0[221] = 557056u32;
__t0[222] = 557056u32;
__t0[223] = 557056u32;
__t0[224] = 557056u32;
__t0[225] = 557056u32;
__t0[226] = 557056u32;
__t0[227] = 557056u32;
__t0[228] = 557056u32;
__t0[229] = 557056u32;
__t0[230] = 524288u32;
__t0[231] = 557056u32;
__t0[232] = 557056u32;
__t0[233] = 557056u32;
__t0[234] = 557056u32;
__t0[235] = 557056u32;
__t0[236] = 557056u32;
__t0[237] = 557056u32;
__t0[238] = 557056u32;
__t0[239] = 557056u32;
__t0[240] = 524288u32;
__t0[241] = 557056u32;
__t0[242] = 557056u32;
__t0[243] = 557056u32;
__t0[244] = 557056u32;
__t0[245] = 557056u32;
__t0[246] = 557056u32;
__t0[247] = 557056u32;
__t0[248] = 557056u32;
__t0[249] = 557056u32;
__t0[250] = 557056u32;
__t0[251] = 557056u32;
__t0[252] = 557056u32;
__t0[253] = 557056u32;
__t0[254] = 557056u32;
__t0
});

#[unsafe(no_mangle)]
static inat_escape_table_1_1: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[16] = 1081344u32;
__t0[17] = 1081344u32;
__t0[18] = 1081344u32;
__t0[19] = 1081344u32;
__t0[20] = 1081344u32;
__t0[21] = 1081344u32;
__t0[22] = 1081344u32;
__t0[23] = 1081344u32;
__t0[26] = 32768u32;
__t0[27] = 32768u32;
__t0[40] = 1081344u32;
__t0[41] = 1081344u32;
__t0[42] = 32768u32;
__t0[43] = 1081344u32;
__t0[44] = 32768u32;
__t0[45] = 32768u32;
__t0[46] = 1081344u32;
__t0[47] = 1081344u32;
__t0[65] = 1081344u32;
__t0[66] = 1081344u32;
__t0[68] = 1081344u32;
__t0[69] = 1081344u32;
__t0[70] = 1081344u32;
__t0[71] = 1081344u32;
__t0[74] = 1081344u32;
__t0[75] = 1081344u32;
__t0[80] = 1081344u32;
__t0[81] = 1081344u32;
__t0[84] = 1081344u32;
__t0[85] = 1081344u32;
__t0[86] = 1081344u32;
__t0[87] = 1081344u32;
__t0[88] = 1081344u32;
__t0[89] = 1081344u32;
__t0[90] = 1081344u32;
__t0[91] = 1081344u32;
__t0[92] = 1081344u32;
__t0[93] = 1081344u32;
__t0[94] = 1081344u32;
__t0[95] = 1081344u32;
__t0[96] = 1081344u32;
__t0[97] = 1081344u32;
__t0[98] = 1081344u32;
__t0[99] = 1081344u32;
__t0[100] = 1081344u32;
__t0[101] = 1081344u32;
__t0[102] = 1081344u32;
__t0[103] = 1081344u32;
__t0[104] = 1081344u32;
__t0[105] = 1081344u32;
__t0[106] = 1081344u32;
__t0[107] = 1081344u32;
__t0[108] = 1081344u32;
__t0[109] = 1081344u32;
__t0[110] = 1081344u32;
__t0[111] = 1081344u32;
__t0[112] = 1085440u32;
__t0[116] = 1081344u32;
__t0[117] = 1081344u32;
__t0[118] = 1081344u32;
__t0[120] = 5275648u32;
__t0[121] = 5275648u32;
__t0[122] = 5275648u32;
__t0[123] = 5275648u32;
__t0[124] = 1081344u32;
__t0[125] = 1081344u32;
__t0[126] = 1081344u32;
__t0[127] = 1081344u32;
__t0[144] = 1081344u32;
__t0[145] = 1081344u32;
__t0[146] = 1081344u32;
__t0[147] = 1081344u32;
__t0[152] = 1081344u32;
__t0[153] = 1081344u32;
__t0[188] = 32768u32;
__t0[189] = 32768u32;
__t0[194] = 1085440u32;
__t0[196] = 1085440u32;
__t0[197] = 1085440u32;
__t0[198] = 1085440u32;
__t0[208] = 1081344u32;
__t0[209] = 1081344u32;
__t0[210] = 1081344u32;
__t0[211] = 1081344u32;
__t0[212] = 1081344u32;
__t0[213] = 1081344u32;
__t0[214] = 1081344u32;
__t0[215] = 1081344u32;
__t0[216] = 1081344u32;
__t0[217] = 1081344u32;
__t0[218] = 1081344u32;
__t0[219] = 1081344u32;
__t0[220] = 1081344u32;
__t0[221] = 1081344u32;
__t0[222] = 1081344u32;
__t0[223] = 1081344u32;
__t0[224] = 1081344u32;
__t0[225] = 1081344u32;
__t0[226] = 1081344u32;
__t0[227] = 1081344u32;
__t0[228] = 1081344u32;
__t0[229] = 1081344u32;
__t0[230] = 1081344u32;
__t0[231] = 1081344u32;
__t0[232] = 1081344u32;
__t0[233] = 1081344u32;
__t0[234] = 1081344u32;
__t0[235] = 1081344u32;
__t0[236] = 1081344u32;
__t0[237] = 1081344u32;
__t0[238] = 1081344u32;
__t0[239] = 1081344u32;
__t0[241] = 1081344u32;
__t0[242] = 1081344u32;
__t0[243] = 1081344u32;
__t0[244] = 1081344u32;
__t0[245] = 1081344u32;
__t0[246] = 1081344u32;
__t0[247] = 1081344u32;
__t0[248] = 1081344u32;
__t0[249] = 1081344u32;
__t0[250] = 1081344u32;
__t0[251] = 1081344u32;
__t0[252] = 1081344u32;
__t0[253] = 1081344u32;
__t0[254] = 1081344u32;
__t0
});

#[unsafe(no_mangle)]
static inat_escape_table_1_2: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[16] = 1081344u32;
__t0[17] = 1081344u32;
__t0[18] = 1081344u32;
__t0[22] = 1081344u32;
__t0[26] = 32768u32;
__t0[27] = 32768u32;
__t0[42] = 1081344u32;
__t0[44] = 1081344u32;
__t0[45] = 1081344u32;
__t0[81] = 1081344u32;
__t0[82] = 1081344u32;
__t0[83] = 1081344u32;
__t0[88] = 1081344u32;
__t0[89] = 1081344u32;
__t0[90] = 1081344u32;
__t0[91] = 1081344u32;
__t0[92] = 1081344u32;
__t0[93] = 1081344u32;
__t0[94] = 1081344u32;
__t0[95] = 1081344u32;
__t0[111] = 1081344u32;
__t0[112] = 1085440u32;
__t0[120] = 5275648u32;
__t0[121] = 5275648u32;
__t0[122] = 5275648u32;
__t0[123] = 5275648u32;
__t0[126] = 1081344u32;
__t0[127] = 1081344u32;
__t0[184] = 32768u32;
__t0[188] = 32768u32;
__t0[189] = 32768u32;
__t0[194] = 1085440u32;
__t0[214] = 32768u32;
__t0[230] = 1081344u32;
__t0
});

#[unsafe(no_mangle)]
static inat_escape_table_1_3: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[16] = 1081344u32;
__t0[17] = 1081344u32;
__t0[18] = 1081344u32;
__t0[26] = 32768u32;
__t0[27] = 32768u32;
__t0[42] = 1081344u32;
__t0[44] = 1081344u32;
__t0[45] = 1081344u32;
__t0[81] = 1081344u32;
__t0[88] = 1081344u32;
__t0[89] = 1081344u32;
__t0[90] = 1081344u32;
__t0[92] = 1081344u32;
__t0[93] = 1081344u32;
__t0[94] = 1081344u32;
__t0[95] = 1081344u32;
__t0[111] = 5275648u32;
__t0[112] = 1085440u32;
__t0[120] = 5275648u32;
__t0[121] = 5275648u32;
__t0[122] = 5275648u32;
__t0[123] = 5275648u32;
__t0[124] = 1081344u32;
__t0[125] = 1081344u32;
__t0[127] = 5275648u32;
__t0[146] = 1081344u32;
__t0[147] = 1081344u32;
__t0[188] = 32768u32;
__t0[189] = 32768u32;
__t0[194] = 1085440u32;
__t0[208] = 1081344u32;
__t0[214] = 32768u32;
__t0[230] = 1081344u32;
__t0[240] = 1081344u32;
__t0
});

#[unsafe(no_mangle)]
static inat_escape_table_2: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[0] = 557056u32;
__t0[1] = 557056u32;
__t0[2] = 557056u32;
__t0[3] = 557056u32;
__t0[4] = 557056u32;
__t0[5] = 557056u32;
__t0[6] = 557056u32;
__t0[7] = 557056u32;
__t0[8] = 557056u32;
__t0[9] = 557056u32;
__t0[10] = 557056u32;
__t0[11] = 557056u32;
__t0[12] = 524288u32;
__t0[13] = 524288u32;
__t0[14] = 524288u32;
__t0[15] = 524288u32;
__t0[16] = 524288u32;
__t0[17] = 524288u32;
__t0[18] = 524288u32;
__t0[19] = 524288u32;
__t0[20] = 524288u32;
__t0[21] = 524288u32;
__t0[22] = 524288u32;
__t0[23] = 524288u32;
__t0[24] = 524288u32;
__t0[25] = 524288u32;
__t0[26] = 524288u32;
__t0[27] = 524288u32;
__t0[28] = 557056u32;
__t0[29] = 557056u32;
__t0[30] = 557056u32;
__t0[31] = 524288u32;
__t0[32] = 524288u32;
__t0[33] = 524288u32;
__t0[34] = 524288u32;
__t0[35] = 524288u32;
__t0[36] = 524288u32;
__t0[37] = 524288u32;
__t0[38] = 524288u32;
__t0[39] = 524288u32;
__t0[40] = 524288u32;
__t0[41] = 524288u32;
__t0[42] = 524288u32;
__t0[43] = 524288u32;
__t0[44] = 524288u32;
__t0[45] = 524288u32;
__t0[46] = 524288u32;
__t0[47] = 524288u32;
__t0[48] = 524288u32;
__t0[49] = 524288u32;
__t0[50] = 524288u32;
__t0[51] = 524288u32;
__t0[52] = 524288u32;
__t0[53] = 524288u32;
__t0[54] = 524288u32;
__t0[55] = 524288u32;
__t0[56] = 524288u32;
__t0[57] = 524288u32;
__t0[58] = 524288u32;
__t0[59] = 524288u32;
__t0[60] = 524288u32;
__t0[61] = 524288u32;
__t0[62] = 524288u32;
__t0[63] = 524288u32;
__t0[64] = 524288u32;
__t0[65] = 524288u32;
__t0[66] = 524288u32;
__t0[67] = 524288u32;
__t0[68] = 524288u32;
__t0[69] = 524288u32;
__t0[70] = 524288u32;
__t0[71] = 524288u32;
__t0[73] = 1605632u32;
__t0[75] = 524288u32;
__t0[76] = 524288u32;
__t0[77] = 524288u32;
__t0[78] = 524288u32;
__t0[79] = 524288u32;
__t0[80] = 3702784u32;
__t0[81] = 3702784u32;
__t0[82] = 524288u32;
__t0[83] = 524288u32;
__t0[84] = 524288u32;
__t0[85] = 524288u32;
__t0[88] = 524288u32;
__t0[89] = 524288u32;
__t0[90] = 524288u32;
__t0[91] = 524288u32;
__t0[92] = 524288u32;
__t0[94] = 1605632u32;
__t0[98] = 524288u32;
__t0[99] = 524288u32;
__t0[100] = 524288u32;
__t0[101] = 524288u32;
__t0[102] = 524288u32;
__t0[104] = 524288u32;
__t0[108] = 1605632u32;
__t0[112] = 524288u32;
__t0[113] = 524288u32;
__t0[114] = 524288u32;
__t0[115] = 524288u32;
__t0[117] = 524288u32;
__t0[118] = 524288u32;
__t0[119] = 524288u32;
__t0[120] = 524288u32;
__t0[121] = 524288u32;
__t0[122] = 524288u32;
__t0[123] = 524288u32;
__t0[124] = 524288u32;
__t0[125] = 524288u32;
__t0[126] = 524288u32;
__t0[127] = 524288u32;
__t0[128] = 524288u32;
__t0[129] = 524288u32;
__t0[130] = 524288u32;
__t0[131] = 524288u32;
__t0[136] = 524288u32;
__t0[137] = 524288u32;
__t0[138] = 524288u32;
__t0[139] = 524288u32;
__t0[140] = 524288u32;
__t0[141] = 524288u32;
__t0[142] = 524288u32;
__t0[143] = 524288u32;
__t0[144] = 524288u32;
__t0[145] = 524288u32;
__t0[146] = 524288u32;
__t0[147] = 524288u32;
__t0[150] = 524288u32;
__t0[151] = 524288u32;
__t0[152] = 524288u32;
__t0[153] = 524288u32;
__t0[154] = 524288u32;
__t0[155] = 524288u32;
__t0[156] = 524288u32;
__t0[157] = 524288u32;
__t0[158] = 524288u32;
__t0[159] = 524288u32;
__t0[160] = 524288u32;
__t0[161] = 524288u32;
__t0[162] = 524288u32;
__t0[163] = 524288u32;
__t0[166] = 524288u32;
__t0[167] = 524288u32;
__t0[168] = 524288u32;
__t0[169] = 524288u32;
__t0[170] = 524288u32;
__t0[171] = 524288u32;
__t0[172] = 524288u32;
__t0[173] = 524288u32;
__t0[174] = 524288u32;
__t0[175] = 524288u32;
__t0[176] = 3702784u32;
__t0[177] = 524288u32;
__t0[180] = 524288u32;
__t0[181] = 524288u32;
__t0[182] = 524288u32;
__t0[183] = 524288u32;
__t0[184] = 524288u32;
__t0[185] = 524288u32;
__t0[186] = 524288u32;
__t0[187] = 524288u32;
__t0[188] = 524288u32;
__t0[189] = 524288u32;
__t0[190] = 524288u32;
__t0[191] = 524288u32;
__t0[196] = 524288u32;
__t0[198] = 35968u32;
__t0[199] = 36096u32;
__t0[200] = 557056u32;
__t0[201] = 32768u32;
__t0[202] = 557056u32;
__t0[203] = 557056u32;
__t0[204] = 557056u32;
__t0[205] = 557056u32;
__t0[207] = 524288u32;
__t0[210] = 3702784u32;
__t0[211] = 3702784u32;
__t0[216] = 524288u32;
__t0[218] = 1605632u32;
__t0[219] = 524288u32;
__t0[220] = 524288u32;
__t0[221] = 524288u32;
__t0[222] = 524288u32;
__t0[223] = 524288u32;
__t0[224] = 524288u32;
__t0[225] = 524288u32;
__t0[226] = 524288u32;
__t0[227] = 524288u32;
__t0[228] = 524288u32;
__t0[229] = 524288u32;
__t0[230] = 524288u32;
__t0[231] = 524288u32;
__t0[232] = 524288u32;
__t0[233] = 524288u32;
__t0[234] = 524288u32;
__t0[235] = 524288u32;
__t0[236] = 524288u32;
__t0[237] = 524288u32;
__t0[238] = 524288u32;
__t0[239] = 524288u32;
__t0[240] = 557056u32;
__t0[241] = 557056u32;
__t0[242] = 3178496u32;
__t0[243] = 36224u32;
__t0[245] = 3702784u32;
__t0[246] = 557056u32;
__t0[247] = 3702784u32;
__t0[248] = 524288u32;
__t0[249] = 32768u32;
__t0[250] = 524288u32;
__t0[251] = 524288u32;
__t0[252] = 557056u32;
__t0
});

#[unsafe(no_mangle)]
static inat_escape_table_2_1: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[0] = 1081344u32;
__t0[1] = 1081344u32;
__t0[2] = 1081344u32;
__t0[3] = 1081344u32;
__t0[4] = 1081344u32;
__t0[5] = 1081344u32;
__t0[6] = 1081344u32;
__t0[7] = 1081344u32;
__t0[8] = 1081344u32;
__t0[9] = 1081344u32;
__t0[10] = 1081344u32;
__t0[11] = 1081344u32;
__t0[12] = 3178496u32;
__t0[13] = 3178496u32;
__t0[14] = 3178496u32;
__t0[15] = 3178496u32;
__t0[16] = 1081344u32;
__t0[17] = 5275648u32;
__t0[18] = 5275648u32;
__t0[19] = 3178496u32;
__t0[20] = 1081344u32;
__t0[21] = 1081344u32;
__t0[22] = 3178496u32;
__t0[23] = 1081344u32;
__t0[24] = 3178496u32;
__t0[25] = 3178496u32;
__t0[26] = 3178496u32;
__t0[27] = 5275648u32;
__t0[28] = 1081344u32;
__t0[29] = 1081344u32;
__t0[30] = 1081344u32;
__t0[31] = 5275648u32;
__t0[32] = 1081344u32;
__t0[33] = 1081344u32;
__t0[34] = 1081344u32;
__t0[35] = 1081344u32;
__t0[36] = 1081344u32;
__t0[37] = 1081344u32;
__t0[38] = 5275648u32;
__t0[39] = 5275648u32;
__t0[40] = 1081344u32;
__t0[41] = 1081344u32;
__t0[42] = 1081344u32;
__t0[43] = 1081344u32;
__t0[44] = 3178496u32;
__t0[45] = 3178496u32;
__t0[46] = 3178496u32;
__t0[47] = 3178496u32;
__t0[48] = 1081344u32;
__t0[49] = 1081344u32;
__t0[50] = 1081344u32;
__t0[51] = 1081344u32;
__t0[52] = 1081344u32;
__t0[53] = 1081344u32;
__t0[54] = 3178496u32;
__t0[55] = 1081344u32;
__t0[56] = 1081344u32;
__t0[57] = 1081344u32;
__t0[58] = 1081344u32;
__t0[59] = 1081344u32;
__t0[60] = 1081344u32;
__t0[61] = 1081344u32;
__t0[62] = 1081344u32;
__t0[63] = 1081344u32;
__t0[64] = 1081344u32;
__t0[65] = 1081344u32;
__t0[66] = 5275648u32;
__t0[67] = 5275648u32;
__t0[68] = 5275648u32;
__t0[69] = 3178496u32;
__t0[70] = 3178496u32;
__t0[71] = 3178496u32;
__t0[73] = 1081344u32;
__t0[75] = 1081344u32;
__t0[76] = 5275648u32;
__t0[77] = 5275648u32;
__t0[78] = 5275648u32;
__t0[79] = 5275648u32;
__t0[80] = 1081344u32;
__t0[81] = 1081344u32;
__t0[82] = 1081344u32;
__t0[83] = 1081344u32;
__t0[84] = 5275648u32;
__t0[85] = 5275648u32;
__t0[88] = 3178496u32;
__t0[89] = 3178496u32;
__t0[90] = 3178496u32;
__t0[91] = 5275648u32;
__t0[94] = 1081344u32;
__t0[98] = 5275648u32;
__t0[99] = 5275648u32;
__t0[100] = 5275648u32;
__t0[101] = 5275648u32;
__t0[102] = 5275648u32;
__t0[108] = 1081344u32;
__t0[112] = 5275648u32;
__t0[113] = 5275648u32;
__t0[114] = 5275648u32;
__t0[115] = 5275648u32;
__t0[117] = 5275648u32;
__t0[118] = 5275648u32;
__t0[119] = 5275648u32;
__t0[120] = 3178496u32;
__t0[121] = 3178496u32;
__t0[122] = 5275648u32;
__t0[123] = 5275648u32;
__t0[124] = 5275648u32;
__t0[125] = 5275648u32;
__t0[126] = 5275648u32;
__t0[127] = 5275648u32;
__t0[128] = 32768u32;
__t0[129] = 32768u32;
__t0[130] = 32768u32;
__t0[131] = 5275648u32;
__t0[136] = 5275648u32;
__t0[137] = 5275648u32;
__t0[138] = 5275648u32;
__t0[139] = 5275648u32;
__t0[140] = 3178496u32;
__t0[141] = 5275648u32;
__t0[142] = 3178496u32;
__t0[143] = 5275648u32;
__t0[144] = 3178496u32;
__t0[145] = 3178496u32;
__t0[146] = 3178496u32;
__t0[147] = 3178496u32;
__t0[150] = 3178496u32;
__t0[151] = 3178496u32;
__t0[152] = 3178496u32;
__t0[153] = 3178496u32;
__t0[154] = 3178496u32;
__t0[155] = 3178496u32;
__t0[156] = 3178496u32;
__t0[157] = 3178496u32;
__t0[158] = 3178496u32;
__t0[159] = 3178496u32;
__t0[160] = 5275648u32;
__t0[161] = 5275648u32;
__t0[162] = 5275648u32;
__t0[163] = 5275648u32;
__t0[166] = 3178496u32;
__t0[167] = 3178496u32;
__t0[168] = 3178496u32;
__t0[169] = 3178496u32;
__t0[170] = 3178496u32;
__t0[171] = 3178496u32;
__t0[172] = 3178496u32;
__t0[173] = 3178496u32;
__t0[174] = 3178496u32;
__t0[175] = 3178496u32;
__t0[176] = 3178496u32;
__t0[177] = 3178496u32;
__t0[180] = 1081344u32;
__t0[181] = 1081344u32;
__t0[182] = 3178496u32;
__t0[183] = 3178496u32;
__t0[184] = 3178496u32;
__t0[185] = 3178496u32;
__t0[186] = 3178496u32;
__t0[187] = 3178496u32;
__t0[188] = 3178496u32;
__t0[189] = 3178496u32;
__t0[190] = 3178496u32;
__t0[191] = 3178496u32;
__t0[196] = 5275648u32;
__t0[200] = 5275648u32;
__t0[202] = 5275648u32;
__t0[203] = 5275648u32;
__t0[204] = 5275648u32;
__t0[205] = 5275648u32;
__t0[207] = 1081344u32;
__t0[210] = 3178496u32;
__t0[211] = 3178496u32;
__t0[218] = 1081344u32;
__t0[219] = 1081344u32;
__t0[220] = 1081344u32;
__t0[221] = 1081344u32;
__t0[222] = 1081344u32;
__t0[223] = 1081344u32;
__t0[224] = 1081344u32;
__t0[225] = 1081344u32;
__t0[226] = 1081344u32;
__t0[227] = 1081344u32;
__t0[228] = 1081344u32;
__t0[229] = 1081344u32;
__t0[230] = 1081344u32;
__t0[231] = 1081344u32;
__t0[232] = 1081344u32;
__t0[233] = 1081344u32;
__t0[234] = 1081344u32;
__t0[235] = 1081344u32;
__t0[236] = 1081344u32;
__t0[237] = 1081344u32;
__t0[238] = 1081344u32;
__t0[239] = 1081344u32;
__t0[240] = 32768u32;
__t0[241] = 32768u32;
__t0[245] = 32768u32;
__t0[246] = 32768u32;
__t0[247] = 3178496u32;
__t0[248] = 32768u32;
__t0[252] = 32768u32;
__t0
});

#[unsafe(no_mangle)]
static inat_escape_table_2_2: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[16] = 5275648u32;
__t0[17] = 5275648u32;
__t0[18] = 5275648u32;
__t0[19] = 5275648u32;
__t0[20] = 5275648u32;
__t0[21] = 5275648u32;
__t0[32] = 5275648u32;
__t0[33] = 5275648u32;
__t0[34] = 5275648u32;
__t0[35] = 5275648u32;
__t0[36] = 5275648u32;
__t0[37] = 5275648u32;
__t0[38] = 5275648u32;
__t0[39] = 5275648u32;
__t0[40] = 5275648u32;
__t0[41] = 5275648u32;
__t0[42] = 5275648u32;
__t0[48] = 5275648u32;
__t0[49] = 5275648u32;
__t0[50] = 5275648u32;
__t0[51] = 5275648u32;
__t0[52] = 5275648u32;
__t0[53] = 5275648u32;
__t0[56] = 5275648u32;
__t0[57] = 5275648u32;
__t0[58] = 5275648u32;
__t0[75] = 3178496u32;
__t0[80] = 3178496u32;
__t0[81] = 3178496u32;
__t0[82] = 5275648u32;
__t0[92] = 1081344u32;
__t0[94] = 1081344u32;
__t0[114] = 1081344u32;
__t0[176] = 3178496u32;
__t0[177] = 3178496u32;
__t0[210] = 3178496u32;
__t0[211] = 3178496u32;
__t0[216] = 32768u32;
__t0[218] = 3178496u32;
__t0[220] = 32768u32;
__t0[221] = 32768u32;
__t0[222] = 32768u32;
__t0[223] = 32768u32;
__t0[245] = 3178496u32;
__t0[246] = 32768u32;
__t0[247] = 3178496u32;
__t0[248] = 32768u32;
__t0[250] = 32768u32;
__t0[251] = 32768u32;
__t0[252] = 32768u32;
__t0
});

#[unsafe(no_mangle)]
static inat_escape_table_2_3: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[73] = 1081344u32;
__t0[75] = 1081344u32;
__t0[80] = 3178496u32;
__t0[81] = 3178496u32;
__t0[82] = 5275648u32;
__t0[83] = 5275648u32;
__t0[92] = 1081344u32;
__t0[94] = 1081344u32;
__t0[104] = 5275648u32;
__t0[114] = 5275648u32;
__t0[154] = 5275648u32;
__t0[155] = 5275648u32;
__t0[170] = 5275648u32;
__t0[171] = 5275648u32;
__t0[176] = 3178496u32;
__t0[203] = 3178496u32;
__t0[204] = 3178496u32;
__t0[205] = 3178496u32;
__t0[218] = 3178496u32;
__t0[240] = 32768u32;
__t0[241] = 32768u32;
__t0[245] = 3178496u32;
__t0[246] = 3178496u32;
__t0[247] = 3178496u32;
__t0[248] = 32768u32;
__t0[252] = 32768u32;
__t0
});

#[unsafe(no_mangle)]
static inat_escape_table_3: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[0] = 524288u32;
__t0[1] = 524288u32;
__t0[2] = 524288u32;
__t0[3] = 524288u32;
__t0[4] = 524288u32;
__t0[5] = 524288u32;
__t0[6] = 524288u32;
__t0[8] = 1609728u32;
__t0[9] = 524288u32;
__t0[10] = 1609728u32;
__t0[11] = 524288u32;
__t0[12] = 524288u32;
__t0[13] = 524288u32;
__t0[14] = 524288u32;
__t0[15] = 561152u32;
__t0[20] = 524288u32;
__t0[21] = 524288u32;
__t0[22] = 524288u32;
__t0[23] = 524288u32;
__t0[24] = 524288u32;
__t0[25] = 524288u32;
__t0[26] = 524288u32;
__t0[27] = 524288u32;
__t0[29] = 524288u32;
__t0[30] = 524288u32;
__t0[31] = 524288u32;
__t0[32] = 524288u32;
__t0[33] = 524288u32;
__t0[34] = 524288u32;
__t0[35] = 524288u32;
__t0[37] = 524288u32;
__t0[38] = 5804032u32;
__t0[39] = 5804032u32;
__t0[48] = 524288u32;
__t0[49] = 524288u32;
__t0[50] = 524288u32;
__t0[51] = 524288u32;
__t0[56] = 524288u32;
__t0[57] = 524288u32;
__t0[58] = 524288u32;
__t0[59] = 524288u32;
__t0[62] = 524288u32;
__t0[63] = 524288u32;
__t0[64] = 524288u32;
__t0[65] = 524288u32;
__t0[66] = 524288u32;
__t0[67] = 524288u32;
__t0[68] = 524288u32;
__t0[70] = 524288u32;
__t0[74] = 524288u32;
__t0[75] = 524288u32;
__t0[76] = 524288u32;
__t0[80] = 524288u32;
__t0[81] = 524288u32;
__t0[84] = 524288u32;
__t0[85] = 524288u32;
__t0[86] = 5804032u32;
__t0[87] = 5804032u32;
__t0[96] = 524288u32;
__t0[97] = 524288u32;
__t0[98] = 524288u32;
__t0[99] = 524288u32;
__t0[102] = 5804032u32;
__t0[103] = 5804032u32;
__t0[112] = 524288u32;
__t0[113] = 524288u32;
__t0[114] = 524288u32;
__t0[115] = 524288u32;
__t0[194] = 5804032u32;
__t0[204] = 36864u32;
__t0[206] = 524288u32;
__t0[207] = 524288u32;
__t0[222] = 524288u32;
__t0[223] = 524288u32;
__t0[240] = 524288u32;
__t0
});

#[unsafe(no_mangle)]
static inat_escape_table_3_1: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[0] = 3182592u32;
__t0[1] = 3182592u32;
__t0[2] = 3182592u32;
__t0[3] = 5279744u32;
__t0[4] = 3182592u32;
__t0[5] = 3182592u32;
__t0[6] = 3182592u32;
__t0[8] = 1085440u32;
__t0[9] = 1085440u32;
__t0[10] = 1085440u32;
__t0[11] = 1085440u32;
__t0[12] = 1085440u32;
__t0[13] = 1085440u32;
__t0[14] = 1085440u32;
__t0[15] = 1085440u32;
__t0[20] = 1085440u32;
__t0[21] = 1085440u32;
__t0[22] = 1085440u32;
__t0[23] = 1085440u32;
__t0[24] = 3182592u32;
__t0[25] = 3182592u32;
__t0[26] = 5279744u32;
__t0[27] = 5279744u32;
__t0[29] = 3182592u32;
__t0[30] = 5279744u32;
__t0[31] = 5279744u32;
__t0[32] = 1085440u32;
__t0[33] = 1085440u32;
__t0[34] = 1085440u32;
__t0[35] = 5279744u32;
__t0[37] = 5279744u32;
__t0[38] = 5279744u32;
__t0[39] = 5279744u32;
__t0[48] = 3182592u32;
__t0[49] = 3182592u32;
__t0[50] = 3182592u32;
__t0[51] = 3182592u32;
__t0[56] = 3182592u32;
__t0[57] = 3182592u32;
__t0[58] = 5279744u32;
__t0[59] = 5279744u32;
__t0[62] = 5279744u32;
__t0[63] = 5279744u32;
__t0[64] = 1085440u32;
__t0[65] = 1085440u32;
__t0[66] = 1085440u32;
__t0[67] = 5279744u32;
__t0[68] = 1085440u32;
__t0[70] = 3182592u32;
__t0[74] = 3182592u32;
__t0[75] = 3182592u32;
__t0[76] = 1085440u32;
__t0[80] = 5279744u32;
__t0[81] = 5279744u32;
__t0[84] = 5279744u32;
__t0[85] = 5279744u32;
__t0[86] = 5279744u32;
__t0[87] = 5279744u32;
__t0[96] = 1085440u32;
__t0[97] = 1085440u32;
__t0[98] = 1085440u32;
__t0[99] = 1085440u32;
__t0[102] = 5279744u32;
__t0[103] = 5279744u32;
__t0[112] = 5279744u32;
__t0[113] = 5279744u32;
__t0[114] = 5279744u32;
__t0[115] = 5279744u32;
__t0[206] = 1085440u32;
__t0[207] = 1085440u32;
__t0[222] = 1085440u32;
__t0[223] = 1085440u32;
__t0
});

#[unsafe(no_mangle)]
static inat_escape_table_3_2: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[194] = 5279744u32;
__t0[240] = 36864u32;
__t0
});

#[unsafe(no_mangle)]
static inat_escape_table_3_3: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[240] = 3182592u32;
__t0
});

#[unsafe(no_mangle)]
static inat_avx_table_4: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[0] = 5275648u32;
__t0[1] = 39354368u32;
__t0[2] = 5275648u32;
__t0[3] = 39354368u32;
__t0[8] = 5275648u32;
__t0[9] = 39354368u32;
__t0[10] = 5275648u32;
__t0[11] = 39354368u32;
__t0[16] = 5275648u32;
__t0[17] = 39354368u32;
__t0[18] = 5275648u32;
__t0[19] = 39354368u32;
__t0[24] = 5275648u32;
__t0[25] = 39354368u32;
__t0[26] = 5275648u32;
__t0[27] = 39354368u32;
__t0[32] = 5275648u32;
__t0[33] = 39354368u32;
__t0[34] = 5275648u32;
__t0[35] = 39354368u32;
__t0[36] = 39358464u32;
__t0[40] = 5275648u32;
__t0[41] = 39354368u32;
__t0[42] = 5275648u32;
__t0[43] = 39354368u32;
__t0[44] = 39358464u32;
__t0[48] = 5275648u32;
__t0[49] = 39354368u32;
__t0[50] = 5275648u32;
__t0[51] = 39354368u32;
__t0[56] = 5275648u32;
__t0[57] = 39354368u32;
__t0[58] = 5275648u32;
__t0[59] = 39354368u32;
__t0[64] = 39354368u32;
__t0[65] = 39354368u32;
__t0[66] = 39354368u32;
__t0[67] = 39354368u32;
__t0[68] = 39354368u32;
__t0[69] = 39354368u32;
__t0[70] = 39354368u32;
__t0[71] = 39354368u32;
__t0[72] = 39354368u32;
__t0[73] = 39354368u32;
__t0[74] = 39354368u32;
__t0[75] = 39354368u32;
__t0[76] = 39354368u32;
__t0[77] = 39354368u32;
__t0[78] = 39354368u32;
__t0[79] = 39354368u32;
__t0[96] = 39354368u32;
__t0[97] = 39354368u32;
__t0[101] = 524288u32;
__t0[102] = 5799936u32;
__t0[105] = 39378944u32;
__t0[107] = 39358464u32;
__t0[128] = 5279872u32;
__t0[129] = 38854784u32;
__t0[131] = 38834304u32;
__t0[132] = 5275648u32;
__t0[133] = 39354368u32;
__t0[136] = 39354368u32;
__t0[143] = 5275648u32;
__t0[165] = 39354368u32;
__t0[173] = 39354368u32;
__t0[175] = 39354368u32;
__t0[192] = 5280128u32;
__t0[193] = 38834560u32;
__t0[208] = 5276032u32;
__t0[209] = 38830464u32;
__t0[210] = 5276032u32;
__t0[211] = 38830464u32;
__t0[240] = 39354368u32;
__t0[241] = 39354368u32;
__t0[242] = 524288u32;
__t0[244] = 39354368u32;
__t0[245] = 39354368u32;
__t0[246] = 5276416u32;
__t0[247] = 38830976u32;
__t0[248] = 524288u32;
__t0[249] = 5275648u32;
__t0[254] = 5276672u32;
__t0[255] = 38831232u32;
__t0
});

#[unsafe(no_mangle)]
static inat_avx_table_4_1: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[1] = 38830080u32;
__t0[3] = 38830080u32;
__t0[9] = 38830080u32;
__t0[11] = 38830080u32;
__t0[17] = 38830080u32;
__t0[19] = 38830080u32;
__t0[25] = 38830080u32;
__t0[27] = 38830080u32;
__t0[33] = 38830080u32;
__t0[35] = 38830080u32;
__t0[36] = 38834176u32;
__t0[41] = 38830080u32;
__t0[43] = 38830080u32;
__t0[44] = 38834176u32;
__t0[49] = 38830080u32;
__t0[51] = 38830080u32;
__t0[57] = 38830080u32;
__t0[59] = 38830080u32;
__t0[64] = 38830080u32;
__t0[65] = 38830080u32;
__t0[66] = 38830080u32;
__t0[67] = 38830080u32;
__t0[68] = 38830080u32;
__t0[69] = 38830080u32;
__t0[70] = 38830080u32;
__t0[71] = 38830080u32;
__t0[72] = 38830080u32;
__t0[73] = 38830080u32;
__t0[74] = 38830080u32;
__t0[75] = 38830080u32;
__t0[76] = 38830080u32;
__t0[77] = 38830080u32;
__t0[78] = 38830080u32;
__t0[79] = 38830080u32;
__t0[96] = 38830080u32;
__t0[97] = 38830080u32;
__t0[101] = 5275648u32;
__t0[102] = 5275648u32;
__t0[105] = 38854656u32;
__t0[107] = 38834176u32;
__t0[133] = 38830080u32;
__t0[136] = 38830080u32;
__t0[165] = 38830080u32;
__t0[173] = 38830080u32;
__t0[175] = 38830080u32;
__t0[241] = 38830080u32;
__t0[244] = 38830080u32;
__t0[245] = 38830080u32;
__t0[248] = 5275648u32;
__t0
});

#[unsafe(no_mangle)]
static inat_avx_table_4_2: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[102] = 5275648u32;
__t0[240] = 5275648u32;
__t0[241] = 5275648u32;
__t0[242] = 5275648u32;
__t0[248] = 5275648u32;
__t0
});

#[unsafe(no_mangle)]
static inat_avx_table_4_3: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[64] = 5275648u32;
__t0[65] = 5275648u32;
__t0[66] = 5275648u32;
__t0[67] = 5275648u32;
__t0[68] = 5275648u32;
__t0[69] = 5275648u32;
__t0[70] = 5275648u32;
__t0[71] = 5275648u32;
__t0[72] = 5275648u32;
__t0[73] = 5275648u32;
__t0[74] = 5275648u32;
__t0[75] = 5275648u32;
__t0[76] = 5275648u32;
__t0[77] = 5275648u32;
__t0[78] = 5275648u32;
__t0[79] = 5275648u32;
__t0[248] = 5275648u32;
__t0
});

#[unsafe(no_mangle)]
static inat_avx_table_5: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[16] = 524288u32;
__t0[17] = 524288u32;
__t0[29] = 5799936u32;
__t0[42] = 524288u32;
__t0[44] = 524288u32;
__t0[45] = 524288u32;
__t0[46] = 5275648u32;
__t0[47] = 5275648u32;
__t0[81] = 5799936u32;
__t0[88] = 5799936u32;
__t0[89] = 5799936u32;
__t0[90] = 5799936u32;
__t0[91] = 5799936u32;
__t0[92] = 5799936u32;
__t0[93] = 5799936u32;
__t0[94] = 5799936u32;
__t0[95] = 5799936u32;
__t0[110] = 524288u32;
__t0[120] = 5799936u32;
__t0[121] = 5799936u32;
__t0[122] = 524288u32;
__t0[123] = 524288u32;
__t0[124] = 5799936u32;
__t0[125] = 5799936u32;
__t0[126] = 524288u32;
__t0
});

#[unsafe(no_mangle)]
static inat_avx_table_5_1: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[29] = 5275648u32;
__t0[90] = 5275648u32;
__t0[91] = 5275648u32;
__t0[110] = 5275648u32;
__t0[120] = 5275648u32;
__t0[121] = 5275648u32;
__t0[122] = 5275648u32;
__t0[123] = 5275648u32;
__t0[124] = 5275648u32;
__t0[125] = 5275648u32;
__t0[126] = 5275648u32;
__t0
});

#[unsafe(no_mangle)]
static inat_avx_table_5_2: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[16] = 5275648u32;
__t0[17] = 5275648u32;
__t0[42] = 5275648u32;
__t0[44] = 5275648u32;
__t0[45] = 5275648u32;
__t0[81] = 5275648u32;
__t0[88] = 5275648u32;
__t0[89] = 5275648u32;
__t0[90] = 5275648u32;
__t0[91] = 5275648u32;
__t0[92] = 5275648u32;
__t0[93] = 5275648u32;
__t0[94] = 5275648u32;
__t0[95] = 5275648u32;
__t0[120] = 5275648u32;
__t0[121] = 5275648u32;
__t0[123] = 5275648u32;
__t0[125] = 5275648u32;
__t0
});

#[unsafe(no_mangle)]
static inat_avx_table_5_3: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[90] = 5275648u32;
__t0[122] = 5275648u32;
__t0[125] = 5275648u32;
__t0
});

#[unsafe(no_mangle)]
static inat_avx_table_6: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[19] = 5799936u32;
__t0[44] = 524288u32;
__t0[45] = 524288u32;
__t0[66] = 524288u32;
__t0[67] = 524288u32;
__t0[76] = 524288u32;
__t0[77] = 524288u32;
__t0[78] = 524288u32;
__t0[79] = 524288u32;
__t0[86] = 524288u32;
__t0[87] = 524288u32;
__t0[150] = 524288u32;
__t0[151] = 524288u32;
__t0[152] = 524288u32;
__t0[153] = 524288u32;
__t0[154] = 524288u32;
__t0[155] = 524288u32;
__t0[156] = 524288u32;
__t0[157] = 524288u32;
__t0[158] = 524288u32;
__t0[159] = 524288u32;
__t0[166] = 524288u32;
__t0[167] = 524288u32;
__t0[168] = 524288u32;
__t0[169] = 524288u32;
__t0[170] = 524288u32;
__t0[171] = 524288u32;
__t0[172] = 524288u32;
__t0[173] = 524288u32;
__t0[174] = 524288u32;
__t0[175] = 524288u32;
__t0[182] = 524288u32;
__t0[183] = 524288u32;
__t0[184] = 524288u32;
__t0[185] = 524288u32;
__t0[186] = 524288u32;
__t0[187] = 524288u32;
__t0[188] = 524288u32;
__t0[189] = 524288u32;
__t0[190] = 524288u32;
__t0[191] = 524288u32;
__t0[214] = 524288u32;
__t0[215] = 524288u32;
__t0
});

#[unsafe(no_mangle)]
static inat_avx_table_6_1: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[19] = 5275648u32;
__t0[44] = 5275648u32;
__t0[45] = 5275648u32;
__t0[66] = 5275648u32;
__t0[67] = 5275648u32;
__t0[76] = 5275648u32;
__t0[77] = 5275648u32;
__t0[78] = 5275648u32;
__t0[79] = 5275648u32;
__t0[150] = 5275648u32;
__t0[151] = 5275648u32;
__t0[152] = 5275648u32;
__t0[153] = 5275648u32;
__t0[154] = 5275648u32;
__t0[155] = 5275648u32;
__t0[156] = 5275648u32;
__t0[157] = 5275648u32;
__t0[158] = 5275648u32;
__t0[159] = 5275648u32;
__t0[166] = 5275648u32;
__t0[167] = 5275648u32;
__t0[168] = 5275648u32;
__t0[169] = 5275648u32;
__t0[170] = 5275648u32;
__t0[171] = 5275648u32;
__t0[172] = 5275648u32;
__t0[173] = 5275648u32;
__t0[174] = 5275648u32;
__t0[175] = 5275648u32;
__t0[182] = 5275648u32;
__t0[183] = 5275648u32;
__t0[184] = 5275648u32;
__t0[185] = 5275648u32;
__t0[186] = 5275648u32;
__t0[187] = 5275648u32;
__t0[188] = 5275648u32;
__t0[189] = 5275648u32;
__t0[190] = 5275648u32;
__t0[191] = 5275648u32;
__t0
});

#[unsafe(no_mangle)]
static inat_avx_table_6_2: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[86] = 5275648u32;
__t0[87] = 5275648u32;
__t0[214] = 5275648u32;
__t0[215] = 5275648u32;
__t0
});

#[unsafe(no_mangle)]
static inat_avx_table_6_3: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[86] = 5275648u32;
__t0[87] = 5275648u32;
__t0[214] = 5275648u32;
__t0[215] = 5275648u32;
__t0
});

#[unsafe(no_mangle)]
static inat_avx_table_7: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[248] = 524288u32;
__t0
});

#[unsafe(no_mangle)]
static inat_avx_table_7_2: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[248] = 1093632u32;
__t0
});

#[unsafe(no_mangle)]
static inat_avx_table_7_3: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[248] = 1093632u32;
__t0
});

#[unsafe(no_mangle)]
static inat_xop_table_0: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[133] = 1085440u32;
__t0[134] = 1085440u32;
__t0[135] = 1085440u32;
__t0[142] = 1085440u32;
__t0[143] = 1085440u32;
__t0[149] = 1085440u32;
__t0[150] = 1085440u32;
__t0[151] = 1085440u32;
__t0[158] = 1085440u32;
__t0[159] = 1085440u32;
__t0[162] = 1085440u32;
__t0[163] = 1085440u32;
__t0[166] = 1085440u32;
__t0[182] = 1085440u32;
__t0[192] = 1085440u32;
__t0[193] = 1085440u32;
__t0[194] = 1085440u32;
__t0[195] = 1085440u32;
__t0[204] = 1085440u32;
__t0[205] = 1085440u32;
__t0[206] = 1085440u32;
__t0[207] = 1085440u32;
__t0[236] = 1085440u32;
__t0[237] = 1085440u32;
__t0[238] = 1085440u32;
__t0[239] = 1085440u32;
__t0
});

#[unsafe(no_mangle)]
static inat_xop_table_1: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[1] = 1084928u32;
__t0[2] = 1085056u32;
__t0[18] = 1085184u32;
__t0[128] = 1081344u32;
__t0[129] = 1081344u32;
__t0[130] = 1081344u32;
__t0[131] = 1081344u32;
__t0[144] = 1081344u32;
__t0[145] = 1081344u32;
__t0[146] = 1081344u32;
__t0[147] = 1081344u32;
__t0[148] = 1081344u32;
__t0[149] = 1081344u32;
__t0[150] = 1081344u32;
__t0[151] = 1081344u32;
__t0[152] = 1081344u32;
__t0[153] = 1081344u32;
__t0[154] = 1081344u32;
__t0[155] = 1081344u32;
__t0[193] = 1081344u32;
__t0[194] = 1081344u32;
__t0[195] = 1081344u32;
__t0[198] = 1081344u32;
__t0[199] = 1081344u32;
__t0[203] = 1081344u32;
__t0[209] = 1081344u32;
__t0[210] = 1081344u32;
__t0[211] = 1081344u32;
__t0[214] = 1081344u32;
__t0[215] = 1081344u32;
__t0[219] = 1081344u32;
__t0[225] = 1081344u32;
__t0[226] = 1081344u32;
__t0[227] = 1081344u32;
__t0
});

#[unsafe(no_mangle)]
static inat_xop_table_2: __SlateAlign16<[u32; 256]> = __SlateAlign16({
let mut __t0: [u32; 256] = unsafe { core::mem::zeroed() };
__t0[16] = 1093632u32;
__t0[18] = 1085312u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_6: __SlateAlign16<[u32; 8]> = __SlateAlign16([36864u32, 36864u32, 32768u32, 32768u32, 32768u32, 32768u32, 32768u32, 32768u32]);

#[unsafe(no_mangle)]
static inat_group_table_7: __SlateAlign16<[u32; 8]> = __SlateAlign16([57344u32, 57344u32, 32768u32, 32768u32, 32768u32, 32768u32, 32768u32, 32768u32]);

#[unsafe(no_mangle)]
static inat_group_table_8: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[0] = 32768u32;
__t0[1] = 32768u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_9: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[0] = 32768u32;
__t0[1] = 32768u32;
__t0[2] = 98304u32;
__t0[3] = 32768u32;
__t0[4] = 98304u32;
__t0[5] = 32768u32;
__t0[6] = 98304u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_10: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[0] = 32768u32;
__t0[1] = 32768u32;
__t0[2] = 32768u32;
__t0[3] = 32768u32;
__t0[4] = 32768u32;
__t0[5] = 32768u32;
__t0[6] = 524288u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_10_3: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[6] = 32768u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_11: __SlateAlign16<[u32; 8]> = __SlateAlign16([32768u32, 32768u32, 32768u32, 32768u32, 32768u32, 524288u32, 32768u32, 32768u32]);

#[unsafe(no_mangle)]
static inat_group_table_11_2: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[5] = 32768u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_24: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[1] = 32768u32;
__t0[6] = 557056u32;
__t0[7] = 557056u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_24_1: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[6] = 32768u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_24_2: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[6] = 32768u32;
__t0[7] = 32768u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_4: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[0] = 36864u32;
__t0[7] = 4096u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_5: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[0] = 57344u32;
__t0[7] = 24576u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_16: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[2] = 561152u32;
__t0[4] = 561152u32;
__t0[6] = 561152u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_16_1: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[2] = 1085440u32;
__t0[4] = 1085440u32;
__t0[6] = 1085440u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_17: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[0] = 524288u32;
__t0[1] = 524288u32;
__t0[2] = 561152u32;
__t0[4] = 561152u32;
__t0[6] = 561152u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_17_1: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[0] = 5279744u32;
__t0[1] = 5279744u32;
__t0[2] = 1085440u32;
__t0[4] = 1085440u32;
__t0[6] = 1085440u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_18: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[2] = 561152u32;
__t0[3] = 524288u32;
__t0[6] = 561152u32;
__t0[7] = 524288u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_18_1: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[2] = 1085440u32;
__t0[3] = 1085440u32;
__t0[6] = 1085440u32;
__t0[7] = 1085440u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_21: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[0] = 524288u32;
__t0[1] = 524288u32;
__t0[2] = 1605632u32;
__t0[3] = 1605632u32;
__t0[4] = 524288u32;
__t0[5] = 524288u32;
__t0[6] = 524288u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_21_1: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[6] = 32768u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_21_2: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[0] = 32768u32;
__t0[1] = 32768u32;
__t0[2] = 32768u32;
__t0[3] = 32768u32;
__t0[4] = 32768u32;
__t0[5] = 32768u32;
__t0[6] = 32768u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_21_3: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[6] = 32768u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_13: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[0] = 32768u32;
__t0[1] = 32768u32;
__t0[2] = 32768u32;
__t0[3] = 32768u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_27: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[1] = 3178496u32;
__t0[2] = 3178496u32;
__t0[3] = 3178496u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_25: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[1] = 524288u32;
__t0[2] = 524288u32;
__t0[5] = 524288u32;
__t0[6] = 524288u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_25_1: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[1] = 5275648u32;
__t0[2] = 5275648u32;
__t0[5] = 5275648u32;
__t0[6] = 5275648u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_26: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[1] = 524288u32;
__t0[2] = 524288u32;
__t0[5] = 524288u32;
__t0[6] = 524288u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_26_1: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[1] = 5275648u32;
__t0[2] = 5275648u32;
__t0[5] = 5275648u32;
__t0[6] = 5275648u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_14: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[0] = 32768u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_15: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[1] = 524288u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_15_2: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[1] = 32768u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_28: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[1] = 1081344u32;
__t0[2] = 1081344u32;
__t0[3] = 1081344u32;
__t0[4] = 1081344u32;
__t0[5] = 1081344u32;
__t0[6] = 1081344u32;
__t0[7] = 1081344u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_29: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[1] = 1081344u32;
__t0[6] = 1081344u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_30: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[0] = 1081344u32;
__t0[1] = 1081344u32;
__t0
});

#[unsafe(no_mangle)]
static inat_group_table_31: __SlateAlign16<[u32; 8]> = __SlateAlign16({
let mut __t0: [u32; 8] = unsafe { core::mem::zeroed() };
__t0[0] = 1093632u32;
__t0[1] = 1093632u32;
__t0
});

#[unsafe(no_mangle)]
static inat_escape_tables: __SlateAlign16<[[*const u32; 4]; 4]> = __SlateAlign16({
let mut __t0: [[*const u32; 4]; 4] = unsafe { core::mem::zeroed() };
__t0[1] = [unsafe { core::ptr::addr_of!(inat_escape_table_1.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_1_1.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_1_2.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_1_3.0) as *const u32 }];
__t0[2] = [unsafe { core::ptr::addr_of!(inat_escape_table_2.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_2_1.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_2_2.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_2_3.0) as *const u32 }];
__t0[3] = [unsafe { core::ptr::addr_of!(inat_escape_table_3.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_3_1.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_3_2.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_3_3.0) as *const u32 }];
__t0
});

#[unsafe(no_mangle)]
static inat_group_tables: __SlateAlign16<[[*const u32; 4]; 32]> = __SlateAlign16({
let mut __t0: [[*const u32; 4]; 32] = unsafe { core::mem::zeroed() };
__t0[4] = {
let mut __t1: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t1[0] = unsafe { core::ptr::addr_of!(inat_group_table_4.0) as *const u32 };
__t1
};
__t0[5] = {
let mut __t2: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t2[0] = unsafe { core::ptr::addr_of!(inat_group_table_5.0) as *const u32 };
__t2
};
__t0[6] = {
let mut __t3: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t3[0] = unsafe { core::ptr::addr_of!(inat_group_table_6.0) as *const u32 };
__t3
};
__t0[7] = {
let mut __t4: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t4[0] = unsafe { core::ptr::addr_of!(inat_group_table_7.0) as *const u32 };
__t4
};
__t0[8] = {
let mut __t5: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t5[0] = unsafe { core::ptr::addr_of!(inat_group_table_8.0) as *const u32 };
__t5
};
__t0[9] = {
let mut __t6: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t6[0] = unsafe { core::ptr::addr_of!(inat_group_table_9.0) as *const u32 };
__t6
};
__t0[10] = {
let mut __t7: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t7[0] = unsafe { core::ptr::addr_of!(inat_group_table_10.0) as *const u32 };
__t7[3] = unsafe { core::ptr::addr_of!(inat_group_table_10_3.0) as *const u32 };
__t7
};
__t0[11] = {
let mut __t8: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t8[0] = unsafe { core::ptr::addr_of!(inat_group_table_11.0) as *const u32 };
__t8[2] = unsafe { core::ptr::addr_of!(inat_group_table_11_2.0) as *const u32 };
__t8
};
__t0[13] = {
let mut __t9: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t9[0] = unsafe { core::ptr::addr_of!(inat_group_table_13.0) as *const u32 };
__t9
};
__t0[14] = {
let mut __t10: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t10[0] = unsafe { core::ptr::addr_of!(inat_group_table_14.0) as *const u32 };
__t10
};
__t0[15] = {
let mut __t11: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t11[0] = unsafe { core::ptr::addr_of!(inat_group_table_15.0) as *const u32 };
__t11[2] = unsafe { core::ptr::addr_of!(inat_group_table_15_2.0) as *const u32 };
__t11
};
__t0[16] = {
let mut __t12: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t12[0] = unsafe { core::ptr::addr_of!(inat_group_table_16.0) as *const u32 };
__t12[1] = unsafe { core::ptr::addr_of!(inat_group_table_16_1.0) as *const u32 };
__t12
};
__t0[17] = {
let mut __t13: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t13[0] = unsafe { core::ptr::addr_of!(inat_group_table_17.0) as *const u32 };
__t13[1] = unsafe { core::ptr::addr_of!(inat_group_table_17_1.0) as *const u32 };
__t13
};
__t0[18] = {
let mut __t14: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t14[0] = unsafe { core::ptr::addr_of!(inat_group_table_18.0) as *const u32 };
__t14[1] = unsafe { core::ptr::addr_of!(inat_group_table_18_1.0) as *const u32 };
__t14
};
__t0[21] = [unsafe { core::ptr::addr_of!(inat_group_table_21.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_group_table_21_1.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_group_table_21_2.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_group_table_21_3.0) as *const u32 }];
__t0[24] = {
let mut __t15: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t15[0] = unsafe { core::ptr::addr_of!(inat_group_table_24.0) as *const u32 };
__t15[1] = unsafe { core::ptr::addr_of!(inat_group_table_24_1.0) as *const u32 };
__t15[2] = unsafe { core::ptr::addr_of!(inat_group_table_24_2.0) as *const u32 };
__t15
};
__t0[25] = {
let mut __t16: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t16[0] = unsafe { core::ptr::addr_of!(inat_group_table_25.0) as *const u32 };
__t16[1] = unsafe { core::ptr::addr_of!(inat_group_table_25_1.0) as *const u32 };
__t16
};
__t0[26] = {
let mut __t17: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t17[0] = unsafe { core::ptr::addr_of!(inat_group_table_26.0) as *const u32 };
__t17[1] = unsafe { core::ptr::addr_of!(inat_group_table_26_1.0) as *const u32 };
__t17
};
__t0[27] = {
let mut __t18: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t18[0] = unsafe { core::ptr::addr_of!(inat_group_table_27.0) as *const u32 };
__t18
};
__t0[28] = {
let mut __t19: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t19[0] = unsafe { core::ptr::addr_of!(inat_group_table_28.0) as *const u32 };
__t19
};
__t0[29] = {
let mut __t20: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t20[0] = unsafe { core::ptr::addr_of!(inat_group_table_29.0) as *const u32 };
__t20
};
__t0[30] = {
let mut __t21: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t21[0] = unsafe { core::ptr::addr_of!(inat_group_table_30.0) as *const u32 };
__t21
};
__t0[31] = {
let mut __t22: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t22[0] = unsafe { core::ptr::addr_of!(inat_group_table_31.0) as *const u32 };
__t22
};
__t0
});

#[unsafe(no_mangle)]
static inat_avx_tables: __SlateAlign16<[[*const u32; 4]; 32]> = __SlateAlign16({
let mut __t0: [[*const u32; 4]; 32] = unsafe { core::mem::zeroed() };
__t0[1] = [unsafe { core::ptr::addr_of!(inat_escape_table_1.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_1_1.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_1_2.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_1_3.0) as *const u32 }];
__t0[2] = [unsafe { core::ptr::addr_of!(inat_escape_table_2.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_2_1.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_2_2.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_2_3.0) as *const u32 }];
__t0[3] = [unsafe { core::ptr::addr_of!(inat_escape_table_3.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_3_1.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_3_2.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_escape_table_3_3.0) as *const u32 }];
__t0[4] = [unsafe { core::ptr::addr_of!(inat_avx_table_4.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_avx_table_4_1.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_avx_table_4_2.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_avx_table_4_3.0) as *const u32 }];
__t0[5] = [unsafe { core::ptr::addr_of!(inat_avx_table_5.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_avx_table_5_1.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_avx_table_5_2.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_avx_table_5_3.0) as *const u32 }];
__t0[6] = [unsafe { core::ptr::addr_of!(inat_avx_table_6.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_avx_table_6_1.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_avx_table_6_2.0) as *const u32 }, unsafe { core::ptr::addr_of!(inat_avx_table_6_3.0) as *const u32 }];
__t0[7] = {
let mut __t1: [*const u32; 4] = unsafe { core::mem::zeroed() };
__t1[0] = unsafe { core::ptr::addr_of!(inat_avx_table_7.0) as *const u32 };
__t1[2] = unsafe { core::ptr::addr_of!(inat_avx_table_7_2.0) as *const u32 };
__t1[3] = unsafe { core::ptr::addr_of!(inat_avx_table_7_3.0) as *const u32 };
__t1
};
__t0
});

#[unsafe(no_mangle)]
static inat_xop_tables: __SlateAlign16<[*const u32; 24]> = __SlateAlign16({
let mut __t0: [*const u32; 24] = unsafe { core::mem::zeroed() };
__t0[0] = unsafe { core::ptr::addr_of!(inat_xop_table_0.0) as *const u32 };
__t0[1] = unsafe { core::ptr::addr_of!(inat_xop_table_1.0) as *const u32 };
__t0[2] = unsafe { core::ptr::addr_of!(inat_xop_table_2.0) as *const u32 };
__t0
});

// __ignore_sync_check__
// Attribute tables are generated from opcode map
/// Attribute search APIs
#[unsafe(no_mangle)]
extern "C" fn inat_get_opcode_attribute(opcode: u8) -> u32 {
unsafe { inat_primary_table.0[(opcode as i32) as usize] }
}

#[unsafe(no_mangle)]
extern "C" fn inat_get_last_prefix_id(last_pfx: u8) -> i32 {
let lpfx_attr: u32 = inat_get_opcode_attribute(last_pfx);
inat_last_prefix_id(lpfx_attr)
}

#[unsafe(no_mangle)]
extern "C" fn inat_get_escape_attribute(opcode: u8, lpfx_id: i32, esc_attr: u32) -> u32 {
let mut table: *const u32 = unsafe { core::mem::zeroed() };
let n: i32 = inat_escape_id(esc_attr);
table = unsafe { inat_escape_tables.0[n as usize][0] };
if table.is_null() {
return 0;
}
if inat_has_variant(unsafe { *table.wrapping_offset((opcode as i32) as isize) }) != 0 && lpfx_id != 0 {
table = unsafe { inat_escape_tables.0[n as usize][lpfx_id as usize] };
if table.is_null() {
return 0;
}
}
unsafe { *table.wrapping_offset((opcode as i32) as isize) }
}

#[unsafe(no_mangle)]
extern "C" fn inat_get_group_attribute(modrm: u8, lpfx_id: i32, grp_attr: u32) -> u32 {
let mut table: *const u32 = unsafe { core::mem::zeroed() };
let n: i32 = inat_group_id(grp_attr);
table = unsafe { inat_group_tables.0[n as usize][0] };
if table.is_null() {
return inat_group_common_attribute(grp_attr) as u32;
}
if inat_has_variant(unsafe { *table.wrapping_offset(X86_MODRM_REG!(modrm) as isize) }) != 0 && lpfx_id != 0 {
table = unsafe { inat_group_tables.0[n as usize][lpfx_id as usize] };
if table.is_null() {
return inat_group_common_attribute(grp_attr) as u32;
}
}
(unsafe { *table.wrapping_offset(X86_MODRM_REG!(modrm) as isize) }) | (inat_group_common_attribute(grp_attr) as u32)
}

#[unsafe(no_mangle)]
extern "C" fn inat_get_avx_attribute(opcode: u8, vex_m: u8, vex_p: u8) -> u32 {
let mut table: *const u32 = unsafe { core::mem::zeroed() };
if (vex_m as i32) > 31 || (vex_p as i32) > 3 {
return 0;
}
// At first, this checks the master table
table = unsafe { inat_avx_tables.0[(vex_m as i32) as usize][0] };
if table.is_null() {
return 0;
}
if inat_is_group(unsafe { *table.wrapping_offset((opcode as i32) as isize) }) == 0 && vex_p != 0 {
// If this is not a group, get attribute directly
table = unsafe { inat_avx_tables.0[(vex_m as i32) as usize][(vex_p as i32) as usize] };
if table.is_null() {
return 0;
}
}
unsafe { *table.wrapping_offset((opcode as i32) as isize) }
}

#[unsafe(no_mangle)]
extern "C" fn inat_get_xop_attribute(opcode: u8, mut map_select: u8) -> u32 {
if (map_select as i32) < 8 || (map_select as i32) > 31 {
return 0;
}
map_select = (map_select as i32).wrapping_sub(8) as u8;
// At first, this checks the master table
let table: *const u32 = unsafe { inat_xop_tables.0[(map_select as i32) as usize] };
if table.is_null() {
return 0;
}
unsafe { *table.wrapping_offset((opcode as i32) as isize) }
}
