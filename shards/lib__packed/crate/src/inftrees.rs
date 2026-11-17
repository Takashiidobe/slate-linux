
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! inftrees.c -- generate Huffman trees for efficient decoding
//! Copyright (C) 1995-2005 Mark Adler
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
/// Optional: only supported since gcc >= 15, clang >= 19
///
///   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
/// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
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
#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

/// Build a set of tables to decode the provided canonical Huffman code.
/// The code lengths are lens[0..codes-1].  The result starts at *table,
/// whose indices are 0..2^bits-1.  work is a writable array of at least
/// lens shorts, which is used as a work area.  type is the type of code
/// to be generated, CODES, LENS, or DISTS.  On return, zero is success,
/// -1 is an invalid code, and +1 means that ENOUGH isn't enough.  table
/// on return points to the next available entry's address.  bits is the
/// requested root table index bits, and on return it is the actual root
/// table index bits.  It will differ if the request is greater than the
/// longest code or if it is less than the shortest code.
#[unsafe(no_mangle)]
extern "C" fn zlib_inflate_table(r#type: u32, lens: *mut u16, codes: u32, table: *mut *mut __SlateRecord_inftrees_h_31, bits: *mut u32, work: *mut u16) -> i32 {
// accumulate lengths for codes (assumes lens[] all in 0..MAXBITS)
let mut len: u32 = 0; // a code's length in bits
let mut sym: u32 = 0; // index of code symbols
let mut min: u32 = 1;
let mut max: u32 = 15; // minimum and maximum code lengths
let mut root: u32 = 0; // number of index bits for root table
let mut curr: u32 = 0; // number of index bits for current table
// current bits to drop from code for index
let mut drop: u32 = 0; // code bits to drop for sub-table
// check for an over-subscribed or incomplete set of lengths
let mut left: i32 = 1; // number of prefix codes available
let mut used: u32 = 0; // code entries in table used
// initialize state for loop
// starting code
let mut huff: u32 = 0; // Huffman code
let mut incr: u32 = 0; // for incrementing code, index
let mut fill: u32 = 0; // index for replicating entries
let mut low: u32 = 0; // low bits for current root entry
let mut mask: u32 = 0; // mask for low root bits
let mut this: __SlateRecord_inftrees_h_31 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(this) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
// table entry for duplication
let mut next: *mut __SlateRecord_inftrees_h_31 = unsafe { core::mem::zeroed() }; // next available space in table
let mut base: *const u16 = unsafe { core::mem::zeroed() }; // base value table to use
let mut extra: *const u16 = unsafe { core::mem::zeroed() }; // extra bits table to use
let mut end: i32 = 0; // use base and extra for symbol > end
let mut count: __SlateAlign16<[u16; 16]> = __SlateAlign16([0u16; 16]); // number of codes of each length
let mut offs: __SlateAlign16<[u16; 16]> = __SlateAlign16([0u16; 16]); // offsets in table for each length
// Length codes 257..285 base
// Length codes 257..285 extra
// Distance codes 0..29 base
// Distance codes 0..29 extra
// Process a set of code lengths to create a canonical Huffman code.  The
// code lengths are lens[0..codes-1].  Each length corresponds to the
// symbols 0..codes-1.  The Huffman code is generated by first sorting the
// symbols by length from short to long, and retaining the symbol order
// for codes with equal lengths.  Then the code starts with all zero bits
// for the first code of the shortest length, and the codes are integer
// increments for the same length, and zeros are appended as the length
// increases.  For the deflate format, these bits are stored backwards
// from their more natural integer increment ordering, and so when the
// decoding tables are built in the large loop below, the integer codes
// are incremented backwards.
//
// This routine assumes, but does not check, that all of the entries in
// lens[] are in the range 0..MAXBITS.  The caller must assure this.
// 1..MAXBITS is interpreted as that code length.  zero means that that
// symbol does not occur in this code.
//
// The codes are sorted by computing a count of codes for each length,
// creating from that a table of starting indices for each length in the
// sorted table, and then entering the symbols in order in the sorted
// table.  The sorted table is work[], with that space being provided by
// the caller.
//
// The length counts are used for other purposes as well, i.e. finding
// the minimum and maximum length codes, determining if there are any
// codes at all, checking for a valid set of lengths, and looking ahead
// at length counts to determine sub-table sizes when building the
// decoding tables.
while len <= 15 {
count.0[len as usize] = 0;
len = len.wrapping_add(1);
}
while sym < codes {
let __v3: *mut u16 = (count.0.as_mut_ptr() as *mut u16).wrapping_offset(((unsafe { *lens.wrapping_offset(sym as isize) }) as i32) as isize);
unsafe {
*__v3 = (*__v3 as i32).wrapping_add(1) as u16;
}
sym = sym.wrapping_add(1);
}
// bound code lengths, force root to be within code lengths
root = unsafe { *bits };
while max >= 1 {
if (count.0[max as usize] as i32) != 0 {
break;
}
max = max.wrapping_sub(1);
}
if root > max {
root = max;
}
if max == 0 {
// no symbols to code at all
this.op = 64; // invalid code marker
this.bits = 1;
this.val = 0;
let __v13: *mut __SlateRecord_inftrees_h_31 = unsafe { *table };
unsafe {
*table = __v13.wrapping_offset(1 as isize);
}
unsafe {
*__v13 = this;
}
// make a table to force an error
let __v16: *mut __SlateRecord_inftrees_h_31 = unsafe { *table };
unsafe {
*table = __v16.wrapping_offset(1 as isize);
}
unsafe {
*__v16 = this;
}
unsafe {
*bits = 1;
}
return 0; // no symbols, but wait for decoding to report error
}
while min < 15 {
if (count.0[min as usize] as i32) != 0 {
break;
}
min = min.wrapping_add(1);
}
if root < min {
root = min;
}
len = 1;
while len <= 15 {
left <<= 1i32;
left = left.wrapping_sub(count.0[len as usize] as i32);
if left < 0 {
return 1i32.wrapping_neg();
}
// over-subscribed
len = len.wrapping_add(1);
}
if left > 0 && ((r#type as u32) == 0 || max != 1) {
return 1i32.wrapping_neg();
}
// incomplete set
// generate offsets into symbol table for each length for sorting
offs.0[1] = 0;
len = 1;
while len < 15 {
offs.0[len.wrapping_add(1) as usize] = (offs.0[len as usize] as i32).wrapping_add(count.0[len as usize] as i32) as u16;
len = len.wrapping_add(1);
}
// sort symbols by length, by symbol order within each length
sym = 0;
while sym < codes {
if ((unsafe { *lens.wrapping_offset(sym as isize) }) as i32) != 0 {
let __v31: *mut u16 = (offs.0.as_mut_ptr() as *mut u16).wrapping_offset(((unsafe { *lens.wrapping_offset(sym as isize) }) as i32) as isize);
let __v32: u16 = unsafe { *__v31 };
unsafe {
*__v31 = (__v32 as i32).wrapping_add(1) as u16;
}
unsafe {
*work.wrapping_offset((__v32 as i32) as isize) = sym as u16;
}
}
sym = sym.wrapping_add(1);
}
// Create and fill in decoding tables.  In this loop, the table being
// filled is at next and has curr index bits.  The code being used is huff
// with length len.  That code is converted to an index by dropping drop
// bits off of the bottom.  For codes where len is less than drop + curr,
// those top drop + curr - len bits are incremented through all values to
// fill the table with replicated entries.
//
// root is the number of index bits for the root table.  When len exceeds
// root, sub-tables are created pointed to by the root entry with an index
// of the low root bits of huff.  This is saved in low to check for when a
// new sub-table should be started.  drop is zero when the root table is
// being filled, and drop is root when sub-tables are being filled.
//
// When a new sub-table is needed, it is necessary to look ahead in the
// code lengths to determine what size sub-table is needed.  The length
// counts are used for this, and so count[] is decremented as codes are
// entered in the tables.
//
// used keeps track of how many table entries have been allocated from the
// provided *table space.  It is checked when a LENS table is being made
// against the space in *table, ENOUGH, minus the maximum space needed by
// the worst case distance code, MAXD.  This should never happen, but the
// sufficiency of ENOUGH has not been proven exhaustively, hence the check.
// This assumes that when type == LENS, bits == 9.
//
// sym increments through all symbols, and the loop terminates when
// all codes of length max, i.e. all codes, have been processed.  This
// routine permits incomplete codes, so another loop after this one fills
// in the rest of the decoding tables with invalid code markers.
// set up for code type
match r#type as u32 {
0 => {
let __v37: *const u16 = work as *const u16;
extra = __v37;
base = __v37; // dummy value--not used
end = 19;
}
1 => {
base = unsafe { core::ptr::addr_of!(lbase.0) as *const u16 };
let __v39: *const u16 = base;
base = __v39.wrapping_offset(-(257 as isize));
extra = unsafe { core::ptr::addr_of!(lext.0) as *const u16 };
let __v41: *const u16 = extra;
extra = __v41.wrapping_offset(-(257 as isize));
end = 256;
}
_ => {
base = unsafe { core::ptr::addr_of!(dbase.0) as *const u16 }; // DISTS
extra = unsafe { core::ptr::addr_of!(dext.0) as *const u16 };
end = 1i32.wrapping_neg();
}
}
sym = 0; // starting code symbol
len = min; // starting code length
next = unsafe { *table }; // current table to fill in
curr = root; // current table index bits
low = 1i32.wrapping_neg() as u32; // trigger new sub-table when len > root
used = 1u32 << root; // use root table entries
mask = used.wrapping_sub(1); // mask for comparing low
// check available table space
if (r#type as u32) == 1 && used >= 1456u32 {
return 1;
}
// process all codes and make table entries
loop {
// create table entry
this.bits = len.wrapping_sub(drop) as u8;
if ((unsafe { *work.wrapping_offset(sym as isize) }) as i32) < end {
this.op = 0;
this.val = unsafe { *work.wrapping_offset(sym as isize) };
} else {
if ((unsafe { *work.wrapping_offset(sym as isize) }) as i32) > end {
this.op = (unsafe { *extra.wrapping_offset((*work.wrapping_offset(sym as isize) as i32) as isize) }) as u8;
this.val = unsafe { *base.wrapping_offset((*work.wrapping_offset(sym as isize) as i32) as isize) };
} else {
this.op = 96u8; // end of block
this.val = 0;
}
}
// replicate for those indices with low len bits equal to huff
incr = 1u32 << len.wrapping_sub(drop);
fill = 1u32 << curr;
min = fill; // save offset to next table
loop {
fill = fill.wrapping_sub(incr);
unsafe {
*next.wrapping_offset((huff >> drop).wrapping_add(fill) as isize) = this;
}
if fill == 0 {
break;
}
}
// backwards increment the len-bit code huff
incr = 1u32 << len.wrapping_sub(1);
while huff & incr != 0 {
incr >>= 1i32;
}
if incr != 0 {
huff &= incr.wrapping_sub(1);
huff = huff.wrapping_add(incr);
} else {
huff = 0;
}
// go to next symbol, update count, len
sym = sym.wrapping_add(1);
let __v55: *mut u16 = (count.0.as_mut_ptr() as *mut u16).wrapping_offset(len as isize);
let __v57: u16 = ((unsafe { *__v55 }) as i32).wrapping_sub(1) as u16;
unsafe {
*__v55 = __v57;
}
if (__v57 as i32) == 0 {
if len == max {
break;
}
len = (unsafe { *lens.wrapping_offset((*work.wrapping_offset(sym as isize) as i32) as isize) }) as u32;
}
// create new sub-table if needed
if len > root && huff & mask != low {
// if first time, transition to sub-tables
if drop == 0 {
drop = root;
}
// increment past last table
let __v59: *mut __SlateRecord_inftrees_h_31 = next;
next = __v59.wrapping_offset(min as isize); // here min is 1 << curr
// determine length of next table
curr = len.wrapping_sub(drop);
left = 1i32 << curr;
while curr.wrapping_add(drop) < max {
left = left.wrapping_sub(count.0[curr.wrapping_add(drop) as usize] as i32);
if left <= 0 {
break;
}
curr = curr.wrapping_add(1);
left <<= 1i32;
}
// check for enough space
used = used.wrapping_add(1u32 << curr);
if (r#type as u32) == 1 && used >= 1456u32 {
return 1;
}
// point entry in root table to sub-table
low = huff & mask;
unsafe {
(*unsafe { *table }.wrapping_offset(low as isize)).op = curr as u8;
}
unsafe {
(*unsafe { *table }.wrapping_offset(low as isize)).bits = root as u8;
}
unsafe {
(*unsafe { *table }.wrapping_offset(low as isize)).val = (next.offset_from(*table as *mut __SlateRecord_inftrees_h_31) as i64) as u16;
}
}
}
// Fill in rest of table for incomplete codes.  This loop is similar to the
// loop above in incrementing huff for table indices.  It is assumed that
// len is equal to curr + drop, so there is no loop needed to increment
// through high index bits.  When the current sub-table is filled, the loop
// drops back to the root table to fill in any remaining entries there.
this.op = 64; // invalid code marker
this.bits = len.wrapping_sub(drop) as u8;
this.val = 0;
while huff != 0 {
// when done with sub-table, drop back to root table
if drop != 0 && huff & mask != low {
drop = 0;
len = root;
next = unsafe { *table };
this.bits = len as u8;
}
// put invalid code marker in table
unsafe {
*next.wrapping_offset((huff >> drop) as isize) = this;
}
// backwards increment the len-bit code huff
incr = 1u32 << len.wrapping_sub(1);
while huff & incr != 0 {
incr >>= 1i32;
}
if incr != 0 {
huff &= incr.wrapping_sub(1);
huff = huff.wrapping_add(incr);
} else {
huff = 0;
}
}
// set return parameters
let __v79: *mut __SlateRecord_inftrees_h_31 = unsafe { *table };
unsafe {
*table = __v79.wrapping_offset(used as isize);
}
unsafe {
*bits = root;
}
0
}

static lbase: __SlateAlign16<[u16; 31]> = __SlateAlign16([3u16, 4u16, 5u16, 6u16, 7u16, 8u16, 9u16, 10u16, 11u16, 13u16, 15u16, 17u16, 19u16, 23u16, 27u16, 31u16, 35u16, 43u16, 51u16, 59u16, 67u16, 83u16, 99u16, 115u16, 131u16, 163u16, 195u16, 227u16, 258u16, 0u16, 0u16]);

static lext: __SlateAlign16<[u16; 31]> = __SlateAlign16([16u16, 16u16, 16u16, 16u16, 16u16, 16u16, 16u16, 16u16, 17u16, 17u16, 17u16, 17u16, 18u16, 18u16, 18u16, 18u16, 19u16, 19u16, 19u16, 19u16, 20u16, 20u16, 20u16, 20u16, 21u16, 21u16, 21u16, 21u16, 16u16, 201u16, 196u16]);

static dbase: __SlateAlign16<[u16; 32]> = __SlateAlign16([1u16, 2u16, 3u16, 4u16, 5u16, 7u16, 9u16, 13u16, 17u16, 25u16, 33u16, 49u16, 65u16, 97u16, 129u16, 193u16, 257u16, 385u16, 513u16, 769u16, 1025u16, 1537u16, 2049u16, 3073u16, 4097u16, 6145u16, 8193u16, 12289u16, 16385u16, 24577u16, 0u16, 0u16]);

static dext: __SlateAlign16<[u16; 32]> = __SlateAlign16([16u16, 16u16, 16u16, 16u16, 17u16, 17u16, 18u16, 18u16, 19u16, 19u16, 20u16, 20u16, 21u16, 21u16, 22u16, 22u16, 23u16, 23u16, 24u16, 24u16, 25u16, 25u16, 26u16, 26u16, 27u16, 27u16, 28u16, 28u16, 29u16, 29u16, 64u16, 64u16]);
