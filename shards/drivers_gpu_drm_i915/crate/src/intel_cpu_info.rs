
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! __context_unsafe() - function attribute, disable context checking
//! @comment: comment explaining why opt-out is safe
//!
//! Function attribute denoting that context analysis is disabled for the
//! whole function. Forces adding an inline comment as argument.
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: MIT
//!
//! Copyright © 2024 Intel Corporation
//!
//! Avoid INTEL_<PLATFORM> name collisions between asm/intel-family.h and
//! intel_device_info.h by having a separate file.
//! small_const_nbits(n) is true precisely when it is known at compile-time
//! that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
//! various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
//! of size 0 are very rare, and a compile-time-known-size 0 is most likely
//! a sign of error. They will be handled correctly by the bit/bitmap APIs,
//! but using the out-of-line functions, so that the inline implementations
//! can unconditionally dereference the pointer(s).
//! Sky Lake
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
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! The __sme_pa() and __sme_pa_nodebug() macros are meant for use when
//! writing to or comparing values from the cr3 register.  Having the
//! encryption mask set in cr3 enables the PGD entry to be encrypted and
//! avoid special case handling of PGD allocations.
//! CR3's layout varies depending on several things.
//!
//! If CR4.PCIDE is set (64-bit only), then CR3[11:0] is the address space ID.
//! If PAE is enabled, then CR3[11:5] is part of the PDPT address
//! (i.e. it's 32-byte aligned, not page-aligned) and CR3[4:0] is ignored.
//! Otherwise (non-PAE, non-PCID), CR3[3] is PWT, CR3[4] is PCD, and
//! CR3[2:0] and CR3[11:5] are ignored.
//!
//! In all cases, Linux puts zeros in the low ignored bits and in PWT and PCD.
//!
//! CR3[63] is always read as zero.  If CR4.PCIDE is set, then CR3[63] may be
//! written as 1 to prevent the write to CR3 from flushing the TLB.
//!
//! On systems with SME, one bit (in a variable position!) is stolen to indicate
//! that the top-level paging structure is encrypted.
//!
//! On systemms with LAM, bits 61 and 62 are used to indicate LAM mode.
//!
//! All of the remaining bits indicate the physical address of the top-level
//! paging structure.
//!
//! CR3_ADDR_MASK is the mask used by read_cr3_pa().
//!
//! Mask off the address space ID and SME encryption bits.
//! Begin/end of an instrumentation safe region
//! __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
//! the symbol table so that objtool can reference it when it generates the
//! .static_call_sites section.
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
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! strlen("ratelimit") + 1
//! label
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! This counts to 15. Any more, it will return 16th argument.
//! The generic per-CPU infrastrucutre is not suitable for
//! reading const-qualified variables.
//! Maximum supported processors.  Setting this smaller saves quite a
//! bit of memory.  Use nr_cpu_ids instead of this except for static bitmaps.
//!
//! Places which use this should consider cpumask_var_t.
//! Base implementations of per-CPU variable declarations and definitions, where
//! the section in which the variable is to be placed is provided by the
//! 'sec' argument.  This may be used to affect the parameters governing the
//! variable's storage.
//!
//! NOTE!  The sections for the DECLARE and for the DEFINE must match, lest
//! linkage errors occur due the compiler generating the wrong code to access
//! that section.
//! Defines x86 CPU feature bits
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
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
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
//! on 4KB pages
//! Try again
//! Value too large for defined data type
//! Operation is not supported
//! IS_ERR_VALUE - Detect an error pointer.
//! @x: The pointer to check.
//!
//! Like IS_ERR(), but does not generate a compiler warning if result is unused.
//! Use alternative patching for paravirt calls:
//! - For replacing an indirect call with a direct one, use the "normal"
//!   ALTERNATIVE() macro with the indirect call as the initial code sequence,
//!   which will be replaced with the related direct call by using the
//!   ALT_FLAG_DIRECT_CALL special case and the "always on" feature.
//! - In case the replacement is either a direct call or a short code sequence
//!   depending on a feature bit, the ALTERNATIVE_2() macro is being used.
//!   The indirect call is the initial code sequence again, while the special
//!   code sequence is selected with the specified feature bit. In case the
//!   feature is not active, the direct call is used as above via the
//!   ALT_FLAG_DIRECT_CALL special case and the "always on" feature.
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! @a is a power of 2 value
//! cpumask_bits - get the bits in a cpumask
//! @maskp: the struct cpumask *
//!
//! You should only assume nr_cpu_ids bits of this mask are valid.  This is
//! a macro so it's const-correct.
//! __sign_use for integer expressions:
//!   bit #0 set if ok for unsigned comparisons
//!   bit #1 set if ok for signed comparisons
//!
//! In particular, statically non-negative signed integer expressions
//! are ok for both.
//!
//! NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
//! in expressions, and are accepted for signed conversions for now.
//! This is debatable.
//!
//! Note that 'x' is the original expression, and 'ux' is the unique variable
//! that contains the value.
//!
//! We use 'ux' for pure type checking, and 'x' for when we need to look at the
//! value (but without evaluating it for side effects!
//! Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
//!
//! Pointers end up being checked by the normal C type rules at the actual
//! comparison, and these expressions only need to be careful to not cause
//! warnings for pointer use.
//! abs - return absolute value of an argument
//! @x: the value.
//!
//! If it is unsigned type, @x is converted to signed type first.
//! char is treated as if it was signed (regardless of whether it really is)
//! but the macro's return type is preserved as char.
//!
//! NOTE, for signed type if @x is the minimum, the returned result is undefined
//! as there is not enough bits to represent it as a positive number.
//!
//! Return: an absolute value of @x.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! list_for_each	-	iterate over a list
//! @pos:	the &struct list_head to use as a loop cursor.
//! @head:	the head for your list.
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
//! for_each_cpu - iterate over every cpu in a mask
//! @cpu: the (optionally unsigned) integer iterator
//! @mask: the cpumask pointer
//!
//! After the loop, cpu is >= nr_cpu_ids.
//! Wrap the arch provided IRQ routines to provide appropriate checks.
//! Instruction encoding provided for binutils backwards compatibility.
//! Defines VMX CPU feature bits
//! DIV_U64_ROUND_UP - unsigned 64bit divide with 32bit divisor rounded up
//! @ll: unsigned 64bit dividend
//! @d: unsigned 32bit divisor
//!
//! Divide unsigned 64bit dividend by unsigned 32bit divisor
//! and round up.
//!
//! Return: dividend / divisor rounded up
//! X86_MATCH_CPU -  Base macro for CPU matching
//! @_vendor:	The vendor name, e.g. INTEL, AMD, HYGON, ..., ANY
//! 		The name is expanded to X86_VENDOR_@_vendor
//! @_family:	The family number or X86_FAMILY_ANY
//! @_model:	The model number, model constant or X86_MODEL_ANY
//! @_steppings:	Bitmask for steppings, stepping constant or X86_STEPPING_ANY
//! @_feature:	A X86_FEATURE bit or X86_FEATURE_ANY
//! @_data:	Driver specific data or NULL. The internal storage
//! 		format is unsigned long. The supplied value, pointer
//! 		etc. is casted to unsigned long internally.
//!
//! Use only if you need all selectors. Otherwise use one of the shorter
//! macros of the X86_MATCH_* family. If there is no matching shorthand
//! macro, consider to add one. If you really need to wrap one of the macros
//! into another macro at the usage site for good reasons, then please
//! start this local macro with X86_MATCH to allow easy grepping.
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
// Sky Lake
// Sky Lake
// Cypress Cove
// Golden Cove / Gracemont
// Golden Cove / Gracemont
// Raptor Cove / Enhanced Gracemont
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
// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
// bool and also pointer types.
// Useful shorthand for "is this condition known at compile-time?"
//
// Note that the condition may involve non-constant values,
// but the compiler may know enough about the details of the
// values to determine that the condition is statically true.
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
// kcsan_check_write - check regular write access for races
//
// @ptr: address of access
// @size: size of access
// kcsan_check_read_write - check regular read-write access for races
//
// @ptr: address of access
// @size: size of access
// BUILD_BUG_ON_MSG - break compile if a condition is true & emit supplied
// 		      error message.
// @cond: the condition which the compiler should know is false.
// @msg: build-time error message
//
// See BUILD_BUG_ON for description.
// BUILD_BUG - break compile if used.
//
// If you have some code that you expect the compiler to eliminate at
// build time, you should use BUILD_BUG to detect if it is
// unexpectedly used.
// static_assert - check integer constant expression at build time
// @expr: expression to be checked
//
// static_assert() is a wrapper for the C11 _Static_assert, with a
// little macro magic to make the message optional (defaulting to the
// stringification of the tested expression).
//
// Contrary to BUILD_BUG_ON(), static_assert() can be used at global
// scope, but requires the expression to be an integer constant
// expression (i.e., it is not enough that __builtin_constant_p() is
// true for expr).
//
// Also note that BUILD_BUG_ON() fails the build if the condition is
// true, while static_assert() fails the build if the expression is
// false.
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
// Alternative instructions for different CPU types or capabilities.
//
// This allows to use optimized instructions even on generic binary
// kernels.
//
// length of oldinstr must be longer or equal the length of newinstr
// It can be padded with nops as needed.
//
// For non barrier like inlines please define new variants
// without volatile and memory clobber.
// Alternative inline assembly with input.
//
// Peculiarities:
// No memory clobber here.
// Argument numbers start with 1.
// Leaving an unused argument 0 to keep API compatibility.
// Like alternative_input, but with a single output argument
// early_idt_handler_array is an array of entry points referenced in the
// early IDT.  For simplicity, it's a real array with one entry point
// every nine bytes.  That leaves room for an optional 'push $0' if the
// vector has no error code (two bytes), a 'push $vector_number' (two
// bytes), and a jump to the common entry code (up to five bytes).
// Load a segment. Fall back on loading the zero segment if something goes
// wrong.  This variant assumes that loading zero fully clears the segment.
// This is always the case on Intel CPUs and, even on 64-bit AMD CPUs, any
// failure to fully clear the cached descriptor is only observable for
// FS and GS.
// Save a segment register away:
// Concatenate two parameters, but allow them to be expanded beforehand.
// this_cpu_read() makes the compiler load the per-CPU variable every time
// it is accessed while this_cpu_read_stable() allows the value to be cached.
// this_cpu_read_stable() is more efficient and can be used if its value
// is guaranteed to be valid across CPUs.  The current users include
// current_task and cpu_current_top_of_stack, both of which are
// actually per-thread variables implemented as per-CPU variables and
// thus stable for the duration of the respective task.
// alpha modules require percpu variables to be defined as
// weak to force the compiler to generate GOT based external
// references for them.  This is necessary because percpu sections
// will be located outside of the usually addressable area.
//
// This definition puts the following two extra restrictions when
// defining percpu variables.
//
// 1. The symbol must be globally unique, even the static ones.
// 2. Static percpu variables cannot be defined inside a function.
//
// Archs which need weak percpu definitions should set
// CONFIG_ARCH_MODULE_NEEDS_WEAK_PER_CPU when necessary.
//
// Normal declaration and definition macros.
// Variant on the per-CPU variable declaration/definition theme used for
// ordinary per-CPU variables.
// Declaration/definition used for per-CPU variables that are frequently
// accessed and should be in a single cacheline.
//
// For use only by architecture and core code.  Only use scalar or pointer
// types to maximize density.
// Declaration/definition used for per-CPU variables that must be page aligned.
// Declaration/definition used for per-CPU variables that must be read mostly.
// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
// @ptr and is invoked once before a percpu area is accessed by all
// accessors and operations.  This is performed in the generic part of
// percpu and arch overrides don't need to worry about it; however, if an
// arch wants to implement an arch-specific percpu accessor or operation,
// it may use __verify_pcpu_ptr() to verify the parameters.
//
// + 0 is required in order to convert the pointer type from a
// potential array type to a pointer to a single item of the array.
// Add an offset to a pointer.  Use RELOC_HIDE() to prevent the compiler
// from making incorrect assumptions about the pointer value.
// N 32-bit words worth of info
// N 32-bit bug flags
// "sse"
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// "erms" Enhanced REP MOVSB/STOSB instructions
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// IA32_TSC_DEADLINE and X2APIC MSRs need fencing
// "wbnoinvd" WBNOINVD instruction
// "la57" 5-level page tables
// BUG word(s)
// "div0" AMD DIV0 speculation bug
// __CPUID_LEAF() - Define a CPUID output and parse info entry
// @_name:	Struct type name of the CPUID leaf/subleaf (e.g. 'leaf_0x7_0'). Such
// 		types are defined at <cpuid/leaf_types.h> and follow the leaf_0xM_N
// 		format, where 0xM is the leaf and N is the subleaf.
// @_count:	Number of storage entries to allocate for this leaf/subleaf.
//
// For a given leaf/subleaf, define an array of CPUID storage entries and an associated
// query info structure.
//
// Use an array of storage entries to accommodate CPUID leaves with multiple subleaves
// having the same output format.  This is common for hierarchical enumeration; e.g.,
// CPUID(0x4), CPUID(0x12), and CPUID(0x8000001d).
// CPUID_LEAF() - Define a 'struct cpuid_leaves' storage entry
// @_leaf:	Leaf number, in compile-time 0xN format
// @_subleaf:	Subleaf number, in compile-time decimal format
//
// Convenience wrapper around __CPUID_LEAF().
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
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
// On 2MB or 1GB pages
// Extracts the PFN from a (pte|pmd|pud|pgd)val_t of a 4KB page
// Extracts the flags from a (pte|pmd|pud|pgd)val_t
// This includes the protection key value.
// No need to mask any bits for !PAE
// Out of memory
// Device or resource busy
// Invalid argument
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// No static inline type checking - see Subtlety (1) above.
// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
// N 32-bit words worth of info
// X86_MATCH_VFM - Match encoded vendor/family/model
// @vfm:	Encoded 8-bits each for vendor, family, model
// @data:	Driver specific data or NULL. The internal storage
// 		format is unsigned long. The supplied value, pointer
// 		etc. is cast to unsigned long internally.
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
unsafe extern "C" {
    fn x86_match_cpu(arg0: *const x86_cpu_id) -> *const x86_cpu_id;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct x86_cpu_id {
    vendor: u16,
    family: u16,
    model: u16,
    steppings: u16,
    feature: u16,
    flags: u16,
    platform_mask: u8,
    r#type: u8,
    driver_data: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_16_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_16_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_17 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_17_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_17_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_18 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_18_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_18_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_19 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_19_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_19_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_20 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_20_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_20_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_21 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_21_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_21_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_22 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_22_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_22_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_23 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_23_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_23_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_24 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_24_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_cpu_info_c_24_3 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

static g8_cpu_ids: __SlateAlign16<[x86_cpu_id; 10]> = __SlateAlign16([{
let mut __t0: x86_cpu_id = unsafe { core::mem::zeroed() };
__t0.vendor = ((1687u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_16>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
__t0.family = ((1687u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_16_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
__t0.model = ((1687u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_16_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
__t0.steppings = 0u16;
__t0.feature = 0u16;
__t0.flags = 1u16;
__t0.r#type = 0u8;
__t0.driver_data = 0usize as u64;
__t0
}, {
let mut __t1: x86_cpu_id = unsafe { core::mem::zeroed() };
__t1.vendor = ((1690u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_17>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
__t1.family = ((1690u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_17_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
__t1.model = ((1690u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_17_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
__t1.steppings = 0u16;
__t1.feature = 0u16;
__t1.flags = 1u16;
__t1.r#type = 0u8;
__t1.driver_data = 0usize as u64;
__t1
}, {
let mut __t2: x86_cpu_id = unsafe { core::mem::zeroed() };
__t2.vendor = ((1701u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_18>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
__t2.family = ((1701u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_18_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
__t2.model = ((1701u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_18_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
__t2.steppings = 0u16;
__t2.feature = 0u16;
__t2.flags = 1u16;
__t2.r#type = 0u8;
__t2.driver_data = 0usize as u64;
__t2
}, {
let mut __t3: x86_cpu_id = unsafe { core::mem::zeroed() };
__t3.vendor = ((1694u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_19>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
__t3.family = ((1694u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_19_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
__t3.model = ((1694u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_19_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
__t3.steppings = 0u16;
__t3.feature = 0u16;
__t3.flags = 1u16;
__t3.r#type = 0u8;
__t3.driver_data = 0usize as u64;
__t3
}, {
let mut __t4: x86_cpu_id = unsafe { core::mem::zeroed() };
__t4.vendor = ((1678u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_20>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
__t4.family = ((1678u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_20_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
__t4.model = ((1678u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_20_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
__t4.steppings = 0u16;
__t4.feature = 0u16;
__t4.flags = 1u16;
__t4.r#type = 0u8;
__t4.driver_data = 0usize as u64;
__t4
}, {
let mut __t5: x86_cpu_id = unsafe { core::mem::zeroed() };
__t5.vendor = ((1719u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_21>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
__t5.family = ((1719u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_21_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
__t5.model = ((1719u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_21_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
__t5.steppings = 0u16;
__t5.feature = 0u16;
__t5.flags = 1u16;
__t5.r#type = 0u8;
__t5.driver_data = 0usize as u64;
__t5
}, {
let mut __t6: x86_cpu_id = unsafe { core::mem::zeroed() };
__t6.vendor = ((1722u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_22>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
__t6.family = ((1722u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_22_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
__t6.model = ((1722u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_22_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
__t6.steppings = 0u16;
__t6.feature = 0u16;
__t6.flags = 1u16;
__t6.r#type = 0u8;
__t6.driver_data = 0usize as u64;
__t6
}, {
let mut __t7: x86_cpu_id = unsafe { core::mem::zeroed() };
__t7.vendor = ((1727u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_23>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
__t7.family = ((1727u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_23_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
__t7.model = ((1727u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_23_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
__t7.steppings = 0u16;
__t7.feature = 0u16;
__t7.flags = 1u16;
__t7.r#type = 0u8;
__t7.driver_data = 0usize as u64;
__t7
}, {
let mut __t8: x86_cpu_id = unsafe { core::mem::zeroed() };
__t8.vendor = ((1703u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_24>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
__t8.family = ((1703u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_24_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
__t8.model = ((1703u64 & (((core::mem::size_of::<__SlateRecord_intel_cpu_info_c_24_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
__t8.steppings = 0u16;
__t8.feature = 0u16;
__t8.flags = 1u16;
__t8.r#type = 0u8;
__t8.driver_data = 0usize as u64;
__t8
}, {
let mut __t9: x86_cpu_id = unsafe { core::mem::zeroed() };
__t9
}]);

/// intel_match_g8_cpu - match current CPU against g8_cpu_ids
///
/// This matches current CPU against g8_cpu_ids, which are applicable
/// for G8 workaround.
///
/// Returns: %true if matches, %false otherwise.
#[unsafe(no_mangle)]
extern "C" fn intel_match_g8_cpu() -> bool {
!unsafe { x86_match_cpu(core::ptr::addr_of!(g8_cpu_ids.0) as *const x86_cpu_id) }.is_null()
}
