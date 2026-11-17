
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! context_unsafe() - disable context checking for contained code
//!
//! Disables context checking for contained statements or expression.
//!
//! .. code-block:: c
//!
//! 	struct some_data {
//! 		spinlock_t lock;
//! 		int counter __guarded_by(&lock);
//! 	};
//!
//! 	int foo(struct some_data *d)
//! 	{
//! 		// ...
//! 		// other code that is still checked ...
//! 		// ...
//! 		return context_unsafe(d->counter);
//! 	}
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0-only
//!
//! Copyright (C) 2022 Linutronix GmbH, John Ogness
//!
//! Copyright (C) 2022 Intel, Thomas Gleixner
//! small_const_nbits(n) is true precisely when it is known at compile-time
//! that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
//! various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
//! of size 0 are very rare, and a compile-time-known-size 0 is most likely
//! a sign of error. They will be handled correctly by the bit/bitmap APIs,
//! but using the out-of-line functions, so that the inline implementations
//! can unconditionally dereference the pointer(s).
//! aligned_u64 should be used in defining kernel<->userspace ABIs to avoid
//! common 32/64-bit compat problems.
//! 64-bit values align to 4-byte boundaries on x86_32 (and possibly other
//! architectures) and to 8-byte boundaries on 64-bit architectures.  The new
//! aligned_64 type enforces 8-byte alignment so that structs containing
//! aligned_64 values have the same alignment on 32-bit and 64-bit architectures.
//! No conversions are necessary between 32-bit user-space and a 64-bit kernel.
//! The type of an index into the pagecache.
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! error conditions
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
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
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Defines x86 CPU feature bits
//! This counts to 15. Any more, it will return 16th argument.
//! Prevent speculative execution past this barrier.
//! Operation not permitted
//! Value too large for defined data type
//! Helper macro for scoped_guard().
//!
//! Note that the "!__is_cond_ptr(_name)" part of the condition ensures that
//! compiler would be sure that for the unconditional locks the body of the
//! loop (caller-provided code glued to the else clause) could not be skipped.
//! It is needed because the other part - "__guard_ptr(_name)(&scope)" - is too
//! hard to deduce (even if could be proven true for unconditional locks).
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
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! writeable
//! 4th level page in 5-level paging case
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
//! @a is a power of 2 value
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
//! restart by calling sys_restart_syscall
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
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
//! Wrap the arch provided IRQ routines to provide appropriate checks.
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
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
//! Parameters used to convert the timespec values:
//! AUX clock support. AUXiliary clocks are dynamically configured by
//! enabling a clock ID. These clock can be steered independently of the
//! core timekeeper. The kernel can support up to 8 auxiliary clocks, but
//! the actual limit depends on eventual architecture constraints vs. VDSO.
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! signal pending
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! If !LOCKDEP, the value is meaningless
//! Shortcut for the queued_spin_lock_slowpath() function that allows
//! virt to hijack it.
//!
//! Returns:
//!   true - lock has been negotiated, all done;
//!   false - queued_spin_lock_slowpath() will do its thing.
//! Remapping spinlock architecture specific functions to the corresponding
//! queued spinlock functions.
//! A writer holds the lock
//! Define the various spin_lock methods.  Note we define these
//! regardless of whether CONFIG_SMP or CONFIG_PREEMPTION are set. The
//! various methods are defined as nops in the case they are not
//! required.
//! Convenience macros for the sake of set_current_state:
//! Dependency order vs. p above.
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
//! 'empty' nodes are nodes that are known not to be inserted in an rbtree
//! Allocated nodes are mutable until they have been inserted into the tree,
//! at which time they cannot change their type until they have been removed
//! from the tree and an RCU grace period has passed.
//!
//! Removed nodes have their ->parent set to point to themselves.  RCU readers
//! check ->parent before relying on the value that they loaded from the
//! slots array.  This lets us reuse the slots array for the RCU head.
//!
//! Nodes in the tree point to their parent unless bit 0 is set.
//!
//! 64bit sizes
//! wait_queue_entry::flags
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
//! Change timeval to jiffies, trying to avoid the
//! most obvious overflows..
//!
//! And some not so obvious.
//!
//! Note that we don't want to return LONG_MAX, because
//! for various timeout reasons we often end up having
//! to wait "jiffies+1" in order to guarantee that we wait
//! at _least_ "jiffies" - so "jiffies+1" had better still
//! be positive.
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
//! The seqlock seqcount_t interface does not prescribe a precise sequence of
//! read begin/retry/end. For readers, typically there is a call to
//! read_seqcount_begin() and read_seqcount_retry(), however, there are more
//! esoteric cases which do not follow this pattern.
//!
//! As a consequence, we take the following best-effort approach for raw usage
//! via seqcount_t under KCSAN: upon beginning a seq-reader critical section,
//! pessimistically mark the next KCSAN_SEQLOCK_REGION_MAX memory accesses as
//! atomics; if there is a matching read_seqcount_retry() call, no following
//! memory operations are considered atomic. Usage of the seqlock_t interface
//! is not affected.
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
//! __MAX_NR_ZONES
//! percpu_counter batch for local add or sub
//! page_folio - Converts from page to folio.
//! @p: The page.
//!
//! Every page is part of a folio.  This function cannot be called on a
//! NULL pointer.
//!
//! Context: No reference, nor lock is required on @page.  If the caller
//! does not hold a reference, this call may race with a folio split, so
//! it should re-check the folio still contains this page after gaining
//! a reference on the folio.
//! Return: The folio which contains this page.
//! local_lock_init - Runtime initialize a lock instance
//! @lock:	The lock variable
//! We do arithmetic on the LRU lists in various places in the code,
//! so it is important to keep the active lists LRU_ACTIVE higher in
//! the array than the corresponding inactive lists, and to keep
//! the *_FILE lists LRU_FILE higher than the corresponding _ANON lists.
//!
//! This has to be kept in sync with the statistics in zone_stat_item
//! above and the descriptions in vmstat_text in mm/vmstat.c
//! Helper macro to avoid gfp flags if they are the default one
//! MXCSR will be initialized
//! mprotect needs to preserve PAT and encryption bits when updating
//! vm_page_prot
//! 6 because 6 should be plenty and struct tlb_state will fit in two cache
//! lines.
//! CAP_BPF allows the following BPF operations:
//! - Creating all types of BPF maps
//! - Advanced verifier features
//!   - Indirect variable access
//!   - Bounded loops
//!   - BPF to BPF function calls
//!   - Scalar precision tracking
//!   - Larger complexity limits
//!   - Dead code elimination
//!   - And potentially other features
//! - Loading BPF Type Format (BTF) data
//! - Retrieve xlated and JITed code of BPF programs
//! - Use bpf_spin_lock() helper
//!
//! CAP_PERFMON relaxes the verifier checks further:
//! - BPF progs can use of pointer-to-integer conversions
//! - speculation attack hardening measures are bypassed
//! - bpf_probe_read to read arbitrary kernel memory is allowed
//! - bpf_trace_printk to print kernel memory is allowed
//!
//! CAP_SYS_ADMIN is required to use bpf_probe_write_user.
//!
//! CAP_SYS_ADMIN is required to iterate system wide loaded
//! programs, maps, links, BTFs and convert their IDs to file descriptors.
//!
//! CAP_PERFMON and CAP_BPF are required to load tracing programs.
//! CAP_NET_ADMIN and CAP_BPF are required to load networking programs.
//! Shifted versions of the command enable bits are be used if the command
//! has no arguments (see kdb_check_flags). This allows commands, such as
//! go, to have different permissions depending upon whether it is called
//! with an argument.
//! This hash multiplies the input by a large odd number and takes the
//! high bits.  Since multiplication propagates changes to the most
//! significant end only, it is essential that the high bits of the
//! product be used for the hash value.
//!
//! Chuck Lever verified the effectiveness of this technique:
//! http://www.citi.umich.edu/techreports/reports/citi-tr-00-1.pdf
//!
//! Although a random odd number will do, it turns out that the golden
//! ratio phi = (sqrt(5)-1)/2, or its negative, has particularly nice
//! properties.  (See Knuth vol 3, section 6.4, exercise 9.)
//!
//! These are the negative, (1 - phi) = phi**2 = (3 - sqrt(5))/2,
//! which is very slightly easier to multiply by and makes no
//! difference to the hash distribution.
//! the maximum size of a formatted record (i.e. with prefix added
//! per line and dropped messages or in extended message format)
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
// __context_unsafe() - function attribute, disable context checking
// @comment: comment explaining why opt-out is safe
//
// Function attribute denoting that context analysis is disabled for the
// whole function. Forces adding an inline comment as argument.
// token_context_lock() - declare an abstract global context lock instance
// @name: token context lock name
//
// Helper that declares an abstract global context lock instance @name, but not
// backed by a real data structure (linker error if accidentally referenced).
// The type name is `__ctx_lock_@name`.
// token_context_lock_instance() - declare another instance of a global context lock
// @ctx: token context lock previously declared with token_context_lock()
// @name: name of additional global context lock instance
//
// Helper that declares an additional instance @name of the same token context
// lock class @ctx. This is helpful where multiple related token contexts are
// declared, to allow using the same underlying type (`__ctx_lock_@ctx`) as
// function arguments.
// __must_hold() - function attribute, caller must hold exclusive context lock
//
// Function attribute declaring that the caller must hold the given context
// lock instance(s) exclusively.
// __acquires() - function attribute, function acquires context lock exclusively
//
// Function attribute declaring that the function acquires the given context
// lock instance(s) exclusively, but does not release them.
// __cond_acquires() - function attribute, function conditionally
//                     acquires a context lock exclusively
// @ret: abstract value returned by function if context lock acquired
// @x: context lock instance pointer
//
// Function attribute declaring that the function conditionally acquires the
// given context lock instance @x exclusively, but does not release it. The
// function return value @ret denotes when the context lock is acquired.
//
// @ret may be one of: true, false, nonzero, 0, nonnull, NULL.
// __releases() - function attribute, function releases a context lock exclusively
//
// Function attribute declaring that the function releases the given context
// lock instance(s) exclusively. The associated context(s) must be active on
// entry.
// __acquire() - function to acquire context lock exclusively
// @x: context lock instance pointer
//
// No-op function that acquires the given context lock instance @x exclusively.
// __release() - function to release context lock exclusively
// @x: context lock instance pointer
//
// No-op function that releases the given context lock instance @x.
// __must_hold_shared() - function attribute, caller must hold shared context lock
//
// Function attribute declaring that the caller must hold the given context
// lock instance(s) with shared access.
// __acquires_shared() - function attribute, function acquires context lock shared
//
// Function attribute declaring that the function acquires the given
// context lock instance(s) with shared access, but does not release them.
// __cond_acquires_shared() - function attribute, function conditionally
//                            acquires a context lock shared
// @ret: abstract value returned by function if context lock acquired
// @x: context lock instance pointer
//
// Function attribute declaring that the function conditionally acquires the
// given context lock instance @x with shared access, but does not release it.
// The function return value @ret denotes when the context lock is acquired.
//
// @ret may be one of: true, false, nonzero, 0, nonnull, NULL.
// __releases_shared() - function attribute, function releases a
//                       context lock shared
//
// Function attribute declaring that the function releases the given context
// lock instance(s) with shared access. The associated context(s) must be
// active on entry.
// __acquire_shared() - function to acquire context lock shared
// @x: context lock instance pointer
//
// No-op function that acquires the given context lock instance @x with shared
// access.
// __release_shared() - function to release context lock shared
// @x: context lock instance pointer
//
// No-op function that releases the given context lock instance @x with shared
// access.
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
// The second argument is optional (default 0), so we use a variadic macro
// to make the shorthand.
//
// Beware: Do not apply this to functions which may return
// ERR_PTRs. Also, it is probably unwise to apply it to functions
// returning extra information in the low bits (but in that case the
// compiler should see some alignment anyway, when the return value is
// massaged by 'flags = ptr & 3; ptr &= ~3;').
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-assume_005faligned-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#assume-aligned
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-cleanup-variable-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#cleanup
// Note the long name.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-const-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-format-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#format
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-malloc-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#malloc
// Note: deliberately not named '__nonnull', to avoid clashing with glibc's
// __nonnull() when kernel and userspace headers are combined.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Attributes.html#index-nonnull
// clang: https://clang.llvm.org/docs/AttributeReference.html#nonnull
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
// data_race - mark an expression as containing intentional data races
//
// This data_race() macro is useful for situations in which data races
// should be forgiven.  One example is diagnostic code that accesses
// shared variables but is not a part of the core synchronization design.
// For example, if accesses to a given variable are protected by a lock,
// except for diagnostic code, then the accesses under the lock should
// be plain C-language accesses and those in the diagnostic code should
// use data_race().  This way, KCSAN will complain if buggy lockless
// accesses to that variable are introduced, even if the buggy accesses
// are protected by READ_ONCE() or WRITE_ONCE().
//
// This macro *does not* affect normal code generation, but is a hint
// to tooling that data races here are to be ignored.  If the access must
// be atomic *and* KCSAN should ignore the access, use both data_race()
// and READ_ONCE(), for example, data_race(READ_ONCE(x)).
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
// objtool annotation to ignore the alternatives and only consider the original
// instruction(s).
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
// reg := -EIO
// reg := -EIO
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
// BUILD_BUG_ON_INVALID() permits the compiler to check the validity of the
// expression but avoids the generation of any code, even if that expression
// has side-effects.
// BUILD_BUG_ON_MSG - break compile if a condition is true & emit supplied
// 		      error message.
// @cond: the condition which the compiler should know is false.
// @msg: build-time error message
//
// See BUILD_BUG_ON for description.
// BUILD_BUG_ON - break compile if a condition is true.
// @condition: the condition which the compiler should know is false.
//
// If you have some code which relies on certain constants being equal, or
// some other compile-time-evaluated condition, you should use BUILD_BUG_ON to
// detect if someone changes it.
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
// modpost check for section mismatches during the kernel build.
// A section mismatch happens when there are references from a
// code or data section to an init section (both code or data).
// The init sections are (for most archs) discarded by the kernel
// when early init has completed so all such references are potential bugs.
// For exit sections the same issue exists.
//
// The following markers are used for the cases where the reference to
// the *init / *exit section (code or data) is valid and will teach
// modpost not to issue a warning.  Intended semantics is that a code or
// data tagged __ref* can reference code or data from init section without
// producing a warning (of course, no warning does not mean code is
// correct, so optimally document why the __ref is needed and why it's OK).
//
// The markers follow same syntax rules as __init / __initdata.
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
// pr_warn - Print a warning-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_WARNING loglevel. It uses pr_fmt()
// to generate the format string.
// new instruction
// feature + flags
// source len
// replacement len
// replacement
// alternative assembly primitive:
// If @feature is set, patch in @newinstr_yes, otherwise @newinstr_no.
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
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "apic" Onboard APIC
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "pti" Kernel Page Table Isolation enabled
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// "erms" Enhanced REP MOVSB/STOSB instructions
// "smap" Supervisor Mode Access Prevention
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// IA32_TSC_DEADLINE and X2APIC MSRs need fencing
// "lass" Linear Address Space Separation
// Fast short REP STOSB
// Non-serializing WRMSR
// INVLPGB and TLBSYNC instructions supported
// "wbnoinvd" WBNOINVD instruction
// "ospke" OS Protection Keys Enable
// Shadow stack
// "la57" 5-level page tables
// "rdpid" RDPID instruction
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "l1tf" CPU is affected by L1 Terminal Fault
// "div0" AMD DIV0 speculation bug
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
// Atomic operations are already serializing on x86
// No such file or directory
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// Invalid argument
// To support Context Analysis, we need to allow the compiler to see the
// acquisition and release of the context lock. However, the "cleanup" helpers
// wrap the lock in a struct passed through separate helper functions, which
// hides the lock alias from the compiler (no inter-procedural analysis).
//
// To make it work, we introduce an explicit alias to the context lock instance
// that is "cleaned" up with a separate cleanup helper. This helper is a dummy
// function that does nothing at runtime, but has the "_unlock" attribute to
// tell the compiler what happens at the end of the scope.
//
// To generalize the pattern, the WITH_LOCK_GUARD_1_ATTRS() macro should be used
// to redefine the constructor, which then also creates the alias variable with
// the right "cleanup" attribute, *after* DECLARE_LOCK_GUARD_1_ATTRS() has been
// used.
//
// Example usage:
//
//   DECLARE_LOCK_GUARD_1_ATTRS(mutex, __acquires(_T), __releases(*(struct mutex **)_T))
//   #define class_mutex_constructor(_T) WITH_LOCK_GUARD_1_ATTRS(mutex, _T)
//
// Note: To support the for-loop based scoped helpers, the auxiliary variable
// must be a pointer to the "class" type because it is defined in the same
// statement as the guard variable. However, we initialize it with the lock
// pointer (despite the type mismatch, the compiler's alias analysis still works
// as expected). The "_unlock" attribute receives a pointer to the auxiliary
// variable (a double pointer to the class type), and must be cast and
// dereferenced appropriately.
// Generate a per-CPU add to memory instruction and optimize code
// if one is added or subtracted.
// Add return operation
// this_cpu_read() makes the compiler load the per-CPU variable every time
// it is accessed while this_cpu_read_stable() allows the value to be cached.
// this_cpu_read_stable() is more efficient and can be used if its value
// is guaranteed to be valid across CPUs.  The current users include
// current_task and cpu_current_top_of_stack, both of which are
// actually per-thread variables implemented as per-CPU variables and
// thus stable for the duration of the respective task.
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
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
// Operation is not supported
// Physical address zone modifiers (see linux/mmzone.h - low four bits)
//
// Do not put any conditional on these. If necessary modify the definitions
// without the underscores and use them consistently. The definitions here may
// be used in bit comparisons.
// ZONE_MOVABLE allowed
// DOC: Page mobility and placement hints
//
// Page mobility and placement hints
//
// These flags provide hints about how mobile the page is. Pages with similar
// mobility are placed within the same pageblocks to minimise problems due
// to external fragmentation.
//
// %__GFP_MOVABLE (also a zone modifier) indicates that the page can be
// moved by page migration during memory compaction or can be reclaimed.
//
// %__GFP_RECLAIMABLE is used for slab allocations that specify
// SLAB_RECLAIM_ACCOUNT and whose pages can be freed via shrinkers.
//
// %__GFP_WRITE indicates the caller intends to dirty the page. Where possible,
// these pages will be spread between local zones to avoid all the dirty
// pages being in one zone (fair zone allocation policy).
//
// %__GFP_HARDWALL enforces the cpuset memory allocation policy.
//
// %__GFP_THISNODE forces the allocation to be satisfied from the requested
// node with no fallbacks or placement policy enforcements.
//
// %__GFP_ACCOUNT causes the allocation to be accounted to the active
// cgroup context.
// DOC: Reclaim modifiers
//
// Reclaim modifiers
// Please note that all the following flags are only applicable to sleepable
// allocations (e.g. %GFP_NOWAIT and %GFP_ATOMIC will ignore them).
//
// %__GFP_IO can start physical IO.
//
// %__GFP_FS can call down to the low-level FS. Clearing the flag avoids the
// allocator recursing into the filesystem which might already be holding
// locks.
//
// %__GFP_DIRECT_RECLAIM indicates that the caller may enter direct reclaim.
// This flag can be cleared to avoid unnecessary delays when a fallback
// option is available.
//
// %__GFP_KSWAPD_RECLAIM indicates that the caller wants to wake kswapd when
// the low watermark is reached and have it reclaim pages until the high
// watermark is reached. A caller may wish to clear this flag when fallback
// options are available and the reclaim is likely to disrupt the system. The
// canonical example is THP allocation where a fallback is cheap but
// reclaim/compaction may cause indirect stalls.
//
// %__GFP_RECLAIM is shorthand to allow/forbid both direct and kswapd reclaim.
//
// The default allocator behavior depends on the request size. We have a concept
// of so-called costly allocations (with order > %PAGE_ALLOC_COSTLY_ORDER).
// !costly allocations are too essential to fail so they are implicitly
// non-failing by default (with some exceptions like OOM victims might fail so
// the caller still has to check for failures) while costly requests try to be
// not disruptive and back off even without invoking the OOM killer.
// The following three modifiers might be used to override some of these
// implicit rules. Please note that all of them must be used along with
// %__GFP_DIRECT_RECLAIM flag.
//
// %__GFP_NORETRY: The VM implementation will try only very lightweight
// memory direct reclaim to get some memory under memory pressure (thus
// it can sleep). It will avoid disruptive actions like OOM killer. The
// caller must handle the failure which is quite likely to happen under
// heavy memory pressure. The flag is suitable when failure can easily be
// handled at small cost, such as reduced throughput.
//
// %__GFP_RETRY_MAYFAIL: The VM implementation will retry memory reclaim
// procedures that have previously failed if there is some indication
// that progress has been made elsewhere.  It can wait for other
// tasks to attempt high-level approaches to freeing memory such as
// compaction (which removes fragmentation) and page-out.
// There is still a definite limit to the number of retries, but it is
// a larger limit than with %__GFP_NORETRY.
// Allocations with this flag may fail, but only when there is
// genuinely little unused memory. While these allocations do not
// directly trigger the OOM killer, their failure indicates that
// the system is likely to need to use the OOM killer soon.  The
// caller must handle failure, but can reasonably do so by failing
// a higher-level request, or completing it only in a much less
// efficient manner.
// If the allocation does fail, and the caller is in a position to
// free some non-essential memory, doing so could benefit the system
// as a whole.
//
// %__GFP_NOFAIL: The VM implementation _must_ retry infinitely: the caller
// cannot handle allocation failures. The allocation could block
// indefinitely but will never return with failure. Testing for
// failure is pointless.
// It _must_ be blockable and used together with __GFP_DIRECT_RECLAIM.
// It should _never_ be used in non-sleepable contexts.
// New users should be evaluated carefully (and the flag should be
// used only when there is no reasonable failure policy) but it is
// definitely preferable to use the flag rather than opencode endless
// loop around allocator.
// Allocating pages from the buddy with __GFP_NOFAIL and order > 1 is
// not supported. Please consider using kvmalloc() instead.
// Caller can reclaim
// DOC: Action modifiers
//
// Action modifiers
//
// %__GFP_NOWARN suppresses allocation failure reports.
//
// %__GFP_COMP address compound page metadata.
//
// %__GFP_ZERO returns a zeroed page on success.
//
// %__GFP_ZEROTAGS zeroes memory tags at allocation time. Setting memory tags at
// the same time as zeroing memory (e.g., with __GFP_ZERO) has minimal
// additional performance impact. However, __GFP_ZEROTAGS also zeroes the tags
// even if memory is not getting zeroed at allocation time (e.g.,
// with init_on_free).
//
// %__GFP_SKIP_KASAN makes KASAN skip unpoisoning on page allocation.
// Used for userspace and vmalloc pages; the latter are unpoisoned by
// kasan_unpoison_vmalloc instead. If passed to vmalloc, kasan_unpoison_vmalloc
// is skipped too. For userspace pages, results in poisoning being skipped as
// well, see should_skip_kasan_poison for details. Only effective in HW_TAGS mode.
// Disable lockdep for GFP context tracking
// DOC: Useful GFP flag combinations
//
// Useful GFP flag combinations
//
// Useful GFP flag combinations that are commonly used. It is recommended
// that subsystems start with one of these combinations and then set/clear
// %__GFP_FOO flags as necessary.
//
// %GFP_ATOMIC users can not sleep and need the allocation to succeed. A lower
// watermark is applied to allow access to "atomic reserves".
// The current implementation doesn't support NMI, nor contexts that disable
// preemption under PREEMPT_RT. This includes raw_spin_lock() and plain
// preempt_disable() - see "Memory allocation" in
// Documentation/core-api/real-time/differences.rst for more info.
//
// %GFP_KERNEL is typical for kernel-internal allocations. The caller requires
// %ZONE_NORMAL or a lower zone for direct access but can direct reclaim.
//
// %GFP_KERNEL_ACCOUNT is the same as GFP_KERNEL, except the allocation is
// accounted to the active cgroup context.
//
// %GFP_NOWAIT is for kernel allocations that should not stall for direct
// reclaim, start physical IO or use any filesystem callback.  It is very
// likely to fail to allocate memory, even for very small allocations.
// The same restrictions on calling contexts apply as for %GFP_ATOMIC.
//
// %GFP_NOIO will use direct reclaim to discard clean pages or slab pages
// that do not require the starting of any physical IO.
// Please try to avoid using this flag directly and instead use
// memalloc_noio_{save,restore} to mark the whole scope which cannot
// perform any IO with a short explanation why. All allocation requests
// will inherit GFP_NOIO implicitly.
//
// %GFP_NOFS will use direct reclaim but will not use any filesystem interfaces.
// Please try to avoid using this flag directly and instead use
// memalloc_nofs_{save,restore} to mark the whole scope which cannot/shouldn't
// recurse into the FS layer with a short explanation why. All allocation
// requests will inherit GFP_NOFS implicitly.
//
// %GFP_USER is for userspace allocations that also need to be directly
// accessibly by the kernel or hardware. It is typically used by hardware
// for buffers that are mapped to userspace (e.g. graphics) that hardware
// still must DMA to. cpuset limits are enforced for these allocations.
//
// %GFP_DMA exists for historical reasons and should be avoided where possible.
// The flags indicates that the caller requires that the lowest zone be
// used (%ZONE_DMA or 16M on x86-64). Ideally, this would be removed but
// it would require careful auditing as some users really require it and
// others use the flag to avoid lowmem reserves in %ZONE_DMA and treat the
// lowest zone as a type of emergency reserve.
//
// %GFP_DMA32 is similar to %GFP_DMA except that the caller requires a 32-bit
// address. Note that kmalloc(..., GFP_DMA32) does not return DMA32 memory
// because the DMA32 kmalloc cache array is not implemented.
// (Reason: there is no such user in kernel).
//
// %GFP_HIGHUSER is for userspace allocations that may be mapped to userspace,
// do not need to be directly accessible by the kernel but that cannot
// move once in use. An example may be a hardware allocation that maps
// data directly into userspace but has no addressing limitations.
//
// %GFP_HIGHUSER_MOVABLE is for userspace allocations that the kernel does not
// need direct access to but can use kmap() when access is required. They
// are expected to be movable via page reclaim or page migration. Typically,
// pages on the LRU would also be allocated with %GFP_HIGHUSER_MOVABLE.
//
// %GFP_TRANSHUGE and %GFP_TRANSHUGE_LIGHT are used for THP allocations. They
// are compound allocations that will generally fail quickly if memory is not
// available and will not wake kswapd/kcompactd on failure. The _LIGHT
// version does not attempt reclaim/compaction at all and is by default used
// in page fault path, while the non-light is used by khugepaged.
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
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
// round_up - round up to next specified power of 2
// @x: the value to round
// @y: multiple to round up to (must be a power of 2)
//
// Rounds @x up to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding up, use roundup() below.
// abs - return absolute value of an argument
// @x: the value.
//
// If it is unsigned type, @x is converted to signed type first.
// char is treated as if it was signed (regardless of whether it really is)
// but the macro's return type is preserved as char.
//
// NOTE, for signed type if @x is the minimum, the returned result is undefined
// as there is not enough bits to represent it as a positive number.
//
// Return: an absolute value of @x.
// list_first_entry - get the first element from a list
// @ptr:	the list head to take the element from.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
//
// Note, that list is expected to be not empty.
// list_next_entry - get the next element in list
// @pos:	the type * to cursor
// @member:	the name of the list_head within the struct.
// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_entry_is_head - test if the entry points to the head of the list
// @pos:	the type * to cursor
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// No static inline type checking - see Subtlety (1) above.
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
// N 32-bit words worth of info
// signal notifications exist
// rescheduling necessary
// re-arm the timer
// The preempt_count offset needed for things like:
//
//  spin_lock_bh()
//
// Which need to disable both preemption (CONFIG_PREEMPT_COUNT) and
// softirqs, such that unlock sequences of:
//
//  spin_unlock();
//  local_bh_enable();
//
// Work as expected.
// disable sas during sighandling
// This function is needed by all SMP systems. It must _always_ be valid
// from the initial startup.
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
// Dependency order vs. p above.
// rcu_assign_pointer() - assign to RCU-protected pointer
// @p: pointer to assign to
// @v: value to assign (publish)
//
// Assigns the specified value to the specified RCU-protected
// pointer, ensuring that any concurrent RCU readers will see
// any prior initialization.
//
// Inserts memory barriers on architectures that require them
// (which is most of them), and also prevents the compiler from
// reordering the code that initializes the structure after the pointer
// assignment.  More importantly, this call documents which pointers
// will be dereferenced by RCU read-side code.
//
// In some special cases, you may use RCU_INIT_POINTER() instead
// of rcu_assign_pointer().  RCU_INIT_POINTER() is a bit faster due
// to the fact that it does not constrain either the CPU or the compiler.
// That said, using RCU_INIT_POINTER() when you should have used
// rcu_assign_pointer() is a very bad thing that results in
// impossible-to-diagnose memory corruption.  So please be careful.
// See the RCU_INIT_POINTER() comment header for details.
//
// Note that rcu_assign_pointer() evaluates each of its arguments only
// once, appearances notwithstanding.  One of the "extra" evaluations
// is in typeof() and the other visible only to sparse (__CHECKER__),
// neither of which actually execute the argument.  As with most cpp
// macros, this execute-arguments-only-once property is important, so
// please be careful when making changes to rcu_assign_pointer() and the
// other macros that it invokes.
// rcu_replace_pointer() - replace an RCU pointer, returning its old value
// @rcu_ptr: RCU pointer, whose old value is returned
// @ptr: regular pointer
// @c: the lockdep conditions under which the dereference will take place
//
// Perform a replacement, where @rcu_ptr is an RCU-annotated
// pointer and @c is the lockdep argument that is passed to the
// rcu_dereference_protected() call used to read that pointer.  The old
// value of @rcu_ptr is returned, and @rcu_ptr is set to @ptr.
// rcu_access_pointer() - fetch RCU pointer with no dereferencing
// @p: The pointer to read
//
// Return the value of the specified RCU-protected pointer, but omit the
// lockdep checks for being in an RCU read-side critical section.  This is
// useful when the value of this pointer is accessed, but the pointer is
// not dereferenced, for example, when testing an RCU-protected pointer
// against NULL.  Within an RCU read-side critical section, there is little
// reason to use rcu_access_pointer().  Although rcu_access_pointer() may
// also be used in cases where update-side locks prevent the value of the
// pointer from changing, you should instead use rcu_dereference_protected()
// for this use case.  It is also permissible to use rcu_access_pointer()
// within lockless updaters to obtain the old value for an atomic operation,
// for example, for cmpxchg().
//
// It is usually best to test the rcu_access_pointer() return value
// directly in order to avoid accidental dereferences being introduced
// by later inattentive changes.  In other words, assigning the
// rcu_access_pointer() return value to a local variable results in an
// accident waiting to happen.
//
// It is also permissible to use rcu_access_pointer() when read-side
// access to the pointer was removed at least one grace period ago, as is
// the case in the context of the RCU callback that is freeing up the data,
// or after a synchronize_rcu() returns.  This can be useful when tearing
// down multi-linked structures after a grace period has elapsed.  However,
// rcu_dereference_protected() is normally preferred for this use case.
// rcu_dereference_check() - rcu_dereference with debug checking
// @p: The pointer to read, prior to dereferencing
// @c: The conditions under which the dereference will take place
//
// Do an rcu_dereference(), but check that the conditions under which the
// dereference will take place are correct.  Typically the conditions
// indicate the various locking conditions that should be held at that
// point.  The check should return true if the conditions are satisfied.
// An implicit check for being in an RCU read-side critical section
// (rcu_read_lock()) is included.
//
// For example:
//
// 	bar = rcu_dereference_check(foo->bar, lockdep_is_held(&foo->lock));
//
// could be used to indicate to lockdep that foo->bar may only be dereferenced
// if either rcu_read_lock() is held, or that the lock required to replace
// the bar struct at foo->bar is held.
//
// Note that the list of conditions may also include indications of when a lock
// need not be held, for example during initialisation or destruction of the
// target struct:
//
// 	bar = rcu_dereference_check(foo->bar, lockdep_is_held(&foo->lock) ||
// 					      atomic_read(&foo->usage) == 0);
//
// Inserts memory barriers on architectures that require them
// (currently only the Alpha), prevents the compiler from refetching
// (and from merging fetches), and, more importantly, documents exactly
// which pointers are protected by RCU and checks that the pointer is
// annotated as __rcu.
// rcu_dereference_protected() - fetch RCU pointer when updates prevented
// @p: The pointer to read, prior to dereferencing
// @c: The conditions under which the dereference will take place
//
// Return the value of the specified RCU-protected pointer, but omit
// the READ_ONCE().  This is useful in cases where update-side locks
// prevent the value of the pointer from changing.  Please note that this
// primitive does *not* prevent the compiler from repeating this reference
// or combining it with other references, so it should not be used without
// protection of appropriate locks.
//
// This function is only for update-side use.  Using this function
// when protected only by rcu_read_lock() will result in infrequent
// but very ugly failures.
// rcu_dereference() - fetch RCU-protected pointer for dereferencing
// @p: The pointer to read, prior to dereferencing
//
// This is a simple wrapper around rcu_dereference_check().
// list_entry_rcu - get the struct for this entry
// @ptr:        the &struct list_head pointer.
// @type:       the type of the struct this is embedded in.
// @member:     the name of the list_head within the struct.
//
// This primitive may safely run concurrently with the _rcu list-mutation
// primitives such as list_add_rcu() as long as it's guarded by rcu_read_lock().
// list_next_or_null_rcu - get the next element from a list
// @head:	the head for the list.
// @ptr:        the list head to take the next element from.
// @type:       the type of the struct this is embedded in.
// @member:     the name of the list_head within the struct.
//
// Note that if the ptr is at the end of the list, NULL is returned.
//
// This primitive may safely run concurrently with the _rcu list-mutation
// primitives such as list_add_rcu() as long as it's guarded by rcu_read_lock().
// return the first or the next element in an RCU protected hlist
// hlist_for_each_entry_srcu - iterate over rcu list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the hlist_node within the struct.
// @cond:	lockdep expression for the lock required to traverse the list.
//
// This list-traversal primitive may safely run concurrently with
// the _rcu list-mutation primitives such as hlist_add_head_rcu()
// as long as the traversal is guarded by srcu_read_lock().
// The lockdep expression srcu_read_lock_held() can be passed as the
// cond argument from read side.
// Same as rb_first(), but O(1)
// SEQCOUNT_LOCKNAME()	- Instantiate seqcount_LOCKNAME_t and helpers
// seqprop_LOCKNAME_*()	- Property accessors for seqcount_LOCKNAME_t
//
// @lockname:		"LOCKNAME" part of seqcount_LOCKNAME_t
// @locktype:		LOCKNAME canonical C data type
// @preemptible:	preemptibility of above locktype
// @lockbase:		prefix for associated lock/unlock
// 									 * Re-read the sequence counter since the (possibly			 * preempted) writer made progress.
// PREEMPT_RT relies on the above LOCK+UNLOCK
// __read_seqcount_begin() - begin a seqcount_t read section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// raw_read_seqcount_begin() - begin a seqcount_t read section w/o lockdep
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// read_seqcount_begin() - begin a seqcount_t read critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// raw_read_seqcount() - read the raw seqcount_t counter value
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// raw_read_seqcount opens a read critical section of the given
// seqcount_t, without any lockdep checking, and without checking or
// masking the sequence counter LSB. Calling code is responsible for
// handling that.
//
// Return: count to be passed to read_seqcount_retry()
// raw_seqcount_try_begin() - begin a seqcount_t read critical section
//                            w/o lockdep and w/o counter stabilization
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
// @start: count to be passed to read_seqcount_retry()
//
// Similar to raw_seqcount_begin(), except it enables eliding the critical
// section entirely if odd, instead of doing the speculation knowing it will
// fail.
//
// Useful when counter stabilization is more or less equivalent to taking
// the lock and there is a slowpath that does that.
//
// If true, start will be set to the (even) sequence count read.
//
// Return: true when a read critical section is started.
// read_seqcount_retry() - end a seqcount_t read critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
// @start: count, from read_seqcount_begin()
//
// read_seqcount_retry closes the read critical section of given
// seqcount_t.  If the critical section was invalid, it must be ignored
// (and typically retried).
//
// Return: true if a read section retry is required, else false
// seqlock_init() - dynamic initializer for seqlock_t
// @sl: Pointer to the seqlock_t instance
// read_seqlock_excl_irqsave() - start a non-interruptible seqlock_t
// 				 locking reader section
// @lock:  Pointer to seqlock_t
// @flags: Stack-allocated storage for saving caller's local interrupt
//         state, to be passed to read_sequnlock_excl_irqrestore().
//
// _irqsave variant of read_seqlock_excl(). Use this only if the seqlock_t
// write side section, *or other read sections*, can be invoked from a
// hardirq context.
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
// Don't call further
// Page flags policies wrt compound pages
//
// PF_POISONED_CHECK
//     check if this struct page poisoned/uninitialized
//
// PF_ANY:
//     the page flag is relevant for small, head and tail pages.
//
// PF_HEAD:
//     for compound page all operations related to the page flag applied to
//     head page.
//
// PF_NO_TAIL:
//     modifications of the page flag must be done on small or head pages,
//     checks can be done on tail pages too.
//
// PF_NO_COMPOUND:
//     the page flag is not relevant for compound pages.
//
// PF_SECOND:
//     the page flag is stored in the first tail page.
// Which page is the flag stored in
// Macros to create function definitions for page flags
// On an anonymous folio mapped into a user virtual memory area,
// folio->mapping points to its anon_vma, not to a struct address_space;
// with the FOLIO_MAPPING_ANON bit set to distinguish it.  See rmap.h.
//
// On an anonymous folio in a VM_MERGEABLE area, if CONFIG_KSM is enabled,
// the FOLIO_MAPPING_ANON_KSM bit may be set along with the FOLIO_MAPPING_ANON
// bit; and then folio->mapping points, not to an anon_vma, but to a private
// structure which KSM associates with that merged folio.  See ksm.h.
//
// Please note that, confusingly, "folio_mapping" refers to the inode
// address_space which maps the folio from disk; whereas "folio_mapped"
// refers to user virtual address space into which the folio is mapped.
//
// For slab pages, since slab reuses the bits in struct page to store its
// internal states, the folio->mapping does not exist as such, nor do
// these flags below.  So in order to avoid testing non-existent bits,
// please make sure that folio_test_slab(folio) actually evaluates to
// false before calling the following functions (e.g., folio_test_anon).
// See mm/slab.h.
// local_lock - Acquire a per CPU local lock
// @lock:	The lock variable
// local_lock_irq - Acquire a per CPU local lock and disable interrupts
// @lock:	The lock variable
// local_lock_irqsave - Acquire a per CPU local lock, save and disable
// 			 interrupts
// @lock:	The lock variable
// @flags:	Storage for interrupt flags
// local_unlock - Release a per CPU local lock
// @lock:	The lock variable
// local_unlock_irq - Release a per CPU local lock and enable interrupts
// @lock:	The lock variable
// local_unlock_irqrestore - Release a per CPU local lock and restore
// 			      interrupt flags
// @lock:	The lock variable
// @flags:      Interrupt flags to restore
// local_trylock_init - Runtime initialize a lock instance
// @lock:	The lock variable
// Maximum number of zones on a zonelist
// zone_idx() returns 0 for the ZONE_DMA zone, 1 for the ZONE_NORMAL zone, etc.
// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
// VMA locks do not behave like most ordinary locks found in the kernel, so we
// cannot quite have full lockdep tracking in the way we would ideally prefer.
//
// Read locks act as shared locks which exclude an exclusive lock being
// taken. We therefore mark these accordingly on read lock acquire/release.
//
// Write locks are acquired exclusively per-VMA, but released in a shared
// fashion, that is upon vma_end_write_all(), we update the mmap's seqcount such
// that write lock is released.
//
// We therefore cannot track write locks per-VMA, nor do we try. Mitigating this
// is the fact that, of course, we do lockdep-track the mmap lock rwsem which
// must be held when taking a VMA write lock.
//
// We do, however, want to indicate that during either acquisition of a VMA
// write lock or detachment of a VMA that we require the lock held be exclusive,
// so we utilise lockdep to do so.
// All top-level MITIGATION_PAGE_TABLE_ISOLATION page tables are order-1 pages
// (8k-aligned and 8k in size).  The kernel one is at the beginning 4k and
// the user one is in the last 4k.  To switch between them, you
// just need to flip the 12th bit in their addresses.
// Please ignore the name of this function.  It should be called
// switch_to_kernel_thread().
//
// enter_lazy_tlb() is a hint from the scheduler that we are entering a
// kernel thread or other context without an mm.  Acceptable implementations
// include doing nothing whatsoever, switching to init_mm, or various clever
// lazy tricks to try to minimize TLB flushes.
//
// The scheduler reserves the right to call enter_lazy_tlb() several times
// in a row.  It will notify us that we're going back to a real mm by
// calling switch_mm_irqs_off().
// Keep stack-allocated flush_tlb_info cacheline aligned, but cap the
// alignment to avoid excessive stack usage on large-cacheline systems.
// kthread_create - create a kthread on the current node
// @threadfn: the function to run in the thread
// @data: data pointer for @threadfn()
// @namefmt: printf-style format string for the thread name
// @arg: arguments for @namefmt.
//
// This macro will create a kthread on the current node, leaving it in
// the stopped state.  This is just a helper for kthread_create_on_node();
// see the documentation there for more details.
// kthread_run - create and wake a thread.
// @threadfn: the function to run until signal_pending(current).
// @data: data ptr for @threadfn.
// @namefmt: printf-style name for the thread.
//
// Description: Convenient wrapper for kthread_create() followed by
// wake_up_process().  Returns the kthread or ERR_PTR(-ENOMEM).
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
macro_rules! BIT_MASK {
($nr:expr) => {
UL!() << ($nr) % 64
};
}

macro_rules! BIT_WORD {
($nr:expr) => {
($nr) / 64
};
}

macro_rules! BIT {
($nr:expr) => {
UL!() << $nr
};
}

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

/// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
/// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
/// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
macro_rules! IS_ENABLED__2 {
() => {
__or!()
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

/// IS_ERR_VALUE - Detect an error pointer.
/// @x: The pointer to check.
///
/// Like IS_ERR(), but does not generate a compiler warning if result is unused.
macro_rules! IS_ERR_VALUE {
($x:expr) => {
unlikely__2!(((($x) as *mut ()) as u64) >= (4095i32.wrapping_neg() as u64))
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

/// RCU_INITIALIZER() - statically initialize an RCU-protected global variable
/// @v: The value to statically initialize with.
macro_rules! RCU_INITIALIZER {
($v:expr) => {
$v
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! ULSEQ_MAX {
() => {
1i32.wrapping_neg()
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! _BITUL {
($x:expr) => {
_UL!() << $x
};
}

macro_rules! _UL {
() => {
_AC!()
};
}

macro_rules! __READ_ONCE {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i32) }
};
}

macro_rules! __READ_ONCE__2 {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i64) }
};
}

macro_rules! __WRITE_ONCE__3 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut i32, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__10 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut task_struct, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__2 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u64, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__11 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut i64, $val) };
if true {
break;
}
}
};
}

macro_rules! ____is_defined {
() => {
0i32
};
}

macro_rules! ____or {
() => {
1i32
};
}

macro_rules! __cmp_once__11 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = $y;
__v = if __v2 > __v3 { __v2 } else { __v3 };
__v
}
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
macro_rules! __native_word__17 {
() => {
((const { core::mem::size_of::<i16>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<i16>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<i16>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<i16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__8 {
() => {
((const { core::mem::size_of::<*mut task_struct>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut task_struct>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut task_struct>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut task_struct>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

macro_rules! __u64seq_to_ulseq {
($u64seq:expr) => {
$u64seq
};
}

macro_rules! __ulseq_to_u64seq {
($ulseq:expr) => {
$ulseq
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__7 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<u32>() as *const ();
if true {
break;
}
}
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__2 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<*mut task_struct>() as *const ();
if true {
break;
}
}
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
};
}

macro_rules! absolute_pointer {
($val:expr) => {
RELOC_HIDE!(($val) as *mut (), 0i32)
};
}

macro_rules! alloc_tag_restore {
() => {
while false {
}
};
}

macro_rules! arch_atomic_sub_return {
($i:expr, $v:expr) => {
arch_atomic_add_return($i.wrapping_neg(), $v)
};
}

macro_rules! cant_migrate {
() => {
while false {
}
};
}

/// check_mul_overflow() - Calculate multiplication with overflow checking
/// @a: first factor
/// @b: second factor
/// @d: pointer to store product
///
/// Returns true on wrap-around, false otherwise.
///
/// *@d holds the results of the attempted multiplication, regardless of whether
/// wrap-around occurred.
macro_rules! check_mul_overflow {
($a:expr, $b:expr, $d:expr) => {
__must_check_overflow({
let __t: i128 = ($a) as i128;
let __t2: i128 = ($b) as i128;
let __t3: i128 = __t.wrapping_mul(__t2);
unsafe {
*$d = __t3 as u64;
}
__t.checked_mul(__t2).is_none() || ((__t3 as u64) as i128) != __t3
})
};
}

macro_rules! cond_resched__5 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/kernel/printk/nbcon.c".as_ptr() as *const u8, 1275i32, 0u32);
__v = _cond_resched();
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__14 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__34!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__27 {
($ptr:expr) => {
{
let mut __v: *mut console;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__59!() as isize)) }) as *mut ()) as *mut console;
__v
}
};
}

macro_rules! debug_normal_state_change {
() => {
while false {
}
};
}

macro_rules! hlist_entry {
($ptr:expr) => {
container_of__27!($ptr)
};
}

macro_rules! kcsan_check_atomic_read {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 4i32);
};
}

macro_rules! kcsan_check_atomic_read_write {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 7i32);
};
}

macro_rules! kcsan_check_atomic_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 5i32);
};
}

/// kcsan_check_read_write - check regular read-write access for races
///
/// @ptr: address of access
/// @size: size of access
macro_rules! kcsan_check_read_write {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 3i32);
};
}

macro_rules! kcsan_mb {
() => {
while false {
}
};
}

macro_rules! kcsan_release {
() => {
while false {
}
};
}

macro_rules! kmalloc_index {
($s:expr) => {
__kmalloc_index($s, true)
};
}

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! local_irq_restore {
($flags:expr) => {
loop {
raw_local_irq_restore!($flags);
if true {
break;
}
}
};
}

macro_rules! local_irq_save__2 {
($flags:expr) => {
loop {
raw_local_irq_save__2!($flags);
if true {
break;
}
}
};
}

macro_rules! local_irq_save {
($flags:expr) => {
loop {
raw_local_irq_save!($flags);
if true {
break;
}
}
};
}

/// max_t - return maximum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! max_t__6 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__11!($type, $x, $y)
};
}

macro_rules! nbcon_context_enter_unsafe {
($c:expr) => {
__nbcon_context_update_unsafe($c, true)
};
}

macro_rules! nbcon_context_exit_unsafe {
($c:expr) => {
__nbcon_context_update_unsafe($c, false)
};
}

macro_rules! offsetof__59 {
() => {
core::mem::offset_of!(console, node) as u64
};
}

macro_rules! offsetof__34 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, irq_work) as u64
};
}

macro_rules! preempt_count_add {
($val:expr) => {
__preempt_count_add($val);
};
}

macro_rules! preempt_count_dec_and_test {
() => {
__preempt_count_dec_and_test()
};
}

macro_rules! preempt_count_inc {
() => {
preempt_count_add!(1i32);
};
}

macro_rules! printk_safe_enter_irqsave {
($flags:expr) => {
loop {
local_irq_save!($flags);
unsafe { __printk_safe_enter() };
if true {
break;
}
}
};
}

macro_rules! printk_safe_exit_irqrestore {
($flags:expr) => {
loop {
unsafe { __printk_safe_exit() };
local_irq_restore!($flags);
if true {
break;
}
}
};
}

macro_rules! raw_check_bogus_irq_restore {
() => {
while false {
}
};
}

macro_rules! raw_local_irq_restore {
($flags:expr) => {
loop {
typecheck!(u64);
raw_check_bogus_irq_restore!();
arch_local_irq_restore($flags);
if true {
break;
}
}
};
}

macro_rules! raw_local_irq_save {
($flags:expr) => {
loop {
typecheck!(u64);
$flags = arch_local_irq_save();
if true {
break;
}
}
};
}

macro_rules! raw_local_irq_save__2 {
($flags:expr) => {
loop {
typecheck!(u64);
unsafe {
$flags = arch_local_irq_save();
}
if true {
break;
}
}
};
}

macro_rules! start_critical_timings {
() => {
while false {
}
};
}

macro_rules! static_branch_unlikely {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const1(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! stop_critical_timings {
() => {
while false {
}
};
}

macro_rules! task_thread_info {
($task:expr) => {
unsafe { core::ptr::addr_of_mut!((*$task).thread_info) }
};
}

macro_rules! test_bit__2 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_a39e427d3b6c2df4($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_bit {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if false && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! trace_set_current_state {
($state_value:expr) => {
loop {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_sched_set_state_tp.key) }) != 0 {
unsafe { __trace_set_current_state($state_value) };
}
if true {
break;
}
}
};
}

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck {
($type:ty) => {
let mut __v: i32;
let mut __dummy: $type = 0;
let mut __dummy2: $type = 0;
core::ptr::addr_of_mut!(__dummy) == core::ptr::addr_of_mut!(__dummy2);
__v = 1;
};
}

macro_rules! unlikely {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely__2 {
($x:expr) => {
!(!$x) as i64
};
}

unsafe extern "C" {
    static mut this_cpu_off: u64;
    static mut __cpu_online_mask: cpumask;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut cpu_number: i32;
    static mut __tracepoint_sched_set_state_tp: tracepoint;
    static mut __SCK__cond_resched: static_call_key;
    static mut sighand_cachep: *mut kmem_cache;
    static mut console_list: hlist_head;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut prb: *mut printk_ringbuffer;
    static mut printk_kthreads_running: bool;
    static mut printk_kthreads_ready: bool;
    static mut have_boot_console: bool;
    static mut have_nbcon_console: bool;
    static mut have_legacy_console: bool;
    static mut legacy_allow_panic_sync: bool;
    static mut console_irqwork_blocked: bool;
    static mut printk_shared_pbufs: printk_buffers;
    fn panic_in_progress() -> bool;
    fn panic_on_this_cpu() -> bool;
    fn panic_on_other_cpu() -> bool;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __xchg_wrong_size();
    fn __cmpxchg_wrong_size();
    fn __xadd_wrong_size();
    fn ____wrong_branch_error() -> bool;
    fn __bad_size_call_parameter();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __cond_resched() -> i32;
    fn __compiletime_assert_59();
    fn schedule();
    fn __trace_set_current_state(arg0: i32);
    fn sched_set_normal(arg0: *mut task_struct, arg1: i32);
    fn wake_up_process(arg0: *mut task_struct) -> i32;
    fn __SCT__cond_resched() -> i32;
    fn rcuwait_wake_up(arg0: *mut rcuwait) -> i32;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_344();
    fn __compiletime_assert_347();
    fn __compiletime_assert_348();
    fn __compiletime_assert_349();
    fn __compiletime_assert_350();
    fn finish_rcuwait(arg0: *mut rcuwait);
    fn irq_work_queue(arg0: *mut irq_work) -> bool;
    fn irq_work_sync(arg0: *mut irq_work);
    fn console_srcu_read_lock() -> i32;
    fn console_srcu_read_unlock(arg0: i32);
    fn __compiletime_assert_353();
    fn console_trylock() -> i32;
    fn console_unlock();
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn kthread_create_on_node(arg0: Option<unsafe extern "C" fn(*mut ()) -> i32>, arg1: *mut (), arg2: i32, arg3: *const u8, ...) -> *mut task_struct;
    fn kthread_stop(arg0: *mut task_struct) -> i32;
    fn kthread_should_stop() -> bool;
    fn kfree(arg0: *const ());
    fn __compiletime_assert_360();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __printk_safe_enter();
    fn __printk_safe_exit();
    fn printk_percpu_data_ready() -> bool;
    fn defer_console_output();
    fn is_printk_legacy_deferred() -> bool;
    fn console_lock_spinning_enable();
    fn console_lock_spinning_disable_and_check(arg0: i32) -> i32;
    fn printk_get_next_message(arg0: *mut printk_message, arg1: u64, arg2: bool, arg3: bool) -> bool;
    fn console_prepend_dropped(arg0: *mut printk_message, arg1: u64);
    fn console_prepend_replay(arg0: *mut printk_message);
    fn prb_read_valid(arg0: *mut printk_ringbuffer, arg1: u64, arg2: *mut printk_record) -> bool;
    fn prb_first_valid_seq(arg0: *mut printk_ringbuffer) -> u64;
    fn prb_next_reserve_seq(arg0: *mut printk_ringbuffer) -> u64;
    fn __compiletime_assert_390();
    fn __compiletime_assert_391();
    fn __compiletime_assert_395();
    fn __compiletime_assert_397();
    fn __compiletime_assert_400();
    fn __compiletime_assert_402();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_f217c4720f04f6db(arg0: u64) -> bool;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_types_h_190 {
    counter: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_types_h_197 {
    counter: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_head {
    next: *mut list_head,
    prev: *mut list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_head {
    first: *mut hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_node {
    next: *mut hlist_node,
    pprev: *mut *mut hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct callback_head {
    next: *mut callback_head,
    func: Option<unsafe extern "C" fn(*mut callback_head)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcuwait {
    task: *mut task_struct,
}


fn kasan_check_read(p: *const (), size: u32) -> bool {
true
}

fn kasan_check_write(p: *const (), size: u32) -> bool {
true
}

fn kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
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


fn __kcsan_enable_current() {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_qspinlock_types_h_44 {
    val: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_qspinlock_types_h_44_2,
    __slate_anon_2: __SlateRecord_qspinlock_types_h_44_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_qspinlock_types_h_44_2 {
    locked: u8,
    pending: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_qspinlock_types_h_44_3 {
    locked_pending: u16,
    tail: u16,
}


fn __kcsan_disable_current() {
}

// kcsan_check_read - check regular read access for races
//
// @ptr: address of access
// @size: size of access
// kcsan_check_write - check regular write access for races
//
// @ptr: address of access
// @size: size of access
// ASSERT_EXCLUSIVE_WRITER - assert no concurrent writes to @var
//
// Assert that there are no concurrent writes to @var; other readers are
// allowed. This assertion can be used to specify properties of concurrent code,
// where violation cannot be detected as a normal data race.
//
// For example, if we only have a single writer, but multiple concurrent
// readers, to avoid data races, all these accesses must be marked; even
// concurrent marked writes racing with the single writer are bugs.
// Unfortunately, due to being marked, they are no longer data races. For cases
// like these, we can use the macro as follows:
//
// .. code-block:: c
//
// 	void writer(void) {
// 		spin_lock(&update_foo_lock);
// 		ASSERT_EXCLUSIVE_WRITER(shared_foo);
// 		WRITE_ONCE(shared_foo, ...);
// 		spin_unlock(&update_foo_lock);
// 	}
// 	void reader(void) {
// 		// update_foo_lock does not need to be held!
// 		... = READ_ONCE(shared_foo);
// 	}
//
// Note: ASSERT_EXCLUSIVE_WRITER_SCOPED(), if applicable, performs more thorough
// checking if a clear scope where no concurrent writes are expected exists.
//
// @var: variable to assert on
// ASSERT_EXCLUSIVE_ACCESS - assert no concurrent accesses to @var
//
// Assert that there are no concurrent accesses to @var (no readers nor
// writers). This assertion can be used to specify properties of concurrent
// code, where violation cannot be detected as a normal data race.
//
// For example, where exclusive access is expected after determining no other
// users of an object are left, but the object is not actually freed. We can
// check that this property actually holds as follows:
//
// .. code-block:: c
//
// 	if (refcount_dec_and_test(&obj->refcnt)) {
// 		ASSERT_EXCLUSIVE_ACCESS(*obj);
// 		do_some_cleanup(obj);
// 		release_for_reuse(obj);
// 	}
//
// Note:
//
// 1. ASSERT_EXCLUSIVE_ACCESS_SCOPED(), if applicable, performs more thorough
//    checking if a clear scope where no concurrent accesses are expected exists.
//
// 2. For cases where the object is freed, `KASAN <kasan.html>`_ is a better
//    fit to detect use-after-free bugs.
//
// @var: variable to assert on
// ASSERT_EXCLUSIVE_BITS - assert no concurrent writes to subset of bits in @var
//
// Bit-granular variant of ASSERT_EXCLUSIVE_WRITER().
//
// Assert that there are no concurrent writes to a subset of bits in @var;
// concurrent readers are permitted. This assertion captures more detailed
// bit-level properties, compared to the other (word granularity) assertions.
// Only the bits set in @mask are checked for concurrent modifications, while
// ignoring the remaining bits, i.e. concurrent writes (or reads) to ~mask bits
// are ignored.
//
// Use this for variables, where some bits must not be modified concurrently,
// yet other bits are expected to be modified concurrently.
//
// For example, variables where, after initialization, some bits are read-only,
// but other bits may still be modified concurrently. A reader may wish to
// assert that this is true as follows:
//
// .. code-block:: c
//
// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
// 	foo = (READ_ONCE(flags) & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
//
// Note: The access that immediately follows ASSERT_EXCLUSIVE_BITS() is assumed
// to access the masked bits only, and KCSAN optimistically assumes it is
// therefore safe, even in the presence of data races, and marking it with
// READ_ONCE() is optional from KCSAN's point-of-view. We caution, however, that
// it may still be advisable to do so, since we cannot reason about all compiler
// optimizations when it comes to bit manipulations (on the reader and writer
// side). If you are sure nothing can go wrong, we can write the above simply
// as:
//
// .. code-block:: c
//
// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
// 	foo = (flags & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
//
// Another example, where this may be used, is when certain bits of @var may
// only be modified when holding the appropriate lock, but other bits may still
// be modified concurrently. Writers, where other bits may change concurrently,
// could use the assertion as follows:
//
// .. code-block:: c
//
// 	spin_lock(&foo_lock);
// 	ASSERT_EXCLUSIVE_BITS(flags, FOO_MASK);
// 	old_flags = flags;
// 	new_flags = (old_flags & ~FOO_MASK) | (new_foo << FOO_SHIFT);
// 	if (cmpxchg(&flags, old_flags, new_flags) != old_flags) { ... }
// 	spin_unlock(&foo_lock);
//
// @var: variable to assert on
// @mask: only check for modifications to bits set in @mask
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
fn __must_check_overflow(overflow: bool) -> bool {
unlikely__2!(overflow) != 0
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


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
#[inline(always)]
fn size_mul(factor1: u64, factor2: u64) -> u64 {
let mut bytes: u64 = 0;
if check_mul_overflow!(factor1, factor2, core::ptr::addr_of_mut!(bytes)) {
return !0;
}
bytes
}

#[repr(C)]
#[derive(Clone, Copy)]
struct qspinlock {
    __slate_anon_0: __SlateRecord_qspinlock_types_h_44,
}


// Bitfields in the atomic value:
//
// When NR_CPUS < 16K
//  0- 7: locked byte
//  8-15: pending byte
// 16-17: tail index
// 18-31: tail cpu (+1)
//
// When NR_CPUS >= 16K
//  0- 7: locked byte
//     8: pending
//  9-10: tail index
// 11-31: tail cpu (+1)
#[repr(C)]
#[derive(Clone, Copy)]
struct raw_spinlock {
    raw_lock: qspinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ratelimit_state {
    lock: raw_spinlock,
    interval: i32,
    burst: i32,
    rs_n_left: __SlateRecord_types_h_190,
    missed: __SlateRecord_types_h_190,
    flags: u32,
    begin: u64,
}


#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn arch_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
__WRITE_ONCE__3!((*v).counter, i);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
}


#[inline(always)]
fn arch_atomic_inc(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock incl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic_add_return(i: i32, mut v: *mut __SlateRecord_types_h_190) -> i32 {
let mut __v0: i32;
let mut __ret: i32 = i;
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
unsafe {
let __t0: u32;
core::arch::asm!("lock xaddb {0:l}, ({1:r})", inlateout(reg) __ret as u32 => __t0, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
}
2 => {
unsafe {
let __t1: u32;
core::arch::asm!("lock xaddw {0:x}, ({1:r})", inlateout(reg) __ret as u32 => __t1, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
}
4 => {
unsafe {
let __t2: u32;
core::arch::asm!("lock xaddl {0:e}, ({1:r})", inlateout(reg) __ret as u32 => __t2, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
}
}
8 => {
unsafe {
let __t3: u32;
core::arch::asm!("lock xaddq {0:r}, ({1:r})", inlateout(reg) __ret as u32 => __t3, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
}
}
_ => {
unsafe { __xadd_wrong_size() };
}
}
__v0 = __ret;
i.wrapping_add(__v0)
}

#[inline(always)]
fn arch_atomic_try_cmpxchg(v: *mut __SlateRecord_types_h_190, _old: *mut i32, mut __new: i32) -> bool {
let mut __v0: i64;
let mut success: bool = false;
let mut __old: i32 = unsafe { *_old };
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u8;
unsafe {
let __t0: u8;
let __t1: u32;
core::arch::asm!("lock cmpxchgb {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t0, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t1, options(nostack, att_syntax));
success = (__t0 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
}
2 => {
let mut __ptr: *mut u16 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u16;
unsafe {
let __t2: u8;
let __t3: u32;
core::arch::asm!("lock cmpxchgw {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t2, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t3, options(nostack, att_syntax));
success = (__t2 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
}
}
4 => {
let mut __ptr: *mut u32 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u32;
unsafe {
let __t4: u8;
let __t5: u32;
core::arch::asm!("lock cmpxchgl {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t4, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t5, options(nostack, att_syntax));
success = (__t4 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t5 as u32) };
}
}
8 => {
let mut __ptr: *mut u64 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u64;
unsafe {
let __t6: u8;
let __t7: u32;
core::arch::asm!("lock cmpxchgq {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t6, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t7, options(nostack, att_syntax));
success = (__t6 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t7 as u32) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
if unlikely__2!(!success) != 0 {
unsafe {
*_old = __old;
}
}
__v0 = success as i64;
__v0 != 0
}

#[inline(always)]
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE__2!((*v).counter)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct file {
}


#[inline(always)]
fn arch_atomic64_set(v: *mut __SlateRecord_types_h_197, i: i64) {
__WRITE_ONCE__11!((*v).counter, i);
}

#[inline(always)]
fn arch_atomic64_try_cmpxchg(v: *mut __SlateRecord_types_h_197, _old: *mut i64, mut __new: i64) -> bool {
let mut __v0: i64;
let mut success: bool = false;
let mut __old: i64 = unsafe { *_old };
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u8;
unsafe {
let __t0: u8;
let __t1: u64;
core::arch::asm!("lock cmpxchgb {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t0, in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t1, options(nostack, att_syntax));
success = (__t0 as u8) != 0;
__old = unsafe { core::mem::transmute::<u64, i64>(__t1 as u64) };
}
}
2 => {
let mut __ptr: *mut u16 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u16;
unsafe {
let __t2: u8;
let __t3: u64;
core::arch::asm!("lock cmpxchgw {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t2, in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t3, options(nostack, att_syntax));
success = (__t2 as u8) != 0;
__old = unsafe { core::mem::transmute::<u64, i64>(__t3 as u64) };
}
}
4 => {
let mut __ptr: *mut u32 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u32;
unsafe {
let __t4: u8;
let __t5: u64;
core::arch::asm!("lock cmpxchgl {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t4, in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t5, options(nostack, att_syntax));
success = (__t4 as u8) != 0;
__old = unsafe { core::mem::transmute::<u64, i64>(__t5 as u64) };
}
}
8 => {
let mut __ptr: *mut u64 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u64;
unsafe {
let __t6: u8;
let __t7: u64;
core::arch::asm!("lock cmpxchgq {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t6, in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t7, options(nostack, att_syntax));
success = (__t6 as u8) != 0;
__old = unsafe { core::mem::transmute::<u64, i64>(__t7 as u64) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
if unlikely__2!(!success) != 0 {
unsafe {
*_old = __old;
}
}
__v0 = success as i64;
__v0 != 0
}

#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[inline(always)]
fn raw_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
arch_atomic_set(v, i);
}

#[inline(always)]
fn raw_atomic_sub_return(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_sub_return!(i, v)
}

#[inline(always)]
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn raw_atomic_dec_return(v: *mut __SlateRecord_types_h_190) -> i32 {
raw_atomic_sub_return(1i32, v)
}

#[inline(always)]
fn raw_atomic_try_cmpxchg(v: *mut __SlateRecord_types_h_190, old: *mut i32, new: i32) -> bool {
arch_atomic_try_cmpxchg(v, old, new)
}

#[inline(always)]
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[inline(always)]
fn raw_atomic64_set(v: *mut __SlateRecord_types_h_197, i: i64) {
arch_atomic64_set(v, i);
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_restart_block_h_29 {
    futex: __SlateRecord_restart_block_h_31,
    nanosleep: __SlateRecord_restart_block_h_40,
    poll: __SlateRecord_restart_block_h_50,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_restart_block_h_31 {
    uaddr: *mut u32,
    val: u32,
    flags: u32,
    bitset: u32,
    time: i64,
    uaddr2: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_restart_block_h_40 {
    clockid: i32,
    r#type: u32,
    __slate_anon_2: __SlateRecord_restart_block_h_43,
    expires: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_restart_block_h_43 {
    rmtp: *mut __kernel_timespec,
    compat_rmtp: *mut old_timespec32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_restart_block_h_50 {
    ufds: *mut pollfd,
    nfds: i32,
    has_timeout: i32,
    end_time: timespec64,
}


#[inline(always)]
fn raw_atomic64_try_cmpxchg(v: *mut __SlateRecord_types_h_197, old: *mut i64, new: i64) -> bool {
arch_atomic64_try_cmpxchg(v, old, new)
}

#[inline(always)]
fn raw_atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
raw_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic_long_set(v: *mut __SlateRecord_types_h_197, i: i64) {
raw_atomic64_set(v, i);
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[inline(always)]
fn raw_atomic_long_try_cmpxchg(v: *mut __SlateRecord_types_h_197, old: *mut i64, new: i64) -> bool {
raw_atomic64_try_cmpxchg(v, old as *mut i64, new)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list_head {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[inline(always)]
fn instrument_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_read_write!(v, size);
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
if unlikely!(__ret_warn_on) != 0 {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[inline(always)]
fn instrument_atomic_read(v: *const (), size: u64) {
kasan_check_read(v, size as u32);
kcsan_check_atomic_read!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[inline(always)]
fn instrument_atomic_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_write!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[inline(always)]
fn instrument_atomic_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_read_write!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


#[inline(always)]
fn atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_read(v)
}

#[inline(always)]
fn atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set(v, i);
}

#[inline(always)]
fn atomic_inc(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc(v);
}

#[inline(always)]
fn atomic_dec_return(v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec_return(v)
}

#[inline(always)]
fn atomic_try_cmpxchg(v: *mut __SlateRecord_types_h_190, old: *mut i32, new: i32) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
instrument_read_write(old as *const (), core::mem::size_of::<i32>() as u64);
raw_atomic_try_cmpxchg(v, old, new)
}

#[inline(always)]
fn atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_read(v)
}

#[inline(always)]
fn atomic_long_set(v: *mut __SlateRecord_types_h_197, i: i64) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_set(v, i);
}

#[inline(always)]
fn atomic_long_try_cmpxchg(v: *mut __SlateRecord_types_h_197, old: *mut i64, new: i64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
instrument_read_write(old as *const (), core::mem::size_of::<i64>() as u64);
raw_atomic_long_try_cmpxchg(v, old, new)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llist_head {
    first: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_node {
    next: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __call_single_node {
    llist: llist_node,
    __slate_anon_1: __SlateRecord_smp_types_h_60,
    src: u16,
    dst: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_work {
    node: __call_single_node,
    func: Option<unsafe extern "C" fn(*mut irq_work)>,
    irqwait: rcuwait,
}


#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
}


#[inline(always)]
fn arch_static_branch__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_sched_set_state_tp, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_false {
    key: static_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_39 {
    _pid: i32,
    _uid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_45 {
    _tid: i32,
    _overrun: i32,
    _sigval: sigval,
    _sys_private: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_53 {
    _pid: i32,
    _uid: u32,
    _sigval: sigval,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_60 {
    _pid: i32,
    _uid: u32,
    _status: i32,
    _utime: i64,
    _stime: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_69 {
    _addr: *mut (),
    __slate_anon_1: __SlateRecord_siginfo_h_74,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_siginfo_h_74 {
    _trapno: i32,
    _addr_lsb: i16,
    _addr_bnd: __SlateRecord_siginfo_h_83,
    _addr_pkey: __SlateRecord_siginfo_h_89,
    _perf: __SlateRecord_siginfo_h_94,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_83 {
    _dummy_bnd: [u8; 8],
    _lower: *mut (),
    _upper: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_89 {
    _dummy_pkey: [u8; 8],
    _pkey: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_94 {
    _data: u64,
    _type: u32,
    _flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_103 {
    _band: i64,
    _fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_109 {
    _call_addr: *mut (),
    _syscall: i32,
    _arch: u32,
}


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
// Declaration/definition used for per-CPU variables that must be cacheline
// aligned under SMP conditions so that, whilst a particular instance of the
// data corresponds to a particular CPU, inefficiencies due to direct access by
// other CPUs are reduced by preventing the data from unnecessarily spanning
// cachelines.
//
// An example of this would be statistical data, where each CPU's set of data
// is updated by that CPU alone, but the data from across all CPUs is collated
// by a CPU processing a read from a proc file.
// Declaration/definition used for per-CPU variables that must be page aligned.
// Declaration/definition used for per-CPU variables that must be read mostly.
// Add an offset to a pointer.  Use RELOC_HIDE() to prevent the compiler
// from making incorrect assumptions about the pointer value.
#[inline(always)]
fn __this_cpu_preempt_check(op: *const u8) {
}

// Operations for contexts where we do not want to do any checks for
// preemptions.  Unless strictly necessary, always use [__]this_cpu_*()
// instead.
//
// If there is no other protection through preempt disable and/or disabling
// interrupts then one of these RMW operations can show unexpected behavior
// because the execution thread was rescheduled on another processor or an
// interrupt occurred and the same percpu variable was modified from the
// interrupt context.
// Operations for contexts that are safe from preemption/interrupts.  These
// operations verify that preemption is disabled.
// Operations with implied preemption/interrupt protection.  These
// operations can be used without worrying about preemption or interrupt.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_types_h_14 {
    si_signo: i32,
    si_errno: i32,
    si_code: i32,
    _sifields: __sifields,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ucounts {
}


#[inline(always)]
fn native_save_fl() -> u64 {
let mut flags: u64 = 0;
// "=rm" is safe here, because "pop" adjusts the stack before
// it evaluates its effective address -- this is part of the
// documented behavior of the "pop" instruction.
unsafe {
let __t0: u64;
core::arch::asm!("# __raw_save_flags\n\tpushf ; pop {0:r}", lateout(reg) __t0, options(nostack, att_syntax));
flags = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
// no input
flags
}

#[inline(always)]
fn native_irq_disable() {
unsafe {
core::arch::asm!("cli", options(nostack, att_syntax));
}
}

#[inline(always)]
fn native_irq_enable() {
unsafe {
core::arch::asm!("sti", options(nostack, att_syntax));
}
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits81U0,
}


// was written to (raised by CPU)
// on 4KB pages
// On 2MB or 1GB pages
// Protection Keys, bit 1/4
// Saved Dirty bit (leaf)
// The hardware requires shadow stack to be Write=0,Dirty=1. However,
// there are valid cases where the kernel might create read-only PTEs that
// are dirty (e.g., fork(), mprotect(), userfaultfd, soft-dirty tracking). In
// this case, the _PAGE_SAVED_DIRTY bit is used instead of the HW-dirty bit,
// to avoid creating a wrong "shadow stack" PTEs. Such PTEs have
// (Write=0,SavedDirty=1,Dirty=0) set.
// Extracts the PFN from a (pte|pmd|pud|pgd)val_t of a 4KB page
// Extracts the flags from a (pte|pmd|pud|pgd)val_t
// This includes the protection key value.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_295 {
    pgd: u64,
}


// No need to mask any bits for !PAE
#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[inline(always)]
fn constant_test_bit(nr: i64, addr: *const u64) -> bool {
1u64 << (nr & 63i64) & unsafe { core::ptr::read_volatile(addr.wrapping_offset((nr >> 6i32) as isize)) } != 0
}

#[inline(always)]
fn variable_test_bit(mut nr: i64, mut addr: *const u64) -> bool {
let mut oldbit: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!(" btq  {2:r},({1:r})\nsetc {0}", lateout(reg_byte) __t0, in(reg) addr as *mut u64, in(reg) nr as u64, options(nostack, att_syntax));
oldbit = (__t0 as u8) != 0;
}
oldbit
}

#[inline(always)]
fn arch_test_bit(nr: u64, addr: *const u64) -> bool {
let __v0: i32 = if unsafe { __slate_intrinsic_f217c4720f04f6db(nr) } {
constant_test_bit(nr as i64, addr) as i32
} else {
variable_test_bit(nr as i64, addr) as i32
};
__v0 != 0
}

// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cpumask {
    bits: [u64; 1],
}


// cpumask_bits - get the bits in a cpumask
// @maskp: the struct cpumask *
//
// You should only assume nr_cpu_ids bits of this mask are valid.  This is
// a macro so it's const-correct.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


fn __might_resched(file: *const u8, line: i32, offsets: u32) {
}

#[inline(always)]
fn cpu_max_bits_warn(cpu: u32, bits: u32) {
}

#[inline(always)]
fn cpumask_check(cpu: u32) -> u32 {
cpu_max_bits_warn(cpu, 64u32);
cpu
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


// for_each_cpu - iterate over every cpu in a mask
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
//
// After the loop, cpu is >= nr_cpu_ids.
#[inline(always)]
fn cpumask_test_cpu(cpu: i32, cpumask: *const cpumask) -> bool {
test_bit!(cpumask_check(cpu as u32), unsafe { (*cpumask).bits.as_ptr() as *const u64 }) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_38 {
    all: u64,
    __slate_anon_1: __SlateRecord_rseq_types_h_40,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_40 {
    __slate_anon_0: __SlateRecord_rseq_types_h_41,
    has_rseq: u8,
    __pad: u8,
    __slate_anon_3: __SlateRecord_rseq_types_h_52,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_41 {
    events: u32,
    __slate_anon_1: __SlateRecord_rseq_types_h_43,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_43 {
    sched_switch: u8,
    ids_changed: u8,
    user_irq: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_52 {
    error: u16,
    __slate_anon_1: __SlateRecord_rseq_types_h_54,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_54 {
    fatal: u8,
    slowpath: u8,
}


// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
#[inline(always)]
fn cpu_online(cpu: u32) -> bool {
cpumask_test_cpu(cpu as i32, (unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask)
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_74 {
    cpu_cid: u64,
    __slate_anon_1: __SlateRecord_rseq_types_h_76,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_76 {
    cpu_id: u32,
    mm_cid: u32,
}


#[inline(always)]
fn arch_local_save_flags() -> u64 {
native_save_fl()
}

#[inline(always)]
fn arch_local_irq_disable() {
native_irq_disable();
}

#[inline(always)]
fn arch_local_irq_enable() {
native_irq_enable();
}

#[inline(always)]
fn arch_local_irq_save() -> u64 {
let flags: u64 = arch_local_save_flags();
arch_local_irq_disable();
flags
}

#[inline(always)]
fn arch_irqs_disabled_flags(flags: u64) -> i32 {
(flags & _BITUL!(9i32) == 0) as i32
}

#[inline(always)]
fn arch_local_irq_restore(flags: u64) {
if arch_irqs_disabled_flags(flags) == 0 {
arch_local_irq_enable();
}
}

static class_irqsave_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_irqflags_h_266 {
    lock: *mut (),
    flags: u64,
}


#[inline(always)]
fn class_irqsave_destructor(_T: *mut __SlateRecord_irqflags_h_266) {
local_irq_restore!(unsafe { (*_T).flags });
}

#[inline(always)]
fn class_irqsave_lock_ptr(_T: &mut __SlateRecord_irqflags_h_266) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_irqflags_h_266)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[inline(always)]
fn class_irqsave_constructor() -> __SlateRecord_irqflags_h_266 {
let mut _t: __SlateRecord_irqflags_h_266 = {
let mut __t0: __SlateRecord_irqflags_h_266 = unsafe { core::mem::zeroed() };
__t0.lock = 1i32 as *mut ();
__t0
};
let _T: *mut __SlateRecord_irqflags_h_266 = core::ptr::addr_of_mut!(_t);
local_irq_save__2!((*_T).flags);
return _t;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_59() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = core::ptr::null_mut::<task_struct>();
return __v0;
}
let mut __v2: *mut task_struct;
let mut pscr_ret__: *mut task_struct = unsafe { core::mem::zeroed() };
__verify_pcpu_ptr__2!();
match (const { core::mem::size_of::<*mut task_struct>() }) as u64 {
1 => {
let mut __v4: *mut task_struct;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v4 = (pfo_val__ as u64) as *mut task_struct;
pscr_ret__ = __v4;
}
2 => {
let mut __v6: *mut task_struct;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v6 = (pfo_val__ as u64) as *mut task_struct;
pscr_ret__ = __v6;
}
4 => {
let mut __v7: *mut task_struct;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v7 = (pfo_val__ as u64) as *mut task_struct;
pscr_ret__ = __v7;
}
8 => {
let mut __v8: *mut task_struct;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v8 = pfo_val__ as *mut task_struct;
pscr_ret__ = __v8;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v2 = pscr_ret__;
__v2
}

#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_plug {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_net_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct capture_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfs_rq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_task {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mempolicy {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nameidata {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nsproxy {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_ctx_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pipe_inode_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct reclaim_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
}


#[inline(always)]
fn native_pause() {
unsafe {
core::arch::asm!("pause", options(nostack, att_syntax));
}
}

#[inline(always)]
fn cpu_relax() {
native_pause();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct thread_struct {
    tls_array: [desc_struct; 3],
    sp: u64,
    es: u16,
    ds: u16,
    fsindex: u16,
    gsindex: u16,
    fsbase: u64,
    gsbase: u64,
    ptrace_bps: [*mut perf_event; 4],
    virtual_dr6: u64,
    ptrace_dr7: u64,
    cr2: u64,
    trap_nr: u64,
    error_code: u64,
    io_bitmap: *mut io_bitmap,
    iopl_emul: u64,
    __slate_bits_0: __slate_bits::__SlateBits88U0,
    pkru: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_delay_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_exec_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timespec64 {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct restart_block {
    arch_data: u64,
    r#fn: Option<unsafe extern "C" fn(*mut restart_block) -> i64>,
    __slate_anon_2: __SlateRecord_restart_block_h_29,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_info {
    flags: u64,
    syscall_work: u64,
    status: u32,
    cpu: u32,
}


fn test_ti_thread_flag(ti: *mut thread_info, flag: i32) -> i32 {
test_bit__2!(flag, unsafe { core::ptr::addr_of_mut!((*ti).flags) })
}

// We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
#[inline(always)]
fn __preempt_count_add(val: i32) {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_a39e427d3b6c2df4(val) }) && (val == 1 || val == 1i32.wrapping_neg()) { val } else { 0 };
if false {
let pao_tmp__: u64 = val as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:{0}(%rip)", sym __preempt_count, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:{0}(%rip)", sym __preempt_count, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = val as u64;
if false {
let pto_tmp__: u64 = val as u64;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:{0}(%rip)", sym __preempt_count, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}

#[inline(always)]
fn __preempt_count_dec_and_test() -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("decq %gs:{0}(%rip)\nsete {1}", sym __preempt_count, lateout(reg_byte) __t0, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex {
    owner: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    osq: optimistic_spin_queue,
    first_waiter: *mut mutex_waiter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_sched_data {
    robust_list: *mut robust_list_head,
    compat_robust_list: *mut compat_robust_list_head,
    pi_state_list: list_head,
    pi_state_cache: *mut futex_pi_state,
    exit_mutex: mutex,
    state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sched_h_826 {
    ipi_mask_ptr: *mut cpumask,
    ipi_mask_val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_class {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_mm_phash {
    lock: mutex,
    hash: *mut futex_private_hash,
    hash_new: *mut futex_private_hash,
    batches: u64,
    rcu: callback_head,
    atomic: __SlateRecord_types_h_197,
    r#ref: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_unlock_cs_range {
    start_ip: u64,
    len: u32,
    pop_size32: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_unlock_cs_ranges {
    cs_ranges: [futex_unlock_cs_range; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_mm_data {
    phash: futex_mm_phash,
    unlock: futex_unlock_cs_ranges,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct files_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_restriction {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rt_mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_set {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_sem {
    undo_list: *mut sem_undo_list,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_shm {
    shm_clist: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct plist_node {
    prio: i32,
    prio_list: list_head,
    node_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_node {
    __rb_parent_color: u64,
    rb_right: *mut rb_node,
    rb_left: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_node_linked {
    node: rb_node,
    prev: *mut rb_node_linked,
    next: *mut rb_node_linked,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root {
    rb_node: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_head {
    rb_root: rb_root_cached,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer {
    node: timerqueue_linked_node,
    base: *mut hrtimer_clock_base,
    is_queued: bool,
    is_rel: bool,
    is_soft: bool,
    is_hard: bool,
    is_lazy: bool,
    _softexpires: i64,
    function: Option<unsafe extern "C" fn(*mut hrtimer) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timer_list {
    entry: hlist_node,
    expires: u64,
    function: Option<unsafe extern "C" fn(*mut timer_list)>,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uprobes_h_31 {
    insn: [u8; 16],
    ixol: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uprobes_h_38 {
    branch: __SlateRecord_uprobes_h_39,
    defparam: __SlateRecord_uprobes_h_44,
    push: __SlateRecord_uprobes_h_48,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_39 {
    offs: i32,
    ilen: u8,
    opc1: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_44 {
    fixups: u8,
    ilen: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_48 {
    reg_offset: u8,
    ilen: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp {
    mode: i32,
    filter_count: __SlateRecord_types_h_190,
    filter: *mut seccomp_filter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uprobes_h_136 {
    __slate_anon_0: __SlateRecord_uprobes_h_137,
    __slate_anon_1: __SlateRecord_uprobes_h_142,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_137 {
    autask: arch_uprobe_task,
    vaddr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_142 {
    dup_xol_work: callback_head,
    dup_xol_addr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


#[repr(C)]
#[derive(Clone, Copy)]
union sigval {
    sival_int: i32,
    sival_ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __sifields {
    _kill: __SlateRecord_siginfo_h_39,
    _timer: __SlateRecord_siginfo_h_45,
    _rt: __SlateRecord_siginfo_h_53,
    _sigchld: __SlateRecord_siginfo_h_60,
    _sigfault: __SlateRecord_siginfo_h_69,
    _sigpoll: __SlateRecord_siginfo_h_103,
    _sigsys: __SlateRecord_siginfo_h_109,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vdso_image {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct address_space {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_89 {
    __slate_anon_0: __SlateRecord_mm_types_h_90,
    __slate_anon_1: __SlateRecord_mm_types_h_118,
    __slate_anon_2: __SlateRecord_mm_types_h_129,
    __slate_anon_3: __SlateRecord_mm_types_h_132,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_90 {
    __slate_anon_0: __SlateRecord_mm_types_h_96,
    mapping: *mut address_space,
    __slate_anon_2: __SlateRecord_mm_types_h_105,
    private: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_96 {
    lru: list_head,
    buddy_list: list_head,
    pcp_list: list_head,
    pcp_llist: llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_105 {
    __folio_index: u64,
    share: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_118 {
    pp_magic: u64,
    pp: *mut page_pool,
    _pp_mapping_pad: u64,
    dma_addr: u64,
    pp_ref_count: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct page_pool {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_129 {
    compound_info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_132 {
    _unused_pgmap_compound_info: *mut (),
    zone_device_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_155 {
    page_type: u32,
    _mapcount: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigaction {
    sa_handler: Option<unsafe extern "C" fn(i32)>,
    sa_flags: u64,
    sa_restorer: Option<unsafe extern "C" fn()>,
    sa_mask: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct k_sigaction {
    sa: sigaction,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kioctx_table {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_mm_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1176 {
    __slate_anon_0: __SlateRecord_mm_types_h_1181,
    mm_mt: maple_tree,
    mmap_base: u64,
    mmap_legacy_base: u64,
    mmap_compat_base: u64,
    mmap_compat_legacy_base: u64,
    task_size: u64,
    pgd: *mut __SlateRecord_pgtable_types_h_295,
    membarrier_state: __SlateRecord_types_h_190,
    mm_users: __SlateRecord_types_h_190,
    mm_cid: mm_mm_cid,
    sched_cache_grp: *mut sched_cache_group,
    pgtables_bytes: __SlateRecord_types_h_197,
    map_count: i32,
    page_table_lock: spinlock,
    mmap_lock: rw_semaphore,
    mmlist: list_head,
    vma_writer_wait: rcuwait,
    mm_lock_seq: seqcount,
    futex: futex_mm_data,
    hiwater_rss: u64,
    hiwater_vm: u64,
    total_vm: u64,
    locked_vm: u64,
    pinned_vm: __SlateRecord_types_h_197,
    data_vm: u64,
    exec_vm: u64,
    stack_vm: u64,
    __slate_anon_28: __SlateRecord_mm_types_h_1289,
    write_protect_seq: seqcount,
    arg_lock: spinlock,
    start_code: u64,
    end_code: u64,
    start_data: u64,
    end_data: u64,
    start_brk: u64,
    brk: u64,
    start_stack: u64,
    arg_start: u64,
    arg_end: u64,
    env_start: u64,
    env_end: u64,
    saved_auxv: [u64; 56],
    rss_stat: [percpu_counter; 4],
    binfmt: *mut linux_binfmt,
    context: __SlateRecord_mmu_h_84,
    flags: __SlateRecord_mm_types_h_1171,
    ioctx_lock: spinlock,
    ioctx_table: *mut kioctx_table,
    exe_file: *mut file,
    notifier_subscriptions: *mut mmu_notifier_subscriptions,
    tlb_flush_pending: __SlateRecord_types_h_190,
    tlb_flush_batched: __SlateRecord_types_h_190,
    uprobes_state: uprobes_state,
    hugetlb_usage: __SlateRecord_types_h_197,
    async_put_work: work_struct,
    iommu_mm: *mut iommu_mm_data,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1181 {
    mm_count: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_1289 {
    def_flags: u64,
    def_vma_flags: __SlateRecord_mm_types_h_871,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct linux_binfmt {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmu_notifier_subscriptions {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_func {
    func: *mut (),
    data: *mut (),
    prio: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dentry {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_ext {
    regfunc: Option<unsafe extern "C" fn() -> i32>,
    unregfunc: Option<unsafe extern "C" fn()>,
    __slate_bits_0: __slate_bits::__SlateBits177U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint {
    name: *const u8,
    key: static_key_false,
    static_call_key: *mut static_call_key,
    static_call_tramp: *mut (),
    iterator: *mut (),
    probestub: *mut (),
    funcs: *mut tracepoint_func,
    ext: *mut tracepoint_ext,
}


// If a tracepoint needs to be called from a header file, it is not
// recommended to call it directly, as tracepoints in header files
// may cause side-effects and bloat the kernel. Instead, use
// tracepoint_enabled() to test if the tracepoint is enabled, then if
// it is, call a wrapper function defined in a C file that will then
// call the tracepoint.
//
// For "trace_foo_bar()", you would need to create a wrapper function
// in a C file to call trace_foo_bar():
//   void do_trace_foo_bar(args) { trace_foo_bar(args); }
// Then in the header file, declare the tracepoint:
//   DECLARE_TRACEPOINT(foo_bar);
// And call your wrapper:
//   static inline void some_inlined_function() {
//            [..]
//            if (tracepoint_enabled(foo_bar))
//                    do_trace_foo_bar(args);
//            [..]
//   }
//
// Note: tracepoint_enabled(foo_bar) is equivalent to trace_foo_bar_enabled()
//   but is safe to have in headers, where trace_foo_bar_enabled() is not.
#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct syscall_user_dispatch {
    selector: *mut u8,
    offset: u64,
    len: u64,
    on_dispatch: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_tlbflush_unmap_batch {
    cpumask: cpumask,
    unmapped_pages: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_117 {
    __slate_anon_0: __SlateRecord_key_h_118,
    x: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_118 {
    desc_len: u16,
    desc: [u8; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct page_frag {
    page: *mut page,
    offset: u32,
    size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tlbflush_unmap_batch {
    arch: arch_tlbflush_unmap_batch,
    flush_required: bool,
    writable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_198 {
    graveyard_link: list_head,
    serial_node: rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_user {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_208 {
    expiry: i64,
    revoked_at: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_246 {
    index_key: keyring_index_key,
    __slate_anon_1: __SlateRecord_key_h_248,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_248 {
    hash: u64,
    len_desc: u64,
    r#type: *mut key_type,
    domain_tag: *mut key_tag,
    description: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_261 {
    payload: key_payload,
    __slate_anon_1: __SlateRecord_key_h_263,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_263 {
    name_link: list_head,
    keys: assoc_array,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_io_accounting {
    rchar: u64,
    wchar: u64,
    syscr: u64,
    syscw: u64,
    read_bytes: u64,
    write_bytes: u64,
    cancelled_write_bytes: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimers {
    bases: [posix_cputimer_base; 3],
    timers_active: u32,
    expiry_active: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimers_work {
    work: callback_head,
    mutex: mutex,
    scheduled: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct work_struct {
    data: __SlateRecord_types_h_197,
    entry: list_head,
    func: Option<unsafe extern "C" fn(*mut work_struct)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct taskstats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_console_h_216 {
    atom: u32,
    __slate_anon_1: __SlateRecord_console_h_218,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_console_h_218 {
    __slate_bits_0: __slate_bits::__SlateBits376U0,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_mm_cid {
    pcpu: *mut mm_cid_pcpu,
    mode: u32,
    max_cids: u32,
    irq_work: irq_work,
    work: work_struct,
    lock: raw_spinlock,
    mutex: mutex,
    user_list: hlist_head,
    nr_cpus_allowed: u32,
    users: u32,
    pcpu_thrs: u32,
    update_deferred: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount {
    sequence: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_driver {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_13 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unwind_cache {
    unwind_completed: u64,
    nr_entries: u32,
    entries: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unwind_task_info {
    unwind_mask: __SlateRecord_types_h_197,
    cache: *mut unwind_cache,
    work: callback_head,
    id: unwind_task_id,
}


// get_task_state():
// set_current_state() includes a barrier so that the write of current->__state
// is correctly serialised wrt the caller's subsequent test of whether to
// actually sleep:
//
//   for (;;) {
// 	set_current_state(TASK_UNINTERRUPTIBLE);
// 	if (CONDITION)
// 	   break;
//
// 	schedule();
//   }
//   __set_current_state(TASK_RUNNING);
//
// If the caller does not need such serialisation (because, for instance, the
// CONDITION test and condition change and wakeup are under the same lock) then
// use __set_current_state().
//
// The above is typically ordered against the wakeup, which does:
//
//   CONDITION = 1;
//   wake_up_state(p, TASK_UNINTERRUPTIBLE);
//
// where wake_up_state()/try_to_wake_up() executes a full memory barrier before
// accessing p->__state.
//
// Wakeup will do: if (@state & p->__state) p->__state = TASK_RUNNING, that is,
// once it observes the TASK_UNINTERRUPTIBLE store the waking CPU can issue a
// TASK_RUNNING store which can collide with __set_current_state(TASK_RUNNING).
//
// However, with slightly different timing the wakeup TASK_RUNNING store can
// also collide with the TASK_UNINTERRUPTIBLE store. Losing that store is not
// a problem either because that will result in one extra go around the loop
// and our @cond test will save the day.
//
// Also see the comments of try_to_wake_up().
// set_special_state() should be used for those states when the blocking task
// can not use the regular condition based wait-loop. In that case we must
// serialize against wakeups such that any possible in-flight TASK_RUNNING
// stores will not collide with our state change.
// may shadow
#[repr(C)]
#[derive(Clone, Copy)]
struct prev_cputime {
    utime: u64,
    stime: u64,
    lock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_info {
    pcount: u64,
    run_delay: u64,
    max_run_delay: u64,
    min_run_delay: u64,
    last_arrival: u64,
    last_queued: u64,
    max_run_delay_ts: timespec64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct load_weight {
    weight: u64,
    inv_weight: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sched_avg {
    last_update_time: u64,
    load_sum: u64,
    runnable_sum: u64,
    util_sum: u32,
    period_contrib: u32,
    load_avg: u64,
    runnable_avg: u64,
    util_avg: u64,
    util_est: u32,
}


mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits237U0 {
        #[bits(1)]
        pub dl_throttled: u32,
        #[bits(1)]
        pub dl_yielded: u32,
        #[bits(1)]
        pub dl_non_contending: u32,
        #[bits(1)]
        pub dl_overrun: u32,
        #[bits(1)]
        pub dl_server: u32,
        #[bits(1)]
        pub dl_server_active: u32,
        #[bits(1)]
        pub dl_defer: u32,
        #[bits(1)]
        pub dl_defer_armed: u32,
        #[bits(1)]
        pub dl_defer_running: u32,
        #[bits(1)]
        pub dl_defer_idle: u32,
        #[bits(1)]
        pub dl_bw_attached: u32,
        #[bits(5, access = na)]
        pub __slate_pad_11: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits51U0 {
        #[bits(1)]
        pub sched_reset_on_fork: u32,
        #[bits(1)]
        pub sched_contributes_to_load: u32,
        #[bits(1)]
        pub sched_migrated: u32,
        #[bits(1)]
        pub sched_task_hot: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits51U1 {
        #[bits(1)]
        pub sched_remote_wakeup: u32,
        #[bits(1)]
        pub sched_rt_mutex: u32,
        #[bits(1)]
        pub in_execve: u32,
        #[bits(1)]
        pub in_iowait: u32,
        #[bits(1)]
        pub restore_sigmask: u32,
        #[bits(1)]
        pub no_cgroup_migration: u32,
        #[bits(1)]
        pub frozen: u32,
        #[bits(1)]
        pub use_memdelay: u32,
        #[bits(1)]
        pub in_eventfd: u32,
        #[bits(1)]
        pub pasid_activated: u32,
        #[bits(1)]
        pub reported_split_lock: u32,
        #[bits(1)]
        pub in_thrashing: u32,
        #[bits(1)]
        pub in_nf_duplicate: u32,
        #[bits(3, access = na)]
        pub __slate_pad_13: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits239U0 {
        #[bits(1)]
        pub is_child_subreaper: u32,
        #[bits(1)]
        pub has_child_subreaper: u32,
        #[bits(1)]
        pub autoreap: u32,
        #[bits(1)]
        pub timer_create_restore_ids: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits51U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits81U0 {
        #[bits(8)]
        pub base1: u16,
        #[bits(4)]
        pub r#type: u16,
        #[bits(1)]
        pub s: u16,
        #[bits(2)]
        pub dpl: u16,
        #[bits(1)]
        pub p: u16,
        #[bits(4)]
        pub limit1: u16,
        #[bits(1)]
        pub avl: u16,
        #[bits(1)]
        pub l: u16,
        #[bits(1)]
        pub d: u16,
        #[bits(1)]
        pub g: u16,
        #[bits(8)]
        pub base2: u16,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits88U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits177U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits389U0 {
        #[bits(5)]
        pub flags: u8,
        #[bits(3)]
        pub level: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits379U0 {
        #[bits(1)]
        pub allow_unsafe_takeover: u32,
        #[bits(1)]
        pub backlog: u32,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits376U0 {
        #[bits(2)]
        pub prio: u32,
        #[bits(2)]
        pub req_prio: u32,
        #[bits(1)]
        pub r#unsafe: u32,
        #[bits(1)]
        pub unsafe_takeover: u32,
        #[bits(24)]
        pub cpu: u32,
        #[bits(2, access = na)]
        pub __slate_pad_5: u8,
    }
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sched_statistics {
    wait_start: u64,
    wait_max: u64,
    wait_count: u64,
    wait_sum: u64,
    iowait_count: u64,
    iowait_sum: u64,
    sleep_start: u64,
    sleep_max: u64,
    sum_sleep_runtime: i64,
    block_start: u64,
    block_max: u64,
    sum_block_runtime: i64,
    exec_max: i64,
    slice_max: u64,
    nr_failed_migrations_affine: u64,
    nr_failed_migrations_running: u64,
    nr_failed_migrations_hot: u64,
    nr_forced_migrations: u64,
    nr_wakeups: u64,
    nr_wakeups_sync: u64,
    nr_wakeups_migrate: u64,
    nr_wakeups_local: u64,
    nr_wakeups_remote: u64,
    nr_wakeups_affine: u64,
    nr_wakeups_affine_attempts: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_entity {
    load: load_weight,
    h_load: load_weight,
    run_node: rb_node,
    deadline: u64,
    min_vruntime: u64,
    min_slice: u64,
    max_slice: u64,
    group_node: list_head,
    on_rq: u8,
    sched_delayed: u8,
    rel_deadline: u8,
    custom_slice: u8,
    exec_start: u64,
    sum_exec_runtime: u64,
    prev_sum_exec_runtime: u64,
    vruntime: u64,
    vlag: i64,
    vprot: u64,
    slice: u64,
    nr_migrations: u64,
    depth: i32,
    parent: *mut sched_entity,
    cfs_rq: *mut cfs_rq,
    my_q: *mut cfs_rq,
    runnable_weight: u64,
    avg: sched_avg,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_rt_entity {
    run_list: list_head,
    timeout: u64,
    watchdog_stamp: u64,
    time_slice: u32,
    on_rq: u16,
    on_list: u16,
    back: *mut sched_rt_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_dl_entity {
    rb_node: rb_node,
    dl_runtime: u64,
    dl_deadline: u64,
    dl_period: u64,
    dl_bw: u64,
    dl_density: u64,
    runtime: i64,
    deadline: u64,
    flags: u32,
    __slate_bits_0: __slate_bits::__SlateBits237U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
}


// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
#[repr(C)]
#[derive(Clone, Copy)]
struct wake_q_node {
    next: *mut wake_q_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmap_ctrl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_struct {
    thread_info: thread_info,
    __state: u32,
    saved_state: u32,
    stack: *mut (),
    usage: refcount_struct,
    flags: u32,
    ptrace: u32,
    on_cpu: u8,
    on_rq: u8,
    is_blocked: u8,
    __pad: u8,
    wake_entry: __call_single_node,
    wakee_flips: u32,
    wakee_flip_decay_ts: u64,
    last_wakee: *mut task_struct,
    recent_used_cpu: i32,
    wake_cpu: i32,
    prio: i32,
    static_prio: i32,
    normal_prio: i32,
    rt_priority: u32,
    se: sched_entity,
    rt: sched_rt_entity,
    dl: sched_dl_entity,
    dl_server: *mut sched_dl_entity,
    sched_class: *const sched_class,
    sched_task_group: *mut task_group,
    stats: sched_statistics,
    btrace_seq: u32,
    policy: u32,
    max_allowed_capacity: u64,
    nr_cpus_allowed: i32,
    cpus_ptr: *const cpumask,
    user_cpus_ptr: *mut cpumask,
    cpus_mask: cpumask,
    migration_pending: *mut (),
    migration_disabled: u16,
    migration_flags: u16,
    rcu_read_lock_nesting: i32,
    rcu_read_unlock_special: rcu_special,
    rcu_node_entry: list_head,
    rcu_blocked_node: *mut rcu_node,
    rcu_tasks_nvcsw: u64,
    rcu_tasks_holdout: u8,
    rcu_tasks_idx: u8,
    rcu_tasks_idle_cpu: i32,
    rcu_tasks_holdout_list: list_head,
    rcu_tasks_exit_cpu: i32,
    rcu_tasks_exit_list: list_head,
    trc_reader_nesting: i32,
    trc_reader_scp: *mut srcu_ctr,
    sched_info: sched_info,
    tasks: list_head,
    pushable_tasks: plist_node,
    pushable_dl_tasks: rb_node,
    mm: *mut mm_struct,
    active_mm: *mut mm_struct,
    exec_state: *mut task_exec_state,
    exit_state: i32,
    exit_code: i32,
    exit_signal: i32,
    pdeath_signal: i32,
    jobctl: u64,
    personality: u32,
    __slate_bits_0: __slate_bits::__SlateBits51U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits51U1,
    atomic_flags: u64,
    restart_block: restart_block,
    pid: i32,
    tgid: i32,
    stack_canary: u64,
    real_parent: *mut task_struct,
    parent: *mut task_struct,
    children: list_head,
    sibling: list_head,
    group_leader: *mut task_struct,
    ptraced: list_head,
    ptrace_entry: list_head,
    thread_pid: *mut pid,
    pid_links: [hlist_node; 4],
    thread_node: list_head,
    vfork_done: *mut completion,
    set_child_tid: *mut i32,
    clear_child_tid: *mut i32,
    worker_private: *mut (),
    utime: u64,
    stime: u64,
    gtime: u64,
    prev_cputime: prev_cputime,
    nvcsw: u64,
    nivcsw: u64,
    start_time: u64,
    start_boottime: u64,
    min_flt: u64,
    maj_flt: u64,
    posix_cputimers: posix_cputimers,
    posix_cputimers_work: posix_cputimers_work,
    ptracer_cred: *const cred,
    real_cred: *const cred,
    cred: *const cred,
    cached_requested_key: *mut key,
    comm: [u8; 16],
    nameidata: *mut nameidata,
    sysvsem: sysv_sem,
    sysvshm: sysv_shm,
    real_fs: *mut fs_struct,
    fs: *mut fs_struct,
    files: *mut files_struct,
    io_uring: *mut io_uring_task,
    io_uring_restrict: *mut io_restriction,
    nsproxy: *mut nsproxy,
    signal: *mut signal_struct,
    sighand: *mut sighand_struct,
    blocked: __SlateRecord_signal_h_25,
    real_blocked: __SlateRecord_signal_h_25,
    saved_sigmask: __SlateRecord_signal_h_25,
    pending: sigpending,
    sas_ss_sp: u64,
    sas_ss_size: u64,
    sas_ss_flags: u32,
    task_works: *mut callback_head,
    audit_context: *mut audit_context,
    loginuid: __SlateRecord_uidgid_types_h_9,
    sessionid: u32,
    seccomp: seccomp,
    syscall_dispatch: syscall_user_dispatch,
    parent_exec_id: u64,
    self_exec_id: u64,
    alloc_lock: spinlock,
    pi_lock: raw_spinlock,
    wake_q: wake_q_node,
    pi_waiters: rb_root_cached,
    pi_top_task: *mut task_struct,
    pi_blocked_on: *mut rt_mutex_waiter,
    blocked_on: *mut mutex,
    blocked_lock: raw_spinlock,
    blocked_donor: *mut task_struct,
    journal_info: *mut (),
    bio_list: *mut bio_list,
    plug: *mut blk_plug,
    reclaim_state: *mut reclaim_state,
    io_context: *mut io_context,
    capture_control: *mut capture_control,
    ptrace_message: u64,
    last_siginfo: *mut kernel_siginfo,
    ioac: task_io_accounting,
    acct_rss_mem1: u64,
    acct_vm_mem1: u64,
    acct_timexpd: u64,
    mems_allowed: __SlateRecord_nodemask_types_h_17,
    mems_allowed_seq: seqcount_spinlock,
    cpuset_mem_spread_rotor: i32,
    cgroups: *mut css_set,
    cg_list: list_head,
    futex: futex_sched_data,
    perf_recursion: [u8; 4],
    perf_event_ctxp: *mut perf_event_context,
    perf_event_mutex: mutex,
    perf_event_list: list_head,
    perf_ctx_data: *mut perf_ctx_data,
    ipi_mask: task_ipi_mask,
    mempolicy: *mut mempolicy,
    il_prev: i16,
    il_weight: u8,
    pref_node_fork: i16,
    cache_work: callback_head,
    sched_cache_grp: *mut sched_cache_group,
    preferred_llc: i32,
    pref_llc_queued: i32,
    rseq: rseq_data,
    mm_cid: sched_mm_cid,
    tlb_ubc: tlbflush_unmap_batch,
    splice_pipe: *mut pipe_inode_info,
    task_frag: page_frag,
    delays: *mut task_delay_info,
    nr_dirtied: i32,
    nr_dirtied_pause: i32,
    dirty_paused_when: u64,
    timer_slack_ns: u64,
    default_timer_slack_ns: u64,
    trace_recursion: u64,
    throttle_disk: *mut gendisk,
    utask: *mut uprobe_task,
    kmap_ctrl: kmap_ctrl,
    rcu: callback_head,
    rcu_users: refcount_struct,
    pagefault_disabled: i32,
    oom_reaper_list: *mut task_struct,
    oom_reaper_timer: timer_list,
    stack_vm_area: *mut vm_struct,
    stack_refcount: refcount_struct,
    security: *mut (),
    bpf_net_context: *mut bpf_net_context,
    mce_vaddr: *mut (),
    mce_kflags: u64,
    mce_addr: u64,
    __slate_bits_2: __slate_bits::__SlateBits51U2,
    mce_kill_me: callback_head,
    mce_count: i32,
    kretprobe_instances: llist_head,
    rethooks: llist_head,
    l1d_flush_kill: callback_head,
    unwind_info: unwind_task_info,
    thread: thread_struct,
}


// I am an IDLE thread
// Allocating memory to free memory. See memalloc_noreclaim_save()
// Kernel thread cloned from userspace thread
// All allocations inherit GFP_NOFS. See memalloc_nfs_save()
// All allocations inherit GFP_NOIO. See memalloc_noio_save()
// I am a kernel thread
// Userland is not allowed to meddle with cpus_mask
// Allocations constrained to zones which allow long term pinning.
// See memalloc_pin_save()
// Per-process atomic flags.
// May not gain new privileges.
// Spread page cache over cpuset
// Speculative Store Bypass disabled
// Speculative Store Bypass force disabled
// Indirect branch speculation restricted
// Indirect branch speculation permanently restricted
// Speculative Store Bypass clear on execve()
fn test_tsk_thread_flag(tsk: *mut task_struct, flag: i32) -> i32 {
test_ti_thread_flag(task_thread_info!(tsk), flag)
}

#[inline(always)]
fn _cond_resched() -> i32 {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__cond_resched as *const ()) };
unsafe { __v0.unwrap()() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__cond_resched_88: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__cond_resched) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
struct sched_cache_time {
    runtime: u64,
    epoch: u64,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sched_cache_group {
    pcpu_sched: *mut sched_cache_time,
    lock: raw_spinlock,
    epoch: u64,
    nr_running_avg: u64,
    next_scan: u64,
    footprint: u64,
    cpu: i32,
    refcnt: refcount_struct,
    rcu: callback_head,
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_nbcon_can_proceed: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_nbcon_enter_unsafe: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_nbcon_exit_unsafe: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_nbcon_reacquire_nobuf: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_nbcon_device_try_acquire: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_nbcon_device_release: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous", sym nbcon_can_proceed, sym nbcon_enter_unsafe, sym nbcon_exit_unsafe, sym nbcon_reacquire_nobuf, sym nbcon_device_try_acquire, sym nbcon_device_release, options(att_syntax));

fn sigismember(set: *mut __SlateRecord_signal_h_25, _sig: i32) -> i32 {
let sig: u64 = _sig.wrapping_sub(1) as u64;
if 1i32 == 1 {
return (1 & (unsafe { *((*set).sig.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) }) >> sig) as i32;
} else {
return (1 & (unsafe { *((*set).sig.as_mut_ptr() as *mut u64).wrapping_offset((sig / 64) as isize) }) >> sig % 64) as i32;
}
}

// 256 bytes including ->parent
// 256 bytes
// 240 bytes
#[repr(C)]
#[derive(Clone, Copy)]
struct maple_tree {
    __slate_anon_0: __SlateRecord_maple_tree_h_261,
    ma_flags: u32,
    ma_root: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rw_semaphore {
    count: __SlateRecord_types_h_197,
    owner: __SlateRecord_types_h_197,
    osq: optimistic_spin_queue,
    wait_lock: raw_spinlock,
    first_waiter: *mut rwsem_waiter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag_counters {
    bytes: u64,
    calls: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
}


fn mem_alloc_profiling_enabled() -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


// Add a ktime_t variable and a scalar nanosecond value.
// res = kt + nsval:
// Subtract a scalar nanosecod from a ktime_t variable
// res = kt - nsval:
// Map the ktime_t to timespec conversion to ns_to_timespec function
fn ktime_to_ns(kt: i64) -> i64 {
kt
}

#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_ctr {
    srcu_locks: __SlateRecord_types_h_197,
    srcu_unlocks: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_uprobe {
    __slate_anon_0: __SlateRecord_uprobes_h_31,
    ops: *const uprobe_xol_ops,
    __slate_anon_2: __SlateRecord_uprobes_h_38,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_uprobe_task {
    saved_scratch_register: u64,
    saved_trap_nr: u32,
    saved_tf: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hprobe {
    state: u32,
    srcu_scp: *mut srcu_ctr,
    uprobe: *mut uprobe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_task {
    state: u32,
    depth: u32,
    return_instances: *mut return_instance,
    ri_pool: *mut return_instance,
    ri_timer: timer_list,
    ri_seqcount: seqcount,
    __slate_anon_6: __SlateRecord_uprobes_h_136,
    active_uprobe: *mut uprobe,
    xol_vaddr: u64,
    signal_denied: bool,
    auprobe: *mut arch_uprobe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct return_consumer {
    cookie: u64,
    id: u64,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct return_instance {
    hprobe: hprobe,
    func: u64,
    stack: u64,
    orig_ret_vaddr: u64,
    chained: bool,
    cons_cnt: i32,
    next: *mut return_instance,
    rcu: callback_head,
    consumer: return_consumer,
    extra_consumers: *mut return_consumer,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobes_state {
    xol_area: *mut xol_area,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_counter {
    lock: raw_spinlock,
    count: i64,
    list: list_head,
    counters: *mut i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mmu_h_84 {
    ctx_id: u64,
    tlb_gen: __SlateRecord_types_h_197,
    next_trim_cpumask: u64,
    ldt_usr_sem: rw_semaphore,
    ldt: *mut ldt_struct,
    flags: u64,
    lock: mutex,
    vdso: *mut (),
    vdso_image: *const vdso_image,
    perf_rdpmc_allowed: __SlateRecord_types_h_190,
    pkey_allocation_map: u16,
    execute_only_pkey: i16,
    global_asid: u16,
    asid_transition: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_41 {
    f: u64,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct page {
    flags: __SlateRecord_mm_types_h_41,
    __slate_anon_1: __SlateRecord_mm_types_h_89,
    __slate_anon_2: __SlateRecord_mm_types_h_155,
    _refcount: __SlateRecord_types_h_190,
}


// Opaque type representing current VMA (vm_area_struct) flag state. Must be
// accessed via vma_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_871 {
    __vma_flags: [u64; 1],
}


// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1171 {
    __mm_flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mm_struct {
    __slate_anon_0: __SlateRecord_mm_types_h_1176,
    flexible_array: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_capability_h_24 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array {
    root: *mut assoc_array_ptr,
    nr_leaves_on_tree: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_tag {
    rcu: callback_head,
    usage: refcount_struct,
    removed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct keyring_index_key {
    hash: u64,
    __slate_anon_1: __SlateRecord_key_h_117,
    r#type: *mut key_type,
    domain_tag: *mut key_tag,
    description: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union key_payload {
    rcu_data0: *mut (),
    data: [*mut (); 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_restriction {
    check: Option<unsafe extern "C" fn(*mut key, *const key_type, *const key_payload, *mut key) -> i32>,
    key: *mut key,
    keytype: *mut key_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key {
    usage: refcount_struct,
    serial: i32,
    __slate_anon_2: __SlateRecord_key_h_198,
    sem: rw_semaphore,
    user: *mut key_user,
    security: *mut (),
    __slate_anon_6: __SlateRecord_key_h_208,
    last_used_at: i64,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    perm: u32,
    quotalen: u16,
    datalen: u16,
    state: i16,
    flags: u64,
    __slate_anon_15: __SlateRecord_key_h_246,
    __slate_anon_16: __SlateRecord_key_h_261,
    restrict_link: *mut key_restriction,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_struct {
    __count: refcount_struct,
    epoll_watches: percpu_counter,
    unix_inflight: u64,
    pipe_bufs: __SlateRecord_types_h_197,
    uidhash_node: hlist_node,
    uid: __SlateRecord_uidgid_types_h_9,
    locked_vm: __SlateRecord_types_h_197,
    ratelimit: ratelimit_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct group_info {
    usage: refcount_struct,
    ngroups: i32,
    gid: [__SlateRecord_uidgid_types_h_13; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cred {
    usage: __SlateRecord_types_h_197,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    suid: __SlateRecord_uidgid_types_h_9,
    sgid: __SlateRecord_uidgid_types_h_13,
    euid: __SlateRecord_uidgid_types_h_9,
    egid: __SlateRecord_uidgid_types_h_13,
    fsuid: __SlateRecord_uidgid_types_h_9,
    fsgid: __SlateRecord_uidgid_types_h_13,
    securebits: u32,
    cap_inheritable: __SlateRecord_capability_h_24,
    cap_permitted: __SlateRecord_capability_h_24,
    cap_effective: __SlateRecord_capability_h_24,
    cap_bset: __SlateRecord_capability_h_24,
    cap_ambient: __SlateRecord_capability_h_24,
    jit_keyring: u8,
    session_keyring: *mut key,
    process_keyring: *mut key,
    thread_keyring: *mut key,
    request_key_auth: *mut key,
    security: *mut (),
    user: *mut user_struct,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    group_info: *mut group_info,
    __slate_anon_25: __SlateRecord_cred_h_147,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhash_head {
    next: *mut rhash_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pid {
    count: refcount_struct,
    level: u32,
    lock: spinlock,
    __slate_anon_3: __SlateRecord_pid_h_62,
    tasks: [hlist_head; 4],
    inodes: hlist_head,
    wait_pidfd: wait_queue_head,
    rcu: callback_head,
    numbers: [upid; 0],
}


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
}

fn hrtimer_start_expires(timer: *mut hrtimer, mode: u32) {
let soft: i64 = hrtimer_get_softexpires(timer as *const hrtimer);
let hard: i64 = hrtimer_get_expires(timer as *const hrtimer);
let delta: u64 = ktime_to_ns(ktime_sub!(hard, soft)) as u64;
unsafe { hrtimer_start_range_ns(timer, soft, delta, mode) };
}

fn hrtimer_restart(timer: *mut hrtimer) {
hrtimer_start_expires(timer, 0u32);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sighand_struct {
    siglock: spinlock,
    count: refcount_struct,
    signalfd_wqh: wait_queue_head,
    action: [k_sigaction; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pacct_struct {
    ac_flag: i32,
    ac_exitcode: i64,
    ac_mem: u64,
    ac_utime: u64,
    ac_stime: u64,
    ac_minflt: u64,
    ac_majflt: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpu_itimer {
    expires: u64,
    incr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_cputime_atomic {
    utime: __SlateRecord_types_h_197,
    stime: __SlateRecord_types_h_197,
    sum_exec_runtime: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_group_cputimer {
    cputime_atomic: task_cputime_atomic,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct core_thread {
    task: *mut task_struct,
    next: *mut core_thread,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct core_state {
    nr_threads: __SlateRecord_types_h_190,
    dumper: core_thread,
    startup: completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct signal_struct {
    sigcnt: refcount_struct,
    live: __SlateRecord_types_h_190,
    nr_threads: i32,
    quick_threads: i32,
    thread_head: list_head,
    wait_chldexit: wait_queue_head,
    curr_target: *mut task_struct,
    shared_pending: sigpending,
    multiprocess: hlist_head,
    group_exit_code: i32,
    notify_count: i32,
    group_exec_task: *mut task_struct,
    group_stop_count: i32,
    flags: u32,
    core_state: *mut core_state,
    __slate_bits_0: __slate_bits::__SlateBits239U0,
    next_posix_timer_id: __SlateRecord_types_h_190,
    posix_timers: hlist_head,
    ignored_posix_timers: hlist_head,
    real_timer: hrtimer,
    it_real_incr: i64,
    it: [cpu_itimer; 2],
    cputimer: thread_group_cputimer,
    posix_cputimers: posix_cputimers,
    pids: [*mut pid; 4],
    tty_old_pgrp: *mut pid,
    leader: i32,
    tty: *mut tty_struct,
    stats_lock: seqlock,
    utime: u64,
    stime: u64,
    cutime: u64,
    cstime: u64,
    gtime: u64,
    cgtime: u64,
    prev_cputime: prev_cputime,
    nvcsw: u64,
    nivcsw: u64,
    cnvcsw: u64,
    cnivcsw: u64,
    min_flt: u64,
    maj_flt: u64,
    cmin_flt: u64,
    cmaj_flt: u64,
    inblock: u64,
    oublock: u64,
    cinblock: u64,
    coublock: u64,
    maxrss: u64,
    cmaxrss: u64,
    ioac: task_io_accounting,
    sum_sched_runtime: u64,
    rlim: [rlimit; 16],
    pacct: pacct_struct,
    stats: *mut taskstats,
    audit_tty: u32,
    tty_audit_buf: *mut tty_audit_buf,
    cgroup_threadgroup_rwsem: rw_semaphore,
    oom_flag_origin: bool,
    oom_score_adj: i16,
    oom_score_adj_min: i16,
    oom_mm: *mut mm_struct,
    cred_guard_mutex: mutex,
    exec_update_lock: rw_semaphore,
}


// group exit in progress
fn task_sigpending(p: *mut task_struct) -> i32 {
unlikely!(test_tsk_thread_flag(p, 1i32)) as i32
}

fn signal_pending(p: *mut task_struct) -> i32 {
// TIF_NOTIFY_SIGNAL isn't really a signal, but it requires the same
// behavior in terms of ensuring that we break out of wait loops
// so that notify signal callbacks can be processed.
if unlikely!(test_tsk_thread_flag(p, 2i32)) != 0 {
return 1;
}
task_sigpending(p)
}

fn __fatal_signal_pending(p: *mut task_struct) -> i32 {
unlikely!(sigismember(unsafe { core::ptr::addr_of_mut!((*p).pending.signal) }, 9i32)) as i32
}

fn signal_pending_state(state: u32, p: *mut task_struct) -> i32 {
if state & 257u32 == 0 {
return 0;
}
if signal_pending(p) == 0 {
return 0;
}
let __v0: bool = state & 1 != 0 || __fatal_signal_pending(p) != 0;
__v0 as i32
}

fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_344() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*((*(*task).signal).rlim.as_mut_ptr() as *mut rlimit).wrapping_offset(limit as isize)).rlim_cur) as *const u64) };
__v0
}

fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
}

fn rcuwait_init(w: *mut rcuwait) {
unsafe {
(*w).task = core::ptr::null_mut::<task_struct>();
}
}

fn rcuwait_active(w: *mut rcuwait) -> i32 {
let mut __v0: *mut task_struct;
let mut __v1: *mut task_struct;
let mut __v2: *mut task_struct;
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut task_struct>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_347() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*w).task) as *const *mut task_struct) };
let __UNIQUE_ID_rcu_346: *mut task_struct = __v2;
__v1 = __UNIQUE_ID_rcu_346;
__v0 = __v1;
!__v0.is_null() as i32
}

fn prepare_to_rcuwait(w: *mut rcuwait) {
let _r_a_p__v: u64 = get_current() as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut task_struct>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_348() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__10!((*w).task, _r_a_p__v as *mut task_struct);
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word__8!() {
unsafe { __compiletime_assert_349() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut task_struct>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_350() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__10!(*unsafe { core::ptr::addr_of_mut!((*w).task) }, RCU_INITIALIZER!(_r_a_p__v as *mut task_struct));
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
}
}

// 									 * Implicit barrier (A) pairs with (B) in				 * rcuwait_wake_up().
#[repr(C)]
#[derive(Clone, Copy)]
struct nbcon_state {
    __slate_anon_0: __SlateRecord_console_h_216,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nbcon_context {
    console: *mut console,
    spinwait_max_us: u32,
    prio: u32,
    __slate_bits_0: __slate_bits::__SlateBits379U0,
    pbufs: *mut printk_buffers,
    seq: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nbcon_write_context {
    ctxt: nbcon_context,
    outbuf: *mut u8,
    len: u32,
    unsafe_takeover: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct console {
    name: [u8; 16],
    write: Option<unsafe extern "C" fn(*mut console, *const u8, u32)>,
    read: Option<unsafe extern "C" fn(*mut console, *mut u8, u32) -> i32>,
    device: Option<unsafe extern "C" fn(*mut console, *mut i32) -> *mut tty_driver>,
    unblank: Option<unsafe extern "C" fn()>,
    setup: Option<unsafe extern "C" fn(*mut console, *mut u8) -> i32>,
    exit: Option<unsafe extern "C" fn(*mut console) -> i32>,
    r#match: Option<unsafe extern "C" fn(*mut console, *mut u8, i32, *mut u8) -> i32>,
    flags: i16,
    index: i16,
    cflag: i32,
    ispeed: u32,
    ospeed: u32,
    seq: u64,
    dropped: u64,
    data: *mut (),
    node: hlist_node,
    write_atomic: Option<unsafe extern "C" fn(*mut console, *mut nbcon_write_context)>,
    write_thread: Option<unsafe extern "C" fn(*mut console, *mut nbcon_write_context)>,
    device_lock: Option<unsafe extern "C" fn(*mut console, *mut u64)>,
    device_unlock: Option<unsafe extern "C" fn(*mut console, u64)>,
    nbcon_state: __SlateRecord_types_h_190,
    nbcon_seq: __SlateRecord_types_h_197,
    nbcon_device_ctxt: nbcon_context,
    nbcon_prev_seq: __SlateRecord_types_h_197,
    pbufs: *mut printk_buffers,
    kthread: *mut task_struct,
    rcuwait: rcuwait,
    irq_work: irq_work,
}


fn lockdep_assert_console_list_lock_held() {
}

fn console_srcu_read_lock_is_held() -> bool {
true
}

fn console_srcu_read_flags(con: *const console) -> i16 {
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!console_srcu_read_lock_is_held())) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("351: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 351b - ., 3; .popsection\n# {0}", const 351 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_console_srcu_read_flags_1, sym __slate_asm_string_console_srcu_read_flags_2, const 524 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("352: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 352b - ., 4; .popsection\n# {0}", const 352 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
// The READ_ONCE() matches the WRITE_ONCE() when @flags are modified
// for registered consoles with console_srcu_write_flags().
let mut __v5: i16;
__kcsan_disable_current();
let mut __v: i16 = 0;
let mut __v6: i16;
loop {
if !(__native_word__17!() || ((const { core::mem::size_of::<i16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_353() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*con).flags) as *const i16) };
__v = __v6;
__kcsan_enable_current();
__v5 = __v;
__v5
}

// for_each_console_srcu() - Iterator over registered consoles
// @con:	struct console pointer used as loop cursor
//
// Although SRCU guarantees the console list will be consistent, the
// struct console fields may be updated by other CPUs while iterating.
//
// Requires console_srcu_read_lock to be held. Can be invoked from
// any context.
fn console_is_usable(con: *mut console, flags: i16, use_atomic: bool) -> bool {
if (flags as i32) & 4 == 0 {
return false;
}
if (flags as i32) & 128 != 0 {
return false;
}
if (flags as i32) & 256 != 0 {
if use_atomic {
// The write_atomic() callback is optional.
if (unsafe { (*con).write_atomic }) == None {
return false;
}
// An unsafe write_atomic() callback is only usable
// when unsafe takeovers are allowed.
let __v0: bool = (flags as i32) & 512 != 0 && !nbcon_allow_unsafe_takeover();
if __v0 {
return false;
}
}
// For the !use_atomic case, @printk_kthreads_running is not
// checked because the write_thread() callback is also used
// via the legacy loop when the printer threads are not
// available.
} else {
if (unsafe { (*con).write }) == None {
return false;
}
}
// Console drivers may assume that per-cpu resources have been
// allocated. So unless they're explicitly marked as being able to
// cope (CON_ANYTIME) don't call them until this CPU is officially up.
let mut __v1: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v3: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v3 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v3;
}
2 => {
let mut __v5: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
4 => {
let mut __v6: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v6 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v6;
}
8 => {
let mut __v7: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v7 = pfo_val__ as i32;
pscr_ret__ = __v7;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v1 = pscr_ret__;
if !cpu_online(__v1 as u32) && (flags as i32) & 16 == 0 {
return false;
}
true
}

// The microseconds/nanosecond delay multiplicators are used to convert a
// constant microseconds/nanoseconds value to a value which can be used by the
// architectures specific implementation to transform it into loops.
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
#[inline(always)]
fn udelay(usec: u64) {
if unsafe { __slate_intrinsic_f217c4720f04f6db(usec) } {
if usec >= 20000 {
unsafe { __bad_udelay() };
} else {
unsafe { __const_udelay(usec.wrapping_mul(4295u64)) };
}
} else {
unsafe { __udelay(usec) };
}
}

fn kdb_printf_on_this_cpu() -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_slab_h_532 {
}


// no-op
// kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
// kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
// and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
#[inline(always)]
fn kmalloc_type(flags: u32, token: __SlateRecord_slab_h_532) -> u32 {
// The most common case is KMALLOC_NORMAL, so test for it
// with a single branch for all the relevant flags.
if likely!(flags & ((BIT!(4i32) as u32) | if IS_ENABLED__2!() != 0 { BIT!(0i32) as u32 } else { 0 } | if IS_ENABLED!() != 0 { BIT!(22i32) as u32 } else { 0 }) == 0) != 0 {
return 0;
}
// At least one of the flags has to be set. Their priorities in
// decreasing order are:
//  1) __GFP_DMA
//  2) __GFP_RECLAIMABLE
//  3) __GFP_ACCOUNT
if IS_ENABLED__2!() != 0 && flags & (BIT!(0i32) as u32) != 0 {
return 2;
}
if IS_ENABLED!() == 0 || flags & (BIT!(4i32) as u32) != 0 {
return 1;
} else {
return 0;
}
}

#[inline(always)]
fn __kmalloc_index(size: u64, size_is_constant: bool) -> u32 {
if size == 0 {
return 0;
}
if size <= 8u64 {
return 3;
}
if 8i32 <= 32 && size > 64 && size <= 96 {
return 1;
}
if 8i32 <= 64 && size > 128 && size <= 192 {
return 2;
}
if size <= 8 {
return 3;
}
if size <= 16 {
return 4;
}
if size <= 32 {
return 5;
}
if size <= 64 {
return 6;
}
if size <= 128 {
return 7;
}
if size <= 256 {
return 8;
}
if size <= 512 {
return 9;
}
if size <= 1024 {
return 10;
}
if size <= 2048u64 {
return 11;
}
if size <= 4096u64 {
return 12;
}
if size <= 8192u64 {
return 13;
}
if size <= 16384u64 {
return 14;
}
if size <= 32768u64 {
return 15;
}
if size <= 65536u64 {
return 16;
}
if size <= 131072u64 {
return 17;
}
if size <= 262144u64 {
return 18;
}
if size <= 524288u64 {
return 19;
}
if size <= 1048576u64 {
return 20;
}
if size <= 2097152u64 {
return 21;
}
if IS_ENABLED!() == 0 && size_is_constant {
loop {
if true {
unsafe { __compiletime_assert_360() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("361: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 361b - ., 3; .popsection\n# {0}", const 361 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string___kmalloc_index_1, sym __slate_asm_string___kmalloc_index_2, const 834 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { core::hint::unreachable_unchecked() };
if true {
break;
}
}
}
// Will never be reached. Needed because the compiler may complain
1i32.wrapping_neg() as u32
}

#[inline(always)]
fn _kmalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
if (unsafe { __slate_intrinsic_f217c4720f04f6db(size) }) && size != 0 {
let mut index: u32 = 0;
if size > 8192u64 {
return unsafe { __kmalloc_large_noprof(size, flags) };
}
index = kmalloc_index!(size);
return unsafe { __kmalloc_cache_noprof(kmalloc_caches[(kmalloc_type(flags, token) as u32) as usize][index as usize], flags, size) };
}
unsafe { __kmalloc_noprof(PASS_TOKEN_PARAMS!(size), flags) }
}

// __alloc_objs - Allocate objects of a given type using
// @KMALLOC: which size-based kmalloc wrapper to allocate with.
// @GFP: GFP flags for the allocation.
// @TYPE: type to allocate space for.
// @COUNT: how many @TYPE objects to allocate.
//
// Returns: Newly allocated pointer to (first) @TYPE of @COUNT-many
// allocated @TYPE objects, or NULL on failure.
// kmalloc_obj - Allocate a single instance of the given type
// @VAR_OR_TYPE: Variable or type to allocate.
// @...: optional GFP flags for the allocation (GFP_KERNEL when not specified).
//
// Returns: newly allocated pointer to a @VAR_OR_TYPE on success, or NULL
// on failure.
// the maximum size allowed to be reserved for a record
#[unsafe(no_mangle)]
extern "C" fn nbcon_kthreads_wake() {
let mut con: *mut console = unsafe { core::mem::zeroed() };
let mut cookie: i32 = 0;
if !unsafe { printk_kthreads_running } {
return;
}
// It is not allowed to call this function when console irq_work
// is blocked.
let mut __v2: i64;
let __ret_warn_on: i32 = !(!unsafe { console_irqwork_blocked }) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("392: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 392b - ., 3; .popsection\n# {0}", const 392 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_nbcon_kthreads_wake_3, sym __slate_asm_string_nbcon_kthreads_wake_4, const 1325 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("393: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 393b - ., 4; .popsection\n# {0}", const 393 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
return;
}
cookie = unsafe { console_srcu_read_lock() };
let mut __v7: *mut console;
let mut __v8: *mut hlist_node;
let mut __v9: *mut hlist_node;
loop {
if !(((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_395() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v9 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(console_list)).first) as *const *mut hlist_node) };
let __UNIQUE_ID_rcu_394: *mut hlist_node = __v9;
__v8 = __UNIQUE_ID_rcu_394;
let ____ptr: *mut hlist_node = __v8;
let __v11: *mut console = if !____ptr.is_null() {
hlist_entry!(____ptr)
} else {
core::ptr::null_mut::<console>()
};
__v7 = __v11;
con = __v7;
while !con.is_null() {
if (console_srcu_read_flags(con as *const console) as i32) & 256 != 0 {
// Only schedule irq_work if the printing thread is
// actively waiting. If not waiting, the thread will
// notice by itself that it has work to do.
if rcuwait_has_sleeper(unsafe { core::ptr::addr_of_mut!((*con).rcuwait) }) {
unsafe { irq_work_queue(core::ptr::addr_of_mut!((*con).irq_work)) };
}
}
let mut __v15: *mut console;
let mut __v16: *mut hlist_node;
let mut __v17: *mut hlist_node;
loop {
if !(((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_397() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v17 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*con).node)).next) as *const *mut hlist_node) };
__v16 = __v17;
let __v0: *mut hlist_node = __v16;
let __v19: *mut console = if !__v0.is_null() {
hlist_entry!(__v0)
} else {
core::ptr::null_mut::<console>()
};
__v15 = __v19;
con = __v15;
}
unsafe { console_srcu_read_unlock(cookie) };
}

fn nbcon_kthread_wake(con: *mut console) {
// Guarantee any new records can be seen by tasks preparing to wait
// before this context checks if the rcuwait is empty.
//
// The full memory barrier in rcuwait_wake_up() pairs with the full
// memory barrier within set_current_state() of
// ___rcuwait_wait_event(), which is called after prepare_to_rcuwait()
// adds the waiter but before it has checked the wait condition.
//
// This pairs with nbcon_kthread_func:A.
unsafe { rcuwait_wake_up(core::ptr::addr_of_mut!((*con).rcuwait)) }; // LMM(nbcon_kthread_wake:A)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct console_flush_type {
    nbcon_atomic: bool,
    nbcon_offload: bool,
    legacy_direct: bool,
    legacy_offload: bool,
}


fn printk_get_console_flush_type(ft: *mut console_flush_type) {
unsafe { memset(ft as *mut (), 0i32, core::mem::size_of::<console_flush_type>() as u64) };
match nbcon_get_default_prio() as u32 {
1 => {
if (unsafe { have_nbcon_console }) && !unsafe { have_boot_console } {
if (unsafe { printk_kthreads_running }) && !unsafe { console_irqwork_blocked } {
unsafe {
(*ft).nbcon_offload = true;
}
} else {
unsafe {
(*ft).nbcon_atomic = true;
}
}
}
// Legacy consoles are flushed directly when possible.
if (unsafe { have_legacy_console }) || unsafe { have_boot_console } {
if !unsafe { is_printk_legacy_deferred() } {
unsafe {
(*ft).legacy_direct = true;
}
} else {
if !unsafe { console_irqwork_blocked } {
unsafe {
(*ft).legacy_offload = true;
}
}
}
}
}
2 => {
if (unsafe { have_nbcon_console }) && !unsafe { have_boot_console } {
unsafe {
(*ft).nbcon_atomic = true;
}
}
// Legacy consoles are flushed directly when possible.
if (unsafe { have_legacy_console }) || unsafe { have_boot_console } {
if !unsafe { is_printk_legacy_deferred() } {
unsafe {
(*ft).legacy_direct = true;
}
} else {
if !unsafe { console_irqwork_blocked } {
unsafe {
(*ft).legacy_offload = true;
}
}
}
}
}
3 => {
if (unsafe { have_nbcon_console }) && !unsafe { have_boot_console } {
unsafe {
(*ft).nbcon_atomic = true;
}
}
// In panic, the nbcon consoles will directly print. But
// only allowed if there are no boot consoles.
if (unsafe { have_legacy_console }) || unsafe { have_boot_console } {
// This is the same decision as NBCON_PRIO_NORMAL
// except that offloading never occurs in panic.
//
// Note that console_flush_on_panic() will flush
// legacy consoles anyway, even if unsafe.
if !unsafe { is_printk_legacy_deferred() } {
unsafe {
(*ft).legacy_direct = true;
}
}
// In panic, if nbcon atomic printing occurs,
// the legacy consoles must remain silent until
// explicitly allowed.
if (unsafe { (*ft).nbcon_atomic }) && !unsafe { legacy_allow_panic_sync } {
unsafe {
(*ft).legacy_direct = false;
}
}
}
}
_ => {
let mut __v1: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("362: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 362b - ., 3; .popsection\n# {0}", const 362 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_printk_get_console_flush_type_2, sym __slate_asm_string_printk_get_console_flush_type_3, const 259 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("363: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 363b - ., 4; .popsection\n# {0}", const 363 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct printk_buffers {
    outbuf: [u8; 2048],
    scratchbuf: [u8; 1024],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct printk_message {
    pbufs: *mut printk_buffers,
    outbuf_len: u32,
    seq: u64,
    dropped: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_printk_info {
    subsystem: [u8; 16],
    device: [u8; 48],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct printk_info {
    seq: u64,
    ts_nsec: u64,
    text_len: u16,
    facility: u8,
    __slate_bits_0: __slate_bits::__SlateBits389U0,
    caller_id: u32,
    dev_info: dev_printk_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct printk_record {
    info: *mut printk_info,
    text_buf: *mut u8,
    text_buf_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct prb_data_blk_lpos {
    begin: u64,
    next: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct prb_desc {
    state_var: __SlateRecord_types_h_197,
    text_blk_lpos: prb_data_blk_lpos,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct prb_data_ring {
    size_bits: u32,
    data: *mut u8,
    head_lpos: __SlateRecord_types_h_197,
    tail_lpos: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct prb_desc_ring {
    count_bits: u32,
    descs: *mut prb_desc,
    infos: *mut printk_info,
    head_id: __SlateRecord_types_h_197,
    tail_id: __SlateRecord_types_h_197,
    last_finalized_seq: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct printk_ringbuffer {
    desc_ring: prb_desc_ring,
    text_data_ring: prb_data_ring,
    fail: __SlateRecord_types_h_197,
}


// Printk console printing implementation for consoles which does not depend
// on the legacy style console_lock mechanism.
//
// The state of the console is maintained in the "nbcon_state" atomic
// variable.
//
// The console is locked when:
//
//   - The 'prio' field contains the priority of the context that owns the
//     console. Only higher priority contexts are allowed to take over the
//     lock. A value of 0 (NBCON_PRIO_NONE) means the console is not locked.
//
//   - The 'cpu' field denotes on which CPU the console is locked. It is used
//     to prevent busy waiting on the same CPU. Also it informs the lock owner
//     that it has lost the lock in a more complex scenario when the lock was
//     taken over by a higher priority context, released, and taken on another
//     CPU with the same priority as the interrupted owner.
//
// The acquire mechanism uses a few more fields:
//
//   - The 'req_prio' field is used by the handover approach to make the
//     current owner aware that there is a context with a higher priority
//     waiting for the friendly handover.
//
//   - The 'unsafe' field allows to take over the console in a safe way in the
//     middle of emitting a message. The field is set only when accessing some
//     shared resources or when the console device is manipulated. It can be
//     cleared, for example, after emitting one character when the console
//     device is in a consistent state.
//
//   - The 'unsafe_takeover' field is set when a hostile takeover took the
//     console in an unsafe state. The console will stay in the unsafe state
//     until re-initialized.
//
// The acquire mechanism uses three approaches:
//
//   1) Direct acquire when the console is not owned or is owned by a lower
//      priority context and is in a safe state.
//
//   2) Friendly handover mechanism uses a request/grant handshake. It is used
//      when the current owner has lower priority and the console is in an
//      unsafe state.
//
//      The requesting context:
//
//        a) Sets its priority into the 'req_prio' field.
//
//        b) Waits (with a timeout) for the owning context to unlock the
//           console.
//
//        c) Takes the lock and clears the 'req_prio' field.
//
//      The owning context:
//
//        a) Observes the 'req_prio' field set on exit from the unsafe
//           console state.
//
//        b) Gives up console ownership by clearing the 'prio' field.
//
//   3) Unsafe hostile takeover allows to take over the lock even when the
//      console is an unsafe state. It is used only in panic() by the final
//      attempt to flush consoles in a try and hope mode.
//
//      Note that separate record buffers are used in panic(). As a result,
//      the messages can be read and formatted without any risk even after
//      using the hostile takeover in unsafe state.
//
// The release function simply clears the 'prio' field.
//
// All operations on @console::nbcon_state are atomic cmpxchg based to
// handle concurrency.
//
// The acquire/release functions implement only minimal policies:
//
//   - Preference for higher priority contexts.
//   - Protection of the panic CPU.
//
// All other policy decisions must be made at the call sites:
//
//   - What is marked as an unsafe section.
//   - Whether to spin-wait if there is already an owner and the console is
//     in an unsafe state.
//   - Whether to attempt an unsafe hostile takeover.
//
// The design allows to implement the well known:
//
//     acquire()
//     output_one_printk_record()
//     release()
//
// The output of one printk record might be interrupted with a higher priority
// context. The new owner is supposed to reprint the entire interrupted record
// from scratch.
/// Counter of active nbcon emergency contexts.
static mut nbcon_cpu_emergency_cnt: __SlateRecord_types_h_190 = __SlateRecord_types_h_190 { counter: 0i32 };

/// nbcon_state_set - Helper function to set the console state
/// @con:	Console to update
/// @new:	The new state to write
///
/// Only to be used when the console is not yet or no longer visible in the
/// system. Otherwise use nbcon_state_try_cmpxchg().
fn nbcon_state_set(con: *mut console, new: &nbcon_state) {
atomic_set(unsafe { core::ptr::addr_of_mut!((*con).nbcon_state) }, (unsafe { (*(new as *const nbcon_state)).__slate_anon_0.atom }) as i32);
}

/// nbcon_state_read - Helper function to read the console state
/// @con:	Console to read
/// @state:	The state to store the result
fn nbcon_state_read(con: *mut console, state: &mut nbcon_state) {
unsafe {
(*(state as *mut nbcon_state)).__slate_anon_0.atom = atomic_read(core::ptr::addr_of_mut!((*con).nbcon_state) as *const __SlateRecord_types_h_190) as u32;
}
}

/// nbcon_state_try_cmpxchg() - Helper function for atomic_try_cmpxchg() on console state
/// @con:	Console to update
/// @cur:	Old/expected state
/// @new:	New state
///
/// Return: True on success. False on fail and @cur is updated.
fn nbcon_state_try_cmpxchg(con: *mut console, cur: &mut nbcon_state, new: &nbcon_state) -> bool {
atomic_try_cmpxchg(unsafe { core::ptr::addr_of_mut!((*con).nbcon_state) }, (unsafe { core::ptr::addr_of_mut!((*(cur as *mut nbcon_state)).__slate_anon_0.atom) }) as *mut i32, (unsafe { (*(new as *const nbcon_state)).__slate_anon_0.atom }) as i32)
}

/// nbcon_seq_read - Read the current console sequence
/// @con:	Console to read the sequence of
///
/// Return:	Sequence number of the next record to print on @con.
#[unsafe(no_mangle)]
extern "C" fn nbcon_seq_read(con: *mut console) -> u64 {
let nbcon_seq: u64 = atomic_long_read((unsafe { core::ptr::addr_of_mut!((*con).nbcon_seq) }) as *const __SlateRecord_types_h_197) as u64;
__ulseq_to_u64seq!(nbcon_seq)
}

/// nbcon_seq_force - Force console sequence to a specific value
/// @con:	Console to work on
/// @seq:	Sequence number value to set
///
/// Only to be used during init (before registration) or in extreme situations
/// (such as panic with CONSOLE_REPLAY_ALL).
#[unsafe(no_mangle)]
extern "C" fn nbcon_seq_force(con: *mut console, seq: u64) {
// If the specified record no longer exists, the oldest available record
// is chosen. This is especially important on 32bit systems because only
// the lower 32 bits of the sequence number are stored. The upper 32 bits
// are derived from the sequence numbers available in the ringbuffer.
let valid_seq: u64 = max_t__6!(u64, seq, unsafe { prb_first_valid_seq(prb) });
atomic_long_set(unsafe { core::ptr::addr_of_mut!((*con).nbcon_seq) }, __u64seq_to_ulseq!(valid_seq) as i64);
}

/// nbcon_seq_try_update - Try to update the console sequence number
/// @ctxt:	Pointer to an acquire context that contains
/// 		all information about the acquire mode
/// @new_seq:	The new sequence number to set
///
/// @ctxt->seq is updated to the new value of @con::nbcon_seq (expanded to
/// the 64bit value). This could be a different value than @new_seq if
/// nbcon_seq_force() was used or the current context no longer owns the
/// console. In the later case, it will stop printing anyway.
fn nbcon_seq_try_update(ctxt: *mut nbcon_context, new_seq: u64) {
let mut nbcon_seq: u64 = __u64seq_to_ulseq!(unsafe { (*ctxt).seq });
let con: *mut console = unsafe { (*ctxt).console };
if atomic_long_try_cmpxchg(unsafe { core::ptr::addr_of_mut!((*con).nbcon_seq) }, core::ptr::addr_of_mut!(nbcon_seq) as *mut i64, __u64seq_to_ulseq!(new_seq) as i64) {
unsafe {
(*ctxt).seq = new_seq;
}
} else {
unsafe {
(*ctxt).seq = nbcon_seq_read(con);
}
}
}

/// nbcon_context_try_acquire_direct - Try to acquire directly
/// @ctxt:		The context of the caller
/// @cur:		The current console state
/// @is_reacquire:	This acquire is a reacquire
///
/// Acquire the console when it is released. Also acquire the console when
/// the current owner has a lower priority and the console is in a safe state.
///
/// Return:	0 on success. Otherwise, an error code on failure. Also @cur
/// 		is updated to the latest state when failed to modify it.
///
/// Errors:
///
/// 	-EPERM:		A panic is in progress and this is neither the panic
/// 			CPU nor is this a reacquire. Or the current owner or
/// 			waiter has the same or higher priority. No acquire
/// 			method can be successful in these cases.
///
/// 	-EBUSY:		The current owner has a lower priority but the console
/// 			in an unsafe state. The caller should try using
/// 			the handover acquire method.
fn nbcon_context_try_acquire_direct(ctxt: *mut nbcon_context, cur: &mut nbcon_state, is_reacquire: bool) -> i32 {
let mut cpu: u32 = 0;
let mut __v0: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v1: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v3: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v3 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v3;
}
2 => {
let mut __v5: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
4 => {
let mut __v6: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v6 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v6;
}
8 => {
let mut __v7: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v7 = pfo_val__ as i32;
pscr_ret__ = __v7;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v1 = pscr_ret__;
__v0 = __v1;
cpu = __v0 as u32;
let con: *mut console = unsafe { (*ctxt).console };
let mut new: nbcon_state = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(new) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
loop {
// Panic does not imply that the console is owned. However,
// since all non-panic CPUs are stopped during panic(), it
// is safer to have them avoid gaining console ownership.
//
// One exception is when kdb has locked for printing on this CPU.
//
// Second exception is a reacquire (and an unsafe takeover
// has not previously occurred) then it is allowed to attempt
// a direct acquire in panic. This gives console drivers an
// opportunity to perform any necessary cleanup if they were
// interrupted by the panic CPU while printing.
let __v8: bool = (unsafe { panic_on_other_cpu() }) && !kdb_printf_on_this_cpu();
if __v8 && (!is_reacquire || ((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe_takeover() }) as i32) != 0) {
return 1i32.wrapping_neg();
}
if ((unsafe { (*ctxt).prio }) as u32) <= (((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_prio() }) as i32) as u32) || ((unsafe { (*ctxt).prio }) as u32) <= (((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_req_prio() }) as i32) as u32) {
return 1i32.wrapping_neg();
}
if ((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe() }) as i32) != 0 {
return 16i32.wrapping_neg();
}
// The console should never be safe for a direct acquire
// if an unsafe hostile takeover has ever happened.
let mut __v9: i64;
let __ret_warn_on: i32 = (((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe_takeover() }) as i32) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("366: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 366b - ., 3; .popsection\n# {0}", const 366 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_nbcon_context_try_acquire_direct_10, sym __slate_asm_string_nbcon_context_try_acquire_direct_11, const 280 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("367: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 367b - ., 4; .popsection\n# {0}", const 367 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v9 = (__ret_warn_on != 0) as i64;
unsafe {
new.__slate_anon_0.atom = (*(cur as *mut nbcon_state)).__slate_anon_0.atom;
}
unsafe {
let __t4 = (*ctxt).prio as u32;
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_prio(__t4);
}
unsafe {
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_req_prio(0u32);
}
unsafe {
let __t5 = ((*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe_takeover() as i32) as u32;
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_unsafe(__t5);
}
unsafe {
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_cpu(cpu);
}
if !(!nbcon_state_try_cmpxchg(con, cur, unsafe { &(*core::ptr::addr_of_mut!(new)) })) {
break;
}
}
0
}

fn nbcon_waiter_matches(cur: &mut nbcon_state, expected_prio: i32) -> bool {
// The request context is well defined by the @req_prio because:
//
// - Only a context with a priority higher than the owner can become
//   a waiter.
// - Only a context with a priority higher than the waiter can
//   directly take over the request.
// - There are only three priorities.
// - Only one CPU is allowed to request PANIC priority.
// - Lower priorities are ignored during panic() until reboot.
//
// As a result, the following scenario is *not* possible:
//
// 1. This context is currently a waiter.
// 2. Another context with a higher priority than this context
//    directly takes ownership.
// 3. The higher priority context releases the ownership.
// 4. Another lower priority context takes the ownership.
// 5. Another context with the same priority as this context
//    creates a request and starts waiting.
//
// Event #1 implies this context is EMERGENCY.
// Event #2 implies the new context is PANIC.
// Event #3 occurs when panic() has flushed the console.
// Event #4 occurs when a non-panic CPU reacquires.
// Event #5 is not possible due to the panic_on_other_cpu() check
//          in nbcon_context_try_acquire_handover().
((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_req_prio() }) as i32) == expected_prio
}

/// nbcon_context_try_acquire_requested - Try to acquire after having
/// 					 requested a handover
/// @ctxt:	The context of the caller
/// @cur:	The current console state
///
/// This is a helper function for nbcon_context_try_acquire_handover().
/// It is called when the console is in an unsafe state. The current
/// owner will release the console on exit from the unsafe region.
///
/// Return:	0 on success and @cur is updated to the new console state.
/// 		Otherwise an error code on failure.
///
/// Errors:
///
/// 	-EPERM:		A panic is in progress and this is not the panic CPU
/// 			or this context is no longer the waiter.
///
/// 	-EBUSY:		The console is still locked. The caller should
/// 			continue waiting.
///
/// Note: The caller must still remove the request when an error has occurred
///       except when this context is no longer the waiter.
fn nbcon_context_try_acquire_requested(ctxt: *mut nbcon_context, cur: &mut nbcon_state) -> i32 {
let mut cpu: u32 = 0;
let mut __v0: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v1: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v3: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v3 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v3;
}
2 => {
let mut __v5: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
4 => {
let mut __v6: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v6 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v6;
}
8 => {
let mut __v7: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v7 = pfo_val__ as i32;
pscr_ret__ = __v7;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v1 = pscr_ret__;
__v0 = __v1;
cpu = __v0 as u32;
let con: *mut console = unsafe { (*ctxt).console };
let mut new: nbcon_state = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(new) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
// Note that the caller must still remove the request!
if unsafe { panic_on_other_cpu() } {
return 1i32.wrapping_neg();
}
// Note that the waiter will also change if there was an unsafe
// hostile takeover.
if !nbcon_waiter_matches(cur, ((unsafe { (*ctxt).prio }) as u32) as i32) {
return 1i32.wrapping_neg();
}
// If still locked, caller should continue waiting.
if ((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_prio() }) as i32) != 0 {
return 16i32.wrapping_neg();
}
// The previous owner should have never released ownership
// in an unsafe region.
let mut __v8: i64;
{
let __ret_warn_on: i32 = (((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe() }) as i32) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("368: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 368b - ., 3; .popsection\n# {0}", const 368 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_nbcon_context_try_acquire_requested_9, sym __slate_asm_string_nbcon_context_try_acquire_requested_10, const 377 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("369: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 369b - ., 4; .popsection\n# {0}", const 369 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v8 = (__ret_warn_on != 0) as i64;
}
unsafe {
new.__slate_anon_0.atom = (*(cur as *mut nbcon_state)).__slate_anon_0.atom;
}
unsafe {
let __t4 = (*ctxt).prio as u32;
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_prio(__t4);
}
unsafe {
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_req_prio(0u32);
}
unsafe {
let __t5 = ((*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe_takeover() as i32) as u32;
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_unsafe(__t5);
}
unsafe {
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_cpu(cpu);
}
if !nbcon_state_try_cmpxchg(con, cur, unsafe { &(*core::ptr::addr_of_mut!(new)) }) {
// The acquire could fail only when it has been taken
// over by a higher priority context.
let mut __v13: i64;
let __ret_warn_on: i32 = !(!nbcon_waiter_matches(cur, ((unsafe { (*ctxt).prio }) as u32) as i32)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("370: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 370b - ., 3; .popsection\n# {0}", const 370 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_nbcon_context_try_acquire_requested_14, sym __slate_asm_string_nbcon_context_try_acquire_requested_15, const 390 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("371: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 371b - ., 4; .popsection\n# {0}", const 371 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v13 = (__ret_warn_on != 0) as i64;
return 1i32.wrapping_neg();
}
// Handover success. This context now owns the console.
0
}

/// nbcon_context_try_acquire_handover - Try to acquire via handover
/// @ctxt:	The context of the caller
/// @cur:	The current console state
///
/// The function must be called only when the context has higher priority
/// than the current owner and the console is in an unsafe state.
/// It is the case when nbcon_context_try_acquire_direct() returns -EBUSY.
///
/// The function sets "req_prio" field to make the current owner aware of
/// the request. Then it waits until the current owner releases the console,
/// or an even higher context takes over the request, or timeout expires.
///
/// The current owner checks the "req_prio" field on exit from the unsafe
/// region and releases the console. It does not touch the "req_prio" field
/// so that the console stays reserved for the waiter.
///
/// Return:	0 on success. Otherwise, an error code on failure. Also @cur
/// 		is updated to the latest state when failed to modify it.
///
/// Errors:
///
/// 	-EPERM:		A panic is in progress and this is not the panic CPU.
/// 			Or a higher priority context has taken over the
/// 			console or the handover request.
///
/// 	-EBUSY:		The current owner is on the same CPU so that the hand
/// 			shake could not work. Or the current owner is not
/// 			willing to wait (zero timeout). Or the console does
/// 			not enter the safe state before timeout passed. The
/// 			caller might still use the unsafe hostile takeover
/// 			when allowed.
///
/// 	-EAGAIN:	@cur has changed when creating the handover request.
/// 			The caller should retry with direct acquire.
fn nbcon_context_try_acquire_handover(ctxt: *mut nbcon_context, cur: &mut nbcon_state) -> i32 {
let mut cpu: u32 = 0;
let mut __v0: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v1: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v3: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v3 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v3;
}
2 => {
let mut __v5: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
4 => {
let mut __v6: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v6 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v6;
}
8 => {
let mut __v7: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v7 = pfo_val__ as i32;
pscr_ret__ = __v7;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v1 = pscr_ret__;
__v0 = __v1;
cpu = __v0 as u32;
let con: *mut console = unsafe { (*ctxt).console };
let mut new: nbcon_state = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(new) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let mut timeout: i32 = 0;
let mut request_err: i32 = 16i32.wrapping_neg();
// Check that the handover is called when the direct acquire failed
// with -EBUSY.
let mut __v8: i64;
{
let __ret_warn_on: i32 = (((unsafe { (*ctxt).prio }) as u32) <= (((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_prio() }) as i32) as u32) || ((unsafe { (*ctxt).prio }) as u32) <= (((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_req_prio() }) as i32) as u32)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("372: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 372b - ., 3; .popsection\n# {0}", const 372 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_nbcon_context_try_acquire_handover_9, sym __slate_asm_string_nbcon_context_try_acquire_handover_10, const 447 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("373: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 373b - ., 4; .popsection\n# {0}", const 373 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v8 = (__ret_warn_on != 0) as i64;
}
let mut __v13: i64;
let __ret_warn_on: i32 = (((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe() }) as i32) == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("374: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 374b - ., 3; .popsection\n# {0}", const 374 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_nbcon_context_try_acquire_handover_14, sym __slate_asm_string_nbcon_context_try_acquire_handover_15, const 448 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("375: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 375b - ., 4; .popsection\n# {0}", const 375 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v13 = (__ret_warn_on != 0) as i64;
// Panic does not imply that the console is owned. However, it
// is critical that non-panic CPUs during panic are unable to
// wait for a handover in order to satisfy the assumptions of
// nbcon_waiter_matches(). In particular, the assumption that
// lower priorities are ignored during panic.
if unsafe { panic_on_other_cpu() } {
return 1i32.wrapping_neg();
}
// Handover is not possible on the same CPU.
if (((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_cpu() }) as i32) as u32) == cpu {
return 16i32.wrapping_neg();
}
// Console stays unsafe after an unsafe takeover until re-initialized.
// Waiting is not going to help in this case.
if ((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe_takeover() }) as i32) != 0 {
return 16i32.wrapping_neg();
}
// Is the caller willing to wait?
if (unsafe { (*ctxt).spinwait_max_us }) == 0 {
return 16i32.wrapping_neg();
}
// Setup a request for the handover. The caller should try to acquire
// the console directly when the current state has been modified.
unsafe {
new.__slate_anon_0.atom = (*(cur as *mut nbcon_state)).__slate_anon_0.atom;
}
unsafe {
let __t4 = (*ctxt).prio as u32;
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_req_prio(__t4);
}
if !nbcon_state_try_cmpxchg(con, cur, unsafe { &(*core::ptr::addr_of_mut!(new)) }) {
return 11i32.wrapping_neg();
}
unsafe {
(*(cur as *mut nbcon_state)).__slate_anon_0.atom = new.__slate_anon_0.atom;
}
// Wait until there is no owner and then acquire the console.
timeout = (unsafe { (*ctxt).spinwait_max_us }) as i32;
while timeout >= 0 {
// On successful acquire, this request is cleared.
request_err = nbcon_context_try_acquire_requested(ctxt, cur);
if request_err == 0 {
return 0;
}
// If the acquire should be aborted, it must be ensured
// that the request is removed before returning to caller.
if request_err == 1i32.wrapping_neg() {
break;
}
udelay(1u64);
// Re-read the state because some time has passed.
nbcon_state_read(con, cur);
timeout = timeout.wrapping_sub(1);
}
// Timed out or aborted. Carefully remove handover request.
loop {
// No need to remove request if there is a new waiter. This
// can only happen if a higher priority context has taken over
// the console or the handover request.
if !nbcon_waiter_matches(cur, ((unsafe { (*ctxt).prio }) as u32) as i32) {
return 1i32.wrapping_neg();
}
// Unset request for handover.
unsafe {
new.__slate_anon_0.atom = (*(cur as *mut nbcon_state)).__slate_anon_0.atom;
}
unsafe {
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_req_prio(0u32);
}
if nbcon_state_try_cmpxchg(con, cur, unsafe { &(*core::ptr::addr_of_mut!(new)) }) {
// Request successfully unset. Report failure of
// acquiring via handover.
unsafe {
(*(cur as *mut nbcon_state)).__slate_anon_0.atom = new.__slate_anon_0.atom;
}
return request_err;
}
// Unable to remove request. Try to acquire in case
// the owner has released the lock.
if nbcon_context_try_acquire_requested(ctxt, cur) == 0 {
break;
}
}
// Lucky timing. The acquire succeeded while removing the request.
0
}

/// nbcon_context_try_acquire_hostile - Acquire via unsafe hostile takeover
/// @ctxt:	The context of the caller
/// @cur:	The current console state
///
/// Acquire the console even in the unsafe state.
///
/// It can be permitted by setting the 'allow_unsafe_takeover' field only
/// by the final attempt to flush messages in panic().
///
/// Return:	0 on success. -EPERM when not allowed by the context.
fn nbcon_context_try_acquire_hostile(ctxt: *mut nbcon_context, cur: &mut nbcon_state) -> i32 {
let mut cpu: u32 = 0;
let mut __v0: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v1: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v3: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v3 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v3;
}
2 => {
let mut __v5: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
4 => {
let mut __v6: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v6 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v6;
}
8 => {
let mut __v7: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v7 = pfo_val__ as i32;
pscr_ret__ = __v7;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v1 = pscr_ret__;
__v0 = __v1;
cpu = __v0 as u32;
let con: *mut console = unsafe { (*ctxt).console };
let mut new: nbcon_state = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(new) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
if ((unsafe { (*ctxt).__slate_bits_0.__get_allow_unsafe_takeover() }) as i32) == 0 {
return 1i32.wrapping_neg();
}
// Ensure caller is allowed to perform unsafe hostile takeovers.
let mut __v8: i64;
{
let __ret_warn_on: i32 = (((unsafe { (*ctxt).prio }) as u32) != 3) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("376: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 376b - ., 3; .popsection\n# {0}", const 376 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_nbcon_context_try_acquire_hostile_9, sym __slate_asm_string_nbcon_context_try_acquire_hostile_10, const 561 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("377: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 377b - ., 4; .popsection\n# {0}", const 377 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v8 = (__ret_warn_on != 0) as i64;
}
if __v8 != 0 {
return 1i32.wrapping_neg();
}
// Check that try_acquire_direct() and try_acquire_handover() returned
// -EBUSY in the right situation.
let mut __v13: i64;
{
let __ret_warn_on: i32 = (((unsafe { (*ctxt).prio }) as u32) <= (((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_prio() }) as i32) as u32) || ((unsafe { (*ctxt).prio }) as u32) <= (((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_req_prio() }) as i32) as u32)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("378: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 378b - ., 3; .popsection\n# {0}", const 378 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_nbcon_context_try_acquire_hostile_14, sym __slate_asm_string_nbcon_context_try_acquire_hostile_15, const 568 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("379: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 379b - ., 4; .popsection\n# {0}", const 379 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v13 = (__ret_warn_on != 0) as i64;
}
let mut __v18: i64;
let __ret_warn_on: i32 = (((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe() }) as i32) != 1) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("380: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 380b - ., 3; .popsection\n# {0}", const 380 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_nbcon_context_try_acquire_hostile_19, sym __slate_asm_string_nbcon_context_try_acquire_hostile_20, const 569 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("381: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 381b - ., 4; .popsection\n# {0}", const 381 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v18 = (__ret_warn_on != 0) as i64;
loop {
unsafe {
new.__slate_anon_0.atom = (*(cur as *mut nbcon_state)).__slate_anon_0.atom;
}
unsafe {
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_cpu(cpu);
}
unsafe {
let __t4 = (*ctxt).prio as u32;
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_prio(__t4);
}
let __v23: u32 = unsafe { new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe() };
let __v24: u32 = ((__v23 as i32) | ((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe_takeover() }) as i32)) as u32;
unsafe {
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_unsafe(__v24);
}
let __v25: u32 = unsafe { new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe_takeover() };
let __v26: u32 = ((__v25 as i32) | ((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe() }) as i32)) as u32;
unsafe {
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_unsafe_takeover(__v26);
}
if !(!nbcon_state_try_cmpxchg(con, cur, unsafe { &(*core::ptr::addr_of_mut!(new)) })) {
break;
}
}
0
}

static mut panic_nbcon_pbufs: printk_buffers = unsafe { core::mem::zeroed() };

/// nbcon_context_try_acquire - Try to acquire nbcon console
/// @ctxt:		The context of the caller
/// @is_reacquire:	This acquire is a reacquire
///
/// Context:	Under @ctxt->con->device_lock() or local_irq_save().
/// Return:	True if the console was acquired. False otherwise.
///
/// If the caller allowed an unsafe hostile takeover, on success the
/// caller should check the current console state to see if it is
/// in an unsafe state. Otherwise, on success the caller may assume
/// the console is not in an unsafe state.
fn nbcon_context_try_acquire(ctxt: *mut nbcon_context, is_reacquire: bool) -> bool {
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut cur: core::mem::MaybeUninit<nbcon_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut nbcon_state = core::ptr::addr_of_mut!(cur) as *mut nbcon_state;
let mut con: *mut console = unsafe { core::mem::zeroed() };
unsafe {
con = (*ctxt).console;
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
nbcon_state_read(con, &mut (*core::ptr::addr_of_mut!(*__slate_slot_1)));
'__join_5: {
loop {
*__slate_slot_0 = nbcon_context_try_acquire_direct(ctxt, &mut (*core::ptr::addr_of_mut!(*__slate_slot_1)), is_reacquire);
if *__slate_slot_0 != 16i32.wrapping_neg() {
break '__join_5;
} else {
*__slate_slot_0 = nbcon_context_try_acquire_handover(ctxt, &mut (*core::ptr::addr_of_mut!(*__slate_slot_1)));
if *__slate_slot_0 != 11i32.wrapping_neg() {
break;
}
}
}
if *__slate_slot_0 == 16i32.wrapping_neg() {
*__slate_slot_0 = nbcon_context_try_acquire_hostile(ctxt, &mut (*core::ptr::addr_of_mut!(*__slate_slot_1)));
}
}
if *__slate_slot_0 != 0 {
return false;
} else {
// Acquire succeeded.
// Assign the appropriate buffer for this context.
if panic_on_this_cpu() {
(*ctxt).pbufs = core::ptr::addr_of_mut!(panic_nbcon_pbufs);
} else {
(*ctxt).pbufs = (*con).pbufs;
}
// Set the record sequence for this context to print.
(*ctxt).seq = nbcon_seq_read((*ctxt).console);
return true;
}
}
}

fn nbcon_owner_matches(cur: &mut nbcon_state, expected_cpu: i32, expected_prio: i32) -> bool {
// A similar function, nbcon_waiter_matches(), only deals with
// EMERGENCY and PANIC priorities. However, this function must also
// deal with the NORMAL priority, which requires additional checks
// and constraints.
//
// For the case where preemption and interrupts are disabled, it is
// enough to also verify that the owning CPU has not changed.
//
// For the case where preemption or interrupts are enabled, an
// external synchronization method *must* be used. In particular,
// the driver-specific locking mechanism used in device_lock()
// (including disabling migration) should be used. It prevents
// scenarios such as:
//
// 1. [Task A] owns a context with NBCON_PRIO_NORMAL on [CPU X] and
//    is scheduled out.
// 2. Another context takes over the lock with NBCON_PRIO_EMERGENCY
//    and releases it.
// 3. [Task B] acquires a context with NBCON_PRIO_NORMAL on [CPU X]
//    and is scheduled out.
// 4. [Task A] gets running on [CPU X] and sees that the console is
//    still owned by a task on [CPU X] with NBON_PRIO_NORMAL. Thus
//    [Task A] thinks it is the owner when it is not.
if ((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_prio() }) as i32) != expected_prio {
return false;
}
if ((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_cpu() }) as i32) != expected_cpu {
return false;
}
true
}

/// nbcon_context_release - Release the console
/// @ctxt:	The nbcon context from nbcon_context_try_acquire()
fn nbcon_context_release(ctxt: *mut nbcon_context) {
let mut cpu: u32 = 0;
let mut __v0: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v1: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v3: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v3 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v3;
}
2 => {
let mut __v5: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
4 => {
let mut __v6: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v6 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v6;
}
8 => {
let mut __v7: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v7 = pfo_val__ as i32;
pscr_ret__ = __v7;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v1 = pscr_ret__;
__v0 = __v1;
cpu = __v0 as u32;
let con: *mut console = unsafe { (*ctxt).console };
let mut cur: nbcon_state = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(cur) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let mut new: nbcon_state = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(new) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
nbcon_state_read(con, unsafe { &mut (*core::ptr::addr_of_mut!(cur)) });
while nbcon_owner_matches(unsafe { &mut (*core::ptr::addr_of_mut!(cur)) }, cpu as i32, ((unsafe { (*ctxt).prio }) as u32) as i32) {
unsafe {
new.__slate_anon_0.atom = cur.__slate_anon_0.atom;
}
unsafe {
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_prio(0u32);
}
// If @unsafe_takeover is set, it is kept set so that
// the state remains permanently unsafe.
let __v9: u32 = unsafe { new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe() };
let __v10: u32 = ((__v9 as i32) | ((unsafe { cur.__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe_takeover() }) as i32)) as u32;
unsafe {
new.__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_unsafe(__v10);
}
if !(!nbcon_state_try_cmpxchg(con, unsafe { &mut (*core::ptr::addr_of_mut!(cur)) }, unsafe { &(*core::ptr::addr_of_mut!(new)) })) {
break;
}
}
unsafe {
(*ctxt).pbufs = core::ptr::null_mut::<printk_buffers>();
}
}

/// nbcon_context_can_proceed - Check whether ownership can proceed
/// @ctxt:	The nbcon context from nbcon_context_try_acquire()
/// @cur:	The current console state
///
/// Return:	True if this context still owns the console. False if
/// 		ownership was handed over or taken.
///
/// Must be invoked when entering the unsafe state to make sure that it still
/// owns the lock. Also must be invoked when exiting the unsafe context
/// to eventually free the lock for a higher priority context which asked
/// for the friendly handover.
///
/// It can be called inside an unsafe section when the console is just
/// temporary in safe state instead of exiting and entering the unsafe
/// state.
///
/// Also it can be called in the safe context before doing an expensive
/// safe operation. It does not make sense to do the operation when
/// a higher priority context took the lock.
///
/// When this function returns false then the calling context no longer owns
/// the console and is no longer allowed to go forward. In this case it must
/// back out immediately and carefully. The buffer content is also no longer
/// trusted since it no longer belongs to the calling context.
fn nbcon_context_can_proceed(ctxt: *mut nbcon_context, cur: &mut nbcon_state) -> bool {
let mut cpu: u32 = 0;
let mut __v0: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v1: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v3: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v3 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v3;
}
2 => {
let mut __v5: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
4 => {
let mut __v6: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v6 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v6;
}
8 => {
let mut __v7: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v7 = pfo_val__ as i32;
pscr_ret__ = __v7;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v1 = pscr_ret__;
__v0 = __v1;
cpu = __v0 as u32;
// Make sure this context still owns the console.
if !nbcon_owner_matches(cur, cpu as i32, ((unsafe { (*ctxt).prio }) as u32) as i32) {
return false;
}
// The console owner can proceed if there is no waiter.
if ((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_req_prio() }) as i32) == 0 {
return true;
}
// A console owner within an unsafe region is always allowed to
// proceed, even if there are waiters. It can perform a handover
// when exiting the unsafe region. Otherwise the waiter will
// need to perform an unsafe hostile takeover.
if ((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe() }) as i32) != 0 {
return true;
}
// Waiters always have higher priorities than owners.
let mut __v8: i64;
let __ret_warn_on: i32 = (((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_req_prio() }) as i32) <= ((unsafe { (*(cur as *mut nbcon_state)).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_prio() }) as i32)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("382: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 382b - ., 3; .popsection\n# {0}", const 382 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_nbcon_context_can_proceed_9, sym __slate_asm_string_nbcon_context_can_proceed_10, const 752 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("383: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 383b - ., 4; .popsection\n# {0}", const 383 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v8 = (__ret_warn_on != 0) as i64;
// Having a safe point for take over and eventually a few
// duplicated characters or a full line is way better than a
// hostile takeover. Post processing can take care of the garbage.
// Release and hand over.
nbcon_context_release(ctxt);
// It is not clear whether the waiter really took over ownership. The
// outermost callsite must make the final decision whether console
// ownership is needed for it to proceed. If yes, it must reacquire
// ownership (possibly hostile) before carefully proceeding.
//
// The calling context no longer owns the console so go back all the
// way instead of trying to implement reacquire heuristics in tons of
// places.
false
}

/// nbcon_can_proceed - Check whether ownership can proceed
/// @wctxt:	The write context that was handed to the write function
///
/// Return:	True if this context still owns the console. False if
/// 		ownership was handed over or taken.
///
/// It is used in nbcon_enter_unsafe() to make sure that it still owns the
/// lock. Also it is used in nbcon_exit_unsafe() to eventually free the lock
/// for a higher priority context which asked for the friendly handover.
///
/// It can be called inside an unsafe section when the console is just
/// temporary in safe state instead of exiting and entering the unsafe state.
///
/// Also it can be called in the safe context before doing an expensive safe
/// operation. It does not make sense to do the operation when a higher
/// priority context took the lock.
///
/// When this function returns false then the calling context no longer owns
/// the console and is no longer allowed to go forward. In this case it must
/// back out immediately and carefully. The buffer content is also no longer
/// trusted since it no longer belongs to the calling context.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.nbcon.nbcon_can_proceed")]
extern "C" fn nbcon_can_proceed(wctxt: *mut nbcon_write_context) -> bool {
let ctxt: *mut nbcon_context = unsafe { core::ptr::addr_of_mut!((*wctxt).ctxt) };
let con: *mut console = unsafe { (*ctxt).console };
let mut cur: nbcon_state = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(cur) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
nbcon_state_read(con, unsafe { &mut (*core::ptr::addr_of_mut!(cur)) });
nbcon_context_can_proceed(ctxt, unsafe { &mut (*core::ptr::addr_of_mut!(cur)) })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nbcon_can_proceed_384: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut nbcon_write_context) -> bool>, *mut ()>(Some(nbcon_can_proceed)) };

/// __nbcon_context_update_unsafe - Update the unsafe bit in @con->nbcon_state
/// @ctxt:	The nbcon context from nbcon_context_try_acquire()
/// @unsafe:	The new value for the unsafe bit
///
/// Return:	True if the unsafe state was updated and this context still
/// 		owns the console. Otherwise false if ownership was handed
/// 		over or taken.
///
/// This function allows console owners to modify the unsafe status of the
/// console.
///
/// When this function returns false then the calling context no longer owns
/// the console and is no longer allowed to go forward. In this case it must
/// back out immediately and carefully. The buffer content is also no longer
/// trusted since it no longer belongs to the calling context.
///
/// Internal helper to avoid duplicated code.
fn __nbcon_context_update_unsafe(ctxt: *mut nbcon_context, r#unsafe: bool) -> bool {
let mut new: core::mem::MaybeUninit<nbcon_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut nbcon_state = core::ptr::addr_of_mut!(new) as *mut nbcon_state;
let mut cur: core::mem::MaybeUninit<nbcon_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut nbcon_state = core::ptr::addr_of_mut!(cur) as *mut nbcon_state;
let mut con: *mut console = unsafe { core::mem::zeroed() };
unsafe {
con = (*ctxt).console;
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
nbcon_state_read(con, &mut (*core::ptr::addr_of_mut!(*__slate_slot_1)));
'__join_0: {
loop {
// The unsafe bit must not be cleared if an
// unsafe hostile takeover has occurred.
if !r#unsafe && ((*__slate_slot_1).__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe_takeover() as i32) != 0 {
break '__join_0;
} else {
if !nbcon_context_can_proceed(ctxt, &mut (*core::ptr::addr_of_mut!(*__slate_slot_1))) {
return false;
} else {
(*__slate_slot_0).__slate_anon_0.atom = (*__slate_slot_1).__slate_anon_0.atom;
(*__slate_slot_0).__slate_anon_0.__slate_anon_1.__slate_bits_0.__set_unsafe(r#unsafe as u32);
if nbcon_state_try_cmpxchg(con, &mut (*core::ptr::addr_of_mut!(*__slate_slot_1)), &(*core::ptr::addr_of_mut!(*__slate_slot_0))) {
break;
}
}
}
}
(*__slate_slot_1).__slate_anon_0.atom = (*__slate_slot_0).__slate_anon_0.atom;
}
return nbcon_context_can_proceed(ctxt, &mut (*core::ptr::addr_of_mut!(*__slate_slot_1)));
}
}

#[unsafe(no_mangle)]
extern "C" fn nbcon_write_context_set_buf(wctxt: *mut nbcon_write_context, buf: *mut u8, len: u32) {
let ctxt: *mut nbcon_context = unsafe { core::ptr::addr_of_mut!((*wctxt).ctxt) };
let con: *mut console = unsafe { (*ctxt).console };
let mut cur: nbcon_state = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(cur) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
unsafe {
(*wctxt).outbuf = buf;
}
unsafe {
(*wctxt).len = len;
}
nbcon_state_read(con, unsafe { &mut (*core::ptr::addr_of_mut!(cur)) });
unsafe {
(*wctxt).unsafe_takeover = (cur.__slate_anon_0.__slate_anon_1.__slate_bits_0.__get_unsafe_takeover() as i32) != 0;
}
}

/// nbcon_enter_unsafe - Enter an unsafe region in the driver
/// @wctxt:	The write context that was handed to the write function
///
/// Return:	True if this context still owns the console. False if
/// 		ownership was handed over or taken.
///
/// When this function returns false then the calling context no longer owns
/// the console and is no longer allowed to go forward. In this case it must
/// back out immediately and carefully. The buffer content is also no longer
/// trusted since it no longer belongs to the calling context.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.nbcon.nbcon_enter_unsafe")]
extern "C" fn nbcon_enter_unsafe(wctxt: *mut nbcon_write_context) -> bool {
let ctxt: *mut nbcon_context = unsafe { core::ptr::addr_of_mut!((*wctxt).ctxt) };
let mut is_owner: bool = false;
is_owner = nbcon_context_enter_unsafe!(ctxt);
if !is_owner {
nbcon_write_context_set_buf(wctxt, core::ptr::null_mut::<u8>(), 0u32);
}
is_owner
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nbcon_enter_unsafe_385: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut nbcon_write_context) -> bool>, *mut ()>(Some(nbcon_enter_unsafe)) };

/// nbcon_exit_unsafe - Exit an unsafe region in the driver
/// @wctxt:	The write context that was handed to the write function
///
/// Return:	True if this context still owns the console. False if
/// 		ownership was handed over or taken.
///
/// When this function returns false then the calling context no longer owns
/// the console and is no longer allowed to go forward. In this case it must
/// back out immediately and carefully. The buffer content is also no longer
/// trusted since it no longer belongs to the calling context.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.nbcon.nbcon_exit_unsafe")]
extern "C" fn nbcon_exit_unsafe(wctxt: *mut nbcon_write_context) -> bool {
let ctxt: *mut nbcon_context = unsafe { core::ptr::addr_of_mut!((*wctxt).ctxt) };
let mut ret: bool = false;
ret = nbcon_context_exit_unsafe!(ctxt);
if !ret {
nbcon_write_context_set_buf(wctxt, core::ptr::null_mut::<u8>(), 0u32);
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nbcon_exit_unsafe_386: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut nbcon_write_context) -> bool>, *mut ()>(Some(nbcon_exit_unsafe)) };

/// nbcon_reacquire_nobuf - Reacquire a console after losing ownership
/// 				while printing
/// @wctxt:	The write context that was handed to the write callback
///
/// Since ownership can be lost at any time due to handover or takeover, a
/// printing context _must_ be prepared to back out immediately and
/// carefully. However, there are scenarios where the printing context must
/// reacquire ownership in order to finalize or revert hardware changes.
///
/// This function allows a printing context to reacquire ownership using the
/// same priority as its previous ownership.
///
/// Note that after a successful reacquire the printing context will have no
/// output buffer because that has been lost. This function cannot be used to
/// resume printing.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.nbcon.nbcon_reacquire_nobuf")]
extern "C" fn nbcon_reacquire_nobuf(wctxt: *mut nbcon_write_context) {
let ctxt: *mut nbcon_context = unsafe { core::ptr::addr_of_mut!((*wctxt).ctxt) };
while !nbcon_context_try_acquire(ctxt, true) {
cpu_relax();
}
nbcon_write_context_set_buf(wctxt, core::ptr::null_mut::<u8>(), 0u32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nbcon_reacquire_nobuf_387: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut nbcon_write_context)>, *mut ()>(Some(nbcon_reacquire_nobuf)) };

fn wctxt_load_execution_ctx(wctxt: *mut nbcon_write_context, pmsg: &printk_message) {
}

// nbcon_emit_next_record - Emit a record in the acquired context
// @wctxt:	The write context that will be handed to the write function
// @use_atomic:	True if the write_atomic() callback is to be used
//
// Return:	True if this context still owns the console. False if
// 		ownership was handed over or taken.
//
// When this function returns false then the calling context no longer owns
// the console and is no longer allowed to go forward. In this case it must
// back out immediately and carefully. The buffer content is also no longer
// trusted since it no longer belongs to the calling context. If the caller
// wants to do more it must reacquire the console first.
//
// When true is returned, @wctxt->ctxt.backlog indicates whether there are
// still records pending in the ringbuffer,
fn nbcon_emit_next_record(wctxt: *mut nbcon_write_context, use_atomic: bool) -> bool {
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut ulseq: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(ulseq) as *mut u64;
let mut dropped: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u64 = core::ptr::addr_of_mut!(dropped) as *mut u64;
let mut cur: core::mem::MaybeUninit<nbcon_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut nbcon_state = core::ptr::addr_of_mut!(cur) as *mut nbcon_state;
let mut con_dropped: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(con_dropped) as *mut u64;
let mut pmsg: core::mem::MaybeUninit<printk_message> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut printk_message = core::ptr::addr_of_mut!(pmsg) as *mut printk_message;
let mut is_extended: bool = unsafe { core::mem::zeroed() };
let mut con: *mut console = unsafe { core::mem::zeroed() };
let mut ctxt: *mut nbcon_context = unsafe { core::mem::zeroed() };
unsafe {
ctxt = core::ptr::addr_of_mut!((*wctxt).ctxt);
con = (*ctxt).console;
is_extended = (console_srcu_read_flags(con as *const console) as i32) & 64 != 0;
core::ptr::write(__slate_slot_11, {
let mut __t1: printk_message = core::mem::zeroed();
__t1.pbufs = (*ctxt).pbufs;
__t1
});
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if use_atomic && (*con).write_atomic == None {
__v4 = true;
} else {
__v4 = (console_srcu_read_flags(con as *const console) as i32) & 256 == 0;
}
__ret_warn_on = __v4 as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("388: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 388b - ., 3; .popsection\n# {0}", const 388 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_nbcon_emit_next_record_16, sym __slate_asm_string_nbcon_emit_next_record_17, const 1003 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("389: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 389b - ., 4; .popsection\n# {0}", const 389 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// This function should never be called for consoles that have not
// implemented the necessary callback for writing: i.e. legacy
// consoles and, when atomic, nbcon consoles with no write_atomic().
// Handle it as if ownership was lost and try to continue.
//
// Note that for nbcon consoles the write_thread() callback is
// mandatory and was already checked in nbcon_alloc().
let __v6: i64 = (__ret_warn_on != 0) as i64;
if __v6 != 0 {
nbcon_context_release(ctxt);
return false;
} else {
// The printk buffers are filled within an unsafe section. This
// prevents NBCON_PRIO_NORMAL and NBCON_PRIO_EMERGENCY from
// clobbering each other.
if !nbcon_context_enter_unsafe!(ctxt) {
return false;
} else {
let __t0 = printk_get_next_message(core::ptr::addr_of_mut!(*__slate_slot_11), (*ctxt).seq, is_extended, true) as u32;
(*ctxt).__slate_bits_0.__set_backlog(__t0);
if ((*ctxt).__slate_bits_0.__get_backlog() as i32) == 0 {
return nbcon_context_exit_unsafe!(ctxt);
} else {
__kcsan_disable_current();
while false {
}
let __v0: u64 = core::ptr::read_volatile(core::ptr::addr_of_mut!((*con).dropped) as *const u64);
let __v: u64 = __v0;
__kcsan_enable_current();
// @con->dropped is not protected in case of an unsafe hostile
// takeover. In that situation the update can be racy so
// annotate it accordingly.
*__slate_slot_10 = __v;
*__slate_slot_8 = (*__slate_slot_10).wrapping_add((*__slate_slot_11).dropped);
if *__slate_slot_8 != 0 && !is_extended {
console_prepend_dropped(core::ptr::addr_of_mut!(*__slate_slot_11), *__slate_slot_8);
}
// If the previous owner was assigned the same record, this context
// has taken over ownership and is replaying the record. Prepend a
// message to let the user know the record is replayed.
*__slate_slot_7 = atomic_long_read(core::ptr::addr_of_mut!((*con).nbcon_prev_seq) as *const __SlateRecord_types_h_197) as u64;
if __ulseq_to_u64seq!(*__slate_slot_7) == (*__slate_slot_11).seq {
console_prepend_replay(core::ptr::addr_of_mut!(*__slate_slot_11));
} else {
// Ensure this context is still the owner before trying to
// update @nbcon_prev_seq. Otherwise the value in @ulseq may
// not be from the previous owner and instead be some later
// value from the context that took over ownership.
nbcon_state_read(con, &mut (*core::ptr::addr_of_mut!(*__slate_slot_9)));
if !nbcon_context_can_proceed(ctxt, &mut (*core::ptr::addr_of_mut!(*__slate_slot_9))) {
return false;
} else {
atomic_long_try_cmpxchg(core::ptr::addr_of_mut!((*con).nbcon_prev_seq), core::ptr::addr_of_mut!(*__slate_slot_7) as *mut i64, __u64seq_to_ulseq!((*__slate_slot_11).seq) as i64);
}
}
if !nbcon_context_exit_unsafe!(ctxt) {
return false;
} else {
// For skipped records just update seq/dropped in @con.
if (*__slate_slot_11).outbuf_len != 0 {
// Initialize the write context for driver callbacks.
nbcon_write_context_set_buf(wctxt, ((*(*__slate_slot_11).pbufs).outbuf.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize), (*__slate_slot_11).outbuf_len);
wctxt_load_execution_ctx(wctxt, &(*core::ptr::addr_of_mut!(*__slate_slot_11)));
if use_atomic {
unsafe { (*con).write_atomic }.unwrap()(con, wctxt);
} else {
unsafe { (*con).write_thread }.unwrap()(con, wctxt);
}
if unsafe { (*wctxt).outbuf }.is_null() {
// Ownership was lost and reacquired by the driver. Handle it
// as if ownership was lost.
nbcon_context_release(ctxt);
return false;
} else {
// Ownership may have been lost but _not_ reacquired by the driver.
// This case is detected and handled when entering unsafe to update
// dropped/seq values.
// Since any dropped message was successfully output, reset the
// dropped count for the console.
*__slate_slot_8 = 0;
}
}
if !nbcon_context_enter_unsafe!(ctxt) {
return false;
} else {
// The dropped count and the sequence number are updated within an
// unsafe section. This limits update races to the panic context and
// allows the panic context to win.
if *__slate_slot_8 != *__slate_slot_10 {
// Counterpart to the READ_ONCE() above.
loop {
if true {
__WRITE_ONCE__2!((*con).dropped, *__slate_slot_8);
if true {
break;
}
}
}
}
nbcon_seq_try_update(ctxt, (*__slate_slot_11).seq.wrapping_add(1));
return nbcon_context_exit_unsafe!(ctxt);
}
}
}
}
}
}
}

/// nbcon_emit_one - Print one record for an nbcon console using the
/// 			specified callback
/// @wctxt:	An initialized write context struct to use for this context
/// @use_atomic:	True if the write_atomic() callback is to be used
///
/// Return:	True, when a record has been printed and there are still
/// 		pending records. The caller might want to continue flushing.
///
/// 		False, when there is no pending record, or when the console
/// 		context cannot be acquired, or the ownership has been lost.
/// 		The caller should give up. Either the job is done, cannot be
/// 		done, or will be handled by the owning context.
///
/// This is an internal helper to handle the locking of the console before
/// calling nbcon_emit_next_record().
fn nbcon_emit_one(wctxt: *mut nbcon_write_context, use_atomic: bool) -> bool {
let mut ret: bool = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut con: *mut console = unsafe { core::mem::zeroed() };
let mut ctxt: *mut nbcon_context = unsafe { core::mem::zeroed() };
unsafe {
ctxt = core::ptr::addr_of_mut!((*wctxt).ctxt);
con = (*ctxt).console;
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = false;
if !use_atomic {
unsafe { (*con).device_lock }.unwrap()(con, core::ptr::addr_of_mut!(*__slate_slot_1));
// Ensure this stays on the CPU to make handover and
// takeover possible.
cant_migrate!();
}
if nbcon_context_try_acquire(ctxt, false) {
// nbcon_emit_next_record() returns false when the console was
// handed over or taken over. In both cases the context is no
// longer valid.
//
// The higher priority printing context takes over responsibility
// to print the pending records.
if nbcon_emit_next_record(wctxt, use_atomic) {
nbcon_context_release(ctxt);
ret = ((*ctxt).__slate_bits_0.__get_backlog() as i32) != 0;
}
}
if !use_atomic {
unsafe { (*con).device_unlock }.unwrap()(con, *__slate_slot_1);
}
return ret;
}
}

/// nbcon_kthread_should_wakeup - Check whether a printer thread should wakeup
/// @con:	Console to operate on
/// @ctxt:	The nbcon context from nbcon_context_try_acquire()
///
/// Return:	True if the thread should shutdown or if the console is
/// 		allowed to print and a record is available. False otherwise.
///
/// After the thread wakes up, it must first check if it should shutdown before
/// attempting any printing.
fn nbcon_kthread_should_wakeup(con: *mut console, ctxt: *mut nbcon_context) -> bool {
let mut ret: bool = false;
let mut flags: i16 = 0;
let mut cookie: i32 = 0;
if unsafe { kthread_should_stop() } {
return true;
}
// Block the kthread when the system is in an emergency or panic mode.
// It increases the chance that these contexts would be able to show
// the messages directly. And it reduces the risk of interrupted writes
// where the context with a higher priority takes over the nbcon console
// ownership in the middle of a message.
let __v0: bool = unlikely!(atomic_read((unsafe { core::ptr::addr_of_mut!(nbcon_cpu_emergency_cnt) }) as *const __SlateRecord_types_h_190)) != 0 || unlikely__2!(unsafe { panic_in_progress() }) != 0;
if __v0 {
return false;
}
cookie = unsafe { console_srcu_read_lock() };
flags = console_srcu_read_flags(con as *const console);
if console_is_usable(con, flags, false) {
// Bring the sequence in @ctxt up to date
unsafe {
(*ctxt).seq = nbcon_seq_read(con);
}
ret = unsafe { prb_read_valid(prb, (*ctxt).seq, core::ptr::null_mut::<printk_record>()) };
}
unsafe { console_srcu_read_unlock(cookie) };
ret
}

/// nbcon_kthread_func - The printer thread function
/// @__console:	Console to operate on
///
/// Return:	0
#[unsafe(link_section = ".text.slate_distinct.nbcon.nbcon_kthread_func")]
extern "C" fn nbcon_kthread_func(__console: *mut ()) -> i32 {
// Block the kthread when the system is in an emergency or panic
// mode. See nbcon_kthread_should_wakeup() for more details.
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __ret: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u32 = core::ptr::addr_of_mut!(__ret) as *mut u32;
let mut __ai_ptr: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut u32 = core::ptr::addr_of_mut!(__ai_ptr) as *mut *mut u32;
let mut __ret: i64 = unsafe { core::mem::zeroed() };
let mut cookie: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(cookie) as *mut i32;
let mut backlog: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut bool = core::ptr::addr_of_mut!(backlog) as *mut bool;
let mut con_flags: core::mem::MaybeUninit<i16> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i16 = core::ptr::addr_of_mut!(con_flags) as *mut i16;
let mut ctxt: *mut nbcon_context = unsafe { core::mem::zeroed() };
let mut wctxt: core::mem::MaybeUninit<nbcon_write_context> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut nbcon_write_context = core::ptr::addr_of_mut!(wctxt) as *mut nbcon_write_context;
let mut con: *mut console = unsafe { core::mem::zeroed() };
unsafe {
con = __console as *mut console;
core::ptr::write(__slate_slot_11, {
let mut __t1: nbcon_write_context = core::mem::zeroed();
__t1.ctxt = {
let mut __t2: nbcon_context = core::mem::zeroed();
__t2.console = con;
__t2.prio = 1u32;
__t2
};
__t1
});
ctxt = core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(*__slate_slot_11)).ctxt);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
'__loop_19: loop {
__ret = 0;
prepare_to_rcuwait(core::ptr::addr_of_mut!((*con).rcuwait));
'__join_9: {
loop {
debug_normal_state_change!();
trace_set_current_state!(1i32);
loop {
kcsan_mb!();
loop {
core::ptr::write(__slate_slot_3, core::ptr::addr_of_mut!((*get_current()).__state));
kcsan_mb!();
instrument_atomic_read_write(*__slate_slot_3 as *const (), core::mem::size_of::<u32>() as u64);
core::ptr::write(__slate_slot_1, 1u32);
let __t0: u32;
core::arch::asm!("xchgl {0:e}, ({1:r})", inlateout(reg) *__slate_slot_1 as u32 => __t0, in(reg) *__slate_slot_3, options(nostack, att_syntax));
*__slate_slot_1 = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
if true {
break;
}
}
if true {
break;
}
}
if true {
if nbcon_kthread_should_wakeup(con, ctxt) {
break '__join_9;
} else {
if signal_pending_state(1u32, get_current()) != 0 {
break;
} else {
schedule();
}
}
}
}
__ret = 4i32.wrapping_neg() as i64;
}
finish_rcuwait(core::ptr::addr_of_mut!((*con).rcuwait));
// Guarantee this task is visible on the rcuwait before
// checking the wake condition.
//
// The full memory barrier within set_current_state() of
// ___rcuwait_wait_event() pairs with the full memory
// barrier within rcuwait_has_sleeper().
//
// This pairs with rcuwait_has_sleeper:A and nbcon_kthread_wake:A.
// LMM(nbcon_kthread_func:A)
loop {
if kthread_should_stop() {
break '__loop_19;
} else {
if unlikely!(atomic_read(core::ptr::addr_of_mut!(nbcon_cpu_emergency_cnt) as *const __SlateRecord_types_h_190)) != 0 {
__v0 = true;
} else {
__v0 = unlikely__2!(panic_in_progress()) != 0;
}
if __v0 {
continue '__loop_19;
} else {
*__slate_slot_8 = false;
// Keep the srcu read lock around the entire operation so that
// synchronize_srcu() can guarantee that the kthread stopped
// or suspended printing.
*__slate_slot_7 = console_srcu_read_lock();
*__slate_slot_9 = console_srcu_read_flags(con as *const console);
if console_is_usable(con, *__slate_slot_9, false) {
*__slate_slot_8 = nbcon_emit_one(core::ptr::addr_of_mut!(*__slate_slot_11), false);
}
console_srcu_read_unlock(*__slate_slot_7);
cond_resched__5!();
if !(*__slate_slot_8) {
continue '__loop_19;
}
}
}
}
}
return 0;
}
}

/// nbcon_irq_work - irq work to wake console printer thread
/// @irq_work:	The irq work to operate on
#[unsafe(link_section = ".text.slate_distinct.nbcon.nbcon_irq_work")]
extern "C" fn nbcon_irq_work(irq_work: *mut irq_work) {
let con: *mut console = container_of__14!(irq_work, console);
nbcon_kthread_wake(con);
}

fn rcuwait_has_sleeper(w: *mut rcuwait) -> bool {
// Guarantee any new records can be seen by tasks preparing to wait
// before this context checks if the rcuwait is empty.
//
// This full memory barrier pairs with the full memory barrier within
// set_current_state() of ___rcuwait_wait_event(), which is called
// after prepare_to_rcuwait() adds the waiter but before it has
// checked the wait condition.
//
// This pairs with nbcon_kthread_func:A.
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
// LMM(rcuwait_has_sleeper:A)
rcuwait_active(w) != 0
}

// nbcon_kthreads_wake - Wake up printing threads using irq_work
/// nbcon_kthread_stop - Stop a console printer thread
/// @con:	Console to operate on
#[unsafe(no_mangle)]
extern "C" fn nbcon_kthread_stop(con: *mut console) {
lockdep_assert_console_list_lock_held();
if unsafe { (*con).kthread }.is_null() {
return;
}
unsafe { kthread_stop((*con).kthread) };
unsafe {
(*con).kthread = core::ptr::null_mut::<task_struct>();
}
}

/// nbcon_kthread_create - Create a console printer thread
/// @con:	Console to operate on
///
/// Return:	True if the kthread was started or already exists.
/// 		Otherwise false and @con must not be registered.
///
/// This function is called when it will be expected that nbcon consoles are
/// flushed using the kthread. The messages printed with NBCON_PRIO_NORMAL
/// will be no longer flushed by the legacy loop. This is why failure must
/// be fatal for console registration.
///
/// If @con was already registered and this function fails, @con must be
/// unregistered before the global state variable @printk_kthreads_running
/// can be set.
#[unsafe(no_mangle)]
extern "C" fn nbcon_kthread_create(con: *mut console) -> bool {
lockdep_assert_console_list_lock_held();
if !unsafe { (*con).kthread }.is_null() {
return true;
}
let mut __v0: *mut task_struct;
let __k: *mut task_struct = unsafe { kthread_create_on_node(Some(nbcon_kthread_func), con as *mut (), 1i32.wrapping_neg(), c"pr/%s%d".as_ptr() as *const u8, (*con).name.as_mut_ptr() as *mut u8, (*con).index as i32) };
if !IS_ERR(__k as *const ()) {
unsafe { wake_up_process(__k) };
}
__v0 = __k;
let kt: *mut task_struct = __v0;
if IS_ERR(kt as *const ()) {
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x013%s%sconsole [%s%d] failed to start printing thread\n".as_ptr() as *const u8, if ((*con).flags as i32) & 256 != 0 { c"".as_ptr() as *mut u8 } else { c"legacy ".as_ptr() as *mut u8 }, if ((*con).flags as i32) & 8 != 0 { c"boot".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, (*con).name.as_mut_ptr() as *mut u8, (*con).index as i32) };
return false;
}
unsafe {
(*con).kthread = kt;
}
// It is important that console printing threads are scheduled
// shortly after a printk call and with generous runtime budgets.
unsafe { sched_set_normal((*con).kthread, 20i32.wrapping_neg()) };
true
}

/// Track the nbcon emergency nesting per CPU.
#[unsafe(link_section = ".data..percpu")]
static mut nbcon_pcpu_emergency_nesting: u32 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.data")]
static mut early_nbcon_pcpu_emergency_nesting: u32 = unsafe { core::mem::zeroed() };

/// nbcon_get_cpu_emergency_nesting - Get the per CPU emergency nesting pointer
///
/// Context:	For reading, any context. For writing, any context which could
/// 		not be migrated to another CPU.
/// Return:	Either a pointer to the per CPU emergency nesting counter of
/// 		the current CPU or to the init data during early boot.
///
/// The function is safe for reading per-CPU variables in any context because
/// preemption is disabled if the current CPU is in the emergency state. See
/// also nbcon_cpu_emergency_enter().
#[inline(never)]
#[unsafe(link_section = ".ref.text")]
fn nbcon_get_cpu_emergency_nesting() -> *mut u32 {
// The value of __printk_percpu_data_ready gets set in normal
// context and before SMP initialization. As a result it could
// never change while inside an nbcon emergency section.
if !unsafe { printk_percpu_data_ready() } {
return unsafe { core::ptr::addr_of_mut!(early_nbcon_pcpu_emergency_nesting) };
}
let mut __v0: *mut u32;
__verify_pcpu_ptr__7!();
let mut __v2: *mut u32;
let mut tcp_ptr__: u64 = 0;
let mut __v3: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v3 = pfo_val__;
tcp_ptr__ = __v3;
let __v5: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(nbcon_pcpu_emergency_nesting) }) as u64);
tcp_ptr__ = __v5;
__v2 = tcp_ptr__ as *mut u32;
__v0 = __v2;
__v0
}

/// nbcon_get_default_prio - The appropriate nbcon priority to use for nbcon
/// 				printing on the current CPU
///
/// Context:	Any context.
/// Return:	The nbcon_prio to use for acquiring an nbcon console in this
/// 		context for printing.
///
/// The function is safe for reading per-CPU data in any context because
/// preemption is disabled if the current CPU is in the emergency or panic
/// state.
#[unsafe(no_mangle)]
extern "C" fn nbcon_get_default_prio() -> u32 {
if unsafe { panic_on_this_cpu() } {
return 3;
}
let cpu_emergency_nesting: *mut u32 = nbcon_get_cpu_emergency_nesting();
if (unsafe { *cpu_emergency_nesting }) != 0 {
return 2;
}
1
}

/// Track if it is allowed to perform unsafe hostile takeovers of console
/// ownership. When true, console drivers might perform unsafe actions while
/// printing. It is externally available via nbcon_allow_unsafe_takeover().
static mut panic_nbcon_allow_unsafe_takeover: bool = unsafe { core::mem::zeroed() };

/// nbcon_allow_unsafe_takeover - Check if unsafe console takeovers are allowed
///
/// Return:	True, when it is permitted to perform unsafe console printing
///
/// This is also used by console_is_usable() to determine if it is allowed to
/// call write_atomic() callbacks flagged as unsafe (CON_NBCON_ATOMIC_UNSAFE).
#[unsafe(no_mangle)]
extern "C" fn nbcon_allow_unsafe_takeover() -> bool {
(unsafe { panic_on_this_cpu() }) && unsafe { panic_nbcon_allow_unsafe_takeover }
}

/// nbcon_legacy_emit_next_record - Print one record for an nbcon console
/// 					in legacy contexts
/// @con:	The console to print on
/// @handover:	Will be set to true if a printk waiter has taken over the
/// 		console_lock, in which case the caller is no longer holding
/// 		both the console_lock and the SRCU read lock. Otherwise it
/// 		is set to false.
/// @cookie:	The cookie from the SRCU read lock.
/// @use_atomic: Set true when called in an atomic or unknown context.
/// 		It affects which nbcon callback will be used: write_atomic()
/// 		or write_thread().
///
/// 		When false, the write_thread() callback is used and would be
/// 		called in a preemptible context unless disabled by the
/// 		device_lock. The legacy handover is not allowed in this mode.
///
/// Context:	Any context except NMI.
/// Return:	True, when a record has been printed and there are still
/// 		pending records. The caller might want to continue flushing.
///
/// 		False, when there is no pending record, or when the console
/// 		context cannot be acquired, or the ownership has been lost.
/// 		The caller should give up. Either the job is done, cannot be
/// 		done, or will be handled by the owning context.
///
/// This function is meant to be called by console_flush_all() to print records
/// on nbcon consoles from legacy context (printing via console unlocking).
/// Essentially it is the nbcon version of console_emit_next_record().
#[unsafe(no_mangle)]
extern "C" fn nbcon_legacy_emit_next_record(con: *mut console, handover: *mut bool, cookie: i32, use_atomic: bool) -> bool {
let mut wctxt: nbcon_write_context = {
let mut __t0: nbcon_write_context = unsafe { core::mem::zeroed() };
__t0
};
let ctxt: *mut nbcon_context = unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(wctxt)).ctxt) };
let mut flags: u64 = 0;
let mut progress: bool = false;
unsafe {
(*ctxt).console = con;
}
unsafe {
(*ctxt).prio = nbcon_get_default_prio();
}
if use_atomic {
// In an atomic or unknown context, use the same procedure as
// in console_emit_next_record(). It allows to handover.
printk_safe_enter_irqsave!(flags);
unsafe { console_lock_spinning_enable() };
stop_critical_timings!();
}
progress = nbcon_emit_one(core::ptr::addr_of_mut!(wctxt), use_atomic);
if use_atomic {
start_critical_timings!();
unsafe {
*handover = console_lock_spinning_disable_and_check(cookie) != 0;
}
printk_safe_exit_irqrestore!(flags);
} else {
// Non-atomic does not perform legacy spinning handovers.
unsafe {
*handover = false;
}
}
progress
}

/// __nbcon_atomic_flush_pending_con - Flush specified nbcon console using its
/// 					write_atomic() callback
/// @con:			The nbcon console to flush
/// @stop_seq:			Flush up until this record
///
/// Return:	0 if @con was flushed up to @stop_seq Otherwise, error code on
/// 		failure.
///
/// Errors:
///
/// 	-EPERM:		Unable to acquire console ownership.
///
/// 	-EAGAIN:	Another context took over ownership while printing.
///
/// 	-ENOENT:	A record before @stop_seq is not available.
///
/// If flushing up to @stop_seq was not successful, it only makes sense for the
/// caller to try again when -EAGAIN was returned. When -EPERM is returned,
/// this context is not allowed to acquire the console. When -ENOENT is
/// returned, it cannot be expected that the unfinalized record will become
/// available.
fn __nbcon_atomic_flush_pending_con(con: *mut console, stop_seq: u64) -> i32 {
// Atomic flushing does not use console driver synchronization
// (i.e. it does not hold the port lock for uart consoles).
// Therefore IRQs must be disabled to avoid being interrupted
// and then calling into a driver that will deadlock trying
// to acquire console ownership.
let mut scope: core::mem::MaybeUninit<__SlateRecord_irqflags_h_266> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut __SlateRecord_irqflags_h_266 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_irqflags_h_266;
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut ctxt: *mut nbcon_context = unsafe { core::mem::zeroed() };
let mut wctxt: core::mem::MaybeUninit<nbcon_write_context> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut nbcon_write_context = core::ptr::addr_of_mut!(wctxt) as *mut nbcon_write_context;
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { class_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
core::ptr::write(__slate_slot_3, {
let mut __t0: nbcon_write_context = core::mem::zeroed();
__t0
});
ctxt = core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(*__slate_slot_3)).ctxt);
err = 0;
(*ctxt).console = con;
(*ctxt).spinwait_max_us = 2000;
(*ctxt).prio = nbcon_get_default_prio();
let __t1 = nbcon_allow_unsafe_takeover() as u32;
(*ctxt).__slate_bits_0.__set_allow_unsafe_takeover(__t1);
'__join_0: {
'__join_3: {
loop {
if nbcon_seq_read(con) < stop_seq {
core::ptr::write(__slate_slot_0, class_irqsave_constructor());
__slate_cleanup_armed_0.set(true);
if !class_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_0))).is_null() || !class_irqsave_is_conditional {
if true {
if !nbcon_context_try_acquire(ctxt, false) {
return 1i32.wrapping_neg();
} else {
// nbcon_emit_next_record() returns false when
// the console was handed over or taken over.
// In both cases the context is no longer valid.
if !nbcon_emit_next_record(core::ptr::addr_of_mut!(*__slate_slot_3), true) {
break;
} else {
nbcon_context_release(ctxt);
}
}
}
}
__slate_cleanup_armed_0.set(false);
class_irqsave_destructor(__slate_slot_0 as *mut _);
if ((*ctxt).__slate_bits_0.__get_backlog() as i32) == 0 {
break '__join_3;
}
} else {
break '__join_0;
}
}
return 11i32.wrapping_neg();
}
// Are there reserved but not yet finalized records?
if nbcon_seq_read(con) < stop_seq {
err = 2i32.wrapping_neg();
}
}
return err;
}
}

/// nbcon_atomic_flush_pending_con - Flush specified nbcon console using its
/// 					write_atomic() callback
/// @con:			The nbcon console to flush
/// @stop_seq:			Flush up until this record
///
/// This will stop flushing before @stop_seq if another context has ownership.
/// That context is then responsible for the flushing. Likewise, if new records
/// are added while this context was flushing and there is no other context
/// to handle the printing, this context must also flush those records.
fn nbcon_atomic_flush_pending_con(con: *mut console, mut stop_seq: u64) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut ft: core::mem::MaybeUninit<console_flush_type> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut console_flush_type = core::ptr::addr_of_mut!(ft) as *mut console_flush_type;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
loop {
*__slate_slot_1 = __nbcon_atomic_flush_pending_con(con, stop_seq);
// If there was a new owner (-EPERM, -EAGAIN), that context is
// responsible for completing.
//
// Do not wait for records not yet finalized (-ENOENT) to avoid a
// possible deadlock. They will either get flushed by the writer or
// eventually skipped on panic CPU.
if *__slate_slot_1 != 0 {
return;
} else {
// If flushing was successful but more records are available, this
// context must flush those remaining records if the printer thread
// is not available do it.
printk_get_console_flush_type(core::ptr::addr_of_mut!(*__slate_slot_2));
if !(*__slate_slot_2).nbcon_offload {
__v0 = prb_read_valid(prb, nbcon_seq_read(con), core::ptr::null_mut::<printk_record>());
} else {
__v0 = false;
}
if __v0 {
stop_seq = prb_next_reserve_seq(prb);
} else {
break;
}
}
}
}
}

// __nbcon_atomic_flush_pending - Flush all nbcon consoles using their
// 					write_atomic() callback
// @stop_seq:			Flush up until this record
fn __nbcon_atomic_flush_pending(stop_seq: u64) {
let mut con: *mut console = unsafe { core::mem::zeroed() };
let cookie: i32 = unsafe { console_srcu_read_lock() };
let mut __v2: *mut console;
let mut __v3: *mut hlist_node;
let mut __v4: *mut hlist_node;
loop {
if !(((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_400() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(console_list)).first) as *const *mut hlist_node) };
let __UNIQUE_ID_rcu_399: *mut hlist_node = __v4;
__v3 = __UNIQUE_ID_rcu_399;
let ____ptr: *mut hlist_node = __v3;
let __v6: *mut console = if !____ptr.is_null() {
hlist_entry!(____ptr)
} else {
core::ptr::null_mut::<console>()
};
__v2 = __v6;
con = __v2;
while !con.is_null() {
let flags: i16 = console_srcu_read_flags(con as *const console);
if (flags as i32) & 256 != 0 {
if console_is_usable(con, flags, true) {
if nbcon_seq_read(con) < stop_seq {
nbcon_atomic_flush_pending_con(con, stop_seq);
}
}
}
let mut __v10: *mut console;
let mut __v11: *mut hlist_node;
let mut __v12: *mut hlist_node;
loop {
if !(((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_402() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v12 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*con).node)).next) as *const *mut hlist_node) };
__v11 = __v12;
let __v0: *mut hlist_node = __v11;
let __v14: *mut console = if !__v0.is_null() {
hlist_entry!(__v0)
} else {
core::ptr::null_mut::<console>()
};
__v10 = __v14;
con = __v10;
}
unsafe { console_srcu_read_unlock(cookie) };
}

/// nbcon_atomic_flush_pending - Flush all nbcon consoles using their
/// 				write_atomic() callback
///
/// Flush the backlog up through the currently newest record. Any new
/// records added while flushing will not be flushed if there is another
/// context available to handle the flushing. This is to avoid one CPU
/// printing unbounded because other CPUs continue to add records.
#[unsafe(no_mangle)]
extern "C" fn nbcon_atomic_flush_pending() {
__nbcon_atomic_flush_pending(unsafe { prb_next_reserve_seq(prb) });
}

/// nbcon_atomic_flush_unsafe - Flush all nbcon consoles using their
/// 	write_atomic() callback and allowing unsafe hostile takeovers
///
/// Flush the backlog up through the currently newest record. Unsafe hostile
/// takeovers will be performed, if necessary.
#[unsafe(no_mangle)]
extern "C" fn nbcon_atomic_flush_unsafe() {
unsafe {
panic_nbcon_allow_unsafe_takeover = true;
}
__nbcon_atomic_flush_pending(unsafe { prb_next_reserve_seq(prb) });
unsafe {
panic_nbcon_allow_unsafe_takeover = false;
}
}

/// nbcon_cpu_emergency_enter - Enter an emergency section where printk()
/// 				messages for that CPU are flushed directly
///
/// Context:	Any context. Disables preemption.
///
/// When within an emergency section, printk() calls will attempt to flush any
/// pending messages in the ringbuffer.
#[unsafe(no_mangle)]
extern "C" fn nbcon_cpu_emergency_enter() {
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
atomic_inc(unsafe { core::ptr::addr_of_mut!(nbcon_cpu_emergency_cnt) });
let cpu_emergency_nesting: *mut u32 = nbcon_get_cpu_emergency_nesting();
let __v2: u32 = unsafe { *cpu_emergency_nesting };
unsafe {
*cpu_emergency_nesting = __v2.wrapping_add(1);
}
}

/// nbcon_cpu_emergency_exit - Exit an emergency section
///
/// Context:	Within an emergency section. Enables preemption.
#[unsafe(no_mangle)]
extern "C" fn nbcon_cpu_emergency_exit() {
let cpu_emergency_nesting: *mut u32 = nbcon_get_cpu_emergency_nesting();
let mut __v0: i64;
{
let __ret_warn_on: i32 = ((unsafe { *cpu_emergency_nesting }) == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("403: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 403b - ., 3; .popsection\n# {0}", const 403 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_nbcon_cpu_emergency_exit_1, sym __slate_asm_string_nbcon_cpu_emergency_exit_2, const 1739 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("404: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 404b - ., 4; .popsection\n# {0}", const 404 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
if __v0 == 0 {
let __v6: u32 = unsafe { *cpu_emergency_nesting };
unsafe {
*cpu_emergency_nesting = __v6.wrapping_sub(1);
}
}
// Wake up kthreads because there might be some pending messages
// added by other CPUs with normal priority since the last flush
// in the emergency context.
let mut __v8: i64;
let __ret_warn_on: i32 = (atomic_read((unsafe { core::ptr::addr_of_mut!(nbcon_cpu_emergency_cnt) }) as *const __SlateRecord_types_h_190) == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("405: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 405b - ., 3; .popsection\n# {0}", const 405 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_nbcon_cpu_emergency_exit_9, sym __slate_asm_string_nbcon_cpu_emergency_exit_10, const 1747 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("406: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 406b - ., 4; .popsection\n# {0}", const 406 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v8 = (__ret_warn_on != 0) as i64;
if __v8 == 0 {
if atomic_dec_return(unsafe { core::ptr::addr_of_mut!(nbcon_cpu_emergency_cnt) }) == 0 {
let mut ft: console_flush_type = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ft) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
printk_get_console_flush_type(core::ptr::addr_of_mut!(ft));
if ft.nbcon_offload {
nbcon_kthreads_wake();
}
}
}
loop {
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if unlikely__2!(preempt_count_dec_and_test!()) != 0 {
loop {
unsafe {
core::arch::asm!("call __SCT__preempt_schedule", options(att_syntax));
}
if true {
break;
}
}
}
if true {
break;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_407: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

/// nbcon_alloc - Allocate and init the nbcon console specific data
/// @con:	Console to initialize
///
/// Return:	True if the console was fully allocated and initialized.
/// 		Otherwise @con must not be registered.
///
/// When allocation and init was successful, the console must be properly
/// freed using nbcon_free() once it is no longer needed.
#[unsafe(no_mangle)]
extern "C" fn nbcon_alloc(con: *mut console) -> bool {
let mut __slate_compound_5: irq_work = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut state: nbcon_state = {
let mut __t0: nbcon_state = unsafe { core::mem::zeroed() };
__t0
};
// Synchronize the kthread start.
lockdep_assert_console_list_lock_held();
// Check for mandatory nbcon callbacks.
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*con).write_thread }) == None || (unsafe { (*con).device_lock }) == None || (unsafe { (*con).device_unlock }) == None) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("408: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 408b - ., 3; .popsection\n# {0}", const 408 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_nbcon_alloc_1, sym __slate_asm_string_nbcon_alloc_2, const 1780 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("409: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 409b - ., 4; .popsection\n# {0}", const 409 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return false;
}
rcuwait_init(unsafe { core::ptr::addr_of_mut!((*con).rcuwait) });
unsafe {
(*con).irq_work = *{
__slate_compound_5 = irq_work { node: {
let mut __t1: __call_single_node = core::mem::zeroed();
__t1.__slate_anon_1 = __SlateRecord_smp_types_h_60 { u_flags: 4 };
__t1
}, func: Some(nbcon_irq_work), irqwait: rcuwait { task: core::ptr::null_mut::<task_struct>() } };
core::ptr::addr_of_mut!(__slate_compound_5)
};
}
atomic_long_set(unsafe { core::ptr::addr_of_mut!((*con).nbcon_prev_seq) }, 1u64.wrapping_neg() as i64);
nbcon_state_set(con, unsafe { &(*core::ptr::addr_of_mut!(state)) });
// Initialize @nbcon_seq to the highest possible sequence number so
// that practically speaking it will have nothing to print until a
// desired initial sequence number has been set via nbcon_seq_force().
atomic_long_set(unsafe { core::ptr::addr_of_mut!((*con).nbcon_seq) }, ULSEQ_MAX!() as i64);
if ((unsafe { (*con).flags }) as i32) & 8 != 0 {
// Boot console printing is synchronized with legacy console
// printing, so boot consoles can share the same global printk
// buffers.
unsafe {
(*con).pbufs = core::ptr::addr_of_mut!(printk_shared_pbufs);
}
} else {
let mut __v6: *mut printk_buffers;
let __obj_size: u64 = size_mul(core::mem::size_of::<printk_buffers>() as u64, 1u64);
let mut __v7: *mut ();
let mut __v8: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
} });
}
__v8 = _res;
__v7 = __v8;
__v6 = __v7 as *mut printk_buffers;
unsafe {
(*con).pbufs = __v6;
}
if unsafe { (*con).pbufs }.is_null() {
let mut __v12: i32;
while false {
}
__v12 = unsafe { _printk(c"\x013%s%sconsole [%s%d] failed to allocate printing buffer\n".as_ptr() as *const u8, if ((*con).flags as i32) & 256 != 0 { c"".as_ptr() as *mut u8 } else { c"legacy ".as_ptr() as *mut u8 }, if ((*con).flags as i32) & 8 != 0 { c"boot".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, (*con).name.as_mut_ptr() as *mut u8, (*con).index as i32) };
return false;
}
if (unsafe { printk_kthreads_ready }) && !unsafe { have_boot_console } {
if !nbcon_kthread_create(con) {
unsafe { kfree((*con).pbufs as *const ()) };
unsafe {
(*con).pbufs = core::ptr::null_mut::<printk_buffers>();
}
return false;
}
// Might be the first kthread.
unsafe {
printk_kthreads_running = true;
}
}
}
true
}

/// nbcon_free - Free and cleanup the nbcon console specific data
/// @con:	Console to free/cleanup nbcon data
///
/// Important: @have_nbcon_console must be updated before calling
/// 	this function. In particular, it can be set only when there
/// 	is still another nbcon console registered.
#[unsafe(no_mangle)]
extern "C" fn nbcon_free(con: *mut console) {
let mut state: nbcon_state = {
let mut __t0: nbcon_state = unsafe { core::mem::zeroed() };
__t0
};
// Synchronize the kthread stop.
lockdep_assert_console_list_lock_held();
unsafe { irq_work_sync(core::ptr::addr_of_mut!((*con).irq_work)) };
if unsafe { printk_kthreads_running } {
nbcon_kthread_stop(con);
// Might be the last nbcon console.
//
// Do not rely on printk_kthreads_check_locked(). It is not
// called in some code paths, see nbcon_free() callers.
if !unsafe { have_nbcon_console } {
unsafe {
printk_kthreads_running = false;
}
}
}
nbcon_state_set(con, unsafe { &(*core::ptr::addr_of_mut!(state)) });
// Boot consoles share global printk buffers.
if ((unsafe { (*con).flags }) as i32) & 8 == 0 {
unsafe { kfree((*con).pbufs as *const ()) };
}
unsafe {
(*con).pbufs = core::ptr::null_mut::<printk_buffers>();
}
}

/// nbcon_device_try_acquire - Try to acquire nbcon console and enter unsafe
/// 				section
/// @con:	The nbcon console to acquire
///
/// Context:	Under the locking mechanism implemented in
/// 		@con->device_lock() including disabling migration.
/// Return:	True if the console was acquired. False otherwise.
///
/// Console drivers will usually use their own internal synchronization
/// mechanism to synchronize between console printing and non-printing
/// activities (such as setting baud rates). However, nbcon console drivers
/// supporting atomic consoles may also want to mark unsafe sections when
/// performing non-printing activities in order to synchronize against their
/// atomic_write() callback.
///
/// This function acquires the nbcon console using priority NBCON_PRIO_NORMAL
/// and marks it unsafe for handover/takeover.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.nbcon.nbcon_device_try_acquire")]
extern "C" fn nbcon_device_try_acquire(con: *mut console) -> bool {
let ctxt: *mut nbcon_context = unsafe { core::ptr::addr_of_mut!((*con).nbcon_device_ctxt) };
cant_migrate!();
unsafe { memset(ctxt as *mut (), 0i32, core::mem::size_of::<nbcon_context>() as u64) };
unsafe {
(*ctxt).console = con;
}
unsafe {
(*ctxt).prio = 1;
}
if !nbcon_context_try_acquire(ctxt, false) {
return false;
}
if !nbcon_context_enter_unsafe!(ctxt) {
return false;
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nbcon_device_try_acquire_410: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut console) -> bool>, *mut ()>(Some(nbcon_device_try_acquire)) };

/// nbcon_device_release - Exit unsafe section and release the nbcon console
/// @con:	The nbcon console acquired in nbcon_device_try_acquire()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.nbcon.nbcon_device_release")]
extern "C" fn nbcon_device_release(con: *mut console) {
let ctxt: *mut nbcon_context = unsafe { core::ptr::addr_of_mut!((*con).nbcon_device_ctxt) };
let mut ft: console_flush_type = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ft) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let mut cookie: i32 = 0;
if !nbcon_context_exit_unsafe!(ctxt) {
return;
}
nbcon_context_release(ctxt);
// This context must flush any new records added while the console
// was locked if the printer thread is not available to do it. The
// console_srcu_read_lock must be taken to ensure the console is
// usable throughout flushing.
cookie = unsafe { console_srcu_read_lock() };
printk_get_console_flush_type(core::ptr::addr_of_mut!(ft));
let __v0: bool = console_is_usable(con, console_srcu_read_flags(con as *const console), true) && !ft.nbcon_offload && unsafe { prb_read_valid(prb, nbcon_seq_read(con), core::ptr::null_mut::<printk_record>()) };
if __v0 {
// If nbcon_atomic flushing is not available, fallback to
// using the legacy loop.
if ft.nbcon_atomic {
__nbcon_atomic_flush_pending_con(con, unsafe { prb_next_reserve_seq(prb) });
} else {
if ft.legacy_direct {
if (unsafe { console_trylock() }) != 0 {
unsafe { console_unlock() };
}
} else {
if ft.legacy_offload {
unsafe { defer_console_output() };
}
}
}
}
unsafe { console_srcu_read_unlock(cookie) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nbcon_device_release_411: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut console)>, *mut ()>(Some(nbcon_device_release)) };

/// nbcon_kdb_try_acquire - Try to acquire nbcon console and enter unsafe
/// 			   section
/// @con:	The nbcon console to acquire
/// @wctxt:	The nbcon write context to be used on success
///
/// Context:	Under console_srcu_read_lock() for emitting a single kdb message
/// 		using the given con->write_atomic() callback. Can be called
/// 		only when the console is usable at the moment.
///
/// Return:	True if the console was acquired. False otherwise.
///
/// kdb emits messages on consoles registered for printk() without
/// storing them into the ring buffer. It has to acquire the console
/// ownership so that it could call con->write_atomic() callback a safe way.
///
/// This function acquires the nbcon console using priority NBCON_PRIO_EMERGENCY
/// and marks it unsafe for handover/takeover.
#[unsafe(no_mangle)]
extern "C" fn nbcon_kdb_try_acquire(con: *mut console, wctxt: *mut nbcon_write_context) -> bool {
let ctxt: *mut nbcon_context = unsafe { core::ptr::addr_of_mut!((*wctxt).ctxt) };
unsafe { memset(ctxt as *mut (), 0i32, core::mem::size_of::<nbcon_context>() as u64) };
unsafe {
(*ctxt).console = con;
}
unsafe {
(*ctxt).prio = 2;
}
if !nbcon_context_try_acquire(ctxt, false) {
return false;
}
if !nbcon_context_enter_unsafe!(ctxt) {
return false;
}
true
}

/// nbcon_kdb_release - Exit unsafe section and release the nbcon console
///
/// @wctxt:	The nbcon write context initialized by a successful
/// 		nbcon_kdb_try_acquire()
#[unsafe(no_mangle)]
extern "C" fn nbcon_kdb_release(wctxt: *mut nbcon_write_context) {
let ctxt: *mut nbcon_context = unsafe { core::ptr::addr_of_mut!((*wctxt).ctxt) };
if !nbcon_context_exit_unsafe!(ctxt) {
return;
}
nbcon_context_release(ctxt);
// Flush any new printk() messages added when the console was blocked.
// Only the console used by the given write context was	blocked.
// The console was locked only when the write_atomic() callback
// was usable.
__nbcon_atomic_flush_pending_con(unsafe { (*ctxt).console }, unsafe { prb_next_reserve_seq(prb) });
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_console_srcu_read_flags_1: [u8; 1] = *b"\0";

static __slate_asm_string_console_srcu_read_flags_2: [u8; 53] = *b"/home/takashi/c-corpus/linux/include/linux/console.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_nbcon_alloc_1: [u8; 1] = *b"\0";

static __slate_asm_string_nbcon_alloc_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/printk/nbcon.c\0";

static __slate_asm_string_nbcon_context_can_proceed_10: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/printk/nbcon.c\0";

static __slate_asm_string_nbcon_context_can_proceed_9: [u8; 1] = *b"\0";

static __slate_asm_string_nbcon_context_try_acquire_direct_10: [u8; 1] = *b"\0";

static __slate_asm_string_nbcon_context_try_acquire_direct_11: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/printk/nbcon.c\0";

static __slate_asm_string_nbcon_context_try_acquire_handover_10: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/printk/nbcon.c\0";

static __slate_asm_string_nbcon_context_try_acquire_handover_14: [u8; 1] = *b"\0";

static __slate_asm_string_nbcon_context_try_acquire_handover_15: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/printk/nbcon.c\0";

static __slate_asm_string_nbcon_context_try_acquire_handover_9: [u8; 1] = *b"\0";

static __slate_asm_string_nbcon_context_try_acquire_hostile_10: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/printk/nbcon.c\0";

static __slate_asm_string_nbcon_context_try_acquire_hostile_14: [u8; 1] = *b"\0";

static __slate_asm_string_nbcon_context_try_acquire_hostile_15: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/printk/nbcon.c\0";

static __slate_asm_string_nbcon_context_try_acquire_hostile_19: [u8; 1] = *b"\0";

static __slate_asm_string_nbcon_context_try_acquire_hostile_20: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/printk/nbcon.c\0";

static __slate_asm_string_nbcon_context_try_acquire_hostile_9: [u8; 1] = *b"\0";

static __slate_asm_string_nbcon_context_try_acquire_requested_10: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/printk/nbcon.c\0";

static __slate_asm_string_nbcon_context_try_acquire_requested_14: [u8; 1] = *b"\0";

static __slate_asm_string_nbcon_context_try_acquire_requested_15: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/printk/nbcon.c\0";

static __slate_asm_string_nbcon_context_try_acquire_requested_9: [u8; 1] = *b"\0";

static __slate_asm_string_nbcon_cpu_emergency_exit_1: [u8; 1] = *b"\0";

static __slate_asm_string_nbcon_cpu_emergency_exit_10: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/printk/nbcon.c\0";

static __slate_asm_string_nbcon_cpu_emergency_exit_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/printk/nbcon.c\0";

static __slate_asm_string_nbcon_cpu_emergency_exit_9: [u8; 1] = *b"\0";

static __slate_asm_string_nbcon_emit_next_record_16: [u8; 1] = *b"\0";

static __slate_asm_string_nbcon_emit_next_record_17: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/printk/nbcon.c\0";

static __slate_asm_string_nbcon_kthreads_wake_3: [u8; 1] = *b"\0";

static __slate_asm_string_nbcon_kthreads_wake_4: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/printk/nbcon.c\0";

static __slate_asm_string_printk_get_console_flush_type_2: [u8; 1] = *b"\0";

static __slate_asm_string_printk_get_console_flush_type_3: [u8; 54] = *b"/home/takashi/c-corpus/linux/kernel/printk/internal.h\0";
