
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
//! SPDX-License-Identifier: GPL-2.0-or-later
//!
//! Timers abstract layer
//! Copyright (c) by Jaroslav Kysela <perex@perex.cz>
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
//! These are the special 64-bit data types that are 8-byte aligned
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! SCHED_ISO: reserved but not implemented yet
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! This counts to 15. Any more, it will return 16th argument.
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
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
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
//! These are for everybody (although not all archs will actually
//! discard it in modules)
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
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! warning conditions
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! Defines x86 CPU feature bits
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! array_index_mask_nospec() - generate a mask that is ~0UL when the
//! 	bounds check succeeds and 0 otherwise
//! @idx: array element index
//! @sz: number of elements in array
//!
//! Returns:
//!     0 - (@idx < @sz)
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
//! writeable
//! 4th level page in 5-level paging case
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
//! These should never be seen by user programs.  To return one of ERESTART*
//! codes, signal_pending() MUST be set.  Note that ptrace can observe these
//! at syscall exit tracing, but they will never be left for the debugged user
//! process to see.
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
//! Helper macro for scoped_guard().
//!
//! Note that the "!__is_cond_ptr(_name)" part of the condition ensures that
//! compiler would be sure that for the unconditional locks the body of the
//! loop (caller-provided code glued to the else clause) could not be skipped.
//! It is needed because the other part - "__guard_ptr(_name)(&scope)" - is too
//! hard to deduce (even if could be proven true for unconditional locks).
//! 64bit FS base
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
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
//! Used to create numbers.
//!
//! NOTE: _IOW means userland is writing and kernel is reading. _IOR
//! means userland is reading and kernel is writing.
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
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
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
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
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
//! __MAX_NR_ZONES
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
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
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
//! Keep unconverted code working
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! Generate SD flag indexes
//! Resource type
//! Desired design of maximum size and alignment (see RFC2553)
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
//! array_index_nospec - sanitize an array index after a bounds check
//!
//! For a code sequence like:
//!
//!     if (index < size) {
//!         index = array_index_nospec(index, size);
//!         val = array[index];
//!     }
//!
//! ...if the CPU speculates past the bounds check then
//! array_index_nospec() will clamp the index within the range of [0,
//! size).
//! VMA locks do not behave like most ordinary locks found in the kernel, so we
//! cannot quite have full lockdep tracking in the way we would ideally prefer.
//!
//! Read locks act as shared locks which exclude an exclusive lock being
//! taken. We therefore mark these accordingly on read lock acquire/release.
//!
//! Write locks are acquired exclusively per-VMA, but released in a shared
//! fashion, that is upon vma_end_write_all(), we update the mmap's seqcount such
//! that write lock is released.
//!
//! We therefore cannot track write locks per-VMA, nor do we try. Mitigating this
//! is the fact that, of course, we do lockdep-track the mmap lock rwsem which
//! must be held when taking a VMA write lock.
//!
//! We do, however, want to indicate that during either acquisition of a VMA
//! write lock or detachment of a VMA that we require the lock held be exclusive,
//! so we utilise lockdep to do so.
//! MXCSR will be initialized
//! mprotect needs to preserve PAT and encryption bits when updating
//! vm_page_prot
//! 6 because 6 should be plenty and struct tlb_state will fit in two cache
//! lines.
//! This is the smallest unsigned integer type that can fit a value
//! (up to 'long long')
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
//! return the first ptr or next element in an RCU protected list
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! per-IO O_SYNC
//! File supports atomic writes
//! The structure layout of user_xstateregs, used for exporting the
//! extended register state through ptrace and core-dump (NT_X86_XSTATE note)
//! interfaces will be same as the memory layout of xsave used by the processor
//! (except for the bytes 464..511, which can be used by the software) and hence
//! the size of this structure varies depending on the features supported by the
//! processor and OS. The size of the structure that users need to use can be
//! obtained by doing:
//!     cpuid_count(0xd, 0, &eax, &ptrace_xstateregs_struct_size, &ecx, &edx);
//! i.e., cpuid.(eax=0xd,ecx=0).ebx will be the size that user (debuggers, etc.)
//! need to use.
//!
//! For now, only the first 8 bytes of the software usable bytes[464..471] will
//! be used and will be set to OS enabled xstate mask (which is same as the
//! 64bit mask returned by the xgetbv's xCR0).  Users (analyzing core dump
//! remotely, etc.) can use this mask as well as the mask saved in the
//! xstate_hdr bytes and interpret what states the processor/OS supports
//! and what states are in modified/initialized conditions for the
//! particular process/thread.
//!
//! Also when the user modifies certain state FP/SSE/etc through the
//! ptrace interface, they must ensure that the header.xfeatures
//! bytes[512..519] of the memory layout are updated correspondingly.
//! i.e., for example when FP state is modified to a non-init state,
//! header.xfeatures's bit 0 must be set to '1', when SSE is modified to
//! non-init state, header.xfeatures's bit 1 must to be set to '1', etc.
//! Generic info of form tag = "info"
//! standard configuration
//! auto trigger is supported
//! optional flags for follower
//! These are specified by iBCS2
//! Epoll event masks
//! SNDRV_MINOR_GLOBAL + 1 * 32
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
// &a[0] degrades to a pointer: a different type from an array
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
// Concatenate two parameters, but allow them to be expanded beforehand.
// objtool annotation to ignore the alternatives and only consider the original
// instruction(s).
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
// reg := -EIO
// reg := -EIO
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
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
// Dummy printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
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
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "pti" Kernel Page Table Isolation enabled
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// Intel-defined CPU features, CPUID level 0x00000007:0 (EBX), word 9
// "fsgsbase" RDFSBASE, WRFSBASE, RDGSBASE, WRGSBASE instructions
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
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "l1tf" CPU is affected by L1 Terminal Fault
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
// Prevent speculative execution past this barrier.
// Atomic operations are already serializing on x86
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
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
// bits_per - calculate the number of bits required for the argument
// @n: parameter
//
// This is constant-capable and can be used for compile time
// initializations, e.g bitfields.
//
// The first few values calculated by this routine:
// bf(0) = 1
// bf(1) = 1
// bf(2) = 2
// bf(3) = 2
// bf(4) = 3
// ... and so on.
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
// No such file or directory
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// No such device
// Invalid argument
// Not a typewriter
// Text file busy
// Invalid system call number
// Value too large for defined data type
// File descriptor in bad state
// Operation not supported on transport endpoint
// No ioctl command
// restart by calling sys_restart_syscall
// Operation is not supported
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
// strscpy - Copy a C-string into a sized buffer
// @dst: Where to copy the string to
// @src: Where to copy the string from
// @...: Size of destination buffer (optional)
//
// Copy the source string @src, or as much of it as fits, into the
// destination @dst buffer. The behavior is undefined if the string
// buffers overlap. The destination @dst buffer is always NUL terminated,
// unless it's zero-sized.
//
// The size argument @... is only required when @dst is not an array, or
// when the copy needs to be smaller than sizeof(@dst).
//
// Preferred to strncpy() since it always returns a valid string, and
// doesn't unnecessarily force the tail of the destination buffer to be
// zero padded. If padding is desired please use strscpy_pad().
//
// Returns the number of characters copied in @dst (not including the
// trailing %NUL) or -E2BIG if @size is 0 or the copy from @src was
// truncated.
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
// used to decode ioctl numbers..
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
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
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
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Change timeval to jiffies, trying to avoid the
// most obvious overflows..
//
// And some not so obvious.
//
// Note that we don't want to return LONG_MAX, because
// for various timeout reasons we often end up having
// to wait "jiffies+1" in order to guarantee that we wait
// at _least_ "jiffies" - so "jiffies+1" had better still
// be positive.
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
// Same as rb_first(), but O(1)
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
// The bottom two bits of the slot determine how the remaining bits in the
// slot are interpreted:
//
// 00 - data pointer
// 10 - internal entry
// x1 - value entry
//
// The internal entry may be a pointer to the next level in the tree, a
// sibling entry, or an indicator that the entry in this slot has been moved
// to another location in the tree and the lookup should be restarted.  While
// NULL fits the 'data pointer' pattern, it means that there is no entry in
// the tree for this index (no matter what level of the tree it is found at).
// This means that storing a NULL entry in the tree is the same as deleting
// the entry from the tree.
// Set the IDR flag and the IDR_FREE tag
// Generate SD flag bits
// Resource extended types
// No address assigned yet
// Implementation specific max size
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
// This is used for both get_user() and __get_user() to expand to
// the proper special function call that has odd calling conventions
// due to returning both a value and an error, and that depends on
// the size of the pointer passed in.
//
// Careful: we have to cast the result to the type of the pointer
// for sign reasons.
//
// The use of _ASM_DX as the register specifier is a bit of a
// simplification, as gcc only cares about it as the starting point
// and not size: for a 64-bit value it will use %ecx:%edx on 32 bits
// (%ecx being the next register in gcc's x86 register sequence), and
// %rdx on 64 bits.
//
// Clang/LLVM cares about the size of the register, but still wants
// the base register for something that ends up being a pair.
// get_user - Get a simple variable from user space.
// @x:   Variable to store result.
// @ptr: Source address, in user space.
//
// Context: User context only. This function may sleep if pagefaults are
//          enabled.
//
// This macro copies a single simple variable from user space to kernel
// space.  It supports simple types like char and int, but not larger
// data types like structures or arrays.
//
// @ptr must have pointer-to-simple-variable type, and the result of
// dereferencing @ptr must be assignable to @x without a cast.
//
// Return: zero on success, or -EFAULT on error.
// On error, the variable @x is set to zero.
// ptr must be evaluated and assigned to the temporary __ptr_pu before
// the assignment of x to __val_pu, to avoid any function calls
// involved in the ptr expression (possibly implicitly generated due
// to KASAN) from clobbering %ax.
// eval x once
// eval ptr once
// put_user - Write a simple value into user space.
// @x:   Value to copy to user space.
// @ptr: Destination address, in user space.
//
// Context: User context only. This function may sleep if pagefaults are
//          enabled.
//
// This macro copies a single simple value from kernel space to user
// space.  It supports simple types like char and int, but not larger
// data types like structures or arrays.
//
// @ptr must have pointer-to-simple-variable type, and @x must be assignable
// to the result of dereferencing @ptr.
//
// Return: zero on success, or -EFAULT on error.
// direct disk access hint
// Reserved kernel ranges [-100], [-10000, -40000].
// Special value for dirfd used to
// indicate openat should use the
// current working directory.
// Reserved for per-syscall flags	0xff.
// Do not follow symbolic
// links.
// Allow empty relative
// pathname to operate on dirfd
// directly.
// The 10 bits between the priority class and the priority level are used to
// optionally define I/O hints for any combination of I/O priority class and
// level. Depending on the kernel configuration, I/O scheduler being used and
// the target I/O device being used, hints can influence how I/Os are processed
// without affecting the I/O scheduling ordering defined by the I/O priority
// class and level.
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
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
unlikely!(((($x) as *mut ()) as u64) >= (4095i32.wrapping_neg() as u64))
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! SNDRV_PROTOCOL_VERSION {
($major:expr, $minor:expr, $subminor:expr) => {
($major) << 16i32 | ($minor) << 8i32 | $subminor
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

macro_rules! _AC__2 {
() => {
16045481047390945280u64
};
}

macro_rules! _IOC_TYPECHECK {
($t:ty) => {
if (core::mem::size_of::<$t>() as u64) == (core::mem::size_of::<[$t; 1]>() as u64) && (core::mem::size_of::<$t>() as u64) < 16384u64 { core::mem::size_of::<$t>() as u64 } else { (unsafe { __invalid_size_argument_for_IOC }) as u64 }
};
}

macro_rules! _IOC {
($dir:expr, $type:expr, $nr:expr, $size:expr) => {
((($dir) << 30i32 | ((($type) << 8i32) as u32) | ((($nr) << 0i32) as u32)) as u64) | ($size) << 16i32
};
}

macro_rules! _IOW {
($type:expr, $nr:expr, $argtype:ty) => {
_IOC!(1u32, $type, $nr, _IOC_TYPECHECK!($argtype))
};
}

macro_rules! _UL {
() => {
_AC!()
};
}

macro_rules! __WRITE_ONCE__2 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut i32, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut list_head, $val) };
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

/// __acquire() - function to acquire context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that acquires the given context lock instance @x exclusively.
macro_rules! __acquire {
() => {
__acquire_ctx_lock!();
};
}

macro_rules! __acquire_ctx_lock {
() => {
while false {
}
};
}

macro_rules! __chk_user_ptr {
() => {
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

macro_rules! __must_be_array__51 {
() => {
core::mem::size_of::<__SlateRecord_timer_c_1289>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__12 {
() => {
core::mem::size_of::<__SlateRecord_timer_c_1725>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__17 {
() => {
core::mem::size_of::<__SlateRecord_timer_compat_c_65_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__23 {
() => {
core::mem::size_of::<__SlateRecord_timer_c_1012>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__26 {
() => {
core::mem::size_of::<__SlateRecord_timer_c_1864_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__38 {
() => {
core::mem::size_of::<__SlateRecord_timer_c_1289_3>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__51 {
() => {
core::mem::size_of::<__SlateRecord_timer_c_1724_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__55 {
() => {
core::mem::size_of::<__SlateRecord_timer_c_1725_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__58 {
() => {
core::mem::size_of::<__SlateRecord_timer_compat_c_66>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__61 {
() => {
core::mem::size_of::<__SlateRecord_timer_compat_c_65>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__65 {
() => {
core::mem::size_of::<__SlateRecord_timer_c_1012_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__69 {
() => {
core::mem::size_of::<__SlateRecord_timer_c_1865>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__71 {
() => {
core::mem::size_of::<__SlateRecord_timer_c_1724>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__72 {
() => {
core::mem::size_of::<__SlateRecord_timer_c_1865_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__79 {
() => {
core::mem::size_of::<__SlateRecord_timer_compat_c_66_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__95 {
() => {
core::mem::size_of::<__SlateRecord_timer_c_1289_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__100 {
() => {
core::mem::size_of::<__SlateRecord_timer_c_1864>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

/// __release() - function to release context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x.
macro_rules! __release {
() => {
__release_ctx_lock!();
};
}

macro_rules! __release_ctx_lock {
() => {
while false {
}
};
}

macro_rules! __timer_init {
($_timer:expr, $_fn:expr, $_flags:expr) => {
unsafe { timer_init_key($_timer, $_fn, ($_flags) as u32, core::ptr::null::<u8>(), core::ptr::null_mut::<lock_class_key>()) };
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
let __vpp_verify: *const () = core::ptr::null_mut::<*mut task_struct>() as *const ();
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

macro_rules! arch_atomic_fetch_sub {
($i:expr, $v:expr) => {
arch_atomic_fetch_add($i.wrapping_neg(), $v)
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

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__15 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__37!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__17 {
($ptr:expr) => {
{
let mut __v: *mut snd_timer_instance;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__43!() as isize)) }) as *mut ()) as *mut snd_timer_instance;
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
macro_rules! container_of__19 {
($ptr:expr) => {
{
let mut __v: *mut snd_timer_system_private;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__45!() as isize)) }) as *mut ()) as *mut snd_timer_system_private;
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
macro_rules! container_of__22 {
($ptr:expr) => {
{
let mut __v: *mut snd_timer;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__48!() as isize)) }) as *mut ()) as *mut snd_timer;
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
macro_rules! container_of__24 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__33!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__30 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__25!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__32 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__49!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__33 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__22!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__39 {
($ptr:expr) => {
{
let mut __v: *mut snd_timer_instance;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__30!() as isize)) }) as *mut ()) as *mut snd_timer_instance;
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
macro_rules! container_of__44 {
($ptr:expr) => {
{
let mut __v: *mut snd_timer_instance;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__46!() as isize)) }) as *mut ()) as *mut snd_timer_instance;
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

/// instrument_get_user() - add instrumentation to get_user()-like macros
/// @to: destination variable, may not be address-taken
///
/// get_user() and friends are fragile, so it may depend on the implementation
/// whether the instrumentation happens before or after the data is copied from
/// the userspace.
macro_rules! instrument_get_user {
($to:expr) => {
let mut __v: u32;
let mut __tmp: u64 = ($to) as u64;
kmsan_unpoison_memory(core::ptr::addr_of_mut!(__tmp) as *const (), core::mem::size_of::<u64>() as u64);
let __v2: u32 = __tmp as u32;
$to = __v2;
__v = __v2;
};
}

/// instrument_put_user() - add instrumentation to put_user()-like macros
/// @from: source address
/// @ptr: userspace pointer to copy to
/// @size: number of bytes to copy
///
/// put_user() and friends are fragile, so it may depend on the implementation
/// whether the instrumentation happens before or after the data is copied from
/// the userspace.
macro_rules! instrument_put_user {
($from:expr, $ptr:expr) => {
kmsan_copy_to_user(($ptr) as *mut (), core::ptr::addr_of_mut!($from) as *const (), core::mem::size_of::<i32>() as u64, 0u64);
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

macro_rules! kcsan_rmb {
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

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__10 {
($ptr:expr) => {
container_of__22!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__7 {
($ptr:expr, $type:ty) => {
container_of__15!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__12 {
($ptr:expr) => {
container_of__17!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__13 {
($ptr:expr) => {
container_of__44!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__16 {
($ptr:expr) => {
container_of__39!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__27 {
($ptr:expr, $type:ty) => {
container_of__33!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__28 {
($ptr:expr, $type:ty) => {
container_of__32!($ptr, $type)
};
}

/// list_entry_is_head - test if the entry points to the head of the list
/// @pos:	the type * to cursor
/// @head:	the head for your list.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry_is_head {
($pos:expr, $head:expr, $member:ident) => {
list_is_head((unsafe { core::ptr::addr_of_mut!((*$pos).$member) }) as *const list_head, ($head) as *const list_head)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__20 {
($ptr:expr) => {
list_entry__16!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__4 {
($ptr:expr) => {
list_entry__12!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__25 {
($ptr:expr, $type:ty) => {
list_entry__27!(unsafe { (*$ptr).next }, $type)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__5 {
($ptr:expr) => {
list_entry__13!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__10 {
($ptr:expr) => {
list_entry__10!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__2 {
($pos:expr, $member:ident) => {
list_entry__12!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__5 {
($pos:expr, $member:ident) => {
list_entry__13!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__18 {
($pos:expr, $member:ident) => {
list_entry__16!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__7 {
($pos:expr, $member:ident) => {
list_entry__10!(unsafe { (*$pos).$member.next })
};
}

macro_rules! offsetof__22 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, ack_list) as u64
};
}

macro_rules! offsetof__25 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, kref) as u64
};
}

macro_rules! offsetof__30 {
() => {
core::mem::offset_of!(snd_timer_instance, master_list) as u64
};
}

macro_rules! offsetof__33 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, task_work) as u64
};
}

macro_rules! offsetof__37 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, device_list) as u64
};
}

macro_rules! offsetof__43 {
() => {
core::mem::offset_of!(snd_timer_instance, open_list) as u64
};
}

macro_rules! offsetof__45 {
() => {
core::mem::offset_of!(snd_timer_system_private, tlist) as u64
};
}

macro_rules! offsetof__46 {
() => {
core::mem::offset_of!(snd_timer_instance, active_list) as u64
};
}

macro_rules! offsetof__48 {
() => {
core::mem::offset_of!(snd_timer, device_list) as u64
};
}

macro_rules! offsetof__49 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, open_list) as u64
};
}

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
};
}

macro_rules! raw_spin_lock_irq {
($lock:expr) => {
unsafe { _raw_spin_lock_irq($lock) };
};
}

macro_rules! raw_spin_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe {
$flags = _raw_spin_lock_irqsave($lock);
}
if true {
break;
}
}
};
}

macro_rules! raw_spin_unlock {
($lock:expr) => {
unsafe { _raw_spin_unlock($lock) };
};
}

macro_rules! raw_spin_unlock_irq {
($lock:expr) => {
unsafe { _raw_spin_unlock_irq($lock) };
};
}

macro_rules! raw_spin_unlock_irqrestore {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe { _raw_spin_unlock_irqrestore($lock, $flags) };
if true {
break;
}
}
};
}

macro_rules! read_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe {
$flags = _raw_read_lock_irqsave($lock);
}
if true {
break;
}
}
};
}

macro_rules! read_unlock_irqrestore {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe { _raw_read_unlock_irqrestore($lock, $flags) };
if true {
break;
}
}
};
}

macro_rules! snd_BUG_ON {
($condition:expr) => {
{
let mut __v: i64;
let __v2: i32 = !(!$condition) as i32;
__v = (__v2 != 0) as i64;
__v
}
};
}

macro_rules! spin_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
raw_spin_lock_irqsave!(spinlock_check($lock), $flags);
__release!();
__acquire!();
if true {
break;
}
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

macro_rules! task_thread_info {
($task:expr) => {
unsafe { core::ptr::addr_of_mut!((*$task).thread_info) }
};
}

macro_rules! test_bit {
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

/// time_after - returns true if the time a is after time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Do this with "<0" and ">=0" to only test the sign of the result. A
/// good compiler would generate better code (and a really good compiler
/// wouldn't care). Gcc is currently neither.
///
/// Return: %true is time a is after time b, otherwise %false.
macro_rules! time_after {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck__2!(u64) != 0 && typecheck__2!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
}
};
}

/// time_before - returns true if the time a is before time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Return: %true is time a is before time b, otherwise %false.
macro_rules! time_before__2 {
($a:expr, $b:expr) => {
time_after!($b, $a)
};
}

macro_rules! timer_container_of {
($callback_timer:expr) => {
container_of__19!($callback_timer)
};
}

/// timer_setup - prepare a timer for first use
/// @timer: the timer in question
/// @callback: the function to call when timer expires
/// @flags: any TIMER_* flags
///
/// Regular timer initialization should use either DEFINE_TIMER() above,
/// or timer_setup(). For timers on the stack, timer_setup_on_stack() must
/// be used and must be balanced with a call to timer_destroy_on_stack().
macro_rules! timer_setup {
($timer:expr, $callback:expr, $flags:expr) => {
__timer_init!($timer, $callback, $flags);
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

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck__2 {
($type:ty) => {
{
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == core::ptr::addr_of_mut!(__v3);
__v = 1;
__v
}
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

macro_rules! wake_up {
($x:expr) => {
unsafe { __wake_up($x, 3u32, 1i32, core::ptr::null_mut::<()>()) };
};
}

macro_rules! write_lock_irq {
($lock:expr) => {
unsafe { _raw_write_lock_irq($lock) };
};
}

macro_rules! write_unlock_irq {
($lock:expr) => {
unsafe { _raw_write_unlock_irq($lock) };
};
}

unsafe extern "C" {
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __invalid_size_argument_for_IOC: u32;
    static mut cpuinfo_op: seq_operations;
    static mut __tracepoint_sched_set_state_tp: tracepoint;
    static mut jiffies: u64;
    static mut system_highpri_wq: *mut workqueue_struct;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut param_ops_int: kernel_param_ops;
    static mut snd_ecards_limit: i32;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn _printk(arg0: *const u8, ...) -> i32;
    fn ____wrong_branch_error() -> bool;
    fn __xchg_wrong_size();
    fn __xadd_wrong_size();
    fn memdup_user(arg0: *const (), arg1: u64) -> *mut ();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn sized_strscpy(arg0: *mut u8, arg1: *const u8, arg2: u64) -> i64;
    fn kstrdup(arg0: *const u8, arg1: u32) -> *mut u8;
    fn sprintf(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn __compiletime_assert_16();
    fn __compiletime_assert_19();
    fn __compiletime_assert_20();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn _raw_write_lock_irq(arg0: *mut rwlock);
    fn _raw_read_lock_irqsave(arg0: *mut rwlock) -> u64;
    fn _raw_write_unlock_irq(arg0: *mut rwlock);
    fn _raw_read_unlock_irqrestore(arg0: *mut rwlock, arg1: u64);
    fn schedule();
    fn __trace_set_current_state(arg0: i32);
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn default_wake_function(arg0: *mut wait_queue_entry, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn __init_waitqueue_head(arg0: *mut wait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn add_wait_queue(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry);
    fn remove_wait_queue(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry);
    fn __wake_up(arg0: *mut wait_queue_head, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn refcount_warn_saturate(arg0: *mut refcount_struct, arg1: u32);
    fn ktime_get_ts64(arg0: *mut timespec64);
    fn ktime_get_real_ts64(arg0: *mut timespec64);
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn mod_timer(arg0: *mut timer_list, arg1: u64) -> i32;
    fn timer_delete_sync(arg0: *mut timer_list) -> i32;
    fn timer_delete(arg0: *mut timer_list) -> i32;
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn disable_work_sync(arg0: *mut work_struct) -> bool;
    fn kfree(arg0: *const ());
    fn __compiletime_assert_208();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __request_module(arg0: bool, arg1: *const u8, ...) -> i32;
    fn __bad_copy_from();
    fn __bad_copy_to();
    fn __copy_overflow(arg0: i32, arg1: u64);
    fn _copy_from_user(arg0: *mut (), arg1: *const (), arg2: u64) -> u64;
    fn _copy_to_user(arg0: *mut (), arg1: *const (), arg2: u64) -> u64;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_395();
    fn __compiletime_assert_416();
    fn stream_open(arg0: *mut inode, arg1: *mut file) -> i32;
    fn try_module_get(arg0: *mut module) -> bool;
    fn module_put(arg0: *mut module);
    fn dev_set_name(arg0: *mut device, arg1: *const u8, ...) -> i32;
    fn get_device(arg0: *mut device) -> *mut device;
    fn put_device(arg0: *mut device);
    fn snd_device_alloc(arg0: *mut *mut device, arg1: *mut snd_card) -> i32;
    fn snd_register_device(arg0: i32, arg1: *mut snd_card, arg2: i32, arg3: *const file_operations, arg4: *mut (), arg5: *mut device) -> i32;
    fn snd_unregister_device(arg0: *mut device) -> i32;
    fn snd_device_new(arg0: *mut snd_card, arg1: u32, arg2: *mut (), arg3: *const snd_device_ops) -> i32;
    fn snd_fasync_helper(arg0: i32, arg1: *mut file, arg2: i32, arg3: *mut *mut snd_fasync) -> i32;
    fn snd_kill_fasync(arg0: *mut snd_fasync, arg1: i32, arg2: i32);
    fn snd_fasync_free(arg0: *mut snd_fasync);
    fn seq_printf(arg0: *mut seq_file, arg1: *const u8, ...);
    fn snd_info_create_module_entry(arg0: *mut module, arg1: *const u8, arg2: *mut snd_info_entry) -> *mut snd_info_entry;
    fn snd_info_free_entry(arg0: *mut snd_info_entry);
    fn snd_info_register(arg0: *mut snd_info_entry) -> i32;
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.objectsize.i64.p0"]
    fn __slate_intrinsic_2913c0303de876b3(arg0: *const (), arg1: bool, arg2: bool, arg3: bool) -> u64;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_f217c4720f04f6db(arg0: u64) -> bool;
}

// generic data direction definitions
// The type of an index into the pagecache.
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
struct io_bitmap {
}


#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
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

#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_0() };
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
__verify_pcpu_ptr!();
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
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
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


// __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
// the symbol table so that objtool can reference it when it generates the
// .static_call_sites section.
#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_key {
    func: *mut (),
    __slate_anon_1: __SlateRecord_static_call_types_h_65,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_qrwlock_types_h_27 {
    cnts: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_qrwlock_types_h_27_2,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_qrwlock_types_h_27_2 {
    wlocked: u8,
    __lstate: [u8; 3],
}


#[inline(always)]
unsafe extern "C" fn __WARN_validate_printf(fmt: *const u8, __va_args: ...) {
}

// inhibit tail-call optimization
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
struct qrwlock {
    __slate_anon_0: __SlateRecord_qrwlock_types_h_27,
    wait_lock: qspinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lock_class_key {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lockdep_map {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct raw_spinlock {
    raw_lock: qspinlock,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


fn kmsan_unpoison_memory(address: *const (), size: u64) {
}

fn kmsan_copy_to_user(to: *mut (), from: *const (), to_copy: u64, left: u64) {
}

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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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
core::arch::asm!("3: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 3b - ., 3; .popsection\n# {0}", const 3 as i32, options(nostack, att_syntax));
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
core::arch::asm!("4: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 4b - ., 4; .popsection\n# {0}", const 4 as i32, options(nostack, att_syntax));
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

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
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
struct pgprot {
    pgprot: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_295 {
    pgd: u64,
}


// No need to mask any bits for !PAE
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits102U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
}

#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[inline(always)]
fn IS_ERR_OR_NULL(ptr: *const ()) -> bool {
unlikely!(ptr.is_null()) != 0 || IS_ERR_VALUE!(ptr as u64) != 0
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits109U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[inline(always)]
fn arch_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
__WRITE_ONCE__2!((*v).counter, i);
}

#[inline(always)]
fn arch_atomic_fetch_add(mut __ret: i32, mut v: *mut __SlateRecord_types_h_190) -> i32 {
let mut __v0: i32;
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
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[inline(always)]
fn raw_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
arch_atomic_set(v, i);
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
fn raw_atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_add(i, v)
}

#[inline(always)]
fn raw_atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_sub!(i, v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[inline(always)]
fn atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set(v, i);
}

#[inline(always)]
fn atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_add_relaxed(i, v)
}

#[inline(always)]
fn atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_release!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_sub_release(i, v)
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


fn might_fault() {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_14() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*list).next, list);
if true {
break;
}
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_15() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*list).prev, list);
if true {
break;
}
}
}

fn __list_add_valid(new: *mut list_head, prev: *mut list_head, next: *mut list_head) -> bool {
true
}

fn __list_del_entry_valid(entry: *mut list_head) -> bool {
true
}

#[inline(always)]
fn __list_add(new: *mut list_head, prev: *mut list_head, next: *mut list_head) {
if !__list_add_valid(new, prev, next) {
return;
}
unsafe {
(*next).prev = new;
}
unsafe {
(*new).next = next;
}
unsafe {
(*new).prev = prev;
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_16() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*prev).next, new);
if true {
break;
}
}
}

fn list_add_tail(new: *mut list_head, head: *mut list_head) {
__list_add(new, unsafe { (*head).prev }, head);
}

fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_19() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*prev).next, next);
if true {
break;
}
}
}

fn __list_del_entry(entry: *mut list_head) {
if !__list_del_entry_valid(entry) {
return;
}
__list_del(unsafe { (*entry).prev }, unsafe { (*entry).next });
}

fn list_del(entry: *mut list_head) {
__list_del_entry(entry);
unsafe {
(*entry).next = ((256i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
unsafe {
(*entry).prev = ((290i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
}

fn list_del_init(entry: *mut list_head) {
__list_del_entry(entry);
INIT_LIST_HEAD(entry);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


fn list_move_tail(list: *mut list_head, head: *mut list_head) {
__list_del_entry(list);
list_add_tail(list, head);
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_20() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).next) as *const *mut list_head) };
(__v0 == (head as *mut list_head)) as i32
}

// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// list_for_each_entry_safe - iterate over list of given type safe against removal of list entry
// @pos:	the type * to use as a loop cursor.
// @n:		another type * to use as temporary storage
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
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
    __slate_bits_0: __slate_bits::__SlateBits110U0,
    pkru: u32,
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


// 32-bit address space on 64 bits
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


fn test_ti_thread_flag(ti: *mut thread_info, flag: i32) -> i32 {
test_bit!(flag, unsafe { core::ptr::addr_of_mut!((*ti).flags) })
}

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


#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
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
struct llist_head {
    first: *mut llist_node,
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
struct sysv_sem {
    undo_list: *mut sem_undo_list,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_shm {
    shm_clist: list_head,
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
struct rb_node {
    __rb_parent_color: u64,
    rb_right: *mut rb_node,
    rb_left: *mut rb_node,
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
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
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
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
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


// SIGPOLL (or any other signal without signal specific si_codes) si_codes
// data input available
#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
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
struct tracepoint_func {
    func: *mut (),
    data: *mut (),
    prio: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_ext {
    regfunc: Option<unsafe extern "C" fn() -> i32>,
    unregfunc: Option<unsafe extern "C" fn()>,
    __slate_bits_0: __slate_bits::__SlateBits205U0,
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


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

#[inline(always)]
fn spin_lock(lock: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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


#[inline(always)]
fn spin_unlock_irq(lock: *mut spinlock) {
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
}

static class_spinlock_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_spinlock_h_610 {
    lock: *mut spinlock,
}


#[inline(always)]
fn class_spinlock_destructor(_T: *mut __SlateRecord_spinlock_h_610) {
spin_unlock(unsafe { (*_T).lock });
}

#[inline(always)]
fn class_spinlock_lock_ptr(_T: &mut __SlateRecord_spinlock_h_610) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_spinlock_h_610)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[inline(always)]
fn class_spinlock_constructor(l: *mut spinlock) -> __SlateRecord_spinlock_h_610 {
let mut _t: __SlateRecord_spinlock_h_610 = __SlateRecord_spinlock_h_610 { lock: l };
let _T: *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(_t);
spin_lock(unsafe { (*_T).lock });
return _t;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
}


#[inline(always)]
fn __class_spinlock_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_610) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
}


static class_spinlock_irq_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_spinlock_h_620 {
    lock: *mut spinlock,
}


#[inline(always)]
fn class_spinlock_irq_destructor(_T: *mut __SlateRecord_spinlock_h_620) {
spin_unlock_irq(unsafe { (*_T).lock });
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


#[inline(always)]
fn class_spinlock_irq_lock_ptr(_T: &mut __SlateRecord_spinlock_h_620) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_spinlock_h_620)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[inline(always)]
fn class_spinlock_irq_constructor(l: *mut spinlock) -> __SlateRecord_spinlock_h_620 {
let mut _t: __SlateRecord_spinlock_h_620 = __SlateRecord_spinlock_h_620 { lock: l };
let _T: *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(_t);
spin_lock_irq(unsafe { (*_T).lock });
return _t;
}

#[inline(always)]
fn __class_spinlock_irq_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_620) {
}

static class_spinlock_irqsave_is_conditional: bool = 0i32 != 0i32;

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
struct __SlateRecord_spinlock_h_642 {
    lock: *mut spinlock,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
}


#[inline(always)]
fn class_spinlock_irqsave_destructor(_T: *mut __SlateRecord_spinlock_h_642) {
spin_unlock_irqrestore(unsafe { (*_T).lock }, unsafe { (*_T).flags });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
}


#[inline(always)]
fn class_spinlock_irqsave_lock_ptr(_T: &mut __SlateRecord_spinlock_h_642) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_spinlock_h_642)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[inline(always)]
fn class_spinlock_irqsave_constructor(l: *mut spinlock) -> __SlateRecord_spinlock_h_642 {
let mut _t: __SlateRecord_spinlock_h_642 = {
let mut __t0: __SlateRecord_spinlock_h_642 = unsafe { core::mem::zeroed() };
__t0.lock = l;
__t0
};
let _T: *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(_t);
spin_lock_irqsave!(unsafe { (*_T).lock }, (*_T).flags);
return _t;
}

#[inline(always)]
fn __class_spinlock_irqsave_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_642) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_spinlock_h_670 {
    lock: *mut rwlock,
    flags: u64,
}


#[inline(always)]
fn class_read_lock_irqsave_destructor(_T: *mut __SlateRecord_spinlock_h_670) {
read_unlock_irqrestore!(unsafe { (*_T).lock }, unsafe { (*_T).flags });
}

#[inline(always)]
fn class_read_lock_irqsave_constructor(l: *mut rwlock) -> __SlateRecord_spinlock_h_670 {
let mut _t: __SlateRecord_spinlock_h_670 = {
let mut __t0: __SlateRecord_spinlock_h_670 = unsafe { core::mem::zeroed() };
__t0.lock = l;
__t0
};
let _T: *mut __SlateRecord_spinlock_h_670 = core::ptr::addr_of_mut!(_t);
read_lock_irqsave!(unsafe { (*_T).lock }, (*_T).flags);
return _t;
}

#[inline(always)]
fn __class_read_lock_irqsave_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_670) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_spinlock_h_683 {
    lock: *mut rwlock,
}


#[inline(always)]
fn class_write_lock_irq_destructor(_T: *mut __SlateRecord_spinlock_h_683) {
write_unlock_irq!(unsafe { (*_T).lock });
}

#[inline(always)]
fn class_write_lock_irq_constructor(l: *mut rwlock) -> __SlateRecord_spinlock_h_683 {
let mut _t: __SlateRecord_spinlock_h_683 = __SlateRecord_spinlock_h_683 { lock: l };
let _T: *mut __SlateRecord_spinlock_h_683 = core::ptr::addr_of_mut!(_t);
write_lock_irq!(unsafe { (*_T).lock });
return _t;
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[inline(always)]
fn __class_write_lock_irq_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_683) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_tlbflush_unmap_batch {
    cpumask: cpumask,
    unmapped_pages: bool,
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
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
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
struct posix_cputimers_work {
    work: callback_head,
    mutex: mutex,
    scheduled: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_work {
    node: __call_single_node,
    func: Option<unsafe extern "C" fn(*mut irq_work)>,
    irqwait: rcuwait,
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
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
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
struct ldt_struct {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
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
union __SlateRecord_mm_types_h_406 {
    __slate_anon_0: __SlateRecord_mm_types_h_407,
    page: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_407 {
    flags: __SlateRecord_mm_types_h_41,
    __slate_anon_1: __SlateRecord_mm_types_h_410,
    mapping: *mut address_space,
    __slate_anon_3: __SlateRecord_mm_types_h_425,
    __slate_anon_4: __SlateRecord_mm_types_h_429,
    _mapcount: __SlateRecord_types_h_190,
    _refcount: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_410 {
    lru: list_head,
    __slate_anon_1: __SlateRecord_mm_types_h_414,
    pgmap: *mut dev_pagemap,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_414 {
    __filler: *mut (),
    mlock_count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pagemap {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_425 {
    index: u64,
    share: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_429 {
    private: *mut (),
    swap: __SlateRecord_mm_types_h_288,
    migrate_info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_451 {
    __slate_anon_0: __SlateRecord_mm_types_h_452,
    __page_1: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_452 {
    _flags_1: u64,
    _head_1: u64,
    __slate_anon_2: __SlateRecord_mm_types_h_455,
    _mapcount_1: __SlateRecord_types_h_190,
    _refcount_1: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_455 {
    __slate_anon_0: __SlateRecord_mm_types_h_456,
    _usable_1: [u64; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_456 {
    _large_mapcount: __SlateRecord_types_h_190,
    _nr_pages_mapped: __SlateRecord_types_h_190,
    _entire_mapcount: __SlateRecord_types_h_190,
    _pincount: __SlateRecord_types_h_190,
    _mm_id_mapcount: [i32; 2],
    __slate_anon_5: __SlateRecord_mm_types_h_465,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_465 {
    _mm_id: [u32; 2],
    _mm_ids: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_483 {
    __slate_anon_0: __SlateRecord_mm_types_h_484,
    __page_2: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_484 {
    _flags_2: u64,
    _head_2: u64,
    _deferred_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_497 {
    __slate_anon_0: __SlateRecord_mm_types_h_498,
    __page_3: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_498 {
    _flags_3: u64,
    _head_3: u64,
    _hugetlb_subpool: *mut (),
    _hugetlb_cgroup: *mut (),
    _hugetlb_cgroup_rsvd: *mut (),
    _hugetlb_hwpoison: *mut (),
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
union __SlateRecord_mm_types_h_830 {
    remap: __SlateRecord_mm_types_h_831,
    simple_ioremap: __SlateRecord_mm_types_h_837,
    map_kernel: __SlateRecord_mm_types_h_841,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_831 {
    start: u64,
    start_pfn: u64,
    size: u64,
    pgprot: pgprot,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_837 {
    start_phys_addr: u64,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_841 {
    start: u64,
    pages: *mut *mut page,
    nr_pages: u64,
    pgoff: u64,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_operations_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_926 {
    __slate_anon_0: __SlateRecord_mm_types_h_927,
    vm_freeptr: __SlateRecord_mm_types_h_701,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_927 {
    vm_start: u64,
    vm_end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_947 {
    vm_flags: u64,
    flags: __SlateRecord_mm_types_h_871,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct anon_vma {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1062 {
    rb: rb_node,
    rb_subtree_last: u64,
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
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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
struct user_namespace {
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
    __slate_bits_0: __slate_bits::__SlateBits283U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_common {
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
struct fs_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
    __slate_bits_0: __slate_bits::__SlateBits65U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits65U1,
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
    __slate_bits_2: __slate_bits::__SlateBits65U2,
    mce_kill_me: callback_head,
    mce_count: i32,
    kretprobe_instances: llist_head,
    rethooks: llist_head,
    l1d_flush_kill: callback_head,
    unwind_info: unwind_task_info,
    thread: thread_struct,
}


// Per process flags
// I'm a virtual CPU
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
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

// wait_queue_entry::flags
#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_entry {
    flags: u32,
    private: *mut (),
    func: Option<unsafe extern "C" fn(*mut wait_queue_entry, u32, i32, *mut ()) -> i32>,
    entry: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sysfs_h_107 {
    is_visible: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, i32) -> u16>,
    is_visible_const: Option<unsafe extern "C" fn(*mut kobject, *const attribute, i32) -> u16>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sysfs_h_118 {
    attrs: *mut *mut attribute,
    attrs_const: *const *const attribute,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


fn init_waitqueue_entry(wq_entry: &mut wait_queue_entry, p: *mut task_struct) {
unsafe {
(*(wq_entry as *mut wait_queue_entry)).flags = 0;
}
unsafe {
(*(wq_entry as *mut wait_queue_entry)).private = p as *mut ();
}
unsafe {
(*(wq_entry as *mut wait_queue_entry)).func = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut wait_queue_entry, u32, i32, *mut ()) -> i32>>(default_wake_function as *const ()) };
}
}

fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
}

static class_mutex_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mutex_h_253 {
    lock: *mut mutex,
}


#[inline(always)]
fn class_mutex_destructor(_T: *mut __SlateRecord_mutex_h_253) {
unsafe { mutex_unlock((*_T).lock) };
}

#[inline(always)]
fn class_mutex_lock_ptr(_T: &mut __SlateRecord_mutex_h_253) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_mutex_h_253)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[inline(always)]
fn class_mutex_constructor(l: *mut mutex) -> __SlateRecord_mutex_h_253 {
let mut _t: __SlateRecord_mutex_h_253 = __SlateRecord_mutex_h_253 { lock: l };
let _T: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(_t);
unsafe { mutex_lock((*_T).lock) };
return _t;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
}


#[inline(always)]
fn __class_mutex_cleanup_ctx(_T: *mut *mut __SlateRecord_mutex_h_253) {
}

fn refcount_set(r: *mut refcount_struct, n: i32) {
atomic_set(unsafe { core::ptr::addr_of_mut!((*r).refs) }, n);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
}


fn __refcount_add(i: i32, r: *mut refcount_struct, oldp: *mut i32) {
let old: i32 = atomic_fetch_add_relaxed(i, unsafe { core::ptr::addr_of_mut!((*r).refs) });
if !oldp.is_null() {
unsafe {
*oldp = old;
}
}
if unlikely!(old == 0) != 0 {
unsafe { refcount_warn_saturate(r, 2u32) };
} else {
if unlikely!(old < 0 || old.wrapping_add(i) < 0) != 0 {
unsafe { refcount_warn_saturate(r, 1u32) };
}
}
}

fn __refcount_inc(r: *mut refcount_struct, oldp: *mut i32) {
__refcount_add(1i32, r, oldp);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_handle {
}


fn refcount_inc(r: *mut refcount_struct) {
__refcount_inc(r, core::ptr::null_mut::<i32>());
}

fn __refcount_sub_and_test(i: i32, r: *mut refcount_struct, oldp: *mut i32) -> bool {
let old: i32 = atomic_fetch_sub_release(i, unsafe { core::ptr::addr_of_mut!((*r).refs) });
if !oldp.is_null() {
unsafe {
*oldp = old;
}
}
if old > 0 && old == i {
loop {
kcsan_rmb!();
loop {
kcsan_rmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
if true {
break;
}
}
return true;
}
if unlikely!(old <= 0 || old.wrapping_sub(i) < 0) != 0 {
unsafe { refcount_warn_saturate(r, 3u32) };
}
false
}

fn __refcount_dec_and_test(r: *mut refcount_struct, oldp: *mut i32) -> bool {
__refcount_sub_and_test(1i32, r, oldp)
}

fn refcount_dec_and_test(r: *mut refcount_struct) -> bool {
__refcount_dec_and_test(r, core::ptr::null_mut::<i32>())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
}


fn kref_init(kref: *mut kref) {
refcount_set(unsafe { core::ptr::addr_of_mut!((*kref).refcount) }, 1i32);
}

fn kref_get(kref: *mut kref) {
refcount_inc(unsafe { core::ptr::addr_of_mut!((*kref).refcount) });
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_58 {
    __ubuf_iovec: iovec,
    __slate_anon_1: __SlateRecord_uio_h_65,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uio_h_65 {
    __slate_anon_0: __SlateRecord_uio_h_66,
    count: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_66 {
    __iov: *const iovec,
    kvec: *const kvec,
    bvec: *const bio_vec,
    folioq: *const folio_queue,
    xarray: *mut xarray,
    ubuf: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_vec {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


fn kref_put(kref: *mut kref, release: Option<unsafe extern "C" fn(*mut kref)>) -> i32 {
if refcount_dec_and_test(unsafe { core::ptr::addr_of_mut!((*kref).refcount) }) {
unsafe { release.unwrap()(kref) };
return 1;
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


// Dependency order vs. p above.
// Dependency order vs. p above.
// RCU_INITIALIZER() - statically initialize an RCU-protected global variable
// @v: The value to statically initialize with.
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
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


fn __init_work(work: *mut work_struct, onstack: i32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


fn queue_work(wq: *mut workqueue_struct, work: *mut work_struct) -> bool {
unsafe { queue_work_on(64i32, wq, work) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_segcblist {
    head: *mut callback_head,
    tails: [*mut *mut callback_head; 4],
    gp_seq: [rcu_gp_seq; 4],
    len: i64,
    seglen: [i64; 4],
    flags: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_ctr {
    srcu_locks: __SlateRecord_types_h_197,
    srcu_unlocks: __SlateRecord_types_h_197,
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct srcu_data {
    srcu_ctrs: [srcu_ctr; 2],
    srcu_reader_flavor: i32,
    __slate_pad_2: [u8; 28],
    lock: raw_spinlock,
    srcu_cblist: rcu_segcblist,
    srcu_gp_seq_needed: u64,
    srcu_gp_seq_needed_exp: u64,
    srcu_cblist_invoking: bool,
    delay_work: timer_list,
    work: work_struct,
    srcu_barrier_head: callback_head,
    srcu_ec_head: callback_head,
    srcu_ec_state: i32,
    mynode: *mut srcu_node,
    grpmask: u64,
    cpu: i32,
    ssp: *mut srcu_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_node {
    lock: raw_spinlock,
    srcu_have_cbs: [u64; 4],
    srcu_data_have_cbs: [u64; 4],
    srcu_gp_seq_needed_exp: u64,
    srcu_parent: *mut srcu_node,
    grplo: i32,
    grphi: i32,
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
struct srcu_usage {
    node: *mut srcu_node,
    level: [*mut srcu_node; 3],
    srcu_size_state: i32,
    srcu_cb_mutex: mutex,
    lock: raw_spinlock,
    srcu_gp_mutex: mutex,
    srcu_gp_seq: u64,
    srcu_gp_seq_needed: u64,
    srcu_gp_seq_needed_exp: u64,
    srcu_gp_start: u64,
    srcu_last_gp_end: u64,
    srcu_size_jiffies: u64,
    srcu_n_lock_retries: u64,
    srcu_n_exp_nodelay: u64,
    sda_is_static: bool,
    srcu_barrier_seq: u64,
    srcu_barrier_mutex: mutex,
    srcu_barrier_completion: completion,
    srcu_barrier_cpu_cnt: __SlateRecord_types_h_190,
    reschedule_jiffies: u64,
    reschedule_count: u64,
    work: delayed_work,
    irq_work: irq_work,
    srcu_ssp: *mut srcu_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_struct {
    srcu_ctrp: *mut srcu_ctr,
    sda: *mut srcu_data,
    srcu_reader_flavor: u8,
    dep_map: lockdep_map,
    srcu_sup: *mut srcu_usage,
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_288 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio {
    __slate_anon_0: __SlateRecord_mm_types_h_406,
    __slate_anon_1: __SlateRecord_mm_types_h_451,
    __slate_anon_2: __SlateRecord_mm_types_h_483,
    __slate_anon_3: __SlateRecord_mm_types_h_497,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pfnmap_track_ctx {
    kref: kref,
    pfn: u64,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits413U0,
}


// Opaque type representing current VMA (vm_area_struct) flag state. Must be
// accessed via vma_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_871 {
    __vma_flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_area_desc {
    mm: *mut mm_struct,
    file: *mut file,
    start: u64,
    end: u64,
    pgoff: u64,
    vm_file: *mut file,
    vma_flags: __SlateRecord_mm_types_h_871,
    page_prot: pgprot,
    vm_ops: *const vm_operations_struct,
    private_data: *mut (),
    action: mmap_action,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct vm_area_struct {
    __slate_anon_0: __SlateRecord_mm_types_h_926,
    vm_mm: *mut mm_struct,
    vm_page_prot: pgprot,
    __slate_anon_3: __SlateRecord_mm_types_h_947,
    vm_lock_seq: u32,
    __vm_anon_pgoff_lo: u32,
    anon_vma_chain: list_head,
    anon_vma: *mut anon_vma,
    vm_ops: *const vm_operations_struct,
    vm_pgoff: u64,
    vm_file: *mut file,
    vm_private_data: *mut (),
    swap_readahead_info: __SlateRecord_types_h_197,
    vm_policy: *mut mempolicy,
    __slate_pad_14: [u8; 8],
    vm_refcnt: refcount_struct,
    __vm_anon_pgoff_hi: u32,
    shared: __SlateRecord_mm_types_h_1062,
    vm_userfaultfd_ctx: vm_userfaultfd_ctx,
    pfnmap_track_ctx: *mut pfnmap_track_ctx,
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
struct __SlateRecord_slab_h_532 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
}


// no-op
#[inline(always)]
fn __free_kfree(p: *mut ()) {
let _T: *mut () = unsafe { *(p as *mut *mut ()) };
if !IS_ERR_OR_NULL(_T as *const ()) {
unsafe { kfree(_T as *const ()) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct export_operations {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fsnotify_sb_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kstatfs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mount {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mtd_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct super_dev {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct writeback_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
}


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
unsafe { __compiletime_assert_208() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("209: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 209b - ., 3; .popsection\n# {0}", const 209 as i32, options(nostack, att_syntax));
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_lockref_h_26 {
    lock_count: u64,
    __slate_anon_1: __SlateRecord_lockref_h_30,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_lockref_h_30 {
    lock: spinlock,
    count: i32,
}


// __alloc_objs - Allocate objects of a given type using
// @KMALLOC: which size-based kmalloc wrapper to allocate with.
// @GFP: GFP flags for the allocation.
// @TYPE: type to allocate space for.
// @COUNT: how many @TYPE objects to allocate.
//
// Returns: Newly allocated pointer to (first) @TYPE of @COUNT-many
// allocated @TYPE objects, or NULL on failure.
// All kzalloc aliases for kmalloc_(obj|objs|flex).
fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dcache_h_50 {
    __slate_anon_0: __SlateRecord_dcache_h_51,
    hash_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dcache_h_51 {
    hash: u32,
    len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_dir {
    subdirs: u64,
    children: rb_root,
    root: *mut kernfs_root,
    rev: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dcache_h_99 {
    __d_name: qstr,
    d_name: qstr,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dcache_h_126 {
    d_alias: hlist_node,
    d_in_lookup_hash: hlist_bl_node,
    d_rcu: callback_head,
    waiters: *mut completion_list,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_symlink {
    target_kn: *mut kernfs_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_attr {
    ops: *const kernfs_ops,
    open: *mut kernfs_open_node,
    size: i64,
    notify_next: *mut kernfs_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_node {
    count: __SlateRecord_types_h_190,
    active: __SlateRecord_types_h_190,
    __parent: *mut kernfs_node,
    name: *const u8,
    rb: rb_node,
    ns: *const ns_common,
    hash: u32,
    flags: u16,
    mode: u16,
    __slate_anon_9: __SlateRecord_kernfs_h_221,
    id: u64,
    r#priv: *mut (),
    iattr: *mut kernfs_iattrs,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_file {
    kn: *mut kernfs_node,
    file: *mut file,
    seq_file: *mut seq_file,
    r#priv: *mut (),
    mutex: mutex,
    prealloc_mutex: mutex,
    event: i32,
    list: list_head,
    prealloc_buf: *mut u8,
    atomic_write_len: u64,
    __slate_bits_0: __slate_bits::__SlateBits461U0,
    vm_ops: *const vm_operations_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_ops {
    open: Option<unsafe extern "C" fn(*mut kernfs_open_file) -> i32>,
    release: Option<unsafe extern "C" fn(*mut kernfs_open_file)>,
    seq_show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
    seq_start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    seq_next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    seq_stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    read: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    atomic_write_len: u64,
    prealloc: bool,
    write: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    poll: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut poll_table_struct) -> u32>,
    mmap: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut vm_area_struct) -> i32>,
    llseek: Option<unsafe extern "C" fn(*mut kernfs_open_file, i64, i32) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fsnotify_mark_connector {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_parameter_spec {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_kattr {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_272 {
    ia_uid: __SlateRecord_uidgid_types_h_9,
    ia_vfsuid: __SlateRecord_mnt_idmapping_h_17,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_276 {
    ia_gid: __SlateRecord_uidgid_types_h_13,
    ia_vfsgid: __SlateRecord_mnt_idmapping_h_21,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct readahead_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_page_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kobj_ns_type_operations {
    r#type: u32,
    current_may_mount: Option<unsafe extern "C" fn() -> bool>,
    grab_current_ns: Option<unsafe extern "C" fn() -> *mut ns_common>,
    netlink_ns: Option<unsafe extern "C" fn(*mut sock) -> *const ns_common>,
    initial_ns: Option<unsafe extern "C" fn() -> *const ns_common>,
    drop_ns: Option<unsafe extern "C" fn(*mut ns_common)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kstat {
    result_mask: u32,
    mode: u16,
    nlink: u32,
    blksize: u32,
    attributes: u64,
    attributes_mask: u64,
    ino: u64,
    dev: u32,
    rdev: u32,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    size: i64,
    atime: timespec64,
    mtime: timespec64,
    ctime: timespec64,
    btime: timespec64,
    blocks: u64,
    mnt_id: u64,
    change_cookie: u64,
    subvol: u64,
    dio_mem_align: u32,
    dio_offset_align: u32,
    dio_read_offset_align: u32,
    atomic_write_unit_min: u32,
    atomic_write_unit_max: u32,
    atomic_write_unit_max_opt: u32,
    atomic_write_segments_max: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct attribute {
    name: *const u8,
    mode: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_790 {
    i_nlink: u32,
    __i_nlink: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_834 {
    i_dentry: hlist_head,
    i_rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_846 {
    i_fop: *const file_operations,
    free_inode: Option<unsafe extern "C" fn(*mut inode)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_lock_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_852 {
    i_devices: list_head,
    i_linklen: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_856 {
    i_pipe: *mut pipe_inode_info,
    i_cdev: *mut cdev,
    i_link: *mut u8,
    i_dir_seq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdev {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct attribute_group {
    name: *const u8,
    __slate_anon_1: __SlateRecord_sysfs_h_107,
    is_bin_visible: Option<unsafe extern "C" fn(*mut kobject, *const bin_attribute, i32) -> u16>,
    bin_size: Option<unsafe extern "C" fn(*mut kobject, *const bin_attribute, i32) -> u64>,
    __slate_anon_4: __SlateRecord_sysfs_h_118,
    bin_attrs: *const *const bin_attribute,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bin_attribute {
    attr: attribute,
    size: u64,
    private: *mut (),
    f_mapping: Option<unsafe extern "C" fn() -> *mut address_space>,
    read: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, *mut u8, i64, u64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, *mut u8, i64, u64) -> i64>,
    llseek: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, i64, i32) -> i64>,
    mmap: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, *mut vm_area_struct) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_1267 {
    f_path: path,
    __f_path: path,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_1271 {
    f_pos_lock: mutex,
    f_pipe: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_1287 {
    f_task_work: callback_head,
    f_llist: llist_node,
    f_ra: file_ra_state,
    f_freeptr: __SlateRecord_mm_types_h_701,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_lock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_lease {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysfs_ops {
    show: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *const u8, u64) -> i64>,
}


// Permissions on a sysfs file: you didn't miss the 0 prefix did you?
// USER_READABLE >= GROUP_READABLE >= OTHER_READABLE
// USER_WRITABLE >= GROUP_WRITABLE
// OTHER_WRITABLE?  Generally considered a bad idea.
#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
}


// buffer for the variables
#[repr(C)]
#[derive(Clone, Copy)]
struct kobject {
    name: *const u8,
    entry: list_head,
    parent: *mut kobject,
    kset: *mut kset,
    ktype: *const kobj_type,
    sd: *mut kernfs_node,
    kref: kref,
    __slate_bits_0: __slate_bits::__SlateBits463U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kobj_type {
    release: Option<unsafe extern "C" fn(*mut kobject)>,
    sysfs_ops: *const sysfs_ops,
    default_groups: *mut *const attribute_group,
    child_ns_type: Option<unsafe extern "C" fn(*const kobject) -> *const kobj_ns_type_operations>,
    namespace: Option<unsafe extern "C" fn(*const kobject) -> *const ns_common>,
    get_ownership: Option<unsafe extern "C" fn(*const kobject, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kobj_uevent_env {
    argv: [*mut u8; 3],
    envp: [*mut u8; 64],
    envp_idx: i32,
    buf: [u8; 2048],
    buflen: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kset_uevent_ops {
    filter: Option<unsafe extern "C" fn(*const kobject) -> i32>,
    name: Option<unsafe extern "C" fn(*const kobject) -> *const u8>,
    uevent: Option<unsafe extern "C" fn(*const kobject, *mut kobj_uevent_env) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_message {
    event: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_ops {
    prepare: Option<unsafe extern "C" fn(*mut device) -> i32>,
    complete: Option<unsafe extern "C" fn(*mut device)>,
    suspend: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_suspend: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_idle: Option<unsafe extern "C" fn(*mut device) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_subsys_data {
    lock: spinlock,
    refcount: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits488U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits488U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits488U2,
    links_count: u32,
    request: u32,
    runtime_status: i32,
    last_status: i32,
    runtime_error: i32,
    autosuspend_delay: i32,
    last_busy: u64,
    active_time: u64,
    suspended_time: u64,
    accounting_timestamp: u64,
    subsys_data: *mut pm_subsys_data,
    set_latency_tolerance: Option<unsafe extern "C" fn(*mut device, i32)>,
    qos: *mut dev_pm_qos,
    __slate_bits_3: __slate_bits::__SlateBits488U3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_domain {
    ops: dev_pm_ops,
    start: Option<unsafe extern "C" fn(*mut device) -> i32>,
    detach: Option<unsafe extern "C" fn(*mut device, bool)>,
    activate: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync: Option<unsafe extern "C" fn(*mut device)>,
    dismiss: Option<unsafe extern "C" fn(*mut device)>,
    set_performance_state: Option<unsafe extern "C" fn(*mut device, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bus_type {
    name: *const u8,
    dev_name: *const u8,
    bus_groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    drv_groups: *const *const attribute_group,
    r#match: Option<unsafe extern "C" fn(*mut device, *const device_driver) -> i32>,
    uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    probe: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync_state: Option<unsafe extern "C" fn(*mut device)>,
    remove: Option<unsafe extern "C" fn(*mut device)>,
    shutdown: Option<unsafe extern "C" fn(*mut device)>,
    irq_get_affinity: Option<unsafe extern "C" fn(*mut device, u32) -> *const cpumask>,
    online: Option<unsafe extern "C" fn(*mut device) -> i32>,
    offline: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend: Option<unsafe extern "C" fn(*mut device, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    num_vf: Option<unsafe extern "C" fn(*mut device) -> i32>,
    dma_configure: Option<unsafe extern "C" fn(*mut device) -> i32>,
    dma_cleanup: Option<unsafe extern "C" fn(*mut device)>,
    pm: *const dev_pm_ops,
    driver_override: bool,
    need_parent_lock: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct class {
    name: *const u8,
    class_groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    dev_uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    devnode: Option<unsafe extern "C" fn(*const device, *mut u16) -> *mut u8>,
    class_release: Option<unsafe extern "C" fn(*const class)>,
    dev_release: Option<unsafe extern "C" fn(*mut device)>,
    shutdown_pre: Option<unsafe extern "C" fn(*mut device) -> i32>,
    ns_type: *const kobj_ns_type_operations,
    namespace: Option<unsafe extern "C" fn(*const device) -> *const ns_common>,
    get_ownership: Option<unsafe extern "C" fn(*const device, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
    pm: *const dev_pm_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhash_head {
    next: *mut rhash_head,
}


fn check_object_size(ptr: *const (), n: u64, to_user: bool) {
}

fn copy_overflow(size: i32, count: u64) {
if IS_ENABLED__2!() != 0 {
unsafe { __copy_overflow(size, count) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_symbol {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_sect_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_notes_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_call {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_eval_map {
}


#[inline(always)]
fn check_copy_size(addr: *const (), bytes: u64, is_source: bool) -> bool {
let sz: i32 = (unsafe { __slate_intrinsic_2913c0303de876b3(addr, const { false as bool }, const { false as bool }, const { false as bool }) }) as i32;
if unlikely!(sz >= 0 && (sz as u64) < bytes) != 0 {
if !unsafe { __slate_intrinsic_f217c4720f04f6db(bytes) } {
copy_overflow(sz, bytes);
} else {
if is_source {
unsafe { __bad_copy_from() };
} else {
unsafe { __bad_copy_to() };
}
}
return false;
}
let mut __v0: i64;
let __ret_warn_on: i32 = (bytes > 2147483647u64) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("249: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 249b - ., 3; .popsection\n# {0}", const 249 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_check_copy_size_1, sym __slate_asm_string_check_copy_size_2, const 57 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("250: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 250b - ., 4; .popsection\n# {0}", const 250 as i32, options(nostack, att_syntax));
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
check_object_size(addr, bytes, is_source);
true
}

#[repr(C)]
#[derive(Clone, Copy)]
struct iovec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kvec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_iommu {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_device_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iov_iter {
    iter_type: u8,
    nofault: bool,
    data_source: bool,
    iov_offset: u64,
    __slate_anon_4: __SlateRecord_uio_h_58,
    __slate_anon_5: __SlateRecord_uio_h_78,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct shrink_control {
    gfp_mask: u32,
    nid: i32,
    nr_to_scan: u64,
    nr_scanned: u64,
    memcg: *mut mem_cgroup,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct shrinker {
    count_objects: Option<unsafe extern "C" fn(*mut shrinker, *mut shrink_control) -> u64>,
    scan_objects: Option<unsafe extern "C" fn(*mut shrinker, *mut shrink_control) -> u64>,
    batch: i64,
    seeks: i32,
    flags: u32,
    refcount: refcount_struct,
    done: completion,
    rcu: callback_head,
    private_data: *mut (),
    list: list_head,
    nr_deferred: *mut __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_device_h_719 {
    name: *const u8,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bus_dma_region {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_tlb_mem {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct list_lru_node {
    lru: list_lru_one,
    nr_items: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru {
    node: *mut list_lru_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_bl_node {
    next: *mut hlist_bl_node,
    pprev: *mut *mut hlist_bl_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uuid_h_21 {
    b: [u8; 16],
}


// The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
// not including trailing NUL.
#[repr(C)]
#[derive(Clone, Copy)]
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_shutdown_f_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_fasync {
}


#[inline(always)]
fn copy_from_user(to: *mut (), from: *const (), n: u64) -> u64 {
if !check_copy_size(to as *const (), n, false) {
return n;
}
unsafe { _copy_from_user(to, from, n) }
}

#[inline(always)]
fn copy_to_user(to: *mut (), from: *const (), n: u64) -> u64 {
if !check_copy_size(from, n, true) {
return n;
}
unsafe { _copy_to_user(to, from, n) }
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


// current_cred - Access the current task's subjective credentials
//
// Access the subjective credentials of the current task.  RCU-safe,
// since nobody else can modify it.
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_info_h_65 {
    text: snd_info_entry_text,
    ops: *const snd_info_entry_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_41 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_41_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_41_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_41_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_41_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_41_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_43 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_43_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_43_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_43_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_43_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_43_6 {
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
struct __SlateRecord_timer_c_1012 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_1012_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpu_itimer {
    expires: u64,
    incr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_1289 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_1289_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_1289_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_1724 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_1724_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_1725 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_1725_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_1864 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_1864_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_1865 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_c_1865_2 {
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
struct __SlateRecord_timer_compat_c_65 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_compat_c_65_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_compat_c_66 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timer_compat_c_66_2 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[repr(transparent)]
struct __SlateConst<T>(T);

unsafe impl<T> Sync for __SlateConst<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits283U0 {
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
    pub struct __SlateBits461U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits463U0 {
        #[bits(1)]
        pub state_initialized: u32,
        #[bits(1)]
        pub state_in_sysfs: u32,
        #[bits(1)]
        pub state_add_uevent_sent: u32,
        #[bits(1)]
        pub state_remove_uevent_sent: u32,
        #[bits(1)]
        pub uevent_suppress: u32,
        #[bits(3, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits109U0 {
        #[bits(4)]
        pub sp_reg: u32,
        #[bits(4)]
        pub bp_reg: u32,
        #[bits(3)]
        pub r#type: u32,
        #[bits(1)]
        pub signal: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits413U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits65U0 {
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
    pub struct __SlateBits65U1 {
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
    pub struct __SlateBits286U0 {
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
    pub struct __SlateBits65U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits102U0 {
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
    pub struct __SlateBits110U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits205U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits488U0 {
        #[bits(1)]
        pub can_wakeup: bool,
        #[bits(1)]
        pub async_suspend: bool,
        #[bits(1)]
        pub in_dpm_list: bool,
        #[bits(1)]
        pub is_prepared: bool,
        #[bits(1)]
        pub is_suspended: bool,
        #[bits(1)]
        pub is_noirq_suspended: bool,
        #[bits(1)]
        pub is_late_suspended: bool,
        #[bits(1)]
        pub no_pm: bool,
        #[bits(1)]
        pub early_init: bool,
        #[bits(1)]
        pub direct_complete: bool,
        #[bits(6, access = na)]
        pub __slate_pad_10: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits485U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits488U1 {
        #[bits(1)]
        pub wakeup_path: bool,
        #[bits(1)]
        pub syscore: bool,
        #[bits(1)]
        pub no_pm_callbacks: bool,
        #[bits(1)]
        pub smart_suspend: bool,
        #[bits(1)]
        pub must_resume: bool,
        #[bits(1)]
        pub may_skip_resume: bool,
        #[bits(1)]
        pub out_band_wakeup: bool,
        #[bits(1)]
        pub strict_midlayer: bool,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits488U2 {
        #[bits(3)]
        pub disable_depth: u32,
        #[bits(1)]
        pub idle_notification: bool,
        #[bits(1)]
        pub request_pending: bool,
        #[bits(1)]
        pub deferred_resume: bool,
        #[bits(1)]
        pub needs_force_resume: bool,
        #[bits(1)]
        pub runtime_auto: bool,
        #[bits(1)]
        pub ignore_children: bool,
        #[bits(1)]
        pub no_callbacks: bool,
        #[bits(1)]
        pub irq_safe: bool,
        #[bits(1)]
        pub use_autosuspend: bool,
        #[bits(1)]
        pub timer_autosuspends: bool,
        #[bits(1)]
        pub memalloc_noio: bool,
        #[bits(2, access = na)]
        pub __slate_pad_12: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits488U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
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
    __slate_bits_0: __slate_bits::__SlateBits286U0,
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
unlikely__2!(test_tsk_thread_flag(p, 1i32)) as i32
}

fn signal_pending(p: *mut task_struct) -> i32 {
// TIF_NOTIFY_SIGNAL isn't really a signal, but it requires the same
// behavior in terms of ensuring that we break out of wait loops
// so that notify signal callbacks can be processed.
if unlikely__2!(test_tsk_thread_flag(p, 2i32)) != 0 {
return 1;
}
task_sigpending(p)
}

fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_395() };
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

#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_sync {
    gp_state: i32,
    gp_count: i32,
    gp_wait: wait_queue_head,
    cb_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_rw_semaphore {
    rss: rcu_sync,
    read_count: *mut u32,
    writer: rcuwait,
    waiters: wait_queue_head,
    block: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_projid_h_24 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kqid {
    __slate_anon_0: __SlateRecord_quota_h_69,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_dqblk {
    dqb_bhardlimit: i64,
    dqb_bsoftlimit: i64,
    dqb_curspace: i64,
    dqb_rsvspace: i64,
    dqb_ihardlimit: i64,
    dqb_isoftlimit: i64,
    dqb_curinodes: i64,
    dqb_btime: i64,
    dqb_itime: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_dqinfo {
    dqi_format: *mut quota_format_type,
    dqi_fmt_id: i32,
    dqi_dirty_list: list_head,
    dqi_flags: u64,
    dqi_bgrace: u32,
    dqi_igrace: u32,
    dqi_max_spc_limit: i64,
    dqi_max_ino_limit: i64,
    dqi_priv: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dquot {
    dq_hash: hlist_node,
    dq_inuse: list_head,
    dq_free: list_head,
    dq_dirty: list_head,
    dq_lock: mutex,
    dq_dqb_lock: spinlock,
    dq_count: __SlateRecord_types_h_190,
    dq_sb: *mut super_block,
    dq_id: kqid,
    dq_off: i64,
    dq_flags: u64,
    dq_dqb: mem_dqblk,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quota_format_ops {
    check_quota_file: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    read_file_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    write_file_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    free_file_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    read_dqblk: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    commit_dqblk: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    release_dqblk: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    get_next_id: Option<unsafe extern "C" fn(*mut super_block, *mut kqid) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dquot_operations {
    write_dquot: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    alloc_dquot: Option<unsafe extern "C" fn(*mut super_block, i32) -> *mut dquot>,
    destroy_dquot: Option<unsafe extern "C" fn(*mut dquot)>,
    acquire_dquot: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    release_dquot: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    mark_dirty: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    write_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    get_reserved_space: Option<unsafe extern "C" fn(*mut inode) -> *mut i64>,
    get_projid: Option<unsafe extern "C" fn(*mut inode, *mut __SlateRecord_projid_h_24) -> i32>,
    get_inode_usage: Option<unsafe extern "C" fn(*mut inode, *mut i64) -> i32>,
    get_next_id: Option<unsafe extern "C" fn(*mut super_block, *mut kqid) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_dqblk {
    d_fieldmask: i32,
    d_spc_hardlimit: u64,
    d_spc_softlimit: u64,
    d_ino_hardlimit: u64,
    d_ino_softlimit: u64,
    d_space: u64,
    d_ino_count: u64,
    d_ino_timer: i64,
    d_spc_timer: i64,
    d_ino_warns: i32,
    d_spc_warns: i32,
    d_rt_spc_hardlimit: u64,
    d_rt_spc_softlimit: u64,
    d_rt_space: u64,
    d_rt_spc_timer: i64,
    d_rt_spc_warns: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_type_state {
    flags: u32,
    spc_timelimit: u32,
    ino_timelimit: u32,
    rt_spc_timelimit: u32,
    spc_warnlimit: u32,
    ino_warnlimit: u32,
    rt_spc_warnlimit: u32,
    ino: u64,
    blocks: u64,
    nextents: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_state {
    s_incoredqs: u32,
    s_state: [qc_type_state; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_info {
    i_fieldmask: i32,
    i_flags: u32,
    i_spc_timelimit: u32,
    i_ino_timelimit: u32,
    i_rt_spc_timelimit: u32,
    i_spc_warnlimit: u32,
    i_ino_warnlimit: u32,
    i_rt_spc_warnlimit: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quotactl_ops {
    quota_on: Option<unsafe extern "C" fn(*mut super_block, i32, i32, *const path) -> i32>,
    quota_off: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    quota_enable: Option<unsafe extern "C" fn(*mut super_block, u32) -> i32>,
    quota_disable: Option<unsafe extern "C" fn(*mut super_block, u32) -> i32>,
    quota_sync: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    set_info: Option<unsafe extern "C" fn(*mut super_block, i32, *mut qc_info) -> i32>,
    get_dqblk: Option<unsafe extern "C" fn(*mut super_block, kqid, *mut qc_dqblk) -> i32>,
    get_nextdqblk: Option<unsafe extern "C" fn(*mut super_block, *mut kqid, *mut qc_dqblk) -> i32>,
    set_dqblk: Option<unsafe extern "C" fn(*mut super_block, kqid, *mut qc_dqblk) -> i32>,
    get_state: Option<unsafe extern "C" fn(*mut super_block, *mut qc_state) -> i32>,
    rm_xquota: Option<unsafe extern "C" fn(*mut super_block, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quota_format_type {
    qf_fmt_id: i32,
    qf_ops: *const quota_format_ops,
    qf_owner: *mut module,
    qf_next: *mut quota_format_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quota_info {
    flags: u32,
    dqio_sem: rw_semaphore,
    files: [*mut inode; 3],
    info: [mem_dqinfo; 3],
    ops: [*const quota_format_ops; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sb_writers {
    frozen: u16,
    freeze_kcount: i32,
    freeze_ucount: i32,
    freeze_owner: *const (),
    rw_sem: [percpu_rw_semaphore; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct super_operations {
    alloc_inode: Option<unsafe extern "C" fn(*mut super_block) -> *mut inode>,
    destroy_inode: Option<unsafe extern "C" fn(*mut inode)>,
    free_inode: Option<unsafe extern "C" fn(*mut inode)>,
    dirty_inode: Option<unsafe extern "C" fn(*mut inode, i32)>,
    write_inode: Option<unsafe extern "C" fn(*mut inode, *mut writeback_control) -> i32>,
    sync_inode_metadata: Option<unsafe extern "C" fn(*mut inode, *mut writeback_control) -> i32>,
    drop_inode: Option<unsafe extern "C" fn(*mut inode) -> i32>,
    evict_inode: Option<unsafe extern "C" fn(*mut inode)>,
    put_super: Option<unsafe extern "C" fn(*mut super_block)>,
    sync_fs: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    freeze_super: Option<unsafe extern "C" fn(*mut super_block, u32, *const ()) -> i32>,
    freeze_fs: Option<unsafe extern "C" fn(*mut super_block) -> i32>,
    thaw_super: Option<unsafe extern "C" fn(*mut super_block, u32, *const ()) -> i32>,
    unfreeze_fs: Option<unsafe extern "C" fn(*mut super_block) -> i32>,
    statfs: Option<unsafe extern "C" fn(*mut dentry, *mut kstatfs) -> i32>,
    umount_begin: Option<unsafe extern "C" fn(*mut super_block)>,
    show_options: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    show_devname: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    show_path: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    show_stats: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    quota_read: Option<unsafe extern "C" fn(*mut super_block, i32, *mut u8, u64, i64) -> i64>,
    quota_write: Option<unsafe extern "C" fn(*mut super_block, i32, *const u8, u64, i64) -> i64>,
    get_dquots: Option<unsafe extern "C" fn(*mut inode) -> *mut *mut dquot>,
    nr_cached_objects: Option<unsafe extern "C" fn(*mut super_block, *mut shrink_control) -> i64>,
    free_cached_objects: Option<unsafe extern "C" fn(*mut super_block, *mut shrink_control) -> i64>,
    remove_bdev: Option<unsafe extern "C" fn(*mut super_block, *mut block_device) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut super_block)>,
    report_error: Option<unsafe extern "C" fn(*const fserror_event)>,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct super_block {
    s_list: list_head,
    s_dev: u32,
    s_super_dev: *mut super_dev,
    s_blocksize_bits: u8,
    s_blocksize: u64,
    s_maxbytes: i64,
    s_type: *mut file_system_type,
    s_op: *const super_operations,
    dq_op: *const dquot_operations,
    s_qcop: *const quotactl_ops,
    s_export_op: *const export_operations,
    s_flags: u64,
    s_iflags: u64,
    s_magic: u64,
    s_root: *mut dentry,
    s_umount: rw_semaphore,
    s_passive: refcount_struct,
    s_active: __SlateRecord_types_h_190,
    s_security: *mut (),
    s_xattr: *const *const xattr_handler,
    s_roots: hlist_head,
    s_roots_lock: spinlock,
    s_mounts: *mut mount,
    s_bdev: *mut block_device,
    s_bdev_file: *mut file,
    s_bdi: *mut backing_dev_info,
    s_mtd: *mut mtd_info,
    s_instances: hlist_node,
    s_quota_types: u32,
    s_dquot: quota_info,
    s_writers: sb_writers,
    s_fs_info: *mut (),
    s_time_gran: u32,
    s_time_min: i64,
    s_time_max: i64,
    s_fsnotify_mask: u32,
    s_fsnotify_info: *mut fsnotify_sb_info,
    s_id: [u8; 32],
    s_uuid: __SlateRecord_uuid_h_21,
    s_uuid_len: u8,
    s_sysfs_name: [u8; 37],
    s_max_links: u32,
    s_d_flags: u32,
    s_vfs_rename_mutex: mutex,
    s_subtype: *const u8,
    __s_d_op: *const dentry_operations,
    s_shrink: *mut shrinker,
    s_remove_count: __SlateRecord_types_h_197,
    s_readonly_remount: i32,
    s_wb_err: u32,
    s_dio_done_wq: *mut workqueue_struct,
    s_pins: hlist_head,
    s_user_ns: *mut user_namespace,
    s_dentry_lru: list_lru,
    s_inode_lru: list_lru,
    rcu: callback_head,
    destroy_work: work_struct,
    s_sync_lock: mutex,
    s_stack_depth: i32,
    __slate_pad_59: [u8; 12],
    s_inode_list_lock: spinlock,
    s_inodes: list_head,
    s_inode_wblist_lock: spinlock,
    s_inodes_wb: list_head,
    s_min_writeback_pages: i64,
    s_pending_errors: refcount_struct,
}


// sb->s_flags.  Note that these mirror the equivalent MS_* flags where
// represented in both.
// Mount read-only
// Update the on-disk [acm]times lazily
#[repr(C)]
#[derive(Clone, Copy)]
struct lockref {
    __slate_anon_0: __SlateRecord_lockref_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qstr {
    __slate_anon_0: __SlateRecord_dcache_h_50,
    name: *const u8,
}


// 192 bytes
#[repr(C)]
#[derive(Clone, Copy)]
union shortname_store {
    string: [u8; 40],
    words: [u64; 5],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dentry {
    d_flags: u32,
    d_seq: seqcount_spinlock,
    d_hash: hlist_bl_node,
    d_parent: *mut dentry,
    __slate_anon_4: __SlateRecord_dcache_h_99,
    d_inode: *mut inode,
    d_shortname: shortname_store,
    d_op: *const dentry_operations,
    d_sb: *mut super_block,
    d_time: u64,
    d_fsdata: *mut (),
    d_lockref: lockref,
    d_lru: list_head,
    d_sib: hlist_node,
    d_children: hlist_head,
    __slate_anon_15: __SlateRecord_dcache_h_126,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct dentry_operations {
    d_revalidate: Option<unsafe extern "C" fn(*mut inode, *const qstr, *mut dentry, u32) -> i32>,
    d_weak_revalidate: Option<unsafe extern "C" fn(*mut dentry, u32) -> i32>,
    d_hash: Option<unsafe extern "C" fn(*const dentry, *mut qstr) -> i32>,
    d_compare: Option<unsafe extern "C" fn(*const dentry, u32, *const u8, *const qstr) -> i32>,
    d_delete: Option<unsafe extern "C" fn(*const dentry) -> i32>,
    d_init: Option<unsafe extern "C" fn(*mut dentry) -> i32>,
    d_release: Option<unsafe extern "C" fn(*mut dentry)>,
    d_prune: Option<unsafe extern "C" fn(*mut dentry)>,
    d_iput: Option<unsafe extern "C" fn(*mut dentry, *mut inode)>,
    d_dname: Option<unsafe extern "C" fn(*mut dentry, *mut u8, i32) -> *mut u8>,
    d_automount: Option<unsafe extern "C" fn(*mut path) -> *mut vfsmount>,
    d_manage: Option<unsafe extern "C" fn(*const path, bool) -> i32>,
    d_real: Option<unsafe extern "C" fn(*mut dentry, u32) -> *mut dentry>,
    d_unalias_trylock: Option<unsafe extern "C" fn(*const dentry) -> bool>,
    d_unalias_unlock: Option<unsafe extern "C" fn(*const dentry)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct path {
    mnt: *mut vfsmount,
    dentry: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_context {
    refcount: __SlateRecord_types_h_197,
    active_ref: __SlateRecord_types_h_190,
    ioprio: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vfsmount {
    mnt_root: *mut dentry,
    mnt_sb: *mut super_block,
    mnt_flags: i32,
    mnt_idmap: *mut mnt_idmap,
}


fn mnt_idmap(mnt: &vfsmount) -> *mut mnt_idmap {
// Pairs with smp_store_release() in do_idmap_mount().
let mut __v0: *mut mnt_idmap;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_416() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(mnt as *const vfsmount)).mnt_idmap) as *const *mut mnt_idmap) };
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mnt_idmapping_h_17 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mnt_idmapping_h_21 {
    val: u32,
}


// vfs{g,u}ids are created from k{g,u}ids.
// We don't allow them to be created from regular {u,g}id.
// Allow a vfs{g,u}id to be used as a k{g,u}id where we want to compare
// whether the mapped value is identical to value of a k{g,u}id.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_file_ref_h_52 {
    refcnt: __SlateRecord_types_h_197,
}


// File is embedded in backing_file object
// File is capable of returning -EAGAIN if I/O will block
// The two FMODE_NONOTIFY* define which fsnotify events should not be generated
// for an open file. These are the possible values of
// (f->f_mode & FMODE_FSNOTIFY_MASK) and their meaning:
//
// FMODE_NONOTIFY - suppress all (incl. non-permission) events.
// FMODE_NONOTIFY_PERM - suppress permission (incl. pre-content) events.
// FMODE_NONOTIFY | FMODE_NONOTIFY_PERM - suppress only FAN_ACCESS_PERM.
// Whiteout is represented by a char device.  The following constants define the
// mode and device number to use.
#[repr(C)]
#[derive(Clone, Copy)]
struct iattr {
    ia_valid: u32,
    ia_mode: u16,
    __slate_anon_2: __SlateRecord_fs_h_272,
    __slate_anon_3: __SlateRecord_fs_h_276,
    ia_size: i64,
    ia_atime: timespec64,
    ia_mtime: timespec64,
    ia_ctime: timespec64,
    ia_file: *mut file,
}


// non-RWF related bits - start at 16
#[repr(C)]
#[derive(Clone, Copy)]
struct kiocb {
    ki_filp: *mut file,
    ki_pos: i64,
    ki_complete: Option<unsafe extern "C" fn(*mut kiocb, i64)>,
    private: *mut (),
    ki_flags: i32,
    ki_ioprio: u16,
    ki_write_stream: u8,
    ki_waitq: *mut wait_page_queue,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct address_space_operations {
    read_folio: Option<unsafe extern "C" fn(*mut file, *mut folio) -> i32>,
    writepages: Option<unsafe extern "C" fn(*mut address_space, *mut writeback_control) -> i32>,
    dirty_folio: Option<unsafe extern "C" fn(*mut address_space, *mut folio) -> bool>,
    readahead: Option<unsafe extern "C" fn(*mut readahead_control)>,
    write_begin: Option<unsafe extern "C" fn(*const kiocb, *mut address_space, i64, u32, *mut *mut folio, *mut *mut ()) -> i32>,
    write_end: Option<unsafe extern "C" fn(*const kiocb, *mut address_space, i64, u32, u32, *mut folio, *mut ()) -> i32>,
    bmap: Option<unsafe extern "C" fn(*mut address_space, u64) -> u64>,
    invalidate_folio: Option<unsafe extern "C" fn(*mut folio, u64, u64)>,
    release_folio: Option<unsafe extern "C" fn(*mut folio, u32) -> bool>,
    free_folio: Option<unsafe extern "C" fn(*mut folio)>,
    direct_IO: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    migrate_folio: Option<unsafe extern "C" fn(*mut address_space, *mut folio, *mut folio, u32) -> i32>,
    launder_folio: Option<unsafe extern "C" fn(*mut folio) -> i32>,
    is_partially_uptodate: Option<unsafe extern "C" fn(*mut folio, u64, u64) -> bool>,
    is_dirty_writeback: Option<unsafe extern "C" fn(*mut folio, *mut bool, *mut bool)>,
    error_remove_folio: Option<unsafe extern "C" fn(*mut address_space, *mut folio) -> i32>,
    swap_activate: Option<unsafe extern "C" fn(*mut swap_info_struct, *mut file, *mut u64) -> i32>,
    swap_deactivate: Option<unsafe extern "C" fn(*mut file)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct address_space {
    host: *mut inode,
    i_pages: xarray,
    invalidate_lock: rw_semaphore,
    gfp_mask: u32,
    i_mmap_writable: __SlateRecord_types_h_190,
    i_mmap: rb_root_cached,
    nrpages: u64,
    writeback_index: u64,
    a_ops: *const address_space_operations,
    flags: u64,
    wb_err: u32,
    i_private_lock: spinlock,
    i_mmap_rwsem: rw_semaphore,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct inode_state_flags {
    __state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct inode {
    i_mode: u16,
    i_opflags: u16,
    i_flags: u32,
    i_acl: *mut posix_acl,
    i_default_acl: *mut posix_acl,
    i_uid: __SlateRecord_uidgid_types_h_9,
    i_gid: __SlateRecord_uidgid_types_h_13,
    i_op: *const inode_operations,
    i_sb: *mut super_block,
    i_mapping: *mut address_space,
    i_security: *mut (),
    i_ino: u64,
    __slate_anon_12: __SlateRecord_fs_h_790,
    i_rdev: u32,
    i_size: i64,
    i_atime_sec: i64,
    i_mtime_sec: i64,
    i_ctime_sec: i64,
    i_atime_nsec: u32,
    i_mtime_nsec: u32,
    i_ctime_nsec: u32,
    i_generation: u32,
    i_lock: spinlock,
    i_bytes: u16,
    i_blkbits: u8,
    i_write_hint: u8,
    i_blocks: u64,
    i_state: inode_state_flags,
    i_rwsem: rw_semaphore,
    dirtied_when: u64,
    dirtied_time_when: u64,
    i_hash: hlist_node,
    i_io_list: list_head,
    i_lru: list_head,
    i_sb_list: list_head,
    i_wb_list: list_head,
    __slate_anon_36: __SlateRecord_fs_h_834,
    i_version: __SlateRecord_types_h_197,
    i_sequence: __SlateRecord_types_h_197,
    i_count: __SlateRecord_types_h_190,
    i_dio_count: __SlateRecord_types_h_190,
    i_writecount: __SlateRecord_types_h_190,
    i_readcount: __SlateRecord_types_h_190,
    __slate_anon_43: __SlateRecord_fs_h_846,
    i_flctx: *mut file_lock_context,
    i_data: address_space,
    __slate_anon_46: __SlateRecord_fs_h_852,
    __slate_anon_47: __SlateRecord_fs_h_856,
    i_fsnotify_mask: u32,
    i_fsnotify_marks: *mut fsnotify_mark_connector,
    i_private: *mut (),
}


// Get bit address from inode->i_state to use with wait_var_event()
// infrastructre.
#[repr(C)]
#[derive(Clone, Copy)]
struct fown_struct {
    file: *mut file,
    lock: rwlock,
    pid: *mut pid,
    pid_type: u32,
    uid: __SlateRecord_uidgid_types_h_9,
    euid: __SlateRecord_uidgid_types_h_9,
    signum: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_ra_state {
    start: u64,
    size: u32,
    async_size: u32,
    ra_pages: u32,
    order: u16,
    mmap_miss: u16,
    prev_pos: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file {
    f_lock: spinlock,
    f_mode: u32,
    f_op: *const file_operations,
    f_mapping: *mut address_space,
    private_data: *mut (),
    f_inode: *mut inode,
    f_flags: u32,
    f_iocb_flags: u32,
    f_cred: *const cred,
    f_owner: *mut fown_struct,
    __slate_anon_10: __SlateRecord_fs_h_1267,
    __slate_anon_11: __SlateRecord_fs_h_1271,
    f_pos: i64,
    f_security: *mut (),
    f_wb_err: u32,
    f_sb_err: u32,
    f_ep: *mut hlist_head,
    __slate_anon_17: __SlateRecord_fs_h_1287,
    f_ref: __SlateRecord_file_ref_h_52,
}


// Multigrain timestamps
//
// Conditionally use fine-grained ctime and mtime timestamps when there
// are users actively observing them via getattr. The primary use-case
// for this is NFS clients that use the ctime to distinguish between
// different states of the file, and that are often fooled by multiple
// operations that occur in the same coarse-grained timer tick.
#[repr(C)]
#[derive(Clone, Copy)]
struct dir_context {
    actor: Option<unsafe extern "C" fn(*mut dir_context, *const u8, i32, i64, u64, u32) -> bool>,
    pos: i64,
    count: i32,
    dt_flags_mask: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_operations {
    owner: *mut module,
    fop_flags: u32,
    llseek: Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>,
    read: Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut file, *const u8, u64, *mut i64) -> i64>,
    read_iter: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    write_iter: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    iopoll: Option<unsafe extern "C" fn(*mut kiocb, *mut io_comp_batch, u32) -> i32>,
    iterate_shared: Option<unsafe extern "C" fn(*mut file, *mut dir_context) -> i32>,
    poll: Option<unsafe extern "C" fn(*mut file, *mut poll_table_struct) -> u32>,
    unlocked_ioctl: Option<unsafe extern "C" fn(*mut file, u32, u64) -> i64>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut file, u32, u64) -> i64>,
    mmap: Option<unsafe extern "C" fn(*mut file, *mut vm_area_struct) -> i32>,
    open: Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>,
    flush: Option<unsafe extern "C" fn(*mut file, *mut ()) -> i32>,
    release: Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>,
    fsync: Option<unsafe extern "C" fn(*mut file, i64, i64, i32) -> i32>,
    fasync: Option<unsafe extern "C" fn(i32, *mut file, i32) -> i32>,
    lock: Option<unsafe extern "C" fn(*mut file, i32, *mut file_lock) -> i32>,
    get_unmapped_area: Option<unsafe extern "C" fn(*mut file, u64, u64, u64, u64) -> u64>,
    check_flags: Option<unsafe extern "C" fn(i32) -> i32>,
    flock: Option<unsafe extern "C" fn(*mut file, i32, *mut file_lock) -> i32>,
    splice_write: Option<unsafe extern "C" fn(*mut pipe_inode_info, *mut file, *mut i64, u64, u32) -> i64>,
    splice_read: Option<unsafe extern "C" fn(*mut file, *mut i64, *mut pipe_inode_info, u64, u32) -> i64>,
    splice_eof: Option<unsafe extern "C" fn(*mut file)>,
    setlease: Option<unsafe extern "C" fn(*mut file, i32, *mut *mut file_lease, *mut *mut ()) -> i32>,
    fallocate: Option<unsafe extern "C" fn(*mut file, i32, i64, i64) -> i64>,
    show_fdinfo: Option<unsafe extern "C" fn(*mut seq_file, *mut file)>,
    copy_file_range: Option<unsafe extern "C" fn(*mut file, i64, *mut file, i64, u64, u32) -> i64>,
    remap_file_range: Option<unsafe extern "C" fn(*mut file, i64, *mut file, i64, i64, u32) -> i64>,
    fadvise: Option<unsafe extern "C" fn(*mut file, i64, i64, i32) -> i32>,
    uring_cmd: Option<unsafe extern "C" fn(*mut io_uring_cmd, u32) -> i32>,
    uring_cmd_iopoll: Option<unsafe extern "C" fn(*mut io_uring_cmd, *mut io_comp_batch, u32) -> i32>,
    mmap_prepare: Option<unsafe extern "C" fn(*mut vm_area_desc) -> i32>,
}


// File system supports uncached read/write buffered IO
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct inode_operations {
    lookup: Option<unsafe extern "C" fn(*mut inode, *mut dentry, u32) -> *mut dentry>,
    get_link: Option<unsafe extern "C" fn(*mut dentry, *mut inode, *mut delayed_call) -> *const u8>,
    permission: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, i32) -> i32>,
    get_inode_acl: Option<unsafe extern "C" fn(*mut inode, i32, bool) -> *mut posix_acl>,
    readlink: Option<unsafe extern "C" fn(*mut dentry, *mut u8, i32) -> i32>,
    create: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, u16) -> i32>,
    link: Option<unsafe extern "C" fn(*mut dentry, *mut inode, *mut dentry) -> i32>,
    unlink: Option<unsafe extern "C" fn(*mut inode, *mut dentry) -> i32>,
    symlink: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, *const u8) -> i32>,
    mkdir: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, u16) -> *mut dentry>,
    rmdir: Option<unsafe extern "C" fn(*mut inode, *mut dentry) -> i32>,
    mknod: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, u16, u32) -> i32>,
    rename: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, *mut inode, *mut dentry, u32) -> i32>,
    setattr: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut iattr) -> i32>,
    getattr: Option<unsafe extern "C" fn(*mut mnt_idmap, *const path, *mut kstat, u32, u32) -> i32>,
    listxattr: Option<unsafe extern "C" fn(*mut dentry, *mut u8, u64) -> i64>,
    fiemap: Option<unsafe extern "C" fn(*mut inode, *mut fiemap_extent_info, u64, u64) -> i32>,
    update_time: Option<unsafe extern "C" fn(*mut inode, u32, u32) -> i32>,
    sync_lazytime: Option<unsafe extern "C" fn(*mut inode)>,
    atomic_open: Option<unsafe extern "C" fn(*mut inode, *mut dentry, *mut file, u32, u16) -> i32>,
    tmpfile: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut file, u16) -> i32>,
    get_acl: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, i32) -> *mut posix_acl>,
    set_acl: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut posix_acl, i32) -> i32>,
    fileattr_set: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut file_kattr) -> i32>,
    fileattr_get: Option<unsafe extern "C" fn(*mut dentry, *mut file_kattr) -> i32>,
    get_offset_ctx: Option<unsafe extern "C" fn(*mut inode) -> *mut offset_ctx>,
}


// no suid or xattr security attributes
// Note that nosuid etc flags are inode-specific: setting some file-system
// flags just means all the inodes inherit those flags by default. It might be
// possible to override it selectively if you really wanted to with some
// ioctl() that is not currently implemented.
//
// Exception: SB_RDONLY is always applied to the entire file system.
//
// Unfortunately, it is possible to change a filesystems flags with it mounted
// with files in use.  This means that all of the inodes will not have their
// i_flags updated.  Hence, i_flags no longer inherit the superblock mount
// flags, so these have to be checked separately. -- rmk@arm.uk.linux.org
#[repr(C)]
#[derive(Clone, Copy)]
struct file_system_type {
    name: *const u8,
    fs_flags: i32,
    init_fs_context: Option<unsafe extern "C" fn(*mut fs_context) -> i32>,
    parameters: *const fs_parameter_spec,
    kill_sb: Option<unsafe extern "C" fn(*mut super_block)>,
    owner: *mut module,
    list: hlist_node,
    fs_supers: hlist_head,
    s_lock_key: lock_class_key,
    s_umount_key: lock_class_key,
    s_vfs_rename_key: lock_class_key,
    s_writers_key: [lock_class_key; 3],
    i_lock_key: lock_class_key,
    i_mutex_key: lock_class_key,
    invalidate_lock_key: lock_class_key,
    i_mutex_dir_key: lock_class_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct offset_ctx {
    mt: maple_tree,
    next_offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list {
    next: u32,
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_snd_timer_instance_new: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timer_instance_free: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timeri_timer_get: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timeri_timer_put: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timer_open: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timer_close: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timer_resolution: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timer_start: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timer_stop: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timer_continue: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timer_pause: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timer_interrupt: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timer_new: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timer_notify: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timer_global_new: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timer_global_free: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_timer_global_register: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_snd_timer__585_2603_alsa_timer_init6:\t\t\t\n.long\t{17} - .\t\n.previous\t\t\t\t\t\n", sym snd_timer_instance_new, sym snd_timer_instance_free, sym snd_timeri_timer_get, sym snd_timeri_timer_put, sym snd_timer_open, sym snd_timer_close, sym snd_timer_resolution, sym snd_timer_start, sym snd_timer_stop, sym snd_timer_continue, sym snd_timer_pause, sym snd_timer_interrupt, sym snd_timer_new, sym snd_timer_notify, sym snd_timer_global_new, sym snd_timer_global_free, sym snd_timer_global_register, sym alsa_timer_init, options(att_syntax));

#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list_head {
    list: compat_robust_list,
    futex_offset: i32,
    list_op_pending: u32,
}


fn compat_ptr(uptr: u32) -> *mut () {
(uptr as u64) as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vdso_image {
    data: *mut (),
    size: u64,
    alt: u64,
    alt_len: u64,
    extable_base: u64,
    extable_len: u64,
    extable: *const (),
    sym___kernel_sigreturn: i64,
    sym___kernel_rt_sigreturn: i64,
    sym___kernel_vsyscall: i64,
    sym_int80_landing_pad: i64,
    sym_vdso32_sigreturn_landing_pad: i64,
    sym_vdso32_rt_sigreturn_landing_pad: i64,
    sym___futex_list64_try_unlock_cs_start: i64,
    sym___futex_list64_try_unlock_cs_end: i64,
    sym___futex_list32_try_unlock_cs_start: i64,
    sym___futex_list32_try_unlock_cs_end: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct elf64_sym {
    st_name: u32,
    st_info: u8,
    st_other: u8,
    st_shndx: u16,
    st_value: u64,
    st_size: u64,
}


// One for each parameter, describing how to use it.  Some files do
// multiple of these per line, so can't just use MODULE_INFO.
#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_param_ops {
    flags: u32,
    set: Option<unsafe extern "C" fn(*const u8, *const kernel_param) -> i32>,
    get: Option<unsafe extern "C" fn(*mut u8, *const kernel_param) -> i32>,
    free: Option<unsafe extern "C" fn(*mut ())>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_param {
    name: *const u8,
    r#mod: *mut module,
    ops: *const kernel_param_ops,
    perm: u16,
    level: i8,
    flags: u8,
    __slate_anon_6: __SlateRecord_moduleparam_h_89,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kparam_string {
    maxlen: u32,
    string: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kparam_array {
    max: u32,
    elemsize: u32,
    num: *mut u32,
    ops: *const kernel_param_ops,
    elem: *mut (),
}


// module_param - typesafe helper for a module/cmdline parameter
// @name: the variable to alter, and exposed parameter name.
// @type: the type of the parameter
// @perm: visibility in sysfs.
//
// @name becomes the module parameter, or (prefixed by KBUILD_MODNAME and a
// ".") the kernel commandline parameter.  Note that - is changed to _, so
// the user can use "foo-bar=1" even for variable "foo_bar".
//
// @perm is 0 if the variable is not to appear in sysfs, or 0444
// for world-readable, 0644 for root-writable, etc.  Note that if it
// is writable, you may need to use kernel_param_lock() around
// accesses (esp. charp, which can be kfreed when it changes).
//
// The @type is simply pasted to refer to a param_ops_##type and a
// param_check_##type: for convenience many standard types are provided but
// you can create your own by defining those variables.
//
// Standard types are:
// 	byte, hexint, short, ushort, int, uint, long, ulong
// 	charp: a character pointer
// 	bool: a bool, values 0/1, y/n, Y/N.
// 	invbool: the above, only sense-reversed (N = true).
// module_param_named - typesafe helper for a renamed module/cmdline parameter
// @name: a valid C identifier which is the parameter name.
// @value: the actual lvalue to alter.
// @type: the type of the parameter
// @perm: visibility in sysfs.
//
// Usually it's a good idea to have variable names and user-exposed names the
// same, but that's harder if the variable must be non-static or is inside a
// structure.  This allows exposure under a different name.
// module_param_cb - general callback for a module/cmdline parameter
// @name: a valid C identifier which is the parameter name.
// @ops: the set & get operations for this parameter.
// @arg: args for @ops
// @perm: visibility in sysfs.
//
// The ops can have NULL set or get functions.
// This is the fundamental function for registering boot/module parameters.
// All the helper functions
//
// The macros to do compile-time type checking stolen from Jakub
// Jelinek, who IIRC came up with this idea for the 2.4 module init code.
#[repr(C)]
#[derive(Clone, Copy)]
struct latch_tree_node {
    node: [rb_node; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct its_array {
    pages: *mut *mut (),
    num: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mod_arch_specific {
    num_orcs: u32,
    orc_unwind_ip: *mut i32,
    orc_unwind: *mut orc_entry,
    its_pages: its_array,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_kobject {
    kobj: kobject,
    r#mod: *mut module,
    drivers_dir: *mut kobject,
    mp: *mut module_param_attrs,
    kobj_completion: *mut completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_attribute {
    attr: attribute,
    show: Option<unsafe extern "C" fn(*const module_attribute, *mut module_kobject, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*const module_attribute, *mut module_kobject, *const u8, u64) -> i64>,
    setup: Option<unsafe extern "C" fn(*mut module, *const u8)>,
    test: Option<unsafe extern "C" fn(*mut module) -> i32>,
    free: Option<unsafe extern "C" fn(*mut module)>,
}


// module_init() - driver initialization entry point
// @x: function to be run at kernel boot time or module insertion
//
// module_init() will either be called during do_initcalls() (if
// builtin) or at module insertion time (if a module).  There can only
// be one per module.
// module_exit() - driver exit entry point
// @x: function to be run when driver is removed
//
// module_exit() will wrap the driver clean-up code
// with cleanup_module() when used with rmmod when
// the driver is a module.  If the driver is statically
// compiled into the kernel, module_exit() has no effect.
// There can only be one per module.
// For userspace: you can also call me...
// The following license idents are currently accepted as indicating free
// software modules
//
// 	"GPL"				[GNU Public License v2]
// 	"GPL v2"			[GNU Public License v2]
// 	"GPL and additional rights"	[GNU Public License v2 rights and more]
// 	"Dual BSD/GPL"			[GNU Public License v2
// 					 or BSD license choice]
// 	"Dual MIT/GPL"			[GNU Public License v2
// 					 or MIT license choice]
// 	"Dual MPL/GPL"			[GNU Public License v2
// 					 or Mozilla license choice]
//
// The following other idents are available
//
// 	"Proprietary"			[Non free products]
//
// Both "GPL v2" and "GPL" (the latter also in dual licensed strings) are
// merely stating that the module is licensed under the GPL v2, but are not
// telling whether "GPL v2 only" or "GPL v2 or later". The reason why there
// are two variants is a historic and failed attempt to convey more
// information in the MODULE_LICENSE string. For module loading the
// "only/or later" distinction is completely irrelevant and does neither
// replace the proper license identifiers in the corresponding source file
// nor amends them in any way. The sole purpose is to make the
// 'Proprietary' flagging work and to refuse to bind symbols which are
// exported with EXPORT_SYMBOL_GPL when a non free module is loaded.
//
// In the same way "BSD" is not a clear license information. It merely
// states, that the module is licensed under one of the compatible BSD
// license variants. The detailed and correct license information is again
// to be found in the corresponding source files.
//
// There are dual licensed components, but when running with Linux it is the
// GPL that is relevant so this is a non issue. Similarly LGPL linked with GPL
// is a GPL combined work.
//
// This exists for several reasons
// 1.	So modinfo can show license info for users wanting to vet their setup
// 	is free
// 2.	So the community can ignore bug reports including proprietary modules
// 3.	So vendors can do likewise based on their own policies
// Author(s), use "Name <email>" or just "Name", for multiple
// authors use multiple MODULE_AUTHOR() statements/lines.
// What your module does.
#[repr(C)]
#[derive(Clone, Copy)]
struct mod_tree_node {
    r#mod: *mut module,
    node: latch_tree_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_memory {
    base: *mut (),
    is_rox: bool,
    size: u32,
    mtn: mod_tree_node,
}


// Only touch one cacheline for common rbtree-for-core-layout case.
#[repr(C)]
#[derive(Clone, Copy)]
struct mod_kallsyms {
    symtab: *mut elf64_sym,
    num_symtab: u32,
    strtab: *mut u8,
    typetab: *mut u8,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct module {
    state: u32,
    list: list_head,
    name: [u8; 56],
    mkobj: module_kobject,
    modinfo_attrs: *mut module_attribute,
    version: *const u8,
    srcversion: *const u8,
    imported_namespaces: *const u8,
    holders_dir: *mut kobject,
    syms: *const kernel_symbol,
    crcs: *const u32,
    flagstab: *const u8,
    num_syms: u32,
    param_lock: mutex,
    kp: *mut kernel_param,
    num_kp: u32,
    using_gplonly_symbols: bool,
    async_probe_requested: bool,
    num_exentries: u32,
    extable: *mut exception_table_entry,
    init: Option<unsafe extern "C" fn() -> i32>,
    __slate_pad_21: [u8; 8],
    mem: [module_memory; 7],
    arch: mod_arch_specific,
    taints: u64,
    num_bugs: u32,
    bug_list: list_head,
    bug_table: *mut bug_entry,
    kallsyms: *mut mod_kallsyms,
    core_kallsyms: mod_kallsyms,
    sect_attrs: *mut module_sect_attrs,
    notes_attrs: *mut module_notes_attrs,
    percpu: *mut (),
    percpu_size: u32,
    noinstr_text_start: *mut (),
    noinstr_text_size: u32,
    num_tracepoints: u32,
    tracepoints_ptrs: *const i32,
    num_srcu_structs: u32,
    srcu_struct_ptrs: *mut *mut srcu_struct,
    jump_entries: *mut jump_entry,
    num_jump_entries: u32,
    num_trace_bprintk_fmt: u32,
    trace_bprintk_fmt_start: *mut *const u8,
    trace_events: *mut *mut trace_event_call,
    num_trace_events: u32,
    trace_evals: *mut *mut trace_eval_map,
    num_trace_evals: u32,
    kprobes_text_start: *mut (),
    kprobes_text_size: u32,
    kprobe_blacklist: *mut u64,
    num_kprobe_blacklist: u32,
    num_static_call_sites: i32,
    static_call_sites: *mut static_call_site,
    source_list: list_head,
    target_list: list_head,
    exit: Option<unsafe extern "C" fn()>,
    refcnt: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_id {
    id: [u8; 16],
    driver_data: u64,
    cls: u32,
    cls_msk: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct of_device_id {
    name: [u8; 32],
    r#type: [u8; 32],
    compatible: [u8; 128],
    data: *const (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_driver {
    name: *const u8,
    bus: *const bus_type,
    owner: *mut module,
    mod_name: *const u8,
    suppress_bind_attrs: bool,
    probe_type: u32,
    of_match_table: *const of_device_id,
    acpi_match_table: *const acpi_device_id,
    probe: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync_state: Option<unsafe extern "C" fn(*mut device)>,
    remove: Option<unsafe extern "C" fn(*mut device) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut device)>,
    suspend: Option<unsafe extern "C" fn(*mut device, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    pm: *const dev_pm_ops,
    coredump: Option<unsafe extern "C" fn(*mut device)>,
    p: *mut driver_private,
    p_cb: __SlateRecord_driver_h_126,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_archdata {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_type {
    name: *const u8,
    groups: *mut *const attribute_group,
    uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    devnode: Option<unsafe extern "C" fn(*const device, *mut u16, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13) -> *mut u8>,
    release: Option<unsafe extern "C" fn(*mut device)>,
    pm: *const dev_pm_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_dma_parameters {
    max_segment_size: u32,
    min_align_mask: u32,
    segment_boundary_mask: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_links_info {
    suppliers: list_head,
    consumers: list_head,
    defer_sync: list_head,
    status: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_msi_info {
    domain: *mut irq_domain,
    data: *mut msi_device_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_physical_location {
    panel: u32,
    vertical_position: u32,
    horizontal_position: u32,
    dock: bool,
    lid: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device {
    kobj: kobject,
    parent: *mut device,
    p: *mut device_private,
    init_name: *const u8,
    r#type: *const device_type,
    bus: *const bus_type,
    driver: *mut device_driver,
    platform_data: *mut (),
    driver_data: *mut (),
    driver_override: __SlateRecord_device_h_719,
    mutex: mutex,
    links: dev_links_info,
    power: dev_pm_info,
    pm_domain: *mut dev_pm_domain,
    msi: dev_msi_info,
    dma_mask: *mut u64,
    coherent_dma_mask: u64,
    bus_dma_limit: u64,
    dma_range_map: *const bus_dma_region,
    dma_parms: *mut device_dma_parameters,
    dma_pools: list_head,
    dma_io_tlb_mem: *mut io_tlb_mem,
    archdata: dev_archdata,
    of_node: *mut device_node,
    fwnode: *mut fwnode_handle,
    numa_node: i32,
    devt: u32,
    id: u32,
    devres_lock: spinlock,
    devres_head: list_head,
    class: *const class,
    groups: *mut *const attribute_group,
    release: Option<unsafe extern "C" fn(*mut device)>,
    iommu_group: *mut iommu_group,
    iommu: *mut dev_iommu,
    physical_location: *mut device_physical_location,
    removable: u32,
    flags: [u64; 1],
}


// Create alias, so I can be autoloaded.
#[repr(C)]
#[derive(Clone, Copy)]
struct wakeup_source {
    name: *const u8,
    id: i32,
    entry: list_head,
    lock: spinlock,
    wakeirq: *mut wake_irq,
    timer: timer_list,
    timer_expires: u64,
    total_time: i64,
    max_time: i64,
    last_time: i64,
    start_prevent_time: i64,
    prevent_sleep_time: i64,
    event_count: u64,
    active_count: u64,
    relax_count: u64,
    expire_count: u64,
    wakeup_count: u64,
    dev: *mut device,
    __slate_bits_0: __slate_bits::__SlateBits485U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_device_ops {
    dev_free: Option<unsafe extern "C" fn(*mut snd_device) -> i32>,
    dev_register: Option<unsafe extern "C" fn(*mut snd_device) -> i32>,
    dev_disconnect: Option<unsafe extern "C" fn(*mut snd_device) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_device {
    list: list_head,
    card: *mut snd_card,
    state: u32,
    r#type: u32,
    device_data: *mut (),
    ops: *const snd_device_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_refcount {
    count: __SlateRecord_types_h_190,
    waiter: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_card {
    number: i32,
    id: [u8; 16],
    driver: [u8; 16],
    shortname: [u8; 32],
    longname: [u8; 80],
    irq_descr: [u8; 32],
    mixername: [u8; 80],
    components: *mut u8,
    components_alloc_size: u32,
    module: *mut module,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_card)>,
    devices: list_head,
    ctl_dev: *mut device,
    last_numid: u32,
    controls_rwsem: rw_semaphore,
    controls_rwlock: rwlock,
    controls_count: i32,
    user_ctl_alloc_size: u64,
    controls: list_head,
    ctl_files: list_head,
    ctl_numids: xarray,
    ctl_hash: xarray,
    ctl_hash_collision: bool,
    proc_root: *mut snd_info_entry,
    proc_root_link: *mut proc_dir_entry,
    files_list: list_head,
    s_f_ops: *mut snd_shutdown_f_ops,
    files_lock: spinlock,
    shutdown: i32,
    release_completion: *mut completion,
    dev: *mut device,
    card_dev: device,
    dev_groups: [*const attribute_group; 4],
    registered: bool,
    managed: bool,
    releasing: bool,
    sync_irq: i32,
    remove_sleep: wait_queue_head,
    total_pcm_alloc_bytes: u64,
    memory_mutex: mutex,
    power_state: u32,
    power_sleep: wait_queue_head,
    power_ref: snd_refcount,
    private_data_area: [u8; 0],
}


// global timers (device member)
// info flags
// cannot be controlled
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_id {
    dev_class: i32,
    dev_sclass: i32,
    card: i32,
    device: i32,
    subdevice: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_ginfo {
    tid: snd_timer_id,
    flags: u32,
    card: i32,
    id: [u8; 64],
    name: [u8; 80],
    reserved0: u64,
    resolution: u64,
    resolution_min: u64,
    resolution_max: u64,
    clients: u32,
    reserved: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_gparams {
    tid: snd_timer_id,
    period_num: u64,
    period_den: u64,
    reserved: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_gstatus {
    tid: snd_timer_id,
    resolution: u64,
    resolution_num: u64,
    resolution_den: u64,
    reserved: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_select {
    id: snd_timer_id,
    reserved: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_info {
    flags: u32,
    card: i32,
    id: [u8; 64],
    name: [u8; 80],
    reserved0: u64,
    resolution: u64,
    reserved: [u8; 64],
}


// auto start, otherwise one-shot
// exclusive use, precise start/stop/pause/continue
// write early event to the poll queue
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_params {
    flags: u32,
    ticks: u32,
    queue_size: u32,
    reserved0: u32,
    filter: u32,
    reserved: [u8; 60],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_uinfo {
    resolution: u64,
    fd: i32,
    id: u32,
    reserved: [u8; 16],
}


// The following four ioctls are changed since 1.0.9 due to confliction
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_read {
    resolution: u32,
    ticks: u32,
}


// virtual mixer device
// call stop before start
// only slave timer (variable resolution)
// first tick can be incomplete
// timer is called from work
// auto restart
// fast callback (do not use work)
// timer callback is active
// exclusive owner - no more instances
// write early event to the poll queue
// need reschedule
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_hardware {
    flags: u32,
    resolution: u64,
    resolution_min: u64,
    resolution_max: u64,
    ticks: u64,
    open: Option<unsafe extern "C" fn(*mut snd_timer) -> i32>,
    close: Option<unsafe extern "C" fn(*mut snd_timer) -> i32>,
    c_resolution: Option<unsafe extern "C" fn(*mut snd_timer) -> u64>,
    start: Option<unsafe extern "C" fn(*mut snd_timer) -> i32>,
    stop: Option<unsafe extern "C" fn(*mut snd_timer) -> i32>,
    set_period: Option<unsafe extern "C" fn(*mut snd_timer, u64, u64) -> i32>,
    precise_resolution: Option<unsafe extern "C" fn(*mut snd_timer, *mut u64, *mut u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer {
    tmr_class: i32,
    card: *mut snd_card,
    module: *mut module,
    tmr_device: i32,
    tmr_subdevice: i32,
    id: [u8; 64],
    name: [u8; 80],
    flags: u32,
    running: i32,
    sticks: u64,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_timer)>,
    hw: snd_timer_hardware,
    lock: spinlock,
    device_list: list_head,
    open_list_head: list_head,
    active_list_head: list_head,
    ack_list_head: list_head,
    sack_list_head: list_head,
    task_work: work_struct,
    kref: kref,
    max_instances: i32,
    num_instances: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_instance {
    timer: *mut snd_timer,
    owner: *mut u8,
    flags: u32,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_timer_instance)>,
    callback: Option<unsafe extern "C" fn(*mut snd_timer_instance, u64, u64)>,
    ccallback: Option<unsafe extern "C" fn(*mut snd_timer_instance, i32, *mut timespec64, u64)>,
    disconnect: Option<unsafe extern "C" fn(*mut snd_timer_instance)>,
    callback_data: *mut (),
    ticks: u64,
    cticks: u64,
    pticks: u64,
    resolution: u64,
    lost: u64,
    slave_class: i32,
    slave_id: u32,
    open_list: list_head,
    active_list: list_head,
    master_list: list_head,
    ack_list: list_head,
    slave_list_head: list_head,
    slave_active_head: list_head,
    master: *mut snd_timer_instance,
}


#[inline(always)]
fn __free_snd_timeri_timer(p: *mut ()) {
let _T: *mut snd_timer = unsafe { *(p as *mut *mut snd_timer) };
if !_T.is_null() {
snd_timeri_timer_put(_T);
}
}

// The rest seem to be more-or-less nonstandard. Check them!
#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
    fd: i32,
    events: i16,
    revents: i16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
    _qproc: Option<unsafe extern "C" fn(*mut file, *mut wait_queue_head, *mut poll_table_struct)>,
    _key: u32,
}


fn poll_wait(filp: *mut file, wait_address: *mut wait_queue_head, p: *mut poll_table_struct) {
if !p.is_null() && (unsafe { (*p)._qproc }) != None {
unsafe { unsafe { (*p)._qproc }.unwrap()(filp, wait_address, p) };
// This memory barrier is paired in the wq_has_sleeper().
// See the comment above prepare_to_wait(), we need to
// ensure that subsequent tests in this thread can't be
// reordered with __add_wait_queue() in _qproc() paths.
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
    buf: *mut u8,
    size: u64,
    from: u64,
    count: u64,
    pad_until: u64,
    index: i64,
    read_pos: i64,
    lock: mutex,
    op: *const seq_operations,
    poll_event: i32,
    file: *const file,
    private: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_operations {
    start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_info_buffer {
    buffer: *mut u8,
    curr: u32,
    size: u32,
    len: u32,
    stop: i32,
    error: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_info_entry_text {
    read: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut snd_info_buffer)>,
    write: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut snd_info_buffer)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_info_entry_ops {
    open: Option<unsafe extern "C" fn(*mut snd_info_entry, u16, *mut *mut ()) -> i32>,
    release: Option<unsafe extern "C" fn(*mut snd_info_entry, u16, *mut ()) -> i32>,
    read: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut file, *mut u8, u64, i64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut file, *const u8, u64, i64) -> i64>,
    llseek: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut file, i64, i32) -> i64>,
    poll: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut file, *mut poll_table_struct) -> u32>,
    ioctl: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut file, u32, u64) -> i32>,
    mmap: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut inode, *mut file, *mut vm_area_struct) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_info_entry {
    name: *const u8,
    mode: u16,
    size: i64,
    content: u16,
    c: __SlateRecord_info_h_65,
    parent: *mut snd_info_entry,
    module: *mut module,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_info_entry)>,
    p: *mut proc_dir_entry,
    access: mutex,
    children: list_head,
    list: list_head,
}


// snd_iprintf - printf on the procfs buffer
// @buf: the procfs buffer
// @fmt: the printf format
//
// Outputs the string on the procfs buffer just like printf().
//
// Return: zero for success, or a negative error code.
// internal flags
static mut timer_limit: i32 = 4;

static mut timer_tstamp_monotonic: i32 = 1;

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_467: [u8; 80] = [115, 110, 100, 95, 116, 105, 109, 101, 114, 46, 97, 117, 116, 104, 111, 114, 61, 74, 97, 114, 111, 115, 108, 97, 118, 32, 75, 121, 115, 101, 108, 97, 32, 60, 112, 101, 114, 101, 120, 64, 112, 101, 114, 101, 120, 46, 99, 122, 62, 44, 32, 84, 97, 107, 97, 115, 104, 105, 32, 73, 119, 97, 105, 32, 60, 116, 105, 119, 97, 105, 64, 115, 117, 115, 101, 46, 100, 101, 62, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_468: [u8; 43] = [115, 110, 100, 95, 116, 105, 109, 101, 114, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 65, 76, 83, 65, 32, 116, 105, 109, 101, 114, 32, 105, 110, 116, 101, 114, 102, 97, 99, 101, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_469: [u8; 36] = [115, 110, 100, 95, 116, 105, 109, 101, 114, 46, 102, 105, 108, 101, 61, 115, 111, 117, 110, 100, 47, 99, 111, 114, 101, 47, 115, 110, 100, 45, 116, 105, 109, 101, 114, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_470: [u8; 22] = [115, 110, 100, 95, 116, 105, 109, 101, 114, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

fn __check_timer_limit() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(timer_limit) }
}

static __param_str_timer_limit: __SlateAlign16<[u8; 22]> = __SlateAlign16([115u8, 110u8, 100u8, 95u8, 116u8, 105u8, 109u8, 101u8, 114u8, 46u8, 116u8, 105u8, 109u8, 101u8, 114u8, 95u8, 108u8, 105u8, 109u8, 105u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_timer_limit: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_timer_limit.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_timer_c_41>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_timer_c_41_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_timer_c_41_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_timer_c_41_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_timer_c_41_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_timer_c_41_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(timer_limit) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_471: [u8; 35] = [115, 110, 100, 95, 116, 105, 109, 101, 114, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 116, 105, 109, 101, 114, 95, 108, 105, 109, 105, 116, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_472: [u8; 60] = [115, 110, 100, 95, 116, 105, 109, 101, 114, 46, 112, 97, 114, 109, 61, 116, 105, 109, 101, 114, 95, 108, 105, 109, 105, 116, 58, 77, 97, 120, 105, 109, 117, 109, 32, 103, 108, 111, 98, 97, 108, 32, 116, 105, 109, 101, 114, 115, 32, 105, 110, 32, 115, 121, 115, 116, 101, 109, 46, 0];

fn __check_timer_tstamp_monotonic() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(timer_tstamp_monotonic) }
}

static __param_str_timer_tstamp_monotonic: __SlateAlign16<[u8; 33]> = __SlateAlign16([115u8, 110u8, 100u8, 95u8, 116u8, 105u8, 109u8, 101u8, 114u8, 46u8, 116u8, 105u8, 109u8, 101u8, 114u8, 95u8, 116u8, 115u8, 116u8, 97u8, 109u8, 112u8, 95u8, 109u8, 111u8, 110u8, 111u8, 116u8, 111u8, 110u8, 105u8, 99u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_timer_tstamp_monotonic: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_timer_tstamp_monotonic.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_timer_c_43>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_timer_c_43_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_timer_c_43_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_timer_c_43_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_timer_c_43_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_timer_c_43_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(timer_tstamp_monotonic) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_473: [u8; 46] = [115, 110, 100, 95, 116, 105, 109, 101, 114, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 116, 105, 109, 101, 114, 95, 116, 115, 116, 97, 109, 112, 95, 109, 111, 110, 111, 116, 111, 110, 105, 99, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_474: [u8; 97] = [115, 110, 100, 95, 116, 105, 109, 101, 114, 46, 112, 97, 114, 109, 61, 116, 105, 109, 101, 114, 95, 116, 115, 116, 97, 109, 112, 95, 109, 111, 110, 111, 116, 111, 110, 105, 99, 58, 85, 115, 101, 32, 112, 111, 115, 105, 120, 32, 109, 111, 110, 111, 116, 111, 110, 105, 99, 32, 99, 108, 111, 99, 107, 32, 115, 111, 117, 114, 99, 101, 32, 102, 111, 114, 32, 116, 105, 109, 101, 115, 116, 97, 109, 112, 115, 32, 40, 100, 101, 102, 97, 117, 108, 116, 41, 46, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_475: [u8; 34] = [115, 110, 100, 95, 116, 105, 109, 101, 114, 46, 97, 108, 105, 97, 115, 61, 99, 104, 97, 114, 45, 109, 97, 106, 111, 114, 45, 49, 49, 54, 45, 51, 51, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_476: [u8; 34] = [115, 110, 100, 95, 116, 105, 109, 101, 114, 46, 97, 108, 105, 97, 115, 61, 100, 101, 118, 110, 97, 109, 101, 58, 115, 110, 100, 47, 116, 105, 109, 101, 114, 0];

#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_tread32 {
    event: i32,
    tstamp_sec: i32,
    tstamp_nsec: i32,
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_tread64 {
    event: i32,
    pad1: [u8; 4],
    tstamp_sec: i64,
    tstamp_nsec: i64,
    val: u32,
    pad2: [u8; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_user {
    timeri: *mut snd_timer_instance,
    /// enhanced read with timestamps and events
    tread: i32,
    ticks: u64,
    overrun: u64,
    qhead: i32,
    qtail: i32,
    qused: i32,
    queue_size: i32,
    disconnected: bool,
    queue: *mut snd_timer_read,
    tqueue: *mut snd_timer_tread64,
    qlock: spinlock,
    last_resolution: u64,
    filter: u32,
    /// trigger tstamp
    tstamp: timespec64,
    qchange_sleep: wait_queue_head,
    fasync: *mut snd_fasync,
    ioctl_lock: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_status32 {
    /// Timestamp - last update
    tstamp_sec: i32,
    tstamp_nsec: i32,
    /// current period resolution in ns
    resolution: u32,
    /// counter of master tick lost
    lost: u32,
    /// count of read queue overruns
    overrun: u32,
    /// used queue size
    queue: u32,
    /// reserved
    reserved: [u8; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_status64 {
    /// Timestamp - last update
    tstamp_sec: i64,
    tstamp_nsec: i64,
    /// current period resolution in ns
    resolution: u32,
    /// counter of master tick lost
    lost: u32,
    /// count of read queue overruns
    overrun: u32,
    /// used queue size
    queue: u32,
    /// reserved
    reserved: [u8; 64],
}


/// list of timers
static mut snd_timer_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(snd_timer_list) }, prev: unsafe { core::ptr::addr_of_mut!(snd_timer_list) } };

/// list of slave instances
static mut snd_timer_slave_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(snd_timer_slave_list) }, prev: unsafe { core::ptr::addr_of_mut!(snd_timer_slave_list) } };

/// list of open master instances that can accept slave links
static mut snd_timer_master_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(snd_timer_master_list) }, prev: unsafe { core::ptr::addr_of_mut!(snd_timer_master_list) } };

/// rwlock for timer instance (for trigger actions)
static mut timeri_lock: rwlock = rwlock { raw_lock: qrwlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qrwlock_types_h_27 = unsafe { core::mem::zeroed() };
__t0.cnts = __SlateRecord_types_h_190 { counter: 0i32 };
__t0
}, wait_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } } };

static mut num_slaves: i32 = unsafe { core::mem::zeroed() };

static mut register_mutex: mutex = {
let mut __t0: mutex = unsafe { core::mem::zeroed() };
__t0.owner = __SlateRecord_types_h_197 { counter: 0i64 };
__t0.wait_lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.first_waiter = core::ptr::null_mut::<mutex_waiter>();
__t0
};

/// create a timer instance with the given owner string.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_instance_new")]
extern "C" fn snd_timer_instance_new(owner: *const u8) -> *mut snd_timer_instance {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut timeri: *mut snd_timer_instance = unsafe { core::mem::zeroed() };
let mut __v0: *mut snd_timer_instance;
let __obj_size: u64 = size_mul(core::mem::size_of::<snd_timer_instance>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut snd_timer_instance;
timeri = __v0;
if timeri.is_null() {
return core::ptr::null_mut::<snd_timer_instance>();
}
unsafe {
(*timeri).owner = kstrdup(owner, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
}
if unsafe { (*timeri).owner }.is_null() {
unsafe { kfree(timeri as *const ()) };
return core::ptr::null_mut::<snd_timer_instance>();
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*timeri).open_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*timeri).active_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*timeri).master_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*timeri).ack_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*timeri).slave_list_head) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*timeri).slave_active_head) });
timeri
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timer_instance_new_477: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> *mut snd_timer_instance>, *mut ()>(Some(snd_timer_instance_new)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_instance_free")]
extern "C" fn snd_timer_instance_free(timeri: *mut snd_timer_instance) {
if !timeri.is_null() {
if (unsafe { (*timeri).private_free }) != None {
unsafe { unsafe { (*timeri).private_free }.unwrap()(timeri) };
}
unsafe { kfree((*timeri).owner as *const ()) };
unsafe { kfree(timeri as *const ()) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timer_instance_free_478: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_timer_instance)>, *mut ()>(Some(snd_timer_instance_free)) };

/// find a timer instance from the given timer id
fn snd_timer_find(tid: *mut snd_timer_id) -> *mut snd_timer {
let mut timer: *mut snd_timer = unsafe { core::mem::zeroed() };
timer = list_first_entry__10!(unsafe { core::ptr::addr_of_mut!(snd_timer_list) });
while list_entry_is_head!(timer, unsafe { core::ptr::addr_of_mut!(snd_timer_list) }, device_list) == 0 {
if (unsafe { (*timer).tmr_class }) == unsafe { (*tid).dev_class } {
if !(((unsafe { (*timer).tmr_class }) == 2 || (unsafe { (*timer).tmr_class }) == 3) && (unsafe { (*timer).card }.is_null() || (unsafe { (*(*timer).card).number }) != unsafe { (*tid).card })) {
if (unsafe { (*timer).tmr_device }) == unsafe { (*tid).device } {
if (unsafe { (*timer).tmr_subdevice }) == unsafe { (*tid).subdevice } {
return timer;
}
}
}
}
timer = list_next_entry__7!(timer, device_list);
}
core::ptr::null_mut::<snd_timer>()
}

fn snd_timer_request(tid: *mut snd_timer_id) {
match unsafe { (*tid).dev_class } {
1 => {
if (unsafe { (*tid).device }) < unsafe { timer_limit } {
unsafe { __request_module(true, c"snd-timer-%i".as_ptr() as *const u8, (*tid).device) };
}
}
2 | 3 => {
if (unsafe { (*tid).card }) < unsafe { snd_ecards_limit } {
unsafe { __request_module(true, c"snd-card-%i".as_ptr() as *const u8, (*tid).card) };
}
}
_ => {
}
}
}

fn snd_timer_ref_get(timer: *mut snd_timer) {
kref_get(unsafe { core::ptr::addr_of_mut!((*timer).kref) });
}

fn snd_timer_ref_put(timer: *mut snd_timer) {
kref_put(unsafe { core::ptr::addr_of_mut!((*timer).kref) }, Some(snd_timer_kref_release));
}

/// Return the assigned timer for the instance, NULL if not present;
/// the caller is responsible to call snd_timeri_timer_put(), or use auto-cleanup
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timeri_timer_get")]
extern "C" fn snd_timeri_timer_get(timeri: *mut snd_timer_instance) -> *mut snd_timer {
let mut __UNIQUE_ID_guard_479: __SlateRecord_spinlock_h_670 = class_read_lock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(timeri_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_479) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_read_lock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_480: *mut __SlateRecord_spinlock_h_670 = ((unsafe { core::ptr::addr_of_mut!(timeri_lock) }) as u64) as *mut __SlateRecord_spinlock_h_670;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_480) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_read_lock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
let t: *mut snd_timer = unsafe { (*timeri).timer };
if t.is_null() {
return core::ptr::null_mut::<snd_timer>();
}
snd_timer_ref_get(t);
t
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timeri_timer_get_481: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_timer_instance) -> *mut snd_timer>, *mut ()>(Some(snd_timeri_timer_get)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timeri_timer_put")]
extern "C" fn snd_timeri_timer_put(timer: *mut snd_timer) {
snd_timer_ref_put(timer);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timeri_timer_put_482: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_timer)>, *mut ()>(Some(snd_timeri_timer_put)) };

/// move the slave if it belongs to the master; return 1 if match
fn check_matching_master_slave(master: *mut snd_timer_instance, slave: *mut snd_timer_instance) -> i32 {
if (unsafe { (*slave).slave_class }) != unsafe { (*master).slave_class } || (unsafe { (*slave).slave_id }) != unsafe { (*master).slave_id } {
return 0;
}
if (unsafe { (*(*master).timer).num_instances }) >= unsafe { (*(*master).timer).max_instances } {
return 16i32.wrapping_neg();
}
list_move_tail(unsafe { core::ptr::addr_of_mut!((*slave).open_list) }, unsafe { core::ptr::addr_of_mut!((*master).slave_list_head) });
let __v0: *mut snd_timer = unsafe { (*master).timer };
let __v1: i32 = unsafe { (*__v0).num_instances };
unsafe {
(*__v0).num_instances = __v1.wrapping_add(1);
}
snd_timer_ref_get(unsafe { (*master).timer });
let mut __UNIQUE_ID_guard_483: __SlateRecord_spinlock_h_683 = class_write_lock_irq_constructor(unsafe { core::ptr::addr_of_mut!(timeri_lock) });
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_483) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || class_write_lock_irq_destructor(__slate_cleanup_ptr_3 as *mut _));
let mut __UNIQUE_ID_unlock_484: *mut __SlateRecord_spinlock_h_683 = ((unsafe { core::ptr::addr_of_mut!(timeri_lock) }) as u64) as *mut __SlateRecord_spinlock_h_683;
let __slate_cleanup_ptr_4 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_484) as *mut _;
let __slate_cleanup_4 = ::slate_support::Guard(&mut || __class_write_lock_irq_cleanup_ctx(__slate_cleanup_ptr_4 as *mut _));
let mut __UNIQUE_ID_guard_485: __SlateRecord_spinlock_h_610 = class_spinlock_constructor(unsafe { core::ptr::addr_of_mut!((*(*master).timer).lock) });
let __slate_cleanup_ptr_5 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_485) as *mut _;
let __slate_cleanup_5 = ::slate_support::Guard(&mut || class_spinlock_destructor(__slate_cleanup_ptr_5 as *mut _));
let mut __UNIQUE_ID_unlock_486: *mut __SlateRecord_spinlock_h_610 = ((unsafe { core::ptr::addr_of_mut!((*(*master).timer).lock) }) as u64) as *mut __SlateRecord_spinlock_h_610;
let __slate_cleanup_ptr_6 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_486) as *mut _;
let __slate_cleanup_6 = ::slate_support::Guard(&mut || __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_6 as *mut _));
unsafe {
(*slave).master = master;
}
unsafe {
(*slave).timer = (*master).timer;
}
if (unsafe { (*slave).flags }) & 2 != 0 {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*slave).active_list) }, unsafe { core::ptr::addr_of_mut!((*master).slave_active_head) });
}
1
}

fn snd_timer_has_slave_key(timeri: *const snd_timer_instance) -> bool {
(unsafe { (*timeri).flags }) & 1 == 0 && (unsafe { (*timeri).slave_class }) > 0
}

/// look for a master instance matching with the slave id of the given slave.
/// when found, relink the open_link of the slave.
///
/// call this with register_mutex down.
fn snd_timer_check_slave(slave: *mut snd_timer_instance) -> i32 {
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut master: core::mem::MaybeUninit<*mut snd_timer_instance> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut snd_timer_instance = core::ptr::addr_of_mut!(master) as *mut *mut snd_timer_instance;
unsafe {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
err = 0;
*__slate_slot_1 = list_first_entry__20!(core::ptr::addr_of_mut!(snd_timer_master_list));
while list_entry_is_head!(*__slate_slot_1, core::ptr::addr_of_mut!(snd_timer_master_list), master_list) == 0 {
err = check_matching_master_slave(*__slate_slot_1, slave);
if err != 0 {
break;
} else {
// match found or error
*__slate_slot_1 = list_next_entry__18!(*__slate_slot_1, master_list);
}
}
return if err < 0 { err } else { 0 };
}
}

/// look for slave instances matching with the slave id of the given master.
/// when found, relink the open_link of slaves.
///
/// call this with register_mutex down.
fn snd_timer_check_master(master: *mut snd_timer_instance) -> i32 {
let mut slave: *mut snd_timer_instance = unsafe { core::mem::zeroed() };
let mut tmp: *mut snd_timer_instance = unsafe { core::mem::zeroed() };
let mut err: i32 = 0;
// check all pending slaves
slave = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!(snd_timer_slave_list) });
tmp = list_next_entry__2!(slave, open_list);
while list_entry_is_head!(slave, unsafe { core::ptr::addr_of_mut!(snd_timer_slave_list) }, open_list) == 0 {
err = check_matching_master_slave(master, slave);
if err < 0 {
break;
}
slave = tmp;
tmp = list_next_entry__2!(tmp, open_list);
}
if err < 0 { err } else { 0 }
}

/// open a timer instance
/// when opening a master, the slave id must be here given.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_open")]
extern "C" fn snd_timer_open(timeri: *mut snd_timer_instance, tid: *mut snd_timer_id, slave_id: u32) -> i32 {
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut card_dev_to_put: *mut device = unsafe { core::mem::zeroed() };
let mut timer: core::mem::MaybeUninit<*mut snd_timer> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut snd_timer = core::ptr::addr_of_mut!(timer) as *mut *mut snd_timer;
unsafe {
'__join_2: {
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
card_dev_to_put = core::ptr::null_mut::<device>();
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
mutex_lock(core::ptr::addr_of_mut!(register_mutex));
if (*tid).dev_class == 0 {
// open a slave instance
if (*tid).dev_sclass <= 0 || (*tid).dev_sclass > 3 {
*__slate_slot_10 = 22i32.wrapping_neg();
break '__join_2;
} else {
if num_slaves >= 1000 {
*__slate_slot_10 = 16i32.wrapping_neg();
break '__join_2;
} else {
(*timeri).slave_class = (*tid).dev_sclass;
(*timeri).slave_id = (*tid).device as u32;
(*timeri).flags = (*timeri).flags | 1;
list_add_tail(core::ptr::addr_of_mut!((*timeri).open_list), core::ptr::addr_of_mut!(snd_timer_slave_list));
let __v5: i32 = num_slaves;
num_slaves = __v5.wrapping_add(1);
*__slate_slot_10 = snd_timer_check_slave(timeri);
}
}
} else {
// open a master instance
*__slate_slot_12 = snd_timer_find(tid);
if (*__slate_slot_12).is_null() {
mutex_unlock(core::ptr::addr_of_mut!(register_mutex));
snd_timer_request(tid);
mutex_lock(core::ptr::addr_of_mut!(register_mutex));
*__slate_slot_12 = snd_timer_find(tid);
}
if (*__slate_slot_12).is_null() {
*__slate_slot_10 = 19i32.wrapping_neg();
break '__join_2;
} else {
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_12)).open_list_head) as *const list_head) == 0 {
let t: *mut snd_timer_instance = list_entry__28!(unsafe { (*(*__slate_slot_12)).open_list_head.next }, snd_timer_instance);
if (*t).flags & 64 != 0 {
*__slate_slot_10 = 16i32.wrapping_neg();
break '__join_2;
}
}
if (*(*__slate_slot_12)).num_instances >= (*(*__slate_slot_12)).max_instances {
*__slate_slot_10 = 16i32.wrapping_neg();
break '__join_2;
} else {
if !try_module_get((*(*__slate_slot_12)).module) {
*__slate_slot_10 = 16i32.wrapping_neg();
break '__join_2;
} else {
// take a card refcount for safe disconnection
if !unsafe { (*(*__slate_slot_12)).card }.is_null() {
get_device(core::ptr::addr_of_mut!((*(*(*__slate_slot_12)).card).card_dev));
card_dev_to_put = core::ptr::addr_of_mut!((*(*(*__slate_slot_12)).card).card_dev);
}
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_12)).open_list_head) as *const list_head) != 0 && (*(*__slate_slot_12)).hw.open != None {
*__slate_slot_10 = unsafe { (*(*__slate_slot_12)).hw.open }.unwrap()(*__slate_slot_12);
if *__slate_slot_10 != 0 {
module_put((*(*__slate_slot_12)).module);
break '__join_2;
}
}
(*timeri).timer = *__slate_slot_12;
(*timeri).slave_class = (*tid).dev_sclass;
(*timeri).slave_id = slave_id;
list_add_tail(core::ptr::addr_of_mut!((*timeri).open_list), core::ptr::addr_of_mut!((*(*__slate_slot_12)).open_list_head));
if snd_timer_has_slave_key(timeri as *const snd_timer_instance) {
list_add_tail(core::ptr::addr_of_mut!((*timeri).master_list), core::ptr::addr_of_mut!(snd_timer_master_list));
}
let __v2: *mut snd_timer = *__slate_slot_12;
let __v1: i32 = (*__v2).num_instances;
(*__v2).num_instances = __v1.wrapping_add(1);
snd_timer_ref_get(*__slate_slot_12);
*__slate_slot_10 = snd_timer_check_master(timeri);
}
}
}
}
if *__slate_slot_10 < 0 {
snd_timer_close_locked(timeri, &mut (*core::ptr::addr_of_mut!(card_dev_to_put)));
}
}
mutex_unlock(core::ptr::addr_of_mut!(register_mutex));
// put_device() is called after unlock for avoiding deadlock
if *__slate_slot_10 < 0 && !card_dev_to_put.is_null() {
put_device(card_dev_to_put);
}
return *__slate_slot_10;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timer_open_487: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_timer_instance, *mut snd_timer_id, u32) -> i32>, *mut ()>(Some(snd_timer_open)) };

/// remove slave links, called from snd_timer_close_locked() below
fn remove_slave_links(timeri: *mut snd_timer_instance, timer: *mut snd_timer) {
let mut slave: *mut snd_timer_instance = unsafe { core::mem::zeroed() };
let mut tmp: *mut snd_timer_instance = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_guard_488: __SlateRecord_spinlock_h_683 = class_write_lock_irq_constructor(unsafe { core::ptr::addr_of_mut!(timeri_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_488) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_write_lock_irq_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_489: *mut __SlateRecord_spinlock_h_683 = ((unsafe { core::ptr::addr_of_mut!(timeri_lock) }) as u64) as *mut __SlateRecord_spinlock_h_683;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_489) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_write_lock_irq_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
let mut __UNIQUE_ID_guard_490: __SlateRecord_spinlock_h_610 = class_spinlock_constructor(unsafe { core::ptr::addr_of_mut!((*timer).lock) });
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_490) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || class_spinlock_destructor(__slate_cleanup_ptr_2 as *mut _));
let mut __UNIQUE_ID_unlock_491: *mut __SlateRecord_spinlock_h_610 = ((unsafe { core::ptr::addr_of_mut!((*timer).lock) }) as u64) as *mut __SlateRecord_spinlock_h_610;
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_491) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _));
unsafe {
(*timeri).timer = core::ptr::null_mut::<snd_timer>();
}
slave = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*timeri).slave_list_head) });
tmp = list_next_entry__2!(slave, open_list);
while list_entry_is_head!(slave, unsafe { core::ptr::addr_of_mut!((*timeri).slave_list_head) }, open_list) == 0 {
list_move_tail(unsafe { core::ptr::addr_of_mut!((*slave).open_list) }, unsafe { core::ptr::addr_of_mut!(snd_timer_slave_list) });
let __v8: i32 = unsafe { (*timer).num_instances };
unsafe {
(*timer).num_instances = __v8.wrapping_sub(1);
}
snd_timer_ref_put(timer);
unsafe {
(*slave).master = core::ptr::null_mut::<snd_timer_instance>();
}
unsafe {
(*slave).timer = core::ptr::null_mut::<snd_timer>();
}
list_del_init(unsafe { core::ptr::addr_of_mut!((*slave).ack_list) });
list_del_init(unsafe { core::ptr::addr_of_mut!((*slave).active_list) });
slave = tmp;
tmp = list_next_entry__2!(tmp, open_list);
}
// the close is done; a reopen must not see the mark
let __v15: u32 = (unsafe { (*timeri).flags }) & 4294836223u32;
unsafe {
(*timeri).flags = __v15;
}
}

/// close a timer instance
/// call this with register_mutex down.
fn snd_timer_close_locked(timeri: *mut snd_timer_instance, card_devp_to_put: &mut *mut device) {
let timer: *mut snd_timer = unsafe { (*timeri).timer };
if !timer.is_null() {
let mut __UNIQUE_ID_guard_492: __SlateRecord_spinlock_h_620 = class_spinlock_irq_constructor(unsafe { core::ptr::addr_of_mut!((*timer).lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_492) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irq_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_493: *mut __SlateRecord_spinlock_h_620 = ((unsafe { core::ptr::addr_of_mut!((*timer).lock) }) as u64) as *mut __SlateRecord_spinlock_h_620;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_493) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irq_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if (unsafe { (*timeri).flags }) & 131072 != 0 {
return;
}
// already closed
unsafe {
(*timeri).flags = (*timeri).flags | 131072;
}
}
if list_empty((unsafe { core::ptr::addr_of_mut!((*timeri).open_list) }) as *const list_head) == 0 {
list_del_init(unsafe { core::ptr::addr_of_mut!((*timeri).open_list) });
if (unsafe { (*timeri).flags }) & 1 != 0 {
let __v5: i32 = unsafe { num_slaves };
unsafe {
num_slaves = __v5.wrapping_sub(1);
}
}
}
if list_empty((unsafe { core::ptr::addr_of_mut!((*timeri).master_list) }) as *const list_head) == 0 {
list_del_init(unsafe { core::ptr::addr_of_mut!((*timeri).master_list) });
}
// force to stop the timer
snd_timer_stop(timeri);
if !timer.is_null() {
let mut slave: *mut snd_timer_instance = unsafe { core::mem::zeroed() };
let mut busy: bool = false;
let __v8: i32 = unsafe { (*timer).num_instances };
unsafe {
(*timer).num_instances = __v8.wrapping_sub(1);
}
// unqueue then drain the slaves' callbacks before remove_slave_links() severs them
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*timer).lock) });
slave = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*timeri).slave_list_head) });
while list_entry_is_head!(slave, unsafe { core::ptr::addr_of_mut!((*timeri).slave_list_head) }, open_list) == 0 {
list_del_init(unsafe { core::ptr::addr_of_mut!((*slave).ack_list) });
slave = list_next_entry__2!(slave, open_list);
}
loop {
busy = (unsafe { (*timeri).flags }) & 32 != 0;
slave = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*timeri).slave_list_head) });
while list_entry_is_head!(slave, unsafe { core::ptr::addr_of_mut!((*timeri).slave_list_head) }, open_list) == 0 {
busy = ((busy as i32) as u32) | (unsafe { (*slave).flags }) & 32 != 0;
slave = list_next_entry__2!(slave, open_list);
}
if !busy {
break;
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*timer).lock) });
udelay(10u64);
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*timer).lock) });
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*timer).lock) });
remove_slave_links(timeri, timer);
// slave doesn't need to release timer resources below
if (unsafe { (*timeri).flags }) & 1 == 0 {
if list_empty((unsafe { core::ptr::addr_of_mut!((*timer).open_list_head) }) as *const list_head) != 0 && (unsafe { (*timer).hw.close }) != None {
unsafe { unsafe { (*timer).hw.close }.unwrap()(timer) };
}
// release a card refcount for safe disconnection
if !unsafe { (*timer).card }.is_null() {
unsafe {
*(card_devp_to_put as *mut *mut device) = core::ptr::addr_of_mut!((*(*timer).card).card_dev);
}
}
unsafe { module_put((*timer).module) };
}
snd_timer_ref_put(timer);
}
}

/// close a timer instance
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_close")]
extern "C" fn snd_timer_close(timeri: *mut snd_timer_instance) {
let mut __UNIQUE_ID_unlock_495: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_495) as *mut *mut __SlateRecord_mutex_h_253;
let mut scope: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_mutex_h_253;
let mut card_dev_to_put: *mut device = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_mutex_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_mutex_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
card_dev_to_put = core::ptr::null_mut::<device>();
if snd_BUG_ON!(timeri.is_null()) != 0 {
return;
} else {
core::ptr::write(__slate_slot_1, class_mutex_constructor(core::ptr::addr_of_mut!(register_mutex)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!(register_mutex) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_0.set(true);
if !class_mutex_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_mutex_is_conditional {
if true {
snd_timer_close_locked(timeri, &mut (*core::ptr::addr_of_mut!(card_dev_to_put)));
}
}
__slate_cleanup_armed_0.set(false);
__class_mutex_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_mutex_destructor(__slate_slot_1 as *mut _);
// put_device() is called after unlock for avoiding deadlock
if !card_dev_to_put.is_null() {
put_device(card_dev_to_put);
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timer_close_496: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_timer_instance)>, *mut ()>(Some(snd_timer_close)) };

fn snd_timer_hw_resolution(timer: *mut snd_timer) -> u64 {
if (unsafe { (*timer).hw.c_resolution }) != None {
return unsafe { unsafe { (*timer).hw.c_resolution }.unwrap()(timer) };
} else {
return unsafe { (*timer).hw.resolution };
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_resolution")]
extern "C" fn snd_timer_resolution(timeri: *mut snd_timer_instance) -> u64 {
let mut ret: u64 = 0;
if timeri.is_null() {
return 0;
}
let mut timer: *mut snd_timer = snd_timeri_timer_get(timeri);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(timer) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_timeri_timer(__slate_cleanup_ptr_0 as *mut _));
if !timer.is_null() {
let mut __UNIQUE_ID_guard_497: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!((*timer).lock) });
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_497) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_1 as *mut _));
let mut __UNIQUE_ID_unlock_498: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!((*timer).lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_498) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _));
ret = snd_timer_hw_resolution(timer);
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timer_resolution_499: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_timer_instance) -> u64>, *mut ()>(Some(snd_timer_resolution)) };

fn snd_timer_notify1(ti: *mut snd_timer_instance, mut event: i32) {
let timer: *mut snd_timer = unsafe { (*ti).timer };
let mut resolution: u64 = 0;
let mut ts: *mut snd_timer_instance = unsafe { core::mem::zeroed() };
let mut tstamp: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tstamp) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
if (unsafe { timer_tstamp_monotonic }) != 0 {
unsafe { ktime_get_ts64(core::ptr::addr_of_mut!(tstamp)) };
} else {
unsafe { ktime_get_real_ts64(core::ptr::addr_of_mut!(tstamp)) };
}
if snd_BUG_ON!(event < 2 || event > 5) != 0 {
return;
}
if !timer.is_null() && (event == 2 || event == 4) {
resolution = snd_timer_hw_resolution(timer);
}
if (unsafe { (*ti).ccallback }) != None {
unsafe { unsafe { (*ti).ccallback }.unwrap()(ti, event, core::ptr::addr_of_mut!(tstamp), resolution) };
}
if (unsafe { (*ti).flags }) & 1 != 0 {
return;
}
if timer.is_null() {
return;
}
if (unsafe { (*timer).hw.flags }) & 4 != 0 {
return;
}
event = event.wrapping_add(10); // convert to SNDRV_TIMER_EVENT_MXXX
ts = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!((*ti).slave_active_head) });
while list_entry_is_head!(ts, unsafe { core::ptr::addr_of_mut!((*ti).slave_active_head) }, active_list) == 0 {
if (unsafe { (*ts).ccallback }) != None {
unsafe { unsafe { (*ts).ccallback }.unwrap()(ts, event, core::ptr::addr_of_mut!(tstamp), resolution) };
}
ts = list_next_entry__5!(ts, active_list);
}
}

/// start/continue a master timer
fn snd_timer_start1(timeri: *mut snd_timer_instance, start: bool, ticks: u64) -> i32 {
let mut __UNIQUE_ID_unlock_501: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_610> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_501) as *mut *mut __SlateRecord_spinlock_h_610;
let mut __UNIQUE_ID_guard_500: core::mem::MaybeUninit<__SlateRecord_spinlock_h_610> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_500) as *mut __SlateRecord_spinlock_h_610;
let mut result: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(result) as *mut i32;
let mut timer: core::mem::MaybeUninit<*mut snd_timer> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut snd_timer = core::ptr::addr_of_mut!(timer) as *mut *mut snd_timer;
let __slate_cleanup_ptr_14 = __slate_slot_14;
let __slate_cleanup_armed_14 = core::cell::Cell::new(false);
let __slate_cleanup_14 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_14.get() {
unsafe { class_spinlock_destructor(__slate_cleanup_ptr_14 as *mut _) };
}
});
let __slate_cleanup_ptr_13 = __slate_slot_13;
let __slate_cleanup_armed_13 = core::cell::Cell::new(false);
let __slate_cleanup_13 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_13.get() {
unsafe { __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_13 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_16 = (*timeri).timer;
if (*__slate_slot_16).is_null() {
return 22i32.wrapping_neg();
} else {
core::ptr::write(__slate_slot_14, class_spinlock_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_16)).lock)));
__slate_cleanup_armed_14.set(true);
core::ptr::write(__slate_slot_13, (core::ptr::addr_of_mut!((*(*__slate_slot_16)).lock) as u64) as *mut __SlateRecord_spinlock_h_610);
__slate_cleanup_armed_13.set(true);
if (*timeri).flags & 131072 != 0 {
return 22i32.wrapping_neg();
} else {
if !unsafe { (*(*__slate_slot_16)).card }.is_null() && (*(*(*__slate_slot_16)).card).shutdown != 0 {
return 19i32.wrapping_neg();
} else {
if (*timeri).flags & 6u32 != 0 {
return 16i32.wrapping_neg();
} else {
// check the actual time for the start tick;
// bail out as error if it's way too low (< 100us)
if start && (*(*__slate_slot_16)).hw.flags & 4 == 0 {
if snd_timer_hw_resolution(*__slate_slot_16).wrapping_mul(ticks) < 100000 {
return 22i32.wrapping_neg();
}
}
if start {
let __v12: u64 = ticks;
(*timeri).cticks = __v12;
(*timeri).ticks = __v12;
} else {
if (*timeri).cticks == 0 {
(*timeri).cticks = 1;
}
}
'__join_0: {
(*timeri).pticks = 0;
list_move_tail(core::ptr::addr_of_mut!((*timeri).active_list), core::ptr::addr_of_mut!((*(*__slate_slot_16)).active_list_head));
if (*(*__slate_slot_16)).running != 0 {
if (*(*__slate_slot_16)).hw.flags & 4 == 0 {
let __v5: *mut snd_timer = *__slate_slot_16;
(*__v5).flags = (*__v5).flags | 2;
(*timeri).flags = (*timeri).flags | 4;
*__slate_slot_15 = 1; // delayed start
break '__join_0;
}
} else {
if start {
(*(*__slate_slot_16)).sticks = ticks;
}
unsafe { (*(*__slate_slot_16)).hw.start }.unwrap()(*__slate_slot_16);
}
let __v11: *mut snd_timer = *__slate_slot_16;
let __v10: i32 = (*__v11).running;
(*__v11).running = __v10.wrapping_add(1);
(*timeri).flags = (*timeri).flags | 2;
*__slate_slot_15 = 0;
}
snd_timer_notify1(timeri, if start { 2i32 } else { 4i32 });
return *__slate_slot_15;
}
}
}
}
}
}

/// start/continue a slave timer
fn snd_timer_start_slave(timeri: *mut snd_timer_instance, start: bool) -> i32 {
if (unsafe { (*timeri).flags }) & 131072 != 0 {
return 22i32.wrapping_neg();
}
if (unsafe { (*timeri).flags }) & 2 != 0 {
return 16i32.wrapping_neg();
}
unsafe {
(*timeri).flags = (*timeri).flags | 2;
}
if !unsafe { (*timeri).master }.is_null() && !unsafe { (*timeri).timer }.is_null() {
let mut __UNIQUE_ID_guard_502: __SlateRecord_spinlock_h_610 = class_spinlock_constructor(unsafe { core::ptr::addr_of_mut!((*(*timeri).timer).lock) });
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_502) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || class_spinlock_destructor(__slate_cleanup_ptr_3 as *mut _));
let mut __UNIQUE_ID_unlock_503: *mut __SlateRecord_spinlock_h_610 = ((unsafe { core::ptr::addr_of_mut!((*(*timeri).timer).lock) }) as u64) as *mut __SlateRecord_spinlock_h_610;
let __slate_cleanup_ptr_4 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_503) as *mut _;
let __slate_cleanup_4 = ::slate_support::Guard(&mut || __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_4 as *mut _));
list_add_tail(unsafe { core::ptr::addr_of_mut!((*timeri).active_list) }, unsafe { core::ptr::addr_of_mut!((*(*timeri).master).slave_active_head) });
snd_timer_notify1(timeri, if start { 2i32 } else { 4i32 });
}
return 1; // delayed start
}

/// stop/pause a master timer
fn snd_timer_stop1(timeri: *mut snd_timer_instance, stop: bool) -> i32 {
let mut timer: *mut snd_timer = unsafe { core::mem::zeroed() };
timer = unsafe { (*timeri).timer };
if timer.is_null() {
return 22i32.wrapping_neg();
}
let mut __UNIQUE_ID_guard_504: __SlateRecord_spinlock_h_610 = class_spinlock_constructor(unsafe { core::ptr::addr_of_mut!((*timer).lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_504) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_505: *mut __SlateRecord_spinlock_h_610 = ((unsafe { core::ptr::addr_of_mut!((*timer).lock) }) as u64) as *mut __SlateRecord_spinlock_h_610;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_505) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
list_del_init(unsafe { core::ptr::addr_of_mut!((*timeri).ack_list) });
list_del_init(unsafe { core::ptr::addr_of_mut!((*timeri).active_list) });
if (unsafe { (*timeri).flags }) & 6u32 == 0 {
return 16i32.wrapping_neg();
}
if !unsafe { (*timer).card }.is_null() && (unsafe { (*(*timer).card).shutdown }) != 0 {
return 0;
}
if stop {
unsafe {
(*timeri).cticks = (*timeri).ticks;
}
unsafe {
(*timeri).pticks = 0;
}
}
let __v2: bool = if (unsafe { (*timeri).flags }) & 2 != 0 {
let __v4: i32 = unsafe { (*timer).running };
let __v5: i32 = __v4.wrapping_sub(1);
unsafe {
(*timer).running = __v5;
}
__v5 == 0
} else {
false
};
if __v2 {
unsafe { unsafe { (*timer).hw.stop }.unwrap()(timer) };
if (unsafe { (*timer).flags }) & 2 != 0 {
let __v8: u32 = (unsafe { (*timer).flags }) & 4294967293u32;
unsafe {
(*timer).flags = __v8;
}
snd_timer_reschedule(timer, 0u64);
if (unsafe { (*timer).flags }) & 1 != 0 {
let __v11: u32 = (unsafe { (*timer).flags }) & 4294967294u32;
unsafe {
(*timer).flags = __v11;
}
unsafe { unsafe { (*timer).hw.start }.unwrap()(timer) };
}
}
}
let __v14: u32 = (unsafe { (*timeri).flags }) & 4294967289u32;
unsafe {
(*timeri).flags = __v14;
}
if stop {
let __v17: u32 = (unsafe { (*timeri).flags }) & 4294901759u32;
unsafe {
(*timeri).flags = __v17;
}
} else {
unsafe {
(*timeri).flags = (*timeri).flags | 65536;
}
}
snd_timer_notify1(timeri, if stop { 3i32 } else { 5i32 });
0
}

/// stop/pause a slave timer
fn snd_timer_stop_slave(timeri: *mut snd_timer_instance, stop: bool) -> i32 {
let mut running: bool = false;
running = (unsafe { (*timeri).flags }) & 2 != 0;
let __v2: u32 = (unsafe { (*timeri).flags }) & 4294967293u32;
unsafe {
(*timeri).flags = __v2;
}
if !unsafe { (*timeri).timer }.is_null() {
let mut __UNIQUE_ID_guard_506: __SlateRecord_spinlock_h_610 = class_spinlock_constructor(unsafe { core::ptr::addr_of_mut!((*(*timeri).timer).lock) });
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_506) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || class_spinlock_destructor(__slate_cleanup_ptr_3 as *mut _));
let mut __UNIQUE_ID_unlock_507: *mut __SlateRecord_spinlock_h_610 = ((unsafe { core::ptr::addr_of_mut!((*(*timeri).timer).lock) }) as u64) as *mut __SlateRecord_spinlock_h_610;
let __slate_cleanup_ptr_4 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_507) as *mut _;
let __slate_cleanup_4 = ::slate_support::Guard(&mut || __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_4 as *mut _));
list_del_init(unsafe { core::ptr::addr_of_mut!((*timeri).ack_list) });
list_del_init(unsafe { core::ptr::addr_of_mut!((*timeri).active_list) });
if running {
snd_timer_notify1(timeri, if stop { 3i32 } else { 5i32 });
}
}
if running { 0 } else { 16i32.wrapping_neg() }
}

/// start the timer instance
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_start")]
extern "C" fn snd_timer_start(timeri: *mut snd_timer_instance, ticks: u32) -> i32 {
if timeri.is_null() || ticks < 1 {
return 22i32.wrapping_neg();
}
let mut __UNIQUE_ID_guard_508: __SlateRecord_spinlock_h_670 = class_read_lock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(timeri_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_508) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_read_lock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_509: *mut __SlateRecord_spinlock_h_670 = ((unsafe { core::ptr::addr_of_mut!(timeri_lock) }) as u64) as *mut __SlateRecord_spinlock_h_670;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_509) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_read_lock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if (unsafe { (*timeri).flags }) & 1 != 0 {
return snd_timer_start_slave(timeri, true);
} else {
return snd_timer_start1(timeri, true, ticks as u64);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timer_start_510: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_timer_instance, u32) -> i32>, *mut ()>(Some(snd_timer_start)) };

/// stop the timer instance.
///
/// do not call this from the timer callback!
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_stop")]
extern "C" fn snd_timer_stop(timeri: *mut snd_timer_instance) -> i32 {
let mut __UNIQUE_ID_guard_511: __SlateRecord_spinlock_h_670 = class_read_lock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(timeri_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_511) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_read_lock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_512: *mut __SlateRecord_spinlock_h_670 = ((unsafe { core::ptr::addr_of_mut!(timeri_lock) }) as u64) as *mut __SlateRecord_spinlock_h_670;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_512) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_read_lock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if (unsafe { (*timeri).flags }) & 1 != 0 {
return snd_timer_stop_slave(timeri, true);
} else {
return snd_timer_stop1(timeri, true);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timer_stop_513: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_timer_instance) -> i32>, *mut ()>(Some(snd_timer_stop)) };

/// start again..  the tick is kept.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_continue")]
extern "C" fn snd_timer_continue(timeri: *mut snd_timer_instance) -> i32 {
// timer can continue only after pause
if (unsafe { (*timeri).flags }) & 65536 == 0 {
return 22i32.wrapping_neg();
}
let mut __UNIQUE_ID_guard_514: __SlateRecord_spinlock_h_670 = class_read_lock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(timeri_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_514) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_read_lock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_515: *mut __SlateRecord_spinlock_h_670 = ((unsafe { core::ptr::addr_of_mut!(timeri_lock) }) as u64) as *mut __SlateRecord_spinlock_h_670;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_515) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_read_lock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if (unsafe { (*timeri).flags }) & 1 != 0 {
return snd_timer_start_slave(timeri, false);
} else {
return snd_timer_start1(timeri, false, 0u64);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timer_continue_516: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_timer_instance) -> i32>, *mut ()>(Some(snd_timer_continue)) };

/// pause.. remember the ticks left
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_pause")]
extern "C" fn snd_timer_pause(timeri: *mut snd_timer_instance) -> i32 {
let mut __UNIQUE_ID_guard_517: __SlateRecord_spinlock_h_670 = class_read_lock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(timeri_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_517) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_read_lock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_518: *mut __SlateRecord_spinlock_h_670 = ((unsafe { core::ptr::addr_of_mut!(timeri_lock) }) as u64) as *mut __SlateRecord_spinlock_h_670;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_518) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_read_lock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if (unsafe { (*timeri).flags }) & 1 != 0 {
return snd_timer_stop_slave(timeri, false);
} else {
return snd_timer_stop1(timeri, false);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timer_pause_519: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_timer_instance) -> i32>, *mut ()>(Some(snd_timer_pause)) };

/// reschedule the timer
///
/// start pending instances and check the scheduling ticks.
/// when the scheduling ticks is changed set CHANGE flag to reprogram the timer.
fn snd_timer_reschedule(timer: *mut snd_timer, ticks_left: u64) {
let mut ti: *mut snd_timer_instance = unsafe { core::mem::zeroed() };
let mut ticks: u64 = !0;
ti = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!((*timer).active_list_head) });
while list_entry_is_head!(ti, unsafe { core::ptr::addr_of_mut!((*timer).active_list_head) }, active_list) == 0 {
if (unsafe { (*ti).flags }) & 4 != 0 {
let __v3: u32 = (unsafe { (*ti).flags }) & 4294967291u32;
unsafe {
(*ti).flags = __v3;
}
unsafe {
(*ti).flags = (*ti).flags | 2;
}
let __v8: i32 = unsafe { (*timer).running };
unsafe {
(*timer).running = __v8.wrapping_add(1);
}
}
if (unsafe { (*ti).flags }) & 2 != 0 {
if ticks > unsafe { (*ti).cticks } {
ticks = unsafe { (*ti).cticks };
}
}
ti = list_next_entry__5!(ti, active_list);
}
if ticks == !0 {
let __v14: u32 = (unsafe { (*timer).flags }) & 4294967293u32;
unsafe {
(*timer).flags = __v14;
}
return;
}
if ticks > unsafe { (*timer).hw.ticks } {
ticks = unsafe { (*timer).hw.ticks };
}
if ticks_left != ticks {
unsafe {
(*timer).flags = (*timer).flags | 1;
}
}
unsafe {
(*timer).sticks = ticks;
}
}

/// call callbacks in timer ack list
fn snd_timer_process_callbacks(timer: *mut snd_timer, head: *mut list_head) {
let mut ti: *mut snd_timer_instance = unsafe { core::mem::zeroed() };
let mut resolution: u64 = 0;
let mut ticks: u64 = 0;
while list_empty(head as *const list_head) == 0 {
ti = list_first_entry__25!(head, snd_timer_instance);
// remove from ack_list and make empty
list_del_init(unsafe { core::ptr::addr_of_mut!((*ti).ack_list) });
if (unsafe { (*ti).flags }) & 131072 == 0 {
ticks = unsafe { (*ti).pticks };
unsafe {
(*ti).pticks = 0;
}
resolution = unsafe { (*ti).resolution };
unsafe {
(*ti).flags = (*ti).flags | 32;
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*timer).lock) });
if (unsafe { (*ti).callback }) != None {
unsafe { unsafe { (*ti).callback }.unwrap()(ti, resolution, ticks) };
}
spin_lock(unsafe { core::ptr::addr_of_mut!((*timer).lock) });
let __v6: u32 = (unsafe { (*ti).flags }) & 4294967263u32;
unsafe {
(*ti).flags = __v6;
}
}
}
}

/// clear pending instances from ack list
fn snd_timer_clear_callbacks(timer: *mut snd_timer, head: *mut list_head) {
let mut __UNIQUE_ID_guard_520: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!((*timer).lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_520) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_521: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!((*timer).lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_521) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
while list_empty(head as *const list_head) == 0 {
list_del_init(unsafe { (*head).next });
}
}

/// timer work
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_work")]
extern "C" fn snd_timer_work(work: *mut work_struct) {
let timer: *mut snd_timer = container_of__24!(work, snd_timer);
if !unsafe { (*timer).card }.is_null() && (unsafe { (*(*timer).card).shutdown }) != 0 {
snd_timer_clear_callbacks(timer, unsafe { core::ptr::addr_of_mut!((*timer).sack_list_head) });
return;
}
let mut __UNIQUE_ID_guard_522: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!((*timer).lock) });
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_522) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_1 as *mut _));
let mut __UNIQUE_ID_unlock_523: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!((*timer).lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_523) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _));
snd_timer_process_callbacks(timer, unsafe { core::ptr::addr_of_mut!((*timer).sack_list_head) });
}

/// timer interrupt
///
/// ticks_left is usually equal to timer->sticks.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_interrupt")]
extern "C" fn snd_timer_interrupt(timer: *mut snd_timer, ticks_left: u64) {
let mut ti: *mut snd_timer_instance = unsafe { core::mem::zeroed() };
let mut ts: *mut snd_timer_instance = unsafe { core::mem::zeroed() };
let mut tmp: *mut snd_timer_instance = unsafe { core::mem::zeroed() };
let mut resolution: u64 = 0;
let mut ack_list_head: *mut list_head = unsafe { core::mem::zeroed() };
if timer.is_null() {
return;
}
if !unsafe { (*timer).card }.is_null() && (unsafe { (*(*timer).card).shutdown }) != 0 {
snd_timer_clear_callbacks(timer, unsafe { core::ptr::addr_of_mut!((*timer).ack_list_head) });
return;
}
let mut __UNIQUE_ID_guard_524: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!((*timer).lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_524) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_525: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!((*timer).lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_525) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
// remember the current resolution
resolution = snd_timer_hw_resolution(timer);
// loop for all active instances
// Here we cannot use list_for_each_entry because the active_list of a
// processed instance is relinked to done_list_head before the callback
// is called.
ti = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!((*timer).active_list_head) });
tmp = list_next_entry__5!(ti, active_list);
while list_entry_is_head!(ti, unsafe { core::ptr::addr_of_mut!((*timer).active_list_head) }, active_list) == 0 {
if (unsafe { (*ti).flags }) & 131072 == 0 {
if (unsafe { (*ti).flags }) & 2 != 0 {
let __v7: u64 = unsafe { (*ti).pticks };
unsafe {
(*ti).pticks = __v7.wrapping_add(ticks_left);
}
unsafe {
(*ti).resolution = resolution;
}
if (unsafe { (*ti).cticks }) < ticks_left {
unsafe {
(*ti).cticks = 0;
}
} else {
let __v10: u64 = unsafe { (*ti).cticks };
unsafe {
(*ti).cticks = __v10.wrapping_sub(ticks_left);
}
}
if (unsafe { (*ti).cticks }) == 0 {
// not expired
if (unsafe { (*ti).flags }) & 8 != 0 {
unsafe {
(*ti).cticks = (*ti).ticks;
}
} else {
let __v14: u32 = (unsafe { (*ti).flags }) & 4294967293u32;
unsafe {
(*ti).flags = __v14;
}
let __v16: i32 = unsafe { (*timer).running };
unsafe {
(*timer).running = __v16.wrapping_sub(1);
}
list_del_init(unsafe { core::ptr::addr_of_mut!((*ti).active_list) });
}
if (unsafe { (*timer).hw.flags }) & 16 != 0 || (unsafe { (*ti).flags }) & 16 != 0 {
ack_list_head = unsafe { core::ptr::addr_of_mut!((*timer).ack_list_head) };
} else {
ack_list_head = unsafe { core::ptr::addr_of_mut!((*timer).sack_list_head) };
}
// don't requeue an instance whose callback is still running
if list_empty((unsafe { core::ptr::addr_of_mut!((*ti).ack_list) }) as *const list_head) != 0 && (unsafe { (*ti).flags }) & 32 == 0 {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*ti).ack_list) }, ack_list_head);
}
ts = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!((*ti).slave_active_head) });
while list_entry_is_head!(ts, unsafe { core::ptr::addr_of_mut!((*ti).slave_active_head) }, active_list) == 0 {
unsafe {
(*ts).pticks = (*ti).pticks;
}
unsafe {
(*ts).resolution = resolution;
}
if list_empty((unsafe { core::ptr::addr_of_mut!((*ts).ack_list) }) as *const list_head) != 0 && (unsafe { (*ts).flags }) & 32 == 0 {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*ts).ack_list) }, ack_list_head);
}
ts = list_next_entry__5!(ts, active_list);
}
}
}
}
ti = tmp;
tmp = list_next_entry__5!(tmp, active_list);
}
if (unsafe { (*timer).flags }) & 2 != 0 {
snd_timer_reschedule(timer, unsafe { (*timer).sticks });
}
if (unsafe { (*timer).running }) != 0 {
if (unsafe { (*timer).hw.flags }) & 2 != 0 {
unsafe { unsafe { (*timer).hw.stop }.unwrap()(timer) };
unsafe {
(*timer).flags = (*timer).flags | 1;
}
}
if (unsafe { (*timer).hw.flags }) & 1 == 0 || (unsafe { (*timer).flags }) & 1 != 0 {
// restart timer
let __v28: u32 = (unsafe { (*timer).flags }) & 4294967294u32;
unsafe {
(*timer).flags = __v28;
}
unsafe { unsafe { (*timer).hw.start }.unwrap()(timer) };
}
} else {
unsafe { unsafe { (*timer).hw.stop }.unwrap()(timer) };
}
// now process all fast callbacks
snd_timer_process_callbacks(timer, unsafe { core::ptr::addr_of_mut!((*timer).ack_list_head) });
// do we have any slow callbacks?
if list_empty((unsafe { core::ptr::addr_of_mut!((*timer).sack_list_head) }) as *const list_head) == 0 {
queue_work(unsafe { system_highpri_wq }, unsafe { core::ptr::addr_of_mut!((*timer).task_work) });
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timer_interrupt_526: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_timer, u64)>, *mut ()>(Some(snd_timer_interrupt)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_new")]
extern "C" fn snd_timer_new(card: *mut snd_card, id: *mut u8, tid: *mut snd_timer_id, rtimer: *mut *mut snd_timer) -> i32 {
let mut __slate_compound_10: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: spinlock = unsafe { core::mem::zeroed() };
let mut __slate_compound_15: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut timer: *mut snd_timer = unsafe { core::mem::zeroed() };
let mut err: i32 = 0;
if snd_BUG_ON!(tid.is_null()) != 0 {
return 22i32.wrapping_neg();
}
if (unsafe { (*tid).dev_class }) == 2 || (unsafe { (*tid).dev_class }) == 3 {
let mut __v2: i64;
let __ret_warn_on: i32 = card.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("527: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 527b - ., 3; .popsection\n# {0}", const 527 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_snd_timer_new_3, sym __slate_asm_string_snd_timer_new_4, const 999 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("528: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 528b - ., 4; .popsection\n# {0}", const 528 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
return 22i32.wrapping_neg();
}
}
if !rtimer.is_null() {
unsafe {
*rtimer = core::ptr::null_mut::<snd_timer>();
}
}
let mut __v7: *mut snd_timer;
let __obj_size: u64 = size_mul(core::mem::size_of::<snd_timer>() as u64, 1u64);
let mut __v8: *mut ();
let mut __v9: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_10 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_10)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_12 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_12)
} });
}
__v9 = _res;
__v8 = __v9;
__v7 = __v8 as *mut snd_timer;
timer = __v7;
if timer.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*timer).tmr_class = (*tid).dev_class;
}
unsafe {
(*timer).card = card;
}
unsafe {
(*timer).tmr_device = (*tid).device;
}
unsafe {
(*timer).tmr_subdevice = (*tid).subdevice;
}
if !id.is_null() {
unsafe { sized_strscpy((*timer).id.as_mut_ptr() as *mut u8, id as *const u8, (core::mem::size_of::<[u8; 64]>() as u64).wrapping_add(__must_be_cstr__23!() as u64).wrapping_add(__must_be_cstr__65!() as u64)) };
}
unsafe {
(*timer).sticks = 1;
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*timer).device_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*timer).open_list_head) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*timer).active_list_head) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*timer).ack_list_head) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*timer).sack_list_head) });
'__slate_break_14: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*timer).lock) });
unsafe {
*core::ptr::addr_of_mut!((*timer).lock) = *{
__slate_compound_13 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_13)
};
}
if true {
break;
}
}
'__slate_break_17: loop {
'__slate_break_16: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*timer).task_work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*timer).task_work)).data = *{
__slate_compound_15 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_15)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*timer).task_work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*timer).task_work)).func = Some(snd_timer_work);
}
if true {
break;
}
}
if true {
break;
}
}
unsafe {
(*timer).max_instances = 1000;
}
// default limit per timer
kref_init(unsafe { core::ptr::addr_of_mut!((*timer).kref) });
if !card.is_null() {
unsafe {
(*timer).module = (*card).module;
}
err = unsafe { snd_device_new(card, 7u32, timer as *mut (), core::ptr::addr_of!(ops.0)) };
if err < 0 {
snd_timer_free(timer);
return err;
}
}
if !rtimer.is_null() {
unsafe {
*rtimer = timer;
}
}
0
}

static ops: __SlateConst<snd_device_ops> = __SlateConst(snd_device_ops { dev_free: Some(snd_timer_dev_free), dev_register: Some(snd_timer_dev_register), dev_disconnect: Some(snd_timer_dev_disconnect) });

static mut __key__1020: lock_class_key = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timer_new_529: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_card, *mut u8, *mut snd_timer_id, *mut *mut snd_timer) -> i32>, *mut ()>(Some(snd_timer_new)) };

/// refcount management of timer object
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_kref_release")]
extern "C" fn snd_timer_kref_release(kref: *mut kref) {
let timer: *mut snd_timer = container_of__30!(kref, snd_timer);
if (unsafe { (*timer).private_free }) != None {
unsafe { unsafe { (*timer).private_free }.unwrap()(timer) };
}
unsafe { kfree(timer as *const ()) };
}

fn snd_timer_free(timer: *mut snd_timer) -> i32 {
let mut card_dev_to_put: *mut device = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_unlock_531: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_531) as *mut *mut __SlateRecord_mutex_h_253;
let mut scope: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_mutex_h_253;
let mut n: core::mem::MaybeUninit<*mut snd_timer_instance> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut snd_timer_instance = core::ptr::addr_of_mut!(n) as *mut *mut snd_timer_instance;
let mut ti: core::mem::MaybeUninit<*mut snd_timer_instance> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut snd_timer_instance = core::ptr::addr_of_mut!(ti) as *mut *mut snd_timer_instance;
let __slate_cleanup_ptr_4 = __slate_slot_4;
let __slate_cleanup_armed_4 = core::cell::Cell::new(false);
let __slate_cleanup_4 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_4.get() {
unsafe { class_mutex_destructor(__slate_cleanup_ptr_4 as *mut _) };
}
});
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { __class_mutex_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if timer.is_null() {
return 0;
} else {
core::ptr::write(__slate_slot_4, class_mutex_constructor(core::ptr::addr_of_mut!(register_mutex)));
__slate_cleanup_armed_4.set(true);
core::ptr::write(__slate_slot_3, (core::ptr::addr_of_mut!(register_mutex) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_3.set(true);
if !class_mutex_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_4))).is_null() || !class_mutex_is_conditional {
if true {
if list_empty(core::ptr::addr_of_mut!((*timer).open_list_head) as *const list_head) == 0 {
*__slate_slot_6 = list_first_entry__4!(core::ptr::addr_of_mut!((*timer).open_list_head));
let __v2: *mut snd_timer_instance = list_next_entry__2!(*__slate_slot_6, open_list);
*__slate_slot_5 = __v2;
while list_entry_is_head!(*__slate_slot_6, core::ptr::addr_of_mut!((*timer).open_list_head), open_list) == 0 {
card_dev_to_put = core::ptr::null_mut::<device>();
snd_timer_close_locked(*__slate_slot_6, &mut (*core::ptr::addr_of_mut!(card_dev_to_put)));
put_device(card_dev_to_put);
*__slate_slot_6 = *__slate_slot_5;
let __v0: *mut snd_timer_instance = list_next_entry__2!(*__slate_slot_5, open_list);
*__slate_slot_5 = __v0;
}
}
list_del(core::ptr::addr_of_mut!((*timer).device_list));
}
}
__slate_cleanup_armed_3.set(false);
__class_mutex_cleanup_ctx(__slate_slot_3 as *mut _);
__slate_cleanup_armed_4.set(false);
class_mutex_destructor(__slate_slot_4 as *mut _);
disable_work_sync(core::ptr::addr_of_mut!((*timer).task_work));
snd_timer_ref_put(timer);
return 0;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_dev_free")]
extern "C" fn snd_timer_dev_free(device: *mut snd_device) -> i32 {
let timer: *mut snd_timer = (unsafe { (*device).device_data }) as *mut snd_timer;
snd_timer_free(timer)
}

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_dev_register")]
extern "C" fn snd_timer_dev_register(dev: *mut snd_device) -> i32 {
let timer: *mut snd_timer = (unsafe { (*dev).device_data }) as *mut snd_timer;
let mut timer1: *mut snd_timer = unsafe { core::mem::zeroed() };
let mut insert_before: *mut list_head = unsafe { core::ptr::addr_of_mut!(snd_timer_list) };
if snd_BUG_ON!(timer.is_null() || (unsafe { (*timer).hw.start }) == None || (unsafe { (*timer).hw.stop }) == None) != 0 {
return 6i32.wrapping_neg();
}
if (unsafe { (*timer).hw.flags }) & 4 == 0 && (unsafe { (*timer).hw.resolution }) == 0 && (unsafe { (*timer).hw.c_resolution }) == None {
return 22i32.wrapping_neg();
}
let mut __UNIQUE_ID_guard_532: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!(register_mutex) });
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_532) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_2 as *mut _));
let mut __UNIQUE_ID_unlock_533: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!(register_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_533) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _));
timer1 = list_first_entry__10!(unsafe { core::ptr::addr_of_mut!(snd_timer_list) });
'__slate_break_5: while list_entry_is_head!(timer1, unsafe { core::ptr::addr_of_mut!(snd_timer_list) }, device_list) == 0 {
'__slate_continue_5: {
if (unsafe { (*timer1).tmr_class }) > unsafe { (*timer).tmr_class } {
insert_before = unsafe { core::ptr::addr_of_mut!((*timer1).device_list) };
break '__slate_break_5;
}
if !((unsafe { (*timer1).tmr_class }) < unsafe { (*timer).tmr_class }) {
if !unsafe { (*timer1).card }.is_null() && !unsafe { (*timer).card }.is_null() {
if (unsafe { (*(*timer1).card).number }) > unsafe { (*(*timer).card).number } {
insert_before = unsafe { core::ptr::addr_of_mut!((*timer1).device_list) };
break '__slate_break_5;
}
if (unsafe { (*(*timer1).card).number }) < unsafe { (*(*timer).card).number } {
break '__slate_continue_5;
}
}
if (unsafe { (*timer1).tmr_device }) > unsafe { (*timer).tmr_device } {
insert_before = unsafe { core::ptr::addr_of_mut!((*timer1).device_list) };
break '__slate_break_5;
}
if !((unsafe { (*timer1).tmr_device }) < unsafe { (*timer).tmr_device }) {
if (unsafe { (*timer1).tmr_subdevice }) > unsafe { (*timer).tmr_subdevice } {
insert_before = unsafe { core::ptr::addr_of_mut!((*timer1).device_list) };
break '__slate_break_5;
}
if !((unsafe { (*timer1).tmr_subdevice }) < unsafe { (*timer).tmr_subdevice }) {
// conflicts..
return 16i32.wrapping_neg();
}
}
}
}
timer1 = list_next_entry__7!(timer1, device_list);
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*timer).device_list) }, insert_before);
0
}

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_dev_disconnect")]
extern "C" fn snd_timer_dev_disconnect(device: *mut snd_device) -> i32 {
let timer: *mut snd_timer = (unsafe { (*device).device_data }) as *mut snd_timer;
let mut ti: *mut snd_timer_instance = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_guard_534: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!(register_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_534) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_535: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!(register_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_535) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
list_del_init(unsafe { core::ptr::addr_of_mut!((*timer).device_list) });
// wake up pending sleepers
ti = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*timer).open_list_head) });
while list_entry_is_head!(ti, unsafe { core::ptr::addr_of_mut!((*timer).open_list_head) }, open_list) == 0 {
if (unsafe { (*ti).disconnect }) != None {
unsafe { unsafe { (*ti).disconnect }.unwrap()(ti) };
}
ti = list_next_entry__2!(ti, open_list);
}
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_notify")]
extern "C" fn snd_timer_notify(timer: *mut snd_timer, event: i32, tstamp: *mut timespec64) {
let mut resolution: u64 = 0;
let mut ti: *mut snd_timer_instance = unsafe { core::mem::zeroed() };
let mut ts: *mut snd_timer_instance = unsafe { core::mem::zeroed() };
if !unsafe { (*timer).card }.is_null() && (unsafe { (*(*timer).card).shutdown }) != 0 {
return;
}
if (unsafe { (*timer).hw.flags }) & 4 == 0 {
return;
}
if snd_BUG_ON!(event < 12 || event > 18) != 0 {
return;
}
let mut __UNIQUE_ID_guard_536: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!((*timer).lock) });
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_536) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_2 as *mut _));
let mut __UNIQUE_ID_unlock_537: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!((*timer).lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_537) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _));
if event == 12 || event == 14 || event == 18 {
resolution = snd_timer_hw_resolution(timer);
}
ti = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!((*timer).active_list_head) });
while list_entry_is_head!(ti, unsafe { core::ptr::addr_of_mut!((*timer).active_list_head) }, active_list) == 0 {
if (unsafe { (*ti).ccallback }) != None {
unsafe { unsafe { (*ti).ccallback }.unwrap()(ti, event, tstamp, resolution) };
}
ts = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!((*ti).slave_active_head) });
while list_entry_is_head!(ts, unsafe { core::ptr::addr_of_mut!((*ti).slave_active_head) }, active_list) == 0 {
if (unsafe { (*ts).ccallback }) != None {
unsafe { unsafe { (*ts).ccallback }.unwrap()(ts, event, tstamp, resolution) };
}
ts = list_next_entry__5!(ts, active_list);
}
ti = list_next_entry__5!(ti, active_list);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timer_notify_538: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_timer, i32, *mut timespec64)>, *mut ()>(Some(snd_timer_notify)) };

/// exported functions for global timers
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_global_new")]
extern "C" fn snd_timer_global_new(id: *mut u8, device: i32, rtimer: *mut *mut snd_timer) -> i32 {
let mut tid: snd_timer_id = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tid) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 20usize);
}
tid.dev_class = 1;
tid.dev_sclass = 0;
tid.card = 1i32.wrapping_neg();
tid.device = device;
tid.subdevice = 0;
snd_timer_new(core::ptr::null_mut::<snd_card>(), id, core::ptr::addr_of_mut!(tid), rtimer)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timer_global_new_539: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut u8, i32, *mut *mut snd_timer) -> i32>, *mut ()>(Some(snd_timer_global_new)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_global_free")]
extern "C" fn snd_timer_global_free(timer: *mut snd_timer) -> i32 {
snd_timer_free(timer)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timer_global_free_540: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_timer) -> i32>, *mut ()>(Some(snd_timer_global_free)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_global_register")]
extern "C" fn snd_timer_global_register(timer: *mut snd_timer) -> i32 {
let mut dev: snd_device = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(dev) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
unsafe { memset(core::ptr::addr_of_mut!(dev) as *mut (), 0i32, core::mem::size_of::<snd_device>() as u64) };
dev.device_data = timer as *mut ();
snd_timer_dev_register(core::ptr::addr_of_mut!(dev))
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_timer_global_register_541: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_timer) -> i32>, *mut ()>(Some(snd_timer_global_register)) };

// System timer
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_system_private {
    tlist: timer_list,
    snd_timer: *mut snd_timer,
    last_expires: u64,
    last_jiffies: u64,
    correction: u64,
}


#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_s_function")]
extern "C" fn snd_timer_s_function(t: *mut timer_list) {
let r#priv: *mut snd_timer_system_private = timer_container_of!(t);
let timer: *mut snd_timer = unsafe { (*r#priv).snd_timer };
let jiff: u64 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) };
if time_after!(jiff, unsafe { (*r#priv).last_expires }) {
let __v9: u64 = unsafe { (*r#priv).correction };
unsafe {
(*r#priv).correction = __v9.wrapping_add((jiff as i64).wrapping_sub((*r#priv).last_expires as i64) as u64);
}
}
snd_timer_interrupt(timer, (jiff as i64).wrapping_sub((unsafe { (*r#priv).last_jiffies }) as i64) as u64);
}

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_s_start")]
extern "C" fn snd_timer_s_start(timer: *mut snd_timer) -> i32 {
let mut r#priv: *mut snd_timer_system_private = unsafe { core::mem::zeroed() };
let mut njiff: u64 = 0;
r#priv = (unsafe { (*timer).private_data }) as *mut snd_timer_system_private;
let __v0: u64 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) };
unsafe {
(*r#priv).last_jiffies = __v0;
}
njiff = __v0;
if (unsafe { (*r#priv).correction }) > unsafe { (*timer).sticks }.wrapping_sub(1) {
let __v2: u64 = unsafe { (*r#priv).correction };
unsafe {
(*r#priv).correction = __v2.wrapping_sub(unsafe { (*timer).sticks }.wrapping_sub(1));
}
njiff = njiff.wrapping_add(1);
} else {
njiff = njiff.wrapping_add(unsafe { (*timer).sticks }.wrapping_sub(unsafe { (*r#priv).correction }));
unsafe {
(*r#priv).correction = 0;
}
}
unsafe {
(*r#priv).last_expires = njiff;
}
unsafe { mod_timer(core::ptr::addr_of_mut!((*r#priv).tlist), njiff) };
0
}

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_s_stop")]
extern "C" fn snd_timer_s_stop(timer: *mut snd_timer) -> i32 {
let mut r#priv: *mut snd_timer_system_private = unsafe { core::mem::zeroed() };
let mut jiff: u64 = 0;
r#priv = (unsafe { (*timer).private_data }) as *mut snd_timer_system_private;
unsafe { timer_delete(core::ptr::addr_of_mut!((*r#priv).tlist)) };
jiff = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) };
if time_before__2!(jiff, unsafe { (*r#priv).last_expires }) {
unsafe {
(*timer).sticks = unsafe { (*r#priv).last_expires }.wrapping_sub(jiff);
}
} else {
unsafe {
(*timer).sticks = 1;
}
}
unsafe {
(*r#priv).correction = 0;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_s_close")]
extern "C" fn snd_timer_s_close(timer: *mut snd_timer) -> i32 {
let mut r#priv: *mut snd_timer_system_private = unsafe { core::mem::zeroed() };
r#priv = (unsafe { (*timer).private_data }) as *mut snd_timer_system_private;
unsafe { timer_delete_sync(core::ptr::addr_of_mut!((*r#priv).tlist)) };
0
}

static snd_timer_system: __SlateConst<snd_timer_hardware> = __SlateConst({
let mut __t0: snd_timer_hardware = unsafe { core::mem::zeroed() };
__t0.flags = 24u32;
__t0.resolution = 1000000u64;
__t0.ticks = 10000000u64;
__t0.close = Some(snd_timer_s_close);
__t0.start = Some(snd_timer_s_start);
__t0.stop = Some(snd_timer_s_stop);
__t0
});

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_free_system")]
extern "C" fn snd_timer_free_system(timer: *mut snd_timer) {
unsafe { kfree((*timer).private_data as *const ()) };
}

fn snd_timer_register_system() -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut timer: *mut snd_timer = unsafe { core::mem::zeroed() };
let mut r#priv: *mut snd_timer_system_private = unsafe { core::mem::zeroed() };
let err: i32 = snd_timer_global_new(c"system".as_ptr() as *mut u8, 0i32, core::ptr::addr_of_mut!(timer));
if err < 0 {
return err;
}
unsafe { sized_strscpy((*timer).name.as_mut_ptr() as *mut u8, c"system timer".as_ptr() as *const u8, (core::mem::size_of::<[u8; 80]>() as u64).wrapping_add(__must_be_array__51!() as u64).wrapping_add(__must_be_cstr__95!() as u64).wrapping_add(__must_be_cstr__38!() as u64)) };
unsafe {
(*timer).hw = snd_timer_system.0;
}
let mut __v0: *mut snd_timer_system_private;
let __obj_size: u64 = size_mul(core::mem::size_of::<snd_timer_system_private>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut snd_timer_system_private;
r#priv = __v0;
if r#priv.is_null() {
snd_timer_free(timer);
return 12i32.wrapping_neg();
}
unsafe {
(*r#priv).snd_timer = timer;
}
timer_setup!(unsafe { core::ptr::addr_of_mut!((*r#priv).tlist) }, Some(snd_timer_s_function), 0i32);
unsafe {
(*timer).private_data = r#priv as *mut ();
}
unsafe {
(*timer).private_free = Some(snd_timer_free_system);
}
snd_timer_global_register(timer)
}

// Info interface
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_proc_read")]
extern "C" fn snd_timer_proc_read(entry: *mut snd_info_entry, buffer: *mut snd_info_buffer) {
let mut __UNIQUE_ID_unlock_545: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_545) as *mut *mut __SlateRecord_spinlock_h_620;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_620;
let mut __UNIQUE_ID_unlock_543: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_543) as *mut *mut __SlateRecord_mutex_h_253;
let mut __UNIQUE_ID_guard_542: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_542) as *mut __SlateRecord_mutex_h_253;
let mut resolution: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u64 = core::ptr::addr_of_mut!(resolution) as *mut u64;
let mut ti: core::mem::MaybeUninit<*mut snd_timer_instance> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut snd_timer_instance = core::ptr::addr_of_mut!(ti) as *mut *mut snd_timer_instance;
let mut timer: core::mem::MaybeUninit<*mut snd_timer> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut snd_timer = core::ptr::addr_of_mut!(timer) as *mut *mut snd_timer;
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { class_mutex_destructor(__slate_cleanup_ptr_3 as *mut _) };
}
});
let __slate_cleanup_ptr_2 = __slate_slot_2;
let __slate_cleanup_armed_2 = core::cell::Cell::new(false);
let __slate_cleanup_2 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_2.get() {
unsafe { __class_mutex_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _) };
}
});
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_spinlock_irq_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_spinlock_irq_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_3, class_mutex_constructor(core::ptr::addr_of_mut!(register_mutex)));
__slate_cleanup_armed_3.set(true);
core::ptr::write(__slate_slot_2, (core::ptr::addr_of_mut!(register_mutex) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_2.set(true);
*__slate_slot_6 = list_first_entry__10!(core::ptr::addr_of_mut!(snd_timer_list));
while list_entry_is_head!(*__slate_slot_6, core::ptr::addr_of_mut!(snd_timer_list), device_list) == 0 {
if !(!unsafe { (*(*__slate_slot_6)).card }.is_null() && (*(*(*__slate_slot_6)).card).shutdown != 0) {
let __t0: i32 = (*(*__slate_slot_6)).tmr_class;
if __t0 == 1 {
seq_printf((*buffer).buffer as *mut seq_file, c"G%i: ".as_ptr() as *const u8, (*(*__slate_slot_6)).tmr_device);
} else {
if __t0 == 2 {
seq_printf((*buffer).buffer as *mut seq_file, c"C%i-%i: ".as_ptr() as *const u8, (*(*(*__slate_slot_6)).card).number, (*(*__slate_slot_6)).tmr_device);
} else {
if __t0 == 3 {
seq_printf((*buffer).buffer as *mut seq_file, c"P%i-%i-%i: ".as_ptr() as *const u8, (*(*(*__slate_slot_6)).card).number, (*(*__slate_slot_6)).tmr_device, (*(*__slate_slot_6)).tmr_subdevice);
} else {
seq_printf((*buffer).buffer as *mut seq_file, c"?%i-%i-%i-%i: ".as_ptr() as *const u8, (*(*__slate_slot_6)).tmr_class, if !unsafe { (*(*__slate_slot_6)).card }.is_null() { (*(*(*__slate_slot_6)).card).number } else { 1i32.wrapping_neg() }, (*(*__slate_slot_6)).tmr_device, (*(*__slate_slot_6)).tmr_subdevice);
}
}
}
seq_printf((*buffer).buffer as *mut seq_file, c"%s :".as_ptr() as *const u8, (*(*__slate_slot_6)).name.as_mut_ptr() as *mut u8);
core::ptr::write(__slate_slot_1, class_spinlock_irq_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_6)).lock)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!((*(*__slate_slot_6)).lock) as u64) as *mut __SlateRecord_spinlock_h_620);
__slate_cleanup_armed_0.set(true);
if !class_spinlock_irq_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_spinlock_irq_is_conditional {
if true {
*__slate_slot_4 = snd_timer_hw_resolution(*__slate_slot_6);
}
}
__slate_cleanup_armed_0.set(false);
__class_spinlock_irq_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_spinlock_irq_destructor(__slate_slot_1 as *mut _);
if *__slate_slot_4 != 0 {
seq_printf((*buffer).buffer as *mut seq_file, c" %lu.%03luus (%lu ticks)".as_ptr() as *const u8, *__slate_slot_4 / 1000, *__slate_slot_4 % 1000, (*(*__slate_slot_6)).hw.ticks);
}
if (*(*__slate_slot_6)).hw.flags & 4 != 0 {
seq_printf((*buffer).buffer as *mut seq_file, c" SLAVE".as_ptr() as *const u8);
}
seq_printf((*buffer).buffer as *mut seq_file, c"\n".as_ptr() as *const u8);
*__slate_slot_5 = list_first_entry__4!(core::ptr::addr_of_mut!((*(*__slate_slot_6)).open_list_head));
while list_entry_is_head!(*__slate_slot_5, core::ptr::addr_of_mut!((*(*__slate_slot_6)).open_list_head), open_list) == 0 {
seq_printf((*buffer).buffer as *mut seq_file, c"  Client %s : %s\n".as_ptr() as *const u8, if !unsafe { (*(*__slate_slot_5)).owner }.is_null() { (*(*__slate_slot_5)).owner } else { c"unknown".as_ptr() as *mut u8 }, if (*(*__slate_slot_5)).flags & 6u32 != 0 { c"running".as_ptr() as *mut u8 } else { c"stopped".as_ptr() as *mut u8 });
*__slate_slot_5 = list_next_entry__2!(*__slate_slot_5, open_list);
}
}
*__slate_slot_6 = list_next_entry__7!(*__slate_slot_6, device_list);
}
__slate_cleanup_armed_2.set(false);
__class_mutex_cleanup_ctx(__slate_slot_2 as *mut _);
__slate_cleanup_armed_3.set(false);
class_mutex_destructor(__slate_slot_3 as *mut _);
}
}

static mut snd_timer_proc_entry: *mut snd_info_entry = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.text")]
fn snd_timer_proc_init() {
let mut entry: *mut snd_info_entry = unsafe { core::mem::zeroed() };
entry = unsafe { snd_info_create_module_entry(core::ptr::null_mut::<module>(), c"timers".as_ptr() as *const u8, core::ptr::null_mut::<snd_info_entry>()) };
if !entry.is_null() {
unsafe {
(*entry).c.text.read = Some(snd_timer_proc_read);
}
if (unsafe { snd_info_register(entry) }) < 0 {
unsafe { snd_info_free_entry(entry) };
entry = core::ptr::null_mut::<snd_info_entry>();
}
}
unsafe {
snd_timer_proc_entry = entry;
}
}

#[unsafe(link_section = ".exit.text")]
fn snd_timer_proc_done() {
unsafe { snd_info_free_entry(snd_timer_proc_entry) };
}

// USER SPACE interface
#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_user_interrupt")]
extern "C" fn snd_timer_user_interrupt(timeri: *mut snd_timer_instance, resolution: u64, ticks: u64) {
let mut __UNIQUE_ID_unlock_547: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_610> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_547) as *mut *mut __SlateRecord_spinlock_h_610;
let mut __UNIQUE_ID_guard_546: core::mem::MaybeUninit<__SlateRecord_spinlock_h_610> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_546) as *mut __SlateRecord_spinlock_h_610;
let mut prev: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(prev) as *mut i32;
let mut r: core::mem::MaybeUninit<*mut snd_timer_read> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut snd_timer_read = core::ptr::addr_of_mut!(r) as *mut *mut snd_timer_read;
let mut tu: *mut snd_timer_user = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_16 = __slate_slot_16;
let __slate_cleanup_armed_16 = core::cell::Cell::new(false);
let __slate_cleanup_16 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_16.get() {
unsafe { class_spinlock_destructor(__slate_cleanup_ptr_16 as *mut _) };
}
});
let __slate_cleanup_ptr_15 = __slate_slot_15;
let __slate_cleanup_armed_15 = core::cell::Cell::new(false);
let __slate_cleanup_15 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_15.get() {
unsafe { __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_15 as *mut _) };
}
});
unsafe {
'__join_0: {
tu = (*timeri).callback_data as *mut snd_timer_user;
let __slate_auto_bytes = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
core::ptr::write(__slate_slot_16, class_spinlock_constructor(core::ptr::addr_of_mut!((*tu).qlock)));
__slate_cleanup_armed_16.set(true);
core::ptr::write(__slate_slot_15, (core::ptr::addr_of_mut!((*tu).qlock) as u64) as *mut __SlateRecord_spinlock_h_610);
__slate_cleanup_armed_15.set(true);
if (*tu).qused > 0 {
*__slate_slot_17 = if (*tu).qtail == 0 { unsafe { (*tu).queue_size }.wrapping_sub(1) } else { unsafe { (*tu).qtail }.wrapping_sub(1) };
*__slate_slot_18 = unsafe { (*tu).queue }.wrapping_offset(*__slate_slot_17 as isize);
if ((*(*__slate_slot_18)).resolution as u64) == resolution {
let __v14: *mut snd_timer_read = *__slate_slot_18;
(*__v14).ticks = ((*__v14).ticks as u64).wrapping_add(ticks) as u32;
break '__join_0;
}
}
if (*tu).qused >= (*tu).queue_size {
let __v1: u64 = (*tu).overrun;
(*tu).overrun = __v1.wrapping_add(1);
} else {
let __v10: i32 = (*tu).qtail;
(*tu).qtail = __v10.wrapping_add(1);
*__slate_slot_18 = unsafe { (*tu).queue }.wrapping_offset(__v10 as isize);
(*tu).qtail = (*tu).qtail % (*tu).queue_size;
(*(*__slate_slot_18)).resolution = resolution as u32;
(*(*__slate_slot_18)).ticks = ticks as u32;
let __v4: i32 = (*tu).qused;
(*tu).qused = __v4.wrapping_add(1);
}
}
snd_kill_fasync((*tu).fasync, 29i32, 1i32);
wake_up!(core::ptr::addr_of_mut!((*tu).qchange_sleep));
__slate_cleanup_armed_15.set(false);
__class_spinlock_cleanup_ctx(__slate_slot_15 as *mut _);
__slate_cleanup_armed_16.set(false);
class_spinlock_destructor(__slate_slot_16 as *mut _);
}
}

fn snd_timer_user_append_to_tqueue(tu: *mut snd_timer_user, tread: *mut snd_timer_tread64) {
if (unsafe { (*tu).qused }) >= unsafe { (*tu).queue_size } {
let __v1: u64 = unsafe { (*tu).overrun };
unsafe {
(*tu).overrun = __v1.wrapping_add(1);
}
} else {
let __v4: i32 = unsafe { (*tu).qtail };
unsafe {
(*tu).qtail = __v4.wrapping_add(1);
}
unsafe { memcpy(unsafe { (*tu).tqueue }.wrapping_offset(__v4 as isize) as *mut (), tread as *const (), core::mem::size_of::<snd_timer_tread64>() as u64) };
unsafe {
(*tu).qtail = (*tu).qtail % (*tu).queue_size;
}
let __v10: i32 = unsafe { (*tu).qused };
unsafe {
(*tu).qused = __v10.wrapping_add(1);
}
}
}

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_user_ccallback")]
extern "C" fn snd_timer_user_ccallback(timeri: *mut snd_timer_instance, event: i32, tstamp: *mut timespec64, resolution: u64) {
let mut __UNIQUE_ID_unlock_549: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_549) as *mut *mut __SlateRecord_spinlock_h_642;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_642;
let mut r1: core::mem::MaybeUninit<snd_timer_tread64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut snd_timer_tread64 = core::ptr::addr_of_mut!(r1) as *mut snd_timer_tread64;
let mut tu: *mut snd_timer_user = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_spinlock_irqsave_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
tu = (*timeri).callback_data as *mut snd_timer_user;
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
if event >= 2 && event <= 5 {
(*tu).tstamp = *tstamp;
}
if (*tu).filter & ((1i32 << event) as u32) == 0 || (*tu).tread == 0 {
return;
} else {
memset(core::ptr::addr_of_mut!(*__slate_slot_2) as *mut (), 0i32, core::mem::size_of::<snd_timer_tread64>() as u64);
(*__slate_slot_2).event = event;
(*__slate_slot_2).tstamp_sec = (*tstamp).tv_sec;
(*__slate_slot_2).tstamp_nsec = (*tstamp).tv_nsec;
(*__slate_slot_2).val = resolution as u32;
core::ptr::write(__slate_slot_1, class_spinlock_irqsave_constructor(core::ptr::addr_of_mut!((*tu).qlock)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!((*tu).qlock) as u64) as *mut __SlateRecord_spinlock_h_642);
__slate_cleanup_armed_0.set(true);
if !class_spinlock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_spinlock_irqsave_is_conditional {
if true {
snd_timer_user_append_to_tqueue(tu, core::ptr::addr_of_mut!(*__slate_slot_2));
}
}
__slate_cleanup_armed_0.set(false);
__class_spinlock_irqsave_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_spinlock_irqsave_destructor(__slate_slot_1 as *mut _);
snd_kill_fasync((*tu).fasync, 29i32, 1i32);
wake_up!(core::ptr::addr_of_mut!((*tu).qchange_sleep));
}
}
}

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_user_disconnect")]
extern "C" fn snd_timer_user_disconnect(timeri: *mut snd_timer_instance) {
let tu: *mut snd_timer_user = (unsafe { (*timeri).callback_data }) as *mut snd_timer_user;
unsafe {
(*tu).disconnected = true;
}
wake_up!(unsafe { core::ptr::addr_of_mut!((*tu).qchange_sleep) });
}

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_user_tinterrupt")]
extern "C" fn snd_timer_user_tinterrupt(timeri: *mut snd_timer_instance, resolution: u64, ticks: u64) {
let mut __UNIQUE_ID_unlock_551: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_610> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_551) as *mut *mut __SlateRecord_spinlock_h_610;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_610> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_610;
let mut append: i32 = unsafe { core::mem::zeroed() };
let mut prev: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(prev) as *mut i32;
let mut tstamp: core::mem::MaybeUninit<timespec64> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut timespec64 = core::ptr::addr_of_mut!(tstamp) as *mut timespec64;
let mut r1: core::mem::MaybeUninit<snd_timer_tread64> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut snd_timer_tread64 = core::ptr::addr_of_mut!(r1) as *mut snd_timer_tread64;
let mut r: core::mem::MaybeUninit<*mut snd_timer_tread64> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut snd_timer_tread64 = core::ptr::addr_of_mut!(r) as *mut *mut snd_timer_tread64;
let mut tu: *mut snd_timer_user = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_10 = __slate_slot_10;
let __slate_cleanup_armed_10 = core::cell::Cell::new(false);
let __slate_cleanup_10 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_10.get() {
unsafe { class_spinlock_destructor(__slate_cleanup_ptr_10 as *mut _) };
}
});
let __slate_cleanup_ptr_9 = __slate_slot_9;
let __slate_cleanup_armed_9 = core::cell::Cell::new(false);
let __slate_cleanup_9 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_9.get() {
unsafe { __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_9 as *mut _) };
}
});
unsafe {
'__join_2: {
tu = (*timeri).callback_data as *mut snd_timer_user;
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 32usize);
let __slate_auto_bytes_2 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 16usize);
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
append = 0;
memset(core::ptr::addr_of_mut!(*__slate_slot_14) as *mut (), 0i32, core::mem::size_of::<snd_timer_tread64>() as u64);
memset(core::ptr::addr_of_mut!(*__slate_slot_13) as *mut (), 0i32, core::mem::size_of::<timespec64>() as u64);
core::ptr::write(__slate_slot_10, class_spinlock_constructor(core::ptr::addr_of_mut!((*tu).qlock)));
__slate_cleanup_armed_10.set(true);
core::ptr::write(__slate_slot_9, (core::ptr::addr_of_mut!((*tu).qlock) as u64) as *mut __SlateRecord_spinlock_h_610);
__slate_cleanup_armed_9.set(true);
if !class_spinlock_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_10))).is_null() || !class_spinlock_is_conditional {
if true {
if (*tu).filter & 3u32 == 0 {
return;
} else {
if (*tu).last_resolution != resolution || ticks > 0 {
if timer_tstamp_monotonic != 0 {
ktime_get_ts64(core::ptr::addr_of_mut!(*__slate_slot_13));
} else {
ktime_get_real_ts64(core::ptr::addr_of_mut!(*__slate_slot_13));
}
}
if (*tu).filter & 1u32 != 0 && (*tu).last_resolution != resolution {
(*__slate_slot_14).event = 0;
(*__slate_slot_14).tstamp_sec = (*__slate_slot_13).tv_sec;
(*__slate_slot_14).tstamp_nsec = (*__slate_slot_13).tv_nsec;
(*__slate_slot_14).val = resolution as u32;
snd_timer_user_append_to_tqueue(tu, core::ptr::addr_of_mut!(*__slate_slot_14));
(*tu).last_resolution = resolution;
append = append.wrapping_add(1);
}
if (*tu).filter & 2u32 != 0 {
if ticks != 0 {
if (*tu).qused > 0 {
*__slate_slot_12 = if (*tu).qtail == 0 { unsafe { (*tu).queue_size }.wrapping_sub(1) } else { unsafe { (*tu).qtail }.wrapping_sub(1) };
*__slate_slot_15 = unsafe { (*tu).tqueue }.wrapping_offset(*__slate_slot_12 as isize);
if (*(*__slate_slot_15)).event == 1 {
(*(*__slate_slot_15)).tstamp_sec = (*__slate_slot_13).tv_sec;
(*(*__slate_slot_15)).tstamp_nsec = (*__slate_slot_13).tv_nsec;
let __v6: *mut snd_timer_tread64 = *__slate_slot_15;
(*__v6).val = ((*__v6).val as u64).wrapping_add(ticks) as u32;
append = append.wrapping_add(1);
break '__join_2;
}
}
(*__slate_slot_14).event = 1;
(*__slate_slot_14).tstamp_sec = (*__slate_slot_13).tv_sec;
(*__slate_slot_14).tstamp_nsec = (*__slate_slot_13).tv_nsec;
(*__slate_slot_14).val = ticks as u32;
snd_timer_user_append_to_tqueue(tu, core::ptr::addr_of_mut!(*__slate_slot_14));
append = append.wrapping_add(1);
}
}
}
}
}
}
__slate_cleanup_armed_9.set(false);
__class_spinlock_cleanup_ctx(__slate_slot_9 as *mut _);
__slate_cleanup_armed_10.set(false);
class_spinlock_destructor(__slate_slot_10 as *mut _);
if append == 0 {
return;
} else {
snd_kill_fasync((*tu).fasync, 29i32, 1i32);
wake_up!(core::ptr::addr_of_mut!((*tu).qchange_sleep));
}
}
}

fn realloc_user_queue(tu: *mut snd_timer_user, size: i32) -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut queue: *mut snd_timer_read = core::ptr::null_mut::<snd_timer_read>();
let mut tqueue: *mut snd_timer_tread64 = core::ptr::null_mut::<snd_timer_tread64>();
if (unsafe { (*tu).tread }) != 0 {
let mut __v0: *mut snd_timer_tread64;
let __obj_size: u64 = size_mul(core::mem::size_of::<snd_timer_tread64>() as u64, size as u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut snd_timer_tread64;
tqueue = __v0;
if tqueue.is_null() {
return 12i32.wrapping_neg();
}
} else {
let mut __v6: *mut snd_timer_read;
let __obj_size: u64 = size_mul(core::mem::size_of::<snd_timer_read>() as u64, size as u64);
let mut __v7: *mut ();
let mut __v8: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
} });
}
__v8 = _res;
__v7 = __v8;
__v6 = __v7 as *mut snd_timer_read;
queue = __v6;
if queue.is_null() {
return 12i32.wrapping_neg();
}
}
let mut __UNIQUE_ID_guard_552: __SlateRecord_spinlock_h_620 = class_spinlock_irq_constructor(unsafe { core::ptr::addr_of_mut!((*tu).qlock) });
let __slate_cleanup_ptr_12 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_552) as *mut _;
let __slate_cleanup_12 = ::slate_support::Guard(&mut || class_spinlock_irq_destructor(__slate_cleanup_ptr_12 as *mut _));
let mut __UNIQUE_ID_unlock_553: *mut __SlateRecord_spinlock_h_620 = ((unsafe { core::ptr::addr_of_mut!((*tu).qlock) }) as u64) as *mut __SlateRecord_spinlock_h_620;
let __slate_cleanup_ptr_13 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_553) as *mut _;
let __slate_cleanup_13 = ::slate_support::Guard(&mut || __class_spinlock_irq_cleanup_ctx(__slate_cleanup_ptr_13 as *mut _));
unsafe { kfree((*tu).queue as *const ()) };
unsafe { kfree((*tu).tqueue as *const ()) };
unsafe {
(*tu).queue_size = size;
}
unsafe {
(*tu).queue = queue;
}
unsafe {
(*tu).tqueue = tqueue;
}
unsafe {
(*tu).qused = 0;
}
unsafe {
(*tu).qtail = 0;
}
unsafe {
(*tu).qhead = 0;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_user_open")]
extern "C" fn snd_timer_user_open(inode: *mut inode, file: *mut file) -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: spinlock = unsafe { core::mem::zeroed() };
let mut tu: *mut snd_timer_user = unsafe { core::mem::zeroed() };
unsafe { stream_open(inode, file) };
let mut __v0: *mut snd_timer_user;
let __obj_size: u64 = size_mul(core::mem::size_of::<snd_timer_user>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut snd_timer_user;
tu = __v0;
if tu.is_null() {
return 12i32.wrapping_neg();
}
'__slate_break_7: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*tu).qlock) });
unsafe {
*core::ptr::addr_of_mut!((*tu).qlock) = *{
__slate_compound_6 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_6)
};
}
if true {
break;
}
}
loop {
unsafe { __init_waitqueue_head(core::ptr::addr_of_mut!((*tu).qchange_sleep), c"&tu->qchange_sleep".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__1556)) };
if true {
break;
}
}
loop {
__mutex_init(unsafe { core::ptr::addr_of_mut!((*tu).ioctl_lock) }, c"&tu->ioctl_lock".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key__1557) });
if true {
break;
}
}
unsafe {
(*tu).ticks = 1;
}
if realloc_user_queue(tu, 128i32) < 0 {
unsafe { kfree(tu as *const ()) };
return 12i32.wrapping_neg();
}
unsafe {
(*file).private_data = tu as *mut ();
}
0
}

static mut __key__1556: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__1557: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_user_release")]
extern "C" fn snd_timer_user_release(inode: *mut inode, file: *mut file) -> i32 {
let mut __UNIQUE_ID_unlock_555: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_555) as *mut *mut __SlateRecord_mutex_h_253;
let mut scope: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_mutex_h_253;
let mut tu: core::mem::MaybeUninit<*mut snd_timer_user> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut snd_timer_user = core::ptr::addr_of_mut!(tu) as *mut *mut snd_timer_user;
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_mutex_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_mutex_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if !unsafe { (*file).private_data }.is_null() {
*__slate_slot_2 = (*file).private_data as *mut snd_timer_user;
(*file).private_data = core::ptr::null_mut::<()>();
core::ptr::write(__slate_slot_1, class_mutex_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_2)).ioctl_lock)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!((*(*__slate_slot_2)).ioctl_lock) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_0.set(true);
if !class_mutex_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_mutex_is_conditional {
if true {
if !unsafe { (*(*__slate_slot_2)).timeri }.is_null() {
snd_timer_close((*(*__slate_slot_2)).timeri);
snd_timer_instance_free((*(*__slate_slot_2)).timeri);
}
}
}
__slate_cleanup_armed_0.set(false);
__class_mutex_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_mutex_destructor(__slate_slot_1 as *mut _);
snd_fasync_free((*(*__slate_slot_2)).fasync);
kfree((*(*__slate_slot_2)).queue as *const ());
kfree((*(*__slate_slot_2)).tqueue as *const ());
kfree(*__slate_slot_2 as *const ());
}
return 0;
}
}

fn snd_timer_user_zero_id(id: &mut snd_timer_id) {
unsafe {
(*(id as *mut snd_timer_id)).dev_class = -1;
}
unsafe {
(*(id as *mut snd_timer_id)).dev_sclass = 0;
}
unsafe {
(*(id as *mut snd_timer_id)).card = 1i32.wrapping_neg();
}
unsafe {
(*(id as *mut snd_timer_id)).device = 1i32.wrapping_neg();
}
unsafe {
(*(id as *mut snd_timer_id)).subdevice = 1i32.wrapping_neg();
}
}

fn snd_timer_user_copy_id(id: &mut snd_timer_id, timer: *mut snd_timer) {
unsafe {
(*(id as *mut snd_timer_id)).dev_class = (*timer).tmr_class;
}
unsafe {
(*(id as *mut snd_timer_id)).dev_sclass = 0;
}
unsafe {
(*(id as *mut snd_timer_id)).card = if !unsafe { (*timer).card }.is_null() { (*(*timer).card).number } else { 1i32.wrapping_neg() };
}
unsafe {
(*(id as *mut snd_timer_id)).device = (*timer).tmr_device;
}
unsafe {
(*(id as *mut snd_timer_id)).subdevice = (*timer).tmr_subdevice;
}
}

fn get_next_device(id: &mut snd_timer_id) {
let mut timer: *mut snd_timer = unsafe { core::mem::zeroed() };
let mut p: *mut list_head = unsafe { core::mem::zeroed() };
if (unsafe { (*(id as *mut snd_timer_id)).dev_class }) < 0 {
// first item
if list_empty((unsafe { core::ptr::addr_of_mut!(snd_timer_list) }) as *const list_head) != 0 {
snd_timer_user_zero_id(id);
} else {
timer = list_entry__7!(unsafe { snd_timer_list.next }, snd_timer);
snd_timer_user_copy_id(id, timer);
}
} else {
match unsafe { (*(id as *mut snd_timer_id)).dev_class } {
1 => {
unsafe {
(*(id as *mut snd_timer_id)).device = if (*(id as *mut snd_timer_id)).device < 0 { 0 } else { unsafe { (*(id as *mut snd_timer_id)).device }.wrapping_add(1) };
}
p = unsafe { (*core::ptr::addr_of_mut!(snd_timer_list)).next };
while list_is_head(p as *const list_head, (unsafe { core::ptr::addr_of_mut!(snd_timer_list) }) as *const list_head) == 0 {
timer = list_entry__7!(p, snd_timer);
if (unsafe { (*timer).tmr_class }) > 1 {
snd_timer_user_copy_id(id, timer);
break;
}
if (unsafe { (*timer).tmr_device }) >= unsafe { (*(id as *mut snd_timer_id)).device } {
snd_timer_user_copy_id(id, timer);
break;
}
p = unsafe { (*p).next };
}
if p == unsafe { core::ptr::addr_of_mut!(snd_timer_list) } {
snd_timer_user_zero_id(id);
}
}
2 | 3 => {
if (unsafe { (*(id as *mut snd_timer_id)).card }) < 0 {
unsafe {
(*(id as *mut snd_timer_id)).card = 0;
}
} else {
if (unsafe { (*(id as *mut snd_timer_id)).device }) < 0 {
unsafe {
(*(id as *mut snd_timer_id)).device = 0;
}
} else {
if (unsafe { (*(id as *mut snd_timer_id)).subdevice }) < 0 {
unsafe {
(*(id as *mut snd_timer_id)).subdevice = 0;
}
} else {
if (unsafe { (*(id as *mut snd_timer_id)).subdevice }) < 2147483647i32 {
let __v5: i32 = unsafe { (*id).subdevice };
unsafe {
(*id).subdevice = __v5.wrapping_add(1);
}
}
}
}
}
p = unsafe { (*core::ptr::addr_of_mut!(snd_timer_list)).next };
while list_is_head(p as *const list_head, (unsafe { core::ptr::addr_of_mut!(snd_timer_list) }) as *const list_head) == 0 {
timer = list_entry__7!(p, snd_timer);
if (unsafe { (*timer).tmr_class }) > unsafe { (*(id as *mut snd_timer_id)).dev_class } {
snd_timer_user_copy_id(id, timer);
break;
}
if !((unsafe { (*timer).tmr_class }) < unsafe { (*(id as *mut snd_timer_id)).dev_class }) {
if (unsafe { (*(*timer).card).number }) > unsafe { (*(id as *mut snd_timer_id)).card } {
snd_timer_user_copy_id(id, timer);
break;
}
if !((unsafe { (*(*timer).card).number }) < unsafe { (*(id as *mut snd_timer_id)).card }) {
if (unsafe { (*timer).tmr_device }) > unsafe { (*(id as *mut snd_timer_id)).device } {
snd_timer_user_copy_id(id, timer);
break;
}
if !((unsafe { (*timer).tmr_device }) < unsafe { (*(id as *mut snd_timer_id)).device }) {
if (unsafe { (*timer).tmr_subdevice }) > unsafe { (*(id as *mut snd_timer_id)).subdevice } {
snd_timer_user_copy_id(id, timer);
break;
}
if !((unsafe { (*timer).tmr_subdevice }) < unsafe { (*(id as *mut snd_timer_id)).subdevice }) {
snd_timer_user_copy_id(id, timer);
break;
}
}
}
}
p = unsafe { (*p).next };
}
if p == unsafe { core::ptr::addr_of_mut!(snd_timer_list) } {
snd_timer_user_zero_id(id);
}
}
_ => {
snd_timer_user_zero_id(id);
}
}
}
}

fn snd_timer_user_next_device(_tid: *mut snd_timer_id) -> i32 {
let mut __UNIQUE_ID_unlock_557: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_557) as *mut *mut __SlateRecord_mutex_h_253;
let mut scope: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_mutex_h_253;
let mut id: core::mem::MaybeUninit<snd_timer_id> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut snd_timer_id = core::ptr::addr_of_mut!(id) as *mut snd_timer_id;
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_mutex_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_mutex_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 20usize);
if copy_from_user(core::ptr::addr_of_mut!(*__slate_slot_2) as *mut (), _tid as *const (), core::mem::size_of::<snd_timer_id>() as u64) != 0 {
return 14i32.wrapping_neg();
} else {
core::ptr::write(__slate_slot_1, class_mutex_constructor(core::ptr::addr_of_mut!(register_mutex)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!(register_mutex) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_0.set(true);
if !class_mutex_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_mutex_is_conditional {
if true {
get_next_device(&mut (*core::ptr::addr_of_mut!(*__slate_slot_2)));
}
}
__slate_cleanup_armed_0.set(false);
__class_mutex_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_mutex_destructor(__slate_slot_1 as *mut _);
if copy_to_user(_tid as *mut (), core::ptr::addr_of_mut!(*__slate_slot_2) as *const (), core::mem::size_of::<snd_timer_id>() as u64) != 0 {
return 14i32.wrapping_neg();
} else {
return 0;
}
}
}
}

fn snd_timer_user_ginfo(file: *mut file, _ginfo: *mut snd_timer_ginfo) -> i32 {
let mut __UNIQUE_ID_unlock_561: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_561) as *mut *mut __SlateRecord_spinlock_h_620;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_620;
let mut __UNIQUE_ID_unlock_559: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_559) as *mut *mut __SlateRecord_mutex_h_253;
let mut scope: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_mutex_h_253;
let mut ginfo: core::mem::MaybeUninit<*mut snd_timer_ginfo> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut snd_timer_ginfo = core::ptr::addr_of_mut!(ginfo) as *mut *mut snd_timer_ginfo;
let mut p: core::mem::MaybeUninit<*mut list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut list_head = core::ptr::addr_of_mut!(p) as *mut *mut list_head;
let mut t: core::mem::MaybeUninit<*mut snd_timer> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut snd_timer = core::ptr::addr_of_mut!(t) as *mut *mut snd_timer;
let mut tid: core::mem::MaybeUninit<snd_timer_id> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut snd_timer_id = core::ptr::addr_of_mut!(tid) as *mut snd_timer_id;
let __slate_cleanup_ptr_10 = __slate_slot_10;
let __slate_cleanup_armed_10 = core::cell::Cell::new(false);
let __slate_cleanup_10 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_10.get() {
unsafe { __free_kfree(__slate_cleanup_ptr_10 as *mut _) };
}
});
let __slate_cleanup_ptr_9 = __slate_slot_9;
let __slate_cleanup_armed_9 = core::cell::Cell::new(false);
let __slate_cleanup_9 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_9.get() {
unsafe { class_mutex_destructor(__slate_cleanup_ptr_9 as *mut _) };
}
});
let __slate_cleanup_ptr_8 = __slate_slot_8;
let __slate_cleanup_armed_8 = core::cell::Cell::new(false);
let __slate_cleanup_8 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_8.get() {
unsafe { __class_mutex_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _) };
}
});
let __slate_cleanup_ptr_4 = __slate_slot_4;
let __slate_cleanup_armed_4 = core::cell::Cell::new(false);
let __slate_cleanup_4 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_4.get() {
unsafe { class_spinlock_irq_destructor(__slate_cleanup_ptr_4 as *mut _) };
}
});
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { __class_spinlock_irq_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 20usize);
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_10, memdup_user(_ginfo as *const (), core::mem::size_of::<snd_timer_ginfo>() as u64) as *mut snd_timer_ginfo);
__slate_cleanup_armed_10.set(true);
if IS_ERR(*__slate_slot_10 as *const ()) {
return PTR_ERR(*__slate_slot_10 as *const ()) as i32;
} else {
*__slate_slot_13 = (*(*__slate_slot_10)).tid;
memset(*__slate_slot_10 as *mut (), 0i32, core::mem::size_of::<snd_timer_ginfo>() as u64);
(*(*__slate_slot_10)).tid = *__slate_slot_13;
core::ptr::write(__slate_slot_9, class_mutex_constructor(core::ptr::addr_of_mut!(register_mutex)));
__slate_cleanup_armed_9.set(true);
core::ptr::write(__slate_slot_8, (core::ptr::addr_of_mut!(register_mutex) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_8.set(true);
if !class_mutex_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_9))).is_null() || !class_mutex_is_conditional {
if true {
*__slate_slot_12 = snd_timer_find(core::ptr::addr_of_mut!(*__slate_slot_13));
if (*__slate_slot_12).is_null() {
return 19i32.wrapping_neg();
} else {
(*(*__slate_slot_10)).card = if !unsafe { (*(*__slate_slot_12)).card }.is_null() { (*(*(*__slate_slot_12)).card).number } else { 1i32.wrapping_neg() };
if (*(*__slate_slot_12)).hw.flags & 4 != 0 {
let __v7: *mut snd_timer_ginfo = *__slate_slot_10;
(*__v7).flags = (*__v7).flags | 1u32;
}
sized_strscpy(((*(*__slate_slot_10)).id.as_mut_ptr() as *mut u8) as *mut u8, ((*(*__slate_slot_12)).id.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 64]>() as u64).wrapping_add(__must_be_cstr__71!() as u64).wrapping_add(__must_be_cstr__51!() as u64));
sized_strscpy(((*(*__slate_slot_10)).name.as_mut_ptr() as *mut u8) as *mut u8, ((*(*__slate_slot_12)).name.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 80]>() as u64).wrapping_add(__must_be_cstr__12!() as u64).wrapping_add(__must_be_cstr__55!() as u64));
core::ptr::write(__slate_slot_4, class_spinlock_irq_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_12)).lock)));
__slate_cleanup_armed_4.set(true);
core::ptr::write(__slate_slot_3, (core::ptr::addr_of_mut!((*(*__slate_slot_12)).lock) as u64) as *mut __SlateRecord_spinlock_h_620);
__slate_cleanup_armed_3.set(true);
if !class_spinlock_irq_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_4))).is_null() || !class_spinlock_irq_is_conditional {
if true {
(*(*__slate_slot_10)).resolution = snd_timer_hw_resolution(*__slate_slot_12);
}
}
__slate_cleanup_armed_3.set(false);
__class_spinlock_irq_cleanup_ctx(__slate_slot_3 as *mut _);
__slate_cleanup_armed_4.set(false);
class_spinlock_irq_destructor(__slate_slot_4 as *mut _);
if (*(*__slate_slot_12)).hw.resolution_min > 0 {
(*(*__slate_slot_10)).resolution_min = (*(*__slate_slot_12)).hw.resolution_min;
(*(*__slate_slot_10)).resolution_max = (*(*__slate_slot_12)).hw.resolution_max;
}
*__slate_slot_11 = (*core::ptr::addr_of_mut!((*(*__slate_slot_12)).open_list_head)).next;
while list_is_head(*__slate_slot_11 as *const list_head, core::ptr::addr_of_mut!((*(*__slate_slot_12)).open_list_head) as *const list_head) == 0 {
let __v2: *mut snd_timer_ginfo = *__slate_slot_10;
let __v1: u32 = (*__v2).clients;
(*__v2).clients = __v1.wrapping_add(1);
*__slate_slot_11 = (*(*__slate_slot_11)).next;
}
}
}
}
__slate_cleanup_armed_8.set(false);
__class_mutex_cleanup_ctx(__slate_slot_8 as *mut _);
__slate_cleanup_armed_9.set(false);
class_mutex_destructor(__slate_slot_9 as *mut _);
if copy_to_user(_ginfo as *mut (), *__slate_slot_10 as *const (), core::mem::size_of::<snd_timer_ginfo>() as u64) != 0 {
return 14i32.wrapping_neg();
} else {
return 0;
}
}
}
}

fn timer_set_gparams(gparams: &mut snd_timer_gparams) -> i32 {
let mut t: *mut snd_timer = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_guard_562: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!(register_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_562) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_563: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!(register_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_563) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
t = snd_timer_find(unsafe { core::ptr::addr_of_mut!((*(gparams as *mut snd_timer_gparams)).tid) });
if t.is_null() {
return 19i32.wrapping_neg();
}
if list_empty((unsafe { core::ptr::addr_of_mut!((*t).open_list_head) }) as *const list_head) == 0 {
return 16i32.wrapping_neg();
}
if (unsafe { (*t).hw.set_period }) == None {
return 38i32.wrapping_neg();
}
unsafe { unsafe { (*t).hw.set_period }.unwrap()(t, (*(gparams as *mut snd_timer_gparams)).period_num, (*(gparams as *mut snd_timer_gparams)).period_den) }
}

fn snd_timer_user_gparams(file: *mut file, _gparams: *mut snd_timer_gparams) -> i32 {
let mut gparams: snd_timer_gparams = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(gparams) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 72usize);
}
if copy_from_user(core::ptr::addr_of_mut!(gparams) as *mut (), _gparams as *const (), core::mem::size_of::<snd_timer_gparams>() as u64) != 0 {
return 14i32.wrapping_neg();
}
timer_set_gparams(unsafe { &mut (*core::ptr::addr_of_mut!(gparams)) })
}

fn snd_timer_user_gstatus(file: *mut file, _gstatus: *mut snd_timer_gstatus) -> i32 {
let mut __UNIQUE_ID_unlock_567: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_567) as *mut *mut __SlateRecord_spinlock_h_620;
let mut __UNIQUE_ID_guard_566: core::mem::MaybeUninit<__SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_566) as *mut __SlateRecord_spinlock_h_620;
let mut __UNIQUE_ID_unlock_565: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_565) as *mut *mut __SlateRecord_mutex_h_253;
let mut scope: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_mutex_h_253;
let mut t: core::mem::MaybeUninit<*mut snd_timer> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut snd_timer = core::ptr::addr_of_mut!(t) as *mut *mut snd_timer;
let mut tid: core::mem::MaybeUninit<snd_timer_id> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut snd_timer_id = core::ptr::addr_of_mut!(tid) as *mut snd_timer_id;
let mut gstatus: core::mem::MaybeUninit<snd_timer_gstatus> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut snd_timer_gstatus = core::ptr::addr_of_mut!(gstatus) as *mut snd_timer_gstatus;
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { class_mutex_destructor(__slate_cleanup_ptr_3 as *mut _) };
}
});
let __slate_cleanup_ptr_2 = __slate_slot_2;
let __slate_cleanup_armed_2 = core::cell::Cell::new(false);
let __slate_cleanup_2 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_2.get() {
unsafe { __class_mutex_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _) };
}
});
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_spinlock_irq_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_spinlock_irq_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 80usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 20usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if copy_from_user(core::ptr::addr_of_mut!(*__slate_slot_6) as *mut (), _gstatus as *const (), core::mem::size_of::<snd_timer_gstatus>() as u64) != 0 {
return 14i32.wrapping_neg();
} else {
*__slate_slot_5 = (*__slate_slot_6).tid;
memset(core::ptr::addr_of_mut!(*__slate_slot_6) as *mut (), 0i32, core::mem::size_of::<snd_timer_gstatus>() as u64);
(*__slate_slot_6).tid = *__slate_slot_5;
core::ptr::write(__slate_slot_3, class_mutex_constructor(core::ptr::addr_of_mut!(register_mutex)));
__slate_cleanup_armed_3.set(true);
core::ptr::write(__slate_slot_2, (core::ptr::addr_of_mut!(register_mutex) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_2.set(true);
if !class_mutex_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_3))).is_null() || !class_mutex_is_conditional {
if true {
*__slate_slot_4 = snd_timer_find(core::ptr::addr_of_mut!(*__slate_slot_5));
if !(*__slate_slot_4).is_null() {
core::ptr::write(__slate_slot_1, class_spinlock_irq_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_4)).lock)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!((*(*__slate_slot_4)).lock) as u64) as *mut __SlateRecord_spinlock_h_620);
__slate_cleanup_armed_0.set(true);
(*__slate_slot_6).resolution = snd_timer_hw_resolution(*__slate_slot_4);
if (*(*__slate_slot_4)).hw.precise_resolution != None {
unsafe { (*(*__slate_slot_4)).hw.precise_resolution }.unwrap()(*__slate_slot_4, core::ptr::addr_of_mut!((*__slate_slot_6).resolution_num), core::ptr::addr_of_mut!((*__slate_slot_6).resolution_den));
} else {
(*__slate_slot_6).resolution_num = (*__slate_slot_6).resolution;
(*__slate_slot_6).resolution_den = 1000000000;
}
__slate_cleanup_armed_0.set(false);
__class_spinlock_irq_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_spinlock_irq_destructor(__slate_slot_1 as *mut _);
} else {
return 19i32.wrapping_neg();
}
}
}
__slate_cleanup_armed_2.set(false);
__class_mutex_cleanup_ctx(__slate_slot_2 as *mut _);
__slate_cleanup_armed_3.set(false);
class_mutex_destructor(__slate_slot_3 as *mut _);
if copy_to_user(_gstatus as *mut (), core::ptr::addr_of_mut!(*__slate_slot_6) as *const (), core::mem::size_of::<snd_timer_gstatus>() as u64) != 0 {
return 14i32.wrapping_neg();
} else {
return 0;
}
}
}
}

fn snd_timer_user_tselect(file: *mut file, _tselect: *mut snd_timer_select) -> i32 {
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut str: core::mem::MaybeUninit<__SlateAlign16<[u8; 32]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut [u8; 32] = core::ptr::addr_of_mut!(str) as *mut [u8; 32];
let mut tselect: core::mem::MaybeUninit<snd_timer_select> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut snd_timer_select = core::ptr::addr_of_mut!(tselect) as *mut snd_timer_select;
let mut tu: core::mem::MaybeUninit<*mut snd_timer_user> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut snd_timer_user = core::ptr::addr_of_mut!(tu) as *mut *mut snd_timer_user;
unsafe {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 52usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
err = 0;
*__slate_slot_6 = (*file).private_data as *mut snd_timer_user;
if !unsafe { (*(*__slate_slot_6)).timeri }.is_null() {
snd_timer_close((*(*__slate_slot_6)).timeri);
snd_timer_instance_free((*(*__slate_slot_6)).timeri);
(*(*__slate_slot_6)).timeri = core::ptr::null_mut::<snd_timer_instance>();
}
if copy_from_user(core::ptr::addr_of_mut!(*__slate_slot_5) as *mut (), _tselect as *const (), core::mem::size_of::<snd_timer_select>() as u64) != 0 {
err = 14i32.wrapping_neg();
} else {
sprintf((*__slate_slot_4).as_mut_ptr() as *mut u8, c"application %i".as_ptr() as *const u8, (*get_current()).pid);
if (*__slate_slot_5).id.dev_class != 0 {
(*__slate_slot_5).id.dev_sclass = 1;
}
(*(*__slate_slot_6)).timeri = snd_timer_instance_new((*__slate_slot_4).as_mut_ptr() as *const u8);
if unsafe { (*(*__slate_slot_6)).timeri }.is_null() {
err = 12i32.wrapping_neg();
} else {
let __v2: *mut snd_timer_instance = (*(*__slate_slot_6)).timeri;
(*__v2).flags = (*__v2).flags | 16;
(*(*(*__slate_slot_6)).timeri).callback = {
let __t0: Option<unsafe extern "C" fn(*mut snd_timer_instance, u64, u64)> = if (*(*__slate_slot_6)).tread != 0 { Some(snd_timer_user_tinterrupt) } else { Some(snd_timer_user_interrupt) };
__t0
};
(*(*(*__slate_slot_6)).timeri).ccallback = Some(snd_timer_user_ccallback);
(*(*(*__slate_slot_6)).timeri).callback_data = *__slate_slot_6 as *mut ();
(*(*(*__slate_slot_6)).timeri).disconnect = Some(snd_timer_user_disconnect);
err = snd_timer_open((*(*__slate_slot_6)).timeri, core::ptr::addr_of_mut!((*__slate_slot_5).id), (*get_current()).pid as u32);
if err < 0 {
snd_timer_instance_free((*(*__slate_slot_6)).timeri);
(*(*__slate_slot_6)).timeri = core::ptr::null_mut::<snd_timer_instance>();
}
}
}
return err;
}
}

fn snd_timer_user_info(file: *mut file, _info: *mut snd_timer_info) -> i32 {
let mut __slate_compound_14: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_16: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_unlock_569: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_569) as *mut *mut __SlateRecord_spinlock_h_620;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_620;
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut info: core::mem::MaybeUninit<*mut snd_timer_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut snd_timer_info = core::ptr::addr_of_mut!(info) as *mut *mut snd_timer_info;
let mut t: core::mem::MaybeUninit<*mut snd_timer> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut snd_timer = core::ptr::addr_of_mut!(t) as *mut *mut snd_timer;
let mut tu: core::mem::MaybeUninit<*mut snd_timer_user> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut snd_timer_user = core::ptr::addr_of_mut!(tu) as *mut *mut snd_timer_user;
let __slate_cleanup_ptr_12 = __slate_slot_12;
let __slate_cleanup_armed_12 = core::cell::Cell::new(false);
let __slate_cleanup_12 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_12.get() {
unsafe { __free_snd_timeri_timer(__slate_cleanup_ptr_12 as *mut _) };
}
});
let __slate_cleanup_ptr_11 = __slate_slot_11;
let __slate_cleanup_armed_11 = core::cell::Cell::new(false);
let __slate_cleanup_11 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_11.get() {
unsafe { __free_kfree(__slate_cleanup_ptr_11 as *mut _) };
}
});
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_spinlock_irq_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_spinlock_irq_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes_2 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
*__slate_slot_13 = (*file).private_data as *mut snd_timer_user;
if unsafe { (*(*__slate_slot_13)).timeri }.is_null() {
return 77i32.wrapping_neg();
} else {
core::ptr::write(__slate_slot_12, snd_timeri_timer_get((*(*__slate_slot_13)).timeri));
__slate_cleanup_armed_12.set(true);
if (*__slate_slot_12).is_null() {
return 77i32.wrapping_neg();
} else {
__slate_cleanup_armed_11.set(true);
__obj_size = size_mul(core::mem::size_of::<snd_timer_info>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_5 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_6 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_14 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_14)
});
alloc_tag_restore!();
} else {
*__slate_slot_6 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_16 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_16)
});
}
*__slate_slot_11 = *__slate_slot_6 as *mut snd_timer_info;
if (*__slate_slot_11).is_null() {
return 12i32.wrapping_neg();
} else {
(*(*__slate_slot_11)).card = if !unsafe { (*(*__slate_slot_12)).card }.is_null() { (*(*(*__slate_slot_12)).card).number } else { 1i32.wrapping_neg() };
if (*(*__slate_slot_12)).hw.flags & 4 != 0 {
let __v4: *mut snd_timer_info = *__slate_slot_11;
(*__v4).flags = (*__v4).flags | 1u32;
}
sized_strscpy(((*(*__slate_slot_11)).id.as_mut_ptr() as *mut u8) as *mut u8, ((*(*__slate_slot_12)).id.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 64]>() as u64).wrapping_add(__must_be_cstr__100!() as u64).wrapping_add(__must_be_cstr__26!() as u64));
sized_strscpy(((*(*__slate_slot_11)).name.as_mut_ptr() as *mut u8) as *mut u8, ((*(*__slate_slot_12)).name.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 80]>() as u64).wrapping_add(__must_be_cstr__69!() as u64).wrapping_add(__must_be_cstr__72!() as u64));
core::ptr::write(__slate_slot_1, class_spinlock_irq_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_12)).lock)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!((*(*__slate_slot_12)).lock) as u64) as *mut __SlateRecord_spinlock_h_620);
__slate_cleanup_armed_0.set(true);
if !class_spinlock_irq_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_spinlock_irq_is_conditional {
if true {
(*(*__slate_slot_11)).resolution = snd_timer_hw_resolution(*__slate_slot_12);
}
}
__slate_cleanup_armed_0.set(false);
__class_spinlock_irq_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_spinlock_irq_destructor(__slate_slot_1 as *mut _);
if copy_to_user(_info as *mut (), *__slate_slot_11 as *const (), core::mem::size_of::<snd_timer_info>() as u64) != 0 {
return 14i32.wrapping_neg();
} else {
return 0;
}
}
}
}
}
}

fn snd_timer_user_params(file: *mut file, _params: *mut snd_timer_params) -> i32 {
let mut tread: core::mem::MaybeUninit<snd_timer_tread64> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut snd_timer_tread64 = core::ptr::addr_of_mut!(tread) as *mut snd_timer_tread64;
let mut __UNIQUE_ID_unlock_573: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_573) as *mut *mut __SlateRecord_spinlock_h_620;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_620;
let mut __UNIQUE_ID_unlock_571: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_571) as *mut *mut __SlateRecord_spinlock_h_620;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_620;
let mut resolution: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u64 = core::ptr::addr_of_mut!(resolution) as *mut u64;
let mut t: core::mem::MaybeUninit<*mut snd_timer> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut *mut snd_timer = core::ptr::addr_of_mut!(t) as *mut *mut snd_timer;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut params: core::mem::MaybeUninit<snd_timer_params> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut snd_timer_params = core::ptr::addr_of_mut!(params) as *mut snd_timer_params;
let mut tu: core::mem::MaybeUninit<*mut snd_timer_user> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut *mut snd_timer_user = core::ptr::addr_of_mut!(tu) as *mut *mut snd_timer_user;
let __slate_cleanup_ptr_27 = __slate_slot_27;
let __slate_cleanup_armed_27 = core::cell::Cell::new(false);
let __slate_cleanup_27 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_27.get() {
unsafe { __free_snd_timeri_timer(__slate_cleanup_ptr_27 as *mut _) };
}
});
let __slate_cleanup_ptr_23 = __slate_slot_23;
let __slate_cleanup_armed_23 = core::cell::Cell::new(false);
let __slate_cleanup_23 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_23.get() {
unsafe { class_spinlock_irq_destructor(__slate_cleanup_ptr_23 as *mut _) };
}
});
let __slate_cleanup_ptr_22 = __slate_slot_22;
let __slate_cleanup_armed_22 = core::cell::Cell::new(false);
let __slate_cleanup_22 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_22.get() {
unsafe { __class_spinlock_irq_cleanup_ctx(__slate_cleanup_ptr_22 as *mut _) };
}
});
let __slate_cleanup_ptr_9 = __slate_slot_9;
let __slate_cleanup_armed_9 = core::cell::Cell::new(false);
let __slate_cleanup_9 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_9.get() {
unsafe { class_spinlock_irq_destructor(__slate_cleanup_ptr_9 as *mut _) };
}
});
let __slate_cleanup_ptr_8 = __slate_slot_8;
let __slate_cleanup_armed_8 = core::cell::Cell::new(false);
let __slate_cleanup_8 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_8.get() {
unsafe { __class_spinlock_irq_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes_1 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 80usize);
let __slate_auto_bytes_3 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
*__slate_slot_30 = (*file).private_data as *mut snd_timer_user;
if unsafe { (*(*__slate_slot_30)).timeri }.is_null() {
return 77i32.wrapping_neg();
} else {
core::ptr::write(__slate_slot_27, snd_timeri_timer_get((*(*__slate_slot_30)).timeri));
__slate_cleanup_armed_27.set(true);
if (*__slate_slot_27).is_null() {
return 77i32.wrapping_neg();
} else {
if copy_from_user(core::ptr::addr_of_mut!(*__slate_slot_29) as *mut (), _params as *const (), core::mem::size_of::<snd_timer_params>() as u64) != 0 {
return 14i32.wrapping_neg();
} else {
'__join_2: {
if (*(*__slate_slot_27)).hw.flags & 4 == 0 {
let __slate_auto_bytes = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if (*__slate_slot_29).ticks < 1 {
*__slate_slot_28 = 22i32.wrapping_neg();
break '__join_2;
} else {
// Don't allow resolution less than 1ms
*__slate_slot_26 = snd_timer_resolution((*(*__slate_slot_30)).timeri);
let __v25: u64 = *__slate_slot_26;
*__slate_slot_26 = __v25.wrapping_mul((*__slate_slot_29).ticks as u64);
if *__slate_slot_26 < 1000000 {
*__slate_slot_28 = 22i32.wrapping_neg();
break '__join_2;
}
}
}
if (*__slate_slot_29).queue_size > 0 && ((*__slate_slot_29).queue_size < 32 || (*__slate_slot_29).queue_size > 1024) {
*__slate_slot_28 = 22i32.wrapping_neg();
} else {
if (*__slate_slot_29).filter & 4294512192u32 != 0 {
*__slate_slot_28 = 22i32.wrapping_neg();
} else {
snd_timer_stop((*(*__slate_slot_30)).timeri);
core::ptr::write(__slate_slot_23, class_spinlock_irq_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_27)).lock)));
__slate_cleanup_armed_23.set(true);
core::ptr::write(__slate_slot_22, (core::ptr::addr_of_mut!((*(*__slate_slot_27)).lock) as u64) as *mut __SlateRecord_spinlock_h_620);
__slate_cleanup_armed_22.set(true);
if !class_spinlock_irq_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_23))).is_null() || !class_spinlock_irq_is_conditional {
if true {
let __v21: *mut snd_timer_instance = (*(*__slate_slot_30)).timeri;
let __v19: u32 = (*__v21).flags & 4294967095u32;
(*__v21).flags = __v19;
if (*__slate_slot_29).flags & 1u32 != 0 {
let __v18: *mut snd_timer_instance = (*(*__slate_slot_30)).timeri;
(*__v18).flags = (*__v18).flags | 8;
}
if (*__slate_slot_29).flags & 2u32 != 0 {
let __v15: *mut snd_timer_instance = (*(*__slate_slot_30)).timeri;
(*__v15).flags = (*__v15).flags | 64;
}
if (*__slate_slot_29).flags & 4u32 != 0 {
let __v12: *mut snd_timer_instance = (*(*__slate_slot_30)).timeri;
(*__v12).flags = (*__v12).flags | 128;
}
}
}
__slate_cleanup_armed_22.set(false);
__class_spinlock_irq_cleanup_ctx(__slate_slot_22 as *mut _);
__slate_cleanup_armed_23.set(false);
class_spinlock_irq_destructor(__slate_slot_23 as *mut _);
if (*__slate_slot_29).queue_size > 0 && ((*(*__slate_slot_30)).queue_size as u32) != (*__slate_slot_29).queue_size {
*__slate_slot_28 = realloc_user_queue(*__slate_slot_30, (*__slate_slot_29).queue_size as i32);
if *__slate_slot_28 < 0 {
break '__join_2;
}
}
core::ptr::write(__slate_slot_9, class_spinlock_irq_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_30)).qlock)));
__slate_cleanup_armed_9.set(true);
core::ptr::write(__slate_slot_8, (core::ptr::addr_of_mut!((*(*__slate_slot_30)).qlock) as u64) as *mut __SlateRecord_spinlock_h_620);
__slate_cleanup_armed_8.set(true);
if !class_spinlock_irq_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_9))).is_null() || !class_spinlock_irq_is_conditional {
if true {
(*(*__slate_slot_30)).qused = 0;
(*(*__slate_slot_30)).qtail = 0;
(*(*__slate_slot_30)).qhead = 0;
if (*(*(*__slate_slot_30)).timeri).flags & 128 != 0 {
if (*(*__slate_slot_30)).tread != 0 {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
memset(core::ptr::addr_of_mut!(*__slate_slot_0) as *mut (), 0i32, core::mem::size_of::<snd_timer_tread64>() as u64);
(*__slate_slot_0).event = 6;
(*__slate_slot_0).tstamp_sec = 0;
(*__slate_slot_0).tstamp_nsec = 0;
(*__slate_slot_0).val = 0;
snd_timer_user_append_to_tqueue(*__slate_slot_30, core::ptr::addr_of_mut!(*__slate_slot_0));
} else {
let r: *mut snd_timer_read = unsafe { (*(*__slate_slot_30)).queue }.wrapping_offset(0 as isize);
(*r).resolution = 0;
(*r).ticks = 0;
let __v6: *mut snd_timer_user = *__slate_slot_30;
let __v5: i32 = (*__v6).qused;
(*__v6).qused = __v5.wrapping_add(1);
let __v3: *mut snd_timer_user = *__slate_slot_30;
let __v2: i32 = (*__v3).qtail;
(*__v3).qtail = __v2.wrapping_add(1);
}
}
(*(*__slate_slot_30)).filter = (*__slate_slot_29).filter;
(*(*__slate_slot_30)).ticks = (*__slate_slot_29).ticks as u64;
}
}
__slate_cleanup_armed_8.set(false);
__class_spinlock_irq_cleanup_ctx(__slate_slot_8 as *mut _);
__slate_cleanup_armed_9.set(false);
class_spinlock_irq_destructor(__slate_slot_9 as *mut _);
*__slate_slot_28 = 0;
}
}
}
if copy_to_user(_params as *mut (), core::ptr::addr_of_mut!(*__slate_slot_29) as *const (), core::mem::size_of::<snd_timer_params>() as u64) != 0 {
return 14i32.wrapping_neg();
} else {
return *__slate_slot_28;
}
}
}
}
}
}

fn snd_timer_user_status32(file: *mut file, _status: *mut snd_timer_status32) -> i32 {
let mut __UNIQUE_ID_unlock_575: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_575) as *mut *mut __SlateRecord_spinlock_h_620;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_620;
let mut status: core::mem::MaybeUninit<snd_timer_status32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut snd_timer_status32 = core::ptr::addr_of_mut!(status) as *mut snd_timer_status32;
let mut tu: core::mem::MaybeUninit<*mut snd_timer_user> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut snd_timer_user = core::ptr::addr_of_mut!(tu) as *mut *mut snd_timer_user;
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_spinlock_irq_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_spinlock_irq_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 88usize);
*__slate_slot_3 = (*file).private_data as *mut snd_timer_user;
if unsafe { (*(*__slate_slot_3)).timeri }.is_null() {
return 77i32.wrapping_neg();
} else {
memset(core::ptr::addr_of_mut!(*__slate_slot_2) as *mut (), 0i32, core::mem::size_of::<snd_timer_status32>() as u64);
(*__slate_slot_2).tstamp_sec = (*(*__slate_slot_3)).tstamp.tv_sec as i32;
(*__slate_slot_2).tstamp_nsec = (*(*__slate_slot_3)).tstamp.tv_nsec as i32;
(*__slate_slot_2).resolution = snd_timer_resolution((*(*__slate_slot_3)).timeri) as u32;
(*__slate_slot_2).lost = (*(*(*__slate_slot_3)).timeri).lost as u32;
(*__slate_slot_2).overrun = (*(*__slate_slot_3)).overrun as u32;
core::ptr::write(__slate_slot_1, class_spinlock_irq_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_3)).qlock)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!((*(*__slate_slot_3)).qlock) as u64) as *mut __SlateRecord_spinlock_h_620);
__slate_cleanup_armed_0.set(true);
if !class_spinlock_irq_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_spinlock_irq_is_conditional {
if true {
(*__slate_slot_2).queue = (*(*__slate_slot_3)).qused as u32;
}
}
__slate_cleanup_armed_0.set(false);
__class_spinlock_irq_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_spinlock_irq_destructor(__slate_slot_1 as *mut _);
if copy_to_user(_status as *mut (), core::ptr::addr_of_mut!(*__slate_slot_2) as *const (), core::mem::size_of::<snd_timer_status32>() as u64) != 0 {
return 14i32.wrapping_neg();
} else {
return 0;
}
}
}
}

fn snd_timer_user_status64(file: *mut file, _status: *mut snd_timer_status64) -> i32 {
let mut __UNIQUE_ID_unlock_577: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_577) as *mut *mut __SlateRecord_spinlock_h_620;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_620;
let mut status: core::mem::MaybeUninit<snd_timer_status64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut snd_timer_status64 = core::ptr::addr_of_mut!(status) as *mut snd_timer_status64;
let mut tu: core::mem::MaybeUninit<*mut snd_timer_user> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut snd_timer_user = core::ptr::addr_of_mut!(tu) as *mut *mut snd_timer_user;
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_spinlock_irq_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_spinlock_irq_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 96usize);
*__slate_slot_3 = (*file).private_data as *mut snd_timer_user;
if unsafe { (*(*__slate_slot_3)).timeri }.is_null() {
return 77i32.wrapping_neg();
} else {
memset(core::ptr::addr_of_mut!(*__slate_slot_2) as *mut (), 0i32, core::mem::size_of::<snd_timer_status64>() as u64);
(*__slate_slot_2).tstamp_sec = (*(*__slate_slot_3)).tstamp.tv_sec;
(*__slate_slot_2).tstamp_nsec = (*(*__slate_slot_3)).tstamp.tv_nsec;
(*__slate_slot_2).resolution = snd_timer_resolution((*(*__slate_slot_3)).timeri) as u32;
(*__slate_slot_2).lost = (*(*(*__slate_slot_3)).timeri).lost as u32;
(*__slate_slot_2).overrun = (*(*__slate_slot_3)).overrun as u32;
core::ptr::write(__slate_slot_1, class_spinlock_irq_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_3)).qlock)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!((*(*__slate_slot_3)).qlock) as u64) as *mut __SlateRecord_spinlock_h_620);
__slate_cleanup_armed_0.set(true);
if !class_spinlock_irq_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_spinlock_irq_is_conditional {
if true {
(*__slate_slot_2).queue = (*(*__slate_slot_3)).qused as u32;
}
}
__slate_cleanup_armed_0.set(false);
__class_spinlock_irq_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_spinlock_irq_destructor(__slate_slot_1 as *mut _);
if copy_to_user(_status as *mut (), core::ptr::addr_of_mut!(*__slate_slot_2) as *const (), core::mem::size_of::<snd_timer_status64>() as u64) != 0 {
return 14i32.wrapping_neg();
} else {
return 0;
}
}
}
}

fn snd_timer_user_start(file: *mut file) -> i32 {
let mut err: i32 = 0;
let tu: *mut snd_timer_user = (unsafe { (*file).private_data }) as *mut snd_timer_user;
if unsafe { (*tu).timeri }.is_null() {
return 77i32.wrapping_neg();
}
snd_timer_stop(unsafe { (*tu).timeri });
unsafe {
(*(*tu).timeri).lost = 0;
}
unsafe {
(*tu).last_resolution = 0;
}
err = snd_timer_start(unsafe { (*tu).timeri }, (unsafe { (*tu).ticks }) as u32);
if err < 0 {
return err;
}
0
}

fn snd_timer_user_stop(file: *mut file) -> i32 {
let mut err: i32 = 0;
let tu: *mut snd_timer_user = (unsafe { (*file).private_data }) as *mut snd_timer_user;
if unsafe { (*tu).timeri }.is_null() {
return 77i32.wrapping_neg();
}
err = snd_timer_stop(unsafe { (*tu).timeri });
if err < 0 {
return err;
}
0
}

fn snd_timer_user_continue(file: *mut file) -> i32 {
let mut err: i32 = 0;
let tu: *mut snd_timer_user = (unsafe { (*file).private_data }) as *mut snd_timer_user;
if unsafe { (*tu).timeri }.is_null() {
return 77i32.wrapping_neg();
}
// start timer instead of continue if it's not used before
if (unsafe { (*(*tu).timeri).flags }) & 65536 == 0 {
return snd_timer_user_start(file);
}
unsafe {
(*(*tu).timeri).lost = 0;
}
err = snd_timer_continue(unsafe { (*tu).timeri });
if err < 0 {
return err;
}
0
}

fn snd_timer_user_pause(file: *mut file) -> i32 {
let mut err: i32 = 0;
let tu: *mut snd_timer_user = (unsafe { (*file).private_data }) as *mut snd_timer_user;
if unsafe { (*tu).timeri }.is_null() {
return 77i32.wrapping_neg();
}
err = snd_timer_pause(unsafe { (*tu).timeri });
if err < 0 {
return err;
}
0
}

fn snd_timer_user_tread(argp: *mut (), tu: *mut snd_timer_user, cmd: u32, compat: bool) -> i32 {
let mut p: *mut i32 = argp as *mut i32;
let mut xarg: i32 = 0;
let mut old_tread: i32 = 0;
if !unsafe { (*tu).timeri }.is_null() {
return 16i32.wrapping_neg();
}
// too late
let mut __v0: i64;
might_fault();
let mut __v1: i64;
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t0: u64;
let __t1: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") p as u64 => __t0, lateout("dx") __t1, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
instrument_get_user!(__val_gu);
xarg = __val_gu as i32;
__v1 = __ret_gu as i64;
__v0 = __v1;
if __v0 != 0 {
return 14i32.wrapping_neg();
}
old_tread = unsafe { (*tu).tread };
if xarg == 0 {
unsafe {
(*tu).tread = 0;
}
} else {
if (cmd as u64) == _IOW!(84i32, 164i32, i32) || IS_ENABLED__2!() != 0 && !compat {
unsafe {
(*tu).tread = 1;
}
} else {
unsafe {
(*tu).tread = 2;
}
}
}
let __v4: bool = (unsafe { (*tu).tread }) != old_tread && realloc_user_queue(tu, unsafe { (*tu).queue_size }) < 0;
if __v4 {
unsafe {
(*tu).tread = old_tread;
}
return 12i32.wrapping_neg();
}
0
}

fn snd_utimer_ioctl_create(file: *mut file, _utimer_info: *mut snd_timer_uinfo) -> i32 {
25i32.wrapping_neg()
}

fn __snd_timer_user_ioctl(file: *mut file, cmd: u32, arg: u64, compat: bool) -> i64 {
let argp: *mut () = arg as *mut ();
let p: *mut i32 = argp as *mut i32;
let tu: *mut snd_timer_user = (unsafe { (*file).private_data }) as *mut snd_timer_user;
match cmd {
2147767296 => {
let mut __v0: i64;
might_fault();
let mut __v1: i64;
let mut __ret_pu: i32 = 0;
let mut __ptr_pu: *mut () = unsafe { core::mem::zeroed() };
let mut __val_pu: i32 = 0;
let mut __x: i32 = SNDRV_PROTOCOL_VERSION!(2i32, 0i32, 8i32);
let __ptr: *mut i32 = p;
__chk_user_ptr!();
__ptr_pu = __ptr as *mut ();
__val_pu = __x;
unsafe {
let __t0: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") __ptr_pu as u64 => __t0, in("ax") __val_pu as u32, options(att_syntax));
__ret_pu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
instrument_put_user!(__x, __ptr);
__v1 = __ret_pu as i64;
__v0 = __v1;
return (if __v0 != 0 { 14i32.wrapping_neg() } else { 0i32 }) as i64;
}
3222557697 => {
return snd_timer_user_next_device(argp as *mut snd_timer_id) as i64;
}
1074025474 | 1074025636 => {
return snd_timer_user_tread(argp, tu, cmd, compat) as i64;
}
3237499907 => {
return snd_timer_user_ginfo(file, argp as *mut snd_timer_ginfo) as i64;
}
1078481924 => {
return snd_timer_user_gparams(file, argp as *mut snd_timer_gparams) as i64;
}
3226489861 => {
return snd_timer_user_gstatus(file, argp as *mut snd_timer_gstatus) as i64;
}
1077171216 => {
return snd_timer_user_tselect(file, argp as *mut snd_timer_select) as i64;
}
2162709521 => {
return snd_timer_user_info(file, argp as *mut snd_timer_info) as i64;
}
1079006226 => {
return snd_timer_user_params(file, argp as *mut snd_timer_params) as i64;
}
2153272340 => {
return snd_timer_user_status32(file, argp as *mut snd_timer_status32) as i64;
}
2153796628 => {
return snd_timer_user_status64(file, argp as *mut snd_timer_status64) as i64;
}
21536 | 21664 => {
return snd_timer_user_start(file) as i64;
}
21537 | 21665 => {
return snd_timer_user_stop(file) as i64;
}
21538 | 21666 => {
return snd_timer_user_continue(file) as i64;
}
21539 | 21667 => {
return snd_timer_user_pause(file) as i64;
}
3223344293 => {
return snd_utimer_ioctl_create(file, argp as *mut snd_timer_uinfo) as i64;
}
_ => {
}
}
25i32.wrapping_neg() as i64
}

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_user_ioctl")]
extern "C" fn snd_timer_user_ioctl(file: *mut file, cmd: u32, arg: u64) -> i64 {
let tu: *mut snd_timer_user = (unsafe { (*file).private_data }) as *mut snd_timer_user;
let mut __UNIQUE_ID_guard_578: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*tu).ioctl_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_578) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_579: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*tu).ioctl_lock) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_579) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
__snd_timer_user_ioctl(file, cmd, arg, false)
}

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_user_fasync")]
extern "C" fn snd_timer_user_fasync(fd: i32, file: *mut file, on: i32) -> i32 {
let mut tu: *mut snd_timer_user = unsafe { core::mem::zeroed() };
tu = (unsafe { (*file).private_data }) as *mut snd_timer_user;
unsafe { snd_fasync_helper(fd, file, on, core::ptr::addr_of_mut!((*tu).fasync)) }
}

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_user_read")]
extern "C" fn snd_timer_user_read(file: *mut file, mut buffer: *mut u8, count: u64, offset: *mut i64) -> i64 {
let mut __slate_compound_31: snd_timer_tread32 = unsafe { core::mem::zeroed() };
let mut __ret: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u32 = core::ptr::addr_of_mut!(__ret) as *mut u32;
let mut __ai_ptr: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut u32 = core::ptr::addr_of_mut!(__ai_ptr) as *mut *mut u32;
let mut wait: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut wait_queue_entry = core::ptr::addr_of_mut!(wait) as *mut wait_queue_entry;
let mut bug: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug) as *mut *mut bug_entry;
let mut __v20: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v20) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut qhead: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut i32 = core::ptr::addr_of_mut!(qhead) as *mut i32;
let mut unit: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut i64 = core::ptr::addr_of_mut!(unit) as *mut i64;
let mut result: i64 = unsafe { core::mem::zeroed() };
let mut tu: core::mem::MaybeUninit<*mut snd_timer_user> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut *mut snd_timer_user = core::ptr::addr_of_mut!(tu) as *mut *mut snd_timer_user;
let mut tread32: core::mem::MaybeUninit<snd_timer_tread32> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut snd_timer_tread32 = core::ptr::addr_of_mut!(tread32) as *mut snd_timer_tread32;
let mut tread: core::mem::MaybeUninit<*mut snd_timer_tread64> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut *mut snd_timer_tread64 = core::ptr::addr_of_mut!(tread) as *mut *mut snd_timer_tread64;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 16usize);
let __slate_auto_bytes_3 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
result = 0;
let __slate_auto_bytes_4 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
err = 0;
*__slate_slot_28 = (*file).private_data as *mut snd_timer_user;
let __t3: i32 = (*(*__slate_slot_28)).tread;
if __t3 == 1 {
*__slate_slot_26 = core::mem::size_of::<snd_timer_tread64>() as i64;
} else {
if __t3 == 2 {
*__slate_slot_26 = core::mem::size_of::<snd_timer_tread32>() as i64;
} else {
if __t3 == 0 {
*__slate_slot_26 = core::mem::size_of::<snd_timer_read>() as i64;
} else {
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"Corrupt snd_timer_user\n".as_ptr() as *const u8);
*__slate_slot_20 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t2: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t2, sym __slate_asm_string_snd_timer_user_read_38, sym __slate_asm_string_snd_timer_user_read_39, const 2421 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_18 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t2 as u64) };
(*__slate_slot_20).unwrap()(*__slate_slot_18);
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
return 524i32.wrapping_neg() as i64;
}
}
}
mutex_lock(core::ptr::addr_of_mut!((*(*__slate_slot_28)).ioctl_lock));
spin_lock_irq(core::ptr::addr_of_mut!((*(*__slate_slot_28)).qlock));
'__join_0: {
'__join_22: {
'__join_15: {
'__loop_1: loop {
if (count as i64).wrapping_sub(result) >= *__slate_slot_26 {
while (*(*__slate_slot_28)).qused == 0 {
let __slate_auto_bytes = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
if (*file).f_flags & 2048u32 != 0 || result > 0 {
break '__join_22;
} else {
loop {
debug_normal_state_change!();
trace_set_current_state!(1i32);
loop {
kcsan_mb!();
loop {
core::ptr::write(__slate_slot_15, core::ptr::addr_of_mut!((*get_current()).__state));
kcsan_mb!();
instrument_atomic_read_write(*__slate_slot_15 as *const (), core::mem::size_of::<u32>() as u64);
core::ptr::write(__slate_slot_13, 1u32);
let __t1: u32;
core::arch::asm!("xchgl {0:e}, ({1:r})", inlateout(reg) *__slate_slot_13 as u32 => __t1, in(reg) *__slate_slot_15, options(nostack, att_syntax));
*__slate_slot_13 = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
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
init_waitqueue_entry(&mut (*core::ptr::addr_of_mut!(*__slate_slot_17)), get_current());
add_wait_queue(core::ptr::addr_of_mut!((*(*__slate_slot_28)).qchange_sleep), core::ptr::addr_of_mut!(*__slate_slot_17));
spin_unlock_irq(core::ptr::addr_of_mut!((*(*__slate_slot_28)).qlock));
mutex_unlock(core::ptr::addr_of_mut!((*(*__slate_slot_28)).ioctl_lock));
schedule();
mutex_lock(core::ptr::addr_of_mut!((*(*__slate_slot_28)).ioctl_lock));
spin_lock_irq(core::ptr::addr_of_mut!((*(*__slate_slot_28)).qlock));
remove_wait_queue(core::ptr::addr_of_mut!((*(*__slate_slot_28)).qchange_sleep), core::ptr::addr_of_mut!(*__slate_slot_17));
if (*(*__slate_slot_28)).disconnected {
break '__join_15;
} else {
if signal_pending(get_current()) != 0 {
break '__loop_1;
}
}
}
}
let __v12: *mut snd_timer_user = *__slate_slot_28;
let __v11: i32 = (*__v12).qhead;
(*__v12).qhead = __v11.wrapping_add(1);
*__slate_slot_25 = __v11;
let __v9: *mut snd_timer_user = *__slate_slot_28;
(*__v9).qhead = (*__v9).qhead % (*(*__slate_slot_28)).queue_size;
let __v6: *mut snd_timer_user = *__slate_slot_28;
let __v5: i32 = (*__v6).qused;
(*__v6).qused = __v5.wrapping_sub(1);
spin_unlock_irq(core::ptr::addr_of_mut!((*(*__slate_slot_28)).qlock));
*__slate_slot_30 = unsafe { (*(*__slate_slot_28)).tqueue }.wrapping_offset(*__slate_slot_25 as isize);
let __t0: i32 = (*(*__slate_slot_28)).tread;
if __t0 == 1 {
if copy_to_user(buffer as *mut (), *__slate_slot_30 as *const (), core::mem::size_of::<snd_timer_tread64>() as u64) != 0 {
err = 14i32.wrapping_neg();
}
} else {
if __t0 == 2 {
memset(core::ptr::addr_of_mut!(*__slate_slot_29) as *mut (), 0i32, core::mem::size_of::<snd_timer_tread32>() as u64);
*__slate_slot_29 = *{
__slate_compound_31 = snd_timer_tread32 { event: (*(*__slate_slot_30)).event, tstamp_sec: (*(*__slate_slot_30)).tstamp_sec as i32, tstamp_nsec: (*(*__slate_slot_30)).tstamp_nsec as i32, val: (*(*__slate_slot_30)).val };
core::ptr::addr_of_mut!(__slate_compound_31)
};
if copy_to_user(buffer as *mut (), core::ptr::addr_of_mut!(*__slate_slot_29) as *const (), core::mem::size_of::<snd_timer_tread32>() as u64) != 0 {
err = 14i32.wrapping_neg();
}
} else {
if __t0 == 0 {
if copy_to_user(buffer as *mut (), unsafe { (*(*__slate_slot_28)).queue }.wrapping_offset(*__slate_slot_25 as isize) as *const (), core::mem::size_of::<snd_timer_read>() as u64) != 0 {
err = 14i32.wrapping_neg();
}
} else {
err = 524i32.wrapping_neg();
}
}
}
spin_lock_irq(core::ptr::addr_of_mut!((*(*__slate_slot_28)).qlock));
if err < 0 {
break '__join_0;
} else {
result = result.wrapping_add(*__slate_slot_26);
let __v1: *mut u8 = buffer;
buffer = __v1.wrapping_offset(*__slate_slot_26 as isize);
}
} else {
break '__join_0;
}
}
err = 512i32.wrapping_neg();
break '__join_0;
}
err = 19i32.wrapping_neg();
break '__join_0;
}
err = 11i32.wrapping_neg();
}
spin_unlock_irq(core::ptr::addr_of_mut!((*(*__slate_slot_28)).qlock));
mutex_unlock(core::ptr::addr_of_mut!((*(*__slate_slot_28)).ioctl_lock));
return if result > 0 { result } else { err as i64 };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_580: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_user_poll")]
extern "C" fn snd_timer_user_poll(file: *mut file, wait: *mut poll_table_struct) -> u32 {
let mut mask: u32 = 0;
let mut tu: *mut snd_timer_user = unsafe { core::mem::zeroed() };
tu = (unsafe { (*file).private_data }) as *mut snd_timer_user;
poll_wait(file, unsafe { core::ptr::addr_of_mut!((*tu).qchange_sleep) }, wait);
let mut __UNIQUE_ID_guard_581: __SlateRecord_spinlock_h_620 = class_spinlock_irq_constructor(unsafe { core::ptr::addr_of_mut!((*tu).qlock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_581) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irq_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_582: *mut __SlateRecord_spinlock_h_620 = ((unsafe { core::ptr::addr_of_mut!((*tu).qlock) }) as u64) as *mut __SlateRecord_spinlock_h_620;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_582) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irq_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if (unsafe { (*tu).qused }) != 0 {
mask |= 1 | 64;
}
if unsafe { (*tu).disconnected } {
mask |= 8;
}
mask
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct snd_timer_gparams32 {
    tid: snd_timer_id,
    period_num: u32,
    period_den: u32,
    reserved: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_info32 {
    flags: u32,
    card: i32,
    id: [u8; 64],
    name: [u8; 80],
    reserved0: u32,
    resolution: u32,
    reserved: [u8; 64],
}


fn snd_timer_user_gparams_compat(file: *mut file, mut user: *mut snd_timer_gparams32) -> i32 {
let mut gparams: snd_timer_gparams = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(gparams) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 72usize);
}
let __v0: bool = if copy_from_user(core::ptr::addr_of_mut!(gparams.tid) as *mut (), (unsafe { core::ptr::addr_of_mut!((*user).tid) }) as *const (), core::mem::size_of::<snd_timer_id>() as u64) != 0 {
true
} else {
let mut __v1: i64;
might_fault();
let mut __v2: i64;
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t0: u64;
let __t1: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*user).period_num) as u64 => __t0, lateout("dx") __t1, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
instrument_get_user!(__val_gu);
gparams.period_num = __val_gu as u64;
__v2 = __ret_gu as i64;
__v1 = __v2;
__v1 != 0
};
let __v5: bool = if __v0 {
true
} else {
let mut __v6: i64;
might_fault();
let mut __v7: i64;
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t2: u64;
let __t3: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*user).period_den) as u64 => __t2, lateout("dx") __t3, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t3 as u32) };
}
instrument_get_user!(__val_gu);
gparams.period_den = __val_gu as u64;
__v7 = __ret_gu as i64;
__v6 = __v7;
__v6 != 0
};
if __v5 {
return 14i32.wrapping_neg();
}
timer_set_gparams(unsafe { &mut (*core::ptr::addr_of_mut!(gparams)) })
}

fn snd_timer_user_info_compat(file: *mut file, _info: *mut snd_timer_info32) -> i32 {
let mut info: snd_timer_info32 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(info) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 224usize);
}
let tu: *mut snd_timer_user = (unsafe { (*file).private_data }) as *mut snd_timer_user;
if unsafe { (*tu).timeri }.is_null() {
return 77i32.wrapping_neg();
}
let mut t: *mut snd_timer = snd_timeri_timer_get(unsafe { (*tu).timeri });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(t) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_timeri_timer(__slate_cleanup_ptr_0 as *mut _));
if t.is_null() {
return 77i32.wrapping_neg();
}
unsafe { memset(core::ptr::addr_of_mut!(info) as *mut (), 0i32, core::mem::size_of::<snd_timer_info32>() as u64) };
info.card = if !unsafe { (*t).card }.is_null() { unsafe { (*(*t).card).number } } else { 1i32.wrapping_neg() };
if (unsafe { (*t).hw.flags }) & 4 != 0 {
info.flags |= 1u32;
}
unsafe { sized_strscpy(info.id.as_mut_ptr() as *mut u8, ((*t).id.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 64]>() as u64).wrapping_add(__must_be_cstr__61!() as u64).wrapping_add(__must_be_cstr__17!() as u64)) };
unsafe { sized_strscpy(info.name.as_mut_ptr() as *mut u8, ((*t).name.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 80]>() as u64).wrapping_add(__must_be_cstr__58!() as u64).wrapping_add(__must_be_cstr__79!() as u64)) };
info.resolution = (unsafe { (*t).hw.resolution }) as u32;
if copy_to_user(_info as *mut (), core::ptr::addr_of_mut!(info) as *const (), core::mem::size_of::<snd_timer_info32>() as u64) != 0 {
return 14i32.wrapping_neg();
}
0
}

fn __snd_timer_user_ioctl_compat(file: *mut file, cmd: u32, arg: u64) -> i64 {
let argp: *mut () = compat_ptr(arg as u32);
match cmd {
21536..=21539 | 21664..=21667 | 1074025474 | 1074025636 | 1077171216 | 1079006226 | 2147767296 | 3222557697 | 3226489861 | 3237499907 => {
return __snd_timer_user_ioctl(file, cmd, argp as u64, true);
}
1077695492 => {
return snd_timer_user_gparams_compat(file, argp as *mut snd_timer_gparams32) as i64;
}
2162185233 => {
return snd_timer_user_info_compat(file, argp as *mut snd_timer_info32) as i64;
}
1079530516 => {
return snd_timer_user_status32(file, argp as *mut snd_timer_status32) as i64;
}
1080054804 => {
return snd_timer_user_status64(file, argp as *mut snd_timer_status64) as i64;
}
_ => {
}
}
515i32.wrapping_neg() as i64
}

#[unsafe(link_section = ".text.slate_distinct.timer.snd_timer_user_ioctl_compat")]
extern "C" fn snd_timer_user_ioctl_compat(file: *mut file, cmd: u32, arg: u64) -> i64 {
let tu: *mut snd_timer_user = (unsafe { (*file).private_data }) as *mut snd_timer_user;
let mut __UNIQUE_ID_guard_583: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*tu).ioctl_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_583) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_584: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*tu).ioctl_lock) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_584) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
__snd_timer_user_ioctl_compat(file, cmd, arg)
}

static snd_timer_f_ops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.owner = core::ptr::null_mut::<module>();
__t0.read = Some(snd_timer_user_read);
__t0.poll = Some(snd_timer_user_poll);
__t0.unlocked_ioctl = Some(snd_timer_user_ioctl);
__t0.compat_ioctl = Some(snd_timer_user_ioctl_compat);
__t0.open = Some(snd_timer_user_open);
__t0.release = Some(snd_timer_user_release);
__t0.fasync = Some(snd_timer_user_fasync);
__t0
});

/// unregister the system timer
fn snd_timer_free_all() {
let mut timer: *mut snd_timer = unsafe { core::mem::zeroed() };
let mut n: *mut snd_timer = unsafe { core::mem::zeroed() };
timer = list_first_entry__10!(unsafe { core::ptr::addr_of_mut!(snd_timer_list) });
n = list_next_entry__7!(timer, device_list);
while list_entry_is_head!(timer, unsafe { core::ptr::addr_of_mut!(snd_timer_list) }, device_list) == 0 {
snd_timer_free(timer);
timer = n;
n = list_next_entry__7!(n, device_list);
}
}

static mut timer_dev: *mut device = unsafe { core::mem::zeroed() };

// ENTRY functions
#[unsafe(link_section = ".init.text")]
extern "C" fn alsa_timer_init() -> i32 {
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_2 = snd_device_alloc(core::ptr::addr_of_mut!(timer_dev), core::ptr::null_mut::<snd_card>());
if *__slate_slot_2 < 0 {
return *__slate_slot_2;
} else {
dev_set_name(timer_dev, c"timer".as_ptr() as *const u8);
*__slate_slot_2 = snd_timer_register_system();
if *__slate_slot_2 < 0 {
while false {
}
_printk(c"\x013ALSA: unable to register system timer (%i)\n".as_ptr() as *const u8, *__slate_slot_2);
} else {
*__slate_slot_2 = snd_register_device(33i32, core::ptr::null_mut::<snd_card>(), 0i32, core::ptr::addr_of!(snd_timer_f_ops.0), core::ptr::null_mut::<()>(), timer_dev);
if *__slate_slot_2 < 0 {
while false {
}
_printk(c"\x013ALSA: unable to register timer device (%i)\n".as_ptr() as *const u8, *__slate_slot_2);
snd_timer_free_all();
} else {
snd_timer_proc_init();
return 0;
}
}
put_device(timer_dev);
return *__slate_slot_2;
}
}
}

#[unsafe(link_section = ".exit.text")]
extern "C" fn alsa_timer_exit() {
unsafe { snd_unregister_device(timer_dev) };
snd_timer_free_all();
unsafe { put_device(timer_dev) };
snd_timer_proc_done();
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_alsa_timer_init_586: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(alsa_timer_init)) };

#[unsafe(link_section = ".exitcall.exit")]
static mut __exitcall_alsa_timer_exit: Option<unsafe extern "C" fn()> = Some(alsa_timer_exit);

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_check_copy_size_1: [u8; 1] = *b"\0";

static __slate_asm_string_check_copy_size_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/include/linux/ucopysize.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_snd_timer_new_3: [u8; 1] = *b"\0";

static __slate_asm_string_snd_timer_new_4: [u8; 48] = *b"/home/takashi/c-corpus/linux/sound/core/timer.c\0";

static __slate_asm_string_snd_timer_user_read_38: [u8; 24] = *b"Corrupt snd_timer_user\n\0";

static __slate_asm_string_snd_timer_user_read_39: [u8; 48] = *b"/home/takashi/c-corpus/linux/sound/core/timer.c\0";
