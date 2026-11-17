
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
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! __DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//!
//! @TYPE: The type of each flexible array element
//! @NAME: The name of the flexible array member
//!
//! In order to have a flexible array member in a union or alone in a
//! struct, it needs to be wrapped in an anonymous struct with at least 1
//! named member, but that member can be empty.
//! aligned_u64 should be used in defining kernel<->userspace ABIs to avoid
//! common 32/64-bit compat problems.
//! 64-bit values align to 4-byte boundaries on x86_32 (and possibly other
//! architectures) and to 8-byte boundaries on 64-bit architectures.  The new
//! aligned_64 type enforces 8-byte alignment so that structs containing
//! aligned_64 values have the same alignment on 32-bit and 64-bit architectures.
//! No conversions are necessary between 32-bit user-space and a 64-bit kernel.
//! These are the special 64-bit data types that are 8-byte aligned
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! # chars in a path name including nul
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
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
//! This counts to 15. Any more, it will return 16th argument.
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! Defines x86 CPU feature bits
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
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
//! u64_to_user_ptr - cast a pointer passed as u64 from user space to void __user *
//! @x: The u64 value from user space, usually via IOCTL
//!
//! u64_to_user_ptr() simply casts a pointer passed as u64 from user space to void
//! __user * correctly. Using this lets us get rid of all the tiresome casts.
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
//! Sign Flag
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
//! Needs two entries
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! The size of each LDT entry.
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
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
//! writeable
//! Operation not permitted
//! Value too large for defined data type
//! restart by calling sys_restart_syscall
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
//! To support Context Analysis, we need to allow the compiler to see the
//! acquisition and release of the context lock. However, the "cleanup" helpers
//! wrap the lock in a struct passed through separate helper functions, which
//! hides the lock alias from the compiler (no inter-procedural analysis).
//!
//! To make it work, we introduce an explicit alias to the context lock instance
//! that is "cleaned" up with a separate cleanup helper. This helper is a dummy
//! function that does nothing at runtime, but has the "_unlock" attribute to
//! tell the compiler what happens at the end of the scope.
//!
//! To generalize the pattern, the WITH_LOCK_GUARD_1_ATTRS() macro should be used
//! to redefine the constructor, which then also creates the alias variable with
//! the right "cleanup" attribute, *after* DECLARE_LOCK_GUARD_1_ATTRS() has been
//! used.
//!
//! Example usage:
//!
//!   DECLARE_LOCK_GUARD_1_ATTRS(mutex, __acquires(_T), __releases(*(struct mutex **)_T))
//!   #define class_mutex_constructor(_T) WITH_LOCK_GUARD_1_ATTRS(mutex, _T)
//!
//! Note: To support the for-loop based scoped helpers, the auxiliary variable
//! must be a pointer to the "class" type because it is defined in the same
//! statement as the guard variable. However, we initialize it with the lock
//! pointer (despite the type mismatch, the compiler's alias analysis still works
//! as expected). The "_unlock" attribute receives a pointer to the auxiliary
//! variable (a double pointer to the class type), and must be cast and
//! dereferenced appropriately.
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
//! strscpy - Copy a C-string into a sized buffer
//! @dst: Where to copy the string to
//! @src: Where to copy the string from
//! @...: Size of destination buffer (optional)
//!
//! Copy the source string @src, or as much of it as fits, into the
//! destination @dst buffer. The behavior is undefined if the string
//! buffers overlap. The destination @dst buffer is always NUL terminated,
//! unless it's zero-sized.
//!
//! The size argument @... is only required when @dst is not an array, or
//! when the copy needs to be smaller than sizeof(@dst).
//!
//! Preferred to strncpy() since it always returns a valid string, and
//! doesn't unnecessarily force the tail of the destination buffer to be
//! zero padded. If padding is desired please use strscpy_pad().
//!
//! Returns the number of characters copied in @dst (not including the
//! trailing %NUL) or -E2BIG if @size is 0 or the copy from @src was
//! truncated.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
//! used to decode ioctl numbers..
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
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! signal pending
//! This function is needed by all SMP systems. It must _always_ be valid
//! from the initial startup.
//! New user namespace
//! These constants are defined in the MSan LLVM instrumentation pass.
//! max stack size
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
//! wait_queue_entry::flags
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
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
//! percpu_counter batch for local add or sub
//! Uprobes on this MM assume 32-bit code
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
//! Helper macro to avoid gfp flags if they are the default one
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
//! MXCSR will be initialized
//! mprotect needs to preserve PAT and encryption bits when updating
//! vm_page_prot
//! Extract the high bits for type
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
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! no-op
//! per-IO O_SYNC
//! File supports atomic writes
//! number of env pointers
//! Generate SD flag indexes
//! Resource type
//! Desired design of maximum size and alignment (see RFC2553)
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
//! Must be a compile-time constant, so implement it as a macro
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! 31 pointers + header align the folio_batch structure to a power of two
//! Flags in backing_dev_info::capability
//!
//! BDI_CAP_WRITEBACK:		Supports dirty page writeback, and dirty pages
//! 				should contribute to accounting
//! BDI_CAP_STRICTLIMIT:		Keep number of dirty pages below bdi threshold
//! Internal flags that share the struct mempolicy flags word with
//! "mode flags".  These flags are allocated from bit 0 up, as they
//! are never OR'ed into the mode in mempolicy API arguments.
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! swapcache pages are stored in the swapper_space radix tree.  We want to
//! get good packing density in that tree, so the index should be dense in
//! the low-order bits.
//!
//! We arrange the `type' and `offset' fields so that `type' is at the six
//! high-order bits of the swp_entry_t and `offset' is right-aligned in the
//! remaining bits.  Although `type' itself needs only five bits, we allow for
//! shmem/tmpfs to shift it all up a further one bit: see swp_to_radix_entry().
//!
//! swp_entry_t's are *never* stored anywhere in their arch-dependent format.
//! Temporary until swp_entry_t eliminated.
//! page can be read
//! stack-like segment
//! If more than 16 keys are ever supported, a thorough audit
//! will be necessary to ensure that the types that store key
//! numbers and masks have sufficient capacity.
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! Set global bits for all 4 regs
//! Define bits that are always set to 1 in DR7, only bit 10 is
//! architecturally reserved to '1'.
//!
//! This is also the init/reset value for DR7.
//! Replace with "lkgs %di" once binutils support LKGS instruction
//! Macro to enforce the same ordering and stack sizes
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! The PERF_METRICS MSR is modeled as several magic fixed-mode PMCs, one for
//! each TopDown metric event.
//!
//! Internally the TopDown metric events are mapped to the FxCtr 3 (SLOTS).
//! This is a list of *what* is being read, not *how* nor *where*.
//! ld/ldx fields
//! Define a list of socket types which can be the argument for
//! skc_to_*_sock() helpers. All these sockets should have
//! sock_common as the first argument in its memory layout.
//! Choose 31 as it makes rqspinlock_held cacheline-aligned.
//! Disallow growing, shrinking, or hole punching in the inode. Combined with
//! folio pinning, makes sure the inode's mapping stays fixed.
//!
//! In some ways similar to F_SEAL_GROW | F_SEAL_SHRINK, but can be removed and
//! isn't directly visible to userspace.
//! INVLPGB can be targeted by virtual address, PCID, ASID, or any combination
//! of the three. For example:
//! - FLAG_VA | FLAG_INCLUDE_GLOBAL: invalidate all TLB entries at the address
//! - FLAG_PCID:			    invalidate all TLB entries matching the PCID
//!
//! The first is used to invalidate (kernel) mappings at a particular
//! address across all processes.
//!
//! The latter invalidates all TLB entries matching a PCID.
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
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-weak-function-attribute
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-weak-variable-attribute
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
// Make the optimizer believe the variable can be manipulated arbitrarily.
// Format: __UNIQUE_ID_<name>_<__COUNTER__>
// &a[0] degrades to a pointer: a different type from an array
// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
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
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
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
// Concatenate two parameters, but allow them to be expanded beforehand.
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "apic" Onboard APIC
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "hypervisor" Running on a hypervisor
// "3dnowprefetch" 3DNow prefetch instructions
// "pti" Kernel Page Table Isolation enabled
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// Xen paravirtual guest
// Intel-defined CPU features, CPUID level 0x00000007:0 (EBX), word 9
// "fsgsbase" RDFSBASE, WRFSBASE, RDGSBASE, WRGSBASE instructions
// "erms" Enhanced REP MOVSB/STOSB instructions
// "smap" Supervisor Mode Access Prevention
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// IA32_TSC_DEADLINE and X2APIC MSRs need fencing
// "lass" Linear Address Space Separation
// Fast short REP STOSB
// "fred" Flexible Return and Event Delivery
// Like MOV_GS except MSR_KERNEL_GS_BASE = GS.base
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
// round_up - round up to next specified power of 2
// @x: the value to round
// @y: multiple to round up to (must be a power of 2)
//
// Rounds @x up to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding up, use roundup() below.
// round_down - round down to next specified power of 2
// @x: the value to round
// @y: multiple to round down to (must be a power of 2)
//
// Rounds @x down to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding down, use rounddown() below.
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// min_t - return minimum of two values, using the specified type
// @type: data type to use
// @x: first value
// @y: second value
// in_range - Determine if a value lies within a range.
// @val: Value to test.
// @start: First value in range.
// @len: Number of values in range.
//
// This is more efficient than "if (start <= val && val < (start + len))".
// It also gives a different answer if @start + @len overflows the size of
// the type by a sufficient amount to encompass @val.  Decide for yourself
// which behaviour you want, or prove that start + len never overflow.
// Do not blindly replace one form with the other.
// swap - swap values of @a and @b
// @a: first value
// @b: second value
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// Overflow Flag
// Needs two entries
// Number of entries in the GDT table:
// Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
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
// No such file or directory
// Interrupted system call
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Permission denied
// Bad address
// Device or resource busy
// No such device
// Invalid argument
// Text file busy
// No space left on device
// Math result not representable
// Operation not supported on transport endpoint
// Memory page has hardware error
// Operation is not supported
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
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
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// SCHED_ISO: reserved but not implemented yet
// disable sas during sighandling
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Same as rb_first(), but O(1)
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
// Don't call further
// folio_page - Return a page from a folio.
// @folio: The folio.
// @n: The page number to return.
//
// @n is relative to the start of the folio.  This function does not
// check that the page number lies within @folio; the caller is presumed
// to have a reference to the page.
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
// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
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
// Shift up (to get rid of type), then down to get value
// Shift the offset up "too far" by TYPE bits, then down again
// The offset is inverted by a binary not operation to make the high
// physical bits set.
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
// kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
// kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
// and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
// __alloc_objs - Allocate objects of a given type using
// @KMALLOC: which size-based kmalloc wrapper to allocate with.
// @GFP: GFP flags for the allocation.
// @TYPE: type to allocate space for.
// @COUNT: how many @TYPE objects to allocate.
//
// Returns: Newly allocated pointer to (first) @TYPE of @COUNT-many
// allocated @TYPE objects, or NULL on failure.
// All kzalloc aliases for kmalloc_(obj|objs|flex).
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
// Generate SD flag bits
// Resource extended types
// No address assigned yet
// Implementation specific max size
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// identify shared policies
// Enabling zone reclaim means the page allocator will attempt to fulfill
// the allocation request on the current node by triggering reclaim and
// trying to shrink the current node.
// Fallback allocations on the next candidates in the zonelist are considered
// when reclaim fails to free up enough memory in the current node/zone.
//
// These bit locations are exposed in the vm.zone_reclaim_mode sysctl.
// New bits are OK, but existing bits should not be changed.
// Enable zone reclaim
// Writeout pages during reclaim
// Unmap pages during reclaim
// hlist_nulls_next_rcu - returns the element of the list after @node.
// @node: element of the list.
// hlist_nulls_pprev_rcu - returns the dereferenced pprev of @node.
// @node: element of the list.
// task_css_check - obtain css for (task, subsys) w/ extra access conds
// @task: the target task
// @subsys_id: the target subsystem ID
// @__c: extra condition expression to be passed to rcu_dereference_check()
//
// Return the cgroup_subsys_state for the (@task, @subsys_id) pair.  The
// synchronization rules are the same as task_css_set_check().
// "Poisoned" here is meant in the very general sense of "future accesses are
// invalid", instead of referring very specifically to hardware memory errors.
// This marker is meant to represent any of various different causes of this.
//
// Note that, when encountered by the faulting logic, PTEs with this marker will
// result in VM_FAULT_HWPOISON and thus regardless trigger hardware memory error
// logic.
// Indicates that, on fault, this PTE will case a SIGSEGV signal to be
// sent. This means guard markers behave in effect as if the region were mapped
// PROT_NONE, rather than if they were a memory hole or equivalent.
// page can be written
// page can be executed
// page may be used for atomic ops
// perform synchronous page faults for the mapping
// pages are locked
// These special macros can be used to get or set a debugging register
// PT_TRACE_* event enable flags
// We abuse bit 3 from flags to pass exact information, see
// perf_arch_misc_flags() and the comment with PERF_EFLAGS_EXACT.
// The implied mode when all bits are clear:
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
/// @a is a power of 2 value
macro_rules! ALIGN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!($x, $a)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__21 {
() => {
((core::mem::size_of::<[*mut vm_area_struct; 8]>() as u64) / (core::mem::size_of::<*mut vm_area_struct>() as u64)).wrapping_add(__must_be_array__21!() as u64)
};
}

/// ASSERT_EXCLUSIVE_WRITER - assert no concurrent writes to @var
///
/// Assert that there are no concurrent writes to @var; other readers are
/// allowed. This assertion can be used to specify properties of concurrent code,
/// where violation cannot be detected as a normal data race.
///
/// For example, if we only have a single writer, but multiple concurrent
/// readers, to avoid data races, all these accesses must be marked; even
/// concurrent marked writes racing with the single writer are bugs.
/// Unfortunately, due to being marked, they are no longer data races. For cases
/// like these, we can use the macro as follows:
///
/// .. code-block:: c
///
/// 	void writer(void) {
/// 		spin_lock(&update_foo_lock);
/// 		ASSERT_EXCLUSIVE_WRITER(shared_foo);
/// 		WRITE_ONCE(shared_foo, ...);
/// 		spin_unlock(&update_foo_lock);
/// 	}
/// 	void reader(void) {
/// 		// update_foo_lock does not need to be held!
/// 		... = READ_ONCE(shared_foo);
/// 	}
///
/// Note: ASSERT_EXCLUSIVE_WRITER_SCOPED(), if applicable, performs more thorough
/// checking if a clear scope where no concurrent writes are expected exists.
///
/// @var: variable to assert on
macro_rules! ASSERT_EXCLUSIVE_WRITER {
($var:expr) => {
__kcsan_check_access((unsafe { core::ptr::addr_of_mut!($var) }) as *const (), core::mem::size_of::<seqcount>() as u64, 8i32);
};
}

macro_rules! BITMAP_LAST_WORD_MASK {
($nbits:expr) => {
18446744073709551615u64 >> ($nbits.wrapping_neg() & 63u32)
};
}

macro_rules! BITS_PER_TYPE {
($type:ty) => {
(core::mem::size_of::<$type>() as u64).wrapping_mul(8i32 as $type)
};
}

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

macro_rules! BIT_WORD__2 {
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

/// BUILD_BUG_ON_INVALID() permits the compiler to check the validity of the
/// expression but avoids the generation of any code, even if that expression
/// has side-effects.
macro_rules! BUILD_BUG_ON_INVALID {
() => {
};
}

macro_rules! GENMASK_INPUT_CHECK__3 {
() => {
core::mem::size_of::<__SlateRecord_vma_h_302>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__2 {
() => {
core::mem::size_of::<__SlateRecord_find_h_207>() as i32
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__2 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__2!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__3 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__3!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK__4 {
($h:expr, $l:expr) => {
GENMASK_TYPE__3!(u64, $h, $l)
};
}

macro_rules! GENMASK__2 {
($h:expr, $l:expr) => {
GENMASK_TYPE__2!(u64, $h, $l)
};
}

macro_rules! IS_ALIGNED__2 {
($x:expr, $a:expr) => {
($x) & (($a) as u32).wrapping_sub(1) == 0
};
}

macro_rules! IS_DAX {
($inode:expr) => {
(unsafe { (*$inode).i_flags }) & 0
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

/// to align the pointer to the (next) page boundary
macro_rules! PAGE_ALIGN {
($addr:expr) => {
ALIGN!($addr, 4096u64)
};
}

macro_rules! PHYS_PFN {
($x:expr) => {
($x) >> 12i32
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

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! VM_BUG_ON_VMA {
() => {
VM_BUG_ON!();
};
}

macro_rules! VM_BUG_ON {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_WARN_ON_ONCE_VMA {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_WARN_ON_ONCE {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_WARN_ON_VMG {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_WARN_ON {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! _AT {
($T:ty, $X:expr) => {
($X) as $T
};
}

macro_rules! _UL {
() => {
_AC!()
};
}

macro_rules! __ALIGN_KERNEL_MASK {
($x:expr, $mask:expr) => {
$x.wrapping_add($mask) & !$mask
};
}

macro_rules! __ALIGN_KERNEL {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as u64).wrapping_sub(1))
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

macro_rules! __WRITE_ONCE__19 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut (), $val) };
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

macro_rules! __assume_ctx_lock {
($var:expr) => {
loop {
$var;
if true {
break;
}
}
};
}

macro_rules! __const_hweight16 {
($w:expr) => {
__const_hweight8!($w).wrapping_add(__const_hweight8!(($w) >> 8i32))
};
}

macro_rules! __const_hweight32 {
($w:expr) => {
__const_hweight16!($w).wrapping_add(__const_hweight16!(($w) >> 16i32))
};
}

macro_rules! __const_hweight64 {
($w:expr) => {
__const_hweight32!($w).wrapping_add(__const_hweight32!(($w) >> 32i32))
};
}

/// Compile time versions of __arch_hweightN()
macro_rules! __const_hweight8 {
($w:expr) => {
((($w) & 1u64 != 0) as i32).wrapping_add((($w) & 2u64 != 0) as i32).wrapping_add((($w) & 4u64 != 0) as i32).wrapping_add((($w) & 8u64 != 0) as i32).wrapping_add((($w) & 16u64 != 0) as i32).wrapping_add((($w) & 32u64 != 0) as i32).wrapping_add((($w) & 64u64 != 0) as i32).wrapping_add((($w) & 128u64 != 0) as i32) as u32
};
}

/// __ffs - find first set bit in word
/// @word: The word to search
///
/// Undefined if no bit exists, so code should check against 0 first.
macro_rules! __ffs {
($word:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($word) } { ($word.trailing_zeros() as i32) as u64 } else { variable__ffs($word) }
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

macro_rules! __must_be_array__21 {
() => {
core::mem::size_of::<__SlateRecord_vma_c_1897>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__34 {
() => {
((const { core::mem::size_of::<*mut anon_vma>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut anon_vma>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut anon_vma>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut anon_vma>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__8 {
() => {
((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__24 {
() => {
((const { core::mem::size_of::<__SlateRecord_types_h_190>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_types_h_190>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_types_h_190>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_types_h_190>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__6 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
() => {
((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__9 {
() => {
((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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

/// The following macros are non-atomic versions of their non-underscored
/// counterparts.
macro_rules! __set_bit__2 {
($nr:expr, $addr:expr) => {
if (unsafe { __slate_intrinsic_a39e427d3b6c2df4($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___set_bit(($nr) as u64, ($addr) as *mut u64);
} else {
___set_bit(($nr) as u64, ($addr) as *mut u64);
}
};
}

/// Atomic operations are already serializing on x86
macro_rules! __smp_mb__before_atomic {
() => {
while false {
}
};
}

macro_rules! __test_and_clear_bit {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___test_and_clear_bit(($nr) as u64, ($addr) as *mut u64) as i32
} else {
___test_and_clear_bit(($nr) as u64, ($addr) as *mut u64) as i32
};
__v
}
};
}

macro_rules! __test_and_set_bit {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___test_and_set_bit(($nr) as u64, ($addr) as *mut u64) as i32
} else {
___test_and_set_bit(($nr) as u64, ($addr) as *mut u64) as i32
};
__v
}
};
}

/// We need to compute the minimum and maximum values representable in a given
/// type. These macros may also be useful elsewhere. It would seem more obvious
/// to do something like:
///
/// #define type_min(T) (T)(is_signed_type(T) ? (T)1 << (8*sizeof(T)-1) : 0)
/// #define type_max(T) (T)(is_signed_type(T) ? ((T)1 << (8*sizeof(T)-1)) - 1 : ~(T)0)
///
/// Unfortunately, the middle expressions, strictly speaking, have
/// undefined behaviour, and at least some versions of gcc warn about
/// the type_max expression (but not if -fsanitize=undefined is in
/// effect; in that case, the warning is deferred to runtime...).
///
/// The slightly excessive casting in type_min is to make sure the
/// macros also produce sensible values for the exotic type _Bool. [The
/// overflow checkers only almost work for _Bool, but that's
/// a-feature-not-a-bug, since people shouldn't be doing arithmetic on
/// _Bools. Besides, the gcc builtins don't allow _Bool* as third
/// argument.]
///
/// Idea stolen from
/// https://mail-index.netbsd.org/tech-misc/2007/02/05/0000.html -
/// credit to Christian Biere.
macro_rules! __type_half_max {
() => {
1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((is_signed_type!() as i32) as u64)
};
}

macro_rules! __type_max {
() => {
__type_half_max!().wrapping_sub(1).wrapping_add(__type_half_max!())
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

macro_rules! absolute_pointer {
($val:expr) => {
RELOC_HIDE!(($val) as *mut (), 0i32)
};
}

macro_rules! arch_atomic_fetch_sub {
($i:expr, $v:expr) => {
arch_atomic_fetch_add($i.wrapping_neg(), $v)
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__18 {
($ptr:expr) => {
{
let mut __v: *mut anon_vma_chain;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__16!() as isize)) }) as *mut ()) as *mut anon_vma_chain;
__v
}
};
}

macro_rules! current_thread_info {
() => {
get_current() as *mut thread_info
};
}

/// data_race - mark an expression as containing intentional data races
///
/// This data_race() macro is useful for situations in which data races
/// should be forgiven.  One example is diagnostic code that accesses
/// shared variables but is not a part of the core synchronization design.
/// For example, if accesses to a given variable are protected by a lock,
/// except for diagnostic code, then the accesses under the lock should
/// be plain C-language accesses and those in the diagnostic code should
/// use data_race().  This way, KCSAN will complain if buggy lockless
/// accesses to that variable are introduced, even if the buggy accesses
/// are protected by READ_ONCE() or WRITE_ONCE().
///
/// This macro *does not* affect normal code generation, but is a hint
/// to tooling that data races here are to be ignored.  If the access must
/// be atomic *and* KCSAN should ignore the access, use both data_race()
/// and READ_ONCE(), for example, data_race(READ_ONCE(x)).
macro_rules! data_race__2 {
($expr:expr) => {
{
let mut __v: u64;
__kcsan_disable_current();
let __v2: u64 = $expr;
__kcsan_enable_current();
__v = __v2;
__v
}
};
}

macro_rules! down_write_nest_lock {
($sem:expr) => {
unsafe { down_write($sem) };
};
}

macro_rules! hweight32 {
($w:expr) => {
{
let __v: u32 = if unsafe { __slate_intrinsic_f217c4720f04f6db($w) } {
__const_hweight32!($w)
} else {
__arch_hweight32(($w) as u32)
};
__v
}
};
}

macro_rules! hweight64 {
($w:expr) => {
{
let __v: u64 = if unsafe { __slate_intrinsic_f217c4720f04f6db($w) } {
__const_hweight64!($w) as u64
} else {
__arch_hweight64($w)
};
__v
}
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
(1i32.wrapping_neg() as u64) < 1
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

/// kcsan_check_read - check regular read access for races
///
/// @ptr: address of access
/// @size: size of access
macro_rules! kcsan_check_read {
($ptr:expr, $size:expr) => {
kcsan_check_access(($ptr) as *const (), $size, 0i32);
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

/// kcsan_check_write - check regular write access for races
///
/// @ptr: address of access
/// @size: size of access
macro_rules! kcsan_check_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 1i32);
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

macro_rules! kcsan_wmb {
() => {
while false {
}
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

macro_rules! likely__8 {
($x:expr) => {
!$x.is_null() as i64
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__4 {
($ptr:expr) => {
container_of__18!($ptr)
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
macro_rules! list_first_entry__3 {
($ptr:expr) => {
list_entry__4!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__3 {
($pos:expr, $member:ident) => {
list_entry__4!(unsafe { (*$pos).$member.next })
};
}

macro_rules! lockdep_assert_held_write {
($l:expr) => {
__assume_ctx_lock!($l);
};
}

macro_rules! might_sleep {
() => {
loop {
might_resched();
if true {
break;
}
}
};
}

macro_rules! mt_on_stack {
() => {
while false {
}
};
}

macro_rules! offset_in_page {
($p:expr) => {
(($p) as u64) & !(!(_AC!() << 12i32).wrapping_sub(1))
};
}

macro_rules! offsetof__16 {
() => {
core::mem::offset_of!(anon_vma_chain, same_vma) as u64
};
}

macro_rules! pgprot_val {
($x:expr) => {
$x.pgprot
};
}

macro_rules! pgtable_supports_soft_dirty {
() => {
IS_ENABLED!()
};
}

/// small_const_nbits(n) is true precisely when it is known at compile-time
/// that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
/// various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
/// of size 0 are very rare, and a compile-time-known-size 0 is most likely
/// a sign of error. They will be handled correctly by the bit/bitmap APIs,
/// but using the out-of-line functions, so that the inline implementations
/// can unconditionally dereference the pointer(s).
macro_rules! small_const_nbits {
($nbits:expr) => {
(unsafe { __slate_intrinsic_1c5364fcd9da51fe($nbits) }) && ($nbits) <= 64 && ($nbits) > 0
};
}

/// small_const_nbits(n) is true precisely when it is known at compile-time
/// that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
/// various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
/// of size 0 are very rare, and a compile-time-known-size 0 is most likely
/// a sign of error. They will be handled correctly by the bit/bitmap APIs,
/// but using the out-of-line functions, so that the inline implementations
/// can unconditionally dereference the pointer(s).
macro_rules! small_const_nbits__2 {
($nbits:expr) => {
(unsafe { __slate_intrinsic_f217c4720f04f6db($nbits) }) && ($nbits) <= 64 && ($nbits) > 0
};
}

macro_rules! static_branch_unlikely__3 {
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

macro_rules! static_branch_unlikely {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const3(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__2 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const2(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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

macro_rules! test_bit__5 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_a39e427d3b6c2df4($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_bit__4 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_thread_flag {
($flag:expr) => {
test_ti_thread_flag(current_thread_info!(), $flag)
};
}

macro_rules! type_max {
() => {
__type_max!()
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

macro_rules! validate_mm {
() => {
while false {
}
};
}

macro_rules! vma_policy {
($vma:expr) => {
unsafe { (*$vma).vm_policy }
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __SCK__might_resched: static_call_key;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut node_data: [*mut pglist_data; 0];
    static mut cpuinfo_op: seq_operations;
    static mut sighand_cachep: *mut kmem_cache;
    static mut __tracepoint_mmap_lock_start_locking: tracepoint;
    static mut __tracepoint_mmap_lock_acquire_returned: tracepoint;
    static mut __tracepoint_mmap_lock_released: tracepoint;
    static mut vma_dummy_vm_ops: vm_operations_struct;
    static mut stack_guard_gap: u64;
    static mut kstat: kernel_stat;
    static mut vm_committed_as: percpu_counter;
    static mut vm_committed_as_batch: i32;
    static mut mmap_min_addr: u64;
    static mut perf_callchain_entry: perf_callchain_entry;
    static mut sysctl_max_map_count: i32;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_5();
    fn ____wrong_branch_error() -> bool;
    fn __cmpxchg_wrong_size();
    fn __xadd_wrong_size();
    fn __compiletime_assert_8();
    fn __compiletime_assert_9();
    fn _find_first_bit(arg0: *const u64, arg1: u64) -> u64;
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn memcmp(arg0: *const (), arg1: *const (), arg2: u64) -> i32;
    fn __bitmap_equal(arg0: *const u64, arg1: *const u64, arg2: u32) -> bool;
    fn __bitmap_and(arg0: *mut u64, arg1: *const u64, arg2: *const u64, arg3: u32) -> bool;
    fn __bitmap_or(arg0: *mut u64, arg1: *const u64, arg2: *const u64, arg3: u32);
    fn __bitmap_xor(arg0: *mut u64, arg1: *const u64, arg2: *const u64, arg3: u32);
    fn __bitmap_andnot(arg0: *mut u64, arg1: *const u64, arg2: *const u64, arg3: u32) -> bool;
    fn __bitmap_intersects(arg0: *const u64, arg1: *const u64, arg2: u32) -> bool;
    fn __bitmap_subset(arg0: *const u64, arg1: *const u64, arg2: u32) -> bool;
    fn __bitmap_weight(arg0: *const u64, arg1: u32) -> u32;
    fn __compiletime_assert_20();
    fn refcount_warn_saturate(arg0: *mut refcount_struct, arg1: u32);
    fn mutex_is_locked(arg0: *mut mutex) -> bool;
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn __mt_destroy(arg0: *mut maple_tree);
    fn mas_walk(arg0: *mut ma_state) -> *mut ();
    fn mas_store_gfp(arg0: *mut ma_state, arg1: *mut (), arg2: u32) -> i32;
    fn mas_store_prealloc(arg0: *mut ma_state, arg1: *mut ());
    fn mas_find(arg0: *mut ma_state, arg1: u64) -> *mut ();
    fn mas_preallocate(arg0: *mut ma_state, arg1: *mut (), arg2: u32) -> i32;
    fn mas_pause(arg0: *mut ma_state);
    fn mas_destroy(arg0: *mut ma_state);
    fn mas_prev(arg0: *mut ma_state, arg1: u64) -> *mut ();
    fn mas_prev_range(arg0: *mut ma_state, arg1: u64) -> *mut ();
    fn mas_next_range(arg0: *mut ma_state, arg1: u64) -> *mut ();
    fn mas_empty_area(arg0: *mut ma_state, arg1: u64, arg2: u64, arg3: u64) -> i32;
    fn mas_empty_area_rev(arg0: *mut ma_state, arg1: u64, arg2: u64, arg3: u64) -> i32;
    fn __compiletime_assert_147();
    fn __compiletime_assert_148();
    fn __compiletime_assert_149();
    fn down_write(arg0: *mut rw_semaphore);
    fn down_write_killable(arg0: *mut rw_semaphore) -> i32;
    fn up_read(arg0: *mut rw_semaphore);
    fn up_write(arg0: *mut rw_semaphore);
    fn downgrade_write(arg0: *mut rw_semaphore);
    fn uprobe_mmap(arg0: *mut vm_area_struct) -> i32;
    fn uprobe_munmap(arg0: *mut vm_area_struct, arg1: u64, arg2: u64);
    fn percpu_counter_add_batch(arg0: *mut percpu_counter, arg1: i64, arg2: i32);
    fn __compiletime_assert_183();
    fn tlb_gather_mmu(arg0: *mut mmu_gather, arg1: *mut mm_struct);
    fn tlb_finish_mmu(arg0: *mut mmu_gather);
    fn __mmap_lock_do_trace_start_locking(arg0: *mut mm_struct, arg1: bool);
    fn __mmap_lock_do_trace_acquire_returned(arg0: *mut mm_struct, arg1: bool, arg2: bool);
    fn __mmap_lock_do_trace_released(arg0: *mut mm_struct, arg1: bool);
    fn __vma_start_write(arg0: *mut vm_area_struct, arg1: i32) -> i32;
    fn __vma_exclude_readers_for_detach(arg0: *mut vm_area_struct);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_359();
    fn __compiletime_assert_384();
    fn compat_vma_mmap(arg0: *mut file, arg1: *mut vm_area_struct) -> i32;
    fn vm_area_alloc(arg0: *mut mm_struct) -> *mut vm_area_struct;
    fn vm_area_dup(arg0: *mut vm_area_struct) -> *mut vm_area_struct;
    fn vm_area_free(arg0: *mut vm_area_struct);
    fn mapping_rmap_tree_insert(arg0: *mut vm_area_struct, arg1: *mut address_space);
    fn mapping_rmap_tree_remove(arg0: *mut vm_area_struct, arg1: *mut address_space);
    fn anon_rmap_tree_insert(arg0: *mut anon_vma_chain, arg1: *mut anon_vma);
    fn anon_rmap_tree_remove(arg0: *mut anon_vma_chain, arg1: *mut anon_vma);
    fn vm_stat_account(arg0: *mut mm_struct, arg1: u64, arg2: i64);
    fn find_vma_prev(arg0: *mut mm_struct, arg1: u64, arg2: *mut *mut vm_area_struct) -> *mut vm_area_struct;
    fn find_vma_intersection(arg0: *mut mm_struct, arg1: u64, arg2: u64) -> *mut vm_area_struct;
    fn mmap_action_prepare(arg0: *mut vm_area_desc) -> i32;
    fn mmap_action_complete(arg0: *mut vm_area_struct, arg1: *mut mmap_action, arg2: bool) -> i32;
    fn vm_get_page_prot(arg0: u64) -> pgprot;
    fn vma_set_page_prot(arg0: *mut vm_area_struct);
    fn get_gate_vma(arg0: *mut mm_struct) -> *mut vm_area_struct;
    fn inode_to_bdi(arg0: *mut inode) -> *mut backing_dev_info;
    fn fput(arg0: *mut file);
    fn __mpol_put(arg0: *mut mempolicy);
    fn __mpol_equal(arg0: *mut mempolicy, arg1: *mut mempolicy) -> bool;
    fn vma_dup_policy(arg0: *mut vm_area_struct, arg1: *mut vm_area_struct) -> i32;
    fn fixup_hugetlb_reservations(arg0: *mut vm_area_struct);
    fn hugetlb_split(arg0: *mut vm_area_struct, arg1: u64);
    fn security_vm_enough_memory_mm(arg0: *mut mm_struct, arg1: i64) -> i32;
    fn perf_event_mmap(arg0: *mut vm_area_struct);
    fn shmem_zero_setup(arg0: *mut vm_area_struct) -> i32;
    fn shmem_mapping(arg0: *const address_space) -> bool;
    fn unmap_vmas(arg0: *mut mmu_gather, arg1: *mut unmap_desc);
    fn anon_vma_clone(arg0: *mut vm_area_struct, arg1: *mut vm_area_struct, arg2: u32) -> i32;
    fn __anon_vma_prepare(arg0: *mut vm_area_struct) -> i32;
    fn unlink_anon_vmas(arg0: *mut vm_area_struct);
    fn free_pgtables(arg0: *mut mmu_gather, arg1: *mut unmap_desc);
    fn mlock_future_ok(arg0: *const mm_struct, arg1: bool, arg2: u64) -> bool;
    fn __compiletime_assert_722();
    fn may_expand_vm(arg0: *mut mm_struct, arg1: *const __SlateRecord_mm_types_h_871, arg2: u64) -> bool;
    fn __compiletime_assert_730();
    fn __compiletime_assert_729();
    fn __compiletime_assert_731();
    fn __builtin_unreachable();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
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
struct __SlateRecord_types_h_202 {
    refcnt: __SlateRecord_types_h_190,
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

fn __kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

fn kcsan_nestable_atomic_begin() {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
}


fn kcsan_nestable_atomic_end() {
}

fn kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

fn __kcsan_enable_current() {
}

fn __kcsan_disable_current() {
}

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


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
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


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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


#[inline(always)]
fn generic___set_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
unsafe {
*p = *p | mask;
}
}

#[inline(always)]
fn generic___test_and_set_bit(nr: u64, addr: *mut u64) -> bool {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
let old: u64 = unsafe { *p };
unsafe {
*p = old | mask;
}
old & mask != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[inline(always)]
fn generic___test_and_clear_bit(nr: u64, addr: *mut u64) -> bool {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
let old: u64 = unsafe { *p };
unsafe {
*p = old & !mask;
}
old & mask != 0
}

// const_*() definitions provide good compile-time optimizations when
// the passed arguments can be resolved at compile time.
#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ptrace_h_144 {
    cs: u16,
    csx: u64,
    fred_cs: fred_cs,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ptrace_h_156 {
    ss: u16,
    ssx: u64,
    fred_ss: fred_ss,
}


#[inline(always)]
fn arch___set_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_test_and_set_bit(mut nr: i64, mut addr: *mut u64) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock  btsq  {2:r}, ({0:r})\nsetc {1}", in(reg) addr, lateout(reg_byte) __t0, in(reg) nr as u64, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

#[inline(always)]
fn arch___test_and_set_bit(mut nr: u64, mut addr: *mut u64) -> bool {
let mut oldbit: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!(" btsq  {2:r},({1:r})\nsetc {0}", lateout(reg_byte) __t0, in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
oldbit = (__t0 as u8) != 0;
}
oldbit
}

#[inline(always)]
fn arch_test_and_clear_bit(mut nr: i64, mut addr: *mut u64) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock  btrq  {2:r}, ({0:r})\nsetc {1}", in(reg) addr, lateout(reg_byte) __t0, in(reg) nr as u64, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

#[inline(always)]
fn arch___test_and_clear_bit(mut nr: u64, mut addr: *mut u64) -> bool {
let mut oldbit: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!(" btrq  {2:r},({1:r})\nsetc {0}", lateout(reg_byte) __t0, in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
oldbit = (__t0 as u8) != 0;
}
oldbit
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

#[inline(always)]
fn variable__ffs(mut word: u64) -> u64 {
unsafe {
let __t0: u64;
core::arch::asm!("tzcnt {1:r},{0:r}", lateout(reg) __t0, in(reg) word as u64, options(pure, nomem, nostack, att_syntax));
word = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
word
}

// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
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
fn __arch_hweight32(mut w: u32) -> u32 {
let mut res: u32 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("# ALT: oldinstr\n771:\n\tcall __sw_hweight32\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 4*32+23)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tpopcntl %edi, %eax\n775:\n.popsection\n", lateout("ax") __t0, in("di") w as u32, options(pure, nomem, att_syntax));
res = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
res
}

#[inline(always)]
fn __arch_hweight64(mut w: u64) -> u64 {
let mut res: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("# ALT: oldinstr\n771:\n\tcall __sw_hweight64\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 4*32+23)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tpopcntq %rdi, %rax\n775:\n.popsection\n", lateout("ax") __t0, in("di") w as u64, options(pure, nomem, att_syntax));
res = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
res
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn instrument_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_write!(v, size);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("2: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 2b - ., 3; .popsection\n# {0}", const 2 as i32, options(nostack, att_syntax));
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
core::arch::asm!("3: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 3b - ., 4; .popsection\n# {0}", const 3 as i32, options(nostack, att_syntax));
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

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_207 {
}


#[inline(always)]
fn instrument_atomic_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_read_write!(v, size);
instrument_atomic_check_alignment(v, size);
}

// instrument_get_user() - add instrumentation to get_user()-like macros
// @to: destination variable, may not be address-taken
//
// get_user() and friends are fragile, so it may depend on the implementation
// whether the instrumentation happens before or after the data is copied from
// the userspace.
#[inline(always)]
fn test_and_set_bit(nr: i64, addr: *mut u64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_set_bit(nr, addr)
}

#[inline(always)]
fn test_and_clear_bit(nr: i64, addr: *mut u64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_clear_bit(nr, addr)
}

#[inline(always)]
fn ___set_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___set_bit(nr, addr);
}

#[inline(always)]
fn __instrument_read_write_bitop(nr: i64, addr: *mut u64) {
if IS_ENABLED!() != 0 {
// We treat non-atomic read-write bitops a little more special.
// Given the operations here only modify a single bit, assuming
// non-atomicity of the writer is sufficient may be reasonable
// for certain usage (and follows the permissible nature of the
// assume-plain-writes-atomic rule):
// 1. report read-modify-write races -> check read;
// 2. do not report races with marked readers, but do report
//    races with unmarked readers -> check "atomic" write.
kcsan_check_read!(addr.wrapping_offset(BIT_WORD__2!(nr) as isize), core::mem::size_of::<i64>() as u64);
// Use generic write instrumentation, in case other sanitizers
// or tools are enabled alongside KCSAN.
instrument_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
} else {
instrument_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
}
}

#[inline(always)]
fn ___test_and_set_bit(nr: u64, addr: *mut u64) -> bool {
__instrument_read_write_bitop(nr as i64, addr);
arch___test_and_set_bit(nr, addr)
}

#[inline(always)]
fn ___test_and_clear_bit(nr: u64, addr: *mut u64) -> bool {
__instrument_read_write_bitop(nr as i64, addr);
arch___test_and_clear_bit(nr, addr)
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


// Check that the bitops prototypes are sane
#[inline(always)]
fn hweight_long(w: u64) -> u64 {
let __v0: u64 = if (core::mem::size_of::<u64>() as u64) == 4 {
hweight32!(w) as u64
} else {
hweight64!(w)
};
__v0
}

// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
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
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
}

// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
// @n: parameter
//
// constant-capable log of base 2 calculation
// - this can be used to initialise global variables from constant data, hence
// the massive ternary operator construction
//
// selects the appropriately-sized optimised version depending on sizeof(n)
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
#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_4: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timespec64 {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fred_cs {
    __slate_bits_0: __slate_bits::__SlateBits104U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fred_ss {
    __slate_bits_0: __slate_bits::__SlateBits105U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pt_regs {
    r15: u64,
    r14: u64,
    r13: u64,
    r12: u64,
    bp: u64,
    bx: u64,
    r11: u64,
    r10: u64,
    r9: u64,
    r8: u64,
    ax: u64,
    cx: u64,
    dx: u64,
    si: u64,
    di: u64,
    orig_ax: u64,
    ip: u64,
    __slate_anon_17: __SlateRecord_ptrace_h_144,
    flags: u64,
    sp: u64,
    __slate_anon_20: __SlateRecord_ptrace_h_156,
}


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_5() };
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

#[inline(always)]
fn task_size_max() -> u64 {
let mut ret: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("# ALT: oldinstr\n771:\n\tmovq ${2},{0:r}\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte (16*32+16)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tmovq ${3},{0:r}\n775:\n.popsection\n\n# {1}", lateout(reg) __t0, const 0 as i32, const 140737488351232 as u64, const 72057594037923840 as u64, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
ret
}

#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_21 {
    pte: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_22 {
    pmd: u64,
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
// PGDIR_SHIFT determines what a top-level page table entry can map
// 4th level page in 5-level paging case
// 3rd level page
// PMD_SHIFT determines the size of the area a middle-level
// page table can map
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
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
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_368 {
    pud: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits130U0,
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
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


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[inline(always)]
fn arch_static_branch__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mmap_lock_start_locking, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const2(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mmap_lock_acquire_returned, const 0 as u8, label {
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
struct seccomp_filter {
}


#[inline(always)]
fn arch_static_branch__const3(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mmap_lock_released, const 0 as u8, label {
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits137U0,
}


#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
}

#[inline(always)]
fn arch_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
__WRITE_ONCE__3!((*v).counter, i);
}

#[inline(always)]
fn arch_atomic_inc(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock incl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
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


#[inline(always)]
fn arch_atomic_dec(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock decl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_types_h_14 {
    si_signo: i32,
    si_errno: i32,
    si_code: i32,
    _sifields: __sifields,
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

#[inline(always)]
fn arch_atomic_try_cmpxchg(v: *mut __SlateRecord_types_h_190, old: &mut i32, mut __new: i32) -> bool {
let mut __v0: i64;
let mut success: bool = false;
let mut __old: i32 = unsafe { *old };
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
if unlikely!(!success) != 0 {
unsafe {
*old = __old;
}
}
__v0 = success as i64;
__v0 != 0
}

#[inline(always)]
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE__2!((*v).counter)
}

#[inline(always)]
fn arch_atomic64_fetch_add(mut __ret: i64, mut v: *mut __SlateRecord_types_h_197) -> i64 {
let mut __v0: i64;
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
unsafe {
let __t0: u64;
core::arch::asm!("lock xaddb {0:l}, ({1:r})", inlateout(reg) __ret as u64 => __t0, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t0 as u64) };
}
}
2 => {
unsafe {
let __t1: u64;
core::arch::asm!("lock xaddw {0:x}, ({1:r})", inlateout(reg) __ret as u64 => __t1, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t1 as u64) };
}
}
4 => {
unsafe {
let __t2: u64;
core::arch::asm!("lock xaddl {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t2, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t2 as u64) };
}
}
8 => {
unsafe {
let __t3: u64;
core::arch::asm!("lock xaddq {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t3, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t3 as u64) };
}
}
_ => {
unsafe { __xadd_wrong_size() };
}
}
__v0 = __ret;
__v0
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
fn raw_atomic_set_release(v: *mut __SlateRecord_types_h_190, i: i32) {
if __native_word__24!() {
loop {
kcsan_release!();
loop {
loop {
if !__native_word__9!() {
unsafe { __compiletime_assert_8() };
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
if !(__native_word__9!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_9() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!(*unsafe { core::ptr::addr_of_mut!((*v).counter) }, i);
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
} else {
loop {
kcsan_mb!();
__smp_mb__before_atomic!();
if true {
break;
}
}
raw_atomic_set(v, i);
}
}

#[inline(always)]
fn raw_atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_sub!(i, v)
}

#[inline(always)]
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
}

#[inline(always)]
fn raw_atomic_dec(v: *mut __SlateRecord_types_h_190) {
arch_atomic_dec(v);
}

#[inline(always)]
fn raw_atomic_try_cmpxchg(v: *mut __SlateRecord_types_h_190, old: &mut i32, new: i32) -> bool {
arch_atomic_try_cmpxchg(v, old, new)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct net_device {
}


#[inline(always)]
fn raw_atomic_inc_unless_negative(v: *mut __SlateRecord_types_h_190) -> bool {
let mut c: i32 = raw_atomic_read(v as *const __SlateRecord_types_h_190);
loop {
if unlikely!(c < 0) != 0 {
return false;
}
if !(!raw_atomic_try_cmpxchg(v, unsafe { &mut (*core::ptr::addr_of_mut!(c)) }, c.wrapping_add(1))) {
break;
}
}
true
}

#[inline(always)]
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic64_fetch_add_relaxed(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
arch_atomic64_fetch_add(i, v)
}

#[inline(always)]
fn raw_atomic64_fetch_inc_relaxed(v: *mut __SlateRecord_types_h_197) -> i64 {
raw_atomic64_fetch_add_relaxed(1i64, v)
}

#[inline(always)]
fn raw_atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
raw_atomic64_read(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[inline(always)]
fn raw_atomic_long_fetch_inc_relaxed(v: *mut __SlateRecord_types_h_197) -> i64 {
raw_atomic64_fetch_inc_relaxed(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_read(v)
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


#[inline(always)]
fn atomic_set_release(v: *mut __SlateRecord_types_h_190, i: i32) {
kcsan_release!();
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set_release(v, i);
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
fn atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_release!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_sub_release(i, v)
}

#[inline(always)]
fn atomic_inc(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc(v);
}

#[inline(always)]
fn atomic_dec(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec(v);
}

#[inline(always)]
fn atomic_inc_unless_negative(v: *mut __SlateRecord_types_h_190) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc_unless_negative(v)
}

#[inline(always)]
fn atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_read(v)
}

#[inline(always)]
fn atomic_long_fetch_inc_relaxed(v: *mut __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_fetch_inc_relaxed(v)
}

#[inline(always)]
fn find_first_bit(addr: *const u64, size: u64) -> u64 {
if small_const_nbits__2!(size) {
let val: u64 = (unsafe { *addr }) & GENMASK__2!(size.wrapping_sub(1), 0i32);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_first_bit(addr, size) }
}

#[inline(always)]
fn bitmap_and(dst: *mut u64, src1: *const u64, src2: *const u64, nbits: u32) -> bool {
if small_const_nbits!(nbits) {
let __v0: u64 = (unsafe { *src1 }) & unsafe { *src2 } & BITMAP_LAST_WORD_MASK!(nbits);
unsafe {
*dst = __v0;
}
return __v0 != 0;
}
unsafe { __bitmap_and(dst, src1, src2, nbits) }
}

#[inline(always)]
fn bitmap_or(dst: *mut u64, src1: *const u64, src2: *const u64, nbits: u32) {
if small_const_nbits!(nbits) {
unsafe {
*dst = *src1 | *src2;
}
} else {
unsafe { __bitmap_or(dst, src1, src2, nbits) };
}
}

#[inline(always)]
fn bitmap_xor(dst: *mut u64, src1: *const u64, src2: *const u64, nbits: u32) {
if small_const_nbits!(nbits) {
unsafe {
*dst = *src1 ^ *src2;
}
} else {
unsafe { __bitmap_xor(dst, src1, src2, nbits) };
}
}

#[inline(always)]
fn bitmap_andnot(dst: *mut u64, src1: *const u64, src2: *const u64, nbits: u32) -> bool {
if small_const_nbits!(nbits) {
let __v0: u64 = (unsafe { *src1 }) & !unsafe { *src2 } & BITMAP_LAST_WORD_MASK!(nbits);
unsafe {
*dst = __v0;
}
return __v0 != 0;
}
unsafe { __bitmap_andnot(dst, src1, src2, nbits) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[inline(always)]
fn bitmap_equal(src1: *const u64, src2: *const u64, nbits: u32) -> bool {
if small_const_nbits!(nbits) {
return ((unsafe { *src1 }) ^ unsafe { *src2 }) & BITMAP_LAST_WORD_MASK!(nbits) == 0;
}
if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(nbits & 7u32) }) && IS_ALIGNED__2!(nbits, 8i32) {
return (unsafe { memcmp(src1 as *const (), src2 as *const (), (nbits / 8) as u64) }) == 0;
}
unsafe { __bitmap_equal(src1, src2, nbits) }
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
struct bpf_local_storage {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_net_context {
}


#[inline(always)]
fn bitmap_intersects(src1: *const u64, src2: *const u64, nbits: u32) -> bool {
if small_const_nbits!(nbits) {
return (unsafe { *src1 }) & unsafe { *src2 } & BITMAP_LAST_WORD_MASK!(nbits) != 0;
} else {
return unsafe { __bitmap_intersects(src1, src2, nbits) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cfs_rq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_struct {
}


#[inline(always)]
fn bitmap_subset(src1: *const u64, src2: *const u64, nbits: u32) -> bool {
if small_const_nbits!(nbits) {
return (unsafe { *src1 }) & !unsafe { *src2 } & BITMAP_LAST_WORD_MASK!(nbits) == 0;
} else {
return unsafe { __bitmap_subset(src1, src2, nbits) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_task {
}


#[inline(always)]
fn bitmap_empty(src: *const u64, nbits: u32) -> bool {
if small_const_nbits!(nbits) {
return (unsafe { *src }) & BITMAP_LAST_WORD_MASK!(nbits) == 0;
}
find_first_bit(src, nbits as u64) == (nbits as u64)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct nameidata {
}


#[inline(always)]
fn bitmap_weight(src: *const u64, nbits: u32) -> u32 {
if small_const_nbits!(nbits) {
return hweight_long((unsafe { *src }) & BITMAP_LAST_WORD_MASK!(nbits)) as u32;
}
unsafe { __bitmap_weight(src, nbits) }
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


#[repr(C)]
#[derive(Clone, Copy)]
struct pipe_inode_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_node {
}


// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
}


fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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

fn list_is_singular(head: *const list_head) -> i32 {
(list_empty(head) == 0 && (unsafe { (*head).next }) == unsafe { (*head).prev }) as i32
}

// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// sizeof(unsigned long) coming from an extra "long" at the end of the
// iobitmap. The limit is inclusive, i.e. the last valid byte.
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
    __slate_bits_0: __slate_bits::__SlateBits138U0,
    pkru: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_head {
    first: *mut llist_node,
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


// uses I/O bitmap
// 32-bit address space on 64 bits
fn test_ti_thread_flag(ti: *mut thread_info, flag: i32) -> i32 {
test_bit__2!(flag, unsafe { core::ptr::addr_of_mut!((*ti).flags) })
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
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
struct sysv_sem {
    undo_list: *mut sem_undo_list,
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
struct gendisk {
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
struct vm_struct {
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
struct timer_list {
    entry: hlist_node,
    expires: u64,
    function: Option<unsafe extern "C" fn(*mut timer_list)>,
    flags: u32,
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
struct mem_cgroup {
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
struct task_cputime {
    stime: u64,
    utime: u64,
    sum_exec_runtime: u64,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_pnode {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_103 {
    slot: [*mut (); 16],
    __slate_anon_1: __SlateRecord_maple_tree_h_105,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_105 {
    pad: [*mut (); 15],
    meta: maple_metadata,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_enode {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_162 {
    node: *mut maple_node,
    max: u64,
    mt: u32,
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
struct __SlateRecord_maple_tree_h_167 {
    node: *mut maple_node,
    max: u64,
    start: u8,
    end: u8,
    mt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_178 {
    pivot: [u64; 3],
    __slate_anon_1: __SlateRecord_maple_tree_h_180,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_180 {
    _pad: [*mut (); 2],
    max: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct k_sigaction {
    sa: sigaction,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_324 {
    __slate_anon_0: __SlateRecord_maple_tree_h_325,
    __slate_anon_1: __SlateRecord_maple_tree_h_329,
    mr64: maple_range_64,
    ma64: maple_arange_64,
    cp: maple_copy,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_325 {
    parent: *mut maple_pnode,
    slot: [*mut (); 31],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_329 {
    pad: *mut (),
    rcu: callback_head,
    piv_parent: *mut maple_enode,
    parent_slot: u8,
    r#type: u32,
    slot_len: u8,
    ma_flags: u32,
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
struct slab_sheaf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_ext {
    regfunc: Option<unsafe extern "C" fn() -> i32>,
    unregfunc: Option<unsafe extern "C" fn()>,
    __slate_bits_0: __slate_bits::__SlateBits232U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
}


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
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
struct uprobe_xol_ops {
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
struct irq_work {
    node: __call_single_node,
    func: Option<unsafe extern "C" fn(*mut irq_work)>,
    irqwait: rcuwait,
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
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct encoded_page {
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
struct unwind_cache {
    unwind_completed: u64,
    nr_entries: u32,
    entries: [u64; 0],
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
struct __SlateRecord_mm_types_h_1062 {
    rb: rb_node,
    rb_subtree_last: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits297U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
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
    __slate_bits_0: __slate_bits::__SlateBits72U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits72U1,
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
    __slate_bits_2: __slate_bits::__SlateBits72U2,
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
// I am kswapd
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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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


fn refcount_set_release(r: *mut refcount_struct, n: i32) {
atomic_set_release(unsafe { core::ptr::addr_of_mut!((*r).refs) }, n);
}

fn refcount_read(r: *const refcount_struct) -> u32 {
atomic_read(unsafe { core::ptr::addr_of!((*r).refs) }) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct memory_tier {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
}


fn __refcount_dec_and_test(r: *mut refcount_struct, oldp: *mut i32) -> bool {
__refcount_sub_and_test(1i32, r, oldp)
}

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
struct shrink_control {
    gfp_mask: u32,
    nid: i32,
    nr_to_scan: u64,
    nr_scanned: u64,
    memcg: *mut mem_cgroup,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
}


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
fn do_raw_write_seqcount_begin(__v0: *mut seqcount) {
kcsan_nestable_atomic_begin();
let __v1: u32 = unsafe { (*__v0).sequence };
unsafe {
(*__v0).sequence = __v1.wrapping_add(1);
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
}

fn do_raw_write_seqcount_end(__v2: *mut seqcount) {
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
let __v3: u32 = unsafe { (*__v2).sequence };
unsafe {
(*__v2).sequence = __v3.wrapping_add(1);
}
kcsan_nestable_atomic_end();
}

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
#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


// Dependency order vs. p above.
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
// 256 bytes including ->parent
// 256 bytes
// 240 bytes
#[repr(C)]
#[derive(Clone, Copy)]
struct maple_metadata {
    end: u8,
    gap: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_range_64 {
    parent: *mut maple_pnode,
    pivot: [u64; 15],
    __slate_anon_2: __SlateRecord_maple_tree_h_103,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_arange_64 {
    parent: *mut maple_pnode,
    pivot: [u64; 9],
    slot: [*mut (); 10],
    gap: [u64; 10],
    meta: maple_metadata,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sysctl_h_254 {
    __slate_anon_0: __SlateRecord_sysctl_h_255,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sysctl_h_255 {
    ctl_table: *const ctl_table,
    ctl_table_size: i32,
    used: i32,
    count: i32,
    nreg: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_copy {
    dst: [__SlateRecord_maple_tree_h_162; 3],
    src: [__SlateRecord_maple_tree_h_167; 4],
    slot: [*mut (); 3],
    gap: [u64; 3],
    min: u64,
    __slate_anon_5: __SlateRecord_maple_tree_h_178,
    end: u8,
    s_count: u8,
    d_count: u8,
    split: u8,
    data: u8,
    height: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_tree {
    __slate_anon_0: __SlateRecord_maple_tree_h_261,
    ma_flags: u32,
    ma_root: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_node {
    __slate_anon_0: __SlateRecord_maple_tree_h_324,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ma_state {
    tree: *mut maple_tree,
    index: u64,
    last: u64,
    node: *mut maple_enode,
    min: u64,
    max: u64,
    sheaf: *mut slab_sheaf,
    alloc: *mut maple_node,
    node_request: u64,
    status: u32,
    depth: u8,
    offset: u8,
    mas_flags: u8,
    end: u8,
    store_type: u32,
}


// When changing MA_STATE, remember to also change rust/kernel/maple_tree.rs
fn mas_init(mas: *mut ma_state, tree: *mut maple_tree, __v0: u64) {
unsafe { memset(mas as *mut (), 0i32, core::mem::size_of::<ma_state>() as u64) };
unsafe {
(*mas).tree = tree;
}
unsafe {
(*mas).last = __v0;
}
unsafe {
(*mas).index = __v0;
}
unsafe {
(*mas).max = !0;
}
unsafe {
(*mas).status = 1;
}
unsafe {
(*mas).node = core::ptr::null_mut::<maple_enode>();
}
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


fn mas_is_active(mas: *mut ma_state) -> bool {
((unsafe { (*mas).status }) as u32) == 0
}

fn mas_is_err(mas: *mut ma_state) -> bool {
((unsafe { (*mas).status }) as u32) == 7
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


#[inline(always)]
fn mas_reset(mas: *mut ma_state) {
unsafe {
(*mas).status = 1;
}
unsafe {
(*mas).node = core::ptr::null_mut::<maple_enode>();
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


// mas_for_each() - Iterate over a range of the maple tree.
// @__mas: Maple Tree operation state (maple_state)
// @__entry: Entry retrieved from the tree
// @__max: maximum index to retrieve from the tree
//
// When returned, mas->index and mas->last will hold the entire range for the
// entry.
//
// Note: may return the zero entry.
fn __mas_set_range(mas: *mut ma_state, start: u64, last: u64) {
// Ensure the range starts within the current slot
let mut __v0: i64;
let __ret_warn_on: i32 = (mas_is_active(mas) && ((unsafe { (*mas).index }) > start || (unsafe { (*mas).last }) < start)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("145: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 145b - ., 3; .popsection\n# {0}", const 145 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___mas_set_range_1, sym __slate_asm_string___mas_set_range_2, const 797 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("146: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 146b - ., 4; .popsection\n# {0}", const 146 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe {
(*mas).index = start;
}
unsafe {
(*mas).last = last;
}
}

fn mas_set_range(mas: *mut ma_state, start: u64, last: u64) {
mas_reset(mas);
__mas_set_range(mas, start, last);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


fn mas_set(mas: *mut ma_state, index: u64) {
mas_set_range(mas, index, index);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


fn mt_external_lock(mt: *const maple_tree) -> bool {
(unsafe { (*mt).ma_flags }) & 768 == 768
}

fn mt_init_flags(mt: *mut maple_tree, flags: u32) {
let mut __slate_compound_0: spinlock = unsafe { core::mem::zeroed() };
unsafe {
(*mt).ma_flags = flags;
}
if !mt_external_lock(mt as *const maple_tree) {
'__slate_break_1: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*mt).__slate_anon_0.ma_lock) });
unsafe {
*core::ptr::addr_of_mut!((*mt).__slate_anon_0.ma_lock) = *{
__slate_compound_0 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
if true {
break;
}
}
}
let _r_a_p__v: u64 = 0;
if true && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_147() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__19!((*mt).ma_root, _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word__5!() {
unsafe { __compiletime_assert_148() };
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
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_149() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__19!(*unsafe { core::ptr::addr_of_mut!((*mt).ma_root) }, RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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

#[repr(C)]
#[derive(Clone, Copy)]
struct rw_semaphore {
    count: __SlateRecord_types_h_197,
    owner: __SlateRecord_types_h_197,
    osq: optimistic_spin_queue,
    wait_lock: raw_spinlock,
    first_waiter: *mut rwsem_waiter,
}


fn rwsem_assert_held_write_nolockdep(sem: *const rw_semaphore) {
let mut __v0: i64;
let __ret_warn_on: i32 = ((atomic_long_read(unsafe { core::ptr::addr_of!((*sem).count) }) as u64) & 1u64 == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("156: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 156b - ., 3; .popsection\n# {0}", const 156 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_rwsem_assert_held_write_nolockdep_1, sym __slate_asm_string_rwsem_assert_held_write_nolockdep_2, const 87 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("157: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 157b - ., 4; .popsection\n# {0}", const 157 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}

fn rwsem_assert_held_write(sem: *const rw_semaphore) {
if IS_ENABLED!() != 0 {
lockdep_assert_held_write!(sem);
} else {
rwsem_assert_held_write_nolockdep(sem);
}
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
struct tty_struct {
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
struct tty_audit_buf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_work {
    work: work_struct,
    rcu: callback_head,
    wq: *mut workqueue_struct,
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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


fn percpu_counter_read_positive(fbc: *mut percpu_counter) -> i64 {
// Prevent reloads of fbc->count
let mut ret: i64 = 0;
let mut __v0: i64;
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_183() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*fbc).count) as *const i64) };
ret = __v0;
if ret >= 0 {
return ret;
}
0
}

// Tracks mm_cpumask
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
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct anon_vma_name {
    kref: kref,
    name: [u8; 0],
}


fn anon_vma_name(vma: *mut vm_area_struct) -> *mut anon_vma_name {
core::ptr::null_mut::<anon_vma_name>()
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


#[repr(C)]
#[derive(Clone, Copy)]
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits453U0,
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


#[inline(always)]
fn vma_flags_empty(flags: &__SlateRecord_mm_types_h_871) -> bool {
let bitmap: *const u64 = unsafe { (*(flags as *const __SlateRecord_mm_types_h_871)).__vma_flags.as_ptr() as *const u64 };
bitmap_empty(bitmap, 64u32)
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


#[repr(C)]
#[derive(Clone, Copy)]
struct request_queue {
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


#[inline(always)]
fn vma_flags_to_legacy(flags: __SlateRecord_mm_types_h_871) -> u64 {
flags.__vma_flags[0]
}

#[inline(always)]
fn vma_flags_overwrite_word(flags: &mut __SlateRecord_mm_types_h_871, value: u64) {
let bitmap: *mut u64 = unsafe { (*(flags as *mut __SlateRecord_mm_types_h_871)).__vma_flags.as_mut_ptr() as *mut u64 };
unsafe {
*bitmap.wrapping_offset(0 as isize) = value;
}
}

#[inline(always)]
fn legacy_to_vma_flags(flags: u64) -> __SlateRecord_mm_types_h_871 {
let mut __slate_compound_0: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut ret: __SlateRecord_mm_types_h_871 = unsafe { *{
__slate_compound_0 = {
let mut __t0: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t0
};
core::ptr::addr_of_mut!(__slate_compound_0)
} };
vma_flags_overwrite_word(unsafe { &mut (*core::ptr::addr_of_mut!(ret)) }, flags);
return ret;
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
struct vma_iterator {
    mas: ma_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
}


fn vma_iter_init(vmi: &mut vma_iterator, mm: *mut mm_struct, addr: u64) {
mas_init(unsafe { core::ptr::addr_of_mut!((*(vmi as *mut vma_iterator)).mas) }, unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_mt) }, addr);
}

// Clear the access bit
// Clear the dirty bit
// MMF_HAS_PINNED: Whether this mm has pinned any pages.  This can be either
// replaced in the future by mm.pinned_vm when it becomes stable, or grow into
// a counter on its own. We're aggresive on this bit for now: even if the
// pinned pages were unpinned later on, we'll still keep this bit set for the
// lifecycle of this mm, just for simplicity.
// FOLL_PIN has run, never cleared
#[repr(C)]
#[derive(Clone, Copy)]
struct zswap_lruvec_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct free_area {
    free_list: [list_head; 4],
    nr_free: u64,
}


// We do arithmetic on the LRU lists in various places in the code,
// so it is important to keep the active lists LRU_ACTIVE higher in
// the array than the corresponding inactive lists, and to keep
// the *_FILE lists LRU_FILE higher than the corresponding _ANON lists.
//
// This has to be kept in sync with the statistics in zone_stat_item
// above and the descriptions in vmstat_text in mm/vmstat.c
#[repr(C)]
#[derive(Clone, Copy)]
struct lru_cost {
    count: u64,
    last_rotated: u64,
    last_io: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fsnotify_mark_connector {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lruvec {
    lists: [list_head; 5],
    lru_lock: spinlock,
    cost: [lru_cost; 2],
    cost_lock: spinlock,
    nonresident_age: __SlateRecord_types_h_197,
    refaults: [u64; 2],
    flags: u64,
    zswap_lruvec_state: zswap_lruvec_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_kattr {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct per_cpu_pages {
    lock: spinlock,
    count: i32,
    high: i32,
    high_min: i32,
    high_max: i32,
    batch: i32,
    flags: u8,
    alloc_factor: u8,
    expire: u8,
    free_count: i16,
    lists: [list_head; 12],
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
struct per_cpu_zonestat {
    vm_stat_diff: [i8; 10],
    stat_threshold: i8,
    vm_numa_event: [u64; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct per_cpu_nodestat {
    stat_threshold: i8,
    vm_node_stat_diff: [i8; 66],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct zone {
    _watermark: [u64; 4],
    watermark_boost: u64,
    nr_reserved_highatomic: u64,
    nr_free_highatomic: u64,
    lowmem_reserve: [i64; 4],
    node: i32,
    zone_pgdat: *mut pglist_data,
    per_cpu_pageset: *mut per_cpu_pages,
    per_cpu_zonestats: *mut per_cpu_zonestat,
    pageset_high_min: i32,
    pageset_high_max: i32,
    pageset_batch: i32,
    zone_start_pfn: u64,
    managed_pages: __SlateRecord_types_h_197,
    spanned_pages: u64,
    present_pages: u64,
    name: *const u8,
    initialized: i32,
    _pad1_: cacheline_padding,
    free_area: [free_area; 11],
    flags: u64,
    lock: spinlock,
    trylock_free_pages: llist_head,
    _pad2_: cacheline_padding,
    percpu_drift_mark: u64,
    compact_cached_free_pfn: u64,
    compact_cached_migrate_pfn: [u64; 2],
    compact_init_migrate_pfn: u64,
    compact_init_free_pfn: u64,
    compact_considered: u32,
    compact_defer_shift: u32,
    compact_order_failed: i32,
    compact_blockskip_flush: bool,
    contiguous: bool,
    _pad3_: cacheline_padding,
    vm_stat: [__SlateRecord_types_h_197; 10],
    vm_numa_event: [__SlateRecord_types_h_197; 6],
    vmemmap_tails: [*mut page; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
}


// Maximum number of zones on a zonelist
#[repr(C)]
#[derive(Clone, Copy)]
struct zoneref {
    zone: *mut zone,
    zone_idx: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct zonelist {
    _zonerefs: [zoneref; 257],
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
struct pglist_data {
    node_zones: [zone; 4],
    node_zonelists: [zonelist; 2],
    nr_zones: i32,
    node_start_pfn: u64,
    node_present_pages: u64,
    node_spanned_pages: u64,
    node_id: i32,
    kswapd_wait: wait_queue_head,
    pfmemalloc_wait: wait_queue_head,
    reclaim_wait: [wait_queue_head; 4],
    nr_writeback_throttled: __SlateRecord_types_h_190,
    nr_reclaim_start: u64,
    kswapd: *mut task_struct,
    kswapd_order: i32,
    kswapd_highest_zoneidx: u32,
    kswapd_failures: __SlateRecord_types_h_190,
    kcompactd_max_order: i32,
    kcompactd_highest_zoneidx: u32,
    kcompactd_wait: wait_queue_head,
    kcompactd: *mut task_struct,
    proactive_compact_trigger: bool,
    totalreserve_pages: u64,
    min_unmapped_pages: u64,
    min_slab_pages: u64,
    _pad1_: cacheline_padding,
    __lruvec: lruvec,
    flags: u64,
    _pad2_: cacheline_padding,
    per_cpu_nodestats: *mut per_cpu_nodestat,
    vm_stat: [__SlateRecord_types_h_197; 66],
    memtier: *mut memory_tier,
}


// zone_idx() returns 0 for the ZONE_DMA zone, 1 for the ZONE_NORMAL zone, etc.
#[repr(C)]
#[derive(Clone, Copy)]
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
// XA_STATE() - Declare an XArray operation state.
// @name: Name of this operation state (usually xas).
// @array: Array to operate on.
// @index: Initial index of interest.
//
// Declare and initialise an xa_state on the stack.
// xas_for_each() - Iterate over a range of an XArray.
// @xas: XArray operation state.
// @entry: Entry retrieved from the array.
// @max: Maximum index to retrieve from array.
//
// The loop body will be executed for each entry present in the xarray
// between the current xas position and @max.  @entry will be set to
// the entry retrieved from the xarray.  It is safe to delete entries
// from the array in the loop body.  You should hold either the RCU lock
// or the xa_lock while iterating.  If you need to drop the lock, call
// xas_pause() first.
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
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct list_lru_node {
    lru: list_lru_one,
    nr_items: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru {
    node: *mut list_lru_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct audit_names {
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
fn __mmap_lock_trace_start_locking(mm: *mut mm_struct, write: bool) {
if static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mmap_lock_start_locking.key) }) != 0 {
unsafe { __mmap_lock_do_trace_start_locking(mm, write) };
}
}

fn __mmap_lock_trace_acquire_returned(mm: *mut mm_struct, write: bool, success: bool) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mmap_lock_acquire_returned.key) }) != 0 {
unsafe { __mmap_lock_do_trace_acquire_returned(mm, write, success) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
}


fn __mmap_lock_trace_released(mm: *mut mm_struct, write: bool) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mmap_lock_released.key) }) != 0 {
unsafe { __mmap_lock_do_trace_released(mm, write) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
}


fn mmap_assert_write_locked(mm: *const mm_struct) {
rwsem_assert_held_write(unsafe { core::ptr::addr_of!((*mm).__slate_anon_0.mmap_lock) });
}

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
fn mm_lock_seqcount_begin(mm: *mut mm_struct) {
do_raw_write_seqcount_begin(unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_lock_seq) });
}

fn mm_lock_seqcount_end(mm: *mut mm_struct) {
ASSERT_EXCLUSIVE_WRITER!((*mm).__slate_anon_0.mm_lock_seq);
do_raw_write_seqcount_end(unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_lock_seq) });
}

fn __vma_refcount_put_return(vma: *mut vm_area_struct) -> u32 {
let mut oldcnt: i32 = 0;
if __refcount_dec_and_test(unsafe { core::ptr::addr_of_mut!((*vma).vm_refcnt) }, core::ptr::addr_of_mut!(oldcnt)) {
return 0;
}
oldcnt.wrapping_sub(1) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


fn __vma_raw_mm_seqnum(vma: *mut vm_area_struct) -> u32 {
let mm: *const mm_struct = (unsafe { (*vma).vm_mm }) as *const mm_struct;
// We must hold an exclusive write lock for this access to be valid.
mmap_assert_write_locked((unsafe { (*vma).vm_mm }) as *const mm_struct);
unsafe { (*mm).__slate_anon_0.mm_lock_seq.sequence }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


fn __is_vma_write_locked(vma: *mut vm_area_struct) -> bool {
// current task is holding mmap_write_lock, both vma->vm_lock_seq and
// mm->mm_lock_seq can't be concurrently modified.
(unsafe { (*vma).vm_lock_seq }) == __vma_raw_mm_seqnum(vma)
}

fn vma_start_write(vma: *mut vm_area_struct) {
if __is_vma_write_locked(vma) {
return;
}
unsafe { __vma_start_write(vma, 2i32) };
}

fn vma_assert_write_locked(vma: *mut vm_area_struct) {
VM_WARN_ON_ONCE_VMA!();
}

fn vma_is_attached(vma: *mut vm_area_struct) -> bool {
refcount_read((unsafe { core::ptr::addr_of_mut!((*vma).vm_refcnt) }) as *const refcount_struct) != 0
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


fn vma_assert_attached(vma: *mut vm_area_struct) {
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!vma_is_attached(vma))) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("278: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 278b - ., 3; .popsection\n# {0}", const 278 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_vma_assert_attached_1, sym __slate_asm_string_vma_assert_attached_2, const 435 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("279: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 279b - ., 4; .popsection\n# {0}", const 279 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}

fn vma_assert_detached(vma: *mut vm_area_struct) {
let mut __v0: i64;
let __ret_warn_on: i32 = !(!vma_is_attached(vma)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("280: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 280b - ., 3; .popsection\n# {0}", const 280 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_vma_assert_detached_1, sym __slate_asm_string_vma_assert_detached_2, const 440 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("281: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 281b - ., 4; .popsection\n# {0}", const 281 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}

fn vma_mark_attached(vma: *mut vm_area_struct) {
vma_assert_write_locked(vma);
vma_assert_detached(vma);
refcount_set_release(unsafe { core::ptr::addr_of_mut!((*vma).vm_refcnt) }, 1i32);
}

fn vma_mark_detached(vma: *mut vm_area_struct) {
vma_assert_write_locked(vma);
vma_assert_attached(vma);
// The VMA still being attached (refcnt > 0) - is unlikely, because the
// vma has been already write-locked and readers can increment vm_refcnt
// only temporarily before they check vm_lock_seq, realize the vma is
// locked and drop back the vm_refcnt. That is a narrow window for
// observing a raised vm_refcnt.
//
// See the comment describing the vm_area_struct->vm_refcnt field for
// details of possible refcnt values.
if likely!(__vma_refcount_put_return(vma) == 0) != 0 {
return;
}
unsafe { __vma_exclude_readers_for_detach(vma) };
}

fn vma_assert_can_modify(vma: *mut vm_area_struct) {
if vma_is_attached(vma) {
vma_assert_write_locked(vma);
}
}

fn mmap_write_lock_killable(mm: *mut mm_struct) -> i32 {
let mut ret: i32 = 0;
__mmap_lock_trace_start_locking(mm, true);
ret = unsafe { down_write_killable(core::ptr::addr_of_mut!((*mm).__slate_anon_0.mmap_lock)) };
if ret == 0 {
mm_lock_seqcount_begin(mm);
}
__mmap_lock_trace_acquire_returned(mm, true, ret == 0);
ret
}

fn vma_end_write_all(mm: *mut mm_struct) {
mmap_assert_write_locked(mm as *const mm_struct);
mm_lock_seqcount_end(mm);
}

fn mmap_write_unlock(mm: *mut mm_struct) {
__mmap_lock_trace_released(mm, true);
vma_end_write_all(mm);
unsafe { up_write(core::ptr::addr_of_mut!((*mm).__slate_anon_0.mmap_lock)) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
}


fn mmap_write_downgrade(mm: *mut mm_struct) {
__mmap_lock_trace_acquire_returned(mm, false, true);
vma_end_write_all(mm);
unsafe { downgrade_write(core::ptr::addr_of_mut!((*mm).__slate_anon_0.mmap_lock)) };
}

fn mmap_read_unlock(mm: *mut mm_struct) {
__mmap_lock_trace_released(mm, false);
unsafe { up_read(core::ptr::addr_of_mut!((*mm).__slate_anon_0.mmap_lock)) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
}


fn cc_get_mask() -> u64 {
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_handle {
}


fn pgprot_modify(oldprot: pgprot, newprot: pgprot) -> pgprot {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
let preservebits: u64 = pgprot_val!(oldprot) & (((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64 | _AT!(u64, 1i32) << 4i32 | _AT!(u64, 1i32) << 3i32 | _AT!(u64, 1i32) << 9i32 | _AT!(u64, 1i32) << 5i32 | (_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 58i32) | _AT!(u64, 0i32) | _AT!(u64, cc_get_mask()) | _AT!(u64, 0i32) | _AT!(u64, 1i32) << 7i32);
let addbits: u64 = pgprot_val!(newprot) & !(((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64 | _AT!(u64, 1i32) << 4i32 | _AT!(u64, 1i32) << 3i32 | _AT!(u64, 1i32) << 9i32 | _AT!(u64, 1i32) << 5i32 | (_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 58i32) | _AT!(u64, 0i32) | _AT!(u64, cc_get_mask()) | _AT!(u64, 0i32) | _AT!(u64, 1i32) << 7i32);
return unsafe { *{
__slate_compound_0 = pgprot { pgprot: preservebits | addbits };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
// All top-level MITIGATION_PAGE_TABLE_ISOLATION page tables are order-1 pages
// (8k-aligned and 8k in size).  The kernel one is at the beginning 4k and
// the user one is in the last 4k.  To switch between them, you
// just need to flip the 12th bit in their addresses.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_capability_h_24 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table {
    procname: *const u8,
    data: *mut (),
    maxlen: i32,
    mode: u16,
    proc_handler: Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>,
    poll: *mut ctl_table_poll,
    extra1: *mut (),
    extra2: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_node {
    node: rb_node,
    header: *mut ctl_table_header,
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
struct ctl_table_header {
    __slate_anon_0: __SlateRecord_sysctl_h_254,
    unregistering: *mut completion,
    ctl_table_arg: *const ctl_table,
    root: *mut ctl_table_root,
    set: *mut ctl_table_set,
    parent: *mut ctl_dir,
    node: *mut ctl_node,
    inodes: hlist_head,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_dir {
    header: ctl_table_header,
    root: rb_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_set {
    is_seen: Option<unsafe extern "C" fn(*mut ctl_table_set) -> i32>,
    dir: ctl_dir,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_root {
    default_set: ctl_table_set,
    lookup: Option<unsafe extern "C" fn(*mut ctl_table_root) -> *mut ctl_table_set>,
    set_ownership: Option<unsafe extern "C" fn(*mut ctl_table_header, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
    permissions: Option<unsafe extern "C" fn(*mut ctl_table_header, *const ctl_table) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array {
    root: *mut assoc_array_ptr,
    nr_leaves_on_tree: u64,
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
struct module_param_attrs {
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
struct rhash_head {
    next: *mut rhash_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhlist_head {
    rhead: rhash_head,
    next: *mut rhlist_head,
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
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
union __SlateRecord_memremap_h_143 {
    range: range,
    __slate_anon_1: __SlateRecord_memremap_h_145,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_memremap_h_145 {
    __empty_ranges: __SlateRecord_memremap_h_145_2,
    ranges: [range; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_memremap_h_145_2 {
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
struct __SlateRecord_mm_h_754 {
    vma: *mut vm_area_struct,
    gfp_mask: u32,
    pgoff: u64,
    address: u64,
    real_address: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_h_768 {
    orig_pte: __SlateRecord_pgtable_64_types_h_21,
    orig_pmd: __SlateRecord_pgtable_64_types_h_22,
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
    __slate_bits_0: __slate_bits::__SlateBits300U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct disk_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_holder_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct partition_meta_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mempolicy_h_54 {
    cpuset_mems_allowed: __SlateRecord_nodemask_types_h_17,
    user_nodemask: __SlateRecord_nodemask_types_h_17,
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
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_359() };
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
struct ipc_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_ns_operations {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct time_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uts_namespace {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_ns_common_types_h_112 {
    __ns_ref: refcount_struct,
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
union __SlateRecord_user_namespace_h_26 {
    __slate_anon_0: __SlateRecord_user_namespace_h_27,
    __slate_anon_1: __SlateRecord_user_namespace_h_31,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_user_namespace_h_27 {
    extent: [uid_gid_extent; 5],
    nr_extents: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_user_namespace_h_31 {
    forward: *mut uid_gid_extent,
    reverse: *mut uid_gid_extent,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct binfmt_misc {
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
struct cgroup_taskset {
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
union __SlateRecord_cgroup_defs_h_644 {
    __slate_anon_0: __SlateRecord_cgroup_defs_h_645,
    __slate_anon_1: __SlateRecord_cgroup_defs_h_646,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_645 {
    __empty_ancestors: __SlateRecord_cgroup_defs_h_645_2,
    ancestors: [*mut cgroup; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_645_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_646 {
    _root_ancestor: *mut cgroup,
    __slate_anon_1: __SlateRecord_cgroup_defs_h_648,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_648 {
    __empty__low_ancestors: __SlateRecord_cgroup_defs_h_648_2,
    _low_ancestors: [*mut cgroup; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_648_2 {
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
struct swap_cluster_info {
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
struct swap_ops {
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
union __SlateRecord_perf_event_h_410 {
    sample_period: u64,
    sample_freq: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_471 {
    wakeup_events: u32,
    wakeup_watermark: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_477 {
    bp_addr: u64,
    kprobe_func: u64,
    uprobe_path: u64,
    config1: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_483 {
    bp_len: u64,
    kprobe_addr: u64,
    probe_offset: u64,
    config2: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_530 {
    aux_action: u32,
    __slate_anon_1: __SlateRecord_perf_event_h_532,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_532 {
    __slate_bits_0: __slate_bits::__SlateBits870U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_1322 {
    __slate_bits_0: __slate_bits::__SlateBits872U0,
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
struct __SlateRecord_perf_event_h_1512 {
    var1_dw: u32,
    var2_w: u16,
    var3_w: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_pin {
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
struct ftrace_ops {
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
struct __SlateRecord_bpf_h_1528 {
    map_type: u32,
    key_size: u32,
    value_size: u32,
    max_entries: u32,
    map_flags: u32,
    inner_map_fd: u32,
    numa_node: u32,
    map_name: [u8; 16],
    map_ifindex: u32,
    btf_fd: u32,
    btf_key_type_id: u32,
    btf_value_type_id: u32,
    btf_vmlinux_value_type_id: u32,
    map_extra: u64,
    value_type_btf_obj_fd: i32,
    map_token_fd: i32,
    excl_prog_hash: u64,
    excl_prog_hash_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1581 {
    map_fd: u32,
    key: u64,
    __slate_anon_2: __SlateRecord_bpf_h_1584,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1584 {
    value: u64,
    next_key: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1591 {
    in_batch: u64,
    out_batch: u64,
    keys: u64,
    values: u64,
    count: u32,
    map_fd: u32,
    elem_flags: u64,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1608 {
    prog_type: u32,
    insn_cnt: u32,
    insns: u64,
    license: u64,
    log_level: u32,
    log_size: u32,
    log_buf: u64,
    kern_version: u32,
    prog_flags: u32,
    prog_name: [u8; 16],
    prog_ifindex: u32,
    expected_attach_type: u32,
    prog_btf_fd: u32,
    func_info_rec_size: u32,
    func_info: u64,
    func_info_cnt: u32,
    line_info_rec_size: u32,
    line_info: u64,
    line_info_cnt: u32,
    attach_btf_id: u32,
    __slate_anon_20: __SlateRecord_bpf_h_1633,
    core_relo_cnt: u32,
    fd_array: u64,
    core_relos: u64,
    core_relo_rec_size: u32,
    log_true_size: u32,
    prog_token_fd: i32,
    fd_array_cnt: u32,
    signature: u64,
    signature_size: u32,
    keyring_id: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1633 {
    attach_prog_fd: u32,
    attach_btf_obj_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1674 {
    pathname: u64,
    bpf_fd: u32,
    file_flags: u32,
    path_fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1687 {
    __slate_anon_0: __SlateRecord_bpf_h_1688,
    attach_bpf_fd: u32,
    attach_type: u32,
    attach_flags: u32,
    replace_bpf_fd: u32,
    __slate_anon_5: __SlateRecord_bpf_h_1696,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1688 {
    target_fd: u32,
    target_ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1696 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1703 {
    prog_fd: u32,
    retval: u32,
    data_size_in: u32,
    data_size_out: u32,
    data_in: u64,
    data_out: u64,
    repeat: u32,
    duration: u32,
    ctx_size_in: u32,
    ctx_size_out: u32,
    ctx_in: u64,
    ctx_out: u64,
    flags: u32,
    cpu: u32,
    batch_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1727 {
    __slate_anon_0: __SlateRecord_bpf_h_1728,
    next_id: u32,
    open_flags: u32,
    fd_by_id_token_fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1728 {
    start_id: u32,
    prog_id: u32,
    map_id: u32,
    btf_id: u32,
    link_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1740 {
    bpf_fd: u32,
    info_len: u32,
    info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1746 {
    __slate_anon_0: __SlateRecord_bpf_h_1747,
    attach_type: u32,
    query_flags: u32,
    attach_flags: u32,
    prog_ids: u64,
    __slate_anon_5: __SlateRecord_bpf_h_1755,
    __slate_bits_0: __slate_bits::__SlateBits901U0,
    prog_attach_flags: u64,
    link_ids: u64,
    link_attach_flags: u64,
    revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1747 {
    target_fd: u32,
    target_ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1755 {
    prog_cnt: u32,
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1769 {
    name: u64,
    prog_fd: u32,
    __slate_bits_0: __slate_bits::__SlateBits904U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1776 {
    btf: u64,
    btf_log_buf: u64,
    btf_size: u32,
    btf_log_size: u32,
    btf_log_level: u32,
    btf_log_true_size: u32,
    btf_flags: u32,
    btf_token_fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1794 {
    pid: u32,
    fd: u32,
    flags: u32,
    buf_len: u32,
    buf: u64,
    prog_id: u32,
    fd_type: u32,
    probe_offset: u64,
    probe_addr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1810 {
    __slate_anon_0: __SlateRecord_bpf_h_1811,
    __slate_anon_1: __SlateRecord_bpf_h_1815,
    attach_type: u32,
    flags: u32,
    __slate_anon_4: __SlateRecord_bpf_h_1821,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1811 {
    prog_fd: u32,
    map_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1815 {
    target_fd: u32,
    target_ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1821 {
    target_btf_id: u32,
    __slate_anon_1: __SlateRecord_bpf_h_1823,
    perf_event: __SlateRecord_bpf_h_1827,
    kprobe_multi: __SlateRecord_bpf_h_1834,
    tracing: __SlateRecord_bpf_h_1841,
    netfilter: __SlateRecord_bpf_h_1850,
    tcx: __SlateRecord_bpf_h_1856,
    uprobe_multi: __SlateRecord_bpf_h_1863,
    netkit: __SlateRecord_bpf_h_1873,
    cgroup: __SlateRecord_bpf_h_1880,
    tracing_multi: __SlateRecord_bpf_h_1887,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1823 {
    iter_info: u64,
    iter_info_len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1827 {
    bpf_cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1834 {
    flags: u32,
    cnt: u32,
    syms: u64,
    addrs: u64,
    cookies: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1841 {
    target_btf_id: u32,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1850 {
    pf: u32,
    hooknum: u32,
    priority: i32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1856 {
    __slate_anon_0: __SlateRecord_bpf_h_1857,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1857 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1863 {
    path: u64,
    offsets: u64,
    ref_ctr_offsets: u64,
    cookies: u64,
    cnt: u32,
    flags: u32,
    pid: u32,
    path_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1873 {
    __slate_anon_0: __SlateRecord_bpf_h_1874,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1874 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1880 {
    __slate_anon_0: __SlateRecord_bpf_h_1881,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1881 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1887 {
    ids: u64,
    cookies: u64,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1895 {
    link_fd: u32,
    __slate_anon_1: __SlateRecord_bpf_h_1897,
    flags: u32,
    __slate_anon_3: __SlateRecord_bpf_h_1904,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1897 {
    new_prog_fd: u32,
    new_map_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1904 {
    old_prog_fd: u32,
    old_map_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1916 {
    link_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1920 {
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1924 {
    link_fd: u32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1929 {
    prog_fd: u32,
    map_fd: u32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1935 {
    flags: u32,
    bpffs_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1940 {
    stream_buf: u64,
    stream_buf_len: u32,
    stream_id: u32,
    prog_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1947 {
    map_fd: u32,
    prog_fd: u32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_6842 {
    raw_tracepoint: __SlateRecord_bpf_h_6843,
    tracing: __SlateRecord_bpf_h_6849,
    cgroup: __SlateRecord_bpf_h_6856,
    iter: __SlateRecord_bpf_h_6860,
    netns: __SlateRecord_bpf_h_6884,
    xdp: __SlateRecord_bpf_h_6888,
    struct_ops: __SlateRecord_bpf_h_6891,
    netfilter: __SlateRecord_bpf_h_6894,
    kprobe_multi: __SlateRecord_bpf_h_6900,
    uprobe_multi: __SlateRecord_bpf_h_6907,
    tracing_multi: __SlateRecord_bpf_h_6917,
    perf_event: __SlateRecord_bpf_h_6926,
    tcx: __SlateRecord_bpf_h_6959,
    netkit: __SlateRecord_bpf_h_6963,
    sockmap: __SlateRecord_bpf_h_6967,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6843 {
    tp_name: u64,
    tp_name_len: u32,
    __slate_bits_0: __slate_bits::__SlateBits937U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6849 {
    attach_type: u32,
    target_obj_id: u32,
    target_btf_id: u32,
    __slate_bits_0: __slate_bits::__SlateBits938U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6856 {
    cgroup_id: u64,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6860 {
    target_name: u64,
    target_name_len: u32,
    __slate_anon_2: __SlateRecord_bpf_h_6868,
    __slate_anon_3: __SlateRecord_bpf_h_6873,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_6868 {
    map: __SlateRecord_bpf_h_6869,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6869 {
    map_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_6873 {
    cgroup: __SlateRecord_bpf_h_6874,
    task: __SlateRecord_bpf_h_6878,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6874 {
    cgroup_id: u64,
    order: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6878 {
    tid: u32,
    pid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6884 {
    netns_ino: u32,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6888 {
    ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6891 {
    map_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6894 {
    pf: u32,
    hooknum: u32,
    priority: i32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6900 {
    addrs: u64,
    count: u32,
    flags: u32,
    missed: u64,
    cookies: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6907 {
    path: u64,
    offsets: u64,
    ref_ctr_offsets: u64,
    cookies: u64,
    path_size: u32,
    count: u32,
    flags: u32,
    pid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6917 {
    attach_type: u32,
    count: u32,
    btf_obj_id: u32,
    __slate_bits_0: __slate_bits::__SlateBits952U0,
    ids: u64,
    addrs: u64,
    cookies: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6926 {
    r#type: u32,
    __slate_bits_0: __slate_bits::__SlateBits953U0,
    __slate_anon_2: __SlateRecord_bpf_h_6929,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_6929 {
    uprobe: __SlateRecord_bpf_h_6930,
    kprobe: __SlateRecord_bpf_h_6937,
    tracepoint: __SlateRecord_bpf_h_6945,
    event: __SlateRecord_bpf_h_6951,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6930 {
    file_name: u64,
    name_len: u32,
    offset: u32,
    cookie: u64,
    ref_ctr_offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6937 {
    func_name: u64,
    name_len: u32,
    offset: u32,
    addr: u64,
    missed: u64,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6945 {
    tp_name: u64,
    name_len: u32,
    __slate_bits_0: __slate_bits::__SlateBits957U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6951 {
    config: u64,
    r#type: u32,
    __slate_bits_0: __slate_bits::__SlateBits958U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6959 {
    ifindex: u32,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6963 {
    ifindex: u32,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6967 {
    map_id: u32,
    attach_type: u32,
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
if !(__native_word__6!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_384() };
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
union __SlateRecord_btf_h_64 {
    size: u32,
    r#type: u32,
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
struct btf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits634U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_verifier_log {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_verifier_env {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_file_ref_h_52 {
    refcnt: __SlateRecord_types_h_197,
}


#[inline(always)]
fn file_ref_inc(r#ref: *mut __SlateRecord_file_ref_h_52) {
let prior: i64 = atomic_long_fetch_inc_relaxed(unsafe { core::ptr::addr_of_mut!((*r#ref).refcnt) });
let mut __v0: i32;
let __ret_warn_on: i32 = (prior < 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"file_ref_inc() on a released file reference".as_ptr() as *const u8) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_file_ref_inc_3, sym __slate_asm_string_file_ref_inc_4, const 121 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = __ret_warn_on;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_arena {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_iter_aux_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_local_storage_map {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_state {
}


#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_392: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

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


// XArray tags, for tagging dirty and writeback pages in the pagecache.
fn i_mmap_lock_write(mapping: *mut address_space) {
unsafe { down_write(core::ptr::addr_of_mut!((*mapping).i_mmap_rwsem)) };
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_256 {
    kptr: btf_field_kptr,
    graph_root: btf_field_graph_root,
}


fn i_mmap_unlock_write(mapping: *mut address_space) {
unsafe { up_write(core::ptr::addr_of_mut!((*mapping).i_mmap_rwsem)) };
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_329 {
    work: work_struct,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_offload_dev {
}


fn mapping_map_writable(mapping: *mut address_space) -> i32 {
if atomic_inc_unless_negative(unsafe { core::ptr::addr_of_mut!((*mapping).i_mmap_writable) }) { 0 } else { 1i32.wrapping_neg() }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_996 {
    __slate_anon_0: __SlateRecord_bpf_h_997,
    arg_type: [u32; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_997 {
    arg1_type: u32,
    arg2_type: u32,
    arg3_type: u32,
    arg4_type: u32,
    arg5_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1006 {
    __slate_anon_0: __SlateRecord_bpf_h_1007,
    arg_btf_id: [*mut u32; 12],
    __slate_anon_2: __SlateRecord_bpf_h_1015,
    arg_size: [u64; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1007 {
    arg1_btf_id: *mut u32,
    arg2_btf_id: *mut u32,
    arg3_btf_id: *mut u32,
    arg4_btf_id: *mut u32,
    arg5_btf_id: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1015 {
    arg1_size: u64,
    arg2_size: u64,
    arg3_size: u64,
    arg4_size: u64,
    arg5_size: u64,
}


fn mapping_unmap_writable(mapping: *mut address_space) {
atomic_dec(unsafe { core::ptr::addr_of_mut!((*mapping).i_mmap_writable) });
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1109 {
    ctx_field_size: i32,
    __slate_anon_1: __SlateRecord_bpf_h_1111,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1111 {
    btf: *mut btf,
    btf_id: u32,
    ref_id: u32,
}


fn mapping_allow_writable(mapping: *mut address_space) {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*mapping).i_mmap_writable) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_reg_state {
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


fn get_file(f: *mut file) -> *mut file {
file_ref_inc(unsafe { core::ptr::addr_of_mut!((*f).f_ref) });
f
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fasync_struct {
    fa_lock: rwlock,
    magic: i32,
    fa_fd: i32,
    fa_next: *mut fasync_struct,
    fa_file: *mut file,
    fa_rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1393 {
    rcu: callback_head,
    work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1423 {
    model: btf_func_model,
    addr: *mut (),
    ftrace_managed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1442 {
    old_image: *mut bpf_tramp_image,
    old_flags: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1705 {
    tail_call: __SlateRecord_bpf_h_1706,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1706 {
    map: *mut bpf_map,
    key: u32,
}


// Contains huge pages
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


fn vfs_mmap(file: *mut file, vma: *mut vm_area_struct) -> i32 {
if (unsafe { (*(*file).f_op).mmap_prepare }) != None {
return unsafe { compat_vma_mmap(file, vma) };
}
unsafe { unsafe { (*(*file).f_op).mmap }.unwrap()(file, vma) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_kfunc_desc_tab {
}


fn vfs_mmap_prepare(file: *mut file, desc: *mut vm_area_desc) -> i32 {
unsafe { unsafe { (*(*file).f_op).mmap_prepare }.unwrap()(desc) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1821_2 {
    keyring_serial: i32,
    keyring_type: u8,
    verdict: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_kfunc_btf_tab {
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
union __SlateRecord_bpf_h_1892 {
    work: work_struct,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1926 {
    digest: [u8; 32],
    tag: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock_fprog_kern {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1937 {
    __slate_anon_0: __SlateRecord_bpf_h_1938,
    __slate_anon_1: __SlateRecord_bpf_h_1939,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1938 {
    __empty_insns: __SlateRecord_bpf_h_1938_2,
    insns: [sock_filter; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1938_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1939 {
    __empty_insnsi: __SlateRecord_bpf_h_1939_2,
    insnsi: [bpf_insn; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1939_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __filename_head {
    name: *const u8,
    refcnt: i32,
    aname: *mut audit_names,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct filename {
    __slate_anon_0: __filename_head,
    iname: [u8; 168],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1964 {
    rcu: callback_head,
    work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct offset_ctx {
    mt: maple_tree,
    next_offset: u64,
}


fn file_is_dax(file: *const file) -> bool {
!file.is_null() && IS_DAX!(unsafe { (*(*file).f_mapping).host }) != 0
}

fn vma_is_dax(vma: *const vm_area_struct) -> bool {
file_is_dax((unsafe { (*vma).vm_file }) as *const file)
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_76 {
    next: *mut perf_raw_frag,
    pad: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct idr {
    idr_rt: xarray,
    idr_base: u32,
    idr_next: u32,
}


// The IDR API does not expose the tagging functionality of the radix tree
// to users.  Use tag 0 to track whether a node has free space below it.
// Set the IDR flag and the IDR_FREE tag
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
union __SlateRecord_perf_event_h_149 {
    __slate_anon_0: __SlateRecord_perf_event_h_150,
    __slate_anon_1: __SlateRecord_perf_event_h_165,
    __slate_anon_2: __SlateRecord_perf_event_h_174,
    __slate_anon_3: __SlateRecord_perf_event_h_177,
    __slate_anon_4: __SlateRecord_perf_event_h_181,
    __slate_anon_5: __SlateRecord_perf_event_h_186,
    __slate_anon_6: __SlateRecord_perf_event_h_196,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_150 {
    config: u64,
    config1: u64,
    last_tag: u64,
    dyn_constraint: u64,
    config_base: u64,
    event_base: u64,
    event_base_rdpmc: i32,
    idx: i32,
    last_cpu: i32,
    flags: i32,
    extra_reg: hw_perf_event_extra,
    branch_reg: hw_perf_event_extra,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_165 {
    aux_config: u64,
    aux_paused: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_174 {
    hrtimer: hrtimer,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_177 {
    tp_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_181 {
    pwr_acc: u64,
    ptsc: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_186 {
    info: arch_hw_breakpoint,
    bp_list: rhlist_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_196 {
    iommu_bank: u8,
    iommu_cntr: u8,
    padding: u16,
    conf: u64,
    conf1: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_244 {
    __slate_anon_0: __SlateRecord_perf_event_h_245,
    __slate_anon_1: __SlateRecord_perf_event_h_259,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_245 {
    last_period: u64,
    period_left: __SlateRecord_local64_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_259 {
    saved_metric: u64,
    saved_slots: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits695U0,
    vm_ops: *const vm_operations_struct,
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
struct attribute {
    name: *const u8,
    mode: u16,
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
struct perf_buffer {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct event_filter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysfs_ops {
    show: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *const u8, u64) -> i64>,
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
    __slate_bits_0: __slate_bits::__SlateBits642U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_1156 {
    flags: u64,
    aux_flags: u64,
    __slate_anon_2: __SlateRecord_perf_event_h_1159,
}


#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_1159 {
    __slate_bits_0: __slate_bits::__SlateBits1081U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_1163 {
    addr: *mut (),
    head: u64,
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
struct __SlateRecord_perf_event_h_1315 {
    pid: u32,
    tid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_1321 {
    cpu: u32,
    reserved: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_context_h_66 {
    string: *mut u8,
    blob: *mut (),
    name: *mut filename,
    file: *mut file,
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
struct pm_message {
    event: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_parser_h_56 {
    boolean: bool,
    int_32: i32,
    uint_32: u32,
    uint_64: u64,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
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
    __slate_bits_0: __slate_bits::__SlateBits720U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits720U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits720U2,
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
    __slate_bits_3: __slate_bits::__SlateBits720U3,
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
union __SlateRecord_vma_h_109 {
    vm_flags: u64,
    vma_flags: __SlateRecord_mm_types_h_871,
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
struct __SlateRecord_vma_h_302 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iovec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_vma_c_23 {
    vm_flags: u64,
    vma_flags: __SlateRecord_mm_types_h_871,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_vma_c_1897 {
}


mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1046U0 {
        #[bits(8)]
        pub r#type: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits634U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits297U0 {
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
    pub struct __SlateBits72U0 {
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
    pub struct __SlateBits72U1 {
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
    pub struct __SlateBits695U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits846U0 {
        #[bits(1)]
        pub early_init: bool,
        #[bits(1)]
        pub implicit_on_dfl: bool,
        #[bits(1)]
        pub threaded: bool,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits300U0 {
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
    pub struct __SlateBits72U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits130U0 {
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
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits720U0 {
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
    pub struct __SlateBits717U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits720U1 {
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
    pub struct __SlateBits720U2 {
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
    pub struct __SlateBits720U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1067U0 {
        #[bits(1)]
        pub embedded: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1081U0 {
        #[bits(1)]
        pub skip_read: u64,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits864U0 {
        #[bits(1)]
        pub disabled: u64,
        #[bits(1)]
        pub inherit: u64,
        #[bits(1)]
        pub pinned: u64,
        #[bits(1)]
        pub exclusive: u64,
        #[bits(1)]
        pub exclude_user: u64,
        #[bits(1)]
        pub exclude_kernel: u64,
        #[bits(1)]
        pub exclude_hv: u64,
        #[bits(1)]
        pub exclude_idle: u64,
        #[bits(1)]
        pub mmap: u64,
        #[bits(1)]
        pub comm: u64,
        #[bits(1)]
        pub freq: u64,
        #[bits(1)]
        pub inherit_stat: u64,
        #[bits(1)]
        pub enable_on_exec: u64,
        #[bits(1)]
        pub task: u64,
        #[bits(1)]
        pub watermark: u64,
        #[bits(2)]
        pub precise_ip: u64,
        #[bits(1)]
        pub mmap_data: u64,
        #[bits(1)]
        pub sample_id_all: u64,
        #[bits(1)]
        pub exclude_host: u64,
        #[bits(1)]
        pub exclude_guest: u64,
        #[bits(1)]
        pub exclude_callchain_kernel: u64,
        #[bits(1)]
        pub exclude_callchain_user: u64,
        #[bits(1)]
        pub mmap2: u64,
        #[bits(1)]
        pub comm_exec: u64,
        #[bits(1)]
        pub use_clockid: u64,
        #[bits(1)]
        pub context_switch: u64,
        #[bits(1)]
        pub write_backward: u64,
        #[bits(1)]
        pub namespaces: u64,
        #[bits(1)]
        pub ksymbol: u64,
        #[bits(1)]
        pub bpf_event: u64,
        #[bits(1)]
        pub aux_output: u64,
        #[bits(1)]
        pub cgroup: u64,
        #[bits(1)]
        pub text_poke: u64,
        #[bits(1)]
        pub build_id: u64,
        #[bits(1)]
        pub inherit_thread: u64,
        #[bits(1)]
        pub remove_on_exec: u64,
        #[bits(1)]
        pub sigtrap: u64,
        #[bits(1)]
        pub defer_callchain: u64,
        #[bits(1)]
        pub defer_output: u64,
        #[bits(24)]
        pub __reserved_1: u64,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits870U0 {
        #[bits(1)]
        pub aux_start_paused: u32,
        #[bits(1)]
        pub aux_pause: u32,
        #[bits(1)]
        pub aux_resume: u32,
        #[bits(29)]
        pub __reserved_3: u32,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits873U0 {
        #[bits(1)]
        pub mispred: u64,
        #[bits(1)]
        pub predicted: u64,
        #[bits(1)]
        pub in_tx: u64,
        #[bits(1)]
        pub abort: u64,
        #[bits(16)]
        pub cycles: u64,
        #[bits(4)]
        pub r#type: u64,
        #[bits(2)]
        pub spec: u64,
        #[bits(4)]
        pub new_type: u64,
        #[bits(3)]
        pub r#priv: u64,
        #[bits(31)]
        pub reserved: u64,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits872U0 {
        #[bits(5)]
        pub mem_op: u64,
        #[bits(14)]
        pub mem_lvl: u64,
        #[bits(5)]
        pub mem_snoop: u64,
        #[bits(2)]
        pub mem_lock: u64,
        #[bits(7)]
        pub mem_dtlb: u64,
        #[bits(4)]
        pub mem_lvl_num: u64,
        #[bits(1)]
        pub mem_remote: u64,
        #[bits(2)]
        pub mem_snoopx: u64,
        #[bits(3)]
        pub mem_blk: u64,
        #[bits(3)]
        pub mem_hops: u64,
        #[bits(5)]
        pub mem_region: u64,
        #[bits(13)]
        pub mem_rsvd: u64,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits104U0 {
        #[bits(16)]
        pub cs: u64,
        #[bits(2)]
        pub sl: u64,
        #[bits(1)]
        pub wfe: u64,
        #[bits(45, access = na)]
        pub __slate_pad_3: u64,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits105U0 {
        #[bits(16)]
        pub ss: u64,
        #[bits(1)]
        pub sti: u64,
        #[bits(1)]
        pub swevent: u64,
        #[bits(1)]
        pub nmi: u64,
        #[bits(13, access = na)]
        pub __slate_pad_4: u16,
        #[bits(8)]
        pub vector: u64,
        #[bits(8, access = na)]
        pub __slate_pad_6: u8,
        #[bits(4)]
        pub r#type: u64,
        #[bits(4, access = na)]
        pub __slate_pad_8: u8,
        #[bits(1)]
        pub enclave: u64,
        #[bits(1)]
        pub l: u64,
        #[bits(1)]
        pub nested: u64,
        #[bits(1, access = na)]
        pub __slate_pad_12: u8,
        #[bits(4)]
        pub insnlen: u64,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits969U0 {
        #[bits(1)]
        pub jited: u32,
        #[bits(1)]
        pub jit_requested: u32,
        #[bits(1)]
        pub jit_required: u32,
        #[bits(1)]
        pub gpl_compatible: u32,
        #[bits(1)]
        pub cb_access: u32,
        #[bits(1)]
        pub dst_needed: u32,
        #[bits(1)]
        pub blinding_requested: u32,
        #[bits(1)]
        pub blinded: u32,
        #[bits(1)]
        pub is_func: u32,
        #[bits(1)]
        pub kprobe_override: u32,
        #[bits(1)]
        pub has_callchain_buf: u32,
        #[bits(1)]
        pub enforce_expected_attach_type: u32,
        #[bits(1)]
        pub call_get_stack: u32,
        #[bits(1)]
        pub call_get_func_ip: u32,
        #[bits(1)]
        pub call_session_cookie: u32,
        #[bits(1)]
        pub tstamp_type_access: u32,
        #[bits(1)]
        pub sleepable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_17: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits881U0 {
        #[bits(4)]
        pub dst_reg: u8,
        #[bits(4)]
        pub src_reg: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits901U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits904U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits937U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits938U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits952U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits953U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits957U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits958U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits138U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits453U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits627U0 {
        #[bits(8)]
        pub purpose: u32,
        #[bits(8)]
        pub phase: u32,
        #[bits(1)]
        pub need_free: bool,
        #[bits(1)]
        pub global: bool,
        #[bits(1)]
        pub oldapi: bool,
        #[bits(1)]
        pub exclusive: bool,
        #[bits(4, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits602U0 {
        #[bits(1)]
        pub for_kupdate: u32,
        #[bits(1)]
        pub for_background: u32,
        #[bits(1)]
        pub tagged_writepages: u32,
        #[bits(1)]
        pub range_cyclic: u32,
        #[bits(1)]
        pub for_sync: u32,
        #[bits(1)]
        pub unpinned_netfs_wb: u32,
        #[bits(1)]
        pub no_cgroup_owner: u32,
        #[bits(1, access = na)]
        pub __slate_pad_7: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits642U0 {
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
    pub struct __SlateBits137U0 {
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
    pub struct __SlateBits232U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits476U0 {
        #[bits(1)]
        pub fullmm: u32,
        #[bits(1)]
        pub need_flush_all: u32,
        #[bits(1)]
        pub freed_tables: u32,
        #[bits(1)]
        pub delayed_rmap: u32,
        #[bits(1)]
        pub cleared_ptes: u32,
        #[bits(1)]
        pub cleared_pmds: u32,
        #[bits(1)]
        pub cleared_puds: u32,
        #[bits(1)]
        pub cleared_p4ds: u32,
        #[bits(1)]
        pub vma_exec: u32,
        #[bits(1)]
        pub vma_huge: u32,
        #[bits(1)]
        pub vma_pfn: u32,
        #[bits(1)]
        pub unshared_tables: u32,
        #[bits(1)]
        pub fully_unshared_tables: u32,
        #[bits(3, access = na)]
        pub __slate_pad_13: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits111U0 {
        #[bits(1)]
        pub just_expand: bool,
        #[bits(1)]
        pub give_up_on_oom: bool,
        #[bits(1)]
        pub skip_vma_uprobe: bool,
        #[bits(1)]
        pub __adjust_middle_start: bool,
        #[bits(1)]
        pub __adjust_next_start: bool,
        #[bits(1)]
        pub __remove_middle: bool,
        #[bits(1)]
        pub __remove_next: bool,
        #[bits(1, access = na)]
        pub __slate_pad_7: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1097U0 {
        #[bits(1)]
        pub skip_vma_uprobe: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1105U0 {
        #[bits(1)]
        pub check_ksm_early: bool,
        #[bits(1)]
        pub file_doesnt_need_get: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kvec {
    iov_base: *mut (),
    iov_len: u64,
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
struct compat_robust_list {
    next: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list_head {
    list: compat_robust_list,
    futex_offset: i32,
    list_op_pending: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits717U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fprop_local_percpu {
    events: percpu_counter,
    period: u32,
    lock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bdi_writeback {
    bdi: *mut backing_dev_info,
    state: u64,
    last_old_flush: u64,
    b_dirty: list_head,
    b_io: list_head,
    b_more_io: list_head,
    b_dirty_time: list_head,
    list_lock: spinlock,
    writeback_inodes: __SlateRecord_types_h_190,
    stat: [percpu_counter; 5],
    bw_time_stamp: u64,
    dirtied_stamp: u64,
    written_stamp: u64,
    write_bandwidth: u64,
    avg_write_bandwidth: u64,
    dirty_ratelimit: u64,
    balanced_dirty_ratelimit: u64,
    completions: fprop_local_percpu,
    dirty_exceeded: i32,
    start_all_reason: u32,
    work_lock: spinlock,
    work_list: list_head,
    dwork: delayed_work,
    bw_dwork: delayed_work,
    bdi_node: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct backing_dev_info {
    id: u64,
    rb_node: rb_node,
    bdi_list: list_head,
    ra_pages: u64,
    io_pages: u64,
    refcnt: kref,
    capabilities: u32,
    min_ratio: u32,
    max_ratio: u32,
    max_prop_frac: u32,
    tot_write_bandwidth: __SlateRecord_types_h_197,
    last_bdp_sleep: u64,
    wb: bdi_writeback,
    wb_list: list_head,
    wb_waitq: wait_queue_head,
    dev: *mut device,
    dev_name: [u8; 64],
    owner: *mut device,
    debug_dir: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vmem_altmap {
    base_pfn: u64,
    end_pfn: u64,
    reserve: u64,
    free: u64,
    align: u64,
    alloc: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pagemap_ops {
    folio_free: Option<unsafe extern "C" fn(*mut folio)>,
    migrate_to_ram: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    memory_failure: Option<unsafe extern "C" fn(*mut dev_pagemap, u64, u64, i32) -> i32>,
    folio_split: Option<unsafe extern "C" fn(*mut folio, *mut folio)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pagemap {
    altmap: vmem_altmap,
    r#ref: percpu_ref,
    done: completion,
    r#type: u32,
    flags: u32,
    vmemmap_shift: u64,
    ops: *const dev_pagemap_ops,
    owner: *mut (),
    nr_range: i32,
    __slate_anon_9: __SlateRecord_memremap_h_143,
}


// Despite the naming, these are FLAGS not bits.
// Bits set in the VMA until the stack is in its final location
// VMA basic access permission flags
// These flags can be updated atomically via VMA/mmap read lock.
// VMA flags we ignore for the purposes of merge, i.e. one VMA possessing one
// of these flags and the other not does not preclude a merge.
//
//    VMA_STICKY_FLAGS - When merging VMAs, VMA flags must match, unless they
//                       are 'sticky'. If any sticky flags exist in either VMA,
//                       we simply set all of them on the merged VMA.
#[repr(C)]
#[derive(Clone, Copy)]
struct vm_fault {
    __slate_anon_0: __SlateRecord_mm_h_754,
    flags: u32,
    pmd: *mut __SlateRecord_pgtable_64_types_h_22,
    pud: *mut __SlateRecord_pgtable_types_h_368,
    __slate_anon_4: __SlateRecord_mm_h_768,
    cow_page: *mut page,
    page: *mut page,
    pte: *mut __SlateRecord_pgtable_64_types_h_21,
    ptl: *mut spinlock,
    prealloc_pte: *mut page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_operations_struct {
    open: Option<unsafe extern "C" fn(*mut vm_area_struct)>,
    close: Option<unsafe extern "C" fn(*mut vm_area_struct)>,
    mapped: Option<unsafe extern "C" fn(u64, u64, u64, *const file, *mut *mut ()) -> i32>,
    may_split: Option<unsafe extern "C" fn(*mut vm_area_struct, u64) -> i32>,
    mremap: Option<unsafe extern "C" fn(*mut vm_area_struct) -> i32>,
    mprotect: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, u64, u64) -> i32>,
    fault: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    huge_fault: Option<unsafe extern "C" fn(*mut vm_fault, u32) -> u32>,
    map_pages: Option<unsafe extern "C" fn(*mut vm_fault, u64, u64) -> u32>,
    pagesize: Option<unsafe extern "C" fn(*mut vm_area_struct) -> u64>,
    page_mkwrite: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    pfn_mkwrite: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    access: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, *mut (), i32, i32) -> i32>,
    name: Option<unsafe extern "C" fn(*mut vm_area_struct) -> *const u8>,
    set_policy: Option<unsafe extern "C" fn(*mut vm_area_struct, *mut mempolicy) -> i32>,
    get_policy: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, *mut u64) -> *mut mempolicy>,
}


fn mm_flags_test(flag: i32, mm: *const mm_struct) -> bool {
test_bit__5!(flag, unsafe { (*core::ptr::addr_of!((*mm).__slate_anon_0.flags)).__mm_flags.as_ptr() as *const u64 }) != 0
}

#[inline(always)]
fn vma_flags_set_flag(flags: &mut __SlateRecord_mm_types_h_871, bit: i32) {
let bitmap: *mut u64 = unsafe { (*(flags as *mut __SlateRecord_mm_types_h_871)).__vma_flags.as_mut_ptr() as *mut u64 };
__set_bit__2!(bit, bitmap);
}

#[inline(always)]
fn __mk_vma_flags(mut flags: __SlateRecord_mm_types_h_871, count: u64, bits: *const i32) -> __SlateRecord_mm_types_h_871 {
let mut i: i32 = 0;
while (i as u64) < count {
vma_flags_set_flag(unsafe { &mut (*core::ptr::addr_of_mut!(flags)) }, unsafe { *bits.wrapping_offset(i as isize) });
i = i.wrapping_add(1);
}
return flags;
}

// Helper macro which bitwise-or combines the specified input flags into a
// vma_flags_t bitmap value. E.g.:
//
// vma_flags_t flags = mk_vma_flags(VMA_IO_BIT, VMA_PFNMAP_BIT,
//              VMA_DONTEXPAND_BIT, VMA_DONTDUMP_BIT);
//
// The compiler cleverly optimises away all of the work and this ends up being
// equivalent to aggregating the values manually.
#[inline(always)]
fn vma_flags_count(flags: &__SlateRecord_mm_types_h_871) -> i32 {
let bitmap: *const u64 = unsafe { (*(flags as *const __SlateRecord_mm_types_h_871)).__vma_flags.as_ptr() as *const u64 };
bitmap_weight(bitmap, 64u32) as i32
}

#[inline(always)]
fn vma_flags_test(flags: *const __SlateRecord_mm_types_h_871, bit: i32) -> bool {
let bitmap: *const u64 = unsafe { (*flags).__vma_flags.as_ptr() as *const u64 };
test_bit__5!(bit, bitmap) != 0
}

#[inline(always)]
fn vma_flags_and_mask(flags: *const __SlateRecord_mm_types_h_871, mut to_and: __SlateRecord_mm_types_h_871) -> __SlateRecord_mm_types_h_871 {
let mut dst: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(dst) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let bitmap_dst: *mut u64 = dst.__vma_flags.as_mut_ptr() as *mut u64;
let bitmap: *const u64 = unsafe { (*flags).__vma_flags.as_ptr() as *const u64 };
let bitmap_to_and: *const u64 = to_and.__vma_flags.as_mut_ptr() as *const u64;
bitmap_and(bitmap_dst, bitmap, bitmap_to_and, 64u32);
return dst;
}

#[inline(always)]
fn vma_flags_test_any_mask(flags: *const __SlateRecord_mm_types_h_871, mut to_test: __SlateRecord_mm_types_h_871) -> bool {
let bitmap: *const u64 = unsafe { (*flags).__vma_flags.as_ptr() as *const u64 };
let bitmap_to_test: *const u64 = to_test.__vma_flags.as_mut_ptr() as *const u64;
bitmap_intersects(bitmap_to_test, bitmap, 64u32)
}

// Test whether any specified VMA flag is set, e.g.:
//
// if (vma_flags_test_any(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
#[inline(always)]
fn vma_flags_test_all_mask(flags: *const __SlateRecord_mm_types_h_871, mut to_test: __SlateRecord_mm_types_h_871) -> bool {
let bitmap: *const u64 = unsafe { (*flags).__vma_flags.as_ptr() as *const u64 };
let bitmap_to_test: *const u64 = to_test.__vma_flags.as_mut_ptr() as *const u64;
bitmap_subset(bitmap_to_test, bitmap, 64u32)
}

// Test whether ALL specified VMA flags are set, e.g.:
//
// if (vma_flags_test_all(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
#[inline(always)]
fn vma_flags_test_single_mask(flags: *const __SlateRecord_mm_types_h_871, flagmask: __SlateRecord_mm_types_h_871) -> bool {
VM_WARN_ON_ONCE!();
vma_flags_test_any_mask(flags, flagmask)
}

#[inline(always)]
fn vma_flags_set_mask(flags: *mut __SlateRecord_mm_types_h_871, mut to_set: __SlateRecord_mm_types_h_871) {
let bitmap: *mut u64 = unsafe { (*flags).__vma_flags.as_mut_ptr() as *mut u64 };
let bitmap_to_set: *const u64 = to_set.__vma_flags.as_mut_ptr() as *const u64;
bitmap_or(bitmap, bitmap as *const u64, bitmap_to_set, 64u32);
}

// Set all specified VMA flags, e.g.:
//
// vma_flags_set(&flags, VMA_READ_BIT, VMA_WRITE_BIT, VMA_EXEC_BIT);
#[inline(always)]
fn vma_flags_clear_mask(flags: *mut __SlateRecord_mm_types_h_871, mut to_clear: __SlateRecord_mm_types_h_871) {
let bitmap: *mut u64 = unsafe { (*flags).__vma_flags.as_mut_ptr() as *mut u64 };
let bitmap_to_clear: *const u64 = to_clear.__vma_flags.as_mut_ptr() as *const u64;
bitmap_andnot(bitmap, bitmap as *const u64, bitmap_to_clear, 64u32);
}

#[inline(always)]
fn vma_flags_diff_pair(flags: *const __SlateRecord_mm_types_h_871, flags_other: *const __SlateRecord_mm_types_h_871) -> __SlateRecord_mm_types_h_871 {
let mut dst: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(dst) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let bitmap_other: *const u64 = unsafe { (*flags_other).__vma_flags.as_ptr() as *const u64 };
let bitmap: *const u64 = unsafe { (*flags).__vma_flags.as_ptr() as *const u64 };
let bitmap_dst: *mut u64 = dst.__vma_flags.as_mut_ptr() as *mut u64;
bitmap_xor(bitmap_dst, bitmap, bitmap_other, 64u32);
return dst;
}

#[inline(always)]
fn vma_flags_same_pair(flags: &__SlateRecord_mm_types_h_871, flags_other: &__SlateRecord_mm_types_h_871) -> bool {
let bitmap: *const u64 = unsafe { (*(flags as *const __SlateRecord_mm_types_h_871)).__vma_flags.as_ptr() as *const u64 };
let bitmap_other: *const u64 = unsafe { (*(flags_other as *const __SlateRecord_mm_types_h_871)).__vma_flags.as_ptr() as *const u64 };
bitmap_equal(bitmap, bitmap_other, 64u32)
}

#[inline(always)]
fn vma_test(vma: *const vm_area_struct, bit: i32) -> bool {
vma_flags_test(unsafe { core::ptr::addr_of!((*vma).__slate_anon_3.flags) }, bit)
}

#[inline(always)]
fn vma_test_any_mask(vma: *const vm_area_struct, flags: __SlateRecord_mm_types_h_871) -> bool {
vma_flags_test_any_mask(unsafe { core::ptr::addr_of!((*vma).__slate_anon_3.flags) }, flags)
}

#[inline(always)]
fn vma_test_all_mask(vma: *const vm_area_struct, flags: __SlateRecord_mm_types_h_871) -> bool {
vma_flags_test_all_mask(unsafe { core::ptr::addr_of!((*vma).__slate_anon_3.flags) }, flags)
}

// Helper macro for checking that ALL specified flags are set in a VMA, e.g.:
//
// if (vma_test_all(vma, VMA_READ_BIT, VMA_MAYREAD_BIT) { ... }
#[inline(always)]
fn vma_test_single_mask(vma: *const vm_area_struct, flagmask: __SlateRecord_mm_types_h_871) -> bool {
vma_flags_test_single_mask(unsafe { core::ptr::addr_of!((*vma).__slate_anon_3.flags) }, flagmask)
}

#[inline(always)]
fn vma_set_flags_mask(vma: *mut vm_area_struct, flags: __SlateRecord_mm_types_h_871) {
vma_flags_set_mask(unsafe { core::ptr::addr_of_mut!((*vma).__slate_anon_3.flags) }, flags);
}

// Helper macro for specifying VMA flags in a VMA, e.g.:
//
// vma_set_flags(vma, VMA_IO_BIT, VMA_PFNMAP_BIT, VMA_DONTEXPAND_BIT,
// 		VMA_DONTDUMP_BIT);
//
// Note: appropriate locks must be held, this function does not acquire them for
// you.
#[inline(always)]
fn vma_clear_flags_mask(vma: *mut vm_area_struct, flags: __SlateRecord_mm_types_h_871) {
vma_flags_clear_mask(unsafe { core::ptr::addr_of_mut!((*vma).__slate_anon_3.flags) }, flags);
}

fn vma_set_anonymous(vma: *mut vm_area_struct) {
unsafe {
(*vma).vm_ops = core::ptr::null::<vm_operations_struct>();
}
}

fn vma_is_anonymous(vma: *const vm_area_struct) -> bool {
unsafe { (*vma).vm_ops }.is_null()
}

fn vma_is_accessible(vma: *const vm_area_struct) -> bool {
(unsafe { (*vma).__slate_anon_3.vm_flags }) & (BIT!(0i32) | BIT!(1i32) | BIT!(2i32)) != 0
}

fn is_shared_maywrite(flags: *const __SlateRecord_mm_types_h_871) -> bool {
let mut __slate_compound_0: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: [i32; 2] = unsafe { core::mem::zeroed() };
vma_flags_test_all_mask(flags, __mk_vma_flags(unsafe { *{
__slate_compound_0 = {
let mut __t0: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t0
};
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 2u64, unsafe { (*{
__slate_compound_1 = [3i32, 5i32];
core::ptr::addr_of_mut!(__slate_compound_1)
}).as_ptr() as *const i32 }))
}

fn vma_is_shared_maywrite(vma: *const vm_area_struct) -> bool {
is_shared_maywrite(unsafe { core::ptr::addr_of!((*vma).__slate_anon_3.flags) })
}

fn vma_kernel_pagesize(vma: *mut vm_area_struct) -> u64 {
if unlikely!(!unsafe { (*vma).vm_ops }.is_null() && (unsafe { (*(*vma).vm_ops).pagesize }) != None) != 0 {
return unsafe { unsafe { (*(*vma).vm_ops).pagesize }.unwrap()(vma) };
}
_AC!() << 12i32
}

fn vma_find(vmi: *mut vma_iterator, max: u64) -> *mut vm_area_struct {
(unsafe { mas_find(core::ptr::addr_of_mut!((*vmi).mas), max.wrapping_sub(1)) }) as *mut vm_area_struct
}

fn vma_next(vmi: *mut vma_iterator) -> *mut vm_area_struct {
// Uses mas_find() to get the first VMA when the iterator starts.
// Calling mas_next() could skip the first entry.
(unsafe { mas_find(core::ptr::addr_of_mut!((*vmi).mas), 18446744073709551615u64) }) as *mut vm_area_struct
}

fn vma_iter_next_range(vmi: *mut vma_iterator) -> *mut vm_area_struct {
(unsafe { mas_next_range(core::ptr::addr_of_mut!((*vmi).mas), 18446744073709551615u64) }) as *mut vm_area_struct
}

fn vma_prev(vmi: *mut vma_iterator) -> *mut vm_area_struct {
(unsafe { mas_prev(core::ptr::addr_of_mut!((*vmi).mas), 0u64) }) as *mut vm_area_struct
}

fn vma_iter_clear_gfp(vmi: *mut vma_iterator, start: u64, end: u64, gfp: u32) -> i32 {
__mas_set_range(unsafe { core::ptr::addr_of_mut!((*vmi).mas) }, start, end.wrapping_sub(1));
unsafe { mas_store_gfp(core::ptr::addr_of_mut!((*vmi).mas), core::ptr::null_mut::<()>(), gfp) };
if unlikely!(mas_is_err(unsafe { core::ptr::addr_of_mut!((*vmi).mas) })) != 0 {
return 12i32.wrapping_neg();
}
0
}

fn vma_iter_free(vmi: *mut vma_iterator) {
unsafe { mas_destroy(core::ptr::addr_of_mut!((*vmi).mas)) };
}

fn vma_iter_invalidate(vmi: *mut vma_iterator) {
unsafe { mas_pause(core::ptr::addr_of_mut!((*vmi).mas)) };
}

fn vma_iter_set(vmi: *mut vma_iterator, addr: u64) {
mas_set(unsafe { core::ptr::addr_of_mut!((*vmi).mas) }, addr);
}

fn vma_adjust_trans_huge(vma: *mut vm_area_struct, start: u64, end: u64, next: *mut vm_area_struct) {
}

// The MM code likes to work with exclusive end addresses
// 127: arbitrary random number, small enough to assemble well
// GUP_PIN_COUNTING_BIAS, and the associated functions that use it, overload
// the page's refcount so that two separate items are tracked: the original page
// reference count, and also a new count of how many pin_user_pages() calls were
// made against the page. ("gup-pinned" is another term for the latter).
//
// With this scheme, pin_user_pages() becomes special: such pages are marked as
// distinct from normal pages. As such, the unpin_user_page() call (and its
// variants) must be used in order to release gup-pinned pages.
//
// Choice of value:
//
// By making GUP_PIN_COUNTING_BIAS a power of two, debugging of page reference
// counts with respect to pin_user_pages() and unpin_user_page() becomes
// simpler, due to the fact that adding an even power of two to the page
// refcount has the effect of using only the upper N bits, for the code that
// counts up using the bias value. This means that the lower bits are left for
// the exclusive use of the original code that increments and decrements by one
// (or at least, by much smaller values than the bias value).
//
// Of course, once the lower bits overflow into the upper bits (and this is
// OK, because subtraction recovers the original values), then visual inspection
// no longer suffices to directly view the separate counts. However, for normal
// applications that don't have huge page reference counts, this won't be an
// issue.
//
// Locking: the lockless algorithm described in folio_try_get_rcu()
// provides safe operation for get_user_pages(), folio_mkclean() and
// other calls that race to set up page table entries.
fn vma_flags_is_cow_mapping(flags: *const __SlateRecord_mm_types_h_871) -> bool {
vma_flags_test(flags, 5i32) && !vma_flags_test(flags, 3i32)
}

fn get_mm_counter(mm: *mut mm_struct, member: i32) -> u64 {
percpu_counter_read_positive(unsafe { (*mm).__slate_anon_0.rss_stat.as_mut_ptr() as *mut percpu_counter }.wrapping_offset(member as isize)) as u64
}

fn get_mm_rss(mm: *mut mm_struct) -> u64 {
get_mm_counter(mm, 0i32).wrapping_add(get_mm_counter(mm, 1i32)).wrapping_add(get_mm_counter(mm, 3i32))
}

fn update_hiwater_rss(mm: *mut mm_struct) {
let _rss: u64 = get_mm_rss(mm);
if data_race__2!(unsafe { (*mm).__slate_anon_0.hiwater_rss }) < _rss {
let mut __v2: u64;
__kcsan_disable_current();
let mut __v: u64 = 0;
let __v3: u64 = _rss;
unsafe {
(*mm).__slate_anon_0.hiwater_rss = __v3;
}
__v = __v3;
__kcsan_enable_current();
__v2 = __v;
}
}

fn update_hiwater_vm(mm: *mut mm_struct) {
if (unsafe { (*mm).__slate_anon_0.hiwater_vm }) < unsafe { (*mm).__slate_anon_0.total_vm } {
unsafe {
(*mm).__slate_anon_0.hiwater_vm = (*mm).__slate_anon_0.total_vm;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vm_unmapped_area_info {
    flags: u64,
    length: u64,
    low_limit: u64,
    high_limit: u64,
    align_mask: u64,
    align_offset: u64,
    start_gap: u64,
}


fn stack_guard_start_gap(vma: *const vm_area_struct) -> u64 {
if (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(8i32) != 0 {
return unsafe { stack_guard_gap };
}
// See reasoning around the VM_SHADOW_STACK definition
if (unsafe { (*vma).__slate_anon_3.vm_flags }) & 0 != 0 {
return _AC!() << 12i32;
}
0
}

fn vm_start_gap(vma: *const vm_area_struct) -> u64 {
let gap: u64 = stack_guard_start_gap(vma);
let mut vm_start: u64 = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start };
vm_start = vm_start.wrapping_sub(gap);
if vm_start > unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start } {
vm_start = 0;
}
vm_start
}

fn vm_end_gap(vma: *const vm_area_struct) -> u64 {
let mut vm_end: u64 = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end };
if (unsafe { (*vma).__slate_anon_3.vm_flags }) & 0 != 0 {
vm_end = vm_end.wrapping_add(unsafe { stack_guard_gap });
if vm_end < unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end } {
vm_end = (_AC!() << 12i32).wrapping_neg();
}
}
vm_end
}

fn vma_pages(vma: *const vm_area_struct) -> u64 {
unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end }.wrapping_sub(unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }) >> 12i32
}

fn vma_start_pgoff(vma: *const vm_area_struct) -> u64 {
unsafe { (*vma).vm_pgoff }
}

fn vma_end_pgoff(vma: *const vm_area_struct) -> u64 {
vma_start_pgoff(vma).wrapping_add(vma_pages(vma))
}

fn vma_start_anon_pgoff(vma: *const vm_area_struct) -> u64 {
let mut pgoff: u64 = 0;
pgoff = pgoff.wrapping_add((unsafe { (*vma).__vm_anon_pgoff_hi }) as u64);
pgoff <<= 32i32;
pgoff = pgoff.wrapping_add((unsafe { (*vma).__vm_anon_pgoff_lo }) as u64);
pgoff
}

fn vma_end_anon_pgoff(vma: *const vm_area_struct) -> u64 {
vma_start_anon_pgoff(vma).wrapping_add(vma_pages(vma))
}

fn vma_flags_to_page_prot(vma_flags: __SlateRecord_mm_types_h_871) -> pgprot {
let vm_flags: u64 = vma_flags_to_legacy(vma_flags);
return unsafe { vm_get_page_prot(vm_flags) };
}

fn vma_get_page_prot(vma: *const vm_area_struct) -> pgprot {
return vma_flags_to_page_prot(unsafe { (*vma).__slate_anon_3.flags });
}

// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
fn flush_dcache_mmap_lock(mapping: *mut address_space) {
}

fn flush_dcache_mmap_unlock(mapping: *mut address_space) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bio_vec {
    bv_page: *mut page,
    bv_len: u32,
    bv_offset: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
    bd_start_sect: u64,
    bd_nr_sectors: u64,
    bd_disk: *mut gendisk,
    bd_queue: *mut request_queue,
    bd_stats: *mut disk_stats,
    bd_stamp: u64,
    __bd_flags: __SlateRecord_types_h_190,
    bd_dev: u32,
    bd_mapping: *mut address_space,
    bd_openers: __SlateRecord_types_h_190,
    bd_size_lock: spinlock,
    bd_claiming: *mut (),
    bd_holder: *mut (),
    bd_holder_ops: *const blk_holder_ops,
    bd_holder_lock: mutex,
    bd_holders: i32,
    bd_holder_dir: *mut kobject,
    bd_fsfreeze_count: __SlateRecord_types_h_190,
    bd_fsfreeze_mutex: mutex,
    bd_meta_info: *mut partition_meta_info,
    bd_writers: i32,
    bd_security: *mut (),
    bd_device: device,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_batch {
    nr: u8,
    i: u8,
    percpu_pvec_drained: bool,
    folios: [*mut folio; 31],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct writeback_control {
    nr_to_write: i64,
    pages_skipped: i64,
    range_start: i64,
    range_end: i64,
    sync_mode: u32,
    __slate_bits_0: __slate_bits::__SlateBits602U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


fn mapping_can_writeback(mapping: *mut address_space) -> bool {
(unsafe { (*inode_to_bdi((*mapping).host)).capabilities }) & 1u32 != 0
}

fn is_vma_hugetlb_flags(flags: *const __SlateRecord_mm_types_h_871) -> bool {
vma_flags_test(flags, 22i32)
}

fn is_vm_hugetlb_page(vma: *const vm_area_struct) -> bool {
is_vma_hugetlb_flags(unsafe { core::ptr::addr_of!((*vma).__slate_anon_3.flags) })
}

fn linear_page_delta(vma: *const vm_area_struct, address: u64) -> u64 {
address.wrapping_sub(unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }) >> 12i32
}

fn linear_page_index(vma: *const vm_area_struct, address: u64) -> u64 {
linear_page_delta(vma, address).wrapping_add(vma_start_pgoff(vma))
}

fn __linear_anon_page_index(vma: *const vm_area_struct, address: u64) -> u64 {
linear_page_delta(vma, address).wrapping_add(vma_start_anon_pgoff(vma))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct wait_page_queue {
    folio: *mut folio,
    bit_nr: i32,
    wait: wait_queue_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct readahead_control {
    file: *mut file,
    mapping: *mut address_space,
    ra: *mut file_ra_state,
    _index: u64,
    _nr_pages: u32,
    _batch_count: u32,
    dropbehind: bool,
    _workingset: bool,
    _pflags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mempolicy {
    refcnt: __SlateRecord_types_h_190,
    mode: u16,
    flags: u16,
    nodes: __SlateRecord_nodemask_types_h_17,
    home_node: i32,
    w: __SlateRecord_mempolicy_h_54,
    rcu: callback_head,
}


fn mpol_put(pol: *mut mempolicy) {
if !pol.is_null() {
unsafe { __mpol_put(pol) };
}
}

fn mpol_equal(a: *mut mempolicy, b: *mut mempolicy) -> bool {
if a == b {
return true;
}
unsafe { __mpol_equal(a, b) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct taskstats {
    version: u16,
    ac_exitcode: u32,
    ac_flag: u8,
    ac_nice: u8,
    cpu_count: u64,
    cpu_delay_total: u64,
    blkio_count: u64,
    blkio_delay_total: u64,
    swapin_count: u64,
    swapin_delay_total: u64,
    cpu_run_real_total: u64,
    cpu_run_virtual_total: u64,
    ac_comm: [u8; 32],
    ac_sched: u8,
    ac_pad: [u8; 3],
    __slate_pad_15: [u8; 4],
    ac_uid: u32,
    ac_gid: u32,
    ac_pid: u32,
    ac_ppid: u32,
    ac_btime: u32,
    ac_etime: u64,
    ac_utime: u64,
    ac_stime: u64,
    ac_minflt: u64,
    ac_majflt: u64,
    coremem: u64,
    virtmem: u64,
    hiwater_rss: u64,
    hiwater_vm: u64,
    read_char: u64,
    write_char: u64,
    read_syscalls: u64,
    write_syscalls: u64,
    read_bytes: u64,
    write_bytes: u64,
    cancelled_write_bytes: u64,
    nvcsw: u64,
    nivcsw: u64,
    ac_utimescaled: u64,
    ac_stimescaled: u64,
    cpu_scaled_run_real_total: u64,
    freepages_count: u64,
    freepages_delay_total: u64,
    thrashing_count: u64,
    thrashing_delay_total: u64,
    ac_btime64: u64,
    compact_count: u64,
    compact_delay_total: u64,
    ac_tgid: u32,
    ac_tgetime: u64,
    ac_exe_dev: u64,
    ac_exe_inode: u64,
    wpcopy_count: u64,
    wpcopy_delay_total: u64,
    irq_count: u64,
    irq_delay_total: u64,
    cpu_delay_max: u64,
    cpu_delay_min: u64,
    blkio_delay_max: u64,
    blkio_delay_min: u64,
    swapin_delay_max: u64,
    swapin_delay_min: u64,
    freepages_delay_max: u64,
    freepages_delay_min: u64,
    thrashing_delay_max: u64,
    thrashing_delay_min: u64,
    compact_delay_max: u64,
    compact_delay_min: u64,
    wpcopy_delay_max: u64,
    wpcopy_delay_min: u64,
    irq_delay_max: u64,
    irq_delay_min: u64,
    cpu_delay_max_ts: __kernel_timespec,
    blkio_delay_max_ts: __kernel_timespec,
    swapin_delay_max_ts: __kernel_timespec,
    freepages_delay_max_ts: __kernel_timespec,
    thrashing_delay_max_ts: __kernel_timespec,
    compact_delay_max_ts: __kernel_timespec,
    wpcopy_delay_max_ts: __kernel_timespec,
    irq_delay_max_ts: __kernel_timespec,
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
struct ns_tree_root {
    ns_rb: rb_root,
    ns_list_head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree_node {
    ns_node: rb_node,
    ns_list_entry: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree {
    ns_id: u64,
    __ns_ref_active: __SlateRecord_types_h_190,
    ns_unified_node: ns_tree_node,
    ns_tree_node: ns_tree_node,
    ns_owner_node: ns_tree_node,
    ns_owner_root: ns_tree_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_common {
    __slate_anon_0: __SlateRecord_ns_common_types_h_112,
    ns_type: u32,
    stashed: *mut dentry,
    ops: *const proc_ns_operations,
    inum: u32,
    __slate_anon_5: ns_tree,
    ns_rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nsproxy {
    count: refcount_struct,
    uts_ns: *mut uts_namespace,
    ipc_ns: *mut ipc_namespace,
    mnt_ns: *mut mnt_namespace,
    pid_ns_for_children: *mut pid_namespace,
    net_ns: *mut net,
    time_ns: *mut time_namespace,
    time_ns_for_children: *mut time_namespace,
    cgroup_ns: *mut cgroup_namespace,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uid_gid_extent {
    first: u32,
    lower_first: u32,
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uid_gid_map {
    __slate_anon_0: __SlateRecord_user_namespace_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
    uid_map: uid_gid_map,
    gid_map: uid_gid_map,
    projid_map: uid_gid_map,
    parent: *mut user_namespace,
    level: i32,
    owner: __SlateRecord_uidgid_types_h_9,
    group: __SlateRecord_uidgid_types_h_13,
    ns: ns_common,
    flags: u64,
    parent_could_setfcap: bool,
    keyring_name_list: list_head,
    user_keyring_register: *mut key,
    keyring_sem: rw_semaphore,
    work: work_struct,
    set: ctl_table_set,
    sysctls: *mut ctl_table_header,
    ucounts: *mut ucounts,
    ucount_max: [i64; 11],
    rlimit_max: [i64; 4],
    binfmt_misc: *mut binfmt_misc,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ucounts {
    node: hlist_nulls_node,
    ns: *mut user_namespace,
    uid: __SlateRecord_uidgid_types_h_9,
    rcu: callback_head,
    count: __SlateRecord_types_h_202,
    ucount: [__SlateRecord_types_h_197; 11],
    rlimit: [__SlateRecord_types_h_197; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_stat {
    irqs_sum: u64,
    softirqs: [u32; 10],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct u64_stats_sync {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_local_h_12 {
    a: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_local64_h_26 {
    a: __SlateRecord_local_h_12,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_bpf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct psi_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_file {
    kn: *mut kernfs_node,
    notified_at: u64,
    notify_timer: timer_list,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_subsys_state {
    cgroup: *mut cgroup,
    ss: *mut cgroup_subsys,
    refcnt: percpu_ref,
    rstat_cpu: *mut css_rstat_cpu,
    sibling: list_head,
    children: list_head,
    id: i32,
    flags: u32,
    serial_nr: u64,
    online_cnt: __SlateRecord_types_h_190,
    destroy_work: work_struct,
    destroy_rwork: rcu_work,
    parent: *mut cgroup_subsys_state,
    nr_descendants: i32,
    nr_populated_csets: i32,
    nr_populated_children: i32,
    kill_finish_work: work_struct,
    rstat_flush_next: *mut cgroup_subsys_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_set {
    subsys: [*mut cgroup_subsys_state; 14],
    refcount: refcount_struct,
    dom_cset: *mut css_set,
    dfl_cgrp: *mut cgroup,
    nr_tasks: i32,
    tasks: list_head,
    mg_tasks: list_head,
    dying_tasks: list_head,
    task_iters: list_head,
    e_cset_node: [list_head; 14],
    threaded_csets: list_head,
    threaded_csets_node: list_head,
    hlist: hlist_node,
    cgrp_links: list_head,
    mg_src_preload_node: list_head,
    mg_dst_preload_node: list_head,
    mg_node: list_head,
    mg_src_cgrp: *mut cgroup,
    mg_dst_cgrp: *mut cgroup,
    mg_dst_cset: *mut css_set,
    dead: bool,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_base_stat {
    cputime: task_cputime,
    ntime: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_rstat_cpu {
    updated_children: *mut cgroup_subsys_state,
    updated_next: *mut cgroup_subsys_state,
    lnode: llist_node,
    owner: *mut cgroup_subsys_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_rstat_base_cpu {
    bsync: u64_stats_sync,
    bstat: cgroup_base_stat,
    last_bstat: cgroup_base_stat,
    subtree_bstat: cgroup_base_stat,
    last_subtree_bstat: cgroup_base_stat,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_freezer_state {
    freeze: bool,
    e_freeze: bool,
    nr_frozen_descendants: i32,
    nr_frozen_tasks: i32,
    freeze_seq: seqcount_spinlock,
    freeze_start_nsec: u64,
    frozen_nsec: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup {
    self_: cgroup_subsys_state,
    flags: u64,
    level: i32,
    max_depth: i32,
    nr_descendants: i32,
    nr_dying_descendants: i32,
    max_descendants: i32,
    nr_populated_domain_children: i32,
    nr_populated_threaded_children: i32,
    nr_threaded_children: i32,
    kill_seq: u32,
    kn: *mut kernfs_node,
    procs_file: cgroup_file,
    events_file: cgroup_file,
    psi_files: [cgroup_file; 0],
    subtree_control: u32,
    subtree_ss_mask: u32,
    old_subtree_control: u32,
    old_subtree_ss_mask: u32,
    subsys: [*mut cgroup_subsys_state; 14],
    nr_dying_subsys: [i32; 14],
    root: *mut cgroup_root,
    cset_links: list_head,
    e_csets: [list_head; 14],
    dom_cgrp: *mut cgroup,
    old_dom_cgrp: *mut cgroup,
    rstat_base_cpu: *mut cgroup_rstat_base_cpu,
    _pad_: cacheline_padding,
    last_bstat: cgroup_base_stat,
    bstat: cgroup_base_stat,
    prev_cputime: prev_cputime,
    pidlists: list_head,
    pidlist_mutex: mutex,
    offline_waitq: wait_queue_head,
    release_agent_work: work_struct,
    psi: *mut psi_group,
    bpf: cgroup_bpf,
    freezer: cgroup_freezer_state,
    __slate_anon_38: __SlateRecord_cgroup_defs_h_644,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_root {
    kf_root: *mut kernfs_root,
    subsys_mask: u32,
    hierarchy_id: i32,
    root_list: list_head,
    rcu: callback_head,
    nr_cgrps: __SlateRecord_types_h_190,
    flags: u32,
    release_agent_path: [u8; 4096],
    name: [u8; 64],
    cgrp: cgroup,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cftype {
    name: [u8; 64],
    private: u64,
    max_write_len: u64,
    flags: u32,
    file_offset: u32,
    ss: *mut cgroup_subsys,
    node: list_head,
    kf_ops: *mut kernfs_ops,
    open: Option<unsafe extern "C" fn(*mut kernfs_open_file) -> i32>,
    release: Option<unsafe extern "C" fn(*mut kernfs_open_file)>,
    read_u64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype) -> u64>,
    read_s64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype) -> i64>,
    seq_show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
    seq_start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    seq_next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    seq_stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    write_u64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype, u64) -> i32>,
    write_s64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype, i64) -> i32>,
    write: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    poll: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut poll_table_struct) -> u32>,
    lockdep_key: lock_class_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_subsys {
    css_alloc: Option<unsafe extern "C" fn(*mut cgroup_subsys_state) -> *mut cgroup_subsys_state>,
    css_online: Option<unsafe extern "C" fn(*mut cgroup_subsys_state) -> i32>,
    css_offline: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_released: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_free: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_reset: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_killed: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_rstat_flush: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, i32)>,
    css_extra_stat_show: Option<unsafe extern "C" fn(*mut seq_file, *mut cgroup_subsys_state) -> i32>,
    css_local_stat_show: Option<unsafe extern "C" fn(*mut seq_file, *mut cgroup_subsys_state) -> i32>,
    can_attach: Option<unsafe extern "C" fn(*mut cgroup_taskset) -> i32>,
    cancel_attach: Option<unsafe extern "C" fn(*mut cgroup_taskset)>,
    attach: Option<unsafe extern "C" fn(*mut cgroup_taskset)>,
    can_fork: Option<unsafe extern "C" fn(*mut task_struct, *mut css_set) -> i32>,
    cancel_fork: Option<unsafe extern "C" fn(*mut task_struct, *mut css_set)>,
    fork: Option<unsafe extern "C" fn(*mut task_struct)>,
    exit: Option<unsafe extern "C" fn(*mut task_struct)>,
    release: Option<unsafe extern "C" fn(*mut task_struct)>,
    bind: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    __slate_bits_0: __slate_bits::__SlateBits846U0,
    id: i32,
    name: *const u8,
    legacy_name: *const u8,
    root: *mut cgroup_root,
    css_idr: idr,
    cfts: list_head,
    dfl_cftypes: *mut cftype,
    legacy_cftypes: *mut cftype,
    depends_on: u32,
    rstat_ss_lock: spinlock,
    lhead: *mut llist_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_namespace {
    ns: ns_common,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    root_cset: *mut css_set,
}


fn is_mergeable_vm_userfaultfd_ctx(vma: *mut vm_area_struct, vm_ctx: vm_userfaultfd_ctx) -> bool {
true
}

fn userfaultfd_wp(vma: *mut vm_area_struct) -> bool {
false
}

fn userfaultfd_unmap_prep(vma: *mut vm_area_struct, start: u64, end: u64, uf: *mut list_head) -> i32 {
0
}

fn userfaultfd_unmap_complete(mm: *mut mm_struct, uf: &list_head) {
}

fn is_file_hugepages(file: *const file) -> bool {
(unsafe { (*(*file).f_op).fop_flags }) & 16u32 != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct reclaim_state {
    reclaimed: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_sequential_cluster {
    next: [u32; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
    users: percpu_ref,
    flags: u64,
    prio: i16,
    list: plist_node,
    r#type: i8,
    max: u32,
    cluster_info: *mut swap_cluster_info,
    free_clusters: list_head,
    full_clusters: list_head,
    nonfull_clusters: [list_head; 1],
    frag_clusters: [list_head; 1],
    pages: u32,
    inuse_pages: __SlateRecord_types_h_197,
    global_cluster: *mut swap_sequential_cluster,
    global_cluster_lock: spinlock,
    swap_extent_root: rb_root,
    bdev: *mut block_device,
    swap_file: *mut file,
    comp: completion,
    lock: spinlock,
    discard_work: work_struct,
    reclaim_work: work_struct,
    discard_clusters: list_head,
    avail_list: plist_node,
    ops: *const swap_ops,
}


fn is_hugepage_only_range(mm: *mut mm_struct, addr: u64, len: u64) -> i32 {
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct anon_vma {
    root: *mut anon_vma,
    rwsem: rw_semaphore,
    refcount: __SlateRecord_types_h_190,
    num_children: u64,
    num_active_vmas: u64,
    parent: *mut anon_vma,
    rb_root: rb_root_cached,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct anon_vma_chain {
    vma: *mut vm_area_struct,
    anon_vma: *mut anon_vma,
    same_vma: list_head,
    rb: rb_node,
    rb_subtree_last: u64,
}


fn ksm_vma_flags(mm: *mut mm_struct, file: *const file, vma_flags: __SlateRecord_mm_types_h_871) -> __SlateRecord_mm_types_h_871 {
return vma_flags;
}

fn khugepaged_enter_vma(vma: *mut vm_area_struct, vm_flags: u64) {
}

fn anon_vma_name_eq(anon_name1: *mut anon_vma_name, anon_name2: *mut anon_vma_name) -> bool {
true
}

fn vm_acct_memory(pages: i64) {
unsafe { percpu_counter_add_batch(core::ptr::addr_of_mut!(vm_committed_as), pages, vm_committed_as_batch) };
}

fn vm_unacct_memory(pages: i64) {
vm_acct_memory(pages.wrapping_neg());
}

fn arch_validate_flags(flags: u64) -> bool {
true
}

// Optimisation macro.  It is equivalent to:
//      (x & bit1) ? bit2 : 0
// but this version is faster.
// ("bit1" and "bit2" must be single bits)
#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
    next: *mut static_call_mod,
    r#mod: *mut module,
    sites: *mut static_call_site,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_eval_map {
    system: *const u8,
    eval_string: *const u8,
    eval_value: u64,
}


// When a tracepoint is used, it's name is added to the __tracepoint_check
// section. This section is only used at build time to make sure all
// defined tracepoints are used. It is discarded after the build.
// Make sure the alignment of the structure in the __tracepoints section will
// not add unwanted padding between the beginning of the section and the
// structure. Force alignment to the same alignment as the section start.
//
// When lockdep is enabled, we make sure to always test if RCU is
// "watching" regardless if the tracepoint is enabled or not. Tracepoints
// require RCU to be active, and it should always warn at the tracepoint
// site if it is not watching, as it will need to be active when the
// tracepoint is enabled.
#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
    entries: *mut desc_struct,
    nr_entries: u32,
    slot: i32,
}


// Init a new mm.  Used on mm copies, like at fork()
// and on mm's that are brand-new, like at execve().
#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_attr {
    r#type: u32,
    size: u32,
    config: u64,
    __slate_anon_3: __SlateRecord_perf_event_h_410,
    sample_type: u64,
    read_format: u64,
    __slate_bits_0: __slate_bits::__SlateBits864U0,
    __slate_anon_46: __SlateRecord_perf_event_h_471,
    bp_type: u32,
    __slate_anon_48: __SlateRecord_perf_event_h_477,
    __slate_anon_49: __SlateRecord_perf_event_h_483,
    branch_sample_type: u64,
    sample_regs_user: u64,
    sample_stack_user: u32,
    clockid: i32,
    sample_regs_intr: u64,
    aux_watermark: u32,
    sample_max_stack: u16,
    __reserved_2: u16,
    aux_sample_size: u32,
    __slate_anon_59: __SlateRecord_perf_event_h_530,
    sig_data: u64,
    config3: u64,
    config4: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union perf_mem_data_src {
    val: u64,
    __slate_anon_1: __SlateRecord_perf_event_h_1322,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_branch_entry {
    from: u64,
    to: u64,
    __slate_bits_0: __slate_bits::__SlateBits873U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union perf_sample_weight {
    full: u64,
    __slate_anon_1: __SlateRecord_perf_event_h_1512,
}


// MFD_EXEC implied if unset
#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
    idr: idr,
    rcu: callback_head,
    pid_allocated: u32,
    memfd_noexec_scope: i32,
    set: ctl_table_set,
    sysctls: *mut ctl_table_header,
    child_reaper: *mut task_struct,
    pid_cachep: *mut kmem_cache,
    level: u32,
    pid_max: i32,
    parent: *mut pid_namespace,
    bacct: *mut fs_pin,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    reboot: i32,
    ns: ns_common,
    work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_hw_breakpoint {
    address: u64,
    mask: u64,
    len: u8,
    r#type: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_regs {
    abi: u64,
    regs: *mut pt_regs,
}


// double word (64-bit)
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_insn {
    code: u8,
    __slate_bits_0: __slate_bits::__SlateBits881U0,
    off: i16,
    imm: i32,
}


// 1 << 13
// insn[0].src_reg:  BPF_PSEUDO_FUNC
// insn[0].imm:      insn offset to the func
// insn[1].imm:      0
// insn[0].off:      0
// insn[1].off:      0
// ldimm64 rewrite:  address of the function
// verifier type:    PTR_TO_FUNC.
#[repr(C)]
#[derive(Clone, Copy)]
union bpf_attr {
    __slate_anon_0: __SlateRecord_bpf_h_1528,
    __slate_anon_1: __SlateRecord_bpf_h_1581,
    batch: __SlateRecord_bpf_h_1591,
    __slate_anon_3: __SlateRecord_bpf_h_1608,
    __slate_anon_4: __SlateRecord_bpf_h_1674,
    __slate_anon_5: __SlateRecord_bpf_h_1687,
    test: __SlateRecord_bpf_h_1703,
    __slate_anon_7: __SlateRecord_bpf_h_1727,
    info: __SlateRecord_bpf_h_1740,
    query: __SlateRecord_bpf_h_1746,
    raw_tracepoint: __SlateRecord_bpf_h_1769,
    __slate_anon_11: __SlateRecord_bpf_h_1776,
    task_fd_query: __SlateRecord_bpf_h_1794,
    link_create: __SlateRecord_bpf_h_1810,
    link_update: __SlateRecord_bpf_h_1895,
    link_detach: __SlateRecord_bpf_h_1916,
    enable_stats: __SlateRecord_bpf_h_1920,
    iter_create: __SlateRecord_bpf_h_1924,
    prog_bind_map: __SlateRecord_bpf_h_1929,
    token_create: __SlateRecord_bpf_h_1935,
    prog_stream_read: __SlateRecord_bpf_h_1940,
    prog_assoc_struct_ops: __SlateRecord_bpf_h_1947,
}


// The description below is an attempt at providing documentation to eBPF
// developers about the multiple available eBPF helper functions. It can be
// parsed and used to produce a manual page. The workflow is the following,
// and requires the rst2man utility:
//
//     $ ./scripts/bpf_doc.py  *             --filename include/uapi/linux/bpf.h > /tmp/bpf-helpers.rst
//     $ rst2man /tmp/bpf-helpers.rst > /tmp/bpf-helpers.7
//     $ man /tmp/bpf-helpers.7
//
// Note that in order to produce this external documentation, some RST
// formatting is used in the descriptions to get "bold" and "italics" in
// manual pages. Also note that the few trailing white spaces are
// intentional, removing them would break paragraphs for rst2man.
//
// Start of BPF helper function descriptions:
//
// void *bpf_map_lookup_elem(struct bpf_map *map, const void *key)
// 	Description
// 		Perform a lookup in *map* for an entry associated to *key*.
// 	Return
// 		Map value associated to *key*, or **NULL** if no entry was
// 		found.
//
// long bpf_map_update_elem(struct bpf_map *map, const void *key, const void *value, u64 flags)
// 	Description
// 		Add or update the value of the entry associated to *key* in
// 		*map* with *value*. *flags* is one of:
//
// 		**BPF_NOEXIST**
// 			The entry for *key* must not exist in the map.
// 		**BPF_EXIST**
// 			The entry for *key* must already exist in the map.
// 		**BPF_ANY**
// 			No condition on the existence of the entry for *key*.
//
// 		Flag value **BPF_NOEXIST** cannot be used for maps of types
// 		**BPF_MAP_TYPE_ARRAY** or **BPF_MAP_TYPE_PERCPU_ARRAY**  (all
// 		elements always exist), the helper would return an error.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_delete_elem(struct bpf_map *map, const void *key)
// 	Description
// 		Delete entry with *key* from *map*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		For tracing programs, safely attempt to read *size* bytes from
// 		kernel space address *unsafe_ptr* and store the data in *dst*.
//
// 		Generally, use **bpf_probe_read_user**\ () or
// 		**bpf_probe_read_kernel**\ () instead.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_ktime_get_ns(void)
// 	Description
// 		Return the time elapsed since system boot, in nanoseconds.
// 		Does not include time the system was suspended.
// 		See: **clock_gettime**\ (**CLOCK_MONOTONIC**)
// 	Return
// 		Current *ktime*.
//
// long bpf_trace_printk(const char *fmt, u32 fmt_size, ...)
// 	Description
// 		This helper is a "printk()-like" facility for debugging. It
// 		prints a message defined by format *fmt* (of size *fmt_size*)
// 		to file *\/sys/kernel/tracing/trace* from TraceFS, if
// 		available. It can take up to three additional **u64**
// 		arguments (as an eBPF helpers, the total number of arguments is
// 		limited to five).
//
// 		Each time the helper is called, it appends a line to the trace.
// 		Lines are discarded while *\/sys/kernel/tracing/trace* is
// 		open, use *\/sys/kernel/tracing/trace_pipe* to avoid this.
// 		The format of the trace is customizable, and the exact output
// 		one will get depends on the options set in
// 		*\/sys/kernel/tracing/trace_options* (see also the
// 		*README* file under the same directory). However, it usually
// 		defaults to something like:
//
// 		::
//
// 			telnet-470   [001] .N.. 419421.045894: 0x00000001: <formatted msg>
//
// 		In the above:
//
// 			* ``telnet`` is the name of the current task.
// 			* ``470`` is the PID of the current task.
// 			* ``001`` is the CPU number on which the task is
// 			  running.
// 			* In ``.N..``, each character refers to a set of
// 			  options (whether irqs are enabled, scheduling
// 			  options, whether hard/softirqs are running, level of
// 			  preempt_disabled respectively). **N** means that
// 			  **TIF_NEED_RESCHED** and **PREEMPT_NEED_RESCHED**
// 			  are set.
// 			* ``419421.045894`` is a timestamp.
// 			* ``0x00000001`` is a fake value used by BPF for the
// 			  instruction pointer register.
// 			* ``<formatted msg>`` is the message formatted with
// 			  *fmt*.
//
// 		The conversion specifiers supported by *fmt* are similar, but
// 		more limited than for printk(). They are **%d**, **%i**,
// 		**%u**, **%x**, **%ld**, **%li**, **%lu**, **%lx**, **%lld**,
// 		**%lli**, **%llu**, **%llx**, **%p**, **%s**. No modifier (size
// 		of field, padding with zeroes, etc.) is available, and the
// 		helper will return **-EINVAL** (but print nothing) if it
// 		encounters an unknown specifier.
//
// 		Also, note that **bpf_trace_printk**\ () is slow, and should
// 		only be used for debugging purposes. For this reason, a notice
// 		block (spanning several lines) is printed to kernel logs and
// 		states that the helper should not be used "for production use"
// 		the first time this helper is used (or more precisely, when
// 		**trace_printk**\ () buffers are allocated). For passing values
// 		to user space, perf events should be preferred.
// 	Return
// 		The number of bytes written to the buffer, or a negative error
// 		in case of failure.
//
// u32 bpf_get_prandom_u32(void)
// 	Description
// 		Get a pseudo-random number.
//
// 		From a security point of view, this helper uses its own
// 		pseudo-random internal state, and cannot be used to infer the
// 		seed of other random functions in the kernel. However, it is
// 		essential to note that the generator used by the helper is not
// 		cryptographically secure.
// 	Return
// 		A random 32-bit unsigned value.
//
// u32 bpf_get_smp_processor_id(void)
// 	Description
// 		Get the SMP (symmetric multiprocessing) processor id. Note that
// 		all programs run with migration disabled, which means that the
// 		SMP processor id is stable during all the execution of the
// 		program.
// 	Return
// 		The SMP id of the processor running the program.
// 	Attributes
// 		__bpf_fastcall
//
// long bpf_skb_store_bytes(struct sk_buff *skb, u32 offset, const void *from, u32 len, u64 flags)
// 	Description
// 		Store *len* bytes from address *from* into the packet
// 		associated to *skb*, at *offset*. The *flags* are a combination
// 		of the following values:
//
// 		**BPF_F_RECOMPUTE_CSUM**
// 			Automatically update *skb*\ **->csum** after storing the
// 			bytes.
// 		**BPF_F_INVALIDATE_HASH**
// 			Set *skb*\ **->hash**, *skb*\ **->swhash** and *skb* * 			**->l4hash** to 0.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_l3_csum_replace(struct sk_buff *skb, u32 offset, u64 from, u64 to, u64 size)
// 	Description
// 		Recompute the layer 3 (e.g. IP) checksum for the packet
// 		associated to *skb*. Computation is incremental, so the helper
// 		must know the former value of the header field that was
// 		modified (*from*), the new value of this field (*to*), and the
// 		number of bytes (2 or 4) for this field, stored in *size*.
// 		Alternatively, it is possible to store the difference between
// 		the previous and the new values of the header field in *to*, by
// 		setting *from* and *size* to 0. For both methods, *offset*
// 		indicates the location of the IP checksum within the packet.
//
// 		This helper works in combination with **bpf_csum_diff**\ (),
// 		which does not update the checksum in-place, but offers more
// 		flexibility and can handle sizes larger than 2 or 4 for the
// 		checksum to update.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_l4_csum_replace(struct sk_buff *skb, u32 offset, u64 from, u64 to, u64 flags)
// 	Description
// 		Recompute the layer 4 (e.g. TCP, UDP or ICMP) checksum for the
// 		packet associated to *skb*. Computation is incremental, so the
// 		helper must know the former value of the header field that was
// 		modified (*from*), the new value of this field (*to*), and the
// 		number of bytes (2 or 4) for this field, stored on the lowest
// 		four bits of *flags*. Alternatively, it is possible to store
// 		the difference between the previous and the new values of the
// 		header field in *to*, by setting *from* and the four lowest
// 		bits of *flags* to 0. For both methods, *offset* indicates the
// 		location of the IP checksum within the packet. In addition to
// 		the size of the field, *flags* can be added (bitwise OR) actual
// 		flags. With **BPF_F_MARK_MANGLED_0**, a null checksum is left
// 		untouched (unless **BPF_F_MARK_ENFORCE** is added as well), and
// 		for updates resulting in a null checksum the value is set to
// 		**CSUM_MANGLED_0** instead. Flag **BPF_F_PSEUDO_HDR** indicates
// 		that the modified header field is part of the pseudo-header.
// 		Flag **BPF_F_IPV6** should be set for IPv6 packets.
//
// 		This helper works in combination with **bpf_csum_diff**\ (),
// 		which does not update the checksum in-place, but offers more
// 		flexibility and can handle sizes larger than 2 or 4 for the
// 		checksum to update.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_tail_call(void *ctx, struct bpf_map *prog_array_map, u32 index)
// 	Description
// 		This special helper is used to trigger a "tail call", or in
// 		other words, to jump into another eBPF program. The same stack
// 		frame is used (but values on stack and in registers for the
// 		caller are not accessible to the callee). This mechanism allows
// 		for program chaining, either for raising the maximum number of
// 		available eBPF instructions, or to execute given programs in
// 		conditional blocks. For security reasons, there is an upper
// 		limit to the number of successive tail calls that can be
// 		performed.
//
// 		Upon call of this helper, the program attempts to jump into a
// 		program referenced at index *index* in *prog_array_map*, a
// 		special map of type **BPF_MAP_TYPE_PROG_ARRAY**, and passes
// 		*ctx*, a pointer to the context.
//
// 		If the call succeeds, the kernel immediately runs the first
// 		instruction of the new program. This is not a function call,
// 		and it never returns to the previous program. If the call
// 		fails, then the helper has no effect, and the caller continues
// 		to run its subsequent instructions. A call can fail if the
// 		destination program for the jump does not exist (i.e. *index*
// 		is superior to the number of entries in *prog_array_map*), or
// 		if the maximum number of tail calls has been reached for this
// 		chain of programs. This limit is defined in the kernel by the
// 		macro **MAX_TAIL_CALL_CNT** (not accessible to user space),
// 		which is currently set to 33.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_clone_redirect(struct sk_buff *skb, u32 ifindex, u64 flags)
// 	Description
// 		Clone and redirect the packet associated to *skb* to another
// 		net device of index *ifindex*. Both ingress and egress
// 		interfaces can be used for redirection. The **BPF_F_INGRESS**
// 		value in *flags* is used to make the distinction (ingress path
// 		is selected if the flag is present, egress path otherwise).
// 		This is the only flag supported for now.
//
// 		In comparison with **bpf_redirect**\ () helper,
// 		**bpf_clone_redirect**\ () has the associated cost of
// 		duplicating the packet buffer, but this can be executed out of
// 		the eBPF program. Conversely, **bpf_redirect**\ () is more
// 		efficient, but it is handled through an action code where the
// 		redirection happens only after the eBPF program has returned.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure. Positive
// 		error indicates a potential drop or congestion in the target
// 		device. The particular positive error codes are not defined.
//
// u64 bpf_get_current_pid_tgid(void)
// 	Description
// 		Get the current pid and tgid.
// 	Return
// 		A 64-bit integer containing the current tgid and pid, and
// 		created as such:
// 		*current_task*\ **->tgid << 32 \|**
// 		*current_task*\ **->pid**.
//
// u64 bpf_get_current_uid_gid(void)
// 	Description
// 		Get the current uid and gid.
// 	Return
// 		A 64-bit integer containing the current GID and UID, and
// 		created as such: *current_gid* **<< 32 \|** *current_uid*.
//
// long bpf_get_current_comm(void *buf, u32 size_of_buf)
// 	Description
// 		Copy the **comm** attribute of the current task into *buf* of
// 		*size_of_buf*. The **comm** attribute contains the name of
// 		the executable (excluding the path) for the current task. The
// 		*size_of_buf* must be strictly positive. On success, the
// 		helper makes sure that the *buf* is NUL-terminated. On failure,
// 		it is filled with zeroes.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u32 bpf_get_cgroup_classid(struct sk_buff *skb)
// 	Description
// 		Retrieve the classid for the current task, i.e. for the net_cls
// 		cgroup to which *skb* belongs.
//
// 		This helper can be used on TC egress path, but not on ingress.
//
// 		The net_cls cgroup provides an interface to tag network packets
// 		based on a user-provided identifier for all traffic coming from
// 		the tasks belonging to the related cgroup. See also the related
// 		kernel documentation, available from the Linux sources in file
// 		*Documentation/admin-guide/cgroup-v1/net_cls.rst*.
//
// 		The Linux kernel has two versions for cgroups: there are
// 		cgroups v1 and cgroups v2. Both are available to users, who can
// 		use a mixture of them, but note that the net_cls cgroup is for
// 		cgroup v1 only. This makes it incompatible with BPF programs
// 		run on cgroups, which is a cgroup-v2-only feature (a socket can
// 		only hold data for one version of cgroups at a time).
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_CGROUP_NET_CLASSID** configuration option set to
// 		"**y**" or to "**m**".
// 	Return
// 		The classid, or 0 for the default unconfigured classid.
//
// long bpf_skb_vlan_push(struct sk_buff *skb, __be16 vlan_proto, u16 vlan_tci)
// 	Description
// 		Push a *vlan_tci* (VLAN tag control information) of protocol
// 		*vlan_proto* to the packet associated to *skb*, then update
// 		the checksum. Note that if *vlan_proto* is different from
// 		**ETH_P_8021Q** and **ETH_P_8021AD**, it is considered to
// 		be **ETH_P_8021Q**.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_vlan_pop(struct sk_buff *skb)
// 	Description
// 		Pop a VLAN header from the packet associated to *skb*.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_get_tunnel_key(struct sk_buff *skb, struct bpf_tunnel_key *key, u32 size, u64 flags)
// 	Description
// 		Get tunnel metadata. This helper takes a pointer *key* to an
// 		empty **struct bpf_tunnel_key** of **size**, that will be
// 		filled with tunnel metadata for the packet associated to *skb*.
// 		The *flags* can be set to **BPF_F_TUNINFO_IPV6**, which
// 		indicates that the tunnel is based on IPv6 protocol instead of
// 		IPv4.
//
// 		The **struct bpf_tunnel_key** is an object that generalizes the
// 		principal parameters used by various tunneling protocols into a
// 		single struct. This way, it can be used to easily make a
// 		decision based on the contents of the encapsulation header,
// 		"summarized" in this struct. In particular, it holds the IP
// 		address of the remote end (IPv4 or IPv6, depending on the case)
// 		in *key*\ **->remote_ipv4** or *key*\ **->remote_ipv6**. Also,
// 		this struct exposes the *key*\ **->tunnel_id**, which is
// 		generally mapped to a VNI (Virtual Network Identifier), making
// 		it programmable together with the **bpf_skb_set_tunnel_key** * 		() helper.
//
// 		Let's imagine that the following code is part of a program
// 		attached to the TC ingress interface, on one end of a GRE
// 		tunnel, and is supposed to filter out all messages coming from
// 		remote ends with IPv4 address other than 10.0.0.1:
//
// 		::
//
// 			int ret;
// 			struct bpf_tunnel_key key = {};
//
// 			ret = bpf_skb_get_tunnel_key(skb, &key, sizeof(key), 0);
// 			if (ret < 0)
// 				return TC_ACT_SHOT;	// drop packet
//
// 			if (key.remote_ipv4 != 0x0a000001)
// 				return TC_ACT_SHOT;	// drop packet
//
// 			return TC_ACT_OK;		// accept packet
//
// 		This interface can also be used with all encapsulation devices
// 		that can operate in "collect metadata" mode: instead of having
// 		one network device per specific configuration, the "collect
// 		metadata" mode only requires a single device where the
// 		configuration can be extracted from this helper.
//
// 		This can be used together with various tunnels such as VXLan,
// 		Geneve, GRE or IP in IP (IPIP).
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_set_tunnel_key(struct sk_buff *skb, struct bpf_tunnel_key *key, u32 size, u64 flags)
// 	Description
// 		Populate tunnel metadata for packet associated to *skb.* The
// 		tunnel metadata is set to the contents of *key*, of *size*. The
// 		*flags* can be set to a combination of the following values:
//
// 		**BPF_F_TUNINFO_IPV6**
// 			Indicate that the tunnel is based on IPv6 protocol
// 			instead of IPv4.
// 		**BPF_F_ZERO_CSUM_TX**
// 			For IPv4 packets, add a flag to tunnel metadata
// 			indicating that checksum computation should be skipped
// 			and checksum set to zeroes.
// 		**BPF_F_DONT_FRAGMENT**
// 			Add a flag to tunnel metadata indicating that the
// 			packet should not be fragmented.
// 		**BPF_F_SEQ_NUMBER**
// 			Add a flag to tunnel metadata indicating that a
// 			sequence number should be added to tunnel header before
// 			sending the packet. This flag was added for GRE
// 			encapsulation, but might be used with other protocols
// 			as well in the future.
// 		**BPF_F_NO_TUNNEL_KEY**
// 			Add a flag to tunnel metadata indicating that no tunnel
// 			key should be set in the resulting tunnel header.
//
// 		Here is a typical usage on the transmit path:
//
// 		::
//
// 			struct bpf_tunnel_key key;
// 			     populate key ...
// 			bpf_skb_set_tunnel_key(skb, &key, sizeof(key), 0);
// 			bpf_clone_redirect(skb, vxlan_dev_ifindex, 0);
//
// 		See also the description of the **bpf_skb_get_tunnel_key**\ ()
// 		helper for additional information.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_perf_event_read(struct bpf_map *map, u64 flags)
// 	Description
// 		Read the value of a perf event counter. This helper relies on a
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. The nature of
// 		the perf event counter is selected when *map* is updated with
// 		perf event file descriptors. The *map* is an array whose size
// 		is the number of available CPUs, and each cell contains a value
// 		relative to one CPU. The value to retrieve is indicated by
// 		*flags*, that contains the index of the CPU to look up, masked
// 		with **BPF_F_INDEX_MASK**. Alternatively, *flags* can be set to
// 		**BPF_F_CURRENT_CPU** to indicate that the value for the
// 		current CPU should be retrieved.
//
// 		Note that before Linux 4.13, only hardware perf event can be
// 		retrieved.
//
// 		Also, be aware that the newer helper
// 		**bpf_perf_event_read_value**\ () is recommended over
// 		**bpf_perf_event_read**\ () in general. The latter has some ABI
// 		quirks where error and counter value are used as a return code
// 		(which is wrong to do since ranges may overlap). This issue is
// 		fixed with **bpf_perf_event_read_value**\ (), which at the same
// 		time provides more features over the **bpf_perf_event_read** * 		() interface. Please refer to the description of
// 		**bpf_perf_event_read_value**\ () for details.
// 	Return
// 		The value of the perf event counter read from the map, or a
// 		negative error code in case of failure.
//
// long bpf_redirect(u32 ifindex, u64 flags)
// 	Description
// 		Redirect the packet to another net device of index *ifindex*.
// 		This helper is somewhat similar to **bpf_clone_redirect** * 		(), except that the packet is not cloned, which provides
// 		increased performance.
//
// 		Except for XDP, both ingress and egress interfaces can be used
// 		for redirection. The **BPF_F_INGRESS** value in *flags* is used
// 		to make the distinction (ingress path is selected if the flag
// 		is present, egress path otherwise). Currently, XDP only
// 		supports redirection to the egress interface, and accepts no
// 		flag at all.
//
// 		The same effect can also be attained with the more generic
// 		**bpf_redirect_map**\ (), which uses a BPF map to store the
// 		redirect target instead of providing it directly to the helper.
// 	Return
// 		For XDP, the helper returns **XDP_REDIRECT** on success or
// 		**XDP_ABORTED** on error. For other program types, the values
// 		are **TC_ACT_REDIRECT** on success or **TC_ACT_SHOT** on
// 		error.
//
// u32 bpf_get_route_realm(struct sk_buff *skb)
// 	Description
// 		Retrieve the realm or the route, that is to say the
// 		**tclassid** field of the destination for the *skb*. The
// 		identifier retrieved is a user-provided tag, similar to the
// 		one used with the net_cls cgroup (see description for
// 		**bpf_get_cgroup_classid**\ () helper), but here this tag is
// 		held by a route (a destination entry), not by a task.
//
// 		Retrieving this identifier works with the clsact TC egress hook
// 		(see also **tc-bpf(8)**), or alternatively on conventional
// 		classful egress qdiscs, but not on TC ingress path. In case of
// 		clsact TC egress hook, this has the advantage that, internally,
// 		the destination entry has not been dropped yet in the transmit
// 		path. Therefore, the destination entry does not need to be
// 		artificially held via **netif_keep_dst**\ () for a classful
// 		qdisc until the *skb* is freed.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_IP_ROUTE_CLASSID** configuration option.
// 	Return
// 		The realm of the route for the packet associated to *skb*, or 0
// 		if none was found.
//
// long bpf_perf_event_output(void *ctx, struct bpf_map *map, u64 flags, void *data, u64 size)
// 	Description
// 		Write raw *data* blob into a special BPF perf event held by
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. This perf
// 		event must have the following attributes: **PERF_SAMPLE_RAW**
// 		as **sample_type**, **PERF_TYPE_SOFTWARE** as **type**, and
// 		**PERF_COUNT_SW_BPF_OUTPUT** as **config**.
//
// 		The *flags* are used to indicate the index in *map* for which
// 		the value must be put, masked with **BPF_F_INDEX_MASK**.
// 		Alternatively, *flags* can be set to **BPF_F_CURRENT_CPU**
// 		to indicate that the index of the current CPU core should be
// 		used.
//
// 		The value to write, of *size*, is passed through eBPF stack and
// 		pointed by *data*.
//
// 		The context of the program *ctx* needs also be passed to the
// 		helper.
//
// 		On user space, a program willing to read the values needs to
// 		call **perf_event_open**\ () on the perf event (either for
// 		one or for all CPUs) and to store the file descriptor into the
// 		*map*. This must be done before the eBPF program can send data
// 		into it. An example is available in file
// 		*samples/bpf/trace_output_user.c* in the Linux kernel source
// 		tree (the eBPF program counterpart is in
// 		*samples/bpf/trace_output.bpf.c*).
//
// 		**bpf_perf_event_output**\ () achieves better performance
// 		than **bpf_trace_printk**\ () for sharing data with user
// 		space, and is much better suitable for streaming data from eBPF
// 		programs.
//
// 		Note that this helper is not restricted to tracing use cases
// 		and can be used with programs attached to TC or XDP as well,
// 		where it allows for passing data to user space listeners. Data
// 		can be:
//
// 		* Only custom structs,
// 		* Only the packet payload, or
// 		* A combination of both.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_load_bytes(const void *skb, u32 offset, void *to, u32 len)
// 	Description
// 		This helper was provided as an easy way to load data from a
// 		packet. It can be used to load *len* bytes from *offset* from
// 		the packet associated to *skb*, into the buffer pointed by
// 		*to*.
//
// 		Since Linux 4.7, usage of this helper has mostly been replaced
// 		by "direct packet access", enabling packet data to be
// 		manipulated with *skb*\ **->data** and *skb*\ **->data_end**
// 		pointing respectively to the first byte of packet data and to
// 		the byte after the last byte of packet data. However, it
// 		remains useful if one wishes to read large quantities of data
// 		at once from a packet into the eBPF stack.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_get_stackid(void *ctx, struct bpf_map *map, u64 flags)
// 	Description
// 		Walk a user or a kernel stack and return its id. To achieve
// 		this, the helper needs *ctx*, which is a pointer to the context
// 		on which the tracing program is executed, and a pointer to a
// 		*map* of type **BPF_MAP_TYPE_STACK_TRACE**.
//
// 		The last argument, *flags*, holds the number of stack frames to
// 		skip (from 0 to 255), masked with
// 		**BPF_F_SKIP_FIELD_MASK**. The next bits can be used to set
// 		a combination of the following flags:
//
// 		**BPF_F_USER_STACK**
// 			Collect a user space stack instead of a kernel stack.
// 		**BPF_F_FAST_STACK_CMP**
// 			Compare stacks by hash only.
// 		**BPF_F_REUSE_STACKID**
// 			If two different stacks hash into the same *stackid*,
// 			discard the old one.
//
// 		The stack id retrieved is a 32 bit long integer handle which
// 		can be further combined with other data (including other stack
// 		ids) and used as a key into maps. This can be useful for
// 		generating a variety of graphs (such as flame graphs or off-cpu
// 		graphs).
//
// 		For walking a stack, this helper is an improvement over
// 		**bpf_probe_read**\ (), which can be used with unrolled loops
// 		but is not efficient and consumes a lot of eBPF instructions.
// 		Instead, **bpf_get_stackid**\ () can collect up to
// 		**PERF_MAX_STACK_DEPTH** both kernel and user frames. Note that
// 		this limit can be controlled with the **sysctl** program, and
// 		that it should be manually increased in order to profile long
// 		user stacks (such as stacks for Java programs). To do so, use:
//
// 		::
//
// 			# sysctl kernel.perf_event_max_stack=<new value>
// 	Return
// 		The positive or null stack id on success, or a negative error
// 		in case of failure.
//
// s64 bpf_csum_diff(__be32 *from, u32 from_size, __be32 *to, u32 to_size, __wsum seed)
// 	Description
// 		Compute a checksum difference, from the raw buffer pointed by
// 		*from*, of length *from_size* (that must be a multiple of 4),
// 		towards the raw buffer pointed by *to*, of size *to_size*
// 		(same remark). An optional *seed* can be added to the value
// 		(this can be cascaded, the seed may come from a previous call
// 		to the helper).
//
// 		This is flexible enough to be used in several ways:
//
// 		* With *from_size* == 0, *to_size* > 0 and *seed* set to
// 		  checksum, it can be used when pushing new data.
// 		* With *from_size* > 0, *to_size* == 0 and *seed* set to
// 		  checksum, it can be used when removing data from a packet.
// 		* With *from_size* > 0, *to_size* > 0 and *seed* set to 0, it
// 		  can be used to compute a diff. Note that *from_size* and
// 		  *to_size* do not need to be equal.
//
// 		This helper can be used in combination with
// 		**bpf_l3_csum_replace**\ () and **bpf_l4_csum_replace**\ (), to
// 		which one can feed in the difference computed with
// 		**bpf_csum_diff**\ ().
// 	Return
// 		The checksum result, or a negative error code in case of
// 		failure.
//
// long bpf_skb_get_tunnel_opt(struct sk_buff *skb, void *opt, u32 size)
// 	Description
// 		Retrieve tunnel options metadata for the packet associated to
// 		*skb*, and store the raw tunnel option data to the buffer *opt*
// 		of *size*.
//
// 		This helper can be used with encapsulation devices that can
// 		operate in "collect metadata" mode (please refer to the related
// 		note in the description of **bpf_skb_get_tunnel_key**\ () for
// 		more details). A particular example where this can be used is
// 		in combination with the Geneve encapsulation protocol, where it
// 		allows for pushing (with **bpf_skb_get_tunnel_opt**\ () helper)
// 		and retrieving arbitrary TLVs (Type-Length-Value headers) from
// 		the eBPF program. This allows for full customization of these
// 		headers.
// 	Return
// 		The size of the option data retrieved.
//
// long bpf_skb_set_tunnel_opt(struct sk_buff *skb, void *opt, u32 size)
// 	Description
// 		Set tunnel options metadata for the packet associated to *skb*
// 		to the option data contained in the raw buffer *opt* of *size*.
//
// 		See also the description of the **bpf_skb_get_tunnel_opt**\ ()
// 		helper for additional information.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_change_proto(struct sk_buff *skb, __be16 proto, u64 flags)
// 	Description
// 		Change the protocol of the *skb* to *proto*. Currently
// 		supported are transition from IPv4 to IPv6, and from IPv6 to
// 		IPv4. The helper takes care of the groundwork for the
// 		transition, including resizing the socket buffer. The eBPF
// 		program is expected to fill the new headers, if any, via
// 		**skb_store_bytes**\ () and to recompute the checksums with
// 		**bpf_l3_csum_replace**\ () and **bpf_l4_csum_replace** * 		(). The main case for this helper is to perform NAT64
// 		operations out of an eBPF program.
//
// 		Internally, the GSO type is marked as dodgy so that headers are
// 		checked and segments are recalculated by the GSO/GRO engine.
// 		The size for GSO target is adapted as well.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_change_type(struct sk_buff *skb, u32 type)
// 	Description
// 		Change the packet type for the packet associated to *skb*. This
// 		comes down to setting *skb*\ **->pkt_type** to *type*, except
// 		the eBPF program does not have a write access to *skb* * 		**->pkt_type** beside this helper. Using a helper here allows
// 		for graceful handling of errors.
//
// 		The major use case is to change incoming *skb*s to
// 		**PACKET_HOST** in a programmatic way instead of having to
// 		recirculate via **redirect**\ (..., **BPF_F_INGRESS**), for
// 		example.
//
// 		Note that *type* only allows certain values. At this time, they
// 		are:
//
// 		**PACKET_HOST**
// 			Packet is for us.
// 		**PACKET_BROADCAST**
// 			Send packet to all.
// 		**PACKET_MULTICAST**
// 			Send packet to group.
// 		**PACKET_OTHERHOST**
// 			Send packet to someone else.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_under_cgroup(struct sk_buff *skb, struct bpf_map *map, u32 index)
// 	Description
// 		Check whether *skb* is a descendant of the cgroup2 held by
// 		*map* of type **BPF_MAP_TYPE_CGROUP_ARRAY**, at *index*.
// 	Return
// 		The return value depends on the result of the test, and can be:
//
// 		* 0, if the *skb* failed the cgroup2 descendant test.
// 		* 1, if the *skb* succeeded the cgroup2 descendant test.
// 		* A negative error code, if an error occurred.
//
// u32 bpf_get_hash_recalc(struct sk_buff *skb)
// 	Description
// 		Retrieve the hash of the packet, *skb*\ **->hash**. If it is
// 		not set, in particular if the hash was cleared due to mangling,
// 		recompute this hash. Later accesses to the hash can be done
// 		directly with *skb*\ **->hash**.
//
// 		Calling **bpf_set_hash_invalid**\ (), changing a packet
// 		prototype with **bpf_skb_change_proto**\ (), or calling
// 		**bpf_skb_store_bytes**\ () with the
// 		**BPF_F_INVALIDATE_HASH** are actions susceptible to clear
// 		the hash and to trigger a new computation for the next call to
// 		**bpf_get_hash_recalc**\ ().
// 	Return
// 		The 32-bit hash.
//
// u64 bpf_get_current_task(void)
// 	Description
// 		Get the current task.
// 	Return
// 		A pointer to the current task struct.
//
// long bpf_probe_write_user(void *dst, const void *src, u32 len)
// 	Description
// 		Attempt in a safe way to write *len* bytes from the buffer
// 		*src* to *dst* in memory. It only works for threads that are in
// 		user context, and *dst* must be a valid user space address.
//
// 		This helper should not be used to implement any kind of
// 		security mechanism because of TOC-TOU attacks, but rather to
// 		debug, divert, and manipulate execution of semi-cooperative
// 		processes.
//
// 		Keep in mind that this feature is meant for experiments, and it
// 		has a risk of crashing the system and running programs.
// 		Therefore, when an eBPF program using this helper is attached,
// 		a warning including PID and process name is printed to kernel
// 		logs.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_current_task_under_cgroup(struct bpf_map *map, u32 index)
// 	Description
// 		Check whether the probe is being run is the context of a given
// 		subset of the cgroup2 hierarchy. The cgroup2 to test is held by
// 		*map* of type **BPF_MAP_TYPE_CGROUP_ARRAY**, at *index*.
// 	Return
// 		The return value depends on the result of the test, and can be:
//
// 		* 1, if current task belongs to the cgroup2.
// 		* 0, if current task does not belong to the cgroup2.
// 		* A negative error code, if an error occurred.
//
// long bpf_skb_change_tail(struct sk_buff *skb, u32 len, u64 flags)
// 	Description
// 		Resize (trim or grow) the packet associated to *skb* to the
// 		new *len*. The *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		The basic idea is that the helper performs the needed work to
// 		change the size of the packet, then the eBPF program rewrites
// 		the rest via helpers like **bpf_skb_store_bytes**\ (),
// 		**bpf_l3_csum_replace**\ (), **bpf_l3_csum_replace**\ ()
// 		and others. This helper is a slow path utility intended for
// 		replies with control messages. And because it is targeted for
// 		slow path, the helper itself can afford to be slow: it
// 		implicitly linearizes, unclones and drops offloads from the
// 		*skb*.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_pull_data(struct sk_buff *skb, u32 len)
// 	Description
// 		Pull in non-linear data in case the *skb* is non-linear and not
// 		all of *len* are part of the linear section. Make *len* bytes
// 		from *skb* readable and writable. If a zero value is passed for
// 		*len*, then all bytes in the linear part of *skb* will be made
// 		readable and writable.
//
// 		This helper is only needed for reading and writing with direct
// 		packet access.
//
// 		For direct packet access, testing that offsets to access
// 		are within packet boundaries (test on *skb*\ **->data_end**) is
// 		susceptible to fail if offsets are invalid, or if the requested
// 		data is in non-linear parts of the *skb*. On failure the
// 		program can just bail out, or in the case of a non-linear
// 		buffer, use a helper to make the data available. The
// 		**bpf_skb_load_bytes**\ () helper is a first solution to access
// 		the data. Another one consists in using **bpf_skb_pull_data**
// 		to pull in once the non-linear parts, then retesting and
// 		eventually access the data.
//
// 		At the same time, this also makes sure the *skb* is uncloned,
// 		which is a necessary condition for direct write. As this needs
// 		to be an invariant for the write part only, the verifier
// 		detects writes and adds a prologue that is calling
// 		**bpf_skb_pull_data()** to effectively unclone the *skb* from
// 		the very beginning in case it is indeed cloned.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// s64 bpf_csum_update(struct sk_buff *skb, __wsum csum)
// 	Description
// 		Add the checksum *csum* into *skb*\ **->csum** in case the
// 		driver has supplied a checksum for the entire packet into that
// 		field. Return an error otherwise. This helper is intended to be
// 		used in combination with **bpf_csum_diff**\ (), in particular
// 		when the checksum needs to be updated after data has been
// 		written into the packet through direct packet access.
// 	Return
// 		The checksum on success, or a negative error code in case of
// 		failure.
//
// void bpf_set_hash_invalid(struct sk_buff *skb)
// 	Description
// 		Invalidate the current *skb*\ **->hash**. It can be used after
// 		mangling on headers through direct packet access, in order to
// 		indicate that the hash is outdated and to trigger a
// 		recalculation the next time the kernel tries to access this
// 		hash or when the **bpf_get_hash_recalc**\ () helper is called.
// 	Return
// 		void.
//
// long bpf_get_numa_node_id(void)
// 	Description
// 		Return the id of the current NUMA node. The primary use case
// 		for this helper is the selection of sockets for the local NUMA
// 		node, when the program is attached to sockets using the
// 		**SO_ATTACH_REUSEPORT_EBPF** option (see also **socket(7)**),
// 		but the helper is also available to other eBPF program types,
// 		similarly to **bpf_get_smp_processor_id**\ ().
// 	Return
// 		The id of current NUMA node.
//
// long bpf_skb_change_head(struct sk_buff *skb, u32 len, u64 flags)
// 	Description
// 		Grows headroom of packet associated to *skb* and adjusts the
// 		offset of the MAC header accordingly, adding *len* bytes of
// 		space. It automatically extends and reallocates memory as
// 		required.
//
// 		This helper can be used on a layer 3 *skb* to push a MAC header
// 		for redirection into a layer 2 device.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_adjust_head(struct xdp_buff *xdp_md, int delta)
// 	Description
// 		Adjust (move) *xdp_md*\ **->data** by *delta* bytes. Note that
// 		it is possible to use a negative value for *delta*. This helper
// 		can be used to prepare the packet for pushing or popping
// 		headers.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_str(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Copy a NUL terminated string from an unsafe kernel address
// 		*unsafe_ptr* to *dst*. See **bpf_probe_read_kernel_str**\ () for
// 		more details.
//
// 		Generally, use **bpf_probe_read_user_str**\ () or
// 		**bpf_probe_read_kernel_str**\ () instead.
// 	Return
// 		On success, the strictly positive length of the string,
// 		including the trailing NUL character. On error, a negative
// 		value.
//
// u64 bpf_get_socket_cookie(struct sk_buff *skb)
// 	Description
// 		If the **struct sk_buff** pointed by *skb* has a known socket,
// 		retrieve the cookie (generated by the kernel) of this socket.
// 		If no cookie has been set yet, generate a new cookie. Once
// 		generated, the socket cookie remains stable for the life of the
// 		socket. This helper can be useful for monitoring per socket
// 		networking traffic statistics as it provides a global socket
// 		identifier that can be assumed unique.
// 	Return
// 		A 8-byte long unique number on success, or 0 if the socket
// 		field is missing inside *skb*.
//
// u64 bpf_get_socket_cookie(struct bpf_sock_addr *ctx)
// 	Description
// 		Equivalent to bpf_get_socket_cookie() helper that accepts
// 		*skb*, but gets socket from **struct bpf_sock_addr** context.
// 	Return
// 		A 8-byte long unique number.
//
// u64 bpf_get_socket_cookie(struct bpf_sock_ops *ctx)
// 	Description
// 		Equivalent to **bpf_get_socket_cookie**\ () helper that accepts
// 		*skb*, but gets socket from **struct bpf_sock_ops** context.
// 	Return
// 		A 8-byte long unique number.
//
// u64 bpf_get_socket_cookie(struct sock *sk)
// 	Description
// 		Equivalent to **bpf_get_socket_cookie**\ () helper that accepts
// 		*sk*, but gets socket from a BTF **struct sock**. This helper
// 		also works for sleepable programs.
// 	Return
// 		A 8-byte long unique number or 0 if *sk* is NULL.
//
// u32 bpf_get_socket_uid(struct sk_buff *skb)
// 	Description
// 		Get the owner UID of the socked associated to *skb*.
// 	Return
// 		The owner UID of the socket associated to *skb*. If the socket
// 		is **NULL**, or if it is not a full socket (i.e. if it is a
// 		time-wait or a request socket instead), **overflowuid** value
// 		is returned (note that **overflowuid** might also be the actual
// 		UID value for the socket).
//
// long bpf_set_hash(struct sk_buff *skb, u32 hash)
// 	Description
// 		Set the full hash for *skb* (set the field *skb*\ **->hash**)
// 		to value *hash*.
// 	Return
// 		0
//
// long bpf_setsockopt(void *bpf_socket, int level, int optname, void *optval, int optlen)
// 	Description
// 		Emulate a call to **setsockopt()** on the socket associated to
// 		*bpf_socket*, which must be a full socket. The *level* at
// 		which the option resides and the name *optname* of the option
// 		must be specified, see **setsockopt(2)** for more information.
// 		The option value of length *optlen* is pointed by *optval*.
//
// 		*bpf_socket* should be one of the following:
//
// 		* **struct bpf_sock_ops** for **BPF_PROG_TYPE_SOCK_OPS**.
// 		* **struct bpf_sock_addr** for **BPF_CGROUP_INET4_CONNECT**,
// 		  **BPF_CGROUP_INET6_CONNECT** and **BPF_CGROUP_UNIX_CONNECT**.
//
// 		This helper actually implements a subset of **setsockopt()**.
// 		It supports the following *level*\ s:
//
// 		* **SOL_SOCKET**, which supports the following *optname*\ s:
// 		  **SO_RCVBUF**, **SO_SNDBUF**, **SO_MAX_PACING_RATE**,
// 		  **SO_PRIORITY**, **SO_RCVLOWAT**, **SO_MARK**,
// 		  **SO_BINDTODEVICE**, **SO_KEEPALIVE**, **SO_REUSEADDR**,
// 		  **SO_REUSEPORT**, **SO_BINDTOIFINDEX**, **SO_TXREHASH**.
// 		* **IPPROTO_TCP**, which supports the following *optname*\ s:
// 		  **TCP_CONGESTION**, **TCP_BPF_IW**,
// 		  **TCP_BPF_SNDCWND_CLAMP**, **TCP_SAVE_SYN**,
// 		  **TCP_KEEPIDLE**, **TCP_KEEPINTVL**, **TCP_KEEPCNT**,
// 		  **TCP_SYNCNT**, **TCP_USER_TIMEOUT**, **TCP_NOTSENT_LOWAT**,
// 		  **TCP_NODELAY**, **TCP_MAXSEG**, **TCP_WINDOW_CLAMP**,
// 		  **TCP_THIN_LINEAR_TIMEOUTS**, **TCP_BPF_DELACK_MAX**,
// 		  **TCP_BPF_RTO_MIN**, **TCP_BPF_SOCK_OPS_CB_FLAGS**.
// 		* **IPPROTO_IP**, which supports *optname* **IP_TOS**.
// 		* **IPPROTO_IPV6**, which supports the following *optname*\ s:
// 		  **IPV6_TCLASS**, **IPV6_AUTOFLOWLABEL**.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_adjust_room(struct sk_buff *skb, s32 len_diff, u32 mode, u64 flags)
// 	Description
// 		Grow or shrink the room for data in the packet associated to
// 		*skb* by *len_diff*, and according to the selected *mode*.
//
// 		By default, the helper will reset any offloaded checksum
// 		indicator of the skb to CHECKSUM_NONE. This can be avoided
// 		by the following flag:
//
// 		* **BPF_F_ADJ_ROOM_NO_CSUM_RESET**: Do not reset offloaded
// 		  checksum data of the skb to CHECKSUM_NONE.
//
// 		There are two supported modes at this time:
//
// 		* **BPF_ADJ_ROOM_MAC**: Adjust room at the mac layer
// 		  (room space is added or removed between the layer 2 and
// 		  layer 3 headers).
//
// 		* **BPF_ADJ_ROOM_NET**: Adjust room at the network layer
// 		  (room space is added or removed between the layer 3 and
// 		  layer 4 headers).
//
// 		The following flags are supported at this time:
//
// 		* **BPF_F_ADJ_ROOM_FIXED_GSO**: Do not adjust gso_size.
// 		  Adjusting mss in this way is not allowed for datagrams.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L3_IPV4**,
// 		  **BPF_F_ADJ_ROOM_ENCAP_L3_IPV6**:
// 		  Any new space is reserved to hold a tunnel header.
// 		  Configure skb offsets and other fields accordingly.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L4_GRE**,
// 		  **BPF_F_ADJ_ROOM_ENCAP_L4_UDP**:
// 		  Use with ENCAP_L3 flags to further specify the tunnel type.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L2**\ (*len*):
// 		  Use with ENCAP_L3/L4 flags to further specify the tunnel
// 		  type; *len* is the length of the inner MAC header.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L2_ETH**:
// 		  Use with BPF_F_ADJ_ROOM_ENCAP_L2 flag to further specify the
// 		  L2 type as Ethernet.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_L3_IPV4**,
// 		  **BPF_F_ADJ_ROOM_DECAP_L3_IPV6**:
// 		  Indicate the new IP header version after decapsulating the
// 		  outer IP header. Used when the inner and outer IP versions
// 		  are different. These flags only trigger a protocol change
// 		  without clearing any tunnel-specific GSO flags.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_L4_GRE**:
// 		  Clear GRE tunnel GSO flags (SKB_GSO_GRE and SKB_GSO_GRE_CSUM)
// 		  when decapsulating a GRE tunnel.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_L4_UDP**:
// 		  Clear UDP tunnel GSO flags (SKB_GSO_UDP_TUNNEL and
// 		  SKB_GSO_UDP_TUNNEL_CSUM) when decapsulating a UDP tunnel.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_IPXIP4**:
// 		  Clear IPIP/SIT tunnel GSO flag (SKB_GSO_IPXIP4) when decapsulating
// 		  a tunnel with an outer IPv4 header (IPv4-in-IPv4 or IPv6-in-IPv4).
//
// 		* **BPF_F_ADJ_ROOM_DECAP_IPXIP6**:
// 		  Clear IPv6 encapsulation tunnel GSO flag (SKB_GSO_IPXIP6) when
// 		  decapsulating a tunnel with an outer IPv6 header (IPv6-in-IPv6
// 		  or IPv4-in-IPv6).
//
// 		When using the decapsulation flags above, the skb->encapsulation
// 		flag is automatically cleared if all tunnel-specific GSO flags
// 		(SKB_GSO_UDP_TUNNEL, SKB_GSO_UDP_TUNNEL_CSUM, SKB_GSO_GRE,
// 		SKB_GSO_GRE_CSUM, SKB_GSO_IPXIP4, SKB_GSO_IPXIP6) have been
// 		removed from the packet. This handles cases where all tunnel
// 		layers have been decapsulated.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_redirect_map(struct bpf_map *map, u64 key, u64 flags)
// 	Description
// 		Redirect the packet to the endpoint referenced by *map* at
// 		index *key*. Depending on its type, this *map* can contain
// 		references to net devices (for forwarding packets through other
// 		ports), or to CPUs (for redirecting XDP frames to another CPU;
// 		but this is only implemented for native XDP (with driver
// 		support) as of this writing).
//
// 		The lower two bits of *flags* are used as the return code if
// 		the map lookup fails. This is so that the return value can be
// 		one of the XDP program return codes up to **XDP_TX**, as chosen
// 		by the caller. The higher bits of *flags* can be set to
// 		BPF_F_BROADCAST or BPF_F_EXCLUDE_INGRESS as defined below.
//
// 		With BPF_F_BROADCAST the packet will be broadcasted to all the
// 		interfaces in the map, with BPF_F_EXCLUDE_INGRESS the ingress
// 		interface will be excluded when do broadcasting.
//
// 		See also **bpf_redirect**\ (), which only supports redirecting
// 		to an ifindex, but doesn't require a map to do so.
// 	Return
// 		**XDP_REDIRECT** on success, or the value of the two lower bits
// 		of the *flags* argument on error.
//
// long bpf_sk_redirect_map(struct sk_buff *skb, struct bpf_map *map, u32 key, u64 flags)
// 	Description
// 		Redirect the packet to the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKMAP**) at index *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress path otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_sock_map_update(struct bpf_sock_ops *skops, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		Add an entry to, or update a *map* referencing sockets. The
// 		*skops* is used as a new value for the entry associated to
// 		*key*. *flags* is one of:
//
// 		**BPF_NOEXIST**
// 			The entry for *key* must not exist in the map.
// 		**BPF_EXIST**
// 			The entry for *key* must already exist in the map.
// 		**BPF_ANY**
// 			No condition on the existence of the entry for *key*.
//
// 		If the *map* has eBPF programs (parser and verdict), those will
// 		be inherited by the socket being added. If the socket is
// 		already attached to eBPF programs, this results in an error.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_adjust_meta(struct xdp_buff *xdp_md, int delta)
// 	Description
// 		Adjust the address pointed by *xdp_md*\ **->data_meta** by
// 		*delta* (which can be positive or negative). Note that this
// 		operation modifies the address stored in *xdp_md*\ **->data**,
// 		so the latter must be loaded only after the helper has been
// 		called.
//
// 		The use of *xdp_md*\ **->data_meta** is optional and programs
// 		are not required to use it. The rationale is that when the
// 		packet is processed with XDP (e.g. as DoS filter), it is
// 		possible to push further meta data along with it before passing
// 		to the stack, and to give the guarantee that an ingress eBPF
// 		program attached as a TC classifier on the same device can pick
// 		this up for further post-processing. Since TC works with socket
// 		buffers, it remains possible to set from XDP the **mark** or
// 		**priority** pointers, or other pointers for the socket buffer.
// 		Having this scratch space generic and programmable allows for
// 		more flexibility as the user is free to store whatever meta
// 		data they need.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_perf_event_read_value(struct bpf_map *map, u64 flags, struct bpf_perf_event_value *buf, u32 buf_size)
// 	Description
// 		Read the value of a perf event counter, and store it into *buf*
// 		of size *buf_size*. This helper relies on a *map* of type
// 		**BPF_MAP_TYPE_PERF_EVENT_ARRAY**. The nature of the perf event
// 		counter is selected when *map* is updated with perf event file
// 		descriptors. The *map* is an array whose size is the number of
// 		available CPUs, and each cell contains a value relative to one
// 		CPU. The value to retrieve is indicated by *flags*, that
// 		contains the index of the CPU to look up, masked with
// 		**BPF_F_INDEX_MASK**. Alternatively, *flags* can be set to
// 		**BPF_F_CURRENT_CPU** to indicate that the value for the
// 		current CPU should be retrieved.
//
// 		This helper behaves in a way close to
// 		**bpf_perf_event_read**\ () helper, save that instead of
// 		just returning the value observed, it fills the *buf*
// 		structure. This allows for additional data to be retrieved: in
// 		particular, the enabled and running times (in *buf* * 		**->enabled** and *buf*\ **->running**, respectively) are
// 		copied. In general, **bpf_perf_event_read_value**\ () is
// 		recommended over **bpf_perf_event_read**\ (), which has some
// 		ABI issues and provides fewer functionalities.
//
// 		These values are interesting, because hardware PMU (Performance
// 		Monitoring Unit) counters are limited resources. When there are
// 		more PMU based perf events opened than available counters,
// 		kernel will multiplex these events so each event gets certain
// 		percentage (but not all) of the PMU time. In case that
// 		multiplexing happens, the number of samples or counter value
// 		will not reflect the case compared to when no multiplexing
// 		occurs. This makes comparison between different runs difficult.
// 		Typically, the counter value should be normalized before
// 		comparing to other experiments. The usual normalization is done
// 		as follows.
//
// 		::
//
// 			normalized_counter = counter * t_enabled / t_running
//
// 		Where t_enabled is the time enabled for event and t_running is
// 		the time running for event since last normalization. The
// 		enabled and running times are accumulated since the perf event
// 		open. To achieve scaling factor between two invocations of an
// 		eBPF program, users can use CPU id as the key (which is
// 		typical for perf array usage model) to remember the previous
// 		value and do the calculation inside the eBPF program.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_perf_prog_read_value(struct bpf_perf_event_data *ctx, struct bpf_perf_event_value *buf, u32 buf_size)
// 	Description
// 		For an eBPF program attached to a perf event, retrieve the
// 		value of the event counter associated to *ctx* and store it in
// 		the structure pointed by *buf* and of size *buf_size*. Enabled
// 		and running times are also stored in the structure (see
// 		description of helper **bpf_perf_event_read_value**\ () for
// 		more details).
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_getsockopt(void *bpf_socket, int level, int optname, void *optval, int optlen)
// 	Description
// 		Emulate a call to **getsockopt()** on the socket associated to
// 		*bpf_socket*, which must be a full socket. The *level* at
// 		which the option resides and the name *optname* of the option
// 		must be specified, see **getsockopt(2)** for more information.
// 		The retrieved value is stored in the structure pointed by
// 		*opval* and of length *optlen*.
//
// 		*bpf_socket* should be one of the following:
//
// 		* **struct bpf_sock_ops** for **BPF_PROG_TYPE_SOCK_OPS**.
// 		* **struct bpf_sock_addr** for **BPF_CGROUP_INET4_CONNECT**,
// 		  **BPF_CGROUP_INET6_CONNECT** and **BPF_CGROUP_UNIX_CONNECT**.
//
// 		This helper actually implements a subset of **getsockopt()**.
// 		It supports the same set of *optname*\ s that is supported by
// 		the **bpf_setsockopt**\ () helper.  The exceptions are
// 		**TCP_BPF_*** is **bpf_setsockopt**\ () only and
// 		**TCP_SAVED_SYN** is **bpf_getsockopt**\ () only.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_override_return(struct pt_regs *regs, u64 rc)
// 	Description
// 		Used for error injection, this helper uses kprobes to override
// 		the return value of the probed function, and to set it to *rc*.
// 		The first argument is the context *regs* on which the kprobe
// 		works.
//
// 		This helper works by setting the PC (program counter)
// 		to an override function which is run in place of the original
// 		probed function. This means the probed function is not run at
// 		all. The replacement function just returns with the required
// 		value.
//
// 		This helper has security implications, and thus is subject to
// 		restrictions. It is only available if the kernel was compiled
// 		with the **CONFIG_BPF_KPROBE_OVERRIDE** configuration
// 		option, and in this case it only works on functions tagged with
// 		**ALLOW_ERROR_INJECTION** in the kernel code.
// 	Return
// 		0
//
// long bpf_sock_ops_cb_flags_set(struct bpf_sock_ops *bpf_sock, int argval)
// 	Description
// 		Attempt to set the value of the **bpf_sock_ops_cb_flags** field
// 		for the full TCP socket associated to *bpf_sock_ops* to
// 		*argval*.
//
// 		The primary use of this field is to determine if there should
// 		be calls to eBPF programs of type
// 		**BPF_PROG_TYPE_SOCK_OPS** at various points in the TCP
// 		code. A program of the same type can change its value, per
// 		connection and as necessary, when the connection is
// 		established. This field is directly accessible for reading, but
// 		this helper must be used for updates in order to return an
// 		error if an eBPF program tries to set a callback that is not
// 		supported in the current kernel.
//
// 		*argval* is a flag array which can combine these flags:
//
// 		* **BPF_SOCK_OPS_RTO_CB_FLAG** (retransmission time out)
// 		* **BPF_SOCK_OPS_RETRANS_CB_FLAG** (retransmission)
// 		* **BPF_SOCK_OPS_STATE_CB_FLAG** (TCP state change)
// 		* **BPF_SOCK_OPS_RTT_CB_FLAG** (every RTT)
//
// 		Therefore, this function can be used to clear a callback flag by
// 		setting the appropriate bit to zero. e.g. to disable the RTO
// 		callback:
//
// 		**bpf_sock_ops_cb_flags_set(bpf_sock,**
// 			**bpf_sock->bpf_sock_ops_cb_flags & ~BPF_SOCK_OPS_RTO_CB_FLAG)**
//
// 		Here are some examples of where one could call such eBPF
// 		program:
//
// 		* When RTO fires.
// 		* When a packet is retransmitted.
// 		* When the connection terminates.
// 		* When a packet is sent.
// 		* When a packet is received.
// 	Return
// 		Code **-EINVAL** if the socket is not a full TCP socket;
// 		otherwise, a positive number containing the bits that could not
// 		be set is returned (which comes down to 0 if all bits were set
// 		as required).
//
// long bpf_msg_redirect_map(struct sk_msg_buff *msg, struct bpf_map *map, u32 key, u64 flags)
// 	Description
// 		This helper is used in programs implementing policies at the
// 		socket level. If the message *msg* is allowed to pass (i.e. if
// 		the verdict eBPF program returns **SK_PASS**), redirect it to
// 		the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKMAP**) at index *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress path otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_msg_apply_bytes(struct sk_msg_buff *msg, u32 bytes)
// 	Description
// 		For socket policies, apply the verdict of the eBPF program to
// 		the next *bytes* (number of bytes) of message *msg*.
//
// 		For example, this helper can be used in the following cases:
//
// 		* A single **sendmsg**\ () or **sendfile**\ () system call
// 		  contains multiple logical messages that the eBPF program is
// 		  supposed to read and for which it should apply a verdict.
// 		* An eBPF program only cares to read the first *bytes* of a
// 		  *msg*. If the message has a large payload, then setting up
// 		  and calling the eBPF program repeatedly for all bytes, even
// 		  though the verdict is already known, would create unnecessary
// 		  overhead.
//
// 		When called from within an eBPF program, the helper sets a
// 		counter internal to the BPF infrastructure, that is used to
// 		apply the last verdict to the next *bytes*. If *bytes* is
// 		smaller than the current data being processed from a
// 		**sendmsg**\ () or **sendfile**\ () system call, the first
// 		*bytes* will be sent and the eBPF program will be re-run with
// 		the pointer for start of data pointing to byte number *bytes*
// 		**+ 1**. If *bytes* is larger than the current data being
// 		processed, then the eBPF verdict will be applied to multiple
// 		**sendmsg**\ () or **sendfile**\ () calls until *bytes* are
// 		consumed.
//
// 		Note that if a socket closes with the internal counter holding
// 		a non-zero value, this is not a problem because data is not
// 		being buffered for *bytes* and is sent as it is received.
// 	Return
// 		0
//
// long bpf_msg_cork_bytes(struct sk_msg_buff *msg, u32 bytes)
// 	Description
// 		For socket policies, prevent the execution of the verdict eBPF
// 		program for message *msg* until *bytes* (byte number) have been
// 		accumulated.
//
// 		This can be used when one needs a specific number of bytes
// 		before a verdict can be assigned, even if the data spans
// 		multiple **sendmsg**\ () or **sendfile**\ () calls. The extreme
// 		case would be a user calling **sendmsg**\ () repeatedly with
// 		1-byte long message segments. Obviously, this is bad for
// 		performance, but it is still valid. If the eBPF program needs
// 		*bytes* bytes to validate a header, this helper can be used to
// 		prevent the eBPF program to be called again until *bytes* have
// 		been accumulated.
// 	Return
// 		0
//
// long bpf_msg_pull_data(struct sk_msg_buff *msg, u32 start, u32 end, u64 flags)
// 	Description
// 		For socket policies, pull in non-linear data from user space
// 		for *msg* and set pointers *msg*\ **->data** and *msg* * 		**->data_end** to *start* and *end* bytes offsets into *msg*,
// 		respectively.
//
// 		If a program of type **BPF_PROG_TYPE_SK_MSG** is run on a
// 		*msg* it can only parse data that the (**data**, **data_end**)
// 		pointers have already consumed. For **sendmsg**\ () hooks this
// 		is likely the first scatterlist element. But for calls relying
// 		on the **sendpage** handler (e.g. **sendfile**\ ()) this will
// 		be the range (**0**, **0**) because the data is shared with
// 		user space and by default the objective is to avoid allowing
// 		user space to modify data while (or after) eBPF verdict is
// 		being decided. This helper can be used to pull in data and to
// 		set the start and end pointer to given values. Data will be
// 		copied if necessary (i.e. if data was not linear and if start
// 		and end pointers do not point to the same chunk).
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_bind(struct bpf_sock_addr *ctx, struct sockaddr *addr, int addr_len)
// 	Description
// 		Bind the socket associated to *ctx* to the address pointed by
// 		*addr*, of length *addr_len*. This allows for making outgoing
// 		connection from the desired IP address, which can be useful for
// 		example when all processes inside a cgroup should use one
// 		single IP address on a host that has multiple IP configured.
//
// 		This helper works for IPv4 and IPv6, TCP and UDP sockets. The
// 		domain (*addr*\ **->sa_family**) must be **AF_INET** (or
// 		**AF_INET6**). It's advised to pass zero port (**sin_port**
// 		or **sin6_port**) which triggers IP_BIND_ADDRESS_NO_PORT-like
// 		behavior and lets the kernel efficiently pick up an unused
// 		port as long as 4-tuple is unique. Passing non-zero port might
// 		lead to degraded performance.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_adjust_tail(struct xdp_buff *xdp_md, int delta)
// 	Description
// 		Adjust (move) *xdp_md*\ **->data_end** by *delta* bytes. It is
// 		possible to both shrink and grow the packet tail.
// 		Shrink done via *delta* being a negative integer.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_get_xfrm_state(struct sk_buff *skb, u32 index, struct bpf_xfrm_state *xfrm_state, u32 size, u64 flags)
// 	Description
// 		Retrieve the XFRM state (IP transform framework, see also
// 		**ip-xfrm(8)**) at *index* in XFRM "security path" for *skb*.
//
// 		The retrieved value is stored in the **struct bpf_xfrm_state**
// 		pointed by *xfrm_state* and of length *size*.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_XFRM** configuration option.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_get_stack(void *ctx, void *buf, u32 size, u64 flags)
// 	Description
// 		Return a user or a kernel stack in bpf program provided buffer.
// 		To achieve this, the helper needs *ctx*, which is a pointer
// 		to the context on which the tracing program is executed.
// 		To store the stacktrace, the bpf program provides *buf* with
// 		a nonnegative *size*.
//
// 		The last argument, *flags*, holds the number of stack frames to
// 		skip (from 0 to 255), masked with
// 		**BPF_F_SKIP_FIELD_MASK**. The next bits can be used to set
// 		the following flags:
//
// 		**BPF_F_USER_STACK**
// 			Collect a user space stack instead of a kernel stack.
// 		**BPF_F_USER_BUILD_ID**
// 			Collect (build_id, file_offset) instead of ips for user
// 			stack, only valid if **BPF_F_USER_STACK** is also
// 			specified.
//
// 			*file_offset* is an offset relative to the beginning
// 			of the executable or shared object file backing the vma
// 			which the *ip* falls in. It is *not* an offset relative
// 			to that object's base address. Accordingly, it must be
// 			adjusted by adding (sh_addr - sh_offset), where
// 			sh_{addr,offset} correspond to the executable section
// 			containing *file_offset* in the object, for comparisons
// 			to symbols' st_value to be valid.
//
// 		**bpf_get_stack**\ () can collect up to
// 		**PERF_MAX_STACK_DEPTH** both kernel and user frames, subject
// 		to sufficient large buffer size. Note that
// 		this limit can be controlled with the **sysctl** program, and
// 		that it should be manually increased in order to profile long
// 		user stacks (such as stacks for Java programs). To do so, use:
//
// 		::
//
// 			# sysctl kernel.perf_event_max_stack=<new value>
// 	Return
// 		The non-negative copied *buf* length equal to or less than
// 		*size* on success, or a negative error in case of failure.
//
// long bpf_skb_load_bytes_relative(const void *skb, u32 offset, void *to, u32 len, u32 start_header)
// 	Description
// 		This helper is similar to **bpf_skb_load_bytes**\ () in that
// 		it provides an easy way to load *len* bytes from *offset*
// 		from the packet associated to *skb*, into the buffer pointed
// 		by *to*. The difference to **bpf_skb_load_bytes**\ () is that
// 		a fifth argument *start_header* exists in order to select a
// 		base offset to start from. *start_header* can be one of:
//
// 		**BPF_HDR_START_MAC**
// 			Base offset to load data from is *skb*'s mac header.
// 		**BPF_HDR_START_NET**
// 			Base offset to load data from is *skb*'s network header.
//
// 		In general, "direct packet access" is the preferred method to
// 		access packet data, however, this helper is in particular useful
// 		in socket filters where *skb*\ **->data** does not always point
// 		to the start of the mac header and where "direct packet access"
// 		is not available.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_fib_lookup(void *ctx, struct bpf_fib_lookup *params, int plen, u32 flags)
// 	Description
// 		Do FIB lookup in kernel tables using parameters in *params*.
// 		If lookup is successful and result shows packet is to be
// 		forwarded, the neighbor tables are searched for the nexthop.
// 		If successful (ie., FIB lookup shows forwarding and nexthop
// 		is resolved), the nexthop address is returned in ipv4_dst
// 		or ipv6_dst based on family, smac is set to mac address of
// 		egress device, dmac is set to nexthop mac address, rt_metric
// 		is set to metric from route (IPv4/IPv6 only), and ifindex
// 		is set to the device index of the nexthop from the FIB lookup.
//
// 		*plen* argument is the size of the passed in struct.
// 		*flags* argument can be a combination of one or more of the
// 		following values:
//
// 		**BPF_FIB_LOOKUP_DIRECT**
// 			Do a direct table lookup vs full lookup using FIB
// 			rules.
// 		**BPF_FIB_LOOKUP_TBID**
// 			Used with BPF_FIB_LOOKUP_DIRECT.
// 			Use the routing table ID present in *params*->tbid
// 			for the fib lookup.
// 		**BPF_FIB_LOOKUP_OUTPUT**
// 			Perform lookup from an egress perspective (default is
// 			ingress).
// 		**BPF_FIB_LOOKUP_SKIP_NEIGH**
// 			Skip the neighbour table lookup. *params*->dmac
// 			and *params*->smac will not be set as output. A common
// 			use case is to call **bpf_redirect_neigh**\ () after
// 			doing **bpf_fib_lookup**\ ().
// 		**BPF_FIB_LOOKUP_SRC**
// 			Derive and set source IP addr in *params*->ipv{4,6}_src
// 			for the nexthop. If the src addr cannot be derived,
// 			**BPF_FIB_LKUP_RET_NO_SRC_ADDR** is returned. In this
// 			case, *params*->dmac and *params*->smac are not set either.
// 		**BPF_FIB_LOOKUP_MARK**
// 			Use the mark present in *params*->mark for the fib lookup.
// 			This option should not be used with BPF_FIB_LOOKUP_DIRECT,
// 			as it only has meaning for full lookups.
// 		**BPF_FIB_LOOKUP_VLAN**
// 			If the fib lookup resolves to a VLAN device whose
// 			parent is a real (non-VLAN) device, set
// 			*params*->h_vlan_proto and *params*->h_vlan_TCI from
// 			the VLAN device and replace *params*->ifindex with the
// 			parent's ifindex. *params*->h_vlan_TCI carries the VID
// 			only, with PCP and DEI bits zero; a consumer wanting to
// 			set egress priority writes PCP itself. *params*->smac is
// 			the VLAN device's own address, which can differ from the
// 			parent's. Only the immediate parent is resolved; if it
// 			is itself a VLAN device (QinQ) or in another namespace,
// 			the egress cannot be reduced to a physical device plus
// 			one tag and the lookup returns
// 			**BPF_FIB_LKUP_RET_VLAN_FAILURE** with *params*->ifindex
// 			left at the input. To obtain the VLAN device's own
// 			ifindex, repeat the lookup without
// 			**BPF_FIB_LOOKUP_VLAN**, re-initializing *params*
// 			first: output fields overwrite the inputs they share
// 			storage with. The swap and the vlan fields
// 			are written only on success; other output fields keep
// 			the helper's existing behaviour, so a frag-needed result
// 			still reports the route mtu in *params*->mtu_result.
// 			This flag is only valid for XDP programs; tc programs
// 			receive -EINVAL since they can redirect to the VLAN
// 			device directly.
// 		**BPF_FIB_LOOKUP_VLAN_INPUT**
// 			Treat *params*->h_vlan_proto and *params*->h_vlan_TCI
// 			as an input VLAN tag and run the lookup as if ingress
// 			had happened on the VLAN subinterface carrying that tag
// 			on *params*->ifindex. The VID is the low 12 bits of
// 			*params*->h_vlan_TCI; *params*->h_vlan_proto must be
// 			ETH_P_8021Q or ETH_P_8021AD in network byte order, else
// 			**-EINVAL**. If *params*->ifindex is itself a VLAN
// 			device, its inner (QinQ) subinterface is matched; for a
// 			bond or team, pass the master's ifindex. An unmatched
// 			tag, a down device, or one in another namespace returns
// 			**BPF_FIB_LKUP_RET_NOT_FWDED**, mirroring real ingress.
// 			A VID of 0 is looked up literally, so do not set this
// 			flag for priority-tagged frames. Cannot be combined with
// 			**BPF_FIB_LOOKUP_TBID** or **BPF_FIB_LOOKUP_OUTPUT**
// 			(returns **-EINVAL**).
//
// 		*ctx* is either **struct xdp_md** for XDP programs or
// 		**struct sk_buff** tc cls_act programs.
// 	Return
// 		* < 0 if any input argument is invalid
// 		*   0 on success (packet is forwarded, nexthop neighbor exists)
// 		* > 0 one of **BPF_FIB_LKUP_RET_** codes explaining why the
// 		  packet is not forwarded or needs assist from full stack
//
// 		If lookup fails with BPF_FIB_LKUP_RET_FRAG_NEEDED, then the MTU
// 		was exceeded and output params->mtu_result contains the MTU.
//
// long bpf_sock_hash_update(struct bpf_sock_ops *skops, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		Add an entry to, or update a sockhash *map* referencing sockets.
// 		The *skops* is used as a new value for the entry associated to
// 		*key*. *flags* is one of:
//
// 		**BPF_NOEXIST**
// 			The entry for *key* must not exist in the map.
// 		**BPF_EXIST**
// 			The entry for *key* must already exist in the map.
// 		**BPF_ANY**
// 			No condition on the existence of the entry for *key*.
//
// 		If the *map* has eBPF programs (parser and verdict), those will
// 		be inherited by the socket being added. If the socket is
// 		already attached to eBPF programs, this results in an error.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_msg_redirect_hash(struct sk_msg_buff *msg, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		This helper is used in programs implementing policies at the
// 		socket level. If the message *msg* is allowed to pass (i.e. if
// 		the verdict eBPF program returns **SK_PASS**), redirect it to
// 		the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKHASH**) using hash *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress path otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_sk_redirect_hash(struct sk_buff *skb, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		This helper is used in programs implementing policies at the
// 		skb socket level. If the sk_buff *skb* is allowed to pass (i.e.
// 		if the verdict eBPF program returns **SK_PASS**), redirect it
// 		to the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKHASH**) using hash *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_lwt_push_encap(struct sk_buff *skb, u32 type, void *hdr, u32 len)
// 	Description
// 		Encapsulate the packet associated to *skb* within a Layer 3
// 		protocol header. This header is provided in the buffer at
// 		address *hdr*, with *len* its size in bytes. *type* indicates
// 		the protocol of the header and can be one of:
//
// 		**BPF_LWT_ENCAP_SEG6**
// 			IPv6 encapsulation with Segment Routing Header
// 			(**struct ipv6_sr_hdr**). *hdr* only contains the SRH,
// 			the IPv6 header is computed by the kernel.
// 		**BPF_LWT_ENCAP_SEG6_INLINE**
// 			Only works if *skb* contains an IPv6 packet. Insert a
// 			Segment Routing Header (**struct ipv6_sr_hdr**) inside
// 			the IPv6 header.
// 		**BPF_LWT_ENCAP_IP**
// 			IP encapsulation (GRE/GUE/IPIP/etc). The outer header
// 			must be IPv4 or IPv6, followed by zero or more
// 			additional headers, up to **LWT_BPF_MAX_HEADROOM**
// 			total bytes in all prepended headers. Please note that
// 			if **skb_is_gso**\ (*skb*) is true, no more than two
// 			headers can be prepended, and the inner header, if
// 			present, should be either GRE or UDP/GUE.
//
// 		**BPF_LWT_ENCAP_SEG6**\ \* types can be called by BPF programs
// 		of type **BPF_PROG_TYPE_LWT_IN**; **BPF_LWT_ENCAP_IP** type can
// 		be called by bpf programs of types **BPF_PROG_TYPE_LWT_IN** and
// 		**BPF_PROG_TYPE_LWT_XMIT**.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_lwt_seg6_store_bytes(struct sk_buff *skb, u32 offset, const void *from, u32 len)
// 	Description
// 		Store *len* bytes from address *from* into the packet
// 		associated to *skb*, at *offset*. Only the flags, tag and TLVs
// 		inside the outermost IPv6 Segment Routing Header can be
// 		modified through this helper.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_lwt_seg6_adjust_srh(struct sk_buff *skb, u32 offset, s32 delta)
// 	Description
// 		Adjust the size allocated to TLVs in the outermost IPv6
// 		Segment Routing Header contained in the packet associated to
// 		*skb*, at position *offset* by *delta* bytes. Only offsets
// 		after the segments are accepted. *delta* can be as well
// 		positive (growing) as negative (shrinking).
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_lwt_seg6_action(struct sk_buff *skb, u32 action, void *param, u32 param_len)
// 	Description
// 		Apply an IPv6 Segment Routing action of type *action* to the
// 		packet associated to *skb*. Each action takes a parameter
// 		contained at address *param*, and of length *param_len* bytes.
// 		*action* can be one of:
//
// 		**SEG6_LOCAL_ACTION_END_X**
// 			End.X action: Endpoint with Layer-3 cross-connect.
// 			Type of *param*: **struct in6_addr**.
// 		**SEG6_LOCAL_ACTION_END_T**
// 			End.T action: Endpoint with specific IPv6 table lookup.
// 			Type of *param*: **int**.
// 		**SEG6_LOCAL_ACTION_END_B6**
// 			End.B6 action: Endpoint bound to an SRv6 policy.
// 			Type of *param*: **struct ipv6_sr_hdr**.
// 		**SEG6_LOCAL_ACTION_END_B6_ENCAP**
// 			End.B6.Encap action: Endpoint bound to an SRv6
// 			encapsulation policy.
// 			Type of *param*: **struct ipv6_sr_hdr**.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_rc_repeat(void *ctx)
// 	Description
// 		This helper is used in programs implementing IR decoding, to
// 		report a successfully decoded repeat key message. This delays
// 		the generation of a key up event for previously generated
// 		key down event.
//
// 		Some IR protocols like NEC have a special IR message for
// 		repeating last button, for when a button is held down.
//
// 		The *ctx* should point to the lirc sample as passed into
// 		the program.
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_BPF_LIRC_MODE2** configuration option set to
// 		"**y**".
// 	Return
// 		0
//
// long bpf_rc_keydown(void *ctx, u32 protocol, u64 scancode, u32 toggle)
// 	Description
// 		This helper is used in programs implementing IR decoding, to
// 		report a successfully decoded key press with *scancode*,
// 		*toggle* value in the given *protocol*. The scancode will be
// 		translated to a keycode using the rc keymap, and reported as
// 		an input key down event. After a period a key up event is
// 		generated. This period can be extended by calling either
// 		**bpf_rc_keydown**\ () again with the same values, or calling
// 		**bpf_rc_repeat**\ ().
//
// 		Some protocols include a toggle bit, in case the button was
// 		released and pressed again between consecutive scancodes.
//
// 		The *ctx* should point to the lirc sample as passed into
// 		the program.
//
// 		The *protocol* is the decoded protocol number (see
// 		**enum rc_proto** for some predefined values).
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_BPF_LIRC_MODE2** configuration option set to
// 		"**y**".
// 	Return
// 		0
//
// u64 bpf_skb_cgroup_id(struct sk_buff *skb)
// 	Description
// 		Return the cgroup v2 id of the socket associated with the *skb*.
// 		This is roughly similar to the **bpf_get_cgroup_classid**\ ()
// 		helper for cgroup v1 by providing a tag resp. identifier that
// 		can be matched on or used for map lookups e.g. to implement
// 		policy. The cgroup v2 id of a given path in the hierarchy is
// 		exposed in user space through the f_handle API in order to get
// 		to the same 64-bit id.
//
// 		This helper can be used on TC egress path, but not on ingress,
// 		and is available only if the kernel was compiled with the
// 		**CONFIG_SOCK_CGROUP_DATA** configuration option.
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// u64 bpf_get_current_cgroup_id(void)
// 	Description
// 		Get the current cgroup id based on the cgroup within which
// 		the current task is running.
// 	Return
// 		A 64-bit integer containing the current cgroup id based
// 		on the cgroup within which the current task is running.
//
// void *bpf_get_local_storage(void *map, u64 flags)
// 	Description
// 		Get the pointer to the local storage area.
// 		The type and the size of the local storage is defined
// 		by the *map* argument.
// 		The *flags* meaning is specific for each map type,
// 		and has to be 0 for cgroup local storage.
//
// 		Depending on the BPF program type, a local storage area
// 		can be shared between multiple instances of the BPF program,
// 		running simultaneously.
//
// 		A user should care about the synchronization by himself.
// 		For example, by using the **BPF_ATOMIC** instructions to alter
// 		the shared data.
// 	Return
// 		A pointer to the local storage area.
//
// long bpf_sk_select_reuseport(struct sk_reuseport_md *reuse, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		Select a **SO_REUSEPORT** socket from a
// 		**BPF_MAP_TYPE_REUSEPORT_SOCKARRAY** *map*.
// 		It checks the selected socket is matching the incoming
// 		request in the socket buffer.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_skb_ancestor_cgroup_id(struct sk_buff *skb, int ancestor_level)
// 	Description
// 		Return id of cgroup v2 that is ancestor of cgroup associated
// 		with the *skb* at the *ancestor_level*.  The root cgroup is at
// 		*ancestor_level* zero and each step down the hierarchy
// 		increments the level. If *ancestor_level* == level of cgroup
// 		associated with *skb*, then return value will be same as that
// 		of **bpf_skb_cgroup_id**\ ().
//
// 		The helper is useful to implement policies based on cgroups
// 		that are upper in hierarchy than immediate cgroup associated
// 		with *skb*.
//
// 		The format of returned id and helper limitations are same as in
// 		**bpf_skb_cgroup_id**\ ().
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// struct bpf_sock *bpf_sk_lookup_tcp(void *ctx, struct bpf_sock_tuple *tuple, u32 tuple_size, u64 netns, u64 flags)
// 	Description
// 		Look for TCP socket matching *tuple*, optionally in a child
// 		network namespace *netns*. The return value must be checked,
// 		and if non-**NULL**, released via **bpf_sk_release**\ ().
//
// 		The *ctx* should point to the context of the program, such as
// 		the skb or socket (depending on the hook in use). This is used
// 		to determine the base network namespace for the lookup.
//
// 		*tuple_size* must be one of:
//
// 		**sizeof**\ (*tuple*\ **->ipv4**)
// 			Look for an IPv4 socket.
// 		**sizeof**\ (*tuple*\ **->ipv6**)
// 			Look for an IPv6 socket.
//
// 		If the *netns* is a negative signed 32-bit integer, then the
// 		socket lookup table in the netns associated with the *ctx*
// 		will be used. For the TC hooks, this is the netns of the device
// 		in the skb. For socket hooks, this is the netns of the socket.
// 		If *netns* is any other signed 32-bit value greater than or
// 		equal to zero then it specifies the ID of the netns relative to
// 		the netns associated with the *ctx*. *netns* values beyond the
// 		range of 32-bit integers are reserved for future use.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_NET** configuration option.
// 	Return
// 		Pointer to **struct bpf_sock**, or **NULL** in case of failure.
// 		For sockets with reuseport option, the **struct bpf_sock**
// 		result is from *reuse*\ **->socks**\ [] using the hash of the
// 		tuple.
//
// struct bpf_sock *bpf_sk_lookup_udp(void *ctx, struct bpf_sock_tuple *tuple, u32 tuple_size, u64 netns, u64 flags)
// 	Description
// 		Look for UDP socket matching *tuple*, optionally in a child
// 		network namespace *netns*. The return value must be checked,
// 		and if non-**NULL**, released via **bpf_sk_release**\ ().
//
// 		The *ctx* should point to the context of the program, such as
// 		the skb or socket (depending on the hook in use). This is used
// 		to determine the base network namespace for the lookup.
//
// 		*tuple_size* must be one of:
//
// 		**sizeof**\ (*tuple*\ **->ipv4**)
// 			Look for an IPv4 socket.
// 		**sizeof**\ (*tuple*\ **->ipv6**)
// 			Look for an IPv6 socket.
//
// 		If the *netns* is a negative signed 32-bit integer, then the
// 		socket lookup table in the netns associated with the *ctx*
// 		will be used. For the TC hooks, this is the netns of the device
// 		in the skb. For socket hooks, this is the netns of the socket.
// 		If *netns* is any other signed 32-bit value greater than or
// 		equal to zero then it specifies the ID of the netns relative to
// 		the netns associated with the *ctx*. *netns* values beyond the
// 		range of 32-bit integers are reserved for future use.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_NET** configuration option.
// 	Return
// 		Pointer to **struct bpf_sock**, or **NULL** in case of failure.
// 		For sockets with reuseport option, the **struct bpf_sock**
// 		result is from *reuse*\ **->socks**\ [] using the hash of the
// 		tuple.
//
// long bpf_sk_release(void *sock)
// 	Description
// 		Release the reference held by *sock*. *sock* must be a
// 		non-**NULL** pointer that was returned from
// 		**bpf_sk_lookup_xxx**\ ().
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_push_elem(struct bpf_map *map, const void *value, u64 flags)
// 	Description
// 		Push an element *value* in *map*. *flags* is one of:
//
// 		**BPF_EXIST**
// 			If the queue/stack is full, the oldest element is
// 			removed to make room for this.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_pop_elem(struct bpf_map *map, void *value)
// 	Description
// 		Pop an element from *map*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_peek_elem(struct bpf_map *map, void *value)
// 	Description
// 		Get an element from *map* without removing it.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_msg_push_data(struct sk_msg_buff *msg, u32 start, u32 len, u64 flags)
// 	Description
// 		For socket policies, insert *len* bytes into *msg* at offset
// 		*start*.
//
// 		If a program of type **BPF_PROG_TYPE_SK_MSG** is run on a
// 		*msg* it may want to insert metadata or options into the *msg*.
// 		This can later be read and used by any of the lower layer BPF
// 		hooks.
//
// 		This helper may fail if under memory pressure (a malloc
// 		fails) in these cases BPF programs will get an appropriate
// 		error and BPF programs will need to handle them.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_msg_pop_data(struct sk_msg_buff *msg, u32 start, u32 len, u64 flags)
// 	Description
// 		Will remove *len* bytes from a *msg* starting at byte *start*.
// 		This may result in **ENOMEM** errors under certain situations if
// 		an allocation and copy are required due to a full ring buffer.
// 		However, the helper will try to avoid doing the allocation
// 		if possible. Other errors can occur if input parameters are
// 		invalid either due to *start* byte not being valid part of *msg*
// 		payload and/or *pop* value being to large.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_rc_pointer_rel(void *ctx, s32 rel_x, s32 rel_y)
// 	Description
// 		This helper is used in programs implementing IR decoding, to
// 		report a successfully decoded pointer movement.
//
// 		The *ctx* should point to the lirc sample as passed into
// 		the program.
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_BPF_LIRC_MODE2** configuration option set to
// 		"**y**".
// 	Return
// 		0
//
// long bpf_spin_lock(struct bpf_spin_lock *lock)
// 	Description
// 		Acquire a spinlock represented by the pointer *lock*, which is
// 		stored as part of a value of a map. Taking the lock allows to
// 		safely update the rest of the fields in that value. The
// 		spinlock can (and must) later be released with a call to
// 		**bpf_spin_unlock**\ (\ *lock*\ ).
//
// 		Spinlocks in BPF programs come with a number of restrictions
// 		and constraints:
//
// 		* **bpf_spin_lock** objects are only allowed inside maps of
// 		  types **BPF_MAP_TYPE_HASH** and **BPF_MAP_TYPE_ARRAY** (this
// 		  list could be extended in the future).
// 		* BTF description of the map is mandatory.
// 		* The BPF program can take ONE lock at a time, since taking two
// 		  or more could cause dead locks.
// 		* Only one **struct bpf_spin_lock** is allowed per map element.
// 		* When the lock is taken, calls (either BPF to BPF or helpers)
// 		  are not allowed.
// 		* The **BPF_LD_ABS** and **BPF_LD_IND** instructions are not
// 		  allowed inside a spinlock-ed region.
// 		* The BPF program MUST call **bpf_spin_unlock**\ () to release
// 		  the lock, on all execution paths, before it returns.
// 		* The BPF program can access **struct bpf_spin_lock** only via
// 		  the **bpf_spin_lock**\ () and **bpf_spin_unlock**\ ()
// 		  helpers. Loading or storing data into the **struct
// 		  bpf_spin_lock** *lock*\ **;** field of a map is not allowed.
// 		* To use the **bpf_spin_lock**\ () helper, the BTF description
// 		  of the map value must be a struct and have **struct
// 		  bpf_spin_lock** *anyname*\ **;** field at the top level.
// 		  Nested lock inside another struct is not allowed.
// 		* The **struct bpf_spin_lock** *lock* field in a map value must
// 		  be aligned on a multiple of 4 bytes in that value.
// 		* Syscall with command **BPF_MAP_LOOKUP_ELEM** does not copy
// 		  the **bpf_spin_lock** field to user space.
// 		* Syscall with command **BPF_MAP_UPDATE_ELEM**, or update from
// 		  a BPF program, do not update the **bpf_spin_lock** field.
// 		* **bpf_spin_lock** cannot be on the stack or inside a
// 		  networking packet (it can only be inside of a map values).
// 		* **bpf_spin_lock** is available to root only.
// 		* Tracing programs and socket filter programs cannot use
// 		  **bpf_spin_lock**\ () due to insufficient preemption checks
// 		  (but this may change in the future).
// 		* **bpf_spin_lock** is not allowed in inner maps of map-in-map.
// 	Return
// 		0
//
// long bpf_spin_unlock(struct bpf_spin_lock *lock)
// 	Description
// 		Release the *lock* previously locked by a call to
// 		**bpf_spin_lock**\ (\ *lock*\ ).
// 	Return
// 		0
//
// struct bpf_sock *bpf_sk_fullsock(struct bpf_sock *sk)
// 	Description
// 		This helper gets a **struct bpf_sock** pointer such
// 		that all the fields in this **bpf_sock** can be accessed.
// 	Return
// 		A **struct bpf_sock** pointer on success, or **NULL** in
// 		case of failure.
//
// struct bpf_tcp_sock *bpf_tcp_sock(struct bpf_sock *sk)
// 	Description
// 		This helper gets a **struct bpf_tcp_sock** pointer from a
// 		**struct bpf_sock** pointer.
// 	Return
// 		A **struct bpf_tcp_sock** pointer on success, or **NULL** in
// 		case of failure.
//
// long bpf_skb_ecn_set_ce(struct sk_buff *skb)
// 	Description
// 		Set ECN (Explicit Congestion Notification) field of IP header
// 		to **CE** (Congestion Encountered) if current value is **ECT**
// 		(ECN Capable Transport). Otherwise, do nothing. Works with IPv6
// 		and IPv4.
// 	Return
// 		1 if the **CE** flag is set (either by the current helper call
// 		or because it was already present), 0 if it is not set.
//
// struct bpf_sock *bpf_get_listener_sock(struct bpf_sock *sk)
// 	Description
// 		Return a **struct bpf_sock** pointer in **TCP_LISTEN** state.
// 		**bpf_sk_release**\ () is unnecessary and not allowed.
// 	Return
// 		A **struct bpf_sock** pointer on success, or **NULL** in
// 		case of failure.
//
// struct bpf_sock *bpf_skc_lookup_tcp(void *ctx, struct bpf_sock_tuple *tuple, u32 tuple_size, u64 netns, u64 flags)
// 	Description
// 		Look for TCP socket matching *tuple*, optionally in a child
// 		network namespace *netns*. The return value must be checked,
// 		and if non-**NULL**, released via **bpf_sk_release**\ ().
//
// 		This function is identical to **bpf_sk_lookup_tcp**\ (), except
// 		that it also returns timewait or request sockets. Use
// 		**bpf_sk_fullsock**\ () or **bpf_tcp_sock**\ () to access the
// 		full structure.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_NET** configuration option.
// 	Return
// 		Pointer to **struct bpf_sock**, or **NULL** in case of failure.
// 		For sockets with reuseport option, the **struct bpf_sock**
// 		result is from *reuse*\ **->socks**\ [] using the hash of the
// 		tuple.
//
// long bpf_tcp_check_syncookie(void *sk, void *iph, u32 iph_len, struct tcphdr *th, u32 th_len)
// 	Description
// 		Check whether *iph* and *th* contain a valid SYN cookie ACK for
// 		the listening socket in *sk*.
//
// 		*iph* points to the start of the IPv4 or IPv6 header, while
// 		*iph_len* contains **sizeof**\ (**struct iphdr**) or
// 		**sizeof**\ (**struct ipv6hdr**).
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		0 if *iph* and *th* are a valid SYN cookie ACK, or a negative
// 		error otherwise.
//
// long bpf_sysctl_get_name(struct bpf_sysctl *ctx, char *buf, size_t buf_len, u64 flags)
// 	Description
// 		Get name of sysctl in /proc/sys/ and copy it into provided by
// 		program buffer *buf* of size *buf_len*.
//
// 		The buffer is always NUL terminated, unless it's zero-sized.
//
// 		If *flags* is zero, full name (e.g. "net/ipv4/tcp_mem") is
// 		copied. Use **BPF_F_SYSCTL_BASE_NAME** flag to copy base name
// 		only (e.g. "tcp_mem").
// 	Return
// 		Number of character copied (not including the trailing NUL).
//
// 		**-E2BIG** if the buffer wasn't big enough (*buf* will contain
// 		truncated name in this case).
//
// long bpf_sysctl_get_current_value(struct bpf_sysctl *ctx, char *buf, size_t buf_len)
// 	Description
// 		Get current value of sysctl as it is presented in /proc/sys
// 		(incl. newline, etc), and copy it as a string into provided
// 		by program buffer *buf* of size *buf_len*.
//
// 		The whole value is copied, no matter what file position user
// 		space issued e.g. sys_read at.
//
// 		The buffer is always NUL terminated, unless it's zero-sized.
// 	Return
// 		Number of character copied (not including the trailing NUL).
//
// 		**-E2BIG** if the buffer wasn't big enough (*buf* will contain
// 		truncated name in this case).
//
// 		**-EINVAL** if current value was unavailable, e.g. because
// 		sysctl is uninitialized and read returns -EIO for it.
//
// long bpf_sysctl_get_new_value(struct bpf_sysctl *ctx, char *buf, size_t buf_len)
// 	Description
// 		Get new value being written by user space to sysctl (before
// 		the actual write happens) and copy it as a string into
// 		provided by program buffer *buf* of size *buf_len*.
//
// 		User space may write new value at file position > 0.
//
// 		The buffer is always NUL terminated, unless it's zero-sized.
// 	Return
// 		Number of character copied (not including the trailing NUL).
//
// 		**-E2BIG** if the buffer wasn't big enough (*buf* will contain
// 		truncated name in this case).
//
// 		**-EINVAL** if sysctl is being read.
//
// long bpf_sysctl_set_new_value(struct bpf_sysctl *ctx, const char *buf, size_t buf_len)
// 	Description
// 		Override new value being written by user space to sysctl with
// 		value provided by program in buffer *buf* of size *buf_len*.
//
// 		*buf* should contain a string in same form as provided by user
// 		space on sysctl write.
//
// 		User space may write new value at file position > 0. To override
// 		the whole sysctl value file position should be set to zero.
// 	Return
// 		0 on success.
//
// 		**-E2BIG** if the *buf_len* is too big.
//
// 		**-EINVAL** if sysctl is being read.
//
// long bpf_strtol(const char *buf, size_t buf_len, u64 flags, long *res)
// 	Description
// 		Convert the initial part of the string from buffer *buf* of
// 		size *buf_len* to a long integer according to the given base
// 		and save the result in *res*.
//
// 		The string may begin with an arbitrary amount of white space
// 		(as determined by **isspace**\ (3)) followed by a single
// 		optional '**-**' sign.
//
// 		Five least significant bits of *flags* encode base, other bits
// 		are currently unused.
//
// 		Base must be either 8, 10, 16 or 0 to detect it automatically
// 		similar to user space **strtol**\ (3).
// 	Return
// 		Number of characters consumed on success. Must be positive but
// 		no more than *buf_len*.
//
// 		**-EINVAL** if no valid digits were found or unsupported base
// 		was provided.
//
// 		**-ERANGE** if resulting value was out of range.
//
// long bpf_strtoul(const char *buf, size_t buf_len, u64 flags, unsigned long *res)
// 	Description
// 		Convert the initial part of the string from buffer *buf* of
// 		size *buf_len* to an unsigned long integer according to the
// 		given base and save the result in *res*.
//
// 		The string may begin with an arbitrary amount of white space
// 		(as determined by **isspace**\ (3)).
//
// 		Five least significant bits of *flags* encode base, other bits
// 		are currently unused.
//
// 		Base must be either 8, 10, 16 or 0 to detect it automatically
// 		similar to user space **strtoul**\ (3).
// 	Return
// 		Number of characters consumed on success. Must be positive but
// 		no more than *buf_len*.
//
// 		**-EINVAL** if no valid digits were found or unsupported base
// 		was provided.
//
// 		**-ERANGE** if resulting value was out of range.
//
// void *bpf_sk_storage_get(struct bpf_map *map, void *sk, void *value, u64 flags)
// 	Description
// 		Get a bpf-local-storage from a *sk*.
//
// 		Logically, it could be thought of getting the value from
// 		a *map* with *sk* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *sk*) except this
// 		helper enforces the key must be a full socket and the map must
// 		be a **BPF_MAP_TYPE_SK_STORAGE** also.
//
// 		Underneath, the value is stored locally at *sk* instead of
// 		the *map*.  The *map* is used as the bpf-local-storage
// 		"type". The bpf-local-storage "type" (i.e. the *map*) is
// 		searched against all bpf-local-storages residing at *sk*.
//
// 		*sk* is a kernel **struct sock** pointer for LSM program.
// 		*sk* is a **struct bpf_sock** pointer for other program types.
//
// 		An optional *flags* (**BPF_SK_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf-local-storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_SK_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf-local-storage.  If *value* is
// 		**NULL**, the new bpf-local-storage will be zero initialized.
// 	Return
// 		A bpf-local-storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf-local-storage.
//
// long bpf_sk_storage_delete(struct bpf_map *map, void *sk)
// 	Description
// 		Delete a bpf-local-storage from a *sk*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf-local-storage cannot be found.
// 		**-EINVAL** if sk is not a fullsock (e.g. a request_sock).
//
// long bpf_send_signal(u32 sig)
// 	Description
// 		Send signal *sig* to the process of the current task.
// 		The signal may be delivered to any of this process's threads.
// 	Return
// 		0 on success or successfully queued.
//
// 		**-EBUSY** if work queue under nmi is full.
//
// 		**-EINVAL** if *sig* is invalid.
//
// 		**-EPERM** if no permission to send the *sig*.
//
// 		**-EAGAIN** if bpf program can try again.
//
// s64 bpf_tcp_gen_syncookie(void *sk, void *iph, u32 iph_len, struct tcphdr *th, u32 th_len)
// 	Description
// 		Try to issue a SYN cookie for the packet with corresponding
// 		IP/TCP headers, *iph* and *th*, on the listening socket in *sk*.
//
// 		*iph* points to the start of the IPv4 or IPv6 header, while
// 		*iph_len* contains **sizeof**\ (**struct iphdr**) or
// 		**sizeof**\ (**struct ipv6hdr**).
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header with options (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		On success, lower 32 bits hold the generated SYN cookie in
// 		followed by 16 bits which hold the MSS value for that cookie,
// 		and the top 16 bits are unused.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EINVAL** SYN cookie cannot be issued due to error
//
// 		**-ENOENT** SYN cookie should not be issued (no SYN flood)
//
// 		**-EOPNOTSUPP** kernel configuration does not enable SYN cookies
//
// 		**-EPROTONOSUPPORT** IP packet version is not 4 or 6
//
// long bpf_skb_output(void *ctx, struct bpf_map *map, u64 flags, void *data, u64 size)
// 	Description
// 		Write raw *data* blob into a special BPF perf event held by
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. This perf
// 		event must have the following attributes: **PERF_SAMPLE_RAW**
// 		as **sample_type**, **PERF_TYPE_SOFTWARE** as **type**, and
// 		**PERF_COUNT_SW_BPF_OUTPUT** as **config**.
//
// 		The *flags* are used to indicate the index in *map* for which
// 		the value must be put, masked with **BPF_F_INDEX_MASK**.
// 		Alternatively, *flags* can be set to **BPF_F_CURRENT_CPU**
// 		to indicate that the index of the current CPU core should be
// 		used.
//
// 		The value to write, of *size*, is passed through eBPF stack and
// 		pointed by *data*.
//
// 		*ctx* is a pointer to in-kernel struct sk_buff.
//
// 		This helper is similar to **bpf_perf_event_output**\ () but
// 		restricted to raw_tracepoint bpf programs.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_user(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Safely attempt to read *size* bytes from user space address
// 		*unsafe_ptr* and store the data in *dst*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_kernel(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Safely attempt to read *size* bytes from kernel space address
// 		*unsafe_ptr* and store the data in *dst*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_user_str(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Copy a NUL terminated string from an unsafe user address
// 		*unsafe_ptr* to *dst*. The *size* should include the
// 		terminating NUL byte. In case the string length is smaller than
// 		*size*, the target is not padded with further NUL bytes. If the
// 		string length is larger than *size*, just *size*-1 bytes are
// 		copied and the last byte is set to NUL.
//
// 		On success, returns the number of bytes that were written,
// 		including the terminal NUL. This makes this helper useful in
// 		tracing programs for reading strings, and more importantly to
// 		get its length at runtime. See the following snippet:
//
// 		::
//
// 			SEC("kprobe/sys_open")
// 			void bpf_sys_open(struct pt_regs *ctx)
// 			{
// 			        char buf[PATHLEN]; // PATHLEN is defined to 256
// 			        int res = bpf_probe_read_user_str(buf, sizeof(buf),
// 				                                  ctx->di);
//
// 				// Consume buf, for example push it to
// 				// userspace via bpf_perf_event_output(); we
// 				// can use res (the string length) as event
// 				// size, after checking its boundaries.
// 			}
//
// 		In comparison, using **bpf_probe_read_user**\ () helper here
// 		instead to read the string would require to estimate the length
// 		at compile time, and would often result in copying more memory
// 		than necessary.
//
// 		Another useful use case is when parsing individual process
// 		arguments or individual environment variables navigating
// 		*current*\ **->mm->arg_start** and *current* * 		**->mm->env_start**: using this helper and the return value,
// 		one can quickly iterate at the right offset of the memory area.
// 	Return
// 		On success, the strictly positive length of the output string,
// 		including the trailing NUL character. On error, a negative
// 		value.
//
// long bpf_probe_read_kernel_str(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Copy a NUL terminated string from an unsafe kernel address *unsafe_ptr*
// 		to *dst*. Same semantics as with **bpf_probe_read_user_str**\ () apply.
// 	Return
// 		On success, the strictly positive length of the string, including
// 		the trailing NUL character. On error, a negative value.
//
// long bpf_tcp_send_ack(void *tp, u32 rcv_nxt)
// 	Description
// 		Send out a tcp-ack. *tp* is the in-kernel struct **tcp_sock**.
// 		*rcv_nxt* is the ack_seq to be sent out.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_send_signal_thread(u32 sig)
// 	Description
// 		Send signal *sig* to the thread corresponding to the current task.
// 	Return
// 		0 on success or successfully queued.
//
// 		**-EBUSY** if work queue under nmi is full.
//
// 		**-EINVAL** if *sig* is invalid.
//
// 		**-EPERM** if no permission to send the *sig*.
//
// 		**-EAGAIN** if bpf program can try again.
//
// u64 bpf_jiffies64(void)
// 	Description
// 		Obtain the 64bit jiffies
// 	Return
// 		The 64 bit jiffies
//
// long bpf_read_branch_records(struct bpf_perf_event_data *ctx, void *buf, u32 size, u64 flags)
// 	Description
// 		For an eBPF program attached to a perf event, retrieve the
// 		branch records (**struct perf_branch_entry**) associated to *ctx*
// 		and store it in the buffer pointed by *buf* up to size
// 		*size* bytes.
// 	Return
// 		On success, number of bytes written to *buf*. On error, a
// 		negative value.
//
// 		The *flags* can be set to **BPF_F_GET_BRANCH_RECORDS_SIZE** to
// 		instead return the number of bytes required to store all the
// 		branch entries. If this flag is set, *buf* may be NULL.
//
// 		**-EINVAL** if arguments invalid or **size** not a multiple
// 		of **sizeof**\ (**struct perf_branch_entry**\ ).
//
// 		**-ENOENT** if architecture does not support branch records.
//
// long bpf_get_ns_current_pid_tgid(u64 dev, u64 ino, struct bpf_pidns_info *nsdata, u32 size)
// 	Description
// 		Returns 0 on success, values for *pid* and *tgid* as seen from the current
// 		*namespace* will be returned in *nsdata*.
// 	Return
// 		0 on success, or one of the following in case of failure:
//
// 		**-EINVAL** if dev and inum supplied don't match dev_t and inode number
//              with nsfs of current task, or if dev conversion to dev_t lost high bits.
//
// 		**-ENOENT** if pidns does not exists for the current task.
//
// long bpf_xdp_output(void *ctx, struct bpf_map *map, u64 flags, void *data, u64 size)
// 	Description
// 		Write raw *data* blob into a special BPF perf event held by
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. This perf
// 		event must have the following attributes: **PERF_SAMPLE_RAW**
// 		as **sample_type**, **PERF_TYPE_SOFTWARE** as **type**, and
// 		**PERF_COUNT_SW_BPF_OUTPUT** as **config**.
//
// 		The *flags* are used to indicate the index in *map* for which
// 		the value must be put, masked with **BPF_F_INDEX_MASK**.
// 		Alternatively, *flags* can be set to **BPF_F_CURRENT_CPU**
// 		to indicate that the index of the current CPU core should be
// 		used.
//
// 		The value to write, of *size*, is passed through eBPF stack and
// 		pointed by *data*.
//
// 		*ctx* is a pointer to in-kernel struct xdp_buff.
//
// 		This helper is similar to **bpf_perf_eventoutput**\ () but
// 		restricted to raw_tracepoint bpf programs.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_get_netns_cookie(void *ctx)
// 	Description
// 		Retrieve the cookie (generated by the kernel) of the network
// 		namespace the input *ctx* is associated with. The network
// 		namespace cookie remains stable for its lifetime and provides
// 		a global identifier that can be assumed unique. If *ctx* is
// 		NULL, then the helper returns the cookie for the initial
// 		network namespace. The cookie itself is very similar to that
// 		of **bpf_get_socket_cookie**\ () helper, but for network
// 		namespaces instead of sockets.
// 	Return
// 		A 8-byte long opaque number.
//
// u64 bpf_get_current_ancestor_cgroup_id(int ancestor_level)
// 	Description
// 		Return id of cgroup v2 that is ancestor of the cgroup associated
// 		with the current task at the *ancestor_level*. The root cgroup
// 		is at *ancestor_level* zero and each step down the hierarchy
// 		increments the level. If *ancestor_level* == level of cgroup
// 		associated with the current task, then return value will be the
// 		same as that of **bpf_get_current_cgroup_id**\ ().
//
// 		The helper is useful to implement policies based on cgroups
// 		that are upper in hierarchy than immediate cgroup associated
// 		with the current task.
//
// 		The format of returned id and helper limitations are same as in
// 		**bpf_get_current_cgroup_id**\ ().
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// long bpf_sk_assign(struct sk_buff *skb, void *sk, u64 flags)
// 	Description
// 		Helper is overloaded depending on BPF program type. This
// 		description applies to **BPF_PROG_TYPE_SCHED_CLS** and
// 		**BPF_PROG_TYPE_SCHED_ACT** programs.
//
// 		Assign the *sk* to the *skb*. When combined with appropriate
// 		routing configuration to receive the packet towards the socket,
// 		will cause *skb* to be delivered to the specified socket.
// 		Subsequent redirection of *skb* via  **bpf_redirect**\ (),
// 		**bpf_clone_redirect**\ () or other methods outside of BPF may
// 		interfere with successful delivery to the socket.
//
// 		This operation is only valid from TC ingress path.
//
// 		The *flags* argument must be zero.
// 	Return
// 		0 on success, or a negative error in case of failure:
//
// 		**-EINVAL** if specified *flags* are not supported.
//
// 		**-ENOENT** if the socket is unavailable for assignment.
//
// 		**-ENETUNREACH** if the socket is unreachable (wrong netns).
//
// 		**-EOPNOTSUPP** if the operation is not supported, for example
// 		a call from outside of TC ingress.
//
// long bpf_sk_assign(struct bpf_sk_lookup *ctx, struct bpf_sock *sk, u64 flags)
// 	Description
// 		Helper is overloaded depending on BPF program type. This
// 		description applies to **BPF_PROG_TYPE_SK_LOOKUP** programs.
//
// 		Select the *sk* as a result of a socket lookup.
//
// 		For the operation to succeed passed socket must be compatible
// 		with the packet description provided by the *ctx* object.
//
// 		L4 protocol (**IPPROTO_TCP** or **IPPROTO_UDP**) must
// 		be an exact match. While IP family (**AF_INET** or
// 		**AF_INET6**) must be compatible, that is IPv6 sockets
// 		that are not v6-only can be selected for IPv4 packets.
//
// 		Only TCP listeners and UDP unconnected sockets can be
// 		selected. *sk* can also be NULL to reset any previous
// 		selection.
//
// 		*flags* argument can combination of following values:
//
// 		* **BPF_SK_LOOKUP_F_REPLACE** to override the previous
// 		  socket selection, potentially done by a BPF program
// 		  that ran before us.
//
// 		* **BPF_SK_LOOKUP_F_NO_REUSEPORT** to skip
// 		  load-balancing within reuseport group for the socket
// 		  being selected.
//
// 		On success *ctx->sk* will point to the selected socket.
//
// 	Return
// 		0 on success, or a negative errno in case of failure.
//
// 		* **-EAFNOSUPPORT** if socket family (*sk->family*) is
// 		  not compatible with packet family (*ctx->family*).
//
// 		* **-EEXIST** if socket has been already selected,
// 		  potentially by another program, and
// 		  **BPF_SK_LOOKUP_F_REPLACE** flag was not specified.
//
// 		* **-EINVAL** if unsupported flags were specified.
//
// 		* **-EPROTOTYPE** if socket L4 protocol
// 		  (*sk->protocol*) doesn't match packet protocol
// 		  (*ctx->protocol*).
//
// 		* **-ESOCKTNOSUPPORT** if socket is not in allowed
// 		  state (TCP listening or UDP unconnected).
//
// u64 bpf_ktime_get_boot_ns(void)
// 	Description
// 		Return the time elapsed since system boot, in nanoseconds.
// 		Does include the time the system was suspended.
// 		See: **clock_gettime**\ (**CLOCK_BOOTTIME**)
// 	Return
// 		Current *ktime*.
//
// long bpf_seq_printf(struct seq_file *m, const char *fmt, u32 fmt_size, const void *data, u32 data_len)
// 	Description
// 		**bpf_seq_printf**\ () uses seq_file **seq_printf**\ () to print
// 		out the format string.
// 		The *m* represents the seq_file. The *fmt* and *fmt_size* are for
// 		the format string itself. The *data* and *data_len* are format string
// 		arguments. The *data* are a **u64** array and corresponding format string
// 		values are stored in the array. For strings and pointers where pointees
// 		are accessed, only the pointer values are stored in the *data* array.
// 		The *data_len* is the size of *data* in bytes - must be a multiple of 8.
//
// 		Formats **%s**, **%p{i,I}{4,6}** requires to read kernel memory.
// 		Reading kernel memory may fail due to either invalid address or
// 		valid address but requiring a major memory fault. If reading kernel memory
// 		fails, the string for **%s** will be an empty string, and the ip
// 		address for **%p{i,I}{4,6}** will be 0. Not returning error to
// 		bpf program is consistent with what **bpf_trace_printk**\ () does for now.
// 	Return
// 		0 on success, or a negative error in case of failure:
//
// 		**-EBUSY** if per-CPU memory copy buffer is busy, can try again
// 		by returning 1 from bpf program.
//
// 		**-EINVAL** if arguments are invalid, or if *fmt* is invalid/unsupported.
//
// 		**-E2BIG** if *fmt* contains too many format specifiers.
//
// 		**-EOVERFLOW** if an overflow happened: The same object will be tried again.
//
// long bpf_seq_write(struct seq_file *m, const void *data, u32 len)
// 	Description
// 		**bpf_seq_write**\ () uses seq_file **seq_write**\ () to write the data.
// 		The *m* represents the seq_file. The *data* and *len* represent the
// 		data to write in bytes.
// 	Return
// 		0 on success, or a negative error in case of failure:
//
// 		**-EOVERFLOW** if an overflow happened: The same object will be tried again.
//
// u64 bpf_sk_cgroup_id(void *sk)
// 	Description
// 		Return the cgroup v2 id of the socket *sk*.
//
// 		*sk* must be a non-**NULL** pointer to a socket, e.g. one
// 		returned from **bpf_sk_lookup_xxx**\ (),
// 		**bpf_sk_fullsock**\ (), etc. The format of returned id is
// 		same as in **bpf_skb_cgroup_id**\ ().
//
// 		This helper is available only if the kernel was compiled with
// 		the **CONFIG_SOCK_CGROUP_DATA** configuration option.
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// u64 bpf_sk_ancestor_cgroup_id(void *sk, int ancestor_level)
// 	Description
// 		Return id of cgroup v2 that is ancestor of cgroup associated
// 		with the *sk* at the *ancestor_level*.  The root cgroup is at
// 		*ancestor_level* zero and each step down the hierarchy
// 		increments the level. If *ancestor_level* == level of cgroup
// 		associated with *sk*, then return value will be same as that
// 		of **bpf_sk_cgroup_id**\ ().
//
// 		The helper is useful to implement policies based on cgroups
// 		that are upper in hierarchy than immediate cgroup associated
// 		with *sk*.
//
// 		The format of returned id and helper limitations are same as in
// 		**bpf_sk_cgroup_id**\ ().
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// long bpf_ringbuf_output(void *ringbuf, void *data, u64 size, u64 flags)
// 	Description
// 		Copy *size* bytes from *data* into a ring buffer *ringbuf*.
// 		If **BPF_RB_NO_WAKEUP** is specified in *flags*, no notification
// 		of new data availability is sent.
// 		If **BPF_RB_FORCE_WAKEUP** is specified in *flags*, notification
// 		of new data availability is sent unconditionally.
// 		If **0** is specified in *flags*, an adaptive notification
// 		of new data availability is sent.
//
// 		An adaptive notification is a notification sent whenever the user-space
// 		process has caught up and consumed all available payloads. In case the user-space
// 		process is still processing a previous payload, then no notification is needed
// 		as it will process the newly added payload automatically.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// void *bpf_ringbuf_reserve(void *ringbuf, u64 size, u64 flags)
// 	Description
// 		Reserve *size* bytes of payload in a ring buffer *ringbuf*.
// 		*flags* must be 0.
// 	Return
// 		Valid pointer with *size* bytes of memory available; NULL,
// 		otherwise.
//
// void bpf_ringbuf_submit(void *data, u64 flags)
// 	Description
// 		Submit reserved ring buffer sample, pointed to by *data*.
// 		If **BPF_RB_NO_WAKEUP** is specified in *flags*, no notification
// 		of new data availability is sent.
// 		If **BPF_RB_FORCE_WAKEUP** is specified in *flags*, notification
// 		of new data availability is sent unconditionally.
// 		If **0** is specified in *flags*, an adaptive notification
// 		of new data availability is sent.
//
// 		See 'bpf_ringbuf_output()' for the definition of adaptive notification.
// 	Return
// 		Nothing. Always succeeds.
//
// void bpf_ringbuf_discard(void *data, u64 flags)
// 	Description
// 		Discard reserved ring buffer sample, pointed to by *data*.
// 		If **BPF_RB_NO_WAKEUP** is specified in *flags*, no notification
// 		of new data availability is sent. Discarded records remain in
// 		the ring buffer until consumed by user space, so a later submit
// 		using adaptive wakeup might not wake up the consumer.
// 		If **BPF_RB_FORCE_WAKEUP** is specified in *flags*, notification
// 		of new data availability is sent unconditionally.
// 		If **0** is specified in *flags*, an adaptive notification
// 		of new data availability is sent.
//
// 		See 'bpf_ringbuf_output()' for the definition of adaptive notification.
// 	Return
// 		Nothing. Always succeeds.
//
// u64 bpf_ringbuf_query(void *ringbuf, u64 flags)
// 	Description
// 		Query various characteristics of provided ring buffer. What
// 		exactly is queries is determined by *flags*:
//
// 		* **BPF_RB_AVAIL_DATA**: Amount of data not yet consumed.
// 		* **BPF_RB_RING_SIZE**: The size of ring buffer.
// 		* **BPF_RB_CONS_POS**: Consumer position (can wrap around).
// 		* **BPF_RB_PROD_POS**: Producer(s) position (can wrap around).
// 		* **BPF_RB_OVERWRITE_POS**: Overwrite position (can wrap around).
//
// 		Data returned is just a momentary snapshot of actual values
// 		and could be inaccurate, so this facility should be used to
// 		power heuristics and for reporting, not to make 100% correct
// 		calculation.
// 	Return
// 		Requested value, or 0, if *flags* are not recognized.
//
// long bpf_csum_level(struct sk_buff *skb, u64 level)
// 	Description
// 		Change the skbs checksum level by one layer up or down, or
// 		reset it entirely to none in order to have the stack perform
// 		checksum validation. The level is applicable to the following
// 		protocols: TCP, UDP, GRE, SCTP, FCOE. For example, a decap of
// 		| ETH | IP | UDP | GUE | IP | TCP | into | ETH | IP | TCP |
// 		through **bpf_skb_adjust_room**\ () helper with passing in
// 		**BPF_F_ADJ_ROOM_NO_CSUM_RESET** flag would require one	call
// 		to **bpf_csum_level**\ () with **BPF_CSUM_LEVEL_DEC** since
// 		the UDP header is removed. Similarly, an encap of the latter
// 		into the former could be accompanied by a helper call to
// 		**bpf_csum_level**\ () with **BPF_CSUM_LEVEL_INC** if the
// 		skb is still intended to be processed in higher layers of the
// 		stack instead of just egressing at tc.
//
// 		There are three supported level settings at this time:
//
// 		* **BPF_CSUM_LEVEL_INC**: Increases skb->csum_level for skbs
// 		  with CHECKSUM_UNNECESSARY.
// 		* **BPF_CSUM_LEVEL_DEC**: Decreases skb->csum_level for skbs
// 		  with CHECKSUM_UNNECESSARY.
// 		* **BPF_CSUM_LEVEL_RESET**: Resets skb->csum_level to 0 and
// 		  sets CHECKSUM_NONE to force checksum validation by the stack.
// 		* **BPF_CSUM_LEVEL_QUERY**: No-op, returns the current
// 		  skb->csum_level.
// 	Return
// 		0 on success, or a negative error in case of failure. In the
// 		case of **BPF_CSUM_LEVEL_QUERY**, the current skb->csum_level
// 		is returned or the error code -EACCES in case the skb is not
// 		subject to CHECKSUM_UNNECESSARY.
//
// struct tcp6_sock *bpf_skc_to_tcp6_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp6_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct tcp_sock *bpf_skc_to_tcp_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct tcp_timewait_sock *bpf_skc_to_tcp_timewait_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp_timewait_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct tcp_request_sock *bpf_skc_to_tcp_request_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp_request_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct udp6_sock *bpf_skc_to_udp6_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *udp6_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// long bpf_get_task_stack(struct task_struct *task, void *buf, u32 size, u64 flags)
// 	Description
// 		Return a user or a kernel stack in bpf program provided buffer.
// 		Note: the user stack will only be populated if the *task* is
// 		the current task; all other tasks will return -EOPNOTSUPP.
// 		To achieve this, the helper needs *task*, which is a valid
// 		pointer to **struct task_struct**. To store the stacktrace, the
// 		bpf program provides *buf* with a nonnegative *size*.
//
// 		The last argument, *flags*, holds the number of stack frames to
// 		skip (from 0 to 255), masked with
// 		**BPF_F_SKIP_FIELD_MASK**. The next bits can be used to set
// 		the following flags:
//
// 		**BPF_F_USER_STACK**
// 			Collect a user space stack instead of a kernel stack.
// 			The *task* must be the current task.
// 		**BPF_F_USER_BUILD_ID**
// 			Collect buildid+offset instead of ips for user stack,
// 			only valid if **BPF_F_USER_STACK** is also specified.
//
// 		**bpf_get_task_stack**\ () can collect up to
// 		**PERF_MAX_STACK_DEPTH** both kernel and user frames, subject
// 		to sufficient large buffer size. Note that
// 		this limit can be controlled with the **sysctl** program, and
// 		that it should be manually increased in order to profile long
// 		user stacks (such as stacks for Java programs). To do so, use:
//
// 		::
//
// 			# sysctl kernel.perf_event_max_stack=<new value>
// 	Return
// 		The non-negative copied *buf* length equal to or less than
// 		*size* on success, or a negative error in case of failure.
//
// long bpf_load_hdr_opt(struct bpf_sock_ops *skops, void *searchby_res, u32 len, u64 flags)
// 	Description
// 		Load header option.  Support reading a particular TCP header
// 		option for bpf program (**BPF_PROG_TYPE_SOCK_OPS**).
//
// 		If *flags* is 0, it will search the option from the
// 		*skops*\ **->skb_data**.  The comment in **struct bpf_sock_ops**
// 		has details on what skb_data contains under different
// 		*skops*\ **->op**.
//
// 		The first byte of the *searchby_res* specifies the
// 		kind that it wants to search.
//
// 		If the searching kind is an experimental kind
// 		(i.e. 253 or 254 according to RFC6994).  It also
// 		needs to specify the "magic" which is either
// 		2 bytes or 4 bytes.  It then also needs to
// 		specify the size of the magic by using
// 		the 2nd byte which is "kind-length" of a TCP
// 		header option and the "kind-length" also
// 		includes the first 2 bytes "kind" and "kind-length"
// 		itself as a normal TCP header option also does.
//
// 		For example, to search experimental kind 254 with
// 		2 byte magic 0xeB9F, the searchby_res should be
// 		[ 254, 4, 0xeB, 0x9F, 0, 0, .... 0 ].
//
// 		To search for the standard window scale option (3),
// 		the *searchby_res* should be [ 3, 0, 0, .... 0 ].
// 		Note, kind-length must be 0 for regular option.
//
// 		Searching for No-Op (0) and End-of-Option-List (1) are
// 		not supported.
//
// 		*len* must be at least 2 bytes which is the minimal size
// 		of a header option.
//
// 		Supported flags:
//
// 		* **BPF_LOAD_HDR_OPT_TCP_SYN** to search from the
// 		  saved_syn packet or the just-received syn packet.
//
// 	Return
// 		> 0 when found, the header option is copied to *searchby_res*.
// 		The return value is the total length copied. On failure, a
// 		negative error code is returned:
//
// 		**-EINVAL** if a parameter is invalid.
//
// 		**-ENOMSG** if the option is not found.
//
// 		**-ENOENT** if no syn packet is available when
// 		**BPF_LOAD_HDR_OPT_TCP_SYN** is used.
//
// 		**-ENOSPC** if there is not enough space.  Only *len* number of
// 		bytes are copied.
//
// 		**-EFAULT** on failure to parse the header options in the
// 		packet.
//
// 		**-EPERM** if the helper cannot be used under the current
// 		*skops*\ **->op**.
//
// long bpf_store_hdr_opt(struct bpf_sock_ops *skops, const void *from, u32 len, u64 flags)
// 	Description
// 		Store header option.  The data will be copied
// 		from buffer *from* with length *len* to the TCP header.
//
// 		The buffer *from* should have the whole option that
// 		includes the kind, kind-length, and the actual
// 		option data.  The *len* must be at least kind-length
// 		long.  The kind-length does not have to be 4 byte
// 		aligned.  The kernel will take care of the padding
// 		and setting the 4 bytes aligned value to th->doff.
//
// 		This helper will check for duplicated option
// 		by searching the same option in the outgoing skb.
//
// 		This helper can only be called during
// 		**BPF_SOCK_OPS_WRITE_HDR_OPT_CB**.
//
// 	Return
// 		0 on success, or negative error in case of failure:
//
// 		**-EINVAL** If param is invalid.
//
// 		**-ENOSPC** if there is not enough space in the header.
// 		Nothing has been written
//
// 		**-EEXIST** if the option already exists.
//
// 		**-EFAULT** on failure to parse the existing header options.
//
// 		**-EPERM** if the helper cannot be used under the current
// 		*skops*\ **->op**.
//
// long bpf_reserve_hdr_opt(struct bpf_sock_ops *skops, u32 len, u64 flags)
// 	Description
// 		Reserve *len* bytes for the bpf header option.  The
// 		space will be used by **bpf_store_hdr_opt**\ () later in
// 		**BPF_SOCK_OPS_WRITE_HDR_OPT_CB**.
//
// 		If **bpf_reserve_hdr_opt**\ () is called multiple times,
// 		the total number of bytes will be reserved.
//
// 		This helper can only be called during
// 		**BPF_SOCK_OPS_HDR_OPT_LEN_CB**.
//
// 	Return
// 		0 on success, or negative error in case of failure:
//
// 		**-EINVAL** if a parameter is invalid.
//
// 		**-ENOSPC** if there is not enough space in the header.
//
// 		**-EPERM** if the helper cannot be used under the current
// 		*skops*\ **->op**.
//
// void *bpf_inode_storage_get(struct bpf_map *map, void *inode, void *value, u64 flags)
// 	Description
// 		Get a bpf_local_storage from an *inode*.
//
// 		Logically, it could be thought of as getting the value from
// 		a *map* with *inode* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *inode*) except this
// 		helper enforces the key must be an inode and the map must also
// 		be a **BPF_MAP_TYPE_INODE_STORAGE**.
//
// 		Underneath, the value is stored locally at *inode* instead of
// 		the *map*.  The *map* is used as the bpf-local-storage
// 		"type". The bpf-local-storage "type" (i.e. the *map*) is
// 		searched against all bpf_local_storage residing at *inode*.
//
// 		An optional *flags* (**BPF_LOCAL_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf_local_storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_LOCAL_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf_local_storage.  If *value* is
// 		**NULL**, the new bpf_local_storage will be zero initialized.
// 	Return
// 		A bpf_local_storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf_local_storage.
//
// int bpf_inode_storage_delete(struct bpf_map *map, void *inode)
// 	Description
// 		Delete a bpf_local_storage from an *inode*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf_local_storage cannot be found.
//
// long bpf_d_path(const struct path *path, char *buf, u32 sz)
// 	Description
// 		Return full path for given **struct path** object, which
// 		needs to be the kernel BTF *path* object. The path is
// 		returned in the provided buffer *buf* of size *sz* and
// 		is zero terminated.
//
// 	Return
// 		On success, the strictly positive length of the string,
// 		including the trailing NUL character. On error, a negative
// 		value.
//
// long bpf_copy_from_user(void *dst, u32 size, const void *user_ptr)
// 	Description
// 		Read *size* bytes from user space address *user_ptr* and store
// 		the data in *dst*. This is a wrapper of **copy_from_user**\ ().
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_snprintf_btf(char *str, u32 str_size, struct btf_ptr *ptr, u32 btf_ptr_size, u64 flags)
// 	Description
// 		Use BTF to store a string representation of *ptr*->ptr in *str*,
// 		using *ptr*->type_id.  This value should specify the type
// 		that *ptr*->ptr points to. LLVM __builtin_btf_type_id(type, 1)
// 		can be used to look up vmlinux BTF type ids. Traversing the
// 		data structure using BTF, the type information and values are
// 		stored in the first *str_size* - 1 bytes of *str*.  Safe copy of
// 		the pointer data is carried out to avoid kernel crashes during
// 		operation.  Smaller types can use string space on the stack;
// 		larger programs can use map data to store the string
// 		representation.
//
// 		The string can be subsequently shared with userspace via
// 		bpf_perf_event_output() or ring buffer interfaces.
// 		bpf_trace_printk() is to be avoided as it places too small
// 		a limit on string size to be useful.
//
// 		*flags* is a combination of
//
// 		**BTF_F_COMPACT**
// 			no formatting around type information
// 		**BTF_F_NONAME**
// 			no struct/union member names/types
// 		**BTF_F_PTR_RAW**
// 			show raw (unobfuscated) pointer values;
// 			equivalent to printk specifier %px.
// 		**BTF_F_ZERO**
// 			show zero-valued struct/union members; they
// 			are not displayed by default
//
// 	Return
// 		The number of bytes that were written (or would have been
// 		written if output had to be truncated due to string size),
// 		or a negative error in cases of failure.
//
// long bpf_seq_printf_btf(struct seq_file *m, struct btf_ptr *ptr, u32 ptr_size, u64 flags)
// 	Description
// 		Use BTF to write to seq_write a string representation of
// 		*ptr*->ptr, using *ptr*->type_id as per bpf_snprintf_btf().
// 		*flags* are identical to those used for bpf_snprintf_btf.
// 	Return
// 		0 on success or a negative error in case of failure.
//
// u64 bpf_skb_cgroup_classid(struct sk_buff *skb)
// 	Description
// 		See **bpf_get_cgroup_classid**\ () for the main description.
// 		This helper differs from **bpf_get_cgroup_classid**\ () in that
// 		the cgroup v1 net_cls class is retrieved only from the *skb*'s
// 		associated socket instead of the current process.
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// long bpf_redirect_neigh(u32 ifindex, struct bpf_redir_neigh *params, int plen, u64 flags)
// 	Description
// 		Redirect the packet to another net device of index *ifindex*
// 		and fill in L2 addresses from neighboring subsystem. This helper
// 		is somewhat similar to **bpf_redirect**\ (), except that it
// 		populates L2 addresses as well, meaning, internally, the helper
// 		relies on the neighbor lookup for the L2 address of the nexthop.
//
// 		The helper will perform a FIB lookup based on the skb's
// 		networking header to get the address of the next hop, unless
// 		this is supplied by the caller in the *params* argument. The
// 		*plen* argument indicates the len of *params* and should be set
// 		to 0 if *params* is NULL.
//
// 		The *flags* argument is reserved and must be 0. The helper is
// 		currently only supported for tc BPF program types, and enabled
// 		for IPv4 and IPv6 protocols.
// 	Return
// 		The helper returns **TC_ACT_REDIRECT** on success or
// 		**TC_ACT_SHOT** on error.
//
// void *bpf_per_cpu_ptr(const void *percpu_ptr, u32 cpu)
//     Description
//             Take a pointer to a percpu ksym, *percpu_ptr*, and return a
//             pointer to the percpu kernel variable on *cpu*. A ksym is an
//             extern variable decorated with '__ksym'. For ksym, there is a
//             global var (either static or global) defined of the same name
//             in the kernel. The ksym is percpu if the global var is percpu.
//             The returned pointer points to the global percpu var on *cpu*.
//
//             bpf_per_cpu_ptr() has the same semantic as per_cpu_ptr() in the
//             kernel, except that bpf_per_cpu_ptr() may return NULL. This
//             happens if *cpu* is larger than nr_cpu_ids. The caller of
//             bpf_per_cpu_ptr() must check the returned value.
//     Return
//             A pointer pointing to the kernel percpu variable on *cpu*, or
//             NULL, if *cpu* is invalid.
//
// void *bpf_this_cpu_ptr(const void *percpu_ptr)
// 	Description
// 		Take a pointer to a percpu ksym, *percpu_ptr*, and return a
// 		pointer to the percpu kernel variable on this cpu. See the
// 		description of 'ksym' in **bpf_per_cpu_ptr**\ ().
//
// 		bpf_this_cpu_ptr() has the same semantic as this_cpu_ptr() in
// 		the kernel. Different from **bpf_per_cpu_ptr**\ (), it would
// 		never return NULL.
// 	Return
// 		A pointer pointing to the kernel percpu variable on this cpu.
//
// long bpf_redirect_peer(u32 ifindex, u64 flags)
// 	Description
// 		Redirect the packet to another net device of index *ifindex*.
// 		This helper is somewhat similar to **bpf_redirect**\ (), except
// 		that the redirection happens to the *ifindex*' peer device. If
// 		*flags* is 0, the netns switch takes place from ingress to
// 		ingress without going through the CPU's backlog queue. If the
// 		**BPF_F_EGRESS** flag is provided then redirection happens in
// 		the egress direction of the peer device.
//
// 		*skb*\ **->mark** and *skb*\ **->tstamp** are not cleared during
// 		the netns switch.
//
// 		If the *flags* argument is 0, the helper is currently only
// 		supported for tc BPF program types at the ingress hook and for
// 		veth and netkit target device types. The peer device must reside
// 		in a different network namespace.
// 	Return
// 		The helper returns **TC_ACT_REDIRECT** on success or
// 		**TC_ACT_SHOT** on error.
//
// void *bpf_task_storage_get(struct bpf_map *map, struct task_struct *task, void *value, u64 flags)
// 	Description
// 		Get a bpf_local_storage from the *task*.
//
// 		Logically, it could be thought of as getting the value from
// 		a *map* with *task* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *task*) except this
// 		helper enforces the key must be a task_struct and the map must also
// 		be a **BPF_MAP_TYPE_TASK_STORAGE**.
//
// 		Underneath, the value is stored locally at *task* instead of
// 		the *map*.  The *map* is used as the bpf-local-storage
// 		"type". The bpf-local-storage "type" (i.e. the *map*) is
// 		searched against all bpf_local_storage residing at *task*.
//
// 		An optional *flags* (**BPF_LOCAL_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf_local_storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_LOCAL_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf_local_storage.  If *value* is
// 		**NULL**, the new bpf_local_storage will be zero initialized.
// 	Return
// 		A bpf_local_storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf_local_storage.
//
// long bpf_task_storage_delete(struct bpf_map *map, struct task_struct *task)
// 	Description
// 		Delete a bpf_local_storage from a *task*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf_local_storage cannot be found.
//
// struct task_struct *bpf_get_current_task_btf(void)
// 	Description
// 		Return a BTF pointer to the "current" task.
// 		This pointer can also be used in helpers that accept an
// 		*ARG_PTR_TO_BTF_ID* of type *task_struct*.
// 	Return
// 		Pointer to the current task.
//
// long bpf_bprm_opts_set(struct linux_binprm *bprm, u64 flags)
// 	Description
// 		Set or clear certain options on *bprm*:
//
// 		**BPF_F_BPRM_SECUREEXEC** Set the secureexec bit
// 		which sets the **AT_SECURE** auxv for glibc. The bit
// 		is cleared if the flag is not specified.
// 	Return
// 		**-EINVAL** if invalid *flags* are passed, zero otherwise.
//
// u64 bpf_ktime_get_coarse_ns(void)
// 	Description
// 		Return a coarse-grained version of the time elapsed since
// 		system boot, in nanoseconds. Does not include time the system
// 		was suspended.
//
// 		See: **clock_gettime**\ (**CLOCK_MONOTONIC_COARSE**)
// 	Return
// 		Current *ktime*.
//
// long bpf_ima_inode_hash(struct inode *inode, void *dst, u32 size)
// 	Description
// 		Returns the stored IMA hash of the *inode* (if it's available).
// 		If the hash is larger than *size*, then only *size*
// 		bytes will be copied to *dst*
// 	Return
// 		The **hash_algo** is returned on success,
// 		**-EOPNOTSUPP** if IMA is disabled or **-EINVAL** if
// 		invalid arguments are passed.
//
// struct socket *bpf_sock_from_file(struct file *file)
// 	Description
// 		If the given file represents a socket, returns the associated
// 		socket.
// 	Return
// 		A pointer to a struct socket on success or NULL if the file is
// 		not a socket.
//
// long bpf_check_mtu(void *ctx, u32 ifindex, u32 *mtu_len, s32 len_diff, u64 flags)
// 	Description
// 		Check packet size against exceeding MTU of net device (based
// 		on *ifindex*).  This helper will likely be used in combination
// 		with helpers that adjust/change the packet size.
//
// 		The argument *len_diff* can be used for querying with a planned
// 		size change. This allows to check MTU prior to changing packet
// 		ctx. Providing a *len_diff* adjustment that is larger than the
// 		actual packet size (resulting in negative packet size) will in
// 		principle not exceed the MTU, which is why it is not considered
// 		a failure.  Other BPF helpers are needed for performing the
// 		planned size change; therefore the responsibility for catching
// 		a negative packet size belongs in those helpers.
//
// 		Specifying *ifindex* zero means the MTU check is performed
// 		against the current net device.  This is practical if this isn't
// 		used prior to redirect.
//
// 		On input *mtu_len* must be a valid pointer, else verifier will
// 		reject BPF program.  If the value *mtu_len* is initialized to
// 		zero then the ctx packet size is use.  When value *mtu_len* is
// 		provided as input this specify the L3 length that the MTU check
// 		is done against. Remember XDP and TC length operate at L2, but
// 		this value is L3 as this correlate to MTU and IP-header tot_len
// 		values which are L3 (similar behavior as bpf_fib_lookup).
//
// 		The Linux kernel route table can configure MTUs on a more
// 		specific per route level, which is not provided by this helper.
// 		For route level MTU checks use the **bpf_fib_lookup**\ ()
// 		helper.
//
// 		*ctx* is either **struct xdp_md** for XDP programs or
// 		**struct sk_buff** for tc cls_act programs.
//
// 		The *flags* argument can be a combination of one or more of the
// 		following values:
//
// 		**BPF_MTU_CHK_SEGS**
// 			This flag will only works for *ctx* **struct sk_buff**.
// 			If packet context contains extra packet segment buffers
// 			(often knows as GSO skb), then MTU check is harder to
// 			check at this point, because in transmit path it is
// 			possible for the skb packet to get re-segmented
// 			(depending on net device features).  This could still be
// 			a MTU violation, so this flag enables performing MTU
// 			check against segments, with a different violation
// 			return code to tell it apart. Check cannot use len_diff.
//
// 		On return *mtu_len* pointer contains the MTU value of the net
// 		device.  Remember the net device configured MTU is the L3 size,
// 		which is returned here and XDP and TC length operate at L2.
// 		Helper take this into account for you, but remember when using
// 		MTU value in your BPF-code.
//
// 	Return
// 		* 0 on success, and populate MTU value in *mtu_len* pointer.
//
// 		* < 0 if any input argument is invalid (*mtu_len* not updated)
//
// 		MTU violations return positive values, but also populate MTU
// 		value in *mtu_len* pointer, as this can be needed for
// 		implementing PMTU handing:
//
// 		* **BPF_MTU_CHK_RET_FRAG_NEEDED**
// 		* **BPF_MTU_CHK_RET_SEGS_TOOBIG**
//
// long bpf_for_each_map_elem(struct bpf_map *map, void *callback_fn, void *callback_ctx, u64 flags)
// 	Description
// 		For each element in **map**, call **callback_fn** function with
// 		**map**, **callback_ctx** and other map-specific parameters.
// 		The **callback_fn** should be a static function and
// 		the **callback_ctx** should be a pointer to the stack.
// 		The **flags** is used to control certain aspects of the helper.
// 		Currently, the **flags** must be 0.
//
// 		The following are a list of supported map types and their
// 		respective expected callback signatures:
//
// 		BPF_MAP_TYPE_HASH, BPF_MAP_TYPE_PERCPU_HASH,
// 		BPF_MAP_TYPE_LRU_HASH, BPF_MAP_TYPE_LRU_PERCPU_HASH,
// 		BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_PERCPU_ARRAY
//
// 		long (\*callback_fn)(struct bpf_map \*map, const void \*key, void \*value, void \*ctx);
//
// 		For per_cpu maps, the map_value is the value on the cpu where the
// 		bpf_prog is running.
//
// 		If **callback_fn** return 0, the helper will continue to the next
// 		element. If return value is 1, the helper will skip the rest of
// 		elements and return. Other return values are not used now.
//
// 	Return
// 		The number of traversed map elements for success, **-EINVAL** for
// 		invalid **flags**.
//
// long bpf_snprintf(char *str, u32 str_size, const char *fmt, u64 *data, u32 data_len)
// 	Description
// 		Outputs a string into the **str** buffer of size **str_size**
// 		based on a format string stored in a read-only map pointed by
// 		**fmt**.
//
// 		Each format specifier in **fmt** corresponds to one u64 element
// 		in the **data** array. For strings and pointers where pointees
// 		are accessed, only the pointer values are stored in the *data*
// 		array. The *data_len* is the size of *data* in bytes - must be
// 		a multiple of 8.
//
// 		Formats **%s** and **%p{i,I}{4,6}** require to read kernel
// 		memory. Reading kernel memory may fail due to either invalid
// 		address or valid address but requiring a major memory fault. If
// 		reading kernel memory fails, the string for **%s** will be an
// 		empty string, and the ip address for **%p{i,I}{4,6}** will be 0.
// 		Not returning error to bpf program is consistent with what
// 		**bpf_trace_printk**\ () does for now.
//
// 	Return
// 		The strictly positive length of the formatted string, including
// 		the trailing zero character. If the return value is greater than
// 		**str_size**, **str** contains a truncated string, guaranteed to
// 		be zero-terminated except when **str_size** is 0.
//
// 		Or **-EBUSY** if the per-CPU memory copy buffer is busy.
//
// long bpf_sys_bpf(u32 cmd, void *attr, u32 attr_size)
// 	Description
// 		Execute bpf syscall with given arguments.
// 	Return
// 		A syscall result.
//
// long bpf_btf_find_by_name_kind(char *name, int name_sz, u32 kind, int flags)
// 	Description
// 		Find BTF type with given name and kind in vmlinux BTF or in module's BTFs.
// 	Return
// 		Returns btf_id and btf_obj_fd in lower and upper 32 bits.
//
// long bpf_sys_close(u32 fd)
// 	Description
// 		Execute close syscall for given FD.
// 	Return
// 		A syscall result.
//
// long bpf_timer_init(struct bpf_timer *timer, struct bpf_map *map, u64 flags)
// 	Description
// 		Initialize the timer.
// 		First 4 bits of *flags* specify clockid.
// 		Only CLOCK_MONOTONIC, CLOCK_REALTIME, CLOCK_BOOTTIME are allowed.
// 		All other bits of *flags* are reserved.
// 		The verifier will reject the program if *timer* is not from
// 		the same *map*.
// 	Return
// 		0 on success.
// 		**-EBUSY** if *timer* is already initialized.
// 		**-EINVAL** if invalid *flags* are passed.
// 		**-EPERM** if *timer* is in a map that doesn't have any user references.
// 		The user space should either hold a file descriptor to a map with timers
// 		or pin such map in bpffs. When map is unpinned or file descriptor is
// 		closed all timers in the map will be cancelled and freed.
//
// long bpf_timer_set_callback(struct bpf_timer *timer, void *callback_fn)
// 	Description
// 		Configure the timer to call *callback_fn* static function.
// 	Return
// 		0 on success.
// 		**-EINVAL** if *timer* was not initialized with bpf_timer_init() earlier.
// 		**-EPERM** if *timer* is in a map that doesn't have any user references.
// 		The user space should either hold a file descriptor to a map with timers
// 		or pin such map in bpffs. When map is unpinned or file descriptor is
// 		closed all timers in the map will be cancelled and freed.
//
// long bpf_timer_start(struct bpf_timer *timer, u64 nsecs, u64 flags)
// 	Description
// 		Set timer expiration N nanoseconds from the current time. The
// 		configured callback will be invoked in soft irq context on some cpu
// 		and will not repeat unless another bpf_timer_start() is made.
// 		In such case the next invocation can migrate to a different cpu.
// 		Since struct bpf_timer is a field inside map element the map
// 		owns the timer. The bpf_timer_set_callback() will increment refcnt
// 		of BPF program to make sure that callback_fn code stays valid.
// 		When user space reference to a map reaches zero all timers
// 		in a map are cancelled and corresponding program's refcnts are
// 		decremented. This is done to make sure that Ctrl-C of a user
// 		process doesn't leave any timers running. If map is pinned in
// 		bpffs the callback_fn can re-arm itself indefinitely.
// 		bpf_map_update/delete_elem() helpers and user space sys_bpf commands
// 		cancel and free the timer in the given map element.
// 		The map can contain timers that invoke callback_fn-s from different
// 		programs. The same callback_fn can serve different timers from
// 		different maps if key/value layout matches across maps.
// 		Every bpf_timer_set_callback() can have different callback_fn.
//
// 		*flags* can be one of:
//
// 		**BPF_F_TIMER_ABS**
// 			Start the timer in absolute expire value instead of the
// 			default relative one.
// 		**BPF_F_TIMER_CPU_PIN**
// 			Timer will be pinned to the CPU of the caller.
//
// 	Return
// 		0 on success.
// 		**-EINVAL** if *timer* was not initialized with bpf_timer_init() earlier
// 		or invalid *flags* are passed.
//
// long bpf_timer_cancel(struct bpf_timer *timer)
// 	Description
// 		Cancel the timer and wait for callback_fn to finish if it was running.
// 	Return
// 		0 if the timer was not active.
// 		1 if the timer was active.
// 		**-EINVAL** if *timer* was not initialized with bpf_timer_init() earlier.
// 		**-EDEADLK** if callback_fn tried to call bpf_timer_cancel() on its
// 		own timer which would have led to a deadlock otherwise.
//
// u64 bpf_get_func_ip(void *ctx)
// 	Description
// 		Get address of the traced function (for tracing and kprobe programs).
//
// 		When called for kprobe program attached as uprobe it returns
// 		probe address for both entry and return uprobe.
//
// 	Return
// 		Address of the traced function for kprobe.
// 		0 for kprobes placed within the function (not at the entry).
// 		Address of the probe for uprobe and return uprobe.
//
// u64 bpf_get_attach_cookie(void *ctx)
// 	Description
// 		Get bpf_cookie value provided (optionally) during the program
// 		attachment. It might be different for each individual
// 		attachment, even if BPF program itself is the same.
// 		Expects BPF program context *ctx* as a first argument.
//
// 		Supported for the following program types:
// 			- kprobe/uprobe;
// 			- tracepoint;
// 			- perf_event.
// 	Return
// 		Value specified by user at BPF link creation/attachment time
// 		or 0, if it was not specified.
//
// long bpf_task_pt_regs(struct task_struct *task)
// 	Description
// 		Get the struct pt_regs associated with **task**.
// 	Return
// 		A pointer to struct pt_regs.
//
// long bpf_get_branch_snapshot(void *entries, u32 size, u64 flags)
// 	Description
// 		Get branch trace from hardware engines like Intel LBR. The
// 		hardware engine is stopped shortly after the helper is
// 		called. Therefore, the user need to filter branch entries
// 		based on the actual use case. To capture branch trace
// 		before the trigger point of the BPF program, the helper
// 		should be called at the beginning of the BPF program.
//
// 		The data is stored as struct perf_branch_entry into output
// 		buffer *entries*. *size* is the size of *entries* in bytes.
// 		*flags* is reserved for now and must be zero.
//
// 	Return
// 		On success, number of bytes written to *buf*. On error, a
// 		negative value.
//
// 		**-EINVAL** if *flags* is not zero.
//
// 		**-ENOENT** if architecture does not support branch records.
//
// long bpf_trace_vprintk(const char *fmt, u32 fmt_size, const void *data, u32 data_len)
// 	Description
// 		Behaves like **bpf_trace_printk**\ () helper, but takes an array of u64
// 		to format and can handle more format args as a result.
//
// 		Arguments are to be used as in **bpf_seq_printf**\ () helper.
// 	Return
// 		The number of bytes written to the buffer, or a negative error
// 		in case of failure.
//
// struct unix_sock *bpf_skc_to_unix_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *unix_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// long bpf_kallsyms_lookup_name(const char *name, int name_sz, int flags, u64 *res)
// 	Description
// 		Get the address of a kernel symbol, returned in *res*. *res* is
// 		set to 0 if the symbol is not found.
// 	Return
// 		On success, zero. On error, a negative value.
//
// 		**-EINVAL** if *flags* is not zero.
//
// 		**-EINVAL** if string *name* is not the same size as *name_sz*.
//
// 		**-ENOENT** if symbol is not found.
//
// 		**-EPERM** if caller does not have permission to obtain kernel address.
//
// long bpf_find_vma(struct task_struct *task, u64 addr, void *callback_fn, void *callback_ctx, u64 flags)
// 	Description
// 		Find vma of *task* that contains *addr*, call *callback_fn*
// 		function with *task*, *vma*, and *callback_ctx*.
// 		The *callback_fn* should be a static function and
// 		the *callback_ctx* should be a pointer to the stack.
// 		The *flags* is used to control certain aspects of the helper.
// 		Currently, the *flags* must be 0.
//
// 		The expected callback signature is
//
// 		long (\*callback_fn)(struct task_struct \*task, struct vm_area_struct \*vma, void \*callback_ctx);
//
// 	Return
// 		0 on success.
// 		**-ENOENT** if *task->mm* is NULL, or no vma contains *addr*.
// 		**-EBUSY** if failed to try lock mmap_lock.
// 		**-EINVAL** for invalid **flags**.
//
// long bpf_loop(u32 nr_loops, void *callback_fn, void *callback_ctx, u64 flags)
// 	Description
// 		For **nr_loops**, call **callback_fn** function
// 		with **callback_ctx** as the context parameter.
// 		The **callback_fn** should be a static function and
// 		the **callback_ctx** should be a pointer to the stack.
// 		The **flags** is used to control certain aspects of the helper.
// 		Currently, the **flags** must be 0. Currently, nr_loops is
// 		limited to 1 << 23 (~8 million) loops.
//
// 		long (\*callback_fn)(u64 index, void \*ctx);
//
// 		where **index** is the current index in the loop. The index
// 		is zero-indexed.
//
// 		If **callback_fn** returns 0, the helper will continue to the next
// 		loop. If return value is 1, the helper will skip the rest of
// 		the loops and return. Other return values are not used now,
// 		and will be rejected by the verifier.
//
// 	Return
// 		The number of loops performed, **-EINVAL** for invalid **flags**,
// 		**-E2BIG** if **nr_loops** exceeds the maximum number of loops.
//
// long bpf_strncmp(const char *s1, u32 s1_sz, const char *s2)
// 	Description
// 		Do strncmp() between **s1** and **s2**. **s1** doesn't need
// 		to be null-terminated and **s1_sz** is the maximum storage
// 		size of **s1**. **s2** must be a read-only string.
// 	Return
// 		An integer less than, equal to, or greater than zero
// 		if the first **s1_sz** bytes of **s1** is found to be
// 		less than, to match, or be greater than **s2**.
//
// long bpf_get_func_arg(void *ctx, u32 n, u64 *value)
// 	Description
// 		Get **n**-th argument register (zero based) of the traced function (for tracing programs)
// 		returned in **value**.
//
// 	Return
// 		0 on success.
// 		**-EINVAL** if n >= argument register count of traced function.
//
// long bpf_get_func_ret(void *ctx, u64 *value)
// 	Description
// 		Get return value of the traced function (for tracing programs)
// 		in **value**.
//
// 	Return
// 		0 on success.
// 		**-EOPNOTSUPP** for tracing programs other than BPF_TRACE_FEXIT or BPF_MODIFY_RETURN.
//
// long bpf_get_func_arg_cnt(void *ctx)
// 	Description
// 		Get number of registers of the traced function (for tracing programs) where
// 		function arguments are stored in these registers.
//
// 	Return
// 		The number of argument registers of the traced function.
//
// int bpf_get_retval(void)
// 	Description
// 		Get the BPF program's return value that will be returned to the upper layers.
//
// 		This helper is currently supported by cgroup programs and only by the hooks
// 		where BPF program's return value is returned to the userspace via errno.
// 	Return
// 		The BPF program's return value.
//
// int bpf_set_retval(int retval)
// 	Description
// 		Set the BPF program's return value that will be returned to the upper layers.
//
// 		This helper is currently supported by cgroup programs and only by the hooks
// 		where BPF program's return value is returned to the userspace via errno.
//
// 		Note that there is the following corner case where the program exports an error
// 		via bpf_set_retval but signals success via 'return 1':
//
// 			bpf_set_retval(-EPERM);
// 			return 1;
//
// 		In this case, the BPF program's return value will use helper's -EPERM. This
// 		still holds true for cgroup/bind{4,6} which supports extra 'return 3' success case.
//
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_xdp_get_buff_len(struct xdp_buff *xdp_md)
// 	Description
// 		Get the total size of a given xdp buff (linear and paged area)
// 	Return
// 		The total size of a given xdp buffer.
//
// long bpf_xdp_load_bytes(struct xdp_buff *xdp_md, u32 offset, void *buf, u32 len)
// 	Description
// 		This helper is provided as an easy way to load data from a
// 		xdp buffer. It can be used to load *len* bytes from *offset* from
// 		the frame associated to *xdp_md*, into the buffer pointed by
// 		*buf*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_store_bytes(struct xdp_buff *xdp_md, u32 offset, void *buf, u32 len)
// 	Description
// 		Store *len* bytes from buffer *buf* into the frame
// 		associated to *xdp_md*, at *offset*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_copy_from_user_task(void *dst, u32 size, const void *user_ptr, struct task_struct *tsk, u64 flags)
// 	Description
// 		Read *size* bytes from user space address *user_ptr* in *tsk*'s
// 		address space, and stores the data in *dst*. *flags* is not
// 		used yet and is provided for future extensibility. This helper
// 		can only be used by sleepable programs.
// 	Return
// 		0 on success, or a negative error in case of failure. On error
// 		*dst* buffer is zeroed out.
//
// long bpf_skb_set_tstamp(struct sk_buff *skb, u64 tstamp, u32 tstamp_type)
// 	Description
// 		Change the __sk_buff->tstamp_type to *tstamp_type*
// 		and set *tstamp* to the __sk_buff->tstamp together.
//
// 		If there is no need to change the __sk_buff->tstamp_type,
// 		the tstamp value can be directly written to __sk_buff->tstamp
// 		instead.
//
// 		BPF_SKB_TSTAMP_DELIVERY_MONO is the only tstamp that
// 		will be kept during bpf_redirect_*().  A non zero
// 		*tstamp* must be used with the BPF_SKB_TSTAMP_DELIVERY_MONO
// 		*tstamp_type*.
//
// 		A BPF_SKB_TSTAMP_UNSPEC *tstamp_type* can only be used
// 		with a zero *tstamp*.
//
// 		Only IPv4 and IPv6 skb->protocol are supported.
//
// 		This function is most useful when it needs to set a
// 		mono delivery time to __sk_buff->tstamp and then
// 		bpf_redirect_*() to the egress of an iface.  For example,
// 		changing the (rcv) timestamp in __sk_buff->tstamp at
// 		ingress to a mono delivery time and then bpf_redirect_*()
// 		to sch_fq@phy-dev.
// 	Return
// 		0 on success.
// 		**-EINVAL** for invalid input
// 		**-EOPNOTSUPP** for unsupported protocol
//
// long bpf_ima_file_hash(struct file *file, void *dst, u32 size)
// 	Description
// 		Returns a calculated IMA hash of the *file*.
// 		If the hash is larger than *size*, then only *size*
// 		bytes will be copied to *dst*
// 	Return
// 		The **hash_algo** is returned on success,
// 		**-EOPNOTSUPP** if the hash calculation failed or **-EINVAL** if
// 		invalid arguments are passed.
//
// void *bpf_kptr_xchg(void *dst, void *ptr)
// 	Description
// 		Exchange kptr at pointer *dst* with *ptr*, and return the old value.
// 		*dst* can be map value or local kptr. *ptr* can be NULL, otherwise
// 		it must be a referenced pointer which will be released when this helper
// 		is called.
// 	Return
// 		The old value of kptr (which can be NULL). The returned pointer
// 		if not NULL, is a reference which must be released using its
// 		corresponding release function, or moved into a BPF map before
// 		program exit.
//
// void *bpf_map_lookup_percpu_elem(struct bpf_map *map, const void *key, u32 cpu)
// 	Description
// 		Perform a lookup in *percpu map* for an entry associated to
// 		*key* on *cpu*.
// 	Return
// 		Map value associated to *key* on *cpu*, or **NULL** if no entry
// 		was found or *cpu* is invalid.
//
// struct mptcp_sock *bpf_skc_to_mptcp_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *mptcp_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// long bpf_dynptr_from_mem(void *data, u64 size, u64 flags, struct bpf_dynptr *ptr)
// 	Description
// 		Get a dynptr to local memory *data*.
//
// 		*data* must be a ptr to a map value.
// 		The maximum *size* supported is DYNPTR_MAX_SIZE.
// 		*flags* is currently unused.
// 	Return
// 		0 on success, -E2BIG if the size exceeds DYNPTR_MAX_SIZE,
// 		-EINVAL if flags is not 0.
//
// long bpf_ringbuf_reserve_dynptr(void *ringbuf, u32 size, u64 flags, struct bpf_dynptr *ptr)
// 	Description
// 		Reserve *size* bytes of payload in a ring buffer *ringbuf*
// 		through the dynptr interface. *flags* must be 0.
//
// 		Please note that a corresponding bpf_ringbuf_submit_dynptr or
// 		bpf_ringbuf_discard_dynptr must be called on *ptr*, even if the
// 		reservation fails. This is enforced by the verifier.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// void bpf_ringbuf_submit_dynptr(struct bpf_dynptr *ptr, u64 flags)
// 	Description
// 		Submit reserved ring buffer sample, pointed to by *data*,
// 		through the dynptr interface. This is a no-op if the dynptr is
// 		invalid/null.
//
// 		For more information on *flags*, please see
// 		'bpf_ringbuf_submit'.
// 	Return
// 		Nothing. Always succeeds.
//
// void bpf_ringbuf_discard_dynptr(struct bpf_dynptr *ptr, u64 flags)
// 	Description
// 		Discard reserved ring buffer sample through the dynptr
// 		interface. This is a no-op if the dynptr is invalid/null.
//
// 		For more information on *flags*, please see
// 		'bpf_ringbuf_discard'.
// 	Return
// 		Nothing. Always succeeds.
//
// long bpf_dynptr_read(void *dst, u64 len, const struct bpf_dynptr *src, u64 offset, u64 flags)
// 	Description
// 		Read *len* bytes from *src* into *dst*, starting from *offset*
// 		into *src*.
// 		*flags* is currently unused.
// 	Return
// 		0 on success, -E2BIG if *offset* + *len* exceeds the length
// 		of *src*'s data, -EINVAL if *src* is an invalid dynptr or if
// 		*flags* is not 0.
//
// long bpf_dynptr_write(const struct bpf_dynptr *dst, u64 offset, void *src, u64 len, u64 flags)
// 	Description
// 		Write *len* bytes from *src* into *dst*, starting from *offset*
// 		into *dst*.
//
// 		*flags* must be 0 except for skb-type dynptrs.
//
// 		For skb-type dynptrs:
// 		    *  All data slices of the dynptr are automatically
// 		       invalidated after **bpf_dynptr_write**\ (). This is
// 		       because writing may pull the skb and change the
// 		       underlying packet buffer.
//
// 		    *  For *flags*, please see the flags accepted by
// 		       **bpf_skb_store_bytes**\ ().
// 	Return
// 		0 on success, -E2BIG if *offset* + *len* exceeds the length
// 		of *dst*'s data, -EINVAL if *dst* is an invalid dynptr or if *dst*
// 		is a read-only dynptr or if *flags* is not correct. For skb-type dynptrs,
// 		other errors correspond to errors returned by **bpf_skb_store_bytes**\ ().
//
// void *bpf_dynptr_data(const struct bpf_dynptr *ptr, u64 offset, u64 len)
// 	Description
// 		Get a pointer to the underlying dynptr data.
//
// 		*len* must be a statically known value. The returned data slice
// 		is invalidated whenever the dynptr is invalidated.
//
// 		skb and xdp type dynptrs may not use bpf_dynptr_data. They should
// 		instead use bpf_dynptr_slice and bpf_dynptr_slice_rdwr.
// 	Return
// 		Pointer to the underlying dynptr data, NULL if the dynptr is
// 		read-only, if the dynptr is invalid, or if the offset and length
// 		is out of bounds.
//
// s64 bpf_tcp_raw_gen_syncookie_ipv4(struct iphdr *iph, struct tcphdr *th, u32 th_len)
// 	Description
// 		Try to issue a SYN cookie for the packet with corresponding
// 		IPv4/TCP headers, *iph* and *th*, without depending on a
// 		listening socket.
//
// 		*iph* points to the IPv4 header.
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		On success, lower 32 bits hold the generated SYN cookie in
// 		followed by 16 bits which hold the MSS value for that cookie,
// 		and the top 16 bits are unused.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EINVAL** if *th_len* is invalid.
//
// s64 bpf_tcp_raw_gen_syncookie_ipv6(struct ipv6hdr *iph, struct tcphdr *th, u32 th_len)
// 	Description
// 		Try to issue a SYN cookie for the packet with corresponding
// 		IPv6/TCP headers, *iph* and *th*, without depending on a
// 		listening socket.
//
// 		*iph* points to the IPv6 header.
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		On success, lower 32 bits hold the generated SYN cookie in
// 		followed by 16 bits which hold the MSS value for that cookie,
// 		and the top 16 bits are unused.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EINVAL** if *th_len* is invalid.
//
// 		**-EPROTONOSUPPORT** if CONFIG_IPV6 is not builtin.
//
// long bpf_tcp_raw_check_syncookie_ipv4(struct iphdr *iph, struct tcphdr *th)
// 	Description
// 		Check whether *iph* and *th* contain a valid SYN cookie ACK
// 		without depending on a listening socket.
//
// 		*iph* points to the IPv4 header.
//
// 		*th* points to the TCP header.
// 	Return
// 		0 if *iph* and *th* are a valid SYN cookie ACK.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EACCES** if the SYN cookie is not valid.
//
// long bpf_tcp_raw_check_syncookie_ipv6(struct ipv6hdr *iph, struct tcphdr *th)
// 	Description
// 		Check whether *iph* and *th* contain a valid SYN cookie ACK
// 		without depending on a listening socket.
//
// 		*iph* points to the IPv6 header.
//
// 		*th* points to the TCP header.
// 	Return
// 		0 if *iph* and *th* are a valid SYN cookie ACK.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EACCES** if the SYN cookie is not valid.
//
// 		**-EPROTONOSUPPORT** if CONFIG_IPV6 is not builtin.
//
// u64 bpf_ktime_get_tai_ns(void)
// 	Description
// 		A nonsettable system-wide clock derived from wall-clock time but
// 		ignoring leap seconds.  This clock does not experience
// 		discontinuities and backwards jumps caused by NTP inserting leap
// 		seconds as CLOCK_REALTIME does.
//
// 		See: **clock_gettime**\ (**CLOCK_TAI**)
// 	Return
// 		Current *ktime*.
//
// long bpf_user_ringbuf_drain(struct bpf_map *map, void *callback_fn, void *ctx, u64 flags)
// 	Description
// 		Drain samples from the specified user ring buffer, and invoke
// 		the provided callback for each such sample:
//
// 		long (\*callback_fn)(const struct bpf_dynptr \*dynptr, void \*ctx);
//
// 		If **callback_fn** returns 0, the helper will continue to try
// 		and drain the next sample, up to a maximum of
// 		BPF_MAX_USER_RINGBUF_SAMPLES samples. If the return value is 1,
// 		the helper will skip the rest of the samples and return. Other
// 		return values are not used now, and will be rejected by the
// 		verifier.
// 	Return
// 		The number of drained samples if no error was encountered while
// 		draining samples, or 0 if no samples were present in the ring
// 		buffer. If a user-space producer was epoll-waiting on this map,
// 		and at least one sample was drained, they will receive an event
// 		notification notifying them of available space in the ring
// 		buffer. If the BPF_RB_NO_WAKEUP flag is passed to this
// 		function, no wakeup notification will be sent. If the
// 		BPF_RB_FORCE_WAKEUP flag is passed, a wakeup notification will
// 		be sent even if no sample was drained.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EBUSY** if the ring buffer is contended, and another calling
// 		context was concurrently draining the ring buffer.
//
// 		**-EINVAL** if user-space is not properly tracking the ring
// 		buffer due to the producer position not being aligned to 8
// 		bytes, a sample not being aligned to 8 bytes, or the producer
// 		position not matching the advertised length of a sample.
//
// 		**-E2BIG** if user-space has tried to publish a sample which is
// 		larger than the size of the ring buffer, or which cannot fit
// 		within a struct bpf_dynptr.
//
// void *bpf_cgrp_storage_get(struct bpf_map *map, struct cgroup *cgroup, void *value, u64 flags)
// 	Description
// 		Get a bpf_local_storage from the *cgroup*.
//
// 		Logically, it could be thought of as getting the value from
// 		a *map* with *cgroup* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *cgroup*) except this
// 		helper enforces the key must be a cgroup struct and the map must also
// 		be a **BPF_MAP_TYPE_CGRP_STORAGE**.
//
// 		In reality, the local-storage value is embedded directly inside of the
// 		*cgroup* object itself, rather than being located in the
// 		**BPF_MAP_TYPE_CGRP_STORAGE** map. When the local-storage value is
// 		queried for some *map* on a *cgroup* object, the kernel will perform an
// 		O(n) iteration over all of the live local-storage values for that
// 		*cgroup* object until the local-storage value for the *map* is found.
//
// 		An optional *flags* (**BPF_LOCAL_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf_local_storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_LOCAL_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf_local_storage.  If *value* is
// 		**NULL**, the new bpf_local_storage will be zero initialized.
// 	Return
// 		A bpf_local_storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf_local_storage.
//
// long bpf_cgrp_storage_delete(struct bpf_map *map, struct cgroup *cgroup)
// 	Description
// 		Delete a bpf_local_storage from a *cgroup*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf_local_storage cannot be found.
// This helper list is effectively frozen. If you are trying to		 * add a new helper, you should add a kfunc instead which has		 * less stability guarantees. See Documentation/bpf/kfuncs.rst
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_link_info {
    r#type: u32,
    id: u32,
    prog_id: u32,
    __slate_anon_3: __SlateRecord_bpf_h_6842,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_info {
    insn_off: u32,
    type_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_line_info {
    insn_off: u32,
    file_name_off: u32,
    line_off: u32,
    line_col: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock_filter {
    code: u16,
    jt: u8,
    jf: u8,
    k: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_type {
    name_off: u32,
    info: u32,
    __slate_anon_2: __SlateRecord_btf_h_64,
}


// BTF_KIND_INT is followed by a u32 and the following
// is the 32 bits arrangement:
// Attributes stored in the BTF_INT_ENCODING
#[repr(C)]
#[derive(Clone, Copy)]
struct btf_member {
    name_off: u32,
    r#type: u32,
    offset: u32,
}


// If the struct/union type info kind_flag is set, the
// following two macros are used to access bitfield_size
// and bit_offset from btf_member.offset.
#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
    name: *const u8,
    prefix: *const u8,
    flags: i32,
    list: Option<unsafe extern "C" fn(*mut dentry) -> bool>,
    get: Option<unsafe extern "C" fn(*const xattr_handler, *mut dentry, *mut inode, *const u8, *mut (), u64) -> i32>,
    set: Option<unsafe extern "C" fn(*const xattr_handler, *mut mnt_idmap, *mut dentry, *mut inode, *const u8, *const (), u64, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_iter_seq_info {
    seq_ops: *const seq_operations,
    init_seq_private: Option<unsafe extern "C" fn(*mut (), *mut bpf_iter_aux_info) -> i32>,
    fini_seq_private: Option<unsafe extern "C" fn(*mut ())>,
    seq_priv_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_map_ops {
    map_alloc_check: Option<unsafe extern "C" fn(*mut bpf_attr) -> i32>,
    map_alloc: Option<unsafe extern "C" fn(*mut bpf_attr) -> *mut bpf_map>,
    map_release: Option<unsafe extern "C" fn(*mut bpf_map, *mut file)>,
    map_free: Option<unsafe extern "C" fn(*mut bpf_map)>,
    map_get_next_key: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut ()) -> i32>,
    map_release_uref: Option<unsafe extern "C" fn(*mut bpf_map)>,
    map_lookup_elem_sys_only: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> *mut ()>,
    map_lookup_batch: Option<unsafe extern "C" fn(*mut bpf_map, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_lookup_and_delete_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut (), u64) -> i32>,
    map_lookup_and_delete_batch: Option<unsafe extern "C" fn(*mut bpf_map, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_update_batch: Option<unsafe extern "C" fn(*mut bpf_map, *mut file, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_delete_batch: Option<unsafe extern "C" fn(*mut bpf_map, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_lookup_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> *mut ()>,
    map_update_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut (), u64) -> i64>,
    map_delete_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> i64>,
    map_push_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), u64) -> i64>,
    map_pop_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> i64>,
    map_peek_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> i64>,
    map_lookup_percpu_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), u32) -> *mut ()>,
    map_get_hash: Option<unsafe extern "C" fn(*mut bpf_map) -> i32>,
    map_fd_get_ptr: Option<unsafe extern "C" fn(*mut bpf_map, *mut file, i32) -> *mut ()>,
    map_fd_put_ptr: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), bool)>,
    map_gen_lookup: Option<unsafe extern "C" fn(*mut bpf_map, *mut bpf_insn) -> i32>,
    map_fd_sys_lookup_elem: Option<unsafe extern "C" fn(*mut ()) -> u32>,
    map_seq_show_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut seq_file)>,
    map_check_btf: Option<unsafe extern "C" fn(*mut bpf_map, *const btf, *const btf_type, *const btf_type) -> i32>,
    map_poke_track: Option<unsafe extern "C" fn(*mut bpf_map, *mut bpf_prog_aux) -> i32>,
    map_poke_untrack: Option<unsafe extern "C" fn(*mut bpf_map, *mut bpf_prog_aux)>,
    map_poke_run: Option<unsafe extern "C" fn(*mut bpf_map, u32, *mut bpf_prog, *mut bpf_prog)>,
    map_direct_value_addr: Option<unsafe extern "C" fn(*const bpf_map, *mut u64, u32) -> i32>,
    map_direct_value_meta: Option<unsafe extern "C" fn(*const bpf_map, u64, *mut u32) -> i32>,
    map_mmap: Option<unsafe extern "C" fn(*mut bpf_map, *mut vm_area_struct) -> i32>,
    map_mmap_fault: Option<unsafe extern "C" fn(*mut bpf_map, *mut vm_fault) -> u32>,
    map_poll: Option<unsafe extern "C" fn(*mut bpf_map, *mut file, *mut poll_table_struct) -> u32>,
    map_get_unmapped_area: Option<unsafe extern "C" fn(*mut file, u64, u64, u64, u64) -> u64>,
    map_local_storage_charge: Option<unsafe extern "C" fn(*mut bpf_local_storage_map, *mut (), u32) -> i32>,
    map_local_storage_uncharge: Option<unsafe extern "C" fn(*mut bpf_local_storage_map, *mut (), u32)>,
    map_owner_storage_ptr: Option<unsafe extern "C" fn(*mut ()) -> *mut *mut bpf_local_storage>,
    map_redirect: Option<unsafe extern "C" fn(*mut bpf_map, u64, u64) -> i64>,
    map_meta_equal: Option<unsafe extern "C" fn(*const bpf_map, *const bpf_map) -> bool>,
    map_set_for_each_callback_args: Option<unsafe extern "C" fn(*mut bpf_verifier_env, *mut bpf_func_state, *mut bpf_func_state) -> i32>,
    map_for_each_callback: Option<unsafe extern "C" fn(*mut bpf_map, Option<unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64>, *mut (), u64) -> i64>,
    map_mem_usage: Option<unsafe extern "C" fn(*const bpf_map) -> u64>,
    map_btf_id: *mut i32,
    iter_seq_info: *const bpf_iter_seq_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_field_kptr {
    btf: *mut btf,
    module: *mut module,
    dtor: Option<unsafe extern "C" fn(*mut ())>,
    btf_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_field_graph_root {
    btf: *mut btf,
    value_btf_id: u32,
    node_offset: u32,
    value_rec: *mut btf_record,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_field {
    offset: u32,
    size: u32,
    r#type: u32,
    __slate_anon_3: __SlateRecord_bpf_h_256,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_record {
    cnt: u32,
    field_mask: u32,
    spin_lock_off: i32,
    res_spin_lock_off: i32,
    timer_off: i32,
    wq_off: i32,
    refcount_off: i32,
    task_work_off: i32,
    fields: [btf_field; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_map_owner {
    r#type: u32,
    jited: bool,
    xdp_has_frags: bool,
    sleepable: bool,
    storage_cookie: [u64; 2],
    attach_func_proto: *const btf_type,
    expected_attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_map {
    sha: [u8; 32],
    ops: *const bpf_map_ops,
    inner_map_meta: *mut bpf_map,
    security: *mut (),
    map_type: u32,
    key_size: u32,
    value_size: u32,
    max_entries: u32,
    map_extra: u64,
    map_flags: u32,
    id: u32,
    record: *mut btf_record,
    numa_node: i32,
    btf_key_type_id: u32,
    btf_value_type_id: u32,
    btf_vmlinux_value_type_id: u32,
    btf: *mut btf,
    name: [u8; 16],
    freeze_mutex: mutex,
    refcnt: __SlateRecord_types_h_197,
    usercnt: __SlateRecord_types_h_197,
    __slate_anon_21: __SlateRecord_bpf_h_329,
    writecnt: __SlateRecord_types_h_197,
    owner_lock: spinlock,
    owner: *mut bpf_map_owner,
    bypass_spec_v1: bool,
    frozen: bool,
    free_after_mult_rcu_gp: bool,
    free_after_rcu_gp: bool,
    sleepable_refcnt: __SlateRecord_types_h_197,
    elem_count: *mut i64,
    cookie: u64,
    excl_prog_sha: *mut u8,
}


// bpf_type_flag contains a set of flags that are applicable to the values of
// arg_type, ret_type and reg_type. For example, a pointer value may be null,
// or a memory is read-only. We classify types into two categories: base types
// and extended types. Extended types are base types combined with a type flag.
//
// Currently there are no more than 32 base types in arg_type, ret_type and
// reg_types.
// Max number of base types.
// Max number of all types.
// The longest tracepoint has 12 args.
// See include/trace/bpf_probe.h
//
// Also reuse this macro for maximum number of arguments a BPF function
// or a kfunc can have. Args 1-5 are passed in registers, args 6-12 via
// stack arg slots. The JIT may map some stack arg slots to registers based
// on the native calling convention (e.g., arg 6 to R9 on x86-64).
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_proto {
    func: Option<unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64>,
    gpl_only: bool,
    pkt_access: bool,
    might_sleep: bool,
    allow_fastcall: bool,
    ret_type: u32,
    __slate_anon_6: __SlateRecord_bpf_h_996,
    __slate_anon_7: __SlateRecord_bpf_h_1006,
    ret_btf_id: *mut i32,
    allowed: Option<unsafe extern "C" fn(*const bpf_prog) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_insn_access_aux {
    reg_type: u32,
    is_ldsx: bool,
    __slate_anon_2: __SlateRecord_bpf_h_1109,
    log: *mut bpf_verifier_log,
    is_retval: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_ops {
    test_run: Option<unsafe extern "C" fn(*mut bpf_prog, *const bpf_attr, *mut bpf_attr) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_verifier_ops {
    get_func_proto: Option<unsafe extern "C" fn(u32, *const bpf_prog) -> *const bpf_func_proto>,
    is_valid_access: Option<unsafe extern "C" fn(i32, i32, u32, *const bpf_prog, *mut bpf_insn_access_aux) -> bool>,
    gen_prologue: Option<unsafe extern "C" fn(*mut bpf_insn, bool, *const bpf_prog) -> i32>,
    gen_epilogue: Option<unsafe extern "C" fn(*mut bpf_insn, *const bpf_prog, i16) -> i32>,
    gen_ld_abs: Option<unsafe extern "C" fn(*const bpf_insn, *mut bpf_insn) -> i32>,
    convert_ctx_access: Option<unsafe extern "C" fn(u32, *const bpf_insn, *mut bpf_insn, *mut bpf_prog, *mut u32) -> u32>,
    btf_struct_access: Option<unsafe extern "C" fn(*mut bpf_verifier_log, *const bpf_reg_state, i32, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_offload {
    prog: *mut bpf_prog,
    netdev: *mut net_device,
    offdev: *mut bpf_offload_dev,
    dev_priv: *mut (),
    offloads: list_head,
    dev_state: bool,
    opt_failed: bool,
    jited_image: *mut (),
    jited_len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_func_model {
    ret_size: u8,
    ret_flags: u8,
    nr_args: u8,
    arg_size: [u8; 12],
    arg_flags: [u8; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_ksym {
    start: u64,
    end: u64,
    name: [u8; 512],
    lnode: list_head,
    tnode: latch_tree_node,
    prog: bool,
    fp_start: u32,
    fp_end: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_tramp_skip {
    prog: *mut bpf_prog,
    nop: *mut (),
    target: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_tramp_image {
    image: *mut (),
    size: i32,
    ksym: bpf_ksym,
    pcref: percpu_ref,
    call_orig: bool,
    tr: *mut bpf_trampoline,
    list: list_head,
    nr_skips: i32,
    skips: *mut bpf_tramp_skip,
    __slate_anon_9: __SlateRecord_bpf_h_1393,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_trampoline {
    hlist_key: hlist_node,
    hlist_ip: hlist_node,
    fops: *mut ftrace_ops,
    refcnt: refcount_struct,
    flags: u32,
    key: u64,
    ip: u64,
    func: __SlateRecord_bpf_h_1423,
    extension_prog: *mut bpf_prog,
    progs_hlist: [hlist_head; 3],
    progs_cnt: [i32; 3],
    cur_image: *mut bpf_tramp_image,
    images: list_head,
    multi_attach: __SlateRecord_bpf_h_1442,
}


// Fits in 2048B
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_info_aux {
    linkage: u16,
    unreliable: bool,
    called: [bool; 2],
    verified: [bool; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_jit_poke_descriptor {
    tailcall_target: *mut (),
    tailcall_bypass: *mut (),
    bypass_addr: *mut (),
    aux: *mut (),
    __slate_anon_4: __SlateRecord_bpf_h_1705,
    tailcall_target_stable: bool,
    adj_off: u8,
    reason: u16,
    insn_idx: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_ctx_arg_aux {
    offset: u32,
    reg_type: u32,
    btf: *mut btf,
    btf_id: u32,
    ref_id: u32,
    refcounted: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_mod_pair {
    btf: *mut btf,
    module: *mut module,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_stream {
    capacity: __SlateRecord_types_h_190,
    log: llist_head,
    lock: mutex,
    backlog_head: *mut llist_node,
    backlog_tail: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_aux {
    refcnt: __SlateRecord_types_h_197,
    used_map_cnt: u32,
    used_btf_cnt: u32,
    max_ctx_offset: u32,
    max_pkt_offset: u32,
    max_tp_access: u32,
    stack_depth: u32,
    id: u32,
    func_cnt: u32,
    real_func_cnt: u32,
    func_idx: u32,
    attach_btf_id: u32,
    attach_st_ops_member_off: u32,
    ctx_arg_info_size: u32,
    max_rdonly_access: u32,
    max_rdwr_access: u32,
    subprog_start: u32,
    attach_btf: *mut btf,
    ctx_arg_info: *mut bpf_ctx_arg_aux,
    priv_stack_ptr: *mut (),
    dst_mutex: mutex,
    dst_prog: *mut bpf_prog,
    dst_trampoline: *mut bpf_trampoline,
    saved_dst_prog_type: u32,
    saved_dst_attach_type: u32,
    verifier_zext: bool,
    dev_bound: bool,
    offload_requested: bool,
    attach_btf_trace: bool,
    attach_tracing_prog: bool,
    tramp_linked: bool,
    func_proto_unreliable: bool,
    tail_call_reachable: bool,
    xdp_has_frags: bool,
    exception_cb: bool,
    exception_boundary: bool,
    is_extended: bool,
    jits_use_priv_stack: bool,
    priv_stack_requested: bool,
    changes_pkt_data: bool,
    might_sleep: bool,
    kprobe_write_ctx: bool,
    sig: __SlateRecord_bpf_h_1821_2,
    prog_array_member_cnt: u64,
    ext_mutex: mutex,
    arena: *mut bpf_arena,
    recursion_detected: Option<unsafe extern "C" fn(*mut bpf_prog)>,
    attach_func_proto: *const btf_type,
    attach_func_name: *const u8,
    func: *mut *mut bpf_prog,
    main_prog_aux: *mut bpf_prog_aux,
    jit_data: *mut (),
    poke_tab: *mut bpf_jit_poke_descriptor,
    kfunc_tab: *mut bpf_kfunc_desc_tab,
    kfunc_btf_tab: *mut bpf_kfunc_btf_tab,
    size_poke_tab: u32,
    ksym: bpf_ksym,
    ops: *const bpf_prog_ops,
    st_ops: *const bpf_struct_ops,
    used_maps: *mut *mut bpf_map,
    used_maps_mutex: mutex,
    used_btfs: *mut btf_mod_pair,
    prog: *mut bpf_prog,
    user: *mut user_struct,
    load_time: u64,
    verified_insns: u32,
    cgroup_atype: i32,
    cgroup_storage: [*mut bpf_map; 2],
    name: [u8; 16],
    bpf_exception_cb: Option<unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64>,
    stack_arg_sp_adjust: u16,
    security: *mut (),
    token: *mut bpf_token,
    offload: *mut bpf_prog_offload,
    btf: *mut btf,
    func_info: *mut bpf_func_info,
    func_info_aux: *mut bpf_func_info_aux,
    linfo: *mut bpf_line_info,
    jited_linfo: *mut *mut (),
    func_info_cnt: u32,
    nr_linfo: u32,
    linfo_idx: u32,
    r#mod: *mut module,
    num_exentries: u32,
    extable: *mut exception_table_entry,
    __slate_anon_85: __SlateRecord_bpf_h_1892,
    stream: [bpf_stream; 2],
    st_ops_assoc_mutex: mutex,
    st_ops_assoc: *mut bpf_map,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog {
    pages: u16,
    __slate_bits_0: __slate_bits::__SlateBits969U0,
    r#type: u32,
    expected_attach_type: u32,
    len: u32,
    jited_len: u32,
    __slate_anon_22: __SlateRecord_bpf_h_1926,
    stats: *mut bpf_prog_stats,
    active: *mut u8,
    bpf_func: Option<unsafe extern "C" fn(*const (), *const bpf_insn) -> u32>,
    aux: *mut bpf_prog_aux,
    orig_prog: *mut sock_fprog_kern,
    __slate_anon_28: __SlateRecord_bpf_h_1937,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_link {
    refcnt: __SlateRecord_types_h_197,
    id: u32,
    r#type: u32,
    ops: *const bpf_link_ops,
    prog: *mut bpf_prog,
    flags: u32,
    attach_type: u32,
    __slate_anon_7: __SlateRecord_bpf_h_1964,
    sleepable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_link_ops {
    release: Option<unsafe extern "C" fn(*mut bpf_link)>,
    dealloc: Option<unsafe extern "C" fn(*mut bpf_link)>,
    dealloc_deferred: Option<unsafe extern "C" fn(*mut bpf_link)>,
    detach: Option<unsafe extern "C" fn(*mut bpf_link) -> i32>,
    update_prog: Option<unsafe extern "C" fn(*mut bpf_link, *mut bpf_prog, *mut bpf_prog) -> i32>,
    show_fdinfo: Option<unsafe extern "C" fn(*const bpf_link, *mut seq_file)>,
    fill_link_info: Option<unsafe extern "C" fn(*const bpf_link, *mut bpf_link_info) -> i32>,
    update_map: Option<unsafe extern "C" fn(*mut bpf_link, *mut bpf_map, *mut bpf_map) -> i32>,
    poll: Option<unsafe extern "C" fn(*mut file, *mut poll_table_struct) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_token {
    work: work_struct,
    refcnt: __SlateRecord_types_h_197,
    userns: *mut user_namespace,
    allowed_cmds: u64,
    allowed_maps: u64,
    allowed_progs: u64,
    allowed_attachs: u64,
    security: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_struct_ops {
    verifier_ops: *const bpf_verifier_ops,
    init: Option<unsafe extern "C" fn(*mut btf) -> i32>,
    check_member: Option<unsafe extern "C" fn(*const btf_type, *const btf_member, *const bpf_prog) -> i32>,
    init_member: Option<unsafe extern "C" fn(*const btf_type, *const btf_member, *mut (), *const ()) -> i32>,
    reg: Option<unsafe extern "C" fn(*mut (), *mut bpf_link) -> i32>,
    unreg: Option<unsafe extern "C" fn(*mut (), *mut bpf_link)>,
    update: Option<unsafe extern "C" fn(*mut (), *mut (), *mut bpf_link) -> i32>,
    validate: Option<unsafe extern "C" fn(*mut ()) -> i32>,
    cfi_stubs: *mut (),
    owner: *mut module,
    name: *const u8,
    func_models: [btf_func_model; 64],
}


// Per-cpu temp buffers used by printf-like helpers to store the bprintf binary
// arguments representation.
#[repr(C)]
#[derive(Clone, Copy)]
struct perf_callchain_entry {
    nr: u64,
    ip: [u64; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct perf_raw_frag {
    __slate_anon_0: __SlateRecord_perf_event_h_76,
    copy: Option<unsafe extern "C" fn(*mut (), *const (), u64, u64) -> u64>,
    data: *mut (),
    size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_raw_record {
    frag: perf_raw_frag,
    size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_branch_stack {
    nr: u64,
    hw_idx: u64,
    entries: [perf_branch_entry; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hw_perf_event_extra {
    config: u64,
    reg: u32,
    alloc: i32,
    idx: i32,
}


// hw_perf_event::flag values
//
// PERF_EVENT_FLAG_ARCH bits are reserved for architecture-specific
// usage.
#[repr(C)]
#[derive(Clone, Copy)]
struct hw_perf_event {
    __slate_anon_0: __SlateRecord_perf_event_h_149,
    target: *mut task_struct,
    addr_filters: *mut (),
    addr_filters_gen: u64,
    state: i32,
    prev_count: __SlateRecord_local64_h_26,
    sample_period: u64,
    __slate_anon_7: __SlateRecord_perf_event_h_244,
    interrupts_seq: u64,
    interrupts: u64,
    freq_time_stamp: u64,
    freq_count_stamp: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pmu {
    entry: list_head,
    events_lock: spinlock,
    events: list_head,
    module: *mut module,
    dev: *mut device,
    parent: *mut device,
    attr_groups: *mut *const attribute_group,
    attr_update: *mut *const attribute_group,
    name: *const u8,
    r#type: i32,
    capabilities: i32,
    scope: u32,
    cpu_pmu_context: *mut *mut perf_cpu_pmu_context,
    exclusive_cnt: __SlateRecord_types_h_190,
    task_ctx_nr: i32,
    hrtimer_interval_ms: i32,
    nr_addr_filters: u32,
    pmu_enable: Option<unsafe extern "C" fn(*mut pmu)>,
    pmu_disable: Option<unsafe extern "C" fn(*mut pmu)>,
    event_init: Option<unsafe extern "C" fn(*mut perf_event) -> i32>,
    event_mapped: Option<unsafe extern "C" fn(*mut perf_event, *mut mm_struct)>,
    event_unmapped: Option<unsafe extern "C" fn(*mut perf_event, *mut mm_struct)>,
    add: Option<unsafe extern "C" fn(*mut perf_event, i32) -> i32>,
    del: Option<unsafe extern "C" fn(*mut perf_event, i32)>,
    start: Option<unsafe extern "C" fn(*mut perf_event, i32)>,
    stop: Option<unsafe extern "C" fn(*mut perf_event, i32)>,
    read: Option<unsafe extern "C" fn(*mut perf_event)>,
    start_txn: Option<unsafe extern "C" fn(*mut pmu, u32)>,
    commit_txn: Option<unsafe extern "C" fn(*mut pmu) -> i32>,
    cancel_txn: Option<unsafe extern "C" fn(*mut pmu)>,
    event_idx: Option<unsafe extern "C" fn(*mut perf_event) -> i32>,
    sched_task: Option<unsafe extern "C" fn(*mut perf_event_pmu_context, *mut task_struct, bool)>,
    task_ctx_cache: *mut kmem_cache,
    setup_aux: Option<unsafe extern "C" fn(*mut perf_event, *mut *mut (), i32, bool) -> *mut ()>,
    free_aux: Option<unsafe extern "C" fn(*mut ())>,
    snapshot_aux: Option<unsafe extern "C" fn(*mut perf_event, *mut perf_output_handle, u64) -> i64>,
    addr_filters_validate: Option<unsafe extern "C" fn(*mut list_head) -> i32>,
    addr_filters_sync: Option<unsafe extern "C" fn(*mut perf_event)>,
    aux_output_match: Option<unsafe extern "C" fn(*mut perf_event) -> i32>,
    filter: Option<unsafe extern "C" fn(*mut pmu, i32) -> bool>,
    check_period: Option<unsafe extern "C" fn(*mut perf_event, u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_addr_filters_head {
    list: list_head,
    lock: raw_spinlock,
    nr_file_filters: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_addr_filter_range {
    start: u64,
    size: u64,
}


// Event capabilities. For event_caps and groups caps.
//
// PERF_EV_CAP_SOFTWARE: Is a software event.
// PERF_EV_CAP_READ_ACTIVE_PKG: A CPU event (or cgroup event) that can be read
// from any CPU in the package where it is active.
// PERF_EV_CAP_SIBLING: An event with this flag must be a group sibling and
// cannot be a group leader. If an event with this flag is detached from the
// group it is scheduled out and moved into an unrecoverable ERROR state.
// PERF_EV_CAP_READ_SCOPE: A CPU event that can be read from any CPU of the
// PMU scope where it is active.
#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
    event_entry: list_head,
    sibling_list: list_head,
    active_list: list_head,
    group_node: rb_node,
    group_index: u64,
    migrate_entry: list_head,
    hlist_entry: hlist_node,
    active_entry: list_head,
    nr_siblings: i32,
    event_caps: i32,
    group_caps: i32,
    group_generation: u32,
    group_leader: *mut perf_event,
    pmu: *mut pmu,
    pmu_private: *mut (),
    state: i32,
    attach_state: u32,
    count: __SlateRecord_local64_h_26,
    child_count: __SlateRecord_types_h_197,
    total_time_enabled: u64,
    total_time_running: u64,
    tstamp: u64,
    attr: perf_event_attr,
    header_size: u16,
    id_header_size: u16,
    read_size: u16,
    hw: hw_perf_event,
    ctx: *mut perf_event_context,
    pmu_ctx: *mut perf_event_pmu_context,
    refcount: __SlateRecord_types_h_197,
    child_total_time_enabled: __SlateRecord_types_h_197,
    child_total_time_running: __SlateRecord_types_h_197,
    child_mutex: mutex,
    child_list: list_head,
    parent: *mut perf_event,
    oncpu: i32,
    cpu: i32,
    owner_entry: list_head,
    owner: *mut task_struct,
    mmap_mutex: mutex,
    mmap_count: refcount_struct,
    rb: *mut perf_buffer,
    rb_entry: list_head,
    rcu_batches: u64,
    rcu_pending: i32,
    waitq: wait_queue_head,
    fasync: *mut fasync_struct,
    pending_wakeup: u32,
    pending_kill: u32,
    pending_disable: u32,
    pending_addr: u64,
    pending_irq: irq_work,
    pending_disable_irq: irq_work,
    pending_task: callback_head,
    pending_work: u32,
    event_limit: __SlateRecord_types_h_190,
    addr_filters: perf_addr_filters_head,
    addr_filter_ranges: *mut perf_addr_filter_range,
    addr_filters_gen: u64,
    aux_event: *mut perf_event,
    destroy: Option<unsafe extern "C" fn(*mut perf_event)>,
    callback_head: callback_head,
    ns: *mut pid_namespace,
    id: u64,
    lost_samples: __SlateRecord_types_h_197,
    clock: Option<unsafe extern "C" fn() -> u64>,
    overflow_handler: Option<unsafe extern "C" fn(*mut perf_event, *mut perf_sample_data, *mut pt_regs)>,
    overflow_handler_context: *mut (),
    prog: *mut bpf_prog,
    bpf_cookie: u64,
    tp_event: *mut trace_event_call,
    filter: *mut event_filter,
    cgrp: *mut perf_cgroup,
    security: *mut (),
    sb_list: list_head,
    pmu_list: list_head,
    orig_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_pmu_context {
    pmu: *mut pmu,
    ctx: *mut perf_event_context,
    pmu_ctx_entry: list_head,
    pinned_active: list_head,
    flexible_active: list_head,
    __slate_bits_0: __slate_bits::__SlateBits1067U0,
    nr_events: u32,
    nr_cgroups: u32,
    nr_freq: u32,
    refcount: __SlateRecord_types_h_190,
    callback_head: callback_head,
    rotate_necessary: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_groups {
    tree: rb_root,
    index: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_time_ctx {
    time: u64,
    stamp: u64,
    offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_context {
    lock: raw_spinlock,
    mutex: mutex,
    pmu_ctx_list: list_head,
    pinned_groups: perf_event_groups,
    flexible_groups: perf_event_groups,
    event_list: list_head,
    nr_events: i32,
    nr_user: i32,
    is_active: i32,
    nr_stat: i32,
    nr_freq: i32,
    rotate_disable: i32,
    refcount: refcount_struct,
    task: *mut task_struct,
    time: perf_time_ctx,
    timeguest: perf_time_ctx,
    parent_ctx: *mut perf_event_context,
    parent_gen: u64,
    generation: u64,
    pin_count: i32,
    nr_cgroups: i32,
    callback_head: callback_head,
    nr_no_switch_fast: __SlateRecord_local_h_12,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_ctx_data {
    callback_head: callback_head,
    refcount: refcount_struct,
    global: i32,
    ctx_cache: *mut kmem_cache,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_cpu_pmu_context {
    epc: perf_event_pmu_context,
    task_epc: *mut perf_event_pmu_context,
    sched_cb_entry: list_head,
    sched_cb_usage: i32,
    active_oncpu: i32,
    exclusive: i32,
    pmu_disable_count: i32,
    hrtimer_lock: raw_spinlock,
    hrtimer: hrtimer,
    hrtimer_interval: i64,
    hrtimer_active: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_output_handle {
    event: *mut perf_event,
    rb: *mut perf_buffer,
    wakeup: u64,
    size: u64,
    __slate_anon_4: __SlateRecord_perf_event_h_1156,
    __slate_anon_5: __SlateRecord_perf_event_h_1163,
    page: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_cgroup_info {
    time: perf_time_ctx,
    timeguest: perf_time_ctx,
    active: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_cgroup {
    css: cgroup_subsys_state,
    info: *mut perf_cgroup_info,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct perf_sample_data {
    sample_flags: u64,
    period: u64,
    dyn_size: u64,
    r#type: u64,
    tid_entry: __SlateRecord_perf_event_h_1315,
    time: u64,
    id: u64,
    cpu_entry: __SlateRecord_perf_event_h_1321,
    ip: u64,
    callchain: *mut perf_callchain_entry,
    raw: *mut perf_raw_record,
    br_stack: *mut perf_branch_stack,
    br_stack_cntr: *mut u64,
    weight: perf_sample_weight,
    data_src: perf_mem_data_src,
    txn: u64,
    regs_user: perf_regs,
    regs_intr: perf_regs,
    stack_user_size: u64,
    stream_id: u64,
    cgroup: u64,
    addr: u64,
    phys_addr: u64,
    data_page_size: u64,
    code_page_size: u64,
    aux_size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_parameter {
    key: *const u8,
    __slate_bits_0: __slate_bits::__SlateBits1046U0,
    __slate_anon_2: __SlateRecord_fs_context_h_66,
    size: u64,
    dirfd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct p_log {
    prefix: *const u8,
    log: *mut fc_log,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
    ops: *const fs_context_operations,
    uapi_mutex: mutex,
    fs_type: *mut file_system_type,
    fs_private: *mut (),
    sget_key: *mut (),
    root: *mut dentry,
    user_ns: *mut user_namespace,
    net_ns: *mut net,
    cred: *const cred,
    log: p_log,
    source: *const u8,
    security: *mut (),
    s_fs_info: *mut (),
    sb_flags: u32,
    sb_flags_mask: u32,
    s_iflags: u32,
    __slate_bits_0: __slate_bits::__SlateBits627U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context_operations {
    free: Option<unsafe extern "C" fn(*mut fs_context)>,
    dup: Option<unsafe extern "C" fn(*mut fs_context, *mut fs_context) -> i32>,
    parse_param: Option<unsafe extern "C" fn(*mut fs_context, *mut fs_parameter) -> i32>,
    parse_monolithic: Option<unsafe extern "C" fn(*mut fs_context, *mut ()) -> i32>,
    get_tree: Option<unsafe extern "C" fn(*mut fs_context) -> i32>,
    reconfigure: Option<unsafe extern "C" fn(*mut fs_context) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fc_log {
    usage: refcount_struct,
    head: u8,
    tail: u8,
    need_free: u8,
    owner: *mut module,
    buffer: [*mut u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_parameter_spec {
    name: *const u8,
    r#type: Option<unsafe extern "C" fn(*mut p_log, *const fs_parameter_spec, *mut fs_parameter, *mut fs_parse_result) -> i32>,
    opt: u8,
    flags: u16,
    data: *const (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_parse_result {
    negated: bool,
    __slate_anon_1: __SlateRecord_fs_parser_h_56,
}


fn shmem_file(file: *mut file) -> bool {
if IS_ENABLED__2!() == 0 {
return false;
}
if file.is_null() || unsafe { (*file).f_mapping }.is_null() {
return false;
}
unsafe { shmem_mapping((*file).f_mapping as *const address_space) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mmu_table_batch {
    rcu: callback_head,
    nr: u32,
    tables: [*mut (); 0],
}


// If we can't allocate a page to make a big batch of page pointers
// to work on, then just handle a few from the on-stack structure.
#[repr(C)]
#[derive(Clone, Copy)]
struct mmu_gather_batch {
    next: *mut mmu_gather_batch,
    nr: u32,
    max: u32,
    encoded_pages: [*mut encoded_page; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmu_gather {
    mm: *mut mm_struct,
    batch: *mut mmu_table_batch,
    start: u64,
    end: u64,
    __slate_bits_0: __slate_bits::__SlateBits476U0,
    batch_count: u32,
    active: *mut mmu_gather_batch,
    local: mmu_gather_batch,
    __pages: [*mut page; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vma_prepare {
    vma: *mut vm_area_struct,
    adj_next: *mut vm_area_struct,
    file: *mut file,
    mapping: *mut address_space,
    anon_vma: *mut anon_vma,
    insert: *mut vm_area_struct,
    remove: *mut vm_area_struct,
    remove2: *mut vm_area_struct,
    __slate_bits_0: __slate_bits::__SlateBits1097U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unlink_vma_file_batch {
    count: i32,
    vmas: [*mut vm_area_struct; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vma_munmap_struct {
    vmi: *mut vma_iterator,
    vma: *mut vm_area_struct,
    prev: *mut vm_area_struct,
    next: *mut vm_area_struct,
    uf: *mut list_head,
    start: u64,
    end: u64,
    unmap_start: u64,
    unmap_end: u64,
    vma_count: i32,
    unlock: bool,
    clear_ptes: bool,
    nr_pages: u64,
    locked_vm: u64,
    nr_accounted: u64,
    exec_vm: u64,
    stack_vm: u64,
    data_vm: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vma_merge_struct {
    mm: *mut mm_struct,
    vmi: *mut vma_iterator,
    prev: *mut vm_area_struct,
    middle: *mut vm_area_struct,
    next: *mut vm_area_struct,
    target: *mut vm_area_struct,
    start: u64,
    end: u64,
    pgoff: u64,
    anon_pgoff: u64,
    __slate_anon_10: __SlateRecord_vma_h_109,
    file: *mut file,
    anon_vma: *mut anon_vma,
    policy: *mut mempolicy,
    uffd_ctx: vm_userfaultfd_ctx,
    anon_name: *mut anon_vma_name,
    state: u32,
    copied_from: *mut vm_area_struct,
    __slate_bits_0: __slate_bits::__SlateBits111U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unmap_desc {
    mas: *mut ma_state,
    first: *mut vm_area_struct,
    pg_start: u64,
    pg_end: u64,
    vma_start: u64,
    vma_end: u64,
    tree_end: u64,
    tree_reset: u64,
    mm_wr_locked: bool,
}


fn vmg_nomem(vmg: *mut vma_merge_struct) -> bool {
((unsafe { (*vmg).state }) as u32) == 1
}

fn vmg_pages(vmg: *const vma_merge_struct) -> u64 {
let size: u64 = unsafe { (*vmg).end }.wrapping_sub(unsafe { (*vmg).start });
size >> 12i32
}

fn vmg_start_pgoff(vmg: *const vma_merge_struct) -> u64 {
unsafe { (*vmg).pgoff }
}

fn vmg_end_pgoff(vmg: *const vma_merge_struct) -> u64 {
vmg_start_pgoff(vmg).wrapping_add(vmg_pages(vmg))
}

fn assert_sane_pgoff(vma: *mut vm_area_struct, pgoff: u64) {
// nommu doesn't set a virtual pgoff for anon VMAs.
if IS_ENABLED__2!() == 0 {
return;
}
// File-backed VMAs have arbitrary page offset (either page offset into
// file or for pfnmap the PFN of the start of the range or drivers may
// set arbitrary page offset).
if !vma_is_anonymous(vma as *const vm_area_struct) {
return;
}
// MAP_PRIVATE-/dev/zero is anon, non-NULL vm_file, but has file pgoff.
if !unsafe { (*vma).vm_file }.is_null() {
return;
}
// If faulted in, could have been remapped.
if !unsafe { (*vma).anon_vma }.is_null() {
return;
}
// OK this is really an anon VMA - expect virtual page offset.
VM_WARN_ON_ONCE!();
}

fn vma_set_pgoff(vma: *mut vm_area_struct, pgoff: u64) {
vma_assert_can_modify(vma);
assert_sane_pgoff(vma, pgoff);
unsafe {
(*vma).vm_pgoff = pgoff;
}
}

fn vmg_start_anon_pgoff(vmg: *const vma_merge_struct) -> u64 {
unsafe { (*vmg).anon_pgoff }
}

fn vmg_end_anon_pgoff(vmg: *const vma_merge_struct) -> u64 {
vmg_start_anon_pgoff(vmg).wrapping_add(vmg_pages(vmg))
}

fn __vma_set_anon_pgoff(vma: *mut vm_area_struct, pgoff: u64) {
unsafe {
(*vma).__vm_anon_pgoff_hi = (pgoff >> 32i32) as u32;
}
unsafe {
(*vma).__vm_anon_pgoff_lo = (pgoff & GENMASK__4!(31i32, 0i32)) as u32;
}
}

fn vma_set_anon_pgoff(vma: *mut vm_area_struct, pgoff: u64) {
vma_assert_can_modify(vma);
__vma_set_anon_pgoff(vma, pgoff);
}

fn vma_add_pgoff(vma: *mut vm_area_struct, delta: u64) {
vma_assert_can_modify(vma);
vma_set_pgoff(vma, vma_start_pgoff(vma as *const vm_area_struct).wrapping_add(delta));
vma_set_anon_pgoff(vma, vma_start_anon_pgoff(vma as *const vm_area_struct).wrapping_add(delta));
}

fn vma_sub_pgoff(vma: *mut vm_area_struct, delta: u64) {
vma_assert_can_modify(vma);
vma_set_pgoff(vma, vma_start_pgoff(vma as *const vm_area_struct).wrapping_sub(delta));
vma_set_anon_pgoff(vma, vma_start_anon_pgoff(vma as *const vm_area_struct).wrapping_sub(delta));
}

fn vma_iter_store_gfp(vmi: *mut vma_iterator, vma: *mut vm_area_struct, gfp: u32) -> i32 {
if ((unsafe { (*vmi).mas.status }) as u32) != 1 && ((unsafe { (*vmi).mas.index }) > unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start } || (unsafe { (*vmi).mas.last }) < unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }) {
vma_iter_invalidate(vmi);
}
__mas_set_range(unsafe { core::ptr::addr_of_mut!((*vmi).mas) }, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end }.wrapping_sub(1));
unsafe { mas_store_gfp(core::ptr::addr_of_mut!((*vmi).mas), vma as *mut (), gfp) };
if unlikely!(mas_is_err(unsafe { core::ptr::addr_of_mut!((*vmi).mas) })) != 0 {
return 12i32.wrapping_neg();
}
vma_mark_attached(vma);
0
}

fn vma_pgprot_modify(oldprot: pgprot, vma_flags: __SlateRecord_mm_types_h_871) -> pgprot {
let prot: pgprot = vma_flags_to_page_prot(vma_flags);
return pgprot_modify(oldprot, prot);
}

fn is_exec_mapping(flags: u64) -> bool {
flags & (BIT!(2i32) | BIT!(1i32) | BIT!(8i32)) == BIT!(2i32)
}

fn is_stack_mapping(flags: u64) -> bool {
flags & BIT!(8i32) == BIT!(8i32) || flags & 0 != 0
}

fn is_data_mapping_vma_flags(vma_flags: *const __SlateRecord_mm_types_h_871) -> bool {
let mut __slate_compound_1: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: [i32; 2] = unsafe { core::mem::zeroed() };
vma_flags_test(vma_flags, 1i32) && !vma_flags_test_any_mask(vma_flags, __mk_vma_flags(unsafe { *{
__slate_compound_1 = {
let mut __t0: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t0
};
core::ptr::addr_of_mut!(__slate_compound_1)
} }, 2u64, unsafe { (*{
__slate_compound_2 = [3i32, 8i32];
core::ptr::addr_of_mut!(__slate_compound_2)
}).as_ptr() as *const i32 }))
}

fn vma_iter_config(vmi: *mut vma_iterator, index: u64, last: u64) {
__mas_set_range(unsafe { core::ptr::addr_of_mut!((*vmi).mas) }, index, last.wrapping_sub(1));
}

fn vma_iter_reset(vmi: &mut vma_iterator) {
mas_reset(unsafe { core::ptr::addr_of_mut!((*(vmi as *mut vma_iterator)).mas) });
}

fn vma_iter_next_range_limit(vmi: &mut vma_iterator, max: u64) -> *mut vm_area_struct {
(unsafe { mas_next_range(core::ptr::addr_of_mut!((*(vmi as *mut vma_iterator)).mas), max) }) as *mut vm_area_struct
}

fn vma_iter_area_lowest(vmi: &mut vma_iterator, min: u64, max: u64, size: u64) -> i32 {
unsafe { mas_empty_area(core::ptr::addr_of_mut!((*(vmi as *mut vma_iterator)).mas), min, max.wrapping_sub(1), size) }
}

fn vma_iter_area_highest(vmi: &mut vma_iterator, min: u64, max: u64, size: u64) -> i32 {
unsafe { mas_empty_area_rev(core::ptr::addr_of_mut!((*(vmi as *mut vma_iterator)).mas), min, max.wrapping_sub(1), size) }
}

fn vma_iter_prealloc(vmi: *mut vma_iterator, vma: *mut vm_area_struct) -> i32 {
unsafe { mas_preallocate(core::ptr::addr_of_mut!((*vmi).mas), vma as *mut (), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) }
}

fn vma_iter_clear(vmi: *mut vma_iterator) {
unsafe { mas_store_prealloc(core::ptr::addr_of_mut!((*vmi).mas), core::ptr::null_mut::<()>()) };
}

fn vma_iter_load(vmi: *mut vma_iterator) -> *mut vm_area_struct {
(unsafe { mas_walk(core::ptr::addr_of_mut!((*vmi).mas)) }) as *mut vm_area_struct
}

fn vma_iter_store_overwrite(vmi: *mut vma_iterator, vma: *mut vm_area_struct) {
vma_assert_attached(vma);
if ((unsafe { (*vmi).mas.status }) as u32) != 1 && ((unsafe { (*vmi).mas.index }) > unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start } || (unsafe { (*vmi).mas.last }) < unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }) {
vma_iter_invalidate(vmi);
}
__mas_set_range(unsafe { core::ptr::addr_of_mut!((*vmi).mas) }, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end }.wrapping_sub(1));
unsafe { mas_store_prealloc(core::ptr::addr_of_mut!((*vmi).mas), vma as *mut ()) };
}

fn vma_iter_store_new(vmi: *mut vma_iterator, vma: *mut vm_area_struct) {
vma_mark_attached(vma);
vma_iter_store_overwrite(vmi, vma);
}

fn vma_iter_addr(vmi: *mut vma_iterator) -> u64 {
unsafe { (*vmi).mas.index }
}

fn vma_iter_end(vmi: &vma_iterator) -> u64 {
unsafe { (*(vmi as *const vma_iterator)).mas.last }.wrapping_add(1)
}

fn vma_iter_prev_range(vmi: *mut vma_iterator) -> *mut vm_area_struct {
(unsafe { mas_prev_range(core::ptr::addr_of_mut!((*vmi).mas), 0u64) }) as *mut vm_area_struct
}

fn vma_iter_next_rewind(vmi: *mut vma_iterator, pprev: *mut *mut vm_area_struct) -> *mut vm_area_struct {
let next: *mut vm_area_struct = vma_next(vmi);
let prev: *mut vm_area_struct = vma_prev(vmi);
// Consider the case where no previous VMA exists. We advance to the
// next VMA, skipping any gap, then rewind to the start of the range.
//
// If we were to unconditionally advance to the next range we'd wind up
// at the next VMA again, so we check to ensure there is a previous VMA
// to skip over.
if !prev.is_null() {
vma_iter_next_range(vmi);
}
if !pprev.is_null() {
unsafe {
*pprev = prev;
}
}
next
}

fn vma_is_sealed(vma: *mut vm_area_struct) -> bool {
(unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(42i32) != 0
}

fn map_deny_write_exec(old: &__SlateRecord_mm_types_h_871, new: &__SlateRecord_mm_types_h_871) -> bool {
// If MDWE is disabled, we have nothing to deny.
if !mm_flags_test(28i32, (unsafe { (*get_current()).mm }) as *const mm_struct) {
return false;
}
// If the new VMA is not executable, we have nothing to deny.
if !vma_flags_test(new as *const __SlateRecord_mm_types_h_871, 2i32) {
return false;
}
// Under MDWE we do not accept newly writably executable VMAs...
if vma_flags_test(new as *const __SlateRecord_mm_types_h_871, 1i32) {
return true;
}
// ...nor previously non-executable VMAs becoming executable.
if !vma_flags_test(old as *const __SlateRecord_mm_types_h_871, 2i32) {
return true;
}
false
}

fn mmap_file(file: *mut file, vma: *mut vm_area_struct) -> i32 {
let err: i32 = vfs_mmap(file, vma);
if likely!(err == 0) != 0 {
return 0;
}
// OK, we tried to call the file hook for mmap(), but an error
// arose. The mapping is in an inconsistent state and we must not invoke
// any further hooks on it.
unsafe {
(*vma).vm_ops = core::ptr::addr_of!(vma_dummy_vm_ops);
}
err
}

fn vma_close(vma: *mut vm_area_struct) {
if !unsafe { (*vma).vm_ops }.is_null() && (unsafe { (*(*vma).vm_ops).close }) != None {
unsafe { unsafe { (*(*vma).vm_ops).close }.unwrap()(vma) };
// The mapping is in an inconsistent state, and no further hooks
// may be invoked upon it.
unsafe {
(*vma).vm_ops = core::ptr::addr_of!(vma_dummy_vm_ops);
}
}
}

fn anon_vma_lock_write(anon_vma: *mut anon_vma) {
unsafe { down_write(core::ptr::addr_of_mut!((*(*anon_vma).root).rwsem)) };
}

fn anon_vma_unlock_write(anon_vma: *mut anon_vma) {
unsafe { up_write(core::ptr::addr_of_mut!((*(*anon_vma).root).rwsem)) };
}

fn anon_vma_prepare(vma: *mut vm_area_struct) -> i32 {
if likely__8!(unsafe { (*vma).anon_vma }) != 0 {
return 0;
}
unsafe { __anon_vma_prepare(vma) }
}

// Compare PTEs respecting the dirty bit.
// Compare PTEs respecting the soft-dirty bit.
// Compare PTEs respecting the writable bit.
// Merge PTE write bits: if any PTE in the batch is writable, modify the
// PTE at @ptentp to be writable.
// Merge PTE young and dirty bits: if any PTE in the batch is young or dirty,
// modify the PTE at @ptentp to be young or dirty, respectively.
#[repr(C)]
#[derive(Clone, Copy)]
struct capture_control {
    zone: *mut zone,
    migratetype: i32,
    order: i32,
    page: *mut page,
}


fn vma_supports_mlock(vma: *const vm_area_struct) -> bool {
let mut __slate_compound_0: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: [i32; 4] = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_3: [i32; 1] = unsafe { core::mem::zeroed() };
if vma_test_any_mask(vma, __mk_vma_flags(unsafe { *{
__slate_compound_0 = {
let mut __t0: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t0
};
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 4u64, unsafe { (*{
__slate_compound_1 = [14i32, 18i32, 10i32, 28i32];
core::ptr::addr_of_mut!(__slate_compound_1)
}).as_ptr() as *const i32 })) {
return false;
}
if vma_test_single_mask(vma, __mk_vma_flags(unsafe { *{
__slate_compound_2 = {
let mut __t1: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t1
};
core::ptr::addr_of_mut!(__slate_compound_2)
} }, 1u64, unsafe { (*{
__slate_compound_3 = [40i32];
core::ptr::addr_of_mut!(__slate_compound_3)
}).as_ptr() as *const i32 })) {
return false;
}
let __v4: bool = vma_is_dax(vma) || is_vm_hugetlb_page(vma);
if __v4 {
return false;
}
vma != ((unsafe { get_gate_vma((*get_current()).mm) }) as *const vm_area_struct)
}

fn vma_soft_dirty_enabled(vma: *mut vm_area_struct) -> bool {
// NOTE: we must check this before VM_SOFTDIRTY on soft-dirty
// enablements, because when without soft-dirty being compiled in,
// VM_SOFTDIRTY is defined as 0x0, then !(vm_flags & VM_SOFTDIRTY)
// will be constantly true.
if pgtable_supports_soft_dirty!() == 0 {
return false;
}
// Soft-dirty is kind of special: its tracking is enabled when the
// vma flags not set.
(unsafe { (*vma).__slate_anon_3.vm_flags }) & 0 == 0
}

fn get_sysctl_max_map_count() -> i32 {
let mut __v0: i32;
loop {
if !(__native_word__9!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_722() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!(sysctl_max_map_count) as *const i32) };
__v0
}

// VMA-specific functions.
// To allow for userland testing we place internal dependencies in
// vma_internal.h and external VMA API declarations in vma.h.
#[repr(C)]
#[derive(Clone, Copy)]
struct mmap_state {
    mm: *mut mm_struct,
    vmi: *mut vma_iterator,
    addr: u64,
    end: u64,
    pgoff: u64,
    anon_pgoff: u64,
    pglen: u64,
    __slate_anon_7: __SlateRecord_vma_c_23,
    file: *mut file,
    page_prot: pgprot,
    /// User-defined fields, perhaps updated by .mmap_prepare().
    vm_ops: *const vm_operations_struct,
    vm_private_data: *mut (),
    charged: u64,
    prev: *mut vm_area_struct,
    next: *mut vm_area_struct,
    /// Unmapping state.
    vms: vma_munmap_struct,
    mas_detach: ma_state,
    mt_detach: maple_tree,
    /// Determine if we can check KSM flags early in mmap() logic.
    __slate_bits_0: __slate_bits::__SlateBits1105U0,
    // If .mmap_prepare changed the file, we don't need to pin.
}


fn __vma_set_range(vma: *mut vm_area_struct, start: u64, end: u64) {
unsafe {
(*vma).__slate_anon_0.__slate_anon_0.vm_start = start;
}
unsafe {
(*vma).__slate_anon_0.__slate_anon_0.vm_end = end;
}
}

fn vma_set_range(vma: *mut vm_area_struct, start: u64, end: u64, pgoff: u64, anon_pgoff: u64) {
__vma_set_range(vma, start, end);
vma_set_pgoff(vma, pgoff);
vma_set_anon_pgoff(vma, anon_pgoff);
}

/// Was this VMA ever forked from a parent, i.e. maybe contains CoW mappings?
fn vma_is_fork_child(vma: *mut vm_area_struct) -> bool {
// The list_is_singular() test is to avoid merging VMA cloned from
// parents. This can improve scalability caused by the anon_vma root
// lock.
!vma.is_null() && !unsafe { (*vma).anon_vma }.is_null() && list_is_singular((unsafe { core::ptr::addr_of_mut!((*vma).anon_vma_chain) }) as *const list_head) == 0
}

fn is_mergeable_vma(vmg: *mut vma_merge_struct, merge_next: bool) -> bool {
let mut __slate_compound_0: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: [i32; 1] = unsafe { core::mem::zeroed() };
let vma: *mut vm_area_struct = if merge_next { unsafe { (*vmg).next } } else { unsafe { (*vmg).prev } };
let mut diff: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(diff) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
if !mpol_equal(unsafe { (*vmg).policy }, vma_policy!(vma)) {
return false;
}
diff = vma_flags_diff_pair((unsafe { core::ptr::addr_of_mut!((*vma).__slate_anon_3.flags) }) as *const __SlateRecord_mm_types_h_871, (unsafe { core::ptr::addr_of_mut!((*vmg).__slate_anon_10.vma_flags) }) as *const __SlateRecord_mm_types_h_871);
vma_flags_clear_mask(core::ptr::addr_of_mut!(diff), __mk_vma_flags(unsafe { *{
__slate_compound_0 = {
let mut __t0: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t0
};
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 1u64, unsafe { (*{
__slate_compound_1 = [11i32];
core::ptr::addr_of_mut!(__slate_compound_1)
}).as_ptr() as *const i32 }));
if !vma_flags_empty(unsafe { &(*(core::ptr::addr_of_mut!(diff) as *const __SlateRecord_mm_types_h_871)) }) {
return false;
}
if (unsafe { (*vma).vm_file }) != unsafe { (*vmg).file } {
return false;
}
if !is_mergeable_vm_userfaultfd_ctx(vma, unsafe { (*vmg).uffd_ctx }) {
return false;
}
if !anon_vma_name_eq(anon_vma_name(vma), unsafe { (*vmg).anon_name }) {
return false;
}
true
}

fn is_mergeable_anon_vma(vmg: *mut vma_merge_struct, merge_next: bool) -> bool {
let tgt: *mut vm_area_struct = if merge_next { unsafe { (*vmg).next } } else { unsafe { (*vmg).prev } };
let src: *mut vm_area_struct = unsafe { (*vmg).middle }; // existing merge case.
let tgt_anon: *mut anon_vma = unsafe { (*tgt).anon_vma };
let src_anon: *mut anon_vma = unsafe { (*vmg).anon_vma };
// We _can_ have !src, vmg->anon_vma via copy_vma(). In this instance we
// will remove the existing VMA's anon_vma's so there's no scalability
// concerns.
VM_WARN_ON!();
// Case 1 - we will dup_anon_vma() from src into tgt.
if tgt_anon.is_null() && !src_anon.is_null() {
let copied_from: *mut vm_area_struct = unsafe { (*vmg).copied_from };
if vma_is_fork_child(src) {
return false;
}
if vma_is_fork_child(copied_from) {
return false;
}
return true;
}
// Case 2 - we will simply use tgt's anon_vma.
if !tgt_anon.is_null() && src_anon.is_null() {
return !vma_is_fork_child(tgt);
}
// Case 3 - the anon_vma's are already shared.
src_anon == tgt_anon
}

/// init_multi_vma_prep() - Initializer for struct vma_prepare
/// @vp: The vma_prepare struct
/// @vma: The vma that will be altered once locked
/// @vmg: The merge state that will be used to determine adjustment and VMA
///       removal.
fn init_multi_vma_prep(vp: *mut vma_prepare, vma: *mut vm_area_struct, vmg: *mut vma_merge_struct) {
let mut adjust: *mut vm_area_struct = unsafe { core::mem::zeroed() };
let mut remove: *mut *mut vm_area_struct = unsafe { core::ptr::addr_of_mut!((*vp).remove) };
unsafe { memset(vp as *mut (), 0i32, core::mem::size_of::<vma_prepare>() as u64) };
unsafe {
(*vp).vma = vma;
}
unsafe {
(*vp).anon_vma = (*vma).anon_vma;
}
if !vmg.is_null() && ((unsafe { (*vmg).__slate_bits_0.__get___remove_middle() }) as i32) != 0 {
unsafe {
*remove = (*vmg).middle;
}
remove = unsafe { core::ptr::addr_of_mut!((*vp).remove2) };
}
if !vmg.is_null() && ((unsafe { (*vmg).__slate_bits_0.__get___remove_next() }) as i32) != 0 {
unsafe {
*remove = (*vmg).next;
}
}
if !vmg.is_null() && ((unsafe { (*vmg).__slate_bits_0.__get___adjust_middle_start() }) as i32) != 0 {
adjust = unsafe { (*vmg).middle };
} else {
if !vmg.is_null() && ((unsafe { (*vmg).__slate_bits_0.__get___adjust_next_start() }) as i32) != 0 {
adjust = unsafe { (*vmg).next };
} else {
adjust = core::ptr::null_mut::<vm_area_struct>();
}
}
unsafe {
(*vp).adj_next = adjust;
}
if unsafe { (*vp).anon_vma }.is_null() && !adjust.is_null() {
unsafe {
(*vp).anon_vma = (*adjust).anon_vma;
}
}
VM_WARN_ON!();
unsafe {
(*vp).file = (*vma).vm_file;
}
if !unsafe { (*vp).file }.is_null() {
unsafe {
(*vp).mapping = (*(*vma).vm_file).f_mapping;
}
}
if !vmg.is_null() && ((unsafe { (*vmg).__slate_bits_0.__get_skip_vma_uprobe() }) as i32) != 0 {
unsafe {
let __t0 = true;
(*vp).__slate_bits_0.__set_skip_vma_uprobe(__t0);
}
}
}

/// Does this merge require that adjacent VMAs must have adjacent anonymous page
/// offsets in addition to having adjacent vma->vm_pgoff?
///
/// This is only required for MAP_PRIVATE-file backed mappings as the page offset
/// for pure anonymous VMAs is equal to the anonymous page offset.
///
/// Read-only shared mappings (with VMA_SHARED_BIT cleared) are always unfaulted
/// so automatically have correct anonymous page offset (as it is always updated
/// on remap).
///
/// 'Special' mappings in the sense of VDSO, VVAR etc. have !file but would in
/// any case not be candidates for merge nor be mergeable.
fn needs_adjacent_anon_pgoff(vmg: *const vma_merge_struct) -> bool {
!unsafe { (*vmg).file }.is_null() && vma_flags_is_cow_mapping(unsafe { core::ptr::addr_of!((*vmg).__slate_anon_10.vma_flags) })
}

/// Return true if we can merge this (vma_flags,anon_vma,file,vm_pgoff)
/// in front of (at a lower virtual address and file offset than) the vma.
///
/// We cannot merge two vmas if they have differently assigned (non-NULL)
/// anon_vmas, nor if same anon_vma is assigned but offsets incompatible.
///
/// We don't check here for the merged mmap wrapping around the end of pagecache
/// indices (16TB on ia32) because do_mmap() does not permit mmap's which
/// wrap, nor mmaps which cover the final page at index -1UL.
///
/// We assume the vma may be removed as part of the merge.
fn can_vma_merge_before(vmg: *mut vma_merge_struct) -> bool {
if !is_mergeable_vma(vmg, true) {
return false;
}
// merge_next =
if !is_mergeable_anon_vma(vmg, true) {
return false;
}
// merge_next =
if vmg_end_pgoff(vmg as *const vma_merge_struct) != vma_start_pgoff((unsafe { (*vmg).next }) as *const vm_area_struct) {
return false;
}
let __v0: bool = needs_adjacent_anon_pgoff(vmg as *const vma_merge_struct) && vmg_end_anon_pgoff(vmg as *const vma_merge_struct) != vma_start_anon_pgoff((unsafe { (*vmg).next }) as *const vm_area_struct);
if __v0 {
return false;
}
true
}

/// Return true if we can merge this (vma_flags,anon_vma,file,vm_pgoff)
/// beyond (at a higher virtual address and file offset than) the vma.
///
/// We cannot merge two vmas if they have differently assigned (non-NULL)
/// anon_vmas, nor if same anon_vma is assigned but offsets incompatible.
///
/// We assume that vma is not removed as part of the merge.
fn can_vma_merge_after(vmg: *mut vma_merge_struct) -> bool {
if !is_mergeable_vma(vmg, false) {
return false;
}
// merge_next =
if !is_mergeable_anon_vma(vmg, false) {
return false;
}
// merge_next =
if vma_end_pgoff((unsafe { (*vmg).prev }) as *const vm_area_struct) != vmg_start_pgoff(vmg as *const vma_merge_struct) {
return false;
}
let __v0: bool = needs_adjacent_anon_pgoff(vmg as *const vma_merge_struct) && vma_end_anon_pgoff((unsafe { (*vmg).prev }) as *const vm_area_struct) != vmg_start_anon_pgoff(vmg as *const vma_merge_struct);
if __v0 {
return false;
}
true
}

fn __vma_link_file(vma: *mut vm_area_struct, mapping: *mut address_space) {
if vma_is_shared_maywrite(vma as *const vm_area_struct) {
mapping_allow_writable(mapping);
}
flush_dcache_mmap_lock(mapping);
unsafe { mapping_rmap_tree_insert(vma, mapping) };
flush_dcache_mmap_unlock(mapping);
}

/// Requires inode->i_mapping->i_mmap_rwsem
fn __remove_shared_vm_struct(vma: *mut vm_area_struct, mapping: *mut address_space) {
if vma_is_shared_maywrite(vma as *const vm_area_struct) {
mapping_unmap_writable(mapping);
}
flush_dcache_mmap_lock(mapping);
unsafe { mapping_rmap_tree_remove(vma, mapping) };
flush_dcache_mmap_unlock(mapping);
}

/// vma has some anon_vma assigned, and is already inserted on that
/// anon_vma's interval trees.
///
/// Before updating the vma's vm_start / vm_end / vm_pgoff fields, the
/// vma must be removed from the anon_vma's interval trees using
/// anon_rmap_tree_pre_update_vma().
///
/// After the update, the vma will be reinserted using
/// anon_rmap_tree_post_update_vma().
///
/// The entire update must be protected by exclusive mmap_lock and by
/// the root anon_vma's mutex.
fn anon_rmap_tree_pre_update_vma(vma: *mut vm_area_struct) {
let mut avc: *mut anon_vma_chain = unsafe { core::mem::zeroed() };
avc = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*vma).anon_vma_chain) });
while list_entry_is_head!(avc, unsafe { core::ptr::addr_of_mut!((*vma).anon_vma_chain) }, same_vma) == 0 {
unsafe { anon_rmap_tree_remove(avc, (*avc).anon_vma) };
avc = list_next_entry__3!(avc, same_vma);
}
}

fn anon_rmap_tree_post_update_vma(vma: *mut vm_area_struct) {
let mut avc: *mut anon_vma_chain = unsafe { core::mem::zeroed() };
avc = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*vma).anon_vma_chain) });
while list_entry_is_head!(avc, unsafe { core::ptr::addr_of_mut!((*vma).anon_vma_chain) }, same_vma) == 0 {
unsafe { anon_rmap_tree_insert(avc, (*avc).anon_vma) };
avc = list_next_entry__3!(avc, same_vma);
}
}

/// vma_prepare() - Helper function for handling locking VMAs prior to altering
/// @vp: The initialized vma_prepare struct
fn vma_prepare(vp: &vma_prepare) {
if !unsafe { (*(vp as *const vma_prepare)).file }.is_null() {
unsafe { uprobe_munmap((*(vp as *const vma_prepare)).vma, (*(*(vp as *const vma_prepare)).vma).__slate_anon_0.__slate_anon_0.vm_start, (*(*(vp as *const vma_prepare)).vma).__slate_anon_0.__slate_anon_0.vm_end) };
if !unsafe { (*(vp as *const vma_prepare)).adj_next }.is_null() {
unsafe { uprobe_munmap((*(vp as *const vma_prepare)).adj_next, (*(*(vp as *const vma_prepare)).adj_next).__slate_anon_0.__slate_anon_0.vm_start, (*(*(vp as *const vma_prepare)).adj_next).__slate_anon_0.__slate_anon_0.vm_end) };
}
i_mmap_lock_write(unsafe { (*(vp as *const vma_prepare)).mapping });
if !unsafe { (*(vp as *const vma_prepare)).insert }.is_null() && !unsafe { (*(*(vp as *const vma_prepare)).insert).vm_file }.is_null() {
// Put into interval tree now, so instantiated pages
// are visible to arm/parisc __flush_dcache_page
// throughout; but we cannot insert into address
// space until vma start or end is updated.
__vma_link_file(unsafe { (*(vp as *const vma_prepare)).insert }, unsafe { (*(*(*(vp as *const vma_prepare)).insert).vm_file).f_mapping });
}
}
if !unsafe { (*(vp as *const vma_prepare)).anon_vma }.is_null() {
anon_vma_lock_write(unsafe { (*(vp as *const vma_prepare)).anon_vma });
anon_rmap_tree_pre_update_vma(unsafe { (*(vp as *const vma_prepare)).vma });
if !unsafe { (*(vp as *const vma_prepare)).adj_next }.is_null() {
anon_rmap_tree_pre_update_vma(unsafe { (*(vp as *const vma_prepare)).adj_next });
}
}
if !unsafe { (*(vp as *const vma_prepare)).file }.is_null() {
flush_dcache_mmap_lock(unsafe { (*(vp as *const vma_prepare)).mapping });
unsafe { mapping_rmap_tree_remove((*(vp as *const vma_prepare)).vma, (*(vp as *const vma_prepare)).mapping) };
if !unsafe { (*(vp as *const vma_prepare)).adj_next }.is_null() {
unsafe { mapping_rmap_tree_remove((*(vp as *const vma_prepare)).adj_next, (*(vp as *const vma_prepare)).mapping) };
}
}
}

/// vma_complete- Helper function for handling the unlocking after altering VMAs,
/// or for inserting a VMA.
///
/// @vp: The vma_prepare struct
/// @vmi: The vma iterator
/// @mm: The mm_struct
fn vma_complete(vp: *mut vma_prepare, vmi: *mut vma_iterator, mm: *mut mm_struct) {
unsafe {
if !unsafe { (*vp).file }.is_null() {
if !unsafe { (*vp).adj_next }.is_null() {
mapping_rmap_tree_insert((*vp).adj_next, (*vp).mapping);
}
mapping_rmap_tree_insert((*vp).vma, (*vp).mapping);
flush_dcache_mmap_unlock((*vp).mapping);
}
if !unsafe { (*vp).remove }.is_null() && !unsafe { (*vp).file }.is_null() {
__remove_shared_vm_struct((*vp).remove, (*vp).mapping);
if !unsafe { (*vp).remove2 }.is_null() {
__remove_shared_vm_struct((*vp).remove2, (*vp).mapping);
}
} else {
if !unsafe { (*vp).insert }.is_null() {
// split_vma has split insert from vma, and needs
// us to insert it before dropping the locks
// (it may either follow vma or precede it).
vma_iter_store_new(vmi, (*vp).insert);
let __v7: i32 = (*mm).__slate_anon_0.map_count;
(*mm).__slate_anon_0.map_count = __v7.wrapping_add(1);
}
}
if !unsafe { (*vp).anon_vma }.is_null() {
anon_rmap_tree_post_update_vma((*vp).vma);
if !unsafe { (*vp).adj_next }.is_null() {
anon_rmap_tree_post_update_vma((*vp).adj_next);
}
anon_vma_unlock_write((*vp).anon_vma);
}
if !unsafe { (*vp).file }.is_null() {
i_mmap_unlock_write((*vp).mapping);
if ((*vp).__slate_bits_0.__get_skip_vma_uprobe() as i32) == 0 {
uprobe_mmap((*vp).vma);
if !unsafe { (*vp).adj_next }.is_null() {
uprobe_mmap((*vp).adj_next);
}
}
}
if !unsafe { (*vp).remove }.is_null() {
loop {
vma_mark_detached((*vp).remove);
if !unsafe { (*vp).file }.is_null() {
uprobe_munmap((*vp).remove, (*(*vp).remove).__slate_anon_0.__slate_anon_0.vm_start, (*(*vp).remove).__slate_anon_0.__slate_anon_0.vm_end);
fput((*vp).file);
}
if !unsafe { (*(*vp).remove).anon_vma }.is_null() {
unlink_anon_vmas((*vp).remove);
}
let __v4: i32 = (*mm).__slate_anon_0.map_count;
(*mm).__slate_anon_0.map_count = __v4.wrapping_sub(1);
mpol_put(vma_policy!(unsafe { (*vp).remove }));
if unsafe { (*vp).remove2 }.is_null() {
let __ret_warn_on: i32 = ((*(*vp).vma).__slate_anon_0.__slate_anon_0.vm_end < (*(*vp).remove).__slate_anon_0.__slate_anon_0.vm_end) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("723: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 723b - ., 3; .popsection\n# {0}", const 723 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_vma_complete_9, sym __slate_asm_string_vma_complete_10, const 438 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("724: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 724b - ., 4; .popsection\n# {0}", const 724 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
vm_area_free((*vp).remove);
// In mprotect's case 6 (see comments on vma_merge),
// we are removing both mid and next vmas
if !unsafe { (*vp).remove2 }.is_null() {
(*vp).remove = (*vp).remove2;
(*vp).remove2 = core::ptr::null_mut::<vm_area_struct>();
} else {
break;
}
}
}
if !unsafe { (*vp).insert }.is_null() && !unsafe { (*vp).file }.is_null() {
uprobe_mmap((*vp).insert);
}
}
}

/// init_vma_prep() - Initializer wrapper for vma_prepare struct
/// @vp: The vma_prepare struct
/// @vma: The vma that will be altered once locked
fn init_vma_prep(vp: *mut vma_prepare, vma: *mut vm_area_struct) {
init_multi_vma_prep(vp, vma, core::ptr::null_mut::<vma_merge_struct>());
}

/// Can the proposed VMA be merged with the left (previous) VMA taking into
/// account the start position of the proposed range.
fn can_vma_merge_left(vmg: *mut vma_merge_struct) -> bool {
!unsafe { (*vmg).prev }.is_null() && (unsafe { (*(*vmg).prev).__slate_anon_0.__slate_anon_0.vm_end }) == unsafe { (*vmg).start } && can_vma_merge_after(vmg)
}

/// Can the proposed VMA be merged with the right (next) VMA taking into
/// account the end position of the proposed range.
///
/// In addition, if we can merge with the left VMA, ensure that left and right
/// anon_vma's are also compatible.
fn can_vma_merge_right(vmg: *mut vma_merge_struct, can_merge_left: bool) -> bool {
let next: *mut vm_area_struct = unsafe { (*vmg).next };
let __v0: bool = next.is_null() || (unsafe { (*vmg).end }) != unsafe { (*next).__slate_anon_0.__slate_anon_0.vm_start } || !can_vma_merge_before(vmg);
if __v0 {
return false;
}
if !can_merge_left {
return true;
}
// If we can merge with prev (left) and next (right), indicating that
// each VMA's anon_vma is compatible with the proposed anon_vma, this
// does not mean prev and next are compatible with EACH OTHER.
//
// We therefore check this in addition to mergeability to either side.
let prev: *mut vm_area_struct = unsafe { (*vmg).prev };
unsafe { (*prev).anon_vma }.is_null() || unsafe { (*next).anon_vma }.is_null() || (unsafe { (*prev).anon_vma }) == unsafe { (*next).anon_vma }
}

/// Close a vm structure and free it.
#[unsafe(no_mangle)]
extern "C" fn remove_vma(vma: *mut vm_area_struct) {
might_sleep!();
vma_close(vma);
if !unsafe { (*vma).vm_file }.is_null() {
unsafe { fput((*vma).vm_file) };
}
mpol_put(vma_policy!(vma));
unsafe { vm_area_free(vma) };
}

/// Get rid of page table information in the indicated region.
///
/// Called with the mm semaphore held.
#[unsafe(no_mangle)]
extern "C" fn unmap_region(unmap: *mut unmap_desc) {
let mm: *mut mm_struct = unsafe { (*(*unmap).first).vm_mm };
let mut tlb: mmu_gather = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tlb) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 128usize);
}
unsafe { tlb_gather_mmu(core::ptr::addr_of_mut!(tlb), mm) };
update_hiwater_rss(mm);
unsafe { unmap_vmas(core::ptr::addr_of_mut!(tlb), unmap) };
mas_set(unsafe { (*unmap).mas }, unsafe { (*unmap).tree_reset });
unsafe { free_pgtables(core::ptr::addr_of_mut!(tlb), unmap) };
unsafe { tlb_finish_mmu(core::ptr::addr_of_mut!(tlb)) };
}

/// __split_vma() bypasses sysctl_max_map_count checking.  We use this where it
/// has already been checked or doesn't make sense to fail.
/// VMA Iterator will point to the original VMA.
fn __split_vma(vmi: *mut vma_iterator, vma: *mut vm_area_struct, addr: u64, new_below: i32) -> i32 {
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut new: core::mem::MaybeUninit<*mut vm_area_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut vm_area_struct = core::ptr::addr_of_mut!(new) as *mut *mut vm_area_struct;
let mut vp: core::mem::MaybeUninit<vma_prepare> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut vma_prepare = core::ptr::addr_of_mut!(vp) as *mut vma_prepare;
unsafe {
{
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 72usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __ret_warn_on: i32 = ((*vma).__slate_anon_0.__slate_anon_0.vm_start >= addr) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("725: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 725b - ., 3; .popsection\n# {0}", const 725 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___split_vma_12, sym __slate_asm_string___split_vma_13, const 551 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("726: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 726b - ., 4; .popsection\n# {0}", const 726 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
let __ret_warn_on: i32 = ((*vma).__slate_anon_0.__slate_anon_0.vm_end <= addr) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("727: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 727b - ., 3; .popsection\n# {0}", const 727 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___split_vma_10, sym __slate_asm_string___split_vma_11, const 552 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("728: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 728b - ., 4; .popsection\n# {0}", const 728 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
if !unsafe { (*vma).vm_ops }.is_null() && (*(*vma).vm_ops).may_split != None {
*__slate_slot_6 = unsafe { (*(*vma).vm_ops).may_split }.unwrap()(vma, addr);
if *__slate_slot_6 != 0 {
return *__slate_slot_6;
}
}
*__slate_slot_7 = vm_area_dup(vma);
if (*__slate_slot_7).is_null() {
return 12i32.wrapping_neg();
} else {
if new_below != 0 {
(*(*__slate_slot_7)).__slate_anon_0.__slate_anon_0.vm_end = addr;
} else {
(*(*__slate_slot_7)).__slate_anon_0.__slate_anon_0.vm_start = addr;
vma_add_pgoff(*__slate_slot_7, linear_page_delta(vma as *const vm_area_struct, addr));
}
*__slate_slot_6 = 12i32.wrapping_neg();
vma_iter_config(vmi, (*(*__slate_slot_7)).__slate_anon_0.__slate_anon_0.vm_start, (*(*__slate_slot_7)).__slate_anon_0.__slate_anon_0.vm_end);
if vma_iter_prealloc(vmi, *__slate_slot_7) == 0 {
*__slate_slot_6 = vma_dup_policy(vma, *__slate_slot_7);
if *__slate_slot_6 == 0 {
*__slate_slot_6 = anon_vma_clone(*__slate_slot_7, vma, 0u32);
if *__slate_slot_6 != 0 {
mpol_put(vma_policy!(*__slate_slot_7));
} else {
if !unsafe { (*(*__slate_slot_7)).vm_file }.is_null() {
get_file((*(*__slate_slot_7)).vm_file);
}
if !unsafe { (*(*__slate_slot_7)).vm_ops }.is_null() && (*(*(*__slate_slot_7)).vm_ops).open != None {
unsafe { (*(*(*__slate_slot_7)).vm_ops).open }.unwrap()(*__slate_slot_7);
}
vma_start_write(vma);
vma_start_write(*__slate_slot_7);
init_vma_prep(core::ptr::addr_of_mut!(*__slate_slot_8), vma);
(*__slate_slot_8).insert = *__slate_slot_7;
vma_prepare(&(*core::ptr::addr_of_mut!(*__slate_slot_8)));
// Get rid of huge pages and shared page tables straddling the split
// boundary.
vma_adjust_trans_huge(vma, (*vma).__slate_anon_0.__slate_anon_0.vm_start, addr, core::ptr::null_mut::<vm_area_struct>());
if is_vm_hugetlb_page(vma as *const vm_area_struct) {
hugetlb_split(vma, addr);
}
if new_below != 0 {
(*vma).__slate_anon_0.__slate_anon_0.vm_start = addr;
vma_add_pgoff(vma, linear_page_delta(*__slate_slot_7 as *const vm_area_struct, addr));
} else {
(*vma).__slate_anon_0.__slate_anon_0.vm_end = addr;
}
// vma_complete stores the new vma
vma_complete(core::ptr::addr_of_mut!(*__slate_slot_8), vmi, (*vma).vm_mm);
validate_mm!();
// Success.
if new_below != 0 {
vma_next(vmi);
} else {
vma_prev(vmi);
}
return 0;
}
}
vma_iter_free(vmi);
}
vm_area_free(*__slate_slot_7);
return *__slate_slot_6;
}
}
}

/// Split a vma into two pieces at address 'addr', a new vma is allocated
/// either for the first part or the tail.
fn split_vma(vmi: *mut vma_iterator, vma: *mut vm_area_struct, addr: u64, new_below: i32) -> i32 {
if (unsafe { (*(*vma).vm_mm).__slate_anon_0.map_count }) >= get_sysctl_max_map_count() {
return 12i32.wrapping_neg();
}
__split_vma(vmi, vma, addr, new_below)
}

/// dup_anon_vma() - Helper function to duplicate anon_vma on VMA merge in the
/// instance that the destination VMA has no anon_vma but the source does.
///
/// @dst: The destination VMA
/// @src: The source VMA
/// @dup: Pointer to the destination VMA when successful.
///
/// Returns: 0 on success.
fn dup_anon_vma(dst: *mut vm_area_struct, src: *mut vm_area_struct, dup: &mut *mut vm_area_struct) -> i32 {
// There are three cases to consider for correctly propagating
// anon_vma's on merge.
//
// The first is trivial - neither VMA has anon_vma, we need not do
// anything.
//
// The second where both have anon_vma is also a no-op, as they must
// then be the same, so there is simply nothing to copy.
//
// Here we cover the third - if the destination VMA has no anon_vma,
// that is it is unfaulted, we need to ensure that the newly merged
// range is referenced by the anon_vma's of the source.
if !unsafe { (*src).anon_vma }.is_null() && unsafe { (*dst).anon_vma }.is_null() {
let mut ret: i32 = 0;
vma_assert_write_locked(dst);
unsafe {
(*dst).anon_vma = (*src).anon_vma;
}
ret = unsafe { anon_vma_clone(dst, src, 1u32) };
if ret != 0 {
return ret;
}
unsafe {
*(dup as *mut *mut vm_area_struct) = dst;
}
}
0
}

/// Based on the vmg flag indicating whether we need to adjust the vm_start field
/// for the middle or next VMA, we calculate what the range of the newly adjusted
/// VMA ought to be, and set the VMA's range accordingly.
fn vmg_adjust_set_range(vmg: *mut vma_merge_struct) {
if ((unsafe { (*vmg).__slate_bits_0.__get___adjust_middle_start() }) as i32) != 0 {
// vmg->start    vmg->end
// |             |
// v    merge    v
// <------------->
//         delta
//        <------>
// |------|----------------|
// | prev |    middle      |
// |------|----------------|
//        ^
//        |
//        middle->vm_start
let middle: *mut vm_area_struct = unsafe { (*vmg).middle };
let delta: u64 = unsafe { (*vmg).end }.wrapping_sub(unsafe { (*middle).__slate_anon_0.__slate_anon_0.vm_start });
__vma_set_range(middle, unsafe { (*vmg).end }, unsafe { (*middle).__slate_anon_0.__slate_anon_0.vm_end });
vma_add_pgoff(middle, delta >> 12i32);
} else {
if ((unsafe { (*vmg).__slate_bits_0.__get___adjust_next_start() }) as i32) != 0 {
//                Originally:
//
//            vmg->start   vmg->end
//            |            |
//            v    merge   v
//            <------------>
//            .            .
// merge_existing_range() updates to:
//            .            .
// vmg->start vmg->end     .
// |          |            .
// v  retain  v            .
// <---------->            .
//             delta       .
//            <----->      .
// |----------------|------|
// |    middle      | next |
// |----------------|------|
//                  ^
//                  |
//                  next->vm_start
let next: *mut vm_area_struct = unsafe { (*vmg).next };
let delta: u64 = unsafe { (*next).__slate_anon_0.__slate_anon_0.vm_start }.wrapping_sub(unsafe { (*vmg).end });
__vma_set_range(next, unsafe { (*vmg).end }, unsafe { (*next).__slate_anon_0.__slate_anon_0.vm_end });
vma_sub_pgoff(next, delta >> 12i32);
}
}
}

/// Actually perform the VMA merge operation.
///
/// IMPORTANT: We guarantee that, should vmg->give_up_on_oom is set, to not
/// modify any VMAs or cause inconsistent state should an OOM condition arise.
///
/// Returns 0 on success, or an error value on failure.
fn commit_merge(vmg: *mut vma_merge_struct) -> i32 {
let mut vma: *mut vm_area_struct = unsafe { core::mem::zeroed() };
let mut vp: vma_prepare = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(vp) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 72usize);
}
if ((unsafe { (*vmg).__slate_bits_0.__get___adjust_next_start() }) as i32) != 0 {
// We manipulate middle and adjust next, which is the target.
vma = unsafe { (*vmg).middle };
vma_iter_config(unsafe { (*vmg).vmi }, unsafe { (*vmg).end }, unsafe { (*(*vmg).next).__slate_anon_0.__slate_anon_0.vm_end });
} else {
vma = unsafe { (*vmg).target };
// Note: vma iterator must be pointing to 'start'.
vma_iter_config(unsafe { (*vmg).vmi }, unsafe { (*vmg).start }, unsafe { (*vmg).end });
}
init_multi_vma_prep(core::ptr::addr_of_mut!(vp), vma, vmg);
// If vmg->give_up_on_oom is set, we're safe, because we don't actually
// manipulate any VMAs until we succeed at preallocation.
//
// Past this point, we will not return an error.
if vma_iter_prealloc(unsafe { (*vmg).vmi }, vma) != 0 {
return 12i32.wrapping_neg();
}
vma_prepare(unsafe { &(*core::ptr::addr_of_mut!(vp)) });
// THP pages may need to do additional splits if we increase
// middle->vm_start.
vma_adjust_trans_huge(vma, unsafe { (*vmg).start }, unsafe { (*vmg).end }, if ((unsafe { (*vmg).__slate_bits_0.__get___adjust_middle_start() }) as i32) != 0 { unsafe { (*vmg).middle } } else { core::ptr::null_mut::<vm_area_struct>() });
vma_set_range(vma, unsafe { (*vmg).start }, unsafe { (*vmg).end }, vmg_start_pgoff(vmg as *const vma_merge_struct), vmg_start_anon_pgoff(vmg as *const vma_merge_struct));
vmg_adjust_set_range(vmg);
vma_iter_store_overwrite(unsafe { (*vmg).vmi }, unsafe { (*vmg).target });
vma_complete(core::ptr::addr_of_mut!(vp), unsafe { (*vmg).vmi }, unsafe { (*vma).vm_mm });
0
}

/// We can only remove VMAs when merging if they do not have a close hook.
fn can_merge_remove_vma(vma: *mut vm_area_struct) -> bool {
unsafe { (*vma).vm_ops }.is_null() || (unsafe { (*(*vma).vm_ops).close }) == None
}

/// vma_merge_existing_range - Attempt to merge VMAs based on a VMA having its
/// attributes modified.
///
/// @vmg: Describes the modifications being made to a VMA and associated
///       metadata.
///
/// When the attributes of a range within a VMA change, then it might be possible
/// for immediately adjacent VMAs to be merged into that VMA due to having
/// identical properties.
///
/// This function checks for the existence of any such mergeable VMAs and updates
/// the maple tree describing the @vmg->middle->vm_mm address space to account
/// for this, as well as any VMAs shrunk/expanded/deleted as a result of this
/// merge.
///
/// As part of this operation, if a merge occurs, the @vmg object will have its
/// vma, start, end, and pgoff fields modified to execute the merge. Subsequent
/// calls to this function should reset these fields.
///
/// Returns: The merged VMA if merge succeeds, or NULL otherwise.
///
/// ASSUMPTIONS:
/// - The caller must assign the VMA to be modified to @vmg->middle.
/// - The caller must have set @vmg->prev to the previous VMA, if there is one.
/// - The caller must not set @vmg->next, as we determine this.
/// - The caller must hold a WRITE lock on the mm_struct->mmap_lock.
/// - vmi must be positioned within [@vmg->middle->vm_start, @vmg->middle->vm_end).
fn vma_merge_existing_range(vmg: *mut vma_merge_struct) -> *mut vm_area_struct {
let mut __slate_compound_20: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_21: [i32; 1] = unsafe { core::mem::zeroed() };
let mut __slate_compound_22: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_23: [i32; 1] = unsafe { core::mem::zeroed() };
let mut __slate_compound_24: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_25: [i32; 1] = unsafe { core::mem::zeroed() };
let mut __slate_compound_26: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_27: [i32; 4] = unsafe { core::mem::zeroed() };
let mut __v0: bool = unsafe { core::mem::zeroed() };
//     |<------------->| OR
// |<----------------->|
// *************-------|
//    middle     next
// shrink/delete extend
let mut pglen: u64 = unsafe { core::mem::zeroed() };
let mut prev_sticky: core::mem::MaybeUninit<__SlateRecord_mm_types_h_871> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut __SlateRecord_mm_types_h_871 = core::ptr::addr_of_mut!(prev_sticky) as *mut __SlateRecord_mm_types_h_871;
let mut next_sticky: core::mem::MaybeUninit<__SlateRecord_mm_types_h_871> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut __SlateRecord_mm_types_h_871 = core::ptr::addr_of_mut!(next_sticky) as *mut __SlateRecord_mm_types_h_871;
// If we cannot delete next, then we can reduce the operation to merging
// prev and middle (thereby deleting middle).
let mut __v4: bool = unsafe { core::mem::zeroed() };
// If we need to remove middle in its entirety but are unable to do so,
// we have no sensible recourse but to abort the merge.
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut merge_both: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut bool = core::ptr::addr_of_mut!(merge_both) as *mut bool;
let mut merge_right: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut bool = core::ptr::addr_of_mut!(merge_right) as *mut bool;
let mut merge_left: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut bool = core::ptr::addr_of_mut!(merge_left) as *mut bool;
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut right_side: bool = unsafe { core::mem::zeroed() };
let mut left_side: bool = unsafe { core::mem::zeroed() };
let mut start: u64 = unsafe { core::mem::zeroed() };
let mut anon_dup: *mut vm_area_struct = unsafe { core::mem::zeroed() };
let mut next: core::mem::MaybeUninit<*mut vm_area_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut vm_area_struct = core::ptr::addr_of_mut!(next) as *mut *mut vm_area_struct;
let mut prev: *mut vm_area_struct = unsafe { core::mem::zeroed() };
let mut middle: *mut vm_area_struct = unsafe { core::mem::zeroed() };
let mut sticky_flags: core::mem::MaybeUninit<__SlateRecord_mm_types_h_871> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut __SlateRecord_mm_types_h_871 = core::ptr::addr_of_mut!(sticky_flags) as *mut __SlateRecord_mm_types_h_871;
unsafe {
core::ptr::write(__slate_slot_19, vma_flags_and_mask(core::ptr::addr_of_mut!((*vmg).__slate_anon_10.vma_flags) as *const __SlateRecord_mm_types_h_871, __mk_vma_flags(*{
__slate_compound_24 = {
let mut __t6: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t6
};
core::ptr::addr_of_mut!(__slate_compound_24)
}, 1u64, (*{
__slate_compound_25 = [11i32];
core::ptr::addr_of_mut!(__slate_compound_25)
}).as_ptr() as *const i32)));
middle = (*vmg).middle;
prev = (*vmg).prev;
let __slate_auto_bytes_1 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
anon_dup = core::ptr::null_mut::<vm_area_struct>();
start = (*vmg).start;
let end: u64 = (*vmg).end;
left_side = !middle.is_null() && start == (*middle).__slate_anon_0.__slate_anon_0.vm_start;
right_side = !middle.is_null() && end == (*middle).__slate_anon_0.__slate_anon_0.vm_end;
err = 0;
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_3 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_4 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 1usize);
mmap_assert_write_locked((*vmg).mm as *const mm_struct);
VM_WARN_ON_VMG!(); // We are modifying a VMA, so caller must specify.
VM_WARN_ON_VMG!(); // We set this.
VM_WARN_ON_VMG!();
VM_WARN_ON_VMG!();
// If middle == prev, then we are offset into a VMA. Otherwise, if we are
// not, we must span a portion of the VMA.
VM_WARN_ON_VMG!();
// The vmi must be positioned within vmg->middle.
VM_WARN_ON_VMG!();
// An existing merge can never be used by the mremap() logic.
VM_WARN_ON_VMG!();
(*vmg).state = 2;
// If a special mapping or if the range being modified is neither at the
// furthermost left or right side of the VMA, then we have no chance of
// merging and should abort.
if vma_flags_test_any_mask(core::ptr::addr_of_mut!((*vmg).__slate_anon_10.vma_flags) as *const __SlateRecord_mm_types_h_871, __mk_vma_flags(*{
__slate_compound_26 = {
let mut __t7: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t7
};
core::ptr::addr_of_mut!(__slate_compound_26)
}, 4u64, (*{
__slate_compound_27 = [14i32, 18i32, 10i32, 28i32];
core::ptr::addr_of_mut!(__slate_compound_27)
}).as_ptr() as *const i32)) || !left_side && !right_side {
return core::ptr::null_mut::<vm_area_struct>();
} else {
if left_side {
*__slate_slot_9 = can_vma_merge_left(vmg);
} else {
*__slate_slot_9 = false;
}
if right_side {
let __v6: *mut vm_area_struct = vma_iter_next_range((*vmg).vmi);
(*vmg).next = __v6;
*__slate_slot_16 = __v6;
vma_iter_prev_range((*vmg).vmi);
*__slate_slot_8 = can_vma_merge_right(vmg, *__slate_slot_9);
} else {
*__slate_slot_8 = false;
*__slate_slot_16 = core::ptr::null_mut::<vm_area_struct>();
}
if *__slate_slot_9 {
vma_prev((*vmg).vmi);
} else {
if !(*__slate_slot_8) {
return core::ptr::null_mut::<vm_area_struct>();
}
}
// If merging prev, position iterator there.
// If we have nothing to merge, abort.
*__slate_slot_7 = *__slate_slot_9 && *__slate_slot_8;
// If we span the entire VMA, a merge implies it will be deleted.
let __t5 = left_side && right_side;
(*vmg).__slate_bits_0.__set___remove_middle(__t5);
if ((*vmg).__slate_bits_0.__get___remove_middle() as i32) != 0 {
__v5 = !can_merge_remove_vma(middle);
} else {
__v5 = false;
}
if __v5 {
return core::ptr::null_mut::<vm_area_struct>();
} else {
// If we merge both VMAs, then next is also deleted. This implies
// merge_will_delete_vma also.
(*vmg).__slate_bits_0.__set___remove_next(*__slate_slot_7);
if ((*vmg).__slate_bits_0.__get___remove_next() as i32) != 0 {
__v4 = !can_merge_remove_vma(*__slate_slot_16);
} else {
__v4 = false;
}
if __v4 {
let __t4 = false;
(*vmg).__slate_bits_0.__set___remove_next(__t4);
*__slate_slot_8 = false;
*__slate_slot_7 = false;
}
// No matter what happens, we will be adjusting middle.
vma_start_write(middle);
if *__slate_slot_8 {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
vma_start_write(*__slate_slot_16);
(*vmg).target = *__slate_slot_16;
*__slate_slot_3 = vma_flags_and_mask(core::ptr::addr_of_mut!((*(*__slate_slot_16)).__slate_anon_3.flags) as *const __SlateRecord_mm_types_h_871, __mk_vma_flags(*{
__slate_compound_22 = {
let mut __t3: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t3
};
core::ptr::addr_of_mut!(__slate_compound_22)
}, 1u64, (*{
__slate_compound_23 = [11i32];
core::ptr::addr_of_mut!(__slate_compound_23)
}).as_ptr() as *const i32));
vma_flags_set_mask(core::ptr::addr_of_mut!(*__slate_slot_19), *__slate_slot_3);
}
if *__slate_slot_9 {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
vma_start_write(prev);
(*vmg).target = prev;
*__slate_slot_2 = vma_flags_and_mask(core::ptr::addr_of_mut!((*prev).__slate_anon_3.flags) as *const __SlateRecord_mm_types_h_871, __mk_vma_flags(*{
__slate_compound_20 = {
let mut __t2: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t2
};
core::ptr::addr_of_mut!(__slate_compound_20)
}, 1u64, (*{
__slate_compound_21 = [11i32];
core::ptr::addr_of_mut!(__slate_compound_21)
}).as_ptr() as *const i32));
vma_flags_set_mask(core::ptr::addr_of_mut!(*__slate_slot_19), *__slate_slot_2);
}
if *__slate_slot_7 {
// |<-------------------->|
// |-------********-------|
//   prev   middle   next
//  extend  delete  delete
(*vmg).start = (*prev).__slate_anon_0.__slate_anon_0.vm_start;
(*vmg).end = (*(*__slate_slot_16)).__slate_anon_0.__slate_anon_0.vm_end;
(*vmg).pgoff = vma_start_pgoff(prev as *const vm_area_struct);
(*vmg).anon_pgoff = vma_start_anon_pgoff(prev as *const vm_area_struct);
// We already ensured anon_vma compatibility above, so now it's
// simply a case of, if prev has no anon_vma object, which of
// next or middle contains the anon_vma we must duplicate.
err = dup_anon_vma(prev, if !unsafe { (*(*__slate_slot_16)).anon_vma }.is_null() { *__slate_slot_16 } else { middle }, &mut (*core::ptr::addr_of_mut!(anon_dup)));
} else {
if *__slate_slot_9 {
// |<------------>|      OR
// |<----------------->|
// |-------*************
//   prev     middle
//  extend shrink/delete
(*vmg).start = (*prev).__slate_anon_0.__slate_anon_0.vm_start;
(*vmg).pgoff = vma_start_pgoff(prev as *const vm_area_struct);
(*vmg).anon_pgoff = vma_start_anon_pgoff(prev as *const vm_area_struct);
if ((*vmg).__slate_bits_0.__get___remove_middle() as i32) == 0 {
let __t0 = true;
(*vmg).__slate_bits_0.__set___adjust_middle_start(__t0);
}
err = dup_anon_vma(prev, middle, &mut (*core::ptr::addr_of_mut!(anon_dup)));
} else {
// merge_right
pglen = vmg_pages(vmg as *const vma_merge_struct);
VM_WARN_ON_VMG!();
// If we are offset into a VMA, then prev must be middle.
VM_WARN_ON_VMG!();
if ((*vmg).__slate_bits_0.__get___remove_middle() as i32) != 0 {
(*vmg).end = (*(*__slate_slot_16)).__slate_anon_0.__slate_anon_0.vm_end;
(*vmg).pgoff = vma_start_pgoff(*__slate_slot_16 as *const vm_area_struct).wrapping_sub(pglen);
(*vmg).anon_pgoff = vma_start_anon_pgoff(*__slate_slot_16 as *const vm_area_struct).wrapping_sub(pglen);
} else {
// We shrink middle and expand next.
let __t1 = true;
(*vmg).__slate_bits_0.__set___adjust_next_start(__t1);
(*vmg).start = (*middle).__slate_anon_0.__slate_anon_0.vm_start;
(*vmg).end = start;
(*vmg).pgoff = vma_start_pgoff(middle as *const vm_area_struct);
(*vmg).anon_pgoff = vma_start_anon_pgoff(middle as *const vm_area_struct);
}
err = dup_anon_vma(*__slate_slot_16, middle, &mut (*core::ptr::addr_of_mut!(anon_dup)));
}
}
if err != 0 {
__v0 = true;
} else {
__v0 = commit_merge(vmg) != 0;
}
if __v0 {
vma_iter_set((*vmg).vmi, start);
vma_iter_load((*vmg).vmi);
if !anon_dup.is_null() {
unlink_anon_vmas(anon_dup);
}
// This means we have failed to clone anon_vma's correctly, but no
// actual changes to VMAs have occurred, so no harm no foul - if the
// user doesn't want this reported and instead just wants to give up on
// the merge, allow it.
if ((*vmg).__slate_bits_0.__get_give_up_on_oom() as i32) == 0 {
(*vmg).state = 1;
}
return core::ptr::null_mut::<vm_area_struct>();
} else {
vma_set_flags_mask((*vmg).target, *__slate_slot_19);
khugepaged_enter_vma((*vmg).target, (*vmg).__slate_anon_10.vm_flags);
(*vmg).state = 3;
return (*vmg).target;
}
}
}
}
}

/// * vma_merge_new_range - Attempt to merge a new VMA into address space
/// *
/// * @vmg: Describes the VMA we are adding, in the range @vmg->start to @vmg->end
/// *       (exclusive), which we try to merge with any adjacent VMAs if possible.
/// *
/// * We are about to add a VMA to the address space starting at @vmg->start and
/// * ending at @vmg->end. There are three different possible scenarios:
/// *
/// * 1. There is a VMA with identical properties immediately adjacent to the
/// *    proposed new VMA [@vmg->start, @vmg->end) either before or after it -
/// *    EXPAND that VMA:
/// *
/// * Proposed:       |-----|  or  |-----|
/// * Existing:  |----|                  |----|
/// *
/// * 2. There are VMAs with identical properties immediately adjacent to the
/// *    proposed new VMA [@vmg->start, @vmg->end) both before AND after it -
/// *    EXPAND the former and REMOVE the latter:
/// *
/// * Proposed:       |-----|
/// * Existing:  |----|     |----|
/// *
/// * 3. There are no VMAs immediately adjacent to the proposed new VMA or those
/// *    VMAs do not have identical attributes - NO MERGE POSSIBLE.
/// *
/// * In instances where we can merge, this function returns the expanded VMA which
/// * will have its range adjusted accordingly and the underlying maple tree also
/// * adjusted.
/// *
/// * Returns: In instances where no merge was possible, NULL. Otherwise, a pointer
/// *          to the VMA we expanded.
/// *
/// * This function adjusts @vmg to provide @vmg->next if not already specified,
/// * and adjusts [@vmg->start, @vmg->end) to span the expanded range.
/// *
/// * ASSUMPTIONS:
/// * - The caller must hold a WRITE lock on the mm_struct->mmap_lock.
/// * - The caller must have determined that [@vmg->start, @vmg->end) is empty,
///     other than VMAs that will be unmapped should the operation succeed.
/// * - The caller must have specified the previous vma in @vmg->prev.
/// * - The caller must have specified the next vma in @vmg->next.
/// * - The caller must have positioned the vmi at or before the gap.
#[unsafe(no_mangle)]
extern "C" fn vma_merge_new_range(vmg: *mut vma_merge_struct) -> *mut vm_area_struct {
let mut __slate_compound_0: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: [i32; 4] = unsafe { core::mem::zeroed() };
let prev: *mut vm_area_struct = unsafe { (*vmg).prev };
let next: *mut vm_area_struct = unsafe { (*vmg).next };
let end: u64 = unsafe { (*vmg).end };
let mut can_merge_left: bool = false;
let mut can_merge_right: bool = false;
mmap_assert_write_locked((unsafe { (*vmg).mm }) as *const mm_struct);
VM_WARN_ON_VMG!();
VM_WARN_ON_VMG!();
// vmi must point at or before the gap.
VM_WARN_ON_VMG!();
unsafe {
(*vmg).state = 2;
}
// Special VMAs are unmergeable, also if no prev/next.
if vma_flags_test_any_mask((unsafe { core::ptr::addr_of_mut!((*vmg).__slate_anon_10.vma_flags) }) as *const __SlateRecord_mm_types_h_871, __mk_vma_flags(unsafe { *{
__slate_compound_0 = {
let mut __t0: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t0
};
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 4u64, unsafe { (*{
__slate_compound_1 = [14i32, 18i32, 10i32, 28i32];
core::ptr::addr_of_mut!(__slate_compound_1)
}).as_ptr() as *const i32 })) || prev.is_null() && next.is_null() {
return core::ptr::null_mut::<vm_area_struct>();
}
can_merge_left = can_vma_merge_left(vmg);
can_merge_right = ((unsafe { (*vmg).__slate_bits_0.__get_just_expand() }) as i32) == 0 && can_vma_merge_right(vmg, can_merge_left);
// If we can merge with the next VMA, adjust vmg accordingly.
if can_merge_right {
unsafe {
(*vmg).end = (*next).__slate_anon_0.__slate_anon_0.vm_end;
}
unsafe {
(*vmg).target = next;
}
}
// If we can merge with the previous VMA, adjust vmg accordingly.
if can_merge_left {
unsafe {
(*vmg).start = (*prev).__slate_anon_0.__slate_anon_0.vm_start;
}
unsafe {
(*vmg).target = prev;
}
unsafe {
(*vmg).pgoff = vma_start_pgoff(prev as *const vm_area_struct);
}
unsafe {
(*vmg).anon_pgoff = vma_start_anon_pgoff(prev as *const vm_area_struct);
}
// If this merge would result in removal of the next VMA but we
// are not permitted to do so, reduce the operation to merging
// prev and vma.
let __v3: bool = can_merge_right && !can_merge_remove_vma(next);
if __v3 {
unsafe {
(*vmg).end = end;
}
}
// In expand-only case we are already positioned at prev.
if ((unsafe { (*vmg).__slate_bits_0.__get_just_expand() }) as i32) == 0 {
// Equivalent to going to the previous range.
vma_prev(unsafe { (*vmg).vmi });
}
}
// Now try to expand adjacent VMA(s). This takes care of removing the
// following VMA if we have VMAs on both sides.
let __v4: bool = !unsafe { (*vmg).target }.is_null() && vma_expand(vmg) == 0;
if __v4 {
khugepaged_enter_vma(unsafe { (*vmg).target }, unsafe { (*vmg).__slate_anon_10.vm_flags });
unsafe {
(*vmg).state = 3;
}
return unsafe { (*vmg).target };
}
core::ptr::null_mut::<vm_area_struct>()
}

/// vma_merge_copied_range - Attempt to merge a VMA that is being copied by
/// mremap()
///
/// @vmg: Describes the VMA we are adding, in the copied-to range @vmg->start to
///       @vmg->end (exclusive), which we try to merge with any adjacent VMAs if
///       possible.
///
/// vmg->prev, next, start, end, pgoff should all be relative to the COPIED TO
/// range, i.e. the target range for the VMA.
///
/// Returns: In instances where no merge was possible, NULL. Otherwise, a pointer
///          to the VMA we expanded.
///
/// ASSUMPTIONS: Same as vma_merge_new_range(), except vmg->middle must contain
///              the copied-from VMA.
fn vma_merge_copied_range(vmg: *mut vma_merge_struct) -> *mut vm_area_struct {
// We must have a copied-from VMA.
VM_WARN_ON_VMG!();
unsafe {
(*vmg).copied_from = (*vmg).middle;
}
unsafe {
(*vmg).middle = core::ptr::null_mut::<vm_area_struct>();
}
vma_merge_new_range(vmg)
}

/// vma_expand - Expand an existing VMA
///
/// @vmg: Describes a VMA expansion operation.
///
/// Expand @vma to vmg->start and vmg->end.  Can expand off the start and end.
/// Will expand over vmg->next if it's different from vmg->target and vmg->end ==
/// vmg->next->vm_end.  Checking if the vmg->target can expand and merge with
/// vmg->next needs to be handled by the caller.
///
/// Returns: 0 on success.
///
/// ASSUMPTIONS:
/// - The caller must hold a WRITE lock on the mm_struct->mmap_lock.
/// - The caller must have set @vmg->target and @vmg->next.
#[unsafe(no_mangle)]
extern "C" fn vma_expand(vmg: *mut vma_merge_struct) -> i32 {
let mut __slate_compound_8: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: [i32; 1] = unsafe { core::mem::zeroed() };
let mut __slate_compound_10: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: [i32; 1] = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: [i32; 1] = unsafe { core::mem::zeroed() };
let mut next_sticky: core::mem::MaybeUninit<__SlateRecord_mm_types_h_871> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut __SlateRecord_mm_types_h_871 = core::ptr::addr_of_mut!(next_sticky) as *mut __SlateRecord_mm_types_h_871;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut target_sticky: core::mem::MaybeUninit<__SlateRecord_mm_types_h_871> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut __SlateRecord_mm_types_h_871 = core::ptr::addr_of_mut!(target_sticky) as *mut __SlateRecord_mm_types_h_871;
let mut sticky_flags: core::mem::MaybeUninit<__SlateRecord_mm_types_h_871> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut __SlateRecord_mm_types_h_871 = core::ptr::addr_of_mut!(sticky_flags) as *mut __SlateRecord_mm_types_h_871;
let mut remove_next: bool = unsafe { core::mem::zeroed() };
let mut next: *mut vm_area_struct = unsafe { core::mem::zeroed() };
let mut target: *mut vm_area_struct = unsafe { core::mem::zeroed() };
let mut anon_dup: *mut vm_area_struct = unsafe { core::mem::zeroed() };
unsafe {
{
anon_dup = core::ptr::null_mut::<vm_area_struct>();
target = (*vmg).target;
next = (*vmg).next;
remove_next = false;
core::ptr::write(__slate_slot_3, vma_flags_and_mask(core::ptr::addr_of_mut!((*vmg).__slate_anon_10.vma_flags) as *const __SlateRecord_mm_types_h_871, __mk_vma_flags(*{
__slate_compound_10 = {
let mut __t2: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t2
};
core::ptr::addr_of_mut!(__slate_compound_10)
}, 1u64, (*{
__slate_compound_11 = [11i32];
core::ptr::addr_of_mut!(__slate_compound_11)
}).as_ptr() as *const i32)));
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 0;
mmap_assert_write_locked((*vmg).mm as *const mm_struct);
vma_start_write(target);
*__slate_slot_2 = vma_flags_and_mask(core::ptr::addr_of_mut!((*target).__slate_anon_3.flags) as *const __SlateRecord_mm_types_h_871, __mk_vma_flags(*{
__slate_compound_12 = {
let mut __t3: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t3
};
core::ptr::addr_of_mut!(__slate_compound_12)
}, 1u64, (*{
__slate_compound_13 = [11i32];
core::ptr::addr_of_mut!(__slate_compound_13)
}).as_ptr() as *const i32));
if !next.is_null() && target != next && (*vmg).end == (*next).__slate_anon_0.__slate_anon_0.vm_end {
remove_next = true;
}
}
// We must have a target.
VM_WARN_ON_VMG!();
// This should have already been checked by this point.
VM_WARN_ON_VMG!();
// Not merging but overwriting any part of next is not handled.
VM_WARN_ON_VMG!();
// Only handles expanding.
VM_WARN_ON_VMG!();
vma_flags_set_mask(core::ptr::addr_of_mut!(*__slate_slot_3), *__slate_slot_2);
// If we are removing the next VMA or copying from a VMA
// (e.g. mremap()'ing), we must propagate anon_vma state.
//
// Note that, by convention, callers ignore OOM for this case, so
// we don't need to account for vmg->give_up_on_mm here.
if remove_next {
ret = dup_anon_vma(target, next, &mut (*core::ptr::addr_of_mut!(anon_dup)));
}
if ret == 0 && !unsafe { (*vmg).copied_from }.is_null() {
ret = dup_anon_vma(target, (*vmg).copied_from, &mut (*core::ptr::addr_of_mut!(anon_dup)));
}
if ret != 0 {
return ret;
} else {
if remove_next {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
vma_start_write(next);
let __t0 = true;
(*vmg).__slate_bits_0.__set___remove_next(__t0);
*__slate_slot_0 = vma_flags_and_mask(core::ptr::addr_of_mut!((*next).__slate_anon_3.flags) as *const __SlateRecord_mm_types_h_871, __mk_vma_flags(*{
__slate_compound_8 = {
let mut __t1: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t1
};
core::ptr::addr_of_mut!(__slate_compound_8)
}, 1u64, (*{
__slate_compound_9 = [11i32];
core::ptr::addr_of_mut!(__slate_compound_9)
}).as_ptr() as *const i32));
vma_flags_set_mask(core::ptr::addr_of_mut!(*__slate_slot_3), *__slate_slot_0);
}
if commit_merge(vmg) != 0 {
if !anon_dup.is_null() {
unlink_anon_vmas(anon_dup);
}
// If the user requests that we just give upon OOM, we are safe to do so
// here, as commit merge provides this contract to us. Nothing has been
// changed - no harm no foul, just don't report it.
if ((*vmg).__slate_bits_0.__get_give_up_on_oom() as i32) == 0 {
(*vmg).state = 1;
}
return 12i32.wrapping_neg();
} else {
vma_set_flags_mask(target, *__slate_slot_3);
return 0;
}
}
}
}

/// vma_shrink() - Shrink the end of a VMA
/// @vmi: The vma iterator
/// @vma: The VMA to modify
/// @end: The new end
///
/// Note that the caller may only shrink the end of the VMA.
///
/// Returns: 0 on success, -ENOMEM otherwise
#[unsafe(no_mangle)]
extern "C" fn vma_shrink(vmi: *mut vma_iterator, vma: *mut vm_area_struct, end: u64) -> i32 {
let mut vp: vma_prepare = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(vp) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 72usize);
}
VM_WARN_ON_ONCE!();
vma_iter_config(vmi, end, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end });
if vma_iter_prealloc(vmi, core::ptr::null_mut::<vm_area_struct>()) != 0 {
return 12i32.wrapping_neg();
}
vma_start_write(vma);
init_vma_prep(core::ptr::addr_of_mut!(vp), vma);
vma_prepare(unsafe { &(*core::ptr::addr_of_mut!(vp)) });
vma_adjust_trans_huge(vma, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }, end, core::ptr::null_mut::<vm_area_struct>());
vma_iter_clear(vmi);
__vma_set_range(vma, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }, end);
vma_complete(core::ptr::addr_of_mut!(vp), vmi, unsafe { (*vma).vm_mm });
validate_mm!();
0
}

fn vms_clear_ptes(vms: *mut vma_munmap_struct, mas_detach: *mut ma_state, mm_wr_locked: bool) {
let mut unmap: unmap_desc = {
let mut __t0: unmap_desc = unsafe { core::mem::zeroed() };
__t0.mas = mas_detach;
__t0.first = unsafe { (*vms).vma };
__t0.pg_start = unsafe { (*vms).unmap_start };
__t0.pg_end = unsafe { (*vms).unmap_end };
__t0.vma_start = unsafe { (*vms).start };
__t0.vma_end = unsafe { (*vms).end };
__t0.tree_end = (unsafe { (*vms).vma_count }) as u64;
__t0.tree_reset = 1u64;
__t0.mm_wr_locked = mm_wr_locked;
__t0
};
// start and end may be different if there is no prev or next vma.
// The tree limits and reset differ from the normal case since it's a
// side-tree
// We can free page tables without write-locking mmap_lock because VMAs
// were isolated before we downgraded mmap_lock.
if !unsafe { (*vms).clear_ptes } {
return;
}
// Nothing to do
mas_set(mas_detach, 1u64);
unmap_region(core::ptr::addr_of_mut!(unmap));
unsafe {
(*vms).clear_ptes = false;
}
}

fn vms_clean_up_area(vms: *mut vma_munmap_struct, mas_detach: *mut ma_state) {
if (unsafe { (*vms).nr_pages }) == 0 {
return;
}
vms_clear_ptes(vms, mas_detach, true);
mas_set(mas_detach, 0u64);
loop {
let __v0: *mut vm_area_struct = (unsafe { mas_find(mas_detach, 18446744073709551615u64) }) as *mut vm_area_struct;
let vma: *mut vm_area_struct = __v0;
if __v0.is_null() {
break;
}
vma_close(vma);
}
}

// vms_complete_munmap_vmas() - Finish the munmap() operation
// @vms: The vma munmap struct
// @mas_detach: The maple state of the detached vmas
//
// This updates the mm_struct, unmaps the region, frees the resources
// used for the munmap() and may downgrade the lock - if requested.  Everything
// needed to be done once the vma maple tree is updated.
fn vms_complete_munmap_vmas(vms: *mut vma_munmap_struct, mas_detach: *mut ma_state) {
let mut mm: *mut mm_struct = unsafe { core::mem::zeroed() };
mm = unsafe { (*get_current()).mm };
let __v1: i32 = unsafe { (*mm).__slate_anon_0.map_count };
unsafe {
(*mm).__slate_anon_0.map_count = __v1.wrapping_sub((*vms).vma_count);
}
let __v4: u64 = unsafe { (*mm).__slate_anon_0.locked_vm };
unsafe {
(*mm).__slate_anon_0.locked_vm = __v4.wrapping_sub((*vms).locked_vm);
}
if unsafe { (*vms).unlock } {
mmap_write_downgrade(mm);
}
if (unsafe { (*vms).nr_pages }) == 0 {
return;
}
vms_clear_ptes(vms, mas_detach, !unsafe { (*vms).unlock });
// Update high watermark before we lower total_vm
update_hiwater_vm(mm);
// Stat accounting
loop {
while false {
}
loop {
let mut __v7: u64;
while false {
}
__v7 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*mm).__slate_anon_0.total_vm) as *const u64) };
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!((*mm).__slate_anon_0.total_vm) as *mut u64, __v7.wrapping_sub((*vms).nr_pages)) };
if true {
break;
}
}
if true {
break;
}
}
// Paranoid bookkeeping
VM_WARN_ON!();
VM_WARN_ON!();
VM_WARN_ON!();
let __v12: u64 = unsafe { (*mm).__slate_anon_0.exec_vm };
unsafe {
(*mm).__slate_anon_0.exec_vm = __v12.wrapping_sub((*vms).exec_vm);
}
let __v15: u64 = unsafe { (*mm).__slate_anon_0.stack_vm };
unsafe {
(*mm).__slate_anon_0.stack_vm = __v15.wrapping_sub((*vms).stack_vm);
}
let __v18: u64 = unsafe { (*mm).__slate_anon_0.data_vm };
unsafe {
(*mm).__slate_anon_0.data_vm = __v18.wrapping_sub((*vms).data_vm);
}
// Remove and clean up vmas
mas_set(mas_detach, 0u64);
loop {
let __v20: *mut vm_area_struct = (unsafe { mas_find(mas_detach, 18446744073709551615u64) }) as *mut vm_area_struct;
let vma: *mut vm_area_struct = __v20;
if __v20.is_null() {
break;
}
remove_vma(vma);
}
vm_unacct_memory((unsafe { (*vms).nr_accounted }) as i64);
validate_mm!();
if unsafe { (*vms).unlock } {
mmap_read_unlock(mm);
}
unsafe { __mt_destroy((*mas_detach).tree) };
}

/// reattach_vmas() - Undo any munmap work and free resources
/// @mas_detach: The maple state with the detached maple tree
///
/// Reattach any detached vmas and free up the maple tree used to track the vmas.
fn reattach_vmas(mas_detach: *mut ma_state) {
mas_set(mas_detach, 0u64);
loop {
let __v0: *mut vm_area_struct = (unsafe { mas_find(mas_detach, 18446744073709551615u64) }) as *mut vm_area_struct;
let vma: *mut vm_area_struct = __v0;
if __v0.is_null() {
break;
}
vma_mark_attached(vma);
}
unsafe { __mt_destroy((*mas_detach).tree) };
}

/// vms_gather_munmap_vmas() - Put all VMAs within a range into a maple tree
/// for removal at a later date.  Handles splitting first and last if necessary
/// and marking the vmas as isolated.
///
/// @vms: The vma munmap struct
/// @mas_detach: The maple state tracking the detached tree
///
/// Return: 0 on success, error otherwise
fn vms_gather_munmap_vmas(vms: *mut vma_munmap_struct, mas_detach: *mut ma_state) -> i32 {
let mut nrpages: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut i64 = core::ptr::addr_of_mut!(nrpages) as *mut i64;
// Make sure that map_count on return from munmap() will
// not exceed its limit; but let map_count go just above
// its limit temporarily, to help free resources as expected.
let mut __v23: bool = unsafe { core::mem::zeroed() };
let mut error: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut i32 = core::ptr::addr_of_mut!(error) as *mut i32;
let mut next: *mut vm_area_struct = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
{
next = core::ptr::null_mut::<vm_area_struct>();
let __slate_auto_bytes = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// If we need to split any vma, do it now to save pain later.
// Does it split the first one?
if (*vms).start > (*(*vms).vma).__slate_anon_0.__slate_anon_0.vm_start {
if (*vms).end < (*(*vms).vma).__slate_anon_0.__slate_anon_0.vm_end {
__v23 = (*(*(*vms).vma).vm_mm).__slate_anon_0.map_count >= get_sysctl_max_map_count();
} else {
__v23 = false;
}
if __v23 {
*__slate_slot_24 = 12i32.wrapping_neg();
break '__join_0;
} else {
// Don't bother splitting the VMA if we can't unmap it anyway
if vma_is_sealed((*vms).vma) {
*__slate_slot_24 = 1i32.wrapping_neg();
break '__join_0;
} else {
*__slate_slot_24 = __split_vma((*vms).vmi, (*vms).vma, (*vms).start, 1i32);
if *__slate_slot_24 != 0 {
break '__join_0;
}
}
}
}
}
(*vms).prev = vma_prev((*vms).vmi);
if !unsafe { (*vms).prev }.is_null() {
(*vms).unmap_start = (*(*vms).prev).__slate_anon_0.__slate_anon_0.vm_end;
}
// Detach a range of VMAs from the mm. Using next as a temp variable as
// it is always overwritten.
'__join_1: {
'__join_23: {
loop {
let __v0: *mut vm_area_struct = vma_find((*vms).vmi, (*vms).end);
next = __v0;
if !__v0.is_null() {
let __slate_auto_bytes = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if vma_is_sealed(next) {
break '__join_23;
} else {
// Does it split the end?
if (*next).__slate_anon_0.__slate_anon_0.vm_end > (*vms).end {
*__slate_slot_24 = __split_vma((*vms).vmi, next, (*vms).end, 0i32);
if *__slate_slot_24 != 0 {
break '__join_1;
}
}
vma_start_write(next);
let __v20: i32 = (*vms).vma_count;
(*vms).vma_count = __v20.wrapping_add(1);
mas_set(mas_detach, __v20 as u64);
*__slate_slot_24 = mas_store_gfp(mas_detach, next as *mut (), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if *__slate_slot_24 != 0 {
break '__join_1;
} else {
vma_mark_detached(next);
*__slate_slot_22 = vma_pages(next as *const vm_area_struct) as i64;
let __v17: u64 = (*vms).nr_pages;
(*vms).nr_pages = __v17.wrapping_add(*__slate_slot_22 as u64);
if vma_test(next as *const vm_area_struct, 13i32) {
let __v14: u64 = (*vms).locked_vm;
(*vms).locked_vm = __v14.wrapping_add(*__slate_slot_22 as u64);
}
if vma_test(next as *const vm_area_struct, 20i32) {
let __v11: u64 = (*vms).nr_accounted;
(*vms).nr_accounted = __v11.wrapping_add(*__slate_slot_22 as u64);
}
if is_exec_mapping((*next).__slate_anon_3.vm_flags) {
let __v2: u64 = (*vms).exec_vm;
(*vms).exec_vm = __v2.wrapping_add(*__slate_slot_22 as u64);
} else {
if is_stack_mapping((*next).__slate_anon_3.vm_flags) {
let __v5: u64 = (*vms).stack_vm;
(*vms).stack_vm = __v5.wrapping_add(*__slate_slot_22 as u64);
} else {
if is_data_mapping_vma_flags(core::ptr::addr_of_mut!((*next).__slate_anon_3.flags) as *const __SlateRecord_mm_types_h_871) {
let __v8: u64 = (*vms).data_vm;
(*vms).data_vm = __v8.wrapping_add(*__slate_slot_22 as u64);
}
}
}
if !unsafe { (*vms).uf }.is_null() {
// If userfaultfd_unmap_prep returns an error the vmas
// will remain split, but userland will get a
// highly unexpected error anyway. This is no
// different than the case where the first of the two
// __split_vma fails, but we don't undo the first
// split, despite we could. This is unlikely enough
// failure that it's not worth optimizing it for.
*__slate_slot_24 = userfaultfd_unmap_prep(next, (*vms).start, (*vms).end, (*vms).uf);
if *__slate_slot_24 != 0 {
break '__join_1;
}
}
}
}
} else {
break;
}
}
(*vms).next = vma_next((*vms).vmi);
if !unsafe { (*vms).next }.is_null() {
(*vms).unmap_end = (*(*vms).next).__slate_anon_0.__slate_anon_0.vm_start;
}
while vma_iter_addr((*vms).vmi) > (*vms).start {
vma_iter_prev_range((*vms).vmi);
}
(*vms).clear_ptes = true;
return 0;
}
*__slate_slot_24 = 1i32.wrapping_neg();
}
reattach_vmas(mas_detach);
}
return *__slate_slot_24;
}
}

/// init_vma_munmap() - Initializer wrapper for vma_munmap_struct
/// @vms: The vma munmap struct
/// @vmi: The vma iterator
/// @vma: The first vm_area_struct to munmap
/// @start: The aligned start address to munmap
/// @end: The aligned end address to munmap
/// @uf: The userfaultfd list_head
/// @unlock: Unlock after the operation.  Only unlocked on success
fn init_vma_munmap(vms: *mut vma_munmap_struct, vmi: *mut vma_iterator, vma: *mut vm_area_struct, start: u64, end: u64, uf: *mut list_head, unlock: bool) {
unsafe {
(*vms).vmi = vmi;
}
unsafe {
(*vms).vma = vma;
}
if !vma.is_null() {
unsafe {
(*vms).start = start;
}
unsafe {
(*vms).end = end;
}
} else {
unsafe {
(*vms).end = 0;
}
unsafe {
(*vms).start = 0;
}
}
unsafe {
(*vms).unlock = unlock;
}
unsafe {
(*vms).uf = uf;
}
unsafe {
(*vms).vma_count = 0;
}
unsafe {
(*vms).nr_accounted = 0;
}
unsafe {
(*vms).locked_vm = 0;
}
unsafe {
(*vms).nr_pages = 0;
}
unsafe {
(*vms).data_vm = 0;
}
unsafe {
(*vms).stack_vm = 0;
}
unsafe {
(*vms).exec_vm = 0;
}
unsafe {
(*vms).unmap_start = 0;
}
unsafe {
(*vms).unmap_end = 0;
}
unsafe {
(*vms).clear_ptes = false;
}
}

/// do_vmi_align_munmap() - munmap the aligned region from @start to @end.
/// @vmi: The vma iterator
/// @vma: The starting vm_area_struct
/// @mm: The mm_struct
/// @start: The aligned start address to munmap.
/// @end: The aligned end address to munmap.
/// @uf: The userfaultfd list_head
/// @unlock: Set to true to drop the mmap_lock.  unlocking only happens on
/// success.
///
/// Return: 0 on success and drops the lock if so directed, error and leaves the
/// lock held otherwise.
#[unsafe(no_mangle)]
extern "C" fn do_vmi_align_munmap(vmi: *mut vma_iterator, vma: *mut vm_area_struct, mm: *mut mm_struct, start: u64, end: u64, uf: *mut list_head, unlock: bool) -> i32 {
let mut error: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(error) as *mut i32;
let mut vms: core::mem::MaybeUninit<vma_munmap_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut vma_munmap_struct = core::ptr::addr_of_mut!(vms) as *mut vma_munmap_struct;
let mut mas_detach: core::mem::MaybeUninit<ma_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut ma_state = core::ptr::addr_of_mut!(mas_detach) as *mut ma_state;
let mut mt_detach: core::mem::MaybeUninit<maple_tree> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut maple_tree = core::ptr::addr_of_mut!(mt_detach) as *mut maple_tree;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
core::ptr::write(__slate_slot_2, {
let mut __t0: ma_state = core::mem::zeroed();
__t0.tree = core::ptr::addr_of_mut!(*__slate_slot_3);
__t0.index = 0u64;
__t0.last = 0u64;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.min = 0u64;
__t0.max = 18446744073709551615u64;
__t0.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t0.alloc = core::ptr::null_mut::<maple_node>();
__t0.node_request = 0u64;
__t0.status = 1u32;
__t0.mas_flags = 0u8;
__t0.store_type = 0u32;
__t0
});
mt_init_flags(core::ptr::addr_of_mut!(*__slate_slot_3), (*(*vmi).mas.tree).ma_flags & 768);
mt_on_stack!();
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 128usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
init_vma_munmap(core::ptr::addr_of_mut!(*__slate_slot_1), vmi, vma, start, end, uf, unlock);
*__slate_slot_0 = vms_gather_munmap_vmas(core::ptr::addr_of_mut!(*__slate_slot_1), core::ptr::addr_of_mut!(*__slate_slot_2));
if *__slate_slot_0 == 0 {
*__slate_slot_0 = vma_iter_clear_gfp(vmi, start, end, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if *__slate_slot_0 != 0 {
reattach_vmas(core::ptr::addr_of_mut!(*__slate_slot_2));
} else {
// Point of no return
vms_complete_munmap_vmas(core::ptr::addr_of_mut!(*__slate_slot_1), core::ptr::addr_of_mut!(*__slate_slot_2));
return 0;
}
}
validate_mm!();
return *__slate_slot_0;
}
}

/// do_vmi_munmap() - munmap a given range.
/// @vmi: The vma iterator
/// @mm: The mm_struct
/// @start: The start address to munmap
/// @len: The length of the range to munmap
/// @uf: The userfaultfd list_head
/// @unlock: set to true if the user wants to drop the mmap_lock on success
///
/// This function takes a @mas that is either pointing to the previous VMA or set
/// to MA_START and sets it up to remove the mapping(s).  The @len will be
/// aligned.
///
/// Return: 0 on success and drops the lock if so directed, error and leaves the
/// lock held otherwise.
#[unsafe(no_mangle)]
extern "C" fn do_vmi_munmap(vmi: *mut vma_iterator, mm: *mut mm_struct, start: u64, len: u64, uf: *mut list_head, unlock: bool) -> i32 {
let mut end: u64 = 0;
let __v0: bool = if offset_in_page!(start) != 0 {
true
} else {
let __v1: u64 = if test_thread_flag!(27i32) != 0 {
(if (unsafe { (*get_current()).personality }) & 134217728 != 0 { 3221225472u32 } else { 4294959104u32 }) as u64
} else {
task_size_max()
};
start > __v1
};
let __v2: bool = if __v0 {
true
} else {
let __v3: u64 = if test_thread_flag!(27i32) != 0 {
(if (unsafe { (*get_current()).personality }) & 134217728 != 0 { 3221225472u32 } else { 4294959104u32 }) as u64
} else {
task_size_max()
};
len > __v3.wrapping_sub(start)
};
if __v2 {
return 22i32.wrapping_neg();
}
end = start.wrapping_add(PAGE_ALIGN!(len));
if end == start {
return 22i32.wrapping_neg();
}
// Find the first overlapping VMA
let vma: *mut vm_area_struct = vma_find(vmi, end);
if vma.is_null() {
if unlock {
mmap_write_unlock(mm);
}
return 0;
}
do_vmi_align_munmap(vmi, vma, mm, start, end, uf, unlock)
}

/// We are about to modify one or multiple of a VMA's flags, policy, userfaultfd
/// context and anonymous VMA name within the range [start, end).
///
/// As a result, we might be able to merge the newly modified VMA range with an
/// adjacent VMA with identical properties.
///
/// If no merge is possible and the range does not span the entirety of the VMA,
/// we then need to split the VMA to accommodate the change.
///
/// The function returns either the merged VMA, the original VMA if a split was
/// required instead, or an error if the split failed.
fn vma_modify(vmg: *mut vma_merge_struct) -> *mut vm_area_struct {
let vma: *mut vm_area_struct = unsafe { (*vmg).middle };
let start: u64 = unsafe { (*vmg).start };
let end: u64 = unsafe { (*vmg).end };
// First, try to merge.
let merged: *mut vm_area_struct = vma_merge_existing_range(vmg);
if !merged.is_null() {
return merged;
}
if vmg_nomem(vmg) {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut vm_area_struct;
}
// Split can fail for reasons other than OOM, so if the user requests
// this it's probably a mistake.
VM_WARN_ON!();
// Split any preceding portion of the VMA.
if (unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }) < start {
let err: i32 = split_vma(unsafe { (*vmg).vmi }, vma, start, 1i32);
if err != 0 {
return ERR_PTR(err as i64) as *mut vm_area_struct;
}
}
// Split any trailing portion of the VMA.
if (unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end }) > end {
let err: i32 = split_vma(unsafe { (*vmg).vmi }, vma, end, 0i32);
if err != 0 {
return ERR_PTR(err as i64) as *mut vm_area_struct;
}
}
vma
}

#[unsafe(no_mangle)]
extern "C" fn vma_modify_flags(vmi: *mut vma_iterator, prev: *mut vm_area_struct, vma: *mut vm_area_struct, start: u64, end: u64, vma_flags_ptr: *mut __SlateRecord_mm_types_h_871) -> *mut vm_area_struct {
let mut vmg: vma_merge_struct = {
let mut __t0: vma_merge_struct = unsafe { core::mem::zeroed() };
__t0.mm = unsafe { (*vma).vm_mm };
__t0.vmi = vmi;
__t0.prev = prev;
__t0.middle = vma;
__t0.next = core::ptr::null_mut::<vm_area_struct>();
__t0.start = start;
__t0.end = end;
__t0.pgoff = linear_page_index(vma as *const vm_area_struct, start);
__t0.anon_pgoff = __linear_anon_page_index(vma as *const vm_area_struct, start);
__t0.__slate_anon_10 = __SlateRecord_vma_h_109 { vm_flags: unsafe { (*vma).__slate_anon_3.vm_flags } };
__t0.file = unsafe { (*vma).vm_file };
__t0.anon_vma = unsafe { (*vma).anon_vma };
__t0.policy = vma_policy!(vma);
__t0.uffd_ctx = unsafe { (*vma).vm_userfaultfd_ctx };
__t0.anon_name = anon_vma_name(vma);
__t0.state = 0u32;
__t0
};
let vma_flags: __SlateRecord_mm_types_h_871 = unsafe { *vma_flags_ptr };
unsafe {
vmg.__slate_anon_10.vma_flags = vma_flags;
}
let ret: *mut vm_area_struct = vma_modify(core::ptr::addr_of_mut!(vmg));
if IS_ERR(ret as *const ()) {
return ret;
}
// For a merge to succeed, the flags must match those
// requested. However, sticky flags may have been retained, so propagate
// them to the caller.
if (vmg.state as u32) == 3 {
unsafe {
*vma_flags_ptr = (*ret).__slate_anon_3.flags;
}
}
ret
}

#[unsafe(no_mangle)]
extern "C" fn vma_modify_name(vmi: *mut vma_iterator, prev: *mut vm_area_struct, vma: *mut vm_area_struct, start: u64, end: u64, new_name: *mut anon_vma_name) -> *mut vm_area_struct {
let mut vmg: vma_merge_struct = {
let mut __t0: vma_merge_struct = unsafe { core::mem::zeroed() };
__t0.mm = unsafe { (*vma).vm_mm };
__t0.vmi = vmi;
__t0.prev = prev;
__t0.middle = vma;
__t0.next = core::ptr::null_mut::<vm_area_struct>();
__t0.start = start;
__t0.end = end;
__t0.pgoff = linear_page_index(vma as *const vm_area_struct, start);
__t0.anon_pgoff = __linear_anon_page_index(vma as *const vm_area_struct, start);
__t0.__slate_anon_10 = __SlateRecord_vma_h_109 { vm_flags: unsafe { (*vma).__slate_anon_3.vm_flags } };
__t0.file = unsafe { (*vma).vm_file };
__t0.anon_vma = unsafe { (*vma).anon_vma };
__t0.policy = vma_policy!(vma);
__t0.uffd_ctx = unsafe { (*vma).vm_userfaultfd_ctx };
__t0.anon_name = anon_vma_name(vma);
__t0.state = 0u32;
__t0
};
vmg.anon_name = new_name;
vma_modify(core::ptr::addr_of_mut!(vmg))
}

#[unsafe(no_mangle)]
extern "C" fn vma_modify_policy(vmi: *mut vma_iterator, prev: *mut vm_area_struct, vma: *mut vm_area_struct, start: u64, end: u64, new_pol: *mut mempolicy) -> *mut vm_area_struct {
let mut vmg: vma_merge_struct = {
let mut __t0: vma_merge_struct = unsafe { core::mem::zeroed() };
__t0.mm = unsafe { (*vma).vm_mm };
__t0.vmi = vmi;
__t0.prev = prev;
__t0.middle = vma;
__t0.next = core::ptr::null_mut::<vm_area_struct>();
__t0.start = start;
__t0.end = end;
__t0.pgoff = linear_page_index(vma as *const vm_area_struct, start);
__t0.anon_pgoff = __linear_anon_page_index(vma as *const vm_area_struct, start);
__t0.__slate_anon_10 = __SlateRecord_vma_h_109 { vm_flags: unsafe { (*vma).__slate_anon_3.vm_flags } };
__t0.file = unsafe { (*vma).vm_file };
__t0.anon_vma = unsafe { (*vma).anon_vma };
__t0.policy = vma_policy!(vma);
__t0.uffd_ctx = unsafe { (*vma).vm_userfaultfd_ctx };
__t0.anon_name = anon_vma_name(vma);
__t0.state = 0u32;
__t0
};
vmg.policy = new_pol;
vma_modify(core::ptr::addr_of_mut!(vmg))
}

#[unsafe(no_mangle)]
extern "C" fn vma_modify_flags_uffd(vmi: *mut vma_iterator, prev: *mut vm_area_struct, vma: *mut vm_area_struct, start: u64, end: u64, vma_flags: *const __SlateRecord_mm_types_h_871, new_ctx: vm_userfaultfd_ctx, give_up_on_oom: bool) -> *mut vm_area_struct {
let mut vmg: vma_merge_struct = {
let mut __t0: vma_merge_struct = unsafe { core::mem::zeroed() };
__t0.mm = unsafe { (*vma).vm_mm };
__t0.vmi = vmi;
__t0.prev = prev;
__t0.middle = vma;
__t0.next = core::ptr::null_mut::<vm_area_struct>();
__t0.start = start;
__t0.end = end;
__t0.pgoff = linear_page_index(vma as *const vm_area_struct, start);
__t0.anon_pgoff = __linear_anon_page_index(vma as *const vm_area_struct, start);
__t0.__slate_anon_10 = __SlateRecord_vma_h_109 { vm_flags: unsafe { (*vma).__slate_anon_3.vm_flags } };
__t0.file = unsafe { (*vma).vm_file };
__t0.anon_vma = unsafe { (*vma).anon_vma };
__t0.policy = vma_policy!(vma);
__t0.uffd_ctx = unsafe { (*vma).vm_userfaultfd_ctx };
__t0.anon_name = anon_vma_name(vma);
__t0.state = 0u32;
__t0
};
unsafe {
vmg.__slate_anon_10.vma_flags = *vma_flags;
}
vmg.uffd_ctx = new_ctx;
if give_up_on_oom {
let __t1 = true;
vmg.__slate_bits_0.__set_give_up_on_oom(__t1);
}
vma_modify(core::ptr::addr_of_mut!(vmg))
}

/// Expand vma by delta bytes, potentially merging with an immediately adjacent
/// VMA with identical properties.
#[unsafe(no_mangle)]
extern "C" fn vma_merge_extend(vmi: *mut vma_iterator, vma: *mut vm_area_struct, delta: u64) -> *mut vm_area_struct {
let mut vmg: vma_merge_struct = {
let mut __t0: vma_merge_struct = unsafe { core::mem::zeroed() };
__t0.mm = unsafe { (*vma).vm_mm };
__t0.vmi = vmi;
__t0.prev = vma;
__t0.middle = vma;
__t0.next = core::ptr::null_mut::<vm_area_struct>();
__t0.start = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end };
__t0.end = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end }.wrapping_add(delta);
__t0.pgoff = linear_page_index(vma as *const vm_area_struct, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end });
__t0.anon_pgoff = __linear_anon_page_index(vma as *const vm_area_struct, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end });
__t0.__slate_anon_10 = __SlateRecord_vma_h_109 { vm_flags: unsafe { (*vma).__slate_anon_3.vm_flags } };
__t0.file = unsafe { (*vma).vm_file };
__t0.anon_vma = unsafe { (*vma).anon_vma };
__t0.policy = vma_policy!(vma);
__t0.uffd_ctx = unsafe { (*vma).vm_userfaultfd_ctx };
__t0.anon_name = anon_vma_name(vma);
__t0.state = 0u32;
__t0
};
vmg.next = vma_iter_next_rewind(vmi, core::ptr::null_mut::<*mut vm_area_struct>());
vmg.middle = core::ptr::null_mut::<vm_area_struct>(); // We use the VMA to populate VMG fields only.
vma_merge_new_range(core::ptr::addr_of_mut!(vmg))
}

#[unsafe(no_mangle)]
extern "C" fn unlink_file_vma_batch_init(vb: *mut unlink_vma_file_batch) {
unsafe {
(*vb).count = 0;
}
}

fn unlink_file_vma_batch_process(vb: *mut unlink_vma_file_batch) {
let mut i: i32 = 0;
let mapping: *mut address_space = unsafe { (*(*(*((*vb).vmas.as_mut_ptr() as *mut *mut vm_area_struct).wrapping_offset(0 as isize))).vm_file).f_mapping };
i_mmap_lock_write(mapping);
while i < unsafe { (*vb).count } {
VM_WARN_ON_ONCE!();
__remove_shared_vm_struct(unsafe { *((*vb).vmas.as_mut_ptr() as *mut *mut vm_area_struct).wrapping_offset(i as isize) }, mapping);
i = i.wrapping_add(1);
}
i_mmap_unlock_write(mapping);
unlink_file_vma_batch_init(vb);
}

#[unsafe(no_mangle)]
extern "C" fn unlink_file_vma_batch_add(vb: *mut unlink_vma_file_batch, vma: *mut vm_area_struct) {
if unsafe { (*vma).vm_file }.is_null() {
return;
}
if (unsafe { (*vb).count }) > 0 && (unsafe { (*(*((*vb).vmas.as_mut_ptr() as *mut *mut vm_area_struct).wrapping_offset(0 as isize))).vm_file }) != unsafe { (*vma).vm_file } || ((unsafe { (*vb).count }) as u64) == ARRAY_SIZE__21!() {
unlink_file_vma_batch_process(vb);
}
unsafe {
*((*vb).vmas.as_mut_ptr() as *mut *mut vm_area_struct).wrapping_offset((*vb).count as isize) = vma;
}
let __v1: i32 = unsafe { (*vb).count };
unsafe {
(*vb).count = __v1.wrapping_add(1);
}
}

#[unsafe(no_mangle)]
extern "C" fn unlink_file_vma_batch_final(vb: *mut unlink_vma_file_batch) {
if (unsafe { (*vb).count }) > 0 {
unlink_file_vma_batch_process(vb);
}
}

fn vma_link_file(vma: *mut vm_area_struct, hold_rmap_lock: bool) {
let file: *mut file = unsafe { (*vma).vm_file };
if !file.is_null() {
let mapping: *mut address_space = unsafe { (*file).f_mapping };
i_mmap_lock_write(mapping);
__vma_link_file(vma, mapping);
if !hold_rmap_lock {
i_mmap_unlock_write(mapping);
}
}
}

fn vma_link(mm: *mut mm_struct, vma: *mut vm_area_struct) -> i32 {
let mut vmi: vma_iterator = vma_iterator { mas: {
let mut __t0: ma_state = unsafe { core::mem::zeroed() };
__t0.tree = unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_mt) };
__t0.index = 0u64;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.status = 1u32;
__t0
} };
vma_iter_config(core::ptr::addr_of_mut!(vmi), unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end });
if vma_iter_prealloc(core::ptr::addr_of_mut!(vmi), vma) != 0 {
return 12i32.wrapping_neg();
}
vma_start_write(vma);
vma_iter_store_new(core::ptr::addr_of_mut!(vmi), vma);
vma_link_file(vma, false); // hold_rmap_lock=
let __v1: i32 = unsafe { (*mm).__slate_anon_0.map_count };
unsafe {
(*mm).__slate_anon_0.map_count = __v1.wrapping_add(1);
}
validate_mm!();
0
}

/// Copy the vma structure to a new location in the same mm,
/// prior to moving page table entries, to effect an mremap move.
#[unsafe(no_mangle)]
extern "C" fn copy_vma(vmap: *mut *mut vm_area_struct, addr: u64, len: u64, mut pgoff: u64, mut anon_pgoff: u64, need_rmap_locks: *mut bool) -> *mut vm_area_struct {
let mut vmg: core::mem::MaybeUninit<vma_merge_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut vma_merge_struct = core::ptr::addr_of_mut!(vmg) as *mut vma_merge_struct;
let mut vmi: core::mem::MaybeUninit<vma_iterator> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut vma_iterator = core::ptr::addr_of_mut!(vmi) as *mut vma_iterator;
let mut can_self_merge: bool = unsafe { core::mem::zeroed() };
let mut new_vma: core::mem::MaybeUninit<*mut vm_area_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut vm_area_struct = core::ptr::addr_of_mut!(new_vma) as *mut *mut vm_area_struct;
let mut mm: *mut mm_struct = unsafe { core::mem::zeroed() };
let mut old_vma_start: u64 = unsafe { core::mem::zeroed() };
let mut vma: *mut vm_area_struct = unsafe { core::mem::zeroed() };
unsafe {
vma = *vmap;
old_vma_start = (*vma).__slate_anon_0.__slate_anon_0.vm_start;
mm = (*vma).vm_mm;
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
can_self_merge = false;
core::ptr::write(__slate_slot_2, vma_iterator { mas: {
let mut __t1: ma_state = core::mem::zeroed();
__t1.tree = core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_mt);
__t1.index = addr;
__t1.node = core::ptr::null_mut::<maple_enode>();
__t1.status = 1u32;
__t1
} });
core::ptr::write(__slate_slot_1, {
let mut __t2: vma_merge_struct = core::mem::zeroed();
__t2.mm = (*vma).vm_mm;
__t2.vmi = core::ptr::addr_of_mut!(*__slate_slot_2);
__t2.prev = core::ptr::null_mut::<vm_area_struct>();
__t2.middle = vma;
__t2.next = core::ptr::null_mut::<vm_area_struct>();
__t2.start = addr;
__t2.end = addr.wrapping_add(len);
__t2.pgoff = linear_page_index(vma as *const vm_area_struct, addr);
__t2.anon_pgoff = __linear_anon_page_index(vma as *const vm_area_struct, addr);
__t2.__slate_anon_10 = __SlateRecord_vma_h_109 { vm_flags: (*vma).__slate_anon_3.vm_flags };
__t2.file = (*vma).vm_file;
__t2.anon_vma = (*vma).anon_vma;
__t2.policy = vma_policy!(vma);
__t2.uffd_ctx = (*vma).vm_userfaultfd_ctx;
__t2.anon_name = anon_vma_name(vma);
__t2.state = 0u32;
__t2
});
// If a vma has not yet been faulted, update its anonymous pgoff to
// match the new location to increase its chance of merging.
if unsafe { (*vma).anon_vma }.is_null() {
anon_pgoff = addr >> 12i32;
if vma_is_anonymous(vma as *const vm_area_struct) {
pgoff = anon_pgoff;
can_self_merge = true;
}
}
// If the VMA we are copying might contain a uprobe PTE, ensure
// that we do not establish one upon merge. Otherwise, when mremap()
// moves page tables, it will orphan the newly created PTE.
if !unsafe { (*vma).vm_file }.is_null() {
let __t0 = true;
(*__slate_slot_1).__slate_bits_0.__set_skip_vma_uprobe(__t0);
}
*__slate_slot_4 = find_vma_prev(mm, addr, core::ptr::addr_of_mut!((*__slate_slot_1).prev));
if !(*__slate_slot_4).is_null() && (*(*__slate_slot_4)).__slate_anon_0.__slate_anon_0.vm_start < addr.wrapping_add(len) {
return core::ptr::null_mut::<vm_area_struct>();
} else {
'__join_6: {
// should never get here
(*__slate_slot_1).pgoff = pgoff;
(*__slate_slot_1).anon_pgoff = anon_pgoff;
(*__slate_slot_1).next = vma_iter_next_rewind(core::ptr::addr_of_mut!(*__slate_slot_2), core::ptr::null_mut::<*mut vm_area_struct>());
*__slate_slot_4 = vma_merge_copied_range(core::ptr::addr_of_mut!(*__slate_slot_1));
if !(*__slate_slot_4).is_null() {
// Self-merged and VMA replaced.
if unlikely!((*(*__slate_slot_4)).__slate_anon_0.__slate_anon_0.vm_start < old_vma_start && (*(*__slate_slot_4)).__slate_anon_0.__slate_anon_0.vm_end > old_vma_start) != 0 {
// The only way a VMA can both self-merge and be
// replaced is if the remap places the new VMA
// immediately prior to its old self ('next') and
// immediately after another VMA ('prev') causing the
// next to be removed and prev to be expanded to cover
// the entire range.
//
// This should only be possible if the anonymous page
// offset was updated, i.e. the VMA is unfaulted.
VM_WARN_ON_ONCE_VMA!();
let __v0: *mut vm_area_struct = *__slate_slot_4;
vma = __v0;
*vmap = __v0;
}
*need_rmap_locks = vma_start_pgoff(*__slate_slot_4 as *const vm_area_struct) <= vma_start_pgoff(vma as *const vm_area_struct);
} else {
*__slate_slot_4 = vm_area_dup(vma);
if !(*__slate_slot_4).is_null() {
vma_set_range(*__slate_slot_4, addr, addr.wrapping_add(len), pgoff, anon_pgoff);
if vma_dup_policy(vma, *__slate_slot_4) == 0 {
if anon_vma_clone(*__slate_slot_4, vma, 2u32) == 0 {
if !unsafe { (*(*__slate_slot_4)).vm_file }.is_null() {
get_file((*(*__slate_slot_4)).vm_file);
}
if !unsafe { (*(*__slate_slot_4)).vm_ops }.is_null() && (*(*(*__slate_slot_4)).vm_ops).open != None {
unsafe { (*(*(*__slate_slot_4)).vm_ops).open }.unwrap()(*__slate_slot_4);
}
if vma_link(mm, *__slate_slot_4) != 0 {
fixup_hugetlb_reservations(*__slate_slot_4);
vma_close(*__slate_slot_4);
if !unsafe { (*(*__slate_slot_4)).vm_file }.is_null() {
fput((*(*__slate_slot_4)).vm_file);
}
unlink_anon_vmas(*__slate_slot_4);
} else {
*need_rmap_locks = false;
break '__join_6;
}
}
mpol_put(vma_policy!(*__slate_slot_4));
}
vm_area_free(*__slate_slot_4);
}
return core::ptr::null_mut::<vm_area_struct>();
}
}
return *__slate_slot_4;
}
}
}

/// Rough compatibility check to quickly see if it's even worth looking
/// at sharing an anon_vma.
///
/// They need to have the same vm_file, and the flags can only differ
/// in things that mprotect may change.
///
/// NOTE! The fact that we share an anon_vma doesn't _have_ to mean that
/// we can merge the two vma's. For example, we refuse to merge a vma if
/// there is a vm_ops->close() function, because that indicates that the
/// driver is doing some kind of reference counting. But that doesn't
/// really matter for the anon_vma sharing case.
fn anon_vma_compatible(a: *mut vm_area_struct, b: *mut vm_area_struct) -> i32 {
let mut __slate_compound_0: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: [i32; 3] = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_3: [i32; 1] = unsafe { core::mem::zeroed() };
let mut diff: __SlateRecord_mm_types_h_871 = vma_flags_diff_pair((unsafe { core::ptr::addr_of_mut!((*a).__slate_anon_3.flags) }) as *const __SlateRecord_mm_types_h_871, (unsafe { core::ptr::addr_of_mut!((*b).__slate_anon_3.flags) }) as *const __SlateRecord_mm_types_h_871);
// Ignore flags that mprotect() can change.
vma_flags_clear_mask(core::ptr::addr_of_mut!(diff), __mk_vma_flags(unsafe { *{
__slate_compound_0 = {
let mut __t0: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t0
};
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 3u64, unsafe { (*{
__slate_compound_1 = [0i32, 1i32, 2i32];
core::ptr::addr_of_mut!(__slate_compound_1)
}).as_ptr() as *const i32 }));
// Ignore flags that do not impact merging.
vma_flags_clear_mask(core::ptr::addr_of_mut!(diff), __mk_vma_flags(unsafe { *{
__slate_compound_2 = {
let mut __t1: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t1
};
core::ptr::addr_of_mut!(__slate_compound_2)
} }, 1u64, unsafe { (*{
__slate_compound_3 = [11i32];
core::ptr::addr_of_mut!(__slate_compound_3)
}).as_ptr() as *const i32 }));
// Must be adjacent.
if (unsafe { (*a).__slate_anon_0.__slate_anon_0.vm_end }) != unsafe { (*b).__slate_anon_0.__slate_anon_0.vm_start } {
return 0;
}
// Must have matching policy.
if !mpol_equal(vma_policy!(a), vma_policy!(b)) {
return 0;
}
// Must both be anon or map the same file (MAP_PRIVATE case).
if (unsafe { (*a).vm_file }) != unsafe { (*b).vm_file } {
return 0;
}
// Flags must be equivalent modulo mprotect().
if !vma_flags_empty(unsafe { &(*(core::ptr::addr_of_mut!(diff) as *const __SlateRecord_mm_types_h_871)) }) {
return 0;
}
// Page offset must align.
if vma_end_pgoff(a as *const vm_area_struct) != vma_start_pgoff(b as *const vm_area_struct) {
return 0;
}
// Only reached from anon path, so either MAP_PRIVATE file or anon.
if vma_end_anon_pgoff(a as *const vm_area_struct) != vma_start_anon_pgoff(b as *const vm_area_struct) {
return 0;
}
1
}

// Do some basic sanity checking to see if we can re-use the anon_vma
// from 'old'. The 'a'/'b' vma's are in VM order - one of them will be
// the same as 'old', the other will be the new one that is trying
// to share the anon_vma.
//
// NOTE! This runs with mmap_lock held for reading, so it is possible that
// the anon_vma of 'old' is concurrently in the process of being set up
// by another page fault trying to merge _that_. But that's ok: if it
// is being set up, that automatically means that it will be a singleton
// acceptable for merging, so we can do all of this optimistically. But
// we do that READ_ONCE() to make sure that we never re-load the pointer.
//
// The READ_ONCE() establishes an address dependency between anon_vma and
// any access to its fields, which pairs with the assignment to
// vma->anon_vma performed with release semantics in __anon_vma_prepare().
//
// This is especially important as anon_vma's are SLAB_TYPESAFE_BY_RCU so
// accessing an uninitialised anon_vma's fields may result in a UAF.
//
// IOW: that the "list_is_singular()" test on the anon_vma_chain only
// matters for the 'stable anon_vma' case (ie the thing we want to avoid
// is to return an anon_vma that is "complex" due to having gone through
// a fork).
//
// We also make sure that the two vma's are compatible (adjacent,
// and with the same memory policies). That's all stable, even with just
// a read lock on the mmap_lock.
fn reusable_anon_vma(old: *mut vm_area_struct, a: *mut vm_area_struct, b: *mut vm_area_struct) -> *mut anon_vma {
if anon_vma_compatible(a, b) != 0 {
let mut __v0: *mut anon_vma;
loop {
if !(__native_word__34!() || ((const { core::mem::size_of::<*mut anon_vma>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_731() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*old).anon_vma) as *const *mut anon_vma) };
// Paired with a memory barrier in __anon_vma_prepare().
let anon_vma: *mut anon_vma = __v0;
let __v2: bool = !anon_vma.is_null() && list_is_singular((unsafe { core::ptr::addr_of_mut!((*old).anon_vma_chain) }) as *const list_head) != 0;
if __v2 {
return anon_vma;
}
}
core::ptr::null_mut::<anon_vma>()
}

/// find_mergeable_anon_vma is used by anon_vma_prepare, to check
/// neighbouring vmas for a suitable anon_vma, before it goes off
/// to allocate a new anon_vma.  It checks because a repetitive
/// sequence of mprotects and faults may otherwise lead to distinct
/// anon_vmas being allocated, preventing vma merge in subsequent
/// mprotect.
#[unsafe(no_mangle)]
extern "C" fn find_mergeable_anon_vma(vma: *mut vm_area_struct) -> *mut anon_vma {
let mut anon_vma: *mut anon_vma = core::ptr::null_mut::<anon_vma>();
let mut prev: *mut vm_area_struct = unsafe { core::mem::zeroed() };
let mut vmi: vma_iterator = vma_iterator { mas: {
let mut __t0: ma_state = unsafe { core::mem::zeroed() };
__t0.tree = unsafe { core::ptr::addr_of_mut!((*(*vma).vm_mm).__slate_anon_0.mm_mt) };
__t0.index = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end };
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.status = 1u32;
__t0
} };
// Try next first.
let next: *mut vm_area_struct = vma_iter_load(core::ptr::addr_of_mut!(vmi));
if !next.is_null() {
anon_vma = reusable_anon_vma(next, vma, next);
if !anon_vma.is_null() {
return anon_vma;
}
}
prev = vma_prev(core::ptr::addr_of_mut!(vmi));
VM_BUG_ON_VMA!();
prev = vma_prev(core::ptr::addr_of_mut!(vmi));
// Try prev next.
if !prev.is_null() {
anon_vma = reusable_anon_vma(prev, prev, vma);
}
// We might reach here with anon_vma == NULL if we can't find
// any reusable anon_vma.
// There's no absolute need to look only at touching neighbours:
// we could search further afield for "compatible" anon_vmas.
// But it would probably just be a waste of time searching,
// or lead to too many vmas hanging off the same anon_vma.
// We're trying to allow mprotect remerging later on,
// not trying to minimize memory used for anon_vmas.
anon_vma
}

fn vm_ops_needs_writenotify(vm_ops: *const vm_operations_struct) -> bool {
!vm_ops.is_null() && ((unsafe { (*vm_ops).page_mkwrite }) != None || (unsafe { (*vm_ops).pfn_mkwrite }) != None)
}

fn vma_is_shared_writable(vma: *mut vm_area_struct) -> bool {
let mut __slate_compound_0: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: [i32; 2] = unsafe { core::mem::zeroed() };
vma_test_all_mask(vma as *const vm_area_struct, __mk_vma_flags(unsafe { *{
__slate_compound_0 = {
let mut __t0: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t0
};
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 2u64, unsafe { (*{
__slate_compound_1 = [1i32, 3i32];
core::ptr::addr_of_mut!(__slate_compound_1)
}).as_ptr() as *const i32 }))
}

fn vma_fs_can_writeback(vma: *mut vm_area_struct) -> bool {
// No managed pages to writeback.
if vma_test(vma as *const vm_area_struct, 10i32) {
return false;
}
!unsafe { (*vma).vm_file }.is_null() && !unsafe { (*(*vma).vm_file).f_mapping }.is_null() && mapping_can_writeback(unsafe { (*(*vma).vm_file).f_mapping })
}

/// Does this VMA require the underlying folios to have their dirty state
/// tracked?
#[unsafe(no_mangle)]
extern "C" fn vma_needs_dirty_tracking(vma: *mut vm_area_struct) -> bool {
// Only shared, writable VMAs require dirty tracking.
if !vma_is_shared_writable(vma) {
return false;
}
// Does the filesystem need to be notified?
if vm_ops_needs_writenotify(unsafe { (*vma).vm_ops }) {
return true;
}
// Even if the filesystem doesn't indicate a need for writenotify, if it
// can writeback, dirty tracking is still required.
vma_fs_can_writeback(vma)
}

/// Some shared mappings will want the pages marked read-only
/// to track write events. If so, we'll downgrade vm_page_prot
/// to the private version (using protection_map[] without the
/// VM_SHARED bit).
#[unsafe(no_mangle)]
extern "C" fn vma_wants_writenotify(vma: *mut vm_area_struct, vm_page_prot: pgprot) -> bool {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
// If it was private or non-writable, the write bit is already clear
if !vma_is_shared_writable(vma) {
return false;
}
// The backer wishes to know when pages are first written to?
if vm_ops_needs_writenotify(unsafe { (*vma).vm_ops }) {
return true;
}
// The open routine did something to the protections that pgprot_modify
// won't preserve?
if pgprot_val!(vm_page_prot) != unsafe { (*{
__slate_compound_0 = vma_pgprot_modify(vm_page_prot, (*vma).__slate_anon_3.flags);
core::ptr::addr_of_mut!(__slate_compound_0)
}).pgprot } {
return false;
}
// Do we need to track softdirty? hugetlb does not support softdirty
// tracking yet.
let __v1: bool = vma_soft_dirty_enabled(vma) && !is_vm_hugetlb_page(vma as *const vm_area_struct);
if __v1 {
return true;
}
// Do we need write faults for uffd-wp tracking?
if userfaultfd_wp(vma) {
return true;
}
// Can the mapping track the dirty pages?
vma_fs_can_writeback(vma)
}

static mut mm_all_locks_mutex: mutex = {
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

fn vm_lock_anon_vma(mm: *mut mm_struct, anon_vma: *mut anon_vma) {
if test_bit__4!(0i32, (unsafe { core::ptr::addr_of_mut!((*(*anon_vma).root).rb_root.rb_root.rb_node) }) as *mut u64) == 0 {
// The LSB of head.next can't change from under us
// because we hold the mm_all_locks_mutex.
down_write_nest_lock!(unsafe { core::ptr::addr_of_mut!((*(*anon_vma).root).rwsem) });
// We can safely modify head.next after taking the
// anon_vma->root->rwsem. If some other vma in this mm shares
// the same anon_vma we won't take it again.
//
// No need of atomic instructions here, head.next
// can't change from under us thanks to the
// anon_vma->root->rwsem.
if __test_and_set_bit!(0i32, (unsafe { core::ptr::addr_of_mut!((*(*anon_vma).root).rb_root.rb_root.rb_node) }) as *mut u64) != 0 {
loop {
unsafe {
core::arch::asm!("732: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 732b - ., 3; .popsection\n# {0}", const 732 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_vm_lock_anon_vma_2, sym __slate_asm_string_vm_lock_anon_vma_3, const 2268 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
}

fn vm_lock_mapping(mm: *mut mm_struct, mapping: *mut address_space) {
if test_bit__4!(2i32, unsafe { core::ptr::addr_of_mut!((*mapping).flags) }) == 0 {
// AS_MM_ALL_LOCKS can't change from under us because
// we hold the mm_all_locks_mutex.
//
// Operations on ->flags have to be atomic because
// even if AS_MM_ALL_LOCKS is stable thanks to the
// mm_all_locks_mutex, there may be other cpus
// changing other bitflags in parallel to us.
if test_and_set_bit(2i64, (unsafe { core::ptr::addr_of_mut!((*mapping).flags) }) as *mut u64) {
loop {
unsafe {
core::arch::asm!("733: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 733b - ., 3; .popsection\n# {0}", const 733 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_vm_lock_mapping_1, sym __slate_asm_string_vm_lock_mapping_2, const 2285 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
down_write_nest_lock!(unsafe { core::ptr::addr_of_mut!((*mapping).i_mmap_rwsem) });
}
}

/// This operation locks against the VM for all pte/vma/mm related
/// operations that could ever happen on a certain mm. This includes
/// vmtruncate, try_to_unmap, and all page faults.
///
/// The caller must take the mmap_lock in write mode before calling
/// mm_take_all_locks(). The caller isn't allowed to release the
/// mmap_lock until mm_drop_all_locks() returns.
///
/// mmap_lock in write mode is required in order to block all operations
/// that could modify pagetables and free pages without need of
/// altering the vma layout. It's also needed in write mode to avoid new
/// anon_vmas to be associated with existing vmas.
///
/// A single task can't take more than one mm_take_all_locks() in a row
/// or it would deadlock.
///
/// The LSB in anon_vma->rb_root.rb_node and the AS_MM_ALL_LOCKS bitflag in
/// mapping->flags avoid to take the same lock twice, if more than one
/// vma in this mm is backed by the same anon_vma or address_space.
///
/// We take locks in following order, accordingly to comment at beginning
/// of mm/rmap.c:
///   - all hugetlbfs_i_mmap_rwsem_key locks (aka mapping->i_mmap_rwsem for
///     hugetlb mapping);
///   - all vmas marked locked
///   - all i_mmap_rwsem locks;
///   - all anon_vma->rwseml
///
/// We can take all locks within these types randomly because the VM code
/// doesn't nest them and we protected from parallel mm_take_all_locks() by
/// mm_all_locks_mutex.
///
/// mm_take_all_locks() and mm_drop_all_locks are expensive operations
/// that may have to take thousand of locks.
///
/// mm_take_all_locks() can fail if it's interrupted by signals.
#[unsafe(no_mangle)]
extern "C" fn mm_take_all_locks(mm: *mut mm_struct) -> i32 {
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut vmi: core::mem::MaybeUninit<vma_iterator> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut vma_iterator = core::ptr::addr_of_mut!(vmi) as *mut vma_iterator;
let mut avc: core::mem::MaybeUninit<*mut anon_vma_chain> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut anon_vma_chain = core::ptr::addr_of_mut!(avc) as *mut *mut anon_vma_chain;
let mut vma: core::mem::MaybeUninit<*mut vm_area_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut vm_area_struct = core::ptr::addr_of_mut!(vma) as *mut *mut vm_area_struct;
unsafe {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_6, vma_iterator { mas: {
let mut __t0: ma_state = core::mem::zeroed();
__t0.tree = core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_mt);
__t0.index = 0u64;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.status = 1u32;
__t0
} });
mmap_assert_write_locked(mm as *const mm_struct);
mutex_lock(core::ptr::addr_of_mut!(mm_all_locks_mutex));
// vma_start_write() does not have a complement in mm_drop_all_locks()
// because vma_start_write() is always asymmetrical; it marks a VMA as
// being written to until mmap_write_unlock() or mmap_write_downgrade()
// is reached.
'__join_0: {
loop {
let __v5: *mut vm_area_struct = vma_next(core::ptr::addr_of_mut!(*__slate_slot_6));
*__slate_slot_8 = __v5;
if !__v5.is_null() {
if signal_pending(get_current()) != 0 {
break '__join_0;
} else {
vma_start_write(*__slate_slot_8);
}
} else {
break;
}
}
vma_iter_init(&mut (*core::ptr::addr_of_mut!(*__slate_slot_6)), mm, 0u64);
loop {
let __v3: *mut vm_area_struct = vma_next(core::ptr::addr_of_mut!(*__slate_slot_6));
*__slate_slot_8 = __v3;
if !__v3.is_null() {
if signal_pending(get_current()) != 0 {
break '__join_0;
} else {
if !unsafe { (*(*__slate_slot_8)).vm_file }.is_null() && !unsafe { (*(*(*__slate_slot_8)).vm_file).f_mapping }.is_null() {
__v4 = is_vm_hugetlb_page(*__slate_slot_8 as *const vm_area_struct);
} else {
__v4 = false;
}
if __v4 {
vm_lock_mapping(mm, (*(*(*__slate_slot_8)).vm_file).f_mapping);
}
}
} else {
break;
}
}
vma_iter_init(&mut (*core::ptr::addr_of_mut!(*__slate_slot_6)), mm, 0u64);
loop {
let __v1: *mut vm_area_struct = vma_next(core::ptr::addr_of_mut!(*__slate_slot_6));
*__slate_slot_8 = __v1;
if !__v1.is_null() {
if signal_pending(get_current()) != 0 {
break '__join_0;
} else {
if !unsafe { (*(*__slate_slot_8)).vm_file }.is_null() && !unsafe { (*(*(*__slate_slot_8)).vm_file).f_mapping }.is_null() {
__v2 = !is_vm_hugetlb_page(*__slate_slot_8 as *const vm_area_struct);
} else {
__v2 = false;
}
if __v2 {
vm_lock_mapping(mm, (*(*(*__slate_slot_8)).vm_file).f_mapping);
}
}
} else {
break;
}
}
vma_iter_init(&mut (*core::ptr::addr_of_mut!(*__slate_slot_6)), mm, 0u64);
'__loop_2: loop {
let __v0: *mut vm_area_struct = vma_next(core::ptr::addr_of_mut!(*__slate_slot_6));
*__slate_slot_8 = __v0;
if !__v0.is_null() {
if signal_pending(get_current()) != 0 {
break '__join_0;
} else {
if !unsafe { (*(*__slate_slot_8)).anon_vma }.is_null() {
*__slate_slot_7 = list_first_entry__3!(core::ptr::addr_of_mut!((*(*__slate_slot_8)).anon_vma_chain));
loop {
if list_entry_is_head!(*__slate_slot_7, core::ptr::addr_of_mut!((*(*__slate_slot_8)).anon_vma_chain), same_vma) == 0 {
vm_lock_anon_vma(mm, (*(*__slate_slot_7)).anon_vma);
*__slate_slot_7 = list_next_entry__3!(*__slate_slot_7, same_vma);
} else {
continue '__loop_2;
}
}
}
}
} else {
break;
}
}
return 0;
}
mm_drop_all_locks(mm);
return 4i32.wrapping_neg();
}
}

fn vm_unlock_anon_vma(anon_vma: *mut anon_vma) {
if test_bit__4!(0i32, (unsafe { core::ptr::addr_of_mut!((*(*anon_vma).root).rb_root.rb_root.rb_node) }) as *mut u64) != 0 {
// The LSB of head.next can't change to 0 from under
// us because we hold the mm_all_locks_mutex.
//
// We must however clear the bitflag before unlocking
// the vma so the users using the anon_vma->rb_root will
// never see our bitflag.
//
// No need of atomic instructions here, head.next
// can't change from under us until we release the
// anon_vma->root->rwsem.
if __test_and_clear_bit!(0i32, (unsafe { core::ptr::addr_of_mut!((*(*anon_vma).root).rb_root.rb_root.rb_node) }) as *mut u64) == 0 {
loop {
unsafe {
core::arch::asm!("734: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 734b - ., 3; .popsection\n# {0}", const 734 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_vm_unlock_anon_vma_2, sym __slate_asm_string_vm_unlock_anon_vma_3, const 2401 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
anon_vma_unlock_write(anon_vma);
}
}

fn vm_unlock_mapping(mapping: *mut address_space) {
if test_bit__4!(2i32, unsafe { core::ptr::addr_of_mut!((*mapping).flags) }) != 0 {
// AS_MM_ALL_LOCKS can't change to 0 from under us
// because we hold the mm_all_locks_mutex.
i_mmap_unlock_write(mapping);
if !test_and_clear_bit(2i64, (unsafe { core::ptr::addr_of_mut!((*mapping).flags) }) as *mut u64) {
loop {
unsafe {
core::arch::asm!("735: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 735b - ., 3; .popsection\n# {0}", const 735 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_vm_unlock_mapping_1, sym __slate_asm_string_vm_unlock_mapping_2, const 2416 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
}

/// The mmap_lock cannot be released by the caller until
/// mm_drop_all_locks() returns.
#[unsafe(no_mangle)]
extern "C" fn mm_drop_all_locks(mm: *mut mm_struct) {
let mut vma: *mut vm_area_struct = unsafe { core::mem::zeroed() };
let mut avc: *mut anon_vma_chain = unsafe { core::mem::zeroed() };
let mut vmi: vma_iterator = vma_iterator { mas: {
let mut __t0: ma_state = unsafe { core::mem::zeroed() };
__t0.tree = unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_mt) };
__t0.index = 0u64;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.status = 1u32;
__t0
} };
mmap_assert_write_locked(mm as *const mm_struct);
loop {
if unlikely!(!unsafe { mutex_is_locked(core::ptr::addr_of_mut!(mm_all_locks_mutex)) }) != 0 {
loop {
unsafe {
core::arch::asm!("736: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 736b - ., 3; .popsection\n# {0}", const 736 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_mm_drop_all_locks_0, sym __slate_asm_string_mm_drop_all_locks_1, const 2431 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if true {
break;
}
}
loop {
let __v8: *mut vm_area_struct = vma_next(core::ptr::addr_of_mut!(vmi));
vma = __v8;
if __v8.is_null() {
break;
}
if !unsafe { (*vma).anon_vma }.is_null() {
avc = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*vma).anon_vma_chain) });
while list_entry_is_head!(avc, unsafe { core::ptr::addr_of_mut!((*vma).anon_vma_chain) }, same_vma) == 0 {
vm_unlock_anon_vma(unsafe { (*avc).anon_vma });
avc = list_next_entry__3!(avc, same_vma);
}
}
if !unsafe { (*vma).vm_file }.is_null() && !unsafe { (*(*vma).vm_file).f_mapping }.is_null() {
vm_unlock_mapping(unsafe { (*(*vma).vm_file).f_mapping });
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(mm_all_locks_mutex)) };
}

/// We account for memory if it's a private writeable mapping,
/// not hugepages and VM_NORESERVE wasn't set.
fn accountable_mapping(map: &mut mmap_state) -> bool {
let mut __slate_compound_2: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_3: [i32; 2] = unsafe { core::mem::zeroed() };
let file: *const file = (unsafe { (*(map as *mut mmap_state)).file }) as *const file;
// hugetlb has its own accounting separate from the core VM
// VM_HUGETLB may not be set yet so we cannot check for that flag.
let __v0: bool = !file.is_null() && is_file_hugepages(file);
if __v0 {
return false;
}
vma_flags_test((unsafe { core::ptr::addr_of_mut!((*(map as *mut mmap_state)).__slate_anon_7.vma_flags) }) as *const __SlateRecord_mm_types_h_871, 1i32) && !vma_flags_test_any_mask((unsafe { core::ptr::addr_of_mut!((*(map as *mut mmap_state)).__slate_anon_7.vma_flags) }) as *const __SlateRecord_mm_types_h_871, __mk_vma_flags(unsafe { *{
__slate_compound_2 = {
let mut __t0: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t0
};
core::ptr::addr_of_mut!(__slate_compound_2)
} }, 2u64, unsafe { (*{
__slate_compound_3 = [21i32, 3i32];
core::ptr::addr_of_mut!(__slate_compound_3)
}).as_ptr() as *const i32 }))
}

/// vms_abort_munmap_vmas() - Undo as much as possible from an aborted munmap()
/// operation.
/// @vms: The vma unmap structure
/// @mas_detach: The maple state with the detached maple tree
///
/// Reattach any detached vmas, free up the maple tree used to track the vmas.
/// If that's not possible because the ptes are cleared (and vm_ops->closed() may
/// have been called), then a NULL is written over the vmas and the vmas are
/// removed (munmap() completed).
fn vms_abort_munmap_vmas(vms: *mut vma_munmap_struct, mas_detach: *mut ma_state) {
let mas: *mut ma_state = unsafe { core::ptr::addr_of_mut!((*(*vms).vmi).mas) };
if (unsafe { (*vms).nr_pages }) == 0 {
return;
}
if unsafe { (*vms).clear_ptes } {
reattach_vmas(mas_detach);
return;
}
// Aborting cannot just call the vm_ops open() because they are often
// not symmetrical and state data has been lost.  Resort to the old
// failure method of leaving a gap where the MAP_FIXED mapping failed.
mas_set_range(mas, unsafe { (*vms).start }, unsafe { (*vms).end }.wrapping_sub(1));
unsafe { mas_store_gfp(mas, core::ptr::null_mut::<()>(), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(15i32) as u32)) };
// Clean up the insertion of the unfortunate gap
vms_complete_munmap_vmas(vms, mas_detach);
}

fn update_ksm_flags(map: &mut mmap_state) {
unsafe {
(*(map as *mut mmap_state)).__slate_anon_7.vma_flags = ksm_vma_flags((*(map as *mut mmap_state)).mm, (*(map as *mut mmap_state)).file as *const file, (*(map as *mut mmap_state)).__slate_anon_7.vma_flags);
}
}

fn set_desc_from_map(desc: &mut vm_area_desc, map: *const mmap_state) {
unsafe {
(*(desc as *mut vm_area_desc)).start = (*map).addr;
}
unsafe {
(*(desc as *mut vm_area_desc)).end = (*map).end;
}
unsafe {
(*(desc as *mut vm_area_desc)).pgoff = (*map).pgoff;
}
unsafe {
(*(desc as *mut vm_area_desc)).vm_file = (*map).file;
}
unsafe {
(*(desc as *mut vm_area_desc)).vma_flags = (*map).__slate_anon_7.vma_flags;
}
unsafe {
(*(desc as *mut vm_area_desc)).page_prot = (*map).page_prot;
}
}

/// __mmap_setup() - Prepare to gather any overlapping VMAs that need to be
/// unmapped once the map operation is completed, check limits, account mapping
/// and clean up any pre-existing VMAs.
///
/// As a result it sets up the @map and @desc objects.
///
/// @map: Mapping state.
/// @desc: VMA descriptor
/// @uf:  Userfaultfd context list.
///
/// Returns: 0 on success, error code otherwise.
fn __mmap_setup(map: &mut mmap_state, desc: &mut vm_area_desc, uf: *mut list_head) -> i32 {
let mut __slate_compound_4: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: [i32; 1] = unsafe { core::mem::zeroed() };
let mut error: i32 = 0;
let vmi: *mut vma_iterator = unsafe { (*(map as *mut mmap_state)).vmi };
let vms: *mut vma_munmap_struct = unsafe { core::ptr::addr_of_mut!((*(map as *mut mmap_state)).vms) };
// Find the first overlapping VMA and initialise unmap state.
unsafe {
(*vms).vma = vma_find(vmi, (*(map as *mut mmap_state)).end);
}
init_vma_munmap(vms, vmi, unsafe { (*vms).vma }, unsafe { (*(map as *mut mmap_state)).addr }, unsafe { (*(map as *mut mmap_state)).end }, uf, false); // unlock =
// OK, we have overlapping VMAs - prepare to unmap them.
if !unsafe { (*vms).vma }.is_null() {
mt_init_flags(unsafe { core::ptr::addr_of_mut!((*(map as *mut mmap_state)).mt_detach) }, (unsafe { (*(*vmi).mas.tree).ma_flags }) & 768);
mt_on_stack!();
mas_init(unsafe { core::ptr::addr_of_mut!((*(map as *mut mmap_state)).mas_detach) }, unsafe { core::ptr::addr_of_mut!((*(map as *mut mmap_state)).mt_detach) }, 0u64); // addr =
// Prepare to unmap any existing mapping in the area
error = vms_gather_munmap_vmas(vms, unsafe { core::ptr::addr_of_mut!((*(map as *mut mmap_state)).mas_detach) });
if error != 0 {
// On error VMAs will already have been reattached.
unsafe {
(*vms).nr_pages = 0;
}
return error;
}
unsafe {
(*(map as *mut mmap_state)).next = (*vms).next;
}
unsafe {
(*(map as *mut mmap_state)).prev = (*vms).prev;
}
} else {
unsafe {
(*(map as *mut mmap_state)).next = vma_iter_next_rewind(vmi, core::ptr::addr_of_mut!((*(map as *mut mmap_state)).prev));
}
}
// Check against address space limit.
if !unsafe { may_expand_vm((*(map as *mut mmap_state)).mm, core::ptr::addr_of_mut!((*(map as *mut mmap_state)).__slate_anon_7.vma_flags) as *const __SlateRecord_mm_types_h_871, unsafe { (*(map as *mut mmap_state)).pglen }.wrapping_sub((*vms).nr_pages)) } {
return 12i32.wrapping_neg();
}
// Private writable mapping: check memory availability.
if accountable_mapping(map) {
unsafe {
(*(map as *mut mmap_state)).charged = (*(map as *mut mmap_state)).pglen;
}
let __v2: u64 = unsafe { (*map).charged };
unsafe {
(*map).charged = __v2.wrapping_sub((*vms).nr_accounted);
}
if (unsafe { (*(map as *mut mmap_state)).charged }) != 0 {
error = unsafe { security_vm_enough_memory_mm((*(map as *mut mmap_state)).mm, (*(map as *mut mmap_state)).charged as i64) };
if error != 0 {
return error;
}
}
unsafe {
(*vms).nr_accounted = 0;
}
vma_flags_set_mask(unsafe { core::ptr::addr_of_mut!((*(map as *mut mmap_state)).__slate_anon_7.vma_flags) }, __mk_vma_flags(unsafe { *{
__slate_compound_4 = {
let mut __t0: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t0
};
core::ptr::addr_of_mut!(__slate_compound_4)
} }, 1u64, unsafe { (*{
__slate_compound_5 = [20i32];
core::ptr::addr_of_mut!(__slate_compound_5)
}).as_ptr() as *const i32 }));
}
// Clear PTEs while the vma is still in the tree so that rmap
// cannot race with the freeing later in the truncate scenario.
// This is also needed for mmap_file(), which is why vm_ops
// close function is called.
vms_clean_up_area(vms, unsafe { core::ptr::addr_of_mut!((*(map as *mut mmap_state)).mas_detach) });
set_desc_from_map(desc, (map as *mut mmap_state) as *const mmap_state);
0
}

fn __mmap_new_file_vma(map: &mut mmap_state, vma: *mut vm_area_struct) -> i32 {
let vmi: *mut vma_iterator = unsafe { (*(map as *mut mmap_state)).vmi };
let mut error: i32 = 0;
unsafe {
(*vma).vm_file = (*(map as *mut mmap_state)).file;
}
if ((unsafe { (*(map as *mut mmap_state)).__slate_bits_0.__get_file_doesnt_need_get() }) as i32) == 0 {
get_file(unsafe { (*(map as *mut mmap_state)).file });
}
if (unsafe { (*(*(*(map as *mut mmap_state)).file).f_op).mmap }) == None {
return 0;
}
error = mmap_file(unsafe { (*vma).vm_file }, vma);
if error != 0 {
let mut unmap: unmap_desc = {
let mut __t0: unmap_desc = unsafe { core::mem::zeroed() };
__t0.mas = unsafe { core::ptr::addr_of_mut!((*vmi).mas) };
__t0.first = vma;
__t0.pg_start = if !unsafe { (*(map as *mut mmap_state)).prev }.is_null() { unsafe { (*(*(map as *mut mmap_state)).prev).__slate_anon_0.__slate_anon_0.vm_end } } else { 0u64 };
__t0.pg_end = if !unsafe { (*(map as *mut mmap_state)).next }.is_null() { unsafe { (*(*(map as *mut mmap_state)).next).__slate_anon_0.__slate_anon_0.vm_start } } else { 0u64 };
__t0.vma_start = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start };
__t0.vma_end = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end };
__t0.tree_end = if !unsafe { (*(map as *mut mmap_state)).next }.is_null() { unsafe { (*(*(map as *mut mmap_state)).next).__slate_anon_0.__slate_anon_0.vm_start } } else { 0u64 };
__t0.tree_reset = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end };
__t0.mm_wr_locked = true;
__t0
};
unsafe { fput((*vma).vm_file) };
unsafe {
(*vma).vm_file = core::ptr::null_mut::<file>();
}
vma_iter_set(vmi, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end });
// Undo any partial mapping done by a device driver.
unmap_region(core::ptr::addr_of_mut!(unmap));
return error;
}
// Drivers cannot alter the address of the VMA.
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*(map as *mut mmap_state)).addr }) != unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("737: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 737b - ., 3; .popsection\n# {0}", const 737 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___mmap_new_file_vma_1, sym __slate_asm_string___mmap_new_file_vma_2, const 2617 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("738: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 738b - ., 4; .popsection\n# {0}", const 738 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
// Drivers should not permit writability when previously it was
// disallowed.
VM_WARN_ON_ONCE!();
unsafe {
(*(map as *mut mmap_state)).file = (*vma).vm_file;
}
unsafe {
(*(map as *mut mmap_state)).__slate_anon_7.vma_flags = (*vma).__slate_anon_3.flags;
}
0
}

/// __mmap_new_vma() - Allocate a new VMA for the region, as merging was not
/// possible.
///
/// @map:  Mapping state.
/// @vmap: Output pointer for the new VMA.
/// @action: Any mmap_prepare action that is still to complete.
///
/// Returns: Zero on success, or an error.
fn __mmap_new_vma(map: &mut mmap_state, vmap: &mut *mut vm_area_struct, action: &mut mmap_action) -> i32 {
let mut vma: core::mem::MaybeUninit<*mut vm_area_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut vm_area_struct = core::ptr::addr_of_mut!(vma) as *mut *mut vm_area_struct;
let mut error: i32 = unsafe { core::mem::zeroed() };
let mut vmi: *mut vma_iterator = unsafe { core::mem::zeroed() };
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut is_anon: bool = unsafe { core::mem::zeroed() };
unsafe {
if unsafe { (*(map as *mut mmap_state)).file }.is_null() {
__v6 = !vma_flags_test(core::ptr::addr_of_mut!((*(map as *mut mmap_state)).__slate_anon_7.vma_flags) as *const __SlateRecord_mm_types_h_871, 3i32);
} else {
__v6 = false;
}
is_anon = __v6;
vmi = (*(map as *mut mmap_state)).vmi;
error = 0;
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// Determine the object being mapped and call the appropriate
// specific mapper. the address has already been validated, but
// not unmapped, but the maps are removed from the list.
*__slate_slot_3 = vm_area_alloc((*(map as *mut mmap_state)).mm);
if (*__slate_slot_3).is_null() {
return 12i32.wrapping_neg();
} else {
vma_iter_config(vmi, (*(map as *mut mmap_state)).addr, (*(map as *mut mmap_state)).end);
if is_anon {
vma_set_anonymous(*__slate_slot_3);
}
vma_set_range(*__slate_slot_3, (*(map as *mut mmap_state)).addr, (*(map as *mut mmap_state)).end, (*(map as *mut mmap_state)).pgoff, (*(map as *mut mmap_state)).anon_pgoff);
(*(*__slate_slot_3)).__slate_anon_3.flags = (*(map as *mut mmap_state)).__slate_anon_7.vma_flags;
(*(*__slate_slot_3)).vm_page_prot = (*(map as *mut mmap_state)).page_prot;
if vma_iter_prealloc(vmi, *__slate_slot_3) != 0 {
error = 12i32.wrapping_neg();
} else {
// Invoke callbacks.
if !unsafe { (*(map as *mut mmap_state)).file }.is_null() {
error = __mmap_new_file_vma(map, *__slate_slot_3);
} else {
if !is_anon {
error = shmem_zero_setup(*__slate_slot_3);
}
}
if error != 0 {
vma_iter_free(vmi);
} else {
if ((*(map as *mut mmap_state)).__slate_bits_0.__get_check_ksm_early() as i32) == 0 {
update_ksm_flags(map);
(*(*__slate_slot_3)).__slate_anon_3.flags = (*(map as *mut mmap_state)).__slate_anon_7.vma_flags;
}
// Lock the VMA since it is modified after insertion into VMA tree
vma_start_write(*__slate_slot_3);
vma_iter_store_new(vmi, *__slate_slot_3);
let __v2: *mut mm_struct = (*(map as *mut mmap_state)).mm;
let __v1: i32 = (*__v2).__slate_anon_0.map_count;
(*__v2).__slate_anon_0.map_count = __v1.wrapping_add(1);
vma_link_file(*__slate_slot_3, ((*(action as *mut mmap_action)).__slate_bits_0.__get_hide_from_rmap_until_complete() as i32) != 0);
// vma_merge_new_range() calls khugepaged_enter_vma() too, the below
// call covers the non-merge case.
if !vma_is_anonymous(*__slate_slot_3 as *const vm_area_struct) {
khugepaged_enter_vma(*__slate_slot_3, (*(map as *mut mmap_state)).__slate_anon_7.vm_flags);
}
*(vmap as *mut *mut vm_area_struct) = *__slate_slot_3;
return 0;
}
}
vm_area_free(*__slate_slot_3);
return error;
}
}
}

/// __mmap_complete() - Unmap any VMAs we overlap, account memory mapping
///                     statistics, handle locking and finalise the VMA.
///
/// @map: Mapping state.
/// @vma: Merged or newly allocated VMA for the mmap()'d region.
fn __mmap_complete(map: &mut mmap_state, vma: *mut vm_area_struct) {
let mut __slate_compound_0: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: [i32; 2] = unsafe { core::mem::zeroed() };
let mm: *mut mm_struct = unsafe { (*(map as *mut mmap_state)).mm };
unsafe { perf_event_mmap(vma) };
// Unmap any existing mapping in the area.
vms_complete_munmap_vmas(unsafe { core::ptr::addr_of_mut!((*(map as *mut mmap_state)).vms) }, unsafe { core::ptr::addr_of_mut!((*(map as *mut mmap_state)).mas_detach) });
unsafe { vm_stat_account(mm, (*vma).__slate_anon_3.vm_flags, (*(map as *mut mmap_state)).pglen as i64) };
if vma_test(vma as *const vm_area_struct, 13i32) {
if !vma_supports_mlock(vma as *const vm_area_struct) {
vma_clear_flags_mask(vma, __mk_vma_flags(unsafe { *{
__slate_compound_0 = {
let mut __t0: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t0
};
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 2u64, unsafe { (*{
__slate_compound_1 = [13i32, 19i32];
core::ptr::addr_of_mut!(__slate_compound_1)
}).as_ptr() as *const i32 }));
} else {
let __v3: u64 = unsafe { (*mm).__slate_anon_0.locked_vm };
unsafe {
(*mm).__slate_anon_0.locked_vm = __v3.wrapping_add((*(map as *mut mmap_state)).pglen);
}
}
}
if !unsafe { (*vma).vm_file }.is_null() {
unsafe { uprobe_mmap(vma) };
}
// New (or expanded) vma always get soft dirty status.
// Otherwise user-space soft-dirty page tracker won't
// be able to distinguish situation when vma area unmapped,
// then new mapped in-place (which must be aimed as
// a completely new data area).
unsafe { vma_set_page_prot(vma) };
}

fn call_action_prepare(map: &mut mmap_state, desc: *mut vm_area_desc) -> i32 {
let err: i32 = unsafe { mmap_action_prepare(desc) };
if err != 0 {
return err;
}
0
}

/// Invoke the f_op->mmap_prepare() callback for a file-backed mapping that
/// specifies it.
///
/// This is called prior to any merge attempt, and updates whitelisted fields
/// that are permitted to be updated by the caller.
///
/// All but user-defined fields will be pre-populated with original values.
///
/// Returns 0 on success, or an error code otherwise.
fn call_mmap_prepare(map: &mut mmap_state, desc: *mut vm_area_desc) -> i32 {
let mut err: i32 = 0;
// Invoke the hook.
err = vfs_mmap_prepare(unsafe { (*(map as *mut mmap_state)).file }, desc);
if err != 0 {
return err;
}
err = call_action_prepare(map, desc);
if err != 0 {
return err;
}
// Update fields permitted to be changed.
unsafe {
(*(map as *mut mmap_state)).pgoff = (*desc).pgoff;
}
if (unsafe { (*desc).vm_file }) != unsafe { (*(map as *mut mmap_state)).file } {
unsafe {
let __t0 = true;
(*(map as *mut mmap_state)).__slate_bits_0.__set_file_doesnt_need_get(__t0);
}
unsafe {
(*(map as *mut mmap_state)).file = (*desc).vm_file;
}
}
unsafe {
(*(map as *mut mmap_state)).__slate_anon_7.vma_flags = (*desc).vma_flags;
}
unsafe {
(*(map as *mut mmap_state)).page_prot = (*desc).page_prot;
}
// User-defined fields.
unsafe {
(*(map as *mut mmap_state)).vm_ops = (*desc).vm_ops;
}
unsafe {
(*(map as *mut mmap_state)).vm_private_data = (*desc).private_data;
}
0
}

fn set_vma_user_defined_fields(vma: *mut vm_area_struct, map: &mmap_state) {
if !unsafe { (*(map as *const mmap_state)).vm_ops }.is_null() {
unsafe {
(*vma).vm_ops = (*(map as *const mmap_state)).vm_ops;
}
} else {
vma_set_anonymous(vma);
}
// Only /dev/zero should do this.
unsafe {
(*vma).vm_private_data = (*(map as *const mmap_state)).vm_private_data;
}
}

/// Are we guaranteed no driver can change state such as to preclude KSM merging?
/// If so, let's set the KSM mergeable flag early so we don't break VMA merging.
fn can_set_ksm_flags_early(map: &mmap_state) -> bool {
let file: *mut file = unsafe { (*(map as *const mmap_state)).file };
// Anonymous mappings have no driver which can change them.
if file.is_null() {
return true;
}
// If .mmap_prepare() is specified, then the driver will have already
// manipulated state prior to updating KSM flags. So no need to worry
// about mmap callbacks modifying VMA flags after the KSM flag has been
// updated here, which could otherwise affect KSM eligibility.
if (unsafe { (*(*file).f_op).mmap_prepare }) != None {
return true;
}
// shmem is safe.
if shmem_file(file) {
return true;
}
// Any other .mmap callback is not safe.
false
}

fn __mmap_region(file: *mut file, addr: u64, len: u64, vma_flags: __SlateRecord_mm_types_h_871, pgoff: u64, uf: *mut list_head) -> u64 {
// vma =
let mut vmg: core::mem::MaybeUninit<vma_merge_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut vma_merge_struct = core::ptr::addr_of_mut!(vmg) as *mut vma_merge_struct;
let mut error: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(error) as *mut i32;
let mut allocated_new: bool = unsafe { core::mem::zeroed() };
// Default to no further action.
let mut desc: core::mem::MaybeUninit<vm_area_desc> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut vm_area_desc = core::ptr::addr_of_mut!(desc) as *mut vm_area_desc;
let mut map: core::mem::MaybeUninit<mmap_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut mmap_state = core::ptr::addr_of_mut!(map) as *mut mmap_state;
let mut vmi: core::mem::MaybeUninit<vma_iterator> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut vma_iterator = core::ptr::addr_of_mut!(vmi) as *mut vma_iterator;
let mut have_mmap_prepare: bool = unsafe { core::mem::zeroed() };
let mut vma: *mut vm_area_struct = unsafe { core::mem::zeroed() };
let mut mm: *mut mm_struct = unsafe { core::mem::zeroed() };
unsafe {
mm = (*get_current()).mm;
vma = core::ptr::null_mut::<vm_area_struct>();
have_mmap_prepare = !file.is_null() && (*(*file).f_op).mmap_prepare != None;
core::ptr::write(__slate_slot_6, vma_iterator { mas: {
let mut __t2: ma_state = core::mem::zeroed();
__t2.tree = core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_mt);
__t2.index = addr;
__t2.node = core::ptr::null_mut::<maple_enode>();
__t2.status = 1u32;
__t2
} });
let anon_pgoff: u64 = addr >> 12i32;
core::ptr::write(__slate_slot_4, {
let mut __t3: mmap_state = core::mem::zeroed();
__t3.mm = mm;
__t3.vmi = core::ptr::addr_of_mut!(*__slate_slot_6);
__t3.addr = addr;
__t3.end = addr.wrapping_add(len);
__t3.pgoff = pgoff;
__t3.anon_pgoff = anon_pgoff;
__t3.pglen = PHYS_PFN!(len);
__t3.__slate_anon_7 = {
let mut __t4: __SlateRecord_vma_c_23 = core::mem::zeroed();
__t4.vma_flags = vma_flags;
__t4
};
__t3.file = file;
__t3.page_prot = vma_flags_to_page_prot(vma_flags);
__t3
});
core::ptr::write(__slate_slot_3, {
let mut __t5: vm_area_desc = core::mem::zeroed();
__t5.mm = mm;
__t5.file = file;
__t5.vm_ops = core::ptr::addr_of!(vma_dummy_vm_ops);
__t5.action = {
let mut __t6: mmap_action = core::mem::zeroed();
__t6.r#type = 0u32;
__t6
};
__t5
});
allocated_new = false;
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __t7 = can_set_ksm_flags_early(&(*core::ptr::addr_of_mut!(*__slate_slot_4)));
(*__slate_slot_4).__slate_bits_0.__set_check_ksm_early(__t7);
*__slate_slot_1 = __mmap_setup(&mut (*core::ptr::addr_of_mut!(*__slate_slot_4)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_3)), uf);
if *__slate_slot_1 == 0 {
if have_mmap_prepare {
*__slate_slot_1 = call_mmap_prepare(&mut (*core::ptr::addr_of_mut!(*__slate_slot_4)), core::ptr::addr_of_mut!(*__slate_slot_3));
}
'__join_4: {
if *__slate_slot_1 == 0 {
if ((*__slate_slot_4).__slate_bits_0.__get_check_ksm_early() as i32) != 0 {
update_ksm_flags(&mut (*core::ptr::addr_of_mut!(*__slate_slot_4)));
}
// Attempt to merge with adjacent VMAs...
if !(*__slate_slot_4).prev.is_null() || !(*__slate_slot_4).next.is_null() {
core::ptr::write(__slate_slot_0, {
let mut __t0: vma_merge_struct = core::mem::zeroed();
__t0.mm = (*core::ptr::addr_of_mut!(*__slate_slot_4)).mm;
__t0.vmi = (*core::ptr::addr_of_mut!(*__slate_slot_4)).vmi;
__t0.prev = (*core::ptr::addr_of_mut!(*__slate_slot_4)).prev;
__t0.middle = core::ptr::null_mut::<vm_area_struct>();
__t0.next = (*core::ptr::addr_of_mut!(*__slate_slot_4)).next;
__t0.start = (*core::ptr::addr_of_mut!(*__slate_slot_4)).addr;
__t0.end = (*core::ptr::addr_of_mut!(*__slate_slot_4)).end;
__t0.pgoff = (*core::ptr::addr_of_mut!(*__slate_slot_4)).pgoff;
__t0.anon_pgoff = (*core::ptr::addr_of_mut!(*__slate_slot_4)).anon_pgoff;
__t0.__slate_anon_10 = {
let mut __t1: __SlateRecord_vma_h_109 = core::mem::zeroed();
__t1.vma_flags = (*core::ptr::addr_of_mut!(*__slate_slot_4)).__slate_anon_7.vma_flags;
__t1
};
__t0.file = (*core::ptr::addr_of_mut!(*__slate_slot_4)).file;
__t0.state = 0u32;
__t0
});
vma = vma_merge_new_range(core::ptr::addr_of_mut!(*__slate_slot_0));
}
// ...but if we can't, allocate a new VMA.
if vma.is_null() {
*__slate_slot_1 = __mmap_new_vma(&mut (*core::ptr::addr_of_mut!(*__slate_slot_4)), &mut (*core::ptr::addr_of_mut!(vma)), &mut (*core::ptr::addr_of_mut!((*__slate_slot_3).action)));
if *__slate_slot_1 != 0 {
break '__join_4;
} else {
allocated_new = true;
}
}
if have_mmap_prepare && allocated_new {
set_vma_user_defined_fields(vma, &(*core::ptr::addr_of_mut!(*__slate_slot_4)));
}
__mmap_complete(&mut (*core::ptr::addr_of_mut!(*__slate_slot_4)), vma);
if have_mmap_prepare && allocated_new {
*__slate_slot_1 = mmap_action_complete(vma, core::ptr::addr_of_mut!((*__slate_slot_3).action), false); // is_compat=
if *__slate_slot_1 != 0 {
return *__slate_slot_1 as u64;
}
}
return addr;
}
}
if (*__slate_slot_4).charged != 0 {
vm_unacct_memory((*__slate_slot_4).charged as i64);
}
}
if ((*__slate_slot_4).__slate_bits_0.__get_file_doesnt_need_get() as i32) != 0 {
fput((*__slate_slot_4).file);
}
// This indicates that .mmap_prepare has set a new file, differing from
// desc->vm_file. But since we're aborting the operation, only the
// original file will be cleaned up. Ensure we clean up both.
vms_abort_munmap_vmas(core::ptr::addr_of_mut!((*__slate_slot_4).vms), core::ptr::addr_of_mut!((*__slate_slot_4).mas_detach));
return *__slate_slot_1 as u64;
}
// Accounting was done by __mmap_setup().
}

/// mmap_region() - Actually perform the userland mapping of a VMA into
/// current->mm with known, aligned and overflow-checked @addr and @len, and
/// correctly determined VMA flags @vm_flags and page offset @pgoff.
///
/// This is an internal memory management function, and should not be used
/// directly.
///
/// The caller must write-lock current->mm->mmap_lock.
///
/// @file: If a file-backed mapping, a pointer to the struct file describing the
/// file to be mapped, otherwise NULL.
/// @addr: The page-aligned address at which to perform the mapping.
/// @len: The page-aligned, non-zero, length of the mapping.
/// @vma_flags: The VMA flags which should be applied to the mapping.
/// @pgoff: If @file is specified, the page offset into the file, if not then
/// the virtual page offset in memory of the anonymous mapping.
/// @uf: Optionally, a pointer to a list head used for tracking userfaultfd unmap
/// events.
///
/// Returns: Either an error, or the address at which the requested mapping has
/// been performed.
#[unsafe(no_mangle)]
extern "C" fn mmap_region(file: *mut file, addr: u64, len: u64, mut vma_flags: __SlateRecord_mm_types_h_871, pgoff: u64, uf: *mut list_head) -> u64 {
let mut ret: u64 = 0;
let mut writable_file_mapping: bool = false;
mmap_assert_write_locked((unsafe { (*get_current()).mm }) as *const mm_struct);
// Check to see if MDWE is applicable.
if map_deny_write_exec(unsafe { &(*(core::ptr::addr_of_mut!(vma_flags) as *const __SlateRecord_mm_types_h_871)) }, unsafe { &(*(core::ptr::addr_of_mut!(vma_flags) as *const __SlateRecord_mm_types_h_871)) }) {
return 13i32.wrapping_neg() as u64;
}
// Allow architectures to sanity-check the vm_flags.
if !arch_validate_flags(vma_flags_to_legacy(vma_flags)) {
return 22i32.wrapping_neg() as u64;
}
// Map writable and ensure this isn't a sealed memfd.
let __v0: bool = !file.is_null() && is_shared_maywrite(core::ptr::addr_of_mut!(vma_flags) as *const __SlateRecord_mm_types_h_871);
if __v0 {
let error: i32 = mapping_map_writable(unsafe { (*file).f_mapping });
if error != 0 {
return error as u64;
}
writable_file_mapping = true;
}
ret = __mmap_region(file, addr, len, vma_flags, pgoff, uf);
// Clear our write mapping regardless of error.
if writable_file_mapping {
mapping_unmap_writable(unsafe { (*file).f_mapping });
}
validate_mm!();
ret
}

/// do_brk_flags() - Increase the brk vma if the flags match.
/// @vmi: The vma iterator
/// @addr: The start address
/// @len: The length of the increase
/// @vma: The vma,
/// @vma_flags: The VMA Flags
///
/// Extend the brk VMA from addr to addr + len.  If the VMA is NULL or the flags
/// do not match then create a new anonymous VMA.  Eventually we may be able to
/// do some brk-specific accounting here.
///
/// Returns: %0 on success, or otherwise an error.
#[unsafe(no_mangle)]
extern "C" fn do_brk_flags(vmi: *mut vma_iterator, mut vma: *mut vm_area_struct, addr: u64, len: u64, mut vma_flags: __SlateRecord_mm_types_h_871) -> i32 {
let mut __slate_compound_16: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_17: [i32; 6] = unsafe { core::mem::zeroed() };
let mut __slate_compound_18: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_19: [i32; 1] = unsafe { core::mem::zeroed() };
let mut vmg: core::mem::MaybeUninit<vma_merge_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut vma_merge_struct = core::ptr::addr_of_mut!(vmg) as *mut vma_merge_struct;
let mut pgoff: u64 = unsafe { core::mem::zeroed() };
let mut mm: *mut mm_struct = unsafe { core::mem::zeroed() };
unsafe {
mm = (*get_current()).mm;
pgoff = addr >> 12i32;
// Check against address space limits by the changed size
// Note: This happens *after* clearing old mappings in some code paths.
vma_flags_set_mask(core::ptr::addr_of_mut!(vma_flags), __mk_vma_flags(*{
__slate_compound_16 = {
let mut __t3: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t3
};
core::ptr::addr_of_mut!(__slate_compound_16)
}, 6u64, (*{
__slate_compound_17 = [0i32, 1i32, if (*get_current()).personality & 4194304 != 0 { 2i32 } else { 0i32 }, 4i32, 5i32, 6i32];
core::ptr::addr_of_mut!(__slate_compound_17)
}).as_ptr() as *const i32));
vma_flags_set_mask(core::ptr::addr_of_mut!(vma_flags), __mk_vma_flags(*{
__slate_compound_18 = {
let mut __t4: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t4
};
core::ptr::addr_of_mut!(__slate_compound_18)
}, 1u64, (*{
__slate_compound_19 = [20i32];
core::ptr::addr_of_mut!(__slate_compound_19)
}).as_ptr() as *const i32));
vma_flags_set_mask(core::ptr::addr_of_mut!(vma_flags), (*mm).__slate_anon_0.__slate_anon_28.def_vma_flags);
vma_flags = ksm_vma_flags(mm, core::ptr::null::<file>(), vma_flags);
if !may_expand_vm(mm, core::ptr::addr_of_mut!(vma_flags) as *const __SlateRecord_mm_types_h_871, len >> 12i32) {
return 12i32.wrapping_neg();
} else {
if (*mm).__slate_anon_0.map_count > get_sysctl_max_map_count() {
return 12i32.wrapping_neg();
} else {
if security_vm_enough_memory_mm(mm, (len >> 12i32) as i64) != 0 {
return 12i32.wrapping_neg();
} else {
'__join_4: {
'__join_0: {
// Expand the existing vma if possible; Note that singular lists do not
// occur after forking, so the expand will only happen on new VMAs.
if !vma.is_null() && (*vma).__slate_anon_0.__slate_anon_0.vm_end == addr {
core::ptr::write(__slate_slot_12, {
let mut __t0: vma_merge_struct = core::mem::zeroed();
__t0.mm = mm;
__t0.vmi = vmi;
__t0.start = addr;
__t0.end = addr.wrapping_add(len);
__t0.pgoff = pgoff;
__t0.anon_pgoff = pgoff;
__t0.__slate_anon_10 = {
let mut __t1: __SlateRecord_vma_h_109 = core::mem::zeroed();
__t1.vma_flags = vma_flags;
__t1
};
__t0.state = 0u32;
__t0
});
(*__slate_slot_12).prev = vma;
// vmi is positioned at prev, which this mode expects.
let __t2 = true;
(*__slate_slot_12).__slate_bits_0.__set_just_expand(__t2);
if !vma_merge_new_range(core::ptr::addr_of_mut!(*__slate_slot_12)).is_null() {
break '__join_4;
} else {
if vmg_nomem(core::ptr::addr_of_mut!(*__slate_slot_12)) {
break '__join_0;
}
}
}
if !vma.is_null() {
vma_iter_next_range(vmi);
}
// create a vma struct for an anonymous mapping
vma = vm_area_alloc(mm);
if !vma.is_null() {
vma_set_anonymous(vma);
vma_set_range(vma, addr, addr.wrapping_add(len), pgoff, pgoff);
(*vma).__slate_anon_3.flags = vma_flags;
(*vma).vm_page_prot = vm_get_page_prot(vma_flags_to_legacy(vma_flags));
vma_start_write(vma);
if vma_iter_store_gfp(vmi, vma, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) != 0 {
vm_area_free(vma);
} else {
let __v10: i32 = (*mm).__slate_anon_0.map_count;
(*mm).__slate_anon_0.map_count = __v10.wrapping_add(1);
validate_mm!();
break '__join_4;
}
}
}
vm_unacct_memory((len >> 12i32) as i64);
return 12i32.wrapping_neg();
}
perf_event_mmap(vma);
let __v7: u64 = (*mm).__slate_anon_0.total_vm;
(*mm).__slate_anon_0.total_vm = __v7.wrapping_add(len >> 12i32);
let __v4: u64 = (*mm).__slate_anon_0.data_vm;
(*mm).__slate_anon_0.data_vm = __v4.wrapping_add(len >> 12i32);
if vma_flags_test(core::ptr::addr_of_mut!(vma_flags) as *const __SlateRecord_mm_types_h_871, 13i32) {
let __v1: u64 = (*mm).__slate_anon_0.locked_vm;
(*mm).__slate_anon_0.locked_vm = __v1.wrapping_add(len >> 12i32);
}
return 0;
}
}
}
}
}

/// unmapped_area() - Find an area between the low_limit and the high_limit with
/// the correct alignment and offset, all from @info. Note: current->mm is used
/// for the search.
///
/// @info: The unmapped area information including the range [low_limit -
/// high_limit), the alignment offset and mask.
///
/// Return: A memory address or -ENOMEM.
#[unsafe(no_mangle)]
extern "C" fn unmapped_area(info: *mut vm_unmapped_area_info) -> u64 {
let mut __slate_compound_10: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: [i32; 1] = unsafe { core::mem::zeroed() };
let mut __v0: bool = unsafe { core::mem::zeroed() };
// Avoid prev check if possible
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut vmi: core::mem::MaybeUninit<vma_iterator> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut vma_iterator = core::ptr::addr_of_mut!(vmi) as *mut vma_iterator;
let mut tmp: core::mem::MaybeUninit<*mut vm_area_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut vm_area_struct = core::ptr::addr_of_mut!(tmp) as *mut *mut vm_area_struct;
let mut high_limit: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u64 = core::ptr::addr_of_mut!(high_limit) as *mut u64;
let mut low_limit: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(low_limit) as *mut u64;
let mut gap: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u64 = core::ptr::addr_of_mut!(gap) as *mut u64;
let mut length: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u64 = core::ptr::addr_of_mut!(length) as *mut u64;
unsafe {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_4, vma_iterator { mas: {
let mut __t1: ma_state = core::mem::zeroed();
__t1.tree = core::ptr::addr_of_mut!((*(*get_current()).mm).__slate_anon_0.mm_mt);
__t1.index = 0u64;
__t1.node = core::ptr::null_mut::<maple_enode>();
__t1.status = 1u32;
__t1
} });
// Adjust search length to account for worst case alignment overhead
*__slate_slot_9 = unsafe { (*info).length }.wrapping_add((*info).align_mask).wrapping_add((*info).start_gap);
if *__slate_slot_9 < (*info).length {
return 12i32.wrapping_neg() as u64;
} else {
*__slate_slot_7 = (*info).low_limit;
if *__slate_slot_7 < mmap_min_addr {
*__slate_slot_7 = mmap_min_addr;
}
*__slate_slot_6 = (*info).high_limit;
loop {
if vma_iter_area_lowest(&mut (*core::ptr::addr_of_mut!(*__slate_slot_4)), *__slate_slot_7, *__slate_slot_6, *__slate_slot_9) != 0 {
return 12i32.wrapping_neg() as u64;
} else {
// Adjust for the gap first so it doesn't interfere with the later
// alignment. The first step is the minimum needed to fulfill the start
// gap, the next step is the minimum to align that. It is the minimum
// needed to fulfill both.
*__slate_slot_8 = vma_iter_addr(core::ptr::addr_of_mut!(*__slate_slot_4)).wrapping_add((*info).start_gap);
let __v3: u64 = *__slate_slot_8;
*__slate_slot_8 = __v3.wrapping_add(unsafe { (*info).align_offset }.wrapping_sub(*__slate_slot_8) & (*info).align_mask);
*__slate_slot_5 = vma_next(core::ptr::addr_of_mut!(*__slate_slot_4));
if !(*__slate_slot_5).is_null() {
__v1 = vma_test_any_mask(*__slate_slot_5 as *const vm_area_struct, __mk_vma_flags(*{
__slate_compound_10 = {
let mut __t0: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t0
};
core::ptr::addr_of_mut!(__slate_compound_10)
}, 1u64, (*{
__slate_compound_11 = [8i32];
core::ptr::addr_of_mut!(__slate_compound_11)
}).as_ptr() as *const i32));
} else {
__v1 = false;
}
if __v1 {
if vm_start_gap(*__slate_slot_5 as *const vm_area_struct) < (*__slate_slot_8).wrapping_add(*__slate_slot_9).wrapping_sub(1) {
*__slate_slot_7 = (*(*__slate_slot_5)).__slate_anon_0.__slate_anon_0.vm_end;
vma_iter_reset(&mut (*core::ptr::addr_of_mut!(*__slate_slot_4)));
} else {
break;
}
} else {
*__slate_slot_5 = vma_prev(core::ptr::addr_of_mut!(*__slate_slot_4));
if !(*__slate_slot_5).is_null() {
__v0 = vm_end_gap(*__slate_slot_5 as *const vm_area_struct) > *__slate_slot_8;
} else {
__v0 = false;
}
if __v0 {
*__slate_slot_7 = vm_end_gap(*__slate_slot_5 as *const vm_area_struct);
vma_iter_reset(&mut (*core::ptr::addr_of_mut!(*__slate_slot_4)));
} else {
break;
}
}
}
}
return *__slate_slot_8;
}
}
}

/// unmapped_area_topdown() - Find an area between the low_limit and the
/// high_limit with the correct alignment and offset at the highest available
/// address, all from @info. Note: current->mm is used for the search.
///
/// @info: The unmapped area information including the range [low_limit -
/// high_limit), the alignment offset and mask.
///
/// Return: A memory address or -ENOMEM.
#[unsafe(no_mangle)]
extern "C" fn unmapped_area_topdown(info: *mut vm_unmapped_area_info) -> u64 {
let mut __slate_compound_11: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: [i32; 1] = unsafe { core::mem::zeroed() };
let mut __v0: bool = unsafe { core::mem::zeroed() };
// Avoid prev check if possible
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut vmi: core::mem::MaybeUninit<vma_iterator> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut vma_iterator = core::ptr::addr_of_mut!(vmi) as *mut vma_iterator;
let mut tmp: core::mem::MaybeUninit<*mut vm_area_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut vm_area_struct = core::ptr::addr_of_mut!(tmp) as *mut *mut vm_area_struct;
let mut high_limit: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u64 = core::ptr::addr_of_mut!(high_limit) as *mut u64;
let mut low_limit: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(low_limit) as *mut u64;
let mut gap_end: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u64 = core::ptr::addr_of_mut!(gap_end) as *mut u64;
let mut gap: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u64 = core::ptr::addr_of_mut!(gap) as *mut u64;
let mut length: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(length) as *mut u64;
unsafe {
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_4, vma_iterator { mas: {
let mut __t1: ma_state = core::mem::zeroed();
__t1.tree = core::ptr::addr_of_mut!((*(*get_current()).mm).__slate_anon_0.mm_mt);
__t1.index = 0u64;
__t1.node = core::ptr::null_mut::<maple_enode>();
__t1.status = 1u32;
__t1
} });
// Adjust search length to account for worst case alignment overhead
*__slate_slot_10 = unsafe { (*info).length }.wrapping_add((*info).align_mask).wrapping_add((*info).start_gap);
if *__slate_slot_10 < (*info).length {
return 12i32.wrapping_neg() as u64;
} else {
*__slate_slot_7 = (*info).low_limit;
if *__slate_slot_7 < mmap_min_addr {
*__slate_slot_7 = mmap_min_addr;
}
*__slate_slot_6 = (*info).high_limit;
loop {
if vma_iter_area_highest(&mut (*core::ptr::addr_of_mut!(*__slate_slot_4)), *__slate_slot_7, *__slate_slot_6, *__slate_slot_10) != 0 {
return 12i32.wrapping_neg() as u64;
} else {
*__slate_slot_9 = vma_iter_end(&(*core::ptr::addr_of_mut!(*__slate_slot_4))).wrapping_sub((*info).length);
let __v3: u64 = *__slate_slot_9;
*__slate_slot_9 = __v3.wrapping_sub((*__slate_slot_9).wrapping_sub((*info).align_offset) & (*info).align_mask);
*__slate_slot_8 = vma_iter_end(&(*core::ptr::addr_of_mut!(*__slate_slot_4)));
*__slate_slot_5 = vma_next(core::ptr::addr_of_mut!(*__slate_slot_4));
if !(*__slate_slot_5).is_null() {
__v1 = vma_test_any_mask(*__slate_slot_5 as *const vm_area_struct, __mk_vma_flags(*{
__slate_compound_11 = {
let mut __t0: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t0
};
core::ptr::addr_of_mut!(__slate_compound_11)
}, 1u64, (*{
__slate_compound_12 = [8i32];
core::ptr::addr_of_mut!(__slate_compound_12)
}).as_ptr() as *const i32));
} else {
__v1 = false;
}
if __v1 {
if vm_start_gap(*__slate_slot_5 as *const vm_area_struct) < *__slate_slot_8 {
*__slate_slot_6 = vm_start_gap(*__slate_slot_5 as *const vm_area_struct);
vma_iter_reset(&mut (*core::ptr::addr_of_mut!(*__slate_slot_4)));
} else {
break;
}
} else {
*__slate_slot_5 = vma_prev(core::ptr::addr_of_mut!(*__slate_slot_4));
if !(*__slate_slot_5).is_null() {
__v0 = vm_end_gap(*__slate_slot_5 as *const vm_area_struct) > *__slate_slot_9;
} else {
__v0 = false;
}
if __v0 {
*__slate_slot_6 = (*(*__slate_slot_5)).__slate_anon_0.__slate_anon_0.vm_start;
vma_iter_reset(&mut (*core::ptr::addr_of_mut!(*__slate_slot_4)));
} else {
break;
}
}
}
}
return *__slate_slot_9;
}
}
}

/// Verify that the stack growth is acceptable and
/// update accounting. This is shared with both the
/// grow-up and grow-down cases.
fn acct_stack_growth(vma: *mut vm_area_struct, size: u64, grow: u64) -> i32 {
let mm: *mut mm_struct = unsafe { (*vma).vm_mm };
let mut new_start: u64 = 0;
// address space limit tests
if !unsafe { may_expand_vm(mm, core::ptr::addr_of_mut!((*vma).__slate_anon_3.flags) as *const __SlateRecord_mm_types_h_871, grow) } {
return 12i32.wrapping_neg();
}
// Stack limit test
if size > rlimit(3u32) {
return 12i32.wrapping_neg();
}
// mlock limit tests
if !unsafe { mlock_future_ok(mm as *const mm_struct, vma_test(vma as *const vm_area_struct, 13i32), grow << 12i32) } {
return 12i32.wrapping_neg();
}
// Check to ensure the stack will not grow into a hugetlb-only region
new_start = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end }.wrapping_sub(size);
if is_hugepage_only_range(unsafe { (*vma).vm_mm }, new_start, size) != 0 {
return 14i32.wrapping_neg();
}
// Overcommit..  This must be the final test, as it will
// update security statistics.
if (unsafe { security_vm_enough_memory_mm(mm, grow as i64) }) != 0 {
return 12i32.wrapping_neg();
}
0
}

/// vma is the first one with address < vma->vm_start.  Have to extend vma.
/// mmap_lock held for writing.
#[unsafe(no_mangle)]
extern "C" fn expand_downwards(vma: *mut vm_area_struct, mut address: u64) -> i32 {
let mm: *mut mm_struct = unsafe { (*vma).vm_mm };
let mut error: i32 = 0;
let mut vmi: vma_iterator = vma_iterator { mas: {
let mut __t0: ma_state = unsafe { core::mem::zeroed() };
__t0.tree = unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_mt) };
__t0.index = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start };
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.status = 1u32;
__t0
} };
if !vma_test(vma as *const vm_area_struct, 8i32) {
return 14i32.wrapping_neg();
}
mmap_assert_write_locked(mm as *const mm_struct);
let __v0: u64 = address;
address = __v0 & !(_AC!() << 12i32).wrapping_sub(1);
if address < unsafe { mmap_min_addr } || address < 0 {
return 1i32.wrapping_neg();
}
// Enforce stack_guard_gap
let prev: *mut vm_area_struct = vma_prev(core::ptr::addr_of_mut!(vmi));
// Check that both stack segments have the same anon_vma?
if !prev.is_null() {
let __v2: bool = !vma_test(prev as *const vm_area_struct, 8i32) && vma_is_accessible(prev as *const vm_area_struct);
if __v2 && address.wrapping_sub(unsafe { (*prev).__slate_anon_0.__slate_anon_0.vm_end }) < unsafe { stack_guard_gap } {
return 12i32.wrapping_neg();
}
}
if !prev.is_null() {
vma_iter_next_range_limit(unsafe { &mut (*core::ptr::addr_of_mut!(vmi)) }, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start });
}
vma_iter_config(core::ptr::addr_of_mut!(vmi), address, unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end });
if vma_iter_prealloc(core::ptr::addr_of_mut!(vmi), vma) != 0 {
return 12i32.wrapping_neg();
}
// We must make sure the anon_vma is allocated.
if unlikely__2!(anon_vma_prepare(vma)) != 0 {
vma_iter_free(core::ptr::addr_of_mut!(vmi));
return 12i32.wrapping_neg();
}
// Lock the VMA before expanding to prevent concurrent page faults
vma_start_write(vma);
// We update the anon VMA tree.
anon_vma_lock_write(unsafe { (*vma).anon_vma });
// Somebody else might have raced and expanded it already
if address < unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start } {
let size: u64 = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end }.wrapping_sub(address);
let grow: u64 = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }.wrapping_sub(address) >> 12i32;
error = 12i32.wrapping_neg();
if grow <= vma_start_pgoff(vma as *const vm_area_struct) {
error = acct_stack_growth(vma, size, grow);
if error == 0 {
if vma_test(vma as *const vm_area_struct, 13i32) {
let __v4: u64 = unsafe { (*mm).__slate_anon_0.locked_vm };
unsafe {
(*mm).__slate_anon_0.locked_vm = __v4.wrapping_add(grow);
}
}
unsafe { vm_stat_account(mm, (*vma).__slate_anon_3.vm_flags, grow as i64) };
anon_rmap_tree_pre_update_vma(vma);
unsafe {
(*vma).__slate_anon_0.__slate_anon_0.vm_start = address;
}
vma_sub_pgoff(vma, grow);
// Overwrite old entry in mtree.
vma_iter_store_overwrite(core::ptr::addr_of_mut!(vmi), vma);
anon_rmap_tree_post_update_vma(vma);
unsafe { perf_event_mmap(vma) };
}
}
}
anon_vma_unlock_write(unsafe { (*vma).anon_vma });
vma_iter_free(core::ptr::addr_of_mut!(vmi));
validate_mm!();
error
}

#[unsafe(no_mangle)]
extern "C" fn __vm_munmap(start: u64, len: u64, unlock: bool) -> i32 {
let mut ret: i32 = 0;
let mm: *mut mm_struct = unsafe { (*get_current()).mm };
let mut uf: list_head = unsafe { core::mem::zeroed() };
uf = list_head { next: core::ptr::addr_of_mut!(uf), prev: core::ptr::addr_of_mut!(uf) };
let mut vmi: vma_iterator = vma_iterator { mas: {
let mut __t0: ma_state = unsafe { core::mem::zeroed() };
__t0.tree = unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_mt) };
__t0.index = start;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.status = 1u32;
__t0
} };
if mmap_write_lock_killable(mm) != 0 {
return 4i32.wrapping_neg();
}
ret = do_vmi_munmap(core::ptr::addr_of_mut!(vmi), mm, start, len, core::ptr::addr_of_mut!(uf), unlock);
if ret != 0 || !unlock {
mmap_write_unlock(mm);
}
userfaultfd_unmap_complete(mm, unsafe { &(*core::ptr::addr_of_mut!(uf)) });
ret
}

/// Insert vm structure into process list sorted by address
/// and into the inode's i_mmap tree if file-backed.
#[unsafe(no_mangle)]
extern "C" fn insert_vm_struct(mm: *mut mm_struct, vma: *mut vm_area_struct) -> i32 {
let charged: u64 = vma_pages(vma as *const vm_area_struct);
if !unsafe { find_vma_intersection(mm, (*vma).__slate_anon_0.__slate_anon_0.vm_start, (*vma).__slate_anon_0.__slate_anon_0.vm_end) }.is_null() {
return 12i32.wrapping_neg();
}
let __v0: bool = vma_test(vma as *const vm_area_struct, 20i32) && (unsafe { security_vm_enough_memory_mm(mm, charged as i64) }) != 0;
if __v0 {
return 12i32.wrapping_neg();
}
// The vm_pgoff of a purely anonymous vma should be irrelevant
// until its first write fault, when page's anon_vma and index
// are set.  But now set the vm_pgoff it will almost certainly
// end up with (unless mremap moves it elsewhere before that
// first wfault), so /proc/pid/maps tells a consistent story.
//
// By setting it to reflect the virtual start address of the
// vma, merges and splits can happen in a seamless way, just
// using the existing file pgoff checks and manipulations.
// Similarly in do_mmap and in do_brk_flags.
if vma_is_anonymous(vma as *const vm_area_struct) {
let mut __v1: i64;
let __ret_warn_on: i32 = !unsafe { (*vma).anon_vma }.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("739: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 739b - ., 3; .popsection\n# {0}", const 739 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_insert_vm_struct_2, sym __slate_asm_string_insert_vm_struct_3, const 3436 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("740: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 740b - ., 4; .popsection\n# {0}", const 740 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
vma_set_pgoff(vma, (unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }) >> 12i32);
}
vma_set_anon_pgoff(vma, (unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start }) >> 12i32);
if vma_link(mm, vma) != 0 {
if vma_test(vma as *const vm_area_struct, 20i32) {
vm_unacct_memory(charged as i64);
}
return 12i32.wrapping_neg();
}
0
}

/// vma_mmu_pagesize - Default MMU page size granularity for this VMA.
/// @vma: The user mapping.
///
/// In the common case, the default page size used by the MMU matches the
/// default page size used by the kernel (see vma_kernel_pagesize()). On
/// architectures where it differs, an architecture-specific 'strong' version
/// of this symbol is required.
///
/// The default MMU page size is not affected by Transparent Huge Pages
/// being in effect, or any usage of larger MMU page sizes (either through
/// architectural huge-page mappings or other explicit/implicit coalescing of
/// virtual ranges performed by the MMU).
///
/// Return: The default MMU page size granularity for this VMA.
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn vma_mmu_pagesize(vma: *mut vm_area_struct) -> u64 {
vma_kernel_pagesize(vma)
}

#[unsafe(no_mangle)]
extern "C" fn __install_special_mapping(mm: *mut mm_struct, addr: u64, len: u64, vm_flags: u64, r#priv: *mut (), ops: *const vm_operations_struct) -> *mut vm_area_struct {
let mut __slate_compound_3: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: [i32; 1] = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_mm_types_h_871 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: [i32; 2] = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut vma: core::mem::MaybeUninit<*mut vm_area_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut vm_area_struct = core::ptr::addr_of_mut!(vma) as *mut *mut vm_area_struct;
let mut vma_flags: core::mem::MaybeUninit<__SlateRecord_mm_types_h_871> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut __SlateRecord_mm_types_h_871 = core::ptr::addr_of_mut!(vma_flags) as *mut __SlateRecord_mm_types_h_871;
unsafe {
core::ptr::write(__slate_slot_2, legacy_to_vma_flags(vm_flags));
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_1 = vm_area_alloc(mm);
if unlikely!((*__slate_slot_1).is_null()) != 0 {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut vm_area_struct;
} else {
vma_flags_set_mask(core::ptr::addr_of_mut!(*__slate_slot_2), (*mm).__slate_anon_0.__slate_anon_28.def_vma_flags);
vma_flags_set_mask(core::ptr::addr_of_mut!(*__slate_slot_2), __mk_vma_flags(*{
__slate_compound_3 = {
let mut __t0: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t0
};
core::ptr::addr_of_mut!(__slate_compound_3)
}, 1u64, (*{
__slate_compound_4 = [18i32];
core::ptr::addr_of_mut!(__slate_compound_4)
}).as_ptr() as *const i32));
vma_flags_clear_mask(core::ptr::addr_of_mut!(*__slate_slot_2), __mk_vma_flags(*{
__slate_compound_5 = {
let mut __t1: __SlateRecord_mm_types_h_871 = core::mem::zeroed();
__t1
};
core::ptr::addr_of_mut!(__slate_compound_5)
}, 2u64, (*{
__slate_compound_6 = [13i32, 19i32];
core::ptr::addr_of_mut!(__slate_compound_6)
}).as_ptr() as *const i32));
(*(*__slate_slot_1)).__slate_anon_3.flags = *__slate_slot_2;
(*(*__slate_slot_1)).vm_page_prot = vma_get_page_prot(*__slate_slot_1 as *const vm_area_struct);
(*(*__slate_slot_1)).vm_ops = ops;
(*(*__slate_slot_1)).vm_private_data = r#priv;
vma_set_range(*__slate_slot_1, addr, addr.wrapping_add(len), 0u64, addr >> 12i32);
*__slate_slot_0 = insert_vm_struct(mm, *__slate_slot_1);
if *__slate_slot_0 != 0 {
vm_area_free(*__slate_slot_1);
return ERR_PTR(*__slate_slot_0 as i64) as *mut vm_area_struct;
} else {
vm_stat_account(mm, (*(*__slate_slot_1)).__slate_anon_3.vm_flags, (len >> 12i32) as i64);
perf_event_mmap(*__slate_slot_1);
return *__slate_slot_1;
}
}
}
}

static __slate_asm_string___mas_set_range_1: [u8; 1] = *b"\0";

static __slate_asm_string___mas_set_range_2: [u8; 56] = *b"/home/takashi/c-corpus/linux/include/linux/maple_tree.h\0";

static __slate_asm_string___mmap_new_file_vma_1: [u8; 1] = *b"\0";

static __slate_asm_string___mmap_new_file_vma_2: [u8; 38] = *b"/home/takashi/c-corpus/linux/mm/vma.c\0";

static __slate_asm_string___split_vma_10: [u8; 1] = *b"\0";

static __slate_asm_string___split_vma_11: [u8; 38] = *b"/home/takashi/c-corpus/linux/mm/vma.c\0";

static __slate_asm_string___split_vma_12: [u8; 1] = *b"\0";

static __slate_asm_string___split_vma_13: [u8; 38] = *b"/home/takashi/c-corpus/linux/mm/vma.c\0";

static __slate_asm_string_file_ref_inc_3: [u8; 44] = *b"file_ref_inc() on a released file reference\0";

static __slate_asm_string_file_ref_inc_4: [u8; 54] = *b"/home/takashi/c-corpus/linux/include/linux/file_ref.h\0";

static __slate_asm_string_insert_vm_struct_2: [u8; 1] = *b"\0";

static __slate_asm_string_insert_vm_struct_3: [u8; 38] = *b"/home/takashi/c-corpus/linux/mm/vma.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_mm_drop_all_locks_0: [u8; 1] = *b"\0";

static __slate_asm_string_mm_drop_all_locks_1: [u8; 38] = *b"/home/takashi/c-corpus/linux/mm/vma.c\0";

static __slate_asm_string_rwsem_assert_held_write_nolockdep_1: [u8; 1] = *b"\0";

static __slate_asm_string_rwsem_assert_held_write_nolockdep_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/include/linux/rwsem.h\0";

static __slate_asm_string_vm_lock_anon_vma_2: [u8; 1] = *b"\0";

static __slate_asm_string_vm_lock_anon_vma_3: [u8; 38] = *b"/home/takashi/c-corpus/linux/mm/vma.c\0";

static __slate_asm_string_vm_lock_mapping_1: [u8; 1] = *b"\0";

static __slate_asm_string_vm_lock_mapping_2: [u8; 38] = *b"/home/takashi/c-corpus/linux/mm/vma.c\0";

static __slate_asm_string_vm_unlock_anon_vma_2: [u8; 1] = *b"\0";

static __slate_asm_string_vm_unlock_anon_vma_3: [u8; 38] = *b"/home/takashi/c-corpus/linux/mm/vma.c\0";

static __slate_asm_string_vm_unlock_mapping_1: [u8; 1] = *b"\0";

static __slate_asm_string_vm_unlock_mapping_2: [u8; 38] = *b"/home/takashi/c-corpus/linux/mm/vma.c\0";

static __slate_asm_string_vma_assert_attached_1: [u8; 1] = *b"\0";

static __slate_asm_string_vma_assert_attached_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/include/linux/mmap_lock.h\0";

static __slate_asm_string_vma_assert_detached_1: [u8; 1] = *b"\0";

static __slate_asm_string_vma_assert_detached_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/include/linux/mmap_lock.h\0";

static __slate_asm_string_vma_complete_10: [u8; 38] = *b"/home/takashi/c-corpus/linux/mm/vma.c\0";

static __slate_asm_string_vma_complete_9: [u8; 1] = *b"\0";
