
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
//! linux/mm/oom_kill.c
//!
//! Copyright (C)  1998,2000  Rik van Riel
//! Thanks go out to Claus Fischer for some serious inspiration and
//! for goading me into coding this file...
//! Copyright (C)  2010  Google, Inc.
//! Rewritten by David Rientjes
//!
//! The routines in this file are used to kill a process when
//! we're seriously out of memory. This gets called from __alloc_pages()
//! in mm/page_alloc.c when we really run out of memory.
//!
//! Since we won't call these routines often (on a well-configured
//! machine) this file will double as a 'coding guide' and a signpost
//! for newbie kernel hackers. It features several pointers to major
//! kernel subsystems and hints as to where to find out what things do.
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
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! Create a dummy function pointer reference to prevent objtool from marking
//! the function as needing to be "sealed" (i.e. ENDBR converted to NOP by
//! apply_seal_endbr()).
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! # chars in a path name including nul
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
//! Defines x86 CPU feature bits
//! This counts to 15. Any more, it will return 16th argument.
//! Operation not permitted
//! Value too large for defined data type
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
//! 64bit FS base
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
//! Needs two entries
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! writeable
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
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! u64_to_user_ptr - cast a pointer passed as u64 from user space to void __user *
//! @x: The u64 value from user space, usually via IOCTL
//!
//! u64_to_user_ptr() simply casts a pointer passed as u64 from user space to void
//! __user * correctly. Using this lets us get rid of all the tiresome casts.
//! used to decode ioctl numbers..
//! Wrap the arch provided IRQ routines to provide appropriate checks.
//! New user namespace
//! The size of each LDT entry.
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
//! Instruction encoding provided for binutils backwards compatibility.
//! Defines VMX CPU feature bits
//! Change personality of the currently running process.
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
//! __MAX_NR_ZONES
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
//! Keep the same order as in fs/proc/proc_sysctl.c
//! /proc/<pid>/oom_score_adj set to OOM_SCORE_ADJ_MIN disables oom killing for
//! pid.
//! Must be a compile-time constant, so implement it as a macro
//! Resource type
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
//! DEBUG: Poison objects
//! return the first ptr or next element in an RCU protected list
//! Keep unconverted code working
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! per-IO O_SYNC
//! File supports atomic writes
//! number of env pointers
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! Generate SD flag indexes
//! Desired design of maximum size and alignment (see RFC2553)
//! Mapping of registers to parameters for syscalls on x86-64 and x32
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
//! 31 pointers + header align the folio_batch structure to a power of two
//! Internal flags that share the struct mempolicy flags word with
//! "mode flags".  These flags are allocated from bit 0 up, as they
//! are never OR'ed into the mode in mempolicy API arguments.
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! The trampoline is 8 bytes and of the general form:
//!
//!   jmp.d32 \func
//!   ud1 %esp, %ecx
//!
//! That trailing #UD provides both a speculation stop and serves as a unique
//! 3 byte signature identifying static call trampolines. Also see tramp_ud[]
//! and __static_call_fixup().
//! These are specified by iBCS2
//! Epoll event masks
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! The PERF_METRICS MSR is modeled as several magic fixed-mode PMCs, one for
//! each TopDown metric event.
//!
//! Internally the TopDown metric events are mapped to the FxCtr 3 (SLOTS).
//! Macro to enforce the same ordering and stack sizes
//! This is a list of *what* is being read, not *how* nor *where*.
//! ld/ldx fields
//! Define a list of socket types which can be the argument for
//! skc_to_*_sock() helpers. All these sockets should have
//! sock_common as the first argument in its memory layout.
//! Choose 31 as it makes rqspinlock_held cacheline-aligned.
//! Used for event string fields when they are NULL
//! If more than 16 keys are ever supported, a thorough audit
//! will be necessary to ensure that the types that store key
//! numbers and masks have sufficient capacity.
//! Set global bits for all 4 regs
//! Define bits that are always set to 1 in DR7, only bit 10 is
//! architecturally reserved to '1'.
//!
//! This is also the init/reset value for DR7.
//! Replace with "lkgs %di" once binutils support LKGS instruction
//! See the comment in asm-generic/io.h about ioremap_np().
//! If we can't allocate a page to make a big batch of page pointers
//! to work on, then just handle a few from the on-stack structure.
//! INVLPGB can be targeted by virtual address, PCID, ASID, or any combination
//! of the three. For example:
//! - FLAG_VA | FLAG_INCLUDE_GLOBAL: invalidate all TLB entries at the address
//! - FLAG_PCID:			    invalidate all TLB entries matching the PCID
//!
//! The first is used to invalidate (kernel) mappings at a particular
//! address across all processes.
//!
//! The latter invalidates all TLB entries matching a PCID.
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
//! Flags passed to __show_mem() and show_free_areas() to suppress output in
//! various contexts.
//! slab's alloc_flags definitions
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
// Make the optimizer believe the variable can be manipulated arbitrarily.
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
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
// flex_array_size() - Calculate size of a flexible array member
//                     within an enclosing structure.
// @p: Pointer to the structure.
// @member: Name of the flexible array member.
// @count: Number of elements in the array.
//
// Calculates size of a flexible array of @count number of @member
// elements, at the end of structure @p.
//
// Return: number of bytes needed or SIZE_MAX on overflow.
// struct_size() - Calculate size of structure with trailing flexible array.
// @p: Pointer to the structure.
// @member: Name of the array member.
// @count: Number of elements in the array.
//
// Calculates size of memory needed for structure of @p followed by an
// array of @count number of @member elements.
//
// Return: number of bytes needed or SIZE_MAX on overflow.
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
// pr_info - Print an info-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_INFO loglevel. It uses pr_fmt() to
// generate the format string.
// pr_cont - Continues a previous log message in the same line.
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_CONT loglevel. It should only be
// used when continuing a log message with no newline ('\n') enclosed. Otherwise
// it defaults back to KERN_DEFAULT loglevel.
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
// Atomic operations are already serializing on x86
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
// No such file or directory
// No such process
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
// Overflow Flag
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
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
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
// SCHED_ISO: reserved but not implemented yet
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
// is terminating due to OOM killer
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// disable sas during sighandling
// This function is needed by all SMP systems. It must _always_ be valid
// from the initial startup.
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
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
// list_for_each_entry_rcu	-	iterate over rcu list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// @cond:	optional lockdep expression if called from non-RCU protection.
//
// This list-traversal primitive may safely run concurrently with
// the _rcu list-mutation primitives such as list_add_rcu()
// as long as the traversal is guarded by rcu_read_lock().
// return the first or the next element in an RCU protected hlist
// Same as rb_first(), but O(1)
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
// write_seqcount_begin() - start a seqcount_t write side critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Context: sequence counter write side sections must be serialized and
// non-preemptible. Preemption will be automatically disabled if and
// only if the seqcount write serialization lock is associated, and
// preemptible.  If readers can be invoked from hardirq or softirq
// context, interrupts or bottom halves must be respectively disabled.
// write_seqcount_end() - end a seqcount_t write side critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Context: Preemption will be automatically re-enabled if and only if
// the seqcount write serialization lock is associated, and preemptible.
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
// Shift up (to get rid of type), then down to get value
// Shift the offset up "too far" by TYPE bits, then down again
// The offset is inverted by a binary not operation to make the high
// physical bits set.
// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
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
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Resource extended types
// No address assigned yet
// Indicate a kmalloc slab
// Use GFP_DMA memory
// DEBUG: Store the last owner for bug hunting
// define SLAB_TYPESAFE_BY_RCU - **WARNING** READ THIS!
//
// This delays freeing the SLAB page by a grace period, it does _NOT_
// delay object freeing. This means that if you do kmem_cache_free()
// that memory location is free to be reused at any time. Thus it may
// be possible to see another object there in the same RCU grace period.
//
// This feature only ensures the memory location backing the object
// stays valid, the trick to using this is relying on an independent
// object validation pass. Something like:
//
// ::
//
//  begin:
//   rcu_read_lock();
//   obj = lockless_lookup(key);
//   if (obj) {
//     if (!try_get_ref(obj)) // might fail for free objects
//       rcu_read_unlock();
//       goto begin;
//
//     if (obj->key != key) { // not the object we expected
//       put_ref(obj);
//       rcu_read_unlock();
//       goto begin;
//     }
//   }
//  rcu_read_unlock();
//
// This is useful if we need to approach a kernel structure obliquely,
// from its address obtained without the usual locking. We can lock
// the structure to stabilize it and check it's still at the given address,
// only if we can be sure that the memory has not been meanwhile reused
// for some other kind of object (which our subsystem's lock might corrupt).
//
// rcu_read_lock before reading the address, then rcu_read_unlock after
// taking the spinlock within the structure expected at that address.
//
// Note that object identity check has to be done *after* acquiring a
// reference, therefore user has to ensure proper ordering for loads.
// Similarly, when initializing objects allocated with SLAB_TYPESAFE_BY_RCU,
// the newly allocated object has to be fully initialized *before* its
// refcount gets initialized and proper ordering for stores is required.
// refcount_{add|inc}_not_zero_acquire() and refcount_set_release() are
// designed with the proper fences required for reference counting objects
// allocated with SLAB_TYPESAFE_BY_RCU.
//
// Note that it is not possible to acquire a lock within a structure
// allocated with SLAB_TYPESAFE_BY_RCU without first acquiring a reference
// as described above.  The reason is that SLAB_TYPESAFE_BY_RCU pages
// are not zeroed before being given to the slab, which means that any
// locks must be initialized after each and every kmem_struct_alloc().
// Alternatively, make the ctor passed to kmem_cache_create() initialize
// the locks at page-allocation time, as is done in __i915_request_ctor(),
// sighand_ctor(), and anon_vma_ctor().  Such a ctor permits readers
// to safely acquire those ctor-initialized locks under rcu_read_lock()
// protection.
//
// Note that SLAB_TYPESAFE_BY_RCU was originally named SLAB_DESTROY_BY_RCU.
// no-op
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
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
// hlist_nulls_next_rcu - returns the element of the list after @node.
// @node: element of the list.
// hlist_nulls_pprev_rcu - returns the dereferenced pprev of @node.
// @node: element of the list.
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
// task_css_check - obtain css for (task, subsys) w/ extra access conds
// @task: the target task
// @subsys_id: the target subsystem ID
// @__c: extra condition expression to be passed to rcu_dereference_check()
//
// Return the cgroup_subsys_state for the (@task, @subsys_id) pair.  The
// synchronization rules are the same as task_css_set_check().
// Generate SD flag bits
// Implementation specific max size
// Mapping of registers to parameters for syscalls on i386
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
// PT_TRACE_* event enable flags
// These special macros can be used to get or set a debugging register
// The implied mode when all bits are clear:
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
macro_rules! ACCESS_PRIVATE {
($p:expr, $member:ident) => {
unsafe { (*$p).$member.as_mut_ptr() as *mut u64 }
};
}

/// @a is a power of 2 value
macro_rules! ALIGN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!($x, $a)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__7 {
() => {
((core::mem::size_of::<[trace_print_flags; 4]>() as u64) / (core::mem::size_of::<trace_print_flags>() as u64)).wrapping_add(__must_be_array__59!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__8 {
() => {
((core::mem::size_of::<[trace_print_flags; 3]>() as u64) / (core::mem::size_of::<trace_print_flags>() as u64)).wrapping_add(__must_be_array__29!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__14 {
() => {
((core::mem::size_of::<[trace_print_flags; 3]>() as u64) / (core::mem::size_of::<trace_print_flags>() as u64)).wrapping_add(__must_be_array__31!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__38 {
() => {
((core::mem::size_of::<[ctl_table; 3]>() as u64) / (core::mem::size_of::<ctl_table>() as u64)).wrapping_add(__must_be_array__51!() as u64)
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

macro_rules! GENMASK_INPUT_CHECK {
() => {
core::mem::size_of::<__SlateRecord_find_h_67>() as i32
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
macro_rules! GENMASK_TYPE {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
};
}

macro_rules! GENMASK {
($h:expr, $l:expr) => {
GENMASK_TYPE!(u64, $h, $l)
};
}

macro_rules! GENMASK__2 {
($h:expr, $l:expr) => {
GENMASK_TYPE__2!(u64, $h, $l)
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

/// in mm/page_alloc.c
macro_rules! K {
($x:expr) => {
($x) << 2i32
};
}

macro_rules! NODE_DATA {
($nid:expr) => {
unsafe { *(core::ptr::addr_of_mut!(node_data) as *mut *mut pglist_data).wrapping_offset(($nid) as isize) }
};
}

macro_rules! PG_COUNT_TO_KB {
($x:expr) => {
($x) << 2i32
};
}

macro_rules! RCU_LOCKDEP_WARN {
($c:expr) => {
loop {
let __v: bool = false && $c;
if !__v {
break;
}
}
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

macro_rules! VM_BUG_ON {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_WARN_ON_ONCE {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! _AC {
() => {
1u64
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

macro_rules! ___wait_cond_timeout {
($condition:expr, $__ret:expr) => {
{
let mut __v: bool;
let __v2: bool = $condition;
if __v2 && ($__ret) == 0 {
$__ret = 1;
}
__v = __v2 || ($__ret) == 0;
__v
}
};
}

macro_rules! ___wait_is_interruptible {
($state:expr) => {
false || ($state) & (1 | 256) != 0
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

/// __acquire_shared() - function to acquire context lock shared
/// @x: context lock instance pointer
///
/// No-op function that acquires the given context lock instance @x with shared
/// access.
macro_rules! __acquire_shared {
() => {
__acquire_shared_ctx_lock!();
};
}

macro_rules! __acquire_shared_ctx_lock {
() => {
while false {
}
};
}

macro_rules! __dynamic_array__5 {
($type:ty, $item:ident, $len:expr, $__item_length:expr, $__data_offsets:expr, $__data_size:expr) => {
$__item_length = $len.wrapping_mul(core::mem::size_of::<$type>() as u64) as i32;
unsafe {
(*$__data_offsets).$item = (($__data_size) as u64).wrapping_add(offsetof__53!()) as u32;
}
let __v: *mut trace_event_data_offsets_mark_victim = $__data_offsets;
let __v2: u32 = unsafe { (*__v).$item };
let __v3: u32 = __v2 | ((($__item_length) << 16i32) as u32);
unsafe {
(*__v).$item = __v3;
}
let __v4: i32 = $__data_size;
let __v5: i32 = __v4.wrapping_add($__item_length);
$__data_size = __v5;
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

macro_rules! __get_str__4 {
($entry:expr) => {
((unsafe { ((($entry) as *mut ()) as *const u8).wrapping_offset(((*$entry).__data_loc_comm & 65535) as isize) }) as *mut ()) as *mut u8
};
}

macro_rules! __get_str__6 {
($field__2:expr) => {
((unsafe { ((($field__2) as *mut ()) as *const u8).wrapping_offset(((*$field__2).__data_loc_comm & 65535) as isize) }) as *mut ()) as *mut u8
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

macro_rules! __is_nonneg {
($ux:expr) => {
statically_true!((($ux) as i64) >= 0)
};
}

macro_rules! __must_be_array__29 {
() => {
core::mem::size_of::<__SlateRecord_oom_h_215>() as i32
};
}

macro_rules! __must_be_array__31 {
() => {
core::mem::size_of::<__SlateRecord_oom_h_214>() as i32
};
}

macro_rules! __must_be_array__51 {
() => {
core::mem::size_of::<__SlateRecord_oom_kill_c_730>() as i32
};
}

macro_rules! __must_be_array__59 {
() => {
core::mem::size_of::<__SlateRecord_oom_h_69>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__8 {
() => {
((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__36 {
() => {
((const { core::mem::size_of::<*mut mm_struct>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mm_struct>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mm_struct>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mm_struct>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__6 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__30 {
() => {
((const { core::mem::size_of::<*const cred>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*const cred>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*const cred>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*const cred>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__26 {
() => {
((const { core::mem::size_of::<*mut task_struct>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut task_struct>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut task_struct>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut task_struct>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__10 {
() => {
((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
() => {
((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__16 {
() => {
((const { core::mem::size_of::<*mut bpf_prog_array>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut bpf_prog_array>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut bpf_prog_array>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut bpf_prog_array>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__11 {
() => {
((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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

macro_rules! __preempt_count_inc {
() => {
__preempt_count_add(1i32);
};
}

macro_rules! __ratelimit__12 {
($state:expr) => {
unsafe { ___ratelimit($state, c"pagefault_out_of_memory".as_ptr() as *const u8) }
};
}

macro_rules! __ratelimit__14 {
($state:expr) => {
unsafe { ___ratelimit($state, c"oom_kill_process".as_ptr() as *const u8) }
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

/// __release_shared() - function to release context lock shared
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x with shared
/// access.
macro_rules! __release_shared {
() => {
__release_shared_ctx_lock!();
};
}

macro_rules! __release_shared_ctx_lock {
() => {
while false {
}
};
}

/// __sign_use for integer expressions:
///   bit #0 set if ok for unsigned comparisons
///   bit #1 set if ok for signed comparisons
///
/// In particular, statically non-negative signed integer expressions
/// are ok for both.
///
/// NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
/// in expressions, and are accepted for signed conversions for now.
/// This is debatable.
///
/// Note that 'x' is the original expression, and 'ux' is the unique variable
/// that contains the value.
///
/// We use 'ux' for pure type checking, and 'x' for when we need to look at the
/// value (but without evaluating it for side effects!
/// Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
///
/// Pointers end up being checked by the normal C type rules at the actual
/// comparison, and these expressions only need to be careful to not cause
/// warnings for pointer use.
macro_rules! __sign_use__2 {
($ux:expr) => {
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
};
}

/// __sign_use for integer expressions:
///   bit #0 set if ok for unsigned comparisons
///   bit #1 set if ok for signed comparisons
///
/// In particular, statically non-negative signed integer expressions
/// are ok for both.
///
/// NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
/// in expressions, and are accepted for signed conversions for now.
/// This is debatable.
///
/// Note that 'x' is the original expression, and 'ux' is the unique variable
/// that contains the value.
///
/// We use 'ux' for pure type checking, and 'x' for when we need to look at the
/// value (but without evaluating it for side effects!
/// Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
///
/// Pointers end up being checked by the normal C type rules at the actual
/// comparison, and these expressions only need to be careful to not cause
/// warnings for pointer use.
macro_rules! __sign_use__4 {
($ux:expr) => {
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<u64>() as u64) < 4) as i32)) }
};
}

macro_rules! __string__2 {
($item:ident, $entry:expr, $__data_offsets:expr) => {
unsafe {
(*$entry).__data_loc_comm = $__data_offsets.$item;
}
};
}

macro_rules! __timer_init {
($_timer:expr, $_fn:expr, $_flags:expr) => {
unsafe { timer_init_key($_timer, $_fn, ($_flags) as u32, core::ptr::null::<u8>(), core::ptr::null_mut::<lock_class_key>()) };
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

macro_rules! __types_ok__3 {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__4!($uy)
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
macro_rules! __verify_pcpu_ptr__6 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<u64>() as *const ();
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
macro_rules! __verify_pcpu_ptr__3 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<__SlateRecord_types_h_197>() as *const ();
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
macro_rules! __verify_pcpu_ptr__4 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i64>() as *const ();
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
let __vpp_verify: *const () = core::ptr::null_mut::<hlist_head>() as *const ();
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

macro_rules! arch_atomic_sub_return {
($i:expr, $v:expr) => {
arch_atomic_add_return($i.wrapping_neg(), $v)
};
}

macro_rules! clear_thread_flag {
($flag:expr) => {
clear_ti_thread_flag(current_thread_info!(), $flag);
};
}

macro_rules! compact_result_to_feedback {
($result:expr) => {
{
let mut __v: i32;
let __v2: u32 = $result;
__v = if (__v2 as u32) == 5 { 1 } else { if (__v2 as u32) == 8 { 3 } else { 2 } };
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
macro_rules! container_of__52 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__33!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

macro_rules! current_thread_info {
() => {
get_current() as *mut thread_info
};
}

macro_rules! first_node__3 {
($src:expr) => {
__first_node(unsafe { core::ptr::addr_of!($src) })
};
}

macro_rules! first_node {
($src:expr) => {
__first_node((unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17)
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__2 {
() => {
1i32.wrapping_neg() < 1
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

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

macro_rules! likely__14 {
($x:expr) => {
!$x.is_null() as i64
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! likely__2 {
($x:expr) => {
(($x) != 0) as i64
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

macro_rules! next_node__3 {
($n:expr, $src:expr) => {
__next_node($n, unsafe { core::ptr::addr_of!($src) })
};
}

macro_rules! next_node {
($n:expr, $src:expr) => {
__next_node($n, (unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17)
};
}

macro_rules! node_present_pages {
($nid:expr) => {
unsafe { (*NODE_DATA!($nid)).node_present_pages }
};
}

macro_rules! nodes_subset__2 {
($src1:expr, $src2:expr) => {
__nodes_subset((unsafe { core::ptr::addr_of_mut!($src1) }) as *const __SlateRecord_nodemask_types_h_17, unsafe { core::ptr::addr_of!($src2) }, 64u32)
};
}

macro_rules! offsetof__33 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, oom_reaper_timer) as u64
};
}

macro_rules! offsetof__53 {
() => {
66u64
};
}

macro_rules! offsetof__9 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, tasks) as u64
};
}

macro_rules! offsetof__6 {
() => {
core::mem::offset_of!(task_struct, thread_node) as u64
};
}

/// We abuse bit 3 from flags to pass exact information, see
/// perf_arch_misc_flags() and the comment with PERF_EFLAGS_EXACT.
macro_rules! perf_arch_fetch_caller_regs {
($regs:expr, $__ip:expr) => {
unsafe {
(*$regs).ip = $__ip;
}
unsafe {
(*$regs).sp = __slate_intrinsic_2d7c29df8fdf6378(const { ((0 as i32) as u32) as i32 }) as u64;
}
unsafe {
(*$regs).__slate_anon_17.cs = 16u16;
}
unsafe {
(*$regs).flags = 0;
}
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
$flags = unsafe { _raw_spin_lock_irqsave($lock) };
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

macro_rules! rcu_lock_acquire {
() => {
while false {
}
};
}

macro_rules! rcu_lock_release {
() => {
while false {
}
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

macro_rules! static_branch_unlikely__9 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const9(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
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

macro_rules! static_branch_unlikely__4 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const4(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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

macro_rules! static_branch_unlikely__8 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const8(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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

macro_rules! static_branch_unlikely__6 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const6(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__5 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const5(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__7 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const7(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

/// Useful shorthand for "is this condition known at compile-time?"
///
/// Note that the condition may involve non-constant values,
/// but the compiler may know enough about the details of the
/// values to determine that the condition is statically true.
macro_rules! statically_true {
($x:expr) => {
(unsafe { __slate_intrinsic_1cfe0293d0a2af8d($x) }) && $x
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

macro_rules! test_bit__6 {
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

macro_rules! type_max {
() => {
__type_max!()
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

macro_rules! unlikely__3 {
($x:expr) => {
(($x) != 0) as i64
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

macro_rules! wake_up_all {
($x:expr) => {
unsafe { __wake_up($x, 3u32, 0i32, core::ptr::null_mut::<()>()) };
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut this_cpu_off: u64;
    static mut __SCK__might_resched: static_call_key;
    static mut nr_node_ids: u32;
    static mut node_states: [__SlateRecord_nodemask_types_h_17; 6];
    static mut node_data: [*mut pglist_data; 0];
    static mut __cpu_online_mask: cpumask;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule_notrace: static_call_key;
    static mut cpu_number: i32;
    static mut sighand_cachep: *mut kmem_cache;
    static mut jiffies: u64;
    static mut __tracepoint_mmap_lock_start_locking: tracepoint;
    static mut __tracepoint_mmap_lock_acquire_returned: tracepoint;
    static mut __tracepoint_mmap_lock_released: tracepoint;
    static mut init_task: task_struct;
    static mut init_user_ns: user_namespace;
    static mut sysctl_vals: [i32; 0];
    static mut _totalram_pages: __SlateRecord_types_h_197;
    static mut vm_event_states: vm_event_state;
    static mut vm_node_stat: [__SlateRecord_types_h_197; 66];
    static mut kstat: kernel_stat;
    static mut total_swap_pages: i64;
    static mut tracepoint_srcu: srcu_struct;
    static mut perf_callchain_entry: perf_callchain_entry;
    static mut cpusets_enabled_key: static_key_false;
    static mut kmem_cache: *mut kmem_cache;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn panic(arg0: *const u8, ...);
    fn ___ratelimit(arg0: *mut ratelimit_state, arg1: *const u8) -> i32;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn dump_stack();
    fn __compiletime_assert_14();
    fn __cmpxchg_wrong_size();
    fn __xadd_wrong_size();
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn ____wrong_branch_error() -> bool;
    fn __bad_size_call_parameter();
    fn _find_next_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn _find_first_bit(arg0: *const u64, arg1: u64) -> u64;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn strlen(arg0: *const u8) -> u64;
    fn __bitmap_subset(arg0: *const u64, arg1: *const u64, arg2: u32) -> bool;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __compiletime_assert_47();
    fn __compiletime_assert_50();
    fn __compiletime_assert_54();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn schedule_timeout(arg0: i64) -> i64;
    fn schedule_timeout_idle(arg0: i64) -> i64;
    fn schedule();
    fn wake_up_process(arg0: *mut task_struct) -> i32;
    fn call_rcu(arg0: *mut callback_head, arg1: Option<unsafe extern "C" fn(*mut callback_head)>);
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn rcu_is_watching() -> bool;
    fn do_send_sig_info(arg0: i32, arg1: *mut kernel_siginfo, arg2: *mut task_struct, arg3: u32) -> i32;
    fn refcount_warn_saturate(arg0: *mut refcount_struct, arg1: u32);
    fn mas_find_rev(arg0: *mut ma_state, arg1: u64) -> *mut ();
    fn down_read_killable(arg0: *mut rw_semaphore) -> i32;
    fn down_read_trylock(arg0: *mut rw_semaphore) -> i32;
    fn up_read(arg0: *mut rw_semaphore);
    fn __wake_up(arg0: *mut wait_queue_head, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn init_wait_entry(arg0: *mut wait_queue_entry, arg1: i32);
    fn prepare_to_wait_event(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry, arg2: i32) -> i64;
    fn finish_wait(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry);
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn add_timer(arg0: *mut timer_list);
    fn mutex_lock_killable(arg0: *mut mutex) -> i32;
    fn mutex_unlock(arg0: *mut mutex);
    fn __compiletime_assert_216();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn blocking_notifier_chain_register(arg0: *mut blocking_notifier_head, arg1: *mut notifier_block) -> i32;
    fn blocking_notifier_chain_unregister(arg0: *mut blocking_notifier_head, arg1: *mut notifier_block) -> i32;
    fn blocking_notifier_call_chain(arg0: *mut blocking_notifier_head, arg1: u64, arg2: *mut ()) -> i32;
    fn __percpu_counter_sum(arg0: *mut percpu_counter) -> i64;
    fn __compiletime_assert_232();
    fn __next_zones_zonelist(arg0: *mut zoneref, arg1: u32, arg2: *const __SlateRecord_nodemask_types_h_17) -> *mut zoneref;
    fn __mmdrop(arg0: *mut mm_struct);
    fn __compiletime_assert_260();
    fn __mmap_lock_do_trace_start_locking(arg0: *mut mm_struct, arg1: bool);
    fn __mmap_lock_do_trace_acquire_returned(arg0: *mut mm_struct, arg1: bool, arg2: bool);
    fn __mmap_lock_do_trace_released(arg0: *mut mm_struct, arg1: bool);
    fn __put_task_struct_rcu_cb(arg0: *mut callback_head);
    fn proc_dointvec(arg0: *const ctl_table, arg1: i32, arg2: *mut (), arg3: *mut u64, arg4: *mut i64) -> i32;
    fn proc_dointvec_minmax(arg0: *const ctl_table, arg1: i32, arg2: *mut (), arg3: *mut u64, arg4: *mut i64) -> i32;
    fn __register_sysctl_init(arg0: *const u8, arg1: *const ctl_table, arg2: *const u8, arg3: u64);
    fn __put_cred(arg0: *mut cred);
    fn get_task_cred(arg0: *mut task_struct) -> *const cred;
    fn pidfd_get_task(arg0: i32, arg1: *mut u32) -> *mut task_struct;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_344();
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn __compiletime_assert_419();
    fn __show_mem(arg0: u32, arg1: *const __SlateRecord_nodemask_types_h_17, arg2: i32);
    fn sched_show_task(arg0: *mut task_struct);
    fn kthread_create_on_node(arg0: Option<unsafe extern "C" fn(*mut ()) -> i32>, arg1: *mut (), arg2: i32, arg3: *const u8, ...) -> *mut task_struct;
    fn trace_print_symbols_seq(arg0: *mut trace_seq, arg1: u64, arg2: *const trace_print_flags, arg3: u64) -> *const u8;
    fn trace_raw_output_prep(arg0: *mut trace_iterator, arg1: *mut trace_event) -> i32;
    fn trace_event_printf(arg0: *mut trace_iterator, arg1: *const u8, ...);
    fn trace_handle_return(arg0: *mut trace_seq) -> u32;
    fn trace_event_reg(arg0: *mut trace_event_call, arg1: u32, arg2: *mut ()) -> i32;
    fn trace_event_buffer_reserve(arg0: *mut trace_event_buffer, arg1: *mut trace_event_file, arg2: u64) -> *mut ();
    fn trace_event_buffer_commit(arg0: *mut trace_event_buffer);
    fn __compiletime_assert_678();
    fn __trace_trigger_soft_disabled(arg0: *mut trace_event_file) -> bool;
    fn trace_event_raw_init(arg0: *mut trace_event_call) -> i32;
    fn perf_trace_buf_alloc(arg0: i32, arg1: *mut *mut pt_regs, arg2: *mut i32) -> *mut ();
    fn perf_trace_run_bpf_submit(arg0: *mut (), arg1: i32, arg2: i32, arg3: *mut trace_event_call, arg4: u64, arg5: *mut pt_regs, arg6: *mut hlist_head, arg7: *mut task_struct);
    fn cpuset_current_node_allowed(arg0: i32, arg1: u32) -> bool;
    fn cpuset_mems_allowed_intersects(arg0: *const task_struct, arg1: *const task_struct) -> i32;
    fn cpuset_print_current_mems_allowed();
    fn mempolicy_in_oom_domain(arg0: *mut task_struct, arg1: *const __SlateRecord_nodemask_types_h_17) -> bool;
    fn thaw_process(arg0: *mut task_struct);
    fn set_freezable() -> bool;
    fn zap_vma_for_reaping(arg0: *mut vm_area_struct) -> i32;
    fn reciprocal_value(arg0: u32) -> reciprocal_value;
    fn dump_unreclaimable_slab();
    fn __SCT__tp_func_oom_score_adj_update(arg0: *mut (), arg1: *mut task_struct) -> i32;
    fn __SCT__tp_func_reclaim_retry_zone(arg0: *mut (), arg1: *mut zoneref, arg2: i32, arg3: u64, arg4: u64, arg5: u64, arg6: i32, arg7: bool) -> i32;
    fn __SCT__tp_func_mark_victim(arg0: *mut (), arg1: *mut task_struct, arg2: u32) -> i32;
    fn __compiletime_assert_753();
    fn __SCT__tp_func_wake_reaper(arg0: *mut (), arg1: i32) -> i32;
    fn __compiletime_assert_760();
    fn __SCT__tp_func_start_task_reaping(arg0: *mut (), arg1: i32) -> i32;
    fn __compiletime_assert_767();
    fn __SCT__tp_func_finish_task_reaping(arg0: *mut (), arg1: i32) -> i32;
    fn __compiletime_assert_774();
    fn __SCT__tp_func_skip_task_reaping(arg0: *mut (), arg1: i32) -> i32;
    fn __compiletime_assert_781();
    fn __SCT__tp_func_compact_retry(arg0: *mut (), arg1: i32, arg2: u32, arg3: u32, arg4: i32, arg5: i32, arg6: bool) -> i32;
    fn __compiletime_assert_792();
    fn __compiletime_assert_793();
    fn __compiletime_assert_795();
    fn __compiletime_assert_796();
    fn __compiletime_assert_798();
    fn __compiletime_assert_799();
    fn __compiletime_assert_801();
    fn __compiletime_assert_802();
    fn __compiletime_assert_804();
    fn __compiletime_assert_805();
    fn __compiletime_assert_807();
    fn __compiletime_assert_808();
    fn __compiletime_assert_810();
    fn __compiletime_assert_811();
    fn __compiletime_assert_813();
    fn __compiletime_assert_814();
    fn __compiletime_assert_831();
    fn __compiletime_assert_832();
    fn __compiletime_assert_834();
    fn __compiletime_assert_835();
    fn __compiletime_assert_836();
    fn __compiletime_assert_838();
    fn __compiletime_assert_839();
    fn __compiletime_assert_841();
    fn __compiletime_assert_842();
    fn __compiletime_assert_843();
    fn __compiletime_assert_844();
    fn __compiletime_assert_849();
    fn __compiletime_assert_851();
    fn __compiletime_assert_852();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.frameaddress.p0"]
    fn __slate_intrinsic_2d7c29df8fdf6378(arg0: i32) -> *mut ();
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
    #[link_name = "llvm.returnaddress.p0"]
    fn __slate_intrinsic_ded315013ea6523(arg0: i32) -> *mut ();
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_e1c548fa55baa574(arg0: i64) -> bool;
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
struct lock_class_key {
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


fn list_is_last(list: *const list_head, head: *const list_head) -> i32 {
((unsafe { (*list).next }) == (head as *mut list_head)) as i32
}

// list_entry - get the struct for this entry
// @ptr:	the &struct list_head pointer.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
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
fn hlist_empty(h: *const hlist_head) -> i32 {
let mut __v0: *mut hlist_node;
loop {
if !(__native_word__11!() || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_14() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*h).first) as *const *mut hlist_node) };
__v0.is_null() as i32
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
fn arch_atomic_inc(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock incl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
}


#[inline(always)]
fn arch_atomic_dec_and_test(mut v: *mut __SlateRecord_types_h_190) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock decl ({0:r})\nsete {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), lateout(reg_byte) __t0, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
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
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE__2!((*v).counter)
}

#[inline(always)]
fn arch_atomic64_sub_and_test(mut i: i64, mut v: *mut __SlateRecord_types_h_197) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock subq {2:r}, ({0:r})\nsete {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), lateout(reg_byte) __t0, in(reg) i as u64, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

#[inline(always)]
fn arch_atomic64_inc(mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock incq ({0:r})\n# {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[inline(always)]
fn raw_atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_add(i, v)
}

#[inline(always)]
fn raw_atomic_sub_return(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_sub_return!(i, v)
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
fn raw_atomic_dec_return(v: *mut __SlateRecord_types_h_190) -> i32 {
raw_atomic_sub_return(1i32, v)
}

#[inline(always)]
fn raw_atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
arch_atomic_dec_and_test(v)
}

#[inline(always)]
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_67 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_207 {
}


#[inline(always)]
fn raw_atomic64_inc(v: *mut __SlateRecord_types_h_197) {
arch_atomic64_inc(v);
}

#[inline(always)]
fn raw_atomic64_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
arch_atomic64_sub_and_test(i, v)
}

#[inline(always)]
fn raw_atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
raw_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
raw_atomic64_inc(v);
}

#[inline(always)]
fn raw_atomic_long_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
raw_atomic64_sub_and_test(i, v)
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
core::arch::asm!("37: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 37b - ., 3; .popsection\n# {0}", const 37 as i32, options(nostack, att_syntax));
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
core::arch::asm!("38: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 38b - ., 4; .popsection\n# {0}", const 38 as i32, options(nostack, att_syntax));
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
struct io_bitmap {
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
fn atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_read(v)
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
fn atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec_and_test(v)
}

#[inline(always)]
fn atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_read(v)
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
fn atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_inc(v);
}

#[inline(always)]
fn atomic_long_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_sub_and_test(i, v)
}

#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
}

#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
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
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
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

#[inline(always)]
fn arch_static_branch__const4(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym cpusets_enabled_key, const 0 as u8, label {
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
struct futex_private_hash {
}


#[inline(always)]
fn arch_static_branch__const5(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mark_victim, const 0 as u8, label {
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
fn arch_static_branch__const6(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_wake_reaper, const 0 as u8, label {
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
fn arch_static_branch__const7(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_start_task_reaping, const 0 as u8, label {
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
fn arch_static_branch__const8(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_finish_task_reaping, const 0 as u8, label {
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
fn arch_static_branch__const9(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_skip_task_reaping, const 0 as u8, label {
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_false {
    key: static_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits102U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits103U0,
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


#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[inline(always)]
fn arch_set_bit(mut nr: i64, mut addr: *mut u64) {
if unsafe { __slate_intrinsic_e1c548fa55baa574(nr) } {
unsafe {
core::arch::asm!("lock orb {1:l},({0:r})", in(reg) ((addr as *const u8).wrapping_offset((nr >> (3 as i32)) as isize) as *mut ()) as *mut u8, in(reg) ((1 as i32) << (nr & ((7 as i32) as i64))) as u32, options(nostack, att_syntax));
}
} else {
unsafe {
core::arch::asm!("lock  btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[inline(always)]
fn arch_clear_bit(mut nr: i64, mut addr: *mut u64) {
if unsafe { __slate_intrinsic_e1c548fa55baa574(nr) } {
unsafe {
core::arch::asm!("lock andb {1:l},({0:r})", in(reg) ((addr as *const u8).wrapping_offset((nr >> (3 as i32)) as isize) as *mut ()) as *mut u8, in(reg) !((1 as i32) << (nr & ((7 as i32) as i64))) as u32, options(nostack, att_syntax));
}
} else {
unsafe {
core::arch::asm!("lock  btrq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
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

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
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
fn set_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_set_bit(nr, addr);
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
fn clear_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_clear_bit(nr, addr);
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
fn test_and_set_bit(nr: i64, addr: *mut u64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_set_bit(nr, addr)
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[inline(always)]
fn find_next_bit(addr: *const u64, size: u64, offset: u64) -> u64 {
if small_const_nbits__2!(size) {
let mut val: u64 = 0;
if unlikely!(offset >= size) != 0 {
return size;
}
val = (unsafe { *addr }) & GENMASK!(size.wrapping_sub(1), offset);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_next_bit(addr, size, offset) }
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
fn bitmap_subset(src1: *const u64, src2: *const u64, nbits: u32) -> bool {
if small_const_nbits!(nbits) {
return (unsafe { *src1 }) & !unsafe { *src2 } & BITMAP_LAST_WORD_MASK!(nbits) == 0;
} else {
return unsafe { __bitmap_subset(src1, src2, nbits) };
}
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


#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
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
static mut __UNIQUE_ID_addressable___SCK__might_resched_41: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

// nodemask_pr_args - printf args to output a nodemask
// @maskp: nodemask to be printed
//
// Can be used to provide arguments for '%*pb[l]' when printing a nodemask.
#[inline(always)]
fn __nodemask_pr_numnodes(m: *const __SlateRecord_nodemask_types_h_17) -> u32 {
if !m.is_null() { unsafe { nr_node_ids } } else { 0 }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct net_device {
}


#[inline(always)]
fn __nodemask_pr_bits(m: *const __SlateRecord_nodemask_types_h_17) -> *const u64 {
if !m.is_null() { unsafe { (*m).bits.as_ptr() as *const u64 } } else { core::ptr::null::<u64>() }
}

// The inline keyword gives the compiler room to decide to inline, or
// not inline a function as it sees best.  However, as these functions
// are called in both __init and non-__init functions, if they are not
// inlined we will end up with a section mismatch error (of the type of
// freeable items not being freed).  So we must use __always_inline here
// to fix the problem.  If other functions in the future also end up in
// this situation they will also need to be annotated as __always_inline
// No static inline type checking - see Subtlety (1) above.
#[inline(always)]
fn __nodes_subset(src1p: *const __SlateRecord_nodemask_types_h_17, src2p: *const __SlateRecord_nodemask_types_h_17, nbits: u32) -> bool {
bitmap_subset(unsafe { (*src1p).bits.as_ptr() as *const u64 }, unsafe { (*src2p).bits.as_ptr() as *const u64 }, nbits)
}

#[inline(always)]
fn __first_node(srcp: *const __SlateRecord_nodemask_types_h_17) -> u32 {
let mut __v0: u64;
let __UNIQUE_ID_x__45: i32 = 64i32;
let __UNIQUE_ID_y__46: u64 = find_first_bit(unsafe { (*srcp).bits.as_ptr() as *const u64 }, 64u64);
loop {
if __types_ok__3!(__UNIQUE_ID_x__45, __UNIQUE_ID_y__46) == 0 {
unsafe { __compiletime_assert_47() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if (__UNIQUE_ID_x__45 as u64) < __UNIQUE_ID_y__46 { __UNIQUE_ID_x__45 as u64 } else { __UNIQUE_ID_y__46 };
__v0 as u32
}

#[inline(always)]
fn __next_node(n: i32, srcp: *const __SlateRecord_nodemask_types_h_17) -> u32 {
let mut __v0: u64;
let __UNIQUE_ID_x__48: i32 = 64i32;
let __UNIQUE_ID_y__49: u64 = find_next_bit(unsafe { (*srcp).bits.as_ptr() as *const u64 }, 64u64, n.wrapping_add(1) as u64);
loop {
if __types_ok__3!(__UNIQUE_ID_x__48, __UNIQUE_ID_y__49) == 0 {
unsafe { __compiletime_assert_50() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if (__UNIQUE_ID_x__48 as u64) < __UNIQUE_ID_y__49 { __UNIQUE_ID_x__48 as u64 } else { __UNIQUE_ID_y__49 };
__v0 as u32
}

#[inline(always)]
fn cpu_max_bits_warn(cpu: u32, bits: u32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
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
test_bit__6!(cpumask_check(cpu as u32), unsafe { (*cpumask).bits.as_ptr() as *const u64 }) != 0
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


// cpumask_of - the cpumask containing just a given cpu
// @cpu: the cpu (<= nr_cpu_ids)
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
}


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_54() };
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
struct fred_cs {
    __slate_bits_0: __slate_bits::__SlateBits140U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fred_ss {
    __slate_bits_0: __slate_bits::__SlateBits141U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


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
    __slate_bits_0: __slate_bits::__SlateBits122U0,
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


// uses I/O bitmap
// 32-bit address space on 64 bits
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


fn clear_ti_thread_flag(ti: *mut thread_info, flag: i32) {
clear_bit(flag as i64, (unsafe { core::ptr::addr_of_mut!((*ti).flags) }) as *mut u64);
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


fn test_and_set_ti_thread_flag(ti: *mut thread_info, flag: i32) -> i32 {
test_and_set_bit(flag as i64, (unsafe { core::ptr::addr_of_mut!((*ti).flags) }) as *mut u64) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cfs_rq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_struct {
}


fn test_ti_thread_flag(ti: *mut thread_info, flag: i32) -> i32 {
test_bit__2!(flag, unsafe { core::ptr::addr_of_mut!((*ti).flags) })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_task {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct nameidata {
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
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_preempt_h_497 {
    lock: *mut (),
}


#[inline(always)]
fn class_preempt_notrace_destructor(_T: *mut __SlateRecord_preempt_h_497) {
loop {
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if unlikely!(__preempt_count_dec_and_test()) != 0 {
loop {
unsafe {
core::arch::asm!("call __SCT__preempt_schedule_notrace", options(att_syntax));
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

#[repr(C)]
#[derive(Clone, Copy)]
struct pipe_inode_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_node {
}


#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_notrace_59: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule_notrace) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
}


#[inline(always)]
fn class_preempt_notrace_constructor() -> __SlateRecord_preempt_h_497 {
let _t: __SlateRecord_preempt_h_497 = __SlateRecord_preempt_h_497 { lock: 1i32 as *mut () };
loop {
__preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
return _t;
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_node {
    next: *mut llist_node,
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
struct gendisk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
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
union sigval {
    sival_int: i32,
    sival_ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_enode {
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
struct __SlateRecord_maple_tree_h_162 {
    node: *mut maple_node,
    max: u64,
    mt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
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
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
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
struct sigaction {
    sa_handler: Option<unsafe extern "C" fn(i32)>,
    sa_flags: u64,
    sa_restorer: Option<unsafe extern "C" fn()>,
    sa_mask: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct slab_sheaf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct k_sigaction {
    sa: sigaction,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_print_flags {
    mask: u64,
    name: *const u8,
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
struct tracepoint_ext {
    regfunc: Option<unsafe extern "C" fn() -> i32>,
    unregfunc: Option<unsafe extern "C" fn()>,
    __slate_bits_0: __slate_bits::__SlateBits234U0,
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
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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

#[inline(always)]
fn spin_unlock_irq(lock: *mut spinlock) {
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
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
struct irq_work {
    node: __call_single_node,
    func: Option<unsafe extern "C" fn(*mut irq_work)>,
    irqwait: rcuwait,
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
struct work_struct {
    data: __SlateRecord_types_h_197,
    entry: list_head,
    func: Option<unsafe extern "C" fn(*mut work_struct)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount {
    sequence: u32,
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
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
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
struct __SlateRecord_mm_types_h_1062 {
    rb: rb_node,
    rb_subtree_last: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits299U0,
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
    __slate_bits_0: __slate_bits::__SlateBits69U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits69U1,
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
    __slate_bits_2: __slate_bits::__SlateBits69U2,
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
// Getting shut down
// Allocating memory to free memory. See memalloc_noreclaim_save()
// Kernel thread cloned from userspace thread
// This thread should not be frozen
// I am kcompactd
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
fn test_and_set_tsk_thread_flag(tsk: *mut task_struct, flag: i32) -> i32 {
test_and_set_ti_thread_flag(task_thread_info!(tsk), flag)
}

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
struct memory_tier {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
}


fn rcu_read_lock_held() -> i32 {
1
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
#[inline(always)]
fn rcu_read_lock() {
unsafe { __rcu_read_lock() };
__acquire_shared!();
rcu_lock_acquire!();
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
}

fn rcu_read_unlock() {
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
rcu_lock_release!(); // Keep acq info for rls diags.
__release_shared!();
unsafe { __rcu_read_unlock() };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rcupdate_h_1216 {
    lock: *mut (),
}


#[inline(always)]
fn class_rcu_destructor(_T: *mut __SlateRecord_rcupdate_h_1216) {
rcu_read_unlock();
}

#[inline(always)]
fn class_rcu_constructor() -> __SlateRecord_rcupdate_h_1216 {
let _t: __SlateRecord_rcupdate_h_1216 = __SlateRecord_rcupdate_h_1216 { lock: 1i32 as *mut () };
rcu_read_lock();
return _t;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


fn sigismember(set: *mut __SlateRecord_signal_h_25, _sig: i32) -> i32 {
let sig: u64 = _sig.wrapping_sub(1) as u64;
if 1i32 == 1 {
return (1 & (unsafe { *((*set).sig.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) }) >> sig) as i32;
} else {
return (1 & (unsafe { *((*set).sig.as_mut_ptr() as *mut u64).wrapping_offset((sig / 64) as isize) }) >> sig % 64) as i32;
}
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


fn __refcount_dec_and_test(r: *mut refcount_struct, oldp: *mut i32) -> bool {
__refcount_sub_and_test(1i32, r, oldp)
}

fn refcount_dec_and_test(r: *mut refcount_struct) -> bool {
__refcount_dec_and_test(r, core::ptr::null_mut::<i32>())
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
struct kref {
    refcount: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


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
struct maple_arange_64 {
    parent: *mut maple_pnode,
    pivot: [u64; 9],
    slot: [*mut (); 10],
    gap: [u64; 10],
    meta: maple_metadata,
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
struct maple_tree {
    __slate_anon_0: __SlateRecord_maple_tree_h_261,
    ma_flags: u32,
    ma_root: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_node {
    __slate_anon_0: __SlateRecord_maple_tree_h_324,
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
// mas_for_each_rev() - Iterate over a range of the maple tree in reverse order.
// @__mas: Maple Tree operation state (maple_state)
// @__entry: Entry retrieved from the tree
// @__min: minimum index to retrieve from the tree
//
// When returned, mas->index and mas->last will hold the entire range for the
// entry.
//
// Note: may return the zero entry.
#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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


// explicit shadow
// wait_event_freezable - sleep (or freeze) until a condition gets true
// @wq_head: the waitqueue to wait on
// @condition: a C expression for the event to wait for
//
// The process is put to sleep (TASK_INTERRUPTIBLE -- so as not to contribute
// to system load) until the @condition evaluates to true. The
// @condition is checked each time the waitqueue @wq_head is woken up.
//
// wake_up() has to be called after changing any variable that could
// change the result of the wait condition.
// wait_event_interruptible_timeout - sleep until a condition gets true or a timeout elapses
// @wq_head: the waitqueue to wait on
// @condition: a C expression for the event to wait for
// @timeout: timeout, in jiffies
//
// The process is put to sleep (TASK_INTERRUPTIBLE) until the
// @condition evaluates to true or a signal is received.
// The @condition is checked each time the waitqueue @wq_head is woken up.
//
// wake_up() has to be called after changing any variable that could
// change the result of the wait condition.
//
// Returns:
// 0 if the @condition evaluated to %false after the @timeout elapsed,
// 1 if the @condition evaluated to %true after the @timeout elapsed,
// the remaining jiffies (at least 1) if the @condition evaluated
// to %true before the @timeout elapsed, or -%ERESTARTSYS if it was
// interrupted by a signal.
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
struct tty_struct {
}


fn debug_show_all_locks() {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
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
struct srcu_struct {
    srcu_ctrp: *mut srcu_ctr,
    sda: *mut srcu_data,
    srcu_reader_flavor: u8,
    dep_map: lockdep_map,
    srcu_sup: *mut srcu_usage,
}


fn __srcu_read_lock_fast(ssp: *mut srcu_struct) -> *mut srcu_ctr {
let mut scp: *mut srcu_ctr = unsafe { core::mem::zeroed() };
let mut __v0: *mut srcu_ctr;
loop {
if !(__native_word__10!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_216() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*ssp).srcu_ctrp) as *const *mut srcu_ctr) };
scp = __v0;
if IS_ENABLED!() == 0 {
loop {
__verify_pcpu_ptr__4!();
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
2 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
4 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
8 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
} else {
let mut __v13: *mut __SlateRecord_types_h_197;
__verify_pcpu_ptr__3!();
let mut __v15: *mut __SlateRecord_types_h_197;
let mut tcp_ptr__: u64 = 0;
let mut __v16: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v16 = pfo_val__;
tcp_ptr__ = __v16;
let __v18: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!((*scp).srcu_locks) }) as u64);
tcp_ptr__ = __v18;
__v15 = tcp_ptr__ as *mut __SlateRecord_types_h_197;
__v13 = __v15;
atomic_long_inc(__v13);
}
// Y, and implicit RCU reader.
// Y, and implicit RCU reader.
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
// Avoid leaking the critical section.
__acquire_shared!();
scp
}

fn __srcu_read_unlock_fast(ssp: *mut srcu_struct, mut scp: *mut srcu_ctr) {
__release_shared!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
// Avoid leaking the critical section.
if IS_ENABLED!() == 0 {
loop {
__verify_pcpu_ptr__4!();
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
2 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
4 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
8 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
} else {
let mut __v12: *mut __SlateRecord_types_h_197;
__verify_pcpu_ptr__3!();
let mut __v14: *mut __SlateRecord_types_h_197;
let mut tcp_ptr__: u64 = 0;
let mut __v15: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v15 = pfo_val__;
tcp_ptr__ = __v15;
let __v17: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!((*scp).srcu_unlocks) }) as u64);
tcp_ptr__ = __v17;
__v14 = tcp_ptr__ as *mut __SlateRecord_types_h_197;
__v12 = __v14;
atomic_long_inc(__v12);
}
// Z, and implicit RCU reader.
// Z, and implicit RCU reader.
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


fn srcu_check_read_flavor(ssp: *mut srcu_struct, read_flavor: i32) {
if IS_ENABLED!() != 0 {
unsafe { __srcu_check_read_flavor(ssp, read_flavor) };
}
}

// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
fn srcu_read_lock_fast_notrace(ssp: *mut srcu_struct) -> *mut srcu_ctr {
srcu_check_read_flavor(ssp, 4i32);
let retval: *mut srcu_ctr = __srcu_read_lock_fast(ssp);
retval
}

fn srcu_read_unlock_fast_notrace(ssp: *mut srcu_struct, scp: *mut srcu_ctr) {
srcu_check_read_flavor(ssp, 4i32);
__srcu_read_unlock_fast(ssp, scp);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcu_h_645 {
    lock: *mut srcu_struct,
    scp: *mut srcu_ctr,
}


#[inline(always)]
fn class_srcu_fast_notrace_destructor(_T: *mut __SlateRecord_srcu_h_645) {
srcu_read_unlock_fast_notrace(unsafe { (*_T).lock }, unsafe { (*_T).scp });
}

#[inline(always)]
fn class_srcu_fast_notrace_constructor(l: *mut srcu_struct) -> __SlateRecord_srcu_h_645 {
let mut _t: __SlateRecord_srcu_h_645 = {
let mut __t0: __SlateRecord_srcu_h_645 = unsafe { core::mem::zeroed() };
__t0.lock = l;
__t0
};
let _T: *mut __SlateRecord_srcu_h_645 = core::ptr::addr_of_mut!(_t);
unsafe {
(*_T).scp = srcu_read_lock_fast_notrace((*_T).lock);
}
return _t;
}

#[inline(always)]
fn __class_srcu_fast_notrace_cleanup_ctx(_T: *mut *mut __SlateRecord_srcu_h_645) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct notifier_block {
    notifier_call: Option<unsafe extern "C" fn(*mut notifier_block, u64, *mut ()) -> i32>,
    next: *mut notifier_block,
    priority: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blocking_notifier_head {
    rwsem: rw_semaphore,
    head: *mut notifier_block,
}


// Suits me
// Don't call further
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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


fn percpu_counter_sum_positive(fbc: *mut percpu_counter) -> i64 {
let ret: i64 = unsafe { __percpu_counter_sum(fbc) };
if ret < 0 { 0 } else { ret }
}

fn percpu_counter_read_positive(fbc: *mut percpu_counter) -> i64 {
// Prevent reloads of fbc->count
let mut ret: i64 = 0;
let mut __v0: i64;
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_232() };
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
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits454U0,
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
struct fserror_event {
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
struct mm_struct {
    __slate_anon_0: __SlateRecord_mm_types_h_1176,
    flexible_array: [u8; 0],
}


// Clear the access bit
// Clear the dirty bit
// mm is of no interest for the OOM killer
// mm is unstable for copy_from_user
// mm was queued for oom_reaper
// MMF_HAS_PINNED: Whether this mm has pinned any pages.  This can be either
// replaced in the future by mm.pinned_vm when it becomes stable, or grow into
// a counter on its own. We're aggresive on this bit for now: even if the
// pinned pages were unpinned later on, we'll still keep this bit set for the
// lifecycle of this mm, just for simplicity.
// FOLL_PIN has run, never cleared
#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct zswap_lruvec_state {
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
struct free_area {
    free_list: [list_head; 4],
    nr_free: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct request_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
}


#[inline(always)]
fn vmstat_item_in_bytes(idx: i32) -> bool {
// Global and per-node slab counters track slab pages.
// It's expected that changes are multiples of PAGE_SIZE.
// Internally values are stored in pages.
//
// Per-memcg and per-lruvec counters track memory, consumed
// by individual slab objects. These counters are actually
// byte-precise.
idx == 5 || idx == 6
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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


// zone_idx() returns 0 for the ZONE_DMA zone, 1 for the ZONE_NORMAL zone, etc.
fn zone_to_nid(zone: *const zone) -> i32 {
unsafe { (*zone).node }
}

fn zonelist_zone(zoneref: *mut zoneref) -> *mut zone {
unsafe { (*zoneref).zone }
}

fn zonelist_zone_idx(zoneref: *const zoneref) -> i32 {
unsafe { (*zoneref).zone_idx }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
}


fn zonelist_node_idx(zoneref: *const zoneref) -> i32 {
zone_to_nid((unsafe { (*zoneref).zone }) as *const zone)
}

#[inline(always)]
fn next_zones_zonelist(z: *mut zoneref, highest_zoneidx: u32, nodes: *const __SlateRecord_nodemask_types_h_17) -> *mut zoneref {
let __v0: bool = nodes.is_null() && (zonelist_zone_idx(z as *const zoneref) as u32) <= (highest_zoneidx as u32);
if __v0 {
return z;
}
unsafe { __next_zones_zonelist(z, highest_zoneidx, nodes) }
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


fn first_zones_zonelist(zonelist: *mut zonelist, highest_zoneidx: u32, nodes: *const __SlateRecord_nodemask_types_h_17) -> *mut zoneref {
next_zones_zonelist(unsafe { (*zonelist)._zonerefs.as_mut_ptr() as *mut zoneref }, highest_zoneidx, nodes)
}

// for_each_zone_zonelist_nodemask - helper macro to iterate over valid zones in a zonelist at or below a given zone index and within a nodemask
// @zone: The current zone in the iterator
// @z: The current pointer within zonelist->_zonerefs being iterated
// @zlist: The zonelist being iterated
// @highidx: The zone index of the highest zone to return
// @nodemask: Nodemask allowed by the allocator
//
// This iterator iterates though all zones at or below a given zone index and
// within a given nodemask
// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
fn gfp_zone(flags: u32) -> u32 {
let mut z: u32 = 0;
let bit: i32 = (flags & ((BIT!(0i32) as u32) | (BIT!(1i32) as u32) | (BIT!(2i32) as u32) | (BIT!(3i32) as u32))) as i32;
z = ((2i32 | 0i32 << BIT!(0i32).wrapping_mul(2) | 2i32 << BIT!(1i32).wrapping_mul(2) | 1i32 << BIT!(2i32).wrapping_mul(2) | 2i32 << BIT!(3i32).wrapping_mul(2) | 0i32 << (BIT!(3i32) | BIT!(0i32)).wrapping_mul(2) | 3i32 << (BIT!(3i32) | BIT!(1i32)).wrapping_mul(2) | 1i32 << (BIT!(3i32) | BIT!(2i32)).wrapping_mul(2)) >> bit.wrapping_mul(2) & 3i32) as u32;
VM_BUG_ON!();
z
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


fn mmgrab(mm: *mut mm_struct) {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.__slate_anon_0.mm_count) });
}

fn mmdrop(mm: *mut mm_struct) {
// The implicit full barrier implied by atomic_dec_and_test() is
// required by the membarrier system call before returning to
// user-space, after storing to rq->curr.
if unlikely!(atomic_dec_and_test(unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.__slate_anon_0.mm_count) })) != 0 {
unsafe { __mmdrop(mm) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
}


fn in_vfork(tsk: *mut task_struct) -> bool {
let mut ret: bool = false;
// need RCU to access ->real_parent if CLONE_VM was used along with
// CLONE_PARENT.
//
// We check real_parent->mm == tsk->mm because CLONE_VFORK does not
// imply CLONE_VM
//
// CLONE_VFORK can be used with CLONE_PARENT/CLONE_THREAD and thus
// ->real_parent is not necessarily the task doing vfork(), so in
// theory we can't rely on task_lock() if we want to dereference it.
//
// And in this case we can't trust the real_parent->mm == tsk->mm
// check, it can be false negative. But we do not care, if init or
// another oom-unkillable task does this it should blame itself.
rcu_read_lock();
let __v0: bool = if !unsafe { (*tsk).vfork_done }.is_null() {
let mut __v1: *mut task_struct;
let mut __v2: *mut task_struct;
loop {
if !(__native_word__26!() || ((const { core::mem::size_of::<*mut task_struct>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_260() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*tsk).real_parent) as *const *mut task_struct) };
let __UNIQUE_ID_rcu_259: *mut task_struct = __v2;
loop {
let __v5: bool = if false {
let __v6: bool = false || rcu_read_lock_held() != 0;
!__v6
} else {
false
};
if !__v5 {
break;
}
}
__v1 = __UNIQUE_ID_rcu_259;
(unsafe { (*__v1).mm }) == unsafe { (*tsk).mm }
} else {
false
};
ret = __v0;
rcu_read_unlock();
ret
}

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
fn mmap_read_lock_killable(mm: *mut mm_struct) -> i32 {
let mut ret: i32 = 0;
__mmap_lock_trace_start_locking(mm, false);
ret = unsafe { down_read_killable(core::ptr::addr_of_mut!((*mm).__slate_anon_0.mmap_lock)) };
__mmap_lock_trace_acquire_returned(mm, false, ret == 0);
ret
}

fn mmap_read_trylock(mm: *mut mm_struct) -> bool {
let mut ret: bool = false;
__mmap_lock_trace_start_locking(mm, false);
ret = (unsafe { down_read_trylock(core::ptr::addr_of_mut!((*mm).__slate_anon_0.mmap_lock)) }) != 0;
__mmap_lock_trace_acquire_returned(mm, false, ret);
ret
}

fn mmap_read_unlock(mm: *mut mm_struct) {
__mmap_lock_trace_released(mm, false);
unsafe { up_read(core::ptr::addr_of_mut!((*mm).__slate_anon_0.mmap_lock)) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


fn get_task_struct(t: *mut task_struct) -> *mut task_struct {
refcount_inc(unsafe { core::ptr::addr_of_mut!((*t).usage) });
t
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


fn put_task_struct(t: *mut task_struct) {
if !refcount_dec_and_test(unsafe { core::ptr::addr_of_mut!((*t).usage) }) {
return;
}
// Under PREEMPT_RT, we can't call __put_task_struct
// in atomic context because it will indirectly
// acquire sleeping locks. The same is true if the
// current process has a mutex enqueued (blocked on
// a PI chain).
//
// In !RT, it is always safe to call __put_task_struct().
// Though, in order to simplify the code, resort to the
// deferred call too.
//
// call_rcu() will schedule __put_task_struct_rcu_cb()
// to be called in process context.
//
// __put_task_struct() is called when
// refcount_dec_and_test(&t->usage) succeeds.
//
// This means that it can't "conflict" with
// put_task_struct_rcu_user() which abuses ->rcu the same
// way; rcu_users has a reference so task->usage can't be
// zero after rcu_users 1 -> 0 transition.
//
// delayed_free_task() also uses ->rcu, but it is only called
// when it fails to fork a process. Therefore, there is no
// way it can conflict with __put_task_struct().
unsafe { call_rcu(core::ptr::addr_of_mut!((*t).rcu), unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut callback_head)>>(__put_task_struct_rcu_cb as *const ()) }) };
}

fn task_lock(p: *mut task_struct) {
spin_lock(unsafe { core::ptr::addr_of_mut!((*p).alloc_lock) });
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


fn task_unlock(p: *mut task_struct) {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*p).alloc_lock) });
}

fn __kuid_val(uid: __SlateRecord_uidgid_types_h_9) -> u32 {
uid.val
}

fn from_kuid(to: *mut user_namespace, kuid: __SlateRecord_uidgid_types_h_9) -> u32 {
__kuid_val(kuid)
}

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
struct ctl_node {
    node: rb_node,
    header: *mut ctl_table_header,
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
struct key_tag {
    rcu: callback_head,
    usage: refcount_struct,
    removed: bool,
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


fn put_cred_many(_cred: *const cred, nr: i32) {
let cred: *mut cred = _cred as *mut cred;
if !cred.is_null() {
if atomic_long_sub_and_test(nr as i64, unsafe { core::ptr::addr_of_mut!((*cred).usage) }) {
unsafe { __put_cred(cred) };
}
}
}

fn put_cred(cred: *const cred) {
put_cred_many(cred, 1i32);
}

// current_cred - Access the current task's subjective credentials
//
// Access the subjective credentials of the current task.  RCU-safe,
// since nobody else can modify it.
// __task_cred - Access a task's objective credentials
// @task: The task to query
//
// Access the objective credentials of a task.  The caller must hold the RCU
// readlock.
//
// The result of this function should not be passed directly to get_cred();
// rather get_task_cred() should be used instead.
#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_taskset {
}


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


fn task_pid_nr(tsk: *mut task_struct) -> i32 {
unsafe { (*tsk).pid }
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


fn task_tgid_nr(tsk: *mut task_struct) -> i32 {
unsafe { (*tsk).tgid }
}

fn is_global_init(tsk: *mut task_struct) -> i32 {
(task_tgid_nr(tsk) == 1) as i32
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
struct wake_irq {
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
struct dev_pm_qos {
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
    __slate_bits_0: __slate_bits::__SlateBits302U0,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


fn __fatal_signal_pending(p: *mut task_struct) -> i32 {
unlikely__2!(sigismember(unsafe { core::ptr::addr_of_mut!((*p).pending.signal) }, 9i32)) as i32
}

fn fatal_signal_pending(p: *mut task_struct) -> i32 {
let __v0: bool = task_sigpending(p) != 0 && __fatal_signal_pending(p) != 0;
__v0 as i32
}

fn thread_group_leader(p: *mut task_struct) -> bool {
(unsafe { (*p).exit_signal }) >= 0
}

fn same_thread_group(p1: *mut task_struct, p2: *mut task_struct) -> bool {
(unsafe { (*p1).signal }) == unsafe { (*p2).signal }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
}


fn thread_group_empty(p: *mut task_struct) -> i32 {
let __v0: bool = thread_group_leader(p) && list_is_last((unsafe { core::ptr::addr_of_mut!((*p).thread_node) }) as *const list_head, (unsafe { core::ptr::addr_of_mut!((*(*p).signal).thread_head) }) as *const list_head) != 0;
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

#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits565U0,
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
struct shrink_control {
    gfp_mask: u32,
    nid: i32,
    nr_to_scan: u64,
    nr_scanned: u64,
    memcg: *mut mem_cgroup,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
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
struct cacheinfo {
    id: u32,
    r#type: u32,
    level: u32,
    coherency_line_size: u32,
    number_of_sets: u32,
    ways_of_associativity: u32,
    physical_line_partition: u32,
    size: u32,
    shared_cpu_map: cpumask,
    attributes: u32,
    fw_token: *mut (),
    disable_sysfs: bool,
    r#priv: *mut (),
}


fn totalram_pages() -> u64 {
atomic_long_read((unsafe { core::ptr::addr_of_mut!(_totalram_pages) }) as *const __SlateRecord_types_h_197) as u64
}

// Despite the naming, these are FLAGS not bits.
// Bits set in the VMA until the stack is in its final location
// VMA basic access permission flags
// These flags can be updated atomically via VMA/mmap read lock.
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


#[repr(C)]
#[derive(Clone, Copy)]
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


fn mm_flags_test(flag: i32, mm: *const mm_struct) -> bool {
test_bit__5!(flag, unsafe { (*core::ptr::addr_of!((*mm).__slate_anon_0.flags)).__mm_flags.as_ptr() as *const u64 }) != 0
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


fn mm_flags_test_and_set(flag: i32, mm: *mut mm_struct) -> bool {
test_and_set_bit(flag as i64, ACCESS_PRIVATE!(unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.flags) }, __mm_flags) as *mut u64)
}

fn mm_flags_set(flag: i32, mm: *mut mm_struct) {
set_bit(flag as i64, ACCESS_PRIVATE!(unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.flags) }, __mm_flags) as *mut u64);
}

// Helper macro which bitwise-or combines the specified input flags into a
// vma_flags_t bitmap value. E.g.:
//
// vma_flags_t flags = mk_vma_flags(VMA_IO_BIT, VMA_PFNMAP_BIT,
//              VMA_DONTEXPAND_BIT, VMA_DONTDUMP_BIT);
//
// The compiler cleverly optimises away all of the work and this ends up being
// equivalent to aggregating the values manually.
// Test whether any specified VMA flag is set, e.g.:
//
// if (vma_flags_test_any(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
// Test whether ALL specified VMA flags are set, e.g.:
//
// if (vma_flags_test_all(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
fn vma_is_anonymous(vma: *const vm_area_struct) -> bool {
unsafe { (*vma).vm_ops }.is_null()
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
struct swap_cluster_info {
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
struct swap_ops {
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
struct trace_buffer {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ring_buffer_iter {
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
    __slate_bits_0: __slate_bits::__SlateBits880U0,
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
struct __SlateRecord_perf_event_h_1322 {
    __slate_bits_0: __slate_bits::__SlateBits882U0,
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
struct ftrace_ops {
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
    __slate_bits_0: __slate_bits::__SlateBits910U0,
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
    __slate_bits_0: __slate_bits::__SlateBits913U0,
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
    __slate_bits_0: __slate_bits::__SlateBits946U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6849 {
    attach_type: u32,
    target_obj_id: u32,
    target_btf_id: u32,
    __slate_bits_0: __slate_bits::__SlateBits947U0,
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
    __slate_bits_0: __slate_bits::__SlateBits961U0,
    ids: u64,
    addrs: u64,
    cookies: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6926 {
    r#type: u32,
    __slate_bits_0: __slate_bits::__SlateBits962U0,
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
    __slate_bits_0: __slate_bits::__SlateBits966U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6951 {
    config: u64,
    r#type: u32,
    __slate_bits_0: __slate_bits::__SlateBits967U0,
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
union __SlateRecord_btf_h_64 {
    size: u32,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_state {
    s_incoredqs: u32,
    s_state: [qc_type_state; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf {
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
struct bpf_verifier_log {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_verifier_env {
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
union __SlateRecord_bpf_h_256 {
    kptr: btf_field_kptr,
    graph_root: btf_field_graph_root,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct path {
    mnt: *mut vfsmount,
    dentry: *mut dentry,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_reg_state {
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
unsafe { __compiletime_assert_419() };
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
union __SlateRecord_bpf_h_1705 {
    tail_call: __SlateRecord_bpf_h_1706,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1706 {
    map: *mut bpf_map,
    key: u32,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct inode_state_flags {
    __state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_kfunc_desc_tab {
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1964 {
    rcu: callback_head,
    work: work_struct,
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
union __SlateRecord_bpf_h_2532 {
    cgroup_storage: [*mut bpf_cgroup_storage; 2],
    bpf_cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_cgroup_storage {
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_76 {
    next: *mut perf_raw_frag,
    pad: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits713U0,
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
struct perf_buffer {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct event_filter {
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
union __SlateRecord_perf_event_h_1156 {
    flags: u64,
    aux_flags: u64,
    __slate_anon_2: __SlateRecord_perf_event_h_1159,
}


#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_1159 {
    __slate_bits_0: __slate_bits::__SlateBits1092U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_1163 {
    addr: *mut (),
    head: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysfs_ops {
    show: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *const u8, u64) -> i64>,
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
struct trace_array {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct array_buffer {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracer {
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
    __slate_bits_0: __slate_bits::__SlateBits663U0,
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
struct vm_event_state {
    event: [u64; 75],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_trace_events_h_275 {
    __slate_anon_0: __SlateRecord_trace_events_h_276,
    define_fields: Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_trace_events_h_276 {
    name: *const u8,
    size: i32,
    align: i32,
    __slate_bits_0: __slate_bits::__SlateBits1110U0,
    filter_type: i32,
    len: i32,
}


fn count_vm_event(mut item: u32) {
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
2 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
4 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
8 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
}

fn global_node_page_state_pages(item: u32) -> u64 {
let mut x: i64 = atomic_long_read(unsafe { core::ptr::addr_of_mut!(vm_node_stat) as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize) as *const __SlateRecord_types_h_197);
if x < 0 {
x = 0;
}
x as u64
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_trace_events_h_379 {
    name: *const u8,
    tp: *mut tracepoint,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_trace_events_h_390 {
    module: *mut (),
    refcnt: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_subsystem_dir {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct eventfs_inode {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mempolicy_h_54 {
    cpuset_mems_allowed: __SlateRecord_nodemask_types_h_17,
    user_nodemask: __SlateRecord_nodemask_types_h_17,
}


fn global_node_page_state(item: u32) -> u64 {
VM_WARN_ON_ONCE!();
global_node_page_state_pages(item)
}

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
fn get_mm_counter(mm: *mut mm_struct, member: i32) -> u64 {
percpu_counter_read_positive(unsafe { (*mm).__slate_anon_0.rss_stat.as_mut_ptr() as *mut percpu_counter }.wrapping_offset(member as isize)) as u64
}

fn get_mm_counter_sum(mm: *mut mm_struct, member: i32) -> u64 {
percpu_counter_sum_positive(unsafe { (*mm).__slate_anon_0.rss_stat.as_mut_ptr() as *mut percpu_counter }.wrapping_offset(member as isize)) as u64
}

fn get_mm_rss_sum(mm: *mut mm_struct) -> u64 {
get_mm_counter_sum(mm, 0i32).wrapping_add(get_mm_counter_sum(mm, 1i32)).wrapping_add(get_mm_counter_sum(mm, 3i32))
}

fn mm_pgtables_bytes(mm: *const mm_struct) -> u64 {
atomic_long_read(unsafe { core::ptr::addr_of!((*mm).__slate_anon_0.pgtables_bytes) }) as u64
}

#[unsafe(no_mangle)]
extern "C" fn process_shares_mm(p: *const task_struct, mm: *const mm_struct) -> bool {
let mut t: *const task_struct = unsafe { core::mem::zeroed() };
let mut __v0: *const task_struct;
let mut __v1: *mut list_head;
loop {
if !(((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_842() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*p).signal).thread_head)).next) as *const *mut list_head) };
__v0 = ((unsafe { (__v1 as *const u8).wrapping_offset(-(offsetof__6!() as isize)) }) as *mut ()) as *const task_struct;
t = __v0;
while (unsafe { core::ptr::addr_of!((*t).thread_node) }) != ((unsafe { core::ptr::addr_of_mut!((*(*p).signal).thread_head) }) as *const list_head) {
let mut __v4: *mut mm_struct;
loop {
if !(__native_word__36!() || ((const { core::mem::size_of::<*mut mm_struct>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_844() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*t).mm) as *const *mut mm_struct) };
let t_mm: *const mm_struct = __v4 as *const mm_struct;
if !t_mm.is_null() {
return t_mm == mm;
}
let mut __v7: *const task_struct;
let mut __v8: *mut list_head;
loop {
if !(((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_843() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v8 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*t).thread_node.next) as *const *mut list_head) };
__v7 = ((unsafe { (__v8 as *const u8).wrapping_offset(-(offsetof__6!() as isize)) }) as *mut ()) as *const task_struct;
t = __v7;
}
false
}

// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
#[repr(C)]
#[derive(Clone, Copy)]
struct node_barn {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct slub_percpu_sheaves {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct oom_control {
    zonelist: *mut zonelist,
    nodemask: *const __SlateRecord_nodemask_types_h_17,
    memcg: *mut mem_cgroup,
    gfp_mask: u32,
    order: i32,
    totalpages: u64,
    chosen: *mut task_struct,
    chosen_points: i64,
    constraint: u32,
}


fn oom_task_origin(p: *const task_struct) -> bool {
unsafe { (*(*p).signal).oom_flag_origin }
}

fn tsk_is_oom_victim(tsk: *mut task_struct) -> bool {
!unsafe { (*(*tsk).signal).oom_mm }.is_null()
}

#[unsafe(no_mangle)]
extern "C" fn find_lock_task_mm(p: *mut task_struct) -> *mut task_struct {
let mut t: *mut task_struct = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_guard_833: __SlateRecord_rcupdate_h_1216 = class_rcu_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_833) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rcu_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __v1: *mut task_struct;
let mut __v2: *mut list_head;
loop {
if !(((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_834() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*p).signal).thread_head)).next) as *const *mut list_head) };
__v1 = ((unsafe { (__v2 as *const u8).wrapping_offset(-(offsetof__6!() as isize)) }) as *mut ()) as *mut task_struct;
t = __v1;
while (unsafe { core::ptr::addr_of_mut!((*t).thread_node) }) != unsafe { core::ptr::addr_of_mut!((*(*p).signal).thread_head) } {
task_lock(t);
if likely__14!(unsafe { (*t).mm }) != 0 {
return t;
}
task_unlock(t);
let mut __v6: *mut task_struct;
let mut __v7: *mut list_head;
loop {
if !(((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_835() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v7 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*t).thread_node.next) as *const *mut list_head) };
__v6 = ((unsafe { (__v7 as *const u8).wrapping_offset(-(offsetof__6!() as isize)) }) as *mut ()) as *mut task_struct;
t = __v6;
}
core::ptr::null_mut::<task_struct>()
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
struct __SlateRecord_oom_h_69 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_214 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_215 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_19 {
    b: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_20 {
    b: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_21 {
    b: i16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_47 {
    b: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_48 {
    b: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_49 {
    b: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_50 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_51 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_52 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_53 {
    b: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_54 {
    b: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_82 {
    b: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_84 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_85 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_86 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_87 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_88 {
    b: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_89 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_90 {
    b: i16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_124 {
    b: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_140 {
    b: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_156 {
    b: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_172 {
    b: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_195 {
    b: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_196 {
    b: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_197 {
    b: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_198 {
    b: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_199 {
    b: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_h_200 {
    b: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_pid__1 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_oom_score_adj__1 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_zone_idx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_order__1 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_reclaimable {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_available {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_min_wmark {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_no_progress_loops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_wmark_check {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_pid__1_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_total_vm {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_anon_rss {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_file_rss {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_shmem_rss {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_uid {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_pgtables {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_oom_score_adj__1_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_pid__1_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_pid__1_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_pid__1_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_pid__1_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_order__1_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_priority {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_result {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_retries {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_max_retries {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_ret {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_kill_c_730 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_kill_c_1195 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_oom_kill_c_1195_2 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits626U0 {
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
    pub struct __SlateBits565U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits299U0 {
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
    pub struct __SlateBits69U0 {
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
    pub struct __SlateBits69U1 {
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
    pub struct __SlateBits454U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits713U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits758U0 {
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
    pub struct __SlateBits302U0 {
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
    pub struct __SlateBits69U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits103U0 {
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
    pub struct __SlateBits788U0 {
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
    pub struct __SlateBits785U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits788U1 {
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
    pub struct __SlateBits788U2 {
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
    pub struct __SlateBits788U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1078U0 {
        #[bits(1)]
        pub embedded: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1092U0 {
        #[bits(1)]
        pub skip_read: u64,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits866U0 {
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
    pub struct __SlateBits880U0 {
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
    pub struct __SlateBits883U0 {
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
    pub struct __SlateBits882U0 {
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
    pub struct __SlateBits140U0 {
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
    pub struct __SlateBits141U0 {
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
    pub struct __SlateBits978U0 {
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
    pub struct __SlateBits891U0 {
        #[bits(4)]
        pub dst_reg: u8,
        #[bits(4)]
        pub src_reg: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits910U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits913U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits946U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits947U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits961U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits962U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits966U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits967U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1110U0 {
        #[bits(1)]
        pub is_signed: u32,
        #[bits(1)]
        pub needs_test: u32,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits234U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits122U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits663U0 {
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
    pub struct __SlateBits102U0 {
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
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits873U0 {
        #[bits(5)]
        pub type_len: u32,
        #[bits(27)]
        pub time_delta: u32,
    }
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
    __slate_bits_0: __slate_bits::__SlateBits758U0,
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
struct bio_vec {
    bv_page: *mut page,
    bv_len: u32,
    bv_offset: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits788U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits788U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits788U2,
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
    __slate_bits_3: __slate_bits::__SlateBits788U3,
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
struct iovec {
    iov_base: *mut (),
    iov_len: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits785U0,
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
    __slate_bits_0: __slate_bits::__SlateBits626U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


fn memcg_memory_event(memcg: *mut mem_cgroup, event: u32) {
}

fn memcg_memory_event_mm(mm: *mut mm_struct, event: u32) {
}

fn mem_cgroup_put(memcg: *mut mem_cgroup) {
}

fn mem_cgroup_scan_tasks(memcg: *mut mem_cgroup, r#fn: Option<unsafe extern "C" fn(*mut task_struct, *mut ()) -> i32>, arg: *mut ()) {
}

fn mem_cgroup_get_max(memcg: *mut mem_cgroup) -> u64 {
0
}

fn mem_cgroup_print_oom_context(memcg: *mut mem_cgroup, p: *mut task_struct) {
}

fn mem_cgroup_print_oom_meminfo(memcg: *mut mem_cgroup) {
}

fn mem_cgroup_get_oom_group(victim: *mut task_struct, oom_domain: *mut mem_cgroup) -> *mut mem_cgroup {
core::ptr::null_mut::<mem_cgroup>()
}

fn mem_cgroup_print_oom_group(memcg: *mut mem_cgroup) {
}

fn mem_cgroup_show_protected_memory(memcg: *mut mem_cgroup) {
}

fn mem_cgroup_oom_synchronize(wait: bool) -> bool {
false
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


// Declare an exported function that Rust code can call to trigger this
// tracepoint. This function does not include the static branch; that is done
// in Rust to avoid a function call when the tracepoint is disabled.
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
// We have no guarantee that gcc and the linker won't up-align the tracepoint
// structures, so we create an array of pointers that will be used for iteration
// on the tracepoints.
//
// it_func[0] is never NULL because there is at least one element in the array
// when the array itself is non NULL.
// 									 * Annotate the probestub 'CFI_NOSEAL' to stop objtool from		 * requesting the kernel remove the ENDBR, because the only		 * references to the function are in the __tracepoint section,		 * that objtool doesn't scan.
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


#[repr(C)]
#[derive(Clone, Copy)]
struct ring_buffer_event {
    __slate_bits_0: __slate_bits::__SlateBits873U0,
    array: [u32; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_buf {
    buffer: *mut u8,
    size: u64,
    len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_seq {
    seq: seq_buf,
    readpos: u64,
    full: i32,
    buffer: [u8; 8156],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_attr {
    r#type: u32,
    size: u32,
    config: u64,
    __slate_anon_3: __SlateRecord_perf_event_h_410,
    sample_type: u64,
    read_format: u64,
    __slate_bits_0: __slate_bits::__SlateBits866U0,
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
    __slate_bits_0: __slate_bits::__SlateBits883U0,
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
    __slate_bits_0: __slate_bits::__SlateBits891U0,
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
    __slate_bits_0: __slate_bits::__SlateBits978U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_array_item {
    prog: *mut bpf_prog,
    __slate_anon_1: __SlateRecord_bpf_h_2532,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_array {
    rcu: callback_head,
    items: [bpf_prog_array_item; 0],
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
    __slate_bits_0: __slate_bits::__SlateBits1078U0,
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


fn perf_fetch_caller_regs(regs: *mut pt_regs) {
perf_arch_fetch_caller_regs!(regs, (unsafe { __slate_intrinsic_ded315013ea6523(const { ((0 as i32) as u32) as i32 }) }) as u64);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct trace_entry {
    r#type: u16,
    flags: u8,
    preempt_count: u8,
    pid: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_iterator {
    tr: *mut trace_array,
    trace: *mut tracer,
    array_buffer: *mut array_buffer,
    private: *mut (),
    cpu_file: i32,
    mutex: mutex,
    buffer_iter: *mut *mut ring_buffer_iter,
    iter_flags: u64,
    temp: *mut (),
    temp_size: u32,
    fmt: *mut u8,
    fmt_size: u32,
    wait_index: __SlateRecord_types_h_190,
    tmp_seq: trace_seq,
    started: [cpumask; 1],
    closed: bool,
    snapshot: bool,
    seq: trace_seq,
    ent: *mut trace_entry,
    lost_events: u64,
    leftover: i32,
    ent_size: i32,
    cpu: i32,
    ts: u64,
    pos: i64,
    idx: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_functions {
    trace: Option<unsafe extern "C" fn(*mut trace_iterator, i32, *mut trace_event) -> u32>,
    raw: Option<unsafe extern "C" fn(*mut trace_iterator, i32, *mut trace_event) -> u32>,
    hex: Option<unsafe extern "C" fn(*mut trace_iterator, i32, *mut trace_event) -> u32>,
    binary: Option<unsafe extern "C" fn(*mut trace_iterator, i32, *mut trace_event) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event {
    node: hlist_node,
    r#type: i32,
    funcs: *mut trace_event_functions,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_fields {
    r#type: *const u8,
    __slate_anon_1: __SlateRecord_trace_events_h_275,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_class {
    system: *const u8,
    probe: *mut (),
    perf_probe: *mut (),
    reg: Option<unsafe extern "C" fn(*mut trace_event_call, u32, *mut ()) -> i32>,
    fields_array: *mut trace_event_fields,
    get_fields: Option<unsafe extern "C" fn(*mut trace_event_call) -> *mut list_head>,
    fields: list_head,
    raw_init: Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_buffer {
    buffer: *mut trace_buffer,
    event: *mut ring_buffer_event,
    trace_file: *mut trace_event_file,
    entry: *mut (),
    trace_ctx: u32,
    regs: *mut pt_regs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_call {
    list: list_head,
    class: *mut trace_event_class,
    __slate_anon_2: __SlateRecord_trace_events_h_379,
    event: trace_event,
    print_fmt: *mut u8,
    __slate_anon_5: __SlateRecord_trace_events_h_390,
    data: *mut (),
    flags: i32,
    perf_refcount: i32,
    perf_events: *mut hlist_head,
    prog_array: *mut bpf_prog_array,
    perf_perm: Option<unsafe extern "C" fn(*mut trace_event_call, *mut perf_event) -> i32>,
}


fn bpf_prog_array_valid(call: *mut trace_event_call) -> bool {
// This inline function checks whether call->prog_array
// is valid or not. The function is called in various places,
// outside rcu_read_lock/unlock, as a heuristic to speed up execution.
//
// If this function returns true, and later call->prog_array
// becomes false inside rcu_read_lock/unlock region,
// we bail out then. If this function return false,
// there is a risk that we might miss a few events if the checking
// were delayed until inside rcu_read_lock/unlock region and
// call->prog_array happened to become non-NULL then.
//
// Here, READ_ONCE() is used instead of rcu_access_pointer().
// rcu_access_pointer() requires the actual definition of
// "struct bpf_prog_array" while READ_ONCE() only needs
// a declaration of the same type.
let mut __v0: *mut bpf_prog_array;
loop {
if !(__native_word__16!() || ((const { core::mem::size_of::<*mut bpf_prog_array>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_678() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*call).prog_array) as *const *mut bpf_prog_array) };
!__v0.is_null()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_file {
    list: list_head,
    event_call: *mut trace_event_call,
    filter: *mut event_filter,
    ei: *mut eventfs_inode,
    tr: *mut trace_array,
    system: *mut trace_subsystem_dir,
    triggers: list_head,
    flags: u64,
    r#ref: refcount_struct,
    sm_ref: __SlateRecord_types_h_190,
    tm_ref: __SlateRecord_types_h_190,
}


#[inline(always)]
fn trace_trigger_soft_disabled(file: *mut trace_event_file) -> bool {
let eflags: u64 = unsafe { (*file).flags };
if likely!(eflags & 352u64 == 0) != 0 {
return false;
}
if likely__2!(eflags & 128) != 0 {
return false;
}
unsafe { __trace_trigger_soft_disabled(file) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
    entries: *mut desc_struct,
    nr_entries: u32,
    slot: i32,
}


// Init a new mm.  Used on mm copies, like at fork()
// and on mm's that are brand-new, like at execve().
fn cpusets_enabled() -> bool {
static_branch_unlikely__4!(unsafe { core::ptr::addr_of_mut!(cpusets_enabled_key) }) != 0
}

fn __cpuset_zone_allowed(z: *mut zone, gfp_mask: u32) -> bool {
unsafe { cpuset_current_node_allowed(zone_to_nid(z as *const zone), gfp_mask) }
}

fn cpuset_zone_allowed(z: *mut zone, gfp_mask: u32) -> bool {
if cpusets_enabled() {
return __cpuset_zone_allowed(z, gfp_mask);
}
true
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


fn touch_softlockup_watchdog() {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_handle {
    secondary: *mut fwnode_handle,
    ops: *const fwnode_operations,
    dev: *mut device,
    suppliers: list_head,
    consumers: list_head,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_endpoint {
    port: u32,
    id: u32,
    local_fwnode: *const fwnode_handle,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_reference_args {
    fwnode: *mut fwnode_handle,
    nargs: u32,
    args: [u64; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_operations {
    get: Option<unsafe extern "C" fn(*mut fwnode_handle) -> *mut fwnode_handle>,
    put: Option<unsafe extern "C" fn(*mut fwnode_handle)>,
    device_is_available: Option<unsafe extern "C" fn(*const fwnode_handle) -> bool>,
    device_get_match_data: Option<unsafe extern "C" fn(*const fwnode_handle, *const device) -> *const ()>,
    device_dma_supported: Option<unsafe extern "C" fn(*const fwnode_handle) -> bool>,
    device_get_dma_attr: Option<unsafe extern "C" fn(*const fwnode_handle) -> u32>,
    property_present: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> bool>,
    property_read_bool: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> bool>,
    property_read_int_array: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, u32, *mut (), u64) -> i32>,
    property_read_string_array: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, *mut *const u8, u64) -> i32>,
    get_name: Option<unsafe extern "C" fn(*const fwnode_handle) -> *const u8>,
    get_name_prefix: Option<unsafe extern "C" fn(*const fwnode_handle) -> *const u8>,
    get_parent: Option<unsafe extern "C" fn(*const fwnode_handle) -> *mut fwnode_handle>,
    get_next_child_node: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_handle) -> *mut fwnode_handle>,
    get_named_child_node: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> *mut fwnode_handle>,
    get_reference_args: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, *const u8, u32, u32, *mut fwnode_reference_args) -> i32>,
    graph_get_next_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_handle) -> *mut fwnode_handle>,
    graph_get_remote_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle) -> *mut fwnode_handle>,
    graph_get_port_parent: Option<unsafe extern "C" fn(*mut fwnode_handle) -> *mut fwnode_handle>,
    graph_parse_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_endpoint) -> i32>,
    iomap: Option<unsafe extern "C" fn(*mut fwnode_handle, i32) -> *mut ()>,
    irq_get: Option<unsafe extern "C" fn(*const fwnode_handle, u32) -> i32>,
    add_links: Option<unsafe extern "C" fn(*mut fwnode_handle) -> i32>,
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


// disallowed nodes
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


#[repr(C)]
#[derive(Clone, Copy)]
struct reciprocal_value {
    m: u32,
    sh1: u8,
    sh2: u8,
}


// no flags
// a kmalloc_nolock() allocation
// prevent obj_exts array allocation
// spinning not allowed
// slab_folio - The folio allocated for a slab
// @s: The slab.
//
// Slabs are allocated as folios that contain the individual objects and are
// using some fields in the first struct page of the folio - those fields are
// now accessed by struct slab. It is occasionally necessary to convert back to
// a folio in order to communicate with the rest of the mm.  Please use this
// helper function instead of casting yourself, as the implementation may change
// in the future.
#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache_order_objects {
    x: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache_per_node_ptrs {
    barn: *mut node_barn,
    node: *mut kmem_cache_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
    cpu_sheaves: *mut slub_percpu_sheaves,
    flags: u32,
    min_partial: u64,
    size: u32,
    object_size: u32,
    reciprocal_size: reciprocal_value,
    offset: u32,
    sheaf_capacity: u32,
    oo: kmem_cache_order_objects,
    min: kmem_cache_order_objects,
    allocflags: u32,
    refcount: i32,
    ctor: Option<unsafe extern "C" fn(*mut ())>,
    inuse: u32,
    align: u32,
    red_left_pad: u32,
    name: *const u8,
    list: list_head,
    kobj: kobject,
    remote_node_defrag_ratio: u32,
    per_node: [kmem_cache_per_node_ptrs; 64],
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__traceiter_oom_score_adj_update")]
extern "C" fn __traceiter_oom_score_adj_update(mut __data: *mut (), task: *mut task_struct) -> i32 {
let mut it_func_ptr: *mut tracepoint_func = unsafe { core::mem::zeroed() };
let mut __v0: *mut tracepoint_func;
let mut __v1: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_792() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_oom_score_adj_update)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_791: *mut tracepoint_func = __v1;
__v0 = __UNIQUE_ID_rcu_791;
it_func_ptr = __v0;
if !it_func_ptr.is_null() {
loop {
let mut __v3: *mut ();
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_793() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*it_func_ptr).func) as *const *mut ()) };
let it_func: *mut () = __v3;
__data = unsafe { (*it_func_ptr).data };
unsafe { unsafe { core::mem::transmute::<*mut (), Option<unsafe extern "C" fn(*mut (), *mut task_struct)>>(it_func) }.unwrap()(__data, task) };
let __v6: *mut tracepoint_func = it_func_ptr;
let __v7: *mut tracepoint_func = __v6.wrapping_offset(1 as isize);
it_func_ptr = __v7;
if unsafe { (*__v7).func }.is_null() {
break;
}
}
}
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__traceiter_reclaim_retry_zone")]
extern "C" fn __traceiter_reclaim_retry_zone(mut __data: *mut (), zoneref: *mut zoneref, order: i32, reclaimable: u64, available: u64, min_wmark: u64, no_progress_loops: i32, wmark_check: bool) -> i32 {
let mut it_func_ptr: *mut tracepoint_func = unsafe { core::mem::zeroed() };
let mut __v0: *mut tracepoint_func;
let mut __v1: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_795() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_reclaim_retry_zone)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_794: *mut tracepoint_func = __v1;
__v0 = __UNIQUE_ID_rcu_794;
it_func_ptr = __v0;
if !it_func_ptr.is_null() {
loop {
let mut __v3: *mut ();
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_796() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*it_func_ptr).func) as *const *mut ()) };
let it_func: *mut () = __v3;
__data = unsafe { (*it_func_ptr).data };
unsafe { unsafe { core::mem::transmute::<*mut (), Option<unsafe extern "C" fn(*mut (), *mut zoneref, i32, u64, u64, u64, i32, bool)>>(it_func) }.unwrap()(__data, zoneref, order, reclaimable, available, min_wmark, no_progress_loops, wmark_check) };
let __v6: *mut tracepoint_func = it_func_ptr;
let __v7: *mut tracepoint_func = __v6.wrapping_offset(1 as isize);
it_func_ptr = __v7;
if unsafe { (*__v7).func }.is_null() {
break;
}
}
}
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__traceiter_mark_victim")]
extern "C" fn __traceiter_mark_victim(mut __data: *mut (), task: *mut task_struct, uid: u32) -> i32 {
let mut it_func_ptr: *mut tracepoint_func = unsafe { core::mem::zeroed() };
let mut __v0: *mut tracepoint_func;
let mut __v1: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_798() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_mark_victim)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_797: *mut tracepoint_func = __v1;
__v0 = __UNIQUE_ID_rcu_797;
it_func_ptr = __v0;
if !it_func_ptr.is_null() {
loop {
let mut __v3: *mut ();
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_799() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*it_func_ptr).func) as *const *mut ()) };
let it_func: *mut () = __v3;
__data = unsafe { (*it_func_ptr).data };
unsafe { unsafe { core::mem::transmute::<*mut (), Option<unsafe extern "C" fn(*mut (), *mut task_struct, u32)>>(it_func) }.unwrap()(__data, task, uid) };
let __v6: *mut tracepoint_func = it_func_ptr;
let __v7: *mut tracepoint_func = __v6.wrapping_offset(1 as isize);
it_func_ptr = __v7;
if unsafe { (*__v7).func }.is_null() {
break;
}
}
}
0
}

fn __do_trace_mark_victim(task: *mut task_struct, uid: u32) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_750: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_750) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_751: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_751) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_753() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_mark_victim)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_752: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_752;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut task_struct, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut task_struct, u32) -> i32>>(__SCT__tp_func_mark_victim as *const ()) };
unsafe { __v12.unwrap()(__data, task, uid) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_mark_victim: [u8; 12] = [109, 97, 114, 107, 95, 118, 105, 99, 116, 105, 109, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_mark_victim_754: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_mark_victim) }) as *mut ();

fn trace_mark_victim(task: *mut task_struct, uid: u32) {
if static_branch_unlikely__5!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mark_victim.key) }) != 0 {
__do_trace_mark_victim(task, uid);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_755: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__traceiter_wake_reaper")]
extern "C" fn __traceiter_wake_reaper(mut __data: *mut (), pid: i32) -> i32 {
let mut it_func_ptr: *mut tracepoint_func = unsafe { core::mem::zeroed() };
let mut __v0: *mut tracepoint_func;
let mut __v1: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_801() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_wake_reaper)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_800: *mut tracepoint_func = __v1;
__v0 = __UNIQUE_ID_rcu_800;
it_func_ptr = __v0;
if !it_func_ptr.is_null() {
loop {
let mut __v3: *mut ();
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_802() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*it_func_ptr).func) as *const *mut ()) };
let it_func: *mut () = __v3;
__data = unsafe { (*it_func_ptr).data };
unsafe { unsafe { core::mem::transmute::<*mut (), Option<unsafe extern "C" fn(*mut (), i32)>>(it_func) }.unwrap()(__data, pid) };
let __v6: *mut tracepoint_func = it_func_ptr;
let __v7: *mut tracepoint_func = __v6.wrapping_offset(1 as isize);
it_func_ptr = __v7;
if unsafe { (*__v7).func }.is_null() {
break;
}
}
}
0
}

fn __do_trace_wake_reaper(pid: i32) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_757: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_757) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_758: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_758) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_760() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_wake_reaper)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_759: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_759;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), i32) -> i32>>(__SCT__tp_func_wake_reaper as *const ()) };
unsafe { __v12.unwrap()(__data, pid) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_wake_reaper: [u8; 12] = [119, 97, 107, 101, 95, 114, 101, 97, 112, 101, 114, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_wake_reaper_761: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_wake_reaper) }) as *mut ();

fn trace_wake_reaper(pid: i32) {
if static_branch_unlikely__6!(unsafe { core::ptr::addr_of_mut!(__tracepoint_wake_reaper.key) }) != 0 {
__do_trace_wake_reaper(pid);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_762: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__traceiter_start_task_reaping")]
extern "C" fn __traceiter_start_task_reaping(mut __data: *mut (), pid: i32) -> i32 {
let mut it_func_ptr: *mut tracepoint_func = unsafe { core::mem::zeroed() };
let mut __v0: *mut tracepoint_func;
let mut __v1: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_804() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_start_task_reaping)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_803: *mut tracepoint_func = __v1;
__v0 = __UNIQUE_ID_rcu_803;
it_func_ptr = __v0;
if !it_func_ptr.is_null() {
loop {
let mut __v3: *mut ();
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_805() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*it_func_ptr).func) as *const *mut ()) };
let it_func: *mut () = __v3;
__data = unsafe { (*it_func_ptr).data };
unsafe { unsafe { core::mem::transmute::<*mut (), Option<unsafe extern "C" fn(*mut (), i32)>>(it_func) }.unwrap()(__data, pid) };
let __v6: *mut tracepoint_func = it_func_ptr;
let __v7: *mut tracepoint_func = __v6.wrapping_offset(1 as isize);
it_func_ptr = __v7;
if unsafe { (*__v7).func }.is_null() {
break;
}
}
}
0
}

fn __do_trace_start_task_reaping(pid: i32) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_764: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_764) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_765: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_765) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_767() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_start_task_reaping)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_766: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_766;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), i32) -> i32>>(__SCT__tp_func_start_task_reaping as *const ()) };
unsafe { __v12.unwrap()(__data, pid) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_start_task_reaping: __SlateAlign16<[u8; 19]> = __SlateAlign16([115u8, 116u8, 97u8, 114u8, 116u8, 95u8, 116u8, 97u8, 115u8, 107u8, 95u8, 114u8, 101u8, 97u8, 112u8, 105u8, 110u8, 103u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_start_task_reaping_768: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_start_task_reaping) }) as *mut ();

fn trace_start_task_reaping(pid: i32) {
if static_branch_unlikely__7!(unsafe { core::ptr::addr_of_mut!(__tracepoint_start_task_reaping.key) }) != 0 {
__do_trace_start_task_reaping(pid);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_769: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__traceiter_finish_task_reaping")]
extern "C" fn __traceiter_finish_task_reaping(mut __data: *mut (), pid: i32) -> i32 {
let mut it_func_ptr: *mut tracepoint_func = unsafe { core::mem::zeroed() };
let mut __v0: *mut tracepoint_func;
let mut __v1: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_807() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_finish_task_reaping)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_806: *mut tracepoint_func = __v1;
__v0 = __UNIQUE_ID_rcu_806;
it_func_ptr = __v0;
if !it_func_ptr.is_null() {
loop {
let mut __v3: *mut ();
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_808() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*it_func_ptr).func) as *const *mut ()) };
let it_func: *mut () = __v3;
__data = unsafe { (*it_func_ptr).data };
unsafe { unsafe { core::mem::transmute::<*mut (), Option<unsafe extern "C" fn(*mut (), i32)>>(it_func) }.unwrap()(__data, pid) };
let __v6: *mut tracepoint_func = it_func_ptr;
let __v7: *mut tracepoint_func = __v6.wrapping_offset(1 as isize);
it_func_ptr = __v7;
if unsafe { (*__v7).func }.is_null() {
break;
}
}
}
0
}

fn __do_trace_finish_task_reaping(pid: i32) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_771: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_771) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_772: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_772) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_774() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_finish_task_reaping)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_773: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_773;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), i32) -> i32>>(__SCT__tp_func_finish_task_reaping as *const ()) };
unsafe { __v12.unwrap()(__data, pid) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_finish_task_reaping: __SlateAlign16<[u8; 20]> = __SlateAlign16([102u8, 105u8, 110u8, 105u8, 115u8, 104u8, 95u8, 116u8, 97u8, 115u8, 107u8, 95u8, 114u8, 101u8, 97u8, 112u8, 105u8, 110u8, 103u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_finish_task_reaping_775: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_finish_task_reaping) }) as *mut ();

core::arch::global_asm!("\t.section \"__tracepoints_ptrs\", \"a\"\t\t\n\t.balign 4\t\t\t\t\t\n\t.long \t{0} - .\t\t\n\t.previous\t\t\t\t\t\n\n.pushsection .discard.ibt_endbr_noseal\n\t .quad {1}\n\t.popsection\n\t\n.pushsection .static_call.text, \"ax\"\t\t\n.align 4\t\t\t\t\t\t\n.globl __SCT__tp_func_oom_score_adj_update\t\t\n__SCT__tp_func_oom_score_adj_update:\t\t\t\n911: .pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 911b - ., 1; .popsection\t\t\t\t\t\n.byte 0xe9; .long {2} - (. + 4)\t\t\t\t\t\t\n.byte 0x0f, 0xb9, 0xcc\t\t\t\t\n.type __SCT__tp_func_oom_score_adj_update, @function\t\n.size __SCT__tp_func_oom_score_adj_update, . - __SCT__tp_func_oom_score_adj_update \n.popsection\t\t\t\t\t\n\n\t.section \"__tracepoints_ptrs\", \"a\"\t\t\n\t.balign 4\t\t\t\t\t\n\t.long \t{3} - .\t\t\n\t.previous\t\t\t\t\t\n\n.pushsection .discard.ibt_endbr_noseal\n\t .quad {4}\n\t.popsection\n\t\n.pushsection .static_call.text, \"ax\"\t\t\n.align 4\t\t\t\t\t\t\n.globl __SCT__tp_func_reclaim_retry_zone\t\t\n__SCT__tp_func_reclaim_retry_zone:\t\t\t\n911: .pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 911b - ., 1; .popsection\t\t\t\t\t\n.byte 0xe9; .long {5} - (. + 4)\t\t\t\t\t\t\n.byte 0x0f, 0xb9, 0xcc\t\t\t\t\n.type __SCT__tp_func_reclaim_retry_zone, @function\t\n.size __SCT__tp_func_reclaim_retry_zone, . - __SCT__tp_func_reclaim_retry_zone \n.popsection\t\t\t\t\t\n\n\t.section \"__tracepoints_ptrs\", \"a\"\t\t\n\t.balign 4\t\t\t\t\t\n\t.long \t{6} - .\t\t\n\t.previous\t\t\t\t\t\n\n.pushsection .discard.ibt_endbr_noseal\n\t .quad {7}\n\t.popsection\n\t\n.pushsection .static_call.text, \"ax\"\t\t\n.align 4\t\t\t\t\t\t\n.globl __SCT__tp_func_mark_victim\t\t\n__SCT__tp_func_mark_victim:\t\t\t\n911: .pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 911b - ., 1; .popsection\t\t\t\t\t\n.byte 0xe9; .long {8} - (. + 4)\t\t\t\t\t\t\n.byte 0x0f, 0xb9, 0xcc\t\t\t\t\n.type __SCT__tp_func_mark_victim, @function\t\n.size __SCT__tp_func_mark_victim, . - __SCT__tp_func_mark_victim \n.popsection\t\t\t\t\t\n\n\t.section \"__tracepoints_ptrs\", \"a\"\t\t\n\t.balign 4\t\t\t\t\t\n\t.long \t{9} - .\t\t\n\t.previous\t\t\t\t\t\n\n.pushsection .discard.ibt_endbr_noseal\n\t .quad {10}\n\t.popsection\n\t\n.pushsection .static_call.text, \"ax\"\t\t\n.align 4\t\t\t\t\t\t\n.globl __SCT__tp_func_wake_reaper\t\t\n__SCT__tp_func_wake_reaper:\t\t\t\n911: .pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 911b - ., 1; .popsection\t\t\t\t\t\n.byte 0xe9; .long {11} - (. + 4)\t\t\t\t\t\t\n.byte 0x0f, 0xb9, 0xcc\t\t\t\t\n.type __SCT__tp_func_wake_reaper, @function\t\n.size __SCT__tp_func_wake_reaper, . - __SCT__tp_func_wake_reaper \n.popsection\t\t\t\t\t\n\n\t.section \"__tracepoints_ptrs\", \"a\"\t\t\n\t.balign 4\t\t\t\t\t\n\t.long \t{12} - .\t\t\n\t.previous\t\t\t\t\t\n\n.pushsection .discard.ibt_endbr_noseal\n\t .quad {13}\n\t.popsection\n\t\n.pushsection .static_call.text, \"ax\"\t\t\n.align 4\t\t\t\t\t\t\n.globl __SCT__tp_func_start_task_reaping\t\t\n__SCT__tp_func_start_task_reaping:\t\t\t\n911: .pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 911b - ., 1; .popsection\t\t\t\t\t\n.byte 0xe9; .long {14} - (. + 4)\t\t\t\t\t\t\n.byte 0x0f, 0xb9, 0xcc\t\t\t\t\n.type __SCT__tp_func_start_task_reaping, @function\t\n.size __SCT__tp_func_start_task_reaping, . - __SCT__tp_func_start_task_reaping \n.popsection\t\t\t\t\t\n\n\t.section \"__tracepoints_ptrs\", \"a\"\t\t\n\t.balign 4\t\t\t\t\t\n\t.long \t{15} - .\t\t\n\t.previous\t\t\t\t\t\n\n.pushsection .discard.ibt_endbr_noseal\n\t .quad {16}\n\t.popsection\n\t\n.pushsection .static_call.text, \"ax\"\t\t\n.align 4\t\t\t\t\t\t\n.globl __SCT__tp_func_finish_task_reaping\t\t\n__SCT__tp_func_finish_task_reaping:\t\t\t\n911: .pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 911b - ., 1; .popsection\t\t\t\t\t\n.byte 0xe9; .long {17} - (. + 4)\t\t\t\t\t\t\n.byte 0x0f, 0xb9, 0xcc\t\t\t\t\n.type __SCT__tp_func_finish_task_reaping, @function\t\n.size __SCT__tp_func_finish_task_reaping, . - __SCT__tp_func_finish_task_reaping \n.popsection\t\t\t\t\t\n\n\t.section \"__tracepoints_ptrs\", \"a\"\t\t\n\t.balign 4\t\t\t\t\t\n\t.long \t{18} - .\t\t\n\t.previous\t\t\t\t\t\n\n.pushsection .discard.ibt_endbr_noseal\n\t .quad {19}\n\t.popsection\n\t\n.pushsection .static_call.text, \"ax\"\t\t\n.align 4\t\t\t\t\t\t\n.globl __SCT__tp_func_skip_task_reaping\t\t\n__SCT__tp_func_skip_task_reaping:\t\t\t\n911: .pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 911b - ., 1; .popsection\t\t\t\t\t\n.byte 0xe9; .long {20} - (. + 4)\t\t\t\t\t\t\n.byte 0x0f, 0xb9, 0xcc\t\t\t\t\n.type __SCT__tp_func_skip_task_reaping, @function\t\n.size __SCT__tp_func_skip_task_reaping, . - __SCT__tp_func_skip_task_reaping \n.popsection\t\t\t\t\t\n\n\t.section \"__tracepoints_ptrs\", \"a\"\t\t\n\t.balign 4\t\t\t\t\t\n\t.long \t{21} - .\t\t\n\t.previous\t\t\t\t\t\n\n.pushsection .discard.ibt_endbr_noseal\n\t .quad {22}\n\t.popsection\n\t\n.pushsection .static_call.text, \"ax\"\t\t\n.align 4\t\t\t\t\t\t\n.globl __SCT__tp_func_compact_retry\t\t\n__SCT__tp_func_compact_retry:\t\t\t\n911: .pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 911b - ., 1; .popsection\t\t\t\t\t\n.byte 0xe9; .long {23} - (. + 4)\t\t\t\t\t\t\n.byte 0x0f, 0xb9, 0xcc\t\t\t\t\n.type __SCT__tp_func_compact_retry, @function\t\n.size __SCT__tp_func_compact_retry, . - __SCT__tp_func_compact_retry \n.popsection\t\t\t\t\t\n\n.section\t\".initcall4.init\", \"a\"\t\t\n__initcall__kmod_oom_kill__845_734_oom_init4:\t\t\t\n.long\t{24} - .\t\n.previous\t\t\t\t\t\n\n.section \".export_symbol\",\"a\" ; __export_symbol_register_oom_notifier: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_unregister_oom_notifier: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous", sym __tracepoint_oom_score_adj_update, sym __probestub_oom_score_adj_update, sym __traceiter_oom_score_adj_update, sym __tracepoint_reclaim_retry_zone, sym __probestub_reclaim_retry_zone, sym __traceiter_reclaim_retry_zone, sym __tracepoint_mark_victim, sym __probestub_mark_victim, sym __traceiter_mark_victim, sym __tracepoint_wake_reaper, sym __probestub_wake_reaper, sym __traceiter_wake_reaper, sym __tracepoint_start_task_reaping, sym __probestub_start_task_reaping, sym __traceiter_start_task_reaping, sym __tracepoint_finish_task_reaping, sym __probestub_finish_task_reaping, sym __traceiter_finish_task_reaping, sym __tracepoint_skip_task_reaping, sym __probestub_skip_task_reaping, sym __traceiter_skip_task_reaping, sym __tracepoint_compact_retry, sym __probestub_compact_retry, sym __traceiter_compact_retry, sym oom_init, sym register_oom_notifier, sym unregister_oom_notifier, options(att_syntax));

fn trace_finish_task_reaping(pid: i32) {
if static_branch_unlikely__8!(unsafe { core::ptr::addr_of_mut!(__tracepoint_finish_task_reaping.key) }) != 0 {
__do_trace_finish_task_reaping(pid);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_776: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__traceiter_skip_task_reaping")]
extern "C" fn __traceiter_skip_task_reaping(mut __data: *mut (), pid: i32) -> i32 {
let mut it_func_ptr: *mut tracepoint_func = unsafe { core::mem::zeroed() };
let mut __v0: *mut tracepoint_func;
let mut __v1: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_810() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_skip_task_reaping)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_809: *mut tracepoint_func = __v1;
__v0 = __UNIQUE_ID_rcu_809;
it_func_ptr = __v0;
if !it_func_ptr.is_null() {
loop {
let mut __v3: *mut ();
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_811() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*it_func_ptr).func) as *const *mut ()) };
let it_func: *mut () = __v3;
__data = unsafe { (*it_func_ptr).data };
unsafe { unsafe { core::mem::transmute::<*mut (), Option<unsafe extern "C" fn(*mut (), i32)>>(it_func) }.unwrap()(__data, pid) };
let __v6: *mut tracepoint_func = it_func_ptr;
let __v7: *mut tracepoint_func = __v6.wrapping_offset(1 as isize);
it_func_ptr = __v7;
if unsafe { (*__v7).func }.is_null() {
break;
}
}
}
0
}

fn __do_trace_skip_task_reaping(pid: i32) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_778: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_778) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_779: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_779) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_781() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_skip_task_reaping)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_780: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_780;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), i32) -> i32>>(__SCT__tp_func_skip_task_reaping as *const ()) };
unsafe { __v12.unwrap()(__data, pid) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_skip_task_reaping: __SlateAlign16<[u8; 18]> = __SlateAlign16([115u8, 107u8, 105u8, 112u8, 95u8, 116u8, 97u8, 115u8, 107u8, 95u8, 114u8, 101u8, 97u8, 112u8, 105u8, 110u8, 103u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_skip_task_reaping_782: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_skip_task_reaping) }) as *mut ();

fn trace_skip_task_reaping(pid: i32) {
if static_branch_unlikely__9!(unsafe { core::ptr::addr_of_mut!(__tracepoint_skip_task_reaping.key) }) != 0 {
__do_trace_skip_task_reaping(pid);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_783: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__traceiter_compact_retry")]
extern "C" fn __traceiter_compact_retry(mut __data: *mut (), order: i32, priority: u32, result: u32, retries: i32, max_retries: i32, ret: bool) -> i32 {
let mut it_func_ptr: *mut tracepoint_func = unsafe { core::mem::zeroed() };
let mut __v0: *mut tracepoint_func;
let mut __v1: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_813() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_compact_retry)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_812: *mut tracepoint_func = __v1;
__v0 = __UNIQUE_ID_rcu_812;
it_func_ptr = __v0;
if !it_func_ptr.is_null() {
loop {
let mut __v3: *mut ();
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_814() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*it_func_ptr).func) as *const *mut ()) };
let it_func: *mut () = __v3;
__data = unsafe { (*it_func_ptr).data };
unsafe { unsafe { core::mem::transmute::<*mut (), Option<unsafe extern "C" fn(*mut (), i32, u32, u32, i32, i32, bool)>>(it_func) }.unwrap()(__data, order, priority, result, retries, max_retries, ret) };
let __v6: *mut tracepoint_func = it_func_ptr;
let __v7: *mut tracepoint_func = __v6.wrapping_offset(1 as isize);
it_func_ptr = __v7;
if unsafe { (*__v7).func }.is_null() {
break;
}
}
}
0
}

#[unsafe(link_section = "__tracepoints_strings")]
static __tpstrtab_oom_score_adj_update: __SlateAlign16<[u8; 21]> = __SlateAlign16([111u8, 111u8, 109u8, 95u8, 115u8, 99u8, 111u8, 114u8, 101u8, 95u8, 97u8, 100u8, 106u8, 95u8, 117u8, 112u8, 100u8, 97u8, 116u8, 101u8, 0u8]);

#[unsafe(no_mangle)]
#[used(compiler)]
#[unsafe(link_section = "__tracepoints")]
static mut __tracepoint_oom_score_adj_update: tracepoint = tracepoint { name: unsafe { core::ptr::addr_of!(__tpstrtab_oom_score_adj_update.0) as *const u8 }, key: static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} }, static_call_key: unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_oom_score_adj_update) }, static_call_tramp: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut task_struct) -> i32>, *mut ()>(unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut task_struct) -> i32>>(__SCT__tp_func_oom_score_adj_update as *const ()) }) }, iterator: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut task_struct) -> i32>, *mut ()>(Some(__traceiter_oom_score_adj_update)) }, probestub: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut task_struct)>, *mut ()>(Some(__probestub_oom_score_adj_update)) }, funcs: core::ptr::null_mut::<tracepoint_func>(), ext: core::ptr::null_mut::<tracepoint_ext>() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__probestub_oom_score_adj_update")]
extern "C" fn __probestub_oom_score_adj_update(__data: *mut (), task: *mut task_struct) {
}

#[unsafe(no_mangle)]
static mut __SCK__tp_func_oom_score_adj_update: static_call_key = static_call_key { func: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut task_struct) -> i32>, *mut ()>(Some(__traceiter_oom_score_adj_update)) }, __slate_anon_1: __SlateRecord_static_call_types_h_65 { r#type: 1u64 } };

#[unsafe(link_section = "__tracepoints_strings")]
static __tpstrtab_reclaim_retry_zone: __SlateAlign16<[u8; 19]> = __SlateAlign16([114u8, 101u8, 99u8, 108u8, 97u8, 105u8, 109u8, 95u8, 114u8, 101u8, 116u8, 114u8, 121u8, 95u8, 122u8, 111u8, 110u8, 101u8, 0u8]);

#[unsafe(no_mangle)]
#[used(compiler)]
#[unsafe(link_section = "__tracepoints")]
static mut __tracepoint_reclaim_retry_zone: tracepoint = tracepoint { name: unsafe { core::ptr::addr_of!(__tpstrtab_reclaim_retry_zone.0) as *const u8 }, key: static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} }, static_call_key: unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_reclaim_retry_zone) }, static_call_tramp: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut zoneref, i32, u64, u64, u64, i32, bool) -> i32>, *mut ()>(unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut zoneref, i32, u64, u64, u64, i32, bool) -> i32>>(__SCT__tp_func_reclaim_retry_zone as *const ()) }) }, iterator: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut zoneref, i32, u64, u64, u64, i32, bool) -> i32>, *mut ()>(Some(__traceiter_reclaim_retry_zone)) }, probestub: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut zoneref, i32, u64, u64, u64, i32, bool)>, *mut ()>(Some(__probestub_reclaim_retry_zone)) }, funcs: core::ptr::null_mut::<tracepoint_func>(), ext: core::ptr::null_mut::<tracepoint_ext>() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__probestub_reclaim_retry_zone")]
extern "C" fn __probestub_reclaim_retry_zone(__data: *mut (), zoneref: *mut zoneref, order: i32, reclaimable: u64, available: u64, min_wmark: u64, no_progress_loops: i32, wmark_check: bool) {
}

#[unsafe(no_mangle)]
static mut __SCK__tp_func_reclaim_retry_zone: static_call_key = static_call_key { func: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut zoneref, i32, u64, u64, u64, i32, bool) -> i32>, *mut ()>(Some(__traceiter_reclaim_retry_zone)) }, __slate_anon_1: __SlateRecord_static_call_types_h_65 { r#type: 1u64 } };

#[unsafe(link_section = "__tracepoints_strings")]
static __tpstrtab_mark_victim: [u8; 12] = [109, 97, 114, 107, 95, 118, 105, 99, 116, 105, 109, 0];

#[unsafe(no_mangle)]
#[used(compiler)]
#[unsafe(link_section = "__tracepoints")]
static mut __tracepoint_mark_victim: tracepoint = tracepoint { name: unsafe { core::ptr::addr_of!(__tpstrtab_mark_victim) as *const u8 }, key: static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} }, static_call_key: unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_mark_victim) }, static_call_tramp: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut task_struct, u32) -> i32>, *mut ()>(unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut task_struct, u32) -> i32>>(__SCT__tp_func_mark_victim as *const ()) }) }, iterator: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut task_struct, u32) -> i32>, *mut ()>(Some(__traceiter_mark_victim)) }, probestub: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut task_struct, u32)>, *mut ()>(Some(__probestub_mark_victim)) }, funcs: core::ptr::null_mut::<tracepoint_func>(), ext: core::ptr::null_mut::<tracepoint_ext>() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__probestub_mark_victim")]
extern "C" fn __probestub_mark_victim(__data: *mut (), task: *mut task_struct, uid: u32) {
}

#[unsafe(no_mangle)]
static mut __SCK__tp_func_mark_victim: static_call_key = static_call_key { func: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut task_struct, u32) -> i32>, *mut ()>(Some(__traceiter_mark_victim)) }, __slate_anon_1: __SlateRecord_static_call_types_h_65 { r#type: 1u64 } };

#[unsafe(link_section = "__tracepoints_strings")]
static __tpstrtab_wake_reaper: [u8; 12] = [119, 97, 107, 101, 95, 114, 101, 97, 112, 101, 114, 0];

#[unsafe(no_mangle)]
#[used(compiler)]
#[unsafe(link_section = "__tracepoints")]
static mut __tracepoint_wake_reaper: tracepoint = tracepoint { name: unsafe { core::ptr::addr_of!(__tpstrtab_wake_reaper) as *const u8 }, key: static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} }, static_call_key: unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_wake_reaper) }, static_call_tramp: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32) -> i32>, *mut ()>(unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), i32) -> i32>>(__SCT__tp_func_wake_reaper as *const ()) }) }, iterator: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32) -> i32>, *mut ()>(Some(__traceiter_wake_reaper)) }, probestub: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32)>, *mut ()>(Some(__probestub_wake_reaper)) }, funcs: core::ptr::null_mut::<tracepoint_func>(), ext: core::ptr::null_mut::<tracepoint_ext>() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__probestub_wake_reaper")]
extern "C" fn __probestub_wake_reaper(__data: *mut (), pid: i32) {
}

#[unsafe(no_mangle)]
static mut __SCK__tp_func_wake_reaper: static_call_key = static_call_key { func: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32) -> i32>, *mut ()>(Some(__traceiter_wake_reaper)) }, __slate_anon_1: __SlateRecord_static_call_types_h_65 { r#type: 1u64 } };

#[unsafe(link_section = "__tracepoints_strings")]
static __tpstrtab_start_task_reaping: __SlateAlign16<[u8; 19]> = __SlateAlign16([115u8, 116u8, 97u8, 114u8, 116u8, 95u8, 116u8, 97u8, 115u8, 107u8, 95u8, 114u8, 101u8, 97u8, 112u8, 105u8, 110u8, 103u8, 0u8]);

#[unsafe(no_mangle)]
#[used(compiler)]
#[unsafe(link_section = "__tracepoints")]
static mut __tracepoint_start_task_reaping: tracepoint = tracepoint { name: unsafe { core::ptr::addr_of!(__tpstrtab_start_task_reaping.0) as *const u8 }, key: static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} }, static_call_key: unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_start_task_reaping) }, static_call_tramp: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32) -> i32>, *mut ()>(unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), i32) -> i32>>(__SCT__tp_func_start_task_reaping as *const ()) }) }, iterator: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32) -> i32>, *mut ()>(Some(__traceiter_start_task_reaping)) }, probestub: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32)>, *mut ()>(Some(__probestub_start_task_reaping)) }, funcs: core::ptr::null_mut::<tracepoint_func>(), ext: core::ptr::null_mut::<tracepoint_ext>() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__probestub_start_task_reaping")]
extern "C" fn __probestub_start_task_reaping(__data: *mut (), pid: i32) {
}

#[unsafe(no_mangle)]
static mut __SCK__tp_func_start_task_reaping: static_call_key = static_call_key { func: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32) -> i32>, *mut ()>(Some(__traceiter_start_task_reaping)) }, __slate_anon_1: __SlateRecord_static_call_types_h_65 { r#type: 1u64 } };

#[unsafe(link_section = "__tracepoints_strings")]
static __tpstrtab_finish_task_reaping: __SlateAlign16<[u8; 20]> = __SlateAlign16([102u8, 105u8, 110u8, 105u8, 115u8, 104u8, 95u8, 116u8, 97u8, 115u8, 107u8, 95u8, 114u8, 101u8, 97u8, 112u8, 105u8, 110u8, 103u8, 0u8]);

#[unsafe(no_mangle)]
#[used(compiler)]
#[unsafe(link_section = "__tracepoints")]
static mut __tracepoint_finish_task_reaping: tracepoint = tracepoint { name: unsafe { core::ptr::addr_of!(__tpstrtab_finish_task_reaping.0) as *const u8 }, key: static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} }, static_call_key: unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_finish_task_reaping) }, static_call_tramp: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32) -> i32>, *mut ()>(unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), i32) -> i32>>(__SCT__tp_func_finish_task_reaping as *const ()) }) }, iterator: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32) -> i32>, *mut ()>(Some(__traceiter_finish_task_reaping)) }, probestub: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32)>, *mut ()>(Some(__probestub_finish_task_reaping)) }, funcs: core::ptr::null_mut::<tracepoint_func>(), ext: core::ptr::null_mut::<tracepoint_ext>() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__probestub_finish_task_reaping")]
extern "C" fn __probestub_finish_task_reaping(__data: *mut (), pid: i32) {
}

#[unsafe(no_mangle)]
static mut __SCK__tp_func_finish_task_reaping: static_call_key = static_call_key { func: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32) -> i32>, *mut ()>(Some(__traceiter_finish_task_reaping)) }, __slate_anon_1: __SlateRecord_static_call_types_h_65 { r#type: 1u64 } };

#[unsafe(link_section = "__tracepoints_strings")]
static __tpstrtab_skip_task_reaping: __SlateAlign16<[u8; 18]> = __SlateAlign16([115u8, 107u8, 105u8, 112u8, 95u8, 116u8, 97u8, 115u8, 107u8, 95u8, 114u8, 101u8, 97u8, 112u8, 105u8, 110u8, 103u8, 0u8]);

#[unsafe(no_mangle)]
#[used(compiler)]
#[unsafe(link_section = "__tracepoints")]
static mut __tracepoint_skip_task_reaping: tracepoint = tracepoint { name: unsafe { core::ptr::addr_of!(__tpstrtab_skip_task_reaping.0) as *const u8 }, key: static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} }, static_call_key: unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_skip_task_reaping) }, static_call_tramp: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32) -> i32>, *mut ()>(unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), i32) -> i32>>(__SCT__tp_func_skip_task_reaping as *const ()) }) }, iterator: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32) -> i32>, *mut ()>(Some(__traceiter_skip_task_reaping)) }, probestub: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32)>, *mut ()>(Some(__probestub_skip_task_reaping)) }, funcs: core::ptr::null_mut::<tracepoint_func>(), ext: core::ptr::null_mut::<tracepoint_ext>() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__probestub_skip_task_reaping")]
extern "C" fn __probestub_skip_task_reaping(__data: *mut (), pid: i32) {
}

#[unsafe(no_mangle)]
static mut __SCK__tp_func_skip_task_reaping: static_call_key = static_call_key { func: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32) -> i32>, *mut ()>(Some(__traceiter_skip_task_reaping)) }, __slate_anon_1: __SlateRecord_static_call_types_h_65 { r#type: 1u64 } };

#[unsafe(link_section = "__tracepoints_strings")]
static __tpstrtab_compact_retry: [u8; 14] = [99, 111, 109, 112, 97, 99, 116, 95, 114, 101, 116, 114, 121, 0];

#[unsafe(no_mangle)]
#[used(compiler)]
#[unsafe(link_section = "__tracepoints")]
static mut __tracepoint_compact_retry: tracepoint = tracepoint { name: unsafe { core::ptr::addr_of!(__tpstrtab_compact_retry) as *const u8 }, key: static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} }, static_call_key: unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_compact_retry) }, static_call_tramp: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32, u32, u32, i32, i32, bool) -> i32>, *mut ()>(unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), i32, u32, u32, i32, i32, bool) -> i32>>(__SCT__tp_func_compact_retry as *const ()) }) }, iterator: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32, u32, u32, i32, i32, bool) -> i32>, *mut ()>(Some(__traceiter_compact_retry)) }, probestub: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32, u32, u32, i32, i32, bool)>, *mut ()>(Some(__probestub_compact_retry)) }, funcs: core::ptr::null_mut::<tracepoint_func>(), ext: core::ptr::null_mut::<tracepoint_ext>() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.__probestub_compact_retry")]
extern "C" fn __probestub_compact_retry(__data: *mut (), order: i32, priority: u32, result: u32, retries: i32, max_retries: i32, ret: bool) {
}

#[unsafe(no_mangle)]
static mut __SCK__tp_func_compact_retry: static_call_key = static_call_key { func: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32, u32, u32, i32, i32, bool) -> i32>, *mut ()>(Some(__traceiter_compact_retry)) }, __slate_anon_1: __SlateRecord_static_call_types_h_65 { r#type: 1u64 } };

static str__oom__trace_system_name: [u8; 4] = [111, 111, 109, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_DMA_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_DMA_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 0u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_DMA_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_DMA_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_HIGHMEM_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_HIGHMEM_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 1u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_HIGHMEM_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_HIGHMEM_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_DMA32_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_DMA32_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 2u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_DMA32_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_DMA32_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_MOVABLE_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_MOVABLE_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 3u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_MOVABLE_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_MOVABLE_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_RECLAIMABLE_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_RECLAIMABLE_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 4u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_RECLAIMABLE_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_RECLAIMABLE_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_HIGH_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_HIGH_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 5u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_HIGH_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_HIGH_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_IO_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_IO_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 6u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_IO_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_IO_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_FS_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_FS_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 7u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_FS_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_FS_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_ZERO_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_ZERO_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 8u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_ZERO_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_ZERO_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_DIRECT_RECLAIM_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_DIRECT_RECLAIM_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 10u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_DIRECT_RECLAIM_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_DIRECT_RECLAIM_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_KSWAPD_RECLAIM_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_KSWAPD_RECLAIM_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 11u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_KSWAPD_RECLAIM_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_KSWAPD_RECLAIM_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_WRITE_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_WRITE_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 12u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_WRITE_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_WRITE_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_NOWARN_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_NOWARN_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 13u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_NOWARN_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_NOWARN_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_RETRY_MAYFAIL_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_RETRY_MAYFAIL_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 14u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_RETRY_MAYFAIL_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_RETRY_MAYFAIL_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_NOFAIL_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_NOFAIL_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 15u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_NOFAIL_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_NOFAIL_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_NORETRY_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_NORETRY_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 16u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_NORETRY_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_NORETRY_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_MEMALLOC_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_MEMALLOC_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 17u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_MEMALLOC_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_MEMALLOC_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_COMP_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_COMP_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 18u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_COMP_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_COMP_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_NOMEMALLOC_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_NOMEMALLOC_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 19u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_NOMEMALLOC_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_NOMEMALLOC_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_HARDWALL_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_HARDWALL_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 20u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_HARDWALL_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_HARDWALL_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_THISNODE_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_THISNODE_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 21u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_THISNODE_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_THISNODE_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_ACCOUNT_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_ACCOUNT_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 22u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_ACCOUNT_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_ACCOUNT_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_ZEROTAGS_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_ZEROTAGS_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 23u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_ZEROTAGS_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_ZEROTAGS_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_UNUSED_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_UNUSED_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 9u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_UNUSED_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_UNUSED_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM____GFP_LAST_BIT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"___GFP_LAST_BIT\0".as_ptr() as *mut u8) as *const u8, eval_value: 24u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM____GFP_LAST_BIT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM____GFP_LAST_BIT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_COMPACT_SKIPPED: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"COMPACT_SKIPPED\0".as_ptr() as *mut u8) as *const u8, eval_value: 1u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_COMPACT_SKIPPED: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_COMPACT_SKIPPED) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_COMPACT_DEFERRED: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"COMPACT_DEFERRED\0".as_ptr() as *mut u8) as *const u8, eval_value: 2u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_COMPACT_DEFERRED: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_COMPACT_DEFERRED) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_COMPACT_CONTINUE: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"COMPACT_CONTINUE\0".as_ptr() as *mut u8) as *const u8, eval_value: 4u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_COMPACT_CONTINUE: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_COMPACT_CONTINUE) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_COMPACT_SUCCESS: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"COMPACT_SUCCESS\0".as_ptr() as *mut u8) as *const u8, eval_value: 8u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_COMPACT_SUCCESS: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_COMPACT_SUCCESS) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_COMPACT_PARTIAL_SKIPPED: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"COMPACT_PARTIAL_SKIPPED\0".as_ptr() as *mut u8) as *const u8, eval_value: 6u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_COMPACT_PARTIAL_SKIPPED: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_COMPACT_PARTIAL_SKIPPED) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_COMPACT_COMPLETE: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"COMPACT_COMPLETE\0".as_ptr() as *mut u8) as *const u8, eval_value: 5u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_COMPACT_COMPLETE: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_COMPACT_COMPLETE) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_COMPACT_NO_SUITABLE_PAGE: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"COMPACT_NO_SUITABLE_PAGE\0".as_ptr() as *mut u8) as *const u8, eval_value: 3u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_COMPACT_NO_SUITABLE_PAGE: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_COMPACT_NO_SUITABLE_PAGE) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_COMPACT_NOT_SUITABLE_ZONE: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"COMPACT_NOT_SUITABLE_ZONE\0".as_ptr() as *mut u8) as *const u8, eval_value: 0u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_COMPACT_NOT_SUITABLE_ZONE: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_COMPACT_NOT_SUITABLE_ZONE) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_COMPACT_CONTENDED: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"COMPACT_CONTENDED\0".as_ptr() as *mut u8) as *const u8, eval_value: 7u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_COMPACT_CONTENDED: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_COMPACT_CONTENDED) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_COMPACT_PRIO_SYNC_FULL: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"COMPACT_PRIO_SYNC_FULL\0".as_ptr() as *mut u8) as *const u8, eval_value: 0u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_COMPACT_PRIO_SYNC_FULL: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_COMPACT_PRIO_SYNC_FULL) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_COMPACT_PRIO_SYNC_LIGHT: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"COMPACT_PRIO_SYNC_LIGHT\0".as_ptr() as *mut u8) as *const u8, eval_value: 1u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_COMPACT_PRIO_SYNC_LIGHT: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_COMPACT_PRIO_SYNC_LIGHT) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_COMPACT_PRIO_ASYNC: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"COMPACT_PRIO_ASYNC\0".as_ptr() as *mut u8) as *const u8, eval_value: 2u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_COMPACT_PRIO_ASYNC: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_COMPACT_PRIO_ASYNC) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_ZONE_DMA: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"ZONE_DMA\0".as_ptr() as *mut u8) as *const u8, eval_value: 0u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_ZONE_DMA: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_ZONE_DMA) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_ZONE_DMA32: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"ZONE_DMA32\0".as_ptr() as *mut u8) as *const u8, eval_value: 1u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_ZONE_DMA32: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_ZONE_DMA32) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_ZONE_NORMAL: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"ZONE_NORMAL\0".as_ptr() as *mut u8) as *const u8, eval_value: 2u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_ZONE_NORMAL: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_ZONE_NORMAL) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_ZONE_MOVABLE: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"ZONE_MOVABLE\0".as_ptr() as *mut u8) as *const u8, eval_value: 3u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_ZONE_MOVABLE: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_ZONE_MOVABLE) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_LRU_INACTIVE_ANON: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"LRU_INACTIVE_ANON\0".as_ptr() as *mut u8) as *const u8, eval_value: 0u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_LRU_INACTIVE_ANON: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_LRU_INACTIVE_ANON) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_LRU_ACTIVE_ANON: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"LRU_ACTIVE_ANON\0".as_ptr() as *mut u8) as *const u8, eval_value: 1u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_LRU_ACTIVE_ANON: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_LRU_ACTIVE_ANON) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_LRU_INACTIVE_FILE: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"LRU_INACTIVE_FILE\0".as_ptr() as *mut u8) as *const u8, eval_value: 2u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_LRU_INACTIVE_FILE: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_LRU_INACTIVE_FILE) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_LRU_ACTIVE_FILE: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"LRU_ACTIVE_FILE\0".as_ptr() as *mut u8) as *const u8, eval_value: 3u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_LRU_ACTIVE_FILE: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_LRU_ACTIVE_FILE) };

#[used(compiler)]
#[unsafe(link_section = ".init.data")]
static mut __TRACE_SYSTEM_LRU_UNEVICTABLE: trace_eval_map = trace_eval_map { system: unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 }, eval_string: (b"LRU_UNEVICTABLE\0".as_ptr() as *mut u8) as *const u8, eval_value: 4u64 };

#[used(compiler)]
#[unsafe(link_section = "_ftrace_eval_map")]
static mut TRACE_SYSTEM_LRU_UNEVICTABLE: *mut trace_eval_map = unsafe { core::ptr::addr_of_mut!(__TRACE_SYSTEM_LRU_UNEVICTABLE) };

#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_raw_oom_score_adj_update {
    ent: trace_entry,
    pid: i32,
    comm: [u8; 16],
    oom_score_adj: i16,
    __data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_raw_reclaim_retry_zone {
    ent: trace_entry,
    node: i32,
    zone_idx: i32,
    order: i32,
    reclaimable: u64,
    available: u64,
    min_wmark: u64,
    no_progress_loops: i32,
    wmark_check: bool,
    __data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_raw_mark_victim {
    ent: trace_entry,
    pid: i32,
    __data_loc_comm: u32,
    total_vm: u64,
    anon_rss: u64,
    file_rss: u64,
    shmem_rss: u64,
    uid: u32,
    pgtables: u64,
    oom_score_adj: i16,
    __data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_raw_wake_reaper {
    ent: trace_entry,
    pid: i32,
    __data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_raw_start_task_reaping {
    ent: trace_entry,
    pid: i32,
    __data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_raw_finish_task_reaping {
    ent: trace_entry,
    pid: i32,
    __data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_raw_skip_task_reaping {
    ent: trace_entry,
    pid: i32,
    __data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_raw_compact_retry {
    ent: trace_entry,
    order: i32,
    priority: i32,
    result: i32,
    retries: i32,
    max_retries: i32,
    ret: bool,
    __data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_data_offsets_oom_score_adj_update {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_data_offsets_reclaim_retry_zone {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_data_offsets_mark_victim {
    comm: u32,
    comm_ptr_: *const (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_data_offsets_wake_reaper {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_data_offsets_start_task_reaping {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_data_offsets_finish_task_reaping {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_data_offsets_skip_task_reaping {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_data_offsets_compact_retry {
}


#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_raw_output_oom_score_adj_update")]
extern "C" fn trace_raw_output_oom_score_adj_update(iter: *mut trace_iterator, flags: i32, trace_event: *mut trace_event) -> u32 {
let s: *mut trace_seq = unsafe { core::ptr::addr_of_mut!((*iter).seq) };
let mut ret: i32 = 0;
let field: *mut trace_event_raw_oom_score_adj_update = (unsafe { (*iter).ent }) as *mut trace_event_raw_oom_score_adj_update;
ret = unsafe { trace_raw_output_prep(iter, trace_event) };
if ret != 1 {
return ret as u32;
}
unsafe { trace_event_printf(iter, c"pid=%d comm=%s oom_score_adj=%hd\n".as_ptr() as *const u8, (*field).pid, (*field).comm.as_mut_ptr() as *mut u8, (*field).oom_score_adj as i32) };
unsafe { trace_handle_return(s) }
}

static mut trace_event_type_funcs_oom_score_adj_update: trace_event_functions = {
let mut __t0: trace_event_functions = unsafe { core::mem::zeroed() };
__t0.trace = Some(trace_raw_output_oom_score_adj_update);
__t0
};

#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_raw_output_reclaim_retry_zone")]
extern "C" fn trace_raw_output_reclaim_retry_zone(iter: *mut trace_iterator, flags: i32, trace_event: *mut trace_event) -> u32 {
let s: *mut trace_seq = unsafe { core::ptr::addr_of_mut!((*iter).seq) };
let p: *mut trace_seq = unsafe { core::ptr::addr_of_mut!((*iter).tmp_seq) };
let mut ret: i32 = 0;
let field: *mut trace_event_raw_reclaim_retry_zone = (unsafe { (*iter).ent }) as *mut trace_event_raw_reclaim_retry_zone;
ret = unsafe { trace_raw_output_prep(iter, trace_event) };
if ret != 1 {
return ret as u32;
}
let mut __v0: *const u8;
__v0 = unsafe { trace_print_symbols_seq(p, (*field).zone_idx as u64, core::ptr::addr_of!(symbols__69.0) as *const trace_print_flags, ARRAY_SIZE__7!()) };
unsafe { trace_event_printf(iter, c"node=%d zone=%-8s order=%d reclaimable=%lu available=%lu min_wmark=%lu no_progress_loops=%d wmark_check=%d\n".as_ptr() as *const u8, (*field).node, __v0, (*field).order, (*field).reclaimable, (*field).available, (*field).min_wmark, (*field).no_progress_loops, (*field).wmark_check as i32) };
unsafe { trace_handle_return(s) }
}

static symbols__69: __SlateAlign16<[trace_print_flags; 4]> = __SlateAlign16([trace_print_flags { mask: 0u64, name: (b"DMA\0".as_ptr() as *mut u8) as *const u8 }, trace_print_flags { mask: 1u64, name: (b"DMA32\0".as_ptr() as *mut u8) as *const u8 }, trace_print_flags { mask: 2u64, name: (b"Normal\0".as_ptr() as *mut u8) as *const u8 }, trace_print_flags { mask: 3u64, name: (b"Movable\0".as_ptr() as *mut u8) as *const u8 }]);

static mut trace_event_type_funcs_reclaim_retry_zone: trace_event_functions = {
let mut __t0: trace_event_functions = unsafe { core::mem::zeroed() };
__t0.trace = Some(trace_raw_output_reclaim_retry_zone);
__t0
};

#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_raw_output_mark_victim")]
extern "C" fn trace_raw_output_mark_victim(iter: *mut trace_iterator, flags: i32, trace_event: *mut trace_event) -> u32 {
let s: *mut trace_seq = unsafe { core::ptr::addr_of_mut!((*iter).seq) };
let mut field: *mut trace_event_raw_mark_victim = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
field = (unsafe { (*iter).ent }) as *mut trace_event_raw_mark_victim;
ret = unsafe { trace_raw_output_prep(iter, trace_event) };
if ret != 1 {
return ret as u32;
}
unsafe { trace_event_printf(iter, c"pid=%d comm=%s total-vm=%lukB anon-rss=%lukB file-rss:%lukB shmem-rss:%lukB uid=%u pgtables=%lukB oom_score_adj=%hd\n".as_ptr() as *const u8, (*field).pid, __get_str__6!(field), (*field).total_vm, (*field).anon_rss, (*field).file_rss, (*field).shmem_rss, (*field).uid, (*field).pgtables, (*field).oom_score_adj as i32) };
unsafe { trace_handle_return(s) }
}

static mut trace_event_type_funcs_mark_victim: trace_event_functions = {
let mut __t0: trace_event_functions = unsafe { core::mem::zeroed() };
__t0.trace = Some(trace_raw_output_mark_victim);
__t0
};

#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_raw_output_wake_reaper")]
extern "C" fn trace_raw_output_wake_reaper(iter: *mut trace_iterator, flags: i32, trace_event: *mut trace_event) -> u32 {
let s: *mut trace_seq = unsafe { core::ptr::addr_of_mut!((*iter).seq) };
let mut ret: i32 = 0;
let field: *mut trace_event_raw_wake_reaper = (unsafe { (*iter).ent }) as *mut trace_event_raw_wake_reaper;
ret = unsafe { trace_raw_output_prep(iter, trace_event) };
if ret != 1 {
return ret as u32;
}
unsafe { trace_event_printf(iter, c"pid=%d\n".as_ptr() as *const u8, (*field).pid) };
unsafe { trace_handle_return(s) }
}

static mut trace_event_type_funcs_wake_reaper: trace_event_functions = {
let mut __t0: trace_event_functions = unsafe { core::mem::zeroed() };
__t0.trace = Some(trace_raw_output_wake_reaper);
__t0
};

#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_raw_output_start_task_reaping")]
extern "C" fn trace_raw_output_start_task_reaping(iter: *mut trace_iterator, flags: i32, trace_event: *mut trace_event) -> u32 {
let s: *mut trace_seq = unsafe { core::ptr::addr_of_mut!((*iter).seq) };
let mut ret: i32 = 0;
let field: *mut trace_event_raw_start_task_reaping = (unsafe { (*iter).ent }) as *mut trace_event_raw_start_task_reaping;
ret = unsafe { trace_raw_output_prep(iter, trace_event) };
if ret != 1 {
return ret as u32;
}
unsafe { trace_event_printf(iter, c"pid=%d\n".as_ptr() as *const u8, (*field).pid) };
unsafe { trace_handle_return(s) }
}

static mut trace_event_type_funcs_start_task_reaping: trace_event_functions = {
let mut __t0: trace_event_functions = unsafe { core::mem::zeroed() };
__t0.trace = Some(trace_raw_output_start_task_reaping);
__t0
};

#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_raw_output_finish_task_reaping")]
extern "C" fn trace_raw_output_finish_task_reaping(iter: *mut trace_iterator, flags: i32, trace_event: *mut trace_event) -> u32 {
let s: *mut trace_seq = unsafe { core::ptr::addr_of_mut!((*iter).seq) };
let mut ret: i32 = 0;
let field: *mut trace_event_raw_finish_task_reaping = (unsafe { (*iter).ent }) as *mut trace_event_raw_finish_task_reaping;
ret = unsafe { trace_raw_output_prep(iter, trace_event) };
if ret != 1 {
return ret as u32;
}
unsafe { trace_event_printf(iter, c"pid=%d\n".as_ptr() as *const u8, (*field).pid) };
unsafe { trace_handle_return(s) }
}

static mut trace_event_type_funcs_finish_task_reaping: trace_event_functions = {
let mut __t0: trace_event_functions = unsafe { core::mem::zeroed() };
__t0.trace = Some(trace_raw_output_finish_task_reaping);
__t0
};

#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_raw_output_skip_task_reaping")]
extern "C" fn trace_raw_output_skip_task_reaping(iter: *mut trace_iterator, flags: i32, trace_event: *mut trace_event) -> u32 {
let s: *mut trace_seq = unsafe { core::ptr::addr_of_mut!((*iter).seq) };
let mut ret: i32 = 0;
let field: *mut trace_event_raw_skip_task_reaping = (unsafe { (*iter).ent }) as *mut trace_event_raw_skip_task_reaping;
ret = unsafe { trace_raw_output_prep(iter, trace_event) };
if ret != 1 {
return ret as u32;
}
unsafe { trace_event_printf(iter, c"pid=%d\n".as_ptr() as *const u8, (*field).pid) };
unsafe { trace_handle_return(s) }
}

static mut trace_event_type_funcs_skip_task_reaping: trace_event_functions = {
let mut __t0: trace_event_functions = unsafe { core::mem::zeroed() };
__t0.trace = Some(trace_raw_output_skip_task_reaping);
__t0
};

#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_raw_output_compact_retry")]
extern "C" fn trace_raw_output_compact_retry(iter: *mut trace_iterator, flags: i32, trace_event: *mut trace_event) -> u32 {
let s: *mut trace_seq = unsafe { core::ptr::addr_of_mut!((*iter).seq) };
let p: *mut trace_seq = unsafe { core::ptr::addr_of_mut!((*iter).tmp_seq) };
let mut ret: i32 = 0;
let field: *mut trace_event_raw_compact_retry = (unsafe { (*iter).ent }) as *mut trace_event_raw_compact_retry;
ret = unsafe { trace_raw_output_prep(iter, trace_event) };
if ret != 1 {
return ret as u32;
}
let mut __v0: *const u8;
__v0 = unsafe { trace_print_symbols_seq(p, (*field).priority as u64, core::ptr::addr_of!(symbols__214.0) as *const trace_print_flags, ARRAY_SIZE__14!()) };
let mut __v1: *const u8;
__v1 = unsafe { trace_print_symbols_seq(p, (*field).result as u64, core::ptr::addr_of!(symbols__215.0) as *const trace_print_flags, ARRAY_SIZE__8!()) };
unsafe { trace_event_printf(iter, c"order=%d priority=%s compaction_result=%s retries=%d max_retries=%d should_retry=%d\n".as_ptr() as *const u8, (*field).order, __v0, __v1, (*field).retries, (*field).max_retries, (*field).ret as i32) };
unsafe { trace_handle_return(s) }
}

static symbols__214: __SlateAlign16<[trace_print_flags; 3]> = __SlateAlign16([trace_print_flags { mask: 0u64, name: (b"COMPACT_PRIO_SYNC_FULL\0".as_ptr() as *mut u8) as *const u8 }, trace_print_flags { mask: 1u64, name: (b"COMPACT_PRIO_SYNC_LIGHT\0".as_ptr() as *mut u8) as *const u8 }, trace_print_flags { mask: 2u64, name: (b"COMPACT_PRIO_ASYNC\0".as_ptr() as *mut u8) as *const u8 }]);

static symbols__215: __SlateAlign16<[trace_print_flags; 3]> = __SlateAlign16([trace_print_flags { mask: 1u64, name: (b"failed\0".as_ptr() as *mut u8) as *const u8 }, trace_print_flags { mask: 2u64, name: (b"withdrawn\0".as_ptr() as *mut u8) as *const u8 }, trace_print_flags { mask: 3u64, name: (b"progress\0".as_ptr() as *mut u8) as *const u8 }]);

static mut trace_event_type_funcs_compact_retry: trace_event_functions = {
let mut __t0: trace_event_functions = unsafe { core::mem::zeroed() };
__t0.trace = Some(trace_raw_output_compact_retry);
__t0
};

static mut trace_event_fields_oom_score_adj_update: __SlateAlign16<[trace_event_fields; 4]> = __SlateAlign16([trace_event_fields { r#type: (b"pid_t\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t0: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t0.__slate_anon_0 = {
let mut __t1: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t1.name = (b"pid\0".as_ptr() as *mut u8) as *const u8;
__t1.size = (core::mem::size_of::<i32>() as u64) as i32;
__t1.align = (core::mem::align_of::<__SlateRecord_oom_h_19>() as u64) as i32;
__t1.__slate_bits_0.__set_is_signed((1i32.wrapping_neg() < 1i32) as u32);
__t1.filter_type = 0i32;
__t1
};
__t0
} }, trace_event_fields { r#type: (b"char[TASK_COMM_LEN]\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t2: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t2.__slate_anon_0 = {
let mut __t3: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t3.name = (b"comm\0".as_ptr() as *mut u8) as *const u8;
__t3.size = (core::mem::size_of::<[u8; 16]>() as u64) as i32;
__t3.align = (core::mem::align_of::<__SlateRecord_oom_h_20>() as u64) as i32;
__t3.__slate_bits_0.__set_is_signed((((1i32.wrapping_neg() as u8) as i32) < 1i32) as u32);
__t3.filter_type = 0i32;
__t3.len = 16i32;
__t3
};
__t2
} }, trace_event_fields { r#type: (b"short\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t4: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t4.__slate_anon_0 = {
let mut __t5: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t5.name = (b"oom_score_adj\0".as_ptr() as *mut u8) as *const u8;
__t5.size = (core::mem::size_of::<i16>() as u64) as i32;
__t5.align = (core::mem::align_of::<__SlateRecord_oom_h_21>() as u64) as i32;
__t5.__slate_bits_0.__set_is_signed((((1i32.wrapping_neg() as i16) as i32) < 1i32) as u32);
__t5.filter_type = 0i32;
__t5
};
__t4
} }, {
let mut __t6: trace_event_fields = unsafe { core::mem::zeroed() };
__t6
}]);

static mut trace_event_fields_reclaim_retry_zone: __SlateAlign16<[trace_event_fields; 9]> = __SlateAlign16([trace_event_fields { r#type: (b"int\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t0: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t0.__slate_anon_0 = {
let mut __t1: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t1.name = (b"node\0".as_ptr() as *mut u8) as *const u8;
__t1.size = (core::mem::size_of::<i32>() as u64) as i32;
__t1.align = (core::mem::align_of::<__SlateRecord_oom_h_47>() as u64) as i32;
__t1.__slate_bits_0.__set_is_signed((1i32.wrapping_neg() < 1i32) as u32);
__t1.filter_type = 0i32;
__t1
};
__t0
} }, trace_event_fields { r#type: (b"int\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t2: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t2.__slate_anon_0 = {
let mut __t3: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t3.name = (b"zone_idx\0".as_ptr() as *mut u8) as *const u8;
__t3.size = (core::mem::size_of::<i32>() as u64) as i32;
__t3.align = (core::mem::align_of::<__SlateRecord_oom_h_48>() as u64) as i32;
__t3.__slate_bits_0.__set_is_signed((1i32.wrapping_neg() < 1i32) as u32);
__t3.filter_type = 0i32;
__t3
};
__t2
} }, trace_event_fields { r#type: (b"int\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t4: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t4.__slate_anon_0 = {
let mut __t5: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t5.name = (b"order\0".as_ptr() as *mut u8) as *const u8;
__t5.size = (core::mem::size_of::<i32>() as u64) as i32;
__t5.align = (core::mem::align_of::<__SlateRecord_oom_h_49>() as u64) as i32;
__t5.__slate_bits_0.__set_is_signed((1i32.wrapping_neg() < 1i32) as u32);
__t5.filter_type = 0i32;
__t5
};
__t4
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t6: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t6.__slate_anon_0 = {
let mut __t7: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t7.name = (b"reclaimable\0".as_ptr() as *mut u8) as *const u8;
__t7.size = (core::mem::size_of::<u64>() as u64) as i32;
__t7.align = (core::mem::align_of::<__SlateRecord_oom_h_50>() as u64) as i32;
__t7.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t7.filter_type = 0i32;
__t7
};
__t6
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t8: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t8.__slate_anon_0 = {
let mut __t9: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t9.name = (b"available\0".as_ptr() as *mut u8) as *const u8;
__t9.size = (core::mem::size_of::<u64>() as u64) as i32;
__t9.align = (core::mem::align_of::<__SlateRecord_oom_h_51>() as u64) as i32;
__t9.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t9.filter_type = 0i32;
__t9
};
__t8
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t10: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t10.__slate_anon_0 = {
let mut __t11: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t11.name = (b"min_wmark\0".as_ptr() as *mut u8) as *const u8;
__t11.size = (core::mem::size_of::<u64>() as u64) as i32;
__t11.align = (core::mem::align_of::<__SlateRecord_oom_h_52>() as u64) as i32;
__t11.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t11.filter_type = 0i32;
__t11
};
__t10
} }, trace_event_fields { r#type: (b"int\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t12: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t12.__slate_anon_0 = {
let mut __t13: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t13.name = (b"no_progress_loops\0".as_ptr() as *mut u8) as *const u8;
__t13.size = (core::mem::size_of::<i32>() as u64) as i32;
__t13.align = (core::mem::align_of::<__SlateRecord_oom_h_53>() as u64) as i32;
__t13.__slate_bits_0.__set_is_signed((1i32.wrapping_neg() < 1i32) as u32);
__t13.filter_type = 0i32;
__t13
};
__t12
} }, trace_event_fields { r#type: (b"bool\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t14: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t14.__slate_anon_0 = {
let mut __t15: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t15.name = (b"wmark_check\0".as_ptr() as *mut u8) as *const u8;
__t15.size = (core::mem::size_of::<bool>() as u64) as i32;
__t15.align = (core::mem::align_of::<__SlateRecord_oom_h_54>() as u64) as i32;
__t15.__slate_bits_0.__set_is_signed((((1i32.wrapping_neg() != 0i32) as i32) < ((1i32 != 0i32) as i32)) as u32);
__t15.filter_type = 0i32;
__t15
};
__t14
} }, {
let mut __t16: trace_event_fields = unsafe { core::mem::zeroed() };
__t16
}]);

static mut trace_event_fields_mark_victim: __SlateAlign16<[trace_event_fields; 10]> = __SlateAlign16([trace_event_fields { r#type: (b"int\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t0: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t0.__slate_anon_0 = {
let mut __t1: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t1.name = (b"pid\0".as_ptr() as *mut u8) as *const u8;
__t1.size = (core::mem::size_of::<i32>() as u64) as i32;
__t1.align = (core::mem::align_of::<__SlateRecord_oom_h_82>() as u64) as i32;
__t1.__slate_bits_0.__set_is_signed((1i32.wrapping_neg() < 1i32) as u32);
__t1.filter_type = 0i32;
__t1
};
__t0
} }, trace_event_fields { r#type: (b"__data_loc char[]\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t2: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t2.__slate_anon_0 = {
let mut __t3: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t3.name = (b"comm\0".as_ptr() as *mut u8) as *const u8;
__t3.size = 4i32;
__t3.align = 4i32;
__t3.__slate_bits_0.__set_is_signed((((1i32.wrapping_neg() as u8) as i32) < 1i32) as u32);
__t3.filter_type = 0i32;
__t3
};
__t2
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t4: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t4.__slate_anon_0 = {
let mut __t5: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t5.name = (b"total_vm\0".as_ptr() as *mut u8) as *const u8;
__t5.size = (core::mem::size_of::<u64>() as u64) as i32;
__t5.align = (core::mem::align_of::<__SlateRecord_oom_h_84>() as u64) as i32;
__t5.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t5.filter_type = 0i32;
__t5
};
__t4
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t6: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t6.__slate_anon_0 = {
let mut __t7: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t7.name = (b"anon_rss\0".as_ptr() as *mut u8) as *const u8;
__t7.size = (core::mem::size_of::<u64>() as u64) as i32;
__t7.align = (core::mem::align_of::<__SlateRecord_oom_h_85>() as u64) as i32;
__t7.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t7.filter_type = 0i32;
__t7
};
__t6
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t8: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t8.__slate_anon_0 = {
let mut __t9: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t9.name = (b"file_rss\0".as_ptr() as *mut u8) as *const u8;
__t9.size = (core::mem::size_of::<u64>() as u64) as i32;
__t9.align = (core::mem::align_of::<__SlateRecord_oom_h_86>() as u64) as i32;
__t9.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t9.filter_type = 0i32;
__t9
};
__t8
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t10: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t10.__slate_anon_0 = {
let mut __t11: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t11.name = (b"shmem_rss\0".as_ptr() as *mut u8) as *const u8;
__t11.size = (core::mem::size_of::<u64>() as u64) as i32;
__t11.align = (core::mem::align_of::<__SlateRecord_oom_h_87>() as u64) as i32;
__t11.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t11.filter_type = 0i32;
__t11
};
__t10
} }, trace_event_fields { r#type: (b"uid_t\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t12: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t12.__slate_anon_0 = {
let mut __t13: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t13.name = (b"uid\0".as_ptr() as *mut u8) as *const u8;
__t13.size = (core::mem::size_of::<u32>() as u64) as i32;
__t13.align = (core::mem::align_of::<__SlateRecord_oom_h_88>() as u64) as i32;
__t13.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u32) < 1u32) as u32);
__t13.filter_type = 0i32;
__t13
};
__t12
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t14: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t14.__slate_anon_0 = {
let mut __t15: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t15.name = (b"pgtables\0".as_ptr() as *mut u8) as *const u8;
__t15.size = (core::mem::size_of::<u64>() as u64) as i32;
__t15.align = (core::mem::align_of::<__SlateRecord_oom_h_89>() as u64) as i32;
__t15.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t15.filter_type = 0i32;
__t15
};
__t14
} }, trace_event_fields { r#type: (b"short\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t16: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t16.__slate_anon_0 = {
let mut __t17: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t17.name = (b"oom_score_adj\0".as_ptr() as *mut u8) as *const u8;
__t17.size = (core::mem::size_of::<i16>() as u64) as i32;
__t17.align = (core::mem::align_of::<__SlateRecord_oom_h_90>() as u64) as i32;
__t17.__slate_bits_0.__set_is_signed((((1i32.wrapping_neg() as i16) as i32) < 1i32) as u32);
__t17.filter_type = 0i32;
__t17
};
__t16
} }, {
let mut __t18: trace_event_fields = unsafe { core::mem::zeroed() };
__t18
}]);

static mut trace_event_fields_wake_reaper: __SlateAlign16<[trace_event_fields; 2]> = __SlateAlign16([trace_event_fields { r#type: (b"int\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t0: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t0.__slate_anon_0 = {
let mut __t1: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t1.name = (b"pid\0".as_ptr() as *mut u8) as *const u8;
__t1.size = (core::mem::size_of::<i32>() as u64) as i32;
__t1.align = (core::mem::align_of::<__SlateRecord_oom_h_124>() as u64) as i32;
__t1.__slate_bits_0.__set_is_signed((1i32.wrapping_neg() < 1i32) as u32);
__t1.filter_type = 0i32;
__t1
};
__t0
} }, {
let mut __t2: trace_event_fields = unsafe { core::mem::zeroed() };
__t2
}]);

static mut trace_event_fields_start_task_reaping: __SlateAlign16<[trace_event_fields; 2]> = __SlateAlign16([trace_event_fields { r#type: (b"int\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t0: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t0.__slate_anon_0 = {
let mut __t1: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t1.name = (b"pid\0".as_ptr() as *mut u8) as *const u8;
__t1.size = (core::mem::size_of::<i32>() as u64) as i32;
__t1.align = (core::mem::align_of::<__SlateRecord_oom_h_140>() as u64) as i32;
__t1.__slate_bits_0.__set_is_signed((1i32.wrapping_neg() < 1i32) as u32);
__t1.filter_type = 0i32;
__t1
};
__t0
} }, {
let mut __t2: trace_event_fields = unsafe { core::mem::zeroed() };
__t2
}]);

static mut trace_event_fields_finish_task_reaping: __SlateAlign16<[trace_event_fields; 2]> = __SlateAlign16([trace_event_fields { r#type: (b"int\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t0: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t0.__slate_anon_0 = {
let mut __t1: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t1.name = (b"pid\0".as_ptr() as *mut u8) as *const u8;
__t1.size = (core::mem::size_of::<i32>() as u64) as i32;
__t1.align = (core::mem::align_of::<__SlateRecord_oom_h_156>() as u64) as i32;
__t1.__slate_bits_0.__set_is_signed((1i32.wrapping_neg() < 1i32) as u32);
__t1.filter_type = 0i32;
__t1
};
__t0
} }, {
let mut __t2: trace_event_fields = unsafe { core::mem::zeroed() };
__t2
}]);

static mut trace_event_fields_skip_task_reaping: __SlateAlign16<[trace_event_fields; 2]> = __SlateAlign16([trace_event_fields { r#type: (b"int\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t0: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t0.__slate_anon_0 = {
let mut __t1: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t1.name = (b"pid\0".as_ptr() as *mut u8) as *const u8;
__t1.size = (core::mem::size_of::<i32>() as u64) as i32;
__t1.align = (core::mem::align_of::<__SlateRecord_oom_h_172>() as u64) as i32;
__t1.__slate_bits_0.__set_is_signed((1i32.wrapping_neg() < 1i32) as u32);
__t1.filter_type = 0i32;
__t1
};
__t0
} }, {
let mut __t2: trace_event_fields = unsafe { core::mem::zeroed() };
__t2
}]);

static mut trace_event_fields_compact_retry: __SlateAlign16<[trace_event_fields; 7]> = __SlateAlign16([trace_event_fields { r#type: (b"int\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t0: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t0.__slate_anon_0 = {
let mut __t1: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t1.name = (b"order\0".as_ptr() as *mut u8) as *const u8;
__t1.size = (core::mem::size_of::<i32>() as u64) as i32;
__t1.align = (core::mem::align_of::<__SlateRecord_oom_h_195>() as u64) as i32;
__t1.__slate_bits_0.__set_is_signed((1i32.wrapping_neg() < 1i32) as u32);
__t1.filter_type = 0i32;
__t1
};
__t0
} }, trace_event_fields { r#type: (b"int\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t2: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t2.__slate_anon_0 = {
let mut __t3: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t3.name = (b"priority\0".as_ptr() as *mut u8) as *const u8;
__t3.size = (core::mem::size_of::<i32>() as u64) as i32;
__t3.align = (core::mem::align_of::<__SlateRecord_oom_h_196>() as u64) as i32;
__t3.__slate_bits_0.__set_is_signed((1i32.wrapping_neg() < 1i32) as u32);
__t3.filter_type = 0i32;
__t3
};
__t2
} }, trace_event_fields { r#type: (b"int\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t4: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t4.__slate_anon_0 = {
let mut __t5: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t5.name = (b"result\0".as_ptr() as *mut u8) as *const u8;
__t5.size = (core::mem::size_of::<i32>() as u64) as i32;
__t5.align = (core::mem::align_of::<__SlateRecord_oom_h_197>() as u64) as i32;
__t5.__slate_bits_0.__set_is_signed((1i32.wrapping_neg() < 1i32) as u32);
__t5.filter_type = 0i32;
__t5
};
__t4
} }, trace_event_fields { r#type: (b"int\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t6: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t6.__slate_anon_0 = {
let mut __t7: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t7.name = (b"retries\0".as_ptr() as *mut u8) as *const u8;
__t7.size = (core::mem::size_of::<i32>() as u64) as i32;
__t7.align = (core::mem::align_of::<__SlateRecord_oom_h_198>() as u64) as i32;
__t7.__slate_bits_0.__set_is_signed((1i32.wrapping_neg() < 1i32) as u32);
__t7.filter_type = 0i32;
__t7
};
__t6
} }, trace_event_fields { r#type: (b"int\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t8: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t8.__slate_anon_0 = {
let mut __t9: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t9.name = (b"max_retries\0".as_ptr() as *mut u8) as *const u8;
__t9.size = (core::mem::size_of::<i32>() as u64) as i32;
__t9.align = (core::mem::align_of::<__SlateRecord_oom_h_199>() as u64) as i32;
__t9.__slate_bits_0.__set_is_signed((1i32.wrapping_neg() < 1i32) as u32);
__t9.filter_type = 0i32;
__t9
};
__t8
} }, trace_event_fields { r#type: (b"bool\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t10: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t10.__slate_anon_0 = {
let mut __t11: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t11.name = (b"ret\0".as_ptr() as *mut u8) as *const u8;
__t11.size = (core::mem::size_of::<bool>() as u64) as i32;
__t11.align = (core::mem::align_of::<__SlateRecord_oom_h_200>() as u64) as i32;
__t11.__slate_bits_0.__set_is_signed((((1i32.wrapping_neg() != 0i32) as i32) < ((1i32 != 0i32) as i32)) as u32);
__t11.filter_type = 0i32;
__t11
};
__t10
} }, {
let mut __t12: trace_event_fields = unsafe { core::mem::zeroed() };
__t12
}]);

fn __string_src(str: *const u8) -> *const u8 {
if str.is_null() {
return c"(null)".as_ptr() as *const u8;
}
str
}

fn trace_event_get_offsets_oom_score_adj_update(__data_offsets: &trace_event_data_offsets_oom_score_adj_update, task: *mut task_struct) -> i32 {
let __data_size: i32 = 0;
__data_size
}

fn trace_event_get_offsets_reclaim_retry_zone(__data_offsets: &trace_event_data_offsets_reclaim_retry_zone, zoneref: *mut zoneref, order: i32, reclaimable: u64, available: u64, min_wmark: u64, no_progress_loops: i32, wmark_check: bool) -> i32 {
let __data_size: i32 = 0;
__data_size
}

fn trace_event_get_offsets_mark_victim(__data_offsets: *mut trace_event_data_offsets_mark_victim, task: *mut task_struct, uid: u32) -> i32 {
let mut __data_size: i32 = 0;
let mut __item_length: i32 = 0;
__dynamic_array__5!(u8, comm, unsafe { strlen(__string_src(((*task).comm.as_mut_ptr() as *mut u8) as *const u8)) }.wrapping_add(1), __item_length, __data_offsets, __data_size);
unsafe {
(*__data_offsets).comm_ptr_ = ((*task).comm.as_mut_ptr() as *mut u8) as *const ();
}
__data_size
}

fn trace_event_get_offsets_wake_reaper(__data_offsets: &trace_event_data_offsets_wake_reaper, pid: i32) -> i32 {
let __data_size: i32 = 0;
__data_size
}

fn trace_event_get_offsets_start_task_reaping(__data_offsets: &trace_event_data_offsets_start_task_reaping, pid: i32) -> i32 {
let __data_size: i32 = 0;
__data_size
}

fn trace_event_get_offsets_finish_task_reaping(__data_offsets: &trace_event_data_offsets_finish_task_reaping, pid: i32) -> i32 {
let __data_size: i32 = 0;
__data_size
}

fn trace_event_get_offsets_skip_task_reaping(__data_offsets: &trace_event_data_offsets_skip_task_reaping, pid: i32) -> i32 {
let __data_size: i32 = 0;
__data_size
}

fn trace_event_get_offsets_compact_retry(__data_offsets: &trace_event_data_offsets_compact_retry, order: i32, priority: u32, result: u32, retries: i32, max_retries: i32, ret: bool) -> i32 {
let __data_size: i32 = 0;
__data_size
}

fn do_trace_event_raw_event_oom_score_adj_update(__data: *mut (), task: *mut task_struct) {
let trace_file: *mut trace_event_file = __data as *mut trace_event_file;
let mut __data_offsets: trace_event_data_offsets_oom_score_adj_update = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut fbuffer: trace_event_buffer = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fbuffer) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
let mut __data_size: i32 = 0;
if trace_trigger_soft_disabled(trace_file) {
return;
}
__data_size = trace_event_get_offsets_oom_score_adj_update(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, task);
let entry: *mut trace_event_raw_oom_score_adj_update = (unsafe { trace_event_buffer_reserve(core::ptr::addr_of_mut!(fbuffer), trace_file, 32u64.wrapping_add(__data_size as u64)) }) as *mut trace_event_raw_oom_score_adj_update;
if entry.is_null() {
return;
}
unsafe {
(*entry).pid = (*task).pid;
}
unsafe { memcpy(((*entry).comm.as_mut_ptr() as *mut u8) as *mut (), ((*task).comm.as_mut_ptr() as *mut u8) as *const (), 16u64) };
unsafe {
(*entry).oom_score_adj = (*(*task).signal).oom_score_adj;
}
unsafe { trace_event_buffer_commit(core::ptr::addr_of_mut!(fbuffer)) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_event_raw_event_oom_score_adj_update")]
extern "C" fn trace_event_raw_event_oom_score_adj_update(__data: *mut (), task: *mut task_struct) {
let mut __UNIQUE_ID_guard_815: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_815) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_trace_event_raw_event_oom_score_adj_update(__data, task);
}

fn do_trace_event_raw_event_reclaim_retry_zone(__data: *mut (), zoneref: *mut zoneref, order: i32, reclaimable: u64, available: u64, min_wmark: u64, no_progress_loops: i32, wmark_check: bool) {
let trace_file: *mut trace_event_file = __data as *mut trace_event_file;
let mut __data_offsets: trace_event_data_offsets_reclaim_retry_zone = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut fbuffer: trace_event_buffer = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fbuffer) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
let mut __data_size: i32 = 0;
if trace_trigger_soft_disabled(trace_file) {
return;
}
__data_size = trace_event_get_offsets_reclaim_retry_zone(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, zoneref, order, reclaimable, available, min_wmark, no_progress_loops, wmark_check);
let entry: *mut trace_event_raw_reclaim_retry_zone = (unsafe { trace_event_buffer_reserve(core::ptr::addr_of_mut!(fbuffer), trace_file, 56u64.wrapping_add(__data_size as u64)) }) as *mut trace_event_raw_reclaim_retry_zone;
if entry.is_null() {
return;
}
unsafe {
(*entry).node = zonelist_node_idx(zoneref as *const zoneref);
}
unsafe {
(*entry).zone_idx = zonelist_zone_idx(zoneref as *const zoneref);
}
unsafe {
(*entry).order = order;
}
unsafe {
(*entry).reclaimable = reclaimable;
}
unsafe {
(*entry).available = available;
}
unsafe {
(*entry).min_wmark = min_wmark;
}
unsafe {
(*entry).no_progress_loops = no_progress_loops;
}
unsafe {
(*entry).wmark_check = wmark_check;
}
unsafe { trace_event_buffer_commit(core::ptr::addr_of_mut!(fbuffer)) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_event_raw_event_reclaim_retry_zone")]
extern "C" fn trace_event_raw_event_reclaim_retry_zone(__data: *mut (), zoneref: *mut zoneref, order: i32, reclaimable: u64, available: u64, min_wmark: u64, no_progress_loops: i32, wmark_check: bool) {
let mut __UNIQUE_ID_guard_816: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_816) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_trace_event_raw_event_reclaim_retry_zone(__data, zoneref, order, reclaimable, available, min_wmark, no_progress_loops, wmark_check);
}

fn do_trace_event_raw_event_mark_victim(__data: *mut (), task: *mut task_struct, uid: u32) {
let trace_file: *mut trace_event_file = __data as *mut trace_event_file;
let mut __data_offsets: trace_event_data_offsets_mark_victim = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut fbuffer: trace_event_buffer = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fbuffer) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
let mut entry: *mut trace_event_raw_mark_victim = unsafe { core::mem::zeroed() };
let mut __data_size: i32 = 0;
if trace_trigger_soft_disabled(trace_file) {
return;
}
__data_size = trace_event_get_offsets_mark_victim(core::ptr::addr_of_mut!(__data_offsets), task, uid);
entry = (unsafe { trace_event_buffer_reserve(core::ptr::addr_of_mut!(fbuffer), trace_file, 72u64.wrapping_add(__data_size as u64)) }) as *mut trace_event_raw_mark_victim;
if entry.is_null() {
return;
}
__string__2!(comm, entry, __data_offsets);
unsafe {
(*entry).pid = (*task).pid;
}
loop {
let __str__: *mut u8 = __get_str__4!(entry);
let __len__: i32 = ((unsafe { (*entry).__data_loc_comm }) >> 16i32 & 65535).wrapping_sub(1) as i32;
let __v0: *const () = __data_offsets.comm_ptr_;
unsafe { memcpy(__str__ as *mut (), if !__v0.is_null() { __v0 } else { b"(null)\0".as_ptr() as *const () }, __len__ as u64) };
unsafe {
*__str__.wrapping_offset(__len__ as isize) = 0;
}
if true {
break;
}
}
unsafe {
(*entry).total_vm = PG_COUNT_TO_KB!(unsafe { (*(*task).mm).__slate_anon_0.total_vm });
}
unsafe {
(*entry).anon_rss = PG_COUNT_TO_KB!(get_mm_counter((*task).mm, 1i32));
}
unsafe {
(*entry).file_rss = PG_COUNT_TO_KB!(get_mm_counter((*task).mm, 0i32));
}
unsafe {
(*entry).shmem_rss = PG_COUNT_TO_KB!(get_mm_counter((*task).mm, 3i32));
}
unsafe {
(*entry).uid = uid;
}
unsafe {
(*entry).pgtables = mm_pgtables_bytes((*task).mm as *const mm_struct) >> 10i32;
}
unsafe {
(*entry).oom_score_adj = (*(*task).signal).oom_score_adj;
}
unsafe { trace_event_buffer_commit(core::ptr::addr_of_mut!(fbuffer)) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_event_raw_event_mark_victim")]
extern "C" fn trace_event_raw_event_mark_victim(__data: *mut (), task: *mut task_struct, uid: u32) {
let mut __UNIQUE_ID_guard_817: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_817) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_trace_event_raw_event_mark_victim(__data, task, uid);
}

fn do_trace_event_raw_event_wake_reaper(__data: *mut (), pid: i32) {
let trace_file: *mut trace_event_file = __data as *mut trace_event_file;
let mut __data_offsets: trace_event_data_offsets_wake_reaper = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut fbuffer: trace_event_buffer = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fbuffer) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
let mut __data_size: i32 = 0;
if trace_trigger_soft_disabled(trace_file) {
return;
}
__data_size = trace_event_get_offsets_wake_reaper(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, pid);
let entry: *mut trace_event_raw_wake_reaper = (unsafe { trace_event_buffer_reserve(core::ptr::addr_of_mut!(fbuffer), trace_file, 12u64.wrapping_add(__data_size as u64)) }) as *mut trace_event_raw_wake_reaper;
if entry.is_null() {
return;
}
unsafe {
(*entry).pid = pid;
}
unsafe { trace_event_buffer_commit(core::ptr::addr_of_mut!(fbuffer)) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_event_raw_event_wake_reaper")]
extern "C" fn trace_event_raw_event_wake_reaper(__data: *mut (), pid: i32) {
let mut __UNIQUE_ID_guard_818: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_818) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_trace_event_raw_event_wake_reaper(__data, pid);
}

fn do_trace_event_raw_event_start_task_reaping(__data: *mut (), pid: i32) {
let trace_file: *mut trace_event_file = __data as *mut trace_event_file;
let mut __data_offsets: trace_event_data_offsets_start_task_reaping = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut fbuffer: trace_event_buffer = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fbuffer) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
let mut __data_size: i32 = 0;
if trace_trigger_soft_disabled(trace_file) {
return;
}
__data_size = trace_event_get_offsets_start_task_reaping(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, pid);
let entry: *mut trace_event_raw_start_task_reaping = (unsafe { trace_event_buffer_reserve(core::ptr::addr_of_mut!(fbuffer), trace_file, 12u64.wrapping_add(__data_size as u64)) }) as *mut trace_event_raw_start_task_reaping;
if entry.is_null() {
return;
}
unsafe {
(*entry).pid = pid;
}
unsafe { trace_event_buffer_commit(core::ptr::addr_of_mut!(fbuffer)) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_event_raw_event_start_task_reaping")]
extern "C" fn trace_event_raw_event_start_task_reaping(__data: *mut (), pid: i32) {
let mut __UNIQUE_ID_guard_819: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_819) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_trace_event_raw_event_start_task_reaping(__data, pid);
}

fn do_trace_event_raw_event_finish_task_reaping(__data: *mut (), pid: i32) {
let trace_file: *mut trace_event_file = __data as *mut trace_event_file;
let mut __data_offsets: trace_event_data_offsets_finish_task_reaping = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut fbuffer: trace_event_buffer = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fbuffer) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
let mut __data_size: i32 = 0;
if trace_trigger_soft_disabled(trace_file) {
return;
}
__data_size = trace_event_get_offsets_finish_task_reaping(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, pid);
let entry: *mut trace_event_raw_finish_task_reaping = (unsafe { trace_event_buffer_reserve(core::ptr::addr_of_mut!(fbuffer), trace_file, 12u64.wrapping_add(__data_size as u64)) }) as *mut trace_event_raw_finish_task_reaping;
if entry.is_null() {
return;
}
unsafe {
(*entry).pid = pid;
}
unsafe { trace_event_buffer_commit(core::ptr::addr_of_mut!(fbuffer)) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_event_raw_event_finish_task_reaping")]
extern "C" fn trace_event_raw_event_finish_task_reaping(__data: *mut (), pid: i32) {
let mut __UNIQUE_ID_guard_820: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_820) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_trace_event_raw_event_finish_task_reaping(__data, pid);
}

fn do_trace_event_raw_event_skip_task_reaping(__data: *mut (), pid: i32) {
let trace_file: *mut trace_event_file = __data as *mut trace_event_file;
let mut __data_offsets: trace_event_data_offsets_skip_task_reaping = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut fbuffer: trace_event_buffer = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fbuffer) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
let mut __data_size: i32 = 0;
if trace_trigger_soft_disabled(trace_file) {
return;
}
__data_size = trace_event_get_offsets_skip_task_reaping(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, pid);
let entry: *mut trace_event_raw_skip_task_reaping = (unsafe { trace_event_buffer_reserve(core::ptr::addr_of_mut!(fbuffer), trace_file, 12u64.wrapping_add(__data_size as u64)) }) as *mut trace_event_raw_skip_task_reaping;
if entry.is_null() {
return;
}
unsafe {
(*entry).pid = pid;
}
unsafe { trace_event_buffer_commit(core::ptr::addr_of_mut!(fbuffer)) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_event_raw_event_skip_task_reaping")]
extern "C" fn trace_event_raw_event_skip_task_reaping(__data: *mut (), pid: i32) {
let mut __UNIQUE_ID_guard_821: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_821) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_trace_event_raw_event_skip_task_reaping(__data, pid);
}

fn do_trace_event_raw_event_compact_retry(__data: *mut (), order: i32, priority: u32, result: u32, retries: i32, max_retries: i32, ret: bool) {
let trace_file: *mut trace_event_file = __data as *mut trace_event_file;
let mut __data_offsets: trace_event_data_offsets_compact_retry = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut fbuffer: trace_event_buffer = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fbuffer) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
let mut __data_size: i32 = 0;
if trace_trigger_soft_disabled(trace_file) {
return;
}
__data_size = trace_event_get_offsets_compact_retry(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, order, priority, result, retries, max_retries, ret);
let entry: *mut trace_event_raw_compact_retry = (unsafe { trace_event_buffer_reserve(core::ptr::addr_of_mut!(fbuffer), trace_file, 32u64.wrapping_add(__data_size as u64)) }) as *mut trace_event_raw_compact_retry;
if entry.is_null() {
return;
}
unsafe {
(*entry).order = order;
}
unsafe {
(*entry).priority = (priority as u32) as i32;
}
unsafe {
(*entry).result = compact_result_to_feedback!(result);
}
unsafe {
(*entry).retries = retries;
}
unsafe {
(*entry).max_retries = max_retries;
}
unsafe {
(*entry).ret = ret;
}
unsafe { trace_event_buffer_commit(core::ptr::addr_of_mut!(fbuffer)) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.trace_event_raw_event_compact_retry")]
extern "C" fn trace_event_raw_event_compact_retry(__data: *mut (), order: i32, priority: u32, result: u32, retries: i32, max_retries: i32, ret: bool) {
let mut __UNIQUE_ID_guard_822: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_822) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_trace_event_raw_event_compact_retry(__data, order, priority, result, retries, max_retries, ret);
}

static mut print_fmt_oom_score_adj_update: __SlateAlign16<[u8; 76]> = __SlateAlign16([34u8, 112u8, 105u8, 100u8, 61u8, 37u8, 100u8, 32u8, 99u8, 111u8, 109u8, 109u8, 61u8, 37u8, 115u8, 32u8, 111u8, 111u8, 109u8, 95u8, 115u8, 99u8, 111u8, 114u8, 101u8, 95u8, 97u8, 100u8, 106u8, 61u8, 37u8, 104u8, 100u8, 34u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 112u8, 105u8, 100u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 99u8, 111u8, 109u8, 109u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 111u8, 111u8, 109u8, 95u8, 115u8, 99u8, 111u8, 114u8, 101u8, 95u8, 97u8, 100u8, 106u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".ref.data")]
static mut event_class_oom_score_adj_update: trace_event_class = {
let mut __t0: trace_event_class = unsafe { core::mem::zeroed() };
__t0.system = unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 };
__t0.probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut task_struct)>, *mut ()>(Some(trace_event_raw_event_oom_score_adj_update)) };
__t0.perf_probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut task_struct)>, *mut ()>(Some(perf_trace_oom_score_adj_update)) };
__t0.reg = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call, u32, *mut ()) -> i32>>(trace_event_reg as *const ()) };
__t0.fields_array = unsafe { core::ptr::addr_of_mut!(trace_event_fields_oom_score_adj_update.0) as *mut trace_event_fields };
__t0.fields = list_head { next: unsafe { core::ptr::addr_of_mut!(event_class_oom_score_adj_update.fields) }, prev: unsafe { core::ptr::addr_of_mut!(event_class_oom_score_adj_update.fields) } };
__t0.raw_init = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>>(trace_event_raw_init as *const ()) };
__t0
};

#[used(compiler)]
static mut event_oom_score_adj_update: trace_event_call = {
let mut __t0: trace_event_call = unsafe { core::mem::zeroed() };
__t0.class = unsafe { core::ptr::addr_of_mut!(event_class_oom_score_adj_update) };
__t0.__slate_anon_2 = __SlateRecord_trace_events_h_379 { tp: unsafe { core::ptr::addr_of_mut!(__tracepoint_oom_score_adj_update) } };
__t0.event = {
let mut __t1: trace_event = unsafe { core::mem::zeroed() };
__t1.funcs = unsafe { core::ptr::addr_of_mut!(trace_event_type_funcs_oom_score_adj_update) };
__t1
};
__t0.print_fmt = unsafe { core::ptr::addr_of_mut!(print_fmt_oom_score_adj_update.0) as *mut u8 };
__t0.flags = 8i32;
__t0
};

#[used(compiler)]
#[unsafe(link_section = "_ftrace_events")]
static mut __event_oom_score_adj_update: *mut trace_event_call = unsafe { core::ptr::addr_of_mut!(event_oom_score_adj_update) };

static mut print_fmt_reclaim_retry_zone: __SlateAlign16<[u8; 351]> = __SlateAlign16([34u8, 110u8, 111u8, 100u8, 101u8, 61u8, 37u8, 100u8, 32u8, 122u8, 111u8, 110u8, 101u8, 61u8, 37u8, 45u8, 56u8, 115u8, 32u8, 111u8, 114u8, 100u8, 101u8, 114u8, 61u8, 37u8, 100u8, 32u8, 114u8, 101u8, 99u8, 108u8, 97u8, 105u8, 109u8, 97u8, 98u8, 108u8, 101u8, 61u8, 37u8, 108u8, 117u8, 32u8, 97u8, 118u8, 97u8, 105u8, 108u8, 97u8, 98u8, 108u8, 101u8, 61u8, 37u8, 108u8, 117u8, 32u8, 109u8, 105u8, 110u8, 95u8, 119u8, 109u8, 97u8, 114u8, 107u8, 61u8, 37u8, 108u8, 117u8, 32u8, 110u8, 111u8, 95u8, 112u8, 114u8, 111u8, 103u8, 114u8, 101u8, 115u8, 115u8, 95u8, 108u8, 111u8, 111u8, 112u8, 115u8, 61u8, 37u8, 100u8, 32u8, 119u8, 109u8, 97u8, 114u8, 107u8, 95u8, 99u8, 104u8, 101u8, 99u8, 107u8, 61u8, 37u8, 100u8, 34u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 110u8, 111u8, 100u8, 101u8, 44u8, 32u8, 95u8, 95u8, 112u8, 114u8, 105u8, 110u8, 116u8, 95u8, 115u8, 121u8, 109u8, 98u8, 111u8, 108u8, 105u8, 99u8, 40u8, 82u8, 69u8, 67u8, 45u8, 62u8, 122u8, 111u8, 110u8, 101u8, 95u8, 105u8, 100u8, 120u8, 44u8, 32u8, 123u8, 90u8, 79u8, 78u8, 69u8, 95u8, 68u8, 77u8, 65u8, 44u8, 32u8, 34u8, 68u8, 77u8, 65u8, 34u8, 125u8, 44u8, 32u8, 123u8, 90u8, 79u8, 78u8, 69u8, 95u8, 68u8, 77u8, 65u8, 51u8, 50u8, 44u8, 32u8, 34u8, 68u8, 77u8, 65u8, 51u8, 50u8, 34u8, 125u8, 44u8, 32u8, 123u8, 90u8, 79u8, 78u8, 69u8, 95u8, 78u8, 79u8, 82u8, 77u8, 65u8, 76u8, 44u8, 32u8, 34u8, 78u8, 111u8, 114u8, 109u8, 97u8, 108u8, 34u8, 125u8, 44u8, 32u8, 123u8, 90u8, 79u8, 78u8, 69u8, 95u8, 77u8, 79u8, 86u8, 65u8, 66u8, 76u8, 69u8, 44u8, 32u8, 34u8, 77u8, 111u8, 118u8, 97u8, 98u8, 108u8, 101u8, 34u8, 125u8, 41u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 111u8, 114u8, 100u8, 101u8, 114u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 114u8, 101u8, 99u8, 108u8, 97u8, 105u8, 109u8, 97u8, 98u8, 108u8, 101u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 97u8, 118u8, 97u8, 105u8, 108u8, 97u8, 98u8, 108u8, 101u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 109u8, 105u8, 110u8, 95u8, 119u8, 109u8, 97u8, 114u8, 107u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 110u8, 111u8, 95u8, 112u8, 114u8, 111u8, 103u8, 114u8, 101u8, 115u8, 115u8, 95u8, 108u8, 111u8, 111u8, 112u8, 115u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 119u8, 109u8, 97u8, 114u8, 107u8, 95u8, 99u8, 104u8, 101u8, 99u8, 107u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".ref.data")]
static mut event_class_reclaim_retry_zone: trace_event_class = {
let mut __t0: trace_event_class = unsafe { core::mem::zeroed() };
__t0.system = unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 };
__t0.probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut zoneref, i32, u64, u64, u64, i32, bool)>, *mut ()>(Some(trace_event_raw_event_reclaim_retry_zone)) };
__t0.perf_probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut zoneref, i32, u64, u64, u64, i32, bool)>, *mut ()>(Some(perf_trace_reclaim_retry_zone)) };
__t0.reg = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call, u32, *mut ()) -> i32>>(trace_event_reg as *const ()) };
__t0.fields_array = unsafe { core::ptr::addr_of_mut!(trace_event_fields_reclaim_retry_zone.0) as *mut trace_event_fields };
__t0.fields = list_head { next: unsafe { core::ptr::addr_of_mut!(event_class_reclaim_retry_zone.fields) }, prev: unsafe { core::ptr::addr_of_mut!(event_class_reclaim_retry_zone.fields) } };
__t0.raw_init = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>>(trace_event_raw_init as *const ()) };
__t0
};

#[used(compiler)]
static mut event_reclaim_retry_zone: trace_event_call = {
let mut __t0: trace_event_call = unsafe { core::mem::zeroed() };
__t0.class = unsafe { core::ptr::addr_of_mut!(event_class_reclaim_retry_zone) };
__t0.__slate_anon_2 = __SlateRecord_trace_events_h_379 { tp: unsafe { core::ptr::addr_of_mut!(__tracepoint_reclaim_retry_zone) } };
__t0.event = {
let mut __t1: trace_event = unsafe { core::mem::zeroed() };
__t1.funcs = unsafe { core::ptr::addr_of_mut!(trace_event_type_funcs_reclaim_retry_zone) };
__t1
};
__t0.print_fmt = unsafe { core::ptr::addr_of_mut!(print_fmt_reclaim_retry_zone.0) as *mut u8 };
__t0.flags = 8i32;
__t0
};

#[used(compiler)]
#[unsafe(link_section = "_ftrace_events")]
static mut __event_reclaim_retry_zone: *mut trace_event_call = unsafe { core::ptr::addr_of_mut!(event_reclaim_retry_zone) };

static mut print_fmt_mark_victim: __SlateAlign16<[u8; 251]> = __SlateAlign16([34u8, 112u8, 105u8, 100u8, 61u8, 37u8, 100u8, 32u8, 99u8, 111u8, 109u8, 109u8, 61u8, 37u8, 115u8, 32u8, 116u8, 111u8, 116u8, 97u8, 108u8, 45u8, 118u8, 109u8, 61u8, 37u8, 108u8, 117u8, 107u8, 66u8, 32u8, 97u8, 110u8, 111u8, 110u8, 45u8, 114u8, 115u8, 115u8, 61u8, 37u8, 108u8, 117u8, 107u8, 66u8, 32u8, 102u8, 105u8, 108u8, 101u8, 45u8, 114u8, 115u8, 115u8, 58u8, 37u8, 108u8, 117u8, 107u8, 66u8, 32u8, 115u8, 104u8, 109u8, 101u8, 109u8, 45u8, 114u8, 115u8, 115u8, 58u8, 37u8, 108u8, 117u8, 107u8, 66u8, 32u8, 117u8, 105u8, 100u8, 61u8, 37u8, 117u8, 32u8, 112u8, 103u8, 116u8, 97u8, 98u8, 108u8, 101u8, 115u8, 61u8, 37u8, 108u8, 117u8, 107u8, 66u8, 32u8, 111u8, 111u8, 109u8, 95u8, 115u8, 99u8, 111u8, 114u8, 101u8, 95u8, 97u8, 100u8, 106u8, 61u8, 37u8, 104u8, 100u8, 34u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 112u8, 105u8, 100u8, 44u8, 32u8, 95u8, 95u8, 103u8, 101u8, 116u8, 95u8, 115u8, 116u8, 114u8, 40u8, 99u8, 111u8, 109u8, 109u8, 41u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 116u8, 111u8, 116u8, 97u8, 108u8, 95u8, 118u8, 109u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 97u8, 110u8, 111u8, 110u8, 95u8, 114u8, 115u8, 115u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 102u8, 105u8, 108u8, 101u8, 95u8, 114u8, 115u8, 115u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 115u8, 104u8, 109u8, 101u8, 109u8, 95u8, 114u8, 115u8, 115u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 117u8, 105u8, 100u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 112u8, 103u8, 116u8, 97u8, 98u8, 108u8, 101u8, 115u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 111u8, 111u8, 109u8, 95u8, 115u8, 99u8, 111u8, 114u8, 101u8, 95u8, 97u8, 100u8, 106u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".ref.data")]
static mut event_class_mark_victim: trace_event_class = {
let mut __t0: trace_event_class = unsafe { core::mem::zeroed() };
__t0.system = unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 };
__t0.probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut task_struct, u32)>, *mut ()>(Some(trace_event_raw_event_mark_victim)) };
__t0.perf_probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *mut task_struct, u32)>, *mut ()>(Some(perf_trace_mark_victim)) };
__t0.reg = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call, u32, *mut ()) -> i32>>(trace_event_reg as *const ()) };
__t0.fields_array = unsafe { core::ptr::addr_of_mut!(trace_event_fields_mark_victim.0) as *mut trace_event_fields };
__t0.fields = list_head { next: unsafe { core::ptr::addr_of_mut!(event_class_mark_victim.fields) }, prev: unsafe { core::ptr::addr_of_mut!(event_class_mark_victim.fields) } };
__t0.raw_init = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>>(trace_event_raw_init as *const ()) };
__t0
};

#[used(compiler)]
static mut event_mark_victim: trace_event_call = {
let mut __t0: trace_event_call = unsafe { core::mem::zeroed() };
__t0.class = unsafe { core::ptr::addr_of_mut!(event_class_mark_victim) };
__t0.__slate_anon_2 = __SlateRecord_trace_events_h_379 { tp: unsafe { core::ptr::addr_of_mut!(__tracepoint_mark_victim) } };
__t0.event = {
let mut __t1: trace_event = unsafe { core::mem::zeroed() };
__t1.funcs = unsafe { core::ptr::addr_of_mut!(trace_event_type_funcs_mark_victim) };
__t1
};
__t0.print_fmt = unsafe { core::ptr::addr_of_mut!(print_fmt_mark_victim.0) as *mut u8 };
__t0.flags = 8i32;
__t0
};

#[used(compiler)]
#[unsafe(link_section = "_ftrace_events")]
static mut __event_mark_victim: *mut trace_event_call = unsafe { core::ptr::addr_of_mut!(event_mark_victim) };

static mut print_fmt_wake_reaper: __SlateAlign16<[u8; 19]> = __SlateAlign16([34u8, 112u8, 105u8, 100u8, 61u8, 37u8, 100u8, 34u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 112u8, 105u8, 100u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".ref.data")]
static mut event_class_wake_reaper: trace_event_class = {
let mut __t0: trace_event_class = unsafe { core::mem::zeroed() };
__t0.system = unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 };
__t0.probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32)>, *mut ()>(Some(trace_event_raw_event_wake_reaper)) };
__t0.perf_probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32)>, *mut ()>(Some(perf_trace_wake_reaper)) };
__t0.reg = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call, u32, *mut ()) -> i32>>(trace_event_reg as *const ()) };
__t0.fields_array = unsafe { core::ptr::addr_of_mut!(trace_event_fields_wake_reaper.0) as *mut trace_event_fields };
__t0.fields = list_head { next: unsafe { core::ptr::addr_of_mut!(event_class_wake_reaper.fields) }, prev: unsafe { core::ptr::addr_of_mut!(event_class_wake_reaper.fields) } };
__t0.raw_init = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>>(trace_event_raw_init as *const ()) };
__t0
};

#[used(compiler)]
static mut event_wake_reaper: trace_event_call = {
let mut __t0: trace_event_call = unsafe { core::mem::zeroed() };
__t0.class = unsafe { core::ptr::addr_of_mut!(event_class_wake_reaper) };
__t0.__slate_anon_2 = __SlateRecord_trace_events_h_379 { tp: unsafe { core::ptr::addr_of_mut!(__tracepoint_wake_reaper) } };
__t0.event = {
let mut __t1: trace_event = unsafe { core::mem::zeroed() };
__t1.funcs = unsafe { core::ptr::addr_of_mut!(trace_event_type_funcs_wake_reaper) };
__t1
};
__t0.print_fmt = unsafe { core::ptr::addr_of_mut!(print_fmt_wake_reaper.0) as *mut u8 };
__t0.flags = 8i32;
__t0
};

#[used(compiler)]
#[unsafe(link_section = "_ftrace_events")]
static mut __event_wake_reaper: *mut trace_event_call = unsafe { core::ptr::addr_of_mut!(event_wake_reaper) };

static mut print_fmt_start_task_reaping: __SlateAlign16<[u8; 19]> = __SlateAlign16([34u8, 112u8, 105u8, 100u8, 61u8, 37u8, 100u8, 34u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 112u8, 105u8, 100u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".ref.data")]
static mut event_class_start_task_reaping: trace_event_class = {
let mut __t0: trace_event_class = unsafe { core::mem::zeroed() };
__t0.system = unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 };
__t0.probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32)>, *mut ()>(Some(trace_event_raw_event_start_task_reaping)) };
__t0.perf_probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32)>, *mut ()>(Some(perf_trace_start_task_reaping)) };
__t0.reg = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call, u32, *mut ()) -> i32>>(trace_event_reg as *const ()) };
__t0.fields_array = unsafe { core::ptr::addr_of_mut!(trace_event_fields_start_task_reaping.0) as *mut trace_event_fields };
__t0.fields = list_head { next: unsafe { core::ptr::addr_of_mut!(event_class_start_task_reaping.fields) }, prev: unsafe { core::ptr::addr_of_mut!(event_class_start_task_reaping.fields) } };
__t0.raw_init = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>>(trace_event_raw_init as *const ()) };
__t0
};

#[used(compiler)]
static mut event_start_task_reaping: trace_event_call = {
let mut __t0: trace_event_call = unsafe { core::mem::zeroed() };
__t0.class = unsafe { core::ptr::addr_of_mut!(event_class_start_task_reaping) };
__t0.__slate_anon_2 = __SlateRecord_trace_events_h_379 { tp: unsafe { core::ptr::addr_of_mut!(__tracepoint_start_task_reaping) } };
__t0.event = {
let mut __t1: trace_event = unsafe { core::mem::zeroed() };
__t1.funcs = unsafe { core::ptr::addr_of_mut!(trace_event_type_funcs_start_task_reaping) };
__t1
};
__t0.print_fmt = unsafe { core::ptr::addr_of_mut!(print_fmt_start_task_reaping.0) as *mut u8 };
__t0.flags = 8i32;
__t0
};

#[used(compiler)]
#[unsafe(link_section = "_ftrace_events")]
static mut __event_start_task_reaping: *mut trace_event_call = unsafe { core::ptr::addr_of_mut!(event_start_task_reaping) };

static mut print_fmt_finish_task_reaping: __SlateAlign16<[u8; 19]> = __SlateAlign16([34u8, 112u8, 105u8, 100u8, 61u8, 37u8, 100u8, 34u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 112u8, 105u8, 100u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".ref.data")]
static mut event_class_finish_task_reaping: trace_event_class = {
let mut __t0: trace_event_class = unsafe { core::mem::zeroed() };
__t0.system = unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 };
__t0.probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32)>, *mut ()>(Some(trace_event_raw_event_finish_task_reaping)) };
__t0.perf_probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32)>, *mut ()>(Some(perf_trace_finish_task_reaping)) };
__t0.reg = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call, u32, *mut ()) -> i32>>(trace_event_reg as *const ()) };
__t0.fields_array = unsafe { core::ptr::addr_of_mut!(trace_event_fields_finish_task_reaping.0) as *mut trace_event_fields };
__t0.fields = list_head { next: unsafe { core::ptr::addr_of_mut!(event_class_finish_task_reaping.fields) }, prev: unsafe { core::ptr::addr_of_mut!(event_class_finish_task_reaping.fields) } };
__t0.raw_init = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>>(trace_event_raw_init as *const ()) };
__t0
};

#[used(compiler)]
static mut event_finish_task_reaping: trace_event_call = {
let mut __t0: trace_event_call = unsafe { core::mem::zeroed() };
__t0.class = unsafe { core::ptr::addr_of_mut!(event_class_finish_task_reaping) };
__t0.__slate_anon_2 = __SlateRecord_trace_events_h_379 { tp: unsafe { core::ptr::addr_of_mut!(__tracepoint_finish_task_reaping) } };
__t0.event = {
let mut __t1: trace_event = unsafe { core::mem::zeroed() };
__t1.funcs = unsafe { core::ptr::addr_of_mut!(trace_event_type_funcs_finish_task_reaping) };
__t1
};
__t0.print_fmt = unsafe { core::ptr::addr_of_mut!(print_fmt_finish_task_reaping.0) as *mut u8 };
__t0.flags = 8i32;
__t0
};

#[used(compiler)]
#[unsafe(link_section = "_ftrace_events")]
static mut __event_finish_task_reaping: *mut trace_event_call = unsafe { core::ptr::addr_of_mut!(event_finish_task_reaping) };

static mut print_fmt_skip_task_reaping: __SlateAlign16<[u8; 19]> = __SlateAlign16([34u8, 112u8, 105u8, 100u8, 61u8, 37u8, 100u8, 34u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 112u8, 105u8, 100u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".ref.data")]
static mut event_class_skip_task_reaping: trace_event_class = {
let mut __t0: trace_event_class = unsafe { core::mem::zeroed() };
__t0.system = unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 };
__t0.probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32)>, *mut ()>(Some(trace_event_raw_event_skip_task_reaping)) };
__t0.perf_probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32)>, *mut ()>(Some(perf_trace_skip_task_reaping)) };
__t0.reg = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call, u32, *mut ()) -> i32>>(trace_event_reg as *const ()) };
__t0.fields_array = unsafe { core::ptr::addr_of_mut!(trace_event_fields_skip_task_reaping.0) as *mut trace_event_fields };
__t0.fields = list_head { next: unsafe { core::ptr::addr_of_mut!(event_class_skip_task_reaping.fields) }, prev: unsafe { core::ptr::addr_of_mut!(event_class_skip_task_reaping.fields) } };
__t0.raw_init = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>>(trace_event_raw_init as *const ()) };
__t0
};

#[used(compiler)]
static mut event_skip_task_reaping: trace_event_call = {
let mut __t0: trace_event_call = unsafe { core::mem::zeroed() };
__t0.class = unsafe { core::ptr::addr_of_mut!(event_class_skip_task_reaping) };
__t0.__slate_anon_2 = __SlateRecord_trace_events_h_379 { tp: unsafe { core::ptr::addr_of_mut!(__tracepoint_skip_task_reaping) } };
__t0.event = {
let mut __t1: trace_event = unsafe { core::mem::zeroed() };
__t1.funcs = unsafe { core::ptr::addr_of_mut!(trace_event_type_funcs_skip_task_reaping) };
__t1
};
__t0.print_fmt = unsafe { core::ptr::addr_of_mut!(print_fmt_skip_task_reaping.0) as *mut u8 };
__t0.flags = 8i32;
__t0
};

#[used(compiler)]
#[unsafe(link_section = "_ftrace_events")]
static mut __event_skip_task_reaping: *mut trace_event_call = unsafe { core::ptr::addr_of_mut!(event_skip_task_reaping) };

static mut print_fmt_compact_retry: __SlateAlign16<[u8; 404]> = __SlateAlign16([34u8, 111u8, 114u8, 100u8, 101u8, 114u8, 61u8, 37u8, 100u8, 32u8, 112u8, 114u8, 105u8, 111u8, 114u8, 105u8, 116u8, 121u8, 61u8, 37u8, 115u8, 32u8, 99u8, 111u8, 109u8, 112u8, 97u8, 99u8, 116u8, 105u8, 111u8, 110u8, 95u8, 114u8, 101u8, 115u8, 117u8, 108u8, 116u8, 61u8, 37u8, 115u8, 32u8, 114u8, 101u8, 116u8, 114u8, 105u8, 101u8, 115u8, 61u8, 37u8, 100u8, 32u8, 109u8, 97u8, 120u8, 95u8, 114u8, 101u8, 116u8, 114u8, 105u8, 101u8, 115u8, 61u8, 37u8, 100u8, 32u8, 115u8, 104u8, 111u8, 117u8, 108u8, 100u8, 95u8, 114u8, 101u8, 116u8, 114u8, 121u8, 61u8, 37u8, 100u8, 34u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 111u8, 114u8, 100u8, 101u8, 114u8, 44u8, 32u8, 95u8, 95u8, 112u8, 114u8, 105u8, 110u8, 116u8, 95u8, 115u8, 121u8, 109u8, 98u8, 111u8, 108u8, 105u8, 99u8, 40u8, 82u8, 69u8, 67u8, 45u8, 62u8, 112u8, 114u8, 105u8, 111u8, 114u8, 105u8, 116u8, 121u8, 44u8, 32u8, 123u8, 67u8, 79u8, 77u8, 80u8, 65u8, 67u8, 84u8, 95u8, 80u8, 82u8, 73u8, 79u8, 95u8, 83u8, 89u8, 78u8, 67u8, 95u8, 70u8, 85u8, 76u8, 76u8, 44u8, 32u8, 34u8, 67u8, 79u8, 77u8, 80u8, 65u8, 67u8, 84u8, 95u8, 80u8, 82u8, 73u8, 79u8, 95u8, 83u8, 89u8, 78u8, 67u8, 95u8, 70u8, 85u8, 76u8, 76u8, 34u8, 125u8, 44u8, 32u8, 123u8, 67u8, 79u8, 77u8, 80u8, 65u8, 67u8, 84u8, 95u8, 80u8, 82u8, 73u8, 79u8, 95u8, 83u8, 89u8, 78u8, 67u8, 95u8, 76u8, 73u8, 71u8, 72u8, 84u8, 44u8, 32u8, 34u8, 67u8, 79u8, 77u8, 80u8, 65u8, 67u8, 84u8, 95u8, 80u8, 82u8, 73u8, 79u8, 95u8, 83u8, 89u8, 78u8, 67u8, 95u8, 76u8, 73u8, 71u8, 72u8, 84u8, 34u8, 125u8, 44u8, 32u8, 123u8, 67u8, 79u8, 77u8, 80u8, 65u8, 67u8, 84u8, 95u8, 80u8, 82u8, 73u8, 79u8, 95u8, 65u8, 83u8, 89u8, 78u8, 67u8, 44u8, 32u8, 34u8, 67u8, 79u8, 77u8, 80u8, 65u8, 67u8, 84u8, 95u8, 80u8, 82u8, 73u8, 79u8, 95u8, 65u8, 83u8, 89u8, 78u8, 67u8, 34u8, 125u8, 41u8, 44u8, 32u8, 95u8, 95u8, 112u8, 114u8, 105u8, 110u8, 116u8, 95u8, 115u8, 121u8, 109u8, 98u8, 111u8, 108u8, 105u8, 99u8, 40u8, 82u8, 69u8, 67u8, 45u8, 62u8, 114u8, 101u8, 115u8, 117u8, 108u8, 116u8, 44u8, 32u8, 123u8, 49u8, 44u8, 32u8, 34u8, 102u8, 97u8, 105u8, 108u8, 101u8, 100u8, 34u8, 125u8, 44u8, 32u8, 123u8, 50u8, 44u8, 32u8, 34u8, 119u8, 105u8, 116u8, 104u8, 100u8, 114u8, 97u8, 119u8, 110u8, 34u8, 125u8, 44u8, 32u8, 123u8, 51u8, 44u8, 32u8, 34u8, 112u8, 114u8, 111u8, 103u8, 114u8, 101u8, 115u8, 115u8, 34u8, 125u8, 41u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 114u8, 101u8, 116u8, 114u8, 105u8, 101u8, 115u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 109u8, 97u8, 120u8, 95u8, 114u8, 101u8, 116u8, 114u8, 105u8, 101u8, 115u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 114u8, 101u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".ref.data")]
static mut event_class_compact_retry: trace_event_class = {
let mut __t0: trace_event_class = unsafe { core::mem::zeroed() };
__t0.system = unsafe { core::ptr::addr_of!(str__oom__trace_system_name) as *const u8 };
__t0.probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32, u32, u32, i32, i32, bool)>, *mut ()>(Some(trace_event_raw_event_compact_retry)) };
__t0.perf_probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), i32, u32, u32, i32, i32, bool)>, *mut ()>(Some(perf_trace_compact_retry)) };
__t0.reg = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call, u32, *mut ()) -> i32>>(trace_event_reg as *const ()) };
__t0.fields_array = unsafe { core::ptr::addr_of_mut!(trace_event_fields_compact_retry.0) as *mut trace_event_fields };
__t0.fields = list_head { next: unsafe { core::ptr::addr_of_mut!(event_class_compact_retry.fields) }, prev: unsafe { core::ptr::addr_of_mut!(event_class_compact_retry.fields) } };
__t0.raw_init = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>>(trace_event_raw_init as *const ()) };
__t0
};

#[used(compiler)]
static mut event_compact_retry: trace_event_call = {
let mut __t0: trace_event_call = unsafe { core::mem::zeroed() };
__t0.class = unsafe { core::ptr::addr_of_mut!(event_class_compact_retry) };
__t0.__slate_anon_2 = __SlateRecord_trace_events_h_379 { tp: unsafe { core::ptr::addr_of_mut!(__tracepoint_compact_retry) } };
__t0.event = {
let mut __t1: trace_event = unsafe { core::mem::zeroed() };
__t1.funcs = unsafe { core::ptr::addr_of_mut!(trace_event_type_funcs_compact_retry) };
__t1
};
__t0.print_fmt = unsafe { core::ptr::addr_of_mut!(print_fmt_compact_retry.0) as *mut u8 };
__t0.flags = 8i32;
__t0
};

#[used(compiler)]
#[unsafe(link_section = "_ftrace_events")]
static mut __event_compact_retry: *mut trace_event_call = unsafe { core::ptr::addr_of_mut!(event_compact_retry) };

fn do_perf_trace_oom_score_adj_update(__data: *mut (), task: *mut task_struct) {
let event_call: *mut trace_event_call = __data as *mut trace_event_call;
let mut __data_offsets: trace_event_data_offsets_oom_score_adj_update = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut __regs: *mut pt_regs = unsafe { core::mem::zeroed() };
let __count: u64 = 1;
let __task: *mut task_struct = core::ptr::null_mut::<task_struct>();
let mut __entry_size: i32 = 0;
let __data_size: i32 = trace_event_get_offsets_oom_score_adj_update(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, task);
let mut rctx: i32 = 0;
let mut __v0: *mut hlist_head;
__verify_pcpu_ptr!();
let mut __v2: *mut hlist_head;
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
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*event_call).perf_events }) as u64);
__v2 = tcp_ptr__ as *mut hlist_head;
__v0 = __v2;
let head: *mut hlist_head = __v0;
let __v6: bool = !bpf_prog_array_valid(event_call) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d(__task.is_null()) } && __task.is_null() && hlist_empty(head as *const hlist_head) != 0;
if __v6 {
return;
}
__entry_size = ALIGN!((__data_size as u64).wrapping_add(32).wrapping_add(core::mem::size_of::<u32>() as u64), core::mem::size_of::<u64>() as u64) as i32;
__entry_size = (__entry_size as u64).wrapping_sub(core::mem::size_of::<u32>() as u64) as i32;
let entry: *mut trace_event_raw_oom_score_adj_update = (unsafe { perf_trace_buf_alloc(__entry_size, core::ptr::addr_of_mut!(__regs), core::ptr::addr_of_mut!(rctx)) }) as *mut trace_event_raw_oom_score_adj_update;
if entry.is_null() {
return;
}
perf_fetch_caller_regs(__regs);
unsafe {
(*entry).pid = (*task).pid;
}
unsafe { memcpy(((*entry).comm.as_mut_ptr() as *mut u8) as *mut (), ((*task).comm.as_mut_ptr() as *mut u8) as *const (), 16u64) };
unsafe {
(*entry).oom_score_adj = (*(*task).signal).oom_score_adj;
}
unsafe { perf_trace_run_bpf_submit(entry as *mut (), __entry_size, rctx, event_call, __count, __regs, head, __task) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.perf_trace_oom_score_adj_update")]
extern "C" fn perf_trace_oom_score_adj_update(__data: *mut (), task: *mut task_struct) {
let mut __UNIQUE_ID_guard_823: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_823) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_perf_trace_oom_score_adj_update(__data, task);
}

fn do_perf_trace_reclaim_retry_zone(__data: *mut (), zoneref: *mut zoneref, order: i32, reclaimable: u64, available: u64, min_wmark: u64, no_progress_loops: i32, wmark_check: bool) {
let event_call: *mut trace_event_call = __data as *mut trace_event_call;
let mut __data_offsets: trace_event_data_offsets_reclaim_retry_zone = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut __regs: *mut pt_regs = unsafe { core::mem::zeroed() };
let __count: u64 = 1;
let __task: *mut task_struct = core::ptr::null_mut::<task_struct>();
let mut __entry_size: i32 = 0;
let __data_size: i32 = trace_event_get_offsets_reclaim_retry_zone(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, zoneref, order, reclaimable, available, min_wmark, no_progress_loops, wmark_check);
let mut rctx: i32 = 0;
let mut __v0: *mut hlist_head;
__verify_pcpu_ptr!();
let mut __v2: *mut hlist_head;
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
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*event_call).perf_events }) as u64);
__v2 = tcp_ptr__ as *mut hlist_head;
__v0 = __v2;
let head: *mut hlist_head = __v0;
let __v6: bool = !bpf_prog_array_valid(event_call) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d(__task.is_null()) } && __task.is_null() && hlist_empty(head as *const hlist_head) != 0;
if __v6 {
return;
}
__entry_size = ALIGN!((__data_size as u64).wrapping_add(56).wrapping_add(core::mem::size_of::<u32>() as u64), core::mem::size_of::<u64>() as u64) as i32;
__entry_size = (__entry_size as u64).wrapping_sub(core::mem::size_of::<u32>() as u64) as i32;
let entry: *mut trace_event_raw_reclaim_retry_zone = (unsafe { perf_trace_buf_alloc(__entry_size, core::ptr::addr_of_mut!(__regs), core::ptr::addr_of_mut!(rctx)) }) as *mut trace_event_raw_reclaim_retry_zone;
if entry.is_null() {
return;
}
perf_fetch_caller_regs(__regs);
unsafe {
(*entry).node = zonelist_node_idx(zoneref as *const zoneref);
}
unsafe {
(*entry).zone_idx = zonelist_zone_idx(zoneref as *const zoneref);
}
unsafe {
(*entry).order = order;
}
unsafe {
(*entry).reclaimable = reclaimable;
}
unsafe {
(*entry).available = available;
}
unsafe {
(*entry).min_wmark = min_wmark;
}
unsafe {
(*entry).no_progress_loops = no_progress_loops;
}
unsafe {
(*entry).wmark_check = wmark_check;
}
unsafe { perf_trace_run_bpf_submit(entry as *mut (), __entry_size, rctx, event_call, __count, __regs, head, __task) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.perf_trace_reclaim_retry_zone")]
extern "C" fn perf_trace_reclaim_retry_zone(__data: *mut (), zoneref: *mut zoneref, order: i32, reclaimable: u64, available: u64, min_wmark: u64, no_progress_loops: i32, wmark_check: bool) {
let mut __UNIQUE_ID_guard_824: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_824) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_perf_trace_reclaim_retry_zone(__data, zoneref, order, reclaimable, available, min_wmark, no_progress_loops, wmark_check);
}

fn do_perf_trace_mark_victim(__data: *mut (), task: *mut task_struct, uid: u32) {
let event_call: *mut trace_event_call = __data as *mut trace_event_call;
let mut __data_offsets: trace_event_data_offsets_mark_victim = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut entry: *mut trace_event_raw_mark_victim = unsafe { core::mem::zeroed() };
let mut __regs: *mut pt_regs = unsafe { core::mem::zeroed() };
let __count: u64 = 1;
let __task: *mut task_struct = core::ptr::null_mut::<task_struct>();
let mut __entry_size: i32 = 0;
let __data_size: i32 = trace_event_get_offsets_mark_victim(core::ptr::addr_of_mut!(__data_offsets), task, uid);
let mut rctx: i32 = 0;
let mut __v0: *mut hlist_head;
__verify_pcpu_ptr!();
let mut __v2: *mut hlist_head;
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
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*event_call).perf_events }) as u64);
__v2 = tcp_ptr__ as *mut hlist_head;
__v0 = __v2;
let head: *mut hlist_head = __v0;
let __v6: bool = !bpf_prog_array_valid(event_call) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d(__task.is_null()) } && __task.is_null() && hlist_empty(head as *const hlist_head) != 0;
if __v6 {
return;
}
__entry_size = ALIGN!((__data_size as u64).wrapping_add(72).wrapping_add(core::mem::size_of::<u32>() as u64), core::mem::size_of::<u64>() as u64) as i32;
__entry_size = (__entry_size as u64).wrapping_sub(core::mem::size_of::<u32>() as u64) as i32;
entry = (unsafe { perf_trace_buf_alloc(__entry_size, core::ptr::addr_of_mut!(__regs), core::ptr::addr_of_mut!(rctx)) }) as *mut trace_event_raw_mark_victim;
if entry.is_null() {
return;
}
perf_fetch_caller_regs(__regs);
__string__2!(comm, entry, __data_offsets);
unsafe {
(*entry).pid = (*task).pid;
}
loop {
let __str__: *mut u8 = __get_str__4!(entry);
let __len__: i32 = ((unsafe { (*entry).__data_loc_comm }) >> 16i32 & 65535).wrapping_sub(1) as i32;
let __v9: *const () = __data_offsets.comm_ptr_;
unsafe { memcpy(__str__ as *mut (), if !__v9.is_null() { __v9 } else { b"(null)\0".as_ptr() as *const () }, __len__ as u64) };
unsafe {
*__str__.wrapping_offset(__len__ as isize) = 0;
}
if true {
break;
}
}
unsafe {
(*entry).total_vm = PG_COUNT_TO_KB!(unsafe { (*(*task).mm).__slate_anon_0.total_vm });
}
unsafe {
(*entry).anon_rss = PG_COUNT_TO_KB!(get_mm_counter((*task).mm, 1i32));
}
unsafe {
(*entry).file_rss = PG_COUNT_TO_KB!(get_mm_counter((*task).mm, 0i32));
}
unsafe {
(*entry).shmem_rss = PG_COUNT_TO_KB!(get_mm_counter((*task).mm, 3i32));
}
unsafe {
(*entry).uid = uid;
}
unsafe {
(*entry).pgtables = mm_pgtables_bytes((*task).mm as *const mm_struct) >> 10i32;
}
unsafe {
(*entry).oom_score_adj = (*(*task).signal).oom_score_adj;
}
unsafe { perf_trace_run_bpf_submit(entry as *mut (), __entry_size, rctx, event_call, __count, __regs, head, __task) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.perf_trace_mark_victim")]
extern "C" fn perf_trace_mark_victim(__data: *mut (), task: *mut task_struct, uid: u32) {
let mut __UNIQUE_ID_guard_825: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_825) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_perf_trace_mark_victim(__data, task, uid);
}

fn do_perf_trace_wake_reaper(__data: *mut (), pid: i32) {
let event_call: *mut trace_event_call = __data as *mut trace_event_call;
let mut __data_offsets: trace_event_data_offsets_wake_reaper = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut __regs: *mut pt_regs = unsafe { core::mem::zeroed() };
let __count: u64 = 1;
let __task: *mut task_struct = core::ptr::null_mut::<task_struct>();
let mut __entry_size: i32 = 0;
let __data_size: i32 = trace_event_get_offsets_wake_reaper(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, pid);
let mut rctx: i32 = 0;
let mut __v0: *mut hlist_head;
__verify_pcpu_ptr!();
let mut __v2: *mut hlist_head;
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
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*event_call).perf_events }) as u64);
__v2 = tcp_ptr__ as *mut hlist_head;
__v0 = __v2;
let head: *mut hlist_head = __v0;
let __v6: bool = !bpf_prog_array_valid(event_call) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d(__task.is_null()) } && __task.is_null() && hlist_empty(head as *const hlist_head) != 0;
if __v6 {
return;
}
__entry_size = ALIGN!((__data_size as u64).wrapping_add(12).wrapping_add(core::mem::size_of::<u32>() as u64), core::mem::size_of::<u64>() as u64) as i32;
__entry_size = (__entry_size as u64).wrapping_sub(core::mem::size_of::<u32>() as u64) as i32;
let entry: *mut trace_event_raw_wake_reaper = (unsafe { perf_trace_buf_alloc(__entry_size, core::ptr::addr_of_mut!(__regs), core::ptr::addr_of_mut!(rctx)) }) as *mut trace_event_raw_wake_reaper;
if entry.is_null() {
return;
}
perf_fetch_caller_regs(__regs);
unsafe {
(*entry).pid = pid;
}
unsafe { perf_trace_run_bpf_submit(entry as *mut (), __entry_size, rctx, event_call, __count, __regs, head, __task) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.perf_trace_wake_reaper")]
extern "C" fn perf_trace_wake_reaper(__data: *mut (), pid: i32) {
let mut __UNIQUE_ID_guard_826: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_826) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_perf_trace_wake_reaper(__data, pid);
}

fn do_perf_trace_start_task_reaping(__data: *mut (), pid: i32) {
let event_call: *mut trace_event_call = __data as *mut trace_event_call;
let mut __data_offsets: trace_event_data_offsets_start_task_reaping = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut __regs: *mut pt_regs = unsafe { core::mem::zeroed() };
let __count: u64 = 1;
let __task: *mut task_struct = core::ptr::null_mut::<task_struct>();
let mut __entry_size: i32 = 0;
let __data_size: i32 = trace_event_get_offsets_start_task_reaping(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, pid);
let mut rctx: i32 = 0;
let mut __v0: *mut hlist_head;
__verify_pcpu_ptr!();
let mut __v2: *mut hlist_head;
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
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*event_call).perf_events }) as u64);
__v2 = tcp_ptr__ as *mut hlist_head;
__v0 = __v2;
let head: *mut hlist_head = __v0;
let __v6: bool = !bpf_prog_array_valid(event_call) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d(__task.is_null()) } && __task.is_null() && hlist_empty(head as *const hlist_head) != 0;
if __v6 {
return;
}
__entry_size = ALIGN!((__data_size as u64).wrapping_add(12).wrapping_add(core::mem::size_of::<u32>() as u64), core::mem::size_of::<u64>() as u64) as i32;
__entry_size = (__entry_size as u64).wrapping_sub(core::mem::size_of::<u32>() as u64) as i32;
let entry: *mut trace_event_raw_start_task_reaping = (unsafe { perf_trace_buf_alloc(__entry_size, core::ptr::addr_of_mut!(__regs), core::ptr::addr_of_mut!(rctx)) }) as *mut trace_event_raw_start_task_reaping;
if entry.is_null() {
return;
}
perf_fetch_caller_regs(__regs);
unsafe {
(*entry).pid = pid;
}
unsafe { perf_trace_run_bpf_submit(entry as *mut (), __entry_size, rctx, event_call, __count, __regs, head, __task) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.perf_trace_start_task_reaping")]
extern "C" fn perf_trace_start_task_reaping(__data: *mut (), pid: i32) {
let mut __UNIQUE_ID_guard_827: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_827) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_perf_trace_start_task_reaping(__data, pid);
}

fn do_perf_trace_finish_task_reaping(__data: *mut (), pid: i32) {
let event_call: *mut trace_event_call = __data as *mut trace_event_call;
let mut __data_offsets: trace_event_data_offsets_finish_task_reaping = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut __regs: *mut pt_regs = unsafe { core::mem::zeroed() };
let __count: u64 = 1;
let __task: *mut task_struct = core::ptr::null_mut::<task_struct>();
let mut __entry_size: i32 = 0;
let __data_size: i32 = trace_event_get_offsets_finish_task_reaping(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, pid);
let mut rctx: i32 = 0;
let mut __v0: *mut hlist_head;
__verify_pcpu_ptr!();
let mut __v2: *mut hlist_head;
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
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*event_call).perf_events }) as u64);
__v2 = tcp_ptr__ as *mut hlist_head;
__v0 = __v2;
let head: *mut hlist_head = __v0;
let __v6: bool = !bpf_prog_array_valid(event_call) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d(__task.is_null()) } && __task.is_null() && hlist_empty(head as *const hlist_head) != 0;
if __v6 {
return;
}
__entry_size = ALIGN!((__data_size as u64).wrapping_add(12).wrapping_add(core::mem::size_of::<u32>() as u64), core::mem::size_of::<u64>() as u64) as i32;
__entry_size = (__entry_size as u64).wrapping_sub(core::mem::size_of::<u32>() as u64) as i32;
let entry: *mut trace_event_raw_finish_task_reaping = (unsafe { perf_trace_buf_alloc(__entry_size, core::ptr::addr_of_mut!(__regs), core::ptr::addr_of_mut!(rctx)) }) as *mut trace_event_raw_finish_task_reaping;
if entry.is_null() {
return;
}
perf_fetch_caller_regs(__regs);
unsafe {
(*entry).pid = pid;
}
unsafe { perf_trace_run_bpf_submit(entry as *mut (), __entry_size, rctx, event_call, __count, __regs, head, __task) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.perf_trace_finish_task_reaping")]
extern "C" fn perf_trace_finish_task_reaping(__data: *mut (), pid: i32) {
let mut __UNIQUE_ID_guard_828: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_828) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_perf_trace_finish_task_reaping(__data, pid);
}

fn do_perf_trace_skip_task_reaping(__data: *mut (), pid: i32) {
let event_call: *mut trace_event_call = __data as *mut trace_event_call;
let mut __data_offsets: trace_event_data_offsets_skip_task_reaping = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut __regs: *mut pt_regs = unsafe { core::mem::zeroed() };
let __count: u64 = 1;
let __task: *mut task_struct = core::ptr::null_mut::<task_struct>();
let mut __entry_size: i32 = 0;
let __data_size: i32 = trace_event_get_offsets_skip_task_reaping(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, pid);
let mut rctx: i32 = 0;
let mut __v0: *mut hlist_head;
__verify_pcpu_ptr!();
let mut __v2: *mut hlist_head;
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
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*event_call).perf_events }) as u64);
__v2 = tcp_ptr__ as *mut hlist_head;
__v0 = __v2;
let head: *mut hlist_head = __v0;
let __v6: bool = !bpf_prog_array_valid(event_call) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d(__task.is_null()) } && __task.is_null() && hlist_empty(head as *const hlist_head) != 0;
if __v6 {
return;
}
__entry_size = ALIGN!((__data_size as u64).wrapping_add(12).wrapping_add(core::mem::size_of::<u32>() as u64), core::mem::size_of::<u64>() as u64) as i32;
__entry_size = (__entry_size as u64).wrapping_sub(core::mem::size_of::<u32>() as u64) as i32;
let entry: *mut trace_event_raw_skip_task_reaping = (unsafe { perf_trace_buf_alloc(__entry_size, core::ptr::addr_of_mut!(__regs), core::ptr::addr_of_mut!(rctx)) }) as *mut trace_event_raw_skip_task_reaping;
if entry.is_null() {
return;
}
perf_fetch_caller_regs(__regs);
unsafe {
(*entry).pid = pid;
}
unsafe { perf_trace_run_bpf_submit(entry as *mut (), __entry_size, rctx, event_call, __count, __regs, head, __task) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.perf_trace_skip_task_reaping")]
extern "C" fn perf_trace_skip_task_reaping(__data: *mut (), pid: i32) {
let mut __UNIQUE_ID_guard_829: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_829) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_perf_trace_skip_task_reaping(__data, pid);
}

fn do_perf_trace_compact_retry(__data: *mut (), order: i32, priority: u32, result: u32, retries: i32, max_retries: i32, ret: bool) {
let event_call: *mut trace_event_call = __data as *mut trace_event_call;
let mut __data_offsets: trace_event_data_offsets_compact_retry = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut __regs: *mut pt_regs = unsafe { core::mem::zeroed() };
let __count: u64 = 1;
let __task: *mut task_struct = core::ptr::null_mut::<task_struct>();
let mut __entry_size: i32 = 0;
let __data_size: i32 = trace_event_get_offsets_compact_retry(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, order, priority, result, retries, max_retries, ret);
let mut rctx: i32 = 0;
let mut __v0: *mut hlist_head;
__verify_pcpu_ptr!();
let mut __v2: *mut hlist_head;
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
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*event_call).perf_events }) as u64);
__v2 = tcp_ptr__ as *mut hlist_head;
__v0 = __v2;
let head: *mut hlist_head = __v0;
let __v6: bool = !bpf_prog_array_valid(event_call) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d(__task.is_null()) } && __task.is_null() && hlist_empty(head as *const hlist_head) != 0;
if __v6 {
return;
}
__entry_size = ALIGN!((__data_size as u64).wrapping_add(32).wrapping_add(core::mem::size_of::<u32>() as u64), core::mem::size_of::<u64>() as u64) as i32;
__entry_size = (__entry_size as u64).wrapping_sub(core::mem::size_of::<u32>() as u64) as i32;
let entry: *mut trace_event_raw_compact_retry = (unsafe { perf_trace_buf_alloc(__entry_size, core::ptr::addr_of_mut!(__regs), core::ptr::addr_of_mut!(rctx)) }) as *mut trace_event_raw_compact_retry;
if entry.is_null() {
return;
}
perf_fetch_caller_regs(__regs);
unsafe {
(*entry).order = order;
}
unsafe {
(*entry).priority = (priority as u32) as i32;
}
unsafe {
(*entry).result = compact_result_to_feedback!(result);
}
unsafe {
(*entry).retries = retries;
}
unsafe {
(*entry).max_retries = max_retries;
}
unsafe {
(*entry).ret = ret;
}
unsafe { perf_trace_run_bpf_submit(entry as *mut (), __entry_size, rctx, event_call, __count, __regs, head, __task) };
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.perf_trace_compact_retry")]
extern "C" fn perf_trace_compact_retry(__data: *mut (), order: i32, priority: u32, result: u32, retries: i32, max_retries: i32, ret: bool) {
let mut __UNIQUE_ID_guard_830: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_830) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_perf_trace_compact_retry(__data, order, priority, result, retries, max_retries, ret);
}

static mut sysctl_panic_on_oom: i32 = unsafe { core::mem::zeroed() };

static mut sysctl_oom_kill_allocating_task: i32 = unsafe { core::mem::zeroed() };

static mut sysctl_oom_dump_tasks: i32 = 1;

/// Serializes oom killer invocations (out_of_memory()) from all contexts to
/// prevent from over eager oom killing (e.g. when the oom killer is invoked
/// from different domains).
///
/// oom_killer_disable() relies on this lock to stabilize oom_killer_disabled
/// and mark_oom_victim
#[unsafe(no_mangle)]
static mut oom_lock: mutex = {
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

/// Serializes oom_score_adj and oom_score_adj_min updates
#[unsafe(no_mangle)]
static mut oom_adj_mutex: mutex = {
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

fn is_memcg_oom(oc: *mut oom_control) -> bool {
!unsafe { (*oc).memcg }.is_null()
}

// oom_cpuset_eligible() - check task eligibility for kill
// @start: task struct of which task to consider
// @oc: pointer to struct oom_control
//
// Task eligibility is determined by whether or not a candidate task, @tsk,
// shares the same mempolicy nodes as current if it is bound by such a policy
// and whether or not it has the same set of allowed cpuset nodes.
//
// This function is assuming oom-killer context and 'current' has triggered
// the oom-killer.
fn oom_cpuset_eligible(start: *mut task_struct, oc: *mut oom_control) -> bool {
let mut tsk: *mut task_struct = unsafe { core::mem::zeroed() };
let mut ret: bool = false;
let mask: *const __SlateRecord_nodemask_types_h_17 = unsafe { (*oc).nodemask };
rcu_read_lock();
let mut __v0: *mut task_struct;
let mut __v1: *mut list_head;
loop {
if !(((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_831() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*start).signal).thread_head)).next) as *const *mut list_head) };
__v0 = ((unsafe { (__v1 as *const u8).wrapping_offset(-(offsetof__6!() as isize)) }) as *mut ()) as *mut task_struct;
tsk = __v0;
while (unsafe { core::ptr::addr_of_mut!((*tsk).thread_node) }) != unsafe { core::ptr::addr_of_mut!((*(*start).signal).thread_head) } {
if !mask.is_null() {
// If this is a mempolicy constrained oom, tsk's
// cpuset is irrelevant.  Only return true if its
// mempolicy intersects current, otherwise it may be
// needlessly killed.
ret = unsafe { mempolicy_in_oom_domain(tsk, mask) };
} else {
// This is not a mempolicy constrained oom, so only
// check the mems of tsk's cpuset.
ret = (unsafe { cpuset_mems_allowed_intersects(get_current() as *const task_struct, tsk as *const task_struct) }) != 0;
}
if ret {
break;
}
let mut __v5: *mut task_struct;
let mut __v6: *mut list_head;
loop {
if !(((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_832() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*tsk).thread_node.next) as *const *mut list_head) };
__v5 = ((unsafe { (__v6 as *const u8).wrapping_offset(-(offsetof__6!() as isize)) }) as *mut ()) as *mut task_struct;
tsk = __v5;
}
rcu_read_unlock();
ret
}

// The process p may have detached its own ->mm while exiting or through
// kthread_use_mm(), but one or more of its subthreads may still have a valid
// pointer.  Return p, or any of its subthreads with a valid ->mm, with
// task_lock() held.
/// order == -1 means the oom kill is required by sysrq, otherwise only
/// for display purposes.
fn is_sysrq_oom(oc: *mut oom_control) -> bool {
(unsafe { (*oc).order }) == 1i32.wrapping_neg()
}

/// return true if the task is not adequate as candidate victim task.
fn oom_unkillable_task(p: *mut task_struct) -> bool {
if is_global_init(p) != 0 {
return true;
}
if (unsafe { (*p).flags }) & 2097152 != 0 {
return true;
}
false
}

/// Check whether unreclaimable slab amount is greater than
/// all user memory(LRU pages).
/// dump_unreclaimable_slab() could help in the case that
/// oom due to too much unreclaimable slab used by kernel.
fn should_dump_unreclaim_slab() -> bool {
let nr_lru: u64 = global_node_page_state(1u32).wrapping_add(global_node_page_state(0u32)).wrapping_add(global_node_page_state(3u32)).wrapping_add(global_node_page_state(2u32)).wrapping_add(global_node_page_state(7u32)).wrapping_add(global_node_page_state(8u32)).wrapping_add(global_node_page_state(4u32));
global_node_page_state_pages(6u32) > nr_lru
}

/// oom_badness - heuristic function to determine which candidate task to kill
/// @p: task struct of which task we should calculate
/// @totalpages: total present RAM allowed for page allocation
///
/// The heuristic for determining which task to kill is made to be as simple and
/// predictable as possible.  The goal is to return the highest value for the
/// task consuming the most memory to avoid subsequent oom failures.
#[unsafe(no_mangle)]
extern "C" fn oom_badness(mut p: *mut task_struct, totalpages: u64) -> i64 {
let mut points: i64 = 0;
let mut adj: i64 = 0;
if oom_unkillable_task(p) {
return 9223372036854775807i64.wrapping_neg().wrapping_sub(1);
}
p = find_lock_task_mm(p);
if p.is_null() {
return 9223372036854775807i64.wrapping_neg().wrapping_sub(1);
}
// Do not even consider tasks which are explicitly marked oom
// unkillable or have been already oom reaped or the are in
// the middle of vfork
adj = (unsafe { (*(*p).signal).oom_score_adj }) as i64;
let __v0: bool = adj == (1000i32.wrapping_neg() as i64) || mm_flags_test(21i32, (unsafe { (*p).mm }) as *const mm_struct);
let __v1: bool = __v0 || in_vfork(p);
if __v1 {
task_unlock(p);
return 9223372036854775807i64.wrapping_neg().wrapping_sub(1);
}
// The baseline for the badness score is the proportion of RAM that each
// task's rss, pagetable and swap space use.
points = get_mm_rss_sum(unsafe { (*p).mm }).wrapping_add(get_mm_counter_sum(unsafe { (*p).mm }, 2i32)).wrapping_add(mm_pgtables_bytes((unsafe { (*p).mm }) as *const mm_struct) / (_AC!() << 12i32)) as i64;
task_unlock(p);
// Normalize to oom_score_adj units
adj = (adj as u64).wrapping_mul(totalpages / 1000) as i64;
points = points.wrapping_add(adj);
points
}

static oom_constraint_text: __SlateAlign16<[*const u8; 4]> = __SlateAlign16([(b"CONSTRAINT_NONE\0".as_ptr() as *mut u8) as *const u8, (b"CONSTRAINT_CPUSET\0".as_ptr() as *mut u8) as *const u8, (b"CONSTRAINT_MEMORY_POLICY\0".as_ptr() as *mut u8) as *const u8, (b"CONSTRAINT_MEMCG\0".as_ptr() as *mut u8) as *const u8]);

/// Determine the type of allocation constraint.
fn constrained_alloc(oc: *mut oom_control) -> u32 {
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
let mut z: *mut zoneref = unsafe { core::mem::zeroed() };
let highest_zoneidx: u32 = gfp_zone(unsafe { (*oc).gfp_mask });
let mut cpuset_limited: bool = false;
let mut nid: i32 = 0;
if is_memcg_oom(oc) {
let __v0: u64 = mem_cgroup_get_max(unsafe { (*oc).memcg });
unsafe {
(*oc).totalpages = if __v0 != 0 { __v0 } else { 1 };
}
return 3;
}
// Default to all available memory
unsafe {
(*oc).totalpages = totalram_pages().wrapping_add(total_swap_pages as u64);
}
if unsafe { (*oc).zonelist }.is_null() {
return 0;
}
// Reach here only when __GFP_NOFAIL is used. So, we should avoid
// to kill current.We have to random task kill in this case.
// Hopefully, CONSTRAINT_THISNODE...but no way to handle it, now.
if (unsafe { (*oc).gfp_mask }) & (BIT!(21i32) as u32) != 0 {
return 0;
}
// This is not a __GFP_THISNODE allocation, so a truncated nodemask in
// the page allocator means a mempolicy is in effect.  Cpuset policy
// is enforced in get_page_from_freelist().
let __v1: bool = !unsafe { (*oc).nodemask }.is_null() && !nodes_subset__2!(node_states[3], *unsafe { (*oc).nodemask });
if __v1 {
unsafe {
(*oc).totalpages = total_swap_pages as u64;
}
nid = first_node__3!(*unsafe { (*oc).nodemask }) as i32;
while nid < 64i32 {
let __v3: u64 = unsafe { (*oc).totalpages };
let __v4: u64 = __v3.wrapping_add(node_present_pages!(nid));
unsafe {
(*oc).totalpages = __v4;
}
nid = next_node__3!(nid, *unsafe { (*oc).nodemask }) as i32;
}
return 2;
}
// Check this allocation failure is caused by cpuset's wall function
z = first_zones_zonelist(unsafe { (*oc).zonelist }, highest_zoneidx, unsafe { (*oc).nodemask });
zone = zonelist_zone(z);
while !zone.is_null() {
if !cpuset_zone_allowed(zone, unsafe { (*oc).gfp_mask }) {
cpuset_limited = true;
}
let __v8: *mut zoneref = z;
let __v9: *mut zoneref = __v8.wrapping_offset(1 as isize);
z = __v9;
z = next_zones_zonelist(__v9, highest_zoneidx, unsafe { (*oc).nodemask });
zone = zonelist_zone(z);
}
if cpuset_limited {
unsafe {
(*oc).totalpages = total_swap_pages as u64;
}
nid = first_node!((*get_current()).mems_allowed) as i32;
while nid < 64i32 {
let __v12: u64 = unsafe { (*oc).totalpages };
let __v13: u64 = __v12.wrapping_add(node_present_pages!(nid));
unsafe {
(*oc).totalpages = __v13;
}
nid = next_node!(nid, (*get_current()).mems_allowed) as i32;
}
return 1;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.oom_evaluate_task")]
extern "C" fn oom_evaluate_task(task: *mut task_struct, arg: *mut ()) -> i32 {
// This task already has access to memory reserves and is being killed.
// Don't allow any other task to have access to the reserves unless
// the task has MMF_OOM_SKIP because chances that it would release
// any memory is quite low.
let mut __v0: bool = unsafe { core::mem::zeroed() };
// p may not have freeable memory in nodemask
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut points: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i64 = core::ptr::addr_of_mut!(points) as *mut i64;
let mut oc: *mut oom_control = unsafe { core::mem::zeroed() };
unsafe {
'__join_3: {
oc = arg as *mut oom_control;
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if !oom_unkillable_task(task) {
if !is_memcg_oom(oc) {
__v1 = !oom_cpuset_eligible(task, oc);
} else {
__v1 = false;
}
if !__v1 {
if !is_sysrq_oom(oc) {
__v0 = tsk_is_oom_victim(task);
} else {
__v0 = false;
}
if __v0 {
if !mm_flags_test(21i32, (*(*task).signal).oom_mm as *const mm_struct) {
if !unsafe { (*oc).chosen }.is_null() {
put_task_struct((*oc).chosen);
}
(*oc).chosen = (1u64.wrapping_neg() as *mut ()) as *mut task_struct;
return 1;
}
} else {
// If task is allocating a lot of memory and has been marked to be
// killed first if it triggers an oom, then select it.
if oom_task_origin(task as *const task_struct) {
*__slate_slot_2 = 9223372036854775807i64;
} else {
*__slate_slot_2 = oom_badness(task, (*oc).totalpages);
if *__slate_slot_2 == 9223372036854775807i64.wrapping_neg().wrapping_sub(1) || *__slate_slot_2 < (*oc).chosen_points {
break '__join_3;
}
}
if !unsafe { (*oc).chosen }.is_null() {
put_task_struct((*oc).chosen);
}
get_task_struct(task);
(*oc).chosen = task;
(*oc).chosen_points = *__slate_slot_2;
}
}
}
}
return 0;
}
}

// Simple selection loop. We choose the process with the highest number of
// 'points'. In case scan was aborted, oc->chosen is set to -1.
fn select_bad_process(oc: *mut oom_control) {
unsafe {
(*oc).chosen_points = 9223372036854775807i64.wrapping_neg().wrapping_sub(1);
}
if is_memcg_oom(oc) {
mem_cgroup_scan_tasks(unsafe { (*oc).memcg }, Some(oom_evaluate_task), oc as *mut ());
} else {
let mut p: *mut task_struct = unsafe { core::mem::zeroed() };
rcu_read_lock();
p = unsafe { core::ptr::addr_of_mut!(init_task) };
loop {
let mut __v1: *mut task_struct;
let mut __v2: *mut list_head;
loop {
if !(((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_836() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*p).tasks.next) as *const *mut list_head) };
__v1 = ((unsafe { (__v2 as *const u8).wrapping_offset(-(offsetof__9!(task_struct) as isize)) }) as *mut ()) as *mut task_struct;
let __v4: *mut task_struct = __v1;
p = __v4;
if __v4 == unsafe { core::ptr::addr_of_mut!(init_task) } {
break;
}
if oom_evaluate_task(p, oc as *mut ()) != 0 {
break;
}
}
rcu_read_unlock();
}
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.dump_task")]
extern "C" fn dump_task(p: *mut task_struct, arg: *mut ()) -> i32 {
let oc: *mut oom_control = arg as *mut oom_control;
let mut task: *mut task_struct = unsafe { core::mem::zeroed() };
if oom_unkillable_task(p) {
return 0;
}
// p may not have freeable memory in nodemask
let __v0: bool = !is_memcg_oom(oc) && !oom_cpuset_eligible(p, oc);
if __v0 {
return 0;
}
task = find_lock_task_mm(p);
if task.is_null() {
// All of p's threads have already detached their mm's. There's
// no need to report them; they can't be oom killed anyway.
return 0;
}
let mut __v1: i32;
while false {
}
let mut __v3: __SlateRecord_uidgid_types_h_9;
let mut ___val: __SlateRecord_uidgid_types_h_9 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(___val) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
rcu_read_lock();
let mut __v4: *const cred;
let mut __v5: *const cred;
loop {
if !(__native_word__30!() || ((const { core::mem::size_of::<*const cred>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_838() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*task).real_cred) as *const *const cred) };
let __UNIQUE_ID_rcu_837: *const cred = __v5;
loop {
let __v8: bool = if false {
let __v9: bool = false || rcu_read_lock_held() != 0;
!__v9
} else {
false
};
if !__v8 {
break;
}
}
__v4 = __UNIQUE_ID_rcu_837;
___val = unsafe { (*__v4).uid };
rcu_read_unlock();
__v3 = ___val;
__v1 = unsafe { _printk(c"\x016[%7d] %5d %5d %8lu %8lu %8lu %8lu %9lu %8ld %8lu         %5hd %s\n".as_ptr() as *const u8, (*task).pid, from_kuid(core::ptr::addr_of_mut!(init_user_ns), __v3), (*task).tgid, (*(*task).mm).__slate_anon_0.total_vm, get_mm_rss_sum((*task).mm), get_mm_counter_sum((*task).mm, 1i32), get_mm_counter_sum((*task).mm, 0i32), get_mm_counter_sum((*task).mm, 3i32), mm_pgtables_bytes((*task).mm as *const mm_struct), get_mm_counter_sum((*task).mm, 2i32), (*(*task).signal).oom_score_adj as i32, (*task).comm.as_mut_ptr() as *mut u8) };
task_unlock(task);
0
}

// dump_tasks - dump current memory state of all system tasks
// @oc: pointer to struct oom_control
//
// Dumps the current memory state of all eligible tasks.  Tasks not in the same
// memcg, not in the same cpuset, or bound to a disjoint set of mempolicy nodes
// are not shown.
// State information includes task's pid, uid, tgid, vm size, rss,
// pgtables_bytes, swapents, oom_score_adj value, and name.
fn dump_tasks(oc: *mut oom_control) {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016Tasks state (memory values in pages):\n".as_ptr() as *const u8) };
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016[  pid  ]   uid  tgid total_vm      rss rss_anon rss_file rss_shmem pgtables_bytes swapents oom_score_adj name\n".as_ptr() as *const u8) };
if is_memcg_oom(oc) {
mem_cgroup_scan_tasks(unsafe { (*oc).memcg }, Some(dump_task), oc as *mut ());
} else {
let mut p: *mut task_struct = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
rcu_read_lock();
p = unsafe { core::ptr::addr_of_mut!(init_task) };
loop {
let mut __v6: *mut task_struct;
let mut __v7: *mut list_head;
loop {
if !(((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_839() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v7 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*p).tasks.next) as *const *mut list_head) };
__v6 = ((unsafe { (__v7 as *const u8).wrapping_offset(-(offsetof__9!(task_struct) as isize)) }) as *mut ()) as *mut task_struct;
let __v9: *mut task_struct = __v6;
p = __v9;
if __v9 == unsafe { core::ptr::addr_of_mut!(init_task) } {
break;
}
// Avoid potential softlockup warning
let __v5: i32 = i.wrapping_add(1);
i = __v5;
if __v5 & 1023 == 0 {
touch_softlockup_watchdog();
}
dump_task(p, oc as *mut ());
}
rcu_read_unlock();
}
}

fn dump_oom_victim(oc: *mut oom_control, victim: *mut task_struct) {
// one line summary of the oom killer context.
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016oom-kill:constraint=%s,nodemask=%*pbl".as_ptr() as *const u8, oom_constraint_text.0[((*oc).constraint as u32) as usize], __nodemask_pr_numnodes((*oc).nodemask), __nodemask_pr_bits((*oc).nodemask)) };
unsafe { cpuset_print_current_mems_allowed() };
mem_cgroup_print_oom_context(unsafe { (*oc).memcg }, victim);
let mut __v2: i32;
while false {
}
let mut __v4: __SlateRecord_uidgid_types_h_9;
let mut ___val: __SlateRecord_uidgid_types_h_9 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(___val) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
rcu_read_lock();
let mut __v5: *const cred;
let mut __v6: *const cred;
loop {
if !(__native_word__30!() || ((const { core::mem::size_of::<*const cred>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_841() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*victim).real_cred) as *const *const cred) };
let __UNIQUE_ID_rcu_840: *const cred = __v6;
loop {
let __v9: bool = if false {
let __v10: bool = false || rcu_read_lock_held() != 0;
!__v10
} else {
false
};
if !__v9 {
break;
}
}
__v5 = __UNIQUE_ID_rcu_840;
___val = unsafe { (*__v5).uid };
rcu_read_unlock();
__v4 = ___val;
__v2 = unsafe { _printk(c"\x01c,task=%s,pid=%d,uid=%d\n".as_ptr() as *const u8, (*victim).comm.as_mut_ptr() as *mut u8, (*victim).pid, from_kuid(core::ptr::addr_of_mut!(init_user_ns), __v4)) };
}

fn dump_header(oc: *mut oom_control) {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014%s invoked oom-killer: gfp_mask=%#x(%pGg), order=%d, oom_score_adj=%d\n".as_ptr() as *const u8, (*get_current()).comm.as_mut_ptr() as *mut u8, (*oc).gfp_mask, core::ptr::addr_of!((*oc).gfp_mask), (*oc).order, (*(*get_current()).signal).oom_score_adj as i32) };
if IS_ENABLED__2!() == 0 && (unsafe { (*oc).order }) != 0 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x014COMPACTION is disabled!!!\n".as_ptr() as *const u8) };
}
unsafe { dump_stack() };
if is_memcg_oom(oc) {
mem_cgroup_print_oom_meminfo(unsafe { (*oc).memcg });
} else {
unsafe { __show_mem(1u32, (*oc).nodemask, (gfp_zone((*oc).gfp_mask) as u32) as i32) };
if should_dump_unreclaim_slab() {
unsafe { dump_unreclaimable_slab() };
}
}
mem_cgroup_show_protected_memory(unsafe { (*oc).memcg });
if (unsafe { sysctl_oom_dump_tasks }) != 0 {
dump_tasks(oc);
}
}

/// Number of OOM victims in flight
static mut oom_victims: __SlateRecord_types_h_190 = __SlateRecord_types_h_190 { counter: 0i32 };

static mut oom_victims_wait: wait_queue_head = {
let mut __t0: wait_queue_head = unsafe { core::mem::zeroed() };
__t0.lock = spinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t1.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t2: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t2.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t2
} } };
__t1
} };
__t0.head = list_head { next: unsafe { core::ptr::addr_of_mut!(oom_victims_wait.head) }, prev: unsafe { core::ptr::addr_of_mut!(oom_victims_wait.head) } };
__t0
};

#[unsafe(link_section = ".data..read_mostly")]
static mut oom_killer_disabled: bool = unsafe { core::mem::zeroed() };

// task->mm can be NULL if the task is the exited group leader.  So to
// determine whether the task is using a particular mm, we examine all the
// task's threads: if one of those is using this mm then this task was also
// using it.
/// OOM Reaper kernel thread which tries to reap the memory used by the OOM
/// victim (if that is possible) to help the OOM killer to move on.
static mut oom_reaper_th: *mut task_struct = unsafe { core::mem::zeroed() };

static mut oom_reaper_wait: wait_queue_head = {
let mut __t0: wait_queue_head = unsafe { core::mem::zeroed() };
__t0.lock = spinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t1.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t2: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t2.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t2
} } };
__t1
} };
__t0.head = list_head { next: unsafe { core::ptr::addr_of_mut!(oom_reaper_wait.head) }, prev: unsafe { core::ptr::addr_of_mut!(oom_reaper_wait.head) } };
__t0
};

static mut oom_reaper_list: *mut task_struct = unsafe { core::mem::zeroed() };

static mut oom_reaper_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

fn __oom_reap_task_mm(mm: *mut mm_struct) -> bool {
let mut ret: bool = true;
let mut mas: ma_state = {
let mut __t0: ma_state = unsafe { core::mem::zeroed() };
__t0.tree = unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_mt) };
__t0.index = 18446744073709551615u64;
__t0.last = 18446744073709551615u64;
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
};
// Tell all users of get_user/copy_from_user etc... that the content
// is no longer stable. No barriers really needed because unmapping
// should imply barriers already and the reader would hit a page fault
// if it stumbled over a reaped memory.
mm_flags_set(22i32, mm);
// It might start racing with the dying task and compete for shared
// resources - e.g. page table lock contention has been observed.
// Reduce those races by reaping the oom victim from the other end
// of the address space.
loop {
let __v1: *mut vm_area_struct = (unsafe { mas_find_rev(core::ptr::addr_of_mut!(mas), 0u64) }) as *mut vm_area_struct;
let vma: *mut vm_area_struct = __v1;
if __v1.is_null() {
break;
}
if (unsafe { (*vma).__slate_anon_3.vm_flags }) & (BIT!(22i32) | BIT!(10i32)) == 0 {
// Only anonymous pages have a good chance to be dropped
// without additional steps which we cannot afford as we
// are OOM already.
//
// We do not even care about fs backed pages because all
// which are reclaimable have already been reclaimed and
// we do not want to block exit_mmap by keeping mm ref
// count elevated without a good reason.
if vma_is_anonymous(vma as *const vm_area_struct) || (unsafe { (*vma).__slate_anon_3.vm_flags }) & BIT!(3i32) == 0 {
if (unsafe { zap_vma_for_reaping(vma) }) != 0 {
ret = false;
}
}
}
}
ret
}

/// Reaps the address space of the given task.
///
/// Returns true on success and false if none or part of the address space
/// has been reclaimed and the caller should retry later.
fn oom_reap_task_mm(tsk: *mut task_struct, mm: *mut mm_struct) -> bool {
let mut ret: bool = unsafe { core::mem::zeroed() };
unsafe {
ret = true;
if !mmap_read_trylock(mm) {
trace_skip_task_reaping((*tsk).pid);
return false;
} else {
// MMF_OOM_SKIP is set by exit_mmap when the OOM reaper can't
// work on the mm anymore. The check for MMF_OOM_SKIP must run
// under mmap_lock for reading because it serializes against the
// mmap_write_lock();mmap_write_unlock() cycle in exit_mmap().
if mm_flags_test(21i32, mm as *const mm_struct) {
trace_skip_task_reaping((*tsk).pid);
} else {
trace_start_task_reaping((*tsk).pid);
// failed to reap part of the address space. Try again later
ret = __oom_reap_task_mm(mm);
if ret {
while false {
}
_printk(c"\x016oom_reaper: reaped process %d (%s), now anon-rss:%lukB, file-rss:%lukB, shmem-rss:%lukB\n".as_ptr() as *const u8, task_pid_nr(tsk), (*tsk).comm.as_mut_ptr() as *mut u8, K!(get_mm_counter_sum(mm, 1i32)), K!(get_mm_counter_sum(mm, 0i32)), K!(get_mm_counter_sum(mm, 3i32)));
}
trace_finish_task_reaping((*tsk).pid);
}
mmap_read_unlock(mm);
return ret;
}
}
}

fn oom_reap_task(tsk: *mut task_struct) {
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut mm: *mut mm_struct = unsafe { core::mem::zeroed() };
let mut attempts: i32 = unsafe { core::mem::zeroed() };
unsafe {
attempts = 0;
mm = (*(*tsk).signal).oom_mm;
// Retry the mmap_read_trylock(mm) a few times
loop {
let __v4: i32 = attempts;
attempts = __v4.wrapping_add(1);
if __v4 < 10 {
__v2 = !oom_reap_task_mm(tsk, mm);
} else {
__v2 = false;
}
if __v2 {
schedule_timeout_idle(100i64);
} else {
break;
}
}
if attempts <= 10 {
__v1 = true;
} else {
__v1 = mm_flags_test(21i32, mm as *const mm_struct);
}
if !__v1 {
while false {
}
_printk(c"\x016oom_reaper: unable to reap pid:%d (%s)\n".as_ptr() as *const u8, task_pid_nr(tsk), (*tsk).comm.as_mut_ptr() as *mut u8);
sched_show_task(tsk);
debug_show_all_locks();
}
(*tsk).oom_reaper_list = core::ptr::null_mut::<task_struct>();
// Hide this mm from OOM killer because it has been either reaped or
// somebody can't call mmap_write_unlock(mm).
mm_flags_set(21i32, mm);
// Drop a reference taken by queue_oom_reaper
put_task_struct(tsk);
}
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.oom_reaper")]
extern "C" fn oom_reaper(unused: *mut ()) -> i32 {
let mut __int: i64 = unsafe { core::mem::zeroed() };
let mut __ret: i64 = unsafe { core::mem::zeroed() };
let mut __wq_entry: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut wait_queue_entry = core::ptr::addr_of_mut!(__wq_entry) as *mut wait_queue_entry;
let mut __ret_1: i32 = unsafe { core::mem::zeroed() };
let mut tsk: *mut task_struct = unsafe { core::mem::zeroed() };
unsafe {
set_freezable();
while true {
tsk = core::ptr::null_mut::<task_struct>();
__ret_1 = 0;
might_sleep!();
if unsafe { oom_reaper_list }.is_null() {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
__ret = 0;
init_wait_entry(core::ptr::addr_of_mut!(*__slate_slot_2), 0i32);
'__join_6: {
'__join_7: {
loop {
__int = prepare_to_wait_event(core::ptr::addr_of_mut!(oom_reaper_wait), core::ptr::addr_of_mut!(*__slate_slot_2), 8193i32);
if !unsafe { oom_reaper_list }.is_null() {
break '__join_7;
} else {
if ___wait_is_interruptible!(8193i32) && __int != 0 {
break;
} else {
schedule();
if !unsafe { oom_reaper_list }.is_null() {
break '__join_7;
}
}
}
}
__ret = __int;
break '__join_6;
}
finish_wait(core::ptr::addr_of_mut!(oom_reaper_wait), core::ptr::addr_of_mut!(*__slate_slot_2));
}
__ret_1 = __ret as i32;
}
spin_lock_irq(core::ptr::addr_of_mut!(oom_reaper_lock));
if !unsafe { oom_reaper_list }.is_null() {
tsk = oom_reaper_list;
oom_reaper_list = (*tsk).oom_reaper_list;
}
spin_unlock_irq(core::ptr::addr_of_mut!(oom_reaper_lock));
if !tsk.is_null() {
oom_reap_task(tsk);
}
}
return 0;
}
}

#[unsafe(link_section = ".text.slate_distinct.oom_kill.wake_oom_reaper")]
extern "C" fn wake_oom_reaper(timer: *mut timer_list) {
let tsk: *mut task_struct = container_of__52!(timer, task_struct);
let mm: *mut mm_struct = unsafe { (*(*tsk).signal).oom_mm };
let mut flags: u64 = 0;
// The victim managed to terminate on its own - see exit_mmap
if mm_flags_test(21i32, mm as *const mm_struct) {
put_task_struct(tsk);
return;
}
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(oom_reaper_lock) }, flags);
unsafe {
(*tsk).oom_reaper_list = oom_reaper_list;
}
unsafe {
oom_reaper_list = tsk;
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!(oom_reaper_lock) }, flags);
trace_wake_reaper(unsafe { (*tsk).pid });
wake_up!(unsafe { core::ptr::addr_of_mut!(oom_reaper_wait) });
}

// Give the OOM victim time to exit naturally before invoking the oom_reaping.
// The timers timeout is arbitrary... the longer it is, the longer the worst
// case scenario for the OOM can take. If it is too small, the oom_reaper can
// get in the way and release resources needed by the process exit path.
// e.g. The futex robust list can sit in Anon|Private memory that gets reaped
// before the exit path is able to wake the futex waiters.
fn queue_oom_reaper(tsk: *mut task_struct) {
// mm is already queued?
if mm_flags_test_and_set(25i32, unsafe { (*(*tsk).signal).oom_mm }) {
return;
}
get_task_struct(tsk);
timer_setup!(unsafe { core::ptr::addr_of_mut!((*tsk).oom_reaper_timer) }, Some(wake_oom_reaper), 0i32);
unsafe {
(*tsk).oom_reaper_timer.expires = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(2000u64);
}
unsafe { add_timer(core::ptr::addr_of_mut!((*tsk).oom_reaper_timer)) };
}

static vm_oom_kill_table: __SlateAlign16<[ctl_table; 3]> = __SlateAlign16([{
let mut __t0: ctl_table = unsafe { core::mem::zeroed() };
__t0.procname = (b"panic_on_oom\0".as_ptr() as *mut u8) as *const u8;
__t0.data = (unsafe { core::ptr::addr_of_mut!(sysctl_panic_on_oom) }) as *mut ();
__t0.maxlen = (core::mem::size_of::<i32>() as u64) as i32;
__t0.mode = 420u16;
__t0.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_dointvec_minmax as *const ()) };
__t0.extra1 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(0i32 as isize) as *mut ();
__t0.extra2 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(2i32 as isize) as *mut ();
__t0
}, {
let mut __t1: ctl_table = unsafe { core::mem::zeroed() };
__t1.procname = (b"oom_kill_allocating_task\0".as_ptr() as *mut u8) as *const u8;
__t1.data = (unsafe { core::ptr::addr_of_mut!(sysctl_oom_kill_allocating_task) }) as *mut ();
__t1.maxlen = (core::mem::size_of::<i32>() as u64) as i32;
__t1.mode = 420u16;
__t1.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_dointvec as *const ()) };
__t1
}, {
let mut __t2: ctl_table = unsafe { core::mem::zeroed() };
__t2.procname = (b"oom_dump_tasks\0".as_ptr() as *mut u8) as *const u8;
__t2.data = (unsafe { core::ptr::addr_of_mut!(sysctl_oom_dump_tasks) }) as *mut ();
__t2.maxlen = (core::mem::size_of::<i32>() as u64) as i32;
__t2.mode = 420u16;
__t2.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_dointvec as *const ()) };
__t2
}]);

#[unsafe(link_section = ".init.text")]
extern "C" fn oom_init() -> i32 {
let mut __v0: *mut task_struct;
let __k: *mut task_struct = unsafe { kthread_create_on_node(Some(oom_reaper), core::ptr::null_mut::<()>(), 1i32.wrapping_neg(), c"oom_reaper".as_ptr() as *const u8) };
if !IS_ERR(__k as *const ()) {
unsafe { wake_up_process(__k) };
}
__v0 = __k;
unsafe {
oom_reaper_th = __v0;
}
unsafe { __register_sysctl_init(c"vm".as_ptr() as *const u8, core::ptr::addr_of!(vm_oom_kill_table.0) as *const ctl_table, c"vm_oom_kill_table".as_ptr() as *const u8, ARRAY_SIZE__38!()) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_oom_init_846: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(oom_init)) };

/// mark_oom_victim - mark the given task as OOM victim
/// @tsk: task to mark
///
/// Has to be called with oom_lock held and never after
/// oom has been disabled already.
///
/// tsk->mm has to be non NULL and caller has to guarantee it is stable (either
/// under task_lock or operate on the current).
fn mark_oom_victim(tsk: *mut task_struct) {
let mm: *mut mm_struct = unsafe { (*tsk).mm };
let mut __v0: i64;
let __ret_warn_on: i32 = !(!unsafe { oom_killer_disabled }) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("847: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 847b - ., 3; .popsection\n# {0}", const 847 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mark_oom_victim_1, sym __slate_asm_string_mark_oom_victim_2, const 756 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("848: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 848b - ., 4; .popsection\n# {0}", const 848 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
// OOM killer might race with memcg OOM
if test_and_set_tsk_thread_flag(tsk, 3i32) != 0 {
return;
}
// oom_mm is bound to the signal struct life time.
let mut __v5: *mut mm_struct;
let __ai_ptr: *mut *mut mm_struct = unsafe { core::ptr::addr_of_mut!((*(*tsk).signal).oom_mm) };
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<*mut mm_struct>() as u64);
let mut __v7: *mut mm_struct;
let mut __ret: *mut mm_struct = unsafe { core::mem::zeroed() };
let mut __old: *mut mm_struct = core::ptr::null_mut::<mm_struct>();
let mut __new: *mut mm_struct = mm;
match (const { core::mem::size_of::<*mut mm_struct>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = __ai_ptr as *mut u8;
unsafe {
let __t0: u64;
core::arch::asm!("lock cmpxchgb {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t0, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut mm_struct>(__t0 as u64) };
}
}
2 => {
let mut __ptr: *mut u16 = __ai_ptr as *mut u16;
unsafe {
let __t1: u64;
core::arch::asm!("lock cmpxchgw {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t1, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut mm_struct>(__t1 as u64) };
}
}
4 => {
let mut __ptr: *mut u32 = __ai_ptr as *mut u32;
unsafe {
let __t2: u64;
core::arch::asm!("lock cmpxchgl {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t2, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut mm_struct>(__t2 as u64) };
}
}
8 => {
let mut __ptr: *mut u64 = __ai_ptr as *mut u64;
unsafe {
let __t3: u64;
core::arch::asm!("lock cmpxchgq {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t3, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut mm_struct>(__t3 as u64) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
__v7 = __ret;
__v5 = __v7;
if __v5.is_null() {
mmgrab(unsafe { (*(*tsk).signal).oom_mm });
}
// Make sure that the process is woken up from uninterruptible sleep
// if it is frozen because OOM killer wouldn't be able to free any
// memory and livelock. The freezer will thaw the tasks that are OOM
// victims regardless of the PM freezing and cgroup freezing states.
unsafe { thaw_process(tsk) };
atomic_inc(unsafe { core::ptr::addr_of_mut!(oom_victims) });
let cred: *const cred = unsafe { get_task_cred(tsk) };
trace_mark_victim(tsk, unsafe { (*cred).uid.val });
put_cred(cred);
}

/// exit_oom_victim - note the exit of an OOM victim
#[unsafe(no_mangle)]
extern "C" fn exit_oom_victim() {
clear_thread_flag!(3i32);
if atomic_dec_return(unsafe { core::ptr::addr_of_mut!(oom_victims) }) == 0 {
wake_up_all!(unsafe { core::ptr::addr_of_mut!(oom_victims_wait) });
}
}

/// oom_killer_enable - enable OOM killer
#[unsafe(no_mangle)]
extern "C" fn oom_killer_enable() {
unsafe {
oom_killer_disabled = false;
}
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016OOM killer enabled.\n".as_ptr() as *const u8) };
}

/// oom_killer_disable - disable OOM killer
/// @timeout: maximum timeout to wait for oom victims in jiffies
///
/// Forces all page allocations to fail rather than trigger OOM killer.
/// Will block and wait until all OOM victims are killed or the given
/// timeout expires.
///
/// The function cannot be called when there are runnable user tasks because
/// the userspace would see unexpected allocation failures as a result. Any
/// new usage of this function should be consulted with MM people.
///
/// Returns true if successful and false if the OOM killer cannot be
/// disabled.
#[unsafe(no_mangle)]
extern "C" fn oom_killer_disable(timeout: i64) -> bool {
let mut __cond: bool = unsafe { core::mem::zeroed() };
let mut __cond_1: bool = unsafe { core::mem::zeroed() };
let mut __int: i64 = unsafe { core::mem::zeroed() };
let mut __ret: i64 = unsafe { core::mem::zeroed() };
let mut __wq_entry: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut wait_queue_entry = core::ptr::addr_of_mut!(__wq_entry) as *mut wait_queue_entry;
let mut __ret_1: i64 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i64 = core::ptr::addr_of_mut!(ret) as *mut i64;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
// Make sure to not race with an ongoing OOM killer. Check that the
// current is not killed (possibly due to sharing the victim's memory).
if mutex_lock_killable(core::ptr::addr_of_mut!(oom_lock)) != 0 {
return false;
} else {
oom_killer_disabled = true;
mutex_unlock(core::ptr::addr_of_mut!(oom_lock));
__ret_1 = timeout;
might_sleep!();
if !___wait_cond_timeout!(atomic_read(core::ptr::addr_of_mut!(oom_victims) as *const __SlateRecord_types_h_190) == 0, __ret_1) {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
__ret = timeout;
init_wait_entry(core::ptr::addr_of_mut!(*__slate_slot_7), if false { 1i32 } else { 0i32 });
'__join_4: {
'__join_5: {
loop {
__int = prepare_to_wait_event(core::ptr::addr_of_mut!(oom_victims_wait), core::ptr::addr_of_mut!(*__slate_slot_7), 1i32);
__cond_1 = atomic_read(core::ptr::addr_of_mut!(oom_victims) as *const __SlateRecord_types_h_190) == 0;
if __cond_1 && __ret == 0 {
__ret = 1;
}
let __v4: bool = __cond_1 || __ret == 0;
if __v4 {
break '__join_5;
} else {
if ___wait_is_interruptible!(1i32) && __int != 0 {
break;
} else {
__ret = schedule_timeout(__ret);
__cond = atomic_read(core::ptr::addr_of_mut!(oom_victims) as *const __SlateRecord_types_h_190) == 0;
if __cond && __ret == 0 {
__ret = 1;
}
if __cond || __ret == 0 {
break '__join_5;
}
}
}
}
__ret = __int;
break '__join_4;
}
finish_wait(core::ptr::addr_of_mut!(oom_victims_wait), core::ptr::addr_of_mut!(*__slate_slot_7));
}
__ret_1 = __ret;
}
*__slate_slot_11 = __ret_1;
if *__slate_slot_11 <= 0 {
oom_killer_enable();
return false;
} else {
while false {
}
_printk(c"\x016OOM killer disabled.\n".as_ptr() as *const u8);
return true;
}
}
}
}

fn __task_will_free_mem(task: *mut task_struct) -> bool {
let sig: *mut signal_struct = unsafe { (*task).signal };
// A coredumping process may sleep for an extended period in
// coredump_task_exit(), so the oom killer cannot assume that
// the process will promptly exit and release memory.
if !unsafe { (*sig).core_state }.is_null() {
return false;
}
if (unsafe { (*sig).flags }) & 4 != 0 {
return true;
}
if thread_group_empty(task) != 0 && (unsafe { (*task).flags }) & 4 != 0 {
return true;
}
false
}

// Checks whether the given task is dying or exiting and likely to
// release its address space. This means that all threads and processes
// sharing the same mm have to be killed or exiting.
// Caller has to make sure that task->mm is stable (hold task_lock or
// it operates on the current).
fn task_will_free_mem(task: *mut task_struct) -> bool {
let mm: *mut mm_struct = unsafe { (*task).mm };
let mut p: *mut task_struct = unsafe { core::mem::zeroed() };
let mut ret: bool = true;
// Skip tasks without mm because it might have passed its exit_mm and
// exit_oom_victim. oom_reaper could have rescued that but do not rely
// on that for now. We can consider find_lock_task_mm in future.
if mm.is_null() {
return false;
}
if !__task_will_free_mem(task) {
return false;
}
// This task has already been drained by the oom reaper so there are
// only small chances it will free some more
if mm_flags_test(21i32, mm as *const mm_struct) {
return false;
}
if atomic_read((unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_users) }) as *const __SlateRecord_types_h_190) <= 1 {
return true;
}
// Make sure that all tasks which share the mm with the given tasks
// are dying as well to make sure that a) nobody pins its mm and
// b) the task is also reapable by the oom reaper.
rcu_read_lock();
p = unsafe { core::ptr::addr_of_mut!(init_task) };
loop {
let mut __v1: *mut task_struct;
let mut __v2: *mut list_head;
loop {
if !(((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_849() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*p).tasks.next) as *const *mut list_head) };
__v1 = ((unsafe { (__v2 as *const u8).wrapping_offset(-(offsetof__9!(task_struct) as isize)) }) as *mut ()) as *mut task_struct;
let __v4: *mut task_struct = __v1;
p = __v4;
if __v4 == unsafe { core::ptr::addr_of_mut!(init_task) } {
break;
}
if process_shares_mm(p as *const task_struct, mm as *const mm_struct) {
if !same_thread_group(task, p) {
ret = __task_will_free_mem(p);
if !ret {
break;
}
}
}
}
rcu_read_unlock();
ret
}

fn __oom_kill_process(mut victim: *mut task_struct, message: *const u8) {
let mut p: *mut task_struct = unsafe { core::mem::zeroed() };
let mut can_oom_reap: bool = true;
p = find_lock_task_mm(victim);
if p.is_null() {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016%s: OOM victim %d (%s) is already exiting. Skip killing the task\n".as_ptr() as *const u8, message, task_pid_nr(victim), (*victim).comm.as_mut_ptr() as *mut u8) };
put_task_struct(victim);
return;
} else {
if victim != p {
get_task_struct(p);
put_task_struct(victim);
victim = p;
}
}
// Get a reference to safely compare mm after task_unlock(victim)
let mm: *mut mm_struct = unsafe { (*victim).mm };
mmgrab(mm);
// Raise event before sending signal: task reaper must see this
count_vm_event(36u32);
memcg_memory_event_mm(mm, 4u32);
// We should send SIGKILL before granting access to memory reserves
// in order to prevent the OOM victim from depleting the memory
// reserves from the user space under its control.
unsafe { do_send_sig_info(9i32, 1i32 as *mut kernel_siginfo, victim, 1u32) };
mark_oom_victim(victim);
let mut __v2: i32;
while false {
}
let mut __v4: __SlateRecord_uidgid_types_h_9;
let mut ___val: __SlateRecord_uidgid_types_h_9 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(___val) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
rcu_read_lock();
let mut __v5: *const cred;
let mut __v6: *const cred;
while false {
}
__v6 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*victim).real_cred) as *const *const cred) };
let __UNIQUE_ID_rcu_850: *const cred = __v6;
loop {
let __v9: bool = if false {
let __v10: bool = false || rcu_read_lock_held() != 0;
!__v10
} else {
false
};
if !__v9 {
break;
}
}
__v5 = __UNIQUE_ID_rcu_850;
___val = unsafe { (*__v5).uid };
rcu_read_unlock();
__v4 = ___val;
__v2 = unsafe { _printk(c"\x013%s: Killed process %d (%s) total-vm:%lukB, anon-rss:%lukB, file-rss:%lukB, shmem-rss:%lukB, UID:%u pgtables:%lukB oom_score_adj:%d\n".as_ptr() as *const u8, message, task_pid_nr(victim), (*victim).comm.as_mut_ptr() as *mut u8, K!(unsafe { (*mm).__slate_anon_0.total_vm }), K!(get_mm_counter_sum(mm, 1i32)), K!(get_mm_counter_sum(mm, 0i32)), K!(get_mm_counter_sum(mm, 3i32)), from_kuid(core::ptr::addr_of_mut!(init_user_ns), __v4), mm_pgtables_bytes(mm as *const mm_struct) >> 10i32, (*(*victim).signal).oom_score_adj as i32) };
task_unlock(victim);
// Kill all user processes sharing victim->mm in other thread groups, if
// any.  They don't get access to memory reserves, though, to avoid
// depletion of all memory.  This prevents mm->mmap_lock livelock when an
// oom killed thread cannot exit because it requires the semaphore and
// its contended by another thread trying to allocate memory itself.
// That thread will now get access to memory reserves since it has a
// pending fatal signal.
rcu_read_lock();
p = unsafe { core::ptr::addr_of_mut!(init_task) };
loop {
let mut __v14: *mut task_struct;
let mut __v15: *mut list_head;
while false {
}
__v15 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*p).tasks.next) as *const *mut list_head) };
__v14 = ((unsafe { (__v15 as *const u8).wrapping_offset(-(offsetof__9!(task_struct) as isize)) }) as *mut ()) as *mut task_struct;
let __v17: *mut task_struct = __v14;
p = __v17;
if __v17 == unsafe { core::ptr::addr_of_mut!(init_task) } {
break;
}
if process_shares_mm(p as *const task_struct, mm as *const mm_struct) {
if !same_thread_group(p, victim) {
if is_global_init(p) != 0 {
can_oom_reap = false;
mm_flags_set(21i32, mm);
let mut __v12: i32;
while false {
}
__v12 = unsafe { _printk(c"\x016oom killer %d (%s) has mm pinned by %d (%s)\n".as_ptr() as *const u8, task_pid_nr(victim), (*victim).comm.as_mut_ptr() as *mut u8, task_pid_nr(p), (*p).comm.as_mut_ptr() as *mut u8) };
} else {
// No kthread_use_mm() user needs to read from the userspace so
// we are ok to reap it.
if unlikely__3!((unsafe { (*p).flags }) & 2097152) == 0 {
unsafe { do_send_sig_info(9i32, 1i32 as *mut kernel_siginfo, p, 1u32) };
}
}
}
}
}
rcu_read_unlock();
if can_oom_reap {
queue_oom_reaper(victim);
}
mmdrop(mm);
put_task_struct(victim);
}

/// Kill provided task unless it's secured by setting
/// oom_score_adj to OOM_SCORE_ADJ_MIN.
#[unsafe(link_section = ".text.slate_distinct.oom_kill.oom_kill_memcg_member")]
extern "C" fn oom_kill_memcg_member(task: *mut task_struct, message: *mut ()) -> i32 {
let __v0: bool = ((unsafe { (*(*task).signal).oom_score_adj }) as i32) != 1000i32.wrapping_neg() && is_global_init(task) == 0;
if __v0 {
get_task_struct(task);
__oom_kill_process(task, message as *const u8);
}
0
}

fn oom_kill_process(oc: *mut oom_control, message: *const u8) {
let victim: *mut task_struct = unsafe { (*oc).chosen };
// If the task is already exiting, don't alarm the sysadmin or kill
// its children or threads, just give it access to memory reserves
// so it can die quickly
task_lock(victim);
if task_will_free_mem(victim) {
mark_oom_victim(victim);
queue_oom_reaper(victim);
task_unlock(victim);
put_task_struct(victim);
return;
}
task_unlock(victim);
if __ratelimit__14!(unsafe { core::ptr::addr_of_mut!(oom_rs) }) != 0 {
dump_header(oc);
dump_oom_victim(oc, victim);
}
// Do we need to kill the entire memory cgroup?
// Or even one of the ancestor memory cgroups?
// Check this out before killing the victim task.
let oom_group: *mut mem_cgroup = mem_cgroup_get_oom_group(victim, unsafe { (*oc).memcg });
__oom_kill_process(victim, message);
// If necessary, kill all tasks in the selected memory cgroup.
if !oom_group.is_null() {
memcg_memory_event(oom_group, 5u32);
mem_cgroup_print_oom_group(oom_group);
mem_cgroup_scan_tasks(oom_group, Some(oom_kill_memcg_member), message as *mut ());
mem_cgroup_put(oom_group);
}
}

static mut oom_rs: ratelimit_state = {
let mut __t0: ratelimit_state = unsafe { core::mem::zeroed() };
__t0.lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.interval = 5000i32;
__t0.burst = 10i32;
__t0.flags = 0u32;
__t0
};

/// Determines whether the kernel must panic because of the panic_on_oom sysctl.
fn check_panic_on_oom(oc: *mut oom_control) {
if likely!((unsafe { sysctl_panic_on_oom }) == 0) != 0 {
return;
}
if (unsafe { sysctl_panic_on_oom }) != 2 {
// panic_on_oom == 1 only affects CONSTRAINT_NONE, the kernel
// does not panic for cpuset, mempolicy, or memcg allocation
// failures.
if ((unsafe { (*oc).constraint }) as u32) != 0 {
return;
}
}
// Do not panic for oom kills triggered by sysrq
if is_sysrq_oom(oc) {
return;
}
dump_header(oc);
unsafe { panic(c"Out of memory: %s panic_on_oom is enabled\n".as_ptr() as *const u8, if sysctl_panic_on_oom == 2 { c"compulsory".as_ptr() as *mut u8 } else { c"system-wide".as_ptr() as *mut u8 }) };
unsafe { core::hint::unreachable_unchecked() };
}

static mut oom_notify_list: blocking_notifier_head = blocking_notifier_head { rwsem: rw_semaphore { count: __SlateRecord_types_h_197 { counter: 0i64 }, owner: __SlateRecord_types_h_197 { counter: 0i64 }, osq: optimistic_spin_queue { tail: __SlateRecord_types_h_190 { counter: 0i32 } }, wait_lock: raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t0.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t0
} } }, first_waiter: core::ptr::null_mut::<rwsem_waiter>() }, head: core::ptr::null_mut::<notifier_block>() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.register_oom_notifier")]
extern "C" fn register_oom_notifier(nb: *mut notifier_block) -> i32 {
unsafe { blocking_notifier_chain_register(core::ptr::addr_of_mut!(oom_notify_list), nb) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_register_oom_notifier_853: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut notifier_block) -> i32>, *mut ()>(Some(register_oom_notifier)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.oom_kill.unregister_oom_notifier")]
extern "C" fn unregister_oom_notifier(nb: *mut notifier_block) -> i32 {
unsafe { blocking_notifier_chain_unregister(core::ptr::addr_of_mut!(oom_notify_list), nb) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_unregister_oom_notifier_854: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut notifier_block) -> i32>, *mut ()>(Some(unregister_oom_notifier)) };

/// out_of_memory - kill the "best" process when we run out of memory
/// @oc: pointer to struct oom_control
///
/// If we run out of memory, we have the choice between either
/// killing a random task (bad), letting the system crash (worse)
/// OR try to be smart about which process to kill. Note that we
/// don't have to be perfect here, we just have to be good.
#[unsafe(no_mangle)]
extern "C" fn out_of_memory(oc: *mut oom_control) -> bool {
let mut freed: u64 = 0;
if unsafe { oom_killer_disabled } {
return false;
}
if !is_memcg_oom(oc) {
unsafe { blocking_notifier_call_chain(core::ptr::addr_of_mut!(oom_notify_list), 0u64, core::ptr::addr_of_mut!(freed) as *mut ()) };
let __v0: bool = freed > 0 && !is_sysrq_oom(oc);
if __v0 {
return true;
}
// Got some memory back in the last second.
}
// If current has a pending SIGKILL or is exiting, then automatically
// select it.  The goal is to allow it to allocate so that it may
// quickly exit and free its memory.
if task_will_free_mem(get_current()) {
mark_oom_victim(get_current());
queue_oom_reaper(get_current());
return true;
}
// The OOM killer does not compensate for IO-less reclaim.
// But mem_cgroup_oom() has to invoke the OOM killer even
// if it is a GFP_NOFS allocation.
let __v1: bool = (unsafe { (*oc).gfp_mask }) & (BIT!(7i32) as u32) == 0 && !is_memcg_oom(oc);
if __v1 {
return true;
}
// Check if there were limitations on the allocation (only relevant for
// NUMA and memcg) that may require different handling.
unsafe {
(*oc).constraint = constrained_alloc(oc);
}
if ((unsafe { (*oc).constraint }) as u32) != 2 {
unsafe {
(*oc).nodemask = core::ptr::null::<__SlateRecord_nodemask_types_h_17>();
}
}
check_panic_on_oom(oc);
let __v2: bool = !is_memcg_oom(oc) && (unsafe { sysctl_oom_kill_allocating_task }) != 0 && !unsafe { (*get_current()).mm }.is_null();
let __v3: bool = __v2 && !oom_unkillable_task(get_current());
let __v4: bool = __v3 && oom_cpuset_eligible(get_current(), oc);
let __v5: bool = __v4 && ((unsafe { (*(*get_current()).signal).oom_score_adj }) as i32) != 1000i32.wrapping_neg();
if __v5 {
get_task_struct(get_current());
unsafe {
(*oc).chosen = get_current();
}
oom_kill_process(oc, c"Out of memory (oom_kill_allocating_task)".as_ptr() as *const u8);
return true;
}
select_bad_process(oc);
// Found nothing?!?!
if unsafe { (*oc).chosen }.is_null() {
dump_header(oc);
let mut __v6: i32;
while false {
}
__v6 = unsafe { _printk(c"\x014Out of memory and no killable processes...\n".as_ptr() as *const u8) };
// If we got here due to an actual allocation at the
// system level, we cannot survive this and will enter
// an endless loop in the allocator. Bail out now.
let __v8: bool = !is_sysrq_oom(oc) && !is_memcg_oom(oc);
if __v8 {
unsafe { panic(c"System is deadlocked on memory\n".as_ptr() as *const u8) };
unsafe { core::hint::unreachable_unchecked() };
}
}
if !unsafe { (*oc).chosen }.is_null() && (unsafe { (*oc).chosen }) != ((1u64.wrapping_neg() as *mut ()) as *mut task_struct) {
oom_kill_process(oc, (if !is_memcg_oom(oc) { c"Out of memory".as_ptr() as *mut u8 } else { c"Memory cgroup out of memory".as_ptr() as *mut u8 }) as *const u8);
}
!unsafe { (*oc).chosen }.is_null()
}

/// The pagefault handler calls here because some allocation has failed. We have
/// to take care of the memcg OOM here because this is the only safe context without
/// any locks held but let the oom killer triggered from the allocation context care
/// about the global OOM.
#[unsafe(no_mangle)]
extern "C" fn pagefault_out_of_memory() {
if mem_cgroup_oom_synchronize(true) {
return;
}
if fatal_signal_pending(get_current()) != 0 {
return;
}
if __ratelimit__12!(unsafe { core::ptr::addr_of_mut!(pfoom_rs) }) != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014Huh VM_FAULT_OOM leaked out to the #PF handler. Retrying PF\n".as_ptr() as *const u8) };
}
}

static mut pfoom_rs: ratelimit_state = {
let mut __t0: ratelimit_state = unsafe { core::mem::zeroed() };
__t0.lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.interval = 5000i32;
__t0.burst = 10i32;
__t0.flags = 0u32;
__t0
};

#[unsafe(no_mangle)]
extern "C" fn __x64_sys_process_mrelease(regs: *const pt_regs) -> i64 {
__se_sys_process_mrelease((unsafe { (*regs).di }) as i64, (unsafe { (*regs).si }) as i64)
}

#[unsafe(no_mangle)]
extern "C" fn __ia32_sys_process_mrelease(regs: *const pt_regs) -> i64 {
__se_sys_process_mrelease(((unsafe { (*regs).bx }) as u32) as i64, ((unsafe { (*regs).cx }) as u32) as i64)
}

fn __se_sys_process_mrelease(pidfd: i64, flags: i64) -> i64 {
let ret: i64 = __do_sys_process_mrelease(pidfd as i32, flags as u32);
while false {
}
ret
}

fn __do_sys_process_mrelease(pidfd: i32, flags: u32) -> i64 {
// Check MMF_OOM_SKIP again under mmap_read_lock protection to ensure
// possible change in exit_mmap is seen
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut ret: i64 = unsafe { core::mem::zeroed() };
let mut reap: bool = unsafe { core::mem::zeroed() };
let mut f_flags: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u32 = core::ptr::addr_of_mut!(f_flags) as *mut u32;
let mut p: core::mem::MaybeUninit<*mut task_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut task_struct = core::ptr::addr_of_mut!(p) as *mut *mut task_struct;
let mut task: core::mem::MaybeUninit<*mut task_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut task_struct = core::ptr::addr_of_mut!(task) as *mut *mut task_struct;
let mut mm: *mut mm_struct = unsafe { core::mem::zeroed() };
unsafe {
mm = core::ptr::null_mut::<mm_struct>();
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
reap = false;
ret = 0;
if flags != 0 {
return 22i32.wrapping_neg() as i64;
} else {
*__slate_slot_5 = pidfd_get_task(pidfd, core::ptr::addr_of_mut!(*__slate_slot_3));
if IS_ERR(*__slate_slot_5 as *const ()) {
return PTR_ERR(*__slate_slot_5 as *const ());
} else {
// Make sure to choose a thread which still has a reference to mm
// during the group exit
*__slate_slot_4 = find_lock_task_mm(*__slate_slot_5);
if (*__slate_slot_4).is_null() {
ret = 3i32.wrapping_neg() as i64;
} else {
mm = (*(*__slate_slot_4)).mm;
mmgrab(mm);
if task_will_free_mem(*__slate_slot_4) {
reap = true;
} else {
// Error only if the work has not been done already
if !mm_flags_test(21i32, mm as *const mm_struct) {
ret = 22i32.wrapping_neg() as i64;
}
}
task_unlock(*__slate_slot_4);
if reap {
if mmap_read_lock_killable(mm) != 0 {
ret = 4i32.wrapping_neg() as i64;
} else {
if !mm_flags_test(21i32, mm as *const mm_struct) {
__v0 = !__oom_reap_task_mm(mm);
} else {
__v0 = false;
}
if __v0 {
ret = 11i32.wrapping_neg() as i64;
}
mmap_read_unlock(mm);
}
}
mmdrop(mm);
}
put_task_struct(*__slate_slot_5);
return ret;
}
}
}
}

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_mark_oom_victim_1: [u8; 1] = *b"\0";

static __slate_asm_string_mark_oom_victim_2: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/oom_kill.c\0";
