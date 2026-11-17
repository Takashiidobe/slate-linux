
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
//! SPDX-License-Identifier: GPL-2.0
//!
//! kernel/cpuset.c
//!
//! Processor and Memory placement constraints for sets of tasks.
//!
//! Copyright (C) 2003 BULL SA.
//! Copyright (C) 2004-2007 Silicon Graphics, Inc.
//! Copyright (C) 2006 Google, Inc
//!
//! Portions derived from Patrick Mochel's sysfs code.
//! sysfs is Copyright (c) 2001-3 Patrick Mochel
//!
//! 2003-10-10 Written by Simon Derr.
//! 2003-10-22 Updates by Stephen Hemminger.
//! 2004 May-July Rework by Paul Jackson.
//! 2006 Rework by Paul Menage to use generic cgroups
//! 2008 Rework of the scheduler domains and CPU hotplug handling
//!      by Max Krasnyansky
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
//! New user namespace
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
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
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! This counts to 15. Any more, it will return 16th argument.
//! # chars in a path name including nul
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
//! Sign Flag
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
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! warning conditions
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Needs two entries
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! The size of each LDT entry.
//! Defines x86 CPU feature bits
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
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
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! writeable
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
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
//! do_div - returns 2 values: calculate remainder and update new dividend
//! @n: uint64_t dividend (will be updated)
//! @base: uint32_t divisor
//!
//! Summary:
//! ``uint32_t remainder = n % base;``
//! ``n = n / base;``
//!
//! Return: (uint32_t)remainder
//!
//! NOTE: macro parameter @n is evaluated multiple times,
//! beware of side effects!
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! u64_to_user_ptr - cast a pointer passed as u64 from user space to void __user *
//! @x: The u64 value from user space, usually via IOCTL
//!
//! u64_to_user_ptr() simply casts a pointer passed as u64 from user space to void
//! __user * correctly. Using this lets us get rid of all the tiresome casts.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
//! used to decode ioctl numbers..
//! cpumask_pr_args - printf args to output a cpumask
//! @maskp: cpumask to be printed
//!
//! Can be used to provide arguments for '%*pb[l]' when printing a cpumask.
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
//! This function is needed by all SMP systems. It must _always_ be valid
//! from the initial startup.
//! If !LOCKDEP, the value is meaningless
//! Shortcut for the queued_spin_lock_slowpath() function that allows
//! virt to hijack it.
//!
//! Returns:
//!   true - lock has been negotiated, all done;
//!   false - queued_spin_lock_slowpath() will do its thing.
//! If a tracepoint needs to be called from a header file, it is not
//! recommended to call it directly, as tracepoints in header files
//! may cause side-effects and bloat the kernel. Instead, use
//! tracepoint_enabled() to test if the tracepoint is enabled, then if
//! it is, call a wrapper function defined in a C file that will then
//! call the tracepoint.
//!
//! For "trace_foo_bar()", you would need to create a wrapper function
//! in a C file to call trace_foo_bar():
//!   void do_trace_foo_bar(args) { trace_foo_bar(args); }
//! Then in the header file, declare the tracepoint:
//!   DECLARE_TRACEPOINT(foo_bar);
//! And call your wrapper:
//!   static inline void some_inlined_function() {
//!            [..]
//!            if (tracepoint_enabled(foo_bar))
//!                    do_trace_foo_bar(args);
//!            [..]
//!   }
//!
//! Note: tracepoint_enabled(foo_bar) is equivalent to trace_foo_bar_enabled()
//!   but is safe to have in headers, where trace_foo_bar_enabled() is not.
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
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
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
//! per-IO O_SYNC
//! File supports atomic writes
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
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
//! Despite the naming, these are FLAGS not bits.
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
//! See the comment in asm-generic/io.h about ioremap_np().
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! Move every page to conform to policy
//! This is a list of *what* is being read, not *how* nor *where*.
//! ld/ldx fields
//! Define a list of socket types which can be the argument for
//! skc_to_*_sock() helpers. All these sockets should have
//! sock_common as the first argument in its memory layout.
//! BTF_KIND_INT is followed by a u32 and the following
//! is the 32 bits arrangement:
//! 31 pointers + header align the folio_batch structure to a power of two
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! Choose 31 as it makes rqspinlock_held cacheline-aligned.
//! bpf_type_flag contains a set of flags that are applicable to the values of
//! arg_type, ret_type and reg_type. For example, a pointer value may be null,
//! or a memory is read-only. We classify types into two categories: base types
//! and extended types. Extended types are base types combined with a type flag.
//!
//! Currently there are no more than 32 base types in arg_type, ret_type and
//! reg_types.
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
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
// SCHED_ISO: reserved but not implemented yet
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
// Overflow Flag
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
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "hypervisor" Running on a hypervisor
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
// Like MOV_GS except MSR_KERNEL_GS_BASE = GS.base
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
// Atomic operations are already serializing on x86
// No such file or directory
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
// Invalid system call number
// Value too large for defined data type
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
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
// Same as rb_first(), but O(1)
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
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
// Resource extended types
// No address assigned yet
// Implementation specific max size
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// These special macros can be used to get or set a debugging register
// Internal flags that share the struct mempolicy flags word with
// "mode flags".  These flags are allocated from bit 0 up, as they
// are never OR'ed into the mode in mempolicy API arguments.
// identify shared policies
// double word (64-bit)
// 1 << 13
// insn[0].src_reg:  BPF_PSEUDO_FUNC
// insn[0].imm:      insn offset to the func
// insn[1].imm:      0
// insn[0].off:      0
// insn[1].off:      0
// ldimm64 rewrite:  address of the function
// verifier type:    PTR_TO_FUNC.
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
// Attributes stored in the BTF_INT_ENCODING
// If the struct/union type info kind_flag is set, the
// following two macros are used to access bitfield_size
// and bit_offset from btf_member.offset.
// PT_TRACE_* event enable flags
// Max number of base types.
// Max number of all types.
// The longest tracepoint has 12 args.
// See include/trace/bpf_probe.h
//
// Also reuse this macro for maximum number of arguments a BPF function
// or a kfunc can have. Args 1-5 are passed in registers, args 6-12 via
// stack arg slots. The JIT may map some stack arg slots to registers based
// on the native calling convention (e.g., arg 6 to R9 on x86-64).
// Fits in 2048B
// Per-cpu temp buffers used by printf-like helpers to store the bprintf binary
// arguments representation.
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
macro_rules! ALIGN__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL__2!($x, $a)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__16 {
() => {
((core::mem::size_of::<[*mut [cpumask; 1]; 3]>() as u64) / (core::mem::size_of::<*mut [cpumask; 1]>() as u64)).wrapping_add(__must_be_array__107!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__41 {
() => {
((core::mem::size_of::<[*mut [cpumask; 1]; 4]>() as u64) / (core::mem::size_of::<*mut [cpumask; 1]>() as u64)).wrapping_add(__must_be_array__78!() as u64)
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

macro_rules! GENMASK_INPUT_CHECK__3 {
() => {
core::mem::size_of::<__SlateRecord_find_h_316>() as i32
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
(GENMASK_INPUT_CHECK__3!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
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

macro_rules! GENMASK__3 {
($h:expr, $l:expr) => {
GENMASK_TYPE__3!(u64, $h, $l)
};
}

macro_rules! IS_ALIGNED {
($x:expr, $a:expr) => {
($x) & (($a) as u32).wrapping_sub(1) == 0
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

macro_rules! NODE_DATA {
($nid:expr) => {
unsafe { *(core::ptr::addr_of_mut!(node_data) as *mut *mut pglist_data).wrapping_offset(($nid) as isize) }
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
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

macro_rules! __ALIGN_KERNEL__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as u32).wrapping_sub(1))
};
}

macro_rules! __IRQ_MASK {
($x:expr) => {
(1u64 << $x).wrapping_sub(1)
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

macro_rules! __WRITE_ONCE__29 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut llist_node, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__7 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u32, $val) };
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

macro_rules! __WRITE_ONCE__14 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u32, ($val) as u32) };
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

macro_rules! __is_nonneg {
($ux:expr) => {
statically_true!((($ux) as i64) >= 0)
};
}

macro_rules! __must_be_array__78 {
() => {
core::mem::size_of::<__SlateRecord_cpuset_c_632>() as i32
};
}

macro_rules! __must_be_array__107 {
() => {
core::mem::size_of::<__SlateRecord_cpuset_c_584>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__20 {
() => {
((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__18 {
() => {
((const { core::mem::size_of::<*mut css_set>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut css_set>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut css_set>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut css_set>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__7 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
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

/// The following macros are non-atomic versions of their non-underscored
/// counterparts.
macro_rules! __set_bit__2 {
($nr:expr, $addr:expr) => {
if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___set_bit(($nr) as u64, ($addr) as *mut u64);
} else {
___set_bit(($nr) as u64, ($addr) as *mut u64);
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
macro_rules! __sign_use__6 {
($ux:expr) => {
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<u64>() as u64) < 4) as i32)) }
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

macro_rules! __types_ok__5 {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__6!($uy)
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

macro_rules! alloc_hooks {
($_do_alloc:expr) => {
{
let mut __v: *mut ();
let mut __v2: *mut ();
let mut __v3: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
__v3 = $_do_alloc;
alloc_tag_restore!();
} else {
__v3 = $_do_alloc;
}
__v2 = __v3;
__v = __v2;
__v
}
};
}

macro_rules! alloc_tag_restore {
() => {
while false {
}
};
}

macro_rules! bitmap_size {
($nbits:expr) => {
ALIGN__2!($nbits, 64i32) / 8
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
macro_rules! container_of__5 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__15!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__9!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__88 {
($ptr:expr) => {
{
let mut __v: *mut cpuset;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__64!() as isize)) }) as *mut ()) as *mut cpuset;
__v
}
};
}

/// cpumask_any_and - pick an arbitrary cpu from *mask1 & *mask2
/// @mask1: the first input cpumask
/// @mask2: the second input cpumask
///
/// Return: >= nr_cpu_ids if no cpus set.
macro_rules! cpumask_any_and__4 {
($mask1:expr, $mask2:expr) => {
cpumask_first_and($mask1, ($mask2) as *const cpumask)
};
}

/// cpumask_bits - get the bits in a cpumask
/// @maskp: the struct cpumask *
///
/// You should only assume nr_cpu_ids bits of this mask are valid.  This is
/// a macro so it's const-correct.
macro_rules! cpumask_bits__2 {
($maskp:expr) => {
unsafe { (*$maskp).bits.as_mut_ptr() as *mut u64 }
};
}

/// cpumask_bits - get the bits in a cpumask
/// @maskp: the struct cpumask *
///
/// You should only assume nr_cpu_ids bits of this mask are valid.  This is
/// a macro so it's const-correct.
macro_rules! cpumask_bits {
($maskp:expr) => {
unsafe { (*$maskp).bits.as_ptr() as *const u64 }
};
}

macro_rules! first_node {
($src:expr) => {
__first_node((unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17)
};
}

macro_rules! flush_workqueue {
($wq:expr) => {
let _wq: *mut workqueue_struct = $wq;
if (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_percpu_wq) }) && _wq == unsafe { system_percpu_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_highpri_wq) }) && _wq == unsafe { system_highpri_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_long_wq) }) && _wq == unsafe { system_long_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_dfl_long_wq) }) && _wq == unsafe { system_dfl_long_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_dfl_wq) }) && _wq == unsafe { system_dfl_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_freezable_wq) }) && _wq == unsafe { system_freezable_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_power_efficient_wq) }) && _wq == unsafe { system_power_efficient_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_freezable_power_efficient_wq) }) && _wq == unsafe { system_freezable_power_efficient_wq } {
unsafe { __warn_flushing_systemwide_wq() };
}
unsafe { __flush_workqueue(_wq) };
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

macro_rules! in_interrupt {
() => {
irq_count!()
};
}

macro_rules! irq_count {
() => {
(preempt_count() as u64) & (__IRQ_MASK!(1i32.wrapping_add(3i32.wrapping_mul(IS_ENABLED__2!()))) << 28i32 | __IRQ_MASK!(4i32) << 24i32 | __IRQ_MASK!(8i32) << 8i32)
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

macro_rules! kcsan_wmb {
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

/// llist_entry - get the struct of this entry
/// @ptr:	the &struct llist_node pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the llist_node within the struct.
macro_rules! llist_entry {
($ptr:expr) => {
container_of__88!($ptr)
};
}

macro_rules! local_irq_disable {
() => {
loop {
raw_local_irq_disable!();
if true {
break;
}
}
};
}

macro_rules! local_irq_enable {
() => {
loop {
raw_local_irq_enable!();
if true {
break;
}
}
};
}

macro_rules! lock_acquire {
() => {
while false {
}
};
}

macro_rules! lock_acquire_exclusive {
() => {
lock_acquire!();
};
}

macro_rules! lock_release {
() => {
while false {
}
};
}

macro_rules! lockdep_assert_held {
($l:expr) => {
__assume_ctx_lock!($l);
};
}

macro_rules! lockdep_assert_preemption_disabled {
() => {
while false {
}
};
}

/// member_address_is_nonnull - check whether the member address is not NULL
/// @ptr:	the object pointer (struct type * that contains the llist_node)
/// @member:	the name of the llist_node within the struct.
///
/// This macro is conceptually the same as
/// 	&ptr->member != NULL
/// but it works around the fact that compilers can decide that taking a member
/// address is never a NULL pointer.
///
/// Real objects that start at a high address and have a member at NULL are
/// unlikely to exist, but such pointers may be returned e.g. by the
/// container_of() macro.
macro_rules! member_address_is_nonnull {
($ptr:expr) => {
(($ptr) as u64).wrapping_add(offsetof__64!()) != 0
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

/// Find the next present node in src, starting after node n, wrapping around to
/// the first node in src if needed.  Returns MAX_NUMNODES if src is empty.
macro_rules! next_node_in {
($n:expr, $src:expr) => {
__next_node_in($n, (unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17)
};
}

/// No static inline type checking - see Subtlety (1) above.
macro_rules! node_isset__2 {
($node:expr, $nodemask:expr) => {
test_bit__8!($node, unsafe { $nodemask.bits.as_mut_ptr() as *mut u64 })
};
}

/// No static inline type checking - see Subtlety (1) above.
macro_rules! node_isset__3 {
($node:expr, $nodemask:expr) => {
test_bit__2!($node, unsafe { $nodemask.bits.as_mut_ptr() as *mut u64 })
};
}

macro_rules! nodelist_parse {
($buf:expr, $dst:expr) => {
__nodelist_parse($buf, unsafe { core::ptr::addr_of_mut!($dst) }, 64i32)
};
}

macro_rules! nodes_and {
($dst:expr, $src1:expr, $src2:expr) => {
__nodes_and(unsafe { core::ptr::addr_of_mut!($dst) }, (unsafe { core::ptr::addr_of_mut!($src1) }) as *const __SlateRecord_nodemask_types_h_17, (unsafe { core::ptr::addr_of_mut!($src2) }) as *const __SlateRecord_nodemask_types_h_17, 64u32)
};
}

macro_rules! nodes_copy {
($dst:expr, $src:expr) => {
__nodes_copy(unsafe { core::ptr::addr_of_mut!($dst) }, (unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17, 64u32);
};
}

macro_rules! nodes_empty {
($src:expr) => {
__nodes_empty((unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17, 64u32)
};
}

macro_rules! nodes_equal__2 {
($src1:expr, $src2:expr) => {
__nodes_equal(unsafe { core::ptr::addr_of!($src1) }, unsafe { core::ptr::addr_of!($src2) }, 64u32)
};
}

macro_rules! nodes_equal {
($src1:expr, $src2:expr) => {
__nodes_equal((unsafe { core::ptr::addr_of_mut!($src1) }) as *const __SlateRecord_nodemask_types_h_17, (unsafe { core::ptr::addr_of_mut!($src2) }) as *const __SlateRecord_nodemask_types_h_17, 64u32)
};
}

macro_rules! nodes_intersects {
($src1:expr, $src2:expr) => {
__nodes_intersects(unsafe { core::ptr::addr_of!($src1) }, (unsafe { core::ptr::addr_of_mut!($src2) }) as *const __SlateRecord_nodemask_types_h_17, 64u32)
};
}

macro_rules! nodes_intersects__2 {
($src1:expr, $src2:expr) => {
__nodes_intersects((unsafe { core::ptr::addr_of_mut!($src1) }) as *const __SlateRecord_nodemask_types_h_17, (unsafe { core::ptr::addr_of_mut!($src2) }) as *const __SlateRecord_nodemask_types_h_17, 64u32)
};
}

macro_rules! nodes_intersects__3 {
($src1:expr, $src2:expr) => {
__nodes_intersects(unsafe { core::ptr::addr_of!($src1) }, unsafe { core::ptr::addr_of!($src2) }, 64u32)
};
}

macro_rules! nodes_or {
($dst:expr, $src1:expr, $src2:expr) => {
__nodes_or(unsafe { core::ptr::addr_of_mut!($dst) }, (unsafe { core::ptr::addr_of_mut!($src1) }) as *const __SlateRecord_nodemask_types_h_17, (unsafe { core::ptr::addr_of_mut!($src2) }) as *const __SlateRecord_nodemask_types_h_17, 64u32);
};
}

macro_rules! nodes_setall {
($dst:expr) => {
__nodes_setall(unsafe { core::ptr::addr_of_mut!($dst) }, 64u32);
};
}

macro_rules! nodes_subset {
($src1:expr, $src2:expr) => {
__nodes_subset((unsafe { core::ptr::addr_of_mut!($src1) }) as *const __SlateRecord_nodemask_types_h_17, (unsafe { core::ptr::addr_of_mut!($src2) }) as *const __SlateRecord_nodemask_types_h_17, 64u32)
};
}

macro_rules! offsetof__9 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, css) as u64
};
}

macro_rules! offsetof__15 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, work) as u64
};
}

macro_rules! offsetof__64 {
() => {
core::mem::offset_of!(cpuset, attach_node) as u64
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

/// Wrap the arch provided IRQ routines to provide appropriate checks.
macro_rules! raw_local_irq_disable {
() => {
arch_local_irq_disable();
};
}

macro_rules! raw_local_irq_enable {
() => {
arch_local_irq_enable();
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

macro_rules! seqcount_acquire {
() => {
lock_acquire_exclusive!();
};
}

macro_rules! seqcount_release {
() => {
lock_release!();
};
}

macro_rules! seqprop_assert__3 {
($s:expr) => {
__seqprop_spinlock_assert(($s) as *const seqcount_spinlock);
};
}

macro_rules! seqprop_preemptible__3 {
($s:expr) => {
__seqprop_spinlock_preemptible(($s) as *const seqcount_spinlock)
};
}

macro_rules! seqprop_ptr__4 {
($s:expr) => {
__seqprop_spinlock_ptr($s)
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
macro_rules! small_const_nbits__3 {
($nbits:expr) => {
true && ($nbits) <= 64 && ($nbits) > 0
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

macro_rules! static_branch_dec_cpuslocked {
($x:expr) => {
unsafe { static_key_slow_dec_cpuslocked(core::ptr::addr_of_mut!((*$x).key)) };
};
}

macro_rules! static_branch_enable_cpuslocked {
($x:expr) => {
unsafe { static_key_enable_cpuslocked(core::ptr::addr_of_mut!((*$x).key)) };
};
}

macro_rules! static_branch_inc_cpuslocked {
($x:expr) => {
unsafe { static_key_slow_inc_cpuslocked(core::ptr::addr_of_mut!((*$x).key)) };
};
}

macro_rules! static_branch_likely__3 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = !arch_static_branch__const2(unsafe { core::ptr::addr_of_mut!((*$x).key) }, true);
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
__v2 = arch_static_branch__const1(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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

macro_rules! task_cpu_possible_mask {
() => {
(unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask
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

macro_rules! test_bit__8 {
($nr:expr, $addr:expr) => {
{
let __v: bool = (unsafe { __slate_intrinsic_a39e427d3b6c2df4($nr) }) && false && (($addr) as u64) != 0;
let __v2: i32 = if __v && false {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v2
}
};
}

macro_rules! test_bit__4 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
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

macro_rules! unlikely__4 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! wake_up {
($x:expr) => {
unsafe { __wake_up($x, 3u32, 1i32, core::ptr::null_mut::<()>()) };
};
}

unsafe extern "C" {
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut kmsg_fops: file_operations;
    static mut __SCK__might_resched: static_call_key;
    static mut nr_node_ids: u32;
    static mut node_states: [__SlateRecord_nodemask_types_h_17; 6];
    static mut node_data: [*mut pglist_data; 0];
    static mut nr_cpu_ids: u32;
    static mut __cpu_possible_mask: cpumask;
    static mut __cpu_active_mask: cpumask;
    static mut __num_online_cpus: __SlateRecord_types_h_190;
    static mut cpuinfo_op: seq_operations;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut sched_domains_mutex: mutex;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut system_highpri_wq: *mut workqueue_struct;
    static mut system_long_wq: *mut workqueue_struct;
    static mut system_dfl_wq: *mut workqueue_struct;
    static mut system_freezable_wq: *mut workqueue_struct;
    static mut system_power_efficient_wq: *mut workqueue_struct;
    static mut system_freezable_power_efficient_wq: *mut workqueue_struct;
    static mut system_dfl_long_wq: *mut workqueue_struct;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut kstat: kernel_stat;
    static mut cgroup_mutex: mutex;
    static mut cpuset_cgrp_subsys_on_dfl_key: static_key_true;
    static mut dl_cookie: u64;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn _printk(arg0: *const u8, ...) -> i32;
    fn static_key_slow_inc_cpuslocked(arg0: *mut static_key) -> bool;
    fn static_key_slow_dec_cpuslocked(arg0: *mut static_key);
    fn static_key_count(arg0: *mut static_key) -> i32;
    fn static_key_enable_cpuslocked(arg0: *mut static_key);
    fn ____wrong_branch_error() -> bool;
    fn __cmpxchg_wrong_size();
    fn _find_next_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn _find_first_bit(arg0: *const u64, arg1: u64) -> u64;
    fn _find_first_and_bit(arg0: *const u64, arg1: *const u64, arg2: u64) -> u64;
    fn find_random_bit(arg0: *const u64, arg1: u64) -> u64;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn memcmp(arg0: *const (), arg1: *const (), arg2: u64) -> i32;
    fn strcmp(arg0: *const u8, arg1: *const u8) -> i32;
    fn strim(arg0: *mut u8) -> *mut u8;
    fn kmemdup_noprof(arg0: *const (), arg1: u64, arg2: u32) -> *mut ();
    fn bitmap_parselist(arg0: *const u8, arg1: *mut u64, arg2: i32) -> i32;
    fn __bitmap_equal(arg0: *const u64, arg1: *const u64, arg2: u32) -> bool;
    fn __bitmap_and(arg0: *mut u64, arg1: *const u64, arg2: *const u64, arg3: u32) -> bool;
    fn __bitmap_or(arg0: *mut u64, arg1: *const u64, arg2: *const u64, arg3: u32);
    fn __bitmap_andnot(arg0: *mut u64, arg1: *const u64, arg2: *const u64, arg3: u32) -> bool;
    fn __bitmap_intersects(arg0: *const u64, arg1: *const u64, arg2: u32) -> bool;
    fn __bitmap_subset(arg0: *const u64, arg1: *const u64, arg2: u32) -> bool;
    fn __bitmap_weight(arg0: *const u64, arg1: u32) -> u32;
    fn __bitmap_weight_and(arg0: *const u64, arg1: *const u64, arg2: u32) -> u32;
    fn __bitmap_weight_andnot(arg0: *const u64, arg1: *const u64, arg2: u32) -> u32;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn __compiletime_assert_48();
    fn __compiletime_assert_51();
    fn __compiletime_assert_60();
    fn __compiletime_assert_61();
    fn __compiletime_assert_64();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn schedule();
    fn cpuset_cpumask_can_shrink(arg0: *const cpumask, arg1: *const cpumask) -> i32;
    fn task_can_attach(arg0: *mut task_struct) -> i32;
    fn dl_bw_alloc(arg0: i32, arg1: u64) -> i32;
    fn dl_bw_free(arg0: i32, arg1: u64);
    fn set_cpus_allowed_force(arg0: *mut task_struct, arg1: *const cpumask);
    fn set_cpus_allowed_ptr(arg0: *mut task_struct, arg1: *const cpumask) -> i32;
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn rcu_is_watching() -> bool;
    fn __wake_up(arg0: *mut wait_queue_head, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn init_wait_entry(arg0: *mut wait_queue_entry, arg1: i32);
    fn prepare_to_wait_event(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry, arg2: i32) -> i64;
    fn finish_wait(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn alloc_workqueue_noprof(arg0: *const u8, arg1: u32, arg2: i32, ...) -> *mut workqueue_struct;
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn __flush_workqueue(arg0: *mut workqueue_struct);
    fn __warn_flushing_systemwide_wq();
    fn __next_zones_zonelist(arg0: *mut zoneref, arg1: u32, arg2: *const __SlateRecord_nodemask_types_h_17) -> *mut zoneref;
    fn mmput(arg0: *mut mm_struct);
    fn get_task_mm(arg0: *mut task_struct) -> *mut mm_struct;
    fn capable(arg0: i32) -> bool;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_359();
    fn __compiletime_assert_384();
    fn __compiletime_assert_385();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_390();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn seq_write(arg0: *mut seq_file, arg1: *const (), arg2: u64) -> i32;
    fn seq_printf(arg0: *mut seq_file, arg1: *const u8, ...);
    fn seq_putc(arg0: *mut seq_file, arg1: u8);
    fn __seq_puts(arg0: *mut seq_file, arg1: *const u8);
    #[link_name = "strlen"]
    fn __builtin_strlen(arg0: *const u8) -> u64;
    fn pr_cont_kernfs_name(arg0: *mut kernfs_node);
    fn cgroup_e_css(arg0: *mut cgroup, arg1: *mut cgroup_subsys) -> *mut cgroup_subsys_state;
    fn cgroup_get_e_css(arg0: *mut cgroup, arg1: *mut cgroup_subsys) -> *mut cgroup_subsys_state;
    fn cgroup_file_notify(arg0: *mut cgroup_file);
    fn css_next_child(arg0: *mut cgroup_subsys_state, arg1: *mut cgroup_subsys_state) -> *mut cgroup_subsys_state;
    fn css_next_descendant_pre(arg0: *mut cgroup_subsys_state, arg1: *mut cgroup_subsys_state) -> *mut cgroup_subsys_state;
    fn css_rightmost_descendant(arg0: *mut cgroup_subsys_state) -> *mut cgroup_subsys_state;
    fn cgroup_taskset_first(arg0: *mut cgroup_taskset, arg1: *mut *mut cgroup_subsys_state) -> *mut task_struct;
    fn cgroup_taskset_next(arg0: *mut cgroup_taskset, arg1: *mut *mut cgroup_subsys_state) -> *mut task_struct;
    fn css_task_iter_start(arg0: *mut cgroup_subsys_state, arg1: u32, arg2: *mut css_task_iter);
    fn css_task_iter_next(arg0: *mut css_task_iter) -> *mut task_struct;
    fn css_task_iter_end(arg0: *mut css_task_iter);
    fn __compiletime_assert_470();
    fn __compiletime_assert_476();
    fn __compiletime_assert_477();
    fn __compiletime_assert_478();
    fn of_css(arg0: *mut kernfs_open_file) -> *mut cgroup_subsys_state;
    fn partition_sched_domains(arg0: i32, arg1: *mut [cpumask; 1], arg2: *mut sched_domain_attr);
    fn alloc_sched_domains(arg0: u32) -> *mut [cpumask; 1];
    fn free_sched_domains(arg0: *mut [cpumask; 1], arg1: u32);
    fn cpus_read_lock();
    fn cpus_read_unlock();
    fn lockdep_assert_cpus_held();
    fn housekeeping_cpumask(arg0: u32) -> *const cpumask;
    fn housekeeping_enabled(arg0: u32) -> bool;
    fn housekeeping_update(arg0: *mut cpumask) -> i32;
    fn mpol_rebind_task(arg0: *mut task_struct, arg1: *const __SlateRecord_nodemask_types_h_17);
    fn mpol_rebind_mm(arg0: *mut mm_struct, arg1: *mut __SlateRecord_nodemask_types_h_17);
    fn do_migrate_pages(arg0: *mut mm_struct, arg1: *const __SlateRecord_nodemask_types_h_17, arg2: *const __SlateRecord_nodemask_types_h_17, arg3: i32) -> i32;
    fn dl_add_task_root_domain(arg0: *mut task_struct);
    fn dl_clear_root_domain_cpu(arg0: i32);
    fn dl_task_needs_bw_move(arg0: *mut task_struct, arg1: *const cpumask) -> bool;
    fn dl_bw_visited(arg0: i32, arg1: u64) -> bool;
    fn security_task_setscheduler(arg0: *mut task_struct) -> i32;
    fn task_work_add(arg0: *mut task_struct, arg1: *mut callback_head, arg2: u32) -> i32;
    fn __compiletime_assert_679();
    fn __compiletime_assert_694();
    fn __compiletime_assert_703();
    fn __compiletime_assert_706();
    fn __compiletime_assert_707();
    fn __compiletime_assert_708();
    fn __compiletime_assert_717();
    fn __compiletime_assert_720();
    fn __compiletime_assert_721();
    fn __compiletime_assert_722();
    fn __compiletime_assert_723();
    fn __compiletime_assert_726();
    fn __compiletime_assert_730();
    fn __compiletime_assert_737();
    fn __compiletime_assert_746();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i8"]
    fn __slate_intrinsic_70b71f185ebb8c4a(arg0: u8) -> bool;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
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

fn kcsan_nestable_atomic_begin() {
}

fn kcsan_nestable_atomic_end() {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


fn kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

// kcsan_check_read - check regular read access for races
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
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
}


#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely__2!(overflow) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[inline(always)]
fn generic___set_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
unsafe {
*p = *p | mask;
}
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

#[inline(always)]
fn arch___set_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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

#[inline(always)]
fn instrument_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_write!(v, size);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_67 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_207 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_316 {
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

// instrument_get_user() - add instrumentation to get_user()-like macros
// @to: destination variable, may not be address-taken
//
// get_user() and friends are fragile, so it may depend on the implementation
// whether the instrumentation happens before or after the data is copied from
// the userspace.
#[inline(always)]
fn set_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_set_bit(nr, addr);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn clear_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_clear_bit(nr, addr);
}

#[inline(always)]
fn ___set_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___set_bit(nr, addr);
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
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
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_21 {
    pte: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_295 {
    pgd: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
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
    __slate_bits_0: __slate_bits::__SlateBits114U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym cpusets_insane_config_key, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym cpuset_cgrp_subsys_on_dfl_key, const 1 as u8, label {
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
struct static_key_true {
    key: static_key,
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
    __slate_bits_0: __slate_bits::__SlateBits121U0,
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

#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
}

#[inline(always)]
fn arch_atomic_add(mut i: i32, mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock addl {1:e}, ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) i as u32, options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic_inc(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock incl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic_dec(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock decl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE__2!((*v).counter)
}

#[inline(always)]
fn arch_atomic64_add(mut i: i64, mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock addq {1:r}, ({0:r})\n# {2}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) i as u64, in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
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
fn arch_atomic64_try_cmpxchg(v: *mut __SlateRecord_types_h_197, old: &mut i64, mut __new: i64) -> bool {
let mut __v0: i64;
let mut success: bool = false;
let mut __old: i64 = unsafe { *old };
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
*old = __old;
}
}
__v0 = success as i64;
__v0 != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[inline(always)]
fn raw_atomic_add(i: i32, v: *mut __SlateRecord_types_h_190) {
arch_atomic_add(i, v);
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
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic64_add(i: i64, v: *mut __SlateRecord_types_h_197) {
arch_atomic64_add(i, v);
}

#[inline(always)]
fn raw_atomic64_try_cmpxchg(v: *mut __SlateRecord_types_h_197, old: &mut i64, new: i64) -> bool {
arch_atomic64_try_cmpxchg(v, old, new)
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
fn raw_atomic64_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
arch_atomic64_sub_and_test(i, v)
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
fn raw_atomic64_fetch_add_unless(v: *mut __SlateRecord_types_h_197, a: i64, u: i64) -> i64 {
let mut c: i64 = raw_atomic64_read(v as *const __SlateRecord_types_h_197);
while unlikely__2!(c == u) == 0 {
if !(!raw_atomic64_try_cmpxchg(v, unsafe { &mut (*core::ptr::addr_of_mut!(c)) }, c.wrapping_add(a))) {
break;
}
}
c
}

#[inline(always)]
fn raw_atomic64_add_unless(v: *mut __SlateRecord_types_h_197, a: i64, u: i64) -> bool {
raw_atomic64_fetch_add_unless(v, a, u) != u
}

#[inline(always)]
fn raw_atomic64_inc_not_zero(v: *mut __SlateRecord_types_h_197) -> bool {
raw_atomic64_add_unless(v, 1i64, 0i64)
}

#[inline(always)]
fn raw_atomic_long_add(i: i64, v: *mut __SlateRecord_types_h_197) {
raw_atomic64_add(i, v);
}

#[inline(always)]
fn raw_atomic_long_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
raw_atomic64_sub_and_test(i, v)
}

#[inline(always)]
fn raw_atomic_long_inc_not_zero(v: *mut __SlateRecord_types_h_197) -> bool {
raw_atomic64_inc_not_zero(v)
}

#[inline(always)]
fn atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_read(v)
}

#[inline(always)]
fn atomic_add(i: i32, v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_add(i, v);
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
fn atomic_long_add(i: i64, v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_add(i, v);
}

#[inline(always)]
fn atomic_long_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_sub_and_test(i, v)
}

#[inline(always)]
fn atomic_long_inc_not_zero(v: *mut __SlateRecord_types_h_197) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_inc_not_zero(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[inline(always)]
fn find_next_bit(addr: *const u64, size: u64, offset: u64) -> u64 {
if small_const_nbits__2!(size) {
let mut val: u64 = 0;
if unlikely__2!(offset >= size) != 0 {
return size;
}
val = (unsafe { *addr }) & GENMASK!(size.wrapping_sub(1), offset);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_next_bit(addr, size, offset) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn find_first_bit(addr: *const u64, size: u64) -> u64 {
if small_const_nbits__2!(size) {
let val: u64 = (unsafe { *addr }) & GENMASK__2!(size.wrapping_sub(1), 0i32);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_first_bit(addr, size) }
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
fn find_first_and_bit(addr1: *const u64, addr2: *const u64, size: u64) -> u64 {
if small_const_nbits__2!(size) {
let val: u64 = (unsafe { *addr1 }) & unsafe { *addr2 } & GENMASK__3!(size.wrapping_sub(1), 0i32);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_first_and_bit(addr1, addr2, size) }
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


fn strstrip(str: *mut u8) -> *mut u8 {
unsafe { strim(str) }
}

#[inline(always)]
fn bitmap_zero(dst: *mut u64, nbits: u32) {
let len: u32 = bitmap_size!(nbits);
if small_const_nbits!(nbits) {
unsafe {
*dst = 0;
}
} else {
unsafe { memset(dst as *mut (), 0i32, len as u64) };
}
}

#[inline(always)]
fn bitmap_fill(dst: *mut u64, nbits: u32) {
let len: u32 = bitmap_size!(nbits);
if small_const_nbits!(nbits) {
unsafe {
*dst = !0;
}
} else {
unsafe { memset(dst as *mut (), 255i32, len as u64) };
}
}

#[inline(always)]
fn bitmap_copy(dst: *mut u64, src: *const u64, nbits: u32) {
let len: u32 = bitmap_size!(nbits);
if small_const_nbits!(nbits) {
unsafe {
*dst = *src;
}
} else {
unsafe { memcpy(dst as *mut (), src as *const (), len as u64) };
}
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

#[inline(always)]
fn bitmap_equal(src1: *const u64, src2: *const u64, nbits: u32) -> bool {
if small_const_nbits!(nbits) {
return ((unsafe { *src1 }) ^ unsafe { *src2 }) & BITMAP_LAST_WORD_MASK!(nbits) == 0;
}
if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(nbits & 7u32) }) && IS_ALIGNED!(nbits, 8i32) {
return (unsafe { memcmp(src1 as *const (), src2 as *const (), (nbits / 8) as u64) }) == 0;
}
unsafe { __bitmap_equal(src1, src2, nbits) }
}

#[inline(always)]
fn bitmap_intersects(src1: *const u64, src2: *const u64, nbits: u32) -> bool {
if small_const_nbits!(nbits) {
return (unsafe { *src1 }) & unsafe { *src2 } & BITMAP_LAST_WORD_MASK!(nbits) != 0;
} else {
return unsafe { __bitmap_intersects(src1, src2, nbits) };
}
}

#[inline(always)]
fn bitmap_subset(src1: *const u64, src2: *const u64, nbits: u32) -> bool {
if small_const_nbits!(nbits) {
return (unsafe { *src1 }) & !unsafe { *src2 } & BITMAP_LAST_WORD_MASK!(nbits) == 0;
} else {
return unsafe { __bitmap_subset(src1, src2, nbits) };
}
}

#[inline(always)]
fn bitmap_empty(src: *const u64, nbits: u32) -> bool {
if small_const_nbits!(nbits) {
return (unsafe { *src }) & BITMAP_LAST_WORD_MASK!(nbits) == 0;
}
find_first_bit(src, nbits as u64) == (nbits as u64)
}

#[inline(always)]
fn bitmap_weight(src: *const u64, nbits: u32) -> u32 {
if small_const_nbits!(nbits) {
return hweight_long((unsafe { *src }) & BITMAP_LAST_WORD_MASK!(nbits)) as u32;
}
unsafe { __bitmap_weight(src, nbits) }
}

#[inline(always)]
fn bitmap_weight_and(src1: *const u64, src2: *const u64, nbits: u32) -> u64 {
if small_const_nbits!(nbits) {
return hweight_long((unsafe { *src1 }) & unsafe { *src2 } & BITMAP_LAST_WORD_MASK!(nbits));
}
(unsafe { __bitmap_weight_and(src1, src2, nbits) }) as u64
}

#[inline(always)]
fn bitmap_weight_andnot(src1: *const u64, src2: *const u64, nbits: u32) -> u64 {
if small_const_nbits!(nbits) {
return hweight_long((unsafe { *src1 }) & !unsafe { *src2 } & BITMAP_LAST_WORD_MASK!(nbits));
}
(unsafe { __bitmap_weight_andnot(src1, src2, nbits) }) as u64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpumask {
    bits: [u64; 1],
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
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_task {
}


#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct nameidata {
}


#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_13: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

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


fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
// nodemask_pr_args - printf args to output a nodemask
// @maskp: nodemask to be printed
//
// Can be used to provide arguments for '%*pb[l]' when printing a nodemask.
#[inline(always)]
fn __nodemask_pr_numnodes(m: *const __SlateRecord_nodemask_types_h_17) -> u32 {
if !m.is_null() { unsafe { nr_node_ids } } else { 0 }
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
#[inline(always)]
fn __nodes_setall(dstp: *mut __SlateRecord_nodemask_types_h_17, nbits: u32) {
bitmap_fill(unsafe { (*dstp).bits.as_mut_ptr() as *mut u64 }, nbits);
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
fn __nodes_and(dstp: *mut __SlateRecord_nodemask_types_h_17, src1p: *const __SlateRecord_nodemask_types_h_17, src2p: *const __SlateRecord_nodemask_types_h_17, nbits: u32) -> bool {
bitmap_and(unsafe { (*dstp).bits.as_mut_ptr() as *mut u64 }, unsafe { (*src1p).bits.as_ptr() as *const u64 }, unsafe { (*src2p).bits.as_ptr() as *const u64 }, nbits)
}

#[inline(always)]
fn __nodes_or(dstp: *mut __SlateRecord_nodemask_types_h_17, src1p: *const __SlateRecord_nodemask_types_h_17, src2p: *const __SlateRecord_nodemask_types_h_17, nbits: u32) {
bitmap_or(unsafe { (*dstp).bits.as_mut_ptr() as *mut u64 }, unsafe { (*src1p).bits.as_ptr() as *const u64 }, unsafe { (*src2p).bits.as_ptr() as *const u64 }, nbits);
}

#[inline(always)]
fn __nodes_copy(dstp: *mut __SlateRecord_nodemask_types_h_17, srcp: *const __SlateRecord_nodemask_types_h_17, nbits: u32) {
bitmap_copy(unsafe { (*dstp).bits.as_mut_ptr() as *mut u64 }, unsafe { (*srcp).bits.as_ptr() as *const u64 }, nbits);
}

#[inline(always)]
fn __nodes_equal(src1p: *const __SlateRecord_nodemask_types_h_17, src2p: *const __SlateRecord_nodemask_types_h_17, nbits: u32) -> bool {
bitmap_equal(unsafe { (*src1p).bits.as_ptr() as *const u64 }, unsafe { (*src2p).bits.as_ptr() as *const u64 }, nbits)
}

#[inline(always)]
fn __nodes_intersects(src1p: *const __SlateRecord_nodemask_types_h_17, src2p: *const __SlateRecord_nodemask_types_h_17, nbits: u32) -> bool {
bitmap_intersects(unsafe { (*src1p).bits.as_ptr() as *const u64 }, unsafe { (*src2p).bits.as_ptr() as *const u64 }, nbits)
}

#[inline(always)]
fn __nodes_subset(src1p: *const __SlateRecord_nodemask_types_h_17, src2p: *const __SlateRecord_nodemask_types_h_17, nbits: u32) -> bool {
bitmap_subset(unsafe { (*src1p).bits.as_ptr() as *const u64 }, unsafe { (*src2p).bits.as_ptr() as *const u64 }, nbits)
}

#[inline(always)]
fn __nodes_empty(srcp: *const __SlateRecord_nodemask_types_h_17, nbits: u32) -> bool {
bitmap_empty(unsafe { (*srcp).bits.as_ptr() as *const u64 }, nbits)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
}


#[inline(always)]
fn __first_node(srcp: *const __SlateRecord_nodemask_types_h_17) -> u32 {
let mut __v0: u64;
let __UNIQUE_ID_x__46: i32 = 64i32;
let __UNIQUE_ID_y__47: u64 = find_first_bit(unsafe { (*srcp).bits.as_ptr() as *const u64 }, 64u64);
loop {
if __types_ok__5!(__UNIQUE_ID_x__46, __UNIQUE_ID_y__47) == 0 {
unsafe { __compiletime_assert_48() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if (__UNIQUE_ID_x__46 as u64) < __UNIQUE_ID_y__47 { __UNIQUE_ID_x__46 as u64 } else { __UNIQUE_ID_y__47 };
__v0 as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
}


#[inline(always)]
fn __next_node(n: i32, srcp: *const __SlateRecord_nodemask_types_h_17) -> u32 {
let mut __v0: u64;
let __UNIQUE_ID_x__49: i32 = 64i32;
let __UNIQUE_ID_y__50: u64 = find_next_bit(unsafe { (*srcp).bits.as_ptr() as *const u64 }, 64u64, n.wrapping_add(1) as u64);
loop {
if __types_ok__5!(__UNIQUE_ID_x__49, __UNIQUE_ID_y__50) == 0 {
unsafe { __compiletime_assert_51() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if (__UNIQUE_ID_x__49 as u64) < __UNIQUE_ID_y__50 { __UNIQUE_ID_x__49 as u64 } else { __UNIQUE_ID_y__50 };
__v0 as u32
}

#[inline(always)]
fn __next_node_in(node: i32, srcp: *const __SlateRecord_nodemask_types_h_17) -> u32 {
let mut ret: u32 = __next_node(node, srcp);
if ret == 64u32 {
ret = __first_node(srcp);
}
ret
}

#[inline(always)]
fn __nodelist_parse(buf: *const u8, dstp: *mut __SlateRecord_nodemask_types_h_17, nbits: i32) -> i32 {
unsafe { bitmap_parselist(buf, (*dstp).bits.as_mut_ptr() as *mut u64, nbits) }
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
fn node_random(maskp: *const __SlateRecord_nodemask_types_h_17) -> i32 {
let node: i32 = (unsafe { find_random_bit((*maskp).bits.as_ptr() as *const u64, 64u64) }) as i32;
if node < 64i32 { node } else { 1i32.wrapping_neg() }
}

#[inline(always)]
fn cpu_max_bits_warn(cpu: u32, bits: u32) {
}

#[inline(always)]
fn cpumask_check(cpu: u32) -> u32 {
cpu_max_bits_warn(cpu, 64u32);
cpu
}

#[inline(always)]
fn cpumask_first_and(srcp1: *const cpumask, srcp2: *const cpumask) -> u32 {
find_first_and_bit(cpumask_bits!(srcp1), cpumask_bits!(srcp2), 64u64) as u32
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


// for_each_cpu - iterate over every cpu in a mask
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
//
// After the loop, cpu is >= nr_cpu_ids.
#[inline(always)]
fn cpumask_set_cpu(cpu: u32, dstp: *mut cpumask) {
set_bit(cpumask_check(cpu) as i64, cpumask_bits__2!(dstp) as *mut u64);
}

#[inline(always)]
fn cpumask_clear_cpu(cpu: i32, dstp: *mut cpumask) {
clear_bit(cpumask_check(cpu as u32) as i64, cpumask_bits__2!(dstp) as *mut u64);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
}


#[inline(always)]
fn cpumask_test_cpu(cpu: i32, cpumask: *const cpumask) -> bool {
test_bit!(cpumask_check(cpu as u32), unsafe { (*cpumask).bits.as_ptr() as *const u64 }) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
}


#[inline(always)]
fn cpumask_setall(dstp: *mut cpumask) {
if small_const_nbits__3!(64u32) {
unsafe {
*((*dstp).bits.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) = BITMAP_LAST_WORD_MASK!(unsafe { nr_cpu_ids });
}
return;
}
bitmap_fill(cpumask_bits__2!(dstp), unsafe { nr_cpu_ids });
}

#[inline(always)]
fn cpumask_clear(dstp: *mut cpumask) {
bitmap_zero(cpumask_bits__2!(dstp), 64u32);
}

#[inline(always)]
fn cpumask_and(dstp: *mut cpumask, src1p: *const cpumask, src2p: *const cpumask) -> bool {
bitmap_and(cpumask_bits__2!(dstp), cpumask_bits!(src1p), cpumask_bits!(src2p), 64u32)
}

#[inline(always)]
fn cpumask_or(dstp: *mut cpumask, src1p: *const cpumask, src2p: *const cpumask) {
bitmap_or(cpumask_bits__2!(dstp), cpumask_bits!(src1p), cpumask_bits!(src2p), 64u32);
}

#[inline(always)]
fn cpumask_andnot(dstp: *mut cpumask, src1p: *const cpumask, src2p: *const cpumask) -> bool {
bitmap_andnot(cpumask_bits__2!(dstp), cpumask_bits!(src1p), cpumask_bits!(src2p), 64u32)
}

#[inline(always)]
fn cpumask_equal(src1p: *const cpumask, src2p: *const cpumask) -> bool {
bitmap_equal(cpumask_bits!(src1p), cpumask_bits!(src2p), 64u32)
}

#[inline(always)]
fn cpumask_intersects(src1p: *const cpumask, src2p: *const cpumask) -> bool {
bitmap_intersects(cpumask_bits!(src1p), cpumask_bits!(src2p), 64u32)
}

#[inline(always)]
fn cpumask_subset(src1p: *const cpumask, src2p: *const cpumask) -> bool {
bitmap_subset(cpumask_bits!(src1p), cpumask_bits!(src2p), 64u32)
}

#[inline(always)]
fn cpumask_empty(srcp: *const cpumask) -> bool {
bitmap_empty(cpumask_bits!(srcp), 64u32)
}

#[inline(always)]
fn cpumask_weight(srcp: *const cpumask) -> u32 {
bitmap_weight(cpumask_bits!(srcp), 64u32)
}

#[inline(always)]
fn cpumask_weight_and(srcp1: *const cpumask, srcp2: *const cpumask) -> u32 {
bitmap_weight_and(cpumask_bits!(srcp1), cpumask_bits!(srcp2), 64u32) as u32
}

#[inline(always)]
fn cpumask_weight_andnot(srcp1: *const cpumask, srcp2: *const cpumask) -> u32 {
bitmap_weight_andnot(cpumask_bits!(srcp1), cpumask_bits!(srcp2), 64u32) as u32
}

#[inline(always)]
fn cpumask_copy(dstp: *mut cpumask, srcp: *const cpumask) {
bitmap_copy(cpumask_bits__2!(dstp), cpumask_bits!(srcp), 64u32);
}

#[inline(always)]
fn cpulist_parse(buf: *const u8, dstp: *mut cpumask) -> i32 {
unsafe { bitmap_parselist(buf, cpumask_bits__2!(dstp), nr_cpu_ids as i32) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
}


#[inline(always)]
fn alloc_cpumask_var(mask: *mut [cpumask; 1], flags: u32) -> bool {
true
}

#[inline(always)]
fn zalloc_cpumask_var(mask: *mut [cpumask; 1], flags: u32) -> bool {
cpumask_clear(unsafe { (*mask).as_mut_ptr() as *mut cpumask });
true
}

#[inline(always)]
fn free_cpumask_var(mask: *mut cpumask) {
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
fn num_online_cpus() -> u32 {
raw_atomic_read((unsafe { core::ptr::addr_of_mut!(__num_online_cpus) }) as *const __SlateRecord_types_h_190) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[inline(always)]
fn arch_local_irq_disable() {
native_irq_disable();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[inline(always)]
fn arch_local_irq_enable() {
native_irq_enable();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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
// We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
#[inline(always)]
fn preempt_count() -> i32 {
let mut __v0: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym __preempt_count, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v0 = pfo_val__;
(__v0 & !(!((1i64.wrapping_neg() as u64) >> 1i32))) as i32
}

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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
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
struct xol_area {
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


fn init_llist_node(node: *mut llist_node) {
loop {
loop {
if !(__native_word__20!() || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_60() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__29!((*node).next, node);
if true {
break;
}
}
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


fn llist_on_list(node: *const llist_node) -> bool {
let mut __v0: *mut llist_node;
loop {
if !(__native_word__20!() || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_61() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*node).next) as *const *mut llist_node) };
__v0 != (node as *mut llist_node)
}

// llist_for_each_entry - iterate over some deleted entries of lock-less list of given type
// @pos:	the type * to use as a loop cursor.
// @node:	the fist entry of deleted list entries.
// @member:	the name of the llist_node with the struct.
//
// In general, some entries of the lock-less list can be traversed
// safely only after being removed from list, so start with an entry
// instead of list head.
//
// If being used on entries deleted from lock-less list directly, the
// traverse order is from the newest to the oldest added entry.  If
// you want to traverse from the oldest to the newest, you must
// reverse the order by yourself before traversing.
// llist_for_each_entry_safe - iterate over some deleted entries of lock-less list of given type
// 			       safe against removal of list entry
// @pos:	the type * to use as a loop cursor.
// @n:		another type * to use as temporary storage
// @node:	the first entry of deleted list entries.
// @member:	the name of the llist_node with the struct.
//
// In general, some entries of the lock-less list can be traversed
// safely only after being removed from list, so start with an entry
// instead of list head.
//
// If being used on entries deleted from lock-less list directly, the
// traverse order is from the newest to the oldest added entry.  If
// you want to traverse from the oldest to the newest, you must
// reverse the order by yourself before traversing.
fn llist_add_batch(new_first: *mut llist_node, new_last: *mut llist_node, head: *mut llist_head) -> bool {
let mut first: *mut llist_node = unsafe { core::mem::zeroed() };
let mut __v10: *mut llist_node;
loop {
if !(__native_word__20!() || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_64() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*head).first) as *const *mut llist_node) };
first = __v10;
loop {
unsafe {
(*new_last).next = first;
}
let mut __v13: i64;
let __v0: *mut *mut llist_node = unsafe { core::ptr::addr_of_mut!((*head).first) };
let __v1: *mut *mut llist_node = core::ptr::addr_of_mut!(first);
kcsan_mb!();
instrument_atomic_read_write(__v0 as *const (), core::mem::size_of::<*mut llist_node>() as u64);
instrument_read_write(__v1 as *const (), core::mem::size_of::<*mut llist_node>() as u64);
let mut __v15: i64;
let mut __v2: bool = false;
let mut __v4: *mut llist_node = unsafe { *__v1 };
let mut __v5: *mut llist_node = new_first;
match (const { core::mem::size_of::<*mut llist_node>() }) as u64 {
1 => {
let mut __v6: *mut u8 = __v0 as *mut u8;
unsafe {
let __t0: u8;
let __t1: u64;
core::arch::asm!("lock cmpxchgb {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t0, in(reg) __v6, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t1, options(nostack, att_syntax));
__v2 = (__t0 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t1 as u64) };
}
}
2 => {
let mut __v7: *mut u16 = __v0 as *mut u16;
unsafe {
let __t2: u8;
let __t3: u64;
core::arch::asm!("lock cmpxchgw {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t2, in(reg) __v7, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t3, options(nostack, att_syntax));
__v2 = (__t2 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t3 as u64) };
}
}
4 => {
let mut __v8: *mut u32 = __v0 as *mut u32;
unsafe {
let __t4: u8;
let __t5: u64;
core::arch::asm!("lock cmpxchgl {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t4, in(reg) __v8, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t5, options(nostack, att_syntax));
__v2 = (__t4 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t5 as u64) };
}
}
8 => {
let mut __v9: *mut u64 = __v0 as *mut u64;
unsafe {
let __t6: u8;
let __t7: u64;
core::arch::asm!("lock cmpxchgq {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t6, in(reg) __v9, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t7, options(nostack, att_syntax));
__v2 = (__t6 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t7 as u64) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
if unlikely__2!(!__v2) != 0 {
unsafe {
*__v1 = __v4;
}
}
__v15 = __v2 as i64;
__v13 = __v15;
if __v13 != 0 {
break;
}
}
first.is_null()
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


fn llist_add(new: *mut llist_node, head: *mut llist_head) -> bool {
llist_add_batch(new, new, head)
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


fn __llist_del_all(head: *mut llist_head) -> *mut llist_node {
let first: *mut llist_node = unsafe { (*head).first };
unsafe {
(*head).first = core::ptr::null_mut::<llist_node>();
}
first
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


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root {
    rb_node: *mut rb_node,
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
struct memory_tier {
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
struct irq_domain {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
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
struct key_type {
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
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount {
    sequence: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_raw_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_rwlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_mutex {
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
struct tty_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
    __slate_bits_0: __slate_bits::__SlateBits290U0,
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
    __slate_bits_0: __slate_bits::__SlateBits67U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits67U1,
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
    __slate_bits_2: __slate_bits::__SlateBits67U2,
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
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
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


#[inline(always)]
fn class_rcu_constructor() -> __SlateRecord_rcupdate_h_1216 {
let _t: __SlateRecord_rcupdate_h_1216 = __SlateRecord_rcupdate_h_1216 { lock: 1i32 as *mut () };
rcu_read_lock();
return _t;
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
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
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
struct alloc_tag_counters {
    bytes: u64,
    calls: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct request_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
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
// wait_event - sleep until a condition gets true
// @wq_head: the waitqueue to wait on
// @condition: a C expression for the event to wait for
//
// The process is put to sleep (TASK_UNINTERRUPTIBLE) until the
// @condition evaluates to true. The @condition is checked each time
// the waitqueue @wq_head is woken up.
//
// wake_up() has to be called after changing any variable that could
// change the result of the wait condition.
#[repr(C)]
#[derive(Clone, Copy)]
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
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
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
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
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
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


// SEQCOUNT_LOCKNAME()	- Instantiate seqcount_LOCKNAME_t and helpers
// seqprop_LOCKNAME_*()	- Property accessors for seqcount_LOCKNAME_t
//
// @lockname:		"LOCKNAME" part of seqcount_LOCKNAME_t
// @locktype:		LOCKNAME canonical C data type
// @preemptible:	preemptibility of above locktype
// @lockbase:		prefix for associated lock/unlock
// 									 * Re-read the sequence counter since the (possibly			 * preempted) writer made progress.
// PREEMPT_RT relies on the above LOCK+UNLOCK
fn __seqprop_ptr(s: &seqcount) -> *mut seqcount {
(s as *const seqcount) as *mut seqcount
}

fn __seqprop_preemptible(s: &seqcount) -> bool {
false
}

fn __seqprop_assert(s: &seqcount) {
lockdep_assert_preemption_disabled!();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
}


#[inline(always)]
fn __seqprop_raw_spinlock_ptr(s: &mut seqcount_raw_spinlock) -> *mut seqcount {
unsafe { core::ptr::addr_of_mut!((*(s as *mut seqcount_raw_spinlock)).seqcount) }
}

#[inline(always)]
fn __seqprop_raw_spinlock_preemptible(s: &seqcount_raw_spinlock) -> bool {
if IS_ENABLED!() == 0 {
return false;
}
false
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


#[inline(always)]
fn __seqprop_raw_spinlock_assert(s: &seqcount_raw_spinlock) {
}

#[inline(always)]
fn __seqprop_spinlock_ptr(s: *mut seqcount_spinlock) -> *mut seqcount {
unsafe { core::ptr::addr_of_mut!((*s).seqcount) }
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


#[inline(always)]
fn __seqprop_spinlock_preemptible(s: *const seqcount_spinlock) -> bool {
if IS_ENABLED!() == 0 {
return false;
}
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
}


#[inline(always)]
fn __seqprop_spinlock_assert(s: *const seqcount_spinlock) {
}

#[inline(always)]
fn __seqprop_rwlock_ptr(s: &mut seqcount_rwlock) -> *mut seqcount {
unsafe { core::ptr::addr_of_mut!((*(s as *mut seqcount_rwlock)).seqcount) }
}

#[inline(always)]
fn __seqprop_rwlock_preemptible(s: &seqcount_rwlock) -> bool {
if IS_ENABLED!() == 0 {
return false;
}
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
}


#[inline(always)]
fn __seqprop_rwlock_assert(s: &seqcount_rwlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
}


#[inline(always)]
fn __seqprop_mutex_ptr(s: &mut seqcount_mutex) -> *mut seqcount {
unsafe { core::ptr::addr_of_mut!((*(s as *mut seqcount_mutex)).seqcount) }
}

#[inline(always)]
fn __seqprop_mutex_preemptible(s: &seqcount_mutex) -> bool {
if IS_ENABLED!() == 0 {
return true;
}
false
}

#[inline(always)]
fn __seqprop_mutex_assert(s: &seqcount_mutex) {
}

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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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

fn do_write_seqcount_begin_nested(s: *mut seqcount, subclass: i32) {
seqcount_acquire!();
do_raw_write_seqcount_begin(s);
}

// write_seqcount_begin() - start a seqcount_t write side critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Context: sequence counter write side sections must be serialized and
// non-preemptible. Preemption will be automatically disabled if and
// only if the seqcount write serialization lock is associated, and
// preemptible.  If readers can be invoked from hardirq or softirq
// context, interrupts or bottom halves must be respectively disabled.
fn do_write_seqcount_begin(s: *mut seqcount) {
do_write_seqcount_begin_nested(s, 0i32);
}

// write_seqcount_end() - end a seqcount_t write side critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Context: Preemption will be automatically re-enabled if and only if
// the seqcount write serialization lock is associated, and preemptible.
fn do_write_seqcount_end(s: *mut seqcount) {
seqcount_release!();
do_raw_write_seqcount_end(s);
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
struct kref {
    refcount: refcount_struct,
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
struct rw_semaphore {
    count: __SlateRecord_types_h_197,
    owner: __SlateRecord_types_h_197,
    osq: optimistic_spin_queue,
    wait_lock: raw_spinlock,
    first_waiter: *mut rwsem_waiter,
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
struct rcu_work {
    work: work_struct,
    rcu: callback_head,
    wq: *mut workqueue_struct,
}


fn __init_work(work: *mut work_struct, onstack: i32) {
}

// alloc_ordered_workqueue - allocate an ordered workqueue
// @fmt: printf format for the name of the workqueue
// @flags: WQ_* flags (only WQ_FREEZABLE and WQ_MEM_RECLAIM are meaningful)
// @args: args for @fmt
//
// Allocate an ordered workqueue.  An ordered workqueue executes at
// most one work item at any given time in the queued order.  They are
// implemented as unbound workqueues with @max_active of one.
//
// RETURNS:
// Pointer to the allocated workqueue on success, %NULL on failure.
fn queue_work(wq: *mut workqueue_struct, work: *mut work_struct) -> bool {
unsafe { queue_work_on(64i32, wq, work) }
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
struct cgroup_taskset {
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
struct srcu_struct {
    srcu_ctrp: *mut srcu_ctr,
    sda: *mut srcu_data,
    srcu_reader_flavor: u8,
    dep_map: lockdep_map,
    srcu_sup: *mut srcu_usage,
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
struct notifier_block {
    notifier_call: Option<unsafe extern "C" fn(*mut notifier_block, u64, *mut ()) -> i32>,
    next: *mut notifier_block,
    priority: i32,
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
struct sock {
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
struct wake_irq {
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
struct dev_pm_qos {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits432U0,
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


// Clear the access bit
// Clear the dirty bit
// mm is unstable for copy_from_user
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
struct folio_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct free_area {
    free_list: [list_head; 4],
    nr_free: u64,
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
fn zonelist_zone(zoneref: *mut zoneref) -> *mut zone {
unsafe { (*zoneref).zone }
}

fn zonelist_zone_idx(zoneref: *const zoneref) -> i32 {
unsafe { (*zoneref).zone_idx }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
}


#[inline(always)]
fn next_zones_zonelist(z: *mut zoneref, highest_zoneidx: u32, nodes: *const __SlateRecord_nodemask_types_h_17) -> *mut zoneref {
let __v0: bool = nodes.is_null() && (zonelist_zone_idx(z as *const zoneref) as u32) <= (highest_zoneidx as u32);
if __v0 {
return z;
}
unsafe { __next_zones_zonelist(z, highest_zoneidx, nodes) }
}

fn first_zones_zonelist(zonelist: *mut zonelist, highest_zoneidx: u32, nodes: *const __SlateRecord_nodemask_types_h_17) -> *mut zoneref {
next_zones_zonelist(unsafe { (*zonelist)._zonerefs.as_mut_ptr() as *mut zoneref }, highest_zoneidx, nodes)
}

fn movable_only_nodes(nodes: *mut __SlateRecord_nodemask_types_h_17) -> bool {
let mut nid: i32 = 0;
if nodes_empty!(*nodes) {
return false;
}
// We can chose arbitrary node from the nodemask to get a
// zonelist as they are interlinked. We just need to find
// at least one zone that can satisfy kernel allocations.
nid = first_node!(*nodes) as i32;
let zonelist: *mut zonelist = unsafe { (*NODE_DATA!(nid)).node_zonelists.as_mut_ptr() as *mut zonelist }.wrapping_offset(0 as isize);
let z: *mut zoneref = first_zones_zonelist(zonelist, 2u32, nodes as *const __SlateRecord_nodemask_types_h_17);
zonelist_zone(z).is_null()
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
struct list_lru {
    node: *mut list_lru_node,
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
struct hlist_bl_node {
    next: *mut hlist_bl_node,
    pprev: *mut *mut hlist_bl_node,
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


fn task_lock(p: *mut task_struct) {
spin_lock(unsafe { core::ptr::addr_of_mut!((*p).alloc_lock) });
}

fn task_unlock(p: *mut task_struct) {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*p).alloc_lock) });
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
struct __SlateRecord_capability_h_24 {
    val: u64,
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
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
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
union __SlateRecord_mempolicy_h_54 {
    cpuset_mems_allowed: __SlateRecord_nodemask_types_h_17,
    user_nodemask: __SlateRecord_nodemask_types_h_17,
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
struct fs_pin {
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
struct __SlateRecord_cpuset_c_584 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cpuset_c_632 {
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


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits290U0 {
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
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits114U0 {
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
    pub struct __SlateBits578U0 {
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
    pub struct __SlateBits611U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits741U0 {
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
    pub struct __SlateBits738U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits741U1 {
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
    pub struct __SlateBits741U2 {
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
    pub struct __SlateBits741U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits672U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits623U0 {
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
    pub struct __SlateBits121U0 {
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
    pub struct __SlateBits432U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits67U0 {
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
    pub struct __SlateBits67U1 {
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
    pub struct __SlateBits697U0 {
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
    pub struct __SlateBits293U0 {
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
    pub struct __SlateBits67U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits122U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
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
    __slate_bits_0: __slate_bits::__SlateBits293U0,
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
fn thread_group_leader(p: *mut task_struct) -> bool {
(unsafe { (*p).exit_signal }) >= 0
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
if !(__native_word__7!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
struct __SlateRecord_mnt_idmapping_h_21 {
    val: u32,
}


// vfs{g,u}ids are created from k{g,u}ids.
// We don't allow them to be created from regular {u,g}id.
// Allow a vfs{g,u}id to be used as a k{g,u}id where we want to compare
// whether the mapped value is identical to value of a k{g,u}id.
#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits611U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


fn __ref_is_percpu(r#ref: *mut percpu_ref, percpu_countp: &mut *mut u64) -> bool {
let mut percpu_ptr: u64 = 0;
// The value of @ref->percpu_count_ptr is tested for
// !__PERCPU_REF_ATOMIC, which may be set asynchronously, and then
// used as a pointer.  If the compiler generates a separate fetch
// when using it as a pointer, __PERCPU_REF_ATOMIC may be set in
// between contaminating the pointer value, meaning that
// READ_ONCE() is required when fetching it.
//
// The dependency ordering from the READ_ONCE() pairs
// with smp_store_release() in __percpu_ref_switch_to_percpu().
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_385() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*r#ref).percpu_count_ptr) as *const u64) };
percpu_ptr = __v0;
// Theoretically, the following could test just ATOMIC; however,
// then we'd have to mask off DEAD separately as DEAD may be
// visible without ATOMIC if we race with percpu_ref_kill().  DEAD
// implies ATOMIC anyway.  Test them together.
if unlikely__4!(percpu_ptr & 3) != 0 {
return false;
}
unsafe {
*(percpu_countp as *mut *mut u64) = percpu_ptr as *mut u64;
}
true
}

fn percpu_ref_get_many(r#ref: *mut percpu_ref, nr: u64) {
let mut percpu_count: *mut u64 = unsafe { core::mem::zeroed() };
rcu_read_lock();
if __ref_is_percpu(r#ref, unsafe { &mut (*core::ptr::addr_of_mut!(percpu_count)) }) {
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = (nr & 255) as u8;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = (nr & 65535) as u16;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = (nr & 4294967295) as u32;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = nr;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
atomic_long_add(nr as i64, unsafe { core::ptr::addr_of_mut!((*(*r#ref).data).count) });
}
rcu_read_unlock();
}

fn percpu_ref_get(r#ref: *mut percpu_ref) {
percpu_ref_get_many(r#ref, 1u64);
}

fn percpu_ref_tryget_live_rcu(r#ref: *mut percpu_ref) -> bool {
let mut percpu_count: *mut u64 = unsafe { core::mem::zeroed() };
let mut ret: bool = false;
let mut __v0: i64;
let __ret_warn_on: i32 = (rcu_read_lock_held() == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("386: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 386b - ., 3; .popsection\n# {0}", const 386 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_percpu_ref_tryget_live_rcu_1, sym __slate_asm_string_percpu_ref_tryget_live_rcu_2, const 281 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("387: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 387b - ., 4; .popsection\n# {0}", const 387 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if likely!(__ref_is_percpu(r#ref, unsafe { &mut (*core::ptr::addr_of_mut!(percpu_count)) })) != 0 {
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
core::arch::asm!("incb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
core::arch::asm!("incw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
core::arch::asm!("incl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
core::arch::asm!("incq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
ret = true;
} else {
if (unsafe { (*r#ref).percpu_count_ptr }) & 2 == 0 {
ret = atomic_long_inc_not_zero(unsafe { core::ptr::addr_of_mut!((*(*r#ref).data).count) });
}
}
ret
}

fn percpu_ref_tryget_live(r#ref: *mut percpu_ref) -> bool {
let mut ret: bool = false;
rcu_read_lock();
ret = percpu_ref_tryget_live_rcu(r#ref);
rcu_read_unlock();
ret
}

fn percpu_ref_put_many(r#ref: *mut percpu_ref, nr: u64) {
let mut percpu_count: *mut u64 = unsafe { core::mem::zeroed() };
rcu_read_lock();
if __ref_is_percpu(r#ref, unsafe { &mut (*core::ptr::addr_of_mut!(percpu_count)) }) {
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = (nr.wrapping_neg() & 255) as u8;
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = (nr.wrapping_neg() & 65535) as u16;
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = (nr.wrapping_neg() & 4294967295) as u32;
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = nr.wrapping_neg();
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
if unlikely__2!(atomic_long_sub_and_test(nr as i64, unsafe { core::ptr::addr_of_mut!((*(*r#ref).data).count) })) != 0 {
unsafe { unsafe { (*(*r#ref).data).release }.unwrap()(r#ref) };
}
}
rcu_read_unlock();
}

fn percpu_ref_put(r#ref: *mut percpu_ref) {
percpu_ref_put_many(r#ref, 1u64);
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
unsafe { __compiletime_assert_390() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("391: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 391b - ., 3; .popsection\n# {0}", const 391 as i32, options(nostack, att_syntax));
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
// kmalloc_objs - Allocate an array of the given type
// @VAR_OR_TYPE: Variable or type to allocate an array of.
// @COUNT: How many elements in the array.
// @...: optional GFP flags for the allocation (GFP_KERNEL when not specified).
//
// Returns: newly allocated pointer to array of @VAR_OR_TYPE on success,
// or NULL on failure.
// All kzalloc aliases for kmalloc_(obj|objs|flex).
fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
}

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


// XArray tags, for tagging dirty and writeback pages in the pagecache.
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


#[inline(always)]
fn seq_puts(m: *mut seq_file, s: *const u8) {
if !unsafe { __slate_intrinsic_70b71f185ebb8c4a(*s) } {
unsafe { __seq_puts(m, s) };
} else {
if (unsafe { *s.wrapping_offset(0 as isize) }) != 0 && (unsafe { *s.wrapping_offset(1 as isize) }) == 0 {
unsafe { seq_putc(m, *s.wrapping_offset(0 as isize)) };
} else {
unsafe { seq_write(m, s as *const (), __builtin_strlen(s)) };
}
}
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
    __slate_bits_0: __slate_bits::__SlateBits672U0,
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
    __slate_bits_0: __slate_bits::__SlateBits697U0,
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
struct css_task_iter {
    ss: *mut cgroup_subsys,
    flags: u32,
    cset_pos: *mut list_head,
    cset_head: *mut list_head,
    tcset_pos: *mut list_head,
    tcset_head: *mut list_head,
    task_pos: *mut list_head,
    cur_tasks_head: *mut list_head,
    cur_cset: *mut css_set,
    cur_dcset: *mut css_set,
    cur_task: *mut task_struct,
    iters_node: list_head,
}


fn css_get(css: *mut cgroup_subsys_state) {
if (unsafe { (*css).flags }) & 1 == 0 {
percpu_ref_get(unsafe { core::ptr::addr_of_mut!((*css).refcnt) });
}
}

fn css_tryget_online(css: *mut cgroup_subsys_state) -> bool {
if (unsafe { (*css).flags }) & 1 == 0 {
return percpu_ref_tryget_live(unsafe { core::ptr::addr_of_mut!((*css).refcnt) });
}
true
}

fn css_put(css: *mut cgroup_subsys_state) {
if (unsafe { (*css).flags }) & 1 == 0 {
percpu_ref_put(unsafe { core::ptr::addr_of_mut!((*css).refcnt) });
}
}

// cgroup_subsys_on_dfl - fast test on whether a subsys is on default hierarchy
// @ss: subsystem in question
// css_for_each_child - iterate through children of a css
// @pos: the css * to use as the loop cursor
// @parent: css whose children to walk
//
// Walk @parent's children.  Must be called under rcu_read_lock().
//
// If a subsystem synchronizes ->css_online() and the start of iteration, a
// css which finished ->css_online() is guaranteed to be visible in the
// future iterations and will stay visible until the last reference is put.
// A css which hasn't finished ->css_online() or already finished
// ->css_offline() may show up during traversal.  It's each subsystem's
// responsibility to synchronize against on/offlining.
//
// It is allowed to temporarily drop RCU read lock during iteration.  The
// caller is responsible for ensuring that @pos remains accessible until
// the start of the next iteration by, for example, bumping the css refcnt.
// css_for_each_descendant_pre - pre-order walk of a css's descendants
// @pos: the css * to use as the loop cursor
// @root: css whose descendants to walk
//
// Walk @root's descendants.  @root is included in the iteration and the
// first node to be visited.  Must be called under rcu_read_lock().
//
// If a subsystem synchronizes ->css_online() and the start of iteration, a
// css which finished ->css_online() is guaranteed to be visible in the
// future iterations and will stay visible until the last reference is put.
// A css which hasn't finished ->css_online() or already finished
// ->css_offline() may show up during traversal.  It's each subsystem's
// responsibility to synchronize against on/offlining.
//
// For example, the following guarantees that a descendant can't escape
// state updates of its ancestors.
//
// my_online(@css)
// {
// 	Lock @css's parent and @css;
// 	Inherit state from the parent;
// 	Unlock both.
// }
//
// my_update_state(@css)
// {
// 	css_for_each_descendant_pre(@pos, @css) {
// 		Lock @pos;
// 		if (@pos == @css)
// 			Update @css's state;
// 		else
// 			Verify @pos is alive and inherit state from its parent;
// 		Unlock @pos;
// 	}
// }
//
// As long as the inheriting step, including checking the parent state, is
// enclosed inside @pos locking, double-locking the parent isn't necessary
// while inheriting.  The state update to the parent is guaranteed to be
// visible by walking order and, as long as inheriting operations to the
// same @pos are atomic to each other, multiple updates racing each other
// still result in the correct state.  It's guaranateed that at least one
// inheritance happens for any css after the latest update to its parent.
//
// If checking parent's state requires locking the parent, each inheriting
// iteration should lock and unlock both @pos->parent and @pos.
//
// Alternatively, a subsystem may choose to use a single global lock to
// synchronize ->css_online() and ->css_offline() against tree-walking
// operations.
//
// It is allowed to temporarily drop RCU read lock during iteration.  The
// caller is responsible for ensuring that @pos remains accessible until
// the start of the next iteration by, for example, bumping the css refcnt.
// cgroup_taskset_for_each - iterate cgroup_taskset
// @task: the loop cursor
// @dst_css: the destination css
// @tset: taskset to iterate
//
// @tset may contain multiple tasks and they may belong to multiple
// processes.
//
// On the v2 hierarchy, there may be tasks from multiple processes and they
// may not share the source or destination csses.
//
// On traditional hierarchies, when there are multiple tasks in @tset, if a
// task of a process is in @tset, all tasks of the process are in @tset.
// Also, all are guaranteed to share the same source and destination csses.
//
// Iteration is not in any specific order.
fn css_is_dying(css: *mut cgroup_subsys_state) -> bool {
(unsafe { (*css).flags }) & 16 != 0
}

fn css_is_online(css: *mut cgroup_subsys_state) -> bool {
(unsafe { (*css).flags }) & 2 != 0
}

// task_css_check - obtain css for (task, subsys) w/ extra access conds
// @task: the target task
// @subsys_id: the target subsystem ID
// @__c: extra condition expression to be passed to rcu_dereference_check()
//
// Return the cgroup_subsys_state for the (@task, @subsys_id) pair.  The
// synchronization rules are the same as task_css_set_check().
fn task_css(task: *mut task_struct, subsys_id: i32) -> *mut cgroup_subsys_state {
let mut __v0: *mut css_set;
let mut __v1: *mut css_set;
loop {
if !(__native_word__18!() || ((const { core::mem::size_of::<*mut css_set>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_470() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*task).cgroups) as *const *mut css_set) };
let __UNIQUE_ID_rcu_469: *mut css_set = __v1;
loop {
let __v4: bool = if false {
let __v5: bool = false || rcu_read_lock_held() != 0;
!__v5
} else {
false
};
if !__v4 {
break;
}
}
__v0 = __UNIQUE_ID_rcu_469;
unsafe { *((*__v0).subsys.as_mut_ptr() as *mut *mut cgroup_subsys_state).wrapping_offset(subsys_id as isize) }
}

fn cgroup_has_tasks(cgrp: *mut cgroup) -> bool {
let mut __v0: i32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_476() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*cgrp).self_.nr_populated_csets) as *const i32) };
__v0 != 0
}

fn css_is_populated(css: *mut cgroup_subsys_state) -> bool {
let mut __v0: i32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_477() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*css).nr_populated_csets) as *const i32) };
let __v2: bool = if __v0 != 0 {
true
} else {
let mut __v3: i32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_478() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*css).nr_populated_children) as *const i32) };
__v3 != 0
};
__v2
}

fn cgroup_is_populated(cgrp: *mut cgroup) -> bool {
css_is_populated(unsafe { core::ptr::addr_of_mut!((*cgrp).self_) })
}

fn of_cft(of: *mut kernfs_open_file) -> *mut cftype {
(unsafe { (*(*of).kn).r#priv }) as *mut cftype
}

fn seq_cft(seq: *mut seq_file) -> *mut cftype {
of_cft((unsafe { (*seq).private }) as *mut kernfs_open_file)
}

fn seq_css(seq: *mut seq_file) -> *mut cgroup_subsys_state {
unsafe { of_css((*seq).private as *mut kernfs_open_file) }
}

fn pr_cont_cgroup_name(cgrp: *mut cgroup) {
unsafe { pr_cont_kernfs_name((*cgrp).kn) };
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
    __slate_bits_0: __slate_bits::__SlateBits623U0,
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


// Generate SD flag bits
#[repr(C)]
#[derive(Clone, Copy)]
struct sched_domain_attr {
    relax_domain_level: i32,
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
    __slate_bits_0: __slate_bits::__SlateBits741U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits741U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits741U2,
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
    __slate_bits_3: __slate_bits::__SlateBits741U3,
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
    __slate_bits_0: __slate_bits::__SlateBits738U0,
}


fn hotplug_node_notifier(r#fn: Option<unsafe extern "C" fn(*mut notifier_block, u64, *mut ()) -> i32>, pri: i32) -> i32 {
0
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


// Helper macro which bitwise-or combines the specified input flags into a
// vma_flags_t bitmap value. E.g.:
//
// vma_flags_t flags = mk_vma_flags(VMA_IO_BIT, VMA_PFNMAP_BIT,
//              VMA_DONTEXPAND_BIT, VMA_DONTDUMP_BIT);
//
// The compiler cleverly optimises away all of the work and this ends up being
// equivalent to aggregating the values manually.
// Test whether ALL specified VMA flags are set, e.g.:
//
// if (vma_flags_test_all(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
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
// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
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
fn cpuset_inc() {
static_branch_inc_cpuslocked!(unsafe { core::ptr::addr_of_mut!(cpusets_pre_enable_key) });
static_branch_inc_cpuslocked!(unsafe { core::ptr::addr_of_mut!(cpusets_enabled_key) });
}

fn cpuset_dec() {
static_branch_dec_cpuslocked!(unsafe { core::ptr::addr_of_mut!(cpusets_enabled_key) });
static_branch_dec_cpuslocked!(unsafe { core::ptr::addr_of_mut!(cpusets_pre_enable_key) });
}

fn cpusets_insane_config() -> bool {
static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(cpusets_insane_config_key) }) != 0
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
struct property {
    name: *mut u8,
    length: i32,
    value: *mut (),
    next: *mut property,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
    name: *const u8,
    phandle: u32,
    full_name: *const u8,
    fwnode: fwnode_handle,
    properties: *mut property,
    deadprops: *mut property,
    parent: *mut device_node,
    child: *mut device_node,
    sibling: *mut device_node,
    _flags: u64,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpuset {
    css: cgroup_subsys_state,
    flags: u64,
    cpus_allowed: [cpumask; 1],
    mems_allowed: __SlateRecord_nodemask_types_h_17,
    effective_cpus: [cpumask; 1],
    effective_mems: __SlateRecord_nodemask_types_h_17,
    effective_xcpus: [cpumask; 1],
    exclusive_cpus: [cpumask; 1],
    old_mems_allowed: __SlateRecord_nodemask_types_h_17,
    attach_node: llist_node,
    partition_root_state: i32,
    remote_partition: bool,
    nr_deadline_tasks: __SlateRecord_types_h_190,
    nr_migrate_dl_tasks: i32,
    sum_migrate_dl_bw: u64,
    dl_bw_cpu: i32,
    prs_err: u32,
    partition_file: cgroup_file,
}


fn css_cs(css: *mut cgroup_subsys_state) -> *mut cpuset {
let __v0: *mut cpuset = if !css.is_null() {
container_of!(css, cpuset)
} else {
core::ptr::null_mut::<cpuset>()
};
__v0
}

fn task_cs(task: *mut task_struct) -> *mut cpuset {
css_cs(task_css(task, 0i32))
}

fn parent_cs(cs: *mut cpuset) -> *mut cpuset {
css_cs(unsafe { (*cs).css.parent })
}

fn is_cpuset_online(cs: *mut cpuset) -> bool {
css_is_online(unsafe { core::ptr::addr_of_mut!((*cs).css) }) && !css_is_dying(unsafe { core::ptr::addr_of_mut!((*cs).css) })
}

fn is_cpu_exclusive(cs: *const cpuset) -> i32 {
test_bit__4!(0i32, unsafe { core::ptr::addr_of!((*cs).flags) })
}

fn is_mem_exclusive(cs: *const cpuset) -> i32 {
test_bit__4!(1i32, unsafe { core::ptr::addr_of!((*cs).flags) })
}

fn is_mem_hardwall(cs: *const cpuset) -> i32 {
test_bit__4!(2i32, unsafe { core::ptr::addr_of!((*cs).flags) })
}

fn is_sched_load_balance(cs: *const cpuset) -> i32 {
test_bit__4!(4i32, unsafe { core::ptr::addr_of!((*cs).flags) })
}

fn is_memory_migrate(cs: *const cpuset) -> i32 {
test_bit__4!(3i32, unsafe { core::ptr::addr_of!((*cs).flags) })
}

fn is_spread_page(cs: *const cpuset) -> i32 {
test_bit__4!(5i32, unsafe { core::ptr::addr_of!((*cs).flags) })
}

fn cpusets_overlap(a: *mut cpuset, b: *mut cpuset) -> i32 {
cpumask_intersects((unsafe { (*a).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { (*b).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask) as i32
}

fn nr_cpusets() -> i32 {
// jump label reference count + the top-level cpuset
unsafe { static_key_count(core::ptr::addr_of_mut!(cpusets_enabled_key.key)) }.wrapping_add(1)
}

fn cpuset_is_populated(cs: *mut cpuset) -> bool {
lockdep_assert_cpuset_lock_held();
cgroup_is_populated(unsafe { (*cs).css.cgroup })
}

// cpuset_for_each_child - traverse online children of a cpuset
// @child_cs: loop cursor pointing to the current child
// @pos_css: used for iteration
// @parent_cs: target cpuset to walk children of
//
// Walk @child_cs through the online children of @parent_cs.  Must be used
// with RCU read locked.
// cpuset_for_each_descendant_pre - pre-order walk of a cpuset's descendants
// @des_cs: loop cursor pointing to the current descendant
// @pos_css: used for iteration
// @root_cs: target cpuset to walk ancestor of
//
// Walk @des_cs through the online descendants of @root_cs.  Must be used
// with RCU read locked.  The caller may modify @pos_css by calling
// css_rightmost_descendant() to skip subtree.  @root_cs is included in the
// iteration and the first node to be visited.
fn cpuset1_update_task_spread_flags(cs: *mut cpuset, tsk: *mut task_struct) {
}

fn cpuset1_update_tasks_flags(cs: *mut cpuset) {
}

fn cpuset1_hotplug_update_tasks(cs: *mut cpuset, new_cpus: *mut cpumask, new_mems: *mut __SlateRecord_nodemask_types_h_17, cpus_updated: bool, mems_updated: bool) {
}

fn cpuset1_validate_change(cur: *mut cpuset, trial: *mut cpuset) -> i32 {
0
}

fn cpuset1_cpus_excl_conflict(cs1: *mut cpuset, cs2: *mut cpuset) -> bool {
false
}

fn cpuset1_init(cs: *mut cpuset) {
}

fn cpuset1_online_css(css: *mut cgroup_subsys_state) {
}

fn cpuset1_generate_sched_domains(domains: &mut *mut [cpumask; 1], attributes: &mut *mut sched_domain_attr) -> i32 {
0
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


fn dl_prio(prio: i32) -> bool {
unlikely__2!(prio < 0) != 0
}

fn dl_task(p: *mut task_struct) -> bool {
dl_prio(unsafe { (*p).prio })
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
    __slate_bits_0: __slate_bits::__SlateBits578U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


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


fn tsk_is_oom_victim(tsk: *mut task_struct) -> bool {
!unsafe { (*(*tsk).signal).oom_mm }.is_null()
}

fn init_task_work(twork: *mut callback_head, func: Option<unsafe extern "C" fn(*mut callback_head)>) {
unsafe {
(*twork).func = func;
}
}

#[unsafe(no_mangle)]
static mut cpusets_pre_enable_key: static_key_false = static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} };

#[unsafe(no_mangle)]
static mut cpusets_enabled_key: static_key_false = static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} };

/// There could be abnormal cpuset configurations for cpu or memory
/// node binding, add this key to provide a quick low-cost judgment
/// of the situation.
#[unsafe(no_mangle)]
static mut cpusets_insane_config_key: static_key_false = static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} };

static perr_strings: __SlateAlign16<[*const u8; 11]> = __SlateAlign16({
let mut __t0: [*const u8; 11] = unsafe { core::mem::zeroed() };
__t0[1] = (b"Invalid cpu list in cpuset.cpus.exclusive\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"Parent is an invalid partition root\0".as_ptr() as *mut u8) as *const u8;
__t0[3] = (b"Parent is not a partition root\0".as_ptr() as *mut u8) as *const u8;
__t0[4] = (b"Cpu list in cpuset.cpus not exclusive\0".as_ptr() as *mut u8) as *const u8;
__t0[5] = (b"Parent unable to distribute cpu downstream\0".as_ptr() as *mut u8) as *const u8;
__t0[6] = (b"No cpu available due to hotplug\0".as_ptr() as *mut u8) as *const u8;
__t0[7] = (b"cpuset.cpus and cpuset.cpus.exclusive are empty\0".as_ptr() as *mut u8) as *const u8;
__t0[8] = (b"partition config conflicts with housekeeping setup\0".as_ptr() as *mut u8) as *const u8;
__t0[9] = (b"Enable partition not permitted\0".as_ptr() as *mut u8) as *const u8;
__t0[10] = (b"Have remote partition underneath\0".as_ptr() as *mut u8) as *const u8;
__t0
});

// CPUSET Locking Convention
//
// Below are the four global/local locks guarding cpuset structures in lock
// acquisition order:
//  - cpuset_top_mutex
//  - cpu_hotplug_lock (cpus_read_lock/cpus_write_lock)
//  - cpuset_mutex
//  - callback_lock (raw spinlock)
//
// As cpuset will now indirectly flush a number of different workqueues in
// housekeeping_update() to update housekeeping cpumasks when the set of
// isolated CPUs is going to be changed, it may be vulnerable to deadlock
// if we hold cpus_read_lock while calling into housekeeping_update().
//
// The first cpuset_top_mutex will be held except when calling into
// cpuset_handle_hotplug() from the CPU hotplug code where cpus_write_lock
// and cpuset_mutex will be held instead. The main purpose of this mutex
// is to prevent regular cpuset control file write actions from interfering
// with the call to housekeeping_update(), though CPU hotplug operation can
// still happen in parallel. This mutex also provides protection for some
// internal variables.
//
// A task must hold all the remaining three locks to modify externally visible
// or used fields of cpusets, though some of the internally used cpuset fields
// and internal variables can be modified without holding callback_lock. If only
// reliable read access of the externally used fields are needed, a task can
// hold either cpuset_mutex or callback_lock which are exposed to other
// external subsystems.
//
// If a task holds cpu_hotplug_lock and cpuset_mutex, it blocks others,
// ensuring that it is the only task able to also acquire callback_lock and
// be able to modify cpusets.  It can perform various checks on the cpuset
// structure first, knowing nothing will change. It can also allocate memory
// without holding callback_lock. While it is performing these checks, various
// callback routines can briefly acquire callback_lock to query cpusets.  Once
// it is ready to make the changes, it takes callback_lock, blocking everyone
// else.
//
// Calls to the kernel memory allocator cannot be made while holding
// callback_lock which is a spinlock, as the memory allocator may sleep or
// call back into cpuset code and acquire callback_lock.
//
// Now, the task_struct fields mems_allowed and mempolicy may be changed
// by other task, we use alloc_lock in the task_struct fields to protect
// them.
//
// The cpuset_common_seq_show() handlers only hold callback_lock across
// small pieces of code, such as when reading out possibly multi-word
// cpumasks and nodemasks.
static mut cpuset_top_mutex: mutex = {
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

static mut cpuset_mutex: mutex = {
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

// File level internal variables below follow one of the following exclusion
// rules.
//
// RWCS: Read/write-able by holding either cpus_write_lock (and optionally
// 	 cpuset_mutex) or both cpus_read_lock and cpuset_mutex.
//
// CSCB: Readable by holding either cpuset_mutex or callback_lock. Writable
// 	 by holding both cpuset_mutex and callback_lock.
//
// T:	 Read/write-able by holding the cpuset_top_mutex.
/// For local partitions, update to subpartitions_cpus & isolated_cpus is done
/// in update_parent_effective_cpumask(). For remote partitions, it is done in
/// the remote_partition_*() and remote_cpus_update() helpers.
///
/// Exclusive CPUs distributed out to local or remote sub-partitions of
/// top_cpuset
/// RWCS
static mut subpartitions_cpus: [cpumask; 1] = unsafe { core::mem::zeroed() };

/// Exclusive CPUs in isolated partitions (shown in cpuset.cpus.isolated)
/// CSCB
static mut isolated_cpus: [cpumask; 1] = unsafe { core::mem::zeroed() };

/// Set if housekeeping cpumasks are to be updated.
/// RWCS
static mut update_housekeeping: bool = unsafe { core::mem::zeroed() };

/// Copy of isolated_cpus to be passed to housekeeping_update()
/// T
static mut isolated_hk_cpus: [cpumask; 1] = unsafe { core::mem::zeroed() };

/// A flag to force sched domain rebuild at the end of an operation.
/// It can be set in
///  - update_partition_sd_lb()
///  - update_cpumasks_hier()
///  - cpuset_update_flag()
///  - cpuset_hotplug_update_tasks()
///  - cpuset_handle_hotplug()
///
/// Protected by cpuset_mutex (with cpus_read_lock held) or cpus_write_lock.
///
/// Note that update_relax_domain_level() in cpuset-v1.c can still call
/// rebuild_sched_domains_locked() directly without using this flag.
/// RWCS
static mut force_sd_rebuild: bool = unsafe { core::mem::zeroed() };

// Partition root states:
//
//   0 - member (not a partition root)
//   1 - partition root
//   2 - partition root without load balancing (isolated)
//  -1 - invalid partition root
//  -2 - invalid isolated partition root
//
//  There are 2 types of partitions - local or remote. Local partitions are
//  those whose parents are partition root themselves. Setting of
//  cpuset.cpus.exclusive are optional in setting up local partitions.
//  Remote partitions are those whose parents are not partition roots. Passing
//  down exclusive CPUs by setting cpuset.cpus.exclusive along its ancestor
//  nodes are mandatory in creating a remote partition.
//
//  For simplicity, a local partition can be created under a local or remote
//  partition but a remote partition cannot have any partition root in its
//  ancestor chain except the cgroup root.
//
//  A valid partition can be formed by setting exclusive_cpus or cpus_allowed
//  if exclusive_cpus is not set. In the case of partition with empty
//  exclusive_cpus, all the conflicting exclusive CPUs specified in the
//  following cpumasks of sibling cpusets will be removed from its
//  cpus_allowed in determining its effective_xcpus.
//  - effective_xcpus
//  - exclusive_cpus
//
//  The "cpuset.cpus.exclusive" control file should be used for setting up
//  partition if the users want to get as many CPUs as possible.
/// Temporary cpumasks for working with partitions that are passed among
/// functions to avoid memory allocation in inner functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct tmpmasks {
    addmask: [cpumask; 1],
    /// For partition root
    delmask: [cpumask; 1],
    /// For update_cpumasks_hier()
    new_cpus: [cpumask; 1],
}


#[unsafe(no_mangle)]
extern "C" fn inc_dl_tasks_cs(p: *mut task_struct) {
let cs: *mut cpuset = task_cs(p);
atomic_inc(unsafe { core::ptr::addr_of_mut!((*cs).nr_deadline_tasks) });
}

#[unsafe(no_mangle)]
extern "C" fn dec_dl_tasks_cs(p: *mut task_struct) {
let cs: *mut cpuset = task_cs(p);
atomic_dec(unsafe { core::ptr::addr_of_mut!((*cs).nr_deadline_tasks) });
}

fn is_partition_valid(cs: *const cpuset) -> bool {
(unsafe { (*cs).partition_root_state }) > 0
}

fn is_partition_invalid(cs: *const cpuset) -> bool {
(unsafe { (*cs).partition_root_state }) < 0
}

fn cs_is_member(cs: *const cpuset) -> bool {
(unsafe { (*cs).partition_root_state }) == 0
}

/// Callers should hold callback_lock to modify partition_root_state.
fn make_partition_invalid(cs: *mut cpuset) {
if (unsafe { (*cs).partition_root_state }) > 0 {
unsafe {
(*cs).partition_root_state = unsafe { (*cs).partition_root_state }.wrapping_neg();
}
}
}

// Send notification event of whenever partition_root_state changes.
fn notify_partition_change(cs: *mut cpuset, old_prs: i32) {
if old_prs == unsafe { (*cs).partition_root_state } {
return;
}
unsafe { cgroup_file_notify(core::ptr::addr_of_mut!((*cs).partition_file)) };
// Reset prs_err if not invalid
if is_partition_valid(cs as *const cpuset) {
loop {
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_679() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__14!((*cs).prs_err, 0i32);
if true {
break;
}
}
}
}

/// The top_cpuset is always synchronized to cpu_active_mask and we should avoid
/// using cpu_online_mask as much as possible. An active CPU is always an online
/// CPU, but not vice versa. cpu_active_mask and cpu_online_mask can differ
/// during hotplug operations. A CPU is marked active at the last stage of CPU
/// bringup (CPUHP_AP_ACTIVE). It is also the stage where cpuset hotplug code
/// will be called to update the sched domains so that the scheduler can move
/// a normal task to a newly active CPU or remove tasks away from a newly
/// inactivated CPU. The online bit is set much earlier in the CPU bringup
/// process and cleared much later in CPU teardown.
///
/// If cpu_online_mask is used while a hotunplug operation is happening in
/// parallel, we may leave an offline CPU in cpu_allowed or some other masks.
#[unsafe(no_mangle)]
static mut top_cpuset: cpuset = {
let mut __t0: cpuset = unsafe { core::mem::zeroed() };
__t0.flags = 19u64;
__t0.partition_root_state = 1i32;
__t0.dl_bw_cpu = 1i32.wrapping_neg();
__t0
};

/// cpuset_lock - Acquire the global cpuset mutex
///
/// This locks the global cpuset mutex to prevent modifications to cpuset
/// hierarchy and configurations. This helper is not enough to make modification.
#[unsafe(no_mangle)]
extern "C" fn cpuset_lock() {
unsafe { mutex_lock(core::ptr::addr_of_mut!(cpuset_mutex)) };
}

#[unsafe(no_mangle)]
extern "C" fn cpuset_unlock() {
unsafe { mutex_unlock(core::ptr::addr_of_mut!(cpuset_mutex)) };
}

#[unsafe(no_mangle)]
extern "C" fn lockdep_assert_cpuset_lock_held() {
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(cpuset_mutex) });
}

/// cpuset_full_lock - Acquire full protection for cpuset modification
///
/// Takes both CPU hotplug read lock (cpus_read_lock()) and cpuset mutex
/// to safely modify cpuset data.
#[unsafe(no_mangle)]
extern "C" fn cpuset_full_lock() {
unsafe { mutex_lock(core::ptr::addr_of_mut!(cpuset_top_mutex)) };
unsafe { cpus_read_lock() };
unsafe { mutex_lock(core::ptr::addr_of_mut!(cpuset_mutex)) };
}

#[unsafe(no_mangle)]
extern "C" fn cpuset_full_unlock() {
unsafe { mutex_unlock(core::ptr::addr_of_mut!(cpuset_mutex)) };
unsafe { cpus_read_unlock() };
unsafe { mutex_unlock(core::ptr::addr_of_mut!(cpuset_top_mutex)) };
}

static mut callback_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

#[unsafe(no_mangle)]
extern "C" fn cpuset_callback_lock_irq() {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
}

#[unsafe(no_mangle)]
extern "C" fn cpuset_callback_unlock_irq() {
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
}

static mut cpuset_migrate_mm_wq: *mut workqueue_struct = unsafe { core::mem::zeroed() };

static mut cpuset_attach_wq: wait_queue_head = {
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
__t0.head = list_head { next: unsafe { core::ptr::addr_of_mut!(cpuset_attach_wq.head) }, prev: unsafe { core::ptr::addr_of_mut!(cpuset_attach_wq.head) } };
__t0
};

/// Cpuset task attach context
/// Protected by cpuset_mutex
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cpuset_c_372 {
    in_progress: i32,
    cpus_updated: bool,
    mems_updated: bool,
    task_work_queued: bool,
    /// Have many destination cpusets
    many_dest_cs: bool,
    /// Source cpuset
    old_cs: *mut cpuset,
    nodemask_to: __SlateRecord_nodemask_types_h_17,
}


static mut attach_ctx: __SlateRecord_cpuset_c_372 = unsafe { core::mem::zeroed() };

static mut src_cs_head: llist_head = llist_head { first: core::ptr::null_mut::<llist_node>() };

static mut dst_cs_head: llist_head = llist_head { first: core::ptr::null_mut::<llist_node>() };

/// Wait if task attach is in progress until it is done and then acquire
/// cpuset_mutex before returning.
fn wait_attach_done_lock() {
let mut __wq_entry: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut wait_queue_entry = core::ptr::addr_of_mut!(__wq_entry) as *mut wait_queue_entry;
unsafe {
'__loop_7: loop {
mutex_lock(core::ptr::addr_of_mut!(cpuset_mutex));
if attach_ctx.in_progress == 0 {
break;
} else {
mutex_unlock(core::ptr::addr_of_mut!(cpuset_mutex));
// Wait until attach operation is done to prevent racing
loop {
might_sleep!();
if attach_ctx.in_progress == 0 {
continue '__loop_7;
} else {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
init_wait_entry(core::ptr::addr_of_mut!(*__slate_slot_2), if false { 1i32 } else { 0i32 });
loop {
prepare_to_wait_event(core::ptr::addr_of_mut!(cpuset_attach_wq), core::ptr::addr_of_mut!(*__slate_slot_2), 2i32);
if attach_ctx.in_progress == 0 {
break;
} else {
schedule();
if attach_ctx.in_progress == 0 {
break;
}
}
}
finish_wait(core::ptr::addr_of_mut!(cpuset_attach_wq), core::ptr::addr_of_mut!(*__slate_slot_2));
if true {
continue '__loop_7;
}
}
}
}
}
return;
}
}

fn check_insane_mems_config(nodes: *mut __SlateRecord_nodemask_types_h_17) {
let __v0: bool = !cpusets_insane_config() && movable_only_nodes(nodes);
if __v0 {
static_branch_enable_cpuslocked!(unsafe { core::ptr::addr_of_mut!(cpusets_insane_config_key) });
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x016Unsupported (movable nodes only) cpuset configuration detected (nmask=%*pbl)!\nCpuset allocations might fail even with a lot of memory available.\n".as_ptr() as *const u8, __nodemask_pr_numnodes(nodes as *const __SlateRecord_nodemask_types_h_17), __nodemask_pr_bits(nodes as *const __SlateRecord_nodemask_types_h_17)) };
}
}

/// decrease attach_ctx.in_progress.
/// wake_up cpuset_attach_wq if attach_ctx.in_progress==0.
fn dec_attach_in_progress_locked() {
lockdep_assert_cpuset_lock_held();
let __v0: i32 = unsafe { attach_ctx.in_progress };
unsafe {
attach_ctx.in_progress = __v0.wrapping_sub(1);
}
if (unsafe { attach_ctx.in_progress }) == 0 {
wake_up!(unsafe { core::ptr::addr_of_mut!(cpuset_attach_wq) });
}
}

fn dec_attach_in_progress() {
unsafe { mutex_lock(core::ptr::addr_of_mut!(cpuset_mutex)) };
dec_attach_in_progress_locked();
unsafe { mutex_unlock(core::ptr::addr_of_mut!(cpuset_mutex)) };
}

fn cpuset_v2() -> bool {
IS_ENABLED!() == 0 || static_branch_likely__3!(unsafe { core::ptr::addr_of_mut!(cpuset_cgrp_subsys_on_dfl_key) }) != 0
}

/// Cgroup v2 behavior is used on the "cpus" and "mems" control files when
/// on default hierarchy or when the cpuset_v2_mode flag is set by mounting
/// the v1 cpuset cgroup filesystem with the "cpuset_v2_mode" mount option.
/// With v2 behavior, "cpus" and "mems" are always what the users have
/// requested and won't be changed by hotplug events. Only the effective
/// cpus or mems will be affected.
fn is_in_v2_mode() -> bool {
cpuset_v2() || (unsafe { (*cpuset_cgrp_subsys.root).flags }) & 65536 != 0
}

/// partition_is_populated - check if partition has tasks
/// @cs: partition root to be checked
/// @excluded_child: a child cpuset to be excluded in task checking
/// Return: true if there are tasks, false otherwise
///
/// @cs should be a valid partition root or going to become a partition root.
/// @excluded_child should be non-NULL when this cpuset is going to become a
/// partition itself.
///
/// Note that a remote partition is not allowed underneath a valid local
/// or remote partition. So if a non-partition root child is populated,
/// the whole partition is considered populated.
fn partition_is_populated(cs: *mut cpuset, excluded_child: *mut cpuset) -> bool {
let mut pos_css: *mut cgroup_subsys_state = unsafe { core::mem::zeroed() };
// We cannot call cs_is_populated(cs) directly, as
// nr_populated_domain_children may include populated
// csets from descendants that are partitions.
if cgroup_has_tasks(unsafe { (*cs).css.cgroup }) {
return true;
}
rcu_read_lock();
pos_css = unsafe { css_next_descendant_pre(core::ptr::null_mut::<cgroup_subsys_state>(), core::ptr::addr_of_mut!((*cs).css)) };
while !pos_css.is_null() {
let __v0: *mut cpuset = css_cs(pos_css);
let cp: *mut cpuset = __v0;
if is_cpuset_online(__v0) {
if !(cp == cs || cp == excluded_child) {
if is_partition_valid(cp as *const cpuset) {
pos_css = unsafe { css_rightmost_descendant(pos_css) };
} else {
if cpuset_is_populated(cp) {
rcu_read_unlock();
return true;
}
}
}
}
pos_css = unsafe { css_next_descendant_pre(pos_css, core::ptr::addr_of_mut!((*cs).css)) };
}
rcu_read_unlock();
false
}

/// Return in pmask the portion of a task's cpusets's cpus_allowed that
/// are online and are capable of running the task.  If none are found,
/// walk up the cpuset hierarchy until we find one that does have some
/// appropriate cpus.
///
/// One way or another, we guarantee to return some non-empty subset
/// of cpu_active_mask.
///
/// Call with callback_lock or cpuset_mutex held.
fn guarantee_active_cpus(tsk: *mut task_struct, pmask: *mut cpumask) {
let possible_mask: *const cpumask = task_cpu_possible_mask!();
let mut cs: *mut cpuset = unsafe { core::mem::zeroed() };
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!cpumask_and(pmask, possible_mask, (unsafe { core::ptr::addr_of_mut!(__cpu_active_mask) }) as *const cpumask))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("680: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 680b - ., 3; .popsection\n# {0}", const 680 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_guarantee_active_cpus_1, sym __slate_asm_string_guarantee_active_cpus_2, const 510 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("681: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 681b - ., 4; .popsection\n# {0}", const 681 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
cpumask_copy(pmask, (unsafe { core::ptr::addr_of_mut!(__cpu_active_mask) }) as *const cpumask);
}
rcu_read_lock();
cs = task_cs(tsk);
while !cpumask_intersects((unsafe { (*cs).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, pmask as *const cpumask) {
cs = parent_cs(cs);
}
cpumask_and(pmask, pmask as *const cpumask, (unsafe { (*cs).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
rcu_read_unlock();
}

/// Return in *pmask the portion of a cpusets's mems_allowed that
/// are online, with memory.  If none are online with memory, walk
/// up the cpuset hierarchy until we find one that does have some
/// online mems.  The top cpuset always has some mems online. With v2,
/// effective_mems should always contain online memory nodes except
/// during the transition period where a memory node hotunplug operation
/// is in progress.
///
/// One way or another, we guarantee to return some non-empty subset
/// of node_states[N_MEMORY].
///
/// Call with callback_lock or cpuset_mutex held.
fn guarantee_online_mems(mut cs: *mut cpuset, pmask: *mut __SlateRecord_nodemask_types_h_17) {
while !nodes_and!(*pmask, (*cs).effective_mems, node_states[3]) {
cs = parent_cs(cs);
}
}

/// alloc_cpumasks - Allocate an array of cpumask variables
/// @pmasks: Pointer to array of cpumask_var_t pointers
/// @size: Number of cpumasks to allocate
/// Return: 0 if successful, -ENOMEM otherwise.
///
/// Allocates @size cpumasks and initializes them to empty. Returns 0 on
/// success, -ENOMEM on allocation failure. On failure, any previously
/// allocated cpumasks are freed.
fn alloc_cpumasks(pmasks: *mut *mut [cpumask; 1], size: u32) -> i32 {
let mut i: i32 = 0;
while (i as u32) < size {
if !zalloc_cpumask_var(unsafe { *pmasks.wrapping_offset(i as isize) }, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) {
loop {
let __v1: i32 = i.wrapping_sub(1);
i = __v1;
if __v1 < 0 {
break;
}
free_cpumask_var(unsafe { (*(*pmasks.wrapping_offset(i as isize))).as_mut_ptr() as *mut cpumask });
}
return 12i32.wrapping_neg();
}
i = i.wrapping_add(1);
}
0
}

/// alloc_tmpmasks - Allocate temporary cpumasks for cpuset operations.
/// @tmp: Pointer to tmpmasks structure to populate
/// Return: 0 on success, -ENOMEM on allocation failure
fn alloc_tmpmasks(tmp: &mut tmpmasks) -> i32 {
// Array of pointers to the three cpumask_var_t fields in tmpmasks.
// Note: Array size must match actual number of masks (3)
let mut pmask: __SlateAlign16<[*mut [cpumask; 1]; 3]> = __SlateAlign16([unsafe { core::ptr::addr_of_mut!((*(tmp as *mut tmpmasks)).new_cpus) }, unsafe { core::ptr::addr_of_mut!((*(tmp as *mut tmpmasks)).addmask) }, unsafe { core::ptr::addr_of_mut!((*(tmp as *mut tmpmasks)).delmask) }]);
alloc_cpumasks(pmask.0.as_mut_ptr() as *mut *mut [cpumask; 1], ARRAY_SIZE__16!() as u32)
}

/// free_tmpmasks - free cpumasks in a tmpmasks structure
/// @tmp: the tmpmasks structure pointer
fn free_tmpmasks(tmp: *mut tmpmasks) {
if tmp.is_null() {
return;
}
free_cpumask_var(unsafe { (*tmp).new_cpus.as_mut_ptr() as *mut cpumask });
free_cpumask_var(unsafe { (*tmp).addmask.as_mut_ptr() as *mut cpumask });
free_cpumask_var(unsafe { (*tmp).delmask.as_mut_ptr() as *mut cpumask });
}

/// dup_or_alloc_cpuset - Duplicate or allocate a new cpuset
/// @cs: Source cpuset to duplicate (NULL for a fresh allocation)
///
/// Creates a new cpuset by either:
/// 1. Duplicating an existing cpuset (if @cs is non-NULL), or
/// 2. Allocating a fresh cpuset with zero-initialized masks (if @cs is NULL)
///
/// Return: Pointer to newly allocated cpuset on success, NULL on failure
fn dup_or_alloc_cpuset(cs: *mut cpuset) -> *mut cpuset {
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut trial: *mut cpuset = unsafe { core::mem::zeroed() };
// Allocate base structure
let __v0: *mut () = if !cs.is_null() {
alloc_hooks!(unsafe { kmemdup_noprof(cs as *const (), core::mem::size_of::<cpuset>() as u64, 3264u32) })
} else {
let mut __v6: *mut cpuset;
let __obj_size: u64 = size_mul(core::mem::size_of::<cpuset>() as u64, 1u64);
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
__v6 = __v7 as *mut cpuset;
__v6 as *mut ()
};
trial = __v0 as *mut cpuset;
if trial.is_null() {
return core::ptr::null_mut::<cpuset>();
}
unsafe {
(*trial).dl_bw_cpu = 1i32.wrapping_neg();
}
init_llist_node(unsafe { core::ptr::addr_of_mut!((*trial).attach_node) });
// Setup cpumask pointer array
let mut pmask: __SlateAlign16<[*mut [cpumask; 1]; 4]> = __SlateAlign16([unsafe { core::ptr::addr_of_mut!((*trial).cpus_allowed) }, unsafe { core::ptr::addr_of_mut!((*trial).effective_cpus) }, unsafe { core::ptr::addr_of_mut!((*trial).effective_xcpus) }, unsafe { core::ptr::addr_of_mut!((*trial).exclusive_cpus) }]);
if alloc_cpumasks(pmask.0.as_mut_ptr() as *mut *mut [cpumask; 1], ARRAY_SIZE__41!() as u32) != 0 {
unsafe { kfree(trial as *const ()) };
return core::ptr::null_mut::<cpuset>();
}
// Copy masks if duplicating
if !cs.is_null() {
cpumask_copy(unsafe { (*trial).cpus_allowed.as_mut_ptr() as *mut cpumask }, (unsafe { (*cs).cpus_allowed.as_mut_ptr() as *mut cpumask }) as *const cpumask);
cpumask_copy(unsafe { (*trial).effective_cpus.as_mut_ptr() as *mut cpumask }, (unsafe { (*cs).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
cpumask_copy(unsafe { (*trial).effective_xcpus.as_mut_ptr() as *mut cpumask }, (unsafe { (*cs).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
cpumask_copy(unsafe { (*trial).exclusive_cpus.as_mut_ptr() as *mut cpumask }, (unsafe { (*cs).exclusive_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
}
trial
}

/// free_cpuset - free the cpuset
/// @cs: the cpuset to be freed
fn free_cpuset(cs: *mut cpuset) {
free_cpumask_var(unsafe { (*cs).cpus_allowed.as_mut_ptr() as *mut cpumask });
free_cpumask_var(unsafe { (*cs).effective_cpus.as_mut_ptr() as *mut cpumask });
free_cpumask_var(unsafe { (*cs).effective_xcpus.as_mut_ptr() as *mut cpumask });
free_cpumask_var(unsafe { (*cs).exclusive_cpus.as_mut_ptr() as *mut cpumask });
unsafe { kfree(cs as *const ()) };
}

/// Return user specified exclusive CPUs
fn user_xcpus(cs: *mut cpuset) -> *mut cpumask {
if cpumask_empty((unsafe { (*cs).exclusive_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask) { unsafe { (*cs).cpus_allowed.as_mut_ptr() as *mut cpumask } } else { unsafe { (*cs).exclusive_cpus.as_mut_ptr() as *mut cpumask } }
}

fn xcpus_empty(cs: *mut cpuset) -> bool {
cpumask_empty((unsafe { (*cs).cpus_allowed.as_mut_ptr() as *mut cpumask }) as *const cpumask) && cpumask_empty((unsafe { (*cs).exclusive_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask)
}

/// cpusets_are_exclusive() - check if two cpusets are exclusive
///
/// Return true if exclusive, false if not
fn cpusets_are_exclusive(cs1: *mut cpuset, cs2: *mut cpuset) -> bool {
let xcpus1: *mut cpumask = user_xcpus(cs1);
let xcpus2: *mut cpumask = user_xcpus(cs2);
if cpumask_intersects(xcpus1 as *const cpumask, xcpus2 as *const cpumask) {
return false;
}
true
}

/// cpus_excl_conflict - Check if two cpusets have exclusive CPU conflicts
/// @trial:	the trial cpuset to be checked
/// @sibling:	a sibling cpuset to be checked against
/// @xcpus_changed: set if exclusive_cpus has been set
///
/// Returns: true if CPU exclusivity conflict exists, false otherwise
///
/// Conflict detection rules:
///  o cgroup v1
///    See cpuset1_cpus_excl_conflict()
///  o cgroup v2
///    - The exclusive_cpus values cannot overlap.
///    - New exclusive_cpus cannot be a superset of a sibling's cpus_allowed.
fn cpus_excl_conflict(trial: *mut cpuset, sibling: *mut cpuset, xcpus_changed: bool) -> bool {
if !cpuset_v2() {
return cpuset1_cpus_excl_conflict(trial, sibling);
}
// The cpus_allowed of a sibling cpuset cannot be a subset of the new exclusive_cpus
let __v0: bool = xcpus_changed && !cpumask_empty((unsafe { (*sibling).cpus_allowed.as_mut_ptr() as *mut cpumask }) as *const cpumask);
let __v1: bool = __v0 && cpumask_subset((unsafe { (*sibling).cpus_allowed.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { (*trial).exclusive_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
if __v1 {
return true;
}
// Exclusive_cpus cannot intersect
cpumask_intersects((unsafe { (*trial).exclusive_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { (*sibling).exclusive_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask)
}

fn mems_excl_conflict(cs1: *mut cpuset, cs2: *mut cpuset) -> bool {
let __v0: bool = is_mem_exclusive(cs1 as *const cpuset) != 0 || is_mem_exclusive(cs2 as *const cpuset) != 0;
if __v0 {
return nodes_intersects__2!((*cs1).mems_allowed, (*cs2).mems_allowed);
}
false
}

// validate_change() - Used to validate that any proposed cpuset change
// 		       follows the structural rules for cpusets.
//
// If we replaced the flag and mask values of the current cpuset
// (cur) with those values in the trial cpuset (trial), would
// our various subset and exclusive rules still be valid?  Presumes
// cpuset_mutex held.
//
// 'cur' is the address of an actual, in-use cpuset.  Operations
// such as list traversal that depend on the actual address of the
// cpuset in the list must use cur below, not trial.
//
// 'trial' is the address of bulk structure copy of cur, with
// perhaps one or more of the fields cpus_allowed, mems_allowed,
// or flags changed to new, trial values.
//
// Return 0 if valid, -errno if not.
fn validate_change(cur: *mut cpuset, trial: *mut cpuset) -> i32 {
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut xcpus_changed: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut bool = core::ptr::addr_of_mut!(xcpus_changed) as *mut bool;
let mut par: core::mem::MaybeUninit<*mut cpuset> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut cpuset = core::ptr::addr_of_mut!(par) as *mut *mut cpuset;
let mut c: core::mem::MaybeUninit<*mut cpuset> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut cpuset = core::ptr::addr_of_mut!(c) as *mut *mut cpuset;
let mut css: core::mem::MaybeUninit<*mut cgroup_subsys_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut cgroup_subsys_state = core::ptr::addr_of_mut!(css) as *mut *mut cgroup_subsys_state;
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
ret = 0;
rcu_read_lock();
if !is_in_v2_mode() {
ret = cpuset1_validate_change(cur, trial);
}
'__join_0: {
if ret == 0 {
// Remaining checks don't apply to root cpuset
if cur != core::ptr::addr_of_mut!(top_cpuset) {
*__slate_slot_5 = parent_cs(cur);
// We can't shrink if we won't have enough room for SCHED_DEADLINE
// tasks. This check is not done when scheduling is disabled as the
// users should know what they are doing.
//
// For v1, effective_cpus == cpus_allowed & user_xcpus() returns
// cpus_allowed.
//
// For v2, is_cpu_exclusive() & is_sched_load_balance() are true only
// for non-isolated partition root. At this point, the target
// effective_cpus isn't computed yet. user_xcpus() is the best
// approximation.
//
// TBD: May need to precompute the real effective_cpus here in case
// incorrect scheduling of SCHED_DEADLINE tasks in a partition
// becomes an issue.
ret = 16i32.wrapping_neg();
if is_cpu_exclusive(cur as *const cpuset) != 0 {
__v2 = is_sched_load_balance(cur as *const cpuset) != 0;
} else {
__v2 = false;
}
if __v2 {
__v1 = cpuset_cpumask_can_shrink(((*cur).effective_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask, user_xcpus(trial) as *const cpumask) == 0;
} else {
__v1 = false;
}
if !__v1 {
// If either I or some sibling (!= me) is exclusive, we can't
// overlap. exclusive_cpus cannot overlap with each other if set.
ret = 22i32.wrapping_neg();
*__slate_slot_4 = !cpumask_equal(((*cur).exclusive_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask, ((*trial).exclusive_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
*__slate_slot_7 = css_next_child(core::ptr::null_mut::<cgroup_subsys_state>(), core::ptr::addr_of_mut!((*(*__slate_slot_5)).css));
while !(*__slate_slot_7).is_null() {
let __v0: *mut cpuset = css_cs(*__slate_slot_7);
*__slate_slot_6 = __v0;
if is_cpuset_online(__v0) {
if *__slate_slot_6 != cur {
if cpus_excl_conflict(trial, *__slate_slot_6, *__slate_slot_4) {
break '__join_0;
} else {
if mems_excl_conflict(trial, *__slate_slot_6) {
break '__join_0;
}
}
}
}
*__slate_slot_7 = css_next_child(*__slate_slot_7, core::ptr::addr_of_mut!((*(*__slate_slot_5)).css));
}
ret = 0;
}
}
}
}
rcu_read_unlock();
return ret;
}
}

/// generate_sched_domains()
///
/// This function builds a partial partition of the systems CPUs
/// A 'partial partition' is a set of non-overlapping subsets whose
/// union is a subset of that set.
/// The output of this function needs to be passed to kernel/sched/core.c
/// partition_sched_domains() routine, which will rebuild the scheduler's
/// load balancing domains (sched domains) as specified by that partial
/// partition.
///
/// See "What is sched_load_balance" in Documentation/admin-guide/cgroup-v1/cpusets.rst
/// for a background explanation of this.
///
/// Does not return errors, on the theory that the callers of this
/// routine would rather not worry about failures to rebuild sched
/// domains when operating in the severe memory shortage situations
/// that could cause allocation failures below.
///
/// Must be called with cpuset_mutex held.
///
/// The three key local variables below are:
///    cp - cpuset pointer, used (together with pos_css) to perform a
/// 	   top-down scan of all cpusets. For our purposes, rebuilding
/// 	   the schedulers sched domains, we can ignore !is_sched_load_
/// 	   balance cpusets.
///  csa  - (for CpuSet Array) Array of pointers to all the cpusets
/// 	   that need to be load balanced, for convenient iterative
/// 	   access by the subsequent code that finds the best partition,
/// 	   i.e the set of domains (subsets) of CPUs such that the
/// 	   cpus_allowed of every cpuset marked is_sched_load_balance
/// 	   is a subset of one of these domains, while there are as
/// 	   many such domains as possible, each as small as possible.
/// doms  - Conversion of 'csa' to an array of cpumasks, for passing to
/// 	   the kernel/sched/core.c routine partition_sched_domains() in a
/// 	   convenient format, that can be easily compared to the prior
/// 	   value to determine what partition elements (sched domains)
/// 	   were changed (added or removed.)
fn generate_sched_domains(domains: &mut *mut [cpumask; 1], attributes: &mut *mut sched_domain_attr) -> i32 {
let mut __slate_compound_34: sched_domain_attr = unsafe { core::mem::zeroed() };
let mut __slate_compound_35: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_37: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_40: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_42: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
// Skip @cp's subtree if not a partition root and has no
// exclusive CPUs to be granted to child cpusets.
let mut __v15: bool = unsafe { core::mem::zeroed() };
// Only valid partition roots that are not isolated and with
// non-empty effective_cpus will be saved into csa[].
let mut __v18: bool = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size_1: u64 = unsafe { core::mem::zeroed() };
let mut pos_css: core::mem::MaybeUninit<*mut cgroup_subsys_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut *mut cgroup_subsys_state = core::ptr::addr_of_mut!(pos_css) as *mut *mut cgroup_subsys_state; // number of sched domains in result
let mut ndoms: i32 = unsafe { core::mem::zeroed() };
// attributes for custom domains
let mut dattr: core::mem::MaybeUninit<*mut sched_domain_attr> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut *mut sched_domain_attr = core::ptr::addr_of_mut!(dattr) as *mut *mut sched_domain_attr; // resulting partition; i.e. sched domains
let mut doms: core::mem::MaybeUninit<*mut [cpumask; 1]> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut *mut [cpumask; 1] = core::ptr::addr_of_mut!(doms) as *mut *mut [cpumask; 1]; // indices for partition finding loops
let mut j: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut i32 = core::ptr::addr_of_mut!(j) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32; // array of all cpuset ptrs
let mut csa: core::mem::MaybeUninit<*mut *mut cpuset> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut *mut *mut cpuset = core::ptr::addr_of_mut!(csa) as *mut *mut *mut cpuset; // top-down scan of cpusets
let mut cp: core::mem::MaybeUninit<*mut cpuset> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut *mut cpuset = core::ptr::addr_of_mut!(cp) as *mut *mut cpuset;
unsafe {
let __slate_auto_bytes_3 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
ndoms = 0;
let __slate_auto_bytes_9 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
if !cpuset_v2() {
return cpuset1_generate_sched_domains(domains, attributes);
} else {
'__join_2: {
*__slate_slot_29 = core::ptr::null_mut::<[cpumask; 1]>();
*__slate_slot_28 = core::ptr::null_mut::<sched_domain_attr>();
*__slate_slot_32 = core::ptr::null_mut::<*mut cpuset>();
// Special case for the 99% of systems with one, full, sched domain
if cpumask_empty((core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask) as *const cpumask) {
ndoms = 1;
// !csa will be checked and can be correctly handled
} else {
__obj_size_1 = size_mul(core::mem::size_of::<*mut cpuset>() as u64, nr_cpusets() as u64);
let __slate_auto_bytes_2 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_20 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_21 = _kmalloc_noprof(__obj_size_1, 3264u32, *{
__slate_compound_40 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_40)
});
alloc_tag_restore!();
} else {
*__slate_slot_21 = _kmalloc_noprof(__obj_size_1, 3264u32, *{
__slate_compound_42 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_42)
});
}
*__slate_slot_32 = *__slate_slot_21 as *mut *mut cpuset;
if (*__slate_slot_32).is_null() {
break '__join_2;
} else {
// Find how many partitions and cache them to csa[]
rcu_read_lock();
*__slate_slot_26 = css_next_descendant_pre(core::ptr::null_mut::<cgroup_subsys_state>(), core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(top_cpuset)).css));
while !(*__slate_slot_26).is_null() {
let __v19: *mut cpuset = css_cs(*__slate_slot_26);
*__slate_slot_33 = __v19;
if is_cpuset_online(__v19) {
if (*(*__slate_slot_33)).partition_root_state == 1 {
__v18 = !cpumask_empty(((*(*__slate_slot_33)).effective_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
} else {
__v18 = false;
}
if __v18 {
let __v17: i32 = ndoms;
ndoms = __v17.wrapping_add(1);
*(*__slate_slot_32).wrapping_offset(__v17 as isize) = *__slate_slot_33;
}
if !is_partition_valid(*__slate_slot_33 as *const cpuset) {
__v15 = cpumask_empty(((*(*__slate_slot_33)).exclusive_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
} else {
__v15 = false;
}
if __v15 {
*__slate_slot_26 = css_rightmost_descendant(*__slate_slot_26);
}
}
*__slate_slot_26 = css_next_descendant_pre(*__slate_slot_26, core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(top_cpuset)).css));
}
rcu_read_unlock();
*__slate_slot_31 = 0;
while *__slate_slot_31 < ndoms {
*__slate_slot_30 = (*__slate_slot_31).wrapping_add(1);
while *__slate_slot_30 < ndoms {
if cpusets_overlap(*(*__slate_slot_32).wrapping_offset(*__slate_slot_31 as isize), *(*__slate_slot_32).wrapping_offset(*__slate_slot_30 as isize)) != 0 {
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("682: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 682b - ., 3; .popsection\n# {0}", const 682 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_generate_sched_domains_38, sym __slate_asm_string_generate_sched_domains_39, const 906 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("683: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 683b - ., 4; .popsection\n# {0}", const 683 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
// Cgroup v2 shouldn't pass down overlapping
// partition root cpusets.
let __v11: i32 = *__slate_slot_30;
*__slate_slot_30 = __v11.wrapping_add(1);
}
let __v9: i32 = *__slate_slot_31;
*__slate_slot_31 = __v9.wrapping_add(1);
}
}
}
*__slate_slot_29 = alloc_sched_domains(ndoms as u32);
if !(*__slate_slot_29).is_null() {
__obj_size = size_mul(core::mem::size_of::<sched_domain_attr>() as u64, ndoms as u64);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_2 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_3 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_35 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_35)
});
alloc_tag_restore!();
} else {
*__slate_slot_3 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_37 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_37)
});
}
let __v5: *mut () = *__slate_slot_3;
// The rest of the code, including the scheduler, can deal with
// dattr==NULL case. No need to abort if alloc fails.
*__slate_slot_28 = __v5 as *mut sched_domain_attr;
// Cgroup v2 doesn't support domain attributes, just set all of them
// to SD_ATTR_INIT. Also non-isolating partition root CPUs are a
// subset of HK_TYPE_DOMAIN_BOOT housekeeping CPUs.
*__slate_slot_31 = 0;
while *__slate_slot_31 < ndoms {
// The top cpuset may contain some boot time isolated
// CPUs that need to be excluded from the sched domain.
if (*__slate_slot_32).is_null() || *(*__slate_slot_32).wrapping_offset(*__slate_slot_31 as isize) == core::ptr::addr_of_mut!(top_cpuset) {
cpumask_and((*(*__slate_slot_29).wrapping_offset(*__slate_slot_31 as isize)).as_mut_ptr() as *mut cpumask, (core::ptr::addr_of_mut!(top_cpuset.effective_cpus) as *mut cpumask) as *const cpumask, housekeeping_cpumask(0u32));
} else {
cpumask_copy((*(*__slate_slot_29).wrapping_offset(*__slate_slot_31 as isize)).as_mut_ptr() as *mut cpumask, ((*(*(*__slate_slot_32).wrapping_offset(*__slate_slot_31 as isize))).effective_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
}
if !(*__slate_slot_28).is_null() {
*(*__slate_slot_28).wrapping_offset(*__slate_slot_31 as isize) = *{
__slate_compound_34 = sched_domain_attr { relax_domain_level: 1i32.wrapping_neg() };
core::ptr::addr_of_mut!(__slate_compound_34)
};
}
let __v1: i32 = *__slate_slot_31;
*__slate_slot_31 = __v1.wrapping_add(1);
}
}
}
kfree(*__slate_slot_32 as *const ());
// Fallback to the default domain if kmalloc() failed.
// See comments in partition_sched_domains().
if (*__slate_slot_29).is_null() {
ndoms = 1;
}
*(domains as *mut *mut [cpumask; 1]) = *__slate_slot_29;
*(attributes as *mut *mut sched_domain_attr) = *__slate_slot_28;
return ndoms;
}
}
}

fn dl_update_tasks_root_domain(cs: *mut cpuset) {
let mut it: css_task_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(it) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 104usize);
}
if atomic_read((unsafe { core::ptr::addr_of_mut!((*cs).nr_deadline_tasks) }) as *const __SlateRecord_types_h_190) == 0 {
return;
}
unsafe { css_task_iter_start(core::ptr::addr_of_mut!((*cs).css), 0u32, core::ptr::addr_of_mut!(it)) };
loop {
let __v0: *mut task_struct = unsafe { css_task_iter_next(core::ptr::addr_of_mut!(it)) };
let task: *mut task_struct = __v0;
if __v0.is_null() {
break;
}
unsafe { dl_add_task_root_domain(task) };
}
unsafe { css_task_iter_end(core::ptr::addr_of_mut!(it)) };
}

#[unsafe(no_mangle)]
extern "C" fn dl_rebuild_rd_accounting() {
let mut cs: *mut cpuset = core::ptr::null_mut::<cpuset>();
let mut pos_css: *mut cgroup_subsys_state = unsafe { core::mem::zeroed() };
let mut cpu: i32 = 0;
let mut cookie: u64 = 0;
let __v0: u64 = unsafe { dl_cookie };
let __v1: u64 = __v0.wrapping_add(1);
unsafe {
dl_cookie = __v1;
}
cookie = __v1;
lockdep_assert_cpuset_lock_held();
unsafe { lockdep_assert_cpus_held() };
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(sched_domains_mutex) });
rcu_read_lock();
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
if !unsafe { dl_bw_visited(cpu, cookie) } {
unsafe { dl_clear_root_domain_cpu(cpu) };
}
cpu = cpu.wrapping_add(1);
}
pos_css = unsafe { css_next_descendant_pre(core::ptr::null_mut::<cgroup_subsys_state>(), core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(top_cpuset)).css)) };
while !pos_css.is_null() {
let __v6: *mut cpuset = css_cs(pos_css);
cs = __v6;
if is_cpuset_online(__v6) {
if cpumask_empty((unsafe { (*cs).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask) {
pos_css = unsafe { css_rightmost_descendant(pos_css) };
} else {
css_get(unsafe { core::ptr::addr_of_mut!((*cs).css) });
rcu_read_unlock();
dl_update_tasks_root_domain(cs);
rcu_read_lock();
css_put(unsafe { core::ptr::addr_of_mut!((*cs).css) });
}
}
pos_css = unsafe { css_next_descendant_pre(pos_css, core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(top_cpuset)).css)) };
}
rcu_read_unlock();
}

/// Rebuild scheduler domains.
///
/// If the flag 'sched_load_balance' of any cpuset with non-empty
/// 'cpus' changes, or if the 'cpus' allowed changes in any cpuset
/// which has that flag enabled, or if any cpuset with a non-empty
/// 'cpus' is removed, then call this routine to rebuild the
/// scheduler's dynamic sched domains.
///
/// Call with cpuset_mutex held.  Takes cpus_read_lock().
#[unsafe(no_mangle)]
extern "C" fn rebuild_sched_domains_locked() {
let mut attr: *mut sched_domain_attr = unsafe { core::mem::zeroed() };
let mut doms: *mut [cpumask; 1] = unsafe { core::mem::zeroed() };
let mut ndoms: i32 = 0;
// cpuset_hotplug_workfn is invoked synchronously now, thus this
// function should not race with CPU hotplug. And the effective CPUs
// must not include any offline CPUs. Passing an offline CPU in the
// doms to partition_sched_domains() will trigger a kernel panic.
//
// We perform a final check here: if the doms contains any
// offline CPUs, a warning is emitted and we return directly to
// prevent the panic.
let mut i: i32 = 0;
unsafe { lockdep_assert_cpus_held() };
lockdep_assert_cpuset_lock_held();
unsafe {
force_sd_rebuild = false;
}
// Generate domain masks and attrs
ndoms = generate_sched_domains(unsafe { &mut (*core::ptr::addr_of_mut!(doms)) }, unsafe { &mut (*core::ptr::addr_of_mut!(attr)) });
while !doms.is_null() && i < ndoms {
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!cpumask_subset((unsafe { (*doms.wrapping_offset(i as isize)).as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { core::ptr::addr_of_mut!(__cpu_active_mask) }) as *const cpumask))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("684: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 684b - ., 3; .popsection\n# {0}", const 684 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_rebuild_sched_domains_locked_1, sym __slate_asm_string_rebuild_sched_domains_locked_2, const 1046 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("685: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 685b - ., 4; .popsection\n# {0}", const 685 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
unsafe { free_sched_domains(doms, ndoms as u32) };
unsafe { kfree(attr as *const ()) };
return;
}
i = i.wrapping_add(1);
}
// Have scheduler rebuild the domains
unsafe { partition_sched_domains(ndoms, doms, attr) };
}

fn rebuild_sched_domains_cpuslocked() {
unsafe { mutex_lock(core::ptr::addr_of_mut!(cpuset_mutex)) };
rebuild_sched_domains_locked();
unsafe { mutex_unlock(core::ptr::addr_of_mut!(cpuset_mutex)) };
}

#[unsafe(no_mangle)]
extern "C" fn rebuild_sched_domains() {
unsafe { cpus_read_lock() };
rebuild_sched_domains_cpuslocked();
unsafe { cpus_read_unlock() };
}

#[unsafe(no_mangle)]
extern "C" fn cpuset_reset_sched_domains() {
unsafe { mutex_lock(core::ptr::addr_of_mut!(cpuset_mutex)) };
unsafe { partition_sched_domains(1i32, core::ptr::null_mut::<[cpumask; 1]>(), core::ptr::null_mut::<sched_domain_attr>()) };
unsafe { mutex_unlock(core::ptr::addr_of_mut!(cpuset_mutex)) };
}

/// cpuset_update_tasks_cpumask - Update the cpumasks of tasks in the cpuset.
/// @cs: the cpuset in which each task's cpus_allowed mask needs to be changed
/// @new_cpus: the temp variable for the new effective_cpus mask
///
/// Iterate through each task of @cs updating its cpus_allowed to the
/// effective cpuset's.  As this function is called with cpuset_mutex held,
/// cpuset membership stays stable.
///
/// For top_cpuset, task_cpu_possible_mask() is used instead of effective_cpus
/// to make sure all offline CPUs are also included as hotplug code won't
/// update cpumasks for tasks in top_cpuset.
///
/// As task_cpu_possible_mask() can be task dependent in arm64, we have to
/// do cpu masking per task instead of doing it once for all.
#[unsafe(no_mangle)]
extern "C" fn cpuset_update_tasks_cpumask(cs: *mut cpuset, new_cpus: *mut cpumask) {
let mut it: css_task_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(it) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 104usize);
}
let top_cs: bool = cs == unsafe { core::ptr::addr_of_mut!(top_cpuset) };
unsafe { css_task_iter_start(core::ptr::addr_of_mut!((*cs).css), 0u32, core::ptr::addr_of_mut!(it)) };
loop {
let __v1: *mut task_struct = unsafe { css_task_iter_next(core::ptr::addr_of_mut!(it)) };
let task: *mut task_struct = __v1;
if __v1.is_null() {
break;
}
'__slate_continue_0: {
let possible_mask: *const cpumask = task_cpu_possible_mask!();
if top_cs {
// PF_KTHREAD tasks are handled by housekeeping.
// PF_NO_SETAFFINITY tasks are ignored.
if (unsafe { (*task).flags }) & 69206016u32 != 0 {
break '__slate_continue_0;
}
cpumask_andnot(new_cpus, possible_mask, (unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask }) as *const cpumask);
} else {
cpumask_and(new_cpus, possible_mask, (unsafe { (*cs).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
}
unsafe { set_cpus_allowed_ptr(task, new_cpus as *const cpumask) };
}
}
unsafe { css_task_iter_end(core::ptr::addr_of_mut!(it)) };
}

/// compute_effective_cpumask - Compute the effective cpumask of the cpuset
/// @new_cpus: the temp variable for the new effective_cpus mask
/// @cs: the cpuset the need to recompute the new effective_cpus mask
/// @parent: the parent cpuset
///
/// For v2, the parent's effective_cpus is inherited if cpumask is empty.
/// The result is valid only if the given cpuset isn't a partition root.
fn compute_effective_cpumask(new_cpus: *mut cpumask, cs: *mut cpuset, parent: *mut cpuset) {
let mut has_cpus: bool = false;
has_cpus = cpumask_and(new_cpus, (unsafe { (*cs).cpus_allowed.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { (*parent).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
let __v0: bool = !has_cpus && is_in_v2_mode();
if __v0 {
cpumask_copy(new_cpus, (unsafe { (*parent).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
}
}

/// compute_effective_nodemask - Compute the effective nodemask of the cpuset
/// @new_mems: the temp variable for the new effective_mems mask
/// @cs: the cpuset the need to recompute the new effective_mems mask
/// @parent: the parent cpuset
///
/// For v2, the parent's effective_mems is inherited if nodemask is empty.
fn compute_effective_nodemask(new_mems: *mut __SlateRecord_nodemask_types_h_17, cs: *mut cpuset, parent: *mut cpuset) {
let mut has_mems: bool = false;
has_mems = nodes_and!(*new_mems, (*cs).mems_allowed, (*parent).effective_mems);
let __v0: bool = !has_mems && is_in_v2_mode();
if __v0 {
nodes_copy!(*new_mems, (*parent).effective_mems);
}
}

// Commands for update_parent_effective_cpumask
/// Update partition exclusive flag
///
/// Return: 0 if successful, an error code otherwise
fn update_partition_exclusive_flag(cs: *mut cpuset, new_prs: i32) -> i32 {
let exclusive: bool = new_prs > 0;
let __v0: bool = exclusive && is_cpu_exclusive(cs as *const cpuset) == 0;
if __v0 {
if cpuset_update_flag(0u32, cs, 1i32) != 0 {
return 4;
}
} else {
let __v1: bool = !exclusive && is_cpu_exclusive(cs as *const cpuset) != 0;
if __v1 {
// Turning off CS_CPU_EXCLUSIVE will not return error
cpuset_update_flag(0u32, cs, 0i32);
}
}
0
}

/// Update partition load balance flag and/or rebuild sched domain
///
/// Changing load balance flag will automatically call
/// rebuild_sched_domains_locked().
/// This function is for cgroup v2 only.
fn update_partition_sd_lb(cs: *mut cpuset, old_prs: i32) {
let new_prs: i32 = unsafe { (*cs).partition_root_state };
let mut rebuild_domains: bool = new_prs > 0 || old_prs > 0;
let mut new_lb: bool = false;
// If cs is not a valid partition root, the load balance state
// will follow its parent.
if new_prs > 0 {
new_lb = new_prs != 2;
} else {
new_lb = is_sched_load_balance(parent_cs(cs) as *const cpuset) != 0;
}
if (new_lb as i32) != ((is_sched_load_balance(cs as *const cpuset) != 0) as i32) {
rebuild_domains = true;
if new_lb {
set_bit(4i64, (unsafe { core::ptr::addr_of_mut!((*cs).flags) }) as *mut u64);
} else {
clear_bit(4i64, (unsafe { core::ptr::addr_of_mut!((*cs).flags) }) as *mut u64);
}
}
if rebuild_domains {
cpuset_force_rebuild();
}
}

/// tasks_nocpu_error - Return true if tasks will have no effective_cpus
fn tasks_nocpu_error(parent: *mut cpuset, cs: *mut cpuset, xcpus: *mut cpumask) -> bool {
// A populated partition (cs or parent) can't have empty effective_cpus
let __v0: bool = cpumask_subset((unsafe { (*parent).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, xcpus as *const cpumask) && partition_is_populated(parent, cs);
__v0 || !cpumask_intersects(xcpus as *const cpumask, (unsafe { core::ptr::addr_of_mut!(__cpu_active_mask) }) as *const cpumask) && partition_is_populated(cs, core::ptr::null_mut::<cpuset>())
}

fn reset_partition_data(cs: *mut cpuset) {
let parent: *mut cpuset = parent_cs(cs);
if !cpuset_v2() {
return;
}
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(callback_lock) });
if cpumask_empty((unsafe { (*cs).exclusive_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask) {
cpumask_clear(unsafe { (*cs).effective_xcpus.as_mut_ptr() as *mut cpumask });
if is_cpu_exclusive(cs as *const cpuset) != 0 {
clear_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*cs).flags) }) as *mut u64);
}
}
if !cpumask_and(unsafe { (*cs).effective_cpus.as_mut_ptr() as *mut cpumask }, (unsafe { (*parent).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { (*cs).cpus_allowed.as_mut_ptr() as *mut cpumask }) as *const cpumask) {
cpumask_copy(unsafe { (*cs).effective_cpus.as_mut_ptr() as *mut cpumask }, (unsafe { (*parent).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
}
}

/// Return true if isolated_cpus changes.
fn isolated_cpu_update(new_prs: i32, cpu: i32) -> bool {
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(callback_lock) });
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(cpuset_mutex) });
if new_prs == 2 {
if cpumask_test_cpu(cpu, (unsafe { core::ptr::addr_of_mut!(isolated_cpus) as *mut cpumask }) as *const cpumask) {
return false;
}
cpumask_set_cpu(cpu as u32, unsafe { core::ptr::addr_of_mut!(isolated_cpus) as *mut cpumask });
return true;
}
// CPUs isolated at boot must remain isolated.
let __v2: bool = !cpumask_test_cpu(cpu, unsafe { housekeeping_cpumask(0u32) }) || !cpumask_test_cpu(cpu, (unsafe { core::ptr::addr_of_mut!(isolated_cpus) as *mut cpumask }) as *const cpumask);
if __v2 {
return false;
}
cpumask_clear_cpu(cpu, unsafe { core::ptr::addr_of_mut!(isolated_cpus) as *mut cpumask });
true
}

/// isolated_cpus_update - Update the isolated_cpus mask
/// @old_prs: old partition_root_state
/// @new_prs: new partition_root_state
/// @xcpus: exclusive CPUs with state change
fn isolated_cpus_update(old_prs: i32, new_prs: i32, xcpus: *mut cpumask) {
let mut updated: bool = false;
let mut cpu: i32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = (old_prs == new_prs) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("686: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 686b - ., 3; .popsection\n# {0}", const 686 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_isolated_cpus_update_1, sym __slate_asm_string_isolated_cpus_update_2, const 1295 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("687: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 687b - ., 4; .popsection\n# {0}", const 687 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(callback_lock) });
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(cpuset_mutex) });
loop {
cpu = find_next_bit(cpumask_bits__2!(xcpus) as *const u64, 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let __v7: bool = isolated_cpu_update(new_prs, cpu);
updated = (updated as i32) | (__v7 as i32) != 0;
cpu = cpu.wrapping_add(1);
}
if updated {
unsafe {
update_housekeeping = true;
}
}
}

/// partition_xcpus_add - Add new exclusive CPUs to partition
/// @new_prs: new partition_root_state
/// @parent: parent cpuset
/// @xcpus: exclusive CPUs to be added
///
/// Remote partition if parent == NULL
fn partition_xcpus_add(new_prs: i32, mut parent: *mut cpuset, xcpus: *mut cpumask) {
let mut __v0: i64;
let __ret_warn_on: i32 = (new_prs < 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("688: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 688b - ., 3; .popsection\n# {0}", const 688 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_partition_xcpus_add_1, sym __slate_asm_string_partition_xcpus_add_2, const 1315 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("689: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 689b - ., 4; .popsection\n# {0}", const 689 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(callback_lock) });
if parent.is_null() {
parent = unsafe { core::ptr::addr_of_mut!(top_cpuset) };
}
if parent == unsafe { core::ptr::addr_of_mut!(top_cpuset) } {
cpumask_or(unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask }, (unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask }) as *const cpumask, xcpus as *const cpumask);
}
if new_prs != unsafe { (*parent).partition_root_state } {
isolated_cpus_update(unsafe { (*parent).partition_root_state }, new_prs, xcpus);
}
cpumask_andnot(unsafe { (*parent).effective_cpus.as_mut_ptr() as *mut cpumask }, (unsafe { (*parent).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, xcpus as *const cpumask);
}

/// partition_xcpus_del - Remove exclusive CPUs from partition
/// @old_prs: old partition_root_state
/// @parent: parent cpuset
/// @xcpus: exclusive CPUs to be removed
///
/// Remote partition if parent == NULL
fn partition_xcpus_del(old_prs: i32, mut parent: *mut cpuset, xcpus: *mut cpumask) {
let mut __v0: i64;
let __ret_warn_on: i32 = (old_prs < 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("690: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 690b - ., 3; .popsection\n# {0}", const 690 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_partition_xcpus_del_1, sym __slate_asm_string_partition_xcpus_del_2, const 1342 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("691: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 691b - ., 4; .popsection\n# {0}", const 691 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(callback_lock) });
if parent.is_null() {
parent = unsafe { core::ptr::addr_of_mut!(top_cpuset) };
}
if parent == unsafe { core::ptr::addr_of_mut!(top_cpuset) } {
cpumask_andnot(unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask }, (unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask }) as *const cpumask, xcpus as *const cpumask);
}
if old_prs != unsafe { (*parent).partition_root_state } {
isolated_cpus_update(old_prs, unsafe { (*parent).partition_root_state }, xcpus);
}
cpumask_or(unsafe { (*parent).effective_cpus.as_mut_ptr() as *mut cpumask }, (unsafe { (*parent).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, xcpus as *const cpumask);
cpumask_and(unsafe { (*parent).effective_cpus.as_mut_ptr() as *mut cpumask }, (unsafe { (*parent).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { core::ptr::addr_of_mut!(__cpu_active_mask) }) as *const cpumask);
}

/// isolated_cpus_can_update - check for isolated & nohz_full conflicts
/// @add_cpus: cpu mask for cpus that are going to be isolated
/// @del_cpus: cpu mask for cpus that are no longer isolated, can be NULL
/// Return: false if there is conflict, true otherwise
///
/// If nohz_full is enabled and we have isolated CPUs, their combination must
/// still leave housekeeping CPUs.
///
/// TBD: Should consider merging this function into
///      prstate_housekeeping_conflict().
fn isolated_cpus_can_update(add_cpus: *mut cpumask, del_cpus: *mut cpumask) -> bool {
let mut full_hk_cpus: [cpumask; 1] = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(full_hk_cpus) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut res: i32 = 1;
if !unsafe { housekeeping_enabled(3u32) } {
return true;
}
let __v0: bool = !del_cpus.is_null() && cpumask_weight_and(del_cpus as *const cpumask, unsafe { housekeeping_cpumask(3u32) }) != 0;
if __v0 {
return true;
}
if !alloc_cpumask_var(core::ptr::addr_of_mut!(full_hk_cpus), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) {
return false;
}
cpumask_and(full_hk_cpus.as_mut_ptr() as *mut cpumask, unsafe { housekeeping_cpumask(3u32) }, unsafe { housekeeping_cpumask(1u32) });
cpumask_andnot(full_hk_cpus.as_mut_ptr() as *mut cpumask, full_hk_cpus.as_mut_ptr() as *const cpumask, (unsafe { core::ptr::addr_of_mut!(isolated_cpus) as *mut cpumask }) as *const cpumask);
cpumask_and(full_hk_cpus.as_mut_ptr() as *mut cpumask, full_hk_cpus.as_mut_ptr() as *const cpumask, (unsafe { core::ptr::addr_of_mut!(__cpu_active_mask) }) as *const cpumask);
if cpumask_weight_andnot(full_hk_cpus.as_mut_ptr() as *const cpumask, add_cpus as *const cpumask) == 0 {
res = 0;
}
free_cpumask_var(full_hk_cpus.as_mut_ptr() as *mut cpumask);
res != 0
}

/// prstate_housekeeping_conflict - check for partition & housekeeping conflicts
/// @prstate: partition root state to be checked
/// @new_cpus: cpu mask
/// Return: true if there is conflict, false otherwise
///
/// CPUs outside of HK_TYPE_DOMAIN_BOOT, if defined, can only be used in an
/// isolated partition.
fn prstate_housekeeping_conflict(prstate: i32, new_cpus: *mut cpumask) -> bool {
if !unsafe { housekeeping_enabled(0u32) } {
return false;
}
let __v0: bool = prstate != 2 && !cpumask_subset(new_cpus as *const cpumask, unsafe { housekeeping_cpumask(0u32) });
if __v0 {
return true;
}
false
}

/// cpuset_update_sd_hk_unlock - Rebuild sched domains, update HK & unlock
///
/// Update housekeeping cpumasks and rebuild sched domains if necessary and
/// then do a cpuset_full_unlock().
/// This should be called at the end of cpuset operation.
fn cpuset_update_sd_hk_unlock() {
// force_sd_rebuild will be cleared in rebuild_sched_domains_locked()
if unsafe { force_sd_rebuild } {
rebuild_sched_domains_locked();
}
if unsafe { update_housekeeping } {
unsafe {
update_housekeeping = false;
}
cpumask_copy(unsafe { core::ptr::addr_of_mut!(isolated_hk_cpus) as *mut cpumask }, (unsafe { core::ptr::addr_of_mut!(isolated_cpus) as *mut cpumask }) as *const cpumask);
// housekeeping_update() is now called without holding
// cpus_read_lock and cpuset_mutex. Only cpuset_top_mutex
// is still being held for mutual exclusion.
unsafe { mutex_unlock(core::ptr::addr_of_mut!(cpuset_mutex)) };
unsafe { cpus_read_unlock() };
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { housekeeping_update(core::ptr::addr_of_mut!(isolated_hk_cpus) as *mut cpumask) }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("692: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 692b - ., 3; .popsection\n# {0}", const 692 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_cpuset_update_sd_hk_unlock_1, sym __slate_asm_string_cpuset_update_sd_hk_unlock_2, const 1444 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("693: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 693b - ., 4; .popsection\n# {0}", const 693 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe { mutex_unlock(core::ptr::addr_of_mut!(cpuset_top_mutex)) };
} else {
cpuset_full_unlock();
}
}

/// Work function to invoke cpuset_update_sd_hk_unlock()
#[unsafe(link_section = ".text.slate_distinct.cpuset.hk_sd_workfn")]
extern "C" fn hk_sd_workfn(work: *mut work_struct) {
cpuset_full_lock();
cpuset_update_sd_hk_unlock();
}

/// rm_siblings_excl_cpus - Remove exclusive CPUs that are used by sibling cpusets
/// @parent: Parent cpuset containing all siblings
/// @cs: Current cpuset (will be skipped)
/// @excpus:  exclusive effective CPU mask to modify
///
/// This function ensures the given @excpus mask doesn't include any CPUs that
/// are exclusively allocated to sibling cpusets. It walks through all siblings
/// of @cs under @parent and removes their exclusive CPUs from @excpus.
fn rm_siblings_excl_cpus(parent: *mut cpuset, cs: *mut cpuset, excpus: *mut cpumask) -> i32 {
let mut css: *mut cgroup_subsys_state = unsafe { core::mem::zeroed() };
let mut retval: i32 = 0;
if cpumask_empty(excpus as *const cpumask) {
return 0;
}
// Remove exclusive CPUs from siblings
rcu_read_lock();
css = unsafe { css_next_child(core::ptr::null_mut::<cgroup_subsys_state>(), core::ptr::addr_of_mut!((*parent).css)) };
while !css.is_null() {
let __v0: *mut cpuset = css_cs(css);
let sibling: *mut cpuset = __v0;
if is_cpuset_online(__v0) {
if sibling != cs {
// If exclusive_cpus is defined, effective_xcpus will always
// be a subset. Otherwise, effective_xcpus will only be set
// in a valid partition root.
let sibling_xcpus: *mut cpumask = if cpumask_empty((unsafe { (*sibling).exclusive_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask) { unsafe { (*sibling).effective_xcpus.as_mut_ptr() as *mut cpumask } } else { unsafe { (*sibling).exclusive_cpus.as_mut_ptr() as *mut cpumask } };
if cpumask_intersects(excpus as *const cpumask, sibling_xcpus as *const cpumask) {
cpumask_andnot(excpus, excpus as *const cpumask, sibling_xcpus as *const cpumask);
retval = retval.wrapping_add(1);
}
}
}
css = unsafe { css_next_child(css, core::ptr::addr_of_mut!((*parent).css)) };
}
rcu_read_unlock();
retval
}

/// compute_excpus - compute effective exclusive CPUs
/// @cs: cpuset
/// @xcpus: effective exclusive CPUs value to be set
/// Return: 0 if there is no sibling conflict, > 0 otherwise
///
/// If exclusive_cpus isn't explicitly set , we have to scan the sibling cpusets
/// and exclude their exclusive_cpus or effective_xcpus as well.
fn compute_excpus(cs: *mut cpuset, excpus: *mut cpumask) -> i32 {
let parent: *mut cpuset = parent_cs(cs);
cpumask_and(excpus, user_xcpus(cs) as *const cpumask, (unsafe { (*parent).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
if !cpumask_empty((unsafe { (*cs).exclusive_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask) {
return 0;
}
rm_siblings_excl_cpus(parent, cs, excpus)
}

/// compute_trialcs_excpus - Compute effective exclusive CPUs for a trial cpuset
/// @trialcs: The trial cpuset containing the proposed new configuration
/// @cs: The original cpuset that the trial configuration is based on
/// Return: 0 if successful with no sibling conflict, >0 if a conflict is found
///
/// Computes the effective_xcpus for a trial configuration. @cs is provided to represent
/// the real cs.
fn compute_trialcs_excpus(trialcs: *mut cpuset, cs: *mut cpuset) -> i32 {
let parent: *mut cpuset = parent_cs(trialcs);
let excpus: *mut cpumask = unsafe { (*trialcs).effective_xcpus.as_mut_ptr() as *mut cpumask };
// trialcs is member, cpuset.cpus has no impact to excpus
if cs_is_member(cs as *const cpuset) {
cpumask_and(excpus, (unsafe { (*trialcs).exclusive_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { (*parent).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
} else {
cpumask_and(excpus, user_xcpus(trialcs) as *const cpumask, (unsafe { (*parent).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
}
rm_siblings_excl_cpus(parent, cs, excpus)
}

fn is_remote_partition(cs: *mut cpuset) -> bool {
unsafe { (*cs).remote_partition }
}

fn is_local_partition(cs: *mut cpuset) -> bool {
is_partition_valid(cs as *const cpuset) && !is_remote_partition(cs)
}

// remote_partition_enable - Enable current cpuset as a remote partition root
// @cs: the cpuset to update
// @new_prs: new partition_root_state
// @tmp: temporary masks
// Return: 0 if successful, errcode if error
//
// Enable the current cpuset to become a remote partition root taking CPUs
// directly from the top cpuset. cpuset_mutex must be held by the caller.
fn remote_partition_enable(cs: *mut cpuset, new_prs: i32, tmp: *mut tmpmasks) -> i32 {
// The user must have sysadmin privilege.
if !unsafe { capable(21i32) } {
return 9;
}
// The requested exclusive_cpus must not be allocated to other
// partitions and it can't use up all the root's effective_cpus.
//
// The effective_xcpus mask can contain offline CPUs, but there must
// be at least one or more online CPUs present before it can be enabled.
//
// Note that creating a remote partition with any local partition root
// above it or remote partition root underneath it is not allowed.
compute_excpus(cs, unsafe { (*tmp).new_cpus.as_mut_ptr() as *mut cpumask });
let __v0: bool = !cpumask_intersects((unsafe { (*tmp).new_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { core::ptr::addr_of_mut!(__cpu_active_mask) }) as *const cpumask) || cpumask_subset((unsafe { core::ptr::addr_of_mut!(top_cpuset.effective_cpus) as *mut cpumask }) as *const cpumask, (unsafe { (*tmp).new_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
if __v0 {
return 1;
}
if cpumask_intersects((unsafe { (*tmp).new_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask }) as *const cpumask) {
return 5;
}
let __v1: bool = new_prs == 2 && !isolated_cpus_can_update(unsafe { (*tmp).new_cpus.as_mut_ptr() as *mut cpumask }, core::ptr::null_mut::<cpumask>());
let __v2: bool = __v1 || prstate_housekeeping_conflict(new_prs, unsafe { (*tmp).new_cpus.as_mut_ptr() as *mut cpumask });
if __v2 {
return 8;
}
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
partition_xcpus_add(new_prs, core::ptr::null_mut::<cpuset>(), unsafe { (*tmp).new_cpus.as_mut_ptr() as *mut cpumask });
unsafe {
(*cs).remote_partition = true;
}
cpumask_copy(unsafe { (*cs).effective_xcpus.as_mut_ptr() as *mut cpumask }, (unsafe { (*tmp).new_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
cpuset_force_rebuild();
loop {
while false {
}
__WRITE_ONCE__14!((*cs).prs_err, 0i32);
if true {
break;
}
}
// Propagate changes in top_cpuset's effective_cpus down the hierarchy.
cpuset_update_tasks_cpumask(unsafe { core::ptr::addr_of_mut!(top_cpuset) }, unsafe { (*tmp).new_cpus.as_mut_ptr() as *mut cpumask });
update_sibling_cpumasks(unsafe { core::ptr::addr_of_mut!(top_cpuset) }, core::ptr::null_mut::<cpuset>(), tmp);
0
}

/// remote_partition_disable - Remove current cpuset from remote partition list
/// @cs: the cpuset to update
/// @tmp: temporary masks
///
/// The effective_cpus is also updated.
///
/// cpuset_mutex must be held by the caller.
fn remote_partition_disable(cs: *mut cpuset, tmp: *mut tmpmasks) {
let mut __v0: i64;
{
let __ret_warn_on: i32 = !(!(!is_remote_partition(cs))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("695: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 695b - ., 3; .popsection\n# {0}", const 695 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_remote_partition_disable_1, sym __slate_asm_string_remote_partition_disable_2, const 1631 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("696: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 696b - ., 4; .popsection\n# {0}", const 696 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
// When a CPU is offlined, top_cpuset may end up with no available CPUs,
// which should clear subpartitions_cpus. We should not emit a warning for this
// scenario: the hierarchy is updated from top to bottom, so subpartitions_cpus
// may already be cleared when disabling the partition.
let mut __v5: i64;
let mut __ret_warn_on: i32 = 0;
let __v6: bool = !cpumask_subset((unsafe { (*cs).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask }) as *const cpumask) && !cpumask_empty((unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask }) as *const cpumask);
__ret_warn_on = __v6 as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("697: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 697b - ., 3; .popsection\n# {0}", const 697 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_remote_partition_disable_7, sym __slate_asm_string_remote_partition_disable_8, const 1639 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("698: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 698b - ., 4; .popsection\n# {0}", const 698 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
unsafe {
(*cs).remote_partition = false;
}
partition_xcpus_del(unsafe { (*cs).partition_root_state }, core::ptr::null_mut::<cpuset>(), unsafe { (*cs).effective_xcpus.as_mut_ptr() as *mut cpumask });
if ((unsafe { (*cs).prs_err }) as u32) != 0 {
unsafe {
(*cs).partition_root_state = unsafe { (*cs).partition_root_state }.wrapping_neg();
}
} else {
unsafe {
(*cs).partition_root_state = 0;
}
}
// effective_xcpus may need to be changed
compute_excpus(cs, unsafe { (*cs).effective_xcpus.as_mut_ptr() as *mut cpumask });
reset_partition_data(cs);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
cpuset_force_rebuild();
// Propagate changes in top_cpuset's effective_cpus down the hierarchy.
cpuset_update_tasks_cpumask(unsafe { core::ptr::addr_of_mut!(top_cpuset) }, unsafe { (*tmp).new_cpus.as_mut_ptr() as *mut cpumask });
update_sibling_cpumasks(unsafe { core::ptr::addr_of_mut!(top_cpuset) }, core::ptr::null_mut::<cpuset>(), tmp);
}

// remote_cpus_update - cpus_exclusive change of remote partition
// @cs: the cpuset to be updated
// @xcpus: the new exclusive_cpus mask, if non-NULL
// @excpus: the new effective_xcpus mask
// @tmp: temporary masks
//
// top_cpuset and subpartitions_cpus will be updated or partition can be
// invalidated.
fn remote_cpus_update(cs: *mut cpuset, xcpus: *mut cpumask, excpus: *mut cpumask, tmp: *mut tmpmasks) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __ret_warn_on_1: i32 = unsafe { core::mem::zeroed() };
let mut prs: i32 = unsafe { core::mem::zeroed() };
let mut deleting: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut bool = core::ptr::addr_of_mut!(deleting) as *mut bool;
let mut adding: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut bool = core::ptr::addr_of_mut!(adding) as *mut bool;
unsafe {
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
prs = (*cs).partition_root_state;
__ret_warn_on_1 = !(!(!is_remote_partition(cs))) as i32;
if unlikely!(__ret_warn_on_1) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("699: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 699b - ., 3; .popsection\n# {0}", const 699 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_remote_cpus_update_22, sym __slate_asm_string_remote_cpus_update_23, const 1678 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("700: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 700b - ., 4; .popsection\n# {0}", const 700 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v10: i64 = (__ret_warn_on_1 != 0) as i64;
if __v10 != 0 {
return;
} else {
{
let __ret_warn_on: i32 = !(!(!cpumask_subset(((*cs).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask, (core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask) as *const cpumask))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("701: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 701b - ., 3; .popsection\n# {0}", const 701 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_remote_cpus_update_20, sym __slate_asm_string_remote_cpus_update_21, const 1681 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("702: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 702b - ., 4; .popsection\n# {0}", const 702 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
'__join_0: {
if cpumask_empty(excpus as *const cpumask) {
loop {
if true {
__WRITE_ONCE__14!((*cs).prs_err, 7i32);
if true {
break;
}
}
}
} else {
*__slate_slot_13 = cpumask_andnot((*tmp).addmask.as_mut_ptr() as *mut cpumask, excpus as *const cpumask, ((*cs).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
*__slate_slot_12 = cpumask_andnot((*tmp).delmask.as_mut_ptr() as *mut cpumask, ((*cs).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask, excpus as *const cpumask);
// Additions of remote CPUs is only allowed if those CPUs are
// not allocated to other partitions and there are effective_cpus
// left in the top cpuset.
if *__slate_slot_13 {
let __ret_warn_on: i32 = !(!cpumask_intersects(((*tmp).addmask.as_mut_ptr() as *mut cpumask) as *const cpumask, (core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask) as *const cpumask)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("704: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 704b - ., 3; .popsection\n# {0}", const 704 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_remote_cpus_update_17, sym __slate_asm_string_remote_cpus_update_18, const 1697 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("705: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 705b - ., 4; .popsection\n# {0}", const 705 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
if !capable(21i32) {
loop {
if true {
__WRITE_ONCE__14!((*cs).prs_err, 9i32);
if true {
break;
}
}
}
} else {
if cpumask_intersects(((*tmp).addmask.as_mut_ptr() as *mut cpumask) as *const cpumask, (core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask) as *const cpumask) {
__v1 = true;
} else {
__v1 = cpumask_subset((core::ptr::addr_of_mut!(top_cpuset.effective_cpus) as *mut cpumask) as *const cpumask, ((*tmp).addmask.as_mut_ptr() as *mut cpumask) as *const cpumask);
}
if __v1 {
loop {
if true {
__WRITE_ONCE__14!((*cs).prs_err, 5i32);
if true {
break;
}
}
}
} else {
if prs == 2 {
__v0 = !isolated_cpus_can_update((*tmp).addmask.as_mut_ptr() as *mut cpumask, (*tmp).delmask.as_mut_ptr() as *mut cpumask);
} else {
__v0 = false;
}
if __v0 {
loop {
if true {
__WRITE_ONCE__14!((*cs).prs_err, 8i32);
if true {
break;
}
}
}
}
}
}
if ((*cs).prs_err as u32) != 0 {
break '__join_0;
}
}
spin_lock_irq(core::ptr::addr_of_mut!(callback_lock));
if *__slate_slot_13 {
partition_xcpus_add(prs, core::ptr::null_mut::<cpuset>(), (*tmp).addmask.as_mut_ptr() as *mut cpumask);
}
if *__slate_slot_12 {
partition_xcpus_del(prs, core::ptr::null_mut::<cpuset>(), (*tmp).delmask.as_mut_ptr() as *mut cpumask);
}
// Need to update effective_xcpus and exclusive_cpus now as
// update_sibling_cpumasks() below may iterate back to the same cs.
cpumask_copy((*cs).effective_xcpus.as_mut_ptr() as *mut cpumask, excpus as *const cpumask);
if !xcpus.is_null() {
cpumask_copy((*cs).exclusive_cpus.as_mut_ptr() as *mut cpumask, xcpus as *const cpumask);
}
spin_unlock_irq(core::ptr::addr_of_mut!(callback_lock));
if *__slate_slot_13 || *__slate_slot_12 {
cpuset_force_rebuild();
}
// Propagate changes in top_cpuset's effective_cpus down the hierarchy.
cpuset_update_tasks_cpumask(core::ptr::addr_of_mut!(top_cpuset), (*tmp).new_cpus.as_mut_ptr() as *mut cpumask);
update_sibling_cpumasks(core::ptr::addr_of_mut!(top_cpuset), core::ptr::null_mut::<cpuset>(), tmp);
return;
}
}
remote_partition_disable(cs, tmp);
}
}
}

// update_parent_effective_cpumask - update effective_cpus mask of parent cpuset
// @cs:      The cpuset that requests change in partition root state
// @cmd:     Partition root state change command
// @newmask: Optional new cpumask for partcmd_update
// @tmp:     Temporary addmask and delmask
// Return:   0 or a partition root state error code
//
// For partcmd_enable*, the cpuset is being transformed from a non-partition
// root to a partition root. The effective_xcpus (cpus_allowed if
// effective_xcpus not set) mask of the given cpuset will be taken away from
// parent's effective_cpus. The function will return 0 if all the CPUs listed
// in effective_xcpus can be granted or an error code will be returned.
//
// For partcmd_disable, the cpuset is being transformed from a partition
// root back to a non-partition root. Any CPUs in effective_xcpus will be
// given back to parent's effective_cpus. 0 will always be returned.
//
// For partcmd_update, if the optional newmask is specified, the cpu list is
// to be changed from effective_xcpus to newmask. Otherwise, effective_xcpus is
// assumed to remain the same. The cpuset should either be a valid or invalid
// partition root. The partition root state may change from valid to invalid
// or vice versa. An error code will be returned if transitioning from
// invalid to valid violates the exclusivity rule.
//
// For partcmd_invalidate, the current partition will be made invalid.
//
// The partcmd_enable* and partcmd_disable commands are used by
// update_prstate(). An error code may be returned and the caller will check
// for error.
//
// The partcmd_update command is used by update_cpumasks_hier() with newmask
// NULL and update_cpumask() with newmask set. The partcmd_invalidate is used
// by update_cpumask() with NULL newmask. In both cases, the callers won't
// check for error and so partition_root_state and prs_err will be updated
// directly.
fn update_parent_effective_cpumask(cs: *mut cpuset, cmd: i32, newmask: *mut cpumask, tmp: *mut tmpmasks) -> i32 {
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut __v8: bool = unsafe { core::mem::zeroed() };
// Make partition invalid if parent's effective_cpus could
// become empty and there are tasks in the parent.
let mut __v9: bool = unsafe { core::mem::zeroed() };
let mut __v13: bool = unsafe { core::mem::zeroed() };
let mut __v14: bool = unsafe { core::mem::zeroed() };
let mut exclusive: bool = unsafe { core::mem::zeroed() };
let mut child: core::mem::MaybeUninit<*mut cpuset> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut cpuset = core::ptr::addr_of_mut!(child) as *mut *mut cpuset;
let mut css: core::mem::MaybeUninit<*mut cgroup_subsys_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *mut cgroup_subsys_state = core::ptr::addr_of_mut!(css) as *mut *mut cgroup_subsys_state;
let mut __v22: bool = unsafe { core::mem::zeroed() };
let mut __v23: bool = unsafe { core::mem::zeroed() };
let mut __v24: bool = unsafe { core::mem::zeroed() };
let mut nocpu: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut bool = core::ptr::addr_of_mut!(nocpu) as *mut bool;
let mut parent_prs: i32 = unsafe { core::mem::zeroed() };
let mut xcpus: *mut cpumask = unsafe { core::mem::zeroed() };
// Partition error?
let mut part_error: i32 = unsafe { core::mem::zeroed() };
let mut new_prs: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut i32 = core::ptr::addr_of_mut!(new_prs) as *mut i32;
let mut old_prs: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut i32 = core::ptr::addr_of_mut!(old_prs) as *mut i32; // Deleting cpus from parent's effective_cpus
let mut deleting: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut i32 = core::ptr::addr_of_mut!(deleting) as *mut i32; // Adding cpus to parent's effective_cpus
let mut adding: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut i32 = core::ptr::addr_of_mut!(adding) as *mut i32;
let mut parent: *mut cpuset = unsafe { core::mem::zeroed() };
unsafe {
{
parent = parent_cs(cs);
let __slate_auto_bytes_1 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
part_error = 0;
xcpus = user_xcpus(cs);
parent_prs = (*parent).partition_root_state;
let __slate_auto_bytes = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
lockdep_assert_cpuset_lock_held();
let __ret_warn_on: i32 = !(!is_remote_partition(cs)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("709: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 709b - ., 3; .popsection\n# {0}", const 709 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_update_parent_effective_cpumask_45, sym __slate_asm_string_update_parent_effective_cpumask_46, const 1788 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("710: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 710b - ., 4; .popsection\n# {0}", const 710 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
// For local partition only
// new_prs will only be changed for the partcmd_update and
// partcmd_invalidate commands.
*__slate_slot_35 = 0;
*__slate_slot_36 = 0;
let __v25: i32 = (*cs).partition_root_state;
*__slate_slot_33 = __v25;
*__slate_slot_34 = __v25;
if cmd == 4 {
if is_partition_invalid(cs as *const cpuset) {
return 0;
} else {
// Make the current partition invalid.
if is_partition_valid(parent as *const cpuset) {
*__slate_slot_36 = cpumask_and((*tmp).addmask.as_mut_ptr() as *mut cpumask, ((*cs).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask, ((*parent).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask) as i32;
}
if *__slate_slot_34 > 0 {
*__slate_slot_33 = (*__slate_slot_34).wrapping_neg();
}
}
} else {
// The parent must be a partition root.
// The new cpumask, if present, or the current cpus_allowed must
// not be empty.
if !is_partition_valid(parent as *const cpuset) {
return if is_partition_invalid(parent as *const cpuset) { 2 } else { 3 };
} else {
if newmask.is_null() {
__v24 = xcpus_empty(cs);
} else {
__v24 = false;
}
if __v24 {
return 7;
} else {
*__slate_slot_29 = tasks_nocpu_error(parent, cs, xcpus);
if cmd == 0 || cmd == 1 {
// Need to call compute_excpus() in case
// exclusive_cpus not set. Sibling conflict should only happen
// if exclusive_cpus isn't set.
xcpus = (*tmp).delmask.as_mut_ptr() as *mut cpumask;
if compute_excpus(cs, xcpus) != 0 {
let __ret_warn_on: i32 = !(!(!cpumask_empty(((*cs).exclusive_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("711: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 711b - ., 3; .popsection\n# {0}", const 711 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_update_parent_effective_cpumask_41, sym __slate_asm_string_update_parent_effective_cpumask_42, const 1836 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("712: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 712b - ., 4; .popsection\n# {0}", const 712 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
*__slate_slot_33 = if cmd == 0 { 1 } else { 2 };
// Enabling partition root is not allowed if its
// effective_xcpus is empty.
if cpumask_empty(xcpus as *const cpumask) {
return 1;
} else {
if prstate_housekeeping_conflict(*__slate_slot_33, xcpus) {
return 8;
} else {
if *__slate_slot_33 == 2 && *__slate_slot_33 != parent_prs {
__v4 = !isolated_cpus_can_update(xcpus, core::ptr::null_mut::<cpumask>());
} else {
__v4 = false;
}
if __v4 {
return 8;
} else {
if tasks_nocpu_error(parent, cs, xcpus) {
return 5;
} else {
// This function will only be called when all the preliminary
// checks have passed. At this point, the following condition
// should hold.
//
// (cs->effective_xcpus & cpu_active_mask) ⊆ parent->effective_cpus
//
// Warn if it is not the case.
cpumask_and((*tmp).new_cpus.as_mut_ptr() as *mut cpumask, xcpus as *const cpumask, core::ptr::addr_of_mut!(__cpu_active_mask) as *const cpumask);
let __ret_warn_on: i32 = !(!(!cpumask_subset(((*tmp).new_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask, ((*parent).effective_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("713: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 713b - ., 3; .popsection\n# {0}", const 713 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_update_parent_effective_cpumask_39, sym __slate_asm_string_update_parent_effective_cpumask_40, const 1866 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("714: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 714b - ., 4; .popsection\n# {0}", const 714 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
*__slate_slot_35 = 1;
}
}
}
}
} else {
if cmd == 2 {
// May need to add cpus back to parent's effective_cpus
// (and maybe removed from subpartitions_cpus/isolated_cpus)
// for valid partition root. xcpus may contain CPUs that
// shouldn't be removed from the two global cpumasks.
if is_partition_valid(cs as *const cpuset) {
cpumask_copy((*tmp).addmask.as_mut_ptr() as *mut cpumask, ((*cs).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
*__slate_slot_36 = 1;
}
*__slate_slot_33 = 0;
} else {
if !newmask.is_null() {
// Empty cpumask is not allowed
if cpumask_empty(newmask as *const cpumask) {
part_error = 7;
} else {
// Check newmask again, whether cpus are available for parent/cs
let __v17: bool = tasks_nocpu_error(parent, cs, newmask);
*__slate_slot_29 = (*__slate_slot_29 as i32) | (__v17 as i32) != 0;
// partcmd_update with newmask:
//
// Compute add/delete mask to/from effective_cpus
//
// For valid partition:
//   addmask = effective_xcpus & ~newmask
// 			      & parent->effective_xcpus
//   delmask = newmask & ~effective_xcpus
// 		       & parent->effective_xcpus
//
// For invalid partition:
//   delmask = newmask & parent->effective_xcpus
//   The partition may become valid soon.
if is_partition_invalid(cs as *const cpuset) {
*__slate_slot_36 = 0;
*__slate_slot_35 = cpumask_and((*tmp).delmask.as_mut_ptr() as *mut cpumask, newmask as *const cpumask, ((*parent).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask) as i32;
} else {
cpumask_andnot((*tmp).addmask.as_mut_ptr() as *mut cpumask, ((*cs).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask, newmask as *const cpumask);
*__slate_slot_36 = cpumask_and((*tmp).addmask.as_mut_ptr() as *mut cpumask, ((*tmp).addmask.as_mut_ptr() as *mut cpumask) as *const cpumask, ((*parent).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask) as i32;
cpumask_andnot((*tmp).delmask.as_mut_ptr() as *mut cpumask, newmask as *const cpumask, ((*cs).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
*__slate_slot_35 = cpumask_and((*tmp).delmask.as_mut_ptr() as *mut cpumask, ((*tmp).delmask.as_mut_ptr() as *mut cpumask) as *const cpumask, ((*parent).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask) as i32;
}
// TBD: Invalidate a currently valid child root partition may
// still break isolated_cpus_can_update() rule if parent is an
// isolated partition.
if is_partition_valid(cs as *const cpuset) && *__slate_slot_34 != parent_prs {
if parent_prs == 1 {
__v14 = !isolated_cpus_can_update((*tmp).delmask.as_mut_ptr() as *mut cpumask, (*tmp).addmask.as_mut_ptr() as *mut cpumask);
} else {
__v14 = false;
}
if __v14 {
part_error = 8;
}
// Adding to parent means removing isolated CPUs
if parent_prs == 2 {
__v13 = !isolated_cpus_can_update((*tmp).addmask.as_mut_ptr() as *mut cpumask, (*tmp).delmask.as_mut_ptr() as *mut cpumask);
} else {
__v13 = false;
}
if __v13 {
part_error = 8;
}
// Adding to parent means adding isolated CPUs
}
// The new CPUs to be removed from parent's effective CPUs
// must be present.
if *__slate_slot_35 != 0 {
cpumask_and((*tmp).new_cpus.as_mut_ptr() as *mut cpumask, ((*tmp).delmask.as_mut_ptr() as *mut cpumask) as *const cpumask, core::ptr::addr_of_mut!(__cpu_active_mask) as *const cpumask);
let __ret_warn_on: i32 = !(!(!cpumask_subset(((*tmp).new_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask, ((*parent).effective_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("715: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 715b - ., 3; .popsection\n# {0}", const 715 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_update_parent_effective_cpumask_43, sym __slate_asm_string_update_parent_effective_cpumask_44, const 1944 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("716: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 716b - ., 4; .popsection\n# {0}", const 716 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
if *__slate_slot_29 {
if *__slate_slot_36 == 0 {
__v8 = true;
} else {
__v8 = !cpumask_intersects(((*tmp).addmask.as_mut_ptr() as *mut cpumask) as *const cpumask, core::ptr::addr_of_mut!(__cpu_active_mask) as *const cpumask);
}
__v9 = __v8;
} else {
__v9 = false;
}
if __v9 {
part_error = 5;
*__slate_slot_35 = 0;
*__slate_slot_36 = cpumask_and((*tmp).addmask.as_mut_ptr() as *mut cpumask, ((*cs).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask, ((*parent).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask) as i32;
}
}
} else {
// partcmd_update w/o newmask
//
// delmask = effective_xcpus & parent->effective_cpus
//
// This can be called from:
// 1) update_cpumasks_hier()
// 2) cpuset_hotplug_update_tasks()
//
// Check to see if it can be transitioned from valid to
// invalid partition or vice versa.
//
// A partition error happens when parent has tasks and all
// its effective CPUs will have to be distributed out.
if *__slate_slot_29 {
part_error = 5;
if is_partition_valid(cs as *const cpuset) {
*__slate_slot_36 = cpumask_and((*tmp).addmask.as_mut_ptr() as *mut cpumask, ((*cs).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask, ((*parent).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask) as i32;
}
} else {
if is_partition_invalid(cs as *const cpuset) {
__v23 = !cpumask_empty(xcpus as *const cpumask);
} else {
__v23 = false;
}
if __v23 {
__v22 = cpumask_subset(xcpus as *const cpumask, ((*parent).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
} else {
__v22 = false;
}
if __v22 {
let __slate_auto_bytes_5 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
exclusive = true;
// Convert invalid partition to valid has to
// pass the cpu exclusivity test.
rcu_read_lock();
*__slate_slot_21 = css_next_child(core::ptr::null_mut::<cgroup_subsys_state>(), core::ptr::addr_of_mut!((*parent).css));
'__join_87: {
loop {
if !(*__slate_slot_21).is_null() {
let __v18: *mut cpuset = css_cs(*__slate_slot_21);
*__slate_slot_20 = __v18;
if is_cpuset_online(__v18) {
if *__slate_slot_20 != cs {
if !cpusets_are_exclusive(cs, *__slate_slot_20) {
break;
}
}
}
*__slate_slot_21 = css_next_child(*__slate_slot_21, core::ptr::addr_of_mut!((*parent).css));
} else {
break '__join_87;
}
}
exclusive = false;
}
rcu_read_unlock();
if exclusive {
*__slate_slot_35 = cpumask_and((*tmp).delmask.as_mut_ptr() as *mut cpumask, xcpus as *const cpumask, ((*parent).effective_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask) as i32;
} else {
part_error = 4;
}
}
}
}
}
}
}
}
}
if part_error != 0 {
loop {
if true {
__WRITE_ONCE__14!((*cs).prs_err, part_error);
if true {
break;
}
}
}
}
'__join_17: {
if cmd == 3 {
'__join_21: {
// Check for possible transition between valid and invalid
// partition root.
let __t0: i32 = (*cs).partition_root_state;
if __t0 == 1 {
break '__join_21;
} else {
if __t0 == 2 {
break '__join_21;
} else {
if __t0 != -1 {
if __t0 != -2 {
break '__join_17;
}
}
}
}
if part_error == 0 {
*__slate_slot_33 = (*__slate_slot_34).wrapping_neg();
break '__join_17;
} else {
break '__join_17;
}
}
if part_error != 0 {
*__slate_slot_33 = (*__slate_slot_34).wrapping_neg();
}
}
}
if *__slate_slot_36 == 0 && *__slate_slot_35 == 0 && *__slate_slot_33 == *__slate_slot_34 {
return 0;
} else {
// Transitioning between invalid to valid or vice versa may require
// changing CS_CPU_EXCLUSIVE. In the case of partcmd_update,
// validate_change() has already been successfully called and
// CPU lists in cs haven't been updated yet. So defer it to later.
if *__slate_slot_34 != *__slate_slot_33 && cmd != 3 {
err = update_partition_exclusive_flag(cs, *__slate_slot_33);
if err != 0 {
return err;
}
}
// Change the parent's effective_cpus & effective_xcpus (top cpuset
// only).
//
// Newly added CPUs will be removed from effective_cpus and
// newly deleted ones will be added back to effective_cpus.
spin_lock_irq(core::ptr::addr_of_mut!(callback_lock));
if *__slate_slot_34 != *__slate_slot_33 {
(*cs).partition_root_state = *__slate_slot_33;
}
// Adding to parent's effective_cpus means deletion CPUs from cs
// and vice versa.
if *__slate_slot_36 != 0 {
partition_xcpus_del(*__slate_slot_34, parent, (*tmp).addmask.as_mut_ptr() as *mut cpumask);
}
if *__slate_slot_35 != 0 {
partition_xcpus_add(*__slate_slot_33, parent, (*tmp).delmask.as_mut_ptr() as *mut cpumask);
}
spin_unlock_irq(core::ptr::addr_of_mut!(callback_lock));
if *__slate_slot_34 != *__slate_slot_33 && cmd == 3 {
update_partition_exclusive_flag(cs, *__slate_slot_33);
}
if *__slate_slot_36 != 0 || *__slate_slot_35 != 0 {
cpuset_update_tasks_cpumask(parent, (*tmp).addmask.as_mut_ptr() as *mut cpumask);
update_sibling_cpumasks(parent, cs, tmp);
}
// For partcmd_update without newmask, it is being called from
// cpuset_handle_hotplug(). Update the load balance flag and
// scheduling domain accordingly.
if cmd == 3 && newmask.is_null() {
update_partition_sd_lb(cs, *__slate_slot_34);
}
notify_partition_change(cs, *__slate_slot_34);
return 0;
}
}
}

// compute_partition_effective_cpumask - compute effective_cpus for partition
// @cs: partition root cpuset
// @new_ecpus: previously computed effective_cpus to be updated
//
// Compute the effective_cpus of a partition root by scanning effective_xcpus
// of child partition roots and excluding their effective_xcpus.
//
// This has the side effect of invalidating valid child partition roots,
// if necessary. Since it is called from either cpuset_hotplug_update_tasks()
// or update_cpumasks_hier() where parent and children are modified
// successively, we don't need to call update_parent_effective_cpumask()
// and the child's effective_cpus will be updated in later iterations.
//
// Note that rcu_read_lock() is assumed to be held.
fn compute_partition_effective_cpumask(cs: *mut cpuset, new_ecpus: *mut cpumask) {
let mut css: *mut cgroup_subsys_state = unsafe { core::mem::zeroed() };
let mut child: *mut cpuset = unsafe { core::mem::zeroed() };
let populated: bool = partition_is_populated(cs, core::ptr::null_mut::<cpuset>());
// Check child partition roots to see if they should be
// invalidated when
//  1) child effective_xcpus not a subset of new
//     excluisve_cpus
//  2) All the effective_cpus will be used up and cp
//     has tasks
compute_excpus(cs, new_ecpus);
cpumask_and(new_ecpus, new_ecpus as *const cpumask, (unsafe { core::ptr::addr_of_mut!(__cpu_active_mask) }) as *const cpumask);
rcu_read_lock();
css = unsafe { css_next_child(core::ptr::null_mut::<cgroup_subsys_state>(), core::ptr::addr_of_mut!((*cs).css)) };
while !css.is_null() {
let __v0: *mut cpuset = css_cs(css);
child = __v0;
if is_cpuset_online(__v0) {
if is_partition_valid(child as *const cpuset) {
// There shouldn't be a remote partition underneath another
// partition root.
let mut __v2: i64;
let __ret_warn_on: i32 = !(!is_remote_partition(child)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("718: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 718b - ., 3; .popsection\n# {0}", const 718 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_compute_partition_effective_cpumask_3, sym __slate_asm_string_compute_partition_effective_cpumask_4, const 2132 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("719: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 719b - ., 4; .popsection\n# {0}", const 719 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
loop {
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_720() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__14!((*child).prs_err, 0i32);
if true {
break;
}
}
if !cpumask_subset((unsafe { (*child).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { (*cs).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask) {
loop {
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_721() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__14!((*child).prs_err, 1i32);
if true {
break;
}
}
} else {
let __v13: bool = populated && cpumask_subset(new_ecpus as *const cpumask, (unsafe { (*child).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
if __v13 {
loop {
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_722() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__14!((*child).prs_err, 5i32);
if true {
break;
}
}
}
}
if ((unsafe { (*child).prs_err }) as u32) != 0 {
let old_prs: i32 = unsafe { (*child).partition_root_state };
// Invalidate child partition
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
make_partition_invalid(child);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
notify_partition_change(child, old_prs);
} else {
cpumask_andnot(new_ecpus, new_ecpus as *const cpumask, (unsafe { (*child).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
}
}
}
css = unsafe { css_next_child(css, core::ptr::addr_of_mut!((*cs).css)) };
}
rcu_read_unlock();
}

// update_cpumasks_hier - Update effective cpumasks and tasks in the subtree
// @cs:  the cpuset to consider
// @tmp: temp variables for calculating effective_cpus & partition setup
// @force: don't skip any descendant cpusets if set
//
// When configured cpumask is changed, the effective cpumasks of this cpuset
// and all its descendants need to be updated.
//
// On legacy hierarchy, effective_cpus will be the same with cpu_allowed.
//
// Called with cpuset_mutex held
fn update_cpumasks_hier(cs: *mut cpuset, tmp: *mut tmpmasks, force: bool) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
// On legacy hierarchy, if the effective cpumask of any non-
// empty cpuset is changed, we need to rebuild sched domains.
// On default hierarchy, the cpuset needs to be a partition
// root as well.
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
// On default hierarchy, inherit the CS_SCHED_LOAD_BALANCE
// from parent if current cpuset isn't a valid partition root
// and their load balance states differ.
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut __v6: bool = unsafe { core::mem::zeroed() };
// Need to compute effective_xcpus if either exclusive_cpus
// is non-empty or it is a valid partition root.
let mut __v9: bool = unsafe { core::mem::zeroed() };
let mut __v10: bool = unsafe { core::mem::zeroed() };
let mut __v11: bool = unsafe { core::mem::zeroed() };
// Skip the whole subtree if
// 1) the cpumask remains the same,
// 2) has no partition root state,
// 3) force flag not set, and
// 4) for v2 load balance state same as its parent.
let mut __v12: bool = unsafe { core::mem::zeroed() };
// A partition with no effective_cpus is allowed as long as
// there is no task associated with it. Call
// update_parent_effective_cpumask() to check it.
let mut __v13: bool = unsafe { core::mem::zeroed() };
let mut __v14: bool = unsafe { core::mem::zeroed() };
let mut __v15: bool = unsafe { core::mem::zeroed() };
let mut update_parent: bool = unsafe { core::mem::zeroed() };
let mut remote: bool = unsafe { core::mem::zeroed() };
let mut parent: *mut cpuset = unsafe { core::mem::zeroed() };
let mut new_prs: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut i32 = core::ptr::addr_of_mut!(new_prs) as *mut i32;
let mut old_prs: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut i32 = core::ptr::addr_of_mut!(old_prs) as *mut i32;
let mut pos_css: core::mem::MaybeUninit<*mut cgroup_subsys_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut *mut cgroup_subsys_state = core::ptr::addr_of_mut!(pos_css) as *mut *mut cgroup_subsys_state;
let mut cp: core::mem::MaybeUninit<*mut cpuset> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut *mut cpuset = core::ptr::addr_of_mut!(cp) as *mut *mut cpuset;
unsafe {
let __slate_auto_bytes = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
rcu_read_lock();
*__slate_slot_23 = css_next_descendant_pre(core::ptr::null_mut::<cgroup_subsys_state>(), core::ptr::addr_of_mut!((*cs).css));
while !(*__slate_slot_23).is_null() {
'__join_2: {
let __v20: *mut cpuset = css_cs(*__slate_slot_23);
*__slate_slot_24 = __v20;
if is_cpuset_online(__v20) {
parent = parent_cs(*__slate_slot_24);
remote = is_remote_partition(*__slate_slot_24);
update_parent = false;
let __v16: i32 = (*(*__slate_slot_24)).partition_root_state;
*__slate_slot_21 = __v16;
*__slate_slot_22 = __v16;
// For child remote partition root (!= cs), we need to call
// remote_cpus_update() if effective_xcpus will be changed.
// Otherwise, we can skip the whole subtree.
//
// remote_cpus_update() will reuse tmp->new_cpus only after
// its value is being processed.
if remote && *__slate_slot_24 != cs {
compute_excpus(*__slate_slot_24, (*tmp).new_cpus.as_mut_ptr() as *mut cpumask);
if cpumask_equal(((*(*__slate_slot_24)).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask, ((*tmp).new_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask) {
*__slate_slot_23 = css_rightmost_descendant(*__slate_slot_23);
break '__join_2;
} else {
rcu_read_unlock();
remote_cpus_update(*__slate_slot_24, core::ptr::null_mut::<cpumask>(), (*tmp).new_cpus.as_mut_ptr() as *mut cpumask, tmp);
rcu_read_lock();
// Remote partition may be invalidated
*__slate_slot_21 = (*(*__slate_slot_24)).partition_root_state;
remote = *__slate_slot_21 == *__slate_slot_22;
}
}
if remote {
__v15 = true;
} else {
if is_partition_valid(parent as *const cpuset) {
__v14 = is_partition_valid(*__slate_slot_24 as *const cpuset);
} else {
__v14 = false;
}
__v15 = __v14;
}
if __v15 {
compute_partition_effective_cpumask(*__slate_slot_24, (*tmp).new_cpus.as_mut_ptr() as *mut cpumask);
} else {
compute_effective_cpumask((*tmp).new_cpus.as_mut_ptr() as *mut cpumask, *__slate_slot_24, parent);
}
if !remote {
// Ready to update cpuset data
if is_partition_valid(*__slate_slot_24 as *const cpuset) {
__v13 = cpumask_empty(((*tmp).new_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
} else {
__v13 = false;
}
if __v13 {
update_parent = true;
} else {
if (*(*__slate_slot_24)).partition_root_state == 0 && !force {
__v12 = cpumask_equal(((*tmp).new_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask, ((*(*__slate_slot_24)).effective_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
} else {
__v12 = false;
}
if __v12 {
if !cpuset_v2() {
__v10 = true;
} else {
__v10 = is_sched_load_balance(parent as *const cpuset) == is_sched_load_balance(*__slate_slot_24 as *const cpuset);
}
__v11 = __v10;
} else {
__v11 = false;
}
if __v11 {
*__slate_slot_23 = css_rightmost_descendant(*__slate_slot_23);
break '__join_2;
}
}
'__join_42: {
if *__slate_slot_24 != cs && *__slate_slot_22 != 0 {
let __t0: i32 = (*parent).partition_root_state;
if __t0 != 1 {
if __t0 != 2 {
if is_partition_valid(*__slate_slot_24 as *const cpuset) {
*__slate_slot_21 = unsafe { (*(*__slate_slot_24)).partition_root_state }.wrapping_neg();
}
// When parent is not a partition root or is
// invalid, child partition roots become
// invalid too.
loop {
if true {
__WRITE_ONCE__14!((*(*__slate_slot_24)).prs_err, if is_partition_invalid(parent as *const cpuset) { 2i32 } else { 3i32 });
if true {
break '__join_42;
}
}
}
}
}
update_parent = true;
}
}
// update_parent_effective_cpumask() should have been called
// for cs already in update_cpumask(). We should also call
// cpuset_update_tasks_cpumask() again for tasks in the parent
// cpuset if the parent's effective_cpus changes.
}
if css_tryget_online(core::ptr::addr_of_mut!((*(*__slate_slot_24)).css)) {
rcu_read_unlock();
if update_parent {
update_parent_effective_cpumask(*__slate_slot_24, 3i32, core::ptr::null_mut::<cpumask>(), tmp);
// The cpuset partition_root_state may become
// invalid. Capture it.
*__slate_slot_21 = (*(*__slate_slot_24)).partition_root_state;
}
spin_lock_irq(core::ptr::addr_of_mut!(callback_lock));
cpumask_copy((*(*__slate_slot_24)).effective_cpus.as_mut_ptr() as *mut cpumask, ((*tmp).new_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
(*(*__slate_slot_24)).partition_root_state = *__slate_slot_21;
if *__slate_slot_21 > 0 {
__v9 = true;
} else {
__v9 = !cpumask_empty(((*(*__slate_slot_24)).exclusive_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
}
if __v9 {
compute_excpus(*__slate_slot_24, (*(*__slate_slot_24)).effective_xcpus.as_mut_ptr() as *mut cpumask);
}
if *__slate_slot_21 <= 0 {
reset_partition_data(*__slate_slot_24);
}
spin_unlock_irq(core::ptr::addr_of_mut!(callback_lock));
notify_partition_change(*__slate_slot_24, *__slate_slot_22);
if !is_in_v2_mode() {
__v6 = !cpumask_equal(((*(*__slate_slot_24)).cpus_allowed.as_mut_ptr() as *mut cpumask) as *const cpumask, ((*(*__slate_slot_24)).effective_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
} else {
__v6 = false;
}
let __ret_warn_on: i32 = __v6 as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("724: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 724b - ., 3; .popsection\n# {0}", const 724 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_update_cpumasks_hier_25, sym __slate_asm_string_update_cpumasks_hier_26, const 2301 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("725: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 725b - ., 4; .popsection\n# {0}", const 725 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
cpuset_update_tasks_cpumask(*__slate_slot_24, (*tmp).new_cpus.as_mut_ptr() as *mut cpumask);
if cpuset_v2() {
__v4 = !is_partition_valid(*__slate_slot_24 as *const cpuset);
} else {
__v4 = false;
}
if __v4 {
__v3 = is_sched_load_balance(parent as *const cpuset) != is_sched_load_balance(*__slate_slot_24 as *const cpuset);
} else {
__v3 = false;
}
if __v3 {
if is_sched_load_balance(parent as *const cpuset) != 0 {
set_bit(4i64, core::ptr::addr_of_mut!((*(*__slate_slot_24)).flags) as *mut u64);
} else {
clear_bit(4i64, core::ptr::addr_of_mut!((*(*__slate_slot_24)).flags) as *mut u64);
}
}
if !cpumask_empty(((*(*__slate_slot_24)).cpus_allowed.as_mut_ptr() as *mut cpumask) as *const cpumask) {
__v2 = is_sched_load_balance(*__slate_slot_24 as *const cpuset) != 0;
} else {
__v2 = false;
}
if __v2 {
if !cpuset_v2() {
__v0 = true;
} else {
__v0 = is_partition_valid(*__slate_slot_24 as *const cpuset);
}
__v1 = __v0;
} else {
__v1 = false;
}
if __v1 {
cpuset_force_rebuild();
}
rcu_read_lock();
css_put(core::ptr::addr_of_mut!((*(*__slate_slot_24)).css));
}
}
}
*__slate_slot_23 = css_next_descendant_pre(*__slate_slot_23, core::ptr::addr_of_mut!((*cs).css));
}
rcu_read_unlock();
}
}

/// update_sibling_cpumasks - Update siblings cpumasks
/// @parent:  Parent cpuset
/// @cs:      Current cpuset
/// @tmp:     Temp variables
fn update_sibling_cpumasks(parent: *mut cpuset, cs: *mut cpuset, tmp: *mut tmpmasks) {
let mut sibling: *mut cpuset = unsafe { core::mem::zeroed() };
let mut pos_css: *mut cgroup_subsys_state = unsafe { core::mem::zeroed() };
lockdep_assert_cpuset_lock_held();
// Check all its siblings and call update_cpumasks_hier()
// if their effective_cpus will need to be changed.
//
// It is possible a change in parent's effective_cpus
// due to a change in a child partition's effective_xcpus will impact
// its siblings even if they do not inherit parent's effective_cpus
// directly. It should not impact valid partition.
//
// The update_cpumasks_hier() function may sleep. So we have to
// release the RCU read lock before calling it.
rcu_read_lock();
pos_css = unsafe { css_next_child(core::ptr::null_mut::<cgroup_subsys_state>(), core::ptr::addr_of_mut!((*parent).css)) };
while !pos_css.is_null() {
let __v0: *mut cpuset = css_cs(pos_css);
sibling = __v0;
if is_cpuset_online(__v0) {
let __v1: bool = sibling == cs || is_partition_valid(sibling as *const cpuset);
if !__v1 {
compute_effective_cpumask(unsafe { (*tmp).new_cpus.as_mut_ptr() as *mut cpumask }, sibling, parent);
if !cpumask_equal((unsafe { (*tmp).new_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { (*sibling).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask) {
if css_tryget_online(unsafe { core::ptr::addr_of_mut!((*sibling).css) }) {
rcu_read_unlock();
update_cpumasks_hier(sibling, tmp, false);
rcu_read_lock();
css_put(unsafe { core::ptr::addr_of_mut!((*sibling).css) });
}
}
}
}
pos_css = unsafe { css_next_child(pos_css, core::ptr::addr_of_mut!((*parent).css)) };
}
rcu_read_unlock();
}

fn parse_cpuset_cpulist(buf: *const u8, out_mask: *mut cpumask) -> i32 {
let retval: i32 = cpulist_parse(buf, out_mask);
if retval < 0 {
return retval;
}
if !cpumask_subset(out_mask as *const cpumask, (unsafe { core::ptr::addr_of_mut!(top_cpuset.cpus_allowed) as *mut cpumask }) as *const cpumask) {
return 22i32.wrapping_neg();
}
0
}

/// validate_partition - Validate a cpuset partition configuration
/// @cs: The cpuset to validate
/// @trialcs: The trial cpuset containing proposed configuration changes
///
/// If any validation check fails, the appropriate error code is set in the
/// cpuset's prs_err field.
///
/// Return: PRS error code (0 if valid, non-zero error code if invalid)
fn validate_partition(cs: *mut cpuset, trialcs: *mut cpuset) -> u32 {
let parent: *mut cpuset = parent_cs(cs);
if cs_is_member(trialcs as *const cpuset) {
return 0;
}
if cpumask_empty((unsafe { (*trialcs).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask) {
return 1;
}
if prstate_housekeeping_conflict(unsafe { (*trialcs).partition_root_state }, unsafe { (*trialcs).effective_xcpus.as_mut_ptr() as *mut cpumask }) {
return 8;
}
if tasks_nocpu_error(parent, cs, unsafe { (*trialcs).effective_xcpus.as_mut_ptr() as *mut cpumask }) {
return 5;
}
0
}

// partition_cpus_change - Handle partition state changes due to CPU mask updates
// @cs: The target cpuset being modified
// @trialcs: The trial cpuset containing proposed configuration changes
// @tmp: Temporary masks for intermediate calculations
//
// This function handles partition state transitions triggered by CPU mask changes.
// CPU modifications may cause a partition to be disabled or require state updates.
fn partition_cpus_change(cs: *mut cpuset, trialcs: *mut cpuset, tmp: *mut tmpmasks) {
let mut prs_err: u32 = 0;
if cs_is_member(cs as *const cpuset) {
return;
}
prs_err = validate_partition(cs, trialcs);
if (prs_err as u32) != 0 {
loop {
while false {
}
__WRITE_ONCE__7!((*cs).prs_err, prs_err);
if true {
break;
}
}
unsafe {
(*trialcs).prs_err = prs_err;
}
}
if is_remote_partition(cs) {
if ((unsafe { (*trialcs).prs_err }) as u32) != 0 {
remote_partition_disable(cs, tmp);
} else {
remote_cpus_update(cs, unsafe { (*trialcs).exclusive_cpus.as_mut_ptr() as *mut cpumask }, unsafe { (*trialcs).effective_xcpus.as_mut_ptr() as *mut cpumask }, tmp);
}
} else {
if ((unsafe { (*trialcs).prs_err }) as u32) != 0 {
update_parent_effective_cpumask(cs, 4i32, core::ptr::null_mut::<cpumask>(), tmp);
} else {
update_parent_effective_cpumask(cs, 3i32, unsafe { (*trialcs).effective_xcpus.as_mut_ptr() as *mut cpumask }, tmp);
}
}
}

/// update_cpumask - update the cpus_allowed mask of a cpuset and all tasks in it
/// @cs: the cpuset to consider
/// @trialcs: trial cpuset
/// @buf: buffer of cpu numbers written to this cpuset
fn update_cpumask(cs: *mut cpuset, trialcs: *mut cpuset, buf: *const u8) -> i32 {
let mut retval: i32 = 0;
let mut tmp: tmpmasks = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tmp) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
let mut force: bool = false;
let old_prs: i32 = unsafe { (*cs).partition_root_state };
retval = parse_cpuset_cpulist(buf, unsafe { (*trialcs).cpus_allowed.as_mut_ptr() as *mut cpumask });
if retval < 0 {
return retval;
}
// Nothing to do if the cpus didn't change
if cpumask_equal((unsafe { (*cs).cpus_allowed.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { (*trialcs).cpus_allowed.as_mut_ptr() as *mut cpumask }) as *const cpumask) {
return 0;
}
compute_trialcs_excpus(trialcs, cs);
unsafe {
(*trialcs).prs_err = 0;
}
retval = validate_change(cs, trialcs);
if retval < 0 {
return retval;
}
if alloc_tmpmasks(unsafe { &mut (*core::ptr::addr_of_mut!(tmp)) }) != 0 {
return 12i32.wrapping_neg();
}
// Check all the descendants in update_cpumasks_hier() if
// effective_xcpus is to be changed.
force = !cpumask_equal((unsafe { (*cs).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { (*trialcs).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
partition_cpus_change(cs, trialcs, core::ptr::addr_of_mut!(tmp));
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
cpumask_copy(unsafe { (*cs).cpus_allowed.as_mut_ptr() as *mut cpumask }, (unsafe { (*trialcs).cpus_allowed.as_mut_ptr() as *mut cpumask }) as *const cpumask);
cpumask_copy(unsafe { (*cs).effective_xcpus.as_mut_ptr() as *mut cpumask }, (unsafe { (*trialcs).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
let __v0: bool = old_prs > 0 && !is_partition_valid(cs as *const cpuset);
if __v0 {
reset_partition_data(cs);
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
// effective_cpus/effective_xcpus will be updated here
update_cpumasks_hier(cs, core::ptr::addr_of_mut!(tmp), force);
// Update CS_SCHED_LOAD_BALANCE and/or sched_domains, if necessary
if (unsafe { (*cs).partition_root_state }) != 0 {
update_partition_sd_lb(cs, old_prs);
}
free_tmpmasks(core::ptr::addr_of_mut!(tmp));
retval
}

/// update_exclusive_cpumask - update the exclusive_cpus mask of a cpuset
/// @cs: the cpuset to consider
/// @trialcs: trial cpuset
/// @buf: buffer of cpu numbers written to this cpuset
///
/// The tasks' cpumask will be updated if cs is a valid partition root.
fn update_exclusive_cpumask(cs: *mut cpuset, trialcs: *mut cpuset, buf: *const u8) -> i32 {
let mut retval: i32 = 0;
let mut tmp: tmpmasks = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tmp) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
let mut force: bool = false;
let old_prs: i32 = unsafe { (*cs).partition_root_state };
retval = parse_cpuset_cpulist(buf, unsafe { (*trialcs).exclusive_cpus.as_mut_ptr() as *mut cpumask });
if retval < 0 {
return retval;
}
// Nothing to do if the CPUs didn't change
if cpumask_equal((unsafe { (*cs).exclusive_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { (*trialcs).exclusive_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask) {
return 0;
}
// Reject the change if there is exclusive CPUs conflict with
// the siblings.
if compute_trialcs_excpus(trialcs, cs) != 0 {
return 22i32.wrapping_neg();
}
// Check all the descendants in update_cpumasks_hier() if
// effective_xcpus is to be changed.
force = !cpumask_equal((unsafe { (*cs).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { (*trialcs).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
retval = validate_change(cs, trialcs);
if retval != 0 {
return retval;
}
if alloc_tmpmasks(unsafe { &mut (*core::ptr::addr_of_mut!(tmp)) }) != 0 {
return 12i32.wrapping_neg();
}
unsafe {
(*trialcs).prs_err = 0;
}
partition_cpus_change(cs, trialcs, core::ptr::addr_of_mut!(tmp));
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
cpumask_copy(unsafe { (*cs).exclusive_cpus.as_mut_ptr() as *mut cpumask }, (unsafe { (*trialcs).exclusive_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
cpumask_copy(unsafe { (*cs).effective_xcpus.as_mut_ptr() as *mut cpumask }, (unsafe { (*trialcs).effective_xcpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
let __v0: bool = old_prs > 0 && !is_partition_valid(cs as *const cpuset);
if __v0 {
reset_partition_data(cs);
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
// Call update_cpumasks_hier() to update effective_cpus/effective_xcpus
// of the subtree when it is a valid partition root or effective_xcpus
// is updated.
if is_partition_valid(cs as *const cpuset) || force {
update_cpumasks_hier(cs, core::ptr::addr_of_mut!(tmp), force);
}
// Update CS_SCHED_LOAD_BALANCE and/or sched_domains, if necessary
if (unsafe { (*cs).partition_root_state }) != 0 {
update_partition_sd_lb(cs, old_prs);
}
free_tmpmasks(core::ptr::addr_of_mut!(tmp));
0
}

// Migrate memory region from one set of nodes to another.  This is
// performed asynchronously as it can be called from process migration path
// holding locks involved in process management.  All mm migrations are
// performed in the queued order and can be waited for by flushing
// cpuset_migrate_mm_wq.
#[repr(C)]
#[derive(Clone, Copy)]
struct cpuset_migrate_mm_work {
    work: work_struct,
    mm: *mut mm_struct,
    from: __SlateRecord_nodemask_types_h_17,
    to: __SlateRecord_nodemask_types_h_17,
}


#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_migrate_mm_workfn")]
extern "C" fn cpuset_migrate_mm_workfn(work: *mut work_struct) {
let mwork: *mut cpuset_migrate_mm_work = container_of__5!(work, cpuset_migrate_mm_work);
// on a wq worker, no need to worry about %current's mems_allowed
unsafe { do_migrate_pages((*mwork).mm, core::ptr::addr_of_mut!((*mwork).from) as *const __SlateRecord_nodemask_types_h_17, core::ptr::addr_of_mut!((*mwork).to) as *const __SlateRecord_nodemask_types_h_17, 4i32) };
unsafe { mmput((*mwork).mm) };
unsafe { kfree(mwork as *const ()) };
}

fn cpuset_migrate_mm(mm: *mut mm_struct, from: *const __SlateRecord_nodemask_types_h_17, to: *const __SlateRecord_nodemask_types_h_17) {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut mwork: *mut cpuset_migrate_mm_work = unsafe { core::mem::zeroed() };
if nodes_equal__2!(*from, *to) {
unsafe { mmput(mm) };
return;
}
let mut __v0: *mut cpuset_migrate_mm_work;
let __obj_size: u64 = size_mul(core::mem::size_of::<cpuset_migrate_mm_work>() as u64, 1u64);
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
__v0 = __v1 as *mut cpuset_migrate_mm_work;
mwork = __v0;
if !mwork.is_null() {
unsafe {
(*mwork).mm = mm;
}
unsafe {
(*mwork).from = *from;
}
unsafe {
(*mwork).to = *to;
}
'__slate_break_8: loop {
'__slate_break_7: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*mwork).work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*mwork).work)).data = *{
__slate_compound_6 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_6)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*mwork).work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*mwork).work)).func = Some(cpuset_migrate_mm_workfn);
}
if true {
break;
}
}
if true {
break;
}
}
queue_work(unsafe { cpuset_migrate_mm_wq }, unsafe { core::ptr::addr_of_mut!((*mwork).work) });
} else {
unsafe { mmput(mm) };
}
}

static mut __key: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.cpuset.flush_migrate_mm_task_workfn")]
extern "C" fn flush_migrate_mm_task_workfn(head: *mut callback_head) {
flush_workqueue!(unsafe { cpuset_migrate_mm_wq });
unsafe { kfree(head as *const ()) };
}

fn schedule_flush_migrate_mm() {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __v0: *mut callback_head;
let __obj_size: u64 = size_mul(core::mem::size_of::<callback_head>() as u64, 1u64);
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
__v0 = __v1 as *mut callback_head;
let flush_cb: *mut callback_head = __v0;
if flush_cb.is_null() {
return;
}
init_task_work(flush_cb, Some(flush_migrate_mm_task_workfn));
if (unsafe { task_work_add(get_current(), flush_cb, 1u32) }) != 0 {
unsafe { kfree(flush_cb as *const ()) };
}
}

/// cpuset_change_task_nodemask - change task's mems_allowed and mempolicy
/// @tsk: the task to change
/// @newmems: new nodes that the task will be set
///
/// We use the mems_allowed_seq seqlock to safely update both tsk->mems_allowed
/// and rebind an eventual tasks' mempolicy. If the task is allocating in
/// parallel, it might temporarily see an empty intersection, which results in
/// a seqlock check and retry before OOM or allocation failure.
fn cpuset_change_task_nodemask(tsk: *mut task_struct, newmems__0: *mut __SlateRecord_nodemask_types_h_17) {
task_lock(tsk);
local_irq_disable!();
loop {
seqprop_assert__3!(unsafe { core::ptr::addr_of_mut!((*tsk).mems_allowed_seq) });
if seqprop_preemptible__3!(unsafe { core::ptr::addr_of_mut!((*tsk).mems_allowed_seq) }) {
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
}
do_write_seqcount_begin(seqprop_ptr__4!(unsafe { core::ptr::addr_of_mut!((*tsk).mems_allowed_seq) }));
if true {
break;
}
}
nodes_or!((*tsk).mems_allowed, (*tsk).mems_allowed, *newmems__0);
unsafe { mpol_rebind_task(tsk, newmems__0 as *const __SlateRecord_nodemask_types_h_17) };
unsafe {
(*tsk).mems_allowed = *newmems__0;
}
loop {
do_write_seqcount_end(seqprop_ptr__4!(unsafe { core::ptr::addr_of_mut!((*tsk).mems_allowed_seq) }));
if seqprop_preemptible__3!(unsafe { core::ptr::addr_of_mut!((*tsk).mems_allowed_seq) }) {
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
if true {
break;
}
}
local_irq_enable!();
task_unlock(tsk);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_727: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

static mut cpuset_being_rebound: *mut () = unsafe { core::mem::zeroed() };

/// cpuset_update_tasks_nodemask - Update the nodemasks of tasks in the cpuset.
/// @cs: the cpuset in which each task's mems_allowed mask needs to be changed
///
/// Iterate through each task of @cs updating its mems_allowed to the
/// effective cpuset's.  As this function is called with cpuset_mutex held,
/// cpuset membership stays stable.
///
/// - cpuset_change_task_nodemask(): guarantee_online_mems()
/// - mpol_rebind_mm(): effective_mems
/// - cpuset_migrate_mm(): guarantee_online_mems()
/// - old_mems_allowed: guarantee_online_mems()
///
/// For v2, guarantee_online_mems() should return a node mask that is the same
/// as the effective_mems of current cpuset.
#[unsafe(no_mangle)]
extern "C" fn cpuset_update_tasks_nodemask(cs: *mut cpuset) {
// protected by cpuset_mutex
let mut it: css_task_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(it) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 104usize);
}
unsafe {
cpuset_being_rebound = cs as *mut ();
}
// causes mpol_dup() rebind
guarantee_online_mems(cs, unsafe { core::ptr::addr_of_mut!(newmems) });
// The mpol_rebind_mm() call takes mmap_lock, which we couldn't
// take while holding tasklist_lock.  Forks can happen - the
// mpol_dup() cpuset_being_rebound check will catch such forks,
// and rebind their vma mempolicies too.  Because we still hold
// the global cpuset_mutex, we know that no other rebind effort
// will be contending for the global variable cpuset_being_rebound.
// It's ok if we rebind the same mm twice; mpol_rebind_mm()
// is idempotent.  Also migrate pages in each mm to new nodes.
unsafe { css_task_iter_start(core::ptr::addr_of_mut!((*cs).css), 0u32, core::ptr::addr_of_mut!(it)) };
loop {
let __v1: *mut task_struct = unsafe { css_task_iter_next(core::ptr::addr_of_mut!(it)) };
let task: *mut task_struct = __v1;
if __v1.is_null() {
break;
}
let mut migrate: bool = false;
cpuset_change_task_nodemask(task, unsafe { core::ptr::addr_of_mut!(newmems) });
// Rebind and migrate mm only for thread group leader
if thread_group_leader(task) {
let mm: *mut mm_struct = unsafe { get_task_mm(task) };
if !mm.is_null() {
migrate = is_memory_migrate(cs as *const cpuset) != 0;
// For v1 we can have empty effective_mems, but we cannot
// attach any tasks (see cpuset_can_attach_check()). For v2,
// effective_mems is guaranteed to not be empty.
unsafe { mpol_rebind_mm(mm, core::ptr::addr_of_mut!((*cs).effective_mems)) };
if migrate {
cpuset_migrate_mm(mm, (unsafe { core::ptr::addr_of_mut!((*cs).old_mems_allowed) }) as *const __SlateRecord_nodemask_types_h_17, (unsafe { core::ptr::addr_of_mut!(newmems) }) as *const __SlateRecord_nodemask_types_h_17);
} else {
unsafe { mmput(mm) };
}
}
}
}
unsafe { css_task_iter_end(core::ptr::addr_of_mut!(it)) };
// All the tasks' nodemasks have been updated, update
// cs->old_mems_allowed.
unsafe {
(*cs).old_mems_allowed = newmems;
}
// We're done rebinding vmas to this cpuset's new mems_allowed.
unsafe {
cpuset_being_rebound = core::ptr::null_mut::<()>();
}
}

static mut newmems: __SlateRecord_nodemask_types_h_17 = unsafe { core::mem::zeroed() };

/// update_nodemasks_hier - Update effective nodemasks and tasks in the subtree
/// @cs: the cpuset to consider
/// @new_mems: a temp variable for calculating new effective_mems
///
/// When configured nodemask is changed, the effective nodemasks of this cpuset
/// and all its descendants need to be updated.
///
/// On legacy hierarchy, effective_mems will be the same with mems_allowed.
///
/// Called with cpuset_mutex held
fn update_nodemasks_hier(cs: *mut cpuset, new_mems: *mut __SlateRecord_nodemask_types_h_17) {
let mut cp: *mut cpuset = unsafe { core::mem::zeroed() };
let mut pos_css: *mut cgroup_subsys_state = unsafe { core::mem::zeroed() };
rcu_read_lock();
pos_css = unsafe { css_next_descendant_pre(core::ptr::null_mut::<cgroup_subsys_state>(), core::ptr::addr_of_mut!((*cs).css)) };
while !pos_css.is_null() {
let __v0: *mut cpuset = css_cs(pos_css);
cp = __v0;
if is_cpuset_online(__v0) {
let parent: *mut cpuset = parent_cs(cp);
compute_effective_nodemask(new_mems, cp, parent);
// Skip the whole subtree if the nodemask remains the same.
if nodes_equal!(*new_mems, (*cp).effective_mems) {
pos_css = unsafe { css_rightmost_descendant(pos_css) };
} else {
if css_tryget_online(unsafe { core::ptr::addr_of_mut!((*cp).css) }) {
rcu_read_unlock();
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
unsafe {
(*cp).effective_mems = *new_mems;
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
let mut __v2: i64;
let mut __ret_warn_on: i32 = 0;
let __v3: bool = !is_in_v2_mode() && !nodes_equal!((*cp).mems_allowed, (*cp).effective_mems);
__ret_warn_on = __v3 as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("728: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 728b - ., 3; .popsection\n# {0}", const 728 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_update_nodemasks_hier_4, sym __slate_asm_string_update_nodemasks_hier_5, const 2803 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("729: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 729b - ., 4; .popsection\n# {0}", const 729 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
cpuset_update_tasks_nodemask(cp);
rcu_read_lock();
css_put(unsafe { core::ptr::addr_of_mut!((*cp).css) });
}
}
}
pos_css = unsafe { css_next_descendant_pre(pos_css, core::ptr::addr_of_mut!((*cs).css)) };
}
rcu_read_unlock();
}

/// Handle user request to change the 'mems' memory placement
/// of a cpuset.  Needs to validate the request, update the
/// cpusets mems_allowed, and for each task in the cpuset,
/// update mems_allowed and rebind task's mempolicy and any vma
/// mempolicies and if the cpuset is marked 'memory_migrate',
/// migrate the tasks pages to the new memory.
///
/// Call with cpuset_mutex held. May take callback_lock during call.
/// Will take tasklist_lock, scan tasklist for tasks in cpuset cs,
/// lock each such tasks mm->mmap_lock, scan its vma's and rebind
/// their mempolicies to the cpusets new mems_allowed.
fn update_nodemask(cs: *mut cpuset, trialcs: *mut cpuset, buf: *const u8) -> i32 {
let mut retval: i32 = 0;
// An empty mems_allowed is ok iff there are no tasks in the cpuset.
// The validate_change() call ensures that cpusets with tasks have memory.
retval = nodelist_parse!(buf, (*trialcs).mems_allowed);
if retval < 0 {
return retval;
}
if !nodes_subset!((*trialcs).mems_allowed, top_cpuset.mems_allowed) {
return 22i32.wrapping_neg();
}
// No change? nothing to do
if nodes_equal!((*cs).mems_allowed, (*trialcs).mems_allowed) {
return 0;
}
retval = validate_change(cs, trialcs);
if retval < 0 {
return retval;
}
check_insane_mems_config(unsafe { core::ptr::addr_of_mut!((*trialcs).mems_allowed) });
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
unsafe {
(*cs).mems_allowed = (*trialcs).mems_allowed;
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
// use trialcs->mems_allowed as a temp variable
update_nodemasks_hier(cs, unsafe { core::ptr::addr_of_mut!((*trialcs).mems_allowed) });
0
}

#[unsafe(no_mangle)]
extern "C" fn current_cpuset_is_being_rebound() -> bool {
let mut ret: bool = false;
rcu_read_lock();
ret = task_cs(get_current()) == ((unsafe { cpuset_being_rebound }) as *mut cpuset);
rcu_read_unlock();
ret
}

// cpuset_update_flag - read a 0 or a 1 in a file and update associated flag
// bit:		the bit to update (see cpuset_flagbits_t)
// cs:		the cpuset to update
// turning_on: 	whether the flag is being set or cleared
//
// Call with cpuset_mutex held.
#[unsafe(no_mangle)]
extern "C" fn cpuset_update_flag(bit: u32, cs: *mut cpuset, turning_on: i32) -> i32 {
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut spread_page_changed: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(spread_page_changed) as *mut i32;
let mut balance_flag_changed: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(balance_flag_changed) as *mut i32;
let mut trialcs: core::mem::MaybeUninit<*mut cpuset> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut cpuset = core::ptr::addr_of_mut!(trialcs) as *mut *mut cpuset;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_3 = dup_or_alloc_cpuset(cs);
if (*__slate_slot_3).is_null() {
return 12i32.wrapping_neg();
} else {
if turning_on != 0 {
set_bit((bit as u32) as i64, core::ptr::addr_of_mut!((*(*__slate_slot_3)).flags) as *mut u64);
} else {
clear_bit((bit as u32) as i64, core::ptr::addr_of_mut!((*(*__slate_slot_3)).flags) as *mut u64);
}
*__slate_slot_0 = validate_change(cs, *__slate_slot_3);
if *__slate_slot_0 >= 0 {
*__slate_slot_2 = (is_sched_load_balance(cs as *const cpuset) != is_sched_load_balance(*__slate_slot_3 as *const cpuset)) as i32;
*__slate_slot_1 = (is_spread_page(cs as *const cpuset) != is_spread_page(*__slate_slot_3 as *const cpuset)) as i32;
spin_lock_irq(core::ptr::addr_of_mut!(callback_lock));
(*cs).flags = (*(*__slate_slot_3)).flags;
spin_unlock_irq(core::ptr::addr_of_mut!(callback_lock));
if !cpumask_empty(((*(*__slate_slot_3)).cpus_allowed.as_mut_ptr() as *mut cpumask) as *const cpumask) && *__slate_slot_2 != 0 {
if cpuset_v2() {
cpuset_force_rebuild();
} else {
rebuild_sched_domains_locked();
}
}
if *__slate_slot_1 != 0 {
cpuset1_update_tasks_flags(cs);
}
}
free_cpuset(*__slate_slot_3);
return *__slate_slot_0;
}
}
}

// update_prstate - update partition_root_state
// @cs: the cpuset to update
// @new_prs: new partition root state
// Return: 0 if successful, != 0 if error
//
// Call with cpuset_mutex held.
fn update_prstate(cs: *mut cpuset, mut new_prs: i32) -> i32 {
let mut __v1: bool = unsafe { core::mem::zeroed() };
// We don't support the creation of a new local partition with
// a remote partition underneath it. This unsupported
// setting can happen only if parent is the top_cpuset because
// a remote partition cannot be created underneath an existing
// local or remote partition.
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut __v6: bool = unsafe { core::mem::zeroed() };
// A change in load balance state only, no change in cpumasks.
// Need to update isolated_cpus.
let mut __v7: bool = unsafe { core::mem::zeroed() };
// Treat a previously invalid partition root as if it is a "member".
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut isolcpus_updated: bool = unsafe { core::mem::zeroed() };
let mut tmpmask: core::mem::MaybeUninit<tmpmasks> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut tmpmasks = core::ptr::addr_of_mut!(tmpmask) as *mut tmpmasks;
let mut parent: *mut cpuset = unsafe { core::mem::zeroed() };
let mut old_prs: i32 = unsafe { core::mem::zeroed() };
let mut err: i32 = unsafe { core::mem::zeroed() };
unsafe {
err = 0;
old_prs = (*cs).partition_root_state;
parent = parent_cs(cs);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
isolcpus_updated = false;
if old_prs == new_prs {
return 0;
} else {
if new_prs != 0 {
__v8 = is_partition_invalid(cs as *const cpuset);
} else {
__v8 = false;
}
if __v8 {
old_prs = 0;
}
if alloc_tmpmasks(&mut (*core::ptr::addr_of_mut!(*__slate_slot_10))) != 0 {
return 12i32.wrapping_neg();
} else {
err = update_partition_exclusive_flag(cs, new_prs);
if err == 0 {
if old_prs == 0 {
// cpus_allowed and exclusive_cpus cannot be both empty.
if xcpus_empty(cs) {
err = 7;
} else {
if parent == core::ptr::addr_of_mut!(top_cpuset) {
__v5 = cpumask_intersects(((*cs).exclusive_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask, (core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask) as *const cpumask);
} else {
__v5 = false;
}
if __v5 {
err = 10;
} else {
// If parent is valid partition, enable local partiion.
// Otherwise, enable a remote partition.
if is_partition_valid(parent as *const cpuset) {
let cmd: u32 = (if new_prs == 1 { 0i32 } else { 1i32 }) as u32;
err = update_parent_effective_cpumask(cs, (cmd as u32) as i32, core::ptr::null_mut::<cpumask>(), core::ptr::addr_of_mut!(*__slate_slot_10));
} else {
err = remote_partition_enable(cs, new_prs, core::ptr::addr_of_mut!(*__slate_slot_10));
}
}
}
} else {
if old_prs != 0 && new_prs != 0 {
if new_prs == 2 {
__v7 = !isolated_cpus_can_update((*cs).effective_xcpus.as_mut_ptr() as *mut cpumask, core::ptr::null_mut::<cpumask>());
} else {
__v7 = false;
}
if __v7 {
__v6 = true;
} else {
__v6 = prstate_housekeeping_conflict(new_prs, (*cs).effective_xcpus.as_mut_ptr() as *mut cpumask);
}
if __v6 {
err = 8;
} else {
isolcpus_updated = true;
}
} else {
// Switching back to member is always allowed even if it
// disables child partitions.
if is_remote_partition(cs) {
remote_partition_disable(cs, core::ptr::addr_of_mut!(*__slate_slot_10));
} else {
update_parent_effective_cpumask(cs, 2i32, core::ptr::null_mut::<cpumask>(), core::ptr::addr_of_mut!(*__slate_slot_10));
}
// Invalidation of child partitions will be done in
// update_cpumasks_hier().
}
}
}
if err != 0 {
new_prs = new_prs.wrapping_neg();
update_partition_exclusive_flag(cs, new_prs);
}
// Make partition invalid & disable CS_CPU_EXCLUSIVE if an error
// happens.
spin_lock_irq(core::ptr::addr_of_mut!(callback_lock));
(*cs).partition_root_state = new_prs;
loop {
if true {
__WRITE_ONCE__14!((*cs).prs_err, err);
if true {
break;
}
}
}
if !is_partition_valid(cs as *const cpuset) {
reset_partition_data(cs);
} else {
if isolcpus_updated {
isolated_cpus_update(old_prs, new_prs, (*cs).effective_xcpus.as_mut_ptr() as *mut cpumask);
}
}
spin_unlock_irq(core::ptr::addr_of_mut!(callback_lock));
// Force update if switching back to member & update effective_xcpus
update_cpumasks_hier(cs, core::ptr::addr_of_mut!(*__slate_slot_10), new_prs == 0);
if old_prs == 0 && new_prs > 0 {
__v1 = cpumask_empty(((*cs).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
} else {
__v1 = false;
}
let __ret_warn_on: i32 = __v1 as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("731: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 731b - ., 3; .popsection\n# {0}", const 731 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_update_prstate_14, sym __slate_asm_string_update_prstate_15, const 3042 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("732: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 732b - ., 4; .popsection\n# {0}", const 732 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// A newly created partition must have effective_xcpus set
// Update sched domains and load balance flag
update_partition_sd_lb(cs, old_prs);
notify_partition_change(cs, old_prs);
if force_sd_rebuild {
rebuild_sched_domains_locked();
}
free_tmpmasks(core::ptr::addr_of_mut!(*__slate_slot_10));
return 0;
}
}
}
}

/// Check to see if a cpuset can accept a new task
/// For v1, cpus_allowed and mems_allowed can't be empty.
/// For v2, effective_cpus can't be empty.
/// Note that in v1, effective_cpus = cpus_allowed.
///
/// Also set the boolean flag passed in by @psetsched depending on if
/// security_task_setscheduler() call is needed and @oldcs is not NULL.
fn cpuset_can_attach_check(cs: *mut cpuset, oldcs: *mut cpuset, psetsched: *mut bool) -> i32 {
let mut cpus_updated: bool = false;
let mut mems_updated: bool = false;
let __v0: bool = cpumask_empty((unsafe { (*cs).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask) || !is_in_v2_mode() && nodes_empty!((*cs).mems_allowed);
if __v0 {
return 28i32.wrapping_neg();
}
if oldcs.is_null() {
return 0;
}
if !llist_on_list((unsafe { core::ptr::addr_of_mut!((*oldcs).attach_node) }) as *const llist_node) {
llist_add(unsafe { core::ptr::addr_of_mut!((*oldcs).attach_node) }, unsafe { core::ptr::addr_of_mut!(src_cs_head) });
}
if !llist_on_list((unsafe { core::ptr::addr_of_mut!((*cs).attach_node) }) as *const llist_node) {
llist_add(unsafe { core::ptr::addr_of_mut!((*cs).attach_node) }, unsafe { core::ptr::addr_of_mut!(dst_cs_head) });
}
cpus_updated = !cpumask_equal((unsafe { (*cs).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { (*oldcs).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
mems_updated = !nodes_equal!((*cs).effective_mems, (*oldcs).effective_mems);
if cpus_updated {
unsafe {
attach_ctx.cpus_updated = true;
}
}
if mems_updated {
unsafe {
attach_ctx.mems_updated = true;
}
}
// Skip rights over task setsched check in v2 when nothing changes for
// the current oldcs/cs pair, migration permission derives from
// hierarchy ownership in cgroup_procs_write_permission()).
unsafe {
*psetsched = !cpuset_v2() || cpus_updated || mems_updated;
}
// A v1 cpuset with tasks will have no CPU left only when CPU hotplug
// brings the last online CPU offline as users are not allowed to empty
// cpuset.cpus when there are active tasks inside. When that happens,
// we should allow tasks to migrate out without security check to make
// sure they will be able to run after migration.
let __v2: bool = !is_in_v2_mode() && cpumask_empty((unsafe { (*oldcs).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
if __v2 {
unsafe {
*psetsched = false;
}
}
0
}

fn cpuset_reserve_dl_bw() -> i32 {
let mut cs: *mut cpuset = unsafe { core::mem::zeroed() };
let mut cpu: i32 = 0;
let mut ret: i32 = 0;
cs = llist_entry!(unsafe { dst_cs_head.first });
while member_address_is_nonnull!(cs) {
if (unsafe { (*cs).sum_migrate_dl_bw }) != 0 {
cpu = cpumask_any_and__4!((unsafe { core::ptr::addr_of_mut!(__cpu_active_mask) }) as *const cpumask, unsafe { (*cs).effective_cpus.as_mut_ptr() as *mut cpumask }) as i32;
if unlikely__2!((cpu as u32) >= unsafe { nr_cpu_ids }) != 0 {
return 22i32.wrapping_neg();
}
ret = unsafe { dl_bw_alloc(cpu, (*cs).sum_migrate_dl_bw) };
if ret != 0 {
return ret;
}
unsafe {
(*cs).dl_bw_cpu = cpu;
}
}
cs = llist_entry!(unsafe { (*cs).attach_node.next });
}
0
}

/// Clear and optionally apply (@cancel is false) the attach related data in the
/// source or destination cpuset.
fn clear_attach_data(head: *mut llist_head, cancel: bool) {
let mut cs: *mut cpuset = unsafe { core::mem::zeroed() };
let mut next: *mut cpuset = unsafe { core::mem::zeroed() };
let lnode: *mut llist_node = __llist_del_all(head);
cs = llist_entry!(lnode);
loop {
let __v1: bool = if member_address_is_nonnull!(cs) {
next = llist_entry!(unsafe { (*cs).attach_node.next });
true
} else {
false
};
if !__v1 {
break;
}
init_llist_node(unsafe { core::ptr::addr_of_mut!((*cs).attach_node) });
if (unsafe { (*cs).nr_migrate_dl_tasks }) != 0 {
if !cancel {
atomic_add(unsafe { (*cs).nr_migrate_dl_tasks }, unsafe { core::ptr::addr_of_mut!((*cs).nr_deadline_tasks) });
} else {
if (unsafe { (*cs).dl_bw_cpu }) >= 0 {
unsafe { dl_bw_free((*cs).dl_bw_cpu, (*cs).sum_migrate_dl_bw) };
}
}
// && cancel
unsafe {
(*cs).nr_migrate_dl_tasks = 0;
}
unsafe {
(*cs).sum_migrate_dl_bw = 0;
}
unsafe {
(*cs).dl_bw_cpu = 1i32.wrapping_neg();
}
}
cs = next;
}
}

/// Called by cgroups to determine if a cpuset is usable; cpuset_mutex held
#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_can_attach")]
extern "C" fn cpuset_can_attach(tset: *mut cgroup_taskset) -> i32 {
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut new_oldcs: *mut cpuset = unsafe { core::mem::zeroed() };
let mut new_cs: *mut cpuset = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut setsched_check: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut bool = core::ptr::addr_of_mut!(setsched_check) as *mut bool;
let mut task: core::mem::MaybeUninit<*mut task_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut task_struct = core::ptr::addr_of_mut!(task) as *mut *mut task_struct;
let mut oldcs: core::mem::MaybeUninit<*mut cpuset> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut cpuset = core::ptr::addr_of_mut!(oldcs) as *mut *mut cpuset;
let mut cs: core::mem::MaybeUninit<*mut cpuset> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *mut cpuset = core::ptr::addr_of_mut!(cs) as *mut *mut cpuset;
let mut css: core::mem::MaybeUninit<*mut cgroup_subsys_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut cgroup_subsys_state = core::ptr::addr_of_mut!(css) as *mut *mut cgroup_subsys_state;
unsafe {
let __slate_auto_bytes = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_20 = core::ptr::null_mut::<cpuset>();
*__slate_slot_21 = core::ptr::null_mut::<cpuset>();
mutex_lock(core::ptr::addr_of_mut!(cpuset_mutex));
attach_ctx.old_cs = core::ptr::null_mut::<cpuset>();
// Used later in cpuset_attach_task()
attach_ctx.cpus_updated = false;
attach_ctx.mems_updated = false;
attach_ctx.many_dest_cs = false;
// The attach_ctx.old_cs is used mainly by cpuset_migrate_mm() to get
// the old_mems_allowed value. There are two ways that many-to-one
// cpuset migration can happen:
// 1) A multithread application with threads in different cpusets is
//    wholely migrated to a new cpuset.
// 2) Disabling v2 cpuset controller will move all the tasks in child
//    cpusets to the parent cpuset.
//
// In the former case, it is the mm setting of the group leader that
// really matters. So attach_ctx.old_cs should track the oldcs of the
// group leader. It falls back to the oldcs of the first task if there
// is no group leader in the taskset. In the latter case, effective_mems
// of child cpusets must always be a subset of the parent. So no real
// page migration will be necessary no matter which child cpuset is
// selected as attach_ctx.old_cs.
//
// For a v2 threaded subtree where cpuset isn't enabled in some of the
// cgroups, it is possible that oldcs == cs for some of the tasks.
// In this case, we can skip checking on those tasks as there is no
// actual migration wrt cpuset.
*__slate_slot_19 = cgroup_taskset_first(tset, core::ptr::addr_of_mut!(*__slate_slot_22));
'__join_3: {
while !(*__slate_slot_19).is_null() {
'__join_15: {
new_cs = css_cs(*__slate_slot_22);
new_oldcs = task_cs(*__slate_slot_19);
if new_oldcs != *__slate_slot_20 || new_cs != *__slate_slot_21 {
if !(*__slate_slot_21).is_null() && new_cs != *__slate_slot_21 {
attach_ctx.many_dest_cs = true;
}
*__slate_slot_21 = new_cs;
*__slate_slot_20 = new_oldcs;
if *__slate_slot_20 == *__slate_slot_21 {
break '__join_15;
} else {
if unsafe { attach_ctx.old_cs }.is_null() {
attach_ctx.old_cs = *__slate_slot_20;
}
*__slate_slot_17 = cpuset_can_attach_check(*__slate_slot_21, *__slate_slot_20, core::ptr::addr_of_mut!(*__slate_slot_18));
if *__slate_slot_17 != 0 {
break '__join_3;
}
}
}
if *__slate_slot_20 != *__slate_slot_21 {
*__slate_slot_17 = task_can_attach(*__slate_slot_19);
if *__slate_slot_17 != 0 {
break '__join_3;
} else {
// Update attach_ctx.old_cs to the latest group leader
if *__slate_slot_19 == (*(*__slate_slot_19)).group_leader {
attach_ctx.old_cs = task_cs(*__slate_slot_19);
}
if *__slate_slot_18 {
*__slate_slot_17 = security_task_setscheduler(*__slate_slot_19);
if *__slate_slot_17 != 0 {
break '__join_3;
}
}
if dl_task(*__slate_slot_19) {
// Count all migrating DL tasks for cpuset task accounting.
// Only tasks that need a root-domain bandwidth move
// contribute to sum_migrate_dl_bw.
let __v14: *mut cpuset = *__slate_slot_21;
let __v13: i32 = (*__v14).nr_migrate_dl_tasks;
(*__v14).nr_migrate_dl_tasks = __v13.wrapping_add(1);
let __v11: *mut cpuset = *__slate_slot_20;
let __v10: i32 = (*__v11).nr_migrate_dl_tasks;
(*__v11).nr_migrate_dl_tasks = __v10.wrapping_sub(1);
if dl_task_needs_bw_move(*__slate_slot_19, ((*(*__slate_slot_21)).effective_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask) {
let __v8: *mut cpuset = *__slate_slot_21;
let __v7: u64 = (*__v8).sum_migrate_dl_bw;
(*__v8).sum_migrate_dl_bw = __v7.wrapping_add((*(*__slate_slot_19)).dl.dl_bw);
}
}
}
}
}
*__slate_slot_19 = cgroup_taskset_next(tset, core::ptr::addr_of_mut!(*__slate_slot_22));
}
if attach_ctx.many_dest_cs {
__v3 = !cpuset_v2() || attach_ctx.cpus_updated || attach_ctx.mems_updated;
} else {
__v3 = false;
}
__ret_warn_on = __v3 as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("733: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 733b - ., 3; .popsection\n# {0}", const 733 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_cpuset_can_attach_23, sym __slate_asm_string_cpuset_can_attach_24, const 3251 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("734: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 734b - ., 4; .popsection\n# {0}", const 734 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// The only case where there are multiple destination cpusets for
// task migration is when enabling a v2 cpuset controllers where
// tasks will be migrated to multiple child cpusets from a parent
// cpuset with the same effective CPUs and memory nodes. IOW,
// both attach_cpus_updated and attach_mems_updated should be false.
// If not, it is a condition that the current code cannot handle.
// Print a warning and abort the attach operation as further code
// change may be needed.
let __v5: i64 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
*__slate_slot_17 = 22i32.wrapping_neg();
} else {
*__slate_slot_17 = cpuset_reserve_dl_bw();
}
}
if *__slate_slot_17 != 0 {
clear_attach_data(core::ptr::addr_of_mut!(src_cs_head), true);
clear_attach_data(core::ptr::addr_of_mut!(dst_cs_head), true);
} else {
let __v1: i32 = attach_ctx.in_progress;
attach_ctx.in_progress = __v1.wrapping_add(1);
}
mutex_unlock(core::ptr::addr_of_mut!(cpuset_mutex));
return *__slate_slot_17;
}
}

#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_cancel_attach")]
extern "C" fn cpuset_cancel_attach(tset: *mut cgroup_taskset) {
unsafe { mutex_lock(core::ptr::addr_of_mut!(cpuset_mutex)) };
dec_attach_in_progress_locked();
clear_attach_data(unsafe { core::ptr::addr_of_mut!(src_cs_head) }, true);
clear_attach_data(unsafe { core::ptr::addr_of_mut!(dst_cs_head) }, true);
unsafe { mutex_unlock(core::ptr::addr_of_mut!(cpuset_mutex)) };
}

/// Protected by cpuset_mutex. cpus_attach is used only by cpuset_attach_task()
/// but we can't allocate it dynamically there.  Define it global and
/// allocate from cpuset_init().
static mut cpus_attach: [cpumask; 1] = unsafe { core::mem::zeroed() };

fn cpuset_attach_task(cs: *mut cpuset, task: *mut task_struct) {
lockdep_assert_cpuset_lock_held();
if cs != unsafe { core::ptr::addr_of_mut!(top_cpuset) } {
guarantee_active_cpus(task, unsafe { core::ptr::addr_of_mut!(cpus_attach) as *mut cpumask });
} else {
cpumask_andnot(unsafe { core::ptr::addr_of_mut!(cpus_attach) as *mut cpumask }, task_cpu_possible_mask!(), (unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask }) as *const cpumask);
}
// can_attach beforehand should guarantee that this doesn't
// fail.  TODO: have a better way to handle failure here
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { set_cpus_allowed_ptr(task, (core::ptr::addr_of_mut!(cpus_attach) as *mut cpumask) as *const cpumask) }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("735: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 735b - ., 3; .popsection\n# {0}", const 735 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_cpuset_attach_task_1, sym __slate_asm_string_cpuset_attach_task_2, const 3301 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("736: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 736b - ., 4; .popsection\n# {0}", const 736 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if cpuset_v2() && !unsafe { attach_ctx.mems_updated } {
return;
}
cpuset_change_task_nodemask(task, unsafe { core::ptr::addr_of_mut!(attach_ctx.nodemask_to) });
cpuset1_update_task_spread_flags(cs, task);
if task != unsafe { (*task).group_leader } || !unsafe { attach_ctx.mems_updated } {
return;
}
// Change mm for threadgroup leader. This is expensive and may
// sleep and should be moved outside migration path proper.
let mm: *mut mm_struct = unsafe { get_task_mm(task) };
if !mm.is_null() {
let oldcs: *mut cpuset = unsafe { attach_ctx.old_cs };
unsafe { mpol_rebind_mm(mm, core::ptr::addr_of_mut!((*cs).effective_mems)) };
// old_mems_allowed is the same with mems_allowed
// here, except if this task is being moved
// automatically due to hotplug.  In that case
// @mems_allowed has been updated and is empty, so
// @old_mems_allowed is the right nodesets that we
// migrate mm from.
if is_memory_migrate(cs as *const cpuset) != 0 {
cpuset_migrate_mm(mm, (unsafe { core::ptr::addr_of_mut!((*oldcs).old_mems_allowed) }) as *const __SlateRecord_nodemask_types_h_17, (unsafe { core::ptr::addr_of_mut!(attach_ctx.nodemask_to) }) as *const __SlateRecord_nodemask_types_h_17);
unsafe {
attach_ctx.task_work_queued = true;
}
} else {
unsafe { mmput(mm) };
}
}
}

#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_attach")]
extern "C" fn cpuset_attach(tset: *mut cgroup_taskset) {
let mut cs: core::mem::MaybeUninit<*mut cpuset> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut cpuset = core::ptr::addr_of_mut!(cs) as *mut *mut cpuset;
let mut css: core::mem::MaybeUninit<*mut cgroup_subsys_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut cgroup_subsys_state = core::ptr::addr_of_mut!(css) as *mut *mut cgroup_subsys_state;
let mut task: core::mem::MaybeUninit<*mut task_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut task_struct = core::ptr::addr_of_mut!(task) as *mut *mut task_struct;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
cgroup_taskset_first(tset, core::ptr::addr_of_mut!(*__slate_slot_1));
*__slate_slot_0 = css_cs(*__slate_slot_1);
lockdep_assert_cpus_held(); // see cgroup_attach_lock()
mutex_lock(core::ptr::addr_of_mut!(cpuset_mutex));
attach_ctx.task_work_queued = false;
guarantee_online_mems(*__slate_slot_0, core::ptr::addr_of_mut!(attach_ctx.nodemask_to));
// attach_ctx.old_cs can only be NULL if no task is actually migrating.
// This is highly unlikely. If it happens at all, we can skip task
// iteration and setting old_mems_allowed.
if unlikely__2!(unsafe { attach_ctx.old_cs }.is_null()) == 0 {
// In the default hierarchy, enabling cpuset in the child cgroups
// will trigger a cpuset_attach() call with no change in effective cpus
// and mems. In that case, we can optimize out by skipping the task
// iteration and the destination cpuset list is iterated to set
// old_mems_allowed.
if cpuset_v2() && !attach_ctx.cpus_updated && !attach_ctx.mems_updated {
*__slate_slot_0 = llist_entry!(unsafe { dst_cs_head.first });
while member_address_is_nonnull!(*__slate_slot_0) {
(*(*__slate_slot_0)).old_mems_allowed = attach_ctx.nodemask_to;
*__slate_slot_0 = llist_entry!(unsafe { (*(*__slate_slot_0)).attach_node.next });
}
} else {
*__slate_slot_2 = cgroup_taskset_first(tset, core::ptr::addr_of_mut!(*__slate_slot_1));
while !(*__slate_slot_2).is_null() {
cpuset_attach_task(*__slate_slot_0, *__slate_slot_2);
*__slate_slot_2 = cgroup_taskset_next(tset, core::ptr::addr_of_mut!(*__slate_slot_1));
}
if attach_ctx.task_work_queued {
schedule_flush_migrate_mm();
}
(*(*__slate_slot_0)).old_mems_allowed = attach_ctx.nodemask_to;
}
}
clear_attach_data(core::ptr::addr_of_mut!(src_cs_head), false);
clear_attach_data(core::ptr::addr_of_mut!(dst_cs_head), false);
dec_attach_in_progress_locked();
mutex_unlock(core::ptr::addr_of_mut!(cpuset_mutex));
}
}

/// Common handling for a write to a "cpus" or "mems" file.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_write_resmask")]
extern "C" fn cpuset_write_resmask(of: *mut kernfs_open_file, mut buf: *mut u8, nbytes: u64, off: i64) -> i64 {
let mut retval: i32 = unsafe { core::mem::zeroed() };
let mut trialcs: core::mem::MaybeUninit<*mut cpuset> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut cpuset = core::ptr::addr_of_mut!(trialcs) as *mut *mut cpuset;
let mut cs: *mut cpuset = unsafe { core::mem::zeroed() };
unsafe {
cs = css_cs(of_css(of));
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
retval = 19i32.wrapping_neg();
// root is read-only
if cs == core::ptr::addr_of_mut!(top_cpuset) {
return 13i32.wrapping_neg() as i64;
} else {
buf = strstrip(buf);
// cpuset_mutex acquired in wait_attach_done_lock()
mutex_lock(core::ptr::addr_of_mut!(cpuset_top_mutex));
cpus_read_lock();
wait_attach_done_lock();
if is_cpuset_online(cs) {
*__slate_slot_2 = dup_or_alloc_cpuset(cs);
if (*__slate_slot_2).is_null() {
retval = 12i32.wrapping_neg();
} else {
let __t0: u64 = (*of_cft(of)).private;
if __t0 == 1 {
retval = update_cpumask(cs, *__slate_slot_2, buf as *const u8);
} else {
if __t0 == 6 {
retval = update_exclusive_cpumask(cs, *__slate_slot_2, buf as *const u8);
} else {
if __t0 == 2 {
retval = update_nodemask(cs, *__slate_slot_2, buf as *const u8);
} else {
retval = 22i32.wrapping_neg();
}
}
}
free_cpuset(*__slate_slot_2);
}
}
cpuset_update_sd_hk_unlock();
if (*of_cft(of)).private == 2 {
schedule_flush_migrate_mm();
}
let __v0: i32 = retval;
return (if __v0 != 0 { __v0 as u64 } else { nbytes }) as i64;
}
}
}

/// These ascii lists should be read in a single call, by using a user
/// buffer large enough to hold the entire map.  If read in smaller
/// chunks, there is no guarantee of atomicity.  Since the display format
/// used, list of ranges of sequential numbers, is variable length,
/// and since these maps can change value dynamically, one could read
/// gibberish by doing partial reads while a list was changing.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_common_seq_show")]
extern "C" fn cpuset_common_seq_show(sf: *mut seq_file, v: *mut ()) -> i32 {
let cs: *mut cpuset = css_cs(seq_css(sf));
let r#type: u32 = (unsafe { (*seq_cft(sf)).private }) as u32;
let mut ret: i32 = 0;
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
match r#type as u32 {
1 => {
unsafe { seq_printf(sf, c"%*pbl\n".as_ptr() as *const u8, nr_cpu_ids, cpumask_bits__2!((*cs).cpus_allowed.as_mut_ptr() as *mut cpumask)) };
}
2 => {
unsafe { seq_printf(sf, c"%*pbl\n".as_ptr() as *const u8, __nodemask_pr_numnodes(core::ptr::addr_of_mut!((*cs).mems_allowed) as *const __SlateRecord_nodemask_types_h_17), __nodemask_pr_bits(core::ptr::addr_of_mut!((*cs).mems_allowed) as *const __SlateRecord_nodemask_types_h_17)) };
}
3 => {
unsafe { seq_printf(sf, c"%*pbl\n".as_ptr() as *const u8, nr_cpu_ids, cpumask_bits__2!((*cs).effective_cpus.as_mut_ptr() as *mut cpumask)) };
}
4 => {
unsafe { seq_printf(sf, c"%*pbl\n".as_ptr() as *const u8, __nodemask_pr_numnodes(core::ptr::addr_of_mut!((*cs).effective_mems) as *const __SlateRecord_nodemask_types_h_17), __nodemask_pr_bits(core::ptr::addr_of_mut!((*cs).effective_mems) as *const __SlateRecord_nodemask_types_h_17)) };
}
6 => {
unsafe { seq_printf(sf, c"%*pbl\n".as_ptr() as *const u8, nr_cpu_ids, cpumask_bits__2!((*cs).exclusive_cpus.as_mut_ptr() as *mut cpumask)) };
}
7 => {
unsafe { seq_printf(sf, c"%*pbl\n".as_ptr() as *const u8, nr_cpu_ids, cpumask_bits__2!((*cs).effective_xcpus.as_mut_ptr() as *mut cpumask)) };
}
5 => {
unsafe { seq_printf(sf, c"%*pbl\n".as_ptr() as *const u8, nr_cpu_ids, cpumask_bits__2!(core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask)) };
}
8 => {
unsafe { seq_printf(sf, c"%*pbl\n".as_ptr() as *const u8, nr_cpu_ids, cpumask_bits__2!(core::ptr::addr_of_mut!(isolated_cpus) as *mut cpumask)) };
}
_ => {
ret = 22i32.wrapping_neg();
}
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
ret
}

#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_partition_show")]
extern "C" fn cpuset_partition_show(seq: *mut seq_file, v: *mut ()) -> i32 {
let cs: *mut cpuset = css_cs(seq_css(seq));
let mut r#type: *const u8 = core::ptr::null::<u8>();
match unsafe { (*cs).partition_root_state } {
1 => {
seq_puts(seq, c"root\n".as_ptr() as *const u8);
}
2 => {
seq_puts(seq, c"isolated\n".as_ptr() as *const u8);
}
0 => {
seq_puts(seq, c"member\n".as_ptr() as *const u8);
}
-1 => {
r#type = c"root".as_ptr() as *const u8;
if r#type.is_null() {
r#type = c"isolated".as_ptr() as *const u8;
}
let mut _v2: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_737() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
_v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*cs).prs_err) as *const u32) };
let mut err: *const u8 = unsafe { perr_strings.0[(_v2 as u32) as usize] };
if !err.is_null() {
unsafe { seq_printf(seq, c"%s invalid (%s)\n".as_ptr() as *const u8, r#type, err) };
} else {
unsafe { seq_printf(seq, c"%s invalid\n".as_ptr() as *const u8, r#type) };
}
}
-2 => {
if r#type.is_null() {
r#type = c"isolated".as_ptr() as *const u8;
}
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_737() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*cs).prs_err) as *const u32) };
let err: *const u8 = unsafe { perr_strings.0[(__v1 as u32) as usize] };
if !err.is_null() {
unsafe { seq_printf(seq, c"%s invalid (%s)\n".as_ptr() as *const u8, r#type, err) };
} else {
unsafe { seq_printf(seq, c"%s invalid\n".as_ptr() as *const u8, r#type) };
}
}
_ => {
}
}
0
}

#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_partition_write")]
extern "C" fn cpuset_partition_write(of: *mut kernfs_open_file, mut buf: *mut u8, nbytes: u64, off: i64) -> i64 {
let cs: *mut cpuset = css_cs(unsafe { of_css(of) });
let mut val: i32 = 0;
let mut retval: i32 = 19i32.wrapping_neg();
buf = strstrip(buf);
if (unsafe { strcmp(buf as *const u8, c"root".as_ptr() as *const u8) }) == 0 {
val = 1;
} else {
if (unsafe { strcmp(buf as *const u8, c"member".as_ptr() as *const u8) }) == 0 {
val = 0;
} else {
if (unsafe { strcmp(buf as *const u8, c"isolated".as_ptr() as *const u8) }) == 0 {
val = 2;
} else {
return 22i32.wrapping_neg() as i64;
}
}
}
unsafe { mutex_lock(core::ptr::addr_of_mut!(cpuset_top_mutex)) };
unsafe { cpus_read_lock() };
wait_attach_done_lock();
if is_cpuset_online(cs) {
retval = update_prstate(cs, val);
}
cpuset_update_sd_hk_unlock();
let __v0: i32 = retval;
(if __v0 != 0 { __v0 as u64 } else { nbytes }) as i64
}

/// This is currently a minimal set for the default hierarchy. It can be
/// expanded later on by migrating more features and control files from v1.
/// terminate
static mut dfl_files: __SlateAlign16<[cftype; 10]> = __SlateAlign16([{
let mut __t0: cftype = unsafe { core::mem::zeroed() };
__t0.name = [99u8, 112u8, 117u8, 115u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t0.private = 1u64;
__t0.max_write_len = 484u64;
__t0.flags = 2u32;
__t0.seq_show = Some(cpuset_common_seq_show);
__t0.write = Some(cpuset_write_resmask);
__t0
}, {
let mut __t1: cftype = unsafe { core::mem::zeroed() };
__t1.name = [109u8, 101u8, 109u8, 115u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t1.private = 2u64;
__t1.max_write_len = 484u64;
__t1.flags = 2u32;
__t1.seq_show = Some(cpuset_common_seq_show);
__t1.write = Some(cpuset_write_resmask);
__t1
}, {
let mut __t2: cftype = unsafe { core::mem::zeroed() };
__t2.name = [99u8, 112u8, 117u8, 115u8, 46u8, 101u8, 102u8, 102u8, 101u8, 99u8, 116u8, 105u8, 118u8, 101u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2.private = 3u64;
__t2.seq_show = Some(cpuset_common_seq_show);
__t2
}, {
let mut __t3: cftype = unsafe { core::mem::zeroed() };
__t3.name = [109u8, 101u8, 109u8, 115u8, 46u8, 101u8, 102u8, 102u8, 101u8, 99u8, 116u8, 105u8, 118u8, 101u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3.private = 4u64;
__t3.seq_show = Some(cpuset_common_seq_show);
__t3
}, {
let mut __t4: cftype = unsafe { core::mem::zeroed() };
__t4.name = [99u8, 112u8, 117u8, 115u8, 46u8, 112u8, 97u8, 114u8, 116u8, 105u8, 116u8, 105u8, 111u8, 110u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t4.private = 13u64;
__t4.flags = 2u32;
__t4.file_offset = (core::mem::offset_of!(cpuset, partition_file) as u64) as u32;
__t4.seq_show = Some(cpuset_partition_show);
__t4.write = Some(cpuset_partition_write);
__t4
}, {
let mut __t5: cftype = unsafe { core::mem::zeroed() };
__t5.name = [99u8, 112u8, 117u8, 115u8, 46u8, 101u8, 120u8, 99u8, 108u8, 117u8, 115u8, 105u8, 118u8, 101u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t5.private = 6u64;
__t5.max_write_len = 484u64;
__t5.flags = 2u32;
__t5.seq_show = Some(cpuset_common_seq_show);
__t5.write = Some(cpuset_write_resmask);
__t5
}, {
let mut __t6: cftype = unsafe { core::mem::zeroed() };
__t6.name = [99u8, 112u8, 117u8, 115u8, 46u8, 101u8, 120u8, 99u8, 108u8, 117u8, 115u8, 105u8, 118u8, 101u8, 46u8, 101u8, 102u8, 102u8, 101u8, 99u8, 116u8, 105u8, 118u8, 101u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t6.private = 7u64;
__t6.flags = 2u32;
__t6.seq_show = Some(cpuset_common_seq_show);
__t6
}, {
let mut __t7: cftype = unsafe { core::mem::zeroed() };
__t7.name = [99u8, 112u8, 117u8, 115u8, 46u8, 115u8, 117u8, 98u8, 112u8, 97u8, 114u8, 116u8, 105u8, 116u8, 105u8, 111u8, 110u8, 115u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t7.private = 5u64;
__t7.flags = 33u32;
__t7.seq_show = Some(cpuset_common_seq_show);
__t7
}, {
let mut __t8: cftype = unsafe { core::mem::zeroed() };
__t8.name = [99u8, 112u8, 117u8, 115u8, 46u8, 105u8, 115u8, 111u8, 108u8, 97u8, 116u8, 101u8, 100u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t8.private = 8u64;
__t8.flags = 1u32;
__t8.seq_show = Some(cpuset_common_seq_show);
__t8
}, {
let mut __t9: cftype = unsafe { core::mem::zeroed() };
__t9
}]);

/// cpuset_css_alloc - Allocate a cpuset css
/// @parent_css: Parent css of the control group that the new cpuset will be
///              part of
/// Return: cpuset css on success, -ENOMEM on failure.
///
/// Allocate and initialize a new cpuset css, for non-NULL @parent_css, return
/// top cpuset css otherwise.
#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_css_alloc")]
extern "C" fn cpuset_css_alloc(parent_css: *mut cgroup_subsys_state) -> *mut cgroup_subsys_state {
let mut cs: *mut cpuset = unsafe { core::mem::zeroed() };
if parent_css.is_null() {
return unsafe { core::ptr::addr_of_mut!(top_cpuset.css) };
}
cs = dup_or_alloc_cpuset(core::ptr::null_mut::<cpuset>());
if cs.is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut cgroup_subsys_state;
}
__set_bit__2!(4i32, unsafe { core::ptr::addr_of_mut!((*cs).flags) });
cpuset1_init(cs);
// Set CS_MEMORY_MIGRATE for default hierarchy
if cpuset_v2() {
__set_bit__2!(3i32, unsafe { core::ptr::addr_of_mut!((*cs).flags) });
}
unsafe { core::ptr::addr_of_mut!((*cs).css) }
}

#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_css_online")]
extern "C" fn cpuset_css_online(css: *mut cgroup_subsys_state) -> i32 {
let cs: *mut cpuset = css_cs(css);
let parent: *mut cpuset = parent_cs(cs);
if parent.is_null() {
return 0;
}
cpuset_full_lock();
// For v2, clear CS_SCHED_LOAD_BALANCE if parent is isolated
let __v0: bool = cpuset_v2() && is_sched_load_balance(parent as *const cpuset) == 0;
if __v0 {
clear_bit(4i64, (unsafe { core::ptr::addr_of_mut!((*cs).flags) }) as *mut u64);
}
cpuset_inc();
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
if is_in_v2_mode() {
cpumask_copy(unsafe { (*cs).effective_cpus.as_mut_ptr() as *mut cpumask }, (unsafe { (*parent).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
unsafe {
(*cs).effective_mems = (*parent).effective_mems;
}
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
cpuset1_online_css(css);
cpuset_full_unlock();
0
}

/// If the cpuset being removed has its flag 'sched_load_balance'
/// enabled, then simulate turning sched_load_balance off, which
/// will call rebuild_sched_domains_locked(). That is not needed
/// in the default hierarchy where only changes in partition
/// will cause repartitioning.
#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_css_offline")]
extern "C" fn cpuset_css_offline(css: *mut cgroup_subsys_state) {
let cs: *mut cpuset = css_cs(css);
cpuset_full_lock();
let __v0: bool = !cpuset_v2() && is_sched_load_balance(cs as *const cpuset) != 0;
if __v0 {
cpuset_update_flag(4u32, cs, 0i32);
}
cpuset_dec();
cpuset_full_unlock();
}

/// If a dying cpuset has the 'cpus.partition' enabled, turn it off by
/// changing it back to member to free its exclusive CPUs back to the pool to
/// be used by other online cpusets.
#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_css_killed")]
extern "C" fn cpuset_css_killed(css: *mut cgroup_subsys_state) {
let cs: *mut cpuset = css_cs(css);
cpuset_full_lock();
// Reset valid partition back to member
if is_partition_valid(cs as *const cpuset) {
update_prstate(cs, 0i32);
}
cpuset_update_sd_hk_unlock();
}

#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_css_free")]
extern "C" fn cpuset_css_free(css: *mut cgroup_subsys_state) {
let cs: *mut cpuset = css_cs(css);
free_cpuset(cs);
}

#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_bind")]
extern "C" fn cpuset_bind(root_css: *mut cgroup_subsys_state) {
unsafe { mutex_lock(core::ptr::addr_of_mut!(cpuset_mutex)) };
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
if is_in_v2_mode() {
cpumask_copy(unsafe { core::ptr::addr_of_mut!(top_cpuset.cpus_allowed) as *mut cpumask }, (unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask);
cpumask_copy(unsafe { core::ptr::addr_of_mut!(top_cpuset.effective_xcpus) as *mut cpumask }, (unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask);
unsafe {
top_cpuset.mems_allowed = node_states[0];
}
} else {
cpumask_copy(unsafe { core::ptr::addr_of_mut!(top_cpuset.cpus_allowed) as *mut cpumask }, (unsafe { core::ptr::addr_of_mut!(top_cpuset.effective_cpus) as *mut cpumask }) as *const cpumask);
unsafe {
top_cpuset.mems_allowed = top_cpuset.effective_mems;
}
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
unsafe { mutex_unlock(core::ptr::addr_of_mut!(cpuset_mutex)) };
}

/// In case the child is cloned into a cpuset different from its parent,
/// additional checks are done to see if the move is allowed.
#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_can_fork")]
extern "C" fn cpuset_can_fork(task: *mut task_struct, cset: *mut css_set) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut same_cs: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut bool = core::ptr::addr_of_mut!(same_cs) as *mut bool;
let mut cs: *mut cpuset = unsafe { core::mem::zeroed() };
unsafe {
cs = css_cs(*((*cset).subsys.as_mut_ptr() as *mut *mut cgroup_subsys_state).wrapping_offset(0 as isize));
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
rcu_read_lock();
*__slate_slot_3 = cs == task_cs(get_current());
rcu_read_unlock();
if *__slate_slot_3 {
return 0;
} else {
lockdep_assert_held!(core::ptr::addr_of_mut!(cgroup_mutex));
mutex_lock(core::ptr::addr_of_mut!(cpuset_mutex));
// Check to see if task is allowed in the cpuset
*__slate_slot_2 = cpuset_can_attach_check(cs, core::ptr::null_mut::<cpuset>(), core::ptr::null_mut::<bool>());
if *__slate_slot_2 == 0 {
*__slate_slot_2 = task_can_attach(task);
if *__slate_slot_2 == 0 {
*__slate_slot_2 = security_task_setscheduler(task);
if *__slate_slot_2 == 0 {
let __v1: i32 = attach_ctx.in_progress;
attach_ctx.in_progress = __v1.wrapping_add(1);
}
}
}
mutex_unlock(core::ptr::addr_of_mut!(cpuset_mutex));
return *__slate_slot_2;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_cancel_fork")]
extern "C" fn cpuset_cancel_fork(task: *mut task_struct, cset: *mut css_set) {
let cs: *mut cpuset = css_cs(unsafe { *((*cset).subsys.as_mut_ptr() as *mut *mut cgroup_subsys_state).wrapping_offset(0 as isize) });
let mut same_cs: bool = false;
rcu_read_lock();
same_cs = cs == task_cs(get_current());
rcu_read_unlock();
if same_cs {
return;
}
dec_attach_in_progress();
}

/// Make sure the new task conform to the current state of its parent,
/// which could have been changed by cpuset just after it inherits the
/// state from the parent and before it sits on the cgroup's task list.
#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_fork")]
extern "C" fn cpuset_fork(task: *mut task_struct) {
rcu_read_lock();
let cs: *mut cpuset = task_cs(task);
let oldcs: *mut cpuset = task_cs(get_current());
rcu_read_unlock();
if cs == oldcs {
if cs == unsafe { core::ptr::addr_of_mut!(top_cpuset) } {
return;
}
unsafe { set_cpus_allowed_ptr(task, (*get_current()).cpus_ptr) };
unsafe {
(*task).mems_allowed = (*get_current()).mems_allowed;
}
return;
}
// CLONE_INTO_CGROUP
unsafe { mutex_lock(core::ptr::addr_of_mut!(cpuset_mutex)) };
guarantee_online_mems(cs, unsafe { core::ptr::addr_of_mut!(attach_ctx.nodemask_to) });
unsafe {
(*cs).old_mems_allowed = attach_ctx.nodemask_to;
}
// Assume CPUs and memory nodes are updated
// A CLONE_INTO_CGROUP operation should have taken the cgroup mutex
// and so there shouldn't be a competing cpuset_attach() operation.
let __v0: bool = true;
unsafe {
attach_ctx.mems_updated = __v0;
}
unsafe {
attach_ctx.cpus_updated = __v0;
}
unsafe {
attach_ctx.task_work_queued = false;
}
unsafe {
attach_ctx.old_cs = oldcs;
}
cpuset_attach_task(cs, task);
if unsafe { attach_ctx.task_work_queued } {
schedule_flush_migrate_mm();
}
dec_attach_in_progress_locked();
unsafe { mutex_unlock(core::ptr::addr_of_mut!(cpuset_mutex)) };
}

#[unsafe(no_mangle)]
static mut cpuset_cgrp_subsys: cgroup_subsys = {
let mut __t0: cgroup_subsys = unsafe { core::mem::zeroed() };
__t0.css_alloc = Some(cpuset_css_alloc);
__t0.css_online = Some(cpuset_css_online);
__t0.css_offline = Some(cpuset_css_offline);
__t0.css_free = Some(cpuset_css_free);
__t0.css_killed = Some(cpuset_css_killed);
__t0.can_attach = Some(cpuset_can_attach);
__t0.cancel_attach = Some(cpuset_cancel_attach);
__t0.attach = Some(cpuset_attach);
__t0.can_fork = Some(cpuset_can_fork);
__t0.cancel_fork = Some(cpuset_cancel_fork);
__t0.fork = Some(cpuset_fork);
__t0.bind = Some(cpuset_bind);
__t0.__slate_bits_0.__set_early_init(1i32 != 0i32);
__t0.__slate_bits_0.__set_threaded(1i32 != 0i32);
__t0.dfl_cftypes = unsafe { core::ptr::addr_of_mut!(dfl_files.0) as *mut cftype };
__t0
};

// cpuset_init - initialize cpusets at system boot
//
// Description: Initialize top_cpuset
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn cpuset_init() -> i32 {
loop {
if unlikely__2!(!alloc_cpumask_var(unsafe { core::ptr::addr_of_mut!(top_cpuset.cpus_allowed) }, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32))) != 0 {
loop {
unsafe {
core::arch::asm!("738: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 738b - ., 3; .popsection\n# {0}", const 738 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_cpuset_init_0, sym __slate_asm_string_cpuset_init_1, const 3876 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if unlikely__2!(!alloc_cpumask_var(unsafe { core::ptr::addr_of_mut!(top_cpuset.effective_cpus) }, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32))) != 0 {
loop {
unsafe {
core::arch::asm!("739: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 739b - ., 3; .popsection\n# {0}", const 739 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_cpuset_init_5, sym __slate_asm_string_cpuset_init_6, const 3877 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if unlikely__2!(!alloc_cpumask_var(unsafe { core::ptr::addr_of_mut!(top_cpuset.effective_xcpus) }, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32))) != 0 {
loop {
unsafe {
core::arch::asm!("740: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 740b - ., 3; .popsection\n# {0}", const 740 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_cpuset_init_10, sym __slate_asm_string_cpuset_init_11, const 3878 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if unlikely__2!(!alloc_cpumask_var(unsafe { core::ptr::addr_of_mut!(top_cpuset.exclusive_cpus) }, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32))) != 0 {
loop {
unsafe {
core::arch::asm!("741: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 741b - ., 3; .popsection\n# {0}", const 741 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_cpuset_init_15, sym __slate_asm_string_cpuset_init_16, const 3879 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if unlikely__2!(!zalloc_cpumask_var(unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) }, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32))) != 0 {
loop {
unsafe {
core::arch::asm!("742: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 742b - ., 3; .popsection\n# {0}", const 742 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_cpuset_init_20, sym __slate_asm_string_cpuset_init_21, const 3880 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if unlikely__2!(!zalloc_cpumask_var(unsafe { core::ptr::addr_of_mut!(isolated_cpus) }, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32))) != 0 {
loop {
unsafe {
core::arch::asm!("743: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 743b - ., 3; .popsection\n# {0}", const 743 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_cpuset_init_25, sym __slate_asm_string_cpuset_init_26, const 3881 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if unlikely__2!(!zalloc_cpumask_var(unsafe { core::ptr::addr_of_mut!(isolated_hk_cpus) }, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32))) != 0 {
loop {
unsafe {
core::arch::asm!("744: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 744b - ., 3; .popsection\n# {0}", const 744 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_cpuset_init_30, sym __slate_asm_string_cpuset_init_31, const 3882 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
cpumask_setall(unsafe { core::ptr::addr_of_mut!(top_cpuset.cpus_allowed) as *mut cpumask });
nodes_setall!(top_cpuset.mems_allowed);
cpumask_setall(unsafe { core::ptr::addr_of_mut!(top_cpuset.effective_cpus) as *mut cpumask });
cpumask_setall(unsafe { core::ptr::addr_of_mut!(top_cpuset.effective_xcpus) as *mut cpumask });
cpumask_setall(unsafe { core::ptr::addr_of_mut!(top_cpuset.exclusive_cpus) as *mut cpumask });
nodes_setall!(top_cpuset.effective_mems);
init_llist_node(unsafe { core::ptr::addr_of_mut!(top_cpuset.attach_node) });
cpuset1_init(unsafe { core::ptr::addr_of_mut!(top_cpuset) });
loop {
if unlikely__2!(!alloc_cpumask_var(unsafe { core::ptr::addr_of_mut!(cpus_attach) }, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32))) != 0 {
loop {
unsafe {
core::arch::asm!("745: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 745b - ., 3; .popsection\n# {0}", const 745 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_cpuset_init_35, sym __slate_asm_string_cpuset_init_36, const 3894 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if unsafe { housekeeping_enabled(0u32) } {
cpumask_andnot(unsafe { core::ptr::addr_of_mut!(isolated_cpus) as *mut cpumask }, (unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask, unsafe { housekeeping_cpumask(0u32) });
}
0
}

fn hotplug_update_tasks(cs: *mut cpuset, new_cpus: *mut cpumask, new_mems: *mut __SlateRecord_nodemask_types_h_17, cpus_updated: bool, mems_updated: bool) {
// A partition root is allowed to have empty effective cpus
let __v0: bool = cpumask_empty(new_cpus as *const cpumask) && !is_partition_valid(cs as *const cpuset);
if __v0 {
cpumask_copy(new_cpus, (unsafe { (*parent_cs(cs)).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask);
}
if nodes_empty!(*new_mems) {
unsafe {
*new_mems = (*parent_cs(cs)).effective_mems;
}
}
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
cpumask_copy(unsafe { (*cs).effective_cpus.as_mut_ptr() as *mut cpumask }, new_cpus as *const cpumask);
unsafe {
(*cs).effective_mems = *new_mems;
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
if cpus_updated {
cpuset_update_tasks_cpumask(cs, new_cpus);
}
if mems_updated {
cpuset_update_tasks_nodemask(cs);
}
}

#[unsafe(no_mangle)]
extern "C" fn cpuset_force_rebuild() {
unsafe {
force_sd_rebuild = true;
}
}

// cpuset_hotplug_update_tasks - update tasks in a cpuset for hotunplug
// @cs: cpuset in interest
// @tmp: the tmpmasks structure pointer
//
// Compare @cs's cpu and mem masks against top_cpuset and if some have gone
// offline, update @cs accordingly.  If @cs ends up with no CPU or memory,
// all its tasks are moved to the nearest ancestor with both resources.
fn cpuset_hotplug_update_tasks(cs: *mut cpuset, tmp: *mut tmpmasks) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __v4: bool = unsafe { core::mem::zeroed() };
// Force the partition to become invalid if either one of
// the following conditions hold:
// 1) empty effective cpus but not valid empty partition.
// 2) parent is invalid or doesn't grant any cpus to child
//    partitions.
// 3) subpartitions_cpus is empty.
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut __v7: bool = unsafe { core::mem::zeroed() };
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut __v9: bool = unsafe { core::mem::zeroed() };
let mut __v10: bool = unsafe { core::mem::zeroed() };
let mut parent: core::mem::MaybeUninit<*mut cpuset> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut cpuset = core::ptr::addr_of_mut!(parent) as *mut *mut cpuset;
let mut partcmd: i32 = unsafe { core::mem::zeroed() };
let mut remote: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut bool = core::ptr::addr_of_mut!(remote) as *mut bool;
let mut mems_updated: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut bool = core::ptr::addr_of_mut!(mems_updated) as *mut bool;
let mut cpus_updated: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut bool = core::ptr::addr_of_mut!(cpus_updated) as *mut bool;
unsafe {
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_1 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
partcmd = 1i32.wrapping_neg();
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
wait_attach_done_lock();
*__slate_slot_11 = parent_cs(cs);
compute_effective_cpumask(core::ptr::addr_of_mut!(new_cpus__3941), cs, *__slate_slot_11);
compute_effective_nodemask(core::ptr::addr_of_mut!(new_mems__3942), cs, *__slate_slot_11);
if !(tmp.is_null() || (*cs).partition_root_state == 0) {
// Compute effective_cpus for valid partition root, may invalidate
// child partition roots if necessary.
*__slate_slot_13 = is_remote_partition(cs);
if *__slate_slot_13 {
__v10 = true;
} else {
if is_partition_valid(cs as *const cpuset) {
__v9 = is_partition_valid(*__slate_slot_11 as *const cpuset);
} else {
__v9 = false;
}
__v10 = __v9;
}
if __v10 {
compute_partition_effective_cpumask(cs, core::ptr::addr_of_mut!(new_cpus__3941));
}
if *__slate_slot_13 {
if cpumask_empty((core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask) as *const cpumask) {
__v7 = true;
} else {
if cpumask_empty(core::ptr::addr_of_mut!(new_cpus__3941) as *const cpumask) {
__v6 = partition_is_populated(cs, core::ptr::null_mut::<cpuset>());
} else {
__v6 = false;
}
__v7 = __v6;
}
__v8 = __v7;
} else {
__v8 = false;
}
if __v8 {
loop {
if true {
__WRITE_ONCE__14!((*cs).prs_err, 6i32);
if true {
break;
}
}
}
remote_partition_disable(cs, tmp);
compute_effective_cpumask(core::ptr::addr_of_mut!(new_cpus__3941), cs, *__slate_slot_11);
*__slate_slot_13 = false;
}
if is_local_partition(cs) {
if !is_partition_valid(*__slate_slot_11 as *const cpuset) {
__v4 = true;
} else {
__v4 = tasks_nocpu_error(*__slate_slot_11, cs, core::ptr::addr_of_mut!(new_cpus__3941));
}
if __v4 {
__v3 = true;
} else {
__v3 = cpumask_empty((core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask) as *const cpumask);
}
__v5 = __v3;
} else {
__v5 = false;
}
if __v5 {
partcmd = 4;
} else {
if is_partition_valid(*__slate_slot_11 as *const cpuset) {
__v2 = is_partition_invalid(cs as *const cpuset);
} else {
__v2 = false;
}
if __v2 {
__v1 = !cpumask_empty(((*cs).effective_xcpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
} else {
__v1 = false;
}
if __v1 {
partcmd = 3;
}
}
// On the other hand, an invalid partition root may be transitioned
// back to a regular one with a non-empty effective xcpus.
if partcmd >= 0 {
update_parent_effective_cpumask(cs, partcmd, core::ptr::null_mut::<cpumask>(), tmp);
if partcmd == 4 {
__v0 = true;
} else {
__v0 = is_partition_valid(cs as *const cpuset);
}
if __v0 {
compute_partition_effective_cpumask(cs, core::ptr::addr_of_mut!(new_cpus__3941));
cpuset_force_rebuild();
}
}
}
*__slate_slot_15 = !cpumask_equal(core::ptr::addr_of_mut!(new_cpus__3941) as *const cpumask, ((*cs).effective_cpus.as_mut_ptr() as *mut cpumask) as *const cpumask);
*__slate_slot_14 = !nodes_equal!(new_mems__3942, (*cs).effective_mems);
if !(!(*__slate_slot_15) && !(*__slate_slot_14)) {
// Hotplug doesn't affect this cpuset
if *__slate_slot_14 {
check_insane_mems_config(core::ptr::addr_of_mut!(new_mems__3942));
}
if is_in_v2_mode() {
hotplug_update_tasks(cs, core::ptr::addr_of_mut!(new_cpus__3941), core::ptr::addr_of_mut!(new_mems__3942), *__slate_slot_15, *__slate_slot_14);
} else {
cpuset1_hotplug_update_tasks(cs, core::ptr::addr_of_mut!(new_cpus__3941), core::ptr::addr_of_mut!(new_mems__3942), *__slate_slot_15, *__slate_slot_14);
}
}
mutex_unlock(core::ptr::addr_of_mut!(cpuset_mutex));
}
}

static mut new_cpus__3941: cpumask = unsafe { core::mem::zeroed() };

static mut new_mems__3942: __SlateRecord_nodemask_types_h_17 = unsafe { core::mem::zeroed() };

/// cpuset_handle_hotplug - handle CPU/memory hot{,un}plug for a cpuset
///
/// This function is called after either CPU or memory configuration has
/// changed and updates cpuset accordingly.  The top_cpuset is always
/// synchronized to cpu_active_mask and N_MEMORY, which is necessary in
/// order to make cpusets transparent (of no affect) on systems that are
/// actively using CPU hotplug but making no active use of cpusets.
///
/// Non-root cpusets are only affected by offlining.  If any CPUs or memory
/// nodes have been taken down, cpuset_hotplug_update_tasks() is invoked on
/// all descendants.
///
/// Note that CPU offlining during suspend is ignored.  We don't modify
/// cpusets across suspend/resume cycles at all.
///
/// CPU / memory hotplug is handled synchronously.
fn cpuset_handle_hotplug() {
let mut cpus_updated: bool = false;
let mut mems_updated: bool = false;
let on_dfl: bool = is_in_v2_mode();
let mut tmp: tmpmasks = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tmp) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
let mut ptmp: *mut tmpmasks = core::ptr::null_mut::<tmpmasks>();
let __v0: bool = on_dfl && alloc_tmpmasks(unsafe { &mut (*core::ptr::addr_of_mut!(tmp)) }) == 0;
if __v0 {
ptmp = core::ptr::addr_of_mut!(tmp);
}
unsafe { lockdep_assert_cpus_held() };
unsafe { mutex_lock(core::ptr::addr_of_mut!(cpuset_mutex)) };
// fetch the available cpus/mems and find out which changed how
cpumask_copy(unsafe { core::ptr::addr_of_mut!(new_cpus__4044) }, (unsafe { core::ptr::addr_of_mut!(__cpu_active_mask) }) as *const cpumask);
unsafe {
new_mems__4045 = node_states[3];
}
// If subpartitions_cpus is populated, it is likely that the check
// below will produce a false positive on cpus_updated when the cpu
// list isn't changed. It is extra work, but it is better to be safe.
cpus_updated = !cpumask_equal((unsafe { core::ptr::addr_of_mut!(top_cpuset.effective_cpus) as *mut cpumask }) as *const cpumask, (unsafe { core::ptr::addr_of_mut!(new_cpus__4044) }) as *const cpumask) || !cpumask_empty((unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask }) as *const cpumask);
mems_updated = !nodes_equal!(top_cpuset.effective_mems, new_mems__4045);
// For v1, synchronize cpus_allowed to cpu_active_mask
if cpus_updated {
cpuset_force_rebuild();
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
if !on_dfl {
cpumask_copy(unsafe { core::ptr::addr_of_mut!(top_cpuset.cpus_allowed) as *mut cpumask }, (unsafe { core::ptr::addr_of_mut!(new_cpus__4044) }) as *const cpumask);
}
// Make sure that CPUs allocated to child partitions
// do not show up in effective_cpus. If no CPU is left,
// we clear the subpartitions_cpus & let the child partitions
// fight for the CPUs again.
if !cpumask_empty((unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask }) as *const cpumask) {
if cpumask_subset((unsafe { core::ptr::addr_of_mut!(new_cpus__4044) }) as *const cpumask, (unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask }) as *const cpumask) {
cpumask_clear(unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask });
} else {
cpumask_andnot(unsafe { core::ptr::addr_of_mut!(new_cpus__4044) }, (unsafe { core::ptr::addr_of_mut!(new_cpus__4044) }) as *const cpumask, (unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask }) as *const cpumask);
}
}
cpumask_copy(unsafe { core::ptr::addr_of_mut!(top_cpuset.effective_cpus) as *mut cpumask }, (unsafe { core::ptr::addr_of_mut!(new_cpus__4044) }) as *const cpumask);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
// we don't mess with cpumasks of tasks in top_cpuset
}
// synchronize mems_allowed to N_MEMORY
if mems_updated {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
if !on_dfl {
unsafe {
top_cpuset.mems_allowed = new_mems__4045;
}
}
unsafe {
top_cpuset.effective_mems = new_mems__4045;
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(callback_lock) });
cpuset_update_tasks_nodemask(unsafe { core::ptr::addr_of_mut!(top_cpuset) });
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(cpuset_mutex)) };
// if cpus or mems changed, we need to propagate to descendants
if cpus_updated || mems_updated {
let mut cs: *mut cpuset = unsafe { core::mem::zeroed() };
let mut pos_css: *mut cgroup_subsys_state = unsafe { core::mem::zeroed() };
rcu_read_lock();
pos_css = unsafe { css_next_descendant_pre(core::ptr::null_mut::<cgroup_subsys_state>(), core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(top_cpuset)).css)) };
while !pos_css.is_null() {
let __v2: *mut cpuset = css_cs(pos_css);
cs = __v2;
if is_cpuset_online(__v2) {
let __v3: bool = cs == unsafe { core::ptr::addr_of_mut!(top_cpuset) } || !css_tryget_online(unsafe { core::ptr::addr_of_mut!((*cs).css) });
if !__v3 {
rcu_read_unlock();
cpuset_hotplug_update_tasks(cs, ptmp);
rcu_read_lock();
css_put(unsafe { core::ptr::addr_of_mut!((*cs).css) });
}
}
pos_css = unsafe { css_next_descendant_pre(pos_css, core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(top_cpuset)).css)) };
}
rcu_read_unlock();
}
// rebuild_sched_domains() will always be called directly if needed
// to make sure that newly added or removed CPU will be reflected in
// the sched domains. However, if isolated partition invalidation
// or recreation is being done (update_housekeeping set), a work item
// will be queued to call housekeeping_update() to update the
// corresponding housekeeping cpumasks after some slight delay.
//
// We rely on WORK_STRUCT_PENDING_BIT to not requeue a work item that
// is still pending. Before the pending bit is cleared, the work data
// is copied out and work item dequeued. So it is possible to queue
// the work again before the hk_sd_workfn() is invoked to process the
// previously queued work. Since hk_sd_workfn() doesn't use the work
// item at all, this is not a problem.
if unsafe { force_sd_rebuild } {
rebuild_sched_domains_cpuslocked();
}
if unsafe { update_housekeeping } {
queue_work(unsafe { system_dfl_wq }, unsafe { core::ptr::addr_of_mut!(hk_sd_work) });
}
free_tmpmasks(ptmp);
}

static mut hk_sd_work: work_struct = work_struct { data: __SlateRecord_types_h_197 { counter: 4503599625273344i64 }, entry: list_head { next: unsafe { core::ptr::addr_of_mut!(hk_sd_work.entry) }, prev: unsafe { core::ptr::addr_of_mut!(hk_sd_work.entry) } }, func: Some(hk_sd_workfn) };

static mut new_cpus__4044: cpumask = unsafe { core::mem::zeroed() };

static mut new_mems__4045: __SlateRecord_nodemask_types_h_17 = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
extern "C" fn cpuset_update_active_cpus() {
// We're inside cpu hotplug critical region which usually nests
// inside cgroup synchronization.  Bounce actual hotplug processing
// to a work item to avoid reverse locking order.
cpuset_handle_hotplug();
}

/// Keep top_cpuset.mems_allowed tracking node_states[N_MEMORY].
/// Call this routine anytime after node_states[N_MEMORY] changes.
/// See cpuset_update_active_cpus() for CPU hotplug handling.
#[unsafe(link_section = ".text.slate_distinct.cpuset.cpuset_track_online_nodes")]
extern "C" fn cpuset_track_online_nodes(self_: *mut notifier_block, action: u64, arg: *mut ()) -> i32 {
cpuset_handle_hotplug();
1
}

/// cpuset_init_smp - initialize cpus_allowed
///
/// Description: Finish top cpuset after cpu, node maps are initialized
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn cpuset_init_smp() {
// cpus_allowd/mems_allowed set to v2 values in the initial
// cpuset_bind() call will be reset to v1 values in another
// cpuset_bind() call when v1 cpuset is mounted.
unsafe {
top_cpuset.old_mems_allowed = top_cpuset.mems_allowed;
}
cpumask_copy(unsafe { core::ptr::addr_of_mut!(top_cpuset.effective_cpus) as *mut cpumask }, (unsafe { core::ptr::addr_of_mut!(__cpu_active_mask) }) as *const cpumask);
unsafe {
top_cpuset.effective_mems = node_states[3];
}
hotplug_node_notifier(Some(cpuset_track_online_nodes), 10i32);
let mut __v0: *mut workqueue_struct;
let mut __v1: *mut workqueue_struct;
let mut _res: *mut workqueue_struct = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = unsafe { alloc_workqueue_noprof(c"cpuset_migrate_mm".as_ptr() as *const u8, 131074u32, 1i32) };
alloc_tag_restore!();
} else {
_res = unsafe { alloc_workqueue_noprof(c"cpuset_migrate_mm".as_ptr() as *const u8, 131074u32, 1i32) };
}
__v1 = _res;
__v0 = __v1;
unsafe {
cpuset_migrate_mm_wq = __v0;
}
loop {
if unlikely__2!(unsafe { cpuset_migrate_mm_wq }.is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("747: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 747b - ., 3; .popsection\n# {0}", const 747 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_cpuset_init_smp_3, sym __slate_asm_string_cpuset_init_smp_4, const 4190 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}

/// Return cpus_allowed mask from a task's cpuset.
fn __cpuset_cpus_allowed_locked(tsk: *mut task_struct, pmask: *mut cpumask) {
let cs: *mut cpuset = task_cs(tsk);
if cs != unsafe { core::ptr::addr_of_mut!(top_cpuset) } {
guarantee_active_cpus(tsk, pmask);
}
// Tasks in the top cpuset won't get update to their cpumasks
// when a hotplug online/offline event happens. So we include all
// offline cpus in the allowed cpu list.
let __v0: bool = cs == unsafe { core::ptr::addr_of_mut!(top_cpuset) } || cpumask_empty(pmask as *const cpumask);
if __v0 {
let possible_mask: *const cpumask = task_cpu_possible_mask!();
// We first exclude cpus allocated to partitions. If there is no
// allowable online cpu left, we fall back to all possible cpus.
cpumask_andnot(pmask, possible_mask, (unsafe { core::ptr::addr_of_mut!(subpartitions_cpus) as *mut cpumask }) as *const cpumask);
if !cpumask_intersects(pmask as *const cpumask, (unsafe { core::ptr::addr_of_mut!(__cpu_active_mask) }) as *const cpumask) {
cpumask_copy(pmask, possible_mask);
}
}
}

/// cpuset_cpus_allowed_locked - return cpus_allowed mask from a task's cpuset.
/// @tsk: pointer to task_struct from which to obtain cpuset->cpus_allowed.
/// @pmask: pointer to struct cpumask variable to receive cpus_allowed set.
///
/// Similir to cpuset_cpus_allowed() except that the caller must have acquired
/// cpuset_mutex.
#[unsafe(no_mangle)]
extern "C" fn cpuset_cpus_allowed_locked(tsk: *mut task_struct, pmask: *mut cpumask) {
lockdep_assert_cpuset_lock_held();
__cpuset_cpus_allowed_locked(tsk, pmask);
}

// cpuset_cpus_allowed - return cpus_allowed mask from a task's cpuset.
// @tsk: pointer to task_struct from which to obtain cpuset->cpus_allowed.
// @pmask: pointer to struct cpumask variable to receive cpus_allowed set.
//
// Description: Returns the cpumask_var_t cpus_allowed of the cpuset
// attached to the specified @tsk.  Guaranteed to return some non-empty
// subset of cpu_active_mask, even if this means going outside the
// tasks cpuset, except when the task is in the top cpuset.
#[unsafe(no_mangle)]
extern "C" fn cpuset_cpus_allowed(tsk: *mut task_struct, pmask: *mut cpumask) {
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(callback_lock) }, flags);
__cpuset_cpus_allowed_locked(tsk, pmask);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!(callback_lock) }, flags);
}

// cpuset_cpus_allowed_fallback - final fallback before complete catastrophe.
// @tsk: pointer to task_struct with which the scheduler is struggling
//
// Description: In the case that the scheduler cannot find an allowed cpu in
// tsk->cpus_allowed, we fall back to task_cs(tsk)->cpus_allowed. In legacy
// mode however, this value is the same as task_cs(tsk)->effective_cpus,
// which will not contain a sane cpumask during cases such as cpu hotplugging.
// This is the absolute last resort for the scheduler and it is only used if
// _every_ other avenue has been traveled.
//
// Returns true if the affinity of @tsk was changed, false otherwise.
#[unsafe(no_mangle)]
extern "C" fn cpuset_cpus_allowed_fallback(tsk: *mut task_struct) -> bool {
let possible_mask: *const cpumask = task_cpu_possible_mask!();
let mut changed: bool = false;
rcu_read_lock();
let cs_mask: *const cpumask = (unsafe { (*task_cs(tsk)).cpus_allowed.as_mut_ptr() as *mut cpumask }) as *const cpumask;
let __v0: bool = is_in_v2_mode() && cpumask_subset(cs_mask, possible_mask);
if __v0 {
unsafe { set_cpus_allowed_force(tsk, cs_mask) };
changed = true;
}
rcu_read_unlock();
// We own tsk->cpus_allowed, nobody can change it under us.
//
// But we used cs && cs->cpus_allowed lockless and thus can
// race with cgroup_attach_task() or update_cpumask() and get
// the wrong tsk->cpus_allowed. However, both cases imply the
// subsequent cpuset_change_cpumask()->set_cpus_allowed_ptr()
// which takes task_rq_lock().
//
// If we are called after it dropped the lock we must see all
// changes in tsk_cs()->cpus_allowed. Otherwise we can temporary
// set any mask even if it is not right from task_cs() pov,
// the pending set_cpus_allowed_ptr() will fix things.
//
// select_fallback_rq() will fix things ups and set cpu_possible_mask
// if required.
changed
}

/// Returns the number of CPUs available for this cgroup.
///
/// This only really works for cgroup-v2 where all the controllers are mounted
/// in the same hierarchy. If not cgroup-v2 or no cpuset controller is
/// configured it reverts to num_online_cpus().
#[unsafe(no_mangle)]
extern "C" fn cpuset_num_cpus(cgrp: *mut cgroup) -> i32 {
let mut nr: i32 = num_online_cpus() as i32;
if is_in_v2_mode() {
let mut __UNIQUE_ID_guard_748: __SlateRecord_rcupdate_h_1216 = class_rcu_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_748) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rcu_destructor(__slate_cleanup_ptr_0 as *mut _));
let cs: *mut cpuset = css_cs(unsafe { cgroup_e_css(cgrp, core::ptr::addr_of_mut!(cpuset_cgrp_subsys)) });
if !cs.is_null() {
nr = cpumask_weight((unsafe { (*cs).effective_cpus.as_mut_ptr() as *mut cpumask }) as *const cpumask) as i32;
}
}
nr
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn cpuset_init_current_mems_allowed() {
nodes_setall!((*get_current()).mems_allowed);
}

// cpuset_mems_allowed - return mems_allowed mask from a tasks cpuset.
// @tsk: pointer to task_struct from which to obtain cpuset->mems_allowed.
//
// Description: Returns the nodemask_t mems_allowed of the cpuset
// attached to the specified @tsk.  Guaranteed to return some non-empty
// subset of node_states[N_MEMORY], even if this means going outside the
// tasks cpuset.
#[unsafe(no_mangle)]
extern "C" fn cpuset_mems_allowed(tsk: *mut task_struct) -> __SlateRecord_nodemask_types_h_17 {
let mut mask: __SlateRecord_nodemask_types_h_17 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(mask) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(callback_lock) }, flags);
guarantee_online_mems(task_cs(tsk), core::ptr::addr_of_mut!(mask));
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!(callback_lock) }, flags);
return mask;
}

/// cpuset_nodemask_valid_mems_allowed - check nodemask vs. current mems_allowed
/// @nodemask: the nodemask to be checked
///
/// Are any of the nodes in the nodemask allowed in current->mems_allowed?
#[unsafe(no_mangle)]
extern "C" fn cpuset_nodemask_valid_mems_allowed(nodemask: *const __SlateRecord_nodemask_types_h_17) -> i32 {
nodes_intersects!(*nodemask, (*get_current()).mems_allowed) as i32
}

/// nearest_hardwall_ancestor() - Returns the nearest mem_exclusive or
/// mem_hardwall ancestor to the specified cpuset.  Call holding
/// callback_lock.  If no ancestor is mem_exclusive or mem_hardwall
/// (an unusual configuration), then returns the root cpuset.
fn nearest_hardwall_ancestor(mut cs: *mut cpuset) -> *mut cpuset {
loop {
let __v0: bool = is_mem_exclusive(cs as *const cpuset) != 0 || is_mem_hardwall(cs as *const cpuset) != 0;
let __v1: bool = !__v0 && !parent_cs(cs).is_null();
if !__v1 {
break;
}
cs = parent_cs(cs);
}
cs
}

/// cpuset_current_node_allowed - Can current task allocate on a memory node?
/// @node: is this an allowed node?
/// @gfp_mask: memory allocation flags
///
/// If we're in interrupt, yes, we can always allocate.  If @node is set in
/// current's mems_allowed, yes.  If it's not a __GFP_HARDWALL request and this
/// node is set in the nearest hardwalled cpuset ancestor to current's cpuset,
/// yes.  If current has access to memory reserves as an oom victim, yes.
/// If the current task is PF_EXITING, yes. Otherwise, no.
///
/// GFP_USER allocations are marked with the __GFP_HARDWALL bit,
/// and do not allow allocations outside the current tasks cpuset
/// unless the task has been OOM killed or is exiting.
/// GFP_KERNEL allocations are not so marked, so can escape to the
/// nearest enclosing hardwalled ancestor cpuset.
///
/// Scanning up parent cpusets requires callback_lock.  The
/// page allocator only calls here with __GFP_HARDWALL bit
/// _not_ set if it's a GFP_KERNEL allocation, and all nodes in the
/// current tasks mems_allowed came up empty on the first pass over
/// the zonelist.  So only GFP_KERNEL allocations, if all nodes in the
/// cpuset are short of memory, might require taking the callback_lock.
///
/// The first call here from mm/page_alloc:get_page_from_freelist()
/// has __GFP_HARDWALL set in gfp_mask, enforcing hardwall cpusets,
/// so no allocation on a node outside the cpuset is allowed (unless
/// in interrupt, of course).  The PF_EXITING check must therefore
/// come before the __GFP_HARDWALL check, otherwise a dying task
/// would be blocked on the fast path.
///
/// The second pass through get_page_from_freelist() doesn't even call here for
/// GFP_ATOMIC calls.  That, and the checks below have the combined affect that:
/// 	in_interrupt - any node ok (current task context irrelevant)
/// 	GFP_ATOMIC   - any node ok
/// 	tsk_is_oom_victim   - any node ok
/// 	PF_EXITING   - any node ok (let dying task exit quickly)
/// 	GFP_KERNEL   - any node in enclosing hardwalled cpuset ok
/// 	GFP_USER     - only nodes in current tasks mems allowed ok.
#[unsafe(no_mangle)]
extern "C" fn cpuset_current_node_allowed(node: i32, gfp_mask: u32) -> bool {
let mut cs: *mut cpuset = unsafe { core::mem::zeroed() }; // current cpuset ancestors
let mut allowed: bool = false; // is allocation in zone z allowed?
let mut flags: u64 = 0;
if in_interrupt!() != 0 {
return true;
}
if node_isset__2!(node, (*get_current()).mems_allowed) != 0 {
return true;
}
// Allow tasks that have access to memory reserves because they have
// been OOM killed to get memory anywhere.
if unlikely__2!(tsk_is_oom_victim(get_current())) != 0 {
return true;
}
if (unsafe { (*get_current()).flags }) & 4 != 0 {
return true;
}
// Let dying task have memory
if gfp_mask & (BIT!(20i32) as u32) != 0 {
return false;
}
// If hardwall request, stop here
if cpuset_v2() {
return true;
}
// Not hardwall and node outside mems_allowed: scan up cpusets
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(callback_lock) }, flags);
cs = nearest_hardwall_ancestor(task_cs(get_current()));
allowed = node_isset__3!(node, (*cs).mems_allowed) != 0;
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!(callback_lock) }, flags);
allowed
}

/// cpuset_nodes_allowed - return effective_mems mask from a cgroup cpuset.
/// @cgroup: pointer to struct cgroup.
/// @mask: pointer to struct nodemask_t to be returned.
///
/// Returns effective_mems mask from a cgroup cpuset if it is cgroup v2 and
/// has cpuset subsys. Otherwise, returns node_states[N_MEMORY].
///
/// This function intentionally avoids taking the cpuset_mutex or callback_lock
/// when accessing effective_mems. This is because the obtained effective_mems
/// is stale immediately after the query anyway (e.g., effective_mems is updated
/// immediately after releasing the lock but before returning).
///
/// As a result, returned @mask may be empty because cs->effective_mems can be
/// rebound during this call. Besides, nodes in @mask are not guaranteed to be
/// online due to hot plugins. Callers should check the mask for validity on
/// return based on its subsequent use.
#[unsafe(no_mangle)]
extern "C" fn cpuset_nodes_allowed(cgroup: *mut cgroup, mask: *mut __SlateRecord_nodemask_types_h_17) {
let mut cs: *mut cpuset = unsafe { core::mem::zeroed() };
// In v1, mem_cgroup and cpuset are unlikely in the same hierarchy
// and mems_allowed is likely to be empty even if we could get to it,
// so return directly to avoid taking a global lock on the empty check.
let __v0: bool = cgroup.is_null() || !cpuset_v2();
if __v0 {
nodes_copy!(*mask, node_states[3]);
return;
}
let css: *mut cgroup_subsys_state = unsafe { cgroup_get_e_css(cgroup, core::ptr::addr_of_mut!(cpuset_cgrp_subsys)) };
if css.is_null() {
nodes_copy!(*mask, node_states[3]);
return;
}
// The reference taken via cgroup_get_e_css is sufficient to
// protect css, but it does not imply safe accesses to effective_mems.
//
// Normally, accessing effective_mems would require the cpuset_mutex
// or callback_lock - but the correctness of this information is stale
// immediately after the query anyway. We do not acquire the lock
// during this process to save lock contention in exchange for racing
// against mems_allowed rebinds.
cs = container_of!(css, cpuset);
nodes_copy!(*mask, (*cs).effective_mems);
css_put(css);
}

/// cpuset_spread_node() - On which node to begin search for a page
/// @rotor: round robin rotor
///
/// If a task is marked PFA_SPREAD_PAGE and a page cache allocation uses
/// cpuset_mem_spread_node() to determine where to start looking, spread the
/// starting node around the task's mems_allowed nodes instead of starting on
/// the local node.
///
/// We don't have to worry about the returned node being offline
/// because "it can't happen", and even if it did, it would be ok.
///
/// The routines calling guarantee_online_mems() are careful to
/// only set nodes in task->mems_allowed that are online.  So it
/// should not be possible for the following code to return an
/// offline node.  But if it did, that would be ok, as this routine
/// is not returning the node where the allocation must be, only
/// the node where the search should start.  The zonelist used by
/// the allocator will include all nodes.  If the slab allocator
/// is passed an offline node, it will fall back to the local node.
/// See kmem_cache_alloc_node().
fn cpuset_spread_node(rotor: *mut i32) -> i32 {
let __v0: i32 = next_node_in!(unsafe { *rotor }, (*get_current()).mems_allowed) as i32;
unsafe {
*rotor = __v0;
}
__v0
}

/// cpuset_mem_spread_node() - On which node to begin search for a file page
#[unsafe(no_mangle)]
extern "C" fn cpuset_mem_spread_node() -> i32 {
if (unsafe { (*get_current()).cpuset_mem_spread_rotor }) == 1i32.wrapping_neg() {
unsafe {
(*get_current()).cpuset_mem_spread_rotor = node_random(core::ptr::addr_of_mut!((*get_current()).mems_allowed) as *const __SlateRecord_nodemask_types_h_17);
}
}
cpuset_spread_node(unsafe { core::ptr::addr_of_mut!((*get_current()).cpuset_mem_spread_rotor) })
}

// cpuset_mems_allowed_intersects - Does @tsk1's mems_allowed intersect @tsk2's?
// @tsk1: pointer to task_struct of some task.
// @tsk2: pointer to task_struct of some other task.
//
// Description: Return true if @tsk1's mems_allowed intersects the
// mems_allowed of @tsk2.  Used by the OOM killer to determine if
// one of the task's memory usage might impact the memory available
// to the other.
#[unsafe(no_mangle)]
extern "C" fn cpuset_mems_allowed_intersects(tsk1: *const task_struct, tsk2: *const task_struct) -> i32 {
nodes_intersects__3!((*tsk1).mems_allowed, (*tsk2).mems_allowed) as i32
}

/// cpuset_print_current_mems_allowed - prints current's cpuset and mems_allowed
///
/// Description: Prints current's name, cpuset name, and cached copy of its
/// mems_allowed to the kernel log.
#[unsafe(no_mangle)]
extern "C" fn cpuset_print_current_mems_allowed() {
rcu_read_lock();
let cgrp: *mut cgroup = unsafe { (*task_cs(get_current())).css.cgroup };
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x01c,cpuset=".as_ptr() as *const u8) };
pr_cont_cgroup_name(cgrp);
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x01c,mems_allowed=%*pbl".as_ptr() as *const u8, __nodemask_pr_numnodes(core::ptr::addr_of_mut!((*get_current()).mems_allowed) as *const __SlateRecord_nodemask_types_h_17), __nodemask_pr_bits(core::ptr::addr_of_mut!((*get_current()).mems_allowed) as *const __SlateRecord_nodemask_types_h_17)) };
rcu_read_unlock();
}

/// Display task mems_allowed in /proc/<pid>/status file.
#[unsafe(no_mangle)]
extern "C" fn cpuset_task_status_allowed(m: *mut seq_file, task: *mut task_struct) {
unsafe { seq_printf(m, c"Mems_allowed:\t%*pb\n".as_ptr() as *const u8, __nodemask_pr_numnodes(core::ptr::addr_of_mut!((*task).mems_allowed) as *const __SlateRecord_nodemask_types_h_17), __nodemask_pr_bits(core::ptr::addr_of_mut!((*task).mems_allowed) as *const __SlateRecord_nodemask_types_h_17)) };
unsafe { seq_printf(m, c"Mems_allowed_list:\t%*pbl\n".as_ptr() as *const u8, __nodemask_pr_numnodes(core::ptr::addr_of_mut!((*task).mems_allowed) as *const __SlateRecord_nodemask_types_h_17), __nodemask_pr_bits(core::ptr::addr_of_mut!((*task).mems_allowed) as *const __SlateRecord_nodemask_types_h_17)) };
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_compute_partition_effective_cpumask_3: [u8; 1] = *b"\0";

static __slate_asm_string_compute_partition_effective_cpumask_4: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_cpuset_attach_task_1: [u8; 1] = *b"\0";

static __slate_asm_string_cpuset_attach_task_2: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_cpuset_can_attach_23: [u8; 1] = *b"\0";

static __slate_asm_string_cpuset_can_attach_24: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_cpuset_init_0: [u8; 1] = *b"\0";

static __slate_asm_string_cpuset_init_1: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_cpuset_init_10: [u8; 1] = *b"\0";

static __slate_asm_string_cpuset_init_11: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_cpuset_init_15: [u8; 1] = *b"\0";

static __slate_asm_string_cpuset_init_16: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_cpuset_init_20: [u8; 1] = *b"\0";

static __slate_asm_string_cpuset_init_21: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_cpuset_init_25: [u8; 1] = *b"\0";

static __slate_asm_string_cpuset_init_26: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_cpuset_init_30: [u8; 1] = *b"\0";

static __slate_asm_string_cpuset_init_31: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_cpuset_init_35: [u8; 1] = *b"\0";

static __slate_asm_string_cpuset_init_36: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_cpuset_init_5: [u8; 1] = *b"\0";

static __slate_asm_string_cpuset_init_6: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_cpuset_init_smp_3: [u8; 1] = *b"\0";

static __slate_asm_string_cpuset_init_smp_4: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_cpuset_update_sd_hk_unlock_1: [u8; 1] = *b"\0";

static __slate_asm_string_cpuset_update_sd_hk_unlock_2: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_generate_sched_domains_38: [u8; 1] = *b"\0";

static __slate_asm_string_generate_sched_domains_39: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_guarantee_active_cpus_1: [u8; 1] = *b"\0";

static __slate_asm_string_guarantee_active_cpus_2: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_isolated_cpus_update_1: [u8; 1] = *b"\0";

static __slate_asm_string_isolated_cpus_update_2: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_partition_xcpus_add_1: [u8; 1] = *b"\0";

static __slate_asm_string_partition_xcpus_add_2: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_partition_xcpus_del_1: [u8; 1] = *b"\0";

static __slate_asm_string_partition_xcpus_del_2: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_percpu_ref_tryget_live_rcu_1: [u8; 1] = *b"\0";

static __slate_asm_string_percpu_ref_tryget_live_rcu_2: [u8; 61] = *b"/home/takashi/c-corpus/linux/include/linux/percpu-refcount.h\0";

static __slate_asm_string_rebuild_sched_domains_locked_1: [u8; 1] = *b"\0";

static __slate_asm_string_rebuild_sched_domains_locked_2: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_remote_cpus_update_17: [u8; 1] = *b"\0";

static __slate_asm_string_remote_cpus_update_18: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_remote_cpus_update_20: [u8; 1] = *b"\0";

static __slate_asm_string_remote_cpus_update_21: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_remote_cpus_update_22: [u8; 1] = *b"\0";

static __slate_asm_string_remote_cpus_update_23: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_remote_partition_disable_1: [u8; 1] = *b"\0";

static __slate_asm_string_remote_partition_disable_2: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_remote_partition_disable_7: [u8; 1] = *b"\0";

static __slate_asm_string_remote_partition_disable_8: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_update_cpumasks_hier_25: [u8; 1] = *b"\0";

static __slate_asm_string_update_cpumasks_hier_26: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_update_nodemasks_hier_4: [u8; 1] = *b"\0";

static __slate_asm_string_update_nodemasks_hier_5: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_update_parent_effective_cpumask_39: [u8; 1] = *b"\0";

static __slate_asm_string_update_parent_effective_cpumask_40: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_update_parent_effective_cpumask_41: [u8; 1] = *b"\0";

static __slate_asm_string_update_parent_effective_cpumask_42: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_update_parent_effective_cpumask_43: [u8; 1] = *b"\0";

static __slate_asm_string_update_parent_effective_cpumask_44: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_update_parent_effective_cpumask_45: [u8; 1] = *b"\0";

static __slate_asm_string_update_parent_effective_cpumask_46: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";

static __slate_asm_string_update_prstate_14: [u8; 1] = *b"\0";

static __slate_asm_string_update_prstate_15: [u8; 52] = *b"/home/takashi/c-corpus/linux/kernel/cgroup/cpuset.c\0";
