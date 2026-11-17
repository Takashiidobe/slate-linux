
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
//! PCI Bus Services, see include/linux/pci.h for further explanation.
//!
//! Copyright 1993 -- 1997 Drew Eckhardt, Frederic Potter,
//! David Mosberger-Tang
//!
//! Copyright 1997 -- 2000 Martin Mares <mj@ucw.cz>
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
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
//! This counts to 15. Any more, it will return 16th argument.
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
//! restart by calling sys_restart_syscall
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! # chars in a path name including nul
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
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
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
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
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
//! Defines x86 CPU feature bits
//! 64bit FS base
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
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
//! @a is a power of 2 value
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
//! upper_32_bits - return bits 32-63 of a number
//! @n: the number we're accessing
//!
//! A basic shift-right of a 64- or 32-bit quantity.  Use this to suppress
//! the "right shift count >= width of type" warning when that quantity is
//! 32-bits.
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
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
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
//! New user namespace
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! Convenience macros for the sake of set_current_state:
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
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
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
//! number of env pointers
//! Generate SD flag indexes
//! Desired design of maximum size and alignment (see RFC2553)
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
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
//! This macro is used to tag functions as "printf-like" because
//! some compilers (like GCC) can catch printf format string problems.
//! Value returned by acpi_os_get_thread_id. There is no standard "thread_id"
//! across operating systems or even the various UNIX systems. Since ACPICA
//! only needs the thread ID as a unique thread identifier, we use a u64
//! as the only common data type - it will accommodate any type of pointer or
//! any type of integer. It is up to the host-dependent OSL to cast the
//! native thread ID type to a u64 (in acpi_os_get_thread_id).
//! For values, see Producer/Consumer above
//! See the comment in asm-generic/io.h about ioremap_np().
//! DWORD 3
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
//! Conventional PCI and PCI-X Mode 1 devices have 256 bytes of
//! configuration space.  PCI-X Mode 2 and PCIe devices have 4096 bytes of
//! configuration space.
//! Vendors and devices.  Sort key: vendor first, device next.
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! Runtime PM flag argument bits
//! ***DANGEROUS*** don't add guard page
//! 8237 DMA controllers
//! AER and DPC capabilities TLP Logging register sizes (PCIe r6.2, sec 7.8.4
//! & 7.9.14).
//! FIELD_PREP() - prepare a bitfield element
//! @_mask: shifted mask defining the field's length and position
//! @_val:  value to put in the field
//!
//! FIELD_PREP() masks and shifts up the value.  The result should
//! be combined with other fields of the bitfield using logical OR.
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
//! 31 pointers + header align the folio_batch structure to a power of two
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! Number of possible devfns: 0.0 to 1f.7 inclusive
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
// No such file or directory
// I/O error
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
// No space left on device
// Math result not representable
// Invalid system call number
// Value too large for defined data type
// Operation not supported on transport endpoint
// Connection timed out
// Memory page has hardware error
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
// Concatenate two parameters, but allow them to be expanded beforehand.
// Operation is not supported
// Force a compilation error if a constant expression is not a power of 2
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
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// Some subsystems have their own custom printk that applies a va_format to a
// generic format, for example, to include a device number or other metadata
// alongside the format supplied by the caller.
//
// In order to store these in the way they would be emitted by the printk
// infrastructure, the subsystem provides us with the start, fixed string, and
// any subsequent text in the format string.
//
// We take a variable argument list as pr_fmt/dev_fmt/etc are sometimes passed
// as multiple arguments (eg: `"%s: ", "blah"`), and we must only take the
// first one.
//
// subsys_fmt_prefix must be known at compile time, or compilation will fail
// (since this is a mistake). If fmt or level is not known at compile time, no
// index entry will be made (since this can legitimately happen).
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
// Overflow Flag
// N 32-bit words worth of info
// N 32-bit bug flags
// "pse" Page Size Extensions
// "tsc" Time Stamp Counter
// "apic" Onboard APIC
// "acpi" ACPI via MSR
// "sse"
// "pdpe1gb" GB pages
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "monitor" MONITOR/MWAIT support
// "est" Enhanced SpeedStep
// "popcnt" POPCNT instruction
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
// Non-serializing WRMSR
// INVLPGB and TLBSYNC instructions supported
// "wbnoinvd" WBNOINVD instruction
// "hwp" Intel Hardware P-states
// "ospke" OS Protection Keys Enable
// Shadow stack
// "la57" 5-level page tables
// "rdpid" RDPID instruction
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "apic_c1e" AMD Erratum 400
// "l1tf" CPU is affected by L1 Terminal Fault
// "div0" AMD DIV0 speculation bug
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
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
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
// lower_32_bits - return bits 0-31 of a number
// @n: the number we're accessing
// No static inline type checking - see Subtlety (1) above.
// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
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
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
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
// SCHED_ISO: reserved but not implemented yet
// disable sas during sighandling
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
// Maximum number of zones on a zonelist
// zone_idx() returns 0 for the ZONE_DMA zone, 1 for the ZONE_NORMAL zone, etc.
// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
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
// Generate SD flag bits
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
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// DWORD 5
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
// Number of standard BARs
// 16 bits
// 16 bits
// Enable response in I/O space
// Enable response in Memory space
// Enable bus mastering
// Use memory write and invalidate
// Enable parity checking
// INTx Emulation Disable
// 16 bits
// Support Capability List
// 8 bits
// 8 bits
// 8 bits
// Offset of first capability list entry
// 8 bits
// 0x3c-0x3d are same as for htype 0
// Forward VGA addresses
// Secondary bus reset
// Header type 2 (CardBus bridges)
// Power Management
// PCI-X
// HyperTransport
// PCI Express
// PCI Advanced Features
// PCI Enhanced Allocation
// Next capability in the list
// PM Capabilities Register
// Version
// D1 power state support
// D2 power state support
// PME Mask of all supported states
// PME# from D0
// PME# from D1
// PME# from D2
// PME# from D3 (hot)
// PME# from D3 (cold)
// PM control and status register
// Current power state (D0 to D3)
// No reset for D3hot->D0
// PME pin enable
// PME pin status
// Number of Capability Entries
// Num Entries Mask
// First EA Entry in List
// Entry Size
// BAR Equivalent Indicator
// Expansion ROM
// 9-14 map to VF BARs 0-5 respectively
// Primary Properties
// Secondary Properties
// Non-Prefetch Memory
// Prefetchable Memory
// I/O Space
// VF Prefetchable Memory
// VF Non-Prefetch Memory
// Bridge I/O Space
// 0x08-0xfc reserved
// Reserved Memory
// Enable for this entry
// bit 0 is reserved
// 64-bit field flag
// For Base & Max Offset
// Modes & Features
// Max Memory Read Byte Count
// PCI-X capabilities
// Designed Max Memory Read Count
// Device/Port type
// Express Endpoint
// Legacy Endpoint
// Root Port
// Upstream Port
// Downstream Port
// PCIe to PCI/PCI-X Bridge
// PCI/PCI-X to PCIe Bridge
// Root Complex Integrated Endpoint
// Root Complex Event Collector
// Function Level Reset
// Device Control
// Max_Payload_Size
// Max_Read_Request_Size
// Bridge Configuration Retry / FLR
// Device Status
// Transactions Pending
// Link Capabilities
// Max Link Speed (prior to PCIe r3.0: Supported Link Speeds)
// LNKCAP2 SLS Vector bit 0
// LNKCAP2 SLS Vector bit 1
// Maximum Link Width
// Link Control
// Retrain Link
// Link Status
// Current Link Speed
// Negotiated Link Width
// Link Training
// Data Link Layer Link Active
// Slot Control
// Root Control
// Root Status
// PME status
// The Device Capabilities 2, Device Status 2, Device Control 2,
// Link Capabilities 2, Link Status 2, Link Control 2,
// Slot Capabilities 2, Slot Status 2, and Slot Control 2 registers
// are only present on devices with PCIe Capability version 2.
// Use pcie_capability_read_word() and similar interfaces to use them
// safely.
// Device Capabilities 2
// Alternative Routing-ID
// Atomic Op routing
// Device Control 2
// Alternative Routing-ID
// Set Atomic requests
// Block atomic egress
// Link Capabilities 2
// Supported Link Speeds Vector
// Supported Speed 2.5GT/s
// Supported Speed 5GT/s
// Link Control 2
// Supported Speed 2.5GT/s
// Supported Speed 5GT/s
// Supported Speed 8GT/s
// Supported Speed 16GT/s
// Supported Speed 32GT/s
// Supported Speed 64GT/s
// Flit Mode Status
// Slot Control 2
// Device Serial Number
// Vendor-Specific
// Access Control Services
// Alternate Routing ID
// Latency Tolerance Reporting
// Designated Vendor-Specific
// Vendor-Specific (VSEC, PCI_EXT_CAP_ID_VNDR)
// Vendor-Specific Header
// HyperTransport sub capability types
//
// Unfortunately there are both 3 bit and 5 bit capability types defined
// in the HT spec, catering for that is a little messy. You probably don't
// want to use these directly, just use pci_find_ht_capability() and it
// will do the right thing for you.
// Slave/Primary link configuration
// Host/Secondary link configuration
// Number of VF BARs
// Access Control Service
// ACS Capability Register
// Source Validation
// Translation Blocking
// P2P Request Redirect
// P2P Completion Redirect
// Upstream Forwarding
// P2P Egress Control
// Direct Translated P2P
// ACS Control Register
// Designated Vendor-Specific (DVSEC, PCI_EXT_CAP_ID_DVSEC)
// Designated Vendor-Specific Header1
// Designated Vendor-Specific Header2
// CXL r4.0, 8.1.5: Extensions DVSEC for Ports
// DMA_ATTR_NO_WARN: This tells the DMA-mapping subsystem to suppress
// allocation failure reports (similarly to __GFP_NOWARN).
// A dma_addr_t can hold any valid DMA or bus address for the platform.  It can
// be given to a device to use as a DMA source or target.  It is specific to a
// given device and there may be a translation between the CPU physical address
// space and the bus address space.
//
// DMA_MAPPING_ERROR is the magic error code if a mapping failed.  It should not
// be used directly in drivers, but checked for using dma_mapping_error()
// instead.
// Use the high bit to mark if we used swiotlb for one or more ranges.
// 8 bit slave DMA, channels 0..3
// 16 bit master DMA, ch 4(=slave input)..7
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// DMA page registers
// FIELD_GET() - extract a bitfield element
// @_mask: shifted mask defining the field's length and position
// @_reg:  value of entire bitfield
//
// FIELD_GET() extracts the field specified by @_mask from the
// bitfield passed in as @_reg by masking and shifting it down.
// FIELD_MODIFY() - modify a bitfield element
// @_mask: shifted mask defining the field's length and position
// @_reg_p: pointer to the memory that should be updated
// @_val: value to store in the bitfield
//
// FIELD_MODIFY() modifies the set of bits in @_reg_p specified by @_mask,
// by replacing them with the bitfield value passed in as @_val.
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
macro_rules! ACPI_COMPANION {
($dev:expr) => {
to_acpi_device_node!(unsafe { (*$dev).fwnode })
};
}

macro_rules! ALIGN_DOWN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL__2!((($x) as i32).wrapping_sub($a.wrapping_sub(1)), $a)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__9 {
() => {
(((const { core::mem::size_of::<[pci_reset_fn_method; 8]>() }) as u64) / ((const { core::mem::size_of::<pci_reset_fn_method>() }) as u64)).wrapping_add(__must_be_array__4!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__13 {
() => {
((core::mem::size_of::<[*const u8; 10]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__18!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__17 {
() => {
((core::mem::size_of::<[*const u8; 10]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__8!() as u64)
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

macro_rules! GENMASK_INPUT_CHECK__5 {
() => {
core::mem::size_of::<__SlateRecord_pci_c_6101>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__7 {
() => {
core::mem::size_of::<__SlateRecord_bitmap_h_539>() as i32
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
macro_rules! GENMASK_TYPE__6 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__7!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__8 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__5!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK__3 {
($h:expr, $l:expr) => {
GENMASK_TYPE__6!(u64, $h, $l)
};
}

macro_rules! GENMASK__4 {
($h:expr, $l:expr) => {
GENMASK_TYPE__8!(u64, $h, $l)
};
}

macro_rules! IS_ALIGNED__2 {
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

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

/// PCIe link information from Link Capabilities 2
macro_rules! PCIE_LNKCAP2_SLS2SPEED {
($lnkcap2:expr) => {
if (($lnkcap2) as i32) & 64 != 0 { 25i32 } else { if (($lnkcap2) as i32) & 32 != 0 { 24i32 } else { if (($lnkcap2) as i32) & 16 != 0 { 23i32 } else { if (($lnkcap2) as i32) & 8 != 0 { 22i32 } else { if (($lnkcap2) as i32) & 4 != 0 { 21i32 } else { if (($lnkcap2) as i32) & 2 != 0 { 20i32 } else { 255i32 } } } } } }
};
}

/// PCIe speed to Mb/s reduced by encoding overhead
macro_rules! PCIE_SPEED2MBS_ENC {
($speed:expr) => {
if (($speed) as u32) == 25 { 64000i32 } else { if (($speed) as u32) == 24 { 31507i32 } else { if (($speed) as u32) == 23 { 15753i32 } else { if (($speed) as u32) == 22 { 7876i32 } else { if (($speed) as u32) == 21 { 4000i32 } else { if (($speed) as u32) == 20 { 2000i32 } else { 0i32 } } } } } }
};
}

/// The PCI interface treats multi-function devices as independent
/// devices.  The slot/function address of each device is encoded
/// in a single byte as follows:
///
/// 	7:3 = slot
/// 	2:0 = function
macro_rules! PCI_DEVFN__3 {
($slot:expr, $func:expr) => {
(($slot) & 31) << 3i32 | ($func) & 7
};
}

/// Extended Capabilities (PCI-X 2.0 and Express)
macro_rules! PCI_EXT_CAP_ID {
($header:expr) => {
($header) & 65535
};
}

macro_rules! PCI_EXT_CAP_NEXT {
($header:expr) => {
($header) >> 20i32 & 4092
};
}

macro_rules! PCI_FUNC__2 {
($devfn:expr) => {
($devfn) & 7
};
}

macro_rules! PCI_FUNC__3 {
($devfn:expr) => {
(($devfn) as i32) & 7
};
}

macro_rules! PCI_POSSIBLE_ERROR {
($val:expr) => {
($val) == 4294967295u32
};
}

macro_rules! PCI_POSSIBLE_ERROR__2 {
($val:expr) => {
(($val) as i32) == 65535i32
};
}

macro_rules! PCI_SLOT__2 {
($devfn:expr) => {
($devfn) >> 3i32 & 31
};
}

macro_rules! PCI_SLOT__3 {
($devfn:expr) => {
(($devfn) as i32) >> 3i32 & 31
};
}

macro_rules! PCI_VNDR_HEADER_ID {
($x:expr) => {
($x) & 65535
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

macro_rules! _AC__2 {
() => {
16045481047390945280u64
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
__ALIGN_KERNEL_MASK!($x, (($a) as i32).wrapping_sub(1))
};
}

macro_rules! __READ_ONCE {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i32) }
};
}

macro_rules! __WRITE_ONCE__2 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut *mut hlist_node, $val) };
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

macro_rules! __WRITE_ONCE__3 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut hlist_node, $val) };
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

macro_rules! __bf_cast_unsigned {
($x:expr) => {
($x) as u32
};
}

macro_rules! __bf_cast_unsigned__3 {
($x:expr) => {
($x) as u64
};
}

macro_rules! __bf_cast_unsigned__2 {
($x:expr) => {
($x) as u16
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

macro_rules! __must_be_array__4 {
() => {
((const { core::mem::size_of::<__SlateRecord_pci_c_5242>() }) as u64) as i32
};
}

macro_rules! __must_be_array__8 {
() => {
core::mem::size_of::<__SlateRecord_pci_c_857>() as i32
};
}

macro_rules! __must_be_array__18 {
() => {
core::mem::size_of::<__SlateRecord_pci_c_860>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
() => {
((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__7 {
() => {
((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
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

macro_rules! __ratelimit {
($state:expr) => {
unsafe { ___ratelimit($state, c"pci_set_full_power_state".as_ptr() as *const u8) }
};
}

macro_rules! __ratelimit__2 {
($state:expr) => {
unsafe { ___ratelimit($state, c"pci_set_low_power_state".as_ptr() as *const u8) }
};
}

macro_rules! __request_mem_region {
($start:expr, $n:expr, $name:expr, $excl:expr) => {
unsafe { __request_region(core::ptr::addr_of_mut!(iomem_resource), $start, $n, $name, $excl) }
};
}

/// The following macros are non-atomic versions of their non-underscored
/// counterparts.
macro_rules! __set_bit {
($nr:expr, $addr:expr) => {
if (unsafe { __slate_intrinsic_1c5364fcd9da51fe($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
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
macro_rules! __sign_use__3 {
($ux:expr) => {
if is_signed_type__3!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
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
macro_rules! __sign_use {
($ux:expr) => {
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u32>() }) as u64) < 4) as i32)) }
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

macro_rules! __types_ok__2 {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
};
}

macro_rules! __types_ok__3 {
($ux:expr, $uy:expr) => {
__sign_use__3!($ux) & __sign_use__3!($uy)
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

macro_rules! arch_atomic_sub_return {
($i:expr, $v:expr) => {
arch_atomic_add_return($i.wrapping_neg(), $v)
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
macro_rules! container_of__3 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__7!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__8 {
($ptr:expr) => {
{
let mut __v: *mut pci_pme_device;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__29!() as isize)) }) as *mut ()) as *mut pci_pme_device;
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
macro_rules! container_of__10 {
($ptr:expr) => {
{
let mut __v: *mut pci_cap_saved_state;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!() as isize)) }) as *mut ()) as *mut pci_cap_saved_state;
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
macro_rules! container_of__15 {
($ptr:expr) => {
{
let mut __v: *const pci_dev;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *const pci_dev;
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
macro_rules! container_of__2 {
($ptr:expr) => {
{
let mut __v: *mut pci_bus;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__5!() as isize)) }) as *mut ()) as *mut pci_bus;
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
($ptr:expr) => {
{
let mut __v: *mut pci_dev;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *mut pci_dev;
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
macro_rules! container_of__6 {
($ptr:expr) => {
{
let mut __v: *mut pci_slot;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__12!() as isize)) }) as *mut ()) as *mut pci_slot;
__v
}
};
}

/// ffs - find first set bit in word
/// @x: the word to search
///
/// This is defined the same way as the libc and compiler builtin ffs
/// routines, therefore differs in spirit from the other bitops.
///
/// ffs(value) returns 0 if value is 0 or the position of the first
/// set bit if value is nonzero. The first (least significant) bit
/// is at position 1.
macro_rules! ffs {
($x:expr) => {
if unsafe { __slate_intrinsic_a39e427d3b6c2df4($x) } { {
let __t = $x;
if __t == 0 { 0i32 } else { (__t.trailing_zeros() as i32) + 1 }
} } else { variable_ffs($x) }
};
}

macro_rules! hlist_entry {
($ptr:expr) => {
container_of__10!($ptr)
};
}

macro_rules! hlist_entry_safe {
($ptr:expr) => {
{
let mut __v: *mut pci_cap_saved_state;
let __v2: *mut hlist_node = $ptr;
let __v3: *mut pci_cap_saved_state = if !__v2.is_null() {
hlist_entry!(__v2)
} else {
core::ptr::null_mut::<pci_cap_saved_state>()
};
__v = __v3;
__v
}
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__2 {
() => {
(1i32.wrapping_neg() as u32) < 1
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__3 {
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
macro_rules! list_entry__6 {
($ptr:expr) => {
container_of__8!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__7 {
($ptr:expr) => {
container_of__15!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry {
($ptr:expr) => {
container_of!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__2 {
($ptr:expr) => {
container_of__2!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__4 {
($ptr:expr) => {
container_of__6!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__5 {
($ptr:expr, $type:ty) => {
container_of__3!($ptr, $type)
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

/// list_entry_is_head - test if the entry points to the head of the list
/// @pos:	the type * to cursor
/// @head:	the head for your list.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry_is_head__3 {
($pos:expr, $head:expr, $member:ident) => {
list_is_head(unsafe { core::ptr::addr_of!((*$pos).$member) }, $head)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__6 {
($ptr:expr) => {
list_entry__4!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry {
($ptr:expr) => {
list_entry!(unsafe { (*$ptr).next })
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
list_entry__2!(unsafe { (*$ptr).next })
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
list_entry__6!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__12 {
($ptr:expr, $type:ty) => {
list_entry__5!(unsafe { (*$ptr).next }, $type)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__13 {
($ptr:expr) => {
list_entry__7!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__7 {
($pos:expr, $member:ident) => {
list_entry__4!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__5 {
($pos:expr, $member:ident) => {
list_entry__2!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__9 {
($pos:expr, $member:ident) => {
list_entry__6!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__11 {
($pos:expr, $member:ident) => {
list_entry__7!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry {
($pos:expr, $member:ident) => {
list_entry!(unsafe { (*$pos).$member.next })
};
}

/// list_prev_entry - get the prev element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_prev_entry {
($pos:expr, $member:ident) => {
list_entry!(unsafe { (*$pos).$member.prev })
};
}

macro_rules! lockdep_assert_held {
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

macro_rules! offsetof__3 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, fwnode) as u64
};
}

macro_rules! offsetof__5 {
() => {
core::mem::offset_of!(pci_bus, node) as u64
};
}

macro_rules! offsetof__7 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, bus_list) as u64
};
}

macro_rules! offsetof__12 {
() => {
core::mem::offset_of!(pci_slot, list) as u64
};
}

macro_rules! offsetof__29 {
() => {
core::mem::offset_of!(pci_pme_device, list) as u64
};
}

macro_rules! offsetof__2 {
() => {
core::mem::offset_of!(pci_dev, bus_list) as u64
};
}

macro_rules! offsetof__4 {
() => {
0u64
};
}

macro_rules! pci_resource_end {
($dev:expr, $bar:expr) => {
unsafe { (*pci_resource_n!($dev, $bar)).end }
};
}

macro_rules! pci_resource_flags {
($dev:expr, $bar:expr) => {
unsafe { (*pci_resource_n!($dev, $bar)).flags }
};
}

macro_rules! pci_resource_len {
($dev:expr, $bar:expr) => {
{
let __v: u64 = if pci_resource_end!($dev, $bar) != 0 {
resource_size(pci_resource_n!($dev, $bar) as *const resource)
} else {
0
};
__v
}
};
}

/// These helpers provide future and backwards compatibility
/// for accessing popular PCI BAR info
macro_rules! pci_resource_n {
($dev:expr, $bar:expr) => {
unsafe { (*$dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(($bar) as isize)
};
}

macro_rules! pci_resource_start {
($dev:expr, $bar:expr) => {
unsafe { (*pci_resource_n!($dev, $bar)).start }
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

macro_rules! release_mem_region {
($start:expr, $n:expr) => {
unsafe { __release_region(core::ptr::addr_of_mut!(iomem_resource), $start, $n) };
};
}

/// Compatibility cruft
macro_rules! release_region {
($start:expr, $n:expr) => {
unsafe { __release_region(core::ptr::addr_of_mut!(ioport_resource), $start, $n) };
};
}

/// Convenience shorthand with allocation
macro_rules! request_region__2 {
($start:expr, $n:expr, $name:expr) => {
unsafe { __request_region(core::ptr::addr_of_mut!(ioport_resource), $start, $n, $name, 0i32) }
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
(unsafe { __slate_intrinsic_1c5364fcd9da51fe($nbits) }) && ($nbits) <= 64 && ($nbits) > 0
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

macro_rules! test_bit__3 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_1c5364fcd9da51fe($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
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
let __v: bool = typecheck__2!(u64) != 0 && typecheck__3!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
}
};
}

/// time_before - returns true if the time a is before time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Return: %true is time a is before time b, otherwise %false.
macro_rules! time_before {
($a:expr, $b:expr) => {
time_after!($b, $a)
};
}

macro_rules! to_acpi_device_node {
($__fwnode:expr) => {
{
let mut __v: *mut acpi_device;
let __v2: *mut fwnode_handle = $__fwnode;
let __v3: *mut acpi_device = if unsafe { is_acpi_device_node(__v2 as *const fwnode_handle) } {
let mut __v4: *mut acpi_device;
__v4 = ((unsafe { (__v2 as *const u8).wrapping_offset(-(offsetof__3!(acpi_device) as isize)) }) as *mut ()) as *mut acpi_device;
__v4
} else {
core::ptr::null_mut::<acpi_device>()
};
__v = __v3;
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

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck__3 {
($type:ty) => {
{
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == (core::ptr::addr_of_mut!(__v3) as *mut $type);
__v = 1;
__v
}
};
}

/// Check at compile time that something is a pointer type.
macro_rules! typecheck_pointer__2 {
() => {
let mut __v: i32;
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
    static mut ioport_resource: resource;
    static mut iomem_resource: resource;
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __SCK__might_resched: static_call_key;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut jiffies: u64;
    static mut system_freezable_wq: *mut workqueue_struct;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut acpi_disabled: i32;
    static mut pci_flags: u32;
    static mut pci_bus_type: bus_type;
    static mut kstat: kernel_stat;
    static mut pm_suspend_global_flags: u32;
    static mut pcie_link_speed: [u8; 0];
    static mut pci_bus_sem: rw_semaphore;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __request_region(arg0: *mut resource, arg1: u64, arg2: u64, arg3: *const u8, arg4: i32) -> *mut resource;
    fn __release_region(arg0: *mut resource, arg1: u64, arg2: u64);
    fn iomem_get_mapping() -> *mut address_space;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn ___ratelimit(arg0: *mut ratelimit_state, arg1: *const u8) -> i32;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __compiletime_assert_14();
    fn __compiletime_assert_17();
    fn __compiletime_assert_18();
    fn __compiletime_assert_29();
    fn __compiletime_assert_30();
    fn __compiletime_assert_31();
    fn __compiletime_assert_32();
    fn __bad_size_call_parameter();
    fn __xadd_wrong_size();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn strlen(arg0: *const u8) -> u64;
    fn strcmp(arg0: *const u8, arg1: *const u8) -> i32;
    fn strncmp(arg0: *const u8, arg1: *const u8, arg2: u64) -> i32;
    fn strchr(arg0: *const u8, arg1: i32) -> *mut u8;
    fn strchrnul(arg0: *const u8, arg1: i32) -> *mut u8;
    fn strrchr(arg0: *const u8, arg1: i32) -> *mut u8;
    fn strstr(arg0: *const u8, arg1: *const u8) -> *mut u8;
    fn kstrdup(arg0: *const u8, arg1: u32) -> *mut u8;
    fn kstrndup(arg0: *const u8, arg1: u64, arg2: u32) -> *mut u8;
    fn kmemdup_nul(arg0: *const u8, arg1: u64, arg2: u32) -> *mut u8;
    fn memparse(arg0: *const u8, arg1: *mut *mut u8) -> u64;
    fn bitmap_zalloc(arg0: u32, arg1: u32) -> *mut u64;
    fn __bitmap_set(arg0: *mut u64, arg1: u32, arg2: i32);
    fn simple_strtoul(arg0: *const u8, arg1: *mut *mut u8, arg2: u32) -> u64;
    fn sscanf(arg0: *const u8, arg1: *const u8, ...) -> i32;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn irq_to_desc(arg0: u32) -> *mut irq_desc;
    fn __compiletime_assert_69();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_trylock(arg0: *mut mutex) -> i32;
    fn mutex_unlock(arg0: *mut mutex);
    fn delayed_work_timer_fn(arg0: *mut timer_list);
    fn down_read(arg0: *mut rw_semaphore);
    fn up_read(arg0: *mut rw_semaphore);
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn queue_delayed_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut delayed_work, arg3: u64) -> bool;
    fn kfree(arg0: *const ());
    fn __compiletime_assert_220();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn ida_alloc_range(arg0: *mut ida, arg1: u32, arg2: u32, arg3: u32) -> i32;
    fn ida_free(arg0: *mut ida, arg1: u32);
    fn sysfs_emit(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn bus_create_file(arg0: *const bus_type, arg1: *mut bus_attribute) -> i32;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_395();
    fn __compiletime_assert_416();
    fn try_module_get(arg0: *mut module) -> bool;
    fn module_put(arg0: *mut module);
    fn device_set_wakeup_capable(arg0: *mut device, arg1: bool);
    fn pm_wakeup_dev_event(arg0: *mut device, arg1: u32, arg2: bool);
    fn dev_driver_string(arg0: *const device) -> *const u8;
    fn acpi_has_method(arg0: *mut (), arg1: *mut u8) -> bool;
    fn is_acpi_device_node(arg0: *const fwnode_handle) -> bool;
    fn acpi_dev_get_pci_dev(arg0: *mut acpi_device) -> *mut pci_dev;
    fn ioremap(arg0: u64, arg1: u64) -> *mut ();
    fn ioremap_wc(arg0: u64, arg1: u64) -> *mut ();
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn msleep(arg0: u32);
    fn usleep_range_state(arg0: u64, arg1: u64, arg2: u32);
    fn dmi_check_system(arg0: *const dmi_system_id) -> i32;
    fn dmi_get_bios_year() -> i32;
    fn pci_dev_reset_iommu_prepare(arg0: *mut pci_dev) -> i32;
    fn pci_dev_reset_iommu_done(arg0: *mut pci_dev);
    fn pci_find_host_bridge(arg0: *mut pci_bus) -> *mut pci_host_bridge;
    fn pci_dev_get(arg0: *mut pci_dev) -> *mut pci_dev;
    fn pci_dev_put(arg0: *mut pci_dev);
    fn pci_bus_read_config_byte(arg0: *mut pci_bus, arg1: u32, arg2: i32, arg3: *mut u8) -> i32;
    fn pci_bus_read_config_word(arg0: *mut pci_bus, arg1: u32, arg2: i32, arg3: *mut u16) -> i32;
    fn pci_bus_read_config_dword(arg0: *mut pci_bus, arg1: u32, arg2: i32, arg3: *mut u32) -> i32;
    fn pci_read_config_byte(arg0: *const pci_dev, arg1: i32, arg2: *mut u8) -> i32;
    fn pci_read_config_word(arg0: *const pci_dev, arg1: i32, arg2: *mut u16) -> i32;
    fn pci_read_config_dword(arg0: *const pci_dev, arg1: i32, arg2: *mut u32) -> i32;
    fn pci_write_config_byte(arg0: *const pci_dev, arg1: i32, arg2: u8) -> i32;
    fn pci_write_config_word(arg0: *const pci_dev, arg1: i32, arg2: u16) -> i32;
    fn pci_write_config_dword(arg0: *const pci_dev, arg1: i32, arg2: u32) -> i32;
    fn pcie_capability_read_word(arg0: *mut pci_dev, arg1: i32, arg2: *mut u16) -> i32;
    fn pcie_capability_read_dword(arg0: *mut pci_dev, arg1: i32, arg2: *mut u32) -> i32;
    fn pcie_capability_write_word(arg0: *mut pci_dev, arg1: i32, arg2: u16) -> i32;
    fn pcie_capability_write_dword(arg0: *mut pci_dev, arg1: i32, arg2: u32) -> i32;
    fn pcie_capability_clear_and_set_word_unlocked(arg0: *mut pci_dev, arg1: i32, arg2: u16, arg3: u16) -> i32;
    fn pcie_capability_clear_and_set_word_locked(arg0: *mut pci_dev, arg1: i32, arg2: u16, arg3: u16) -> i32;
    fn pci_update_resource(arg0: *mut pci_dev, arg1: i32);
    fn pci_enable_resources(arg0: *mut pci_dev, arg1: i32) -> i32;
    fn pci_bus_resource_n(arg0: *const pci_bus, arg1: i32) -> *mut resource;
    fn pci_walk_bus(arg0: *mut pci_bus, arg1: Option<unsafe extern "C" fn(*mut pci_dev, *mut ()) -> i32>, arg2: *mut ());
    fn pci_restore_msi_state(arg0: *mut pci_dev);
    fn pci_cfg_access_lock(arg0: *mut pci_dev);
    fn pci_cfg_access_trylock(arg0: *mut pci_dev) -> bool;
    fn pci_cfg_access_unlock(arg0: *mut pci_dev);
    fn pci_fixup_device(arg0: u32, arg1: *mut pci_dev);
    fn __compiletime_assert_516();
    fn __pm_runtime_idle(arg0: *mut device, arg1: i32) -> i32;
    fn __pm_runtime_resume(arg0: *mut device, arg1: i32) -> i32;
    fn pm_runtime_get_if_active(arg0: *mut device) -> i32;
    fn __pm_runtime_set_status(arg0: *mut device, arg1: u32) -> i32;
    fn pm_runtime_barrier(arg0: *mut device);
    fn pm_runtime_forbid(arg0: *mut device);
    fn pciehp_is_native(arg0: *mut pci_dev) -> bool;
    fn pci_save_vc_state(arg0: *mut pci_dev) -> i32;
    fn pci_restore_vc_state(arg0: *mut pci_dev);
    fn pci_allocate_vc_save_buffers(arg0: *mut pci_dev);
    fn pci_no_msi();
    fn pci_realloc_get_opt(arg0: *mut u8);
    fn pci_setup_cardbus(arg0: *mut u8) -> i32;
    fn pci_bus_read_dev_vendor_id(arg0: *mut pci_bus, arg1: i32, arg2: *mut u32, arg3: i32) -> bool;
    fn pci_walk_bus_locked(arg0: *mut pci_bus, arg1: Option<unsafe extern "C" fn(*mut pci_dev, *mut ()) -> i32>, arg2: *mut ());
    fn pci_disable_bridge_window(arg0: *mut pci_dev);
    fn pci_speed_string(arg0: u32) -> *const u8;
    fn pcie_update_link_speed(arg0: *mut pci_bus, arg1: u32);
    fn pci_restore_ats_state(arg0: *mut pci_dev);
    fn pci_restore_pri_state(arg0: *mut pci_dev);
    fn pci_restore_pasid_state(arg0: *mut pci_dev);
    fn pci_dev_specific_acs_enabled(arg0: *mut pci_dev, arg1: u16) -> i32;
    fn pci_dev_specific_enable_acs(arg0: *mut pci_dev) -> i32;
    fn pci_dev_specific_disable_acs_redir(arg0: *mut pci_dev) -> i32;
    fn pci_disable_broken_acs_cap(arg0: *mut pci_dev);
    fn pcie_failed_link_retrain(arg0: *mut pci_dev) -> i32;
    fn pci_save_ltr_state(arg0: *mut pci_dev);
    fn pci_restore_ltr_state(arg0: *mut pci_dev);
    fn pci_save_aspm_l1ss_state(arg0: *mut pci_dev);
    fn pci_restore_aspm_l1ss_state(arg0: *mut pci_dev);
    fn pcie_aspm_pm_state_change(arg0: *mut pci_dev, arg1: bool);
    fn pcie_aspm_powersave_config_link(arg0: *mut pci_dev);
    fn pci_bridge_reconfigure_ltr(arg0: *mut pci_dev);
    fn pcie_reset_lbms(arg0: *mut pci_dev);
    fn pci_dev_specific_reset(arg0: *mut pci_dev, arg1: bool) -> i32;
    fn pci_restore_rebar_state(arg0: *mut pci_dev);
    fn pci_dev_acpi_reset(arg0: *mut pci_dev, arg1: bool) -> i32;
    fn acpi_pci_power_manageable(arg0: *mut pci_dev) -> bool;
    fn acpi_pci_bridge_d3(arg0: *mut pci_dev) -> bool;
    fn acpi_pci_set_power_state(arg0: *mut pci_dev, arg1: i32) -> i32;
    fn acpi_pci_get_power_state(arg0: *mut pci_dev) -> i32;
    fn acpi_pci_refresh_power_state(arg0: *mut pci_dev);
    fn acpi_pci_wakeup(arg0: *mut pci_dev, arg1: bool) -> i32;
    fn acpi_pci_need_resume(arg0: *mut pci_dev) -> bool;
    fn acpi_pci_choose_state(arg0: *mut pci_dev) -> i32;
    fn __compiletime_assert_684();
    fn __compiletime_assert_687();
    fn __compiletime_assert_693();
    fn __compiletime_assert_694();
    fn __compiletime_assert_695();
    fn __compiletime_assert_696();
    fn __compiletime_assert_697();
    fn __compiletime_assert_698();
    fn __compiletime_assert_699();
    fn __compiletime_assert_700();
    fn __compiletime_assert_701();
    fn __compiletime_assert_702();
    fn __compiletime_assert_709();
    fn __compiletime_assert_710();
    fn __compiletime_assert_711();
    fn __compiletime_assert_712();
    fn __compiletime_assert_713();
    fn __compiletime_assert_714();
    fn __compiletime_assert_715();
    fn __compiletime_assert_716();
    fn __compiletime_assert_717();
    fn __compiletime_assert_718();
    fn __compiletime_assert_719();
    fn __compiletime_assert_720();
    fn __compiletime_assert_721();
    fn __compiletime_assert_722();
    fn __compiletime_assert_723();
    fn __compiletime_assert_724();
    fn __compiletime_assert_725();
    fn __compiletime_assert_726();
    fn __compiletime_assert_727();
    fn __compiletime_assert_728();
    fn __compiletime_assert_760();
    fn __compiletime_assert_761();
    fn __compiletime_assert_762();
    fn __compiletime_assert_763();
    fn __compiletime_assert_764();
    fn __compiletime_assert_765();
    fn __compiletime_assert_766();
    fn __compiletime_assert_767();
    fn __compiletime_assert_768();
    fn __compiletime_assert_769();
    fn __compiletime_assert_770();
    fn __compiletime_assert_771();
    fn __compiletime_assert_772();
    fn __compiletime_assert_773();
    fn __compiletime_assert_774();
    fn __compiletime_assert_775();
    fn __compiletime_assert_776();
    fn __compiletime_assert_777();
    fn __compiletime_assert_778();
    fn __compiletime_assert_779();
    fn __compiletime_assert_780();
    fn __compiletime_assert_781();
    fn __compiletime_assert_782();
    fn __compiletime_assert_783();
    fn __compiletime_assert_784();
    fn __compiletime_assert_810();
    fn __compiletime_assert_816();
    fn __compiletime_assert_824();
    fn __compiletime_assert_825();
    fn __compiletime_assert_826();
    fn __compiletime_assert_827();
    fn __compiletime_assert_828();
    fn __compiletime_assert_830();
    fn __compiletime_assert_831();
    fn __compiletime_assert_832();
    fn __compiletime_assert_833();
    fn __compiletime_assert_834();
    fn __compiletime_assert_836();
    fn __compiletime_assert_837();
    fn __compiletime_assert_838();
    fn __compiletime_assert_839();
    fn __compiletime_assert_840();
    fn __compiletime_assert_841();
    fn __compiletime_assert_842();
    fn __compiletime_assert_843();
    fn __compiletime_assert_844();
    fn __compiletime_assert_845();
    fn __compiletime_assert_846();
    fn __compiletime_assert_847();
    fn __compiletime_assert_848();
    fn __compiletime_assert_849();
    fn __compiletime_assert_850();
    fn __compiletime_assert_852();
    fn __compiletime_assert_853();
    fn __compiletime_assert_854();
    fn __compiletime_assert_855();
    fn __compiletime_assert_856();
    fn __compiletime_assert_858();
    fn __compiletime_assert_859();
    fn __compiletime_assert_860();
    fn __compiletime_assert_861();
    fn __compiletime_assert_862();
    fn __compiletime_assert_864();
    fn __compiletime_assert_865();
    fn __compiletime_assert_866();
    fn __compiletime_assert_867();
    fn __compiletime_assert_868();
    fn __compiletime_assert_870();
    fn __compiletime_assert_871();
    fn __compiletime_assert_872();
    fn __compiletime_assert_873();
    fn __compiletime_assert_874();
    fn __compiletime_assert_876();
    fn __compiletime_assert_877();
    fn __compiletime_assert_878();
    fn __compiletime_assert_879();
    fn __compiletime_assert_880();
    fn __compiletime_assert_882();
    fn __compiletime_assert_883();
    fn __compiletime_assert_884();
    fn __compiletime_assert_885();
    fn __compiletime_assert_886();
    fn __compiletime_assert_889();
    fn __compiletime_assert_890();
    fn __compiletime_assert_891();
    fn __compiletime_assert_892();
    fn __compiletime_assert_893();
    fn __compiletime_assert_902();
    fn __compiletime_assert_909();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
    #[link_name = "llvm.is.constant.i16"]
    fn __slate_intrinsic_b52e897e44daaa8(arg0: u16) -> bool;
    #[link_name = "llvm.returnaddress.p0"]
    fn __slate_intrinsic_ded315013ea6523(arg0: i32) -> *mut ();
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
#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely__2!(overflow) != 0
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
struct resource {
    start: u64,
    end: u64,
    name: *const u8,
    flags: u64,
    desc: u64,
    parent: *mut resource,
    sibling: *mut resource,
    child: *mut resource,
}


// Resource type
// PCI/ISA I/O ports
// No side effects
// size indicates alignment
// start field is alignment
// Resource extended types
// Userland may not map this resource
// No address assigned yet
// PCI control bits.  Shares IORESOURCE_BITS with above PCI ROM.
// Do not move resource
// BAR Equivalent Indicator
fn resource_set_size(res: *mut resource, size: u64) {
unsafe {
(*res).end = unsafe { (*res).start }.wrapping_add(size).wrapping_sub(1);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
}


fn resource_set_range(res: *mut resource, start: u64, size: u64) {
unsafe {
(*res).start = start;
}
resource_set_size(res, size);
}

fn resource_size(res: *const resource) -> u64 {
unsafe { (*res).end }.wrapping_sub(unsafe { (*res).start }).wrapping_add(1)
}

fn resource_type(res: *const resource) -> u64 {
(unsafe { (*res).flags }) & 7936
}

fn __resource_contains_unbound(r1: *const resource, r2: *const resource) -> bool {
if resource_type(r1) != resource_type(r2) {
return false;
}
(unsafe { (*r1).start }) <= unsafe { (*r2).start } && (unsafe { (*r1).end }) >= unsafe { (*r2).end }
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


fn resource_contains(r1: *const resource, r2: *const resource) -> bool {
if (unsafe { (*r1).flags }) & 536870912 != 0 || (unsafe { (*r2).flags }) & 536870912 != 0 {
return false;
}
__resource_contains_unbound(r1, r2)
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
// A "pure" initcall has no dependencies on anything else, and purely
// initializes variables that couldn't be statically initialized.
//
// This only exists for built-in code, not for modules.
// Keep main.c:initcall_level_names[] in sync.
#[repr(C)]
#[derive(Clone, Copy)]
struct obs_kernel_param {
    str: *const u8,
    setup_func: Option<unsafe extern "C" fn(*mut u8) -> i32>,
    early: i32,
}


// Only for really core code.  See moduleparam.h for the normal way.
//
// Force the alignment so the compiler doesn't space elements of the
// obs_kernel_param "array" too far apart in .init.setup.
// NOTE: __setup functions return values:
// @fn returns 1 (or non-zero) if the option argument is "handled"
// and returns 0 if the option argument is "not handled".
// NOTE: @fn is as per module_param, not __setup!
// I.e., @fn returns 0 for no error or non-zero for error
// (possibly @fn returns a -errno value, but it does not matter).
// Emits warning if @fn returns non-zero.
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


// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
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
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_14() };
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

#[inline(always)]
fn list_add(new: *mut list_head, head: *mut list_head) {
__list_add(new, head, unsafe { (*head).next });
}

fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_17() };
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

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_18() };
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

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bitmap_h_539 {
}


// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// list_for_each_entry_continue_reverse - iterate backwards from the given point
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
//
// Start to iterate over list of given type backwards, continuing after
// the current position.
// list_for_each_entry_safe - iterate over list of given type safe against removal of list entry
// @pos:	the type * to use as a loop cursor.
// @n:		another type * to use as temporary storage
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
fn hlist_add_head(n: *mut hlist_node, h: *mut hlist_head) {
let first: *mut hlist_node = unsafe { (*h).first };
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_29() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!((*n).next, first);
if true {
break;
}
}
if !first.is_null() {
loop {
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_30() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*first).pprev, unsafe { core::ptr::addr_of_mut!((*n).next) });
if true {
break;
}
}
}
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_31() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!((*h).first, n);
if true {
break;
}
}
loop {
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_32() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*n).pprev, unsafe { core::ptr::addr_of_mut!((*h).first) });
if true {
break;
}
}
}

// hlist_for_each_entry	- iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the hlist_node within the struct.
// hlist_for_each_entry_safe - iterate over list of given type safe against removal of list entry
// @pos:	the type * to use as a loop cursor.
// @n:		a &struct hlist_node to use as temporary storage
// @head:	the head for your list.
// @member:	the name of the hlist_node within the struct.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
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
struct io_bitmap {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits104U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits105U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[inline(always)]
fn arch_atomic_inc(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock incl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
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


#[inline(always)]
fn arch_atomic_dec(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock decl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[inline(always)]
fn raw_atomic_add_return(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_add_return(i, v)
}

#[inline(always)]
fn raw_atomic_sub_return(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_sub_return!(i, v)
}

#[inline(always)]
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
}

#[inline(always)]
fn raw_atomic_inc_return(v: *mut __SlateRecord_types_h_190) -> i32 {
raw_atomic_add_return(1i32, v)
}

#[inline(always)]
fn raw_atomic_dec(v: *mut __SlateRecord_types_h_190) {
arch_atomic_dec(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[inline(always)]
fn raw_atomic_dec_return(v: *mut __SlateRecord_types_h_190) -> i32 {
raw_atomic_sub_return(1i32, v)
}

#[inline(always)]
fn instrument_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_write!(v, size);
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
core::arch::asm!("52: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 52b - ., 3; .popsection\n# {0}", const 52 as i32, options(nostack, att_syntax));
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
core::arch::asm!("53: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 53b - ., 4; .popsection\n# {0}", const 53 as i32, options(nostack, att_syntax));
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
fn atomic_inc(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc(v);
}

#[inline(always)]
fn atomic_inc_return(v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc_return(v)
}

#[inline(always)]
fn atomic_dec(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec(v);
}

#[inline(always)]
fn atomic_dec_return(v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec_return(v)
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
fn arch___set_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
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
struct futex_private_hash {
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
fn variable_ffs(mut x: i32) -> i32 {
let mut r: i32 = 0;
// AMD64 says BSFL won't clobber the dest reg if x==0; Intel64 says the
// dest reg is undefined if x==0, but their CPU architect says its
// value is written to set it to the same as before, except that the
// top 32 bits will be cleared.
//
// We cannot do this on 32 bits because at the very least some
// 486 CPUs did not behave this way.
unsafe {
let __t0: u32;
core::arch::asm!("bsfl {1:e},{0:e}", inlateout(reg) (1 as i32).wrapping_neg() as u32 => __t0, in(reg) x as u32, options(pure, nomem, nostack, att_syntax));
r = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
r.wrapping_add(1)
}

#[inline(always)]
fn ___set_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___set_bit(nr, addr);
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
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[inline(always)]
fn bitmap_set(map: *mut u64, start: u32, nbits: u32) {
if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(nbits) }) && nbits == 1 {
__set_bit!(start, map);
} else {
if small_const_nbits__2!(start.wrapping_add(nbits)) {
let __v1: u64 = unsafe { *map };
let __v2: u64 = __v1 | GENMASK__3!(start.wrapping_add(nbits).wrapping_sub(1), start);
unsafe {
*map = __v2;
}
} else {
if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(start & 7u32) }) && IS_ALIGNED__2!(start, 8i32) && unsafe { __slate_intrinsic_1c5364fcd9da51fe(nbits & 7u32) } && IS_ALIGNED__2!(nbits, 8i32) {
unsafe { memset((map as *mut u8).wrapping_offset((start / 8) as isize) as *mut (), 255i32, (nbits / 8) as u64) };
} else {
unsafe { __bitmap_set(map, start, nbits as i32) };
}
}
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
fn is_power_of_2(n: u64) -> bool {
n.wrapping_sub(1) < n ^ n.wrapping_sub(1)
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
struct hrtimer_clock_base {
}


#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_56: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_69() };
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
struct range {
    start: u64,
    end: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits124U0,
    pkru: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
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
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
}


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
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

#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
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
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
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
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
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
struct mutex {
    owner: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    osq: optimistic_spin_queue,
    first_waiter: *mut mutex_waiter,
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
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
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
struct rb_node {
    __rb_parent_color: u64,
    rb_right: *mut rb_node,
    rb_left: *mut rb_node,
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
struct rb_node_linked {
    node: rb_node,
    prev: *mut rb_node_linked,
    next: *mut rb_node_linked,
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
struct rb_root {
    rb_node: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
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
struct gendisk {
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
struct syscall_user_dispatch {
    selector: *mut u8,
    offset: u64,
    len: u64,
    on_dispatch: bool,
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
struct rwsem_waiter {
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
struct uprobe {
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
struct uprobe_xol_ops {
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
struct xol_area {
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
struct ldt_struct {
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
    __slate_bits_0: __slate_bits::__SlateBits293U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
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
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
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
    __slate_bits_0: __slate_bits::__SlateBits70U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits70U1,
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
    __slate_bits_2: __slate_bits::__SlateBits70U2,
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
// This thread should not be frozen
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
struct completion {
    done: u32,
    wait: swait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kioctx_table {
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
fn _msecs_to_jiffies(m: u32) -> u64 {
((m as i64).wrapping_add(1000 / 1000).wrapping_sub(1) / (1000 / 1000)) as u64
}

#[inline(always)]
fn msecs_to_jiffies(m: u32) -> u64 {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe(m) } {
if (m as i32) < 0 {
return 4611686018427387902u64;
}
return _msecs_to_jiffies(m);
} else {
return unsafe { __msecs_to_jiffies(m) };
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
struct kmem_cache {
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


fn queue_delayed_work(wq: *mut workqueue_struct, dwork: *mut delayed_work, delay: u64) -> bool {
unsafe { queue_delayed_work_on(64i32, wq, dwork, delay) }
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
struct poll_table_struct {
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
struct hprobe {
    state: u32,
    srcu_scp: *mut srcu_ctr,
    uprobe: *mut uprobe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
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
    __slate_bits_0: __slate_bits::__SlateBits415U0,
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
// MMF_HAS_PINNED: Whether this mm has pinned any pages.  This can be either
// replaced in the future by mm.pinned_vm when it becomes stable, or grow into
// a counter on its own. We're aggresive on this bit for now: even if the
// pinned pages were unpinned later on, we'll still keep this bit set for the
// lifecycle of this mm, just for simplicity.
// FOLL_PIN has run, never cleared
#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits447U0,
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
struct dev_pm_qos {
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
unsafe { __compiletime_assert_220() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("221: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 221b - ., 3; .popsection\n# {0}", const 221 as i32, options(nostack, att_syntax));
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
struct ida {
    xa: xarray,
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
    __slate_bits_0: __slate_bits::__SlateBits476U0,
    vm_ops: *const vm_operations_struct,
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
struct folio_queue {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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
    __slate_bits_0: __slate_bits::__SlateBits478U0,
}


fn kobject_name(kobj: *const kobject) -> *const u8 {
unsafe { (*kobj).name }
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
struct mnt_idmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_subsys_data {
    lock: spinlock,
    refcount: u32,
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
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits504U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits504U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits504U2,
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
    __slate_bits_3: __slate_bits::__SlateBits504U3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
struct bus_attribute {
    attr: attribute,
    show: Option<unsafe extern "C" fn(*const bus_type, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*const bus_type, *const u8, u64) -> i64>,
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
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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
struct rhash_head {
    next: *mut rhash_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iovec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
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
struct sighand_struct {
    siglock: spinlock,
    count: refcount_struct,
    signalfd_wqh: wait_queue_head,
    action: [k_sigaction; 64],
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
struct request_queue {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
    __slate_bits_0: __slate_bits::__SlateBits296U0,
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


// group exit in progress
fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
struct posix_acl {
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
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct path {
    mnt: *mut vfsmount,
    dentry: *mut dentry,
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
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_device_data {
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
union __SlateRecord_device_h_108 {
    show: Option<unsafe extern "C" fn(*mut device, *mut device_attribute, *mut u8) -> i64>,
    show_const: Option<unsafe extern "C" fn(*mut device, *const device_attribute, *mut u8) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_device_h_114 {
    store: Option<unsafe extern "C" fn(*mut device, *mut device_attribute, *const u8, u64) -> i64>,
    store_const: Option<unsafe extern "C" fn(*mut device, *const device_attribute, *const u8, u64) -> i64>,
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
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
union __SlateRecord_property_h_397 {
    pointer: *const (),
    value: __SlateRecord_property_h_399,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_property_h_399 {
    u8_data: [u8; 8],
    u16_data: [u16; 4],
    u32_data: [u32; 2],
    u64_data: [u64; 1],
    str: [*const u8; 1],
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
struct __SlateRecord_actypes_h_910 {
    r#type: u32,
    value: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_915 {
    r#type: u32,
    length: u32,
    pointer: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_921 {
    r#type: u32,
    length: u32,
    pointer: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_927 {
    r#type: u32,
    count: u32,
    elements: *mut acpi_object,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_933 {
    r#type: u32,
    actual_type: u32,
    handle: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_939 {
    r#type: u32,
    proc_id: u32,
    pblk_address: u64,
    pblk_length: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_946 {
    r#type: u32,
    system_level: u32,
    resource_order: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
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
struct __SlateRecord_acpi_bus_h_249 {
    __slate_bits_0: __slate_bits::__SlateBits784U0,
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
struct __SlateRecord_acpi_bus_h_281 {
    __slate_bits_0: __slate_bits::__SlateBits788U0,
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
struct acpi_gpio_mapping {
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
struct of_device_id {
    name: [u8; 32],
    r#type: [u8; 32],
    compatible: [u8; 128],
    data: *const (),
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
struct iova_bitmap {
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
struct device_attribute {
    attr: attribute,
    __slate_anon_1: __SlateRecord_device_h_108,
    __slate_anon_2: __SlateRecord_device_h_114,
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
struct iommu_dma_cookie {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_dma_msi_cookie {
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
struct iommufd_viommu {
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
    __slate_bits_0: __slate_bits::__SlateBits501U0,
}


fn device_can_wakeup(dev: *mut device) -> bool {
((unsafe { (*dev).power.__slate_bits_0.__get_can_wakeup() }) as i32) != 0
}

fn device_may_wakeup(dev: *mut device) -> bool {
((unsafe { (*dev).power.__slate_bits_0.__get_can_wakeup() }) as i32) != 0 && !unsafe { (*dev).power.wakeup }.is_null()
}

fn pm_wakeup_event(dev: *mut device, msec: u32) {
unsafe { pm_wakeup_dev_event(dev, msec, false) };
}

fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
}

fn device_is_registered(dev: *mut device) -> i32 {
(unsafe { (*dev).kobj.__slate_bits_0.__get_state_in_sysfs() }) as i32
}

fn device_enable_async_suspend(dev: *mut device) {
if ((unsafe { (*dev).power.__slate_bits_0.__get_is_prepared() }) as i32) == 0 {
unsafe {
let __t0 = true;
(*dev).power.__slate_bits_0.__set_async_suspend(__t0);
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_iommu_h_234 {
    iova_cookie: *mut iommu_dma_cookie,
    msi_cookie: *mut iommu_dma_msi_cookie,
    iommufd_hwpt: *mut iommufd_hw_pagetable,
    __slate_anon_3: __SlateRecord_iommu_h_238,
    __slate_anon_4: __SlateRecord_iommu_h_242,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommufd_hw_pagetable {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_h_238 {
    handler: Option<unsafe extern "C" fn(*mut iommu_domain, *mut device, u64, i32, *mut ()) -> i32>,
    handler_token: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_h_242 {
    mm: *mut mm_struct,
    users: i32,
    next: list_head,
}


fn device_lock(dev: *mut device) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*dev).mutex)) };
}

fn device_trylock(dev: *mut device) -> i32 {
unsafe { mutex_trylock(core::ptr::addr_of_mut!((*dev).mutex)) }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_iommu_h_364 {
    pgsize: u64,
    pt: __SlateRecord_iommu_h_370,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_iommu_h_370 {
    leaf_levels_bitmap: u8,
    table_levels_bitmap: u8,
}


fn device_unlock(dev: *mut device) {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*dev).mutex)) };
}

fn device_lock_assert(dev: *mut device) {
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*dev).mutex) });
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
struct software_node_ref_args {
    swnode: *const software_node,
    fwnode: *mut fwnode_handle,
    nargs: u32,
    args: [u64; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct property_entry {
    name: *const u8,
    length: u64,
    is_inline: bool,
    r#type: u32,
    __slate_anon_4: __SlateRecord_property_h_397,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct software_node {
    name: *const u8,
    parent: *const software_node,
    properties: *const property_entry,
}


// Fixed by ACPI spec
#[repr(C)]
#[derive(Clone, Copy)]
union acpi_object {
    r#type: u32,
    integer: __SlateRecord_actypes_h_910,
    string: __SlateRecord_actypes_h_915,
    buffer: __SlateRecord_actypes_h_921,
    package: __SlateRecord_actypes_h_927,
    reference: __SlateRecord_actypes_h_933,
    processor: __SlateRecord_actypes_h_939,
    power_resource: __SlateRecord_actypes_h_946,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_hotplug_profile {
    kobj: kobject,
    scan_dependent: Option<unsafe extern "C" fn(*mut acpi_device) -> i32>,
    notify_online: Option<unsafe extern "C" fn(*mut acpi_device)>,
    __slate_bits_0: __slate_bits::__SlateBits769U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_scan_handler {
    list_node: list_head,
    ids: *const acpi_device_id,
    r#match: Option<unsafe extern "C" fn(*const u8, *mut *const acpi_device_id) -> bool>,
    attach: Option<unsafe extern "C" fn(*mut acpi_device, *const acpi_device_id) -> i32>,
    detach: Option<unsafe extern "C" fn(*mut acpi_device)>,
    post_eject: Option<unsafe extern "C" fn(*mut acpi_device)>,
    bind: Option<unsafe extern "C" fn(*mut device)>,
    unbind: Option<unsafe extern "C" fn(*mut device)>,
    hotplug: acpi_hotplug_profile,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_hotplug_context {
    self_: *mut acpi_device,
    notify: Option<unsafe extern "C" fn(*mut acpi_device, u32) -> i32>,
    uevent: Option<unsafe extern "C" fn(*mut acpi_device, u32)>,
    fixup: Option<unsafe extern "C" fn(*mut acpi_device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_status {
    __slate_bits_0: __slate_bits::__SlateBits775U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_flags {
    __slate_bits_0: __slate_bits::__SlateBits776U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_dir {
    entry: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_pnp_type {
    __slate_bits_0: __slate_bits::__SlateBits780U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_hw_irq_h_82 {
    ioapic: ioapic_alloc_info,
    uv: uv_alloc_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_pnp {
    bus_id: [u8; 8],
    instance_no: i32,
    r#type: acpi_pnp_type,
    bus_address: u64,
    unique_id: *mut u8,
    ids: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_power_flags {
    __slate_bits_0: __slate_bits::__SlateBits782U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_power_state {
    resources: list_head,
    flags: __SlateRecord_acpi_bus_h_249,
    power: i32,
    latency: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_power {
    state: i32,
    flags: acpi_device_power_flags,
    states: [acpi_device_power_state; 5],
    state_for_enumeration: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_perf_flags {
    __slate_bits_0: __slate_bits::__SlateBits786U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_perf_state {
    flags: __SlateRecord_acpi_bus_h_281,
    power: u8,
    performance: u8,
    latency: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_perf {
    state: i32,
    flags: acpi_device_perf_flags,
    state_count: i32,
    states: *mut acpi_device_perf_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_wakeup_flags {
    __slate_bits_0: __slate_bits::__SlateBits790U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_26 {
    __slate_anon_0: __SlateRecord_msi_h_26_2,
    dmar_subhandle: u32,
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_26_2 {
    __slate_bits_0: __slate_bits::__SlateBits883U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_wakeup_context {
    func: Option<unsafe extern "C" fn(*mut acpi_device_wakeup_context)>,
    dev: *mut device,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_49 {
    __slate_anon_0: __SlateRecord_msi_h_49_2,
    __slate_anon_1: __SlateRecord_msi_h_49_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_49_2 {
    __slate_bits_0: __slate_bits::__SlateBits887U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_49_3 {
    __slate_bits_0: __slate_bits::__SlateBits888U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_wakeup {
    gpe_device: *mut (),
    gpe_number: u64,
    sleep_state: u64,
    resources: list_head,
    flags: acpi_device_wakeup_flags,
    context: acpi_device_wakeup_context,
    ws: *mut wakeup_source,
    prepare_count: i32,
    enable_count: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_62 {
    address_lo: u32,
    arch_addr_lo: x86_msi_addr_lo,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_66 {
    address_hi: u32,
    arch_addr_hi: x86_msi_addr_hi,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_70 {
    data: u32,
    arch_data: x86_msi_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_data {
    pointer: *const acpi_object,
    properties: list_head,
    of_compatible: *const acpi_object,
    subnodes: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_109 {
    msi_mask: u32,
    msix_ctrl: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_113 {
    __slate_bits_0: __slate_bits::__SlateBits897U0,
    default_irq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_122 {
    mask_pos: u8,
    mask_base: *mut (),
}


// The maximum expected number of CSI-2 data lanes.
//
// This number is not expected to ever have to be equal to or greater than the
// number of bits in an unsigned long variable, but if it needs to be increased
// above that limit, code will need to be adjusted accordingly.
#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_software_node_port {
    port_name: [u8; 9],
    data_lanes: [u32; 8],
    lane_polarities: [u32; 9],
    link_frequencies: [u64; 8],
    port_nr: u32,
    crs_csi2_local: bool,
    port_props: [property_entry; 2],
    ep_props: [property_entry; 8],
    remote_ep: [software_node_ref_args; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_software_nodes {
    dev_props: [property_entry; 6],
    nodes: *mut software_node,
    nodeptrs: *mut *const software_node,
    ports: *mut acpi_device_software_node_port,
    num_ports: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_202 {
    pci: pci_msi_desc,
    data: msi_desc_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device {
    pld_crc: u32,
    device_type: i32,
    handle: *mut (),
    fwnode: fwnode_handle,
    wakeup_list: list_head,
    del_list: list_head,
    status: acpi_device_status,
    flags: acpi_device_flags,
    pnp: acpi_device_pnp,
    power: acpi_device_power,
    wakeup: acpi_device_wakeup,
    performance: acpi_device_perf,
    dir: acpi_device_dir,
    data: acpi_device_data,
    handler: *mut acpi_scan_handler,
    hp: *mut acpi_hotplug_context,
    swnodes: *mut acpi_device_software_nodes,
    driver_gpios: *const acpi_gpio_mapping,
    driver_data: *mut (),
    dev: device,
    physical_node_count: u32,
    dep_unmet: u32,
    physical_node_list: list_head,
    physical_node_lock: mutex,
    remove: Option<unsafe extern "C" fn(*mut acpi_device)>,
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

fn usleep_range(min: u64, max: u64) {
unsafe { usleep_range_state(min, max, 2u32) };
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_interrupt_h_125 {
    dev_id: *mut (),
    percpu_dev_id: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dmi_strmatch {
    __slate_bits_0: __slate_bits::__SlateBits803U0,
    substr: [u8; 79],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dmi_system_id {
    callback: Option<unsafe extern "C" fn(*const dmi_system_id) -> i32>,
    ident: *const u8,
    matches: [dmi_strmatch; 4],
    driver_data: *mut (),
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
struct pcie_bwctrl_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pcie_link_state {
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
union __SlateRecord_pci_h_547 {
    sriov: *mut pci_sriov,
    physfn: *mut pci_dev,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gpio_desc {
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
struct of_phandle_args {
    np: *mut device_node,
    args_count: i32,
    args: [u32; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_fault_page_request {
    flags: u32,
    pasid: u32,
    grpid: u32,
    perm: u32,
    addr: u64,
    private_data: [u64; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_fault {
    r#type: u32,
    prm: iommu_fault_page_request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_page_response {
    pasid: u32,
    grpid: u32,
    code: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_vmalloc_h_53 {
    next: *mut vm_struct,
    llnode: llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iopf_fault {
    fault: iommu_fault,
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iopf_group {
    last_fault: iopf_fault,
    faults: list_head,
    fault_count: u64,
    pending_node: list_head,
    work: work_struct,
    attach_handle: *mut iommu_attach_handle,
    fault_param: *mut iommu_fault_param,
    node: list_head,
    cookie: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iopf_queue {
    wq: *mut workqueue_struct,
    devices: list_head,
    lock: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_domain_geometry {
    aperture_start: u64,
    aperture_end: u64,
    force_aperture: bool,
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


// Domain for use in DMA-API
// implementation
#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_domain {
    r#type: u32,
    cookie_type: u32,
    is_iommupt: bool,
    ops: *const iommu_domain_ops,
    dirty_ops: *const iommu_dirty_ops,
    owner: *const iommu_ops,
    pgsize_bitmap: u64,
    geometry: iommu_domain_geometry,
    iopf_handler: Option<unsafe extern "C" fn(*mut iopf_group) -> i32>,
    __slate_anon_9: __SlateRecord_iommu_h_234,
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
struct iommu_pages_list {
    pages: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_iotlb_gather {
    start: u64,
    end: u64,
    __slate_anon_2: __SlateRecord_iommu_h_364,
    freelist: iommu_pages_list,
    queued: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_dirty_bitmap {
    bitmap: *mut iova_bitmap,
    gather: *mut iommu_iotlb_gather,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_dirty_ops {
    set_dirty_tracking: Option<unsafe extern "C" fn(*mut iommu_domain, bool) -> i32>,
    read_and_clear_dirty: Option<unsafe extern "C" fn(*mut iommu_domain, u64, u64, u64, *mut iommu_dirty_bitmap) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_user_data {
    r#type: u32,
    uptr: *mut (),
    len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_user_data_array {
    r#type: u32,
    uptr: *mut (),
    entry_len: u64,
    entry_num: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_ops {
    capable: Option<unsafe extern "C" fn(*mut device, u32) -> bool>,
    hw_info: Option<unsafe extern "C" fn(*mut device, *mut u32, *mut u32) -> *mut ()>,
    domain_alloc_identity: Option<unsafe extern "C" fn(*mut device) -> *mut iommu_domain>,
    domain_alloc_paging_flags: Option<unsafe extern "C" fn(*mut device, u32, *const iommu_user_data) -> *mut iommu_domain>,
    domain_alloc_paging: Option<unsafe extern "C" fn(*mut device) -> *mut iommu_domain>,
    domain_alloc_sva: Option<unsafe extern "C" fn(*mut device, *mut mm_struct) -> *mut iommu_domain>,
    domain_alloc_nested: Option<unsafe extern "C" fn(*mut device, *mut iommu_domain, u32, *const iommu_user_data) -> *mut iommu_domain>,
    probe_device: Option<unsafe extern "C" fn(*mut device) -> *mut iommu_device>,
    release_device: Option<unsafe extern "C" fn(*mut device)>,
    probe_finalize: Option<unsafe extern "C" fn(*mut device)>,
    device_group: Option<unsafe extern "C" fn(*mut device) -> *mut iommu_group>,
    get_resv_regions: Option<unsafe extern "C" fn(*mut device, *mut list_head)>,
    of_xlate: Option<unsafe extern "C" fn(*mut device, *const of_phandle_args) -> i32>,
    is_attach_deferred: Option<unsafe extern "C" fn(*mut device) -> bool>,
    page_response: Option<unsafe extern "C" fn(*mut device, *mut iopf_fault, *mut iommu_page_response)>,
    def_domain_type: Option<unsafe extern "C" fn(*mut device) -> i32>,
    get_viommu_size: Option<unsafe extern "C" fn(*mut device, u32) -> u64>,
    viommu_init: Option<unsafe extern "C" fn(*mut iommufd_viommu, *mut iommu_domain, *const iommu_user_data) -> i32>,
    default_domain_ops: *const iommu_domain_ops,
    owner: *mut module,
    identity_domain: *mut iommu_domain,
    blocked_domain: *mut iommu_domain,
    release_domain: *mut iommu_domain,
    default_domain: *mut iommu_domain,
    __slate_bits_0: __slate_bits::__SlateBits800U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_taskset {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_domain_ops {
    attach_dev: Option<unsafe extern "C" fn(*mut iommu_domain, *mut device, *mut iommu_domain) -> i32>,
    set_dev_pasid: Option<unsafe extern "C" fn(*mut iommu_domain, *mut device, u32, *mut iommu_domain) -> i32>,
    map_pages: Option<unsafe extern "C" fn(*mut iommu_domain, u64, u64, u64, u64, i32, u32, *mut u64) -> i32>,
    unmap_pages: Option<unsafe extern "C" fn(*mut iommu_domain, u64, u64, u64, *mut iommu_iotlb_gather) -> u64>,
    flush_iotlb_all: Option<unsafe extern "C" fn(*mut iommu_domain)>,
    iotlb_sync_map: Option<unsafe extern "C" fn(*mut iommu_domain, u64, u64) -> i32>,
    iotlb_sync: Option<unsafe extern "C" fn(*mut iommu_domain, *mut iommu_iotlb_gather)>,
    cache_invalidate_user: Option<unsafe extern "C" fn(*mut iommu_domain, *mut iommu_user_data_array) -> i32>,
    iova_to_phys: Option<unsafe extern "C" fn(*mut iommu_domain, u64) -> u64>,
    enforce_cache_coherency: Option<unsafe extern "C" fn(*mut iommu_domain) -> bool>,
    set_pgtable_quirks: Option<unsafe extern "C" fn(*mut iommu_domain, u64) -> i32>,
    free: Option<unsafe extern "C" fn(*mut iommu_domain)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_device {
    list: list_head,
    ops: *const iommu_ops,
    fwnode: *mut fwnode_handle,
    dev: *mut device,
    singleton_group: *mut iommu_group,
    max_pasids: u32,
    ready: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_fault_param {
    lock: mutex,
    users: refcount_struct,
    rcu: callback_head,
    dev: *mut device,
    queue: *mut iopf_queue,
    queue_list: list_head,
    partial: list_head,
    faults: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_iommu {
    lock: mutex,
    fault_param: *mut iommu_fault_param,
    fwspec: *mut iommu_fwspec,
    iommu_dev: *mut iommu_device,
    r#priv: *mut (),
    max_pasids: u32,
    __slate_bits_0: __slate_bits::__SlateBits727U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_fwspec {
    iommu_fwnode: *mut fwnode_handle,
    flags: u32,
    num_ids: u32,
    ids: [u32; 0],
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
struct iommu_attach_handle {
    domain: *mut iommu_domain,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_mm_data {
    pasid: u32,
    mm: *mut mm_struct,
    sva_domains: list_head,
    mm_list_elm: list_head,
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
union msi_instance_cookie {
    value: u64,
    ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_common_data {
    state_use_accessors: u32,
    node: u32,
    handler_data: *mut (),
    msi_desc: *mut msi_desc,
    affinity: [cpumask; 1],
    effective_affinity: [cpumask; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_data {
    mask: u32,
    irq: u32,
    hwirq: u64,
    common: *mut irq_common_data,
    chip: *mut irq_chip,
    domain: *mut irq_domain,
    parent_data: *mut irq_data,
    chip_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip {
    name: *const u8,
    irq_startup: Option<unsafe extern "C" fn(*mut irq_data) -> u32>,
    irq_shutdown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_enable: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_disable: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_ack: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_mask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_mask_ack: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_unmask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_eoi: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_set_affinity: Option<unsafe extern "C" fn(*mut irq_data, *const cpumask, bool) -> i32>,
    irq_pre_redirect: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_retrigger: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_set_type: Option<unsafe extern "C" fn(*mut irq_data, u32) -> i32>,
    irq_set_wake: Option<unsafe extern "C" fn(*mut irq_data, u32) -> i32>,
    irq_bus_lock: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_bus_sync_unlock: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_suspend: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_resume: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_pm_shutdown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_calc_mask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_print_chip: Option<unsafe extern "C" fn(*mut irq_data, *mut seq_file)>,
    irq_request_resources: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_release_resources: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_compose_msi_msg: Option<unsafe extern "C" fn(*mut irq_data, *mut msi_msg)>,
    irq_write_msi_msg: Option<unsafe extern "C" fn(*mut irq_data, *mut msi_msg)>,
    irq_get_irqchip_state: Option<unsafe extern "C" fn(*mut irq_data, u32, *mut bool) -> i32>,
    irq_set_irqchip_state: Option<unsafe extern "C" fn(*mut irq_data, u32, bool) -> i32>,
    irq_set_vcpu_affinity: Option<unsafe extern "C" fn(*mut irq_data, *mut ()) -> i32>,
    ipi_send_single: Option<unsafe extern "C" fn(*mut irq_data, u32)>,
    ipi_send_mask: Option<unsafe extern "C" fn(*mut irq_data, *const cpumask)>,
    irq_nmi_setup: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_nmi_teardown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_force_complete_move: Option<unsafe extern "C" fn(*mut irq_data)>,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irqstat {
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pci_c_857 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pci_c_860 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_redirect {
    work: irq_work,
    target_cpu: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pci_c_5242 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pci_c_6101 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pci_c_6679 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pci_c_6679_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pci_c_6679_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pci_c_6679_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pci_c_6679_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pci_c_6679_6 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits618U0 {
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
    pub struct __SlateBits447U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits293U0 {
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
    pub struct __SlateBits70U0 {
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
    pub struct __SlateBits70U1 {
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
    pub struct __SlateBits415U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits476U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits967U0 {
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
    pub struct __SlateBits296U0 {
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
    pub struct __SlateBits70U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits105U0 {
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
    pub struct __SlateBits124U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits504U0 {
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
    pub struct __SlateBits501U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits504U1 {
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
    pub struct __SlateBits504U2 {
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
    pub struct __SlateBits504U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits887U0 {
        #[bits(2)]
        pub reserved_0: u32,
        #[bits(1)]
        pub dest_mode_logical: u32,
        #[bits(1)]
        pub redirect_hint: u32,
        #[bits(1)]
        pub reserved_1: u32,
        #[bits(7)]
        pub virt_destid_8_14: u32,
        #[bits(8)]
        pub destid_0_7: u32,
        #[bits(12)]
        pub base_address: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits888U0 {
        #[bits(2)]
        pub dmar_reserved_0: u32,
        #[bits(1)]
        pub dmar_index_15: u32,
        #[bits(1)]
        pub dmar_subhandle_valid: u32,
        #[bits(1)]
        pub dmar_format: u32,
        #[bits(15)]
        pub dmar_index_0_14: u32,
        #[bits(12)]
        pub dmar_base_address: u32,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits890U0 {
        #[bits(8)]
        pub reserved: u32,
        #[bits(24)]
        pub destid_8_31: u32,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits883U0 {
        #[bits(8)]
        pub vector: u32,
        #[bits(3)]
        pub delivery_mode: u32,
        #[bits(1)]
        pub dest_mode_logical: u32,
        #[bits(2)]
        pub reserved: u32,
        #[bits(1)]
        pub active_low: u32,
        #[bits(1)]
        pub is_level: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits858U0 {
        #[bits(1)]
        pub is_managed: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits897U0 {
        #[bits(1)]
        pub is_msix: u8,
        #[bits(3)]
        pub multiple: u8,
        #[bits(3)]
        pub multi_cap: u8,
        #[bits(1)]
        pub can_mask: u8,
        #[bits(1)]
        pub is_64: u8,
        #[bits(1)]
        pub is_virtual: u8,
        #[bits(6, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits868U0 {
        #[bits(1)]
        pub is_level: u32,
        #[bits(1)]
        pub active_low: u32,
        #[bits(1)]
        pub valid: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits800U0 {
        #[bits(1)]
        pub user_pasid_table: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits727U0 {
        #[bits(1)]
        pub attach_deferred: u32,
        #[bits(1)]
        pub pci_32bit_workaround: u32,
        #[bits(1)]
        pub require_direct: u32,
        #[bits(1)]
        pub shadow_on_flush: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits478U0 {
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
    pub struct __SlateBits104U0 {
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
    pub struct __SlateBits444U0 {
        #[bits(1)]
        pub is_added: u32,
        #[bits(1)]
        pub unsafe_warn: u32,
        #[bits(1)]
        pub flit_mode: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits909U0 {
        #[bits(1)]
        pub per_func_slot: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits801U0 {
        #[bits(3)]
        pub pcie_mpss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits801U1 {
        #[bits(5)]
        pub pme_support: u32,
        #[bits(1)]
        pub pme_poll: u32,
        #[bits(1)]
        pub pinned: u32,
        #[bits(1)]
        pub config_rrs_sv: u32,
        #[bits(1)]
        pub imm_ready: u32,
        #[bits(1)]
        pub d1_support: u32,
        #[bits(1)]
        pub d2_support: u32,
        #[bits(1)]
        pub no_d1d2: u32,
        #[bits(1)]
        pub no_d3cold: u32,
        #[bits(1)]
        pub bridge_d3: u32,
        #[bits(1)]
        pub d3cold_allowed: u32,
        #[bits(1)]
        pub mmio_always_on: u32,
        #[bits(1)]
        pub wakeup_prepared: u32,
        #[bits(1)]
        pub skip_bus_pm: u32,
        #[bits(1)]
        pub ignore_hotplug: u32,
        #[bits(1)]
        pub hotplug_user_indicators: u32,
        #[bits(1)]
        pub clear_retrain_link: u32,
        #[bits(1)]
        pub no_bw_notif: u32,
        #[bits(2, access = na)]
        pub __slate_pad_18: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits801U2 {
        #[bits(1)]
        pub aspm_l0s_support: u32,
        #[bits(1)]
        pub aspm_l1_support: u32,
        #[bits(1)]
        pub ltr_path: u32,
        #[bits(1)]
        pub pasid_no_tlp: u32,
        #[bits(3)]
        pub eetlp_prefix_max: u32,
        #[bits(1, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield([u8; 6], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits801U3 {
        #[bits(1)]
        pub transparent: u32,
        #[bits(1)]
        pub io_window: u32,
        #[bits(1)]
        pub pref_window: u32,
        #[bits(1)]
        pub pref_64_window: u32,
        #[bits(1)]
        pub multifunction: u32,
        #[bits(1)]
        pub is_busmaster: u32,
        #[bits(1)]
        pub no_msi: u32,
        #[bits(1)]
        pub block_cfg_access: u32,
        #[bits(1)]
        pub broken_parity_status: u32,
        #[bits(2)]
        pub irq_reroute_variant: u32,
        #[bits(1)]
        pub msi_enabled: u32,
        #[bits(1)]
        pub msix_enabled: u32,
        #[bits(1)]
        pub ari_enabled: u32,
        #[bits(1)]
        pub ats_enabled: u32,
        #[bits(1)]
        pub pasid_enabled: u32,
        #[bits(1)]
        pub pri_enabled: u32,
        #[bits(1)]
        pub tph_enabled: u32,
        #[bits(1)]
        pub fm_enabled: u32,
        #[bits(1)]
        pub is_managed: u32,
        #[bits(1)]
        pub is_msi_managed: u32,
        #[bits(1)]
        pub needs_freset: u32,
        #[bits(1)]
        pub state_saved: u32,
        #[bits(1)]
        pub is_physfn: u32,
        #[bits(1)]
        pub is_virtfn: u32,
        #[bits(1)]
        pub is_hotplug_bridge: u32,
        #[bits(1)]
        pub is_pciehp: u32,
        #[bits(1)]
        pub shpc_managed: u32,
        #[bits(1)]
        pub is_thunderbolt: u32,
        #[bits(1)]
        pub is_cxl: u32,
        #[bits(1)]
        pub untrusted: u32,
        #[bits(1)]
        pub external_facing: u32,
        #[bits(1)]
        pub broken_intx_masking: u32,
        #[bits(1)]
        pub io_window_1k: u32,
        #[bits(1)]
        pub irq_managed: u32,
        #[bits(1)]
        pub non_compliant_bars: u32,
        #[bits(1)]
        pub is_probed: u32,
        #[bits(1)]
        pub link_active_reporting: u32,
        #[bits(1)]
        pub no_vf_scan: u32,
        #[bits(1)]
        pub no_command_memory: u32,
        #[bits(1)]
        pub rom_bar_overlap: u32,
        #[bits(1)]
        pub rom_attr_enabled: u32,
        #[bits(1)]
        pub non_mappable_bars: u32,
        #[bits(5, access = na)]
        pub __slate_pad_42: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits801U4 {
        #[bits(1)]
        pub pasid_required: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits803U0 {
        #[bits(7)]
        pub slot: u8,
        #[bits(1)]
        pub exact_match: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits775U0 {
        #[bits(1)]
        pub present: u32,
        #[bits(1)]
        pub enabled: u32,
        #[bits(1)]
        pub show_in_ui: u32,
        #[bits(1)]
        pub functional: u32,
        #[bits(1)]
        pub battery_present: u32,
        #[bits(27)]
        pub reserved: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits776U0 {
        #[bits(1)]
        pub dynamic_status: u32,
        #[bits(1)]
        pub removable: u32,
        #[bits(1)]
        pub ejectable: u32,
        #[bits(1)]
        pub power_manageable: u32,
        #[bits(1)]
        pub initialized: u32,
        #[bits(1)]
        pub visited: u32,
        #[bits(1)]
        pub hotplug_notify: u32,
        #[bits(1)]
        pub is_dock_station: u32,
        #[bits(1)]
        pub of_compatible_ok: u32,
        #[bits(1)]
        pub coherent_dma: u32,
        #[bits(1)]
        pub cca_seen: u32,
        #[bits(1)]
        pub enumeration_by_parent: u32,
        #[bits(1)]
        pub honor_deps: u32,
        #[bits(19)]
        pub reserved: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits780U0 {
        #[bits(1)]
        pub hardware_id: u32,
        #[bits(1)]
        pub bus_address: u32,
        #[bits(1)]
        pub platform_id: u32,
        #[bits(1)]
        pub backlight: u32,
        #[bits(28)]
        pub reserved: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits782U0 {
        #[bits(1)]
        pub explicit_get: u32,
        #[bits(1)]
        pub power_resources: u32,
        #[bits(1)]
        pub inrush_current: u32,
        #[bits(1)]
        pub power_removed: u32,
        #[bits(1)]
        pub ignore_parent: u32,
        #[bits(1)]
        pub dsw_present: u32,
        #[bits(26)]
        pub reserved: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits784U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub explicit_set: u8,
        #[bits(6)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits790U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub notifier_present: u8,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits786U0 {
        #[bits(8)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits788U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(7)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits769U0 {
        #[bits(1)]
        pub enabled: bool,
        #[bits(1)]
        pub demand_offline: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits931U0 {
        #[bits(1)]
        pub ignore_reset_delay: u32,
        #[bits(1)]
        pub no_ext_tags: u32,
        #[bits(1)]
        pub no_inc_mrrs: u32,
        #[bits(1)]
        pub native_aer: u32,
        #[bits(1)]
        pub native_pcie_hotplug: u32,
        #[bits(1)]
        pub native_shpc_hotplug: u32,
        #[bits(1)]
        pub native_pme: u32,
        #[bits(1)]
        pub native_ltr: u32,
        #[bits(1)]
        pub native_dpc: u32,
        #[bits(1)]
        pub native_cxl_error: u32,
        #[bits(1)]
        pub preserve_config: u32,
        #[bits(1)]
        pub size_windows: u32,
        #[bits(1)]
        pub msi_domain: u32,
        #[bits(1)]
        pub broken_l1ss_resume: u32,
        #[bits(2, access = na)]
        pub __slate_pad_14: u8,
    }
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct irq_desc {
    irq_common_data: irq_common_data,
    irq_data: irq_data,
    kstat_irqs: *mut irqstat,
    handle_irq: Option<unsafe extern "C" fn(*mut irq_desc)>,
    action: *mut irqaction,
    status_use_accessors: u32,
    core_internal_state__do_not_mess_with_it: u32,
    depth: u32,
    wake_depth: u32,
    tot_count: u64,
    last_unhandled: u64,
    irq_count: u32,
    irqs_unhandled: u32,
    threads_handled: __SlateRecord_types_h_190,
    threads_handled_last: i32,
    lock: raw_spinlock,
    percpu_enabled: *mut cpumask,
    redirect: irq_redirect,
    affinity_hint: *const cpumask,
    affinity_notify: *mut irq_affinity_notify,
    pending_mask: [cpumask; 1],
    threads_oneshot: u64,
    threads_active: __SlateRecord_types_h_190,
    wait_for_threads: wait_queue_head,
    nr_actions: u32,
    no_suspend_depth: u32,
    cond_suspend_depth: u32,
    force_resume_depth: u32,
    dir: *mut proc_dir_entry,
    refcnt: __SlateRecord_types_h_202,
    rcu: callback_head,
    kobj: kobject,
    request_mutex: mutex,
    parent_irq: i32,
    owner: *mut module,
    name: *const u8,
    resend_node: hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ioapic_alloc_info {
    pin: i32,
    node: i32,
    __slate_bits_0: __slate_bits::__SlateBits868U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uv_alloc_info {
    limit: i32,
    blade: i32,
    offset: u64,
    name: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_alloc_info {
    r#type: u32,
    flags: u32,
    devid: u32,
    hwirq: u64,
    mask: *const cpumask,
    desc: *mut msi_desc,
    data: *mut (),
    __slate_anon_7: __SlateRecord_hw_irq_h_82,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_regs {
    enable: u64,
    disable: u64,
    mask: u64,
    ack: u64,
    eoi: u64,
    r#type: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_type {
    chip: irq_chip,
    regs: irq_chip_regs,
    handler: Option<unsafe extern "C" fn(*mut irq_desc)>,
    r#type: u32,
    mask_cache_priv: u32,
    mask_cache: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_generic {
    lock: raw_spinlock,
    reg_base: *mut (),
    reg_readl: Option<unsafe extern "C" fn(*mut ()) -> u32>,
    reg_writel: Option<unsafe extern "C" fn(u32, *mut ())>,
    suspend: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    resume: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    irq_base: u32,
    irq_cnt: u32,
    mask_cache: u32,
    wake_enabled: u32,
    wake_active: u32,
    num_ct: u32,
    private: *mut (),
    installed: u64,
    unused: u64,
    domain: *mut irq_domain,
    list: list_head,
    chip_types: [irq_chip_type; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_chip_generic {
    irqs_per_chip: u32,
    num_chips: u32,
    irq_flags_to_clear: u32,
    irq_flags_to_set: u32,
    gc_flags: u32,
    exit: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    gc: [*mut irq_chip_generic; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_fwspec {
    fwnode: *mut fwnode_handle,
    param_count: i32,
    param: [u32; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_fwspec_info {
    flags: u64,
    affinity: *const cpumask,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_ops {
    r#match: Option<unsafe extern "C" fn(*mut irq_domain, *mut device_node, u32) -> i32>,
    select: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, u32) -> i32>,
    map: Option<unsafe extern "C" fn(*mut irq_domain, u32, u64) -> i32>,
    unmap: Option<unsafe extern "C" fn(*mut irq_domain, u32)>,
    xlate: Option<unsafe extern "C" fn(*mut irq_domain, *mut device_node, *const u32, u32, *mut u64, *mut u32) -> i32>,
    alloc: Option<unsafe extern "C" fn(*mut irq_domain, u32, u32, *mut ()) -> i32>,
    free: Option<unsafe extern "C" fn(*mut irq_domain, u32, u32)>,
    activate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_data, bool) -> i32>,
    deactivate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_data)>,
    translate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, *mut u64, *mut u32) -> i32>,
    get_fwspec_info: Option<unsafe extern "C" fn(*mut irq_fwspec, *mut irq_fwspec_info) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
    link: list_head,
    name: *const u8,
    ops: *const irq_domain_ops,
    host_data: *mut (),
    flags: u32,
    mapcount: u32,
    mutex: mutex,
    root: *mut irq_domain,
    fwnode: *mut fwnode_handle,
    bus_token: u32,
    gc: *mut irq_domain_chip_generic,
    dev: *mut device,
    pm_dev: *mut device,
    parent: *mut irq_domain,
    msi_parent_ops: *const msi_parent_ops,
    exit: Option<unsafe extern "C" fn(*mut irq_domain)>,
    hwirq_max: u64,
    revmap_size: u32,
    revmap_tree: xarray,
    revmap: [*mut irq_data; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct x86_msi_data {
    __slate_anon_0: core::mem::MaybeUninit<[u8; 4]>,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct x86_msi_addr_lo {
    __slate_anon_0: __SlateRecord_msi_h_49,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_msi_addr_hi {
    __slate_bits_0: __slate_bits::__SlateBits890U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_msg {
    __slate_anon_0: __SlateRecord_msi_h_62,
    __slate_anon_1: __SlateRecord_msi_h_66,
    __slate_anon_2: __SlateRecord_msi_h_70,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_msi_desc {
    __slate_anon_0: __SlateRecord_msi_h_109,
    msi_attrib: __SlateRecord_msi_h_113,
    __slate_anon_2: __SlateRecord_msi_h_122,
}


#[repr(C)]
#[derive(Clone, Copy)]
union msi_domain_cookie {
    value: u64,
    ptr: *mut (),
    iobase: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_desc_data {
    dcookie: msi_domain_cookie,
    icookie: msi_instance_cookie,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_desc {
    irq: u32,
    nvec_used: u32,
    dev: *mut device,
    msg: msi_msg,
    affinity: *mut irq_affinity_desc,
    sysfs_attrs: *mut device_attribute,
    write_msi_msg: Option<unsafe extern "C" fn(*mut msi_desc, *mut ())>,
    write_msi_msg_data: *mut (),
    msi_index: u16,
    __slate_anon_9: __SlateRecord_msi_h_202,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_domain_ops {
    get_hwirq: Option<unsafe extern "C" fn(*mut msi_domain_info, *mut irq_alloc_info) -> u64>,
    msi_init: Option<unsafe extern "C" fn(*mut irq_domain, *mut msi_domain_info, u32, u64, *mut irq_alloc_info) -> i32>,
    msi_free: Option<unsafe extern "C" fn(*mut irq_domain, *mut msi_domain_info, u32)>,
    msi_prepare: Option<unsafe extern "C" fn(*mut irq_domain, *mut device, i32, *mut irq_alloc_info) -> i32>,
    msi_teardown: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_alloc_info)>,
    prepare_desc: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_alloc_info, *mut msi_desc)>,
    set_desc: Option<unsafe extern "C" fn(*mut irq_alloc_info, *mut msi_desc)>,
    domain_alloc_irqs: Option<unsafe extern "C" fn(*mut irq_domain, *mut device, i32) -> i32>,
    domain_free_irqs: Option<unsafe extern "C" fn(*mut irq_domain, *mut device)>,
    msi_translate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, *mut u64, *mut u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_domain_info {
    flags: u32,
    bus_token: u32,
    hwsize: u32,
    ops: *mut msi_domain_ops,
    dev: *mut device,
    chip: *mut irq_chip,
    chip_data: *mut (),
    handler: Option<unsafe extern "C" fn(*mut irq_desc)>,
    handler_data: *mut (),
    handler_name: *const u8,
    alloc_data: *mut irq_alloc_info,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_parent_ops {
    supported_flags: u32,
    required_flags: u32,
    chip_flags: u32,
    bus_select_token: u32,
    bus_select_mask: u32,
    prefix: *const u8,
    init_dev_msi_info: Option<unsafe extern "C" fn(*mut device, *mut irq_domain, *mut irq_domain, *mut msi_domain_info) -> bool>,
}


fn pci_configure_of_wake_gpio(dev: *mut pci_dev) {
}

fn pci_remove_of_wake_gpio(dev: *mut pci_dev) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_device_id {
    vendor: u32,
    device: u32,
    subvendor: u32,
    subdevice: u32,
    class: u32,
    class_mask: u32,
    driver_data: u64,
    override_only: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct irqaction {
    handler: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>,
    __slate_anon_1: __SlateRecord_interrupt_h_125,
    affinity: *const cpumask,
    next: *mut irqaction,
    thread_fn: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>,
    thread: *mut task_struct,
    secondary: *mut irqaction,
    irq: u32,
    flags: u32,
    thread_flags: u64,
    thread_mask: u64,
    name: *const u8,
    dir: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_affinity_notify {
    irq: u32,
    kref: kref,
    work: work_struct,
    notify: Option<unsafe extern "C" fn(*mut irq_affinity_notify, *const cpumask)>,
    release: Option<unsafe extern "C" fn(*mut kref)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_affinity_desc {
    mask: cpumask,
    __slate_bits_0: __slate_bits::__SlateBits858U0,
}


// The PCI interface treats multi-function devices as independent
// devices.  The slot/function address of each device is encoded
// in a single byte as follows:
//
// 	7:3 = slot
// 	2:0 = function
//
// PCI_DEVFN(), PCI_SLOT(), and PCI_FUNC() are defined in uapi/linux/pci.h.
// In the interest of not exposing interfaces to user-space unnecessarily,
// the following kernel-only defines are being added here.
#[repr(C)]
#[derive(Clone, Copy)]
struct pci_slot {
    bus: *mut pci_bus,
    list: list_head,
    hotplug: *mut hotplug_slot,
    number: u16,
    __slate_bits_0: __slate_bits::__SlateBits909U0,
    kobj: kobject,
}


#[unsafe(no_mangle)]
static mut pci_power_names: __SlateAlign16<[*const u8; 7]> = __SlateAlign16([(b"error\0".as_ptr() as *mut u8) as *const u8, (b"D0\0".as_ptr() as *mut u8) as *const u8, (b"D1\0".as_ptr() as *mut u8) as *const u8, (b"D2\0".as_ptr() as *mut u8) as *const u8, (b"D3hot\0".as_ptr() as *mut u8) as *const u8, (b"D3cold\0".as_ptr() as *mut u8) as *const u8, (b"unknown\0".as_ptr() as *mut u8) as *const u8]);

fn pci_power_name(state: i32) -> *const u8 {
unsafe { *(core::ptr::addr_of_mut!(pci_power_names.0) as *mut *const u8).wrapping_offset(1i32.wrapping_add(state) as isize) }
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pcie_get_width_cap")]
extern "C" fn pcie_get_width_cap(dev: *mut pci_dev) -> u32 {
let mut lnkcap: u32 = 0;
unsafe { pcie_capability_read_dword(dev, 12i32, core::ptr::addr_of_mut!(lnkcap)) };
if lnkcap != 0 {
let mut __v0: i32;
loop {
if __bf_cast_unsigned!(1008i32) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_889() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: i32;
loop {
if false {
unsafe { __compiletime_assert_890() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_891() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(1008i32 >> (1008u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_892() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 1008u64.wrapping_add(1u64 << (1008u64.trailing_zeros() as i32)) & 1008u64.wrapping_add(1u64 << (1008u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_893() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((lnkcap & 1008) >> (1008u64.trailing_zeros() as i32)) as i32;
__v0 = __v2;
return __v0 as u32;
}
255
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_vpd {
    lock: mutex,
    len: u32,
    cap: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dev {
    bus_list: list_head,
    bus: *mut pci_bus,
    subordinate: *mut pci_bus,
    sysdata: *mut (),
    procent: *mut proc_dir_entry,
    slot: *mut pci_slot,
    devfn: u32,
    vendor: u16,
    device: u16,
    subsystem_vendor: u16,
    subsystem_device: u16,
    class: u32,
    revision: u8,
    hdr_type: u8,
    rcec_ea: *mut rcec_ea,
    rcec: *mut pci_dev,
    devcap: u32,
    rebar_cap: u16,
    pcie_cap: u8,
    msi_cap: u8,
    msix_cap: u8,
    __slate_bits_0: __slate_bits::__SlateBits801U0,
    rom_base_reg: u8,
    pin: u8,
    pcie_flags_reg: u16,
    dma_alias_mask: *mut u64,
    driver: *mut pci_driver,
    dma_mask: u64,
    msi_addr_mask: u64,
    dma_parms: device_dma_parameters,
    current_state: i32,
    pm_cap: u8,
    __slate_bits_1: __slate_bits::__SlateBits801U1,
    d3hot_delay: u32,
    d3cold_delay: u32,
    l1ss: u16,
    link_state: *mut pcie_link_state,
    __slate_bits_2: __slate_bits::__SlateBits801U2,
    error_state: u32,
    dev: device,
    cfg_size: i32,
    irq: u32,
    resource: [resource; 11],
    driver_exclusive_resource: resource,
    __slate_bits_3: __slate_bits::__SlateBits801U3,
    dev_flags: u16,
    enable_cnt: __SlateRecord_types_h_190,
    pcie_cap_lock: spinlock,
    saved_config_space: [u32; 16],
    saved_cap_space: hlist_head,
    msix_base: *mut (),
    msi_lock: raw_spinlock,
    vpd: pci_vpd,
    link_bwctrl: *mut pcie_bwctrl_data,
    __slate_anon_116: __SlateRecord_pci_h_547,
    ats_cap: u16,
    ats_stu: u8,
    pri_cap: u16,
    pri_reqs_alloc: u32,
    __slate_bits_4: __slate_bits::__SlateBits801U4,
    pasid_cap: u16,
    pasid_features: u16,
    acs_cap: u16,
    acs_capabilities: u16,
    supported_speeds: u8,
    rom: u64,
    romlen: u64,
    priv_flags: u64,
    reset_methods: [u8; 8],
    wake: *mut gpio_desc,
}


fn pci_physfn(dev: *mut pci_dev) -> *mut pci_dev {
dev
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct pci_host_bridge {
    dev: device,
    bus: *mut pci_bus,
    ops: *mut pci_ops,
    child_ops: *mut pci_ops,
    sysdata: *mut (),
    busnr: i32,
    domain_nr: i32,
    windows: list_head,
    dma_ranges: list_head,
    ports: list_head,
    swizzle_irq: Option<unsafe extern "C" fn(*mut pci_dev, *mut u8) -> u8>,
    map_irq: Option<unsafe extern "C" fn(*const pci_dev, u8, u8) -> i32>,
    release_fn: Option<unsafe extern "C" fn(*mut pci_host_bridge)>,
    enable_device: Option<unsafe extern "C" fn(*mut pci_host_bridge, *mut pci_dev) -> i32>,
    disable_device: Option<unsafe extern "C" fn(*mut pci_host_bridge, *mut pci_dev)>,
    reset_root_port: Option<unsafe extern "C" fn(*mut pci_host_bridge, *mut pci_dev) -> i32>,
    release_data: *mut (),
    __slate_bits_0: __slate_bits::__SlateBits931U0,
    align_resource: Option<unsafe extern "C" fn(*mut pci_dev, *const resource, u64, u64, u64) -> u64>,
    __slate_pad_19: [u8; 40],
    private: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_bus {
    node: list_head,
    parent: *mut pci_bus,
    children: list_head,
    devices: list_head,
    self_: *mut pci_dev,
    slots: list_head,
    resource: [*mut resource; 4],
    resources: list_head,
    busn_res: resource,
    ops: *mut pci_ops,
    sysdata: *mut (),
    procdir: *mut proc_dir_entry,
    number: u8,
    primary: u8,
    max_bus_speed: u8,
    cur_bus_speed: u8,
    name: [u8; 48],
    bridge_ctl: u16,
    bus_flags: u16,
    bridge: *mut device,
    dev: device,
    __slate_bits_0: __slate_bits::__SlateBits444U0,
}


fn pci_is_root_bus(pbus: *mut pci_bus) -> bool {
unsafe { (*pbus).parent }.is_null()
}

fn pci_is_bridge(dev: *mut pci_dev) -> bool {
((unsafe { (*dev).hdr_type }) as i32) == 1 || ((unsafe { (*dev).hdr_type }) as i32) == 2
}

fn pcie_is_cxl(pci_dev: *mut pci_dev) -> bool {
((unsafe { (*pci_dev).__slate_bits_3.__get_is_cxl() }) as i32) != 0
}

fn pci_upstream_bridge(mut dev: *mut pci_dev) -> *mut pci_dev {
dev = pci_physfn(dev);
if pci_is_root_bus(unsafe { (*dev).bus }) {
return core::ptr::null_mut::<pci_dev>();
}
unsafe { (*(*dev).bus).self_ }
}

// Error values that may be returned by PCI functions
fn pcibios_err_to_errno(err: i32) -> i32 {
if err <= 0 {
return err;
}
// Assume already errno
match err {
129 => {
return 2i32.wrapping_neg();
}
131 => {
return 25i32.wrapping_neg();
}
134 => {
return 19i32.wrapping_neg();
}
135 => {
return 14i32.wrapping_neg();
}
136 => {
return 5i32.wrapping_neg();
}
137 => {
return 28i32.wrapping_neg();
}
_ => {
}
}
34i32.wrapping_neg()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_ops {
    add_bus: Option<unsafe extern "C" fn(*mut pci_bus) -> i32>,
    remove_bus: Option<unsafe extern "C" fn(*mut pci_bus)>,
    map_bus: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32) -> *mut ()>,
    read: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32, i32, *mut u32) -> i32>,
    write: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32, i32, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dynids {
    lock: spinlock,
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_error_handlers {
    error_detected: Option<unsafe extern "C" fn(*mut pci_dev, u32) -> u32>,
    mmio_enabled: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    slot_reset: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    reset_prepare: Option<unsafe extern "C" fn(*mut pci_dev)>,
    reset_done: Option<unsafe extern "C" fn(*mut pci_dev)>,
    resume: Option<unsafe extern "C" fn(*mut pci_dev)>,
    cor_error_detected: Option<unsafe extern "C" fn(*mut pci_dev)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_driver {
    name: *const u8,
    id_table: *const pci_device_id,
    probe: Option<unsafe extern "C" fn(*mut pci_dev, *const pci_device_id) -> i32>,
    remove: Option<unsafe extern "C" fn(*mut pci_dev)>,
    suspend: Option<unsafe extern "C" fn(*mut pci_dev, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut pci_dev) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut pci_dev)>,
    sriov_configure: Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>,
    sriov_set_msix_vec_count: Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>,
    sriov_get_vf_total_msix: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    err_handler: *const pci_error_handlers,
    groups: *mut *const attribute_group,
    dev_groups: *mut *const attribute_group,
    driver: device_driver,
    dynids: pci_dynids,
    driver_managed_dma: bool,
}


// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
fn pci_add_flags(flags: i32) {
unsafe {
pci_flags = pci_flags | (flags as u32);
}
}

fn pci_has_flag(flag: i32) -> i32 {
((unsafe { pci_flags }) & (flag as u32)) as i32
}

#[inline(always)]
fn __free_pci_dev_put(p: *mut ()) {
let _T: *mut pci_dev = unsafe { *(p as *mut *mut pci_dev) };
if !_T.is_null() {
unsafe { pci_dev_put(_T) };
}
}

fn pcie_capability_clear_and_set_word(dev: *mut pci_dev, pos: i32, clear: u16, set: u16) -> i32 {
match pos {
16 | 28 | 48 => {
return unsafe { pcie_capability_clear_and_set_word_locked(dev, pos, clear, set) };
}
_ => {
return unsafe { pcie_capability_clear_and_set_word_unlocked(dev, pos, clear, set) };
}
}
unsafe { core::mem::zeroed() }
}

fn pcie_capability_set_word(dev: *mut pci_dev, pos: i32, set: u16) -> i32 {
pcie_capability_clear_and_set_word(dev, pos, 0u16, set)
}

fn pcie_capability_clear_word(dev: *mut pci_dev, pos: i32, clear: u16) -> i32 {
pcie_capability_clear_and_set_word(dev, pos, clear, 0u16)
}

fn pci_is_enabled(pdev: *mut pci_dev) -> i32 {
(atomic_read((unsafe { core::ptr::addr_of_mut!((*pdev).enable_cnt) }) as *const __SlateRecord_types_h_190) > 0) as i32
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pcix_get_max_mmrbc")]
extern "C" fn pcix_get_max_mmrbc(dev: *mut pci_dev) -> i32 {
let cap: i32 = pci_find_capability(dev, 7i32) as i32;
let mut stat: u32 = 0;
if cap == 0 {
return 22i32.wrapping_neg();
}
if (unsafe { pci_read_config_dword(dev as *const pci_dev, cap.wrapping_add(4), core::ptr::addr_of_mut!(stat)) }) != 0 {
return 22i32.wrapping_neg();
}
let mut __v0: i32;
loop {
if __bf_cast_unsigned!(6291456i32) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_824() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: i32;
loop {
if false {
unsafe { __compiletime_assert_825() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_826() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(6291456i32 >> (6291456u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_827() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 6291456u64.wrapping_add(1u64 << (6291456u64.trailing_zeros() as i32)) & 6291456u64.wrapping_add(1u64 << (6291456u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_828() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((stat & 6291456) >> (6291456u64.trailing_zeros() as i32)) as i32;
__v0 = __v2;
512i32 << __v0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pcix_get_mmrbc")]
extern "C" fn pcix_get_mmrbc(dev: *mut pci_dev) -> i32 {
let cap: i32 = pci_find_capability(dev, 7i32) as i32;
let mut cmd: u16 = 0;
if cap == 0 {
return 22i32.wrapping_neg();
}
if (unsafe { pci_read_config_word(dev as *const pci_dev, cap.wrapping_add(2), core::ptr::addr_of_mut!(cmd)) }) != 0 {
return 22i32.wrapping_neg();
}
let mut __v0: i32;
loop {
if __bf_cast_unsigned!(12i32) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u32) {
unsafe { __compiletime_assert_830() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: i32;
loop {
if false {
unsafe { __compiletime_assert_831() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_832() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(12i32 >> (12u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_833() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 12u64.wrapping_add(1u64 << (12u64.trailing_zeros() as i32)) & 12u64.wrapping_add(1u64 << (12u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_834() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((cmd as i32) & 12) >> (12u64.trailing_zeros() as i32);
__v0 = __v2;
512i32 << __v0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pcix_set_mmrbc")]
extern "C" fn pcix_set_mmrbc(dev: *mut pci_dev, mmrbc: i32) -> i32 {
let mut cap: i32 = 0;
let mut stat: u32 = 0;
let mut v: u32 = 0;
let mut o: u32 = 0;
let mut cmd: u16 = 0;
if mmrbc < 512 || mmrbc > 4096 || !is_power_of_2(mmrbc as u64) {
return 22i32.wrapping_neg();
}
v = ffs!(mmrbc).wrapping_sub(10) as u32;
cap = pci_find_capability(dev, 7i32) as i32;
if cap == 0 {
return 22i32.wrapping_neg();
}
if (unsafe { pci_read_config_dword(dev as *const pci_dev, cap.wrapping_add(4), core::ptr::addr_of_mut!(stat)) }) != 0 {
return 22i32.wrapping_neg();
}
let mut __v0: i32;
loop {
if __bf_cast_unsigned!(6291456i32) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_836() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: i32;
loop {
if false {
unsafe { __compiletime_assert_837() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_838() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(6291456i32 >> (6291456u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_839() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 6291456u64.wrapping_add(1u64 << (6291456u64.trailing_zeros() as i32)) & 6291456u64.wrapping_add(1u64 << (6291456u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_840() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((stat & 6291456) >> (6291456u64.trailing_zeros() as i32)) as i32;
__v0 = __v2;
if v > (__v0 as u32) {
return 7i32.wrapping_neg();
}
if (unsafe { pci_read_config_word(dev as *const pci_dev, cap.wrapping_add(2), core::ptr::addr_of_mut!(cmd)) }) != 0 {
return 22i32.wrapping_neg();
}
let mut __v7: i32;
loop {
if __bf_cast_unsigned!(12i32) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u32) {
unsafe { __compiletime_assert_841() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v9: i32;
loop {
if false {
unsafe { __compiletime_assert_842() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_843() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(12i32 >> (12u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_844() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 12u64.wrapping_add(1u64 << (12u64.trailing_zeros() as i32)) & 12u64.wrapping_add(1u64 << (12u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_845() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v9 = ((cmd as i32) & 12) >> (12u64.trailing_zeros() as i32);
__v7 = __v9;
o = __v7 as u32;
if o != v {
if v > o && ((unsafe { (*(*dev).bus).bus_flags }) as i32) & 2 != 0 {
return 5i32.wrapping_neg();
}
let mut __v14: u16;
typecheck_pointer__2!();
loop {
if false {
unsafe { __compiletime_assert_846() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_847() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if unsafe { __slate_intrinsic_1c5364fcd9da51fe(v) } { (!(12i32 >> (12u64.trailing_zeros() as i32)) as u32) & 0u32.wrapping_add(v) } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_848() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 12u64.wrapping_add(1u64 << (12u64.trailing_zeros() as i32)) & 12u64.wrapping_add(1u64 << (12u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_849() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __bf_cast_unsigned!(12i32) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u32) {
unsafe { __compiletime_assert_850() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let __v21: *mut u16 = core::ptr::addr_of_mut!(cmd);
let __v23: u16 = (((unsafe { *__v21 }) as i32) & !12) as u16;
unsafe {
*__v21 = __v23;
}
let __v24: *mut u16 = core::ptr::addr_of_mut!(cmd);
let __v26: u16 = (((unsafe { *__v24 }) as i32) | (v as i32) << (12u64.trailing_zeros() as i32) & 12) as u16;
unsafe {
*__v24 = __v26;
}
__v14 = __v26;
if (unsafe { pci_write_config_word(dev as *const pci_dev, cap.wrapping_add(2), cmd) }) != 0 {
return 5i32.wrapping_neg();
}
}
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pcie_get_readrq")]
extern "C" fn pcie_get_readrq(dev: *mut pci_dev) -> i32 {
let mut ctl: u16 = 0;
unsafe { pcie_capability_read_word(dev, 8i32, core::ptr::addr_of_mut!(ctl)) };
let mut __v0: i32;
loop {
if __bf_cast_unsigned!(28672i32) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u32) {
unsafe { __compiletime_assert_852() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: i32;
loop {
if false {
unsafe { __compiletime_assert_853() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_854() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(28672i32 >> (28672u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_855() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 28672u64.wrapping_add(1u64 << (28672u64.trailing_zeros() as i32)) & 28672u64.wrapping_add(1u64 << (28672u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_856() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((ctl as i32) & 28672) >> (28672u64.trailing_zeros() as i32);
__v0 = __v2;
128i32 << __v0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pcie_set_readrq")]
extern "C" fn pcie_set_readrq(dev: *mut pci_dev, mut rq: i32) -> i32 {
let mut v: u16 = 0;
let mut ret: i32 = 0;
let mut firstbit: u32 = 0;
let bridge: *mut pci_host_bridge = unsafe { pci_find_host_bridge((*dev).bus) };
if rq < 128 || rq > 4096 || !is_power_of_2(rq as u64) {
return 22i32.wrapping_neg();
}
// If using the "performance" PCIe config, we clamp the read rq
// size to the max packet size to keep the host bridge from
// generating requests larger than we can cope with.
if ((unsafe { pcie_bus_config }) as u32) == 3 {
let mps: i32 = pcie_get_mps(dev);
if mps < rq {
rq = mps;
}
}
firstbit = ffs!(rq) as u32;
if firstbit < 8 {
return 22i32.wrapping_neg();
}
let mut __v0: i32;
loop {
if (__bf_cast_unsigned!(28672i32) as u64) > __bf_cast_unsigned__3!(18446744073709551615u64) {
unsafe { __compiletime_assert_858() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: i32;
loop {
if false {
unsafe { __compiletime_assert_859() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_860() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if unsafe { __slate_intrinsic_1c5364fcd9da51fe(firstbit.wrapping_sub(8)) } { (!(28672i32 >> (28672u64.trailing_zeros() as i32)) as u32) & 0u32.wrapping_add(firstbit.wrapping_sub(8)) } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_861() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 28672u64.wrapping_add(1u64 << (28672u64.trailing_zeros() as i32)) & 28672u64.wrapping_add(1u64 << (28672u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_862() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = (firstbit.wrapping_sub(8) as i32) << (28672u64.trailing_zeros() as i32) & 28672;
__v0 = __v2;
v = __v0 as u16;
if ((unsafe { (*bridge).__slate_bits_0.__get_no_inc_mrrs() }) as i32) != 0 {
let max_mrrs: i32 = pcie_get_readrq(dev);
if rq > max_mrrs {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"can't set Max_Read_Request_Size to %d; max is %d\n".as_ptr() as *const u8, rq, max_mrrs) };
return 22i32.wrapping_neg();
}
}
ret = pcie_capability_clear_and_set_word(dev, 8i32, 28672u16, v);
pcibios_err_to_errno(ret)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pcie_get_mps")]
extern "C" fn pcie_get_mps(dev: *mut pci_dev) -> i32 {
let mut ctl: u16 = 0;
unsafe { pcie_capability_read_word(dev, 8i32, core::ptr::addr_of_mut!(ctl)) };
let mut __v0: i32;
loop {
if __bf_cast_unsigned!(224i32) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u32) {
unsafe { __compiletime_assert_864() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: i32;
loop {
if false {
unsafe { __compiletime_assert_865() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_866() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(224i32 >> (224u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_867() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 224u64.wrapping_add(1u64 << (224u64.trailing_zeros() as i32)) & 224u64.wrapping_add(1u64 << (224u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_868() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((ctl as i32) & 224) >> (224u64.trailing_zeros() as i32);
__v0 = __v2;
128i32 << __v0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pcie_set_mps")]
extern "C" fn pcie_set_mps(dev: *mut pci_dev, mps: i32) -> i32 {
let mut v: u16 = 0;
let mut ret: i32 = 0;
if mps < 128 || mps > 4096 || !is_power_of_2(mps as u64) {
return 22i32.wrapping_neg();
}
v = ffs!(mps).wrapping_sub(8) as u16;
if (v as i32) > ((unsafe { (*dev).__slate_bits_0.__get_pcie_mpss() }) as i32) {
return 22i32.wrapping_neg();
}
let mut __v0: i32;
loop {
if (__bf_cast_unsigned!(224i32) as u64) > __bf_cast_unsigned__3!(18446744073709551615u64) {
unsafe { __compiletime_assert_870() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: i32;
loop {
if false {
unsafe { __compiletime_assert_871() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_872() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if unsafe { __slate_intrinsic_b52e897e44daaa8(v) } { !(224i32 >> (224u64.trailing_zeros() as i32)) & 0i32.wrapping_add(v as i32) } else { 0i32 }) != 0 {
unsafe { __compiletime_assert_873() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 224u64.wrapping_add(1u64 << (224u64.trailing_zeros() as i32)) & 224u64.wrapping_add(1u64 << (224u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_874() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = (v as i32) << (224u64.trailing_zeros() as i32) & 224;
__v0 = __v2;
v = __v0 as u16;
ret = pcie_capability_clear_and_set_word(dev, 8i32, 224u16, v);
pcibios_err_to_errno(ret)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pcie_bandwidth_available")]
extern "C" fn pcie_bandwidth_available(mut dev: *mut pci_dev, limiting_dev: *mut *mut pci_dev, speed: *mut u32, width: *mut u32) -> u32 {
let mut lnksta: u16 = 0;
let mut next_speed: u32 = 0;
let mut next_width: u32 = 0;
let mut bw: u32 = 0;
let mut next_bw: u32 = 0;
if !speed.is_null() {
unsafe {
*speed = 255;
}
}
if !width.is_null() {
unsafe {
*width = 255;
}
}
while !dev.is_null() {
unsafe { pcie_capability_read_word(dev, 18i32, core::ptr::addr_of_mut!(lnksta)) };
next_speed = to_pcie_link_speed(lnksta);
let mut __v0: i32;
loop {
if __bf_cast_unsigned!(1008i32) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u32) {
unsafe { __compiletime_assert_882() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: i32;
loop {
if false {
unsafe { __compiletime_assert_883() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_884() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(1008i32 >> (1008u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_885() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 1008u64.wrapping_add(1u64 << (1008u64.trailing_zeros() as i32)) & 1008u64.wrapping_add(1u64 << (1008u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_886() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((lnksta as i32) & 1008) >> (1008u64.trailing_zeros() as i32);
__v0 = __v2;
next_width = __v0 as u32;
next_bw = (next_width as u32).wrapping_mul(PCIE_SPEED2MBS_ENC!(next_speed) as u32);
// Check if current device limits the total bandwidth
if bw == 0 || next_bw <= bw {
bw = next_bw;
if !limiting_dev.is_null() {
unsafe {
*limiting_dev = dev;
}
}
if !speed.is_null() {
unsafe {
*speed = next_speed;
}
}
if !width.is_null() {
unsafe {
*width = next_width;
}
}
}
dev = pci_upstream_bridge(dev);
}
bw
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_bridge_secondary_bus_reset")]
extern "C" fn pci_bridge_secondary_bus_reset(dev: *mut pci_dev) -> i32 {
if ((unsafe { (*dev).__slate_bits_3.__get_block_cfg_access() }) as i32) == 0 {
loop {
if !unsafe { __print_once } {
unsafe {
__print_once = true;
}
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"unlocked secondary bus reset via: %pS\n".as_ptr() as *const u8, __slate_intrinsic_ded315013ea6523(const { ((0 as i32) as u32) as i32 })) };
}
if true {
break;
}
}
}
pcibios_reset_secondary_bus(dev);
pci_bridge_wait_for_secondary_bus(dev, c"bus reset".as_ptr() as *mut u8)
}

// pci_bus_for_each_resource - iterate over PCI bus resources
// @bus: the PCI bus
// @res: pointer to the current resource
// @...: optional index of the current resource
//
// Iterate over PCI bus resources. The first part is to go over PCI bus
// resource array, which has at most the %PCI_BRIDGE_RESOURCE_NUM entries.
// After that continue with the separate list of the additional resources,
// if not empty. That's why the Logical OR is being used.
//
// Possible usage:
//
// 	struct pci_bus *bus = ...;
// 	struct resource *res;
// 	unsigned int i;
//
// 	// With optional index
// 	pci_bus_for_each_resource(bus, res, i)
// 		pr_info("PCI bus resource[%u]: %pR\n", i, res);
//
// 	// Without index
// 	pci_bus_for_each_resource(bus, res)
// 		_do_something_(res);
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_bus_find_emul_domain_nr")]
extern "C" fn pci_bus_find_emul_domain_nr(__UNIQUE_ID_x__907: u32, __UNIQUE_ID_y__908: u32, max: u32) -> i32 {
let mut __v0: u32;
loop {
if __types_ok__2!(__UNIQUE_ID_x__907, __UNIQUE_ID_y__908) == 0 {
unsafe { __compiletime_assert_909() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__907 > __UNIQUE_ID_y__908 { __UNIQUE_ID_x__907 } else { __UNIQUE_ID_y__908 };
unsafe { ida_alloc_range(core::ptr::addr_of_mut!(pci_domain_nr_dynamic_ida), __v0, max, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_sysdata {
    domain: i32,
    node: i32,
    companion: *mut acpi_device,
    iommu: *mut (),
    fwnode: *mut (),
}


fn to_pci_sysdata(bus: *const pci_bus) -> *mut pci_sysdata {
(unsafe { (*bus).sysdata }) as *mut pci_sysdata
}

fn pci_domain_nr(bus: *mut pci_bus) -> i32 {
unsafe { (*to_pci_sysdata(bus as *const pci_bus)).domain }
}

fn pci_name(pdev: *const pci_dev) -> *const u8 {
dev_name(unsafe { core::ptr::addr_of!((*pdev).dev) })
}

#[unsafe(no_mangle)]
static mut pci_pci_problems: i32 = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..ro_after_init")]
static mut pci_cache_line_size: u8 = unsafe { core::mem::zeroed() };

fn pci_pcie_cap(dev: *mut pci_dev) -> i32 {
(unsafe { (*dev).pcie_cap }) as i32
}

fn pci_is_pcie(dev: *mut pci_dev) -> bool {
pci_pcie_cap(dev) != 0
}

fn pcie_caps_reg(dev: *const pci_dev) -> u16 {
unsafe { (*dev).pcie_flags_reg }
}

fn pci_pcie_type(dev: *const pci_dev) -> i32 {
((pcie_caps_reg(dev) as i32) & 240) >> 4i32
}

fn pcie_find_root_port(mut dev: *mut pci_dev) -> *mut pci_dev {
while !dev.is_null() {
let __v0: bool = pci_is_pcie(dev) && pci_pcie_type(dev as *const pci_dev) == 4;
if __v0 {
return dev;
}
dev = pci_upstream_bridge(dev);
}
core::ptr::null_mut::<pci_dev>()
}

fn pci_dev_is_disconnected(dev: *const pci_dev) -> bool {
// error_state is set in pci_dev_set_io_state() using xchg/cmpxchg()
// and read w/o common lock. READ_ONCE() ensures compiler cannot cache
// the value (e.g. inside the loop in pci_dev_wait()).
let mut __v0: u32;
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_516() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*dev).error_state) as *const u32) };
__v0 == 3
}

#[unsafe(no_mangle)]
extern "C" fn pci_add_dma_alias(dev: *mut pci_dev, devfn_from: u8, mut nr_devfns: u32) {
let mut devfn_to: i32 = 0;
let mut __v0: u32;
let __UNIQUE_ID_x__900: u32 = nr_devfns;
let __UNIQUE_ID_y__901: u32 = 256u32.wrapping_sub((devfn_from as i32) as u32);
loop {
if __types_ok__2!(__UNIQUE_ID_x__900, __UNIQUE_ID_y__901) == 0 {
unsafe { __compiletime_assert_902() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__900 < __UNIQUE_ID_y__901 { __UNIQUE_ID_x__900 } else { __UNIQUE_ID_y__901 };
nr_devfns = __v0;
devfn_to = ((devfn_from as i32) as u32).wrapping_add(nr_devfns).wrapping_sub(1) as i32;
if unsafe { (*dev).dma_alias_mask }.is_null() {
unsafe {
(*dev).dma_alias_mask = bitmap_zalloc(256u32, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
}
}
if unsafe { (*dev).dma_alias_mask }.is_null() {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Unable to allocate DMA alias mask\n".as_ptr() as *const u8) };
return;
}
bitmap_set(unsafe { (*dev).dma_alias_mask }, devfn_from as u32, nr_devfns);
if nr_devfns == 1 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Enabling fixed DMA alias to %02x.%d\n".as_ptr() as *const u8, PCI_SLOT__3!(devfn_from), PCI_FUNC__3!(devfn_from)) };
} else {
if nr_devfns > 1 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Enabling fixed DMA alias for devfn range from %02x.%d to %02x.%d\n".as_ptr() as *const u8, PCI_SLOT__3!(devfn_from), PCI_FUNC__3!(devfn_from), PCI_SLOT__2!(devfn_to), PCI_FUNC__2!(devfn_to)) };
}
}
}

// Request is asynchronous
// Increment/decrement the
// usage_count
// Use autosuspend_delay
// Succeed if runtime PM is disabled
fn pm_runtime_get_noresume(dev: *mut device) {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*dev).power.usage_count) });
}

fn pm_runtime_suspended(dev: *mut device) -> bool {
((unsafe { (*dev).power.runtime_status }) as i32) == 2 && ((unsafe { (*dev).power.__slate_bits_2.__get_disable_depth() }) as i32) == 0
}

fn pm_runtime_resume(dev: *mut device) -> i32 {
unsafe { __pm_runtime_resume(dev, 0i32) }
}

fn pm_request_resume(dev: *mut device) -> i32 {
unsafe { __pm_runtime_resume(dev, 1i32) }
}

fn pm_runtime_get_sync(dev: *mut device) -> i32 {
unsafe { __pm_runtime_resume(dev, 4i32) }
}

fn pm_runtime_put(dev: *mut device) {
unsafe { __pm_runtime_idle(dev, 5i32) };
}

fn pm_runtime_put_sync(dev: *mut device) -> i32 {
unsafe { __pm_runtime_idle(dev, 4i32) }
}

fn pm_runtime_set_active(dev: *mut device) -> i32 {
unsafe { __pm_runtime_set_status(dev, 0u32) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hotplug_slot_ops {
    enable_slot: Option<unsafe extern "C" fn(*mut hotplug_slot) -> i32>,
    disable_slot: Option<unsafe extern "C" fn(*mut hotplug_slot) -> i32>,
    set_attention_status: Option<unsafe extern "C" fn(*mut hotplug_slot, u8) -> i32>,
    hardware_test: Option<unsafe extern "C" fn(*mut hotplug_slot, u32) -> i32>,
    get_power_status: Option<unsafe extern "C" fn(*mut hotplug_slot, *mut u8) -> i32>,
    get_attention_status: Option<unsafe extern "C" fn(*mut hotplug_slot, *mut u8) -> i32>,
    get_latch_status: Option<unsafe extern "C" fn(*mut hotplug_slot, *mut u8) -> i32>,
    get_adapter_status: Option<unsafe extern "C" fn(*mut hotplug_slot, *mut u8) -> i32>,
    reset_slot: Option<unsafe extern "C" fn(*mut hotplug_slot, bool) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hotplug_slot {
    ops: *const hotplug_slot_ops,
    pci_slot: *mut pci_slot,
    owner: *mut module,
    mod_name: *const u8,
}


// reset direct map and flush TLB on unmap, can't be freed in atomic context
#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
    __slate_anon_0: __SlateRecord_vmalloc_h_53,
    addr: *mut (),
    size: u64,
    flags: u64,
    pages: *mut *mut page,
    page_order: u32,
    nr_pages: u64,
    phys_addr: u64,
    caller: *const (),
    requested_size: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits967U0,
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
    __slate_bits_0: __slate_bits::__SlateBits618U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
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


fn pm_suspend_via_firmware() -> bool {
((unsafe { pm_suspend_global_flags }) as u64) & BIT!(0i32) != 0
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
#[unsafe(no_mangle)]
static mut pci_early_dump: bool = unsafe { core::mem::zeroed() };

// Standard Capability finder
//
// PCI_FIND_NEXT_CAP - Find a PCI standard capability
// @read_cfg: Function pointer for reading PCI config space
// @start: Starting position to begin search
// @cap: Capability ID to find
// @prev_ptr: Pointer to store position of preceding capability (optional)
// @args: Arguments to pass to read_cfg function
//
// Search the capability list in PCI config space to find @cap. If
// found, update *prev_ptr with the position of the preceding capability
// (if prev_ptr != NULL)
// Implements TTL (time-to-live) protection against infinite loops.
//
// Return: Position of the capability if found, 0 otherwise.
// Extended Capability finder
//
// PCI_FIND_NEXT_EXT_CAP - Find a PCI extended capability
// @read_cfg: Function pointer for reading PCI config space
// @start: Starting position to begin search (0 for initial search)
// @cap: Extended capability ID to find
// @prev_ptr: Pointer to store position of preceding capability (optional)
// @args: Arguments to pass to read_cfg function
//
// Search the extended capability list in PCI config space to find @cap.
// If found, update *prev_ptr with the position of the preceding capability
// (if prev_ptr != NULL)
// Implements TTL protection against infinite loops using a calculated
// maximum search count.
//
// Return: Position of the capability if found, 0 otherwise.
#[unsafe(no_mangle)]
extern "C" fn pci_init_reset_methods(dev: *mut pci_dev) {
let mut i: i32 = 0;
let mut rc: i32 = 0;
loop {
if ARRAY_SIZE__9!() != 8 {
unsafe { __compiletime_assert_816() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
might_sleep!();
for m in 1i32..8i32 {
rc = unsafe { unsafe { pci_reset_fn_methods.0[m as usize].reset_fn }.unwrap()(dev, true) };
if rc == 0 {
let __v2: i32 = i;
i = __v2.wrapping_add(1);
unsafe {
*((*dev).reset_methods.as_mut_ptr() as *mut u8).wrapping_offset(__v2 as isize) = m as u8;
}
} else {
if rc != 25i32.wrapping_neg() {
break;
}
}
}
unsafe {
*((*dev).reset_methods.as_mut_ptr() as *mut u8).wrapping_offset(i as isize) = 0;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_cap_saved_data {
    cap_nr: u16,
    cap_extended: bool,
    size: u32,
    data: [u32; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_cap_saved_state {
    next: hlist_node,
    cap: pci_cap_saved_data,
}


// usec; see PCIe r4.0, sec 5.9.1
// msec
// msec
#[unsafe(no_mangle)]
extern "C" fn pci_pm_init(dev: *mut pci_dev) {
let mut pmc: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u16 = core::ptr::addr_of_mut!(pmc) as *mut u16;
let mut pm: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(pm) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
device_enable_async_suspend(core::ptr::addr_of_mut!((*dev).dev));
(*dev).__slate_bits_1.__set_wakeup_prepared(0u32);
(*dev).pm_cap = 0;
(*dev).__slate_bits_1.__set_pme_support(0u32);
// find PCI PM capability in list
*__slate_slot_5 = pci_find_capability(dev, 1i32) as i32;
if *__slate_slot_5 != 0 {
// Check device's ability to generate PME#
pci_read_config_word(dev as *const pci_dev, (*__slate_slot_5).wrapping_add(2), core::ptr::addr_of_mut!(*__slate_slot_4));
if (*__slate_slot_4 as i32) & 7 > 3 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"unsupported PM cap regs version (%u)\n".as_ptr() as *const u8, (*__slate_slot_4 as i32) & 7);
} else {
(*dev).pm_cap = *__slate_slot_5 as u8;
(*dev).d3hot_delay = 10;
(*dev).d3cold_delay = 100;
let __t0 = pci_bridge_d3_possible(dev) as u32;
(*dev).__slate_bits_1.__set_bridge_d3(__t0);
(*dev).__slate_bits_1.__set_d3cold_allowed(1u32);
(*dev).__slate_bits_1.__set_d1_support(0u32);
(*dev).__slate_bits_1.__set_d2_support(0u32);
if pci_no_d1d2(dev) == 0 {
if (*__slate_slot_4 as i32) & 512 != 0 {
(*dev).__slate_bits_1.__set_d1_support(1u32);
}
if (*__slate_slot_4 as i32) & 1024 != 0 {
(*dev).__slate_bits_1.__set_d2_support(1u32);
}
if ((*dev).__slate_bits_1.__get_d1_support() as i32) != 0 || ((*dev).__slate_bits_1.__get_d2_support() as i32) != 0 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"supports%s%s\n".as_ptr() as *const u8, if ((*dev).__slate_bits_1.__get_d1_support() as i32) != 0 { c" D1".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if ((*dev).__slate_bits_1.__get_d2_support() as i32) != 0 { c" D2".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 });
}
}
*__slate_slot_4 = ((*__slate_slot_4 as i32) & 63488) as u16;
if *__slate_slot_4 != 0 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"PME# supported from%s%s%s%s%s\n".as_ptr() as *const u8, if (*__slate_slot_4 as i32) & 2048 != 0 { c" D0".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if (*__slate_slot_4 as i32) & 4096 != 0 { c" D1".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if (*__slate_slot_4 as i32) & 8192 != 0 { c" D2".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if (*__slate_slot_4 as i32) & 16384 != 0 { c" D3hot".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if (*__slate_slot_4 as i32) & 32768 != 0 { c" D3cold".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 });
while false {
}
while false {
}
while false {
}
loop {
if (if true { (!(63488i32 >> (63488u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
__compiletime_assert_763();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 63488u64.wrapping_add(1u64 << (63488u64.trailing_zeros() as i32)) & 63488u64.wrapping_add(1u64 << (63488u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_764();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
(*dev).__slate_bits_1.__set_pme_support((((*__slate_slot_4 as i32) & 63488) >> (63488u64.trailing_zeros() as i32)) as u32);
(*dev).__slate_bits_1.__set_pme_poll(1u32);
// Make device's PM flags reflect the wake-up capability, but
// let the user space enable it to wake up the system as needed.
device_set_wakeup_capable(core::ptr::addr_of_mut!((*dev).dev), true);
// Disable the PME# generation functionality
pci_pme_active(dev, false);
}
}
}
pci_pm_power_up_and_verify_state(dev);
pm_runtime_forbid(core::ptr::addr_of_mut!((*dev).dev));
// Runtime PM will be enabled for the device when it has been fully
// configured, but since its parent and suppliers may suspend in
// the meantime, prevent them from doing so by changing the
// device's runtime PM status to "active".
pm_runtime_set_active(core::ptr::addr_of_mut!((*dev).dev));
}
}

fn pci_bus_rrs_vendor_id(l: u32) -> bool {
l & 65535 == 1
}

fn pci_wakeup_event(dev: *mut pci_dev) {
// Wait 100 ms before the system can be put into a sleep state.
pm_wakeup_event(unsafe { core::ptr::addr_of_mut!((*dev).dev) }, 100u32);
}

fn pci_bar_index_is_valid(bar: i32) -> bool {
if bar >= 0 && bar < 11 {
return true;
}
false
}

fn pci_has_subordinate(pci_dev: *mut pci_dev) -> bool {
!unsafe { (*pci_dev).subordinate }.is_null()
}

fn pci_power_manageable(pci_dev: *mut pci_dev) -> bool {
// Currently we allow normal PCI devices and PCI bridges transition
// into D3 if their bridge_d3 is set.
!pci_has_subordinate(pci_dev) || ((unsafe { (*pci_dev).__slate_bits_1.__get_bridge_d3() }) as i32) != 0
}

fn pcie_downstream_port(dev: *const pci_dev) -> bool {
let r#type: i32 = pci_pcie_type(dev);
r#type == 4 || r#type == 6 || r#type == 8
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_pci_power_names: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_pci_problems: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_ats_disabled: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_bus_max_busnr: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_status_get_and_clear_errors: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_ioremap_bar: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_ioremap_wc_bar: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_find_next_capability: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_find_capability: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_bus_find_capability: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_find_next_ext_capability: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_find_ext_capability: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_get_dsn: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_find_next_ht_capability: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_find_ht_capability: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_find_vsec_capability: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_find_dvsec_capability: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_find_parent_resource: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_find_resource: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_platform_power_transition: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_set_power_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_set_power_state_locked: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_save_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_restore_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_store_saved_state: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_load_saved_state: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_load_and_free_saved_state: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_reenable_device: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_enable_device_mem: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {28} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_enable_device: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {29} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_disable_device: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {30} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_set_pcie_reset_state: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {31} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_pme_capable: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {32} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_pme_active: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {33} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_enable_wake: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {34} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_wake_from_d3: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {35} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_prepare_to_sleep: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {36} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_back_from_sleep: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {37} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_dev_run_wake: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {38} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_choose_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {39} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_suspend_retains_context: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {40} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_d3cold_enable: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {41} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_d3cold_disable: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {42} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_enable_atomic_ops_to_root: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {43} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_release_region: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {44} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_request_region: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {45} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_release_selected_regions: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {46} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_request_selected_regions: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {47} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_request_selected_regions_exclusive: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {48} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_release_regions: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {49} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_request_regions: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {50} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_request_regions_exclusive: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {51} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_pio_to_address: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {52} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_remap_iospace: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {53} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_unmap_iospace: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {54} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_set_master: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {55} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_clear_master: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {56} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_set_cacheline_size: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {57} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_set_mwi: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {58} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_try_set_mwi: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {59} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_clear_mwi: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {60} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_intx: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {61} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_wait_for_pending_transaction: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {62} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_flr: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {63} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_reset_flr: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {64} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_bridge_secondary_bus_reset: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {65} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_dev_lock: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {66} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_dev_trylock: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {67} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_dev_unlock: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {68} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___pci_reset_function_locked: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {69} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_reset_function: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {70} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_reset_function_locked: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {71} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_try_reset_function: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {72} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_probe_reset_slot: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {73} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_bus_error_reset: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {74} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_probe_reset_bus: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {75} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_reset_bus: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {76} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcix_get_max_mmrbc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {77} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcix_get_mmrbc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {78} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcix_set_mmrbc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {79} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_get_readrq: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {80} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_set_readrq: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {81} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_get_mps: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {82} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_set_mps: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {83} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_link_speed_mbps: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {84} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_bandwidth_available: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {85} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_get_speed_cap: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {86} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_get_width_cap: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {87} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pcie_print_link_status: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {88} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_select_bars: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {89} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_pr3_present: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {90} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_device_is_present: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {91} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_ignore_hotplug: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {92} ; .previous\n.section\t\".initcall7.init\", \"a\"\t\t\n__initcall__kmod_pci__905_6686_pci_resource_alignment_sysfs_init7:\t\t\t\n.long\t{93} - .\t\n.previous\t\t\t\t\t\n\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_bus_find_emul_domain_nr: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {94} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pci_bus_release_emul_domain_nr: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {95} ; .previous\n.section\t\".initcall0.init\", \"a\"\t\t\n__initcall__kmod_pci__912_6898_pci_realloc_setup_params0:\t\t\t\n.long\t{96} - .\t\n.previous\t\t\t\t\t\n", sym pci_power_names, sym pci_pci_problems, sym pci_ats_disabled, sym pci_bus_max_busnr, sym pci_status_get_and_clear_errors, sym pci_ioremap_bar, sym pci_ioremap_wc_bar, sym pci_find_next_capability, sym pci_find_capability, sym pci_bus_find_capability, sym pci_find_next_ext_capability, sym pci_find_ext_capability, sym pci_get_dsn, sym pci_find_next_ht_capability, sym pci_find_ht_capability, sym pci_find_vsec_capability, sym pci_find_dvsec_capability, sym pci_find_parent_resource, sym pci_find_resource, sym pci_platform_power_transition, sym pci_set_power_state, sym pci_set_power_state_locked, sym pci_save_state, sym pci_restore_state, sym pci_store_saved_state, sym pci_load_saved_state, sym pci_load_and_free_saved_state, sym pci_reenable_device, sym pci_enable_device_mem, sym pci_enable_device, sym pci_disable_device, sym pci_set_pcie_reset_state, sym pci_pme_capable, sym pci_pme_active, sym pci_enable_wake, sym pci_wake_from_d3, sym pci_prepare_to_sleep, sym pci_back_from_sleep, sym pci_dev_run_wake, sym pci_choose_state, sym pci_suspend_retains_context, sym pci_d3cold_enable, sym pci_d3cold_disable, sym pci_enable_atomic_ops_to_root, sym pci_release_region, sym pci_request_region, sym pci_release_selected_regions, sym pci_request_selected_regions, sym pci_request_selected_regions_exclusive, sym pci_release_regions, sym pci_request_regions, sym pci_request_regions_exclusive, sym pci_pio_to_address, sym pci_remap_iospace, sym pci_unmap_iospace, sym pci_set_master, sym pci_clear_master, sym pci_set_cacheline_size, sym pci_set_mwi, sym pci_try_set_mwi, sym pci_clear_mwi, sym pci_intx, sym pci_wait_for_pending_transaction, sym pcie_flr, sym pcie_reset_flr, sym pci_bridge_secondary_bus_reset, sym pci_dev_lock, sym pci_dev_trylock, sym pci_dev_unlock, sym __pci_reset_function_locked, sym pci_reset_function, sym pci_reset_function_locked, sym pci_try_reset_function, sym pci_probe_reset_slot, sym pci_bus_error_reset, sym pci_probe_reset_bus, sym pci_reset_bus, sym pcix_get_max_mmrbc, sym pcix_get_mmrbc, sym pcix_set_mmrbc, sym pcie_get_readrq, sym pcie_set_readrq, sym pcie_get_mps, sym pcie_set_mps, sym pcie_link_speed_mbps, sym pcie_bandwidth_available, sym pcie_get_speed_cap, sym pcie_get_width_cap, sym pcie_print_link_status, sym pci_select_bars, sym pci_pr3_present, sym pci_device_is_present, sym pci_ignore_hotplug, sym pci_resource_alignment_sysfs_init, sym pci_bus_find_emul_domain_nr, sym pci_bus_release_emul_domain_nr, sym pci_realloc_setup_params, options(att_syntax));

#[unsafe(no_mangle)]
static mut pci_pm_d3hot_delay: u32 = unsafe { core::mem::zeroed() };

fn pci_no_d1d2(dev: *mut pci_dev) -> i32 {
let mut parent_dstates: u32 = 0;
if !unsafe { (*(*dev).bus).self_ }.is_null() {
parent_dstates = ((unsafe { (*(*(*dev).bus).self_).__slate_bits_1.__get_no_d1d2() }) as i32) as u32;
}
(((unsafe { (*dev).__slate_bits_1.__get_no_d1d2() }) as i32) != 0 || parent_dstates != 0) as i32
}

fn pcie_dev_speed_mbps(speed: u32) -> i32 {
match speed as u32 {
20 => {
return 2500;
}
21 => {
return 5000;
}
22 => {
return 8000;
}
23 => {
return 16000;
}
24 => {
return 32000;
}
25 => {
return 64000;
}
_ => {
}
}
22i32.wrapping_neg()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_sriov {
    pos: i32,
    nres: i32,
    cap: u32,
    ctrl: u16,
    total_VFs: u16,
    initial_VFs: u16,
    num_VFs: u16,
    offset: u16,
    stride: u16,
    vf_device: u16,
    pgsz: u32,
    link: u8,
    max_VF_buses: u8,
    driver_max_VFs: u16,
    dev: *mut pci_dev,
    self_: *mut pci_dev,
    class: u32,
    hdr_type: u8,
    subsystem_vendor: u16,
    subsystem_device: u16,
    barsz: [u64; 6],
    vf_rebar_cap: u16,
    drivers_autoprobe: bool,
}


// pci_dev priv_flags
#[repr(C)]
#[derive(Clone, Copy)]
struct rcec_ea {
    nextbusn: u8,
    lastbusn: u8,
    bitmap: u32,
}


fn pci_save_dpc_state(dev: *mut pci_dev) {
}

fn pci_restore_dpc_state(dev: *mut pci_dev) {
}

fn pci_restore_iov_state(dev: *mut pci_dev) {
}

fn pci_restore_tph_state(dev: *mut pci_dev) {
}

fn pci_save_tph_state(dev: *mut pci_dev) {
}

fn pci_no_tph() {
}

fn pci_save_ptm_state(dev: *mut pci_dev) {
}

fn pci_restore_ptm_state(dev: *mut pci_dev) {
}

fn pcie_ecrc_get_policy(str: *mut u8) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_reset_fn_method {
    reset_fn: Option<unsafe extern "C" fn(*mut pci_dev, bool) -> i32>,
    name: *mut u8,
}


#[unsafe(no_mangle)]
static pci_reset_fn_methods: __SlateAlign16<[pci_reset_fn_method; 8]> = __SlateAlign16([{
let mut __t0: pci_reset_fn_method = unsafe { core::mem::zeroed() };
__t0
}, pci_reset_fn_method { reset_fn: unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut pci_dev, bool) -> i32>>(pci_dev_specific_reset as *const ()) }, name: b"device_specific\0".as_ptr() as *mut u8 }, pci_reset_fn_method { reset_fn: unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut pci_dev, bool) -> i32>>(pci_dev_acpi_reset as *const ()) }, name: b"acpi\0".as_ptr() as *mut u8 }, pci_reset_fn_method { reset_fn: Some(pcie_reset_flr), name: b"flr\0".as_ptr() as *mut u8 }, pci_reset_fn_method { reset_fn: Some(pci_af_flr), name: b"af_flr\0".as_ptr() as *mut u8 }, pci_reset_fn_method { reset_fn: Some(pci_pm_reset), name: b"pm\0".as_ptr() as *mut u8 }, pci_reset_fn_method { reset_fn: Some(pci_reset_bus_function), name: b"bus\0".as_ptr() as *mut u8 }, pci_reset_fn_method { reset_fn: Some(cxl_reset_bus_function), name: b"cxl_bus\0".as_ptr() as *mut u8 }]);

fn pci_no_aer() {
}

fn pci_aer_clear_status(dev: *mut pci_dev) -> i32 {
22i32.wrapping_neg()
}

fn pci_save_aer_state(dev: *mut pci_dev) {
}

fn pci_restore_aer_state(dev: *mut pci_dev) {
}

fn pci_use_mid_pm() -> bool {
false
}

fn mid_pci_set_power_state(pdev: *mut pci_dev, state: i32) -> i32 {
19i32.wrapping_neg()
}

fn mid_pci_get_power_state(pdev: *mut pci_dev) -> i32 {
5
}

#[unsafe(no_mangle)]
static mut pci_slot_mutex: mutex = {
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

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_power_names_680: *mut () = (unsafe { core::ptr::addr_of_mut!(pci_power_names.0) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_pci_problems_681: *mut () = (unsafe { core::ptr::addr_of_mut!(pci_pci_problems) }) as *mut ();

static mut pci_pme_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(pci_pme_list) }, prev: unsafe { core::ptr::addr_of_mut!(pci_pme_list) } };

static mut pci_pme_list_mutex: mutex = {
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

static mut pci_pme_work: delayed_work = {
let mut __t0: delayed_work = unsafe { core::mem::zeroed() };
__t0.work = work_struct { data: __SlateRecord_types_h_197 { counter: 4503599625273344i64 }, entry: list_head { next: unsafe { core::ptr::addr_of_mut!(pci_pme_work.work.entry) }, prev: unsafe { core::ptr::addr_of_mut!(pci_pme_work.work.entry) } }, func: Some(pci_pme_list_scan) };
__t0.timer = {
let mut __t1: timer_list = unsafe { core::mem::zeroed() };
__t1.entry = {
let mut __t2: hlist_node = unsafe { core::mem::zeroed() };
__t2.next = ((unsafe { ((768i32 as *mut ()) as *const u8).wrapping_offset(16045481047390945280u64 as isize) }) as *mut ()) as *mut hlist_node;
__t2
};
__t1.function = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) };
__t1.flags = 2097152u32;
__t1
};
__t0
};

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_pme_device {
    list: list_head,
    dev: *mut pci_dev,
}


// How long between PME checks
// Following exit from Conventional Reset, devices must be ready within 1 sec
// (PCIe r6.0 sec 6.6.1).  A D3cold to D0 transition implies a Conventional
// Reset (PCIe r6.0 sec 5.8).
// msec
// Devices may extend the 1 sec period through Request Retry Status
// completions (PCIe r6.0 sec 2.3.1).  The spec does not provide an upper
// limit, but 60 sec ought to be enough for any device to become
// responsive.
// msec
fn pci_dev_d3_sleep(dev: *mut pci_dev) {
let mut delay_ms: u32 = 0;
let mut __v0: u32;
let __UNIQUE_ID_x__682: u32 = unsafe { (*dev).d3hot_delay };
let __UNIQUE_ID_y__683: u32 = unsafe { pci_pm_d3hot_delay };
loop {
if __types_ok__2!(__UNIQUE_ID_x__682, __UNIQUE_ID_y__683) == 0 {
unsafe { __compiletime_assert_684() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__682 > __UNIQUE_ID_y__683 { __UNIQUE_ID_x__682 } else { __UNIQUE_ID_y__683 };
delay_ms = __v0;
let mut upper: u32 = 0;
if delay_ms != 0 {
// Use a 20% upper bound, 1ms minimum
let mut __v2: u32;
let mut __UNIQUE_ID_x__685: u32 = 0;
let mut __v3: u32;
let __x: u32 = delay_ms;
let __d: i32 = 5;
__v3 = if (1i32.wrapping_neg() as u32) > 0 || 1i32.wrapping_neg() > 0 || ((__x > 0) as i32) == ((__d > 0) as i32) { __x.wrapping_add((__d / 2) as u32) / (__d as u32) } else { __x.wrapping_sub((__d / 2) as u32) / (__d as u32) };
__UNIQUE_ID_x__685 = __v3;
let __UNIQUE_ID_y__686: u32 = 1;
loop {
if __types_ok__2!(__UNIQUE_ID_x__685, __UNIQUE_ID_y__686) == 0 {
unsafe { __compiletime_assert_687() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__685 > __UNIQUE_ID_y__686 { __UNIQUE_ID_x__685 } else { __UNIQUE_ID_y__686 };
upper = __v2;
usleep_range((delay_ms as i64).wrapping_mul(1000) as u64, (delay_ms.wrapping_add(upper) as i64).wrapping_mul(1000) as u64);
}
}

#[unsafe(no_mangle)]
extern "C" fn pci_reset_supported(dev: *mut pci_dev) -> bool {
((unsafe { *((*dev).reset_methods.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) != 0
}

#[unsafe(no_mangle)]
static mut pci_domains_supported: i32 = 1;

/// hpiosize=nn can override this
#[unsafe(no_mangle)]
static mut pci_hotplug_io_size: u64 = 256;

/// pci=hpmmiosize=nnM overrides non-prefetchable MMIO size,
/// pci=hpmmioprefsize=nnM overrides prefetchable MMIO size;
/// pci=hpmemsize=nnM overrides both
#[unsafe(no_mangle)]
static mut pci_hotplug_mmio_size: u64 = 2097152;

#[unsafe(no_mangle)]
static mut pci_hotplug_mmio_pref_size: u64 = 2097152;

#[unsafe(no_mangle)]
static mut pci_hotplug_bus_size: u64 = 1;

/// PCIe MPS/MRRS strategy; can be overridden by kernel command-line param
#[unsafe(no_mangle)]
static mut pcie_bus_config: u32 = 1;

/// The default CLS is used if arch didn't set CLS explicitly and not
/// all pci devices agree on the same value.  Arch can override either
/// the dfl or actual value as it sees fit.  Don't forget this is
/// measured in 32-bit words, not bytes.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..ro_after_init")]
static mut pci_dfl_cache_line_size: u8 = 16;

/// If we set up a device for bus mastering, we need to check the latency
/// timer as certain BIOSes forget to set it properly.
#[unsafe(no_mangle)]
static mut pcibios_max_latency: u32 = 255;

/// If set, the PCIe ARI capability will not be used.
static mut pcie_ari_disabled: bool = unsafe { core::mem::zeroed() };

/// If set, the PCIe ATS capability will not be used.
static mut pcie_ats_disabled: bool = unsafe { core::mem::zeroed() };

/// If set, the PCI config space of each device is printed during boot.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_ats_disabled")]
extern "C" fn pci_ats_disabled() -> bool {
unsafe { pcie_ats_disabled }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_ats_disabled_688: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> bool>, *mut ()>(Some(pci_ats_disabled)) };

/// Disable bridge_d3 for all PCIe ports
static mut pci_bridge_d3_disable: bool = unsafe { core::mem::zeroed() };

/// Force bridge_d3 for all PCIe ports
static mut pci_bridge_d3_force: bool = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.text")]
extern "C" fn pcie_port_pm_setup(str: *mut u8) -> i32 {
if (unsafe { strcmp(str as *const u8, c"off".as_ptr() as *const u8) }) == 0 {
unsafe {
pci_bridge_d3_disable = true;
}
} else {
if (unsafe { strcmp(str as *const u8, c"force".as_ptr() as *const u8) }) == 0 {
unsafe {
pci_bridge_d3_force = true;
}
}
}
1
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_pcie_port_pm_setup: [u8; 14] = [112, 99, 105, 101, 95, 112, 111, 114, 116, 95, 112, 109, 61, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_pcie_port_pm_setup: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_pcie_port_pm_setup) as *const u8 };
__t0.setup_func = Some(pcie_port_pm_setup);
__t0.early = 0i32;
__t0
};

/// pci_bus_max_busnr - returns maximum PCI bus number of given bus' children
/// @bus: pointer to PCI bus structure to search
///
/// Given a PCI bus, returns the highest PCI bus number present in the set
/// including the given PCI bus and its list of child PCI buses.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_bus_max_busnr")]
extern "C" fn pci_bus_max_busnr(bus: *mut pci_bus) -> u8 {
let mut tmp: *mut pci_bus = unsafe { core::mem::zeroed() };
let mut max: u8 = (unsafe { (*bus).busn_res.end }) as u8;
let mut n: u8 = 0;
tmp = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*bus).children) });
while list_entry_is_head!(tmp, unsafe { core::ptr::addr_of_mut!((*bus).children) }, node) == 0 {
n = pci_bus_max_busnr(tmp);
if (n as i32) > (max as i32) {
max = n;
}
tmp = list_next_entry__5!(tmp, node);
}
max
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_bus_max_busnr_689: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_bus) -> u8>, *mut ()>(Some(pci_bus_max_busnr)) };

/// pci_status_get_and_clear_errors - return and clear error bits in PCI_STATUS
/// @pdev: the PCI device
///
/// Returns error bits set in PCI_STATUS and clears them.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_status_get_and_clear_errors")]
extern "C" fn pci_status_get_and_clear_errors(pdev: *mut pci_dev) -> i32 {
let mut status: u16 = 0;
let ret: i32 = unsafe { pci_read_config_word(pdev as *const pci_dev, 6i32, core::ptr::addr_of_mut!(status)) };
if ret != 0 {
return 5i32.wrapping_neg();
}
status = ((status as i32) & (32768 | 16384 | 8192 | 4096 | 2048 | 256)) as u16;
if status != 0 {
unsafe { pci_write_config_word(pdev as *const pci_dev, 6i32, status) };
}
status as i32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_status_get_and_clear_errors_690: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_status_get_and_clear_errors)) };

fn __pci_ioremap_resource(pdev: *mut pci_dev, bar: i32, write_combine: bool) -> *mut () {
let res: *mut resource = unsafe { (*pdev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(bar as isize);
let start: u64 = unsafe { (*res).start };
let size: u64 = resource_size(res as *const resource);
// Make sure the BAR is actually a memory resource, not an IO resource
if (unsafe { (*res).flags }) & 536870912 != 0 || (unsafe { (*res).flags }) & 512 == 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"can't ioremap BAR %d: %pR\n".as_ptr() as *const u8, bar, res) };
return core::ptr::null_mut::<()>();
}
if write_combine {
return unsafe { ioremap_wc(start, size) };
}
unsafe { ioremap(start, size) }
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_ioremap_bar")]
extern "C" fn pci_ioremap_bar(pdev: *mut pci_dev, bar: i32) -> *mut () {
__pci_ioremap_resource(pdev, bar, false)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_ioremap_bar_691: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32) -> *mut ()>, *mut ()>(Some(pci_ioremap_bar)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_ioremap_wc_bar")]
extern "C" fn pci_ioremap_wc_bar(pdev: *mut pci_dev, bar: i32) -> *mut () {
__pci_ioremap_resource(pdev, bar, true)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_ioremap_wc_bar_692: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32) -> *mut ()>, *mut ()>(Some(pci_ioremap_wc_bar)) };

/// pci_dev_str_match_path - test if a path string matches a device
/// @dev: the PCI device to test
/// @path: string to match the device against
/// @endptr: pointer to the string after the match
///
/// Test if a string (typically from a kernel parameter) formatted as a
/// path of device/function addresses matches a PCI device. The string must
/// be of the form:
///
///   [<domain>:]<bus>:<device>.<func>[/<device>.<func>]*
///
/// A path for a device can be obtained using 'lspci -t'.  Using a path
/// is more robust against bus renumbering than using only a single bus,
/// device and function address.
///
/// Returns 1 if the string matches the device, 0 if it does not and
/// a negative error code if it fails to parse the string.
fn pci_dev_str_match_path(mut dev: *mut pci_dev, path: *const u8, endptr: *mut *const u8) -> i32 {
let mut end: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u8 = core::ptr::addr_of_mut!(end) as *mut u8;
let mut p: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut u8 = core::ptr::addr_of_mut!(p) as *mut *mut u8;
let mut wpath: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut u8 = core::ptr::addr_of_mut!(wpath) as *mut *mut u8;
let mut func: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u32 = core::ptr::addr_of_mut!(func) as *mut u32;
let mut slot: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(slot) as *mut u32;
let mut bus: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u32 = core::ptr::addr_of_mut!(bus) as *mut u32;
let mut seg: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(seg) as *mut u32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*endptr = strchrnul(path, 59i32) as *const u8;
*__slate_slot_2 = kmemdup_nul(path, (unsafe { *endptr }.offset_from(path as *const u8) as i64) as u64, (BIT!(5i32) as u32) | (BIT!(11i32) as u32));
if (*__slate_slot_2).is_null() {
return 12i32.wrapping_neg();
} else {
'__join_0: {
'__join_4: {
'__join_11: {
'__join_9: {
loop {
if true {
*__slate_slot_1 = strrchr(*__slate_slot_2 as *const u8, 47i32);
if (*__slate_slot_1).is_null() {
break '__join_4;
} else {
*__slate_slot_7 = sscanf(*__slate_slot_1 as *const u8, c"/%x.%x%c".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_4), core::ptr::addr_of_mut!(*__slate_slot_3), core::ptr::addr_of_mut!(*__slate_slot_0));
if *__slate_slot_7 != 2 {
break '__join_11;
} else {
if (*dev).devfn != PCI_DEVFN__3!(*__slate_slot_4, *__slate_slot_3) {
break '__join_9;
} else {
// Note: we don't need to get a reference to the upstream
// bridge because we hold a reference to the top level
// device which should hold a reference to the bridge,
// and so on.
dev = pci_upstream_bridge(dev);
if dev.is_null() {
break;
} else {
*(*__slate_slot_1) = 0;
}
}
}
}
} else {
break '__join_4;
}
}
*__slate_slot_7 = 0;
break '__join_0;
}
*__slate_slot_7 = 0;
break '__join_0;
}
*__slate_slot_7 = 22i32.wrapping_neg();
break '__join_0;
}
*__slate_slot_7 = sscanf(*__slate_slot_2 as *const u8, c"%x:%x:%x.%x%c".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_6), core::ptr::addr_of_mut!(*__slate_slot_5), core::ptr::addr_of_mut!(*__slate_slot_4), core::ptr::addr_of_mut!(*__slate_slot_3), core::ptr::addr_of_mut!(*__slate_slot_0));
if *__slate_slot_7 != 4 {
*__slate_slot_6 = 0;
*__slate_slot_7 = sscanf(*__slate_slot_2 as *const u8, c"%x:%x.%x%c".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_5), core::ptr::addr_of_mut!(*__slate_slot_4), core::ptr::addr_of_mut!(*__slate_slot_3), core::ptr::addr_of_mut!(*__slate_slot_0));
if *__slate_slot_7 != 3 {
*__slate_slot_7 = 22i32.wrapping_neg();
break '__join_0;
}
}
*__slate_slot_7 = (*__slate_slot_6 == (pci_domain_nr((*dev).bus) as u32) && *__slate_slot_5 == (((*(*dev).bus).number as i32) as u32) && (*dev).devfn == PCI_DEVFN__3!(*__slate_slot_4, *__slate_slot_3)) as i32;
}
kfree(*__slate_slot_2 as *const ());
return *__slate_slot_7;
}
}
}

/// pci_dev_str_match - test if a string matches a device
/// @dev: the PCI device to test
/// @p: string to match the device against
/// @endptr: pointer to the string after the match
///
/// Test if a string (typically from a kernel parameter) matches a specified
/// PCI device. The string may be of one of the following formats:
///
///   [<domain>:]<bus>:<device>.<func>[/<device>.<func>]*
///   pci:<vendor>:<device>[:<subvendor>:<subdevice>]
///
/// The first format specifies a PCI bus/device/function address which
/// may change if new hardware is inserted, if motherboard firmware changes,
/// or due to changes caused in kernel parameters. If the domain is
/// left unspecified, it is taken to be 0.  In order to be robust against
/// bus renumbering issues, a path of PCI device/function numbers may be used
/// to address the specific device.  The path for a device can be determined
/// through the use of 'lspci -t'.
///
/// The second format matches devices using IDs in the configuration
/// space which may match multiple devices in the system. A value of 0
/// for any field will match all devices. (Note: this differs from
/// in-kernel code that uses PCI_ANY_ID which is ~0; this is for
/// legacy reasons and convenience so users don't have to specify
/// FFFFFFFFs on the command line.)
///
/// Returns 1 if the string matches the device, 0 if it does not and
/// a negative error code if the string cannot be parsed.
fn pci_dev_str_match(dev: *mut pci_dev, mut p: *const u8, endptr: *mut *const u8) -> i32 {
let mut subsystem_device: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u16 = core::ptr::addr_of_mut!(subsystem_device) as *mut u16;
let mut subsystem_vendor: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u16 = core::ptr::addr_of_mut!(subsystem_vendor) as *mut u16;
let mut device: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u16 = core::ptr::addr_of_mut!(device) as *mut u16;
let mut vendor: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u16 = core::ptr::addr_of_mut!(vendor) as *mut u16;
let mut count: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(count) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
'__join_0: {
'__join_1: {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_3 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_4 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 2usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
if strncmp(p, c"pci:".as_ptr() as *const u8, 4u64) == 0 {
// PCI vendor/device (subvendor/subdevice) IDs are specified
let __v3: *const u8 = p;
p = __v3.wrapping_offset(4 as isize);
*__slate_slot_9 = sscanf(p, c"%hx:%hx:%hx:%hx%n".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_7), core::ptr::addr_of_mut!(*__slate_slot_6), core::ptr::addr_of_mut!(*__slate_slot_5), core::ptr::addr_of_mut!(*__slate_slot_4), core::ptr::addr_of_mut!(*__slate_slot_8));
if *__slate_slot_9 != 4 {
*__slate_slot_9 = sscanf(p, c"%hx:%hx%n".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_7), core::ptr::addr_of_mut!(*__slate_slot_6), core::ptr::addr_of_mut!(*__slate_slot_8));
if *__slate_slot_9 != 2 {
return 22i32.wrapping_neg();
} else {
*__slate_slot_5 = 0;
*__slate_slot_4 = 0;
}
}
let __v1: *const u8 = p;
p = __v1.wrapping_offset(*__slate_slot_8 as isize);
if (*__slate_slot_7 == 0 || (*__slate_slot_7 as i32) == ((*dev).vendor as i32)) && (*__slate_slot_6 == 0 || (*__slate_slot_6 as i32) == ((*dev).device as i32)) && (*__slate_slot_5 == 0 || (*__slate_slot_5 as i32) == ((*dev).subsystem_vendor as i32)) && (*__slate_slot_4 == 0 || (*__slate_slot_4 as i32) == ((*dev).subsystem_device as i32)) {
break '__join_0;
}
} else {
// PCI Bus, Device, Function IDs are specified
// (optionally, may include a path of devfns following it)
*__slate_slot_9 = pci_dev_str_match_path(dev, p, core::ptr::addr_of_mut!(p));
if *__slate_slot_9 < 0 {
return *__slate_slot_9;
} else {
if *__slate_slot_9 != 0 {
break '__join_0;
} else {
break '__join_1;
}
}
}
}
*endptr = p;
return 0;
}
*endptr = p;
return 1;
}
}

fn __pci_find_next_cap(bus: *mut pci_bus, devfn: u32, pos: u8, cap: i32) -> u8 {
let mut __v0: u8;
let mut __ttl: i32 = 48;
let mut __id: u8 = 0;
let mut __found_pos: u8 = 0;
let mut __prev_pos: u8 = pos;
let mut __pos: u8 = pos;
let mut __ent: u16 = 0;
unsafe { pci_bus_read_config_byte(bus, devfn, __pos as i32, core::ptr::addr_of_mut!(__pos)) };
loop {
let __v16: i32 = __ttl;
__ttl = __v16.wrapping_sub(1);
if __v16 == 0 {
break;
}
if (__pos as i32) < 64 {
break;
}
__pos = ALIGN_DOWN!(__pos, 4i32) as u8;
unsafe { pci_bus_read_config_word(bus, devfn, __pos as i32, core::ptr::addr_of_mut!(__ent)) };
let mut __v2: i32;
loop {
if __bf_cast_unsigned!(255i32) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u32) {
unsafe { __compiletime_assert_693() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v4: i32;
loop {
if false {
unsafe { __compiletime_assert_694() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_695() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(255i32 >> (255u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_696() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 255u64.wrapping_add(1u64 << (255u64.trailing_zeros() as i32)) & 255u64.wrapping_add(1u64 << (255u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_697() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = ((__ent as i32) & 255) >> (255u64.trailing_zeros() as i32);
__v2 = __v4;
__id = __v2 as u8;
if (__id as i32) == 255 {
break;
}
if (__id as i32) == cap {
__found_pos = __pos;
if core::ptr::null_mut::<()>() != core::ptr::null_mut::<()>() {
unsafe {
*core::ptr::null_mut::<u8>() = __prev_pos;
}
}
break;
}
__prev_pos = __pos;
let mut __v9: i32;
loop {
if __bf_cast_unsigned!(65280i32) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u32) {
unsafe { __compiletime_assert_698() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v11: i32;
loop {
if false {
unsafe { __compiletime_assert_699() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_700() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(65280i32 >> (65280u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_701() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)) & 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_702() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v11 = ((__ent as i32) & 65280) >> (65280u64.trailing_zeros() as i32);
__v9 = __v11;
__pos = __v9 as u8;
}
__v0 = __found_pos;
__v0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_find_next_capability")]
extern "C" fn pci_find_next_capability(dev: *mut pci_dev, pos: u8, cap: i32) -> u8 {
__pci_find_next_cap(unsafe { (*dev).bus }, unsafe { (*dev).devfn }, (pos as i32).wrapping_add(1) as u8, cap)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_find_next_capability_703: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, u8, i32) -> u8>, *mut ()>(Some(pci_find_next_capability)) };

fn __pci_bus_find_cap_start(bus: *mut pci_bus, devfn: u32, hdr_type: u8) -> u8 {
let mut status: u16 = 0;
unsafe { pci_bus_read_config_word(bus, devfn, 6i32, core::ptr::addr_of_mut!(status)) };
if (status as i32) & 16 == 0 {
return 0;
}
match hdr_type as i32 {
0 | 1 => {
return 52;
}
2 => {
return 20;
}
_ => {
}
}
0
}

/// pci_find_capability - query for devices' capabilities
/// @dev: PCI device to query
/// @cap: capability code
///
/// Tell if a device supports a given PCI capability.
/// Returns the address of the requested capability structure within the
/// device's PCI configuration space or 0 in case the device does not
/// support it.  Possible values for @cap include:
///
///  %PCI_CAP_ID_PM           Power Management
///  %PCI_CAP_ID_AGP          Accelerated Graphics Port
///  %PCI_CAP_ID_VPD          Vital Product Data
///  %PCI_CAP_ID_SLOTID       Slot Identification
///  %PCI_CAP_ID_MSI          Message Signalled Interrupts
///  %PCI_CAP_ID_CHSWP        CompactPCI HotSwap
///  %PCI_CAP_ID_PCIX         PCI-X
///  %PCI_CAP_ID_EXP          PCI Express
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_find_capability")]
extern "C" fn pci_find_capability(dev: *mut pci_dev, cap: i32) -> u8 {
let mut pos: u8 = __pci_bus_find_cap_start(unsafe { (*dev).bus }, unsafe { (*dev).devfn }, unsafe { (*dev).hdr_type });
if pos != 0 {
pos = __pci_find_next_cap(unsafe { (*dev).bus }, unsafe { (*dev).devfn }, pos, cap);
}
pos
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_find_capability_704: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32) -> u8>, *mut ()>(Some(pci_find_capability)) };

/// pci_bus_find_capability - query for devices' capabilities
/// @bus: the PCI bus to query
/// @devfn: PCI device to query
/// @cap: capability code
///
/// Like pci_find_capability() but works for PCI devices that do not have a
/// pci_dev structure set up yet.
///
/// Returns the address of the requested capability structure within the
/// device's PCI configuration space or 0 in case the device does not
/// support it.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_bus_find_capability")]
extern "C" fn pci_bus_find_capability(bus: *mut pci_bus, devfn: u32, cap: i32) -> u8 {
let mut hdr_type: u8 = 0;
let mut pos: u8 = 0;
unsafe { pci_bus_read_config_byte(bus, devfn, 14i32, core::ptr::addr_of_mut!(hdr_type)) };
pos = __pci_bus_find_cap_start(bus, devfn, ((hdr_type as i32) & 127) as u8);
if pos != 0 {
pos = __pci_find_next_cap(bus, devfn, pos, cap);
}
pos
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_bus_find_capability_705: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_bus, u32, i32) -> u8>, *mut ()>(Some(pci_bus_find_capability)) };

/// pci_find_next_ext_capability - Find an extended capability
/// @dev: PCI device to query
/// @start: address at which to start looking (0 to start at beginning of list)
/// @cap: capability code
///
/// Returns the address of the next matching extended capability structure
/// within the device's PCI configuration space or 0 if the device does
/// not support it.  Some capabilities can occur several times, e.g., the
/// vendor-specific capability, and this provides a way to find them all.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_find_next_ext_capability")]
extern "C" fn pci_find_next_ext_capability(dev: *mut pci_dev, start: u16, cap: i32) -> u16 {
if (unsafe { (*dev).cfg_size }) <= 256 {
return 0;
}
let mut __v0: u16;
let mut __pos: u16 = 0;
let __v1: u16 = start;
__pos = (if __v1 != 0 { __v1 as i32 } else { 256i32 }) as u16;
let mut __found_pos: u16 = 0;
let mut __prev_pos: u16 = __pos;
let mut __ttl: i32 = 480i32;
let mut __ret: i32 = 0;
let mut __header: u32 = 0;
loop {
let __v3: i32 = __ttl;
__ttl = __v3.wrapping_sub(1);
if !(__v3 > 0 && (__pos as i32) >= 256) {
break;
}
__ret = unsafe { pci_bus_read_config_dword((*dev).bus, (*dev).devfn, __pos as i32, core::ptr::addr_of_mut!(__header)) };
if __ret != 0 {
break;
}
if __header == 0 {
break;
}
if PCI_EXT_CAP_ID!(__header) == (cap as u32) && (__pos as i32) != (start as i32) {
__found_pos = __pos;
if core::ptr::null_mut::<()>() != core::ptr::null_mut::<()>() {
unsafe {
*core::ptr::null_mut::<u16>() = __prev_pos;
}
}
break;
}
__prev_pos = __pos;
__pos = PCI_EXT_CAP_NEXT!(__header) as u16;
}
__v0 = __found_pos;
__v0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_find_next_ext_capability_706: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, u16, i32) -> u16>, *mut ()>(Some(pci_find_next_ext_capability)) };

/// pci_find_ext_capability - Find an extended capability
/// @dev: PCI device to query
/// @cap: capability code
///
/// Returns the address of the requested extended capability structure
/// within the device's PCI configuration space or 0 if the device does
/// not support it.  Possible values for @cap include:
///
///  %PCI_EXT_CAP_ID_ERR		Advanced Error Reporting
///  %PCI_EXT_CAP_ID_VC		Virtual Channel
///  %PCI_EXT_CAP_ID_DSN		Device Serial Number
///  %PCI_EXT_CAP_ID_PWR		Power Budgeting
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_find_ext_capability")]
extern "C" fn pci_find_ext_capability(dev: *mut pci_dev, cap: i32) -> u16 {
pci_find_next_ext_capability(dev, 0u16, cap)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_find_ext_capability_707: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32) -> u16>, *mut ()>(Some(pci_find_ext_capability)) };

/// pci_get_dsn - Read and return the 8-byte Device Serial Number
/// @dev: PCI device to query
///
/// Looks up the PCI_EXT_CAP_ID_DSN and reads the 8 bytes of the Device Serial
/// Number.
///
/// Returns the DSN, or zero if the capability does not exist.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_get_dsn")]
extern "C" fn pci_get_dsn(dev: *mut pci_dev) -> u64 {
let mut dword: u32 = 0;
let mut dsn: u64 = 0;
let mut pos: i32 = pci_find_ext_capability(dev, 3i32) as i32;
if pos == 0 {
return 0;
}
// The Device Serial Number is two dwords offset 4 bytes from the
// capability position. The specification says that the first dword is
// the lower half, and the second dword is the upper half.
pos = pos.wrapping_add(4);
unsafe { pci_read_config_dword(dev as *const pci_dev, pos, core::ptr::addr_of_mut!(dword)) };
dsn = dword as u64;
unsafe { pci_read_config_dword(dev as *const pci_dev, pos.wrapping_add(4), core::ptr::addr_of_mut!(dword)) };
dsn |= (dword as u64) << 32i32;
dsn
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_get_dsn_708: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> u64>, *mut ()>(Some(pci_get_dsn)) };

fn __pci_find_next_ht_cap(dev: *mut pci_dev, mut pos: u8, ht_cap: i32) -> u8 {
let mut rc: i32 = 0;
let mut cap: u8 = 0;
let mask: u8 = if ht_cap == 0 || ht_cap == 32 {
224
} else {
248
};
let mut __v0: u8;
{
let mut __ttl: i32 = 48;
let mut __id: u8 = 0;
let mut __found_pos: u8 = 0;
let mut __prev_pos: u8 = pos;
let mut __pos: u8 = pos;
let mut __ent: u16 = 0;
unsafe { pci_bus_read_config_byte((*dev).bus, (*dev).devfn, __pos as i32, core::ptr::addr_of_mut!(__pos)) };
loop {
let __v16: i32 = __ttl;
__ttl = __v16.wrapping_sub(1);
if __v16 == 0 {
break;
}
if (__pos as i32) < 64 {
break;
}
__pos = ALIGN_DOWN!(__pos, 4i32) as u8;
unsafe { pci_bus_read_config_word((*dev).bus, (*dev).devfn, __pos as i32, core::ptr::addr_of_mut!(__ent)) };
let mut __v2: i32;
loop {
if __bf_cast_unsigned!(255i32) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u32) {
unsafe { __compiletime_assert_709() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v4: i32;
loop {
if false {
unsafe { __compiletime_assert_710() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_711() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(255i32 >> (255u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_712() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 255u64.wrapping_add(1u64 << (255u64.trailing_zeros() as i32)) & 255u64.wrapping_add(1u64 << (255u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_713() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = ((__ent as i32) & 255) >> (255u64.trailing_zeros() as i32);
__v2 = __v4;
__id = __v2 as u8;
if (__id as i32) == 255 {
break;
}
if (__id as i32) == 8 {
__found_pos = __pos;
if core::ptr::null_mut::<()>() != core::ptr::null_mut::<()>() {
unsafe {
*core::ptr::null_mut::<u8>() = __prev_pos;
}
}
break;
}
__prev_pos = __pos;
let mut __v9: i32;
loop {
if __bf_cast_unsigned!(65280i32) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u32) {
unsafe { __compiletime_assert_714() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v11: i32;
loop {
if false {
unsafe { __compiletime_assert_715() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_716() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(65280i32 >> (65280u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_717() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)) & 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_718() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v11 = ((__ent as i32) & 65280) >> (65280u64.trailing_zeros() as i32);
__v9 = __v11;
__pos = __v9 as u8;
}
__v0 = __found_pos;
}
pos = __v0;
while pos != 0 {
rc = unsafe { pci_read_config_byte(dev as *const pci_dev, (pos as i32).wrapping_add(3), core::ptr::addr_of_mut!(cap)) };
if rc != 0 {
return 0;
}
if (cap as i32) & (mask as i32) == ht_cap {
return pos;
}
let mut __v18: u8;
let mut __ttl: i32 = 48;
let mut __id: u8 = 0;
let mut __found_pos: u8 = 0;
let mut __prev_pos: u8 = (pos as i32).wrapping_add(1) as u8;
let mut __pos: u8 = (pos as i32).wrapping_add(1) as u8;
let mut __ent: u16 = 0;
unsafe { pci_bus_read_config_byte((*dev).bus, (*dev).devfn, __pos as i32, core::ptr::addr_of_mut!(__pos)) };
loop {
let __v34: i32 = __ttl;
__ttl = __v34.wrapping_sub(1);
if __v34 == 0 {
break;
}
if (__pos as i32) < 64 {
break;
}
__pos = ALIGN_DOWN!(__pos, 4i32) as u8;
unsafe { pci_bus_read_config_word((*dev).bus, (*dev).devfn, __pos as i32, core::ptr::addr_of_mut!(__ent)) };
let mut __v20: i32;
loop {
if __bf_cast_unsigned!(255i32) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u32) {
unsafe { __compiletime_assert_719() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v22: i32;
loop {
if false {
unsafe { __compiletime_assert_720() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_721() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(255i32 >> (255u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_722() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 255u64.wrapping_add(1u64 << (255u64.trailing_zeros() as i32)) & 255u64.wrapping_add(1u64 << (255u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_723() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v22 = ((__ent as i32) & 255) >> (255u64.trailing_zeros() as i32);
__v20 = __v22;
__id = __v20 as u8;
if (__id as i32) == 255 {
break;
}
if (__id as i32) == 8 {
__found_pos = __pos;
if core::ptr::null_mut::<()>() != core::ptr::null_mut::<()>() {
unsafe {
*core::ptr::null_mut::<u8>() = __prev_pos;
}
}
break;
}
__prev_pos = __pos;
let mut __v27: i32;
loop {
if __bf_cast_unsigned!(65280i32) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u32) {
unsafe { __compiletime_assert_724() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v29: i32;
loop {
if false {
unsafe { __compiletime_assert_725() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_726() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(65280i32 >> (65280u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_727() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)) & 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_728() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v29 = ((__ent as i32) & 65280) >> (65280u64.trailing_zeros() as i32);
__v27 = __v29;
__pos = __v27 as u8;
}
__v18 = __found_pos;
pos = __v18;
}
0
}

/// pci_find_next_ht_capability - query a device's HyperTransport capabilities
/// @dev: PCI device to query
/// @pos: Position from which to continue searching
/// @ht_cap: HyperTransport capability code
///
/// To be used in conjunction with pci_find_ht_capability() to search for
/// all capabilities matching @ht_cap. @pos should always be a value returned
/// from pci_find_ht_capability().
///
/// NB. To be 100% safe against broken PCI devices, the caller should take
/// steps to avoid an infinite loop.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_find_next_ht_capability")]
extern "C" fn pci_find_next_ht_capability(dev: *mut pci_dev, pos: u8, ht_cap: i32) -> u8 {
__pci_find_next_ht_cap(dev, (pos as i32).wrapping_add(1) as u8, ht_cap)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_find_next_ht_capability_729: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, u8, i32) -> u8>, *mut ()>(Some(pci_find_next_ht_capability)) };

/// pci_find_ht_capability - query a device's HyperTransport capabilities
/// @dev: PCI device to query
/// @ht_cap: HyperTransport capability code
///
/// Tell if a device supports a given HyperTransport capability.
/// Returns an address within the device's PCI configuration space
/// or 0 in case the device does not support the request capability.
/// The address points to the PCI capability, of type PCI_CAP_ID_HT,
/// which has a HyperTransport capability matching @ht_cap.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_find_ht_capability")]
extern "C" fn pci_find_ht_capability(dev: *mut pci_dev, ht_cap: i32) -> u8 {
let mut pos: u8 = __pci_bus_find_cap_start(unsafe { (*dev).bus }, unsafe { (*dev).devfn }, unsafe { (*dev).hdr_type });
if pos != 0 {
pos = __pci_find_next_ht_cap(dev, pos, ht_cap);
}
pos
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_find_ht_capability_730: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32) -> u8>, *mut ()>(Some(pci_find_ht_capability)) };

/// pci_find_vsec_capability - Find a vendor-specific extended capability
/// @dev: PCI device to query
/// @vendor: Vendor ID for which capability is defined
/// @cap: Vendor-specific capability ID
///
/// If @dev has Vendor ID @vendor, search for a VSEC capability with
/// VSEC ID @cap. If found, return the capability offset in
/// config space; otherwise return 0.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_find_vsec_capability")]
extern "C" fn pci_find_vsec_capability(dev: *mut pci_dev, vendor: u16, cap: i32) -> u16 {
let mut vsec: u16 = 0;
let mut header: u32 = 0;
let mut ret: i32 = 0;
if (vendor as i32) != ((unsafe { (*dev).vendor }) as i32) {
return 0;
}
loop {
let __v1: u16 = pci_find_next_ext_capability(dev, vsec, 11i32);
vsec = __v1;
if __v1 == 0 {
break;
}
ret = unsafe { pci_read_config_dword(dev as *const pci_dev, (vsec as i32).wrapping_add(4), core::ptr::addr_of_mut!(header)) };
if ret == 0 {
if PCI_VNDR_HEADER_ID!(header) == (cap as u32) {
return vsec;
}
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_find_vsec_capability_731: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, u16, i32) -> u16>, *mut ()>(Some(pci_find_vsec_capability)) };

/// pci_find_dvsec_capability - Find DVSEC for vendor
/// @dev: PCI device to query
/// @vendor: Vendor ID to match for the DVSEC
/// @dvsec: Designated Vendor-specific capability ID
///
/// If DVSEC has Vendor ID @vendor and DVSEC ID @dvsec return the capability
/// offset in config space; otherwise return 0.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_find_dvsec_capability")]
extern "C" fn pci_find_dvsec_capability(dev: *mut pci_dev, vendor: u16, dvsec: u16) -> u16 {
let mut pos: i32 = pci_find_ext_capability(dev, 35i32) as i32;
if pos == 0 {
return 0;
}
while pos != 0 {
let mut v: u16 = 0;
let mut id: u16 = 0;
unsafe { pci_read_config_word(dev as *const pci_dev, pos.wrapping_add(4), core::ptr::addr_of_mut!(v)) };
unsafe { pci_read_config_word(dev as *const pci_dev, pos.wrapping_add(8), core::ptr::addr_of_mut!(id)) };
if (vendor as i32) == (v as i32) && (dvsec as i32) == (id as i32) {
return pos as u16;
}
pos = pci_find_next_ext_capability(dev, pos as u16, 35i32) as i32;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_find_dvsec_capability_732: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, u16, u16) -> u16>, *mut ()>(Some(pci_find_dvsec_capability)) };

fn pci_dev_config_accessible(dev: *mut pci_dev, msg: *mut u8) -> bool {
let mut val: u32 = 0;
// If a device's config space is inaccessible, reads typically
// return ~0.  Since Device and Vendor ID are always ~0 for VFs,
// check the Command and Status registers instead.
//
// N.B. This is racy because the device may become inaccessible
// before the next access.
unsafe { pci_read_config_dword(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(val)) };
if PCI_POSSIBLE_ERROR!(val) {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Device config space inaccessible; unable to %s\n".as_ptr() as *const u8, msg) };
return false;
}
true
}

/// pci_find_parent_resource - return resource region of parent bus of given
/// 			      region
/// @dev: PCI device structure contains resources to be searched
/// @res: child resource record for which parent is sought
///
/// For given resource region of given device, return the resource region of
/// parent bus the given region is contained in.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_find_parent_resource")]
extern "C" fn pci_find_parent_resource(dev: *const pci_dev, res: *mut resource) -> *mut resource {
let bus: *const pci_bus = (unsafe { (*dev).bus }) as *const pci_bus;
let mut __b: u32 = 0;
loop {
let __v1: *mut resource = unsafe { pci_bus_resource_n(bus, __b as i32) };
let r: *mut resource = __v1;
if !(!__v1.is_null() || __b < 4) {
break;
}
if !r.is_null() {
if resource_contains(r as *const resource, res as *const resource) {
// If the window is prefetchable but the BAR is
// not, the allocator made a mistake.
if (unsafe { (*r).flags }) & 8192 != 0 && (unsafe { (*res).flags }) & 8192 == 0 {
return core::ptr::null_mut::<resource>();
}
// If we're below a transparent bridge, there may
// be both a positively-decoded aperture and a
// subtractively-decoded region that contain the BAR.
// We want the positively-decoded one, so this depends
// on pci_bus_for_each_resource() giving us those
// first.
return r;
}
}
__b = __b.wrapping_add(1);
}
core::ptr::null_mut::<resource>()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_find_parent_resource_733: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const pci_dev, *mut resource) -> *mut resource>, *mut ()>(Some(pci_find_parent_resource)) };

/// pci_find_resource - Return matching PCI device resource
/// @dev: PCI device to query
/// @res: Resource to look for
///
/// Goes over standard PCI resources (BARs) and checks if the given resource
/// is partially or fully contained in any of them. In that case the
/// matching resource is returned, %NULL otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_find_resource")]
extern "C" fn pci_find_resource(dev: *mut pci_dev, res: *mut resource) -> *mut resource {
for i in 0i32..6i32 {
let r: *mut resource = unsafe { (*dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(i as isize);
let __v0: bool = (unsafe { (*r).start }) != 0 && resource_contains(r as *const resource, res as *const resource);
if __v0 {
return r;
}
}
core::ptr::null_mut::<resource>()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_find_resource_734: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, *mut resource) -> *mut resource>, *mut ()>(Some(pci_find_resource)) };

/// pci_resource_name - Return the name of the PCI resource
/// @dev: PCI device to query
/// @i: index of the resource
///
/// Return the standard PCI resource (BAR) name according to their index.
#[unsafe(no_mangle)]
extern "C" fn pci_resource_name(dev: *mut pci_dev, i: u32) -> *const u8 {
// "io" included in %pR
// "mem" included in %pR
// "mem pref" included in %pR
// I/O
// I/O
// mem
// mem
if ((unsafe { (*dev).hdr_type }) as i32) == 2 && (i as u64) < ARRAY_SIZE__17!() {
return unsafe { cardbus_name.0[i as usize] };
}
if (i as u64) < ARRAY_SIZE__13!() {
return unsafe { bar_name.0[i as usize] };
}
c"unknown".as_ptr() as *const u8
}

static bar_name: __SlateAlign16<[*const u8; 10]> = __SlateAlign16([(b"BAR 0\0".as_ptr() as *mut u8) as *const u8, (b"BAR 1\0".as_ptr() as *mut u8) as *const u8, (b"BAR 2\0".as_ptr() as *mut u8) as *const u8, (b"BAR 3\0".as_ptr() as *mut u8) as *const u8, (b"BAR 4\0".as_ptr() as *mut u8) as *const u8, (b"BAR 5\0".as_ptr() as *mut u8) as *const u8, (b"ROM\0".as_ptr() as *mut u8) as *const u8, (b"bridge window\0".as_ptr() as *mut u8) as *const u8, (b"bridge window\0".as_ptr() as *mut u8) as *const u8, (b"bridge window\0".as_ptr() as *mut u8) as *const u8]);

static cardbus_name: __SlateAlign16<[*const u8; 10]> = __SlateAlign16([(b"BAR 1\0".as_ptr() as *mut u8) as *const u8, (b"unknown\0".as_ptr() as *mut u8) as *const u8, (b"unknown\0".as_ptr() as *mut u8) as *const u8, (b"unknown\0".as_ptr() as *mut u8) as *const u8, (b"unknown\0".as_ptr() as *mut u8) as *const u8, (b"unknown\0".as_ptr() as *mut u8) as *const u8, (b"CardBus bridge window 0\0".as_ptr() as *mut u8) as *const u8, (b"CardBus bridge window 1\0".as_ptr() as *mut u8) as *const u8, (b"CardBus bridge window 0\0".as_ptr() as *mut u8) as *const u8, (b"CardBus bridge window 1\0".as_ptr() as *mut u8) as *const u8]);

/// pci_wait_for_pending - wait for @mask bit(s) to clear in status word @pos
/// @dev: the PCI device to operate on
/// @pos: config space offset of status word
/// @mask: mask of bit(s) to care about in status word
///
/// Return 1 when mask bit(s) in status word clear, 0 otherwise.
#[unsafe(no_mangle)]
extern "C" fn pci_wait_for_pending(dev: *mut pci_dev, pos: i32, mask: u16) -> i32 {
// Wait for Transaction Pending bit clean
for i in 0i32..4i32 {
let mut status: u16 = 0;
if i != 0 {
unsafe { msleep((1i32 << i.wrapping_sub(1)).wrapping_mul(100) as u32) };
}
unsafe { pci_read_config_word(dev as *const pci_dev, pos, core::ptr::addr_of_mut!(status)) };
if (status as i32) & (mask as i32) == 0 {
return 1;
}
}
0
}

static mut pci_acs_enable: i32 = unsafe { core::mem::zeroed() };

/// pci_request_acs - ask for ACS to be enabled if supported
#[unsafe(no_mangle)]
extern "C" fn pci_request_acs() {
unsafe {
pci_acs_enable = 1;
}
}

static mut disable_acs_redir_param: *const u8 = unsafe { core::mem::zeroed() };

static mut config_acs_param: *const u8 = unsafe { core::mem::zeroed() };

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_acs {
    ctrl: u16,
    fw_ctrl: u16,
}


fn __pci_config_acs(dev: *mut pci_dev, caps: &mut pci_acs, mut p: *const u8, acs_mask: u16, mut flags: u16) {
let mut mask: u16 = acs_mask;
let mut ret: i32 = 0;
if p.is_null() {
return;
}
while (unsafe { *p }) != 0 {
if acs_mask == 0 {
// Check for ACS flags
let delimit: *mut u8 = unsafe { strstr(p, c"@".as_ptr() as *const u8) };
if !delimit.is_null() {
let mut end: i32 = ((unsafe { delimit.offset_from(p as *mut u8) }) as i64).wrapping_sub(1) as i32;
let mut shift: u32 = 0;
mask = 0;
flags = 0;
while end > 1i32.wrapping_neg() {
if ((unsafe { *p.wrapping_offset(end as isize) }) as i32) == 48 {
mask = ((mask as i32) | 1i32 << shift) as u16;
shift = shift.wrapping_add(1);
end = end.wrapping_sub(1);
} else {
if ((unsafe { *p.wrapping_offset(end as isize) }) as i32) == 49 {
mask = ((mask as i32) | 1i32 << shift) as u16;
flags = ((flags as i32) | 1i32 << shift) as u16;
shift = shift.wrapping_add(1);
end = end.wrapping_sub(1);
} else {
if ((unsafe { *p.wrapping_offset(end as isize) }) as i32) == 120 || ((unsafe { *p.wrapping_offset(end as isize) }) as i32) == 88 {
shift = shift.wrapping_add(1);
end = end.wrapping_sub(1);
} else {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Invalid ACS flags... Ignoring\n".as_ptr() as *const u8) };
return;
}
}
}
}
p = delimit.wrapping_offset(1 as isize) as *const u8;
} else {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"ACS Flags missing\n".as_ptr() as *const u8) };
return;
}
}
if (mask as i32) & !(1 | 2 | 4 | 8 | 16 | 32 | 64) != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Invalid ACS flags specified\n".as_ptr() as *const u8) };
return;
}
ret = pci_dev_str_match(dev, p, core::ptr::addr_of_mut!(p));
if ret < 0 {
let mut __v22: i64;
let __ret_do_once: bool = true;
let mut __v23: i64;
let __ret_cond: bool = __ret_do_once;
let mut __ret_once: bool = false;
if unlikely__2!(__ret_cond) != 0 && unlikely__2!(!unsafe { __already_done__966 }) != 0 {
unsafe {
__already_done__966 = true;
}
__ret_once = true;
}
__v23 = __ret_once as i64;
if __v23 != 0 {
let mut __v24: i32;
while false {
}
__v24 = unsafe { _printk(c"\x014PCI: Can't parse ACS command line parameter\n".as_ptr() as *const u8) };
}
__v22 = __ret_do_once as i64;
break;
} else {
if ret == 1 {
// Found a match
break;
}
}
if ((unsafe { *p }) as i32) != 59 && ((unsafe { *p }) as i32) != 44 {
// End of param or invalid format
break;
}
let __v27: *const u8 = p;
p = __v27.wrapping_offset(1 as isize);
}
if ret != 1 {
return;
}
if (unsafe { pci_dev_specific_disable_acs_redir(dev) }) == 0 {
return;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"ACS mask  = %#06x\n".as_ptr() as *const u8, mask as i32) };
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"ACS flags = %#06x\n".as_ptr() as *const u8, flags as i32) };
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"ACS control = %#06x\n".as_ptr() as *const u8, (*(caps as *mut pci_acs)).ctrl as i32) };
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"ACS fw_ctrl = %#06x\n".as_ptr() as *const u8, (*(caps as *mut pci_acs)).fw_ctrl as i32) };
}
// For mask bits that are 0, copy them from the firmware setting
// and apply flags for all the mask bits that are 1.
unsafe {
(*(caps as *mut pci_acs)).ctrl = (((*(caps as *mut pci_acs)).fw_ctrl as i32) & !(mask as i32) | (flags as i32) & (mask as i32)) as u16;
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Configured ACS to %#06x\n".as_ptr() as *const u8, (*(caps as *mut pci_acs)).ctrl as i32) };
}

#[unsafe(link_section = ".data..once")]
static mut __already_done__966: bool = unsafe { core::mem::zeroed() };

/// pci_std_enable_acs - enable ACS on devices using standard ACS capabilities
/// @dev: the PCI device
/// @caps: default ACS controls
fn pci_std_enable_acs(dev: *mut pci_dev, caps: &mut pci_acs) {
// Source Validation
unsafe {
(*caps).ctrl = (((*caps).ctrl as i32) | ((*dev).acs_capabilities as i32) & 1) as u16;
}
// P2P Request Redirect
unsafe {
(*caps).ctrl = (((*caps).ctrl as i32) | ((*dev).acs_capabilities as i32) & 4) as u16;
}
// P2P Completion Redirect
unsafe {
(*caps).ctrl = (((*caps).ctrl as i32) | ((*dev).acs_capabilities as i32) & 8) as u16;
}
// Upstream Forwarding
unsafe {
(*caps).ctrl = (((*caps).ctrl as i32) | ((*dev).acs_capabilities as i32) & 16) as u16;
}
// Enable Translation Blocking for external devices and noats
if pci_ats_disabled() || ((unsafe { (*dev).__slate_bits_3.__get_external_facing() }) as i32) != 0 || ((unsafe { (*dev).__slate_bits_3.__get_untrusted() }) as i32) != 0 {
unsafe {
(*caps).ctrl = (((*caps).ctrl as i32) | ((*dev).acs_capabilities as i32) & 2) as u16;
}
}
}

/// pci_enable_acs - enable ACS if hardware support it
/// @dev: the PCI device
#[unsafe(no_mangle)]
extern "C" fn pci_enable_acs(dev: *mut pci_dev) {
let mut caps: pci_acs = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(caps) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let mut enable_acs: bool = false;
let mut pos: i32 = 0;
// If an iommu is present we start with kernel default caps
if (unsafe { pci_acs_enable }) != 0 {
if (unsafe { pci_dev_specific_enable_acs(dev) }) != 0 {
enable_acs = true;
}
}
pos = (unsafe { (*dev).acs_cap }) as i32;
if pos == 0 {
return;
}
unsafe { pci_read_config_word(dev as *const pci_dev, pos.wrapping_add(6), core::ptr::addr_of_mut!(caps.ctrl)) };
caps.fw_ctrl = caps.ctrl;
if enable_acs {
pci_std_enable_acs(dev, unsafe { &mut (*core::ptr::addr_of_mut!(caps)) });
}
// Always apply caps from the command line, even if there is no iommu.
// Trust that the admin has a reason to change the ACS settings.
__pci_config_acs(dev, unsafe { &mut (*core::ptr::addr_of_mut!(caps)) }, unsafe { disable_acs_redir_param }, 44u16, 65491u16);
__pci_config_acs(dev, unsafe { &mut (*core::ptr::addr_of_mut!(caps)) }, unsafe { config_acs_param }, 0u16, 0u16);
unsafe { pci_write_config_word(dev as *const pci_dev, pos.wrapping_add(6), caps.ctrl) };
}

/// pci_restore_bars - restore a device's BAR values (e.g. after wake-up)
/// @dev: PCI device to have its BARs restored
///
/// Restore the BAR values for a given device, so as to make it
/// accessible by its driver.
fn pci_restore_bars(dev: *mut pci_dev) {
for i in 0i32..7i32 {
unsafe { pci_update_resource(dev, i) };
}
}

fn platform_pci_power_manageable(dev: *mut pci_dev) -> bool {
if pci_use_mid_pm() {
return true;
}
unsafe { acpi_pci_power_manageable(dev) }
}

fn platform_pci_set_power_state(dev: *mut pci_dev, t: i32) -> i32 {
if pci_use_mid_pm() {
return mid_pci_set_power_state(dev, t);
}
unsafe { acpi_pci_set_power_state(dev, t) }
}

fn platform_pci_get_power_state(dev: *mut pci_dev) -> i32 {
if pci_use_mid_pm() {
return mid_pci_get_power_state(dev);
}
unsafe { acpi_pci_get_power_state(dev) }
}

fn platform_pci_refresh_power_state(dev: *mut pci_dev) {
if !pci_use_mid_pm() {
unsafe { acpi_pci_refresh_power_state(dev) };
}
}

fn platform_pci_choose_state(dev: *mut pci_dev) -> i32 {
if pci_use_mid_pm() {
return 1i32.wrapping_neg();
}
unsafe { acpi_pci_choose_state(dev) }
}

fn platform_pci_set_wakeup(dev: *mut pci_dev, enable: bool) -> i32 {
if pci_use_mid_pm() {
return 1i32.wrapping_neg();
}
unsafe { acpi_pci_wakeup(dev, enable) }
}

fn platform_pci_need_resume(dev: *mut pci_dev) -> bool {
if pci_use_mid_pm() {
return false;
}
unsafe { acpi_pci_need_resume(dev) }
}

fn platform_pci_bridge_d3(dev: *mut pci_dev) -> bool {
if pci_use_mid_pm() {
return false;
}
unsafe { acpi_pci_bridge_d3(dev) }
}

#[unsafe(no_mangle)]
extern "C" fn platform_pci_configure_wake(dev: *mut pci_dev) {
pci_configure_of_wake_gpio(dev);
}

#[unsafe(no_mangle)]
extern "C" fn platform_pci_remove_wake(dev: *mut pci_dev) {
pci_remove_of_wake_gpio(dev);
}

/// pci_update_current_state - Read power state of given device and cache it
/// @dev: PCI device to handle.
/// @state: State to cache in case the device doesn't have the PM capability
///
/// The power state is read from the PMCSR register, which however is
/// inaccessible in D3cold.  The platform firmware is therefore queried first
/// to detect accessibility of the register.  In case the platform firmware
/// reports an incorrect state or the device isn't power manageable by the
/// platform at all, we try to detect D3cold by testing accessibility of the
/// vendor ID in config space.
#[unsafe(no_mangle)]
extern "C" fn pci_update_current_state(dev: *mut pci_dev, state: i32) {
if platform_pci_get_power_state(dev) == 4 {
unsafe {
(*dev).current_state = 4;
}
} else {
if (unsafe { (*dev).pm_cap }) != 0 {
let mut pmcsr: u16 = 0;
unsafe { pci_read_config_word(dev as *const pci_dev, ((*dev).pm_cap as i32).wrapping_add(4), core::ptr::addr_of_mut!(pmcsr)) };
if PCI_POSSIBLE_ERROR__2!(pmcsr) {
unsafe {
(*dev).current_state = 4;
}
return;
}
unsafe {
(*dev).current_state = (pmcsr as i32) & 3;
}
} else {
unsafe {
(*dev).current_state = state;
}
}
}
}

/// pci_refresh_power_state - Refresh the given device's power state data
/// @dev: Target PCI device.
///
/// Ask the platform to refresh the devices power state information and invoke
/// pci_update_current_state() to update its current PCI power state.
#[unsafe(no_mangle)]
extern "C" fn pci_refresh_power_state(dev: *mut pci_dev) {
platform_pci_refresh_power_state(dev);
pci_update_current_state(dev, unsafe { (*dev).current_state });
}

/// pci_platform_power_transition - Use platform to change device power state
/// @dev: PCI device to handle.
/// @state: State to put the device into.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_platform_power_transition")]
extern "C" fn pci_platform_power_transition(dev: *mut pci_dev, state: i32) -> i32 {
let error: i32 = platform_pci_set_power_state(dev, state);
if error == 0 {
pci_update_current_state(dev, state);
} else {
if (unsafe { (*dev).pm_cap }) == 0 {
unsafe {
(*dev).current_state = 0;
}
}
}
// Fall back to PCI_D0
error
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_platform_power_transition_735: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>, *mut ()>(Some(pci_platform_power_transition)) };

#[unsafe(link_section = ".text.slate_distinct.pci.pci_resume_one")]
extern "C" fn pci_resume_one(pci_dev: *mut pci_dev, ign: *mut ()) -> i32 {
pm_request_resume(unsafe { core::ptr::addr_of_mut!((*pci_dev).dev) });
0
}

/// pci_resume_bus - Walk given bus and runtime resume devices on it
/// @bus: Top bus of the subtree to walk.
#[unsafe(no_mangle)]
extern "C" fn pci_resume_bus(bus: *mut pci_bus) {
if !bus.is_null() {
unsafe { pci_walk_bus(bus, Some(pci_resume_one), core::ptr::null_mut::<()>()) };
}
}

fn pci_dev_wait(dev: *mut pci_dev, reset_type: *mut u8, timeout: i32) -> i32 {
let mut delay: i32 = 1;
let mut retrain: bool = false;
let mut bridge: *mut pci_dev = unsafe { core::mem::zeroed() };
let root: *mut pci_dev = pcie_find_root_port(dev);
if pci_is_pcie(dev) {
bridge = pci_upstream_bridge(dev);
if !bridge.is_null() {
retrain = true;
}
}
// The caller has already waited long enough after a reset that the
// device should respond to config requests, but it may respond
// with Request Retry Status (RRS) if it needs more time to
// initialize.
//
// If the device is below a Root Port with Configuration RRS
// Software Visibility enabled, reading the Vendor ID returns a
// special data value if the device responded with RRS.  Read the
// Vendor ID until we get non-RRS status.
//
// If there's no Root Port or Configuration RRS Software Visibility
// is not enabled, the device may still respond with RRS, but
// hardware may retry the config request.  If no retries receive
// Successful Completion, hardware generally synthesizes ~0
// (PCI_ERROR_RESPONSE) data to complete the read.  Reading Vendor
// ID for VFs and non-existent devices also returns ~0, so read the
// Command register until it returns something other than ~0.
'__slate_break_0: loop {
'__slate_continue_0: {
let mut id: u32 = 0;
if pci_dev_is_disconnected(dev as *const pci_dev) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"disconnected; not waiting\n".as_ptr() as *const u8) };
}
return 25i32.wrapping_neg();
}
if !root.is_null() && ((unsafe { (*root).__slate_bits_1.__get_config_rrs_sv() }) as i32) != 0 {
unsafe { pci_read_config_dword(dev as *const pci_dev, 0i32, core::ptr::addr_of_mut!(id)) };
if !pci_bus_rrs_vendor_id(id) {
break '__slate_break_0;
}
} else {
unsafe { pci_read_config_dword(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(id)) };
if !PCI_POSSIBLE_ERROR!(id) {
break '__slate_break_0;
}
}
if delay > timeout {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"not ready %dms after %s; giving up\n".as_ptr() as *const u8, delay.wrapping_sub(1), reset_type) };
return 25i32.wrapping_neg();
}
if delay > 1000 {
if retrain {
retrain = false;
if (unsafe { pcie_failed_link_retrain(bridge) }) == 0 {
delay = 1;
break '__slate_continue_0;
}
}
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"not ready %dms after %s; waiting\n".as_ptr() as *const u8, delay.wrapping_sub(1), reset_type) };
}
unsafe { msleep(delay as u32) };
delay = delay.wrapping_mul(2);
}
}
if delay > 1000 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"ready %dms after %s\n".as_ptr() as *const u8, delay.wrapping_sub(1), reset_type) };
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"ready %dms after %s\n".as_ptr() as *const u8, delay.wrapping_sub(1), reset_type) };
}
}
0
}

/// pci_power_up - Put the given device into D0
/// @dev: PCI device to power up
///
/// On success, return 0 or 1, depending on whether or not it is necessary to
/// restore the device's BARs subsequently (1 is returned in that case).
///
/// On failure, return a negative error code.  Always return failure if @dev
/// lacks a Power Management Capability, even if the platform was able to
/// put the device in D0 via non-PCI means.
#[unsafe(no_mangle)]
extern "C" fn pci_power_up(dev: *mut pci_dev) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut pmcsr: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u16 = core::ptr::addr_of_mut!(pmcsr) as *mut u16;
let mut state: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(state) as *mut i32;
let mut need_restore: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut bool = core::ptr::addr_of_mut!(need_restore) as *mut bool;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 2usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// When setting power state to D0, platform_pci_set_power_state()
// ensures main power is on.  If it puts the device in D0, it also
// completes any required delays after the transition; if it leaves
// the device in D1, D2, or D3hot, we use the PM Capability to
// transition to D0.
//
// In all cases, the device is either Configuration-Ready or
// inaccessible upon return.
platform_pci_set_power_state(dev, 0i32);
if (*dev).pm_cap == 0 {
*__slate_slot_2 = platform_pci_get_power_state(dev);
if *__slate_slot_2 == 5 {
(*dev).current_state = 0;
} else {
(*dev).current_state = *__slate_slot_2;
}
return 5i32.wrapping_neg();
} else {
if pci_dev_is_disconnected(dev as *const pci_dev) {
(*dev).current_state = 4;
return 5i32.wrapping_neg();
} else {
pci_read_config_word(dev as *const pci_dev, ((*dev).pm_cap as i32).wrapping_add(4), core::ptr::addr_of_mut!(*__slate_slot_1));
if PCI_POSSIBLE_ERROR__2!(*__slate_slot_1) {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Unable to change power state from %s to D0, device inaccessible\n".as_ptr() as *const u8, pci_power_name((*dev).current_state));
(*dev).current_state = 4;
return 5i32.wrapping_neg();
} else {
*__slate_slot_2 = (*__slate_slot_1 as i32) & 3;
*__slate_slot_3 = (*__slate_slot_2 == 3 || (*dev).current_state >= 3) && (*__slate_slot_1 as i32) & 8 == 0;
if *__slate_slot_2 != 0 {
// Force the entire word to 0. This doesn't affect PME_Status, disables
// PME_En, and sets PowerState to 0.
pci_write_config_word(dev as *const pci_dev, ((*dev).pm_cap as i32).wrapping_add(4), 0u16);
// Mandatory transition delays; see PCI PM 1.2.
if *__slate_slot_2 == 3 {
pci_dev_d3_sleep(dev);
if (*__slate_slot_1 as i32) & 8 == 0 {
*__slate_slot_0 = pci_dev_wait(dev, c"power up D3hot->D0uninitialized".as_ptr() as *mut u8, 60000i32);
if *__slate_slot_0 != 0 {
(*dev).current_state = 4;
return 5i32.wrapping_neg();
}
}
} else {
if *__slate_slot_2 == 2 {
udelay(200u64);
}
}
}
(*dev).current_state = 0;
if *__slate_slot_3 {
return 1;
} else {
return 0;
}
}
}
}
}
}

/// pci_set_full_power_state - Put a PCI device into D0 and update its state
/// @dev: PCI device to power up
/// @locked: whether pci_bus_sem is held
///
/// Call pci_power_up() to put @dev into D0, read from its PCI_PM_CTRL register
/// to confirm the state change, restore its BARs if they might be lost and
/// reconfigure ASPM in accordance with the new power state.
///
/// If pci_restore_state() is going to be called right after a power state change
/// to D0, it is more efficient to use pci_power_up() directly instead of this
/// function.
fn pci_set_full_power_state(dev: *mut pci_dev, locked: bool) -> i32 {
let mut pmcsr: u16 = 0;
let ret: i32 = pci_power_up(dev);
if ret < 0 {
if (unsafe { (*dev).current_state }) == 0 {
return 0;
}
return ret;
}
unsafe { pci_read_config_word(dev as *const pci_dev, ((*dev).pm_cap as i32).wrapping_add(4), core::ptr::addr_of_mut!(pmcsr)) };
unsafe {
(*dev).current_state = (pmcsr as i32) & 3;
}
if (unsafe { (*dev).current_state }) != 0 {
loop {
if __ratelimit!(unsafe { core::ptr::addr_of_mut!(_rs__1430) }) != 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Refused to change power state from %s to D0\n".as_ptr() as *const u8, pci_power_name((*dev).current_state)) };
}
if true {
break;
}
}
} else {
if ret > 0 {
// According to section 5.4.1 of the "PCI BUS POWER MANAGEMENT
// INTERFACE SPECIFICATION, REV. 1.2", a device transitioning
// from D3hot to D0 _may_ perform an internal reset, thereby
// going to "D0 Uninitialized" rather than "D0 Initialized".
// For example, at least some versions of the 3c905B and the
// 3c556B exhibit this behaviour.
//
// At least some laptop BIOSen (e.g. the Thinkpad T21) leave
// devices in a D3hot state at boot.  Consequently, we need to
// restore at least the BARs so that the device will be
// accessible to its driver.
pci_restore_bars(dev);
}
}
if !unsafe { (*(*dev).bus).self_ }.is_null() {
unsafe { pcie_aspm_pm_state_change((*(*dev).bus).self_, locked) };
}
0
}

static mut _rs__1430: ratelimit_state = {
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

/// __pci_dev_set_current_state - Set current state of a PCI device
/// @dev: Device to handle
/// @data: pointer to state to be set
#[unsafe(link_section = ".text.slate_distinct.pci.__pci_dev_set_current_state")]
extern "C" fn __pci_dev_set_current_state(dev: *mut pci_dev, data: *mut ()) -> i32 {
let state: i32 = unsafe { *(data as *mut i32) };
unsafe {
(*dev).current_state = state;
}
0
}

/// pci_bus_set_current_state - Walk given bus and set current state of devices
/// @bus: Top bus of the subtree to walk.
/// @state: state to be set
#[unsafe(no_mangle)]
extern "C" fn pci_bus_set_current_state(bus: *mut pci_bus, mut state: i32) {
if !bus.is_null() {
unsafe { pci_walk_bus(bus, Some(__pci_dev_set_current_state), core::ptr::addr_of_mut!(state) as *mut ()) };
}
}

fn __pci_bus_set_current_state(bus: *mut pci_bus, mut state: i32, locked: bool) {
if bus.is_null() {
return;
}
if locked {
unsafe { pci_walk_bus_locked(bus, Some(__pci_dev_set_current_state), core::ptr::addr_of_mut!(state) as *mut ()) };
} else {
unsafe { pci_walk_bus(bus, Some(__pci_dev_set_current_state), core::ptr::addr_of_mut!(state) as *mut ()) };
}
}

/// pci_set_low_power_state - Put a PCI device into a low-power state.
/// @dev: PCI device to handle.
/// @state: PCI power state (D1, D2, D3hot) to put the device into.
/// @locked: whether pci_bus_sem is held
///
/// Use the device's PCI_PM_CTRL register to put it into a low-power state.
///
/// RETURN VALUE:
/// -EINVAL if the requested state is invalid.
/// -EIO if device does not support PCI PM or its PM capabilities register has a
/// wrong version, or device doesn't support the requested state.
/// 0 if device already is in the requested state.
/// 0 if device's power state has been successfully changed.
fn pci_set_low_power_state(dev: *mut pci_dev, state: i32, locked: bool) -> i32 {
let mut pmcsr: u16 = 0;
if (unsafe { (*dev).pm_cap }) == 0 {
return 5i32.wrapping_neg();
}
// Validate transition: We can enter D0 from any state, but if
// we're already in a low-power state, we can only go deeper.  E.g.,
// we can go from D1 to D3, but we can't go directly from D3 to D1;
// we'd have to go from D3 to D0, then to D1.
if (unsafe { (*dev).current_state }) <= 4 && (unsafe { (*dev).current_state }) > state {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Invalid power transition (from %s to %s)\n".as_ptr() as *const u8, pci_power_name((*dev).current_state), pci_power_name(state)) };
}
return 22i32.wrapping_neg();
}
// Check if this device supports the desired state
if state == 1 && ((unsafe { (*dev).__slate_bits_1.__get_d1_support() }) as i32) == 0 || state == 2 && ((unsafe { (*dev).__slate_bits_1.__get_d2_support() }) as i32) == 0 {
return 5i32.wrapping_neg();
}
if (unsafe { (*dev).current_state }) == state {
return 0;
}
unsafe { pci_read_config_word(dev as *const pci_dev, ((*dev).pm_cap as i32).wrapping_add(4), core::ptr::addr_of_mut!(pmcsr)) };
if PCI_POSSIBLE_ERROR__2!(pmcsr) {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Unable to change power state from %s to %s, device inaccessible\n".as_ptr() as *const u8, pci_power_name((*dev).current_state), pci_power_name(state)) };
unsafe {
(*dev).current_state = 4;
}
return 5i32.wrapping_neg();
}
let __v2: u16 = ((pmcsr as i32) & !3) as u16;
pmcsr = __v2;
pmcsr = ((pmcsr as i32) | state) as u16;
// Enter specified state
unsafe { pci_write_config_word(dev as *const pci_dev, ((*dev).pm_cap as i32).wrapping_add(4), pmcsr) };
// Mandatory power management transition delays; see PCI PM 1.2.
if state == 3 {
pci_dev_d3_sleep(dev);
} else {
if state == 2 {
udelay(200u64);
}
}
unsafe { pci_read_config_word(dev as *const pci_dev, ((*dev).pm_cap as i32).wrapping_add(4), core::ptr::addr_of_mut!(pmcsr)) };
unsafe {
(*dev).current_state = (pmcsr as i32) & 3;
}
if (unsafe { (*dev).current_state }) != state {
loop {
if __ratelimit__2!(unsafe { core::ptr::addr_of_mut!(_rs__1557) }) != 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Refused to change power state from %s to %s\n".as_ptr() as *const u8, pci_power_name((*dev).current_state), pci_power_name(state)) };
}
if true {
break;
}
}
}
0
}

static mut _rs__1557: ratelimit_state = {
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

fn __pci_set_power_state(dev: *mut pci_dev, mut state: i32, locked: bool) -> i32 {
let mut error: i32 = 0;
// Bound the state we're entering
if state > 4 {
state = 4;
} else {
if state < 0 {
state = 0;
} else {
let __v0: bool = (state == 1 || state == 2) && pci_no_d1d2(dev) != 0;
if __v0 {
return 0;
}
}
}
// If the device or the parent bridge do not support PCI
// PM, ignore the request if we're doing anything other
// than putting it into D0 (which would only happen on
// boot).
// Check if we're already there
if (unsafe { (*dev).current_state }) == state {
return 0;
}
if state == 0 {
return pci_set_full_power_state(dev, locked);
}
// This device is quirked not to be put into D3, so don't put it in
// D3
if state >= 3 && ((unsafe { (*dev).dev_flags }) as i32) & 2 != 0 {
return 0;
}
if state == 4 {
// To put the device in D3cold, put it into D3hot in the native
// way, then put it into D3cold using platform ops.
error = pci_set_low_power_state(dev, 3i32, locked);
if pci_platform_power_transition(dev, 4i32) != 0 {
return error;
}
// Powering off a bridge may power off the whole hierarchy
if (unsafe { (*dev).current_state }) == 4 {
__pci_bus_set_current_state(unsafe { (*dev).subordinate }, 4i32, locked);
}
} else {
error = pci_set_low_power_state(dev, state, locked);
if pci_platform_power_transition(dev, state) != 0 {
return error;
}
}
0
}

/// pci_set_power_state - Set the power state of a PCI device
/// @dev: PCI device to handle.
/// @state: PCI power state (D0, D1, D2, D3hot) to put the device into.
///
/// Transition a device to a new power state, using the platform firmware and/or
/// the device's PCI PM registers.
///
/// RETURN VALUE:
/// -EINVAL if the requested state is invalid.
/// -EIO if device does not support PCI PM or its PM capabilities register has a
/// wrong version, or device doesn't support the requested state.
/// 0 if the transition is to D1 or D2 but D1 and D2 are not supported.
/// 0 if device already is in the requested state.
/// 0 if the transition is to D3 but D3 is not supported.
/// 0 if device's power state has been successfully changed.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_set_power_state")]
extern "C" fn pci_set_power_state(dev: *mut pci_dev, state: i32) -> i32 {
__pci_set_power_state(dev, state, false)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_set_power_state_736: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>, *mut ()>(Some(pci_set_power_state)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_set_power_state_locked")]
extern "C" fn pci_set_power_state_locked(dev: *mut pci_dev, state: i32) -> i32 {
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(pci_bus_sem) });
__pci_set_power_state(dev, state, true)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_set_power_state_locked_737: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>, *mut ()>(Some(pci_set_power_state_locked)) };

fn _pci_find_saved_cap(pci_dev: *mut pci_dev, cap: u16, extended: bool) -> *mut pci_cap_saved_state {
let mut tmp: *mut pci_cap_saved_state = unsafe { core::mem::zeroed() };
tmp = hlist_entry_safe!(unsafe { (*core::ptr::addr_of_mut!((*pci_dev).saved_cap_space)).first });
while !tmp.is_null() {
if ((unsafe { (*tmp).cap.cap_extended }) as i32) == (extended as i32) && ((unsafe { (*tmp).cap.cap_nr }) as i32) == (cap as i32) {
return tmp;
}
tmp = hlist_entry_safe!(unsafe { (*tmp).next.next });
}
core::ptr::null_mut::<pci_cap_saved_state>()
}

#[unsafe(no_mangle)]
extern "C" fn pci_find_saved_cap(dev: *mut pci_dev, cap: u8) -> *mut pci_cap_saved_state {
_pci_find_saved_cap(dev, cap as u16, false)
}

#[unsafe(no_mangle)]
extern "C" fn pci_find_saved_ext_cap(dev: *mut pci_dev, cap: u16) -> *mut pci_cap_saved_state {
_pci_find_saved_cap(dev, cap, true)
}

fn pci_save_pcie_state(dev: *mut pci_dev) -> i32 {
let mut i: i32 = 0;
let mut save_state: *mut pci_cap_saved_state = unsafe { core::mem::zeroed() };
if !pci_is_pcie(dev) {
return 0;
}
save_state = pci_find_saved_cap(dev, 16u8);
if save_state.is_null() {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"buffer not found in %s\n".as_ptr() as *const u8, c"pci_save_pcie_state".as_ptr() as *mut u8) };
return 12i32.wrapping_neg();
}
let cap: *mut u16 = unsafe { core::ptr::addr_of_mut!((*save_state).cap.data) as *mut u32 }.wrapping_offset(0 as isize) as *mut u16;
let __v1: i32 = i;
i = __v1.wrapping_add(1);
unsafe { pcie_capability_read_word(dev, 8i32, cap.wrapping_offset(__v1 as isize)) };
let __v3: i32 = i;
i = __v3.wrapping_add(1);
unsafe { pcie_capability_read_word(dev, 16i32, cap.wrapping_offset(__v3 as isize)) };
let __v5: i32 = i;
i = __v5.wrapping_add(1);
unsafe { pcie_capability_read_word(dev, 24i32, cap.wrapping_offset(__v5 as isize)) };
let __v7: i32 = i;
i = __v7.wrapping_add(1);
unsafe { pcie_capability_read_word(dev, 28i32, cap.wrapping_offset(__v7 as isize)) };
let __v9: i32 = i;
i = __v9.wrapping_add(1);
unsafe { pcie_capability_read_word(dev, 40i32, cap.wrapping_offset(__v9 as isize)) };
let __v11: i32 = i;
i = __v11.wrapping_add(1);
unsafe { pcie_capability_read_word(dev, 48i32, cap.wrapping_offset(__v11 as isize)) };
let __v13: i32 = i;
i = __v13.wrapping_add(1);
unsafe { pcie_capability_read_word(dev, 56i32, cap.wrapping_offset(__v13 as isize)) };
unsafe { pci_save_aspm_l1ss_state(dev) };
unsafe { pci_save_ltr_state(dev) };
0
}

fn pci_restore_pcie_state(dev: *mut pci_dev) {
let mut i: i32 = 0;
let mut save_state: *mut pci_cap_saved_state = unsafe { core::mem::zeroed() };
// Restore max latencies (in the LTR capability) before enabling
// LTR itself in PCI_EXP_DEVCTL2.
unsafe { pci_restore_ltr_state(dev) };
unsafe { pci_restore_aspm_l1ss_state(dev) };
save_state = pci_find_saved_cap(dev, 16u8);
if save_state.is_null() {
return;
}
// Downstream ports reset the LTR enable bit when link goes down.
// Check and re-configure the bit here before restoring device.
// PCIe r5.0, sec 7.5.3.16.
unsafe { pci_bridge_reconfigure_ltr(dev) };
let cap: *mut u16 = unsafe { core::ptr::addr_of_mut!((*save_state).cap.data) as *mut u32 }.wrapping_offset(0 as isize) as *mut u16;
let __v0: i32 = i;
i = __v0.wrapping_add(1);
unsafe { pcie_capability_write_word(dev, 8i32, *cap.wrapping_offset(__v0 as isize)) };
let __v2: i32 = i;
i = __v2.wrapping_add(1);
unsafe { pcie_capability_write_word(dev, 16i32, *cap.wrapping_offset(__v2 as isize)) };
let __v4: i32 = i;
i = __v4.wrapping_add(1);
unsafe { pcie_capability_write_word(dev, 24i32, *cap.wrapping_offset(__v4 as isize)) };
let __v6: i32 = i;
i = __v6.wrapping_add(1);
unsafe { pcie_capability_write_word(dev, 28i32, *cap.wrapping_offset(__v6 as isize)) };
let __v8: i32 = i;
i = __v8.wrapping_add(1);
unsafe { pcie_capability_write_word(dev, 40i32, *cap.wrapping_offset(__v8 as isize)) };
let __v10: i32 = i;
i = __v10.wrapping_add(1);
unsafe { pcie_capability_write_word(dev, 48i32, *cap.wrapping_offset(__v10 as isize)) };
let __v12: i32 = i;
i = __v12.wrapping_add(1);
unsafe { pcie_capability_write_word(dev, 56i32, *cap.wrapping_offset(__v12 as isize)) };
}

fn pci_save_pcix_state(dev: *mut pci_dev) -> i32 {
let mut pos: i32 = 0;
let mut save_state: *mut pci_cap_saved_state = unsafe { core::mem::zeroed() };
pos = pci_find_capability(dev, 7i32) as i32;
if pos == 0 {
return 0;
}
save_state = pci_find_saved_cap(dev, 7u8);
if save_state.is_null() {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"buffer not found in %s\n".as_ptr() as *const u8, c"pci_save_pcix_state".as_ptr() as *mut u8) };
return 12i32.wrapping_neg();
}
unsafe { pci_read_config_word(dev as *const pci_dev, pos.wrapping_add(2), (core::ptr::addr_of_mut!((*save_state).cap.data) as *mut u32) as *mut u16) };
0
}

fn pci_restore_pcix_state(dev: *mut pci_dev) {
let mut i: i32 = 0;
let mut pos: i32 = 0;
let mut save_state: *mut pci_cap_saved_state = unsafe { core::mem::zeroed() };
save_state = pci_find_saved_cap(dev, 7u8);
pos = pci_find_capability(dev, 7i32) as i32;
if save_state.is_null() || pos == 0 {
return;
}
let cap: *mut u16 = unsafe { core::ptr::addr_of_mut!((*save_state).cap.data) as *mut u32 }.wrapping_offset(0 as isize) as *mut u16;
let __v0: i32 = i;
i = __v0.wrapping_add(1);
unsafe { pci_write_config_word(dev as *const pci_dev, pos.wrapping_add(2), *cap.wrapping_offset(__v0 as isize)) };
}

/// pci_save_state - save the PCI configuration space of a device before
/// 		    suspending
/// @dev: PCI device that we're dealing with
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_save_state")]
extern "C" fn pci_save_state(dev: *mut pci_dev) -> i32 {
// XXX: 100% dword access ok here?
let mut i: i32 = 0;
while i < 16 {
unsafe { pci_read_config_dword(dev as *const pci_dev, i.wrapping_mul(4), ((*dev).saved_config_space.as_mut_ptr() as *mut u32).wrapping_offset(i as isize)) };
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"save config %#04x: %#010x\n".as_ptr() as *const u8, i.wrapping_mul(4), *((*dev).saved_config_space.as_mut_ptr() as *mut u32).wrapping_offset(i as isize)) };
}
i = i.wrapping_add(1);
}
unsafe {
(*dev).__slate_bits_3.__set_state_saved(1u32);
}
i = pci_save_pcie_state(dev);
if i != 0 {
return i;
}
i = pci_save_pcix_state(dev);
if i != 0 {
return i;
}
pci_save_dpc_state(dev);
pci_save_aer_state(dev);
pci_save_ptm_state(dev);
pci_save_tph_state(dev);
unsafe { pci_save_vc_state(dev) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_save_state_738: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_save_state)) };

fn pci_restore_config_dword(pdev: *mut pci_dev, offset: i32, saved_val: u32, force: bool) {
let mut val: u32 = 0;
unsafe { pci_read_config_dword(pdev as *const pci_dev, offset, core::ptr::addr_of_mut!(val)) };
if !force && val == saved_val {
return;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"restore config %#04x: %#010x -> %#010x\n".as_ptr() as *const u8, offset, val, saved_val) };
}
unsafe { pci_write_config_dword(pdev as *const pci_dev, offset, saved_val) };
}

fn pci_restore_config_space_range(pdev: *mut pci_dev, start: i32, mut index: i32, force: bool) {
while index >= start {
pci_restore_config_dword(pdev, 4i32.wrapping_mul(index), unsafe { *((*pdev).saved_config_space.as_mut_ptr() as *mut u32).wrapping_offset(index as isize) }, force);
index = index.wrapping_sub(1);
}
}

fn pci_restore_config_space(pdev: *mut pci_dev) {
if ((unsafe { (*pdev).hdr_type }) as i32) == 0 {
pci_restore_config_space_range(pdev, 10i32, 15i32, false);
// Restore BARs before the command register.
pci_restore_config_space_range(pdev, 4i32, 9i32, false);
pci_restore_config_space_range(pdev, 0i32, 3i32, false);
} else {
if ((unsafe { (*pdev).hdr_type }) as i32) == 1 {
pci_restore_config_space_range(pdev, 12i32, 15i32, false);
// Force rewriting of prefetch registers to avoid S3 resume
// issues on Intel PCI bridges that occur when these
// registers are not explicitly written.
pci_restore_config_space_range(pdev, 9i32, 11i32, true);
pci_restore_config_space_range(pdev, 0i32, 8i32, false);
} else {
pci_restore_config_space_range(pdev, 0i32, 15i32, false);
}
}
}

/// pci_restore_state - Restore the saved state of a PCI device
/// @dev: PCI device that we're dealing with
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_restore_state")]
extern "C" fn pci_restore_state(dev: *mut pci_dev) {
pci_restore_pcie_state(dev);
unsafe { pci_restore_pasid_state(dev) };
unsafe { pci_restore_pri_state(dev) };
unsafe { pci_restore_ats_state(dev) };
unsafe { pci_restore_vc_state(dev) };
unsafe { pci_restore_rebar_state(dev) };
pci_restore_dpc_state(dev);
pci_restore_ptm_state(dev);
pci_restore_tph_state(dev);
pci_aer_clear_status(dev);
pci_restore_aer_state(dev);
pci_restore_config_space(dev);
pci_restore_pcix_state(dev);
unsafe { pci_restore_msi_state(dev) };
// Restore ACS and IOV configuration state
pci_enable_acs(dev);
pci_restore_iov_state(dev);
unsafe {
(*dev).__slate_bits_3.__set_state_saved(0u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_restore_state_739: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_restore_state)) };

#[repr(C)]
#[derive(Clone, Copy)]
struct pci_saved_state {
    config_space: [u32; 16],
    cap: [pci_cap_saved_data; 0],
}


/// pci_store_saved_state - Allocate and return an opaque struct containing
/// 			   the device saved state.
/// @dev: PCI device that we're dealing with
///
/// Return NULL if no state or error.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_store_saved_state")]
extern "C" fn pci_store_saved_state(dev: *mut pci_dev) -> *mut pci_saved_state {
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_15: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut state: *mut pci_saved_state = unsafe { core::mem::zeroed() };
let mut tmp: *mut pci_cap_saved_state = unsafe { core::mem::zeroed() };
let mut cap: *mut pci_cap_saved_data = unsafe { core::mem::zeroed() };
let mut size: u64 = 0;
if ((unsafe { (*dev).__slate_bits_3.__get_state_saved() }) as i32) == 0 {
return core::ptr::null_mut::<pci_saved_state>();
}
size = 72u64;
tmp = hlist_entry_safe!(unsafe { (*core::ptr::addr_of_mut!((*dev).saved_cap_space)).first });
while !tmp.is_null() {
size = size.wrapping_add(8u64.wrapping_add((unsafe { (*tmp).cap.size }) as u64));
tmp = hlist_entry_safe!(unsafe { (*tmp).next.next });
}
let mut __v11: *mut ();
let mut __v12: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(size, 3264u32, unsafe { *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(size, 3264u32, unsafe { *{
__slate_compound_15 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_15)
} });
}
__v12 = _res;
__v11 = __v12;
state = __v11 as *mut pci_saved_state;
if state.is_null() {
return core::ptr::null_mut::<pci_saved_state>();
}
unsafe { memcpy(((*state).config_space.as_mut_ptr() as *mut u32) as *mut (), ((*dev).saved_config_space.as_mut_ptr() as *mut u32) as *const (), core::mem::size_of::<[u32; 16]>() as u64) };
cap = unsafe { core::ptr::addr_of_mut!((*state).cap) as *mut pci_cap_saved_data };
tmp = hlist_entry_safe!(unsafe { (*core::ptr::addr_of_mut!((*dev).saved_cap_space)).first });
while !tmp.is_null() {
let len: u64 = 8u64.wrapping_add((unsafe { (*tmp).cap.size }) as u64);
unsafe { memcpy(cap as *mut (), core::ptr::addr_of_mut!((*tmp).cap) as *const (), len) };
cap = (cap as *mut u8).wrapping_offset(len as isize) as *mut pci_cap_saved_data;
tmp = hlist_entry_safe!(unsafe { (*tmp).next.next });
}
// Empty cap_save terminates list
state
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_store_saved_state_740: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> *mut pci_saved_state>, *mut ()>(Some(pci_store_saved_state)) };

/// pci_load_saved_state - Reload the provided save state into struct pci_dev.
/// @dev: PCI device that we're dealing with
/// @state: Saved state returned from pci_store_saved_state()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_load_saved_state")]
extern "C" fn pci_load_saved_state(dev: *mut pci_dev, state: *mut pci_saved_state) -> i32 {
let mut cap: *mut pci_cap_saved_data = unsafe { core::mem::zeroed() };
unsafe {
(*dev).__slate_bits_3.__set_state_saved(0u32);
}
if state.is_null() {
return 0;
}
unsafe { memcpy(((*dev).saved_config_space.as_mut_ptr() as *mut u32) as *mut (), ((*state).config_space.as_mut_ptr() as *mut u32) as *const (), core::mem::size_of::<[u32; 16]>() as u64) };
cap = unsafe { core::ptr::addr_of_mut!((*state).cap) as *mut pci_cap_saved_data };
while (unsafe { (*cap).size }) != 0 {
let mut tmp: *mut pci_cap_saved_state = unsafe { core::mem::zeroed() };
tmp = _pci_find_saved_cap(dev, unsafe { (*cap).cap_nr }, unsafe { (*cap).cap_extended });
if tmp.is_null() || (unsafe { (*tmp).cap.size }) != unsafe { (*cap).size } {
return 22i32.wrapping_neg();
}
unsafe { memcpy((core::ptr::addr_of_mut!((*tmp).cap.data) as *mut u32) as *mut (), (core::ptr::addr_of_mut!((*cap).data) as *mut u32) as *const (), (*tmp).cap.size as u64) };
cap = (cap as *mut u8).wrapping_offset(8 as isize).wrapping_offset((unsafe { (*cap).size }) as isize) as *mut pci_cap_saved_data;
}
unsafe {
(*dev).__slate_bits_3.__set_state_saved(1u32);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_load_saved_state_741: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, *mut pci_saved_state) -> i32>, *mut ()>(Some(pci_load_saved_state)) };

/// pci_load_and_free_saved_state - Reload the save state pointed to by state,
/// 				   and free the memory allocated for it.
/// @dev: PCI device that we're dealing with
/// @state: Pointer to saved state returned from pci_store_saved_state()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_load_and_free_saved_state")]
extern "C" fn pci_load_and_free_saved_state(dev: *mut pci_dev, state: *mut *mut pci_saved_state) -> i32 {
let ret: i32 = pci_load_saved_state(dev, unsafe { *state });
unsafe { kfree(*state as *const ()) };
unsafe {
*state = core::ptr::null_mut::<pci_saved_state>();
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_load_and_free_saved_state_742: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, *mut *mut pci_saved_state) -> i32>, *mut ()>(Some(pci_load_and_free_saved_state)) };

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pcibios_enable_device(dev: *mut pci_dev, bars: i32) -> i32 {
unsafe { pci_enable_resources(dev, bars) }
}

fn pci_host_bridge_enable_device(dev: *mut pci_dev) -> i32 {
let host_bridge: *mut pci_host_bridge = unsafe { pci_find_host_bridge((*dev).bus) };
let mut err: i32 = 0;
if !host_bridge.is_null() && (unsafe { (*host_bridge).enable_device }) != None {
err = unsafe { unsafe { (*host_bridge).enable_device }.unwrap()(host_bridge, dev) };
if err != 0 {
return err;
}
}
0
}

fn pci_host_bridge_disable_device(dev: *mut pci_dev) {
let host_bridge: *mut pci_host_bridge = unsafe { pci_find_host_bridge((*dev).bus) };
if !host_bridge.is_null() && (unsafe { (*host_bridge).disable_device }) != None {
unsafe { unsafe { (*host_bridge).disable_device }.unwrap()(host_bridge, dev) };
}
}

fn do_pci_enable_device(dev: *mut pci_dev, bars: i32) -> i32 {
let mut pin: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u8 = core::ptr::addr_of_mut!(pin) as *mut u8;
let mut cmd: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u16 = core::ptr::addr_of_mut!(cmd) as *mut u16;
let mut bridge: core::mem::MaybeUninit<*mut pci_dev> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut pci_dev = core::ptr::addr_of_mut!(bridge) as *mut *mut pci_dev;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 2usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_3 = pci_set_power_state(dev, 0i32);
if *__slate_slot_3 < 0 && *__slate_slot_3 != 5i32.wrapping_neg() {
return *__slate_slot_3;
} else {
*__slate_slot_2 = pci_upstream_bridge(dev);
if !(*__slate_slot_2).is_null() {
pcie_aspm_powersave_config_link(*__slate_slot_2);
}
*__slate_slot_3 = pci_host_bridge_enable_device(dev);
if *__slate_slot_3 != 0 {
return *__slate_slot_3;
} else {
*__slate_slot_3 = pcibios_enable_device(dev, bars);
if *__slate_slot_3 < 0 {
pci_host_bridge_disable_device(dev);
return *__slate_slot_3;
} else {
pci_fixup_device(3u32, dev);
if ((*dev).__slate_bits_3.__get_msi_enabled() as i32) != 0 || ((*dev).__slate_bits_3.__get_msix_enabled() as i32) != 0 {
return 0;
} else {
pci_read_config_byte(dev as *const pci_dev, 61i32, core::ptr::addr_of_mut!(*__slate_slot_0));
if *__slate_slot_0 != 0 {
pci_read_config_word(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(*__slate_slot_1));
if (*__slate_slot_1 as i32) & 1024 != 0 {
pci_write_config_word(dev as *const pci_dev, 4i32, ((*__slate_slot_1 as i32) & !1024) as u16);
}
}
return 0;
}
}
}
}
}
}

/// pci_reenable_device - Resume abandoned device
/// @dev: PCI device to be resumed
///
/// NOTE: This function is a backend of pci_default_resume() and is not supposed
/// to be called by normal code, write proper resume handler and use it instead.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_reenable_device")]
extern "C" fn pci_reenable_device(dev: *mut pci_dev) -> i32 {
if pci_is_enabled(dev) != 0 {
return do_pci_enable_device(dev, 2047i32);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_reenable_device_743: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_reenable_device)) };

fn pci_enable_bridge(dev: *mut pci_dev) {
let mut retval: i32 = 0;
let bridge: *mut pci_dev = pci_upstream_bridge(dev);
if !bridge.is_null() {
pci_enable_bridge(bridge);
}
if pci_is_enabled(dev) != 0 {
if ((unsafe { (*dev).__slate_bits_3.__get_is_busmaster() }) as i32) == 0 {
pci_set_master(dev);
}
return;
}
retval = pci_enable_device(dev);
if retval != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Error enabling bridge (%d), continuing\n".as_ptr() as *const u8, retval) };
}
pci_set_master(dev);
}

fn pci_enable_device_flags(dev: *mut pci_dev, flags: u64) -> i32 {
let mut err: i32 = 0;
// only skip sriov related
let mut i: i32 = 0;
let mut bars: i32 = 0;
// Power state could be unknown at this point, either due to a fresh
// boot or a device removal call.  So get the current power state
// so that things like MSI message writing will behave as expected
// (e.g. if the device really is in D0 at enable time).
pci_update_current_state(dev, unsafe { (*dev).current_state });
if atomic_inc_return(unsafe { core::ptr::addr_of_mut!((*dev).enable_cnt) }) > 1 {
return 0;
}
// already enabled
let bridge: *mut pci_dev = pci_upstream_bridge(dev);
if !bridge.is_null() {
pci_enable_bridge(bridge);
}
while i <= 6 {
if (unsafe { (*((*dev).resource.as_mut_ptr() as *mut resource).wrapping_offset(i as isize)).flags }) & flags != 0 {
bars |= 1i32 << i;
}
i = i.wrapping_add(1);
}
i = 7;
while i < 11 {
if (unsafe { (*((*dev).resource.as_mut_ptr() as *mut resource).wrapping_offset(i as isize)).flags }) & flags != 0 {
bars |= 1i32 << i;
}
i = i.wrapping_add(1);
}
err = do_pci_enable_device(dev, bars);
if err < 0 {
atomic_dec(unsafe { core::ptr::addr_of_mut!((*dev).enable_cnt) });
}
err
}

/// pci_enable_device_mem - Initialize a device for use with Memory space
/// @dev: PCI device to be initialized
///
/// Initialize device before it's used by a driver. Ask low-level code
/// to enable Memory resources. Wake up the device if it was suspended.
/// Beware, this function can fail.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_enable_device_mem")]
extern "C" fn pci_enable_device_mem(dev: *mut pci_dev) -> i32 {
pci_enable_device_flags(dev, 512u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_enable_device_mem_744: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_enable_device_mem)) };

/// pci_enable_device - Initialize device before it's used by a driver.
/// @dev: PCI device to be initialized
///
/// Initialize device before it's used by a driver. Ask low-level code
/// to enable I/O and memory. Wake up the device if it was suspended.
/// Beware, this function can fail.
///
/// Note we don't actually enable the device many times if we call
/// this function repeatedly (we just increment the count).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_enable_device")]
extern "C" fn pci_enable_device(dev: *mut pci_dev) -> i32 {
pci_enable_device_flags(dev, 768u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_enable_device_745: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_enable_device)) };

/// pcibios_device_add - provide arch specific hooks when adding device dev
/// @dev: the PCI device being added
///
/// Permits the platform to provide architecture specific functionality when
/// devices are added. This is the default implementation. Architecture
/// implementations can override this.
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pcibios_device_add(dev: *mut pci_dev) -> i32 {
0
}

/// pcibios_release_device - provide arch specific hooks when releasing
/// 			    device dev
/// @dev: the PCI device being released
///
/// Permits the platform to provide architecture specific functionality when
/// devices are released. This is the default implementation. Architecture
/// implementations can override this.
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pcibios_release_device(dev: *mut pci_dev) {
}

/// pcibios_disable_device - disable arch specific PCI resources for device dev
/// @dev: the PCI device to disable
///
/// Disables architecture specific PCI resources for the device. This
/// is the default implementation. Architecture implementations can
/// override this.
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pcibios_disable_device(dev: *mut pci_dev) {
}

fn do_pci_disable_device(dev: *mut pci_dev) {
let mut pci_command: u16 = 0;
unsafe { pci_read_config_word(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(pci_command)) };
if (pci_command as i32) & 4 != 0 {
let __v1: u16 = ((pci_command as i32) & !4) as u16;
pci_command = __v1;
unsafe { pci_write_config_word(dev as *const pci_dev, 4i32, pci_command) };
}
pcibios_disable_device(dev);
}

/// pci_disable_enabled_device - Disable device without updating enable_cnt
/// @dev: PCI device to disable
///
/// NOTE: This function is a backend of PCI power management routines and is
/// not supposed to be called drivers.
#[unsafe(no_mangle)]
extern "C" fn pci_disable_enabled_device(dev: *mut pci_dev) {
if pci_is_enabled(dev) != 0 {
do_pci_disable_device(dev);
}
}

/// pci_disable_device - Disable PCI device after use
/// @dev: PCI device to be disabled
///
/// Signal to the system that the PCI device is not in use by the system
/// anymore.  This only involves disabling PCI bus-mastering, if active.
///
/// Note we don't actually disable the device until all callers of
/// pci_enable_device() have called pci_disable_device().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_disable_device")]
extern "C" fn pci_disable_device(dev: *mut pci_dev) {
let mut __v0: i32;
let __ret_warn_on: i32 = (atomic_read((unsafe { core::ptr::addr_of_mut!((*dev).enable_cnt) }) as *const __SlateRecord_types_h_190) <= 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: disabling already-disabled device".as_ptr() as *const u8, dev_driver_string(core::ptr::addr_of_mut!((*dev).dev) as *const device), dev_name(core::ptr::addr_of_mut!((*dev).dev) as *const device)) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_pci_disable_device_3, sym __slate_asm_string_pci_disable_device_4, const 2232 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, dev_driver_string(core::ptr::addr_of_mut!((*dev).dev) as *const device), dev_name(core::ptr::addr_of_mut!((*dev).dev) as *const device)) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = __ret_warn_on;
if atomic_dec_return(unsafe { core::ptr::addr_of_mut!((*dev).enable_cnt) }) != 0 {
return;
}
pci_host_bridge_disable_device(dev);
do_pci_disable_device(dev);
unsafe {
(*dev).__slate_bits_3.__set_is_busmaster(0u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_746: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_disable_device_747: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_disable_device)) };

/// pcibios_set_pcie_reset_state - set reset state for device dev
/// @dev: the PCIe device reset
/// @state: Reset state to enter into
///
/// Set the PCIe reset state for the device. This is the default
/// implementation. Architecture implementations can override this.
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pcibios_set_pcie_reset_state(dev: *mut pci_dev, state: u32) -> i32 {
22i32.wrapping_neg()
}

/// pci_set_pcie_reset_state - set reset state for device dev
/// @dev: the PCIe device reset
/// @state: Reset state to enter into
///
/// Sets the PCI reset state for the device.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_set_pcie_reset_state")]
extern "C" fn pci_set_pcie_reset_state(dev: *mut pci_dev, state: u32) -> i32 {
pcibios_set_pcie_reset_state(dev, state)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_set_pcie_reset_state_748: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, u32) -> i32>, *mut ()>(Some(pci_set_pcie_reset_state)) };

/// pcie_clear_root_pme_status - Clear root port PME interrupt status.
/// @dev: PCIe root port or event collector.
#[unsafe(no_mangle)]
extern "C" fn pcie_clear_root_pme_status(dev: *mut pci_dev) {
unsafe { pcie_capability_write_dword(dev, 32i32, 65536u32) };
}

/// pci_check_pme_status - Check if given device has generated PME.
/// @dev: Device to check.
///
/// Check the PME status of the device and if set, clear it and clear PME enable
/// (if set).  Return 'true' if PME status and PME enable were both set or
/// 'false' otherwise.
#[unsafe(no_mangle)]
extern "C" fn pci_check_pme_status(dev: *mut pci_dev) -> bool {
let mut pmcsr_pos: i32 = 0;
let mut pmcsr: u16 = 0;
let mut ret: bool = false;
if (unsafe { (*dev).pm_cap }) == 0 {
return false;
}
pmcsr_pos = ((unsafe { (*dev).pm_cap }) as i32).wrapping_add(4);
unsafe { pci_read_config_word(dev as *const pci_dev, pmcsr_pos, core::ptr::addr_of_mut!(pmcsr)) };
if (pmcsr as i32) & 32768 == 0 {
return false;
}
// Clear PME status.
pmcsr = ((pmcsr as i32) | 32768) as u16;
if (pmcsr as i32) & 256 != 0 {
// Disable PME to avoid interrupt flood.
let __v3: u16 = ((pmcsr as i32) & !256) as u16;
pmcsr = __v3;
ret = true;
}
unsafe { pci_write_config_word(dev as *const pci_dev, pmcsr_pos, pmcsr) };
ret
}

/// pci_pme_wakeup - Wake up a PCI device if its PME Status bit is set.
/// @dev: Device to handle.
/// @pme_poll_reset: Whether or not to reset the device's pme_poll flag.
///
/// Check if @dev has generated PME and queue a resume request for it in that
/// case.
#[unsafe(link_section = ".text.slate_distinct.pci.pci_pme_wakeup")]
extern "C" fn pci_pme_wakeup(dev: *mut pci_dev, pme_poll_reset: *mut ()) -> i32 {
if !pme_poll_reset.is_null() && ((unsafe { (*dev).__slate_bits_1.__get_pme_poll() }) as i32) != 0 {
unsafe {
(*dev).__slate_bits_1.__set_pme_poll(0u32);
}
}
if pci_check_pme_status(dev) {
pci_wakeup_event(dev);
pm_request_resume(unsafe { core::ptr::addr_of_mut!((*dev).dev) });
}
0
}

/// pci_pme_wakeup_bus - Walk given bus and wake up devices on it, if necessary.
/// @bus: Top bus of the subtree to walk.
#[unsafe(no_mangle)]
extern "C" fn pci_pme_wakeup_bus(bus: *mut pci_bus) {
if !bus.is_null() {
unsafe { pci_walk_bus(bus, Some(pci_pme_wakeup), 1i32 as *mut ()) };
}
}

/// pci_pme_capable - check the capability of PCI device to generate PME#
/// @dev: PCI device to handle.
/// @state: PCI state from which device will issue PME#.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_pme_capable")]
extern "C" fn pci_pme_capable(dev: *mut pci_dev, state: i32) -> bool {
if (unsafe { (*dev).pm_cap }) == 0 {
return false;
}
((unsafe { (*dev).__slate_bits_1.__get_pme_support() }) as i32) & 1i32 << state != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_pme_capable_749: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32) -> bool>, *mut ()>(Some(pci_pme_capable)) };

#[unsafe(link_section = ".text.slate_distinct.pci.pci_pme_list_scan")]
extern "C" fn pci_pme_list_scan(work: *mut work_struct) {
let mut bref: i32 = unsafe { core::mem::zeroed() };
let mut bdev: *mut device = unsafe { core::mem::zeroed() };
let mut dev: *mut device = unsafe { core::mem::zeroed() };
let mut bridge: *mut pci_dev = unsafe { core::mem::zeroed() };
let mut pdev: *mut pci_dev = unsafe { core::mem::zeroed() };
let mut n: core::mem::MaybeUninit<*mut pci_pme_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut pci_pme_device = core::ptr::addr_of_mut!(n) as *mut *mut pci_pme_device;
let mut pme_dev: core::mem::MaybeUninit<*mut pci_pme_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut pci_pme_device = core::ptr::addr_of_mut!(pme_dev) as *mut *mut pci_pme_device;
unsafe {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
mutex_lock(core::ptr::addr_of_mut!(pci_pme_list_mutex));
*__slate_slot_8 = list_first_entry__10!(core::ptr::addr_of_mut!(pci_pme_list));
let __v6: *mut pci_pme_device = list_next_entry__9!(*__slate_slot_8, list);
*__slate_slot_7 = __v6;
while list_entry_is_head!(*__slate_slot_8, core::ptr::addr_of_mut!(pci_pme_list), list) == 0 {
'__join_4: {
pdev = (*(*__slate_slot_8)).dev;
if ((*pdev).__slate_bits_1.__get_pme_poll() as i32) != 0 {
'__join_6: {
bridge = (*(*pdev).bus).self_;
dev = core::ptr::addr_of_mut!((*pdev).dev);
bdev = if !bridge.is_null() { core::ptr::addr_of_mut!((*bridge).dev) } else { core::ptr::null_mut::<device>() };
bref = 0;
// If we have a bridge, it should be in an active/D0
// state or the configuration space of subordinate
// devices may not be accessible or stable over the
// course of the call.
if !bdev.is_null() {
bref = pm_runtime_get_if_active(bdev);
if bref == 0 {
break '__join_4;
} else {
if (*bridge).current_state != 0 {
break '__join_6;
}
}
}
// The device itself should be suspended but config
// space must be accessible, therefore it cannot be in
// D3cold.
if pm_runtime_suspended(dev) && (*pdev).current_state != 4 {
pci_pme_wakeup(pdev, core::ptr::null_mut::<()>());
}
}
if bref > 0 {
pm_runtime_put(bdev);
}
} else {
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_8)).list));
kfree(*__slate_slot_8 as *const ());
}
}
*__slate_slot_8 = *__slate_slot_7;
let __v0: *mut pci_pme_device = list_next_entry__9!(*__slate_slot_7, list);
*__slate_slot_7 = __v0;
}
if list_empty(core::ptr::addr_of_mut!(pci_pme_list) as *const list_head) == 0 {
queue_delayed_work(system_freezable_wq, core::ptr::addr_of_mut!(pci_pme_work), msecs_to_jiffies(1000u32));
}
mutex_unlock(core::ptr::addr_of_mut!(pci_pme_list_mutex));
}
}

fn __pci_pme_active(dev: *mut pci_dev, enable: bool) {
let mut pmcsr: u16 = 0;
if ((unsafe { (*dev).__slate_bits_1.__get_pme_support() }) as i32) == 0 {
return;
}
unsafe { pci_read_config_word(dev as *const pci_dev, ((*dev).pm_cap as i32).wrapping_add(4), core::ptr::addr_of_mut!(pmcsr)) };
// Clear PME_Status by writing 1 to it and enable PME#
pmcsr = ((pmcsr as i32) | (32768 | 256)) as u16;
if !enable {
let __v3: u16 = ((pmcsr as i32) & !256) as u16;
pmcsr = __v3;
}
unsafe { pci_write_config_word(dev as *const pci_dev, ((*dev).pm_cap as i32).wrapping_add(4), pmcsr) };
}

/// pci_pme_restore - Restore PME configuration after config space restore.
/// @dev: PCI device to update.
#[unsafe(no_mangle)]
extern "C" fn pci_pme_restore(dev: *mut pci_dev) {
let mut pmcsr: u16 = 0;
if ((unsafe { (*dev).__slate_bits_1.__get_pme_support() }) as i32) == 0 {
return;
}
unsafe { pci_read_config_word(dev as *const pci_dev, ((*dev).pm_cap as i32).wrapping_add(4), core::ptr::addr_of_mut!(pmcsr)) };
if ((unsafe { (*dev).__slate_bits_1.__get_wakeup_prepared() }) as i32) != 0 {
pmcsr = ((pmcsr as i32) | 256) as u16;
let __v3: u16 = ((pmcsr as i32) & !32768) as u16;
pmcsr = __v3;
} else {
let __v5: u16 = ((pmcsr as i32) & !256) as u16;
pmcsr = __v5;
pmcsr = ((pmcsr as i32) | 32768) as u16;
}
unsafe { pci_write_config_word(dev as *const pci_dev, ((*dev).pm_cap as i32).wrapping_add(4), pmcsr) };
}

/// pci_pme_active - enable or disable PCI device's PME# function
/// @dev: PCI device to handle.
/// @enable: 'true' to enable PME# generation; 'false' to disable it.
///
/// The caller must verify that the device is capable of generating PME# before
/// calling this function with @enable equal to 'true'.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_pme_active")]
extern "C" fn pci_pme_active(dev: *mut pci_dev, enable: bool) {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
__pci_pme_active(dev, enable);
// PCI (as opposed to PCIe) PME requires that the device have
// its PME# line hooked up correctly. Not all hardware vendors
// do this, so the PME never gets delivered and the device
// remains asleep. The easiest way around this is to
// periodically walk the list of suspended devices and check
// whether any have their PME flag set. The assumption is that
// we'll wake up often enough anyway that this won't be a huge
// hit, and the power savings from the devices will still be a
// win.
//
// Although PCIe uses in-band PME message instead of PME# line
// to report PME, PME does not work for some PCIe devices in
// reality.  For example, there are devices that set their PME
// status bits, but don't really bother to send a PME message;
// there are PCI Express Root Ports that don't bother to
// trigger interrupts when they receive PME messages from the
// devices below.  So PME poll is used for PCIe devices too.
if ((unsafe { (*dev).__slate_bits_1.__get_pme_poll() }) as i32) != 0 {
let mut pme_dev: *mut pci_pme_device = unsafe { core::mem::zeroed() };
if enable {
let mut __v0: *mut pci_pme_device;
let __obj_size: u64 = size_mul(core::mem::size_of::<pci_pme_device>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut pci_pme_device;
pme_dev = __v0;
if pme_dev.is_null() {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"can't enable PME#\n".as_ptr() as *const u8) };
return;
}
unsafe {
(*pme_dev).dev = dev;
}
unsafe { mutex_lock(core::ptr::addr_of_mut!(pci_pme_list_mutex)) };
list_add(unsafe { core::ptr::addr_of_mut!((*pme_dev).list) }, unsafe { core::ptr::addr_of_mut!(pci_pme_list) });
if list_is_singular((unsafe { core::ptr::addr_of_mut!(pci_pme_list) }) as *const list_head) != 0 {
queue_delayed_work(unsafe { system_freezable_wq }, unsafe { core::ptr::addr_of_mut!(pci_pme_work) }, msecs_to_jiffies(1000u32));
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(pci_pme_list_mutex)) };
} else {
unsafe { mutex_lock(core::ptr::addr_of_mut!(pci_pme_list_mutex)) };
pme_dev = list_first_entry__10!(unsafe { core::ptr::addr_of_mut!(pci_pme_list) });
while list_entry_is_head!(pme_dev, unsafe { core::ptr::addr_of_mut!(pci_pme_list) }, list) == 0 {
if (unsafe { (*pme_dev).dev }) == dev {
list_del(unsafe { core::ptr::addr_of_mut!((*pme_dev).list) });
unsafe { kfree(pme_dev as *const ()) };
break;
}
pme_dev = list_next_entry__9!(pme_dev, list);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(pci_pme_list_mutex)) };
}
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"PME# %s\n".as_ptr() as *const u8, if enable { c"enabled".as_ptr() as *mut u8 } else { c"disabled".as_ptr() as *mut u8 }) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_pme_active_750: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, bool)>, *mut ()>(Some(pci_pme_active)) };

/// __pci_enable_wake - enable PCI device as wakeup event source
/// @dev: PCI device affected
/// @state: PCI state from which device will issue wakeup events
/// @enable: True to enable event generation; false to disable
///
/// This enables the device as a wakeup event source, or disables it.
/// When such events involves platform-specific hooks, those hooks are
/// called automatically by this routine.
///
/// Devices with legacy power management (no standard PCI PM capabilities)
/// always require such platform hooks.
///
/// RETURN VALUE:
/// 0 is returned on success
/// -EINVAL is returned if device is not supposed to wake up the system
/// Error code depending on the platform is returned if both the platform and
/// the native mechanism fail to enable the generation of wake-up events
fn __pci_enable_wake(dev: *mut pci_dev, state: i32, enable: bool) -> i32 {
let mut ret: i32 = 0;
// Bridges that are not power-manageable directly only signal
// wakeup on behalf of subordinate devices which is set up
// elsewhere, so skip them. However, bridges that are
// power-manageable may signal wakeup for themselves (for example,
// on a hotplug event) and they need to be covered here.
if !pci_power_manageable(dev) {
return 0;
}
// Don't do the same thing twice in a row for one device.
if (enable as i32) == ((((unsafe { (*dev).__slate_bits_1.__get_wakeup_prepared() }) as i32) != 0) as i32) {
return 0;
}
// According to "PCI System Architecture" 4th ed. by Tom Shanley & Don
// Anderson we should be doing PME# wake enable followed by ACPI wake
// enable.  To disable wake-up we call the platform first, for symmetry.
if enable {
let mut error: i32 = 0;
// Enable PME signaling if the device can signal PME from
// D3cold regardless of whether or not it can signal PME from
// the current target state, because that will allow it to
// signal PME when the hierarchy above it goes into D3cold and
// the device itself ends up in D3cold as a result of that.
let __v0: bool = pci_pme_capable(dev, state) || pci_pme_capable(dev, 4i32);
if __v0 {
pci_pme_active(dev, true);
} else {
ret = 1;
}
error = platform_pci_set_wakeup(dev, true);
if ret != 0 {
ret = error;
}
if ret == 0 {
unsafe {
(*dev).__slate_bits_1.__set_wakeup_prepared(1u32);
}
}
} else {
platform_pci_set_wakeup(dev, false);
pci_pme_active(dev, false);
unsafe {
(*dev).__slate_bits_1.__set_wakeup_prepared(0u32);
}
}
ret
}

/// pci_enable_wake - change wakeup settings for a PCI device
/// @pci_dev: Target device
/// @state: PCI state from which device will issue wakeup events
/// @enable: Whether or not to enable event generation
///
/// If @enable is set, check device_may_wakeup() for the device before calling
/// __pci_enable_wake() for it.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_enable_wake")]
extern "C" fn pci_enable_wake(pci_dev: *mut pci_dev, state: i32, enable: bool) -> i32 {
let __v0: bool = enable && !device_may_wakeup(unsafe { core::ptr::addr_of_mut!((*pci_dev).dev) });
if __v0 {
return 22i32.wrapping_neg();
}
__pci_enable_wake(pci_dev, state, enable)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_enable_wake_751: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32, bool) -> i32>, *mut ()>(Some(pci_enable_wake)) };

/// pci_wake_from_d3 - enable/disable device to wake up from D3_hot or D3_cold
/// @dev: PCI device to prepare
/// @enable: True to enable wake-up event generation; false to disable
///
/// Many drivers want the device to wake up the system from D3_hot or D3_cold
/// and this function allows them to set that up cleanly - pci_enable_wake()
/// should not be called twice in a row to enable wake-up due to PCI PM vs ACPI
/// ordering constraints.
///
/// This function only returns error code if the device is not allowed to wake
/// up the system from sleep or it is not capable of generating PME# from both
/// D3_hot and D3_cold and the platform is unable to enable wake-up power for it.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_wake_from_d3")]
extern "C" fn pci_wake_from_d3(dev: *mut pci_dev, enable: bool) -> i32 {
let __v0: i32 = if pci_pme_capable(dev, 4i32) {
pci_enable_wake(dev, 4i32, enable)
} else {
pci_enable_wake(dev, 3i32, enable)
};
__v0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_wake_from_d3_752: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, bool) -> i32>, *mut ()>(Some(pci_wake_from_d3)) };

/// pci_target_state - find an appropriate low power state for a given PCI dev
/// @dev: PCI device
/// @wakeup: Whether or not wakeup functionality will be enabled for the device.
///
/// Use underlying platform code to find a supported low power state for @dev.
/// If the platform can't manage @dev, return the deepest state from which it
/// can generate wake events, based on any available PME info.
fn pci_target_state(dev: *mut pci_dev, wakeup: bool) -> i32 {
if platform_pci_power_manageable(dev) {
// Call the platform to find the target state for the device.
let state: i32 = platform_pci_choose_state(dev);
match state {
-1 | 5 => {
return 3;
}
1 | 2 => {
if pci_no_d1d2(dev) != 0 {
return 3;
}
}
_ => {
}
}
return state;
}
// If the device is in D3cold even though it's not power-manageable by
// the platform, it may have been powered down by non-standard means.
// Best to let it slumber.
if (unsafe { (*dev).current_state }) == 4 {
return 4;
} else {
if (unsafe { (*dev).pm_cap }) == 0 {
return 0;
}
}
if wakeup && ((unsafe { (*dev).__slate_bits_1.__get_pme_support() }) as i32) != 0 {
let mut state: i32 = 3;
// Find the deepest state from which the device can generate
// PME#.
while state != 0 && ((unsafe { (*dev).__slate_bits_1.__get_pme_support() }) as i32) & 1i32 << state == 0 {
state = state.wrapping_sub(1);
}
if state != 0 {
return state;
} else {
if ((unsafe { (*dev).__slate_bits_1.__get_pme_support() }) as i32) & 1 != 0 {
return 0;
}
}
}
3
}

/// pci_prepare_to_sleep - prepare PCI device for system-wide transition
/// 			  into a sleep state
/// @dev: Device to handle.
///
/// Choose the power state appropriate for the device depending on whether
/// it can wake up the system and/or is power manageable by the platform
/// (PCI_D3hot is the default) and put the device into that state.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_prepare_to_sleep")]
extern "C" fn pci_prepare_to_sleep(dev: *mut pci_dev) -> i32 {
let wakeup: bool = device_may_wakeup(unsafe { core::ptr::addr_of_mut!((*dev).dev) });
let target_state: i32 = pci_target_state(dev, wakeup);
let mut error: i32 = 0;
if target_state == 1i32.wrapping_neg() {
return 5i32.wrapping_neg();
}
pci_enable_wake(dev, target_state, wakeup);
error = pci_set_power_state(dev, target_state);
if error != 0 {
pci_enable_wake(dev, target_state, false);
}
error
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_prepare_to_sleep_753: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_prepare_to_sleep)) };

/// pci_back_from_sleep - turn PCI device on during system-wide transition
/// 			 into working state
/// @dev: Device to handle.
///
/// Disable device's system wake-up capability and put it into D0.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_back_from_sleep")]
extern "C" fn pci_back_from_sleep(dev: *mut pci_dev) -> i32 {
let ret: i32 = pci_set_power_state(dev, 0i32);
if ret != 0 {
return ret;
}
pci_enable_wake(dev, 0i32, false);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_back_from_sleep_754: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_back_from_sleep)) };

/// pci_finish_runtime_suspend - Carry out PCI-specific part of runtime suspend.
/// @dev: PCI device being suspended.
///
/// Prepare @dev to generate wake-up events at run time and put it into a low
/// power state.
#[unsafe(no_mangle)]
extern "C" fn pci_finish_runtime_suspend(dev: *mut pci_dev) -> i32 {
let target_state: i32 = pci_target_state(dev, device_can_wakeup(unsafe { core::ptr::addr_of_mut!((*dev).dev) }));
let mut error: i32 = 0;
if target_state == 1i32.wrapping_neg() {
return 5i32.wrapping_neg();
}
__pci_enable_wake(dev, target_state, pci_dev_run_wake(dev));
error = pci_set_power_state(dev, target_state);
if error != 0 {
pci_enable_wake(dev, target_state, false);
}
error
}

/// pci_dev_run_wake - Check if device can generate run-time wake-up events.
/// @dev: Device to check.
///
/// Return true if the device itself is capable of generating wake-up events
/// (through the platform or using the native PCIe PME) or if the device supports
/// PME and one of its upstream bridges can generate wake-up events.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_dev_run_wake")]
extern "C" fn pci_dev_run_wake(dev: *mut pci_dev) -> bool {
let mut bus: *mut pci_bus = unsafe { (*dev).bus };
if ((unsafe { (*dev).__slate_bits_1.__get_pme_support() }) as i32) == 0 {
return false;
}
// PME-capable in principle, but not from the target power state
if !pci_pme_capable(dev, pci_target_state(dev, true)) {
return false;
}
if device_can_wakeup(unsafe { core::ptr::addr_of_mut!((*dev).dev) }) {
return true;
}
while !unsafe { (*bus).parent }.is_null() {
let bridge: *mut pci_dev = unsafe { (*bus).self_ };
if device_can_wakeup(unsafe { core::ptr::addr_of_mut!((*bridge).dev) }) {
return true;
}
bus = unsafe { (*bus).parent };
}
// We have reached the root bus.
if !unsafe { (*bus).bridge }.is_null() {
return device_can_wakeup(unsafe { (*bus).bridge });
}
false
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_dev_run_wake_755: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> bool>, *mut ()>(Some(pci_dev_run_wake)) };

/// pci_dev_need_resume - Check if it is necessary to resume the device.
/// @pci_dev: Device to check.
///
/// Return 'true' if the device is not runtime-suspended or it has to be
/// reconfigured due to wakeup settings difference between system and runtime
/// suspend, or the current power state of it is not suitable for the upcoming
/// (system-wide) transition.
#[unsafe(no_mangle)]
extern "C" fn pci_dev_need_resume(pci_dev: *mut pci_dev) -> bool {
let dev: *mut device = unsafe { core::ptr::addr_of_mut!((*pci_dev).dev) };
let mut target_state: i32 = 0;
let __v0: bool = !pm_runtime_suspended(dev) || platform_pci_need_resume(pci_dev);
if __v0 {
return true;
}
target_state = pci_target_state(pci_dev, device_may_wakeup(dev));
// If the earlier platform check has not triggered, D3cold is just power
// removal on top of D3hot, so no need to resume the device in that
// case.
target_state != unsafe { (*pci_dev).current_state } && target_state != 4 && (unsafe { (*pci_dev).current_state }) != 3
}

/// pci_dev_adjust_pme - Adjust PME setting for a suspended device.
/// @pci_dev: Device to check.
///
/// If the device is suspended and it is not configured for system wakeup,
/// disable PME for it to prevent it from waking up the system unnecessarily.
///
/// Note that if the device's power state is D3cold and the platform check in
/// pci_dev_need_resume() has not triggered, the device's configuration need not
/// be changed.
#[unsafe(no_mangle)]
extern "C" fn pci_dev_adjust_pme(pci_dev: *mut pci_dev) {
let dev: *mut device = unsafe { core::ptr::addr_of_mut!((*pci_dev).dev) };
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*dev).power.lock) });
let __v0: bool = pm_runtime_suspended(dev) && !device_may_wakeup(dev);
if __v0 && (unsafe { (*pci_dev).current_state }) < 4 {
__pci_pme_active(pci_dev, false);
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*dev).power.lock) });
}

/// pci_dev_complete_resume - Finalize resume from system sleep for a device.
/// @pci_dev: Device to handle.
///
/// If the device is runtime suspended and wakeup-capable, enable PME for it as
/// it might have been disabled during the prepare phase of system suspend if
/// the device was not configured for system wakeup.
#[unsafe(no_mangle)]
extern "C" fn pci_dev_complete_resume(pci_dev: *mut pci_dev) {
let dev: *mut device = unsafe { core::ptr::addr_of_mut!((*pci_dev).dev) };
if !pci_dev_run_wake(pci_dev) {
return;
}
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*dev).power.lock) });
if pm_runtime_suspended(dev) && (unsafe { (*pci_dev).current_state }) < 4 {
__pci_pme_active(pci_dev, true);
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*dev).power.lock) });
}

/// pci_choose_state - Choose the power state of a PCI device.
/// @dev: Target PCI device.
/// @state: Target state for the whole system.
///
/// Returns PCI power state suitable for @dev and @state.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_choose_state")]
extern "C" fn pci_choose_state(dev: *mut pci_dev, state: pm_message) -> i32 {
if state.event == 0 {
return 0;
}
pci_target_state(dev, false)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_choose_state_756: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, pm_message) -> i32>, *mut ()>(Some(pci_choose_state)) };

#[unsafe(no_mangle)]
extern "C" fn pci_config_pm_runtime_get(pdev: *mut pci_dev) {
let dev: *mut device = unsafe { core::ptr::addr_of_mut!((*pdev).dev) };
let parent: *mut device = unsafe { (*dev).parent };
if !parent.is_null() {
pm_runtime_get_sync(parent);
}
pm_runtime_get_noresume(dev);
// pdev->current_state is set to PCI_D3cold during suspending,
// so wait until suspending completes
unsafe { pm_runtime_barrier(dev) };
// Only need to resume devices in D3cold, because config
// registers are still accessible for devices suspended but
// not in D3cold.
if (unsafe { (*pdev).current_state }) == 4 {
pm_runtime_resume(dev);
}
}

#[unsafe(no_mangle)]
extern "C" fn pci_config_pm_runtime_put(pdev: *mut pci_dev) {
let dev: *mut device = unsafe { core::ptr::addr_of_mut!((*pdev).dev) };
let parent: *mut device = unsafe { (*dev).parent };
pm_runtime_put(dev);
if !parent.is_null() {
pm_runtime_put_sync(parent);
}
}

/// pci_suspend_retains_context - Check if the platform can retain the device
/// 				 context during system suspend
/// @pdev: PCI device to check
///
/// Return: true if the platform can guarantee to retain the device context,
/// false otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_suspend_retains_context")]
extern "C" fn pci_suspend_retains_context(pdev: *mut pci_dev) -> bool {
let bridge: *mut pci_host_bridge = unsafe { pci_find_host_bridge((*pdev).bus) };
// If the platform firmware (like ACPI) is involved at the end of
// system suspend, device context may not be retained.
if pm_suspend_via_firmware() {
return false;
}
// Some host bridges power off the PHY to enter deep low-power
// modes during system suspend. Exiting L1SS from this condition
// may violate timing requirements and result in Link Down (LDn),
// which causes a reset of the device.  On such platforms, the
// endpoint must be prepared for context loss.
if !bridge.is_null() && ((unsafe { (*bridge).__slate_bits_0.__get_broken_l1ss_resume() }) as i32) != 0 {
return false;
}
// Assume that the context is retained by default
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_suspend_retains_context_757: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> bool>, *mut ()>(Some(pci_suspend_retains_context)) };

/// Gigabyte X299 root port is not marked as hotplug capable
/// which allows Linux to power manage it.  However, this
/// confuses the BIOS SMI handler so don't power manage root
/// ports on that system.
/// Downstream device is not accessible after putting a root port
/// into D3cold and back into D0 on Elo Continental Z2 board
/// Changing power state of root port dGPU is connected fails
/// https://gitlab.freedesktop.org/drm/amd/-/issues/3229
static bridge_d3_blacklist: __SlateAlign16<[dmi_system_id; 4]> = __SlateAlign16([{
let mut __t0: dmi_system_id = unsafe { core::mem::zeroed() };
__t0.ident = (b"X299 DESIGNARE EX-CF\0".as_ptr() as *mut u8) as *const u8;
__t0.matches = {
let mut __t1: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t1[0] = {
let mut __t2: dmi_strmatch = unsafe { core::mem::zeroed() };
__t2.__slate_bits_0.__set_slot(13u8);
__t2.substr = [71u8, 105u8, 103u8, 97u8, 98u8, 121u8, 116u8, 101u8, 32u8, 84u8, 101u8, 99u8, 104u8, 110u8, 111u8, 108u8, 111u8, 103u8, 121u8, 32u8, 67u8, 111u8, 46u8, 44u8, 32u8, 76u8, 116u8, 100u8, 46u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2
};
__t1[1] = {
let mut __t3: dmi_strmatch = unsafe { core::mem::zeroed() };
__t3.__slate_bits_0.__set_slot(14u8);
__t3.substr = [88u8, 50u8, 57u8, 57u8, 32u8, 68u8, 69u8, 83u8, 73u8, 71u8, 78u8, 65u8, 82u8, 69u8, 32u8, 69u8, 88u8, 45u8, 67u8, 70u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3
};
__t1
};
__t0
}, {
let mut __t4: dmi_system_id = unsafe { core::mem::zeroed() };
__t4.ident = (b"Elo Continental Z2\0".as_ptr() as *mut u8) as *const u8;
__t4.matches = {
let mut __t5: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t5[0] = {
let mut __t6: dmi_strmatch = unsafe { core::mem::zeroed() };
__t6.__slate_bits_0.__set_slot(13u8);
__t6.substr = [69u8, 108u8, 111u8, 32u8, 84u8, 111u8, 117u8, 99u8, 104u8, 32u8, 83u8, 111u8, 108u8, 117u8, 116u8, 105u8, 111u8, 110u8, 115u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t6
};
__t5[1] = {
let mut __t7: dmi_strmatch = unsafe { core::mem::zeroed() };
__t7.__slate_bits_0.__set_slot(14u8);
__t7.substr = [71u8, 101u8, 109u8, 105u8, 110u8, 105u8, 108u8, 97u8, 107u8, 101u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t7
};
__t5[2] = {
let mut __t8: dmi_strmatch = unsafe { core::mem::zeroed() };
__t8.__slate_bits_0.__set_slot(15u8);
__t8.substr = [67u8, 111u8, 110u8, 116u8, 105u8, 110u8, 101u8, 110u8, 116u8, 97u8, 108u8, 32u8, 90u8, 50u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t8
};
__t5
};
__t4
}, {
let mut __t9: dmi_system_id = unsafe { core::mem::zeroed() };
__t9.ident = (b"Hewlett-Packard HP Pavilion 17 Notebook PC/1972\0".as_ptr() as *mut u8) as *const u8;
__t9.matches = {
let mut __t10: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t10[0] = {
let mut __t11: dmi_strmatch = unsafe { core::mem::zeroed() };
__t11.__slate_bits_0.__set_slot(13u8);
__t11.substr = [72u8, 101u8, 119u8, 108u8, 101u8, 116u8, 116u8, 45u8, 80u8, 97u8, 99u8, 107u8, 97u8, 114u8, 100u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t11
};
__t10[1] = {
let mut __t12: dmi_strmatch = unsafe { core::mem::zeroed() };
__t12.__slate_bits_0.__set_slot(14u8);
__t12.substr = [49u8, 57u8, 55u8, 50u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t12
};
__t10[2] = {
let mut __t13: dmi_strmatch = unsafe { core::mem::zeroed() };
__t13.__slate_bits_0.__set_slot(15u8);
__t13.substr = [57u8, 53u8, 46u8, 51u8, 51u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t13
};
__t10
};
__t9
}, {
let mut __t14: dmi_system_id = unsafe { core::mem::zeroed() };
__t14
}]);

/// pci_bridge_d3_possible - Is it possible to put the bridge into D3
/// @bridge: Bridge to check
///
/// Currently we only allow D3 for some PCIe ports and for Thunderbolt.
///
/// Return: Whether it is possible to move the bridge to D3.
///
/// The return value is guaranteed to be constant across the entire lifetime
/// of the bridge, including its hot-removal.
#[unsafe(no_mangle)]
extern "C" fn pci_bridge_d3_possible(bridge: *mut pci_dev) -> bool {
if !pci_is_pcie(bridge) {
return false;
}
match pci_pcie_type(bridge as *const pci_dev) {
4..=6 => {
if unsafe { pci_bridge_d3_disable } {
return false;
}
// Hotplug ports handled by platform firmware may not be put
// into D3 by the OS, e.g. ACPI slots ...
if ((unsafe { (*bridge).__slate_bits_3.__get_is_hotplug_bridge() }) as i32) != 0 && ((unsafe { (*bridge).__slate_bits_3.__get_is_pciehp() }) as i32) == 0 {
return false;
}
// ... or PCIe hotplug ports not handled natively by the OS.
let __v0: bool = ((unsafe { (*bridge).__slate_bits_3.__get_is_pciehp() }) as i32) != 0 && !unsafe { pciehp_is_native(bridge) };
if __v0 {
return false;
}
if unsafe { pci_bridge_d3_force } {
return true;
}
// Even the oldest 2010 Thunderbolt controller supports D3.
if ((unsafe { (*bridge).__slate_bits_3.__get_is_thunderbolt() }) as i32) != 0 {
return true;
}
// Platform might know better if the bridge supports D3
if platform_pci_bridge_d3(bridge) {
return true;
}
// Hotplug ports handled natively by the OS on x86 platforms
// were not validated by vendors for runtime D3 at least until
// 2018 because there was no OS support.
if IS_ENABLED__2!() != 0 && ((unsafe { (*bridge).__slate_bits_3.__get_is_pciehp() }) as i32) != 0 {
return false;
}
if (unsafe { dmi_check_system(core::ptr::addr_of!(bridge_d3_blacklist.0) as *const dmi_system_id) }) != 0 {
return false;
}
// Out of caution, we only allow PCIe ports from 2015 or newer
// into D3 on x86.
let __v1: bool = IS_ENABLED__2!() == 0 || (unsafe { dmi_get_bios_year() }) >= 2015;
if __v1 {
return true;
}
}
_ => {
}
}
false
}

#[unsafe(link_section = ".text.slate_distinct.pci.pci_dev_check_d3cold")]
extern "C" fn pci_dev_check_d3cold(dev: *mut pci_dev, data: *mut ()) -> i32 {
let __v0: bool = ((unsafe { (*dev).__slate_bits_1.__get_no_d3cold() }) as i32) != 0 || ((unsafe { (*dev).__slate_bits_1.__get_d3cold_allowed() }) as i32) == 0 || device_may_wakeup(unsafe { core::ptr::addr_of_mut!((*dev).dev) }) && !pci_pme_capable(dev, 4i32);
let __v2: bool = __v0 || !pci_power_manageable(dev);
if __v2 {
unsafe {
*(data as *mut bool) = false;
}
}
// The device needs to be allowed to go D3cold ...
// ... and if it is wakeup capable to do so from D3cold.
// If it is a bridge it must be allowed to go to D3.
!unsafe { *(data as *mut bool) } as i32
}

/// pci_bridge_d3_update - Update bridge D3 capabilities
/// @dev: PCI device which is changed
///
/// Update upstream bridge PM capabilities accordingly depending on if the
/// device PM configuration was changed or the device is being removed.  The
/// change is also propagated upstream.
#[unsafe(no_mangle)]
extern "C" fn pci_bridge_d3_update(dev: *mut pci_dev) {
let remove: bool = device_is_registered(unsafe { core::ptr::addr_of_mut!((*dev).dev) }) == 0;
let mut d3cold_ok: bool = true;
let bridge: *mut pci_dev = pci_upstream_bridge(dev);
let __v0: bool = bridge.is_null() || !pci_bridge_d3_possible(bridge);
if __v0 {
return;
}
// If D3 is currently allowed for the bridge, removing one of its
// children won't change that.
if remove && ((unsafe { (*bridge).__slate_bits_1.__get_bridge_d3() }) as i32) != 0 {
return;
}
// If D3 is currently allowed for the bridge and a child is added or
// changed, disallowance of D3 can only be caused by that child, so
// we only need to check that single device, not any of its siblings.
//
// If D3 is currently not allowed for the bridge, checking the device
// first may allow us to skip checking its siblings.
if !remove {
pci_dev_check_d3cold(dev, core::ptr::addr_of_mut!(d3cold_ok) as *mut ());
}
// If D3 is currently not allowed for the bridge, this may be caused
// either by the device being changed/removed or any of its siblings,
// so we need to go through all children to find out if one of them
// continues to block D3.
if d3cold_ok && ((unsafe { (*bridge).__slate_bits_1.__get_bridge_d3() }) as i32) == 0 {
unsafe { pci_walk_bus((*bridge).subordinate, Some(pci_dev_check_d3cold), core::ptr::addr_of_mut!(d3cold_ok) as *mut ()) };
}
if ((unsafe { (*bridge).__slate_bits_1.__get_bridge_d3() }) as i32) != (d3cold_ok as i32) {
unsafe {
(*bridge).__slate_bits_1.__set_bridge_d3(d3cold_ok as u32);
}
// Propagate change to upstream bridges
pci_bridge_d3_update(bridge);
}
}

/// pci_d3cold_enable - Enable D3cold for device
/// @dev: PCI device to handle
///
/// This function can be used in drivers to enable D3cold from the device
/// they handle.  It also updates upstream PCI bridge PM capabilities
/// accordingly.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_d3cold_enable")]
extern "C" fn pci_d3cold_enable(dev: *mut pci_dev) {
if ((unsafe { (*dev).__slate_bits_1.__get_no_d3cold() }) as i32) != 0 {
unsafe {
(*dev).__slate_bits_1.__set_no_d3cold(0u32);
}
pci_bridge_d3_update(dev);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_d3cold_enable_758: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_d3cold_enable)) };

/// pci_d3cold_disable - Disable D3cold for device
/// @dev: PCI device to handle
///
/// This function can be used in drivers to disable D3cold from the device
/// they handle.  It also updates upstream PCI bridge PM capabilities
/// accordingly.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_d3cold_disable")]
extern "C" fn pci_d3cold_disable(dev: *mut pci_dev) {
if ((unsafe { (*dev).__slate_bits_1.__get_no_d3cold() }) as i32) == 0 {
unsafe {
(*dev).__slate_bits_1.__set_no_d3cold(1u32);
}
pci_bridge_d3_update(dev);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_d3cold_disable_759: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_d3cold_disable)) };

#[unsafe(no_mangle)]
extern "C" fn pci_pm_power_up_and_verify_state(pci_dev: *mut pci_dev) {
pci_power_up(pci_dev);
pci_update_current_state(pci_dev, 0i32);
}

// pci_pm_init - Initialize PM functions of given PCI device
// @dev: PCI device to handle.
fn pci_ea_flags(dev: *mut pci_dev, prop: u8) -> u64 {
let mut flags: u64 = 48u64;
match prop as i32 {
0 | 4 => {
flags |= 512;
}
1 | 3 => {
flags |= 8704u64;
}
2 => {
flags |= 256;
}
_ => {
return 0;
}
}
flags
}

fn pci_ea_get_resource(dev: *mut pci_dev, bei: u8, prop: u8) -> *mut resource {
if (bei as i32) <= 5 && (prop as i32) <= 2 {
return unsafe { (*dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset((bei as i32) as isize);
} else {
if (bei as i32) == 8 {
return unsafe { (*dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(6 as isize);
} else {
return core::ptr::null_mut::<resource>();
}
}
}

// Read an Enhanced Allocation (EA) entry
fn pci_ea_read(dev: *mut pci_dev, offset: i32) -> i32 {
let mut max_offset_upper: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(max_offset_upper) as *mut u32;
let mut base_upper: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u32 = core::ptr::addr_of_mut!(base_upper) as *mut u32;
let mut support_64: bool = unsafe { core::mem::zeroed() };
let mut prop: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut u8 = core::ptr::addr_of_mut!(prop) as *mut u8;
let mut max_offset: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut u32 = core::ptr::addr_of_mut!(max_offset) as *mut u32;
let mut base: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut u32 = core::ptr::addr_of_mut!(base) as *mut u32;
let mut bei: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut u32 = core::ptr::addr_of_mut!(bei) as *mut u32;
let mut dw0: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut u32 = core::ptr::addr_of_mut!(dw0) as *mut u32;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut end: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut u64 = core::ptr::addr_of_mut!(end) as *mut u64;
let mut start: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut u64 = core::ptr::addr_of_mut!(start) as *mut u64;
let mut ent_offset: i32 = unsafe { core::mem::zeroed() };
let mut ent_size: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut i32 = core::ptr::addr_of_mut!(ent_size) as *mut i32;
let mut res_name: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_39: *mut *const u8 = core::ptr::addr_of_mut!(res_name) as *mut *const u8;
let mut res: core::mem::MaybeUninit<*mut resource> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut *mut resource = core::ptr::addr_of_mut!(res) as *mut *mut resource;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_39 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
ent_offset = offset;
let __slate_auto_bytes_4 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_10 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_11 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 1usize);
support_64 = (core::mem::size_of::<u64>() as u64) >= 8;
pci_read_config_dword(dev as *const pci_dev, ent_offset, core::ptr::addr_of_mut!(*__slate_slot_33));
ent_offset = ent_offset.wrapping_add(4);
while false {
}
while false {
}
while false {
}
loop {
if (if true { (!(7i32 >> (7u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
__compiletime_assert_768();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 7u64.wrapping_add(1u64 << (7u64.trailing_zeros() as i32)) & 7u64.wrapping_add(1u64 << (7u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_769();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
'__join_0: {
let __v24: i32 = ((*__slate_slot_33 & 7) >> (7u64.trailing_zeros() as i32)) as i32;
// Entry size field indicates DWORDs after 1st
*__slate_slot_38 = __v24.wrapping_add(1) << 2i32;
if *__slate_slot_33 & 2147483648 != 0 {
// Entry not enabled
while false {
}
while false {
}
while false {
}
loop {
if (if true { (!(240i32 >> (240u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
__compiletime_assert_773();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 240u64.wrapping_add(1u64 << (240u64.trailing_zeros() as i32)) & 240u64.wrapping_add(1u64 << (240u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_774();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
*__slate_slot_32 = (((*__slate_slot_33 & 240) >> (240u64.trailing_zeros() as i32)) as i32) as u32;
while false {
}
while false {
}
while false {
}
loop {
if (if true { (!(65280i32 >> (65280u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
__compiletime_assert_778();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)) & 65280u64.wrapping_add(1u64 << (65280u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_779();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
*__slate_slot_29 = (((*__slate_slot_33 & 65280) >> (65280u64.trailing_zeros() as i32)) as i32) as u8;
// If the Property is in the reserved range, try the Secondary
// Property instead.
if (*__slate_slot_29 as i32) > 7 && (*__slate_slot_29 as i32) < 253 {
while false {
}
while false {
}
while false {
}
loop {
if (if true { (!(16711680i32 >> (16711680u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
__compiletime_assert_783();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if 16711680u64.wrapping_add(1u64 << (16711680u64.trailing_zeros() as i32)) & 16711680u64.wrapping_add(1u64 << (16711680u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
__compiletime_assert_784();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
*__slate_slot_29 = (((*__slate_slot_33 & 16711680) >> (16711680u64.trailing_zeros() as i32)) as i32) as u8;
}
if (*__slate_slot_29 as i32) <= 7 {
*__slate_slot_40 = pci_ea_get_resource(dev, *__slate_slot_32 as u8, *__slate_slot_29);
*__slate_slot_39 = pci_resource_name(dev, *__slate_slot_32);
if (*__slate_slot_40).is_null() {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Unsupported EA entry BEI: %u\n".as_ptr() as *const u8, *__slate_slot_32);
} else {
*__slate_slot_34 = pci_ea_flags(dev, *__slate_slot_29);
if *__slate_slot_34 == 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Unsupported EA properties: %#x\n".as_ptr() as *const u8, *__slate_slot_29 as i32);
} else {
// Read Base
pci_read_config_dword(dev as *const pci_dev, ent_offset, core::ptr::addr_of_mut!(*__slate_slot_31));
*__slate_slot_36 = (*__slate_slot_31 & 4294967292) as u64;
ent_offset = ent_offset.wrapping_add(4);
// Read MaxOffset
pci_read_config_dword(dev as *const pci_dev, ent_offset, core::ptr::addr_of_mut!(*__slate_slot_30));
ent_offset = ent_offset.wrapping_add(4);
// Read Base MSBs (if 64-bit entry)
if *__slate_slot_31 & 2 != 0 {
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
pci_read_config_dword(dev as *const pci_dev, ent_offset, core::ptr::addr_of_mut!(*__slate_slot_13));
ent_offset = ent_offset.wrapping_add(4);
*__slate_slot_34 |= 1048576;
// entry starts above 32-bit boundary, can't use
if !support_64 && *__slate_slot_13 != 0 {
break '__join_0;
} else {
if support_64 {
*__slate_slot_36 |= (*__slate_slot_13 as u64) << 32i32;
}
}
}
*__slate_slot_35 = (*__slate_slot_36).wrapping_add((*__slate_slot_30 | 3) as u64);
// Read MaxOffset MSBs (if 64-bit entry)
if *__slate_slot_30 & 2 != 0 {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
pci_read_config_dword(dev as *const pci_dev, ent_offset, core::ptr::addr_of_mut!(*__slate_slot_6));
ent_offset = ent_offset.wrapping_add(4);
*__slate_slot_34 |= 1048576;
// entry too big, can't use
if !support_64 && *__slate_slot_6 != 0 {
break '__join_0;
} else {
if support_64 {
let __v1: u64 = *__slate_slot_35;
*__slate_slot_35 = __v1.wrapping_add((*__slate_slot_6 as u64) << 32i32);
}
}
}
if *__slate_slot_35 < *__slate_slot_36 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"EA Entry crosses address boundary\n".as_ptr() as *const u8);
} else {
if *__slate_slot_38 != ent_offset.wrapping_sub(offset) {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"EA Entry Size (%d) does not match length read (%d)\n".as_ptr() as *const u8, *__slate_slot_38, ent_offset.wrapping_sub(offset));
} else {
(*(*__slate_slot_40)).name = pci_name(dev as *const pci_dev);
(*(*__slate_slot_40)).start = *__slate_slot_36;
(*(*__slate_slot_40)).end = *__slate_slot_35;
(*(*__slate_slot_40)).flags = *__slate_slot_34;
if *__slate_slot_32 <= 5 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s %pR: from Enhanced Allocation, properties %#02x\n".as_ptr() as *const u8, *__slate_slot_39, *__slate_slot_40, *__slate_slot_29 as i32);
} else {
if *__slate_slot_32 == 8 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s %pR: from Enhanced Allocation, properties %#02x\n".as_ptr() as *const u8, *__slate_slot_39, *__slate_slot_40, *__slate_slot_29 as i32);
} else {
if *__slate_slot_32 >= 9 && *__slate_slot_32 <= 14 {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s %pR: from Enhanced Allocation, properties %#02x\n".as_ptr() as *const u8, *__slate_slot_39, *__slate_slot_40, *__slate_slot_29 as i32);
} else {
while false {
}
_dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"BEI %d %pR: from Enhanced Allocation, properties %#02x\n".as_ptr() as *const u8, *__slate_slot_32, *__slate_slot_40, *__slate_slot_29 as i32);
}
}
}
}
}
}
}
}
}
}
return offset.wrapping_add(*__slate_slot_38);
}
}

/// Enhanced Allocation Initialization
#[unsafe(no_mangle)]
extern "C" fn pci_ea_init(dev: *mut pci_dev) {
let mut ea: i32 = 0;
let mut num_ent: u8 = 0;
let mut offset: i32 = 0;
let mut i: i32 = 0;
// find PCI EA capability in list
ea = pci_find_capability(dev, 20i32) as i32;
if ea == 0 {
return;
}
// determine the number of entries
unsafe { pci_bus_read_config_byte((*dev).bus, (*dev).devfn, ea.wrapping_add(2), core::ptr::addr_of_mut!(num_ent)) };
num_ent = ((num_ent as i32) & 63) as u8;
offset = ea.wrapping_add(4);
// Skip DWORD 2 for type 1 functions
if ((unsafe { (*dev).hdr_type }) as i32) == 1 {
offset = offset.wrapping_add(4);
}
// parse each EA entry
while i < (num_ent as i32) {
offset = pci_ea_read(dev, offset);
i = i.wrapping_add(1);
}
}

fn pci_add_saved_cap(pci_dev: *mut pci_dev, new_cap: *mut pci_cap_saved_state) {
hlist_add_head(unsafe { core::ptr::addr_of_mut!((*new_cap).next) }, unsafe { core::ptr::addr_of_mut!((*pci_dev).saved_cap_space) });
}

/// _pci_add_cap_save_buffer - allocate buffer for saving given
/// 			      capability registers
/// @dev: the PCI device
/// @cap: the capability to allocate the buffer for
/// @extended: Standard or Extended capability ID
/// @size: requested size of the buffer
fn _pci_add_cap_save_buffer(dev: *mut pci_dev, cap: u16, extended: bool, size: u32) -> i32 {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let pos: i32 = if extended {
pci_find_ext_capability(dev, cap as i32) as i32
} else {
pci_find_capability(dev, cap as i32) as i32
};
if pos == 0 {
return 0;
}
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(24u64.wrapping_add(size as u64), 3264u32, unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(24u64.wrapping_add(size as u64), 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
__v0 = __v1;
let save_state: *mut pci_cap_saved_state = __v0 as *mut pci_cap_saved_state;
if save_state.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*save_state).cap.cap_nr = cap;
}
unsafe {
(*save_state).cap.cap_extended = extended;
}
unsafe {
(*save_state).cap.size = size;
}
pci_add_saved_cap(dev, save_state);
0
}

#[unsafe(no_mangle)]
extern "C" fn pci_add_cap_save_buffer(dev: *mut pci_dev, cap: u8, size: u32) -> i32 {
_pci_add_cap_save_buffer(dev, cap as u16, false, size)
}

#[unsafe(no_mangle)]
extern "C" fn pci_add_ext_cap_save_buffer(dev: *mut pci_dev, cap: u16, size: u32) -> i32 {
_pci_add_cap_save_buffer(dev, cap, true, size)
}

/// pci_allocate_cap_save_buffers - allocate buffers for saving capabilities
/// @dev: the PCI device
#[unsafe(no_mangle)]
extern "C" fn pci_allocate_cap_save_buffers(dev: *mut pci_dev) {
let mut error: i32 = pci_add_cap_save_buffer(dev, 16u8, 7u64.wrapping_mul(core::mem::size_of::<u16>() as u64) as u32);
if error != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"unable to preallocate PCI Express save buffer\n".as_ptr() as *const u8) };
}
error = pci_add_cap_save_buffer(dev, 7u8, core::mem::size_of::<u16>() as u32);
if error != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"unable to preallocate PCI-X save buffer\n".as_ptr() as *const u8) };
}
error = pci_add_ext_cap_save_buffer(dev, 24u16, 2u64.wrapping_mul(core::mem::size_of::<u16>() as u64) as u32);
if error != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"unable to allocate suspend buffer for LTR\n".as_ptr() as *const u8) };
}
unsafe { pci_allocate_vc_save_buffers(dev) };
}

#[unsafe(no_mangle)]
extern "C" fn pci_free_cap_save_buffers(dev: *mut pci_dev) {
let mut tmp: *mut pci_cap_saved_state = unsafe { core::mem::zeroed() };
let mut n: *mut hlist_node = unsafe { core::mem::zeroed() };
tmp = hlist_entry_safe!(unsafe { (*core::ptr::addr_of_mut!((*dev).saved_cap_space)).first });
loop {
let __v4: bool = if !tmp.is_null() {
let mut __v5: i32;
n = unsafe { (*tmp).next.next };
__v5 = 1;
__v5 != 0
} else {
false
};
if !__v4 {
break;
}
unsafe { kfree(tmp as *const ()) };
tmp = hlist_entry_safe!(n);
}
}

/// pci_configure_ari - enable or disable ARI forwarding
/// @dev: the PCI device
///
/// If @dev and its upstream bridge both support ARI, enable ARI in the
/// bridge.  Otherwise, disable ARI in the bridge.
#[unsafe(no_mangle)]
extern "C" fn pci_configure_ari(dev: *mut pci_dev) {
let mut cap: u32 = 0;
let __v0: bool = (unsafe { pcie_ari_disabled }) || !pci_is_pcie(dev);
if __v0 || (unsafe { (*dev).devfn }) != 0 {
return;
}
let bridge: *mut pci_dev = unsafe { (*(*dev).bus).self_ };
if bridge.is_null() {
return;
}
unsafe { pcie_capability_read_dword(bridge, 36i32, core::ptr::addr_of_mut!(cap)) };
if cap & 32 == 0 {
return;
}
if pci_find_ext_capability(dev, 14i32) != 0 {
pcie_capability_set_word(bridge, 40i32, 32u16);
unsafe {
(*bridge).__slate_bits_3.__set_ari_enabled(1u32);
}
} else {
pcie_capability_clear_word(bridge, 40i32, 32u16);
unsafe {
(*bridge).__slate_bits_3.__set_ari_enabled(0u32);
}
}
}

fn pci_acs_flags_enabled(pdev: *mut pci_dev, mut acs_flags: u16) -> bool {
let pos: i32 = (unsafe { (*pdev).acs_cap }) as i32;
let mut ctrl: u16 = 0;
if pos == 0 {
return false;
}
// Except for egress control, capabilities are either required
// or only required if controllable.  Features missing from the
// capability field can therefore be assumed as hard-wired enabled.
acs_flags = ((acs_flags as i32) & (((unsafe { (*pdev).acs_capabilities }) as i32) | 32)) as u16;
unsafe { pci_read_config_word(pdev as *const pci_dev, pos.wrapping_add(6), core::ptr::addr_of_mut!(ctrl)) };
(ctrl as i32) & (acs_flags as i32) == (acs_flags as i32)
}

/// pci_acs_enabled - test ACS against required flags for a given device
/// @pdev: device to test
/// @acs_flags: required PCI ACS flags
///
/// Return true if the device supports the provided flags.  Automatically
/// filters out flags that are not implemented on multifunction devices.
///
/// Note that this interface checks the effective ACS capabilities of the
/// device rather than the actual capabilities.  For instance, most single
/// function endpoints are not required to support ACS because they have no
/// opportunity for peer-to-peer access.  We therefore return 'true'
/// regardless of whether the device exposes an ACS capability.  This makes
/// it much easier for callers of this function to ignore the actual type
/// or topology of the device when testing ACS support.
#[unsafe(no_mangle)]
extern "C" fn pci_acs_enabled(pdev: *mut pci_dev, acs_flags: u16) -> bool {
let ret: i32 = unsafe { pci_dev_specific_acs_enabled(pdev, acs_flags) };
if ret >= 0 {
return ret > 0;
}
// Conventional PCI and PCI-X devices never support ACS, either
// effectively or actually.  The shared bus topology implies that
// any device on the bus can receive or snoop DMA.
if !pci_is_pcie(pdev) {
return false;
}
match pci_pcie_type(pdev as *const pci_dev) {
7 | 8 | 10 => {
// PCI/X-to-PCIe bridges are not specifically mentioned by the spec,
// but since their primary interface is PCI/X, we conservatively
// handle them as we would a non-PCIe device.
return false;
// PCIe 3.0, 6.12.1 excludes ACS on these devices.  "ACS is never
// applicable... must never implement an ACS Extended Capability...".
// This seems arbitrary, but we take a conservative interpretation
// of this statement.
// PCIe 3.0, 6.12.1.1 specifies that downstream and root ports should
// implement ACS in order to indicate their peer-to-peer capabilities,
// regardless of whether they are single- or multi-function devices.
}
4 | 6 => {
return pci_acs_flags_enabled(pdev, acs_flags);
// PCIe 3.0, 6.12.1.2 specifies ACS capabilities that should be
// implemented by the remaining PCIe types to indicate peer-to-peer
// capabilities, but only when they are part of a multifunction
// device.  The footnote for section 6.12 indicates the specific
// PCIe types included here.
}
0 | 1 | 5 | 9 => {
if ((unsafe { (*pdev).__slate_bits_3.__get_multifunction() }) as i32) != 0 {
return pci_acs_flags_enabled(pdev, acs_flags);
}
}
_ => {
}
}
// PCIe 3.0, 6.12.1.3 specifies no ACS capabilities are applicable
// to single function devices with the exception of downstream ports.
true
}

/// pci_acs_path_enabled - test ACS flags from start to end in a hierarchy
/// @start: starting downstream device
/// @end: ending upstream device or NULL to search to the root bus
/// @acs_flags: required flags
///
/// Walk up a device tree from start to end testing PCI ACS support.  If
/// any step along the way does not support the required flags, return false.
#[unsafe(no_mangle)]
extern "C" fn pci_acs_path_enabled(mut parent: *mut pci_dev, end: *mut pci_dev, acs_flags: u16) -> bool {
let mut pdev: *mut pci_dev = unsafe { core::mem::zeroed() };
loop {
pdev = parent;
if !pci_acs_enabled(pdev, acs_flags) {
return false;
}
if pci_is_root_bus(unsafe { (*pdev).bus }) {
return end.is_null();
}
parent = unsafe { (*(*pdev).bus).self_ };
if pdev == end {
break;
}
}
true
}

/// pci_acs_init - Initialize ACS if hardware supports it
/// @dev: the PCI device
#[unsafe(no_mangle)]
extern "C" fn pci_acs_init(dev: *mut pci_dev) {
let mut pos: i32 = 0;
unsafe {
(*dev).acs_cap = pci_find_ext_capability(dev, 13i32);
}
pos = (unsafe { (*dev).acs_cap }) as i32;
if pos == 0 {
return;
}
unsafe { pci_read_config_word(dev as *const pci_dev, pos.wrapping_add(4), core::ptr::addr_of_mut!((*dev).acs_capabilities)) };
unsafe { pci_disable_broken_acs_cap(dev) };
}

/// pci_enable_atomic_ops_to_root - enable AtomicOp requests to root port
/// @dev: the PCI device
/// @cap_mask: mask of desired AtomicOp sizes, including one or more of:
/// 	PCI_EXP_DEVCAP2_ATOMIC_COMP32
/// 	PCI_EXP_DEVCAP2_ATOMIC_COMP64
/// 	PCI_EXP_DEVCAP2_ATOMIC_COMP128
///
/// Return 0 if all upstream bridges support AtomicOp routing, egress
/// blocking is disabled on all upstream ports, and the root port supports
/// the requested completion capabilities (32-bit, 64-bit and/or 128-bit
/// AtomicOp completion), or negative otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_enable_atomic_ops_to_root")]
extern "C" fn pci_enable_atomic_ops_to_root(dev: *mut pci_dev, cap_mask: u32) -> i32 {
let mut bridge: *mut pci_dev = unsafe { core::mem::zeroed() };
let mut cap: u32 = 0;
let mut ctl2: u32 = 0;
// Per PCIe r7.0, sec 7.5.3.16, the AtomicOp Requester Enable bit
// in Device Control 2 is reserved in VFs and the PF value applies
// to all associated VFs.
if ((unsafe { (*dev).__slate_bits_3.__get_is_virtfn() }) as i32) != 0 {
return 22i32.wrapping_neg();
}
if !pci_is_pcie(dev) {
return 22i32.wrapping_neg();
}
// Per PCIe r7.0, sec 6.15, endpoints and root ports may be
// AtomicOp requesters.  For now, we only support (legacy) endpoints
// as requesters and root ports as completers.  No endpoints as
// completers, and no peer-to-peer.
match pci_pcie_type(dev as *const pci_dev) {
0 | 1 => {
}
_ => {
return 22i32.wrapping_neg();
}
}
let root: *mut pci_dev = pcie_find_root_port(dev);
if root.is_null() {
// A hypervisor may expose a topology with the root port
// not visible to the guest.  If the hypervisor has already
// set AtomicOp Requester Enable in the endpoint, we assume
// it has verified support in the root port, so it is safe
// for the driver to use AtomicOps.
unsafe { pcie_capability_read_dword(dev, 40i32, core::ptr::addr_of_mut!(ctl2)) };
if ctl2 & 64 != 0 {
return 0;
}
return 22i32.wrapping_neg();
}
unsafe { pcie_capability_read_dword(root, 36i32, core::ptr::addr_of_mut!(cap)) };
if cap & cap_mask != cap_mask {
return 22i32.wrapping_neg();
}
bridge = pci_upstream_bridge(dev);
while bridge != root {
// Upstream ports must not block AtomicOps on egress
// All switch ports need to route AtomicOps
match pci_pcie_type(bridge as *const pci_dev) {
5 => {
unsafe { pcie_capability_read_dword(bridge, 40i32, core::ptr::addr_of_mut!(ctl2)) }; // Upstream ports must not block AtomicOps on egress
if ctl2 & 128 != 0 {
return 22i32.wrapping_neg();
}
// All switch ports need to route AtomicOps
unsafe { pcie_capability_read_dword(bridge, 36i32, core::ptr::addr_of_mut!(cap)) };
if cap & 64 == 0 {
return 22i32.wrapping_neg();
}
}
6 => {
unsafe { pcie_capability_read_dword(bridge, 36i32, core::ptr::addr_of_mut!(cap)) };
if cap & 64 == 0 {
return 22i32.wrapping_neg();
}
}
_ => {
}
}
bridge = pci_upstream_bridge(bridge);
}
pcie_capability_set_word(dev, 40i32, 64u16);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_enable_atomic_ops_to_root_785: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, u32) -> i32>, *mut ()>(Some(pci_enable_atomic_ops_to_root)) };

/// pci_release_region - Release a PCI bar
/// @pdev: PCI device whose resources were previously reserved by
/// 	  pci_request_region()
/// @bar: BAR to release
///
/// Releases the PCI I/O and memory resources previously reserved by a
/// successful call to pci_request_region().  Call this function only
/// after all use of the PCI regions has ceased.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_release_region")]
extern "C" fn pci_release_region(pdev: *mut pci_dev, bar: i32) {
if !pci_bar_index_is_valid(bar) {
return;
}
if pci_resource_len!(pdev, bar) == 0 {
return;
}
if pci_resource_flags!(pdev, bar) & 256 != 0 {
release_region!(pci_resource_start!(pdev, bar), pci_resource_len!(pdev, bar));
} else {
if pci_resource_flags!(pdev, bar) & 512 != 0 {
release_mem_region!(pci_resource_start!(pdev, bar), pci_resource_len!(pdev, bar));
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_release_region_786: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32)>, *mut ()>(Some(pci_release_region)) };

/// __pci_request_region - Reserved PCI I/O and memory resource
/// @pdev: PCI device whose resources are to be reserved
/// @bar: BAR to be reserved
/// @name: name of the driver requesting the resource
/// @exclusive: whether the region access is exclusive or not
///
/// Returns: 0 on success, negative error code on failure.
///
/// Mark the PCI region associated with PCI device @pdev BAR @bar as being
/// reserved by owner @name. Do not access any address inside the PCI regions
/// unless this call returns successfully.
///
/// If @exclusive is set, then the region is marked so that userspace
/// is explicitly not allowed to map the resource via /dev/mem or
/// sysfs MMIO access.
///
/// Returns 0 on success, or %EBUSY on error.  A warning
/// message is also printed on failure.
fn __pci_request_region(pdev: *mut pci_dev, bar: i32, name: *const u8, exclusive: i32) -> i32 {
unsafe {
if !pci_bar_index_is_valid(bar) {
return 22i32.wrapping_neg();
} else {
if pci_resource_len!(pdev, bar) == 0 {
return 0;
} else {
'__join_1: {
if pci_resource_flags!(pdev, bar) & 256 != 0 {
if request_region__2!(pci_resource_start!(pdev, bar), pci_resource_len!(pdev, bar), name).is_null() {
break '__join_1;
}
} else {
if pci_resource_flags!(pdev, bar) & 512 != 0 {
if __request_mem_region!(pci_resource_start!(pdev, bar), pci_resource_len!(pdev, bar), name, exclusive).is_null() {
break '__join_1;
}
}
}
return 0;
}
while false {
}
_dev_warn(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"BAR %d: can't reserve %pR\n".as_ptr() as *const u8, bar, ((*pdev).resource.as_mut_ptr() as *mut resource).wrapping_offset(bar as isize));
return 16i32.wrapping_neg();
}
}
}
}

/// pci_request_region - Reserve PCI I/O and memory resource
/// @pdev: PCI device whose resources are to be reserved
/// @bar: BAR to be reserved
/// @name: name of the driver requesting the resource
///
/// Returns: 0 on success, negative error code on failure.
///
/// Mark the PCI region associated with PCI device @pdev BAR @bar as being
/// reserved by owner @name. Do not access any address inside the PCI regions
/// unless this call returns successfully.
///
/// Returns 0 on success, or %EBUSY on error.  A warning
/// message is also printed on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_request_region")]
extern "C" fn pci_request_region(pdev: *mut pci_dev, bar: i32, name: *const u8) -> i32 {
__pci_request_region(pdev, bar, name, 0i32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_request_region_787: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32, *const u8) -> i32>, *mut ()>(Some(pci_request_region)) };

/// pci_release_selected_regions - Release selected PCI I/O and memory resources
/// @pdev: PCI device whose resources were previously reserved
/// @bars: Bitmask of BARs to be released
///
/// Release selected PCI I/O and memory resources previously reserved.
/// Call this function only after all use of the PCI regions has ceased.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_release_selected_regions")]
extern "C" fn pci_release_selected_regions(pdev: *mut pci_dev, bars: i32) {
for i in 0i32..6i32 {
if bars & 1i32 << i != 0 {
pci_release_region(pdev, i);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_release_selected_regions_788: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32)>, *mut ()>(Some(pci_release_selected_regions)) };

fn __pci_request_selected_regions(pdev: *mut pci_dev, bars: i32, name: *const u8, excl: i32) -> i32 {
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_4 = 0;
'__join_1: {
while *__slate_slot_4 < 6 {
if bars & 1i32 << *__slate_slot_4 != 0 {
if __pci_request_region(pdev, *__slate_slot_4, name, excl) != 0 {
break '__join_1;
}
}
let __v3: i32 = *__slate_slot_4;
*__slate_slot_4 = __v3.wrapping_add(1);
}
return 0;
}
loop {
let __v1: i32 = *__slate_slot_4;
let __v0: i32 = __v1.wrapping_sub(1);
*__slate_slot_4 = __v0;
if __v0 >= 0 {
if bars & 1i32 << *__slate_slot_4 != 0 {
pci_release_region(pdev, *__slate_slot_4);
}
} else {
break;
}
}
return 16i32.wrapping_neg();
}
}

/// pci_request_selected_regions - Reserve selected PCI I/O and memory resources
/// @pdev: PCI device whose resources are to be reserved
/// @bars: Bitmask of BARs to be requested
/// @name: Name of the driver requesting the resources
///
/// Returns: 0 on success, negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_request_selected_regions")]
extern "C" fn pci_request_selected_regions(pdev: *mut pci_dev, bars: i32, name: *const u8) -> i32 {
__pci_request_selected_regions(pdev, bars, name, 0i32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_request_selected_regions_789: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32, *const u8) -> i32>, *mut ()>(Some(pci_request_selected_regions)) };

/// pci_request_selected_regions_exclusive - Request regions exclusively
/// @pdev: PCI device to request regions from
/// @bars: bit mask of BARs to request
/// @name: name of the driver requesting the resources
///
/// Returns: 0 on success, negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_request_selected_regions_exclusive")]
extern "C" fn pci_request_selected_regions_exclusive(pdev: *mut pci_dev, bars: i32, name: *const u8) -> i32 {
__pci_request_selected_regions(pdev, bars, name, 134217728i32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_request_selected_regions_exclusive_790: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32, *const u8) -> i32>, *mut ()>(Some(pci_request_selected_regions_exclusive)) };

/// pci_release_regions - Release reserved PCI I/O and memory resources
/// @pdev: PCI device whose resources were previously reserved by
/// 	  pci_request_regions()
///
/// Releases all PCI I/O and memory resources previously reserved by a
/// successful call to pci_request_regions().  Call this function only
/// after all use of the PCI regions has ceased.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_release_regions")]
extern "C" fn pci_release_regions(pdev: *mut pci_dev) {
pci_release_selected_regions(pdev, 63i32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_release_regions_791: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_release_regions)) };

/// pci_request_regions - Reserve PCI I/O and memory resources
/// @pdev: PCI device whose resources are to be reserved
/// @name: name of the driver requesting the resources
///
/// Mark all PCI regions associated with PCI device @pdev as being reserved by
/// owner @name. Do not access any address inside the PCI regions unless this
/// call returns successfully.
///
/// Returns 0 on success, or %EBUSY on error.  A warning
/// message is also printed on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_request_regions")]
extern "C" fn pci_request_regions(pdev: *mut pci_dev, name: *const u8) -> i32 {
pci_request_selected_regions(pdev, 63i32, name)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_request_regions_792: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, *const u8) -> i32>, *mut ()>(Some(pci_request_regions)) };

/// pci_request_regions_exclusive - Reserve PCI I/O and memory resources
/// @pdev: PCI device whose resources are to be reserved
/// @name: name of the driver requesting the resources
///
/// Returns: 0 on success, negative error code on failure.
///
/// Mark all PCI regions associated with PCI device @pdev as being reserved
/// by owner @name. Do not access any address inside the PCI regions
/// unless this call returns successfully.
///
/// pci_request_regions_exclusive() will mark the region so that /dev/mem
/// and the sysfs MMIO access will not be allowed.
///
/// Returns 0 on success, or %EBUSY on error.  A warning message is also
/// printed on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_request_regions_exclusive")]
extern "C" fn pci_request_regions_exclusive(pdev: *mut pci_dev, name: *const u8) -> i32 {
pci_request_selected_regions_exclusive(pdev, 63i32, name)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_request_regions_exclusive_793: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, *const u8) -> i32>, *mut ()>(Some(pci_request_regions_exclusive)) };

/// Record the PCI IO range (expressed as CPU physical address + size).
/// Return a negative value if an error has occurred, zero otherwise
#[unsafe(no_mangle)]
extern "C" fn pci_register_io_range(fwnode: *const fwnode_handle, addr: u64, size: u64) -> i32 {
let ret: i32 = 0;
ret
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_pio_to_address")]
extern "C" fn pci_pio_to_address(pio: u64) -> u64 {
1i32.wrapping_neg() as u64
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_pio_to_address_794: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64) -> u64>, *mut ()>(Some(pci_pio_to_address)) };

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pci_address_to_pio(address: u64) -> u64 {
if address > 65535 {
return 1i32.wrapping_neg() as u64;
}
address
}

/// pci_remap_iospace - Remap the memory mapped I/O space
/// @res: Resource describing the I/O space
/// @phys_addr: physical address of range to be mapped
///
/// Remap the memory mapped I/O space described by the @res and the CPU
/// physical address @phys_addr into virtual address space.  Only
/// architectures that have memory mapped IO functions defined (and the
/// PCI_IOBASE value defined) should call this function.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_remap_iospace")]
extern "C" fn pci_remap_iospace(res: *const resource, phys_addr: u64) -> i32 {
// This architecture does not have memory mapped I/O space,
// so this function should never be called
let mut __v0: i32;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"This architecture does not support memory mapped I/O\n".as_ptr() as *const u8) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_pci_remap_iospace_3, sym __slate_asm_string_pci_remap_iospace_4, const 4130 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
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
19i32.wrapping_neg()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_795: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_remap_iospace_796: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const resource, u64) -> i32>, *mut ()>(Some(pci_remap_iospace)) };

/// pci_unmap_iospace - Unmap the memory mapped I/O space
/// @res: resource to be unmapped
///
/// Unmap the CPU virtual address @res from virtual address space.  Only
/// architectures that have memory mapped IO functions defined (and the
/// PCI_IOBASE value defined) should call this function.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_unmap_iospace")]
extern "C" fn pci_unmap_iospace(res: *mut resource) {
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_unmap_iospace_797: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut resource)>, *mut ()>(Some(pci_unmap_iospace)) };

fn __pci_set_master(dev: *mut pci_dev, enable: bool) {
let mut old_cmd: u16 = 0;
let mut cmd: u16 = 0;
unsafe { pci_read_config_word(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(old_cmd)) };
if enable {
cmd = ((old_cmd as i32) | 4) as u16;
} else {
cmd = ((old_cmd as i32) & !4) as u16;
}
if (cmd as i32) != (old_cmd as i32) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s bus mastering\n".as_ptr() as *const u8, if enable { c"enabling".as_ptr() as *mut u8 } else { c"disabling".as_ptr() as *mut u8 }) };
}
unsafe { pci_write_config_word(dev as *const pci_dev, 4i32, cmd) };
}
unsafe {
(*dev).__slate_bits_3.__set_is_busmaster(enable as u32);
}
}

/// pcibios_setup - process "pci=" kernel boot arguments
/// @str: string used to pass in "pci=" kernel boot arguments
///
/// Process kernel boot arguments.  This is the default implementation.
/// Architecture specific implementations can override this as necessary.
#[unsafe(no_mangle)]
#[linkage = "weak"]
#[unsafe(link_section = ".init.text")]
extern "C" fn pcibios_setup(str: *mut u8) -> *mut u8 {
str
}

/// pcibios_set_master - enable PCI bus-mastering for device dev
/// @dev: the PCI device to enable
///
/// Enables PCI bus-mastering for the device.  This is the default
/// implementation.  Architecture specific implementations can override
/// this if necessary.
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pcibios_set_master(dev: *mut pci_dev) {
let mut lat: u8 = 0;
// The latency timer doesn't apply to PCIe (either Type 0 or Type 1)
if pci_is_pcie(dev) {
return;
}
unsafe { pci_read_config_byte(dev as *const pci_dev, 13i32, core::ptr::addr_of_mut!(lat)) };
if (lat as i32) < 16 {
lat = (if 64 <= unsafe { pcibios_max_latency } { 64u32 } else { unsafe { pcibios_max_latency } }) as u8;
} else {
if ((lat as i32) as u32) > unsafe { pcibios_max_latency } {
lat = (unsafe { pcibios_max_latency }) as u8;
} else {
return;
}
}
unsafe { pci_write_config_byte(dev as *const pci_dev, 13i32, lat) };
}

/// pci_set_master - enables bus-mastering for device dev
/// @dev: the PCI device to enable
///
/// Enables bus-mastering on the device and calls pcibios_set_master()
/// to do the needed arch specific settings.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_set_master")]
extern "C" fn pci_set_master(dev: *mut pci_dev) {
__pci_set_master(dev, true);
pcibios_set_master(dev);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_set_master_798: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_set_master)) };

/// pci_clear_master - disables bus-mastering for device dev
/// @dev: the PCI device to disable
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_clear_master")]
extern "C" fn pci_clear_master(dev: *mut pci_dev) {
__pci_set_master(dev, false);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_clear_master_799: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_clear_master)) };

/// pci_set_cacheline_size - ensure the CACHE_LINE_SIZE register is programmed
/// @dev: the PCI device for which MWI is to be enabled
///
/// Helper function for pci_set_mwi.
/// Originally copied from drivers/net/acenic.c.
/// Copyright 1998-2001 by Jes Sorensen, <jes@trained-monkey.org>.
///
/// RETURNS: An appropriate -ERRNO error value on error, or zero for success.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_set_cacheline_size")]
extern "C" fn pci_set_cacheline_size(dev: *mut pci_dev) -> i32 {
let mut cacheline_size: u8 = 0;
if (unsafe { pci_cache_line_size }) == 0 {
return 22i32.wrapping_neg();
}
// Validate current setting: the PCI_CACHE_LINE_SIZE must be
// equal to or multiple of the right value.
unsafe { pci_read_config_byte(dev as *const pci_dev, 12i32, core::ptr::addr_of_mut!(cacheline_size)) };
if (cacheline_size as i32) >= ((unsafe { pci_cache_line_size }) as i32) && (cacheline_size as i32) % ((unsafe { pci_cache_line_size }) as i32) == 0 {
return 0;
}
// Write the correct value.
unsafe { pci_write_config_byte(dev as *const pci_dev, 12i32, pci_cache_line_size) };
// Read it back.
unsafe { pci_read_config_byte(dev as *const pci_dev, 12i32, core::ptr::addr_of_mut!(cacheline_size)) };
if (cacheline_size as i32) == ((unsafe { pci_cache_line_size }) as i32) {
return 0;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"cache line size of %d is not supported\n".as_ptr() as *const u8, (pci_cache_line_size as i32) << 2i32) };
}
22i32.wrapping_neg()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_set_cacheline_size_800: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_set_cacheline_size)) };

/// pci_set_mwi - enables memory-write-invalidate PCI transaction
/// @dev: the PCI device for which MWI is enabled
///
/// Enables the Memory-Write-Invalidate transaction in %PCI_COMMAND.
///
/// RETURNS: An appropriate -ERRNO error value on error, or zero for success.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_set_mwi")]
extern "C" fn pci_set_mwi(dev: *mut pci_dev) -> i32 {
let rc: i32 = pci_set_cacheline_size(dev);
let mut cmd: u16 = 0;
if rc != 0 {
return rc;
}
unsafe { pci_read_config_word(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(cmd)) };
if (cmd as i32) & 16 == 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"enabling Mem-Wr-Inval\n".as_ptr() as *const u8) };
}
cmd = ((cmd as i32) | 16) as u16;
unsafe { pci_write_config_word(dev as *const pci_dev, 4i32, cmd) };
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_set_mwi_801: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_set_mwi)) };

/// pci_try_set_mwi - enables memory-write-invalidate PCI transaction
/// @dev: the PCI device for which MWI is enabled
///
/// Enables the Memory-Write-Invalidate transaction in %PCI_COMMAND.
/// Callers are not required to check the return value.
///
/// RETURNS: An appropriate -ERRNO error value on error, or zero for success.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_try_set_mwi")]
extern "C" fn pci_try_set_mwi(dev: *mut pci_dev) -> i32 {
pci_set_mwi(dev)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_try_set_mwi_802: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_try_set_mwi)) };

/// pci_clear_mwi - disables Memory-Write-Invalidate for device dev
/// @dev: the PCI device to disable
///
/// Disables PCI Memory-Write-Invalidate transaction on the device
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_clear_mwi")]
extern "C" fn pci_clear_mwi(dev: *mut pci_dev) {
let mut cmd: u16 = 0;
unsafe { pci_read_config_word(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(cmd)) };
if (cmd as i32) & 16 != 0 {
let __v1: u16 = ((cmd as i32) & !16) as u16;
cmd = __v1;
unsafe { pci_write_config_word(dev as *const pci_dev, 4i32, cmd) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_clear_mwi_803: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_clear_mwi)) };

/// pci_disable_parity - disable parity checking for device
/// @dev: the PCI device to operate on
///
/// Disable parity checking for device @dev
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_disable_parity")]
extern "C" fn pci_disable_parity(dev: *mut pci_dev) {
let mut cmd: u16 = 0;
unsafe { pci_read_config_word(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(cmd)) };
if (cmd as i32) & 64 != 0 {
let __v1: u16 = ((cmd as i32) & !64) as u16;
cmd = __v1;
unsafe { pci_write_config_word(dev as *const pci_dev, 4i32, cmd) };
}
}

/// pci_intx - enables/disables PCI INTx for device dev
/// @pdev: the PCI device to operate on
/// @enable: boolean: whether to enable or disable PCI INTx
///
/// Enables/disables PCI INTx for device @pdev
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_intx")]
extern "C" fn pci_intx(pdev: *mut pci_dev, enable: i32) {
let mut pci_command: u16 = 0;
let mut new: u16 = 0;
unsafe { pci_read_config_word(pdev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(pci_command)) };
if enable != 0 {
new = ((pci_command as i32) & !1024) as u16;
} else {
new = ((pci_command as i32) | 1024) as u16;
}
if (new as i32) == (pci_command as i32) {
return;
}
unsafe { pci_write_config_word(pdev as *const pci_dev, 4i32, new) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_intx_804: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32)>, *mut ()>(Some(pci_intx)) };

/// pci_wait_for_pending_transaction - wait for pending transaction
/// @dev: the PCI device to operate on
///
/// Return 0 if transaction is pending 1 otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_wait_for_pending_transaction")]
extern "C" fn pci_wait_for_pending_transaction(dev: *mut pci_dev) -> i32 {
if !pci_is_pcie(dev) {
return 1;
}
pci_wait_for_pending(dev, pci_pcie_cap(dev).wrapping_add(10), 32u16)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_wait_for_pending_transaction_805: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_wait_for_pending_transaction)) };

/// pcie_flr - initiate a PCIe function level reset
/// @dev: device to reset
///
/// Initiate a function level reset unconditionally on @dev without
/// checking any flags and DEVCAP
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pcie_flr")]
extern "C" fn pcie_flr(dev: *mut pci_dev) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if !pci_dev_config_accessible(dev, c"FLR".as_ptr() as *mut u8) {
return 25i32.wrapping_neg();
} else {
if pci_wait_for_pending_transaction(dev) == 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"timed out waiting for pending transaction; performing function level reset anyway\n".as_ptr() as *const u8);
}
// Have to call it after waiting for pending DMA transaction
*__slate_slot_0 = pci_dev_reset_iommu_prepare(dev);
if *__slate_slot_0 != 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"failed to stop IOMMU for a PCI reset: %d\n".as_ptr() as *const u8, *__slate_slot_0);
return *__slate_slot_0;
} else {
pcie_capability_set_word(dev, 8i32, 32768u16);
if ((*dev).__slate_bits_1.__get_imm_ready() as i32) == 0 {
// Per PCIe r4.0, sec 6.6.2, a device must complete an FLR within
// 100ms, but may silently discard requests while the FLR is in
// progress.  Wait 100ms before trying to access the device.
msleep(100u32);
*__slate_slot_0 = pci_dev_wait(dev, c"FLR".as_ptr() as *mut u8, 60000i32);
}
pci_dev_reset_iommu_done(dev);
return *__slate_slot_0;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_flr_806: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pcie_flr)) };

/// pcie_reset_flr - initiate a PCIe function level reset
/// @dev: device to reset
/// @probe: if true, return 0 if device can be reset this way
///
/// Initiate a function level reset on @dev.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pcie_reset_flr")]
extern "C" fn pcie_reset_flr(dev: *mut pci_dev, probe: bool) -> i32 {
if ((unsafe { (*dev).dev_flags }) as i32) & 1024 != 0 {
return 25i32.wrapping_neg();
}
if (unsafe { (*dev).devcap }) & 268435456 == 0 {
return 25i32.wrapping_neg();
}
if probe {
return 0;
}
pcie_flr(dev)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_reset_flr_807: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, bool) -> i32>, *mut ()>(Some(pcie_reset_flr)) };

#[unsafe(link_section = ".text.slate_distinct.pci.pci_af_flr")]
extern "C" fn pci_af_flr(dev: *mut pci_dev, probe: bool) -> i32 {
let mut cap: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u8 = core::ptr::addr_of_mut!(cap) as *mut u8;
let mut pos: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(pos) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_1 = pci_find_capability(dev, 19i32) as i32;
if *__slate_slot_1 == 0 {
return 25i32.wrapping_neg();
} else {
if ((*dev).dev_flags as i32) & 1024 != 0 {
return 25i32.wrapping_neg();
} else {
pci_read_config_byte(dev as *const pci_dev, (*__slate_slot_1).wrapping_add(3), core::ptr::addr_of_mut!(*__slate_slot_0));
if (*__slate_slot_0 as i32) & 1 == 0 || (*__slate_slot_0 as i32) & 2 == 0 {
return 25i32.wrapping_neg();
} else {
if probe {
return 0;
} else {
// Wait for Transaction Pending bit to clear.  A word-aligned test
// is used, so we use the control offset rather than status and shift
// the test bit to match.
if pci_wait_for_pending(dev, (*__slate_slot_1).wrapping_add(4), 256u16) == 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"timed out waiting for pending transaction; performing AF function level reset anyway\n".as_ptr() as *const u8);
}
// Have to call it after waiting for pending DMA transaction
*__slate_slot_2 = pci_dev_reset_iommu_prepare(dev);
if *__slate_slot_2 != 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"failed to stop IOMMU for a PCI reset: %d\n".as_ptr() as *const u8, *__slate_slot_2);
return *__slate_slot_2;
} else {
pci_write_config_byte(dev as *const pci_dev, (*__slate_slot_1).wrapping_add(4), 1u8);
if ((*dev).__slate_bits_1.__get_imm_ready() as i32) == 0 {
// Per Advanced Capabilities for Conventional PCI ECN, 13 April 2006,
// updated 27 July 2006; a device must complete an FLR within
// 100ms, but may silently discard requests while the FLR is in
// progress.  Wait 100ms before trying to access the device.
msleep(100u32);
*__slate_slot_2 = pci_dev_wait(dev, c"AF_FLR".as_ptr() as *mut u8, 60000i32);
}
pci_dev_reset_iommu_done(dev);
return *__slate_slot_2;
}
}
}
}
}
}
}

/// pci_pm_reset - Put device into PCI_D3 and back into PCI_D0.
/// @dev: Device to reset.
/// @probe: if true, return 0 if the device can be reset this way.
///
/// If @dev supports native PCI PM and its PCI_PM_CTRL_NO_SOFT_RESET flag is
/// unset, it will be reinitialized internally when going from PCI_D3hot to
/// PCI_D0.  If that's the case and the device is not in a low-power state
/// already, force it into PCI_D3hot and back to PCI_D0, causing it to be reset.
///
/// NOTE: This causes the caller to sleep for twice the device power transition
/// cooldown period, which for the D0->D3hot and D3hot->D0 transitions is 10 ms
/// by default (i.e. unless the @dev's d3hot_delay field has a different value).
/// Moreover, only devices in D0 can be reset by this function.
#[unsafe(link_section = ".text.slate_distinct.pci.pci_pm_reset")]
extern "C" fn pci_pm_reset(dev: *mut pci_dev, probe: bool) -> i32 {
let mut csr: u16 = 0;
let mut ret: i32 = 0;
if (unsafe { (*dev).pm_cap }) == 0 || ((unsafe { (*dev).dev_flags }) as i32) & 128 != 0 {
return 25i32.wrapping_neg();
}
unsafe { pci_read_config_word(dev as *const pci_dev, ((*dev).pm_cap as i32).wrapping_add(4), core::ptr::addr_of_mut!(csr)) };
if (csr as i32) & 8 != 0 {
return 25i32.wrapping_neg();
}
if probe {
return 0;
}
if (unsafe { (*dev).current_state }) != 0 {
return 22i32.wrapping_neg();
}
ret = unsafe { pci_dev_reset_iommu_prepare(dev) };
if ret != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"failed to stop IOMMU for a PCI reset: %d\n".as_ptr() as *const u8, ret) };
return ret;
}
let __v2: u16 = ((csr as i32) & !3) as u16;
csr = __v2;
csr = ((csr as i32) | 3) as u16;
unsafe { pci_write_config_word(dev as *const pci_dev, ((*dev).pm_cap as i32).wrapping_add(4), csr) };
pci_dev_d3_sleep(dev);
let __v6: u16 = ((csr as i32) & !3) as u16;
csr = __v6;
csr = ((csr as i32) | 0) as u16;
unsafe { pci_write_config_word(dev as *const pci_dev, ((*dev).pm_cap as i32).wrapping_add(4), csr) };
pci_dev_d3_sleep(dev);
ret = pci_dev_wait(dev, c"PM D3hot->D0".as_ptr() as *mut u8, 60000i32);
unsafe { pci_dev_reset_iommu_done(dev) };
ret
}

/// pcie_wait_for_link_status - Wait for link status change
/// @pdev: Device whose link to wait for.
/// @use_lt: Use the LT bit if TRUE, or the DLLLA bit if FALSE.
/// @active: Waiting for active or inactive?
///
/// Return 0 if successful, or -ETIMEDOUT if status has not changed within
/// PCIE_LINK_RETRAIN_TIMEOUT_MS milliseconds.
fn pcie_wait_for_link_status(pdev: *mut pci_dev, use_lt: bool, active: bool) -> i32 {
let lnksta_mask: u16 = (if use_lt { 2048i32 } else { 8192i32 }) as u16;
let lnksta_match: u16 = (if active { lnksta_mask as i32 } else { 0i32 }) as u16;
let end_jiffies: u64 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }.wrapping_add(msecs_to_jiffies(1000u32));
let mut lnksta: u16 = 0;
loop {
unsafe { pcie_capability_read_word(pdev, 18i32, core::ptr::addr_of_mut!(lnksta)) };
if (lnksta as i32) & (lnksta_mask as i32) == (lnksta_match as i32) {
return 0;
}
unsafe { msleep(1u32) };
if !time_before!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, end_jiffies) {
break;
}
}
110i32.wrapping_neg()
}

/// pcie_retrain_link - Request a link retrain and wait for it to complete
/// @pdev: Device whose link to retrain.
/// @use_lt: Use the LT bit if TRUE, or the DLLLA bit if FALSE, for status.
///
/// Trigger retraining of the PCIe Link and wait for the completion of the
/// retraining. As link retraining is known to asserts LBMS and may change
/// the Link Speed, LBMS is cleared after the retraining and the Link Speed
/// of the subordinate bus is updated.
///
/// Retrain completion status is retrieved from the Link Status Register
/// according to @use_lt.  It is not verified whether the use of the DLLLA
/// bit is valid.
///
/// Return 0 if successful, or -ETIMEDOUT if training has not completed
/// within PCIE_LINK_RETRAIN_TIMEOUT_MS milliseconds.
#[unsafe(no_mangle)]
extern "C" fn pcie_retrain_link(pdev: *mut pci_dev, use_lt: bool) -> i32 {
let mut rc: i32 = 0;
// Ensure the updated LNKCTL parameters are used during link
// training by checking that there is no ongoing link training that
// may have started before link parameters were changed, so as to
// avoid LTSSM race as recommended in Implementation Note at the end
// of PCIe r6.1 sec 7.5.3.7.
rc = pcie_wait_for_link_status(pdev, true, false);
if rc != 0 {
return rc;
}
pcie_capability_set_word(pdev, 16i32, 32u16);
if ((unsafe { (*pdev).__slate_bits_1.__get_clear_retrain_link() }) as i32) != 0 {
// Due to an erratum in some devices the Retrain Link bit
// needs to be cleared again manually to allow the link
// training to succeed.
pcie_capability_clear_word(pdev, 16i32, 32u16);
}
rc = pcie_wait_for_link_status(pdev, use_lt, !use_lt);
// Clear LBMS after a manual retrain so that the bit can be used
// to track link speed or width changes made by hardware itself
// in attempt to correct unreliable link operation.
unsafe { pcie_reset_lbms(pdev) };
// Ensure the Link Speed updates after retraining in case the Link
// Speed was changed because of the retraining. While the bwctrl's
// IRQ handler normally picks up the new Link Speed, clearing LBMS
// races with the IRQ handler reading the Link Status register and
// can result in the handler returning early without updating the
// Link Speed.
if !unsafe { (*pdev).subordinate }.is_null() {
unsafe { pcie_update_link_speed((*pdev).subordinate, 0u32) };
}
rc
}

/// pcie_wait_for_link_delay - Wait until link is active or inactive
/// @pdev: Bridge device
/// @active: waiting for active or inactive?
/// @delay: Delay to wait after link has become active (in ms)
///
/// Use this to wait till link becomes active or inactive.
fn pcie_wait_for_link_delay(pdev: *mut pci_dev, active: bool, delay: i32) -> bool {
let mut rc: i32 = 0;
// Some controllers might not implement link active reporting. In this
// case, we wait for 1000 ms + any delay requested by the caller.
if ((unsafe { (*pdev).__slate_bits_3.__get_link_active_reporting() }) as i32) == 0 {
unsafe { msleep(1000i32.wrapping_add(delay) as u32) };
return true;
}
// PCIe r4.0 sec 6.6.1, a component must enter LTSSM Detect within 20ms,
// after which we should expect the link to be active if the reset was
// successful. If so, software must wait a minimum 100ms before sending
// configuration requests to devices downstream this port.
//
// If the link fails to activate, either the device was physically
// removed or the link is permanently failed.
if active {
unsafe { msleep(20u32) };
}
rc = pcie_wait_for_link_status(pdev, false, active);
if active {
if rc != 0 {
rc = unsafe { pcie_failed_link_retrain(pdev) };
}
if rc != 0 {
return false;
}
unsafe { msleep(delay as u32) };
return true;
}
if rc != 0 {
return false;
}
true
}

/// pcie_wait_for_link - Wait until link is active or inactive
/// @pdev: Bridge device
/// @active: waiting for active or inactive?
///
/// Use this to wait till link becomes active or inactive.
#[unsafe(no_mangle)]
extern "C" fn pcie_wait_for_link(pdev: *mut pci_dev, active: bool) -> bool {
pcie_wait_for_link_delay(pdev, active, 100i32)
}

// Find maximum D3cold delay required by all the devices on the bus.  The
// spec says 100 ms, but firmware can lower it and we allow drivers to
// increase it as well.
//
// Context: Called with @pci_bus_sem locked for reading.
fn pci_bus_max_d3cold_delay(bus: *const pci_bus) -> i32 {
let mut pdev: *const pci_dev = unsafe { core::mem::zeroed() };
let mut min_delay: i32 = 100;
let mut max_delay: i32 = 0;
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(pci_bus_sem) });
pdev = list_first_entry__13!(unsafe { core::ptr::addr_of!((*bus).devices) });
while list_entry_is_head__3!(pdev, unsafe { core::ptr::addr_of!((*bus).devices) }, bus_list) == 0 {
if (unsafe { (*pdev).d3cold_delay }) < (min_delay as u32) {
min_delay = (unsafe { (*pdev).d3cold_delay }) as i32;
}
if (unsafe { (*pdev).d3cold_delay }) > (max_delay as u32) {
max_delay = (unsafe { (*pdev).d3cold_delay }) as i32;
}
pdev = list_next_entry__11!(pdev, bus_list);
}
let mut __v4: i32;
let __UNIQUE_ID_x__808: i32 = min_delay;
let __UNIQUE_ID_y__809: i32 = max_delay;
loop {
if __types_ok__3!(__UNIQUE_ID_x__808, __UNIQUE_ID_y__809) == 0 {
unsafe { __compiletime_assert_810() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = if __UNIQUE_ID_x__808 > __UNIQUE_ID_y__809 { __UNIQUE_ID_x__808 } else { __UNIQUE_ID_y__809 };
__v4
}

/// pci_bridge_wait_for_secondary_bus - Wait for secondary bus to be accessible
/// @dev: PCI bridge
/// @reset_type: reset type in human-readable form
///
/// Handle necessary delays before access to the devices on the secondary
/// side of the bridge are permitted after D3cold to D0 transition
/// or Conventional Reset.
///
/// For PCIe this means the delays in PCIe 5.0 section 6.6.1. For
/// conventional PCI it means Tpvrh + Trhfa specified in PCI 3.0 section
/// 4.3.2.
///
/// Return 0 on success or -ENOTTY if the first device on the secondary bus
/// failed to become accessible.
#[unsafe(no_mangle)]
extern "C" fn pci_bridge_wait_for_secondary_bus(dev: *mut pci_dev, reset_type: *mut u8) -> i32 {
let mut child: *mut pci_dev = core::ptr::null_mut::<pci_dev>();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(child) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_pci_dev_put(__slate_cleanup_ptr_0 as *mut _));
let mut delay: i32 = 0;
if pci_dev_is_disconnected(dev as *const pci_dev) {
return 0;
}
if !pci_is_bridge(dev) {
return 0;
}
unsafe { down_read(core::ptr::addr_of_mut!(pci_bus_sem)) };
// We only deal with devices that are present currently on the bus.
// For any hot-added devices the access delay is handled in pciehp
// board_added(). In case of ACPI hotplug the firmware is expected
// to configure the devices before OS is notified.
let __v1: bool = unsafe { (*dev).subordinate }.is_null() || list_empty((unsafe { core::ptr::addr_of_mut!((*(*dev).subordinate).devices) }) as *const list_head) != 0;
if __v1 {
unsafe { up_read(core::ptr::addr_of_mut!(pci_bus_sem)) };
return 0;
}
// Take d3cold_delay requirements into account
delay = pci_bus_max_d3cold_delay((unsafe { (*dev).subordinate }) as *const pci_bus);
if delay == 0 {
unsafe { up_read(core::ptr::addr_of_mut!(pci_bus_sem)) };
return 0;
}
child = unsafe { pci_dev_get(list_first_entry__12!(core::ptr::addr_of_mut!((*(*dev).subordinate).devices), pci_dev)) };
unsafe { up_read(core::ptr::addr_of_mut!(pci_bus_sem)) };
// Conventional PCI and PCI-X we need to wait Tpvrh + Trhfa before
// accessing the device after reset (that is 1000 ms + 100 ms).
if !pci_is_pcie(dev) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"waiting %d ms for secondary bus\n".as_ptr() as *const u8, 1000i32.wrapping_add(delay)) };
}
unsafe { msleep(1000i32.wrapping_add(delay) as u32) };
return 0;
}
// For PCIe downstream and root ports that do not support speeds
// greater than 5 GT/s need to wait minimum 100 ms. For higher
// speeds (gen3) we need to wait first for the data link layer to
// become active.
//
// However, 100 ms is the minimum and the PCIe spec says the
// software must allow at least 1s before it can determine that the
// device that did not respond is a broken device. Also device can
// take longer than that to respond if it indicates so through Request
// Retry Status completions.
//
// Therefore we wait for 100 ms and check for the device presence
// until the timeout expires.
if !pcie_downstream_port(dev as *const pci_dev) {
return 0;
}
if (pcie_get_speed_cap(dev) as u32) <= 21 {
let mut status: u16 = 0;
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"waiting %d ms for downstream link\n".as_ptr() as *const u8, delay) };
}
unsafe { msleep(delay as u32) };
if pci_dev_wait(child, reset_type, 1000i32.wrapping_sub(delay)) == 0 {
return 0;
}
// If the port supports active link reporting we now check
// whether the link is active and if not bail out early with
// the assumption that the device is not present anymore.
if ((unsafe { (*dev).__slate_bits_3.__get_link_active_reporting() }) as i32) == 0 {
return 25i32.wrapping_neg();
}
unsafe { pcie_capability_read_word(dev, 18i32, core::ptr::addr_of_mut!(status)) };
if (status as i32) & 8192 == 0 {
return 25i32.wrapping_neg();
}
return pci_dev_wait(child, reset_type, 59000i32);
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"waiting %d ms for downstream link, after activation\n".as_ptr() as *const u8, delay) };
}
if !pcie_wait_for_link_delay(dev, true, delay) {
// Did not train, no need to wait any further
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Data Link Layer Link Active not set in %d msec\n".as_ptr() as *const u8, delay) };
return 25i32.wrapping_neg();
}
pci_dev_wait(child, reset_type, 60000i32.wrapping_sub(delay))
}

#[unsafe(no_mangle)]
extern "C" fn pci_reset_secondary_bus(dev: *mut pci_dev) {
let mut ctrl: u16 = 0;
unsafe { pci_read_config_word(dev as *const pci_dev, 62i32, core::ptr::addr_of_mut!(ctrl)) };
ctrl = ((ctrl as i32) | 64) as u16;
unsafe { pci_write_config_word(dev as *const pci_dev, 62i32, ctrl) };
// PCI spec v3.0 7.6.4.2 requires minimum Trst of 1ms.  Double
// this to 2ms to ensure that we meet the minimum requirement.
unsafe { msleep(2u32) };
let __v3: u16 = ((ctrl as i32) & !64) as u16;
ctrl = __v3;
unsafe { pci_write_config_word(dev as *const pci_dev, 62i32, ctrl) };
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pcibios_reset_secondary_bus(dev: *mut pci_dev) {
let host: *mut pci_host_bridge = unsafe { pci_find_host_bridge((*dev).bus) };
let mut ret: i32 = 0;
if pci_is_root_bus(unsafe { (*dev).bus }) && (unsafe { (*host).reset_root_port }) != None {
ret = unsafe { unsafe { (*host).reset_root_port }.unwrap()(host, dev) };
if ret != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Failed to reset Root Port: %d\n".as_ptr() as *const u8, ret) };
} else {
pci_restore_state(dev);
}
return;
}
pci_reset_secondary_bus(dev);
}

// pci_bridge_secondary_bus_reset - Reset the secondary bus on a PCI bridge.
// @dev: Bridge device
//
// Use the bridge control register to assert reset on the secondary bus.
// Devices on the secondary bus are left in power-on state.
#[unsafe(link_section = ".data..read_mostly")]
static mut __print_once: bool = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_bridge_secondary_bus_reset_811: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_bridge_secondary_bus_reset)) };

fn pci_parent_bus_reset(dev: *mut pci_dev, probe: bool) -> i32 {
let mut pdev: *mut pci_dev = unsafe { core::mem::zeroed() };
if pci_is_root_bus(unsafe { (*dev).bus }) || !unsafe { (*dev).subordinate }.is_null() || unsafe { (*(*dev).bus).self_ }.is_null() || ((unsafe { (*dev).dev_flags }) as i32) & 64 != 0 {
return 25i32.wrapping_neg();
}
pdev = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*(*dev).bus).devices) });
while list_entry_is_head!(pdev, unsafe { core::ptr::addr_of_mut!((*(*dev).bus).devices) }, bus_list) == 0 {
if pdev != dev {
return 25i32.wrapping_neg();
}
pdev = list_next_entry!(pdev, bus_list);
}
if probe {
return 0;
}
pci_bridge_secondary_bus_reset(unsafe { (*(*dev).bus).self_ })
}

fn pci_reset_hotplug_slot(hotplug: *mut hotplug_slot, probe: bool) -> i32 {
let mut rc: i32 = 25i32.wrapping_neg();
let __v0: bool = hotplug.is_null() || !unsafe { try_module_get((*hotplug).owner) };
if __v0 {
return rc;
}
if (unsafe { (*(*hotplug).ops).reset_slot }) != None {
rc = unsafe { unsafe { (*(*hotplug).ops).reset_slot }.unwrap()(hotplug, probe) };
}
unsafe { module_put((*hotplug).owner) };
rc
}

fn pci_dev_reset_slot_function(dev: *mut pci_dev, probe: bool) -> i32 {
if !unsafe { (*dev).subordinate }.is_null() || unsafe { (*dev).slot }.is_null() || ((unsafe { (*dev).dev_flags }) as i32) & 64 != 0 || ((unsafe { (*dev).__slate_bits_3.__get_multifunction() }) as i32) != 0 && ((unsafe { (*(*dev).slot).__slate_bits_0.__get_per_func_slot() }) as i32) == 0 {
return 25i32.wrapping_neg();
}
pci_reset_hotplug_slot(unsafe { (*(*dev).slot).hotplug }, probe)
}

fn cxl_port_dvsec(dev: *mut pci_dev) -> u16 {
pci_find_dvsec_capability(dev, 7832u16, 3u16)
}

fn cxl_sbr_masked(dev: *mut pci_dev) -> bool {
let dvsec: u16 = cxl_port_dvsec(dev);
let mut reg: u16 = 0;
let mut rc: i32 = 0;
if dvsec == 0 {
return false;
}
rc = unsafe { pci_read_config_word(dev as *const pci_dev, (dvsec as i32).wrapping_add(12), core::ptr::addr_of_mut!(reg)) };
if rc != 0 || PCI_POSSIBLE_ERROR__2!(reg) {
return false;
}
// Per CXL spec r3.1, sec 8.1.5.2, when "Unmask SBR" is 0, the SBR
// bit in Bridge Control has no effect.  When 1, the Port generates
// hot reset when the SBR bit is set to 1.
if (reg as i32) & 1 != 0 {
return false;
}
true
}

#[unsafe(link_section = ".text.slate_distinct.pci.pci_reset_bus_function")]
extern "C" fn pci_reset_bus_function(dev: *mut pci_dev, probe: bool) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
// If "dev" is below a CXL port that has SBR control masked, SBR
// won't do anything, so return error.
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut rc: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(rc) as *mut i32;
let mut bridge: *mut pci_dev = unsafe { core::mem::zeroed() };
unsafe {
bridge = pci_upstream_bridge(dev);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if !bridge.is_null() {
__v1 = pcie_is_cxl(bridge);
} else {
__v1 = false;
}
if __v1 {
__v0 = cxl_sbr_masked(bridge);
} else {
__v0 = false;
}
if __v0 {
return 25i32.wrapping_neg();
} else {
*__slate_slot_2 = pci_dev_reset_iommu_prepare(dev);
if *__slate_slot_2 != 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"failed to stop IOMMU for a PCI reset: %d\n".as_ptr() as *const u8, *__slate_slot_2);
return *__slate_slot_2;
} else {
*__slate_slot_2 = pci_dev_reset_slot_function(dev, probe);
if *__slate_slot_2 == 25i32.wrapping_neg() {
*__slate_slot_2 = pci_parent_bus_reset(dev, probe);
}
pci_dev_reset_iommu_done(dev);
return *__slate_slot_2;
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.pci.cxl_reset_bus_function")]
extern "C" fn cxl_reset_bus_function(dev: *mut pci_dev, probe: bool) -> i32 {
let mut dvsec: u16 = 0;
let mut reg: u16 = 0;
let mut val: u16 = 0;
let mut rc: i32 = 0;
let bridge: *mut pci_dev = pci_upstream_bridge(dev);
if bridge.is_null() {
return 25i32.wrapping_neg();
}
dvsec = cxl_port_dvsec(bridge);
if dvsec == 0 {
return 25i32.wrapping_neg();
}
if probe {
return 0;
}
rc = unsafe { pci_read_config_word(bridge as *const pci_dev, (dvsec as i32).wrapping_add(12), core::ptr::addr_of_mut!(reg)) };
if rc != 0 {
return 25i32.wrapping_neg();
}
rc = unsafe { pci_dev_reset_iommu_prepare(dev) };
if rc != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"failed to stop IOMMU for a PCI reset: %d\n".as_ptr() as *const u8, rc) };
return rc;
}
if (reg as i32) & 1 != 0 {
val = reg;
} else {
val = ((reg as i32) | 1) as u16;
unsafe { pci_write_config_word(bridge as *const pci_dev, (dvsec as i32).wrapping_add(12), val) };
}
rc = pci_reset_bus_function(dev, probe);
if (reg as i32) != (val as i32) {
unsafe { pci_write_config_word(bridge as *const pci_dev, (dvsec as i32).wrapping_add(12), reg) };
}
unsafe { pci_dev_reset_iommu_done(dev) };
rc
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_dev_lock")]
extern "C" fn pci_dev_lock(dev: *mut pci_dev) {
// block PM suspend, driver probe, etc.
device_lock(unsafe { core::ptr::addr_of_mut!((*dev).dev) });
unsafe { pci_cfg_access_lock(dev) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_dev_lock_812: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_dev_lock)) };

/// Return 1 on successful lock, 0 on contention
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_dev_trylock")]
extern "C" fn pci_dev_trylock(dev: *mut pci_dev) -> i32 {
if device_trylock(unsafe { core::ptr::addr_of_mut!((*dev).dev) }) != 0 {
if unsafe { pci_cfg_access_trylock(dev) } {
return 1;
}
device_unlock(unsafe { core::ptr::addr_of_mut!((*dev).dev) });
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_dev_trylock_813: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_dev_trylock)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_dev_unlock")]
extern "C" fn pci_dev_unlock(dev: *mut pci_dev) {
unsafe { pci_cfg_access_unlock(dev) };
device_unlock(unsafe { core::ptr::addr_of_mut!((*dev).dev) });
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_dev_unlock_814: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_dev_unlock)) };

fn pci_dev_save_and_disable(dev: *mut pci_dev) {
let err_handler: *const pci_error_handlers = if !unsafe { (*dev).driver }.is_null() { unsafe { (*(*dev).driver).err_handler } } else { core::ptr::null::<pci_error_handlers>() };
// dev->driver->err_handler->reset_prepare() is protected against
// races with ->remove() by the device lock, which must be held by
// the caller.
device_lock_assert(unsafe { core::ptr::addr_of_mut!((*dev).dev) });
if !err_handler.is_null() && (unsafe { (*err_handler).reset_prepare }) != None {
unsafe { unsafe { (*err_handler).reset_prepare }.unwrap()(dev) };
} else {
if !unsafe { (*dev).driver }.is_null() {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"resetting".as_ptr() as *const u8) };
}
}
// Wake-up device prior to save.  PM registers default to D0 after
// reset and a simple register restore doesn't reliably return
// to a non-D0 state anyway.
pci_set_power_state(dev, 0i32);
if !pci_dev_config_accessible(dev, c"save state".as_ptr() as *mut u8) {
return;
}
pci_save_state(dev);
// Disable the device by clearing the Command register, except for
// INTx-disable which is set.  This not only disables MMIO and I/O port
// BARs, but also prevents the device from being Bus Master, preventing
// DMA from the device including MSI/MSI-X interrupts.  For PCI 2.3
// compliant devices, INTx-disable prevents legacy interrupts.
unsafe { pci_write_config_word(dev as *const pci_dev, 4i32, 1024u16) };
}

fn pci_dev_restore(dev: *mut pci_dev) {
let err_handler: *const pci_error_handlers = if !unsafe { (*dev).driver }.is_null() { unsafe { (*(*dev).driver).err_handler } } else { core::ptr::null::<pci_error_handlers>() };
pci_restore_state(dev);
// dev->driver->err_handler->reset_done() is protected against
// races with ->remove() by the device lock, which must be held by
// the caller.
if !err_handler.is_null() && (unsafe { (*err_handler).reset_done }) != None {
unsafe { unsafe { (*err_handler).reset_done }.unwrap()(dev) };
} else {
if !unsafe { (*dev).driver }.is_null() {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"reset done".as_ptr() as *const u8) };
}
}
}

// dev->reset_methods[] is a 0-terminated list of indices into this array
/// __pci_reset_function_locked - reset a PCI device function while holding
/// the @dev mutex lock.
/// @dev: PCI device to reset
///
/// Some devices allow an individual function to be reset without affecting
/// other functions in the same device.  The PCI device must be responsive
/// to PCI config space in order to use this function.
///
/// The device function is presumed to be unused and the caller is holding
/// the device mutex lock when this function is called.
///
/// Resetting the device will make the contents of PCI configuration space
/// random, so any caller of this must be prepared to reinitialise the
/// device including MSI, bus mastering, BARs, decoding IO and memory spaces,
/// etc.
///
/// Context: The caller must hold the device lock.
///
/// Return: 0 if the device function was successfully reset or negative if the
/// device doesn't support resetting a single function.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.__pci_reset_function_locked")]
extern "C" fn __pci_reset_function_locked(dev: *mut pci_dev) -> i32 {
let mut m: i32 = 0;
let mut rc: i32 = 0;
might_sleep!();
device_lock_assert(unsafe { core::ptr::addr_of_mut!((*dev).dev) });
// A reset method returns -ENOTTY if it doesn't support this device and
// we should try the next method.
//
// If it returns 0 (success), we're finished.  If it returns any other
// error, we're also finished: this indicates that further reset
// mechanisms might be broken on the device.
for i in 0i32..8i32 {
m = (unsafe { *((*dev).reset_methods.as_mut_ptr() as *mut u8).wrapping_offset(i as isize) }) as i32;
if m == 0 {
return 25i32.wrapping_neg();
}
let method: *const pci_reset_fn_method = unsafe { core::ptr::addr_of!(pci_reset_fn_methods.0) as *const pci_reset_fn_method }.wrapping_offset(m as isize);
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"reset via %s\n".as_ptr() as *const u8, (*method).name) };
}
rc = unsafe { unsafe { (*method).reset_fn }.unwrap()(dev, false) };
if rc == 0 {
return 0;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s failed with %d\n".as_ptr() as *const u8, (*method).name, rc) };
}
if rc != 25i32.wrapping_neg() {
return rc;
}
}
25i32.wrapping_neg()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___pci_reset_function_locked_815: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(__pci_reset_function_locked)) };

// pci_init_reset_methods - check whether device can be safely reset
// and store supported reset mechanisms.
// @dev: PCI device to check for reset mechanisms
//
// Some devices allow an individual function to be reset without affecting
// other functions in the same device.  The PCI device must be in D0-D3hot
// state.
//
// Stores reset mechanisms supported by device in reset_methods byte array
// which is a member of struct pci_dev.
/// pci_reset_function - quiesce and reset a PCI device function
/// @dev: PCI device to reset
///
/// Some devices allow an individual function to be reset without affecting
/// other functions in the same device.  The PCI device must be responsive
/// to PCI config space in order to use this function.
///
/// This function does not just reset the PCI portion of a device, but
/// clears all the state associated with the device.  This function differs
/// from __pci_reset_function_locked() in that it saves and restores device state
/// over the reset and takes the PCI device lock.
///
/// Returns 0 if the device function was successfully reset or negative if the
/// device doesn't support resetting a single function.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_reset_function")]
extern "C" fn pci_reset_function(dev: *mut pci_dev) -> i32 {
let mut rc: i32 = 0;
if !pci_reset_supported(dev) {
return 25i32.wrapping_neg();
}
// If there's no upstream bridge, no locking is needed since there is
// no upstream bridge configuration to hold consistent.
let bridge: *mut pci_dev = pci_upstream_bridge(dev);
if !bridge.is_null() {
pci_dev_lock(bridge);
}
pci_dev_lock(dev);
pci_dev_save_and_disable(dev);
rc = __pci_reset_function_locked(dev);
pci_dev_restore(dev);
pci_dev_unlock(dev);
if !bridge.is_null() {
pci_dev_unlock(bridge);
}
rc
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_reset_function_817: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_reset_function)) };

/// pci_reset_function_locked - quiesce and reset a PCI device function
/// @dev: PCI device to reset
///
/// Some devices allow an individual function to be reset without affecting
/// other functions in the same device.  The PCI device must be responsive
/// to PCI config space in order to use this function.
///
/// This function does not just reset the PCI portion of a device, but
/// clears all the state associated with the device.  This function differs
/// from __pci_reset_function_locked() in that it saves and restores device state
/// over the reset.  It also differs from pci_reset_function() in that it
/// requires the PCI device lock to be held.
///
/// Context: The caller must hold the device lock.
///
/// Return: 0 if the device function was successfully reset or negative if the
/// device doesn't support resetting a single function.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_reset_function_locked")]
extern "C" fn pci_reset_function_locked(dev: *mut pci_dev) -> i32 {
let mut rc: i32 = 0;
device_lock_assert(unsafe { core::ptr::addr_of_mut!((*dev).dev) });
if !pci_reset_supported(dev) {
return 25i32.wrapping_neg();
}
pci_dev_save_and_disable(dev);
rc = __pci_reset_function_locked(dev);
pci_dev_restore(dev);
rc
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_reset_function_locked_818: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_reset_function_locked)) };

/// pci_try_reset_function - quiesce and reset a PCI device function
/// @dev: PCI device to reset
///
/// Same as above, except return -EAGAIN if unable to lock device.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_try_reset_function")]
extern "C" fn pci_try_reset_function(dev: *mut pci_dev) -> i32 {
let mut rc: i32 = 0;
if !pci_reset_supported(dev) {
return 25i32.wrapping_neg();
}
if pci_dev_trylock(dev) == 0 {
return 11i32.wrapping_neg();
}
pci_dev_save_and_disable(dev);
rc = __pci_reset_function_locked(dev);
pci_dev_restore(dev);
pci_dev_unlock(dev);
rc
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_try_reset_function_819: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_try_reset_function)) };

/// Do any devices on or below this bus prevent a bus reset?
fn pci_bus_resettable(bus: *mut pci_bus) -> bool {
let mut dev: *mut pci_dev = unsafe { core::mem::zeroed() };
if !unsafe { (*bus).self_ }.is_null() && ((unsafe { (*(*bus).self_).dev_flags }) as i32) & 64 != 0 {
return false;
}
dev = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*bus).devices) });
while list_entry_is_head!(dev, unsafe { core::ptr::addr_of_mut!((*bus).devices) }, bus_list) == 0 {
let __v1: bool = ((unsafe { (*dev).dev_flags }) as i32) & 64 != 0 || !unsafe { (*dev).subordinate }.is_null() && !pci_bus_resettable(unsafe { (*dev).subordinate });
if __v1 {
return false;
}
dev = list_next_entry!(dev, bus_list);
}
true
}

/// Lock devices from the top of the tree down
fn __pci_bus_lock(bus: *mut pci_bus, slot: *mut pci_slot) {
let mut dev: *mut pci_dev = unsafe { core::mem::zeroed() };
let bridge: *mut pci_dev = unsafe { (*bus).self_ };
if !bridge.is_null() {
pci_dev_lock(bridge);
}
dev = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*bus).devices) });
while list_entry_is_head!(dev, unsafe { core::ptr::addr_of_mut!((*bus).devices) }, bus_list) == 0 {
if !(!slot.is_null() && (unsafe { (*dev).slot }.is_null() || (unsafe { (*dev).slot }) != slot)) {
if !unsafe { (*dev).subordinate }.is_null() {
pci_bus_lock(unsafe { (*dev).subordinate });
} else {
pci_dev_lock(dev);
}
}
dev = list_next_entry!(dev, bus_list);
}
}

/// Unlock devices from the bottom of the tree up
fn __pci_bus_unlock(bus: *mut pci_bus, slot: *mut pci_slot) {
let mut dev: *mut pci_dev = unsafe { core::mem::zeroed() };
let bridge: *mut pci_dev = unsafe { (*bus).self_ };
dev = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*bus).devices) });
while list_entry_is_head!(dev, unsafe { core::ptr::addr_of_mut!((*bus).devices) }, bus_list) == 0 {
if !(!slot.is_null() && (unsafe { (*dev).slot }.is_null() || (unsafe { (*dev).slot }) != slot)) {
if !unsafe { (*dev).subordinate }.is_null() {
pci_bus_unlock(unsafe { (*dev).subordinate });
} else {
pci_dev_unlock(dev);
}
}
dev = list_next_entry!(dev, bus_list);
}
if !bridge.is_null() {
pci_dev_unlock(bridge);
}
}

/// Return 1 on successful lock, 0 on contention
fn __pci_bus_trylock(bus: *mut pci_bus, slot: *mut pci_slot) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut bridge: *mut pci_dev = unsafe { core::mem::zeroed() };
let mut dev: core::mem::MaybeUninit<*mut pci_dev> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut pci_dev = core::ptr::addr_of_mut!(dev) as *mut *mut pci_dev;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
bridge = (*bus).self_;
if !bridge.is_null() {
__v0 = pci_dev_trylock(bridge) == 0;
} else {
__v0 = false;
}
if __v0 {
return 0;
} else {
*__slate_slot_2 = list_first_entry!(core::ptr::addr_of_mut!((*bus).devices));
'__join_9: {
while list_entry_is_head!(*__slate_slot_2, core::ptr::addr_of_mut!((*bus).devices), bus_list) == 0 {
if !(!slot.is_null() && (unsafe { (*(*__slate_slot_2)).slot }.is_null() || (*(*__slate_slot_2)).slot != slot)) {
if !unsafe { (*(*__slate_slot_2)).subordinate }.is_null() {
if pci_bus_trylock((*(*__slate_slot_2)).subordinate) == 0 {
break '__join_9;
}
} else {
if pci_dev_trylock(*__slate_slot_2) == 0 {
break '__join_9;
}
}
}
*__slate_slot_2 = list_next_entry!(*__slate_slot_2, bus_list);
}
return 1;
}
*__slate_slot_2 = list_prev_entry!(*__slate_slot_2, bus_list);
while list_entry_is_head!(*__slate_slot_2, core::ptr::addr_of_mut!((*bus).devices), bus_list) == 0 {
if !(!slot.is_null() && (unsafe { (*(*__slate_slot_2)).slot }.is_null() || (*(*__slate_slot_2)).slot != slot)) {
if !unsafe { (*(*__slate_slot_2)).subordinate }.is_null() {
pci_bus_unlock((*(*__slate_slot_2)).subordinate);
} else {
pci_dev_unlock(*__slate_slot_2);
}
}
*__slate_slot_2 = list_prev_entry!(*__slate_slot_2, bus_list);
}
if !bridge.is_null() {
pci_dev_unlock(bridge);
}
return 0;
}
}
}

/// Lock devices from the top of the tree down
fn pci_bus_lock(bus: *mut pci_bus) {
__pci_bus_lock(bus, core::ptr::null_mut::<pci_slot>());
}

/// Unlock devices from the bottom of the tree up
fn pci_bus_unlock(bus: *mut pci_bus) {
__pci_bus_unlock(bus, core::ptr::null_mut::<pci_slot>());
}

/// Return 1 on successful lock, 0 on contention
fn pci_bus_trylock(bus: *mut pci_bus) -> i32 {
__pci_bus_trylock(bus, core::ptr::null_mut::<pci_slot>())
}

/// Do any devices on or below this slot prevent a bus reset?
fn pci_slot_resettable(slot: *mut pci_slot) -> bool {
let mut dev: *mut pci_dev = unsafe { core::mem::zeroed() };
let bridge: *mut pci_dev = unsafe { (*(*slot).bus).self_ };
if !bridge.is_null() && ((unsafe { (*bridge).dev_flags }) as i32) & 64 != 0 {
return false;
}
dev = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*(*slot).bus).devices) });
while list_entry_is_head!(dev, unsafe { core::ptr::addr_of_mut!((*(*slot).bus).devices) }, bus_list) == 0 {
if !(unsafe { (*dev).slot }.is_null() || (unsafe { (*dev).slot }) != slot) {
let __v2: bool = ((unsafe { (*dev).dev_flags }) as i32) & 64 != 0 || !unsafe { (*dev).subordinate }.is_null() && !pci_bus_resettable(unsafe { (*dev).subordinate });
if __v2 {
return false;
}
}
dev = list_next_entry!(dev, bus_list);
}
true
}

/// Lock devices from the top of the tree down
fn pci_slot_lock(slot: *mut pci_slot) {
__pci_bus_lock(unsafe { (*slot).bus }, slot);
}

/// Unlock devices from the bottom of the tree up
fn pci_slot_unlock(slot: *mut pci_slot) {
__pci_bus_unlock(unsafe { (*slot).bus }, slot);
}

/// Return 1 on successful lock, 0 on contention
fn pci_slot_trylock(slot: *mut pci_slot) -> i32 {
__pci_bus_trylock(unsafe { (*slot).bus }, slot)
}

/// Save and disable devices from the top of the tree down while holding
/// the @dev mutex lock for the entire tree.
fn pci_bus_save_and_disable_locked(bus: *mut pci_bus) {
let mut dev: *mut pci_dev = unsafe { core::mem::zeroed() };
dev = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*bus).devices) });
while list_entry_is_head!(dev, unsafe { core::ptr::addr_of_mut!((*bus).devices) }, bus_list) == 0 {
pci_dev_save_and_disable(dev);
if !unsafe { (*dev).subordinate }.is_null() {
pci_bus_save_and_disable_locked(unsafe { (*dev).subordinate });
}
dev = list_next_entry!(dev, bus_list);
}
}

/// Restore devices from top of the tree down while holding @dev mutex lock
/// for the entire tree.  Parent bridges need to be restored before we can
/// get to subordinate devices.
fn pci_bus_restore_locked(bus: *mut pci_bus) {
let mut dev: *mut pci_dev = unsafe { core::mem::zeroed() };
dev = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*bus).devices) });
while list_entry_is_head!(dev, unsafe { core::ptr::addr_of_mut!((*bus).devices) }, bus_list) == 0 {
pci_dev_restore(dev);
if !unsafe { (*dev).subordinate }.is_null() {
pci_bridge_wait_for_secondary_bus(dev, c"bus reset".as_ptr() as *mut u8);
pci_bus_restore_locked(unsafe { (*dev).subordinate });
}
dev = list_next_entry!(dev, bus_list);
}
}

/// Save and disable devices from the top of the tree down while holding
/// the @dev mutex lock for the entire tree.
fn pci_slot_save_and_disable_locked(slot: *mut pci_slot) {
let mut dev: *mut pci_dev = unsafe { core::mem::zeroed() };
dev = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*(*slot).bus).devices) });
while list_entry_is_head!(dev, unsafe { core::ptr::addr_of_mut!((*(*slot).bus).devices) }, bus_list) == 0 {
if !(unsafe { (*dev).slot }.is_null() || (unsafe { (*dev).slot }) != slot) {
pci_dev_save_and_disable(dev);
if !unsafe { (*dev).subordinate }.is_null() {
pci_bus_save_and_disable_locked(unsafe { (*dev).subordinate });
}
}
dev = list_next_entry!(dev, bus_list);
}
}

/// Restore devices from top of the tree down while holding @dev mutex lock
/// for the entire tree.  Parent bridges need to be restored before we can
/// get to subordinate devices.
fn pci_slot_restore_locked(slot: *mut pci_slot) {
let mut dev: *mut pci_dev = unsafe { core::mem::zeroed() };
dev = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*(*slot).bus).devices) });
while list_entry_is_head!(dev, unsafe { core::ptr::addr_of_mut!((*(*slot).bus).devices) }, bus_list) == 0 {
if !(unsafe { (*dev).slot }.is_null() || (unsafe { (*dev).slot }) != slot) {
pci_dev_restore(dev);
if !unsafe { (*dev).subordinate }.is_null() {
pci_bridge_wait_for_secondary_bus(dev, c"slot reset".as_ptr() as *mut u8);
pci_bus_restore_locked(unsafe { (*dev).subordinate });
}
}
dev = list_next_entry!(dev, bus_list);
}
}

fn pci_slot_reset(slot: *mut pci_slot, probe: bool) -> i32 {
let mut rc: i32 = 0;
let __v0: bool = slot.is_null() || !pci_slot_resettable(slot);
if __v0 {
return 25i32.wrapping_neg();
}
if !probe {
pci_slot_lock(slot);
}
might_sleep!();
rc = pci_reset_hotplug_slot(unsafe { (*slot).hotplug }, probe);
if !probe {
pci_slot_unlock(slot);
}
rc
}

/// pci_probe_reset_slot - probe whether a PCI slot can be reset
/// @slot: PCI slot to probe
///
/// Return 0 if slot can be reset, negative if a slot reset is not supported.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_probe_reset_slot")]
extern "C" fn pci_probe_reset_slot(slot: *mut pci_slot) -> i32 {
pci_slot_reset(slot, true)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_probe_reset_slot_820: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_slot) -> i32>, *mut ()>(Some(pci_probe_reset_slot)) };

/// pci_try_reset_slot - Try to reset a PCI slot
/// @slot: PCI slot to reset
///
/// A PCI bus may host multiple slots, each slot may support a reset mechanism
/// independent of other slots.  For instance, some slots may support slot power
/// control.  In the case of a 1:1 bus to slot architecture, this function may
/// wrap the bus reset to avoid spurious slot related events such as hotplug.
/// Generally a slot reset should be attempted before a bus reset.  All of the
/// function of the slot and any subordinate buses behind the slot are reset
/// through this function.  PCI config space of all devices in the slot and
/// behind the slot is saved before and restored after reset.
///
/// Same as above except return -EAGAIN if the slot cannot be locked
fn pci_try_reset_slot(slot: *mut pci_slot) -> i32 {
let mut rc: i32 = pci_slot_reset(slot, true);
if rc != 0 {
return rc;
}
if pci_slot_trylock(slot) != 0 {
pci_slot_save_and_disable_locked(slot);
might_sleep!();
rc = pci_reset_hotplug_slot(unsafe { (*slot).hotplug }, false);
pci_slot_restore_locked(slot);
pci_slot_unlock(slot);
} else {
rc = 11i32.wrapping_neg();
}
rc
}

fn pci_bus_reset(bus: *mut pci_bus, probe: bool) -> i32 {
let mut ret: i32 = 0;
let __v0: bool = unsafe { (*bus).self_ }.is_null() || !pci_bus_resettable(bus);
if __v0 {
return 25i32.wrapping_neg();
}
if probe {
return 0;
}
pci_bus_lock(bus);
might_sleep!();
ret = pci_bridge_secondary_bus_reset(unsafe { (*bus).self_ });
pci_bus_unlock(bus);
ret
}

/// pci_try_reset_bus - Try to reset a PCI bus
/// @bus: top level PCI bus to reset
///
/// Same as above except return -EAGAIN if the bus cannot be locked
fn pci_try_reset_bus(bus: *mut pci_bus) -> i32 {
let mut rc: i32 = pci_bus_reset(bus, true);
if rc != 0 {
return rc;
}
if pci_bus_trylock(bus) != 0 {
pci_bus_save_and_disable_locked(bus);
might_sleep!();
rc = pci_bridge_secondary_bus_reset(unsafe { (*bus).self_ });
pci_bus_restore_locked(bus);
pci_bus_unlock(bus);
} else {
rc = 11i32.wrapping_neg();
}
rc
}

/// pci_reset_bridge - reset a bridge's subordinate bus
/// @bridge: bridge that connects to the bus to reset
/// @restore: when true use a reset method that invokes pci_dev_restore() post
///           reset for affected devices
///
/// This function will first try to reset the slots on this bus if the method is
/// available. If slot reset is not available, this will fall back to a
/// secondary bus reset.
fn pci_reset_bridge(bridge: *mut pci_dev, restore: bool) -> i32 {
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut slot: core::mem::MaybeUninit<*mut pci_slot> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut pci_slot = core::ptr::addr_of_mut!(slot) as *mut *mut pci_slot;
let mut bus: *mut pci_bus = unsafe { core::mem::zeroed() };
unsafe {
bus = (*bridge).subordinate;
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 0;
if bus.is_null() {
return 25i32.wrapping_neg();
} else {
'__join_2: {
mutex_lock(core::ptr::addr_of_mut!(pci_slot_mutex));
if list_empty(core::ptr::addr_of_mut!((*bus).slots) as *const list_head) == 0 {
*__slate_slot_1 = list_first_entry__6!(core::ptr::addr_of_mut!((*bus).slots));
while list_entry_is_head!(*__slate_slot_1, core::ptr::addr_of_mut!((*bus).slots), list) == 0 {
if pci_probe_reset_slot(*__slate_slot_1) != 0 {
break '__join_2;
} else {
*__slate_slot_1 = list_next_entry__7!(*__slate_slot_1, list);
}
}
*__slate_slot_1 = list_first_entry__6!(core::ptr::addr_of_mut!((*bus).slots));
while list_entry_is_head!(*__slate_slot_1, core::ptr::addr_of_mut!((*bus).slots), list) == 0 {
if restore {
ret = pci_try_reset_slot(*__slate_slot_1);
} else {
ret = pci_slot_reset(*__slate_slot_1, false);
}
if ret != 0 {
break;
} else {
*__slate_slot_1 = list_next_entry__7!(*__slate_slot_1, list);
}
}
mutex_unlock(core::ptr::addr_of_mut!(pci_slot_mutex));
return ret;
}
}
mutex_unlock(core::ptr::addr_of_mut!(pci_slot_mutex));
if restore {
return pci_try_reset_bus(bus);
} else {
return pci_bus_reset((*bridge).subordinate, false);
}
}
}
}

/// pci_bus_error_reset - reset the bridge's subordinate bus
/// @bridge: The parent device that connects to the bus to reset
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_bus_error_reset")]
extern "C" fn pci_bus_error_reset(bridge: *mut pci_dev) -> i32 {
pci_reset_bridge(bridge, false)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_bus_error_reset_821: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_bus_error_reset)) };

#[unsafe(no_mangle)]
extern "C" fn pci_try_reset_bridge(bridge: *mut pci_dev) -> i32 {
pci_reset_bridge(bridge, true)
}

/// pci_probe_reset_bus - probe whether a PCI bus can be reset
/// @bus: PCI bus to probe
///
/// Return 0 if bus can be reset, negative if a bus reset is not supported.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_probe_reset_bus")]
extern "C" fn pci_probe_reset_bus(bus: *mut pci_bus) -> i32 {
pci_bus_reset(bus, true)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_probe_reset_bus_822: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_bus) -> i32>, *mut ()>(Some(pci_probe_reset_bus)) };

/// pci_reset_bus - Try to reset a PCI bus
/// @pdev: top level PCI device to reset via slot/bus
///
/// Same as above except return -EAGAIN if the bus cannot be locked
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_reset_bus")]
extern "C" fn pci_reset_bus(pdev: *mut pci_dev) -> i32 {
let __v0: i32 = if pci_probe_reset_slot(unsafe { (*pdev).slot }) == 0 {
pci_try_reset_slot(unsafe { (*pdev).slot })
} else {
pci_try_reset_bus(unsafe { (*pdev).bus })
};
__v0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_reset_bus_823: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pci_reset_bus)) };

// pcix_get_max_mmrbc - get PCI-X maximum designed memory read byte count
// @dev: PCI device to query
//
// Returns mmrbc: maximum designed memory read count in bytes or
// appropriate error value.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcix_get_max_mmrbc_829: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pcix_get_max_mmrbc)) };

// pcix_get_mmrbc - get PCI-X maximum memory read byte count
// @dev: PCI device to query
//
// Returns mmrbc: maximum memory read count in bytes or appropriate error
// value.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcix_get_mmrbc_835: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pcix_get_mmrbc)) };

// pcix_set_mmrbc - set PCI-X maximum memory read byte count
// @dev: PCI device to query
// @mmrbc: maximum memory read count in bytes
//    valid values are 512, 1024, 2048, 4096
//
// If possible sets maximum memory read byte count, some bridges have errata
// that prevent this.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcix_set_mmrbc_851: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>, *mut ()>(Some(pcix_set_mmrbc)) };

// pcie_get_readrq - get PCI Express read request size
// @dev: PCI device to query
//
// Returns maximum memory read request in bytes or appropriate error value.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_get_readrq_857: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pcie_get_readrq)) };

// pcie_set_readrq - set PCI Express maximum memory read request
// @dev: PCI device to query
// @rq: maximum memory read count in bytes
//    valid values are 128, 256, 512, 1024, 2048, 4096
//
// If possible sets maximum memory read request in bytes
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_set_readrq_863: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>, *mut ()>(Some(pcie_set_readrq)) };

// pcie_get_mps - get PCI Express maximum payload size
// @dev: PCI device to query
//
// Returns maximum payload size in bytes
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_get_mps_869: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pcie_get_mps)) };

// pcie_set_mps - set PCI Express maximum payload size
// @dev: PCI device to query
// @mps: maximum payload size in bytes
//    valid values are 128, 256, 512, 1024, 2048, 4096
//
// If possible sets maximum payload size
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_set_mps_875: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>, *mut ()>(Some(pcie_set_mps)) };

fn to_pcie_link_speed(lnksta: u16) -> u32 {
let mut __v0: i32;
loop {
if __bf_cast_unsigned!(15i32) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u32) {
unsafe { __compiletime_assert_876() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: i32;
loop {
if false {
unsafe { __compiletime_assert_877() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if false {
unsafe { __compiletime_assert_878() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!(15i32 >> (15u64.trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_879() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 15u64.wrapping_add(1u64 << (15u64.trailing_zeros() as i32)) & 15u64.wrapping_add(1u64 << (15u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_880() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((lnksta as i32) & 15) >> (15u64.trailing_zeros() as i32);
__v0 = __v2;
(unsafe { *(core::ptr::addr_of!(pcie_link_speed) as *const u8).wrapping_offset(__v0 as isize) }) as u32
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pcie_link_speed_mbps")]
extern "C" fn pcie_link_speed_mbps(pdev: *mut pci_dev) -> i32 {
let mut lnksta: u16 = 0;
let err: i32 = unsafe { pcie_capability_read_word(pdev, 18i32, core::ptr::addr_of_mut!(lnksta)) };
if err != 0 {
return err;
}
pcie_dev_speed_mbps(to_pcie_link_speed(lnksta))
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_link_speed_mbps_881: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(pcie_link_speed_mbps)) };

// pcie_bandwidth_available - determine minimum link settings of a PCIe
// 			      device and its bandwidth limitation
// @dev: PCI device to query
// @limiting_dev: storage for device causing the bandwidth limitation
// @speed: storage for speed of limiting device
// @width: storage for width of limiting device
//
// Walk up the PCI device chain and find the point where the minimum
// bandwidth is available.  Return the bandwidth available there and (if
// limiting_dev, speed, and width pointers are supplied) information about
// that point.  The bandwidth returned is in Mb/s, i.e., megabits/second of
// raw bandwidth.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_bandwidth_available_887: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, *mut *mut pci_dev, *mut u32, *mut u32) -> u32>, *mut ()>(Some(pcie_bandwidth_available)) };

/// pcie_get_supported_speeds - query Supported Link Speed Vector
/// @dev: PCI device to query
///
/// Query @dev supported link speeds.
///
/// Implementation Note in PCIe r6.0 sec 7.5.3.18 recommends determining
/// supported link speeds using the Supported Link Speeds Vector in the Link
/// Capabilities 2 Register (when available).
///
/// Link Capabilities 2 was added in PCIe r3.0, sec 7.8.18.
///
/// Without Link Capabilities 2, i.e., prior to PCIe r3.0, Supported Link
/// Speeds field in Link Capabilities is used and only 2.5 GT/s and 5.0 GT/s
/// speeds were defined.
///
/// For @dev without Supported Link Speed Vector, the field is synthesized
/// from the Max Link Speed field in the Link Capabilities Register.
///
/// Return: Supported Link Speeds Vector (+ reserved 0 at LSB).
#[unsafe(no_mangle)]
extern "C" fn pcie_get_supported_speeds(dev: *mut pci_dev) -> u8 {
let mut lnkcap2: u32 = 0;
let mut lnkcap: u32 = 0;
let mut speeds: u8 = 0;
// Speeds retain the reserved 0 at LSB before PCIe Supported Link
// Speeds Vector to allow using SLS Vector bit defines directly.
unsafe { pcie_capability_read_dword(dev, 44i32, core::ptr::addr_of_mut!(lnkcap2)) };
speeds = (lnkcap2 & 254) as u8;
// Ignore speeds higher than Max Link Speed
unsafe { pcie_capability_read_dword(dev, 12i32, core::ptr::addr_of_mut!(lnkcap)) };
let __v0: u8 = speeds;
speeds = (((__v0 as i32) as u64) & GENMASK__4!(lnkcap & 15, 0i32)) as u8;
// PCIe r3.0-compliant
if speeds != 0 {
return speeds;
}
// Synthesize from the Max Link Speed field
if lnkcap & 15 == 2 {
speeds = 6u8;
} else {
if lnkcap & 15 == 1 {
speeds = 2;
}
}
speeds
}

/// pcie_get_speed_cap - query for the PCI device's link speed capability
/// @dev: PCI device to query
///
/// Query the PCI device speed capability.
///
/// Return: the maximum link speed supported by the device.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pcie_get_speed_cap")]
extern "C" fn pcie_get_speed_cap(dev: *mut pci_dev) -> u32 {
PCIE_LNKCAP2_SLS2SPEED!(unsafe { (*dev).supported_speeds }) as u32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_get_speed_cap_888: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> u32>, *mut ()>(Some(pcie_get_speed_cap)) };

// pcie_get_width_cap - query for the PCI device's link width capability
// @dev: PCI device to query
//
// Query the PCI device width capability.  Return the maximum link width
// supported by the device.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_get_width_cap_894: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> u32>, *mut ()>(Some(pcie_get_width_cap)) };

/// pcie_bandwidth_capable - calculate a PCI device's link bandwidth capability
/// @dev: PCI device
/// @speed: storage for link speed
/// @width: storage for link width
///
/// Calculate a PCI device's link bandwidth by querying for its link speed
/// and width, multiplying them, and applying encoding overhead.  The result
/// is in Mb/s, i.e., megabits/second of raw bandwidth.
fn pcie_bandwidth_capable(dev: *mut pci_dev, speed: *mut u32, width: &mut u32) -> u32 {
unsafe {
*speed = pcie_get_speed_cap(dev);
}
unsafe {
*(width as *mut u32) = pcie_get_width_cap(dev);
}
if ((unsafe { *speed }) as u32) == 255 || ((unsafe { *(width as *mut u32) }) as u32) == 255 {
return 0;
}
((unsafe { *(width as *mut u32) }) as u32).wrapping_mul(PCIE_SPEED2MBS_ENC!(unsafe { *speed }) as u32)
}

/// __pcie_print_link_status - Report the PCI device's link speed and width
/// @dev: PCI device to query
/// @verbose: Print info even when enough bandwidth is available
///
/// If the available bandwidth at the device is less than the device is
/// capable of, report the device's maximum possible bandwidth and the
/// upstream link that limits its performance.  If @verbose, always print
/// the available bandwidth, even if the device isn't constrained.
#[unsafe(no_mangle)]
extern "C" fn __pcie_print_link_status(dev: *mut pci_dev, verbose: bool) {
let mut width: u32 = 0;
let mut width_cap: u32 = 0;
let mut speed: u32 = 0;
let mut speed_cap: u32 = 0;
let mut limiting_dev: *mut pci_dev = core::ptr::null_mut::<pci_dev>();
let mut bw_avail: u32 = 0;
let mut bw_cap: u32 = 0;
let mut flit_mode: *mut u8 = c"".as_ptr() as *mut u8;
bw_cap = pcie_bandwidth_capable(dev, core::ptr::addr_of_mut!(speed_cap), unsafe { &mut (*core::ptr::addr_of_mut!(width_cap)) });
bw_avail = pcie_bandwidth_available(dev, core::ptr::addr_of_mut!(limiting_dev), core::ptr::addr_of_mut!(speed), core::ptr::addr_of_mut!(width));
if !unsafe { (*dev).bus }.is_null() && ((unsafe { (*(*dev).bus).__slate_bits_0.__get_flit_mode() }) as i32) != 0 {
flit_mode = c", in Flit mode".as_ptr() as *mut u8;
}
if bw_avail >= bw_cap && verbose {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%u.%03u Gb/s available PCIe bandwidth (%s x%d link)%s\n".as_ptr() as *const u8, bw_cap / 1000, bw_cap % 1000, pci_speed_string(speed_cap), width_cap as u32, flit_mode) };
} else {
if bw_avail < bw_cap {
while false {
}
let __v2: *const u8 = if !limiting_dev.is_null() {
pci_name(limiting_dev as *const pci_dev)
} else {
c"<unknown>".as_ptr() as *const u8
};
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%u.%03u Gb/s available PCIe bandwidth, limited by %s x%d link at %s (capable of %u.%03u Gb/s with %s x%d link)%s\n".as_ptr() as *const u8, bw_avail / 1000, bw_avail % 1000, pci_speed_string(speed), width as u32, __v2, bw_cap / 1000, bw_cap % 1000, pci_speed_string(speed_cap), width_cap as u32, flit_mode) };
}
}
}

/// pcie_print_link_status - Report the PCI device's link speed and width
/// @dev: PCI device to query
///
/// Report the available bandwidth at the device.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pcie_print_link_status")]
extern "C" fn pcie_print_link_status(dev: *mut pci_dev) {
__pcie_print_link_status(dev, true);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pcie_print_link_status_895: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pcie_print_link_status)) };

/// pci_select_bars - Make BAR mask from the type of resource
/// @dev: the PCI device for which BAR mask is made
/// @flags: resource type mask to be selected
///
/// This helper routine makes bar mask from the type of resource.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_select_bars")]
extern "C" fn pci_select_bars(dev: *mut pci_dev, flags: u64) -> i32 {
let mut bars: i32 = 0;
for i in 0i32..11i32 {
if pci_resource_flags!(dev, i) & flags != 0 {
bars |= 1i32 << i;
}
}
bars
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_select_bars_896: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, u64) -> i32>, *mut ()>(Some(pci_select_bars)) };

/// Some architectures require additional programming to enable VGA
static mut arch_set_vga_state: Option<unsafe extern "C" fn(*mut pci_dev, bool, u32, u32) -> i32> = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn pci_register_set_vga_state(func: Option<unsafe extern "C" fn(*mut pci_dev, bool, u32, u32) -> i32>) {
unsafe {
arch_set_vga_state = func;
}
// NULL disables
}

fn pci_set_vga_state_arch(dev: *mut pci_dev, decode: bool, command_bits: u32, flags: u32) -> i32 {
if (unsafe { arch_set_vga_state }) != None {
return unsafe { unsafe { arch_set_vga_state }.unwrap()(dev, decode, command_bits, flags) };
}
0
}

/// pci_set_vga_state - set VGA decode state on device and parents if requested
/// @dev: the PCI device
/// @decode: true = enable decoding, false = disable decoding
/// @command_bits: PCI_COMMAND_IO and/or PCI_COMMAND_MEMORY
/// @flags: traverse ancestors and change bridges
/// CHANGE_BRIDGE_ONLY / CHANGE_BRIDGE
#[unsafe(no_mangle)]
extern "C" fn pci_set_vga_state(dev: *mut pci_dev, decode: bool, command_bits: u32, flags: u32) -> i32 {
let mut bus: *mut pci_bus = unsafe { core::mem::zeroed() };
let mut cmd: u16 = 0;
let mut rc: i32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = (flags & 2u32 != 0 && command_bits & 4294967292u32 != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("897: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 897b - ., 3; .popsection\n# {0}", const 897 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pci_set_vga_state_1, sym __slate_asm_string_pci_set_vga_state_2, const 6271 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("898: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 898b - ., 4; .popsection\n# {0}", const 898 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
// ARCH specific VGA enables
rc = pci_set_vga_state_arch(dev, decode, command_bits, flags);
if rc != 0 {
return rc;
}
if flags & 2u32 != 0 {
unsafe { pci_read_config_word(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(cmd)) };
if decode {
cmd = (((cmd as i32) as u32) | command_bits) as u16;
} else {
let __v8: u16 = (((cmd as i32) as u32) & !command_bits) as u16;
cmd = __v8;
}
unsafe { pci_write_config_word(dev as *const pci_dev, 4i32, cmd) };
}
if flags & 1u32 == 0 {
return 0;
}
bus = unsafe { (*dev).bus };
while !bus.is_null() {
let bridge: *mut pci_dev = unsafe { (*bus).self_ };
if !bridge.is_null() {
unsafe { pci_read_config_word(bridge as *const pci_dev, 62i32, core::ptr::addr_of_mut!(cmd)) };
if decode {
cmd = ((cmd as i32) | 8) as u16;
} else {
let __v12: u16 = ((cmd as i32) & !8) as u16;
cmd = __v12;
}
unsafe { pci_write_config_word(bridge as *const pci_dev, 62i32, cmd) };
// VGA Enable may not be writable if bridge doesn't
// support it.
if decode {
unsafe { pci_read_config_word(bridge as *const pci_dev, 62i32, core::ptr::addr_of_mut!(cmd)) };
if (cmd as i32) & 8 == 0 {
return 5i32.wrapping_neg();
}
}
}
bus = unsafe { (*bus).parent };
}
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_pr3_present")]
extern "C" fn pci_pr3_present(pdev: *mut pci_dev) -> bool {
if (unsafe { acpi_disabled }) != 0 {
return false;
}
let adev: *mut acpi_device = ACPI_COMPANION!(unsafe { core::ptr::addr_of_mut!((*pdev).dev) });
if adev.is_null() {
return false;
}
((unsafe { (*adev).power.flags.__slate_bits_0.__get_power_resources() }) as i32) != 0 && unsafe { acpi_has_method((*adev).handle, c"_PR3".as_ptr() as *mut u8) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_pr3_present_899: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> bool>, *mut ()>(Some(pci_pr3_present)) };

// pci_add_dma_alias - Add a DMA devfn alias for a device
// @dev: the PCI device for which alias is added
// @devfn_from: alias slot and function
// @nr_devfns: number of subsequent devfns to alias
//
// This helper encodes an 8-bit devfn as a bit number in dma_alias_mask
// which is used to program permissible bus-devfn source addresses for DMA
// requests in an IOMMU.  These aliases factor into IOMMU group creation
// and are useful for devices generating DMA requests beyond or different
// from their logical bus-devfn.  Examples include device quirks where the
// device simply uses the wrong devfn, as well as non-transparent bridges
// where the alias may be a proxy for devices in another domain.
//
// IOMMU group creation is performed during device discovery or addition,
// prior to any potential DMA mapping and therefore prior to driver probing
// (especially for userspace assigned devices where IOMMU group definition
// cannot be left as a userspace activity).  DMA aliases should therefore
// be configured via quirks, such as the PCI fixup header quirk.
#[unsafe(no_mangle)]
extern "C" fn pci_devs_are_dma_aliases(dev1: *mut pci_dev, dev2: *mut pci_dev) -> bool {
let __v0: bool = !unsafe { (*dev1).dma_alias_mask }.is_null() && test_bit__3!(unsafe { (*dev2).devfn }, unsafe { (*dev1).dma_alias_mask }) != 0;
let __v2: bool = __v0 || !unsafe { (*dev2).dma_alias_mask }.is_null() && test_bit__3!(unsafe { (*dev1).devfn }, unsafe { (*dev2).dma_alias_mask }) != 0;
let __v5: bool = __v2 || pci_real_dma_dev(dev1) == dev2;
__v5 || pci_real_dma_dev(dev2) == dev1
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_device_is_present")]
extern "C" fn pci_device_is_present(mut pdev: *mut pci_dev) -> bool {
let mut v: u32 = 0;
// Check PF if pdev is a VF, since VF Vendor/Device IDs are 0xffff
pdev = pci_physfn(pdev);
if pci_dev_is_disconnected(pdev as *const pci_dev) {
return false;
}
unsafe { pci_bus_read_dev_vendor_id((*pdev).bus, (*pdev).devfn as i32, core::ptr::addr_of_mut!(v), 0i32) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_device_is_present_903: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> bool>, *mut ()>(Some(pci_device_is_present)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_ignore_hotplug")]
extern "C" fn pci_ignore_hotplug(dev: *mut pci_dev) {
let bridge: *mut pci_dev = unsafe { (*(*dev).bus).self_ };
unsafe {
(*dev).__slate_bits_1.__set_ignore_hotplug(1u32);
}
// Propagate the "ignore hotplug" setting to the parent bridge.
if !bridge.is_null() {
unsafe {
(*bridge).__slate_bits_1.__set_ignore_hotplug(1u32);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_ignore_hotplug_904: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev)>, *mut ()>(Some(pci_ignore_hotplug)) };

/// pci_real_dma_dev - Get PCI DMA device for PCI device
/// @dev: the PCI device that may have a PCI DMA alias
///
/// Permits the platform to provide architecture-specific functionality to
/// devices needing to alias DMA to another PCI device on another PCI bus. If
/// the PCI device is on the same bus, it is recommended to use
/// pci_add_dma_alias(). This is the default implementation. Architecture
/// implementations can override this.
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pci_real_dma_dev(dev: *mut pci_dev) -> *mut pci_dev {
dev
}

#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pcibios_default_alignment() -> u64 {
0
}

/// Arches that don't want to expose struct resource to userland as-is in
/// sysfs and /proc can implement their own pci_resource_to_user().
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pci_resource_to_user(dev: *const pci_dev, bar: i32, rsrc: *const resource, start: *mut u64, end: *mut u64) {
unsafe {
*start = (*rsrc).start;
}
unsafe {
*end = (*rsrc).end;
}
}

static mut resource_alignment_param: *mut u8 = unsafe { core::mem::zeroed() };

static mut resource_alignment_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

/// pci_specified_resource_alignment - get resource alignment specified by user.
/// @dev: the PCI device to get
/// @resize: whether or not to change resources' size when reassigning alignment
///
/// RETURNS: Resource alignment if it is specified.
///          Zero if it is not specified.
fn pci_specified_resource_alignment(dev: *mut pci_dev, resize: &mut bool) -> u64 {
let mut __ret_once: bool = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut p: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *const u8 = core::ptr::addr_of_mut!(p) as *mut *const u8;
let mut align: u64 = unsafe { core::mem::zeroed() };
let mut count: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(count) as *mut i32;
let mut align_order: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(align_order) as *mut i32;
unsafe {
'__join_0: {
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
align = pcibios_default_alignment();
let __slate_auto_bytes_2 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
spin_lock(core::ptr::addr_of_mut!(resource_alignment_lock));
*__slate_slot_13 = resource_alignment_param as *const u8;
if !((*__slate_slot_13).is_null() || *(*__slate_slot_13) == 0) {
if pci_has_flag(4i32) != 0 {
align = 0;
let __ret_do_once: bool = true;
let __ret_cond: bool = __ret_do_once;
__ret_once = false;
if unlikely__2!(__ret_cond) != 0 && unlikely__2!(!__already_done__6474) != 0 {
__already_done__6474 = true;
__ret_once = true;
}
let __v9: i64 = __ret_once as i64;
if __v9 != 0 {
while false {
}
_printk(c"\x016PCI: Ignoring requested alignments (PCI_PROBE_ONLY)\n".as_ptr() as *const u8);
}
} else {
'__join_5: {
'__join_7: {
loop {
if *(*__slate_slot_13) != 0 {
*__slate_slot_15 = 0;
if sscanf(*__slate_slot_13, c"%d%n".as_ptr() as *const u8, core::ptr::addr_of_mut!(*__slate_slot_16), core::ptr::addr_of_mut!(*__slate_slot_15)) == 1 && (*(*__slate_slot_13).wrapping_offset(*__slate_slot_15 as isize) as i32) == 64 {
let __v5: *const u8 = *__slate_slot_13;
*__slate_slot_13 = __v5.wrapping_offset((*__slate_slot_15).wrapping_add(1) as isize);
if *__slate_slot_16 > 63 {
while false {
}
_printk(c"\x013PCI: Invalid requested alignment (order %d)\n".as_ptr() as *const u8, *__slate_slot_16);
*__slate_slot_16 = 12;
}
} else {
*__slate_slot_16 = 12;
}
*__slate_slot_12 = pci_dev_str_match(dev, *__slate_slot_13, core::ptr::addr_of_mut!(*__slate_slot_13));
if *__slate_slot_12 == 1 {
break '__join_5;
} else {
if *__slate_slot_12 < 0 {
break '__join_7;
} else {
if (*(*__slate_slot_13) as i32) != 59 && (*(*__slate_slot_13) as i32) != 44 {
break;
} else {
let __v1: *const u8 = *__slate_slot_13;
*__slate_slot_13 = __v1.wrapping_offset(1 as isize);
}
}
}
} else {
break '__join_0;
}
}
// End of param or invalid format
break '__join_0;
}
while false {
}
_printk(c"\x013PCI: Can't parse resource_alignment parameter: %s\n".as_ptr() as *const u8, *__slate_slot_13);
break '__join_0;
}
*(resize as *mut bool) = true;
align = 1u64 << *__slate_slot_16;
}
}
}
spin_unlock(core::ptr::addr_of_mut!(resource_alignment_lock));
return align;
}
}

#[unsafe(link_section = ".data..once")]
static mut __already_done__6474: bool = unsafe { core::mem::zeroed() };

fn pci_request_resource_alignment(dev: *mut pci_dev, bar: i32, align: u64, resize: bool) {
let r: *mut resource = unsafe { (*dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(bar as isize);
let r_name: *const u8 = pci_resource_name(dev, bar as u32);
let mut size: u64 = 0;
if (unsafe { (*r).flags }) & 512 == 0 {
return;
}
if (unsafe { (*r).flags }) & 16u64 != 0 {
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s %pR: ignoring requested alignment %#llx\n".as_ptr() as *const u8, r_name, r, align) };
return;
}
size = resource_size(r as *const resource);
if size >= align {
return;
}
// Increase the alignment of the resource.  There are two ways we
// can do this:
//
// 1) Increase the size of the resource.  BARs are aligned on their
//    size, so when we reallocate space for this resource, we'll
//    allocate it with the larger alignment.  This also prevents
//    assignment of any other BARs inside the alignment region, so
//    if we're requesting page alignment, this means no other BARs
//    will share the page.
//
//    The disadvantage is that this makes the resource larger than
//    the hardware BAR, which may break drivers that compute things
//    based on the resource size, e.g., to find registers at a
//    fixed offset before the end of the BAR.
//
// 2) Retain the resource size, but use IORESOURCE_STARTALIGN and
//    set r->start to the desired alignment.  By itself this
//    doesn't prevent other BARs being put inside the alignment
//    region, but if we realign *every* resource of every device in
//    the system, none of them will share an alignment region.
//
// When the user has requested alignment for only some devices via
// the "pci=resource_alignment" argument, "resize" is true and we
// use the first method.  Otherwise we assume we're aligning all
// devices and we use the second.
while false {
}
unsafe { _dev_info(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"%s %pR: requesting alignment to %#llx\n".as_ptr() as *const u8, r_name, r, align) };
if resize {
unsafe {
(*r).start = 0;
}
unsafe {
(*r).end = align.wrapping_sub(1);
}
} else {
let __v4: u64 = (unsafe { (*r).flags }) & 18446744073709289471u64;
unsafe {
(*r).flags = __v4;
}
unsafe {
(*r).flags = (*r).flags | 524288;
}
resource_set_range(r, align, size);
}
unsafe {
(*r).flags = (*r).flags | 536870912;
}
}

/// This function disables memory decoding and releases memory resources
/// of the device specified by kernel's boot parameter 'pci=resource_alignment='.
/// It also rounds up size to specified alignment.
/// Later on, the kernel will assign page-aligned memory resource back
/// to the device.
#[unsafe(no_mangle)]
extern "C" fn pci_reassigndev_resource_alignment(dev: *mut pci_dev) {
let mut i: i32 = 0;
let mut align: u64 = 0;
let mut command: u16 = 0;
let mut resize: bool = false;
// VF BARs are read-only zero according to SR-IOV spec r1.1, sec
// 3.4.1.11.  Their resources are allocated from the space
// described by the VF BARx register in the PF's SR-IOV capability.
// We can't influence their alignment here.
if ((unsafe { (*dev).__slate_bits_3.__get_is_virtfn() }) as i32) != 0 {
return;
}
// check if specified PCI is target device to reassign
align = pci_specified_resource_alignment(dev, unsafe { &mut (*core::ptr::addr_of_mut!(resize)) });
if align == 0 {
return;
}
if ((unsafe { (*dev).hdr_type }) as i32) == 0 && (unsafe { (*dev).class }) >> 8i32 == 1536 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*dev).dev) as *const device, c"Can't reassign resources to host bridge\n".as_ptr() as *const u8) };
return;
}
unsafe { pci_read_config_word(dev as *const pci_dev, 4i32, core::ptr::addr_of_mut!(command)) };
let __v2: u16 = ((command as i32) & !2) as u16;
command = __v2;
unsafe { pci_write_config_word(dev as *const pci_dev, 4i32, command) };
while i <= 6 {
pci_request_resource_alignment(dev, i, align, resize);
i = i.wrapping_add(1);
}
// Need to disable bridge's resource window,
// to enable the kernel to reassign new resource
// window later on.
if ((unsafe { (*dev).hdr_type }) as i32) == 1 {
i = 7;
while i < 11 {
let r: *mut resource = unsafe { (*dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(i as isize);
if (unsafe { (*r).flags }) & 512 != 0 {
unsafe {
(*r).flags = (*r).flags | 536870912;
}
unsafe {
(*r).end = resource_size(r as *const resource).wrapping_sub(1);
}
unsafe {
(*r).start = 0;
}
}
i = i.wrapping_add(1);
}
unsafe { pci_disable_bridge_window(dev) };
}
}

#[unsafe(link_section = ".text.slate_distinct.pci.resource_alignment_show")]
extern "C" fn resource_alignment_show(bus: *const bus_type, buf: *mut u8) -> i64 {
let mut count: u64 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!(resource_alignment_lock) });
if !unsafe { resource_alignment_param }.is_null() {
count = (unsafe { sysfs_emit(buf, c"%s\n".as_ptr() as *const u8, resource_alignment_param) }) as u64;
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(resource_alignment_lock) });
count as i64
}

#[unsafe(link_section = ".text.slate_distinct.pci.resource_alignment_store")]
extern "C" fn resource_alignment_store(bus: *const bus_type, buf: *const u8, count: u64) -> i64 {
if count >= (_AC!() << 12i32).wrapping_sub(1) {
return 22i32.wrapping_neg() as i64;
}
let param: *mut u8 = unsafe { kstrndup(buf, count, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) };
if param.is_null() {
return 12i32.wrapping_neg() as i64;
}
let end: *mut u8 = unsafe { strchr(param as *const u8, 10i32) };
if !end.is_null() {
unsafe {
*end = 0;
}
}
spin_lock(unsafe { core::ptr::addr_of_mut!(resource_alignment_lock) });
let old: *mut u8 = unsafe { resource_alignment_param };
if (unsafe { strlen(param as *const u8) }) != 0 {
unsafe {
resource_alignment_param = param;
}
} else {
unsafe { kfree(param as *const ()) };
unsafe {
resource_alignment_param = core::ptr::null_mut::<u8>();
}
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(resource_alignment_lock) });
unsafe { kfree(old as *const ()) };
count as i64
}

static mut bus_attr_resource_alignment: bus_attribute = bus_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"resource_alignment\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_pci_c_6679>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_pci_c_6679_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_pci_c_6679_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_pci_c_6679_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_pci_c_6679_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_pci_c_6679_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, show: Some(resource_alignment_show), store: Some(resource_alignment_store) };

#[unsafe(link_section = ".init.text")]
extern "C" fn pci_resource_alignment_sysfs_init() -> i32 {
unsafe { bus_create_file(core::ptr::addr_of!(pci_bus_type), core::ptr::addr_of_mut!(bus_attr_resource_alignment)) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_resource_alignment_sysfs_init_906: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(pci_resource_alignment_sysfs_init)) };

fn pci_no_domains() {
unsafe {
pci_domains_supported = 0;
}
}

static mut pci_domain_nr_dynamic_ida: ida = ida { xa: xarray { xa_lock: spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} }, xa_flags: 16777221u32, xa_head: core::ptr::null_mut::<()>() } };

// pci_bus_find_emul_domain_nr() - allocate a PCI domain number per constraints
// @hint: desired domain, 0 if any ID in the range of @min to @max is acceptable
// @min: minimum allowable domain
// @max: maximum allowable domain, no IDs higher than INT_MAX will be returned
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_bus_find_emul_domain_nr_910: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, u32, u32) -> i32>, *mut ()>(Some(pci_bus_find_emul_domain_nr)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.pci.pci_bus_release_emul_domain_nr")]
extern "C" fn pci_bus_release_emul_domain_nr(domain_nr: i32) {
unsafe { ida_free(core::ptr::addr_of_mut!(pci_domain_nr_dynamic_ida), domain_nr as u32) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_bus_release_emul_domain_nr_911: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32)>, *mut ()>(Some(pci_bus_release_emul_domain_nr)) };

/// pci_ext_cfg_avail - can we access extended PCI config space?
///
/// Returns 1 if we can access PCI extended config space (offsets
/// greater than 0xff). This is the default implementation. Architecture
/// implementations can override this.
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn pci_ext_cfg_avail() -> i32 {
1
}

#[unsafe(link_section = ".init.text")]
extern "C" fn pci_setup(mut str: *mut u8) -> i32 {
while !str.is_null() {
let mut k: *mut u8 = unsafe { strchr(str as *const u8, 44i32) };
if !k.is_null() {
let __v0: *mut u8 = k;
k = __v0.wrapping_offset(1 as isize);
unsafe {
*__v0 = 0;
}
}
let __v2: bool = if (unsafe { *str }) != 0 {
let __v3: *mut u8 = pcibios_setup(str);
str = __v3;
!__v3.is_null()
} else {
false
};
if __v2 && (unsafe { *str }) != 0 {
if (unsafe { pci_setup_cardbus(str) }) == 0 {
// Function handled the parameters
} else {
if (unsafe { strcmp(str as *const u8, c"nomsi".as_ptr() as *const u8) }) == 0 {
unsafe { pci_no_msi() };
} else {
if (unsafe { strncmp(str as *const u8, c"noats".as_ptr() as *const u8, 5u64) }) == 0 {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x016PCIe: ATS is disabled\n".as_ptr() as *const u8) };
unsafe {
pcie_ats_disabled = true;
}
} else {
if (unsafe { strcmp(str as *const u8, c"noaer".as_ptr() as *const u8) }) == 0 {
pci_no_aer();
} else {
if (unsafe { strcmp(str as *const u8, c"earlydump".as_ptr() as *const u8) }) == 0 {
unsafe {
pci_early_dump = true;
}
} else {
if (unsafe { strncmp(str as *const u8, c"realloc=".as_ptr() as *const u8, 8u64) }) == 0 {
unsafe { pci_realloc_get_opt(str.wrapping_offset(8 as isize)) };
} else {
if (unsafe { strncmp(str as *const u8, c"realloc".as_ptr() as *const u8, 7u64) }) == 0 {
unsafe { pci_realloc_get_opt(c"on".as_ptr() as *mut u8) };
} else {
if (unsafe { strcmp(str as *const u8, c"nodomains".as_ptr() as *const u8) }) == 0 {
pci_no_domains();
} else {
if (unsafe { strncmp(str as *const u8, c"noari".as_ptr() as *const u8, 5u64) }) == 0 {
unsafe {
pcie_ari_disabled = true;
}
} else {
if (unsafe { strncmp(str as *const u8, c"notph".as_ptr() as *const u8, 5u64) }) == 0 {
pci_no_tph();
} else {
if (unsafe { strncmp(str as *const u8, c"resource_alignment=".as_ptr() as *const u8, 19u64) }) == 0 {
unsafe {
resource_alignment_param = str.wrapping_offset(19 as isize);
}
} else {
if (unsafe { strncmp(str as *const u8, c"ecrc=".as_ptr() as *const u8, 5u64) }) == 0 {
pcie_ecrc_get_policy(str.wrapping_offset(5 as isize));
} else {
if (unsafe { strncmp(str as *const u8, c"hpiosize=".as_ptr() as *const u8, 9u64) }) == 0 {
unsafe {
pci_hotplug_io_size = memparse(str.wrapping_offset(9 as isize) as *const u8, core::ptr::addr_of_mut!(str));
}
} else {
if (unsafe { strncmp(str as *const u8, c"hpmmiosize=".as_ptr() as *const u8, 11u64) }) == 0 {
unsafe {
pci_hotplug_mmio_size = memparse(str.wrapping_offset(11 as isize) as *const u8, core::ptr::addr_of_mut!(str));
}
} else {
if (unsafe { strncmp(str as *const u8, c"hpmmioprefsize=".as_ptr() as *const u8, 15u64) }) == 0 {
unsafe {
pci_hotplug_mmio_pref_size = memparse(str.wrapping_offset(15 as isize) as *const u8, core::ptr::addr_of_mut!(str));
}
} else {
if (unsafe { strncmp(str as *const u8, c"hpmemsize=".as_ptr() as *const u8, 10u64) }) == 0 {
unsafe {
pci_hotplug_mmio_size = memparse(str.wrapping_offset(10 as isize) as *const u8, core::ptr::addr_of_mut!(str));
}
unsafe {
pci_hotplug_mmio_pref_size = pci_hotplug_mmio_size;
}
} else {
if (unsafe { strncmp(str as *const u8, c"hpbussize=".as_ptr() as *const u8, 10u64) }) == 0 {
unsafe {
pci_hotplug_bus_size = simple_strtoul(str.wrapping_offset(10 as isize) as *const u8, core::ptr::addr_of_mut!(str), 0u32);
}
if (unsafe { pci_hotplug_bus_size }) > 255 {
unsafe {
pci_hotplug_bus_size = 1;
}
}
} else {
if (unsafe { strncmp(str as *const u8, c"pcie_bus_tune_off".as_ptr() as *const u8, 17u64) }) == 0 {
unsafe {
pcie_bus_config = 0;
}
} else {
if (unsafe { strncmp(str as *const u8, c"pcie_bus_safe".as_ptr() as *const u8, 13u64) }) == 0 {
unsafe {
pcie_bus_config = 2;
}
} else {
if (unsafe { strncmp(str as *const u8, c"pcie_bus_perf".as_ptr() as *const u8, 13u64) }) == 0 {
unsafe {
pcie_bus_config = 3;
}
} else {
if (unsafe { strncmp(str as *const u8, c"pcie_bus_peer2peer".as_ptr() as *const u8, 18u64) }) == 0 {
unsafe {
pcie_bus_config = 4;
}
} else {
if (unsafe { strncmp(str as *const u8, c"pcie_scan_all".as_ptr() as *const u8, 13u64) }) == 0 {
pci_add_flags(64i32);
} else {
if (unsafe { strncmp(str as *const u8, c"disable_acs_redir=".as_ptr() as *const u8, 18u64) }) == 0 {
unsafe {
disable_acs_redir_param = str.wrapping_offset(18 as isize) as *const u8;
}
} else {
if (unsafe { strncmp(str as *const u8, c"config_acs=".as_ptr() as *const u8, 11u64) }) == 0 {
unsafe {
config_acs_param = str.wrapping_offset(11 as isize) as *const u8;
}
} else {
let mut __v6: i32;
while false {
}
__v6 = unsafe { _printk(c"\x013PCI: Unknown option `%s'\n".as_ptr() as *const u8, str) };
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
str = k;
}
0
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_pci_setup: [u8; 4] = [112, 99, 105, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_pci_setup: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_pci_setup) as *const u8 };
__t0.setup_func = Some(pci_setup);
__t0.early = 1i32;
__t0
};

/// 'resource_alignment_param' and 'disable_acs_redir_param' are initialized
/// in pci_setup(), above, to point to data in the __initdata section which
/// will be freed after the init sequence is complete. We can't allocate memory
/// in pci_setup() because some architectures do not have any memory allocation
/// service available during an early_param() call. So we allocate memory and
/// copy the variable here before the init section is freed.
#[unsafe(link_section = ".init.text")]
extern "C" fn pci_realloc_setup_params() -> i32 {
unsafe {
resource_alignment_param = kstrdup(resource_alignment_param as *const u8, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
}
unsafe {
disable_acs_redir_param = kstrdup(disable_acs_redir_param, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *const u8;
}
unsafe {
config_acs_param = kstrdup(config_acs_param, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *const u8;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pci_realloc_setup_params_913: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(pci_realloc_setup_params)) };

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_pci_disable_device_3: [u8; 41] = *b"%s %s: disabling already-disabled device\0";

static __slate_asm_string_pci_disable_device_4: [u8; 47] = *b"/home/takashi/c-corpus/linux/drivers/pci/pci.c\0";

static __slate_asm_string_pci_remap_iospace_3: [u8; 54] = *b"This architecture does not support memory mapped I/O\n\0";

static __slate_asm_string_pci_remap_iospace_4: [u8; 47] = *b"/home/takashi/c-corpus/linux/drivers/pci/pci.c\0";

static __slate_asm_string_pci_set_vga_state_1: [u8; 1] = *b"\0";

static __slate_asm_string_pci_set_vga_state_2: [u8; 47] = *b"/home/takashi/c-corpus/linux/drivers/pci/pci.c\0";
