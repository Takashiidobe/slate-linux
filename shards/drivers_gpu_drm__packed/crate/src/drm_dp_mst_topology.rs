
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
//! Copyright © 2014 Red Hat
//!
//! Permission to use, copy, modify, distribute, and sell this software and its
//! documentation for any purpose is hereby granted without fee, provided that
//! the above copyright notice appear in all copies and that both that copyright
//! notice and this permission notice appear in supporting documentation, and
//! that the name of the copyright holders not be used in advertising or
//! publicity pertaining to distribution of the software without specific,
//! written prior permission.  The copyright holders make no representations
//! about the suitability of this software for any purpose.  It is provided "as
//! is" without express or implied warranty.
//!
//! THE COPYRIGHT HOLDERS DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
//! INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO
//! EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY SPECIAL, INDIRECT OR
//! CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
//! DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
//! TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
//! OF THIS SOFTWARE.
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! small_const_nbits(n) is true precisely when it is known at compile-time
//! that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
//! various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
//! of size 0 are very rare, and a compile-time-known-size 0 is most likely
//! a sign of error. They will be handled correctly by the bit/bitmap APIs,
//! but using the out-of-line functions, so that the inline implementations
//! can unconditionally dereference the pointer(s).
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
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! FIELD_PREP() - prepare a bitfield element
//! @_mask: shifted mask defining the field's length and position
//! @_val:  value to put in the field
//!
//! FIELD_PREP() masks and shifts up the value.  The result should
//! be combined with other fields of the bitfield using logical OR.
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! New user namespace
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
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
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! warning conditions
//! Internal kernel timer frequency
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
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
//! Prevent speculative execution past this barrier.
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
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
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
//! @a is a power of 2 value
//! for_each_set_bit_wrap - iterate over all set bits starting from @start, and
//! wrapping around the end of bitmap.
//! @bit: offset for current iteration
//! @addr: bitmap address to base the search on
//! @size: bitmap size in number of bits
//! @start: Starting bit for bitmap traversing, wrapping around the bitmap end
//! for_each_if - helper for handling conditionals in various for_each macros
//! @condition: The condition to check
//!
//! Typical use::
//!
//! 	#define for_each_foo_bar(x, y) \'
//! 		list_for_each_entry(x, y->list, head) \'
//! 			for_each_if(x->something == SOMETHING)
//!
//! The for_each_if() macro makes the use of for_each_foo_bar() less error
//! prone.
//! lower_32_bits - return bits 0-31 of a number
//! @n: the number we're accessing
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
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
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
//! The microseconds/nanosecond delay multiplicators are used to convert a
//! constant microseconds/nanoseconds value to a value which can be used by the
//! architectures specific implementation to transform it into loops.
//! Resource type
//! wait_queue_entry::flags
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
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! 31 pointers + header align the folio_batch structure to a power of two
//! Receiver Capability
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! Legacy last prefixes
//! REX2 M0
//! Kprobe status flags
//! When adding a new stereo mode don't forget to adjust DRM_MODE_FLAGS_3D_MAX
//! (define not exposed to user space).
//! Number of bits in the handle that stack depot doesn't use. Users may store
//! information in them via stack_depot_set/get_extra_bits.
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
// __must_not_hold() - function attribute, caller must not hold context lock
//
// Function attribute declaring that the caller must not hold the given context
// lock instance(s).
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// FIELD_GET() - extract a bitfield element
// @_mask: shifted mask defining the field's length and position
// @_reg:  value of entire bitfield
//
// FIELD_GET() extracts the field specified by @_mask from the
// bitfield passed in as @_reg by masking and shifting it down.
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
// Same as above but for u64 dividends. divisor must be a 32-bit
// number.
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
// debug-level messages
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
// "acpi" ACPI via MSR
// "sse"
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
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "apic_c1e" AMD Erratum 400
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
// Text file busy
// No space left on device
// Invalid system call number
// Protocol error
// Value too large for defined data type
// Operation not supported on transport endpoint
// Remote I/O error
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
// min3 - return minimum of three values
// @x: first value
// @y: second value
// @z: third value
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
// container_of_const - cast a member of a structure out to the containing
// 			structure and preserve the const-ness of the pointer
// @ptr:		the pointer to the member
// @type:		the type of the container struct this is embedded in.
// @member:		the name of the member within the struct.
//
// Always prefer container_of_const() instead of container_of() in new code.
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
// No static inline type checking - see Subtlety (1) above.
// for_each_cpu_wrap - iterate over every cpu in a mask, starting at a specified location
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
// @start: the start location
//
// The implementation does not assume any bit in @mask is set (including @start).
//
// After the loop, cpu is >= nr_cpu_ids.
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
// Are we running in atomic context?  WARNING: this macro cannot
// always detect atomic context; in particular, it cannot know about
// held spinlocks in non-preemptible kernels.  Thus it should not be
// used in the general case to determine whether sleeping is possible.
// Do not use in_atomic() in driver code.
// disable sas during sighandling
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
// Resource extended types
// No address assigned yet
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
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// 1.3
// 1.2
// 2.0
// eDP
// XXX 1.2?
// Multiple stream transport
// 1.2
// 1.2
// 2.0
// 1.2
// DP 1.4
// eDP 1.4
// eDP 1.4
// DP Forward error Correction Registers
// 1.4
// DP-HDMI2.1 PCON DSC ENCODER SUPPORT
// 0x92 through 0x9E
// 1.2
// 1.2 MST
// 1.2 MST
// 1.2 MST
// Branch Device-specific
// Sideband MSG Buffers
// 1.2 MST
// 1.2 MST
// 1.2 MST
// 1.2 MST
// DP 1.2 Sideband message defines
//
// peer device type - DP 1.2a Table 2-92
// DP 1.2 MST sideband request names DP 1.2a Table 2-80
// DP 1.3
// DP 1.2 MST sideband reply types
// DP 1.2 MST sideband nak reasons - table 2.84
// DSC Capabilities 0x60 through 0x6F
// PT_TRACE_* event enable flags
// 0x66
//
// LPFX1
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
// Escape opcodes
// Group opcodes (1-16)
// Immediates
// VEX bit flags
// VEX3 Byte2
// VEX bit fields
// EVEX Byte1
// VEX3 Byte1
// VEX2.M always 1
// VEX3 Byte2, VEX2 Byte1
// XOP Byte2
// XOP Byte3
// breakpoint has already gone
// probe is temporarily disabled
// probe is really optimized.
// NOTE:
// this flag is only for optimized_kprobe.
// probe is using ftrace
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
/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE {
() => {
((core::mem::size_of::<[u8; 48]>() as u64) / (core::mem::size_of::<u8>() as u64)).wrapping_add(__must_be_array__24!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__9 {
() => {
((core::mem::size_of::<[*const u8; 57]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__7!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__10 {
() => {
((core::mem::size_of::<[u8; 48]>() as u64) / (core::mem::size_of::<u8>() as u64)).wrapping_add(__must_be_array__13!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__11 {
() => {
((core::mem::size_of::<[u8; 7]>() as u64) / (core::mem::size_of::<u8>() as u64)).wrapping_add(__must_be_array__26!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__12 {
() => {
((core::mem::size_of::<[u8; 256]>() as u64) / (core::mem::size_of::<u8>() as u64)).wrapping_add(__must_be_array__28!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__23 {
() => {
((core::mem::size_of::<[*const u8; 5]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__32!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__27 {
() => {
((core::mem::size_of::<[*const u8; 11]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__3!() as u64)
};
}

macro_rules! BITS_PER_TYPE {
($type:ty) => {
(core::mem::size_of::<$type>() as u64).wrapping_mul(8i32 as $type)
};
}

macro_rules! BIT {
($nr:expr) => {
UL!() << $nr
};
}

macro_rules! BIT__2 {
($nr:expr) => {
UL!() << (($nr) as u32)
};
}

macro_rules! BIT__3 {
($nr:expr) => {
UL!() << (($nr) as i32)
};
}

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

/// DIV64_U64_ROUND_UP - unsigned 64bit divide with 64bit divisor rounded up
/// @ll: unsigned 64bit dividend
/// @d: unsigned 64bit divisor
///
/// Divide unsigned 64bit dividend by unsigned 64bit divisor
/// and round up.
///
/// Return: dividend / divisor rounded up
macro_rules! DIV64_U64_ROUND_UP {
($ll:expr, $d:expr) => {
{
let mut __v: u64;
let __v2: u64 = $d;
__v = div64_u64($ll.wrapping_add(__v2).wrapping_sub(1), __v2);
__v
}
};
}

macro_rules! DIV_ROUND_DOWN_ULL {
($ll:expr, $d:expr) => {
{
let mut __v: u64;
let mut __v2: u64 = $ll;
do_div!(__v2, $d);
__v = __v2;
__v
}
};
}

/// prior to 1.2 bit 7 was reserved mbz
macro_rules! DP_GET_SINK_COUNT {
($x:expr) => {
((($x) as i32) & 128) >> 1i32 | (($x) as i32) & 63
};
}

macro_rules! GENMASK_INPUT_CHECK__3 {
() => {
core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_1032>() as i32
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
(GENMASK_INPUT_CHECK__3!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
unlikely__2!(((($x) as *mut ()) as u64) >= (4095i32.wrapping_neg() as u64))
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

macro_rules! __READ_ONCE {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i32) }
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

macro_rules! __bf_cast_unsigned {
($x:expr) => {
($x) as u64
};
}

macro_rules! __bf_cast_unsigned__2 {
($x:expr) => {
($x) as u8
};
}

macro_rules! __cmp_once__2 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = ($x) as $type;
let __v3: $type = ($y) as $type;
__v = if __v2 < __v3 { __v2 } else { __v3 };
__v
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

/// Compile time versions of __arch_hweightN()
macro_rules! __const_hweight8 {
($w:expr) => {
(((($w) as u64) & 1u64 != 0) as i32).wrapping_add(((($w) as u64) & 2u64 != 0) as i32).wrapping_add(((($w) as u64) & 4u64 != 0) as i32).wrapping_add(((($w) as u64) & 8u64 != 0) as i32).wrapping_add(((($w) as u64) & 16u64 != 0) as i32).wrapping_add(((($w) as u64) & 32u64 != 0) as i32).wrapping_add(((($w) as u64) & 64u64 != 0) as i32).wrapping_add(((($w) as u64) & 128u64 != 0) as i32) as u32
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

macro_rules! __must_be_array__3 {
() => {
core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_148>() as i32
};
}

macro_rules! __must_be_array__7 {
() => {
core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_123>() as i32
};
}

macro_rules! __must_be_array__13 {
() => {
core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_795>() as i32
};
}

macro_rules! __must_be_array__24 {
() => {
core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_794>() as i32
};
}

macro_rules! __must_be_array__26 {
() => {
core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_685>() as i32
};
}

macro_rules! __must_be_array__28 {
() => {
core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_809>() as i32
};
}

macro_rules! __must_be_array__32 {
() => {
core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_168>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__2 {
() => {
core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_5996_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__6 {
() => {
core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_5996>() as i32
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
macro_rules! __native_word__12 {
() => {
((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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
if is_signed_type__4!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u8>() }) as u64) < 4) as i32)) }
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

macro_rules! __types_ok3 {
($ux:expr, $uy:expr, $uz:expr) => {
__sign_use!($ux) & __sign_use!($uy) & __sign_use!($uz)
};
}

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
};
}

macro_rules! __types_ok__5 {
($ux:expr, $uy:expr) => {
__sign_use__4!($ux) & __sign_use__4!($uy)
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
macro_rules! container_of__9 {
($ptr:expr) => {
{
let mut __v: *mut drm_dp_mst_port;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__28!() as isize)) }) as *mut ()) as *mut drm_dp_mst_port;
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
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__48!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__3 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__30!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__16 {
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
macro_rules! container_of__4 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__32!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__20 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__36!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__23 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__14!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__26 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__54!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__29 {
($ptr:expr) => {
{
let mut __v: *mut drm_dp_mst_atomic_payload;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__35!() as isize)) }) as *mut ()) as *mut drm_dp_mst_atomic_payload;
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
($ptr:expr) => {
{
let mut __v: *mut drm_dp_sideband_msg_tx;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__8!() as isize)) }) as *mut ()) as *mut drm_dp_sideband_msg_tx;
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
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__41!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__7 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

///  + ((B + 0.000122)*4096))
macro_rules! dfixed_const {
($A:expr) => {
(($A) << 12i32) as u32
};
}

macro_rules! dfixed_trunc {
($A:expr) => {
(unsafe { $A.full }) >> 12i32
};
}

/// do_div - returns 2 values: calculate remainder and update new dividend
/// @n: uint64_t dividend (will be updated)
/// @base: uint32_t divisor
///
/// Summary:
/// ``uint32_t remainder = n % base;``
/// ``n = n / base;``
///
/// Return: (uint32_t)remainder
///
/// NOTE: macro parameter @n is evaluated multiple times,
/// beware of side effects!
macro_rules! do_div {
($n:expr, $base:expr) => {
let mut __v: u32;
let __v2: u32 = ($base) as u32;
let __v3: u32 = ((($n) as u64) % (__v2 as u64)) as u32;
$n = (($n) as u64) / (__v2 as u64);
__v = __v3;
};
}

macro_rules! drm_debug_enabled {
($category:expr) => {
drm_debug_enabled_raw(($category) as u32)
};
}

/// ffz - find first zero bit in word
/// @word: The word to search
///
/// Undefined if no zero exists, so code should check against ~0UL first.
macro_rules! ffz {
($word:expr) => {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe($word) } { ((!$word as u64).trailing_zeros() as i32) as u64 } else { variable_ffz(($word) as u64) }
};
}

macro_rules! hweight32 {
($w:expr) => {
{
let __v: u32 = if unsafe { __slate_intrinsic_1c5364fcd9da51fe($w) } {
__const_hweight32!($w)
} else {
__arch_hweight32($w)
};
__v
}
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__4 {
() => {
((1i32.wrapping_neg() as u8) as i32) < 1
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
macro_rules! list_entry__13 {
($ptr:expr, $type:ty) => {
container_of__10!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__8 {
($ptr:expr) => {
container_of__33!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__2 {
($ptr:expr, $type:ty) => {
container_of__15!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__10 {
($ptr:expr) => {
container_of__29!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__12 {
($ptr:expr) => {
container_of__9!($ptr)
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
macro_rules! list_first_entry__9 {
($ptr:expr, $member:ident) => {
list_entry__8!(unsafe { (*$ptr).$member })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__2 {
($ptr:expr, $member:ident) => {
list_entry__10!(unsafe { (*$ptr).$member })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__13 {
($ptr:expr, $type:ty, $member:ident) => {
list_entry__2!(unsafe { (*$ptr).$member }, $type)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry {
($ptr:expr, $member:ident) => {
list_entry__12!(unsafe { (*$ptr).$member })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__2 {
($pos:expr, $member:ident) => {
list_entry__10!(unsafe { (*$pos).$member.$member })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__10 {
($pos:expr, $member:ident) => {
list_entry__8!(unsafe { (*$pos).$member.$member })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry {
($pos:expr, $member:ident) => {
list_entry__12!(unsafe { (*$pos).$member.$member })
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

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t__2 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__2!($type, $x, $y)
};
}

macro_rules! offsetof__8 {
() => {
core::mem::offset_of!(drm_dp_sideband_msg_tx, next) as u64
};
}

macro_rules! offsetof__14 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, tx_work) as u64
};
}

macro_rules! offsetof__25 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, next) as u64
};
}

macro_rules! offsetof__28 {
() => {
core::mem::offset_of!(drm_dp_mst_port, next) as u64
};
}

macro_rules! offsetof__30 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, aux) as u64
};
}

macro_rules! offsetof__32 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, topology_kref) as u64
};
}

macro_rules! offsetof__35 {
() => {
core::mem::offset_of!(drm_dp_mst_atomic_payload, next) as u64
};
}

macro_rules! offsetof__36 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, up_req_work) as u64
};
}

macro_rules! offsetof__37 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, work) as u64
};
}

macro_rules! offsetof__41 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, malloc_kref) as u64
};
}

macro_rules! offsetof__48 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, destroy_next) as u64
};
}

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, base) as u64
};
}

macro_rules! offsetof__54 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, delayed_destroy_work) as u64
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
let __v: bool = typecheck__5!(u64) != 0 && typecheck__4!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
}
};
}

macro_rules! to_dp_mst_topology_mgr {
($x:expr) => {
{
let mut __v: *mut drm_dp_mst_topology_mgr;
__v = ((unsafe { ((($x) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(drm_dp_mst_topology_mgr) as isize)) }) as *mut ()) as *mut drm_dp_mst_topology_mgr;
__v
}
};
}

macro_rules! to_dp_mst_topology_state {
($x:expr) => {
{
let mut __v: *mut drm_dp_mst_topology_state;
__v = ((unsafe { ((($x) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(drm_dp_mst_topology_state) as isize)) }) as *mut ()) as *mut drm_dp_mst_topology_state;
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
macro_rules! typecheck__4 {
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
macro_rules! typecheck__5 {
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

macro_rules! unlikely__3 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! wake_up_all {
($x:expr) => {
unsafe { __wake_up($x, 3u32, 0i32, core::ptr::null_mut::<()>()) };
};
}

unsafe extern "C" {
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __SCK__might_resched: static_call_key;
    static mut cpuinfo_op: seq_operations;
    static mut jiffies: u64;
    static mut system_long_wq: *mut workqueue_struct;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut guid_null: __SlateRecord_uuid_h_17;
    static mut kstat: kernel_stat;
    static mut __drm_debug: u64;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn print_hex_dump(arg0: *const u8, arg1: *const u8, arg2: i32, arg3: i32, arg4: i32, arg5: *const (), arg6: u64, arg7: bool);
    fn __cmpxchg_wrong_size();
    fn __xadd_wrong_size();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn memcmp(arg0: *const (), arg1: *const (), arg2: u64) -> i32;
    fn sized_strscpy(arg0: *mut u8, arg1: *const u8, arg2: u64) -> i64;
    fn strlcat(arg0: *mut u8, arg1: *const u8, arg2: u64) -> u64;
    fn kmemdup_noprof(arg0: *const (), arg1: u64, arg2: u32) -> *mut ();
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn __compiletime_assert_16();
    fn __compiletime_assert_19();
    fn __compiletime_assert_20();
    fn get_random_bytes(arg0: *mut (), arg1: u64);
    fn schedule_timeout(arg0: i64) -> i64;
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn iomem_get_mapping() -> *mut address_space;
    fn __init_waitqueue_head(arg0: *mut wait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn __wake_up(arg0: *mut wait_queue_head, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn init_wait_entry(arg0: *mut wait_queue_entry, arg1: i32);
    fn prepare_to_wait_event(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry, arg2: i32) -> i64;
    fn finish_wait(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry);
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_is_locked(arg0: *mut mutex) -> bool;
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn refcount_warn_saturate(arg0: *mut refcount_struct, arg1: u32);
    fn alloc_workqueue_noprof(arg0: *const u8, arg1: u32, arg2: i32, ...) -> *mut workqueue_struct;
    fn destroy_workqueue(arg0: *mut workqueue_struct);
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn flush_work(arg0: *mut work_struct) -> bool;
    fn kfree(arg0: *const ());
    fn __compiletime_assert_220();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn guid_gen(arg0: *mut __SlateRecord_uuid_h_17);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_395();
    fn up(arg0: *mut semaphore);
    fn __compiletime_assert_416();
    fn dev_driver_string(arg0: *const device) -> *const u8;
    fn acpi_create_platform_device(arg0: *mut acpi_device, arg1: *const property_entry) -> *mut platform_device;
    fn seq_printf(arg0: *mut seq_file, arg1: *const u8, ...);
    fn seq_putc(arg0: *mut seq_file, arg1: u8);
    fn i2c_add_adapter(arg0: *mut i2c_adapter) -> i32;
    fn i2c_del_adapter(arg0: *mut i2c_adapter);
    fn drm_mode_object_put(arg0: *mut drm_mode_object);
    fn drm_connector_dynamic_register(arg0: *mut drm_connector) -> i32;
    fn drm_connector_unregister(arg0: *mut drm_connector);
    fn drm_dp_dpcd_read(arg0: *mut drm_dp_aux, arg1: u32, arg2: *mut (), arg3: u64) -> i64;
    fn drm_dp_dpcd_write(arg0: *mut drm_dp_aux, arg1: u32, arg2: *mut (), arg3: u64) -> i64;
    fn drm_dp_read_dpcd_caps(arg0: *mut drm_dp_aux, arg1: *mut u8) -> i32;
    fn drm_dp_dpcd_write_payload(arg0: *mut drm_dp_aux, arg1: i32, arg2: u8, arg3: u8) -> i32;
    fn drm_dp_dpcd_clear_payload(arg0: *mut drm_dp_aux) -> i32;
    fn drm_dp_dpcd_poll_act_handled(arg0: *mut drm_dp_aux, arg1: i32) -> i32;
    fn drm_dp_remote_aux_init(arg0: *mut drm_dp_aux);
    fn drm_dp_read_desc(arg0: *mut drm_dp_aux, arg1: *mut drm_dp_desc, arg2: bool) -> i32;
    fn drm_dp_bw_overhead(arg0: i32, arg1: i32, arg2: i32, arg3: i32, arg4: u64) -> i32;
    fn drm_dp_bw_channel_coding_efficiency(arg0: bool) -> i32;
    fn drm_modeset_lock(arg0: *mut drm_modeset_lock, arg1: *mut drm_modeset_acquire_ctx) -> i32;
    fn drm_modeset_lock_single_interruptible(arg0: *mut drm_modeset_lock) -> i32;
    fn drm_modeset_unlock(arg0: *mut drm_modeset_lock);
    fn __drm_crtc_commit_free(arg0: *mut kref);
    fn drm_crtc_commit_wait(arg0: *mut drm_crtc_commit) -> i32;
    fn drm_atomic_get_crtc_state(arg0: *mut drm_atomic_commit, arg1: *mut drm_crtc) -> *mut drm_crtc_state;
    fn drm_atomic_get_connector_state(arg0: *mut drm_atomic_commit, arg1: *mut drm_connector) -> *mut drm_connector_state;
    fn drm_atomic_private_obj_init(arg0: *mut drm_device, arg1: *mut drm_private_obj, arg2: *const drm_private_state_funcs) -> i32;
    fn drm_atomic_private_obj_fini(arg0: *mut drm_private_obj);
    fn drm_atomic_get_private_obj_state(arg0: *mut drm_atomic_commit, arg1: *mut drm_private_obj) -> *mut drm_private_state;
    fn drm_atomic_get_old_private_obj_state(arg0: *const drm_atomic_commit, arg1: *mut drm_private_obj) -> *mut drm_private_state;
    fn drm_atomic_get_new_private_obj_state(arg0: *const drm_atomic_commit, arg1: *mut drm_private_obj) -> *mut drm_private_state;
    fn drm_atomic_commit(arg0: *mut drm_atomic_commit) -> i32;
    fn __drm_atomic_helper_private_obj_create_state(arg0: *mut drm_private_obj, arg1: *mut drm_private_state);
    fn __drm_atomic_helper_private_obj_duplicate_state(arg0: *mut drm_private_obj, arg1: *mut drm_private_state);
    fn drm_edid_duplicate(arg0: *const edid) -> *mut edid;
    fn drm_edid_get_monitor_name(arg0: *const edid, arg1: *mut u8, arg2: i32);
    fn drm_edid_dup(arg0: *const drm_edid) -> *const drm_edid;
    fn drm_edid_free(arg0: *const drm_edid);
    fn drm_edid_raw(arg0: *const drm_edid) -> *const edid;
    fn drm_edid_read_ddc(arg0: *mut drm_connector, arg1: *mut i2c_adapter) -> *const drm_edid;
    fn __drm_printfn_dbg(arg0: *mut drm_printer, arg1: *mut va_format);
    fn drm_printf(arg0: *mut drm_printer, arg1: *const u8, ...);
    fn __drm_dev_dbg(arg0: *mut _ddebug, arg1: *const device, arg2: u32, arg3: *const u8, ...);
    fn __drm_err(arg0: *const u8, ...);
    fn drm_kms_helper_hotplug_event(arg0: *mut drm_device);
    fn __compiletime_assert_660();
    fn __compiletime_assert_661();
    fn __compiletime_assert_662();
    fn __compiletime_assert_663();
    fn __compiletime_assert_664();
    fn __compiletime_assert_665();
    fn __compiletime_assert_666();
    fn __compiletime_assert_667();
    fn __compiletime_assert_668();
    fn __compiletime_assert_669();
    fn __compiletime_assert_674();
    fn __compiletime_assert_675();
    fn __compiletime_assert_676();
    fn __compiletime_assert_677();
    fn __compiletime_assert_678();
    fn __compiletime_assert_679();
    fn __compiletime_assert_680();
    fn __compiletime_assert_681();
    fn __compiletime_assert_682();
    fn __compiletime_assert_683();
    fn __compiletime_assert_684();
    fn __compiletime_assert_685();
    fn __compiletime_assert_686();
    fn __compiletime_assert_687();
    fn __compiletime_assert_688();
    fn __compiletime_assert_689();
    fn __compiletime_assert_690();
    fn __compiletime_assert_691();
    fn __compiletime_assert_692();
    fn __compiletime_assert_693();
    fn __compiletime_assert_694();
    fn __compiletime_assert_707();
    fn __compiletime_assert_710();
    fn __compiletime_assert_731();
    fn __compiletime_assert_734();
    fn __compiletime_assert_738();
    fn __compiletime_assert_739();
    fn __compiletime_assert_740();
    fn __compiletime_assert_742();
    fn __compiletime_assert_759();
    fn __compiletime_assert_760();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i8"]
    fn __slate_intrinsic_70b71f185ebb8c4a(arg0: u8) -> bool;
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


#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


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
struct va_format {
    fmt: *const u8,
    va: *mut core::ffi::VaList<'static>,
}


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
// pr_warn - Print a warning-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_WARNING loglevel. It uses pr_fmt()
// to generate the format string.
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
fn variable__ffs(mut word: u64) -> u64 {
unsafe {
let __t0: u64;
core::arch::asm!("tzcnt {1:r},{0:r}", lateout(reg) __t0, in(reg) word as u64, options(pure, nomem, nostack, att_syntax));
word = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
word
}

// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
#[inline(always)]
fn variable_ffz(word: u64) -> u64 {
variable__ffs(!word)
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
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_21 {
    pte: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits124U0,
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
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
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
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
struct perf_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_true {
    key: static_key,
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
    __slate_bits_0: __slate_bits::__SlateBits131U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
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
fn raw_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
arch_atomic_set(v, i);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[inline(always)]
fn raw_atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_add(i, v)
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
fn raw_atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_sub!(i, v)
}

#[inline(always)]
fn raw_atomic_try_cmpxchg_relaxed(v: *mut __SlateRecord_types_h_190, old: *mut i32, new: i32) -> bool {
arch_atomic_try_cmpxchg(v, old, new)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
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
fn atomic_try_cmpxchg_relaxed(v: *mut __SlateRecord_types_h_190, old: *mut i32, new: i32) -> bool {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
instrument_read_write(old as *const (), core::mem::size_of::<i32>() as u64);
raw_atomic_try_cmpxchg_relaxed(v, old, new)
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_13: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

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

#[inline(always)]
fn list_add(new: *mut list_head, head: *mut list_head) {
__list_add(new, head, unsafe { (*head).next });
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

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
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

fn list_is_singular(head: *const list_head) -> i32 {
(list_empty(head) == 0 && (unsafe { (*head).next }) == unsafe { (*head).prev }) as i32
}

// list_first_entry_or_null - get the first element from a list
// @ptr:	the list head to take the element from.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
//
// Note that if the list is empty, it returns NULL.
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
fn div64_u64(dividend: u64, divisor: u64) -> u64 {
dividend / divisor
}

fn mul_u32_u32(a: u32, b: u32) -> u64 {
(a as u64).wrapping_mul(b as u64)
}

// DIV_U64_ROUND_UP - unsigned 64bit divide with 32bit divisor rounded up
// @ll: unsigned 64bit dividend
// @d: unsigned 32bit divisor
//
// Divide unsigned 64bit dividend by unsigned 32bit divisor
// and round up.
//
// Return: dividend / divisor rounded up
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
    __slate_bits_0: __slate_bits::__SlateBits132U0,
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
struct timespec64 {
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
struct workqueue_struct {
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
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_sem {
    undo_list: *mut sem_undo_list,
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
struct sysv_shm {
    shm_clist: list_head,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
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
struct task_cputime {
    stime: u64,
    utime: u64,
    sum_exec_runtime: u64,
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
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
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
union sigval {
    sival_int: i32,
    sival_ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct gendisk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount {
    sequence: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
struct rwsem_waiter {
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
struct uprobe {
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
    __slate_bits_0: __slate_bits::__SlateBits288U0,
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
    __slate_bits_0: __slate_bits::__SlateBits74U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits74U1,
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
    __slate_bits_2: __slate_bits::__SlateBits74U2,
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
struct uprobe_xol_ops {
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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


// explicit shadow
// wait_event_timeout - sleep until a condition gets true or a timeout elapses
// @wq_head: the waitqueue to wait on
// @condition: a C expression for the event to wait for
// @timeout: timeout, in jiffies
//
// The process is put to sleep (TASK_UNINTERRUPTIBLE) until the
// @condition evaluates to true. The @condition is checked each time
// the waitqueue @wq_head is woken up.
//
// wake_up() has to be called after changing any variable that could
// change the result of the wait condition.
//
// Returns:
// 0 if the @condition evaluated to %false after the @timeout elapsed,
// 1 if the @condition evaluated to %true after the @timeout elapsed,
// or the remaining jiffies (at least 1) if the @condition evaluated
// to %true before the @timeout elapsed.
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


fn mutex_destroy(lock: *mut mutex) {
}

// mutex_init - initialize the mutex
// @mutex: the mutex to be initialized
//
// Initialize the mutex to unlocked state.
//
// It is not allowed to initialize an already locked mutex.
fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
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


fn refcount_set(r: *mut refcount_struct, n: i32) {
atomic_set(unsafe { core::ptr::addr_of_mut!((*r).refs) }, n);
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


fn refcount_read(r: *const refcount_struct) -> u32 {
atomic_read(unsafe { core::ptr::addr_of!((*r).refs) }) as u32
}

fn __refcount_add_not_zero(i: i32, r: *mut refcount_struct, oldp: *mut i32) -> bool {
let mut old: i32 = refcount_read(r as *const refcount_struct) as i32;
while old != 0 {
if !(!atomic_try_cmpxchg_relaxed(unsafe { core::ptr::addr_of_mut!((*r).refs) }, core::ptr::addr_of_mut!(old), old.wrapping_add(i))) {
break;
}
}
if !oldp.is_null() {
unsafe {
*oldp = old;
}
}
if unlikely__2!(old < 0 || old.wrapping_add(i) < 0) != 0 {
unsafe { refcount_warn_saturate(r, 0u32) };
}
old != 0
}

fn __refcount_add(i: i32, r: *mut refcount_struct, oldp: *mut i32) {
let old: i32 = atomic_fetch_add_relaxed(i, unsafe { core::ptr::addr_of_mut!((*r).refs) });
if !oldp.is_null() {
unsafe {
*oldp = old;
}
}
if unlikely__2!(old == 0) != 0 {
unsafe { refcount_warn_saturate(r, 2u32) };
} else {
if unlikely__2!(old < 0 || old.wrapping_add(i) < 0) != 0 {
unsafe { refcount_warn_saturate(r, 1u32) };
}
}
}

fn __refcount_inc_not_zero(r: *mut refcount_struct, oldp: *mut i32) -> bool {
__refcount_add_not_zero(1i32, r, oldp)
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


fn refcount_inc_not_zero(r: *mut refcount_struct) -> bool {
__refcount_inc_not_zero(r, core::ptr::null_mut::<i32>())
}

fn __refcount_inc(r: *mut refcount_struct, oldp: *mut i32) {
__refcount_add(1i32, r, oldp);
}

fn refcount_inc(r: *mut refcount_struct) {
__refcount_inc(r, core::ptr::null_mut::<i32>());
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
if unlikely__2!(old <= 0 || old.wrapping_sub(i) < 0) != 0 {
unsafe { refcount_warn_saturate(r, 3u32) };
}
false
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


fn kref_init(kref: *mut kref) {
refcount_set(unsafe { core::ptr::addr_of_mut!((*kref).refcount) }, 1i32);
}

fn kref_read(kref: *const kref) -> u32 {
refcount_read(unsafe { core::ptr::addr_of!((*kref).refcount) })
}

fn kref_get(kref: *mut kref) {
refcount_inc(unsafe { core::ptr::addr_of_mut!((*kref).refcount) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
}


fn kref_put(kref: *mut kref, release: Option<unsafe extern "C" fn(*mut kref)>) -> i32 {
if refcount_dec_and_test(unsafe { core::ptr::addr_of_mut!((*kref).refcount) }) {
unsafe { release.unwrap()(kref) };
return 1;
}
0
}

fn kref_get_unless_zero(kref: *mut kref) -> i32 {
refcount_inc_not_zero(unsafe { core::ptr::addr_of_mut!((*kref).refcount) }) as i32
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
struct rcu_work {
    work: work_struct,
    rcu: callback_head,
    wq: *mut workqueue_struct,
}


fn __init_work(work: *mut work_struct, onstack: i32) {
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
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_288 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
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
struct dev_pm_qos {
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
    __slate_bits_0: __slate_bits::__SlateBits421U0,
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
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits452U0,
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
struct folio_queue {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sg_table {
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
    __slate_bits_0: __slate_bits::__SlateBits479U0,
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
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysfs_ops {
    show: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *const u8, u64) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
    __slate_bits_0: __slate_bits::__SlateBits481U0,
}


fn kobject_name(kobj: *const kobject) -> *const u8 {
unsafe { (*kobj).name }
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
struct kset_uevent_ops {
    filter: Option<unsafe extern "C" fn(*const kobject) -> i32>,
    name: Option<unsafe extern "C" fn(*const kobject) -> *const u8>,
    uevent: Option<unsafe extern "C" fn(*const kobject, *mut kobj_uevent_env) -> i32>,
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
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
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
    __slate_bits_0: __slate_bits::__SlateBits506U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits506U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits506U2,
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
    __slate_bits_3: __slate_bits::__SlateBits506U3,
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
struct tty_struct {
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
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct __SlateRecord_uuid_h_17 {
    b: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uuid_h_21 {
    b: [u8; 16],
}


// The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
// not including trailing NUL.
fn guid_equal(u1: *const __SlateRecord_uuid_h_17, u2: *const __SlateRecord_uuid_h_17) -> bool {
(unsafe { memcmp(u1 as *const (), u2 as *const (), core::mem::size_of::<__SlateRecord_uuid_h_17>() as u64) }) == 0
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


fn guid_copy(dst: *mut __SlateRecord_uuid_h_17, src: *const __SlateRecord_uuid_h_17) {
unsafe { memcpy(dst as *mut (), src as *const (), core::mem::size_of::<__SlateRecord_uuid_h_17>() as u64) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
}


fn import_guid(dst: *mut __SlateRecord_uuid_h_17, src: *const u8) {
unsafe { memcpy(dst as *mut (), src as *const (), core::mem::size_of::<__SlateRecord_uuid_h_17>() as u64) };
}

fn export_guid(dst: *mut u8, src: *const __SlateRecord_uuid_h_17) {
unsafe { memcpy(dst as *mut (), src as *const (), core::mem::size_of::<__SlateRecord_uuid_h_17>() as u64) };
}

fn guid_is_null(guid: *const __SlateRecord_uuid_h_17) -> bool {
guid_equal(guid, unsafe { core::ptr::addr_of!(guid_null) })
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
struct assoc_array {
    root: *mut assoc_array_ptr,
    nr_leaves_on_tree: u64,
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
struct key_tag {
    rcu: callback_head,
    usage: refcount_struct,
    removed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct semaphore_waiter {
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
struct request_queue {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct posix_acl {
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
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
    __slate_bits_0: __slate_bits::__SlateBits291U0,
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
union __SlateRecord_dynamic_debug_h_55 {
    dd_key_true: static_key_true,
    dd_key_false: static_key_false,
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
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
struct qc_state {
    s_incoredqs: u32,
    s_state: [qc_type_state; 3],
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
struct path {
    mnt: *mut vfsmount,
    dentry: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct semaphore {
    lock: raw_spinlock,
    count: u32,
    first_waiter: *mut semaphore_waiter,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_acpi_bus_h_249 {
    __slate_bits_0: __slate_bits::__SlateBits791U0,
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
struct __SlateRecord_acpi_bus_h_281 {
    __slate_bits_0: __slate_bits::__SlateBits795U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_gpio_mapping {
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
struct platform_device {
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
struct compat_robust_list_head {
    list: compat_robust_list,
    futex_offset: i32,
    list_op_pending: u32,
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
struct cgroup_taskset {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _ddebug {
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
    format: *const u8,
    __slate_bits_0: __slate_bits::__SlateBits709U0,
    key: __SlateRecord_dynamic_debug_h_55,
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


// Only touch one cacheline for common rbtree-for-core-layout case.
#[repr(C)]
#[derive(Clone, Copy)]
struct mod_kallsyms {
    symtab: *mut elf64_sym,
    num_symtab: u32,
    strtab: *mut u8,
    typetab: *mut u8,
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
struct acpi_device_id {
    id: [u8; 16],
    driver_data: u64,
    cls: u32,
    cls_msk: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct regulator {
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
struct irq_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_parent_ops {
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
struct irq_domain_chip_generic {
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
union __SlateRecord_i2c_h_551 {
    xfer: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
    master_xfer: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_i2c_h_557 {
    xfer_atomic: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
    master_xfer_atomic: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
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
    __slate_bits_0: __slate_bits::__SlateBits503U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gpio_desc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pinctrl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pinctrl_state {
}


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
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
struct __SlateRecord_hdmi_h_214 {
    x: u16,
    y: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hdmi_h_217 {
    x: u16,
    y: u16,
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
union __SlateRecord_hdmi_h_406 {
    hdmi_type1: hdr_static_metadata,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_hotplug_profile {
    kobj: kobject,
    scan_dependent: Option<unsafe extern "C" fn(*mut acpi_device) -> i32>,
    notify_online: Option<unsafe extern "C" fn(*mut acpi_device)>,
    __slate_bits_0: __slate_bits::__SlateBits776U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hdmi_h_419 {
    r#type: u32,
    version: u8,
    length: u8,
    oui: u32,
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
struct drm_file {
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
struct drm_master {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_status {
    __slate_bits_0: __slate_bits::__SlateBits782U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_flags {
    __slate_bits_0: __slate_bits::__SlateBits783U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_dir {
    entry: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_pin {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_pnp_type {
    __slate_bits_0: __slate_bits::__SlateBits787U0,
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
    __slate_bits_0: __slate_bits::__SlateBits789U0,
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
    __slate_bits_0: __slate_bits::__SlateBits793U0,
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
    __slate_bits_0: __slate_bits::__SlateBits797U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_privacy_screen {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_edid {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_wakeup_context {
    func: Option<unsafe extern "C" fn(*mut acpi_device_wakeup_context)>,
    dev: *mut device,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_codec_daifmt {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_codec_params {
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
struct acpi_device_data {
    pointer: *const acpi_object,
    properties: list_head,
    of_compatible: *const acpi_object,
    subnodes: list_head,
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
struct __SlateRecord_drm_connector_h_1088 {
    avi: drm_connector_hdmi_infoframe,
    hdr_drm: drm_connector_hdmi_infoframe,
    spd: drm_connector_hdmi_infoframe,
    hdmi: drm_connector_hdmi_infoframe,
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
struct drm_writeback_job {
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
struct __SlateRecord_drm_connector_h_2062 {
    lock: mutex,
    audio: drm_connector_hdmi_infoframe,
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
struct cec_adapter {
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
    __slate_bits_0: __slate_bits::__SlateBits833U0,
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
struct drm_format_info {
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
struct drm_framebuffer {
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
struct fprop_local_percpu {
    events: percpu_counter,
    period: u32,
    lock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_minor {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vblank_crtc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vma_offset_manager {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vram_mm {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_fb_helper {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_device_h_113 {
    resources: list_head,
    final_kfree: *mut (),
    lock: spinlock,
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
struct dma_fence {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_colorop {
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
    __slate_bits_0: __slate_bits::__SlateBits622U0,
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
struct drm_self_refresh_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_pending_vblank_event {
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
struct rt_mutex_base {
    wait_lock: raw_spinlock,
    waiters: rb_root_cached,
    owner: *mut task_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rt_mutex {
    rtmutex: rt_mutex_base,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_colorop_state {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_msg {
    addr: u16,
    flags: u16,
    len: u16,
    buf: *mut u8,
}


// required for I2C_M_TEN
// SMBus 2.0 or later
// required for I2C_M_RECV_LEN
// Data for SMBus Messages
// As specified in SMBus standard
#[repr(C)]
#[derive(Clone, Copy)]
union i2c_smbus_data {
    byte: u8,
    word: u16,
    block: [u8; 34],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_algorithm {
    __slate_anon_0: __SlateRecord_i2c_h_551,
    __slate_anon_1: __SlateRecord_i2c_h_557,
    smbus_xfer: Option<unsafe extern "C" fn(*mut i2c_adapter, u16, u16, u8, u8, i32, *mut i2c_smbus_data) -> i32>,
    smbus_xfer_atomic: Option<unsafe extern "C" fn(*mut i2c_adapter, u16, u16, u8, u8, i32, *mut i2c_smbus_data) -> i32>,
    functionality: Option<unsafe extern "C" fn(*mut i2c_adapter) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_lock_operations {
    lock_bus: Option<unsafe extern "C" fn(*mut i2c_adapter, u32)>,
    trylock_bus: Option<unsafe extern "C" fn(*mut i2c_adapter, u32) -> i32>,
    unlock_bus: Option<unsafe extern "C" fn(*mut i2c_adapter, u32)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_bus_recovery_info {
    recover_bus: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    get_scl: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    set_scl: Option<unsafe extern "C" fn(*mut i2c_adapter, i32)>,
    get_sda: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    set_sda: Option<unsafe extern "C" fn(*mut i2c_adapter, i32)>,
    get_bus_free: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    prepare_recovery: Option<unsafe extern "C" fn(*mut i2c_adapter)>,
    unprepare_recovery: Option<unsafe extern "C" fn(*mut i2c_adapter)>,
    scl_gpiod: *mut gpio_desc,
    sda_gpiod: *mut gpio_desc,
    pinctrl: *mut pinctrl,
    pins_default: *mut pinctrl_state,
    pins_gpio: *mut pinctrl_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_adapter_quirks {
    flags: u64,
    max_num_msgs: i32,
    max_write_len: u16,
    max_read_len: u16,
    max_comb_1st_msg_len: u16,
    max_comb_2nd_msg_len: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_adapter {
    owner: *mut module,
    class: u32,
    algo: *const i2c_algorithm,
    algo_data: *mut (),
    lock_ops: *const i2c_lock_operations,
    bus_lock: rt_mutex,
    mux_lock: rt_mutex,
    timeout: i32,
    retries: i32,
    dev: device,
    locked_flags: u64,
    nr: i32,
    name: [u8; 48],
    dev_released: completion,
    userspace_clients_lock: mutex,
    userspace_clients: list_head,
    bus_recovery_info: *mut i2c_bus_recovery_info,
    quirks: *const i2c_adapter_quirks,
    host_notify_domain: *mut irq_domain,
    bus_regulator: *mut regulator,
    debugfs: *mut dentry,
    addrs_in_instantiation: [u64; 2],
}


// Adapter locking functions, exported for shared pin cases
#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_any_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_avi_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    itc: bool,
    pixel_repeat: u8,
    colorspace: u32,
    scan_mode: u32,
    colorimetry: u32,
    picture_aspect: u32,
    active_aspect: u32,
    extended_colorimetry: u32,
    quantization_range: u32,
    nups: u32,
    video_code: u8,
    ycc_quantization_range: u32,
    content_type: u32,
    top_bar: u16,
    bottom_bar: u16,
    left_bar: u16,
    right_bar: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_link_addr_reply_port {
    input_port: bool,
    peer_device_type: u8,
    port_number: u8,
    mcs: bool,
    ddps: bool,
    legacy_device_plug_status: bool,
    dpcd_revision: u8,
    peer_guid: __SlateRecord_uuid_h_17,
    num_sdp_streams: u8,
    num_sdp_stream_sinks: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_drm_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    eotf: u32,
    metadata_type: u32,
    display_primaries: [__SlateRecord_hdmi_h_214; 3],
    white_point: __SlateRecord_hdmi_h_217,
    max_display_mastering_luminance: u16,
    min_display_mastering_luminance: u16,
    max_cll: u16,
    max_fall: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_spd_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    vendor: [u8; 8],
    product: [u8; 16],
    sdi: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_audio_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    channels: u8,
    coding_type: u32,
    sample_size: u32,
    sample_frequency: u32,
    coding_type_ext: u32,
    channel_allocation: u8,
    level_shift_value: u8,
    downmix_inhibit: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_vendor_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    oui: u32,
    vic: u8,
    s3d_struct: i32,
    s3d_ext_data: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdr_static_metadata {
    eotf: u8,
    metadata_type: u8,
    max_cll: u16,
    max_fall: u16,
    min_cll: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdr_sink_metadata {
    metadata_type: u32,
    __slate_anon_1: __SlateRecord_hdmi_h_406,
}


#[repr(C)]
#[derive(Clone, Copy)]
union hdmi_vendor_any_infoframe {
    any: __SlateRecord_hdmi_h_419,
    hdmi: hdmi_vendor_infoframe,
}


#[repr(C)]
#[derive(Clone, Copy)]
union hdmi_infoframe {
    any: hdmi_any_infoframe,
    avi: hdmi_avi_infoframe,
    spd: hdmi_spd_infoframe,
    vendor: hdmi_vendor_any_infoframe,
    audio: hdmi_audio_infoframe,
    drm: hdmi_drm_infoframe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_object {
    id: u32,
    r#type: u32,
    properties: *mut drm_object_properties,
    refcount: kref,
    free_cb: Option<unsafe extern "C" fn(*mut kref)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_object_properties {
    count: i32,
    properties: [*mut drm_property; 64],
    values: [u64; 64],
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


// Link Status options
// extended-types: rather than continue to consume a bit per type,
// grab a chunk of the bits to use as integer type id.
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_fb_cmd2 {
    fb_id: u32,
    width: u32,
    height: u32,
    pixel_format: u32,
    flags: u32,
    handles: [u32; 4],
    pitches: [u32; 4],
    offsets: [u32; 4],
    modifier: [u64; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_create_dumb {
    height: u32,
    width: u32,
    bpp: u32,
    flags: u32,
    handle: u32,
    pitch: u32,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_remote_i2c_read_tx {
    i2c_dev_id: u8,
    num_bytes: u8,
    bytes: *mut u8,
    no_stop_bit: u8,
    i2c_transaction_delay: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_property {
    head: list_head,
    base: drm_mode_object,
    flags: u32,
    name: [u8; 32],
    num_values: u32,
    values: *mut u64,
    dev: *mut drm_device,
    enum_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_property_blob {
    base: drm_mode_object,
    dev: *mut drm_device,
    head_global: list_head,
    head_file: list_head,
    length: u64,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_scrambling {
    supported: bool,
    low_rates: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_scdc {
    supported: bool,
    read_request: bool,
    scrambling: drm_scrambling,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_hdmi_dsc_cap {
    v_1p2: bool,
    native_420: bool,
    all_bpp: bool,
    bpc_supported: u8,
    max_slices: u8,
    clk_per_slice: i32,
    max_lanes: u8,
    max_frl_rate_per_lane: u8,
    total_chunk_kbytes: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_hdmi_info {
    scdc: drm_scdc,
    y420_vdb_modes: [u64; 4],
    y420_cmdb_modes: [u64; 4],
    y420_dc_modes: u8,
    max_frl_rate_per_lane: u8,
    max_lanes: u8,
    dsc_cap: drm_hdmi_dsc_cap,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_monitor_range_info {
    min_vfreq: u16,
    max_vfreq: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_luminance_range_info {
    min_luminance: u32,
    max_luminance: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_amd_vsdb_info {
    version: u8,
    replay_mode: bool,
    panel_type: u8,
    luminance_range1: drm_luminance_range_info,
    luminance_range2: drm_luminance_range_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
union ack_req {
    conn_stat: drm_dp_connection_status_notify,
    port_num: drm_dp_port_number_req,
    resource_stat: drm_dp_resource_status_notify,
    query_payload: drm_dp_query_payload,
    allocate_payload: drm_dp_allocate_payload,
    dpcd_read: drm_dp_remote_dpcd_read,
    dpcd_write: drm_dp_remote_dpcd_write,
    i2c_read: drm_dp_remote_i2c_read,
    i2c_write: drm_dp_remote_i2c_write,
    enc_status: drm_dp_query_stream_enc_status,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_display_info {
    width_mm: u32,
    height_mm: u32,
    bpc: u32,
    subpixel_order: u32,
    panel_orientation: i32,
    color_formats: u32,
    bus_formats: *const u32,
    num_bus_formats: u32,
    bus_flags: u32,
    max_tmds_clock: i32,
    dvi_dual: bool,
    is_hdmi: bool,
    has_audio: bool,
    has_hdmi_infoframe: bool,
    rgb_quant_range_selectable: bool,
    edid_hdmi_rgb444_dc_modes: u8,
    edid_hdmi_ycbcr444_dc_modes: u8,
    cea_rev: u8,
    hdmi: drm_hdmi_info,
    hdr_sink_metadata: hdr_sink_metadata,
    non_desktop: bool,
    monitor_range: drm_monitor_range_info,
    luminance_range: drm_luminance_range_info,
    mso_stream_count: u8,
    mso_pixel_overlap: u8,
    max_dsc_bpp: u32,
    vics: *mut u8,
    vics_len: i32,
    quirks: u32,
    source_physical_address: u16,
    amd_vsdb: drm_amd_vsdb_info,
    panel_type: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union ack_replies {
    nak: drm_dp_nak_reply,
    link_addr: drm_dp_link_address_ack_reply,
    port_number: drm_dp_port_number_rep,
    path_resources: drm_dp_enum_path_resources_ack_reply,
    allocate_payload: drm_dp_allocate_payload_ack_reply,
    query_payload: drm_dp_query_payload_ack_reply,
    remote_dpcd_read_ack: drm_dp_remote_dpcd_read_ack_reply,
    remote_dpcd_write_ack: drm_dp_remote_dpcd_write_ack_reply,
    remote_dpcd_write_nack: drm_dp_remote_dpcd_write_nak_reply,
    remote_i2c_read_ack: drm_dp_remote_i2c_read_ack_reply,
    remote_i2c_read_nack: drm_dp_remote_i2c_read_nak_reply,
    remote_i2c_write_ack: drm_dp_remote_i2c_write_ack_reply,
    enc_status: drm_dp_query_stream_enc_status_ack_reply,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_tv_margins {
    bottom: u32,
    left: u32,
    right: u32,
    top: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_tv_connector_state {
    select_subconnector: u32,
    subconnector: u32,
    margins: drm_connector_tv_margins,
    legacy_mode: u32,
    mode: u32,
    brightness: u32,
    contrast: u32,
    flicker_reduction: u32,
    overscan: u32,
    saturation: u32,
    hue: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_infoframe {
    data: hdmi_infoframe,
    set: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_state {
    broadcast_rgb: u32,
    infoframes: __SlateRecord_drm_connector_h_1088,
    is_limited_range: bool,
    output_bpc: u32,
    output_format: u32,
    tmds_char_rate: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_state {
    connector: *mut drm_connector,
    crtc: *mut drm_crtc,
    best_encoder: *mut drm_encoder,
    link_status: u32,
    state: *mut drm_atomic_commit,
    commit: *mut drm_crtc_commit,
    tv: drm_tv_connector_state,
    self_refresh_aware: bool,
    picture_aspect_ratio: u32,
    content_type: u32,
    hdcp_content_type: u32,
    scaling_mode: u32,
    content_protection: u32,
    colorspace: u32,
    color_format: u32,
    writeback_job: *mut drm_writeback_job,
    max_requested_bpc: u8,
    max_bpc: u8,
    privacy_screen_sw_state: u32,
    hdr_output_metadata: *mut drm_property_blob,
    hdmi: drm_connector_hdmi_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_audio_funcs {
    startup: Option<unsafe extern "C" fn(*mut drm_connector) -> i32>,
    prepare: Option<unsafe extern "C" fn(*mut drm_connector, *mut hdmi_codec_daifmt, *mut hdmi_codec_params) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut drm_connector)>,
    mute_stream: Option<unsafe extern "C" fn(*mut drm_connector, bool, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_scanout_buffer {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_writeback_connector {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_fb_helper_surface_size {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_gem_object {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_buf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_buf_attachment {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_ioctl_desc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_cec_funcs {
    phys_addr_invalidate: Option<unsafe extern "C" fn(*mut drm_connector)>,
    phys_addr_set: Option<unsafe extern "C" fn(*mut drm_connector, u16)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_infoframe_funcs {
    clear_infoframe: Option<unsafe extern "C" fn(*mut drm_connector) -> i32>,
    write_infoframe: Option<unsafe extern "C" fn(*mut drm_connector, *const u8, u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_funcs {
    tmds_char_rate_valid: Option<unsafe extern "C" fn(*const drm_connector, *const drm_display_mode, u64) -> i32>,
    read_edid: Option<unsafe extern "C" fn(*mut drm_connector) -> *const drm_edid>,
    avi: drm_connector_infoframe_funcs,
    hdmi: drm_connector_infoframe_funcs,
    audio: drm_connector_infoframe_funcs,
    hdr_drm: drm_connector_infoframe_funcs,
    spd: drm_connector_infoframe_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_funcs {
    dpms: Option<unsafe extern "C" fn(*mut drm_connector, i32) -> i32>,
    reset: Option<unsafe extern "C" fn(*mut drm_connector)>,
    detect: Option<unsafe extern "C" fn(*mut drm_connector, bool) -> u32>,
    force: Option<unsafe extern "C" fn(*mut drm_connector)>,
    fill_modes: Option<unsafe extern "C" fn(*mut drm_connector, u32, u32) -> i32>,
    set_property: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_property, u64) -> i32>,
    late_register: Option<unsafe extern "C" fn(*mut drm_connector) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_connector)>,
    destroy: Option<unsafe extern "C" fn(*mut drm_connector)>,
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_connector) -> *mut drm_connector_state>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_connector) -> *mut drm_connector_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_connector_state)>,
    atomic_set_property: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_connector_state, *mut drm_property, u64) -> i32>,
    atomic_get_property: Option<unsafe extern "C" fn(*mut drm_connector, *const drm_connector_state, *mut drm_property, *mut u64) -> i32>,
    atomic_print_state: Option<unsafe extern "C" fn(*mut drm_printer, *const drm_connector_state)>,
    oob_hotplug_event: Option<unsafe extern "C" fn(*mut drm_connector, u32)>,
    debugfs_init: Option<unsafe extern "C" fn(*mut drm_connector, *mut dentry)>,
    color_format: Option<unsafe extern "C" fn(*const drm_connector_state) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_cmdline_mode {
    name: [u8; 32],
    specified: bool,
    refresh_specified: bool,
    bpp_specified: bool,
    pixel_clock: u32,
    xres: i32,
    yres: i32,
    bpp: i32,
    refresh: i32,
    rb: bool,
    interlace: bool,
    cvt: bool,
    margins: bool,
    force: u32,
    rotation_reflection: u32,
    panel_orientation: i32,
    tv_margins: drm_connector_tv_margins,
    tv_mode: u32,
    tv_mode_specified: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_edid_h_126 {
    gtf2: __SlateRecord_drm_edid_h_127,
    cvt: __SlateRecord_drm_edid_h_135,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_h_127 {
    reserved: u8,
    hfreq_start_khz: u8,
    c: u8,
    m: u16,
    k: u8,
    j: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_h_135 {
    version: u8,
    data1: u8,
    data2: u8,
    supported_aspects: u8,
    flags: u8,
    supported_scalings: u8,
    preferred_refresh: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_audio {
    funcs: *const drm_connector_hdmi_audio_funcs,
    codec_pdev: *mut platform_device,
    lock: mutex,
    plugged_cb: Option<unsafe extern "C" fn(*mut device, bool)>,
    plugged_cb_dev: *mut device,
    last_state: bool,
    dai_port: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi {
    vendor: [u8; 8],
    product: [u8; 16],
    supported_formats: u64,
    funcs: *const drm_connector_hdmi_funcs,
    infoframes: __SlateRecord_drm_connector_h_2062,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_cec {
    mutex: mutex,
    funcs: *const drm_connector_cec_funcs,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_edid_h_171 {
    str: detailed_data_string,
    range: detailed_data_monitor_range,
    color: detailed_data_wpindex,
    timings: [std_timing; 6],
    cvt: [cvt_timing; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector {
    dev: *mut drm_device,
    kdev: *mut device,
    attr: *mut device_attribute,
    fwnode: *mut fwnode_handle,
    head: list_head,
    global_connector_list_entry: list_head,
    base: drm_mode_object,
    name: *mut u8,
    mutex: mutex,
    index: u32,
    connector_type: i32,
    connector_type_id: i32,
    interlace_allowed: bool,
    doublescan_allowed: bool,
    stereo_allowed: bool,
    ycbcr_420_allowed: bool,
    registration_state: u32,
    modes: list_head,
    status: u32,
    probed_modes: list_head,
    display_info: drm_display_info,
    funcs: *const drm_connector_funcs,
    edid_blob_ptr: *mut drm_property_blob,
    properties: drm_object_properties,
    scaling_mode_property: *mut drm_property,
    vrr_capable_property: *mut drm_property,
    colorspace_property: *mut drm_property,
    color_format_property: *mut drm_property,
    path_blob_ptr: *mut drm_property_blob,
    max_bpc: u32,
    max_bpc_property: *mut drm_property,
    privacy_screen: *mut drm_privacy_screen,
    privacy_screen_notifier: notifier_block,
    privacy_screen_sw_state_property: *mut drm_property,
    privacy_screen_hw_state_property: *mut drm_property,
    broadcast_rgb_property: *mut drm_property,
    polled: u8,
    dpms: i32,
    helper_private: *const drm_connector_helper_funcs,
    cmdline_mode: drm_cmdline_mode,
    force: u32,
    edid_override: *const drm_edid,
    edid_override_mutex: mutex,
    epoch_counter: u64,
    possible_encoders: u32,
    encoder: *mut drm_encoder,
    eld: [u8; 128],
    eld_mutex: mutex,
    latency_present: [bool; 2],
    video_latency: [i32; 2],
    audio_latency: [i32; 2],
    ddc: *mut i2c_adapter,
    null_edid_counter: i32,
    bad_edid_counter: u32,
    edid_corrupt: bool,
    real_edid_checksum: u8,
    debugfs_entry: *mut dentry,
    state: *mut drm_connector_state,
    tile_blob_ptr: *mut drm_property_blob,
    has_tile: bool,
    tile_group: *mut drm_tile_group,
    tile_is_single_monitor: bool,
    num_h_tile: u8,
    num_v_tile: u8,
    tile_h_loc: u8,
    tile_v_loc: u8,
    tile_h_size: u16,
    tile_v_size: u16,
    free_node: llist_node,
    hdmi: drm_connector_hdmi,
    hdmi_audio: drm_connector_hdmi_audio,
    cec: drm_connector_cec,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_edid_h_192 {
    pixel_data: detailed_pixel_timing,
    other_data: detailed_non_pixel,
}


fn drm_connector_index(connector: *const drm_connector) -> u32 {
unsafe { (*connector).index }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_edid_h_294 {
    product_id: drm_edid_product_id,
    __slate_anon_1: __SlateRecord_drm_edid_h_296,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_edid_h_296 {
    mfg_id: [u8; 2],
    prod_code: [u8; 2],
    serial: u32,
    mfg_week: u8,
    mfg_year: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_print_h_181 {
    series: u32,
    counter: u32,
}


fn drm_connector_put(connector: *mut drm_connector) {
unsafe { drm_mode_object_put(core::ptr::addr_of_mut!((*connector).base)) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_123 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_148 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_168 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_454_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_454_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_454_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_454_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_454_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_454_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_454_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_454_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_454_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_454_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_456_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_456_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_456_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_456_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_456_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_456_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_456_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_456_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_456_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_456_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_594_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_594_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_594_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_594_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_594_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_594_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_594_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_594_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_594_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_594_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_598_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_598_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_598_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_598_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_598_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_598_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_598_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_598_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_598_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_598_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_685 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_794 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_795 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_809 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_1032 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_5996 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_mst_topology_c_5996_2 {
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
    pub struct __SlateBits288U0 {
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
    pub struct __SlateBits622U0 {
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
    pub struct __SlateBits452U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits506U0 {
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
    pub struct __SlateBits503U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits506U1 {
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
    pub struct __SlateBits506U2 {
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
    pub struct __SlateBits506U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits479U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits481U0 {
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
    pub struct __SlateBits131U0 {
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
    pub struct __SlateBits421U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits74U0 {
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
    pub struct __SlateBits74U1 {
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
    pub struct __SlateBits833U0 {
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
    pub struct __SlateBits291U0 {
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
    pub struct __SlateBits74U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits124U0 {
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
    pub struct __SlateBits132U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits971U0 {
        #[bits(1)]
        pub allow_modeset: bool,
        #[bits(1)]
        pub legacy_cursor_update: bool,
        #[bits(1)]
        pub async_update: bool,
        #[bits(1)]
        pub duplicated: bool,
        #[bits(1)]
        pub checked: bool,
        #[bits(1)]
        pub plane_color_pipeline: bool,
        #[bits(2, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1030U0 {
        #[bits(1)]
        pub planes_changed: bool,
        #[bits(1)]
        pub mode_changed: bool,
        #[bits(1)]
        pub active_changed: bool,
        #[bits(1)]
        pub connectors_changed: bool,
        #[bits(1)]
        pub zpos_changed: bool,
        #[bits(1)]
        pub color_mgmt_changed: bool,
        #[bits(2, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1018U0 {
        #[bits(1)]
        pub color_mgmt_changed: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1046U0 {
        #[bits(1)]
        pub mst_state: bool,
        #[bits(1)]
        pub payload_id_table_cleared: bool,
        #[bits(1)]
        pub reset_rx_state: bool,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1080U0 {
        #[bits(1)]
        pub delete: bool,
        #[bits(1)]
        pub dsc_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits782U0 {
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
    pub struct __SlateBits783U0 {
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
    pub struct __SlateBits787U0 {
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
    pub struct __SlateBits789U0 {
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
    pub struct __SlateBits791U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub explicit_set: u8,
        #[bits(6)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits797U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub notifier_present: u8,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits793U0 {
        #[bits(8)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits795U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(7)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits776U0 {
        #[bits(1)]
        pub enabled: bool,
        #[bits(1)]
        pub demand_offline: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits709U0 {
        #[bits(18)]
        pub lineno: u32,
        #[bits(6)]
        pub class_id: u32,
        #[bits(8)]
        pub flags: u32,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_tile_group {
    refcount: kref,
    dev: *mut drm_device,
    id: i32,
    group_data: [u8; 9],
}


fn drm_dp_is_uhbr_rate(link_rate: i32) -> bool {
link_rate >= 1000000
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_aux_msg {
    address: u32,
    request: u8,
    reply: u8,
    buffer: *mut (),
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_aux_cec {
    lock: mutex,
    adap: *mut cec_adapter,
    connector: *mut drm_connector,
    unregister_work: delayed_work,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_aux {
    name: *const u8,
    ddc: i2c_adapter,
    dev: *mut device,
    drm_dev: *mut drm_device,
    crtc: *mut drm_crtc,
    hw_mutex: mutex,
    crc_work: work_struct,
    crc_count: u8,
    transfer: Option<unsafe extern "C" fn(*mut drm_dp_aux, *mut drm_dp_aux_msg) -> i64>,
    wait_hpd_asserted: Option<unsafe extern "C" fn(*mut drm_dp_aux, u64) -> i32>,
    i2c_nack_count: u32,
    i2c_defer_count: u32,
    cec: drm_dp_aux_cec,
    is_remote: bool,
    powered_down: bool,
    no_zero_sized: bool,
    dpcd_probe_disabled: bool,
}


fn drm_dp_dpcd_readb(aux: *mut drm_dp_aux, offset: u32, valuep: *mut u8) -> i64 {
unsafe { drm_dp_dpcd_read(aux, offset, valuep as *mut (), 1u64) }
}

fn drm_dp_dpcd_read_data(aux: *mut drm_dp_aux, offset: u32, buffer: *mut (), size: u64) -> i32 {
let mut ret: i32 = 0;
let buf: *mut u8 = buffer as *mut u8;
ret = (unsafe { drm_dp_dpcd_read(aux, offset, buffer, size) }) as i32;
if ret >= 0 {
if (ret as u64) < size {
return 71i32.wrapping_neg();
}
return 0;
}
// Workaround for USB-C hubs/adapters with buggy firmware that fail
// multi-byte AUX reads but work with single-byte reads.
// Known affected devices:
// - Lenovo USB-C to VGA adapter (VIA VL817, idVendor=17ef, idProduct=7217)
// - Dell DA310 USB-C hub (idVendor=413c, idProduct=c010)
// Attempt byte-by-byte reading as a fallback.
for i in 0..size {
ret = drm_dp_dpcd_readb(aux, (offset as u64).wrapping_add(i) as u32, buf.wrapping_offset(i as isize)) as i32;
if ret < 0 {
return ret;
}
}
0
}

fn drm_dp_dpcd_write_data(aux: *mut drm_dp_aux, offset: u32, buffer: *mut (), size: u64) -> i32 {
let ret: i32 = (unsafe { drm_dp_dpcd_write(aux, offset, buffer, size) }) as i32;
if ret < 0 {
return ret;
}
if (ret as u64) < size {
return 71i32.wrapping_neg();
}
0
}

fn drm_dp_dpcd_read_byte(aux: *mut drm_dp_aux, offset: u32, valuep: *mut u8) -> i32 {
drm_dp_dpcd_read_data(aux, offset, valuep as *mut (), 1u64)
}

fn drm_dp_dpcd_write_byte(aux: *mut drm_dp_aux, offset: u32, mut value: u8) -> i32 {
drm_dp_dpcd_write_data(aux, offset, core::ptr::addr_of_mut!(value) as *mut (), 1u64)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_dpcd_ident {
    oui: [u8; 3],
    device_id: [u8; 6],
    hw_rev: u8,
    sw_major_rev: u8,
    sw_minor_rev: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_desc {
    ident: drm_dp_dpcd_ident,
    quirks: u32,
}


fn drm_dp_has_quirk(desc: &drm_dp_desc, quirk: u32) -> bool {
((unsafe { (*(desc as *const drm_dp_desc)).quirks }) as u64) & BIT__2!(quirk) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ww_mutex {
    base: mutex,
    ctx: *mut ww_acquire_ctx,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ww_acquire_ctx {
    task: *mut task_struct,
    stamp: u64,
    acquired: u32,
    wounded: u16,
    is_wait_die: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_modeset_acquire_ctx {
    ww_ctx: ww_acquire_ctx,
    contended: *mut drm_modeset_lock,
    stack_depot: u32,
    locked: list_head,
    trylock_only: bool,
    interruptible: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_modeset_lock {
    mutex: ww_mutex,
    head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_display_mode {
    clock: i32,
    hdisplay: u16,
    hsync_start: u16,
    hsync_end: u16,
    htotal: u16,
    hskew: u16,
    vdisplay: u16,
    vsync_start: u16,
    vsync_end: u16,
    vtotal: u16,
    vscan: u16,
    flags: u32,
    crtc_clock: i32,
    crtc_hdisplay: u16,
    crtc_hblank_start: u16,
    crtc_hblank_end: u16,
    crtc_hsync_start: u16,
    crtc_hsync_end: u16,
    crtc_htotal: u16,
    crtc_hskew: u16,
    crtc_vdisplay: u16,
    crtc_vblank_start: u16,
    crtc_vblank_end: u16,
    crtc_vsync_start: u16,
    crtc_vsync_end: u16,
    crtc_vtotal: u16,
    width_mm: u16,
    height_mm: u16,
    r#type: u8,
    expose_to_userspace: bool,
    head: list_head,
    name: [u8; 32],
    status: i32,
    picture_aspect_ratio: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_config_funcs {
    fb_create: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file, *const drm_format_info, *const drm_mode_fb_cmd2) -> *mut drm_framebuffer>,
    get_format_info: Option<unsafe extern "C" fn(u32, u64) -> *const drm_format_info>,
    mode_valid: Option<unsafe extern "C" fn(*mut drm_device, *const drm_display_mode) -> i32>,
    atomic_check: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit) -> i32>,
    atomic_commit: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit, bool) -> i32>,
    atomic_state_alloc: Option<unsafe extern "C" fn(*mut drm_device) -> *mut drm_atomic_commit>,
    atomic_state_clear: Option<unsafe extern "C" fn(*mut drm_atomic_commit)>,
    atomic_state_free: Option<unsafe extern "C" fn(*mut drm_atomic_commit)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_config {
    mutex: mutex,
    connection_mutex: drm_modeset_lock,
    acquire_ctx: *mut drm_modeset_acquire_ctx,
    idr_mutex: mutex,
    object_idr: idr,
    tile_idr: idr,
    fb_lock: mutex,
    num_fb: i32,
    fb_list: list_head,
    connector_list_lock: spinlock,
    num_connector: i32,
    connector_ida: ida,
    connector_list: list_head,
    connector_free_list: llist_head,
    connector_free_work: work_struct,
    num_encoder: i32,
    encoder_list: list_head,
    num_total_plane: i32,
    plane_list: list_head,
    panic_lock: raw_spinlock,
    num_colorop: i32,
    colorop_list: list_head,
    num_crtc: i32,
    crtc_list: list_head,
    property_list: list_head,
    privobj_list: list_head,
    min_width: u32,
    min_height: u32,
    max_width: u32,
    max_height: u32,
    funcs: *const drm_mode_config_funcs,
    poll_enabled: bool,
    poll_running: bool,
    delayed_event: bool,
    output_poll_work: delayed_work,
    blob_lock: mutex,
    property_blob_list: list_head,
    edid_property: *mut drm_property,
    dpms_property: *mut drm_property,
    path_property: *mut drm_property,
    tile_property: *mut drm_property,
    panel_type_property: *mut drm_property,
    link_status_property: *mut drm_property,
    plane_type_property: *mut drm_property,
    prop_src_x: *mut drm_property,
    prop_src_y: *mut drm_property,
    prop_src_w: *mut drm_property,
    prop_src_h: *mut drm_property,
    prop_crtc_x: *mut drm_property,
    prop_crtc_y: *mut drm_property,
    prop_crtc_w: *mut drm_property,
    prop_crtc_h: *mut drm_property,
    prop_fb_id: *mut drm_property,
    prop_in_fence_fd: *mut drm_property,
    prop_out_fence_ptr: *mut drm_property,
    prop_crtc_id: *mut drm_property,
    prop_fb_damage_clips: *mut drm_property,
    prop_active: *mut drm_property,
    prop_mode_id: *mut drm_property,
    prop_vrr_enabled: *mut drm_property,
    dvi_i_subconnector_property: *mut drm_property,
    dvi_i_select_subconnector_property: *mut drm_property,
    dp_subconnector_property: *mut drm_property,
    tv_subconnector_property: *mut drm_property,
    tv_select_subconnector_property: *mut drm_property,
    legacy_tv_mode_property: *mut drm_property,
    tv_mode_property: *mut drm_property,
    tv_left_margin_property: *mut drm_property,
    tv_right_margin_property: *mut drm_property,
    tv_top_margin_property: *mut drm_property,
    tv_bottom_margin_property: *mut drm_property,
    tv_brightness_property: *mut drm_property,
    tv_contrast_property: *mut drm_property,
    tv_flicker_reduction_property: *mut drm_property,
    tv_overscan_property: *mut drm_property,
    tv_saturation_property: *mut drm_property,
    tv_hue_property: *mut drm_property,
    scaling_mode_property: *mut drm_property,
    aspect_ratio_property: *mut drm_property,
    content_type_property: *mut drm_property,
    degamma_lut_property: *mut drm_property,
    degamma_lut_size_property: *mut drm_property,
    ctm_property: *mut drm_property,
    gamma_lut_property: *mut drm_property,
    gamma_lut_size_property: *mut drm_property,
    background_color_property: *mut drm_property,
    suggested_x_property: *mut drm_property,
    suggested_y_property: *mut drm_property,
    non_desktop_property: *mut drm_property,
    panel_orientation_property: *mut drm_property,
    writeback_fb_id_property: *mut drm_property,
    writeback_pixel_formats_property: *mut drm_property,
    writeback_out_fence_ptr_property: *mut drm_property,
    hdr_output_metadata_property: *mut drm_property,
    content_protection_property: *mut drm_property,
    hdcp_content_type_property: *mut drm_property,
    preferred_depth: u32,
    prefer_shadow: u32,
    quirk_addfb_prefer_xbgr_30bpp: bool,
    quirk_addfb_prefer_host_byte_order: bool,
    async_page_flip: bool,
    fb_modifiers_not_supported: bool,
    normalize_zpos: bool,
    modifiers_property: *mut drm_property,
    async_modifiers_property: *mut drm_property,
    size_hints_property: *mut drm_property,
    cursor_width: u32,
    cursor_height: u32,
    suspend_state: *mut drm_atomic_commit,
    helper_private: *const drm_mode_config_helper_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_device {
    if_version: i32,
    r#ref: kref,
    dev: *mut device,
    dma_dev: *mut device,
    managed: __SlateRecord_drm_device_h_113,
    driver: *const drm_driver,
    dev_private: *mut (),
    primary: *mut drm_minor,
    render: *mut drm_minor,
    accel: *mut drm_minor,
    registered: bool,
    master: *mut drm_master,
    driver_features: u32,
    unplugged: bool,
    anon_inode: *mut inode,
    unique: *mut u8,
    master_mutex: mutex,
    open_count: __SlateRecord_types_h_190,
    filelist_mutex: mutex,
    filelist: list_head,
    filelist_internal: list_head,
    clientlist_mutex: mutex,
    clientlist: list_head,
    client_sysrq_list: list_head,
    vblank_disable_immediate: bool,
    vblank: *mut drm_vblank_crtc,
    vblank_time_lock: spinlock,
    vbl_lock: spinlock,
    max_vblank_count: u32,
    vblank_event_list: list_head,
    event_lock: spinlock,
    num_crtcs: u32,
    mode_config: drm_mode_config,
    object_name_lock: mutex,
    object_name_idr: idr,
    vma_offset_manager: *mut drm_vma_offset_manager,
    vram_mm: *mut drm_vram_mm,
    switch_power_state: u32,
    fb_helper: *mut drm_fb_helper,
    debugfs_root: *mut dentry,
    gem_lru_mutex: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmsg_dump_detail {
    reason: u32,
    description: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmsg_dumper {
    list: list_head,
    dump: Option<unsafe extern "C" fn(*mut kmsg_dumper, *mut kmsg_dump_detail)>,
    max_reason: u32,
    registered: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_rect {
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane_state {
    plane: *mut drm_plane,
    crtc: *mut drm_crtc,
    fb: *mut drm_framebuffer,
    fence: *mut dma_fence,
    crtc_x: i32,
    crtc_y: i32,
    crtc_w: u32,
    crtc_h: u32,
    src_x: u32,
    src_y: u32,
    src_h: u32,
    src_w: u32,
    hotspot_x: i32,
    hotspot_y: i32,
    alpha: u16,
    pixel_blend_mode: u16,
    rotation: u32,
    zpos: u32,
    normalized_zpos: u32,
    color_encoding: u32,
    color_range: u32,
    fb_damage_clips: *mut drm_property_blob,
    ignore_damage_clips: bool,
    src: drm_rect,
    dst: drm_rect,
    visible: bool,
    scaling_filter: u32,
    color_pipeline: *mut drm_colorop,
    commit: *mut drm_crtc_commit,
    state: *mut drm_atomic_commit,
    __slate_bits_0: __slate_bits::__SlateBits1018U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane_funcs {
    update_plane: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_crtc, *mut drm_framebuffer, i32, i32, u32, u32, u32, u32, u32, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    disable_plane: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_modeset_acquire_ctx) -> i32>,
    destroy: Option<unsafe extern "C" fn(*mut drm_plane)>,
    reset: Option<unsafe extern "C" fn(*mut drm_plane)>,
    set_property: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_property, u64) -> i32>,
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_plane) -> *mut drm_plane_state>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_plane) -> *mut drm_plane_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state)>,
    atomic_set_property: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state, *mut drm_property, u64) -> i32>,
    atomic_get_property: Option<unsafe extern "C" fn(*mut drm_plane, *const drm_plane_state, *mut drm_property, *mut u64) -> i32>,
    late_register: Option<unsafe extern "C" fn(*mut drm_plane) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_plane)>,
    atomic_print_state: Option<unsafe extern "C" fn(*mut drm_printer, *const drm_plane_state)>,
    format_mod_supported: Option<unsafe extern "C" fn(*mut drm_plane, u32, u64) -> bool>,
    format_mod_supported_async: Option<unsafe extern "C" fn(*mut drm_plane, u32, u64) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane {
    dev: *mut drm_device,
    head: list_head,
    name: *mut u8,
    mutex: drm_modeset_lock,
    base: drm_mode_object,
    possible_crtcs: u32,
    format_types: *mut u32,
    format_count: u32,
    format_default: bool,
    modifiers: *mut u64,
    modifier_count: u32,
    crtc: *mut drm_crtc,
    fb: *mut drm_framebuffer,
    old_fb: *mut drm_framebuffer,
    funcs: *const drm_plane_funcs,
    properties: drm_object_properties,
    r#type: u32,
    index: u32,
    helper_private: *const drm_plane_helper_funcs,
    state: *mut drm_plane_state,
    alpha_property: *mut drm_property,
    zpos_property: *mut drm_property,
    rotation_property: *mut drm_property,
    blend_mode_property: *mut drm_property,
    color_encoding_property: *mut drm_property,
    color_range_property: *mut drm_property,
    color_pipeline_property: *mut drm_property,
    scaling_filter_property: *mut drm_property,
    hotspot_x_property: *mut drm_property,
    hotspot_y_property: *mut drm_property,
    kmsg_panic: kmsg_dumper,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_crc_entry {
    has_frame_counter: bool,
    frame: u32,
    crcs: [u32; 10],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_crc {
    lock: spinlock,
    source: *const u8,
    opened: bool,
    overflow: bool,
    entries: *mut drm_crtc_crc_entry,
    head: i32,
    tail: i32,
    values_cnt: u64,
    wq: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_state {
    crtc: *mut drm_crtc,
    enable: bool,
    active: bool,
    __slate_bits_0: __slate_bits::__SlateBits1030U0,
    no_vblank: bool,
    plane_mask: u32,
    connector_mask: u32,
    encoder_mask: u32,
    adjusted_mode: drm_display_mode,
    mode: drm_display_mode,
    mode_blob: *mut drm_property_blob,
    degamma_lut: *mut drm_property_blob,
    ctm: *mut drm_property_blob,
    gamma_lut: *mut drm_property_blob,
    background_color: u64,
    target_vblank: u32,
    async_flip: bool,
    vrr_enabled: bool,
    self_refresh_active: bool,
    scaling_filter: u32,
    sharpness_strength: u8,
    event: *mut drm_pending_vblank_event,
    commit: *mut drm_crtc_commit,
    state: *mut drm_atomic_commit,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_funcs {
    reset: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    cursor_set: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_file, u32, u32, u32) -> i32>,
    cursor_set2: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_file, u32, u32, u32, i32, i32) -> i32>,
    cursor_move: Option<unsafe extern "C" fn(*mut drm_crtc, i32, i32) -> i32>,
    gamma_set: Option<unsafe extern "C" fn(*mut drm_crtc, *mut u16, *mut u16, *mut u16, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    destroy: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    set_config: Option<unsafe extern "C" fn(*mut drm_mode_set, *mut drm_modeset_acquire_ctx) -> i32>,
    page_flip: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_framebuffer, *mut drm_pending_vblank_event, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    page_flip_target: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_framebuffer, *mut drm_pending_vblank_event, u32, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    set_property: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_property, u64) -> i32>,
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_crtc) -> *mut drm_crtc_state>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_crtc) -> *mut drm_crtc_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_crtc_state)>,
    atomic_set_property: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_crtc_state, *mut drm_property, u64) -> i32>,
    atomic_get_property: Option<unsafe extern "C" fn(*mut drm_crtc, *const drm_crtc_state, *mut drm_property, *mut u64) -> i32>,
    late_register: Option<unsafe extern "C" fn(*mut drm_crtc) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    set_crc_source: Option<unsafe extern "C" fn(*mut drm_crtc, *const u8) -> i32>,
    verify_crc_source: Option<unsafe extern "C" fn(*mut drm_crtc, *const u8, *mut u64) -> i32>,
    get_crc_sources: Option<unsafe extern "C" fn(*mut drm_crtc, *mut u64) -> *const *const u8>,
    atomic_print_state: Option<unsafe extern "C" fn(*mut drm_printer, *const drm_crtc_state)>,
    get_vblank_counter: Option<unsafe extern "C" fn(*mut drm_crtc) -> u32>,
    enable_vblank: Option<unsafe extern "C" fn(*mut drm_crtc) -> i32>,
    disable_vblank: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    get_vblank_timestamp: Option<unsafe extern "C" fn(*mut drm_crtc, *mut i32, *mut i64, bool) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc {
    dev: *mut drm_device,
    port: *mut device_node,
    head: list_head,
    name: *mut u8,
    mutex: drm_modeset_lock,
    base: drm_mode_object,
    primary: *mut drm_plane,
    cursor: *mut drm_plane,
    index: u32,
    cursor_x: i32,
    cursor_y: i32,
    enabled: bool,
    mode: drm_display_mode,
    hwmode: drm_display_mode,
    x: i32,
    y: i32,
    funcs: *const drm_crtc_funcs,
    gamma_size: u32,
    gamma_store: *mut u16,
    helper_private: *const drm_crtc_helper_funcs,
    properties: drm_object_properties,
    scaling_filter_property: *mut drm_property,
    sharpness_strength_property: *mut drm_property,
    state: *mut drm_crtc_state,
    commit_list: list_head,
    commit_lock: spinlock,
    debugfs_entry: *mut dentry,
    crc: drm_crtc_crc,
    fence_context: u32,
    fence_lock: spinlock,
    fence_seqno: u64,
    timeline_name: [u8; 32],
    self_refresh_data: *mut drm_self_refresh_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_set {
    fb: *mut drm_framebuffer,
    crtc: *mut drm_crtc,
    mode: *mut drm_display_mode,
    x: u32,
    y: u32,
    connectors: *mut *mut drm_connector,
    num_connectors: u64,
}


fn drm_crtc_index(crtc: *const drm_crtc) -> u32 {
unsafe { (*crtc).index }
}

fn drm_crtc_mask(crtc: *const drm_crtc) -> u32 {
(1i32 << drm_crtc_index(crtc)) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_commit {
    crtc: *mut drm_crtc,
    r#ref: kref,
    flip_done: completion,
    hw_done: completion,
    cleanup_done: completion,
    commit_entry: list_head,
    event: *mut drm_pending_vblank_event,
    abort_completion: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __drm_colorops_state {
    ptr: *mut drm_colorop,
    state: *mut drm_colorop_state,
    old_state: *mut drm_colorop_state,
    new_state: *mut drm_colorop_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __drm_planes_state {
    ptr: *mut drm_plane,
    state_to_destroy: *mut drm_plane_state,
    old_state: *mut drm_plane_state,
    new_state: *mut drm_plane_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __drm_crtcs_state {
    ptr: *mut drm_crtc,
    state_to_destroy: *mut drm_crtc_state,
    old_state: *mut drm_crtc_state,
    new_state: *mut drm_crtc_state,
    commit: *mut drm_crtc_commit,
    out_fence_ptr: *mut i32,
    last_vblank_count: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __drm_connnectors_state {
    ptr: *mut drm_connector,
    state_to_destroy: *mut drm_connector_state,
    old_state: *mut drm_connector_state,
    new_state: *mut drm_connector_state,
    out_fence_ptr: *mut i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_private_state_funcs {
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_private_obj) -> *mut drm_private_state>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_private_obj) -> *mut drm_private_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_private_obj, *mut drm_private_state)>,
    atomic_print_state: Option<unsafe extern "C" fn(*mut drm_printer, *const drm_private_state)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_private_obj {
    dev: *mut drm_device,
    head: list_head,
    lock: drm_modeset_lock,
    state: *mut drm_private_state,
    funcs: *const drm_private_state_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_private_state {
    state: *mut drm_atomic_commit,
    obj: *mut drm_private_obj,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __drm_private_objs_state {
    ptr: *mut drm_private_obj,
    state_to_destroy: *mut drm_private_state,
    old_state: *mut drm_private_state,
    new_state: *mut drm_private_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_atomic_commit {
    r#ref: kref,
    dev: *mut drm_device,
    __slate_bits_0: __slate_bits::__SlateBits971U0,
    colorops: *mut __drm_colorops_state,
    planes: *mut __drm_planes_state,
    crtcs: *mut __drm_crtcs_state,
    num_connector: i32,
    connectors: *mut __drm_connnectors_state,
    num_private_objs: i32,
    private_objs: *mut __drm_private_objs_state,
    acquire_ctx: *mut drm_modeset_acquire_ctx,
    fake_commit: *mut drm_crtc_commit,
    commit_work: work_struct,
}


fn drm_crtc_commit_get(commit: *mut drm_crtc_commit) -> *mut drm_crtc_commit {
kref_get(unsafe { core::ptr::addr_of_mut!((*commit).r#ref) });
commit
}

fn drm_crtc_commit_put(commit: *mut drm_crtc_commit) {
kref_put(unsafe { core::ptr::addr_of_mut!((*commit).r#ref) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut kref)>>(__drm_crtc_commit_free as *const ()) });
}

fn drm_atomic_get_new_crtc_state(state: *const drm_atomic_commit, crtc: *mut drm_crtc) -> *mut drm_crtc_state {
unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(drm_crtc_index(crtc as *const drm_crtc) as isize)).new_state }
}

fn drm_atomic_get_old_connector_state(state: *const drm_atomic_commit, connector: *mut drm_connector) -> *mut drm_connector_state {
let index: i32 = drm_connector_index(connector as *const drm_connector) as i32;
if index >= unsafe { (*state).num_connector } {
return core::ptr::null_mut::<drm_connector_state>();
}
unsafe { (*unsafe { (*state).connectors }.wrapping_offset(index as isize)).old_state }
}

fn drm_atomic_get_new_connector_state(state: *const drm_atomic_commit, connector: *mut drm_connector) -> *mut drm_connector_state {
let index: i32 = drm_connector_index(connector as *const drm_connector) as i32;
if index >= unsafe { (*state).num_connector } {
return core::ptr::null_mut::<drm_connector_state>();
}
unsafe { (*unsafe { (*state).connectors }.wrapping_offset(index as isize)).new_state }
}

// for_each_new_crtc_in_state - iterate over all CRTCs in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @crtc: &struct drm_crtc iteration cursor
// @new_crtc_state: &struct drm_crtc_state iteration cursor for the new state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all CRTCs in an atomic update, tracking only the new
// state. This is useful in enable functions, where we need the new state the
// hardware should be in when the atomic commit operation has completed.
// Only to avoid unused-but-set-variable warning
// Only to avoid unused-but-set-variable warning
fn drm_atomic_crtc_needs_modeset(state: *const drm_crtc_state) -> bool {
((unsafe { (*state).__slate_bits_0.__get_mode_changed() }) as i32) != 0 || ((unsafe { (*state).__slate_bits_0.__get_active_changed() }) as i32) != 0 || ((unsafe { (*state).__slate_bits_0.__get_connectors_changed() }) as i32) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union dfixed {
    full: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_mst_port {
    topology_kref: kref,
    malloc_kref: kref,
    port_num: u8,
    input: bool,
    mcs: bool,
    ddps: bool,
    pdt: u8,
    ldps: bool,
    dpcd_rev: u8,
    num_sdp_streams: u8,
    num_sdp_stream_sinks: u8,
    full_pbn: u16,
    next: list_head,
    mstb: *mut drm_dp_mst_branch,
    aux: drm_dp_aux,
    passthrough_aux: *mut drm_dp_aux,
    parent: *mut drm_dp_mst_branch,
    connector: *mut drm_connector,
    mgr: *mut drm_dp_mst_topology_mgr,
    cached_edid: *const drm_edid,
    fec_capable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_sideband_msg_hdr {
    lct: u8,
    lcr: u8,
    rad: [u8; 8],
    broadcast: bool,
    path_msg: bool,
    msg_len: u8,
    somt: bool,
    eomt: bool,
    seqno: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_sideband_msg_rx {
    chunk: [u8; 48],
    msg: [u8; 256],
    curchunk_len: u8,
    curchunk_idx: u8,
    curchunk_hdrlen: u8,
    curlen: u8,
    have_somt: bool,
    have_eomt: bool,
    initial_hdr: drm_dp_sideband_msg_hdr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_mst_branch {
    topology_kref: kref,
    malloc_kref: kref,
    destroy_next: list_head,
    rad: [u8; 8],
    lct: u8,
    num_ports: i32,
    ports: list_head,
    port_parent: *mut drm_dp_mst_port,
    mgr: *mut drm_dp_mst_topology_mgr,
    link_address_sent: bool,
    guid: __SlateRecord_uuid_h_17,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_nak_reply {
    guid: __SlateRecord_uuid_h_17,
    reason: u8,
    nak_data: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_link_address_ack_reply {
    guid: __SlateRecord_uuid_h_17,
    nports: u8,
    ports: [drm_dp_link_addr_reply_port; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_remote_dpcd_read_ack_reply {
    port_number: u8,
    num_bytes: u8,
    bytes: [u8; 255],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_remote_dpcd_write_ack_reply {
    port_number: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_remote_dpcd_write_nak_reply {
    port_number: u8,
    reason: u8,
    bytes_written_before_failure: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_remote_i2c_read_ack_reply {
    port_number: u8,
    num_bytes: u8,
    bytes: [u8; 255],
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_get_port_malloc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_put_port_malloc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_get_mst_payload_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_connector_late_register: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_connector_early_unregister: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_send_power_updown_phy: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_send_query_stream_enc_status: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_add_payload_part1: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_remove_payload_part1: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_remove_payload_part2: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_add_payload_part2: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_get_vc_payload_bw: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_read_mst_cap: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_topology_mgr_set_mst: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_topology_queue_probe: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_topology_mgr_suspend: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_topology_mgr_resume: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_hpd_irq_handle_event: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_hpd_irq_send_new_request: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_detect_port: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_edid_read: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_get_edid: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_atomic_find_time_slots: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_atomic_release_time_slots: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_atomic_setup_commit: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_atomic_wait_for_dependencies: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_root_conn_atomic_check: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_update_slots: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_check_act_status: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {28} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_calc_pbn_mode: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {29} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_dump_topology: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {30} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_port_downstream_of_parent: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {31} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_add_affected_dsc_crtcs: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {32} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_atomic_enable_dsc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {33} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_atomic_check_mgr: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {34} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_atomic_check: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {35} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_topology_state_funcs: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {36} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_get_mst_topology_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {37} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_get_old_mst_topology_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {38} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_get_new_mst_topology_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {39} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_topology_mgr_init: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {40} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_topology_mgr_destroy: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {41} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_aux_for_parent: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {42} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_mst_dsc_aux_for_port: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {43} ; .previous", sym drm_dp_mst_get_port_malloc, sym drm_dp_mst_put_port_malloc, sym drm_atomic_get_mst_payload_state, sym drm_dp_mst_connector_late_register, sym drm_dp_mst_connector_early_unregister, sym drm_dp_send_power_updown_phy, sym drm_dp_send_query_stream_enc_status, sym drm_dp_add_payload_part1, sym drm_dp_remove_payload_part1, sym drm_dp_remove_payload_part2, sym drm_dp_add_payload_part2, sym drm_dp_get_vc_payload_bw, sym drm_dp_read_mst_cap, sym drm_dp_mst_topology_mgr_set_mst, sym drm_dp_mst_topology_queue_probe, sym drm_dp_mst_topology_mgr_suspend, sym drm_dp_mst_topology_mgr_resume, sym drm_dp_mst_hpd_irq_handle_event, sym drm_dp_mst_hpd_irq_send_new_request, sym drm_dp_mst_detect_port, sym drm_dp_mst_edid_read, sym drm_dp_mst_get_edid, sym drm_dp_atomic_find_time_slots, sym drm_dp_atomic_release_time_slots, sym drm_dp_mst_atomic_setup_commit, sym drm_dp_mst_atomic_wait_for_dependencies, sym drm_dp_mst_root_conn_atomic_check, sym drm_dp_mst_update_slots, sym drm_dp_check_act_status, sym drm_dp_calc_pbn_mode, sym drm_dp_mst_dump_topology, sym drm_dp_mst_port_downstream_of_parent, sym drm_dp_mst_add_affected_dsc_crtcs, sym drm_dp_mst_atomic_enable_dsc, sym drm_dp_mst_atomic_check_mgr, sym drm_dp_mst_atomic_check, sym drm_dp_mst_topology_state_funcs, sym drm_atomic_get_mst_topology_state, sym drm_atomic_get_old_mst_topology_state, sym drm_atomic_get_new_mst_topology_state, sym drm_dp_mst_topology_mgr_init, sym drm_dp_mst_topology_mgr_destroy, sym drm_dp_mst_aux_for_parent, sym drm_dp_mst_dsc_aux_for_port, options(att_syntax));

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_remote_i2c_read_nak_reply {
    port_number: u8,
    nak_reason: u8,
    i2c_nak_transaction: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_remote_i2c_write_ack_reply {
    port_number: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_query_stream_enc_status_ack_reply {
    stream_id: u8,
    reply_signed: bool,
    unauthorizable_device_present: bool,
    legacy_device_present: bool,
    query_capable_device_present: bool,
    hdcp_1x_device_present: bool,
    hdcp_2x_device_present: bool,
    auth_completed: bool,
    encryption_enabled: bool,
    repeater_present: bool,
    state: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_allocate_payload {
    port_number: u8,
    number_sdp_streams: u8,
    vcpi: u8,
    pbn: u16,
    sdp_stream_sink: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_allocate_payload_ack_reply {
    port_number: u8,
    vcpi: u8,
    allocated_pbn: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_connection_status_notify {
    guid: __SlateRecord_uuid_h_17,
    port_number: u8,
    legacy_device_plug_status: bool,
    displayport_device_plug_status: bool,
    message_capability_status: bool,
    input_port: bool,
    peer_device_type: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_remote_dpcd_read {
    port_number: u8,
    dpcd_address: u32,
    num_bytes: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_remote_dpcd_write {
    port_number: u8,
    dpcd_address: u32,
    num_bytes: u8,
    bytes: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_remote_i2c_read {
    num_transactions: u8,
    port_number: u8,
    transactions: [drm_dp_remote_i2c_read_tx; 4],
    read_i2c_device_id: u8,
    num_bytes_read: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_remote_i2c_write {
    port_number: u8,
    write_i2c_device_id: u8,
    num_bytes: u8,
    bytes: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_query_stream_enc_status {
    stream_id: u8,
    client_id: [u8; 7],
    stream_event: u8,
    valid_stream_event: bool,
    stream_behavior: u8,
    valid_stream_behavior: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_port_number_req {
    port_number: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_enum_path_resources_ack_reply {
    port_number: u8,
    fec_capable: bool,
    full_payload_bw_number: u16,
    avail_payload_bw_number: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_port_number_rep {
    port_number: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_query_payload {
    port_number: u8,
    vcpi: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_resource_status_notify {
    port_number: u8,
    guid: __SlateRecord_uuid_h_17,
    available_pbn: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_query_payload_ack_reply {
    port_number: u8,
    allocated_pbn: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_sideband_msg_req_body {
    req_type: u8,
    u: ack_req,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_sideband_msg_reply_body {
    reply_type: u8,
    req_type: u8,
    u: ack_replies,
}


// msg is queued to be put into a slot
// msg has started transmitting on a slot - still on msgq
// msg has finished transmitting on a slot - removed from msgq only in slot
// msg has received a response - removed from slot
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_sideband_msg_tx {
    msg: [u8; 256],
    chunk: [u8; 48],
    cur_offset: u8,
    cur_len: u8,
    dst: *mut drm_dp_mst_branch,
    next: list_head,
    seqno: i32,
    state: i32,
    path_msg: bool,
    reply: drm_dp_sideband_msg_reply_body,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_mst_topology_cbs {
    add_connector: Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_port, *const u8) -> *mut drm_connector>,
    poll_hpd_irq: Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_mst_atomic_payload {
    port: *mut drm_dp_mst_port,
    vc_start_slot: i8,
    vcpi: u8,
    time_slots: i32,
    pbn: i32,
    __slate_bits_0: __slate_bits::__SlateBits1080U0,
    payload_allocation_status: u32,
    next: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_mst_topology_state {
    base: drm_private_state,
    mgr: *mut drm_dp_mst_topology_mgr,
    pending_crtc_mask: u32,
    commit_deps: *mut *mut drm_crtc_commit,
    num_commit_deps: u64,
    payload_mask: u32,
    payloads: list_head,
    total_avail_slots: u8,
    start_slot: u8,
    pbn_div: dfixed,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_mst_topology_mgr {
    base: drm_private_obj,
    dev: *mut drm_device,
    cbs: *const drm_dp_mst_topology_cbs,
    max_dpcd_transaction_bytes: i32,
    aux: *mut drm_dp_aux,
    max_payloads: i32,
    conn_base_id: i32,
    up_req_recv: drm_dp_sideband_msg_rx,
    down_rep_recv: drm_dp_sideband_msg_rx,
    lock: mutex,
    probe_lock: mutex,
    __slate_bits_0: __slate_bits::__SlateBits1046U0,
    payload_count: u8,
    next_start_slot: u8,
    mst_primary: *mut drm_dp_mst_branch,
    dpcd: [u8; 15],
    sink_count: u8,
    funcs: *const drm_private_state_funcs,
    qlock: mutex,
    tx_msg_downq: list_head,
    tx_waitq: wait_queue_head,
    work: work_struct,
    tx_work: work_struct,
    destroy_port_list: list_head,
    destroy_branch_device_list: list_head,
    delayed_destroy_lock: mutex,
    delayed_destroy_wq: *mut workqueue_struct,
    delayed_destroy_work: work_struct,
    up_req_list: list_head,
    up_req_lock: mutex,
    up_req_work: work_struct,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_hpd_irq_send_new_request")]
extern "C" fn drm_dp_mst_hpd_irq_send_new_request(mgr: *mut drm_dp_mst_topology_mgr) {
let mut kick: bool = true;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).qlock)) };
let mut __v0: *mut drm_dp_sideband_msg_tx;
let head__: *mut list_head = unsafe { core::ptr::addr_of_mut!((*mgr).tx_msg_downq) };
let mut __v1: *mut list_head;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_742() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*head__).next) as *const *mut list_head) };
let pos__: *mut list_head = __v1;
let __v3: *mut drm_dp_sideband_msg_tx = if pos__ != head__ {
list_entry__2!(pos__, drm_dp_sideband_msg_tx)
} else {
core::ptr::null_mut::<drm_dp_sideband_msg_tx>()
};
__v0 = __v3;
let txmsg: *mut drm_dp_sideband_msg_tx = __v0;
// If last transaction is not completed yet
if txmsg.is_null() || (unsafe { (*txmsg).state }) == 1 || (unsafe { (*txmsg).state }) == 2 {
kick = false;
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).qlock)) };
if kick {
drm_dp_mst_kick_tx(mgr);
}
}

fn drm_dp_mst_port_is_logical(port: *mut drm_dp_mst_port) -> bool {
((unsafe { (*port).port_num }) as i32) >= 8
}

fn to_drm_dp_mst_topology_state(state: *mut drm_private_state) -> *mut drm_dp_mst_topology_state {
container_of__7!(state, drm_dp_mst_topology_state)
}

fn __drm_dp_mst_state_iter_get(state: *mut drm_atomic_commit, mgr: &mut *mut drm_dp_mst_topology_mgr, old_state: *mut *mut drm_dp_mst_topology_state, new_state: *mut *mut drm_dp_mst_topology_state, i: i32) -> bool {
let objs_state: *mut __drm_private_objs_state = unsafe { (*state).private_objs }.wrapping_offset(i as isize);
if (unsafe { (*(*objs_state).ptr).funcs }) != unsafe { core::ptr::addr_of!(drm_dp_mst_topology_state_funcs.0) } {
return false;
}
unsafe {
*(mgr as *mut *mut drm_dp_mst_topology_mgr) = to_dp_mst_topology_mgr!(unsafe { (*objs_state).ptr });
}
if !old_state.is_null() {
unsafe {
*old_state = to_dp_mst_topology_state!(unsafe { (*objs_state).old_state });
}
}
if !new_state.is_null() {
unsafe {
*new_state = to_dp_mst_topology_state!(unsafe { (*objs_state).new_state });
}
}
true
}

// for_each_oldnew_mst_mgr_in_state - iterate over all DP MST topology
// managers in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @mgr: &struct drm_dp_mst_topology_mgr iteration cursor
// @old_state: &struct drm_dp_mst_topology_state iteration cursor for the old
// state
// @new_state: &struct drm_dp_mst_topology_state iteration cursor for the new
// state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all DRM DP MST topology managers in an atomic update,
// tracking both old and new state. This is useful in places where the state
// delta needs to be considered, for example in atomic check functions.
// for_each_new_mst_mgr_in_state - iterate over all DP MST topology managers
// in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @mgr: &struct drm_dp_mst_topology_mgr iteration cursor
// @new_state: &struct drm_dp_mst_topology_state iteration cursor for the new
// state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all DRM DP MST topology managers in an atomic update,
// tracking only the new state. This is useful in enable functions, where we
// need the new state the hardware should be in when the atomic commit
// operation has completed.
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_encoder_funcs {
    reset: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    destroy: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    late_register: Option<unsafe extern "C" fn(*mut drm_encoder) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    debugfs_init: Option<unsafe extern "C" fn(*mut drm_encoder, *mut dentry)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_encoder {
    dev: *mut drm_device,
    head: list_head,
    base: drm_mode_object,
    name: *mut u8,
    encoder_type: i32,
    index: u32,
    possible_crtcs: u32,
    possible_clones: u32,
    crtc: *mut drm_crtc,
    bridge_chain: list_head,
    bridge_chain_mutex: mutex,
    funcs: *const drm_encoder_funcs,
    helper_private: *const drm_encoder_helper_funcs,
    debugfs_entry: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_helper_funcs {
    dpms: Option<unsafe extern "C" fn(*mut drm_crtc, i32)>,
    prepare: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    commit: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    mode_valid: Option<unsafe extern "C" fn(*mut drm_crtc, *const drm_display_mode) -> i32>,
    mode_fixup: Option<unsafe extern "C" fn(*mut drm_crtc, *const drm_display_mode, *mut drm_display_mode) -> bool>,
    mode_set: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_display_mode, *mut drm_display_mode, i32, i32, *mut drm_framebuffer) -> i32>,
    mode_set_nofb: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    mode_set_base: Option<unsafe extern "C" fn(*mut drm_crtc, i32, i32, *mut drm_framebuffer) -> i32>,
    disable: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    atomic_check: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_atomic_commit) -> i32>,
    atomic_begin: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_atomic_commit)>,
    atomic_flush: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_atomic_commit)>,
    atomic_enable: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_atomic_commit)>,
    atomic_disable: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_atomic_commit)>,
    get_scanout_position: Option<unsafe extern "C" fn(*mut drm_crtc, bool, *mut i32, *mut i32, *mut i64, *mut i64, *const drm_display_mode) -> bool>,
    handle_vblank_timeout: Option<unsafe extern "C" fn(*mut drm_crtc) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_encoder_helper_funcs {
    dpms: Option<unsafe extern "C" fn(*mut drm_encoder, i32)>,
    mode_valid: Option<unsafe extern "C" fn(*mut drm_encoder, *const drm_display_mode) -> i32>,
    mode_fixup: Option<unsafe extern "C" fn(*mut drm_encoder, *const drm_display_mode, *mut drm_display_mode) -> bool>,
    prepare: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    commit: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    mode_set: Option<unsafe extern "C" fn(*mut drm_encoder, *mut drm_display_mode, *mut drm_display_mode)>,
    atomic_mode_set: Option<unsafe extern "C" fn(*mut drm_encoder, *mut drm_crtc_state, *mut drm_connector_state)>,
    detect: Option<unsafe extern "C" fn(*mut drm_encoder, *mut drm_connector) -> u32>,
    atomic_disable: Option<unsafe extern "C" fn(*mut drm_encoder, *mut drm_atomic_commit)>,
    atomic_enable: Option<unsafe extern "C" fn(*mut drm_encoder, *mut drm_atomic_commit)>,
    disable: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    enable: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    atomic_check: Option<unsafe extern "C" fn(*mut drm_encoder, *mut drm_crtc_state, *mut drm_connector_state) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_helper_funcs {
    get_modes: Option<unsafe extern "C" fn(*mut drm_connector) -> i32>,
    detect_ctx: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_modeset_acquire_ctx, bool) -> i32>,
    mode_valid: Option<unsafe extern "C" fn(*mut drm_connector, *const drm_display_mode) -> i32>,
    mode_valid_ctx: Option<unsafe extern "C" fn(*mut drm_connector, *const drm_display_mode, *mut drm_modeset_acquire_ctx, *mut i32) -> i32>,
    best_encoder: Option<unsafe extern "C" fn(*mut drm_connector) -> *mut drm_encoder>,
    atomic_best_encoder: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_atomic_commit) -> *mut drm_encoder>,
    atomic_check: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_atomic_commit) -> i32>,
    atomic_commit: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_atomic_commit)>,
    prepare_writeback_job: Option<unsafe extern "C" fn(*mut drm_writeback_connector, *mut drm_writeback_job) -> i32>,
    cleanup_writeback_job: Option<unsafe extern "C" fn(*mut drm_writeback_connector, *mut drm_writeback_job)>,
    enable_hpd: Option<unsafe extern "C" fn(*mut drm_connector)>,
    disable_hpd: Option<unsafe extern "C" fn(*mut drm_connector)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane_helper_funcs {
    prepare_fb: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state) -> i32>,
    cleanup_fb: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state)>,
    begin_fb_access: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state) -> i32>,
    end_fb_access: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state)>,
    atomic_check: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_atomic_commit) -> i32>,
    atomic_update: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_atomic_commit)>,
    atomic_enable: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_atomic_commit)>,
    atomic_disable: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_atomic_commit)>,
    atomic_async_check: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_atomic_commit, bool) -> i32>,
    atomic_async_update: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_atomic_commit)>,
    get_scanout_buffer: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_scanout_buffer) -> i32>,
    panic_flush: Option<unsafe extern "C" fn(*mut drm_plane)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_config_helper_funcs {
    atomic_commit_tail: Option<unsafe extern "C" fn(*mut drm_atomic_commit)>,
    atomic_commit_setup: Option<unsafe extern "C" fn(*mut drm_atomic_commit) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_driver {
    load: Option<unsafe extern "C" fn(*mut drm_device, u64) -> i32>,
    open: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file) -> i32>,
    postclose: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file)>,
    unload: Option<unsafe extern "C" fn(*mut drm_device)>,
    release: Option<unsafe extern "C" fn(*mut drm_device)>,
    master_set: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file, bool)>,
    master_drop: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file)>,
    debugfs_init: Option<unsafe extern "C" fn(*mut drm_minor)>,
    gem_create_object: Option<unsafe extern "C" fn(*mut drm_device, u64) -> *mut drm_gem_object>,
    prime_handle_to_fd: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file, u32, u32, *mut i32) -> i32>,
    prime_fd_to_handle: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file, i32, *mut u32) -> i32>,
    gem_prime_import: Option<unsafe extern "C" fn(*mut drm_device, *mut dma_buf) -> *mut drm_gem_object>,
    gem_prime_import_sg_table: Option<unsafe extern "C" fn(*mut drm_device, *mut dma_buf_attachment, *mut sg_table) -> *mut drm_gem_object>,
    dumb_create: Option<unsafe extern "C" fn(*mut drm_file, *mut drm_device, *mut drm_mode_create_dumb) -> i32>,
    dumb_map_offset: Option<unsafe extern "C" fn(*mut drm_file, *mut drm_device, u32, *mut u64) -> i32>,
    fbdev_probe: Option<unsafe extern "C" fn(*mut drm_fb_helper, *mut drm_fb_helper_surface_size) -> i32>,
    show_fdinfo: Option<unsafe extern "C" fn(*mut drm_printer, *mut drm_file)>,
    major: i32,
    minor: i32,
    patchlevel: i32,
    name: *mut u8,
    desc: *mut u8,
    driver_features: u32,
    ioctls: *const drm_ioctl_desc,
    num_ioctls: i32,
    fops: *const file_operations,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct est_timings {
    t1: u8,
    t2: u8,
    mfg_rsvd: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct std_timing {
    hsize: u8,
    vfreq_aspect: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct detailed_pixel_timing {
    hactive_lo: u8,
    hblank_lo: u8,
    hactive_hblank_hi: u8,
    vactive_lo: u8,
    vblank_lo: u8,
    vactive_vblank_hi: u8,
    hsync_offset_lo: u8,
    hsync_pulse_width_lo: u8,
    vsync_offset_pulse_width_lo: u8,
    hsync_vsync_offset_pulse_width_hi: u8,
    width_mm_lo: u8,
    height_mm_lo: u8,
    width_height_mm_hi: u8,
    hborder: u8,
    vborder: u8,
    misc: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct detailed_data_string {
    str: [u8; 13],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct detailed_data_monitor_range {
    min_vfreq: u8,
    max_vfreq: u8,
    min_hfreq_khz: u8,
    max_hfreq_khz: u8,
    pixel_clock_mhz: u8,
    flags: u8,
    formula: __SlateRecord_drm_edid_h_126,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct detailed_data_wpindex {
    white_yx_lo: u8,
    white_x_hi: u8,
    white_y_hi: u8,
    gamma: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cvt_timing {
    code: [u8; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct detailed_non_pixel {
    pad1: u8,
    r#type: u8,
    pad2: u8,
    data: __SlateRecord_drm_edid_h_171,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct detailed_timing {
    pixel_clock: u16,
    data: __SlateRecord_drm_edid_h_192,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct drm_edid_product_id {
    manufacturer_name: u16,
    product_code: u16,
    serial_number: u32,
    week_of_manufacture: u8,
    year_of_manufacture: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct edid {
    header: [u8; 8],
    __slate_anon_1: __SlateRecord_drm_edid_h_294,
    version: u8,
    revision: u8,
    input: u8,
    width_cm: u8,
    height_cm: u8,
    gamma: u8,
    features: u8,
    red_green_lo: u8,
    blue_white_lo: u8,
    red_x: u8,
    red_y: u8,
    green_x: u8,
    green_y: u8,
    blue_x: u8,
    blue_y: u8,
    white_x: u8,
    white_y: u8,
    established_timings: est_timings,
    standard_timings: [std_timing; 8],
    detailed_timings: [detailed_timing; 4],
    extensions: u8,
    checksum: u8,
}


fn drm_debug_enabled_raw(category: u32) -> bool {
unlikely__3!((unsafe { __drm_debug }) & BIT__2!(category)) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_printer {
    printfn: Option<unsafe extern "C" fn(*mut drm_printer, *mut va_format)>,
    puts: Option<unsafe extern "C" fn(*mut drm_printer, *const u8)>,
    arg: *mut (),
    origin: *const (),
    prefix: *const u8,
    line: __SlateRecord_drm_print_h_181,
    category: u32,
}


// drm_printf_indent - Print to a &drm_printer stream with indentation
// @printer: DRM printer
// @indent: Tab indentation level (max 5)
// @fmt: Format string
fn drm_dbg_printer(drm: *mut drm_device, category: u32, prefix: *const u8) -> drm_printer {
let mut __v0: u64;
let mut __here: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("lea 0(%rip), {0:r}", lateout(reg) __t0, options(pure, nomem, nostack, att_syntax));
__here = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v0 = __here;
let p: drm_printer = {
let mut __t1: drm_printer = unsafe { core::mem::zeroed() };
__t1.printfn = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_printer, *mut va_format)>>(__drm_printfn_dbg as *const ()) };
__t1.arg = drm as *mut ();
__t1.origin = __v0 as *const ();
__t1.prefix = prefix;
__t1.category = category;
__t1
};
// it's fine as we will be inlined
return p;
}

fn __drm_to_dev(drm: *const drm_device) -> *mut device {
if !drm.is_null() { unsafe { (*drm).dev } } else { core::ptr::null_mut::<device>() }
}

// Helper for struct drm_device based logging.
// NOTE: this is deprecated in favor of pr_err().
// NOTE: this is deprecated in favor of drm_dbg_kms(NULL, ...).
// Helper for struct drm_device based WARNs
fn drm_dp_aux_register_devnode(aux: *mut drm_dp_aux) -> i32 {
0
}

fn drm_dp_aux_unregister_devnode(aux: *mut drm_dp_aux) {
}

#[unsafe(no_mangle)]
extern "C" fn drm_dp_encode_sideband_req(req: *const drm_dp_sideband_msg_req_body, raw: *mut drm_dp_sideband_msg_tx) {
let mut idx: i32 = 0;
let mut i: i32 = 0;
let buf: *mut u8 = unsafe { (*raw).msg.as_mut_ptr() as *mut u8 };
let __v0: i32 = idx;
idx = __v0.wrapping_add(1);
unsafe {
*buf.wrapping_offset(__v0 as isize) = (((*req).req_type as i32) & 127) as u8;
}
match (unsafe { (*req).req_type }) as i32 {
16 | 36 | 37 => {
unsafe {
*buf.wrapping_offset(idx as isize) = ((((*req).u.port_num.port_number as i32) & 15) << 4i32) as u8;
}
idx = idx.wrapping_add(1);
}
17 => {
unsafe {
*buf.wrapping_offset(idx as isize) = ((((*req).u.allocate_payload.port_number as i32) & 15) << 4i32 | ((*req).u.allocate_payload.number_sdp_streams as i32) & 15) as u8;
}
idx = idx.wrapping_add(1);
unsafe {
*buf.wrapping_offset(idx as isize) = (((*req).u.allocate_payload.vcpi as i32) & 127) as u8;
}
idx = idx.wrapping_add(1);
unsafe {
*buf.wrapping_offset(idx as isize) = (((*req).u.allocate_payload.pbn as i32) >> 8i32) as u8;
}
idx = idx.wrapping_add(1);
unsafe {
*buf.wrapping_offset(idx as isize) = (((*req).u.allocate_payload.pbn as i32) & 255) as u8;
}
idx = idx.wrapping_add(1);
i = 0;
while i < ((unsafe { (*req).u.allocate_payload.number_sdp_streams }) as i32) / 2 {
unsafe {
*buf.wrapping_offset(idx as isize) = (((*((*req).u.allocate_payload.sdp_stream_sink.as_ptr() as *const u8).wrapping_offset(i.wrapping_mul(2) as isize) as i32) & 15) << 4i32 | (*((*req).u.allocate_payload.sdp_stream_sink.as_ptr() as *const u8).wrapping_offset(i.wrapping_mul(2).wrapping_add(1) as isize) as i32) & 15) as u8;
}
idx = idx.wrapping_add(1);
i = i.wrapping_add(1);
}
if ((unsafe { (*req).u.allocate_payload.number_sdp_streams }) as i32) & 1 != 0 {
i = ((unsafe { (*req).u.allocate_payload.number_sdp_streams }) as i32).wrapping_sub(1);
unsafe {
*buf.wrapping_offset(idx as isize) = (((*((*req).u.allocate_payload.sdp_stream_sink.as_ptr() as *const u8).wrapping_offset(i as isize) as i32) & 15) << 4i32) as u8;
}
idx = idx.wrapping_add(1);
}
}
18 => {
unsafe {
*buf.wrapping_offset(idx as isize) = ((((*req).u.query_payload.port_number as i32) & 15) << 4i32) as u8;
}
idx = idx.wrapping_add(1);
unsafe {
*buf.wrapping_offset(idx as isize) = (((*req).u.query_payload.vcpi as i32) & 127) as u8;
}
idx = idx.wrapping_add(1);
}
32 => {
unsafe {
*buf.wrapping_offset(idx as isize) = ((((*req).u.dpcd_read.port_number as i32) & 15) << 4i32) as u8;
}
let __v24: *mut u8 = buf.wrapping_offset(idx as isize);
unsafe {
*__v24 = (((*__v24 as i32) as u32) | ((*req).u.dpcd_read.dpcd_address & 983040) >> 16i32 & 15) as u8;
}
idx = idx.wrapping_add(1);
unsafe {
*buf.wrapping_offset(idx as isize) = (((*req).u.dpcd_read.dpcd_address & 65280) >> 8i32) as u8;
}
idx = idx.wrapping_add(1);
unsafe {
*buf.wrapping_offset(idx as isize) = ((*req).u.dpcd_read.dpcd_address & 255) as u8;
}
idx = idx.wrapping_add(1);
unsafe {
*buf.wrapping_offset(idx as isize) = (*req).u.dpcd_read.num_bytes;
}
idx = idx.wrapping_add(1);
}
33 => {
unsafe {
*buf.wrapping_offset(idx as isize) = ((((*req).u.dpcd_write.port_number as i32) & 15) << 4i32) as u8;
}
let __v35: *mut u8 = buf.wrapping_offset(idx as isize);
unsafe {
*__v35 = (((*__v35 as i32) as u32) | ((*req).u.dpcd_write.dpcd_address & 983040) >> 16i32 & 15) as u8;
}
idx = idx.wrapping_add(1);
unsafe {
*buf.wrapping_offset(idx as isize) = (((*req).u.dpcd_write.dpcd_address & 65280) >> 8i32) as u8;
}
idx = idx.wrapping_add(1);
unsafe {
*buf.wrapping_offset(idx as isize) = ((*req).u.dpcd_write.dpcd_address & 255) as u8;
}
idx = idx.wrapping_add(1);
unsafe {
*buf.wrapping_offset(idx as isize) = (*req).u.dpcd_write.num_bytes;
}
idx = idx.wrapping_add(1);
unsafe { memcpy(buf.wrapping_offset(idx as isize) as *mut (), (*req).u.dpcd_write.bytes as *const (), (*req).u.dpcd_write.num_bytes as u64) };
idx = idx.wrapping_add((unsafe { (*req).u.dpcd_write.num_bytes }) as i32);
}
34 => {
unsafe {
*buf.wrapping_offset(idx as isize) = ((((*req).u.i2c_read.port_number as i32) & 15) << 4i32) as u8;
}
let __v48: *mut u8 = buf.wrapping_offset(idx as isize);
unsafe {
*__v48 = ((*__v48 as i32) | ((*req).u.i2c_read.num_transactions as i32) & 3) as u8;
}
idx = idx.wrapping_add(1);
i = 0;
while i < ((unsafe { (*req).u.i2c_read.num_transactions }) as i32) & 3 {
unsafe {
*buf.wrapping_offset(idx as isize) = (((*((*req).u.i2c_read.transactions.as_ptr() as *const drm_dp_remote_i2c_read_tx).wrapping_offset(i as isize)).i2c_dev_id as i32) & 127) as u8;
}
idx = idx.wrapping_add(1);
unsafe {
*buf.wrapping_offset(idx as isize) = (*((*req).u.i2c_read.transactions.as_ptr() as *const drm_dp_remote_i2c_read_tx).wrapping_offset(i as isize)).num_bytes;
}
idx = idx.wrapping_add(1);
unsafe { memcpy(buf.wrapping_offset(idx as isize) as *mut (), (*((*req).u.i2c_read.transactions.as_ptr() as *const drm_dp_remote_i2c_read_tx).wrapping_offset(i as isize)).bytes as *const (), (*((*req).u.i2c_read.transactions.as_ptr() as *const drm_dp_remote_i2c_read_tx).wrapping_offset(i as isize)).num_bytes as u64) };
let __v57: i32 = idx;
idx = __v57.wrapping_add((unsafe { (*((*req).u.i2c_read.transactions.as_ptr() as *const drm_dp_remote_i2c_read_tx).wrapping_offset(i as isize)).num_bytes }) as i32);
unsafe {
*buf.wrapping_offset(idx as isize) = ((((*((*req).u.i2c_read.transactions.as_ptr() as *const drm_dp_remote_i2c_read_tx).wrapping_offset(i as isize)).no_stop_bit as i32) & 1) << 4i32) as u8;
}
let __v59: *mut u8 = buf.wrapping_offset(idx as isize);
let __v60: u8 = unsafe { *__v59 };
let __v61: u8 = ((__v60 as i32) | ((unsafe { (*((*req).u.i2c_read.transactions.as_ptr() as *const drm_dp_remote_i2c_read_tx).wrapping_offset(i as isize)).i2c_transaction_delay }) as i32) & 15) as u8;
unsafe {
*__v59 = __v61;
}
idx = idx.wrapping_add(1);
i = i.wrapping_add(1);
}
unsafe {
*buf.wrapping_offset(idx as isize) = (((*req).u.i2c_read.read_i2c_device_id as i32) & 127) as u8;
}
idx = idx.wrapping_add(1);
unsafe {
*buf.wrapping_offset(idx as isize) = (*req).u.i2c_read.num_bytes_read;
}
idx = idx.wrapping_add(1);
}
35 => {
unsafe {
*buf.wrapping_offset(idx as isize) = ((((*req).u.i2c_write.port_number as i32) & 15) << 4i32) as u8;
}
idx = idx.wrapping_add(1);
unsafe {
*buf.wrapping_offset(idx as isize) = (((*req).u.i2c_write.write_i2c_device_id as i32) & 127) as u8;
}
idx = idx.wrapping_add(1);
unsafe {
*buf.wrapping_offset(idx as isize) = (*req).u.i2c_write.num_bytes;
}
idx = idx.wrapping_add(1);
unsafe { memcpy(buf.wrapping_offset(idx as isize) as *mut (), (*req).u.i2c_write.bytes as *const (), (*req).u.i2c_write.num_bytes as u64) };
idx = idx.wrapping_add((unsafe { (*req).u.i2c_write.num_bytes }) as i32);
}
56 => {
let msg: *const drm_dp_query_stream_enc_status = unsafe { core::ptr::addr_of!((*req).u.enc_status) };
unsafe {
*buf.wrapping_offset(idx as isize) = (*msg).stream_id;
}
idx = idx.wrapping_add(1);
unsafe { memcpy(buf.wrapping_offset(idx as isize) as *mut (), ((*msg).client_id.as_ptr() as *const u8) as *const (), core::mem::size_of::<[u8; 7]>() as u64) };
idx = (idx as u64).wrapping_add(core::mem::size_of::<[u8; 7]>() as u64) as i32;
unsafe {
*buf.wrapping_offset(idx as isize) = 0;
}
let mut __v83: u64;
loop {
if __bf_cast_unsigned!(((((const { core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_454_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1))) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_660() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v85: u64;
loop {
if false {
unsafe { __compiletime_assert_661() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_454_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)) == 0 {
unsafe { __compiletime_assert_662() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if unsafe { __slate_intrinsic_70b71f185ebb8c4a((*msg).stream_event) } { !((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_454_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)) >> ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_454_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)).trailing_zeros() as i32)) & (0i32.wrapping_add((unsafe { (*msg).stream_event }) as i32) as u64) } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_663() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_454_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_454_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_454_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_454_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_664() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v85 = ((unsafe { (*msg).stream_event }) as u64) << ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_454_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_454_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1));
__v83 = __v85;
let __v91: *mut u8 = buf.wrapping_offset(idx as isize);
unsafe {
*__v91 = (((*__v91 as i32) as u64) | __v83) as u8;
}
let __v94: *mut u8 = buf.wrapping_offset(idx as isize);
let __v95: u8 = unsafe { *__v94 };
let __v96: u8 = (((__v95 as i32) as u64) | if unsafe { (*msg).valid_stream_event } { BIT!(2i32) } else { 0 }) as u8;
unsafe {
*__v94 = __v96;
}
let mut __v97: u64;
loop {
if __bf_cast_unsigned!(((((const { core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_456_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4))) > __bf_cast_unsigned!(18446744073709551615u64) {
unsafe { __compiletime_assert_665() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v99: u64;
loop {
if false {
unsafe { __compiletime_assert_666() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_456_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)) == 0 {
unsafe { __compiletime_assert_667() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if unsafe { __slate_intrinsic_70b71f185ebb8c4a((*msg).stream_behavior) } { !((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_456_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)) >> ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_456_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)).trailing_zeros() as i32)) & (0i32.wrapping_add((unsafe { (*msg).stream_behavior }) as i32) as u64) } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_668() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_456_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_456_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_456_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_456_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_669() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v99 = ((unsafe { (*msg).stream_behavior }) as u64) << ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_456_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_456_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4));
__v97 = __v99;
let __v105: *mut u8 = buf.wrapping_offset(idx as isize);
unsafe {
*__v105 = (((*__v105 as i32) as u64) | __v97) as u8;
}
let __v108: *mut u8 = buf.wrapping_offset(idx as isize);
let __v109: u8 = unsafe { *__v108 };
let __v110: u8 = (((__v109 as i32) as u64) | if (unsafe { (*msg).valid_stream_behavior }) != 0 { BIT!(5i32) } else { 0 }) as u8;
unsafe {
*__v108 = __v110;
}
idx = idx.wrapping_add(1);
}
_ => {
}
}
unsafe {
(*raw).cur_len = idx as u8;
}
}

#[unsafe(no_mangle)]
extern "C" fn drm_dp_decode_sideband_req(raw: *const drm_dp_sideband_msg_tx, req: *mut drm_dp_sideband_msg_req_body) -> i32 {
let buf: *const u8 = unsafe { (*raw).msg.as_ptr() as *const u8 };
let mut i: i32 = 0;
let mut idx: i32 = 0;
let __v0: i32 = idx;
idx = __v0.wrapping_add(1);
unsafe {
(*req).req_type = ((*buf.wrapping_offset(__v0 as isize) as i32) & 127) as u8;
}
match (unsafe { (*req).req_type }) as i32 {
16 | 36 | 37 => {
unsafe {
(*req).u.port_num.port_number = ((*buf.wrapping_offset(idx as isize) as i32) >> 4i32 & 15) as u8;
}
}
17 => {
let a: *mut drm_dp_allocate_payload = unsafe { core::ptr::addr_of_mut!((*req).u.allocate_payload) };
unsafe {
(*a).number_sdp_streams = ((*buf.wrapping_offset(idx as isize) as i32) & 15) as u8;
}
unsafe {
(*a).port_number = ((*buf.wrapping_offset(idx as isize) as i32) >> 4i32 & 15) as u8;
}
let mut __v3: i64;
let mut __ret_warn_on: i32 = 0;
let __v5: i32 = idx.wrapping_add(1);
idx = __v5;
__ret_warn_on = (((unsafe { *buf.wrapping_offset(__v5 as isize) }) as i32) & 128 != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("670: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 670b - ., 3; .popsection\n# {0}", const 670 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_dp_decode_sideband_req_6, sym __slate_asm_string_drm_dp_decode_sideband_req_7, const 489 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("671: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 671b - ., 4; .popsection\n# {0}", const 671 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
unsafe {
(*a).vcpi = ((*buf.wrapping_offset(idx as isize) as i32) & 127) as u8;
}
let __v11: i32 = idx.wrapping_add(1);
idx = __v11;
unsafe {
(*a).pbn = ((*buf.wrapping_offset(__v11 as isize) as i32) << 8i32) as u16;
}
let __v13: i32 = idx.wrapping_add(1);
idx = __v13;
unsafe {
(*a).pbn = (((*a).pbn as i32) | (*buf.wrapping_offset(__v13 as isize) as i32)) as u16;
}
idx = idx.wrapping_add(1);
i = 0;
while i < ((unsafe { (*a).number_sdp_streams }) as i32) {
unsafe {
*((*a).sdp_stream_sink.as_mut_ptr() as *mut u8).wrapping_offset(i as isize) = ((*buf.wrapping_offset(idx.wrapping_add(i / 2) as isize) as i32) >> if i % 2 != 0 { 0i32 } else { 4i32 } & 15) as u8;
}
i = i.wrapping_add(1);
}
}
18 => {
unsafe {
(*req).u.query_payload.port_number = ((*buf.wrapping_offset(idx as isize) as i32) >> 4i32 & 15) as u8;
}
let mut __v23: i64;
let mut __ret_warn_on: i32 = 0;
let __v25: i32 = idx.wrapping_add(1);
idx = __v25;
__ret_warn_on = (((unsafe { *buf.wrapping_offset(__v25 as isize) }) as i32) & 128 != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("672: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 672b - ., 3; .popsection\n# {0}", const 672 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_dp_decode_sideband_req_26, sym __slate_asm_string_drm_dp_decode_sideband_req_27, const 504 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("673: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 673b - ., 4; .popsection\n# {0}", const 673 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v23 = (__ret_warn_on != 0) as i64;
unsafe {
(*req).u.query_payload.vcpi = ((*buf.wrapping_offset(idx as isize) as i32) & 127) as u8;
}
}
32 => {
let r: *mut drm_dp_remote_dpcd_read = unsafe { core::ptr::addr_of_mut!((*req).u.dpcd_read) };
unsafe {
(*r).port_number = ((*buf.wrapping_offset(idx as isize) as i32) >> 4i32 & 15) as u8;
}
unsafe {
(*r).dpcd_address = ((*buf.wrapping_offset(idx as isize) as i32) << 16i32 & 983040) as u32;
}
let __v31: i32 = idx.wrapping_add(1);
idx = __v31;
unsafe {
(*r).dpcd_address = (*r).dpcd_address | (((*buf.wrapping_offset(__v31 as isize) as i32) << 8i32 & 65280) as u32);
}
let __v37: i32 = idx.wrapping_add(1);
idx = __v37;
unsafe {
(*r).dpcd_address = (*r).dpcd_address | (((*buf.wrapping_offset(__v37 as isize) as i32) & 255) as u32);
}
let __v43: i32 = idx.wrapping_add(1);
idx = __v43;
unsafe {
(*r).num_bytes = *buf.wrapping_offset(__v43 as isize);
}
}
33 => {
let w: *mut drm_dp_remote_dpcd_write = unsafe { core::ptr::addr_of_mut!((*req).u.dpcd_write) };
unsafe {
(*w).port_number = ((*buf.wrapping_offset(idx as isize) as i32) >> 4i32 & 15) as u8;
}
unsafe {
(*w).dpcd_address = ((*buf.wrapping_offset(idx as isize) as i32) << 16i32 & 983040) as u32;
}
let __v45: i32 = idx.wrapping_add(1);
idx = __v45;
unsafe {
(*w).dpcd_address = (*w).dpcd_address | (((*buf.wrapping_offset(__v45 as isize) as i32) << 8i32 & 65280) as u32);
}
let __v51: i32 = idx.wrapping_add(1);
idx = __v51;
unsafe {
(*w).dpcd_address = (*w).dpcd_address | (((*buf.wrapping_offset(__v51 as isize) as i32) & 255) as u32);
}
let __v57: i32 = idx.wrapping_add(1);
idx = __v57;
unsafe {
(*w).num_bytes = *buf.wrapping_offset(__v57 as isize);
}
let mut __v58: *mut ();
let mut __v59: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
let __v61: i32 = idx.wrapping_add(1);
idx = __v61;
_res = unsafe { kmemdup_noprof(buf.wrapping_offset(__v61 as isize) as *const (), (*w).num_bytes as u64, 3264u32) };
alloc_tag_restore!();
} else {
let __v64: i32 = idx.wrapping_add(1);
idx = __v64;
_res = unsafe { kmemdup_noprof(buf.wrapping_offset(__v64 as isize) as *const (), (*w).num_bytes as u64, 3264u32) };
}
__v59 = _res;
__v58 = __v59;
unsafe {
(*w).bytes = __v58 as *mut u8;
}
if unsafe { (*w).bytes }.is_null() {
return 12i32.wrapping_neg();
}
}
34 => {
let r: *mut drm_dp_remote_i2c_read = unsafe { core::ptr::addr_of_mut!((*req).u.i2c_read) };
let mut tx: *mut drm_dp_remote_i2c_read_tx = unsafe { core::mem::zeroed() };
let mut failed: bool = false;
unsafe {
(*r).num_transactions = ((*buf.wrapping_offset(idx as isize) as i32) & 3) as u8;
}
unsafe {
(*r).port_number = ((*buf.wrapping_offset(idx as isize) as i32) >> 4i32 & 15) as u8;
}
i = 0;
while i < ((unsafe { (*r).num_transactions }) as i32) {
tx = unsafe { (*r).transactions.as_mut_ptr() as *mut drm_dp_remote_i2c_read_tx }.wrapping_offset(i as isize);
let __v66: i32 = idx.wrapping_add(1);
idx = __v66;
unsafe {
(*tx).i2c_dev_id = ((*buf.wrapping_offset(__v66 as isize) as i32) & 127) as u8;
}
let __v68: i32 = idx.wrapping_add(1);
idx = __v68;
unsafe {
(*tx).num_bytes = *buf.wrapping_offset(__v68 as isize);
}
let mut __v69: *mut ();
let mut __v70: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
let __v72: i32 = idx.wrapping_add(1);
idx = __v72;
_res = unsafe { kmemdup_noprof(buf.wrapping_offset(__v72 as isize) as *const (), (*tx).num_bytes as u64, 3264u32) };
alloc_tag_restore!();
} else {
let __v75: i32 = idx.wrapping_add(1);
idx = __v75;
_res = unsafe { kmemdup_noprof(buf.wrapping_offset(__v75 as isize) as *const (), (*tx).num_bytes as u64, 3264u32) };
}
__v70 = _res;
__v69 = __v70;
unsafe {
(*tx).bytes = __v69 as *mut u8;
}
if unsafe { (*tx).bytes }.is_null() {
failed = true;
break;
}
idx = idx.wrapping_add((unsafe { (*tx).num_bytes }) as i32);
unsafe {
(*tx).no_stop_bit = ((*buf.wrapping_offset(idx as isize) as i32) >> 5i32 & 1) as u8;
}
unsafe {
(*tx).i2c_transaction_delay = ((*buf.wrapping_offset(idx as isize) as i32) & 15) as u8;
}
i = i.wrapping_add(1);
}
if failed {
i = 0;
while i < ((unsafe { (*r).num_transactions }) as i32) {
tx = unsafe { (*r).transactions.as_mut_ptr() as *mut drm_dp_remote_i2c_read_tx }.wrapping_offset(i as isize);
unsafe { kfree((*tx).bytes as *const ()) };
i = i.wrapping_add(1);
}
return 12i32.wrapping_neg();
}
let __v85: i32 = idx.wrapping_add(1);
idx = __v85;
unsafe {
(*r).read_i2c_device_id = ((*buf.wrapping_offset(__v85 as isize) as i32) & 127) as u8;
}
let __v87: i32 = idx.wrapping_add(1);
idx = __v87;
unsafe {
(*r).num_bytes_read = *buf.wrapping_offset(__v87 as isize);
}
}
35 => {
let w: *mut drm_dp_remote_i2c_write = unsafe { core::ptr::addr_of_mut!((*req).u.i2c_write) };
unsafe {
(*w).port_number = ((*buf.wrapping_offset(idx as isize) as i32) >> 4i32 & 15) as u8;
}
let __v89: i32 = idx.wrapping_add(1);
idx = __v89;
unsafe {
(*w).write_i2c_device_id = ((*buf.wrapping_offset(__v89 as isize) as i32) & 127) as u8;
}
let __v91: i32 = idx.wrapping_add(1);
idx = __v91;
unsafe {
(*w).num_bytes = *buf.wrapping_offset(__v91 as isize);
}
let mut __v92: *mut ();
let mut __v93: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
let __v95: i32 = idx.wrapping_add(1);
idx = __v95;
_res = unsafe { kmemdup_noprof(buf.wrapping_offset(__v95 as isize) as *const (), (*w).num_bytes as u64, 3264u32) };
alloc_tag_restore!();
} else {
let __v98: i32 = idx.wrapping_add(1);
idx = __v98;
_res = unsafe { kmemdup_noprof(buf.wrapping_offset(__v98 as isize) as *const (), (*w).num_bytes as u64, 3264u32) };
}
__v93 = _res;
__v92 = __v93;
unsafe {
(*w).bytes = __v92 as *mut u8;
}
if unsafe { (*w).bytes }.is_null() {
return 12i32.wrapping_neg();
}
}
56 => {
let __v99: i32 = idx;
idx = __v99.wrapping_add(1);
unsafe {
(*req).u.enc_status.stream_id = *buf.wrapping_offset(__v99 as isize);
}
i = 0;
while (i as u64) < (core::mem::size_of::<[u8; 7]>() as u64) {
let __v101: i32 = idx;
idx = __v101.wrapping_add(1);
unsafe {
*((*req).u.enc_status.client_id.as_mut_ptr() as *mut u8).wrapping_offset(i as isize) = *buf.wrapping_offset(__v101 as isize);
}
i = i.wrapping_add(1);
}
let mut __v106: u64;
loop {
if __bf_cast_unsigned!(((((const { core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_594_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1))) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u64) {
unsafe { __compiletime_assert_674() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v108: u64;
loop {
if false {
unsafe { __compiletime_assert_675() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_594_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)) == 0 {
unsafe { __compiletime_assert_676() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_594_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)) >> ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_594_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_677() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_594_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_594_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_594_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_594_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_678() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v108 = ((((unsafe { *buf.wrapping_offset(idx as isize) }) as i32) as u64) & (core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_594_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1))) >> ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_594_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(1)).trailing_zeros() as i32);
__v106 = __v108;
unsafe {
(*req).u.enc_status.stream_event = __v106 as u8;
}
let mut __v113: u64;
loop {
if __bf_cast_unsigned!(4u64) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u64) {
unsafe { __compiletime_assert_679() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v115: u64;
loop {
if false {
unsafe { __compiletime_assert_680() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 4u64 == 0 {
unsafe { __compiletime_assert_681() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !(4u64 >> (4u64.trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_682() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 4u64.wrapping_add(1u64 << (4u64.trailing_zeros() as i32)) & 4u64.wrapping_add(1u64 << (4u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_683() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v115 = ((((unsafe { *buf.wrapping_offset(idx as isize) }) as i32) as u64) & 4u64) >> (4u64.trailing_zeros() as i32);
__v113 = __v115;
unsafe {
(*req).u.enc_status.valid_stream_event = __v113 != 0;
}
let mut __v120: u64;
loop {
if __bf_cast_unsigned!(((((const { core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_598_3>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4))) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u64) {
unsafe { __compiletime_assert_684() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v122: u64;
loop {
if false {
unsafe { __compiletime_assert_685() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((const { core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_598_5>() }) as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> ((const { core::mem::size_of::<u64>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)) == 0 {
unsafe { __compiletime_assert_686() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_598_6>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)) >> ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_598_7>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)).trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_687() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_598_8>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_598_9>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)).trailing_zeros() as i32)) & (core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_598_10>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)).wrapping_add(1u64 << ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_598_11>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_688() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v122 = ((((unsafe { *buf.wrapping_offset(idx as isize) }) as i32) as u64) & (core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_598_13>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4))) >> ((core::mem::size_of::<__SlateRecord_drm_dp_mst_topology_c_598_14>() as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) << 3i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u64) < 1) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)).trailing_zeros() as i32);
__v120 = __v122;
unsafe {
(*req).u.enc_status.stream_behavior = __v120 as u8;
}
let mut __v127: u64;
loop {
if __bf_cast_unsigned!(32u64) > ((__bf_cast_unsigned__2!(18446744073709551615u64) as i32) as u64) {
unsafe { __compiletime_assert_689() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v129: u64;
loop {
if false {
unsafe { __compiletime_assert_690() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 32u64 == 0 {
unsafe { __compiletime_assert_691() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { !(32u64 >> (32u64.trailing_zeros() as i32)) & 0u64 } else { 0u64 }) != 0 {
unsafe { __compiletime_assert_692() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if 32u64.wrapping_add(1u64 << (32u64.trailing_zeros() as i32)) & 32u64.wrapping_add(1u64 << (32u64.trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_693() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v129 = ((((unsafe { *buf.wrapping_offset(idx as isize) }) as i32) as u64) & 32u64) >> (32u64.trailing_zeros() as i32);
__v127 = __v129;
unsafe {
(*req).u.enc_status.valid_stream_behavior = __v127 as u8;
}
}
_ => {
}
}
0
}

/// DOC: dp mst helper
///
/// These functions contain parts of the DisplayPort 1.2a MultiStream Transport
/// protocol. The helpers contain a topology manager and bandwidth manager.
/// The helpers encapsulate the sending and received of sideband msgs.
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_pending_up_req {
    hdr: drm_dp_sideband_msg_hdr,
    msg: drm_dp_sideband_msg_req_body,
    next: list_head,
}


fn drm_dp_mst_req_type_str(req_type: u8) -> *const u8 {
if ((req_type as i32) as u64) >= ARRAY_SIZE__9!() || unsafe { req_type_str.0[(req_type as i32) as usize] }.is_null() {
return c"unknown".as_ptr() as *const u8;
}
unsafe { req_type_str.0[(req_type as i32) as usize] }
}

static req_type_str: __SlateAlign16<[*const u8; 57]> = __SlateAlign16({
let mut __t0: [*const u8; 57] = unsafe { core::mem::zeroed() };
__t0[0] = (b"GET_MSG_TRANSACTION_VERSION\0".as_ptr() as *mut u8) as *const u8;
__t0[1] = (b"LINK_ADDRESS\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"CONNECTION_STATUS_NOTIFY\0".as_ptr() as *mut u8) as *const u8;
__t0[16] = (b"ENUM_PATH_RESOURCES\0".as_ptr() as *mut u8) as *const u8;
__t0[17] = (b"ALLOCATE_PAYLOAD\0".as_ptr() as *mut u8) as *const u8;
__t0[18] = (b"QUERY_PAYLOAD\0".as_ptr() as *mut u8) as *const u8;
__t0[19] = (b"RESOURCE_STATUS_NOTIFY\0".as_ptr() as *mut u8) as *const u8;
__t0[20] = (b"CLEAR_PAYLOAD_ID_TABLE\0".as_ptr() as *mut u8) as *const u8;
__t0[32] = (b"REMOTE_DPCD_READ\0".as_ptr() as *mut u8) as *const u8;
__t0[33] = (b"REMOTE_DPCD_WRITE\0".as_ptr() as *mut u8) as *const u8;
__t0[34] = (b"REMOTE_I2C_READ\0".as_ptr() as *mut u8) as *const u8;
__t0[35] = (b"REMOTE_I2C_WRITE\0".as_ptr() as *mut u8) as *const u8;
__t0[36] = (b"POWER_UP_PHY\0".as_ptr() as *mut u8) as *const u8;
__t0[37] = (b"POWER_DOWN_PHY\0".as_ptr() as *mut u8) as *const u8;
__t0[48] = (b"SINK_EVENT_NOTIFY\0".as_ptr() as *mut u8) as *const u8;
__t0[56] = (b"QUERY_STREAM_ENC_STATUS\0".as_ptr() as *mut u8) as *const u8;
__t0
});

fn drm_dp_mst_nak_reason_str(nak_reason: u8) -> *const u8 {
if ((nak_reason as i32) as u64) >= ARRAY_SIZE__27!() || unsafe { nak_reason_str.0[(nak_reason as i32) as usize] }.is_null() {
return c"unknown".as_ptr() as *const u8;
}
unsafe { nak_reason_str.0[(nak_reason as i32) as usize] }
}

static nak_reason_str: __SlateAlign16<[*const u8; 11]> = __SlateAlign16({
let mut __t0: [*const u8; 11] = unsafe { core::mem::zeroed() };
__t0[1] = (b"WRITE_FAILURE\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"INVALID_READ\0".as_ptr() as *mut u8) as *const u8;
__t0[3] = (b"CRC_FAILURE\0".as_ptr() as *mut u8) as *const u8;
__t0[4] = (b"BAD_PARAM\0".as_ptr() as *mut u8) as *const u8;
__t0[5] = (b"DEFER\0".as_ptr() as *mut u8) as *const u8;
__t0[6] = (b"LINK_FAILURE\0".as_ptr() as *mut u8) as *const u8;
__t0[7] = (b"NO_RESOURCES\0".as_ptr() as *mut u8) as *const u8;
__t0[8] = (b"DPCD_FAIL\0".as_ptr() as *mut u8) as *const u8;
__t0[9] = (b"I2C_NAK\0".as_ptr() as *mut u8) as *const u8;
__t0[10] = (b"ALLOCATE_FAIL\0".as_ptr() as *mut u8) as *const u8;
__t0
});

fn drm_dp_mst_sideband_tx_state_str(state: i32) -> *const u8 {
if (state as u64) >= ARRAY_SIZE__23!() || unsafe { sideband_reason_str.0[state as usize] }.is_null() {
return c"unknown".as_ptr() as *const u8;
}
unsafe { sideband_reason_str.0[state as usize] }
}

static sideband_reason_str: __SlateAlign16<[*const u8; 5]> = __SlateAlign16([(b"QUEUED\0".as_ptr() as *mut u8) as *const u8, (b"START_SEND\0".as_ptr() as *mut u8) as *const u8, (b"SENT\0".as_ptr() as *mut u8) as *const u8, (b"RX\0".as_ptr() as *mut u8) as *const u8, (b"TIMEOUT\0".as_ptr() as *mut u8) as *const u8]);

fn drm_dp_mst_get_ufp_num_at_lct_from_rad(lct: u8, rad: *const u8) -> u8 {
let idx: i32 = ((lct as i32) / 2).wrapping_sub(1);
let shift: i32 = if (lct as i32) % 2 != 0 { 0 } else { 4 };
let mut ufp_num: u8 = 0;
// mst_primary, it's rad is unset
if (lct as i32) == 1 {
return 0;
}
ufp_num = (((unsafe { *rad.wrapping_offset(idx as isize) }) as i32) >> shift & 15) as u8;
ufp_num
}

fn drm_dp_mst_rad_to_str(rad: *const u8, lct: u8, out: *mut u8, len: u64) -> i32 {
let mut unpacked_rad: __SlateAlign16<[u8; 16]> = __SlateAlign16({
let mut __t0: [u8; 16] = unsafe { core::mem::zeroed() };
__t0
});
for i in 0..lct as i32 {
unpacked_rad.0[i as usize] = drm_dp_mst_get_ufp_num_at_lct_from_rad(i.wrapping_add(1) as u8, rad);
}
// TODO: Eventually add something to printk so we can format the rad
// like this: 1.2.3
unsafe { snprintf(out, len, c"%*phC".as_ptr() as *const u8, lct as i32, unpacked_rad.0.as_mut_ptr() as *mut u8) }
}

/// sideband msg handling
fn drm_dp_msg_header_crc4(data: *const u8, num_nibbles: u64) -> u8 {
let mut bitmask: u8 = 128;
let mut bitshift: u8 = 7;
let mut array_index: u8 = 0;
let mut number_of_bits: i32 = num_nibbles.wrapping_mul(4) as i32;
let mut remainder: u8 = 0;
while number_of_bits != 0 {
number_of_bits = number_of_bits.wrapping_sub(1);
remainder = ((remainder as i32) << 1i32) as u8;
remainder = ((remainder as i32) | (((unsafe { *data.wrapping_offset((array_index as i32) as isize) }) as i32) & (bitmask as i32)) >> (bitshift as i32)) as u8;
bitmask = ((bitmask as i32) >> 1i32) as u8;
bitshift = (bitshift as i32).wrapping_sub(1) as u8;
if (bitmask as i32) == 0 {
bitmask = 128;
bitshift = 7;
array_index = (array_index as i32).wrapping_add(1) as u8;
}
if (remainder as i32) & 16 == 16 {
remainder = ((remainder as i32) ^ 19) as u8;
}
}
number_of_bits = 4;
while number_of_bits != 0 {
number_of_bits = number_of_bits.wrapping_sub(1);
remainder = ((remainder as i32) << 1i32) as u8;
if (remainder as i32) & 16 != 0 {
remainder = ((remainder as i32) ^ 19) as u8;
}
}
remainder
}

fn drm_dp_msg_data_crc4(data: *const u8, number_of_bytes: u8) -> u8 {
let mut bitmask: u8 = 128;
let mut bitshift: u8 = 7;
let mut array_index: u8 = 0;
let mut number_of_bits: i32 = (number_of_bytes as i32).wrapping_mul(8);
let mut remainder: u16 = 0;
while number_of_bits != 0 {
number_of_bits = number_of_bits.wrapping_sub(1);
remainder = ((remainder as i32) << 1i32) as u16;
remainder = ((remainder as i32) | (((unsafe { *data.wrapping_offset((array_index as i32) as isize) }) as i32) & (bitmask as i32)) >> (bitshift as i32)) as u16;
bitmask = ((bitmask as i32) >> 1i32) as u8;
bitshift = (bitshift as i32).wrapping_sub(1) as u8;
if (bitmask as i32) == 0 {
bitmask = 128;
bitshift = 7;
array_index = (array_index as i32).wrapping_add(1) as u8;
}
if (remainder as i32) & 256 == 256 {
remainder = ((remainder as i32) ^ 213) as u16;
}
}
number_of_bits = 8;
while number_of_bits != 0 {
number_of_bits = number_of_bits.wrapping_sub(1);
remainder = ((remainder as i32) << 1i32) as u16;
if (remainder as i32) & 256 != 0 {
remainder = ((remainder as i32) ^ 213) as u16;
}
}
((remainder as i32) & 255) as u8
}

fn drm_dp_calc_sb_hdr_size(hdr: &drm_dp_sideband_msg_hdr) -> u8 {
let mut size: u8 = 3;
size = (size as i32).wrapping_add(((unsafe { (*(hdr as *const drm_dp_sideband_msg_hdr)).lct }) as i32) / 2) as u8;
size
}

fn drm_dp_encode_sideband_msg_hdr(hdr: &mut drm_dp_sideband_msg_hdr, buf: *mut u8, len: &mut i32) {
let mut idx: i32 = 0;
let mut i: i32 = 0;
let mut crc4: u8 = 0;
let __v0: i32 = idx;
idx = __v0.wrapping_add(1);
unsafe {
*buf.wrapping_offset(__v0 as isize) = ((((*(hdr as *mut drm_dp_sideband_msg_hdr)).lct as i32) & 15) << 4i32 | ((*(hdr as *mut drm_dp_sideband_msg_hdr)).lcr as i32) & 15) as u8;
}
while i < ((unsafe { (*(hdr as *mut drm_dp_sideband_msg_hdr)).lct }) as i32) / 2 {
let __v2: i32 = idx;
idx = __v2.wrapping_add(1);
unsafe {
*buf.wrapping_offset(__v2 as isize) = *((*(hdr as *mut drm_dp_sideband_msg_hdr)).rad.as_mut_ptr() as *mut u8).wrapping_offset(i as isize);
}
i = i.wrapping_add(1);
}
let __v7: i32 = idx;
idx = __v7.wrapping_add(1);
unsafe {
*buf.wrapping_offset(__v7 as isize) = (((*(hdr as *mut drm_dp_sideband_msg_hdr)).broadcast as i32) << 7i32 | ((*(hdr as *mut drm_dp_sideband_msg_hdr)).path_msg as i32) << 6i32 | ((*(hdr as *mut drm_dp_sideband_msg_hdr)).msg_len as i32) & 63) as u8;
}
let __v9: i32 = idx;
idx = __v9.wrapping_add(1);
unsafe {
*buf.wrapping_offset(__v9 as isize) = (((*(hdr as *mut drm_dp_sideband_msg_hdr)).somt as i32) << 7i32 | ((*(hdr as *mut drm_dp_sideband_msg_hdr)).eomt as i32) << 6i32 | ((*(hdr as *mut drm_dp_sideband_msg_hdr)).seqno as i32) << 4i32) as u8;
}
crc4 = drm_dp_msg_header_crc4(buf as *const u8, idx.wrapping_mul(2).wrapping_sub(1) as u64);
let __v11: *mut u8 = buf.wrapping_offset(idx.wrapping_sub(1) as isize);
unsafe {
*__v11 = ((*__v11 as i32) | (crc4 as i32) & 15) as u8;
}
unsafe {
*(len as *mut i32) = idx;
}
}

fn drm_dp_decode_sideband_msg_hdr(mgr: *const drm_dp_mst_topology_mgr, hdr: &mut drm_dp_sideband_msg_hdr, buf: *mut u8, buflen: i32, hdrlen: &mut u8) -> bool {
let mut crc4: u8 = 0;
let mut len: u8 = 3;
let mut i: i32 = 0;
let mut idx: u8 = 1;
if ((unsafe { *buf.wrapping_offset(0 as isize) }) as i32) == 0 {
return false;
}
len = (len as i32).wrapping_add(((((unsafe { *buf.wrapping_offset(0 as isize) }) as i32) & 240) >> 4i32) / 2) as u8;
if (len as i32) > buflen {
return false;
}
crc4 = drm_dp_msg_header_crc4(buf as *const u8, (len as i32).wrapping_mul(2).wrapping_sub(1) as u64);
if (crc4 as i32) & 15 != ((unsafe { *buf.wrapping_offset((len as i32).wrapping_sub(1) as isize) }) as i32) & 15 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"crc4 mismatch 0x%x 0x%x\n".as_ptr() as *const u8, crc4 as i32, *buf.wrapping_offset((len as i32).wrapping_sub(1) as isize) as i32) };
return false;
}
unsafe {
(*(hdr as *mut drm_dp_sideband_msg_hdr)).lct = (((*buf.wrapping_offset(0 as isize) as i32) & 240) >> 4i32) as u8;
}
unsafe {
(*(hdr as *mut drm_dp_sideband_msg_hdr)).lcr = ((*buf.wrapping_offset(0 as isize) as i32) & 15) as u8;
}
while i < ((unsafe { (*(hdr as *mut drm_dp_sideband_msg_hdr)).lct }) as i32) / 2 {
let __v2: u8 = idx;
idx = (__v2 as i32).wrapping_add(1) as u8;
unsafe {
*((*(hdr as *mut drm_dp_sideband_msg_hdr)).rad.as_mut_ptr() as *mut u8).wrapping_offset(i as isize) = *buf.wrapping_offset((__v2 as i32) as isize);
}
i = i.wrapping_add(1);
}
unsafe {
(*(hdr as *mut drm_dp_sideband_msg_hdr)).broadcast = (*buf.wrapping_offset((idx as i32) as isize) as i32) >> 7i32 & 1 != 0;
}
unsafe {
(*(hdr as *mut drm_dp_sideband_msg_hdr)).path_msg = (*buf.wrapping_offset((idx as i32) as isize) as i32) >> 6i32 & 1 != 0;
}
unsafe {
(*(hdr as *mut drm_dp_sideband_msg_hdr)).msg_len = ((*buf.wrapping_offset((idx as i32) as isize) as i32) & 63) as u8;
}
if ((unsafe { (*(hdr as *mut drm_dp_sideband_msg_hdr)).msg_len }) as i32) < 1 {
return false;
}
// min space for body CRC
idx = (idx as i32).wrapping_add(1) as u8;
unsafe {
(*(hdr as *mut drm_dp_sideband_msg_hdr)).somt = (*buf.wrapping_offset((idx as i32) as isize) as i32) >> 7i32 & 1 != 0;
}
unsafe {
(*(hdr as *mut drm_dp_sideband_msg_hdr)).eomt = (*buf.wrapping_offset((idx as i32) as isize) as i32) >> 6i32 & 1 != 0;
}
unsafe {
(*(hdr as *mut drm_dp_sideband_msg_hdr)).seqno = (*buf.wrapping_offset((idx as i32) as isize) as i32) >> 4i32 & 1 != 0;
}
idx = (idx as i32).wrapping_add(1) as u8;
unsafe {
*(hdrlen as *mut u8) = idx;
}
true
}

// Decode a sideband request we've encoded, mainly used for debugging
#[unsafe(no_mangle)]
extern "C" fn drm_dp_dump_sideband_msg_req_body(req: *const drm_dp_sideband_msg_req_body, mut indent: i32, printer: *mut drm_printer) {
let mut i: i32 = 0;
if ((unsafe { (*req).req_type }) as i32) == 1 {
// No contents to print
unsafe { drm_printf(printer, c"%.*stype=%s\n".as_ptr() as *const u8, indent, c"\t\t\t\t\tX".as_ptr() as *mut u8, drm_dp_mst_req_type_str((*req).req_type)) };
return;
}
unsafe { drm_printf(printer, c"%.*stype=%s contents:\n".as_ptr() as *const u8, indent, c"\t\t\t\t\tX".as_ptr() as *mut u8, drm_dp_mst_req_type_str((*req).req_type)) };
indent = indent.wrapping_add(1);
match (unsafe { (*req).req_type }) as i32 {
16 | 36 | 37 => {
unsafe { drm_printf(printer, c"%.*sport=%d\n".as_ptr() as *const u8, indent, c"\t\t\t\t\tX".as_ptr() as *mut u8, (*req).u.port_num.port_number as i32) };
}
17 => {
unsafe { drm_printf(printer, c"%.*sport=%d vcpi=%d pbn=%d sdp_streams=%d %*ph\n".as_ptr() as *const u8, indent, c"\t\t\t\t\tX".as_ptr() as *mut u8, (*req).u.allocate_payload.port_number as i32, (*req).u.allocate_payload.vcpi as i32, (*req).u.allocate_payload.pbn as i32, (*req).u.allocate_payload.number_sdp_streams as i32, (*req).u.allocate_payload.number_sdp_streams as i32, (*req).u.allocate_payload.sdp_stream_sink.as_ptr() as *const u8) };
}
18 => {
unsafe { drm_printf(printer, c"%.*sport=%d vcpi=%d\n".as_ptr() as *const u8, indent, c"\t\t\t\t\tX".as_ptr() as *mut u8, (*req).u.query_payload.port_number as i32, (*req).u.query_payload.vcpi as i32) };
}
32 => {
unsafe { drm_printf(printer, c"%.*sport=%d dpcd_addr=%05x len=%d\n".as_ptr() as *const u8, indent, c"\t\t\t\t\tX".as_ptr() as *mut u8, (*req).u.dpcd_read.port_number as i32, (*req).u.dpcd_read.dpcd_address, (*req).u.dpcd_read.num_bytes as i32) };
}
33 => {
unsafe { drm_printf(printer, c"%.*sport=%d addr=%05x len=%d: %*ph\n".as_ptr() as *const u8, indent, c"\t\t\t\t\tX".as_ptr() as *mut u8, (*req).u.dpcd_write.port_number as i32, (*req).u.dpcd_write.dpcd_address, (*req).u.dpcd_write.num_bytes as i32, (*req).u.dpcd_write.num_bytes as i32, (*req).u.dpcd_write.bytes) };
}
34 => {
unsafe { drm_printf(printer, c"%.*sport=%d num_tx=%d id=%d size=%d:\n".as_ptr() as *const u8, indent, c"\t\t\t\t\tX".as_ptr() as *mut u8, (*req).u.i2c_read.port_number as i32, (*req).u.i2c_read.num_transactions as i32, (*req).u.i2c_read.read_i2c_device_id as i32, (*req).u.i2c_read.num_bytes_read as i32) };
indent = indent.wrapping_add(1);
i = 0;
while i < ((unsafe { (*req).u.i2c_read.num_transactions }) as i32) {
let rtx: *const drm_dp_remote_i2c_read_tx = unsafe { (*req).u.i2c_read.transactions.as_ptr() as *const drm_dp_remote_i2c_read_tx }.wrapping_offset(i as isize);
unsafe { drm_printf(printer, c"%.*s%d: id=%03d size=%03d no_stop_bit=%d tx_delay=%03d: %*ph\n".as_ptr() as *const u8, indent, c"\t\t\t\t\tX".as_ptr() as *mut u8, i, (*rtx).i2c_dev_id as i32, (*rtx).num_bytes as i32, (*rtx).no_stop_bit as i32, (*rtx).i2c_transaction_delay as i32, (*rtx).num_bytes as i32, (*rtx).bytes) };
i = i.wrapping_add(1);
}
}
35 => {
unsafe { drm_printf(printer, c"%.*sport=%d id=%d size=%d: %*ph\n".as_ptr() as *const u8, indent, c"\t\t\t\t\tX".as_ptr() as *mut u8, (*req).u.i2c_write.port_number as i32, (*req).u.i2c_write.write_i2c_device_id as i32, (*req).u.i2c_write.num_bytes as i32, (*req).u.i2c_write.num_bytes as i32, (*req).u.i2c_write.bytes) };
}
56 => {
unsafe { drm_printf(printer, c"%.*sstream_id=%u client_id=%*ph stream_event=%x valid_event=%d stream_behavior=%x valid_behavior=%d".as_ptr() as *const u8, indent, c"\t\t\t\t\tX".as_ptr() as *mut u8, (*req).u.enc_status.stream_id as i32, ARRAY_SIZE__11!() as i32, (*req).u.enc_status.client_id.as_ptr() as *const u8, (*req).u.enc_status.stream_event as i32, (*req).u.enc_status.valid_stream_event as i32, (*req).u.enc_status.stream_behavior as i32, (*req).u.enc_status.valid_stream_behavior as i32) };
}
_ => {
unsafe { drm_printf(printer, c"%.*s???\n".as_ptr() as *const u8, indent, c"\t\t\t\t\tX".as_ptr() as *mut u8) };
}
}
}

fn drm_dp_mst_dump_sideband_msg_tx(p: *mut drm_printer, txmsg: *const drm_dp_sideband_msg_tx) {
let mut req: drm_dp_sideband_msg_req_body = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(req) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 120usize);
}
let mut buf: __SlateAlign16<[u8; 64]> = __SlateAlign16([0u8; 64]);
let mut ret: i32 = 0;
let mut i: i32 = 0;
drm_dp_mst_rad_to_str((unsafe { (*(*txmsg).dst).rad.as_mut_ptr() as *mut u8 }) as *const u8, unsafe { (*(*txmsg).dst).lct }, buf.0.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 64]>() as u64);
unsafe { drm_printf(p, c"txmsg cur_offset=%x cur_len=%x seqno=%x state=%s path_msg=%d dst=%s\n".as_ptr() as *const u8, (*txmsg).cur_offset as i32, (*txmsg).cur_len as i32, (*txmsg).seqno, drm_dp_mst_sideband_tx_state_str((*txmsg).state), (*txmsg).path_msg as i32, buf.0.as_mut_ptr() as *mut u8) };
ret = drm_dp_decode_sideband_req(txmsg, core::ptr::addr_of_mut!(req));
if ret != 0 {
unsafe { drm_printf(p, c"<failed to decode sideband req: %d>\n".as_ptr() as *const u8, ret) };
return;
}
drm_dp_dump_sideband_msg_req_body(core::ptr::addr_of_mut!(req) as *const drm_dp_sideband_msg_req_body, 1i32, p);
match req.req_type as i32 {
33 => {
unsafe { kfree(req.u.dpcd_write.bytes as *const ()) };
}
34 => {
i = 0;
while i < ((unsafe { req.u.i2c_read.num_transactions }) as i32) {
unsafe { kfree(req.u.i2c_read.transactions[i as usize].bytes as *const ()) };
i = i.wrapping_add(1);
}
}
35 => {
unsafe { kfree(req.u.i2c_write.bytes as *const ()) };
}
_ => {
}
}
}

fn drm_dp_crc_sideband_chunk_req(msg: *mut u8, len: u8) {
let crc4: u8 = drm_dp_msg_data_crc4(msg as *const u8, len);
unsafe {
*msg.wrapping_offset((len as i32) as isize) = crc4;
}
}

fn drm_dp_encode_sideband_reply(rep: &drm_dp_sideband_msg_reply_body, raw: *mut drm_dp_sideband_msg_tx) {
let mut idx: i32 = 0;
let buf: *mut u8 = unsafe { (*raw).msg.as_mut_ptr() as *mut u8 };
let __v0: i32 = idx;
idx = __v0.wrapping_add(1);
unsafe {
*buf.wrapping_offset(__v0 as isize) = ((((*(rep as *const drm_dp_sideband_msg_reply_body)).reply_type as i32) & 1) << 7i32 | ((*(rep as *const drm_dp_sideband_msg_reply_body)).req_type as i32) & 127) as u8;
}
unsafe {
(*raw).cur_len = idx as u8;
}
}

fn drm_dp_sideband_msg_set_header(msg: *mut drm_dp_sideband_msg_rx, hdr: *mut drm_dp_sideband_msg_hdr, hdrlen: u8) -> i32 {
// ignore out-of-order messages or messages that are part of a
// failed transaction
if !unsafe { (*hdr).somt } && !unsafe { (*msg).have_somt } {
return 0;
}
// get length contained in this portion
unsafe {
(*msg).curchunk_idx = 0;
}
unsafe {
(*msg).curchunk_len = (*hdr).msg_len;
}
unsafe {
(*msg).curchunk_hdrlen = hdrlen;
}
// we have already gotten an somt - don't bother parsing
if (unsafe { (*hdr).somt }) && unsafe { (*msg).have_somt } {
return 0;
}
if unsafe { (*hdr).somt } {
unsafe { memcpy(core::ptr::addr_of_mut!((*msg).initial_hdr) as *mut (), hdr as *const (), core::mem::size_of::<drm_dp_sideband_msg_hdr>() as u64) };
unsafe {
(*msg).have_somt = true;
}
}
if unsafe { (*hdr).eomt } {
unsafe {
(*msg).have_eomt = true;
}
}
1
}

/// this adds a chunk of msg to the builder to get the final msg
fn drm_dp_sideband_append_payload(msg: *mut drm_dp_sideband_msg_rx, replybuf: *mut u8, replybuflen: u8) -> bool {
let mut crc4: u8 = 0;
// curchunk_len must be >= 1 (min 1 CRC byte) and fit in chunk[]
if (unsafe { (*msg).curchunk_len }) == 0 || (((unsafe { (*msg).curchunk_len }) as i32) as u64) > ARRAY_SIZE!() || (((unsafe { (*msg).curchunk_idx }) as i32).wrapping_add(replybuflen as i32) as u64) > ARRAY_SIZE__10!() {
return false;
}
unsafe { memcpy(((*msg).chunk.as_mut_ptr() as *mut u8).wrapping_offset(((*msg).curchunk_idx as i32) as isize) as *mut (), replybuf as *const (), replybuflen as u64) };
unsafe {
(*msg).curchunk_idx = ((*msg).curchunk_idx as i32).wrapping_add(replybuflen as i32) as u8;
}
if ((unsafe { (*msg).curchunk_idx }) as i32) >= ((unsafe { (*msg).curchunk_len }) as i32) {
// do CRC
crc4 = drm_dp_msg_data_crc4((unsafe { (*msg).chunk.as_mut_ptr() as *mut u8 }) as *const u8, ((unsafe { (*msg).curchunk_len }) as i32).wrapping_sub(1) as u8);
if (crc4 as i32) != ((unsafe { *((*msg).chunk.as_mut_ptr() as *mut u8).wrapping_offset(((*msg).curchunk_len as i32).wrapping_sub(1) as isize) }) as i32) {
unsafe { print_hex_dump(c"\x017".as_ptr() as *const u8, c"wrong crc".as_ptr() as *const u8, 0i32, 16i32, 1i32, ((*msg).chunk.as_mut_ptr() as *mut u8) as *const (), (*msg).curchunk_len as u64, false) };
}
// Guard against accumulated msg[] overflow
if (((unsafe { (*msg).curlen }) as i32).wrapping_add((unsafe { (*msg).curchunk_len }) as i32).wrapping_sub(1) as u64) > ARRAY_SIZE__12!() {
return false;
}
// copy chunk into bigger msg
unsafe { memcpy(((*msg).msg.as_mut_ptr() as *mut u8).wrapping_offset(((*msg).curlen as i32) as isize) as *mut (), ((*msg).chunk.as_mut_ptr() as *mut u8) as *const (), ((*msg).curchunk_len as i32).wrapping_sub(1) as u64) };
unsafe {
(*msg).curlen = ((*msg).curlen as i32).wrapping_add(((*msg).curchunk_len as i32).wrapping_sub(1)) as u8;
}
}
true
}

fn drm_dp_sideband_parse_link_address(mgr: *const drm_dp_mst_topology_mgr, raw: *mut drm_dp_sideband_msg_rx, repmsg: *mut drm_dp_sideband_msg_reply_body) -> bool {
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut idx: i32 = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
idx = 1;
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
import_guid(core::ptr::addr_of_mut!((*repmsg).u.link_addr.guid), ((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as *const u8);
idx = idx.wrapping_add(16);
(*repmsg).u.link_addr.nports = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) & 15) as u8;
idx = idx.wrapping_add(1);
if idx <= ((*raw).curlen as i32) {
*__slate_slot_16 = 0;
while *__slate_slot_16 < ((*repmsg).u.link_addr.nports as i32) {
if (*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) & 128 != 0 {
(*((*repmsg).u.link_addr.ports.as_mut_ptr() as *mut drm_dp_link_addr_reply_port).wrapping_offset(*__slate_slot_16 as isize)).input_port = true;
}
(*((*repmsg).u.link_addr.ports.as_mut_ptr() as *mut drm_dp_link_addr_reply_port).wrapping_offset(*__slate_slot_16 as isize)).peer_device_type = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) >> 4i32 & 7) as u8;
(*((*repmsg).u.link_addr.ports.as_mut_ptr() as *mut drm_dp_link_addr_reply_port).wrapping_offset(*__slate_slot_16 as isize)).port_number = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) & 15) as u8;
idx = idx.wrapping_add(1);
if idx > ((*raw).curlen as i32) {
break '__join_0;
} else {
(*((*repmsg).u.link_addr.ports.as_mut_ptr() as *mut drm_dp_link_addr_reply_port).wrapping_offset(*__slate_slot_16 as isize)).mcs = (*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) >> 7i32 & 1 != 0;
(*((*repmsg).u.link_addr.ports.as_mut_ptr() as *mut drm_dp_link_addr_reply_port).wrapping_offset(*__slate_slot_16 as isize)).ddps = (*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) >> 6i32 & 1 != 0;
if ((*((*repmsg).u.link_addr.ports.as_mut_ptr() as *mut drm_dp_link_addr_reply_port).wrapping_offset(*__slate_slot_16 as isize)).input_port as i32) == 0 {
(*((*repmsg).u.link_addr.ports.as_mut_ptr() as *mut drm_dp_link_addr_reply_port).wrapping_offset(*__slate_slot_16 as isize)).legacy_device_plug_status = (*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) >> 5i32 & 1 != 0;
}
idx = idx.wrapping_add(1);
if idx > ((*raw).curlen as i32) {
break '__join_0;
} else {
if ((*((*repmsg).u.link_addr.ports.as_mut_ptr() as *mut drm_dp_link_addr_reply_port).wrapping_offset(*__slate_slot_16 as isize)).input_port as i32) == 0 {
(*((*repmsg).u.link_addr.ports.as_mut_ptr() as *mut drm_dp_link_addr_reply_port).wrapping_offset(*__slate_slot_16 as isize)).dpcd_revision = *((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize);
idx = idx.wrapping_add(1);
if idx > ((*raw).curlen as i32) {
break '__join_0;
} else {
import_guid(core::ptr::addr_of_mut!((*((*repmsg).u.link_addr.ports.as_mut_ptr() as *mut drm_dp_link_addr_reply_port).wrapping_offset(*__slate_slot_16 as isize)).peer_guid), ((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as *const u8);
idx = idx.wrapping_add(16);
if idx > ((*raw).curlen as i32) {
break '__join_0;
} else {
(*((*repmsg).u.link_addr.ports.as_mut_ptr() as *mut drm_dp_link_addr_reply_port).wrapping_offset(*__slate_slot_16 as isize)).num_sdp_streams = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) >> 4i32 & 15) as u8;
(*((*repmsg).u.link_addr.ports.as_mut_ptr() as *mut drm_dp_link_addr_reply_port).wrapping_offset(*__slate_slot_16 as isize)).num_sdp_stream_sinks = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) & 15) as u8;
idx = idx.wrapping_add(1);
}
}
}
if idx > ((*raw).curlen as i32) {
break '__join_0;
} else {
let __v1: i32 = *__slate_slot_16;
*__slate_slot_16 = __v1.wrapping_add(1);
}
}
}
}
return true;
}
}
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"link address reply parse length fail %d %d\n".as_ptr() as *const u8, idx, (*raw).curlen as i32);
return false;
}
}

fn drm_dp_sideband_parse_remote_dpcd_read(raw: *mut drm_dp_sideband_msg_rx, repmsg: *mut drm_dp_sideband_msg_reply_body) -> bool {
let mut idx: i32 = unsafe { core::mem::zeroed() };
unsafe {
idx = 1;
(*repmsg).u.remote_dpcd_read_ack.port_number = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) & 15) as u8;
idx = idx.wrapping_add(1);
if idx <= ((*raw).curlen as i32) {
(*repmsg).u.remote_dpcd_read_ack.num_bytes = *((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize);
idx = idx.wrapping_add(1);
if idx.wrapping_add((*repmsg).u.remote_dpcd_read_ack.num_bytes as i32) <= ((*raw).curlen as i32) {
memcpy(((*repmsg).u.remote_dpcd_read_ack.bytes.as_mut_ptr() as *mut u8) as *mut (), ((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as *const (), (*repmsg).u.remote_dpcd_read_ack.num_bytes as u64);
return true;
}
}
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"link address reply parse length fail %d %d\n".as_ptr() as *const u8, idx, (*raw).curlen as i32);
return false;
}
}

fn drm_dp_sideband_parse_remote_dpcd_write(raw: *mut drm_dp_sideband_msg_rx, repmsg: *mut drm_dp_sideband_msg_reply_body) -> bool {
let mut idx: i32 = unsafe { core::mem::zeroed() };
unsafe {
idx = 1;
(*repmsg).u.remote_dpcd_write_ack.port_number = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) & 15) as u8;
idx = idx.wrapping_add(1);
if idx > ((*raw).curlen as i32) {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"parse length fail %d %d\n".as_ptr() as *const u8, idx, (*raw).curlen as i32);
return false;
} else {
return true;
}
}
}

fn drm_dp_sideband_parse_remote_i2c_read_ack(raw: *mut drm_dp_sideband_msg_rx, repmsg: *mut drm_dp_sideband_msg_reply_body) -> bool {
let mut idx: i32 = unsafe { core::mem::zeroed() };
unsafe {
idx = 1;
(*repmsg).u.remote_i2c_read_ack.port_number = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) & 15) as u8;
idx = idx.wrapping_add(1);
if idx <= ((*raw).curlen as i32) {
(*repmsg).u.remote_i2c_read_ack.num_bytes = *((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize);
idx = idx.wrapping_add(1);
if idx.wrapping_add((*repmsg).u.remote_i2c_read_ack.num_bytes as i32) <= ((*raw).curlen as i32) {
memcpy(((*repmsg).u.remote_i2c_read_ack.bytes.as_mut_ptr() as *mut u8) as *mut (), ((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as *const (), (*repmsg).u.remote_i2c_read_ack.num_bytes as u64);
return true;
}
}
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"remote i2c reply parse length fail %d %d\n".as_ptr() as *const u8, idx, (*raw).curlen as i32);
return false;
}
}

fn drm_dp_sideband_parse_enum_path_resources_ack(raw: *mut drm_dp_sideband_msg_rx, repmsg: *mut drm_dp_sideband_msg_reply_body) -> bool {
let mut idx: i32 = unsafe { core::mem::zeroed() };
unsafe {
idx = 1;
(*repmsg).u.path_resources.port_number = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) >> 4i32 & 15) as u8;
(*repmsg).u.path_resources.fec_capable = (*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) & 1 != 0;
idx = idx.wrapping_add(1);
if idx.wrapping_add(2) <= ((*raw).curlen as i32) {
(*repmsg).u.path_resources.full_payload_bw_number = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) << 8i32 | (*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx.wrapping_add(1) as isize) as i32)) as u16;
idx = idx.wrapping_add(2);
if idx.wrapping_add(2) <= ((*raw).curlen as i32) {
(*repmsg).u.path_resources.avail_payload_bw_number = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) << 8i32 | (*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx.wrapping_add(1) as isize) as i32)) as u16;
return true;
}
}
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"enum resource parse length fail %d %d\n".as_ptr() as *const u8, idx, (*raw).curlen as i32);
return false;
}
}

fn drm_dp_sideband_parse_allocate_payload_ack(raw: *mut drm_dp_sideband_msg_rx, repmsg: *mut drm_dp_sideband_msg_reply_body) -> bool {
let mut idx: i32 = unsafe { core::mem::zeroed() };
unsafe {
idx = 1;
(*repmsg).u.allocate_payload.port_number = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) >> 4i32 & 15) as u8;
idx = idx.wrapping_add(1);
if idx <= ((*raw).curlen as i32) {
(*repmsg).u.allocate_payload.vcpi = *((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize);
idx = idx.wrapping_add(1);
if idx.wrapping_add(2) <= ((*raw).curlen as i32) {
(*repmsg).u.allocate_payload.allocated_pbn = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) << 8i32 | (*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx.wrapping_add(1) as isize) as i32)) as u16;
return true;
}
}
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"allocate payload parse length fail %d %d\n".as_ptr() as *const u8, idx, (*raw).curlen as i32);
return false;
}
}

fn drm_dp_sideband_parse_query_payload_ack(raw: *mut drm_dp_sideband_msg_rx, repmsg: *mut drm_dp_sideband_msg_reply_body) -> bool {
let mut idx: i32 = unsafe { core::mem::zeroed() };
unsafe {
idx = 1;
(*repmsg).u.query_payload.port_number = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) >> 4i32 & 15) as u8;
idx = idx.wrapping_add(1);
if idx.wrapping_add(2) > ((*raw).curlen as i32) {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"query payload parse length fail %d %d\n".as_ptr() as *const u8, idx, (*raw).curlen as i32);
return false;
} else {
(*repmsg).u.query_payload.allocated_pbn = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) << 8i32 | (*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx.wrapping_add(1) as isize) as i32)) as u16;
return true;
}
}
}

fn drm_dp_sideband_parse_power_updown_phy_ack(raw: *mut drm_dp_sideband_msg_rx, repmsg: *mut drm_dp_sideband_msg_reply_body) -> bool {
let mut idx: i32 = 1;
unsafe {
(*repmsg).u.port_number.port_number = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) >> 4i32 & 15) as u8;
}
idx = idx.wrapping_add(1);
if idx > ((unsafe { (*raw).curlen }) as i32) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"power up/down phy parse length fail %d %d\n".as_ptr() as *const u8, idx, (*raw).curlen as i32) };
return false;
}
true
}

fn drm_dp_sideband_parse_query_stream_enc_status(raw: *mut drm_dp_sideband_msg_rx, repmsg: *mut drm_dp_sideband_msg_reply_body) -> bool {
let reply: *mut drm_dp_query_stream_enc_status_ack_reply = unsafe { core::ptr::addr_of_mut!((*repmsg).u.enc_status) };
unsafe {
(*reply).stream_id = *((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize);
}
unsafe {
(*reply).reply_signed = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) as i32) as u64) & BIT!(0i32) != 0;
}
// NOTE: It's my impression from reading the spec that the below parsing
// is correct. However I noticed while testing with an HDCP 1.4 display
// through an HDCP 2.2 hub that only bit 3 was set. In that case, I
// would expect both bits to be set. So keep the parsing following the
// spec, but beware reality might not match the spec (at least for some
// configurations).
unsafe {
(*reply).hdcp_1x_device_present = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) as i32) as u64) & BIT!(4i32) != 0;
}
unsafe {
(*reply).hdcp_2x_device_present = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) as i32) as u64) & BIT!(3i32) != 0;
}
unsafe {
(*reply).query_capable_device_present = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) as i32) as u64) & BIT!(5i32) != 0;
}
unsafe {
(*reply).legacy_device_present = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) as i32) as u64) & BIT!(6i32) != 0;
}
unsafe {
(*reply).unauthorizable_device_present = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) as i32) as u64) & BIT!(7i32) != 0;
}
unsafe {
(*reply).auth_completed = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) as i32) as u64) & BIT!(3i32) != 0;
}
unsafe {
(*reply).encryption_enabled = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) as i32) as u64) & BIT!(4i32) != 0;
}
unsafe {
(*reply).repeater_present = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) as i32) as u64) & BIT!(5i32) != 0;
}
unsafe {
(*reply).state = ((((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) as i32) as u64) & GENMASK__2!(7i32, 6i32)) >> 6i32) as u8;
}
true
}

fn drm_dp_sideband_parse_reply(mgr: *const drm_dp_mst_topology_mgr, raw: *mut drm_dp_sideband_msg_rx, msg: *mut drm_dp_sideband_msg_reply_body) -> bool {
unsafe { memset(msg as *mut (), 0i32, core::mem::size_of::<drm_dp_sideband_msg_reply_body>() as u64) };
unsafe {
(*msg).reply_type = (((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32) & 128) >> 7i32) as u8;
}
unsafe {
(*msg).req_type = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32) & 127) as u8;
}
if ((unsafe { (*msg).reply_type }) as i32) == 1 {
import_guid(unsafe { core::ptr::addr_of_mut!((*msg).u.nak.guid) }, unsafe { (*raw).msg.as_mut_ptr() as *mut u8 }.wrapping_offset(1 as isize) as *const u8);
unsafe {
(*msg).u.nak.reason = *((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(17 as isize);
}
unsafe {
(*msg).u.nak.nak_data = *((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(18 as isize);
}
return false;
}
match (unsafe { (*msg).req_type }) as i32 {
1 => {
return drm_dp_sideband_parse_link_address(mgr, raw, msg);
}
18 => {
return drm_dp_sideband_parse_query_payload_ack(raw, msg);
}
32 => {
return drm_dp_sideband_parse_remote_dpcd_read(raw, msg);
}
33 => {
return drm_dp_sideband_parse_remote_dpcd_write(raw, msg);
}
34 => {
return drm_dp_sideband_parse_remote_i2c_read_ack(raw, msg);
}
35 => {
return true; // since there's nothing to parse
}
16 => {
return drm_dp_sideband_parse_enum_path_resources_ack(raw, msg);
}
17 => {
return drm_dp_sideband_parse_allocate_payload_ack(raw, msg);
}
36 | 37 => {
return drm_dp_sideband_parse_power_updown_phy_ack(raw, msg);
}
20 => {
return true; // since there's nothing to parse
}
56 => {
return drm_dp_sideband_parse_query_stream_enc_status(raw, msg);
}
_ => {
while false {
}
unsafe { _dev_err(__drm_to_dev((*mgr).dev as *const drm_device) as *const device, c"[drm] *ERROR* Got unknown reply 0x%02x (%s)\n".as_ptr() as *const u8, (*msg).req_type as i32, drm_dp_mst_req_type_str((*msg).req_type)) };
return false;
}
}
unsafe { core::mem::zeroed() }
}

fn drm_dp_sideband_parse_connection_status_notify(mgr: *const drm_dp_mst_topology_mgr, raw: *mut drm_dp_sideband_msg_rx, msg: *mut drm_dp_sideband_msg_req_body) -> bool {
let mut idx: i32 = unsafe { core::mem::zeroed() };
unsafe {
idx = 1;
(*msg).u.conn_stat.port_number = (((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) & 240) >> 4i32) as u8;
idx = idx.wrapping_add(1);
if idx <= ((*raw).curlen as i32) {
import_guid(core::ptr::addr_of_mut!((*msg).u.conn_stat.guid), ((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as *const u8);
idx = idx.wrapping_add(16);
if idx <= ((*raw).curlen as i32) {
(*msg).u.conn_stat.legacy_device_plug_status = (*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) >> 6i32 & 1 != 0;
(*msg).u.conn_stat.displayport_device_plug_status = (*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) >> 5i32 & 1 != 0;
(*msg).u.conn_stat.message_capability_status = (*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) >> 4i32 & 1 != 0;
(*msg).u.conn_stat.input_port = (*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) >> 3i32 & 1 != 0;
(*msg).u.conn_stat.peer_device_type = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) & 7) as u8;
idx = idx.wrapping_add(1);
return true;
}
}
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"connection status reply parse length fail %d %d\n".as_ptr() as *const u8, idx, (*raw).curlen as i32);
return false;
}
}

fn drm_dp_sideband_parse_resource_status_notify(mgr: *const drm_dp_mst_topology_mgr, raw: *mut drm_dp_sideband_msg_rx, msg: *mut drm_dp_sideband_msg_req_body) -> bool {
let mut idx: i32 = unsafe { core::mem::zeroed() };
unsafe {
idx = 1;
(*msg).u.resource_stat.port_number = (((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) & 240) >> 4i32) as u8;
idx = idx.wrapping_add(1);
if idx <= ((*raw).curlen as i32) {
import_guid(core::ptr::addr_of_mut!((*msg).u.resource_stat.guid), ((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as *const u8);
idx = idx.wrapping_add(16);
if idx <= ((*raw).curlen as i32) {
(*msg).u.resource_stat.available_pbn = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as i32) << 8i32 | (*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(idx.wrapping_add(1) as isize) as i32)) as u16;
idx = idx.wrapping_add(1);
return true;
}
}
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"resource status reply parse length fail %d %d\n".as_ptr() as *const u8, idx, (*raw).curlen as i32);
return false;
}
}

fn drm_dp_sideband_parse_req(mgr: *const drm_dp_mst_topology_mgr, raw: *mut drm_dp_sideband_msg_rx, msg: *mut drm_dp_sideband_msg_req_body) -> bool {
unsafe { memset(msg as *mut (), 0i32, core::mem::size_of::<drm_dp_sideband_msg_req_body>() as u64) };
unsafe {
(*msg).req_type = ((*((*raw).msg.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32) & 127) as u8;
}
match (unsafe { (*msg).req_type }) as i32 {
2 => {
return drm_dp_sideband_parse_connection_status_notify(mgr, raw, msg);
}
19 => {
return drm_dp_sideband_parse_resource_status_notify(mgr, raw, msg);
}
_ => {
while false {
}
unsafe { _dev_err(__drm_to_dev((*mgr).dev as *const drm_device) as *const device, c"[drm] *ERROR* Got unknown request 0x%02x (%s)\n".as_ptr() as *const u8, (*msg).req_type as i32, drm_dp_mst_req_type_str((*msg).req_type)) };
return false;
}
}
unsafe { core::mem::zeroed() }
}

fn build_dpcd_write(msg: *mut drm_dp_sideband_msg_tx, port_num: u8, offset: u32, num_bytes: u8, bytes: *mut u8) {
let mut req: drm_dp_sideband_msg_req_body = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(req) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 120usize);
}
req.req_type = 33;
unsafe {
req.u.dpcd_write.port_number = port_num;
}
unsafe {
req.u.dpcd_write.dpcd_address = offset;
}
unsafe {
req.u.dpcd_write.num_bytes = num_bytes;
}
unsafe {
req.u.dpcd_write.bytes = bytes;
}
drm_dp_encode_sideband_req(core::ptr::addr_of_mut!(req) as *const drm_dp_sideband_msg_req_body, msg);
}

fn build_link_address(msg: *mut drm_dp_sideband_msg_tx) {
let mut req: drm_dp_sideband_msg_req_body = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(req) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 120usize);
}
req.req_type = 1;
drm_dp_encode_sideband_req(core::ptr::addr_of_mut!(req) as *const drm_dp_sideband_msg_req_body, msg);
}

fn build_clear_payload_id_table(msg: *mut drm_dp_sideband_msg_tx) {
let mut req: drm_dp_sideband_msg_req_body = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(req) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 120usize);
}
req.req_type = 20;
drm_dp_encode_sideband_req(core::ptr::addr_of_mut!(req) as *const drm_dp_sideband_msg_req_body, msg);
unsafe {
(*msg).path_msg = true;
}
}

fn build_enum_path_resources(msg: *mut drm_dp_sideband_msg_tx, port_num: i32) -> i32 {
let mut req: drm_dp_sideband_msg_req_body = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(req) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 120usize);
}
req.req_type = 16;
unsafe {
req.u.port_num.port_number = port_num as u8;
}
drm_dp_encode_sideband_req(core::ptr::addr_of_mut!(req) as *const drm_dp_sideband_msg_req_body, msg);
unsafe {
(*msg).path_msg = true;
}
0
}

fn build_allocate_payload(msg: *mut drm_dp_sideband_msg_tx, port_num: i32, vcpi: u8, pbn: u16, number_sdp_streams: u8, sdp_stream_sink: *mut u8) {
let mut req: drm_dp_sideband_msg_req_body = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(req) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 120usize);
}
unsafe { memset(core::ptr::addr_of_mut!(req) as *mut (), 0i32, core::mem::size_of::<drm_dp_sideband_msg_req_body>() as u64) };
req.req_type = 17;
unsafe {
req.u.allocate_payload.port_number = port_num as u8;
}
unsafe {
req.u.allocate_payload.vcpi = vcpi;
}
unsafe {
req.u.allocate_payload.pbn = pbn;
}
unsafe {
req.u.allocate_payload.number_sdp_streams = number_sdp_streams;
}
unsafe { memcpy((req.u.allocate_payload.sdp_stream_sink.as_mut_ptr() as *mut u8) as *mut (), sdp_stream_sink as *const (), number_sdp_streams as u64) };
drm_dp_encode_sideband_req(core::ptr::addr_of_mut!(req) as *const drm_dp_sideband_msg_req_body, msg);
unsafe {
(*msg).path_msg = true;
}
}

fn build_power_updown_phy(msg: *mut drm_dp_sideband_msg_tx, port_num: i32, power_up: bool) {
let mut req: drm_dp_sideband_msg_req_body = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(req) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 120usize);
}
if power_up {
req.req_type = 36;
} else {
req.req_type = 37;
}
unsafe {
req.u.port_num.port_number = port_num as u8;
}
drm_dp_encode_sideband_req(core::ptr::addr_of_mut!(req) as *const drm_dp_sideband_msg_req_body, msg);
unsafe {
(*msg).path_msg = true;
}
}

fn build_query_stream_enc_status(msg: *mut drm_dp_sideband_msg_tx, stream_id: u8, q_id: *mut u8) -> i32 {
let mut req: drm_dp_sideband_msg_req_body = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(req) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 120usize);
}
req.req_type = 56;
unsafe {
req.u.enc_status.stream_id = stream_id;
}
unsafe { memcpy((req.u.enc_status.client_id.as_mut_ptr() as *mut u8) as *mut (), q_id as *const (), core::mem::size_of::<[u8; 7]>() as u64) };
unsafe {
req.u.enc_status.stream_event = 0;
}
unsafe {
req.u.enc_status.valid_stream_event = false;
}
unsafe {
req.u.enc_status.stream_behavior = 0;
}
unsafe {
req.u.enc_status.valid_stream_behavior = 0;
}
drm_dp_encode_sideband_req(core::ptr::addr_of_mut!(req) as *const drm_dp_sideband_msg_req_body, msg);
0
}

fn check_txmsg_state(mgr: *mut drm_dp_mst_topology_mgr, txmsg: *mut drm_dp_sideband_msg_tx) -> bool {
let mut state: u32 = 0;
// All updates to txmsg->state are protected by mgr->qlock, and the two
// cases we check here are terminal states. For those the barriers
// provided by the wake_up/wait_event pair are enough.
let mut __v0: i32;
loop {
if !(__native_word__12!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_694() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*txmsg).state) as *const i32) };
state = __v0 as u32;
state == 3 || state == 4
}

fn drm_dp_mst_wait_tx_reply(mstb: *mut drm_dp_mst_branch, txmsg: *mut drm_dp_sideband_msg_tx) -> i32 {
let mut p: core::mem::MaybeUninit<drm_printer> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut drm_printer = core::ptr::addr_of_mut!(p) as *mut drm_printer;
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __cond: bool = unsafe { core::mem::zeroed() };
let mut __cond_1: bool = unsafe { core::mem::zeroed() };
let mut __v8: u64 = unsafe { core::mem::zeroed() };
let mut __ret: i64 = unsafe { core::mem::zeroed() };
let mut __wq_entry: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut wait_queue_entry = core::ptr::addr_of_mut!(__wq_entry) as *mut wait_queue_entry;
let mut __v12: u64 = unsafe { core::mem::zeroed() };
let mut __ret_1: i64 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut wait_expires: u64 = unsafe { core::mem::zeroed() };
let mut wait_timeout: u64 = unsafe { core::mem::zeroed() };
let mut mgr: *mut drm_dp_mst_topology_mgr = unsafe { core::mem::zeroed() };
unsafe {
mgr = (*mstb).mgr;
wait_timeout = msecs_to_jiffies(4000u32);
wait_expires = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(wait_timeout);
let __slate_auto_bytes_1 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
loop {
if (*(*mgr).cbs).poll_hpd_irq != None {
__v12 = msecs_to_jiffies(50u32);
} else {
__v12 = wait_timeout;
}
__ret_1 = __v12 as i64;
might_sleep!();
if !___wait_cond_timeout!(check_txmsg_state(mgr, txmsg), __ret_1) {
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
if (*(*mgr).cbs).poll_hpd_irq != None {
__v8 = msecs_to_jiffies(50u32);
} else {
__v8 = wait_timeout;
}
__ret = __v8 as i64;
init_wait_entry(core::ptr::addr_of_mut!(*__slate_slot_10), if false { 1i32 } else { 0i32 });
loop {
prepare_to_wait_event(core::ptr::addr_of_mut!((*mgr).tx_waitq), core::ptr::addr_of_mut!(*__slate_slot_10), 2i32);
__cond_1 = check_txmsg_state(mgr, txmsg);
if __cond_1 && __ret == 0 {
__ret = 1;
}
let __v6: bool = __cond_1 || __ret == 0;
if __v6 {
break;
} else {
__ret = schedule_timeout(__ret);
__cond = check_txmsg_state(mgr, txmsg);
if __cond && __ret == 0 {
__ret = 1;
}
if __cond || __ret == 0 {
break;
}
}
}
finish_wait(core::ptr::addr_of_mut!((*mgr).tx_waitq), core::ptr::addr_of_mut!(*__slate_slot_10));
__ret_1 = __ret;
}
// If the driver provides a way for this, change to
// poll-waiting for the MST reply interrupt if we didn't receive
// it for 50 msec. This would cater for cases where the HPD
// pulse signal got lost somewhere, even though the sink raised
// the corresponding MST interrupt correctly. One example is the
// Club 3D CAC-1557 TypeC -> DP adapter which for some reason
// filters out short pulses with a duration less than ~540 usec.
//
// The poll period is 50 msec to avoid missing an interrupt
// after the sink has cleared it (after a 110msec timeout
// since it raised the interrupt).
*__slate_slot_15 = __ret_1 as i32;
if *__slate_slot_15 != 0 || (*(*mgr).cbs).poll_hpd_irq == None {
__v2 = true;
} else {
__v2 = time_after!(core::ptr::read_volatile(core::ptr::addr_of!(jiffies)), wait_expires);
}
if __v2 {
break;
} else {
unsafe { (*(*mgr).cbs).poll_hpd_irq }.unwrap()(mgr);
}
}
mutex_lock(core::ptr::addr_of_mut!((*mgr).qlock));
if *__slate_slot_15 > 0 {
if (*txmsg).state == 4 {
*__slate_slot_15 = 5i32.wrapping_neg();
}
} else {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"timedout msg send %p %d %d\n".as_ptr() as *const u8, txmsg, (*txmsg).state, (*txmsg).seqno);
// dump some state
*__slate_slot_15 = 5i32.wrapping_neg();
// remove from q
if (*txmsg).state == 0 || (*txmsg).state == 1 || (*txmsg).state == 2 {
list_del(core::ptr::addr_of_mut!((*txmsg).next));
}
}
if unlikely__2!(*__slate_slot_15 == 5i32.wrapping_neg()) != 0 {
__v1 = drm_debug_enabled!(8i32);
} else {
__v1 = false;
}
if __v1 {
core::ptr::write(__slate_slot_0, drm_dbg_printer((*mgr).dev, 8u32, c"[dp_mst]".as_ptr() as *const u8));
drm_dp_mst_dump_sideband_msg_tx(core::ptr::addr_of_mut!(*__slate_slot_0), txmsg as *const drm_dp_sideband_msg_tx);
}
mutex_unlock(core::ptr::addr_of_mut!((*mgr).qlock));
drm_dp_mst_kick_tx(mgr);
return *__slate_slot_15;
}
}

fn drm_dp_add_mst_branch_device(lct: u8, rad: *mut u8) -> *mut drm_dp_mst_branch {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut mstb: *mut drm_dp_mst_branch = unsafe { core::mem::zeroed() };
let mut __v0: *mut drm_dp_mst_branch;
let __obj_size: u64 = size_mul(core::mem::size_of::<drm_dp_mst_branch>() as u64, 1u64);
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
__v0 = __v1 as *mut drm_dp_mst_branch;
mstb = __v0;
if mstb.is_null() {
return core::ptr::null_mut::<drm_dp_mst_branch>();
}
unsafe {
(*mstb).lct = lct;
}
if (lct as i32) > 1 {
unsafe { memcpy(((*mstb).rad.as_mut_ptr() as *mut u8) as *mut (), rad as *const (), ((lct as i32) / 2) as u64) };
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*mstb).ports) });
kref_init(unsafe { core::ptr::addr_of_mut!((*mstb).topology_kref) });
kref_init(unsafe { core::ptr::addr_of_mut!((*mstb).malloc_kref) });
mstb
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_free_mst_branch_device")]
extern "C" fn drm_dp_free_mst_branch_device(kref: *mut kref) {
let mstb: *mut drm_dp_mst_branch = container_of__6!(kref, drm_dp_mst_branch);
if !unsafe { (*mstb).port_parent }.is_null() {
drm_dp_mst_put_port_malloc(unsafe { (*mstb).port_parent });
}
unsafe { kfree(mstb as *const ()) };
}

// DOC: Branch device and port refcounting
//
// Topology refcount overview
//
// The refcounting schemes for &struct drm_dp_mst_branch and &struct
// drm_dp_mst_port are somewhat unusual. Both ports and branch devices have
// two different kinds of refcounts: topology refcounts, and malloc refcounts.
//
// Topology refcounts are not exposed to drivers, and are handled internally
// by the DP MST helpers. The helpers use them in order to prevent the
// in-memory topology state from being changed in the middle of critical
// operations like changing the internal state of payload allocations. This
// means each branch and port will be considered to be connected to the rest
// of the topology until its topology refcount reaches zero. Additionally,
// for ports this means that their associated &struct drm_connector will stay
// registered with userspace until the port's refcount reaches 0.
//
// Malloc refcount overview
//
// Malloc references are used to keep a &struct drm_dp_mst_port or &struct
// drm_dp_mst_branch allocated even after all of its topology references have
// been dropped, so that the driver or MST helpers can safely access each
// branch's last known state before it was disconnected from the topology.
// When the malloc refcount of a port or branch reaches 0, the memory
// allocation containing the &struct drm_dp_mst_branch or &struct
// drm_dp_mst_port respectively will be freed.
//
// For &struct drm_dp_mst_branch, malloc refcounts are not currently exposed
// to drivers. As of writing this documentation, there are no drivers that
// have a usecase for accessing &struct drm_dp_mst_branch outside of the MST
// helpers. Exposing this API to drivers in a race-free manner would take more
// tweaking of the refcounting scheme, however patches are welcome provided
// there is a legitimate driver usecase for this.
//
// Refcount relationships in a topology
//
// Let's take a look at why the relationship between topology and malloc
// refcounts is designed the way it is.
//
// .. kernel-figure:: dp-mst/topology-figure-1.dot
//
//    An example of topology and malloc refs in a DP MST topology with two
//    active payloads. Topology refcount increments are indicated by solid
//    lines, and malloc refcount increments are indicated by dashed lines.
//    Each starts from the branch which incremented the refcount, and ends at
//    the branch to which the refcount belongs to, i.e. the arrow points the
//    same way as the C pointers used to reference a structure.
//
// As you can see in the above figure, every branch increments the topology
// refcount of its children, and increments the malloc refcount of its
// parent. Additionally, every payload increments the malloc refcount of its
// assigned port by 1.
//
// So, what would happen if MSTB #3 from the above figure was unplugged from
// the system, but the driver hadn't yet removed payload #2 from port #3? The
// topology would start to look like the figure below.
//
// .. kernel-figure:: dp-mst/topology-figure-2.dot
//
//    Ports and branch devices which have been released from memory are
//    colored grey, and references which have been removed are colored red.
//
// Whenever a port or branch device's topology refcount reaches zero, it will
// decrement the topology refcounts of all its children, the malloc refcount
// of its parent, and finally its own malloc refcount. For MSTB #4 and port
// #4, this means they both have been disconnected from the topology and freed
// from memory. But, because payload #2 is still holding a reference to port
// #3, port #3 is removed from the topology but its &struct drm_dp_mst_port
// is still accessible from memory. This also means port #3 has not yet
// decremented the malloc refcount of MSTB #3, so its &struct
// drm_dp_mst_branch will also stay allocated in memory until port #3's
// malloc refcount reaches 0.
//
// This relationship is necessary because in order to release payload #2, we
// need to be able to figure out the last relative of port #3 that's still
// connected to the topology. In this case, we would travel up the topology as
// shown below.
//
// .. kernel-figure:: dp-mst/topology-figure-3.dot
//
// And finally, remove payload #2 by communicating with port #2 through
// sideband transactions.
/// drm_dp_mst_get_mstb_malloc() - Increment the malloc refcount of a branch
/// device
/// @mstb: The &struct drm_dp_mst_branch to increment the malloc refcount of
///
/// Increments &drm_dp_mst_branch.malloc_kref. When
/// &drm_dp_mst_branch.malloc_kref reaches 0, the memory allocation for @mstb
/// will be released and @mstb may no longer be used.
///
/// See also: drm_dp_mst_put_mstb_malloc()
fn drm_dp_mst_get_mstb_malloc(mstb: *mut drm_dp_mst_branch) {
kref_get(unsafe { core::ptr::addr_of_mut!((*mstb).malloc_kref) });
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*mstb).mgr).dev as *const drm_device) as *const device, 1u32, c"mstb %p (%d)\n".as_ptr() as *const u8, mstb, kref_read(core::ptr::addr_of_mut!((*mstb).malloc_kref) as *const kref)) };
}

/// drm_dp_mst_put_mstb_malloc() - Decrement the malloc refcount of a branch
/// device
/// @mstb: The &struct drm_dp_mst_branch to decrement the malloc refcount of
///
/// Decrements &drm_dp_mst_branch.malloc_kref. When
/// &drm_dp_mst_branch.malloc_kref reaches 0, the memory allocation for @mstb
/// will be released and @mstb may no longer be used.
///
/// See also: drm_dp_mst_get_mstb_malloc()
fn drm_dp_mst_put_mstb_malloc(mstb: *mut drm_dp_mst_branch) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*mstb).mgr).dev as *const drm_device) as *const device, 1u32, c"mstb %p (%d)\n".as_ptr() as *const u8, mstb, kref_read(core::ptr::addr_of_mut!((*mstb).malloc_kref) as *const kref).wrapping_sub(1)) };
kref_put(unsafe { core::ptr::addr_of_mut!((*mstb).malloc_kref) }, Some(drm_dp_free_mst_branch_device));
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_free_mst_port")]
extern "C" fn drm_dp_free_mst_port(kref: *mut kref) {
let port: *mut drm_dp_mst_port = container_of__6!(kref, drm_dp_mst_port);
drm_dp_mst_put_mstb_malloc(unsafe { (*port).parent });
unsafe { kfree(port as *const ()) };
}

/// drm_dp_mst_get_port_malloc() - Increment the malloc refcount of an MST port
/// @port: The &struct drm_dp_mst_port to increment the malloc refcount of
///
/// Increments &drm_dp_mst_port.malloc_kref. When &drm_dp_mst_port.malloc_kref
/// reaches 0, the memory allocation for @port will be released and @port may
/// no longer be used.
///
/// Because @port could potentially be freed at any time by the DP MST helpers
/// if &drm_dp_mst_port.malloc_kref reaches 0, including during a call to this
/// function, drivers that which to make use of &struct drm_dp_mst_port should
/// ensure that they grab at least one main malloc reference to their MST ports
/// in &drm_dp_mst_topology_cbs.add_connector. This callback is called before
/// there is any chance for &drm_dp_mst_port.malloc_kref to reach 0.
///
/// See also: drm_dp_mst_put_port_malloc()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_get_port_malloc")]
extern "C" fn drm_dp_mst_get_port_malloc(port: *mut drm_dp_mst_port) {
kref_get(unsafe { core::ptr::addr_of_mut!((*port).malloc_kref) });
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*port).mgr).dev as *const drm_device) as *const device, 1u32, c"port %p (%d)\n".as_ptr() as *const u8, port, kref_read(core::ptr::addr_of_mut!((*port).malloc_kref) as *const kref)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_get_port_malloc_695: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_port)>, *mut ()>(Some(drm_dp_mst_get_port_malloc)) };

/// drm_dp_mst_put_port_malloc() - Decrement the malloc refcount of an MST port
/// @port: The &struct drm_dp_mst_port to decrement the malloc refcount of
///
/// Decrements &drm_dp_mst_port.malloc_kref. When &drm_dp_mst_port.malloc_kref
/// reaches 0, the memory allocation for @port will be released and @port may
/// no longer be used.
///
/// See also: drm_dp_mst_get_port_malloc()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_put_port_malloc")]
extern "C" fn drm_dp_mst_put_port_malloc(port: *mut drm_dp_mst_port) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*port).mgr).dev as *const drm_device) as *const device, 1u32, c"port %p (%d)\n".as_ptr() as *const u8, port, kref_read(core::ptr::addr_of_mut!((*port).malloc_kref) as *const kref).wrapping_sub(1)) };
kref_put(unsafe { core::ptr::addr_of_mut!((*port).malloc_kref) }, Some(drm_dp_free_mst_port));
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_put_port_malloc_696: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_port)>, *mut ()>(Some(drm_dp_mst_put_port_malloc)) };

fn topology_ref_history_lock(mgr: *mut drm_dp_mst_topology_mgr) {
}

fn topology_ref_history_unlock(mgr: *mut drm_dp_mst_topology_mgr) {
}

fn drm_dp_mst_dump_mstb_topology_history(mstb: *mut drm_dp_mst_branch) {
}

fn drm_dp_mst_dump_port_topology_history(port: *mut drm_dp_mst_port) {
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_atomic_get_mst_payload_state")]
extern "C" fn drm_atomic_get_mst_payload_state(state: *mut drm_dp_mst_topology_state, port: *mut drm_dp_mst_port) -> *mut drm_dp_mst_atomic_payload {
let mut payload: *mut drm_dp_mst_atomic_payload = unsafe { core::mem::zeroed() };
payload = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*state).payloads) }, next);
while list_entry_is_head!(payload, unsafe { core::ptr::addr_of_mut!((*state).payloads) }, next) == 0 {
if (unsafe { (*payload).port }) == port {
return payload;
}
payload = list_next_entry__2!(payload, next);
}
core::ptr::null_mut::<drm_dp_mst_atomic_payload>()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_get_mst_payload_state_697: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_state, *mut drm_dp_mst_port) -> *mut drm_dp_mst_atomic_payload>, *mut ()>(Some(drm_atomic_get_mst_payload_state)) };

#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_destroy_mst_branch_device")]
extern "C" fn drm_dp_destroy_mst_branch_device(kref: *mut kref) {
let mstb: *mut drm_dp_mst_branch = container_of__4!(kref, drm_dp_mst_branch);
let mgr: *mut drm_dp_mst_topology_mgr = unsafe { (*mstb).mgr };
drm_dp_mst_dump_mstb_topology_history(mstb);
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*mstb).destroy_next) });
// This can get called under mgr->mutex, so we need to perform the
// actual destruction of the mstb in another worker
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).delayed_destroy_lock)) };
list_add(unsafe { core::ptr::addr_of_mut!((*mstb).destroy_next) }, unsafe { core::ptr::addr_of_mut!((*mgr).destroy_branch_device_list) });
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).delayed_destroy_lock)) };
queue_work(unsafe { (*mgr).delayed_destroy_wq }, unsafe { core::ptr::addr_of_mut!((*mgr).delayed_destroy_work) });
}

/// drm_dp_mst_topology_try_get_mstb() - Increment the topology refcount of a
/// branch device unless it's zero
/// @mstb: &struct drm_dp_mst_branch to increment the topology refcount of
///
/// Attempts to grab a topology reference to @mstb, if it hasn't yet been
/// removed from the topology (e.g. &drm_dp_mst_branch.topology_kref has
/// reached 0). Holding a topology reference implies that a malloc reference
/// will be held to @mstb as long as the user holds the topology reference.
///
/// Care should be taken to ensure that the user has at least one malloc
/// reference to @mstb. If you already have a topology reference to @mstb, you
/// should use drm_dp_mst_topology_get_mstb() instead.
///
/// See also:
/// drm_dp_mst_topology_get_mstb()
/// drm_dp_mst_topology_put_mstb()
///
/// Returns:
/// * 1: A topology reference was grabbed successfully
/// * 0: @port is no longer in the topology, no reference was grabbed
fn drm_dp_mst_topology_try_get_mstb(mstb: *mut drm_dp_mst_branch) -> i32 {
let mut ret: i32 = 0;
topology_ref_history_lock(unsafe { (*mstb).mgr });
ret = kref_get_unless_zero(unsafe { core::ptr::addr_of_mut!((*mstb).topology_kref) });
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*mstb).mgr).dev as *const drm_device) as *const device, 1u32, c"mstb %p (%d)\n".as_ptr() as *const u8, mstb, kref_read(core::ptr::addr_of_mut!((*mstb).topology_kref) as *const kref)) };
}
topology_ref_history_unlock(unsafe { (*mstb).mgr });
ret
}

/// drm_dp_mst_topology_get_mstb() - Increment the topology refcount of a
/// branch device
/// @mstb: The &struct drm_dp_mst_branch to increment the topology refcount of
///
/// Increments &drm_dp_mst_branch.topology_refcount without checking whether or
/// not it's already reached 0. This is only valid to use in scenarios where
/// you are already guaranteed to have at least one active topology reference
/// to @mstb. Otherwise, drm_dp_mst_topology_try_get_mstb() must be used.
///
/// See also:
/// drm_dp_mst_topology_try_get_mstb()
/// drm_dp_mst_topology_put_mstb()
fn drm_dp_mst_topology_get_mstb(mstb: *mut drm_dp_mst_branch) {
topology_ref_history_lock(unsafe { (*mstb).mgr });
let mut __v0: i64;
let __ret_warn_on: i32 = (kref_read((unsafe { core::ptr::addr_of_mut!((*mstb).topology_kref) }) as *const kref) == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("698: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 698b - ., 3; .popsection\n# {0}", const 698 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_dp_mst_topology_get_mstb_1, sym __slate_asm_string_drm_dp_mst_topology_get_mstb_2, const 1798 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("699: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 699b - ., 4; .popsection\n# {0}", const 699 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
kref_get(unsafe { core::ptr::addr_of_mut!((*mstb).topology_kref) });
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*mstb).mgr).dev as *const drm_device) as *const device, 1u32, c"mstb %p (%d)\n".as_ptr() as *const u8, mstb, kref_read(core::ptr::addr_of_mut!((*mstb).topology_kref) as *const kref)) };
topology_ref_history_unlock(unsafe { (*mstb).mgr });
}

/// drm_dp_mst_topology_put_mstb() - release a topology reference to a branch
/// device
/// @mstb: The &struct drm_dp_mst_branch to release the topology reference from
///
/// Releases a topology reference from @mstb by decrementing
/// &drm_dp_mst_branch.topology_kref.
///
/// See also:
/// drm_dp_mst_topology_try_get_mstb()
/// drm_dp_mst_topology_get_mstb()
fn drm_dp_mst_topology_put_mstb(mstb: *mut drm_dp_mst_branch) {
topology_ref_history_lock(unsafe { (*mstb).mgr });
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*mstb).mgr).dev as *const drm_device) as *const device, 1u32, c"mstb %p (%d)\n".as_ptr() as *const u8, mstb, kref_read(core::ptr::addr_of_mut!((*mstb).topology_kref) as *const kref).wrapping_sub(1)) };
topology_ref_history_unlock(unsafe { (*mstb).mgr });
kref_put(unsafe { core::ptr::addr_of_mut!((*mstb).topology_kref) }, Some(drm_dp_destroy_mst_branch_device));
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_destroy_port")]
extern "C" fn drm_dp_destroy_port(kref: *mut kref) {
let port: *mut drm_dp_mst_port = container_of__4!(kref, drm_dp_mst_port);
let mgr: *mut drm_dp_mst_topology_mgr = unsafe { (*port).mgr };
drm_dp_mst_dump_port_topology_history(port);
// There's nothing that needs locking to destroy an input port yet
if unsafe { (*port).input } {
drm_dp_mst_put_port_malloc(port);
return;
}
unsafe { drm_edid_free((*port).cached_edid) };
// we can't destroy the connector here, as we might be holding the
// mode_config.mutex from an EDID retrieval
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).delayed_destroy_lock)) };
list_add(unsafe { core::ptr::addr_of_mut!((*port).next) }, unsafe { core::ptr::addr_of_mut!((*mgr).destroy_port_list) });
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).delayed_destroy_lock)) };
queue_work(unsafe { (*mgr).delayed_destroy_wq }, unsafe { core::ptr::addr_of_mut!((*mgr).delayed_destroy_work) });
}

/// drm_dp_mst_topology_try_get_port() - Increment the topology refcount of a
/// port unless it's zero
/// @port: &struct drm_dp_mst_port to increment the topology refcount of
///
/// Attempts to grab a topology reference to @port, if it hasn't yet been
/// removed from the topology (e.g. &drm_dp_mst_port.topology_kref has reached
/// 0). Holding a topology reference implies that a malloc reference will be
/// held to @port as long as the user holds the topology reference.
///
/// Care should be taken to ensure that the user has at least one malloc
/// reference to @port. If you already have a topology reference to @port, you
/// should use drm_dp_mst_topology_get_port() instead.
///
/// See also:
/// drm_dp_mst_topology_get_port()
/// drm_dp_mst_topology_put_port()
///
/// Returns:
/// * 1: A topology reference was grabbed successfully
/// * 0: @port is no longer in the topology, no reference was grabbed
fn drm_dp_mst_topology_try_get_port(port: *mut drm_dp_mst_port) -> i32 {
let mut ret: i32 = 0;
topology_ref_history_lock(unsafe { (*port).mgr });
ret = kref_get_unless_zero(unsafe { core::ptr::addr_of_mut!((*port).topology_kref) });
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*port).mgr).dev as *const drm_device) as *const device, 1u32, c"port %p (%d)\n".as_ptr() as *const u8, port, kref_read(core::ptr::addr_of_mut!((*port).topology_kref) as *const kref)) };
}
topology_ref_history_unlock(unsafe { (*port).mgr });
ret
}

/// drm_dp_mst_topology_get_port() - Increment the topology refcount of a port
/// @port: The &struct drm_dp_mst_port to increment the topology refcount of
///
/// Increments &drm_dp_mst_port.topology_refcount without checking whether or
/// not it's already reached 0. This is only valid to use in scenarios where
/// you are already guaranteed to have at least one active topology reference
/// to @port. Otherwise, drm_dp_mst_topology_try_get_port() must be used.
///
/// See also:
/// drm_dp_mst_topology_try_get_port()
/// drm_dp_mst_topology_put_port()
fn drm_dp_mst_topology_get_port(port: *mut drm_dp_mst_port) {
topology_ref_history_lock(unsafe { (*port).mgr });
let mut __v0: i64;
let __ret_warn_on: i32 = (kref_read((unsafe { core::ptr::addr_of_mut!((*port).topology_kref) }) as *const kref) == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("700: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 700b - ., 3; .popsection\n# {0}", const 700 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_dp_mst_topology_get_port_1, sym __slate_asm_string_drm_dp_mst_topology_get_port_2, const 1910 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("701: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 701b - ., 4; .popsection\n# {0}", const 701 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
kref_get(unsafe { core::ptr::addr_of_mut!((*port).topology_kref) });
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*port).mgr).dev as *const drm_device) as *const device, 1u32, c"port %p (%d)\n".as_ptr() as *const u8, port, kref_read(core::ptr::addr_of_mut!((*port).topology_kref) as *const kref)) };
topology_ref_history_unlock(unsafe { (*port).mgr });
}

/// drm_dp_mst_topology_put_port() - release a topology reference to a port
/// @port: The &struct drm_dp_mst_port to release the topology reference from
///
/// Releases a topology reference from @port by decrementing
/// &drm_dp_mst_port.topology_kref.
///
/// See also:
/// drm_dp_mst_topology_try_get_port()
/// drm_dp_mst_topology_get_port()
fn drm_dp_mst_topology_put_port(port: *mut drm_dp_mst_port) {
topology_ref_history_lock(unsafe { (*port).mgr });
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*port).mgr).dev as *const drm_device) as *const device, 1u32, c"port %p (%d)\n".as_ptr() as *const u8, port, kref_read(core::ptr::addr_of_mut!((*port).topology_kref) as *const kref).wrapping_sub(1)) };
topology_ref_history_unlock(unsafe { (*port).mgr });
kref_put(unsafe { core::ptr::addr_of_mut!((*port).topology_kref) }, Some(drm_dp_destroy_port));
}

fn drm_dp_mst_topology_get_mstb_validated_locked(mstb: *mut drm_dp_mst_branch, to_find: *mut drm_dp_mst_branch) -> *mut drm_dp_mst_branch {
let mut port: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
if to_find == mstb {
return mstb;
}
port = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next);
while list_entry_is_head!(port, unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next) == 0 {
if !unsafe { (*port).mstb }.is_null() {
let rmstb: *mut drm_dp_mst_branch = drm_dp_mst_topology_get_mstb_validated_locked(unsafe { (*port).mstb }, to_find);
if !rmstb.is_null() {
return rmstb;
}
}
port = list_next_entry!(port, next);
}
core::ptr::null_mut::<drm_dp_mst_branch>()
}

fn drm_dp_mst_topology_get_mstb_validated(mgr: *mut drm_dp_mst_topology_mgr, mstb: *mut drm_dp_mst_branch) -> *mut drm_dp_mst_branch {
let mut rmstb: *mut drm_dp_mst_branch = core::ptr::null_mut::<drm_dp_mst_branch>();
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).lock)) };
if !unsafe { (*mgr).mst_primary }.is_null() {
rmstb = drm_dp_mst_topology_get_mstb_validated_locked(unsafe { (*mgr).mst_primary }, mstb);
let __v0: bool = !rmstb.is_null() && drm_dp_mst_topology_try_get_mstb(rmstb) == 0;
if __v0 {
rmstb = core::ptr::null_mut::<drm_dp_mst_branch>();
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock)) };
rmstb
}

fn drm_dp_mst_topology_get_port_validated_locked(mstb: *mut drm_dp_mst_branch, to_find: *mut drm_dp_mst_port) -> *mut drm_dp_mst_port {
let mut port: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
port = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next);
while list_entry_is_head!(port, unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next) == 0 {
if port == to_find {
return port;
}
if !unsafe { (*port).mstb }.is_null() {
let mport: *mut drm_dp_mst_port = drm_dp_mst_topology_get_port_validated_locked(unsafe { (*port).mstb }, to_find);
if !mport.is_null() {
return mport;
}
}
port = list_next_entry!(port, next);
}
core::ptr::null_mut::<drm_dp_mst_port>()
}

fn drm_dp_mst_topology_get_port_validated(mgr: *mut drm_dp_mst_topology_mgr, port: *mut drm_dp_mst_port) -> *mut drm_dp_mst_port {
let mut rport: *mut drm_dp_mst_port = core::ptr::null_mut::<drm_dp_mst_port>();
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).lock)) };
if !unsafe { (*mgr).mst_primary }.is_null() {
rport = drm_dp_mst_topology_get_port_validated_locked(unsafe { (*mgr).mst_primary }, port);
let __v0: bool = !rport.is_null() && drm_dp_mst_topology_try_get_port(rport) == 0;
if __v0 {
rport = core::ptr::null_mut::<drm_dp_mst_port>();
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock)) };
rport
}

fn drm_dp_get_port(mstb: *mut drm_dp_mst_branch, port_num: u8) -> *mut drm_dp_mst_port {
let mut port: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
port = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next);
while list_entry_is_head!(port, unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next) == 0 {
if ((unsafe { (*port).port_num }) as i32) == (port_num as i32) {
ret = drm_dp_mst_topology_try_get_port(port);
return if ret != 0 { port } else { core::ptr::null_mut::<drm_dp_mst_port>() };
}
port = list_next_entry!(port, next);
}
core::ptr::null_mut::<drm_dp_mst_port>()
}

/// calculate a new RAD for this MST branch device
/// if parent has an LCT of 2 then it has 1 nibble of RAD,
/// if parent has an LCT of 3 then it has 2 nibbles of RAD,
fn drm_dp_calculate_rad(port: *mut drm_dp_mst_port, rad: *mut u8) -> u8 {
let parent_lct: i32 = (unsafe { (*(*port).parent).lct }) as i32;
let mut shift: i32 = 4;
let idx: i32 = parent_lct.wrapping_sub(1) / 2;
if parent_lct > 1 {
unsafe { memcpy(rad as *mut (), ((*(*port).parent).rad.as_mut_ptr() as *mut u8) as *const (), idx.wrapping_add(1) as u64) };
shift = if parent_lct % 2 != 0 { 4 } else { 0 };
} else {
unsafe {
*rad.wrapping_offset(0 as isize) = 0;
}
}
let __v0: *mut u8 = rad.wrapping_offset(idx as isize);
unsafe {
*__v0 = ((*__v0 as i32) | ((*port).port_num as i32) << shift) as u8;
}
parent_lct.wrapping_add(1) as u8
}

fn drm_dp_mst_is_end_device(pdt: u8, mcs: bool) -> bool {
match pdt as i32 {
3 | 4 => {
return true;
}
2 => {
if !mcs {
return true;
}
// For sst branch device
return false;
}
_ => {
}
}
true
}

fn drm_dp_port_set_pdt(port: *mut drm_dp_mst_port, new_pdt: u8, new_mcs: bool) -> i32 {
// If the new PDT would also have an i2c bus,
// don't bother with reregistering it
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut lct: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u8 = core::ptr::addr_of_mut!(lct) as *mut u8;
let mut rad: core::mem::MaybeUninit<[u8; 8]> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut [u8; 8] = core::ptr::addr_of_mut!(rad) as *mut [u8; 8];
let mut mstb: core::mem::MaybeUninit<*mut drm_dp_mst_branch> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut drm_dp_mst_branch = core::ptr::addr_of_mut!(mstb) as *mut *mut drm_dp_mst_branch;
let mut mgr: *mut drm_dp_mst_topology_mgr = unsafe { core::mem::zeroed() };
unsafe {
mgr = (*port).mgr;
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
ret = 0;
if ((*port).pdt as i32) == (new_pdt as i32) && ((*port).mcs as i32) == (new_mcs as i32) {
return 0;
} else {
// Teardown the old pdt, if there is one
if ((*port).pdt as i32) != 0 {
if drm_dp_mst_is_end_device((*port).pdt, (*port).mcs) {
if (new_pdt as i32) != 0 {
__v0 = drm_dp_mst_is_end_device(new_pdt, new_mcs);
} else {
__v0 = false;
}
if __v0 {
(*port).pdt = new_pdt;
(*port).mcs = new_mcs;
return 0;
} else {
// remove i2c over sideband
drm_dp_mst_unregister_i2c_bus(port);
}
} else {
mutex_lock(core::ptr::addr_of_mut!((*mgr).lock));
drm_dp_mst_topology_put_mstb((*port).mstb);
(*port).mstb = core::ptr::null_mut::<drm_dp_mst_branch>();
mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock));
}
}
(*port).pdt = new_pdt;
(*port).mcs = new_mcs;
if ((*port).pdt as i32) != 0 {
if drm_dp_mst_is_end_device((*port).pdt, (*port).mcs) {
// add i2c over sideband
ret = drm_dp_mst_register_i2c_bus(port);
} else {
*__slate_slot_2 = drm_dp_calculate_rad(port, (*__slate_slot_3).as_mut_ptr() as *mut u8);
*__slate_slot_4 = drm_dp_add_mst_branch_device(*__slate_slot_2, (*__slate_slot_3).as_mut_ptr() as *mut u8);
if (*__slate_slot_4).is_null() {
ret = 12i32.wrapping_neg();
while false {
}
_dev_err(__drm_to_dev((*mgr).dev as *const drm_device) as *const device, c"[drm] *ERROR* Failed to create MSTB for port %p".as_ptr() as *const u8, port);
} else {
mutex_lock(core::ptr::addr_of_mut!((*mgr).lock));
(*port).mstb = *__slate_slot_4;
(*(*__slate_slot_4)).mgr = (*port).mgr;
(*(*__slate_slot_4)).port_parent = port;
// Make sure this port's memory allocation stays
// around until its child MSTB releases it
drm_dp_mst_get_port_malloc(port);
mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock));
// And make sure we send a link address for this
ret = 1;
}
}
}
if ret < 0 {
(*port).pdt = 0;
}
return ret;
}
}
}

/// drm_dp_mst_dpcd_read() - read a series of bytes from the DPCD via sideband
/// @aux: Fake sideband AUX CH
/// @offset: address of the (first) register to read
/// @buffer: buffer to store the register values
/// @size: number of bytes in @buffer
///
/// Performs the same functionality for remote devices via
/// sideband messaging as drm_dp_dpcd_read() does for local
/// devices via actual AUX CH.
///
/// Return: Number of bytes read, or negative error code on failure.
#[unsafe(no_mangle)]
extern "C" fn drm_dp_mst_dpcd_read(aux: *mut drm_dp_aux, offset: u32, buffer: *mut (), size: u64) -> i64 {
let port: *mut drm_dp_mst_port = container_of__3!(aux, drm_dp_mst_port);
drm_dp_send_dpcd_read(unsafe { (*port).mgr }, port, offset as i32, size as i32, buffer as *mut u8) as i64
}

/// drm_dp_mst_dpcd_write() - write a series of bytes to the DPCD via sideband
/// @aux: Fake sideband AUX CH
/// @offset: address of the (first) register to write
/// @buffer: buffer containing the values to write
/// @size: number of bytes in @buffer
///
/// Performs the same functionality for remote devices via
/// sideband messaging as drm_dp_dpcd_write() does for local
/// devices via actual AUX CH.
///
/// Return: number of bytes written on success, negative error code on failure.
#[unsafe(no_mangle)]
extern "C" fn drm_dp_mst_dpcd_write(aux: *mut drm_dp_aux, offset: u32, buffer: *mut (), size: u64) -> i64 {
let port: *mut drm_dp_mst_port = container_of__3!(aux, drm_dp_mst_port);
drm_dp_send_dpcd_write(unsafe { (*port).mgr }, port, offset as i32, size as i32, buffer as *mut u8) as i64
}

fn drm_dp_check_mstb_guid(mstb: *mut drm_dp_mst_branch, guid: *mut __SlateRecord_uuid_h_17) -> i32 {
let mut ret: i32 = 0;
guid_copy(unsafe { core::ptr::addr_of_mut!((*mstb).guid) }, guid as *const __SlateRecord_uuid_h_17);
if !drm_dp_validate_guid(unsafe { (*mstb).mgr }, unsafe { core::ptr::addr_of_mut!((*mstb).guid) }) {
let mut aux: *mut drm_dp_aux = unsafe { core::mem::zeroed() };
let mut buf: __SlateAlign16<[u8; 16]> = __SlateAlign16([0u8; 16]);
export_guid(buf.0.as_mut_ptr() as *mut u8, (unsafe { core::ptr::addr_of_mut!((*mstb).guid) }) as *const __SlateRecord_uuid_h_17);
if !unsafe { (*mstb).port_parent }.is_null() {
aux = unsafe { core::ptr::addr_of_mut!((*(*mstb).port_parent).aux) };
} else {
aux = unsafe { (*(*mstb).mgr).aux };
}
ret = drm_dp_dpcd_write_data(aux, 48u32, buf.0.as_mut_ptr() as *mut (), core::mem::size_of::<[u8; 16]>() as u64);
}
ret
}

fn build_mst_prop_path(mstb: *const drm_dp_mst_branch, pnum: i32, proppath: *mut u8, proppath_size: u64) {
let mut i: i32 = 0;
let mut temp: [u8; 8] = [0u8; 8];
unsafe { snprintf(proppath, proppath_size, c"mst:%d".as_ptr() as *const u8, (*(*mstb).mgr).conn_base_id) };
while i < ((unsafe { (*mstb).lct }) as i32).wrapping_sub(1) {
let shift: i32 = if i % 2 != 0 { 0 } else { 4 };
let port_num: i32 = ((unsafe { *((*mstb).rad.as_ptr() as *const u8).wrapping_offset((i / 2) as isize) }) as i32) >> shift & 15;
unsafe { snprintf(temp.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 8]>() as u64, c"-%d".as_ptr() as *const u8, port_num) };
unsafe { strlcat(proppath, temp.as_mut_ptr() as *const u8, proppath_size) };
i = i.wrapping_add(1);
}
unsafe { snprintf(temp.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 8]>() as u64, c"-%d".as_ptr() as *const u8, pnum) };
unsafe { strlcat(proppath, temp.as_mut_ptr() as *const u8, proppath_size) };
}

/// drm_dp_mst_connector_late_register() - Late MST connector registration
/// @connector: The MST connector
/// @port: The MST port for this connector
///
/// Helper to register the remote aux device for this MST port. Drivers should
/// call this from their mst connector's late_register hook to enable MST aux
/// devices.
///
/// Return: 0 on success, negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_connector_late_register")]
extern "C" fn drm_dp_mst_connector_late_register(connector: *mut drm_connector, port: *mut drm_dp_mst_port) -> i32 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*port).mgr).dev as *const drm_device) as *const device, 2u32, c"registering %s remote bus for %s\n".as_ptr() as *const u8, (*port).aux.name, (*(*connector).kdev).kobj.name) };
unsafe {
(*port).aux.dev = (*connector).kdev;
}
drm_dp_aux_register_devnode(unsafe { core::ptr::addr_of_mut!((*port).aux) })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_connector_late_register_702: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_dp_mst_port) -> i32>, *mut ()>(Some(drm_dp_mst_connector_late_register)) };

/// drm_dp_mst_connector_early_unregister() - Early MST connector unregistration
/// @connector: The MST connector
/// @port: The MST port for this connector
///
/// Helper to unregister the remote aux device for this MST port, registered by
/// drm_dp_mst_connector_late_register(). Drivers should call this from their mst
/// connector's early_unregister hook.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_connector_early_unregister")]
extern "C" fn drm_dp_mst_connector_early_unregister(connector: *mut drm_connector, port: *mut drm_dp_mst_port) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*port).mgr).dev as *const drm_device) as *const device, 2u32, c"unregistering %s remote bus for %s\n".as_ptr() as *const u8, (*port).aux.name, (*(*connector).kdev).kobj.name) };
drm_dp_aux_unregister_devnode(unsafe { core::ptr::addr_of_mut!((*port).aux) });
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_connector_early_unregister_703: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_dp_mst_port)>, *mut ()>(Some(drm_dp_mst_connector_early_unregister)) };

fn drm_dp_mst_port_add_connector(mstb: *mut drm_dp_mst_branch, port: *mut drm_dp_mst_port) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut proppath: core::mem::MaybeUninit<__SlateAlign16<[u8; 255]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut [u8; 255] = core::ptr::addr_of_mut!(proppath) as *mut [u8; 255];
let mut mgr: *mut drm_dp_mst_topology_mgr = unsafe { core::mem::zeroed() };
unsafe {
mgr = (*port).mgr;
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 255usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
build_mst_prop_path(mstb as *const drm_dp_mst_branch, (*port).port_num as i32, (*__slate_slot_3).as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 255]>() as u64);
(*port).connector = unsafe { (*(*mgr).cbs).add_connector }.unwrap()(mgr, port, (*__slate_slot_3).as_mut_ptr() as *const u8);
if unsafe { (*port).connector }.is_null() {
*__slate_slot_2 = 12i32.wrapping_neg();
while false {
}
_dev_err(__drm_to_dev((*mgr).dev as *const drm_device) as *const device, c"[drm] *ERROR* Failed to create connector for port %p: %d\n".as_ptr() as *const u8, port, *__slate_slot_2);
} else {
if ((*port).pdt as i32) != 0 {
__v1 = drm_dp_mst_is_end_device((*port).pdt, (*port).mcs);
} else {
__v1 = false;
}
if __v1 {
__v0 = drm_dp_mst_port_is_logical(port);
} else {
__v0 = false;
}
if __v0 {
(*port).cached_edid = drm_edid_read_ddc((*port).connector, core::ptr::addr_of_mut!((*port).aux.ddc));
}
drm_connector_dynamic_register((*port).connector);
return;
}
}
}

/// Drop a topology reference, and unlink the port from the in-memory topology
/// layout
fn drm_dp_mst_topology_unlink_port(mgr: *mut drm_dp_mst_topology_mgr, port: *mut drm_dp_mst_port) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).lock)) };
let __v0: *mut drm_dp_mst_branch = unsafe { (*port).parent };
let __v1: i32 = unsafe { (*__v0).num_ports };
unsafe {
(*__v0).num_ports = __v1.wrapping_sub(1);
}
list_del(unsafe { core::ptr::addr_of_mut!((*port).next) });
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock)) };
drm_dp_mst_topology_put_port(port);
}

fn drm_dp_mst_add_port(dev: *mut drm_device, mgr: *mut drm_dp_mst_topology_mgr, mstb: *mut drm_dp_mst_branch, port_number: u8) -> *mut drm_dp_mst_port {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut port: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
let mut __v0: *mut drm_dp_mst_port;
let __obj_size: u64 = size_mul(core::mem::size_of::<drm_dp_mst_port>() as u64, 1u64);
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
__v0 = __v1 as *mut drm_dp_mst_port;
port = __v0;
if port.is_null() {
return core::ptr::null_mut::<drm_dp_mst_port>();
}
kref_init(unsafe { core::ptr::addr_of_mut!((*port).topology_kref) });
kref_init(unsafe { core::ptr::addr_of_mut!((*port).malloc_kref) });
unsafe {
(*port).parent = mstb;
}
unsafe {
(*port).port_num = port_number;
}
unsafe {
(*port).mgr = mgr;
}
unsafe {
(*port).aux.name = c"DPMST".as_ptr() as *const u8;
}
unsafe {
(*port).aux.dev = (*dev).dev;
}
unsafe {
(*port).aux.is_remote = true;
}
// initialize the MST downstream port's AUX crc work queue
unsafe {
(*port).aux.drm_dev = dev;
}
unsafe { drm_dp_remote_aux_init(core::ptr::addr_of_mut!((*port).aux)) };
// Make sure the memory allocation for our parent branch stays
// around until our own memory allocation is released
drm_dp_mst_get_mstb_malloc(mstb);
port
}

fn drm_dp_mst_handle_link_address_port(mstb: *mut drm_dp_mst_branch, dev: *mut drm_device, port_msg: *mut drm_dp_link_addr_reply_port) -> i32 {
let mut changed: bool = unsafe { core::mem::zeroed() };
let mut send_link_addr: bool = unsafe { core::mem::zeroed() };
let mut created: bool = unsafe { core::mem::zeroed() };
let mut new_mcs: bool = unsafe { core::mem::zeroed() };
let mut new_pdt: u8 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut port: core::mem::MaybeUninit<*mut drm_dp_mst_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut drm_dp_mst_port = core::ptr::addr_of_mut!(port) as *mut *mut drm_dp_mst_port;
let mut mgr: *mut drm_dp_mst_topology_mgr = unsafe { core::mem::zeroed() };
unsafe {
mgr = (*mstb).mgr;
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
new_pdt = 0;
new_mcs = false;
created = false;
send_link_addr = false;
changed = false;
*__slate_slot_9 = drm_dp_get_port(mstb, (*port_msg).port_number);
if (*__slate_slot_9).is_null() {
*__slate_slot_9 = drm_dp_mst_add_port(dev, mgr, mstb, (*port_msg).port_number);
if (*__slate_slot_9).is_null() {
return 12i32.wrapping_neg();
} else {
created = true;
changed = true;
}
} else {
if !(*(*__slate_slot_9)).input && (*port_msg).input_port && !unsafe { (*(*__slate_slot_9)).connector }.is_null() {
// Since port->connector can't be changed here, we create a
// new port if input_port changes from 0 to 1
drm_dp_mst_topology_unlink_port(mgr, *__slate_slot_9);
drm_dp_mst_topology_put_port(*__slate_slot_9);
*__slate_slot_9 = drm_dp_mst_add_port(dev, mgr, mstb, (*port_msg).port_number);
if (*__slate_slot_9).is_null() {
return 12i32.wrapping_neg();
} else {
changed = true;
created = true;
}
} else {
if (*(*__slate_slot_9)).input && !(*port_msg).input_port {
changed = true;
} else {
if !unsafe { (*(*__slate_slot_9)).connector }.is_null() {
// We're updating a port that's exposed to userspace, so do it
// under lock
drm_modeset_lock(core::ptr::addr_of_mut!((*mgr).base.lock), core::ptr::null_mut::<drm_modeset_acquire_ctx>());
changed = ((*(*__slate_slot_9)).ddps as i32) != ((*port_msg).ddps as i32) || (*(*__slate_slot_9)).ddps && (((*(*__slate_slot_9)).ldps as i32) != ((*port_msg).legacy_device_plug_status as i32) || ((*(*__slate_slot_9)).dpcd_rev as i32) != ((*port_msg).dpcd_revision as i32) || ((*(*__slate_slot_9)).mcs as i32) != ((*port_msg).mcs as i32) || ((*(*__slate_slot_9)).pdt as i32) != ((*port_msg).peer_device_type as i32) || ((*(*__slate_slot_9)).num_sdp_stream_sinks as i32) != ((*port_msg).num_sdp_stream_sinks as i32));
}
}
}
}
(*(*__slate_slot_9)).input = (*port_msg).input_port;
if !(*(*__slate_slot_9)).input {
new_pdt = (*port_msg).peer_device_type;
}
new_mcs = (*port_msg).mcs;
(*(*__slate_slot_9)).ddps = (*port_msg).ddps;
(*(*__slate_slot_9)).ldps = (*port_msg).legacy_device_plug_status;
(*(*__slate_slot_9)).dpcd_rev = (*port_msg).dpcd_revision;
(*(*__slate_slot_9)).num_sdp_streams = (*port_msg).num_sdp_streams;
(*(*__slate_slot_9)).num_sdp_stream_sinks = (*port_msg).num_sdp_stream_sinks;
// manage mstb port lists with mgr lock - take a reference
// for this list
if created {
mutex_lock(core::ptr::addr_of_mut!((*mgr).lock));
drm_dp_mst_topology_get_port(*__slate_slot_9);
list_add(core::ptr::addr_of_mut!((*(*__slate_slot_9)).next), core::ptr::addr_of_mut!((*mstb).ports));
let __v1: i32 = (*mstb).num_ports;
(*mstb).num_ports = __v1.wrapping_add(1);
mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock));
}
// Reprobe PBN caps on both hotplug, and when re-probing the link
// for our parent mstb
if (*(*__slate_slot_9)).ddps && !(*(*__slate_slot_9)).input {
*__slate_slot_8 = drm_dp_send_enum_path_resources(mgr, mstb, *__slate_slot_9);
if *__slate_slot_8 == 1 {
changed = true;
}
} else {
(*(*__slate_slot_9)).full_pbn = 0;
}
'__join_0: {
*__slate_slot_8 = drm_dp_port_set_pdt(*__slate_slot_9, new_pdt, new_mcs);
if *__slate_slot_8 == 1 {
send_link_addr = true;
} else {
if *__slate_slot_8 < 0 {
while false {
}
_dev_err(__drm_to_dev(dev as *const drm_device) as *const device, c"[drm] *ERROR* Failed to change PDT on port %p: %d\n".as_ptr() as *const u8, *__slate_slot_9, *__slate_slot_8);
drm_dp_mst_topology_unlink_port(mgr, *__slate_slot_9);
if !unsafe { (*(*__slate_slot_9)).connector }.is_null() {
drm_modeset_unlock(core::ptr::addr_of_mut!((*mgr).base.lock));
break '__join_0;
} else {
break '__join_0;
}
}
}
// If this port wasn't just created, then we're reprobing because
// we're coming out of suspend. In this case, always resend the link
// address if there's an MSTB on this port
if !created && ((*(*__slate_slot_9)).pdt as i32) == 2 && (*(*__slate_slot_9)).mcs {
send_link_addr = true;
}
if !unsafe { (*(*__slate_slot_9)).connector }.is_null() {
drm_modeset_unlock(core::ptr::addr_of_mut!((*mgr).base.lock));
} else {
if !(*(*__slate_slot_9)).input {
drm_dp_mst_port_add_connector(mstb, *__slate_slot_9);
}
}
if send_link_addr && !unsafe { (*(*__slate_slot_9)).mstb }.is_null() {
*__slate_slot_8 = drm_dp_send_link_address(mgr, (*(*__slate_slot_9)).mstb);
if *__slate_slot_8 == 1 {
changed = true;
} else {
if *__slate_slot_8 < 0 {
break '__join_0;
}
}
// MSTB below us changed
}
// put reference to this port
drm_dp_mst_topology_put_port(*__slate_slot_9);
return changed as i32;
}
drm_dp_mst_topology_put_port(*__slate_slot_9);
return *__slate_slot_8;
}
}

fn drm_dp_mst_handle_conn_stat(mstb: *mut drm_dp_mst_branch, conn_stat: *mut drm_dp_connection_status_notify) -> i32 {
let mut create_connector: bool = unsafe { core::mem::zeroed() };
let mut dowork: bool = unsafe { core::mem::zeroed() };
let mut new_mcs: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut bool = core::ptr::addr_of_mut!(new_mcs) as *mut bool;
let mut new_pdt: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u8 = core::ptr::addr_of_mut!(new_pdt) as *mut u8;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut old_ddps: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(old_ddps) as *mut i32;
let mut port: core::mem::MaybeUninit<*mut drm_dp_mst_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut drm_dp_mst_port = core::ptr::addr_of_mut!(port) as *mut *mut drm_dp_mst_port;
let mut mgr: *mut drm_dp_mst_topology_mgr = unsafe { core::mem::zeroed() };
unsafe {
mgr = (*mstb).mgr;
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
dowork = false;
create_connector = false;
*__slate_slot_6 = drm_dp_get_port(mstb, (*conn_stat).port_number);
if (*__slate_slot_6).is_null() {
return 0;
} else {
'__join_0: {
if !unsafe { (*(*__slate_slot_6)).connector }.is_null() {
if !(*(*__slate_slot_6)).input && (*conn_stat).input_port {
// We can't remove a connector from an already exposed
// port, so just throw the port out and make sure we
// reprobe the link address of it's parent MSTB
drm_dp_mst_topology_unlink_port(mgr, *__slate_slot_6);
(*mstb).link_address_sent = false;
dowork = true;
break '__join_0;
} else {
// Locking is only needed if the port's exposed to userspace
drm_modeset_lock(core::ptr::addr_of_mut!((*mgr).base.lock), core::ptr::null_mut::<drm_modeset_acquire_ctx>());
}
} else {
if (*(*__slate_slot_6)).input && !(*conn_stat).input_port {
create_connector = true;
// Reprobe link address so we get num_sdp_streams
(*mstb).link_address_sent = false;
dowork = true;
}
}
*__slate_slot_5 = (*(*__slate_slot_6)).ddps as i32;
(*(*__slate_slot_6)).input = (*conn_stat).input_port;
(*(*__slate_slot_6)).ldps = (*conn_stat).legacy_device_plug_status;
(*(*__slate_slot_6)).ddps = (*conn_stat).displayport_device_plug_status;
if *__slate_slot_5 != ((*(*__slate_slot_6)).ddps as i32) {
if (*(*__slate_slot_6)).ddps && !(*(*__slate_slot_6)).input {
drm_dp_send_enum_path_resources(mgr, mstb, *__slate_slot_6);
} else {
(*(*__slate_slot_6)).full_pbn = 0;
}
}
*__slate_slot_3 = (if (*(*__slate_slot_6)).input { 0i32 } else { (*conn_stat).peer_device_type as i32 }) as u8;
*__slate_slot_2 = (*conn_stat).message_capability_status;
*__slate_slot_4 = drm_dp_port_set_pdt(*__slate_slot_6, *__slate_slot_3, *__slate_slot_2);
if *__slate_slot_4 == 1 {
dowork = true;
} else {
if *__slate_slot_4 < 0 {
while false {
}
_dev_err(__drm_to_dev((*mgr).dev as *const drm_device) as *const device, c"[drm] *ERROR* Failed to change PDT for port %p: %d\n".as_ptr() as *const u8, *__slate_slot_6, *__slate_slot_4);
dowork = false;
}
}
if !unsafe { (*(*__slate_slot_6)).connector }.is_null() {
drm_modeset_unlock(core::ptr::addr_of_mut!((*mgr).base.lock));
} else {
if create_connector {
drm_dp_mst_port_add_connector(mstb, *__slate_slot_6);
}
}
}
drm_dp_mst_topology_put_port(*__slate_slot_6);
return dowork as i32;
}
}
}

fn drm_dp_get_mst_branch_device(mgr: *mut drm_dp_mst_topology_mgr, lct: u8, rad: *mut u8) -> *mut drm_dp_mst_branch {
let mut port_num: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut port: core::mem::MaybeUninit<*mut drm_dp_mst_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut drm_dp_mst_port = core::ptr::addr_of_mut!(port) as *mut *mut drm_dp_mst_port;
let mut mstb: core::mem::MaybeUninit<*mut drm_dp_mst_branch> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut drm_dp_mst_branch = core::ptr::addr_of_mut!(mstb) as *mut *mut drm_dp_mst_branch;
unsafe {
'__join_0: {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// find the port by iterating down
mutex_lock(core::ptr::addr_of_mut!((*mgr).lock));
*__slate_slot_6 = (*mgr).mst_primary;
if !(*__slate_slot_6).is_null() {
*__slate_slot_4 = 1;
'__join_8: {
while *__slate_slot_4 < (lct as i32) {
port_num = drm_dp_mst_get_ufp_num_at_lct_from_rad((*__slate_slot_4).wrapping_add(1) as u8, rad as *const u8) as i32;
*__slate_slot_5 = list_first_entry!(core::ptr::addr_of_mut!((*(*__slate_slot_6)).ports), next);
'__join_4: {
loop {
if list_entry_is_head!(*__slate_slot_5, core::ptr::addr_of_mut!((*(*__slate_slot_6)).ports), next) == 0 {
if ((*(*__slate_slot_5)).port_num as i32) == port_num {
break;
} else {
*__slate_slot_5 = list_next_entry!(*__slate_slot_5, next);
}
} else {
break '__join_4;
}
}
*__slate_slot_6 = (*(*__slate_slot_5)).mstb;
if (*__slate_slot_6).is_null() {
break '__join_8;
}
}
let __v1: i32 = *__slate_slot_4;
*__slate_slot_4 = __v1.wrapping_add(1);
}
*__slate_slot_3 = drm_dp_mst_topology_try_get_mstb(*__slate_slot_6);
if *__slate_slot_3 == 0 {
*__slate_slot_6 = core::ptr::null_mut::<drm_dp_mst_branch>();
break '__join_0;
} else {
break '__join_0;
}
}
while false {
}
_dev_err(__drm_to_dev((*mgr).dev as *const drm_device) as *const device, c"[drm] *ERROR* failed to lookup MSTB with lct %d, rad %02x\n".as_ptr() as *const u8, lct as i32, *rad.wrapping_offset(0 as isize) as i32);
}
}
mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock));
return *__slate_slot_6;
}
}

fn get_mst_branch_device_by_guid_helper(mstb: *mut drm_dp_mst_branch, guid: *const __SlateRecord_uuid_h_17) -> *mut drm_dp_mst_branch {
let mut port: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
if mstb.is_null() {
return core::ptr::null_mut::<drm_dp_mst_branch>();
}
if guid_equal((unsafe { core::ptr::addr_of_mut!((*mstb).guid) }) as *const __SlateRecord_uuid_h_17, guid) {
return mstb;
}
port = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next);
while list_entry_is_head!(port, unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next) == 0 {
let found_mstb: *mut drm_dp_mst_branch = get_mst_branch_device_by_guid_helper(unsafe { (*port).mstb }, guid);
if !found_mstb.is_null() {
return found_mstb;
}
port = list_next_entry!(port, next);
}
core::ptr::null_mut::<drm_dp_mst_branch>()
}

fn drm_dp_get_mst_branch_device_by_guid(mgr: *mut drm_dp_mst_topology_mgr, guid: *const __SlateRecord_uuid_h_17) -> *mut drm_dp_mst_branch {
let mut mstb: *mut drm_dp_mst_branch = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
// find the port by iterating down
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).lock)) };
mstb = get_mst_branch_device_by_guid_helper(unsafe { (*mgr).mst_primary }, guid);
if !mstb.is_null() {
ret = drm_dp_mst_topology_try_get_mstb(mstb);
if ret == 0 {
mstb = core::ptr::null_mut::<drm_dp_mst_branch>();
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock)) };
mstb
}

fn drm_dp_check_and_send_link_address(mgr: *mut drm_dp_mst_topology_mgr, mstb: *mut drm_dp_mst_branch) -> i32 {
let mut port: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
let mut changed: bool = false;
if !unsafe { (*mstb).link_address_sent } {
ret = drm_dp_send_link_address(mgr, mstb);
if ret == 1 {
changed = true;
} else {
if ret < 0 {
return ret;
}
}
}
port = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next);
while list_entry_is_head!(port, unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next) == 0 {
if !((unsafe { (*port).input }) || !unsafe { (*port).ddps } || unsafe { (*port).mstb }.is_null()) {
ret = drm_dp_check_and_send_link_address(mgr, unsafe { (*port).mstb });
if ret == 1 {
changed = true;
} else {
if ret < 0 {
return ret;
}
}
}
port = list_next_entry!(port, next);
}
changed as i32
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_link_probe_work")]
extern "C" fn drm_dp_mst_link_probe_work(work: *mut work_struct) {
let mgr: *mut drm_dp_mst_topology_mgr = container_of__16!(work, drm_dp_mst_topology_mgr);
let dev: *mut drm_device = unsafe { (*mgr).dev };
let mut mstb: *mut drm_dp_mst_branch = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
let mut clear_payload_id_table: bool = false;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).probe_lock)) };
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).lock)) };
clear_payload_id_table = ((unsafe { (*mgr).__slate_bits_0.__get_payload_id_table_cleared() }) as i32) == 0;
unsafe {
let __t0 = true;
(*mgr).__slate_bits_0.__set_payload_id_table_cleared(__t0);
}
mstb = unsafe { (*mgr).mst_primary };
if !mstb.is_null() {
ret = drm_dp_mst_topology_try_get_mstb(mstb);
if ret == 0 {
mstb = core::ptr::null_mut::<drm_dp_mst_branch>();
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock)) };
if mstb.is_null() {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).probe_lock)) };
return;
}
// Certain branch devices seem to incorrectly report an available_pbn
// of 0 on downstream sinks, even after clearing the
// DP_PAYLOAD_ALLOCATE_* registers in
// drm_dp_mst_topology_mgr_set_mst(). Namely, the CableMatters USB-C
// 2x DP hub. Sending a CLEAR_PAYLOAD_ID_TABLE message seems to make
// things work again.
if clear_payload_id_table {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Clearing payload ID table\n".as_ptr() as *const u8) };
drm_dp_send_clear_payload_id_table(mgr, mstb);
}
ret = drm_dp_check_and_send_link_address(mgr, mstb);
drm_dp_mst_topology_put_mstb(mstb);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).probe_lock)) };
if ret > 0 {
unsafe { drm_kms_helper_hotplug_event(dev) };
}
}

fn drm_dp_mst_queue_probe_work(mgr: *mut drm_dp_mst_topology_mgr) {
queue_work(unsafe { system_long_wq }, unsafe { core::ptr::addr_of_mut!((*mgr).work) });
}

fn drm_dp_validate_guid(mgr: *mut drm_dp_mst_topology_mgr, guid: *mut __SlateRecord_uuid_h_17) -> bool {
if !guid_is_null(guid as *const __SlateRecord_uuid_h_17) {
return true;
}
unsafe { guid_gen(guid) };
false
}

fn build_dpcd_read(msg: *mut drm_dp_sideband_msg_tx, port_num: u8, offset: u32, num_bytes: u8) {
let mut req: drm_dp_sideband_msg_req_body = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(req) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 120usize);
}
req.req_type = 32;
unsafe {
req.u.dpcd_read.port_number = port_num;
}
unsafe {
req.u.dpcd_read.dpcd_address = offset;
}
unsafe {
req.u.dpcd_read.num_bytes = num_bytes;
}
drm_dp_encode_sideband_req(core::ptr::addr_of_mut!(req) as *const drm_dp_sideband_msg_req_body, msg);
}

fn drm_dp_send_sideband_msg(mgr: *mut drm_dp_mst_topology_mgr, up: bool, msg: *mut u8, len: i32) -> i32 {
let mut __UNIQUE_ID_z__706: i32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__705: i32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__704: i32 = unsafe { core::mem::zeroed() };
let mut retries: i32 = unsafe { core::mem::zeroed() };
let mut offset: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(offset) as *mut i32;
let mut total: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(total) as *mut i32;
let mut tosend: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i32 = core::ptr::addr_of_mut!(tosend) as *mut i32;
let mut regbase: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
regbase = if up { 4608 } else { 4096 };
let __slate_auto_bytes_1 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
retries = 0;
'__loop_10: loop {
*__slate_slot_12 = len;
*__slate_slot_11 = 0;
loop {
__UNIQUE_ID_x__704 = (*mgr).max_dpcd_transaction_bytes;
__UNIQUE_ID_y__705 = 16;
__UNIQUE_ID_z__706 = *__slate_slot_12;
loop {
if __types_ok3!(__UNIQUE_ID_x__704, __UNIQUE_ID_y__705, __UNIQUE_ID_z__706) == 0 {
__compiletime_assert_707();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
*__slate_slot_13 = if __UNIQUE_ID_x__704 < if __UNIQUE_ID_y__705 < __UNIQUE_ID_z__706 { __UNIQUE_ID_y__705 } else { __UNIQUE_ID_z__706 } { __UNIQUE_ID_x__704 } else { if __UNIQUE_ID_y__705 < __UNIQUE_ID_z__706 { __UNIQUE_ID_y__705 } else { __UNIQUE_ID_z__706 } };
*__slate_slot_15 = drm_dp_dpcd_write_data((*mgr).aux, regbase.wrapping_add(*__slate_slot_11) as u32, msg.wrapping_offset(*__slate_slot_11 as isize) as *mut (), *__slate_slot_13 as u64);
if *__slate_slot_15 == 5i32.wrapping_neg() && retries < 5 {
break;
} else {
if *__slate_slot_15 < 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"failed to dpcd write %d %d\n".as_ptr() as *const u8, *__slate_slot_13, *__slate_slot_15);
return 5i32.wrapping_neg();
} else {
let __v3: i32 = *__slate_slot_11;
*__slate_slot_11 = __v3.wrapping_add(*__slate_slot_13);
let __v1: i32 = *__slate_slot_12;
*__slate_slot_12 = __v1.wrapping_sub(*__slate_slot_13);
if *__slate_slot_12 <= 0 {
break '__loop_10;
}
}
}
}
retries = retries.wrapping_add(1);
}
return 0;
}
}

fn set_hdr_from_dst_qlock(hdr: &mut drm_dp_sideband_msg_hdr, txmsg: *mut drm_dp_sideband_msg_tx) -> i32 {
let mstb: *mut drm_dp_mst_branch = unsafe { (*txmsg).dst };
let req_type: u8 = (((unsafe { *((*txmsg).msg.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) & 127) as u8;
if (req_type as i32) == 2 || (req_type as i32) == 19 || (req_type as i32) == 20 {
unsafe {
(*(hdr as *mut drm_dp_sideband_msg_hdr)).broadcast = true;
}
} else {
unsafe {
(*(hdr as *mut drm_dp_sideband_msg_hdr)).broadcast = false;
}
}
unsafe {
(*(hdr as *mut drm_dp_sideband_msg_hdr)).path_msg = (*txmsg).path_msg;
}
if unsafe { (*(hdr as *mut drm_dp_sideband_msg_hdr)).broadcast } {
unsafe {
(*(hdr as *mut drm_dp_sideband_msg_hdr)).lct = 1;
}
unsafe {
(*(hdr as *mut drm_dp_sideband_msg_hdr)).lcr = 6;
}
} else {
unsafe {
(*(hdr as *mut drm_dp_sideband_msg_hdr)).lct = (*mstb).lct;
}
unsafe {
(*(hdr as *mut drm_dp_sideband_msg_hdr)).lcr = ((*mstb).lct as i32).wrapping_sub(1) as u8;
}
}
unsafe { memcpy(((*(hdr as *mut drm_dp_sideband_msg_hdr)).rad.as_mut_ptr() as *mut u8) as *mut (), ((*mstb).rad.as_mut_ptr() as *mut u8) as *const (), (((*(hdr as *mut drm_dp_sideband_msg_hdr)).lct as i32) / 2) as u64) };
0
}

// process a single block of the next message in the sideband queue
fn process_single_tx_qlock(mgr: *mut drm_dp_mst_topology_mgr, txmsg: *mut drm_dp_sideband_msg_tx, up: bool) -> i32 {
let mut chunk: __SlateAlign16<[u8; 48]> = __SlateAlign16([0u8; 48]);
let mut hdr: drm_dp_sideband_msg_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(hdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut len: i32 = 0;
let mut space: i32 = 0;
let mut idx: i32 = 0;
let mut tosend: i32 = 0;
let mut ret: i32 = 0;
if (unsafe { (*txmsg).state }) == 2 {
return 0;
}
unsafe { memset(core::ptr::addr_of_mut!(hdr) as *mut (), 0i32, core::mem::size_of::<drm_dp_sideband_msg_hdr>() as u64) };
if (unsafe { (*txmsg).state }) == 0 {
unsafe {
(*txmsg).state = 1;
}
}
// make hdr from dst mst
ret = set_hdr_from_dst_qlock(unsafe { &mut (*core::ptr::addr_of_mut!(hdr)) }, txmsg);
if ret < 0 {
return ret;
}
// amount left to send in this message
len = ((unsafe { (*txmsg).cur_len }) as i32).wrapping_sub((unsafe { (*txmsg).cur_offset }) as i32);
// 48 - sideband msg size - 1 byte for data CRC, x header bytes
space = 47i32.wrapping_sub(drm_dp_calc_sb_hdr_size(unsafe { &(*core::ptr::addr_of_mut!(hdr)) }) as i32);
let mut __v0: i32;
let __UNIQUE_ID_x__708: i32 = len;
let __UNIQUE_ID_y__709: i32 = space;
loop {
if __types_ok!(__UNIQUE_ID_x__708, __UNIQUE_ID_y__709) == 0 {
unsafe { __compiletime_assert_710() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__708 < __UNIQUE_ID_y__709 { __UNIQUE_ID_x__708 } else { __UNIQUE_ID_y__709 };
tosend = __v0;
if len == ((unsafe { (*txmsg).cur_len }) as i32) {
hdr.somt = true;
}
if space >= len {
hdr.eomt = true;
}
hdr.msg_len = tosend.wrapping_add(1) as u8;
drm_dp_encode_sideband_msg_hdr(unsafe { &mut (*core::ptr::addr_of_mut!(hdr)) }, chunk.0.as_mut_ptr() as *mut u8, unsafe { &mut (*core::ptr::addr_of_mut!(idx)) });
unsafe { memcpy((chunk.0.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize) as *mut (), ((*txmsg).msg.as_mut_ptr() as *mut u8).wrapping_offset(((*txmsg).cur_offset as i32) as isize) as *const (), tosend as u64) };
// add crc at end
drm_dp_crc_sideband_chunk_req((chunk.0.as_mut_ptr() as *mut u8).wrapping_offset(idx as isize), tosend as u8);
idx = idx.wrapping_add(tosend.wrapping_add(1));
ret = drm_dp_send_sideband_msg(mgr, up, chunk.0.as_mut_ptr() as *mut u8, idx);
if ret != 0 {
if drm_debug_enabled!(8i32) {
let mut p: drm_printer = drm_dbg_printer(unsafe { (*mgr).dev }, 8u32, c"[dp_mst]".as_ptr() as *const u8);
unsafe { drm_printf(core::ptr::addr_of_mut!(p), c"sideband msg failed to send\n".as_ptr() as *const u8) };
drm_dp_mst_dump_sideband_msg_tx(core::ptr::addr_of_mut!(p), txmsg as *const drm_dp_sideband_msg_tx);
}
return ret;
}
unsafe {
(*txmsg).cur_offset = ((*txmsg).cur_offset as i32).wrapping_add(tosend) as u8;
}
if ((unsafe { (*txmsg).cur_offset }) as i32) == ((unsafe { (*txmsg).cur_len }) as i32) {
unsafe {
(*txmsg).state = 2;
}
return 1;
}
0
}

fn process_single_down_tx_qlock(mgr: *mut drm_dp_mst_topology_mgr) {
let mut txmsg: *mut drm_dp_sideband_msg_tx = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!unsafe { mutex_is_locked(core::ptr::addr_of_mut!((*mgr).qlock)) })) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("711: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 711b - ., 3; .popsection\n# {0}", const 711 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_process_single_down_tx_qlock_1, sym __slate_asm_string_process_single_down_tx_qlock_2, const 2860 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("712: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 712b - ., 4; .popsection\n# {0}", const 712 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
// construct a chunk from the first msg in the tx_msg queue
if list_empty((unsafe { core::ptr::addr_of_mut!((*mgr).tx_msg_downq) }) as *const list_head) != 0 {
return;
}
txmsg = list_first_entry__13!(unsafe { core::ptr::addr_of_mut!((*mgr).tx_msg_downq) }, drm_dp_sideband_msg_tx, next);
ret = process_single_tx_qlock(mgr, txmsg, false);
if ret < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"failed to send msg in q %d\n".as_ptr() as *const u8, ret) };
list_del(unsafe { core::ptr::addr_of_mut!((*txmsg).next) });
unsafe {
(*txmsg).state = 4;
}
wake_up_all!(unsafe { core::ptr::addr_of_mut!((*mgr).tx_waitq) });
}
}

fn drm_dp_queue_down_tx(mgr: *mut drm_dp_mst_topology_mgr, txmsg: *mut drm_dp_sideband_msg_tx) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).qlock)) };
list_add_tail(unsafe { core::ptr::addr_of_mut!((*txmsg).next) }, unsafe { core::ptr::addr_of_mut!((*mgr).tx_msg_downq) });
if drm_debug_enabled!(8i32) {
let mut p: drm_printer = drm_dbg_printer(unsafe { (*mgr).dev }, 8u32, c"[dp_mst]".as_ptr() as *const u8);
drm_dp_mst_dump_sideband_msg_tx(core::ptr::addr_of_mut!(p), txmsg as *const drm_dp_sideband_msg_tx);
}
if list_is_singular((unsafe { core::ptr::addr_of_mut!((*mgr).tx_msg_downq) }) as *const list_head) != 0 {
process_single_down_tx_qlock(mgr);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).qlock)) };
}

fn drm_dp_dump_link_address(mgr: *const drm_dp_mst_topology_mgr, reply: *mut drm_dp_link_address_ack_reply) {
let mut i: i32 = 0;
while i < ((unsafe { (*reply).nports }) as i32) {
let port_reply: *mut drm_dp_link_addr_reply_port = unsafe { (*reply).ports.as_mut_ptr() as *mut drm_dp_link_addr_reply_port }.wrapping_offset(i as isize);
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"port %d: input %d, pdt: %d, pn: %d, dpcd_rev: %02x, mcs: %d, ddps: %d, ldps %d, sdp %d/%d\n".as_ptr() as *const u8, i, (*port_reply).input_port as i32, (*port_reply).peer_device_type as i32, (*port_reply).port_number as i32, (*port_reply).dpcd_revision as i32, (*port_reply).mcs as i32, (*port_reply).ddps as i32, (*port_reply).legacy_device_plug_status as i32, (*port_reply).num_sdp_streams as i32, (*port_reply).num_sdp_stream_sinks as i32) };
i = i.wrapping_add(1);
}
}

fn drm_dp_send_link_address(mgr: *mut drm_dp_mst_topology_mgr, mstb: *mut drm_dp_mst_branch) -> i32 {
let mut __slate_compound_24: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_26: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut buf: core::mem::MaybeUninit<__SlateAlign16<[u8; 64]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut [u8; 64] = core::ptr::addr_of_mut!(buf) as *mut [u8; 64];
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut changed: bool = unsafe { core::mem::zeroed() };
let mut port_mask: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut tmp: core::mem::MaybeUninit<*mut drm_dp_mst_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut drm_dp_mst_port = core::ptr::addr_of_mut!(tmp) as *mut *mut drm_dp_mst_port;
let mut port: core::mem::MaybeUninit<*mut drm_dp_mst_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut drm_dp_mst_port = core::ptr::addr_of_mut!(port) as *mut *mut drm_dp_mst_port;
let mut reply: core::mem::MaybeUninit<*mut drm_dp_link_address_ack_reply> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut drm_dp_link_address_ack_reply = core::ptr::addr_of_mut!(reply) as *mut *mut drm_dp_link_address_ack_reply;
let mut txmsg: core::mem::MaybeUninit<*mut drm_dp_sideband_msg_tx> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut drm_dp_sideband_msg_tx = core::ptr::addr_of_mut!(txmsg) as *mut *mut drm_dp_sideband_msg_tx;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
port_mask = 0;
changed = false;
__obj_size = size_mul(core::mem::size_of::<drm_dp_sideband_msg_tx>() as u64, 1u64);
let __slate_auto_bytes_7 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_7 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_8 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_24 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_24)
});
alloc_tag_restore!();
} else {
*__slate_slot_8 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_26 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_26)
});
}
*__slate_slot_20 = *__slate_slot_8 as *mut drm_dp_sideband_msg_tx;
if (*__slate_slot_20).is_null() {
return 12i32.wrapping_neg();
} else {
'__join_2: {
(*(*__slate_slot_20)).dst = mstb;
build_link_address(*__slate_slot_20);
(*mstb).link_address_sent = true;
drm_dp_queue_down_tx(mgr, *__slate_slot_20);
// FIXME: Actually do some real error handling here
*__slate_slot_15 = drm_dp_mst_wait_tx_reply(mstb, *__slate_slot_20);
if *__slate_slot_15 < 0 {
while false {
}
_dev_err(__drm_to_dev((*mgr).dev as *const drm_device) as *const device, c"[drm] *ERROR* Sending link address failed with %d\n".as_ptr() as *const u8, *__slate_slot_15);
} else {
if ((*(*__slate_slot_20)).reply.reply_type as i32) == 1 {
while false {
}
_dev_err(__drm_to_dev((*mgr).dev as *const drm_device) as *const device, c"[drm] *ERROR* link address NAK received\n".as_ptr() as *const u8);
*__slate_slot_15 = 5i32.wrapping_neg();
} else {
*__slate_slot_19 = core::ptr::addr_of_mut!((*(*__slate_slot_20)).reply.u.link_addr);
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"link address reply: %d\n".as_ptr() as *const u8, (*(*__slate_slot_19)).nports as i32);
drm_dp_dump_link_address(mgr as *const drm_dp_mst_topology_mgr, *__slate_slot_19);
*__slate_slot_15 = drm_dp_check_mstb_guid(mstb, core::ptr::addr_of_mut!((*(*__slate_slot_19)).guid));
if *__slate_slot_15 != 0 {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
drm_dp_mst_rad_to_str(((*mstb).rad.as_mut_ptr() as *mut u8) as *const u8, (*mstb).lct, (*__slate_slot_6).as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 64]>() as u64);
while false {
}
_dev_err(__drm_to_dev((*mgr).dev as *const drm_device) as *const device, c"[drm] *ERROR* GUID check on %s failed: %d\n".as_ptr() as *const u8, (*__slate_slot_6).as_mut_ptr() as *mut u8, *__slate_slot_15);
} else {
*__slate_slot_16 = 0;
while *__slate_slot_16 < ((*(*__slate_slot_19)).nports as i32) {
let __v5: i32 = port_mask;
port_mask = ((__v5 as u64) | BIT__3!(unsafe { (*((*(*__slate_slot_19)).ports.as_mut_ptr() as *mut drm_dp_link_addr_reply_port).wrapping_offset(*__slate_slot_16 as isize)).port_number })) as i32;
*__slate_slot_15 = drm_dp_mst_handle_link_address_port(mstb, (*mgr).dev, ((*(*__slate_slot_19)).ports.as_mut_ptr() as *mut drm_dp_link_addr_reply_port).wrapping_offset(*__slate_slot_16 as isize));
if *__slate_slot_15 == 1 {
changed = true;
} else {
if *__slate_slot_15 < 0 {
break '__join_2;
}
}
let __v3: i32 = *__slate_slot_16;
*__slate_slot_16 = __v3.wrapping_add(1);
}
// Prune any ports that are currently a part of mstb in our in-memory
// topology, but were not seen in this link address. Usually this
// means that they were removed while the topology was out of sync,
// e.g. during suspend/resume
mutex_lock(core::ptr::addr_of_mut!((*mgr).lock));
*__slate_slot_18 = list_first_entry!(core::ptr::addr_of_mut!((*mstb).ports), next);
let __v1: *mut drm_dp_mst_port = list_next_entry!(*__slate_slot_18, next);
*__slate_slot_17 = __v1;
while list_entry_is_head!(*__slate_slot_18, core::ptr::addr_of_mut!((*mstb).ports), next) == 0 {
if (port_mask as u64) & BIT__3!(unsafe { (*(*__slate_slot_18)).port_num }) == 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"port %d was not in link address, removing\n".as_ptr() as *const u8, (*(*__slate_slot_18)).port_num as i32);
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_18)).next));
drm_dp_mst_topology_put_port(*__slate_slot_18);
changed = true;
}
*__slate_slot_18 = *__slate_slot_17;
let __v0: *mut drm_dp_mst_port = list_next_entry!(*__slate_slot_17, next);
*__slate_slot_17 = __v0;
}
mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock));
}
}
}
}
if *__slate_slot_15 < 0 {
(*mstb).link_address_sent = false;
}
kfree(*__slate_slot_20 as *const ());
return if *__slate_slot_15 < 0 { *__slate_slot_15 } else { changed as i32 };
}
}
}

fn drm_dp_send_clear_payload_id_table(mgr: *mut drm_dp_mst_topology_mgr, mstb: *mut drm_dp_mst_branch) {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
let mut __v0: *mut drm_dp_sideband_msg_tx;
let __obj_size: u64 = size_mul(core::mem::size_of::<drm_dp_sideband_msg_tx>() as u64, 1u64);
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
__v0 = __v1 as *mut drm_dp_sideband_msg_tx;
let txmsg: *mut drm_dp_sideband_msg_tx = __v0;
if txmsg.is_null() {
return;
}
unsafe {
(*txmsg).dst = mstb;
}
build_clear_payload_id_table(txmsg);
drm_dp_queue_down_tx(mgr, txmsg);
ret = drm_dp_mst_wait_tx_reply(mstb, txmsg);
if ret > 0 && ((unsafe { (*txmsg).reply.reply_type }) as i32) == 1 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"clear payload table id nak received\n".as_ptr() as *const u8) };
}
unsafe { kfree(txmsg as *const ()) };
}

fn drm_dp_send_enum_path_resources(mgr: *mut drm_dp_mst_topology_mgr, mstb: *mut drm_dp_mst_branch, port: *mut drm_dp_mst_port) -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut txmsg: *mut drm_dp_sideband_msg_tx = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
let mut __v0: *mut drm_dp_sideband_msg_tx;
let __obj_size: u64 = size_mul(core::mem::size_of::<drm_dp_sideband_msg_tx>() as u64, 1u64);
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
__v0 = __v1 as *mut drm_dp_sideband_msg_tx;
txmsg = __v0;
if txmsg.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*txmsg).dst = mstb;
}
build_enum_path_resources(txmsg, (unsafe { (*port).port_num }) as i32);
drm_dp_queue_down_tx(mgr, txmsg);
ret = drm_dp_mst_wait_tx_reply(mstb, txmsg);
if ret > 0 {
ret = 0;
let path_res: *mut drm_dp_enum_path_resources_ack_reply = unsafe { core::ptr::addr_of_mut!((*txmsg).reply.u.path_resources) };
if ((unsafe { (*txmsg).reply.reply_type }) as i32) == 1 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"enum path resources nak received\n".as_ptr() as *const u8) };
} else {
if ((unsafe { (*port).port_num }) as i32) != ((unsafe { (*path_res).port_number }) as i32) {
unsafe { __drm_err(c"got incorrect port in response\n".as_ptr() as *const u8) };
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"enum path resources %d: %d %d\n".as_ptr() as *const u8, (*path_res).port_number as i32, (*path_res).full_payload_bw_number as i32, (*path_res).avail_payload_bw_number as i32) };
// If something changed, make sure we send a
// hotplug
if ((unsafe { (*port).full_pbn }) as i32) != ((unsafe { (*path_res).full_payload_bw_number }) as i32) || ((unsafe { (*port).fec_capable }) as i32) != ((unsafe { (*path_res).fec_capable }) as i32) {
ret = 1;
}
unsafe {
(*port).full_pbn = (*path_res).full_payload_bw_number;
}
unsafe {
(*port).fec_capable = (*path_res).fec_capable;
}
}
}
unsafe { kfree(txmsg as *const ()) };
ret
}

fn drm_dp_get_last_connected_port_to_mstb(mstb: *mut drm_dp_mst_branch) -> *mut drm_dp_mst_port {
if unsafe { (*mstb).port_parent }.is_null() {
return core::ptr::null_mut::<drm_dp_mst_port>();
}
if (unsafe { (*(*mstb).port_parent).mstb }) != mstb {
return unsafe { (*mstb).port_parent };
}
drm_dp_get_last_connected_port_to_mstb(unsafe { (*(*mstb).port_parent).parent })
}

/// Searches upwards in the topology starting from mstb to try to find the
/// closest available parent of mstb that's still connected to the rest of the
/// topology. This can be used in order to perform operations like releasing
/// payloads, where the branch device which owned the payload may no longer be
/// around and thus would require that the payload on the last living relative
/// be freed instead.
fn drm_dp_get_last_connected_port_and_mstb(mgr: *mut drm_dp_mst_topology_mgr, mut mstb: *mut drm_dp_mst_branch, port_num: &mut i32) -> *mut drm_dp_mst_branch {
let mut found_port: core::mem::MaybeUninit<*mut drm_dp_mst_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut drm_dp_mst_port = core::ptr::addr_of_mut!(found_port) as *mut *mut drm_dp_mst_port;
let mut rmstb: *mut drm_dp_mst_branch = unsafe { core::mem::zeroed() };
unsafe {
rmstb = core::ptr::null_mut::<drm_dp_mst_branch>();
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
mutex_lock(core::ptr::addr_of_mut!((*mgr).lock));
if !unsafe { (*mgr).mst_primary }.is_null() {
loop {
*__slate_slot_0 = drm_dp_get_last_connected_port_to_mstb(mstb);
if (*__slate_slot_0).is_null() {
break;
} else {
if drm_dp_mst_topology_try_get_mstb((*(*__slate_slot_0)).parent) != 0 {
rmstb = (*(*__slate_slot_0)).parent;
*(port_num as *mut i32) = (*(*__slate_slot_0)).port_num as i32;
} else {
// Search again, starting from this parent
mstb = (*(*__slate_slot_0)).parent;
}
if !rmstb.is_null() {
break;
}
}
}
}
mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock));
return rmstb;
}
}

fn drm_dp_payload_send_msg(mgr: *mut drm_dp_mst_topology_mgr, port: *mut drm_dp_mst_port, id: i32, pbn: i32) -> i32 {
let mut __slate_compound_14: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_16: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut sinks: core::mem::MaybeUninit<__SlateAlign16<[u8; 16]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut [u8; 16] = core::ptr::addr_of_mut!(sinks) as *mut [u8; 16];
let mut port_num: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(port_num) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut mstb: core::mem::MaybeUninit<*mut drm_dp_mst_branch> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut drm_dp_mst_branch = core::ptr::addr_of_mut!(mstb) as *mut *mut drm_dp_mst_branch;
let mut txmsg: core::mem::MaybeUninit<*mut drm_dp_sideband_msg_tx> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut drm_dp_sideband_msg_tx = core::ptr::addr_of_mut!(txmsg) as *mut *mut drm_dp_sideband_msg_tx;
unsafe {
{
let __slate_auto_bytes_2 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 16usize);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_10 = (*port).port_num as i32;
*__slate_slot_12 = drm_dp_mst_topology_get_mstb_validated(mgr, (*port).parent);
if (*__slate_slot_12).is_null() {
*__slate_slot_12 = drm_dp_get_last_connected_port_and_mstb(mgr, (*port).parent, &mut (*core::ptr::addr_of_mut!(*__slate_slot_10)));
if (*__slate_slot_12).is_null() {
return 22i32.wrapping_neg();
}
}
}
__obj_size = size_mul(core::mem::size_of::<drm_dp_sideband_msg_tx>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_2 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_3 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_14 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_14)
});
alloc_tag_restore!();
} else {
*__slate_slot_3 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_16 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_16)
});
}
*__slate_slot_13 = *__slate_slot_3 as *mut drm_dp_sideband_msg_tx;
if (*__slate_slot_13).is_null() {
*__slate_slot_11 = 12i32.wrapping_neg();
} else {
*__slate_slot_8 = 0;
while *__slate_slot_8 < ((*port).num_sdp_streams as i32) {
(*__slate_slot_9)[*__slate_slot_8 as usize] = *__slate_slot_8 as u8;
let __v1: i32 = *__slate_slot_8;
*__slate_slot_8 = __v1.wrapping_add(1);
}
(*(*__slate_slot_13)).dst = *__slate_slot_12;
build_allocate_payload(*__slate_slot_13, *__slate_slot_10, id as u8, pbn as u16, (*port).num_sdp_streams, (*__slate_slot_9).as_mut_ptr() as *mut u8);
drm_dp_queue_down_tx(mgr, *__slate_slot_13);
// FIXME: there is a small chance that between getting the last
// connected mstb and sending the payload message, the last connected
// mstb could also be removed from the topology. In the future, this
// needs to be fixed by restarting the
// drm_dp_get_last_connected_port_and_mstb() search in the event of a
// timeout if the topology is still connected to the system.
*__slate_slot_11 = drm_dp_mst_wait_tx_reply(*__slate_slot_12, *__slate_slot_13);
if *__slate_slot_11 > 0 {
if ((*(*__slate_slot_13)).reply.reply_type as i32) == 1 {
*__slate_slot_11 = 22i32.wrapping_neg();
} else {
*__slate_slot_11 = 0;
}
}
kfree(*__slate_slot_13 as *const ());
}
drm_dp_mst_topology_put_mstb(*__slate_slot_12);
return *__slate_slot_11;
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_send_power_updown_phy")]
extern "C" fn drm_dp_send_power_updown_phy(mgr: *mut drm_dp_mst_topology_mgr, mut port: *mut drm_dp_mst_port, power_up: bool) -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
port = drm_dp_mst_topology_get_port_validated(mgr, port);
if port.is_null() {
return 22i32.wrapping_neg();
}
let mut __v0: *mut drm_dp_sideband_msg_tx;
let __obj_size: u64 = size_mul(core::mem::size_of::<drm_dp_sideband_msg_tx>() as u64, 1u64);
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
__v0 = __v1 as *mut drm_dp_sideband_msg_tx;
let txmsg: *mut drm_dp_sideband_msg_tx = __v0;
if txmsg.is_null() {
drm_dp_mst_topology_put_port(port);
return 12i32.wrapping_neg();
}
unsafe {
(*txmsg).dst = (*port).parent;
}
build_power_updown_phy(txmsg, (unsafe { (*port).port_num }) as i32, power_up);
drm_dp_queue_down_tx(mgr, txmsg);
ret = drm_dp_mst_wait_tx_reply(unsafe { (*port).parent }, txmsg);
if ret > 0 {
if ((unsafe { (*txmsg).reply.reply_type }) as i32) == 1 {
ret = 22i32.wrapping_neg();
} else {
ret = 0;
}
}
unsafe { kfree(txmsg as *const ()) };
drm_dp_mst_topology_put_port(port);
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_send_power_updown_phy_713: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_port, bool) -> i32>, *mut ()>(Some(drm_dp_send_power_updown_phy)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_send_query_stream_enc_status")]
extern "C" fn drm_dp_send_query_stream_enc_status(mgr: *mut drm_dp_mst_topology_mgr, mut port: *mut drm_dp_mst_port, status__0: *mut drm_dp_query_stream_enc_status_ack_reply) -> i32 {
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut nonce: core::mem::MaybeUninit<[u8; 7]> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut [u8; 7] = core::ptr::addr_of_mut!(nonce) as *mut [u8; 7];
let mut txmsg: core::mem::MaybeUninit<*mut drm_dp_sideband_msg_tx> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut drm_dp_sideband_msg_tx = core::ptr::addr_of_mut!(txmsg) as *mut *mut drm_dp_sideband_msg_tx;
let mut payload: core::mem::MaybeUninit<*mut drm_dp_mst_atomic_payload> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut drm_dp_mst_atomic_payload = core::ptr::addr_of_mut!(payload) as *mut *mut drm_dp_mst_atomic_payload;
let mut state: core::mem::MaybeUninit<*mut drm_dp_mst_topology_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut drm_dp_mst_topology_state = core::ptr::addr_of_mut!(state) as *mut *mut drm_dp_mst_topology_state;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 7usize);
let __slate_auto_bytes_5 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
__obj_size = size_mul(core::mem::size_of::<drm_dp_sideband_msg_tx>() as u64, 1u64);
let __slate_auto_bytes_6 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
});
}
*__slate_slot_8 = *__slate_slot_1 as *mut drm_dp_sideband_msg_tx;
if (*__slate_slot_8).is_null() {
return 12i32.wrapping_neg();
} else {
port = drm_dp_mst_topology_get_port_validated(mgr, port);
if port.is_null() {
*__slate_slot_6 = 22i32.wrapping_neg();
} else {
get_random_bytes((*__slate_slot_7).as_mut_ptr() as *mut (), core::mem::size_of::<[u8; 7]>() as u64);
drm_modeset_lock(core::ptr::addr_of_mut!((*mgr).base.lock), core::ptr::null_mut::<drm_modeset_acquire_ctx>());
*__slate_slot_10 = to_drm_dp_mst_topology_state((*mgr).base.state);
*__slate_slot_9 = drm_atomic_get_mst_payload_state(*__slate_slot_10, port);
// "Source device targets the QUERY_STREAM_ENCRYPTION_STATUS message
//  transaction at the MST Branch device directly connected to the
//  Source"
(*(*__slate_slot_8)).dst = (*mgr).mst_primary;
build_query_stream_enc_status(*__slate_slot_8, (*(*__slate_slot_9)).vcpi, (*__slate_slot_7).as_mut_ptr() as *mut u8);
drm_dp_queue_down_tx(mgr, *__slate_slot_8);
*__slate_slot_6 = drm_dp_mst_wait_tx_reply((*mgr).mst_primary, *__slate_slot_8);
if *__slate_slot_6 >= 0 {
if ((*(*__slate_slot_8)).reply.reply_type as i32) == 1 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"query encryption status nak received\n".as_ptr() as *const u8);
*__slate_slot_6 = 6i32.wrapping_neg();
} else {
*__slate_slot_6 = 0;
memcpy(status__0 as *mut (), core::ptr::addr_of_mut!((*(*__slate_slot_8)).reply.u.enc_status) as *const (), core::mem::size_of::<drm_dp_query_stream_enc_status_ack_reply>() as u64);
}
}
drm_modeset_unlock(core::ptr::addr_of_mut!((*mgr).base.lock));
drm_dp_mst_topology_put_port(port);
}
kfree(*__slate_slot_8 as *const ());
return *__slate_slot_6;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_send_query_stream_enc_status_714: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_port, *mut drm_dp_query_stream_enc_status_ack_reply) -> i32>, *mut ()>(Some(drm_dp_send_query_stream_enc_status)) };

fn drm_dp_create_payload_at_dfp(mgr: *mut drm_dp_mst_topology_mgr, payload: *mut drm_dp_mst_atomic_payload) -> i32 {
unsafe { drm_dp_dpcd_write_payload((*mgr).aux, (*payload).vcpi as i32, (*payload).vc_start_slot as u8, (*payload).time_slots as u8) }
}

fn drm_dp_create_payload_to_remote(mgr: *mut drm_dp_mst_topology_mgr, payload: *mut drm_dp_mst_atomic_payload) -> i32 {
let mut ret: i32 = 0;
let port: *mut drm_dp_mst_port = drm_dp_mst_topology_get_port_validated(mgr, unsafe { (*payload).port });
if port.is_null() {
return 5i32.wrapping_neg();
}
ret = drm_dp_payload_send_msg(mgr, port, (unsafe { (*payload).vcpi }) as i32, unsafe { (*payload).pbn });
drm_dp_mst_topology_put_port(port);
ret
}

fn drm_dp_destroy_payload_at_remote_and_dfp(mgr: *mut drm_dp_mst_topology_mgr, mst_state: *mut drm_dp_mst_topology_state, payload: *mut drm_dp_mst_atomic_payload) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"\n".as_ptr() as *const u8) };
// it's okay for these to fail
if ((unsafe { (*payload).payload_allocation_status }) as u32) == 3 {
drm_dp_payload_send_msg(mgr, unsafe { (*payload).port }, (unsafe { (*payload).vcpi }) as i32, 0i32);
unsafe {
(*payload).payload_allocation_status = 2;
}
}
if ((unsafe { (*payload).payload_allocation_status }) as u32) == 2 {
unsafe { drm_dp_dpcd_write_payload((*mgr).aux, (*payload).vcpi as i32, (*payload).vc_start_slot as u8, 0u8) };
}
}

/// drm_dp_add_payload_part1() - Execute payload update part 1
/// @mgr: Manager to use.
/// @mst_state: The MST atomic state
/// @payload: The payload to write
///
/// Determines the starting time slot for the given payload, and programs the VCPI for this payload
/// into the DPCD of DPRX. After calling this, the driver should generate ACT and payload packets.
///
/// Returns: 0 on success, error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_add_payload_part1")]
extern "C" fn drm_dp_add_payload_part1(mgr: *mut drm_dp_mst_topology_mgr, mst_state: *mut drm_dp_mst_topology_state, payload: *mut drm_dp_mst_atomic_payload) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut port: core::mem::MaybeUninit<*mut drm_dp_mst_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut drm_dp_mst_port = core::ptr::addr_of_mut!(port) as *mut *mut drm_dp_mst_port;
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// Update mst mgr info
if ((*mgr).payload_count as i32) == 0 {
(*mgr).next_start_slot = (*mst_state).start_slot;
}
(*payload).vc_start_slot = (*mgr).next_start_slot as i8;
(*mgr).payload_count = ((*mgr).payload_count as i32).wrapping_add(1) as u8;
(*mgr).next_start_slot = ((*mgr).next_start_slot as i32).wrapping_add((*payload).time_slots) as u8;
(*payload).payload_allocation_status = 1;
// Allocate payload to immediate downstream facing port
*__slate_slot_7 = drm_dp_mst_topology_get_port_validated(mgr, (*payload).port);
if (*__slate_slot_7).is_null() {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"VCPI %d for port %p not in topology, not creating a payload to remote\n".as_ptr() as *const u8, (*payload).vcpi as i32, (*payload).port);
return 5i32.wrapping_neg();
} else {
*__slate_slot_6 = drm_dp_create_payload_at_dfp(mgr, payload);
if *__slate_slot_6 < 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"Failed to create MST payload for port %p: %d\n".as_ptr() as *const u8, (*payload).port, *__slate_slot_6);
} else {
(*payload).payload_allocation_status = 2;
}
drm_dp_mst_topology_put_port(*__slate_slot_7);
return *__slate_slot_6;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_add_payload_part1_715: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_topology_state, *mut drm_dp_mst_atomic_payload) -> i32>, *mut ()>(Some(drm_dp_add_payload_part1)) };

/// drm_dp_remove_payload_part1() - Remove an MST payload along the virtual channel
/// @mgr: Manager to use.
/// @mst_state: The MST atomic state
/// @payload: The payload to remove
///
/// Removes a payload along the virtual channel if it was successfully allocated.
/// After calling this, the driver should set HW to generate ACT and then switch to new
/// payload allocation state.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_remove_payload_part1")]
extern "C" fn drm_dp_remove_payload_part1(mgr: *mut drm_dp_mst_topology_mgr, mst_state: *mut drm_dp_mst_topology_state, payload: *mut drm_dp_mst_atomic_payload) {
// Remove remote payload allocation
let mut send_remove: bool = false;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).lock)) };
send_remove = drm_dp_mst_port_downstream_of_branch(unsafe { (*payload).port }, unsafe { (*mgr).mst_primary });
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock)) };
if send_remove {
drm_dp_destroy_payload_at_remote_and_dfp(mgr, mst_state, payload);
} else {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"Payload for VCPI %d not in topology, not sending remove\n".as_ptr() as *const u8, (*payload).vcpi as i32) };
}
unsafe {
(*payload).payload_allocation_status = 1;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_remove_payload_part1_716: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_topology_state, *mut drm_dp_mst_atomic_payload)>, *mut ()>(Some(drm_dp_remove_payload_part1)) };

/// drm_dp_remove_payload_part2() - Remove an MST payload locally
/// @mgr: Manager to use.
/// @mst_state: The MST atomic state
/// @old_payload: The payload with its old state
/// @new_payload: The payload with its latest state
///
/// Updates the starting time slots of all other payloads which would have been shifted towards
/// the start of the payload ID table as a result of removing a payload. Driver should call this
/// function whenever it removes a payload in its HW. It's independent to the result of payload
/// allocation/deallocation at branch devices along the virtual channel.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_remove_payload_part2")]
extern "C" fn drm_dp_remove_payload_part2(mgr: *mut drm_dp_mst_topology_mgr, mst_state: *mut drm_dp_mst_topology_state, old_payload: *const drm_dp_mst_atomic_payload, new_payload: *mut drm_dp_mst_atomic_payload) {
let mut pos: *mut drm_dp_mst_atomic_payload = unsafe { core::mem::zeroed() };
// Remove local payload allocation
pos = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*mst_state).payloads) }, next);
while list_entry_is_head!(pos, unsafe { core::ptr::addr_of_mut!((*mst_state).payloads) }, next) == 0 {
if pos != new_payload && ((unsafe { (*pos).vc_start_slot }) as i32) > ((unsafe { (*new_payload).vc_start_slot }) as i32) {
unsafe {
(*pos).vc_start_slot = ((*pos).vc_start_slot as i32).wrapping_sub((*old_payload).time_slots) as i8;
}
}
pos = list_next_entry__2!(pos, next);
}
unsafe {
(*new_payload).vc_start_slot = 1i32.wrapping_neg() as i8;
}
unsafe {
(*mgr).payload_count = ((*mgr).payload_count as i32).wrapping_sub(1) as u8;
}
unsafe {
(*mgr).next_start_slot = ((*mgr).next_start_slot as i32).wrapping_sub((*old_payload).time_slots) as u8;
}
if ((unsafe { (*new_payload).__slate_bits_0.__get_delete() }) as i32) != 0 {
drm_dp_mst_put_port_malloc(unsafe { (*new_payload).port });
}
unsafe {
(*new_payload).payload_allocation_status = 0;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_remove_payload_part2_717: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_topology_state, *const drm_dp_mst_atomic_payload, *mut drm_dp_mst_atomic_payload)>, *mut ()>(Some(drm_dp_remove_payload_part2)) };

/// drm_dp_add_payload_part2() - Execute payload update part 2
/// @mgr: Manager to use.
/// @payload: The payload to update
///
/// If @payload was successfully assigned a starting time slot by drm_dp_add_payload_part1(), this
/// function will send the sideband messages to finish allocating this payload.
///
/// Returns: 0 on success, negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_add_payload_part2")]
extern "C" fn drm_dp_add_payload_part2(mgr: *mut drm_dp_mst_topology_mgr, payload: *mut drm_dp_mst_atomic_payload) -> i32 {
let mut ret: i32 = 0;
// Skip failed payloads
if ((unsafe { (*payload).payload_allocation_status }) as u32) != 2 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"Part 1 of payload creation for %s failed, skipping part 2\n".as_ptr() as *const u8, (*(*(*payload).port).connector).name) };
return 5i32.wrapping_neg();
}
// Allocate payload to remote end
ret = drm_dp_create_payload_to_remote(mgr, payload);
if ret < 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*mgr).dev as *const drm_device) as *const device, c"[drm] *ERROR* Step 2 of creating MST payload for %p failed: %d\n".as_ptr() as *const u8, (*payload).port, ret) };
} else {
unsafe {
(*payload).payload_allocation_status = 3;
}
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_add_payload_part2_718: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_atomic_payload) -> i32>, *mut ()>(Some(drm_dp_add_payload_part2)) };

fn drm_dp_send_dpcd_read(mgr: *mut drm_dp_mst_topology_mgr, port: *mut drm_dp_mst_port, offset: i32, size: i32, bytes: *mut u8) -> i32 {
let mut __slate_compound_12: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_14: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut mstb: core::mem::MaybeUninit<*mut drm_dp_mst_branch> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut drm_dp_mst_branch = core::ptr::addr_of_mut!(mstb) as *mut *mut drm_dp_mst_branch;
let mut txmsg: core::mem::MaybeUninit<*mut drm_dp_sideband_msg_tx> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut drm_dp_sideband_msg_tx = core::ptr::addr_of_mut!(txmsg) as *mut *mut drm_dp_sideband_msg_tx;
let mut ret: i32 = unsafe { core::mem::zeroed() };
unsafe {
ret = 0;
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
*__slate_slot_6 = drm_dp_mst_topology_get_mstb_validated(mgr, (*port).parent);
if (*__slate_slot_6).is_null() {
return 22i32.wrapping_neg();
} else {
__obj_size = size_mul(core::mem::size_of::<drm_dp_sideband_msg_tx>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_12 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_12)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_14 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_14)
});
}
*__slate_slot_7 = *__slate_slot_1 as *mut drm_dp_sideband_msg_tx;
if (*__slate_slot_7).is_null() {
ret = 12i32.wrapping_neg();
} else {
build_dpcd_read(*__slate_slot_7, (*port).port_num, offset as u32, size as u8);
(*(*__slate_slot_7)).dst = (*port).parent;
drm_dp_queue_down_tx(mgr, *__slate_slot_7);
ret = drm_dp_mst_wait_tx_reply(*__slate_slot_6, *__slate_slot_7);
if ret >= 0 {
if ((*(*__slate_slot_7)).reply.reply_type as i32) == 1 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"mstb %p port %d: DPCD read on addr 0x%x for %d bytes NAKed\n".as_ptr() as *const u8, *__slate_slot_6, (*port).port_num as i32, offset, size);
ret = 5i32.wrapping_neg();
} else {
if ((*(*__slate_slot_7)).reply.u.remote_dpcd_read_ack.num_bytes as i32) != size {
ret = 71i32.wrapping_neg();
} else {
ret = min_t__2!(u64, unsafe { (*(*__slate_slot_7)).reply.u.remote_dpcd_read_ack.num_bytes }, size) as i32;
memcpy(bytes as *mut (), ((*(*__slate_slot_7)).reply.u.remote_dpcd_read_ack.bytes.as_mut_ptr() as *mut u8) as *const (), ret as u64);
}
}
}
kfree(*__slate_slot_7 as *const ());
}
drm_dp_mst_topology_put_mstb(*__slate_slot_6);
return ret;
}
}
}

fn drm_dp_send_dpcd_write(mgr: *mut drm_dp_mst_topology_mgr, port: *mut drm_dp_mst_port, offset: i32, size: i32, bytes: *mut u8) -> i32 {
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut mstb: core::mem::MaybeUninit<*mut drm_dp_mst_branch> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut drm_dp_mst_branch = core::ptr::addr_of_mut!(mstb) as *mut *mut drm_dp_mst_branch;
let mut txmsg: core::mem::MaybeUninit<*mut drm_dp_sideband_msg_tx> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut drm_dp_sideband_msg_tx = core::ptr::addr_of_mut!(txmsg) as *mut *mut drm_dp_sideband_msg_tx;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
*__slate_slot_6 = drm_dp_mst_topology_get_mstb_validated(mgr, (*port).parent);
if (*__slate_slot_6).is_null() {
return 22i32.wrapping_neg();
} else {
__obj_size = size_mul(core::mem::size_of::<drm_dp_sideband_msg_tx>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
});
}
*__slate_slot_7 = *__slate_slot_1 as *mut drm_dp_sideband_msg_tx;
if (*__slate_slot_7).is_null() {
*__slate_slot_8 = 12i32.wrapping_neg();
} else {
build_dpcd_write(*__slate_slot_7, (*port).port_num, offset as u32, size as u8, bytes);
(*(*__slate_slot_7)).dst = *__slate_slot_6;
drm_dp_queue_down_tx(mgr, *__slate_slot_7);
*__slate_slot_8 = drm_dp_mst_wait_tx_reply(*__slate_slot_6, *__slate_slot_7);
if *__slate_slot_8 > 0 {
if ((*(*__slate_slot_7)).reply.reply_type as i32) == 1 {
*__slate_slot_8 = 5i32.wrapping_neg();
} else {
*__slate_slot_8 = size;
}
}
kfree(*__slate_slot_7 as *const ());
}
drm_dp_mst_topology_put_mstb(*__slate_slot_6);
return *__slate_slot_8;
}
}
}

fn drm_dp_encode_up_ack_reply(msg: *mut drm_dp_sideband_msg_tx, req_type: u8) -> i32 {
let mut reply: drm_dp_sideband_msg_reply_body = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(reply) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 420usize);
}
reply.reply_type = 0;
reply.req_type = req_type;
drm_dp_encode_sideband_reply(unsafe { &(*core::ptr::addr_of_mut!(reply)) }, msg);
0
}

fn drm_dp_send_up_ack_reply(mgr: *mut drm_dp_mst_topology_mgr, mstb: *mut drm_dp_mst_branch, req_type: i32, broadcast: bool) -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __v0: *mut drm_dp_sideband_msg_tx;
let __obj_size: u64 = size_mul(core::mem::size_of::<drm_dp_sideband_msg_tx>() as u64, 1u64);
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
__v0 = __v1 as *mut drm_dp_sideband_msg_tx;
let txmsg: *mut drm_dp_sideband_msg_tx = __v0;
if txmsg.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*txmsg).dst = mstb;
}
drm_dp_encode_up_ack_reply(txmsg, req_type as u8);
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).qlock)) };
// construct a chunk from the first msg in the tx_msg queue
process_single_tx_qlock(mgr, txmsg, true);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).qlock)) };
unsafe { kfree(txmsg as *const ()) };
0
}

/// drm_dp_get_vc_payload_bw - get the VC payload BW for an MTP link
/// @link_rate: link rate in 10kbits/s units
/// @link_lane_count: lane count
///
/// Calculate the total bandwidth of a MultiStream Transport link. The returned
/// value is in units of PBNs/(timeslots/1 MTP). This value can be used to
/// convert the number of PBNs required for a given stream to the number of
/// timeslots this stream requires in each MTP.
///
/// Returns the BW / timeslot value in 20.12 fixed point format.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_get_vc_payload_bw")]
extern "C" fn drm_dp_get_vc_payload_bw(link_rate: i32, link_lane_count: i32) -> dfixed {
let ch_coding_efficiency: i32 = unsafe { drm_dp_bw_channel_coding_efficiency(drm_dp_is_uhbr_rate(link_rate)) };
let mut ret: dfixed = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ret) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
// See DP v2.0 2.6.4.2, 2.7.6.3 VCPayload_Bandwidth_for_OneTimeSlotPer_MTP_Allocation
unsafe {
ret.full = DIV_ROUND_DOWN_ULL!(mul_u32_u32(link_rate.wrapping_mul(link_lane_count) as u32, ch_coding_efficiency as u32), 10546875u64) as u32;
}
return ret;
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_get_vc_payload_bw_721: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, i32) -> dfixed>, *mut ()>(Some(drm_dp_get_vc_payload_bw)) };

/// drm_dp_read_mst_cap() - Read the sink's MST mode capability
/// @aux: The DP AUX channel to use
/// @dpcd: A cached copy of the DPCD capabilities for this sink
///
/// Returns: enum drm_dp_mst_mode to indicate MST mode capability
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_read_mst_cap")]
extern "C" fn drm_dp_read_mst_cap(aux: *mut drm_dp_aux, dpcd: *const u8) -> u32 {
let mut mstm_cap: u8 = 0;
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) < 18 {
return 0;
}
if drm_dp_dpcd_read_byte(aux, 33u32, core::ptr::addr_of_mut!(mstm_cap)) < 0 {
return 0;
}
if (mstm_cap as i32) & 1i32 != 0 {
return 1;
}
if (mstm_cap as i32) & 2i32 != 0 {
return 2;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_read_mst_cap_722: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *const u8) -> u32>, *mut ()>(Some(drm_dp_read_mst_cap)) };

/// drm_dp_mst_topology_mgr_set_mst() - Set the MST state for a topology manager
/// @mgr: manager to set state for
/// @mst_state: true to enable MST on this connector - false to disable.
///
/// This is called by the driver when it detects an MST capable device plugged
/// into a DP MST capable port, or when a DP MST capable device is unplugged.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_topology_mgr_set_mst")]
extern "C" fn drm_dp_mst_topology_mgr_set_mst(mgr: *mut drm_dp_mst_topology_mgr, mst_state: bool) -> i32 {
let mut mstb: *mut drm_dp_mst_branch = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
unsafe {
ret = 0;
mstb = core::ptr::null_mut::<drm_dp_mst_branch>();
mutex_lock(core::ptr::addr_of_mut!((*mgr).lock));
if (mst_state as i32) != ((*mgr).__slate_bits_0.__get_mst_state() as i32) {
(*mgr).__slate_bits_0.__set_mst_state(mst_state);
// set the device into MST mode
if mst_state {
let __ret_warn_on: i32 = !unsafe { (*mgr).mst_primary }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("723: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 723b - ., 3; .popsection\n# {0}", const 723 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_dp_mst_topology_mgr_set_mst_5, sym __slate_asm_string_drm_dp_mst_topology_mgr_set_mst_6, const 3658 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
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
// get dpcd info
ret = drm_dp_read_dpcd_caps((*mgr).aux, (*mgr).dpcd.as_mut_ptr() as *mut u8);
if ret < 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"%s: failed to read DPCD, ret %d\n".as_ptr() as *const u8, (*(*mgr).aux).name, ret);
} else {
// add initial branch device at LCT 1
mstb = drm_dp_add_mst_branch_device(1u8, core::ptr::null_mut::<u8>());
if mstb.is_null() {
ret = 12i32.wrapping_neg();
} else {
(*mstb).mgr = mgr;
// give this the main reference
(*mgr).mst_primary = mstb;
drm_dp_mst_topology_get_mstb((*mgr).mst_primary);
ret = drm_dp_dpcd_write_byte((*mgr).aux, 273u32, 7u8);
if ret >= 0 {
// Write reset payload
drm_dp_dpcd_clear_payload((*mgr).aux);
drm_dp_mst_queue_probe_work(mgr);
ret = 0;
}
}
}
} else {
// disable MST on the device
mstb = (*mgr).mst_primary;
(*mgr).mst_primary = core::ptr::null_mut::<drm_dp_mst_branch>();
// this can fail if the device is gone
drm_dp_dpcd_write_byte((*mgr).aux, 273u32, 0u8);
ret = 0;
let __t0 = false;
(*mgr).__slate_bits_0.__set_payload_id_table_cleared(__t0);
let __t1 = true;
(*mgr).__slate_bits_0.__set_reset_rx_state(__t1);
}
}
mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock));
if !mstb.is_null() {
drm_dp_mst_topology_put_mstb(mstb);
}
return ret;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_topology_mgr_set_mst_725: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr, bool) -> i32>, *mut ()>(Some(drm_dp_mst_topology_mgr_set_mst)) };

fn drm_dp_mst_topology_mgr_invalidate_mstb(mstb: *mut drm_dp_mst_branch) {
let mut port: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
// The link address will need to be re-sent on resume
unsafe {
(*mstb).link_address_sent = false;
}
port = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next);
while list_entry_is_head!(port, unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next) == 0 {
if !unsafe { (*port).mstb }.is_null() {
drm_dp_mst_topology_mgr_invalidate_mstb(unsafe { (*port).mstb });
}
port = list_next_entry!(port, next);
}
}

/// drm_dp_mst_topology_queue_probe - Queue a topology probe
/// @mgr: manager to probe
///
/// Queue a work to probe the MST topology. Driver's should call this only to
/// sync the topology's HW->SW state after the MST link's parameters have
/// changed in a way the state could've become out-of-sync. This is the case
/// for instance when the link rate between the source and first downstream
/// branch device has switched between UHBR and non-UHBR rates. Except of those
/// cases - for instance when a sink gets plugged/unplugged to a port - the SW
/// state will get updated automatically via MST UP message notifications.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_topology_queue_probe")]
extern "C" fn drm_dp_mst_topology_queue_probe(mgr: *mut drm_dp_mst_topology_mgr) {
unsafe {
mutex_lock(core::ptr::addr_of_mut!((*mgr).lock));
if ((*mgr).__slate_bits_0.__get_mst_state() as i32) == 0 || unsafe { (*mgr).mst_primary }.is_null() {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"queue_probe skipped: topology torn down\n".as_ptr() as *const u8);
} else {
drm_dp_mst_topology_mgr_invalidate_mstb((*mgr).mst_primary);
drm_dp_mst_queue_probe_work(mgr);
}
mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock));
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_topology_queue_probe_726: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr)>, *mut ()>(Some(drm_dp_mst_topology_queue_probe)) };

/// drm_dp_mst_topology_mgr_suspend() - suspend the MST manager
/// @mgr: manager to suspend
///
/// This function tells the MST device that we can't handle UP messages
/// anymore. This should stop it from sending any since we are suspended.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_topology_mgr_suspend")]
extern "C" fn drm_dp_mst_topology_mgr_suspend(mgr: *mut drm_dp_mst_topology_mgr) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).lock)) };
drm_dp_dpcd_write_byte(unsafe { (*mgr).aux }, 273u32, 5u8);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock)) };
unsafe { flush_work(core::ptr::addr_of_mut!((*mgr).up_req_work)) };
unsafe { flush_work(core::ptr::addr_of_mut!((*mgr).work)) };
unsafe { flush_work(core::ptr::addr_of_mut!((*mgr).delayed_destroy_work)) };
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).lock)) };
if ((unsafe { (*mgr).__slate_bits_0.__get_mst_state() }) as i32) != 0 && !unsafe { (*mgr).mst_primary }.is_null() {
drm_dp_mst_topology_mgr_invalidate_mstb(unsafe { (*mgr).mst_primary });
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_topology_mgr_suspend_727: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr)>, *mut ()>(Some(drm_dp_mst_topology_mgr_suspend)) };

/// drm_dp_mst_topology_mgr_resume() - resume the MST manager
/// @mgr: manager to resume
/// @sync: whether or not to perform topology reprobing synchronously
///
/// This will fetch DPCD and see if the device is still there,
/// if it is, it will rewrite the MSTM control bits, and return.
///
/// If the device fails this returns -1, and the driver should do
/// a full MST reprobe, in case we were undocked.
///
/// During system resume (where it is assumed that the driver will be calling
/// drm_atomic_helper_resume()) this function should be called beforehand with
/// @sync set to true. In contexts like runtime resume where the driver is not
/// expected to be calling drm_atomic_helper_resume(), this function should be
/// called with @sync set to false in order to avoid deadlocking.
///
/// Returns: -1 if the MST topology was removed while we were suspended, 0
/// otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_topology_mgr_resume")]
extern "C" fn drm_dp_mst_topology_mgr_resume(mgr: *mut drm_dp_mst_topology_mgr, sync: bool) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut guid: core::mem::MaybeUninit<__SlateRecord_uuid_h_17> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_uuid_h_17 = core::ptr::addr_of_mut!(guid) as *mut __SlateRecord_uuid_h_17;
let mut buf: core::mem::MaybeUninit<__SlateAlign16<[u8; 16]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut [u8; 16] = core::ptr::addr_of_mut!(buf) as *mut [u8; 16];
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 16usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
mutex_lock(core::ptr::addr_of_mut!((*mgr).lock));
if !unsafe { (*mgr).mst_primary }.is_null() {
if drm_dp_read_dpcd_caps((*mgr).aux, (*mgr).dpcd.as_mut_ptr() as *mut u8) < 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"dpcd read failed - undocked during suspend?\n".as_ptr() as *const u8);
} else {
*__slate_slot_0 = drm_dp_dpcd_write_byte((*mgr).aux, 273u32, 7u8);
if *__slate_slot_0 < 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"mst write failed - undocked during suspend?\n".as_ptr() as *const u8);
} else {
// Some hubs forget their guids after they resume
*__slate_slot_0 = drm_dp_dpcd_read_data((*mgr).aux, 48u32, (*__slate_slot_2).as_mut_ptr() as *mut (), core::mem::size_of::<[u8; 16]>() as u64);
if *__slate_slot_0 < 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"dpcd read failed - undocked during suspend?\n".as_ptr() as *const u8);
} else {
import_guid(core::ptr::addr_of_mut!(*__slate_slot_1), (*__slate_slot_2).as_mut_ptr() as *const u8);
*__slate_slot_0 = drm_dp_check_mstb_guid((*mgr).mst_primary, core::ptr::addr_of_mut!(*__slate_slot_1));
if *__slate_slot_0 != 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"check mstb failed - undocked during suspend?\n".as_ptr() as *const u8);
} else {
// For the final step of resuming the topology, we need to bring the
// state of our in-memory topology back into sync with reality. So,
// restart the probing process as if we're probing a new hub
drm_dp_mst_queue_probe_work(mgr);
mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock));
if sync {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"Waiting for link probe work to finish re-syncing topology...\n".as_ptr() as *const u8);
flush_work(core::ptr::addr_of_mut!((*mgr).work));
}
return 0;
}
}
}
}
}
mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock));
return 1i32.wrapping_neg();
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_topology_mgr_resume_728: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr, bool) -> i32>, *mut ()>(Some(drm_dp_mst_topology_mgr_resume)) };

fn reset_msg_rx_state(msg: *mut drm_dp_sideband_msg_rx) {
unsafe { memset(msg as *mut (), 0i32, core::mem::size_of::<drm_dp_sideband_msg_rx>() as u64) };
}

fn drm_dp_get_one_sb_msg(mgr: *mut drm_dp_mst_topology_mgr, up: bool, mstb: *mut *mut drm_dp_mst_branch) -> bool {
let mut len: i32 = 0;
let mut replyblock: __SlateAlign16<[u8; 32]> = __SlateAlign16([0u8; 32]);
let mut replylen: i32 = 0;
let mut curreply: i32 = 0;
let mut ret: i32 = 0;
let mut hdrlen: u8 = 0;
let mut hdr: drm_dp_sideband_msg_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(hdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let msg: *mut drm_dp_sideband_msg_rx = if up { unsafe { core::ptr::addr_of_mut!((*mgr).up_req_recv) } } else { unsafe { core::ptr::addr_of_mut!((*mgr).down_rep_recv) } };
let basereg: i32 = if up { 5632 } else { 5120 };
if !up {
unsafe {
*mstb = core::ptr::null_mut::<drm_dp_mst_branch>();
}
}
let mut __v0: i32;
let __UNIQUE_ID_x__729: i32 = unsafe { (*mgr).max_dpcd_transaction_bytes };
let __UNIQUE_ID_y__730: i32 = 16;
loop {
if __types_ok!(__UNIQUE_ID_x__729, __UNIQUE_ID_y__730) == 0 {
unsafe { __compiletime_assert_731() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__729 < __UNIQUE_ID_y__730 { __UNIQUE_ID_x__729 } else { __UNIQUE_ID_y__730 };
len = __v0;
ret = drm_dp_dpcd_read_data(unsafe { (*mgr).aux }, basereg as u32, replyblock.0.as_mut_ptr() as *mut (), len as u64);
if ret < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"failed to read DPCD down rep %d %d\n".as_ptr() as *const u8, len, ret) };
return false;
}
ret = drm_dp_decode_sideband_msg_hdr(mgr as *const drm_dp_mst_topology_mgr, unsafe { &mut (*core::ptr::addr_of_mut!(hdr)) }, replyblock.0.as_mut_ptr() as *mut u8, len, unsafe { &mut (*core::ptr::addr_of_mut!(hdrlen)) }) as i32;
if ret == 0 {
unsafe { print_hex_dump(c"\x017".as_ptr() as *const u8, c"failed hdr".as_ptr() as *const u8, 0i32, 16i32, 1i32, replyblock.0.as_mut_ptr() as *const (), len as u64, false) };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"ERROR: failed header\n".as_ptr() as *const u8) };
return false;
}
if !up {
// Caller is responsible for giving back this reference
unsafe {
*mstb = drm_dp_get_mst_branch_device(mgr, hdr.lct, hdr.rad.as_mut_ptr() as *mut u8);
}
if unsafe { *mstb }.is_null() {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"Got MST reply from unknown device %d\n".as_ptr() as *const u8, hdr.lct as i32) };
return false;
}
}
if drm_dp_sideband_msg_set_header(msg, core::ptr::addr_of_mut!(hdr), hdrlen) == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"sideband msg set header failed %d\n".as_ptr() as *const u8, replyblock.0[0] as i32) };
return false;
}
let mut __v2: i32;
let __UNIQUE_ID_x__732: u8 = unsafe { (*msg).curchunk_len };
let __UNIQUE_ID_y__733: u8 = len.wrapping_sub(hdrlen as i32) as u8;
loop {
if __types_ok__5!(__UNIQUE_ID_x__732, __UNIQUE_ID_y__733) == 0 {
unsafe { __compiletime_assert_734() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if (__UNIQUE_ID_x__732 as i32) < (__UNIQUE_ID_y__733 as i32) { __UNIQUE_ID_x__732 as i32 } else { __UNIQUE_ID_y__733 as i32 };
replylen = __v2;
ret = drm_dp_sideband_append_payload(msg, (replyblock.0.as_mut_ptr() as *mut u8).wrapping_offset((hdrlen as i32) as isize), replylen as u8) as i32;
if ret == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"sideband msg build failed %d\n".as_ptr() as *const u8, replyblock.0[0] as i32) };
return false;
}
replylen = ((unsafe { (*msg).curchunk_len }) as i32).wrapping_add((unsafe { (*msg).curchunk_hdrlen }) as i32).wrapping_sub(len);
curreply = len;
while replylen > 0 {
let mut __v4: i32;
let __UNIQUE_ID_x__735: i32 = replylen;
let __UNIQUE_ID_y__736: i32 = unsafe { (*mgr).max_dpcd_transaction_bytes };
let __UNIQUE_ID_z__737: i32 = 16;
loop {
if __types_ok3!(__UNIQUE_ID_x__735, __UNIQUE_ID_y__736, __UNIQUE_ID_z__737) == 0 {
unsafe { __compiletime_assert_738() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = if __UNIQUE_ID_x__735 < if __UNIQUE_ID_y__736 < __UNIQUE_ID_z__737 { __UNIQUE_ID_y__736 } else { __UNIQUE_ID_z__737 } { __UNIQUE_ID_x__735 } else { if __UNIQUE_ID_y__736 < __UNIQUE_ID_z__737 { __UNIQUE_ID_y__736 } else { __UNIQUE_ID_z__737 } };
len = __v4;
ret = drm_dp_dpcd_read_data(unsafe { (*mgr).aux }, basereg.wrapping_add(curreply) as u32, replyblock.0.as_mut_ptr() as *mut (), len as u64);
if ret < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"failed to read a chunk (len %d, ret %d)\n".as_ptr() as *const u8, len, ret) };
return false;
}
ret = drm_dp_sideband_append_payload(msg, replyblock.0.as_mut_ptr() as *mut u8, len as u8) as i32;
if ret == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"failed to build sideband msg\n".as_ptr() as *const u8) };
return false;
}
curreply = curreply.wrapping_add(len);
replylen = replylen.wrapping_sub(len);
}
true
}

fn get_msg_request_type(data: u8) -> i32 {
(data as i32) & 127
}

fn verify_rx_request_type(mgr: *mut drm_dp_mst_topology_mgr, txmsg: *const drm_dp_sideband_msg_tx, rxmsg: *const drm_dp_sideband_msg_rx) -> bool {
let hdr: *const drm_dp_sideband_msg_hdr = unsafe { core::ptr::addr_of!((*rxmsg).initial_hdr) };
let mstb: *const drm_dp_mst_branch = (unsafe { (*txmsg).dst }) as *const drm_dp_mst_branch;
let tx_req_type: i32 = get_msg_request_type(unsafe { *((*txmsg).msg.as_ptr() as *const u8).wrapping_offset(0 as isize) });
let rx_req_type: i32 = get_msg_request_type(unsafe { *((*rxmsg).msg.as_ptr() as *const u8).wrapping_offset(0 as isize) });
let mut rad_str: __SlateAlign16<[u8; 64]> = __SlateAlign16([0u8; 64]);
if tx_req_type == rx_req_type {
return true;
}
drm_dp_mst_rad_to_str(unsafe { (*mstb).rad.as_ptr() as *const u8 }, unsafe { (*mstb).lct }, rad_str.0.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 64]>() as u64);
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"Got unexpected MST reply, mstb: %p seqno: %d lct: %d rad: %s rx_req_type: %s (%02x) != tx_req_type: %s (%02x)\n".as_ptr() as *const u8, mstb, (*hdr).seqno as i32, (*mstb).lct as i32, rad_str.0.as_mut_ptr() as *mut u8, drm_dp_mst_req_type_str(rx_req_type as u8), rx_req_type, drm_dp_mst_req_type_str(tx_req_type as u8), tx_req_type) };
false
}

fn drm_dp_mst_handle_down_rep(mgr: *mut drm_dp_mst_topology_mgr) -> i32 {
let mut hdr: core::mem::MaybeUninit<*mut drm_dp_sideband_msg_hdr> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut drm_dp_sideband_msg_hdr = core::ptr::addr_of_mut!(hdr) as *mut *mut drm_dp_sideband_msg_hdr;
let mut __v1: *mut drm_dp_sideband_msg_tx = unsafe { core::mem::zeroed() };
let mut pos__: *mut list_head = unsafe { core::mem::zeroed() };
let mut head__: *mut list_head = unsafe { core::mem::zeroed() };
let mut msg: *mut drm_dp_sideband_msg_rx = unsafe { core::mem::zeroed() };
let mut mstb: *mut drm_dp_mst_branch = unsafe { core::mem::zeroed() };
let mut txmsg: core::mem::MaybeUninit<*mut drm_dp_sideband_msg_tx> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut drm_dp_sideband_msg_tx = core::ptr::addr_of_mut!(txmsg) as *mut *mut drm_dp_sideband_msg_tx;
unsafe {
'__join_2: {
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
mstb = core::ptr::null_mut::<drm_dp_mst_branch>();
msg = core::ptr::addr_of_mut!((*mgr).down_rep_recv);
if drm_dp_get_one_sb_msg(mgr, false, core::ptr::addr_of_mut!(mstb)) {
// Multi-packet message transmission, don't clear the reply
if !(*msg).have_eomt {
break '__join_2;
} else {
// find the message
mutex_lock(core::ptr::addr_of_mut!((*mgr).qlock));
head__ = core::ptr::addr_of_mut!((*mgr).tx_msg_downq);
while false {
}
let __v2: *mut list_head = core::ptr::read_volatile(core::ptr::addr_of_mut!((*head__).next) as *const *mut list_head);
pos__ = __v2;
if pos__ != head__ {
__v1 = list_entry__2!(pos__, drm_dp_sideband_msg_tx);
} else {
__v1 = core::ptr::null_mut::<drm_dp_sideband_msg_tx>();
}
*__slate_slot_8 = __v1;
// Were we actually expecting a response, and from this mstb?
if (*__slate_slot_8).is_null() || (*(*__slate_slot_8)).dst != mstb {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::addr_of_mut!((*msg).initial_hdr);
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"Got MST reply with no msg %p %d %d %02x %02x\n".as_ptr() as *const u8, mstb, (*(*__slate_slot_0)).seqno as i32, (*(*__slate_slot_0)).lct as i32, *((*(*__slate_slot_0)).rad.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32, *((*msg).msg.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32);
mutex_unlock(core::ptr::addr_of_mut!((*mgr).qlock));
} else {
if !verify_rx_request_type(mgr, *__slate_slot_8 as *const drm_dp_sideband_msg_tx, msg as *const drm_dp_sideband_msg_rx) {
mutex_unlock(core::ptr::addr_of_mut!((*mgr).qlock));
} else {
drm_dp_sideband_parse_reply(mgr as *const drm_dp_mst_topology_mgr, msg, core::ptr::addr_of_mut!((*(*__slate_slot_8)).reply));
if ((*(*__slate_slot_8)).reply.reply_type as i32) == 1 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"Got NAK reply: req 0x%02x (%s), reason 0x%02x (%s), nak data 0x%02x\n".as_ptr() as *const u8, (*(*__slate_slot_8)).reply.req_type as i32, drm_dp_mst_req_type_str((*(*__slate_slot_8)).reply.req_type), (*(*__slate_slot_8)).reply.u.nak.reason as i32, drm_dp_mst_nak_reason_str((*(*__slate_slot_8)).reply.u.nak.reason), (*(*__slate_slot_8)).reply.u.nak.nak_data as i32);
}
(*(*__slate_slot_8)).state = 3;
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_8)).next));
mutex_unlock(core::ptr::addr_of_mut!((*mgr).qlock));
wake_up_all!(core::ptr::addr_of_mut!((*mgr).tx_waitq));
}
}
}
}
reset_msg_rx_state(msg);
}
if !mstb.is_null() {
drm_dp_mst_topology_put_mstb(mstb);
}
return 0;
}
}

fn primary_mstb_probing_is_done(mgr: *mut drm_dp_mst_topology_mgr) -> bool {
let mut probing_done: bool = false;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).lock)) };
let __v0: bool = !unsafe { (*mgr).mst_primary }.is_null() && drm_dp_mst_topology_try_get_mstb(unsafe { (*mgr).mst_primary }) != 0;
if __v0 {
probing_done = unsafe { (*(*mgr).mst_primary).link_address_sent };
drm_dp_mst_topology_put_mstb(unsafe { (*mgr).mst_primary });
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock)) };
probing_done
}

fn drm_dp_mst_process_up_req(mgr: *mut drm_dp_mst_topology_mgr, up_req: *mut drm_dp_pending_up_req) -> bool {
let mut mstb: *mut drm_dp_mst_branch = core::ptr::null_mut::<drm_dp_mst_branch>();
let msg: *mut drm_dp_sideband_msg_req_body = unsafe { core::ptr::addr_of_mut!((*up_req).msg) };
let hdr: *mut drm_dp_sideband_msg_hdr = unsafe { core::ptr::addr_of_mut!((*up_req).hdr) };
let mut hotplug: bool = false;
let mut dowork: bool = false;
if unsafe { (*hdr).broadcast } {
let mut guid: *const __SlateRecord_uuid_h_17 = core::ptr::null::<__SlateRecord_uuid_h_17>();
if ((unsafe { (*msg).req_type }) as i32) == 2 {
guid = (unsafe { core::ptr::addr_of_mut!((*msg).u.conn_stat.guid) }) as *const __SlateRecord_uuid_h_17;
} else {
if ((unsafe { (*msg).req_type }) as i32) == 19 {
guid = (unsafe { core::ptr::addr_of_mut!((*msg).u.resource_stat.guid) }) as *const __SlateRecord_uuid_h_17;
}
}
if !guid.is_null() {
mstb = drm_dp_get_mst_branch_device_by_guid(mgr, guid);
}
} else {
mstb = drm_dp_get_mst_branch_device(mgr, unsafe { (*hdr).lct }, unsafe { (*hdr).rad.as_mut_ptr() as *mut u8 });
}
if mstb.is_null() {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"Got MST reply from unknown device %d\n".as_ptr() as *const u8, (*hdr).lct as i32) };
return false;
}
// TODO: Add missing handler for DP_RESOURCE_STATUS_NOTIFY events
if ((unsafe { (*msg).req_type }) as i32) == 2 {
if !primary_mstb_probing_is_done(mgr) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"Got CSN before finish topology probing. Skip it.\n".as_ptr() as *const u8) };
} else {
dowork = drm_dp_mst_handle_conn_stat(mstb, unsafe { core::ptr::addr_of_mut!((*msg).u.conn_stat) }) != 0;
hotplug = true;
}
}
drm_dp_mst_topology_put_mstb(mstb);
if dowork {
queue_work(unsafe { system_long_wq }, unsafe { core::ptr::addr_of_mut!((*mgr).work) });
}
hotplug
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_up_req_work")]
extern "C" fn drm_dp_mst_up_req_work(work: *mut work_struct) {
let mgr: *mut drm_dp_mst_topology_mgr = container_of__20!(work, drm_dp_mst_topology_mgr);
let mut up_req: *mut drm_dp_pending_up_req = unsafe { core::mem::zeroed() };
let mut send_hotplug: bool = false;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).probe_lock)) };
while true {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).up_req_lock)) };
let mut __v1: *mut drm_dp_pending_up_req;
let head__: *mut list_head = unsafe { core::ptr::addr_of_mut!((*mgr).up_req_list) };
let mut __v2: *mut list_head;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_740() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*head__).next) as *const *mut list_head) };
let pos__: *mut list_head = __v2;
let __v4: *mut drm_dp_pending_up_req = if pos__ != head__ {
list_entry__2!(pos__, drm_dp_pending_up_req)
} else {
core::ptr::null_mut::<drm_dp_pending_up_req>()
};
__v1 = __v4;
up_req = __v1;
if !up_req.is_null() {
list_del(unsafe { core::ptr::addr_of_mut!((*up_req).next) });
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).up_req_lock)) };
if up_req.is_null() {
break;
}
let __v7: bool = drm_dp_mst_process_up_req(mgr, up_req);
send_hotplug = (send_hotplug as i32) | (__v7 as i32) != 0;
unsafe { kfree(up_req as *const ()) };
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).probe_lock)) };
if send_hotplug {
unsafe { drm_kms_helper_hotplug_event((*mgr).dev) };
}
}

fn drm_dp_mst_handle_up_req(mgr: *mut drm_dp_mst_topology_mgr) -> i32 {
let mut __slate_compound_12: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_14: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut mst_primary: core::mem::MaybeUninit<*mut drm_dp_mst_branch> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut drm_dp_mst_branch = core::ptr::addr_of_mut!(mst_primary) as *mut *mut drm_dp_mst_branch;
let mut up_req: core::mem::MaybeUninit<*mut drm_dp_pending_up_req> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut drm_dp_pending_up_req = core::ptr::addr_of_mut!(up_req) as *mut *mut drm_dp_pending_up_req;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
ret = 0;
if drm_dp_get_one_sb_msg(mgr, true, core::ptr::null_mut::<*mut drm_dp_mst_branch>()) {
if !(*mgr).up_req_recv.have_eomt {
return 0;
} else {
__obj_size = size_mul(core::mem::size_of::<drm_dp_pending_up_req>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_3 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_4 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_12 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_12)
});
alloc_tag_restore!();
} else {
*__slate_slot_4 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_14 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_14)
});
}
*__slate_slot_11 = *__slate_slot_4 as *mut drm_dp_pending_up_req;
if (*__slate_slot_11).is_null() {
ret = 12i32.wrapping_neg();
} else {
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_11)).next));
drm_dp_sideband_parse_req(mgr as *const drm_dp_mst_topology_mgr, core::ptr::addr_of_mut!((*mgr).up_req_recv), core::ptr::addr_of_mut!((*(*__slate_slot_11)).msg));
if ((*(*__slate_slot_11)).msg.req_type as i32) != 2 && ((*(*__slate_slot_11)).msg.req_type as i32) != 19 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"Received unknown up req type, ignoring: %x\n".as_ptr() as *const u8, (*(*__slate_slot_11)).msg.req_type as i32);
kfree(*__slate_slot_11 as *const ());
} else {
mutex_lock(core::ptr::addr_of_mut!((*mgr).lock));
*__slate_slot_10 = (*mgr).mst_primary;
if (*__slate_slot_10).is_null() {
__v2 = true;
} else {
__v2 = drm_dp_mst_topology_try_get_mstb(*__slate_slot_10) == 0;
}
if __v2 {
mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock));
kfree(*__slate_slot_11 as *const ());
} else {
mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock));
drm_dp_send_up_ack_reply(mgr, *__slate_slot_10, (*(*__slate_slot_11)).msg.req_type as i32, false);
drm_dp_mst_topology_put_mstb(*__slate_slot_10);
if ((*(*__slate_slot_11)).msg.req_type as i32) == 2 {
let conn_stat: *const drm_dp_connection_status_notify = core::ptr::addr_of_mut!((*(*__slate_slot_11)).msg.u.conn_stat) as *const drm_dp_connection_status_notify;
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"Got CSN: pn: %d ldps:%d ddps: %d mcs: %d ip: %d pdt: %d\n".as_ptr() as *const u8, (*conn_stat).port_number as i32, (*conn_stat).legacy_device_plug_status as i32, (*conn_stat).displayport_device_plug_status as i32, (*conn_stat).message_capability_status as i32, (*conn_stat).input_port as i32, (*conn_stat).peer_device_type as i32);
} else {
if ((*(*__slate_slot_11)).msg.req_type as i32) == 19 {
let res_stat: *const drm_dp_resource_status_notify = core::ptr::addr_of_mut!((*(*__slate_slot_11)).msg.u.resource_stat) as *const drm_dp_resource_status_notify;
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"Got RSN: pn: %d avail_pbn %d\n".as_ptr() as *const u8, (*res_stat).port_number as i32, (*res_stat).available_pbn as i32);
}
}
(*(*__slate_slot_11)).hdr = (*mgr).up_req_recv.initial_hdr;
mutex_lock(core::ptr::addr_of_mut!((*mgr).up_req_lock));
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_11)).next), core::ptr::addr_of_mut!((*mgr).up_req_list));
mutex_unlock(core::ptr::addr_of_mut!((*mgr).up_req_lock));
queue_work(system_long_wq, core::ptr::addr_of_mut!((*mgr).up_req_work));
}
}
}
}
}
reset_msg_rx_state(core::ptr::addr_of_mut!((*mgr).up_req_recv));
return ret;
}
}

fn update_msg_rx_state(mgr: *mut drm_dp_mst_topology_mgr) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).lock)) };
if ((unsafe { (*mgr).__slate_bits_0.__get_reset_rx_state() }) as i32) != 0 {
unsafe {
let __t0 = false;
(*mgr).__slate_bits_0.__set_reset_rx_state(__t0);
}
reset_msg_rx_state(unsafe { core::ptr::addr_of_mut!((*mgr).down_rep_recv) });
reset_msg_rx_state(unsafe { core::ptr::addr_of_mut!((*mgr).up_req_recv) });
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock)) };
}

/// drm_dp_mst_hpd_irq_handle_event() - MST hotplug IRQ handle MST event
/// @mgr: manager to notify irq for.
/// @esi: 4 bytes from SINK_COUNT_ESI
/// @ack: 4 bytes used to ack events starting from SINK_COUNT_ESI
/// @handled: whether the hpd interrupt was consumed or not
///
/// This should be called from the driver when it detects a HPD IRQ,
/// along with the value of the DEVICE_SERVICE_IRQ_VECTOR_ESI0. The
/// topology manager will process the sideband messages received
/// as indicated in the DEVICE_SERVICE_IRQ_VECTOR_ESI0 and set the
/// corresponding flags that Driver has to ack the DP receiver later.
///
/// Note that driver shall also call
/// drm_dp_mst_hpd_irq_send_new_request() if the 'handled' is set
/// after calling this function, to try to kick off a new request in
/// the queue if the previous message transaction is completed.
///
/// See also:
/// drm_dp_mst_hpd_irq_send_new_request()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_hpd_irq_handle_event")]
extern "C" fn drm_dp_mst_hpd_irq_handle_event(mgr: *mut drm_dp_mst_topology_mgr, esi: *const u8, ack: *mut u8, handled: *mut bool) -> i32 {
let mut ret: i32 = 0;
let mut sc: i32 = 0;
unsafe {
*handled = false;
}
sc = DP_GET_SINK_COUNT!(unsafe { *esi.wrapping_offset(0 as isize) });
if sc != ((unsafe { (*mgr).sink_count }) as i32) {
unsafe {
(*mgr).sink_count = sc as u8;
}
unsafe {
*handled = true;
}
}
update_msg_rx_state(mgr);
if ((unsafe { *esi.wrapping_offset(1 as isize) }) as i32) & 16i32 != 0 {
ret = drm_dp_mst_handle_down_rep(mgr);
unsafe {
*handled = true;
}
let __v0: *mut u8 = ack.wrapping_offset(1 as isize);
unsafe {
*__v0 = ((*__v0 as i32) | 16i32) as u8;
}
}
if ((unsafe { *esi.wrapping_offset(1 as isize) }) as i32) & 32i32 != 0 {
let __v3: i32 = drm_dp_mst_handle_up_req(mgr);
ret |= __v3;
unsafe {
*handled = true;
}
let __v6: *mut u8 = ack.wrapping_offset(1 as isize);
unsafe {
*__v6 = ((*__v6 as i32) | 32i32) as u8;
}
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_hpd_irq_handle_event_741: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr, *const u8, *mut u8, *mut bool) -> i32>, *mut ()>(Some(drm_dp_mst_hpd_irq_handle_event)) };

// drm_dp_mst_hpd_irq_send_new_request() - MST hotplug IRQ kick off new request
// @mgr: manager to notify irq for.
//
// This should be called from the driver when mst irq event is handled
// and acked. Note that new down request should only be sent when
// previous message transaction is completed. Source is not supposed to generate
// interleaved message transactions.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_hpd_irq_send_new_request_743: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr)>, *mut ()>(Some(drm_dp_mst_hpd_irq_send_new_request)) };

/// drm_dp_mst_detect_port() - get connection status for an MST port
/// @connector: DRM connector for this port
/// @ctx: The acquisition context to use for grabbing locks
/// @mgr: manager for this port
/// @port: pointer to a port
///
/// This returns the current connection state for a port.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_detect_port")]
extern "C" fn drm_dp_mst_detect_port(connector: *mut drm_connector, ctx: *mut drm_modeset_acquire_ctx, mgr: *mut drm_dp_mst_topology_mgr, mut port: *mut drm_dp_mst_port) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// we need to search for the port in the mgr in case it's gone
port = drm_dp_mst_topology_get_port_validated(mgr, port);
if port.is_null() {
return 2;
} else {
*__slate_slot_0 = drm_modeset_lock(core::ptr::addr_of_mut!((*mgr).base.lock), ctx);
if *__slate_slot_0 == 0 {
*__slate_slot_0 = 2;
if (*port).ddps {
let __t0: i32 = (*port).pdt as i32;
if __t0 != 0 {
if __t0 == 2 {
if !(*port).mcs {
*__slate_slot_0 = 1;
}
} else {
if __t0 == 3 {
*__slate_slot_0 = 1;
// for logical ports - cache the EDID
if drm_dp_mst_port_is_logical(port) && unsafe { (*port).cached_edid }.is_null() {
(*port).cached_edid = drm_edid_read_ddc(connector, core::ptr::addr_of_mut!((*port).aux.ddc));
}
} else {
if __t0 == 4 {
if (*port).ldps {
*__slate_slot_0 = 1;
}
}
}
}
}
}
}
drm_dp_mst_topology_put_port(port);
return *__slate_slot_0;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_detect_port_744: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_modeset_acquire_ctx, *mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_port) -> i32>, *mut ()>(Some(drm_dp_mst_detect_port)) };

/// drm_dp_mst_edid_read() - get EDID for an MST port
/// @connector: toplevel connector to get EDID for
/// @mgr: manager for this port
/// @port: unverified pointer to a port.
///
/// This returns an EDID for the port connected to a connector,
/// It validates the pointer still exists so the caller doesn't require a
/// reference.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_edid_read")]
extern "C" fn drm_dp_mst_edid_read(connector: *mut drm_connector, mgr: *mut drm_dp_mst_topology_mgr, mut port: *mut drm_dp_mst_port) -> *const drm_edid {
let mut drm_edid: *const drm_edid = unsafe { core::mem::zeroed() };
// we need to search for the port in the mgr in case it's gone
port = drm_dp_mst_topology_get_port_validated(mgr, port);
if port.is_null() {
return core::ptr::null::<drm_edid>();
}
if !unsafe { (*port).cached_edid }.is_null() {
drm_edid = unsafe { drm_edid_dup((*port).cached_edid) };
} else {
drm_edid = unsafe { drm_edid_read_ddc(connector, core::ptr::addr_of_mut!((*port).aux.ddc)) };
}
drm_dp_mst_topology_put_port(port);
drm_edid
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_edid_read_745: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_port) -> *const drm_edid>, *mut ()>(Some(drm_dp_mst_edid_read)) };

/// drm_dp_mst_get_edid() - get EDID for an MST port
/// @connector: toplevel connector to get EDID for
/// @mgr: manager for this port
/// @port: unverified pointer to a port.
///
/// This function is deprecated; please use drm_dp_mst_edid_read() instead.
///
/// This returns an EDID for the port connected to a connector,
/// It validates the pointer still exists so the caller doesn't require a
/// reference.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_get_edid")]
extern "C" fn drm_dp_mst_get_edid(connector: *mut drm_connector, mgr: *mut drm_dp_mst_topology_mgr, port: *mut drm_dp_mst_port) -> *mut edid {
let drm_edid: *const drm_edid = drm_dp_mst_edid_read(connector, mgr, port);
let edid: *mut edid = unsafe { drm_edid_duplicate(drm_edid_raw(drm_edid)) };
unsafe { drm_edid_free(drm_edid) };
edid
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_get_edid_746: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_port) -> *mut edid>, *mut ()>(Some(drm_dp_mst_get_edid)) };

/// drm_dp_atomic_find_time_slots() - Find and add time slots to the state
/// @state: global atomic state
/// @mgr: MST topology manager for the port
/// @port: port to find time slots for
/// @pbn: bandwidth required for the mode in PBN
///
/// Allocates time slots to @port, replacing any previous time slot allocations it may
/// have had. Any atomic drivers which support MST must call this function in
/// their &drm_encoder_helper_funcs.atomic_check() callback unconditionally to
/// change the current time slot allocation for the new state, and ensure the MST
/// atomic state is added whenever the state of payloads in the topology changes.
///
/// Allocations set by this function are not checked against the bandwidth
/// restraints of @mgr until the driver calls drm_dp_mst_atomic_check().
///
/// Additionally, it is OK to call this function multiple times on the same
/// @port as needed. It is not OK however, to call this function and
/// drm_dp_atomic_release_time_slots() in the same atomic check phase.
///
/// See also:
/// drm_dp_atomic_release_time_slots()
/// drm_dp_mst_atomic_check()
///
/// Returns:
/// Total slots in the atomic state assigned for this port, or a negative error
/// code if the port no longer exists
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_atomic_find_time_slots")]
extern "C" fn drm_dp_atomic_find_time_slots(state: *mut drm_atomic_commit, mgr: *mut drm_dp_mst_topology_mgr, port: *mut drm_dp_mst_port, pbn: i32) -> i32 {
let mut __slate_compound_14: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_16: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut topology_state: *mut drm_dp_mst_topology_state = unsafe { core::mem::zeroed() };
let mut payload: *mut drm_dp_mst_atomic_payload = core::ptr::null_mut::<drm_dp_mst_atomic_payload>();
let mut prev_slots: i32 = 0;
let mut prev_bw: i32 = 0;
let mut req_slots: i32 = 0;
topology_state = drm_atomic_get_mst_topology_state(state, mgr);
if IS_ERR(topology_state as *const ()) {
return PTR_ERR(topology_state as *const ()) as i32;
}
let conn_state: *mut drm_connector_state = drm_atomic_get_new_connector_state(state as *const drm_atomic_commit, unsafe { (*port).connector });
let __v0: u32 = drm_crtc_mask((unsafe { (*conn_state).crtc }) as *const drm_crtc);
unsafe {
(*topology_state).pending_crtc_mask = (*topology_state).pending_crtc_mask | __v0;
}
// Find the current allocation for this port, if any
payload = drm_atomic_get_mst_payload_state(topology_state, port);
if !payload.is_null() {
prev_slots = unsafe { (*payload).time_slots };
prev_bw = unsafe { (*payload).pbn };
// This should never happen, unless the driver tries
// releasing and allocating the same timeslot allocation,
// which is an error
let mut __v4: i64;
let __ret_warn_on: i32 = (((unsafe { (*payload).__slate_bits_0.__get_delete() }) as i32) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*mgr).dev as *const drm_device) as *const device), dev_name(__drm_to_dev((*mgr).dev as *const drm_device) as *const device), c"drm_WARN_ON(payload->delete)".as_ptr() as *mut u8) };
let mut __v5: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v5 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v6: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_drm_dp_atomic_find_time_slots_7, sym __slate_asm_string_drm_dp_atomic_find_time_slots_8, const 4467 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v6 = bug;
unsafe { __v5.unwrap()(__v6, dev_driver_string(__drm_to_dev((*mgr).dev as *const drm_device) as *const device), dev_name(__drm_to_dev((*mgr).dev as *const drm_device) as *const device), c"drm_WARN_ON(payload->delete)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v4 = (__ret_warn_on != 0) as i64;
if __v4 != 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*mgr).dev as *const drm_device) as *const device, c"[drm] *ERROR* cannot allocate and release time slots on [MST PORT:%p] in the same state\n".as_ptr() as *const u8, port) };
return 22i32.wrapping_neg();
}
}
req_slots = (dfixed_const!(pbn).wrapping_add(unsafe { (*topology_state).pbn_div.full }).wrapping_sub(1) / unsafe { (*topology_state).pbn_div.full }) as i32;
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 4u32, c"[CONNECTOR:%d:%s] [MST PORT:%p] TU %d -> %d\n".as_ptr() as *const u8, (*(*port).connector).base.id, (*(*port).connector).name, port, prev_slots, req_slots) };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 4u32, c"[CONNECTOR:%d:%s] [MST PORT:%p] PBN %d -> %d\n".as_ptr() as *const u8, (*(*port).connector).base.id, (*(*port).connector).name, port, prev_bw, pbn) };
// Add the new allocation to the state, note the VCPI isn't assigned until the end
if payload.is_null() {
let mut __v11: *mut drm_dp_mst_atomic_payload;
let __obj_size: u64 = size_mul(core::mem::size_of::<drm_dp_mst_atomic_payload>() as u64, 1u64);
let mut __v12: *mut ();
let mut __v13: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_14 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_14)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_16 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_16)
} });
}
__v13 = _res;
__v12 = __v13;
__v11 = __v12 as *mut drm_dp_mst_atomic_payload;
payload = __v11;
if payload.is_null() {
return 12i32.wrapping_neg();
}
drm_dp_mst_get_port_malloc(port);
unsafe {
(*payload).port = port;
}
unsafe {
(*payload).vc_start_slot = 1i32.wrapping_neg() as i8;
}
unsafe {
(*payload).payload_allocation_status = 0;
}
list_add(unsafe { core::ptr::addr_of_mut!((*payload).next) }, unsafe { core::ptr::addr_of_mut!((*topology_state).payloads) });
}
unsafe {
(*payload).time_slots = req_slots;
}
unsafe {
(*payload).pbn = pbn;
}
req_slots
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_747: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_atomic_find_time_slots_748: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit, *mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_port, i32) -> i32>, *mut ()>(Some(drm_dp_atomic_find_time_slots)) };

/// drm_dp_atomic_release_time_slots() - Release allocated time slots
/// @state: global atomic state
/// @mgr: MST topology manager for the port
/// @port: The port to release the time slots from
///
/// Releases any time slots that have been allocated to a port in the atomic
/// state. Any atomic drivers which support MST must call this function
/// unconditionally in their &drm_connector_helper_funcs.atomic_check() callback.
/// This helper will check whether time slots would be released by the new state and
/// respond accordingly, along with ensuring the MST state is always added to the
/// atomic state whenever a new state would modify the state of payloads on the
/// topology.
///
/// It is OK to call this even if @port has been removed from the system.
/// Additionally, it is OK to call this function multiple times on the same
/// @port as needed. It is not OK however, to call this function and
/// drm_dp_atomic_find_time_slots() on the same @port in a single atomic check
/// phase.
///
/// See also:
/// drm_dp_atomic_find_time_slots()
/// drm_dp_mst_atomic_check()
///
/// Returns:
/// 0 on success, negative error code otherwise
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_atomic_release_time_slots")]
extern "C" fn drm_dp_atomic_release_time_slots(state: *mut drm_atomic_commit, mgr: *mut drm_dp_mst_topology_mgr, port: *mut drm_dp_mst_port) -> i32 {
let mut topology_state: *mut drm_dp_mst_topology_state = unsafe { core::mem::zeroed() };
let mut update_payload: bool = true;
let old_conn_state: *mut drm_connector_state = drm_atomic_get_old_connector_state(state as *const drm_atomic_commit, unsafe { (*port).connector });
if unsafe { (*old_conn_state).crtc }.is_null() {
return 0;
}
// If the CRTC isn't disabled by this state, don't release it's payload
let new_conn_state: *mut drm_connector_state = drm_atomic_get_new_connector_state(state as *const drm_atomic_commit, unsafe { (*port).connector });
if !unsafe { (*new_conn_state).crtc }.is_null() {
let crtc_state: *mut drm_crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, unsafe { (*new_conn_state).crtc });
// No modeset means no payload changes, so it's safe to not pull in the MST state
let __v0: bool = crtc_state.is_null() || !drm_atomic_crtc_needs_modeset(crtc_state as *const drm_crtc_state);
if __v0 {
return 0;
}
if ((unsafe { (*crtc_state).__slate_bits_0.__get_mode_changed() }) as i32) == 0 && ((unsafe { (*crtc_state).__slate_bits_0.__get_connectors_changed() }) as i32) == 0 {
update_payload = false;
}
}
topology_state = drm_atomic_get_mst_topology_state(state, mgr);
if IS_ERR(topology_state as *const ()) {
return PTR_ERR(topology_state as *const ()) as i32;
}
let __v1: u32 = drm_crtc_mask((unsafe { (*old_conn_state).crtc }) as *const drm_crtc);
unsafe {
(*topology_state).pending_crtc_mask = (*topology_state).pending_crtc_mask | __v1;
}
if !update_payload {
return 0;
}
let payload: *mut drm_dp_mst_atomic_payload = drm_atomic_get_mst_payload_state(topology_state, port);
let mut __v5: i64;
let __ret_warn_on: i32 = payload.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("749: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 749b - ., 3; .popsection\n# {0}", const 749 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_dp_atomic_release_time_slots_6, sym __slate_asm_string_drm_dp_atomic_release_time_slots_7, const 4566 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("750: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 750b - ., 4; .popsection\n# {0}", const 750 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*mgr).dev as *const drm_device) as *const device, c"[drm] *ERROR* No payload for [MST PORT:%p] found in mst state %p\n".as_ptr() as *const u8, port, core::ptr::addr_of_mut!((*topology_state).base)) };
return 22i32.wrapping_neg();
}
if !unsafe { (*new_conn_state).crtc }.is_null() {
return 0;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 4u32, c"[MST PORT:%p] TU %d -> 0\n".as_ptr() as *const u8, port, (*payload).time_slots) };
if ((unsafe { (*payload).__slate_bits_0.__get_delete() }) as i32) == 0 {
unsafe {
(*payload).pbn = 0;
}
unsafe {
let __t0 = true;
(*payload).__slate_bits_0.__set_delete(__t0);
}
if ((unsafe { (*payload).vcpi }) as i32) > 0 {
let __v12: u32 = unsafe { (*topology_state).payload_mask };
let __v13: u32 = ((__v12 as u64) & !BIT!(((unsafe { (*payload).vcpi }) as i32).wrapping_sub(1))) as u32;
unsafe {
(*topology_state).payload_mask = __v13;
}
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_atomic_release_time_slots_751: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit, *mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_port) -> i32>, *mut ()>(Some(drm_dp_atomic_release_time_slots)) };

/// drm_dp_mst_atomic_setup_commit() - setup_commit hook for MST helpers
/// @state: global atomic state
///
/// This function saves all of the &drm_crtc_commit structs in an atomic state that touch any CRTCs
/// currently assigned to an MST topology. Drivers must call this hook from their
/// &drm_mode_config_helper_funcs.atomic_commit_setup hook.
///
/// Returns:
/// 0 if all CRTC commits were retrieved successfully, negative error code otherwise
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_atomic_setup_commit")]
extern "C" fn drm_dp_mst_atomic_setup_commit(state: *mut drm_atomic_commit) -> i32 {
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_7: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut mgr: *mut drm_dp_mst_topology_mgr = unsafe { core::mem::zeroed() };
let mut mst_state: *mut drm_dp_mst_topology_state = unsafe { core::mem::zeroed() };
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut j: i32 = 0;
let mut commit_idx: i32 = 0;
let mut num_commit_deps: i32 = 0;
'__slate_break_0: while i < unsafe { (*state).num_private_objs } {
if __drm_dp_mst_state_iter_get(state, unsafe { &mut (*core::ptr::addr_of_mut!(mgr)) }, core::ptr::null_mut::<*mut drm_dp_mst_topology_state>(), core::ptr::addr_of_mut!(mst_state), i) {
if (unsafe { (*mst_state).pending_crtc_mask }) != 0 {
num_commit_deps = hweight32!(unsafe { (*mst_state).pending_crtc_mask }) as i32;
let mut __v2: *mut *mut drm_crtc_commit;
let __obj_size: u64 = size_mul(core::mem::size_of::<*mut drm_crtc_commit>() as u64, num_commit_deps as u64);
let mut __v3: *mut ();
let mut __v4: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_7 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_7)
} });
}
__v4 = _res;
__v3 = __v4;
__v2 = __v3 as *mut *mut drm_crtc_commit;
unsafe {
(*mst_state).commit_deps = __v2;
}
if unsafe { (*mst_state).commit_deps }.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*mst_state).num_commit_deps = num_commit_deps as u64;
}
commit_idx = 0;
j = 0;
while j < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v8: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(j as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(j as isize)).ptr };
crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(j as isize)).new_state };
true
} else {
false
};
if __v8 {
if (unsafe { (*mst_state).pending_crtc_mask }) & drm_crtc_mask(crtc as *const drm_crtc) != 0 {
let __v10: i32 = commit_idx;
commit_idx = __v10.wrapping_add(1);
unsafe {
*unsafe { (*mst_state).commit_deps }.wrapping_offset(__v10 as isize) = drm_crtc_commit_get((*crtc_state).commit);
}
}
}
j = j.wrapping_add(1);
}
}
}
i = i.wrapping_add(1);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_atomic_setup_commit_752: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit) -> i32>, *mut ()>(Some(drm_dp_mst_atomic_setup_commit)) };

/// drm_dp_mst_atomic_wait_for_dependencies() - Wait for all pending commits on MST topologies,
/// prepare new MST state for commit
/// @state: global atomic state
///
/// Goes through any MST topologies in this atomic state, and waits for any pending commits which
/// touched CRTCs that were/are on an MST topology to be programmed to hardware and flipped to before
/// returning. This is to prevent multiple non-blocking commits affecting an MST topology from racing
/// with eachother by forcing them to be executed sequentially in situations where the only resources
/// the modeset objects in these commits share are an MST topology.
///
/// This function also prepares the new MST state for commit by performing some state preparation
/// which can't be done until this point, such as reading back the final VC start slots (which are
/// determined at commit-time) from the previous state.
///
/// All MST drivers must call this function after calling drm_atomic_helper_wait_for_dependencies(),
/// or whatever their equivalent of that is.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_atomic_wait_for_dependencies")]
extern "C" fn drm_dp_mst_atomic_wait_for_dependencies(state: *mut drm_atomic_commit) {
let mut old_mst_state: *mut drm_dp_mst_topology_state = unsafe { core::mem::zeroed() };
let mut new_mst_state: *mut drm_dp_mst_topology_state = unsafe { core::mem::zeroed() };
let mut mgr: *mut drm_dp_mst_topology_mgr = unsafe { core::mem::zeroed() };
let mut old_payload: *mut drm_dp_mst_atomic_payload = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut j: i32 = 0;
let mut ret: i32 = 0;
while i < unsafe { (*state).num_private_objs } {
if __drm_dp_mst_state_iter_get(state, unsafe { &mut (*core::ptr::addr_of_mut!(mgr)) }, core::ptr::addr_of_mut!(old_mst_state), core::ptr::addr_of_mut!(new_mst_state), i) {
j = 0;
while (j as u64) < unsafe { (*old_mst_state).num_commit_deps } {
ret = unsafe { drm_crtc_commit_wait(*unsafe { (*old_mst_state).commit_deps }.wrapping_offset(j as isize)) };
if ret < 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*state).dev as *const drm_device) as *const device, c"[drm] *ERROR* Failed to wait for %s: %d\n".as_ptr() as *const u8, (*(*(*unsafe { (*old_mst_state).commit_deps }.wrapping_offset(j as isize))).crtc).name, ret) };
}
j = j.wrapping_add(1);
}
// Now that previous state is committed, it's safe to copy over the start slot
// and allocation status assignments
old_payload = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*old_mst_state).payloads) }, next);
while list_entry_is_head!(old_payload, unsafe { core::ptr::addr_of_mut!((*old_mst_state).payloads) }, next) == 0 {
if ((unsafe { (*old_payload).__slate_bits_0.__get_delete() }) as i32) == 0 {
let new_payload: *mut drm_dp_mst_atomic_payload = drm_atomic_get_mst_payload_state(new_mst_state, unsafe { (*old_payload).port });
unsafe {
(*new_payload).vc_start_slot = (*old_payload).vc_start_slot;
}
unsafe {
(*new_payload).payload_allocation_status = (*old_payload).payload_allocation_status;
}
}
old_payload = list_next_entry__2!(old_payload, next);
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_atomic_wait_for_dependencies_753: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit)>, *mut ()>(Some(drm_dp_mst_atomic_wait_for_dependencies)) };

/// drm_dp_mst_root_conn_atomic_check() - Serialize CRTC commits on MST-capable connectors operating
/// in SST mode
/// @new_conn_state: The new connector state of the &drm_connector
/// @mgr: The MST topology manager for the &drm_connector
///
/// Since MST uses fake &drm_encoder structs, the generic atomic modesetting code isn't able to
/// serialize non-blocking commits happening on the real DP connector of an MST topology switching
/// into/away from MST mode - as the CRTC on the real DP connector and the CRTCs on the connector's
/// MST topology will never share the same &drm_encoder.
///
/// This function takes care of this serialization issue, by checking a root MST connector's atomic
/// state to determine if it is about to have a modeset - and then pulling in the MST topology state
/// if so, along with adding any relevant CRTCs to &drm_dp_mst_topology_state.pending_crtc_mask.
///
/// Drivers implementing MST must call this function from the
/// &drm_connector_helper_funcs.atomic_check hook of any physical DP &drm_connector capable of
/// driving MST sinks.
///
/// Returns:
/// 0 on success, negative error code otherwise
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_root_conn_atomic_check")]
extern "C" fn drm_dp_mst_root_conn_atomic_check(new_conn_state: *mut drm_connector_state, mgr: *mut drm_dp_mst_topology_mgr) -> i32 {
let state: *mut drm_atomic_commit = unsafe { (*new_conn_state).state };
let old_conn_state: *mut drm_connector_state = drm_atomic_get_old_connector_state(state as *const drm_atomic_commit, unsafe { (*new_conn_state).connector });
let mut crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut mst_state: *mut drm_dp_mst_topology_state = core::ptr::null_mut::<drm_dp_mst_topology_state>();
if !unsafe { (*new_conn_state).crtc }.is_null() {
crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, unsafe { (*new_conn_state).crtc });
let __v0: bool = !crtc_state.is_null() && drm_atomic_crtc_needs_modeset(crtc_state as *const drm_crtc_state);
if __v0 {
mst_state = drm_atomic_get_mst_topology_state(state, mgr);
if IS_ERR(mst_state as *const ()) {
return PTR_ERR(mst_state as *const ()) as i32;
}
let __v1: u32 = drm_crtc_mask((unsafe { (*new_conn_state).crtc }) as *const drm_crtc);
unsafe {
(*mst_state).pending_crtc_mask = (*mst_state).pending_crtc_mask | __v1;
}
}
}
if !unsafe { (*old_conn_state).crtc }.is_null() {
crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, unsafe { (*old_conn_state).crtc });
let __v5: bool = !crtc_state.is_null() && drm_atomic_crtc_needs_modeset(crtc_state as *const drm_crtc_state);
if __v5 {
if mst_state.is_null() {
mst_state = drm_atomic_get_mst_topology_state(state, mgr);
if IS_ERR(mst_state as *const ()) {
return PTR_ERR(mst_state as *const ()) as i32;
}
}
let __v6: u32 = drm_crtc_mask((unsafe { (*old_conn_state).crtc }) as *const drm_crtc);
unsafe {
(*mst_state).pending_crtc_mask = (*mst_state).pending_crtc_mask | __v6;
}
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_root_conn_atomic_check_754: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector_state, *mut drm_dp_mst_topology_mgr) -> i32>, *mut ()>(Some(drm_dp_mst_root_conn_atomic_check)) };

/// drm_dp_mst_update_slots() - updates the slot info depending on the DP ecoding format
/// @mst_state: mst_state to update
/// @link_encoding_cap: the ecoding format on the link
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_update_slots")]
extern "C" fn drm_dp_mst_update_slots(mst_state: *mut drm_dp_mst_topology_state, link_encoding_cap: u8) {
if (link_encoding_cap as i32) == 2i32 {
unsafe {
(*mst_state).total_avail_slots = 64;
}
unsafe {
(*mst_state).start_slot = 0;
}
} else {
unsafe {
(*mst_state).total_avail_slots = 63;
}
unsafe {
(*mst_state).start_slot = 1;
}
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"%s encoding format on mst_state 0x%p\n".as_ptr() as *const u8, if (link_encoding_cap as i32) == 2i32 { c"128b/132b".as_ptr() as *mut u8 } else { c"8b/10b".as_ptr() as *mut u8 }, mst_state) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_update_slots_755: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_state, u8)>, *mut ()>(Some(drm_dp_mst_update_slots)) };

/// drm_dp_check_act_status() - Polls for ACT handled status.
/// @mgr: manager to use
///
/// Tries waiting for the MST hub to finish updating it's payload table by
/// polling for the ACT handled bit for up to 3 seconds (yes-some hubs really
/// take that long).
///
/// Returns:
/// 0 if the ACT was handled in time, negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_check_act_status")]
extern "C" fn drm_dp_check_act_status(mgr: *mut drm_dp_mst_topology_mgr) -> i32 {
// There doesn't seem to be any recommended retry count or timeout in
// the MST specification. Since some hubs have been observed to take
// over 1 second to update their payload allocations under certain
// conditions, we use a rather large timeout value of 3 seconds.
unsafe { drm_dp_dpcd_poll_act_handled((*mgr).aux, 3000i32) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_check_act_status_756: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr) -> i32>, *mut ()>(Some(drm_dp_check_act_status)) };

/// drm_dp_calc_pbn_mode() - Calculate the PBN for a mode.
/// @clock: dot clock
/// @bpp: bpp as .4 binary fixed point
///
/// This uses the formula in the spec to calculate the PBN value for a mode.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_calc_pbn_mode")]
extern "C" fn drm_dp_calc_pbn_mode(clock: i32, bpp: i32) -> i32 {
// The unit of 54/64Mbytes/sec is an arbitrary unit chosen based on
// common multiplier to render an integer PBN for all link rate/lane
// counts combinations
// calculate
// peak_kbps = clock * bpp / 16
// peak_kbps *= SSC overhead / 1000000
// peak_kbps /= 8    convert to Kbytes
// peak_kBps *= (64/54) / 1000    convert to PBN
//
// TODO: Use the actual link and mode parameters to calculate
// the overhead. For now it's assumed that these are
// 4 link lanes, 4096 hactive pixels, which don't add any
// significant data padding overhead and that there is no DSC
// or FEC overhead.
let overhead: i32 = unsafe { drm_dp_bw_overhead(4i32, 4096i32, 0i32, bpp, BIT!(0i32) | BIT!(2i32)) };
DIV64_U64_ROUND_UP!(mul_u32_u32(clock.wrapping_mul(bpp) as u32, (64i32.wrapping_mul(overhead) >> 4i32) as u32), 432000000000u64) as i32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_calc_pbn_mode_757: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, i32) -> i32>, *mut ()>(Some(drm_dp_calc_pbn_mode)) };

/// we want to kick the TX after we've ack the up/down IRQs.
fn drm_dp_mst_kick_tx(mgr: *mut drm_dp_mst_topology_mgr) {
queue_work(unsafe { system_long_wq }, unsafe { core::ptr::addr_of_mut!((*mgr).tx_work) });
}

/// Helper function for parsing DP device types into convenient strings
/// for use with dp_mst_topology
fn pdt_to_string(pdt: u8) -> *const u8 {
match pdt as i32 {
0 => {
return c"NONE".as_ptr() as *const u8;
}
1 => {
return c"SOURCE OR SST".as_ptr() as *const u8;
}
2 => {
return c"MST BRANCHING".as_ptr() as *const u8;
}
3 => {
return c"SST SINK".as_ptr() as *const u8;
}
4 => {
return c"DP LEGACY CONV".as_ptr() as *const u8;
}
_ => {
return c"ERR".as_ptr() as *const u8;
}
}
unsafe { core::mem::zeroed() }
}

fn drm_dp_mst_dump_mstb(m: *mut seq_file, mstb: *mut drm_dp_mst_branch) {
let mut port: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
let tabs: i32 = (unsafe { (*mstb).lct }) as i32;
let mut prefix: [u8; 10] = [0u8; 10];
let mut i: i32 = 0;
while i < tabs {
prefix[i as usize] = 9;
i = i.wrapping_add(1);
}
prefix[i as usize] = 0;
unsafe { seq_printf(m, c"%smstb - [%p]: num_ports: %d\n".as_ptr() as *const u8, prefix.as_mut_ptr() as *mut u8, mstb, (*mstb).num_ports) };
port = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next);
while list_entry_is_head!(port, unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next) == 0 {
unsafe { seq_printf(m, c"%sport %d - [%p] (%s - %s): ddps: %d, ldps: %d, sdp: %d/%d, fec: %s, conn: %p\n".as_ptr() as *const u8, prefix.as_mut_ptr() as *mut u8, (*port).port_num as i32, port, if (*port).input { c"input".as_ptr() as *mut u8 } else { c"output".as_ptr() as *mut u8 }, pdt_to_string((*port).pdt), (*port).ddps as i32, (*port).ldps as i32, (*port).num_sdp_streams as i32, (*port).num_sdp_stream_sinks as i32, if (*port).fec_capable { c"true".as_ptr() as *mut u8 } else { c"false".as_ptr() as *mut u8 }, (*port).connector) };
if !unsafe { (*port).mstb }.is_null() {
drm_dp_mst_dump_mstb(m, unsafe { (*port).mstb });
}
port = list_next_entry!(port, next);
}
}

fn dump_dp_payload_table(mgr: *mut drm_dp_mst_topology_mgr, buf: *mut u8) -> bool {
let mut i: i32 = 0;
while i < 64 {
if drm_dp_dpcd_read_data(unsafe { (*mgr).aux }, 704i32.wrapping_add(i) as u32, buf.wrapping_offset(i as isize) as *mut (), 16u64) < 0 {
return false;
}
i = i.wrapping_add(16);
}
true
}

fn fetch_monitor_name(mgr: *mut drm_dp_mst_topology_mgr, port: *mut drm_dp_mst_port, name: *mut u8, namelen: i32) {
let mst_edid: *mut edid = drm_dp_mst_get_edid(unsafe { (*port).connector }, mgr, port);
unsafe { drm_edid_get_monitor_name(mst_edid as *const edid, name, namelen) };
unsafe { kfree(mst_edid as *const ()) };
}

/// drm_dp_mst_dump_topology(): dump topology to seq file.
/// @m: seq_file to dump output to
/// @mgr: manager to dump current topology for.
///
/// helper to dump MST topology to a seq file for debugfs.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_dump_topology")]
extern "C" fn drm_dp_mst_dump_topology(m: *mut seq_file, mgr: *mut drm_dp_mst_topology_mgr) {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut buf: core::mem::MaybeUninit<__SlateAlign16<[u8; 64]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut [u8; 64] = core::ptr::addr_of_mut!(buf) as *mut [u8; 64];
let mut name: core::mem::MaybeUninit<[u8; 14]> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut [u8; 14] = core::ptr::addr_of_mut!(name) as *mut [u8; 14];
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut payload: core::mem::MaybeUninit<*mut drm_dp_mst_atomic_payload> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut drm_dp_mst_atomic_payload = core::ptr::addr_of_mut!(payload) as *mut *mut drm_dp_mst_atomic_payload;
let mut state: core::mem::MaybeUninit<*mut drm_dp_mst_topology_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut drm_dp_mst_topology_state = core::ptr::addr_of_mut!(state) as *mut *mut drm_dp_mst_topology_state;
unsafe {
{
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
mutex_lock(core::ptr::addr_of_mut!((*mgr).lock));
if !unsafe { (*mgr).mst_primary }.is_null() {
drm_dp_mst_dump_mstb(m, (*mgr).mst_primary);
}
}
// dump VCPIs
mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock));
*__slate_slot_7 = drm_modeset_lock_single_interruptible(core::ptr::addr_of_mut!((*mgr).base.lock));
if *__slate_slot_7 < 0 {
return;
} else {
*__slate_slot_10 = to_drm_dp_mst_topology_state((*mgr).base.state);
seq_printf(m, c"\n*** Atomic state info ***\n".as_ptr() as *const u8);
seq_printf(m, c"payload_mask: %x, max_payloads: %d, start_slot: %u, pbn_div: %d\n".as_ptr() as *const u8, (*(*__slate_slot_10)).payload_mask, (*mgr).max_payloads, (*(*__slate_slot_10)).start_slot as i32, dfixed_trunc!((*(*__slate_slot_10)).pbn_div));
seq_printf(m, c"\n| idx | port | vcpi | slots | pbn | dsc | status |     sink name     |\n".as_ptr() as *const u8);
*__slate_slot_8 = 0;
while *__slate_slot_8 < (*mgr).max_payloads {
*__slate_slot_9 = list_first_entry__2!(core::ptr::addr_of_mut!((*(*__slate_slot_10)).payloads), next);
while list_entry_is_head!(*__slate_slot_9, core::ptr::addr_of_mut!((*(*__slate_slot_10)).payloads), next) == 0 {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 14usize);
if !(((*(*__slate_slot_9)).vcpi as i32) != *__slate_slot_8 || ((*(*__slate_slot_9)).__slate_bits_0.__get_delete() as i32) != 0) {
fetch_monitor_name(mgr, (*(*__slate_slot_9)).port, (*__slate_slot_6).as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 14]>() as i32);
seq_printf(m, c" %5d %6d %6d %02d - %02d %5d %5s %8s %19s\n".as_ptr() as *const u8, *__slate_slot_8, (*(*(*__slate_slot_9)).port).port_num as i32, (*(*__slate_slot_9)).vcpi as i32, (*(*__slate_slot_9)).vc_start_slot as i32, ((*(*__slate_slot_9)).vc_start_slot as i32).wrapping_add((*(*__slate_slot_9)).time_slots).wrapping_sub(1), (*(*__slate_slot_9)).pbn, if ((*(*__slate_slot_9)).__slate_bits_0.__get_dsc_enabled() as i32) != 0 { c"Y".as_ptr() as *mut u8 } else { c"N".as_ptr() as *mut u8 }, status.0[((*(*__slate_slot_9)).payload_allocation_status as u32) as usize], if (*((*__slate_slot_6).as_mut_ptr() as *mut u8) as i32) != 0 { (*__slate_slot_6).as_mut_ptr() as *mut u8 } else { c"Unknown".as_ptr() as *mut u8 });
}
*__slate_slot_9 = list_next_entry__2!(*__slate_slot_9, next);
}
let __v5: i32 = *__slate_slot_8;
*__slate_slot_8 = __v5.wrapping_add(1);
}
seq_printf(m, c"\n*** DPCD Info ***\n".as_ptr() as *const u8);
mutex_lock(core::ptr::addr_of_mut!((*mgr).lock));
if !unsafe { (*mgr).mst_primary }.is_null() {
let __slate_auto_bytes_4 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 64usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if drm_dp_read_dpcd_caps((*mgr).aux, (*__slate_slot_3).as_mut_ptr() as *mut u8) < 0 {
seq_printf(m, c"dpcd read failed\n".as_ptr() as *const u8);
} else {
seq_printf(m, c"dpcd: %*ph\n".as_ptr() as *const u8, 15i32, (*__slate_slot_3).as_mut_ptr() as *mut u8);
*__slate_slot_2 = drm_dp_dpcd_read_data((*mgr).aux, 32u32, (*__slate_slot_3).as_mut_ptr() as *mut (), 2u64);
if *__slate_slot_2 < 0 {
seq_printf(m, c"faux/mst read failed\n".as_ptr() as *const u8);
} else {
seq_printf(m, c"faux/mst: %*ph\n".as_ptr() as *const u8, 2i32, (*__slate_slot_3).as_mut_ptr() as *mut u8);
*__slate_slot_2 = drm_dp_dpcd_read_data((*mgr).aux, 273u32, (*__slate_slot_3).as_mut_ptr() as *mut (), 1u64);
if *__slate_slot_2 < 0 {
seq_printf(m, c"mst ctrl read failed\n".as_ptr() as *const u8);
} else {
seq_printf(m, c"mst ctrl: %*ph\n".as_ptr() as *const u8, 1i32, (*__slate_slot_3).as_mut_ptr() as *mut u8);
// dump the standard OUI branch header
*__slate_slot_2 = drm_dp_dpcd_read_data((*mgr).aux, 1280u32, (*__slate_slot_3).as_mut_ptr() as *mut (), 12u64);
if *__slate_slot_2 < 0 {
seq_printf(m, c"branch oui read failed\n".as_ptr() as *const u8);
} else {
seq_printf(m, c"branch oui: %*phN devid: ".as_ptr() as *const u8, 3i32, (*__slate_slot_3).as_mut_ptr() as *mut u8);
*__slate_slot_8 = 3;
while *__slate_slot_8 < 8 && (*__slate_slot_3)[*__slate_slot_8 as usize] != 0 {
seq_putc(m, (*__slate_slot_3)[*__slate_slot_8 as usize]);
let __v1: i32 = *__slate_slot_8;
*__slate_slot_8 = __v1.wrapping_add(1);
}
seq_printf(m, c" revision: hw: %x.%x sw: %x.%x\n".as_ptr() as *const u8, ((*__slate_slot_3)[9] as i32) >> 4i32, ((*__slate_slot_3)[9] as i32) & 15, (*__slate_slot_3)[10] as i32, (*__slate_slot_3)[11] as i32);
if dump_dp_payload_table(mgr, (*__slate_slot_3).as_mut_ptr() as *mut u8) {
seq_printf(m, c"payload table: %*ph\n".as_ptr() as *const u8, 64i32, (*__slate_slot_3).as_mut_ptr() as *mut u8);
}
}
}
}
}
}
mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock));
drm_modeset_unlock(core::ptr::addr_of_mut!((*mgr).base.lock));
}
}
}

static status: __SlateAlign16<[*const u8; 4]> = __SlateAlign16([(b"None\0".as_ptr() as *mut u8) as *const u8, (b"Local\0".as_ptr() as *mut u8) as *const u8, (b"DFP\0".as_ptr() as *mut u8) as *const u8, (b"Remote\0".as_ptr() as *mut u8) as *const u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_dump_topology_758: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut seq_file, *mut drm_dp_mst_topology_mgr)>, *mut ()>(Some(drm_dp_mst_dump_topology)) };

#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_tx_work")]
extern "C" fn drm_dp_tx_work(work: *mut work_struct) {
let mgr: *mut drm_dp_mst_topology_mgr = container_of__23!(work, drm_dp_mst_topology_mgr);
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).qlock)) };
if list_empty((unsafe { core::ptr::addr_of_mut!((*mgr).tx_msg_downq) }) as *const list_head) == 0 {
process_single_down_tx_qlock(mgr);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).qlock)) };
}

fn drm_dp_delayed_destroy_port(port: *mut drm_dp_mst_port) {
drm_dp_port_set_pdt(port, 0u8, unsafe { (*port).mcs });
if !unsafe { (*port).connector }.is_null() {
unsafe { drm_connector_unregister((*port).connector) };
drm_connector_put(unsafe { (*port).connector });
}
drm_dp_mst_put_port_malloc(port);
}

fn drm_dp_delayed_destroy_mstb(mstb: *mut drm_dp_mst_branch) {
let mgr: *mut drm_dp_mst_topology_mgr = unsafe { (*mstb).mgr };
let mut port: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
let mut port_tmp: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
let mut txmsg: *mut drm_dp_sideband_msg_tx = unsafe { core::mem::zeroed() };
let mut txmsg_tmp: *mut drm_dp_sideband_msg_tx = unsafe { core::mem::zeroed() };
let mut wake_tx: bool = false;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).lock)) };
port = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next);
port_tmp = list_next_entry!(port, next);
while list_entry_is_head!(port, unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next) == 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*port).next) });
drm_dp_mst_topology_put_port(port);
port = port_tmp;
port_tmp = list_next_entry!(port_tmp, next);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock)) };
// drop any tx slot msg
unsafe { mutex_lock(core::ptr::addr_of_mut!((*(*mstb).mgr).qlock)) };
txmsg = list_first_entry__9!(unsafe { core::ptr::addr_of_mut!((*mgr).tx_msg_downq) }, next);
txmsg_tmp = list_next_entry__10!(txmsg, next);
while list_entry_is_head!(txmsg, unsafe { core::ptr::addr_of_mut!((*mgr).tx_msg_downq) }, next) == 0 {
if (unsafe { (*txmsg).dst }) == mstb {
unsafe {
(*txmsg).state = 4;
}
list_del(unsafe { core::ptr::addr_of_mut!((*txmsg).next) });
wake_tx = true;
}
txmsg = txmsg_tmp;
txmsg_tmp = list_next_entry__10!(txmsg_tmp, next);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*(*mstb).mgr).qlock)) };
if wake_tx {
wake_up_all!(unsafe { core::ptr::addr_of_mut!((*(*mstb).mgr).tx_waitq) });
}
drm_dp_mst_put_mstb_malloc(mstb);
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_delayed_destroy_work")]
extern "C" fn drm_dp_delayed_destroy_work(work: *mut work_struct) {
let mgr: *mut drm_dp_mst_topology_mgr = container_of__26!(work, drm_dp_mst_topology_mgr);
let mut send_hotplug: bool = false;
let mut go_again: bool = false;
// Not a regular list traverse as we have to drop the destroy
// connector lock before destroying the mstb/port, to avoid AB->BA
// ordering between this lock and the config mutex.
loop {
go_again = false;
loop {
let mut mstb: *mut drm_dp_mst_branch = unsafe { core::mem::zeroed() };
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).delayed_destroy_lock)) };
let mut __v1: *mut drm_dp_mst_branch;
let head__: *mut list_head = unsafe { core::ptr::addr_of_mut!((*mgr).destroy_branch_device_list) };
let mut __v2: *mut list_head;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_759() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*head__).next) as *const *mut list_head) };
let pos__: *mut list_head = __v2;
let __v4: *mut drm_dp_mst_branch = if pos__ != head__ {
list_entry__13!(pos__, drm_dp_mst_branch)
} else {
core::ptr::null_mut::<drm_dp_mst_branch>()
};
__v1 = __v4;
mstb = __v1;
if !mstb.is_null() {
list_del(unsafe { core::ptr::addr_of_mut!((*mstb).destroy_next) });
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).delayed_destroy_lock)) };
if mstb.is_null() {
break;
}
drm_dp_delayed_destroy_mstb(mstb);
go_again = true;
}
loop {
let mut port: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).delayed_destroy_lock)) };
let mut __v7: *mut drm_dp_mst_port;
let head__: *mut list_head = unsafe { core::ptr::addr_of_mut!((*mgr).destroy_port_list) };
let mut __v8: *mut list_head;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_760() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v8 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*head__).next) as *const *mut list_head) };
let pos__: *mut list_head = __v8;
let __v10: *mut drm_dp_mst_port = if pos__ != head__ {
list_entry__2!(pos__, drm_dp_mst_port)
} else {
core::ptr::null_mut::<drm_dp_mst_port>()
};
__v7 = __v10;
port = __v7;
if !port.is_null() {
list_del(unsafe { core::ptr::addr_of_mut!((*port).next) });
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).delayed_destroy_lock)) };
if port.is_null() {
break;
}
drm_dp_delayed_destroy_port(port);
send_hotplug = true;
go_again = true;
}
if !go_again {
break;
}
}
if send_hotplug {
unsafe { drm_kms_helper_hotplug_event((*mgr).dev) };
}
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_duplicate_state")]
extern "C" fn drm_dp_mst_duplicate_state(obj: *mut drm_private_obj) -> *mut drm_private_state {
let mut payload: core::mem::MaybeUninit<*mut drm_dp_mst_atomic_payload> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut drm_dp_mst_atomic_payload = core::ptr::addr_of_mut!(payload) as *mut *mut drm_dp_mst_atomic_payload;
let mut pos: core::mem::MaybeUninit<*mut drm_dp_mst_atomic_payload> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut drm_dp_mst_atomic_payload = core::ptr::addr_of_mut!(pos) as *mut *mut drm_dp_mst_atomic_payload;
let mut old_state: *mut drm_dp_mst_topology_state = unsafe { core::mem::zeroed() };
let mut state: core::mem::MaybeUninit<*mut drm_dp_mst_topology_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut drm_dp_mst_topology_state = core::ptr::addr_of_mut!(state) as *mut *mut drm_dp_mst_topology_state;
unsafe {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
old_state = to_dp_mst_topology_state!(unsafe { (*obj).state });
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_5 = alloc_hooks!(kmemdup_noprof(old_state as *const (), core::mem::size_of::<drm_dp_mst_topology_state>() as u64, 3264u32)) as *mut drm_dp_mst_topology_state;
if (*__slate_slot_5).is_null() {
return core::ptr::null_mut::<drm_private_state>();
} else {
__drm_atomic_helper_private_obj_duplicate_state(obj, core::ptr::addr_of_mut!((*(*__slate_slot_5)).base));
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_5)).payloads));
(*(*__slate_slot_5)).commit_deps = core::ptr::null_mut::<*mut drm_crtc_commit>();
(*(*__slate_slot_5)).num_commit_deps = 0;
(*(*__slate_slot_5)).pending_crtc_mask = 0;
*__slate_slot_3 = list_first_entry__2!(core::ptr::addr_of_mut!((*old_state).payloads), next);
'__join_3: {
while list_entry_is_head!(*__slate_slot_3, core::ptr::addr_of_mut!((*old_state).payloads), next) == 0 {
// Prune leftover freed timeslot allocations
if ((*(*__slate_slot_3)).__slate_bits_0.__get_delete() as i32) == 0 {
*__slate_slot_2 = alloc_hooks!(kmemdup_noprof(*__slate_slot_3 as *const (), core::mem::size_of::<drm_dp_mst_atomic_payload>() as u64, 3264u32)) as *mut drm_dp_mst_atomic_payload;
if (*__slate_slot_2).is_null() {
break '__join_3;
} else {
drm_dp_mst_get_port_malloc((*(*__slate_slot_2)).port);
list_add(core::ptr::addr_of_mut!((*(*__slate_slot_2)).next), core::ptr::addr_of_mut!((*(*__slate_slot_5)).payloads));
}
}
*__slate_slot_3 = list_next_entry__2!(*__slate_slot_3, next);
}
return core::ptr::addr_of_mut!((*(*__slate_slot_5)).base);
}
*__slate_slot_3 = list_first_entry__2!(core::ptr::addr_of_mut!((*(*__slate_slot_5)).payloads), next);
let __v1: *mut drm_dp_mst_atomic_payload = list_next_entry__2!(*__slate_slot_3, next);
*__slate_slot_2 = __v1;
while list_entry_is_head!(*__slate_slot_3, core::ptr::addr_of_mut!((*(*__slate_slot_5)).payloads), next) == 0 {
drm_dp_mst_put_port_malloc((*(*__slate_slot_3)).port);
kfree(*__slate_slot_3 as *const ());
*__slate_slot_3 = *__slate_slot_2;
let __v0: *mut drm_dp_mst_atomic_payload = list_next_entry__2!(*__slate_slot_2, next);
*__slate_slot_2 = __v0;
}
kfree(*__slate_slot_5 as *const ());
return core::ptr::null_mut::<drm_private_state>();
}
}
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_destroy_state")]
extern "C" fn drm_dp_mst_destroy_state(obj: *mut drm_private_obj, state: *mut drm_private_state) {
let mst_state: *mut drm_dp_mst_topology_state = to_dp_mst_topology_state!(state);
let mut pos: *mut drm_dp_mst_atomic_payload = unsafe { core::mem::zeroed() };
let mut tmp: *mut drm_dp_mst_atomic_payload = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
pos = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*mst_state).payloads) }, next);
tmp = list_next_entry__2!(pos, next);
while list_entry_is_head!(pos, unsafe { core::ptr::addr_of_mut!((*mst_state).payloads) }, next) == 0 {
// We only keep references to ports with active payloads
if ((unsafe { (*pos).__slate_bits_0.__get_delete() }) as i32) == 0 {
drm_dp_mst_put_port_malloc(unsafe { (*pos).port });
}
unsafe { kfree(pos as *const ()) };
pos = tmp;
tmp = list_next_entry__2!(tmp, next);
}
while (i as u64) < unsafe { (*mst_state).num_commit_deps } {
drm_crtc_commit_put(unsafe { *unsafe { (*mst_state).commit_deps }.wrapping_offset(i as isize) });
i = i.wrapping_add(1);
}
unsafe { kfree((*mst_state).commit_deps as *const ()) };
unsafe { kfree(mst_state as *const ()) };
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_atomic_create_state")]
extern "C" fn drm_dp_mst_atomic_create_state(obj: *mut drm_private_obj) -> *mut drm_private_state {
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mgr: *mut drm_dp_mst_topology_mgr = to_dp_mst_topology_mgr!(obj);
let mut mst_state: *mut drm_dp_mst_topology_state = unsafe { core::mem::zeroed() };
let mut __v1: *mut drm_dp_mst_topology_state;
let __obj_size: u64 = size_mul(core::mem::size_of::<drm_dp_mst_topology_state>() as u64, 1u64);
let mut __v2: *mut ();
let mut __v3: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_6 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_6)
} });
}
__v3 = _res;
__v2 = __v3;
__v1 = __v2 as *mut drm_dp_mst_topology_state;
mst_state = __v1;
if mst_state.is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut drm_private_state;
}
unsafe { __drm_atomic_helper_private_obj_create_state(obj, core::ptr::addr_of_mut!((*mst_state).base)) };
unsafe {
(*mst_state).total_avail_slots = 63;
}
unsafe {
(*mst_state).start_slot = 1;
}
unsafe {
(*mst_state).mgr = mgr;
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*mst_state).payloads) });
unsafe { core::ptr::addr_of_mut!((*mst_state).base) }
}

fn drm_dp_mst_port_downstream_of_branch(mut port: *mut drm_dp_mst_port, branch: *mut drm_dp_mst_branch) -> bool {
while !unsafe { (*port).parent }.is_null() {
if (unsafe { (*port).parent }) == branch {
return true;
}
if !unsafe { (*(*port).parent).port_parent }.is_null() {
port = unsafe { (*(*port).parent).port_parent };
} else {
break;
}
}
false
}

fn drm_dp_mst_port_downstream_of_parent_locked(mgr: *mut drm_dp_mst_topology_mgr, mut port: *mut drm_dp_mst_port, mut parent: *mut drm_dp_mst_port) -> bool {
if unsafe { (*mgr).mst_primary }.is_null() {
return false;
}
port = drm_dp_mst_topology_get_port_validated_locked(unsafe { (*mgr).mst_primary }, port);
if port.is_null() {
return false;
}
if parent.is_null() {
return true;
}
parent = drm_dp_mst_topology_get_port_validated_locked(unsafe { (*mgr).mst_primary }, parent);
if parent.is_null() {
return false;
}
if unsafe { (*parent).mstb }.is_null() {
return false;
}
drm_dp_mst_port_downstream_of_branch(port, unsafe { (*parent).mstb })
}

/// drm_dp_mst_port_downstream_of_parent - check if a port is downstream of a parent port
/// @mgr: MST topology manager
/// @port: the port being looked up
/// @parent: the parent port
///
/// The function returns %true if @port is downstream of @parent. If @parent is
/// %NULL - denoting the root port - the function returns %true if @port is in
/// @mgr's topology.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_port_downstream_of_parent")]
extern "C" fn drm_dp_mst_port_downstream_of_parent(mgr: *mut drm_dp_mst_topology_mgr, port: *mut drm_dp_mst_port, parent: *mut drm_dp_mst_port) -> bool {
let mut ret: bool = false;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).lock)) };
ret = drm_dp_mst_port_downstream_of_parent_locked(mgr, port, parent);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock)) };
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_port_downstream_of_parent_761: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_port, *mut drm_dp_mst_port) -> bool>, *mut ()>(Some(drm_dp_mst_port_downstream_of_parent)) };

fn drm_dp_mst_atomic_check_mstb_bw_limit(mstb: *mut drm_dp_mst_branch, state: *mut drm_dp_mst_topology_state, failing_port: *mut *mut drm_dp_mst_port) -> i32 {
let mut payload: *mut drm_dp_mst_atomic_payload = unsafe { core::mem::zeroed() };
let mut port: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
let mut pbn_used: i32 = 0;
let mut ret: i32 = 0;
let mut found: bool = false;
// Check that we have at least one port in our state that's downstream
// of this branch, otherwise we can skip this branch
payload = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*state).payloads) }, next);
while list_entry_is_head!(payload, unsafe { core::ptr::addr_of_mut!((*state).payloads) }, next) == 0 {
let __v1: bool = (unsafe { (*payload).pbn }) == 0 || !drm_dp_mst_port_downstream_of_branch(unsafe { (*payload).port }, mstb);
if !__v1 {
found = true;
break;
}
payload = list_next_entry__2!(payload, next);
}
if !found {
return 0;
}
if !unsafe { (*mstb).port_parent }.is_null() {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*mstb).mgr).dev as *const drm_device) as *const device, 4u32, c"[MSTB:%p] [MST PORT:%p] Checking bandwidth limits on [MSTB:%p]\n".as_ptr() as *const u8, (*(*mstb).port_parent).parent, (*mstb).port_parent, mstb) };
} else {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*mstb).mgr).dev as *const drm_device) as *const device, 4u32, c"[MSTB:%p] Checking bandwidth limits\n".as_ptr() as *const u8, mstb) };
}
port = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next);
while list_entry_is_head!(port, unsafe { core::ptr::addr_of_mut!((*mstb).ports) }, next) == 0 {
ret = drm_dp_mst_atomic_check_port_bw_limit(port, state, failing_port);
if ret < 0 {
return ret;
}
pbn_used = pbn_used.wrapping_add(ret);
port = list_next_entry!(port, next);
}
pbn_used
}

fn drm_dp_mst_atomic_check_port_bw_limit(port: *mut drm_dp_mst_port, state: *mut drm_dp_mst_topology_state, failing_port: *mut *mut drm_dp_mst_port) -> i32 {
let mut pbn_used: i32 = 0;
if ((unsafe { (*port).pdt }) as i32) == 0 {
return 0;
}
if drm_dp_mst_is_end_device(unsafe { (*port).pdt }, unsafe { (*port).mcs }) {
let payload: *mut drm_dp_mst_atomic_payload = drm_atomic_get_mst_payload_state(state, port);
if payload.is_null() {
return 0;
}
// This could happen if the sink deasserted its HPD line, but
// the branch device still reports it as attached (PDT != NONE).
if (unsafe { (*port).full_pbn }) == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*port).mgr).dev as *const drm_device) as *const device, 4u32, c"[MSTB:%p] [MST PORT:%p] no BW available for the port\n".as_ptr() as *const u8, (*port).parent, port) };
unsafe {
*failing_port = port;
}
return 22i32.wrapping_neg();
}
pbn_used = unsafe { (*payload).pbn };
} else {
pbn_used = drm_dp_mst_atomic_check_mstb_bw_limit(unsafe { (*port).mstb }, state, failing_port);
if pbn_used <= 0 {
return pbn_used;
}
}
if pbn_used > ((unsafe { (*port).full_pbn }) as i32) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*port).mgr).dev as *const drm_device) as *const device, 4u32, c"[MSTB:%p] [MST PORT:%p] required PBN of %d exceeds port limit of %d\n".as_ptr() as *const u8, (*port).parent, port, pbn_used, (*port).full_pbn as i32) };
unsafe {
*failing_port = port;
}
return 28i32.wrapping_neg();
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*port).mgr).dev as *const drm_device) as *const device, 4u32, c"[MSTB:%p] [MST PORT:%p] uses %d out of %d PBN\n".as_ptr() as *const u8, (*port).parent, port, pbn_used, (*port).full_pbn as i32) };
pbn_used
}

fn drm_dp_mst_atomic_check_payload_alloc_limits(mgr: *mut drm_dp_mst_topology_mgr, mst_state: *mut drm_dp_mst_topology_state) -> i32 {
let mut payload: *mut drm_dp_mst_atomic_payload = unsafe { core::mem::zeroed() };
let mut avail_slots: i32 = (unsafe { (*mst_state).total_avail_slots }) as i32;
let mut payload_count: i32 = 0;
payload = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*mst_state).payloads) }, next);
while list_entry_is_head!(payload, unsafe { core::ptr::addr_of_mut!((*mst_state).payloads) }, next) == 0 {
// Releasing payloads is always OK-even if the port is gone
if ((unsafe { (*payload).__slate_bits_0.__get_delete() }) as i32) != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 4u32, c"[MST PORT:%p] releases all time slots\n".as_ptr() as *const u8, (*payload).port) };
} else {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 4u32, c"[MST PORT:%p] requires %d time slots\n".as_ptr() as *const u8, (*payload).port, (*payload).time_slots) };
avail_slots = avail_slots.wrapping_sub(unsafe { (*payload).time_slots });
if avail_slots < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 4u32, c"[MST PORT:%p] not enough time slots in mst state %p (avail=%d)\n".as_ptr() as *const u8, (*payload).port, mst_state, avail_slots.wrapping_add((*payload).time_slots)) };
return 28i32.wrapping_neg();
}
let __v5: i32 = payload_count.wrapping_add(1);
payload_count = __v5;
if __v5 > unsafe { (*mgr).max_payloads } {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 4u32, c"[MST MGR:%p] state %p has too many payloads (max=%d)\n".as_ptr() as *const u8, mgr, mst_state, (*mgr).max_payloads) };
return 22i32.wrapping_neg();
}
// Assign a VCPI
if (unsafe { (*payload).vcpi }) == 0 {
unsafe {
(*payload).vcpi = ffz!(unsafe { (*mst_state).payload_mask }).wrapping_add(1) as u8;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 4u32, c"[MST PORT:%p] assigned VCPI #%d\n".as_ptr() as *const u8, (*payload).port, (*payload).vcpi as i32) };
let __v7: u32 = unsafe { (*mst_state).payload_mask };
let __v8: u32 = ((__v7 as u64) | BIT!(((unsafe { (*payload).vcpi }) as i32).wrapping_sub(1))) as u32;
unsafe {
(*mst_state).payload_mask = __v8;
}
}
}
payload = list_next_entry__2!(payload, next);
}
if payload_count == 0 {
unsafe {
(*mst_state).pbn_div.full = dfixed_const!(0i32);
}
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 4u32, c"[MST MGR:%p] mst state %p TU pbn_div=%d avail=%d used=%d\n".as_ptr() as *const u8, mgr, mst_state, dfixed_trunc!((*mst_state).pbn_div), avail_slots, ((*mst_state).total_avail_slots as i32).wrapping_sub(avail_slots)) };
0
}

/// drm_dp_mst_add_affected_dsc_crtcs
/// @state: Pointer to the new struct drm_dp_mst_topology_state
/// @mgr: MST topology manager
///
/// Whenever there is a change in mst topology
/// DSC configuration would have to be recalculated
/// therefore we need to trigger modeset on all affected
/// CRTCs in that topology
///
/// See also:
/// drm_dp_mst_atomic_enable_dsc()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_add_affected_dsc_crtcs")]
extern "C" fn drm_dp_mst_add_affected_dsc_crtcs(state: *mut drm_atomic_commit, mgr: *mut drm_dp_mst_topology_mgr) -> i32 {
let mut mst_state: *mut drm_dp_mst_topology_state = unsafe { core::mem::zeroed() };
let mut pos: *mut drm_dp_mst_atomic_payload = unsafe { core::mem::zeroed() };
mst_state = drm_atomic_get_mst_topology_state(state, mgr);
if IS_ERR(mst_state as *const ()) {
return PTR_ERR(mst_state as *const ()) as i32;
}
pos = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*mst_state).payloads) }, next);
while list_entry_is_head!(pos, unsafe { core::ptr::addr_of_mut!((*mst_state).payloads) }, next) == 0 {
let connector: *mut drm_connector = unsafe { (*(*pos).port).connector };
if connector.is_null() {
return 22i32.wrapping_neg();
}
let conn_state: *mut drm_connector_state = unsafe { drm_atomic_get_connector_state(state, connector) };
if IS_ERR(conn_state as *const ()) {
return PTR_ERR(conn_state as *const ()) as i32;
}
let crtc: *mut drm_crtc = unsafe { (*conn_state).crtc };
if !crtc.is_null() {
if !drm_dp_mst_dsc_aux_for_port(unsafe { (*pos).port }).is_null() {
let crtc_state: *mut drm_crtc_state = unsafe { drm_atomic_get_crtc_state((*mst_state).base.state, crtc) };
if IS_ERR(crtc_state as *const ()) {
return PTR_ERR(crtc_state as *const ()) as i32;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 4u32, c"[MST MGR:%p] Setting mode_changed flag on CRTC %p\n".as_ptr() as *const u8, mgr, crtc) };
unsafe {
let __t0 = true;
(*crtc_state).__slate_bits_0.__set_mode_changed(__t0);
}
}
}
pos = list_next_entry__2!(pos, next);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_add_affected_dsc_crtcs_762: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit, *mut drm_dp_mst_topology_mgr) -> i32>, *mut ()>(Some(drm_dp_mst_add_affected_dsc_crtcs)) };

/// drm_dp_mst_atomic_enable_dsc - Set DSC Enable Flag to On/Off
/// @state: Pointer to the new drm_atomic_commit
/// @port: Pointer to the affected MST Port
/// @pbn: Newly recalculated bw required for link with DSC enabled
/// @enable: Boolean flag to enable or disable DSC on the port
///
/// This function enables DSC on the given Port
/// by recalculating its vcpi from pbn provided
/// and sets dsc_enable flag to keep track of which
/// ports have DSC enabled
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_atomic_enable_dsc")]
extern "C" fn drm_dp_mst_atomic_enable_dsc(state: *mut drm_atomic_commit, port: *mut drm_dp_mst_port, pbn: i32, enable: bool) -> i32 {
let mut time_slots: i32 = 0;
let mst_state: *mut drm_dp_mst_topology_state = drm_atomic_get_mst_topology_state(state, unsafe { (*port).mgr });
if IS_ERR(mst_state as *const ()) {
return PTR_ERR(mst_state as *const ()) as i32;
}
let payload: *mut drm_dp_mst_atomic_payload = drm_atomic_get_mst_payload_state(mst_state, port);
if payload.is_null() {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*state).dev as *const drm_device) as *const device, 4u32, c"[MST PORT:%p] Couldn't find payload in mst state %p\n".as_ptr() as *const u8, port, mst_state) };
return 22i32.wrapping_neg();
}
if ((unsafe { (*payload).__slate_bits_0.__get_dsc_enabled() }) as i32) == (enable as i32) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*state).dev as *const drm_device) as *const device, 4u32, c"[MST PORT:%p] DSC flag is already set to %d, returning %d time slots\n".as_ptr() as *const u8, port, enable as i32, (*payload).time_slots) };
time_slots = unsafe { (*payload).time_slots };
}
if enable {
time_slots = drm_dp_atomic_find_time_slots(state, unsafe { (*port).mgr }, port, pbn);
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*state).dev as *const drm_device) as *const device, 4u32, c"[MST PORT:%p] Enabling DSC flag, reallocating %d time slots on the port\n".as_ptr() as *const u8, port, time_slots) };
if time_slots < 0 {
return 22i32.wrapping_neg();
}
}
unsafe {
(*payload).__slate_bits_0.__set_dsc_enabled(enable);
}
time_slots
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_atomic_enable_dsc_763: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit, *mut drm_dp_mst_port, i32, bool) -> i32>, *mut ()>(Some(drm_dp_mst_atomic_enable_dsc)) };

/// drm_dp_mst_atomic_check_mgr - Check the atomic state of an MST topology manager
/// @state: The global atomic state
/// @mgr: Manager to check
/// @mst_state: The MST atomic state for @mgr
/// @failing_port: Returns the port with a BW limitation
///
/// Checks the given MST manager's topology state for an atomic update to ensure
/// that it's valid. This includes checking whether there's enough bandwidth to
/// support the new timeslot allocations in the atomic update.
///
/// Any atomic drivers supporting DP MST must make sure to call this or
/// the drm_dp_mst_atomic_check() function after checking the rest of their state
/// in their &drm_mode_config_funcs.atomic_check() callback.
///
/// See also:
/// drm_dp_mst_atomic_check()
/// drm_dp_atomic_find_time_slots()
/// drm_dp_atomic_release_time_slots()
///
/// Returns:
///   - 0 if the new state is valid
///   - %-ENOSPC, if the new state is invalid, because of BW limitation
///         @failing_port is set to:
///
///         - The non-root port where a BW limit check failed
///           with all the ports downstream of @failing_port passing
///           the BW limit check.
///           The returned port pointer is valid until at least
///           one payload downstream of it exists.
///         - %NULL if the BW limit check failed at the root port
///           with all the ports downstream of the root port passing
///           the BW limit check.
///
///   - %-EINVAL, if the new state is invalid, because the root port has
///     too many payloads.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_atomic_check_mgr")]
extern "C" fn drm_dp_mst_atomic_check_mgr(state: *mut drm_atomic_commit, mgr: *mut drm_dp_mst_topology_mgr, mst_state: *mut drm_dp_mst_topology_state, failing_port: *mut *mut drm_dp_mst_port) -> i32 {
let mut ret: i32 = 0;
unsafe {
*failing_port = core::ptr::null_mut::<drm_dp_mst_port>();
}
if ((unsafe { (*mgr).__slate_bits_0.__get_mst_state() }) as i32) == 0 {
return 0;
}
unsafe { mutex_lock(core::ptr::addr_of_mut!((*mgr).lock)) };
ret = drm_dp_mst_atomic_check_mstb_bw_limit(unsafe { (*mgr).mst_primary }, mst_state, failing_port);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*mgr).lock)) };
if ret < 0 {
return ret;
}
drm_dp_mst_atomic_check_payload_alloc_limits(mgr, mst_state)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_atomic_check_mgr_764: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit, *mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_topology_state, *mut *mut drm_dp_mst_port) -> i32>, *mut ()>(Some(drm_dp_mst_atomic_check_mgr)) };

/// drm_dp_mst_atomic_check - Check that the new state of an MST topology in an
/// atomic update is valid
/// @state: Pointer to the new &struct drm_dp_mst_topology_state
///
/// Checks the given topology state for an atomic update to ensure that it's
/// valid, calling drm_dp_mst_atomic_check_mgr() for all MST manager in the
/// atomic state. This includes checking whether there's enough bandwidth to
/// support the new timeslot allocations in the atomic update.
///
/// Any atomic drivers supporting DP MST must make sure to call this after
/// checking the rest of their state in their
/// &drm_mode_config_funcs.atomic_check() callback.
///
/// See also:
/// drm_dp_mst_atomic_check_mgr()
/// drm_dp_atomic_find_time_slots()
/// drm_dp_atomic_release_time_slots()
///
/// Returns:
/// 0 if the new state is valid, negative error code otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_atomic_check")]
extern "C" fn drm_dp_mst_atomic_check(state: *mut drm_atomic_commit) -> i32 {
let mut mgr: *mut drm_dp_mst_topology_mgr = unsafe { core::mem::zeroed() };
let mut mst_state: *mut drm_dp_mst_topology_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut ret: i32 = 0;
while i < unsafe { (*state).num_private_objs } {
if __drm_dp_mst_state_iter_get(state, unsafe { &mut (*core::ptr::addr_of_mut!(mgr)) }, core::ptr::null_mut::<*mut drm_dp_mst_topology_state>(), core::ptr::addr_of_mut!(mst_state), i) {
let mut tmp_port: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
ret = drm_dp_mst_atomic_check_mgr(state, mgr, mst_state, core::ptr::addr_of_mut!(tmp_port));
if ret != 0 {
break;
}
}
i = i.wrapping_add(1);
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_atomic_check_765: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit) -> i32>, *mut ()>(Some(drm_dp_mst_atomic_check)) };

#[unsafe(no_mangle)]
static drm_dp_mst_topology_state_funcs: __SlateConst<drm_private_state_funcs> = __SlateConst({
let mut __t0: drm_private_state_funcs = unsafe { core::mem::zeroed() };
__t0.atomic_create_state = Some(drm_dp_mst_atomic_create_state);
__t0.atomic_duplicate_state = Some(drm_dp_mst_duplicate_state);
__t0.atomic_destroy_state = Some(drm_dp_mst_destroy_state);
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_topology_state_funcs_766: *mut () = (unsafe { core::ptr::addr_of!(drm_dp_mst_topology_state_funcs.0) }) as *mut ();

/// drm_atomic_get_mst_topology_state: get MST topology state
/// @state: global atomic state
/// @mgr: MST topology manager, also the private object in this case
///
/// This function wraps drm_atomic_get_priv_obj_state() passing in the MST atomic
/// state vtable so that the private object state returned is that of a MST
/// topology object.
///
/// RETURNS:
/// The MST topology state or error pointer.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_atomic_get_mst_topology_state")]
extern "C" fn drm_atomic_get_mst_topology_state(state: *mut drm_atomic_commit, mgr: *mut drm_dp_mst_topology_mgr) -> *mut drm_dp_mst_topology_state {
to_dp_mst_topology_state!(unsafe { drm_atomic_get_private_obj_state(state, core::ptr::addr_of_mut!((*mgr).base)) })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_get_mst_topology_state_767: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit, *mut drm_dp_mst_topology_mgr) -> *mut drm_dp_mst_topology_state>, *mut ()>(Some(drm_atomic_get_mst_topology_state)) };

/// drm_atomic_get_old_mst_topology_state: get old MST topology state in atomic state, if any
/// @state: global atomic state
/// @mgr: MST topology manager, also the private object in this case
///
/// This function wraps drm_atomic_get_old_private_obj_state() passing in the MST atomic
/// state vtable so that the private object state returned is that of a MST
/// topology object.
///
/// Returns:
/// The old MST topology state, or NULL if there's no topology state for this MST mgr
/// in the global atomic state
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_atomic_get_old_mst_topology_state")]
extern "C" fn drm_atomic_get_old_mst_topology_state(state: *mut drm_atomic_commit, mgr: *mut drm_dp_mst_topology_mgr) -> *mut drm_dp_mst_topology_state {
let old_priv_state: *mut drm_private_state = unsafe { drm_atomic_get_old_private_obj_state(state as *const drm_atomic_commit, core::ptr::addr_of_mut!((*mgr).base)) };
let __v0: *mut drm_dp_mst_topology_state = if !old_priv_state.is_null() {
to_dp_mst_topology_state!(old_priv_state)
} else {
core::ptr::null_mut::<drm_dp_mst_topology_state>()
};
__v0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_get_old_mst_topology_state_768: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit, *mut drm_dp_mst_topology_mgr) -> *mut drm_dp_mst_topology_state>, *mut ()>(Some(drm_atomic_get_old_mst_topology_state)) };

/// drm_atomic_get_new_mst_topology_state: get new MST topology state in atomic state, if any
/// @state: global atomic state
/// @mgr: MST topology manager, also the private object in this case
///
/// This function wraps drm_atomic_get_new_private_obj_state() passing in the MST atomic
/// state vtable so that the private object state returned is that of a MST
/// topology object.
///
/// Returns:
/// The new MST topology state, or NULL if there's no topology state for this MST mgr
/// in the global atomic state
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_atomic_get_new_mst_topology_state")]
extern "C" fn drm_atomic_get_new_mst_topology_state(state: *mut drm_atomic_commit, mgr: *mut drm_dp_mst_topology_mgr) -> *mut drm_dp_mst_topology_state {
let new_priv_state: *mut drm_private_state = unsafe { drm_atomic_get_new_private_obj_state(state as *const drm_atomic_commit, core::ptr::addr_of_mut!((*mgr).base)) };
let __v0: *mut drm_dp_mst_topology_state = if !new_priv_state.is_null() {
to_dp_mst_topology_state!(new_priv_state)
} else {
core::ptr::null_mut::<drm_dp_mst_topology_state>()
};
__v0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_get_new_mst_topology_state_769: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit, *mut drm_dp_mst_topology_mgr) -> *mut drm_dp_mst_topology_state>, *mut ()>(Some(drm_atomic_get_new_mst_topology_state)) };

/// drm_dp_mst_topology_mgr_init - initialise a topology manager
/// @mgr: manager struct to initialise
/// @dev: device providing this structure - for i2c addition.
/// @aux: DP helper aux channel to talk to this device
/// @max_dpcd_transaction_bytes: hw specific DPCD transaction limit
/// @max_payloads: maximum number of payloads this GPU can source
/// @conn_base_id: the connector object ID the MST device is connected to.
///
/// Return 0 for success, or negative error code on failure
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_topology_mgr_init")]
extern "C" fn drm_dp_mst_topology_mgr_init(mgr: *mut drm_dp_mst_topology_mgr, dev: *mut drm_device, aux: *mut drm_dp_aux, max_dpcd_transaction_bytes: i32, max_payloads: i32, conn_base_id: i32) -> i32 {
let mut __slate_compound_8: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_14: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_17: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
loop {
__mutex_init(unsafe { core::ptr::addr_of_mut!((*mgr).lock) }, c"&mgr->lock".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key__5738) });
if true {
break;
}
}
loop {
__mutex_init(unsafe { core::ptr::addr_of_mut!((*mgr).qlock) }, c"&mgr->qlock".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key__5739) });
if true {
break;
}
}
loop {
__mutex_init(unsafe { core::ptr::addr_of_mut!((*mgr).delayed_destroy_lock) }, c"&mgr->delayed_destroy_lock".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key__5740) });
if true {
break;
}
}
loop {
__mutex_init(unsafe { core::ptr::addr_of_mut!((*mgr).up_req_lock) }, c"&mgr->up_req_lock".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key__5741) });
if true {
break;
}
}
loop {
__mutex_init(unsafe { core::ptr::addr_of_mut!((*mgr).probe_lock) }, c"&mgr->probe_lock".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key__5742) });
if true {
break;
}
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*mgr).tx_msg_downq) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*mgr).destroy_port_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*mgr).destroy_branch_device_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*mgr).up_req_list) });
// delayed_destroy_work will be queued on a dedicated WQ, so that any
// requeuing will be also flushed when deiniting the topology manager.
let mut __v5: *mut workqueue_struct;
let mut __v6: *mut workqueue_struct;
let mut _res: *mut workqueue_struct = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = unsafe { alloc_workqueue_noprof(c"drm_dp_mst_wq".as_ptr() as *const u8, 131074u32, 1i32) };
alloc_tag_restore!();
} else {
_res = unsafe { alloc_workqueue_noprof(c"drm_dp_mst_wq".as_ptr() as *const u8, 131074u32, 1i32) };
}
__v6 = _res;
__v5 = __v6;
unsafe {
(*mgr).delayed_destroy_wq = __v5;
}
if unsafe { (*mgr).delayed_destroy_wq }.is_null() {
return 12i32.wrapping_neg();
}
'__slate_break_10: loop {
'__slate_break_9: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*mgr).work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*mgr).work)).data = *{
__slate_compound_8 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_8)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*mgr).work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*mgr).work)).func = Some(drm_dp_mst_link_probe_work);
}
if true {
break;
}
}
if true {
break;
}
}
'__slate_break_13: loop {
'__slate_break_12: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*mgr).tx_work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*mgr).tx_work)).data = *{
__slate_compound_11 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_11)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*mgr).tx_work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*mgr).tx_work)).func = Some(drm_dp_tx_work);
}
if true {
break;
}
}
if true {
break;
}
}
'__slate_break_16: loop {
'__slate_break_15: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*mgr).delayed_destroy_work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*mgr).delayed_destroy_work)).data = *{
__slate_compound_14 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_14)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*mgr).delayed_destroy_work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*mgr).delayed_destroy_work)).func = Some(drm_dp_delayed_destroy_work);
}
if true {
break;
}
}
if true {
break;
}
}
'__slate_break_19: loop {
'__slate_break_18: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*mgr).up_req_work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*mgr).up_req_work)).data = *{
__slate_compound_17 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_17)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*mgr).up_req_work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*mgr).up_req_work)).func = Some(drm_dp_mst_up_req_work);
}
if true {
break;
}
}
if true {
break;
}
}
loop {
unsafe { __init_waitqueue_head(core::ptr::addr_of_mut!((*mgr).tx_waitq), c"&mgr->tx_waitq".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__5764)) };
if true {
break;
}
}
unsafe {
(*mgr).dev = dev;
}
unsafe {
(*mgr).aux = aux;
}
unsafe {
(*mgr).max_dpcd_transaction_bytes = max_dpcd_transaction_bytes;
}
unsafe {
(*mgr).max_payloads = max_payloads;
}
unsafe {
(*mgr).conn_base_id = conn_base_id;
}
unsafe { drm_atomic_private_obj_init(dev, core::ptr::addr_of_mut!((*mgr).base), core::ptr::addr_of!(drm_dp_mst_topology_state_funcs.0)) };
0
}

static mut __key__5738: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__5739: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__5740: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__5741: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__5742: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__5760: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__5761: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__5762: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__5763: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__5764: lock_class_key = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_topology_mgr_init_770: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr, *mut drm_device, *mut drm_dp_aux, i32, i32, i32) -> i32>, *mut ()>(Some(drm_dp_mst_topology_mgr_init)) };

/// drm_dp_mst_topology_mgr_destroy() - destroy topology manager.
/// @mgr: manager to destroy
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_topology_mgr_destroy")]
extern "C" fn drm_dp_mst_topology_mgr_destroy(mgr: *mut drm_dp_mst_topology_mgr) {
drm_dp_mst_topology_mgr_set_mst(mgr, false);
unsafe { flush_work(core::ptr::addr_of_mut!((*mgr).work)) };
// The following will also drain any requeued work on the WQ.
if !unsafe { (*mgr).delayed_destroy_wq }.is_null() {
unsafe { destroy_workqueue((*mgr).delayed_destroy_wq) };
unsafe {
(*mgr).delayed_destroy_wq = core::ptr::null_mut::<workqueue_struct>();
}
}
unsafe {
(*mgr).dev = core::ptr::null_mut::<drm_device>();
}
unsafe {
(*mgr).aux = core::ptr::null_mut::<drm_dp_aux>();
}
unsafe { drm_atomic_private_obj_fini(core::ptr::addr_of_mut!((*mgr).base)) };
unsafe {
(*mgr).funcs = core::ptr::null::<drm_private_state_funcs>();
}
mutex_destroy(unsafe { core::ptr::addr_of_mut!((*mgr).delayed_destroy_lock) });
mutex_destroy(unsafe { core::ptr::addr_of_mut!((*mgr).qlock) });
mutex_destroy(unsafe { core::ptr::addr_of_mut!((*mgr).lock) });
mutex_destroy(unsafe { core::ptr::addr_of_mut!((*mgr).up_req_lock) });
mutex_destroy(unsafe { core::ptr::addr_of_mut!((*mgr).probe_lock) });
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_topology_mgr_destroy_771: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr)>, *mut ()>(Some(drm_dp_mst_topology_mgr_destroy)) };

fn remote_i2c_read_ok(msgs: *const i2c_msg, num: i32) -> bool {
let mut i: i32 = 0;
if num.wrapping_sub(1) > 4 {
return false;
}
while i < num.wrapping_sub(1) {
if ((unsafe { (*msgs.wrapping_offset(i as isize)).flags }) as i32) & 1 != 0 || ((unsafe { (*msgs.wrapping_offset(i as isize)).len }) as i32) > 255 {
return false;
}
i = i.wrapping_add(1);
}
((unsafe { (*msgs.wrapping_offset(num.wrapping_sub(1) as isize)).flags }) as i32) & 1 != 0 && ((unsafe { (*msgs.wrapping_offset(num.wrapping_sub(1) as isize)).len }) as i32) <= 255
}

fn remote_i2c_write_ok(msgs: *const i2c_msg, num: i32) -> bool {
let mut i: i32 = 0;
while i < num.wrapping_sub(1) {
if ((unsafe { (*msgs.wrapping_offset(i as isize)).flags }) as i32) & 1 != 0 || ((unsafe { (*msgs.wrapping_offset(i as isize)).flags }) as i32) & 32768 == 0 || ((unsafe { (*msgs.wrapping_offset(i as isize)).len }) as i32) > 255 {
return false;
}
i = i.wrapping_add(1);
}
((unsafe { (*msgs.wrapping_offset(num.wrapping_sub(1) as isize)).flags }) as i32) & 1 == 0 && ((unsafe { (*msgs.wrapping_offset(num.wrapping_sub(1) as isize)).len }) as i32) <= 255
}

fn drm_dp_mst_i2c_read(mstb: *mut drm_dp_mst_branch, port: *mut drm_dp_mst_port, msgs: *mut i2c_msg, num: i32) -> i32 {
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_15: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut txmsg: *mut drm_dp_sideband_msg_tx = unsafe { core::mem::zeroed() };
let mut msg: core::mem::MaybeUninit<drm_dp_sideband_msg_req_body> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut drm_dp_sideband_msg_req_body = core::ptr::addr_of_mut!(msg) as *mut drm_dp_sideband_msg_req_body;
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
let mut mgr: *mut drm_dp_mst_topology_mgr = unsafe { core::mem::zeroed() };
unsafe {
mgr = (*port).mgr;
let __slate_auto_bytes_2 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 120usize);
txmsg = core::ptr::null_mut::<drm_dp_sideband_msg_tx>();
let __slate_auto_bytes_4 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
memset(core::ptr::addr_of_mut!(*__slate_slot_10) as *mut (), 0i32, core::mem::size_of::<drm_dp_sideband_msg_req_body>() as u64);
(*__slate_slot_10).req_type = 34;
(*__slate_slot_10).u.i2c_read.num_transactions = num.wrapping_sub(1) as u8;
(*__slate_slot_10).u.i2c_read.port_number = (*port).port_num;
*__slate_slot_11 = 0;
while *__slate_slot_11 < (num.wrapping_sub(1) as u32) {
(*__slate_slot_10).u.i2c_read.transactions[*__slate_slot_11 as usize].i2c_dev_id = (*msgs.wrapping_offset(*__slate_slot_11 as isize)).addr as u8;
(*__slate_slot_10).u.i2c_read.transactions[*__slate_slot_11 as usize].num_bytes = (*msgs.wrapping_offset(*__slate_slot_11 as isize)).len as u8;
(*__slate_slot_10).u.i2c_read.transactions[*__slate_slot_11 as usize].bytes = (*msgs.wrapping_offset(*__slate_slot_11 as isize)).buf;
(*__slate_slot_10).u.i2c_read.transactions[*__slate_slot_11 as usize].no_stop_bit = (((*msgs.wrapping_offset(*__slate_slot_11 as isize)).flags as i32) & 32768 == 0) as u8;
let __v7: u32 = *__slate_slot_11;
*__slate_slot_11 = __v7.wrapping_add(1);
}
(*__slate_slot_10).u.i2c_read.read_i2c_device_id = (*msgs.wrapping_offset(num.wrapping_sub(1) as isize)).addr as u8;
(*__slate_slot_10).u.i2c_read.num_bytes_read = (*msgs.wrapping_offset(num.wrapping_sub(1) as isize)).len as u8;
__obj_size = size_mul(core::mem::size_of::<drm_dp_sideband_msg_tx>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_15 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_15)
});
}
txmsg = *__slate_slot_1 as *mut drm_dp_sideband_msg_tx;
if txmsg.is_null() {
*__slate_slot_8 = 12i32.wrapping_neg();
} else {
(*txmsg).dst = mstb;
drm_dp_encode_sideband_req(core::ptr::addr_of_mut!(*__slate_slot_10) as *const drm_dp_sideband_msg_req_body, txmsg);
drm_dp_queue_down_tx(mgr, txmsg);
*__slate_slot_8 = drm_dp_mst_wait_tx_reply(mstb, txmsg);
if *__slate_slot_8 > 0 {
if ((*txmsg).reply.reply_type as i32) == 1 {
*__slate_slot_8 = 121i32.wrapping_neg();
} else {
if ((*txmsg).reply.u.remote_i2c_read_ack.num_bytes as i32) != ((*msgs.wrapping_offset(num.wrapping_sub(1) as isize)).len as i32) {
*__slate_slot_8 = 5i32.wrapping_neg();
} else {
memcpy((*msgs.wrapping_offset(num.wrapping_sub(1) as isize)).buf as *mut (), ((*txmsg).reply.u.remote_i2c_read_ack.bytes.as_mut_ptr() as *mut u8) as *const (), (*msgs.wrapping_offset(num.wrapping_sub(1) as isize)).len as u64);
*__slate_slot_8 = num;
}
}
}
}
kfree(txmsg as *const ());
return *__slate_slot_8;
}
}

fn drm_dp_mst_i2c_write(mstb: *mut drm_dp_mst_branch, port: *mut drm_dp_mst_port, msgs: *mut i2c_msg, num: i32) -> i32 {
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_15: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut txmsg: *mut drm_dp_sideband_msg_tx = unsafe { core::mem::zeroed() };
let mut msg: core::mem::MaybeUninit<drm_dp_sideband_msg_req_body> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut drm_dp_sideband_msg_req_body = core::ptr::addr_of_mut!(msg) as *mut drm_dp_sideband_msg_req_body;
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
let mut mgr: *mut drm_dp_mst_topology_mgr = unsafe { core::mem::zeroed() };
unsafe {
mgr = (*port).mgr;
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 120usize);
txmsg = core::ptr::null_mut::<drm_dp_sideband_msg_tx>();
let __slate_auto_bytes_3 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
__obj_size = size_mul(core::mem::size_of::<drm_dp_sideband_msg_tx>() as u64, 1u64);
let __slate_auto_bytes_4 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_2 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_3 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
});
alloc_tag_restore!();
} else {
*__slate_slot_3 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_15 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_15)
});
}
'__join_0: {
txmsg = *__slate_slot_3 as *mut drm_dp_sideband_msg_tx;
if txmsg.is_null() {
*__slate_slot_8 = 12i32.wrapping_neg();
} else {
*__slate_slot_11 = 0;
'__join_4: {
while *__slate_slot_11 < (num as u32) {
memset(core::ptr::addr_of_mut!(*__slate_slot_10) as *mut (), 0i32, core::mem::size_of::<drm_dp_sideband_msg_req_body>() as u64);
(*__slate_slot_10).req_type = 35;
(*__slate_slot_10).u.i2c_write.port_number = (*port).port_num;
(*__slate_slot_10).u.i2c_write.write_i2c_device_id = (*msgs.wrapping_offset(*__slate_slot_11 as isize)).addr as u8;
(*__slate_slot_10).u.i2c_write.num_bytes = (*msgs.wrapping_offset(*__slate_slot_11 as isize)).len as u8;
(*__slate_slot_10).u.i2c_write.bytes = (*msgs.wrapping_offset(*__slate_slot_11 as isize)).buf;
memset(txmsg as *mut (), 0i32, core::mem::size_of::<drm_dp_sideband_msg_tx>() as u64);
(*txmsg).dst = mstb;
drm_dp_encode_sideband_req(core::ptr::addr_of_mut!(*__slate_slot_10) as *const drm_dp_sideband_msg_req_body, txmsg);
drm_dp_queue_down_tx(mgr, txmsg);
*__slate_slot_8 = drm_dp_mst_wait_tx_reply(mstb, txmsg);
if *__slate_slot_8 > 0 {
if ((*txmsg).reply.reply_type as i32) == 1 {
break '__join_4;
} else {
let __v1: u32 = *__slate_slot_11;
*__slate_slot_11 = __v1.wrapping_add(1);
}
} else {
break '__join_0;
}
}
*__slate_slot_8 = num;
break '__join_0;
}
*__slate_slot_8 = 121i32.wrapping_neg();
}
}
kfree(txmsg as *const ());
return *__slate_slot_8;
}
}

/// I2C device
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_i2c_xfer")]
extern "C" fn drm_dp_mst_i2c_xfer(adapter: *mut i2c_adapter, msgs: *mut i2c_msg, num: i32) -> i32 {
let aux: *mut drm_dp_aux = (unsafe { (*adapter).algo_data }) as *mut drm_dp_aux;
let port: *mut drm_dp_mst_port = container_of__3!(aux, drm_dp_mst_port);
let mgr: *mut drm_dp_mst_topology_mgr = unsafe { (*port).mgr };
let mut ret: i32 = 0;
let mstb: *mut drm_dp_mst_branch = drm_dp_mst_topology_get_mstb_validated(mgr, unsafe { (*port).parent });
if mstb.is_null() {
return 121i32.wrapping_neg();
}
if remote_i2c_read_ok(msgs as *const i2c_msg, num) {
ret = drm_dp_mst_i2c_read(mstb, port, msgs, num);
} else {
if remote_i2c_write_ok(msgs as *const i2c_msg, num) {
ret = drm_dp_mst_i2c_write(mstb, port, msgs, num);
} else {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*mgr).dev as *const drm_device) as *const device, 2u32, c"Unsupported I2C transaction for MST device\n".as_ptr() as *const u8) };
ret = 5i32.wrapping_neg();
}
}
drm_dp_mst_topology_put_mstb(mstb);
ret
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_i2c_functionality")]
extern "C" fn drm_dp_mst_i2c_functionality(adapter: *mut i2c_adapter) -> u32 {
(1 | (65536 | (131072 | 262144) | (524288 | 1048576) | (2097152 | 4194304) | 8388608 | 33554432 | (67108864 | 134217728) | 8) | 16777216 | 32768 | 2) as u32
}

static drm_dp_mst_i2c_algo: __SlateConst<i2c_algorithm> = __SlateConst({
let mut __t0: i2c_algorithm = unsafe { core::mem::zeroed() };
__t0.__slate_anon_0 = __SlateRecord_i2c_h_551 { master_xfer: Some(drm_dp_mst_i2c_xfer) };
__t0.functionality = Some(drm_dp_mst_i2c_functionality);
__t0
});

/// drm_dp_mst_register_i2c_bus() - register an I2C adapter for I2C-over-AUX
/// @port: The port to add the I2C bus on
///
/// Returns 0 on success or a negative error code on failure.
fn drm_dp_mst_register_i2c_bus(port: *mut drm_dp_mst_port) -> i32 {
let aux: *mut drm_dp_aux = unsafe { core::ptr::addr_of_mut!((*port).aux) };
let parent_dev: *mut device = unsafe { (*(*(*port).mgr).dev).dev };
unsafe {
(*aux).ddc.algo = core::ptr::addr_of!(drm_dp_mst_i2c_algo.0);
}
unsafe {
(*aux).ddc.algo_data = aux as *mut ();
}
unsafe {
(*aux).ddc.retries = 3;
}
unsafe {
(*aux).ddc.owner = core::ptr::null_mut::<module>();
}
// FIXME: set the kdev of the port's connector as parent
unsafe {
(*aux).ddc.dev.parent = parent_dev;
}
unsafe {
(*aux).ddc.dev.of_node = (*parent_dev).of_node;
}
let __v0: *const u8 = if !unsafe { (*aux).name }.is_null() {
unsafe { (*aux).name }
} else {
dev_name(parent_dev as *const device)
};
unsafe { sized_strscpy((*aux).ddc.name.as_mut_ptr() as *mut u8, __v0, (core::mem::size_of::<[u8; 48]>() as u64).wrapping_add(__must_be_cstr__6!() as u64).wrapping_add(__must_be_cstr__2!() as u64)) };
unsafe { i2c_add_adapter(core::ptr::addr_of_mut!((*aux).ddc)) }
}

/// drm_dp_mst_unregister_i2c_bus() - unregister an I2C-over-AUX adapter
/// @port: The port to remove the I2C bus from
fn drm_dp_mst_unregister_i2c_bus(port: *mut drm_dp_mst_port) {
unsafe { i2c_del_adapter(core::ptr::addr_of_mut!((*port).aux.ddc)) };
}

/// drm_dp_mst_is_virtual_dpcd() - Is the given port a virtual DP Peer Device
/// @port: The port to check
///
/// A single physical MST hub object can be represented in the topology
/// by multiple branches, with virtual ports between those branches.
///
/// As of DP1.4, An MST hub with internal (virtual) ports must expose
/// certain DPCD registers over those ports. See sections 2.6.1.1.1
/// and 2.6.1.1.2 of Display Port specification v1.4 for details.
///
/// May acquire mgr->lock
///
/// Returns:
/// true if the port is a virtual DP peer device, false otherwise
fn drm_dp_mst_is_virtual_dpcd(port: *mut drm_dp_mst_port) -> bool {
let mut downstream_port: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
if port.is_null() || ((unsafe { (*port).dpcd_rev }) as i32) < 20 {
return false;
}
// Virtual DP Sink (Internal Display Panel)
if drm_dp_mst_port_is_logical(port) {
return true;
}
// DP-to-HDMI Protocol Converter
if ((unsafe { (*port).pdt }) as i32) == 4 && !unsafe { (*port).mcs } && unsafe { (*port).ldps } {
return true;
}
// DP-to-DP
unsafe { mutex_lock(core::ptr::addr_of_mut!((*(*port).mgr).lock)) };
if ((unsafe { (*port).pdt }) as i32) == 2 && !unsafe { (*port).mstb }.is_null() && (unsafe { (*(*port).mstb).num_ports }) == 2 {
downstream_port = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*(*port).mstb).ports) }, next);
while list_entry_is_head!(downstream_port, unsafe { core::ptr::addr_of_mut!((*(*port).mstb).ports) }, next) == 0 {
if ((unsafe { (*downstream_port).pdt }) as i32) == 3 && !unsafe { (*downstream_port).input } {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*(*port).mgr).lock)) };
return true;
}
downstream_port = list_next_entry!(downstream_port, next);
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*(*port).mgr).lock)) };
false
}

/// drm_dp_mst_aux_for_parent() - Get the AUX device for an MST port's parent
/// @port: MST port whose parent's AUX device is returned
///
/// Return the AUX device for @port's parent or NULL if port's parent is the
/// root port.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_aux_for_parent")]
extern "C" fn drm_dp_mst_aux_for_parent(port: *mut drm_dp_mst_port) -> *mut drm_dp_aux {
if unsafe { (*port).parent }.is_null() || unsafe { (*(*port).parent).port_parent }.is_null() {
return core::ptr::null_mut::<drm_dp_aux>();
}
unsafe { core::ptr::addr_of_mut!((*(*(*port).parent).port_parent).aux) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_aux_for_parent_772: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_port) -> *mut drm_dp_aux>, *mut ()>(Some(drm_dp_mst_aux_for_parent)) };

/// drm_dp_mst_dsc_aux_for_port() - Find the correct aux for DSC
/// @port: The port to check. A leaf of the MST tree with an attached display.
///
/// Depending on the situation, DSC may be enabled via the endpoint aux,
/// the immediately upstream aux, or the connector's physical aux.
///
/// This is both the correct aux to read DSC_CAPABILITY and the
/// correct aux to write DSC_ENABLED.
///
/// This operation can be expensive (up to four aux reads), so
/// the caller should cache the return.
///
/// Returns:
/// NULL if DSC cannot be enabled on this port, otherwise the aux device
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_mst_topology.drm_dp_mst_dsc_aux_for_port")]
extern "C" fn drm_dp_mst_dsc_aux_for_port(port: *mut drm_dp_mst_port) -> *mut drm_dp_aux {
let mut immediate_upstream_port: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
let mut immediate_upstream_aux: *mut drm_dp_aux = unsafe { core::mem::zeroed() };
let mut fec_port: *mut drm_dp_mst_port = unsafe { core::mem::zeroed() };
let mut desc: drm_dp_desc = {
let mut __t0: drm_dp_desc = unsafe { core::mem::zeroed() };
__t0
};
let mut upstream_dsc: u8 = 0;
let mut endpoint_fec: u8 = 0;
let mut endpoint_dsc: u8 = 0;
if port.is_null() {
return core::ptr::null_mut::<drm_dp_aux>();
}
if !unsafe { (*(*port).parent).port_parent }.is_null() {
immediate_upstream_port = unsafe { (*(*port).parent).port_parent };
} else {
immediate_upstream_port = core::ptr::null_mut::<drm_dp_mst_port>();
}
fec_port = immediate_upstream_port;
while !fec_port.is_null() {
// Each physical link (i.e. not a virtual port) between the
// output and the primary device must support FEC
if !drm_dp_mst_is_virtual_dpcd(fec_port) && !unsafe { (*fec_port).fec_capable } {
return core::ptr::null_mut::<drm_dp_aux>();
}
fec_port = unsafe { (*(*fec_port).parent).port_parent };
}
// DP-to-DP peer device
if drm_dp_mst_is_virtual_dpcd(immediate_upstream_port) {
if drm_dp_dpcd_read_data(unsafe { core::ptr::addr_of_mut!((*port).aux) }, 96u32, core::ptr::addr_of_mut!(endpoint_dsc) as *mut (), 1u64) < 0 {
return core::ptr::null_mut::<drm_dp_aux>();
}
if drm_dp_dpcd_read_data(unsafe { core::ptr::addr_of_mut!((*port).aux) }, 144u32, core::ptr::addr_of_mut!(endpoint_fec) as *mut (), 1u64) < 0 {
return core::ptr::null_mut::<drm_dp_aux>();
}
if drm_dp_dpcd_read_data(unsafe { core::ptr::addr_of_mut!((*immediate_upstream_port).aux) }, 96u32, core::ptr::addr_of_mut!(upstream_dsc) as *mut (), 1u64) < 0 {
return core::ptr::null_mut::<drm_dp_aux>();
}
// Enpoint decompression with DP-to-DP peer device
if (endpoint_dsc as i32) & 1i32 != 0 && (endpoint_fec as i32) & 1i32 != 0 && (upstream_dsc as i32) & 2i32 != 0 {
unsafe {
(*port).passthrough_aux = core::ptr::addr_of_mut!((*immediate_upstream_port).aux);
}
return unsafe { core::ptr::addr_of_mut!((*port).aux) };
}
// Virtual DPCD decompression with DP-to-DP peer device
return unsafe { core::ptr::addr_of_mut!((*immediate_upstream_port).aux) };
}
// Virtual DPCD decompression with DP-to-HDMI or Virtual DP Sink
if drm_dp_mst_is_virtual_dpcd(port) {
return unsafe { core::ptr::addr_of_mut!((*port).aux) };
}
// Synaptics quirk
// Applies to ports for which:
// - Physical aux has Synaptics OUI
// - DPv1.4 or higher
// - Port is on primary branch device
// - Not a VGA adapter (DP_DWN_STRM_PORT_TYPE_ANALOG)
if !immediate_upstream_port.is_null() {
immediate_upstream_aux = unsafe { core::ptr::addr_of_mut!((*immediate_upstream_port).aux) };
} else {
immediate_upstream_aux = unsafe { (*(*port).mgr).aux };
}
if (unsafe { drm_dp_read_desc(immediate_upstream_aux, core::ptr::addr_of_mut!(desc), true) }) != 0 {
return core::ptr::null_mut::<drm_dp_aux>();
}
if drm_dp_has_quirk(unsafe { &(*(core::ptr::addr_of_mut!(desc) as *const drm_dp_desc)) }, 3u32) {
let mut dpcd_ext: [u8; 15] = [0u8; 15];
if drm_dp_dpcd_read_data(immediate_upstream_aux, 96u32, core::ptr::addr_of_mut!(upstream_dsc) as *mut (), 1u64) < 0 {
return core::ptr::null_mut::<drm_dp_aux>();
}
if (upstream_dsc as i32) & 1i32 == 0 {
return core::ptr::null_mut::<drm_dp_aux>();
}
if (unsafe { drm_dp_read_dpcd_caps(immediate_upstream_aux, dpcd_ext.as_mut_ptr() as *mut u8) }) < 0 {
return core::ptr::null_mut::<drm_dp_aux>();
}
if (dpcd_ext[0] as i32) >= 20 && ((dpcd_ext[5] as i32) & 1i32 != 0 && (dpcd_ext[5] as i32) & 6 != 2i32) {
return immediate_upstream_aux;
}
}
// The check below verifies if the MST sink
// connected to the GPU is capable of DSC -
// therefore the endpoint needs to be
// both DSC and FEC capable.
if drm_dp_dpcd_read_data(unsafe { core::ptr::addr_of_mut!((*port).aux) }, 96u32, core::ptr::addr_of_mut!(endpoint_dsc) as *mut (), 1u64) < 0 {
return core::ptr::null_mut::<drm_dp_aux>();
}
if drm_dp_dpcd_read_data(unsafe { core::ptr::addr_of_mut!((*port).aux) }, 144u32, core::ptr::addr_of_mut!(endpoint_fec) as *mut (), 1u64) < 0 {
return core::ptr::null_mut::<drm_dp_aux>();
}
if (endpoint_dsc as i32) & 1i32 != 0 && (endpoint_fec as i32) & 1i32 != 0 {
return unsafe { core::ptr::addr_of_mut!((*port).aux) };
}
core::ptr::null_mut::<drm_dp_aux>()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_mst_dsc_aux_for_port_773: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_mst_port) -> *mut drm_dp_aux>, *mut ()>(Some(drm_dp_mst_dsc_aux_for_port)) };

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_drm_dp_atomic_find_time_slots_7: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_drm_dp_atomic_find_time_slots_8: [u8; 75] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_mst_topology.c\0";

static __slate_asm_string_drm_dp_atomic_release_time_slots_6: [u8; 1] = *b"\0";

static __slate_asm_string_drm_dp_atomic_release_time_slots_7: [u8; 75] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_mst_topology.c\0";

static __slate_asm_string_drm_dp_decode_sideband_req_26: [u8; 1] = *b"\0";

static __slate_asm_string_drm_dp_decode_sideband_req_27: [u8; 75] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_mst_topology.c\0";

static __slate_asm_string_drm_dp_decode_sideband_req_6: [u8; 1] = *b"\0";

static __slate_asm_string_drm_dp_decode_sideband_req_7: [u8; 75] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_mst_topology.c\0";

static __slate_asm_string_drm_dp_mst_topology_get_mstb_1: [u8; 1] = *b"\0";

static __slate_asm_string_drm_dp_mst_topology_get_mstb_2: [u8; 75] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_mst_topology.c\0";

static __slate_asm_string_drm_dp_mst_topology_get_port_1: [u8; 1] = *b"\0";

static __slate_asm_string_drm_dp_mst_topology_get_port_2: [u8; 75] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_mst_topology.c\0";

static __slate_asm_string_drm_dp_mst_topology_mgr_set_mst_5: [u8; 1] = *b"\0";

static __slate_asm_string_drm_dp_mst_topology_mgr_set_mst_6: [u8; 75] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_mst_topology.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_process_single_down_tx_qlock_1: [u8; 1] = *b"\0";

static __slate_asm_string_process_single_down_tx_qlock_2: [u8; 75] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_mst_topology.c\0";
