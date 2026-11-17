
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
//! Copyright © 2008 Intel Corporation
//!
//! Permission is hereby granted, free of charge, to any person obtaining a
//! copy of this software and associated documentation files (the "Software"),
//! to deal in the Software without restriction, including without limitation
//! the rights to use, copy, modify, merge, publish, distribute, sublicense,
//! and/or sell copies of the Software, and to permit persons to whom the
//! Software is furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice (including the next
//! paragraph) shall be included in all copies or substantial portions of the
//! Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
//! THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//! FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//! IN THE SOFTWARE.
//!
//! Authors:
//!    Keith Packard <keithp@keithp.com>
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
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! Operation not permitted
//! Resource deadlock would occur
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
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
//! Physical address zone modifiers (see linux/mmzone.h - low four bits)
//!
//! Do not put any conditional on these. If necessary modify the definitions
//! without the underscores and use them consistently. The definitions here may
//! be used in bit comparisons.
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
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
//! for_each_set_bit_wrap - iterate over all set bits starting from @start, and
//! wrapping around the end of bitmap.
//! @bit: offset for current iteration
//! @addr: bitmap address to base the search on
//! @size: bitmap size in number of bits
//! @start: Starting bit for bitmap traversing, wrapping around the bitmap end
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
//! no-op
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
//! Generic info of form tag = "info"
//! This macro is used to tag functions as "printf-like" because
//! some compilers (like GCC) can catch printf format string problems.
//! Value returned by acpi_os_get_thread_id. There is no standard "thread_id"
//! across operating systems or even the various UNIX systems. Since ACPICA
//! only needs the thread ID as a unique thread identifier, we use a u64
//! as the only common data type - it will accommodate any type of pointer or
//! any type of integer. It is up to the host-dependent OSL to cast the
//! native thread ID type to a u64 (in acpi_os_get_thread_id).
//! For values, see Producer/Consumer above
//! Let people know that we have them
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
//! The microseconds/nanosecond delay multiplicators are used to convert a
//! constant microseconds/nanoseconds value to a value which can be used by the
//! architectures specific implementation to transform it into loops.
//! poll_timeout_us - Periodically poll and perform an operation until
//!                   a condition is met or a timeout occurs
//!
//! @op: Operation
//! @cond: Break condition
//! @sleep_us: Maximum time to sleep between operations in us (0 tight-loops).
//!            Please read usleep_range() function description for details and
//!            limitations.
//! @timeout_us: Timeout in us, 0 means never timeout
//! @sleep_before_op: if it is true, sleep @sleep_us before operation.
//!
//! When available, you'll probably want to use one of the specialized
//! macros defined below rather than this macro directly.
//!
//! Returns: 0 on success and -ETIMEDOUT upon a timeout. Must not
//! be called from atomic context if sleep_us or timeout_us are used.
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
//! VESA Display Stream Compression DSC 1.2 constants
//! Number of bits in the handle that stack depot doesn't use. Users may store
//! information in them via stack_depot_set/get_extra_bits.
//!  + ((B + 0.000122)*4096))
//! Display platforms and subplatforms. Keep platforms in display version based
//! order, chronological order within a version, and subplatforms next to the
//! platform.
//! Invoke param, a function-like macro, for each intel display param, with
//! arguments:
//!
//! param(type, name, value, mode)
//!
//! type: parameter type, one of {bool, int, unsigned int, unsigned long, char *}
//! name: name of the parameter
//! value: initial/default value of the parameter
//! mode: debugfs file permissions, one of {0400, 0600, 0}, use 0 to not create
//!       debugfs file
//! -ENOENT means we got the ref, but there's no tracking
//! including 0
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! Amount of SAGV/QGV points, BSpec precisely defines this
//! HSW and later:
//! Helpful configuration macro to be used in tables etc.
//! DRM_FORMAT_MAX_PLANES - maximum number of planes a DRM format can have
//! Following Macros take a byte at a time for bit(s) masking
//!
//! TODO: HDCP_2_2_MAX_CONTENT_STREAMS_CNT is based upon actual
//! H/W MST streams capacity.
//! This required to be moved out to platform specific header.
//! Each region is a minimum of 16k, and there are at most 255 of them.
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
// This error code is special: arch syscall entry code will return
// -ENOSYS if users try to call a syscall that doesn't exist.  To keep
// failures of syscalls that really do exist distinguishable from
// failures due to attempts to use a nonexistent syscall, syscall
// implementations should refrain from returning -ENOSYS.
// Invalid system call number
// Protocol error
// Value too large for defined data type
// Operation not supported on transport endpoint
// Memory page has hardware error
// Concatenate two parameters, but allow them to be expanded beforehand.
// Operation is not supported
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
// clamp - return a value clamped to a given range with typechecking
// @val: current value
// @lo: lowest allowable value
// @hi: highest allowable value
//
// This macro checks @val/@lo/@hi to make sure they have compatible
// signedness.
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
// Are we running in atomic context?  WARNING: this macro cannot
// always detect atomic context; in particular, it cannot know about
// held spinlocks in non-preemptible kernels.  Thus it should not be
// used in the general case to determine whether sleeping is possible.
// Do not use in_atomic() in driver code.
// Overflow Flag
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "apic" Onboard APIC
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
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
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
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
// guarantee 'op' and 'cond' are evaluated after timeout expired
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
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
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
// ADL and later:
// table size 2k - maximum due to use
// of chars for next/prev indices
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
macro_rules! ARRAY_SIZE__2 {
() => {
((core::mem::size_of::<[i32; 6]>() as u64) / (core::mem::size_of::<i32>() as u64)).wrapping_add(__must_be_array__14!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__9 {
() => {
((core::mem::size_of::<[i32; 11]>() as u64) / (core::mem::size_of::<i32>() as u64)).wrapping_add(__must_be_array__116!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__25 {
() => {
((core::mem::size_of::<[i32; 10]>() as u64) / (core::mem::size_of::<i32>() as u64)).wrapping_add(__must_be_array__58!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__30 {
() => {
((core::mem::size_of::<[i32; 4]>() as u64) / (core::mem::size_of::<i32>() as u64)).wrapping_add(__must_be_array__51!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__51 {
() => {
((core::mem::size_of::<[i32; 6]>() as u64) / (core::mem::size_of::<i32>() as u64)).wrapping_add(__must_be_array__180!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__52 {
() => {
((core::mem::size_of::<[i32; 3]>() as u64) / (core::mem::size_of::<i32>() as u64)).wrapping_add(__must_be_array__40!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__65 {
() => {
(((const { core::mem::size_of::<[i32; 8]>() }) as u64) / ((const { core::mem::size_of::<i32>() }) as u64)).wrapping_add(__must_be_array__151!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__83 {
() => {
((core::mem::size_of::<[u8; 5]>() as u64) / (core::mem::size_of::<u8>() as u64)).wrapping_add(__must_be_array__142!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__94 {
() => {
((core::mem::size_of::<[u16; 8]>() as u64) / (core::mem::size_of::<u16>() as u64)).wrapping_add(__must_be_array__6!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__97 {
() => {
((core::mem::size_of::<[i32; 3]>() as u64) / (core::mem::size_of::<i32>() as u64)).wrapping_add(__must_be_array__150!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__149 {
() => {
((core::mem::size_of::<[u8; 5]>() as u64) / (core::mem::size_of::<u8>() as u64)).wrapping_add(__must_be_array__34!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__171 {
() => {
(((const { core::mem::size_of::<[i32; 4]>() }) as u64) / ((const { core::mem::size_of::<i32>() }) as u64)).wrapping_add(__must_be_array__64!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__172 {
() => {
((core::mem::size_of::<[i32; 7]>() as u64) / (core::mem::size_of::<i32>() as u64)).wrapping_add(__must_be_array__48!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__181 {
() => {
((core::mem::size_of::<[i32; 2]>() as u64) / (core::mem::size_of::<i32>() as u64)).wrapping_add(__must_be_array__204!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__192 {
() => {
((core::mem::size_of::<[i32; 11]>() as u64) / (core::mem::size_of::<i32>() as u64)).wrapping_add(__must_be_array__166!() as u64)
};
}

macro_rules! BITS_PER_TYPE__4 {
() => {
(core::mem::size_of::<u32>() as u64).wrapping_mul(8)
};
}

macro_rules! BITS_PER_TYPE__2 {
($type:ty) => {
(core::mem::size_of::<$type>() as u64).wrapping_mul(8i32 as $type)
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1487 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_5351>() as i32
};
}

macro_rules! BIT_MASK {
($nr:expr) => {
UL!() << ($nr) % 64
};
}

macro_rules! BIT_TYPE__1387 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1487!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_U32__1120 {
($nr:expr) => {
BIT_TYPE__1387!(u32, $nr)
};
}

macro_rules! BIT_ULL {
($nr:expr) => {
ULL!() << $nr
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

macro_rules! BIT__3 {
($nr:expr) => {
UL!() << (($nr) as u32)
};
}

macro_rules! BIT__2 {
($nr:expr) => {
UL!() << (($nr) as i32)
};
}

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

macro_rules! CAN_PANEL_REPLAY {
($intel_dp:expr) => {
(unsafe { (*$intel_dp).psr.sink_panel_replay_support }) && unsafe { (*$intel_dp).psr.source_panel_replay_support }
};
}

macro_rules! DISPLAY_INFO {
($__display:expr) => {
unsafe { (*$__display).info.__device_info }
};
}

macro_rules! DISPLAY_MMIO_BASE {
($display:expr) => {
unsafe { (*DISPLAY_INFO!($display)).mmio_offset }
};
}

macro_rules! DISPLAY_RUNTIME_INFO {
($__display:expr) => {
unsafe { core::ptr::addr_of_mut!((*$__display).info.__runtime_info) }
};
}

macro_rules! DISPLAY_VER {
($__display:expr) => {
unsafe { (*DISPLAY_RUNTIME_INFO!($__display)).ip.ver }
};
}

macro_rules! DISPLAY_VERx100 {
($__display:expr) => {
((unsafe { (*DISPLAY_RUNTIME_INFO!($__display)).ip.ver }) as i32).wrapping_mul(100).wrapping_add((unsafe { (*DISPLAY_RUNTIME_INFO!($__display)).ip.rel }) as i32)
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

macro_rules! DIV_ROUND_UP_ULL {
($ll:expr, $d:expr) => {
DIV_ROUND_DOWN_ULL!((($ll) as u64).wrapping_add(($d) as u64).wrapping_sub(1), $d)
};
}

/// prior to 1.2 bit 7 was reserved mbz
macro_rules! DP_GET_SINK_COUNT {
($x:expr) => {
((($x) as i32) & 128) >> 1i32 | (($x) as i32) & 63
};
}

macro_rules! GENMASK_INPUT_CHECK__9 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_5401>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__122 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_5399>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__156 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_2951>() as i32
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
macro_rules! GENMASK_TYPE__34 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__122!() as $t).wrapping_add(type_max__3!() << $l & type_max__3!() >> BITS_PER_TYPE__2!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__113 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__9!() as $t).wrapping_add(type_max__3!() << $l & type_max__3!() >> BITS_PER_TYPE__2!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__210 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__156!() as $t).wrapping_add(type_max__3!() << (($l) as i32) & type_max__3!() >> BITS_PER_TYPE__2!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK__11 {
($h:expr, $l:expr) => {
GENMASK_TYPE__210!(u64, $h, $l)
};
}

macro_rules! GENMASK__12 {
($h:expr, $l:expr) => {
GENMASK_TYPE__34!(u64, $h, $l)
};
}

macro_rules! GENMASK__15 {
($h:expr, $l:expr) => {
GENMASK_TYPE__113!(u64, $h, $l)
};
}

macro_rules! HAS_AS_SDP {
($__display:expr) => {
(DISPLAY_VER!($__display) as i32) >= 13
};
}

macro_rules! HAS_BIGJOINER {
($__display:expr) => {
(DISPLAY_VER!($__display) as i32) >= 11 && HAS_DSC!($__display)
};
}

macro_rules! HAS_DDI {
($__display:expr) => {
(unsafe { (*DISPLAY_INFO!($__display)).__slate_bits_0.__get_has_ddi() }) as i32
};
}

macro_rules! HAS_DISPLAY {
($__display:expr) => {
((unsafe { (*DISPLAY_RUNTIME_INFO!($__display)).pipe_mask }) as i32) != 0
};
}

macro_rules! HAS_DOUBLE_BUFFERED_M_N {
($__display:expr) => {
IS_DISPLAY_VER!($__display, 9i32, 14i32) || ((unsafe { (*$__display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_broadwell() }) as i32) != 0
};
}

macro_rules! HAS_DSC_MST {
($__display:expr) => {
(DISPLAY_VER!($__display) as i32) >= 12 && HAS_DSC!($__display)
};
}

macro_rules! HAS_DSC {
($__display:expr) => {
unsafe { (*DISPLAY_RUNTIME_INFO!($__display)).has_dsc }
};
}

macro_rules! HAS_GMCH {
($__display:expr) => {
(unsafe { (*DISPLAY_INFO!($__display)).__slate_bits_0.__get_has_gmch() }) as i32
};
}

macro_rules! HAS_MSO {
($__display:expr) => {
(DISPLAY_VER!($__display) as i32) >= 12
};
}

macro_rules! HAS_PCH_CPT {
($display:expr) => {
(INTEL_PCH_TYPE!($display) as i32) == 2
};
}

macro_rules! HAS_PCH_IBX {
($display:expr) => {
(INTEL_PCH_TYPE!($display) as i32) == 1
};
}

macro_rules! HAS_ULTRAJOINER {
($__display:expr) => {
((unsafe { (*$__display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_dgfx() }) as i32) != 0 && (DISPLAY_VER!($__display) as i32) == 14 && HAS_DSC!($__display)
};
}

macro_rules! HAS_UNCOMPRESSED_JOINER {
($__display:expr) => {
(DISPLAY_VER!($__display) as i32) >= 13
};
}

macro_rules! HAS_VRR {
($__display:expr) => {
(DISPLAY_VER!($__display) as i32) >= 11
};
}

macro_rules! INTEL_DISPLAY_DEVICE_TRANS_OFFSET {
($display:expr, $trans:expr) => {
unsafe { *((*DISPLAY_INFO!($display)).trans_offsets.as_ptr() as *const u32).wrapping_offset((($trans) as i32) as isize) }.wrapping_sub(unsafe { *((*DISPLAY_INFO!($display)).trans_offsets.as_ptr() as *const u32).wrapping_offset(0 as isize) }).wrapping_add(DISPLAY_MMIO_BASE!($display))
};
}

macro_rules! INTEL_PCH_TYPE {
($_display:expr) => {
unsafe { (*$_display).pch_type }
};
}

macro_rules! IS_DISPLAY_VER {
($__display:expr, $from:expr, $until:expr) => {
(DISPLAY_VER!($__display) as i32) >= $from && (DISPLAY_VER!($__display) as i32) <= $until
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

macro_rules! REG_BIT__1211 {
($n:expr) => {
BIT_U32__1120!($n)
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! ULL {
() => {
_ULL!()
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

macro_rules! _ULL {
() => {
_AC!()
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
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut bool, $val) };
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

macro_rules! __assign_bit {
($nr:expr, $addr:expr, $value:expr) => {
if $value {
if (unsafe { __slate_intrinsic_1c5364fcd9da51fe($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___set_bit(($nr) as u64, ($addr) as *mut u64);
} else {
___set_bit(($nr) as u64, ($addr) as *mut u64);
}
} else {
if (unsafe { __slate_intrinsic_1c5364fcd9da51fe($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___clear_bit(($nr) as u64, ($addr) as *mut u64);
} else {
___clear_bit(($nr) as u64, ($addr) as *mut u64);
}
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

macro_rules! __cmp_once__2 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = ($y) as $type;
__v = if __v2 < __v3 { __v2 } else { __v3 };
__v
}
};
}

macro_rules! __cmp_once {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = ($x) as $type;
let __v3: $type = $y;
__v = if __v2 < __v3 { __v2 } else { __v3 };
__v
}
};
}

/// Conversion functions/macros from various pointer types to struct
/// intel_display pointer.
macro_rules! __drm_device_to_intel_display {
($p:expr) => {
{
let __v: *mut intel_display = if !$p.is_null() {
unsafe { __drm_to_display($p) }
} else {
core::ptr::null_mut::<intel_display>()
};
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

macro_rules! __is_nonneg {
($ux:expr) => {
statically_true!((($ux) as i64) >= 0)
};
}

macro_rules! __must_be_array__6 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_4748>() as i32
};
}

macro_rules! __must_be_array__14 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_603>() as i32
};
}

macro_rules! __must_be_array__34 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_735>() as i32
};
}

macro_rules! __must_be_array__40 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_607>() as i32
};
}

macro_rules! __must_be_array__48 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_600>() as i32
};
}

macro_rules! __must_be_array__51 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_245>() as i32
};
}

macro_rules! __must_be_array__58 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_588>() as i32
};
}

macro_rules! __must_be_array__64 {
() => {
((const { core::mem::size_of::<__SlateRecord_intel_dp_c_262_2>() }) as u64) as i32
};
}

macro_rules! __must_be_array__116 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_580>() as i32
};
}

macro_rules! __must_be_array__142 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_749>() as i32
};
}

macro_rules! __must_be_array__150 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_232>() as i32
};
}

macro_rules! __must_be_array__151 {
() => {
((const { core::mem::size_of::<__SlateRecord_intel_dp_c_262>() }) as u64) as i32
};
}

macro_rules! __must_be_array__166 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_583>() as i32
};
}

macro_rules! __must_be_array__180 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_4112>() as i32
};
}

macro_rules! __must_be_array__204 {
() => {
core::mem::size_of::<__SlateRecord_intel_dp_c_610>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__7 {
() => {
((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
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

/// This looks more complex than it should be. But we need to
/// get the type for the ~ right in round_down (it needs to be
/// as wide as the result!), and we want to evaluate the macro
/// arguments just once each.
macro_rules! __round_mask {
($y:expr) => {
$y.wrapping_sub(1)
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
macro_rules! __sign_use__2 {
($ux:expr) => {
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u32>() }) as u64) < 4) as i32)) }
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
macro_rules! __sign_use__9 {
($ux:expr) => {
if is_signed_type__4!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<u8>() as u64) < 4) as i32)) }
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
macro_rules! __type_half_max__2 {
() => {
1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((is_signed_type__3!() as i32) as u64)
};
}

macro_rules! __type_max__2 {
() => {
__type_half_max__2!().wrapping_sub(1).wrapping_add(__type_half_max__2!())
};
}

macro_rules! __types_ok3__2 {
($ux:expr, $uy:expr, $uz:expr) => {
__sign_use!($ux) & __sign_use!($uy) & __sign_use!($uz)
};
}

macro_rules! __types_ok__2 {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__2!($uy)
};
}

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
};
}

macro_rules! __types_ok__7 {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use__9!($uy)
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
let mut __v: *mut drm_display_mode;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__13!() as isize)) }) as *mut ()) as *mut drm_display_mode;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!($type) as isize)) }) as *mut ()) as *mut $type;
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
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__5!($type) as isize)) }) as *mut ()) as *mut $type;
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
($ptr:expr) => {
{
let mut __v: *mut intel_crtc;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__3!() as isize)) }) as *mut ()) as *mut intel_crtc;
__v
}
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

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__4 {
() => {
((1i32.wrapping_neg() as u8) as i32) < 1
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
(1i32.wrapping_neg() as u32) < 1
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
macro_rules! is_signed_type__3 {
() => {
(1i32.wrapping_neg() as u64) < 1
};
}

macro_rules! kcsan_check_atomic_read {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 4i32);
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

/// Add a ktime_t variable and a scalar nanosecond value.
/// res = kt + nsval:
macro_rules! ktime_add_ns {
($kt:expr, $nsval:expr) => {
(($kt) as u64).wrapping_add($nsval)
};
}

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry {
($ptr:expr) => {
container_of__3!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__4 {
($ptr:expr) => {
container_of__8!($ptr)
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
macro_rules! list_first_entry__4 {
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

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__3 {
($pos:expr, $member:ident) => {
list_entry__4!(unsafe { (*$pos).$member.next })
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

macro_rules! might_sleep_if {
($cond:expr) => {
loop {
if $cond {
might_sleep!();
}
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
macro_rules! min_t {
($type:ty, $x:expr, $y:expr) => {
__cmp_once!($type, $x, $y)
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

macro_rules! offsetof__4 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dp) as u64
};
}

macro_rules! offsetof__7 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, uapi) as u64
};
}

macro_rules! offsetof__3 {
() => {
core::mem::offset_of!(intel_crtc, pipe_head) as u64
};
}

macro_rules! offsetof__6 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, base.base) as u64
};
}

macro_rules! offsetof__5 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, hdmi) as u64
};
}

macro_rules! offsetof__13 {
() => {
core::mem::offset_of!(drm_display_mode, head) as u64
};
}

macro_rules! offsetof__2 {
() => {
core::mem::offset_of!(intel_encoder, base.head) as u64
};
}

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, base) as u64
};
}

macro_rules! raw_spin_lock_irq {
($lock:expr) => {
unsafe { _raw_spin_lock_irq($lock) };
};
}

macro_rules! raw_spin_unlock_irq {
($lock:expr) => {
unsafe { _raw_spin_unlock_irq($lock) };
};
}

/// round_down - round down to next specified power of 2
/// @x: the value to round
/// @y: multiple to round down to (must be a power of 2)
///
/// Rounds @x down to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding down, use rounddown() below.
macro_rules! round_down {
($x:expr, $y:expr) => {
($x) & !__round_mask!($y)
};
}

/// round_up - round up to next specified power of 2
/// @x: the value to round
/// @y: multiple to round up to (must be a power of 2)
///
/// Rounds @x up to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding up, use roundup() below.
macro_rules! round_up {
($x:expr, $y:expr) => {
($x.wrapping_sub(1) | __round_mask!($y)).wrapping_add(1)
};
}

/// rounddown - round down to next specified multiple
/// @x: the value to round
/// @y: multiple to round down to
///
/// Rounds @x down to next multiple of @y. If @y will always be a power
/// of 2, consider using the faster round_down().
macro_rules! rounddown__2 {
($x:expr, $y:expr) => {
{
let mut __v: i32;
let __v2: i32 = $x;
__v = __v2.wrapping_sub(__v2 % $y);
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

macro_rules! test_bit__5 {
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
macro_rules! time_after__2 {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck__2!(u64) != 0 && typecheck__2!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
}
};
}

macro_rules! to_intel_atomic_state {
($x:expr) => {
{
let mut __v: *mut intel_atomic_state;
__v = ((unsafe { ((($x) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(intel_atomic_state) as isize)) }) as *mut ()) as *mut intel_atomic_state;
__v
}
};
}

macro_rules! to_intel_connector {
($x:expr) => {
{
let mut __v: *mut intel_connector;
__v = ((unsafe { ((($x) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(intel_connector) as isize)) }) as *mut ()) as *mut intel_connector;
__v
}
};
}

macro_rules! to_intel_crtc {
($x:expr) => {
{
let mut __v: *mut intel_crtc;
__v = ((unsafe { ((($x) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(intel_crtc) as isize)) }) as *mut ()) as *mut intel_crtc;
__v
}
};
}

macro_rules! to_intel_crtc_state {
($crtc_state:expr) => {
{
let mut __v: *mut intel_crtc_state;
__v = ((unsafe { ((($crtc_state) as *mut ()) as *const u8).wrapping_offset(-(offsetof__7!(intel_crtc_state) as isize)) }) as *mut ()) as *mut intel_crtc_state;
__v
}
};
}

macro_rules! to_intel_digital_connector_state__2 {
($conn_state:expr) => {
{
let mut __v: *mut intel_digital_connector_state;
__v = ((unsafe { ((($conn_state) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(intel_digital_connector_state) as isize)) }) as *mut ()) as *mut intel_digital_connector_state;
__v as *const intel_digital_connector_state
}
};
}

macro_rules! to_intel_digital_connector_state {
($conn_state:expr) => {
{
let mut __v: *mut intel_digital_connector_state;
__v = ((unsafe { ((($conn_state) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(intel_digital_connector_state) as isize)) }) as *mut ()) as *mut intel_digital_connector_state;
__v
}
};
}

/// Convert various pointer types to struct intel_display pointer.
macro_rules! to_intel_display__3 {
($p:expr) => {
__drm_device_to_intel_display!(unsafe { (*(*(($p) as *mut intel_crtc_state)).uapi.crtc).dev })
};
}

/// Convert various pointer types to struct intel_display pointer.
macro_rules! to_intel_display__4 {
($p:expr) => {
{
let __v: *mut intel_display = if !$p.is_null() {
unsafe { __drm_to_display($p) }
} else {
core::ptr::null_mut::<intel_display>()
};
__v
}
};
}

/// Convert various pointer types to struct intel_display pointer.
macro_rules! to_intel_display__2 {
($p:expr) => {
__drm_device_to_intel_display!(unsafe { (*dp_to_dig_port($p)).base.base.dev })
};
}

/// Convert various pointer types to struct intel_display pointer.
macro_rules! to_intel_display {
($p:expr) => {
__drm_device_to_intel_display!(unsafe { (*$p).base.dev })
};
}

/// Convert various pointer types to struct intel_display pointer.
macro_rules! to_intel_display__6 {
($p:expr) => {
__drm_device_to_intel_display!(unsafe { (*$p).base.base.dev })
};
}

/// Convert various pointer types to struct intel_display pointer.
macro_rules! to_intel_display__5 {
($p:expr) => {
__drm_device_to_intel_display!(unsafe { (*(*$p).uapi.crtc).dev })
};
}

/// Convert various pointer types to struct intel_display pointer.
macro_rules! to_intel_display__10 {
($p:expr) => {
__drm_device_to_intel_display!(unsafe { (*(($p) as *mut intel_connector)).base.dev })
};
}

macro_rules! to_intel_encoder {
($x:expr) => {
{
let mut __v: *mut intel_encoder;
__v = ((unsafe { ((($x) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(intel_encoder) as isize)) }) as *mut ()) as *mut intel_encoder;
__v
}
};
}

macro_rules! type_max__3 {
() => {
__type_max__2!()
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

macro_rules! ww_mutex_base_is_locked {
($b:expr) => {
unsafe { mutex_is_locked($b) }
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __SCK__might_resched: static_call_key;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut jiffies: u64;
    static mut kstat: kernel_stat;
    static mut __drm_debug: u64;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn iomem_get_mapping() -> *mut address_space;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __bad_size_call_parameter();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn memcmp(arg0: *const (), arg1: *const (), arg2: u64) -> i32;
    fn memchr_inv(arg0: *const (), arg1: i32, arg2: u64) -> *mut ();
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn irq_to_desc(arg0: u32) -> *mut irq_desc;
    fn __compiletime_assert_69();
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn mutex_is_locked(arg0: *mut mutex) -> bool;
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn schedule_timeout_uninterruptible(arg0: i64) -> i64;
    fn wait_for_completion_timeout(arg0: *mut completion, arg1: u64) -> u64;
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn ktime_get() -> i64;
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_395();
    fn __compiletime_assert_416();
    fn dev_driver_string(arg0: *const device) -> *const u8;
    fn acpi_dev_get_pci_dev(arg0: *mut acpi_device) -> *mut pci_dev;
    fn acpi_create_platform_device(arg0: *mut acpi_device, arg1: *const property_entry) -> *mut platform_device;
    fn msleep(arg0: u32);
    fn usleep_range_state(arg0: u64, arg1: u64, arg2: u32);
    fn seq_buf_printf(arg0: *mut seq_buf, arg1: *const u8, ...) -> i32;
    fn hdmi_drm_infoframe_pack_only(arg0: *const hdmi_drm_infoframe, arg1: *mut (), arg2: u64) -> i64;
    fn hdmi_drm_infoframe_unpack_only(arg0: *mut hdmi_drm_infoframe, arg1: *const (), arg2: u64) -> i32;
    fn drm_connector_init_with_ddc(arg0: *mut drm_device, arg1: *mut drm_connector, arg2: *const drm_connector_funcs, arg3: i32, arg4: *mut i2c_adapter) -> i32;
    fn drm_connector_cleanup(arg0: *mut drm_connector);
    fn drm_connector_attach_dp_subconnector_property(arg0: *mut drm_connector);
    fn drm_connector_attach_content_type_property(arg0: *mut drm_connector) -> i32;
    fn drm_connector_attach_vrr_capable_property(arg0: *mut drm_connector) -> i32;
    fn drm_connector_attach_hdr_output_metadata_property(arg0: *mut drm_connector);
    fn drm_connector_set_vrr_capable_property(arg0: *mut drm_connector, arg1: bool);
    fn drm_connector_set_panel_orientation_with_quirk(arg0: *mut drm_connector, arg1: i32, arg2: i32, arg3: i32) -> i32;
    fn drm_connector_attach_max_bpc_property(arg0: *mut drm_connector, arg1: i32, arg2: i32) -> i32;
    fn drm_connector_list_iter_begin(arg0: *mut drm_device, arg1: *mut drm_connector_list_iter);
    fn drm_connector_list_iter_next(arg0: *mut drm_connector_list_iter) -> *mut drm_connector;
    fn drm_connector_list_iter_end(arg0: *mut drm_connector_list_iter);
    fn drm_dp_link_rate_to_bw_code(arg0: i32) -> u8;
    fn drm_dp_bw_code_to_link_rate(arg0: u8) -> i32;
    fn drm_dp_as_sdp_supported(arg0: *mut drm_dp_aux, arg1: *const u8) -> bool;
    fn drm_dp_dsc_sink_bpp_incr(arg0: *const u8) -> u8;
    fn drm_dp_dsc_slice_count_to_mask(arg0: i32) -> u32;
    fn drm_dp_dsc_sink_slice_count_mask(arg0: *const u8, arg1: bool) -> u32;
    fn drm_dp_dsc_sink_max_slice_count(arg0: *const u8, arg1: bool) -> u8;
    fn drm_dp_dsc_sink_line_buf_depth(arg0: *const u8) -> u8;
    fn drm_dp_dsc_sink_supported_input_bpcs(arg0: *const u8, arg1: *mut u8) -> i32;
    fn drm_dp_dsc_sink_max_slice_throughput(arg0: *const u8, arg1: i32, arg2: bool) -> i32;
    fn drm_dp_dsc_branch_max_overall_throughput(arg0: *const u8, arg1: bool) -> i32;
    fn drm_dp_dsc_branch_max_line_width(arg0: *const u8) -> i32;
    fn drm_dp_dpcd_set_probe(arg0: *mut drm_dp_aux, arg1: bool);
    fn drm_dp_dpcd_read(arg0: *mut drm_dp_aux, arg1: u32, arg2: *mut (), arg3: u64) -> i64;
    fn drm_dp_dpcd_write(arg0: *mut drm_dp_aux, arg1: u32, arg2: *mut (), arg3: u64) -> i64;
    fn drm_dp_read_dpcd_caps(arg0: *mut drm_dp_aux, arg1: *mut u8) -> i32;
    fn drm_dp_read_downstream_info(arg0: *mut drm_dp_aux, arg1: *const u8, arg2: *mut u8) -> i32;
    fn drm_dp_downstream_max_dotclock(arg0: *const u8, arg1: *const u8) -> i32;
    fn drm_dp_downstream_max_tmds_clock(arg0: *const u8, arg1: *const u8, arg2: *const drm_edid) -> i32;
    fn drm_dp_downstream_min_tmds_clock(arg0: *const u8, arg1: *const u8, arg2: *const drm_edid) -> i32;
    fn drm_dp_downstream_max_bpc(arg0: *const u8, arg1: *const u8, arg2: *const drm_edid) -> i32;
    fn drm_dp_downstream_420_passthrough(arg0: *const u8, arg1: *const u8) -> bool;
    fn drm_dp_downstream_444_to_420_conversion(arg0: *const u8, arg1: *const u8) -> bool;
    fn drm_dp_downstream_mode(arg0: *mut drm_device, arg1: *const u8, arg2: *const u8) -> *mut drm_display_mode;
    fn drm_dp_set_subconnector_property(arg0: *mut drm_connector, arg1: u32, arg2: *const u8, arg3: *const u8);
    fn drm_dp_read_sink_count_cap(arg0: *mut drm_connector, arg1: *const u8, arg2: *const drm_dp_desc) -> bool;
    fn drm_dp_read_sink_count(arg0: *mut drm_dp_aux) -> i32;
    fn drm_dp_lttpr_count(arg0: *const u8) -> i32;
    fn drm_dp_lttpr_max_link_rate(arg0: *const u8) -> i32;
    fn drm_dp_lttpr_max_lane_count(arg0: *const u8) -> i32;
    fn drm_dp_aux_register(arg0: *mut drm_dp_aux) -> i32;
    fn drm_dp_aux_unregister(arg0: *mut drm_dp_aux);
    fn drm_dp_read_desc(arg0: *mut drm_dp_aux, arg1: *mut drm_dp_desc, arg2: bool) -> i32;
    fn drm_dp_get_pcon_max_frl_bw(arg0: *const u8, arg1: *const u8) -> i32;
    fn drm_dp_pcon_frl_prepare(arg0: *mut drm_dp_aux, arg1: bool) -> i32;
    fn drm_dp_pcon_is_frl_ready(arg0: *mut drm_dp_aux) -> bool;
    fn drm_dp_pcon_frl_configure_1(arg0: *mut drm_dp_aux, arg1: i32, arg2: u8) -> i32;
    fn drm_dp_pcon_frl_configure_2(arg0: *mut drm_dp_aux, arg1: i32, arg2: u8) -> i32;
    fn drm_dp_pcon_frl_enable(arg0: *mut drm_dp_aux) -> i32;
    fn drm_dp_pcon_hdmi_link_active(arg0: *mut drm_dp_aux) -> bool;
    fn drm_dp_pcon_hdmi_link_mode(arg0: *mut drm_dp_aux, arg1: *mut u8) -> i32;
    fn drm_dp_pcon_hdmi_frl_link_error_count(arg0: *mut drm_dp_aux, arg1: *mut drm_connector);
    fn drm_dp_pcon_enc_is_dsc_1_2(arg0: *const u8) -> bool;
    fn drm_dp_pcon_dsc_max_slices(arg0: *const u8) -> i32;
    fn drm_dp_pcon_dsc_max_slice_width(arg0: *const u8) -> i32;
    fn drm_dp_pcon_dsc_bpp_incr(arg0: *const u8) -> i32;
    fn drm_dp_pcon_pps_override_param(arg0: *mut drm_dp_aux, arg1: *mut u8) -> i32;
    fn drm_dp_downstream_rgb_to_ycbcr_conversion(arg0: *const u8, arg1: *const u8, arg2: u8) -> bool;
    fn drm_dp_pcon_convert_rgb_to_ycbcr(arg0: *mut drm_dp_aux, arg1: u8) -> i32;
    fn drm_dp_bw_overhead(arg0: i32, arg1: i32, arg2: i32, arg3: i32, arg4: u64) -> i32;
    fn drm_dp_max_dprx_data_rate(arg0: i32, arg1: i32) -> i32;
    fn drm_dp_vsc_sdp_pack(arg0: *const drm_dp_vsc_sdp, arg1: *mut dp_sdp) -> i64;
    fn drm_dp_link_symbol_cycles(arg0: i32, arg1: i32, arg2: i32, arg3: i32, arg4: i32, arg5: bool) -> i32;
    fn drm_dsc_compute_rc_parameters(arg0: *mut drm_dsc_config) -> i32;
    fn drm_hdmi_infoframe_set_hdr_metadata(arg0: *mut hdmi_drm_infoframe, arg1: *const drm_connector_state) -> i32;
    fn drm_modeset_lock(arg0: *mut drm_modeset_lock, arg1: *mut drm_modeset_acquire_ctx) -> i32;
    fn drm_mode_probed_add(arg0: *mut drm_connector, arg1: *mut drm_display_mode);
    fn drm_mode_is_420_only(arg0: *const drm_display_info, arg1: *const drm_display_mode) -> bool;
    fn drm_mode_is_420_also(arg0: *const drm_display_info, arg1: *const drm_display_mode) -> bool;
    fn drm_mode_is_420(arg0: *const drm_display_info, arg1: *const drm_display_mode) -> bool;
    fn drm_mode_set_name(arg0: *mut drm_display_mode);
    fn drm_mode_vrefresh(arg0: *const drm_display_mode) -> i32;
    fn drm_atomic_helper_connector_destroy_state(arg0: *mut drm_connector, arg1: *mut drm_connector_state);
    fn drm_probe_ddc(arg0: *mut i2c_adapter) -> bool;
    fn drm_default_rgb_quant_range(arg0: *const drm_display_mode) -> u32;
    fn drm_edid_dup(arg0: *const drm_edid) -> *const drm_edid;
    fn drm_edid_free(arg0: *const drm_edid);
    fn drm_edid_read_ddc(arg0: *mut drm_connector, arg1: *mut i2c_adapter) -> *const drm_edid;
    fn drm_edid_connector_update(arg0: *mut drm_connector, arg1: *const drm_edid) -> i32;
    fn drm_edid_connector_add_modes(arg0: *mut drm_connector) -> i32;
    fn drm_edid_has_quirk(arg0: *mut drm_connector, arg1: u32) -> bool;
    fn __drm_dev_dbg(arg0: *mut _ddebug, arg1: *const device, arg2: u32, arg3: *const u8, ...);
    fn drm_helper_probe_single_connector_modes(arg0: *mut drm_connector, arg1: u32, arg2: u32) -> i32;
    fn vlv_get_dpll(arg0: *mut intel_display) -> *const dpll;
    fn intel_alpm_init(arg0: *mut intel_dp);
    fn intel_alpm_lobf_compute_config(arg0: *mut intel_dp, arg1: *mut intel_crtc_state, arg2: *mut drm_connector_state);
    fn intel_alpm_disable(arg0: *mut intel_dp);
    fn intel_alpm_get_error(arg0: *mut intel_dp) -> bool;
    fn intel_alpm_lobf_compute_config_late(arg0: *mut intel_dp, arg1: *mut intel_crtc_state);
    fn intel_digital_connector_atomic_get_property(arg0: *mut drm_connector, arg1: *const drm_connector_state, arg2: *mut drm_property, arg3: *mut u64) -> i32;
    fn intel_digital_connector_atomic_set_property(arg0: *mut drm_connector, arg1: *mut drm_connector_state, arg2: *mut drm_property, arg3: u64) -> i32;
    fn intel_digital_connector_atomic_check(arg0: *mut drm_connector, arg1: *mut drm_atomic_commit) -> i32;
    fn intel_digital_connector_duplicate_state(arg0: *mut drm_connector) -> *mut drm_connector_state;
    fn intel_connector_needs_modeset(arg0: *mut intel_atomic_state, arg1: *mut drm_connector) -> bool;
    fn intel_atomic_get_digital_connector_state(arg0: *mut intel_atomic_state, arg1: *mut intel_connector) -> *mut intel_digital_connector_state;
    fn intel_atomic_get_crtc_state(arg0: *mut drm_atomic_commit, arg1: *mut intel_crtc) -> *mut intel_crtc_state;
    fn intel_audio_compute_config(arg0: *mut intel_encoder, arg1: *mut intel_crtc_state, arg2: *mut drm_connector_state) -> bool;
    fn intel_backlight_setup(arg0: *mut intel_connector, arg1: i32) -> i32;
    fn intel_backlight_enable(arg0: *const intel_crtc_state, arg1: *const drm_connector_state);
    fn intel_backlight_disable(arg0: *const drm_connector_state);
    fn intel_connector_destroy(arg0: *mut drm_connector);
    fn intel_connector_register(arg0: *mut drm_connector) -> i32;
    fn intel_connector_unregister(arg0: *mut drm_connector);
    fn intel_connector_attach_encoder(arg0: *mut intel_connector, arg1: *mut intel_encoder);
    fn intel_connector_get_hw_state(arg0: *mut intel_connector) -> bool;
    fn intel_attach_force_audio_property(arg0: *mut drm_connector);
    fn intel_attach_broadcast_rgb_property(arg0: *mut drm_connector);
    fn intel_attach_hdmi_colorspace_property(arg0: *mut drm_connector);
    fn intel_attach_dp_colorspace_property(arg0: *mut drm_connector);
    fn intel_attach_scaling_mode_property(arg0: *mut drm_connector);
    fn intel_connector_queue_modeset_retry_work(arg0: *mut intel_connector);
    fn intel_encoder_is_c10phy(arg0: *mut intel_encoder) -> bool;
    fn intel_ddi_connector_get_hw_state(arg0: *mut intel_connector) -> bool;
    fn intel_display_device_enabled(arg0: *mut intel_display) -> bool;
    fn intel_display_power_get(arg0: *mut intel_display, arg1: u32) -> *mut ref_tracker;
    fn __intel_display_power_put_async(arg0: *mut intel_display, arg1: u32, arg2: *mut ref_tracker, arg3: i32);
    fn intel_display_power_flush_work(arg0: *mut intel_display);
    fn intel_dmc_wl_get(arg0: *mut intel_display, arg1: __SlateRecord_i915_reg_defs_h_14);
    fn intel_dmc_wl_put(arg0: *mut intel_display, arg1: __SlateRecord_i915_reg_defs_h_14);
    fn intel_opregion_get_edid(arg0: *mut intel_connector) -> *const drm_edid;
    fn to_intel_uncore(arg0: *mut drm_device) -> *mut intel_uncore;
    fn intel_display_driver_check_access(arg0: *mut intel_display) -> bool;
    fn intel_display_rpm_suspended(arg0: *mut intel_display) -> bool;
    fn drm_atomic_get_connector_state(arg0: *mut drm_atomic_commit, arg1: *mut drm_connector) -> *mut drm_connector_state;
    fn drm_atomic_add_affected_connectors(arg0: *mut drm_atomic_commit, arg1: *mut drm_crtc) -> i32;
    fn drm_atomic_add_affected_planes(arg0: *mut drm_atomic_commit, arg1: *mut drm_crtc) -> i32;
    fn drm_atomic_commit(arg0: *mut drm_atomic_commit) -> i32;
    fn drm_dp_read_mst_cap(arg0: *mut drm_dp_aux, arg1: *const u8) -> u32;
    fn drm_dp_mst_topology_mgr_set_mst(arg0: *mut drm_dp_mst_topology_mgr, arg1: bool) -> i32;
    fn drm_dp_mst_hpd_irq_handle_event(arg0: *mut drm_dp_mst_topology_mgr, arg1: *const u8, arg2: *mut u8, arg3: *mut bool) -> i32;
    fn drm_dp_mst_hpd_irq_send_new_request(arg0: *mut drm_dp_mst_topology_mgr);
    fn drm_dp_mst_topology_mgr_suspend(arg0: *mut drm_dp_mst_topology_mgr);
    fn drm_dp_mst_topology_mgr_resume(arg0: *mut drm_dp_mst_topology_mgr, arg1: bool) -> i32;
    fn drm_dp_mst_root_conn_atomic_check(arg0: *mut drm_connector_state, arg1: *mut drm_dp_mst_topology_mgr) -> i32;
    fn drm_format_info(arg0: u32) -> *const drm_format_info;
    fn intel_bios_init_panel_early(arg0: *mut intel_display, arg1: *mut intel_panel, arg2: *const intel_bios_encoder_data);
    fn intel_bios_init_panel_late(arg0: *mut intel_display, arg1: *mut intel_panel, arg2: *const intel_bios_encoder_data, arg3: *const drm_edid);
    fn intel_bios_fini_panel(arg0: *mut intel_panel);
    fn intel_bios_encoder_data_lookup(arg0: *mut intel_display, arg1: i32) -> *const intel_bios_encoder_data;
    fn intel_bios_encoder_supports_edp(arg0: *const intel_bios_encoder_data) -> bool;
    fn intel_bios_encoder_is_lspcon(arg0: *const intel_bios_encoder_data) -> bool;
    fn intel_bios_encoder_reject_edp_rate(arg0: *const intel_bios_encoder_data, arg1: i32) -> bool;
    fn intel_bios_dp_max_lane_count(arg0: *const intel_bios_encoder_data) -> i32;
    fn intel_bios_dp_max_link_rate(arg0: *const intel_bios_encoder_data) -> i32;
    fn intel_bios_dp_has_shared_aux_ch(arg0: *const intel_bios_encoder_data) -> bool;
    fn intel_link_compute_m_n(arg0: u16, arg1: i32, arg2: i32, arg3: i32, arg4: i32, arg5: *mut intel_link_m_n);
    fn intel_mode_valid_max_plane_size(arg0: *mut intel_display, arg1: *const drm_display_mode, arg2: i32) -> i32;
    fn intel_cpu_transcoder_mode_valid(arg0: *mut intel_display, arg1: *const drm_display_mode) -> i32;
    fn intel_encoder_is_combo(arg0: *mut intel_encoder) -> bool;
    fn intel_encoder_is_tc(arg0: *mut intel_encoder) -> bool;
    fn intel_zero_m_n(arg0: *mut intel_link_m_n);
    fn intel_cpu_transcoder_has_m2_n2(arg0: *mut intel_display, arg1: i32) -> bool;
    fn intel_max_uncompressed_dotclock(arg0: *mut intel_display) -> i32;
    fn intel_display_min_pipe_bpp() -> i32;
    fn intel_crtc_num_joined_pipes(arg0: *const intel_crtc_state) -> i32;
    fn __drm_to_display(arg0: *mut drm_device) -> *mut intel_display;
    fn intel_dp_link_caps_iter_start(arg0: *mut intel_dp_link_caps_iter, arg1: *mut intel_dp_link_caps, arg2: intel_dp_link_caps_order, arg3: intel_dp_link_caps_filter);
    fn intel_dp_link_caps_iter_end(arg0: *mut intel_dp_link_caps_iter);
    fn intel_dp_link_caps_connector_compute_order(arg0: *mut intel_connector) -> intel_dp_link_caps_order;
    fn intel_dp_link_caps_print_common_rates(arg0: *mut intel_dp_link_caps);
    fn intel_dp_link_caps_filter_add(arg0: *mut intel_dp_link_caps, arg1: *mut intel_dp_link_caps_filter, arg2: *const intel_dp_link_config) -> bool;
    fn intel_dp_link_caps_get_max_config(arg0: *mut intel_dp_link_caps, arg1: u32, arg2: intel_dp_link_caps_filter, arg3: *mut intel_dp_link_config) -> bool;
    fn intel_dp_link_caps_get_max_bw_config(arg0: *mut intel_dp_link_caps, arg1: *mut intel_dp_link_config);
    fn intel_dp_link_caps_update(arg0: *mut intel_dp_link_caps, arg1: *const i32, arg2: i32, arg3: i32, arg4: bool) -> bool;
    fn intel_dp_link_caps_reset(arg0: *mut intel_dp_link_caps);
    fn intel_dp_link_caps_init(arg0: *mut intel_dp) -> *mut intel_dp_link_caps;
    fn intel_dp_link_caps_cleanup(arg0: *mut intel_dp_link_caps);
    fn intel_dp_aux_fini(arg0: *mut intel_dp);
    fn intel_dp_aux_init(arg0: *mut intel_dp);
    fn intel_dp_hdcp_init(arg0: *mut intel_digital_port, arg1: *mut intel_connector) -> i32;
    fn intel_dp_read_dprx_caps(arg0: *mut intel_dp, arg1: *mut u8) -> i32;
    fn intel_dp_init_lttpr_and_dprx_caps(arg0: *mut intel_dp) -> i32;
    fn intel_dp_link_training_get_force_retrain(arg0: *mut intel_dp_link_training) -> bool;
    fn intel_dp_check_link_state(arg0: *mut intel_dp);
    fn intel_dp_link_training_reset(arg0: *mut intel_dp_link_training);
    fn intel_dp_link_training_init(arg0: *mut intel_dp) -> *mut intel_dp_link_training;
    fn intel_dp_link_training_cleanup(arg0: *mut intel_dp_link_training);
    fn intel_dp_mst_encoder_init(arg0: *mut intel_digital_port, arg1: i32) -> i32;
    fn intel_dp_mst_encoder_cleanup(arg0: *mut intel_digital_port);
    fn intel_dp_mst_source_support(arg0: *mut intel_dp) -> bool;
    fn intel_dp_mst_prepare_probe(arg0: *mut intel_dp);
    fn intel_dp_mst_verify_dpcd_state(arg0: *mut intel_dp) -> bool;
    fn intel_dp_mtp_tu_compute_config(arg0: *mut intel_dp, arg1: *mut intel_crtc_state, arg2: *mut drm_connector_state, arg3: i32, arg4: i32, arg5: i32, arg6: bool) -> i32;
    fn intel_dp_test_reset(arg0: *mut intel_dp);
    fn intel_dp_test_request(arg0: *mut intel_dp);
    fn intel_dp_test_compute_config(arg0: *mut intel_connector, arg1: *mut intel_crtc_state, arg2: *mut link_config_limits) -> bool;
    fn intel_dp_test_short_pulse(arg0: *mut intel_dp) -> bool;
    fn intel_cpu_transcoder_has_drrs(arg0: *mut intel_display, arg1: i32) -> bool;
    fn intel_encoder_link_check_flush_work(arg0: *mut intel_encoder);
    fn is_hdcp_supported(arg0: *mut intel_display, arg1: i32) -> bool;
    fn intel_hdcp_handle_cp_irq(arg0: *mut intel_connector);
    fn intel_hdmi_infoframe_enable(arg0: u32) -> u32;
    fn intel_hdmi_bpc_possible(arg0: *const intel_crtc_state, arg1: i32, arg2: bool) -> bool;
    fn intel_hdmi_tmds_clock(arg0: i32, arg1: i32, arg2: u32) -> i32;
    fn intel_hdmi_dsc_get_bpp(arg0: i32, arg1: i32, arg2: i32, arg3: i32, arg4: bool, arg5: i32) -> i32;
    fn intel_hdmi_dsc_get_num_slices(arg0: *const intel_crtc_state, arg1: i32, arg2: i32, arg3: i32, arg4: i32) -> i32;
    fn intel_hdmi_dsc_get_slice_height(arg0: i32) -> i32;
    fn intel_hpd_schedule_detection(arg0: *mut intel_display) -> bool;
    fn intel_hpd_enable_detection(arg0: *mut intel_encoder);
    fn intel_lspcon_init(arg0: *mut intel_digital_port) -> bool;
    fn intel_lspcon_active(arg0: *mut intel_digital_port) -> bool;
    fn intel_lspcon_detect_hdr_capability(arg0: *mut intel_digital_port) -> bool;
    fn intel_lspcon_resume(arg0: *mut intel_digital_port);
    fn intel_lspcon_wait_pcon_mode(arg0: *mut intel_digital_port);
    fn intel_get_lvds_encoder(arg0: *mut intel_display) -> *mut intel_encoder;
    fn intel_panel_init(arg0: *mut intel_connector, arg1: *const drm_edid) -> i32;
    fn intel_panel_preferred_fixed_mode(arg0: *mut intel_connector) -> *const drm_display_mode;
    fn intel_panel_downclock_mode(arg0: *mut intel_connector, arg1: *const drm_display_mode) -> *const drm_display_mode;
    fn intel_panel_highest_mode(arg0: *mut intel_connector, arg1: *const drm_display_mode) -> *const drm_display_mode;
    fn intel_panel_get_modes(arg0: *mut intel_connector) -> i32;
    fn intel_panel_drrs_type(arg0: *mut intel_connector) -> u32;
    fn intel_panel_mode_valid(arg0: *mut intel_connector, arg1: *const drm_display_mode, arg2: *mut i32) -> i32;
    fn intel_panel_compute_config(arg0: *mut intel_atomic_state, arg1: *mut intel_crtc_state, arg2: *mut intel_connector) -> i32;
    fn intel_panel_add_edid_fixed_modes(arg0: *mut intel_connector, arg1: bool);
    fn intel_panel_add_vbt_lfp_fixed_mode(arg0: *mut intel_connector);
    fn intel_pfit_compute_config(arg0: *mut intel_crtc_state, arg1: *const drm_connector_state) -> i32;
    fn intel_pfit_mode_valid(arg0: *mut intel_display, arg1: *const drm_display_mode, arg2: u32, arg3: i32) -> i32;
    fn intel_pps_backlight_on(arg0: *mut intel_dp);
    fn intel_pps_backlight_off(arg0: *mut intel_dp);
    fn intel_pps_vdd_on(arg0: *mut intel_dp);
    fn intel_pps_vdd_off(arg0: *mut intel_dp);
    fn intel_pps_vdd_off_sync(arg0: *mut intel_dp);
    fn intel_pps_have_panel_power_or_vdd(arg0: *mut intel_dp) -> bool;
    fn intel_pps_wait_power_cycle(arg0: *mut intel_dp);
    fn intel_pps_init(arg0: *mut intel_dp) -> bool;
    fn intel_pps_init_late(arg0: *mut intel_dp);
    fn vlv_pps_pipe_init(arg0: *mut intel_dp);
    fn vlv_pps_backlight_initial_pipe(arg0: *mut intel_dp) -> i32;
    fn intel_psr_init_dpcd(arg0: *mut intel_dp, arg1: *mut intel_connector);
    fn intel_psr_init(arg0: *mut intel_dp);
    fn intel_psr_compute_config(arg0: *mut intel_dp, arg1: *mut intel_crtc_state, arg2: *mut drm_connector_state);
    fn intel_psr_short_pulse(arg0: *mut intel_dp);
    fn intel_psr_needs_alpm_aux_less(arg0: *mut intel_dp, arg1: *const intel_crtc_state) -> bool;
    fn intel_psr_compute_config_late(arg0: *mut intel_dp, arg1: *mut intel_crtc_state);
    fn intel_psr_pr_async_video_timing_supported(arg0: *mut intel_dp) -> bool;
    fn intel_init_dpcd_quirks(arg0: *mut intel_dp, arg1: *const drm_dp_dpcd_ident);
    fn intel_has_quirk(arg0: *mut intel_display, arg1: u32) -> bool;
    fn intel_tc_port_handles_hpd_glitches(arg0: *mut intel_digital_port) -> bool;
    fn intel_tc_port_max_lane_count(arg0: *mut intel_digital_port) -> i32;
    fn intel_crtc_vblank_length(arg0: *const intel_crtc_state) -> i32;
    fn intel_dsc_source_support(arg0: *const intel_crtc_state) -> bool;
    fn intel_dsc_line_slice_count(arg0: *const intel_dsc_slice_config) -> i32;
    fn intel_dsc_get_slice_config(arg0: *mut intel_display, arg1: i32, arg2: i32, arg3: *mut intel_dsc_slice_config) -> bool;
    fn intel_dsc_compute_params(arg0: *mut intel_crtc_state) -> i32;
    fn intel_dsc_enable_on_crtc(arg0: *mut intel_crtc_state);
    fn intel_dsc_enabled_on_link(arg0: *const intel_crtc_state) -> bool;
    fn intel_dsc_get_pixel_rate_with_dsc_bubbles(arg0: *mut intel_display, arg1: i32, arg2: i32, arg3: i32) -> i32;
    fn intel_vrr_is_capable(arg0: *mut intel_connector) -> bool;
    fn intel_vrr_possible(arg0: *const intel_crtc_state) -> bool;
    fn intel_vrr_compute_config(arg0: *mut intel_crtc_state, arg1: *mut drm_connector_state);
    fn intel_vrr_vmin_vtotal(arg0: *const intel_crtc_state) -> i32;
    fn __compiletime_assert_716();
    fn __compiletime_assert_719();
    fn __compiletime_assert_720();
    fn __compiletime_assert_723();
    fn __compiletime_assert_726();
    fn __compiletime_assert_730();
    fn __compiletime_assert_735();
    fn __compiletime_assert_738();
    fn __compiletime_assert_741();
    fn __compiletime_assert_745();
    fn __compiletime_assert_752();
    fn __compiletime_assert_755();
    fn __compiletime_assert_758();
    fn __compiletime_assert_761();
    fn __compiletime_assert_764();
    fn __compiletime_assert_767();
    fn __compiletime_assert_773();
    fn __compiletime_assert_778();
    fn __compiletime_assert_783();
    fn __compiletime_assert_786();
    fn __compiletime_assert_789();
    fn __compiletime_assert_792();
    fn __compiletime_assert_795();
    fn __compiletime_assert_802();
    fn __compiletime_assert_806();
    fn __compiletime_assert_809();
    fn __compiletime_assert_812();
    fn __compiletime_assert_815();
    fn __compiletime_assert_818();
    fn __compiletime_assert_821();
    fn __compiletime_assert_824();
    fn __compiletime_assert_827();
    fn __compiletime_assert_830();
    fn __compiletime_assert_833();
    fn __compiletime_assert_836();
    fn __compiletime_assert_840();
    fn __compiletime_assert_841();
    fn __compiletime_assert_845();
    fn __compiletime_assert_849();
    fn __compiletime_assert_852();
    fn __compiletime_assert_856();
    fn __compiletime_assert_857();
    fn __compiletime_assert_858();
    fn __compiletime_assert_859();
    fn __compiletime_assert_862();
    fn __compiletime_assert_865();
    fn __compiletime_assert_869();
    fn __compiletime_assert_880();
    fn __compiletime_assert_881();
    fn __compiletime_assert_883();
    fn __compiletime_assert_894();
    fn __compiletime_assert_897();
    fn __compiletime_assert_900();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
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
// Resource extended types
// Userland may not map this resource
// No address assigned yet
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
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


fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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
struct static_key_false {
    key: static_key,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits108U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits109U0,
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

// instrument_get_user() - add instrumentation to get_user()-like macros
// @to: destination variable, may not be address-taken
//
// get_user() and friends are fragile, so it may depend on the implementation
// whether the instrumentation happens before or after the data is copied from
// the userspace.
#[inline(always)]
fn generic___set_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
unsafe {
*p = *p | mask;
}
}

#[inline(always)]
fn generic___clear_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
let __v2: u64 = (unsafe { *p }) & !mask;
unsafe {
*p = __v2;
}
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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

#[inline(always)]
fn arch___clear_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btrq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}

#[inline(always)]
fn constant_test_bit(nr: i64, addr: *const u64) -> bool {
1u64 << (nr & 63i64) & unsafe { core::ptr::read_volatile(addr.wrapping_offset((nr >> 6i32) as isize)) } != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
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
#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn ___set_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___set_bit(nr, addr);
}

#[inline(always)]
fn ___clear_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___clear_bit(nr, addr);
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


fn mem_is_zero(s: *const (), n: u64) -> bool {
unsafe { memchr_inv(s, 0i32, n) }.is_null()
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
struct pollfd {
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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

fn div_u64_rem(dividend: u64, divisor: u32, remainder: &mut u32) -> u64 {
unsafe {
*(remainder as *mut u32) = (dividend % (divisor as u64)) as u32;
}
dividend / (divisor as u64)
}

fn div_u64(dividend: u64, divisor: u32) -> u64 {
let mut remainder: u32 = 0;
div_u64_rem(dividend, divisor, unsafe { &mut (*core::ptr::addr_of_mut!(remainder)) })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
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
    __slate_bits_0: __slate_bits::__SlateBits128U0,
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
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irq(lock: *mut spinlock) {
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
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
struct futex_private_hash {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
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
struct sem_undo_list {
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
struct hrtimer_clock_base {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
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
struct seccomp_filter {
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
struct futex_unlock_cs_ranges {
    cs_ranges: [futex_unlock_cs_range; 2],
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
struct plist_head {
    node_list: list_head,
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
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
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
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
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
struct posix_cputimers {
    bases: [posix_cputimer_base; 3],
    timers_active: u32,
    expiry_active: u32,
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
struct posix_cputimers_work {
    work: callback_head,
    mutex: mutex,
    scheduled: u32,
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
struct irq_work {
    node: __call_single_node,
    func: Option<unsafe extern "C" fn(*mut irq_work)>,
    irqwait: rcuwait,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
union __SlateRecord_sched_h_826 {
    ipi_mask_ptr: *mut cpumask,
    ipi_mask_val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_class {
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
    __slate_bits_0: __slate_bits::__SlateBits298U0,
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
struct gendisk {
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
struct vm_struct {
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
    __slate_bits_0: __slate_bits::__SlateBits73U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits73U1,
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
    __slate_bits_2: __slate_bits::__SlateBits73U2,
    mce_kill_me: callback_head,
    mce_count: i32,
    kretprobe_instances: llist_head,
    rethooks: llist_head,
    l1d_flush_kill: callback_head,
    unwind_info: unwind_task_info,
    thread: thread_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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

// Subtract a scalar nanosecod from a ktime_t variable
// res = kt - nsval:
// Map the ktime_t to timespec conversion to ns_to_timespec function
fn ktime_to_ns(kt: i64) -> i64 {
kt
}

fn ktime_compare(cmp1: i64, cmp2: i64) -> i32 {
if cmp1 < cmp2 {
return 1i32.wrapping_neg();
}
if cmp1 > cmp2 {
return 1;
}
0
}

fn ktime_add_us(kt: i64, usec: u64) -> i64 {
ktime_add_ns!(kt, usec.wrapping_mul(1000)) as i64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
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
struct xol_area {
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
struct ldt_struct {
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
struct mem_cgroup {
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
struct hprobe {
    state: u32,
    srcu_scp: *mut srcu_ctr,
    uprobe: *mut uprobe,
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
    __slate_bits_0: __slate_bits::__SlateBits417U0,
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
struct kmem_cache {
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
    __slate_bits_0: __slate_bits::__SlateBits449U0,
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
struct kernfs_elem_symlink {
    target_kn: *mut kernfs_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
    __slate_bits_0: __slate_bits::__SlateBits472U0,
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
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
    __slate_bits_0: __slate_bits::__SlateBits474U0,
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
struct pm_subsys_data {
    lock: spinlock,
    refcount: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits499U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits499U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits499U2,
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
    __slate_bits_3: __slate_bits::__SlateBits499U3,
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
struct folio_queue {
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
struct hlist_bl_node {
    next: *mut hlist_bl_node,
    pprev: *mut *mut hlist_bl_node,
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
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
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
struct key_type {
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


fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
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


fn hrtimer_start_expires(timer: *mut hrtimer, mode: u32) {
let soft: i64 = hrtimer_get_softexpires(timer as *const hrtimer);
let hard: i64 = hrtimer_get_expires(timer as *const hrtimer);
let delta: u64 = ktime_to_ns(ktime_sub!(hard, soft)) as u64;
unsafe { hrtimer_start_range_ns(timer, soft, delta, mode) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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
struct tty_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct core_thread {
    task: *mut task_struct,
    next: *mut core_thread,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
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
    __slate_bits_0: __slate_bits::__SlateBits301U0,
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct sb_writers {
    frozen: u16,
    freeze_kcount: i32,
    freeze_ucount: i32,
    freeze_owner: *const (),
    rw_sem: [percpu_rw_semaphore; 3],
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
struct request_queue {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
}


fn mnt_idmap(mnt: &vfsmount) -> *mut mnt_idmap {
// Pairs with smp_store_release() in do_idmap_mount().
let mut __v0: *mut mnt_idmap;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
struct posix_acl {
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
struct io_uring_cmd {
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
union __SlateRecord_dynamic_debug_h_55 {
    dd_key_true: static_key_true,
    dd_key_false: static_key_false,
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
struct module_param_attrs {
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
struct _ddebug {
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
    format: *const u8,
    __slate_bits_0: __slate_bits::__SlateBits699U0,
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
struct device_attribute {
    attr: attribute,
    __slate_anon_1: __SlateRecord_device_h_108,
    __slate_anon_2: __SlateRecord_device_h_114,
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
    __slate_bits_0: __slate_bits::__SlateBits496U0,
}


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
}

fn dev_get_drvdata(dev: *const device) -> *mut () {
unsafe { (*dev).driver_data }
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
struct __SlateRecord_acpi_bus_h_249 {
    __slate_bits_0: __slate_bits::__SlateBits781U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_acpi_bus_h_281 {
    __slate_bits_0: __slate_bits::__SlateBits785U0,
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
    __slate_bits_0: __slate_bits::__SlateBits766U0,
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
    __slate_bits_0: __slate_bits::__SlateBits772U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_gpio_mapping {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_flags {
    __slate_bits_0: __slate_bits::__SlateBits773U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_dir {
    entry: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_pnp_type {
    __slate_bits_0: __slate_bits::__SlateBits777U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct platform_device {
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
    __slate_bits_0: __slate_bits::__SlateBits779U0,
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
struct acpi_device_perf_flags {
    __slate_bits_0: __slate_bits::__SlateBits783U0,
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
struct acpi_device_perf_state {
    flags: __SlateRecord_acpi_bus_h_281,
    power: u8,
    performance: u8,
    latency: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_interrupt_h_125 {
    dev_id: *mut (),
    percpu_dev_id: *mut (),
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
    __slate_bits_0: __slate_bits::__SlateBits787U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_wakeup_context {
    func: Option<unsafe extern "C" fn(*mut acpi_device_wakeup_context)>,
    dev: *mut device,
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


fn str_enable_disable(v: bool) -> *const u8 {
(if v { c"enable".as_ptr() as *mut u8 } else { c"disable".as_ptr() as *mut u8 }) as *const u8
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_taskset {
}


fn str_on_off(v: bool) -> *const u8 {
(if v { c"on".as_ptr() as *mut u8 } else { c"off".as_ptr() as *mut u8 }) as *const u8
}

fn str_yes_no(v: bool) -> *const u8 {
(if v { c"yes".as_ptr() as *mut u8 } else { c"no".as_ptr() as *mut u8 }) as *const u8
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
struct uid_gid_extent {
    first: u32,
    lower_first: u32,
    count: u32,
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
struct kernel_stat {
    irqs_sum: u64,
    softirqs: [u32; 10],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_parent_ops {
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
struct kthread_worker {
    flags: u32,
    lock: raw_spinlock,
    work_list: list_head,
    delayed_work_list: list_head,
    task: *mut task_struct,
    current_work: *mut kthread_work,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kthread_work {
    node: list_head,
    func: Option<unsafe extern "C" fn(*mut kthread_work)>,
    worker: *mut kthread_worker,
    canceling: i32,
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
    __slate_bits_0: __slate_bits::__SlateBits834U0,
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
union __SlateRecord_hdmi_h_406 {
    hdmi_type1: hdr_static_metadata,
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
struct __SlateRecord_hdmi_h_419 {
    r#type: u32,
    version: u8,
    length: u8,
    oui: u32,
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
struct drm_file {
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
struct drm_master {
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
struct fs_pin {
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
    __slate_bits_0: __slate_bits::__SlateBits614U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_mode_h_1025 {
    x: u16,
    y: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_mode_h_1036 {
    x: u16,
    y: u16,
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
union __SlateRecord_drm_mode_h_1079 {
    hdmi_metadata_type1: hdr_metadata_infoframe,
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
struct drm_privacy_screen {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_edid {
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
struct of_phandle_args {
    np: *mut device_node,
    args_count: i32,
    args: [u32; 16],
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
struct __SlateRecord_drm_connector_h_1088 {
    avi: drm_connector_hdmi_infoframe,
    hdr_drm: drm_connector_hdmi_infoframe,
    spd: drm_connector_hdmi_infoframe,
    hdmi: drm_connector_hdmi_infoframe,
}


fn usleep_range(min: u64, max: u64) {
unsafe { usleep_range_state(min, max, 2u32) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seq_buf {
    buffer: *mut u8,
    size: u64,
    len: u64,
}


fn seq_buf_clear(s: &mut seq_buf) {
unsafe {
(*(s as *mut seq_buf)).len = 0;
}
if (unsafe { (*(s as *mut seq_buf)).size }) != 0 {
unsafe {
*unsafe { (*(s as *mut seq_buf)).buffer }.wrapping_offset(0 as isize) = 0;
}
}
}

fn seq_buf_has_overflowed(s: &seq_buf) -> bool {
(unsafe { (*(s as *const seq_buf)).len }) > unsafe { (*(s as *const seq_buf)).size }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_writeback_job {
}


fn seq_buf_buffer_left(s: &seq_buf) -> u32 {
if seq_buf_has_overflowed(s) {
return 0;
}
unsafe { (*(s as *const seq_buf)).size }.wrapping_sub(unsafe { (*(s as *const seq_buf)).len }) as u32
}

fn seq_buf_str(s: &seq_buf) -> *const u8 {
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*(s as *const seq_buf)).size }) == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("629: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 629b - ., 3; .popsection\n# {0}", const 629 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_seq_buf_str_1, sym __slate_asm_string_seq_buf_str_2, const 100 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("630: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 630b - ., 4; .popsection\n# {0}", const 630 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return c"".as_ptr() as *const u8;
}
if seq_buf_buffer_left(s) != 0 {
unsafe {
*unsafe { (*(s as *const seq_buf)).buffer }.wrapping_offset((*(s as *const seq_buf)).len as isize) = 0;
}
} else {
unsafe {
*unsafe { (*(s as *const seq_buf)).buffer }.wrapping_offset(unsafe { (*(s as *const seq_buf)).size }.wrapping_sub(1) as isize) = 0;
}
}
(unsafe { (*(s as *const seq_buf)).buffer }) as *const u8
}

// 1.3
// 1.2
// 2.0
// eDP
// XXX 1.2?
// eDP 1.4
// 16-bit little-endian
// eDP 1.4
// DP 1.4
// eDP 1.4
// eDP 1.4
// eDP 1.5 & DP 2.0
// DP Forward error Correction Registers
// 1.4
// DP-HDMI2.1 PCON DSC ENCODER SUPPORT
// 0x92 through 0x9E
// DP Extended DSC Capabilities
// DP 1.4a SCR
// DP 1.4
// Link/Sink Device Status
// 1.2 MST
// 1.2 MST
// Source Device-specific
// Link/Sink Device Power Control
// eDP-specific
// eDP 1.2
// eDP 1.4
// DPRX Event Status Indicator
// same as 0x200
// 1.2
// DP 1.3
// DP 1.4
// DP 1.3
// 2.0 E11
// 2.1
// 2.0
// PCON CONFIGURE-1 FRL FOR HDMI SINK
// DP 1.3
// DP 1.3
// DP 1.3
// DP 1.3
// LTTPR: Link Training (LT)-tunable PHY Repeaters
// 1.3
// 2.0
// See DP_128B132B_SUPPORTED_LINK_RATES for values
// 2.0
// DSC Capabilities 0x60 through 0x6F
// DP 1.2
// DP 1.4
// DP 1.4
// DP 1.4
// DP 1.4
#[repr(C)]
#[derive(Clone, Copy)]
struct dp_sdp_header {
    HB0: u8,
    HB1: u8,
    HB2: u8,
    HB3: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dp_sdp {
    sdp_header: dp_sdp_header,
    db: [u8; 32],
}


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
struct __SlateRecord_drm_connector_h_2062 {
    lock: mutex,
    audio: drm_connector_hdmi_infoframe,
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
struct cec_adapter {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_clip_rect {
    x1: u16,
    y1: u16,
    x2: u16,
    y2: u16,
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
struct hdr_metadata_infoframe {
    eotf: u8,
    metadata_type: u8,
    display_primaries: [__SlateRecord_drm_mode_h_1025; 3],
    white_point: __SlateRecord_drm_mode_h_1036,
    max_display_mastering_luminance: u16,
    min_display_mastering_luminance: u16,
    max_cll: u16,
    max_fall: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_tunnel_mgr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ref_tracker {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdr_output_metadata {
    metadata_type: u32,
    __slate_anon_1: __SlateRecord_drm_mode_h_1079,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_tunnel {
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
struct drm_connector_tv_margins {
    bottom: u32,
    left: u32,
    right: u32,
    top: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_driver {
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
struct dma_fence {
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
struct drm_self_refresh_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_pending_vblank_event {
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


fn drm_connector_index(connector: *const drm_connector) -> u32 {
unsafe { (*connector).index }
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
struct drm_tile_group {
    refcount: kref,
    dev: *mut drm_device,
    id: i32,
    group_data: [u8; 9],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_list_iter {
    dev: *mut drm_device,
    conn: *mut drm_connector,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_print_h_181 {
    series: u32,
    counter: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_vsc_sdp {
    sdp_type: u8,
    revision: u8,
    length: u8,
    pixelformat: u32,
    colorimetry: u32,
    bpc: i32,
    dynamic_range: u32,
    content_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_as_sdp {
    sdp_type: u8,
    revision: u8,
    length: u8,
    vtotal: i32,
    target_rr: i32,
    duration_incr_ms: i32,
    duration_decr_ms: i32,
    target_rr_divider: bool,
    mode: u32,
    coasting_vtotal: i32,
}


fn drm_dp_max_lane_count(dpcd: *const u8) -> u8 {
(((unsafe { *dpcd.wrapping_offset(2 as isize) }) as i32) & 31) as u8
}

fn drm_dp_enhanced_frame_cap(dpcd: *const u8) -> bool {
((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) >= 17 && ((unsafe { *dpcd.wrapping_offset(2 as isize) }) as i32) & 128i32 != 0
}

fn drm_dp_is_branch(dpcd: *const u8) -> bool {
((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 1i32 != 0
}

fn drm_dp_sink_supports_dsc(dsc_dpcd: *const u8) -> bool {
((unsafe { *dsc_dpcd.wrapping_offset(0i32 as isize) }) as i32) & 1i32 != 0
}

fn drm_edp_dsc_sink_output_bpp(dsc_dpcd: *const u8) -> u16 {
(((unsafe { *dsc_dpcd.wrapping_offset(7i32 as isize) }) as i32) | (((unsafe { *dsc_dpcd.wrapping_offset(8i32 as isize) }) as i32) & 3i32) << 8i32) as u16
}

fn drm_dp_dsc_sink_max_slice_width(dsc_dpcd: *const u8) -> u32 {
// Max Slicewidth = Number of Pixels * 320
((unsafe { *dsc_dpcd.wrapping_offset(12i32 as isize) }) as i32).wrapping_mul(320) as u32
}

fn drm_dp_dsc_sink_supports_format(dsc_dpcd: *const u8, output_format: u8) -> bool {
((unsafe { *dsc_dpcd.wrapping_offset(9i32 as isize) }) as i32) & (output_format as i32) != 0
}

fn drm_dp_sink_supports_fec(fec_capable: u8) -> bool {
(fec_capable as i32) & 1i32 != 0
}

fn drm_dp_128b132b_supported(dpcd: *const u8) -> bool {
((unsafe { *dpcd.wrapping_offset(6 as isize) }) as i32) & 2i32 != 0
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

#[repr(C)]
#[derive(Clone, Copy)]
struct intel_bios_encoder_data {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_parent_interface {
}


fn drm_dp_dpcd_writeb(aux: *mut drm_dp_aux, offset: u32, mut value: u8) -> i64 {
unsafe { drm_dp_dpcd_write(aux, offset, core::ptr::addr_of_mut!(value) as *mut (), 1u64) }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_device_h_117 {
    __slate_anon_0: __SlateRecord_intel_display_device_h_118,
    bitmap: [u64; 2],
}


#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_device_h_118 {
    __slate_bits_0: __slate_bits::__SlateBits1106U0,
}


fn drm_dp_dpcd_read_byte(aux: *mut drm_dp_aux, offset: u32, valuep: *mut u8) -> i32 {
drm_dp_dpcd_read_data(aux, offset, valuep as *mut (), 1u64)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_ip_ver {
    ver: u16,
    rel: u16,
    step: u16,
}


fn drm_dp_dpcd_write_byte(aux: *mut drm_dp_aux, offset: u32, mut value: u8) -> i32 {
drm_dp_dpcd_write_data(aux, offset, core::ptr::addr_of_mut!(value) as *mut (), 1u64)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_device_h_320 {
    size: u16,
    slice_mask: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_device_h_337 {
    degamma_lut_size: u32,
    gamma_lut_size: u32,
    degamma_lut_tests: u32,
    gamma_lut_tests: u32,
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
struct i915_power_well {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_desc {
    ident: drm_dp_dpcd_ident,
    quirks: u32,
}


fn drm_dp_has_quirk(desc: *const drm_dp_desc, quirk: u32) -> bool {
((unsafe { (*desc).quirks }) as u64) & BIT__3!(quirk) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_edp_backlight_info {
    pwmgen_bit_count: u8,
    pwm_freq_pre_divider: u8,
    max: u32,
    __slate_bits_0: __slate_bits::__SlateBits1017U0,
}


fn drm_dp_cec_irq(aux: *mut drm_dp_aux) {
}

fn drm_dp_cec_register_connector(aux: *mut drm_dp_aux, connector: *mut drm_connector) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_funcs {
}


fn drm_dp_cec_unregister_connector(aux: *mut drm_dp_aux) {
}

fn drm_dp_cec_attach(aux: *mut drm_dp_aux, source_physical_address: u16) {
}

fn drm_dp_cec_unset_edid(aux: *mut drm_dp_aux) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_phy_test_params {
    link_rate: i32,
    num_lanes: u8,
    phy_pattern: u8,
    hbr2_reset: [u8; 2],
    custom80: [u8; 10],
    enhanced_frame_cap: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_tunnel_ref {
    tunnel: *mut drm_dp_tunnel,
    tracker: *mut ref_tracker,
}


fn drm_dp_tunnel_handle_irq(mgr: *mut drm_dp_tunnel_mgr, aux: *mut drm_dp_aux) -> i32 {
95i32.wrapping_neg()
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_dpll_mgr_h_252 {
    mplla: [u16; 10],
    mpllb: [u16; 11],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_c20pll_vdr_state {
    custom_width: u8,
    serdes_rate: u8,
    hdmi_rate: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_dpll_mgr_h_264 {
    c10: intel_c10pll_state,
    c20: intel_c20pll_state,
}


fn drm_dp_tunnel_128b132b_supported(tunnel: *const drm_dp_tunnel) -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_dpll_mgr_h_285 {
    i9xx: i9xx_dpll_hw_state,
    hsw: hsw_dpll_hw_state,
    skl: skl_dpll_hw_state,
    bxt: bxt_dpll_hw_state,
    icl: icl_dpll_hw_state,
    mpllb: intel_mpllb_state,
    cx0pll: intel_cx0pll_state,
    ltpll: intel_lt_phy_pll_state,
}


fn drm_dp_tunnel_128b132b_dprx_rates(tunnel: *const drm_dp_tunnel) -> u8 {
0
}

fn drm_dp_tunnel_max_dprx_rate(tunnel: *const drm_dp_tunnel) -> i32 {
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dsb {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_fbc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_global_commit {
}


fn drm_dp_tunnel_max_dprx_lane_count(tunnel: *const drm_dp_tunnel) -> i32 {
0
}

fn drm_dp_tunnel_available_bw(tunnel: *const drm_dp_tunnel) -> i32 {
1i32.wrapping_neg()
}

// DSC Rate Control Constants
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dsc_rc_range_parameters {
    range_min_qp: u8,
    range_max_qp: u8,
    range_bpg_offset: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dsc_config {
    line_buf_depth: u8,
    bits_per_component: u8,
    convert_rgb: bool,
    slice_count: u8,
    slice_width: u16,
    slice_height: u16,
    simple_422: bool,
    pic_width: u16,
    pic_height: u16,
    rc_tgt_offset_high: u8,
    rc_tgt_offset_low: u8,
    bits_per_pixel: u16,
    rc_edge_factor: u8,
    rc_quant_incr_limit1: u8,
    rc_quant_incr_limit0: u8,
    initial_xmit_delay: u16,
    initial_dec_delay: u16,
    block_pred_enable: bool,
    first_line_bpg_offset: u8,
    initial_offset: u16,
    rc_buf_thresh: [u16; 14],
    rc_range_params: [drm_dsc_rc_range_parameters; 15],
    rc_model_size: u16,
    flatness_min_qp: u8,
    flatness_max_qp: u8,
    initial_scale_value: u8,
    scale_decrement_interval: u16,
    scale_increment_interval: u16,
    nfl_bpg_offset: u16,
    slice_bpg_offset: u16,
    final_offset: u16,
    vbr_enable: bool,
    mux_word_size: u8,
    slice_chunk_size: u16,
    rc_bits: u16,
    dsc_version_minor: u8,
    dsc_version_major: u8,
    native_422: bool,
    native_420: bool,
    second_line_bpg_offset: u8,
    nsl_bpg_offset: u16,
    second_line_offset_adj: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ww_mutex {
    base: mutex,
    ctx: *mut ww_acquire_ctx,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hotplug_slot {
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
struct pcie_bwctrl_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pcie_link_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_sriov {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcec_ea {
}


fn ww_mutex_is_locked(lock: *mut ww_mutex) -> bool {
ww_mutex_base_is_locked!(unsafe { core::ptr::addr_of_mut!((*lock).base) })
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_pci_h_547 {
    sriov: *mut pci_sriov,
    physfn: *mut pci_dev,
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


fn drm_modeset_is_locked(lock: *mut drm_modeset_lock) -> bool {
ww_mutex_is_locked(unsafe { core::ptr::addr_of_mut!((*lock).mutex) })
}

fn drm_modeset_lock_assert_held(lock: *mut drm_modeset_lock) {
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*lock).mutex.base) });
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


// DRM_MODE_FMT - printf string for &struct drm_display_mode
// DRM_MODE_ARG - printf arguments for &struct drm_display_mode
// @m: display mode
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
struct i915_audio_component {
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
struct intel_audio_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_cdclk_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_cdclk_vals {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_color_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dmc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_global_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_mgr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_fbdev {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_fdi_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hotplug_irq_funcs {
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
    __slate_bits_0: __slate_bits::__SlateBits1050U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_opregion {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_overlay {
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
struct __SlateRecord_intel_display_core_h_120 {
    platdev: *mut platform_device,
    irq: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_state {
    crtc: *mut drm_crtc,
    enable: bool,
    active: bool,
    __slate_bits_0: __slate_bits::__SlateBits1062U0,
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
struct __SlateRecord_intel_display_core_h_141 {
    nssc: i32,
    ssc: i32,
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
struct __SlateRecord_intel_display_core_h_170 {
    last_jiffies: u64,
    count: i32,
    blocked_count: i32,
    state: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct sdvo_device_mapping {
    initialized: u8,
    dvo_port: u8,
    target_addr: u8,
    dvo_wiring: u8,
    i2c_pin: u8,
    ddc_pin: u8,
}


fn drm_crtc_index(crtc: *const drm_crtc) -> u32 {
unsafe { (*crtc).index }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_core_h_280 {
    hw: ilk_wm_values,
    vlv: vlv_wm_values,
    g4x: g4x_wm_values,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_314 {
    any_task_allowed: bool,
    allowed_task: *mut task_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_319 {
    lock: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_324 {
    obj: intel_global_obj,
    max: [intel_bw_info; 6],
    psf_bw: [u32; 3],
    peakbw: [u32; 8],
    num_qgv_points: u8,
    num_psf_gv_points: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_bw_info {
    deratedbw: [u32; 8],
    num_planes: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_340 {
    funcs: *const intel_cdclk_funcs,
    hw: intel_cdclk_config,
    table: *const intel_cdclk_vals,
    obj: intel_global_obj,
    max_cdclk_freq: u32,
    max_dotclk_freq: u32,
    skl_preferred_vco_freq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_357 {
    funcs: *const intel_color_funcs,
    glk_linear_degamma_lut: *mut drm_property_blob,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_364 {
    enabled_slices: u8,
    obj: intel_global_obj,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_371 {
    obj: intel_global_obj,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_375 {
    phy_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_383 {
    dmc: *mut intel_dmc,
    wakeref: *mut ref_tracker,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_388 {
    mmio_base: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_393 {
    info: *const dram_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dram_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_397 {
    instances: [*mut intel_fbc; 4],
    sys_cache: sys_cache_cfg,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sys_cache_cfg {
    lock: mutex,
    id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_408 {
    fbdev: *mut intel_fbdev,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_413 {
    funcs: *const intel_fdi_funcs,
    pll_freq: u32,
    rx_config: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_421 {
    obj_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_425 {
    mmio_base: u32,
    mutex: mutex,
    bus: [*mut intel_gmbus; 15],
    wait_queue: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_gmbus {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_443 {
    arbiter: *mut i915_hdcp_arbiter,
    comp_added: bool,
    gsc_context: *mut intel_hdcp_gsc_context,
    hdcp_mutex: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hdcp_gsc_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_457 {
    state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_467 {
    __device_info: *const intel_display_device_info,
    __runtime_info: intel_display_runtime_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_475 {
    false_color: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_479 {
    funcs: *const intel_display_irq_funcs,
    lock: spinlock,
    vlv_display_irqs_enabled: bool,
    vblank_enabled: u8,
    vblank_enable_count: i32,
    vblank_status_last_notified: bool,
    vblank_notify_work: work_struct,
    vlv_imr_mask: u32,
    ilk_de_imr_mask: u32,
    de_pipe_imr_mask: [u32; 4],
    pipestat_irq_mask: [u32; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_irq_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_521 {
    funcs: *const intel_modeset_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_526 {
    linetime: [u16; 4],
    disable: [bool; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_532 {
    waitqueue: wait_queue_head,
    lock: mutex,
    obj: intel_global_obj,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_541 {
    domains: i915_power_domains,
    dc3co: intel_dc3co_state,
    chv_phy_control: u32,
    chv_phy_assert: [bool; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_553 {
    mmio_base: u32,
    mutex: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_560 {
    broadcast_rgb: *mut drm_property,
    force_audio: *mut drm_property,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_565 {
    mask: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_569 {
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_573 {
    modeset_state: *mut drm_atomic_commit,
    reset_ctx: drm_modeset_acquire_ctx,
    saveDSPARB: u32,
    saveSWF0: [u32; 16],
    saveSWF1: [u32; 16],
    saveSWF3: [u32; 3],
    saveGCDGMBUS: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_585 {
    status: u32,
    block_time_us: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_596 {
    lock: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_601 {
    phy_failed_calibration: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_609 {
    chv_dpll_md: [u32; 4],
    bxt_phy_grc: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_619 {
    hpll_freq: u32,
    czclk_freq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_624 {
    modeset: *mut workqueue_struct,
    flip: *mut workqueue_struct,
    cleanup: *mut workqueue_struct,
    unordered: *mut workqueue_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_i915_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_runtime_pm {
}


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
struct intel_gt {
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
struct i915_mmio_range {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_uncore_forcewake_domain {
    uncore: *mut intel_uncore,
    id: u32,
    mask: u32,
    wake_count: u32,
    active: bool,
    timer: hrtimer,
    reg_set: *mut u32,
    reg_ack: *mut u32,
}


fn drm_connector_helper_add(connector: *mut drm_connector, funcs: *const drm_connector_helper_funcs) {
unsafe {
(*connector).helper_private = funcs;
}
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


fn fxp_q4_from_int(val_int: i32) -> i32 {
val_int << 4i32
}

fn fxp_q4_to_int(val_q4: i32) -> i32 {
val_q4 >> 4i32
}

fn fxp_q4_to_frac(val_q4: i32) -> i32 {
val_q4 & 15
}

fn drm_debug_enabled_raw(category: u32) -> bool {
unlikely__3!((unsafe { __drm_debug }) & BIT__3!(category)) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_pm_qos_h_109 {
    pnode: plist_node,
    flr: pm_qos_flags_request,
    freq: freq_qos_request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_msg {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_desc {
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


fn __drm_to_dev(drm: *const drm_device) -> *mut device {
if !drm.is_null() { unsafe { (*drm).dev } } else { core::ptr::null_mut::<device>() }
}

// Helper for struct drm_device based logging.
// Helper for struct drm_device based WARNs
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_i915_reg_defs_h_14 {
    reg: u32,
}


// These macros can be used on either i915_reg_t or i915_mcr_reg_t since they're
// simply operations on the register's offset and don't care about the MCR vs
// non-MCR nature of the register.
#[repr(C)]
#[derive(Clone, Copy)]
struct intel_cdclk_config {
    cdclk: u32,
    vco: u32,
    r#ref: u32,
    bypass: u32,
    voltage_level: u8,
    joined_mbus: bool,
}


// Platform group aliases
// g45 and gm45
// mobile platforms
// discrete graphics
// Display ver 2
// Display ver 3
// Display ver 4
// Display ver 5
// Display ver 6
// Display ver 7
// Display ver 8
// Display ver 9
// Display ver 11
// Display ver 12
// Display ver 13
// Display ver 14 (based on GMD ID)
// Display ver 20 (based on GMD ID)
// Display ver 14.1 (based on GMD ID)
// Display ver 30 (based on GMD ID)
// Display ver 35 (based on GMD ID)
#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_platforms {
    __slate_anon_0: __SlateRecord_intel_display_device_h_117,
}


// Keep in alphabetical order
#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_runtime_info {
    ip: intel_display_ip_ver,
    step: i32,
    step_name: [u8; 3],
    rawclk_freq: u32,
    pipe_mask: u8,
    cpu_transcoder_mask: u16,
    port_mask: u16,
    num_sprites: [u8; 4],
    num_scalers: [u8; 4],
    fbc_mask: u8,
    has_hdcp: bool,
    has_dmc: bool,
    has_dsc: bool,
    edp_typec_support: bool,
    has_dbuf_overlap_detection: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_device_info {
    __runtime_defaults: intel_display_runtime_info,
    abox_mask: u8,
    dbuf: __SlateRecord_intel_display_device_h_320,
    __slate_bits_0: __slate_bits::__SlateBits1109U0,
    mmio_offset: u32,
    pipe_offsets: [u32; 9],
    trans_offsets: [u32; 9],
    cursor_offsets: [u32; 4],
    color: __SlateRecord_intel_display_device_h_337,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gpio_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_params {
    dmc_firmware_path: *mut u8,
    vbt_firmware: *mut u8,
    lvds_channel_mode: i32,
    panel_use_ssc: i32,
    vbt_sdvo_panel_type: i32,
    enable_dc: i32,
    enable_dpt: bool,
    enable_dsb: bool,
    enable_flipq: bool,
    enable_sagv: bool,
    disable_power_well: i32,
    enable_ips: bool,
    invert_brightness: i32,
    edp_vswing: i32,
    enable_dpcd_backlight: i32,
    load_detect_test: bool,
    force_reset_modeset_test: bool,
    disable_display: bool,
    verbose_state_checks: bool,
    nuclear_pageflip: bool,
    enable_dp_mst: bool,
    enable_fbc: i32,
    enable_psr: i32,
    enable_dc_balance: bool,
    enable_panel_replay: i32,
    psr_safest_params: bool,
    enable_psr2_sel_fetch: bool,
    enable_dmc_wl: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_power_domain_mask {
    bits: [u64; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dc3co_state {
    lock: mutex,
    allowed: bool,
    trigger: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i915_power_domains {
    initializing: bool,
    display_core_suspended: bool,
    dc3co_to_dc6: bool,
    power_well_count: i32,
    dc_state: u32,
    target_dc_state: u32,
    allowed_dc_mask: u32,
    init_wakeref: *mut ref_tracker,
    disable_wakeref: *mut ref_tracker,
    lock: mutex,
    domain_use_count: [i32; 76],
    async_put_work: delayed_work,
    async_put_wakeref: *mut ref_tracker,
    async_put_domains: [intel_power_domain_mask; 2],
    async_put_next_delay: i32,
    power_wells: *mut i915_power_well,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_power_domain_set {
    mask: intel_power_domain_mask,
}


fn intel_display_power_put_async(display: *mut intel_display, domain: u32, wakeref: *mut ref_tracker) {
unsafe { __intel_display_power_put_async(display, domain, ERR_PTR(2i32.wrapping_neg() as i64) as *mut ref_tracker, 1i32.wrapping_neg()) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dmc_wl {
    lock: spinlock,
    enabled: bool,
    taken: bool,
    refcount: refcount_struct,
    dc_state: u32,
    work: delayed_work,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_pwm_h_348 {
    nonatomic_lock: mutex,
    atomic_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i9xx_dpll_hw_state {
    dpll: u32,
    dpll_md: u32,
    fp0: u32,
    fp1: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hsw_dpll_hw_state {
    wrpll: u32,
    spll: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct skl_dpll_hw_state {
    ctrl1: u32,
    cfgcr1: u32,
    cfgcr2: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bxt_dpll_hw_state {
    ebb0: u32,
    ebb4: u32,
    pll0: u32,
    pll1: u32,
    pll2: u32,
    pll3: u32,
    pll6: u32,
    pll8: u32,
    pll9: u32,
    pll10: u32,
    pcsdw12: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct icl_dpll_hw_state {
    cfgcr0: u32,
    cfgcr1: u32,
    div0: u32,
    mg_refclkin_ctl: u32,
    mg_clktop2_coreclkctl1: u32,
    mg_clktop2_hsclkctl: u32,
    mg_pll_div0: u32,
    mg_pll_div1: u32,
    mg_pll_lf: u32,
    mg_pll_frac_lock: u32,
    mg_pll_ssc: u32,
    mg_pll_bias: u32,
    mg_pll_tdc_coldst_bias: u32,
    mg_pll_bias_mask: u32,
    mg_pll_tdc_coldst_bias_mask: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_mpllb_state {
    clock: u32,
    ref_control: u32,
    mpllb_cp: u32,
    mpllb_div: u32,
    mpllb_div2: u32,
    mpllb_fracn1: u32,
    mpllb_fracn2: u32,
    mpllb_sscen: u32,
    mpllb_sscstep: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_c10pll_state {
    tx: u8,
    cmn: u8,
    pll: [u8; 20],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_c20pll_state {
    tx: [u16; 3],
    cmn: [u16; 4],
    __slate_anon_2: __SlateRecord_intel_dpll_mgr_h_252,
    vdr: intel_c20pll_vdr_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_cx0pll_state {
    __slate_anon_0: __SlateRecord_intel_dpll_mgr_h_264,
    lane_count: i32,
    ssc_enabled: bool,
    use_c10: bool,
    tbt_mode: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_lt_phy_pll_state {
    addr_msb: [u8; 13],
    addr_lsb: [u8; 13],
    data: [[u8; 4]; 13],
    config: [u8; 3],
    ssc_enabled: bool,
    tbt_mode: bool,
    lane_count: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_hw_state {
    __slate_anon_0: __SlateRecord_intel_dpll_mgr_h_285,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_state {
    pipe_mask: u8,
    hw_state: intel_dpll_hw_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dpll_info {
    name: *const u8,
    funcs: *const intel_dpll_funcs,
    id: i32,
    power_domain: u32,
    always_on: bool,
    is_alt_port_dpll: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll {
    state: intel_dpll_state,
    index: u8,
    active_mask: u8,
    on: bool,
    info: *const dpll_info,
    wakeref: *mut ref_tracker,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_fourcc_h_77 {
    cpp: [u8; 4],
    char_per_block: [u8; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_gem_object {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_global_state_funcs {
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut intel_global_obj) -> *mut intel_global_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut intel_global_obj, *mut intel_global_state)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct backlight_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct display_timing {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_global_obj {
    head: list_head,
    state: *mut intel_global_state,
    funcs: *const intel_global_state_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_global_state {
    obj: *mut intel_global_obj,
    state: *mut intel_atomic_state,
    commit: *mut intel_global_commit,
    r#ref: kref,
    changed: bool,
    serialized: bool,
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
    __slate_bits_0: __slate_bits::__SlateBits1146U0,
    kobj: kobject,
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
    __slate_bits_0: __slate_bits::__SlateBits797U0,
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
    __slate_bits_1: __slate_bits::__SlateBits797U1,
    d3hot_delay: u32,
    d3cold_delay: u32,
    l1ss: u16,
    link_state: *mut pcie_link_state,
    __slate_bits_2: __slate_bits::__SlateBits797U2,
    error_state: u32,
    dev: device,
    cfg_size: i32,
    irq: u32,
    resource: [resource; 11],
    driver_exclusive_resource: resource,
    __slate_bits_3: __slate_bits::__SlateBits797U3,
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
    __slate_bits_4: __slate_bits::__SlateBits797U4,
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
    __slate_bits_0: __slate_bits::__SlateBits446U0,
}


// Error values that may be returned by PCI functions
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
// These helpers provide future and backwards compatibility
// for accessing popular PCI BAR info
fn pci_get_drvdata(pdev: &mut pci_dev) -> *mut () {
dev_get_drvdata((unsafe { core::ptr::addr_of_mut!((*(pdev as *mut pci_dev)).dev) }) as *const device)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ilk_wm_values {
    wm_pipe: [u32; 3],
    wm_lp: [u32; 3],
    wm_lp_spr: [u32; 3],
    enable_fbc_wm: bool,
    partitioning: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct g4x_pipe_wm {
    plane: [u16; 8],
    fbc: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct g4x_sr_wm {
    plane: u16,
    cursor: u16,
    fbc: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vlv_wm_ddl_values {
    plane: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vlv_wm_values {
    pipe: [g4x_pipe_wm; 3],
    sr: g4x_sr_wm,
    ddl: [vlv_wm_ddl_values; 3],
    level: u8,
    cxsr: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct g4x_wm_values {
    pipe: [g4x_pipe_wm; 2],
    sr: g4x_sr_wm,
    hpll: g4x_sr_wm,
    cxsr: bool,
    hpll_en: bool,
    fbc_en: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct skl_ddb_entry {
    start: u16,
    end: u16,
}


// Amount of PSF GV points, BSpec precisely defines this
#[repr(C)]
#[derive(Clone, Copy)]
struct intel_modeset_funcs {
    get_pipe_config: Option<unsafe extern "C" fn(*mut intel_crtc, *mut intel_crtc_state) -> bool>,
    get_initial_plane_config: Option<unsafe extern "C" fn(*mut intel_crtc, *mut intel_initial_plane_config)>,
    fixup_initial_plane_config: Option<unsafe extern "C" fn(*mut intel_crtc, *const intel_initial_plane_config) -> bool>,
    crtc_enable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc)>,
    crtc_disable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc)>,
    commit_modeset_enables: Option<unsafe extern "C" fn(*mut intel_atomic_state)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_wm_funcs {
    update_wm: Option<unsafe extern "C" fn(*mut intel_display)>,
    compute_watermarks: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc) -> i32>,
    initial_watermarks: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc)>,
    atomic_update_watermarks: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc)>,
    optimize_watermarks: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc)>,
    compute_global_watermarks: Option<unsafe extern "C" fn(*mut intel_atomic_state) -> i32>,
    get_hw_state: Option<unsafe extern "C" fn(*mut intel_display)>,
    sanitize: Option<unsafe extern "C" fn(*mut intel_display)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_50 {
    __slate_bits_0: __slate_bits::__SlateBits1348U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_84 {
    __slate_bits_0: __slate_bits::__SlateBits1349U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_102 {
    __slate_bits_0: __slate_bits::__SlateBits1350U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_113 {
    __slate_bits_0: __slate_bits::__SlateBits1351U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_122 {
    __slate_bits_0: __slate_bits::__SlateBits1352U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_140 {
    __slate_bits_0: __slate_bits::__SlateBits1353U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_audio_state {
    encoder: *mut intel_encoder,
    eld: [u8; 128],
    needs_cpu_transcoder_id: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cec_notifier {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_ddi_buf_trans {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_link_caps {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_link_training {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_global_objs_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hdcp_shim {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_tc_port {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_audio {
    funcs: *const intel_audio_funcs,
    component: *mut i915_audio_component,
    component_registered: bool,
    mutex: mutex,
    power_refcount: i32,
    freq_cntrl: u32,
    state: [intel_audio_state; 9],
    lpe: __SlateRecord_intel_display_core_h_120,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_global {
    funcs: *const intel_dpll_global_funcs,
    lock: mutex,
    num_dpll: i32,
    dplls: [intel_dpll; 9],
    mgr: *const intel_dpll_mgr,
    ref_clks: __SlateRecord_intel_display_core_h_141,
    pch_ssc_use: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_frontbuffer_tracking {
    lock: spinlock,
    busy_bits: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_345 {
    max_link_rate: i32,
    rate: i32,
    lanes: i32,
    preemphasis: i32,
    vswing: i32,
    bpp: i32,
    pps: intel_pps_delays,
    drrs_msa_timing_delay: u8,
    low_vswing: bool,
    hobl: bool,
    dsc_disable: bool,
    pipe_joiner_enable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_360 {
    enable: bool,
    full_link: bool,
    require_aux_wakeup: bool,
    idle_frames: i32,
    tp1_wakeup_time_us: i32,
    tp2_tp3_wakeup_time_us: i32,
    psr2_tp2_tp3_wakeup_time_us: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_370 {
    pwm_freq_hz: u16,
    brightness_precision_bits: u16,
    hdr_dpcd_refresh_timeout: u16,
    present: bool,
    active_low_pwm: bool,
    min_brightness: u8,
    controller: i8,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_382 {
    panel_id: u16,
    config: *mut mipi_config,
    pps: *mut mipi_pps_data,
    bl_ports: u16,
    cabc_ports: u16,
    seq_version: u8,
    size: u32,
    data: *mut u8,
    sequence: [*const u8; 12],
    deassert_seq: *mut u8,
    orientation: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_407 {
    present: bool,
    level: u32,
    min: u32,
    max: u32,
    enabled: bool,
    combination_mode: bool,
    active_low_pwm: bool,
    alternate_pwm_increment: bool,
    pwm_level_min: u32,
    pwm_level_max: u32,
    pwm_enabled: bool,
    util_pin_active_low: bool,
    controller: u8,
    pwm: *mut pwm_device,
    pwm_state: pwm_state,
    edp: __SlateRecord_intel_display_types_h_427,
    device: *mut backlight_device,
    funcs: *const intel_panel_bl_funcs,
    pwm_funcs: *const intel_panel_bl_funcs,
    power: Option<unsafe extern "C" fn(*mut intel_connector, bool)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_types_h_427 {
    vesa: __SlateRecord_intel_display_types_h_428,
    intel_cap: __SlateRecord_intel_display_types_h_432,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_428 {
    info: drm_edp_backlight_info,
    luminance_control_support: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_432 {
    sdr_uses_aux: bool,
    supports_2084_decode: bool,
    supports_2020_gamut: bool,
    supports_segmented_backlight: bool,
    supports_sdp_colorimetry: bool,
    supports_tone_mapping: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hotplug {
    funcs: *const intel_hotplug_irq_funcs,
    hotplug_work: delayed_work,
    hpd: *const u32,
    pch_hpd: *const u32,
    stats: [__SlateRecord_intel_display_core_h_170; 15],
    event_bits: u32,
    retry_bits: u32,
    reenable_work: delayed_work,
    long_hpd_pin_mask: u32,
    short_hpd_pin_mask: u32,
    dig_port_work: work_struct,
    poll_init_work: work_struct,
    poll_enabled: bool,
    detection_work_enabled: bool,
    hpd_storm_threshold: u32,
    hpd_short_storm_enabled: u8,
    oob_hotplug_last_state: u64,
    dp_wq: *mut workqueue_struct,
    ignore_long_hpd: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_559 {
    dsc_decompression_aux: *mut drm_dp_aux,
    dsc_dpcd: [u8; 16],
    fec_capability: u8,
    __slate_bits_0: __slate_bits::__SlateBits1382U0,
    dsc_branch_caps: __SlateRecord_intel_display_types_h_568,
    panel_replay_caps: __SlateRecord_intel_display_types_h_576,
    psr_caps: __SlateRecord_intel_display_types_h_588,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_568 {
    overall_throughput: __SlateRecord_intel_display_types_h_569,
    max_line_width: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_569 {
    rgb_yuv444: i32,
    yuv422_420: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_576 {
    dpcd: [u8; 7],
    support: bool,
    su_support: bool,
    dsc_support: u32,
    su_w_granularity: u16,
    su_y_granularity: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_588 {
    dpcd: [u8; 2],
    intel_wa_dpcd: u8,
    support: bool,
    su_support: bool,
    su_w_granularity: u16,
    su_y_granularity: u16,
    sync_latency: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_602 {
    port: *mut drm_dp_mst_port,
    dp: *mut intel_dp,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_607 {
    force_bpp_x16: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_tunnel_inherited_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i915_vma {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_vbt_data {
    version: u16,
    __slate_bits_0: __slate_bits::__SlateBits1200U0,
    lvds_ssc_freq: i32,
    orientation: i32,
    override_afc_startup: bool,
    override_afc_startup_val: u8,
    crt_ddc_pin: i32,
    display_devices: list_head,
    bdb_blocks: list_head,
    sdvo_mappings: [sdvo_device_mapping; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_wm {
    funcs: *const intel_wm_funcs,
    pri_latency: [u16; 5],
    spr_latency: [u16; 5],
    cur_latency: [u16; 5],
    skl_latency: [u16; 8],
    __slate_anon_5: __SlateRecord_intel_display_core_h_280,
    num_levels: u8,
    wm_mutex: mutex,
    ipc_enabled: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display {
    drm: *mut drm_device,
    platform: intel_display_platforms,
    pch_type: i32,
    parent: *const intel_display_parent_interface,
    pipe_list: list_head,
    access: __SlateRecord_intel_display_core_h_314,
    backlight: __SlateRecord_intel_display_core_h_319,
    bw: __SlateRecord_intel_display_core_h_324,
    cdclk: __SlateRecord_intel_display_core_h_340,
    color: __SlateRecord_intel_display_core_h_357,
    dbuf: __SlateRecord_intel_display_core_h_364,
    dbuf_bw: __SlateRecord_intel_display_core_h_371,
    dkl: __SlateRecord_intel_display_core_h_375,
    dmc: __SlateRecord_intel_display_core_h_383,
    dsi: __SlateRecord_intel_display_core_h_388,
    dram: __SlateRecord_intel_display_core_h_393,
    fbc: __SlateRecord_intel_display_core_h_397,
    fbdev: __SlateRecord_intel_display_core_h_408,
    fdi: __SlateRecord_intel_display_core_h_413,
    global: __SlateRecord_intel_display_core_h_421,
    gmbus: __SlateRecord_intel_display_core_h_425,
    hdcp: __SlateRecord_intel_display_core_h_443,
    hti: __SlateRecord_intel_display_core_h_457,
    info: __SlateRecord_intel_display_core_h_467,
    ips: __SlateRecord_intel_display_core_h_475,
    irq: __SlateRecord_intel_display_core_h_479,
    modeset: __SlateRecord_intel_display_core_h_521,
    pkgc: __SlateRecord_intel_display_core_h_526,
    pmdemand: __SlateRecord_intel_display_core_h_532,
    power: __SlateRecord_intel_display_core_h_541,
    pps: __SlateRecord_intel_display_core_h_553,
    properties: __SlateRecord_intel_display_core_h_560,
    quirks: __SlateRecord_intel_display_core_h_565,
    reset: __SlateRecord_intel_display_core_h_569,
    restore: __SlateRecord_intel_display_core_h_573,
    sagv: __SlateRecord_intel_display_core_h_585,
    sbi: __SlateRecord_intel_display_core_h_596,
    snps: __SlateRecord_intel_display_core_h_601,
    state: __SlateRecord_intel_display_core_h_609,
    vlv_clock: __SlateRecord_intel_display_core_h_619,
    wq: __SlateRecord_intel_display_core_h_624,
    dp_tunnel_mgr: *mut drm_dp_tunnel_mgr,
    audio: intel_audio,
    dpll: intel_dpll_global,
    fb_tracking: intel_frontbuffer_tracking,
    hotplug: intel_hotplug,
    opregion: *mut intel_opregion,
    overlay: *mut intel_overlay,
    params: intel_display_params,
    vbt: intel_vbt_data,
    wl: intel_dmc_wl,
    wm: intel_wm,
    psr_dc5_dc6_wa_work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_uncore_mmio_debug {
    lock: spinlock,
    unclaimed_mmio_check: i32,
    saved_mmio_check: i32,
    suspend_count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_uncore_fw_get {
    force_wake_get: Option<unsafe extern "C" fn(*mut intel_uncore, u32)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_uncore_funcs {
    read_fw_domains: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14) -> u32>,
    write_fw_domains: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14) -> u32>,
    mmio_readb: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, bool) -> u8>,
    mmio_readw: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, bool) -> u16>,
    mmio_readl: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, bool) -> u32>,
    mmio_readq: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, bool) -> u64>,
    mmio_writeb: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, u8, bool)>,
    mmio_writew: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, u16, bool)>,
    mmio_writel: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, u32, bool)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_853 {
    wm0: skl_wm_level,
    trans_wm: skl_wm_level,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_forcewake_range {
    start: u32,
    end: u32,
    domains: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_uncore {
    regs: *mut (),
    i915: *mut drm_i915_private,
    gt: *mut intel_gt,
    rpm: *mut intel_runtime_pm,
    lock: spinlock,
    gsi_offset: u32,
    flags: u32,
    fw_domains_table: *const intel_forcewake_range,
    fw_domains_table_entries: u32,
    shadowed_reg_table: *const i915_mmio_range,
    shadowed_reg_table_entries: u32,
    pmic_bus_access_nb: notifier_block,
    fw_get_funcs: *const intel_uncore_fw_get,
    funcs: intel_uncore_funcs,
    fifo_count: u32,
    fw_domains: u32,
    fw_domains_active: u32,
    fw_domains_timer: u32,
    fw_domains_saved: u32,
    fw_domain: [*mut intel_uncore_forcewake_domain; 16],
    user_forcewake_count: u32,
    debug: *mut intel_uncore_mmio_debug,
}


// register access functions
fn intel_uncore_read(uncore: *mut intel_uncore, reg: __SlateRecord_i915_reg_defs_h_14) -> u32 {
unsafe { unsafe { (*uncore).funcs.mmio_readl }.unwrap()(uncore, reg, true) }
}

fn intel_uncore_read_notrace(uncore: *mut intel_uncore, reg: __SlateRecord_i915_reg_defs_h_14) -> u32 {
unsafe { unsafe { (*uncore).funcs.mmio_readl }.unwrap()(uncore, reg, false) }
}

fn intel_uncore_write(uncore: *mut intel_uncore, reg: __SlateRecord_i915_reg_defs_h_14, val: u32) {
unsafe { unsafe { (*uncore).funcs.mmio_writel }.unwrap()(uncore, reg, val, true) };
}

// These are untraced mmio-accessors that are only valid to be used inside
// critical sections, such as inside IRQ handlers, where forcewake is explicitly
// controlled.
//
// Think twice, and think again, before using these.
//
// As an example, these accessors can possibly be used between:
//
// spin_lock_irq(&uncore->lock);
// intel_uncore_forcewake_get__locked();
//
// and
//
// intel_uncore_forcewake_put__locked();
// spin_unlock_irq(&uncore->lock);
//
//
// Note: some registers may not need forcewake held, so
// intel_uncore_forcewake_{get,put} can be omitted, see
// intel_uncore_forcewake_for_reg().
//
// Certain architectures will die if the same cacheline is concurrently accessed
// by different clients (e.g. on Ivybridge). Access to registers should
// therefore generally be serialised, by either the dev_priv->uncore.lock or
// a more localised lock guarding all access to that bank of registers.
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_types_h_899 {
    ilk: __SlateRecord_intel_display_types_h_922,
    skl: __SlateRecord_intel_display_types_h_927,
    vlv: __SlateRecord_intel_display_types_h_948,
    g4x: __SlateRecord_intel_display_types_h_955,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_922 {
    intermediate: intel_pipe_wm,
    optimal: intel_pipe_wm,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_927 {
    raw: skl_pipe_wm,
    optimal: skl_pipe_wm,
    ddb: skl_ddb_entry,
    plane_ddb: [skl_ddb_entry; 8],
    plane_ddb_y: [skl_ddb_entry; 8],
    plane_min_ddb: [u16; 8],
    plane_interim_ddb: [u16; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_948 {
    raw: [g4x_pipe_wm; 3],
    intermediate: vlv_wm_state,
    optimal: vlv_wm_state,
    fifo_state: vlv_fifo_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_955 {
    raw: [g4x_pipe_wm; 3],
    intermediate: g4x_wm_state,
    optimal: g4x_wm_state,
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
fn __to_uncore(display: *mut intel_display) -> *mut intel_uncore {
unsafe { to_intel_uncore((*display).drm) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1036 {
    active: bool,
    enable: bool,
    degamma_lut: *mut drm_property_blob,
    gamma_lut: *mut drm_property_blob,
    ctm: *mut drm_property_blob,
    mode: drm_display_mode,
    pipe_mode: drm_display_mode,
    adjusted_mode: drm_display_mode,
    background_color: u32,
    scaling_filter: u32,
    sharpness_strength: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct icl_port_dpll {
    pll: *mut intel_dpll,
    hw_state: intel_dpll_hw_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1171 {
    ctrl: u32,
    div: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1230 {
    control: u32,
    pgm_ratios: u32,
    lvds_border_bits: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1237 {
    casf: intel_casf,
    dst: drm_rect,
    enabled: bool,
    force_thru: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_types_h_1277 {
    csc_mode: u32,
    cgm_mode: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1303 {
    enable: u32,
    gcp: u32,
    avi: hdmi_infoframe,
    spd: hdmi_infoframe,
    hdmi: hdmi_infoframe,
    drm: hdmi_infoframe,
    vsc: drm_dp_vsc_sdp,
    as_sdp: drm_dp_as_sdp,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1347 {
    compression_enabled_on_link: bool,
    compression_enable: bool,
    slice_config: intel_dsc_slice_config,
    compressed_bpp_x16: u16,
    config: drm_dsc_config,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dsc_slice_config {
    pipes_per_line: i32,
    streams_per_pipe: i32,
    slices_per_stream: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1401 {
    enable: bool,
    in_range: bool,
    pipeline_full: u8,
    flipline: u16,
    vmin: u16,
    vmax: u16,
    guardband: u16,
    vsync_end: u32,
    vsync_start: u32,
    dc_balance: __SlateRecord_intel_display_types_h_1406,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1406 {
    enable: bool,
    vmin: u16,
    vmax: u16,
    guardband: u16,
    slope: u16,
    max_increase: u16,
    max_decrease: u16,
    vblank_target: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1416 {
    enable: bool,
    cmrr_n: u64,
    cmrr_m: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1422 {
    enable: bool,
    link_count: u8,
    pixel_overlap: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1437 {
    io_wake_lines: u8,
    fast_wake_lines: u8,
    check_entry_lines: u8,
    aux_less_wake_lines: u8,
    silence_period_sym_clocks: u8,
    lfps_half_cycle_num_of_syms: u8,
}


fn intel_de_read(display: *mut intel_display, reg: __SlateRecord_i915_reg_defs_h_14) -> u32 {
let mut val: u32 = 0;
unsafe { intel_dmc_wl_get(display, reg) };
val = intel_uncore_read(__to_uncore(display), reg);
unsafe { intel_dmc_wl_put(display, reg) };
val
}

fn intel_de_posting_read(display: *mut intel_display, reg: __SlateRecord_i915_reg_defs_h_14) {
unsafe { intel_dmc_wl_get(display, reg) };
intel_uncore_read_notrace(__to_uncore(display), reg);
unsafe { intel_dmc_wl_put(display, reg) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1535 {
    active: __SlateRecord_intel_display_types_h_1537,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_types_h_1537 {
    ilk: intel_pipe_wm,
    vlv: vlv_wm_state,
    g4x: g4x_wm_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1544 {
    mutex: mutex,
    work: delayed_work,
    refresh_rate: u32,
    frontbuffer_bits: u32,
    busy_frontbuffer_bits: u32,
    cpu_transcoder: i32,
    m_n: intel_link_m_n,
    m2_n2: intel_link_m_n,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1554 {
    flip_count: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1560 {
    start_vbl_count: u32,
    start_vbl_time: i64,
    min_vbl: i32,
    max_vbl: i32,
    scanline_start: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1588 {
    enabled: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1683 {
    r#type: u32,
    max_tmds_clock: i32,
}


fn intel_de_write(display: *mut intel_display, reg: __SlateRecord_i915_reg_defs_h_14, val: u32) {
unsafe { intel_dmc_wl_get(display, reg) };
intel_uncore_write(__to_uncore(display), reg, val);
unsafe { intel_dmc_wl_put(display, reg) };
}

fn msecs_to_jiffies_timeout(m: u32) -> u64 {
let j: u64 = msecs_to_jiffies(m);
min_t!(u64, 4611686018427387902i64, j.wrapping_add(1))
}

fn wait_remaining_ms_from_jiffies(timestamp_jiffies: u64, to_wait_ms: i32) {
let mut target_jiffies: u64 = 0;
let mut tmp_jiffies: u64 = 0;
let mut remaining_jiffies: u64 = 0;
// Don't re-read the value of "jiffies" every time since it may change
// behind our back and break the math.
tmp_jiffies = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) };
target_jiffies = timestamp_jiffies.wrapping_add(msecs_to_jiffies_timeout(to_wait_ms as u32));
if time_after__2!(target_jiffies, tmp_jiffies) {
remaining_jiffies = target_jiffies.wrapping_sub(tmp_jiffies);
while remaining_jiffies != 0 {
remaining_jiffies = (unsafe { schedule_timeout_uninterruptible(remaining_jiffies as i64) }) as u64;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pm_qos_constraints {
    list: plist_head,
    target_value: i32,
    default_value: i32,
    no_constraint_value: i32,
    r#type: u32,
    notifiers: *mut blocking_notifier_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_qos_request {
    node: plist_node,
    qos: *mut pm_qos_constraints,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_types_h_1728 {
    vlv_pps_pipe: i32,
    pps_idx: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_qos_flags_request {
    node: list_head,
    flags: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_qos_flags {
    list: list_head,
    effective_flags: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1848 {
    active: bool,
    mst_probed_lane_count: i32,
    mst_probed_rate: i32,
    training: *mut intel_dp_link_training,
    caps: *mut intel_dp_link_caps,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1885 {
    stream_encoders: [*mut intel_dp_mst_encoder; 4],
    mgr: drm_dp_mst_topology_mgr,
    active_streams: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1919 {
    min_tmds_clock: i32,
    max_tmds_clock: i32,
    max_dotclock: i32,
    pcon_max_frl_bw: i32,
    max_bpc: u8,
    ycbcr_444_to_420: bool,
    ycbcr420_passthrough: bool,
    rgb_to_ycbcr: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1953 {
    transcoder: i32,
    lock: mutex,
    lobf_disable_debug: bool,
    sink_alpm_error: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1963 {
    mask: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct freq_constraints {
    min_freq: pm_qos_constraints,
    min_freq_notifiers: blocking_notifier_head,
    max_freq: pm_qos_constraints,
    max_freq_notifiers: blocking_notifier_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_2000 {
    mutex: mutex,
    num_streams: u32,
    auth_status: bool,
    port_data: hdcp_port_data,
    mst_type1_capable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct freq_qos_request {
    r#type: u32,
    pnode: plist_node,
    qos: *mut freq_constraints,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos_request {
    r#type: u32,
    data: __SlateRecord_pm_qos_h_109,
    dev: *mut device,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
    resume_latency: pm_qos_constraints,
    latency_tolerance: pm_qos_constraints,
    freq: freq_constraints,
    flags: pm_qos_flags,
    resume_latency_req: *mut dev_pm_qos_request,
    latency_tolerance_req: *mut dev_pm_qos_request,
    flags_req: *mut dev_pm_qos_request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdev {
    kobj: kobject,
    owner: *mut module,
    ops: *const file_operations,
    list: list_head,
    dev: u32,
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_h_32 {
    min_bpp: i32,
    max_bpp: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_h_36 {
    min_bpp_x16: i32,
    max_bpp_x16: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_232 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_245 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_262 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_262_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_580 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_583 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_588 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_600 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_603 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_607 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_610 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_735 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_749 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_2951 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_4112 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_4748 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_5351 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_5399 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dp_c_5401 {
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
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits614U0 {
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
    pub struct __SlateBits449U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits298U0 {
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
    pub struct __SlateBits73U0 {
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
    pub struct __SlateBits73U1 {
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
    pub struct __SlateBits417U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits472U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits834U0 {
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
    pub struct __SlateBits301U0 {
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
    pub struct __SlateBits73U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits109U0 {
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
    pub struct __SlateBits128U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits499U0 {
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
    pub struct __SlateBits496U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits499U1 {
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
    pub struct __SlateBits499U2 {
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
    pub struct __SlateBits499U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits474U0 {
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
    pub struct __SlateBits108U0 {
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
    pub struct __SlateBits992U0 {
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
    pub struct __SlateBits1062U0 {
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
    pub struct __SlateBits1050U0 {
        #[bits(1)]
        pub color_mgmt_changed: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits446U0 {
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
    pub struct __SlateBits1146U0 {
        #[bits(1)]
        pub per_func_slot: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits797U0 {
        #[bits(3)]
        pub pcie_mpss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits797U1 {
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
    pub struct __SlateBits797U2 {
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
    pub struct __SlateBits797U3 {
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
    pub struct __SlateBits797U4 {
        #[bits(1)]
        pub pasid_required: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits772U0 {
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
    pub struct __SlateBits773U0 {
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
    pub struct __SlateBits777U0 {
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
    pub struct __SlateBits779U0 {
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
    pub struct __SlateBits781U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub explicit_set: u8,
        #[bits(6)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits787U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub notifier_present: u8,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits783U0 {
        #[bits(8)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits785U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(7)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits766U0 {
        #[bits(1)]
        pub enabled: bool,
        #[bits(1)]
        pub demand_offline: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits699U0 {
        #[bits(18)]
        pub lineno: u32,
        #[bits(6)]
        pub class_id: u32,
        #[bits(8)]
        pub flags: u32,
    }
    #[bitfields::bitfield([u8; 9], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1106U0 {
        #[bits(1)]
        pub g4x: u64,
        #[bits(1)]
        pub mobile: u64,
        #[bits(1)]
        pub dgfx: u64,
        #[bits(1)]
        pub i830: u64,
        #[bits(1)]
        pub i845g: u64,
        #[bits(1)]
        pub i85x: u64,
        #[bits(1)]
        pub i865g: u64,
        #[bits(1)]
        pub i915g: u64,
        #[bits(1)]
        pub i915gm: u64,
        #[bits(1)]
        pub i945g: u64,
        #[bits(1)]
        pub i945gm: u64,
        #[bits(1)]
        pub g33: u64,
        #[bits(1)]
        pub pineview: u64,
        #[bits(1)]
        pub i965g: u64,
        #[bits(1)]
        pub i965gm: u64,
        #[bits(1)]
        pub g45: u64,
        #[bits(1)]
        pub gm45: u64,
        #[bits(1)]
        pub ironlake: u64,
        #[bits(1)]
        pub sandybridge: u64,
        #[bits(1)]
        pub ivybridge: u64,
        #[bits(1)]
        pub valleyview: u64,
        #[bits(1)]
        pub haswell: u64,
        #[bits(1)]
        pub haswell_ult: u64,
        #[bits(1)]
        pub haswell_ulx: u64,
        #[bits(1)]
        pub broadwell: u64,
        #[bits(1)]
        pub broadwell_ult: u64,
        #[bits(1)]
        pub broadwell_ulx: u64,
        #[bits(1)]
        pub cherryview: u64,
        #[bits(1)]
        pub skylake: u64,
        #[bits(1)]
        pub skylake_ult: u64,
        #[bits(1)]
        pub skylake_ulx: u64,
        #[bits(1)]
        pub broxton: u64,
        #[bits(1)]
        pub kabylake: u64,
        #[bits(1)]
        pub kabylake_ult: u64,
        #[bits(1)]
        pub kabylake_ulx: u64,
        #[bits(1)]
        pub geminilake: u64,
        #[bits(1)]
        pub coffeelake: u64,
        #[bits(1)]
        pub coffeelake_ult: u64,
        #[bits(1)]
        pub coffeelake_ulx: u64,
        #[bits(1)]
        pub cometlake: u64,
        #[bits(1)]
        pub cometlake_ult: u64,
        #[bits(1)]
        pub cometlake_ulx: u64,
        #[bits(1)]
        pub icelake: u64,
        #[bits(1)]
        pub icelake_port_f: u64,
        #[bits(1)]
        pub jasperlake: u64,
        #[bits(1)]
        pub elkhartlake: u64,
        #[bits(1)]
        pub tigerlake: u64,
        #[bits(1)]
        pub tigerlake_uy: u64,
        #[bits(1)]
        pub rocketlake: u64,
        #[bits(1)]
        pub dg1: u64,
        #[bits(1)]
        pub alderlake_s: u64,
        #[bits(1)]
        pub alderlake_s_raptorlake_s: u64,
        #[bits(1)]
        pub alderlake_p: u64,
        #[bits(1)]
        pub alderlake_p_alderlake_n: u64,
        #[bits(1)]
        pub alderlake_p_raptorlake_p: u64,
        #[bits(1)]
        pub alderlake_p_raptorlake_u: u64,
        #[bits(1)]
        pub dg2: u64,
        #[bits(1)]
        pub dg2_g10: u64,
        #[bits(1)]
        pub dg2_g11: u64,
        #[bits(1)]
        pub dg2_g12: u64,
        #[bits(1)]
        pub meteorlake: u64,
        #[bits(1)]
        pub meteorlake_u: u64,
        #[bits(1)]
        pub lunarlake: u64,
        #[bits(1)]
        pub battlemage: u64,
        #[bits(1)]
        pub pantherlake: u64,
        #[bits(1)]
        pub pantherlake_wildcatlake: u64,
        #[bits(1)]
        pub novalake: u64,
        #[bits(5, access = na)]
        pub __slate_pad_67: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1109U0 {
        #[bits(1)]
        pub cursor_needs_physical: u8,
        #[bits(1)]
        pub has_cdclk_crawl: u8,
        #[bits(1)]
        pub has_cdclk_squash: u8,
        #[bits(1)]
        pub has_ddi: u8,
        #[bits(1)]
        pub has_dp_mst: u8,
        #[bits(1)]
        pub has_dsb: u8,
        #[bits(1)]
        pub has_fpga_dbg: u8,
        #[bits(1)]
        pub has_gmch: u8,
        #[bits(1)]
        pub has_hotplug: u8,
        #[bits(1)]
        pub has_hti: u8,
        #[bits(1)]
        pub has_ipc: u8,
        #[bits(1)]
        pub has_overlay: u8,
        #[bits(1)]
        pub has_psr: u8,
        #[bits(1)]
        pub has_psr_hw_tracking: u8,
        #[bits(1)]
        pub overlay_needs_physical: u8,
        #[bits(1)]
        pub supports_tv: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1017U0 {
        #[bits(1)]
        pub lsb_reg_used: bool,
        #[bits(1)]
        pub aux_enable: bool,
        #[bits(1)]
        pub aux_set: bool,
        #[bits(1)]
        pub luminance_set: bool,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1371U0 {
        #[bits(1)]
        pub lvds_dither: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1348U0 {
        #[bits(1)]
        pub enable_dithering: u32,
        #[bits(1)]
        pub rsvd1: u32,
        #[bits(1)]
        pub is_bridge: u32,
        #[bits(2)]
        pub panel_arch_type: u32,
        #[bits(1)]
        pub is_cmd_mode: u32,
        #[bits(2)]
        pub video_transfer_mode: u32,
        #[bits(1)]
        pub cabc_supported: u32,
        #[bits(1)]
        pub pwm_blc: u32,
        #[bits(4)]
        pub videomode_color_format: u32,
        #[bits(2)]
        pub rotation: u32,
        #[bits(1)]
        pub bta_disable: u32,
        #[bits(15)]
        pub rsvd2: u32,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1349U0 {
        #[bits(2)]
        pub dual_link: u16,
        #[bits(2)]
        pub lane_cnt: u16,
        #[bits(3)]
        pub pixel_overlap: u16,
        #[bits(1)]
        pub rgb_flip: u16,
        #[bits(2)]
        pub dl_dcs_cabc_ports: u16,
        #[bits(2)]
        pub dl_dcs_backlight_ports: u16,
        #[bits(1)]
        pub port_sync: u16,
        #[bits(3)]
        pub rsvd3: u16,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1350U0 {
        #[bits(1)]
        pub dsi_usage: u16,
        #[bits(15)]
        pub rsvd4: u16,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1351U0 {
        #[bits(2)]
        pub byte_clk_sel: u8,
        #[bits(6)]
        pub rsvd6: u8,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1352U0 {
        #[bits(1)]
        pub dphy_param_valid: u16,
        #[bits(1)]
        pub eot_pkt_disabled: u16,
        #[bits(1)]
        pub enable_clk_stop: u16,
        #[bits(1)]
        pub blanking_packets_during_bllp: u16,
        #[bits(1)]
        pub lp_clock_during_lpm: u16,
        #[bits(11)]
        pub rsvd7: u16,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1353U0 {
        #[bits(6)]
        pub prepare_cnt: u32,
        #[bits(2)]
        pub rsvd8: u32,
        #[bits(8)]
        pub clk_zero_cnt: u32,
        #[bits(5)]
        pub trail_cnt: u32,
        #[bits(3)]
        pub rsvd9: u32,
        #[bits(6)]
        pub exit_zero_cnt: u32,
        #[bits(2)]
        pub rsvd10: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1382U0 {
        #[bits(1)]
        pub dsc_hblank_expansion_quirk: u8,
        #[bits(1)]
        pub dsc_throughput_quirk: u8,
        #[bits(1)]
        pub dsc_decompression_enabled: u8,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1301U0 {
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
        pub tunnel_suspended: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1200U0 {
        #[bits(1)]
        pub int_tv_support: u32,
        #[bits(1)]
        pub int_crt_support: u32,
        #[bits(1)]
        pub lvds_use_ssc: u32,
        #[bits(1)]
        pub int_lvds_support: u32,
        #[bits(1)]
        pub display_clock_mode: u32,
        #[bits(1)]
        pub fdi_rx_polarity_inverted: u32,
        #[bits(2, access = na)]
        pub __slate_pad_6: u8,
    }
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
struct irq_redirect {
    work: irq_work,
    target_cpu: u32,
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
struct gpio_chip {
    label: *const u8,
    gpiodev: *mut gpio_device,
    parent: *mut device,
    fwnode: *mut fwnode_handle,
    owner: *mut module,
    request: Option<unsafe extern "C" fn(*mut gpio_chip, u32) -> i32>,
    free: Option<unsafe extern "C" fn(*mut gpio_chip, u32)>,
    get_direction: Option<unsafe extern "C" fn(*mut gpio_chip, u32) -> i32>,
    direction_input: Option<unsafe extern "C" fn(*mut gpio_chip, u32) -> i32>,
    direction_output: Option<unsafe extern "C" fn(*mut gpio_chip, u32, i32) -> i32>,
    get: Option<unsafe extern "C" fn(*mut gpio_chip, u32) -> i32>,
    get_multiple: Option<unsafe extern "C" fn(*mut gpio_chip, *mut u64, *mut u64) -> i32>,
    set: Option<unsafe extern "C" fn(*mut gpio_chip, u32, i32) -> i32>,
    set_multiple: Option<unsafe extern "C" fn(*mut gpio_chip, *mut u64, *mut u64) -> i32>,
    set_config: Option<unsafe extern "C" fn(*mut gpio_chip, u32, u64) -> i32>,
    to_irq: Option<unsafe extern "C" fn(*mut gpio_chip, u32) -> i32>,
    dbg_show: Option<unsafe extern "C" fn(*mut seq_file, *mut gpio_chip)>,
    init_valid_mask: Option<unsafe extern "C" fn(*mut gpio_chip, *mut u64, u32) -> i32>,
    add_pin_ranges: Option<unsafe extern "C" fn(*mut gpio_chip) -> i32>,
    en_hw_timestamp: Option<unsafe extern "C" fn(*mut gpio_chip, u32, u64) -> i32>,
    dis_hw_timestamp: Option<unsafe extern "C" fn(*mut gpio_chip, u32, u64) -> i32>,
    base: i32,
    ngpio: u16,
    offset: u16,
    names: *const *const u8,
    can_sleep: bool,
}


#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_713: [u8; 19] = [105, 57, 49, 53, 46, 105, 109, 112, 111, 114, 116, 95, 110, 115, 61, 80, 87, 77, 0];

#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_args {
    period: u64,
    polarity: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_waveform {
    period_length_ns: u64,
    duty_length_ns: u64,
    duty_offset_ns: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_state {
    period: u64,
    duty_cycle: u64,
    polarity: u32,
    enabled: bool,
    usage_power: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_device {
    label: *const u8,
    flags: u64,
    hwpwm: u32,
    chip: *mut pwm_chip,
    args: pwm_args,
    state: pwm_state,
    last: pwm_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_capture {
    period: u32,
    duty_cycle: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_ops {
    request: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device) -> i32>,
    free: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device)>,
    capture: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *mut pwm_capture, u64) -> i32>,
    sizeof_wfhw: u64,
    round_waveform_tohw: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *const pwm_waveform, *mut ()) -> i32>,
    round_waveform_fromhw: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *const (), *mut pwm_waveform) -> i32>,
    read_waveform: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *mut ()) -> i32>,
    write_waveform: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *const ()) -> i32>,
    apply: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *const pwm_state) -> i32>,
    get_state: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *mut pwm_state) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_chip {
    dev: device,
    cdev: cdev,
    ops: *const pwm_ops,
    owner: *mut module,
    id: u32,
    npwm: u32,
    of_xlate: Option<unsafe extern "C" fn(*mut pwm_chip, *const of_phandle_args) -> *mut pwm_device>,
    atomic: bool,
    gpio: gpio_chip,
    uses_pwmchip_alloc: bool,
    operational: bool,
    __slate_anon_11: __SlateRecord_pwm_h_348,
    pwms: [pwm_device; 0],
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
    __slate_bits_0: __slate_bits::__SlateBits992U0,
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


fn drm_atomic_get_old_crtc_state(state: *const drm_atomic_commit, crtc: *mut drm_crtc) -> *mut drm_crtc_state {
unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(drm_crtc_index(crtc as *const drm_crtc) as isize)).old_state }
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

// for_each_oldnew_connector_in_state - iterate over all connectors in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @connector: &struct drm_connector iteration cursor
// @old_connector_state: &struct drm_connector_state iteration cursor for the
// 	old state
// @new_connector_state: &struct drm_connector_state iteration cursor for the
// 	new state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all connectors in an atomic update, tracking both old and
// new state. This is useful in places where the state delta needs to be
// considered, for example in atomic check functions.
// Only to avoid unused-but-set-variable warning
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
struct drm_dp_mst_topology_cbs {
    add_connector: Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_port, *const u8) -> *mut drm_connector>,
    poll_hpd_irq: Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr)>,
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
    __slate_bits_0: __slate_bits::__SlateBits1301U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_colorop_state {
    colorop: *mut drm_colorop,
    bypass: bool,
    curve_1d_type: u32,
    multiplier: u64,
    data: *mut drm_property_blob,
    lut1d_interpolation: u32,
    lut3d_interpolation: u32,
    state: *mut drm_atomic_commit,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_colorop_funcs {
    destroy: Option<unsafe extern "C" fn(*mut drm_colorop)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_colorop {
    dev: *mut drm_device,
    head: list_head,
    index: u32,
    base: drm_mode_object,
    plane: *mut drm_plane,
    state: *mut drm_colorop_state,
    properties: drm_object_properties,
    r#type: u32,
    next: *mut drm_colorop,
    type_property: *mut drm_property,
    bypass_property: *mut drm_property,
    size: u32,
    lut1d_interpolation_property: *mut drm_property,
    curve_1d_type_property: *mut drm_property,
    multiplier_property: *mut drm_property,
    size_property: *mut drm_property,
    lut3d_interpolation_property: *mut drm_property,
    data_property: *mut drm_property,
    next_property: *mut drm_property,
    funcs: *const drm_colorop_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_format_info {
    format: u32,
    depth: u8,
    num_planes: u8,
    __slate_anon_3: __SlateRecord_drm_fourcc_h_77,
    block_w: [u8; 4],
    block_h: [u8; 4],
    hsub: u8,
    vsub: u8,
    has_alpha: bool,
    is_yuv: bool,
    is_color_indexed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_framebuffer_funcs {
    destroy: Option<unsafe extern "C" fn(*mut drm_framebuffer)>,
    create_handle: Option<unsafe extern "C" fn(*mut drm_framebuffer, *mut drm_file, *mut u32) -> i32>,
    dirty: Option<unsafe extern "C" fn(*mut drm_framebuffer, *mut drm_file, u32, u32, *mut drm_clip_rect, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_framebuffer {
    dev: *mut drm_device,
    head: list_head,
    base: drm_mode_object,
    comm: [u8; 16],
    format: *const drm_format_info,
    funcs: *const drm_framebuffer_funcs,
    pitches: [u32; 4],
    offsets: [u32; 4],
    modifier: u64,
    width: u32,
    height: u32,
    flags: i32,
    internal_flags: u32,
    filp_head: list_head,
    obj: [*mut drm_gem_object; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_panel_funcs {
    prepare: Option<unsafe extern "C" fn(*mut drm_panel) -> i32>,
    enable: Option<unsafe extern "C" fn(*mut drm_panel) -> i32>,
    disable: Option<unsafe extern "C" fn(*mut drm_panel) -> i32>,
    unprepare: Option<unsafe extern "C" fn(*mut drm_panel) -> i32>,
    get_modes: Option<unsafe extern "C" fn(*mut drm_panel, *mut drm_connector) -> i32>,
    get_orientation: Option<unsafe extern "C" fn(*mut drm_panel) -> i32>,
    get_timings: Option<unsafe extern "C" fn(*mut drm_panel, u32, *mut display_timing) -> i32>,
    debugfs_init: Option<unsafe extern "C" fn(*mut drm_panel, *mut dentry)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_panel {
    dev: *mut device,
    backlight: *mut backlight_device,
    funcs: *const drm_panel_funcs,
    connector_type: i32,
    list: list_head,
    followers: list_head,
    follower_lock: mutex,
    prepare_prev_first: bool,
    prepared: bool,
    enabled: bool,
    container: *mut (),
    refcount: kref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vblank_work {
    base: kthread_work,
    vblank: *mut drm_vblank_crtc,
    count: u64,
    cancelling: i32,
    node: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_cert_rx {
    receiver_id: [u8; 5],
    kpub_rx: [u8; 131],
    reserved: [u8; 2],
    dcp_signature: [u8; 384],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_streamid_type {
    stream_id: u8,
    stream_type: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_tx_caps {
    version: u8,
    tx_cap_mask: [u8; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ake_init {
    msg_id: u8,
    r_tx: [u8; 8],
    tx_caps: hdcp2_tx_caps,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ake_send_cert {
    msg_id: u8,
    cert_rx: hdcp2_cert_rx,
    r_rx: [u8; 8],
    rx_caps: [u8; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ake_no_stored_km {
    msg_id: u8,
    e_kpub_km: [u8; 128],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ake_send_hprime {
    msg_id: u8,
    h_prime: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ake_send_pairing_info {
    msg_id: u8,
    e_kh_km: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_lc_init {
    msg_id: u8,
    r_n: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_lc_send_lprime {
    msg_id: u8,
    l_prime: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ske_send_eks {
    msg_id: u8,
    e_dkey_ks: [u8; 16],
    riv: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_rep_send_receiverid_list {
    msg_id: u8,
    rx_info: [u8; 2],
    seq_num_v: [u8; 3],
    v_prime: [u8; 16],
    receiver_ids: [u8; 155],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_rep_send_ack {
    msg_id: u8,
    v: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_rep_stream_ready {
    msg_id: u8,
    m_prime: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp_port_data {
    hdcp_ddi: u32,
    hdcp_transcoder: u32,
    port_type: u8,
    protocol: u8,
    k: u16,
    seq_num_m: u32,
    streams: *mut hdcp2_streamid_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i915_hdcp_ops {
    owner: *mut module,
    initiate_hdcp2_session: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_ake_init) -> i32>,
    verify_receiver_cert_prepare_km: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_ake_send_cert, *mut bool, *mut hdcp2_ake_no_stored_km, *mut u64) -> i32>,
    verify_hprime: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_ake_send_hprime) -> i32>,
    store_pairing_info: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_ake_send_pairing_info) -> i32>,
    initiate_locality_check: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_lc_init) -> i32>,
    verify_lprime: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_lc_send_lprime) -> i32>,
    get_session_key: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_ske_send_eks) -> i32>,
    repeater_check_flow_prepare_ack: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_rep_send_receiverid_list, *mut hdcp2_rep_send_ack) -> i32>,
    verify_mprime: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_rep_stream_ready) -> i32>,
    enable_hdcp_authentication: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data) -> i32>,
    close_hdcp_session: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i915_hdcp_arbiter {
    hdcp_dev: *mut device,
    ops: *const i915_hdcp_ops,
    mutex: mutex,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct mipi_config {
    panel_id: u16,
    __slate_anon_1: __SlateRecord_intel_dsi_vbt_defs_h_50,
    __slate_anon_2: __SlateRecord_intel_dsi_vbt_defs_h_84,
    __slate_anon_3: __SlateRecord_intel_dsi_vbt_defs_h_102,
    rsvd5: u8,
    target_burst_mode_freq: u32,
    dsi_ddr_clk: u32,
    bridge_ref_clk: u32,
    __slate_anon_8: __SlateRecord_intel_dsi_vbt_defs_h_113,
    __slate_anon_9: __SlateRecord_intel_dsi_vbt_defs_h_122,
    hs_tx_timeout: u32,
    lp_rx_timeout: u32,
    turn_around_timeout: u32,
    device_reset_timer: u32,
    master_init_timer: u32,
    dbi_bw_timer: u32,
    lp_byte_clk_val: u32,
    __slate_anon_17: __SlateRecord_intel_dsi_vbt_defs_h_140,
    clk_lane_switch_cnt: u32,
    hl_switch_cnt: u32,
    rsvd11: [u32; 6],
    tclk_miss: u8,
    tclk_post: u8,
    rsvd12: u8,
    tclk_pre: u8,
    tclk_prepare: u8,
    tclk_settle: u8,
    tclk_term_enable: u8,
    tclk_trail: u8,
    tclk_prepare_clkzero: u16,
    rsvd13: u8,
    td_term_enable: u8,
    teot: u8,
    ths_exit: u8,
    ths_prepare: u8,
    ths_prepare_hszero: u16,
    rsvd14: u8,
    ths_settle: u8,
    ths_skip: u8,
    ths_trail: u8,
    tinit: u8,
    tlpx: u8,
    rsvd15: [u8; 3],
    panel_enable: u8,
    bl_enable: u8,
    pwm_enable: u8,
    reset_r_n: u8,
    pwr_down_r: u8,
    stdby_r_n: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct mipi_pps_data {
    panel_on_delay: u16,
    bl_enable_delay: u16,
    bl_disable_delay: u16,
    panel_off_delay: u16,
    panel_power_cycle_delay: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_encoder {
    base: drm_encoder,
    r#type: u32,
    port: i32,
    cloneable: u16,
    pipe_mask: u8,
    link_check_work: delayed_work,
    link_check: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    hotplug: Option<unsafe extern "C" fn(*mut intel_encoder, *mut intel_connector) -> u32>,
    compute_output_type: Option<unsafe extern "C" fn(*mut intel_encoder, *mut intel_crtc_state, *mut drm_connector_state) -> u32>,
    compute_config: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *mut intel_crtc_state, *mut drm_connector_state) -> i32>,
    compute_config_late: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *mut intel_crtc_state, *mut drm_connector_state) -> i32>,
    pre_pll_enable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    pre_enable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    enable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    disable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    post_disable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    post_pll_disable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    update_pipe: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    audio_enable: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    audio_disable: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    get_hw_state: Option<unsafe extern "C" fn(*mut intel_encoder, *mut i32) -> bool>,
    get_config: Option<unsafe extern "C" fn(*mut intel_encoder, *mut intel_crtc_state)>,
    sync_state: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state)>,
    initial_fastset_check: Option<unsafe extern "C" fn(*mut intel_encoder, *mut intel_crtc_state) -> bool>,
    get_power_domains: Option<unsafe extern "C" fn(*mut intel_encoder, *mut intel_crtc_state)>,
    suspend: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    suspend_complete: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    shutdown: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    shutdown_complete: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    enable_clock: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state)>,
    disable_clock: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    is_clock_enabled: Option<unsafe extern "C" fn(*mut intel_encoder) -> bool>,
    port_pll_type: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state) -> u32>,
    get_buf_trans: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state, *mut i32) -> *const intel_ddi_buf_trans>,
    set_signal_levels: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state)>,
    hpd_pin: u32,
    power_domain: u32,
    devdata: *const intel_bios_encoder_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_panel_bl_funcs {
    setup: Option<unsafe extern "C" fn(*mut intel_connector, i32) -> i32>,
    get: Option<unsafe extern "C" fn(*mut intel_connector, i32) -> u32>,
    set: Option<unsafe extern "C" fn(*const drm_connector_state, u32)>,
    disable: Option<unsafe extern "C" fn(*const drm_connector_state, u32)>,
    enable: Option<unsafe extern "C" fn(*const intel_crtc_state, *const drm_connector_state, u32)>,
    hz_to_pwm: Option<unsafe extern "C" fn(*mut intel_connector, u32) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_pps_delays {
    power_up: u16,
    backlight_on: u16,
    backlight_off: u16,
    power_down: u16,
    power_cycle: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_vbt_panel_data {
    lfp_vbt_mode: *mut drm_display_mode,
    sdvo_lvds_vbt_mode: *mut drm_display_mode,
    panel_type: i32,
    __slate_bits_0: __slate_bits::__SlateBits1371U0,
    bios_lvds_val: u32,
    vrr: bool,
    seamless_drrs_min_refresh_rate: u8,
    drrs_type: u32,
    edp: __SlateRecord_intel_display_types_h_345,
    psr: __SlateRecord_intel_display_types_h_360,
    backlight: __SlateRecord_intel_display_types_h_370,
    dsi: __SlateRecord_intel_display_types_h_382,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_panel {
    base: *mut drm_panel,
    fixed_edid: *const drm_edid,
    fixed_modes: list_head,
    backlight: __SlateRecord_intel_display_types_h_407,
    vbt: intel_vbt_panel_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hdcp {
    shim: *const intel_hdcp_shim,
    mutex: mutex,
    value: u64,
    check_work: delayed_work,
    prop_work: work_struct,
    hdcp_encrypted: bool,
    hdcp2_supported: bool,
    hdcp2_encrypted: bool,
    content_type: u8,
    is_paired: bool,
    is_repeater: bool,
    seq_num_v: u32,
    seq_num_m: u32,
    cp_irq_queue: wait_queue_head,
    cp_irq_count: __SlateRecord_types_h_190,
    cp_irq_count_cached: i32,
    cpu_transcoder: i32,
    stream_transcoder: i32,
    force_hdcp14: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_connector {
    base: drm_connector,
    encoder: *mut intel_encoder,
    acpi_device_id: u32,
    get_hw_state: Option<unsafe extern "C" fn(*mut intel_connector) -> bool>,
    sync_state: Option<unsafe extern "C" fn(*mut intel_connector, *const intel_crtc_state)>,
    panel: intel_panel,
    detect_edid: *const drm_edid,
    hotplug_retries: i32,
    polled: u8,
    force_joined_pipes: i32,
    dp: __SlateRecord_intel_display_types_h_559,
    mst: __SlateRecord_intel_display_types_h_602,
    link: __SlateRecord_intel_display_types_h_607,
    modeset_retry_work: work_struct,
    hdcp: intel_hdcp,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_digital_connector_state {
    base: drm_connector_state,
    force_audio: i32,
    broadcast_rgb: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dpll {
    n: i32,
    m1: i32,
    m2: i32,
    p1: i32,
    p2: i32,
    dot: i32,
    vco: i32,
    m: i32,
    p: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_atomic_state {
    base: drm_atomic_commit,
    wakeref: *mut ref_tracker,
    global_objs: *mut intel_global_objs_state,
    num_global_objs: i32,
    internal: bool,
    dpll_set: bool,
    modeset: bool,
    dpll_state: [intel_dpll_state; 9],
    inherited_dp_tunnels: *mut intel_dp_tunnel_inherited_state,
    skip_intermediate_wm: bool,
    rps_interactive: bool,
    cleanup_work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_initial_plane_config {
    fb: *mut drm_framebuffer,
    vma: *mut i915_vma,
    size: i32,
    base: u32,
    rotation: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_scaler {
    mode: u32,
    in_use: bool,
    hscale: i32,
    vscale: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_crtc_scaler_state {
    scalers: [intel_scaler; 2],
    scaler_users: u32,
    scaler_id: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_wm_level {
    enable: bool,
    pri_val: u32,
    spr_val: u32,
    cur_val: u32,
    fbc_val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_pipe_wm {
    wm: [intel_wm_level; 5],
    fbc_wm_enabled: bool,
    pipe_enabled: bool,
    sprites_enabled: bool,
    sprites_scaled: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct skl_wm_level {
    min_ddb_alloc: u16,
    min_ddb_alloc_uv: u16,
    blocks: u16,
    lines: u8,
    enable: bool,
    ignore_lines: bool,
    auto_min_alloc_wm_enable: bool,
    can_sagv: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct skl_plane_wm {
    wm: [skl_wm_level; 8],
    trans_wm: skl_wm_level,
    sagv: __SlateRecord_intel_display_types_h_853,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct skl_pipe_wm {
    planes: [skl_plane_wm; 8],
    use_sagv_wm: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vlv_wm_state {
    wm: [g4x_pipe_wm; 3],
    sr: [g4x_sr_wm; 3],
    num_levels: u8,
    cxsr: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vlv_fifo_state {
    plane: [u16; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct g4x_wm_state {
    wm: g4x_pipe_wm,
    sr: g4x_sr_wm,
    hpll: g4x_sr_wm,
    cxsr: bool,
    hpll_en: bool,
    fbc_en: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_crtc_wm_state {
    __slate_anon_0: __SlateRecord_intel_display_types_h_899,
    need_postvbl_update: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_link_m_n {
    tu: u32,
    data_m: u32,
    data_n: u32,
    link_m: u32,
    link_n: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_csc_matrix {
    coeff: [u16; 9],
    preoff: [u16; 3],
    postoff: [u16; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scaler_filter_coeff {
    sign: u16,
    exp: u16,
    mantissa: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_casf {
    coeff: [scaler_filter_coeff; 7],
    strength: u8,
    win_size: u8,
    enable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_crtc_state {
    uapi: drm_crtc_state,
    hw: __SlateRecord_intel_display_types_h_1036,
    pre_csc_lut: *mut drm_property_blob,
    post_csc_lut: *mut drm_property_blob,
    csc: intel_csc_matrix,
    output_csc: intel_csc_matrix,
    quirks: u64,
    fb_bits: u32,
    update_pipe: bool,
    update_m_n: bool,
    update_lrr: bool,
    disable_cxsr: bool,
    update_wm_pre: bool,
    update_wm_post: bool,
    fifo_changed: bool,
    preload_luts: bool,
    inherited: bool,
    do_async_flip: bool,
    pipe_src: drm_rect,
    pixel_rate: u32,
    pixel_rate_cdclk: u32,
    has_pch_encoder: bool,
    has_infoframe: bool,
    cpu_transcoder: i32,
    limited_color_range: bool,
    output_types: u32,
    has_hdmi_sink: bool,
    has_audio: bool,
    dither: bool,
    dither_force_disable: bool,
    clock_set: bool,
    sdvo_tv_clock: bool,
    bw_constrained: bool,
    dpll: dpll,
    intel_dpll: *mut intel_dpll,
    dpll_hw_state: intel_dpll_hw_state,
    icl_port_dplls: [icl_port_dpll; 2],
    dsi_pll: __SlateRecord_intel_display_types_h_1171,
    max_link_bpp_x16: i32,
    max_pipe_bpp: i32,
    pipe_bpp: i32,
    min_hblank: i32,
    dp_m_n: intel_link_m_n,
    dp_m2_n2: intel_link_m_n,
    has_drrs: bool,
    has_psr: bool,
    has_sel_update: bool,
    enable_psr2_sel_fetch: bool,
    enable_psr2_su_region_et: bool,
    clear_psr2_sel_fetch: bool,
    req_psr2_sdp_prior_scanline: bool,
    has_panel_replay: bool,
    link_off_after_as_sdp_when_pr_active: bool,
    disable_as_sdp_when_pr_active: bool,
    wm_level_disabled: bool,
    pkg_c_latency_used: bool,
    panel_replay_dsc_support: u32,
    su_y_granularity: u16,
    active_non_psr_pipes: u8,
    entry_setup_frames: u8,
    no_psr_reason: *const u8,
    port_clock: i32,
    pixel_multiplier: u32,
    mode_flags: u8,
    lane_count: u8,
    lane_lat_optim_mask: u8,
    min_voltage_level: u8,
    gmch_pfit: __SlateRecord_intel_display_types_h_1230,
    pch_pfit: __SlateRecord_intel_display_types_h_1237,
    fdi_lanes: i32,
    fdi_m_n: intel_link_m_n,
    ips_enabled: bool,
    crc_enabled: bool,
    double_wide: bool,
    scaler_state: intel_crtc_scaler_state,
    hsw_workaround_pipe: i32,
    wm: intel_crtc_wm_state,
    min_cdclk: i32,
    plane_min_cdclk: [i32; 8],
    data_rate: [u32; 8],
    data_rate_y: [u32; 8],
    rel_data_rate: [u64; 8],
    rel_data_rate_y: [u64; 8],
    gamma_mode: u32,
    __slate_anon_84: __SlateRecord_intel_display_types_h_1277,
    enabled_planes: u8,
    active_planes: u8,
    scaled_planes: u8,
    nv12_planes: u8,
    c8_planes: u8,
    update_planes: u8,
    async_flip_planes: u8,
    framestart_delay: u8,
    msa_timing_delay: u8,
    infoframes: __SlateRecord_intel_display_types_h_1303,
    eld: [u8; 128],
    hdmi_scrambling: bool,
    hdmi_high_tmds_clock_ratio: bool,
    output_format: u32,
    sink_format: u32,
    gamma_enable: bool,
    csc_enable: bool,
    wgc_enable: bool,
    joiner_pipes: u8,
    dsc: __SlateRecord_intel_display_types_h_1347,
    dp_tunnel_ref: drm_dp_tunnel_ref,
    linetime: u16,
    ips_linetime: u16,
    enhanced_framing: bool,
    fec_enable: bool,
    sdp_split_enable: bool,
    master_transcoder: i32,
    sync_mode_slaves_mask: u16,
    mst_master_transcoder: i32,
    dsb_color: *mut intel_dsb,
    dsb_commit: *mut intel_dsb,
    use_dsb: bool,
    use_flipq: bool,
    psr2_man_track_ctl: u32,
    pipe_srcsz_early_tpt: u32,
    psr2_su_area: drm_rect,
    vrr: __SlateRecord_intel_display_types_h_1401,
    cmrr: __SlateRecord_intel_display_types_h_1416,
    splitter: __SlateRecord_intel_display_types_h_1422,
    vblank_work: drm_vblank_work,
    has_lobf: bool,
    set_context_latency: u16,
    alpm_state: __SlateRecord_intel_display_types_h_1437,
    plane_color_changed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_pipe_crc {
    lock: spinlock,
    skipped: i32,
    source: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_flipq {
    start_mmioaddr: u32,
    flipq_id: u32,
    tail: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_crtc {
    base: drm_crtc,
    pipe_head: list_head,
    pipe: i32,
    active: bool,
    plane_ids_mask: u8,
    mode_flags: u8,
    vmax_vblank_start: u16,
    enabled_power_domains: intel_display_power_domain_set,
    hw_readout_power_domains: intel_display_power_domain_set,
    overlay: *mut intel_overlay,
    config: *mut intel_crtc_state,
    flip_done_event: *mut drm_pending_vblank_event,
    dsb_event: *mut drm_pending_vblank_event,
    flipq_event: *mut drm_pending_vblank_event,
    cpu_fifo_underrun_disabled: bool,
    pch_fifo_underrun_disabled: bool,
    flipq: [intel_flipq; 5],
    wm: __SlateRecord_intel_display_types_h_1535,
    drrs: __SlateRecord_intel_display_types_h_1544,
    dc_balance: __SlateRecord_intel_display_types_h_1554,
    scanline_offset: i32,
    debug: __SlateRecord_intel_display_types_h_1560,
    num_scalers: i32,
    vblank_pm_qos: pm_qos_request,
    pipe_crc: intel_pipe_crc,
    vblank_psr_notify: bool,
    cmtg: __SlateRecord_intel_display_types_h_1588,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hdmi {
    hdmi_reg: __SlateRecord_i915_reg_defs_h_14,
    dp_dual_mode: __SlateRecord_intel_display_types_h_1683,
    attached_connector: *mut intel_connector,
    cec_notifier: *mut cec_notifier,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_compliance_data {
    edid: u64,
    video_pattern: u8,
    hdisplay: u16,
    vdisplay: u16,
    bpc: u8,
    phytest: drm_dp_phy_test_params,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_compliance {
    test_type: u64,
    test_data: intel_dp_compliance_data,
    test_active: bool,
    test_link_rate: i32,
    test_lane_count: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_pcon_frl {
    is_trained: bool,
    trained_rate_gbps: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_pps {
    panel_power_up_delay: i32,
    panel_power_down_delay: i32,
    panel_power_cycle_delay: i32,
    backlight_on_delay: i32,
    backlight_off_delay: i32,
    panel_vdd_work: delayed_work,
    want_panel_vdd: bool,
    initializing: bool,
    last_power_on: u64,
    last_backlight_off: u64,
    panel_power_off_time: i64,
    vdd_wakeref: *mut ref_tracker,
    __slate_anon_12: __SlateRecord_intel_display_types_h_1728,
    vlv_active_pipe: i32,
    bxt_pps_reset: bool,
    pps_delays: intel_pps_delays,
    bios_pps_delays: intel_pps_delays,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_psr {
    lock: mutex,
    debug: u32,
    sink_support: bool,
    source_support: bool,
    enabled: bool,
    pause_counter: i32,
    pipe: i32,
    transcoder: i32,
    active: bool,
    work: work_struct,
    busy_frontbuffer_bits: u32,
    link_standby: bool,
    sel_update_enabled: bool,
    psr2_sel_fetch_enabled: bool,
    psr2_sel_fetch_cff_enabled: bool,
    su_region_et_enabled: bool,
    req_psr2_sdp_prior_scanline: bool,
    last_entry_attempt: i64,
    last_exit: i64,
    sink_not_reliable: bool,
    irq_aux_error: bool,
    dc3co_allowed: bool,
    dc3co_work: delayed_work,
    su_w_granularity: u16,
    su_y_granularity: u16,
    source_panel_replay_support: bool,
    sink_panel_replay_support: bool,
    panel_replay_enabled: bool,
    dc3co_exit_delay: u32,
    entry_setup_frames: u8,
    io_wake_lines: u8,
    fast_wake_lines: u8,
    link_ok: bool,
    pkg_c_latency_used: bool,
    active_non_psr_pipes: u8,
    no_psr_reason: *const u8,
    vblank_wakeref: *mut ref_tracker,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_link_config {
    rate: i32,
    lane_count: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp {
    output_reg: __SlateRecord_i915_reg_defs_h_14,
    DP: u32,
    link_rate: i32,
    lane_count: u8,
    sink_count: u8,
    downstream_port_changed: bool,
    needs_modeset_retry: bool,
    use_max_params: bool,
    dpcd: [u8; 15],
    downstream_ports: [u8; 16],
    edp_dpcd: [u8; 5],
    lttpr_common_caps: [u8; 8],
    lttpr_phy_caps: [[u8; 3]; 8],
    pcon_dsc_dpcd: [u8; 13],
    num_source_rates: i32,
    source_rates: *const i32,
    num_sink_rates: i32,
    sink_rates: [i32; 8],
    use_rate_select: bool,
    max_sink_lane_count: i32,
    link: __SlateRecord_intel_display_types_h_1848,
    reset_link_params: bool,
    mso_link_count: i32,
    mso_pixel_overlap: i32,
    desc: drm_dp_desc,
    aux: drm_dp_aux,
    aux_busy_last_status: u32,
    train_set: [u8; 4],
    pps: intel_pps,
    is_mst: bool,
    mst_detect: u32,
    attached_connector: *mut intel_connector,
    as_sdp_supported: bool,
    as_sdp_v2_supported: bool,
    tunnel: *mut drm_dp_tunnel,
    __slate_bits_0: __slate_bits::__SlateBits1080U0,
    disabled_uhbr_lane_mask: u8,
    mst: __SlateRecord_intel_display_types_h_1885,
    get_aux_clock_divider: Option<unsafe extern "C" fn(*mut intel_dp, i32) -> u32>,
    get_aux_send_ctl: Option<unsafe extern "C" fn(*mut intel_dp, i32, u32) -> u32>,
    aux_ch_ctl_reg: Option<unsafe extern "C" fn(*mut intel_dp) -> __SlateRecord_i915_reg_defs_h_14>,
    aux_ch_data_reg: Option<unsafe extern "C" fn(*mut intel_dp, i32) -> __SlateRecord_i915_reg_defs_h_14>,
    prepare_link_retrain: Option<unsafe extern "C" fn(*mut intel_dp, *const intel_crtc_state)>,
    set_link_train: Option<unsafe extern "C" fn(*mut intel_dp, *const intel_crtc_state, u8)>,
    set_idle_link_train: Option<unsafe extern "C" fn(*mut intel_dp, *const intel_crtc_state)>,
    preemph_max: Option<unsafe extern "C" fn(*mut intel_dp) -> u8>,
    voltage_max: Option<unsafe extern "C" fn(*mut intel_dp, *const intel_crtc_state) -> u8>,
    compliance: intel_dp_compliance,
    dfp: __SlateRecord_intel_display_types_h_1919,
    pm_qos: pm_qos_request,
    force_dsc_en: bool,
    force_dsc_output_format: i32,
    force_dsc_fractional_bpp_en: bool,
    force_dsc_bpc: i32,
    hobl_failed: bool,
    hobl_active: bool,
    frl: intel_dp_pcon_frl,
    psr: intel_psr,
    last_oui_write: u64,
    oui_valid: bool,
    colorimetry_support: bool,
    sst_split_sdp_support: bool,
    alpm: __SlateRecord_intel_display_types_h_1953,
    alpm_dpcd: u8,
    quirks: __SlateRecord_intel_display_types_h_1963,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_lspcon {
    active: bool,
    hdr_supported: bool,
    mode: u32,
    vendor: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_digital_port {
    base: intel_encoder,
    dp: intel_dp,
    hdmi: intel_hdmi,
    lspcon: intel_lspcon,
    hpd_pulse: Option<unsafe extern "C" fn(*mut intel_digital_port, bool) -> u32>,
    lane_reversal: bool,
    ddi_a_4_lanes: bool,
    release_cl2_override: bool,
    dedicated_external: bool,
    max_lanes: u8,
    aux_ch: i32,
    ddi_io_power_domain: u32,
    ddi_io_wakeref: *mut ref_tracker,
    aux_wakeref: *mut ref_tracker,
    tc: *mut intel_tc_port,
    hdcp: __SlateRecord_intel_display_types_h_2000,
    write_infoframe: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state, u32, *const (), i64)>,
    read_infoframe: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state, u32, *mut (), i64)>,
    set_infoframes: Option<unsafe extern "C" fn(*mut intel_encoder, bool, *const intel_crtc_state, *const drm_connector_state)>,
    infoframes_enabled: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state) -> u32>,
    connected: Option<unsafe extern "C" fn(*mut intel_encoder) -> bool>,
    lock: Option<unsafe extern "C" fn(*mut intel_digital_port)>,
    unlock: Option<unsafe extern "C" fn(*mut intel_digital_port)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_mst_encoder {
    base: intel_encoder,
    pipe: i32,
    primary: *mut intel_digital_port,
    connector: *mut intel_connector,
}


fn intel_attached_encoder(connector: *mut intel_connector) -> *mut intel_encoder {
unsafe { (*connector).encoder }
}

fn intel_encoder_is_dig_port(encoder: *mut intel_encoder) -> bool {
match (unsafe { (*encoder).r#type }) as u32 {
6..=8 | 10 => {
return true;
}
_ => {
return false;
}
}
unsafe { core::mem::zeroed() }
}

fn intel_encoder_is_mst(encoder: *mut intel_encoder) -> bool {
((unsafe { (*encoder).r#type }) as u32) == 11
}

fn enc_to_mst(encoder: *mut intel_encoder) -> *mut intel_dp_mst_encoder {
let mut __v0: *mut intel_dp_mst_encoder;
__v0 = ((unsafe { ((core::ptr::addr_of_mut!((*encoder).base) as *mut ()) as *const u8).wrapping_offset(-(offsetof__6!(intel_dp_mst_encoder) as isize)) }) as *mut ()) as *mut intel_dp_mst_encoder;
__v0
}

fn enc_to_dig_port(encoder: *mut intel_encoder) -> *mut intel_digital_port {
let intel_encoder: *mut intel_encoder = encoder;
if intel_encoder_is_dig_port(intel_encoder) {
let mut __v0: *mut intel_digital_port;
__v0 = ((unsafe { ((core::ptr::addr_of_mut!((*encoder).base) as *mut ()) as *const u8).wrapping_offset(-(offsetof__6!(intel_digital_port) as isize)) }) as *mut ()) as *mut intel_digital_port;
return __v0;
} else {
if intel_encoder_is_mst(intel_encoder) {
return unsafe { (*enc_to_mst(encoder)).primary };
} else {
return core::ptr::null_mut::<intel_digital_port>();
}
}
}

fn enc_to_intel_dp(encoder: *mut intel_encoder) -> *mut intel_dp {
unsafe { core::ptr::addr_of_mut!((*enc_to_dig_port(encoder)).dp) }
}

fn intel_attached_dp(connector: *mut intel_connector) -> *mut intel_dp {
if !unsafe { (*connector).mst.dp }.is_null() {
return unsafe { (*connector).mst.dp };
} else {
return enc_to_intel_dp(intel_attached_encoder(connector));
}
}

fn dp_to_dig_port(intel_dp: *mut intel_dp) -> *mut intel_digital_port {
container_of!(intel_dp, intel_digital_port)
}

fn hdmi_to_dig_port(intel_hdmi: *mut intel_hdmi) -> *mut intel_digital_port {
container_of__2!(intel_hdmi, intel_digital_port)
}

fn intel_atomic_get_old_crtc_state(state: *mut intel_atomic_state, crtc: *mut intel_crtc) -> *mut intel_crtc_state {
to_intel_crtc_state!(drm_atomic_get_old_crtc_state((unsafe { core::ptr::addr_of_mut!((*state).base) }) as *const drm_atomic_commit, unsafe { core::ptr::addr_of_mut!((*crtc).base) }))
}

fn intel_atomic_get_new_crtc_state(state: *mut intel_atomic_state, crtc: *mut intel_crtc) -> *mut intel_crtc_state {
to_intel_crtc_state!(drm_atomic_get_new_crtc_state((unsafe { core::ptr::addr_of_mut!((*state).base) }) as *const drm_atomic_commit, unsafe { core::ptr::addr_of_mut!((*crtc).base) }))
}

fn intel_crtc_has_type(crtc_state: *const intel_crtc_state, r#type: u32) -> bool {
((unsafe { (*crtc_state).output_types }) as u64) & BIT__3!(r#type) != 0
}

fn intel_crtc_has_dp_encoder(crtc_state: *const intel_crtc_state) -> bool {
((unsafe { (*crtc_state).output_types }) as u64) & (BIT!(7i32) | BIT!(11i32) | BIT!(8i32)) != 0
}

// Helper for generic association. Map types to conversion functions/macros.
#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_link_caps_order {
    key: u32,
    dir: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_link_caps_filter {
    config_mask: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_link_caps_iter {
    link_caps: *mut intel_dp_link_caps,
    pos: i32,
    order: intel_dp_link_caps_order,
    filter: intel_dp_link_caps_filter,
    get_next_config: Option<unsafe extern "C" fn(*mut intel_dp_link_caps_iter, *mut intel_dp_link_config) -> bool>,
}


// for_each_dp_link_config - iterate allowed link configurations
// @__iter:
//   &struct intel_dp_link_caps_iter being iterated
// @__config:
//   pointer to &struct intel_dp_link_config filled for each match
#[repr(C)]
#[derive(Clone, Copy)]
struct link_config_limits {
    link_config_filter: intel_dp_link_caps_filter,
    pipe: __SlateRecord_intel_dp_h_32,
    link: __SlateRecord_intel_dp_h_36,
}


#[unsafe(no_mangle)]
extern "C" fn intel_dp_max_source_lane_count(dig_port: *mut intel_digital_port) -> i32 {
let vbt_max_lanes: i32 = unsafe { intel_bios_dp_max_lane_count((*dig_port).base.devdata) };
let mut max_lanes: i32 = (unsafe { (*dig_port).max_lanes }) as i32;
if vbt_max_lanes != 0 {
let mut __v0: i32;
let __UNIQUE_ID_x__721: i32 = max_lanes;
let __UNIQUE_ID_y__722: i32 = vbt_max_lanes;
loop {
if __types_ok!(__UNIQUE_ID_x__721, __UNIQUE_ID_y__722) == 0 {
unsafe { __compiletime_assert_723() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__721 < __UNIQUE_ID_y__722 { __UNIQUE_ID_x__721 } else { __UNIQUE_ID_y__722 };
max_lanes = __v0;
}
max_lanes
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_link_bw_overhead(link_clock: i32, lane_count: i32, hdisplay: i32, dsc_slice_count: i32, bpp_x16: i32, mut flags: u64) -> i32 {
let mut overhead: i32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = (flags & !(BIT!(0i32) | BIT!(2i32) | BIT!(3i32)) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("731: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 731b - ., 3; .popsection\n# {0}", const 731 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_intel_dp_link_bw_overhead_1, sym __slate_asm_string_intel_dp_link_bw_overhead_2, const 390 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("732: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 732b - ., 4; .popsection\n# {0}", const 732 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if drm_dp_is_uhbr_rate(link_clock) {
let __v5: u64 = flags;
flags = __v5 | BIT!(1i32);
}
if dsc_slice_count != 0 {
let __v7: u64 = flags;
flags = __v7 | BIT!(4i32);
}
overhead = unsafe { drm_dp_bw_overhead(lane_count, hdisplay, dsc_slice_count, bpp_x16, flags) };
// TODO: clarify whether a minimum required by the fixed FEC overhead
// in the bspec audio programming sequence is required here.
let mut __v9: i32;
let __UNIQUE_ID_x__733: i32 = overhead;
let __UNIQUE_ID_y__734: i32 = intel_dp_bw_fec_overhead(flags & BIT!(3i32) != 0);
loop {
if __types_ok!(__UNIQUE_ID_x__733, __UNIQUE_ID_y__734) == 0 {
unsafe { __compiletime_assert_735() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v9 = if __UNIQUE_ID_x__733 > __UNIQUE_ID_y__734 { __UNIQUE_ID_x__733 } else { __UNIQUE_ID_y__734 };
__v9
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_max_link_data_rate(intel_dp: *mut intel_dp, max_dprx_rate: i32, max_dprx_lanes: i32) -> i32 {
let mut max_rate: i32 = unsafe { drm_dp_max_dprx_data_rate(max_dprx_rate, max_dprx_lanes) };
if intel_dp_tunnel_bw_alloc_is_enabled(intel_dp) {
let mut __v0: i32;
let __UNIQUE_ID_x__736: i32 = max_rate;
let __UNIQUE_ID_y__737: i32 = drm_dp_tunnel_available_bw((unsafe { (*intel_dp).tunnel }) as *const drm_dp_tunnel);
loop {
if __types_ok!(__UNIQUE_ID_x__736, __UNIQUE_ID_y__737) == 0 {
unsafe { __compiletime_assert_738() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__736 < __UNIQUE_ID_y__737 { __UNIQUE_ID_x__736 } else { __UNIQUE_ID_y__737 };
max_rate = __v0;
}
max_rate
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_dsc_compute_max_bpp(connector: *const intel_connector, __UNIQUE_ID_y__785: u8) -> i32 {
let display: *mut intel_display = to_intel_display__10!(connector);
let mut dsc_max_bpc: i32 = intel_dp_dsc_max_src_input_bpc(display);
if dsc_max_bpc == 0 {
return dsc_max_bpc;
}
let mut __v1: i32;
let __UNIQUE_ID_x__784: i32 = dsc_max_bpc;
loop {
if __types_ok__7!(__UNIQUE_ID_x__784, __UNIQUE_ID_y__785) == 0 {
unsafe { __compiletime_assert_786() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = if __UNIQUE_ID_x__784 < (__UNIQUE_ID_y__785 as i32) { __UNIQUE_ID_x__784 } else { __UNIQUE_ID_y__785 as i32 };
dsc_max_bpc = __v1;
align_max_sink_dsc_input_bpp(connector, dsc_max_bpc.wrapping_mul(3))
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_compute_min_compressed_bpp_x16(connector: *mut intel_connector, output_format: u32) -> i32 {
let dsc_src_min_bpp: i32 = intel_dp_dsc_min_src_compressed_bpp();
let dsc_sink_min_bpp: i32 = intel_dp_dsc_sink_min_compressed_bpp(output_format);
let mut dsc_min_bpp: i32 = 0;
let mut min_bpp_x16: i32 = 0;
let mut __v0: i32;
let __UNIQUE_ID_x__807: i32 = dsc_src_min_bpp;
let __UNIQUE_ID_y__808: i32 = dsc_sink_min_bpp;
loop {
if __types_ok!(__UNIQUE_ID_x__807, __UNIQUE_ID_y__808) == 0 {
unsafe { __compiletime_assert_809() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__807 > __UNIQUE_ID_y__808 { __UNIQUE_ID_x__807 } else { __UNIQUE_ID_y__808 };
dsc_min_bpp = __v0;
min_bpp_x16 = fxp_q4_from_int(dsc_min_bpp);
min_bpp_x16 = align_min_compressed_bpp_x16(connector as *const intel_connector, min_bpp_x16);
min_bpp_x16
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_invalidate_source_oui(intel_dp: *mut intel_dp) {
loop {
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_859() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*intel_dp).oui_valid, false);
if true {
break;
}
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_compute_config_limits(intel_dp: *mut intel_dp, conn_state: *mut drm_connector_state, crtc_state: *mut intel_crtc_state, respect_downstream_limits: bool, dsc: bool, limits: *mut link_config_limits) -> bool {
let mut __slate_compound_2: intel_dp_link_caps_filter = unsafe { core::mem::zeroed() };
let mut __slate_compound_16: intel_dp_link_caps_filter = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let link_caps: *mut intel_dp_link_caps = unsafe { (*intel_dp).link.caps };
let is_mst: bool = intel_crtc_has_type(crtc_state as *const intel_crtc_state, 11u32);
let connector: *mut intel_connector = to_intel_connector!(unsafe { (*conn_state).connector });
unsafe {
(*limits).link_config_filter = *{
__slate_compound_2 = intel_dp_link_caps_filter { config_mask: 1i32.wrapping_neg() as u32 };
core::ptr::addr_of_mut!(__slate_compound_2)
};
}
unsafe {
(*limits).pipe.min_bpp = intel_dp_min_bpp((*crtc_state).output_format);
}
if is_mst {
// FIXME: If all the streams can't fit into the link with their
// current pipe_bpp we should reduce pipe_bpp across the board
// until things start to fit. Until then we limit to <= 8bpc
// since that's what was hardcoded for all MST streams
// previously. This hack should be removed once we have the
// proper retry logic in place.
let mut __v3: i32;
let __UNIQUE_ID_x__828: i32 = unsafe { (*crtc_state).max_pipe_bpp };
let __UNIQUE_ID_y__829: i32 = 24;
loop {
if __types_ok!(__UNIQUE_ID_x__828, __UNIQUE_ID_y__829) == 0 {
unsafe { __compiletime_assert_830() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__828 < __UNIQUE_ID_y__829 { __UNIQUE_ID_x__828 } else { __UNIQUE_ID_y__829 };
unsafe {
(*limits).pipe.max_bpp = __v3;
}
} else {
unsafe {
(*limits).pipe.max_bpp = intel_dp_max_bpp(intel_dp, crtc_state as *const intel_crtc_state, respect_downstream_limits);
}
}
let __v5: bool = !dsc && intel_dp_in_hdr_mode(conn_state as *const drm_connector_state);
if __v5 {
if intel_dp_supports_dsc(intel_dp, connector as *const intel_connector, crtc_state as *const intel_crtc_state) && (unsafe { (*limits).pipe.max_bpp }) >= 30 {
let mut __v6: i32;
let __UNIQUE_ID_x__831: i32 = unsafe { (*limits).pipe.min_bpp };
let __UNIQUE_ID_y__832: i32 = 30;
loop {
if __types_ok!(__UNIQUE_ID_x__831, __UNIQUE_ID_y__832) == 0 {
unsafe { __compiletime_assert_833() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = if __UNIQUE_ID_x__831 > __UNIQUE_ID_y__832 { __UNIQUE_ID_x__831 } else { __UNIQUE_ID_y__832 };
unsafe {
(*limits).pipe.min_bpp = __v6;
}
} else {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Can't force 30 bpp for HDR (pipe bpp: %d-%d DSC-support: %s)\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name, (*limits).pipe.min_bpp, (*limits).pipe.max_bpp, str_yes_no(intel_dp_supports_dsc(intel_dp, connector as *const intel_connector, crtc_state as *const intel_crtc_state))) };
}
}
// HDMI knows no 6 bpc transport format, and DSC with an at least
// 8 bpc input provides a better output quality than a dithered
// 6 bpc output. Prefer it for HDMI sinks, by failing the
// uncompressed link config for modes which would fit only with a
// 6 bpc pipe BPP. Honor a lower limit set via the max bpc
// connector property.
let __v8: bool = !dsc && intel_dp_has_hdmi_sink(intel_dp);
let __v9: bool = __v8 && intel_dp_supports_dsc(intel_dp, connector as *const intel_connector, crtc_state as *const intel_crtc_state);
if __v9 && (unsafe { (*limits).pipe.max_bpp }) >= 24 && (unsafe { (*crtc_state).pipe_bpp }) >= 24 {
let mut __v10: i32;
let __UNIQUE_ID_x__834: i32 = unsafe { (*limits).pipe.min_bpp };
let __UNIQUE_ID_y__835: i32 = 24;
loop {
if __types_ok!(__UNIQUE_ID_x__834, __UNIQUE_ID_y__835) == 0 {
unsafe { __compiletime_assert_836() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = if __UNIQUE_ID_x__834 > __UNIQUE_ID_y__835 { __UNIQUE_ID_x__834 } else { __UNIQUE_ID_y__835 };
unsafe {
(*limits).pipe.min_bpp = __v10;
}
}
if (unsafe { (*limits).pipe.min_bpp }) <= 0 || (unsafe { (*limits).pipe.min_bpp }) > unsafe { (*limits).pipe.max_bpp } {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Invalid pipe bpp range: %d-%d\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name, (*limits).pipe.min_bpp, (*limits).pipe.max_bpp) };
return false;
}
let __v12: bool = dsc && !intel_dp_dsc_compute_pipe_bpp_limits(connector, limits);
if __v12 {
return false;
}
// crtc_state->pipe_bpp is the non-DP specific baseline (platform /
// EDID) maximum pipe BPP limited by the max-BPC connector property
// request. Since by now pipe.max_bpp is <= the above baseline
// maximum BPP, the only remaining reason for adjusting pipe.max_bpp
// is the max-BPC connector property request. Adjust pipe.max_bpp to
// this request within the current valid pipe.min_bpp .. pipe.max_bpp
// range.
let mut __v13: i32;
let __UNIQUE_ID_v__837: i32 = unsafe { (*crtc_state).pipe_bpp };
let __UNIQUE_ID_l__838: i32 = unsafe { (*limits).pipe.min_bpp };
let __UNIQUE_ID_h__839: i32 = unsafe { (*limits).pipe.max_bpp };
loop {
if !(!statically_true!(__UNIQUE_ID_l__838 > __UNIQUE_ID_h__839)) {
unsafe { __compiletime_assert_840() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __types_ok3__2!(__UNIQUE_ID_v__837, __UNIQUE_ID_l__838, __UNIQUE_ID_h__839) == 0 {
unsafe { __compiletime_assert_841() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v13 = if __UNIQUE_ID_v__837 >= __UNIQUE_ID_h__839 { __UNIQUE_ID_h__839 } else { if __UNIQUE_ID_v__837 <= __UNIQUE_ID_l__838 { __UNIQUE_ID_l__838 } else { __UNIQUE_ID_v__837 } };
unsafe {
(*limits).pipe.max_bpp = __v13;
}
if dsc {
unsafe {
(*limits).pipe.max_bpp = align_max_sink_dsc_input_bpp(connector as *const intel_connector, (*limits).pipe.max_bpp);
}
}
if (unsafe { (*limits).pipe.max_bpp }) != unsafe { (*crtc_state).pipe_bpp } {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Adjusting requested max pipe bpp %d -> %d\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name, (*crtc_state).pipe_bpp, (*limits).pipe.max_bpp) };
}
if is_mst || unsafe { (*intel_dp).use_max_params } {
let mut new_filter: intel_dp_link_caps_filter = unsafe { *{
__slate_compound_16 = intel_dp_link_caps_filter { config_mask: 0 };
core::ptr::addr_of_mut!(__slate_compound_16)
} };
let mut max_config: intel_dp_link_config = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(max_config) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
// For MST we always configure max link bw - the spec doesn't
// seem to suggest we should do otherwise.
//
// Use the maximum clock and number of lanes the eDP panel
// advertizes being capable of in case the initial fast
// optimal params failed us. The panels are generally
// designed to support only a single clock and lane
// configuration, and typically on older panels these
// values correspond to the native resolution of the panel.
if !intel_dp_get_connector_max_link_config(connector, limits as *const link_config_limits, core::ptr::addr_of_mut!(max_config)) {
return false;
}
if !unsafe { intel_dp_link_caps_filter_add(link_caps, core::ptr::addr_of_mut!(new_filter), core::ptr::addr_of_mut!(max_config) as *const intel_dp_link_config) } {
return false;
}
unsafe {
(*limits).link_config_filter = new_filter;
}
}
if !unsafe { intel_dp_test_compute_config(connector, crtc_state, limits) } {
return false;
}
intel_dp_compute_config_link_bpp_limits(connector, crtc_state as *const intel_crtc_state, dsc, limits)
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_compute_min_hblank(crtc_state: *mut intel_crtc_state, conn_state: *const drm_connector_state) -> i32 {
let display: *mut intel_display = to_intel_display__5!(crtc_state);
let adjusted_mode: *const drm_display_mode = (unsafe { core::ptr::addr_of_mut!((*crtc_state).hw.adjusted_mode) }) as *const drm_display_mode;
let connector: *mut intel_connector = to_intel_connector!(unsafe { (*conn_state).connector });
let symbol_size: i32 = if intel_dp_is_uhbr(crtc_state as *const intel_crtc_state) { 32 } else { 8 };
// min symbol cycles is 3(BS,VBID, BE) for 128b/132b and
// 5(BS, VBID, MVID, MAUD, BE) for 8b/10b
let min_sym_cycles: i32 = if intel_dp_is_uhbr(crtc_state as *const intel_crtc_state) { 3 } else { 5 };
let is_mst: bool = intel_crtc_has_type(crtc_state as *const intel_crtc_state, 11u32);
let num_joined_pipes: i32 = unsafe { intel_crtc_num_joined_pipes(crtc_state as *const intel_crtc_state) };
let mut min_hblank: i32 = 0;
let max_lane_count: i32 = 4;
let mut hactive_sym_cycles: i32 = 0;
let mut htotal_sym_cycles: i32 = 0;
let mut dsc_slices: i32 = 0;
let mut link_bpp_x16: i32 = 0;
if (DISPLAY_VER!(display) as i32) < 30 {
return 0;
}
// MIN_HBLANK should be set only for 8b/10b MST or for 128b/132b SST/MST
let __v2: bool = !is_mst && !intel_dp_is_uhbr(crtc_state as *const intel_crtc_state);
if __v2 {
return 0;
}
if unsafe { (*crtc_state).dsc.compression_enable } {
dsc_slices = intel_dp_dsc_get_slice_count(connector as *const intel_connector, unsafe { (*adjusted_mode).crtc_clock }, (unsafe { (*adjusted_mode).crtc_hdisplay }) as i32, num_joined_pipes) as i32;
if dsc_slices == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 1u32, c"failed to calculate dsc slice count\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg();
}
}
if unsafe { (*crtc_state).dsc.compression_enable } {
link_bpp_x16 = (unsafe { (*crtc_state).dsc.compressed_bpp_x16 }) as i32;
} else {
link_bpp_x16 = intel_dp_output_format_link_bpp_x16(unsafe { (*crtc_state).output_format }, unsafe { (*crtc_state).pipe_bpp });
}
// Calculate min Hblank Link Layer Symbol Cycle Count for 8b/10b MST & 128b/132b
hactive_sym_cycles = unsafe { drm_dp_link_symbol_cycles(max_lane_count, (*adjusted_mode).hdisplay as i32, dsc_slices, link_bpp_x16, symbol_size, is_mst) };
htotal_sym_cycles = ((unsafe { (*adjusted_mode).htotal }) as i32).wrapping_mul(hactive_sym_cycles) / ((unsafe { (*adjusted_mode).hdisplay }) as i32);
min_hblank = htotal_sym_cycles.wrapping_sub(hactive_sym_cycles);
// minimum Hblank calculation: https://groups.vesa.org/wg/DP/document/20494
let mut __v3: i32;
let __UNIQUE_ID_x__843: i32 = min_hblank;
let __UNIQUE_ID_y__844: i32 = min_sym_cycles;
loop {
if __types_ok!(__UNIQUE_ID_x__843, __UNIQUE_ID_y__844) == 0 {
unsafe { __compiletime_assert_845() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__843 > __UNIQUE_ID_y__844 { __UNIQUE_ID_x__843 } else { __UNIQUE_ID_y__844 };
min_hblank = __v3;
// adjust the BlankingStart/BlankingEnd framing control from
// the calculated value
min_hblank = min_hblank.wrapping_sub(2);
// min_hblank formula is undergoing a change, to avoid underrun use the
// recomended value in spec to compare with the calculated one and use the
// minimum value
if intel_dp_is_uhbr(crtc_state as *const intel_crtc_state) {
// Note: Bspec requires a min_hblank of 2 for YCBCR420
// with compressed bpp 6, but the minimum compressed bpp
// supported by the driver is 8.
let mut __v5: i64;
let mut __ret_warn_on: i32 = 0;
let __v6: bool = (unsafe { (*crtc_state).dsc.compression_enable }) && ((unsafe { (*crtc_state).output_format }) as u32) == 1 && ((unsafe { (*crtc_state).dsc.compressed_bpp_x16 }) as i32) < fxp_q4_from_int(8i32);
__ret_warn_on = __v6 as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON((crtc_state->dsc.compression_enable && crtc_state->output_format == INTEL_OUTPUT_FORMAT_YCBCR420 && crtc_state->dsc.compressed_bpp_x16 < fxp_q4_from_int(8)))".as_ptr() as *mut u8) };
let mut __v7: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v7 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v8: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_compute_min_hblank_9, sym __slate_asm_string_intel_dp_compute_min_hblank_10, const 3538 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v8 = bug;
unsafe { __v7.unwrap()(__v8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON((crtc_state->dsc.compression_enable && crtc_state->output_format == INTEL_OUTPUT_FORMAT_YCBCR420 && crtc_state->dsc.compressed_bpp_x16 < fxp_q4_from_int(8)))".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
let mut __v12: i32;
let __UNIQUE_ID_x__847: i32 = 3;
let __UNIQUE_ID_y__848: i32 = min_hblank;
loop {
if __types_ok!(__UNIQUE_ID_x__847, __UNIQUE_ID_y__848) == 0 {
unsafe { __compiletime_assert_849() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v12 = if __UNIQUE_ID_x__847 < __UNIQUE_ID_y__848 { __UNIQUE_ID_x__847 } else { __UNIQUE_ID_y__848 };
min_hblank = __v12;
} else {
let mut __v14: i32;
let __UNIQUE_ID_x__850: i32 = 10;
let __UNIQUE_ID_y__851: i32 = min_hblank;
loop {
if __types_ok!(__UNIQUE_ID_x__850, __UNIQUE_ID_y__851) == 0 {
unsafe { __compiletime_assert_852() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v14 = if __UNIQUE_ID_x__850 < __UNIQUE_ID_y__851 { __UNIQUE_ID_x__850 } else { __UNIQUE_ID_y__851 };
min_hblank = __v14;
}
unsafe {
(*crtc_state).min_hblank = min_hblank;
}
0
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_sdp_min_guardband(crtc_state: *const intel_crtc_state, assume_all_enabled: bool) -> i32 {
let mut sdp_guardband: i32 = 0;
let __v0: bool = assume_all_enabled || (unsafe { (*crtc_state).infoframes.enable }) & unsafe { intel_hdmi_infoframe_enable(10u32) } != 0;
if __v0 {
let mut __v1: i32;
let __UNIQUE_ID_x__892: i32 = sdp_guardband;
let __UNIQUE_ID_y__893: i32 = intel_dp_get_lines_for_sdp(crtc_state, 10u32);
loop {
if __types_ok!(__UNIQUE_ID_x__892, __UNIQUE_ID_y__893) == 0 {
unsafe { __compiletime_assert_894() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = if __UNIQUE_ID_x__892 > __UNIQUE_ID_y__893 { __UNIQUE_ID_x__892 } else { __UNIQUE_ID_y__893 };
sdp_guardband = __v1;
}
if assume_all_enabled || unsafe { (*crtc_state).dsc.compression_enable } {
let mut __v3: i32;
let __UNIQUE_ID_x__895: i32 = sdp_guardband;
let __UNIQUE_ID_y__896: i32 = intel_dp_get_lines_for_sdp(crtc_state, 16u32);
loop {
if __types_ok!(__UNIQUE_ID_x__895, __UNIQUE_ID_y__896) == 0 {
unsafe { __compiletime_assert_897() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__895 > __UNIQUE_ID_y__896 { __UNIQUE_ID_x__895 } else { __UNIQUE_ID_y__896 };
sdp_guardband = __v3;
}
if (unsafe { (*crtc_state).infoframes.enable }) & unsafe { intel_hdmi_infoframe_enable(34u32) } != 0 {
let mut __v5: i32;
let __UNIQUE_ID_x__898: i32 = sdp_guardband;
let __UNIQUE_ID_y__899: i32 = intel_dp_get_lines_for_sdp(crtc_state, 34u32);
loop {
if __types_ok!(__UNIQUE_ID_x__898, __UNIQUE_ID_y__899) == 0 {
unsafe { __compiletime_assert_900() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = if __UNIQUE_ID_x__898 > __UNIQUE_ID_y__899 { __UNIQUE_ID_x__898 } else { __UNIQUE_ID_y__899 };
sdp_guardband = __v5;
}
sdp_guardband
}

fn intel_dp_tunnel_detect(intel_dp: *mut intel_dp, ctx: *mut drm_modeset_acquire_ctx) -> i32 {
95i32.wrapping_neg()
}

fn intel_dp_tunnel_disconnect(intel_dp: *mut intel_dp) {
}

fn intel_dp_tunnel_destroy(intel_dp: *mut intel_dp) {
}

fn intel_dp_tunnel_resume(intel_dp: *mut intel_dp, crtc_state: *const intel_crtc_state, dpcd_updated: bool) {
}

fn intel_dp_tunnel_suspend(intel_dp: *mut intel_dp) {
}

fn intel_dp_tunnel_bw_alloc_is_enabled(intel_dp: *mut intel_dp) -> bool {
false
}

fn intel_dp_tunnel_atomic_compute_stream_bw(state: *mut intel_atomic_state, intel_dp: *mut intel_dp, connector: *const intel_connector, crtc_state: *mut intel_crtc_state) -> i32 {
0
}

fn intel_dp_tunnel_atomic_check_state(state: *mut intel_atomic_state, intel_dp: *mut intel_dp, connector: *mut intel_connector) -> i32 {
0
}

fn intel_dp_tunnel_uhbr_lanes_wa_apply(intel_dp: *mut intel_dp) {
}

fn intel_dp_tunnel_uhbr_lanes_wa_setup(intel_dp: *mut intel_dp) -> bool {
false
}

fn intel_dp_tunnel_uhbr_lanes_wa_reset(intel_dp: *mut intel_dp) {
}

// Max DSC line buffer depth supported by HW.
// DP DSC FEC Overhead factor in ppm = 1/(0.972261) = 1.028530
/// Constants for DP DSC configurations
static valid_dsc_bpp: [u8; 5] = [6, 8, 10, 12, 15];

/// intel_dp_is_edp - is the given port attached to an eDP panel (either CPU or PCH)
/// @intel_dp: DP struct
///
/// If a CPU or PCH DP output is attached to an eDP panel, this function
/// will return true, and false otherwise.
///
/// This function is not safe to use prior to encoder type being set.
#[unsafe(no_mangle)]
extern "C" fn intel_dp_is_edp(intel_dp: *mut intel_dp) -> bool {
let dig_port: *mut intel_digital_port = dp_to_dig_port(intel_dp);
((unsafe { (*dig_port).base.r#type }) as u32) == 8
}

/// Is link rate UHBR and thus 128b/132b?
#[unsafe(no_mangle)]
extern "C" fn intel_dp_is_uhbr(crtc_state: *const intel_crtc_state) -> bool {
drm_dp_is_uhbr_rate(unsafe { (*crtc_state).port_clock })
}

/// intel_dp_link_symbol_size - get the link symbol size for a given link rate
/// @rate: link rate in 10kbit/s units
///
/// Returns the link symbol size in bits/symbol units depending on the link
/// rate -> channel coding.
#[unsafe(no_mangle)]
extern "C" fn intel_dp_link_symbol_size(rate: i32) -> i32 {
if drm_dp_is_uhbr_rate(rate) { 32 } else { 10 }
}

/// intel_dp_link_symbol_clock - convert link rate to link symbol clock
/// @rate: link rate in 10kbit/s units
///
/// Returns the link symbol clock frequency in kHz units depending on the
/// link rate and channel coding.
#[unsafe(no_mangle)]
extern "C" fn intel_dp_link_symbol_clock(rate: i32) -> i32 {
let mut __v0: i32;
let __x: i32 = rate.wrapping_mul(10);
let __d: i32 = intel_dp_link_symbol_size(rate);
__v0 = if 1i32.wrapping_neg() > 0 || 1i32.wrapping_neg() > 0 || ((__x > 0) as i32) == ((__d > 0) as i32) { __x.wrapping_add(__d / 2) / __d } else { __x.wrapping_sub(__d / 2) / __d };
__v0
}

fn max_dprx_rate(intel_dp: *mut intel_dp) -> i32 {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let mut max_rate: i32 = if intel_dp_tunnel_bw_alloc_is_enabled(intel_dp) {
drm_dp_tunnel_max_dprx_rate((unsafe { (*intel_dp).tunnel }) as *const drm_dp_tunnel)
} else {
unsafe { drm_dp_bw_code_to_link_rate(*((*intel_dp).dpcd.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize)) }
};
// Some platforms + eDP panels may not reliably support HBR3
// due to signal integrity limitations, despite advertising it.
// Cap the link rate to HBR2 to avoid unstable configurations for the
// known machines.
let __v1: bool = intel_dp_is_edp(intel_dp) && unsafe { intel_has_quirk(display, 7u32) };
if __v1 {
let mut __v2: i32;
let __UNIQUE_ID_x__714: i32 = max_rate;
let __UNIQUE_ID_y__715: i32 = 540000;
loop {
if __types_ok!(__UNIQUE_ID_x__714, __UNIQUE_ID_y__715) == 0 {
unsafe { __compiletime_assert_716() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__714 < __UNIQUE_ID_y__715 { __UNIQUE_ID_x__714 } else { __UNIQUE_ID_y__715 };
max_rate = __v2;
}
max_rate
}

fn max_dprx_lane_count(intel_dp: *mut intel_dp) -> i32 {
if intel_dp_tunnel_bw_alloc_is_enabled(intel_dp) {
return drm_dp_tunnel_max_dprx_lane_count((unsafe { (*intel_dp).tunnel }) as *const drm_dp_tunnel);
}
drm_dp_max_lane_count((unsafe { (*intel_dp).dpcd.as_mut_ptr() as *mut u8 }) as *const u8) as i32
}

fn intel_dp_set_default_sink_rates(intel_dp: *mut intel_dp) {
unsafe {
*((*intel_dp).sink_rates.as_mut_ptr() as *mut i32).wrapping_offset(0 as isize) = 162000;
}
unsafe {
(*intel_dp).num_sink_rates = 1;
}
}

fn dprx_supports_128b132b(intel_dp: *mut intel_dp) -> bool {
if intel_dp_tunnel_bw_alloc_is_enabled(intel_dp) {
return drm_dp_tunnel_128b132b_supported((unsafe { (*intel_dp).tunnel }) as *const drm_dp_tunnel);
} else {
return drm_dp_128b132b_supported((unsafe { (*intel_dp).dpcd.as_mut_ptr() as *mut u8 }) as *const u8);
}
}

fn dprx_128b132b_link_rates(intel_dp: *mut intel_dp, no_bwa_rates: u8) -> u8 {
let mut ret: u8 = 0;
if !intel_dp_tunnel_bw_alloc_is_enabled(intel_dp) {
return no_bwa_rates;
}
ret = drm_dp_tunnel_128b132b_dprx_rates((unsafe { (*intel_dp).tunnel }) as *const drm_dp_tunnel);
ret
}

// update sink rates from dpcd
fn intel_dp_set_dpcd_sink_rates(intel_dp: *mut intel_dp) {
let mut i: i32 = 0;
let mut max_rate: i32 = 0;
let mut max_lttpr_rate: i32 = 0;
let mut uhbr_rates: u8 = 0;
if drm_dp_has_quirk((unsafe { core::ptr::addr_of_mut!((*intel_dp).desc) }) as *const drm_dp_desc, 4u32) {
// Needed, e.g., for Apple MBP 2017, 15 inch eDP Retina panel
unsafe { memcpy(((*intel_dp).sink_rates.as_mut_ptr() as *mut i32) as *mut (), (core::ptr::addr_of!(quirk_rates) as *const i32) as *const (), core::mem::size_of::<[i32; 3]>() as u64) };
unsafe {
(*intel_dp).num_sink_rates = ARRAY_SIZE__97!() as i32;
}
return;
}
// Sink rates for 8b/10b.
max_rate = max_dprx_rate(intel_dp);
max_lttpr_rate = unsafe { drm_dp_lttpr_max_link_rate(((*intel_dp).lttpr_common_caps.as_mut_ptr() as *mut u8) as *const u8) };
if max_lttpr_rate != 0 {
let mut __v0: i32;
let __UNIQUE_ID_x__717: i32 = max_rate;
let __UNIQUE_ID_y__718: i32 = max_lttpr_rate;
loop {
if __types_ok!(__UNIQUE_ID_x__717, __UNIQUE_ID_y__718) == 0 {
unsafe { __compiletime_assert_719() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__717 < __UNIQUE_ID_y__718 { __UNIQUE_ID_x__717 } else { __UNIQUE_ID_y__718 };
max_rate = __v0;
}
while (i as u64) < ARRAY_SIZE__30!() {
if (unsafe { dp_rates.0[i as usize] }) > max_rate {
break;
}
unsafe {
*((*intel_dp).sink_rates.as_mut_ptr() as *mut i32).wrapping_offset(i as isize) = dp_rates.0[i as usize];
}
i = i.wrapping_add(1);
}
// The following register must be read unconditionally for the later
// DP tunnel 128b132b detection to work, see DP Standard v2.1 5.14.3 .
drm_dp_dpcd_read_byte(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 8725u32, core::ptr::addr_of_mut!(uhbr_rates));
// Sink rates for 128b/132b. If set, sink should support all 8b/10b
// rates and 10 Gbps.
if dprx_supports_128b132b(intel_dp) {
loop {
if ARRAY_SIZE__65!() < ARRAY_SIZE__171!().wrapping_add(3) {
unsafe { __compiletime_assert_720() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
uhbr_rates = dprx_128b132b_link_rates(intel_dp, uhbr_rates);
if (unsafe { drm_dp_lttpr_count(((*intel_dp).lttpr_common_caps.as_mut_ptr() as *mut u8) as *const u8) }) != 0 {
// We have a repeater
if ((unsafe { *((*intel_dp).lttpr_common_caps.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) >= 32 && ((unsafe { *((*intel_dp).lttpr_common_caps.as_mut_ptr() as *mut u8).wrapping_offset(6i32 as isize) }) as i32) & 1i32 != 0 {
// Repeater supports 128b/132b, valid UHBR rates
let __v6: u8 = uhbr_rates;
uhbr_rates = ((__v6 as i32) & ((unsafe { *((*intel_dp).lttpr_common_caps.as_mut_ptr() as *mut u8).wrapping_offset(7i32 as isize) }) as i32)) as u8;
} else {
// Does not support 128b/132b
uhbr_rates = 0;
}
}
if (uhbr_rates as i32) & 1i32 != 0 {
let __v8: i32 = i;
i = __v8.wrapping_add(1);
unsafe {
*((*intel_dp).sink_rates.as_mut_ptr() as *mut i32).wrapping_offset(__v8 as isize) = 1000000;
}
}
if (uhbr_rates as i32) & 4i32 != 0 {
let __v10: i32 = i;
i = __v10.wrapping_add(1);
unsafe {
*((*intel_dp).sink_rates.as_mut_ptr() as *mut i32).wrapping_offset(__v10 as isize) = 1350000;
}
}
if (uhbr_rates as i32) & 2i32 != 0 {
let __v12: i32 = i;
i = __v12.wrapping_add(1);
unsafe {
*((*intel_dp).sink_rates.as_mut_ptr() as *mut i32).wrapping_offset(__v12 as isize) = 2000000;
}
}
}
unsafe {
(*intel_dp).num_sink_rates = i;
}
}

static dp_rates: __SlateAlign16<[i32; 4]> = __SlateAlign16([162000i32, 270000i32, 540000i32, 810000i32]);

static quirk_rates: [i32; 3] = [162000, 270000, 324000];

fn intel_dp_set_sink_rates(intel_dp: *mut intel_dp) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let intel_dig_port: *mut intel_digital_port = dp_to_dig_port(intel_dp);
let encoder: *mut intel_encoder = unsafe { core::ptr::addr_of_mut!((*intel_dig_port).base) };
intel_dp_set_dpcd_sink_rates(intel_dp);
if (unsafe { (*intel_dp).num_sink_rates }) != 0 {
return;
}
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* [CONNECTOR:%d:%s][ENCODER:%d:%s] Invalid DPCD with no link rates, using defaults\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name, (*encoder).base.base.id, (*encoder).base.name) };
intel_dp_set_default_sink_rates(intel_dp);
}

fn intel_dp_set_default_max_sink_lane_count(intel_dp: *mut intel_dp) {
unsafe {
(*intel_dp).max_sink_lane_count = 1;
}
}

fn intel_dp_set_max_sink_lane_count(intel_dp: *mut intel_dp) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let intel_dig_port: *mut intel_digital_port = dp_to_dig_port(intel_dp);
let encoder: *mut intel_encoder = unsafe { core::ptr::addr_of_mut!((*intel_dig_port).base) };
unsafe {
(*intel_dp).max_sink_lane_count = max_dprx_lane_count(intel_dp);
}
match unsafe { (*intel_dp).max_sink_lane_count } {
1 | 2 | 4 => {
return;
}
_ => {
}
}
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* [CONNECTOR:%d:%s][ENCODER:%d:%s] Invalid DPCD max lane count (%d), using default\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name, (*encoder).base.base.id, (*encoder).base.name, (*intel_dp).max_sink_lane_count) };
intel_dp_set_default_max_sink_lane_count(intel_dp);
}

/// Get length of rates array potentially limited by max_rate.
#[unsafe(no_mangle)]
extern "C" fn intel_dp_rate_limit_len(rates: *const i32, len: i32, max_rate: i32) -> i32 {
// Limit results by potentially reduced max rate
for i in 0..len {
if (unsafe { *rates.wrapping_offset(len.wrapping_sub(i).wrapping_sub(1) as isize) }) <= max_rate {
return len.wrapping_sub(i);
}
}
0
}

// Theoretical max between source and sink.
fn intel_dp_get_max_common_lane_count(intel_dp: *mut intel_dp) -> i32 {
let dig_port: *mut intel_digital_port = dp_to_dig_port(intel_dp);
let source_max: i32 = intel_dp_max_source_lane_count(dig_port);
let mut sink_max: i32 = unsafe { (*intel_dp).max_sink_lane_count };
let lane_max: i32 = unsafe { intel_tc_port_max_lane_count(dig_port) };
let lttpr_max: i32 = unsafe { drm_dp_lttpr_max_lane_count(((*intel_dp).lttpr_common_caps.as_mut_ptr() as *mut u8) as *const u8) };
if lttpr_max != 0 {
let mut __v0: i32;
let __UNIQUE_ID_x__724: i32 = sink_max;
let __UNIQUE_ID_y__725: i32 = lttpr_max;
loop {
if __types_ok!(__UNIQUE_ID_x__724, __UNIQUE_ID_y__725) == 0 {
unsafe { __compiletime_assert_726() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__724 < __UNIQUE_ID_y__725 { __UNIQUE_ID_x__724 } else { __UNIQUE_ID_y__725 };
sink_max = __v0;
}
let mut __v2: i32;
let __UNIQUE_ID_x__727: i32 = source_max;
let __UNIQUE_ID_y__728: i32 = sink_max;
let __UNIQUE_ID_z__729: i32 = lane_max;
loop {
if __types_ok3__2!(__UNIQUE_ID_x__727, __UNIQUE_ID_y__728, __UNIQUE_ID_z__729) == 0 {
unsafe { __compiletime_assert_730() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__727 < if __UNIQUE_ID_y__728 < __UNIQUE_ID_z__729 { __UNIQUE_ID_y__728 } else { __UNIQUE_ID_z__729 } { __UNIQUE_ID_x__727 } else { if __UNIQUE_ID_y__728 < __UNIQUE_ID_z__729 { __UNIQUE_ID_y__728 } else { __UNIQUE_ID_z__729 } };
__v2
}

/// The required data bandwidth for a mode with given pixel clock and bpp. This
/// is the required net bandwidth independent of the data bandwidth efficiency.
#[unsafe(no_mangle)]
extern "C" fn intel_dp_link_required(link_clock: i32, lane_count: i32, mode_clock: i32, mode_hdisplay: i32, link_bpp_x16: i32, bw_overhead_flags: u64) -> i32 {
let bw_overhead: i32 = intel_dp_link_bw_overhead(link_clock, lane_count, mode_hdisplay, 0i32, link_bpp_x16, bw_overhead_flags);
intel_dp_effective_data_rate(mode_clock, link_bpp_x16, bw_overhead)
}

/// intel_dp_effective_data_rate - Return the pixel data rate accounting for BW allocation overhead
/// @pixel_clock: pixel clock in kHz
/// @bpp_x16: bits per pixel .4 fixed point format
/// @bw_overhead: BW allocation overhead in 1ppm units
///
/// Return the effective pixel data rate in kB/sec units taking into account
/// the provided SSC, FEC, DSC BW allocation overhead.
#[unsafe(no_mangle)]
extern "C" fn intel_dp_effective_data_rate(pixel_clock: i32, bpp_x16: i32, bw_overhead: i32) -> i32 {
DIV_ROUND_UP_ULL!(mul_u32_u32(pixel_clock.wrapping_mul(bpp_x16) as u32, bw_overhead as u32), 128000000i32) as i32
}

// intel_dp_max_link_data_rate: Calculate the maximum rate for the given link params
// @intel_dp: Intel DP object
// @max_dprx_rate: Maximum data rate of the DPRX
// @max_dprx_lanes: Maximum lane count of the DPRX
//
// Calculate the maximum data rate for the provided link parameters taking into
// account any BW limitations by a DP tunnel attached to @intel_dp.
//
// Returns the maximum data rate in kBps units.
#[unsafe(no_mangle)]
extern "C" fn intel_dp_has_joiner(intel_dp: *mut intel_dp) -> bool {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let intel_dig_port: *mut intel_digital_port = dp_to_dig_port(intel_dp);
let encoder: *mut intel_encoder = unsafe { core::ptr::addr_of_mut!((*intel_dig_port).base) };
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
// eDP MSO is not compatible with joiner
if (unsafe { (*intel_dp).mso_link_count }) != 0 {
return false;
}
if intel_dp_is_edp(intel_dp) && !unsafe { (*connector).panel.vbt.edp.pipe_joiner_enable } {
return false;
}
(DISPLAY_VER!(display) as i32) >= 12 || (DISPLAY_VER!(display) as i32) == 11 && ((unsafe { (*encoder).port }) as i32) != 0
}

fn dg2_max_source_rate(intel_dp: *mut intel_dp) -> i32 {
if intel_dp_is_edp(intel_dp) { 810000 } else { 1350000 }
}

fn icl_max_source_rate(intel_dp: *mut intel_dp) -> i32 {
let encoder: *mut intel_encoder = unsafe { core::ptr::addr_of_mut!((*dp_to_dig_port(intel_dp)).base) };
let __v0: bool = (unsafe { intel_encoder_is_combo(encoder) }) && !intel_dp_is_edp(intel_dp);
if __v0 {
return 540000;
}
810000
}

fn ehl_max_source_rate(intel_dp: *mut intel_dp) -> i32 {
if intel_dp_is_edp(intel_dp) {
return 540000;
}
810000
}

fn mtl_max_source_rate(intel_dp: *mut intel_dp) -> i32 {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let encoder: *mut intel_encoder = unsafe { core::ptr::addr_of_mut!((*dp_to_dig_port(intel_dp)).base) };
if (unsafe { intel_encoder_is_c10phy(encoder) }) || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_pantherlake_wildcatlake() }) as i32) != 0 {
return 810000;
}
if DISPLAY_VERx100!(display) == 1401 {
return 1350000;
}
2000000
}

fn vbt_max_link_rate(intel_dp: *mut intel_dp) -> i32 {
let encoder: *mut intel_encoder = unsafe { core::ptr::addr_of_mut!((*dp_to_dig_port(intel_dp)).base) };
let mut max_rate: i32 = unsafe { intel_bios_dp_max_link_rate((*encoder).devdata) };
if intel_dp_is_edp(intel_dp) {
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let edp_max_rate: i32 = unsafe { (*connector).panel.vbt.edp.max_link_rate };
if max_rate != 0 && edp_max_rate != 0 {
let mut __v0: i32;
let __UNIQUE_ID_x__739: i32 = max_rate;
let __UNIQUE_ID_y__740: i32 = edp_max_rate;
loop {
if __types_ok!(__UNIQUE_ID_x__739, __UNIQUE_ID_y__740) == 0 {
unsafe { __compiletime_assert_741() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__739 < __UNIQUE_ID_y__740 { __UNIQUE_ID_x__739 } else { __UNIQUE_ID_y__740 };
max_rate = __v0;
} else {
if edp_max_rate != 0 {
max_rate = edp_max_rate;
}
}
}
max_rate
}

fn intel_dp_set_source_rates(intel_dp: *mut intel_dp) {
// The values must be in increasing order
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let mut source_rates: *const i32 = unsafe { core::mem::zeroed() };
let mut size: i32 = 0;
let mut max_rate: i32 = 0;
let mut vbt_max_rate: i32 = 0;
// This should only be done once
let mut __v1: i64;
let __ret_warn_on: i32 = (!unsafe { (*intel_dp).source_rates }.is_null() || (unsafe { (*intel_dp).num_source_rates }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(intel_dp->source_rates || intel_dp->num_source_rates)".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_set_source_rates_4, sym __slate_asm_string_intel_dp_set_source_rates_5, const 575 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(intel_dp->source_rates || intel_dp->num_source_rates)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if (DISPLAY_VER!(display) as i32) >= 14 {
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_battlemage() }) as i32) != 0 {
source_rates = unsafe { core::ptr::addr_of!(bmg_rates.0) as *const i32 };
size = ARRAY_SIZE__9!() as i32;
} else {
source_rates = unsafe { core::ptr::addr_of!(mtl_rates.0) as *const i32 };
size = ARRAY_SIZE__192!() as i32;
}
max_rate = mtl_max_source_rate(intel_dp);
} else {
if (DISPLAY_VER!(display) as i32) >= 11 {
source_rates = unsafe { core::ptr::addr_of!(icl_rates.0) as *const i32 };
size = ARRAY_SIZE__25!() as i32;
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_dg2() }) as i32) != 0 {
max_rate = dg2_max_source_rate(intel_dp);
} else {
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_alderlake_p() }) as i32) != 0 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_alderlake_s() }) as i32) != 0 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_dg1() }) as i32) != 0 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_rocketlake() }) as i32) != 0 {
max_rate = 810000;
} else {
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_jasperlake() }) as i32) != 0 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_elkhartlake() }) as i32) != 0 {
max_rate = ehl_max_source_rate(intel_dp);
} else {
max_rate = icl_max_source_rate(intel_dp);
}
}
}
} else {
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_geminilake() }) as i32) != 0 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_broxton() }) as i32) != 0 {
source_rates = unsafe { core::ptr::addr_of!(bxt_rates.0) as *const i32 };
size = ARRAY_SIZE__172!() as i32;
} else {
if (DISPLAY_VER!(display) as i32) == 9 {
source_rates = unsafe { core::ptr::addr_of!(skl_rates.0) as *const i32 };
size = ARRAY_SIZE__2!() as i32;
} else {
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_haswell() }) as i32) != 0 && ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_haswell_ulx() }) as i32) == 0 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_broadwell() }) as i32) != 0 {
source_rates = unsafe { core::ptr::addr_of!(hsw_rates) as *const i32 };
size = ARRAY_SIZE__52!() as i32;
} else {
source_rates = unsafe { core::ptr::addr_of!(g4x_rates) as *const i32 };
size = ARRAY_SIZE__181!() as i32;
}
}
}
}
}
vbt_max_rate = vbt_max_link_rate(intel_dp);
if max_rate != 0 && vbt_max_rate != 0 {
let mut __v7: i32;
let __UNIQUE_ID_x__743: i32 = max_rate;
let __UNIQUE_ID_y__744: i32 = vbt_max_rate;
loop {
if __types_ok!(__UNIQUE_ID_x__743, __UNIQUE_ID_y__744) == 0 {
unsafe { __compiletime_assert_745() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v7 = if __UNIQUE_ID_x__743 < __UNIQUE_ID_y__744 { __UNIQUE_ID_x__743 } else { __UNIQUE_ID_y__744 };
max_rate = __v7;
} else {
if vbt_max_rate != 0 {
max_rate = vbt_max_rate;
}
}
if max_rate != 0 {
size = intel_dp_rate_limit_len(source_rates, size, max_rate);
}
unsafe {
(*intel_dp).source_rates = source_rates;
}
unsafe {
(*intel_dp).num_source_rates = size;
}
}

static bmg_rates: __SlateAlign16<[i32; 11]> = __SlateAlign16([162000i32, 216000i32, 243000i32, 270000i32, 324000i32, 432000i32, 540000i32, 675000i32, 810000i32, 1000000i32, 1350000i32]);

static mtl_rates: __SlateAlign16<[i32; 11]> = __SlateAlign16([162000i32, 216000i32, 243000i32, 270000i32, 324000i32, 432000i32, 540000i32, 675000i32, 810000i32, 1000000i32, 2000000i32]);

static icl_rates: __SlateAlign16<[i32; 10]> = __SlateAlign16([162000i32, 216000i32, 270000i32, 324000i32, 432000i32, 540000i32, 648000i32, 810000i32, 1000000i32, 1350000i32]);

static bxt_rates: __SlateAlign16<[i32; 7]> = __SlateAlign16([162000i32, 216000i32, 243000i32, 270000i32, 324000i32, 432000i32, 540000i32]);

static skl_rates: __SlateAlign16<[i32; 6]> = __SlateAlign16([162000i32, 216000i32, 270000i32, 324000i32, 432000i32, 540000i32]);

static hsw_rates: [i32; 3] = [162000, 270000, 540000];

static g4x_rates: [i32; 2] = [162000, 270000];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_742: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intersect_rates(source_rates: *const i32, source_len: i32, sink_rates: *const i32, sink_len: i32, common_rates: *mut i32) -> i32 {
let mut i: i32 = 0;
let mut j: i32 = 0;
let mut k: i32 = 0;
while i < source_len && j < sink_len {
if (unsafe { *source_rates.wrapping_offset(i as isize) }) == unsafe { *sink_rates.wrapping_offset(j as isize) } {
let mut __v0: i64;
let __ret_warn_on: i32 = (k >= 8) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("746: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 746b - ., 3; .popsection\n# {0}", const 746 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_intersect_rates_1, sym __slate_asm_string_intersect_rates_2, const 634 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("747: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 747b - ., 4; .popsection\n# {0}", const 747 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return k;
}
unsafe {
*common_rates.wrapping_offset(k as isize) = *source_rates.wrapping_offset(i as isize);
}
k = k.wrapping_add(1);
i = i.wrapping_add(1);
j = j.wrapping_add(1);
} else {
if (unsafe { *source_rates.wrapping_offset(i as isize) }) < unsafe { *sink_rates.wrapping_offset(j as isize) } {
i = i.wrapping_add(1);
} else {
j = j.wrapping_add(1);
}
}
}
k
}

/// return index of rate in rates array, or -1 if not found
#[unsafe(no_mangle)]
extern "C" fn intel_dp_rate_index(rates: *const i32, len: i32, rate: i32) -> i32 {
for i in 0..len {
if rate == unsafe { *rates.wrapping_offset(i as isize) } {
return i;
}
}
1i32.wrapping_neg()
}

fn intel_dp_get_common_rates(intel_dp: *mut intel_dp, common_rates: *mut i32, num_common_rates: &mut i32) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let mut __v1: i64;
{
let __ret_warn_on: i32 = ((unsafe { (*intel_dp).num_source_rates }) == 0 || (unsafe { (*intel_dp).num_sink_rates }) == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!intel_dp->num_source_rates || !intel_dp->num_sink_rates)".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_get_common_rates_4, sym __slate_asm_string_intel_dp_get_common_rates_5, const 668 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!intel_dp->num_source_rates || !intel_dp->num_sink_rates)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
}
unsafe {
*(num_common_rates as *mut i32) = intersect_rates((*intel_dp).source_rates, (*intel_dp).num_source_rates, ((*intel_dp).sink_rates.as_mut_ptr() as *mut i32) as *const i32, (*intel_dp).num_sink_rates, common_rates);
}
// Paranoia, there should always be something in common.
let mut __v7: i64;
let __ret_warn_on: i32 = ((unsafe { *(num_common_rates as *mut i32) }) == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(*num_common_rates == 0)".as_ptr() as *mut u8) };
let mut __v8: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v8 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v9: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t1: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t1, sym __slate_asm_string_intel_dp_get_common_rates_10, sym __slate_asm_string_intel_dp_get_common_rates_11, const 677 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t1 as u64) };
}
__v9 = bug;
unsafe { __v8.unwrap()(__v9, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(*num_common_rates == 0)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v7 = (__ret_warn_on != 0) as i64;
if __v7 != 0 {
unsafe {
*common_rates.wrapping_offset(0 as isize) = 162000;
}
unsafe {
*(num_common_rates as *mut i32) = 1;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_748: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_749: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

/// Return %true if any common link param changed.
fn intel_dp_set_common_link_params(intel_dp: *mut intel_dp) -> bool {
let mut num_common_rates: i32 = 0;
let mut common_rates: __SlateAlign16<[i32; 8]> = __SlateAlign16([0i32; 8]);
let mut params_changed: bool = false;
intel_dp_get_common_rates(intel_dp, common_rates.0.as_mut_ptr() as *mut i32, unsafe { &mut (*core::ptr::addr_of_mut!(num_common_rates)) });
if unsafe { intel_dp_link_caps_update((*intel_dp).link.caps, common_rates.0.as_mut_ptr() as *const i32, num_common_rates, intel_dp_get_max_common_lane_count(intel_dp), (*intel_dp).reset_link_params) } {
params_changed = true;
}
params_changed
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_mode_to_fec_clock(mode_clock: u32) -> u32 {
div_u64(mul_u32_u32(mode_clock, 1028530u32), 1000000u32) as u32
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_bw_fec_overhead(fec_enabled: bool) -> i32 {
// TODO: Calculate the actual overhead for a given mode.
// The hard-coded 1/0.972261=2.853% overhead factor
// corresponds (for instance) to the 8b/10b DP FEC 2.4% +
// 0.453% DSC overhead. This is enough for a 3840 width mode,
// which has a DSC overhead of up to ~0.2%, but may not be
// enough for a 1024 width mode where this is ~0.8% (on a 4
// lane DP link, with 2 DSC slices and 8 bpp color depth).
if fec_enabled { 1028530 } else { 1000000 }
}

fn small_joiner_ram_size_bits(display: *mut intel_display) -> i32 {
if (DISPLAY_VER!(display) as i32) >= 13 {
return 138240i32;
} else {
if (DISPLAY_VER!(display) as i32) >= 11 {
return 61440i32;
} else {
return 49152i32;
}
}
}

fn align_min_vesa_compressed_bpp_x16(min_link_bpp_x16: i32) -> i32 {
let mut i: i32 = 0;
while (i as u64) < ARRAY_SIZE__149!() {
let vesa_bpp_x16: i32 = fxp_q4_from_int((unsafe { valid_dsc_bpp[i as usize] }) as i32);
if vesa_bpp_x16 >= min_link_bpp_x16 {
return vesa_bpp_x16;
}
i = i.wrapping_add(1);
}
0
}

fn align_max_vesa_compressed_bpp_x16(max_link_bpp_x16: i32) -> i32 {
let mut i: i32 = ARRAY_SIZE__83!().wrapping_sub(1) as i32;
while i >= 0 {
let vesa_bpp_x16: i32 = fxp_q4_from_int((unsafe { valid_dsc_bpp[i as usize] }) as i32);
if vesa_bpp_x16 <= max_link_bpp_x16 {
return vesa_bpp_x16;
}
i = i.wrapping_sub(1);
}
0
}

fn bigjoiner_interface_bits(display: *mut intel_display) -> i32 {
if (DISPLAY_VER!(display) as i32) >= 14 { 36 } else { 24 }
}

fn bigjoiner_bw_max_bpp(display: *mut intel_display, mode_clock: u32, num_joined_pipes: i32) -> u32 {
let mut max_bpp: u32 = 0;
// With bigjoiner multiple dsc engines are used in parallel so PPC is 2
let ppc: i32 = 2;
let num_big_joiners: i32 = num_joined_pipes / 2;
max_bpp = unsafe { (*display).cdclk.max_cdclk_freq }.wrapping_mul(ppc as u32).wrapping_mul(bigjoiner_interface_bits(display) as u32) / intel_dp_mode_to_fec_clock(mode_clock);
max_bpp = max_bpp.wrapping_mul(num_big_joiners as u32);
max_bpp
}

fn small_joiner_ram_max_bpp(display: *mut intel_display, mode_hdisplay: u32, num_joined_pipes: i32) -> u32 {
let mut max_bpp: u32 = 0;
// Small Joiner Check: output bpp <= joiner RAM (bits) / Horiz. width
max_bpp = (small_joiner_ram_size_bits(display) as u32) / mode_hdisplay;
max_bpp = max_bpp.wrapping_mul(num_joined_pipes as u32);
max_bpp
}

fn ultrajoiner_ram_bits() -> i32 {
147456i32
}

fn ultrajoiner_ram_max_bpp(mode_hdisplay: u32) -> u32 {
(ultrajoiner_ram_bits() as u32) / mode_hdisplay
}

// TODO: return a bpp_x16 value
fn get_max_compressed_bpp_with_joiner(display: *mut intel_display, mode_clock: u32, mode_hdisplay: u32, num_joined_pipes: i32) -> u32 {
let mut max_bpp: u32 = small_joiner_ram_max_bpp(display, mode_hdisplay, num_joined_pipes);
if num_joined_pipes > 1 {
let mut __v0: u32;
let __UNIQUE_ID_x__750: u32 = max_bpp;
let __UNIQUE_ID_y__751: u32 = bigjoiner_bw_max_bpp(display, mode_clock, num_joined_pipes);
loop {
if __types_ok__2!(__UNIQUE_ID_x__750, __UNIQUE_ID_y__751) == 0 {
unsafe { __compiletime_assert_752() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__750 < __UNIQUE_ID_y__751 { __UNIQUE_ID_x__750 } else { __UNIQUE_ID_y__751 };
max_bpp = __v0;
}
if num_joined_pipes == 4 {
let mut __v2: u32;
let __UNIQUE_ID_x__753: u32 = max_bpp;
let __UNIQUE_ID_y__754: u32 = ultrajoiner_ram_max_bpp(mode_hdisplay);
loop {
if __types_ok__2!(__UNIQUE_ID_x__753, __UNIQUE_ID_y__754) == 0 {
unsafe { __compiletime_assert_755() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__753 < __UNIQUE_ID_y__754 { __UNIQUE_ID_x__753 } else { __UNIQUE_ID_y__754 };
max_bpp = __v2;
}
max_bpp
}

fn intel_dp_dsc_min_slice_count(connector: *const intel_connector, mode_clock: i32, mode_hdisplay: i32) -> i32 {
let display: *mut intel_display = to_intel_display__10!(connector);
let is_edp: bool = (unsafe { (*connector).base.connector_type }) == 14;
let mut min_slice_count: i32 = 0;
let mut max_slice_width: i32 = 0;
let mut tp_rgb_yuv444: i32 = 0;
let mut tp_yuv422_420: i32 = 0;
// TODO: allow using less than the maximum number of slices
// supported by the eDP sink, to allow using fewer DSC engines.
if is_edp {
return (unsafe { drm_dp_dsc_sink_max_slice_count((*connector).dp.dsc_dpcd.as_ptr() as *const u8, true) }) as i32;
}
// TODO: Use the throughput value specific to the actual RGB/YUV
// format of the output.
// The RGB/YUV444 throughput value should be always either equal
// or smaller than the YUV422/420 value, but let's not depend on
// this assumption.
let mut __v1: i32;
let __UNIQUE_ID_x__756: i32 = unsafe { (*connector).dp.dsc_branch_caps.overall_throughput.rgb_yuv444 };
let __UNIQUE_ID_y__757: i32 = unsafe { (*connector).dp.dsc_branch_caps.overall_throughput.yuv422_420 };
loop {
if __types_ok!(__UNIQUE_ID_x__756, __UNIQUE_ID_y__757) == 0 {
unsafe { __compiletime_assert_758() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = if __UNIQUE_ID_x__756 > __UNIQUE_ID_y__757 { __UNIQUE_ID_x__756 } else { __UNIQUE_ID_y__757 };
if mode_clock > __v1 {
return 0;
}
if mode_hdisplay > unsafe { (*connector).dp.dsc_branch_caps.max_line_width } {
return 0;
}
// TODO: Pass the total pixel rate of all the streams transferred to
// an MST tiled display, calculate the total slice count for all tiles
// from this and the per-tile slice count from the total slice count.
tp_rgb_yuv444 = unsafe { drm_dp_dsc_sink_max_slice_throughput((*connector).dp.dsc_dpcd.as_ptr() as *const u8, mode_clock, true) };
tp_yuv422_420 = unsafe { drm_dp_dsc_sink_max_slice_throughput((*connector).dp.dsc_dpcd.as_ptr() as *const u8, mode_clock, false) };
// TODO: Use the throughput value specific to the actual RGB/YUV
// format of the output.
// For now use the smaller of these, which is ok, potentially
// resulting in a higher than required minimum slice count.
// The RGB/YUV444 throughput value should be always either equal
// or smaller than the YUV422/420 value, but let's not depend on
// this assumption.
let mut __v3: i32;
{
let __UNIQUE_ID_x__759: i32 = tp_rgb_yuv444;
let __UNIQUE_ID_y__760: i32 = tp_yuv422_420;
loop {
if (if 1i32.wrapping_neg() < 1 { 2i32.wrapping_add(((unsafe { __slate_intrinsic_1cfe0293d0a2af8d((__UNIQUE_ID_x__759 as i64) >= 0) }) && (__UNIQUE_ID_x__759 as i64) >= 0) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }) & if 1i32.wrapping_neg() < 1 { 2i32.wrapping_add(((unsafe { __slate_intrinsic_1cfe0293d0a2af8d((__UNIQUE_ID_y__760 as i64) >= 0) }) && (__UNIQUE_ID_y__760 as i64) >= 0) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) } == 0 {
unsafe { __compiletime_assert_761() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__759 < __UNIQUE_ID_y__760 { __UNIQUE_ID_x__759 } else { __UNIQUE_ID_y__760 };
}
let mut __v5: i32;
let __UNIQUE_ID_x__759: i32 = tp_rgb_yuv444;
let __UNIQUE_ID_y__760: i32 = tp_yuv422_420;
loop {
if (if 1i32.wrapping_neg() < 1 { 2i32.wrapping_add(((unsafe { __slate_intrinsic_1cfe0293d0a2af8d((__UNIQUE_ID_x__759 as i64) >= 0) }) && (__UNIQUE_ID_x__759 as i64) >= 0) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }) & if 1i32.wrapping_neg() < 1 { 2i32.wrapping_add(((unsafe { __slate_intrinsic_1cfe0293d0a2af8d((__UNIQUE_ID_y__760 as i64) >= 0) }) && (__UNIQUE_ID_y__760 as i64) >= 0) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) } == 0 {
unsafe { __compiletime_assert_761() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = if __UNIQUE_ID_x__759 < __UNIQUE_ID_y__760 { __UNIQUE_ID_x__759 } else { __UNIQUE_ID_y__760 };
min_slice_count = mode_clock.wrapping_add(__v3).wrapping_sub(1) / __v5;
// Due to some DSC engine BW limitations, we need to enable second
// slice and VDSC engine, whenever we approach close enough to max CDCLK
if (mode_clock as u32) >= unsafe { (*display).cdclk.max_cdclk_freq }.wrapping_mul(85) / 100 {
let mut __v7: i32;
let __UNIQUE_ID_x__762: i32 = min_slice_count;
let __UNIQUE_ID_y__763: i32 = 2;
loop {
if __types_ok!(__UNIQUE_ID_x__762, __UNIQUE_ID_y__763) == 0 {
unsafe { __compiletime_assert_764() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v7 = if __UNIQUE_ID_x__762 > __UNIQUE_ID_y__763 { __UNIQUE_ID_x__762 } else { __UNIQUE_ID_y__763 };
min_slice_count = __v7;
}
max_slice_width = drm_dp_dsc_sink_max_slice_width(unsafe { (*connector).dp.dsc_dpcd.as_ptr() as *const u8 }) as i32;
if max_slice_width < 2560 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Unsupported slice width %d by DP DSC Sink device\n".as_ptr() as *const u8, max_slice_width) };
return 0;
}
// Also take into account max slice width
let mut __v9: i32;
let __UNIQUE_ID_x__765: i32 = min_slice_count;
let __UNIQUE_ID_y__766: i32 = mode_hdisplay.wrapping_add(max_slice_width).wrapping_sub(1) / max_slice_width;
loop {
if __types_ok!(__UNIQUE_ID_x__765, __UNIQUE_ID_y__766) == 0 {
unsafe { __compiletime_assert_767() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v9 = if __UNIQUE_ID_x__765 > __UNIQUE_ID_y__766 { __UNIQUE_ID_x__765 } else { __UNIQUE_ID_y__766 };
min_slice_count = __v9;
min_slice_count
}

fn intel_dp_dsc_get_slice_config(connector: *const intel_connector, mode_clock: i32, mode_hdisplay: i32, num_joined_pipes: i32, config_ret: *mut intel_dsc_slice_config) -> bool {
let display: *mut intel_display = to_intel_display__10!(connector);
let min_slice_count: i32 = intel_dp_dsc_min_slice_count(connector, mode_clock, mode_hdisplay);
let is_edp: bool = (unsafe { (*connector).base.connector_type }) == 14;
let mut sink_slice_count_mask: u32 = unsafe { drm_dp_dsc_sink_slice_count_mask((*connector).dp.dsc_dpcd.as_ptr() as *const u8, is_edp) };
// Find the closest match to the valid slice count values
//
// Max HW DSC-per-pipe x slice-per-DSC (= slice-per-pipe) capability:
// ICL:  2x2
// BMG:  2x2, or for ultrajoined 4 pipes: 3x1
// TGL+: 2x4 (TODO: Add support for this)
//
// TODO: Explore if it's worth increasing the number of slices (from 1
// to 2 or 3), so that multiple VDSC engines can be used, thus
// reducing the minimum CDCLK requirement, which in turn is determined
// by the 1 pixel per clock VDSC engine throughput in
// intel_vdsc_min_cdclk().
let mut slices_per_pipe: i32 = 1;
while slices_per_pipe <= 4 {
let mut config: intel_dsc_slice_config = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(config) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 12usize);
}
let mut slices_per_line: i32 = 0;
if unsafe { intel_dsc_get_slice_config(display, num_joined_pipes, slices_per_pipe, core::ptr::addr_of_mut!(config)) } {
slices_per_line = unsafe { intel_dsc_line_slice_count(core::ptr::addr_of_mut!(config) as *const intel_dsc_slice_config) };
if (unsafe { drm_dp_dsc_slice_count_to_mask(slices_per_line) }) & sink_slice_count_mask != 0 {
if mode_hdisplay % slices_per_line == 0 {
if min_slice_count <= slices_per_line {
unsafe {
*config_ret = config;
}
return true;
}
}
}
}
slices_per_pipe = slices_per_pipe.wrapping_add(1);
}
// Print slice count 1,2,4,..24 if bit#0,1,3,..23 is set in the mask.
sink_slice_count_mask <<= 1i32;
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Unsupported slice count (min: %d, sink supported: %*pbl)\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name, min_slice_count, BITS_PER_TYPE__4!() as i32, core::ptr::addr_of_mut!(sink_slice_count_mask)) };
false
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_dsc_get_slice_count(connector: *const intel_connector, mode_clock: i32, mode_hdisplay: i32, num_joined_pipes: i32) -> u8 {
let mut config: intel_dsc_slice_config = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(config) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 12usize);
}
if !intel_dp_dsc_get_slice_config(connector, mode_clock, mode_hdisplay, num_joined_pipes, core::ptr::addr_of_mut!(config)) {
return 0;
}
(unsafe { intel_dsc_line_slice_count(core::ptr::addr_of_mut!(config) as *const intel_dsc_slice_config) }) as u8
}

fn source_can_output(intel_dp: *mut intel_dp, format: u32) -> bool {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
match format as u32 {
0 => {
return true;
}
2 => {
return HAS_GMCH!(display) == 0 && ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_ironlake() }) as i32) == 0;
// No YCbCr output support on gmch platforms.
// Also, ILK doesn't seem capable of DP YCbCr output.
// The displayed image is severely corrupted. SNB+ is fine.
}
1 => {
return (DISPLAY_VER!(display) as i32) >= 11; // Platform < Gen 11 cannot output YCbCr420 format
}
_ => {
let mut __v1: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"format".as_ptr() as *mut u8, (format as u32) as i64) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_source_can_output_4, sym __slate_asm_string_source_can_output_5, const 997 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, c"format".as_ptr() as *mut u8, (format as u32) as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
return false;
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_768: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn dfp_can_convert_from_rgb(intel_dp: *mut intel_dp, sink_format: u32) -> bool {
if !drm_dp_is_branch((unsafe { (*intel_dp).dpcd.as_mut_ptr() as *mut u8 }) as *const u8) {
return false;
}
if (sink_format as u32) == 2 {
return unsafe { (*intel_dp).dfp.rgb_to_ycbcr };
}
if (sink_format as u32) == 1 {
return (unsafe { (*intel_dp).dfp.rgb_to_ycbcr }) && unsafe { (*intel_dp).dfp.ycbcr_444_to_420 };
}
false
}

fn dfp_can_convert_from_ycbcr444(intel_dp: *mut intel_dp, sink_format: u32) -> bool {
if !drm_dp_is_branch((unsafe { (*intel_dp).dpcd.as_mut_ptr() as *mut u8 }) as *const u8) {
return false;
}
if (sink_format as u32) == 1 {
return unsafe { (*intel_dp).dfp.ycbcr_444_to_420 };
}
false
}

fn dfp_can_convert(intel_dp: *mut intel_dp, output_format: u32, sink_format: u32) -> bool {
match output_format as u32 {
0 => {
return dfp_can_convert_from_rgb(intel_dp, sink_format);
}
2 => {
return dfp_can_convert_from_ycbcr444(intel_dp, sink_format);
}
_ => {
let mut __v0: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"output_format".as_ptr() as *mut u8, (output_format as u32) as i64) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_dfp_can_convert_3, sym __slate_asm_string_dfp_can_convert_4, const 1043 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, c"output_format".as_ptr() as *mut u8, (output_format as u32) as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
return false;
}
}
false
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_769: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intel_dp_output_format(connector: *mut intel_connector, sink_format: u32) -> u32 {
let display: *mut intel_display = to_intel_display!(connector);
let intel_dp: *mut intel_dp = intel_attached_dp(connector);
let force_dsc_output_format: u32 = (unsafe { (*intel_dp).force_dsc_output_format }) as u32;
let mut output_format: u32 = 0;
if (force_dsc_output_format as u32) != 0 {
let __v1: bool = source_can_output(intel_dp, force_dsc_output_format) && (!drm_dp_is_branch((unsafe { (*intel_dp).dpcd.as_mut_ptr() as *mut u8 }) as *const u8) || (sink_format as u32) != (force_dsc_output_format as u32) || dfp_can_convert(intel_dp, force_dsc_output_format, sink_format));
if __v1 {
return force_dsc_output_format;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Cannot force DSC output format\n".as_ptr() as *const u8) };
}
let __v3: bool = (sink_format as u32) == 0 || dfp_can_convert_from_rgb(intel_dp, sink_format);
if __v3 {
output_format = 0;
} else {
let __v4: bool = (sink_format as u32) == 2 || dfp_can_convert_from_ycbcr444(intel_dp, sink_format);
if __v4 {
output_format = 2;
} else {
output_format = 1;
}
}
let mut __v5: i64;
let __ret_warn_on: i32 = !(!(!source_can_output(intel_dp, output_format))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!source_can_output(intel_dp, output_format))".as_ptr() as *mut u8) };
let mut __v6: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v6 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v7: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_output_format_8, sym __slate_asm_string_intel_dp_output_format_9, const 1080 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v7 = bug;
unsafe { __v6.unwrap()(__v7, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!source_can_output(intel_dp, output_format))".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
output_format
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_770: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn intel_dp_min_bpp(output_format: u32) -> i32 {
if (output_format as u32) == 0 {
return unsafe { intel_display_min_pipe_bpp() };
} else {
return 24i32;
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_output_format_link_bpp_x16(output_format: u32, mut pipe_bpp: i32) -> i32 {
// bpp value was assumed to RGB format. And YCbCr 4:2:0 output
// format of the number of bytes per pixel will be half the number
// of bytes of RGB pixel.
if (output_format as u32) == 1 {
pipe_bpp /= 2;
}
fxp_q4_from_int(pipe_bpp)
}

fn intel_dp_mode_min_link_bpp_x16(connector: *mut intel_connector, mode: *const drm_display_mode, output_format: u32) -> i32 {
intel_dp_output_format_link_bpp_x16(output_format, intel_dp_min_bpp(output_format))
}

fn intel_dp_hdisplay_bad(display: *mut intel_display, hdisplay: i32) -> bool {
// Older platforms don't like hdisplay==4096 with DP.
//
// On ILK/SNB/IVB the pipe seems to be somewhat running (scanline
// and frame counter increment), but we don't get vblank interrupts,
// and the pipe underruns immediately. The link also doesn't seem
// to get trained properly.
//
// On CHV the vblank interrupts don't seem to disappear but
// otherwise the symptoms are similar.
//
// TODO: confirm the behaviour on HSW+
hdisplay == 4096 && HAS_DDI!(display) == 0
}

fn intel_dp_max_tmds_clock(intel_dp: *mut intel_dp) -> i32 {
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let info: *const drm_display_info = (unsafe { core::ptr::addr_of_mut!((*connector).base.display_info) }) as *const drm_display_info;
let mut max_tmds_clock: i32 = unsafe { (*intel_dp).dfp.max_tmds_clock };
// Only consider the sink's max TMDS clock if we know this is a HDMI DFP
if max_tmds_clock != 0 && (unsafe { (*info).max_tmds_clock }) != 0 {
let mut __v0: i32;
let __UNIQUE_ID_x__771: i32 = max_tmds_clock;
let __UNIQUE_ID_y__772: i32 = unsafe { (*info).max_tmds_clock };
loop {
if __types_ok!(__UNIQUE_ID_x__771, __UNIQUE_ID_y__772) == 0 {
unsafe { __compiletime_assert_773() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__771 < __UNIQUE_ID_y__772 { __UNIQUE_ID_x__771 } else { __UNIQUE_ID_y__772 };
max_tmds_clock = __v0;
}
max_tmds_clock
}

fn intel_dp_tmds_clock_valid(intel_dp: *mut intel_dp, clock: i32, bpc: i32, sink_format: u32, respect_downstream_limits: bool) -> i32 {
let mut tmds_clock: i32 = 0;
let mut min_tmds_clock: i32 = 0;
let mut max_tmds_clock: i32 = 0;
if !respect_downstream_limits {
return 0;
}
tmds_clock = unsafe { intel_hdmi_tmds_clock(clock, bpc, sink_format) };
min_tmds_clock = unsafe { (*intel_dp).dfp.min_tmds_clock };
max_tmds_clock = intel_dp_max_tmds_clock(intel_dp);
if min_tmds_clock != 0 && tmds_clock < min_tmds_clock {
return 16;
}
if max_tmds_clock != 0 && tmds_clock > max_tmds_clock {
return 15;
}
0
}

fn frl_required_bw(mut clock: i32, bpc: i32, sink_format: u32) -> i32 {
if (sink_format as u32) == 1 {
clock /= 2;
}
clock.wrapping_mul(bpc).wrapping_mul(3)
}

fn intel_dp_mode_valid_downstream(connector: *mut intel_connector, mode: *const drm_display_mode, target_clock: i32, sink_format: u32) -> i32 {
let intel_dp: *mut intel_dp = intel_attached_dp(connector);
// If PCON supports FRL MODE, check FRL bandwidth constraints
if (unsafe { (*intel_dp).dfp.pcon_max_frl_bw }) != 0 {
let mut target_bw: i32 = 0;
let mut max_frl_bw: i32 = 0;
// Assume 8bpc for the FRL bandwidth check
target_bw = frl_required_bw(target_clock, 8i32, sink_format);
max_frl_bw = unsafe { (*intel_dp).dfp.pcon_max_frl_bw };
// converting bw from Gbps to Kbps
max_frl_bw = max_frl_bw.wrapping_mul(1000000);
if target_bw > max_frl_bw {
return 15;
}
return 0;
}
if (unsafe { (*intel_dp).dfp.max_dotclock }) != 0 && target_clock > unsafe { (*intel_dp).dfp.max_dotclock } {
return 15;
}
// Assume 8bpc for the DP++/HDMI/DVI TMDS clock check
intel_dp_tmds_clock_valid(intel_dp, target_clock, 8i32, sink_format, true)
}

fn intel_dp_sink_format_valid(connector: *mut intel_connector, mode: *const drm_display_mode, sink_format: u32) -> i32 {
let intel_dp: *mut intel_dp = intel_attached_dp(connector);
let info: *const drm_display_info = (unsafe { core::ptr::addr_of_mut!((*connector).base.display_info) }) as *const drm_display_info;
match sink_format as u32 {
1 => {
let __v0: bool = (unsafe { (*intel_dp).dfp.min_tmds_clock }) != 0 && !intel_dp_has_hdmi_sink(intel_dp);
if __v0 {
return 36;
}
let __v1: bool = !unsafe { (*connector).base.ycbcr_420_allowed } || !unsafe { drm_mode_is_420(info, mode) };
if __v1 {
return 36;
}
return 0;
}
0 => {
return 0;
}
_ => {
let mut __v2: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"sink_format".as_ptr() as *mut u8, (sink_format as u32) as i64) };
let mut __v3: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v3 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v4: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_sink_format_valid_5, sym __slate_asm_string_intel_dp_sink_format_valid_6, const 1238 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v4 = bug;
unsafe { __v3.unwrap()(__v4, c"sink_format".as_ptr() as *mut u8, (sink_format as u32) as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
return -2;
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_774: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn intel_dp_max_hdisplay_per_pipe(display: *mut intel_display) -> i32 {
if (DISPLAY_VER!(display) as i32) >= 30 { 6144 } else { 5120 }
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_has_dsc(connector: *const intel_connector) -> bool {
let display: *mut intel_display = to_intel_display__10!(connector);
if !HAS_DSC!(display) {
return false;
}
if !unsafe { (*connector).mst.dp }.is_null() && !HAS_DSC_MST!(display) {
return false;
}
if (unsafe { (*connector).base.connector_type }) == 14 && unsafe { (*connector).panel.vbt.edp.dsc_disable } {
return false;
}
if !drm_dp_sink_supports_dsc(unsafe { (*connector).dp.dsc_dpcd.as_ptr() as *const u8 }) {
return false;
}
true
}

fn intel_dp_can_join(intel_dp: *mut intel_dp, num_joined_pipes: i32) -> bool {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let __v1: bool = num_joined_pipes > 1 && !intel_dp_has_joiner(intel_dp);
if __v1 {
return false;
}
match num_joined_pipes {
1 => {
return true;
}
2 => {
return HAS_BIGJOINER!(display) || HAS_UNCOMPRESSED_JOINER!(display);
}
4 => {
return HAS_ULTRAJOINER!(display);
}
_ => {
return false;
}
}
unsafe { core::mem::zeroed() }
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_dotclk_valid(display: *mut intel_display, mut target_clock: i32, htotal: i32, dsc_slice_count: i32, num_joined_pipes: i32) -> bool {
let max_dotclk: i32 = (unsafe { (*display).cdclk.max_dotclk_freq }) as i32;
let mut effective_dotclk_limit: i32 = max_dotclk.wrapping_mul(num_joined_pipes);
if dsc_slice_count != 0 {
target_clock = unsafe { intel_dsc_get_pixel_rate_with_dsc_bubbles(display, target_clock, htotal, dsc_slice_count) };
} else {
effective_dotclk_limit = unsafe { intel_max_uncompressed_dotclock(display) }.wrapping_mul(num_joined_pipes);
}
target_clock <= effective_dotclk_limit
}

fn intel_dp_mode_valid_format(connector: *mut intel_connector, mode: *const drm_display_mode, target_clock: i32, sink_format: u32) -> i32 {
let display: *mut intel_display = to_intel_display!(connector);
let intel_dp: *mut intel_dp = intel_attached_dp(connector);
let mut output_format: u32 = 0;
let mut max_rate: i32 = 0;
let mut mode_rate: i32 = 0;
let mut max_lanes: i32 = 0;
let mut max_link_clock: i32 = 0;
let mut max_bw_config: intel_dp_link_config = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(max_bw_config) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut dsc_max_compressed_bpp: u16 = 0;
let mut status: i32 = intel_dp_sink_format_valid(connector, mode, sink_format);
let mut dsc: bool = false;
let mut num_joined_pipes: i32 = 1;
let mut link_bpp_x16: i32 = 0;
if (status as i32) != 0 {
return status;
}
output_format = intel_dp_output_format(connector, sink_format);
unsafe { intel_dp_link_caps_get_max_bw_config((*intel_dp).link.caps, core::ptr::addr_of_mut!(max_bw_config)) };
max_link_clock = max_bw_config.rate;
max_lanes = max_bw_config.lane_count;
max_rate = intel_dp_max_link_data_rate(intel_dp, max_link_clock, max_lanes);
link_bpp_x16 = intel_dp_mode_min_link_bpp_x16(connector, mode, output_format);
mode_rate = intel_dp_link_required(max_link_clock, max_lanes, target_clock, (unsafe { (*mode).hdisplay }) as i32, link_bpp_x16, 0u64);
// We cannot determine the required pipe‑join count before knowing whether
// DSC is needed, nor can we determine DSC need without knowing the pipe
// count.
// Because of this dependency cycle, the only correct approach is to iterate
// over candidate pipe counts and evaluate each combination.
status = 15;
while num_joined_pipes <= 4 {
if intel_dp_joiner_candidate_valid(connector, (unsafe { (*mode).hdisplay }) as i32, num_joined_pipes) {
let mut dsc_slice_count: i32 = 0;
status = unsafe { intel_pfit_mode_valid(display, mode, output_format, num_joined_pipes) };
if (status as i32) == 0 {
if intel_dp_has_dsc(connector as *const intel_connector) {
let mut pipe_bpp: i32 = 0;
dsc_slice_count = intel_dp_dsc_get_slice_count(connector as *const intel_connector, target_clock, (unsafe { (*mode).hdisplay }) as i32, num_joined_pipes) as i32;
// TBD pass the connector BPC,
// for now U8_MAX so that max BPC on that platform would be picked
pipe_bpp = intel_dp_dsc_compute_max_bpp(connector as *const intel_connector, 255u8);
// Output bpp is stored in 6.4 format so right shift by 4 to get the
// integer value since we support only integer values of bpp.
if intel_dp_is_edp(intel_dp) {
dsc_max_compressed_bpp = ((drm_edp_dsc_sink_output_bpp((unsafe { (*connector).dp.dsc_dpcd.as_mut_ptr() as *mut u8 }) as *const u8) as i32) >> 4i32) as u16;
dsc = dsc_max_compressed_bpp != 0 && dsc_slice_count != 0;
} else {
if drm_dp_sink_supports_fec(unsafe { (*connector).dp.fec_capability }) {
let mut bw_overhead_flags: u64 = 0;
if !drm_dp_is_uhbr_rate(max_link_clock) {
let __v2: u64 = bw_overhead_flags;
bw_overhead_flags = __v2 | BIT!(3i32);
}
dsc = intel_dp_mode_valid_with_dsc(connector, max_link_clock, max_lanes, target_clock, (unsafe { (*mode).hdisplay }) as i32, num_joined_pipes, output_format, pipe_bpp, bw_overhead_flags);
}
}
}
if intel_dp_joiner_needs_dsc(display, num_joined_pipes) && !dsc {
status = 15;
} else {
if mode_rate > max_rate && !dsc {
status = 15;
} else {
status = unsafe { intel_mode_valid_max_plane_size(display, mode, num_joined_pipes) };
if (status as i32) == 0 {
if !dsc {
dsc_slice_count = 0;
}
if !intel_dp_dotclk_valid(display, target_clock, (unsafe { (*mode).htotal }) as i32, dsc_slice_count, num_joined_pipes) {
status = 15;
} else {
break;
}
}
}
}
}
}
num_joined_pipes = num_joined_pipes.wrapping_add(1);
}
if (status as i32) != 0 {
return status;
}
intel_dp_mode_valid_downstream(connector, mode, target_clock, sink_format)
}

#[unsafe(link_section = ".text.slate_distinct.intel_dp.intel_dp_mode_valid")]
extern "C" fn intel_dp_mode_valid(_connector: *mut drm_connector, mode: *const drm_display_mode) -> i32 {
let display: *mut intel_display = to_intel_display__4!(unsafe { (*_connector).dev });
let connector: *mut intel_connector = to_intel_connector!(_connector);
let info: *const drm_display_info = (unsafe { core::ptr::addr_of_mut!((*connector).base.display_info) }) as *const drm_display_info;
let intel_dp: *mut intel_dp = intel_attached_dp(connector);
let mut target_clock: i32 = unsafe { (*mode).clock };
let mut status: i32 = unsafe { intel_cpu_transcoder_mode_valid(display, mode) };
if (status as i32) != 0 {
return status;
}
if (unsafe { (*mode).flags }) & 4096u32 != 0 {
return 3;
}
if (unsafe { (*mode).clock }) < 10000 {
return 16;
}
if intel_dp_hdisplay_bad(display, (unsafe { (*mode).hdisplay }) as i32) {
return 3;
}
if intel_dp_is_edp(intel_dp) {
status = unsafe { intel_panel_mode_valid(connector, mode, core::ptr::addr_of_mut!(target_clock)) };
if (status as i32) != 0 {
return status;
}
}
// TODO: Even when using a 4:2:0 sink_format intel_dp_output_format()
// will always choose a 4:4:4 output_format if the DFP can do the
// 4:4:4->4:2:0 conversion for us. Thus a mode may still be rejected
// if we only have enough DP link bandwidth for 4:2:0 but not for
// 4:4:4. Another attempt with an explicit 4:2:0 output_format might
// be needed here. intel_dp_compute_config() would need the same
// logic, or else the actual modeset would still fail.
//
// Also a lot of the checks only depend on output_format but not
// sink_format, so we are potentially doing redundant work by
// testing the same output_format for two different sink_formats.
if unsafe { drm_mode_is_420_only(info, mode) } {
status = intel_dp_mode_valid_format(connector, mode, target_clock, 1u32);
} else {
status = intel_dp_mode_valid_format(connector, mode, target_clock, 0u32);
let __v2: bool = (status as i32) != 0 && unsafe { drm_mode_is_420_also(info, mode) };
if __v2 {
status = intel_dp_mode_valid_format(connector, mode, target_clock, 1u32);
}
}
status
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_source_supports_tps3(display: *mut intel_display) -> bool {
(DISPLAY_VER!(display) as i32) >= 9 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_broadwell() }) as i32) != 0 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_haswell() }) as i32) != 0
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_source_supports_tps4(display: *mut intel_display) -> bool {
(DISPLAY_VER!(display) as i32) >= 10
}

fn seq_buf_print_array(s: *mut seq_buf, array: *const i32, nelem: i32) {
for i in 0..nelem {
unsafe { seq_buf_printf(s, c"%s%d".as_ptr() as *const u8, if i != 0 { c", ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, *array.wrapping_offset(i as isize)) };
}
}

fn intel_dp_print_rates(intel_dp: *mut intel_dp) {
let mut __slate_compound_1: [u8; 128] = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let mut s: seq_buf = {
let mut __t0: seq_buf = unsafe { core::mem::zeroed() };
__t0.buffer = unsafe { (*{
__slate_compound_1 = {
let mut __t1: [u8; 128] = core::mem::zeroed();
__t1[0] = 0u8;
__t1
};
core::ptr::addr_of_mut!(__slate_compound_1)
}).as_mut_ptr() as *mut u8 };
__t0.size = 128u64;
__t0
};
// FIXME: too big for stack?
if !drm_debug_enabled!(2i32) {
return;
}
seq_buf_print_array(core::ptr::addr_of_mut!(s), unsafe { (*intel_dp).source_rates }, unsafe { (*intel_dp).num_source_rates });
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"source rates: %s\n".as_ptr() as *const u8, seq_buf_str(&(*core::ptr::addr_of_mut!(s)))) };
seq_buf_clear(unsafe { &mut (*core::ptr::addr_of_mut!(s)) });
seq_buf_print_array(core::ptr::addr_of_mut!(s), (unsafe { (*intel_dp).sink_rates.as_mut_ptr() as *mut i32 }) as *const i32, unsafe { (*intel_dp).num_sink_rates });
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"sink rates: %s\n".as_ptr() as *const u8, seq_buf_str(&(*core::ptr::addr_of_mut!(s)))) };
unsafe { intel_dp_link_caps_print_common_rates((*intel_dp).link.caps) };
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_rate_select(intel_dp: *mut intel_dp, rate: i32) -> i32 {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let mut i: i32 = intel_dp_rate_index((unsafe { (*intel_dp).sink_rates.as_mut_ptr() as *mut i32 }) as *const i32, unsafe { (*intel_dp).num_sink_rates }, rate);
let mut __v1: i64;
let __ret_warn_on: i32 = (i < 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(i < 0)".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_rate_select_4, sym __slate_asm_string_intel_dp_rate_select_5, const 1538 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(i < 0)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if __v1 != 0 {
i = 0;
}
i
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_775: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn intel_dp_compute_rate(intel_dp: *mut intel_dp, mut port_clock: i32, link_bw: *mut u8, rate_select: *mut u8) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
// FIXME g4x can't generate an exact 2.7GHz with the 96MHz non-SSC refclk
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_g4x() }) as i32) != 0 && port_clock == 268800 {
port_clock = 270000;
}
// eDP 1.4 rate select method.
if unsafe { (*intel_dp).use_rate_select } {
unsafe {
*link_bw = 0;
}
unsafe {
*rate_select = intel_dp_rate_select(intel_dp, port_clock) as u8;
}
} else {
unsafe {
*link_bw = drm_dp_link_rate_to_bw_code(port_clock);
}
unsafe {
*rate_select = 0;
}
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_has_hdmi_sink(intel_dp: *mut intel_dp) -> bool {
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
unsafe { (*connector).base.display_info.is_hdmi }
}

fn intel_dp_source_supports_fec(intel_dp: *mut intel_dp, pipe_config: *const intel_crtc_state) -> bool {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let encoder: *mut intel_encoder = unsafe { core::ptr::addr_of_mut!((*dp_to_dig_port(intel_dp)).base) };
if (DISPLAY_VER!(display) as i32) >= 12 {
return true;
}
let __v1: bool = (DISPLAY_VER!(display) as i32) == 11 && ((unsafe { (*encoder).port }) as i32) != 0 && !intel_crtc_has_type(pipe_config, 11u32);
if __v1 {
return true;
}
false
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_supports_fec(intel_dp: *mut intel_dp, connector: *const intel_connector, pipe_config: *const intel_crtc_state) -> bool {
intel_dp_source_supports_fec(intel_dp, pipe_config) && drm_dp_sink_supports_fec(unsafe { (*connector).dp.fec_capability })
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_supports_dsc(intel_dp: *mut intel_dp, connector: *const intel_connector, crtc_state: *const intel_crtc_state) -> bool {
if !intel_dp_has_dsc(connector) {
return false;
}
let __v0: bool = intel_crtc_has_type(crtc_state, 7u32) && !intel_dp_supports_fec(intel_dp, connector, crtc_state);
if __v0 {
return false;
}
unsafe { intel_dsc_source_support(crtc_state) }
}

fn intel_dp_hdmi_compute_bpc(intel_dp: *mut intel_dp, crtc_state: *const intel_crtc_state, mut bpc: i32, respect_downstream_limits: bool) -> i32 {
let clock: i32 = unsafe { (*crtc_state).hw.adjusted_mode.crtc_clock };
// Current bpc could already be below 8bpc due to
// FDI bandwidth constraints or other limits.
// HDMI minimum is 8bpc however.
let mut __v0: i32;
let __UNIQUE_ID_x__776: i32 = bpc;
let __UNIQUE_ID_y__777: i32 = 8;
loop {
if __types_ok!(__UNIQUE_ID_x__776, __UNIQUE_ID_y__777) == 0 {
unsafe { __compiletime_assert_778() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__776 > __UNIQUE_ID_y__777 { __UNIQUE_ID_x__776 } else { __UNIQUE_ID_y__777 };
bpc = __v0;
// We will never exceed downstream TMDS clock limits while
// attempting deep color. If the user insists on forcing an
// out of spec mode they will have to be satisfied with 8bpc.
if !respect_downstream_limits {
bpc = 8;
}
while bpc >= 8 {
let __v2: bool = (unsafe { intel_hdmi_bpc_possible(crtc_state, bpc, intel_dp_has_hdmi_sink(intel_dp)) }) && (intel_dp_tmds_clock_valid(intel_dp, clock, bpc, unsafe { (*crtc_state).sink_format }, respect_downstream_limits) as i32) == 0;
if __v2 {
return bpc;
}
bpc = bpc.wrapping_sub(2);
}
22i32.wrapping_neg()
}

fn intel_dp_max_bpp(intel_dp: *mut intel_dp, crtc_state: *const intel_crtc_state, respect_downstream_limits: bool) -> i32 {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let mut bpp: i32 = 0;
let mut bpc: i32 = (unsafe { (*crtc_state).max_pipe_bpp }) / 3;
if (unsafe { (*intel_dp).dfp.max_bpc }) != 0 {
bpc = min_t__2!(i32, bpc, unsafe { (*intel_dp).dfp.max_bpc });
}
if (unsafe { (*intel_dp).dfp.min_tmds_clock }) != 0 {
let max_hdmi_bpc: i32 = intel_dp_hdmi_compute_bpc(intel_dp, crtc_state, bpc, respect_downstream_limits);
if max_hdmi_bpc < 0 {
return 0;
}
let mut __v4: i32;
let __UNIQUE_ID_x__781: i32 = bpc;
let __UNIQUE_ID_y__782: i32 = max_hdmi_bpc;
loop {
if __types_ok!(__UNIQUE_ID_x__781, __UNIQUE_ID_y__782) == 0 {
unsafe { __compiletime_assert_783() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = if __UNIQUE_ID_x__781 < __UNIQUE_ID_y__782 { __UNIQUE_ID_x__781 } else { __UNIQUE_ID_y__782 };
bpc = __v4;
}
bpp = bpc.wrapping_mul(3);
if intel_dp_is_edp(intel_dp) {
// Get bpp from vbt only for panels that dont have bpp in edid
if (unsafe { (*connector).base.display_info.bpc }) == 0 && (unsafe { (*connector).panel.vbt.edp.bpp }) != 0 && (unsafe { (*connector).panel.vbt.edp.bpp }) < bpp {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"clamping bpp for eDP panel to BIOS-provided %i\n".as_ptr() as *const u8, (*connector).panel.vbt.edp.bpp) };
bpp = unsafe { (*connector).panel.vbt.edp.bpp };
}
}
bpp
}

fn has_seamless_m_n(connector: *mut intel_connector) -> bool {
let display: *mut intel_display = to_intel_display!(connector);
// Seamless M/N reprogramming only implemented
// for BDW+ double buffered M/N registers so far.
HAS_DOUBLE_BUFFERED_M_N!(display) && ((unsafe { intel_panel_drrs_type(connector) }) as u32) == 2
}

fn intel_dp_mode_clock(crtc_state: *const intel_crtc_state, conn_state: *const drm_connector_state) -> i32 {
let connector: *mut intel_connector = to_intel_connector!(unsafe { (*conn_state).connector });
let adjusted_mode: *const drm_display_mode = unsafe { core::ptr::addr_of!((*crtc_state).hw.adjusted_mode) };
// FIXME a bit of a mess wrt clock vs. crtc_clock
if has_seamless_m_n(connector) {
return unsafe { (*intel_panel_highest_mode(connector, adjusted_mode)).clock };
} else {
return unsafe { (*adjusted_mode).crtc_clock };
}
}

/// Optimize link config in order: max bpp, min clock, min lanes
fn intel_dp_compute_link_config_wide(intel_dp: *mut intel_dp, pipe_config: *mut intel_crtc_state, conn_state: *const drm_connector_state, limits: &link_config_limits) -> i32 {
let connector: *mut intel_connector = to_intel_connector!(unsafe { (*conn_state).connector });
let mut bpp: i32 = 0;
let clock: i32 = intel_dp_mode_clock(pipe_config as *const intel_crtc_state, conn_state);
let link_caps: *mut intel_dp_link_caps = unsafe { (*intel_dp).link.caps };
let order: intel_dp_link_caps_order = unsafe { intel_dp_link_caps_connector_compute_order(connector) };
let mut err: i32 = 22i32.wrapping_neg();
let mut link_avail: i32 = 0;
bpp = fxp_q4_to_int(unsafe { (*(limits as *const link_config_limits)).link.max_bpp_x16 });
while bpp >= fxp_q4_to_int(unsafe { (*(limits as *const link_config_limits)).link.min_bpp_x16 }) {
let link_bpp_x16: i32 = intel_dp_output_format_link_bpp_x16(unsafe { (*pipe_config).output_format }, bpp);
let mut link_config: intel_dp_link_config = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(link_config) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut iter: intel_dp_link_caps_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
unsafe { intel_dp_link_caps_iter_start(core::ptr::addr_of_mut!(iter), link_caps, order, (*(limits as *const link_config_limits)).link_config_filter) };
while unsafe { unsafe { (*core::ptr::addr_of_mut!(iter)).get_next_config }.unwrap()(core::ptr::addr_of_mut!(iter), core::ptr::addr_of_mut!(link_config)) } {
let adjusted_mode: *const drm_display_mode = (unsafe { core::ptr::addr_of_mut!((*pipe_config).hw.adjusted_mode) }) as *const drm_display_mode;
let mode_rate: i32 = intel_dp_link_required(link_config.rate, link_config.lane_count, clock, (unsafe { (*adjusted_mode).hdisplay }) as i32, link_bpp_x16, 0u64);
link_avail = intel_dp_max_link_data_rate(intel_dp, link_config.rate, link_config.lane_count);
if mode_rate <= link_avail {
unsafe {
(*pipe_config).lane_count = link_config.lane_count as u8;
}
unsafe {
(*pipe_config).pipe_bpp = bpp;
}
unsafe {
(*pipe_config).port_clock = link_config.rate;
}
err = 0;
break;
}
}
unsafe { intel_dp_link_caps_iter_end(core::ptr::addr_of_mut!(iter)) };
if err == 0 {
break;
}
bpp = bpp.wrapping_sub(6i32);
}
err
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_dsc_max_src_input_bpc(display: *mut intel_display) -> i32 {
// Max DSC Input BPC for ICL is 10 and for TGL+ is 12
if (DISPLAY_VER!(display) as i32) >= 12 {
return 12;
}
if (DISPLAY_VER!(display) as i32) == 11 {
return 10;
}
intel_dp_dsc_min_src_input_bpc()
}

fn align_min_sink_dsc_input_bpp(connector: *const intel_connector, min_pipe_bpp: i32) -> i32 {
let mut dsc_bpc: [u8; 3] = [0u8; 3];
let num_bpc: i32 = unsafe { drm_dp_dsc_sink_supported_input_bpcs((*connector).dp.dsc_dpcd.as_ptr() as *const u8, dsc_bpc.as_mut_ptr() as *mut u8) };
let mut i: i32 = num_bpc.wrapping_sub(1);
while i >= 0 {
if (dsc_bpc[i as usize] as i32).wrapping_mul(3) >= min_pipe_bpp {
return (dsc_bpc[i as usize] as i32).wrapping_mul(3);
}
i = i.wrapping_sub(1);
}
0
}

fn align_max_sink_dsc_input_bpp(connector: *const intel_connector, max_pipe_bpp: i32) -> i32 {
let mut dsc_bpc: [u8; 3] = [0u8; 3];
let num_bpc: i32 = unsafe { drm_dp_dsc_sink_supported_input_bpcs((*connector).dp.dsc_dpcd.as_ptr() as *const u8, dsc_bpc.as_mut_ptr() as *mut u8) };
for i in 0..num_bpc {
if (dsc_bpc[i as usize] as i32).wrapping_mul(3) <= max_pipe_bpp {
return (dsc_bpc[i as usize] as i32).wrapping_mul(3);
}
}
0
}

fn intel_dp_source_dsc_version_minor(display: *mut intel_display) -> i32 {
if (DISPLAY_VER!(display) as i32) >= 14 { 2 } else { 1 }
}

fn intel_dp_sink_dsc_version_minor(dsc_dpcd: *const u8) -> i32 {
(((unsafe { *dsc_dpcd.wrapping_offset(1i32 as isize) }) as i32) & 240i32) >> 4i32
}

fn intel_dp_get_slice_height(vactive: i32) -> i32 {
// VDSC 1.2a spec in Section 3.8 Options for Slices implies that 108
// lines is an optimal slice height, but any size can be used as long as
// vertical active integer multiple and maximum vertical slice count
// requirements are met.
let mut slice_height: i32 = 108;
while slice_height <= vactive {
if vactive % slice_height == 0 {
return slice_height;
}
slice_height = slice_height.wrapping_add(2);
}
// Highly unlikely we reach here as most of the resolutions will end up
// finding appropriate slice_height in above loop but returning
// slice_height as 2 here as it should work with all resolutions.
2
}

fn intel_dp_dsc_compute_params(connector: *const intel_connector, crtc_state: *mut intel_crtc_state) -> i32 {
let display: *mut intel_display = to_intel_display__10!(connector);
let vdsc_cfg: *mut drm_dsc_config = unsafe { core::ptr::addr_of_mut!((*crtc_state).dsc.config) };
let mut ret: i32 = 0;
// RC_MODEL_SIZE is currently a constant across all configurations.
//
// FIXME: Look into using sink defined DPCD DP_DSC_RC_BUF_BLK_SIZE and
// DP_DSC_RC_BUF_SIZE for this.
unsafe {
(*vdsc_cfg).rc_model_size = 8192;
}
unsafe {
(*vdsc_cfg).pic_height = (*crtc_state).hw.adjusted_mode.crtc_vdisplay;
}
unsafe {
(*vdsc_cfg).slice_height = intel_dp_get_slice_height((*vdsc_cfg).pic_height as i32) as u16;
}
ret = unsafe { intel_dsc_compute_params(crtc_state) };
if ret != 0 {
return ret;
}
unsafe {
(*vdsc_cfg).dsc_version_major = (((*((*connector).dp.dsc_dpcd.as_ptr() as *const u8).wrapping_offset(1i32 as isize) as i32) & 15i32) >> 0i32) as u8;
}
let mut __v1: i32;
let __UNIQUE_ID_x__787: i32 = intel_dp_source_dsc_version_minor(display);
let __UNIQUE_ID_y__788: i32 = intel_dp_sink_dsc_version_minor(unsafe { (*connector).dp.dsc_dpcd.as_ptr() as *const u8 });
loop {
if __types_ok!(__UNIQUE_ID_x__787, __UNIQUE_ID_y__788) == 0 {
unsafe { __compiletime_assert_789() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = if __UNIQUE_ID_x__787 < __UNIQUE_ID_y__788 { __UNIQUE_ID_x__787 } else { __UNIQUE_ID_y__788 };
unsafe {
(*vdsc_cfg).dsc_version_minor = __v1 as u8;
}
if unsafe { (*vdsc_cfg).convert_rgb } {
unsafe {
(*vdsc_cfg).convert_rgb = (*((*connector).dp.dsc_dpcd.as_ptr() as *const u8).wrapping_offset(9i32 as isize) as i32) & 1i32 != 0;
}
}
let mut __v3: i32;
let __UNIQUE_ID_x__790: i32 = 13;
let __UNIQUE_ID_y__791: u8 = unsafe { drm_dp_dsc_sink_line_buf_depth((*connector).dp.dsc_dpcd.as_ptr() as *const u8) };
loop {
if __types_ok__7!(__UNIQUE_ID_x__790, __UNIQUE_ID_y__791) == 0 {
unsafe { __compiletime_assert_792() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__790 < (__UNIQUE_ID_y__791 as i32) { __UNIQUE_ID_x__790 } else { __UNIQUE_ID_y__791 as i32 };
unsafe {
(*vdsc_cfg).line_buf_depth = __v3 as u8;
}
if (unsafe { (*vdsc_cfg).line_buf_depth }) == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"DSC Sink Line Buffer Depth invalid\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg();
}
unsafe {
(*vdsc_cfg).block_pred_enable = (*((*connector).dp.dsc_dpcd.as_ptr() as *const u8).wrapping_offset(6i32 as isize) as i32) & 1i32 != 0;
}
unsafe { drm_dsc_compute_rc_parameters(vdsc_cfg) }
}

fn intel_dp_dsc_supports_format(connector: *const intel_connector, output_format: u32) -> bool {
let display: *mut intel_display = to_intel_display__10!(connector);
let mut sink_dsc_format: u8 = 0;
match output_format as u32 {
0 => {
sink_dsc_format = 1u8;
}
2 => {
sink_dsc_format = 2u8;
}
1 => {
let mut __v2: i32;
let __UNIQUE_ID_x__793: i32 = intel_dp_source_dsc_version_minor(display);
let __UNIQUE_ID_y__794: i32 = intel_dp_sink_dsc_version_minor(unsafe { (*connector).dp.dsc_dpcd.as_ptr() as *const u8 });
loop {
if __types_ok!(__UNIQUE_ID_x__793, __UNIQUE_ID_y__794) == 0 {
unsafe { __compiletime_assert_795() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__793 < __UNIQUE_ID_y__794 { __UNIQUE_ID_x__793 } else { __UNIQUE_ID_y__794 };
if __v2 < 2 {
return false;
}
sink_dsc_format = 16u8;
}
_ => {
return false;
}
}
drm_dp_dsc_sink_supports_format(unsafe { (*connector).dp.dsc_dpcd.as_ptr() as *const u8 }, sink_dsc_format)
}

fn is_bw_sufficient_for_dsc_config(intel_dp: *mut intel_dp, link_clock: i32, lane_count: i32, mode_clock: i32, mode_hdisplay: i32, dsc_slice_count: i32, link_bpp_x16: i32, bw_overhead_flags: u64) -> bool {
let available_bw: i32 = intel_dp_max_link_data_rate(intel_dp, link_clock, lane_count);
let required_bw: i32 = intel_dp_link_required(link_clock, lane_count, mode_clock, mode_hdisplay, link_bpp_x16, bw_overhead_flags);
available_bw >= required_bw
}

fn dsc_compute_link_config(intel_dp: *mut intel_dp, pipe_config: *mut intel_crtc_state, conn_state: *mut drm_connector_state, limits: *const link_config_limits, dsc_bpp_x16: i32) -> i32 {
let adjusted_mode: *const drm_display_mode = (unsafe { core::ptr::addr_of_mut!((*pipe_config).hw.adjusted_mode) }) as *const drm_display_mode;
let connector: *mut intel_connector = to_intel_connector!(unsafe { (*conn_state).connector });
let link_caps: *mut intel_dp_link_caps = unsafe { (*intel_dp).link.caps };
let order: intel_dp_link_caps_order = unsafe { intel_dp_link_caps_connector_compute_order(connector) };
let mut link_config: intel_dp_link_config = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(link_config) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut iter: intel_dp_link_caps_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
let mut err: i32 = 22i32.wrapping_neg();
unsafe { intel_dp_link_caps_iter_start(core::ptr::addr_of_mut!(iter), link_caps, order, (*limits).link_config_filter) };
'__slate_break_1: while unsafe { unsafe { (*core::ptr::addr_of_mut!(iter)).get_next_config }.unwrap()(core::ptr::addr_of_mut!(iter), core::ptr::addr_of_mut!(link_config)) } {
'__slate_continue_1: {
// FIXME: intel_dp_mtp_tu_compute_config() requires
// ->lane_count and ->port_clock set before we know
// they'll work. If we end up failing altogether,
// they'll remain in crtc state. This shouldn't matter,
// as we'd then bail out from compute config, but it's
// just ugly.
unsafe {
(*pipe_config).lane_count = link_config.lane_count as u8;
}
unsafe {
(*pipe_config).port_clock = link_config.rate;
}
if drm_dp_is_uhbr_rate(link_config.rate) {
let ret: i32 = unsafe { intel_dp_mtp_tu_compute_config(intel_dp, pipe_config, conn_state, dsc_bpp_x16, dsc_bpp_x16, 0i32, true) };
if ret != 0 {
break '__slate_continue_1;
}
} else {
let bw_overhead_flags: u64 = if unsafe { (*pipe_config).fec_enable } { BIT!(3i32) } else { 0 };
let line_slice_count: i32 = unsafe { intel_dsc_line_slice_count(core::ptr::addr_of_mut!((*pipe_config).dsc.slice_config) as *const intel_dsc_slice_config) };
if !is_bw_sufficient_for_dsc_config(intel_dp, link_config.rate, link_config.lane_count, unsafe { (*adjusted_mode).crtc_clock }, (unsafe { (*adjusted_mode).hdisplay }) as i32, line_slice_count, dsc_bpp_x16, bw_overhead_flags) {
break '__slate_continue_1;
}
}
err = 0;
break '__slate_break_1;
}
}
unsafe { intel_dp_link_caps_iter_end(core::ptr::addr_of_mut!(iter)) };
err
}

fn intel_dp_dsc_max_delta_bppx16(connector: *const intel_connector, output_format: u32) -> u16 {
let dsc_dpcd: *const u8 = unsafe { (*connector).dp.dsc_dpcd.as_ptr() as *const u8 };
let max_bpp_delta_v1: u8 = unsafe { *dsc_dpcd.wrapping_offset(14i32 as isize) };
let mut max_bpp: i32 = 0;
if ((unsafe { *dsc_dpcd.wrapping_offset(8i32 as isize) }) as i32) & 128i32 == 0 {
return 0;
}
match output_format as u32 {
0 | 2 => {
max_bpp = (max_bpp_delta_v1 as i32) & 31;
if max_bpp >= 1 && max_bpp <= 21 {
max_bpp = max_bpp.wrapping_add(16).wrapping_sub(1);
} else {
max_bpp = 0;
}
}
1 => {
max_bpp = ((max_bpp_delta_v1 as i32) & 224) >> 5i32;
if max_bpp >= 1 && max_bpp <= 7 {
max_bpp = max_bpp.wrapping_add(12).wrapping_sub(1);
}
}
_ => {
let mut __v1: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"output_format".as_ptr() as *mut u8, (output_format as u32) as i64) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_dsc_max_delta_bppx16_4, sym __slate_asm_string_intel_dp_dsc_max_delta_bppx16_5, const 2040 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, c"output_format".as_ptr() as *mut u8, (output_format as u32) as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
return 0;
}
}
(max_bpp << 4i32) as u16
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_796: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intel_dp_dsc_max_sink_compressed_bppx16(connector: *const intel_connector, output_format: u32, bpc: i32) -> u16 {
let mut max_bppx16: u16 = intel_dp_dsc_max_delta_bppx16(connector, output_format);
if max_bppx16 != 0 {
return max_bppx16;
}
max_bppx16 = drm_edp_dsc_sink_output_bpp(unsafe { (*connector).dp.dsc_dpcd.as_ptr() as *const u8 });
if max_bppx16 != 0 {
return max_bppx16;
}
// If support not given in DPCD 67h, 68h, 6Eh, 6Fh use the Maximum Allowed bit rate
// values as given in spec Table 2-157 DP v2.0
match output_format as u32 {
0 | 2 => {
return (3i32.wrapping_mul(bpc) << 4i32) as u16;
}
1 => {
return (3i32.wrapping_mul(bpc / 2) << 4i32) as u16;
}
_ => {
let mut __v0: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"output_format".as_ptr() as *mut u8, (output_format as u32) as i64) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_dsc_max_sink_compressed_bppx16_3, sym __slate_asm_string_intel_dp_dsc_max_sink_compressed_bppx16_4, const 2072 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, c"output_format".as_ptr() as *mut u8, (output_format as u32) as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_797: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intel_dp_dsc_sink_min_compressed_bpp(output_format: u32) -> i32 {
// From Mandatory bit rate range Support Table 2-157 (DP v2.0)
match output_format as u32 {
0 | 2 => {
return 8;
}
1 => {
return 6;
}
_ => {
let mut __v0: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"output_format".as_ptr() as *mut u8, (output_format as u32) as i64) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_dsc_sink_min_compressed_bpp_3, sym __slate_asm_string_intel_dp_dsc_sink_min_compressed_bpp_4, const 2089 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, c"output_format".as_ptr() as *mut u8, (output_format as u32) as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_798: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intel_dp_dsc_sink_max_compressed_bpp(connector: *const intel_connector, output_format: u32, bpc: i32) -> i32 {
(intel_dp_dsc_max_sink_compressed_bppx16(connector, output_format, bpc) as i32) >> 4i32
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_dsc_min_src_compressed_bpp() -> i32 {
// Min Compressed bpp supported by source is 8
8
}

fn dsc_src_max_compressed_bpp(intel_dp: *mut intel_dp) -> i32 {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
// Forcing DSC and using the platform's max compressed bpp is seen to cause
// underruns. Since DSC isn't needed in these cases, limit the
// max compressed bpp to 18, which is a safe value across platforms with different
// pipe bpps.
if unsafe { (*intel_dp).force_dsc_en } {
return 18;
}
// Max Compressed bpp for Gen 13+ is 27bpp.
// For earlier platform is 23bpp. (Bspec:49259).
if (DISPLAY_VER!(display) as i32) < 13 {
return 23;
} else {
return 27;
}
}

/// Note: for pre-13 display you still need to check the validity of each step.
#[unsafe(no_mangle)]
extern "C" fn intel_dp_dsc_bpp_step_x16(connector: *const intel_connector) -> i32 {
let display: *mut intel_display = to_intel_display__10!(connector);
let incr: u8 = unsafe { drm_dp_dsc_sink_bpp_incr((*connector).dp.dsc_dpcd.as_ptr() as *const u8) };
if (DISPLAY_VER!(display) as i32) < 14 || incr == 0 {
return fxp_q4_from_int(1i32);
}
if !unsafe { (*connector).mst.dp }.is_null() && (unsafe { (*connector).link.force_bpp_x16 }) == 0 && !unsafe { (*(*connector).mst.dp).force_dsc_fractional_bpp_en } {
return fxp_q4_from_int(1i32);
}
// fxp q4
fxp_q4_from_int(1i32) / (incr as i32)
}

/// Note: for bpp_x16 to be valid it must be also within the source/sink's
/// min..max bpp capability range.
#[unsafe(no_mangle)]
extern "C" fn intel_dp_dsc_valid_compressed_bpp(intel_dp: *mut intel_dp, bpp_x16: i32) -> bool {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
if (DISPLAY_VER!(display) as i32) >= 13 {
let __v1: bool = (unsafe { (*intel_dp).force_dsc_fractional_bpp_en }) && fxp_q4_to_frac(bpp_x16) == 0;
if __v1 {
return false;
}
return true;
}
if fxp_q4_to_frac(bpp_x16) != 0 {
return false;
}
align_max_vesa_compressed_bpp_x16(bpp_x16) == bpp_x16
}

fn align_min_compressed_bpp_x16(connector: *const intel_connector, min_bpp_x16: i32) -> i32 {
let display: *mut intel_display = to_intel_display__10!(connector);
if (DISPLAY_VER!(display) as i32) >= 13 {
let bpp_step_x16: i32 = intel_dp_dsc_bpp_step_x16(connector);
let mut __v1: i64;
let __ret_warn_on: i32 = !(!(!is_power_of_2(bpp_step_x16 as u64))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!is_power_of_2(bpp_step_x16))".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_align_min_compressed_bpp_x16_4, sym __slate_asm_string_align_min_compressed_bpp_x16_5, const 2180 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!is_power_of_2(bpp_step_x16))".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
return round_up!(min_bpp_x16, bpp_step_x16);
} else {
return align_min_vesa_compressed_bpp_x16(min_bpp_x16);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_799: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn align_max_compressed_bpp_x16(connector: *const intel_connector, output_format: u32, pipe_bpp: i32, mut max_bpp_x16: i32) -> i32 {
let display: *mut intel_display = to_intel_display__10!(connector);
let link_bpp_x16: i32 = intel_dp_output_format_link_bpp_x16(output_format, pipe_bpp);
let bpp_step_x16: i32 = intel_dp_dsc_bpp_step_x16(connector);
let mut __v1: i32;
let __UNIQUE_ID_x__800: i32 = max_bpp_x16;
let __UNIQUE_ID_y__801: i32 = link_bpp_x16.wrapping_sub(bpp_step_x16);
loop {
if __types_ok!(__UNIQUE_ID_x__800, __UNIQUE_ID_y__801) == 0 {
unsafe { __compiletime_assert_802() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = if __UNIQUE_ID_x__800 < __UNIQUE_ID_y__801 { __UNIQUE_ID_x__800 } else { __UNIQUE_ID_y__801 };
max_bpp_x16 = __v1;
if (DISPLAY_VER!(display) as i32) >= 13 {
let mut __v3: i64;
let __ret_warn_on: i32 = !(!(!is_power_of_2(bpp_step_x16 as u64))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!is_power_of_2(bpp_step_x16))".as_ptr() as *mut u8) };
let mut __v4: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v4 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v5: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_align_max_compressed_bpp_x16_6, sym __slate_asm_string_align_max_compressed_bpp_x16_7, const 2199 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v5 = bug;
unsafe { __v4.unwrap()(__v5, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!is_power_of_2(bpp_step_x16))".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
return round_down!(max_bpp_x16, bpp_step_x16);
} else {
return align_max_vesa_compressed_bpp_x16(max_bpp_x16);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_803: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

/// Find the max compressed BPP we can find a link configuration for. The BPPs to
/// try depend on the source (platform) and sink.
fn dsc_compute_compressed_bpp(intel_dp: *mut intel_dp, pipe_config: *mut intel_crtc_state, conn_state: *mut drm_connector_state, limits: *const link_config_limits, pipe_bpp: i32) -> i32 {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let connector: *mut intel_connector = to_intel_connector!(unsafe { (*conn_state).connector });
let min_bpp_x16: i32 = unsafe { (*limits).link.min_bpp_x16 };
let mut max_bpp_x16: i32 = unsafe { (*limits).link.max_bpp_x16 };
let bpp_step_x16: i32 = intel_dp_dsc_bpp_step_x16(connector as *const intel_connector);
let mut bpp_x16: i32 = 0;
let mut ret: i32 = 0;
max_bpp_x16 = align_max_compressed_bpp_x16(connector as *const intel_connector, unsafe { (*pipe_config).output_format }, pipe_bpp, max_bpp_x16);
if intel_dp_is_edp(intel_dp) {
let mut max_link_config: intel_dp_link_config = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(max_link_config) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
// FIXME: Clarify why eDP does not use the regular SST BW
// check and instead always uses the maximum link config,
// regardless of intel_dp::use_max_params. Then unify this eDP
// path with the regular DP path.
if !intel_dp_get_connector_max_link_config(connector, limits, core::ptr::addr_of_mut!(max_link_config)) {
return 22i32.wrapping_neg();
}
unsafe {
(*pipe_config).port_clock = max_link_config.rate;
}
unsafe {
(*pipe_config).lane_count = max_link_config.lane_count as u8;
}
unsafe {
(*pipe_config).dsc.compressed_bpp_x16 = max_bpp_x16 as u16;
}
return 0;
}
bpp_x16 = max_bpp_x16;
while bpp_x16 >= min_bpp_x16 {
if intel_dp_dsc_valid_compressed_bpp(intel_dp, bpp_x16) {
ret = dsc_compute_link_config(intel_dp, pipe_config, conn_state, limits, bpp_x16);
if ret == 0 {
unsafe {
(*pipe_config).dsc.compressed_bpp_x16 = bpp_x16 as u16;
}
let __v3: bool = (unsafe { (*intel_dp).force_dsc_fractional_bpp_en }) && fxp_q4_to_frac(bpp_x16) != 0;
if __v3 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Forcing DSC fractional bpp\n".as_ptr() as *const u8) };
}
return 0;
}
}
bpp_x16 = bpp_x16.wrapping_sub(bpp_step_x16);
}
22i32.wrapping_neg()
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_dsc_min_src_input_bpc() -> i32 {
// Min DSC Input BPC for ICL+ is 8
8
}

fn is_dsc_pipe_bpp_sufficient(limits: *const link_config_limits, pipe_bpp: i32) -> bool {
pipe_bpp >= unsafe { (*limits).pipe.min_bpp } && pipe_bpp <= unsafe { (*limits).pipe.max_bpp }
}

fn intel_dp_force_dsc_pipe_bpp(intel_dp: *mut intel_dp, limits: *const link_config_limits) -> i32 {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let mut forced_bpp: i32 = 0;
if (unsafe { (*intel_dp).force_dsc_bpc }) == 0 {
return 0;
}
forced_bpp = unsafe { (*intel_dp).force_dsc_bpc }.wrapping_mul(3);
if is_dsc_pipe_bpp_sufficient(limits, forced_bpp) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Input DSC BPC forced to %d\n".as_ptr() as *const u8, (*intel_dp).force_dsc_bpc) };
return forced_bpp;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Cannot force DSC BPC:%d, due to DSC BPC limits\n".as_ptr() as *const u8, (*intel_dp).force_dsc_bpc) };
0
}

fn intel_dp_dsc_compute_pipe_bpp(intel_dp: *mut intel_dp, pipe_config: *mut intel_crtc_state, conn_state: *mut drm_connector_state, limits: *const link_config_limits) -> i32 {
let forced_bpp: i32 = intel_dp_force_dsc_pipe_bpp(intel_dp, limits);
let mut pipe_bpp: i32 = 0;
let mut ret: i32 = 0;
if forced_bpp != 0 {
pipe_bpp = forced_bpp;
} else {
pipe_bpp = unsafe { (*limits).pipe.max_bpp };
}
ret = dsc_compute_compressed_bpp(intel_dp, pipe_config, conn_state, limits, pipe_bpp);
if ret != 0 {
return 22i32.wrapping_neg();
}
unsafe {
(*pipe_config).pipe_bpp = pipe_bpp;
}
0
}

/// Return whether FEC must be enabled for 8b10b SST or MST links. On 128b132b
/// links FEC is always enabled implicitly by the HW, so this function returns
/// false for that case.
#[unsafe(no_mangle)]
extern "C" fn intel_dp_needs_8b10b_fec(crtc_state: *const intel_crtc_state, dsc_enabled_on_crtc: bool) -> bool {
if intel_dp_is_uhbr(crtc_state) {
return false;
}
// Though eDP v1.5 supports FEC with DSC, unlike DP, it is optional.
// Since, FEC is a bandwidth overhead, continue to not enable it for
// eDP. Until, there is a good reason to do so.
if intel_crtc_has_type(crtc_state, 8u32) {
return false;
}
dsc_enabled_on_crtc || unsafe { intel_dsc_enabled_on_link(crtc_state) }
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_dsc_reset_config(crtc_state: *mut intel_crtc_state) {
unsafe {
(*crtc_state).fec_enable = false;
}
unsafe {
(*crtc_state).dsc.compression_enable = false;
}
unsafe {
(*crtc_state).dsc.compressed_bpp_x16 = 0;
}
unsafe { memset(core::ptr::addr_of_mut!((*crtc_state).dsc.slice_config) as *mut (), 0i32, core::mem::size_of::<intel_dsc_slice_config>() as u64) };
unsafe { memset(core::ptr::addr_of_mut!((*crtc_state).dsc.config) as *mut (), 0i32, core::mem::size_of::<drm_dsc_config>() as u64) };
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_dsc_compute_config(intel_dp: *mut intel_dp, pipe_config: *mut intel_crtc_state, conn_state: *mut drm_connector_state, limits: *const link_config_limits, timeslots: i32) -> i32 {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let connector: *const intel_connector = to_intel_connector!(unsafe { (*conn_state).connector }) as *const intel_connector;
let adjusted_mode: *const drm_display_mode = (unsafe { core::ptr::addr_of_mut!((*pipe_config).hw.adjusted_mode) }) as *const drm_display_mode;
let num_joined_pipes: i32 = unsafe { intel_crtc_num_joined_pipes(pipe_config as *const intel_crtc_state) };
let is_mst: bool = intel_crtc_has_type(pipe_config as *const intel_crtc_state, 11u32);
let mut ret: i32 = 0;
// FIXME: set the FEC enabled state once pipe_config->port_clock is
// already known, so the UHBR/non-UHBR mode can be determined.
unsafe {
(*pipe_config).fec_enable = intel_dp_needs_8b10b_fec(pipe_config as *const intel_crtc_state, true);
}
if !intel_dp_dsc_supports_format(connector, unsafe { (*pipe_config).output_format }) {
return 22i32.wrapping_neg();
}
// Link parameters, pipe bpp and compressed bpp have already been
// figured out for DP MST DSC.
if !is_mst {
ret = intel_dp_dsc_compute_pipe_bpp(intel_dp, pipe_config, conn_state, limits);
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"No Valid pipe bpp for given mode ret = %d\n".as_ptr() as *const u8, ret) };
return ret;
}
}
if !intel_dp_dsc_get_slice_config(connector, unsafe { (*adjusted_mode).crtc_clock }, (unsafe { (*adjusted_mode).crtc_hdisplay }) as i32, num_joined_pipes, unsafe { core::ptr::addr_of_mut!((*pipe_config).dsc.slice_config) }) {
return 22i32.wrapping_neg();
}
ret = intel_dp_dsc_compute_params(connector, pipe_config);
if ret < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Cannot compute valid DSC parameters for Input Bpp = %dCompressed BPP = %d.%04d\n".as_ptr() as *const u8, (*pipe_config).pipe_bpp, fxp_q4_to_int((*pipe_config).dsc.compressed_bpp_x16 as i32), fxp_q4_to_frac((*pipe_config).dsc.compressed_bpp_x16 as i32).wrapping_mul(625)) };
return ret;
}
unsafe { intel_dsc_enable_on_crtc(pipe_config) };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"DP DSC computed with Input Bpp = %d Compressed Bpp = %d.%04d Slice Count = %d\n".as_ptr() as *const u8, (*pipe_config).pipe_bpp, fxp_q4_to_int((*pipe_config).dsc.compressed_bpp_x16 as i32), fxp_q4_to_frac((*pipe_config).dsc.compressed_bpp_x16 as i32).wrapping_mul(625), intel_dsc_line_slice_count(core::ptr::addr_of_mut!((*pipe_config).dsc.slice_config) as *const intel_dsc_slice_config)) };
0
}

fn dsc_throughput_quirk_max_bpp_x16(connector: *const intel_connector, mode_clock: i32) -> i32 {
if ((unsafe { (*connector).dp.__slate_bits_0.__get_dsc_throughput_quirk() }) as i32) == 0 {
return 2147483647i32;
}
// Synaptics Panamera branch devices have a problem decompressing a
// stream with a compressed link-bpp higher than 12, if the pixel
// clock is higher than ~50 % of the maximum overall throughput
// reported by the branch device. Work around this by limiting the
// maximum link bpp for such pixel clocks.
//
// TODO: Use the throughput value specific to the actual RGB/YUV
// format of the output, after determining the pixel clock limit for
// YUV modes. For now use the smaller of the throughput values, which
// may result in limiting the link-bpp value already at a lower than
// required mode clock in case of native YUV422/420 output formats.
// The RGB/YUV444 throughput value should be always either equal or
// smaller than the YUV422/420 value, but let's not depend on this
// assumption.
let mut __v0: i32;
let __UNIQUE_ID_x__804: i32 = unsafe { (*connector).dp.dsc_branch_caps.overall_throughput.rgb_yuv444 };
let __UNIQUE_ID_y__805: i32 = unsafe { (*connector).dp.dsc_branch_caps.overall_throughput.yuv422_420 };
loop {
if __types_ok!(__UNIQUE_ID_x__804, __UNIQUE_ID_y__805) == 0 {
unsafe { __compiletime_assert_806() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__804 < __UNIQUE_ID_y__805 { __UNIQUE_ID_x__804 } else { __UNIQUE_ID_y__805 };
if mode_clock < __v0 / 2 {
return 2147483647i32;
}
fxp_q4_from_int(12i32)
}

fn compute_max_compressed_bpp_x16(connector: *mut intel_connector, mode_clock: i32, mode_hdisplay: i32, num_joined_pipes: i32, output_format: u32, pipe_max_bpp: i32, mut max_link_bpp_x16: i32) -> i32 {
let display: *mut intel_display = to_intel_display!(connector);
let intel_dp: *mut intel_dp = intel_attached_dp(connector);
let dsc_src_max_bpp: i32 = dsc_src_max_compressed_bpp(intel_dp);
let mut dsc_sink_max_bpp: i32 = 0;
let mut dsc_max_bpp: i32 = 0;
let mut throughput_max_bpp_x16: i32 = 0;
let joiner_max_bpp: i32 = get_max_compressed_bpp_with_joiner(display, mode_clock as u32, mode_hdisplay as u32, num_joined_pipes) as i32;
dsc_sink_max_bpp = intel_dp_dsc_sink_max_compressed_bpp(connector as *const intel_connector, output_format, pipe_max_bpp / 3);
let mut __v1: i32;
let __UNIQUE_ID_x__810: i32 = dsc_sink_max_bpp;
let __UNIQUE_ID_y__811: i32 = dsc_src_max_bpp;
loop {
if __types_ok!(__UNIQUE_ID_x__810, __UNIQUE_ID_y__811) == 0 {
unsafe { __compiletime_assert_812() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = if __UNIQUE_ID_x__810 < __UNIQUE_ID_y__811 { __UNIQUE_ID_x__810 } else { __UNIQUE_ID_y__811 };
dsc_max_bpp = __v1;
let mut __v3: i32;
let __UNIQUE_ID_x__813: i32 = dsc_max_bpp;
let __UNIQUE_ID_y__814: i32 = joiner_max_bpp;
loop {
if __types_ok!(__UNIQUE_ID_x__813, __UNIQUE_ID_y__814) == 0 {
unsafe { __compiletime_assert_815() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__813 < __UNIQUE_ID_y__814 { __UNIQUE_ID_x__813 } else { __UNIQUE_ID_y__814 };
dsc_max_bpp = __v3;
let mut __v5: i32;
let __UNIQUE_ID_x__816: i32 = max_link_bpp_x16;
let __UNIQUE_ID_y__817: i32 = fxp_q4_from_int(dsc_max_bpp);
loop {
if __types_ok!(__UNIQUE_ID_x__816, __UNIQUE_ID_y__817) == 0 {
unsafe { __compiletime_assert_818() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = if __UNIQUE_ID_x__816 < __UNIQUE_ID_y__817 { __UNIQUE_ID_x__816 } else { __UNIQUE_ID_y__817 };
max_link_bpp_x16 = __v5;
throughput_max_bpp_x16 = dsc_throughput_quirk_max_bpp_x16(connector as *const intel_connector, mode_clock);
if throughput_max_bpp_x16 < max_link_bpp_x16 {
max_link_bpp_x16 = throughput_max_bpp_x16;
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Decreasing link max bpp to %d.%04d due to DSC throughput quirk\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name, fxp_q4_to_int(max_link_bpp_x16), fxp_q4_to_frac(max_link_bpp_x16).wrapping_mul(625)) };
}
max_link_bpp_x16 = align_max_compressed_bpp_x16(connector as *const intel_connector, output_format, pipe_max_bpp, max_link_bpp_x16);
max_link_bpp_x16
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_mode_valid_with_dsc(connector: *mut intel_connector, link_clock: i32, lane_count: i32, mode_clock: i32, mode_hdisplay: i32, num_joined_pipes: i32, output_format: u32, pipe_bpp: i32, bw_overhead_flags: u64) -> bool {
let intel_dp: *mut intel_dp = intel_attached_dp(connector);
let min_bpp_x16: i32 = intel_dp_compute_min_compressed_bpp_x16(connector, output_format);
let max_bpp_x16: i32 = compute_max_compressed_bpp_x16(connector, mode_clock, mode_hdisplay, num_joined_pipes, output_format, pipe_bpp, 2147483647i32);
let dsc_slice_count: i32 = intel_dp_dsc_get_slice_count(connector as *const intel_connector, mode_clock, mode_hdisplay, num_joined_pipes) as i32;
if min_bpp_x16 <= 0 || min_bpp_x16 > max_bpp_x16 {
return false;
}
if dsc_slice_count == 0 {
return false;
}
is_bw_sufficient_for_dsc_config(intel_dp, link_clock, lane_count, mode_clock, mode_hdisplay, dsc_slice_count, min_bpp_x16, bw_overhead_flags)
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_get_connector_max_link_config(connector: *mut intel_connector, limits: *const link_config_limits, max_link_config: *mut intel_dp_link_config) -> bool {
let intel_dp: *mut intel_dp = intel_attached_dp(connector);
let link_caps: *mut intel_dp_link_caps = unsafe { (*intel_dp).link.caps };
let order: intel_dp_link_caps_order = unsafe { intel_dp_link_caps_connector_compute_order(connector) };
unsafe { intel_dp_link_caps_get_max_config(link_caps, order.key, (*limits).link_config_filter, max_link_config) }
}

// Calculate the output link min, max bpp values in limits based on the pipe bpp
// range, crtc_state and dsc mode. Return true on success.
fn intel_dp_compute_config_link_bpp_limits(connector: *mut intel_connector, crtc_state: *const intel_crtc_state, dsc: bool, limits: *mut link_config_limits) -> bool {
let display: *mut intel_display = to_intel_display!(connector);
let intel_dp: *mut intel_dp = intel_attached_dp(connector);
let adjusted_mode: *const drm_display_mode = unsafe { core::ptr::addr_of!((*crtc_state).hw.adjusted_mode) };
let crtc: *const intel_crtc = to_intel_crtc!(unsafe { (*crtc_state).uapi.crtc }) as *const intel_crtc;
let encoder: *const intel_encoder = (unsafe { core::ptr::addr_of_mut!((*dp_to_dig_port(intel_dp)).base) }) as *const intel_encoder;
let mut max_link_config: intel_dp_link_config = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(max_link_config) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut max_link_bpp_x16: i32 = 0;
let mut __v2: i32;
let __UNIQUE_ID_x__819: i32 = unsafe { (*crtc_state).max_link_bpp_x16 };
let __UNIQUE_ID_y__820: i32 = fxp_q4_from_int(unsafe { (*limits).pipe.max_bpp });
loop {
if __types_ok!(__UNIQUE_ID_x__819, __UNIQUE_ID_y__820) == 0 {
unsafe { __compiletime_assert_821() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__819 < __UNIQUE_ID_y__820 { __UNIQUE_ID_x__819 } else { __UNIQUE_ID_y__820 };
max_link_bpp_x16 = __v2;
if !dsc {
max_link_bpp_x16 = rounddown__2!(max_link_bpp_x16, fxp_q4_from_int(6i32));
if max_link_bpp_x16 < fxp_q4_from_int(unsafe { (*limits).pipe.min_bpp }) {
return false;
}
unsafe {
(*limits).link.min_bpp_x16 = fxp_q4_from_int((*limits).pipe.min_bpp);
}
} else {
unsafe {
(*limits).link.min_bpp_x16 = intel_dp_compute_min_compressed_bpp_x16(connector, (*crtc_state).output_format);
}
max_link_bpp_x16 = compute_max_compressed_bpp_x16(connector, unsafe { (*adjusted_mode).crtc_clock }, (unsafe { (*adjusted_mode).hdisplay }) as i32, unsafe { intel_crtc_num_joined_pipes(crtc_state) }, unsafe { (*crtc_state).output_format }, unsafe { (*limits).pipe.max_bpp }, max_link_bpp_x16);
}
unsafe {
(*limits).link.max_bpp_x16 = max_link_bpp_x16;
}
if !intel_dp_get_connector_max_link_config(connector, limits as *const link_config_limits, core::ptr::addr_of_mut!(max_link_config)) {
return false;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[ENCODER:%d:%s][CRTC:%d:%s] DP link limits: pixel clock %d kHz DSC %s max link %dx%d max pipe_bpp %d min link_bpp %d.%04d max link_bpp %d.%04d\n".as_ptr() as *const u8, (*encoder).base.base.id, (*encoder).base.name, (*crtc).base.base.id, (*crtc).base.name, (*adjusted_mode).crtc_clock, str_on_off(dsc), max_link_config.lane_count, max_link_config.rate, (*limits).pipe.max_bpp, fxp_q4_to_int((*limits).link.min_bpp_x16), fxp_q4_to_frac((*limits).link.min_bpp_x16).wrapping_mul(625), fxp_q4_to_int((*limits).link.max_bpp_x16), fxp_q4_to_frac((*limits).link.max_bpp_x16).wrapping_mul(625)) };
if (unsafe { (*limits).link.min_bpp_x16 }) <= 0 || (unsafe { (*limits).link.min_bpp_x16 }) > unsafe { (*limits).link.max_bpp_x16 } {
return false;
}
true
}

fn intel_dp_dsc_compute_pipe_bpp_limits(connector: *mut intel_connector, limits: *mut link_config_limits) -> bool {
let display: *mut intel_display = to_intel_display!(connector);
let orig_limits: link_config_limits = unsafe { *limits };
let dsc_min_bpc: i32 = intel_dp_dsc_min_src_input_bpc();
let dsc_max_bpc: i32 = intel_dp_dsc_max_src_input_bpc(display);
let mut __v1: i32;
let __UNIQUE_ID_x__822: i32 = unsafe { (*limits).pipe.min_bpp };
let __UNIQUE_ID_y__823: i32 = dsc_min_bpc.wrapping_mul(3);
loop {
if __types_ok!(__UNIQUE_ID_x__822, __UNIQUE_ID_y__823) == 0 {
unsafe { __compiletime_assert_824() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = if __UNIQUE_ID_x__822 > __UNIQUE_ID_y__823 { __UNIQUE_ID_x__822 } else { __UNIQUE_ID_y__823 };
unsafe {
(*limits).pipe.min_bpp = __v1;
}
unsafe {
(*limits).pipe.min_bpp = align_min_sink_dsc_input_bpp(connector as *const intel_connector, (*limits).pipe.min_bpp);
}
let mut __v3: i32;
let __UNIQUE_ID_x__825: i32 = unsafe { (*limits).pipe.max_bpp };
let __UNIQUE_ID_y__826: i32 = dsc_max_bpc.wrapping_mul(3);
loop {
if __types_ok!(__UNIQUE_ID_x__825, __UNIQUE_ID_y__826) == 0 {
unsafe { __compiletime_assert_827() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__825 < __UNIQUE_ID_y__826 { __UNIQUE_ID_x__825 } else { __UNIQUE_ID_y__826 };
unsafe {
(*limits).pipe.max_bpp = __v3;
}
unsafe {
(*limits).pipe.max_bpp = align_max_sink_dsc_input_bpp(connector as *const intel_connector, (*limits).pipe.max_bpp);
}
if (unsafe { (*limits).pipe.min_bpp }) <= 0 || (unsafe { (*limits).pipe.min_bpp }) > unsafe { (*limits).pipe.max_bpp } {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Invalid DSC src/sink input BPP (src:%d-%d pipe:%d-%d sink-align:%d-%d)\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name, dsc_min_bpc.wrapping_mul(3), dsc_max_bpc.wrapping_mul(3), orig_limits.pipe.min_bpp, orig_limits.pipe.max_bpp, (*limits).pipe.min_bpp, (*limits).pipe.max_bpp) };
return false;
}
true
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_config_required_rate(crtc_state: *const intel_crtc_state) -> i32 {
let adjusted_mode: *const drm_display_mode = unsafe { core::ptr::addr_of!((*crtc_state).hw.adjusted_mode) };
let mut link_bpp_x16: i32 = 0;
let __v0: i32 = if unsafe { (*crtc_state).dsc.compression_enable } {
(unsafe { (*crtc_state).dsc.compressed_bpp_x16 }) as i32
} else {
fxp_q4_from_int(unsafe { (*crtc_state).pipe_bpp })
};
link_bpp_x16 = __v0;
intel_dp_link_required(unsafe { (*crtc_state).port_clock }, (unsafe { (*crtc_state).lane_count }) as i32, unsafe { (*adjusted_mode).crtc_clock }, (unsafe { (*adjusted_mode).hdisplay }) as i32, link_bpp_x16, 0u64)
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_joiner_needs_dsc(display: *mut intel_display, num_joined_pipes: i32) -> bool {
// Pipe joiner needs compression up to display 12 due to bandwidth
// limitation. DG2 onwards pipe joiner can be enabled without
// compression.
// Ultrajoiner always needs compression.
!HAS_UNCOMPRESSED_JOINER!(display) && num_joined_pipes == 2 || num_joined_pipes == 4
}

fn intel_dp_compute_link_for_joined_pipes(encoder: *mut intel_encoder, pipe_config: *mut intel_crtc_state, conn_state: *mut drm_connector_state, respect_downstream_limits: bool) -> i32 {
let display: *mut intel_display = to_intel_display!(encoder);
let num_joined_pipes: i32 = unsafe { intel_crtc_num_joined_pipes(pipe_config as *const intel_crtc_state) };
let connector: *mut intel_connector = to_intel_connector!(unsafe { (*conn_state).connector });
let adjusted_mode: *const drm_display_mode = (unsafe { core::ptr::addr_of_mut!((*pipe_config).hw.adjusted_mode) }) as *const drm_display_mode;
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
let mut limits: link_config_limits = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(limits) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 20usize);
}
let mut dsc_needed: bool = false;
let mut joiner_needs_dsc: bool = false;
let mut ret: i32 = 0;
intel_dp_dsc_reset_config(pipe_config);
joiner_needs_dsc = intel_dp_joiner_needs_dsc(display, num_joined_pipes);
dsc_needed = joiner_needs_dsc || unsafe { (*intel_dp).force_dsc_en } || !intel_dp_compute_config_limits(intel_dp, conn_state, pipe_config, respect_downstream_limits, false, core::ptr::addr_of_mut!(limits));
if !dsc_needed {
// Optimize for slow and wide for everything, because there are some
// eDP 1.3 and 1.4 panels don't work well with fast and narrow.
ret = intel_dp_compute_link_config_wide(intel_dp, pipe_config, conn_state as *const drm_connector_state, unsafe { &(*(core::ptr::addr_of_mut!(limits) as *const link_config_limits)) });
let __v3: bool = ret == 0 && intel_dp_is_uhbr(pipe_config as *const intel_crtc_state);
if __v3 {
ret = unsafe { intel_dp_mtp_tu_compute_config(intel_dp, pipe_config, conn_state, fxp_q4_from_int((*pipe_config).pipe_bpp), fxp_q4_from_int((*pipe_config).pipe_bpp), 0i32, false) };
}
let __v4: bool = ret != 0 || !intel_dp_dotclk_valid(display, unsafe { (*adjusted_mode).crtc_clock }, (unsafe { (*adjusted_mode).crtc_htotal }) as i32, 0i32, num_joined_pipes);
if __v4 {
dsc_needed = true;
}
}
let __v5: bool = dsc_needed && !intel_dp_supports_dsc(intel_dp, connector as *const intel_connector, pipe_config as *const intel_crtc_state);
if __v5 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"DSC required but not available\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg();
}
if dsc_needed {
let mut dsc_slice_count: i32 = 0;
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Try DSC (fallback=%s, joiner=%s, force=%s)\n".as_ptr() as *const u8, str_yes_no(ret != 0), str_yes_no(joiner_needs_dsc), str_yes_no((*intel_dp).force_dsc_en)) };
if !intel_dp_compute_config_limits(intel_dp, conn_state, pipe_config, respect_downstream_limits, true, core::ptr::addr_of_mut!(limits)) {
return 22i32.wrapping_neg();
}
ret = intel_dp_dsc_compute_config(intel_dp, pipe_config, conn_state, core::ptr::addr_of_mut!(limits) as *const link_config_limits, 64i32);
if ret < 0 {
return ret;
}
dsc_slice_count = unsafe { intel_dsc_line_slice_count(core::ptr::addr_of_mut!((*pipe_config).dsc.slice_config) as *const intel_dsc_slice_config) };
if !intel_dp_dotclk_valid(display, unsafe { (*adjusted_mode).crtc_clock }, (unsafe { (*adjusted_mode).crtc_htotal }) as i32, dsc_slice_count, num_joined_pipes) {
return 22i32.wrapping_neg();
}
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"DP lane count %d clock %d bpp input %d compressed %d.%04d HDR %s link rate required %d available %d\n".as_ptr() as *const u8, (*pipe_config).lane_count as i32, (*pipe_config).port_clock, (*pipe_config).pipe_bpp, fxp_q4_to_int((*pipe_config).dsc.compressed_bpp_x16 as i32), fxp_q4_to_frac((*pipe_config).dsc.compressed_bpp_x16 as i32).wrapping_mul(625), str_yes_no(intel_dp_in_hdr_mode(conn_state as *const drm_connector_state)), intel_dp_config_required_rate(pipe_config as *const intel_crtc_state), intel_dp_max_link_data_rate(intel_dp, (*pipe_config).port_clock, (*pipe_config).lane_count as i32)) };
0
}

fn intel_dp_compute_link_config(encoder: *mut intel_encoder, crtc_state: *mut intel_crtc_state, conn_state: *mut drm_connector_state, respect_downstream_limits: bool) -> i32 {
let crtc: *mut intel_crtc = to_intel_crtc!(unsafe { (*crtc_state).uapi.crtc });
let connector: *mut intel_connector = to_intel_connector!(unsafe { (*conn_state).connector });
let adjusted_mode: *const drm_display_mode = (unsafe { core::ptr::addr_of_mut!((*crtc_state).hw.adjusted_mode) }) as *const drm_display_mode;
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
let mut num_joined_pipes: i32 = 1;
let mut ret: i32 = 22i32.wrapping_neg();
let __v2: bool = (unsafe { (*crtc_state).fec_enable }) && !intel_dp_supports_fec(intel_dp, connector as *const intel_connector, crtc_state as *const intel_crtc_state);
if __v2 {
return 22i32.wrapping_neg();
}
while num_joined_pipes <= 4 {
if intel_dp_joiner_candidate_valid(connector, (unsafe { (*adjusted_mode).hdisplay }) as i32, num_joined_pipes) {
// NOTE:
// The crtc_state->joiner_pipes should have been set at the end
// only if all the conditions are met. However that would mean
// that num_joined_pipes is passed around to all helpers and
// make them use it instead of using crtc_state->joiner_pipes
// directly or indirectly (via intel_crtc_num_joined_pipes()).
//
// For now, setting crtc_state->joiner_pipes to the candidate
// value to avoid the above churn and resetting it to 0, in case
// no joiner candidate is found to be suitable for the given
// configuration.
if num_joined_pipes > 1 {
unsafe {
(*crtc_state).joiner_pipes = GENMASK__11!(((*crtc).pipe as i32).wrapping_add(num_joined_pipes).wrapping_sub(1), unsafe { (*crtc).pipe }) as u8;
}
}
ret = intel_dp_compute_link_for_joined_pipes(encoder, crtc_state, conn_state, respect_downstream_limits);
if ret == 0 || ret == 35i32.wrapping_neg() {
break;
}
}
num_joined_pipes = num_joined_pipes.wrapping_add(1);
}
if ret < 0 {
unsafe {
(*crtc_state).joiner_pipes = 0;
}
}
ret
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_limited_color_range(crtc_state: *const intel_crtc_state, conn_state: *const drm_connector_state) -> bool {
let intel_conn_state: *const intel_digital_connector_state = to_intel_digital_connector_state__2!(conn_state);
let adjusted_mode: *const drm_display_mode = unsafe { core::ptr::addr_of!((*crtc_state).hw.adjusted_mode) };
// Our YCbCr output is always limited range.
// crtc_state->limited_color_range only applies to RGB,
// and it must never be set for YCbCr or we risk setting
// some conflicting bits in TRANSCONF which will mess up
// the colors on the monitor.
if ((unsafe { (*crtc_state).output_format }) as u32) != 0 {
return false;
}
if (unsafe { (*intel_conn_state).broadcast_rgb }) == 0 {
// See:
// CEA-861-E - 5.1 Default Encoding Parameters
// VESA DisplayPort Ver.1.2a - 5.1.1.1 Video Colorimetry
return (unsafe { (*crtc_state).pipe_bpp }) != 18 && ((unsafe { drm_default_rgb_quant_range(adjusted_mode) }) as u32) == 1;
} else {
return (unsafe { (*intel_conn_state).broadcast_rgb }) == 2;
}
}

fn intel_dp_port_has_audio(display: *mut intel_display, port: i32) -> bool {
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_g4x() }) as i32) != 0 {
return false;
}
if (DISPLAY_VER!(display) as i32) < 12 && (port as i32) == 0 {
return false;
}
true
}

fn intel_dp_compute_vsc_colorimetry(crtc_state: *const intel_crtc_state, conn_state: *const drm_connector_state, vsc: *mut drm_dp_vsc_sdp) {
let display: *mut intel_display = to_intel_display__3!(crtc_state);
if unsafe { (*crtc_state).has_panel_replay } {
// Prepare VSC Header for SU as per DP 2.0 spec, Table 2-223
// VSC SDP supporting 3D stereo, Panel Replay, and Pixel
// Encoding/Colorimetry Format indication.
unsafe {
(*vsc).revision = 7;
}
} else {
// Prepare VSC Header for SU as per DP 1.4 spec, Table 2-118
// VSC SDP supporting 3D stereo, PSR2, and Pixel Encoding/
// Colorimetry Format indication.
unsafe {
(*vsc).revision = 5;
}
}
unsafe {
(*vsc).length = 19;
}
// DP 1.4a spec, Table 2-120
match (unsafe { (*crtc_state).output_format }) as u32 {
2 => {
unsafe {
(*vsc).pixelformat = 1;
}
}
1 => {
unsafe {
(*vsc).pixelformat = 3;
}
}
_ => {
unsafe {
(*vsc).pixelformat = 0;
}
}
}
match (unsafe { (*conn_state).colorspace }) as u32 {
2 => {
unsafe {
(*vsc).colorimetry = 1;
}
}
3 => {
unsafe {
(*vsc).colorimetry = 2;
}
}
4 => {
unsafe {
(*vsc).colorimetry = 3;
}
}
5 => {
unsafe {
(*vsc).colorimetry = 4;
}
}
6 => {
unsafe {
(*vsc).colorimetry = 5;
}
}
8 => {
unsafe {
(*vsc).colorimetry = 6;
}
}
9 => {
unsafe {
(*vsc).colorimetry = 6;
}
}
10 => {
unsafe {
(*vsc).colorimetry = 7;
}
}
11 | 12 => {
unsafe {
(*vsc).colorimetry = 4;
}
}
_ => {
if ((unsafe { (*crtc_state).output_format }) as u32) == 1 {
unsafe {
(*vsc).colorimetry = 1;
}
} else {
unsafe {
(*vsc).colorimetry = 0;
}
}
// RGB->YCBCR color conversion uses the BT.709
// color space.
}
}
unsafe {
(*vsc).bpc = (*crtc_state).pipe_bpp / 3;
}
// only RGB pixelformat supports 6 bpc
let mut __v3: i64;
let __ret_warn_on: i32 = ((unsafe { (*vsc).bpc }) == 6 && ((unsafe { (*vsc).pixelformat }) as u32) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(vsc->bpc == 6 && vsc->pixelformat != DP_PIXELFORMAT_RGB)".as_ptr() as *mut u8) };
let mut __v4: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v4 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v5: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_compute_vsc_colorimetry_6, sym __slate_asm_string_intel_dp_compute_vsc_colorimetry_7, const 3091 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v5 = bug;
unsafe { __v4.unwrap()(__v5, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(vsc->bpc == 6 && vsc->pixelformat != DP_PIXELFORMAT_RGB)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
// All YCbCr formats are always limited range.
if ((unsafe { (*vsc).pixelformat }) as u32) == 0 {
unsafe {
(*vsc).dynamic_range = (if (*crtc_state).limited_color_range { 1i32 } else { 0i32 }) as u32;
}
} else {
unsafe {
(*vsc).dynamic_range = 1;
}
}
unsafe {
(*vsc).content_type = 0;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_842: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intel_dp_needs_as_sdp(intel_dp: *mut intel_dp, crtc_state: *mut intel_crtc_state) -> bool {
if !unsafe { (*intel_dp).as_sdp_v2_supported } {
return false;
}
// #TODO: Add AS SDP v1 support for PCONs (DP branch devices).
if drm_dp_is_branch((unsafe { (*intel_dp).dpcd.as_mut_ptr() as *mut u8 }) as *const u8) {
return false;
}
let __v0: bool = (unsafe { intel_psr_needs_alpm_aux_less(intel_dp, crtc_state as *const intel_crtc_state) }) && !unsafe { intel_psr_pr_async_video_timing_supported(intel_dp) };
if __v0 {
return true;
}
unsafe { intel_vrr_possible(crtc_state as *const intel_crtc_state) }
}

fn intel_dp_compute_as_sdp(intel_dp: *mut intel_dp, crtc_state: *mut intel_crtc_state) {
let as_sdp: *mut drm_dp_as_sdp = unsafe { core::ptr::addr_of_mut!((*crtc_state).infoframes.as_sdp) };
let adjusted_mode: *const drm_display_mode = (unsafe { core::ptr::addr_of_mut!((*crtc_state).hw.adjusted_mode) }) as *const drm_display_mode;
// #FIXME: SDP/infoframe updates aren’t truly atomic, and with the new
// cdclk->tc clock crossing we may transiently send a corrupted packet
// if the update lands mid‑transmission.
if !intel_dp_needs_as_sdp(intel_dp, crtc_state) {
return;
}
let __v0: u32 = unsafe { intel_hdmi_infoframe_enable(34u32) };
unsafe {
(*crtc_state).infoframes.enable = (*crtc_state).infoframes.enable | __v0;
}
unsafe {
(*as_sdp).sdp_type = 34;
}
unsafe {
(*as_sdp).length = 9;
}
unsafe {
(*as_sdp).duration_incr_ms = 0;
}
unsafe {
(*as_sdp).revision = 2;
}
unsafe {
(*as_sdp).vtotal = intel_vrr_vmin_vtotal(crtc_state as *const intel_crtc_state);
}
if unsafe { (*crtc_state).cmrr.enable } {
unsafe {
(*as_sdp).mode = 3;
}
unsafe {
(*as_sdp).target_rr = drm_mode_vrefresh(adjusted_mode);
}
unsafe {
(*as_sdp).target_rr_divider = true;
}
} else {
if unsafe { (*crtc_state).vrr.enable } {
unsafe {
(*as_sdp).mode = 0;
}
} else {
unsafe {
(*as_sdp).mode = 1;
}
}
}
// For Panel Replay with Async Video Timing support, the source can
// disable sending the AS SDP during PR Active state. In that case,
// the sink needs the coasting vtotal value to maintain the refresh
// rate. The HW only samples this on PR_ALPM_CTL[AS SDP Transmission
// in Active Disable], which we never program, so providing the value
// unconditionally when the sink advertises the capability is safe.
//
// #TODO:
// If we ever advertise support for coasting at other refresh targets,
// this logic could be revisited. For now, use the minimum refresh rate
// as the only safe coasting value.
if unsafe { intel_psr_pr_async_video_timing_supported(intel_dp) } {
unsafe {
(*as_sdp).coasting_vtotal = (*crtc_state).vrr.vmax as i32;
}
}
}

fn intel_dp_compute_vsc_sdp(intel_dp: *mut intel_dp, crtc_state: *mut intel_crtc_state, conn_state: *const drm_connector_state) {
if !intel_dp_needs_vsc_colorimetry(crtc_state as *const intel_crtc_state, conn_state) && !unsafe { (*crtc_state).has_psr } {
return;
}
let vsc: *mut drm_dp_vsc_sdp = unsafe { core::ptr::addr_of_mut!((*crtc_state).infoframes.vsc) };
let __v0: u32 = unsafe { intel_hdmi_infoframe_enable(7u32) };
unsafe {
(*crtc_state).infoframes.enable = (*crtc_state).infoframes.enable | __v0;
}
unsafe {
(*vsc).sdp_type = 7;
}
if intel_dp_needs_vsc_colorimetry(crtc_state as *const intel_crtc_state, conn_state) {
intel_dp_compute_vsc_colorimetry(crtc_state as *const intel_crtc_state, conn_state, vsc);
} else {
if unsafe { (*crtc_state).has_panel_replay } {
// [Panel Replay without colorimetry info]
// Prepare VSC Header for SU as per DP 2.0 spec, Table 2-223
// VSC SDP supporting 3D stereo + Panel Replay.
unsafe {
(*vsc).revision = 6;
}
unsafe {
(*vsc).length = 16;
}
} else {
if unsafe { (*crtc_state).has_sel_update } {
// [PSR2 without colorimetry]
// Prepare VSC Header for SU as per eDP 1.4 spec, Table 6-11
// 3D stereo + PSR/PSR2 + Y-coordinate.
unsafe {
(*vsc).revision = 4;
}
unsafe {
(*vsc).length = 14;
}
} else {
// [PSR1]
// Prepare VSC Header for SU as per DP 1.4 spec, Table 2-118
// VSC SDP supporting 3D stereo + PSR (applies to eDP v1.3 or
// higher).
unsafe {
(*vsc).revision = 2;
}
unsafe {
(*vsc).length = 8;
}
}
}
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_in_hdr_mode(conn_state: *const drm_connector_state) -> bool {
if unsafe { (*conn_state).hdr_output_metadata }.is_null() {
return false;
}
let hdr_metadata: *mut hdr_output_metadata = (unsafe { (*(*conn_state).hdr_output_metadata).data }) as *mut hdr_output_metadata;
((unsafe { (*hdr_metadata).__slate_anon_1.hdmi_metadata_type1.eotf }) as i32) == 2
}

fn intel_dp_compute_hdr_metadata_infoframe_sdp(intel_dp: *mut intel_dp, crtc_state: *mut intel_crtc_state, conn_state: *const drm_connector_state) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let mut ret: i32 = 0;
let drm_infoframe: *mut hdmi_drm_infoframe = unsafe { core::ptr::addr_of_mut!((*crtc_state).infoframes.drm.drm) };
if unsafe { (*conn_state).hdr_output_metadata }.is_null() {
return;
}
ret = unsafe { drm_hdmi_infoframe_set_hdr_metadata(drm_infoframe, conn_state) };
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"couldn't set HDR metadata in infoframe\n".as_ptr() as *const u8) };
return;
}
let __v1: u32 = unsafe { intel_hdmi_infoframe_enable(10u32) };
unsafe {
(*crtc_state).infoframes.enable = (*crtc_state).infoframes.enable | __v1;
}
}

fn can_enable_drrs(connector: *mut intel_connector, pipe_config: *const intel_crtc_state, downclock_mode: *const drm_display_mode) -> bool {
let display: *mut intel_display = to_intel_display!(connector);
if unsafe { (*pipe_config).vrr.enable } {
return false;
}
// DRRS and PSR can't be enable together, so giving preference to PSR
// as it allows more power-savings by complete shutting down display,
// so to guarantee this, intel_drrs_compute_config() must be called
// after intel_psr_compute_config().
if unsafe { (*pipe_config).has_psr } {
return false;
}
// FIXME missing FDI M2/N2 etc.
if unsafe { (*pipe_config).has_pch_encoder } {
return false;
}
if !unsafe { intel_cpu_transcoder_has_drrs(display, (*pipe_config).cpu_transcoder) } {
return false;
}
!downclock_mode.is_null() && ((unsafe { intel_panel_drrs_type(connector) }) as u32) == 2
}

fn intel_dp_drrs_compute_config(connector: *mut intel_connector, pipe_config: *mut intel_crtc_state, link_bpp_x16: i32) {
let display: *mut intel_display = to_intel_display!(connector);
let downclock_mode: *const drm_display_mode = unsafe { intel_panel_downclock_mode(connector, core::ptr::addr_of_mut!((*pipe_config).hw.adjusted_mode) as *const drm_display_mode) };
let mut pixel_clock: i32 = 0;
// FIXME all joined pipes share the same transcoder.
// Need to account for that when updating M/N live.
if has_seamless_m_n(connector) && (unsafe { (*pipe_config).joiner_pipes }) == 0 {
unsafe {
(*pipe_config).update_m_n = true;
}
}
if !can_enable_drrs(connector, pipe_config as *const intel_crtc_state, downclock_mode) {
if unsafe { intel_cpu_transcoder_has_m2_n2(display, (*pipe_config).cpu_transcoder) } {
unsafe { intel_zero_m_n(core::ptr::addr_of_mut!((*pipe_config).dp_m2_n2)) };
}
return;
}
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_ironlake() }) as i32) != 0 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_sandybridge() }) as i32) != 0 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_ivybridge() }) as i32) != 0 {
unsafe {
(*pipe_config).msa_timing_delay = (*connector).panel.vbt.edp.drrs_msa_timing_delay;
}
}
unsafe {
(*pipe_config).has_drrs = true;
}
pixel_clock = unsafe { (*downclock_mode).clock };
if unsafe { (*pipe_config).splitter.enable } {
pixel_clock /= (unsafe { (*pipe_config).splitter.link_count }) as i32;
}
unsafe { intel_link_compute_m_n(link_bpp_x16 as u16, (*pipe_config).lane_count as i32, pixel_clock, (*pipe_config).port_clock, intel_dp_bw_fec_overhead((*pipe_config).fec_enable), core::ptr::addr_of_mut!((*pipe_config).dp_m2_n2)) };
// FIXME: abstract this better
if unsafe { (*pipe_config).splitter.enable } {
let __v4: u32 = unsafe { (*pipe_config).dp_m2_n2.data_m };
unsafe {
(*pipe_config).dp_m2_n2.data_m = __v4.wrapping_mul(((*pipe_config).splitter.link_count as i32) as u32);
}
}
}

fn intel_dp_has_audio(encoder: *mut intel_encoder, conn_state: *const drm_connector_state) -> bool {
let display: *mut intel_display = to_intel_display!(encoder);
let intel_conn_state: *const intel_digital_connector_state = to_intel_digital_connector_state__2!(conn_state);
let connector: *mut intel_connector = to_intel_connector!(unsafe { (*conn_state).connector });
if !intel_dp_port_has_audio(display, unsafe { (*encoder).port }) {
return false;
}
if ((unsafe { (*intel_conn_state).force_audio }) as i32) == 0 {
return unsafe { (*connector).base.display_info.has_audio };
} else {
return ((unsafe { (*intel_conn_state).force_audio }) as i32) == 1;
}
}

fn intel_dp_compute_output_format(encoder: *mut intel_encoder, crtc_state: *mut intel_crtc_state, conn_state: *mut drm_connector_state, respect_downstream_limits: bool, sink_format: u32) -> i32 {
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let adjusted_mode: *const drm_display_mode = (unsafe { core::ptr::addr_of_mut!((*crtc_state).hw.adjusted_mode) }) as *const drm_display_mode;
if (intel_dp_sink_format_valid(connector, adjusted_mode, sink_format) as i32) != 0 {
return 22i32.wrapping_neg();
}
unsafe {
(*crtc_state).sink_format = sink_format;
}
unsafe {
(*crtc_state).output_format = intel_dp_output_format(connector, (*crtc_state).sink_format);
}
intel_dp_compute_link_config(encoder, crtc_state, conn_state, respect_downstream_limits)
}

fn intel_dp_compute_formats(encoder: *mut intel_encoder, crtc_state: *mut intel_crtc_state, conn_state: *mut drm_connector_state, respect_downstream_limits: bool) -> i32 {
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let info: *const drm_display_info = (unsafe { core::ptr::addr_of_mut!((*connector).base.display_info) }) as *const drm_display_info;
let adjusted_mode: *const drm_display_mode = (unsafe { core::ptr::addr_of_mut!((*crtc_state).hw.adjusted_mode) }) as *const drm_display_mode;
let mut ret: i32 = 0;
if unsafe { drm_mode_is_420_only(info, adjusted_mode) } {
ret = intel_dp_compute_output_format(encoder, crtc_state, conn_state, respect_downstream_limits, 1u32);
if ret != 0 && !respect_downstream_limits {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"YCbCr 4:2:0 mode but YCbCr 4:2:0 output not possible. Falling back to RGB.\n".as_ptr() as *const u8) };
ret = intel_dp_compute_output_format(encoder, crtc_state, conn_state, respect_downstream_limits, 0u32);
}
} else {
ret = intel_dp_compute_output_format(encoder, crtc_state, conn_state, respect_downstream_limits, 0u32);
let __v1: bool = ret != 0 && unsafe { drm_mode_is_420_also(info, adjusted_mode) };
if __v1 {
ret = intel_dp_compute_output_format(encoder, crtc_state, conn_state, respect_downstream_limits, 1u32);
}
}
ret
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_audio_compute_config(encoder: *mut intel_encoder, pipe_config: *mut intel_crtc_state, conn_state: *mut drm_connector_state) {
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
let __v0: bool = intel_dp_has_audio(encoder, conn_state as *const drm_connector_state) && unsafe { intel_audio_compute_config(encoder, pipe_config, conn_state) };
unsafe {
(*pipe_config).has_audio = __v0;
}
let __v1: bool = (unsafe { (*pipe_config).has_audio }) && intel_dp_is_uhbr(pipe_config as *const intel_crtc_state);
unsafe {
(*pipe_config).sdp_split_enable = __v1;
}
// SDP splitting for UHBR audio requires explicit sink capability in
// SST mode, whereas in MST mode it is inherently supported.
let __v2: bool = (unsafe { (*pipe_config).sdp_split_enable }) && !intel_crtc_has_type(pipe_config as *const intel_crtc_state, 11u32);
if __v2 {
unsafe {
(*pipe_config).sdp_split_enable = (*intel_dp).sst_split_sdp_support;
}
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_queue_modeset_retry_for_link(state: *mut intel_atomic_state, encoder: *mut intel_encoder, crtc_state: *const intel_crtc_state) {
let mut connector: *mut intel_connector = unsafe { core::mem::zeroed() };
let mut conn_state: *mut intel_digital_connector_state = unsafe { core::mem::zeroed() };
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
let mut i: i32 = 0;
if unsafe { (*intel_dp).needs_modeset_retry } {
return;
}
unsafe {
(*intel_dp).needs_modeset_retry = true;
}
if !intel_crtc_has_type(crtc_state, 11u32) {
unsafe { intel_connector_queue_modeset_retry_work((*intel_dp).attached_connector) };
return;
}
while i < unsafe { (*state).base.num_connector } {
let __v0: bool = if !unsafe { (*unsafe { (*state).base.connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
connector = to_intel_connector!(unsafe { (*unsafe { (*state).base.connectors }.wrapping_offset(i as isize)).ptr });
conn_state = to_intel_digital_connector_state!(unsafe { (*unsafe { (*state).base.connectors }.wrapping_offset(i as isize)).new_state });
true
} else {
false
};
if __v0 {
if !unsafe { (*conn_state).base.crtc }.is_null() {
if (unsafe { (*connector).mst.dp }) == intel_dp {
unsafe { intel_connector_queue_modeset_retry_work(connector) };
}
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_846: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn intel_dp_compute_config(state: *mut intel_atomic_state, encoder: *mut intel_encoder, pipe_config: *mut intel_crtc_state, conn_state: *mut drm_connector_state) -> i32 {
let display: *mut intel_display = to_intel_display!(encoder);
let adjusted_mode: *mut drm_display_mode = unsafe { core::ptr::addr_of_mut!((*pipe_config).hw.adjusted_mode) };
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let mut ret: i32 = 0;
let mut link_bpp_x16: i32 = 0;
if intel_dp_is_edp(intel_dp) {
ret = unsafe { intel_panel_compute_config(state, pipe_config, connector) };
if ret != 0 {
return ret;
}
}
if (unsafe { (*adjusted_mode).flags }) & 32u32 != 0 {
return 22i32.wrapping_neg();
}
if !unsafe { (*connector).base.interlace_allowed } && (unsafe { (*adjusted_mode).flags }) & 16u32 != 0 {
return 22i32.wrapping_neg();
}
if (unsafe { (*adjusted_mode).flags }) & 4096u32 != 0 {
return 22i32.wrapping_neg();
}
if intel_dp_hdisplay_bad(display, (unsafe { (*adjusted_mode).crtc_hdisplay }) as i32) {
return 22i32.wrapping_neg();
}
// Try to respect downstream TMDS clock limits first, if
// that fails assume the user might know something we don't.
ret = intel_dp_compute_formats(encoder, pipe_config, conn_state, true);
if ret != 0 {
ret = intel_dp_compute_formats(encoder, pipe_config, conn_state, false);
}
if ret != 0 {
return ret;
}
ret = unsafe { intel_pfit_compute_config(pipe_config, conn_state as *const drm_connector_state) };
if ret != 0 {
return ret;
}
unsafe {
(*pipe_config).limited_color_range = intel_dp_limited_color_range(pipe_config as *const intel_crtc_state, conn_state as *const drm_connector_state);
}
if intel_dp_is_uhbr(pipe_config as *const intel_crtc_state) {
// 128b/132b SST also needs this
unsafe {
(*pipe_config).mst_master_transcoder = (*pipe_config).cpu_transcoder;
}
} else {
unsafe {
(*pipe_config).enhanced_framing = drm_dp_enhanced_frame_cap(((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8);
}
}
if unsafe { (*pipe_config).dsc.compression_enable } {
link_bpp_x16 = (unsafe { (*pipe_config).dsc.compressed_bpp_x16 }) as i32;
} else {
link_bpp_x16 = intel_dp_output_format_link_bpp_x16(unsafe { (*pipe_config).output_format }, unsafe { (*pipe_config).pipe_bpp });
}
if (unsafe { (*intel_dp).mso_link_count }) != 0 {
let n: i32 = unsafe { (*intel_dp).mso_link_count };
let overlap: i32 = unsafe { (*intel_dp).mso_pixel_overlap };
unsafe {
(*pipe_config).splitter.enable = true;
}
unsafe {
(*pipe_config).splitter.link_count = n as u8;
}
unsafe {
(*pipe_config).splitter.pixel_overlap = overlap as u8;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"MSO link count %d, pixel overlap %d\n".as_ptr() as *const u8, n, overlap) };
unsafe {
(*adjusted_mode).crtc_hdisplay = (((*adjusted_mode).crtc_hdisplay as i32) / n).wrapping_add(overlap) as u16;
}
unsafe {
(*adjusted_mode).crtc_hblank_start = (((*adjusted_mode).crtc_hblank_start as i32) / n).wrapping_add(overlap) as u16;
}
unsafe {
(*adjusted_mode).crtc_hblank_end = (((*adjusted_mode).crtc_hblank_end as i32) / n).wrapping_add(overlap) as u16;
}
unsafe {
(*adjusted_mode).crtc_hsync_start = (((*adjusted_mode).crtc_hsync_start as i32) / n).wrapping_add(overlap) as u16;
}
unsafe {
(*adjusted_mode).crtc_hsync_end = (((*adjusted_mode).crtc_hsync_end as i32) / n).wrapping_add(overlap) as u16;
}
unsafe {
(*adjusted_mode).crtc_htotal = (((*adjusted_mode).crtc_htotal as i32) / n).wrapping_add(overlap) as u16;
}
unsafe {
(*adjusted_mode).crtc_clock = (*adjusted_mode).crtc_clock / n;
}
}
intel_dp_audio_compute_config(encoder, pipe_config, conn_state);
if !intel_dp_is_uhbr(pipe_config as *const intel_crtc_state) {
unsafe { intel_link_compute_m_n(link_bpp_x16 as u16, (*pipe_config).lane_count as i32, (*adjusted_mode).crtc_clock, (*pipe_config).port_clock, intel_dp_bw_fec_overhead((*pipe_config).fec_enable), core::ptr::addr_of_mut!((*pipe_config).dp_m_n)) };
}
ret = intel_dp_compute_min_hblank(pipe_config, conn_state as *const drm_connector_state);
if ret != 0 {
return ret;
}
// FIXME: abstract this better
if unsafe { (*pipe_config).splitter.enable } {
let __v5: u32 = unsafe { (*pipe_config).dp_m_n.data_m };
unsafe {
(*pipe_config).dp_m_n.data_m = __v5.wrapping_mul(((*pipe_config).splitter.link_count as i32) as u32);
}
}
unsafe { intel_vrr_compute_config(pipe_config, conn_state) };
unsafe { intel_psr_compute_config(intel_dp, pipe_config, conn_state) };
intel_dp_compute_as_sdp(intel_dp, pipe_config);
unsafe { intel_alpm_lobf_compute_config(intel_dp, pipe_config, conn_state) };
intel_dp_drrs_compute_config(connector, pipe_config, link_bpp_x16);
intel_dp_compute_vsc_sdp(intel_dp, pipe_config, conn_state as *const drm_connector_state);
intel_dp_compute_hdr_metadata_infoframe_sdp(intel_dp, pipe_config, conn_state as *const drm_connector_state);
intel_dp_tunnel_atomic_compute_stream_bw(state, intel_dp, connector as *const intel_connector, pipe_config)
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_set_link_params(intel_dp: *mut intel_dp, link_rate: i32, lane_count: i32) {
unsafe { memset(((*intel_dp).train_set.as_mut_ptr() as *mut u8) as *mut (), 0i32, core::mem::size_of::<[u8; 4]>() as u64) };
unsafe {
(*intel_dp).link.active = false;
}
unsafe {
(*intel_dp).needs_modeset_retry = false;
}
unsafe {
(*intel_dp).link_rate = link_rate;
}
unsafe {
(*intel_dp).lane_count = lane_count as u8;
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_reset_link_params(intel_dp: *mut intel_dp) {
// TODO: Remove the following reset of link capabilities, as
// this isn't needed after intel_dp_link_caps_update(reset=true)
// was called.
unsafe { intel_dp_link_caps_reset((*intel_dp).link.caps) };
intel_dp_tunnel_uhbr_lanes_wa_apply(intel_dp);
unsafe {
(*intel_dp).link.mst_probed_lane_count = 0;
}
unsafe {
(*intel_dp).link.mst_probed_rate = 0;
}
unsafe { intel_dp_link_training_reset((*intel_dp).link.training) };
}

/// Enable backlight PWM and backlight PP control.
#[unsafe(no_mangle)]
extern "C" fn intel_edp_backlight_on(crtc_state: *const intel_crtc_state, conn_state: *const drm_connector_state) {
let display: *mut intel_display = to_intel_display__3!(crtc_state);
let intel_dp: *mut intel_dp = enc_to_intel_dp(to_intel_encoder!(unsafe { (*conn_state).best_encoder }));
if !intel_dp_is_edp(intel_dp) {
return;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"\n".as_ptr() as *const u8) };
unsafe { intel_backlight_enable(crtc_state, conn_state) };
unsafe { intel_pps_backlight_on(intel_dp) };
}

/// Disable backlight PP control and backlight PWM.
#[unsafe(no_mangle)]
extern "C" fn intel_edp_backlight_off(old_conn_state: *const drm_connector_state) {
let intel_dp: *mut intel_dp = enc_to_intel_dp(to_intel_encoder!(unsafe { (*old_conn_state).best_encoder }));
let display: *mut intel_display = to_intel_display__2!(intel_dp);
if !intel_dp_is_edp(intel_dp) {
return;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"\n".as_ptr() as *const u8) };
unsafe { intel_pps_backlight_off(intel_dp) };
unsafe { intel_backlight_disable(old_conn_state) };
}

fn downstream_hpd_needs_d0(intel_dp: *mut intel_dp) -> bool {
// DPCD 1.2+ should support BRANCH_DEVICE_CTRL, and thus
// be capable of signalling downstream hpd with a long pulse.
// Whether or not that means D3 is safe to use is not clear,
// but let's assume so until proven otherwise.
//
// FIXME should really check all downstream ports...
let __v0: bool = ((unsafe { *((*intel_dp).dpcd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) == 17 && drm_dp_is_branch((unsafe { (*intel_dp).dpcd.as_mut_ptr() as *mut u8 }) as *const u8);
__v0 && ((unsafe { *((*intel_dp).downstream_ports.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) & 8i32 != 0
}

fn write_dsc_decompression_flag(aux: *mut drm_dp_aux, flag: u8, set: bool) -> i32 {
let mut err: i32 = 0;
let mut val: u8 = 0;
err = drm_dp_dpcd_readb(aux, 352u32, core::ptr::addr_of_mut!(val)) as i32;
if err < 0 {
return err;
}
if set {
val = ((val as i32) | (flag as i32)) as u8;
} else {
let __v3: u8 = ((val as i32) & !(flag as i32)) as u8;
val = __v3;
}
drm_dp_dpcd_writeb(aux, 352u32, val) as i32
}

fn intel_dp_sink_set_dsc_decompression(connector: *mut intel_connector, enable: bool) {
let display: *mut intel_display = to_intel_display!(connector);
if write_dsc_decompression_flag(unsafe { (*connector).dp.dsc_decompression_aux }, 1u8, enable) < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Failed to %s sink decompression state\n".as_ptr() as *const u8, str_enable_disable(enable)) };
}
}

fn intel_dp_sink_set_dsc_passthrough(connector: *const intel_connector, enable: bool) {
let display: *mut intel_display = to_intel_display__10!(connector);
let aux: *mut drm_dp_aux = if !unsafe { (*connector).mst.port }.is_null() { unsafe { (*(*connector).mst.port).passthrough_aux } } else { core::ptr::null_mut::<drm_dp_aux>() };
if aux.is_null() {
return;
}
if write_dsc_decompression_flag(aux, 2u8, enable) < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Failed to %s sink compression passthrough state\n".as_ptr() as *const u8, str_enable_disable(enable)) };
}
}

fn intel_dp_dsc_aux_ref_count(state: *mut intel_atomic_state, connector: *const intel_connector, for_get_ref: bool) -> i32 {
let display: *mut intel_display = to_intel_display!(state);
let mut _connector_iter: *mut drm_connector = unsafe { core::mem::zeroed() };
let mut old_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut new_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut ref_count: i32 = 0;
let mut i: i32 = 0;
// On SST the decompression AUX device won't be shared, each connector
// uses for this its own AUX targeting the sink device.
if unsafe { (*connector).mst.dp }.is_null() {
return if ((unsafe { (*connector).dp.__slate_bits_0.__get_dsc_decompression_enabled() }) as i32) != 0 { 1 } else { 0 };
}
while i < unsafe { (*core::ptr::addr_of_mut!((*state).base)).num_connector } {
let __v1: bool = if !unsafe { (*unsafe { (*core::ptr::addr_of_mut!((*state).base)).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
_connector_iter = unsafe { (*unsafe { (*core::ptr::addr_of_mut!((*state).base)).connectors }.wrapping_offset(i as isize)).ptr };
let __v2: *mut drm_connector_state = unsafe { (*unsafe { (*core::ptr::addr_of_mut!((*state).base)).connectors }.wrapping_offset(i as isize)).old_state };
old_conn_state = __v2;
let __v3: *mut drm_connector_state = unsafe { (*unsafe { (*core::ptr::addr_of_mut!((*state).base)).connectors }.wrapping_offset(i as isize)).new_state };
new_conn_state = __v3;
true
} else {
false
};
if __v1 {
let connector_iter: *const intel_connector = to_intel_connector!(_connector_iter) as *const intel_connector;
if (unsafe { (*connector_iter).mst.dp }) == unsafe { (*connector).mst.dp } {
if ((unsafe { (*connector_iter).dp.__slate_bits_0.__get_dsc_decompression_enabled() }) as i32) != 0 {
let mut __v6: i64;
let __ret_warn_on: i32 = (for_get_ref && unsafe { (*new_conn_state).crtc }.is_null() || !for_get_ref && unsafe { (*old_conn_state).crtc }.is_null()) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON((for_get_ref && !new_conn_state->crtc) || (!for_get_ref && !old_conn_state->crtc))".as_ptr() as *mut u8) };
let mut __v7: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v7 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v8: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_dsc_aux_ref_count_9, sym __slate_asm_string_intel_dp_dsc_aux_ref_count_10, const 3814 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v8 = bug;
unsafe { __v7.unwrap()(__v8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON((for_get_ref && !new_conn_state->crtc) || (!for_get_ref && !old_conn_state->crtc))".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v6 = (__ret_warn_on != 0) as i64;
if (unsafe { (*connector_iter).dp.dsc_decompression_aux }) == unsafe { (*connector).dp.dsc_decompression_aux } {
ref_count = ref_count.wrapping_add(1);
}
}
}
}
i = i.wrapping_add(1);
}
ref_count
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_853: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intel_dp_dsc_aux_get_ref(state: *mut intel_atomic_state, connector: *mut intel_connector) -> bool {
let ret: bool = intel_dp_dsc_aux_ref_count(state, connector as *const intel_connector, true) == 0;
unsafe {
(*connector).dp.__slate_bits_0.__set_dsc_decompression_enabled(1u8);
}
ret
}

fn intel_dp_dsc_aux_put_ref(state: *mut intel_atomic_state, connector: *mut intel_connector) -> bool {
unsafe {
(*connector).dp.__slate_bits_0.__set_dsc_decompression_enabled(0u8);
}
intel_dp_dsc_aux_ref_count(state, connector as *const intel_connector, false) == 0
}

/// intel_dp_sink_enable_decompression - Enable DSC decompression in sink/last branch device
/// @state: atomic state
/// @connector: connector to enable the decompression for
/// @new_crtc_state: new state for the CRTC driving @connector
///
/// Enable the DSC decompression if required in the %DP_DSC_ENABLE DPCD
/// register of the appropriate sink/branch device. On SST this is always the
/// sink device, whereas on MST based on each device's DSC capabilities it's
/// either the last branch device (enabling decompression in it) or both the
/// last branch device (enabling passthrough in it) and the sink device
/// (enabling decompression in it).
#[unsafe(no_mangle)]
extern "C" fn intel_dp_sink_enable_decompression(state: *mut intel_atomic_state, connector: *mut intel_connector, new_crtc_state: *const intel_crtc_state) {
let display: *mut intel_display = to_intel_display!(state);
if !unsafe { (*new_crtc_state).dsc.compression_enable } {
return;
}
let mut __v1: i64;
let __ret_warn_on: i32 = (unsafe { (*connector).dp.dsc_decompression_aux }.is_null() || ((unsafe { (*connector).dp.__slate_bits_0.__get_dsc_decompression_enabled() }) as i32) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!connector->dp.dsc_decompression_aux || connector->dp.dsc_decompression_enabled)".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_sink_enable_decompression_4, sym __slate_asm_string_intel_dp_sink_enable_decompression_5, const 3866 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!connector->dp.dsc_decompression_aux || connector->dp.dsc_decompression_enabled)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if __v1 != 0 {
return;
}
if !intel_dp_dsc_aux_get_ref(state, connector) {
return;
}
intel_dp_sink_set_dsc_passthrough(connector as *const intel_connector, true);
intel_dp_sink_set_dsc_decompression(connector, true);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_854: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

/// intel_dp_sink_disable_decompression - Disable DSC decompression in sink/last branch device
/// @state: atomic state
/// @connector: connector to disable the decompression for
/// @old_crtc_state: old state for the CRTC driving @connector
///
/// Disable the DSC decompression if required in the %DP_DSC_ENABLE DPCD
/// register of the appropriate sink/branch device, corresponding to the
/// sequence in intel_dp_sink_enable_decompression().
#[unsafe(no_mangle)]
extern "C" fn intel_dp_sink_disable_decompression(state: *mut intel_atomic_state, connector: *mut intel_connector, old_crtc_state: *const intel_crtc_state) {
let display: *mut intel_display = to_intel_display!(state);
if !unsafe { (*old_crtc_state).dsc.compression_enable } {
return;
}
let mut __v1: i64;
let __ret_warn_on: i32 = (unsafe { (*connector).dp.dsc_decompression_aux }.is_null() || ((unsafe { (*connector).dp.__slate_bits_0.__get_dsc_decompression_enabled() }) as i32) == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!connector->dp.dsc_decompression_aux || !connector->dp.dsc_decompression_enabled)".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_sink_disable_decompression_4, sym __slate_asm_string_intel_dp_sink_disable_decompression_5, const 3897 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!connector->dp.dsc_decompression_aux || !connector->dp.dsc_decompression_enabled)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if __v1 != 0 {
return;
}
if !intel_dp_dsc_aux_put_ref(state, connector) {
return;
}
intel_dp_sink_set_dsc_decompression(connector, false);
intel_dp_sink_set_dsc_passthrough(connector as *const intel_connector, false);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_855: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intel_dp_init_source_oui(intel_dp: *mut intel_dp) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let mut oui: [u8; 3] = [0, 170, 1];
let mut buf: [u8; 3] = {
let mut __t0: [u8; 3] = unsafe { core::mem::zeroed() };
__t0
};
let mut __v1: bool;
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_856() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*intel_dp).oui_valid) as *const bool) };
if __v1 {
return;
}
loop {
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_857() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*intel_dp).oui_valid, true);
if true {
break;
}
}
// During driver init, we want to be careful and avoid changing the source OUI if it's
// already set to what we want, so as to avoid clearing any state by accident
if (unsafe { drm_dp_dpcd_read(core::ptr::addr_of_mut!((*intel_dp).aux), 768u32, buf.as_mut_ptr() as *mut (), core::mem::size_of::<[u8; 3]>() as u64) }) < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Failed to read source OUI\n".as_ptr() as *const u8) };
}
if (unsafe { memcmp(oui.as_mut_ptr() as *const (), buf.as_mut_ptr() as *const (), core::mem::size_of::<[u8; 3]>() as u64) }) == 0 {
// Assume the OUI was written now.
unsafe {
(*intel_dp).last_oui_write = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
}
return;
}
if (unsafe { drm_dp_dpcd_write(core::ptr::addr_of_mut!((*intel_dp).aux), 768u32, oui.as_mut_ptr() as *mut (), core::mem::size_of::<[u8; 3]>() as u64) }) < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Failed to write source OUI\n".as_ptr() as *const u8) };
loop {
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_858() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*intel_dp).oui_valid, false);
if true {
break;
}
}
}
unsafe {
(*intel_dp).last_oui_write = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_wait_source_oui(intel_dp: *mut intel_dp) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Performing OUI wait (%u ms)\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name, (*connector).panel.vbt.backlight.hdr_dpcd_refresh_timeout as i32) };
wait_remaining_ms_from_jiffies(unsafe { (*intel_dp).last_oui_write }, (unsafe { (*connector).panel.vbt.backlight.hdr_dpcd_refresh_timeout }) as i32);
}

/// If the device supports it, try to set the power state appropriately
#[unsafe(no_mangle)]
extern "C" fn intel_dp_set_power(intel_dp: *mut intel_dp, mode: u8) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let encoder: *mut intel_encoder = unsafe { core::ptr::addr_of_mut!((*dp_to_dig_port(intel_dp)).base) };
let mut ret: i32 = 0;
// Should have a valid DPCD by this point
if ((unsafe { *((*intel_dp).dpcd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) < 17 {
return;
}
if (mode as i32) != 1 {
if downstream_hpd_needs_d0(intel_dp) {
return;
}
ret = drm_dp_dpcd_writeb(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 1536u32, mode) as i32;
} else {
let dig_port: *mut intel_digital_port = dp_to_dig_port(intel_dp);
unsafe { intel_lspcon_resume(dig_port) };
// Write the source OUI as early as possible
intel_dp_init_source_oui(intel_dp);
// When turning on, we need to retry for 1ms to give the sink
// time to wake up.
for i in 0i32..3i32 {
ret = drm_dp_dpcd_writeb(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 1536u32, mode) as i32;
if ret == 1 {
break;
}
unsafe { msleep(1u32) };
}
let __v4: bool = ret == 1 && unsafe { intel_lspcon_active(dig_port) };
if __v4 {
unsafe { intel_lspcon_wait_pcon_mode(dig_port) };
}
}
if ret != 1 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[ENCODER:%d:%s] Set power to %s failed\n".as_ptr() as *const u8, (*encoder).base.base.id, (*encoder).base.name, if (mode as i32) == 1 { c"D0".as_ptr() as *mut u8 } else { c"D3".as_ptr() as *mut u8 }) };
}
}

/// intel_dp_sync_state - sync the encoder state during init/resume
/// @encoder: intel encoder to sync
/// @crtc_state: state for the CRTC connected to the encoder
///
/// Sync any state stored in the encoder wrt. HW state during driver init
/// and system resume.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.intel_dp.intel_dp_sync_state")]
extern "C" fn intel_dp_sync_state(encoder: *mut intel_encoder, crtc_state: *const intel_crtc_state) {
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
let mut dpcd_updated: bool = false;
// Don't clobber DPCD if it's been already read out during output
// setup (eDP) or detect.
if !crtc_state.is_null() && ((unsafe { *((*intel_dp).dpcd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) == 0 {
intel_dp_get_dpcd(intel_dp);
dpcd_updated = true;
}
intel_dp_tunnel_resume(intel_dp, crtc_state, dpcd_updated);
if !crtc_state.is_null() {
intel_dp_reset_link_params(intel_dp);
intel_dp_set_link_params(intel_dp, unsafe { (*crtc_state).port_clock }, (unsafe { (*crtc_state).lane_count }) as i32);
unsafe {
(*intel_dp).link.active = true;
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.intel_dp.intel_dp_initial_fastset_check")]
extern "C" fn intel_dp_initial_fastset_check(encoder: *mut intel_encoder, crtc_state: *mut intel_crtc_state) -> bool {
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
let mut fastset: bool = true;
// If BIOS has set an unsupported or non-standard link rate for some
// reason force an encoder recompute and full modeset.
if intel_dp_rate_index(unsafe { (*intel_dp).source_rates }, unsafe { (*intel_dp).num_source_rates }, unsafe { (*crtc_state).port_clock }) < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[ENCODER:%d:%s] Forcing full modeset due to unsupported link rate\n".as_ptr() as *const u8, (*encoder).base.base.id, (*encoder).base.name) };
unsafe {
let __t0 = true;
(*crtc_state).uapi.__slate_bits_0.__set_connectors_changed(__t0);
}
fastset = false;
}
// FIXME hack to force full modeset when DSC is being used.
//
// As long as we do not have full state readout and config comparison
// of crtc_state->dsc, we have no way to ensure reliable fastset.
// Remove once we have readout for DSC.
if unsafe { (*crtc_state).dsc.compression_enable } {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[ENCODER:%d:%s] Forcing full modeset due to DSC being enabled\n".as_ptr() as *const u8, (*encoder).base.base.id, (*encoder).base.name) };
unsafe {
let __t1 = true;
(*crtc_state).uapi.__slate_bits_0.__set_mode_changed(__t1);
}
fastset = false;
}
if CAN_PANEL_REPLAY!(intel_dp) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[ENCODER:%d:%s] Forcing full modeset to compute panel replay state\n".as_ptr() as *const u8, (*encoder).base.base.id, (*encoder).base.name) };
unsafe {
let __t2 = true;
(*crtc_state).uapi.__slate_bits_0.__set_mode_changed(__t2);
}
fastset = false;
}
fastset
}

fn intel_dp_get_pcon_dsc_cap(intel_dp: *mut intel_dp) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
// Clear the cached register set to avoid using stale values
unsafe { memset(((*intel_dp).pcon_dsc_dpcd.as_mut_ptr() as *mut u8) as *mut (), 0i32, core::mem::size_of::<[u8; 13]>() as u64) };
if !drm_dp_is_branch((unsafe { (*intel_dp).dpcd.as_mut_ptr() as *mut u8 }) as *const u8) {
return;
}
if (unsafe { drm_dp_dpcd_read(core::ptr::addr_of_mut!((*intel_dp).aux), 146u32, ((*intel_dp).pcon_dsc_dpcd.as_mut_ptr() as *mut u8) as *mut (), core::mem::size_of::<[u8; 13]>() as u64) }) < 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* Failed to read DPCD register 0x%x\n".as_ptr() as *const u8, 146i32) };
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"PCON ENCODER DSC DPCD: %*ph\n".as_ptr() as *const u8, core::mem::size_of::<[u8; 13]>() as i32, (*intel_dp).pcon_dsc_dpcd.as_mut_ptr() as *mut u8) };
}

fn intel_dp_pcon_get_frl_mask(frl_bw_mask: u8) -> i32 {
let mut i: i32 = ARRAY_SIZE__51!().wrapping_sub(1) as i32;
while i >= 0 {
if (frl_bw_mask as i32) & 1i32 << i != 0 {
return unsafe { bw_gbps.0[i as usize] };
}
i = i.wrapping_sub(1);
}
0
}

static bw_gbps: __SlateAlign16<[i32; 6]> = __SlateAlign16([9i32, 18i32, 24i32, 32i32, 40i32, 48i32]);

fn intel_dp_pcon_set_frl_mask(max_frl: i32) -> i32 {
match max_frl {
48 => {
return 32i32;
}
40 => {
return 16i32;
}
32 => {
return 8i32;
}
24 => {
return 4i32;
}
18 => {
return 2i32;
}
9 => {
return 1i32;
}
_ => {
}
}
0
}

fn intel_dp_hdmi_sink_max_frl(intel_dp: *mut intel_dp) -> i32 {
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let info: *const drm_display_info = (unsafe { core::ptr::addr_of_mut!((*connector).base.display_info) }) as *const drm_display_info;
let mut max_frl_rate: i32 = 0;
let max_lanes: i32 = (unsafe { (*info).hdmi.max_lanes }) as i32;
let rate_per_lane: i32 = (unsafe { (*info).hdmi.max_frl_rate_per_lane }) as i32;
let mut max_dsc_lanes: i32 = 0;
let mut dsc_rate_per_lane: i32 = 0;
max_frl_rate = max_lanes.wrapping_mul(rate_per_lane);
// The sink's DSC max FRL rate only applies to compressed video
// transport, which requires a DSC 1.2 encoder in the PCON. Without
// one the HDMI link always carries uncompressed video, for which
// the regular max FRL rate is the limit.
if (unsafe { drm_dp_pcon_enc_is_dsc_1_2(((*intel_dp).pcon_dsc_dpcd.as_mut_ptr() as *mut u8) as *const u8) }) && unsafe { (*info).hdmi.dsc_cap.v_1p2 } {
max_dsc_lanes = (unsafe { (*info).hdmi.dsc_cap.max_lanes }) as i32;
dsc_rate_per_lane = (unsafe { (*info).hdmi.dsc_cap.max_frl_rate_per_lane }) as i32;
if max_dsc_lanes != 0 && dsc_rate_per_lane != 0 {
let mut __v0: i32;
let __UNIQUE_ID_x__860: i32 = max_frl_rate;
let __UNIQUE_ID_y__861: i32 = max_dsc_lanes.wrapping_mul(dsc_rate_per_lane);
loop {
if __types_ok!(__UNIQUE_ID_x__860, __UNIQUE_ID_y__861) == 0 {
unsafe { __compiletime_assert_862() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__860 < __UNIQUE_ID_y__861 { __UNIQUE_ID_x__860 } else { __UNIQUE_ID_y__861 };
max_frl_rate = __v0;
}
}
max_frl_rate
}

fn intel_dp_pcon_is_frl_trained(intel_dp: *mut intel_dp, max_frl_bw_mask: u8, frl_trained_mask: *mut u8) -> bool {
let __v0: bool = (unsafe { drm_dp_pcon_hdmi_link_active(core::ptr::addr_of_mut!((*intel_dp).aux)) }) && (unsafe { drm_dp_pcon_hdmi_link_mode(core::ptr::addr_of_mut!((*intel_dp).aux), frl_trained_mask) }) == 1;
if __v0 && ((unsafe { *frl_trained_mask }) as i32) >= (max_frl_bw_mask as i32) {
return true;
}
false
}

fn intel_dp_pcon_start_frl_training(intel_dp: *mut intel_dp) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __expired: bool = unsafe { core::mem::zeroed() };
let mut ___ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(___ret) as *mut i32;
let mut __timeout: i64 = unsafe { core::mem::zeroed() };
let mut __sleep_us: u64 = unsafe { core::mem::zeroed() };
let mut __timeout_us: u64 = unsafe { core::mem::zeroed() };
let mut __v7: bool = unsafe { core::mem::zeroed() };
let mut __expired_1: bool = unsafe { core::mem::zeroed() };
let mut ___ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(___ret) as *mut i32;
let mut __timeout_1: i64 = unsafe { core::mem::zeroed() };
let mut __sleep_us_1: u64 = unsafe { core::mem::zeroed() };
let mut __timeout_us_1: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__864: i32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__863: i32 = unsafe { core::mem::zeroed() };
let mut is_active: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut bool = core::ptr::addr_of_mut!(is_active) as *mut bool;
let mut frl_trained_mask: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut u8 = core::ptr::addr_of_mut!(frl_trained_mask) as *mut u8;
let mut max_frl_bw_mask: u8 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut max_edid_frl_bw: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut i32 = core::ptr::addr_of_mut!(max_edid_frl_bw) as *mut i32;
let mut max_pcon_frl_bw: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut i32 = core::ptr::addr_of_mut!(max_pcon_frl_bw) as *mut i32;
let mut max_frl_bw: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut i32 = core::ptr::addr_of_mut!(max_frl_bw) as *mut i32;
let mut display: *mut intel_display = unsafe { core::mem::zeroed() };
unsafe {
display = to_intel_display__2!(intel_dp);
let __slate_auto_bytes_2 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
max_frl_bw_mask = 0;
let __slate_auto_bytes_6 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_7 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 1usize);
*__slate_slot_22 = (*intel_dp).dfp.pcon_max_frl_bw;
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 1u32, c"PCON max rate = %d Gbps\n".as_ptr() as *const u8, *__slate_slot_22);
*__slate_slot_21 = intel_dp_hdmi_sink_max_frl(intel_dp);
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 1u32, c"Sink max rate from EDID = %d Gbps\n".as_ptr() as *const u8, *__slate_slot_21);
__UNIQUE_ID_x__863 = *__slate_slot_21;
__UNIQUE_ID_y__864 = *__slate_slot_22;
loop {
if __types_ok!(__UNIQUE_ID_x__863, __UNIQUE_ID_y__864) == 0 {
__compiletime_assert_865();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
*__slate_slot_23 = if __UNIQUE_ID_x__863 < __UNIQUE_ID_y__864 { __UNIQUE_ID_x__863 } else { __UNIQUE_ID_y__864 };
if *__slate_slot_23 <= 0 {
return 22i32.wrapping_neg();
} else {
max_frl_bw_mask = intel_dp_pcon_set_frl_mask(*__slate_slot_23) as u8;
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 1u32, c"MAX_FRL_BW_MASK = %u\n".as_ptr() as *const u8, max_frl_bw_mask as i32);
if !intel_dp_pcon_is_frl_trained(intel_dp, max_frl_bw_mask, core::ptr::addr_of_mut!(*__slate_slot_18)) {
*__slate_slot_20 = drm_dp_pcon_frl_prepare(core::ptr::addr_of_mut!((*intel_dp).aux), false);
if *__slate_slot_20 < 0 {
return *__slate_slot_20;
} else {
__timeout_us_1 = 500000u64;
__sleep_us_1 = 1000;
__timeout_1 = ktime_add_us(ktime_get(), __timeout_us_1);
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
might_sleep_if!(__sleep_us_1 != 0);
'__join_21: {
'__join_27: {
loop {
if __timeout_us_1 != 0 {
__v7 = ktime_compare(ktime_get(), __timeout_1) > 0;
} else {
__v7 = false;
}
__expired_1 = __v7;
core::arch::asm!("", options(nostack, att_syntax));
*__slate_slot_17 = drm_dp_pcon_is_frl_ready(core::ptr::addr_of_mut!((*intel_dp).aux));
if *__slate_slot_17 {
break '__join_27;
} else {
if __expired_1 {
break;
} else {
if __sleep_us_1 != 0 {
usleep_range((__sleep_us_1 >> 2i32).wrapping_add(1), __sleep_us_1);
}
cpu_relax();
}
}
}
*__slate_slot_9 = 110i32.wrapping_neg();
break '__join_21;
}
*__slate_slot_9 = 0;
}
// Wait for PCON to be FRL Ready
*__slate_slot_20 = *__slate_slot_9;
if *__slate_slot_20 != 0 {
return *__slate_slot_20;
} else {
*__slate_slot_20 = drm_dp_pcon_frl_configure_1(core::ptr::addr_of_mut!((*intel_dp).aux), *__slate_slot_23, 0u8);
if *__slate_slot_20 < 0 {
return *__slate_slot_20;
} else {
*__slate_slot_20 = drm_dp_pcon_frl_configure_2(core::ptr::addr_of_mut!((*intel_dp).aux), max_frl_bw_mask as i32, 0u8);
if *__slate_slot_20 < 0 {
return *__slate_slot_20;
} else {
*__slate_slot_20 = drm_dp_pcon_frl_enable(core::ptr::addr_of_mut!((*intel_dp).aux));
if *__slate_slot_20 < 0 {
return *__slate_slot_20;
} else {
__timeout_us = 1000000u64;
__sleep_us = 1000;
__timeout = ktime_add_us(ktime_get(), __timeout_us);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
might_sleep_if!(__sleep_us != 0);
'__join_2: {
'__join_8: {
loop {
if __timeout_us != 0 {
__v0 = ktime_compare(ktime_get(), __timeout) > 0;
} else {
__v0 = false;
}
__expired = __v0;
core::arch::asm!("", options(nostack, att_syntax));
*__slate_slot_17 = intel_dp_pcon_is_frl_trained(intel_dp, max_frl_bw_mask, core::ptr::addr_of_mut!(*__slate_slot_18));
if *__slate_slot_17 {
break '__join_8;
} else {
if __expired {
break;
} else {
if __sleep_us != 0 {
usleep_range((__sleep_us >> 2i32).wrapping_add(1), __sleep_us);
}
cpu_relax();
}
}
}
*__slate_slot_2 = 110i32.wrapping_neg();
break '__join_2;
}
*__slate_slot_2 = 0;
}
// Wait for FRL to be completed
// Check if the HDMI Link is up and active.
*__slate_slot_20 = *__slate_slot_2;
if *__slate_slot_20 != 0 {
return *__slate_slot_20;
}
}
}
}
}
}
}
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 1u32, c"FRL_TRAINED_MASK = %u\n".as_ptr() as *const u8, *__slate_slot_18 as i32);
(*intel_dp).frl.trained_rate_gbps = intel_dp_pcon_get_frl_mask(*__slate_slot_18);
(*intel_dp).frl.is_trained = true;
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 1u32, c"FRL trained with : %d Gbps\n".as_ptr() as *const u8, (*intel_dp).frl.trained_rate_gbps);
return 0;
}
}
}

fn intel_dp_is_hdmi_2_1_sink(intel_dp: *mut intel_dp) -> bool {
let __v0: bool = drm_dp_is_branch((unsafe { (*intel_dp).dpcd.as_mut_ptr() as *mut u8 }) as *const u8) && intel_dp_has_hdmi_sink(intel_dp);
let __v1: bool = __v0 && intel_dp_hdmi_sink_max_frl(intel_dp) > 0;
if __v1 {
return true;
}
false
}

fn intel_dp_pcon_set_tmds_mode(intel_dp: *mut intel_dp) -> i32 {
let mut ret: i32 = 0;
let mut buf: u8 = 0;
// Set PCON source control mode
buf = ((buf as i32) | 8i32) as u8;
ret = drm_dp_dpcd_writeb(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 12378u32, buf) as i32;
if ret < 0 {
return ret;
}
// Set HDMI LINK ENABLE
buf = ((buf as i32) | 128i32) as u8;
ret = drm_dp_dpcd_writeb(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 12378u32, buf) as i32;
if ret < 0 {
return ret;
}
0
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_check_frl_training(intel_dp: *mut intel_dp) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
// Always go for FRL training if:
// -PCON supports SRC_CTL_MODE (VESA DP2.0-HDMI2.1 PCON Spec Draft-1 Sec-7)
// -sink is HDMI2.1
let __v1: bool = ((unsafe { *((*intel_dp).downstream_ports.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) }) as i32) & 32i32 == 0 || !intel_dp_is_hdmi_2_1_sink(intel_dp);
if __v1 || unsafe { (*intel_dp).frl.is_trained } {
return;
}
if intel_dp_pcon_start_frl_training(intel_dp) < 0 {
let mut ret: i32 = 0;
let mut mode: i32 = 0;
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 1u32, c"Couldn't set FRL mode, continuing with TMDS mode\n".as_ptr() as *const u8) };
ret = intel_dp_pcon_set_tmds_mode(intel_dp);
mode = unsafe { drm_dp_pcon_hdmi_link_mode(core::ptr::addr_of_mut!((*intel_dp).aux), core::ptr::null_mut::<u8>()) };
if ret < 0 || mode != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 1u32, c"Issue with PCON, cannot set TMDS mode\n".as_ptr() as *const u8) };
}
} else {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 1u32, c"FRL training Completed\n".as_ptr() as *const u8) };
}
}

fn intel_dp_pcon_dsc_enc_slice_height(crtc_state: *const intel_crtc_state) -> i32 {
let vactive: i32 = (unsafe { (*crtc_state).hw.adjusted_mode.vdisplay }) as i32;
unsafe { intel_hdmi_dsc_get_slice_height(vactive) }
}

fn intel_dp_pcon_dsc_enc_slices(intel_dp: *mut intel_dp, crtc_state: *const intel_crtc_state) -> i32 {
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let info: *const drm_display_info = (unsafe { core::ptr::addr_of_mut!((*connector).base.display_info) }) as *const drm_display_info;
let hdmi_throughput: i32 = unsafe { (*info).hdmi.dsc_cap.clk_per_slice };
let hdmi_max_slices: i32 = (unsafe { (*info).hdmi.dsc_cap.max_slices }) as i32;
let pcon_max_slices: i32 = unsafe { drm_dp_pcon_dsc_max_slices(((*intel_dp).pcon_dsc_dpcd.as_mut_ptr() as *mut u8) as *const u8) };
let pcon_max_slice_width: i32 = unsafe { drm_dp_pcon_dsc_max_slice_width(((*intel_dp).pcon_dsc_dpcd.as_mut_ptr() as *mut u8) as *const u8) };
unsafe { intel_hdmi_dsc_get_num_slices(crtc_state, pcon_max_slices, pcon_max_slice_width, hdmi_max_slices, hdmi_throughput) }
}

fn intel_dp_pcon_dsc_enc_bpp(intel_dp: *mut intel_dp, crtc_state: *const intel_crtc_state, num_slices: i32, slice_width: i32) -> i32 {
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let info: *const drm_display_info = (unsafe { core::ptr::addr_of_mut!((*connector).base.display_info) }) as *const drm_display_info;
let output_format: i32 = ((unsafe { (*crtc_state).output_format }) as u32) as i32;
let hdmi_all_bpp: bool = unsafe { (*info).hdmi.dsc_cap.all_bpp };
let pcon_fractional_bpp: i32 = unsafe { drm_dp_pcon_dsc_bpp_incr(((*intel_dp).pcon_dsc_dpcd.as_mut_ptr() as *mut u8) as *const u8) };
let hdmi_max_chunk_bytes: i32 = ((unsafe { (*info).hdmi.dsc_cap.total_chunk_kbytes }) as i32).wrapping_mul(1024);
unsafe { intel_hdmi_dsc_get_bpp(pcon_fractional_bpp, slice_width, num_slices, output_format, hdmi_all_bpp, hdmi_max_chunk_bytes) }
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_pcon_dsc_configure(intel_dp: *mut intel_dp, crtc_state: *const intel_crtc_state) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let mut pps_param: [u8; 6] = [0u8; 6];
let mut slice_height: i32 = 0;
let mut slice_width: i32 = 0;
let mut num_slices: i32 = 0;
let mut bits_per_pixel: i32 = 0;
let mut ret: i32 = 0;
let mut hdmi_is_dsc_1_2: bool = false;
if !intel_dp_is_hdmi_2_1_sink(intel_dp) {
return;
}
if connector.is_null() {
return;
}
let info: *const drm_display_info = (unsafe { core::ptr::addr_of_mut!((*connector).base.display_info) }) as *const drm_display_info;
hdmi_is_dsc_1_2 = unsafe { (*info).hdmi.dsc_cap.v_1p2 };
if !unsafe { drm_dp_pcon_enc_is_dsc_1_2(((*intel_dp).pcon_dsc_dpcd.as_mut_ptr() as *mut u8) as *const u8) } || !hdmi_is_dsc_1_2 {
return;
}
slice_height = intel_dp_pcon_dsc_enc_slice_height(crtc_state);
if slice_height == 0 {
return;
}
num_slices = intel_dp_pcon_dsc_enc_slices(intel_dp, crtc_state);
if num_slices == 0 {
return;
}
slice_width = ((unsafe { (*crtc_state).hw.adjusted_mode.hdisplay }) as i32).wrapping_add(num_slices).wrapping_sub(1) / num_slices;
bits_per_pixel = intel_dp_pcon_dsc_enc_bpp(intel_dp, crtc_state, num_slices, slice_width);
if bits_per_pixel == 0 {
return;
}
pps_param[0] = (slice_height & 255) as u8;
pps_param[1] = (slice_height >> 8i32) as u8;
pps_param[2] = (slice_width & 255) as u8;
pps_param[3] = (slice_width >> 8i32) as u8;
pps_param[4] = (bits_per_pixel & 255) as u8;
pps_param[5] = (bits_per_pixel >> 8i32 & 3) as u8;
ret = unsafe { drm_dp_pcon_pps_override_param(core::ptr::addr_of_mut!((*intel_dp).aux), pps_param.as_mut_ptr() as *mut u8) };
if ret < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Failed to set pcon DSC\n".as_ptr() as *const u8) };
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_configure_protocol_converter(intel_dp: *mut intel_dp, crtc_state: *const intel_crtc_state) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let mut ycbcr444_to_420: bool = false;
let mut rgb_to_ycbcr: bool = false;
let mut tmp: u8 = 0;
if ((unsafe { *((*intel_dp).dpcd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) < 19 {
return;
}
if !drm_dp_is_branch((unsafe { (*intel_dp).dpcd.as_mut_ptr() as *mut u8 }) as *const u8) {
return;
}
tmp = (if intel_dp_has_hdmi_sink(intel_dp) { 1i32 } else { 0i32 }) as u8;
if drm_dp_dpcd_writeb(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 12368u32, tmp) != 1 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Failed to %s protocol converter HDMI mode\n".as_ptr() as *const u8, str_enable_disable(intel_dp_has_hdmi_sink(intel_dp))) };
}
if ((unsafe { (*crtc_state).sink_format }) as u32) == 1 {
match (unsafe { (*crtc_state).output_format }) as u32 {
1 => {
}
2 => {
ycbcr444_to_420 = true;
}
0 => {
rgb_to_ycbcr = true;
ycbcr444_to_420 = true;
}
_ => {
let mut __v2: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"crtc_state->output_format".as_ptr() as *mut u8, ((*crtc_state).output_format as u32) as i64) };
let mut __v3: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v3 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v4: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_configure_protocol_converter_5, sym __slate_asm_string_intel_dp_configure_protocol_converter_6, const 4443 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v4 = bug;
unsafe { __v3.unwrap()(__v4, c"crtc_state->output_format".as_ptr() as *mut u8, ((*crtc_state).output_format as u32) as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
}
} else {
if ((unsafe { (*crtc_state).sink_format }) as u32) == 2 {
match (unsafe { (*crtc_state).output_format }) as u32 {
2 => {
}
0 => {
rgb_to_ycbcr = true;
}
_ => {
let mut __v9: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"crtc_state->output_format".as_ptr() as *mut u8, ((*crtc_state).output_format as u32) as i64) };
let mut __v10: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v10 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v11: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t1: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t1, sym __slate_asm_string_intel_dp_configure_protocol_converter_12, sym __slate_asm_string_intel_dp_configure_protocol_converter_13, const 4454 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t1 as u64) };
}
__v11 = bug;
unsafe { __v10.unwrap()(__v11, c"crtc_state->output_format".as_ptr() as *mut u8, ((*crtc_state).output_format as u32) as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v9 = (__ret_warn_on != 0) as i64;
}
}
}
}
tmp = (if ycbcr444_to_420 { 1i32 } else { 0i32 }) as u8;
if drm_dp_dpcd_writeb(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 12369u32, tmp) != 1 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Failed to %s protocol converter YCbCr 4:2:0 conversion mode\n".as_ptr() as *const u8, str_enable_disable((*intel_dp).dfp.ycbcr_444_to_420)) };
}
tmp = (if rgb_to_ycbcr { 32i32 } else { 0i32 }) as u8;
if (unsafe { drm_dp_pcon_convert_rgb_to_ycbcr(core::ptr::addr_of_mut!((*intel_dp).aux), tmp) }) < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Failed to %s protocol converter RGB->YCbCr conversion mode\n".as_ptr() as *const u8, str_enable_disable(tmp != 0)) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_866: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_867: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intel_dp_read_dprx_feature_enum(intel_dp: *mut intel_dp) -> u8 {
let mut dprx: u8 = 0;
drm_dp_dpcd_read_data(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 8720u32, core::ptr::addr_of_mut!(dprx) as *mut (), core::mem::size_of::<u8>() as u64);
dprx
}

fn intel_dp_get_colorimetry_status(dprx: u8) -> bool {
(dprx as i32) & 8i32 != 0
}

fn intel_dp_get_sst_split_sdp_status(dprx: u8) -> bool {
(dprx as i32) & 2i32 != 0
}

fn intel_dp_read_dsc_dpcd(aux: *mut drm_dp_aux, dsc_dpcd: *mut u8) -> i32 {
let ret: i32 = drm_dp_dpcd_read_data(aux, 96u32, dsc_dpcd as *mut (), 16u64);
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"Could not read DSC DPCD register 0x%x Error: %pe\n".as_ptr() as *const u8, 96i32, ERR_PTR(ret as i64)) };
return ret;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"DSC DPCD: %*ph\n".as_ptr() as *const u8, 16i32, dsc_dpcd) };
0
}

fn init_dsc_overall_throughput_limits(connector: *mut intel_connector, is_branch: bool) {
let mut branch_caps: [u8; 3] = [0u8; 3];
let mut line_width: i32 = 0;
unsafe {
(*connector).dp.dsc_branch_caps.overall_throughput.rgb_yuv444 = 2147483647i32;
}
unsafe {
(*connector).dp.dsc_branch_caps.overall_throughput.yuv422_420 = 2147483647i32;
}
unsafe {
(*connector).dp.dsc_branch_caps.max_line_width = 2147483647i32;
}
if !is_branch {
return;
}
if drm_dp_dpcd_read_data(unsafe { (*connector).dp.dsc_decompression_aux }, 160u32, branch_caps.as_mut_ptr() as *mut (), core::mem::size_of::<[u8; 3]>() as u64) != 0 {
return;
}
let __v0: i32 = unsafe { drm_dp_dsc_branch_max_overall_throughput(branch_caps.as_mut_ptr() as *const u8, true) };
unsafe {
(*connector).dp.dsc_branch_caps.overall_throughput.rgb_yuv444 = if __v0 != 0 { __v0 } else { 2147483647i32 };
}
let __v1: i32 = unsafe { drm_dp_dsc_branch_max_overall_throughput(branch_caps.as_mut_ptr() as *const u8, false) };
unsafe {
(*connector).dp.dsc_branch_caps.overall_throughput.yuv422_420 = if __v1 != 0 { __v1 } else { 2147483647i32 };
}
line_width = unsafe { drm_dp_dsc_branch_max_line_width(branch_caps.as_mut_ptr() as *const u8) };
unsafe {
(*connector).dp.dsc_branch_caps.max_line_width = if line_width > 0 { line_width } else { 2147483647i32 };
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_get_dsc_sink_cap(dpcd_rev: u8, desc: *const drm_dp_desc, is_branch: bool, connector: *mut intel_connector) {
let display: *mut intel_display = to_intel_display!(connector);
// Clear the cached register set to avoid using stale values
// for the sinks that do not support DSC.
unsafe { memset(((*connector).dp.dsc_dpcd.as_mut_ptr() as *mut u8) as *mut (), 0i32, core::mem::size_of::<[u8; 16]>() as u64) };
// Clear fec_capable to avoid using stale values
unsafe {
(*connector).dp.fec_capability = 0;
}
unsafe { memset(core::ptr::addr_of_mut!((*connector).dp.dsc_branch_caps) as *mut (), 0i32, core::mem::size_of::<__SlateRecord_intel_display_types_h_568>() as u64) };
unsafe {
(*connector).dp.__slate_bits_0.__set_dsc_throughput_quirk(0u8);
}
if (dpcd_rev as i32) < 20 {
return;
}
if intel_dp_read_dsc_dpcd(unsafe { (*connector).dp.dsc_decompression_aux }, unsafe { (*connector).dp.dsc_dpcd.as_mut_ptr() as *mut u8 }) < 0 {
return;
}
if drm_dp_dpcd_readb(unsafe { (*connector).dp.dsc_decompression_aux }, 144u32, unsafe { core::ptr::addr_of_mut!((*connector).dp.fec_capability) }) < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Could not read FEC DPCD register\n".as_ptr() as *const u8) };
return;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"FEC CAPABILITY: %x\n".as_ptr() as *const u8, (*connector).dp.fec_capability as i32) };
if ((unsafe { *((*connector).dp.dsc_dpcd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) & 1i32 == 0 {
return;
}
init_dsc_overall_throughput_limits(connector, is_branch);
// TODO: Move the HW rev check as well to the DRM core quirk table if
// that's required after clarifying the list of affected devices.
if drm_dp_has_quirk(desc, 6u32) && ((unsafe { (*desc).ident.hw_rev }) as i32) == 16 {
unsafe {
(*connector).dp.__slate_bits_0.__set_dsc_throughput_quirk(1u8);
}
}
}

fn intel_edp_get_dsc_sink_cap(edp_dpcd_rev: u8, connector: *mut intel_connector) {
if (edp_dpcd_rev as i32) < 3 {
return;
}
if intel_dp_read_dsc_dpcd(unsafe { (*connector).dp.dsc_decompression_aux }, unsafe { (*connector).dp.dsc_dpcd.as_mut_ptr() as *mut u8 }) < 0 {
return;
}
if ((unsafe { *((*connector).dp.dsc_dpcd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) & 1i32 != 0 {
init_dsc_overall_throughput_limits(connector, false);
}
}

fn intel_dp_detect_dsc_caps(intel_dp: *mut intel_dp, connector: *mut intel_connector) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
// Read DP Sink DSC Cap DPCD regs for DP v1.4
if !HAS_DSC!(display) {
return;
}
if intel_dp_is_edp(intel_dp) {
intel_edp_get_dsc_sink_cap(unsafe { *((*intel_dp).edp_dpcd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }, connector);
} else {
intel_dp_get_dsc_sink_cap(unsafe { *((*intel_dp).dpcd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }, (unsafe { core::ptr::addr_of_mut!((*intel_dp).desc) }) as *const drm_dp_desc, drm_dp_is_branch((unsafe { (*intel_dp).dpcd.as_mut_ptr() as *mut u8 }) as *const u8), connector);
}
}

fn intel_edp_mso_mode_fixup(connector: *mut intel_connector, mode: *mut drm_display_mode) {
let display: *mut intel_display = to_intel_display!(connector);
let intel_dp: *mut intel_dp = intel_attached_dp(connector);
let n: i32 = unsafe { (*intel_dp).mso_link_count };
let overlap: i32 = unsafe { (*intel_dp).mso_pixel_overlap };
if mode.is_null() || n == 0 {
return;
}
unsafe {
(*mode).hdisplay = ((*mode).hdisplay as i32).wrapping_sub(overlap).wrapping_mul(n) as u16;
}
unsafe {
(*mode).hsync_start = ((*mode).hsync_start as i32).wrapping_sub(overlap).wrapping_mul(n) as u16;
}
unsafe {
(*mode).hsync_end = ((*mode).hsync_end as i32).wrapping_sub(overlap).wrapping_mul(n) as u16;
}
unsafe {
(*mode).htotal = ((*mode).htotal as i32).wrapping_sub(overlap).wrapping_mul(n) as u16;
}
let __v2: i32 = unsafe { (*mode).clock };
unsafe {
(*mode).clock = __v2.wrapping_mul(n);
}
unsafe { drm_mode_set_name(mode) };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] using generated MSO mode: \"%s\": %d %d %d %d %d %d %d %d %d %d 0x%x 0x%x\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name, (*mode).name.as_mut_ptr() as *mut u8, drm_mode_vrefresh(mode as *const drm_display_mode), (*mode).clock, (*mode).hdisplay as i32, (*mode).hsync_start as i32, (*mode).hsync_end as i32, (*mode).htotal as i32, (*mode).vdisplay as i32, (*mode).vsync_start as i32, (*mode).vsync_end as i32, (*mode).vtotal as i32, (*mode).r#type as i32, (*mode).flags) };
}

#[unsafe(no_mangle)]
extern "C" fn intel_edp_fixup_vbt_bpp(encoder: *mut intel_encoder, pipe_bpp: i32) {
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
if (unsafe { (*connector).panel.vbt.edp.bpp }) != 0 && pipe_bpp > unsafe { (*connector).panel.vbt.edp.bpp } {
// This is a big fat ugly hack.
//
// Some machines in UEFI boot mode provide us a VBT that has 18
// bpp and 1.62 GHz link bandwidth for eDP, which for reasons
// unknown we fail to light up. Yet the same BIOS boots up with
// 24 bpp and 2.7 GHz link. Use the same bpp as the BIOS uses as
// max, not what it tells us to use.
//
// Note: This will still be broken if the eDP panel is not lit
// up by the BIOS, and thus we can't get the mode at module
// load.
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"pipe has %d bpp for eDP panel, overriding BIOS-provided max %d bpp\n".as_ptr() as *const u8, pipe_bpp, (*connector).panel.vbt.edp.bpp) };
unsafe {
(*connector).panel.vbt.edp.bpp = pipe_bpp;
}
}
}

fn intel_edp_mso_init(intel_dp: *mut intel_dp) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let info: *mut drm_display_info = unsafe { core::ptr::addr_of_mut!((*connector).base.display_info) };
let mut mso: u8 = 0;
if ((unsafe { *((*intel_dp).edp_dpcd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) < 3 {
return;
}
if drm_dp_dpcd_readb(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 1956u32, core::ptr::addr_of_mut!(mso)) != 1 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* Failed to read MSO cap\n".as_ptr() as *const u8) };
return;
}
// Valid configurations are SST or MSO 2x1, 2x2, 4x1
mso = ((mso as i32) & 7i32) as u8;
let __v4: bool = (mso as i32) % 2 != 0 || (mso as i32) > (drm_dp_max_lane_count((unsafe { (*intel_dp).dpcd.as_mut_ptr() as *mut u8 }) as *const u8) as i32);
if __v4 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* Invalid MSO link count cap %u\n".as_ptr() as *const u8, mso as i32) };
mso = 0;
}
if mso != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Sink MSO %ux%u configuration, pixel overlap %u\n".as_ptr() as *const u8, mso as i32, (drm_dp_max_lane_count(((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8) as i32) / (mso as i32), (*info).mso_pixel_overlap as i32) };
if !HAS_MSO!(display) {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* No source MSO support, disabling\n".as_ptr() as *const u8) };
mso = 0;
}
}
unsafe {
(*intel_dp).mso_link_count = mso as i32;
}
unsafe {
(*intel_dp).mso_pixel_overlap = if mso != 0 { (*info).mso_pixel_overlap as i32 } else { 0 };
}
}

fn intel_edp_set_data_override_rates(intel_dp: *mut intel_dp) {
let encoder: *mut intel_encoder = unsafe { core::ptr::addr_of_mut!((*dp_to_dig_port(intel_dp)).base) };
let sink_rates: *mut i32 = unsafe { (*intel_dp).sink_rates.as_mut_ptr() as *mut i32 };
let mut i: i32 = 0;
let mut count: i32 = 0;
while i < unsafe { (*intel_dp).num_sink_rates } {
if !unsafe { intel_bios_encoder_reject_edp_rate((*encoder).devdata, *((*intel_dp).sink_rates.as_mut_ptr() as *mut i32).wrapping_offset(i as isize)) } {
let __v1: i32 = count;
count = __v1.wrapping_add(1);
unsafe {
*sink_rates.wrapping_offset(__v1 as isize) = *((*intel_dp).sink_rates.as_mut_ptr() as *mut i32).wrapping_offset(i as isize);
}
}
i = i.wrapping_add(1);
}
unsafe {
(*intel_dp).num_sink_rates = count;
}
}

fn intel_edp_set_sink_rates(intel_dp: *mut intel_dp) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
unsafe {
(*intel_dp).num_sink_rates = 0;
}
if ((unsafe { *((*intel_dp).edp_dpcd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) >= 3 {
let mut sink_rates: __SlateAlign16<[u16; 8]> = __SlateAlign16([0u16; 8]);
let ret: i32 = drm_dp_dpcd_read_data(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 16u32, sink_rates.0.as_mut_ptr() as *mut (), core::mem::size_of::<[u16; 8]>() as u64);
let mut i: i32 = 0;
if ret < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Unable to read eDP supported link rates, using default rates\n".as_ptr() as *const u8) };
unsafe { memset(sink_rates.0.as_mut_ptr() as *mut (), 0i32, core::mem::size_of::<[u16; 8]>() as u64) };
}
while (i as u64) < ARRAY_SIZE__94!() {
let mut rate: i32 = 0;
// Value read multiplied by 200kHz gives the per-lane
// link rate in kHz. The source rates are, however,
// stored in terms of LS_Clk kHz. The full conversion
// back to symbols is
// (val * 200kHz)*(8/10 ch. encoding)*(1/8 bit to Byte)
rate = (sink_rates.0[i as usize] as i32).wrapping_mul(200) / 10;
if rate == 0 {
break;
}
// Some platforms cannot reliably drive HBR3 rates due to PHY limitations,
// even if the sink advertises support. Reject any sink rates above HBR2 on
// the known machines for stable output.
let __v2: bool = rate > 540000 && unsafe { intel_has_quirk(display, 7u32) };
if __v2 {
break;
}
unsafe {
*((*intel_dp).sink_rates.as_mut_ptr() as *mut i32).wrapping_offset(i as isize) = rate;
}
i = i.wrapping_add(1);
}
unsafe {
(*intel_dp).num_sink_rates = i;
}
}
// Use DP_LINK_RATE_SET if DP_SUPPORTED_LINK_RATES are available,
// default to DP_MAX_LINK_RATE and DP_LINK_BW_SET otherwise.
if (unsafe { (*intel_dp).num_sink_rates }) != 0 {
unsafe {
(*intel_dp).use_rate_select = true;
}
} else {
intel_dp_set_sink_rates(intel_dp);
}
intel_edp_set_data_override_rates(intel_dp);
}

fn intel_edp_init_dpcd(intel_dp: *mut intel_dp, connector: *mut intel_connector) -> bool {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let mut ret: i32 = 0;
let mut dprx: u8 = 0;
// this function is meant to be called only once
let mut __v1: i64;
let __ret_warn_on: i32 = (((unsafe { *((*intel_dp).dpcd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(intel_dp->dpcd[0x000] != 0)".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_edp_init_dpcd_4, sym __slate_asm_string_intel_edp_init_dpcd_5, const 4796 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(intel_dp->dpcd[0x000] != 0)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if (unsafe { drm_dp_read_dpcd_caps(core::ptr::addr_of_mut!((*intel_dp).aux), (*intel_dp).dpcd.as_mut_ptr() as *mut u8) }) != 0 {
return false;
}
unsafe { drm_dp_read_desc(core::ptr::addr_of_mut!((*intel_dp).aux), core::ptr::addr_of_mut!((*intel_dp).desc), drm_dp_is_branch(((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8)) };
unsafe { intel_init_dpcd_quirks(intel_dp, core::ptr::addr_of_mut!((*intel_dp).desc.ident) as *const drm_dp_dpcd_ident) };
dprx = intel_dp_read_dprx_feature_enum(intel_dp);
unsafe {
(*intel_dp).colorimetry_support = intel_dp_get_colorimetry_status(dprx);
}
unsafe {
(*intel_dp).sst_split_sdp_support = intel_dp_get_sst_split_sdp_status(dprx);
}
// Read the eDP display control registers.
//
// Do this independent of DP_DPCD_DISPLAY_CONTROL_CAPABLE bit in
// DP_EDP_CONFIGURATION_CAP, because some buggy displays do not have it
// set, but require eDP 1.4+ detection (e.g. for supported link rates
// method). The display control registers should read zero if they're
// not supported anyway.
if ((unsafe { drm_dp_dpcd_read(core::ptr::addr_of_mut!((*intel_dp).aux), 1792u32, ((*intel_dp).edp_dpcd.as_mut_ptr() as *mut u8) as *mut (), core::mem::size_of::<[u8; 5]>() as u64) }) as u64) == (core::mem::size_of::<[u8; 5]>() as u64) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"eDP DPCD: %*ph\n".as_ptr() as *const u8, core::mem::size_of::<[u8; 5]>() as i32, (*intel_dp).edp_dpcd.as_mut_ptr() as *mut u8) };
unsafe {
(*intel_dp).use_max_params = (*((*intel_dp).edp_dpcd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32) < 3;
}
}
// If needed, program our source OUI so we can make various Intel-specific AUX services
// available (such as HDR backlight controls)
intel_dp_init_source_oui(intel_dp);
// Read the ALPM DPCD caps
ret = drm_dp_dpcd_read_byte(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 46u32, unsafe { core::ptr::addr_of_mut!((*intel_dp).alpm_dpcd) });
if ret < 0 {
return false;
}
// This has to be called after intel_dp->edp_dpcd is filled, PSR checks
// for SET_POWER_CAPABLE bit in intel_dp->edp_dpcd[1]
unsafe { intel_psr_init_dpcd(intel_dp, connector) };
intel_edp_set_sink_rates(intel_dp);
intel_dp_set_max_sink_lane_count(intel_dp);
// Read the eDP DSC DPCD registers
intel_dp_detect_dsc_caps(intel_dp, connector);
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_868: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intel_dp_has_sink_count(intel_dp: *mut intel_dp) -> bool {
if unsafe { (*intel_dp).attached_connector }.is_null() {
return false;
}
unsafe { drm_dp_read_sink_count_cap(core::ptr::addr_of_mut!((*(*intel_dp).attached_connector).base), ((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8, core::ptr::addr_of_mut!((*intel_dp).desc) as *const drm_dp_desc) }
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_update_sink_caps(intel_dp: *mut intel_dp) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
intel_dp_set_sink_rates(intel_dp);
intel_dp_set_max_sink_lane_count(intel_dp);
// Handle unexpected sink cap changes, or a race between setting
// the deferred link params flag in the HPD IRQ handler and
// clearing the flag during connector detect.
if intel_dp_set_common_link_params(intel_dp) && !unsafe { (*intel_dp).reset_link_params } {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"DPRX capabilities changed before long HPD or RX_CAP_CHANGED signal\n".as_ptr() as *const u8) };
unsafe {
(*intel_dp).reset_link_params = true;
}
}
}

fn intel_dp_get_dpcd(intel_dp: *mut intel_dp) -> bool {
let mut ret: i32 = 0;
if (unsafe { intel_dp_init_lttpr_and_dprx_caps(intel_dp) }) < 0 {
return false;
}
// Don't clobber cached eDP rates. Also skip re-reading
// the OUI/ID since we know it won't change.
if !intel_dp_is_edp(intel_dp) {
let mut dprx: u8 = 0;
unsafe { drm_dp_read_desc(core::ptr::addr_of_mut!((*intel_dp).aux), core::ptr::addr_of_mut!((*intel_dp).desc), drm_dp_is_branch(((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8)) };
unsafe { intel_init_dpcd_quirks(intel_dp, core::ptr::addr_of_mut!((*intel_dp).desc.ident) as *const drm_dp_dpcd_ident) };
dprx = intel_dp_read_dprx_feature_enum(intel_dp);
unsafe {
(*intel_dp).colorimetry_support = intel_dp_get_colorimetry_status(dprx);
}
unsafe {
(*intel_dp).sst_split_sdp_support = intel_dp_get_sst_split_sdp_status(dprx);
}
intel_dp_update_sink_caps(intel_dp);
}
if intel_dp_has_sink_count(intel_dp) {
ret = unsafe { drm_dp_read_sink_count(core::ptr::addr_of_mut!((*intel_dp).aux)) };
if ret < 0 {
return false;
}
// Sink count can change between short pulse hpd hence
// a member variable in intel_dp will track any changes
// between short pulse interrupts.
unsafe {
(*intel_dp).sink_count = ret as u8;
}
// SINK_COUNT == 0 and DOWNSTREAM_PORT_PRESENT == 1 implies that
// a dongle is present but no display. Unless we require to know
// if a dongle is present or not, we don't need to update
// downstream port information. So, an early return here saves
// time from performing other operations which are not required.
if (unsafe { (*intel_dp).sink_count }) == 0 {
return false;
}
}
(unsafe { drm_dp_read_downstream_info(core::ptr::addr_of_mut!((*intel_dp).aux), ((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8, (*intel_dp).downstream_ports.as_mut_ptr() as *mut u8) }) == 0
}

fn intel_dp_mst_mode_str(mst_mode: u32) -> *const u8 {
if (mst_mode as u32) == 1 {
return c"MST".as_ptr() as *const u8;
} else {
if (mst_mode as u32) == 2 {
return c"SST w/ sideband messaging".as_ptr() as *const u8;
} else {
return c"SST".as_ptr() as *const u8;
}
}
}

fn intel_dp_mst_mode_choose(intel_dp: *mut intel_dp, sink_mst_mode: u32) -> u32 {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
if !unsafe { (*display).params.enable_dp_mst } {
return 0;
}
if !unsafe { intel_dp_mst_source_support(intel_dp) } {
return 0;
}
if (sink_mst_mode as u32) == 2 && ((unsafe { *((*intel_dp).dpcd.as_mut_ptr() as *mut u8).wrapping_offset(6 as isize) }) as i32) & 2i32 == 0 {
return 0;
}
sink_mst_mode
}

fn intel_dp_mst_detect(intel_dp: *mut intel_dp) -> u32 {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let encoder: *mut intel_encoder = unsafe { core::ptr::addr_of_mut!((*dp_to_dig_port(intel_dp)).base) };
let sink_mst_mode: u32 = unsafe { drm_dp_read_mst_cap(core::ptr::addr_of_mut!((*intel_dp).aux), ((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8) };
let mst_detect: u32 = intel_dp_mst_mode_choose(intel_dp, sink_mst_mode);
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[ENCODER:%d:%s] MST support: port: %s, sink: %s, modparam: %s -> enable: %s\n".as_ptr() as *const u8, (*encoder).base.base.id, (*encoder).base.name, str_yes_no(intel_dp_mst_source_support(intel_dp)), intel_dp_mst_mode_str(sink_mst_mode), str_yes_no((*display).params.enable_dp_mst), intel_dp_mst_mode_str(mst_detect)) };
mst_detect
}

fn intel_dp_mst_configure(intel_dp: *mut intel_dp) {
if !unsafe { intel_dp_mst_source_support(intel_dp) } {
return;
}
unsafe {
(*intel_dp).is_mst = ((*intel_dp).mst_detect as u32) != 0;
}
if unsafe { (*intel_dp).is_mst } {
unsafe { intel_dp_mst_prepare_probe(intel_dp) };
}
unsafe { drm_dp_mst_topology_mgr_set_mst(core::ptr::addr_of_mut!((*intel_dp).mst.mgr), (*intel_dp).is_mst) };
// Avoid stale info on the next detect cycle.
unsafe {
(*intel_dp).mst_detect = 0;
}
}

fn intel_dp_mst_disconnect(intel_dp: *mut intel_dp) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
if !unsafe { (*intel_dp).is_mst } {
return;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"MST device may have disappeared %d vs %d\n".as_ptr() as *const u8, (*intel_dp).is_mst as i32, (*intel_dp).mst.mgr.__slate_bits_0.__get_mst_state() as i32) };
unsafe {
(*intel_dp).is_mst = false;
}
unsafe { drm_dp_mst_topology_mgr_set_mst(core::ptr::addr_of_mut!((*intel_dp).mst.mgr), (*intel_dp).is_mst) };
}

fn intel_dp_get_sink_irq_esi(intel_dp: *mut intel_dp, esi: *mut u8) -> bool {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
// Display WA for HSD #13013007775: mtl/arl/lnl
// Read the sink count and link service IRQ registers in separate
// transactions to prevent disconnecting the sink on a TBT link
// inadvertently.
if IS_DISPLAY_VER!(display, 14i32, 20i32) && ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_battlemage() }) as i32) == 0 {
if (unsafe { drm_dp_dpcd_read(core::ptr::addr_of_mut!((*intel_dp).aux), 8194u32, esi as *mut (), 3u64) }) != 3 {
return false;
}
// DP_SINK_COUNT_ESI + 3 == DP_LINK_SERVICE_IRQ_VECTOR_ESI0
return drm_dp_dpcd_readb(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 8197u32, esi.wrapping_offset(3 as isize)) == 1;
}
(unsafe { drm_dp_dpcd_read(core::ptr::addr_of_mut!((*intel_dp).aux), 8194u32, esi as *mut (), 4u64) }) == 4
}

fn intel_dp_ack_sink_irq_esi(intel_dp: *mut intel_dp, esi: *mut u8) -> bool {
for retry in 0i32..3i32 {
if (unsafe { drm_dp_dpcd_write(core::ptr::addr_of_mut!((*intel_dp).aux), 8195u32, esi.wrapping_offset(1 as isize) as *mut (), 3u64) }) == 3 {
return true;
}
}
false
}

/// Return %true if reading the ESI vector succeeded, %false otherwise.
fn intel_dp_get_sink_irq_esi_sst(intel_dp: *mut intel_dp, esi: *mut u8) -> bool {
unsafe { memset(esi as *mut (), 0i32, 4u64) };
// TODO: For DP_DPCD_REV >= 0x12 read
// DP_SINK_COUNT_ESI and DP_DEVICE_SERVICE_IRQ_VECTOR_ESI0.
if drm_dp_dpcd_read_data(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 512u32, esi as *mut (), 2u64) != 0 {
return false;
}
if ((unsafe { *((*intel_dp).dpcd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) < 18 {
return true;
}
// TODO: Read DP_DEVICE_SERVICE_IRQ_VECTOR_ESI1 as well
if drm_dp_dpcd_read_byte(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 8197u32, esi.wrapping_offset(3 as isize)) != 0 {
return false;
}
true
}

/// Return %true if acking the ESI vector IRQ events succeeded, %false otherwise.
fn intel_dp_ack_sink_irq_esi_sst(intel_dp: *mut intel_dp, esi: *mut u8) -> bool {
// TODO: For DP_DPCD_REV >= 0x12 write
// DP_DEVICE_SERVICE_IRQ_VECTOR_ESI0
if drm_dp_dpcd_write_byte(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 513u32, unsafe { *esi.wrapping_offset(1 as isize) }) != 0 {
return false;
}
if ((unsafe { *((*intel_dp).dpcd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) < 18 {
return true;
}
// TODO: Read DP_DEVICE_SERVICE_IRQ_VECTOR_ESI1 as well
if drm_dp_dpcd_write_byte(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 8197u32, unsafe { *esi.wrapping_offset(3 as isize) }) != 0 {
return false;
}
true
}

/// Return %true if reading the ESI vector and acking the ESI IRQ events succeeded,
/// %false otherwise.
fn intel_dp_get_and_ack_sink_irq_esi_sst(intel_dp: *mut intel_dp, esi: *mut u8) -> bool {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let encoder: *mut intel_encoder = unsafe { core::ptr::addr_of_mut!((*dp_to_dig_port(intel_dp)).base) };
if !intel_dp_get_sink_irq_esi_sst(intel_dp, esi) {
return false;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s][ENCODER:%d:%s] DPRX ESI: %4ph\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name, (*encoder).base.base.id, (*encoder).base.name, esi) };
let __v1: *mut u8 = esi.wrapping_offset(1 as isize);
unsafe {
*__v1 = ((*__v1 as i32) & 70i32) as u8;
}
let __v4: *mut u8 = esi.wrapping_offset(3 as isize);
unsafe {
*__v4 = ((*__v4 as i32) & 59i32) as u8;
}
if mem_is_zero(esi.wrapping_offset(1 as isize) as *const (), 3u64) {
return true;
}
if !intel_dp_ack_sink_irq_esi_sst(intel_dp, esi) {
return false;
}
true
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_needs_vsc_colorimetry(crtc_state: *const intel_crtc_state, conn_state: *const drm_connector_state) -> bool {
let intel_dp: *mut intel_dp = enc_to_intel_dp(to_intel_encoder!(unsafe { (*conn_state).best_encoder }));
// As per DP 1.4a spec section 2.2.4.3 [MSA Field for Indication
// of Color Encoding Format and Content Color Gamut], in order to
// send YCBCR 420 or HDR BT.2020 signals we should use DP VSC SDP.
// Only signal this when the sink advertises VSC SDP colorimetry
// support.
if !unsafe { (*intel_dp).colorimetry_support } {
return false;
}
if ((unsafe { (*crtc_state).output_format }) as u32) == 1 {
return true;
}
match (unsafe { (*conn_state).colorspace }) as u32 {
5 | 6 | 8..=10 => {
return true;
}
_ => {
}
}
false
}

fn intel_dp_as_sdp_pack(as_sdp: *const drm_dp_as_sdp, sdp: *mut dp_sdp, size: u64) -> i64 {
let length: u64 = core::mem::size_of::<dp_sdp>() as u64;
if size < length {
return 28i32.wrapping_neg() as i64;
}
unsafe { memset(sdp as *mut (), 0i32, size) };
// Prepare AS (Adaptive Sync) SDP Header
unsafe {
(*sdp).sdp_header.HB0 = 0;
}
unsafe {
(*sdp).sdp_header.HB1 = (*as_sdp).sdp_type;
}
unsafe {
(*sdp).sdp_header.HB2 = (*as_sdp).revision;
}
unsafe {
(*sdp).sdp_header.HB3 = (*as_sdp).length;
}
// Fill AS (Adaptive Sync) SDP Payload
unsafe {
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = ((*as_sdp).mode as u32) as u8;
}
unsafe {
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) = ((*as_sdp).vtotal & 255) as u8;
}
unsafe {
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) = ((*as_sdp).vtotal >> 8i32 & 255) as u8;
}
unsafe {
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) = ((*as_sdp).target_rr & 255) as u8;
}
unsafe {
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(4 as isize) = ((*as_sdp).target_rr >> 8i32 & 3) as u8;
}
if unsafe { (*as_sdp).target_rr_divider } {
let __v0: *mut u8 = unsafe { (*sdp).db.as_mut_ptr() as *mut u8 }.wrapping_offset(4 as isize);
unsafe {
*__v0 = ((*__v0 as i32) | 32) as u8;
}
}
unsafe {
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(7 as isize) = ((*as_sdp).coasting_vtotal & 255) as u8;
}
unsafe {
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(8 as isize) = ((*as_sdp).coasting_vtotal >> 8i32 & 255) as u8;
}
length as i64
}

fn intel_dp_hdr_metadata_infoframe_sdp_pack(display: *mut intel_display, drm_infoframe: *const hdmi_drm_infoframe, sdp: *mut dp_sdp, size: u64) -> i64 {
let length: u64 = core::mem::size_of::<dp_sdp>() as u64;
let infoframe_size: i32 = 30i32;
let mut buf: __SlateAlign16<[u8; 30]> = __SlateAlign16([0u8; 30]);
let mut len: i64 = 0;
if size < length {
return 28i32.wrapping_neg() as i64;
}
unsafe { memset(sdp as *mut (), 0i32, size) };
len = unsafe { hdmi_drm_infoframe_pack_only(drm_infoframe, buf.0.as_mut_ptr() as *mut (), core::mem::size_of::<[u8; 30]>() as u64) };
if len < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"buffer size is smaller than hdr metadata infoframe\n".as_ptr() as *const u8) };
return 28i32.wrapping_neg() as i64;
}
if len != (infoframe_size as i64) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"wrong static hdr metadata size\n".as_ptr() as *const u8) };
return 28i32.wrapping_neg() as i64;
}
// Set up the infoframe sdp packet for HDR static metadata.
// Prepare VSC Header for SU as per DP 1.4a spec,
// Table 2-100 and Table 2-101
// Secondary-Data Packet ID, 00h for non-Audio INFOFRAME
unsafe {
(*sdp).sdp_header.HB0 = 0;
}
// Packet Type 80h + Non-audio INFOFRAME Type value
// HDMI_INFOFRAME_TYPE_DRM: 0x87
// - 80h + Non-audio INFOFRAME Type value
// - InfoFrame Type: 0x07
//    [CTA-861-G Table-42 Dynamic Range and Mastering InfoFrame]
unsafe {
(*sdp).sdp_header.HB1 = ((*drm_infoframe).r#type as u32) as u8;
}
// Least Significant Eight Bits of (Data Byte Count – 1)
// infoframe_size - 1
unsafe {
(*sdp).sdp_header.HB2 = 29;
}
// INFOFRAME SDP Version Number
unsafe {
(*sdp).sdp_header.HB3 = 76u8;
}
// CTA Header Byte 2 (INFOFRAME Version Number)
unsafe {
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = (*drm_infoframe).version;
}
// CTA Header Byte 3 (Length of INFOFRAME): HDMI_DRM_INFOFRAME_SIZE
unsafe {
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) = (*drm_infoframe).length;
}
// Copy HDMI_DRM_INFOFRAME_SIZE size from a buffer after
// HDMI_INFOFRAME_HEADER_SIZE
loop {
if ((const { core::mem::size_of::<[u8; 32]>() }) as u64) < 28u64 {
unsafe { __compiletime_assert_869() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe { memcpy(((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) as *mut (), (buf.0.as_mut_ptr() as *mut u8).wrapping_offset(4 as isize) as *const (), 26u64) };
// Size of DP infoframe sdp packet for HDR static metadata consists of
// - DP SDP Header(struct dp_sdp_header): 4 bytes
// - Two Data Blocks: 2 bytes
//    CTA Header Byte2 (INFOFRAME Version Number)
//    CTA Header Byte3 (Length of INFOFRAME)
// - HDMI_DRM_INFOFRAME_SIZE: 26 bytes
//
// Prior to GEN11's GMP register size is identical to DP HDR static metadata
// infoframe size. But GEN11+ has larger than that size, write_infoframe
// will pad rest of the size.
(core::mem::size_of::<dp_sdp_header>() as u64).wrapping_add(2).wrapping_add(26) as i64
}

fn intel_write_dp_sdp(encoder: *mut intel_encoder, crtc_state: *const intel_crtc_state, r#type: u32) {
let display: *mut intel_display = to_intel_display!(encoder);
let dig_port: *mut intel_digital_port = enc_to_dig_port(encoder);
let mut sdp: dp_sdp = {
let mut __t0: dp_sdp = unsafe { core::mem::zeroed() };
__t0
};
let mut len: i64 = 0;
if (unsafe { (*crtc_state).infoframes.enable }) & unsafe { intel_hdmi_infoframe_enable(r#type) } == 0 {
return;
}
match r#type {
7 => {
len = unsafe { drm_dp_vsc_sdp_pack(core::ptr::addr_of!((*crtc_state).infoframes.vsc), core::ptr::addr_of_mut!(sdp)) };
}
10 => {
len = intel_dp_hdr_metadata_infoframe_sdp_pack(display, unsafe { core::ptr::addr_of!((*crtc_state).infoframes.drm.drm) }, core::ptr::addr_of_mut!(sdp), core::mem::size_of::<dp_sdp>() as u64);
}
34 => {
len = intel_dp_as_sdp_pack(unsafe { core::ptr::addr_of!((*crtc_state).infoframes.as_sdp) }, core::ptr::addr_of_mut!(sdp), core::mem::size_of::<dp_sdp>() as u64);
}
_ => {
let mut __v2: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"type".as_ptr() as *mut u8, r#type as i64) };
let mut __v3: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v3 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v4: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t1: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t1, sym __slate_asm_string_intel_write_dp_sdp_5, sym __slate_asm_string_intel_write_dp_sdp_6, const 5329 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t1 as u64) };
}
__v4 = bug;
unsafe { __v3.unwrap()(__v4, c"type".as_ptr() as *mut u8, r#type as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
return;
}
}
let mut __v8: i64;
let __ret_warn_on: i32 = (len < 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(len < 0)".as_ptr() as *mut u8) };
let mut __v9: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v9 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v10: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t2: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t2, sym __slate_asm_string_intel_write_dp_sdp_11, sym __slate_asm_string_intel_write_dp_sdp_12, const 5333 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t2 as u64) };
}
__v10 = bug;
unsafe { __v9.unwrap()(__v10, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(len < 0)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v8 = (__ret_warn_on != 0) as i64;
if __v8 != 0 {
return;
}
unsafe { unsafe { (*dig_port).write_infoframe }.unwrap()(encoder, crtc_state, r#type, core::ptr::addr_of_mut!(sdp) as *const (), len) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_870: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_871: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn intel_dp_set_infoframes(encoder: *mut intel_encoder, enable: bool, crtc_state: *const intel_crtc_state, conn_state: *const drm_connector_state) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let reg: __SlateRecord_i915_reg_defs_h_14 = unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_TRANS_OFFSET!(display, unsafe { (*crtc_state).cpu_transcoder }).wrapping_add(393728) };
core::ptr::addr_of_mut!(__slate_compound_1)
} };
let mut dip_enable: u32 = 268505361u32;
if HAS_AS_SDP!(display) {
let __v2: u32 = dip_enable;
dip_enable = __v2 | REG_BIT__1211!(23i32);
}
let mut val: u32 = intel_de_read(display, reg) & !dip_enable;
// TODO: Sanitize DSC enabling wrt. intel_dsc_dp_pps_write().
if !enable && HAS_DSC!(display) {
let __v5: u32 = val & 4278190079u32;
val = __v5;
}
// This routine disables VSC DIP if the function is called
// to disable SDP or if it does not have PSR
if !enable || !unsafe { (*crtc_state).has_psr } {
let __v7: u32 = val & 4293918719u32;
val = __v7;
}
intel_de_write(display, reg, val);
intel_de_posting_read(display, reg);
if !enable {
return;
}
intel_write_dp_sdp(encoder, crtc_state, 7u32);
intel_write_dp_sdp(encoder, crtc_state, 34u32);
intel_write_dp_sdp(encoder, crtc_state, 10u32);
}

fn intel_dp_as_sdp_unpack(as_sdp: *mut drm_dp_as_sdp, buffer: &(), size: u64) -> i32 {
let sdp: *const dp_sdp = (buffer as *const ()) as *const dp_sdp;
if size < (core::mem::size_of::<dp_sdp>() as u64) {
return 22i32.wrapping_neg();
}
unsafe { memset(as_sdp as *mut (), 0i32, core::mem::size_of::<drm_dp_as_sdp>() as u64) };
if ((unsafe { (*sdp).sdp_header.HB0 }) as i32) != 0 {
return 22i32.wrapping_neg();
}
if ((unsafe { (*sdp).sdp_header.HB1 }) as i32) != 34 {
return 22i32.wrapping_neg();
}
if ((unsafe { (*sdp).sdp_header.HB3 }) as i32) & 63 != 9 {
return 22i32.wrapping_neg();
}
unsafe {
(*as_sdp).sdp_type = (*sdp).sdp_header.HB1;
}
unsafe {
(*as_sdp).length = ((((*sdp).sdp_header.HB3 as i32) as u64) & GENMASK__12!(5i32, 0i32)) as u8;
}
unsafe {
(*as_sdp).revision = (*sdp).sdp_header.HB2;
}
unsafe {
(*as_sdp).mode = (((*((*sdp).db.as_ptr() as *const u8).wrapping_offset(0 as isize) as i32) as u64) & GENMASK__15!(1i32, 0i32)) as u32;
}
unsafe {
(*as_sdp).vtotal = (*((*sdp).db.as_ptr() as *const u8).wrapping_offset(2 as isize) as i32) << 8i32 | (*((*sdp).db.as_ptr() as *const u8).wrapping_offset(1 as isize) as i32);
}
unsafe {
(*as_sdp).target_rr = ((*((*sdp).db.as_ptr() as *const u8).wrapping_offset(4 as isize) as i32) & 3) << 8i32 | (*((*sdp).db.as_ptr() as *const u8).wrapping_offset(3 as isize) as i32);
}
unsafe {
(*as_sdp).target_rr_divider = (*((*sdp).db.as_ptr() as *const u8).wrapping_offset(4 as isize) as i32) & 32 != 0;
}
unsafe {
(*as_sdp).coasting_vtotal = (*((*sdp).db.as_ptr() as *const u8).wrapping_offset(8 as isize) as i32) << 8i32 | (*((*sdp).db.as_ptr() as *const u8).wrapping_offset(7 as isize) as i32);
}
0
}

fn intel_dp_vsc_sdp_unpack(vsc: *mut drm_dp_vsc_sdp, buffer: &(), size: u64) -> i32 {
if size < (core::mem::size_of::<dp_sdp>() as u64) {
return 22i32.wrapping_neg();
}
unsafe { memset(vsc as *mut (), 0i32, core::mem::size_of::<drm_dp_vsc_sdp>() as u64) };
if ((unsafe { (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB0 }) as i32) != 0 {
return 22i32.wrapping_neg();
}
if ((unsafe { (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB1 }) as i32) != 7 {
return 22i32.wrapping_neg();
}
unsafe {
(*vsc).sdp_type = (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB1;
}
unsafe {
(*vsc).revision = (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB2;
}
unsafe {
(*vsc).length = (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB3;
}
if ((unsafe { (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB2 }) as i32) == 2 && ((unsafe { (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB3 }) as i32) == 8 || ((unsafe { (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB2 }) as i32) == 4 && ((unsafe { (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB3 }) as i32) == 14 || ((unsafe { (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB2 }) as i32) == 6 && ((unsafe { (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB3 }) as i32) == 16 {
// - HB2 = 0x2, HB3 = 0x8
//   VSC SDP supporting 3D stereo + PSR
// - HB2 = 0x4, HB3 = 0xe
//   VSC SDP supporting 3D stereo + PSR2 with Y-coordinate of
//   first scan line of the SU region (applies to eDP v1.4b
//   and higher).
// - HB2 = 0x6, HB3 = 0x10
//   VSC SDP supporting 3D stereo + Panel Replay.
return 0;
} else {
if ((unsafe { (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB2 }) as i32) == 5 && ((unsafe { (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB3 }) as i32) == 19 {
// - HB2 = 0x5, HB3 = 0x13
//   VSC SDP supporting 3D stereo + PSR2 + Pixel Encoding/Colorimetry
//   Format.
unsafe {
(*vsc).pixelformat = ((*((*((buffer as *const ()) as *const dp_sdp)).db.as_ptr() as *const u8).wrapping_offset(16 as isize) as i32) >> 4i32 & 15) as u32;
}
unsafe {
(*vsc).colorimetry = ((*((*((buffer as *const ()) as *const dp_sdp)).db.as_ptr() as *const u8).wrapping_offset(16 as isize) as i32) & 15) as u32;
}
unsafe {
(*vsc).dynamic_range = ((*((*((buffer as *const ()) as *const dp_sdp)).db.as_ptr() as *const u8).wrapping_offset(17 as isize) as i32) >> 7i32 & 1) as u32;
}
match ((unsafe { *((*((buffer as *const ()) as *const dp_sdp)).db.as_ptr() as *const u8).wrapping_offset(17 as isize) }) as i32) & 7 {
0 => {
unsafe {
(*vsc).bpc = 6;
}
}
1 => {
unsafe {
(*vsc).bpc = 8;
}
}
2 => {
unsafe {
(*vsc).bpc = 10;
}
}
3 => {
unsafe {
(*vsc).bpc = 12;
}
}
4 => {
unsafe {
(*vsc).bpc = 16;
}
}
_ => {
let mut __v1: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"sdp->db[17] & 0x7".as_ptr() as *mut u8, ((*((*((buffer as *const ()) as *const dp_sdp)).db.as_ptr() as *const u8).wrapping_offset(17 as isize) as i32) & 7) as i64) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_vsc_sdp_unpack_4, sym __slate_asm_string_intel_dp_vsc_sdp_unpack_5, const 5471 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, c"sdp->db[17] & 0x7".as_ptr() as *mut u8, ((*((*((buffer as *const ()) as *const dp_sdp)).db.as_ptr() as *const u8).wrapping_offset(17 as isize) as i32) & 7) as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
return 22i32.wrapping_neg();
}
}
unsafe {
(*vsc).content_type = ((*((*((buffer as *const ()) as *const dp_sdp)).db.as_ptr() as *const u8).wrapping_offset(18 as isize) as i32) & 7) as u32;
}
} else {
return 22i32.wrapping_neg();
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_872: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intel_read_dp_as_sdp(encoder: *mut intel_encoder, crtc_state: *mut intel_crtc_state, as_sdp: *mut drm_dp_as_sdp) {
let display: *mut intel_display = to_intel_display!(encoder);
let dig_port: *mut intel_digital_port = enc_to_dig_port(encoder);
let r#type: u32 = 34;
let mut sdp: dp_sdp = {
let mut __t0: dp_sdp = unsafe { core::mem::zeroed() };
__t0
};
let mut ret: i32 = 0;
if (unsafe { (*crtc_state).infoframes.enable }) & unsafe { intel_hdmi_infoframe_enable(r#type) } == 0 {
return;
}
unsafe { unsafe { (*dig_port).read_infoframe }.unwrap()(encoder, crtc_state as *const intel_crtc_state, r#type, core::ptr::addr_of_mut!(sdp) as *mut (), core::mem::size_of::<dp_sdp>() as i64) };
ret = intel_dp_as_sdp_unpack(as_sdp, unsafe { &(*(core::ptr::addr_of_mut!(sdp) as *const ())) }, core::mem::size_of::<dp_sdp>() as u64);
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Failed to unpack DP AS SDP\n".as_ptr() as *const u8) };
}
}

fn intel_dp_hdr_metadata_infoframe_sdp_unpack(drm_infoframe: *mut hdmi_drm_infoframe, buffer: &(), size: u64) -> i32 {
let mut ret: i32 = 0;
if size < (core::mem::size_of::<dp_sdp>() as u64) {
return 22i32.wrapping_neg();
}
if ((unsafe { (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB0 }) as i32) != 0 {
return 22i32.wrapping_neg();
}
if ((unsafe { (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB1 }) as i32) != 135 {
return 22i32.wrapping_neg();
}
// Least Significant Eight Bits of (Data Byte Count – 1)
// 1Dh (i.e., Data Byte Count = 30 bytes).
if ((unsafe { (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB2 }) as i32) != 29 {
return 22i32.wrapping_neg();
}
// Most Significant Two Bits of (Data Byte Count – 1), Clear to 00b.
if ((unsafe { (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB3 }) as i32) & 3 != 0 {
return 22i32.wrapping_neg();
}
// INFOFRAME SDP Version Number
if ((unsafe { (*((buffer as *const ()) as *const dp_sdp)).sdp_header.HB3 }) as i32) >> 2i32 & 63 != 19 {
return 22i32.wrapping_neg();
}
// CTA Header Byte 2 (INFOFRAME Version Number)
if ((unsafe { *((*((buffer as *const ()) as *const dp_sdp)).db.as_ptr() as *const u8).wrapping_offset(0 as isize) }) as i32) != 1 {
return 22i32.wrapping_neg();
}
// CTA Header Byte 3 (Length of INFOFRAME): HDMI_DRM_INFOFRAME_SIZE
if ((unsafe { *((*((buffer as *const ()) as *const dp_sdp)).db.as_ptr() as *const u8).wrapping_offset(1 as isize) }) as i32) != 26 {
return 22i32.wrapping_neg();
}
ret = unsafe { hdmi_drm_infoframe_unpack_only(drm_infoframe, ((*((buffer as *const ()) as *const dp_sdp)).db.as_ptr() as *const u8).wrapping_offset(2 as isize) as *const (), 26u64) };
ret
}

fn intel_read_dp_vsc_sdp(encoder: *mut intel_encoder, crtc_state: *mut intel_crtc_state, vsc: *mut drm_dp_vsc_sdp) {
let display: *mut intel_display = to_intel_display!(encoder);
let dig_port: *mut intel_digital_port = enc_to_dig_port(encoder);
let r#type: u32 = 7;
let mut sdp: dp_sdp = {
let mut __t0: dp_sdp = unsafe { core::mem::zeroed() };
__t0
};
let mut ret: i32 = 0;
if (unsafe { (*crtc_state).infoframes.enable }) & unsafe { intel_hdmi_infoframe_enable(r#type) } == 0 {
return;
}
unsafe { unsafe { (*dig_port).read_infoframe }.unwrap()(encoder, crtc_state as *const intel_crtc_state, r#type, core::ptr::addr_of_mut!(sdp) as *mut (), core::mem::size_of::<dp_sdp>() as i64) };
ret = intel_dp_vsc_sdp_unpack(vsc, unsafe { &(*(core::ptr::addr_of_mut!(sdp) as *const ())) }, core::mem::size_of::<dp_sdp>() as u64);
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Failed to unpack DP VSC SDP\n".as_ptr() as *const u8) };
}
}

fn intel_read_dp_hdr_metadata_infoframe_sdp(encoder: *mut intel_encoder, crtc_state: *mut intel_crtc_state, drm_infoframe: *mut hdmi_drm_infoframe) {
let display: *mut intel_display = to_intel_display!(encoder);
let dig_port: *mut intel_digital_port = enc_to_dig_port(encoder);
let r#type: u32 = 10;
let mut sdp: dp_sdp = {
let mut __t0: dp_sdp = unsafe { core::mem::zeroed() };
__t0
};
let mut ret: i32 = 0;
if (unsafe { (*crtc_state).infoframes.enable }) & unsafe { intel_hdmi_infoframe_enable(r#type) } == 0 {
return;
}
unsafe { unsafe { (*dig_port).read_infoframe }.unwrap()(encoder, crtc_state as *const intel_crtc_state, r#type, core::ptr::addr_of_mut!(sdp) as *mut (), core::mem::size_of::<dp_sdp>() as i64) };
ret = intel_dp_hdr_metadata_infoframe_sdp_unpack(drm_infoframe, unsafe { &(*(core::ptr::addr_of_mut!(sdp) as *const ())) }, core::mem::size_of::<dp_sdp>() as u64);
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Failed to unpack DP HDR Metadata Infoframe SDP\n".as_ptr() as *const u8) };
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_read_dp_sdp(encoder: *mut intel_encoder, crtc_state: *mut intel_crtc_state, r#type: u32) {
match r#type {
7 => {
intel_read_dp_vsc_sdp(encoder, crtc_state, unsafe { core::ptr::addr_of_mut!((*crtc_state).infoframes.vsc) });
}
10 => {
intel_read_dp_hdr_metadata_infoframe_sdp(encoder, crtc_state, unsafe { core::ptr::addr_of_mut!((*crtc_state).infoframes.drm.drm) });
}
34 => {
intel_read_dp_as_sdp(encoder, crtc_state, unsafe { core::ptr::addr_of_mut!((*crtc_state).infoframes.as_sdp) });
}
_ => {
let mut __v1: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"type".as_ptr() as *mut u8, r#type as i64) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_read_dp_sdp_4, sym __slate_asm_string_intel_read_dp_sdp_5, const 5617 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, c"type".as_ptr() as *mut u8, r#type as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
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

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_873: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intel_dp_mst_hpd_irq(intel_dp: *mut intel_dp, esi: *mut u8, ack: *mut u8) {
let mut handled: bool = false;
unsafe { drm_dp_mst_hpd_irq_handle_event(core::ptr::addr_of_mut!((*intel_dp).mst.mgr), esi as *const u8, ack, core::ptr::addr_of_mut!(handled)) };
if ((unsafe { *esi.wrapping_offset(1 as isize) }) as i32) & 4i32 != 0 {
unsafe { intel_hdcp_handle_cp_irq((*intel_dp).attached_connector) };
let __v0: *mut u8 = ack.wrapping_offset(1 as isize);
unsafe {
*__v0 = ((*__v0 as i32) | 4i32) as u8;
}
}
}

/// intel_dp_check_mst_status - service any pending MST interrupts, check link status
/// @intel_dp: Intel DP struct
///
/// Read any pending MST interrupts, call MST core to handle these and ack the
/// interrupts. Check if the main and AUX link state is ok.
///
/// Returns:
/// - %true if pending interrupts were serviced (or no interrupts were
///   pending) w/o detecting an error condition.
/// - %false if an error condition - like AUX failure or a loss of link - is
///   detected, or another condition - like a DP tunnel BW state change - needs
///   servicing from the hotplug work.
fn intel_dp_check_mst_status(intel_dp: *mut intel_dp) -> bool {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let mut force_retrain: bool = unsafe { intel_dp_link_training_get_force_retrain((*intel_dp).link.training) };
let mut reprobe_needed: bool = false;
let mut tries: i32 = 33;
'__slate_break_1: loop {
let __v22: i32 = tries.wrapping_sub(1);
tries = __v22;
if __v22 == 0 {
break;
}
let mut esi: [u8; 4] = {
let mut __t0: [u8; 4] = unsafe { core::mem::zeroed() };
__t0
};
let mut ack: [u8; 4] = {
let mut __t1: [u8; 4] = unsafe { core::mem::zeroed() };
__t1
};
let mut new_irqs: bool = false;
if !intel_dp_get_sink_irq_esi(intel_dp, esi.as_mut_ptr() as *mut u8) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"failed to get ESI - device may have failed\n".as_ptr() as *const u8) };
reprobe_needed = true;
break '__slate_break_1;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"DPRX ESI: %4ph\n".as_ptr() as *const u8, esi.as_mut_ptr() as *mut u8) };
let __v2: *mut u8 = (ack.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize);
unsafe {
*__v2 = ((*__v2 as i32) | (esi[3] as i32) & 35i32) as u8;
}
intel_dp_mst_hpd_irq(intel_dp, esi.as_mut_ptr() as *mut u8, ack.as_mut_ptr() as *mut u8);
new_irqs = !mem_is_zero(ack.as_mut_ptr() as *const (), core::mem::size_of::<[u8; 4]>() as u64);
let mut __v5: i64;
{
let __ret_warn_on: i32 = ((ack[1] as i32) & -53i32 != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(ack[1] & ~((1 << 2) | (1 << 4) | (1 << 5)))".as_ptr() as *mut u8) };
let mut __v6: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v6 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v7: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t2: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t2, sym __slate_asm_string_intel_dp_check_mst_status_8, sym __slate_asm_string_intel_dp_check_mst_status_9, const 5680 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t2 as u64) };
}
__v7 = bug;
unsafe { __v6.unwrap()(__v7, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(ack[1] & ~((1 << 2) | (1 << 4) | (1 << 5)))".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
}
let mut __v11: i64;
let __ret_warn_on: i32 = ((ack[3] as i32) & -36i32 != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(ack[3] & ~((1 << 0) | (1 << 1) | (1 << 5)))".as_ptr() as *mut u8) };
let mut __v12: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v13: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t3: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t3, sym __slate_asm_string_intel_dp_check_mst_status_14, sym __slate_asm_string_intel_dp_check_mst_status_15, const 5681 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t3 as u64) };
}
__v13 = bug;
unsafe { __v12.unwrap()(__v13, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(ack[3] & ~((1 << 0) | (1 << 1) | (1 << 5)))".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v11 = (__ret_warn_on != 0) as i64;
let __v17: bool = new_irqs && !intel_dp_ack_sink_irq_esi(intel_dp, ack.as_mut_ptr() as *mut u8);
if __v17 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Failed to ack ESI\n".as_ptr() as *const u8) };
}
if (ack[1] as i32) & 48i32 != 0 {
unsafe { drm_dp_mst_hpd_irq_send_new_request(core::ptr::addr_of_mut!((*intel_dp).mst.mgr)) };
}
if force_retrain {
// Defer forced retraining to the regular link status check.
let __v18: *mut u8 = (ack.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize);
unsafe {
*__v18 = ((*__v18 as i32) | 2i32) as u8;
}
force_retrain = false;
}
if intel_dp_handle_link_service_irq(intel_dp, ack[3]) {
reprobe_needed = true;
}
if !new_irqs {
break '__slate_break_1;
}
}
if tries == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"DPRX ESI not clearing, device may be stuck\n".as_ptr() as *const u8) };
reprobe_needed = true;
}
!reprobe_needed
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_874: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_875: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intel_dp_handle_hdmi_link_status_change(intel_dp: *mut intel_dp) {
let mut is_active: bool = false;
let mut buf: u8 = 0;
is_active = unsafe { drm_dp_pcon_hdmi_link_active(core::ptr::addr_of_mut!((*intel_dp).aux)) };
if (unsafe { (*intel_dp).frl.is_trained }) && !is_active {
if drm_dp_dpcd_readb(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 12378u32, core::ptr::addr_of_mut!(buf)) < 0 {
return;
}
let __v1: u8 = ((buf as i32) & -129i32) as u8;
buf = __v1;
if drm_dp_dpcd_writeb(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 12378u32, buf) < 0 {
return;
}
unsafe { drm_dp_pcon_hdmi_frl_link_error_count(core::ptr::addr_of_mut!((*intel_dp).aux), core::ptr::addr_of_mut!((*(*intel_dp).attached_connector).base)) };
unsafe {
(*intel_dp).frl.is_trained = false;
}
// Restart FRL training or fall back to TMDS mode
intel_dp_check_frl_training(intel_dp);
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_has_connector(intel_dp: *mut intel_dp, conn_state: *const drm_connector_state) -> bool {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let mut encoder: *mut intel_encoder = unsafe { core::mem::zeroed() };
// MST
let mut pipe: i32 = 0;
if unsafe { (*conn_state).best_encoder }.is_null() {
return false;
}
// SST
encoder = unsafe { core::ptr::addr_of_mut!((*dp_to_dig_port(intel_dp)).base) };
if (unsafe { (*conn_state).best_encoder }) == unsafe { core::ptr::addr_of_mut!((*encoder).base) } {
return true;
}
while (pipe as i32) < 4 {
if (((unsafe { (*DISPLAY_RUNTIME_INFO!(display)).pipe_mask }) as i32) as u64) & BIT__2!(pipe) != 0 {
encoder = unsafe { core::ptr::addr_of_mut!((*(*((*intel_dp).mst.stream_encoders.as_mut_ptr() as *mut *mut intel_dp_mst_encoder).wrapping_offset((pipe as i32) as isize))).base) };
if (unsafe { (*conn_state).best_encoder }) == unsafe { core::ptr::addr_of_mut!((*encoder).base) } {
return true;
}
}
pipe = (pipe as i32).wrapping_add(1) as i32;
}
false
}

fn wait_for_connector_hw_done(conn_state: *const drm_connector_state) {
let connector: *mut intel_connector = to_intel_connector!(unsafe { (*conn_state).connector });
let display: *mut intel_display = to_intel_display!(connector);
drm_modeset_lock_assert_held(unsafe { core::ptr::addr_of_mut!((*(*display).drm).mode_config.connection_mutex) });
if unsafe { (*conn_state).commit }.is_null() {
return;
}
let mut __v2: i64;
let __ret_warn_on: i32 = ((unsafe { wait_for_completion_timeout(core::ptr::addr_of_mut!((*(*conn_state).commit).hw_done), msecs_to_jiffies(5000u32)) }) == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!wait_for_completion_timeout(&conn_state->commit->hw_done, msecs_to_jiffies(5000)))".as_ptr() as *mut u8) };
let mut __v3: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v3 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v4: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_wait_for_connector_hw_done_5, sym __slate_asm_string_wait_for_connector_hw_done_6, const 5771 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v4 = bug;
unsafe { __v3.unwrap()(__v4, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!wait_for_completion_timeout(&conn_state->commit->hw_done, msecs_to_jiffies(5000)))".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_876: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn intel_dp_get_active_pipes(intel_dp: *mut intel_dp, ctx: *mut drm_modeset_acquire_ctx, pipe_mask: *mut u8) -> i32 {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let mut conn_iter: drm_connector_list_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(conn_iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut ret: i32 = 0;
unsafe {
*pipe_mask = 0;
}
unsafe { drm_connector_list_iter_begin((*display).drm, core::ptr::addr_of_mut!(conn_iter)) };
loop {
let __v13: *mut intel_connector = to_intel_connector!(unsafe { drm_connector_list_iter_next(core::ptr::addr_of_mut!(conn_iter)) });
if __v13.is_null() {
break;
}
let conn_state: *mut drm_connector_state = unsafe { (*__v13).base.state };
let mut crtc: *mut intel_crtc = unsafe { core::mem::zeroed() };
if intel_dp_has_connector(intel_dp, conn_state as *const drm_connector_state) {
crtc = to_intel_crtc!(unsafe { (*conn_state).crtc });
if !crtc.is_null() {
ret = unsafe { drm_modeset_lock(core::ptr::addr_of_mut!((*crtc).base.mutex), ctx) };
if ret != 0 {
break;
}
let crtc_state: *mut intel_crtc_state = to_intel_crtc_state!(unsafe { (*crtc).base.state });
let mut __v4: i64;
let __ret_warn_on: i32 = !(!(!intel_crtc_has_dp_encoder(crtc_state as *const intel_crtc_state))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!intel_crtc_has_dp_encoder(crtc_state))".as_ptr() as *mut u8) };
let mut __v5: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v5 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v6: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_get_active_pipes_7, sym __slate_asm_string_intel_dp_get_active_pipes_8, const 5806 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v6 = bug;
unsafe { __v5.unwrap()(__v6, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!intel_crtc_has_dp_encoder(crtc_state))".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v4 = (__ret_warn_on != 0) as i64;
if unsafe { (*crtc_state).hw.active } {
wait_for_connector_hw_done(conn_state as *const drm_connector_state);
let __v11: u8 = unsafe { *pipe_mask };
let __v12: u8 = (((__v11 as i32) as u64) | BIT__2!(unsafe { (*crtc).pipe })) as u8;
unsafe {
*pipe_mask = __v12;
}
}
}
}
}
unsafe { drm_connector_list_iter_end(core::ptr::addr_of_mut!(conn_iter)) };
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_877: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn intel_dp_flush_connector_commits(connector: *mut intel_connector) {
wait_for_connector_hw_done((unsafe { (*connector).base.state }) as *const drm_connector_state);
}

fn intel_dp_handle_device_service_irq(intel_dp: *mut intel_dp, irq_mask: u8) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let mut __v1: i64;
let __ret_warn_on: i32 = ((irq_mask as i32) & -71i32 != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(irq_mask & ~((1 << 1) | (1 << 2) | (1 << 6)))".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_handle_device_service_irq_4, sym __slate_asm_string_intel_dp_handle_device_service_irq_5, const 5829 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(irq_mask & ~((1 << 1) | (1 << 2) | (1 << 6)))".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if (irq_mask as i32) & 2i32 != 0 {
unsafe { intel_dp_test_request(intel_dp) };
}
if (irq_mask as i32) & 4i32 != 0 {
unsafe { intel_hdcp_handle_cp_irq((*intel_dp).attached_connector) };
}
if (irq_mask as i32) & 64i32 != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Sink specific irq unhandled\n".as_ptr() as *const u8) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_878: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

/// Return %true if a full connector reprobe is required after handling a link
/// service IRQ event.
fn intel_dp_handle_link_service_irq(intel_dp: *mut intel_dp, irq_mask: u8) -> bool {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let encoder: *mut intel_encoder = unsafe { core::ptr::addr_of_mut!((*dp_to_dig_port(intel_dp)).base) };
let mut reprobe_needed: bool = false;
let mut __v1: i64;
let __ret_warn_on: i32 = ((irq_mask as i32) & -60i32 != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(irq_mask & ~(((1 << 0) | (1 << 1) | (1 << 3) | (1 << 4) | (1 << 5)) | ((1 << 0) | (1 << 1) | (1 << 5))))".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_handle_link_service_irq_4, sym __slate_asm_string_intel_dp_handle_link_service_irq_5, const 5854 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(irq_mask & ~(((1 << 0) | (1 << 1) | (1 << 3) | (1 << 4) | (1 << 5)) | ((1 << 0) | (1 << 1) | (1 << 5))))".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if (irq_mask as i32) & 1i32 != 0 {
unsafe {
(*intel_dp).reset_link_params = true;
}
reprobe_needed = true;
}
if (irq_mask as i32) & 2i32 != 0 {
unsafe { intel_dp_check_link_state(intel_dp) };
}
if (irq_mask as i32) & 8i32 != 0 {
intel_dp_handle_hdmi_link_status_change(intel_dp);
}
if (irq_mask as i32) & 16i32 != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s][ENCODER:%d:%s] Allowing connected off request\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name, (*encoder).base.base.id, (*encoder).base.name) };
}
let __v7: bool = (irq_mask as i32) & 32i32 != 0 && drm_dp_tunnel_handle_irq(unsafe { (*display).dp_tunnel_mgr }, unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }) != 0;
if __v7 {
reprobe_needed = true;
}
reprobe_needed
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_879: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

// According to DP spec
// 5.1.2:
//  1. Read DPCD
//  2. Configure link according to Receiver Capabilities
//  3. Use Link Training from 2.5.3.3 and 3.5.1.3
//  4. Check link status on receipt of hot-plug interrupt
//
// intel_dp_short_pulse -  handles short pulse interrupts
// when full detection is not required.
// Returns %true if short pulse is handled and full detection
// is NOT required and %false otherwise.
fn intel_dp_short_pulse(intel_dp: *mut intel_dp) -> bool {
let mut reprobe_needed: bool = false;
let mut esi: [u8; 4] = {
let mut __t0: [u8; 4] = unsafe { core::mem::zeroed() };
__t0
};
unsafe { intel_dp_test_reset(intel_dp) };
if !intel_dp_get_and_ack_sink_irq_esi_sst(intel_dp, esi.as_mut_ptr() as *mut u8) {
return false;
}
// If the current value of sink count doesn't match with
// the value that was stored earlier we need to do full
// detection.
if intel_dp_has_sink_count(intel_dp) && DP_GET_SINK_COUNT!(esi[0]) != ((unsafe { (*intel_dp).sink_count }) as i32) {
return false;
}
// No need to proceed if we are going to do full detect
intel_dp_handle_device_service_irq(intel_dp, esi[1]);
// Force checking the link status for DPCD_REV < 1.2
// TODO: let the link status check depend on LINK_STATUS_CHANGED
// or intel_dp->link.training.force_retrain for DPCD_REV >= 1.2
let __v0: *mut u8 = (esi.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize);
unsafe {
*__v0 = ((*__v0 as i32) | 2i32) as u8;
}
if intel_dp_handle_link_service_irq(intel_dp, esi[3]) {
reprobe_needed = true;
}
// Handle CEC interrupts, if any
drm_dp_cec_irq(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) });
let mut __v3: bool;
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_880() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*intel_dp).downstream_port_changed) as *const bool) };
if __v3 {
loop {
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_881() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*intel_dp).downstream_port_changed, false);
if true {
break;
}
}
reprobe_needed = true;
}
unsafe { intel_psr_short_pulse(intel_dp) };
if unsafe { intel_alpm_get_error(intel_dp) } {
unsafe { intel_alpm_disable(intel_dp) };
unsafe {
(*intel_dp).alpm.sink_alpm_error = true;
}
}
if unsafe { intel_dp_test_short_pulse(intel_dp) } {
reprobe_needed = true;
}
!reprobe_needed
}

// XXX this is probably wrong for multiple downstream ports
fn intel_dp_detect_dpcd(intel_dp: *mut intel_dp) -> u32 {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let dig_port: *mut intel_digital_port = dp_to_dig_port(intel_dp);
let dpcd: *mut u8 = unsafe { (*intel_dp).dpcd.as_mut_ptr() as *mut u8 };
let mut r#type: u8 = 0;
let mut __v1: i64;
let __ret_warn_on: i32 = !(!intel_dp_is_edp(intel_dp)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(intel_dp_is_edp(intel_dp))".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_detect_dpcd_4, sym __slate_asm_string_intel_dp_detect_dpcd_5, const 5956 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(intel_dp_is_edp(intel_dp))".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if __v1 != 0 {
return 1;
}
loop {
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_883() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*intel_dp).downstream_port_changed, false);
if true {
break;
}
}
unsafe { intel_lspcon_resume(dig_port) };
if !intel_dp_get_dpcd(intel_dp) {
return 2;
}
unsafe {
(*intel_dp).mst_detect = intel_dp_mst_detect(intel_dp);
}
// if there's no downstream port, we're done
if !drm_dp_is_branch(dpcd as *const u8) {
return 1;
}
// If we're HPD-aware, SINK_COUNT changes dynamically
if intel_dp_has_sink_count(intel_dp) && ((unsafe { *((*intel_dp).downstream_ports.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) & 8i32 != 0 {
return (if (unsafe { (*intel_dp).sink_count }) != 0 { 1i32 } else { 2i32 }) as u32;
}
if ((unsafe { (*intel_dp).mst_detect }) as u32) == 1 {
return 1;
}
// If no HPD, poke DDC gently
if unsafe { drm_probe_ddc(core::ptr::addr_of_mut!((*intel_dp).aux.ddc)) } {
return 1;
}
// Well we tried, say unknown for unreliable port types
if ((unsafe { *((*intel_dp).dpcd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) >= 17 {
r#type = (((unsafe { *((*intel_dp).downstream_ports.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) & 7i32) as u8;
if (r#type as i32) == 1 || (r#type as i32) == 4 {
return 3;
}
} else {
r#type = (((unsafe { *((*intel_dp).dpcd.as_mut_ptr() as *mut u8).wrapping_offset(5 as isize) }) as i32) & 6) as u8;
if (r#type as i32) == 2i32 || (r#type as i32) == 6i32 {
return 3;
}
}
// Anything else is out of spec, warn and ignore
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Broken DP branch device, ignoring\n".as_ptr() as *const u8) };
2
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_882: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn edp_detect(intel_dp: *mut intel_dp) -> u32 {
1
}

#[unsafe(no_mangle)]
extern "C" fn intel_digital_port_lock(encoder: *mut intel_encoder) {
let dig_port: *mut intel_digital_port = enc_to_dig_port(encoder);
if (unsafe { (*dig_port).lock }) != None {
unsafe { unsafe { (*dig_port).lock }.unwrap()(dig_port) };
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_digital_port_unlock(encoder: *mut intel_encoder) {
let dig_port: *mut intel_digital_port = enc_to_dig_port(encoder);
if (unsafe { (*dig_port).unlock }) != None {
unsafe { unsafe { (*dig_port).unlock }.unwrap()(dig_port) };
}
}

/// intel_digital_port_connected_locked - is the specified port connected?
/// @encoder: intel_encoder
///
/// In cases where there's a connector physically connected but it can't be used
/// by our hardware we also return false, since the rest of the driver should
/// pretty much treat the port as disconnected. This is relevant for type-C
/// (starting on ICL) where there's ownership involved.
///
/// The caller must hold the lock acquired by calling intel_digital_port_lock()
/// when calling this function.
///
/// Return %true if port is connected, %false otherwise.
#[unsafe(no_mangle)]
extern "C" fn intel_digital_port_connected_locked(encoder: *mut intel_encoder) -> bool {
let display: *mut intel_display = to_intel_display!(encoder);
let dig_port: *mut intel_digital_port = enc_to_dig_port(encoder);
let is_glitch_free: bool = unsafe { intel_tc_port_handles_hpd_glitches(dig_port) };
let mut is_connected: bool = false;
let mut __UNIQUE_ID_wakeref_884: *mut ref_tracker = unsafe { intel_display_power_get(display, 0u32) };
while !__UNIQUE_ID_wakeref_884.is_null() {
let mut __v1: i32;
let __timeout_us: u64 = 4000;
let __sleep_us: u64 = 30;
let __timeout: i64 = ktime_add_us(unsafe { ktime_get() }, __timeout_us);
let mut ___ret: i32 = 0;
might_sleep_if!(__sleep_us != 0);
if false && __sleep_us != 0 {
usleep_range((__sleep_us >> 2i32).wrapping_add(1), __sleep_us);
}
loop {
let mut __expired: bool = false;
__expired = __timeout_us != 0 && ktime_compare(unsafe { ktime_get() }, __timeout) > 0;
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
is_connected = unsafe { unsafe { (*dig_port).connected }.unwrap()(encoder) };
if is_connected || is_glitch_free {
___ret = 0;
break;
}
if __expired {
___ret = 110i32.wrapping_neg();
break;
}
if __sleep_us != 0 {
usleep_range((__sleep_us >> 2i32).wrapping_add(1), __sleep_us);
}
cpu_relax();
}
__v1 = ___ret;
intel_display_power_put_async(display, 0u32, __UNIQUE_ID_wakeref_884);
__UNIQUE_ID_wakeref_884 = core::ptr::null_mut::<ref_tracker>();
}
is_connected
}

#[unsafe(no_mangle)]
extern "C" fn intel_digital_port_connected(encoder: *mut intel_encoder) -> bool {
let mut ret: bool = false;
intel_digital_port_lock(encoder);
ret = intel_digital_port_connected_locked(encoder);
intel_digital_port_unlock(encoder);
ret
}

fn intel_dp_get_edid(intel_dp: *mut intel_dp) -> *const drm_edid {
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let fixed_edid: *const drm_edid = unsafe { (*connector).panel.fixed_edid };
// Use panel fixed edid if we have one
if !fixed_edid.is_null() {
// invalid edid
if IS_ERR(fixed_edid as *const ()) {
return core::ptr::null::<drm_edid>();
}
return unsafe { drm_edid_dup(fixed_edid) };
}
unsafe { drm_edid_read_ddc(core::ptr::addr_of_mut!((*connector).base), core::ptr::addr_of_mut!((*intel_dp).aux.ddc)) }
}

fn intel_dp_update_dfp(intel_dp: *mut intel_dp, drm_edid: *const drm_edid) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
unsafe {
(*intel_dp).dfp.max_bpc = drm_dp_downstream_max_bpc(((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8, ((*intel_dp).downstream_ports.as_mut_ptr() as *mut u8) as *const u8, drm_edid) as u8;
}
unsafe {
(*intel_dp).dfp.max_dotclock = drm_dp_downstream_max_dotclock(((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8, ((*intel_dp).downstream_ports.as_mut_ptr() as *mut u8) as *const u8);
}
unsafe {
(*intel_dp).dfp.min_tmds_clock = drm_dp_downstream_min_tmds_clock(((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8, ((*intel_dp).downstream_ports.as_mut_ptr() as *mut u8) as *const u8, drm_edid);
}
unsafe {
(*intel_dp).dfp.max_tmds_clock = drm_dp_downstream_max_tmds_clock(((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8, ((*intel_dp).downstream_ports.as_mut_ptr() as *mut u8) as *const u8, drm_edid);
}
unsafe {
(*intel_dp).dfp.pcon_max_frl_bw = drm_dp_get_pcon_max_frl_bw(((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8, ((*intel_dp).downstream_ports.as_mut_ptr() as *mut u8) as *const u8);
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] DFP max bpc %d, max dotclock %d, TMDS clock %d-%d, PCON Max FRL BW %dGbps\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name, (*intel_dp).dfp.max_bpc as i32, (*intel_dp).dfp.max_dotclock, (*intel_dp).dfp.min_tmds_clock, (*intel_dp).dfp.max_tmds_clock, (*intel_dp).dfp.pcon_max_frl_bw) };
intel_dp_get_pcon_dsc_cap(intel_dp);
}

fn intel_dp_can_ycbcr420(intel_dp: *mut intel_dp) -> bool {
let __v0: bool = source_can_output(intel_dp, 1u32) && (!drm_dp_is_branch((unsafe { (*intel_dp).dpcd.as_mut_ptr() as *mut u8 }) as *const u8) || unsafe { (*intel_dp).dfp.ycbcr420_passthrough });
if __v0 {
return true;
}
let __v1: bool = source_can_output(intel_dp, 0u32) && dfp_can_convert_from_rgb(intel_dp, 1u32);
if __v1 {
return true;
}
let __v2: bool = source_can_output(intel_dp, 2u32) && dfp_can_convert_from_ycbcr444(intel_dp, 1u32);
if __v2 {
return true;
}
false
}

fn intel_dp_update_420(intel_dp: *mut intel_dp) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
unsafe {
(*intel_dp).dfp.ycbcr420_passthrough = drm_dp_downstream_420_passthrough(((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8, ((*intel_dp).downstream_ports.as_mut_ptr() as *mut u8) as *const u8);
}
// on-board LSPCON always assumed to support 4:4:4->4:2:0 conversion
let __v1: bool = (unsafe { intel_lspcon_active(dp_to_dig_port(intel_dp)) }) || unsafe { drm_dp_downstream_444_to_420_conversion(((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8, ((*intel_dp).downstream_ports.as_mut_ptr() as *mut u8) as *const u8) };
unsafe {
(*intel_dp).dfp.ycbcr_444_to_420 = __v1;
}
unsafe {
(*intel_dp).dfp.rgb_to_ycbcr = drm_dp_downstream_rgb_to_ycbcr_conversion(((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8, ((*intel_dp).downstream_ports.as_mut_ptr() as *mut u8) as *const u8, 64u8);
}
unsafe {
(*connector).base.ycbcr_420_allowed = intel_dp_can_ycbcr420(intel_dp);
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] RGB->YcbCr conversion? %s, YCbCr 4:2:0 allowed? %s, YCbCr 4:4:4->4:2:0 conversion? %s\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name, str_yes_no((*intel_dp).dfp.rgb_to_ycbcr), str_yes_no((*connector).base.ycbcr_420_allowed), str_yes_no((*intel_dp).dfp.ycbcr_444_to_420)) };
}

fn intel_dp_set_edid(intel_dp: *mut intel_dp) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let mut vrr_capable: bool = false;
intel_dp_unset_edid(intel_dp);
let drm_edid: *const drm_edid = intel_dp_get_edid(intel_dp);
unsafe {
(*connector).detect_edid = drm_edid;
}
// Below we depend on display info having been updated
unsafe { drm_edid_connector_update(core::ptr::addr_of_mut!((*connector).base), drm_edid) };
vrr_capable = unsafe { intel_vrr_is_capable(connector) };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] VRR capable: %s\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name, str_yes_no(vrr_capable)) };
unsafe { drm_connector_set_vrr_capable_property(core::ptr::addr_of_mut!((*connector).base), vrr_capable) };
intel_dp_update_dfp(intel_dp, drm_edid);
intel_dp_update_420(intel_dp);
drm_dp_cec_attach(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, unsafe { (*connector).base.display_info.source_physical_address });
}

fn intel_dp_unset_edid(intel_dp: *mut intel_dp) {
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
drm_dp_cec_unset_edid(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) });
unsafe { drm_edid_free((*connector).detect_edid) };
unsafe {
(*connector).detect_edid = core::ptr::null::<drm_edid>();
}
unsafe {
(*intel_dp).dfp.max_bpc = 0;
}
unsafe {
(*intel_dp).dfp.max_dotclock = 0;
}
unsafe {
(*intel_dp).dfp.min_tmds_clock = 0;
}
unsafe {
(*intel_dp).dfp.max_tmds_clock = 0;
}
unsafe {
(*intel_dp).dfp.pcon_max_frl_bw = 0;
}
unsafe {
(*intel_dp).dfp.ycbcr_444_to_420 = false;
}
unsafe {
(*connector).base.ycbcr_420_allowed = false;
}
unsafe { drm_connector_set_vrr_capable_property(core::ptr::addr_of_mut!((*connector).base), false) };
}

fn intel_dp_sink_supports_as_sdp_v2(intel_dp: *mut intel_dp) -> bool {
let mut rx_features: u8 = 0;
// The DP spec does not explicitly provide the AS SDP v2 capability.
// So based on the DP v2.1 SCR, we infer it from the following bits:
//
// DP_AS_SDP_FAVT_PAYLOAD_FIELDS_PARSING_SUPPORTED indicates support for
// FAVT, which is explicitly defined to use AS SDP v2.
//
// DP_ASYNC_VIDEO_TIMING_NOT_SUPPORTED_IN_PR indicates that the sink
// does not support asynchronous video timing while in PR Active,
// requiring the source to keep transmitting Adaptive-Sync SDPs. The
// spec mandates that such sinks shall support AS SDP v2.
//
// #TODO: Check the Adaptive-Sync DisplayID 2.1 block once DisplayID
// parsing is available. This may help detect AS SDP v2 support for
// native DP 2.1 sinks that do not expose FAVT or PR-based capability
// bits.
//
// In the presence of PCONs, check PCON support from DPCD and sink
// support from Display ID.
if drm_dp_dpcd_read_byte(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, 8724u32, core::ptr::addr_of_mut!(rx_features)) == 1 {
if ((rx_features as i32) as u64) & BIT!(2i32) != 0 {
return true;
}
}
let __v0: bool = (unsafe { (*intel_dp).psr.sink_panel_replay_support }) && !unsafe { intel_psr_pr_async_video_timing_supported(intel_dp) };
if __v0 {
return true;
}
false
}

fn intel_dp_detect_sdp_caps(intel_dp: *mut intel_dp) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let __v1: bool = HAS_AS_SDP!(display) && unsafe { drm_dp_as_sdp_supported(core::ptr::addr_of_mut!((*intel_dp).aux), ((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8) };
unsafe {
(*intel_dp).as_sdp_supported = __v1;
}
if !unsafe { (*intel_dp).as_sdp_supported } {
return;
}
// eDP Adaptive-Sync SDP always uses AS SDP v2
if intel_dp_is_edp(intel_dp) {
unsafe {
(*intel_dp).as_sdp_v2_supported = true;
}
} else {
unsafe {
(*intel_dp).as_sdp_v2_supported = intel_dp_sink_supports_as_sdp_v2(intel_dp);
}
}
}

fn intel_dp_needs_dpcd_probe(intel_dp: *mut intel_dp, force_on_external: bool) -> bool {
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
if intel_dp_is_edp(intel_dp) {
return false;
}
if force_on_external {
return true;
}
if unsafe { (*intel_dp).is_mst } {
return false;
}
unsafe { drm_edid_has_quirk(core::ptr::addr_of_mut!((*connector).base), 0u32) }
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_dpcd_set_probe(intel_dp: *mut intel_dp, force_on_external: bool) {
unsafe { drm_dp_dpcd_set_probe(core::ptr::addr_of_mut!((*intel_dp).aux), intel_dp_needs_dpcd_probe(intel_dp, force_on_external)) };
}

#[unsafe(link_section = ".text.slate_distinct.intel_dp.intel_dp_detect")]
extern "C" fn intel_dp_detect(_connector: *mut drm_connector, ctx: *mut drm_modeset_acquire_ctx, force: bool) -> i32 {
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut bug: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug) as *mut *mut bug_entry;
let mut __v6: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v6) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(status) as *mut u32;
let mut encoder: *mut intel_encoder = unsafe { core::mem::zeroed() };
let mut dig_port: *mut intel_digital_port = unsafe { core::mem::zeroed() };
let mut intel_dp: *mut intel_dp = unsafe { core::mem::zeroed() };
let mut connector: *mut intel_connector = unsafe { core::mem::zeroed() };
let mut display: *mut intel_display = unsafe { core::mem::zeroed() };
unsafe {
display = to_intel_display__4!(unsafe { (*_connector).dev });
connector = to_intel_connector!(_connector);
intel_dp = intel_attached_dp(connector);
dig_port = dp_to_dig_port(intel_dp);
encoder = core::ptr::addr_of_mut!((*dig_port).base);
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s]\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name);
let __ret_warn_on: i32 = !(!(!drm_modeset_is_locked(core::ptr::addr_of_mut!((*(*display).drm).mode_config.connection_mutex)))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!drm_modeset_is_locked(&display->drm->mode_config.connection_mutex))".as_ptr() as *mut u8);
*__slate_slot_6 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_detect_17, sym __slate_asm_string_intel_dp_detect_18, const 6319 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_4 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
(*__slate_slot_6).unwrap()(*__slate_slot_4, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!drm_modeset_is_locked(&display->drm->mode_config.connection_mutex))".as_ptr() as *mut u8);
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
if !intel_display_device_enabled(display) {
return 2;
} else {
if !intel_display_driver_check_access(display) {
return ((*connector).base.status as u32) as i32;
} else {
intel_dp_flush_connector_commits(connector);
intel_pps_vdd_on(intel_dp);
// Can't disconnect eDP
if intel_dp_is_edp(intel_dp) {
*__slate_slot_11 = edp_detect(intel_dp);
} else {
if intel_digital_port_connected(encoder) {
*__slate_slot_11 = intel_dp_detect_dpcd(intel_dp);
} else {
*__slate_slot_11 = 2;
}
}
if (*__slate_slot_11 as u32) != 2 {
__v3 = !intel_dp_mst_verify_dpcd_state(intel_dp);
} else {
__v3 = false;
}
if __v3 {
*__slate_slot_11 = 2;
}
'__join_0: {
// This requires retrying detection for instance to re-enable
// the MST mode that got reset via a long HPD pulse. The retry
// will happen either via the hotplug handler's retry logic,
// ensured by setting the connector here to SST/disconnected,
// or via a userspace connector probing in response to the
// hotplug uevent sent when removing the MST connectors.
if (*__slate_slot_11 as u32) == 2 {
intel_dp_test_reset(intel_dp);
// FIXME: Resetting these caps here cause
// state computation fail if the connector need to be
// modeset after sink disconnect. Move resetting them
// to where new sink is connected.
memset(((*connector).dp.dsc_dpcd.as_mut_ptr() as *mut u8) as *mut (), 0i32, core::mem::size_of::<[u8; 16]>() as u64);
memset(((*connector).dp.panel_replay_caps.dpcd.as_mut_ptr() as *mut u8) as *mut (), 0i32, core::mem::size_of::<[u8; 7]>() as u64);
(*intel_dp).psr.sink_panel_replay_support = false;
(*connector).dp.panel_replay_caps.support = false;
(*connector).dp.panel_replay_caps.su_support = false;
(*connector).dp.panel_replay_caps.dsc_support = 0;
intel_dp_mst_disconnect(intel_dp);
intel_dp_tunnel_disconnect(intel_dp);
intel_dp_tunnel_uhbr_lanes_wa_reset(intel_dp);
} else {
intel_dp_init_source_oui(intel_dp);
*__slate_slot_10 = intel_dp_tunnel_detect(intel_dp, ctx);
if *__slate_slot_10 == 35i32.wrapping_neg() {
*__slate_slot_11 = *__slate_slot_10 as u32;
break '__join_0;
} else {
if *__slate_slot_10 == 1 {
let __v1: u64 = (*connector).base.epoch_counter;
(*connector).base.epoch_counter = __v1.wrapping_add(1);
}
if !intel_dp_is_edp(intel_dp) {
intel_psr_init_dpcd(intel_dp, connector);
}
intel_dp_detect_dsc_caps(intel_dp, connector);
intel_dp_detect_sdp_caps(intel_dp);
if (*intel_dp).reset_link_params {
intel_dp_reset_link_params(intel_dp);
(*intel_dp).reset_link_params = false;
}
intel_dp_mst_configure(intel_dp);
intel_dp_print_rates(intel_dp);
if (*intel_dp).is_mst {
// If we are in MST mode then this connector
// won't appear connected or have anything
// with EDID on it
*__slate_slot_11 = 2;
} else {
// Some external monitors do not signal loss of link synchronization
// with an IRQ_HPD, so force a link status check.
//
// TODO: this probably became redundant, so remove it: the link state
// is rechecked/recovered now after modesets, where the loss of
// synchronization tends to occur.
if !intel_dp_is_edp(intel_dp) {
intel_dp_check_link_state(intel_dp);
}
// Clearing NACK and defer counts to get their exact values
// while reading EDID which are required by Compliance tests
// 4.2.2.4 and 4.2.2.5
(*intel_dp).aux.i2c_nack_count = 0;
(*intel_dp).aux.i2c_defer_count = 0;
intel_dp_set_edid(intel_dp);
if intel_dp_is_edp(intel_dp) || !unsafe { (*connector).detect_edid }.is_null() {
*__slate_slot_11 = 1;
}
if intel_dp_tunnel_uhbr_lanes_wa_setup(intel_dp) {
intel_dp_tunnel_uhbr_lanes_wa_apply(intel_dp);
}
}
}
}
if (*__slate_slot_11 as u32) != 1 && !(*intel_dp).is_mst {
intel_dp_unset_edid(intel_dp);
}
intel_dp_dpcd_set_probe(intel_dp, false);
if !intel_dp_is_edp(intel_dp) {
drm_dp_set_subconnector_property(core::ptr::addr_of_mut!((*connector).base), *__slate_slot_11, ((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8, ((*intel_dp).downstream_ports.as_mut_ptr() as *mut u8) as *const u8);
}
}
intel_pps_vdd_off(intel_dp);
return (*__slate_slot_11 as u32) as i32;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_885: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.intel_dp.intel_dp_force")]
extern "C" fn intel_dp_force(_connector: *mut drm_connector) {
let connector: *mut intel_connector = to_intel_connector!(_connector);
let display: *mut intel_display = to_intel_display!(connector);
let intel_dp: *mut intel_dp = intel_attached_dp(connector);
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s]\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name) };
if !unsafe { intel_display_driver_check_access(display) } {
return;
}
intel_dp_unset_edid(intel_dp);
if ((unsafe { (*connector).base.status }) as u32) != 1 {
return;
}
intel_dp_set_edid(intel_dp);
intel_dp_dpcd_set_probe(intel_dp, false);
}

#[unsafe(link_section = ".text.slate_distinct.intel_dp.intel_dp_get_modes")]
extern "C" fn intel_dp_get_modes(_connector: *mut drm_connector) -> i32 {
let display: *mut intel_display = to_intel_display__4!(unsafe { (*_connector).dev });
let connector: *mut intel_connector = to_intel_connector!(_connector);
let intel_dp: *mut intel_dp = intel_attached_dp(connector);
let mut num_modes: i32 = 0;
// drm_edid_connector_update() done in ->detect() or ->force()
num_modes = unsafe { drm_edid_connector_add_modes(core::ptr::addr_of_mut!((*connector).base)) };
// Also add fixed mode, which may or may not be present in EDID
if intel_dp_is_edp(intel_dp) {
let __v2: i32 = unsafe { intel_panel_get_modes(connector) };
num_modes = num_modes.wrapping_add(__v2);
}
if num_modes != 0 {
return num_modes;
}
if unsafe { (*connector).detect_edid }.is_null() {
let mode: *mut drm_display_mode = unsafe { drm_dp_downstream_mode((*display).drm, ((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8, ((*intel_dp).downstream_ports.as_mut_ptr() as *mut u8) as *const u8) };
if !mode.is_null() {
unsafe { drm_mode_probed_add(core::ptr::addr_of_mut!((*connector).base), mode) };
num_modes = num_modes.wrapping_add(1);
}
}
num_modes
}

#[unsafe(link_section = ".text.slate_distinct.intel_dp.intel_dp_connector_register")]
extern "C" fn intel_dp_connector_register(_connector: *mut drm_connector) -> i32 {
let connector: *mut intel_connector = to_intel_connector!(_connector);
let display: *mut intel_display = to_intel_display!(connector);
let intel_dp: *mut intel_dp = intel_attached_dp(connector);
let dig_port: *mut intel_digital_port = dp_to_dig_port(intel_dp);
let mut ret: i32 = unsafe { intel_connector_register(core::ptr::addr_of_mut!((*connector).base)) };
if ret != 0 {
return ret;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"registering %s bus for %s\n".as_ptr() as *const u8, (*intel_dp).aux.name, (*(*connector).base.kdev).kobj.name) };
unsafe {
(*intel_dp).aux.dev = (*connector).base.kdev;
}
ret = unsafe { drm_dp_aux_register(core::ptr::addr_of_mut!((*intel_dp).aux)) };
if ret == 0 {
drm_dp_cec_register_connector(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) }, unsafe { core::ptr::addr_of_mut!((*connector).base) });
}
if !unsafe { intel_bios_encoder_is_lspcon((*dig_port).base.devdata) } {
return ret;
}
// ToDo: Clean this up to handle lspcon init and resume more
// efficiently and streamlined.
if unsafe { intel_lspcon_init(dig_port) } {
if unsafe { intel_lspcon_detect_hdr_capability(dig_port) } {
unsafe { drm_connector_attach_hdr_output_metadata_property(core::ptr::addr_of_mut!((*connector).base)) };
}
}
ret
}

#[unsafe(link_section = ".text.slate_distinct.intel_dp.intel_dp_connector_unregister")]
extern "C" fn intel_dp_connector_unregister(_connector: *mut drm_connector) {
let connector: *mut intel_connector = to_intel_connector!(_connector);
let intel_dp: *mut intel_dp = intel_attached_dp(connector);
drm_dp_cec_unregister_connector(unsafe { core::ptr::addr_of_mut!((*intel_dp).aux) });
unsafe { drm_dp_aux_unregister(core::ptr::addr_of_mut!((*intel_dp).aux)) };
unsafe { intel_connector_unregister(core::ptr::addr_of_mut!((*connector).base)) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.intel_dp.intel_dp_connector_sync_state")]
extern "C" fn intel_dp_connector_sync_state(connector: *mut intel_connector, crtc_state: *const intel_crtc_state) {
let display: *mut intel_display = to_intel_display!(connector);
if !crtc_state.is_null() && unsafe { (*crtc_state).dsc.compression_enable } {
let mut __v1: i64;
let __ret_warn_on: i32 = unsafe { (*connector).dp.dsc_decompression_aux }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!connector->dp.dsc_decompression_aux)".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_connector_sync_state_4, sym __slate_asm_string_intel_dp_connector_sync_state_5, const 6567 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!connector->dp.dsc_decompression_aux)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
unsafe {
(*connector).dp.__slate_bits_0.__set_dsc_decompression_enabled(1u8);
}
} else {
unsafe {
(*connector).dp.__slate_bits_0.__set_dsc_decompression_enabled(0u8);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_886: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn intel_dp_encoder_flush_work(_encoder: *mut drm_encoder) {
let encoder: *mut intel_encoder = to_intel_encoder!(_encoder);
let dig_port: *mut intel_digital_port = enc_to_dig_port(encoder);
let intel_dp: *mut intel_dp = unsafe { core::ptr::addr_of_mut!((*dig_port).dp) };
unsafe { intel_encoder_link_check_flush_work(encoder) };
unsafe { intel_dp_mst_encoder_cleanup(dig_port) };
intel_dp_tunnel_destroy(intel_dp);
unsafe { intel_pps_vdd_off_sync(intel_dp) };
// Ensure power off delay is respected on module remove, so that we can
// reduce delays at driver probe. See pps_init_timestamps().
unsafe { intel_pps_wait_power_cycle(intel_dp) };
unsafe { intel_dp_aux_fini(intel_dp) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.intel_dp.intel_dp_encoder_suspend")]
extern "C" fn intel_dp_encoder_suspend(encoder: *mut intel_encoder) {
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
unsafe { intel_pps_vdd_off_sync(intel_dp) };
intel_dp_tunnel_suspend(intel_dp);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.intel_dp.intel_dp_encoder_shutdown")]
extern "C" fn intel_dp_encoder_shutdown(encoder: *mut intel_encoder) {
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
unsafe { intel_pps_wait_power_cycle(intel_dp) };
}

fn intel_modeset_tile_group(state: *mut intel_atomic_state, tile_group_id: i32) -> i32 {
let display: *mut intel_display = to_intel_display!(state);
let mut conn_iter: drm_connector_list_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(conn_iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut connector: *mut intel_connector = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
unsafe { drm_connector_list_iter_begin((*display).drm, core::ptr::addr_of_mut!(conn_iter)) };
loop {
let __v3: *mut intel_connector = to_intel_connector!(unsafe { drm_connector_list_iter_next(core::ptr::addr_of_mut!(conn_iter)) });
connector = __v3;
if __v3.is_null() {
break;
}
let mut crtc: *mut intel_crtc = unsafe { core::mem::zeroed() };
if !(!unsafe { (*connector).base.has_tile } || (unsafe { (*(*connector).base.tile_group).id }) != tile_group_id) {
let conn_state: *mut drm_connector_state = unsafe { drm_atomic_get_connector_state(core::ptr::addr_of_mut!((*state).base), core::ptr::addr_of_mut!((*connector).base)) };
if IS_ERR(conn_state as *const ()) {
ret = PTR_ERR(conn_state as *const ()) as i32;
break;
}
crtc = to_intel_crtc!(unsafe { (*conn_state).crtc });
if !crtc.is_null() {
let crtc_state: *mut intel_crtc_state = intel_atomic_get_new_crtc_state(state, crtc);
unsafe {
let __t0 = true;
(*crtc_state).uapi.__slate_bits_0.__set_mode_changed(__t0);
}
ret = unsafe { drm_atomic_add_affected_planes(core::ptr::addr_of_mut!((*state).base), core::ptr::addr_of_mut!((*crtc).base)) };
if ret != 0 {
break;
}
}
}
}
unsafe { drm_connector_list_iter_end(core::ptr::addr_of_mut!(conn_iter)) };
ret
}

fn intel_modeset_affected_transcoders(state: *mut intel_atomic_state, mut transcoders: u16) -> i32 {
let display: *mut intel_display = to_intel_display!(state);
let mut crtc: *mut intel_crtc = unsafe { core::mem::zeroed() };
if (transcoders as i32) == 0 {
return 0;
}
crtc = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*display).pipe_list) });
while list_entry_is_head!(crtc, unsafe { core::ptr::addr_of_mut!((*display).pipe_list) }, pipe_head) == 0 {
let mut ret: i32 = 0;
let crtc_state: *mut intel_crtc_state = unsafe { intel_atomic_get_crtc_state(core::ptr::addr_of_mut!((*state).base), crtc) };
if IS_ERR(crtc_state as *const ()) {
return PTR_ERR(crtc_state as *const ()) as i32;
}
if unsafe { (*crtc_state).hw.enable } {
if ((transcoders as i32) as u64) & BIT__2!(unsafe { (*crtc_state).cpu_transcoder }) != 0 {
unsafe {
let __t0 = true;
(*crtc_state).uapi.__slate_bits_0.__set_mode_changed(__t0);
}
ret = unsafe { drm_atomic_add_affected_connectors(core::ptr::addr_of_mut!((*state).base), core::ptr::addr_of_mut!((*crtc).base)) };
if ret != 0 {
return ret;
}
ret = unsafe { drm_atomic_add_affected_planes(core::ptr::addr_of_mut!((*state).base), core::ptr::addr_of_mut!((*crtc).base)) };
if ret != 0 {
return ret;
}
let __v3: u16 = transcoders;
transcoders = (((__v3 as i32) as u64) & !BIT__2!(unsafe { (*crtc_state).cpu_transcoder })) as u16;
}
}
crtc = list_next_entry!(crtc, pipe_head);
}
let mut __v6: i64;
let __ret_warn_on: i32 = ((transcoders as i32) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(transcoders != 0)".as_ptr() as *mut u8) };
let mut __v7: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v7 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v8: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t1: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t1, sym __slate_asm_string_intel_modeset_affected_transcoders_9, sym __slate_asm_string_intel_modeset_affected_transcoders_10, const 6690 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t1 as u64) };
}
__v8 = bug;
unsafe { __v7.unwrap()(__v8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(transcoders != 0)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v6 = (__ret_warn_on != 0) as i64;
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_887: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intel_modeset_synced_crtcs(state: *mut intel_atomic_state, _connector: *mut drm_connector) -> i32 {
let connector: *mut intel_connector = to_intel_connector!(_connector);
let old_conn_state: *const drm_connector_state = drm_atomic_get_old_connector_state((unsafe { core::ptr::addr_of_mut!((*state).base) }) as *const drm_atomic_commit, unsafe { core::ptr::addr_of_mut!((*connector).base) }) as *const drm_connector_state;
let mut transcoders: u16 = 0;
let crtc: *mut intel_crtc = to_intel_crtc!(unsafe { (*old_conn_state).crtc });
if crtc.is_null() {
return 0;
}
let old_crtc_state: *const intel_crtc_state = intel_atomic_get_old_crtc_state(state, crtc) as *const intel_crtc_state;
if !unsafe { (*old_crtc_state).hw.active } {
return 0;
}
transcoders = unsafe { (*old_crtc_state).sync_mode_slaves_mask };
if ((unsafe { (*old_crtc_state).master_transcoder }) as i32) != -1 {
let __v2: u16 = transcoders;
transcoders = (((__v2 as i32) as u64) | BIT__2!(unsafe { (*old_crtc_state).master_transcoder })) as u16;
}
intel_modeset_affected_transcoders(state, transcoders)
}

#[unsafe(link_section = ".text.slate_distinct.intel_dp.intel_dp_connector_atomic_check")]
extern "C" fn intel_dp_connector_atomic_check(_connector: *mut drm_connector, _state: *mut drm_atomic_commit) -> i32 {
let connector: *mut intel_connector = to_intel_connector!(_connector);
let display: *mut intel_display = to_intel_display!(connector);
let state: *mut intel_atomic_state = to_intel_atomic_state!(_state);
let conn_state: *mut drm_connector_state = drm_atomic_get_new_connector_state(_state as *const drm_atomic_commit, unsafe { core::ptr::addr_of_mut!((*connector).base) });
let intel_dp: *mut intel_dp = enc_to_intel_dp(unsafe { (*connector).encoder });
let mut ret: i32 = unsafe { intel_digital_connector_atomic_check(core::ptr::addr_of_mut!((*connector).base), core::ptr::addr_of_mut!((*state).base)) };
if ret != 0 {
return ret;
}
if unsafe { intel_dp_mst_source_support(intel_dp) } {
ret = unsafe { drm_dp_mst_root_conn_atomic_check(conn_state, core::ptr::addr_of_mut!((*intel_dp).mst.mgr)) };
if ret != 0 {
return ret;
}
}
if !unsafe { intel_connector_needs_modeset(state, core::ptr::addr_of_mut!((*connector).base)) } {
return 0;
}
ret = intel_dp_tunnel_atomic_check_state(state, intel_dp, connector);
if ret != 0 {
return ret;
}
// We don't enable port sync on BDW due to missing w/as and
// due to not having adjusted the modeset sequence appropriately.
if (DISPLAY_VER!(display) as i32) < 9 {
return 0;
}
if unsafe { (*connector).base.has_tile } {
ret = intel_modeset_tile_group(state, unsafe { (*(*connector).base.tile_group).id });
if ret != 0 {
return ret;
}
}
intel_modeset_synced_crtcs(state, unsafe { core::ptr::addr_of_mut!((*connector).base) })
}

#[unsafe(link_section = ".text.slate_distinct.intel_dp.intel_dp_oob_hotplug_event")]
extern "C" fn intel_dp_oob_hotplug_event(_connector: *mut drm_connector, hpd_state: u32) {
let connector: *mut intel_connector = to_intel_connector!(_connector);
let display: *mut intel_display = to_intel_display!(connector);
let encoder: *mut intel_encoder = intel_attached_encoder(connector);
let hpd_high: bool = (hpd_state as u32) == 1;
let hpd_pin: u32 = (unsafe { (*encoder).hpd_pin }) as u32;
let mut need_work: bool = false;
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*display).irq.lock) });
if (hpd_high as i32) != test_bit__5!(hpd_pin, unsafe { core::ptr::addr_of_mut!((*display).hotplug.oob_hotplug_last_state) }) {
let __v4: u32 = unsafe { (*display).hotplug.event_bits };
let __v5: u32 = ((__v4 as u64) | BIT!(hpd_pin)) as u32;
unsafe {
(*display).hotplug.event_bits = __v5;
}
__assign_bit!(hpd_pin, unsafe { core::ptr::addr_of_mut!((*display).hotplug.oob_hotplug_last_state) }, hpd_high);
need_work = true;
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*display).irq.lock) });
if need_work {
unsafe { intel_hpd_schedule_detection(display) };
}
}

static intel_dp_connector_funcs: __SlateConst<drm_connector_funcs> = __SlateConst({
let mut __t0: drm_connector_funcs = unsafe { core::mem::zeroed() };
__t0.force = Some(intel_dp_force);
__t0.fill_modes = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector, u32, u32) -> i32>>(drm_helper_probe_single_connector_modes as *const ()) };
__t0.late_register = Some(intel_dp_connector_register);
__t0.early_unregister = Some(intel_dp_connector_unregister);
__t0.destroy = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector)>>(intel_connector_destroy as *const ()) };
__t0.atomic_duplicate_state = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector) -> *mut drm_connector_state>>(intel_digital_connector_duplicate_state as *const ()) };
__t0.atomic_destroy_state = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_connector_state)>>(drm_atomic_helper_connector_destroy_state as *const ()) };
__t0.atomic_set_property = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_connector_state, *mut drm_property, u64) -> i32>>(intel_digital_connector_atomic_set_property as *const ()) };
__t0.atomic_get_property = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector, *const drm_connector_state, *mut drm_property, *mut u64) -> i32>>(intel_digital_connector_atomic_get_property as *const ()) };
__t0.oob_hotplug_event = Some(intel_dp_oob_hotplug_event);
__t0
});

static intel_dp_connector_helper_funcs: __SlateConst<drm_connector_helper_funcs> = __SlateConst({
let mut __t0: drm_connector_helper_funcs = unsafe { core::mem::zeroed() };
__t0.get_modes = Some(intel_dp_get_modes);
__t0.detect_ctx = Some(intel_dp_detect);
__t0.mode_valid = Some(intel_dp_mode_valid);
__t0.atomic_check = Some(intel_dp_connector_atomic_check);
__t0
});

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.intel_dp.intel_dp_hpd_pulse")]
extern "C" fn intel_dp_hpd_pulse(dig_port: *mut intel_digital_port, long_hpd: bool) -> u32 {
let display: *mut intel_display = to_intel_display__6!(dig_port);
let intel_dp: *mut intel_dp = unsafe { core::ptr::addr_of_mut!((*dig_port).dp) };
let mut dpcd: [u8; 15] = [0u8; 15];
let __v1: bool = if ((unsafe { (*dig_port).base.r#type }) as u32) == 8 {
let __v2: bool = long_hpd || unsafe { intel_display_rpm_suspended(display) };
let __v3: bool = __v2 || !unsafe { intel_pps_have_panel_power_or_vdd(intel_dp) };
__v3
} else {
false
};
if __v1 {
// vdd off can generate a long/short pulse on eDP which
// would require vdd on to handle it, and thus we
// would end up in an endless cycle of
// "vdd off -> long/short hpd -> vdd on -> detect -> vdd off -> ..."
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"ignoring %s hpd on eDP [ENCODER:%d:%s]\n".as_ptr() as *const u8, if long_hpd { c"long".as_ptr() as *mut u8 } else { c"short".as_ptr() as *mut u8 }, (*dig_port).base.base.base.id, (*dig_port).base.base.name) };
return 1;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"got hpd irq on [ENCODER:%d:%s] - %s\n".as_ptr() as *const u8, (*dig_port).base.base.base.id, (*dig_port).base.base.name, if long_hpd { c"long".as_ptr() as *mut u8 } else { c"short".as_ptr() as *mut u8 }) };
// TBT DP tunnels require the GFX driver to read out the DPRX caps in
// response to long HPD pulses. The DP hotplug handler does that,
// however the hotplug handler may be blocked by another
// connector's/encoder's hotplug handler. Since the TBT CM may not
// complete the DP tunnel BW request for the latter connector/encoder
// waiting for this encoder's DPRX read, perform a dummy read here.
if long_hpd {
intel_dp_dpcd_set_probe(intel_dp, true);
unsafe { intel_dp_read_dprx_caps(intel_dp, dpcd.as_mut_ptr() as *mut u8) };
unsafe {
(*intel_dp).reset_link_params = true;
}
intel_dp_invalidate_source_oui(intel_dp);
return 0;
}
if unsafe { (*intel_dp).is_mst } {
if !intel_dp_check_mst_status(intel_dp) {
return 0;
}
} else {
if !intel_dp_short_pulse(intel_dp) {
return 0;
}
}
1
}

fn _intel_dp_is_port_edp(display: *mut intel_display, devdata: *const intel_bios_encoder_data, port: i32) -> bool {
// eDP not supported on g4x. so bail out early just
// for a bit extra safety in case the VBT is bonkers.
if (DISPLAY_VER!(display) as i32) < 5 {
return false;
}
if (DISPLAY_VER!(display) as i32) < 9 && (port as i32) == 0 {
return true;
}
!devdata.is_null() && unsafe { intel_bios_encoder_supports_edp(devdata) }
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_is_port_edp(display: *mut intel_display, port: i32) -> bool {
let devdata: *const intel_bios_encoder_data = unsafe { intel_bios_encoder_data_lookup(display, port) };
_intel_dp_is_port_edp(display, devdata, port)
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_has_gamut_metadata_dip(encoder: *mut intel_encoder) -> bool {
let display: *mut intel_display = to_intel_display!(encoder);
let port: i32 = unsafe { (*encoder).port };
if unsafe { intel_bios_encoder_is_lspcon((*encoder).devdata) } {
return false;
}
if (DISPLAY_VER!(display) as i32) >= 11 {
return true;
}
if (port as i32) == 0 {
return false;
}
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_haswell() }) as i32) != 0 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_broadwell() }) as i32) != 0 || (DISPLAY_VER!(display) as i32) >= 9 {
return true;
}
false
}

fn intel_dp_add_properties(intel_dp: *mut intel_dp, _connector: *mut drm_connector) {
let connector: *mut intel_connector = to_intel_connector!(_connector);
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let port: i32 = unsafe { (*dp_to_dig_port(intel_dp)).base.port };
if !intel_dp_is_edp(intel_dp) {
unsafe { drm_connector_attach_dp_subconnector_property(core::ptr::addr_of_mut!((*connector).base)) };
}
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_g4x() }) as i32) == 0 && (port as i32) != 0 {
unsafe { intel_attach_force_audio_property(core::ptr::addr_of_mut!((*connector).base)) };
}
unsafe { intel_attach_broadcast_rgb_property(core::ptr::addr_of_mut!((*connector).base)) };
if HAS_GMCH!(display) != 0 {
unsafe { drm_connector_attach_max_bpc_property(core::ptr::addr_of_mut!((*connector).base), 6i32, 10i32) };
} else {
if (DISPLAY_VER!(display) as i32) >= 5 {
unsafe { drm_connector_attach_max_bpc_property(core::ptr::addr_of_mut!((*connector).base), 6i32, 12i32) };
}
}
// Register HDMI colorspace for case of lspcon
if unsafe { intel_bios_encoder_is_lspcon((*dp_to_dig_port(intel_dp)).base.devdata) } {
unsafe { drm_connector_attach_content_type_property(core::ptr::addr_of_mut!((*connector).base)) };
unsafe { intel_attach_hdmi_colorspace_property(core::ptr::addr_of_mut!((*connector).base)) };
} else {
unsafe { intel_attach_dp_colorspace_property(core::ptr::addr_of_mut!((*connector).base)) };
}
if intel_dp_has_gamut_metadata_dip(unsafe { core::ptr::addr_of_mut!((*dp_to_dig_port(intel_dp)).base) }) {
unsafe { drm_connector_attach_hdr_output_metadata_property(core::ptr::addr_of_mut!((*connector).base)) };
}
if HAS_VRR!(display) {
unsafe { drm_connector_attach_vrr_capable_property(core::ptr::addr_of_mut!((*connector).base)) };
}
}

fn intel_edp_add_properties(intel_dp: *mut intel_dp) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let connector: *mut intel_connector = unsafe { (*intel_dp).attached_connector };
let fixed_mode: *const drm_display_mode = unsafe { intel_panel_preferred_fixed_mode(connector) };
unsafe { intel_attach_scaling_mode_property(core::ptr::addr_of_mut!((*connector).base)) };
unsafe { drm_connector_set_panel_orientation_with_quirk(core::ptr::addr_of_mut!((*connector).base), (*display).vbt.orientation, (*fixed_mode).hdisplay as i32, (*fixed_mode).vdisplay as i32) };
}

fn intel_edp_backlight_setup(intel_dp: *mut intel_dp, connector: *mut intel_connector) {
let display: *mut intel_display = to_intel_display__2!(intel_dp);
let mut pipe: i32 = -1;
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_valleyview() }) as i32) != 0 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_cherryview() }) as i32) != 0 {
pipe = unsafe { vlv_pps_backlight_initial_pipe(intel_dp) };
}
unsafe { intel_backlight_setup(connector, pipe) };
}

fn intel_edp_init_connector(intel_dp: *mut intel_dp, connector: *mut intel_connector) -> bool {
let mut __v0: bool = unsafe { core::mem::zeroed() };
// Unfortunately even the HPD based detection fails on
// eg. Asus B360M-A (CFL+CNP), so as a last resort fall
// back to checking for a VGA branch device. Only do this
// on known affected platforms to minimize false positives.
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut bug: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug) as *mut *mut bug_entry;
let mut __v4: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v4) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut drm_edid: core::mem::MaybeUninit<*const drm_edid> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *const drm_edid = core::ptr::addr_of_mut!(drm_edid) as *mut *const drm_edid;
let mut has_dpcd: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut bool = core::ptr::addr_of_mut!(has_dpcd) as *mut bool;
let mut encoder: *mut intel_encoder = unsafe { core::mem::zeroed() };
let mut fixed_mode: core::mem::MaybeUninit<*mut drm_display_mode> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut drm_display_mode = core::ptr::addr_of_mut!(fixed_mode) as *mut *mut drm_display_mode;
let mut display: *mut intel_display = unsafe { core::mem::zeroed() };
unsafe {
display = to_intel_display__2!(intel_dp);
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
encoder = core::ptr::addr_of_mut!((*dp_to_dig_port(intel_dp)).base);
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_3 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if !intel_dp_is_edp(intel_dp) {
return true;
} else {
// On IBX/CPT we may get here with LVDS already registered. Since the
// driver uses the only internal power sequencer available for both
// eDP and LVDS bail out early in this case to prevent interfering
// with an already powered-on LVDS power sequencer.
if !intel_get_lvds_encoder(display).is_null() {
let __ret_warn_on: i32 = !(HAS_PCH_IBX!(display) || HAS_PCH_CPT!(display)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!((((display)->pch_type) == PCH_IBX) || (((display)->pch_type) == PCH_CPT)))".as_ptr() as *mut u8);
*__slate_slot_4 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_edp_init_connector_15, sym __slate_asm_string_intel_edp_init_connector_16, const 7001 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_2 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
(*__slate_slot_4).unwrap()(*__slate_slot_2, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!((((display)->pch_type) == PCH_IBX) || (((display)->pch_type) == PCH_CPT)))".as_ptr() as *mut u8);
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
while false {
}
_dev_info(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] LVDS was detected, not registering eDP\n".as_ptr() as *const u8);
return false;
} else {
'__join_0: {
intel_bios_init_panel_early(display, core::ptr::addr_of_mut!((*connector).panel), (*encoder).devdata);
if !intel_pps_init(intel_dp) {
while false {
}
_dev_info(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] [ENCODER:%d:%s] unusable PPS, disabling eDP\n".as_ptr() as *const u8, (*encoder).base.base.id, (*encoder).base.name);
// The BIOS may have still enabled VDD on the PPS even
// though it's unusable. Make sure we turn it back off
// and to release the power domain references/etc.
} else {
// Enable HPD sense for live status check.
// intel_hpd_irq_setup() will turn it off again
// if it's no longer needed later.
//
// The DPCD probe below will make sure VDD is on.
intel_hpd_enable_detection(encoder);
intel_alpm_init(intel_dp);
// Cache DPCD and EDID for edp.
*__slate_slot_9 = intel_edp_init_dpcd(intel_dp, connector);
if !(*__slate_slot_9) {
// if this fails, presume the device is a ghost
while false {
}
_dev_info(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] [ENCODER:%d:%s] failed to retrieve link info, disabling eDP\n".as_ptr() as *const u8, (*encoder).base.base.id, (*encoder).base.name);
} else {
// VBT and straps are liars. Also check HPD as that seems
// to be the most reliable piece of information available.
//
// ... expect on devices that forgot to hook HPD up for eDP
// (eg. Acer Chromebook C710), so we'll check it only if multiple
// ports are attempting to use the same AUX CH, according to VBT.
if intel_bios_dp_has_shared_aux_ch((*encoder).devdata) {
// If this fails, presume the DPCD answer came
// from some other port using the same AUX CH.
//
// FIXME maybe cleaner to check this before the
// DPCD read? Would need sort out the VDD handling...
if !intel_digital_port_connected(encoder) {
while false {
}
_dev_info(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] [ENCODER:%d:%s] HPD is down, disabling eDP\n".as_ptr() as *const u8, (*encoder).base.base.id, (*encoder).base.name);
break '__join_0;
} else {
if (DISPLAY_VER!(display) as i32) == 9 {
__v1 = drm_dp_is_branch(((*intel_dp).dpcd.as_mut_ptr() as *mut u8) as *const u8);
} else {
__v1 = false;
}
if __v1 && (*((*intel_dp).dpcd.as_mut_ptr() as *mut u8).wrapping_offset(5 as isize) as i32) & 6 == 2i32 {
while false {
}
_dev_info(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] [ENCODER:%d:%s] VGA converter detected, disabling eDP\n".as_ptr() as *const u8, (*encoder).base.base.id, (*encoder).base.name);
break '__join_0;
}
}
}
mutex_lock(core::ptr::addr_of_mut!((*(*display).drm).mode_config.mutex));
*__slate_slot_8 = drm_edid_read_ddc(core::ptr::addr_of_mut!((*connector).base), (*connector).base.ddc);
if (*__slate_slot_8).is_null() {
// Fallback to EDID from ACPI OpRegion, if any
*__slate_slot_8 = intel_opregion_get_edid(connector);
if !(*__slate_slot_8).is_null() {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[CONNECTOR:%d:%s] Using OpRegion EDID\n".as_ptr() as *const u8, (*connector).base.base.id, (*connector).base.name);
}
}
if !(*__slate_slot_8).is_null() {
if drm_edid_connector_update(core::ptr::addr_of_mut!((*connector).base), *__slate_slot_8) != 0 {
__v0 = true;
} else {
__v0 = drm_edid_connector_add_modes(core::ptr::addr_of_mut!((*connector).base)) == 0;
}
if __v0 {
drm_edid_connector_update(core::ptr::addr_of_mut!((*connector).base), core::ptr::null::<drm_edid>());
drm_edid_free(*__slate_slot_8);
*__slate_slot_8 = ERR_PTR(22i32.wrapping_neg() as i64) as *const drm_edid;
}
} else {
*__slate_slot_8 = ERR_PTR(2i32.wrapping_neg() as i64) as *const drm_edid;
}
intel_bios_init_panel_late(display, core::ptr::addr_of_mut!((*connector).panel), (*encoder).devdata, if IS_ERR(*__slate_slot_8 as *const ()) { core::ptr::null::<drm_edid>() } else { *__slate_slot_8 });
intel_panel_add_edid_fixed_modes(connector, true);
// MSO requires information from the EDID
intel_edp_mso_init(intel_dp);
// multiply the mode clock and horizontal timings for MSO
*__slate_slot_11 = list_first_entry__4!(core::ptr::addr_of_mut!((*connector).panel.fixed_modes));
while list_entry_is_head!(*__slate_slot_11, core::ptr::addr_of_mut!((*connector).panel.fixed_modes), head) == 0 {
intel_edp_mso_mode_fixup(connector, *__slate_slot_11);
*__slate_slot_11 = list_next_entry__3!(*__slate_slot_11, head);
}
// fallback to VBT if available for eDP
if intel_panel_preferred_fixed_mode(connector).is_null() {
intel_panel_add_vbt_lfp_fixed_mode(connector);
}
mutex_unlock(core::ptr::addr_of_mut!((*(*display).drm).mode_config.mutex));
if intel_panel_preferred_fixed_mode(connector).is_null() {
while false {
}
_dev_info(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] [ENCODER:%d:%s] failed to find fixed mode for the panel, disabling eDP\n".as_ptr() as *const u8, (*encoder).base.base.id, (*encoder).base.name);
} else {
intel_panel_init(connector, *__slate_slot_8);
intel_edp_backlight_setup(intel_dp, connector);
intel_edp_add_properties(intel_dp);
intel_pps_init_late(intel_dp);
return true;
}
}
}
}
intel_pps_vdd_off_sync(intel_dp);
intel_bios_fini_panel(core::ptr::addr_of_mut!((*connector).panel));
return false;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_888: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn intel_dp_init_connector(dig_port: *mut intel_digital_port, connector: *mut intel_connector) -> bool {
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut bug: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug) as *mut *mut bug_entry;
let mut __v4: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v4) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut bug_1: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug_1) as *mut *mut bug_entry;
let mut __v10: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v10) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut bug: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug) as *mut *mut bug_entry;
let mut __v16: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v16) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut __ret_warn_on_1: i32 = unsafe { core::mem::zeroed() };
let mut r#type: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut i32 = core::ptr::addr_of_mut!(r#type) as *mut i32;
let mut port: i32 = unsafe { core::mem::zeroed() };
let mut dev: *mut drm_device = unsafe { core::mem::zeroed() };
let mut encoder: *mut intel_encoder = unsafe { core::mem::zeroed() };
let mut intel_dp: *mut intel_dp = unsafe { core::mem::zeroed() };
let mut display: *mut intel_display = unsafe { core::mem::zeroed() };
unsafe {
display = to_intel_display__6!(dig_port);
intel_dp = core::ptr::addr_of_mut!((*dig_port).dp);
encoder = core::ptr::addr_of_mut!((*dig_port).base);
dev = (*encoder).base.dev;
port = (*encoder).port;
let __slate_auto_bytes_1 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
__ret_warn_on_1 = (((*dig_port).max_lanes as i32) < 1) as i32;
if unlikely!(__ret_warn_on_1) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"%s %s: [drm] Not enough lanes (%d) for DP on [ENCODER:%d:%s]\n".as_ptr() as *const u8, dev_driver_string(__drm_to_dev(dev as *const drm_device) as *const device), dev_name(__drm_to_dev(dev as *const drm_device) as *const device), (*dig_port).max_lanes as i32, (*encoder).base.base.id, (*encoder).base.name);
*__slate_slot_16 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t2: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t2, sym __slate_asm_string_intel_dp_init_connector_30, sym __slate_asm_string_intel_dp_init_connector_31, const 7161 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_14 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t2 as u64) };
(*__slate_slot_16).unwrap()(*__slate_slot_14, dev_driver_string(__drm_to_dev(dev as *const drm_device) as *const device), dev_name(__drm_to_dev(dev as *const drm_device) as *const device), (*dig_port).max_lanes as i32, (*encoder).base.base.id, (*encoder).base.name);
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
let __v19: i64 = (__ret_warn_on_1 != 0) as i64;
if __v19 != 0 {
return false;
} else {
'__join_17: {
(*intel_dp).reset_link_params = true;
// Preserve the current hw state.
(*intel_dp).DP = intel_de_read(display, (*intel_dp).output_reg);
(*intel_dp).attached_connector = connector;
if _intel_dp_is_port_edp(display, (*encoder).devdata, port) {
{
let __ret_warn_on: i32 = (intel_encoder_is_tc(encoder) && (DISPLAY_VER!(display) as i32) < 30) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev(dev as *const drm_device) as *const device), dev_name(__drm_to_dev(dev as *const drm_device) as *const device), c"drm_WARN_ON(intel_encoder_is_tc(encoder) && ((&(display)->info.__runtime_info)->ip.ver) < 30)".as_ptr() as *mut u8);
*__slate_slot_10 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t1: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t1, sym __slate_asm_string_intel_dp_init_connector_28, sym __slate_asm_string_intel_dp_init_connector_29, const 7176 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_8 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t1 as u64) };
(*__slate_slot_10).unwrap()(*__slate_slot_8, dev_driver_string(__drm_to_dev(dev as *const drm_device) as *const device), dev_name(__drm_to_dev(dev as *const drm_device) as *const device), c"drm_WARN_ON(intel_encoder_is_tc(encoder) && ((&(display)->info.__runtime_info)->ip.ver) < 30)".as_ptr() as *mut u8);
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
}
// Currently we don't support eDP on TypeC ports for DISPLAY_VER < 30,
// although in theory it could work on TypeC legacy ports.
*__slate_slot_20 = 14;
(*encoder).r#type = 8;
__ret_warn_on = ((((*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_valleyview() as i32) != 0 || ((*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_cherryview() as i32) != 0) && (port as i32) != 1 && (port as i32) != 2) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev(dev as *const drm_device) as *const device), dev_name(__drm_to_dev(dev as *const drm_device) as *const device), c"drm_WARN_ON((display->platform.valleyview || display->platform.cherryview) && port != PORT_B && port != PORT_C)".as_ptr() as *mut u8);
*__slate_slot_4 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_dp_init_connector_26, sym __slate_asm_string_intel_dp_init_connector_27, const 7183 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_2 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
(*__slate_slot_4).unwrap()(*__slate_slot_2, dev_driver_string(__drm_to_dev(dev as *const drm_device) as *const device), dev_name(__drm_to_dev(dev as *const drm_device) as *const device), c"drm_WARN_ON((display->platform.valleyview || display->platform.cherryview) && port != PORT_B && port != PORT_C)".as_ptr() as *mut u8);
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
// eDP only on port B and/or C on vlv/chv
if __ret_warn_on != 0 {
return false;
} else {
break '__join_17;
}
} else {
*__slate_slot_20 = 10;
}
}
intel_dp_set_default_sink_rates(intel_dp);
intel_dp_set_default_max_sink_lane_count(intel_dp);
if ((*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_valleyview() as i32) != 0 || ((*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_cherryview() as i32) != 0 {
vlv_pps_pipe_init(intel_dp);
}
intel_dp_aux_init(intel_dp);
(*connector).dp.dsc_decompression_aux = core::ptr::addr_of_mut!((*intel_dp).aux);
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Adding %s connector on [ENCODER:%d:%s]\n".as_ptr() as *const u8, if *__slate_slot_20 == 14 { c"eDP".as_ptr() as *mut u8 } else { c"DP".as_ptr() as *mut u8 }, (*encoder).base.base.id, (*encoder).base.name);
drm_connector_init_with_ddc(dev, core::ptr::addr_of_mut!((*connector).base), core::ptr::addr_of!(intel_dp_connector_funcs.0), *__slate_slot_20, core::ptr::addr_of_mut!((*intel_dp).aux.ddc));
drm_connector_helper_add(core::ptr::addr_of_mut!((*connector).base), core::ptr::addr_of!(intel_dp_connector_helper_funcs.0));
if HAS_GMCH!(display) == 0 && (DISPLAY_VER!(display) as i32) < 12 {
(*connector).base.interlace_allowed = true;
}
if *__slate_slot_20 != 14 {
(*connector).polled = 1u8;
}
(*connector).base.polled = (*connector).polled;
intel_connector_attach_encoder(connector, encoder);
if HAS_DDI!(display) != 0 {
(*connector).get_hw_state = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut intel_connector) -> bool>>(intel_ddi_connector_get_hw_state as *const ()) };
} else {
(*connector).get_hw_state = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut intel_connector) -> bool>>(intel_connector_get_hw_state as *const ()) };
}
(*connector).sync_state = Some(intel_dp_connector_sync_state);
if !intel_edp_init_connector(intel_dp, connector) {
intel_dp_aux_fini(intel_dp);
intel_display_power_flush_work(display);
drm_connector_cleanup(core::ptr::addr_of_mut!((*connector).base));
return false;
} else {
intel_dp_set_source_rates(intel_dp);
intel_dp_set_common_link_params(intel_dp);
intel_dp_reset_link_params(intel_dp);
// init MST on ports that can support it
intel_dp_mst_encoder_init(dig_port, (*connector).base.base.id as i32);
intel_dp_add_properties(intel_dp, core::ptr::addr_of_mut!((*connector).base));
if is_hdcp_supported(display, port) {
__v1 = !intel_dp_is_edp(intel_dp);
} else {
__v1 = false;
}
if __v1 {
let ret: i32 = intel_dp_hdcp_init(dig_port, connector);
if ret != 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"HDCP init failed, skipping.\n".as_ptr() as *const u8);
}
}
(*intel_dp).frl.is_trained = false;
(*intel_dp).frl.trained_rate_gbps = 0;
intel_psr_init(intel_dp);
return true;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_889: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_890: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_891: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn intel_dp_cleanup_connector(dig_port: *mut intel_digital_port, connector: *mut intel_connector) {
let display: *mut intel_display = to_intel_display!(connector);
let intel_dp: *mut intel_dp = unsafe { core::ptr::addr_of_mut!((*dig_port).dp) };
unsafe { intel_display_power_flush_work(display) };
unsafe { intel_dp_mst_encoder_cleanup(dig_port) };
unsafe { intel_dp_aux_fini(intel_dp) };
unsafe { drm_connector_cleanup(core::ptr::addr_of_mut!((*connector).base)) };
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_mst_suspend(display: *mut intel_display) {
let mut encoder: *mut intel_encoder = unsafe { core::mem::zeroed() };
if !HAS_DISPLAY!(display) {
return;
}
let mut __v0: *mut intel_encoder;
__v0 = ((unsafe { (((*core::ptr::addr_of_mut!((*(*display).drm).mode_config.encoder_list)).next as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *mut intel_encoder;
encoder = __v0;
while list_is_head((unsafe { core::ptr::addr_of_mut!((*encoder).base.head) }) as *const list_head, (unsafe { core::ptr::addr_of_mut!((*(*display).drm).mode_config.encoder_list) }) as *const list_head) == 0 {
let mut intel_dp: *mut intel_dp = unsafe { core::mem::zeroed() };
if ((unsafe { (*encoder).r#type }) as u32) == 10 {
intel_dp = enc_to_intel_dp(encoder);
if unsafe { intel_dp_mst_source_support(intel_dp) } {
if unsafe { (*intel_dp).is_mst } {
unsafe { drm_dp_mst_topology_mgr_suspend(core::ptr::addr_of_mut!((*intel_dp).mst.mgr)) };
}
}
}
let mut __v2: *mut intel_encoder;
__v2 = ((unsafe { (((*encoder).base.head.next as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *mut intel_encoder;
encoder = __v2;
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_mst_resume(display: *mut intel_display) {
let mut encoder: *mut intel_encoder = unsafe { core::mem::zeroed() };
if !HAS_DISPLAY!(display) {
return;
}
let mut __v0: *mut intel_encoder;
__v0 = ((unsafe { (((*core::ptr::addr_of_mut!((*(*display).drm).mode_config.encoder_list)).next as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *mut intel_encoder;
encoder = __v0;
while list_is_head((unsafe { core::ptr::addr_of_mut!((*encoder).base.head) }) as *const list_head, (unsafe { core::ptr::addr_of_mut!((*(*display).drm).mode_config.encoder_list) }) as *const list_head) == 0 {
let mut intel_dp: *mut intel_dp = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
if ((unsafe { (*encoder).r#type }) as u32) == 10 {
intel_dp = enc_to_intel_dp(encoder);
if unsafe { intel_dp_mst_source_support(intel_dp) } {
ret = unsafe { drm_dp_mst_topology_mgr_resume(core::ptr::addr_of_mut!((*intel_dp).mst.mgr), true) };
if ret != 0 {
unsafe {
(*intel_dp).is_mst = false;
}
unsafe { drm_dp_mst_topology_mgr_set_mst(core::ptr::addr_of_mut!((*intel_dp).mst.mgr), false) };
}
}
}
let mut __v2: *mut intel_encoder;
__v2 = ((unsafe { (((*encoder).base.head.next as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *mut intel_encoder;
encoder = __v2;
}
}

fn intel_dp_sdp_compute_config_late(crtc_state: *mut intel_crtc_state) -> i32 {
let display: *mut intel_display = to_intel_display__5!(crtc_state);
let guardband: i32 = unsafe { intel_crtc_vblank_length(crtc_state as *const intel_crtc_state) };
let min_sdp_guardband: i32 = intel_dp_sdp_min_guardband(crtc_state as *const intel_crtc_state, false);
if guardband < min_sdp_guardband {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"guardband %d < min sdp guardband %d\n".as_ptr() as *const u8, guardband, min_sdp_guardband) };
return 22i32.wrapping_neg();
}
0
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_compute_config_late(encoder: *mut intel_encoder, crtc_state: *mut intel_crtc_state, conn_state: *mut drm_connector_state) -> i32 {
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
let mut ret: i32 = 0;
unsafe { intel_psr_compute_config_late(intel_dp, crtc_state) };
ret = intel_dp_sdp_compute_config_late(crtc_state);
if ret != 0 {
return ret;
}
unsafe { intel_alpm_lobf_compute_config_late(intel_dp, crtc_state) };
0
}

fn intel_dp_get_lines_for_sdp(crtc_state: *const intel_crtc_state, r#type: u32) -> i32 {
match r#type {
32 | 33 => {
return 10;
}
10 => {
return 8;
}
16 => {
return 7;
}
34 => {
return unsafe { (*crtc_state).vrr.vsync_start }.wrapping_add(1) as i32;
}
_ => {
}
}
0
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_joiner_candidate_valid(connector: *mut intel_connector, hdisplay: i32, num_joined_pipes: i32) -> bool {
let display: *mut intel_display = to_intel_display!(connector);
let intel_dp: *mut intel_dp = intel_attached_dp(connector);
if !intel_dp_can_join(intel_dp, num_joined_pipes) {
return false;
}
if hdisplay > num_joined_pipes.wrapping_mul(intel_dp_max_hdisplay_per_pipe(display)) {
return false;
}
if (unsafe { (*connector).force_joined_pipes }) != 0 && (unsafe { (*connector).force_joined_pipes }) != num_joined_pipes {
return false;
}
true
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_as_sdp_transmission_time() -> u8 {
// DP allows AS SDP position to move during PR active in some cases, but
// software-controlled refresh rate changes with DC6v / ALPM require the
// AS SDP to remain at T1. Use T1 unconditionally for now.
0u8
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_link_init(intel_dp: *mut intel_dp) -> i32 {
unsafe {
(*intel_dp).link.training = intel_dp_link_training_init(intel_dp);
}
if unsafe { (*intel_dp).link.training }.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*intel_dp).link.caps = intel_dp_link_caps_init(intel_dp);
}
if unsafe { (*intel_dp).link.caps }.is_null() {
unsafe { intel_dp_link_training_cleanup((*intel_dp).link.training) };
return 12i32.wrapping_neg();
}
0
}

#[unsafe(no_mangle)]
extern "C" fn intel_dp_link_cleanup(intel_dp: *mut intel_dp) {
unsafe { intel_dp_link_caps_cleanup((*intel_dp).link.caps) };
unsafe { intel_dp_link_training_cleanup((*intel_dp).link.training) };
}

static __slate_asm_string_align_max_compressed_bpp_x16_6: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_align_max_compressed_bpp_x16_7: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_align_min_compressed_bpp_x16_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_align_min_compressed_bpp_x16_5: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_dfp_can_convert_3: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_dfp_can_convert_4: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_intel_dp_check_mst_status_14: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_check_mst_status_15: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_check_mst_status_8: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_check_mst_status_9: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_compute_min_hblank_10: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_compute_min_hblank_9: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_compute_vsc_colorimetry_6: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_compute_vsc_colorimetry_7: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_configure_protocol_converter_12: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_intel_dp_configure_protocol_converter_13: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_configure_protocol_converter_5: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_intel_dp_configure_protocol_converter_6: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_connector_sync_state_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_connector_sync_state_5: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_detect_17: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_detect_18: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_detect_dpcd_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_detect_dpcd_5: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_dsc_aux_ref_count_10: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_dsc_aux_ref_count_9: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_dsc_max_delta_bppx16_4: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_intel_dp_dsc_max_delta_bppx16_5: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_dsc_max_sink_compressed_bppx16_3: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_intel_dp_dsc_max_sink_compressed_bppx16_4: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_dsc_sink_min_compressed_bpp_3: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_intel_dp_dsc_sink_min_compressed_bpp_4: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_get_active_pipes_7: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_get_active_pipes_8: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_get_common_rates_10: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_get_common_rates_11: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_get_common_rates_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_get_common_rates_5: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_handle_device_service_irq_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_handle_device_service_irq_5: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_handle_link_service_irq_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_handle_link_service_irq_5: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_init_connector_26: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_init_connector_27: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_init_connector_28: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_init_connector_29: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_init_connector_30: [u8; 62] = *b"%s %s: [drm] Not enough lanes (%d) for DP on [ENCODER:%d:%s]\n\0";

static __slate_asm_string_intel_dp_init_connector_31: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_link_bw_overhead_1: [u8; 1] = *b"\0";

static __slate_asm_string_intel_dp_link_bw_overhead_2: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_output_format_8: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_output_format_9: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_rate_select_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_rate_select_5: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_set_source_rates_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_set_source_rates_5: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_sink_disable_decompression_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_sink_disable_decompression_5: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_sink_enable_decompression_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_dp_sink_enable_decompression_5: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_sink_format_valid_5: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_intel_dp_sink_format_valid_6: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_dp_vsc_sdp_unpack_4: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_intel_dp_vsc_sdp_unpack_5: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_edp_init_connector_15: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_edp_init_connector_16: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_edp_init_dpcd_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_edp_init_dpcd_5: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_modeset_affected_transcoders_10: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_modeset_affected_transcoders_9: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_read_dp_sdp_4: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_intel_read_dp_sdp_5: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_write_dp_sdp_11: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_write_dp_sdp_12: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intel_write_dp_sdp_5: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_intel_write_dp_sdp_6: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_intersect_rates_1: [u8; 1] = *b"\0";

static __slate_asm_string_intersect_rates_2: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_seq_buf_str_1: [u8; 1] = *b"\0";

static __slate_asm_string_seq_buf_str_2: [u8; 53] = *b"/home/takashi/c-corpus/linux/include/linux/seq_buf.h\0";

static __slate_asm_string_source_can_output_4: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_source_can_output_5: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";

static __slate_asm_string_wait_for_connector_hw_done_5: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_wait_for_connector_hw_done_6: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_dp.c\0";
