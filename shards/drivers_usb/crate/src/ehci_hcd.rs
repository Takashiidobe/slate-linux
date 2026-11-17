
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
//! SPDX-License-Identifier: GPL-2.0+
//!
//! Enhanced Host Controller Interface (EHCI) driver for USB.
//!
//! Maintainer: Alan Stern <stern@rowland.harvard.edu>
//!
//! Copyright (c) 2000-2004 by David Brownell
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
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
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
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! @a is a power of 2 value
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
//! ignore signal
//! bit-flags
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
//! number of env pointers
//! Generic info of form tag = "info"
//! Generate SD flag indexes
//! See the comment in asm-generic/io.h about ioremap_np().
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! ***DANGEROUS*** don't add guard page
//! USB directions
//!
//! This bit flag is used in endpoint descriptors' bEndpointAddress field.
//! It's also one of three fields in control requests bRequestType.
//! Runtime PM flag argument bits
//! This is arbitrary.
//! From USB 2.0 spec Table 11-13, offset 7, a hub can
//! have up to 255 ports. The most yet reported is 10.
//! Upcoming hardware might raise that limit.
//! Because the arrays need to add a bit for hub status data, we
//! use 31, so plus one evens out to four bytes.
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
//! __get_unaligned_t - read an unaligned value from memory.
//! @type:	the type to load from the pointer.
//! @ptr:	the pointer to load from.
//!
//! Use memcpy to affect an unaligned type sized load avoiding undefined behavior
//! from approaches like type punning that require -fno-strict-aliasing in order
//! to be correct. As type may be const, use __unqual_scalar_typeof to map to a
//! non-const type - you can't memcpy into a const type. The
//! __get_unaligned_ctrl_type gives __unqual_scalar_typeof its required
//! expression rather than type, a pointer is used to avoid warnings about mixing
//! the use of 0 and NULL. The void* cast silences ubsan warnings.
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
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
// N 32-bit words worth of info
// N 32-bit bug flags
// "pse" Page Size Extensions
// "tsc" Time Stamp Counter
// "apic" Onboard APIC
// "sse"
// "pdpe1gb" GB pages
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
// File too large
// No space left on device
// Broken pipe
// Math result not representable
// Invalid system call number
// Out of streams resources
// Communication error on send
// Protocol error
// Value too large for defined data type
// Message too long
// Operation not supported on transport endpoint
// Cannot send after transport endpoint shutdown
// Connection timed out
// Operation now in progress
// Remote I/O error
// Memory page has hardware error
// Operation is not supported
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
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
// Generate SD flag bits
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
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
// Error values that may be returned by PCI functions
// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// These helpers provide future and backwards compatibility
// for accessing popular PCI BAR info
// Request is asynchronous
// Increment/decrement the
// usage_count
// Use autosuspend_delay
// Succeed if runtime PM is disabled
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
macro_rules! ACTIVE_BIT {
($ehci:expr) => {
cpu_to_hc32(($ehci) as *const ehci_hcd, 128u32)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__13 {
() => {
((core::mem::size_of::<[*mut platform_driver; 0]>() as u64) / (core::mem::size_of::<*mut platform_driver>() as u64)).wrapping_add(__must_be_array__17!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__26 {
() => {
((core::mem::size_of::<[*mut platform_driver; 0]>() as u64) / (core::mem::size_of::<*mut platform_driver>() as u64)).wrapping_add(__must_be_array__12!() as u64)
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

/// with integer truncation
macro_rules! BitTime {
($bytecount:expr) => {
56i32.wrapping_mul($bytecount) / 6
};
}

/// with integer truncation
macro_rules! BitTime__2 {
($bytecount:expr) => {
56u32.wrapping_mul($bytecount) / 6
};
}

macro_rules! EHCI_ITD_LENGTH {
($tok:expr) => {
($tok) >> 16i32 & 4095
};
}

/// for periodic/async schedules and qtd lists, mark end of list
/// "null pointer" to hw
macro_rules! EHCI_LIST_END {
($ehci:expr) => {
cpu_to_hc32(($ehci) as *const ehci_hcd, 1u32)
};
}

macro_rules! GENMASK_INPUT_CHECK {
() => {
core::mem::size_of::<__SlateRecord_find_h_67>() as i32
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

macro_rules! HALT_BIT {
($ehci:expr) => {
cpu_to_hc32(($ehci) as *const ehci_hcd, 64u32)
};
}

macro_rules! HCC_64BIT_ADDR {
($p:expr) => {
($p) & 1
};
}

macro_rules! HCC_CANPARK {
($p:expr) => {
($p) & 4u32
};
}

macro_rules! HCC_ISOC_CACHE {
($p:expr) => {
($p) & 128u32
};
}

macro_rules! HCC_ISOC_THRES {
($p:expr) => {
($p) >> 4i32 & 7
};
}

macro_rules! HCC_PER_PORT_CHANGE_EVENT {
($p:expr) => {
($p) & 262144u32
};
}

macro_rules! HCC_PGM_FRAMELISTLEN {
($p:expr) => {
($p) & 2u32
};
}

macro_rules! HCD_HW_ACCESSIBLE {
($hcd:expr) => {
(unsafe { (*$hcd).flags }) & 1u64
};
}

macro_rules! HCD_WAKEUP_PENDING {
($hcd:expr) => {
(unsafe { (*$hcd).flags }) & 16u64
};
}

macro_rules! HCS_DEBUG_PORT {
($p:expr) => {
($p) >> 20i32 & 15
};
}

macro_rules! HCS_N_PORTS {
($p:expr) => {
($p) >> 0i32 & 15
};
}

macro_rules! HCS_PPC {
($p:expr) => {
($p) & 16u32
};
}

macro_rules! HC_LENGTH {
($p:expr) => {
255 & ($p) >> if ehci_big_endian_capbase!() != 0 { 24i32 } else { 0i32 }
};
}

macro_rules! HC_VERSION {
($p:expr) => {
65535 & ($p) >> if ehci_big_endian_capbase!() != 0 { 0i32 } else { 16i32 }
};
}

macro_rules! HS_NSECS_ISO__2 {
($bytes:expr) => {
(633232u64.wrapping_add(2083u64.wrapping_mul(3u32.wrapping_add(BitTime__2!($bytes)) as u64)) / 1000).wrapping_add(5)
};
}

macro_rules! HS_NSECS_ISO {
($bytes:expr) => {
(633232u64.wrapping_add(2083u64.wrapping_mul(3i32.wrapping_add(BitTime!($bytes)) as u64)) / 1000).wrapping_add(5)
};
}

macro_rules! HS_NSECS {
($bytes:expr) => {
(916520u64.wrapping_add(2083u64.wrapping_mul(3i32.wrapping_add(BitTime!($bytes)) as u64)) / 1000).wrapping_add(5)
};
}

macro_rules! HS_USECS_ISO {
($bytes:expr) => {
NS_TO_US!(HS_NSECS_ISO__2!($bytes))
};
}

macro_rules! HS_USECS_ISO__2 {
($bytes:expr) => {
NS_TO_US!(HS_NSECS_ISO!($bytes))
};
}

macro_rules! HS_USECS {
($bytes:expr) => {
NS_TO_US!(HS_NSECS!($bytes))
};
}

/// class requests from the USB 2.0 hub spec, table 11-15
macro_rules! HUB_CLASS_REQ {
($dir:expr, $type:expr, $request:expr) => {
(($dir) | $type) << 8i32 | $request
};
}

macro_rules! INCR {
() => {
while false {
}
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

macro_rules! IS_SHORT_READ {
($token:expr) => {
QTD_LENGTH!($token) != 0 && QTD_PID!($token) == 1
};
}

macro_rules! ITD_ACTIVE {
($ehci:expr) => {
cpu_to_hc32(($ehci) as *const ehci_hcd, 2147483648u32)
};
}

macro_rules! NS_TO_US {
($ns:expr) => {
$ns.wrapping_add(1000).wrapping_sub(1) / 1000
};
}

macro_rules! NS_TO_US__2 {
($ns:expr) => {
$ns.wrapping_add(1000).wrapping_sub(1) / 1000
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! PORT_TEST {
($x:expr) => {
(($x) & 15) << 16i32
};
}

macro_rules! PORT_USB11 {
($x:expr) => {
($x) & 3072u32 == 1024u32
};
}

/// next async queue entry, or pointer to interrupt/periodic QH
macro_rules! QH_NEXT {
($ehci:expr, $dma:expr) => {
cpu_to_hc32(($ehci) as *const ehci_hcd, (($dma) as u32) & 4294967264u32 | 2u32)
};
}

macro_rules! QTD_CERR {
($tok:expr) => {
($tok) >> 10i32 & 3
};
}

macro_rules! QTD_LENGTH {
($tok:expr) => {
($tok) >> 16i32 & 32767
};
}

macro_rules! QTD_NEXT {
($ehci:expr, $dma:expr) => {
cpu_to_hc32(($ehci) as *const ehci_hcd, ($dma) as u32)
};
}

macro_rules! QTD_PID {
($tok:expr) => {
($tok) >> 8i32 & 3
};
}

/// type tag from {qh,itd,sitd,fstn}->hw_next
macro_rules! Q_NEXT_TYPE {
($ehci:expr, $dma:expr) => {
($dma) & cpu_to_hc32(($ehci) as *const ehci_hcd, 6u32)
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! SITD_ACTIVE {
($ehci:expr) => {
cpu_to_hc32(($ehci) as *const ehci_hcd, 128u32)
};
}

macro_rules! SITD_LENGTH {
($x:expr) => {
($x) >> 16i32 & 1023
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! USB_EP_MAXP_MULT {
($m:expr) => {
(($m) & 6144i32) >> 11i32
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

macro_rules! __cmp_once__4 {
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

macro_rules! __le16_to_cpu {
($x:expr) => {
($x) as u16
};
}

macro_rules! __must_be_array__3 {
() => {
core::mem::size_of::<__SlateRecord_ehci_sched_c_1169_4>() as i32
};
}

macro_rules! __must_be_array__12 {
() => {
core::mem::size_of::<__SlateRecord_ehci_hcd_c_1396>() as i32
};
}

macro_rules! __must_be_array__17 {
() => {
core::mem::size_of::<__SlateRecord_ehci_hcd_c_1366>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

/// __put_unaligned_t - write an unaligned value to memory.
/// @type:	the type of the value to store.
/// @val:	the value to store.
/// @ptr:	the pointer to store to.
///
/// Use memcpy to affect an unaligned type sized store avoiding undefined
/// behavior from approaches like type punning that require -fno-strict-aliasing
/// in order to be correct. The void* cast silences ubsan warnings.
macro_rules! __put_unaligned_t {
($type:ty, $val:expr, $ptr:expr) => {
loop {
let mut __put_unaligned_val: $type = $val;
unsafe { __builtin_memcpy($ptr, core::ptr::addr_of_mut!(__put_unaligned_val) as *const (), core::mem::size_of::<$type>() as u64) };
if true {
break;
}
}
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
if is_signed_type__3!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u32>() }) as u64) < 4) as i32)) }
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

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__2!($uy)
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

/// check_add_overflow() - Calculate addition with overflow checking
/// @a: first addend
/// @b: second addend
/// @d: pointer to store sum
///
/// Returns true on wrap-around, false otherwise.
///
/// *@d holds the results of the attempted addition, regardless of whether
/// wrap-around occurred.
macro_rules! check_add_overflow {
($a:expr, $b:expr, $d:expr) => {
__must_check_overflow({
let __t: i128 = ($a) as i128;
let __t2: i128 = ($b) as i128;
let __t3: i128 = __t.wrapping_add(__t2);
unsafe {
*$d = __t3 as u64;
}
__t.checked_add(__t2).is_none() || ((__t3 as u64) as i128) != __t3
})
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
macro_rules! container_of__8 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__58!($type) as isize)) }) as *mut ()) as *mut $type;
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
__v = ((unsafe { (($ptr) as *const u8).wrapping_offset(-(offsetof__5!() as isize)) }) as *mut ()) as *mut $type;
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
($ptr:expr) => {
{
let mut __v: *mut ehci_itd;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__68!() as isize)) }) as *mut ()) as *mut ehci_itd;
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
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__63!($type) as isize)) }) as *mut ()) as *mut $type;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *mut $type;
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
let mut __v: *mut ehci_per_sched;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__79!() as isize)) }) as *mut ()) as *mut ehci_per_sched;
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
macro_rules! container_of__53 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__71!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__55 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__74!() as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__59 {
($ptr:expr) => {
{
let mut __v: *mut ehci_qh;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__37!() as isize)) }) as *mut ()) as *mut ehci_qh;
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
macro_rules! container_of__63 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__47!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__64 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__57!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__66 {
($ptr:expr) => {
{
let mut __v: *mut ehci_sitd;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__44!() as isize)) }) as *mut ()) as *mut ehci_sitd;
__v
}
};
}

macro_rules! ehci_big_endian_capbase {
() => {
0i32
};
}

macro_rules! ehci_big_endian_mmio {
() => {
0i32
};
}

/// Some Freescale/NXP processors using ChipIdea IP have a bug in which
/// disabling the port (PE is cleared) does not cause PEC to be asserted
/// when frame babble is detected.
macro_rules! ehci_has_ci_pec_bug {
($e:expr, $portsc:expr) => {
((unsafe { (*$e).__slate_bits_1.__get_has_ci_pec_bug() }) as i32) != 0 && (unsafe { (*$e).command }) & 16u32 != 0 && ($portsc) & 8u32 == 0 && ($portsc) & 4u32 == 0
};
}

/// Some Freescale/NXP processors have an erratum (USB A-005697)
/// in which we need to wait for 10ms for bus to enter suspend mode
/// after setting SUSP bit.
macro_rules! ehci_has_fsl_susp_errata {
($e:expr) => {
(unsafe { (*$e).__slate_bits_1.__get_has_fsl_susp_errata() }) as i32
};
}

macro_rules! ehci_is_TDI {
() => {
0i32
};
}

macro_rules! ehci_port_speed {
() => {
1024i32
};
}

macro_rules! ehci_prepare_ports_for_controller_resume {
($ehci:expr) => {
ehci_adjust_port_wakeup_flags($ehci, false, false);
};
}

macro_rules! ehci_prepare_ports_for_controller_suspend {
($ehci:expr, $do_wakeup:expr) => {
ehci_adjust_port_wakeup_flags($ehci, true, $do_wakeup);
};
}

/// flex_array_size() - Calculate size of a flexible array member
///                     within an enclosing structure.
/// @p: Pointer to the structure.
/// @member: Name of the flexible array member.
/// @count: Number of elements in the array.
///
/// Calculates size of a flexible array of @count number of @member
/// elements, at the end of structure @p.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! flex_array_size__2 {
($count:expr) => {
size_mul($count, (core::mem::size_of::<ehci_iso_packet>() as u64).wrapping_add(__must_be_array__3!() as u64))
};
}

/// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
/// @n: parameter
///
/// constant-capable log of base 2 calculation
/// - this can be used to initialise global variables from constant data, hence
/// the massive ternary operator construction
///
/// selects the appropriately-sized optimised version depending on sizeof(n)
macro_rules! ilog2__2 {
($n:expr) => {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe($n) } { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { if (core::mem::size_of::<u32>() as u64) <= 4 { __ilog2_u32($n) } else { __ilog2_u64(($n) as u64) } }
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__3 {
() => {
(1i32.wrapping_neg() as u32) < 1
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

macro_rules! kcsan_check_atomic_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 5i32);
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

macro_rules! kmalloc_index {
($s:expr) => {
__kmalloc_index($s, true)
};
}

/// Add two ktime_t variables. res = lhs + rhs:
macro_rules! ktime_add {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_add(($rhs) as i64)
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
macro_rules! list_entry__16 {
($ptr:expr) => {
container_of__59!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__4 {
($ptr:expr, $type:ty) => {
container_of__64!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__10 {
($ptr:expr, $type:ty) => {
container_of__19!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__40 {
($ptr:expr, $type:ty) => {
container_of__53!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__42 {
($ptr:expr, $type:ty) => {
container_of__63!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__11 {
($ptr:expr, $type:ty) => {
container_of__8!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__23 {
($ptr:expr) => {
container_of__44!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__31 {
($ptr:expr) => {
container_of__16!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__34 {
($ptr:expr) => {
container_of__66!($ptr)
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
macro_rules! list_first_entry__15 {
($ptr:expr) => {
list_entry__23!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__16 {
($ptr:expr) => {
list_entry__31!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__17 {
($ptr:expr) => {
list_entry__34!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__23 {
($ptr:expr, $type:ty) => {
list_entry__11!(unsafe { (*$ptr).next }, $type)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__28 {
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
macro_rules! list_first_entry__7 {
($ptr:expr, $type:ty) => {
list_entry__42!(unsafe { (*$ptr).next }, $type)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__29 {
($ptr:expr, $type:ty) => {
list_entry__10!(unsafe { (*$ptr).next }, $type)
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__11 {
($pos:expr, $member:ident) => {
list_entry__34!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__18 {
($pos:expr, $member:ident) => {
list_entry__23!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__15 {
($pos:expr, $member:ident) => {
list_entry__31!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__16 {
($pos:expr, $member:ident) => {
list_entry__16!(unsafe { (*$pos).$member.next })
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

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t__3 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__4!($type, $x, $y)
};
}

macro_rules! offsetof__5 {
() => {
584u64
};
}

macro_rules! offsetof__37 {
() => {
core::mem::offset_of!(ehci_qh, intr_node) as u64
};
}

macro_rules! offsetof__44 {
() => {
core::mem::offset_of!(ehci_sitd, sitd_list) as u64
};
}

macro_rules! offsetof__47 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, unlink_node) as u64
};
}

macro_rules! offsetof__57 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, qtd_list) as u64
};
}

macro_rules! offsetof__58 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, sitd_list) as u64
};
}

macro_rules! offsetof__63 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, itd_list) as u64
};
}

macro_rules! offsetof__68 {
() => {
core::mem::offset_of!(ehci_itd, itd_list) as u64
};
}

macro_rules! offsetof__71 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, intr_node) as u64
};
}

macro_rules! offsetof__74 {
() => {
104u64
};
}

macro_rules! offsetof__2 {
() => {
0u64
};
}

macro_rules! offsetof__79 {
() => {
core::mem::offset_of!(ehci_per_sched, ps_list) as u64
};
}

macro_rules! platform_register_drivers {
($drivers:expr, $count:expr) => {
unsafe { __platform_register_drivers($drivers, ($count) as u32, core::ptr::null_mut::<module>(), c"ehci_hcd".as_ptr() as *const u8) }
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

/// rounddown_pow_of_two - round the given value down to nearest power of two
/// @n: parameter
///
/// round the given value down to the nearest power of two
/// - the result is undefined when n == 0
/// - this can be used to initialise global variables from constant data
macro_rules! rounddown_pow_of_two__2 {
($n:expr) => {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe($n) } { 1u64 << ilog2__2!($n) } else { __rounddown_pow_of_two(($n) as u64) }
};
}

/// These macros should be used after a dma_map_sg call has been done
/// to get bus addresses of each of the SG entries and their lengths.
/// You should only work with the number of sg entries dma_map_sg
/// returns, or alternatively stop on the first sg_dma_len(sg) which
/// is 0.
macro_rules! sg_dma_address {
($sg:expr) => {
unsafe { (*$sg).dma_address }
};
}

macro_rules! sg_dma_len {
($sg:expr) => {
unsafe { (*$sg).dma_length }
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

/// struct_size() - Calculate size of structure with trailing flexible array.
/// @p: Pointer to the structure.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure of @p followed by an
/// array of @count number of @member elements.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size__5 {
($count:expr) => {
size_add(24u64, flex_array_size__2!($count))
};
}

/// struct_size_t() - Calculate size of structure with trailing flexible array
/// @type: structure type name.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure @type followed by an
/// array of @count number of @member elements. Prefer using struct_size()
/// when possible instead, to keep calculations associated with a specific
/// instance variable of type @type.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size_t__4 {
($count:expr) => {
struct_size__5!($count)
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

macro_rules! test_bit__5 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_b52e897e44daaa8($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
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

/// time_after_eq - returns true if the time a is after or the same as time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Return: %true is time a is after or the same as time b, otherwise %false.
macro_rules! time_after_eq {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck__3!(u64) != 0 && typecheck__2!(u64) != 0;
__v && ($a.wrapping_sub($b) as i64) >= 0
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

macro_rules! unlikely__3 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely__5 {
($x:expr) => {
!$x.is_null() as i64
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

macro_rules! unlikely__6 {
($x:expr) => {
(($x) != 0) as i64
};
}

/// The D0/D1 toggle bits ... USE WITH CAUTION (they're almost hcd-internal)
macro_rules! usb_gettoggle {
($dev:expr, $ep:expr, $out:expr) => {
(unsafe { *((*$dev).toggle.as_mut_ptr() as *mut u32).wrapping_offset(($out) as isize) }) >> $ep & 1
};
}

macro_rules! usb_pipecontrol {
($pipe:expr) => {
usb_pipetype!($pipe) == 2
};
}

macro_rules! usb_pipedevice {
($pipe:expr) => {
($pipe) >> 8i32 & 127
};
}

macro_rules! usb_pipeendpoint {
($pipe:expr) => {
($pipe) >> 15i32 & 15
};
}

macro_rules! usb_pipein {
($pipe:expr) => {
($pipe) & 128
};
}

macro_rules! usb_pipeint {
($pipe:expr) => {
usb_pipetype!($pipe) == 1
};
}

macro_rules! usb_pipeout {
($pipe:expr) => {
usb_pipein!($pipe) == 0
};
}

macro_rules! usb_pipetype {
($pipe:expr) => {
($pipe) >> 30i32 & 3
};
}

macro_rules! usb_settoggle {
($dev:expr, $ep:expr, $out:expr, $bit:expr) => {
unsafe {
*((*$dev).toggle.as_mut_ptr() as *mut u32).wrapping_offset(($out) as isize) = *((*$dev).toggle.as_mut_ptr() as *mut u32).wrapping_offset(($out) as isize) & (!(1i32 << $ep) as u32) | ((($bit) << $ep) as u32);
}
};
}

macro_rules! usb_settoggle__2 {
($dev:expr, $ep:expr, $out:expr, $bit:expr) => {
unsafe {
*((*$dev).toggle.as_mut_ptr() as *mut u32).wrapping_offset((($out) as i32) as isize) = *((*$dev).toggle.as_mut_ptr() as *mut u32).wrapping_offset((($out) as i32) as isize) & (!(1i32 << $ep) as u32) | ((($bit) << $ep) as u32);
}
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut jiffies: u64;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut param_ops_int: kernel_param_ops;
    static mut param_ops_uint: kernel_param_ops;
    static mut param_ops_bool: kernel_param_ops;
    static mut ehci_cf_port_reset_rwsem: rw_semaphore;
    static mut kstat: kernel_stat;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __compiletime_assert_0();
    fn __compiletime_assert_1();
    fn __compiletime_assert_2();
    fn __compiletime_assert_5();
    fn __compiletime_assert_6();
    fn __bad_size_call_parameter();
    fn __compiletime_assert_29();
    fn _find_next_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn kstrtouint(arg0: *const u8, arg1: u32, arg2: *mut u32) -> i32;
    fn sscanf(arg0: *const u8, arg1: *const u8, ...) -> i32;
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn schedule_timeout_uninterruptible(arg0: i64) -> i64;
    fn down_write(arg0: *mut rw_semaphore);
    fn up_write(arg0: *mut rw_semaphore);
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn ktime_get() -> i64;
    fn ktime_get_with_offset(arg0: u32) -> i64;
    fn mod_timer(arg0: *mut timer_list, arg1: u64) -> i32;
    fn hrtimer_setup(arg0: *mut hrtimer, arg1: Option<unsafe extern "C" fn(*mut hrtimer) -> u32>, arg2: i32, arg3: u32);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn hrtimer_cancel(arg0: *mut hrtimer) -> i32;
    fn __compiletime_assert_361();
    fn __compiletime_assert_386();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_392();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn sysfs_emit(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn sysfs_emit_at(arg0: *mut u8, arg1: i32, arg2: *const u8, ...) -> i32;
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn device_create_file(arg0: *mut device, arg1: *const device_attribute) -> i32;
    fn device_remove_file(arg0: *mut device, arg1: *const device_attribute);
    fn synchronize_irq(arg0: u32);
    fn dma_pool_create_node(arg0: *const u8, arg1: *mut device, arg2: u64, arg3: u64, arg4: u64, arg5: i32) -> *mut dma_pool;
    fn dma_pool_destroy(arg0: *mut dma_pool);
    fn dma_pool_alloc(arg0: *mut dma_pool, arg1: u32, arg2: *mut u64) -> *mut ();
    fn dma_pool_free(arg0: *mut dma_pool, arg1: *mut (), arg2: u64);
    fn dma_alloc_attrs(arg0: *mut device, arg1: u64, arg2: *mut u64, arg3: u32, arg4: u64) -> *mut ();
    fn dma_free_attrs(arg0: *mut device, arg1: u64, arg2: *mut (), arg3: u64, arg4: u64);
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn msleep(arg0: u32);
    fn usleep_range_state(arg0: u64, arg1: u64, arg2: u32);
    fn usb_for_each_dev(arg0: *mut (), arg1: Option<unsafe extern "C" fn(*mut usb_device, *mut ()) -> i32>) -> i32;
    fn usb_disabled() -> i32;
    fn usb_hcd_link_urb_to_ep(arg0: *mut usb_hcd, arg1: *mut urb) -> i32;
    fn usb_hcd_check_unlink_urb(arg0: *mut usb_hcd, arg1: *mut urb, arg2: i32) -> i32;
    fn usb_hcd_unlink_urb_from_ep(arg0: *mut usb_hcd, arg1: *mut urb);
    fn usb_hcd_giveback_urb(arg0: *mut usb_hcd, arg1: *mut urb, arg2: i32);
    fn usb_hc_died(arg0: *mut usb_hcd);
    fn usb_hcd_poll_rh_status(arg0: *mut usb_hcd);
    fn usb_hcd_start_port_resume(arg0: *mut usb_bus, arg1: i32);
    fn usb_hcd_end_port_resume(arg0: *mut usb_bus, arg1: i32);
    fn usb_hub_clear_tt_buffer(arg0: *mut urb) -> i32;
    fn usb_calc_bus_time(arg0: i32, arg1: i32, arg2: i32, arg3: i32) -> i64;
    fn usb_root_hub_lost_power(arg0: *mut usb_device);
    fn usb_hcd_resume_root_hub(arg0: *mut usb_hcd);
    fn __platform_register_drivers(arg0: *const *mut platform_driver, arg1: u32, arg2: *mut module, arg3: *const u8) -> i32;
    fn platform_unregister_drivers(arg0: *const *mut platform_driver, arg1: u32);
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn dbgp_external_startup(arg0: *mut usb_hcd) -> i32;
    fn dbgp_reset_prep(arg0: *mut usb_hcd) -> i32;
    fn usb_amd_dev_put();
    fn usb_amd_quirk_pll_disable();
    fn usb_amd_quirk_pll_enable();
    fn __compiletime_assert_681();
    fn __compiletime_assert_689();
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


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
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


#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
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
fn size_add(addend1: u64, addend2: u64) -> u64 {
let mut bytes: u64 = 0;
if check_add_overflow!(addend1, addend2, core::ptr::addr_of_mut!(bytes)) {
return !0;
}
bytes
}

// __set_flex_counter() - Set the counter associated with the given flexible
//                        array member that has been annoated by __counted_by().
// @FAM: Instance of flexible array member within a given struct.
// @COUNT: Value to store to the __counted_by annotated @FAM_PTR's counter.
//
// This is a no-op if no annotation exists. Count needs to be checked with
// overflows_flex_counter_type() before using this function.
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
#[inline(always)]
fn __le32_to_cpup(p: *const u32) -> u32 {
unsafe { *p }
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
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_0() };
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
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1() };
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

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_2() };
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
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_5() };
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

fn list_move_tail(list: *mut list_head, head: *mut list_head) {
__list_del_entry(list);
list_add_tail(list, head);
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_6() };
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

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_67 {
}


fn __list_splice(list: *const list_head, prev: *mut list_head, next: *mut list_head) {
let first: *mut list_head = unsafe { (*list).next };
let last: *mut list_head = unsafe { (*list).prev };
unsafe {
(*first).prev = prev;
}
unsafe {
(*prev).next = first;
}
unsafe {
(*last).next = next;
}
unsafe {
(*next).prev = last;
}
}

fn list_splice(list: *const list_head, head: *mut list_head) {
if list_empty(list) == 0 {
__list_splice(list, head, unsafe { (*head).next });
}
}

fn list_splice_tail(list: *mut list_head, head: *mut list_head) {
if list_empty(list as *const list_head) == 0 {
__list_splice(list as *const list_head, unsafe { (*head).prev }, head);
}
}

fn list_splice_tail_init(list: *mut list_head, head: *mut list_head) {
if list_empty(list as *const list_head) == 0 {
__list_splice(list as *const list_head, unsafe { (*head).prev }, head);
INIT_LIST_HEAD(list);
}
}

// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_safe - iterate over a list safe against removal of list entry
// @pos:	the &struct list_head to use as a loop cursor.
// @n:		another &struct list_head to use as temporary storage
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
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_29() };
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
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
core::arch::asm!("32: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 32b - ., 3; .popsection\n# {0}", const 32 as i32, options(nostack, att_syntax));
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
core::arch::asm!("33: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 33b - ., 4; .popsection\n# {0}", const 33 as i32, options(nostack, att_syntax));
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
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
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

// instrument_get_user() - add instrumentation to get_user()-like macros
// @to: destination variable, may not be address-taken
//
// get_user() and friends are fragile, so it may depend on the implementation
// whether the instrumentation happens before or after the data is copied from
// the userspace.
#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[inline(always)]
fn set_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_set_bit(nr, addr);
}

#[inline(always)]
fn clear_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_clear_bit(nr, addr);
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
fn fls_long(l: u64) -> u32 {
if ((const { core::mem::size_of::<u64>() }) as u64) == 4 {
return fls(l as u32) as u32;
}
fls64(l) as u32
}

// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
}

fn __rounddown_pow_of_two(n: u64) -> u64 {
1u64 << fls_long(n).wrapping_sub(1)
}

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


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits123U0,
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
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
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits128U0,
}


#[inline(always)]
fn find_next_bit(addr: *const u64, size: u64, offset: u64) -> u64 {
if small_const_nbits!(size) {
let mut val: u64 = 0;
if unlikely!(offset >= size) != 0 {
return size;
}
val = (unsafe { *addr }) & GENMASK!(size.wrapping_sub(1), offset);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_next_bit(addr, size, offset) }
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
    __slate_bits_0: __slate_bits::__SlateBits129U0,
    pkru: u32,
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
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
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
struct workqueue_struct {
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
struct rseq {
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
struct futex_unlock_cs_range {
    start_ip: u64,
    len: u32,
    pop_size32: u32,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct timer_list {
    entry: hlist_node,
    expires: u64,
    function: Option<unsafe extern "C" fn(*mut timer_list)>,
    flags: u32,
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
struct seccomp {
    mode: i32,
    filter_count: __SlateRecord_types_h_190,
    filter: *mut seccomp_filter,
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
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
}


#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
struct gendisk {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
struct uprobe_xol_ops {
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
    __slate_bits_0: __slate_bits::__SlateBits286U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
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
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
    __slate_bits_0: __slate_bits::__SlateBits71U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits71U1,
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
    __slate_bits_2: __slate_bits::__SlateBits71U2,
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
struct ldt_struct {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
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
struct wait_queue_entry {
    flags: u32,
    private: *mut (),
    func: Option<unsafe extern "C" fn(*mut wait_queue_entry, u32, i32, *mut ()) -> i32>,
    entry: list_head,
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
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
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


// Add a ktime_t variable and a scalar nanosecond value.
// res = kt + nsval:
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

fn ktime_get_real() -> i64 {
unsafe { ktime_get_with_offset(0u32) }
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
struct irq_domain {
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
struct notifier_block {
    notifier_call: Option<unsafe extern "C" fn(*mut notifier_block, u64, *mut ()) -> i32>,
    next: *mut notifier_block,
    priority: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct atomic_notifier_head {
    lock: spinlock,
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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
struct sock {
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
struct kmem_cache {
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
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits421U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
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
struct tty_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhash_head {
    next: *mut rhash_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
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


fn hrtimer_restart(timer: *mut hrtimer) {
hrtimer_start_expires(timer, 0u32);
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
struct sighand_struct {
    siglock: spinlock,
    count: refcount_struct,
    signalfd_wqh: wait_queue_head,
    action: [k_sigaction; 64],
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
    __slate_bits_0: __slate_bits::__SlateBits289U0,
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
}


// group exit in progress
fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_361() };
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
struct poll_table_struct {
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
struct percpu_rw_semaphore {
    rss: rcu_sync,
    read_count: *mut u32,
    writer: rcuwait,
    waiters: wait_queue_head,
    block: __SlateRecord_types_h_190,
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
struct posix_acl {
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
struct io_uring_cmd {
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
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
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
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_386() };
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
    __slate_bits_0: __slate_bits::__SlateBits594U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
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
unsafe { __compiletime_assert_392() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("393: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 393b - ., 3; .popsection\n# {0}", const 393 as i32, options(nostack, att_syntax));
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
// __alloc_flex - Allocate an object that has a trailing flexible array
// @KMALLOC: kmalloc wrapper function to use for allocation.
// @GFP: GFP flags for the allocation.
// @TYPE: type of structure to allocate space for.
// @FAM: The name of the flexible array member of @TYPE structure.
// @COUNT: how many @FAM elements to allocate space for.
//
// Returns: Newly allocated pointer to @TYPE with @COUNT-many trailing
// @FAM elements, or NULL on failure or if @COUNT cannot be represented
// by the member of @TYPE that counts the @FAM elements (annotated via
// __counted_by()).
// All kzalloc aliases for kmalloc_(obj|objs|flex).
fn _kmalloc_array_noprof(n: u64, size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
let mut bytes: u64 = 0;
if unlikely!(check_mul_overflow!(n, size, core::ptr::addr_of_mut!(bytes))) != 0 {
return core::ptr::null_mut::<()>();
}
_kmalloc_noprof(bytes, flags, token)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
}


// kcalloc - allocate memory for an array. The memory is set to zero.
// @n: number of elements.
// @size: element size.
// @flags: the type of memory to allocate (see kmalloc).
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
struct wake_irq {
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
struct dev_pm_qos {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
    __slate_bits_0: __slate_bits::__SlateBits663U0,
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
struct dma_pool {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_vmalloc_h_53 {
    next: *mut vm_struct,
    llnode: llist_node,
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
    __slate_bits_0: __slate_bits::__SlateBits606U0,
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
union __SlateRecord_ch9_h_1010 {
    legacy_padding: u32,
    __slate_anon_1: __SlateRecord_ch9_h_1013,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ch9_h_1013 {
    __empty_bmSublinkSpeedAttr: __SlateRecord_ch9_h_1013_2,
    bmSublinkSpeedAttr: [u32; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ch9_h_1013_2 {
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
struct ep_device {
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
struct mon_bus {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits709U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits709U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits709U2,
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
    __slate_bits_3: __slate_bits::__SlateBits709U3,
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
struct pdev_archdata {
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
struct usb_phy_roothub {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gen_pool {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_attribute {
    attr: attribute,
    __slate_anon_1: __SlateRecord_device_h_108,
    __slate_anon_2: __SlateRecord_device_h_114,
}


// DEVICE_ATTR_RW - Define a read-write device attribute.
// @_name: Attribute name.
//
// Like DEVICE_ATTR(), but @_mode is 0644, @_show is <_name>_show,
// and @_store is <_name>_store.
#[repr(C)]
#[derive(Clone, Copy)]
struct device_dma_parameters {
    max_segment_size: u32,
    min_align_mask: u32,
    segment_boundary_mask: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ch11_h_269 {
    hs: __SlateRecord_ch11_h_270,
    ss: __SlateRecord_ch11_h_276,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ch11_h_270 {
    DeviceRemovable: [u8; 4],
    PortPwrCtrlMask: [u8; 4],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_ch11_h_276 {
    bHubHdrDecLat: u8,
    wHubDelay: u16,
    DeviceRemovable: u16,
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
struct device_link {
    supplier: *mut device,
    s_node: list_head,
    consumer: *mut device,
    c_node: list_head,
    link_dev: device,
    status: i32,
    flags: u32,
    rpm_active: refcount_struct,
    kref: kref,
    rm_work: work_struct,
    supplier_preactivated: bool,
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
    __slate_bits_0: __slate_bits::__SlateBits706U0,
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


fn dev_get_drvdata(dev: *const device) -> *mut () {
unsafe { (*dev).driver_data }
}

fn readl(mut addr: *const ()) -> u32 {
let mut ret: u32 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movl ({1:r}),{0:e}", lateout(reg) __t0, in(reg) addr as *mut u32, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
ret
}

fn writel(mut val: u32, mut addr: *mut ()) {
unsafe {
core::arch::asm!("movl {0:e},({1:r})", in(reg) val as u32, in(reg) addr as *mut u32, options(nostack, att_syntax));
}
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
struct cgroup_taskset {
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
struct scatterlist {
    page_link: u64,
    offset: u32,
    length: u32,
    dma_address: u64,
    dma_length: u32,
    dma_flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sg_table {
    sgl: *mut scatterlist,
    nents: u32,
    orig_nents: u32,
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


// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
fn __sg_flags(sg: *mut scatterlist) -> u32 {
((unsafe { (*sg).page_link }) & (1 | 2)) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct swap_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct regulator {
}


fn sg_chain_ptr(sg: *mut scatterlist) -> *mut scatterlist {
((unsafe { (*sg).page_link }) & !(1 | 2)) as *mut scatterlist
}

fn sg_is_chain(sg: *mut scatterlist) -> bool {
(__sg_flags(sg) as u64) & 1 != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_phy_hdmi_h_27 {
    tmds_char_rate: u64,
    frl: __SlateRecord_phy_hdmi_h_29,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_phy_hdmi_h_29 {
    rate_per_lane: u8,
    lanes: u8,
}


fn sg_is_last(sg: *mut scatterlist) -> bool {
(__sg_flags(sg) as u64) & 2 != 0
}

fn sg_next(mut sg: *mut scatterlist) -> *mut scatterlist {
if sg_is_last(sg) {
return core::ptr::null_mut::<scatterlist>();
}
let __v0: *mut scatterlist = sg;
sg = __v0.wrapping_offset(1 as isize);
if unlikely!(sg_is_chain(sg)) != 0 {
sg = sg_chain_ptr(sg);
}
sg
}

fn dma_pool_create(name: *const u8, dev: *mut device, size: u64, align: u64, boundary: u64) -> *mut dma_pool {
unsafe { dma_pool_create_node(name, dev, size, align, boundary, 1i32.wrapping_neg()) }
}

fn dma_pool_zalloc(pool: *mut dma_pool, mem_flags: u32, handle: *mut u64) -> *mut () {
unsafe { dma_pool_alloc(pool, mem_flags | (BIT!(8i32) as u32), handle) }
}

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
fn dma_alloc_coherent(dev: *mut device, size: u64, dma_handle: *mut u64, gfp: u32) -> *mut () {
unsafe { dma_alloc_attrs(dev, size, dma_handle, gfp, if gfp & (BIT!(13i32) as u32) != 0 { 256u64 } else { 0u64 }) }
}

fn dma_free_coherent(dev: *mut device, size: u64, cpu_addr: *mut (), dma_handle: u64) {
unsafe { dma_free_attrs(dev, size, cpu_addr, dma_handle, 0u64) };
return;
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

#[repr(C)]
#[derive(Clone, Copy)]
struct extcon_dev {
}


fn usleep_range(min: u64, max: u64) {
unsafe { usleep_range_state(min, max, 2u32) };
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


// to device
// to host
// Status Type
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ctrlrequest {
    bRequestType: u8,
    bRequest: u8,
    wValue: u16,
    wIndex: u16,
    wLength: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_gadget {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_device_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bcdUSB: u16,
    bDeviceClass: u8,
    bDeviceSubClass: u8,
    bDeviceProtocol: u8,
    bMaxPacketSize0: u8,
    idVendor: u16,
    idProduct: u16,
    bcdDevice: u16,
    iManufacturer: u8,
    iProduct: u8,
    iSerialNumber: u8,
    bNumConfigurations: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mfd_cell {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_config_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    wTotalLength: u16,
    bNumInterfaces: u8,
    bConfigurationValue: u8,
    iConfiguration: u8,
    bmAttributes: u8,
    bMaxPower: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_81 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_81_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_81_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_81_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_81_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_81_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_86 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_86_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_86_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_86_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_86_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_86_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_91 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_91_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_91_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_91_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_91_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_91_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_interface_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bInterfaceNumber: u8,
    bAlternateSetting: u8,
    bNumEndpoints: u8,
    bInterfaceClass: u8,
    bInterfaceSubClass: u8,
    bInterfaceProtocol: u8,
    iInterface: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_endpoint_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bEndpointAddress: u8,
    bmAttributes: u8,
    wMaxPacketSize: u16,
    bInterval: u8,
    bRefresh: u8,
    bSynchAddress: u8,
}


// Endpoints
// in bEndpointAddress
// in bmAttributes
// The USB 3.0 spec redefines bits 5:4 of bmAttributes as interrupt ep type.
fn usb_endpoint_num(epd: *const usb_endpoint_descriptor) -> i32 {
((unsafe { (*epd).bEndpointAddress }) as i32) & 15
}

fn usb_endpoint_type(epd: *const usb_endpoint_descriptor) -> i32 {
((unsafe { (*epd).bmAttributes }) as i32) & 3
}

fn usb_endpoint_dir_in(epd: &usb_endpoint_descriptor) -> i32 {
(((unsafe { (*(epd as *const usb_endpoint_descriptor)).bEndpointAddress }) as i32) & 128 == 128) as i32
}

fn usb_endpoint_dir_out(epd: *const usb_endpoint_descriptor) -> i32 {
(((unsafe { (*epd).bEndpointAddress }) as i32) & 128 == 0) as i32
}

fn usb_endpoint_maxp(epd: *const usb_endpoint_descriptor) -> i32 {
(__le16_to_cpu!(unsafe { (*epd).wMaxPacketSize }) as i32) & 2047
}

fn usb_endpoint_maxp_mult(epd: *const usb_endpoint_descriptor) -> i32 {
let maxp: i32 = __le16_to_cpu!(unsafe { (*epd).wMaxPacketSize }) as i32;
USB_EP_MAXP_MULT!(maxp).wrapping_add(1)
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_eusb2_isoc_ep_comp_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    wMaxPacketSize: u16,
    dwBytesPerInterval: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ehci_def_h_116 {
    port_status: [u32; 15],
    __slate_anon_1: __SlateRecord_ehci_def_h_155,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_def_h_155 {
    reserved3: [u32; 9],
    usbmode: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ehci_def_h_170 {
    __slate_anon_0: __SlateRecord_ehci_def_h_171,
    brcm_insnreg: [u32; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_def_h_171 {
    reserved4: u32,
    hostpc: [u32; 15],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ssp_isoc_ep_comp_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    wReseved: u16,
    dwBytesPerInterval: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ss_ep_comp_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bMaxBurst: u8,
    bmAttributes: u8,
    wBytesPerInterval: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_interface_assoc_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bFirstInterface: u8,
    bInterfaceCount: u8,
    bFunctionClass: u8,
    bFunctionSubClass: u8,
    bFunctionProtocol: u8,
    iFunction: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_bos_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    wTotalLength: u16,
    bNumDeviceCaps: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ext_cap_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
    bmAttributes: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ss_cap_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
    bmAttributes: u8,
    wSpeedSupported: u16,
    bFunctionalitySupport: u8,
    bU1devExitLat: u8,
    bU2DevExitLat: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_ss_container_id_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
    bReserved: u8,
    ContainerID: [u8; 16],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_ssp_cap_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
    bReserved: u8,
    bmAttributes: u32,
    wFunctionalitySupport: u16,
    wReserved: u16,
    __slate_anon_7: __SlateRecord_ch9_h_1010,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_sched_c_1169_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_sysfs_c_58 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_sysfs_c_58_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_sysfs_c_58_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_sysfs_c_58_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_sysfs_c_58_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_sysfs_c_58_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_sysfs_c_140 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_sysfs_c_140_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_sysfs_c_140_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_sysfs_c_140_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_sysfs_c_140_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_sysfs_c_140_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_1366 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ehci_hcd_c_1396 {
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
    pub struct __SlateBits562U0 {
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
    pub struct __SlateBits594U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits286U0 {
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
    pub struct __SlateBits71U0 {
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
    pub struct __SlateBits71U1 {
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
    pub struct __SlateBits663U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits849U0 {
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
    pub struct __SlateBits289U0 {
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
    pub struct __SlateBits71U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits123U0 {
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
    pub struct __SlateBits129U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits709U0 {
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
    pub struct __SlateBits706U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits709U1 {
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
    pub struct __SlateBits709U2 {
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
    pub struct __SlateBits709U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits606U0 {
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
    pub struct __SlateBits128U0 {
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
    pub struct __SlateBits924U0 {
        #[bits(1)]
        pub scanning: bool,
        #[bits(1)]
        pub need_rescan: bool,
        #[bits(1)]
        pub intr_unlinking: bool,
        #[bits(1)]
        pub iaa_in_progress: bool,
        #[bits(1)]
        pub async_unlinking: bool,
        #[bits(1)]
        pub shutdown: bool,
        #[bits(2, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits810U0 {
        #[bits(1)]
        pub poisoned: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits802U0 {
        #[bits(1)]
        pub is_b_host: u32,
        #[bits(1)]
        pub b_hnp_enable: u32,
        #[bits(1)]
        pub no_stop_on_short: u32,
        #[bits(1)]
        pub no_sg_constraint: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits798U0 {
        #[bits(1)]
        pub sysfs_files_created: u32,
        #[bits(1)]
        pub ep_devs_created: u32,
        #[bits(1)]
        pub unregistering: u32,
        #[bits(1)]
        pub needs_remote_wakeup: u32,
        #[bits(1)]
        pub needs_altsetting0: u32,
        #[bits(1)]
        pub needs_binding: u32,
        #[bits(1)]
        pub resetting_device: u32,
        #[bits(1)]
        pub authorized: u32,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits792U0 {
        #[bits(1)]
        pub can_submit: u32,
        #[bits(1)]
        pub persist_enabled: u32,
        #[bits(1)]
        pub reset_in_progress: u32,
        #[bits(1)]
        pub have_langid: u32,
        #[bits(1)]
        pub authorized: u32,
        #[bits(1)]
        pub authenticated: u32,
        #[bits(1)]
        pub lpm_capable: u32,
        #[bits(1)]
        pub lpm_devinit_allow: u32,
        #[bits(1)]
        pub usb2_hw_lpm_capable: u32,
        #[bits(1)]
        pub usb2_hw_lpm_besl_capable: u32,
        #[bits(1)]
        pub usb2_hw_lpm_enabled: u32,
        #[bits(1)]
        pub usb2_hw_lpm_allowed: u32,
        #[bits(1)]
        pub usb3_lpm_u1_enabled: u32,
        #[bits(1)]
        pub usb3_lpm_u2_enabled: u32,
        #[bits(2, access = na)]
        pub __slate_pad_14: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits792U1 {
        #[bits(1)]
        pub do_remote_wakeup: u32,
        #[bits(1)]
        pub reset_resume: u32,
        #[bits(1)]
        pub port_is_suspended: u32,
        #[bits(1)]
        pub offload_pm_locked: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits792U2 {
        #[bits(1)]
        pub use_generic_driver: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits928U0 {
        #[bits(1)]
        pub is_out: u32,
        #[bits(1)]
        pub clearing_tt: u32,
        #[bits(1)]
        pub dequeue_during_giveback: u32,
        #[bits(1)]
        pub should_be_inactive: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits924U1 {
        #[bits(1)]
        pub no_selective_suspend: u32,
        #[bits(1)]
        pub has_fsl_port_bug: u32,
        #[bits(1)]
        pub has_fsl_hs_errata: u32,
        #[bits(1)]
        pub has_fsl_susp_errata: u32,
        #[bits(1)]
        pub has_ci_pec_bug: u32,
        #[bits(1)]
        pub big_endian_mmio: u32,
        #[bits(1)]
        pub big_endian_desc: u32,
        #[bits(1)]
        pub big_endian_capbase: u32,
        #[bits(1)]
        pub has_amcc_usb23: u32,
        #[bits(1)]
        pub need_io_watchdog: u32,
        #[bits(1)]
        pub amd_pll_fix: u32,
        #[bits(1)]
        pub use_dummy_qh: u32,
        #[bits(1)]
        pub has_synopsys_hc_bug: u32,
        #[bits(1)]
        pub frame_index_bug: u32,
        #[bits(1)]
        pub need_oc_pp_cycle: u32,
        #[bits(1)]
        pub imx28_write_fix: u32,
        #[bits(1)]
        pub spurious_oc: u32,
        #[bits(1)]
        pub is_aspeed: u32,
        #[bits(1)]
        pub zx_wakeup_clear_needed: u32,
        #[bits(5, access = na)]
        pub __slate_pad_19: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits924U2 {
        #[bits(1)]
        pub has_hostpc: u32,
        #[bits(1)]
        pub has_tdi_phy_lpm: u32,
        #[bits(1)]
        pub has_ppcd: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits875U0 {
        #[bits(1)]
        pub ssc: u8,
        #[bits(1)]
        pub set_rate: u8,
        #[bits(1)]
        pub set_lanes: u8,
        #[bits(1)]
        pub set_voltages: u8,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits814U0 {
        #[bits(1)]
        pub rh_registered: u32,
        #[bits(1)]
        pub rh_pollable: u32,
        #[bits(1)]
        pub msix_enabled: u32,
        #[bits(1)]
        pub msi_enabled: u32,
        #[bits(1)]
        pub skip_phy_initialization: u32,
        #[bits(1)]
        pub uses_new_polling: u32,
        #[bits(1)]
        pub has_tt: u32,
        #[bits(1)]
        pub amd_resume_bug: u32,
        #[bits(1)]
        pub can_do_streams: u32,
        #[bits(1)]
        pub tpl_support: u32,
        #[bits(1)]
        pub cant_recv_wakeups: u32,
        #[bits(5, access = na)]
        pub __slate_pad_11: u8,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct usb_ptm_cap_descriptor {
    bLength: u8,
    bDescriptorType: u8,
    bDevCapabilityType: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_host_endpoint {
    desc: usb_endpoint_descriptor,
    ss_ep_comp: usb_ss_ep_comp_descriptor,
    ssp_isoc_ep_comp: usb_ssp_isoc_ep_comp_descriptor,
    eusb2_isoc_ep_comp: usb_eusb2_isoc_ep_comp_descriptor,
    urb_list: list_head,
    hcpriv: *mut (),
    ep_dev: *mut ep_device,
    extra: *mut u8,
    extralen: i32,
    enabled: i32,
    streams: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_host_interface {
    desc: usb_interface_descriptor,
    extralen: i32,
    extra: *mut u8,
    endpoint: *mut usb_host_endpoint,
    string: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_interface {
    altsetting: *mut usb_host_interface,
    cur_altsetting: *mut usb_host_interface,
    num_altsetting: u32,
    intf_assoc: *mut usb_interface_assoc_descriptor,
    minor: i32,
    condition: u32,
    __slate_bits_0: __slate_bits::__SlateBits798U0,
    wireless_status: u32,
    wireless_status_work: work_struct,
    dev: device,
    usb_dev: *mut device,
    reset_ws: work_struct,
}


// this maximum is arbitrary
// USB Resume Timer: Every Host controller driver should drive the resume
// signalling on the bus for the amount of time defined by this macro.
//
// That way we will have a 'stable' behavior among all HCDs supported by Linux.
//
// Note that the USB Specification states we should drive resume for *at least*
// 20 ms, but it doesn't give an upper bound. This creates two possible
// situations which we want to avoid:
//
// (a) sometimes an msleep(20) might expire slightly before 20 ms, which causes
// us to fail USB Electrical Tests, thus failing Certification
//
// (b) Some (many) devices actually need more than 20 ms of resume signalling,
// and while we can argue that's against the USB Specification, we don't have
// control over which devices a certification laboratory will be using for
// certification. If CertLab uses a device which was tested against Windows and
// that happens to have relaxed resume signalling rules, we might fall into
// situations where we fail interoperability and electrical tests.
//
// In order to avoid both conditions, we're using a 40 ms resume timeout, which
// should cope with both LPJ calibration errors and devices not following every
// detail of the USB Specification.
// ms
#[repr(C)]
#[derive(Clone, Copy)]
struct usb_interface_cache {
    num_altsetting: u32,
    r#ref: kref,
    altsetting: [usb_host_interface; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_host_config {
    desc: usb_config_descriptor,
    string: *mut u8,
    intf_assoc: [*mut usb_interface_assoc_descriptor; 16],
    interface: [*mut usb_interface; 32],
    intf_cache: [*mut usb_interface_cache; 32],
    extra: *mut u8,
    extralen: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_host_bos {
    desc: *mut usb_bos_descriptor,
    ext_cap: *mut usb_ext_cap_descriptor,
    ss_cap: *mut usb_ss_cap_descriptor,
    ssp_cap: *mut usb_ssp_cap_descriptor,
    ss_id: *mut usb_ss_container_id_descriptor,
    ptm_cap: *mut usb_ptm_cap_descriptor,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_bus {
    controller: *mut device,
    sysdev: *mut device,
    busnum: i32,
    bus_name: *const u8,
    uses_pio_for_control: u8,
    otg_port: u8,
    __slate_bits_0: __slate_bits::__SlateBits802U0,
    sg_tablesize: u32,
    devnum_next: i32,
    devnum_next_mutex: mutex,
    devmap: [u64; 2],
    root_hub: *mut usb_device,
    hs_companion: *mut usb_bus,
    bandwidth_allocated: i32,
    bandwidth_int_reqs: i32,
    bandwidth_isoc_reqs: i32,
    resuming_ports: u32,
    mon_bus: *mut mon_bus,
    monitored: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb2_lpm_parameters {
    besl: u32,
    timeout: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb3_lpm_parameters {
    mel: u32,
    pel: u32,
    sel: u32,
    timeout: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_device {
    devnum: i32,
    devpath: [u8; 16],
    route: u32,
    state: u32,
    speed: u32,
    rx_lanes: u32,
    tx_lanes: u32,
    ssp_rate: u32,
    tt: *mut usb_tt,
    ttport: i32,
    toggle: [u32; 2],
    parent: *mut usb_device,
    bus: *mut usb_bus,
    ep0: usb_host_endpoint,
    dev: device,
    descriptor: usb_device_descriptor,
    bos: *mut usb_host_bos,
    config: *mut usb_host_config,
    actconfig: *mut usb_host_config,
    ep_in: [*mut usb_host_endpoint; 16],
    ep_out: [*mut usb_host_endpoint; 16],
    rawdescriptors: *mut *mut u8,
    bus_mA: u16,
    portnum: u8,
    level: u8,
    devaddr: u8,
    __slate_bits_0: __slate_bits::__SlateBits792U0,
    string_langid: i32,
    product: *mut u8,
    manufacturer: *mut u8,
    serial: *mut u8,
    filelist: list_head,
    maxchild: i32,
    quirks: u32,
    urbnum: __SlateRecord_types_h_190,
    active_duration: u64,
    connect_time: u64,
    __slate_bits_1: __slate_bits::__SlateBits792U1,
    offload_usage: i32,
    offload_lock: spinlock,
    tunnel_mode: u32,
    usb4_link: *mut device_link,
    slot_id: i32,
    l1_params: usb2_lpm_parameters,
    u1_params: usb3_lpm_parameters,
    u2_params: usb3_lpm_parameters,
    lpm_disable_count: u32,
    hub_delay: u16,
    __slate_bits_2: __slate_bits::__SlateBits792U2,
}


// urb->transfer_flags:
//
// Note: URB_DIR_IN/OUT is automatically set in usb_submit_urb().
// report short reads as errors
// iso-only; use the first unexpired
// slot in the schedule
// Finish bulk OUT with short packet
// HINT: no non-error interrupt
// needed
// The following flags are used internally by usbcore and HCDs
// Transfer from device to host
#[repr(C)]
#[derive(Clone, Copy)]
struct usb_iso_packet_descriptor {
    offset: u32,
    length: u32,
    actual_length: u32,
    status: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_anchor {
    urb_list: list_head,
    wait: wait_queue_head,
    lock: spinlock,
    suspend_wakeups: __SlateRecord_types_h_190,
    __slate_bits_0: __slate_bits::__SlateBits810U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct urb {
    kref: kref,
    unlinked: i32,
    hcpriv: *mut (),
    use_count: __SlateRecord_types_h_190,
    reject: __SlateRecord_types_h_190,
    urb_list: list_head,
    anchor_list: list_head,
    anchor: *mut usb_anchor,
    dev: *mut usb_device,
    ep: *mut usb_host_endpoint,
    pipe: u32,
    stream_id: u32,
    status: i32,
    transfer_flags: u32,
    transfer_buffer: *mut (),
    transfer_dma: u64,
    sg: *mut scatterlist,
    sgt: *mut sg_table,
    num_mapped_sgs: i32,
    num_sgs: i32,
    transfer_buffer_length: u32,
    actual_length: u32,
    setup_packet: *mut u8,
    setup_dma: u64,
    start_frame: i32,
    number_of_packets: i32,
    interval: i32,
    error_count: i32,
    context: *mut (),
    complete: Option<unsafe extern "C" fn(*mut urb)>,
    iso_frame_desc: [usb_iso_packet_descriptor; 0],
}


// NOTE:  these are not the standard USB_ENDPOINT_XFER_* values!!
//
// (yet ... they're the values used by usbfs)
fn usb_pipe_endpoint(dev: *mut usb_device, pipe: u32) -> *mut usb_host_endpoint {
let eps: *mut *mut usb_host_endpoint = if usb_pipein!(pipe) != 0 { unsafe { (*dev).ep_in.as_mut_ptr() as *mut *mut usb_host_endpoint } } else { unsafe { (*dev).ep_out.as_mut_ptr() as *mut *mut usb_host_endpoint } };
unsafe { *eps.wrapping_offset(usb_pipeendpoint!(pipe) as isize) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct giveback_urb_bh {
    running: bool,
    high_prio: bool,
    lock: spinlock,
    head: list_head,
    bh: work_struct,
    completing_ep: *mut usb_host_endpoint,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_hcd {
    self_: usb_bus,
    kref: kref,
    product_desc: *const u8,
    speed: i32,
    irq_descr: [u8; 24],
    rh_timer: timer_list,
    status_urb: *mut urb,
    wakeup_work: work_struct,
    died_work: work_struct,
    driver: *const hc_driver,
    usb_phy: *mut usb_phy,
    phy_roothub: *mut usb_phy_roothub,
    flags: u64,
    dev_policy: u32,
    __slate_bits_0: __slate_bits::__SlateBits814U0,
    irq: u32,
    regs: *mut (),
    rsrc_start: u64,
    rsrc_len: u64,
    power_budget: u32,
    high_prio_bh: giveback_urb_bh,
    low_prio_bh: giveback_urb_bh,
    address0_mutex: *mut mutex,
    bandwidth_mutex: *mut mutex,
    shared_hcd: *mut usb_hcd,
    primary_hcd: *mut usb_hcd,
    pool: [*mut dma_pool; 4],
    state: i32,
    localmem_pool: *mut gen_pool,
    hcd_priv: [u64; 0],
}


fn bus_to_hcd(bus: *mut usb_bus) -> *mut usb_hcd {
container_of!(bus, usb_hcd)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hc_driver {
    description: *const u8,
    product_desc: *const u8,
    hcd_priv_size: u64,
    irq: Option<unsafe extern "C" fn(*mut usb_hcd) -> u32>,
    flags: i32,
    reset: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    start: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    pci_suspend: Option<unsafe extern "C" fn(*mut usb_hcd, bool) -> i32>,
    pci_resume: Option<unsafe extern "C" fn(*mut usb_hcd, pm_message) -> i32>,
    pci_poweroff_late: Option<unsafe extern "C" fn(*mut usb_hcd, bool) -> i32>,
    stop: Option<unsafe extern "C" fn(*mut usb_hcd)>,
    shutdown: Option<unsafe extern "C" fn(*mut usb_hcd)>,
    get_frame_number: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    urb_enqueue: Option<unsafe extern "C" fn(*mut usb_hcd, *mut urb, u32) -> i32>,
    urb_dequeue: Option<unsafe extern "C" fn(*mut usb_hcd, *mut urb, i32) -> i32>,
    map_urb_for_dma: Option<unsafe extern "C" fn(*mut usb_hcd, *mut urb, u32) -> i32>,
    unmap_urb_for_dma: Option<unsafe extern "C" fn(*mut usb_hcd, *mut urb)>,
    endpoint_disable: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_host_endpoint)>,
    endpoint_reset: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_host_endpoint)>,
    hub_status_data: Option<unsafe extern "C" fn(*mut usb_hcd, *mut u8) -> i32>,
    hub_control: Option<unsafe extern "C" fn(*mut usb_hcd, u16, u16, u16, *mut u8, u16) -> i32>,
    bus_suspend: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    bus_resume: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    start_port_reset: Option<unsafe extern "C" fn(*mut usb_hcd, u32) -> i32>,
    get_resuming_ports: Option<unsafe extern "C" fn(*mut usb_hcd) -> u64>,
    relinquish_port: Option<unsafe extern "C" fn(*mut usb_hcd, i32)>,
    port_handed_over: Option<unsafe extern "C" fn(*mut usb_hcd, i32) -> i32>,
    clear_tt_buffer_complete: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_host_endpoint)>,
    alloc_dev: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    free_dev: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device)>,
    alloc_streams: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut *mut usb_host_endpoint, u32, u32, u32) -> i32>,
    free_streams: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut *mut usb_host_endpoint, u32, u32) -> i32>,
    add_endpoint: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut usb_host_endpoint) -> i32>,
    drop_endpoint: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut usb_host_endpoint) -> i32>,
    check_bandwidth: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    reset_bandwidth: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device)>,
    address_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, u32) -> i32>,
    enable_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    update_hub_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, *mut usb_tt, u32) -> i32>,
    reset_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    update_device: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device) -> i32>,
    set_usb2_hw_lpm: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, i32) -> i32>,
    enable_usb3_lpm_timeout: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, u32) -> i32>,
    disable_usb3_lpm_timeout: Option<unsafe extern "C" fn(*mut usb_hcd, *mut usb_device, u32) -> i32>,
    find_raw_port_number: Option<unsafe extern "C" fn(*mut usb_hcd, i32) -> i32>,
    port_power: Option<unsafe extern "C" fn(*mut usb_hcd, i32, bool) -> i32>,
    submit_single_step_set_feature: Option<unsafe extern "C" fn(*mut usb_hcd, *mut urb, i32) -> i32>,
}


fn hcd_periodic_completion_in_progress(hcd: *mut usb_hcd, ep: *mut usb_host_endpoint) -> bool {
(unsafe { (*hcd).high_prio_bh.completing_ep }) == ep
}

// Hub Class feature numbers
// See USB 2.0 spec Table 11-17
// L2 suspend
// wPortStatus bit field
// See USB 2.0 spec Table 11-21
// bits 6 to 7 are reserved
// wPortChange bit field
// See USB 2.0 spec Table 11-22 and USB 2.0 LPM ECN Table-4.10
// Bits 0 to 5 shown, bits 6 to 15 are reserved
// per-port power control
// no power switching
// per-port Over-current reporting
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct usb_hub_descriptor {
    bDescLength: u8,
    bDescriptorType: u8,
    bNbrPorts: u8,
    wHubCharacteristics: u16,
    bPwrOn2PwrGood: u8,
    bHubContrCurrent: u8,
    u: __SlateRecord_ch11_h_269,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_tt {
    hub: *mut usb_device,
    multi: i32,
    think_time: u32,
    hcpriv: *mut (),
    lock: spinlock,
    clear_list: list_head,
    clear_work: work_struct,
}


// GetBusState and SetHubDescriptor are optional, omitted
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


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_ehci_handshake: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ehci_reset: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ehci_adjust_port_wakeup_flags: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ehci_hub_control: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ehci_setup: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ehci_suspend: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ehci_resume: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ehci_init_driver: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_ehci_hcd__704_1389_ehci_hcd_init6:\t\t\t\n.long\t{8} - .\t\n.previous\t\t\t\t\t\n", sym ehci_handshake, sym ehci_reset, sym ehci_adjust_port_wakeup_flags, sym ehci_hub_control, sym ehci_setup, sym ehci_suspend, sym ehci_resume, sym ehci_init_driver, sym ehci_hcd_init, options(att_syntax));

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
    __slate_bits_0: __slate_bits::__SlateBits849U0,
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
    __slate_bits_0: __slate_bits::__SlateBits562U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_configure_opts_dp {
    link_rate: u32,
    lanes: u32,
    voltage: [u32; 4],
    pre: [u32; 4],
    __slate_bits_0: __slate_bits::__SlateBits875U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_configure_opts_hdmi {
    bpc: u32,
    __slate_anon_1: __SlateRecord_phy_hdmi_h_27,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_configure_opts_lvds {
    bits_per_lane_and_dclk_cycle: u32,
    differential_clk_rate: u64,
    lanes: u32,
    is_slave: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_configure_opts_mipi_dphy {
    clk_miss: u32,
    clk_post: u32,
    clk_pre: u32,
    clk_prepare: u32,
    clk_settle: u32,
    clk_term_en: u32,
    clk_trail: u32,
    clk_zero: u32,
    d_term_en: u32,
    eot: u32,
    hs_exit: u32,
    hs_prepare: u32,
    hs_settle: u32,
    hs_skip: u32,
    hs_trail: u32,
    hs_zero: u32,
    init: u32,
    lpx: u32,
    ta_get: u32,
    ta_go: u32,
    ta_sure: u32,
    wakeup: u32,
    hs_clk_rate: u64,
    lp_clk_rate: u64,
    lanes: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union phy_notify {
    ufs_state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union phy_configure_opts {
    mipi_dphy: phy_configure_opts_mipi_dphy,
    dp: phy_configure_opts_dp,
    lvds: phy_configure_opts_lvds,
    hdmi: phy_configure_opts_hdmi,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_ops {
    init: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    exit: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    power_on: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    power_off: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    set_mode: Option<unsafe extern "C" fn(*mut phy, u32, i32) -> i32>,
    set_media: Option<unsafe extern "C" fn(*mut phy, u32) -> i32>,
    set_speed: Option<unsafe extern "C" fn(*mut phy, i32) -> i32>,
    configure: Option<unsafe extern "C" fn(*mut phy, *mut phy_configure_opts) -> i32>,
    validate: Option<unsafe extern "C" fn(*mut phy, u32, i32, *mut phy_configure_opts) -> i32>,
    reset: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    calibrate: Option<unsafe extern "C" fn(*mut phy) -> i32>,
    connect: Option<unsafe extern "C" fn(*mut phy, i32) -> i32>,
    disconnect: Option<unsafe extern "C" fn(*mut phy, i32) -> i32>,
    notify_phystate: Option<unsafe extern "C" fn(*mut phy, phy_notify) -> i32>,
    release: Option<unsafe extern "C" fn(*mut phy)>,
    owner: *mut module,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy_attrs {
    bus_width: u32,
    max_link_rate: u32,
    mode: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct phy {
    dev: device,
    id: i32,
    ops: *const phy_ops,
    mutex: mutex,
    lockdep_key: lock_class_key,
    init_count: i32,
    power_count: i32,
    attrs: phy_attrs,
    pwr: *mut regulator,
    debugfs: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_phy_io_ops {
    read: Option<unsafe extern "C" fn(*mut usb_phy, u32) -> i32>,
    write: Option<unsafe extern "C" fn(*mut usb_phy, u32, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_charger_current {
    sdp_min: u32,
    sdp_max: u32,
    dcp_min: u32,
    dcp_max: u32,
    cdp_min: u32,
    cdp_max: u32,
    aca_min: u32,
    aca_max: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_phy {
    dev: *mut device,
    label: *const u8,
    flags: u32,
    r#type: u32,
    last_event: u32,
    otg: *mut usb_otg,
    io_dev: *mut device,
    io_ops: *mut usb_phy_io_ops,
    io_priv: *mut (),
    edev: *mut extcon_dev,
    id_edev: *mut extcon_dev,
    vbus_nb: notifier_block,
    id_nb: notifier_block,
    type_nb: notifier_block,
    chg_type: u32,
    chg_state: u32,
    chg_cur: usb_charger_current,
    chg_work: work_struct,
    notifier: atomic_notifier_head,
    port_status: u16,
    port_change: u16,
    head: list_head,
    init: Option<unsafe extern "C" fn(*mut usb_phy) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut usb_phy)>,
    set_vbus: Option<unsafe extern "C" fn(*mut usb_phy, i32) -> i32>,
    set_power: Option<unsafe extern "C" fn(*mut usb_phy, u32) -> i32>,
    set_suspend: Option<unsafe extern "C" fn(*mut usb_phy, i32) -> i32>,
    set_wakeup: Option<unsafe extern "C" fn(*mut usb_phy, bool) -> i32>,
    notify_connect: Option<unsafe extern "C" fn(*mut usb_phy, u32) -> i32>,
    notify_disconnect: Option<unsafe extern "C" fn(*mut usb_phy, u32) -> i32>,
    charger_detect: Option<unsafe extern "C" fn(*mut usb_phy) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct usb_otg {
    default_a: u8,
    phy: *mut phy,
    usb_phy: *mut usb_phy,
    host: *mut usb_bus,
    gadget: *mut usb_gadget,
    state: u32,
    set_host: Option<unsafe extern "C" fn(*mut usb_otg, *mut usb_bus) -> i32>,
    set_peripheral: Option<unsafe extern "C" fn(*mut usb_otg, *mut usb_gadget) -> i32>,
    set_vbus: Option<unsafe extern "C" fn(*mut usb_otg, bool) -> i32>,
    start_srp: Option<unsafe extern "C" fn(*mut usb_otg) -> i32>,
    start_hnp: Option<unsafe extern "C" fn(*mut usb_otg) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct platform_device_id {
    name: [u8; 24],
    driver_data: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct platform_device {
    name: *const u8,
    id: i32,
    id_auto: bool,
    dev: device,
    platform_dma_mask: u64,
    dma_parms: device_dma_parameters,
    num_resources: u32,
    resource: *mut resource,
    id_entry: *const platform_device_id,
    mfd_cell: *mut mfd_cell,
    archdata: pdev_archdata,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct platform_driver {
    probe: Option<unsafe extern "C" fn(*mut platform_device) -> i32>,
    remove: Option<unsafe extern "C" fn(*mut platform_device)>,
    shutdown: Option<unsafe extern "C" fn(*mut platform_device)>,
    suspend: Option<unsafe extern "C" fn(*mut platform_device, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut platform_device) -> i32>,
    driver: device_driver,
    id_table: *const platform_device_id,
    prevent_deferred_probe: bool,
    driver_managed_dma: bool,
}


fn put_unaligned_le32(val: u32, p: *mut ()) {
__put_unaligned_t!(u32, val, p);
}

// EHCI hc_driver implementation ... experimental, incomplete.
// Based on the final 1.0 register interface specification.
//
// USB 2.0 shows up in upcoming www.pcmcia.org technology.
// First was PCMCIA, like ISA; then CardBus, which is PCI.
// Next comes "CardBay", using USB 2.0 signals.
//
// Contains additional contributions by Brad Hards, Rory Bolt, and others.
// Special thanks to Intel and VIA for providing host controllers to
// test this driver on, and Cypress (including In-System Design) for
// providing early devices for those host controllers to talk to!
static hcd_name: [u8; 9] = [101, 104, 99, 105, 95, 104, 99, 100, 0];

// magic numbers that can affect system performance
// 0-3 qtd retries; 0 == don't stop
// nak throttle; see 4.9
// 1-3 transactions/uframe; 4.10.3
// Some drivers think it's safe to schedule isochronous transfers more than
// 256 ms into the future (partly as a result of an old bug in the scheduling
// code).  In an attempt to avoid trouble, we will use a minimum scheduling
// length of 512 frames instead of 256.
// (medium) 512-frame schedule
/// Initial IRQ latency:  faster than hw default
/// 0 to 6
static mut log2_irq_thresh: i32 = unsafe { core::mem::zeroed() };

fn __check_log2_irq_thresh() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(log2_irq_thresh) }
}

static __param_str_log2_irq_thresh: __SlateAlign16<[u8; 25]> = __SlateAlign16([101u8, 104u8, 99u8, 105u8, 95u8, 104u8, 99u8, 100u8, 46u8, 108u8, 111u8, 103u8, 50u8, 95u8, 105u8, 114u8, 113u8, 95u8, 116u8, 104u8, 114u8, 101u8, 115u8, 104u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_log2_irq_thresh: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_log2_irq_thresh.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_ehci_hcd_c_81>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_hcd_c_81_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_hcd_c_81_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_hcd_c_81_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_hcd_c_81_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_hcd_c_81_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(log2_irq_thresh) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_639: [u8; 38] = [101, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 108, 111, 103, 50, 95, 105, 114, 113, 95, 116, 104, 114, 101, 115, 104, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_640: [u8; 65] = [101, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 61, 108, 111, 103, 50, 95, 105, 114, 113, 95, 116, 104, 114, 101, 115, 104, 58, 108, 111, 103, 50, 32, 73, 82, 81, 32, 108, 97, 116, 101, 110, 99, 121, 44, 32, 49, 45, 54, 52, 32, 109, 105, 99, 114, 111, 102, 114, 97, 109, 101, 115, 0];

/// initial park setting:  slower than hw default
static mut park: u32 = unsafe { core::mem::zeroed() };

fn __check_park() -> *mut u32 {
unsafe { core::ptr::addr_of_mut!(park) }
}

static __param_str_park: [u8; 14] = [101, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 107, 0];

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_park: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_park) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_uint) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_ehci_hcd_c_86>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_hcd_c_86_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_hcd_c_86_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_hcd_c_86_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_hcd_c_86_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_hcd_c_86_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(park) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_641: [u8; 28] = [101, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 112, 97, 114, 107, 58, 117, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_642: [u8; 64] = [101, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 61, 112, 97, 114, 107, 58, 112, 97, 114, 107, 32, 115, 101, 116, 116, 105, 110, 103, 59, 32, 49, 45, 51, 32, 98, 97, 99, 107, 45, 116, 111, 45, 98, 97, 99, 107, 32, 97, 115, 121, 110, 99, 32, 112, 97, 99, 107, 101, 116, 115, 0];

/// for flakey hardware, ignore overcurrent indicators
static mut ignore_oc: bool = unsafe { core::mem::zeroed() };

fn __check_ignore_oc() -> *mut bool {
unsafe { core::ptr::addr_of_mut!(ignore_oc) }
}

static __param_str_ignore_oc: __SlateAlign16<[u8; 19]> = __SlateAlign16([101u8, 104u8, 99u8, 105u8, 95u8, 104u8, 99u8, 100u8, 46u8, 105u8, 103u8, 110u8, 111u8, 114u8, 101u8, 95u8, 111u8, 99u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_ignore_oc: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_ignore_oc.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_bool) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_ehci_hcd_c_91>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_hcd_c_91_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_hcd_c_91_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_hcd_c_91_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_hcd_c_91_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_hcd_c_91_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(ignore_oc) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_643: [u8; 33] = [101, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 105, 103, 110, 111, 114, 101, 95, 111, 99, 58, 98, 111, 111, 108, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_644: [u8; 70] = [101, 104, 99, 105, 95, 104, 99, 100, 46, 112, 97, 114, 109, 61, 105, 103, 110, 111, 114, 101, 95, 111, 99, 58, 105, 103, 110, 111, 114, 101, 32, 98, 111, 103, 117, 115, 32, 104, 97, 114, 100, 119, 97, 114, 101, 32, 111, 118, 101, 114, 99, 117, 114, 114, 101, 110, 116, 32, 105, 110, 100, 105, 99, 97, 116, 105, 111, 110, 115, 0];

#[repr(C)]
#[derive(Clone, Copy)]
struct ehci_per_sched {
    udev: *mut usb_device,
    ep: *mut usb_host_endpoint,
    ps_list: list_head,
    tt_usecs: u16,
    cs_mask: u16,
    period: u16,
    phase: u16,
    bw_phase: u8,
    phase_uf: u8,
    usecs: u8,
    c_usecs: u8,
    bw_uperiod: u8,
    bw_period: u8,
}


// frame not assigned yet
// see HCS_N_PORTS
#[repr(C)]
#[derive(Clone, Copy)]
struct ehci_hcd {
    next_hrtimer_event: u32,
    enabled_hrtimer_events: u32,
    hr_timeouts: [i64; 12],
    hrtimer: hrtimer,
    PSS_poll_count: i32,
    ASS_poll_count: i32,
    died_poll_count: i32,
    caps: *mut ehci_caps,
    regs: *mut ehci_regs,
    debug: *mut ehci_dbg_port,
    hcs_params: u32,
    lock: spinlock,
    rh_state: u32,
    __slate_bits_0: __slate_bits::__SlateBits924U0,
    qh_scan_next: *mut ehci_qh,
    r#async: *mut ehci_qh,
    dummy: *mut ehci_qh,
    async_unlink: list_head,
    async_idle: list_head,
    async_unlink_cycle: u32,
    async_count: u32,
    old_current: u32,
    old_token: u32,
    periodic_size: u32,
    periodic: *mut u32,
    periodic_dma: u64,
    intr_qh_list: list_head,
    i_thresh: u32,
    pshadow: *mut ehci_shadow,
    intr_unlink_wait: list_head,
    intr_unlink: list_head,
    intr_unlink_wait_cycle: u32,
    intr_unlink_cycle: u32,
    now_frame: u32,
    last_iso_frame: u32,
    intr_count: u32,
    isoc_count: u32,
    periodic_count: u32,
    uframe_periodic_max: u32,
    cached_itd_list: list_head,
    last_itd_to_free: *mut ehci_itd,
    cached_sitd_list: list_head,
    last_sitd_to_free: *mut ehci_sitd,
    reset_done: [u64; 15],
    bus_suspended: u64,
    companion_ports: u64,
    owned_ports: u64,
    port_c_suspend: u64,
    suspended_ports: u64,
    resuming_ports: u64,
    qh_pool: *mut dma_pool,
    qtd_pool: *mut dma_pool,
    itd_pool: *mut dma_pool,
    sitd_pool: *mut dma_pool,
    random_frame: u32,
    next_statechange: u64,
    last_periodic_enable: i64,
    command: u32,
    __slate_bits_1: __slate_bits::__SlateBits924U1,
    ohci_hcctrl_reg: *mut u32,
    __slate_bits_2: __slate_bits::__SlateBits924U2,
    sbrn: u8,
    bandwidth: [u8; 64],
    tt_budget: [u8; 64],
    tt_list: list_head,
    r#priv: [u64; 0],
}


fn hcd_to_ehci(hcd: *mut usb_hcd) -> *mut ehci_hcd {
(unsafe { core::ptr::addr_of_mut!((*hcd).hcd_priv) as *mut u64 }) as *mut ehci_hcd
}

fn ehci_to_hcd(ehci: *mut ehci_hcd) -> *mut usb_hcd {
container_of__2!(ehci as *mut (), usb_hcd)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ehci_dbg_port {
    control: u32,
    pids: u32,
    data03: u32,
    data47: u32,
    address: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ehci_caps {
    hc_capbase: u32,
    hcs_params: u32,
    hcc_params: u32,
    portroute: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ehci_regs {
    command: u32,
    status: u32,
    intr_enable: u32,
    frame_index: u32,
    segment: u32,
    frame_list: u32,
    async_next: u32,
    reserved1: [u32; 2],
    txfill_tuning: u32,
    reserved2: [u32; 6],
    configured_flag: u32,
    __slate_anon_11: __SlateRecord_ehci_def_h_116,
    __slate_anon_12: __SlateRecord_ehci_def_h_170,
    reserved5: [u32; 2],
    usbmode_ex: u32,
}


#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct ehci_qtd {
    hw_next: u32,
    hw_alt_next: u32,
    hw_token: u32,
    hw_buf: [u32; 5],
    hw_buf_hi: [u32; 5],
    qtd_dma: u64,
    qtd_list: list_head,
    urb: *mut urb,
    length: u64,
}


// Now the following defines are not converted using the
// cpu_to_le32() macro anymore, since we have to support
// "dynamic" switching between be and le support, so that the driver
// can be used on one system with SoC EHCI controller using big-endian
// descriptors as well as a normal little-endian PCI EHCI controller.
//
// values for that type tag
#[repr(C)]
#[derive(Clone, Copy)]
union ehci_shadow {
    qh: *mut ehci_qh,
    itd: *mut ehci_itd,
    sitd: *mut ehci_sitd,
    fstn: *mut ehci_fstn,
    hw_next: *mut u32,
    ptr: *mut (),
}


#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct ehci_qh_hw {
    hw_next: u32,
    hw_info1: u32,
    hw_info2: u32,
    hw_current: u32,
    hw_qtd_next: u32,
    hw_alt_next: u32,
    hw_token: u32,
    hw_buf: [u32; 5],
    hw_buf_hi: [u32; 5],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ehci_qh {
    hw: *mut ehci_qh_hw,
    qh_dma: u64,
    qh_next: ehci_shadow,
    qtd_list: list_head,
    intr_node: list_head,
    dummy: *mut ehci_qtd,
    unlink_node: list_head,
    ps: ehci_per_sched,
    unlink_cycle: u32,
    qh_state: u8,
    xacterrs: u8,
    unlink_reason: u8,
    gap_uf: u8,
    __slate_bits_0: __slate_bits::__SlateBits928U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ehci_iso_packet {
    bufp: u64,
    transaction: u32,
    cross: u8,
    buf1: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ehci_iso_sched {
    td_list: list_head,
    span: u32,
    first_packet: u32,
    packet: [ehci_iso_packet; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ehci_iso_stream {
    hw: *mut ehci_qh_hw,
    bEndpointAddress: u8,
    highspeed: u8,
    td_list: list_head,
    free_list: list_head,
    ps: ehci_per_sched,
    next_uframe: u32,
    splits: u32,
    uperiod: u16,
    maxp: u16,
    bandwidth: u32,
    buf0: u32,
    buf1: u32,
    buf2: u32,
    address: u32,
}


#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct ehci_itd {
    hw_next: u32,
    hw_transaction: [u32; 8],
    hw_bufp: [u32; 7],
    hw_bufp_hi: [u32; 7],
    itd_dma: u64,
    itd_next: ehci_shadow,
    urb: *mut urb,
    stream: *mut ehci_iso_stream,
    itd_list: list_head,
    frame: u32,
    pg: u32,
    index: [u32; 8],
}


#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct ehci_sitd {
    hw_next: u32,
    hw_fullspeed_ep: u32,
    hw_uframe: u32,
    hw_results: u32,
    hw_buf: [u32; 2],
    hw_backpointer: u32,
    hw_buf_hi: [u32; 2],
    sitd_dma: u64,
    sitd_next: ehci_shadow,
    urb: *mut urb,
    stream: *mut ehci_iso_stream,
    sitd_list: list_head,
    frame: u32,
    index: u32,
}


#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct ehci_fstn {
    hw_next: u32,
    hw_prev: u32,
    fstn_dma: u64,
    fstn_next: ehci_shadow,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ehci_tt {
    bandwidth: [u16; 8],
    tt_list: list_head,
    ps_list: list_head,
    usb_tt: *mut usb_tt,
    tt_port: i32,
}


// Port Force Full-Speed Connect
fn ehci_readl(ehci: *const ehci_hcd, regs: *mut u32) -> u32 {
readl(regs as *const ())
}

fn imx28_ehci_writel(val: u32, addr: *mut u32) {
}

fn ehci_writel(ehci: *const ehci_hcd, val: u32, regs: *mut u32) {
if ((unsafe { (*ehci).__slate_bits_1.__get_imx28_write_fix() }) as i32) != 0 {
imx28_ehci_writel(val, regs as *mut u32);
} else {
writel(val, regs as *mut ());
}
}

fn set_ohci_hcfs(ehci: *mut ehci_hcd, operational: i32) {
}

fn cpu_to_hc32(ehci: *const ehci_hcd, x: u32) -> u32 {
x
}

fn hc32_to_cpu(ehci: *const ehci_hcd, x: u32) -> u32 {
x
}

fn hc32_to_cpup(ehci: *const ehci_hcd, x: *const u32) -> u32 {
__le32_to_cpup(x)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ehci_driver_overrides {
    extra_priv_size: u64,
    reset: Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>,
    port_power: Option<unsafe extern "C" fn(*mut usb_hcd, i32, bool) -> i32>,
}


/// The MosChip MCS9990 controller updates its microframe counter
/// a little before the frame counter, and occasionally we will read
/// the invalid intermediate value.  Avoid problems by checking the
/// microframe number (the low-order 3 bits); if they are 0 then
/// re-read the register to get the correct value.
fn ehci_moschip_read_frame_index(ehci: *mut ehci_hcd) -> u32 {
let mut uf: u32 = ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).frame_index) });
if unlikely!(uf & 7 == 0) != 0 {
uf = ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).frame_index) });
}
uf
}

fn ehci_read_frame_index(ehci: *mut ehci_hcd) -> u32 {
if ((unsafe { (*ehci).__slate_bits_1.__get_frame_index_bug() }) as i32) != 0 {
return ehci_moschip_read_frame_index(ehci);
}
ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).frame_index) })
}

fn dbg_hcs_params(ehci: *mut ehci_hcd, label: *mut u8) {
}

fn dbg_hcc_params(ehci: *mut ehci_hcd, label: *mut u8) {
}

fn dbg_status(ehci: *mut ehci_hcd, label: *const u8, status: u32) {
}

fn dbg_cmd(ehci: *mut ehci_hcd, label: *const u8, command: u32) {
}

fn dbg_port(ehci: *mut ehci_hcd, label: *const u8, port: i32, status: u32) {
}

fn create_debug_files(bus: *mut ehci_hcd) {
}

fn remove_debug_files(bus: *mut ehci_hcd) {
}

/// ehci_handshake - spin reading hc until handshake completes or fails
/// @ptr: address of hc register to be read
/// @mask: bits to look at in result of read
/// @done: value of those bits when handshake succeeds
/// @usec: timeout in microseconds
///
/// Returns negative errno, or zero on success
///
/// Success happens when the "mask" bits have the specified value (hardware
/// handshake done).  There are two failure modes:  "usec" have passed (major
/// hardware flakeout), or the register reads as all-ones (hardware removed).
///
/// That last failure should_only happen in cases like physical cardbus eject
/// before driver shutdown. But it also seems to be caused by bugs in cardbus
/// bridge shutdown:  shutting down the bridge before the devices using it.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_handshake")]
extern "C" fn ehci_handshake(ehci: *mut ehci_hcd, ptr: *mut (), mask: u32, done: u32, mut usec: i32) -> i32 {
let mut result: u32 = 0;
loop {
result = ehci_readl(ehci as *const ehci_hcd, ptr as *mut u32);
if result == !0 {
return 19i32.wrapping_neg();
}
// card removed
result &= mask;
if result == done {
return 0;
}
udelay(1u64);
usec = usec.wrapping_sub(1);
if usec <= 0 {
break;
}
}
110i32.wrapping_neg()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ehci_handshake_649: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ehci_hcd, *mut (), u32, u32, i32) -> i32>, *mut ()>(Some(ehci_handshake)) };

/// check TDI/ARC silicon is in host mode
fn tdi_in_host_mode(ehci: *mut ehci_hcd) -> i32 {
let tmp: u32 = ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).__slate_anon_11.__slate_anon_1.usbmode) });
(tmp & 3 == 3u32) as i32
}

/// Force HC to halt state from unknown (EHCI spec section 2.3).
/// Must be called with interrupts enabled and the lock not held.
fn ehci_halt(ehci: *mut ehci_hcd) -> i32 {
let mut temp: u32 = 0;
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
// disable any irqs left enabled by previous code
ehci_writel(ehci as *const ehci_hcd, 0u32, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).intr_enable) });
let __v0: bool = ehci_is_TDI!() != 0 && tdi_in_host_mode(ehci) == 0;
if __v0 {
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
return 0;
}
// This routine gets called during probe before ehci->command
// has been initialized, so we can't rely on its value.
let __v3: u32 = (unsafe { (*ehci).command }) & 4294967294u32;
unsafe {
(*ehci).command = __v3;
}
temp = ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) });
let __v5: u32 = temp & 4294967230u32;
temp = __v5;
ehci_writel(ehci as *const ehci_hcd, temp, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) });
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
unsafe { synchronize_irq((*ehci_to_hcd(ehci)).irq) };
ehci_handshake(ehci, (unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).status) }) as *mut (), 4096u32, 4096u32, 2000i32)
}

/// put TDI/ARC silicon into EHCI mode
fn tdi_reset(ehci: *mut ehci_hcd) {
let mut tmp: u32 = ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).__slate_anon_11.__slate_anon_1.usbmode) });
tmp |= 3u32;
// The default byte access to MMR space is LE after
// controller reset. Set the required endian mode
// for transfer buffers to match the host microprocessor
if ehci_big_endian_mmio!() != 0 {
tmp |= 4u32;
}
ehci_writel(ehci as *const ehci_hcd, tmp, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).__slate_anon_11.__slate_anon_1.usbmode) });
}

/// Reset a non-running (STS_HALT == 1) controller.
/// Must be called with interrupts enabled and the lock not held.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_reset")]
extern "C" fn ehci_reset(ehci: *mut ehci_hcd) -> i32 {
let mut retval: i32 = 0;
let mut command: u32 = ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) });
// If the EHCI debug controller is active, special care must be
// taken before and after a host controller reset
let __v0: bool = !unsafe { (*ehci).debug }.is_null() && (unsafe { dbgp_reset_prep(ehci_to_hcd(ehci)) }) == 0;
if __v0 {
unsafe {
(*ehci).debug = core::ptr::null_mut::<ehci_dbg_port>();
}
}
command |= 2u32;
dbg_cmd(ehci, c"reset".as_ptr() as *const u8, command);
ehci_writel(ehci as *const ehci_hcd, command, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) });
unsafe {
(*ehci).rh_state = 0;
}
unsafe {
(*ehci).next_statechange = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
}
retval = ehci_handshake(ehci, (unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) }) as *mut (), 2u32, 0u32, 250000i32);
if ((unsafe { (*ehci).__slate_bits_2.__get_has_hostpc() }) as i32) != 0 {
ehci_writel(ehci as *const ehci_hcd, 35u32, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).usbmode_ex) });
ehci_writel(ehci as *const ehci_hcd, 524288u32, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).txfill_tuning) });
}
if retval != 0 {
return retval;
}
if ehci_is_TDI!() != 0 {
tdi_reset(ehci);
}
if !unsafe { (*ehci).debug }.is_null() {
unsafe { dbgp_external_startup(ehci_to_hcd(ehci)) };
}
unsafe {
(*ehci).resuming_ports = 0;
}
unsafe {
(*ehci).suspended_ports = 0;
}
unsafe {
(*ehci).port_c_suspend = 0;
}
retval
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ehci_reset_650: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ehci_hcd) -> i32>, *mut ()>(Some(ehci_reset)) };

/// Idle the controller (turn off the schedules).
/// Must be called with interrupts enabled and the lock not held.
fn ehci_quiesce(ehci: *mut ehci_hcd) {
let mut temp: u32 = 0;
if ((unsafe { (*ehci).rh_state }) as u32) != 2 {
return;
}
// wait for any schedule enables/disables to take effect
temp = (unsafe { (*ehci).command }) << 10i32 & 49152u32;
ehci_handshake(ehci, (unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).status) }) as *mut (), 49152u32, temp, 2000i32);
// then disable anything that's still active
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
let __v2: u32 = (unsafe { (*ehci).command }) & 4294967247u32;
unsafe {
(*ehci).command = __v2;
}
ehci_writel(ehci as *const ehci_hcd, unsafe { (*ehci).command }, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) });
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
// hardware can take 16 microframes to turn off ...
ehci_handshake(ehci, (unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).status) }) as *mut (), 49152u32, 0u32, 2000i32);
}

fn ehci_set_command_bit(ehci: *mut ehci_hcd, bit: u32) {
unsafe {
(*ehci).command = (*ehci).command | bit;
}
ehci_writel(ehci as *const ehci_hcd, unsafe { (*ehci).command }, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) });
// unblock posted write
ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) });
}

fn ehci_clear_command_bit(ehci: *mut ehci_hcd, bit: u32) {
let __v2: u32 = (unsafe { (*ehci).command }) & !bit;
unsafe {
(*ehci).command = __v2;
}
ehci_writel(ehci as *const ehci_hcd, unsafe { (*ehci).command }, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) });
// unblock posted write
ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) });
}

static mut event_delays_ns: __SlateAlign16<[u32; 12]> = __SlateAlign16([1000000u32, 1000000u32, 1000000u32, 1125000u32, 2000000u32, 2000000u32, 5000000u32, 6000000u32, 10000000u32, 10000000u32, 15000000u32, 100000000u32]);

fn ehci_enable_event(ehci: *mut ehci_hcd, event: u32, resched: bool) {
let timeout: *mut i64 = unsafe { (*ehci).hr_timeouts.as_mut_ptr() as *mut i64 }.wrapping_offset(event as isize);
if resched {
unsafe {
*timeout = ktime_add!(ktime_get(), unsafe { event_delays_ns.0[event as usize] });
}
}
unsafe {
(*ehci).enabled_hrtimer_events = (*ehci).enabled_hrtimer_events | ((1i32 << event) as u32);
}
// Track only the lowest-numbered pending event
if event < ((unsafe { (*ehci).next_hrtimer_event }) as u32) {
unsafe {
(*ehci).next_hrtimer_event = event as u32;
}
unsafe { hrtimer_start_range_ns(core::ptr::addr_of_mut!((*ehci).hrtimer), *timeout, 1000000u64, 0u32) };
}
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_poll_ASS")]
extern "C" fn ehci_poll_ASS(ehci: *mut ehci_hcd) {
let mut actual: u32 = 0;
let mut want: u32 = 0;
// Don't enable anything if the controller isn't running (e.g., died)
if ((unsafe { (*ehci).rh_state }) as u32) != 2 {
return;
}
want = (if (unsafe { (*ehci).command }) & 32u32 != 0 { 32768i32 } else { 0i32 }) as u32;
actual = ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).status) }) & 32768u32;
if want != actual {
// Poll again later, but give up after about 2-4 ms
let __v1: i32 = unsafe { (*ehci).ASS_poll_count };
unsafe {
(*ehci).ASS_poll_count = __v1.wrapping_add(1);
}
if __v1 < 2 {
ehci_enable_event(ehci, 0u32, true);
return;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.controller as *const device, c"Waited too long for the async schedule status (%x/%x), giving up\n".as_ptr() as *const u8, want, actual) };
}
}
unsafe {
(*ehci).ASS_poll_count = 0;
}
// The status is up-to-date; restart or stop the schedule as needed
if want == 0 {
// Stopped
if (unsafe { (*ehci).async_count }) > 0 {
ehci_set_command_bit(ehci, 32u32);
}
} else {
// Running
if (unsafe { (*ehci).async_count }) == 0 {
// Turn off the schedule after a while
ehci_enable_event(ehci, 10u32, true);
}
}
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_disable_ASE")]
extern "C" fn ehci_disable_ASE(ehci: *mut ehci_hcd) {
ehci_clear_command_bit(ehci, 32u32);
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_poll_PSS")]
extern "C" fn ehci_poll_PSS(ehci: *mut ehci_hcd) {
let mut actual: u32 = 0;
let mut want: u32 = 0;
// Don't do anything if the controller isn't running (e.g., died)
if ((unsafe { (*ehci).rh_state }) as u32) != 2 {
return;
}
want = (if (unsafe { (*ehci).command }) & 16u32 != 0 { 16384i32 } else { 0i32 }) as u32;
actual = ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).status) }) & 16384u32;
if want != actual {
// Poll again later, but give up after about 2-4 ms
let __v1: i32 = unsafe { (*ehci).PSS_poll_count };
unsafe {
(*ehci).PSS_poll_count = __v1.wrapping_add(1);
}
if __v1 < 2 {
ehci_enable_event(ehci, 1u32, true);
return;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.controller as *const device, c"Waited too long for the periodic schedule status (%x/%x), giving up\n".as_ptr() as *const u8, want, actual) };
}
}
unsafe {
(*ehci).PSS_poll_count = 0;
}
// The status is up-to-date; restart or stop the schedule as needed
if want == 0 {
// Stopped
if (unsafe { (*ehci).periodic_count }) > 0 {
ehci_set_command_bit(ehci, 16u32);
}
} else {
// Running
if (unsafe { (*ehci).periodic_count }) == 0 {
// Turn off the schedule after a while
ehci_enable_event(ehci, 9u32, true);
}
}
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_disable_PSE")]
extern "C" fn ehci_disable_PSE(ehci: *mut ehci_hcd) {
ehci_clear_command_bit(ehci, 16u32);
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_handle_controller_death")]
extern "C" fn ehci_handle_controller_death(ehci: *mut ehci_hcd) {
if ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).status) }) & 4096u32 == 0 {
// Give up after a few milliseconds
let __v1: i32 = unsafe { (*ehci).died_poll_count };
unsafe {
(*ehci).died_poll_count = __v1.wrapping_add(1);
}
if __v1 < 5 {
// Try again later
ehci_enable_event(ehci, 2u32, true);
return;
}
while false {
}
unsafe { _dev_warn((*ehci_to_hcd(ehci)).self_.controller as *const device, c"Waited too long for the controller to stop, giving up\n".as_ptr() as *const u8) };
}
// Clean up the mess
unsafe {
(*ehci).rh_state = 0;
}
ehci_writel(ehci as *const ehci_hcd, 0u32, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).configured_flag) });
ehci_writel(ehci as *const ehci_hcd, 0u32, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).intr_enable) });
ehci_work(ehci);
end_unlink_async(ehci);
// Not in process context, so don't try to reset the controller
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_handle_start_intr_unlinks")]
extern "C" fn ehci_handle_start_intr_unlinks(ehci: *mut ehci_hcd) {
let stopped: bool = ((unsafe { (*ehci).rh_state }) as u32) < 2;
// Process all the QHs on the intr_unlink list that were added
// before the current unlink cycle began.  The list is in
// temporal order, so stop when we reach the first entry in the
// current cycle.  But if the root hub isn't running then
// process all the QHs on the list.
while list_empty((unsafe { core::ptr::addr_of_mut!((*ehci).intr_unlink_wait) }) as *const list_head) == 0 {
let mut qh: *mut ehci_qh = unsafe { core::mem::zeroed() };
qh = list_first_entry__7!(unsafe { core::ptr::addr_of_mut!((*ehci).intr_unlink_wait) }, ehci_qh);
if !stopped && (unsafe { (*qh).unlink_cycle }) == unsafe { (*ehci).intr_unlink_wait_cycle } {
break;
}
list_del_init(unsafe { core::ptr::addr_of_mut!((*qh).unlink_node) });
unsafe {
(*qh).unlink_reason = (((*qh).unlink_reason as i32) | 16) as u8;
}
start_unlink_intr(ehci, qh);
}
// Handle remaining entries later
if list_empty((unsafe { core::ptr::addr_of_mut!((*ehci).intr_unlink_wait) }) as *const list_head) == 0 {
ehci_enable_event(ehci, 6u32, true);
let __v6: u32 = unsafe { (*ehci).intr_unlink_wait_cycle };
unsafe {
(*ehci).intr_unlink_wait_cycle = __v6.wrapping_add(1);
}
}
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_handle_intr_unlinks")]
extern "C" fn ehci_handle_intr_unlinks(ehci: *mut ehci_hcd) {
let stopped: bool = ((unsafe { (*ehci).rh_state }) as u32) < 2;
// Process all the QHs on the intr_unlink list that were added
// before the current unlink cycle began.  The list is in
// temporal order, so stop when we reach the first entry in the
// current cycle.  But if the root hub isn't running then
// process all the QHs on the list.
unsafe {
let __t0 = true;
(*ehci).__slate_bits_0.__set_intr_unlinking(__t0);
}
while list_empty((unsafe { core::ptr::addr_of_mut!((*ehci).intr_unlink) }) as *const list_head) == 0 {
let mut qh: *mut ehci_qh = unsafe { core::mem::zeroed() };
qh = list_first_entry__7!(unsafe { core::ptr::addr_of_mut!((*ehci).intr_unlink) }, ehci_qh);
if !stopped && (unsafe { (*qh).unlink_cycle }) == unsafe { (*ehci).intr_unlink_cycle } {
break;
}
list_del_init(unsafe { core::ptr::addr_of_mut!((*qh).unlink_node) });
end_unlink_intr(ehci, qh);
}
// Handle remaining entries later
if list_empty((unsafe { core::ptr::addr_of_mut!((*ehci).intr_unlink) }) as *const list_head) == 0 {
ehci_enable_event(ehci, 3u32, true);
let __v3: u32 = unsafe { (*ehci).intr_unlink_cycle };
unsafe {
(*ehci).intr_unlink_cycle = __v3.wrapping_add(1);
}
}
unsafe {
let __t1 = false;
(*ehci).__slate_bits_0.__set_intr_unlinking(__t1);
}
}

fn start_free_itds(ehci: *mut ehci_hcd) {
if ((unsafe { (*ehci).enabled_hrtimer_events }) as u64) & BIT!(4i32) == 0 {
unsafe {
(*ehci).last_itd_to_free = list_entry__10!(unsafe { (*ehci).cached_itd_list.prev }, ehci_itd);
}
unsafe {
(*ehci).last_sitd_to_free = list_entry__11!(unsafe { (*ehci).cached_sitd_list.prev }, ehci_sitd);
}
ehci_enable_event(ehci, 4u32, true);
}
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.end_free_itds")]
extern "C" fn end_free_itds(ehci: *mut ehci_hcd) {
let mut itd: *mut ehci_itd = unsafe { core::mem::zeroed() };
let mut n: *mut ehci_itd = unsafe { core::mem::zeroed() };
let mut sitd: *mut ehci_sitd = unsafe { core::mem::zeroed() };
let mut sn: *mut ehci_sitd = unsafe { core::mem::zeroed() };
if ((unsafe { (*ehci).rh_state }) as u32) < 2 {
unsafe {
(*ehci).last_itd_to_free = core::ptr::null_mut::<ehci_itd>();
}
unsafe {
(*ehci).last_sitd_to_free = core::ptr::null_mut::<ehci_sitd>();
}
}
itd = list_first_entry__16!(unsafe { core::ptr::addr_of_mut!((*ehci).cached_itd_list) });
n = list_next_entry__15!(itd, itd_list);
while list_entry_is_head!(itd, unsafe { core::ptr::addr_of_mut!((*ehci).cached_itd_list) }, itd_list) == 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*itd).itd_list) });
unsafe { dma_pool_free((*ehci).itd_pool, itd as *mut (), (*itd).itd_dma) };
if itd == unsafe { (*ehci).last_itd_to_free } {
break;
}
itd = n;
n = list_next_entry__15!(n, itd_list);
}
sitd = list_first_entry__17!(unsafe { core::ptr::addr_of_mut!((*ehci).cached_sitd_list) });
sn = list_next_entry__11!(sitd, sitd_list);
while list_entry_is_head!(sitd, unsafe { core::ptr::addr_of_mut!((*ehci).cached_sitd_list) }, sitd_list) == 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*sitd).sitd_list) });
unsafe { dma_pool_free((*ehci).sitd_pool, sitd as *mut (), (*sitd).sitd_dma) };
if sitd == unsafe { (*ehci).last_sitd_to_free } {
break;
}
sitd = sn;
sn = list_next_entry__11!(sn, sitd_list);
}
let __v12: bool = list_empty((unsafe { core::ptr::addr_of_mut!((*ehci).cached_itd_list) }) as *const list_head) == 0 || list_empty((unsafe { core::ptr::addr_of_mut!((*ehci).cached_sitd_list) }) as *const list_head) == 0;
if __v12 {
start_free_itds(ehci);
}
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_iaa_watchdog")]
extern "C" fn ehci_iaa_watchdog(ehci: *mut ehci_hcd) {
let mut cmd: u32 = 0;
let mut status: u32 = 0;
// Lost IAA irqs wedge things badly; seen first with a vt8235.
// So we need this watchdog, but must protect it against both
// (a) SMP races against real IAA firing and retriggering, and
// (b) clean HC shutdown, when IAA watchdog was pending.
if ((unsafe { (*ehci).__slate_bits_0.__get_iaa_in_progress() }) as i32) == 0 || ((unsafe { (*ehci).rh_state }) as u32) != 2 {
return;
}
// If we get here, IAA is *REALLY* late.  It's barely
// conceivable that the system is so busy that CMD_IAAD
// is still legitimately set, so let's be sure it's
// clear before we read STS_IAA.  (The HC should clear
// CMD_IAAD when it sets STS_IAA.)
cmd = ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) });
// If IAA is set here it either legitimately triggered
// after the watchdog timer expired (_way_ late, so we'll
// still count it as lost) ... or a silicon erratum:
// - VIA seems to set IAA without triggering the IRQ;
// - IAAD potentially cleared without setting IAA.
status = ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).status) });
if status & 32u32 != 0 || cmd & 64u32 == 0 {
INCR!();
ehci_writel(ehci as *const ehci_hcd, 32u32, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).status) });
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.controller as *const device, c"IAA watchdog: status %x cmd %x\n".as_ptr() as *const u8, status, cmd) };
}
end_iaa_cycle(ehci);
}

fn turn_on_io_watchdog(ehci: *mut ehci_hcd) {
// Not needed if the controller isn't running or it's already enabled
if ((unsafe { (*ehci).rh_state }) as u32) != 2 || ((unsafe { (*ehci).enabled_hrtimer_events }) as u64) & BIT!(11i32) != 0 {
return;
}
// Isochronous transfers always need the watchdog.
// For other sorts we use it only if the flag is set.
if (unsafe { (*ehci).isoc_count }) > 0 || ((unsafe { (*ehci).__slate_bits_1.__get_need_io_watchdog() }) as i32) != 0 && unsafe { (*ehci).async_count }.wrapping_add(unsafe { (*ehci).intr_count }) > 0 {
ehci_enable_event(ehci, 11u32, true);
}
}

static mut event_handlers: __SlateAlign16<[Option<unsafe extern "C" fn(*mut ehci_hcd)>; 12]> = __SlateAlign16([Some(ehci_poll_ASS), Some(ehci_poll_PSS), Some(ehci_handle_controller_death), Some(ehci_handle_intr_unlinks), Some(end_free_itds), Some(end_unlink_async), Some(ehci_handle_start_intr_unlinks), Some(unlink_empty_async), Some(ehci_iaa_watchdog), Some(ehci_disable_PSE), Some(ehci_disable_ASE), Some(ehci_work)]);

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_hrtimer_func")]
extern "C" fn ehci_hrtimer_func(t: *mut hrtimer) -> u32 {
let ehci: *mut ehci_hcd = container_of__55!(t, ehci_hcd);
let mut now: i64 = 0;
let mut events: u64 = 0;
let mut flags: u64 = 0;
let mut e: u32 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*ehci).lock) }, flags);
events = (unsafe { (*ehci).enabled_hrtimer_events }) as u64;
unsafe {
(*ehci).enabled_hrtimer_events = 0;
}
unsafe {
(*ehci).next_hrtimer_event = 99;
}
// Check each pending event.  If its time has expired, handle
// the event; otherwise re-enable it.
now = unsafe { ktime_get() };
loop {
e = find_next_bit(core::ptr::addr_of_mut!(events) as *const u64, 12u64, e as u64) as u32;
if e >= 12 {
break;
}
if ktime_compare(now, unsafe { *((*ehci).hr_timeouts.as_mut_ptr() as *mut i64).wrapping_offset(e as isize) }) >= 0 {
unsafe { unsafe { event_handlers.0[e as usize] }.unwrap()(ehci) };
} else {
ehci_enable_event(ehci, e, false);
}
e = e.wrapping_add(1);
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*ehci).lock) }, flags);
0
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.persist_enabled_on_companion")]
extern "C" fn persist_enabled_on_companion(udev: *mut usb_device, unused: *mut ()) -> i32 {
((unsafe { (*udev).maxchild }) == 0 && ((unsafe { (*udev).__slate_bits_0.__get_persist_enabled() }) as i32) != 0 && ((unsafe { (*(*(*udev).bus).root_hub).speed }) as u32) < 3) as i32
}

fn ehci_handover_companion_ports(ehci: *mut ehci_hcd) {
let mut reg: *mut u32 = unsafe { core::mem::zeroed() };
let mut status: u32 = 0;
let mut port: i32 = 0;
let mut buf: u32 = 0;
let hcd: *mut usb_hcd = ehci_to_hcd(ehci);
if (unsafe { (*ehci).owned_ports }) == 0 {
return;
}
// USB 1.1 devices are mostly HIDs, which don't need to persist across
// suspends. If we ensure that none of our companion's devices have
// persist_enabled (by looking through all USB 1.1 buses in the system),
// we can skip this and avoid slowing resume down. Devices without
// persist will just get reenumerated shortly after resume anyway.
if (unsafe { usb_for_each_dev(core::ptr::null_mut::<()>(), Some(persist_enabled_on_companion)) }) == 0 {
return;
}
// Make sure the ports are powered
port = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
loop {
let __v1: i32 = port;
port = __v1.wrapping_sub(1);
if __v1 == 0 {
break;
}
if test_bit!(port, unsafe { core::ptr::addr_of_mut!((*ehci).owned_ports) }) != 0 {
reg = unsafe { (*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32 }.wrapping_offset(port as isize);
status = ehci_readl(ehci as *const ehci_hcd, reg) & 4294967253u32;
if status & 4096u32 == 0 {
ehci_port_power(ehci, port, true);
}
}
}
// Give the connections some time to appear
unsafe { msleep(20u32) };
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
port = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
loop {
let __v6: i32 = port;
port = __v6.wrapping_sub(1);
if __v6 == 0 {
break;
}
if test_bit!(port, unsafe { core::ptr::addr_of_mut!((*ehci).owned_ports) }) != 0 {
reg = unsafe { (*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32 }.wrapping_offset(port as isize);
status = ehci_readl(ehci as *const ehci_hcd, reg) & 4294967253u32;
// Port already owned by companion?
if status & 8192u32 != 0 {
clear_bit(port as i64, (unsafe { core::ptr::addr_of_mut!((*ehci).owned_ports) }) as *mut u64);
} else {
if test_bit!(port, unsafe { core::ptr::addr_of_mut!((*ehci).companion_ports) }) != 0 {
ehci_writel(ehci as *const ehci_hcd, status & 4294967291u32, reg);
} else {
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
ehci_hub_control(hcd, HUB_CLASS_REQ!(0i32, 35i32, 3i32) as u16, 4u16, port.wrapping_add(1) as u16, core::ptr::null_mut::<u8>(), 0u16);
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
}
}
}
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
if (unsafe { (*ehci).owned_ports }) == 0 {
return;
}
unsafe { msleep(90u32) }; // Wait for resets to complete
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
port = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
loop {
let __v10: i32 = port;
port = __v10.wrapping_sub(1);
if __v10 == 0 {
break;
}
if test_bit!(port, unsafe { core::ptr::addr_of_mut!((*ehci).owned_ports) }) != 0 {
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
ehci_hub_control(hcd, HUB_CLASS_REQ!(128i32, 35i32, 0i32) as u16, 0u16, port.wrapping_add(1) as u16, core::ptr::addr_of_mut!(buf) as *mut u8, core::mem::size_of::<u32>() as u16);
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
// The companion should now own the port,
// but if something went wrong the port must not
// remain enabled.
reg = unsafe { (*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32 }.wrapping_offset(port as isize);
status = ehci_readl(ehci as *const ehci_hcd, reg) & 4294967253u32;
if status & 8192u32 != 0 {
ehci_writel(ehci as *const ehci_hcd, status | 2u32, reg);
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.controller as *const device, c"failed handover port %d: %x\n".as_ptr() as *const u8, port.wrapping_add(1), status) };
}
ehci_writel(ehci as *const ehci_hcd, status & 4294967291u32, reg);
}
}
}
unsafe {
(*ehci).owned_ports = 0;
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
}

fn ehci_port_change(ehci: *mut ehci_hcd) -> i32 {
let mut i: i32 = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
// First check if the controller indicates a change event
if ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).status) }) & 4u32 != 0 {
return 1;
}
// Not all controllers appear to update this while going from D3 to D0,
// so check the individual port status registers as well
loop {
let __v0: i32 = i;
i = __v0.wrapping_sub(1);
if __v0 == 0 {
break;
}
if ehci_readl(ehci as *const ehci_hcd, unsafe { (*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32 }.wrapping_offset(i as isize)) & 2u32 != 0 {
return 1;
}
}
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_adjust_port_wakeup_flags")]
extern "C" fn ehci_adjust_port_wakeup_flags(ehci: *mut ehci_hcd, suspending: bool, do_wakeup: bool) {
let mut port: i32 = 0;
let mut temp: u32 = 0;
// If remote wakeup is enabled for the root hub but disabled
// for the controller, we must adjust all the port wakeup flags
// when the controller is suspended or resumed.  In all other
// cases they don't need to be changed.
if ((unsafe { (*(*ehci_to_hcd(ehci)).self_.root_hub).__slate_bits_1.__get_do_remote_wakeup() }) as i32) == 0 || do_wakeup {
return;
}
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
// clear phy low-power mode before changing wakeup flags
if ((unsafe { (*ehci).__slate_bits_2.__get_has_tdi_phy_lpm() }) as i32) != 0 {
port = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
loop {
let __v0: i32 = port;
port = __v0.wrapping_sub(1);
if __v0 == 0 {
break;
}
let hostpc_reg: *mut u32 = unsafe { (*(*ehci).regs).__slate_anon_12.__slate_anon_0.hostpc.as_mut_ptr() as *mut u32 }.wrapping_offset(port as isize);
temp = ehci_readl(ehci as *const ehci_hcd, hostpc_reg);
ehci_writel(ehci as *const ehci_hcd, temp & 4290772991u32, hostpc_reg);
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
unsafe { msleep(5u32) };
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
}
port = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
loop {
let __v7: i32 = port;
port = __v7.wrapping_sub(1);
if __v7 == 0 {
break;
}
let reg: *mut u32 = unsafe { (*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32 }.wrapping_offset(port as isize);
let t1: u32 = ehci_readl(ehci as *const ehci_hcd, reg) & 4294967253u32;
let mut t2: u32 = t1 & 4287627263u32;
// If we are suspending the controller, clear the flags.
// If we are resuming the controller, set the wakeup flags.
if !suspending {
if t1 & 1u32 != 0 {
t2 |= 6291456u32;
} else {
t2 |= 5242880u32;
}
}
ehci_writel(ehci as *const ehci_hcd, t2, reg);
}
// enter phy low-power mode again
if ((unsafe { (*ehci).__slate_bits_2.__get_has_tdi_phy_lpm() }) as i32) != 0 {
port = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
loop {
let __v10: i32 = port;
port = __v10.wrapping_sub(1);
if __v10 == 0 {
break;
}
let hostpc_reg: *mut u32 = unsafe { (*(*ehci).regs).__slate_anon_12.__slate_anon_0.hostpc.as_mut_ptr() as *mut u32 }.wrapping_offset(port as isize);
temp = ehci_readl(ehci as *const ehci_hcd, hostpc_reg);
ehci_writel(ehci as *const ehci_hcd, temp | 4194304u32, hostpc_reg);
}
}
// Does the root hub have a port wakeup pending?
let __v13: bool = !suspending && ehci_port_change(ehci) != 0;
if __v13 {
unsafe { usb_hcd_resume_root_hub(ehci_to_hcd(ehci)) };
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ehci_adjust_port_wakeup_flags_651: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ehci_hcd, bool, bool)>, *mut ()>(Some(ehci_adjust_port_wakeup_flags)) };

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_bus_suspend")]
extern "C" fn ehci_bus_suspend(hcd: *mut usb_hcd) -> i32 {
let mut t3: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(t3) as *mut u32;
let mut t2: u32 = unsafe { core::mem::zeroed() };
let mut t1: u32 = unsafe { core::mem::zeroed() };
let mut reg: *mut u32 = unsafe { core::mem::zeroed() };
let mut fs_idle_delay: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut bool = core::ptr::addr_of_mut!(fs_idle_delay) as *mut bool;
let mut changed: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(changed) as *mut i32;
let mut mask: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(mask) as *mut i32;
let mut port: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut i32 = core::ptr::addr_of_mut!(port) as *mut i32;
let mut ehci: *mut ehci_hcd = unsafe { core::mem::zeroed() };
unsafe {
{
ehci = hcd_to_ehci(hcd);
let __slate_auto_bytes_1 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
if time_before!(core::ptr::read_volatile(core::ptr::addr_of!(jiffies)), unsafe { (*ehci).next_statechange }) {
msleep(5u32);
}
}
// stop the schedules
ehci_quiesce(ehci);
spin_lock_irq(core::ptr::addr_of_mut!((*ehci).lock));
if ((*ehci).rh_state as u32) >= 2 {
// Once the controller is stopped, port resumes that are already
// in progress won't complete.  Hence if remote wakeup is enabled
// for the root hub and any ports are in the middle of a resume or
// remote wakeup, we must fail the suspend.
if ((*(*hcd).self_.root_hub).__slate_bits_1.__get_do_remote_wakeup() as i32) != 0 {
if (*ehci).resuming_ports != 0 {
spin_unlock_irq(core::ptr::addr_of_mut!((*ehci).lock));
return 16i32.wrapping_neg();
}
}
// Unlike other USB host controller types, EHCI doesn't have
// any notion of "global" or bus-wide suspend.  The driver has
// to manually suspend all the active unsuspended ports, and
// then manually resume them in the bus_resume() routine.
(*ehci).bus_suspended = 0;
(*ehci).owned_ports = 0;
*__slate_slot_18 = 0;
*__slate_slot_17 = false;
*__slate_slot_20 = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
loop {
let __v7: i32 = *__slate_slot_20;
*__slate_slot_20 = __v7.wrapping_sub(1);
if __v7 != 0 {
reg = ((*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_20 as isize);
t1 = ehci_readl(ehci as *const ehci_hcd, reg) & 4294967253u32;
t2 = t1 & 4287627263u32;
// keep track of which ports we suspend
if t1 & 8192u32 != 0 {
set_bit(*__slate_slot_20 as i64, core::ptr::addr_of_mut!((*ehci).owned_ports) as *mut u64);
} else {
if t1 & 4u32 != 0 && t1 & 128u32 == 0 {
t2 |= 128u32;
set_bit(*__slate_slot_20 as i64, core::ptr::addr_of_mut!((*ehci).bus_suspended) as *mut u64);
}
}
// enable remote wakeup on all ports, if told to do so
if ((*(*hcd).self_.root_hub).__slate_bits_1.__get_do_remote_wakeup() as i32) != 0 {
// only enable appropriate wake bits, otherwise the
// hardware can not go phy low power mode. If a race
// condition happens here(connection change during bits
// set), the port change detection will finally fix it.
if t1 & 1u32 != 0 {
t2 |= 6291456u32;
} else {
t2 |= 5242880u32;
}
}
if t1 != t2 {
// On some controllers, Wake-On-Disconnect will
// generate false wakeup signals until the bus
// switches over to full-speed idle.  For their
// sake, add a delay if we need one.
if t2 & 2097152u32 != 0 && ehci_port_speed!() == 1024 {
*__slate_slot_17 = true;
}
ehci_writel(ehci as *const ehci_hcd, t2, reg);
*__slate_slot_18 = 1;
}
} else {
break;
}
}
spin_unlock_irq(core::ptr::addr_of_mut!((*ehci).lock));
if *__slate_slot_18 != 0 && ehci_has_fsl_susp_errata!(ehci) != 0 {
usleep_range(10000u64, 20000u64);
}
// Wait for at least 10 millisecondes to ensure the controller
// enter the suspend status before initiating a port resume
// using the Force Port Resume bit (Not-EHCI compatible).
if *__slate_slot_18 != 0 && ((*ehci).__slate_bits_2.__get_has_tdi_phy_lpm() as i32) != 0 || *__slate_slot_17 {
// Wait for HCD to enter low-power mode or for the bus
// to switch to full-speed idle.
usleep_range(5000u64, 5500u64);
}
if *__slate_slot_18 != 0 && ((*ehci).__slate_bits_2.__get_has_tdi_phy_lpm() as i32) != 0 {
spin_lock_irq(core::ptr::addr_of_mut!((*ehci).lock));
*__slate_slot_20 = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
loop {
let __v3: i32 = *__slate_slot_20;
*__slate_slot_20 = __v3.wrapping_sub(1);
if __v3 != 0 {
let hostpc_reg: *mut u32 = ((*(*ehci).regs).__slate_anon_12.__slate_anon_0.hostpc.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_20 as isize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_4 = ehci_readl(ehci as *const ehci_hcd, hostpc_reg);
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_4 | 4194304u32, hostpc_reg);
*__slate_slot_4 = ehci_readl(ehci as *const ehci_hcd, hostpc_reg);
} else {
break;
}
}
spin_unlock_irq(core::ptr::addr_of_mut!((*ehci).lock));
}
// Apparently some devices need a >= 1-uframe delay here
if (*ehci).bus_suspended != 0 {
udelay(150u64);
}
// turn off now-idle HC
ehci_halt(ehci);
spin_lock_irq(core::ptr::addr_of_mut!((*ehci).lock));
if ((*ehci).enabled_hrtimer_events as u64) & BIT!(2i32) != 0 {
ehci_handle_controller_death(ehci);
}
if ((*ehci).rh_state as u32) == 2 {
(*ehci).rh_state = 1;
unlink_empty_async_suspended(ehci);
// Some Synopsys controllers mistakenly leave IAA turned on
ehci_writel(ehci as *const ehci_hcd, 32u32, core::ptr::addr_of_mut!((*(*ehci).regs).status));
// Any IAA cycle that started before the suspend is now invalid
end_iaa_cycle(ehci);
ehci_handle_start_intr_unlinks(ehci);
ehci_handle_intr_unlinks(ehci);
end_free_itds(ehci);
// allow remote wakeup
*__slate_slot_19 = 55i32;
if ((*(*hcd).self_.root_hub).__slate_bits_1.__get_do_remote_wakeup() as i32) == 0 {
let __v0: i32 = *__slate_slot_19 & -5i32;
*__slate_slot_19 = __v0;
}
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_19 as u32, core::ptr::addr_of_mut!((*(*ehci).regs).intr_enable));
ehci_readl(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*(*ehci).regs).intr_enable));
}
}
(*ehci).next_statechange = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(10u32));
(*ehci).enabled_hrtimer_events = 0;
(*ehci).next_hrtimer_event = 99;
spin_unlock_irq(core::ptr::addr_of_mut!((*ehci).lock));
hrtimer_cancel(core::ptr::addr_of_mut!((*ehci).hrtimer));
return 0;
}
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_bus_resume")]
extern "C" fn ehci_bus_resume(hcd: *mut usb_hcd) -> i32 {
let mut resume_needed: u64 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut power_okay: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u32 = core::ptr::addr_of_mut!(power_okay) as *mut u32;
let mut temp: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u32 = core::ptr::addr_of_mut!(temp) as *mut u32;
let mut ehci: *mut ehci_hcd = unsafe { core::mem::zeroed() };
unsafe {
ehci = hcd_to_ehci(hcd);
let __slate_auto_bytes = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
resume_needed = 0;
if time_before!(core::ptr::read_volatile(core::ptr::addr_of!(jiffies)), unsafe { (*ehci).next_statechange }) {
msleep(5u32);
}
'__join_0: {
spin_lock_irq(core::ptr::addr_of_mut!((*ehci).lock));
if !(HCD_HW_ACCESSIBLE!(hcd) == 0 || ((*ehci).__slate_bits_0.__get_shutdown() as i32) != 0) {
if unlikely__5!(unsafe { (*ehci).debug }) != 0 {
if dbgp_reset_prep(hcd) == 0 {
(*ehci).debug = core::ptr::null_mut::<ehci_dbg_port>();
} else {
dbgp_external_startup(hcd);
}
}
// Ideally and we've got a real resume here, and no port's power
// was lost.  (For PCI, that means Vaux was maintained.)  But we
// could instead be restoring a swsusp snapshot -- so that BIOS was
// the last user of the controller, not reset/pm hardware keeping
// state we gave to it.
*__slate_slot_20 = ehci_readl(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*(*ehci).regs).intr_enable));
// at least some APM implementations will try to deliver
// IRQs right away, so delay them until we're ready.
ehci_writel(ehci as *const ehci_hcd, 0u32, core::ptr::addr_of_mut!((*(*ehci).regs).intr_enable));
// re-init operational registers
ehci_writel(ehci as *const ehci_hcd, 0u32, core::ptr::addr_of_mut!((*(*ehci).regs).segment));
ehci_writel(ehci as *const ehci_hcd, (*ehci).periodic_dma as u32, core::ptr::addr_of_mut!((*(*ehci).regs).frame_list));
ehci_writel(ehci as *const ehci_hcd, (*(*ehci).r#async).qh_dma as u32, core::ptr::addr_of_mut!((*(*ehci).regs).async_next));
// restore CMD_RUN, framelist size, and irq threshold
(*ehci).command = (*ehci).command | 1u32;
ehci_writel(ehci as *const ehci_hcd, (*ehci).command, core::ptr::addr_of_mut!((*(*ehci).regs).command));
(*ehci).rh_state = 2;
// According to Bugzilla #8190, the port status for some controllers
// will be wrong without a delay. At their wrong status, the port
// is enabled, but not suspended neither resumed.
*__slate_slot_19 = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
'__join_20: {
loop {
let __v14: i32 = *__slate_slot_19;
*__slate_slot_19 = __v14.wrapping_sub(1);
if __v14 != 0 {
*__slate_slot_21 = ehci_readl(ehci as *const ehci_hcd, ((*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_19 as isize));
if *__slate_slot_21 & 4u32 != 0 && *__slate_slot_21 & 192u32 == 0 {
break;
}
} else {
break '__join_20;
}
}
spin_unlock_irq(core::ptr::addr_of_mut!((*ehci).lock));
msleep(8u32);
spin_lock_irq(core::ptr::addr_of_mut!((*ehci).lock));
}
if ((*ehci).__slate_bits_0.__get_shutdown() as i32) == 0 {
// clear phy low-power mode before resume
if (*ehci).bus_suspended != 0 && ((*ehci).__slate_bits_2.__get_has_tdi_phy_lpm() as i32) != 0 {
*__slate_slot_19 = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
loop {
let __v11: i32 = *__slate_slot_19;
*__slate_slot_19 = __v11.wrapping_sub(1);
if __v11 != 0 {
if test_bit!(*__slate_slot_19, core::ptr::addr_of_mut!((*ehci).bus_suspended)) != 0 {
let hostpc_reg: *mut u32 = ((*(*ehci).regs).__slate_anon_12.__slate_anon_0.hostpc.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_19 as isize);
*__slate_slot_21 = ehci_readl(ehci as *const ehci_hcd, hostpc_reg);
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_21 & 4290772991u32, hostpc_reg);
}
} else {
break;
}
}
spin_unlock_irq(core::ptr::addr_of_mut!((*ehci).lock));
msleep(5u32);
spin_lock_irq(core::ptr::addr_of_mut!((*ehci).lock));
if ((*ehci).__slate_bits_0.__get_shutdown() as i32) != 0 {
break '__join_0;
}
}
// manually resume the ports we suspended during bus_suspend()
*__slate_slot_19 = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
loop {
let __v5: i32 = *__slate_slot_19;
*__slate_slot_19 = __v5.wrapping_sub(1);
if __v5 != 0 {
*__slate_slot_21 = ehci_readl(ehci as *const ehci_hcd, ((*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_19 as isize));
let __v8: u32 = *__slate_slot_21 & 4287627221u32;
*__slate_slot_21 = __v8;
if test_bit!(*__slate_slot_19, core::ptr::addr_of_mut!((*ehci).bus_suspended)) != 0 && *__slate_slot_21 & 128u32 != 0 {
*__slate_slot_21 |= 64u32;
set_bit(*__slate_slot_19 as i64, core::ptr::addr_of_mut!(resume_needed) as *mut u64);
}
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_21, ((*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_19 as isize));
} else {
break;
}
}
// msleep for USB_RESUME_TIMEOUT ms only if code is trying to resume
// port
if resume_needed != 0 {
spin_unlock_irq(core::ptr::addr_of_mut!((*ehci).lock));
msleep(40u32);
spin_lock_irq(core::ptr::addr_of_mut!((*ehci).lock));
if ((*ehci).__slate_bits_0.__get_shutdown() as i32) != 0 {
break '__join_0;
}
}
*__slate_slot_19 = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
loop {
let __v1: i32 = *__slate_slot_19;
*__slate_slot_19 = __v1.wrapping_sub(1);
if __v1 != 0 {
*__slate_slot_21 = ehci_readl(ehci as *const ehci_hcd, ((*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_19 as isize));
if test_bit!(*__slate_slot_19, core::ptr::addr_of_mut!(resume_needed)) != 0 {
let __v2: u32 = *__slate_slot_21 & 4294967061u32;
*__slate_slot_21 = __v2;
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_21, ((*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_19 as isize));
}
} else {
break;
}
}
(*ehci).next_statechange = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(5u32));
spin_unlock_irq(core::ptr::addr_of_mut!((*ehci).lock));
ehci_handover_companion_ports(ehci);
// Now we can safely re-enable irqs
spin_lock_irq(core::ptr::addr_of_mut!((*ehci).lock));
if ((*ehci).__slate_bits_0.__get_shutdown() as i32) == 0 {
ehci_writel(ehci as *const ehci_hcd, 55u32, core::ptr::addr_of_mut!((*(*ehci).regs).intr_enable));
ehci_readl(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*(*ehci).regs).intr_enable));
spin_unlock_irq(core::ptr::addr_of_mut!((*ehci).lock));
return 0;
}
}
}
}
spin_unlock_irq(core::ptr::addr_of_mut!((*ehci).lock));
return 108i32.wrapping_neg();
}
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_get_resuming_ports")]
extern "C" fn ehci_get_resuming_ports(hcd: *mut usb_hcd) -> u64 {
let ehci: *mut ehci_hcd = hcd_to_ehci(hcd);
unsafe { (*ehci).resuming_ports }
}

fn set_owner(ehci: *mut ehci_hcd, portnum: i32, new_owner: i32) {
let mut port_status: u32 = 0;
let mut r#try: i32 = 4;
let status_reg: *mut u32 = unsafe { (*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32 }.wrapping_offset(portnum as isize);
// The controller won't set the OWNER bit if the port is
// enabled, so this loop will sometimes require at least two
// iterations: one to disable the port and one to set OWNER.
while r#try > 0 {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
port_status = ehci_readl(ehci as *const ehci_hcd, status_reg);
if port_status & 8192u32 == (new_owner as u32) || port_status & 8193u32 == 0 {
r#try = 0;
} else {
port_status ^= 8192u32;
let __v3: u32 = port_status & 4294967249u32;
port_status = __v3;
ehci_writel(ehci as *const ehci_hcd, port_status, status_reg);
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
if r#try > 1 {
unsafe { msleep(5u32) };
}
r#try = r#try.wrapping_sub(1);
}
}

fn check_reset_complete(ehci: *mut ehci_hcd, index: i32, status_reg: *mut u32, mut port_status: i32) -> i32 {
if port_status & 1i32 == 0 {
return port_status;
}
// if reset finished and it's still not enabled -- handoff
if port_status & 4i32 == 0 {
// with integrated TT, there's nobody to hand it to!
if ehci_is_TDI!() != 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.controller as *const device, c"Failed to enable port %d on root hub TT\n".as_ptr() as *const u8, index.wrapping_add(1)) };
}
return port_status;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.controller as *const device, c"port %d full speed --> companion\n".as_ptr() as *const u8, index.wrapping_add(1)) };
}
// what happens if HCS_N_CC(params) == 0 ?
port_status |= 8192i32;
let __v3: i32 = port_status & -43i32;
port_status = __v3;
ehci_writel(ehci as *const ehci_hcd, port_status as u32, status_reg);
// ensure 440EPX ohci controller state is operational
if ((unsafe { (*ehci).__slate_bits_1.__get_has_amcc_usb23() }) as i32) != 0 {
set_ohci_hcfs(ehci, 1i32);
}
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.controller as *const device, c"port %d reset complete, port enabled\n".as_ptr() as *const u8, index.wrapping_add(1)) };
}
// ensure 440EPx ohci controller state is suspended
if ((unsafe { (*ehci).__slate_bits_1.__get_has_amcc_usb23() }) as i32) != 0 {
set_ohci_hcfs(ehci, 0i32);
}
}
port_status
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_hub_status_data")]
extern "C" fn ehci_hub_status_data(hcd: *mut usb_hcd, buf: *mut u8) -> i32 {
let ehci: *mut ehci_hcd = hcd_to_ehci(hcd);
let mut temp: u32 = 0;
let mut status: u32 = 0;
let mut mask: u32 = 0;
let mut ports: i32 = 0;
let mut retval: i32 = 1;
let mut flags: u64 = 0;
let mut ppcd: u32 = 4294967295u32;
// init status to no-changes
unsafe {
*buf.wrapping_offset(0 as isize) = 0;
}
ports = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
if ports > 7 {
unsafe {
*buf.wrapping_offset(1 as isize) = 0;
}
retval = retval.wrapping_add(1);
}
// Inform the core about resumes-in-progress by returning
// a non-zero value even if there are no status changes.
status = (unsafe { (*ehci).resuming_ports }) as u32;
// Some boards (mostly VIA?) report bogus overcurrent indications,
// causing massive log spam unless we completely ignore them.  It
// may be relevant that VIA VT8235 controllers, where PORT_POWER is
// always set, seem to clear PORT_OCC and PORT_CSC when writing to
// PORT_POWER; that's surprising, but maybe within-spec.
if !unsafe { ignore_oc } && ((unsafe { (*ehci).__slate_bits_1.__get_spurious_oc() }) as i32) == 0 {
mask = 42u32;
} else {
mask = 10u32;
}
// PORT_RESUME from hardware ~= PORT_STAT_C_SUSPEND
// no hub change reports (bit 0) for now (power, ...)
// port N changes (bit N)?
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*ehci).lock) }, flags);
// get per-port change detect bits
if ((unsafe { (*ehci).__slate_bits_2.__get_has_ppcd() }) as i32) != 0 {
ppcd = ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).status) }) >> 16i32;
}
for i in 0..ports {
// leverage per-port change bits feature
if ppcd & ((1i32 << i) as u32) != 0 {
temp = ehci_readl(ehci as *const ehci_hcd, unsafe { (*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32 }.wrapping_offset(i as isize));
} else {
temp = 0;
}
// Return status information even for ports with OWNER set.
// Otherwise hub_wq wouldn't see the disconnect event when a
// high-speed device is switched over to the companion
// controller by the user.
let __v7: bool = temp & mask != 0 || test_bit!(i, unsafe { core::ptr::addr_of_mut!((*ehci).port_c_suspend) }) != 0;
let __v9: bool = __v7 || (unsafe { *((*ehci).reset_done.as_mut_ptr() as *mut u64).wrapping_offset(i as isize) }) != 0 && time_after_eq!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, unsafe { *((*ehci).reset_done.as_mut_ptr() as *mut u64).wrapping_offset(i as isize) });
if __v9 || ehci_has_ci_pec_bug!(ehci, temp) {
if i < 7 {
let __v18: *mut u8 = buf.wrapping_offset(0 as isize);
unsafe {
*__v18 = ((*__v18 as i32) | 1i32 << i.wrapping_add(1)) as u8;
}
} else {
let __v21: *mut u8 = buf.wrapping_offset(1 as isize);
unsafe {
*__v21 = ((*__v21 as i32) | 1i32 << i.wrapping_sub(7)) as u8;
}
}
status = 4u32;
}
}
// If a resume is in progress, make sure it can finish
if (unsafe { (*ehci).resuming_ports }) != 0 {
unsafe { mod_timer(core::ptr::addr_of_mut!((*hcd).rh_timer), core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(25u32))) };
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*ehci).lock) }, flags);
if status != 0 { retval } else { 0 }
}

fn ehci_hub_descriptor(ehci: *mut ehci_hcd, desc: *mut usb_hub_descriptor) {
let ports: i32 = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
let mut temp: u16 = 0;
unsafe {
(*desc).bDescriptorType = 41u8;
}
unsafe {
(*desc).bPwrOn2PwrGood = 10;
}
// ehci 1.0, 2.3.9 says 20ms max
unsafe {
(*desc).bHubContrCurrent = 0;
}
unsafe {
(*desc).bNbrPorts = ports as u8;
}
temp = 1i32.wrapping_add(ports / 8) as u16;
unsafe {
(*desc).bDescLength = 7i32.wrapping_add(2i32.wrapping_mul(temp as i32)) as u8;
}
// two bitmaps:  ports removable, and usb 1.0 legacy PortPwrCtrlMask
unsafe { memset((core::ptr::addr_of_mut!((*desc).u.hs.DeviceRemovable) as *mut u8).wrapping_offset(0 as isize) as *mut (), 0i32, temp as u64) };
unsafe { memset((core::ptr::addr_of_mut!((*desc).u.hs.DeviceRemovable) as *mut u8).wrapping_offset((temp as i32) as isize) as *mut (), 255i32, temp as u64) };
temp = 8; // per-port overcurrent reporting
if HCS_PPC!(unsafe { (*ehci).hcs_params }) != 0 {
temp = ((temp as i32) | 1) as u16;
} else {
temp = ((temp as i32) | 2) as u16;
}
// per-port power control
// no power switching
unsafe {
(*desc).wHubCharacteristics = temp;
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_hub_control")]
extern "C" fn ehci_hub_control(hcd: *mut usb_hcd, typeReq: u16, wValue: u16, mut wIndex: u16, buf: *mut u8, wLength: u16) -> i32 {
let mut sreg: *mut u32 = unsafe { core::mem::zeroed() };
// If the debug port is active any port
// feature requests should get denied
let mut __v21: bool = unsafe { core::mem::zeroed() };
// transfer dedicated ports to the companion hc
let mut __v46: bool = unsafe { core::mem::zeroed() };
let mut selector: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_67: *mut u32 = core::ptr::addr_of_mut!(selector) as *mut u32;
let mut retval: i32 = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_69: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_70: *mut u32 = core::ptr::addr_of_mut!(status) as *mut u32;
let mut temp1: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_71: *mut u32 = core::ptr::addr_of_mut!(temp1) as *mut u32;
let mut temp: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_72: *mut u32 = core::ptr::addr_of_mut!(temp) as *mut u32;
let mut hostpc_reg: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_73: *mut *mut u32 = core::ptr::addr_of_mut!(hostpc_reg) as *mut *mut u32;
let mut status_reg: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_74: *mut *mut u32 = core::ptr::addr_of_mut!(status_reg) as *mut *mut u32;
let mut ports: i32 = unsafe { core::mem::zeroed() };
let mut ehci: *mut ehci_hcd = unsafe { core::mem::zeroed() };
unsafe {
ehci = hcd_to_ehci(hcd);
ports = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
let __slate_auto_bytes = __slate_slot_74 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_73 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_72 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_71 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_70 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_69 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
retval = 0;
let __slate_auto_bytes = __slate_slot_67 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// Avoid out-of-bounds values while calculating the port index
// from wIndex.  The compiler doesn't like pointers to invalid
// addresses, even if they are never used.
*__slate_slot_72 = ((wIndex as i32).wrapping_sub(1) & 255) as u32;
if *__slate_slot_72 >= 15 {
*__slate_slot_72 = 0;
}
'__join_0: {
'__join_1: {
*__slate_slot_74 = ((*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_72 as isize);
*__slate_slot_73 = ((*(*ehci).regs).__slate_anon_12.__slate_anon_0.hostpc.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_72 as isize);
// FIXME:  support SetPortFeatures USB_PORT_FEAT_INDICATOR.
// HCS_INDICATOR may say we can change LEDs to off/amber/green.
// (track current state ourselves) ... blink for diagnostics,
// power, "this is the one", etc.  EHCI spec supports this.
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
let __t4: i32 = typeReq as i32;
if __t4 == 8193 {
let __t3: i32 = wValue as i32;
if __t3 == 0 {
break '__join_0;
} else {
if __t3 == 1 {
break '__join_0;
}
}
} else {
if __t4 == 8961 {
if !(wIndex == 0 || (wIndex as i32) > ports) {
wIndex = (wIndex as i32).wrapping_sub(1) as u16;
*__slate_slot_72 = ehci_readl(ehci as *const ehci_hcd, *__slate_slot_74);
let __v63: u32 = *__slate_slot_72 & 4294967253u32;
*__slate_slot_72 = __v63;
// Even if OWNER is set, so the port is owned by the
// companion controller, hub_wq needs to be able to clear
// the port-change status bits (especially
// USB_PORT_STAT_C_CONNECTION).
let __t2: i32 = wValue as i32;
if __t2 == 1 {
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_72 & 4294967291u32, *__slate_slot_74);
} else {
if __t2 == 17 {
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_72 | 8u32, *__slate_slot_74);
} else {
if __t2 == 2 {
if *__slate_slot_72 & 256u32 != 0 {
break '__join_1;
} else {
if ((*ehci).__slate_bits_1.__get_no_selective_suspend() as i32) == 0 {
if *__slate_slot_72 & 128u32 != 0 {
if *__slate_slot_72 & 4u32 == 0 {
break '__join_1;
} else {
// clear phy low-power mode before resume
if ((*ehci).__slate_bits_2.__get_has_tdi_phy_lpm() as i32) != 0 {
*__slate_slot_71 = ehci_readl(ehci as *const ehci_hcd, *__slate_slot_73);
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_71 & 4290772991u32, *__slate_slot_73);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
msleep(5u32); // wait to leave low-power mode
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
}
// resume signaling for 20 msec
let __v61: u32 = *__slate_slot_72 & 4287627263u32;
*__slate_slot_72 = __v61;
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_72 | 64u32, *__slate_slot_74);
*((*ehci).reset_done.as_mut_ptr() as *mut u64).wrapping_offset((wIndex as i32) as isize) = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(40u32));
set_bit(wIndex as i64, core::ptr::addr_of_mut!((*ehci).resuming_ports) as *mut u64);
usb_hcd_start_port_resume(core::ptr::addr_of_mut!((*hcd).self_), wIndex as i32);
}
}
}
}
} else {
if __t2 == 18 {
clear_bit(wIndex as i64, core::ptr::addr_of_mut!((*ehci).port_c_suspend) as *mut u64);
} else {
if __t2 == 8 {
if HCS_PPC!(unsafe { (*ehci).hcs_params }) != 0 {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
ehci_port_power(ehci, wIndex as i32, false);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
}
} else {
if __t2 == 16 {
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_72 | 2u32, *__slate_slot_74);
} else {
if __t2 == 19 {
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_72 | 32u32, *__slate_slot_74);
} else {
if __t2 != 20 {
break '__join_1;
}
}
}
}
}
}
}
}
ehci_readl(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*(*ehci).regs).command)); // unblock posted write
break '__join_0;
}
} else {
if __t4 == 40966 {
ehci_hub_descriptor(ehci, buf as *mut usb_hub_descriptor);
break '__join_0;
} else {
if __t4 == 40960 {
memset(buf as *mut (), 0i32, 4u64); // no hub-wide feature/status flags
// cpu_to_le32s ((u32 *) buf);
break '__join_0;
} else {
if __t4 == 41728 {
if !(wIndex == 0 || (wIndex as i32) > ports) {
wIndex = (wIndex as i32).wrapping_sub(1) as u16;
*__slate_slot_70 = 0;
*__slate_slot_72 = ehci_readl(ehci as *const ehci_hcd, *__slate_slot_74);
// wPortChange bits
if *__slate_slot_72 & 2u32 != 0 {
*__slate_slot_70 |= 65536u32;
}
if *__slate_slot_72 & 8u32 != 0 {
*__slate_slot_70 |= 131072u32;
}
if ehci_has_ci_pec_bug!(ehci, *__slate_slot_72) {
*__slate_slot_70 |= 131072u32;
while false {
}
_dev_info((*ehci_to_hcd(ehci)).self_.controller as *const device, c"PE is cleared by HW port:%d PORTSC:%08x\n".as_ptr() as *const u8, (wIndex as i32).wrapping_add(1), *__slate_slot_72);
}
if *__slate_slot_72 & 32u32 != 0 && (!ignore_oc && ((*ehci).__slate_bits_1.__get_spurious_oc() as i32) == 0) {
*__slate_slot_70 |= 524288u32;
// Hubs should disable port power on over-current.
// However, not all EHCI implementations do this
// automatically, even if they _do_ support per-port
// power switching; they're allowed to just limit the
// current.  hub_wq will turn the power back on.
if (*__slate_slot_72 & 16u32 != 0 || ((*ehci).__slate_bits_1.__get_need_oc_pp_cycle() as i32) != 0) && HCS_PPC!(unsafe { (*ehci).hcs_params }) != 0 {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
ehci_port_power(ehci, wIndex as i32, false);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
*__slate_slot_72 = ehci_readl(ehci as *const ehci_hcd, *__slate_slot_74);
}
}
// no reset or resume pending
if *((*ehci).reset_done.as_mut_ptr() as *mut u64).wrapping_offset((wIndex as i32) as isize) == 0 {
// Remote Wakeup received?
if *__slate_slot_72 & 64u32 != 0 {
// resume signaling for 20 msec
*((*ehci).reset_done.as_mut_ptr() as *mut u64).wrapping_offset((wIndex as i32) as isize) = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(20u32));
usb_hcd_start_port_resume(core::ptr::addr_of_mut!((*hcd).self_), wIndex as i32);
set_bit(wIndex as i64, core::ptr::addr_of_mut!((*ehci).resuming_ports) as *mut u64);
// check the port again
mod_timer(core::ptr::addr_of_mut!((*ehci_to_hcd(ehci)).rh_timer), *((*ehci).reset_done.as_mut_ptr() as *mut u64).wrapping_offset((wIndex as i32) as isize));
}
// reset or resume not yet complete
} else {
if !time_after_eq!(core::ptr::read_volatile(core::ptr::addr_of!(jiffies)), unsafe { *((*ehci).reset_done.as_mut_ptr() as *mut u64).wrapping_offset((wIndex as i32) as isize) }) {
// wait until it is complete
// resume completed
} else {
if test_bit__5!(wIndex, core::ptr::addr_of_mut!((*ehci).resuming_ports)) != 0 {
clear_bit(wIndex as i64, core::ptr::addr_of_mut!((*ehci).suspended_ports) as *mut u64);
set_bit(wIndex as i64, core::ptr::addr_of_mut!((*ehci).port_c_suspend) as *mut u64);
*((*ehci).reset_done.as_mut_ptr() as *mut u64).wrapping_offset((wIndex as i32) as isize) = 0;
usb_hcd_end_port_resume(core::ptr::addr_of_mut!((*hcd).self_), wIndex as i32);
// stop resume signaling
let __v47: u32 = *__slate_slot_72 & 4294967061u32;
*__slate_slot_72 = __v47;
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_72, *__slate_slot_74);
clear_bit(wIndex as i64, core::ptr::addr_of_mut!((*ehci).resuming_ports) as *mut u64);
retval = ehci_handshake(ehci, *__slate_slot_74 as *mut (), 64u32, 0u32, 2000i32); // 2msec
if retval != 0 {
while false {
}
_dev_err((*ehci_to_hcd(ehci)).self_.controller as *const device, c"port %d resume error %d\n".as_ptr() as *const u8, (wIndex as i32).wrapping_add(1), retval);
break '__join_1;
} else {
*__slate_slot_72 = ehci_readl(ehci as *const ehci_hcd, *__slate_slot_74);
// whoever resets must GetPortStatus to complete it!!
}
} else {
*__slate_slot_70 |= 1048576u32;
*((*ehci).reset_done.as_mut_ptr() as *mut u64).wrapping_offset((wIndex as i32) as isize) = 0;
// force reset to complete
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_72 & 4294966997u32, *__slate_slot_74);
// REVISIT:  some hardware needs 550+ usec to clear
// this bit; seems too long to spin routinely...
retval = ehci_handshake(ehci, *__slate_slot_74 as *mut (), 256u32, 0u32, 1000i32);
if retval != 0 {
while false {
}
_dev_err((*ehci_to_hcd(ehci)).self_.controller as *const device, c"port %d reset error %d\n".as_ptr() as *const u8, (wIndex as i32).wrapping_add(1), retval);
break '__join_1;
} else {
// see what we found out
*__slate_slot_72 = check_reset_complete(ehci, wIndex as i32, *__slate_slot_74, ehci_readl(ehci as *const ehci_hcd, *__slate_slot_74) as i32) as u32;
}
}
}
}
if *__slate_slot_72 & 1u32 != 0 {
__v46 = test_bit__5!(wIndex, core::ptr::addr_of_mut!((*ehci).companion_ports)) != 0;
} else {
__v46 = false;
}
if __v46 {
let __v44: u32 = *__slate_slot_72 & 4294967253u32;
*__slate_slot_72 = __v44;
*__slate_slot_72 |= 8192u32;
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_72, *__slate_slot_74);
*__slate_slot_72 = ehci_readl(ehci as *const ehci_hcd, *__slate_slot_74);
}
// Even if OWNER is set, there's no harm letting hub_wq
// see the wPortStatus values (they should all be 0 except
// for PORT_POWER anyway).
if *__slate_slot_72 & 1u32 != 0 {
*__slate_slot_70 |= 1;
// status may be from integrated TT
if ((*ehci).__slate_bits_2.__get_has_hostpc() as i32) != 0 {
*__slate_slot_71 = ehci_readl(ehci as *const ehci_hcd, *__slate_slot_73);
let __v37: u32 = *__slate_slot_70;
let __v36: u32 = __v37 | (ehci_port_speed!() as u32);
*__slate_slot_70 = __v36;
} else {
let __v39: u32 = *__slate_slot_70;
let __v38: u32 = __v39 | (ehci_port_speed!() as u32);
*__slate_slot_70 = __v38;
}
}
if *__slate_slot_72 & 4u32 != 0 {
*__slate_slot_70 |= 2;
}
// maybe the port was unsuspended without our knowledge
if *__slate_slot_72 & 192u32 != 0 {
*__slate_slot_70 |= 4;
} else {
if test_bit__5!(wIndex, core::ptr::addr_of_mut!((*ehci).suspended_ports)) != 0 {
clear_bit(wIndex as i64, core::ptr::addr_of_mut!((*ehci).suspended_ports) as *mut u64);
clear_bit(wIndex as i64, core::ptr::addr_of_mut!((*ehci).resuming_ports) as *mut u64);
*((*ehci).reset_done.as_mut_ptr() as *mut u64).wrapping_offset((wIndex as i32) as isize) = 0;
if *__slate_slot_72 & 4u32 != 0 {
set_bit(wIndex as i64, core::ptr::addr_of_mut!((*ehci).port_c_suspend) as *mut u64);
}
usb_hcd_end_port_resume(core::ptr::addr_of_mut!((*hcd).self_), wIndex as i32);
}
}
if *__slate_slot_72 & 16u32 != 0 {
*__slate_slot_70 |= 8;
}
if *__slate_slot_72 & 256u32 != 0 {
*__slate_slot_70 |= 16;
}
if *__slate_slot_72 & 4096u32 != 0 {
*__slate_slot_70 |= 256;
}
if test_bit__5!(wIndex, core::ptr::addr_of_mut!((*ehci).port_c_suspend)) != 0 {
*__slate_slot_70 |= 262144u32;
}
if *__slate_slot_70 & 4294901760u32 != 0 {
dbg_port(ehci, c"GetStatus".as_ptr() as *const u8, (wIndex as i32).wrapping_add(1), *__slate_slot_72);
}
// only if wPortChange is interesting
put_unaligned_le32(*__slate_slot_70, buf as *mut ());
break '__join_0;
}
} else {
if __t4 == 8195 {
let __t1: i32 = wValue as i32;
if __t1 == 0 {
break '__join_0;
} else {
if __t1 == 1 {
break '__join_0;
}
}
} else {
if __t4 == 8963 {
*__slate_slot_67 = ((wIndex as i32) >> 8i32) as u32;
wIndex = ((wIndex as i32) & 255) as u16;
if unlikely__5!(unsafe { (*ehci).debug }) != 0 {
if ((wIndex as i32) as u32) == HCS_DEBUG_PORT!(unsafe { (*ehci).hcs_params }) {
__v21 = readl(core::ptr::addr_of_mut!((*(*ehci).debug).control) as *const ()) & 268435456u32 != 0;
} else {
__v21 = false;
}
if __v21 {
retval = 19i32.wrapping_neg();
break '__join_0;
}
}
if !(wIndex == 0 || (wIndex as i32) > ports) {
wIndex = (wIndex as i32).wrapping_sub(1) as u16;
*__slate_slot_72 = ehci_readl(ehci as *const ehci_hcd, *__slate_slot_74);
if *__slate_slot_72 & 8192u32 != 0 {
break '__join_0;
} else {
let __v17: u32 = *__slate_slot_72 & 4294967253u32;
*__slate_slot_72 = __v17;
let __t0: i32 = wValue as i32;
if __t0 == 2 {
if ((*ehci).__slate_bits_1.__get_no_selective_suspend() as i32) == 0 {
if *__slate_slot_72 & 4u32 == 0 || *__slate_slot_72 & 256u32 != 0 {
break '__join_1;
} else {
// After above check the port must be connected.
// Set appropriate bit thus could put phy into low power
// mode if we have tdi_phy_lpm feature
let __v15: u32 = *__slate_slot_72 & 4293918719u32;
*__slate_slot_72 = __v15;
*__slate_slot_72 |= 6291456u32;
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_72 | 128u32, *__slate_slot_74);
if ((*ehci).__slate_bits_2.__get_has_tdi_phy_lpm() as i32) != 0 {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
msleep(5u32); // 5ms for HCD enter low pwr mode
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
*__slate_slot_71 = ehci_readl(ehci as *const ehci_hcd, *__slate_slot_73);
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_71 | 4194304u32, *__slate_slot_73);
*__slate_slot_71 = ehci_readl(ehci as *const ehci_hcd, *__slate_slot_73);
}
if ehci_has_fsl_susp_errata!(ehci) != 0 {
// 10ms for HCD enter suspend
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
usleep_range(10000u64, 20000u64);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
}
set_bit(wIndex as i64, core::ptr::addr_of_mut!((*ehci).suspended_ports) as *mut u64);
}
}
} else {
if __t0 == 8 {
if HCS_PPC!(unsafe { (*ehci).hcs_params }) != 0 {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
ehci_port_power(ehci, wIndex as i32, true);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
}
} else {
if __t0 == 4 {
if *__slate_slot_72 & 192u32 != 0 {
break '__join_1;
} else {
// line status bits may report this as low speed,
// which can be fine if this root hub has a
// transaction translator built in.
if *__slate_slot_72 & 5u32 == 1u32 && ehci_is_TDI!() == 0 && PORT_USB11!(*__slate_slot_72) {
*__slate_slot_72 |= 8192u32;
} else {
*__slate_slot_72 |= 256u32;
let __v9: u32 = *__slate_slot_72 & 4294967291u32;
*__slate_slot_72 = __v9;
// caller must wait, then call GetPortStatus
// usb 2.0 spec says 50 ms resets on root
*((*ehci).reset_done.as_mut_ptr() as *mut u64).wrapping_offset((wIndex as i32) as isize) = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(50u32));
// Force full-speed connect for FSL high-speed
// erratum; disable HS Chirp by setting PFSC bit
}
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_72, *__slate_slot_74);
}
} else {
if __t0 == 21 {
if *__slate_slot_67 == 0 || *__slate_slot_67 > 5 {
break '__join_1;
} else {
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
ehci_quiesce(ehci);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
// Put all enabled ports into suspend
loop {
let __v3: i32 = ports;
ports = __v3.wrapping_sub(1);
if __v3 != 0 {
sreg = ((*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32).wrapping_offset(ports as isize);
*__slate_slot_72 = ehci_readl(ehci as *const ehci_hcd, sreg) & 4294967253u32;
if *__slate_slot_72 & 4u32 != 0 {
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_72 | 128u32, sreg);
}
} else {
break;
}
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
ehci_halt(ehci);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
*__slate_slot_72 = ehci_readl(ehci as *const ehci_hcd, *__slate_slot_74);
*__slate_slot_72 |= *__slate_slot_67 << 16i32;
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_72, *__slate_slot_74);
}
} else {
break '__join_1;
}
}
}
}
ehci_readl(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*(*ehci).regs).command)); // unblock posted writes
break '__join_0;
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
retval = 32i32.wrapping_neg(); // "stall" on error
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_69);
return retval;
}
// no hub-wide feature/status flags
// GetPortStatus clears reset
// no hub-wide feature/status flags
// For downstream facing ports (these):  one hub port is put
// into test mode according to USB2 11.24.2.13, then the hub
// must be reset (which for root hub now means rmmod+modprobe,
// or else system reboot).  See EHCI 2.3.9 and 4.14 for info
// about the EHCI-specific stuff.
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ehci_hub_control_652: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd, u16, u16, u16, *mut u8, u16) -> i32>, *mut ()>(Some(ehci_hub_control)) };

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_relinquish_port")]
extern "C" fn ehci_relinquish_port(hcd: *mut usb_hcd, mut portnum: i32) {
let ehci: *mut ehci_hcd = hcd_to_ehci(hcd);
if ehci_is_TDI!() != 0 {
return;
}
let __v1: i32 = portnum.wrapping_sub(1);
portnum = __v1;
set_owner(ehci, __v1, 8192i32);
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_port_handed_over")]
extern "C" fn ehci_port_handed_over(hcd: *mut usb_hcd, portnum: i32) -> i32 {
let ehci: *mut ehci_hcd = hcd_to_ehci(hcd);
if ehci_is_TDI!() != 0 {
return 0;
}
let reg: *mut u32 = unsafe { (*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32 }.wrapping_offset(portnum.wrapping_sub(1) as isize);
(ehci_readl(ehci as *const ehci_hcd, reg) & 8192u32) as i32
}

fn ehci_port_power(ehci: *mut ehci_hcd, portnum: i32, enable: bool) -> i32 {
let hcd: *mut usb_hcd = ehci_to_hcd(ehci);
let status_reg: *mut u32 = unsafe { (*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32 }.wrapping_offset(portnum as isize);
let temp: u32 = ehci_readl(ehci as *const ehci_hcd, status_reg) & 4294967253u32;
if enable {
ehci_writel(ehci as *const ehci_hcd, temp | 4096u32, status_reg);
} else {
ehci_writel(ehci as *const ehci_hcd, temp & 4294963199u32, status_reg);
}
if (unsafe { (*(*hcd).driver).port_power }) != None {
unsafe { unsafe { (*(*hcd).driver).port_power }.unwrap()(hcd, portnum, enable) };
}
0
}

fn ehci_qtd_init(ehci: *mut ehci_hcd, qtd: *mut ehci_qtd, dma: u64) {
unsafe { memset(qtd as *mut (), 0i32, core::mem::size_of::<ehci_qtd>() as u64) };
unsafe {
(*qtd).qtd_dma = dma;
}
unsafe {
(*qtd).hw_token = cpu_to_hc32(ehci as *const ehci_hcd, 64u32);
}
unsafe {
(*qtd).hw_next = EHCI_LIST_END!(ehci);
}
unsafe {
(*qtd).hw_alt_next = EHCI_LIST_END!(ehci);
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*qtd).qtd_list) });
}

fn ehci_qtd_alloc(ehci: *mut ehci_hcd, flags: u32) -> *mut ehci_qtd {
let mut dma: u64 = 0;
let qtd: *mut ehci_qtd = (unsafe { dma_pool_alloc((*ehci).qtd_pool, flags, core::ptr::addr_of_mut!(dma)) }) as *mut ehci_qtd;
if !qtd.is_null() {
ehci_qtd_init(ehci, qtd, dma);
}
qtd
}

fn ehci_qtd_free(ehci: *mut ehci_hcd, qtd: *mut ehci_qtd) {
unsafe { dma_pool_free((*ehci).qtd_pool, qtd as *mut (), (*qtd).qtd_dma) };
}

fn qh_destroy(ehci: *mut ehci_hcd, qh: *mut ehci_qh) {
// clean qtds first, and know this is not linked
if list_empty((unsafe { core::ptr::addr_of_mut!((*qh).qtd_list) }) as *const list_head) == 0 || !unsafe { (*qh).qh_next.ptr }.is_null() {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.controller as *const device, c"unused qh not empty!\n".as_ptr() as *const u8) };
}
loop {
unsafe {
core::arch::asm!("653: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 653b - ., 3; .popsection\n# {0}", const 653 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_qh_destroy_0, sym __slate_asm_string_qh_destroy_1, const 59 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if !unsafe { (*qh).dummy }.is_null() {
ehci_qtd_free(ehci, unsafe { (*qh).dummy });
}
unsafe { dma_pool_free((*ehci).qh_pool, (*qh).hw as *mut (), (*qh).qh_dma) };
unsafe { kfree(qh as *const ()) };
}

fn ehci_qh_alloc(ehci: *mut ehci_hcd, flags: u32) -> *mut ehci_qh {
let mut __slate_compound_8: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_10: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut dma: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u64 = core::ptr::addr_of_mut!(dma) as *mut u64;
let mut qh: core::mem::MaybeUninit<*mut ehci_qh> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut ehci_qh = core::ptr::addr_of_mut!(qh) as *mut *mut ehci_qh;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
__obj_size = size_mul(core::mem::size_of::<ehci_qh>() as u64, 1u64);
let __slate_auto_bytes_3 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 2080u32, *{
__slate_compound_8 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_8)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 2080u32, *{
__slate_compound_10 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_10)
});
}
'__join_2: {
*__slate_slot_7 = *__slate_slot_1 as *mut ehci_qh;
if !(*__slate_slot_7).is_null() {
(*(*__slate_slot_7)).hw = dma_pool_zalloc((*ehci).qh_pool, flags, core::ptr::addr_of_mut!(*__slate_slot_6)) as *mut ehci_qh_hw;
if !unsafe { (*(*__slate_slot_7)).hw }.is_null() {
(*(*__slate_slot_7)).qh_dma = *__slate_slot_6;
// INIT_LIST_HEAD (&qh->qh_list);
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_7)).qtd_list));
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_7)).unlink_node));
// dummy td enables safe urb queuing
(*(*__slate_slot_7)).dummy = ehci_qtd_alloc(ehci, flags);
if unsafe { (*(*__slate_slot_7)).dummy }.is_null() {
dma_pool_free((*ehci).qh_pool, (*(*__slate_slot_7)).hw as *mut (), (*(*__slate_slot_7)).qh_dma);
} else {
break '__join_2;
}
}
kfree(*__slate_slot_7 as *const ());
return core::ptr::null_mut::<ehci_qh>();
}
}
return *__slate_slot_7;
}
}

fn ehci_mem_cleanup(ehci: *mut ehci_hcd) {
if !unsafe { (*ehci).r#async }.is_null() {
qh_destroy(ehci, unsafe { (*ehci).r#async });
}
unsafe {
(*ehci).r#async = core::ptr::null_mut::<ehci_qh>();
}
if !unsafe { (*ehci).dummy }.is_null() {
qh_destroy(ehci, unsafe { (*ehci).dummy });
}
unsafe {
(*ehci).dummy = core::ptr::null_mut::<ehci_qh>();
}
// DMA consistent memory and pools
unsafe { dma_pool_destroy((*ehci).qtd_pool) };
unsafe {
(*ehci).qtd_pool = core::ptr::null_mut::<dma_pool>();
}
unsafe { dma_pool_destroy((*ehci).qh_pool) };
unsafe {
(*ehci).qh_pool = core::ptr::null_mut::<dma_pool>();
}
unsafe { dma_pool_destroy((*ehci).itd_pool) };
unsafe {
(*ehci).itd_pool = core::ptr::null_mut::<dma_pool>();
}
unsafe { dma_pool_destroy((*ehci).sitd_pool) };
unsafe {
(*ehci).sitd_pool = core::ptr::null_mut::<dma_pool>();
}
if !unsafe { (*ehci).periodic }.is_null() {
dma_free_coherent(unsafe { (*ehci_to_hcd(ehci)).self_.sysdev }, ((unsafe { (*ehci).periodic_size }) as u64).wrapping_mul(core::mem::size_of::<u32>() as u64), (unsafe { (*ehci).periodic }) as *mut (), unsafe { (*ehci).periodic_dma });
}
unsafe {
(*ehci).periodic = core::ptr::null_mut::<u32>();
}
// shadow periodic table
unsafe { kfree((*ehci).pshadow as *const ()) };
unsafe {
(*ehci).pshadow = core::ptr::null_mut::<ehci_shadow>();
}
}

fn ehci_mem_init(ehci: *mut ehci_hcd, flags: u32) -> i32 {
let mut __slate_compound_10: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut hw: core::mem::MaybeUninit<*mut ehci_qh_hw> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut ehci_qh_hw = core::ptr::addr_of_mut!(hw) as *mut *mut ehci_qh_hw;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
unsafe {
'__join_0: {
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
// QTDs for control/bulk/intr transfers
(*ehci).qtd_pool = dma_pool_create(c"ehci_qtd".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.sysdev, core::mem::size_of::<ehci_qtd>() as u64, 32u64, 4096u64);
// byte alignment (for hw parts)
// can't cross 4K
if !unsafe { (*ehci).qtd_pool }.is_null() {
// QHs for control/bulk/intr transfers
(*ehci).qh_pool = dma_pool_create(c"ehci_qh".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.sysdev, core::mem::size_of::<ehci_qh_hw>() as u64, 32u64, 4096u64);
// byte alignment (for hw parts)
// can't cross 4K
if !unsafe { (*ehci).qh_pool }.is_null() {
(*ehci).r#async = ehci_qh_alloc(ehci, flags);
if !unsafe { (*ehci).r#async }.is_null() {
// ITD for high speed ISO transfers
(*ehci).itd_pool = dma_pool_create(c"ehci_itd".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.sysdev, core::mem::size_of::<ehci_itd>() as u64, 32u64, 4096u64);
// byte alignment (for hw parts)
// can't cross 4K
if !unsafe { (*ehci).itd_pool }.is_null() {
// SITD for full/low speed split ISO transfers
(*ehci).sitd_pool = dma_pool_create(c"ehci_sitd".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.sysdev, core::mem::size_of::<ehci_sitd>() as u64, 32u64, 4096u64);
// byte alignment (for hw parts)
// can't cross 4K
if !unsafe { (*ehci).sitd_pool }.is_null() {
// Hardware periodic table
(*ehci).periodic = dma_alloc_coherent((*ehci_to_hcd(ehci)).self_.sysdev, ((*ehci).periodic_size as u64).wrapping_mul(core::mem::size_of::<u32>() as u64), core::ptr::addr_of_mut!((*ehci).periodic_dma), flags) as *mut u32;
if !unsafe { (*ehci).periodic }.is_null() {
if ((*ehci).__slate_bits_1.__get_use_dummy_qh() as i32) != 0 {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
(*ehci).dummy = ehci_qh_alloc(ehci, flags);
if unsafe { (*ehci).dummy }.is_null() {
break '__join_0;
} else {
*__slate_slot_6 = (*(*ehci).dummy).hw;
(*(*__slate_slot_6)).hw_next = EHCI_LIST_END!(ehci);
(*(*__slate_slot_6)).hw_qtd_next = EHCI_LIST_END!(ehci);
(*(*__slate_slot_6)).hw_alt_next = EHCI_LIST_END!(ehci);
(*(*ehci).dummy).hw = *__slate_slot_6;
*__slate_slot_9 = 0;
while (*__slate_slot_9 as u32) < (*ehci).periodic_size {
*unsafe { (*ehci).periodic }.wrapping_offset(*__slate_slot_9 as isize) = cpu_to_hc32(ehci as *const ehci_hcd, (*(*ehci).dummy).qh_dma as u32);
let __v5: i32 = *__slate_slot_9;
*__slate_slot_9 = __v5.wrapping_add(1);
}
}
} else {
*__slate_slot_9 = 0;
while (*__slate_slot_9 as u32) < (*ehci).periodic_size {
*unsafe { (*ehci).periodic }.wrapping_offset(*__slate_slot_9 as isize) = EHCI_LIST_END!(ehci);
let __v8: i32 = *__slate_slot_9;
*__slate_slot_9 = __v8.wrapping_add(1);
}
}
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kmalloc_array_noprof((*ehci).periodic_size as u64, core::mem::size_of::<*mut ()>() as u64, flags | 256u32, *{
__slate_compound_10 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_10)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kmalloc_array_noprof((*ehci).periodic_size as u64, core::mem::size_of::<*mut ()>() as u64, flags | 256u32, *{
__slate_compound_12 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_12)
});
}
let __v2: *mut () = *__slate_slot_1;
// software shadow of hardware table
(*ehci).pshadow = __v2 as *mut ehci_shadow;
if !unsafe { (*ehci).pshadow }.is_null() {
return 0;
}
}
}
}
}
}
}
}
ehci_mem_cleanup(ehci);
return 12i32.wrapping_neg();
}
}

fn qtd_fill(ehci: *mut ehci_hcd, qtd: *mut ehci_qtd, mut buf: u64, len: u64, token: i32, maxpacket: i32) -> u32 {
let mut count: u32 = 0;
let mut addr: u64 = buf;
let mut i: i32 = 0;
// one buffer entry per 4K ... first might be short or unaligned
unsafe {
*((*qtd).hw_buf.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) = cpu_to_hc32(ehci as *const ehci_hcd, addr as u32);
}
unsafe {
*((*qtd).hw_buf_hi.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) = cpu_to_hc32(ehci as *const ehci_hcd, (addr >> 32i32) as u32);
}
count = 4096u64.wrapping_sub(buf & 4095) as u32; // rest of that page
if likely!(len < (count as u64)) != 0 {
count = len as u32;
} else {
buf = buf.wrapping_add(4096);
let __v3: u64 = buf & 18446744073709547520u64;
buf = __v3;
// per-qtd limit: from 16K to 20K (best alignment)
i = 1;
while (count as u64) < len && i < 5 {
addr = buf;
unsafe {
*((*qtd).hw_buf.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) = cpu_to_hc32(ehci as *const ehci_hcd, addr as u32);
}
unsafe {
*((*qtd).hw_buf_hi.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) = cpu_to_hc32(ehci as *const ehci_hcd, (addr >> 32i32) as u32);
}
buf = buf.wrapping_add(4096);
if (count.wrapping_add(4096) as u64) < len {
count = count.wrapping_add(4096);
} else {
count = len as u32;
}
i = i.wrapping_add(1);
}
// short packets may only terminate transfers
if (count as u64) != len {
count = count.wrapping_sub(count % (maxpacket as u32));
}
}
// ... iff needed
unsafe {
(*qtd).hw_token = cpu_to_hc32(ehci as *const ehci_hcd, count << 16i32 | (token as u32));
}
unsafe {
(*qtd).length = count as u64;
}
count
}

fn qh_update(ehci: *mut ehci_hcd, qh: *mut ehci_qh, qtd: *mut ehci_qtd) {
let hw: *mut ehci_qh_hw = unsafe { (*qh).hw };
// writes to an active overlay are unsafe
let mut __v0: i64;
let __ret_warn_on: i32 = (((unsafe { (*qh).qh_state }) as i32) != 3) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("654: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 654b - ., 3; .popsection\n# {0}", const 654 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_qh_update_1, sym __slate_asm_string_qh_update_2, const 81 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("655: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 655b - ., 4; .popsection\n# {0}", const 655 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe {
(*hw).hw_qtd_next = QTD_NEXT!(ehci, unsafe { (*qtd).qtd_dma });
}
unsafe {
(*hw).hw_alt_next = EHCI_LIST_END!(ehci);
}
// Except for control endpoints, we make hardware maintain data
// toggle (like OHCI) ... here (re)initialize the toggle in the QH,
// and set the pseudo-toggle in udev. Only usb_clear_halt() will
// ever clear it.
if (unsafe { (*hw).hw_info1 }) & cpu_to_hc32(ehci as *const ehci_hcd, 16384u32) == 0 {
let is_out: u32 = ((unsafe { (*qh).__slate_bits_0.__get_is_out() }) as i32) as u32;
let epnum: u32 = hc32_to_cpup(ehci as *const ehci_hcd, (unsafe { core::ptr::addr_of_mut!((*hw).hw_info1) }) as *const u32) >> 8i32 & 15;
if unlikely!(usb_gettoggle!(unsafe { (*qh).ps.udev }, epnum, is_out) == 0) != 0 {
let __v5: u32 = !cpu_to_hc32(ehci as *const ehci_hcd, 2147483648u32);
unsafe {
(*hw).hw_token = (*hw).hw_token & __v5;
}
usb_settoggle!(unsafe { (*qh).ps.udev }, epnum, is_out, 1i32);
}
}
let __v9: u32 = cpu_to_hc32(ehci as *const ehci_hcd, 2147483649u32);
unsafe {
(*hw).hw_token = (*hw).hw_token & __v9;
}
}

fn qh_refresh(ehci: *mut ehci_hcd, qh: *mut ehci_qh) {
let qtd: *mut ehci_qtd = list_entry__4!(unsafe { (*qh).qtd_list.next }, ehci_qtd);
// first qtd may already be partially processed.
// If we come here during unlink, the QH overlay region
// might have reference to the just unlinked qtd. The
// qtd is updated in qh_completions(). Update the QH
// overlay here.
if (unsafe { (*(*qh).hw).hw_token }) & ACTIVE_BIT!(ehci) != 0 {
unsafe {
(*(*qh).hw).hw_qtd_next = (*qtd).hw_next;
}
if ((unsafe { (*qh).__slate_bits_0.__get_should_be_inactive() }) as i32) != 0 {
while false {
}
unsafe { _dev_warn((*ehci_to_hcd(ehci)).self_.controller as *const device, c"qh %p should be inactive!\n".as_ptr() as *const u8, qh) };
}
} else {
qh_update(ehci, qh, qtd);
}
unsafe {
(*qh).__slate_bits_0.__set_should_be_inactive(0u32);
}
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_clear_tt_buffer_complete")]
extern "C" fn ehci_clear_tt_buffer_complete(hcd: *mut usb_hcd, ep: *mut usb_host_endpoint) {
let ehci: *mut ehci_hcd = hcd_to_ehci(hcd);
let qh: *mut ehci_qh = (unsafe { (*ep).hcpriv }) as *mut ehci_qh;
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*ehci).lock) }, flags);
unsafe {
(*qh).__slate_bits_0.__set_clearing_tt(0u32);
}
let __v5: bool = ((unsafe { (*qh).qh_state }) as i32) == 3 && list_empty((unsafe { core::ptr::addr_of_mut!((*qh).qtd_list) }) as *const list_head) == 0;
if __v5 && ((unsafe { (*ehci).rh_state }) as u32) == 2 {
qh_link_async(ehci, qh);
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*ehci).lock) }, flags);
}

fn ehci_clear_tt_buffer(ehci: *mut ehci_hcd, qh: *mut ehci_qh, urb: *mut urb, token: u32) {
// If an async split transaction gets an error or is unlinked,
// the TT buffer may be left in an indeterminate state.  We
// have to clear the TT buffer.
//
// Note: this routine is never called for Isochronous transfers.
if !unsafe { (*(*urb).dev).tt }.is_null() && !usb_pipeint!(unsafe { (*urb).pipe }) && ((unsafe { (*qh).__slate_bits_0.__get_clearing_tt() }) as i32) == 0 {
let __v0: bool = ehci_is_TDI!() == 0 || (unsafe { (*(*(*urb).dev).tt).hub }) != unsafe { (*ehci_to_hcd(ehci)).self_.root_hub };
if __v0 {
if (unsafe { usb_hub_clear_tt_buffer(urb) }) == 0 {
unsafe {
(*qh).__slate_bits_0.__set_clearing_tt(1u32);
}
}
}
// REVISIT ARC-derived cores don't clear the root
// hub TT buffer in this way...
}
}

fn qtd_copy_status(ehci: *mut ehci_hcd, urb: *mut urb, length: u64, token: u32) -> i32 {
let mut status: i32 = 115i32.wrapping_neg();
// count IN/OUT bytes, not SETUP (even short packets)
if likely!(QTD_PID!(token) != 2) != 0 {
let __v1: u32 = unsafe { (*urb).actual_length };
let __v2: u32 = (__v1 as u64).wrapping_add(length.wrapping_sub(QTD_LENGTH!(token) as u64)) as u32;
unsafe {
(*urb).actual_length = __v2;
}
}
// don't modify error codes
if unlikely__2!(unsafe { (*urb).unlinked }) != 0 {
return status;
}
// force cleanup after short read; not always an error
if unlikely!(IS_SHORT_READ!(token)) != 0 {
status = 121i32.wrapping_neg();
}
// serious "can't proceed" faults reported by the hardware
if token & 64u32 != 0 {
if token & 16u32 != 0 {
// FIXME "must" disable babbling device's port too
status = 75i32.wrapping_neg();
// When MMF is active and PID Code is IN, queue is halted.
// EHCI Specification, Table 4-13.
} else {
if token & 4u32 != 0 && QTD_PID!(token) == 1 {
status = 71i32.wrapping_neg();
// CERR nonzero + halt --> stall
} else {
if QTD_CERR!(token) != 0 {
status = 32i32.wrapping_neg();
// In theory, more than one of the following bits can be set
// since they are sticky and the transaction is retried.
// Which to test first is rather arbitrary.
} else {
if token & 4u32 != 0 {
// fs/ls interrupt xfer missed the complete-split
status = 71i32.wrapping_neg();
} else {
if token & 32u32 != 0 {
status = if QTD_PID!(token) == 1 { 63i32.wrapping_neg() } else { 70i32.wrapping_neg() }; // IN ?
// hc couldn't read data
// hc couldn't write data
} else {
if token & 8u32 != 0 {
// timeout, bad CRC, wrong PID, etc
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.controller as *const device, c"devpath %s ep%d%s 3strikes\n".as_ptr() as *const u8, (*(*urb).dev).devpath.as_mut_ptr() as *mut u8, usb_pipeendpoint!(unsafe { (*urb).pipe }), if usb_pipein!(unsafe { (*urb).pipe }) != 0 { c"in".as_ptr() as *mut u8 } else { c"out".as_ptr() as *mut u8 }) };
}
status = 71i32.wrapping_neg();
} else {
// unknown
status = 71i32.wrapping_neg();
}
}
}
}
}
}
}
status
}

fn ehci_urb_done(ehci: *mut ehci_hcd, urb: *mut urb, mut status: i32) {
if usb_pipetype!(unsafe { (*urb).pipe }) == 1 {
// ... update hc-wide periodic stats
let __v0: *mut usb_hcd = ehci_to_hcd(ehci);
let __v1: i32 = unsafe { (*__v0).self_.bandwidth_int_reqs };
unsafe {
(*__v0).self_.bandwidth_int_reqs = __v1.wrapping_sub(1);
}
}
if unlikely__2!(unsafe { (*urb).unlinked }) != 0 {
INCR!();
} else {
// report non-error and short read status as zero
if status == 115i32.wrapping_neg() || status == 121i32.wrapping_neg() {
status = 0;
}
INCR!();
}
unsafe { usb_hcd_unlink_urb_from_ep(ehci_to_hcd(ehci), urb) };
unsafe { usb_hcd_giveback_urb(ehci_to_hcd(ehci), urb, status) };
}

fn qh_completions(ehci: *mut ehci_hcd, qh: *mut ehci_qh) -> u32 {
// be sure the hardware's done with the qh before refreshing
// it after fault cleanup, or recovering from silicon wrongly
// overlaying the dummy qtd (which reduces DMA chatter).
//
// We won't refresh a QH that's linked (after the HC
// stopped the queue).  That avoids a race:
//  - HC reads first part of QH;
//  - CPU updates that first part and the token;
//  - HC reads rest of that QH, including token
// Result:  HC gets an inconsistent image, and then
// DMAs to/from the wrong memory (corrupting it).
//
// That should be rare for interrupt transfers,
// except maybe high bandwidth ...
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __v5: bool = unsafe { core::mem::zeroed() };
// retry transaction errors until we
// reach the software xacterr limit
let mut __v16: bool = unsafe { core::mem::zeroed() };
let mut __v20: bool = unsafe { core::mem::zeroed() };
// If this was the active qtd when the qh was unlinked
// and the overlay's token is active, then the overlay
// hasn't been written back to the qtd yet so use its
// token instead of the qtd's.  After the qtd is
// processed and removed, the overlay won't be valid
// any more.
let mut __v25: bool = unsafe { core::mem::zeroed() };
let mut token: u32 = unsafe { core::mem::zeroed() };
let mut urb: core::mem::MaybeUninit<*mut urb> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut *mut urb = core::ptr::addr_of_mut!(urb) as *mut *mut urb;
let mut qtd: core::mem::MaybeUninit<*mut ehci_qtd> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut *mut ehci_qtd = core::ptr::addr_of_mut!(qtd) as *mut *mut ehci_qtd;
let mut hw: *mut ehci_qh_hw = unsafe { core::mem::zeroed() };
let mut state: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut u8 = core::ptr::addr_of_mut!(state) as *mut u8;
let mut stopped: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut i32 = core::ptr::addr_of_mut!(stopped) as *mut i32;
let mut last_status: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut i32 = core::ptr::addr_of_mut!(last_status) as *mut i32;
let mut tmp: core::mem::MaybeUninit<*mut list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_37: *mut *mut list_head = core::ptr::addr_of_mut!(tmp) as *mut *mut list_head;
let mut entry: core::mem::MaybeUninit<*mut list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut *mut list_head = core::ptr::addr_of_mut!(entry) as *mut *mut list_head;
let mut end: *mut ehci_qtd = unsafe { core::mem::zeroed() };
let mut last: core::mem::MaybeUninit<*mut ehci_qtd> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut *mut ehci_qtd = core::ptr::addr_of_mut!(last) as *mut *mut ehci_qtd;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
end = (*qh).dummy;
let __slate_auto_bytes_2 = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_37 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 1usize);
hw = (*qh).hw;
// completions (or tasks on other cpus) must never clobber HALT
// till we've gone through and cleaned everything up, even when
// they add urbs to this qh's queue or mark them for unlinking.
//
// NOTE:  unlinking expects to be done in queue order.
//
// It's a bug for qh->qh_state to be anything other than
// QH_STATE_IDLE, unless our caller is scan_async() or
// scan_intr().
*__slate_slot_34 = (*qh).qh_state;
(*qh).qh_state = 5;
*__slate_slot_35 = ((*__slate_slot_34 as i32) == 3) as i32;
'__join_5: {
loop {
*__slate_slot_40 = core::ptr::null_mut::<ehci_qtd>();
*__slate_slot_36 = 115i32.wrapping_neg();
(*qh).__slate_bits_0.__set_dequeue_during_giveback(0u32);
// remove de-activated QTDs from front of queue.
// after faults (including short reads), cleanup this urb
// then let the queue advance.
// if queue is stopped, handles unlinks.
*__slate_slot_38 = (*core::ptr::addr_of_mut!((*qh).qtd_list)).next;
*__slate_slot_37 = (*(*__slate_slot_38)).next;
'__loop_11: while list_is_head(*__slate_slot_38 as *const list_head, core::ptr::addr_of_mut!((*qh).qtd_list) as *const list_head) == 0 {
let __slate_auto_bytes = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
token = 0;
*__slate_slot_31 = list_entry__4!(*__slate_slot_38, ehci_qtd);
*__slate_slot_30 = (*(*__slate_slot_31)).urb;
// clean up any state from previous QTD ...
if !(*__slate_slot_40).is_null() {
if likely!((*(*__slate_slot_40)).urb != *__slate_slot_30) != 0 {
ehci_urb_done(ehci, (*(*__slate_slot_40)).urb, *__slate_slot_36);
*__slate_slot_36 = 115i32.wrapping_neg();
}
ehci_qtd_free(ehci, *__slate_slot_40);
*__slate_slot_40 = core::ptr::null_mut::<ehci_qtd>();
}
// ignore urbs submitted during completions we reported
if *__slate_slot_31 == end {
break;
} else {
// hardware copies qtd out of qh overlay
loop {
kcsan_rmb!();
core::arch::asm!("lfence", options(nostack, att_syntax));
if true {
break;
}
}
token = hc32_to_cpu(ehci as *const ehci_hcd, (*(*__slate_slot_31)).hw_token);
// always clean up qtds the hc de-activated
'__join_12: {
'__join_24: {
'__join_25: {
'__join_26: {
'__join_38: {
while token & 128u32 == 0 {
// Report Data Buffer Error: non-fatal but useful
if token & 32u32 != 0 {
}
// on STALL, error, and short reads this urb must
// complete and all its qtds must be recycled.
if token & 64u32 != 0 {
if token & 8u32 != 0 && QTD_CERR!(token) == 0 {
let __v13: u8 = ((*qh).xacterrs as i32).wrapping_add(1) as u8;
(*qh).xacterrs = __v13;
__v16 = (__v13 as i32) < 32;
} else {
__v16 = false;
}
if __v16 && (*(*__slate_slot_30)).unlinked == 0 {
// reset the token in the qtd and the
// qh overlay (which still contains
// the qtd) so that we pick up from
// where we left off
let __v11: u32 = token & 4294967231u32;
token = __v11;
token |= 3200u32;
(*(*__slate_slot_31)).hw_token = cpu_to_hc32(ehci as *const ehci_hcd, token);
loop {
kcsan_wmb!();
core::arch::asm!("sfence", options(nostack, att_syntax));
if true {
break;
}
}
(*hw).hw_token = cpu_to_hc32(ehci as *const ehci_hcd, token);
} else {
break '__join_26;
}
} else {
break '__join_38;
}
}
if likely!(*__slate_slot_35 == 0 && ((*ehci).rh_state as u32) >= 2) != 0 {
break '__loop_11;
} else {
*__slate_slot_35 = 1;
// cancel everything if we halt, suspend, etc
if ((*ehci).rh_state as u32) < 2 {
*__slate_slot_36 = 108i32.wrapping_neg();
(*qh).unlink_reason = (((*qh).unlink_reason as i32) | 8) as u8;
} else {
if *__slate_slot_36 == 115i32.wrapping_neg() && (*(*__slate_slot_30)).unlinked == 0 {
break '__join_12;
}
}
// this qtd is active; skip it unless a previous qtd
// for its urb faulted, or its urb was canceled.
if (*__slate_slot_34 as i32) == 3 && (*qh).qtd_list.next == core::ptr::addr_of_mut!((*(*__slate_slot_31)).qtd_list) {
__v25 = (*hw).hw_token & ACTIVE_BIT!(ehci) != 0;
} else {
__v25 = false;
}
if __v25 {
token = hc32_to_cpu(ehci as *const ehci_hcd, (*hw).hw_token);
let __v24: u32 = !ACTIVE_BIT!(ehci);
(*hw).hw_token = (*hw).hw_token & __v24;
(*qh).__slate_bits_0.__set_should_be_inactive(1u32);
// An unlink may leave an incomplete
// async transaction in the TT buffer.
// We have to clear it.
ehci_clear_tt_buffer(ehci, qh, *__slate_slot_30, token);
break '__join_24;
} else {
break '__join_24;
}
}
}
if IS_SHORT_READ!(token) {
__v20 = (*(*__slate_slot_31)).hw_alt_next & EHCI_LIST_END!(ehci) == 0;
} else {
__v20 = false;
}
if __v20 {
*__slate_slot_35 = 1;
(*qh).unlink_reason = (((*qh).unlink_reason as i32) | 2) as u8;
break '__join_25;
} else {
break '__join_25;
}
}
*__slate_slot_35 = 1;
(*qh).unlink_reason = (((*qh).unlink_reason as i32) | 1) as u8;
// magic dummy for some short reads; qh won't advance.
// that silicon quirk can kick in with this dummy too.
//
// other short reads won't stop the queue, including
// control transfers (status stage handles that) or
// most other single-qtd reads ... the queue stops if
// URB_SHORT_NOT_OK was set so the driver submitting
// the urbs could clean it up.
}
// stop scanning when we reach qtds the hc is using
}
// unless we already know the urb's status, collect qtd status
// and update count of bytes transferred.  in common short read
// cases with only one data qtd (including control transfers),
// queue processing won't halt.  but with two or more qtds (for
// example, with a 32 KB transfer), when the first qtd gets a
// short read the second must be removed by hand.
if *__slate_slot_36 == 115i32.wrapping_neg() {
*__slate_slot_36 = qtd_copy_status(ehci, *__slate_slot_30, (*(*__slate_slot_31)).length, token);
if *__slate_slot_36 == 121i32.wrapping_neg() {
__v5 = (*(*__slate_slot_31)).hw_alt_next & EHCI_LIST_END!(ehci) != 0;
} else {
__v5 = false;
}
if __v5 {
*__slate_slot_36 = 115i32.wrapping_neg();
}
// As part of low/full-speed endpoint-halt processing
// we must clear the TT buffer (11.17.5).
if unlikely!(*__slate_slot_36 != 115i32.wrapping_neg() && *__slate_slot_36 != 121i32.wrapping_neg()) != 0 {
// The TT's in some hubs malfunction when they
// receive this request following a STALL (they
// stop sending isochronous packets).  Since a
// STALL can't leave the TT buffer in a busy
// state (if you believe Figures 11-48 - 11-51
// in the USB 2.0 spec), we won't clear the TT
// buffer in this case.  Strictly speaking this
// is a violation of the spec.
if *__slate_slot_36 != 32i32.wrapping_neg() {
ehci_clear_tt_buffer(ehci, qh, *__slate_slot_30, token);
}
}
}
// if we're removing something not at the queue head,
// patch the hardware queue pointer.
if *__slate_slot_35 != 0 && (*(*__slate_slot_31)).qtd_list.prev != core::ptr::addr_of_mut!((*qh).qtd_list) {
*__slate_slot_40 = list_entry__4!(unsafe { (*(*__slate_slot_31)).qtd_list.prev }, ehci_qtd);
(*(*__slate_slot_40)).hw_next = (*(*__slate_slot_31)).hw_next;
}
// remove qtd; it's recycled after possible urb completion
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_31)).qtd_list));
*__slate_slot_40 = *__slate_slot_31;
// reinit the xacterr counter for the next qtd
(*qh).xacterrs = 0;
}
*__slate_slot_38 = *__slate_slot_37;
*__slate_slot_37 = (*(*__slate_slot_38)).next;
}
}
// last urb's completion might still need calling
if likely!(!(*__slate_slot_40).is_null()) != 0 {
ehci_urb_done(ehci, (*(*__slate_slot_40)).urb, *__slate_slot_36);
ehci_qtd_free(ehci, *__slate_slot_40);
}
// Do we need to rescan for URBs dequeued during a giveback?
if unlikely__2!((*qh).__slate_bits_0.__get_dequeue_during_giveback() as i32) != 0 {
// If the QH is already unlinked, do the rescan now.
if (*__slate_slot_34 as i32) != 3 {
break;
}
} else {
break '__join_5;
}
}
// Otherwise the caller must unlink the QH.
}
// restore original state; caller must unlink or relink
(*qh).qh_state = *__slate_slot_34;
if *__slate_slot_35 != 0 {
__v3 = true;
} else {
__v3 = (*hw).hw_qtd_next == EHCI_LIST_END!(ehci);
}
if __v3 {
(*qh).unlink_reason = (((*qh).unlink_reason as i32) | 4) as u8;
}
// Let the caller know if the QH needs to be unlinked.
return (*qh).unlink_reason as u32;
}
// scan the whole queue for unlinks whenever it stops
}

fn qtd_list_free(ehci: *mut ehci_hcd, urb: *mut urb, qtd_list: *mut list_head) {
let mut entry: *mut list_head = unsafe { core::mem::zeroed() };
let mut temp: *mut list_head = unsafe { core::mem::zeroed() };
entry = unsafe { (*qtd_list).next };
temp = unsafe { (*entry).next };
while list_is_head(entry as *const list_head, qtd_list as *const list_head) == 0 {
let mut qtd: *mut ehci_qtd = unsafe { core::mem::zeroed() };
qtd = list_entry__4!(entry, ehci_qtd);
list_del(unsafe { core::ptr::addr_of_mut!((*qtd).qtd_list) });
ehci_qtd_free(ehci, qtd);
entry = temp;
temp = unsafe { (*entry).next };
}
}

fn qh_urb_transaction(ehci: *mut ehci_hcd, urb: *mut urb, head: *mut list_head, flags: u32) -> *mut list_head {
let mut one_more: i32 = unsafe { core::mem::zeroed() };
let mut this_qtd_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut u32 = core::ptr::addr_of_mut!(this_qtd_len) as *mut u32;
let mut sg: core::mem::MaybeUninit<*mut scatterlist> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut *mut scatterlist = core::ptr::addr_of_mut!(sg) as *mut *mut scatterlist;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut token: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut u32 = core::ptr::addr_of_mut!(token) as *mut u32;
let mut is_input: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut i32 = core::ptr::addr_of_mut!(is_input) as *mut i32;
let mut maxpacket: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut i32 = core::ptr::addr_of_mut!(maxpacket) as *mut i32;
let mut this_sg_len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut i32 = core::ptr::addr_of_mut!(this_sg_len) as *mut i32;
let mut len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut i32 = core::ptr::addr_of_mut!(len) as *mut i32;
let mut buf: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut u64 = core::ptr::addr_of_mut!(buf) as *mut u64;
let mut qtd_prev: core::mem::MaybeUninit<*mut ehci_qtd> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut *mut ehci_qtd = core::ptr::addr_of_mut!(qtd_prev) as *mut *mut ehci_qtd;
let mut qtd: core::mem::MaybeUninit<*mut ehci_qtd> = core::mem::MaybeUninit::uninit();
let __slate_slot_37: *mut *mut ehci_qtd = core::ptr::addr_of_mut!(qtd) as *mut *mut ehci_qtd;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_37 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_10 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
// URBs map to sequences of QTDs:  one logical transaction
*__slate_slot_37 = ehci_qtd_alloc(ehci, flags);
if unlikely!((*__slate_slot_37).is_null()) != 0 {
return core::ptr::null_mut::<list_head>();
} else {
'__join_0: {
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_37)).qtd_list), head);
(*(*__slate_slot_37)).urb = urb;
*__slate_slot_30 = 128u32;
*__slate_slot_30 |= 3072u32;
// for split transactions, SplitXState initialized to zero
*__slate_slot_34 = (*urb).transfer_buffer_length as i32;
*__slate_slot_31 = usb_pipein!(unsafe { (*urb).pipe }) as i32;
if usb_pipecontrol!(unsafe { (*urb).pipe }) {
// SETUP pid
qtd_fill(ehci, *__slate_slot_37, (*urb).setup_dma, core::mem::size_of::<usb_ctrlrequest>() as u64, (*__slate_slot_30 | 512u32) as i32, 8i32);
// ... and always at least one more pid
*__slate_slot_30 ^= 2147483648u32;
*__slate_slot_36 = *__slate_slot_37;
*__slate_slot_37 = ehci_qtd_alloc(ehci, flags);
if unlikely!((*__slate_slot_37).is_null()) != 0 {
break '__join_0;
} else {
(*(*__slate_slot_37)).urb = urb;
(*(*__slate_slot_36)).hw_next = QTD_NEXT!(ehci, unsafe { (*(*__slate_slot_37)).qtd_dma });
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_37)).qtd_list), head);
// for zero length DATA stages, STATUS is always IN
if *__slate_slot_34 == 0 {
*__slate_slot_30 |= 256u32;
}
}
}
// data transfer stage:  buffer setup
*__slate_slot_29 = (*urb).num_mapped_sgs;
if *__slate_slot_34 > 0 && *__slate_slot_29 > 0 {
*__slate_slot_28 = (*urb).sg;
*__slate_slot_35 = sg_dma_address!(*__slate_slot_28);
// urb->transfer_buffer_length may be smaller than the
// size of the scatterlist (or vice versa)
*__slate_slot_33 = min_t!(i32, sg_dma_len!(*__slate_slot_28), *__slate_slot_34);
} else {
*__slate_slot_28 = core::ptr::null_mut::<scatterlist>();
*__slate_slot_35 = (*urb).transfer_dma;
*__slate_slot_33 = *__slate_slot_34;
}
if *__slate_slot_31 != 0 {
*__slate_slot_30 |= 256u32;
}
// else it's already initted to "out" pid (0 << 8)
*__slate_slot_32 = usb_endpoint_maxp(core::ptr::addr_of_mut!((*(*urb).ep).desc) as *const usb_endpoint_descriptor);
// buffer gets wrapped in one or more qtds;
// last one may be "short" (including zero len)
// and may serve as a control status ack
loop {
let __slate_auto_bytes = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_19 = qtd_fill(ehci, *__slate_slot_37, *__slate_slot_35, *__slate_slot_33 as u64, *__slate_slot_30 as i32, *__slate_slot_32);
*__slate_slot_33 = (*__slate_slot_33 as u32).wrapping_sub(*__slate_slot_19) as i32;
*__slate_slot_34 = (*__slate_slot_34 as u32).wrapping_sub(*__slate_slot_19) as i32;
let __v14: u64 = *__slate_slot_35;
*__slate_slot_35 = __v14.wrapping_add(*__slate_slot_19 as u64);
// short reads advance to a "magic" dummy instead of the next
// qtd ... that forces the queue to stop, for manual cleanup.
// (this will usually be overridden later.)
if *__slate_slot_31 != 0 {
(*(*__slate_slot_37)).hw_alt_next = (*(*(*ehci).r#async).hw).hw_alt_next;
}
// qh makes control packets use qtd toggle; maybe switch it
if (*__slate_slot_32 as u32) & (*__slate_slot_19).wrapping_add((*__slate_slot_32).wrapping_sub(1) as u32) == 0 {
*__slate_slot_30 ^= 2147483648u32;
}
if likely!(*__slate_slot_33 <= 0) != 0 {
let __v10: i32 = *__slate_slot_29;
let __v9: i32 = __v10.wrapping_sub(1);
*__slate_slot_29 = __v9;
if __v9 <= 0 || *__slate_slot_34 <= 0 {
break;
} else {
*__slate_slot_28 = sg_next(*__slate_slot_28);
*__slate_slot_35 = sg_dma_address!(*__slate_slot_28);
*__slate_slot_33 = min_t!(i32, sg_dma_len!(*__slate_slot_28), *__slate_slot_34);
}
}
*__slate_slot_36 = *__slate_slot_37;
*__slate_slot_37 = ehci_qtd_alloc(ehci, flags);
if unlikely!((*__slate_slot_37).is_null()) != 0 {
break '__join_0;
} else {
(*(*__slate_slot_37)).urb = urb;
(*(*__slate_slot_36)).hw_next = QTD_NEXT!(ehci, unsafe { (*(*__slate_slot_37)).qtd_dma });
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_37)).qtd_list), head);
}
}
// unless the caller requires manual cleanup after short reads,
// have the alt_next mechanism keep the queue running after the
// last data qtd (the only one, for control and most other cases).
if likely!((*urb).transfer_flags & 1 == 0 || usb_pipecontrol!(unsafe { (*urb).pipe })) != 0 {
(*(*__slate_slot_37)).hw_alt_next = EHCI_LIST_END!(ehci);
}
// control requests may need a terminating data "status" ack;
// other OUT ones may need a terminating short packet
// (zero length).
if likely!((*urb).transfer_buffer_length != 0) != 0 {
one_more = 0;
if usb_pipecontrol!(unsafe { (*urb).pipe }) {
one_more = 1;
*__slate_slot_30 ^= 256u32; // "in" <--> "out"
*__slate_slot_30 |= 2147483648u32; // force DATA1
} else {
if usb_pipeout!(unsafe { (*urb).pipe }) && (*urb).transfer_flags & 64 != 0 && (*urb).transfer_buffer_length % (*__slate_slot_32 as u32) == 0 {
one_more = 1;
}
}
if one_more != 0 {
*__slate_slot_36 = *__slate_slot_37;
*__slate_slot_37 = ehci_qtd_alloc(ehci, flags);
if unlikely!((*__slate_slot_37).is_null()) != 0 {
break '__join_0;
} else {
(*(*__slate_slot_37)).urb = urb;
(*(*__slate_slot_36)).hw_next = QTD_NEXT!(ehci, unsafe { (*(*__slate_slot_37)).qtd_dma });
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_37)).qtd_list), head);
// never any data in such packets
qtd_fill(ehci, *__slate_slot_37, 0u64, 0u64, *__slate_slot_30 as i32, 0i32);
}
}
}
// by default, enable interrupt on urb completion
if likely!((*urb).transfer_flags & 128 == 0) != 0 {
let __v3: u32 = cpu_to_hc32(ehci as *const ehci_hcd, 32768u32);
let __v2: *mut ehci_qtd = *__slate_slot_37;
(*__v2).hw_token = (*__v2).hw_token | __v3;
}
return head;
}
qtd_list_free(ehci, urb, head);
return core::ptr::null_mut::<list_head>();
}
}
}

fn qh_make(ehci: *mut ehci_hcd, urb: *mut urb, flags: u32) -> *mut ehci_qh {
// set the address of the TT; for TDI's integrated
// root hub tt, leave it zeroed.
let mut __v22: bool = unsafe { core::mem::zeroed() };
let mut think_time: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_42: *mut i32 = core::ptr::addr_of_mut!(think_time) as *mut i32;
let mut tmp: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_43: *mut u32 = core::ptr::addr_of_mut!(tmp) as *mut u32;
let mut hw: core::mem::MaybeUninit<*mut ehci_qh_hw> = core::mem::MaybeUninit::uninit();
let __slate_slot_48: *mut *mut ehci_qh_hw = core::ptr::addr_of_mut!(hw) as *mut *mut ehci_qh_hw;
let mut tt: *mut usb_tt = unsafe { core::mem::zeroed() };
let mut mult: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_50: *mut i32 = core::ptr::addr_of_mut!(mult) as *mut i32;
let mut maxp: i32 = unsafe { core::mem::zeroed() };
let mut r#type: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_52: *mut i32 = core::ptr::addr_of_mut!(r#type) as *mut i32;
let mut is_input: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_53: *mut i32 = core::ptr::addr_of_mut!(is_input) as *mut i32;
let mut info2: u32 = unsafe { core::mem::zeroed() };
let mut info1: u32 = unsafe { core::mem::zeroed() };
let mut ep: core::mem::MaybeUninit<*mut usb_host_endpoint> = core::mem::MaybeUninit::uninit();
let __slate_slot_56: *mut *mut usb_host_endpoint = core::ptr::addr_of_mut!(ep) as *mut *mut usb_host_endpoint;
let mut qh: *mut ehci_qh = unsafe { core::mem::zeroed() };
unsafe {
qh = ehci_qh_alloc(ehci, flags);
let __slate_auto_bytes_2 = __slate_slot_56 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
info1 = 0;
info2 = 0;
let __slate_auto_bytes_3 = __slate_slot_53 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_52 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
maxp = 0;
let __slate_auto_bytes_5 = __slate_slot_50 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
tt = (*(*urb).dev).tt;
let __slate_auto_bytes_6 = __slate_slot_48 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
if qh.is_null() {
return qh;
} else {
'__join_1: {
// init endpoint/device data for this QH
let __v47: u32 = info1;
info1 = __v47 | usb_pipeendpoint!(unsafe { (*urb).pipe }) << 8i32;
let __v45: u32 = info1;
info1 = __v45 | usb_pipedevice!(unsafe { (*urb).pipe }) << 0i32;
*__slate_slot_53 = usb_pipein!(unsafe { (*urb).pipe }) as i32;
*__slate_slot_52 = usb_pipetype!(unsafe { (*urb).pipe }) as i32;
*__slate_slot_56 = usb_pipe_endpoint((*urb).dev, (*urb).pipe);
maxp = usb_endpoint_maxp(core::ptr::addr_of_mut!((*(*__slate_slot_56)).desc) as *const usb_endpoint_descriptor);
*__slate_slot_50 = usb_endpoint_maxp_mult(core::ptr::addr_of_mut!((*(*__slate_slot_56)).desc) as *const usb_endpoint_descriptor);
// 1024 byte maxpacket is a hardware ceiling.  High bandwidth
// acts like up to 3KB, but is built from smaller packets.
if maxp <= 1024 {
// Compute interrupt scheduling parameters just once, and save.
// - allowing for high bandwidth, how many nsec/uframe are used?
// - split transactions need a second CSPLIT uframe; same question
// - splits also need a schedule gap (for full/low speed I/O)
// - qh has a polling interval
//
// For control/bulk requests, the HC or TT handles these.
if *__slate_slot_52 == 1 {
let __slate_auto_bytes_1 = __slate_slot_43 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
(*qh).ps.usecs = NS_TO_US__2!(usb_calc_bus_time(3i32, *__slate_slot_53, 0i32, (*__slate_slot_50).wrapping_mul(maxp))) as u8;
(*qh).ps.phase = 29999;
if ((*(*urb).dev).speed as u32) == 3 {
(*qh).ps.c_usecs = 0;
(*qh).gap_uf = 0;
if (*urb).interval > 1 && (*urb).interval < 8 {
// NOTE interval 2 or 4 uframes could work.
// But interval 1 scheduling is simpler, and
// includes high bandwidth.
(*urb).interval = 1;
} else {
if ((*urb).interval as u32) > (*ehci).periodic_size << 3i32 {
(*urb).interval = ((*ehci).periodic_size << 3i32) as i32;
}
}
(*qh).ps.period = ((*urb).interval >> 3i32) as u16;
// period for bandwidth allocation
*__slate_slot_43 = min_t__3!(u32, 64i32, 1i32 << ((*(*urb).ep).desc.bInterval as i32).wrapping_sub(1));
// Allow urb->interval to override
(*qh).ps.bw_uperiod = min_t__2!(u32, *__slate_slot_43, unsafe { (*urb).interval }) as u8;
(*qh).ps.bw_period = (((*qh).ps.bw_uperiod as i32) >> 3i32) as u8;
} else {
let __slate_auto_bytes = __slate_slot_42 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// gap is f(FS/LS transfer times)
(*qh).gap_uf = 1i64.wrapping_add(usb_calc_bus_time(((*(*urb).dev).speed as u32) as i32, *__slate_slot_53, 0i32, maxp) / 125000i64) as u8;
// FIXME this just approximates SPLIT/CSPLIT times
if *__slate_slot_53 != 0 {
// SPLIT, gap, CSPLIT+DATA
(*qh).ps.c_usecs = (((*qh).ps.usecs as i32) as u64).wrapping_add(HS_USECS!(0i32)) as u8;
(*qh).ps.usecs = HS_USECS!(1i32) as u8;
} else {
// SPLIT+DATA, gap, CSPLIT
let __v40: u8 = (*qh).ps.usecs;
let __v39: u8 = ((__v40 as i32) as u64).wrapping_add(HS_USECS!(1i32)) as u8;
(*qh).ps.usecs = __v39;
(*qh).ps.c_usecs = HS_USECS!(0i32) as u8;
}
*__slate_slot_42 = (if !tt.is_null() { (*tt).think_time } else { 0u32 }) as i32;
(*qh).ps.tt_usecs = NS_TO_US__2!((*__slate_slot_42 as i64).wrapping_add(usb_calc_bus_time(((*(*urb).dev).speed as u32) as i32, *__slate_slot_53, 0i32, maxp))) as u16;
if ((*urb).interval as u32) > (*ehci).periodic_size {
(*urb).interval = (*ehci).periodic_size as i32;
}
(*qh).ps.period = (*urb).interval as u16;
// period for bandwidth allocation
*__slate_slot_43 = min_t__3!(u32, 8i32, unsafe { (*(*urb).ep).desc.bInterval });
*__slate_slot_43 = rounddown_pow_of_two__2!(*__slate_slot_43) as u32;
// Allow urb->interval to override
(*qh).ps.bw_period = min_t__2!(u32, *__slate_slot_43, unsafe { (*urb).interval }) as u8;
(*qh).ps.bw_uperiod = (((*qh).ps.bw_period as i32) << 3i32) as u8;
}
}
'__join_0: {
// support for tt scheduling, and access to toggles
(*qh).ps.udev = (*urb).dev;
(*qh).ps.ep = (*urb).ep;
// using TT?
let __t1: u32 = (*(*urb).dev).speed as u32;
if __t1 == 1 {
info1 |= 4096u32;
} else {
if __t1 != 2 {
if __t1 == 3 {
info1 |= 8192u32; // no TT involved
if *__slate_slot_52 == 2 {
info1 |= 1073741824u32;
info1 |= 4194304u32; // usb2 fixed maxpacket
info1 |= 16384u32; // toggle from qtd
info2 |= 1073741824u32;
break '__join_0;
} else {
if *__slate_slot_52 == 3 {
info1 |= 1073741824u32;
// The USB spec says that high speed bulk endpoints
// always use 512 byte maxpacket.  But some device
// vendors decided to ignore that, and MSFT is happy
// to help them do so.  So now people expect to use
// such nonconformant devices with Linux too; sigh.
info1 |= (maxp << 16i32) as u32;
info2 |= 1073741824u32;
break '__join_0;
} else {
// PIPE_INTERRUPT
info1 |= (maxp << 16i32) as u32;
info2 |= (*__slate_slot_50 << 30i32) as u32;
break '__join_0;
}
}
} else {
break '__join_1;
}
}
}
if *__slate_slot_52 != 1 {
info1 |= 0u32;
}
// EPS 0 means "full"
if *__slate_slot_52 == 2 {
info1 |= 134217728u32; // for TT
info1 |= 16384u32; // toggle from qtd
}
info1 |= (maxp << 16i32) as u32;
info2 |= 1073741824u32;
// Some Freescale processors have an erratum in which the
// port number in the queue head was 0..N-1 instead of 1..N.
info2 |= ((*(*urb).dev).ttport << 23i32) as u32;
if !tt.is_null() {
__v22 = (*tt).hub != (*ehci_to_hcd(ehci)).self_.root_hub;
} else {
__v22 = false;
}
if __v22 {
info2 |= ((*(*tt).hub).devnum << 16i32) as u32;
}
// NOTE:  if (PIPE_INTERRUPT) { scheduler sets c-mask }
}
// NOTE:  if (PIPE_INTERRUPT) { scheduler sets s-mask }
// init as live, toggle clear
(*qh).qh_state = 3;
*__slate_slot_48 = (*qh).hw;
(*(*__slate_slot_48)).hw_info1 = cpu_to_hc32(ehci as *const ehci_hcd, info1);
(*(*__slate_slot_48)).hw_info2 = cpu_to_hc32(ehci as *const ehci_hcd, info2);
let __t0 = (*__slate_slot_53 == 0) as u32;
(*qh).__slate_bits_0.__set_is_out(__t0);
usb_settoggle__2!(unsafe { (*urb).dev }, (*urb).pipe >> 15i32 & 15, *__slate_slot_53 == 0, 1i32);
return qh;
}
}
qh_destroy(ehci, qh);
return core::ptr::null_mut::<ehci_qh>();
}
}
}

fn enable_async(ehci: *mut ehci_hcd) {
let __v1: u32 = unsafe { (*ehci).async_count };
unsafe {
(*ehci).async_count = __v1.wrapping_add(1);
}
if __v1 != 0 {
return;
}
// Stop waiting to turn off the async schedule
let __v4: u32 = unsafe { (*ehci).enabled_hrtimer_events };
let __v5: u32 = ((__v4 as u64) & !BIT!(10i32)) as u32;
unsafe {
(*ehci).enabled_hrtimer_events = __v5;
}
// Don't start the schedule until ASS is 0
ehci_poll_ASS(ehci);
turn_on_io_watchdog(ehci);
}

fn disable_async(ehci: *mut ehci_hcd) {
let __v1: u32 = unsafe { (*ehci).async_count };
let __v2: u32 = __v1.wrapping_sub(1);
unsafe {
(*ehci).async_count = __v2;
}
if __v2 != 0 {
return;
}
// The async schedule and unlink lists are supposed to be empty
let mut __v3: i64;
let mut __ret_warn_on: i32 = 0;
let __v4: bool = !unsafe { (*(*ehci).r#async).qh_next.qh }.is_null() || list_empty((unsafe { core::ptr::addr_of_mut!((*ehci).async_unlink) }) as *const list_head) == 0;
let __v5: bool = __v4 || list_empty((unsafe { core::ptr::addr_of_mut!((*ehci).async_idle) }) as *const list_head) == 0;
__ret_warn_on = __v5 as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("668: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 668b - ., 3; .popsection\n# {0}", const 668 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_disable_async_6, sym __slate_asm_string_disable_async_7, const 970 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("669: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 669b - ., 4; .popsection\n# {0}", const 669 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
// Don't turn off the schedule until ASS is 1
ehci_poll_ASS(ehci);
}

fn qh_link_async(ehci: *mut ehci_hcd, qh: *mut ehci_qh) {
let dma: u32 = QH_NEXT!(ehci, unsafe { (*qh).qh_dma });
// Don't link a QH if there's a Clear-TT-Buffer pending
if unlikely__2!((unsafe { (*qh).__slate_bits_0.__get_clearing_tt() }) as i32) != 0 {
return;
}
let mut __v0: i64;
let __ret_warn_on: i32 = (((unsafe { (*qh).qh_state }) as i32) != 3) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("670: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 670b - ., 3; .popsection\n# {0}", const 670 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_qh_link_async_1, sym __slate_asm_string_qh_link_async_2, const 987 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
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
__v0 = (__ret_warn_on != 0) as i64;
// clear halt and/or toggle; and maybe recover from silicon quirk
qh_refresh(ehci, qh);
// splice right after start
let head: *mut ehci_qh = unsafe { (*ehci).r#async };
unsafe {
(*qh).qh_next = (*head).qh_next;
}
unsafe {
(*(*qh).hw).hw_next = (*(*head).hw).hw_next;
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
(*head).qh_next.qh = qh;
}
unsafe {
(*(*head).hw).hw_next = dma;
}
unsafe {
(*qh).qh_state = 1;
}
unsafe {
(*qh).xacterrs = 0;
}
unsafe {
(*qh).unlink_reason = 0;
}
// qtd completions reported later by interrupt
enable_async(ehci);
}

fn qh_append_tds(ehci: *mut ehci_hcd, urb: *mut urb, qtd_list: *mut list_head, epnum: i32, ptr: *mut *mut ()) -> *mut ehci_qh {
let mut qh: *mut ehci_qh = core::ptr::null_mut::<ehci_qh>();
let qh_addr_mask: u32 = cpu_to_hc32(ehci as *const ehci_hcd, 127u32);
qh = (unsafe { *ptr }) as *mut ehci_qh;
if unlikely!(qh.is_null()) != 0 {
// can't sleep here, we have ehci->lock...
qh = qh_make(ehci, urb, (BIT!(5i32) as u32) | (BIT!(11i32) as u32));
unsafe {
*ptr = qh as *mut ();
}
}
if likely!(!qh.is_null()) != 0 {
let mut qtd: *mut ehci_qtd = unsafe { core::mem::zeroed() };
if unlikely__2!(list_empty(qtd_list as *const list_head)) != 0 {
qtd = core::ptr::null_mut::<ehci_qtd>();
} else {
qtd = list_entry__4!(unsafe { (*qtd_list).next }, ehci_qtd);
}
// control qh may need patching ...
if unlikely!(epnum == 0) != 0 {
// usb_reset_device() briefly reverts to address 0
if usb_pipedevice!(unsafe { (*urb).pipe }) == 0 {
let __v1: *mut ehci_qh_hw = unsafe { (*qh).hw };
let __v3: u32 = (unsafe { (*__v1).hw_info1 }) & !qh_addr_mask;
unsafe {
(*__v1).hw_info1 = __v3;
}
}
}
// just one way to queue requests: swap with the dummy qtd.
// only hc or qh_refresh() ever modify the overlay.
if likely!(!qtd.is_null()) != 0 {
let mut dummy: *mut ehci_qtd = unsafe { core::mem::zeroed() };
let mut dma: u64 = 0;
let mut token: u32 = 0;
// to avoid racing the HC, use the dummy td instead of
// the first td of our list (becomes new dummy).  both
// tds stay deactivated until we're done, when the
// HC is allowed to fetch the old dummy (4.10.2).
token = unsafe { (*qtd).hw_token };
unsafe {
(*qtd).hw_token = HALT_BIT!(ehci);
}
dummy = unsafe { (*qh).dummy };
dma = unsafe { (*dummy).qtd_dma };
unsafe {
*dummy = *qtd;
}
unsafe {
(*dummy).qtd_dma = dma;
}
list_del(unsafe { core::ptr::addr_of_mut!((*qtd).qtd_list) });
list_add(unsafe { core::ptr::addr_of_mut!((*dummy).qtd_list) }, qtd_list);
list_splice_tail(qtd_list, unsafe { core::ptr::addr_of_mut!((*qh).qtd_list) });
ehci_qtd_init(ehci, qtd, unsafe { (*qtd).qtd_dma });
unsafe {
(*qh).dummy = qtd;
}
// hc must see the new dummy at list end
dma = unsafe { (*qtd).qtd_dma };
qtd = list_entry__4!(unsafe { (*qh).qtd_list.prev }, ehci_qtd);
unsafe {
(*qtd).hw_next = QTD_NEXT!(ehci, dma);
}
// let the hc process these next qtds
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
(*dummy).hw_token = token;
}
unsafe {
(*urb).hcpriv = qh as *mut ();
}
}
}
qh
}

fn submit_async(ehci: *mut ehci_hcd, urb: *mut urb, qtd_list: *mut list_head, mem_flags: u32) -> i32 {
let mut rc: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(rc) as *mut i32;
let mut qh: *mut ehci_qh = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut epnum: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(epnum) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
qh = core::ptr::null_mut::<ehci_qh>();
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_3 = (*(*urb).ep).desc.bEndpointAddress as i32;
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_2);
if unlikely!(HCD_HW_ACCESSIBLE!(ehci_to_hcd(ehci)) == 0) != 0 {
*__slate_slot_0 = 108i32.wrapping_neg();
} else {
*__slate_slot_0 = usb_hcd_link_urb_to_ep(ehci_to_hcd(ehci), urb);
if unlikely__2!(*__slate_slot_0) == 0 {
qh = qh_append_tds(ehci, urb, qtd_list, *__slate_slot_3, core::ptr::addr_of_mut!((*(*urb).ep).hcpriv));
if unlikely!(qh.is_null()) != 0 {
usb_hcd_unlink_urb_from_ep(ehci_to_hcd(ehci), urb);
*__slate_slot_0 = 12i32.wrapping_neg();
} else {
// Control/bulk operations through TTs don't need scheduling,
// the HC and TT handle it when the TT has a buffer ready.
if likely!(((*qh).qh_state as i32) == 3) != 0 {
qh_link_async(ehci, qh);
}
}
}
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_2);
if unlikely!(qh.is_null()) != 0 {
qtd_list_free(ehci, urb, qtd_list);
}
return *__slate_slot_0;
}
}

fn single_unlink_async(ehci: *mut ehci_hcd, qh: *mut ehci_qh) {
let mut prev: *mut ehci_qh = unsafe { core::mem::zeroed() };
// Add to the end of the list of QHs waiting for the next IAAD
unsafe {
(*qh).qh_state = 4;
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*qh).unlink_node) }, unsafe { core::ptr::addr_of_mut!((*ehci).async_unlink) });
// Unlink it from the schedule
prev = unsafe { (*ehci).r#async };
while (unsafe { (*prev).qh_next.qh }) != qh {
prev = unsafe { (*prev).qh_next.qh };
}
unsafe {
(*(*prev).hw).hw_next = (*(*qh).hw).hw_next;
}
unsafe {
(*prev).qh_next = (*qh).qh_next;
}
if (unsafe { (*ehci).qh_scan_next }) == qh {
unsafe {
(*ehci).qh_scan_next = (*qh).qh_next.qh;
}
}
}

fn start_iaa_cycle(ehci: *mut ehci_hcd) {
// If the controller isn't running, we don't have to wait for it
if unlikely!(((unsafe { (*ehci).rh_state }) as u32) < 2) != 0 {
end_unlink_async(ehci);
// Otherwise start a new IAA cycle if one isn't already running
} else {
if ((unsafe { (*ehci).rh_state }) as u32) == 2 && ((unsafe { (*ehci).__slate_bits_0.__get_iaa_in_progress() }) as i32) == 0 {
// Make sure the unlinks are all visible to the hardware
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
ehci_writel(ehci as *const ehci_hcd, (unsafe { (*ehci).command }) | 64u32, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) });
ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) });
unsafe {
let __t0 = true;
(*ehci).__slate_bits_0.__set_iaa_in_progress(__t0);
}
ehci_enable_event(ehci, 8u32, true);
}
}
}

fn end_iaa_cycle(ehci: *mut ehci_hcd) {
if ((unsafe { (*ehci).__slate_bits_1.__get_has_synopsys_hc_bug() }) as i32) != 0 {
ehci_writel(ehci as *const ehci_hcd, (unsafe { (*(*ehci).r#async).qh_dma }) as u32, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).async_next) });
}
// The current IAA cycle has ended
unsafe {
let __t0 = false;
(*ehci).__slate_bits_0.__set_iaa_in_progress(__t0);
}
end_unlink_async(ehci);
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.end_unlink_async")]
extern "C" fn end_unlink_async(ehci: *mut ehci_hcd) {
let mut qh_token: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u32 = core::ptr::addr_of_mut!(qh_token) as *mut u32;
let mut qh_current: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u32 = core::ptr::addr_of_mut!(qh_current) as *mut u32;
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut early_exit: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut bool = core::ptr::addr_of_mut!(early_exit) as *mut bool;
let mut qh: core::mem::MaybeUninit<*mut ehci_qh> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut ehci_qh = core::ptr::addr_of_mut!(qh) as *mut *mut ehci_qh;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
if list_empty(core::ptr::addr_of_mut!((*ehci).async_unlink) as *const list_head) != 0 {
return;
} else {
*__slate_slot_4 = list_first_entry__7!(core::ptr::addr_of_mut!((*ehci).async_unlink), ehci_qh); // QH whose IAA cycle just ended
// If async_unlinking is set then this routine is already running,
// either on the stack or on another CPU.
*__slate_slot_3 = ((*ehci).__slate_bits_0.__get_async_unlinking() as i32) != 0;
// If the controller isn't running, process all the waiting QHs
if ((*ehci).rh_state as u32) < 2 {
list_splice_tail_init(core::ptr::addr_of_mut!((*ehci).async_unlink), core::ptr::addr_of_mut!((*ehci).async_idle));
} else {
if ((*(*__slate_slot_4)).qh_state as i32) == 2 {
// Second IAA cycle has finished.  Process only the first
// waiting QH (NVIDIA (?) bug).
list_move_tail(core::ptr::addr_of_mut!((*(*__slate_slot_4)).unlink_node), core::ptr::addr_of_mut!((*ehci).async_idle));
} else {
if ((*(*__slate_slot_4)).unlink_reason as i32) & (1 | 2 | 4) == 0 {
if ((*(*__slate_slot_4)).unlink_reason as i32) & 16 != 0 {
__v2 = list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_4)).qtd_list) as *const list_head) != 0;
} else {
__v2 = false;
}
if !__v2 {
if (*(*(*__slate_slot_4)).hw).hw_token & cpu_to_hc32(ehci as *const ehci_hcd, 64u32) == 0 {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_1 = (*(*(*__slate_slot_4)).hw).hw_current;
*__slate_slot_0 = (*(*(*__slate_slot_4)).hw).hw_token;
if *__slate_slot_1 != (*ehci).old_current || *__slate_slot_0 != (*ehci).old_token {
(*ehci).old_current = *__slate_slot_1;
(*ehci).old_token = *__slate_slot_0;
ehci_enable_event(ehci, 5u32, true);
return;
}
}
}
}
(*(*__slate_slot_4)).qh_state = 2;
*__slate_slot_3 = true;
}
}
// Intel (?) bug: The HC can write back the overlay region even
// after the IAA interrupt occurs.  In self-defense, always go
// through two IAA cycles for each QH.
// AMD/ATI (?) bug: The HC can continue to use an active QH long
// after the IAA interrupt occurs.  To prevent problems, QHs that
// may still be active will wait until 2 ms have passed with no
// change to the hw_current and hw_token fields (this delay occurs
// between the two IAA cycles).
//
// The EHCI spec (4.8.2) says that active QHs must not be removed
// from the async schedule and recommends waiting until the QH
// goes inactive.  This is ridiculous because the QH will _never_
// become inactive if the endpoint NAKs indefinitely.
// Some reasons for unlinking guarantee the QH can't be active
// The QH can't be active if the queue was and still is empty...
// ... or if the QH has halted
// Otherwise we have to wait until the QH stops changing
(*ehci).old_current = 4294967295u32;
// Prepare for next QH
// Start a new IAA cycle if any QHs are waiting for it
if list_empty(core::ptr::addr_of_mut!((*ehci).async_unlink) as *const list_head) == 0 {
start_iaa_cycle(ehci);
}
// Don't allow nesting or concurrent calls,
// or wait for the second IAA cycle for the next QH.
if *__slate_slot_3 {
return;
} else {
// Process the idle QHs
let __t1 = true;
(*ehci).__slate_bits_0.__set_async_unlinking(__t1);
while list_empty(core::ptr::addr_of_mut!((*ehci).async_idle) as *const list_head) == 0 {
*__slate_slot_4 = list_first_entry__7!(core::ptr::addr_of_mut!((*ehci).async_idle), ehci_qh);
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_4)).unlink_node));
(*(*__slate_slot_4)).qh_state = 3;
(*(*__slate_slot_4)).qh_next.qh = core::ptr::null_mut::<ehci_qh>();
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_4)).qtd_list) as *const list_head) == 0 {
qh_completions(ehci, *__slate_slot_4);
}
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_4)).qtd_list) as *const list_head) == 0 && ((*ehci).rh_state as u32) == 2 {
qh_link_async(ehci, *__slate_slot_4);
}
disable_async(ehci);
}
let __t0 = false;
(*ehci).__slate_bits_0.__set_async_unlinking(__t0);
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.unlink_empty_async")]
extern "C" fn unlink_empty_async(ehci: *mut ehci_hcd) {
let mut qh: *mut ehci_qh = unsafe { core::mem::zeroed() };
let mut qh_to_unlink: *mut ehci_qh = core::ptr::null_mut::<ehci_qh>();
let mut count: i32 = 0;
// Find the last async QH which has been empty for a timer cycle
qh = unsafe { (*(*ehci).r#async).qh_next.qh };
while !qh.is_null() {
if list_empty((unsafe { core::ptr::addr_of_mut!((*qh).qtd_list) }) as *const list_head) != 0 && ((unsafe { (*qh).qh_state }) as i32) == 1 {
count = count.wrapping_add(1);
if (unsafe { (*qh).unlink_cycle }) != unsafe { (*ehci).async_unlink_cycle } {
qh_to_unlink = qh;
}
}
qh = unsafe { (*qh).qh_next.qh };
}
// If nothing else is being unlinked, unlink the last empty QH
if list_empty((unsafe { core::ptr::addr_of_mut!((*ehci).async_unlink) }) as *const list_head) != 0 && !qh_to_unlink.is_null() {
unsafe {
(*qh_to_unlink).unlink_reason = (((*qh_to_unlink).unlink_reason as i32) | 16) as u8;
}
start_unlink_async(ehci, qh_to_unlink);
count = count.wrapping_sub(1);
}
// Other QHs will be handled later
if count > 0 {
ehci_enable_event(ehci, 7u32, true);
let __v9: u32 = unsafe { (*ehci).async_unlink_cycle };
unsafe {
(*ehci).async_unlink_cycle = __v9.wrapping_add(1);
}
}
}

fn unlink_empty_async_suspended(ehci: *mut ehci_hcd) {
let mut qh: *mut ehci_qh = unsafe { core::mem::zeroed() };
while !unsafe { (*(*ehci).r#async).qh_next.qh }.is_null() {
qh = unsafe { (*(*ehci).r#async).qh_next.qh };
let mut __v0: i64;
let __ret_warn_on: i32 = (list_empty((unsafe { core::ptr::addr_of_mut!((*qh).qtd_list) }) as *const list_head) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("672: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 672b - ., 3; .popsection\n# {0}", const 672 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_unlink_empty_async_suspended_1, sym __slate_asm_string_unlink_empty_async_suspended_2, const 1460 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
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
__v0 = (__ret_warn_on != 0) as i64;
single_unlink_async(ehci, qh);
}
}

fn start_unlink_async(ehci: *mut ehci_hcd, qh: *mut ehci_qh) {
// If the QH isn't linked then there's nothing we can do.
if ((unsafe { (*qh).qh_state }) as i32) != 1 {
return;
}
single_unlink_async(ehci, qh);
start_iaa_cycle(ehci);
}

fn scan_async(ehci: *mut ehci_hcd) {
let mut qh: *mut ehci_qh = unsafe { core::mem::zeroed() };
let mut check_unlinks_later: bool = false;
unsafe {
(*ehci).qh_scan_next = (*(*ehci).r#async).qh_next.qh;
}
while !unsafe { (*ehci).qh_scan_next }.is_null() {
qh = unsafe { (*ehci).qh_scan_next };
unsafe {
(*ehci).qh_scan_next = (*qh).qh_next.qh;
}
// clean any finished work for this qh
if list_empty((unsafe { core::ptr::addr_of_mut!((*qh).qtd_list) }) as *const list_head) == 0 {
let mut temp: i32 = 0;
// Unlinks could happen here; completion reporting
// drops the lock.  That's why ehci->qh_scan_next
// always holds the next qh to scan; if the next qh
// gets unlinked then ehci->qh_scan_next is adjusted
// in single_unlink_async().
temp = qh_completions(ehci, qh) as i32;
if unlikely__2!(temp) != 0 {
start_unlink_async(ehci, qh);
} else {
if list_empty((unsafe { core::ptr::addr_of_mut!((*qh).qtd_list) }) as *const list_head) != 0 && ((unsafe { (*qh).qh_state }) as i32) == 1 {
unsafe {
(*qh).unlink_cycle = (*ehci).async_unlink_cycle;
}
check_unlinks_later = true;
}
}
}
}
// Unlink empty entries, reducing DMA usage as well
// as HCD schedule-scanning costs.  Delay for any qh
// we just scanned, there's a not-unusual case that it
// doesn't stay idle for long.
if check_unlinks_later && ((unsafe { (*ehci).rh_state }) as u32) == 2 && ((unsafe { (*ehci).enabled_hrtimer_events }) as u64) & BIT!(7i32) == 0 {
ehci_enable_event(ehci, 7u32, true);
let __v2: u32 = unsafe { (*ehci).async_unlink_cycle };
unsafe {
(*ehci).async_unlink_cycle = __v2.wrapping_add(1);
}
}
}

fn periodic_next_shadow(ehci: *mut ehci_hcd, periodic: *mut ehci_shadow, tag: u32) -> *mut ehci_shadow {
match hc32_to_cpu(ehci as *const ehci_hcd, tag) {
2 => {
return unsafe { core::ptr::addr_of_mut!((*(*periodic).qh).qh_next) };
}
6 => {
return unsafe { core::ptr::addr_of_mut!((*(*periodic).fstn).fstn_next) };
}
0 => {
return unsafe { core::ptr::addr_of_mut!((*(*periodic).itd).itd_next) };
// case Q_TYPE_SITD:
}
_ => {
return unsafe { core::ptr::addr_of_mut!((*(*periodic).sitd).sitd_next) };
}
}
unsafe { core::mem::zeroed() }
}

fn shadow_next_periodic(ehci: *mut ehci_hcd, periodic: &ehci_shadow, tag: u32) -> *mut u32 {
match hc32_to_cpu(ehci as *const ehci_hcd, tag) {
2 => {
// our ehci_shadow.qh is actually software part
return unsafe { core::ptr::addr_of_mut!((*(*(*(periodic as *const ehci_shadow)).qh).hw).hw_next) };
// others are hw parts
}
_ => {
return unsafe { (*(periodic as *const ehci_shadow)).hw_next };
}
}
unsafe { core::mem::zeroed() }
}

fn periodic_unlink(ehci: *mut ehci_hcd, frame: u32, ptr: *mut ()) {
let mut prev_p: *mut ehci_shadow = unsafe { (*ehci).pshadow }.wrapping_offset(frame as isize);
let mut hw_p: *mut u32 = unsafe { (*ehci).periodic }.wrapping_offset(frame as isize);
let mut here: ehci_shadow = unsafe { *prev_p };
// find predecessor of "ptr"; hw and shadow lists are in sync
while !unsafe { here.ptr }.is_null() && (unsafe { here.ptr }) != ptr {
prev_p = periodic_next_shadow(ehci, prev_p, Q_NEXT_TYPE!(ehci, unsafe { *hw_p }));
hw_p = shadow_next_periodic(ehci, unsafe { &(*core::ptr::addr_of_mut!(here)) }, Q_NEXT_TYPE!(ehci, unsafe { *hw_p }));
here = unsafe { *prev_p };
}
// an interrupt entry (at list end) could have been shared
if unsafe { here.ptr }.is_null() {
return;
}
// update shadow and hardware lists ... the old "next" pointers
// from ptr may still be in use, the caller updates them.
unsafe {
*prev_p = *periodic_next_shadow(ehci, core::ptr::addr_of_mut!(here), Q_NEXT_TYPE!(ehci, unsafe { *hw_p }));
}
let __v1: bool = ((unsafe { (*ehci).__slate_bits_1.__get_use_dummy_qh() }) as i32) == 0 || (unsafe { *shadow_next_periodic(ehci, &(*core::ptr::addr_of_mut!(here)), Q_NEXT_TYPE!(ehci, unsafe { *hw_p })) }) != EHCI_LIST_END!(ehci);
if __v1 {
unsafe {
*hw_p = *shadow_next_periodic(ehci, &(*core::ptr::addr_of_mut!(here)), Q_NEXT_TYPE!(ehci, unsafe { *hw_p }));
}
} else {
unsafe {
*hw_p = cpu_to_hc32(ehci as *const ehci_hcd, (*(*ehci).dummy).qh_dma as u32);
}
}
}

fn find_tt(udev: *mut usb_device) -> *mut ehci_tt {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let utt: *mut usb_tt = unsafe { (*udev).tt };
let mut tt: *mut ehci_tt = unsafe { core::mem::zeroed() };
let mut tt_index: *mut *mut ehci_tt = unsafe { core::mem::zeroed() };
let mut ptt: *mut *mut ehci_tt = unsafe { core::mem::zeroed() };
let mut port: u32 = 0;
let mut allocated_index: bool = false;
if utt.is_null() {
return core::ptr::null_mut::<ehci_tt>();
}
// Not below a TT
// Find/create our data structure.
// For hubs with a single TT, we get it directly.
// For hubs with multiple TTs, there's an extra level of pointers.
tt_index = core::ptr::null_mut::<*mut ehci_tt>();
if (unsafe { (*utt).multi }) != 0 {
tt_index = (unsafe { (*utt).hcpriv }) as *mut *mut ehci_tt;
if tt_index.is_null() {
// Create the index array
let mut __v0: *mut *mut ehci_tt;
let __obj_size: u64 = size_mul(core::mem::size_of::<*mut ehci_tt>() as u64, (unsafe { (*(*utt).hub).maxchild }) as u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 2080u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 2080u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut *mut ehci_tt;
tt_index = __v0;
if tt_index.is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut ehci_tt;
}
unsafe {
(*utt).hcpriv = tt_index as *mut ();
}
allocated_index = true;
}
port = unsafe { (*udev).ttport }.wrapping_sub(1) as u32;
ptt = tt_index.wrapping_offset(port as isize);
} else {
port = 0;
ptt = (unsafe { core::ptr::addr_of_mut!((*utt).hcpriv) }) as *mut *mut ehci_tt;
}
tt = unsafe { *ptt };
if tt.is_null() {
// Create the ehci_tt
let ehci: *mut ehci_hcd = hcd_to_ehci(bus_to_hcd(unsafe { (*udev).bus }));
let mut __v6: *mut ehci_tt;
let __obj_size: u64 = size_mul(core::mem::size_of::<ehci_tt>() as u64, 1u64);
let mut __v7: *mut ();
let mut __v8: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 2080u32, unsafe { *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 2080u32, unsafe { *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
} });
}
__v8 = _res;
__v7 = __v8;
__v6 = __v7 as *mut ehci_tt;
tt = __v6;
if tt.is_null() {
if allocated_index {
unsafe {
(*utt).hcpriv = core::ptr::null_mut::<()>();
}
unsafe { kfree(tt_index as *const ()) };
}
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut ehci_tt;
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*tt).tt_list) }, unsafe { core::ptr::addr_of_mut!((*ehci).tt_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*tt).ps_list) });
unsafe {
(*tt).usb_tt = utt;
}
unsafe {
(*tt).tt_port = port as i32;
}
unsafe {
*ptt = tt;
}
}
tt
}

fn drop_tt(udev: *mut usb_device) {
let utt: *mut usb_tt = unsafe { (*udev).tt };
let mut tt: *mut ehci_tt = unsafe { core::mem::zeroed() };
let mut tt_index: *mut *mut ehci_tt = unsafe { core::mem::zeroed() };
let mut ptt: *mut *mut ehci_tt = unsafe { core::mem::zeroed() };
let mut cnt: i32 = 0;
let mut i: i32 = 0;
if utt.is_null() || unsafe { (*utt).hcpriv }.is_null() {
return;
}
// Not below a TT, or never allocated
if (unsafe { (*utt).multi }) != 0 {
tt_index = (unsafe { (*utt).hcpriv }) as *mut *mut ehci_tt;
ptt = tt_index.wrapping_offset(unsafe { (*udev).ttport }.wrapping_sub(1) as isize);
// How many entries are left in tt_index?
i = 0;
while i < unsafe { (*(*utt).hub).maxchild } {
let __v0: i32 = cnt;
cnt = __v0.wrapping_add(!unsafe { *tt_index.wrapping_offset(i as isize) }.is_null() as i32);
i = i.wrapping_add(1);
}
} else {
tt_index = core::ptr::null_mut::<*mut ehci_tt>();
ptt = (unsafe { core::ptr::addr_of_mut!((*utt).hcpriv) }) as *mut *mut ehci_tt;
}
tt = unsafe { *ptt };
let __v5: bool = tt.is_null() || list_empty((unsafe { core::ptr::addr_of_mut!((*tt).ps_list) }) as *const list_head) == 0;
if __v5 {
return;
}
// never allocated, or still in use
list_del(unsafe { core::ptr::addr_of_mut!((*tt).tt_list) });
unsafe {
*ptt = core::ptr::null_mut::<ehci_tt>();
}
unsafe { kfree(tt as *const ()) };
if cnt == 1 {
unsafe {
(*utt).hcpriv = core::ptr::null_mut::<()>();
}
unsafe { kfree(tt_index as *const ()) };
}
}

fn bandwidth_dbg(ehci: *mut ehci_hcd, sign: i32, r#type: *mut u8, ps: *mut ehci_per_sched) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*ps).udev).dev) as *const device, c"ep %02x: %s %s @ %u+%u (%u.%u+%u) [%u/%u us] mask %04x\n".as_ptr() as *const u8, (*(*ps).ep).desc.bEndpointAddress as i32, if sign >= 0 { c"reserve".as_ptr() as *mut u8 } else { c"release".as_ptr() as *mut u8 }, r#type, (((*ps).bw_phase as i32) << 3i32).wrapping_add((*ps).phase_uf as i32), (*ps).bw_uperiod as i32, (*ps).phase as i32, (*ps).phase_uf as i32, (*ps).period as i32, (*ps).usecs as i32, (*ps).c_usecs as i32, (*ps).cs_mask as i32) };
}
}

fn reserve_release_intr_bandwidth(ehci: *mut ehci_hcd, qh: *mut ehci_qh, sign: i32) {
let mut start_uf: u32 = 0;
let mut i: u32 = 0;
let mut j: u32 = 0;
let mut m: u32 = 0;
let mut usecs: i32 = (unsafe { (*qh).ps.usecs }) as i32;
let mut c_usecs: i32 = (unsafe { (*qh).ps.c_usecs }) as i32;
let mut tt_usecs: i32 = (unsafe { (*qh).ps.tt_usecs }) as i32;
let mut tt: *mut ehci_tt = unsafe { core::mem::zeroed() };
if ((unsafe { (*qh).ps.phase }) as i32) == 29999 {
return;
}
// Bandwidth wasn't reserved
start_uf = (((unsafe { (*qh).ps.bw_phase }) as i32) << 3i32) as u32;
bandwidth_dbg(ehci, sign, c"intr".as_ptr() as *mut u8, unsafe { core::ptr::addr_of_mut!((*qh).ps) });
if sign < 0 {
// Release bandwidth
usecs = usecs.wrapping_neg();
c_usecs = c_usecs.wrapping_neg();
tt_usecs = tt_usecs.wrapping_neg();
}
// Entire transaction (high speed) or start-split (full/low speed)
i = start_uf.wrapping_add(((unsafe { (*qh).ps.phase_uf }) as i32) as u32);
while i < 64 {
let __v0: *mut u8 = unsafe { (*ehci).bandwidth.as_mut_ptr() as *mut u8 }.wrapping_offset(i as isize);
unsafe {
*__v0 = (*__v0 as i32).wrapping_add(usecs) as u8;
}
i = i.wrapping_add(((unsafe { (*qh).ps.bw_uperiod }) as i32) as u32);
}
// Complete-split (full/low speed)
if (unsafe { (*qh).ps.c_usecs }) != 0 {
// NOTE: adjustments needed for FSTN
i = start_uf;
while i < 64 {
j = 2;
m = (1i32 << j.wrapping_add(8)) as u32;
while j < 8 {
if (((unsafe { (*qh).ps.cs_mask }) as i32) as u32) & m != 0 {
let __v7: *mut u8 = unsafe { (*ehci).bandwidth.as_mut_ptr() as *mut u8 }.wrapping_offset(i.wrapping_add(j) as isize);
unsafe {
*__v7 = (*__v7 as i32).wrapping_add(c_usecs) as u8;
}
}
j = j.wrapping_add(1);
m <<= 1i32;
}
i = i.wrapping_add(((unsafe { (*qh).ps.bw_uperiod }) as i32) as u32);
}
}
// FS/LS bus bandwidth
if tt_usecs != 0 {
// find_tt() will not return any error here as we have
// already called find_tt() before calling this function
// and checked for any error return. The previous call
// would have created the data structure.
tt = find_tt(unsafe { (*qh).ps.udev });
if sign > 0 {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*qh).ps.ps_list) }, unsafe { core::ptr::addr_of_mut!((*tt).ps_list) });
} else {
list_del(unsafe { core::ptr::addr_of_mut!((*qh).ps.ps_list) });
}
i = start_uf >> 3i32;
while i < 8u32 {
let __v18: *mut u16 = unsafe { (*tt).bandwidth.as_mut_ptr() as *mut u16 }.wrapping_offset(i as isize);
unsafe {
*__v18 = (*__v18 as i32).wrapping_add(tt_usecs) as u16;
}
i = i.wrapping_add(((unsafe { (*qh).ps.bw_period }) as i32) as u32);
}
}
}

fn compute_tt_budget(budget_table: *mut u8, tt: *mut ehci_tt) {
let mut ps: *mut ehci_per_sched = unsafe { core::mem::zeroed() };
let mut uframe: u32 = 0;
let mut x: u32 = 0;
if tt.is_null() {
return;
}
unsafe { memset(budget_table as *mut (), 0i32, 64u64) };
// Add up the contributions from all the endpoints using this TT
ps = list_first_entry__15!(unsafe { core::ptr::addr_of_mut!((*tt).ps_list) });
while list_entry_is_head!(ps, unsafe { core::ptr::addr_of_mut!((*tt).ps_list) }, ps_list) == 0 {
uframe = (((unsafe { (*ps).bw_phase }) as i32) << 3i32) as u32;
while uframe < 64 {
let budget_line: *mut u8 = budget_table.wrapping_offset(uframe as isize);
x = (unsafe { (*ps).tt_usecs }) as u32;
// propagate the time forward
for uf in (unsafe { (*ps).phase_uf }) as u32..8 {
x = x.wrapping_add(((unsafe { *budget_line.wrapping_offset(uf as isize) }) as i32) as u32);
// Each microframe lasts 125 us
if x <= 125 {
unsafe {
*budget_line.wrapping_offset(uf as isize) = x as u8;
}
break;
}
unsafe {
*budget_line.wrapping_offset(uf as isize) = 125;
}
x = x.wrapping_sub(125);
}
uframe = uframe.wrapping_add(((unsafe { (*ps).bw_uperiod }) as i32) as u32);
}
ps = list_next_entry__18!(ps, ps_list);
}
}

static max_tt_usecs: [u8; 8] = [125, 125, 125, 125, 125, 125, 30, 0];

fn carryover_tt_bandwidth(tt_usecs: *mut u16) {
for i in 0i32..7i32 {
if ((unsafe { max_tt_usecs[i as usize] }) as i32) < ((unsafe { *tt_usecs.wrapping_offset(i as isize) }) as i32) {
let __v0: *mut u16 = tt_usecs.wrapping_offset(i.wrapping_add(1) as isize);
unsafe {
*__v0 = (*__v0 as i32).wrapping_add((*tt_usecs.wrapping_offset(i as isize) as i32).wrapping_sub(max_tt_usecs[i as usize] as i32)) as u16;
}
unsafe {
*tt_usecs.wrapping_offset(i as isize) = max_tt_usecs[i as usize] as u16;
}
}
}
}

fn tt_available(ehci: *mut ehci_hcd, ps: *mut ehci_per_sched, tt: *mut ehci_tt, mut frame: u32, uframe: u32) -> i32 {
let period: u32 = (unsafe { (*ps).bw_period }) as u32;
let usecs: u32 = (unsafe { (*ps).tt_usecs }) as u32;
if period == 0 || uframe >= 7 {
return 0;
}
// error
frame &= period.wrapping_sub(1);
while frame < 8u32 {
let mut i: u32 = 0;
let mut uf: u32 = 0;
let mut tt_usecs: __SlateAlign16<[u16; 8]> = __SlateAlign16([0u16; 8]);
if (((unsafe { *((*tt).bandwidth.as_mut_ptr() as *mut u16).wrapping_offset(frame as isize) }) as i32) as u32).wrapping_add(usecs) > 900 {
return 0;
}
uf = frame << 3i32;
while i < 8 {
tt_usecs.0[i as usize] = (unsafe { *((*ehci).tt_budget.as_mut_ptr() as *mut u8).wrapping_offset(uf as isize) }) as u16;
i = i.wrapping_add(1);
uf = uf.wrapping_add(1);
}
if ((unsafe { max_tt_usecs[uframe as usize] }) as i32) <= (tt_usecs.0[uframe as usize] as i32) {
return 0;
}
// special case for isoc transfers larger than 125us:
// the first and each subsequent fully used uframe
// must be empty, so as to not illegally delay
// already scheduled transactions
if usecs > 125 {
let ufs: i32 = (usecs / 125) as i32;
i = uframe;
while i < uframe.wrapping_add(ufs as u32) && i < 8 {
if (tt_usecs.0[i as usize] as i32) > 0 {
return 0;
}
i = i.wrapping_add(1);
}
}
let __v10: *mut u16 = (tt_usecs.0.as_mut_ptr() as *mut u16).wrapping_offset(uframe as isize);
unsafe {
*__v10 = ((*__v10 as i32) as u32).wrapping_add(usecs) as u16;
}
carryover_tt_bandwidth(tt_usecs.0.as_mut_ptr() as *mut u16);
// fail if the carryover pushed bw past the last uframe's limit
if ((unsafe { max_tt_usecs[7] }) as i32) < (tt_usecs.0[7] as i32) {
return 0;
}
frame = frame.wrapping_add(period);
}
1
}

fn enable_periodic(ehci: *mut ehci_hcd) {
unsafe {
let __v4: u32 = (*ehci).periodic_count;
(*ehci).periodic_count = __v4.wrapping_add(1);
if __v4 == 0 {
// Stop waiting to turn off the periodic schedule
let __v1: u32 = (*ehci).enabled_hrtimer_events;
let __v0: u32 = ((__v1 as u64) & !BIT!(9i32)) as u32;
(*ehci).enabled_hrtimer_events = __v0;
// Don't start the schedule until PSS is 0
ehci_poll_PSS(ehci);
}
turn_on_io_watchdog(ehci);
}
}

fn disable_periodic(ehci: *mut ehci_hcd) {
let __v1: u32 = unsafe { (*ehci).periodic_count };
let __v2: u32 = __v1.wrapping_sub(1);
unsafe {
(*ehci).periodic_count = __v2;
}
if __v2 != 0 {
return;
}
// Don't turn off the schedule until PSS is 1
ehci_poll_PSS(ehci);
}

fn qh_link_periodic(ehci: *mut ehci_hcd, qh: *mut ehci_qh) {
let mut i: u32 = 0;
let mut period: u32 = (unsafe { (*qh).ps.period }) as u32;
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*qh).ps.udev).dev) as *const device, c"link qh%d-%04x/%p start %d [%d/%d us]\n".as_ptr() as *const u8, period, hc32_to_cpup(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*(*qh).hw).hw_info2) as *const u32) & 65535u32, qh, (*qh).ps.phase as i32, (*qh).ps.usecs as i32, (*qh).ps.c_usecs as i32) };
}
// high bandwidth, or otherwise every microframe
if period == 0 {
period = 1;
}
i = (unsafe { (*qh).ps.phase }) as u32;
while i < unsafe { (*ehci).periodic_size } {
let mut prev: *mut ehci_shadow = unsafe { (*ehci).pshadow }.wrapping_offset(i as isize);
let mut hw_p: *mut u32 = unsafe { (*ehci).periodic }.wrapping_offset(i as isize);
let mut here: ehci_shadow = unsafe { *prev };
let mut r#type: u32 = 0;
// skip the iso nodes at list head
while !unsafe { here.ptr }.is_null() {
r#type = Q_NEXT_TYPE!(ehci, unsafe { *hw_p });
if r#type == cpu_to_hc32(ehci as *const ehci_hcd, 2u32) {
break;
}
prev = periodic_next_shadow(ehci, prev, r#type);
hw_p = shadow_next_periodic(ehci, unsafe { &(*core::ptr::addr_of_mut!(here)) }, r#type);
here = unsafe { *prev };
}
// sorting each branch by period (slow-->fast)
// enables sharing interior tree nodes
while !unsafe { here.ptr }.is_null() && qh != unsafe { here.qh } {
if ((unsafe { (*qh).ps.period }) as i32) > ((unsafe { (*here.qh).ps.period }) as i32) {
break;
}
prev = unsafe { core::ptr::addr_of_mut!((*here.qh).qh_next) };
hw_p = unsafe { core::ptr::addr_of_mut!((*(*here.qh).hw).hw_next) };
here = unsafe { *prev };
}
// link in this qh, unless some earlier pass did that
if qh != unsafe { here.qh } {
unsafe {
(*qh).qh_next = here;
}
if !unsafe { here.qh }.is_null() {
unsafe {
(*(*qh).hw).hw_next = *hw_p;
}
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
(*prev).qh = qh;
}
unsafe {
*hw_p = QH_NEXT!(ehci, unsafe { (*qh).qh_dma });
}
}
i = i.wrapping_add(period);
}
unsafe {
(*qh).qh_state = 1;
}
unsafe {
(*qh).xacterrs = 0;
}
unsafe {
(*qh).unlink_reason = 0;
}
// update per-qh bandwidth for debugfs
let __v7: *mut usb_hcd = ehci_to_hcd(ehci);
let __v8: i32 = unsafe { (*__v7).self_.bandwidth_allocated };
unsafe {
(*__v7).self_.bandwidth_allocated = __v8.wrapping_add(if (*qh).ps.bw_period != 0 { ((*qh).ps.usecs as i32).wrapping_add((*qh).ps.c_usecs as i32) / ((*qh).ps.bw_period as i32) } else { ((*qh).ps.usecs as i32).wrapping_mul(8) });
}
list_add(unsafe { core::ptr::addr_of_mut!((*qh).intr_node) }, unsafe { core::ptr::addr_of_mut!((*ehci).intr_qh_list) });
// maybe enable periodic schedule processing
let __v11: u32 = unsafe { (*ehci).intr_count };
unsafe {
(*ehci).intr_count = __v11.wrapping_add(1);
}
enable_periodic(ehci);
}

fn qh_unlink_periodic(ehci: *mut ehci_hcd, qh: *mut ehci_qh) {
let mut i: u32 = 0;
let mut period: u32 = 0;
// If qh is for a low/full-speed device, simply unlinking it
// could interfere with an ongoing split transaction.  To unlink
// it safely would require setting the QH_INACTIVATE bit and
// waiting at least one frame, as described in EHCI 4.12.2.5.
//
// We won't bother with any of this.  Instead, we assume that the
// only reason for unlinking an interrupt QH while the current URB
// is still active is to dequeue all the URBs (flush the whole
// endpoint queue).
//
// If rebalancing the periodic schedule is ever implemented, this
// approach will no longer be valid.
// high bandwidth, or otherwise part of every microframe
let __v0: u16 = unsafe { (*qh).ps.period };
period = (if __v0 != 0 { __v0 as i32 } else { 1i32 }) as u32;
i = (unsafe { (*qh).ps.phase }) as u32;
while i < unsafe { (*ehci).periodic_size } {
periodic_unlink(ehci, i, qh as *mut ());
i = i.wrapping_add(period);
}
// update per-qh bandwidth for debugfs
let __v4: *mut usb_hcd = ehci_to_hcd(ehci);
let __v5: i32 = unsafe { (*__v4).self_.bandwidth_allocated };
unsafe {
(*__v4).self_.bandwidth_allocated = __v5.wrapping_sub(if (*qh).ps.bw_period != 0 { ((*qh).ps.usecs as i32).wrapping_add((*qh).ps.c_usecs as i32) / ((*qh).ps.bw_period as i32) } else { ((*qh).ps.usecs as i32).wrapping_mul(8) });
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*qh).ps.udev).dev) as *const device, c"unlink qh%d-%04x/%p start %d [%d/%d us]\n".as_ptr() as *const u8, (*qh).ps.period as i32, hc32_to_cpup(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*(*qh).hw).hw_info2) as *const u32) & 65535u32, qh, (*qh).ps.phase as i32, (*qh).ps.usecs as i32, (*qh).ps.c_usecs as i32) };
}
// qh->qh_next still "live" to HC
unsafe {
(*qh).qh_state = 2;
}
unsafe {
(*qh).qh_next.ptr = core::ptr::null_mut::<()>();
}
if (unsafe { (*ehci).qh_scan_next }) == qh {
unsafe {
(*ehci).qh_scan_next = list_entry__40!(unsafe { (*qh).intr_node.next }, ehci_qh);
}
}
list_del(unsafe { core::ptr::addr_of_mut!((*qh).intr_node) });
}

fn cancel_unlink_wait_intr(ehci: *mut ehci_hcd, qh: *mut ehci_qh) {
let __v0: bool = ((unsafe { (*qh).qh_state }) as i32) != 1 || list_empty((unsafe { core::ptr::addr_of_mut!((*qh).unlink_node) }) as *const list_head) != 0;
if __v0 {
return;
}
list_del_init(unsafe { core::ptr::addr_of_mut!((*qh).unlink_node) });
// TODO: disable the event of EHCI_HRTIMER_START_UNLINK_INTR for
// avoiding unnecessary CPU wakeup
}

fn start_unlink_intr(ehci: *mut ehci_hcd, qh: *mut ehci_qh) {
// If the QH isn't linked then there's nothing we can do.
if ((unsafe { (*qh).qh_state }) as i32) != 1 {
return;
}
// if the qh is waiting for unlink, cancel it now
cancel_unlink_wait_intr(ehci, qh);
qh_unlink_periodic(ehci, qh);
// Make sure the unlinks are visible before starting the timer
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
// The EHCI spec doesn't say how long it takes the controller to
// stop accessing an unlinked interrupt QH.  The timer delay is
// 9 uframes; presumably that will be long enough.
unsafe {
(*qh).unlink_cycle = (*ehci).intr_unlink_cycle;
}
// New entries go at the end of the intr_unlink list
list_add_tail(unsafe { core::ptr::addr_of_mut!((*qh).unlink_node) }, unsafe { core::ptr::addr_of_mut!((*ehci).intr_unlink) });
if ((unsafe { (*ehci).__slate_bits_0.__get_intr_unlinking() }) as i32) == 0 {
if ((unsafe { (*ehci).rh_state }) as u32) < 2 {
ehci_handle_intr_unlinks(ehci);
} else {
if (unsafe { (*ehci).intr_unlink.next }) == unsafe { core::ptr::addr_of_mut!((*qh).unlink_node) } {
ehci_enable_event(ehci, 3u32, true);
let __v3: u32 = unsafe { (*ehci).intr_unlink_cycle };
unsafe {
(*ehci).intr_unlink_cycle = __v3.wrapping_add(1);
}
}
}
}
// Avoid recursive calls
}

fn start_unlink_intr_wait(ehci: *mut ehci_hcd, qh: *mut ehci_qh) {
unsafe {
(*qh).unlink_cycle = (*ehci).intr_unlink_wait_cycle;
}
// New entries go at the end of the intr_unlink_wait list
list_add_tail(unsafe { core::ptr::addr_of_mut!((*qh).unlink_node) }, unsafe { core::ptr::addr_of_mut!((*ehci).intr_unlink_wait) });
if ((unsafe { (*ehci).rh_state }) as u32) < 2 {
ehci_handle_start_intr_unlinks(ehci);
} else {
if (unsafe { (*ehci).intr_unlink_wait.next }) == unsafe { core::ptr::addr_of_mut!((*qh).unlink_node) } {
ehci_enable_event(ehci, 6u32, true);
let __v1: u32 = unsafe { (*ehci).intr_unlink_wait_cycle };
unsafe {
(*ehci).intr_unlink_wait_cycle = __v1.wrapping_add(1);
}
}
}
}

fn end_unlink_intr(ehci: *mut ehci_hcd, qh: *mut ehci_qh) {
let hw: *mut ehci_qh_hw = unsafe { (*qh).hw };
let mut rc: i32 = 0;
unsafe {
(*qh).qh_state = 3;
}
unsafe {
(*hw).hw_next = EHCI_LIST_END!(ehci);
}
if list_empty((unsafe { core::ptr::addr_of_mut!((*qh).qtd_list) }) as *const list_head) == 0 {
qh_completions(ehci, qh);
}
// reschedule QH iff another request is queued
if list_empty((unsafe { core::ptr::addr_of_mut!((*qh).qtd_list) }) as *const list_head) == 0 && ((unsafe { (*ehci).rh_state }) as u32) == 2 {
rc = qh_schedule(ehci, qh);
if rc == 0 {
qh_refresh(ehci, qh);
qh_link_periodic(ehci, qh);
} else {
while false {
}
unsafe { _dev_err((*ehci_to_hcd(ehci)).self_.controller as *const device, c"can't reschedule qh %p, err %d\n".as_ptr() as *const u8, qh, rc) };
}
// An error here likely indicates handshake failure
// or no space left in the schedule.  Neither fault
// should happen often ...
//
// FIXME kill the now-dysfunctional queued urbs
}
// maybe turn off periodic schedule
let __v2: u32 = unsafe { (*ehci).intr_count };
unsafe {
(*ehci).intr_count = __v2.wrapping_sub(1);
}
disable_periodic(ehci);
}

fn check_period(ehci: *mut ehci_hcd, frame: u32, mut uframe: u32, uperiod: u32, mut usecs: u32) -> i32 {
// complete split running into next frame?
// given FSTN support, we could sometimes check...
if uframe >= 8 {
return 0;
}
// convert "usecs we need" to "max already claimed"
usecs = unsafe { (*ehci).uframe_periodic_max }.wrapping_sub(usecs);
uframe = uframe.wrapping_add(frame << 3i32);
while uframe < 64 {
if (((unsafe { *((*ehci).bandwidth.as_mut_ptr() as *mut u8).wrapping_offset(uframe as isize) }) as i32) as u32) > usecs {
return 0;
}
uframe = uframe.wrapping_add(uperiod);
}
// success!
1
}

fn check_intr_schedule(ehci: *mut ehci_hcd, frame: u32, uframe: u32, qh: *mut ehci_qh, c_maskp: &mut u32, tt: *mut ehci_tt) -> i32 {
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
let mut mask: u8 = unsafe { core::mem::zeroed() };
let mut retval: i32 = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
retval = 28i32.wrapping_neg();
mask = 0;
if !((*qh).ps.c_usecs != 0 && uframe >= 6) {
// FSTN territory?
if check_period(ehci, frame, uframe, (*qh).ps.bw_uperiod as u32, (*qh).ps.usecs as u32) != 0 {
if (*qh).ps.c_usecs == 0 {
retval = 0;
*(c_maskp as *mut u32) = 0;
} else {
if tt_available(ehci, core::ptr::addr_of_mut!((*qh).ps), tt, frame, uframe) != 0 {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// TODO : this may need FSTN for SSPLIT in uframe 5.
*__slate_slot_4 = uframe.wrapping_add(2);
while *__slate_slot_4 < 8 && *__slate_slot_4 <= uframe.wrapping_add(4) {
if check_period(ehci, frame, *__slate_slot_4, (*qh).ps.bw_uperiod as u32, (*qh).ps.c_usecs as u32) == 0 {
break '__join_0;
} else {
mask = ((mask as i32) | 1i32 << *__slate_slot_4) as u8;
let __v1: u32 = *__slate_slot_4;
*__slate_slot_4 = __v1.wrapping_add(1);
}
}
retval = 0;
*(c_maskp as *mut u32) = mask as u32;
}
}
}
}
}
return retval;
}
}

fn qh_schedule(ehci: *mut ehci_hcd, qh: *mut ehci_qh) -> i32 {
let mut frame: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u32 = core::ptr::addr_of_mut!(frame) as *mut u32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut tt: core::mem::MaybeUninit<*mut ehci_tt> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut ehci_tt = core::ptr::addr_of_mut!(tt) as *mut *mut ehci_tt;
let mut hw: *mut ehci_qh_hw = unsafe { core::mem::zeroed() };
let mut c_mask: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut u32 = core::ptr::addr_of_mut!(c_mask) as *mut u32;
let mut uframe: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u32 = core::ptr::addr_of_mut!(uframe) as *mut u32;
let mut status: i32 = unsafe { core::mem::zeroed() };
unsafe {
status = 0;
let __slate_auto_bytes_1 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
hw = (*qh).hw;
let __slate_auto_bytes_3 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
(*hw).hw_next = EHCI_LIST_END!(ehci);
// reuse the previous schedule slots, if we can
if ((*qh).ps.phase as i32) != 29999 {
return 0;
} else {
'__join_0: {
*__slate_slot_20 = 0;
*__slate_slot_19 = 0;
*__slate_slot_17 = find_tt((*qh).ps.udev);
if IS_ERR(*__slate_slot_17 as *const ()) {
status = PTR_ERR(*__slate_slot_17 as *const ()) as i32;
} else {
'__join_1: {
compute_tt_budget((*ehci).tt_budget.as_mut_ptr() as *mut u8, *__slate_slot_17);
// else scan the schedule to find a group of slots such that all
// uframes have enough periodic bandwidth available.
//
// "normal" case, uframing flexible except with splits
if (*qh).ps.bw_period != 0 {
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_16 = (*qh).ps.bw_period as i32;
while *__slate_slot_16 > 0 {
let __v13: u32 = (*ehci).random_frame;
let __v12: u32 = __v13.wrapping_add(1);
(*ehci).random_frame = __v12;
*__slate_slot_15 = __v12 & (((*qh).ps.bw_period as i32).wrapping_sub(1) as u32);
*__slate_slot_20 = 0;
while *__slate_slot_20 < 8 {
status = check_intr_schedule(ehci, *__slate_slot_15, *__slate_slot_20, qh, &mut (*core::ptr::addr_of_mut!(*__slate_slot_19)), *__slate_slot_17);
if status == 0 {
break '__join_1;
} else {
let __v11: u32 = *__slate_slot_20;
*__slate_slot_20 = __v11.wrapping_add(1);
}
}
let __v9: i32 = *__slate_slot_16;
*__slate_slot_16 = __v9.wrapping_sub(1);
}
// qh->ps.bw_period == 0 means every uframe
} else {
status = check_intr_schedule(ehci, 0u32, 0u32, qh, &mut (*core::ptr::addr_of_mut!(*__slate_slot_19)), *__slate_slot_17);
}
if status != 0 {
break '__join_0;
}
}
(*qh).ps.phase = (if (*qh).ps.period != 0 { (*ehci).random_frame & (((*qh).ps.period as i32).wrapping_sub(1) as u32) } else { 0u32 }) as u16;
(*qh).ps.bw_phase = (((*qh).ps.phase as i32) & ((*qh).ps.bw_period as i32).wrapping_sub(1)) as u8;
(*qh).ps.phase_uf = *__slate_slot_20 as u8;
(*qh).ps.cs_mask = (if (*qh).ps.period != 0 { *__slate_slot_19 << 8i32 | ((1i32 << *__slate_slot_20) as u32) } else { 255u32 }) as u16;
// reset S-frame and (maybe) C-frame masks
let __v7: u32 = cpu_to_hc32(ehci as *const ehci_hcd, 4294901760u32);
(*hw).hw_info2 = (*hw).hw_info2 & __v7;
let __v3: u32 = cpu_to_hc32(ehci as *const ehci_hcd, (*qh).ps.cs_mask as u32);
(*hw).hw_info2 = (*hw).hw_info2 | __v3;
reserve_release_intr_bandwidth(ehci, qh, 1i32);
}
}
return status;
}
}
}

fn intr_submit(ehci: *mut ehci_hcd, urb: *mut urb, qtd_list: *mut list_head, mem_flags: u32) -> i32 {
let mut empty: core::mem::MaybeUninit<list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut list_head = core::ptr::addr_of_mut!(empty) as *mut list_head;
let mut status: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(status) as *mut i32;
let mut qh: core::mem::MaybeUninit<*mut ehci_qh> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut ehci_qh = core::ptr::addr_of_mut!(qh) as *mut *mut ehci_qh;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut epnum: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(epnum) as *mut u32;
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
// get endpoint and transfer/schedule data
*__slate_slot_7 = (*(*urb).ep).desc.bEndpointAddress as u32;
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_6);
if unlikely!(HCD_HW_ACCESSIBLE!(ehci_to_hcd(ehci)) == 0) != 0 {
*__slate_slot_4 = 108i32.wrapping_neg();
} else {
*__slate_slot_4 = usb_hcd_link_urb_to_ep(ehci_to_hcd(ehci), urb);
if unlikely__2!(*__slate_slot_4) == 0 {
'__join_4: {
// get qh and force any scheduling errors
INIT_LIST_HEAD(core::ptr::addr_of_mut!(*__slate_slot_3));
*__slate_slot_5 = qh_append_tds(ehci, urb, core::ptr::addr_of_mut!(*__slate_slot_3), *__slate_slot_7 as i32, core::ptr::addr_of_mut!((*(*urb).ep).hcpriv));
if (*__slate_slot_5).is_null() {
*__slate_slot_4 = 12i32.wrapping_neg();
} else {
if ((*(*__slate_slot_5)).qh_state as i32) == 3 {
*__slate_slot_4 = qh_schedule(ehci, *__slate_slot_5);
if *__slate_slot_4 != 0 {
break '__join_4;
}
}
// then queue the urb's tds to the qh
*__slate_slot_5 = qh_append_tds(ehci, urb, qtd_list, *__slate_slot_7 as i32, core::ptr::addr_of_mut!((*(*urb).ep).hcpriv));
'__join_11: {
loop {
if unlikely!((*__slate_slot_5).is_null()) != 0 {
break '__join_11;
} else {
if true {
break;
}
}
}
// stuff into the periodic schedule
if ((*(*__slate_slot_5)).qh_state as i32) == 3 {
qh_refresh(ehci, *__slate_slot_5);
qh_link_periodic(ehci, *__slate_slot_5);
} else {
// cancel unlink wait for the qh
cancel_unlink_wait_intr(ehci, *__slate_slot_5);
}
// ... update usbfs periodic stats
let __v2: *mut usb_hcd = ehci_to_hcd(ehci);
let __v1: i32 = (*__v2).self_.bandwidth_int_reqs;
(*__v2).self_.bandwidth_int_reqs = __v1.wrapping_add(1);
break '__join_4;
}
core::arch::asm!("674: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 674b - ., 3; .popsection\n# {0}", const 674 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_intr_submit_8, sym __slate_asm_string_intr_submit_9, const 943 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
if unlikely__2!(*__slate_slot_4) != 0 {
usb_hcd_unlink_urb_from_ep(ehci_to_hcd(ehci), urb);
}
}
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_6);
if *__slate_slot_4 != 0 {
qtd_list_free(ehci, urb, qtd_list);
}
return *__slate_slot_4;
}
}

fn scan_intr(ehci: *mut ehci_hcd) {
let mut qh: *mut ehci_qh = unsafe { core::mem::zeroed() };
qh = list_first_entry__28!(unsafe { core::ptr::addr_of_mut!((*ehci).intr_qh_list) });
let __v1: *mut ehci_qh = list_next_entry__16!(qh, intr_node);
unsafe {
(*ehci).qh_scan_next = __v1;
}
while list_entry_is_head!(qh, unsafe { core::ptr::addr_of_mut!((*ehci).intr_qh_list) }, intr_node) == 0 {
// clean any finished work for this qh
if list_empty((unsafe { core::ptr::addr_of_mut!((*qh).qtd_list) }) as *const list_head) == 0 {
let mut temp: i32 = 0;
// Unlinks could happen here; completion reporting
// drops the lock.  That's why ehci->qh_scan_next
// always holds the next qh to scan; if the next qh
// gets unlinked then ehci->qh_scan_next is adjusted
// in qh_unlink_periodic().
temp = qh_completions(ehci, qh) as i32;
if unlikely__2!(temp) != 0 {
start_unlink_intr(ehci, qh);
} else {
if unlikely!(list_empty((unsafe { core::ptr::addr_of_mut!((*qh).qtd_list) }) as *const list_head) != 0 && ((unsafe { (*qh).qh_state }) as i32) == 1) != 0 {
start_unlink_intr_wait(ehci, qh);
}
}
}
qh = unsafe { (*ehci).qh_scan_next };
let __v4: *mut ehci_qh = list_next_entry__16!(unsafe { (*ehci).qh_scan_next }, intr_node);
unsafe {
(*ehci).qh_scan_next = __v4;
}
}
}

fn iso_stream_alloc(mem_flags: u32) -> *mut ehci_iso_stream {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut stream: *mut ehci_iso_stream = unsafe { core::mem::zeroed() };
let mut __v0: *mut ehci_iso_stream;
let __obj_size: u64 = size_mul(core::mem::size_of::<ehci_iso_stream>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, mem_flags, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, mem_flags, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut ehci_iso_stream;
stream = __v0;
if likely!(!stream.is_null()) != 0 {
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*stream).td_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*stream).free_list) });
unsafe {
(*stream).next_uframe = 29999;
}
unsafe {
(*stream).ps.phase = 29999;
}
}
stream
}

fn iso_stream_init(ehci: *mut ehci_hcd, stream: *mut ehci_iso_stream, urb: *mut urb) {
let dev: *mut usb_device = unsafe { (*urb).dev };
let mut buf1: u32 = 0;
let mut epnum: u32 = 0;
let mut maxp: u32 = 0;
let mut is_input: i32 = 0;
let mut tmp: u32 = 0;
// this might be a "high bandwidth" highspeed endpoint,
// as encoded in the ep descriptor's wMaxPacket field
epnum = usb_pipeendpoint!(unsafe { (*urb).pipe });
is_input = if usb_pipein!(unsafe { (*urb).pipe }) != 0 { 128 } else { 0 };
maxp = usb_endpoint_maxp((unsafe { core::ptr::addr_of_mut!((*(*urb).ep).desc) }) as *const usb_endpoint_descriptor) as u32;
buf1 = (if is_input != 0 { 2048i32 } else { 0i32 }) as u32;
// knows about ITD vs SITD
if ((unsafe { (*dev).speed }) as u32) == 3 {
let multi: u32 = usb_endpoint_maxp_mult((unsafe { core::ptr::addr_of_mut!((*(*urb).ep).desc) }) as *const usb_endpoint_descriptor) as u32;
unsafe {
(*stream).highspeed = 1;
}
buf1 |= maxp;
maxp = maxp.wrapping_mul(multi);
unsafe {
(*stream).buf0 = cpu_to_hc32(ehci as *const ehci_hcd, epnum << 8i32 | ((*dev).devnum as u32));
}
unsafe {
(*stream).buf1 = cpu_to_hc32(ehci as *const ehci_hcd, buf1);
}
unsafe {
(*stream).buf2 = cpu_to_hc32(ehci as *const ehci_hcd, multi);
}
// usbfs wants to report the average usecs per frame tied up
// when transfers on this endpoint are scheduled ...
unsafe {
(*stream).ps.usecs = HS_USECS_ISO!(maxp) as u8;
}
// period for bandwidth allocation
tmp = min_t__3!(u32, 64i32, 1i32 << ((unsafe { (*(*urb).ep).desc.bInterval }) as i32).wrapping_sub(1));
// Allow urb->interval to override
unsafe {
(*stream).ps.bw_uperiod = min_t__2!(u32, tmp, unsafe { (*urb).interval }) as u8;
}
unsafe {
(*stream).uperiod = (*urb).interval as u16;
}
unsafe {
(*stream).ps.period = ((*urb).interval >> 3i32) as u16;
}
unsafe {
(*stream).bandwidth = (((*stream).ps.usecs as i32).wrapping_mul(8) / ((*stream).ps.bw_uperiod as i32)) as u32;
}
} else {
let mut addr: u32 = ((unsafe { (*dev).ttport }) << 24i32) as u32;
let mut think_time: i32 = 0;
let mut hs_transfers: i32 = 0;
let __v10: bool = ehci_is_TDI!() == 0 || (unsafe { (*(*dev).tt).hub }) != unsafe { (*ehci_to_hcd(ehci)).self_.root_hub };
if __v10 {
addr |= ((unsafe { (*(*(*dev).tt).hub).devnum }) << 16i32) as u32;
}
addr |= epnum << 8i32;
addr |= (unsafe { (*dev).devnum }) as u32;
unsafe {
(*stream).ps.usecs = HS_USECS_ISO!(maxp) as u8;
}
think_time = (unsafe { (*(*dev).tt).think_time }) as i32;
unsafe {
(*stream).ps.tt_usecs = NS_TO_US__2!((think_time as i64).wrapping_add(usb_calc_bus_time(((*dev).speed as u32) as i32, is_input, 1i32, maxp as i32))) as u16;
}
let mut __v17: u32;
let __UNIQUE_ID_x__679: u32 = 1;
let __UNIQUE_ID_y__680: u32 = maxp.wrapping_add(187) / 188;
loop {
if __types_ok!(__UNIQUE_ID_x__679, __UNIQUE_ID_y__680) == 0 {
unsafe { __compiletime_assert_681() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v17 = if __UNIQUE_ID_x__679 > __UNIQUE_ID_y__680 { __UNIQUE_ID_x__679 } else { __UNIQUE_ID_y__680 };
hs_transfers = __v17 as i32;
if is_input != 0 {
let mut tmp: u32 = 0;
addr |= 2147483648u32;
unsafe {
(*stream).ps.c_usecs = (*stream).ps.usecs;
}
unsafe {
(*stream).ps.usecs = HS_USECS_ISO__2!(1i32) as u8;
}
unsafe {
(*stream).ps.cs_mask = 1;
}
// c-mask as specified in USB 2.0 11.18.4 3.c
tmp = (1i32 << hs_transfers.wrapping_add(2)).wrapping_sub(1) as u32;
unsafe {
(*stream).ps.cs_mask = ((((*stream).ps.cs_mask as i32) as u32) | tmp << 10i32) as u16;
}
} else {
unsafe {
(*stream).ps.cs_mask = smask_out[hs_transfers.wrapping_sub(1) as usize] as u16;
}
}
// period for bandwidth allocation
tmp = min_t__3!(u32, 8i32, 1i32 << ((unsafe { (*(*urb).ep).desc.bInterval }) as i32).wrapping_sub(1));
// Allow urb->interval to override
unsafe {
(*stream).ps.bw_period = min_t__2!(u32, tmp, unsafe { (*urb).interval }) as u8;
}
unsafe {
(*stream).ps.bw_uperiod = (((*stream).ps.bw_period as i32) << 3i32) as u8;
}
unsafe {
(*stream).ps.period = (*urb).interval as u16;
}
unsafe {
(*stream).uperiod = ((*urb).interval << 3i32) as u16;
}
unsafe {
(*stream).bandwidth = (((*stream).ps.usecs as i32).wrapping_add((*stream).ps.c_usecs as i32) / ((*stream).ps.bw_period as i32)) as u32;
}
// stream->splits gets created from cs_mask later
unsafe {
(*stream).address = cpu_to_hc32(ehci as *const ehci_hcd, addr);
}
}
unsafe {
(*stream).ps.udev = dev;
}
unsafe {
(*stream).ps.ep = (*urb).ep;
}
unsafe {
(*stream).bEndpointAddress = ((is_input as u32) | epnum) as u8;
}
unsafe {
(*stream).maxp = maxp as u16;
}
}

static smask_out: [u8; 6] = [1, 3, 7, 15, 31, 63];

fn iso_stream_find(ehci: *mut ehci_hcd, urb: *mut urb) -> *mut ehci_iso_stream {
let mut epnum: u32 = 0;
let mut stream: *mut ehci_iso_stream = unsafe { core::mem::zeroed() };
let mut ep: *mut usb_host_endpoint = unsafe { core::mem::zeroed() };
let mut flags: u64 = 0;
epnum = usb_pipeendpoint!(unsafe { (*urb).pipe });
if usb_pipein!(unsafe { (*urb).pipe }) != 0 {
ep = unsafe { *((*(*urb).dev).ep_in.as_mut_ptr() as *mut *mut usb_host_endpoint).wrapping_offset(epnum as isize) };
} else {
ep = unsafe { *((*(*urb).dev).ep_out.as_mut_ptr() as *mut *mut usb_host_endpoint).wrapping_offset(epnum as isize) };
}
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*ehci).lock) }, flags);
stream = (unsafe { (*ep).hcpriv }) as *mut ehci_iso_stream;
if unlikely!(stream.is_null()) != 0 {
stream = iso_stream_alloc((BIT!(5i32) as u32) | (BIT!(11i32) as u32));
if likely!(!stream.is_null()) != 0 {
unsafe {
(*ep).hcpriv = stream as *mut ();
}
iso_stream_init(ehci, stream, urb);
}
// if dev->ep [epnum] is a QH, hw is set
} else {
if unlikely!(!unsafe { (*stream).hw }.is_null()) != 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.controller as *const device, c"dev %s ep%d%s, not iso??\n".as_ptr() as *const u8, (*(*urb).dev).devpath.as_mut_ptr() as *mut u8, epnum, if usb_pipein!(unsafe { (*urb).pipe }) != 0 { c"in".as_ptr() as *mut u8 } else { c"out".as_ptr() as *mut u8 }) };
}
stream = core::ptr::null_mut::<ehci_iso_stream>();
}
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*ehci).lock) }, flags);
stream
}

fn iso_sched_alloc(packets: u32, mem_flags: u32) -> *mut ehci_iso_sched {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: u64 = unsafe { core::mem::zeroed() };
let mut iso_sched: *mut ehci_iso_sched = unsafe { core::mem::zeroed() };
let mut __v0: *mut ehci_iso_sched;
let __count: u64 = packets as u64;
let __obj_size: u64 = struct_size_t__4!(__count);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, mem_flags, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, mem_flags, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
let __obj_ptr: *mut ehci_iso_sched = __v1 as *mut ehci_iso_sched;
if !__obj_ptr.is_null() {
let mut __v6: u64;
let __v7: u64 = __count;
unsafe {
*core::ptr::addr_of_mut!(*{
__slate_compound_8 = 0u64;
core::ptr::addr_of_mut!(__slate_compound_8)
}) = __v7;
}
__v6 = __v7;
}
__v0 = __obj_ptr;
iso_sched = __v0;
if likely!(!iso_sched.is_null()) != 0 {
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*iso_sched).td_list) });
}
iso_sched
}

fn itd_sched_init(ehci: *mut ehci_hcd, iso_sched: *mut ehci_iso_sched, stream: *mut ehci_iso_stream, urb: *mut urb) {
let mut i: u32 = 0;
let dma: u64 = unsafe { (*urb).transfer_dma };
// how many uframes are needed for these transfers
unsafe {
(*iso_sched).span = unsafe { (*urb).number_of_packets }.wrapping_mul((*stream).uperiod as i32) as u32;
}
// figure out per-uframe itd fields that we'll need later
// when we fit new itds into the schedule.
while i < ((unsafe { (*urb).number_of_packets }) as u32) {
let uframe: *mut ehci_iso_packet = unsafe { core::ptr::addr_of_mut!((*iso_sched).packet) as *mut ehci_iso_packet }.wrapping_offset(i as isize);
let length: u32 = unsafe { (*(core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(i as isize)).length };
let mut buf: u64 = dma.wrapping_add((unsafe { (*(core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(i as isize)).offset }) as u64);
let mut trans: u32 = 2147483648u32;
trans = ((trans as u64) | buf & 4095) as u32;
if unlikely!(i.wrapping_add(1) == ((unsafe { (*urb).number_of_packets }) as u32)) != 0 && (unsafe { (*urb).transfer_flags }) & 128 == 0 {
trans |= 32768u32;
}
trans |= length << 16i32;
unsafe {
(*uframe).transaction = cpu_to_hc32(ehci as *const ehci_hcd, trans);
}
// might need to cross a buffer page within a uframe
unsafe {
(*uframe).bufp = buf & !4095;
}
buf = buf.wrapping_add(length as u64);
if unlikely!((unsafe { (*uframe).bufp }) != buf & !4095) != 0 {
unsafe {
(*uframe).cross = 1;
}
}
i = i.wrapping_add(1);
}
}

fn iso_sched_free(stream: *mut ehci_iso_stream, iso_sched: *mut ehci_iso_sched) {
if iso_sched.is_null() {
return;
}
// caller must hold ehci->lock!
list_splice((unsafe { core::ptr::addr_of_mut!((*iso_sched).td_list) }) as *const list_head, unsafe { core::ptr::addr_of_mut!((*stream).free_list) });
unsafe { kfree(iso_sched as *const ()) };
}

fn itd_urb_transaction(stream: *mut ehci_iso_stream, ehci: *mut ehci_hcd, urb: *mut urb, mem_flags: u32) -> i32 {
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut sched: core::mem::MaybeUninit<*mut ehci_iso_sched> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut ehci_iso_sched = core::ptr::addr_of_mut!(sched) as *mut *mut ehci_iso_sched;
let mut num_itds: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(num_itds) as *mut u32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut itd_dma: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u64 = core::ptr::addr_of_mut!(itd_dma) as *mut u64;
let mut itd: core::mem::MaybeUninit<*mut ehci_itd> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut ehci_itd = core::ptr::addr_of_mut!(itd) as *mut *mut ehci_itd;
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_3 = iso_sched_alloc((*urb).number_of_packets as u32, mem_flags);
if unlikely!((*__slate_slot_3).is_null()) != 0 {
return 12i32.wrapping_neg();
} else {
itd_sched_init(ehci, *__slate_slot_3, stream, urb);
if (*urb).interval < 8 {
*__slate_slot_4 = 1u32.wrapping_add(unsafe { (*(*__slate_slot_3)).span }.wrapping_add(7) / 8);
} else {
*__slate_slot_4 = (*urb).number_of_packets as u32;
}
// allocate/init ITDs
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_2);
*__slate_slot_5 = 0;
while (*__slate_slot_5 as u32) < *__slate_slot_4 {
'__join_2: {
// Use iTDs from the free list, but not iTDs that may
// still be in use by the hardware.
if likely!(list_empty(core::ptr::addr_of_mut!((*stream).free_list) as *const list_head) == 0) != 0 {
*__slate_slot_7 = list_first_entry__29!(core::ptr::addr_of_mut!((*stream).free_list), ehci_itd);
if (*(*__slate_slot_7)).frame != (*ehci).now_frame {
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_7)).itd_list));
*__slate_slot_6 = (*(*__slate_slot_7)).itd_dma;
break '__join_2;
}
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_2);
*__slate_slot_7 = dma_pool_alloc((*ehci).itd_pool, mem_flags, core::ptr::addr_of_mut!(*__slate_slot_6)) as *mut ehci_itd;
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_2);
if (*__slate_slot_7).is_null() {
iso_sched_free(stream, *__slate_slot_3);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_2);
return 12i32.wrapping_neg();
}
}
memset(*__slate_slot_7 as *mut (), 0i32, core::mem::size_of::<ehci_itd>() as u64);
(*(*__slate_slot_7)).itd_dma = *__slate_slot_6;
(*(*__slate_slot_7)).frame = 29999;
list_add(core::ptr::addr_of_mut!((*(*__slate_slot_7)).itd_list), core::ptr::addr_of_mut!((*(*__slate_slot_3)).td_list));
let __v1: i32 = *__slate_slot_5;
*__slate_slot_5 = __v1.wrapping_add(1);
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_2);
// temporarily store schedule info in hcpriv
(*urb).hcpriv = *__slate_slot_3 as *mut ();
(*urb).error_count = 0;
return 0;
}
}
}

fn reserve_release_iso_bandwidth(ehci: *mut ehci_hcd, stream: *mut ehci_iso_stream, sign: i32) {
let mut uframe: u32 = 0;
let mut i: u32 = 0;
let mut j: u32 = 0;
let mut s_mask: u32 = 0;
let mut c_mask: u32 = 0;
let mut m: u32 = 0;
let mut usecs: i32 = (unsafe { (*stream).ps.usecs }) as i32;
let mut c_usecs: i32 = (unsafe { (*stream).ps.c_usecs }) as i32;
let mut tt_usecs: i32 = (unsafe { (*stream).ps.tt_usecs }) as i32;
let mut tt: *mut ehci_tt = unsafe { core::mem::zeroed() };
if ((unsafe { (*stream).ps.phase }) as i32) == 29999 {
return;
}
// Bandwidth wasn't reserved
uframe = (((unsafe { (*stream).ps.bw_phase }) as i32) << 3i32) as u32;
bandwidth_dbg(ehci, sign, c"iso".as_ptr() as *mut u8, unsafe { core::ptr::addr_of_mut!((*stream).ps) });
if sign < 0 {
// Release bandwidth
usecs = usecs.wrapping_neg();
c_usecs = c_usecs.wrapping_neg();
tt_usecs = tt_usecs.wrapping_neg();
}
if (unsafe { (*stream).splits }) == 0 {
// High speed
i = uframe.wrapping_add(((unsafe { (*stream).ps.phase_uf }) as i32) as u32);
while i < 64 {
let __v0: *mut u8 = unsafe { (*ehci).bandwidth.as_mut_ptr() as *mut u8 }.wrapping_offset(i as isize);
unsafe {
*__v0 = (*__v0 as i32).wrapping_add(usecs) as u8;
}
i = i.wrapping_add(((unsafe { (*stream).ps.bw_uperiod }) as i32) as u32);
}
} else {
// Full speed
s_mask = (unsafe { (*stream).ps.cs_mask }) as u32;
c_mask = s_mask >> 8i32;
// NOTE: adjustment needed for frame overflow
i = uframe;
while i < 64 {
j = (unsafe { (*stream).ps.phase_uf }) as u32;
m = (1i32 << j) as u32;
while j < 8 {
if s_mask & m != 0 {
let __v7: *mut u8 = unsafe { (*ehci).bandwidth.as_mut_ptr() as *mut u8 }.wrapping_offset(i.wrapping_add(j) as isize);
unsafe {
*__v7 = (*__v7 as i32).wrapping_add(usecs) as u8;
}
} else {
if c_mask & m != 0 {
let __v10: *mut u8 = unsafe { (*ehci).bandwidth.as_mut_ptr() as *mut u8 }.wrapping_offset(i.wrapping_add(j) as isize);
unsafe {
*__v10 = (*__v10 as i32).wrapping_add(c_usecs) as u8;
}
}
}
j = j.wrapping_add(1);
m <<= 1i32;
}
i = i.wrapping_add(((unsafe { (*stream).ps.bw_uperiod }) as i32) as u32);
}
// find_tt() will not return any error here as we have
// already called find_tt() before calling this function
// and checked for any error return. The previous call
// would have created the data structure.
tt = find_tt(unsafe { (*stream).ps.udev });
if sign > 0 {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*stream).ps.ps_list) }, unsafe { core::ptr::addr_of_mut!((*tt).ps_list) });
} else {
list_del(unsafe { core::ptr::addr_of_mut!((*stream).ps.ps_list) });
}
i = uframe >> 3i32;
while i < 8u32 {
let __v21: *mut u16 = unsafe { (*tt).bandwidth.as_mut_ptr() as *mut u16 }.wrapping_offset(i as isize);
unsafe {
*__v21 = (*__v21 as i32).wrapping_add(tt_usecs) as u16;
}
i = i.wrapping_add(((unsafe { (*stream).ps.bw_period }) as i32) as u32);
}
}
}

fn itd_slot_ok(ehci: *mut ehci_hcd, stream: *mut ehci_iso_stream, mut uframe: u32) -> i32 {
let mut usecs: u32 = 0;
// convert "usecs we need" to "max already claimed"
usecs = unsafe { (*ehci).uframe_periodic_max }.wrapping_sub(((unsafe { (*stream).ps.usecs }) as i32) as u32);
uframe &= ((unsafe { (*stream).ps.bw_uperiod }) as i32).wrapping_sub(1) as u32;
while uframe < 64 {
if (((unsafe { *((*ehci).bandwidth.as_mut_ptr() as *mut u8).wrapping_offset(uframe as isize) }) as i32) as u32) > usecs {
return 0;
}
uframe = uframe.wrapping_add(((unsafe { (*stream).ps.bw_uperiod }) as i32) as u32);
}
1
}

fn sitd_slot_ok(ehci: *mut ehci_hcd, stream: *mut ehci_iso_stream, mut uframe: u32, sched: *mut ehci_iso_sched, tt: *mut ehci_tt) -> i32 {
let mask: u32 = (((unsafe { (*stream).ps.cs_mask }) as i32) << (uframe & 7)) as u32;
let mut tmp: u32 = 0;
let mut frame: u32 = 0;
let mut uf: u32 = 0;
// for OUT, don't wrap SSPLIT into H-microframe 7
if (((unsafe { (*stream).ps.cs_mask }) as i32) & 255) << (uframe & 7) >= 128i32 {
return 0;
}
// for IN, don't wrap CSPLIT into the next frame
if mask & 4294901760u32 != 0 {
return 0;
}
// check bandwidth
uframe &= ((unsafe { (*stream).ps.bw_uperiod }) as i32).wrapping_sub(1) as u32;
frame = uframe >> 3i32;
// The tt's fullspeed bus bandwidth must be available.
// tt_available scheduling guarantees 10+% for control/bulk.
uf = uframe & 7;
if tt_available(ehci, unsafe { core::ptr::addr_of_mut!((*stream).ps) }, tt, frame, uf) == 0 {
return 0;
}
loop {
let mut max_used: u32 = 0;
let mut i: u32 = 0;
// check starts (OUT uses more than one)
uf = uframe;
max_used = unsafe { (*ehci).uframe_periodic_max }.wrapping_sub(((unsafe { (*stream).ps.usecs }) as i32) as u32);
tmp = (((unsafe { (*stream).ps.cs_mask }) as i32) & 255) as u32;
while tmp != 0 {
if (((unsafe { *((*ehci).bandwidth.as_mut_ptr() as *mut u8).wrapping_offset(uf as isize) }) as i32) as u32) > max_used {
return 0;
}
tmp >>= 1i32;
uf = uf.wrapping_add(1);
}
// for IN, check CSPLIT
if (unsafe { (*stream).ps.c_usecs }) != 0 {
max_used = unsafe { (*ehci).uframe_periodic_max }.wrapping_sub(((unsafe { (*stream).ps.c_usecs }) as i32) as u32);
uf = uframe & 4294967288u32;
tmp = 1024u32;
i = (uframe & 7).wrapping_add(2);
while i < 8 {
if (((unsafe { (*stream).ps.cs_mask }) as i32) as u32) & tmp != 0 {
if (((unsafe { *((*ehci).bandwidth.as_mut_ptr() as *mut u8).wrapping_offset(uf.wrapping_add(i) as isize) }) as i32) as u32) > max_used {
return 0;
}
}
i = i.wrapping_add(1);
tmp <<= 1i32;
}
}
uframe = uframe.wrapping_add(((unsafe { (*stream).ps.bw_uperiod }) as i32) as u32);
if uframe >= 64 {
break;
}
}
unsafe {
(*stream).ps.cs_mask = (((*stream).ps.cs_mask as i32) << (uframe & 7)) as u16;
}
unsafe {
(*stream).splits = cpu_to_hc32(ehci as *const ehci_hcd, (*stream).ps.cs_mask as u32);
}
1
}

fn iso_stream_schedule(ehci: *mut ehci_hcd, urb: *mut urb, stream: *mut ehci_iso_stream) -> i32 {
let mut tt: *mut ehci_tt = unsafe { core::mem::zeroed() };
let mut done: i32 = unsafe { core::mem::zeroed() };
// Start a new isochronous stream?
let mut __v16: bool = unsafe { core::mem::zeroed() };
let mut new_stream: bool = unsafe { core::mem::zeroed() };
let mut empty: bool = unsafe { core::mem::zeroed() };
let mut sched: *mut ehci_iso_sched = unsafe { core::mem::zeroed() };
let mut r#mod: u32 = unsafe { core::mem::zeroed() };
let mut status: i32 = unsafe { core::mem::zeroed() };
let mut skip: u32 = unsafe { core::mem::zeroed() };
let mut wrap: u32 = unsafe { core::mem::zeroed() };
let mut now2: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u32 = core::ptr::addr_of_mut!(now2) as *mut u32;
let mut span: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u32 = core::ptr::addr_of_mut!(span) as *mut u32;
let mut period: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut u32 = core::ptr::addr_of_mut!(period) as *mut u32;
let mut start: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut u32 = core::ptr::addr_of_mut!(start) as *mut u32;
let mut next: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut u32 = core::ptr::addr_of_mut!(next) as *mut u32;
let mut base: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut u32 = core::ptr::addr_of_mut!(base) as *mut u32;
let mut now: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut u32 = core::ptr::addr_of_mut!(now) as *mut u32;
unsafe {
let __slate_auto_bytes = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
wrap = 0;
skip = 0;
status = 0;
r#mod = (*ehci).periodic_size << 3i32;
sched = (*urb).hcpriv as *mut ehci_iso_sched;
empty = list_empty(core::ptr::addr_of_mut!((*stream).td_list) as *const list_head) != 0;
new_stream = false;
*__slate_slot_28 = (*stream).uperiod as u32;
*__slate_slot_27 = (*sched).span;
if (*stream).highspeed == 0 {
*__slate_slot_27 <<= 3i32;
}
if empty {
__v16 = !hcd_periodic_completion_in_progress(ehci_to_hcd(ehci), (*urb).ep);
} else {
__v16 = false;
}
'__join_0: {
if __v16 {
// Schedule the endpoint
if ((*stream).ps.phase as i32) == 29999 {
done = 0;
tt = find_tt((*stream).ps.udev);
if IS_ERR(tt as *const ()) {
status = PTR_ERR(tt as *const ()) as i32;
break '__join_0;
} else {
compute_tt_budget((*ehci).tt_budget.as_mut_ptr() as *mut u8, tt);
let __v12: u32 = (*ehci).random_frame;
let __v11: u32 = __v12.wrapping_add(1);
(*ehci).random_frame = __v11;
*__slate_slot_29 = __v11.wrapping_neg() << 3i32 & (*__slate_slot_28).wrapping_sub(1);
// find a uframe slot with enough bandwidth.
// Early uframes are more precious because full-speed
// iso IN transfers can't use late uframes,
// and therefore they should be allocated last.
*__slate_slot_30 = *__slate_slot_29;
let __v10: u32 = *__slate_slot_29;
*__slate_slot_29 = __v10.wrapping_add(*__slate_slot_28);
loop {
let __v8: u32 = *__slate_slot_29;
*__slate_slot_29 = __v8.wrapping_sub(1);
// check schedule: enough space?
if (*stream).highspeed != 0 {
if itd_slot_ok(ehci, stream, *__slate_slot_29) != 0 {
done = 1;
}
} else {
if *__slate_slot_29 % 8 < 6 {
if sitd_slot_ok(ehci, stream, *__slate_slot_29, sched, tt) != 0 {
done = 1;
}
}
}
if !(*__slate_slot_29 > *__slate_slot_30 && done == 0) {
break;
}
}
// no room in the schedule
if done == 0 {
status = 28i32.wrapping_neg();
break '__join_0;
} else {
(*stream).ps.phase = (*__slate_slot_29 >> 3i32 & (((*stream).ps.period as i32).wrapping_sub(1) as u32)) as u16;
(*stream).ps.bw_phase = (((*stream).ps.phase as i32) & ((*stream).ps.bw_period as i32).wrapping_sub(1)) as u8;
(*stream).ps.phase_uf = (*__slate_slot_29 & 7) as u8;
reserve_release_iso_bandwidth(ehci, stream, 1i32);
}
}
} else {
*__slate_slot_29 = (((*stream).ps.phase as i32) << 3i32).wrapping_add((*stream).ps.phase_uf as i32) as u32;
}
// New stream is already scheduled; use the upcoming slot
(*stream).next_uframe = *__slate_slot_29;
new_stream = true;
}
*__slate_slot_32 = ehci_read_frame_index(ehci) & r#mod.wrapping_sub(1);
// Take the isochronous scheduling threshold into account
if (*ehci).i_thresh != 0 {
*__slate_slot_30 = (*__slate_slot_32).wrapping_add((*ehci).i_thresh);
} else {
*__slate_slot_30 = (*__slate_slot_32).wrapping_add(2).wrapping_add(7) & 4294967288u32;
}
// uframe cache
// full frame cache
// If needed, initialize last_iso_frame so that this URB will be seen
if (*ehci).isoc_count == 0 {
(*ehci).last_iso_frame = *__slate_slot_32 >> 3i32;
}
'__join_5: {
'__join_6: {
// Use ehci->last_iso_frame as the base.  There can't be any
// TDs scheduled for earlier than that.
*__slate_slot_31 = (*ehci).last_iso_frame << 3i32;
*__slate_slot_30 = (*__slate_slot_30).wrapping_sub(*__slate_slot_31) & r#mod.wrapping_sub(1);
*__slate_slot_29 = unsafe { (*stream).next_uframe }.wrapping_sub(*__slate_slot_31) & r#mod.wrapping_sub(1);
if unlikely!(new_stream) == 0 {
// Typical case: reuse current schedule, stream may still be active.
// Hopefully there are no gaps from the host falling behind
// (irq delays etc).  If there are, the behavior depends on
// whether URB_ISO_ASAP is set.
*__slate_slot_26 = (*__slate_slot_32).wrapping_sub(*__slate_slot_31) & r#mod.wrapping_sub(1);
// Is the schedule about to wrap around?
if unlikely!(!empty && *__slate_slot_29 < *__slate_slot_28) != 0 {
status = 27i32.wrapping_neg();
break '__join_0;
} else {
// Is the next packet scheduled after the base time?
if likely!(!empty || *__slate_slot_29 <= (*__slate_slot_26).wrapping_add(*__slate_slot_28)) != 0 {
// URB_ISO_ASAP: make sure that start >= next
if unlikely!(*__slate_slot_29 < *__slate_slot_30 && (*urb).transfer_flags & 2 != 0) != 0 {
break '__join_6;
} else {
// Otherwise use start, if it's not in the past
if likely!(*__slate_slot_29 >= *__slate_slot_26) != 0 {
break '__join_5;
}
// Otherwise we got an underrun while the queue was empty
}
} else {
if (*urb).transfer_flags & 2 != 0 {
break '__join_6;
} else {
wrap = r#mod;
let __v6: u32 = *__slate_slot_26;
*__slate_slot_26 = __v6.wrapping_add(r#mod);
}
}
// How many uframes and packets do we need to skip?
skip = (*__slate_slot_26).wrapping_sub(*__slate_slot_29).wrapping_add(*__slate_slot_28).wrapping_sub(1) & (*__slate_slot_28).wrapping_neg();
if skip >= *__slate_slot_27 {
// Entirely in the past?
// Try to keep the last TD intact for scanning later
skip = (*__slate_slot_27).wrapping_sub(*__slate_slot_28);
// Will it come before the current scan position?
if empty {
skip = *__slate_slot_27; // Skip the entire URB
status = 1; // and give it back immediately
iso_sched_free(stream, sched);
sched = core::ptr::null_mut::<ehci_iso_sched>();
(*urb).hcpriv = core::ptr::null_mut::<()>();
}
}
(*urb).error_count = (skip / *__slate_slot_28) as i32;
if !sched.is_null() {
(*sched).first_packet = (*urb).error_count as u32;
break '__join_5;
} else {
break '__join_5;
}
}
}
}
*__slate_slot_29 = (*__slate_slot_30).wrapping_add((*__slate_slot_29).wrapping_sub(*__slate_slot_30) & (*__slate_slot_28).wrapping_sub(1)); // Use the first slot after "next"
}
if unlikely!((*__slate_slot_29).wrapping_add(*__slate_slot_27).wrapping_sub(*__slate_slot_28) >= r#mod.wrapping_add(wrap)) != 0 {
status = 27i32.wrapping_neg();
} else {
// Tried to schedule too far into the future?
let __v4: u32 = *__slate_slot_29;
*__slate_slot_29 = __v4.wrapping_add(*__slate_slot_31);
(*stream).next_uframe = (*__slate_slot_29).wrapping_add(skip) & r#mod.wrapping_sub(1);
// report high speed start in uframes; full speed, in frames
(*urb).start_frame = (*__slate_slot_29 & r#mod.wrapping_sub(1)) as i32;
if (*stream).highspeed == 0 {
(*urb).start_frame = (*urb).start_frame >> 3i32;
}
return status;
}
}
return status;
}
}

fn itd_init(ehci: *mut ehci_hcd, stream: *mut ehci_iso_stream, itd: *mut ehci_itd) {
// it's been recently zeroed
unsafe {
(*itd).hw_next = EHCI_LIST_END!(ehci);
}
unsafe {
*((*itd).hw_bufp.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) = (*stream).buf0;
}
unsafe {
*((*itd).hw_bufp.as_mut_ptr() as *mut u32).wrapping_offset(1 as isize) = (*stream).buf1;
}
unsafe {
*((*itd).hw_bufp.as_mut_ptr() as *mut u32).wrapping_offset(2 as isize) = (*stream).buf2;
}
for i in 0i32..8i32 {
unsafe {
*((*itd).index.as_mut_ptr() as *mut u32).wrapping_offset(i as isize) = 1i32.wrapping_neg() as u32;
}
}
// All other fields are filled when scheduling
}

fn itd_patch(ehci: *mut ehci_hcd, itd: *mut ehci_itd, iso_sched: *mut ehci_iso_sched, index: u32, mut uframe: u16) {
let uf: *mut ehci_iso_packet = unsafe { core::ptr::addr_of_mut!((*iso_sched).packet) as *mut ehci_iso_packet }.wrapping_offset(index as isize);
let mut pg: u32 = unsafe { (*itd).pg };
// BUG_ON(pg == 6 && uf->cross);
uframe = ((uframe as i32) & 7) as u16;
unsafe {
*((*itd).index.as_mut_ptr() as *mut u32).wrapping_offset((uframe as i32) as isize) = index;
}
unsafe {
*((*itd).hw_transaction.as_mut_ptr() as *mut u32).wrapping_offset((uframe as i32) as isize) = (*uf).transaction;
}
let __v2: u32 = cpu_to_hc32(ehci as *const ehci_hcd, pg << 12i32);
let __v3: *mut u32 = unsafe { (*itd).hw_transaction.as_mut_ptr() as *mut u32 }.wrapping_offset((uframe as i32) as isize);
unsafe {
*__v3 = *__v3 | __v2;
}
let __v6: u32 = cpu_to_hc32(ehci as *const ehci_hcd, ((unsafe { (*uf).bufp }) & 4294967295u64) as u32);
let __v7: *mut u32 = unsafe { (*itd).hw_bufp.as_mut_ptr() as *mut u32 }.wrapping_offset(pg as isize);
unsafe {
*__v7 = *__v7 | __v6;
}
let __v10: u32 = cpu_to_hc32(ehci as *const ehci_hcd, ((unsafe { (*uf).bufp }) >> 32i32) as u32);
let __v11: *mut u32 = unsafe { (*itd).hw_bufp_hi.as_mut_ptr() as *mut u32 }.wrapping_offset(pg as isize);
unsafe {
*__v11 = *__v11 | __v10;
}
// iso_frame_desc[].offset must be strictly increasing
if unlikely__6!(unsafe { (*uf).cross }) != 0 {
let bufp: u64 = unsafe { (*uf).bufp }.wrapping_add(4096);
let __v15: u32 = pg.wrapping_add(1);
pg = __v15;
unsafe {
(*itd).pg = __v15;
}
let __v16: u32 = cpu_to_hc32(ehci as *const ehci_hcd, (bufp & 4294967295u64) as u32);
let __v17: *mut u32 = unsafe { (*itd).hw_bufp.as_mut_ptr() as *mut u32 }.wrapping_offset(pg as isize);
unsafe {
*__v17 = *__v17 | __v16;
}
let __v20: u32 = cpu_to_hc32(ehci as *const ehci_hcd, (bufp >> 32i32) as u32);
let __v21: *mut u32 = unsafe { (*itd).hw_bufp_hi.as_mut_ptr() as *mut u32 }.wrapping_offset(pg as isize);
unsafe {
*__v21 = *__v21 | __v20;
}
}
}

fn itd_link(ehci: *mut ehci_hcd, frame: u32, itd: *mut ehci_itd) {
let mut prev: *mut ehci_shadow = unsafe { (*ehci).pshadow }.wrapping_offset(frame as isize);
let mut hw_p: *mut u32 = unsafe { (*ehci).periodic }.wrapping_offset(frame as isize);
let mut here: ehci_shadow = unsafe { *prev };
let mut r#type: u32 = 0;
// skip any iso nodes which might belong to previous microframes
while !unsafe { here.ptr }.is_null() {
r#type = Q_NEXT_TYPE!(ehci, unsafe { *hw_p });
if r#type == cpu_to_hc32(ehci as *const ehci_hcd, 2u32) {
break;
}
prev = periodic_next_shadow(ehci, prev, r#type);
hw_p = shadow_next_periodic(ehci, unsafe { &(*core::ptr::addr_of_mut!(here)) }, r#type);
here = unsafe { *prev };
}
unsafe {
(*itd).itd_next = here;
}
unsafe {
(*itd).hw_next = *hw_p;
}
unsafe {
(*prev).itd = itd;
}
unsafe {
(*itd).frame = frame;
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
*hw_p = cpu_to_hc32(ehci as *const ehci_hcd, ((*itd).itd_dma | 0u64) as u32);
}
}

fn itd_link_urb(ehci: *mut ehci_hcd, urb: *mut urb, r#mod: u32, stream: *mut ehci_iso_stream) {
let mut packet: i32 = 0;
let mut next_uframe: u32 = 0;
let mut uframe: u32 = 0;
let mut frame: u32 = 0;
let iso_sched: *mut ehci_iso_sched = (unsafe { (*urb).hcpriv }) as *mut ehci_iso_sched;
let mut itd: *mut ehci_itd = unsafe { core::mem::zeroed() };
next_uframe = (unsafe { (*stream).next_uframe }) & r#mod.wrapping_sub(1);
if unlikely__2!(list_empty((unsafe { core::ptr::addr_of_mut!((*stream).td_list) }) as *const list_head)) != 0 {
let __v0: *mut usb_hcd = ehci_to_hcd(ehci);
unsafe {
(*__v0).self_.bandwidth_allocated = ((*__v0).self_.bandwidth_allocated as u32).wrapping_add((*stream).bandwidth) as i32;
}
}
if (unsafe { (*ehci_to_hcd(ehci)).self_.bandwidth_isoc_reqs }) == 0 {
if ((unsafe { (*ehci).__slate_bits_1.__get_amd_pll_fix() }) as i32) == 1 {
unsafe { usb_amd_quirk_pll_disable() };
}
}
let __v3: *mut usb_hcd = ehci_to_hcd(ehci);
let __v4: i32 = unsafe { (*__v3).self_.bandwidth_isoc_reqs };
unsafe {
(*__v3).self_.bandwidth_isoc_reqs = __v4.wrapping_add(1);
}
// fill iTDs uframe by uframe
packet = (unsafe { (*iso_sched).first_packet }) as i32;
itd = core::ptr::null_mut::<ehci_itd>();
while packet < unsafe { (*urb).number_of_packets } {
if itd.is_null() {
// ASSERT:  we have all necessary itds
//
// BUG_ON(list_empty(&iso_sched->td_list));
// ASSERT:  no itds for this endpoint in this uframe
itd = list_entry__10!(unsafe { (*iso_sched).td_list.next }, ehci_itd);
list_move_tail(unsafe { core::ptr::addr_of_mut!((*itd).itd_list) }, unsafe { core::ptr::addr_of_mut!((*stream).td_list) });
unsafe {
(*itd).stream = stream;
}
unsafe {
(*itd).urb = urb;
}
itd_init(ehci, stream, itd);
}
uframe = next_uframe & 7;
frame = next_uframe >> 3i32;
itd_patch(ehci, itd, iso_sched, packet as u32, uframe as u16);
next_uframe = next_uframe.wrapping_add(((unsafe { (*stream).uperiod }) as i32) as u32);
next_uframe &= r#mod.wrapping_sub(1);
packet = packet.wrapping_add(1);
// link completed itds into the schedule
if next_uframe >> 3i32 != frame || packet == unsafe { (*urb).number_of_packets } {
itd_link(ehci, frame & unsafe { (*ehci).periodic_size }.wrapping_sub(1), itd);
itd = core::ptr::null_mut::<ehci_itd>();
}
}
unsafe {
(*stream).next_uframe = next_uframe;
}
// don't need that schedule data any more
iso_sched_free(stream, iso_sched);
unsafe {
(*urb).hcpriv = stream as *mut ();
}
let __v15: u32 = unsafe { (*ehci).isoc_count };
unsafe {
(*ehci).isoc_count = __v15.wrapping_add(1);
}
enable_periodic(ehci);
}

fn itd_complete(ehci: *mut ehci_hcd, itd: *mut ehci_itd) -> bool {
let mut retval: bool = unsafe { core::mem::zeroed() };
let mut stream: *mut ehci_iso_stream = unsafe { core::mem::zeroed() };
let mut urb_index: i32 = unsafe { core::mem::zeroed() };
let mut uframe: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u32 = core::ptr::addr_of_mut!(uframe) as *mut u32;
let mut t: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u32 = core::ptr::addr_of_mut!(t) as *mut u32;
let mut desc: core::mem::MaybeUninit<*mut usb_iso_packet_descriptor> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut *mut usb_iso_packet_descriptor = core::ptr::addr_of_mut!(desc) as *mut *mut usb_iso_packet_descriptor;
let mut urb: *mut urb = unsafe { core::mem::zeroed() };
unsafe {
urb = (*itd).urb;
let __slate_auto_bytes = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
urb_index = 1i32.wrapping_neg();
stream = (*itd).stream;
retval = false;
// for each uframe with a packet
*__slate_slot_26 = 0;
while *__slate_slot_26 < 8 {
if likely!(*((*itd).index.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_26 as isize) == (1i32.wrapping_neg() as u32)) == 0 {
urb_index = *((*itd).index.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_26 as isize) as i32;
*__slate_slot_28 = (core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(urb_index as isize);
*__slate_slot_27 = hc32_to_cpup(ehci as *const ehci_hcd, ((*itd).hw_transaction.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_26 as isize) as *const u32);
*((*itd).hw_transaction.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_26 as isize) = 0;
// report transfer status
if unlikely__3!(*__slate_slot_27 & 1879048192u32) != 0 {
let __v15: i32 = (*urb).error_count;
(*urb).error_count = __v15.wrapping_add(1);
if *__slate_slot_27 & 1073741824u32 != 0 {
(*(*__slate_slot_28)).status = if usb_pipein!(unsafe { (*urb).pipe }) != 0 { 63i32.wrapping_neg() } else { 70i32.wrapping_neg() };
} else {
if *__slate_slot_27 & 536870912u32 != 0 {
(*(*__slate_slot_28)).status = 75i32.wrapping_neg();
} else {
(*(*__slate_slot_28)).status = 71i32.wrapping_neg();
}
}
// hc couldn't read
// hc couldn't write
// (t & EHCI_ISOC_XACTERR)
// HC need not update length with this error
if *__slate_slot_27 & 536870912u32 == 0 {
(*(*__slate_slot_28)).actual_length = EHCI_ITD_LENGTH!(*__slate_slot_27);
let __v12: u32 = (*urb).actual_length;
(*urb).actual_length = __v12.wrapping_add((*(*__slate_slot_28)).actual_length);
}
} else {
if likely!(*__slate_slot_27 & 2147483648u32 == 0) != 0 {
(*(*__slate_slot_28)).status = 0;
(*(*__slate_slot_28)).actual_length = EHCI_ITD_LENGTH!(*__slate_slot_27);
let __v18: u32 = (*urb).actual_length;
(*urb).actual_length = __v18.wrapping_add((*(*__slate_slot_28)).actual_length);
} else {
// URB was too late
let __v21: i32 = (*urb).error_count;
(*urb).error_count = __v21.wrapping_add(1);
}
}
}
let __v10: u32 = *__slate_slot_26;
*__slate_slot_26 = __v10.wrapping_add(1);
}
// handle completion now?
if likely!(urb_index.wrapping_add(1) != (*urb).number_of_packets) == 0 {
// ASSERT: it's really the last itd for this urb
// list_for_each_entry (itd, &stream->td_list, itd_list)
// 	 BUG_ON(itd->urb == urb);
// give urb back to the driver; completion often (re)submits
ehci_urb_done(ehci, urb, 0i32);
retval = true;
urb = core::ptr::null_mut::<urb>();
let __v7: u32 = (*ehci).isoc_count;
(*ehci).isoc_count = __v7.wrapping_sub(1);
disable_periodic(ehci);
let __v5: *mut usb_hcd = ehci_to_hcd(ehci);
let __v4: i32 = (*__v5).self_.bandwidth_isoc_reqs;
(*__v5).self_.bandwidth_isoc_reqs = __v4.wrapping_sub(1);
if (*ehci_to_hcd(ehci)).self_.bandwidth_isoc_reqs == 0 {
if ((*ehci).__slate_bits_1.__get_amd_pll_fix() as i32) == 1 {
usb_amd_quirk_pll_enable();
}
}
if unlikely__2!(list_is_singular(core::ptr::addr_of_mut!((*stream).td_list) as *const list_head)) != 0 {
let __v2: *mut usb_hcd = ehci_to_hcd(ehci);
(*__v2).self_.bandwidth_allocated = ((*__v2).self_.bandwidth_allocated as u32).wrapping_sub((*stream).bandwidth) as i32;
}
}
(*itd).urb = core::ptr::null_mut::<urb>();
// Add to the end of the free list for later reuse
list_move_tail(core::ptr::addr_of_mut!((*itd).itd_list), core::ptr::addr_of_mut!((*stream).free_list));
// Recycle the iTDs when the pipeline is empty (ep no longer in use)
if list_empty(core::ptr::addr_of_mut!((*stream).td_list) as *const list_head) != 0 {
list_splice_tail_init(core::ptr::addr_of_mut!((*stream).free_list), core::ptr::addr_of_mut!((*ehci).cached_itd_list));
start_free_itds(ehci);
}
return retval;
}
}

fn itd_submit(ehci: *mut ehci_hcd, urb: *mut urb, mem_flags: u32) -> i32 {
let mut stream: core::mem::MaybeUninit<*mut ehci_iso_stream> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut ehci_iso_stream = core::ptr::addr_of_mut!(stream) as *mut *mut ehci_iso_stream;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut status: i32 = unsafe { core::mem::zeroed() };
unsafe {
status = 22i32.wrapping_neg();
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// Get iso_stream head
*__slate_slot_0 = iso_stream_find(ehci, urb);
if unlikely!((*__slate_slot_0).is_null()) != 0 {
return 12i32.wrapping_neg();
} else {
if unlikely!((*urb).interval != ((*(*__slate_slot_0)).uperiod as i32)) == 0 {
// allocate ITDs w/o locking anything
status = itd_urb_transaction(*__slate_slot_0, ehci, urb, mem_flags);
if unlikely!(status < 0) == 0 {
// schedule ... need to lock
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_1);
if unlikely!(HCD_HW_ACCESSIBLE!(ehci_to_hcd(ehci)) == 0) != 0 {
status = 108i32.wrapping_neg();
} else {
status = usb_hcd_link_urb_to_ep(ehci_to_hcd(ehci), urb);
if unlikely__2!(status) == 0 {
status = iso_stream_schedule(ehci, urb, *__slate_slot_0);
if likely!(status == 0) != 0 {
itd_link_urb(ehci, urb, (*ehci).periodic_size << 3i32, *__slate_slot_0);
} else {
if status > 0 {
status = 0;
ehci_urb_done(ehci, urb, 0i32);
} else {
usb_hcd_unlink_urb_from_ep(ehci_to_hcd(ehci), urb);
}
}
}
}
if status < 0 {
iso_sched_free(*__slate_slot_0, (*urb).hcpriv as *mut ehci_iso_sched);
(*urb).hcpriv = core::ptr::null_mut::<()>();
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_1);
}
}
return status;
}
}
}

fn sitd_sched_init(ehci: *mut ehci_hcd, iso_sched: *mut ehci_iso_sched, stream: *mut ehci_iso_stream, urb: *mut urb) {
let mut i: u32 = 0;
let dma: u64 = unsafe { (*urb).transfer_dma };
// how many frames are needed for these transfers
unsafe {
(*iso_sched).span = unsafe { (*urb).number_of_packets }.wrapping_mul((*stream).ps.period as i32) as u32;
}
// figure out per-frame sitd fields that we'll need later
// when we fit new sitds into the schedule.
while i < ((unsafe { (*urb).number_of_packets }) as u32) {
let packet: *mut ehci_iso_packet = unsafe { core::ptr::addr_of_mut!((*iso_sched).packet) as *mut ehci_iso_packet }.wrapping_offset(i as isize);
let mut length: u32 = (unsafe { (*(core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(i as isize)).length }) & 1023;
let buf: u64 = dma.wrapping_add((unsafe { (*(core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(i as isize)).offset }) as u64);
let mut trans: u32 = 128u32;
if i.wrapping_add(1) == ((unsafe { (*urb).number_of_packets }) as u32) && (unsafe { (*urb).transfer_flags }) & 128 == 0 {
trans |= 2147483648u32;
}
trans |= length << 16i32;
unsafe {
(*packet).transaction = cpu_to_hc32(ehci as *const ehci_hcd, trans);
}
// might need to cross a buffer page within a td
unsafe {
(*packet).bufp = buf;
}
unsafe {
(*packet).buf1 = (buf.wrapping_add(length as u64) & 18446744073709547520u64) as u32;
}
if ((unsafe { (*packet).buf1 }) as u64) != buf & !4095 {
unsafe {
(*packet).cross = 1;
}
}
// OUT uses multiple start-splits
if ((unsafe { (*stream).bEndpointAddress }) as i32) & 128 == 0 {
length = length.wrapping_add(187) / 188;
if length > 1 {
length |= 8u32;
}
// BEGIN vs ALL
unsafe {
(*packet).buf1 = (*packet).buf1 | length;
}
}
i = i.wrapping_add(1);
}
}

fn sitd_urb_transaction(stream: *mut ehci_iso_stream, ehci: *mut ehci_hcd, urb: *mut urb, mem_flags: u32) -> i32 {
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut iso_sched: core::mem::MaybeUninit<*mut ehci_iso_sched> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut ehci_iso_sched = core::ptr::addr_of_mut!(iso_sched) as *mut *mut ehci_iso_sched;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut sitd_dma: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u64 = core::ptr::addr_of_mut!(sitd_dma) as *mut u64;
let mut sitd: core::mem::MaybeUninit<*mut ehci_sitd> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut ehci_sitd = core::ptr::addr_of_mut!(sitd) as *mut *mut ehci_sitd;
unsafe {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_3 = iso_sched_alloc((*urb).number_of_packets as u32, mem_flags);
if (*__slate_slot_3).is_null() {
return 12i32.wrapping_neg();
} else {
sitd_sched_init(ehci, *__slate_slot_3, stream, urb);
// allocate/init sITDs
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_2);
*__slate_slot_4 = 0;
while *__slate_slot_4 < (*urb).number_of_packets {
'__join_2: {
// NOTE:  for now, we don't try to handle wraparound cases
// for IN (using sitd->hw_backpointer, like a FSTN), which
// means we never need two sitds for full speed packets.
// Use siTDs from the free list, but not siTDs that may
// still be in use by the hardware.
if likely!(list_empty(core::ptr::addr_of_mut!((*stream).free_list) as *const list_head) == 0) != 0 {
*__slate_slot_6 = list_first_entry__23!(core::ptr::addr_of_mut!((*stream).free_list), ehci_sitd);
if (*(*__slate_slot_6)).frame != (*ehci).now_frame {
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_6)).sitd_list));
*__slate_slot_5 = (*(*__slate_slot_6)).sitd_dma;
break '__join_2;
}
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_2);
*__slate_slot_6 = dma_pool_alloc((*ehci).sitd_pool, mem_flags, core::ptr::addr_of_mut!(*__slate_slot_5)) as *mut ehci_sitd;
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_2);
if (*__slate_slot_6).is_null() {
iso_sched_free(stream, *__slate_slot_3);
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_2);
return 12i32.wrapping_neg();
}
}
memset(*__slate_slot_6 as *mut (), 0i32, core::mem::size_of::<ehci_sitd>() as u64);
(*(*__slate_slot_6)).sitd_dma = *__slate_slot_5;
(*(*__slate_slot_6)).frame = 29999;
list_add(core::ptr::addr_of_mut!((*(*__slate_slot_6)).sitd_list), core::ptr::addr_of_mut!((*(*__slate_slot_3)).td_list));
let __v1: i32 = *__slate_slot_4;
*__slate_slot_4 = __v1.wrapping_add(1);
}
// temporarily store schedule info in hcpriv
(*urb).hcpriv = *__slate_slot_3 as *mut ();
(*urb).error_count = 0;
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_2);
return 0;
}
}
}

fn sitd_patch(ehci: *mut ehci_hcd, stream: *mut ehci_iso_stream, sitd: *mut ehci_sitd, iso_sched: *mut ehci_iso_sched, index: u32) {
let uf: *mut ehci_iso_packet = unsafe { core::ptr::addr_of_mut!((*iso_sched).packet) as *mut ehci_iso_packet }.wrapping_offset(index as isize);
let mut bufp: u64 = 0;
unsafe {
(*sitd).hw_next = EHCI_LIST_END!(ehci);
}
unsafe {
(*sitd).hw_fullspeed_ep = (*stream).address;
}
unsafe {
(*sitd).hw_uframe = (*stream).splits;
}
unsafe {
(*sitd).hw_results = (*uf).transaction;
}
unsafe {
(*sitd).hw_backpointer = EHCI_LIST_END!(ehci);
}
bufp = unsafe { (*uf).bufp };
unsafe {
*((*sitd).hw_buf.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) = cpu_to_hc32(ehci as *const ehci_hcd, bufp as u32);
}
unsafe {
*((*sitd).hw_buf_hi.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) = cpu_to_hc32(ehci as *const ehci_hcd, (bufp >> 32i32) as u32);
}
unsafe {
*((*sitd).hw_buf.as_mut_ptr() as *mut u32).wrapping_offset(1 as isize) = cpu_to_hc32(ehci as *const ehci_hcd, (*uf).buf1);
}
if (unsafe { (*uf).cross }) != 0 {
bufp = bufp.wrapping_add(4096);
}
unsafe {
*((*sitd).hw_buf_hi.as_mut_ptr() as *mut u32).wrapping_offset(1 as isize) = cpu_to_hc32(ehci as *const ehci_hcd, (bufp >> 32i32) as u32);
}
unsafe {
(*sitd).index = index;
}
}

fn sitd_link(ehci: *mut ehci_hcd, frame: u32, sitd: *mut ehci_sitd) {
// note: sitd ordering could matter (CSPLIT then SSPLIT)
unsafe {
(*sitd).sitd_next = *unsafe { (*ehci).pshadow }.wrapping_offset(frame as isize);
}
unsafe {
(*sitd).hw_next = *unsafe { (*ehci).periodic }.wrapping_offset(frame as isize);
}
unsafe {
(*unsafe { (*ehci).pshadow }.wrapping_offset(frame as isize)).sitd = sitd;
}
unsafe {
(*sitd).frame = frame;
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("sfence", options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
*unsafe { (*ehci).periodic }.wrapping_offset(frame as isize) = cpu_to_hc32(ehci as *const ehci_hcd, ((*sitd).sitd_dma | 4u64) as u32);
}
}

fn sitd_link_urb(ehci: *mut ehci_hcd, urb: *mut urb, r#mod: u32, stream: *mut ehci_iso_stream) {
let mut packet: i32 = 0;
let mut next_uframe: u32 = 0;
let sched: *mut ehci_iso_sched = (unsafe { (*urb).hcpriv }) as *mut ehci_iso_sched;
let mut sitd: *mut ehci_sitd = unsafe { core::mem::zeroed() };
next_uframe = unsafe { (*stream).next_uframe };
if list_empty((unsafe { core::ptr::addr_of_mut!((*stream).td_list) }) as *const list_head) != 0 {
let __v0: *mut usb_hcd = ehci_to_hcd(ehci);
unsafe {
(*__v0).self_.bandwidth_allocated = ((*__v0).self_.bandwidth_allocated as u32).wrapping_add((*stream).bandwidth) as i32;
}
}
// usbfs ignores TT bandwidth
if (unsafe { (*ehci_to_hcd(ehci)).self_.bandwidth_isoc_reqs }) == 0 {
if ((unsafe { (*ehci).__slate_bits_1.__get_amd_pll_fix() }) as i32) == 1 {
unsafe { usb_amd_quirk_pll_disable() };
}
}
let __v3: *mut usb_hcd = ehci_to_hcd(ehci);
let __v4: i32 = unsafe { (*__v3).self_.bandwidth_isoc_reqs };
unsafe {
(*__v3).self_.bandwidth_isoc_reqs = __v4.wrapping_add(1);
}
// fill sITDs frame by frame
packet = (unsafe { (*sched).first_packet }) as i32;
sitd = core::ptr::null_mut::<ehci_sitd>();
while packet < unsafe { (*urb).number_of_packets } {
// ASSERT:  we have all necessary sitds
loop {
if unlikely__2!(list_empty((unsafe { core::ptr::addr_of_mut!((*sched).td_list) }) as *const list_head)) != 0 {
loop {
unsafe {
core::arch::asm!("686: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 686b - ., 3; .popsection\n# {0}", const 686 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_sitd_link_urb_6, sym __slate_asm_string_sitd_link_urb_7, const 2176 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
// ASSERT:  no itds for this endpoint in this frame
sitd = list_entry__11!(unsafe { (*sched).td_list.next }, ehci_sitd);
list_move_tail(unsafe { core::ptr::addr_of_mut!((*sitd).sitd_list) }, unsafe { core::ptr::addr_of_mut!((*stream).td_list) });
unsafe {
(*sitd).stream = stream;
}
unsafe {
(*sitd).urb = urb;
}
sitd_patch(ehci, stream, sitd, sched, packet as u32);
sitd_link(ehci, next_uframe >> 3i32 & unsafe { (*ehci).periodic_size }.wrapping_sub(1), sitd);
next_uframe = next_uframe.wrapping_add(((unsafe { (*stream).uperiod }) as i32) as u32);
packet = packet.wrapping_add(1);
}
unsafe {
(*stream).next_uframe = next_uframe & r#mod.wrapping_sub(1);
}
// don't need that schedule data any more
iso_sched_free(stream, sched);
unsafe {
(*urb).hcpriv = stream as *mut ();
}
let __v18: u32 = unsafe { (*ehci).isoc_count };
unsafe {
(*ehci).isoc_count = __v18.wrapping_add(1);
}
enable_periodic(ehci);
}

fn sitd_complete(ehci: *mut ehci_hcd, sitd: *mut ehci_sitd) -> bool {
let mut retval: bool = unsafe { core::mem::zeroed() };
let mut stream: *mut ehci_iso_stream = unsafe { core::mem::zeroed() };
let mut urb_index: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut i32 = core::ptr::addr_of_mut!(urb_index) as *mut i32;
let mut t: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u32 = core::ptr::addr_of_mut!(t) as *mut u32;
let mut desc: core::mem::MaybeUninit<*mut usb_iso_packet_descriptor> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut usb_iso_packet_descriptor = core::ptr::addr_of_mut!(desc) as *mut *mut usb_iso_packet_descriptor;
let mut urb: *mut urb = unsafe { core::mem::zeroed() };
unsafe {
urb = (*sitd).urb;
let __slate_auto_bytes = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
stream = (*sitd).stream;
retval = false;
*__slate_slot_20 = (*sitd).index as i32;
*__slate_slot_22 = (core::ptr::addr_of_mut!((*urb).iso_frame_desc) as *mut usb_iso_packet_descriptor).wrapping_offset(*__slate_slot_20 as isize);
*__slate_slot_21 = hc32_to_cpup(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*sitd).hw_results) as *const u32);
// report transfer status
if unlikely__3!(*__slate_slot_21 & 124u32) != 0 {
let __v10: i32 = (*urb).error_count;
(*urb).error_count = __v10.wrapping_add(1);
if *__slate_slot_21 & 32u32 != 0 {
(*(*__slate_slot_22)).status = if usb_pipein!(unsafe { (*urb).pipe }) != 0 { 63i32.wrapping_neg() } else { 70i32.wrapping_neg() };
} else {
if *__slate_slot_21 & 16u32 != 0 {
(*(*__slate_slot_22)).status = 75i32.wrapping_neg();
} else {
(*(*__slate_slot_22)).status = 71i32.wrapping_neg();
}
}
// hc couldn't read
// hc couldn't write
// XACT, MMF, etc
} else {
if unlikely__3!(*__slate_slot_21 & 128u32) != 0 {
// URB was too late
let __v13: i32 = (*urb).error_count;
(*urb).error_count = __v13.wrapping_add(1);
} else {
(*(*__slate_slot_22)).status = 0;
(*(*__slate_slot_22)).actual_length = unsafe { (*(*__slate_slot_22)).length }.wrapping_sub(SITD_LENGTH!(*__slate_slot_21));
let __v16: u32 = (*urb).actual_length;
(*urb).actual_length = __v16.wrapping_add((*(*__slate_slot_22)).actual_length);
}
}
// handle completion now?
if (*__slate_slot_20).wrapping_add(1) == (*urb).number_of_packets {
// ASSERT: it's really the last sitd for this urb
// list_for_each_entry (sitd, &stream->td_list, sitd_list)
// 	 BUG_ON(sitd->urb == urb);
// give urb back to the driver; completion often (re)submits
ehci_urb_done(ehci, urb, 0i32);
retval = true;
urb = core::ptr::null_mut::<urb>();
let __v7: u32 = (*ehci).isoc_count;
(*ehci).isoc_count = __v7.wrapping_sub(1);
disable_periodic(ehci);
let __v5: *mut usb_hcd = ehci_to_hcd(ehci);
let __v4: i32 = (*__v5).self_.bandwidth_isoc_reqs;
(*__v5).self_.bandwidth_isoc_reqs = __v4.wrapping_sub(1);
if (*ehci_to_hcd(ehci)).self_.bandwidth_isoc_reqs == 0 {
if ((*ehci).__slate_bits_1.__get_amd_pll_fix() as i32) == 1 {
usb_amd_quirk_pll_enable();
}
}
if list_is_singular(core::ptr::addr_of_mut!((*stream).td_list) as *const list_head) != 0 {
let __v2: *mut usb_hcd = ehci_to_hcd(ehci);
(*__v2).self_.bandwidth_allocated = ((*__v2).self_.bandwidth_allocated as u32).wrapping_sub((*stream).bandwidth) as i32;
}
}
(*sitd).urb = core::ptr::null_mut::<urb>();
// Add to the end of the free list for later reuse
list_move_tail(core::ptr::addr_of_mut!((*sitd).sitd_list), core::ptr::addr_of_mut!((*stream).free_list));
// Recycle the siTDs when the pipeline is empty (ep no longer in use)
if list_empty(core::ptr::addr_of_mut!((*stream).td_list) as *const list_head) != 0 {
list_splice_tail_init(core::ptr::addr_of_mut!((*stream).free_list), core::ptr::addr_of_mut!((*ehci).cached_sitd_list));
start_free_itds(ehci);
}
return retval;
}
}

fn sitd_submit(ehci: *mut ehci_hcd, urb: *mut urb, mem_flags: u32) -> i32 {
let mut stream: core::mem::MaybeUninit<*mut ehci_iso_stream> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut ehci_iso_stream = core::ptr::addr_of_mut!(stream) as *mut *mut ehci_iso_stream;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut status: i32 = unsafe { core::mem::zeroed() };
unsafe {
status = 22i32.wrapping_neg();
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// Get iso_stream head
*__slate_slot_0 = iso_stream_find(ehci, urb);
if (*__slate_slot_0).is_null() {
return 12i32.wrapping_neg();
} else {
if (*urb).interval == ((*(*__slate_slot_0)).ps.period as i32) {
// allocate SITDs
status = sitd_urb_transaction(*__slate_slot_0, ehci, urb, mem_flags);
if status >= 0 {
// schedule ... need to lock
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_1);
if unlikely!(HCD_HW_ACCESSIBLE!(ehci_to_hcd(ehci)) == 0) != 0 {
status = 108i32.wrapping_neg();
} else {
status = usb_hcd_link_urb_to_ep(ehci_to_hcd(ehci), urb);
if unlikely__2!(status) == 0 {
status = iso_stream_schedule(ehci, urb, *__slate_slot_0);
if likely!(status == 0) != 0 {
sitd_link_urb(ehci, urb, (*ehci).periodic_size << 3i32, *__slate_slot_0);
} else {
if status > 0 {
status = 0;
ehci_urb_done(ehci, urb, 0i32);
} else {
usb_hcd_unlink_urb_from_ep(ehci_to_hcd(ehci), urb);
}
}
}
}
if status < 0 {
iso_sched_free(*__slate_slot_0, (*urb).hcpriv as *mut ehci_iso_sched);
(*urb).hcpriv = core::ptr::null_mut::<()>();
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_1);
}
}
return status;
}
}
}

fn scan_isoc(ehci: *mut ehci_hcd) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut hw_p: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut u32 = core::ptr::addr_of_mut!(hw_p) as *mut *mut u32;
let mut r#type: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(r#type) as *mut u32;
let mut q_p: core::mem::MaybeUninit<*mut ehci_shadow> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut ehci_shadow = core::ptr::addr_of_mut!(q_p) as *mut *mut ehci_shadow;
let mut q: core::mem::MaybeUninit<ehci_shadow> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut ehci_shadow = core::ptr::addr_of_mut!(q) as *mut ehci_shadow;
let mut live: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut bool = core::ptr::addr_of_mut!(live) as *mut bool;
let mut modified: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut bool = core::ptr::addr_of_mut!(modified) as *mut bool;
let mut fmask: u32 = unsafe { core::mem::zeroed() };
let mut frame: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u32 = core::ptr::addr_of_mut!(frame) as *mut u32;
let mut now_frame: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u32 = core::ptr::addr_of_mut!(now_frame) as *mut u32;
let mut uf: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u32 = core::ptr::addr_of_mut!(uf) as *mut u32;
unsafe {
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
fmask = unsafe { (*ehci).periodic_size }.wrapping_sub(1);
let __slate_auto_bytes_3 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_4 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_5 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// When running, scan from last scan point up to "now"
// else clean up by scanning everything that's left.
// Touches as few pages as possible:  cache-friendly.
if ((*ehci).rh_state as u32) >= 2 {
*__slate_slot_14 = ehci_read_frame_index(ehci);
*__slate_slot_13 = *__slate_slot_14 >> 3i32 & fmask;
*__slate_slot_9 = true;
} else {
*__slate_slot_13 = unsafe { (*ehci).last_iso_frame }.wrapping_sub(1) & fmask;
*__slate_slot_9 = false;
}
(*ehci).now_frame = *__slate_slot_13;
*__slate_slot_12 = (*ehci).last_iso_frame;
'__loop_39: loop {
*__slate_slot_7 = unsafe { (*ehci).pshadow }.wrapping_offset(*__slate_slot_12 as isize); // Scan each element in frame's queue for completions
*__slate_slot_5 = unsafe { (*ehci).periodic }.wrapping_offset(*__slate_slot_12 as isize);
(*__slate_slot_8).ptr = (*(*__slate_slot_7)).ptr;
*__slate_slot_6 = Q_NEXT_TYPE!(ehci, unsafe { *(*__slate_slot_5) });
*__slate_slot_10 = false;
while !unsafe { (*__slate_slot_8).ptr }.is_null() {
'__join_4: {
let __t0: u32 = hc32_to_cpu(ehci as *const ehci_hcd, *__slate_slot_6);
if __t0 == 0 {
if *__slate_slot_12 == *__slate_slot_13 && *__slate_slot_9 {
loop {
kcsan_rmb!();
core::arch::asm!("lfence", options(nostack, att_syntax));
if true {
break;
}
}
*__slate_slot_14 = 0;
while *__slate_slot_14 < 8 {
if *((*(*__slate_slot_8).itd).hw_transaction.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_14 as isize) & ITD_ACTIVE!(ehci) != 0 {
break;
} else {
let __v4: u32 = *__slate_slot_14;
*__slate_slot_14 = __v4.wrapping_add(1);
}
}
if *__slate_slot_14 < 8 {
*__slate_slot_7 = core::ptr::addr_of_mut!((*(*__slate_slot_8).itd).itd_next);
*__slate_slot_5 = core::ptr::addr_of_mut!((*(*__slate_slot_8).itd).hw_next);
*__slate_slot_6 = Q_NEXT_TYPE!(ehci, unsafe { (*(*__slate_slot_8).itd).hw_next });
*__slate_slot_8 = *(*__slate_slot_7);
break '__join_4;
}
}
// If this ITD is still active, leave it for
// later processing ... check the next entry.
// No need to check for activity unless the
// frame is current.
// Take finished ITDs out of the schedule
// and process them:  recycle, maybe report
// URB completion.  HC won't cache the
// pointer for much longer, if at all.
*(*__slate_slot_7) = (*(*__slate_slot_8).itd).itd_next;
if ((*ehci).__slate_bits_1.__get_use_dummy_qh() as i32) == 0 {
__v2 = true;
} else {
__v2 = (*(*__slate_slot_8).itd).hw_next != EHCI_LIST_END!(ehci);
}
if __v2 {
*(*__slate_slot_5) = (*(*__slate_slot_8).itd).hw_next;
} else {
*(*__slate_slot_5) = cpu_to_hc32(ehci as *const ehci_hcd, (*(*ehci).dummy).qh_dma as u32);
}
*__slate_slot_6 = Q_NEXT_TYPE!(ehci, unsafe { (*(*__slate_slot_8).itd).hw_next });
loop {
kcsan_wmb!();
core::arch::asm!("sfence", options(nostack, att_syntax));
if true {
break;
}
}
*__slate_slot_10 = itd_complete(ehci, (*__slate_slot_8).itd);
*__slate_slot_8 = *(*__slate_slot_7);
break '__join_4;
} else {
if __t0 == 4 {
if (*__slate_slot_12 == *__slate_slot_13 || (*__slate_slot_12).wrapping_add(1) & fmask == *__slate_slot_13) && *__slate_slot_9 {
__v1 = (*(*__slate_slot_8).sitd).hw_results & SITD_ACTIVE!(ehci) != 0;
} else {
__v1 = false;
}
if __v1 {
*__slate_slot_7 = core::ptr::addr_of_mut!((*(*__slate_slot_8).sitd).sitd_next);
*__slate_slot_5 = core::ptr::addr_of_mut!((*(*__slate_slot_8).sitd).hw_next);
*__slate_slot_6 = Q_NEXT_TYPE!(ehci, unsafe { (*(*__slate_slot_8).sitd).hw_next });
*__slate_slot_8 = *(*__slate_slot_7);
break '__join_4;
} else {
// If this SITD is still active, leave it for
// later processing ... check the next entry.
// No need to check for activity unless the
// frame is current.
// Take finished SITDs out of the schedule
// and process them:  recycle, maybe report
// URB completion.
*(*__slate_slot_7) = (*(*__slate_slot_8).sitd).sitd_next;
if ((*ehci).__slate_bits_1.__get_use_dummy_qh() as i32) == 0 {
__v0 = true;
} else {
__v0 = (*(*__slate_slot_8).sitd).hw_next != EHCI_LIST_END!(ehci);
}
if __v0 {
*(*__slate_slot_5) = (*(*__slate_slot_8).sitd).hw_next;
} else {
*(*__slate_slot_5) = cpu_to_hc32(ehci as *const ehci_hcd, (*(*ehci).dummy).qh_dma as u32);
}
*__slate_slot_6 = Q_NEXT_TYPE!(ehci, unsafe { (*(*__slate_slot_8).sitd).hw_next });
loop {
kcsan_wmb!();
core::arch::asm!("sfence", options(nostack, att_syntax));
if true {
break;
}
}
*__slate_slot_10 = sitd_complete(ehci, (*__slate_slot_8).sitd);
*__slate_slot_8 = *(*__slate_slot_7);
break '__join_4;
}
}
// BUG();
}
(*__slate_slot_8).ptr = core::ptr::null_mut::<()>();
// End of the iTDs and siTDs
}
// Assume completion callbacks modify the queue
if unlikely!(*__slate_slot_10 && (*ehci).isoc_count > 0) != 0 {
continue '__loop_39;
}
}
// Stop when we have reached the current frame
if *__slate_slot_12 == *__slate_slot_13 {
break;
} else {
// The last frame may still have active siTDs
(*ehci).last_iso_frame = *__slate_slot_12;
*__slate_slot_12 = (*__slate_slot_12).wrapping_add(1) & fmask;
}
}
return;
}
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.companion_show")]
extern "C" fn companion_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let mut ehci: *mut ehci_hcd = unsafe { core::mem::zeroed() };
let mut nports: i32 = 0;
let mut len: i32 = 0;
ehci = hcd_to_ehci(dev_get_drvdata(dev as *const device) as *mut usb_hcd);
nports = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
for index in 0..nports {
if test_bit!(index, unsafe { core::ptr::addr_of_mut!((*ehci).companion_ports) }) != 0 {
let __v1: i32 = unsafe { sysfs_emit_at(buf, len, c"%d\n".as_ptr() as *const u8, index.wrapping_add(1)) };
len = len.wrapping_add(__v1);
}
}
len as i64
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.companion_store")]
extern "C" fn companion_store(dev: *mut device, attr: *mut device_attribute, buf: *const u8, count: u64) -> i64 {
let mut ehci: *mut ehci_hcd = unsafe { core::mem::zeroed() };
let mut portnum: i32 = 0;
// Owned by companion
let mut new_owner: i32 = 8192i32;
ehci = hcd_to_ehci(dev_get_drvdata(dev as *const device) as *mut usb_hcd);
if (unsafe { sscanf(buf, c"%d".as_ptr() as *const u8, core::ptr::addr_of_mut!(portnum)) }) != 1 {
return 22i32.wrapping_neg() as i64;
}
if portnum < 0 {
portnum = portnum.wrapping_neg();
new_owner = 0; // Owned by EHCI
}
if portnum <= 0 || (portnum as u32) > HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) {
return 2i32.wrapping_neg() as i64;
}
portnum = portnum.wrapping_sub(1);
if new_owner != 0 {
set_bit(portnum as i64, (unsafe { core::ptr::addr_of_mut!((*ehci).companion_ports) }) as *mut u64);
} else {
clear_bit(portnum as i64, (unsafe { core::ptr::addr_of_mut!((*ehci).companion_ports) }) as *mut u64);
}
set_owner(ehci, portnum, new_owner);
count as i64
}

static mut dev_attr_companion: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"companion\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_ehci_sysfs_c_58>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_sysfs_c_58_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_sysfs_c_58_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_sysfs_c_58_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_sysfs_c_58_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_sysfs_c_58_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(companion_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(companion_store) } };

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.uframe_periodic_max_show")]
extern "C" fn uframe_periodic_max_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let ehci: *mut ehci_hcd = hcd_to_ehci(dev_get_drvdata(dev as *const device) as *mut usb_hcd);
(unsafe { sysfs_emit(buf, c"%d\n".as_ptr() as *const u8, (*ehci).uframe_periodic_max) }) as i64
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.uframe_periodic_max_store")]
extern "C" fn uframe_periodic_max_store(dev: *mut device, attr: *mut device_attribute, buf: *const u8, count: u64) -> i64 {
let mut __UNIQUE_ID_y__688: u8 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__687: u8 = unsafe { core::mem::zeroed() };
let mut allocated_max: u8 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i64 = core::ptr::addr_of_mut!(ret) as *mut i64;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut uframe: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(uframe) as *mut u32;
let mut uframe_periodic_max: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(uframe_periodic_max) as *mut u32;
let mut ehci: core::mem::MaybeUninit<*mut ehci_hcd> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut ehci_hcd = core::ptr::addr_of_mut!(ehci) as *mut *mut ehci_hcd;
unsafe {
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_10 = hcd_to_ehci(dev_get_drvdata(dev as *const device) as *mut usb_hcd);
if kstrtouint(buf, 0u32, core::ptr::addr_of_mut!(*__slate_slot_9)) < 0 {
return 22i32.wrapping_neg() as i64;
} else {
if *__slate_slot_9 < 100 || *__slate_slot_9 >= 125 {
while false {
}
_dev_info((*ehci_to_hcd(*__slate_slot_10)).self_.controller as *const device, c"rejecting invalid request for uframe_periodic_max=%u\n".as_ptr() as *const u8, *__slate_slot_9);
return 22i32.wrapping_neg() as i64;
} else {
'__join_0: {
*__slate_slot_6 = 22i32.wrapping_neg() as i64;
// lock, so that our checking does not race with possible periodic
// bandwidth allocation through submitting new urbs.
spin_lock_irqsave!(core::ptr::addr_of_mut!((*(*__slate_slot_10)).lock), *__slate_slot_7);
// for request to decrease max periodic bandwidth, we have to check
// to see whether the decrease is possible.
if *__slate_slot_9 < (*(*__slate_slot_10)).uframe_periodic_max {
allocated_max = 0;
*__slate_slot_8 = 0;
while *__slate_slot_8 < 64 {
__UNIQUE_ID_x__687 = allocated_max;
__UNIQUE_ID_y__688 = *((*(*__slate_slot_10)).bandwidth.as_mut_ptr() as *mut u8).wrapping_offset(*__slate_slot_8 as isize);
while false {
}
allocated_max = (if (__UNIQUE_ID_x__687 as i32) > (__UNIQUE_ID_y__688 as i32) { __UNIQUE_ID_x__687 as i32 } else { __UNIQUE_ID_y__688 as i32 }) as u8;
let __v1: u32 = *__slate_slot_8;
*__slate_slot_8 = __v1.wrapping_add(1);
}
if ((allocated_max as i32) as u32) > *__slate_slot_9 {
while false {
}
_dev_info((*ehci_to_hcd(*__slate_slot_10)).self_.controller as *const device, c"cannot decrease uframe_periodic_max because periodic bandwidth is already allocated (%u > %u)\n".as_ptr() as *const u8, allocated_max as i32, *__slate_slot_9);
break '__join_0;
}
}
// increasing is always ok
while false {
}
_dev_info((*ehci_to_hcd(*__slate_slot_10)).self_.controller as *const device, c"setting max periodic bandwidth to %u%% (== %u usec/uframe)\n".as_ptr() as *const u8, 100u32.wrapping_mul(*__slate_slot_9) / 125, *__slate_slot_9);
if *__slate_slot_9 != 100 {
while false {
}
_dev_warn((*ehci_to_hcd(*__slate_slot_10)).self_.controller as *const device, c"max periodic bandwidth set is non-standard\n".as_ptr() as *const u8);
}
(*(*__slate_slot_10)).uframe_periodic_max = *__slate_slot_9;
*__slate_slot_6 = count as i64;
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*__slate_slot_10)).lock), *__slate_slot_7);
return *__slate_slot_6;
}
}
}
}

static mut dev_attr_uframe_periodic_max: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"uframe_periodic_max\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_ehci_sysfs_c_140>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_sysfs_c_140_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_sysfs_c_140_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_sysfs_c_140_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_sysfs_c_140_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_ehci_sysfs_c_140_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(uframe_periodic_max_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(uframe_periodic_max_store) } };

fn create_sysfs_files(ehci: *mut ehci_hcd) -> i32 {
let mut i: i32 = unsafe { core::mem::zeroed() };
let mut controller: *mut device = unsafe { core::mem::zeroed() };
unsafe {
controller = (*ehci_to_hcd(ehci)).self_.controller;
i = 0;
// with integrated TT there is no companion!
i = device_create_file(controller, core::ptr::addr_of_mut!(dev_attr_companion) as *const device_attribute);
if i == 0 {
i = device_create_file(controller, core::ptr::addr_of_mut!(dev_attr_uframe_periodic_max) as *const device_attribute);
}
return i;
}
}

fn remove_sysfs_files(ehci: *mut ehci_hcd) {
let controller: *mut device = unsafe { (*ehci_to_hcd(ehci)).self_.controller };
// with integrated TT there is no companion!
if ehci_is_TDI!() == 0 {
unsafe { device_remove_file(controller, core::ptr::addr_of_mut!(dev_attr_companion) as *const device_attribute) };
}
unsafe { device_remove_file(controller, core::ptr::addr_of_mut!(dev_attr_uframe_periodic_max) as *const device_attribute) };
}

/// On some systems, leaving remote wakeup enabled prevents system shutdown.
/// The firmware seems to think that powering off is a wakeup event!
/// This routine turns off remote wakeup and everything else, on all ports.
fn ehci_turn_off_all_ports(ehci: *mut ehci_hcd) {
let mut port: i32 = HCS_N_PORTS!(unsafe { (*ehci).hcs_params }) as i32;
loop {
let __v0: i32 = port;
port = __v0.wrapping_sub(1);
if __v0 == 0 {
break;
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
ehci_port_power(ehci, port, false);
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
ehci_writel(ehci as *const ehci_hcd, 42u32, unsafe { (*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32 }.wrapping_offset(port as isize));
}
}

/// Halt HC, turn off all ports, and let the BIOS use the companion controllers.
/// Must be called with interrupts enabled and the lock not held.
fn ehci_silence_controller(ehci: *mut ehci_hcd) {
ehci_halt(ehci);
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
unsafe {
(*ehci).rh_state = 0;
}
ehci_turn_off_all_ports(ehci);
// make BIOS/etc use companion controller during reboot
ehci_writel(ehci as *const ehci_hcd, 0u32, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).configured_flag) });
// unblock posted writes
ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).configured_flag) });
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
}

/// ehci_shutdown kick in for silicon on any bus (not just pci, etc).
/// This forcibly disables dma and IRQs, helping kexec and other cases
/// where the next system software may expect clean state.
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_shutdown")]
extern "C" fn ehci_shutdown(hcd: *mut usb_hcd) {
let ehci: *mut ehci_hcd = hcd_to_ehci(hcd);
// Protect the system from crashing at system shutdown in cases where
// usb host is not added yet from OTG controller driver.
// As ehci_setup() not done yet, so stop accessing registers or
// variables initialized in ehci_setup()
if (unsafe { (*ehci).sbrn }) == 0 {
return;
}
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
unsafe {
let __t0 = true;
(*ehci).__slate_bits_0.__set_shutdown(__t0);
}
unsafe {
(*ehci).rh_state = 3;
}
unsafe {
(*ehci).enabled_hrtimer_events = 0;
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
ehci_silence_controller(ehci);
unsafe { hrtimer_cancel(core::ptr::addr_of_mut!((*ehci).hrtimer)) };
}

/// ehci_work is called from some interrupts, timers, and so on.
/// it calls driver completion functions, after dropping ehci->lock.
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_work")]
extern "C" fn ehci_work(ehci: *mut ehci_hcd) {
unsafe {
// another CPU may drop ehci->lock during a schedule scan while
// it reports urb completions.  this flag guards against bogus
// attempts at re-entrant schedule scanning.
if ((*ehci).__slate_bits_0.__get_scanning() as i32) != 0 {
let __t3 = true;
(*ehci).__slate_bits_0.__set_need_rescan(__t3);
return;
} else {
let __t2 = true;
(*ehci).__slate_bits_0.__set_scanning(__t2);
loop {
let __t1 = false;
(*ehci).__slate_bits_0.__set_need_rescan(__t1);
if (*ehci).async_count != 0 {
scan_async(ehci);
}
if (*ehci).intr_count > 0 {
scan_intr(ehci);
}
if (*ehci).isoc_count > 0 {
scan_isoc(ehci);
}
if ((*ehci).__slate_bits_0.__get_need_rescan() as i32) == 0 {
break;
}
}
let __t0 = false;
(*ehci).__slate_bits_0.__set_scanning(__t0);
// the IO watchdog guards against hardware or driver bugs that
// misplace IRQs, and should let us run completely without IRQs.
// such lossage has been observed on both VT6202 and VT8235.
turn_on_io_watchdog(ehci);
}
}
}

/// Called when the ehci_hcd module is removed.
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_stop")]
extern "C" fn ehci_stop(hcd: *mut usb_hcd) {
let ehci: *mut ehci_hcd = hcd_to_ehci(hcd);
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.controller as *const device, c"stop\n".as_ptr() as *const u8) };
}
// no more interrupts ...
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
unsafe {
(*ehci).enabled_hrtimer_events = 0;
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
ehci_quiesce(ehci);
ehci_silence_controller(ehci);
ehci_reset(ehci);
unsafe { hrtimer_cancel(core::ptr::addr_of_mut!((*ehci).hrtimer)) };
remove_sysfs_files(ehci);
remove_debug_files(ehci);
// root hub is shut down separately (first, when possible)
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
end_free_itds(ehci);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
ehci_mem_cleanup(ehci);
if ((unsafe { (*ehci).__slate_bits_1.__get_amd_pll_fix() }) as i32) == 1 {
unsafe { usb_amd_dev_put() };
}
dbg_status(ehci, c"ehci_stop completed".as_ptr() as *const u8, ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).status) }));
}

/// one-time init, only for memory state
fn ehci_init(hcd: *mut usb_hcd) -> i32 {
let mut __slate_compound_0: spinlock = unsafe { core::mem::zeroed() };
let ehci: *mut ehci_hcd = hcd_to_ehci(hcd);
let mut temp: u32 = 0;
let mut retval: i32 = 0;
let mut hcc_params: u32 = 0;
'__slate_break_1: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
unsafe {
*core::ptr::addr_of_mut!((*ehci).lock) = *{
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
// keep io watchdog by default, those good HCDs could turn off it later
unsafe {
(*ehci).__slate_bits_1.__set_need_io_watchdog(1u32);
}
unsafe { hrtimer_setup(core::ptr::addr_of_mut!((*ehci).hrtimer), Some(ehci_hrtimer_func), 1i32, 0u32) };
unsafe {
(*ehci).next_hrtimer_event = 99;
}
hcc_params = ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).caps).hcc_params) });
// by default set standard 80% (== 100 usec/uframe) max periodic
// bandwidth as required by USB 2.0
unsafe {
(*ehci).uframe_periodic_max = 100;
}
// hw default: 1K periodic list heads, one per frame.
// periodic_size can shrink by USBCMD update if hcc_params allows.
unsafe {
(*ehci).periodic_size = 1024;
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*ehci).async_unlink) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*ehci).async_idle) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*ehci).intr_unlink_wait) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*ehci).intr_unlink) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*ehci).intr_qh_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*ehci).cached_itd_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*ehci).cached_sitd_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*ehci).tt_list) });
if HCC_PGM_FRAMELISTLEN!(hcc_params) != 0 {
// periodic schedule size can be smaller than default
match 1i32 {
0 => {
unsafe {
(*ehci).periodic_size = 1024;
}
}
1 => {
unsafe {
(*ehci).periodic_size = 512;
}
}
2 => {
unsafe {
(*ehci).periodic_size = 256;
}
}
_ => {
loop {
unsafe {
core::arch::asm!("690: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 690b - ., 3; .popsection\n# {0}", const 690 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ehci_init_3, sym __slate_asm_string_ehci_init_4, const 500 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
let __v7: i32 = ehci_mem_init(ehci, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
retval = __v7;
if __v7 < 0 {
return retval;
}
// controllers may cache some of the periodic schedule ...
if HCC_ISOC_CACHE!(hcc_params) != 0 {
unsafe {
(*ehci).i_thresh = 0;
}
} else {
unsafe {
(*ehci).i_thresh = 2u32.wrapping_add(HCC_ISOC_THRES!(hcc_params));
}
}
// full frame cache
// N microframes cached
// dedicate a qh for the async ring head, since we couldn't unlink
// a 'real' qh without stopping the async schedule [4.8].  use it
// as the 'reclamation list head' too.
// its dummy is used in hw_alt_next of many tds, to prevent the qh
// from automatically advancing to the next td after short reads.
unsafe {
(*(*ehci).r#async).qh_next.qh = core::ptr::null_mut::<ehci_qh>();
}
let hw: *mut ehci_qh_hw = unsafe { (*(*ehci).r#async).hw };
unsafe {
(*hw).hw_next = QH_NEXT!(ehci, unsafe { (*(*ehci).r#async).qh_dma });
}
unsafe {
(*hw).hw_info1 = cpu_to_hc32(ehci as *const ehci_hcd, 32768u32);
}
unsafe {
(*hw).hw_token = cpu_to_hc32(ehci as *const ehci_hcd, 64u32);
}
unsafe {
(*hw).hw_qtd_next = EHCI_LIST_END!(ehci);
}
unsafe {
(*(*ehci).r#async).qh_state = 1;
}
unsafe {
(*hw).hw_alt_next = QTD_NEXT!(ehci, unsafe { (*(*(*ehci).r#async).dummy).qtd_dma });
}
// clear interrupt enables, set irq latency
if (unsafe { log2_irq_thresh }) < 0 || (unsafe { log2_irq_thresh }) > 6 {
unsafe {
log2_irq_thresh = 0;
}
}
temp = (1i32 << 16i32.wrapping_add(unsafe { log2_irq_thresh })) as u32;
if HCC_PER_PORT_CHANGE_EVENT!(hcc_params) != 0 {
unsafe {
(*ehci).__slate_bits_2.__set_has_ppcd(1u32);
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.controller as *const device, c"enable per-port change event\n".as_ptr() as *const u8) };
}
temp |= 32768u32;
}
if HCC_CANPARK!(hcc_params) != 0 {
// HW default park == 3, on hardware that supports it (like
// NVidia and ALI silicon), maximizes throughput on the async
// schedule by avoiding QH fetches between transfers.
//
// With fast usb storage devices and NForce2, "park" seems to
// make problems:  throughput reduction (!), data errors...
if (unsafe { park }) != 0 {
unsafe {
park = min_t__2!(u32, unsafe { park }, 3i32);
}
temp |= 2048u32;
temp |= (unsafe { park }) << 8i32;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*ehci_to_hcd(ehci)).self_.controller as *const device, c"park %d\n".as_ptr() as *const u8, park) };
}
}
if HCC_PGM_FRAMELISTLEN!(hcc_params) != 0 {
// periodic schedule size can be smaller than default
let __v18: u32 = temp & 4294967283u32;
temp = __v18;
temp |= 4u32;
}
unsafe {
(*ehci).command = temp;
}
// Accept arbitrarily long scatter-gather lists
if unsafe { (*hcd).localmem_pool }.is_null() {
unsafe {
(*hcd).self_.sg_tablesize = 4294967295u32;
}
}
// Prepare for unlinking active QHs
unsafe {
(*ehci).old_current = 4294967295u32;
}
0
}

/// start HC running; it's halted, ehci_init() has been run (once)
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_run")]
extern "C" fn ehci_run(hcd: *mut usb_hcd) -> i32 {
let ehci: *mut ehci_hcd = hcd_to_ehci(hcd);
let mut temp: u32 = 0;
let mut hcc_params: u32 = 0;
let mut rc: i32 = 0;
unsafe {
(*hcd).__slate_bits_0.__set_uses_new_polling(1u32);
}
// EHCI spec section 4.1
ehci_writel(ehci as *const ehci_hcd, (unsafe { (*ehci).periodic_dma }) as u32, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).frame_list) });
ehci_writel(ehci as *const ehci_hcd, (unsafe { (*(*ehci).r#async).qh_dma }) as u32, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).async_next) });
// hcc_params controls whether ehci->regs->segment must (!!!)
// be used; it constrains QH/ITD/SITD and QTD locations.
// dma_pool consistent memory always uses segment zero.
// streaming mappings for I/O buffers, like dma_map_single(),
// can return segments above 4GB, if the device allows.
//
// NOTE:  the dma mask is visible through dev->dma_mask, so
// drivers can pass this info along ... like NETIF_F_HIGHDMA,
// Scsi_Host.highmem_io, and so forth.  It's readonly to all
// host side drivers though.
hcc_params = ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).caps).hcc_params) });
if HCC_64BIT_ADDR!(hcc_params) != 0 {
ehci_writel(ehci as *const ehci_hcd, 0u32, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).segment) });
}
// Philips, Intel, and maybe others need CMD_RUN before the
//
// root hub will detect new devices (why?); NEC doesn't
let __v2: u32 = (unsafe { (*ehci).command }) & 4294967053u32;
unsafe {
(*ehci).command = __v2;
}
unsafe {
(*ehci).command = (*ehci).command | 1u32;
}
ehci_writel(ehci as *const ehci_hcd, unsafe { (*ehci).command }, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) });
dbg_cmd(ehci, c"init".as_ptr() as *const u8, unsafe { (*ehci).command });
// Start, enabling full USB 2.0 functionality ... usb 1.1 devices
// are explicitly handed to companion controller(s), so no TT is
// involved with the root hub.  (Except where one is integrated,
// and there's no companion controller unless maybe for USB OTG.)
//
// Turning on the CF flag will transfer ownership of all ports
// from the companions to the EHCI controller.  If any of the
// companions are in the middle of a port reset at the time, it
// could cause trouble.  Write-locking ehci_cf_port_reset_rwsem
// guarantees that no resets are in progress.  After we set CF,
// a short delay lets the hardware catch up; new resets shouldn't
// be started before the port switching actions could complete.
unsafe { down_write(core::ptr::addr_of_mut!(ehci_cf_port_reset_rwsem)) };
unsafe {
(*ehci).rh_state = 2;
}
ehci_writel(ehci as *const ehci_hcd, 1u32, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).configured_flag) });
// Wait until HC become operational
ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) }); // unblock posted writes
unsafe { msleep(5u32) };
// For Aspeed, STS_HALT also depends on ASS/PSS status.
// Check CMD_RUN instead.
if ((unsafe { (*ehci).__slate_bits_1.__get_is_aspeed() }) as i32) != 0 {
rc = ehci_handshake(ehci, (unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).command) }) as *mut (), 1u32, 1u32, 100000i32);
} else {
rc = ehci_handshake(ehci, (unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).status) }) as *mut (), 4096u32, 0u32, 100000i32);
}
unsafe { up_write(core::ptr::addr_of_mut!(ehci_cf_port_reset_rwsem)) };
if rc != 0 {
while false {
}
unsafe { _dev_err((*ehci_to_hcd(ehci)).self_.controller as *const device, c"USB %x.%x, controller refused to start: %d\n".as_ptr() as *const u8, (((*ehci).sbrn as i32) & 240) >> 4i32, ((*ehci).sbrn as i32) & 15, rc) };
return rc;
}
unsafe {
(*ehci).last_periodic_enable = ktime_get_real();
}
temp = HC_VERSION!(ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).caps).hc_capbase) }));
while false {
}
unsafe { _dev_info((*ehci_to_hcd(ehci)).self_.controller as *const device, c"USB %x.%x started, EHCI %x.%02x%s\n".as_ptr() as *const u8, (((*ehci).sbrn as i32) & 240) >> 4i32, ((*ehci).sbrn as i32) & 15, temp >> 8i32, temp & 255, if ignore_oc || ((*ehci).__slate_bits_1.__get_spurious_oc() as i32) != 0 { c", overcurrent ignored".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
ehci_writel(ehci as *const ehci_hcd, 55u32, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).intr_enable) }); // Turn On Interrupts
// GRR this is run-once init(), being done every time the HC starts.
// So long as they're part of class devices, we can't do it init()
// since the class device isn't created that early.
create_debug_files(ehci);
create_sysfs_files(ehci);
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_setup")]
extern "C" fn ehci_setup(hcd: *mut usb_hcd) -> i32 {
let ehci: *mut ehci_hcd = hcd_to_ehci(hcd);
let mut retval: i32 = 0;
unsafe {
(*ehci).regs = ((((*ehci).caps as *mut ()) as *const u8).wrapping_offset(HC_LENGTH!(ehci_readl(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*(*ehci).caps).hc_capbase))) as isize) as *mut ()) as *mut ehci_regs;
}
dbg_hcs_params(ehci, c"reset".as_ptr() as *mut u8);
dbg_hcc_params(ehci, c"reset".as_ptr() as *mut u8);
// cache this readonly data; minimize chip reads
unsafe {
(*ehci).hcs_params = ehci_readl(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*(*ehci).caps).hcs_params));
}
unsafe {
(*ehci).sbrn = 32;
}
// data structure init
retval = ehci_init(hcd);
if retval != 0 {
return retval;
}
retval = ehci_halt(ehci);
if retval != 0 {
ehci_mem_cleanup(ehci);
return retval;
}
ehci_reset(ehci);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ehci_setup_693: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd) -> i32>, *mut ()>(Some(ehci_setup)) };

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_irq")]
extern "C" fn ehci_irq(hcd: *mut usb_hcd) -> u32 {
let mut pstatus: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(pstatus) as *mut i32;
let mut ppcd: u32 = unsafe { core::mem::zeroed() };
let mut i: u32 = unsafe { core::mem::zeroed() };
let mut bh: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(bh) as *mut i32;
let mut cmd: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut u32 = core::ptr::addr_of_mut!(cmd) as *mut u32;
let mut pcd_status: u32 = unsafe { core::mem::zeroed() };
let mut masked_status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u32 = core::ptr::addr_of_mut!(masked_status) as *mut u32;
let mut current_status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u32 = core::ptr::addr_of_mut!(current_status) as *mut u32;
let mut status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u32 = core::ptr::addr_of_mut!(status) as *mut u32;
let mut ehci: *mut ehci_hcd = unsafe { core::mem::zeroed() };
unsafe {
ehci = hcd_to_ehci(hcd);
let __slate_auto_bytes_1 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
pcd_status = 0;
let __slate_auto_bytes_4 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
spin_lock(core::ptr::addr_of_mut!((*ehci).lock));
*__slate_slot_23 = 0;
*__slate_slot_22 = ehci_readl(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*(*ehci).regs).status));
'__join_4: {
'__join_5: {
loop {
if *__slate_slot_22 == !0 {
break '__join_5;
} else {
// e.g. cardbus physical eject
*__slate_slot_23 |= *__slate_slot_22;
// We don't use STS_FLR, but some controllers don't like it to
// remain on, so mask it out along with the other status bits.
*__slate_slot_21 = *__slate_slot_22 & 63u32;
// Shared IRQ?
if *__slate_slot_21 == 0 || unlikely!(((*ehci).rh_state as u32) == 0) != 0 {
spin_unlock(core::ptr::addr_of_mut!((*ehci).lock));
return 0;
} else {
// clear (just) interrupts
ehci_writel(ehci as *const ehci_hcd, *__slate_slot_21, core::ptr::addr_of_mut!((*(*ehci).regs).status));
// For edge interrupts, don't race with an interrupt bit being raised
*__slate_slot_22 = ehci_readl(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*(*ehci).regs).status));
if *__slate_slot_22 & 55u32 == 0 {
break;
}
}
}
}
*__slate_slot_19 = ehci_readl(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*(*ehci).regs).command));
*__slate_slot_18 = 0;
// normal [4.15.1.2] or error [4.15.1.1] completion
if likely!(*__slate_slot_23 & 3u32 != 0) != 0 {
if likely!(*__slate_slot_23 & 2u32 == 0) != 0 {
INCR!();
} else {
// Force to check port status
if ((*ehci).__slate_bits_1.__get_has_ci_pec_bug() as i32) != 0 {
*__slate_slot_23 |= 4u32;
}
INCR!();
}
*__slate_slot_18 = 1;
}
// complete the unlinking of some qh [4.15.2.3]
if *__slate_slot_23 & 32u32 != 0 {
// Turn off the IAA watchdog
let __v12: u32 = (*ehci).enabled_hrtimer_events;
let __v11: u32 = ((__v12 as u64) & !BIT!(8i32)) as u32;
(*ehci).enabled_hrtimer_events = __v11;
// Mild optimization: Allow another IAAD to reset the
// hrtimer, if one occurs before the next expiration.
// In theory we could always cancel the hrtimer, but
// tests show that about half the time it will be reset
// for some other event anyway.
if ((*ehci).next_hrtimer_event as u32) == 8 {
(*ehci).next_hrtimer_event = ((*ehci).next_hrtimer_event as u32).wrapping_add(1) as u32;
}
// guard against (alleged) silicon errata
if *__slate_slot_19 & 64u32 != 0 {
}
if ((*ehci).__slate_bits_0.__get_iaa_in_progress() as i32) != 0 {
INCR!();
}
end_iaa_cycle(ehci);
}
// remote wakeup [4.3.1]
if *__slate_slot_23 & 4u32 != 0 {
i = HCS_N_PORTS!(unsafe { (*ehci).hcs_params });
ppcd = 4294967295u32;
// kick root hub later
pcd_status = *__slate_slot_23;
// resume root hub?
if ((*ehci).rh_state as u32) == 1 {
usb_hcd_resume_root_hub(hcd);
}
// get per-port change detect bits
if ((*ehci).__slate_bits_2.__get_has_ppcd() as i32) != 0 {
ppcd = *__slate_slot_23 >> 16i32;
}
loop {
let __v4: u32 = i;
i = __v4.wrapping_sub(1);
if __v4 != 0 {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// leverage per-port change bits feature
if ppcd & ((1i32 << i) as u32) != 0 {
*__slate_slot_5 = ehci_readl(ehci as *const ehci_hcd, ((*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32).wrapping_offset(i as isize)) as i32;
if *__slate_slot_5 & 8192i32 == 0 {
if test_bit__3!(i, core::ptr::addr_of_mut!((*ehci).suspended_ports)) != 0 && (*__slate_slot_5 & 64i32 != 0 || *__slate_slot_5 & 128i32 == 0) && *__slate_slot_5 & 4i32 != 0 && *((*ehci).reset_done.as_mut_ptr() as *mut u64).wrapping_offset(i as isize) == 0 {
// start USB_RESUME_TIMEOUT msec resume signaling from
// this port, and make hub_wq collect
// PORT_STAT_C_SUSPEND to stop that signaling.
*((*ehci).reset_done.as_mut_ptr() as *mut u64).wrapping_offset(i as isize) = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(40u32));
set_bit(i as i64, core::ptr::addr_of_mut!((*ehci).resuming_ports) as *mut u64);
usb_hcd_start_port_resume(core::ptr::addr_of_mut!((*hcd).self_), i as i32);
mod_timer(core::ptr::addr_of_mut!((*hcd).rh_timer), *((*ehci).reset_done.as_mut_ptr() as *mut u64).wrapping_offset(i as isize));
}
}
}
} else {
break;
}
}
}
// PCI errors [4.15.2.4]
if unlikely!(*__slate_slot_23 & 16u32 != 0) != 0 {
while false {
}
_dev_err((*ehci_to_hcd(ehci)).self_.controller as *const device, c"fatal error\n".as_ptr() as *const u8);
dbg_cmd(ehci, c"fatal".as_ptr() as *const u8, *__slate_slot_19);
dbg_status(ehci, c"fatal".as_ptr() as *const u8, *__slate_slot_23);
} else {
break '__join_4;
}
}
usb_hc_died(hcd);
// Don't let the controller do anything more
let __t0 = true;
(*ehci).__slate_bits_0.__set_shutdown(__t0);
(*ehci).rh_state = 3;
let __v0: u32 = (*ehci).command & 4294967246u32;
(*ehci).command = __v0;
ehci_writel(ehci as *const ehci_hcd, (*ehci).command, core::ptr::addr_of_mut!((*(*ehci).regs).command));
ehci_writel(ehci as *const ehci_hcd, 0u32, core::ptr::addr_of_mut!((*(*ehci).regs).intr_enable));
ehci_handle_controller_death(ehci);
// Handle completions when the controller stops
*__slate_slot_18 = 0;
}
if *__slate_slot_18 != 0 {
ehci_work(ehci);
}
spin_unlock(core::ptr::addr_of_mut!((*ehci).lock));
if pcd_status != 0 {
usb_hcd_poll_rh_status(hcd);
}
return 1;
}
}

/// non-error returns are a promise to giveback() the urb later
/// we drop ownership so next owner (or urb unlink) can get it
///
/// urb + dev is in hcd.self.controller.urb_list
/// we're queueing TDs onto software and hardware lists
///
/// hcd-specific init for hcpriv hasn't been done yet
///
/// NOTE:  control, bulk, and interrupt share the same code to append TDs
/// to a (possibly active) QH, and the same QH scanning code.
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_urb_enqueue")]
extern "C" fn ehci_urb_enqueue(hcd: *mut usb_hcd, urb: *mut urb, mem_flags: u32) -> i32 {
let ehci: *mut ehci_hcd = hcd_to_ehci(hcd);
let mut qtd_list: list_head = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(qtd_list) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
INIT_LIST_HEAD(core::ptr::addr_of_mut!(qtd_list));
// qh_completions() code doesn't handle all the fault cases
// in multi-TD control transfers.  Even 1KB is rare anyway.
// case PIPE_BULK:
match usb_pipetype!(unsafe { (*urb).pipe }) {
2 => {
if (unsafe { (*urb).transfer_buffer_length }) > 16384u32 {
return 90i32.wrapping_neg();
}
// qh_completions() code doesn't handle all the fault cases
// in multi-TD control transfers.  Even 1KB is rare anyway.
// case PIPE_BULK:
if qh_urb_transaction(ehci, urb, core::ptr::addr_of_mut!(qtd_list), mem_flags).is_null() {
return 12i32.wrapping_neg();
}
return submit_async(ehci, urb, core::ptr::addr_of_mut!(qtd_list), mem_flags);
}
1 => {
if qh_urb_transaction(ehci, urb, core::ptr::addr_of_mut!(qtd_list), mem_flags).is_null() {
return 12i32.wrapping_neg();
}
return intr_submit(ehci, urb, core::ptr::addr_of_mut!(qtd_list), mem_flags);
}
0 => {
if ((unsafe { (*(*urb).dev).speed }) as u32) == 3 {
return itd_submit(ehci, urb, mem_flags);
} else {
return sitd_submit(ehci, urb, mem_flags);
}
}
_ => {
if qh_urb_transaction(ehci, urb, core::ptr::addr_of_mut!(qtd_list), mem_flags).is_null() {
return 12i32.wrapping_neg();
}
return submit_async(ehci, urb, core::ptr::addr_of_mut!(qtd_list), mem_flags);
}
}
unsafe { core::mem::zeroed() }
}

// remove from hardware lists
// completions normally happen asynchronously
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_urb_dequeue")]
extern "C" fn ehci_urb_dequeue(hcd: *mut usb_hcd, urb: *mut urb, status: i32) -> i32 {
let mut rc: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(rc) as *mut i32;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut qh: core::mem::MaybeUninit<*mut ehci_qh> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut ehci_qh = core::ptr::addr_of_mut!(qh) as *mut *mut ehci_qh;
let mut ehci: *mut ehci_hcd = unsafe { core::mem::zeroed() };
unsafe {
ehci = hcd_to_ehci(hcd);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_4);
*__slate_slot_3 = usb_hcd_check_unlink_urb(hcd, urb, status);
if *__slate_slot_3 == 0 {
if usb_pipetype!(unsafe { (*urb).pipe }) == 0 {
// We don't expedite dequeue for isochronous URBs.
// Just wait until they complete normally or their
// time slot expires.
} else {
*__slate_slot_5 = (*urb).hcpriv as *mut ehci_qh;
let __v2: *mut ehci_qh = *__slate_slot_5;
(*__v2).unlink_reason = (((*__v2).unlink_reason as i32) | 32) as u8;
let __t0: i32 = (*(*__slate_slot_5)).qh_state as i32;
if __t0 == 1 {
if usb_pipetype!(unsafe { (*urb).pipe }) == 1 {
start_unlink_intr(ehci, *__slate_slot_5);
} else {
start_unlink_async(ehci, *__slate_slot_5);
}
} else {
if __t0 == 5 {
(*(*__slate_slot_5)).__slate_bits_0.__set_dequeue_during_giveback(1u32);
} else {
if __t0 != 2 {
if __t0 != 4 {
if __t0 == 3 {
qh_completions(ehci, *__slate_slot_5); // QH might be waiting for a Clear-TT-Buffer
}
}
}
}
}
}
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_4);
return *__slate_slot_3;
}
// already started
}

// bulk qh holds the data toggle
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_endpoint_disable")]
extern "C" fn ehci_endpoint_disable(hcd: *mut usb_hcd, ep: *mut usb_host_endpoint) {
let mut stream: *mut ehci_iso_stream = unsafe { core::mem::zeroed() };
let mut qh: core::mem::MaybeUninit<*mut ehci_qh> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut ehci_qh = core::ptr::addr_of_mut!(qh) as *mut *mut ehci_qh;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut ehci: *mut ehci_hcd = unsafe { core::mem::zeroed() };
unsafe {
ehci = hcd_to_ehci(hcd);
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// ASSERT:  any requests/urbs are being unlinked
//
// ASSERT:  nobody can be submitting urbs for this any more
'__join_0: {
'__join_22: {
'__join_2: {
loop {
spin_lock_irqsave!(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_11);
*__slate_slot_10 = (*ep).hcpriv as *mut ehci_qh;
if (*__slate_slot_10).is_null() {
break '__join_0;
} else {
// endpoints can be iso streams.  for now, we don't
// accelerate iso completions ... so spin a while.
if unsafe { (*(*__slate_slot_10)).hw }.is_null() {
stream = (*ep).hcpriv as *mut ehci_iso_stream;
if list_empty(core::ptr::addr_of_mut!((*stream).td_list) as *const list_head) != 0 {
break '__join_22;
}
} else {
let __v8: *mut ehci_qh = *__slate_slot_10;
(*__v8).unlink_reason = (((*__v8).unlink_reason as i32) | 32) as u8;
let __t0: i32 = (*(*__slate_slot_10)).qh_state as i32;
if __t0 == 1 {
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_10)).qtd_list) as *const list_head) != 0 {
let __v2: *mut ehci_qh = *__slate_slot_10;
(*__v2).unlink_reason = (((*__v2).unlink_reason as i32) | 16) as u8;
} else {
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("694: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 694b - ., 3; .popsection\n# {0}", const 694 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ehci_endpoint_disable_13, sym __slate_asm_string_ehci_endpoint_disable_14, const 1009 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("695: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 695b - ., 4; .popsection\n# {0}", const 695 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
if usb_endpoint_type(core::ptr::addr_of_mut!((*ep).desc) as *const usb_endpoint_descriptor) != 3 {
start_unlink_async(ehci, *__slate_slot_10);
} else {
start_unlink_intr(ehci, *__slate_slot_10);
}
} else {
if __t0 != 5 {
if __t0 != 2 {
if __t0 != 4 {
if __t0 == 3 {
if ((*(*__slate_slot_10)).__slate_bits_0.__get_clearing_tt() as i32) == 0 {
break;
}
} else {
break '__join_2;
}
}
}
}
}
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_11); // already in unlinking
// wait for hw to finish?
schedule_timeout_uninterruptible(1i64);
}
}
// fully unlinked
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_10)).qtd_list) as *const list_head) != 0 {
if (*(*__slate_slot_10)).ps.bw_uperiod != 0 {
reserve_release_intr_bandwidth(ehci, *__slate_slot_10, 1i32.wrapping_neg());
}
qh_destroy(ehci, *__slate_slot_10);
break '__join_0;
}
}
while false {
}
_dev_err((*ehci_to_hcd(ehci)).self_.controller as *const device, c"qh %p (#%02x) state %d%s\n".as_ptr() as *const u8, *__slate_slot_10, (*ep).desc.bEndpointAddress as i32, (*(*__slate_slot_10)).qh_state as i32, if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_10)).qtd_list) as *const list_head) != 0 { c"".as_ptr() as *mut u8 } else { c"(has tds)".as_ptr() as *mut u8 });
// caller was supposed to have unlinked any requests;
// that's not our job.  just leak this memory.
break '__join_0;
}
// BUG_ON(!list_empty(&stream->free_list));
reserve_release_iso_bandwidth(ehci, stream, 1i32.wrapping_neg());
kfree(stream as *const ());
}
(*ep).hcpriv = core::ptr::null_mut::<()>();
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*ehci).lock), *__slate_slot_11);
}
}

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_endpoint_reset")]
extern "C" fn ehci_endpoint_reset(hcd: *mut usb_hcd, ep: *mut usb_host_endpoint) {
let ehci: *mut ehci_hcd = hcd_to_ehci(hcd);
let mut qh: *mut ehci_qh = unsafe { core::mem::zeroed() };
let eptype: i32 = usb_endpoint_type((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor);
let epnum: i32 = usb_endpoint_num((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor);
let is_out: i32 = usb_endpoint_dir_out((unsafe { core::ptr::addr_of_mut!((*ep).desc) }) as *const usb_endpoint_descriptor);
let mut flags: u64 = 0;
if eptype != 2 && eptype != 3 {
return;
}
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*ehci).lock) }, flags);
qh = (unsafe { (*ep).hcpriv }) as *mut ehci_qh;
// For Bulk and Interrupt endpoints we maintain the toggle state
// in the hardware; the toggle bits in udev aren't used at all.
// When an endpoint is reset by usb_clear_halt() we must reset
// the toggle bit in the QH.
if !qh.is_null() {
if list_empty((unsafe { core::ptr::addr_of_mut!((*qh).qtd_list) }) as *const list_head) == 0 {
let mut __v5: i32;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"clear_halt for a busy endpoint\n".as_ptr() as *const u8) };
let mut __v6: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v6 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v7: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_ehci_endpoint_reset_8, sym __slate_asm_string_ehci_endpoint_reset_9, const 1069 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v7 = bug;
unsafe { __v6.unwrap()(__v7) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v5 = __ret_warn_on;
} else {
// The toggle value in the QH can't be updated
// while the QH is active.  Unlink it now;
// re-linking will call qh_refresh().
usb_settoggle!(unsafe { (*qh).ps.udev }, epnum, is_out, 0i32);
unsafe {
(*qh).unlink_reason = (((*qh).unlink_reason as i32) | 32) as u8;
}
if eptype == 2 {
start_unlink_async(ehci, qh);
} else {
start_unlink_intr(ehci, qh);
}
}
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*ehci).lock) }, flags);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_696: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_get_frame")]
extern "C" fn ehci_get_frame(hcd: *mut usb_hcd) -> i32 {
let ehci: *mut ehci_hcd = hcd_to_ehci(hcd);
((ehci_read_frame_index(ehci) >> 3i32) % unsafe { (*ehci).periodic_size }) as i32
}

// Device addition and removal
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_remove_device")]
extern "C" fn ehci_remove_device(hcd: *mut usb_hcd, udev: *mut usb_device) {
let ehci: *mut ehci_hcd = hcd_to_ehci(hcd);
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
drop_tt(udev);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
}

/// Clear wakeup signal locked in zhaoxin platform when device plug in.
fn ehci_zx_wakeup_clear(ehci: *mut ehci_hcd) {
let reg: *mut u32 = unsafe { (*(*ehci).regs).__slate_anon_11.port_status.as_mut_ptr() as *mut u32 }.wrapping_offset(4 as isize);
let mut t1: u32 = ehci_readl(ehci as *const ehci_hcd, reg);
let __v1: u32 = t1 & 4293984255u32;
t1 = __v1;
let __v2: u32 = t1;
t1 = __v2 | (PORT_TEST!(5i32) as u32);
ehci_writel(ehci as *const ehci_hcd, t1, reg);
t1 = ehci_readl(ehci as *const ehci_hcd, reg);
unsafe { msleep(1u32) };
let __v5: u32 = t1 & 4293984255u32;
t1 = __v5;
ehci_writel(ehci as *const ehci_hcd, t1, reg);
ehci_readl(ehci as *const ehci_hcd, reg);
unsafe { msleep(1u32) };
t1 = ehci_readl(ehci as *const ehci_hcd, reg);
ehci_writel(ehci as *const ehci_hcd, t1 | 2u32, reg);
ehci_readl(ehci as *const ehci_hcd, reg);
}

// suspend/resume, section 4.3
// These routines handle the generic parts of controller suspend/resume
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_suspend")]
extern "C" fn ehci_suspend(hcd: *mut usb_hcd, do_wakeup: bool) -> i32 {
let ehci: *mut ehci_hcd = hcd_to_ehci(hcd);
if time_before!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, unsafe { (*ehci).next_statechange }) {
unsafe { msleep(10u32) };
}
// Root hub was already suspended.  Disable IRQ emission and
// mark HW unaccessible.  The PM and USB cores make sure that
// the root hub is either suspended or stopped.
ehci_prepare_ports_for_controller_suspend!(ehci, do_wakeup);
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
ehci_writel(ehci as *const ehci_hcd, 0u32, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).intr_enable) });
ehci_readl(ehci as *const ehci_hcd, unsafe { core::ptr::addr_of_mut!((*(*ehci).regs).intr_enable) });
clear_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*hcd).flags) }) as *mut u64);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*ehci).lock) });
unsafe { synchronize_irq((*hcd).irq) };
// Check for race with a wakeup request
if do_wakeup && HCD_WAKEUP_PENDING!(hcd) != 0 {
ehci_resume(hcd, false);
return 16i32.wrapping_neg();
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ehci_suspend_697: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd, bool) -> i32>, *mut ()>(Some(ehci_suspend)) };

/// Returns 0 if power was preserved, 1 if power was lost
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_resume")]
extern "C" fn ehci_resume(hcd: *mut usb_hcd, force_reset: bool) -> i32 {
let mut mask: i32 = unsafe { core::mem::zeroed() };
let mut ehci: *mut ehci_hcd = unsafe { core::mem::zeroed() };
unsafe {
ehci = hcd_to_ehci(hcd);
if time_before!(core::ptr::read_volatile(core::ptr::addr_of!(jiffies)), unsafe { (*ehci).next_statechange }) {
msleep(100u32);
}
// Mark hardware accessible again as we are back to full power by now
set_bit(0i64, core::ptr::addr_of_mut!((*hcd).flags) as *mut u64);
if ((*ehci).__slate_bits_0.__get_shutdown() as i32) != 0 {
return 0;
} else {
// Controller is dead
if ((*ehci).__slate_bits_1.__get_zx_wakeup_clear_needed() as i32) != 0 {
ehci_zx_wakeup_clear(ehci);
}
// If CF is still set and reset isn't forced
// then we maintained suspend power.
// Just undo the effect of ehci_suspend().
if ehci_readl(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*(*ehci).regs).configured_flag)) == 1u32 && !force_reset {
mask = 55i32;
ehci_prepare_ports_for_controller_resume!(ehci);
spin_lock_irq(core::ptr::addr_of_mut!((*ehci).lock));
if ((*ehci).__slate_bits_0.__get_shutdown() as i32) == 0 {
if ((*(*hcd).self_.root_hub).__slate_bits_1.__get_do_remote_wakeup() as i32) == 0 {
let __v0: i32 = mask & -5i32;
mask = __v0;
}
ehci_writel(ehci as *const ehci_hcd, mask as u32, core::ptr::addr_of_mut!((*(*ehci).regs).intr_enable));
ehci_readl(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*(*ehci).regs).intr_enable));
}
} else {
// Else reset, to cope with power loss or resume from hibernation
// having let the firmware kick in during reboot.
usb_root_hub_lost_power((*hcd).self_.root_hub);
ehci_halt(ehci);
ehci_reset(ehci);
spin_lock_irq(core::ptr::addr_of_mut!((*ehci).lock));
if ((*ehci).__slate_bits_0.__get_shutdown() as i32) == 0 {
ehci_writel(ehci as *const ehci_hcd, (*ehci).command, core::ptr::addr_of_mut!((*(*ehci).regs).command));
ehci_writel(ehci as *const ehci_hcd, 1u32, core::ptr::addr_of_mut!((*(*ehci).regs).configured_flag));
ehci_readl(ehci as *const ehci_hcd, core::ptr::addr_of_mut!((*(*ehci).regs).command)); // unblock posted writes
(*ehci).rh_state = 1;
spin_unlock_irq(core::ptr::addr_of_mut!((*ehci).lock));
return 1;
}
}
spin_unlock_irq(core::ptr::addr_of_mut!((*ehci).lock));
return 0;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ehci_resume_698: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut usb_hcd, bool) -> i32>, *mut ()>(Some(ehci_resume)) };

// Generic structure: This gets copied for platform drivers so that
// individual entries can be overridden as needed.
/// generic hardware linkage
/// basic lifecycle operations
/// managing i/o requests and associated device resources
/// scheduling support
/// root hub support
/// device support
static ehci_hc_driver: __SlateConst<hc_driver> = __SlateConst({
let mut __t0: hc_driver = unsafe { core::mem::zeroed() };
__t0.description = unsafe { core::ptr::addr_of!(hcd_name) as *const u8 };
__t0.product_desc = (b"EHCI Host Controller\0".as_ptr() as *mut u8) as *const u8;
__t0.hcd_priv_size = 872u64;
__t0.irq = Some(ehci_irq);
__t0.flags = 291i32;
__t0.reset = Some(ehci_setup);
__t0.start = Some(ehci_run);
__t0.stop = Some(ehci_stop);
__t0.shutdown = Some(ehci_shutdown);
__t0.get_frame_number = Some(ehci_get_frame);
__t0.urb_enqueue = Some(ehci_urb_enqueue);
__t0.urb_dequeue = Some(ehci_urb_dequeue);
__t0.endpoint_disable = Some(ehci_endpoint_disable);
__t0.endpoint_reset = Some(ehci_endpoint_reset);
__t0.hub_status_data = Some(ehci_hub_status_data);
__t0.hub_control = Some(ehci_hub_control);
__t0.bus_suspend = Some(ehci_bus_suspend);
__t0.bus_resume = Some(ehci_bus_resume);
__t0.get_resuming_ports = Some(ehci_get_resuming_ports);
__t0.relinquish_port = Some(ehci_relinquish_port);
__t0.port_handed_over = Some(ehci_port_handed_over);
__t0.clear_tt_buffer_complete = Some(ehci_clear_tt_buffer_complete);
__t0.free_dev = Some(ehci_remove_device);
__t0
});

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ehci_hcd.ehci_init_driver")]
extern "C" fn ehci_init_driver(drv: *mut hc_driver, over: *const ehci_driver_overrides) {
// Copy the generic table to drv and then apply the overrides
unsafe {
*drv = ehci_hc_driver.0;
}
if !over.is_null() {
let __v1: u64 = unsafe { (*drv).hcd_priv_size };
unsafe {
(*drv).hcd_priv_size = __v1.wrapping_add((*over).extra_priv_size);
}
if (unsafe { (*over).reset }) != None {
unsafe {
(*drv).reset = (*over).reset;
}
}
if (unsafe { (*over).port_power }) != None {
unsafe {
(*drv).port_power = (*over).port_power;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ehci_init_driver_699: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hc_driver, *const ehci_driver_overrides)>, *mut ()>(Some(ehci_init_driver)) };

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_700: [u8; 70] = [101, 104, 99, 105, 95, 104, 99, 100, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 85, 83, 66, 32, 50, 46, 48, 32, 39, 69, 110, 104, 97, 110, 99, 101, 100, 39, 32, 72, 111, 115, 116, 32, 67, 111, 110, 116, 114, 111, 108, 108, 101, 114, 32, 40, 69, 72, 67, 73, 41, 32, 68, 114, 105, 118, 101, 114, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_701: [u8; 31] = [101, 104, 99, 105, 95, 104, 99, 100, 46, 97, 117, 116, 104, 111, 114, 61, 68, 97, 118, 105, 100, 32, 66, 114, 111, 119, 110, 101, 108, 108, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_702: [u8; 40] = [101, 104, 99, 105, 95, 104, 99, 100, 46, 102, 105, 108, 101, 61, 100, 114, 105, 118, 101, 114, 115, 47, 117, 115, 98, 47, 104, 111, 115, 116, 47, 101, 104, 99, 105, 45, 104, 99, 100, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_703: [u8; 21] = [101, 104, 99, 105, 95, 104, 99, 100, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

static platform_drivers: __SlateConst<[*mut platform_driver; 0]> = __SlateConst([]);

#[unsafe(link_section = ".init.text")]
extern "C" fn ehci_hcd_init() -> i32 {
let mut retval: i32 = unsafe { core::mem::zeroed() };
unsafe {
retval = 0;
if usb_disabled() != 0 {
return 19i32.wrapping_neg();
} else {
retval = platform_register_drivers!(core::ptr::addr_of!(platform_drivers.0) as *const *mut platform_driver, ARRAY_SIZE__13!());
if retval < 0 {
return retval;
} else {
return 0;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ehci_hcd_init_705: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(ehci_hcd_init)) };

#[unsafe(link_section = ".exit.text")]
extern "C" fn ehci_hcd_cleanup() {
unsafe { platform_unregister_drivers(core::ptr::addr_of!(platform_drivers.0) as *const *mut platform_driver, ARRAY_SIZE__26!() as u32) };
}

#[unsafe(link_section = ".exitcall.exit")]
static mut __exitcall_ehci_hcd_cleanup: Option<unsafe extern "C" fn()> = Some(ehci_hcd_cleanup);

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_disable_async_6: [u8; 1] = *b"\0";

static __slate_asm_string_disable_async_7: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/ehci-q.c\0";

static __slate_asm_string_ehci_endpoint_disable_13: [u8; 1] = *b"\0";

static __slate_asm_string_ehci_endpoint_disable_14: [u8; 57] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/ehci-hcd.c\0";

static __slate_asm_string_ehci_endpoint_reset_8: [u8; 32] = *b"clear_halt for a busy endpoint\n\0";

static __slate_asm_string_ehci_endpoint_reset_9: [u8; 57] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/ehci-hcd.c\0";

static __slate_asm_string_ehci_init_3: [u8; 1] = *b"\0";

static __slate_asm_string_ehci_init_4: [u8; 57] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/ehci-hcd.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_intr_submit_8: [u8; 1] = *b"\0";

static __slate_asm_string_intr_submit_9: [u8; 59] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/ehci-sched.c\0";

static __slate_asm_string_qh_destroy_0: [u8; 1] = *b"\0";

static __slate_asm_string_qh_destroy_1: [u8; 57] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/ehci-mem.c\0";

static __slate_asm_string_qh_link_async_1: [u8; 1] = *b"\0";

static __slate_asm_string_qh_link_async_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/ehci-q.c\0";

static __slate_asm_string_qh_update_1: [u8; 1] = *b"\0";

static __slate_asm_string_qh_update_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/ehci-q.c\0";

static __slate_asm_string_sitd_link_urb_6: [u8; 1] = *b"\0";

static __slate_asm_string_sitd_link_urb_7: [u8; 59] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/ehci-sched.c\0";

static __slate_asm_string_unlink_empty_async_suspended_1: [u8; 1] = *b"\0";

static __slate_asm_string_unlink_empty_async_suspended_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/usb/host/ehci-q.c\0";
