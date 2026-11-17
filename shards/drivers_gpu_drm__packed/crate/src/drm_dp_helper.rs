
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
//! Copyright © 2009 Keith Packard
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
//! # chars in a path name including nul
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
//! New user namespace
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! This counts to 15. Any more, it will return 16th argument.
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
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
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
//! instrument_get_user() - add instrumentation to get_user()-like macros
//! @to: destination variable, may not be address-taken
//!
//! get_user() and friends are fragile, so it may depend on the implementation
//! whether the instrumentation happens before or after the data is copied from
//! the userspace.
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
//! for_each_set_bit_wrap - iterate over all set bits starting from @start, and
//! wrapping around the end of bitmap.
//! @bit: offset for current iteration
//! @addr: bitmap address to base the search on
//! @size: bitmap size in number of bits
//! @start: Starting bit for bitmap traversing, wrapping around the bitmap end
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
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
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
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
//! Keep unconverted code working
//! number of env pointers
//! Generate SD flag indexes
//! Resource type
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
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! no-op
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
//! The microseconds/nanosecond delay multiplicators are used to convert a
//! constant microseconds/nanoseconds value to a value which can be used by the
//! architectures specific implementation to transform it into loops.
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
//! Number of bits in the handle that stack depot doesn't use. Users may store
//! information in them via stack_depot_set/get_extra_bits.
//!  + ((B + 0.000122)*4096))
//! These are specified by iBCS2
//! Epoll event masks
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
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
// Math result not representable
// Invalid system call number
// Protocol error
// Value too large for defined data type
// Operation not supported on transport endpoint
// Connection timed out
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
// min_t - return minimum of two values, using the specified type
// @type: data type to use
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
// container_of_const - cast a member of a structure out to the containing
// 			structure and preserve the const-ness of the pointer
// @ptr:		the pointer to the member
// @type:		the type of the container struct this is embedded in.
// @member:		the name of the member within the struct.
//
// Always prefer container_of_const() instead of container_of() in new code.
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
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
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
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
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
// guarantee 'op' and 'cond' are evaluated after timeout expired
// read_poll_timeout - Periodically poll an address until a condition is
//                     met or a timeout occurs
// @op: accessor function (takes @args as its arguments)
// @val: Variable to read the value into
// @cond: Break condition (usually involving @val)
// @sleep_us: Maximum time to sleep between reads in us (0 tight-loops). Please
//            read usleep_range() function description for details and
//            limitations.
// @timeout_us: Timeout in us, 0 means never timeout
// @sleep_before_read: if it is true, sleep @sleep_us before read.
// @args: arguments for @op poll
//
// When available, you'll probably want to use one of the specialized
// macros defined below rather than this macro directly.
//
// Returns: 0 on success and -ETIMEDOUT upon a timeout. In either
// case, the last read value at @args is stored in @val. Must not
// be called from atomic context if sleep_us or timeout_us are used.
// readx_poll_timeout - Periodically poll an address until a condition is met or a timeout occurs
// @op: accessor function (takes @addr as its only argument)
// @addr: Address to poll
// @val: Variable to read the value into
// @cond: Break condition (usually involving @val)
// @sleep_us: Maximum time to sleep between reads in us (0 tight-loops). Please
//            read usleep_range() function description for details and
//            limitations.
// @timeout_us: Timeout in us, 0 means never timeout
//
// When available, you'll probably want to use one of the specialized
// macros defined below rather than this macro directly.
//
// Returns: 0 on success and -ETIMEDOUT upon a timeout. In either
// case, the last read value at @addr is stored in @val. Must not
// be called from atomic context if sleep_us or timeout_us are used.
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
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
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
macro_rules! ARRAY_SIZE__5 {
() => {
((core::mem::size_of::<[u16; 7]>() as u64) / (core::mem::size_of::<u16>() as u64)).wrapping_add(__must_be_array__23!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__7 {
() => {
((core::mem::size_of::<[*const u8; 9]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__16!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__22 {
() => {
((core::mem::size_of::<[dpcd_quirk; 12]>() as u64) / (core::mem::size_of::<dpcd_quirk>() as u64)).wrapping_add(__must_be_array__12!() as u64)
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

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
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

macro_rules! DP_128B132B_TRAINING_AUX_RD_INTERVAL_PHY_REPEATER {
($dp_phy:expr) => {
DP_LTTPR_REG!($dp_phy, 983074i32)
};
}

/// prior to 1.2 bit 7 was reserved mbz
macro_rules! DP_GET_SINK_COUNT {
($x:expr) => {
((($x) as i32) & 128) >> 1i32 | (($x) as i32) & 63
};
}

macro_rules! DP_LANE0_1_STATUS_PHY_REPEATER {
($dp_phy:expr) => {
DP_LTTPR_REG!($dp_phy, 983088i32)
};
}

macro_rules! DP_LTTPR_BASE {
($dp_phy:expr) => {
983056i32.wrapping_add(80i32.wrapping_mul($dp_phy.wrapping_sub(1)))
};
}

macro_rules! DP_LTTPR_BASE__2 {
($dp_phy:expr) => {
983056u32.wrapping_add(80u32.wrapping_mul((($dp_phy) as u32).wrapping_sub(1)))
};
}

macro_rules! DP_LTTPR_REG {
($dp_phy:expr, $lttpr1_reg:expr) => {
DP_LTTPR_BASE__2!($dp_phy).wrapping_sub(DP_LTTPR_BASE!(1i32) as u32).wrapping_add(($lttpr1_reg) as u32)
};
}

macro_rules! DP_OUI_PHY_REPEATER {
($dp_phy:expr) => {
DP_LTTPR_REG!($dp_phy, 983101i32)
};
}

macro_rules! DP_TRAINING_AUX_RD_INTERVAL_PHY_REPEATER {
($dp_phy:expr) => {
DP_LTTPR_REG!($dp_phy, 983072i32)
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
__ALIGN_KERNEL_MASK!($x, (($a) as i32).wrapping_sub(1))
};
}

macro_rules! __WRITE_ONCE__3 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut bool, $val) };
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

/// Compile time versions of __arch_hweightN()
macro_rules! __const_hweight8__2 {
($w:expr) => {
((((($w) as i32) as u64) & 1u64 != 0) as i32).wrapping_add((((($w) as i32) as u64) & 2u64 != 0) as i32).wrapping_add((((($w) as i32) as u64) & 4u64 != 0) as i32).wrapping_add((((($w) as i32) as u64) & 8u64 != 0) as i32).wrapping_add((((($w) as i32) as u64) & 16u64 != 0) as i32).wrapping_add((((($w) as i32) as u64) & 32u64 != 0) as i32).wrapping_add((((($w) as i32) as u64) & 64u64 != 0) as i32).wrapping_add((((($w) as i32) as u64) & 128u64 != 0) as i32) as u32
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

macro_rules! __must_be_array__12 {
() => {
core::mem::size_of::<__SlateRecord_drm_dp_helper_c_2581>() as i32
};
}

macro_rules! __must_be_array__16 {
() => {
core::mem::size_of::<__SlateRecord_drm_dp_helper_c_442>() as i32
};
}

macro_rules! __must_be_array__23 {
() => {
core::mem::size_of::<__SlateRecord_drm_dp_helper_c_2462>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr {
() => {
core::mem::size_of::<__SlateRecord_drm_dp_helper_c_2410>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__3 {
() => {
core::mem::size_of::<__SlateRecord_drm_dp_helper_c_2410_2>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
() => {
((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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

macro_rules! __ratelimit__2 {
($state:expr) => {
unsafe { ___ratelimit($state, c"drm_dp_i2c_do_msg".as_ptr() as *const u8) }
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
macro_rules! __sign_use__3 {
($ux:expr) => {
if is_signed_type__3!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u32>() }) as u64) < 4) as i32)) }
};
}

macro_rules! __types_ok3 {
($ux:expr, $uy:expr, $uz:expr) => {
__sign_use!($ux) & __sign_use!($uy) & __sign_use!($uz)
};
}

macro_rules! __types_ok__3 {
($ux:expr, $uy:expr) => {
__sign_use__3!($ux) & __sign_use__3!($uy)
};
}

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__46!($type) as isize)) }) as *mut ()) as *mut $type;
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
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__23!($type) as isize)) }) as *mut ()) as *mut $type;
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

/// Generic interface.
macro_rules! hweight8 {
($w:expr) => {
{
let __v: u32 = if unsafe { __slate_intrinsic_70b71f185ebb8c4a($w) } {
__const_hweight8__2!($w)
} else {
__arch_hweight8(($w) as u32)
};
__v
}
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
macro_rules! ilog2 {
($n:expr) => {
if unsafe { __slate_intrinsic_70b71f185ebb8c4a($n) } { if (($n) as i32) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { if (core::mem::size_of::<u8>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64(($n) as u64) } }
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
macro_rules! is_signed_type__2 {
() => {
1i32.wrapping_neg() < 1
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

macro_rules! offsetof__23 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, ddc) as u64
};
}

macro_rules! offsetof__46 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, crc_work) as u64
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

unsafe extern "C" {
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __SCK__might_resched: static_call_key;
    static mut cpuinfo_op: seq_operations;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut sighand_cachep: *mut kmem_cache;
    static mut param_ops_int: kernel_param_ops;
    static mut kstat: kernel_stat;
    static mut __drm_debug: u64;
    fn ___ratelimit(arg0: *mut ratelimit_state, arg1: *const u8) -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn memmove(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memcmp(arg0: *const (), arg1: *const (), arg2: u64) -> i32;
    fn sized_strscpy(arg0: *mut u8, arg1: *const u8, arg2: u64) -> i64;
    fn strnlen(arg0: *const u8, arg1: u64) -> u64;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_trylock(arg0: *mut mutex) -> i32;
    fn mutex_unlock(arg0: *mut mutex);
    fn ktime_get() -> i64;
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn flush_work(arg0: *mut work_struct) -> bool;
    fn devm_kmalloc(arg0: *mut device, arg1: u64, arg2: u32) -> *mut ();
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_388();
    fn __compiletime_assert_409();
    fn dev_driver_string(arg0: *const device) -> *const u8;
    fn devm_backlight_device_register(arg0: *mut device, arg1: *const u8, arg2: *mut device, arg3: *mut (), arg4: *const backlight_ops, arg5: *const backlight_properties) -> *mut backlight_device;
    fn usleep_range_state(arg0: u64, arg1: u64, arg2: u32);
    fn acpi_create_platform_device(arg0: *mut acpi_device, arg1: *const property_entry) -> *mut platform_device;
    fn seq_write(arg0: *mut seq_file, arg1: *const (), arg2: u64) -> i32;
    fn seq_printf(arg0: *mut seq_file, arg1: *const u8, ...);
    fn seq_putc(arg0: *mut seq_file, arg1: u8);
    fn __seq_puts(arg0: *mut seq_file, arg1: *const u8);
    #[link_name = "strlen"]
    fn __builtin_strlen(arg0: *const u8) -> u64;
    fn i2c_add_adapter(arg0: *mut i2c_adapter) -> i32;
    fn i2c_del_adapter(arg0: *mut i2c_adapter);
    fn drm_object_property_set_value(arg0: *mut drm_mode_object, arg1: *mut drm_property, arg2: u64) -> i32;
    fn drm_modeset_lock(arg0: *mut drm_modeset_lock, arg1: *mut drm_modeset_acquire_ctx) -> i32;
    fn drm_crtc_add_crc_entry(arg0: *mut drm_crtc, arg1: bool, arg2: u32, arg3: *mut u32) -> i32;
    fn drm_atomic_commit(arg0: *mut drm_atomic_commit) -> i32;
    fn drm_dp_mst_dpcd_read(arg0: *mut drm_dp_aux, arg1: u32, arg2: *mut (), arg3: u64) -> i64;
    fn drm_dp_mst_dpcd_write(arg0: *mut drm_dp_aux, arg1: u32, arg2: *mut (), arg3: u64) -> i64;
    fn drm_display_mode_from_cea_vic(arg0: *mut drm_device, arg1: u8) -> *mut drm_display_mode;
    fn drm_edid_raw(arg0: *const drm_edid) -> *const edid;
    fn drm_printf(arg0: *mut drm_printer, arg1: *const u8, ...);
    fn drm_dev_printk(arg0: *const device, arg1: *const u8, arg2: *const u8, ...);
    fn __drm_dev_dbg(arg0: *mut _ddebug, arg1: *const device, arg2: u32, arg3: *const u8, ...);
    fn drm_crtc_wait_one_vblank(arg0: *mut drm_crtc) -> i32;
    fn __compiletime_assert_687();
    fn __compiletime_assert_692();
    fn __compiletime_assert_694();
    fn __compiletime_assert_727();
    fn __compiletime_assert_733();
    fn __compiletime_assert_734();
    fn __compiletime_assert_737();
    fn __compiletime_assert_804();
    fn __compiletime_assert_805();
    fn __compiletime_assert_809();
    fn __compiletime_assert_810();
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


#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[inline(always)]
unsafe extern "C" fn __WARN_validate_printf(fmt: *const u8, __va_args: ...) {
}

// inhibit tail-call optimization
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
struct range {
    start: u64,
    end: u64,
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

fn __arch_hweight8(w: u32) -> u32 {
__arch_hweight32(w & 255)
}

#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits123U0,
}


#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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
    __slate_bits_0: __slate_bits::__SlateBits130U0,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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
    __slate_bits_0: __slate_bits::__SlateBits131U0,
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
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
struct futex_private_hash {
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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
struct sem_undo_list {
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
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
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
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
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct load_weight {
    weight: u64,
    inv_weight: u32,
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
struct rq {
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
    __slate_bits_0: __slate_bits::__SlateBits287U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
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
    __slate_bits_0: __slate_bits::__SlateBits75U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits75U1,
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
    __slate_bits_2: __slate_bits::__SlateBits75U2,
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
struct rq_flags {
}


fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
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
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
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
struct gendisk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
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

fn queue_work(wq: *mut workqueue_struct, work: *mut work_struct) -> bool {
unsafe { queue_work_on(64i32, wq, work) }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


fn schedule_work(work: *mut work_struct) -> bool {
queue_work(unsafe { system_percpu_wq }, work)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
struct uprobe_xol_ops {
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
struct percpu_counter {
    lock: raw_spinlock,
    count: i64,
    list: list_head,
    counters: *mut i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_701 {
    v: u64,
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
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits415U0,
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
    __slate_bits_0: __slate_bits::__SlateBits465U0,
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
    __slate_bits_0: __slate_bits::__SlateBits467U0,
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
struct kobj_uevent_env {
    argv: [*mut u8; 3],
    envp: [*mut u8; 64],
    envp_idx: i32,
    buf: [u8; 2048],
    buflen: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_subsys_data {
    lock: spinlock,
    refcount: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits492U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits492U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits492U2,
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
    __slate_bits_3: __slate_bits::__SlateBits492U3,
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


fn devm_kzalloc(dev: *mut device, size: u64, gfp: u32) -> *mut () {
unsafe { devm_kmalloc(dev, size, gfp | (BIT!(8i32) as u32)) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
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
struct wake_irq {
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
struct dev_pm_qos {
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
struct folio_queue {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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
struct kmem_cache {
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
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_group_cputimer {
    cputime_atomic: task_cputime_atomic,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
    __slate_bits_0: __slate_bits::__SlateBits290U0,
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
unsafe { __compiletime_assert_388() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*((*(*task).signal).rlim.as_mut_ptr() as *mut rlimit).wrapping_offset(limit as isize)).rlim_cur) as *const u64) };
__v0
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


fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
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
struct rcu_sync {
    gp_state: i32,
    gp_count: i32,
    gp_wait: wait_queue_head,
    cb_head: callback_head,
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
struct tty_struct {
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
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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


fn mnt_idmap(mnt: &vfsmount) -> *mut mnt_idmap {
// Pairs with smp_store_release() in do_idmap_mount().
let mut __v0: *mut mnt_idmap;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_409() };
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
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
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
    __slate_bits_0: __slate_bits::__SlateBits637U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
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
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_file_ref_h_52 {
    refcnt: __SlateRecord_types_h_197,
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
struct request_queue {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct posix_acl {
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
struct io_uring_cmd {
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


// module_param_unsafe - same as module_param but taints kernel
// @name: the variable to alter, and exposed parameter name.
// @type: the type of the parameter
// @perm: visibility in sysfs.
// module_param_named_unsafe - same as module_param_named but taints kernel
// @name: a valid C identifier which is the parameter name.
// @value: the actual lvalue to alter.
// @type: the type of the parameter
// @perm: visibility in sysfs.
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _ddebug {
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
    format: *const u8,
    __slate_bits_0: __slate_bits::__SlateBits695U0,
    key: __SlateRecord_dynamic_debug_h_55,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ddebug_class_map {
    link: list_head,
    r#mod: *mut module,
    mod_name: *const u8,
    class_names: *mut *const u8,
    length: i32,
    base: i32,
    map_type: u32,
}


// DECLARE_DYNDBG_CLASSMAP - declare classnames known by a module
// @_var:   a struct ddebug_class_map, passed to module_param_cb
// @_type:  enum class_map_type, chooses bits/verbose, numeric/symbolic
// @_base:  offset of 1st class-name. splits .class_id space
// @classes: class-names used to control class'd prdbgs
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
union __SlateRecord_dynamic_debug_h_55 {
    dd_key_true: static_key_true,
    dd_key_false: static_key_false,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
struct device_attribute {
    attr: attribute,
    __slate_anon_1: __SlateRecord_device_h_108,
    __slate_anon_2: __SlateRecord_device_h_114,
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
    __slate_bits_0: __slate_bits::__SlateBits489U0,
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
struct backlight_ops {
    options: u32,
    update_status: Option<unsafe extern "C" fn(*mut backlight_device) -> i32>,
    get_brightness: Option<unsafe extern "C" fn(*mut backlight_device) -> i32>,
    controls_device: Option<unsafe extern "C" fn(*mut backlight_device, *mut device) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct backlight_properties {
    brightness: i32,
    max_brightness: i32,
    power: i32,
    r#type: u32,
    state: u32,
    scale: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct backlight_device {
    props: backlight_properties,
    update_lock: mutex,
    ops_lock: mutex,
    ops: *const backlight_ops,
    entry: list_head,
    dev: device,
    use_count: i32,
}


fn backlight_update_status(bd: *mut backlight_device) -> i32 {
let mut ret: i32 = 2i32.wrapping_neg();
unsafe { mutex_lock(core::ptr::addr_of_mut!((*bd).update_lock)) };
if !unsafe { (*bd).ops }.is_null() && (unsafe { (*(*bd).ops).update_status }) != None {
ret = unsafe { unsafe { (*(*bd).ops).update_status }.unwrap()(bd) };
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*bd).update_lock)) };
ret
}

fn backlight_disable(bd: *mut backlight_device) -> i32 {
if bd.is_null() {
return 0;
}
unsafe {
(*bd).props.power = 4;
}
unsafe {
(*bd).props.state = (*bd).props.state | 2u32;
}
backlight_update_status(bd)
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


fn backlight_is_blank(bd: *const backlight_device) -> bool {
(unsafe { (*bd).props.power }) != 0 || (unsafe { (*bd).props.state }) & 3u32 != 0
}

fn backlight_get_brightness(bd: *const backlight_device) -> i32 {
if backlight_is_blank(bd) {
return 0;
} else {
return unsafe { (*bd).props.brightness };
}
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


fn bl_get_data(bl_dev: *mut backlight_device) -> *mut () {
dev_get_drvdata((unsafe { core::ptr::addr_of_mut!((*bl_dev).dev) }) as *const device)
}

fn usleep_range(min: u64, max: u64) {
unsafe { usleep_range_state(min, max, 2u32) };
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
struct __SlateRecord_acpi_bus_h_249 {
    __slate_bits_0: __slate_bits::__SlateBits784U0,
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
struct __SlateRecord_acpi_bus_h_281 {
    __slate_bits_0: __slate_bits::__SlateBits788U0,
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
struct acpi_gpio_mapping {
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
struct platform_device {
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
struct acpi_device_wakeup_flags {
    __slate_bits_0: __slate_bits::__SlateBits790U0,
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
struct cgroup_taskset {
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
struct uid_gid_extent {
    first: u32,
    lower_first: u32,
    count: u32,
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
struct uid_gid_map {
    __slate_anon_0: __SlateRecord_user_namespace_h_26,
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
struct swap_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct regulator {
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
struct irq_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_parent_ops {
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
struct irq_domain_chip_generic {
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
    __slate_bits_0: __slate_bits::__SlateBits829U0,
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
struct drm_master {
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
struct fs_pin {
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
    __slate_bits_0: __slate_bits::__SlateBits606U0,
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
struct drm_connector_helper_funcs {
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
struct drm_encoder {
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
struct drm_privacy_screen {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_edid {
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
struct hdmi_codec_daifmt {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_codec_params {
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
// 1.3
// 1.2
// DPI
// 2.0
// eDP
// XXX 1.2?
// XXX 1.2?
// DP 1.3
// DP 1.3
// DP 1.4
// DP 2.1a, in units of 2 MPixels/sec
// eDP 1.4
// eDP 1.4
// offset 0
// offset 1 for VGA is maximum megapixels per second / 8
//
// offset 1 for DVI/HDMI is maximum TMDS clock in Mbps / 2.5
//
// offset 2 for VGA/DVI/HDMI
// HDMI2.1 PCON FRL CONFIGURATION
// DP-HDMI2.1 PCON DSC ENCODER SUPPORT
// 0x92 through 0x9E
// DP-HDMI2.1 PCON DSC Version
// DP-HDMI2.1 PCON DSC Slice capabilities-1
// DP-HDMI2.1 PCON DSC Slice capabilities-2
// DP-HDMI2.1 PCON HDMI TX Encoder Bits/pixel increment
// DP Extended DSC Capabilities
// DP 1.4a SCR
// 2.0 128b/132b Link Layer
// 2.0 128b/132b Link Layer
// 2.0 128b/132b Link Layer
// DPCD >= 1.2
// 1.4a
// Link/Sink Device Status
// 1.3
// 2.0 E11
// 2.0 E11
// 2.0 E11
// 1.2 MST
// Sink Device-specific
// Branch Device-specific
// Link/Sink Device Power Control
// eDP-specific
// eDP 1.2
// eDP 1.5
// Extended Receiver Capability: See DP_DPCD_REV for definitions
// DP 1.3
// DP 1.3
// 1.4a
// 2.0 E11
// 2.0
// PCON CONFIGURE-1 FRL FOR HDMI SINK
// PCON CONFIGURE-2 FRL FOR HDMI SINK
// PCON HDMI LINK STATUS
// PCON HDMI POST FRL STATUS
// DP 1.3
// PCON Downstream HDMI ERROR Status per Lane
// PCON HDMI CONFIG PPS Override Buffer
// Valid Offsets to be added to Base : 0-127
// PCON HDMI CONFIG PPS Override Parameter: Slice height
// Offset-0 8LSBs of the Slice height.
// Offset-1 8MSBs of the Slice height.
// PCON HDMI CONFIG PPS Override Parameter: Slice width
// Offset-0 8LSBs of the Slice width.
// Offset-1 8MSBs of the Slice width.
// PCON HDMI CONFIG PPS Override Parameter: bits_per_pixel
// Offset-0 8LSBs of the bits_per_pixel.
// Offset-1 2MSBs of the bits_per_pixel.
// LTTPR: Link Training (LT)-tunable PHY Repeaters
// 1.3
// 1.4a
// 1.3
// 1.3
// 1.4a
// 1.4a
// 1.4a
// 1.4a
// Repeater modes
// 1.3
// 1.3
// DSC Capabilities 0x60 through 0x6F
// DP 1.1
// DP 1.2
// DP 1.2
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
struct __SlateRecord_drm_connector_h_1088 {
    avi: drm_connector_hdmi_infoframe,
    hdr_drm: drm_connector_hdmi_infoframe,
    spd: drm_connector_hdmi_infoframe,
    hdmi: drm_connector_hdmi_infoframe,
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
struct drm_writeback_job {
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
struct __SlateRecord_drm_connector_h_2062 {
    lock: mutex,
    audio: drm_connector_hdmi_infoframe,
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
struct drm_event {
    r#type: u32,
    length: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_event_vblank {
    base: drm_event,
    user_data: u64,
    tv_sec: u32,
    tv_usec: u32,
    sequence: u32,
    crtc_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_event_crtc_sequence {
    base: drm_event,
    user_data: u64,
    time_ns: i64,
    sequence: u64,
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
struct cec_adapter {
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
struct drm_format_info {
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
struct drm_framebuffer {
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
struct drm_mode_config_helper_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_driver {
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
struct drm_connector_cec_funcs {
    phys_addr_invalidate: Option<unsafe extern "C" fn(*mut drm_connector)>,
    phys_addr_set: Option<unsafe extern "C" fn(*mut drm_connector, u16)>,
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
struct drm_connector_infoframe_funcs {
    clear_infoframe: Option<unsafe extern "C" fn(*mut drm_connector) -> i32>,
    write_infoframe: Option<unsafe extern "C" fn(*mut drm_connector, *const u8, u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane_helper_funcs {
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
struct drm_crtc_helper_funcs {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_colorop_state {
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


fn drm_dp_is_branch(dpcd: *const u8) -> bool {
((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 1i32 != 0
}

fn drm_dp_sink_supports_dsc(dsc_dpcd: *const u8) -> bool {
((unsafe { *dsc_dpcd.wrapping_offset(0i32 as isize) }) as i32) & 1i32 != 0
}

fn drm_edp_backlight_supported(edp_dpcd: *const u8) -> bool {
((unsafe { *edp_dpcd.wrapping_offset(1 as isize) }) as i32) & 1i32 != 0
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


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dpcd_set_probe")]
extern "C" fn drm_dp_dpcd_set_probe(aux: *mut drm_dp_aux, enable: bool) {
loop {
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_692() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!((*aux).dpcd_probe_disabled, !enable);
if true {
break;
}
}
}

fn drm_dp_dpcd_readb(aux: *mut drm_dp_aux, offset: u32, valuep: *mut u8) -> i64 {
drm_dp_dpcd_read(aux, offset, valuep as *mut (), 1u64)
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


fn drm_dp_dpcd_read_data(aux: *mut drm_dp_aux, offset: u32, buffer: *mut (), size: u64) -> i32 {
let mut ret: i32 = 0;
let buf: *mut u8 = buffer as *mut u8;
ret = drm_dp_dpcd_read(aux, offset, buffer, size) as i32;
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
let ret: i32 = drm_dp_dpcd_write(aux, offset, buffer, size) as i32;
if ret < 0 {
return ret;
}
if (ret as u64) < size {
return 71i32.wrapping_neg();
}
0
}

fn drm_dp_dpcd_writeb(aux: *mut drm_dp_aux, offset: u32, mut value: u8) -> i64 {
drm_dp_dpcd_write(aux, offset, core::ptr::addr_of_mut!(value) as *mut (), 1u64)
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


fn drm_dp_dpcd_read_byte(aux: *mut drm_dp_aux, offset: u32, valuep: *mut u8) -> i32 {
drm_dp_dpcd_read_data(aux, offset, valuep as *mut (), 1u64)
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_edid_h_192 {
    pixel_data: detailed_pixel_timing,
    other_data: detailed_non_pixel,
}


fn drm_dp_dpcd_write_byte(aux: *mut drm_dp_aux, offset: u32, mut value: u8) -> i32 {
drm_dp_dpcd_write_data(aux, offset, core::ptr::addr_of_mut!(value) as *mut (), 1u64)
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_vblank_h_59 {
    base: drm_event,
    vbl: drm_event_vblank,
    seq: drm_event_crtc_sequence,
}


fn drm_dp_has_quirk(desc: *const drm_dp_desc, quirk: u32) -> bool {
((unsafe { (*desc).quirks }) as u64) & BIT__2!(quirk) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_edp_backlight_info {
    pwmgen_bit_count: u8,
    pwm_freq_pre_divider: u8,
    max: u32,
    __slate_bits_0: __slate_bits::__SlateBits1003U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct display_timing {
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
struct ww_mutex {
    base: mutex,
    ctx: *mut ww_acquire_ctx,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_442 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_1979 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_1979_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_1979_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_1979_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_1979_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_1979_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_2144 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_2144_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_2144_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_2144_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_2144_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_2144_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_2410 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_2410_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_dp_helper_c_2462 {
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
struct __SlateRecord_drm_dp_helper_c_2581 {
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
    pub struct __SlateBits287U0 {
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
    pub struct __SlateBits606U0 {
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
    pub struct __SlateBits637U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits492U0 {
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
    pub struct __SlateBits489U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits492U1 {
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
    pub struct __SlateBits492U2 {
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
    pub struct __SlateBits492U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits465U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits467U0 {
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
    pub struct __SlateBits130U0 {
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
    pub struct __SlateBits415U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits75U0 {
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
    pub struct __SlateBits75U1 {
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
    pub struct __SlateBits829U0 {
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
    pub struct __SlateBits290U0 {
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
    pub struct __SlateBits75U2 {
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
    pub struct __SlateBits131U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits979U0 {
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
    pub struct __SlateBits1042U0 {
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
    pub struct __SlateBits1030U0 {
        #[bits(1)]
        pub color_mgmt_changed: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1003U0 {
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
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits695U0 {
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
    __slate_bits_0: __slate_bits::__SlateBits1030U0,
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
    __slate_bits_0: __slate_bits::__SlateBits1042U0,
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
    __slate_bits_0: __slate_bits::__SlateBits979U0,
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


fn fxp_q4_to_int(val_q4: i32) -> i32 {
val_q4 >> 4i32
}

fn fxp_q4_to_frac(val_q4: i32) -> i32 {
val_q4 & 15
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


// 1.4
// 1.4
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
// NOTE: this is deprecated in favor of drm_info() or dev_info().
fn __drm_to_dev(drm: *const drm_device) -> *mut device {
if !drm.is_null() { unsafe { (*drm).dev } } else { core::ptr::null_mut::<device>() }
}

// Helper for struct drm_device based logging.
// NOTE: this is deprecated in favor of drm_dbg_kms(NULL, ...).
// Helper for struct drm_device based WARNs
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
struct drm_prime_file_private {
    lock: mutex,
    dmabufs: rb_root,
    handles: rb_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_minor {
    index: i32,
    r#type: i32,
    kdev: *mut device,
    dev: *mut drm_device,
    debugfs_symlink: *mut dentry,
    debugfs_root: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_pending_event {
    completion: *mut completion,
    completion_release: Option<unsafe extern "C" fn(*mut completion)>,
    event: *mut drm_event,
    fence: *mut dma_fence,
    file_priv: *mut drm_file,
    link: list_head,
    pending_link: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_file {
    authenticated: bool,
    stereo_allowed: bool,
    universal_planes: bool,
    atomic: bool,
    aspect_ratio_allowed: bool,
    writeback_connectors: bool,
    plane_color_pipeline: bool,
    was_master: bool,
    is_master: bool,
    supports_virtualized_cursor_plane: bool,
    master: *mut drm_master,
    master_lookup_lock: spinlock,
    pid: *mut pid,
    client_id: u64,
    magic: u32,
    lhead: list_head,
    minor: *mut drm_minor,
    object_idr: idr,
    table_lock: spinlock,
    syncobj_xa: xarray,
    filp: *mut file,
    driver_priv: *mut (),
    fbs: list_head,
    fbs_lock: mutex,
    blobs: list_head,
    event_wait: wait_queue_head,
    pending_event_list: list_head,
    event_list: list_head,
    event_space: i32,
    event_read_lock: mutex,
    prime: drm_prime_file_private,
    client_name: *const u8,
    client_name_lock: mutex,
    debugfs_client: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_pending_vblank_event {
    base: drm_pending_event,
    pipe: u32,
    sequence: u64,
    event: __SlateRecord_drm_vblank_h_59,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vblank_crtc_config {
    offdelay_ms: i32,
    disable_immediate: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vblank_crtc_timer {
    timer: hrtimer,
    interval_lock: spinlock,
    interval: i64,
    crtc: *mut drm_crtc,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vblank_crtc {
    dev: *mut drm_device,
    queue: wait_queue_head,
    disable_timer: timer_list,
    seqlock: seqlock,
    count: __SlateRecord_types_h_197,
    time: i64,
    refcount: __SlateRecord_types_h_190,
    last: u32,
    max_vblank_count: u32,
    inmodeset: u32,
    pipe: u32,
    framedur_ns: i32,
    linedur_ns: i32,
    hwmode: drm_display_mode,
    config: drm_vblank_crtc_config,
    enabled: bool,
    worker: *mut kthread_worker,
    pending_work: list_head,
    work_wait_queue: wait_queue_head,
    vblank_timer: drm_vblank_crtc_timer,
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


fn drm_dp_aux_register_devnode(aux: *mut drm_dp_aux) -> i32 {
0
}

fn drm_dp_aux_unregister_devnode(aux: *mut drm_dp_aux) {
}

static mut drm_debug_classes_classnames: __SlateAlign16<[*const u8; 10]> = __SlateAlign16([(b"DRM_UT_CORE\0".as_ptr() as *mut u8) as *const u8, (b"DRM_UT_DRIVER\0".as_ptr() as *mut u8) as *const u8, (b"DRM_UT_KMS\0".as_ptr() as *mut u8) as *const u8, (b"DRM_UT_PRIME\0".as_ptr() as *mut u8) as *const u8, (b"DRM_UT_ATOMIC\0".as_ptr() as *mut u8) as *const u8, (b"DRM_UT_VBL\0".as_ptr() as *mut u8) as *const u8, (b"DRM_UT_STATE\0".as_ptr() as *mut u8) as *const u8, (b"DRM_UT_LEASE\0".as_ptr() as *mut u8) as *const u8, (b"DRM_UT_DP\0".as_ptr() as *mut u8) as *const u8, (b"DRM_UT_DRMRES\0".as_ptr() as *mut u8) as *const u8]);

#[used(compiler)]
#[unsafe(link_section = "__dyndbg_classes")]
static mut drm_debug_classes: ddebug_class_map = {
let mut __t0: ddebug_class_map = unsafe { core::mem::zeroed() };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.mod_name = (b"drm_display_helper\0".as_ptr() as *mut u8) as *const u8;
__t0.class_names = unsafe { core::ptr::addr_of_mut!(drm_debug_classes_classnames.0) as *mut *const u8 };
__t0.length = ((core::mem::size_of::<[*mut u8; 10]>() as u64) / (core::mem::size_of::<*mut u8>() as u64)) as i32;
__t0.base = 0i32;
__t0.map_type = 0u32;
__t0
};

#[repr(C)]
#[derive(Clone, Copy)]
struct dp_aux_backlight {
    base: *mut backlight_device,
    aux: *mut drm_dp_aux,
    info: drm_edp_backlight_info,
    enabled: bool,
}


// DOC: dp helpers
//
// These functions contain some common logic and helpers at various abstraction
// levels to deal with Display Port sink devices and related things like DP aux
// channel transfers, EDID reading over DP aux channels, decoding certain DPCD
// blocks, ...
/// Helpers for DP link training
fn dp_link_status(link_status: *const u8, r: i32) -> u8 {
unsafe { *link_status.wrapping_offset(r.wrapping_sub(514) as isize) }
}

fn dp_get_lane_status(link_status: *const u8, lane: i32) -> u8 {
let i: i32 = 514i32.wrapping_add(lane >> 1i32);
let s: i32 = (lane & 1).wrapping_mul(4);
let l: u8 = dp_link_status(link_status, i);
((l as i32) >> s & 15) as u8
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_channel_eq_ok")]
extern "C" fn drm_dp_channel_eq_ok(link_status: *const u8, lane_count: i32) -> bool {
let lane_align: u8 = dp_link_status(link_status, 516i32);
let mut lane_status: u8 = 0;
if (lane_align as i32) & 1i32 == 0 {
return false;
}
for lane in 0..lane_count {
lane_status = dp_get_lane_status(link_status, lane);
if (lane_status as i32) & 7i32 != 7i32 {
return false;
}
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_channel_eq_ok_661: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, i32) -> bool>, *mut ()>(Some(drm_dp_channel_eq_ok)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_clock_recovery_ok")]
extern "C" fn drm_dp_clock_recovery_ok(link_status: *const u8, lane_count: i32) -> bool {
let mut lane_status: u8 = 0;
for lane in 0..lane_count {
lane_status = dp_get_lane_status(link_status, lane);
if (lane_status as i32) & 1i32 == 0 {
return false;
}
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_clock_recovery_ok_662: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, i32) -> bool>, *mut ()>(Some(drm_dp_clock_recovery_ok)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_post_lt_adj_req_in_progress")]
extern "C" fn drm_dp_post_lt_adj_req_in_progress(link_status: *const u8) -> bool {
let lane_align: u8 = dp_link_status(link_status, 516i32);
(lane_align as i32) & 2i32 != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_post_lt_adj_req_in_progress_663: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> bool>, *mut ()>(Some(drm_dp_post_lt_adj_req_in_progress)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_get_adjust_request_voltage")]
extern "C" fn drm_dp_get_adjust_request_voltage(link_status: *const u8, lane: i32) -> u8 {
let i: i32 = 518i32.wrapping_add(lane >> 1i32);
let s: i32 = if lane & 1 != 0 { 4 } else { 0 };
let l: u8 = dp_link_status(link_status, i);
(((l as i32) >> s & 3) << 0i32) as u8
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_get_adjust_request_voltage_664: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, i32) -> u8>, *mut ()>(Some(drm_dp_get_adjust_request_voltage)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_get_adjust_request_pre_emphasis")]
extern "C" fn drm_dp_get_adjust_request_pre_emphasis(link_status: *const u8, lane: i32) -> u8 {
let i: i32 = 518i32.wrapping_add(lane >> 1i32);
let s: i32 = if lane & 1 != 0 { 6 } else { 2 };
let l: u8 = dp_link_status(link_status, i);
(((l as i32) >> s & 3) << 3i32) as u8
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_get_adjust_request_pre_emphasis_665: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, i32) -> u8>, *mut ()>(Some(drm_dp_get_adjust_request_pre_emphasis)) };

/// DP 2.0 128b/132b
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_get_adjust_tx_ffe_preset")]
extern "C" fn drm_dp_get_adjust_tx_ffe_preset(link_status: *const u8, lane: i32) -> u8 {
let i: i32 = 518i32.wrapping_add(lane >> 1i32);
let s: i32 = if lane & 1 != 0 { 4 } else { 0 };
let l: u8 = dp_link_status(link_status, i);
((l as i32) >> s & 15) as u8
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_get_adjust_tx_ffe_preset_666: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, i32) -> u8>, *mut ()>(Some(drm_dp_get_adjust_tx_ffe_preset)) };

/// DP 2.0 errata for 128b/132b
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_128b132b_lane_channel_eq_done")]
extern "C" fn drm_dp_128b132b_lane_channel_eq_done(link_status: *const u8, lane_count: i32) -> bool {
let lane_align: u8 = dp_link_status(link_status, 516i32);
let mut lane_status: u8 = 0;
if (lane_align as i32) & 1i32 == 0 {
return false;
}
for lane in 0..lane_count {
lane_status = dp_get_lane_status(link_status, lane);
if (lane_status as i32) & 2i32 == 0 {
return false;
}
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_128b132b_lane_channel_eq_done_667: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, i32) -> bool>, *mut ()>(Some(drm_dp_128b132b_lane_channel_eq_done)) };

/// DP 2.0 errata for 128b/132b
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_128b132b_lane_symbol_locked")]
extern "C" fn drm_dp_128b132b_lane_symbol_locked(link_status: *const u8, lane_count: i32) -> bool {
let mut lane_status: u8 = 0;
for lane in 0..lane_count {
lane_status = dp_get_lane_status(link_status, lane);
if (lane_status as i32) & 4i32 == 0 {
return false;
}
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_128b132b_lane_symbol_locked_668: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, i32) -> bool>, *mut ()>(Some(drm_dp_128b132b_lane_symbol_locked)) };

/// DP 2.0 errata for 128b/132b
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_128b132b_eq_interlane_align_done")]
extern "C" fn drm_dp_128b132b_eq_interlane_align_done(link_status: *const u8) -> bool {
let status: u8 = dp_link_status(link_status, 516i32);
(status as i32) & 4i32 != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_128b132b_eq_interlane_align_done_669: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> bool>, *mut ()>(Some(drm_dp_128b132b_eq_interlane_align_done)) };

/// DP 2.0 errata for 128b/132b
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_128b132b_cds_interlane_align_done")]
extern "C" fn drm_dp_128b132b_cds_interlane_align_done(link_status: *const u8) -> bool {
let status: u8 = dp_link_status(link_status, 516i32);
(status as i32) & 8i32 != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_128b132b_cds_interlane_align_done_670: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> bool>, *mut ()>(Some(drm_dp_128b132b_cds_interlane_align_done)) };

/// DP 2.0 errata for 128b/132b
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_128b132b_link_training_failed")]
extern "C" fn drm_dp_128b132b_link_training_failed(link_status: *const u8) -> bool {
let status: u8 = dp_link_status(link_status, 516i32);
(status as i32) & 16i32 != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_128b132b_link_training_failed_671: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> bool>, *mut ()>(Some(drm_dp_128b132b_link_training_failed)) };

#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.__8b10b_clock_recovery_delay_us")]
extern "C" fn __8b10b_clock_recovery_delay_us(aux: *const drm_dp_aux, rd_interval: u8) -> i32 {
if (rd_interval as i32) > 4 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: invalid AUX interval 0x%02x (max 4)\n".as_ptr() as *const u8, (*aux).name, rd_interval as i32) };
}
if (rd_interval as i32) == 0 {
return 100;
}
((rd_interval as i32).wrapping_mul(4) as i64).wrapping_mul(1000) as i32
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.__8b10b_channel_eq_delay_us")]
extern "C" fn __8b10b_channel_eq_delay_us(aux: *const drm_dp_aux, rd_interval: u8) -> i32 {
if (rd_interval as i32) > 4 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: invalid AUX interval 0x%02x (max 4)\n".as_ptr() as *const u8, (*aux).name, rd_interval as i32) };
}
if (rd_interval as i32) == 0 {
return 400;
}
((rd_interval as i32).wrapping_mul(4) as i64).wrapping_mul(1000) as i32
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.__128b132b_channel_eq_delay_us")]
extern "C" fn __128b132b_channel_eq_delay_us(aux: *const drm_dp_aux, rd_interval: u8) -> i32 {
match rd_interval as i32 {
0 => {
return 400;
}
1 => {
return 4000;
}
2 => {
return 8000;
}
3 => {
return 12000;
}
4 => {
return 16000;
}
5 => {
return 32000;
}
6 => {
return 64000;
}
_ => {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: invalid AUX interval 0x%02x\n".as_ptr() as *const u8, (*aux).name, rd_interval as i32) };
return 400;
}
}
unsafe { core::mem::zeroed() }
}

/// The link training delays are different for:
///
///  - Clock recovery vs. channel equalization
///  - DPRX vs. LTTPR
///  - 128b/132b vs. 8b/10b
///  - DPCD rev 1.3 vs. later
///
/// Get the correct delay in us, reading DPCD if necessary.
fn __read_delay(aux: *mut drm_dp_aux, dpcd: *const u8, dp_phy: u32, uhbr: bool, cr: bool) -> i32 {
let mut parse: Option<unsafe extern "C" fn(*const drm_dp_aux, u8) -> i32> = unsafe { core::mem::zeroed() };
let mut offset: u32 = 0;
let mut rd_interval: u8 = 0;
let mut mask: u8 = 0;
if (dp_phy as u32) == 0 {
if uhbr {
if cr {
return 100;
}
offset = 8726;
mask = 127;
parse = Some(__128b132b_channel_eq_delay_us);
} else {
if cr && ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) >= 20 {
return 100;
}
offset = 14;
mask = 127;
if cr {
parse = Some(__8b10b_clock_recovery_delay_us);
} else {
parse = Some(__8b10b_channel_eq_delay_us);
}
}
} else {
if uhbr {
offset = DP_128B132B_TRAINING_AUX_RD_INTERVAL_PHY_REPEATER!(dp_phy);
mask = 127;
parse = Some(__128b132b_channel_eq_delay_us);
} else {
if cr {
return 100;
}
offset = DP_TRAINING_AUX_RD_INTERVAL_PHY_REPEATER!(dp_phy);
mask = 127;
parse = Some(__8b10b_channel_eq_delay_us);
}
}
if offset < 15 {
rd_interval = unsafe { *dpcd.wrapping_offset(offset as isize) };
} else {
if drm_dp_dpcd_read_byte(aux, offset, core::ptr::addr_of_mut!(rd_interval)) < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: failed rd interval read\n".as_ptr() as *const u8, (*aux).name) };
// arbitrary default delay
return 400;
}
}
unsafe { parse.unwrap()(aux as *const drm_dp_aux, ((rd_interval as i32) & (mask as i32)) as u8) }
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_read_clock_recovery_delay")]
extern "C" fn drm_dp_read_clock_recovery_delay(aux: *mut drm_dp_aux, dpcd: *const u8, dp_phy: u32, uhbr: bool) -> i32 {
__read_delay(aux, dpcd, dp_phy, uhbr, true)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_read_clock_recovery_delay_672: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *const u8, u32, bool) -> i32>, *mut ()>(Some(drm_dp_read_clock_recovery_delay)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_read_channel_eq_delay")]
extern "C" fn drm_dp_read_channel_eq_delay(aux: *mut drm_dp_aux, dpcd: *const u8, dp_phy: u32, uhbr: bool) -> i32 {
__read_delay(aux, dpcd, dp_phy, uhbr, false)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_read_channel_eq_delay_673: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *const u8, u32, bool) -> i32>, *mut ()>(Some(drm_dp_read_channel_eq_delay)) };

/// Per DP 2.0 Errata
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_128b132b_read_aux_rd_interval")]
extern "C" fn drm_dp_128b132b_read_aux_rd_interval(aux: *mut drm_dp_aux) -> i32 {
let mut unit: i32 = 0;
let mut val: u8 = 0;
if drm_dp_dpcd_read_byte(aux, 8726u32, core::ptr::addr_of_mut!(val)) < 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, c"[drm] *ERROR* %s: failed rd interval read\n".as_ptr() as *const u8, (*aux).name) };
// default to max
val = 127;
}
unit = if (val as i32) & 128i32 != 0 { 1 } else { 2 };
val = ((val as i32) & 127) as u8;
(val as i32).wrapping_add(1).wrapping_mul(unit).wrapping_mul(1000)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_128b132b_read_aux_rd_interval_674: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux) -> i32>, *mut ()>(Some(drm_dp_128b132b_read_aux_rd_interval)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_link_train_clock_recovery_delay")]
extern "C" fn drm_dp_link_train_clock_recovery_delay(aux: *const drm_dp_aux, dpcd: *const u8) {
let rd_interval: u8 = (((unsafe { *dpcd.wrapping_offset(14 as isize) }) as i32) & 127) as u8;
let delay_us: i32 = if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) >= 20 {
100
} else {
__8b10b_clock_recovery_delay_us(aux, rd_interval)
};
usleep_range(delay_us as u64, delay_us.wrapping_mul(2) as u64);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_link_train_clock_recovery_delay_675: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_dp_aux, *const u8)>, *mut ()>(Some(drm_dp_link_train_clock_recovery_delay)) };

fn __drm_dp_link_train_channel_eq_delay(aux: *const drm_dp_aux, rd_interval: u8) {
let delay_us: i32 = __8b10b_channel_eq_delay_us(aux, rd_interval);
usleep_range(delay_us as u64, delay_us.wrapping_mul(2) as u64);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_link_train_channel_eq_delay")]
extern "C" fn drm_dp_link_train_channel_eq_delay(aux: *const drm_dp_aux, dpcd: *const u8) {
__drm_dp_link_train_channel_eq_delay(aux, (((unsafe { *dpcd.wrapping_offset(14 as isize) }) as i32) & 127) as u8);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_link_train_channel_eq_delay_676: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_dp_aux, *const u8)>, *mut ()>(Some(drm_dp_link_train_channel_eq_delay)) };

/// drm_dp_phy_name() - Get the name of the given DP PHY
/// @dp_phy: The DP PHY identifier
///
/// Given the @dp_phy, get a user friendly name of the DP PHY, either "DPRX" or
/// "LTTPR <N>", or "<INVALID DP PHY>" on errors. The returned string is always
/// non-NULL and valid.
///
/// Returns: Name of the DP PHY.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_phy_name")]
extern "C" fn drm_dp_phy_name(dp_phy: u32) -> *const u8 {
let __v0: bool = if (dp_phy as u32) < 0 || ((dp_phy as u32) as u64) >= ARRAY_SIZE__7!() {
true
} else {
let mut __v1: i64;
let __ret_warn_on: i32 = unsafe { phy_names.0[(dp_phy as u32) as usize] }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("677: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 677b - ., 3; .popsection\n# {0}", const 677 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_dp_phy_name_2, sym __slate_asm_string_drm_dp_phy_name_3, const 443 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("678: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 678b - ., 4; .popsection\n# {0}", const 678 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
__v1 != 0
};
if __v0 {
return c"<INVALID DP PHY>".as_ptr() as *const u8;
}
unsafe { phy_names.0[(dp_phy as u32) as usize] }
}

static phy_names: __SlateAlign16<[*const u8; 9]> = __SlateAlign16([(b"DPRX\0".as_ptr() as *mut u8) as *const u8, (b"LTTPR 1\0".as_ptr() as *mut u8) as *const u8, (b"LTTPR 2\0".as_ptr() as *mut u8) as *const u8, (b"LTTPR 3\0".as_ptr() as *mut u8) as *const u8, (b"LTTPR 4\0".as_ptr() as *mut u8) as *const u8, (b"LTTPR 5\0".as_ptr() as *mut u8) as *const u8, (b"LTTPR 6\0".as_ptr() as *mut u8) as *const u8, (b"LTTPR 7\0".as_ptr() as *mut u8) as *const u8, (b"LTTPR 8\0".as_ptr() as *mut u8) as *const u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_phy_name_679: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32) -> *const u8>, *mut ()>(Some(drm_dp_phy_name)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_lttpr_link_train_clock_recovery_delay")]
extern "C" fn drm_dp_lttpr_link_train_clock_recovery_delay() {
usleep_range(100u64, 200u64);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_lttpr_link_train_clock_recovery_delay_680: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn()>, *mut ()>(Some(drm_dp_lttpr_link_train_clock_recovery_delay)) };

fn dp_lttpr_phy_cap(phy_cap: *const u8, r: i32) -> u8 {
unsafe { *phy_cap.wrapping_offset(r.wrapping_sub(983072) as isize) }
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_lttpr_link_train_channel_eq_delay")]
extern "C" fn drm_dp_lttpr_link_train_channel_eq_delay(aux: *const drm_dp_aux, phy_cap: *const u8) {
let interval: u8 = ((dp_lttpr_phy_cap(phy_cap, 983072i32) as i32) & 127) as u8;
__drm_dp_link_train_channel_eq_delay(aux, interval);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_lttpr_link_train_channel_eq_delay_681: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_dp_aux, *const u8)>, *mut ()>(Some(drm_dp_lttpr_link_train_channel_eq_delay)) };

/// drm_dp_lttpr_wake_timeout_setup() - Grant extended time for sink to wake up
/// @aux: The DP AUX channel to use
/// @transparent_mode: This is true if lttpr is in transparent mode
///
/// This function checks if the sink needs any extended wake time, if it does
/// it grants this request. Post this setup the source device can keep trying
/// the Aux transaction till the granted wake timeout.
/// If this function is not called all Aux transactions are expected to take
/// a default of 1ms before they throw an error.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_lttpr_wake_timeout_setup")]
extern "C" fn drm_dp_lttpr_wake_timeout_setup(aux: *mut drm_dp_aux, transparent_mode: bool) {
let mut val: u8 = 1;
let mut ret: i32 = 0;
if transparent_mode {
ret = drm_dp_dpcd_readb(aux, 8721u32, core::ptr::addr_of_mut!(val)) as i32;
if ret != 1 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"Failed to read Extended sleep wake timeout request\n".as_ptr() as *const u8) };
return;
}
val = (if ((val as i32) as u64) < (core::mem::size_of::<[u8; 6]>() as u64) && (unsafe { timeout_mapping[(val as i32) as usize] }) != 0 { (unsafe { timeout_mapping[(val as i32) as usize] }) as i32 } else { 1i32 }) as u8;
if (val as i32) > 1 {
drm_dp_dpcd_writeb(aux, 281u32, 1u8);
}
} else {
ret = drm_dp_dpcd_readb(aux, 983045u32, core::ptr::addr_of_mut!(val)) as i32;
if ret != 1 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"Failed to read Extended sleep wake timeout request\n".as_ptr() as *const u8) };
return;
}
val = (if (val as i32) & 127 != 0 { ((val as i32) & 127).wrapping_mul(10) } else { 1i32 }) as u8;
if (val as i32) > 1 {
drm_dp_dpcd_writeb(aux, 983045u32, 128u8);
}
}
}

static timeout_mapping: [u8; 6] = [1, 20, 40, 60, 80, 100];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_lttpr_wake_timeout_setup_682: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, bool)>, *mut ()>(Some(drm_dp_lttpr_wake_timeout_setup)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_link_rate_to_bw_code")]
extern "C" fn drm_dp_link_rate_to_bw_code(link_rate: i32) -> u8 {
match link_rate {
1000000 => {
return 1;
}
1350000 => {
return 4;
}
2000000 => {
return 2;
}
_ => {
return (link_rate / 27000) as u8; // Spec says link_bw = link_rate / 0.27Gbps
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_link_rate_to_bw_code_683: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32) -> u8>, *mut ()>(Some(drm_dp_link_rate_to_bw_code)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_bw_code_to_link_rate")]
extern "C" fn drm_dp_bw_code_to_link_rate(link_bw: u8) -> i32 {
match link_bw as i32 {
1 => {
return 1000000;
}
4 => {
return 1350000;
}
2 => {
return 2000000;
}
_ => {
return (link_bw as i32).wrapping_mul(27000); // Spec says link_rate = link_bw * 0.27Gbps
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_bw_code_to_link_rate_684: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u8) -> i32>, *mut ()>(Some(drm_dp_bw_code_to_link_rate)) };

// us
fn drm_dp_dump_access(aux: *const drm_dp_aux, request: u8, offset: u32, buffer: *mut (), ret: i32) {
let arrow: *const u8 = (if (request as i32) == 9 { c"->".as_ptr() as *mut u8 } else { c"<-".as_ptr() as *mut u8 }) as *const u8;
if ret > 0 {
let mut __v0: i32;
let __UNIQUE_ID_x__685: i32 = ret;
let __UNIQUE_ID_y__686: i32 = 20;
loop {
if __types_ok!(__UNIQUE_ID_x__685, __UNIQUE_ID_y__686) == 0 {
unsafe { __compiletime_assert_687() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__685 < __UNIQUE_ID_y__686 { __UNIQUE_ID_x__685 } else { __UNIQUE_ID_y__686 };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 8u32, c"%s: 0x%05x AUX %s (ret=%3d) %*ph\n".as_ptr() as *const u8, (*aux).name, offset, arrow, ret, __v0, buffer) };
} else {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 8u32, c"%s: 0x%05x AUX %s (ret=%3d)\n".as_ptr() as *const u8, (*aux).name, offset, arrow, ret) };
}
}

// DOC: dp helpers
//
// The DisplayPort AUX channel is an abstraction to allow generic, driver-
// independent access to AUX functionality. Drivers can take advantage of
// this by filling in the fields of the drm_dp_aux structure.
//
// Transactions are described using a hardware-independent drm_dp_aux_msg
// structure, which is passed into a driver's .transfer() implementation.
// Both native and I2C-over-AUX transactions are supported.
fn drm_dp_dpcd_access(aux: *mut drm_dp_aux, request: u8, offset: u32, buffer: *mut (), size: u64) -> i32 {
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut native_reply: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(native_reply) as *mut u32;
let mut retry: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u32 = core::ptr::addr_of_mut!(retry) as *mut u32;
let mut msg: core::mem::MaybeUninit<drm_dp_aux_msg> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut drm_dp_aux_msg = core::ptr::addr_of_mut!(msg) as *mut drm_dp_aux_msg;
unsafe {
'__join_0: {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
err = 0;
ret = 0;
memset(core::ptr::addr_of_mut!(*__slate_slot_6) as *mut (), 0i32, core::mem::size_of::<drm_dp_aux_msg>() as u64);
(*__slate_slot_6).address = offset;
(*__slate_slot_6).request = request;
(*__slate_slot_6).buffer = buffer;
(*__slate_slot_6).size = size;
mutex_lock(core::ptr::addr_of_mut!((*aux).hw_mutex));
// If the device attached to the aux bus is powered down then there's
// no reason to attempt a transfer. Error out immediately.
if (*aux).powered_down {
ret = 16i32.wrapping_neg();
} else {
// The specification doesn't give any recommendation on how often to
// retry native transactions. We used to retry 7 times like for
// aux i2c transactions but real world devices this wasn't
// sufficient, bump to 32 which makes Dell 4k monitors happier.
*__slate_slot_5 = 0;
while *__slate_slot_5 < 32 {
if ret != 0 && ret != 110i32.wrapping_neg() {
usleep_range(500u64, 600u64);
}
ret = unsafe { (*aux).transfer }.unwrap()(aux, core::ptr::addr_of_mut!(*__slate_slot_6)) as i32;
if ret >= 0 {
*__slate_slot_4 = (((*__slate_slot_6).reply as i32) & 3i32) as u32;
if *__slate_slot_4 == 0u32 {
if (ret as u64) == size {
break '__join_0;
} else {
ret = 71i32.wrapping_neg();
}
} else {
ret = 5i32.wrapping_neg();
}
}
// We want the error we return to be the error we received on
// the first transaction, since we may get a different error the
// next time we retry
if err == 0 {
err = ret;
}
let __v1: u32 = *__slate_slot_5;
*__slate_slot_5 = __v1.wrapping_add(1);
}
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Too many retries, giving up. First error: %d\n".as_ptr() as *const u8, (*aux).name, err);
ret = err;
}
}
mutex_unlock(core::ptr::addr_of_mut!((*aux).hw_mutex));
return ret;
}
}

/// drm_dp_dpcd_probe() - probe a given DPCD address with a 1-byte read access
/// @aux: DisplayPort AUX channel (SST)
/// @offset: address of the register to probe
///
/// Probe the provided DPCD address by reading 1 byte from it. The function can
/// be used to trigger some side-effect the read access has, like waking up the
/// sink, without the need for the read-out value.
///
/// Returns 0 if the read access suceeded, or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dpcd_probe")]
extern "C" fn drm_dp_dpcd_probe(aux: *mut drm_dp_aux, offset: u32) -> i32 {
let mut buffer: u8 = 0;
let ret: i32 = drm_dp_dpcd_access(aux, 9u8, offset, core::ptr::addr_of_mut!(buffer) as *mut (), 1u64);
let mut __v0: i64;
let __ret_warn_on: i32 = (ret == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("688: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 688b - ., 3; .popsection\n# {0}", const 688 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_dp_dpcd_probe_1, sym __slate_asm_string_drm_dp_dpcd_probe_2, const 673 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
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
drm_dp_dump_access(aux as *const drm_dp_aux, 9u8, offset, core::ptr::addr_of_mut!(buffer) as *mut (), ret);
if ret < 0 { ret } else { 0 }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dpcd_probe_690: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, u32) -> i32>, *mut ()>(Some(drm_dp_dpcd_probe)) };

/// drm_dp_dpcd_set_powered() - Set whether the DP device is powered
/// @aux: DisplayPort AUX channel; for convenience it's OK to pass NULL here
///       and the function will be a no-op.
/// @powered: true if powered; false if not
///
/// If the endpoint device on the DP AUX bus is known to be powered down
/// then this function can be called to make future transfers fail immediately
/// instead of needing to time out.
///
/// If this function is never called then a device defaults to being powered.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dpcd_set_powered")]
extern "C" fn drm_dp_dpcd_set_powered(aux: *mut drm_dp_aux, powered: bool) {
if aux.is_null() {
return;
}
unsafe { mutex_lock(core::ptr::addr_of_mut!((*aux).hw_mutex)) };
unsafe {
(*aux).powered_down = !powered;
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*aux).hw_mutex)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dpcd_set_powered_691: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, bool)>, *mut ()>(Some(drm_dp_dpcd_set_powered)) };

// drm_dp_dpcd_set_probe() - Set whether a probing before DPCD access is done
// @aux: DisplayPort AUX channel
// @enable: Enable the probing if required
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dpcd_set_probe_693: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, bool)>, *mut ()>(Some(drm_dp_dpcd_set_probe)) };

fn dpcd_access_needs_probe(aux: *mut drm_dp_aux) -> bool {
// HP ZR24w corrupts the first DPCD access after entering power save
// mode. Eg. on a read, the entire buffer will be filled with the same
// byte. Do a throw away read to avoid corrupting anything we care
// about. Afterwards things will work correctly until the monitor
// gets woken up and subsequently re-enters power save mode.
//
// The user pressing any button on the monitor is enough to wake it
// up, so there is no particularly good place to do the workaround.
// We just have to do it before any DPCD access and hope that the
// monitor doesn't power down exactly after the throw away read.
let __v0: bool = if !unsafe { (*aux).is_remote } {
let mut __v1: bool;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_694() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*aux).dpcd_probe_disabled) as *const bool) };
!__v1
} else {
false
};
__v0
}

/// drm_dp_dpcd_read() - read a series of bytes from the DPCD
/// @aux: DisplayPort AUX channel (SST or MST)
/// @offset: address of the (first) register to read
/// @buffer: buffer to store the register values
/// @size: number of bytes in @buffer
///
/// Returns the number of bytes transferred on success, or a negative error
/// code on failure. -EIO is returned if the request was NAKed by the sink or
/// if the retry count was exceeded. If not all bytes were transferred, this
/// function returns -EPROTO. Errors from the underlying AUX channel transfer
/// function, with the exception of -EBUSY (which causes the transaction to
/// be retried), are propagated to the caller.
///
/// In most of the cases you want to use drm_dp_dpcd_read_data() instead.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dpcd_read")]
extern "C" fn drm_dp_dpcd_read(aux: *mut drm_dp_aux, offset: u32, buffer: *mut (), size: u64) -> i64 {
let mut ret: i32 = 0;
if dpcd_access_needs_probe(aux) {
ret = drm_dp_dpcd_probe(aux, 258u32);
if ret < 0 {
return ret as i64;
}
}
if unsafe { (*aux).is_remote } {
ret = (unsafe { drm_dp_mst_dpcd_read(aux, offset, buffer, size) }) as i32;
} else {
ret = drm_dp_dpcd_access(aux, 9u8, offset, buffer, size);
}
drm_dp_dump_access(aux as *const drm_dp_aux, 9u8, offset, buffer, ret);
ret as i64
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dpcd_read_695: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, u32, *mut (), u64) -> i64>, *mut ()>(Some(drm_dp_dpcd_read)) };

/// drm_dp_dpcd_write() - write a series of bytes to the DPCD
/// @aux: DisplayPort AUX channel (SST or MST)
/// @offset: address of the (first) register to write
/// @buffer: buffer containing the values to write
/// @size: number of bytes in @buffer
///
/// Returns the number of bytes transferred on success, or a negative error
/// code on failure. -EIO is returned if the request was NAKed by the sink or
/// if the retry count was exceeded. If not all bytes were transferred, this
/// function returns -EPROTO. Errors from the underlying AUX channel transfer
/// function, with the exception of -EBUSY (which causes the transaction to
/// be retried), are propagated to the caller.
///
/// In most of the cases you want to use drm_dp_dpcd_write_data() instead.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dpcd_write")]
extern "C" fn drm_dp_dpcd_write(aux: *mut drm_dp_aux, offset: u32, buffer: *mut (), size: u64) -> i64 {
let ret: i32 = if unsafe { (*aux).is_remote } {
(unsafe { drm_dp_mst_dpcd_write(aux, offset, buffer, size) }) as i32
} else {
drm_dp_dpcd_access(aux, 8u8, offset, buffer, size)
};
drm_dp_dump_access(aux as *const drm_dp_aux, 8u8, offset, buffer, ret);
ret as i64
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dpcd_write_696: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, u32, *mut (), u64) -> i64>, *mut ()>(Some(drm_dp_dpcd_write)) };

/// drm_dp_dpcd_read_link_status() - read DPCD link status (bytes 0x202-0x207)
/// @aux: DisplayPort AUX channel
/// @status: buffer to store the link status in (must be at least 6 bytes)
///
/// Returns a negative error code on failure or 0 on success.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dpcd_read_link_status")]
extern "C" fn drm_dp_dpcd_read_link_status(aux: *mut drm_dp_aux, status: *mut u8) -> i32 {
drm_dp_dpcd_read_data(aux, 514u32, status as *mut (), 6u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dpcd_read_link_status_697: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *mut u8) -> i32>, *mut ()>(Some(drm_dp_dpcd_read_link_status)) };

/// drm_dp_dpcd_read_phy_link_status - get the link status information for a DP PHY
/// @aux: DisplayPort AUX channel
/// @dp_phy: the DP PHY to get the link status for
/// @link_status: buffer to return the status in
///
/// Fetch the AUX DPCD registers for the DPRX or an LTTPR PHY link status. The
/// layout of the returned @link_status matches the DPCD register layout of the
/// DPRX PHY link status.
///
/// Returns 0 if the information was read successfully or a negative error code
/// on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dpcd_read_phy_link_status")]
extern "C" fn drm_dp_dpcd_read_phy_link_status(aux: *mut drm_dp_aux, dp_phy: u32, link_status: *mut u8) -> i32 {
let mut ret: i32 = 0;
if (dp_phy as u32) == 0 {
return drm_dp_dpcd_read_data(aux, 514u32, link_status as *mut (), 6u64);
}
ret = drm_dp_dpcd_read_data(aux, DP_LANE0_1_STATUS_PHY_REPEATER!(dp_phy), link_status as *mut (), 5u64);
if ret < 0 {
return ret;
}
// Convert the LTTPR to the sink PHY link status layout
unsafe { memmove(link_status.wrapping_offset(4i32 as isize) as *mut (), link_status.wrapping_offset(3i32 as isize) as *const (), 2u64) };
unsafe {
*link_status.wrapping_offset(3i32 as isize) = 0;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dpcd_read_phy_link_status_698: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, u32, *mut u8) -> i32>, *mut ()>(Some(drm_dp_dpcd_read_phy_link_status)) };

/// drm_dp_link_power_up() - power up a DisplayPort link
/// @aux: DisplayPort AUX channel
/// @revision: DPCD revision supported on the link
///
/// Returns 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_link_power_up")]
extern "C" fn drm_dp_link_power_up(aux: *mut drm_dp_aux, revision: u8) -> i32 {
let mut value: u8 = 0;
let mut err: i32 = 0;
// DP_SET_POWER register is only available on DPCD v1.1 and later
if (revision as i32) < 17 {
return 0;
}
err = drm_dp_dpcd_readb(aux, 1536u32, core::ptr::addr_of_mut!(value)) as i32;
if err < 0 {
return err;
}
let __v1: u8 = ((value as i32) & !3) as u8;
value = __v1;
value = ((value as i32) | 1) as u8;
err = drm_dp_dpcd_writeb(aux, 1536u32, value) as i32;
if err < 0 {
return err;
}
// According to the DP 1.1 specification, a "Sink Device must exit the
// power saving state within 1 ms" (Section 2.5.3.1, Table 5-52, "Sink
// Control Field" (register 0x600).
usleep_range(1000u64, 2000u64);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_link_power_up_699: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, u8) -> i32>, *mut ()>(Some(drm_dp_link_power_up)) };

/// drm_dp_link_power_down() - power down a DisplayPort link
/// @aux: DisplayPort AUX channel
/// @revision: DPCD revision supported on the link
///
/// Returns 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_link_power_down")]
extern "C" fn drm_dp_link_power_down(aux: *mut drm_dp_aux, revision: u8) -> i32 {
let mut value: u8 = 0;
let mut err: i32 = 0;
// DP_SET_POWER register is only available on DPCD v1.1 and later
if (revision as i32) < 17 {
return 0;
}
err = drm_dp_dpcd_readb(aux, 1536u32, core::ptr::addr_of_mut!(value)) as i32;
if err < 0 {
return err;
}
let __v1: u8 = ((value as i32) & !3) as u8;
value = __v1;
value = ((value as i32) | 2) as u8;
err = drm_dp_dpcd_writeb(aux, 1536u32, value) as i32;
if err < 0 {
return err;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_link_power_down_700: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, u8) -> i32>, *mut ()>(Some(drm_dp_link_power_down)) };

fn read_payload_update_status(aux: *mut drm_dp_aux) -> i32 {
let mut ret: i32 = 0;
let mut status: u8 = 0;
ret = drm_dp_dpcd_read_byte(aux, 704u32, core::ptr::addr_of_mut!(status));
if ret < 0 {
return ret;
}
status as i32
}

/// drm_dp_dpcd_write_payload() - Write Virtual Channel information to payload table
/// @aux: DisplayPort AUX channel
/// @vcpid: Virtual Channel Payload ID
/// @start_time_slot: Starting time slot
/// @time_slot_count: Time slot count
///
/// Write the Virtual Channel payload allocation table, checking the payload
/// update status and retrying as necessary.
///
/// Returns:
/// 0 on success, negative error otherwise
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dpcd_write_payload")]
extern "C" fn drm_dp_dpcd_write_payload(aux: *mut drm_dp_aux, vcpid: i32, start_time_slot: u8, time_slot_count: u8) -> i32 {
let mut retries: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut status: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u8 = core::ptr::addr_of_mut!(status) as *mut u8;
let mut payload_alloc: core::mem::MaybeUninit<[u8; 3]> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut [u8; 3] = core::ptr::addr_of_mut!(payload_alloc) as *mut [u8; 3];
unsafe {
'__join_0: {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 3usize);
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
retries = 0;
drm_dp_dpcd_write_byte(aux, 704u32, 1u8);
(*__slate_slot_5)[0] = vcpid as u8;
(*__slate_slot_5)[1] = start_time_slot;
(*__slate_slot_5)[2] = time_slot_count;
*__slate_slot_3 = drm_dp_dpcd_write_data(aux, 448u32, (*__slate_slot_5).as_mut_ptr() as *mut (), 3u64);
if *__slate_slot_3 < 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"failed to write payload allocation %d\n".as_ptr() as *const u8, *__slate_slot_3);
} else {
'__join_6: {
'__join_2: {
loop {
*__slate_slot_3 = drm_dp_dpcd_read_byte(aux, 704u32, core::ptr::addr_of_mut!(*__slate_slot_4));
if *__slate_slot_3 < 0 {
break '__join_6;
} else {
if (*__slate_slot_4 as i32) & 1i32 == 0 {
retries = retries.wrapping_add(1);
if retries < 20 {
usleep_range(10000u64, 20000u64);
} else {
break '__join_2;
}
} else {
break;
}
}
}
*__slate_slot_3 = 0;
break '__join_0;
}
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"status not set after read payload table status %d\n".as_ptr() as *const u8, *__slate_slot_4 as i32);
*__slate_slot_3 = 22i32.wrapping_neg();
break '__join_0;
}
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"failed to read payload table status %d\n".as_ptr() as *const u8, *__slate_slot_3);
}
}
return *__slate_slot_3;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dpcd_write_payload_701: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, i32, u8, u8) -> i32>, *mut ()>(Some(drm_dp_dpcd_write_payload)) };

/// drm_dp_dpcd_clear_payload() - Clear the entire VC Payload ID table
/// @aux: DisplayPort AUX channel
///
/// Clear the entire VC Payload ID table.
///
/// Returns: 0 on success, negative error code on errors.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dpcd_clear_payload")]
extern "C" fn drm_dp_dpcd_clear_payload(aux: *mut drm_dp_aux) -> i32 {
drm_dp_dpcd_write_payload(aux, 0i32, 0u8, 63u8)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dpcd_clear_payload_702: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux) -> i32>, *mut ()>(Some(drm_dp_dpcd_clear_payload)) };

/// drm_dp_dpcd_poll_act_handled() - Poll for ACT handled status
/// @aux: DisplayPort AUX channel
/// @timeout_ms: Timeout in ms
///
/// Try waiting for the sink to finish updating its payload table by polling for
/// the ACT handled bit of DP_PAYLOAD_TABLE_UPDATE_STATUS for up to @timeout_ms
/// milliseconds, defaulting to 3000 ms if 0.
///
/// Returns:
/// 0 if the ACT was handled in time, negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dpcd_poll_act_handled")]
extern "C" fn drm_dp_dpcd_poll_act_handled(aux: *mut drm_dp_aux, mut timeout_ms: i32) -> i32 {
let mut ret: i32 = 0;
let mut status: i32 = 0;
// default to 3 seconds, this is arbitrary
let __v0: i32 = timeout_ms;
timeout_ms = if __v0 != 0 { __v0 } else { 3000 };
let mut __v1: i32;
let __timeout_us: u64 = (timeout_ms as i64).wrapping_mul(1000) as u64;
let __sleep_us: u64 = 200;
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
status = read_payload_update_status(aux);
if status & 2i32 != 0 || status < 0 {
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
ret = __v1;
if ret < 0 && status >= 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, c"[drm] *ERROR* Failed to get ACT after %d ms, last status: %02x\n".as_ptr() as *const u8, timeout_ms, status) };
return 22i32.wrapping_neg();
} else {
if status < 0 {
// Failure here isn't unexpected - the hub may have
// just been unplugged
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"Failed to read payload table status: %d\n".as_ptr() as *const u8, status) };
return status;
}
}
0
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_channel_eq_ok: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_clock_recovery_ok: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_post_lt_adj_req_in_progress: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_get_adjust_request_voltage: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_get_adjust_request_pre_emphasis: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_get_adjust_tx_ffe_preset: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_128b132b_lane_channel_eq_done: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_128b132b_lane_symbol_locked: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_128b132b_eq_interlane_align_done: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_128b132b_cds_interlane_align_done: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_128b132b_link_training_failed: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_read_clock_recovery_delay: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_read_channel_eq_delay: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_128b132b_read_aux_rd_interval: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_link_train_clock_recovery_delay: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_link_train_channel_eq_delay: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_phy_name: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_lttpr_link_train_clock_recovery_delay: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_lttpr_link_train_channel_eq_delay: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_lttpr_wake_timeout_setup: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_link_rate_to_bw_code: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_bw_code_to_link_rate: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dpcd_probe: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dpcd_set_powered: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dpcd_set_probe: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dpcd_read: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dpcd_write: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dpcd_read_link_status: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dpcd_read_phy_link_status: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {28} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_link_power_up: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {29} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_link_power_down: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {30} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dpcd_write_payload: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {31} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dpcd_clear_payload: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {32} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dpcd_poll_act_handled: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {33} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_downstream_is_type: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {34} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_downstream_is_tmds: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {35} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_send_real_edid_checksum: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {36} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_read_dpcd_caps: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {37} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_read_downstream_info: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {38} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_downstream_max_dotclock: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {39} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_downstream_max_tmds_clock: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {40} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_downstream_min_tmds_clock: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {41} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_downstream_max_bpc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {42} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_downstream_420_passthrough: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {43} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_downstream_444_to_420_conversion: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {44} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_downstream_rgb_to_ycbcr_conversion: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {45} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_downstream_mode: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {46} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_downstream_id: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {47} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_downstream_debug: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {48} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_subconnector_type: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {49} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_set_subconnector_property: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {50} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_read_sink_count_cap: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {51} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_read_sink_count: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {52} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_remote_aux_init: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {53} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_aux_init: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {54} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_aux_register: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {55} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_aux_unregister: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {56} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_psr_setup_time: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {57} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_start_crc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {58} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_stop_crc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {59} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_read_desc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {60} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dump_lttpr_desc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {61} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dsc_sink_bpp_incr: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {62} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dsc_slice_count_to_mask: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {63} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dsc_sink_slice_count_mask: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {64} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dsc_sink_max_slice_count: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {65} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dsc_sink_line_buf_depth: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {66} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dsc_sink_supported_input_bpcs: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {67} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dsc_sink_max_slice_throughput: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {68} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dsc_branch_max_overall_throughput: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {69} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_dsc_branch_max_line_width: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {70} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_read_lttpr_common_caps: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {71} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_read_lttpr_phy_caps: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {72} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_lttpr_count: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {73} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_lttpr_max_link_rate: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {74} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_lttpr_set_transparent_mode: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {75} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_lttpr_init: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {76} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_lttpr_max_lane_count: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {77} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_lttpr_voltage_swing_level_3_supported: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {78} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_lttpr_pre_emphasis_level_3_supported: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {79} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_get_phy_test_pattern: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {80} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_set_phy_test_pattern: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {81} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_vsc_sdp_log: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {82} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_as_sdp_log: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {83} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_as_sdp_supported: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {84} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_vsc_sdp_supported: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {85} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_vsc_sdp_pack: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {86} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_get_pcon_max_frl_bw: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {87} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_frl_prepare: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {88} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_is_frl_ready: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {89} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_frl_configure_1: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {90} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_frl_configure_2: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {91} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_reset_frl_config: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {92} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_frl_enable: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {93} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_hdmi_link_active: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {94} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_hdmi_link_mode: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {95} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_hdmi_frl_link_error_count: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {96} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_enc_is_dsc_1_2: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {97} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_dsc_max_slices: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {98} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_dsc_max_slice_width: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {99} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_dsc_bpp_incr: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {100} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_pps_default: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {101} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_pps_override_buf: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {102} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_pps_override_param: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {103} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_pcon_convert_rgb_to_ycbcr: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {104} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edp_backlight_set_level: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {105} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edp_backlight_enable: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {106} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edp_backlight_disable: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {107} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_edp_backlight_init: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {108} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_panel_dp_aux_backlight: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {109} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_link_symbol_cycles: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {110} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_bw_overhead: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {111} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_bw_channel_coding_efficiency: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {112} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_dp_max_dprx_data_rate: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {113} ; .previous", sym drm_dp_channel_eq_ok, sym drm_dp_clock_recovery_ok, sym drm_dp_post_lt_adj_req_in_progress, sym drm_dp_get_adjust_request_voltage, sym drm_dp_get_adjust_request_pre_emphasis, sym drm_dp_get_adjust_tx_ffe_preset, sym drm_dp_128b132b_lane_channel_eq_done, sym drm_dp_128b132b_lane_symbol_locked, sym drm_dp_128b132b_eq_interlane_align_done, sym drm_dp_128b132b_cds_interlane_align_done, sym drm_dp_128b132b_link_training_failed, sym drm_dp_read_clock_recovery_delay, sym drm_dp_read_channel_eq_delay, sym drm_dp_128b132b_read_aux_rd_interval, sym drm_dp_link_train_clock_recovery_delay, sym drm_dp_link_train_channel_eq_delay, sym drm_dp_phy_name, sym drm_dp_lttpr_link_train_clock_recovery_delay, sym drm_dp_lttpr_link_train_channel_eq_delay, sym drm_dp_lttpr_wake_timeout_setup, sym drm_dp_link_rate_to_bw_code, sym drm_dp_bw_code_to_link_rate, sym drm_dp_dpcd_probe, sym drm_dp_dpcd_set_powered, sym drm_dp_dpcd_set_probe, sym drm_dp_dpcd_read, sym drm_dp_dpcd_write, sym drm_dp_dpcd_read_link_status, sym drm_dp_dpcd_read_phy_link_status, sym drm_dp_link_power_up, sym drm_dp_link_power_down, sym drm_dp_dpcd_write_payload, sym drm_dp_dpcd_clear_payload, sym drm_dp_dpcd_poll_act_handled, sym drm_dp_downstream_is_type, sym drm_dp_downstream_is_tmds, sym drm_dp_send_real_edid_checksum, sym drm_dp_read_dpcd_caps, sym drm_dp_read_downstream_info, sym drm_dp_downstream_max_dotclock, sym drm_dp_downstream_max_tmds_clock, sym drm_dp_downstream_min_tmds_clock, sym drm_dp_downstream_max_bpc, sym drm_dp_downstream_420_passthrough, sym drm_dp_downstream_444_to_420_conversion, sym drm_dp_downstream_rgb_to_ycbcr_conversion, sym drm_dp_downstream_mode, sym drm_dp_downstream_id, sym drm_dp_downstream_debug, sym drm_dp_subconnector_type, sym drm_dp_set_subconnector_property, sym drm_dp_read_sink_count_cap, sym drm_dp_read_sink_count, sym drm_dp_remote_aux_init, sym drm_dp_aux_init, sym drm_dp_aux_register, sym drm_dp_aux_unregister, sym drm_dp_psr_setup_time, sym drm_dp_start_crc, sym drm_dp_stop_crc, sym drm_dp_read_desc, sym drm_dp_dump_lttpr_desc, sym drm_dp_dsc_sink_bpp_incr, sym drm_dp_dsc_slice_count_to_mask, sym drm_dp_dsc_sink_slice_count_mask, sym drm_dp_dsc_sink_max_slice_count, sym drm_dp_dsc_sink_line_buf_depth, sym drm_dp_dsc_sink_supported_input_bpcs, sym drm_dp_dsc_sink_max_slice_throughput, sym drm_dp_dsc_branch_max_overall_throughput, sym drm_dp_dsc_branch_max_line_width, sym drm_dp_read_lttpr_common_caps, sym drm_dp_read_lttpr_phy_caps, sym drm_dp_lttpr_count, sym drm_dp_lttpr_max_link_rate, sym drm_dp_lttpr_set_transparent_mode, sym drm_dp_lttpr_init, sym drm_dp_lttpr_max_lane_count, sym drm_dp_lttpr_voltage_swing_level_3_supported, sym drm_dp_lttpr_pre_emphasis_level_3_supported, sym drm_dp_get_phy_test_pattern, sym drm_dp_set_phy_test_pattern, sym drm_dp_vsc_sdp_log, sym drm_dp_as_sdp_log, sym drm_dp_as_sdp_supported, sym drm_dp_vsc_sdp_supported, sym drm_dp_vsc_sdp_pack, sym drm_dp_get_pcon_max_frl_bw, sym drm_dp_pcon_frl_prepare, sym drm_dp_pcon_is_frl_ready, sym drm_dp_pcon_frl_configure_1, sym drm_dp_pcon_frl_configure_2, sym drm_dp_pcon_reset_frl_config, sym drm_dp_pcon_frl_enable, sym drm_dp_pcon_hdmi_link_active, sym drm_dp_pcon_hdmi_link_mode, sym drm_dp_pcon_hdmi_frl_link_error_count, sym drm_dp_pcon_enc_is_dsc_1_2, sym drm_dp_pcon_dsc_max_slices, sym drm_dp_pcon_dsc_max_slice_width, sym drm_dp_pcon_dsc_bpp_incr, sym drm_dp_pcon_pps_default, sym drm_dp_pcon_pps_override_buf, sym drm_dp_pcon_pps_override_param, sym drm_dp_pcon_convert_rgb_to_ycbcr, sym drm_edp_backlight_set_level, sym drm_edp_backlight_enable, sym drm_edp_backlight_disable, sym drm_edp_backlight_init, sym drm_panel_dp_aux_backlight, sym drm_dp_link_symbol_cycles, sym drm_dp_bw_overhead, sym drm_dp_bw_channel_coding_efficiency, sym drm_dp_max_dprx_data_rate, options(att_syntax));

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dpcd_poll_act_handled_703: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, i32) -> i32>, *mut ()>(Some(drm_dp_dpcd_poll_act_handled)) };

fn is_edid_digital_input_dp(drm_edid: *const drm_edid) -> bool {
// FIXME: get rid of drm_edid_raw()
let edid: *const edid = unsafe { drm_edid_raw(drm_edid) };
!edid.is_null() && ((unsafe { (*edid).revision }) as i32) >= 4 && ((unsafe { (*edid).input }) as i32) & 128i32 != 0 && ((unsafe { (*edid).input }) as i32) & 7i32 == 5i32
}

/// drm_dp_downstream_is_type() - is the downstream facing port of certain type?
/// @dpcd: DisplayPort configuration data
/// @port_cap: port capabilities
/// @type: port type to be checked. Can be:
/// 	  %DP_DS_PORT_TYPE_DP, %DP_DS_PORT_TYPE_VGA, %DP_DS_PORT_TYPE_DVI,
/// 	  %DP_DS_PORT_TYPE_HDMI, %DP_DS_PORT_TYPE_NON_EDID,
/// 	  %DP_DS_PORT_TYPE_DP_DUALMODE or %DP_DS_PORT_TYPE_WIRELESS.
///
/// Caveat: Only works with DPCD 1.1+ port caps.
///
/// Returns: whether the downstream facing port matches the type.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_downstream_is_type")]
extern "C" fn drm_dp_downstream_is_type(dpcd: *const u8, port_cap: *const u8, r#type: u8) -> bool {
drm_dp_is_branch(dpcd) && ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) >= 17 && ((unsafe { *port_cap.wrapping_offset(0 as isize) }) as i32) & 7i32 == (r#type as i32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_downstream_is_type_704: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8, u8) -> bool>, *mut ()>(Some(drm_dp_downstream_is_type)) };

/// drm_dp_downstream_is_tmds() - is the downstream facing port TMDS?
/// @dpcd: DisplayPort configuration data
/// @port_cap: port capabilities
/// @drm_edid: EDID
///
/// Returns: whether the downstream facing port is TMDS (HDMI/DVI).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_downstream_is_tmds")]
extern "C" fn drm_dp_downstream_is_tmds(dpcd: *const u8, port_cap: *const u8, drm_edid: *const drm_edid) -> bool {
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) < 17 {
match ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 6 {
4 => {
return true;
}
_ => {
return false;
}
}
}
match ((unsafe { *port_cap.wrapping_offset(0 as isize) }) as i32) & 7i32 {
5 => {
if is_edid_digital_input_dp(drm_edid) {
return false;
}
return true;
}
2 | 3 => {
return true;
}
_ => {
return false;
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_downstream_is_tmds_705: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8, *const drm_edid) -> bool>, *mut ()>(Some(drm_dp_downstream_is_tmds)) };

/// drm_dp_send_real_edid_checksum() - send back real edid checksum value
/// @aux: DisplayPort AUX channel
/// @real_edid_checksum: real edid checksum for the last block
///
/// Returns:
/// True on success
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_send_real_edid_checksum")]
extern "C" fn drm_dp_send_real_edid_checksum(aux: *mut drm_dp_aux, real_edid_checksum: u8) -> bool {
let mut link_edid_read: u8 = 0;
let mut auto_test_req: u8 = 0;
let mut test_resp: u8 = 0;
if drm_dp_dpcd_read_byte(aux, 513u32, core::ptr::addr_of_mut!(auto_test_req)) < 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, c"[drm] *ERROR* %s: DPCD failed read at register 0x%x\n".as_ptr() as *const u8, (*aux).name, 513i32) };
return false;
}
auto_test_req = ((auto_test_req as i32) & 2i32) as u8;
if drm_dp_dpcd_read_byte(aux, 536u32, core::ptr::addr_of_mut!(link_edid_read)) < 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, c"[drm] *ERROR* %s: DPCD failed read at register 0x%x\n".as_ptr() as *const u8, (*aux).name, 536i32) };
return false;
}
link_edid_read = ((link_edid_read as i32) & 4i32) as u8;
if auto_test_req == 0 || link_edid_read == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Source DUT does not support TEST_EDID_READ\n".as_ptr() as *const u8, (*aux).name) };
return false;
}
if drm_dp_dpcd_write_byte(aux, 513u32, auto_test_req) < 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, c"[drm] *ERROR* %s: DPCD failed write at register 0x%x\n".as_ptr() as *const u8, (*aux).name, 513i32) };
return false;
}
// send back checksum for the last edid extension block data
if drm_dp_dpcd_write_byte(aux, 609u32, real_edid_checksum) < 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, c"[drm] *ERROR* %s: DPCD failed write at register 0x%x\n".as_ptr() as *const u8, (*aux).name, 609i32) };
return false;
}
test_resp = ((test_resp as i32) | 4i32) as u8;
if drm_dp_dpcd_write_byte(aux, 608u32, test_resp) < 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, c"[drm] *ERROR* %s: DPCD failed write at register 0x%x\n".as_ptr() as *const u8, (*aux).name, 608i32) };
return false;
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_send_real_edid_checksum_706: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, u8) -> bool>, *mut ()>(Some(drm_dp_send_real_edid_checksum)) };

fn drm_dp_downstream_port_count(dpcd: *const u8) -> u8 {
let mut port_count: u8 = (((unsafe { *dpcd.wrapping_offset(7 as isize) }) as i32) & 15) as u8;
if ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 16i32 != 0 && (port_count as i32) > 4 {
port_count = 4;
}
port_count
}

fn drm_dp_read_extended_dpcd_caps(aux: *mut drm_dp_aux, dpcd: *mut u8) -> i32 {
let mut dpcd_ext: [u8; 15] = [0u8; 15];
let mut ret: i32 = 0;
// Prior to DP1.3 the bit represented by
// DP_EXTENDED_RECEIVER_CAP_FIELD_PRESENT was reserved.
// If it is set DP_DPCD_REV at 0000h could be at a value less than
// the true capability of the panel. The only way to check is to
// then compare 0000h and 2200h.
if ((unsafe { *dpcd.wrapping_offset(14 as isize) }) as i32) & 128i32 == 0 {
return 0;
}
ret = drm_dp_dpcd_read_data(aux, 8704u32, core::ptr::addr_of_mut!(dpcd_ext) as *mut (), core::mem::size_of::<[u8; 15]>() as u64);
if ret < 0 {
return ret;
}
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) > (dpcd_ext[0] as i32) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Extended DPCD rev less than base DPCD rev (%d > %d)\n".as_ptr() as *const u8, (*aux).name, *dpcd.wrapping_offset(0 as isize) as i32, dpcd_ext[0] as i32) };
return 0;
}
if (unsafe { memcmp(dpcd as *const (), dpcd_ext.as_mut_ptr() as *const (), core::mem::size_of::<[u8; 15]>() as u64) }) == 0 {
return 0;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Base DPCD: %*ph\n".as_ptr() as *const u8, (*aux).name, 15i32, dpcd) };
unsafe { memcpy(dpcd as *mut (), dpcd_ext.as_mut_ptr() as *const (), core::mem::size_of::<[u8; 15]>() as u64) };
0
}

/// drm_dp_read_dpcd_caps() - read DPCD caps and extended DPCD caps if
/// available
/// @aux: DisplayPort AUX channel
/// @dpcd: Buffer to store the resulting DPCD in
///
/// Attempts to read the base DPCD caps for @aux. Additionally, this function
/// checks for and reads the extended DPRX caps (%DP_DP13_DPCD_REV) if
/// present.
///
/// Returns: %0 if the DPCD was read successfully, negative error code
/// otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_read_dpcd_caps")]
extern "C" fn drm_dp_read_dpcd_caps(aux: *mut drm_dp_aux, dpcd: *mut u8) -> i32 {
let mut ret: i32 = drm_dp_dpcd_read_data(aux, 0u32, dpcd as *mut (), 15u64);
if ret < 0 {
return ret;
}
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) == 0 {
return 5i32.wrapping_neg();
}
ret = drm_dp_read_extended_dpcd_caps(aux, dpcd);
if ret < 0 {
return ret;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: DPCD: %*ph\n".as_ptr() as *const u8, (*aux).name, 15i32, dpcd) };
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_read_dpcd_caps_707: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *mut u8) -> i32>, *mut ()>(Some(drm_dp_read_dpcd_caps)) };

/// drm_dp_read_downstream_info() - read DPCD downstream port info if available
/// @aux: DisplayPort AUX channel
/// @dpcd: A cached copy of the port's DPCD
/// @downstream_ports: buffer to store the downstream port info in
///
/// See also:
/// drm_dp_downstream_max_clock()
/// drm_dp_downstream_max_bpc()
///
/// Returns: 0 if either the downstream port info was read successfully or
/// there was no downstream info to read, or a negative error code otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_read_downstream_info")]
extern "C" fn drm_dp_read_downstream_info(aux: *mut drm_dp_aux, dpcd: *const u8, downstream_ports: *mut u8) -> i32 {
let mut ret: i32 = 0;
let mut len: u8 = 0;
unsafe { memset(downstream_ports as *mut (), 0i32, 16u64) };
// No downstream info to read
if !drm_dp_is_branch(dpcd) || ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) == 16 {
return 0;
}
// Some branches advertise having 0 downstream ports, despite also advertising they have a
// downstream port present. The DP spec isn't clear on if this is allowed or not, but since
// some branches do it we need to handle it regardless.
len = drm_dp_downstream_port_count(dpcd);
if len == 0 {
return 0;
}
if ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 16i32 != 0 {
len = (len as i32).wrapping_mul(4) as u8;
}
ret = drm_dp_dpcd_read_data(aux, 128u32, downstream_ports as *mut (), len as u64);
if ret < 0 {
return ret;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: DPCD DFP: %*ph\n".as_ptr() as *const u8, (*aux).name, len as i32, downstream_ports) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_read_downstream_info_708: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *const u8, *mut u8) -> i32>, *mut ()>(Some(drm_dp_read_downstream_info)) };

/// drm_dp_downstream_max_dotclock() - extract downstream facing port max dot clock
/// @dpcd: DisplayPort configuration data
/// @port_cap: port capabilities
///
/// Returns: Downstream facing port max dot clock in kHz on success,
/// or 0 if max clock not defined
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_downstream_max_dotclock")]
extern "C" fn drm_dp_downstream_max_dotclock(dpcd: *const u8, port_cap: *const u8) -> i32 {
if !drm_dp_is_branch(dpcd) {
return 0;
}
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) < 17 {
return 0;
}
match ((unsafe { *port_cap.wrapping_offset(0 as isize) }) as i32) & 7i32 {
1 => {
if ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 16i32 == 0 {
return 0;
}
return ((unsafe { *port_cap.wrapping_offset(1 as isize) }) as i32).wrapping_mul(8000);
}
_ => {
return 0;
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_downstream_max_dotclock_709: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8) -> i32>, *mut ()>(Some(drm_dp_downstream_max_dotclock)) };

/// drm_dp_downstream_max_tmds_clock() - extract downstream facing port max TMDS clock
/// @dpcd: DisplayPort configuration data
/// @port_cap: port capabilities
/// @drm_edid: EDID
///
/// Returns: HDMI/DVI downstream facing port max TMDS clock in kHz on success,
/// or 0 if max TMDS clock not defined
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_downstream_max_tmds_clock")]
extern "C" fn drm_dp_downstream_max_tmds_clock(dpcd: *const u8, port_cap: *const u8, drm_edid: *const drm_edid) -> i32 {
if !drm_dp_is_branch(dpcd) {
return 0;
}
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) < 17 {
match ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 6 {
4 => {
return 165000;
}
_ => {
return 0;
}
}
}
// It's left up to the driver to check the
// DP dual mode adapter's max TMDS clock.
//
// Unfortunately it looks like branch devices
// may not fordward that the DP dual mode i2c
// access so we just usually get i2c nak :(
// We should perhaps assume 165 MHz when detailed cap
// info is not available. But looks like many typical
// branch devices fall into that category and so we'd
// probably end up with users complaining that they can't
// get high resolution modes with their favorite dongle.
//
// So let's limit to 300 MHz instead since DPCD 1.4
// HDMI 2.0 DFPs are required to have the detailed cap
// info. So it's more likely we're dealing with a HDMI 1.4
// compatible* device here.
// FIXME what to do about DVI dual link?
match ((unsafe { *port_cap.wrapping_offset(0 as isize) }) as i32) & 7i32 {
5 => {
if is_edid_digital_input_dp(drm_edid) {
return 0;
}
// It's left up to the driver to check the
// DP dual mode adapter's max TMDS clock.
//
// Unfortunately it looks like branch devices
// may not fordward that the DP dual mode i2c
// access so we just usually get i2c nak :(
if ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 16i32 == 0 {
return 300000;
}
// We should perhaps assume 165 MHz when detailed cap
// info is not available. But looks like many typical
// branch devices fall into that category and so we'd
// probably end up with users complaining that they can't
// get high resolution modes with their favorite dongle.
//
// So let's limit to 300 MHz instead since DPCD 1.4
// HDMI 2.0 DFPs are required to have the detailed cap
// info. So it's more likely we're dealing with a HDMI 1.4
// compatible* device here.
return ((unsafe { *port_cap.wrapping_offset(1 as isize) }) as i32).wrapping_mul(2500);
}
3 => {
if ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 16i32 == 0 {
return 300000;
}
// We should perhaps assume 165 MHz when detailed cap
// info is not available. But looks like many typical
// branch devices fall into that category and so we'd
// probably end up with users complaining that they can't
// get high resolution modes with their favorite dongle.
//
// So let's limit to 300 MHz instead since DPCD 1.4
// HDMI 2.0 DFPs are required to have the detailed cap
// info. So it's more likely we're dealing with a HDMI 1.4
// compatible* device here.
return ((unsafe { *port_cap.wrapping_offset(1 as isize) }) as i32).wrapping_mul(2500);
}
2 => {
if ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 16i32 == 0 {
return 165000;
}
// FIXME what to do about DVI dual link?
return ((unsafe { *port_cap.wrapping_offset(1 as isize) }) as i32).wrapping_mul(2500);
}
_ => {
return 0;
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_downstream_max_tmds_clock_710: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8, *const drm_edid) -> i32>, *mut ()>(Some(drm_dp_downstream_max_tmds_clock)) };

/// drm_dp_downstream_min_tmds_clock() - extract downstream facing port min TMDS clock
/// @dpcd: DisplayPort configuration data
/// @port_cap: port capabilities
/// @drm_edid: EDID
///
/// Returns: HDMI/DVI downstream facing port min TMDS clock in kHz on success,
/// or 0 if max TMDS clock not defined
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_downstream_min_tmds_clock")]
extern "C" fn drm_dp_downstream_min_tmds_clock(dpcd: *const u8, port_cap: *const u8, drm_edid: *const drm_edid) -> i32 {
if !drm_dp_is_branch(dpcd) {
return 0;
}
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) < 17 {
match ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 6 {
4 => {
return 25000;
}
_ => {
return 0;
}
}
}
// Unclear whether the protocol converter could
// utilize pixel replication. Assume it won't.
match ((unsafe { *port_cap.wrapping_offset(0 as isize) }) as i32) & 7i32 {
5 => {
if is_edid_digital_input_dp(drm_edid) {
return 0;
}
return 25000;
// Unclear whether the protocol converter could
// utilize pixel replication. Assume it won't.
}
2 | 3 => {
return 25000;
// Unclear whether the protocol converter could
// utilize pixel replication. Assume it won't.
}
_ => {
return 0;
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_downstream_min_tmds_clock_711: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8, *const drm_edid) -> i32>, *mut ()>(Some(drm_dp_downstream_min_tmds_clock)) };

/// drm_dp_downstream_max_bpc() - extract downstream facing port max
///                               bits per component
/// @dpcd: DisplayPort configuration data
/// @port_cap: downstream facing port capabilities
/// @drm_edid: EDID
///
/// Returns: Max bpc on success or 0 if max bpc not defined
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_downstream_max_bpc")]
extern "C" fn drm_dp_downstream_max_bpc(dpcd: *const u8, port_cap: *const u8, drm_edid: *const drm_edid) -> i32 {
if !drm_dp_is_branch(dpcd) {
return 0;
}
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) < 17 {
match ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 6 {
0 => {
return 0;
}
_ => {
return 8;
}
}
}
match ((unsafe { *port_cap.wrapping_offset(0 as isize) }) as i32) & 7i32 {
0 => {
return 0;
}
5 => {
if is_edid_digital_input_dp(drm_edid) {
return 0;
}
if ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 16i32 == 0 {
return 8;
}
match ((unsafe { *port_cap.wrapping_offset(2 as isize) }) as i32) & 3i32 {
0 => {
return 8;
}
1 => {
return 10;
}
2 => {
return 12;
}
3 => {
return 16;
}
_ => {
return 8;
}
}
}
1..=3 => {
if ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 16i32 == 0 {
return 8;
}
match ((unsafe { *port_cap.wrapping_offset(2 as isize) }) as i32) & 3i32 {
0 => {
return 8;
}
1 => {
return 10;
}
2 => {
return 12;
}
3 => {
return 16;
}
_ => {
return 8;
}
}
}
_ => {
return 8;
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_downstream_max_bpc_712: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8, *const drm_edid) -> i32>, *mut ()>(Some(drm_dp_downstream_max_bpc)) };

/// drm_dp_downstream_420_passthrough() - determine downstream facing port
///                                       YCbCr 4:2:0 pass-through capability
/// @dpcd: DisplayPort configuration data
/// @port_cap: downstream facing port capabilities
///
/// Returns: whether the downstream facing port can pass through YCbCr 4:2:0
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_downstream_420_passthrough")]
extern "C" fn drm_dp_downstream_420_passthrough(dpcd: *const u8, port_cap: *const u8) -> bool {
if !drm_dp_is_branch(dpcd) {
return false;
}
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) < 19 {
return false;
}
match ((unsafe { *port_cap.wrapping_offset(0 as isize) }) as i32) & 7i32 {
0 => {
return true;
}
3 => {
if ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 16i32 == 0 {
return false;
}
return ((unsafe { *port_cap.wrapping_offset(3 as isize) }) as i32) & 4i32 != 0;
}
_ => {
return false;
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_downstream_420_passthrough_713: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8) -> bool>, *mut ()>(Some(drm_dp_downstream_420_passthrough)) };

/// drm_dp_downstream_444_to_420_conversion() - determine downstream facing port
///                                             YCbCr 4:4:4->4:2:0 conversion capability
/// @dpcd: DisplayPort configuration data
/// @port_cap: downstream facing port capabilities
///
/// Returns: whether the downstream facing port can convert YCbCr 4:4:4 to 4:2:0
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_downstream_444_to_420_conversion")]
extern "C" fn drm_dp_downstream_444_to_420_conversion(dpcd: *const u8, port_cap: *const u8) -> bool {
if !drm_dp_is_branch(dpcd) {
return false;
}
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) < 19 {
return false;
}
match ((unsafe { *port_cap.wrapping_offset(0 as isize) }) as i32) & 7i32 {
3 => {
if ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 16i32 == 0 {
return false;
}
return ((unsafe { *port_cap.wrapping_offset(3 as isize) }) as i32) & 16i32 != 0;
}
_ => {
return false;
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_downstream_444_to_420_conversion_714: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8) -> bool>, *mut ()>(Some(drm_dp_downstream_444_to_420_conversion)) };

/// drm_dp_downstream_rgb_to_ycbcr_conversion() - determine downstream facing port
///                                               RGB->YCbCr conversion capability
/// @dpcd: DisplayPort configuration data
/// @port_cap: downstream facing port capabilities
/// @color_spc: Colorspace for which conversion cap is sought
///
/// Returns: whether the downstream facing port can convert RGB->YCbCr for a given
/// colorspace.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_downstream_rgb_to_ycbcr_conversion")]
extern "C" fn drm_dp_downstream_rgb_to_ycbcr_conversion(dpcd: *const u8, port_cap: *const u8, color_spc: u8) -> bool {
if !drm_dp_is_branch(dpcd) {
return false;
}
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) < 19 {
return false;
}
match ((unsafe { *port_cap.wrapping_offset(0 as isize) }) as i32) & 7i32 {
3 => {
if ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 16i32 == 0 {
return false;
}
return ((unsafe { *port_cap.wrapping_offset(3 as isize) }) as i32) & (color_spc as i32) != 0;
}
_ => {
return false;
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_downstream_rgb_to_ycbcr_conversion_715: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8, u8) -> bool>, *mut ()>(Some(drm_dp_downstream_rgb_to_ycbcr_conversion)) };

/// drm_dp_downstream_mode() - return a mode for downstream facing port
/// @dev: DRM device
/// @dpcd: DisplayPort configuration data
/// @port_cap: port capabilities
///
/// Provides a suitable mode for downstream facing ports without EDID.
///
/// Returns: A new drm_display_mode on success or NULL on failure
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_downstream_mode")]
extern "C" fn drm_dp_downstream_mode(dev: *mut drm_device, dpcd: *const u8, port_cap: *const u8) -> *mut drm_display_mode {
let mut vic: u8 = 0;
if !drm_dp_is_branch(dpcd) {
return core::ptr::null_mut::<drm_display_mode>();
}
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) < 17 {
return core::ptr::null_mut::<drm_display_mode>();
}
match ((unsafe { *port_cap.wrapping_offset(0 as isize) }) as i32) & 7i32 {
4 => {
match ((unsafe { *port_cap.wrapping_offset(0 as isize) }) as i32) & 240i32 {
16 => {
vic = 6;
}
32 => {
vic = 21;
}
48 => {
vic = 5;
}
64 => {
vic = 20;
}
80 => {
vic = 4;
}
112 => {
vic = 19;
}
_ => {
return core::ptr::null_mut::<drm_display_mode>();
}
}
return unsafe { drm_display_mode_from_cea_vic(dev, vic) };
}
_ => {
return core::ptr::null_mut::<drm_display_mode>();
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_downstream_mode_716: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *const u8, *const u8) -> *mut drm_display_mode>, *mut ()>(Some(drm_dp_downstream_mode)) };

/// drm_dp_downstream_id() - identify branch device
/// @aux: DisplayPort AUX channel
/// @id: DisplayPort branch device id
///
/// Returns branch device id on success or NULL on failure
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_downstream_id")]
extern "C" fn drm_dp_downstream_id(aux: *mut drm_dp_aux, id: *mut u8) -> i32 {
drm_dp_dpcd_read_data(aux, 1283u32, id as *mut (), 6u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_downstream_id_717: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *mut u8) -> i32>, *mut ()>(Some(drm_dp_downstream_id)) };

/// drm_dp_downstream_debug() - debug DP branch devices
/// @m: pointer for debugfs file
/// @dpcd: DisplayPort configuration data
/// @port_cap: port capabilities
/// @drm_edid: EDID
/// @aux: DisplayPort AUX channel
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_downstream_debug")]
extern "C" fn drm_dp_downstream_debug(m: *mut seq_file, dpcd: *const u8, port_cap: *const u8, drm_edid: *const drm_edid, aux: *mut drm_dp_aux) {
let detailed_cap_info: bool = ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 16i32 != 0;
let mut clk: i32 = 0;
let mut bpc: i32 = 0;
let mut id: [u8; 7] = [0u8; 7];
let mut len: i32 = 0;
let mut rev: [u8; 2] = [0u8; 2];
let r#type: i32 = ((unsafe { *port_cap.wrapping_offset(0 as isize) }) as i32) & 7i32;
let branch_device: bool = drm_dp_is_branch(dpcd);
unsafe { seq_printf(m, c"\tDP branch device present: %s\n".as_ptr() as *const u8, str_yes_no(branch_device)) };
if !branch_device {
return;
}
match r#type {
0 => {
seq_puts(m, c"\t\tType: DisplayPort\n".as_ptr() as *const u8);
}
1 => {
seq_puts(m, c"\t\tType: VGA\n".as_ptr() as *const u8);
}
2 => {
seq_puts(m, c"\t\tType: DVI\n".as_ptr() as *const u8);
}
3 => {
seq_puts(m, c"\t\tType: HDMI\n".as_ptr() as *const u8);
}
4 => {
seq_puts(m, c"\t\tType: others without EDID support\n".as_ptr() as *const u8);
}
5 => {
seq_puts(m, c"\t\tType: DP++\n".as_ptr() as *const u8);
}
6 => {
seq_puts(m, c"\t\tType: Wireless\n".as_ptr() as *const u8);
}
_ => {
seq_puts(m, c"\t\tType: N/A\n".as_ptr() as *const u8);
}
}
unsafe { memset(id.as_mut_ptr() as *mut (), 0i32, core::mem::size_of::<[u8; 7]>() as u64) };
drm_dp_downstream_id(aux, id.as_mut_ptr() as *mut u8);
unsafe { seq_printf(m, c"\t\tID: %s\n".as_ptr() as *const u8, id.as_mut_ptr() as *mut u8) };
len = drm_dp_dpcd_read_data(aux, 1289u32, (rev.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as *mut (), 1u64);
if len == 0 {
unsafe { seq_printf(m, c"\t\tHW: %d.%d\n".as_ptr() as *const u8, ((rev[0] as i32) & 240) >> 4i32, (rev[0] as i32) & 15) };
}
len = drm_dp_dpcd_read_data(aux, 1290u32, rev.as_mut_ptr() as *mut (), 2u64);
if len == 0 {
unsafe { seq_printf(m, c"\t\tSW: %d.%d\n".as_ptr() as *const u8, rev[0] as i32, rev[1] as i32) };
}
if detailed_cap_info {
clk = drm_dp_downstream_max_dotclock(dpcd, port_cap);
if clk > 0 {
unsafe { seq_printf(m, c"\t\tMax dot clock: %d kHz\n".as_ptr() as *const u8, clk) };
}
clk = drm_dp_downstream_max_tmds_clock(dpcd, port_cap, drm_edid);
if clk > 0 {
unsafe { seq_printf(m, c"\t\tMax TMDS clock: %d kHz\n".as_ptr() as *const u8, clk) };
}
clk = drm_dp_downstream_min_tmds_clock(dpcd, port_cap, drm_edid);
if clk > 0 {
unsafe { seq_printf(m, c"\t\tMin TMDS clock: %d kHz\n".as_ptr() as *const u8, clk) };
}
bpc = drm_dp_downstream_max_bpc(dpcd, port_cap, drm_edid);
if bpc > 0 {
unsafe { seq_printf(m, c"\t\tMax bpc: %d\n".as_ptr() as *const u8, bpc) };
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_downstream_debug_718: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut seq_file, *const u8, *const u8, *const drm_edid, *mut drm_dp_aux)>, *mut ()>(Some(drm_dp_downstream_debug)) };

/// drm_dp_subconnector_type() - get DP branch device type
/// @dpcd: DisplayPort configuration data
/// @port_cap: port capabilities
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_subconnector_type")]
extern "C" fn drm_dp_subconnector_type(dpcd: *const u8, port_cap: *const u8) -> u32 {
let mut r#type: i32 = 0;
if !drm_dp_is_branch(dpcd) {
return 15;
}
// DP 1.0 approach
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) == 16 {
r#type = ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 6;
match r#type {
4 => {
return 3; // Can be HDMI or DVI-D, DVI-D is a safer option
}
2 => {
return 1; // Can be VGA or DVI-A, VGA is more popular
}
0 => {
return 10;
}
_ => {
return 0;
}
}
}
r#type = ((unsafe { *port_cap.wrapping_offset(0 as isize) }) as i32) & 7i32;
match r#type {
0 | 5 => {
return 10;
}
1 => {
return 1;
}
2 => {
return 3;
}
3 => {
return 11;
}
6 => {
return 18;
}
_ => {
return 0;
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_subconnector_type_719: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8) -> u32>, *mut ()>(Some(drm_dp_subconnector_type)) };

/// drm_dp_set_subconnector_property - set subconnector for DP connector
/// @connector: connector to set property on
/// @status: connector status
/// @dpcd: DisplayPort configuration data
/// @port_cap: port capabilities
///
/// Called by a driver on every detect event.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_set_subconnector_property")]
extern "C" fn drm_dp_set_subconnector_property(connector: *mut drm_connector, status: u32, dpcd: *const u8, port_cap: *const u8) {
let mut subconnector: u32 = 0;
if (status as u32) == 1 {
subconnector = drm_dp_subconnector_type(dpcd, port_cap);
}
unsafe { drm_object_property_set_value(core::ptr::addr_of_mut!((*connector).base), (*(*connector).dev).mode_config.dp_subconnector_property, (subconnector as u32) as u64) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_set_subconnector_property_720: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, u32, *const u8, *const u8)>, *mut ()>(Some(drm_dp_set_subconnector_property)) };

/// drm_dp_read_sink_count_cap() - Check whether a given connector has a valid sink
/// count
/// @connector: The DRM connector to check
/// @dpcd: A cached copy of the connector's DPCD RX capabilities
/// @desc: A cached copy of the connector's DP descriptor
///
/// See also: drm_dp_read_sink_count()
///
/// Returns: %True if the (e)DP connector has a valid sink count that should
/// be probed, %false otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_read_sink_count_cap")]
extern "C" fn drm_dp_read_sink_count_cap(connector: *mut drm_connector, dpcd: *const u8, desc: *const drm_dp_desc) -> bool {
// Some eDP panels don't set a valid value for the sink count
(unsafe { (*connector).connector_type }) != 14 && ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) >= 17 && ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 1i32 != 0 && !drm_dp_has_quirk(desc, 2u32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_read_sink_count_cap_721: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *const u8, *const drm_dp_desc) -> bool>, *mut ()>(Some(drm_dp_read_sink_count_cap)) };

/// drm_dp_read_sink_count() - Retrieve the sink count for a given sink
/// @aux: The DP AUX channel to use
///
/// See also: drm_dp_read_sink_count_cap()
///
/// Returns: The current sink count reported by @aux, or a negative error code
/// otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_read_sink_count")]
extern "C" fn drm_dp_read_sink_count(aux: *mut drm_dp_aux) -> i32 {
let mut count: u8 = 0;
let ret: i32 = drm_dp_dpcd_read_byte(aux, 512u32, core::ptr::addr_of_mut!(count));
if ret < 0 {
return ret;
}
DP_GET_SINK_COUNT!(count)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_read_sink_count_722: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux) -> i32>, *mut ()>(Some(drm_dp_read_sink_count)) };

// I2C-over-AUX implementation
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_i2c_functionality")]
extern "C" fn drm_dp_i2c_functionality(adapter: *mut i2c_adapter) -> u32 {
(1 | (65536 | (131072 | 262144) | (524288 | 1048576) | (2097152 | 4194304) | 8388608 | 33554432 | (67108864 | 134217728) | 8) | 16777216 | 32768 | 2) as u32
}

fn drm_dp_i2c_msg_write_status_update(msg: *mut drm_dp_aux_msg) {
// In case of i2c defer or short i2c ack reply to a write,
// we need to switch to WRITE_STATUS_UPDATE to drain the
// rest of the message
if ((unsafe { (*msg).request }) as i32) & !4 == 0 {
unsafe {
(*msg).request = (((*msg).request as i32) & 4) as u8;
}
unsafe {
(*msg).request = (((*msg).request as i32) | 2) as u8;
}
}
}

// 10 to 16
// preamble + AUX_SYNC_END
/// Calculate the duration of the AUX request/reply in usec. Gives the
/// "best" case estimate, ie. successful while as short as possible.
fn drm_dp_aux_req_duration(msg: *const drm_dp_aux_msg) -> i32 {
let mut len: i32 = 66i32;
if ((unsafe { (*msg).request }) as i32) & 1 == 0 {
len = (len as u64).wrapping_add(unsafe { (*msg).size }.wrapping_mul(8)) as i32;
}
len
}

fn drm_dp_aux_reply_duration(msg: *const drm_dp_aux_msg) -> i32 {
let mut len: i32 = 42i32;
// For read we expect what was asked. For writes there will
// be 0 or 1 data bytes. Assume 0 for the "best" case.
if ((unsafe { (*msg).request }) as i32) & 1 != 0 {
len = (len as u64).wrapping_add(unsafe { (*msg).size }.wrapping_mul(8)) as i32;
}
len
}

// ADDRESS + R/W + ACK/NACK
// DATA + ACK/NACK
/// Calculate the length of the i2c transfer in usec, assuming
/// the i2c bus speed is as specified. Gives the "worst"
/// case estimate, ie. successful while as long as possible.
/// Doesn't account the "MOT" bit, and instead assumes each
/// message includes a START, ADDRESS and STOP. Neither does it
/// account for additional random variables such as clock stretching.
fn drm_dp_i2c_msg_duration(msg: *const drm_dp_aux_msg, i2c_speed_khz: i32) -> i32 {
// AUX bitrate is 1MHz, i2c bitrate as specified
(10u64.wrapping_add(unsafe { (*msg).size }.wrapping_mul(9)).wrapping_add(1).wrapping_mul(1000).wrapping_add(i2c_speed_khz as u64).wrapping_sub(1) / (i2c_speed_khz as u64)) as i32
}

/// Determine how many retries should be attempted to successfully transfer
/// the specified message, based on the estimated durations of the
/// i2c and AUX transfers.
fn drm_dp_i2c_retry_count(msg: *const drm_dp_aux_msg, i2c_speed_khz: i32) -> i32 {
let aux_time_us: i32 = drm_dp_aux_req_duration(msg).wrapping_add(drm_dp_aux_reply_duration(msg));
let i2c_time_us: i32 = drm_dp_i2c_msg_duration(msg, i2c_speed_khz);
i2c_time_us.wrapping_add(aux_time_us.wrapping_add(500)).wrapping_sub(1) / aux_time_us.wrapping_add(500)
}

/// FIXME currently assumes 10 kHz as some real world devices seem
/// to require it. We should query/set the speed via DPCD if supported.
#[unsafe(link_section = ".data..read_mostly")]
static mut dp_aux_i2c_speed_khz: i32 = 10;

fn __check_dp_aux_i2c_speed_khz() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(dp_aux_i2c_speed_khz) }
}

static __param_str_dp_aux_i2c_speed_khz: __SlateAlign16<[u8; 40]> = __SlateAlign16([100u8, 114u8, 109u8, 95u8, 100u8, 105u8, 115u8, 112u8, 108u8, 97u8, 121u8, 95u8, 104u8, 101u8, 108u8, 112u8, 101u8, 114u8, 46u8, 100u8, 112u8, 95u8, 97u8, 117u8, 120u8, 95u8, 105u8, 50u8, 99u8, 95u8, 115u8, 112u8, 101u8, 101u8, 100u8, 95u8, 107u8, 104u8, 122u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_dp_aux_i2c_speed_khz: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_dp_aux_i2c_speed_khz.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_drm_dp_helper_c_1979>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_drm_dp_helper_c_1979_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_drm_dp_helper_c_1979_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_drm_dp_helper_c_1979_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_drm_dp_helper_c_1979_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_drm_dp_helper_c_1979_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 1u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(dp_aux_i2c_speed_khz) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_723: [u8; 53] = [100, 114, 109, 95, 100, 105, 115, 112, 108, 97, 121, 95, 104, 101, 108, 112, 101, 114, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 100, 112, 95, 97, 117, 120, 95, 105, 50, 99, 95, 115, 112, 101, 101, 100, 95, 107, 104, 122, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_724: [u8; 102] = [100, 114, 109, 95, 100, 105, 115, 112, 108, 97, 121, 95, 104, 101, 108, 112, 101, 114, 46, 112, 97, 114, 109, 61, 100, 112, 95, 97, 117, 120, 95, 105, 50, 99, 95, 115, 112, 101, 101, 100, 95, 107, 104, 122, 58, 65, 115, 115, 117, 109, 101, 100, 32, 115, 112, 101, 101, 100, 32, 111, 102, 32, 116, 104, 101, 32, 105, 50, 99, 32, 98, 117, 115, 32, 105, 110, 32, 107, 72, 122, 44, 32, 40, 49, 45, 52, 48, 48, 44, 32, 100, 101, 102, 97, 117, 108, 116, 32, 49, 48, 41, 0];

// Transfer a single I2C-over-AUX message and handle various error conditions,
// retrying the transaction as appropriate.  It is assumed that the
// &drm_dp_aux.transfer function does not modify anything in the msg other than the
// reply field.
//
// Returns bytes transferred on success, or a negative error code on failure.
fn drm_dp_i2c_do_msg(aux: *mut drm_dp_aux, msg: *mut drm_dp_aux_msg) -> i32 {
let mut retry: u32 = 0;
let mut defer_i2c: u32 = 0;
let mut ret: i32 = 0;
// DP1.2 sections 2.7.7.1.5.6.1 and 2.7.7.1.6.6.1: A DP Source device
// is required to retry at least seven times upon receiving AUX_DEFER
// before giving up the AUX transaction.
//
// We also try to account for the i2c bus speed.
let mut max_retries: i32 = 0;
let mut __v0: i32;
let __UNIQUE_ID_x__725: i32 = 7;
let __UNIQUE_ID_y__726: i32 = drm_dp_i2c_retry_count(msg as *const drm_dp_aux_msg, unsafe { dp_aux_i2c_speed_khz });
loop {
if __types_ok!(__UNIQUE_ID_x__725, __UNIQUE_ID_y__726) == 0 {
unsafe { __compiletime_assert_727() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__725 > __UNIQUE_ID_y__726 { __UNIQUE_ID_x__725 } else { __UNIQUE_ID_y__726 };
max_retries = __v0;
while retry < (max_retries as u32).wrapping_add(defer_i2c) {
'__slate_continue_2: {
ret = (unsafe { unsafe { (*aux).transfer }.unwrap()(aux, msg) }) as i32;
if ret < 0 {
if ret == 16i32.wrapping_neg() {
break '__slate_continue_2;
}
// While timeouts can be errors, they're usually normal
// behavior (for instance, when a driver tries to
// communicate with a non-existent DisplayPort device).
// Avoid spamming the kernel log with timeout errors.
if ret == 110i32.wrapping_neg() {
let __v3: bool = drm_debug_enabled!(2i32) && __ratelimit__2!(unsafe { core::ptr::addr_of_mut!(rs_) }) != 0;
if __v3 {
unsafe { drm_dev_printk(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, c"\x017".as_ptr() as *const u8, c"%s: transaction timed out\n".as_ptr() as *const u8, (*aux).name) };
}
} else {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: transaction failed: %d\n".as_ptr() as *const u8, (*aux).name, ret) };
}
return ret;
}
'__slate_break_4: {
match ((unsafe { (*msg).reply }) as i32) & 3i32 {
0 => {
break '__slate_break_4;
// For I2C-over-AUX transactions this isn't enough, we
// need to check for the I2C ACK reply.
}
1 => {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: native nack (result=%d, size=%zu)\n".as_ptr() as *const u8, (*aux).name, ret, (*msg).size) };
return 121i32.wrapping_neg();
}
2 => {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: native defer\n".as_ptr() as *const u8, (*aux).name) };
// We could check for I2C bit rate capabilities and if
// available adjust this interval. We could also be
// more careful with DP-to-legacy adapters where a
// long legacy cable may force very low I2C bit rates.
//
// For now just defer for long enough to hopefully be
// safe for all use-cases.
usleep_range(500u64, 600u64);
break '__slate_continue_2;
}
_ => {
while false {
}
unsafe { _dev_err(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, c"[drm] *ERROR* %s: invalid native reply %#04x\n".as_ptr() as *const u8, (*aux).name, (*msg).reply as i32) };
return 121i32.wrapping_neg();
}
}
}
match ((unsafe { (*msg).reply }) as i32) & 12i32 {
0 => {
if (ret as u64) != unsafe { (*msg).size } {
drm_dp_i2c_msg_write_status_update(msg);
}
// Both native ACK and I2C ACK replies received. We
// can assume the transfer was successful.
return ret;
}
4 => {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: I2C nack (result=%d, size=%zu)\n".as_ptr() as *const u8, (*aux).name, ret, (*msg).size) };
let __v7: u32 = unsafe { (*aux).i2c_nack_count };
unsafe {
(*aux).i2c_nack_count = __v7.wrapping_add(1);
}
return 121i32.wrapping_neg();
}
8 => {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: I2C defer\n".as_ptr() as *const u8, (*aux).name) };
// DP Compliance Test 4.2.2.5 Requirement:
// Must have at least 7 retries for I2C defers on the
// transaction to pass this test
let __v10: u32 = unsafe { (*aux).i2c_defer_count };
unsafe {
(*aux).i2c_defer_count = __v10.wrapping_add(1);
}
if defer_i2c < 7 {
defer_i2c = defer_i2c.wrapping_add(1);
}
usleep_range(500u64, 600u64);
drm_dp_i2c_msg_write_status_update(msg);
}
_ => {
while false {
}
unsafe { _dev_err(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, c"[drm] *ERROR* %s: invalid I2C reply %#04x\n".as_ptr() as *const u8, (*aux).name, (*msg).reply as i32) };
return 121i32.wrapping_neg();
}
}
}
retry = retry.wrapping_add(1);
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Too many retries, giving up\n".as_ptr() as *const u8, (*aux).name) };
121i32.wrapping_neg()
}

static mut rs_: ratelimit_state = {
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

fn drm_dp_i2c_msg_set_request(msg: &mut drm_dp_aux_msg, i2c_msg: *const i2c_msg) {
unsafe {
(*(msg as *mut drm_dp_aux_msg)).request = (if ((*i2c_msg).flags as i32) & 1 != 0 { 1i32 } else { 0i32 }) as u8;
}
if ((unsafe { (*i2c_msg).flags }) as i32) & 32768 == 0 {
unsafe {
(*msg).request = (((*msg).request as i32) | 4) as u8;
}
}
}

/// Keep retrying drm_dp_i2c_do_msg until all data has been transferred.
///
/// Returns an error code on failure, or a recommended transfer size on success.
fn drm_dp_i2c_drain_msg(aux: *mut drm_dp_aux, orig_msg: &drm_dp_aux_msg) -> i32 {
let mut err: i32 = 0;
let mut ret: i32 = (unsafe { (*(orig_msg as *const drm_dp_aux_msg)).size }) as i32;
let mut msg: drm_dp_aux_msg = unsafe { *(orig_msg as *const drm_dp_aux_msg) };
while msg.size > 0 {
err = drm_dp_i2c_do_msg(aux, core::ptr::addr_of_mut!(msg));
if err <= 0 {
return if err == 0 { 71i32.wrapping_neg() } else { err };
}
if (err as u64) < msg.size && err < ret {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Partial I2C reply: requested %zu bytes got %d bytes\n".as_ptr() as *const u8, (*aux).name, msg.size, err) };
ret = err;
}
let __v0: u64 = msg.size;
msg.size = __v0.wrapping_sub(err as u64);
msg.buffer = (unsafe { (msg.buffer as *const u8).wrapping_offset(err as isize) }) as *mut ();
}
ret
}

/// Bizlink designed DP->DVI-D Dual Link adapters require the I2C over AUX
/// packets to be as large as possible. If not, the I2C transactions never
/// succeed. Hence the default is maximum.
#[unsafe(link_section = ".data..read_mostly")]
static mut dp_aux_i2c_transfer_size: i32 = 16;

fn __check_dp_aux_i2c_transfer_size() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(dp_aux_i2c_transfer_size) }
}

static __param_str_dp_aux_i2c_transfer_size: __SlateAlign16<[u8; 44]> = __SlateAlign16([100u8, 114u8, 109u8, 95u8, 100u8, 105u8, 115u8, 112u8, 108u8, 97u8, 121u8, 95u8, 104u8, 101u8, 108u8, 112u8, 101u8, 114u8, 46u8, 100u8, 112u8, 95u8, 97u8, 117u8, 120u8, 95u8, 105u8, 50u8, 99u8, 95u8, 116u8, 114u8, 97u8, 110u8, 115u8, 102u8, 101u8, 114u8, 95u8, 115u8, 105u8, 122u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_dp_aux_i2c_transfer_size: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_dp_aux_i2c_transfer_size.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_drm_dp_helper_c_2144>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_drm_dp_helper_c_2144_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_drm_dp_helper_c_2144_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_drm_dp_helper_c_2144_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_drm_dp_helper_c_2144_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_drm_dp_helper_c_2144_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 1u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(dp_aux_i2c_transfer_size) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_728: [u8; 57] = [100, 114, 109, 95, 100, 105, 115, 112, 108, 97, 121, 95, 104, 101, 108, 112, 101, 114, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 100, 112, 95, 97, 117, 120, 95, 105, 50, 99, 95, 116, 114, 97, 110, 115, 102, 101, 114, 95, 115, 105, 122, 101, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_729: [u8; 136] = [100, 114, 109, 95, 100, 105, 115, 112, 108, 97, 121, 95, 104, 101, 108, 112, 101, 114, 46, 112, 97, 114, 109, 61, 100, 112, 95, 97, 117, 120, 95, 105, 50, 99, 95, 116, 114, 97, 110, 115, 102, 101, 114, 95, 115, 105, 122, 101, 58, 78, 117, 109, 98, 101, 114, 32, 111, 102, 32, 98, 121, 116, 101, 115, 32, 116, 111, 32, 116, 114, 97, 110, 115, 102, 101, 114, 32, 105, 110, 32, 97, 32, 115, 105, 110, 103, 108, 101, 32, 73, 50, 67, 32, 111, 118, 101, 114, 32, 68, 80, 32, 65, 85, 88, 32, 67, 72, 32, 109, 101, 115, 115, 97, 103, 101, 44, 32, 40, 49, 45, 49, 54, 44, 32, 100, 101, 102, 97, 117, 108, 116, 32, 49, 54, 41, 0];

#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_i2c_xfer")]
extern "C" fn drm_dp_i2c_xfer(adapter: *mut i2c_adapter, msgs: *mut i2c_msg, num: i32) -> i32 {
let aux: *mut drm_dp_aux = (unsafe { (*adapter).algo_data }) as *mut drm_dp_aux;
let mut j: u32 = 0;
let mut transfer_size: u32 = 0;
let mut msg: drm_dp_aux_msg = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(msg) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
let mut err: i32 = 0;
if unsafe { (*aux).powered_down } {
return 16i32.wrapping_neg();
}
let mut __v0: i32;
let __UNIQUE_ID_v__730: i32 = unsafe { dp_aux_i2c_transfer_size };
let __UNIQUE_ID_l__731: i32 = 1;
let __UNIQUE_ID_h__732: i32 = 16;
loop {
if !(!statically_true!(__UNIQUE_ID_l__731 > __UNIQUE_ID_h__732)) {
unsafe { __compiletime_assert_733() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __types_ok3!(__UNIQUE_ID_v__730, __UNIQUE_ID_l__731, __UNIQUE_ID_h__732) == 0 {
unsafe { __compiletime_assert_734() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_v__730 >= __UNIQUE_ID_h__732 { __UNIQUE_ID_h__732 } else { if __UNIQUE_ID_v__730 <= __UNIQUE_ID_l__731 { __UNIQUE_ID_l__731 } else { __UNIQUE_ID_v__730 } };
unsafe {
dp_aux_i2c_transfer_size = __v0;
}
unsafe { memset(core::ptr::addr_of_mut!(msg) as *mut (), 0i32, core::mem::size_of::<drm_dp_aux_msg>() as u64) };
for i in 0..num as u32 {
msg.address = (unsafe { (*msgs.wrapping_offset(i as isize)).addr }) as u32;
if !unsafe { (*aux).no_zero_sized } {
drm_dp_i2c_msg_set_request(unsafe { &mut (*core::ptr::addr_of_mut!(msg)) }, msgs.wrapping_offset(i as isize) as *const i2c_msg);
// Send a bare address packet to start the transaction.
// Zero sized messages specify an address only (bare
// address) transaction.
msg.buffer = core::ptr::null_mut::<()>();
msg.size = 0;
err = drm_dp_i2c_do_msg(aux, core::ptr::addr_of_mut!(msg));
}
// Reset msg.request in case in case it got
// changed into a WRITE_STATUS_UPDATE.
drm_dp_i2c_msg_set_request(unsafe { &mut (*core::ptr::addr_of_mut!(msg)) }, msgs.wrapping_offset(i as isize) as *const i2c_msg);
if err < 0 {
break;
}
// We want each transaction to be as large as possible, but
// we'll go to smaller sizes if the hardware gives us a
// short reply.
transfer_size = (unsafe { dp_aux_i2c_transfer_size }) as u32;
j = 0;
while j < (((unsafe { (*msgs.wrapping_offset(i as isize)).len }) as i32) as u32) {
msg.buffer = unsafe { (*msgs.wrapping_offset(i as isize)).buf }.wrapping_offset(j as isize) as *mut ();
let mut __v4: u32;
let __UNIQUE_ID_x__735: u32 = transfer_size;
let __UNIQUE_ID_y__736: u32 = (((unsafe { (*msgs.wrapping_offset(i as isize)).len }) as i32) as u32).wrapping_sub(j);
loop {
if __types_ok__3!(__UNIQUE_ID_x__735, __UNIQUE_ID_y__736) == 0 {
unsafe { __compiletime_assert_737() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = if __UNIQUE_ID_x__735 < __UNIQUE_ID_y__736 { __UNIQUE_ID_x__735 } else { __UNIQUE_ID_y__736 };
msg.size = __v4 as u64;
if (j as u64).wrapping_add(msg.size) == (((unsafe { (*msgs.wrapping_offset(i as isize)).len }) as i32) as u64) && unsafe { (*aux).no_zero_sized } {
let __v7: u8 = ((msg.request as i32) & !4) as u8;
msg.request = __v7;
}
err = drm_dp_i2c_drain_msg(aux, unsafe { &(*core::ptr::addr_of_mut!(msg)) });
// Reset msg.request in case in case it got
// changed into a WRITE_STATUS_UPDATE.
drm_dp_i2c_msg_set_request(unsafe { &mut (*core::ptr::addr_of_mut!(msg)) }, msgs.wrapping_offset(i as isize) as *const i2c_msg);
if err < 0 {
break;
}
transfer_size = err as u32;
j = (j as u64).wrapping_add(msg.size) as u32;
}
if err < 0 {
break;
}
}
if err >= 0 {
err = num;
}
if !unsafe { (*aux).no_zero_sized } {
// Send a bare address packet to close out the transaction.
// Zero sized messages specify an address only (bare
// address) transaction.
let __v14: u8 = ((msg.request as i32) & !4) as u8;
msg.request = __v14;
msg.buffer = core::ptr::null_mut::<()>();
msg.size = 0;
drm_dp_i2c_do_msg(aux, core::ptr::addr_of_mut!(msg));
}
err
}

static drm_dp_i2c_algo: __SlateConst<i2c_algorithm> = __SlateConst({
let mut __t0: i2c_algorithm = unsafe { core::mem::zeroed() };
__t0.__slate_anon_0 = __SlateRecord_i2c_h_551 { master_xfer: Some(drm_dp_i2c_xfer) };
__t0.functionality = Some(drm_dp_i2c_functionality);
__t0
});

fn i2c_to_aux(i2c: *mut i2c_adapter) -> *mut drm_dp_aux {
container_of__39!(i2c, drm_dp_aux)
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.lock_bus")]
extern "C" fn lock_bus(i2c: *mut i2c_adapter, flags: u32) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*i2c_to_aux(i2c)).hw_mutex)) };
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.trylock_bus")]
extern "C" fn trylock_bus(i2c: *mut i2c_adapter, flags: u32) -> i32 {
unsafe { mutex_trylock(core::ptr::addr_of_mut!((*i2c_to_aux(i2c)).hw_mutex)) }
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.unlock_bus")]
extern "C" fn unlock_bus(i2c: *mut i2c_adapter, flags: u32) {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*i2c_to_aux(i2c)).hw_mutex)) };
}

static drm_dp_i2c_lock_ops: __SlateConst<i2c_lock_operations> = __SlateConst(i2c_lock_operations { lock_bus: Some(lock_bus), trylock_bus: Some(trylock_bus), unlock_bus: Some(unlock_bus) });

fn drm_dp_aux_get_crc(aux: *mut drm_dp_aux, crc: *mut u8) -> i32 {
let mut buf: u8 = 0;
let mut count: u8 = 0;
let mut ret: i32 = drm_dp_dpcd_read_byte(aux, 624u32, core::ptr::addr_of_mut!(buf));
if ret < 0 {
return ret;
}
let mut __v0: i64;
let __ret_warn_on: i32 = ((buf as i32) & 1i32 == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("738: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 738b - ., 3; .popsection\n# {0}", const 738 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_dp_aux_get_crc_1, sym __slate_asm_string_drm_dp_aux_get_crc_2, const 2268 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("739: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 739b - ., 4; .popsection\n# {0}", const 739 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
ret = drm_dp_dpcd_read_byte(aux, 582u32, core::ptr::addr_of_mut!(buf));
if ret < 0 {
return ret;
}
count = ((buf as i32) & 15) as u8;
if (count as i32) == ((unsafe { (*aux).crc_count }) as i32) {
return 11i32.wrapping_neg();
}
// No CRC yet
unsafe {
(*aux).crc_count = count;
}
// At DP_TEST_CRC_R_CR, there's 6 bytes containing CRC data, 2 bytes
// per component (RGB or CrYCb).
drm_dp_dpcd_read_data(aux, 576u32, crc as *mut (), 6u64)
}

#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_aux_crc_work")]
extern "C" fn drm_dp_aux_crc_work(work: *mut work_struct) {
let aux: *mut drm_dp_aux = container_of__32!(work, drm_dp_aux);
let mut crc_bytes: [u8; 6] = [0u8; 6];
let mut crcs: [u32; 3] = [0u32; 3];
let mut ret: i32 = 0;
let mut __v1: i64;
let __ret_warn_on: i32 = unsafe { (*aux).crtc }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("740: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 740b - ., 3; .popsection\n# {0}", const 740 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_dp_aux_crc_work_2, sym __slate_asm_string_drm_dp_aux_crc_work_3, const 2296 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("741: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 741b - ., 4; .popsection\n# {0}", const 741 as i32, options(nostack, att_syntax));
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
let crtc: *mut drm_crtc = unsafe { (*aux).crtc };
'__slate_break_6: while unsafe { (*crtc).crc.opened } {
'__slate_continue_6: {
unsafe { drm_crtc_wait_one_vblank(crtc) };
if !unsafe { (*crtc).crc.opened } {
break '__slate_break_6;
}
ret = drm_dp_aux_get_crc(aux, crc_bytes.as_mut_ptr() as *mut u8);
if ret == 11i32.wrapping_neg() {
usleep_range(1000u64, 2000u64);
ret = drm_dp_aux_get_crc(aux, crc_bytes.as_mut_ptr() as *mut u8);
}
if ret == 11i32.wrapping_neg() {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Get CRC failed after retrying: %d\n".as_ptr() as *const u8, (*aux).name, ret) };
break '__slate_continue_6;
} else {
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Failed to get a CRC: %d\n".as_ptr() as *const u8, (*aux).name, ret) };
break '__slate_continue_6;
}
}
crcs[0] = ((crc_bytes[0] as i32) | (crc_bytes[1] as i32) << 8i32) as u32;
crcs[1] = ((crc_bytes[2] as i32) | (crc_bytes[3] as i32) << 8i32) as u32;
crcs[2] = ((crc_bytes[4] as i32) | (crc_bytes[5] as i32) << 8i32) as u32;
unsafe { drm_crtc_add_crc_entry(crtc, false, 0u32, crcs.as_mut_ptr() as *mut u32) };
}
}
}

/// drm_dp_remote_aux_init() - minimally initialise a remote aux channel
/// @aux: DisplayPort AUX channel
///
/// Used for remote aux channel in general. Merely initialize the crc work
/// struct.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_remote_aux_init")]
extern "C" fn drm_dp_remote_aux_init(aux: *mut drm_dp_aux) {
let mut __slate_compound_0: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
'__slate_break_2: loop {
'__slate_break_1: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*aux).crc_work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*aux).crc_work)).data = *{
__slate_compound_0 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*aux).crc_work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*aux).crc_work)).func = Some(drm_dp_aux_crc_work);
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

static mut __key__2336: lock_class_key = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_remote_aux_init_742: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux)>, *mut ()>(Some(drm_dp_remote_aux_init)) };

/// drm_dp_aux_init() - minimally initialise an aux channel
/// @aux: DisplayPort AUX channel
///
/// If you need to use the drm_dp_aux's i2c adapter prior to registering it with
/// the outside world, call drm_dp_aux_init() first. For drivers which are
/// grandparents to their AUX adapters (e.g. the AUX adapter is parented by a
/// &drm_connector), you must still call drm_dp_aux_register() once the connector
/// has been registered to allow userspace access to the auxiliary DP channel.
/// Likewise, for such drivers you should also assign &drm_dp_aux.drm_dev as
/// early as possible so that the &drm_device that corresponds to the AUX adapter
/// may be mentioned in debugging output from the DRM DP helpers.
///
/// For devices which use a separate platform device for their AUX adapters, this
/// may be called as early as required by the driver.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_aux_init")]
extern "C" fn drm_dp_aux_init(aux: *mut drm_dp_aux) {
let mut __slate_compound_2: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
loop {
__mutex_init(unsafe { core::ptr::addr_of_mut!((*aux).hw_mutex) }, c"&aux->hw_mutex".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key__2359) });
if true {
break;
}
}
loop {
__mutex_init(unsafe { core::ptr::addr_of_mut!((*aux).cec.lock) }, c"&aux->cec.lock".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key__2360) });
if true {
break;
}
}
'__slate_break_4: loop {
'__slate_break_3: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*aux).crc_work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*aux).crc_work)).data = *{
__slate_compound_2 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_2)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*aux).crc_work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*aux).crc_work)).func = Some(drm_dp_aux_crc_work);
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
(*aux).ddc.algo = core::ptr::addr_of!(drm_dp_i2c_algo.0);
}
unsafe {
(*aux).ddc.algo_data = aux as *mut ();
}
unsafe {
(*aux).ddc.retries = 3;
}
unsafe {
(*aux).ddc.lock_ops = core::ptr::addr_of!(drm_dp_i2c_lock_ops.0);
}
}

static mut __key__2359: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__2360: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__2361: lock_class_key = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_aux_init_743: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux)>, *mut ()>(Some(drm_dp_aux_init)) };

/// drm_dp_aux_register() - initialise and register aux channel
/// @aux: DisplayPort AUX channel
///
/// Automatically calls drm_dp_aux_init() if this hasn't been done yet. This
/// should only be called once the parent of @aux, &drm_dp_aux.dev, is
/// initialized. For devices which are grandparents of their AUX channels,
/// &drm_dp_aux.dev will typically be the &drm_connector &device which
/// corresponds to @aux. For these devices, it's advised to call
/// drm_dp_aux_register() in &drm_connector_funcs.late_register, and likewise to
/// call drm_dp_aux_unregister() in &drm_connector_funcs.early_unregister.
/// Functions which don't follow this will likely Oops when
/// %CONFIG_DRM_DISPLAY_DP_AUX_CHARDEV is enabled.
///
/// For devices where the AUX channel is a device that exists independently of
/// the &drm_device that uses it, such as SoCs and bridge devices, it is
/// recommended to call drm_dp_aux_register() after a &drm_device has been
/// assigned to &drm_dp_aux.drm_dev, and likewise to call
/// drm_dp_aux_unregister() once the &drm_device should no longer be associated
/// with the AUX channel (e.g. on bridge detach).
///
/// Drivers which need to use the aux channel before either of the two points
/// mentioned above need to call drm_dp_aux_init() in order to use the AUX
/// channel before registration.
///
/// Returns 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_aux_register")]
extern "C" fn drm_dp_aux_register(aux: *mut drm_dp_aux) -> i32 {
let mut ret: i32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = unsafe { (*aux).drm_dev }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("744: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 744b - ., 3; .popsection\n# {0}", const 744 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_dp_aux_register_1, sym __slate_asm_string_drm_dp_aux_register_2, const 2402 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("745: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 745b - ., 4; .popsection\n# {0}", const 745 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if unsafe { (*aux).ddc.algo }.is_null() {
drm_dp_aux_init(aux);
}
unsafe {
(*aux).ddc.owner = core::ptr::null_mut::<module>();
}
unsafe {
(*aux).ddc.dev.parent = (*aux).dev;
}
let __v5: *const u8 = if !unsafe { (*aux).name }.is_null() {
unsafe { (*aux).name }
} else {
dev_name((unsafe { (*aux).dev }) as *const device)
};
unsafe { sized_strscpy((*aux).ddc.name.as_mut_ptr() as *mut u8, __v5, (core::mem::size_of::<[u8; 48]>() as u64).wrapping_add(__must_be_cstr!() as u64).wrapping_add(__must_be_cstr__3!() as u64)) };
ret = drm_dp_aux_register_devnode(aux);
if ret != 0 {
return ret;
}
ret = unsafe { i2c_add_adapter(core::ptr::addr_of_mut!((*aux).ddc)) };
if ret != 0 {
drm_dp_aux_unregister_devnode(aux);
return ret;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_aux_register_746: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux) -> i32>, *mut ()>(Some(drm_dp_aux_register)) };

/// drm_dp_aux_unregister() - unregister an AUX adapter
/// @aux: DisplayPort AUX channel
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_aux_unregister")]
extern "C" fn drm_dp_aux_unregister(aux: *mut drm_dp_aux) {
drm_dp_aux_unregister_devnode(aux);
unsafe { i2c_del_adapter(core::ptr::addr_of_mut!((*aux).ddc)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_aux_unregister_747: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux)>, *mut ()>(Some(drm_dp_aux_unregister)) };

/// drm_dp_psr_setup_time() - PSR setup in time usec
/// @psr_cap: PSR capabilities from DPCD
///
/// Returns:
/// PSR setup time for the panel in microseconds,  negative
/// error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_psr_setup_time")]
extern "C" fn drm_dp_psr_setup_time(psr_cap: *const u8) -> i32 {
let i: i32 = (((unsafe { *psr_cap.wrapping_offset(1 as isize) }) as i32) & 14i32) >> 1i32;
if (i as u64) >= ARRAY_SIZE__5!() {
return 22i32.wrapping_neg();
}
(unsafe { psr_setup_time_us[i as usize] }) as i32
}

static psr_setup_time_us: [u16; 7] = [330, 275, 220, 165, 110, 55, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_psr_setup_time_748: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> i32>, *mut ()>(Some(drm_dp_psr_setup_time)) };

/// drm_dp_start_crc() - start capture of frame CRCs
/// @aux: DisplayPort AUX channel
/// @crtc: CRTC displaying the frames whose CRCs are to be captured
///
/// Returns 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_start_crc")]
extern "C" fn drm_dp_start_crc(aux: *mut drm_dp_aux, crtc: *mut drm_crtc) -> i32 {
let mut buf: u8 = 0;
let mut ret: i32 = drm_dp_dpcd_read_byte(aux, 624u32, core::ptr::addr_of_mut!(buf));
if ret < 0 {
return ret;
}
ret = drm_dp_dpcd_write_byte(aux, 624u32, ((buf as i32) | 1i32) as u8);
if ret < 0 {
return ret;
}
unsafe {
(*aux).crc_count = 0;
}
unsafe {
(*aux).crtc = crtc;
}
schedule_work(unsafe { core::ptr::addr_of_mut!((*aux).crc_work) });
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_start_crc_749: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *mut drm_crtc) -> i32>, *mut ()>(Some(drm_dp_start_crc)) };

/// drm_dp_stop_crc() - stop capture of frame CRCs
/// @aux: DisplayPort AUX channel
///
/// Returns 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_stop_crc")]
extern "C" fn drm_dp_stop_crc(aux: *mut drm_dp_aux) -> i32 {
let mut buf: u8 = 0;
let mut ret: i32 = drm_dp_dpcd_read_byte(aux, 624u32, core::ptr::addr_of_mut!(buf));
if ret < 0 {
return ret;
}
ret = drm_dp_dpcd_write_byte(aux, 624u32, ((buf as i32) & -2i32) as u8);
if ret < 0 {
return ret;
}
unsafe { flush_work(core::ptr::addr_of_mut!((*aux).crc_work)) };
unsafe {
(*aux).crtc = core::ptr::null_mut::<drm_crtc>();
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_stop_crc_750: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux) -> i32>, *mut ()>(Some(drm_dp_stop_crc)) };

#[repr(C)]
#[derive(Clone, Copy)]
struct dpcd_quirk {
    oui: [u8; 3],
    device_id: [u8; 6],
    is_branch: bool,
    quirks: u32,
}


/// Analogix 7737 needs reduced M and N at HBR2 link rates
/// LG LP140WF6-SPM1 eDP panel
/// Apple panels need some additional handling to support PSR
/// CH7511 seems to leave SINK_COUNT zeroed
/// Synaptics DP1.4 MST hubs can support DSC without virtual DPCD
/// Realtek DP1.4 MST hubs can support DSC without virtual DPCD
/// Synaptics DP1.4 MST hubs require DSC for some modes on which it applies HBLANK expansion.
/// MediaTek panels (at least in U3224KBA) require DSC for modes with a short HBLANK on UHBR links.
/// Apple MacBookPro 2017 15 inch eDP Retina panel reports too low DP_MAX_LINK_RATE
/// Synaptics Panamera supports only a compressed bpp of 12 above 50% of its max DSC pixel throughput
static dpcd_quirk_list: __SlateAlign16<[dpcd_quirk; 12]> = __SlateAlign16([{
let mut __t0: dpcd_quirk = unsafe { core::mem::zeroed() };
__t0.oui = [0u8, 34u8, 185u8];
__t0.device_id = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t0.is_branch = 1i32 != 0i32;
__t0.quirks = 1u32;
__t0
}, {
let mut __t1: dpcd_quirk = unsafe { core::mem::zeroed() };
__t1.oui = [0u8, 34u8, 185u8];
__t1.device_id = [115u8, 105u8, 118u8, 97u8, 114u8, 84u8];
__t1.is_branch = 0i32 != 0i32;
__t1.quirks = 1u32;
__t1
}, {
let mut __t2: dpcd_quirk = unsafe { core::mem::zeroed() };
__t2.oui = [0u8, 16u8, 250u8];
__t2.device_id = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2.is_branch = 0i32 != 0i32;
__t2.quirks = 2u32;
__t2
}, {
let mut __t3: dpcd_quirk = unsafe { core::mem::zeroed() };
__t3.oui = [0u8, 0u8, 0u8];
__t3.device_id = [67u8, 72u8, 55u8, 53u8, 49u8, 49u8];
__t3.is_branch = 0i32 != 0i32;
__t3.quirks = 4u32;
__t3
}, {
let mut __t4: dpcd_quirk = unsafe { core::mem::zeroed() };
__t4.oui = [144u8, 204u8, 36u8];
__t4.device_id = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t4.is_branch = 1i32 != 0i32;
__t4.quirks = 8u32;
__t4
}, {
let mut __t5: dpcd_quirk = unsafe { core::mem::zeroed() };
__t5.oui = [0u8, 224u8, 76u8];
__t5.device_id = [68u8, 112u8, 49u8, 46u8, 52u8, 0u8];
__t5.is_branch = 1i32 != 0i32;
__t5.quirks = 8u32;
__t5
}, {
let mut __t6: dpcd_quirk = unsafe { core::mem::zeroed() };
__t6.oui = [144u8, 204u8, 36u8];
__t6.device_id = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t6.is_branch = 1i32 != 0i32;
__t6.quirks = 32u32;
__t6
}, {
let mut __t7: dpcd_quirk = unsafe { core::mem::zeroed() };
__t7.oui = [0u8, 12u8, 231u8];
__t7.device_id = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t7.is_branch = 0i32 != 0i32;
__t7.quirks = 32u32;
__t7
}, {
let mut __t8: dpcd_quirk = unsafe { core::mem::zeroed() };
__t8.oui = [0u8, 16u8, 250u8];
__t8.device_id = [101u8, 68u8, 21u8, 101u8, 98u8, 97u8];
__t8.is_branch = 0i32 != 0i32;
__t8.quirks = 16u32;
__t8
}, {
let mut __t9: dpcd_quirk = unsafe { core::mem::zeroed() };
__t9.oui = [144u8, 204u8, 36u8];
__t9.device_id = [83u8, 89u8, 78u8, 65u8, 83u8, 34u8];
__t9.is_branch = 1i32 != 0i32;
__t9.quirks = 64u32;
__t9
}, {
let mut __t10: dpcd_quirk = unsafe { core::mem::zeroed() };
__t10.oui = [144u8, 204u8, 36u8];
__t10.device_id = [83u8, 89u8, 78u8, 65u8, 83u8, 49u8];
__t10.is_branch = 1i32 != 0i32;
__t10.quirks = 64u32;
__t10
}, {
let mut __t11: dpcd_quirk = unsafe { core::mem::zeroed() };
__t11.oui = [144u8, 204u8, 36u8];
__t11.device_id = [83u8, 89u8, 78u8, 65u8, 83u8, 51u8];
__t11.is_branch = 1i32 != 0i32;
__t11.quirks = 64u32;
__t11
}]);

/// Get a bit mask of DPCD quirks for the sink/branch device identified by
/// ident. The quirk data is shared but it's up to the drivers to act on the
/// data.
///
/// For now, only the OUI (first three bytes) is used, but this may be extended
/// to device identification string and hardware/firmware revisions later.
fn drm_dp_get_quirks(ident: *const drm_dp_dpcd_ident, is_branch: bool) -> u32 {
let mut quirks: u32 = 0;
let mut i: i32 = 0;
let mut any_device: [u8; 6] = [0, 0, 0, 0, 0, 0];
while (i as u64) < ARRAY_SIZE__22!() {
let quirk: *const dpcd_quirk = unsafe { core::ptr::addr_of!(dpcd_quirk_list.0) as *const dpcd_quirk }.wrapping_offset(i as isize);
if ((unsafe { (*quirk).is_branch }) as i32) == (is_branch as i32) {
if (unsafe { memcmp(((*quirk).oui.as_ptr() as *const u8) as *const (), ((*ident).oui.as_ptr() as *const u8) as *const (), core::mem::size_of::<[u8; 3]>() as u64) }) == 0 {
let __v1: bool = (unsafe { memcmp(((*quirk).device_id.as_ptr() as *const u8) as *const (), any_device.as_mut_ptr() as *const (), core::mem::size_of::<[u8; 6]>() as u64) }) != 0 && (unsafe { memcmp(((*quirk).device_id.as_ptr() as *const u8) as *const (), ((*ident).device_id.as_ptr() as *const u8) as *const (), core::mem::size_of::<[u8; 6]>() as u64) }) != 0;
if !__v1 {
quirks |= unsafe { (*quirk).quirks };
}
}
}
i = i.wrapping_add(1);
}
quirks
}

fn drm_dp_read_ident(aux: *mut drm_dp_aux, offset: u32, ident: *mut drm_dp_dpcd_ident) -> i32 {
drm_dp_dpcd_read_data(aux, offset, ident as *mut (), core::mem::size_of::<drm_dp_dpcd_ident>() as u64)
}

fn drm_dp_dump_desc(aux: *mut drm_dp_aux, device_name: *const u8, desc: *const drm_dp_desc) {
let ident: *const drm_dp_dpcd_ident = unsafe { core::ptr::addr_of!((*desc).ident) };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: %s: OUI %*phD dev-ID %*pE HW-rev %d.%d SW-rev %d.%d quirks 0x%04x\n".as_ptr() as *const u8, (*aux).name, device_name, core::mem::size_of::<[u8; 3]>() as i32, (*ident).oui.as_ptr() as *const u8, strnlen(((*ident).device_id.as_ptr() as *const u8) as *const u8, core::mem::size_of::<[u8; 6]>() as u64) as i32, (*ident).device_id.as_ptr() as *const u8, ((*ident).hw_rev as i32) >> 4i32, ((*ident).hw_rev as i32) & 15, (*ident).sw_major_rev as i32, (*ident).sw_minor_rev as i32, (*desc).quirks) };
}

/// drm_dp_read_desc - read sink/branch descriptor from DPCD
/// @aux: DisplayPort AUX channel
/// @desc: Device descriptor to fill from DPCD
/// @is_branch: true for branch devices, false for sink devices
///
/// Read DPCD 0x400 (sink) or 0x500 (branch) into @desc. Also debug log the
/// identification.
///
/// Returns 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_read_desc")]
extern "C" fn drm_dp_read_desc(aux: *mut drm_dp_aux, desc: *mut drm_dp_desc, is_branch: bool) -> i32 {
let ident: *mut drm_dp_dpcd_ident = unsafe { core::ptr::addr_of_mut!((*desc).ident) };
let offset: u32 = (if is_branch { 1280i32 } else { 1024i32 }) as u32;
let ret: i32 = drm_dp_read_ident(aux, offset, ident);
if ret < 0 {
return ret;
}
unsafe {
(*desc).quirks = drm_dp_get_quirks(ident as *const drm_dp_dpcd_ident, is_branch);
}
drm_dp_dump_desc(aux, (if is_branch { c"DP branch".as_ptr() as *mut u8 } else { c"DP sink".as_ptr() as *mut u8 }) as *const u8, desc as *const drm_dp_desc);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_read_desc_751: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *mut drm_dp_desc, bool) -> i32>, *mut ()>(Some(drm_dp_read_desc)) };

/// drm_dp_dump_lttpr_desc - read and dump the DPCD descriptor for an LTTPR PHY
/// @aux: DisplayPort AUX channel
/// @dp_phy: LTTPR PHY instance
///
/// Read the DPCD LTTPR PHY descriptor for @dp_phy and print a debug message
/// with its details to dmesg.
///
/// Returns 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dump_lttpr_desc")]
extern "C" fn drm_dp_dump_lttpr_desc(aux: *mut drm_dp_aux, dp_phy: u32) -> i32 {
let mut desc: drm_dp_desc = {
let mut __t0: drm_dp_desc = unsafe { core::mem::zeroed() };
__t0
};
let mut ret: i32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = ((dp_phy as u32) < 1 || (dp_phy as u32) > 8) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device), dev_name(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device), c"drm_WARN_ON(dp_phy < DP_PHY_LTTPR1 || dp_phy > DP_MAX_LTTPR_COUNT)".as_ptr() as *mut u8) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t1: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t1, sym __slate_asm_string_drm_dp_dump_lttpr_desc_3, sym __slate_asm_string_drm_dp_dump_lttpr_desc_4, const 2669 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t1 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, dev_driver_string(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device), dev_name(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device), c"drm_WARN_ON(dp_phy < DP_PHY_LTTPR1 || dp_phy > DP_MAX_LTTPR_COUNT)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return 22i32.wrapping_neg();
}
ret = drm_dp_read_ident(aux, DP_OUI_PHY_REPEATER!(dp_phy), core::ptr::addr_of_mut!(desc.ident));
if ret < 0 {
return ret;
}
drm_dp_dump_desc(aux, drm_dp_phy_name(dp_phy), core::ptr::addr_of_mut!(desc) as *const drm_dp_desc);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_752: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dump_lttpr_desc_753: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, u32) -> i32>, *mut ()>(Some(drm_dp_dump_lttpr_desc)) };

/// drm_dp_dsc_sink_bpp_incr() - Get bits per pixel increment
/// @dsc_dpcd: DSC capabilities from DPCD
///
/// Returns the bpp precision supported by the DP sink.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dsc_sink_bpp_incr")]
extern "C" fn drm_dp_dsc_sink_bpp_incr(dsc_dpcd: *const u8) -> u8 {
let bpp_increment_dpcd: u8 = unsafe { *dsc_dpcd.wrapping_offset(15i32 as isize) };
match (bpp_increment_dpcd as i32) & 7 {
0 => {
return 16;
}
1 => {
return 8;
}
2 => {
return 4;
}
3 => {
return 2;
}
4 => {
return 1;
}
_ => {
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dsc_sink_bpp_incr_754: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> u8>, *mut ()>(Some(drm_dp_dsc_sink_bpp_incr)) };

/// drm_dp_dsc_slice_count_to_mask() - Convert a slice count to a slice count mask
/// @slice_count: slice count
///
/// Convert @slice_count to a slice count mask.
///
/// Returns the slice count mask.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dsc_slice_count_to_mask")]
extern "C" fn drm_dp_dsc_slice_count_to_mask(slice_count: i32) -> u32 {
BIT!(slice_count.wrapping_sub(1)) as u32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dsc_slice_count_to_mask_755: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32) -> u32>, *mut ()>(Some(drm_dp_dsc_slice_count_to_mask)) };

/// drm_dp_dsc_sink_slice_count_mask() - Get the mask of valid DSC sink slice counts
/// @dsc_dpcd: the sink's DSC DPCD capabilities
/// @is_edp: %true for an eDP sink
///
/// Get the mask of supported slice counts from the sink's DSC DPCD register.
///
/// Returns:
/// Mask of slice counts supported by the DSC sink:
/// - > 0: bit#0,1,3,5..,23 set if the sink supports 1,2,4,6..,24 slices
/// - 0:   if the sink doesn't support any slices
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dsc_sink_slice_count_mask")]
extern "C" fn drm_dp_dsc_sink_slice_count_mask(dsc_dpcd: *const u8, is_edp: bool) -> u32 {
let slice_cap1: u8 = unsafe { *dsc_dpcd.wrapping_offset(4i32 as isize) };
let mut mask: u32 = 0;
if !is_edp {
// For DP, use values from DSC_SLICE_CAP_1 and DSC_SLICE_CAP2
let slice_cap2: u8 = unsafe { *dsc_dpcd.wrapping_offset(13i32 as isize) };
if (slice_cap2 as i32) & 4i32 != 0 {
let __v0: u32 = drm_dp_dsc_slice_count_to_mask(24i32);
mask |= __v0;
}
if (slice_cap2 as i32) & 2i32 != 0 {
let __v3: u32 = drm_dp_dsc_slice_count_to_mask(20i32);
mask |= __v3;
}
if (slice_cap2 as i32) & 1i32 != 0 {
let __v6: u32 = drm_dp_dsc_slice_count_to_mask(16i32);
mask |= __v6;
}
}
// DP, eDP v1.5+
if (slice_cap1 as i32) & 128i32 != 0 {
let __v9: u32 = drm_dp_dsc_slice_count_to_mask(12i32);
mask |= __v9;
}
if (slice_cap1 as i32) & 64i32 != 0 {
let __v12: u32 = drm_dp_dsc_slice_count_to_mask(10i32);
mask |= __v12;
}
if (slice_cap1 as i32) & 32i32 != 0 {
let __v15: u32 = drm_dp_dsc_slice_count_to_mask(8i32);
mask |= __v15;
}
if (slice_cap1 as i32) & 16i32 != 0 {
let __v18: u32 = drm_dp_dsc_slice_count_to_mask(6i32);
mask |= __v18;
}
// DP, eDP v1.4+
if (slice_cap1 as i32) & 8i32 != 0 {
let __v21: u32 = drm_dp_dsc_slice_count_to_mask(4i32);
mask |= __v21;
}
if (slice_cap1 as i32) & 2i32 != 0 {
let __v24: u32 = drm_dp_dsc_slice_count_to_mask(2i32);
mask |= __v24;
}
if (slice_cap1 as i32) & 1i32 != 0 {
let __v27: u32 = drm_dp_dsc_slice_count_to_mask(1i32);
mask |= __v27;
}
mask
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dsc_sink_slice_count_mask_756: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, bool) -> u32>, *mut ()>(Some(drm_dp_dsc_sink_slice_count_mask)) };

/// drm_dp_dsc_sink_max_slice_count() - Get the max slice count
/// supported by the DSC sink.
/// @dsc_dpcd: DSC capabilities from DPCD
/// @is_edp: true if its eDP, false for DP
///
/// Read the slice capabilities DPCD register from DSC sink to get
/// the maximum slice count supported. This is used to populate
/// the DSC parameters in the &struct drm_dsc_config by the driver.
/// Driver creates an infoframe using these parameters to populate
/// &struct drm_dsc_pps_infoframe. These are sent to the sink using DSC
/// infoframe using the helper function drm_dsc_pps_infoframe_pack()
///
/// Returns:
/// Maximum slice count supported by DSC sink or 0 its invalid
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dsc_sink_max_slice_count")]
extern "C" fn drm_dp_dsc_sink_max_slice_count(dsc_dpcd: *const u8, is_edp: bool) -> u8 {
fls(drm_dp_dsc_sink_slice_count_mask(dsc_dpcd, is_edp)) as u8
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dsc_sink_max_slice_count_757: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, bool) -> u8>, *mut ()>(Some(drm_dp_dsc_sink_max_slice_count)) };

/// drm_dp_dsc_sink_line_buf_depth() - Get the line buffer depth in bits
/// @dsc_dpcd: DSC capabilities from DPCD
///
/// Read the DSC DPCD register to parse the line buffer depth in bits which is
/// number of bits of precision within the decoder line buffer supported by
/// the DSC sink. This is used to populate the DSC parameters in the
/// &struct drm_dsc_config by the driver.
/// Driver creates an infoframe using these parameters to populate
/// &struct drm_dsc_pps_infoframe. These are sent to the sink using DSC
/// infoframe using the helper function drm_dsc_pps_infoframe_pack()
///
/// Returns:
/// Line buffer depth supported by DSC panel or 0 its invalid
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dsc_sink_line_buf_depth")]
extern "C" fn drm_dp_dsc_sink_line_buf_depth(dsc_dpcd: *const u8) -> u8 {
let line_buf_depth: u8 = unsafe { *dsc_dpcd.wrapping_offset(5i32 as isize) };
match (line_buf_depth as i32) & 15i32 {
0 => {
return 9;
}
1 => {
return 10;
}
2 => {
return 11;
}
3 => {
return 12;
}
4 => {
return 13;
}
5 => {
return 14;
}
6 => {
return 15;
}
7 => {
return 16;
}
8 => {
return 8;
}
_ => {
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dsc_sink_line_buf_depth_758: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> u8>, *mut ()>(Some(drm_dp_dsc_sink_line_buf_depth)) };

/// drm_dp_dsc_sink_supported_input_bpcs() - Get all the input bits per component
/// values supported by the DSC sink.
/// @dsc_dpcd: DSC capabilities from DPCD
/// @dsc_bpc: An array to be filled by this helper with supported
///           input bpcs.
///
/// Read the DSC DPCD from the sink device to parse the supported bits per
/// component values. This is used to populate the DSC parameters
/// in the &struct drm_dsc_config by the driver.
/// Driver creates an infoframe using these parameters to populate
/// &struct drm_dsc_pps_infoframe. These are sent to the sink using DSC
/// infoframe using the helper function drm_dsc_pps_infoframe_pack()
///
/// Returns:
/// Number of input BPC values parsed from the DPCD
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dsc_sink_supported_input_bpcs")]
extern "C" fn drm_dp_dsc_sink_supported_input_bpcs(dsc_dpcd: *const u8, dsc_bpc: *mut u8) -> i32 {
let mut num_bpc: i32 = 0;
let color_depth: u8 = unsafe { *dsc_dpcd.wrapping_offset(10i32 as isize) };
if !drm_dp_sink_supports_dsc(dsc_dpcd) {
return 0;
}
if (color_depth as i32) & 8i32 != 0 {
let __v0: i32 = num_bpc;
num_bpc = __v0.wrapping_add(1);
unsafe {
*dsc_bpc.wrapping_offset(__v0 as isize) = 12;
}
}
if (color_depth as i32) & 4i32 != 0 {
let __v2: i32 = num_bpc;
num_bpc = __v2.wrapping_add(1);
unsafe {
*dsc_bpc.wrapping_offset(__v2 as isize) = 10;
}
}
// A DP DSC Sink device shall support 8 bpc.
let __v4: i32 = num_bpc;
num_bpc = __v4.wrapping_add(1);
unsafe {
*dsc_bpc.wrapping_offset(__v4 as isize) = 8;
}
num_bpc
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dsc_sink_supported_input_bpcs_759: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *mut u8) -> i32>, *mut ()>(Some(drm_dp_dsc_sink_supported_input_bpcs)) };

/// See DP Standard v2.1a 2.8.4 Minimum Slices/Display, Table 2-159 and
/// Appendix L.1 Derivation of Slice Count Requirements.
fn dsc_sink_min_slice_throughput(peak_pixel_rate: i32) -> i32 {
if peak_pixel_rate >= 4800000 {
return 600000;
} else {
if peak_pixel_rate >= 2700000 {
return 400000;
} else {
return 340000;
}
}
}

/// drm_dp_dsc_sink_max_slice_throughput() - Get a DSC sink's maximum pixel throughput per slice
/// @dsc_dpcd: DSC sink's capabilities from DPCD
/// @peak_pixel_rate: Cumulative peak pixel rate in kHz
/// @is_rgb_yuv444: The mode is either RGB or YUV444
///
/// Return the DSC sink device's maximum pixel throughput per slice, based on
/// the device's @dsc_dpcd capabilities, the @peak_pixel_rate of the transferred
/// stream(s) and whether the output format @is_rgb_yuv444 or yuv422/yuv420.
///
/// Note that @peak_pixel_rate is the total pixel rate transferred to the same
/// DSC/display sink. For instance to calculate a tile's slice count of an MST
/// multi-tiled display sink (not considering here the required
/// rounding/alignment of slice count)::
///
///   @peak_pixel_rate = tile_pixel_rate * tile_count
///   total_slice_count = @peak_pixel_rate / drm_dp_dsc_sink_max_slice_throughput(@peak_pixel_rate)
///   tile_slice_count = total_slice_count / tile_count
///
/// Returns:
/// The maximum pixel throughput per slice supported by the DSC sink device
/// in kPixels/sec.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dsc_sink_max_slice_throughput")]
extern "C" fn drm_dp_dsc_sink_max_slice_throughput(dsc_dpcd: *const u8, peak_pixel_rate: i32, is_rgb_yuv444: bool) -> i32 {
let mut throughput: i32 = (unsafe { *dsc_dpcd.wrapping_offset(11i32 as isize) }) as i32;
let mut delta: i32 = 0;
let mut base: i32 = 0;
if is_rgb_yuv444 {
throughput = (throughput & 15i32) >> 0i32;
delta = (((unsafe { *dsc_dpcd.wrapping_offset(2i32 as isize) }) as i32) & 248i32) >> 3i32; // in units of 2 MPixels/sec
delta = delta.wrapping_mul(2000);
} else {
throughput = (throughput & 240i32) >> 4i32;
}
match throughput {
0 => {
return dsc_sink_min_slice_throughput(peak_pixel_rate);
}
1 => {
base = 340000;
}
2..=14 => {
base = 400000i32.wrapping_add(50000i32.wrapping_mul(throughput.wrapping_sub(2)));
}
15 => {
base = 170000;
}
_ => {
}
}
base.wrapping_add(delta)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dsc_sink_max_slice_throughput_760: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, i32, bool) -> i32>, *mut ()>(Some(drm_dp_dsc_sink_max_slice_throughput)) };

fn dsc_branch_dpcd_cap(dpcd: *const u8, reg: i32) -> u8 {
unsafe { *dpcd.wrapping_offset(reg.wrapping_sub(160) as isize) }
}

/// drm_dp_dsc_branch_max_overall_throughput() - Branch device's max overall DSC pixel throughput
/// @dsc_branch_dpcd: DSC branch capabilities from DPCD
/// @is_rgb_yuv444: The mode is either RGB or YUV444
///
/// Return the branch device's maximum overall DSC pixel throughput, based on
/// the device's DPCD DSC branch capabilities, and whether the output
/// format @is_rgb_yuv444 or yuv422/yuv420.
///
/// Returns:
/// - 0:   The maximum overall throughput capability is not indicated by
///        the device separately and it must be determined from the per-slice
///        max throughput (see @drm_dp_dsc_branch_slice_max_throughput())
///        and the maximum slice count supported by the device.
/// - > 0: The maximum overall DSC pixel throughput supported by the branch
///        device in kPixels/sec.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dsc_branch_max_overall_throughput")]
extern "C" fn drm_dp_dsc_branch_max_overall_throughput(dsc_branch_dpcd: *const u8, is_rgb_yuv444: bool) -> i32 {
let throughput: i32 = if is_rgb_yuv444 {
dsc_branch_dpcd_cap(dsc_branch_dpcd, 160i32) as i32
} else {
dsc_branch_dpcd_cap(dsc_branch_dpcd, 161i32) as i32
};
match throughput {
0 => {
return 0;
}
1 => {
return 680000;
}
_ => {
return 600000i32.wrapping_add(50000i32.wrapping_mul(throughput));
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dsc_branch_max_overall_throughput_761: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, bool) -> i32>, *mut ()>(Some(drm_dp_dsc_branch_max_overall_throughput)) };

/// drm_dp_dsc_branch_max_line_width() - Branch device's max DSC line width
/// @dsc_branch_dpcd: DSC branch capabilities from DPCD
///
/// Return the branch device's maximum overall DSC line width, based on
/// the device's @dsc_branch_dpcd capabilities.
///
/// Returns:
/// - 0:        The maximum line width is not indicated by the device
///             separately and it must be determined from the maximum
///             slice count and slice-width supported by the device.
/// - %-EINVAL: The device indicates an invalid maximum line width
///             (< 5120 pixels).
/// - >= 5120:  The maximum line width in pixels.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_dsc_branch_max_line_width")]
extern "C" fn drm_dp_dsc_branch_max_line_width(dsc_branch_dpcd: *const u8) -> i32 {
let line_width: i32 = dsc_branch_dpcd_cap(dsc_branch_dpcd, 162i32) as i32;
match line_width {
0 => {
return 0;
}
1..=15 => {
return 22i32.wrapping_neg();
}
_ => {
return line_width.wrapping_mul(320);
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_dsc_branch_max_line_width_762: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> i32>, *mut ()>(Some(drm_dp_dsc_branch_max_line_width)) };

fn drm_dp_read_lttpr_regs(aux: *mut drm_dp_aux, dpcd: *const u8, address: i32, buf: *mut u8, buf_size: i32) -> i32 {
// At least the DELL P2715Q monitor with a DPCD_REV < 0x14 returns
// corrupted values when reading from the 0xF0000- range with a block
// size bigger than 1.
let block_size: i32 = if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) < 20 { 1 } else { buf_size };
let mut offset: i32 = 0;
let mut ret: i32 = 0;
while offset < buf_size {
ret = drm_dp_dpcd_read_data(aux, address.wrapping_add(offset) as u32, buf.wrapping_offset(offset as isize) as *mut (), block_size as u64);
if ret < 0 {
return ret;
}
offset = offset.wrapping_add(block_size);
}
0
}

/// drm_dp_read_lttpr_common_caps - read the LTTPR common capabilities
/// @aux: DisplayPort AUX channel
/// @dpcd: DisplayPort configuration data
/// @caps: buffer to return the capability info in
///
/// Read capabilities common to all LTTPRs.
///
/// Returns 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_read_lttpr_common_caps")]
extern "C" fn drm_dp_read_lttpr_common_caps(aux: *mut drm_dp_aux, dpcd: *const u8, caps: *mut u8) -> i32 {
drm_dp_read_lttpr_regs(aux, dpcd, 983040i32, caps, 8i32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_read_lttpr_common_caps_763: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *const u8, *mut u8) -> i32>, *mut ()>(Some(drm_dp_read_lttpr_common_caps)) };

/// drm_dp_read_lttpr_phy_caps - read the capabilities for a given LTTPR PHY
/// @aux: DisplayPort AUX channel
/// @dpcd: DisplayPort configuration data
/// @dp_phy: LTTPR PHY to read the capabilities for
/// @caps: buffer to return the capability info in
///
/// Read the capabilities for the given LTTPR PHY.
///
/// Returns 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_read_lttpr_phy_caps")]
extern "C" fn drm_dp_read_lttpr_phy_caps(aux: *mut drm_dp_aux, dpcd: *const u8, dp_phy: u32, caps: *mut u8) -> i32 {
drm_dp_read_lttpr_regs(aux, dpcd, DP_TRAINING_AUX_RD_INTERVAL_PHY_REPEATER!(dp_phy) as i32, caps, 3i32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_read_lttpr_phy_caps_764: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *const u8, u32, *mut u8) -> i32>, *mut ()>(Some(drm_dp_read_lttpr_phy_caps)) };

fn dp_lttpr_common_cap(caps: *const u8, r: i32) -> u8 {
unsafe { *caps.wrapping_offset(r.wrapping_sub(983040) as isize) }
}

/// drm_dp_lttpr_count - get the number of detected LTTPRs
/// @caps: LTTPR common capabilities
///
/// Get the number of detected LTTPRs from the LTTPR common capabilities info.
///
/// Returns:
///   -ERANGE if more than supported number (8) of LTTPRs are detected
///   -EINVAL if the DP_PHY_REPEATER_CNT register contains an invalid value
///   otherwise the number of detected LTTPRs
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_lttpr_count")]
extern "C" fn drm_dp_lttpr_count(caps: *const u8) -> i32 {
let count: u8 = dp_lttpr_common_cap(caps, 983042i32);
match hweight8!(count) {
0 => {
return 0;
}
1 => {
return 8i32.wrapping_sub(ilog2!(count));
}
8 => {
return 34i32.wrapping_neg();
}
_ => {
return 22i32.wrapping_neg();
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_lttpr_count_765: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> i32>, *mut ()>(Some(drm_dp_lttpr_count)) };

/// drm_dp_lttpr_max_link_rate - get the maximum link rate supported by all LTTPRs
/// @caps: LTTPR common capabilities
///
/// Returns the maximum link rate supported by all detected LTTPRs.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_lttpr_max_link_rate")]
extern "C" fn drm_dp_lttpr_max_link_rate(caps: *const u8) -> i32 {
let rate: u8 = dp_lttpr_common_cap(caps, 983041i32);
drm_dp_bw_code_to_link_rate(rate)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_lttpr_max_link_rate_766: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> i32>, *mut ()>(Some(drm_dp_lttpr_max_link_rate)) };

/// drm_dp_lttpr_set_transparent_mode() - set the LTTPR in transparent mode
/// @aux: DisplayPort AUX channel
/// @enable: Enable or disable transparent mode
///
/// Returns: 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_lttpr_set_transparent_mode")]
extern "C" fn drm_dp_lttpr_set_transparent_mode(aux: *mut drm_dp_aux, enable: bool) -> i32 {
let val: u8 = (if enable { 85i32 } else { 170i32 }) as u8;
let ret: i32 = drm_dp_dpcd_writeb(aux, 983043u32, val) as i32;
if ret < 0 {
return ret;
}
if ret == 1 { 0 } else { 5i32.wrapping_neg() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_lttpr_set_transparent_mode_767: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, bool) -> i32>, *mut ()>(Some(drm_dp_lttpr_set_transparent_mode)) };

/// drm_dp_lttpr_init() - init LTTPR transparency mode according to DP standard
/// @aux: DisplayPort AUX channel
/// @lttpr_count: Number of LTTPRs. Between 0 and 8, according to DP standard.
///               Negative error code for any non-valid number.
///               See drm_dp_lttpr_count().
///
/// Returns: 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_lttpr_init")]
extern "C" fn drm_dp_lttpr_init(aux: *mut drm_dp_aux, lttpr_count: i32) -> i32 {
let mut ret: i32 = 0;
if lttpr_count == 0 {
return 0;
}
// See DP Standard v2.0 3.6.6.1 about the explicit disabling of
// non-transparent mode and the disable->enable non-transparent mode
// sequence.
ret = drm_dp_lttpr_set_transparent_mode(aux, true);
if ret != 0 {
return ret;
}
if lttpr_count < 0 {
return 19i32.wrapping_neg();
}
if drm_dp_lttpr_set_transparent_mode(aux, false) != 0 {
// Roll-back to transparent mode if setting non-transparent
// mode has failed
drm_dp_lttpr_set_transparent_mode(aux, true);
return 22i32.wrapping_neg();
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_lttpr_init_768: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, i32) -> i32>, *mut ()>(Some(drm_dp_lttpr_init)) };

/// drm_dp_lttpr_max_lane_count - get the maximum lane count supported by all LTTPRs
/// @caps: LTTPR common capabilities
///
/// Returns the maximum lane count supported by all detected LTTPRs.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_lttpr_max_lane_count")]
extern "C" fn drm_dp_lttpr_max_lane_count(caps: *const u8) -> i32 {
let max_lanes: u8 = dp_lttpr_common_cap(caps, 983044i32);
(max_lanes as i32) & 31
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_lttpr_max_lane_count_769: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> i32>, *mut ()>(Some(drm_dp_lttpr_max_lane_count)) };

/// drm_dp_lttpr_voltage_swing_level_3_supported - check for LTTPR vswing3 support
/// @caps: LTTPR PHY capabilities
///
/// Returns true if the @caps for an LTTPR TX PHY indicate support for
/// voltage swing level 3.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_lttpr_voltage_swing_level_3_supported")]
extern "C" fn drm_dp_lttpr_voltage_swing_level_3_supported(caps: *const u8) -> bool {
let txcap: u8 = dp_lttpr_phy_cap(caps, 983073i32);
((txcap as i32) as u64) & BIT!(0i32) != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_lttpr_voltage_swing_level_3_supported_770: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> bool>, *mut ()>(Some(drm_dp_lttpr_voltage_swing_level_3_supported)) };

/// drm_dp_lttpr_pre_emphasis_level_3_supported - check for LTTPR preemph3 support
/// @caps: LTTPR PHY capabilities
///
/// Returns true if the @caps for an LTTPR TX PHY indicate support for
/// pre-emphasis level 3.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_lttpr_pre_emphasis_level_3_supported")]
extern "C" fn drm_dp_lttpr_pre_emphasis_level_3_supported(caps: *const u8) -> bool {
let txcap: u8 = dp_lttpr_phy_cap(caps, 983073i32);
((txcap as i32) as u64) & BIT!(1i32) != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_lttpr_pre_emphasis_level_3_supported_771: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> bool>, *mut ()>(Some(drm_dp_lttpr_pre_emphasis_level_3_supported)) };

/// drm_dp_get_phy_test_pattern() - get the requested pattern from the sink.
/// @aux: DisplayPort AUX channel
/// @data: DP phy compliance test parameters.
///
/// Returns 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_get_phy_test_pattern")]
extern "C" fn drm_dp_get_phy_test_pattern(aux: *mut drm_dp_aux, data: *mut drm_dp_phy_test_params) -> i32 {
let mut err: i32 = 0;
let mut rate: u8 = 0;
let mut lanes: u8 = 0;
err = drm_dp_dpcd_read_byte(aux, 537u32, core::ptr::addr_of_mut!(rate));
if err < 0 {
return err;
}
unsafe {
(*data).link_rate = drm_dp_bw_code_to_link_rate(rate);
}
err = drm_dp_dpcd_read_byte(aux, 544u32, core::ptr::addr_of_mut!(lanes));
if err < 0 {
return err;
}
unsafe {
(*data).num_lanes = ((lanes as i32) & 31) as u8;
}
if (lanes as i32) & 128i32 != 0 {
unsafe {
(*data).enhanced_frame_cap = true;
}
}
err = drm_dp_dpcd_read_byte(aux, 584u32, unsafe { core::ptr::addr_of_mut!((*data).phy_pattern) });
if err < 0 {
return err;
}
match (unsafe { (*data).phy_pattern }) as i32 {
4 => {
err = drm_dp_dpcd_read_data(aux, 592u32, (unsafe { core::ptr::addr_of_mut!((*data).custom80) }) as *mut (), core::mem::size_of::<[u8; 10]>() as u64);
if err < 0 {
return err;
}
}
5 => {
err = drm_dp_dpcd_read_data(aux, 586u32, (unsafe { core::ptr::addr_of_mut!((*data).hbr2_reset) }) as *mut (), core::mem::size_of::<[u8; 2]>() as u64);
if err < 0 {
return err;
}
}
_ => {
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_get_phy_test_pattern_772: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *mut drm_dp_phy_test_params) -> i32>, *mut ()>(Some(drm_dp_get_phy_test_pattern)) };

/// drm_dp_set_phy_test_pattern() - set the pattern to the sink.
/// @aux: DisplayPort AUX channel
/// @data: DP phy compliance test parameters.
/// @dp_rev: DP revision to use for compliance testing
///
/// Returns 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_set_phy_test_pattern")]
extern "C" fn drm_dp_set_phy_test_pattern(aux: *mut drm_dp_aux, data: *mut drm_dp_phy_test_params, dp_rev: u8) -> i32 {
let mut err: i32 = 0;
let mut i: i32 = 0;
let mut test_pattern: u8 = unsafe { (*data).phy_pattern };
if (dp_rev as i32) < 18 {
test_pattern = ((test_pattern as i32) << 2i32 & 12i32) as u8;
err = drm_dp_dpcd_write_byte(aux, 258u32, test_pattern);
if err < 0 {
return err;
}
} else {
i = 0;
while i < ((unsafe { (*data).num_lanes }) as i32) {
err = drm_dp_dpcd_write_byte(aux, 267i32.wrapping_add(i) as u32, test_pattern);
if err < 0 {
return err;
}
i = i.wrapping_add(1);
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_set_phy_test_pattern_773: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *mut drm_dp_phy_test_params, u8) -> i32>, *mut ()>(Some(drm_dp_set_phy_test_pattern)) };

fn dp_pixelformat_get_name(pixelformat: u32) -> *const u8 {
if (pixelformat as u32) < 0 || (pixelformat as u32) > 6 {
return c"Invalid".as_ptr() as *const u8;
}
match pixelformat as u32 {
0 => {
return c"RGB".as_ptr() as *const u8;
}
1 => {
return c"YUV444".as_ptr() as *const u8;
}
2 => {
return c"YUV422".as_ptr() as *const u8;
}
3 => {
return c"YUV420".as_ptr() as *const u8;
}
4 => {
return c"Y_ONLY".as_ptr() as *const u8;
}
5 => {
return c"RAW".as_ptr() as *const u8;
}
_ => {
return c"Reserved".as_ptr() as *const u8;
}
}
unsafe { core::mem::zeroed() }
}

fn dp_colorimetry_get_name(pixelformat: u32, colorimetry: u32) -> *const u8 {
if (pixelformat as u32) < 0 || (pixelformat as u32) > 6 {
return c"Invalid".as_ptr() as *const u8;
}
let mut __t0: i64 = match colorimetry as u32 { 0 => 0, 1 => 1, 2 => 2, 3 => 3, 4 => 4, 5 => 5, 6 => 6, 7 => 7, _ => 8 };
loop {
match __t0 {
0 => {
match pixelformat as u32 {
0 => {
return c"sRGB".as_ptr() as *const u8;
}
1..=3 => {
return c"BT.601".as_ptr() as *const u8;
}
4 => {
return c"DICOM PS3.14".as_ptr() as *const u8;
}
5 => {
return c"Custom Color Profile".as_ptr() as *const u8;
}
_ => {
return c"Reserved".as_ptr() as *const u8;
}
}
__t0 = 1;
continue;
}
1 => {
match pixelformat as u32 {
0 => {
return c"Wide Fixed".as_ptr() as *const u8;
}
1..=3 => {
return c"BT.709".as_ptr() as *const u8;
}
_ => {
return c"Reserved".as_ptr() as *const u8;
}
}
// and DP_COLORIMETRY_BT709_YCC
__t0 = 2;
continue;
}
2 => {
match pixelformat as u32 {
0 => {
return c"Wide Float".as_ptr() as *const u8;
}
1..=3 => {
return c"xvYCC 601".as_ptr() as *const u8;
}
_ => {
return c"Reserved".as_ptr() as *const u8;
}
}
// and DP_COLORIMETRY_XVYCC_601
__t0 = 3;
continue;
}
3 => {
match pixelformat as u32 {
0 => {
return c"OpRGB".as_ptr() as *const u8;
}
1..=3 => {
return c"xvYCC 709".as_ptr() as *const u8;
}
_ => {
return c"Reserved".as_ptr() as *const u8;
}
}
// and DP_COLORIMETRY_XVYCC_709
__t0 = 4;
continue;
}
4 => {
match pixelformat as u32 {
0 => {
return c"DCI-P3".as_ptr() as *const u8;
}
1..=3 => {
return c"sYCC 601".as_ptr() as *const u8;
}
_ => {
return c"Reserved".as_ptr() as *const u8;
}
}
// and DP_COLORIMETRY_SYCC_601
__t0 = 5;
continue;
}
5 => {
match pixelformat as u32 {
0 => {
return c"Custom Profile".as_ptr() as *const u8;
}
1..=3 => {
return c"OpYCC 601".as_ptr() as *const u8;
}
_ => {
return c"Reserved".as_ptr() as *const u8;
}
}
// and DP_COLORIMETRY_OPYCC_601
__t0 = 6;
continue;
}
6 => {
match pixelformat as u32 {
0 => {
return c"BT.2020 RGB".as_ptr() as *const u8;
}
1..=3 => {
return c"BT.2020 CYCC".as_ptr() as *const u8;
}
_ => {
return c"Reserved".as_ptr() as *const u8;
}
}
// and DP_COLORIMETRY_BT2020_CYCC
__t0 = 7;
continue;
}
7 => {
match pixelformat as u32 {
1..=3 => {
return c"BT.2020 YCC".as_ptr() as *const u8;
}
_ => {
return c"Reserved".as_ptr() as *const u8;
}
}
__t0 = 8;
continue;
}
8 => {
return c"Invalid".as_ptr() as *const u8;
}
_ => {
break;
}
}
}
unsafe { core::mem::zeroed() }
}

fn dp_dynamic_range_get_name(dynamic_range: u32) -> *const u8 {
match dynamic_range as u32 {
0 => {
return c"VESA range".as_ptr() as *const u8;
}
1 => {
return c"CTA range".as_ptr() as *const u8;
}
_ => {
return c"Invalid".as_ptr() as *const u8;
}
}
unsafe { core::mem::zeroed() }
}

fn dp_content_type_get_name(content_type: u32) -> *const u8 {
match content_type as u32 {
0 => {
return c"Not defined".as_ptr() as *const u8;
}
1 => {
return c"Graphics".as_ptr() as *const u8;
}
2 => {
return c"Photo".as_ptr() as *const u8;
}
3 => {
return c"Video".as_ptr() as *const u8;
}
4 => {
return c"Game".as_ptr() as *const u8;
}
_ => {
return c"Reserved".as_ptr() as *const u8;
}
}
unsafe { core::mem::zeroed() }
}

fn dp_sdp_type_get_name(r#type: u8) -> *const u8 {
match r#type as i32 {
1 => {
return c"Audio_TimeStamp".as_ptr() as *const u8;
}
2 => {
return c"Audio_Stream".as_ptr() as *const u8;
}
4 => {
return c"Extension".as_ptr() as *const u8;
}
5 => {
return c"Audio_CopyManagement".as_ptr() as *const u8;
}
6 => {
return c"ISRC".as_ptr() as *const u8;
}
7 => {
return c"VSC".as_ptr() as *const u8;
}
16 => {
return c"PPS".as_ptr() as *const u8;
}
32 => {
return c"VSC_EXT_VESA".as_ptr() as *const u8;
}
33 => {
return c"VSC_EXT_CEA".as_ptr() as *const u8;
}
34 => {
return c"Adaptive-Sync".as_ptr() as *const u8;
}
_ => {
return c"Unknown".as_ptr() as *const u8;
}
}
unsafe { core::mem::zeroed() }
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_vsc_sdp_log")]
extern "C" fn drm_dp_vsc_sdp_log(p: *mut drm_printer, vsc: *const drm_dp_vsc_sdp) {
unsafe { drm_printf(p, c"DP SDP: %s, revision %u, length %u\n".as_ptr() as *const u8, dp_sdp_type_get_name((*vsc).sdp_type), (*vsc).revision as i32, (*vsc).length as i32) };
unsafe { drm_printf(p, c"%.*spixelformat: %s\n".as_ptr() as *const u8, 1i32, c"\t\t\t\t\tX".as_ptr() as *mut u8, dp_pixelformat_get_name((*vsc).pixelformat)) };
unsafe { drm_printf(p, c"%.*scolorimetry: %s\n".as_ptr() as *const u8, 1i32, c"\t\t\t\t\tX".as_ptr() as *mut u8, dp_colorimetry_get_name((*vsc).pixelformat, (*vsc).colorimetry)) };
unsafe { drm_printf(p, c"%.*sbpc: %u\n".as_ptr() as *const u8, 1i32, c"\t\t\t\t\tX".as_ptr() as *mut u8, (*vsc).bpc) };
unsafe { drm_printf(p, c"%.*sdynamic range: %s\n".as_ptr() as *const u8, 1i32, c"\t\t\t\t\tX".as_ptr() as *mut u8, dp_dynamic_range_get_name((*vsc).dynamic_range)) };
unsafe { drm_printf(p, c"%.*scontent type: %s\n".as_ptr() as *const u8, 1i32, c"\t\t\t\t\tX".as_ptr() as *mut u8, dp_content_type_get_name((*vsc).content_type)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_vsc_sdp_log_774: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_printer, *const drm_dp_vsc_sdp)>, *mut ()>(Some(drm_dp_vsc_sdp_log)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_as_sdp_log")]
extern "C" fn drm_dp_as_sdp_log(p: *mut drm_printer, as_sdp: *const drm_dp_as_sdp) {
unsafe { drm_printf(p, c"DP SDP: %s, revision %u, length %u\n".as_ptr() as *const u8, dp_sdp_type_get_name((*as_sdp).sdp_type), (*as_sdp).revision as i32, (*as_sdp).length as i32) };
unsafe { drm_printf(p, c"%.*svtotal: %d\n".as_ptr() as *const u8, 1i32, c"\t\t\t\t\tX".as_ptr() as *mut u8, (*as_sdp).vtotal) };
unsafe { drm_printf(p, c"%.*starget rr: %d\n".as_ptr() as *const u8, 1i32, c"\t\t\t\t\tX".as_ptr() as *mut u8, (*as_sdp).target_rr) };
unsafe { drm_printf(p, c"%.*sduration increase ms: %d\n".as_ptr() as *const u8, 1i32, c"\t\t\t\t\tX".as_ptr() as *mut u8, (*as_sdp).duration_incr_ms) };
unsafe { drm_printf(p, c"%.*sduration decrease ms: %d\n".as_ptr() as *const u8, 1i32, c"\t\t\t\t\tX".as_ptr() as *mut u8, (*as_sdp).duration_decr_ms) };
unsafe { drm_printf(p, c"%.*soperation mode: %d\n".as_ptr() as *const u8, 1i32, c"\t\t\t\t\tX".as_ptr() as *mut u8, (*as_sdp).mode as u32) };
unsafe { drm_printf(p, c"%.*starget rr divider: %s\n".as_ptr() as *const u8, 1i32, c"\t\t\t\t\tX".as_ptr() as *mut u8, if (*as_sdp).target_rr_divider { c"1.001".as_ptr() as *mut u8 } else { c"1.000".as_ptr() as *mut u8 }) };
unsafe { drm_printf(p, c"%.*scoasting vtotal: %d\n".as_ptr() as *const u8, 1i32, c"\t\t\t\t\tX".as_ptr() as *mut u8, (*as_sdp).coasting_vtotal) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_as_sdp_log_775: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_printer, *const drm_dp_as_sdp)>, *mut ()>(Some(drm_dp_as_sdp_log)) };

/// drm_dp_as_sdp_supported() - check if adaptive sync sdp is supported
/// @aux: DisplayPort AUX channel
/// @dpcd: DisplayPort configuration data
///
/// Returns true if adaptive sync sdp is supported, else returns false
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_as_sdp_supported")]
extern "C" fn drm_dp_as_sdp_supported(aux: *mut drm_dp_aux, dpcd: *const u8) -> bool {
let mut rx_feature: u8 = 0;
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) < 19 {
return false;
}
if drm_dp_dpcd_read_byte(aux, 8724u32, core::ptr::addr_of_mut!(rx_feature)) < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 8u32, c"Failed to read DP_DPRX_FEATURE_ENUMERATION_LIST_CONT_1\n".as_ptr() as *const u8) };
return false;
}
((rx_feature as i32) as u64) & BIT!(0i32) != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_as_sdp_supported_776: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *const u8) -> bool>, *mut ()>(Some(drm_dp_as_sdp_supported)) };

/// drm_dp_vsc_sdp_supported() - check if vsc sdp is supported
/// @aux: DisplayPort AUX channel
/// @dpcd: DisplayPort configuration data
///
/// Returns true if vsc sdp is supported, else returns false
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_vsc_sdp_supported")]
extern "C" fn drm_dp_vsc_sdp_supported(aux: *mut drm_dp_aux, dpcd: *const u8) -> bool {
let mut rx_feature: u8 = 0;
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) < 19 {
return false;
}
if drm_dp_dpcd_read_byte(aux, 8720u32, core::ptr::addr_of_mut!(rx_feature)) < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 8u32, c"failed to read DP_DPRX_FEATURE_ENUMERATION_LIST\n".as_ptr() as *const u8) };
return false;
}
(rx_feature as i32) & 8i32 != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_vsc_sdp_supported_777: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *const u8) -> bool>, *mut ()>(Some(drm_dp_vsc_sdp_supported)) };

/// drm_dp_vsc_sdp_pack() - pack a given vsc sdp into generic dp_sdp
/// @vsc: vsc sdp initialized according to its purpose as defined in
///       table 2-118 - table 2-120 in DP 1.4a specification
/// @sdp: valid handle to the generic dp_sdp which will be packed
///
/// Returns length of sdp on success and error code on failure
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_vsc_sdp_pack")]
extern "C" fn drm_dp_vsc_sdp_pack(vsc: *const drm_dp_vsc_sdp, sdp: *mut dp_sdp) -> i64 {
let mut bug: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug) as *mut *mut bug_entry;
let mut __v5: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v5) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut length: u64 = unsafe { core::mem::zeroed() };
unsafe {
length = core::mem::size_of::<dp_sdp>() as u64;
memset(sdp as *mut (), 0i32, core::mem::size_of::<dp_sdp>() as u64);
// Prepare VSC Header for SU as per DP 1.4a spec, Table 2-119
// VSC SDP Header Bytes
(*sdp).sdp_header.HB0 = 0;
// Secondary-Data Packet ID = 0
(*sdp).sdp_header.HB1 = (*vsc).sdp_type;
// Secondary-data Packet Type
(*sdp).sdp_header.HB2 = (*vsc).revision;
// Revision Number
(*sdp).sdp_header.HB3 = (*vsc).length;
// Number of Valid Data Bytes
if ((*vsc).revision as i32) == 6 {
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = 1;
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) = 1;
}
// Revision 0x5 and revision 0x7 supports Pixel Encoding/Colorimetry
// Format as per DP 1.4a spec and DP 2.0 respectively.
if ((*vsc).revision as i32) == 5 || ((*vsc).revision as i32) == 7 {
// VSC SDP Payload for DB16 through DB18
//
// Pixel Encoding and Colorimetry Formats
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(16 as isize) = ((((*vsc).pixelformat as u32) & 15) << 4i32) as u8;
// DB16[7:4]
let __v11: *mut u8 = ((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(16 as isize);
*__v11 = (((*__v11 as i32) as u32) | ((*vsc).colorimetry as u32) & 15) as u8;
// DB16[3:0]
let __t1: i32 = (*vsc).bpc;
if __t1 != 6 {
if __t1 == 8 {
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(17 as isize) = 1;
// DB17[3:0]
} else {
if __t1 == 10 {
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(17 as isize) = 2;
} else {
if __t1 == 12 {
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(17 as isize) = 3;
} else {
if __t1 == 16 {
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(17 as isize) = 4;
} else {
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"Missing case %d\n".as_ptr() as *const u8, (*vsc).bpc);
*__slate_slot_5 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_drm_dp_vsc_sdp_pack_13, sym __slate_asm_string_drm_dp_vsc_sdp_pack_14, const 3659 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_3 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
(*__slate_slot_5).unwrap()(*__slate_slot_3, (*vsc).bpc);
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
return 22i32.wrapping_neg() as i64;
}
}
}
}
}
// Dynamic Range and Component Bit Depth
if ((*vsc).dynamic_range as u32) == 1 {
let __v2: *mut u8 = ((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(17 as isize);
*__v2 = ((*__v2 as i32) | 128) as u8;
}
// DB17[7]
// Content Type
*((*sdp).db.as_mut_ptr() as *mut u8).wrapping_offset(18 as isize) = (((*vsc).content_type as u32) & 7) as u8;
}
return length as i64;
}
// 6bpc: 0x0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_778: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_vsc_sdp_pack_779: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_dp_vsc_sdp, *mut dp_sdp) -> i64>, *mut ()>(Some(drm_dp_vsc_sdp_pack)) };

/// drm_dp_get_pcon_max_frl_bw() - maximum frl supported by PCON
/// @dpcd: DisplayPort configuration data
/// @port_cap: port capabilities
///
/// Returns maximum frl bandwidth supported by PCON in GBPS,
/// returns 0 if not supported.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_get_pcon_max_frl_bw")]
extern "C" fn drm_dp_get_pcon_max_frl_bw(dpcd: *const u8, port_cap: *const u8) -> i32 {
let mut bw: i32 = 0;
let mut buf: u8 = 0;
if !drm_dp_is_branch(dpcd) {
return 0;
}
if ((unsafe { *dpcd.wrapping_offset(0 as isize) }) as i32) < 17 {
return 0;
}
if ((unsafe { *dpcd.wrapping_offset(5 as isize) }) as i32) & 16i32 == 0 {
return 0;
}
if ((unsafe { *port_cap.wrapping_offset(0 as isize) }) as i32) & 7i32 != 3 {
return 0;
}
buf = unsafe { *port_cap.wrapping_offset(2 as isize) };
bw = (buf as i32) & 28i32;
match bw {
4 => {
return 9;
}
8 => {
return 18;
}
12 => {
return 24;
}
16 => {
return 32;
}
20 => {
return 40;
}
24 => {
return 48;
}
_ => {
return 0;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_get_pcon_max_frl_bw_780: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const u8) -> i32>, *mut ()>(Some(drm_dp_get_pcon_max_frl_bw)) };

/// drm_dp_pcon_frl_prepare() - Prepare PCON for FRL.
/// @aux: DisplayPort AUX channel
/// @enable_frl_ready_hpd: Configure DP_PCON_ENABLE_HPD_READY.
///
/// Returns 0 if success, else returns negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_frl_prepare")]
extern "C" fn drm_dp_pcon_frl_prepare(aux: *mut drm_dp_aux, enable_frl_ready_hpd: bool) -> i32 {
let mut buf: u8 = 40u8;
if enable_frl_ready_hpd {
buf = ((buf as i32) | 64i32) as u8;
}
drm_dp_dpcd_write_byte(aux, 12378u32, buf)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_frl_prepare_781: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, bool) -> i32>, *mut ()>(Some(drm_dp_pcon_frl_prepare)) };

/// drm_dp_pcon_is_frl_ready() - Is PCON ready for FRL
/// @aux: DisplayPort AUX channel
///
/// Returns true if success, else returns false.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_is_frl_ready")]
extern "C" fn drm_dp_pcon_is_frl_ready(aux: *mut drm_dp_aux) -> bool {
let mut ret: i32 = 0;
let mut buf: u8 = 0;
ret = drm_dp_dpcd_read_byte(aux, 12347u32, core::ptr::addr_of_mut!(buf));
if ret < 0 {
return false;
}
if (buf as i32) & 2i32 != 0 {
return true;
}
false
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_is_frl_ready_782: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux) -> bool>, *mut ()>(Some(drm_dp_pcon_is_frl_ready)) };

// drm_dp_pcon_frl_configure_1() - Set HDMI LINK Configuration-Step1
// @aux: DisplayPort AUX channel
// @max_frl_gbps: maximum frl bw to be configured between PCON and HDMI sink
// @frl_mode: FRL Training mode, it can be either Concurrent or Sequential.
// In Concurrent Mode, the FRL link bring up can be done along with
// DP Link training. In Sequential mode, the FRL link bring up is done prior to
// the DP Link training.
//
// Returns 0 if success, else returns negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_frl_configure_1")]
extern "C" fn drm_dp_pcon_frl_configure_1(aux: *mut drm_dp_aux, max_frl_gbps: i32, frl_mode: u8) -> i32 {
let mut ret: i32 = 0;
let mut buf: u8 = 0;
ret = drm_dp_dpcd_read_byte(aux, 12378u32, core::ptr::addr_of_mut!(buf));
if ret < 0 {
return ret;
}
if (frl_mode as i32) == 16i32 {
buf = ((buf as i32) | 16i32) as u8;
} else {
let __v3: u8 = ((buf as i32) & -17i32) as u8;
buf = __v3;
}
match max_frl_gbps {
9 => {
buf = ((buf as i32) | 1) as u8;
}
18 => {
buf = ((buf as i32) | 2) as u8;
}
24 => {
buf = ((buf as i32) | 3) as u8;
}
32 => {
buf = ((buf as i32) | 4) as u8;
}
40 => {
buf = ((buf as i32) | 5) as u8;
}
48 => {
buf = ((buf as i32) | 6) as u8;
}
0 => {
buf = ((buf as i32) | 0) as u8;
}
_ => {
return 22i32.wrapping_neg();
}
}
drm_dp_dpcd_write_byte(aux, 12378u32, buf)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_frl_configure_1_783: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, i32, u8) -> i32>, *mut ()>(Some(drm_dp_pcon_frl_configure_1)) };

/// drm_dp_pcon_frl_configure_2() - Set HDMI Link configuration Step-2
/// @aux: DisplayPort AUX channel
/// @max_frl_mask : Max FRL BW to be tried by the PCON with HDMI Sink
/// @frl_type : FRL training type, can be Extended, or Normal.
/// In Normal FRL training, the PCON tries each frl bw from the max_frl_mask
/// starting from min, and stops when link training is successful. In Extended
/// FRL training, all frl bw selected in the mask are trained by the PCON.
///
/// Returns 0 if success, else returns negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_frl_configure_2")]
extern "C" fn drm_dp_pcon_frl_configure_2(aux: *mut drm_dp_aux, max_frl_mask: i32, frl_type: u8) -> i32 {
let ret: i32 = 0;
let mut buf: u8 = max_frl_mask as u8;
if (frl_type as i32) == 64i32 {
buf = ((buf as i32) | 64i32) as u8;
} else {
let __v3: u8 = ((buf as i32) & -65i32) as u8;
buf = __v3;
}
drm_dp_dpcd_write_byte(aux, 12379u32, buf)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_frl_configure_2_784: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, i32, u8) -> i32>, *mut ()>(Some(drm_dp_pcon_frl_configure_2)) };

/// drm_dp_pcon_reset_frl_config() - Re-Set HDMI Link configuration.
/// @aux: DisplayPort AUX channel
///
/// Returns 0 if success, else returns negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_reset_frl_config")]
extern "C" fn drm_dp_pcon_reset_frl_config(aux: *mut drm_dp_aux) -> i32 {
drm_dp_dpcd_write_byte(aux, 12378u32, 0u8)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_reset_frl_config_785: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux) -> i32>, *mut ()>(Some(drm_dp_pcon_reset_frl_config)) };

/// drm_dp_pcon_frl_enable() - Enable HDMI link through FRL
/// @aux: DisplayPort AUX channel
///
/// Returns 0 if success, else returns negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_frl_enable")]
extern "C" fn drm_dp_pcon_frl_enable(aux: *mut drm_dp_aux) -> i32 {
let mut ret: i32 = 0;
let mut buf: u8 = 0;
ret = drm_dp_dpcd_read_byte(aux, 12378u32, core::ptr::addr_of_mut!(buf));
if ret < 0 {
return ret;
}
if (buf as i32) & 8i32 == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: PCON in Autonomous mode, can't enable FRL\n".as_ptr() as *const u8, (*aux).name) };
return 22i32.wrapping_neg();
}
buf = ((buf as i32) | 128i32) as u8;
drm_dp_dpcd_write_byte(aux, 12378u32, buf)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_frl_enable_786: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux) -> i32>, *mut ()>(Some(drm_dp_pcon_frl_enable)) };

/// drm_dp_pcon_hdmi_link_active() - check if the PCON HDMI LINK status is active.
/// @aux: DisplayPort AUX channel
///
/// Returns true if link is active else returns false.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_hdmi_link_active")]
extern "C" fn drm_dp_pcon_hdmi_link_active(aux: *mut drm_dp_aux) -> bool {
let mut buf: u8 = 0;
let ret: i32 = drm_dp_dpcd_read_byte(aux, 12347u32, core::ptr::addr_of_mut!(buf));
if ret < 0 {
return false;
}
(buf as i32) & 1i32 != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_hdmi_link_active_787: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux) -> bool>, *mut ()>(Some(drm_dp_pcon_hdmi_link_active)) };

/// drm_dp_pcon_hdmi_link_mode() - get the PCON HDMI LINK MODE
/// @aux: DisplayPort AUX channel
/// @frl_trained_mask: pointer to store bitmask of the trained bw configuration.
/// Valid only if the MODE returned is FRL. For Normal Link training mode
/// only 1 of the bits will be set, but in case of Extended mode, more than
/// one bits can be set.
///
/// Returns the link mode : TMDS or FRL on success, else returns negative error
/// code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_hdmi_link_mode")]
extern "C" fn drm_dp_pcon_hdmi_link_mode(aux: *mut drm_dp_aux, frl_trained_mask: *mut u8) -> i32 {
let mut buf: u8 = 0;
let mut mode: i32 = 0;
let ret: i32 = drm_dp_dpcd_read_byte(aux, 12342u32, core::ptr::addr_of_mut!(buf));
if ret < 0 {
return ret;
}
mode = (buf as i32) & 1i32;
if !frl_trained_mask.is_null() && 1 == mode {
unsafe {
*frl_trained_mask = (((buf as i32) & 126i32) >> 1i32) as u8;
}
}
mode
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_hdmi_link_mode_788: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *mut u8) -> i32>, *mut ()>(Some(drm_dp_pcon_hdmi_link_mode)) };

// drm_dp_pcon_hdmi_frl_link_error_count() - print the error count per lane
// during link failure between PCON and HDMI sink
// @aux: DisplayPort AUX channel
// @connector: DRM connector
// code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_hdmi_frl_link_error_count")]
extern "C" fn drm_dp_pcon_hdmi_frl_link_error_count(aux: *mut drm_dp_aux, connector: *mut drm_connector) {
let mut buf: u8 = 0;
let mut error_count: u8 = 0;
let mut i: i32 = 0;
let mut num_error: i32 = 0;
let hdmi: *mut drm_hdmi_info = unsafe { core::ptr::addr_of_mut!((*connector).display_info.hdmi) };
while i < ((unsafe { (*hdmi).max_lanes }) as i32) {
if drm_dp_dpcd_read_byte(aux, 12343i32.wrapping_add(i) as u32, core::ptr::addr_of_mut!(buf)) < 0 {
return;
}
error_count = ((buf as i32) & 7i32) as u8;
match error_count as i32 {
4 => {
num_error = 100;
}
2 => {
num_error = 10;
}
1 => {
num_error = 3;
}
_ => {
num_error = 0;
}
}
while false {
}
unsafe { _dev_err(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, c"[drm] *ERROR* %s: More than %d errors since the last read for lane %d".as_ptr() as *const u8, (*aux).name, num_error, i) };
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_hdmi_frl_link_error_count_789: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *mut drm_connector)>, *mut ()>(Some(drm_dp_pcon_hdmi_frl_link_error_count)) };

/// drm_dp_pcon_enc_is_dsc_1_2 - Does PCON Encoder supports DSC 1.2
/// @pcon_dsc_dpcd: DSC capabilities of the PCON DSC Encoder
///
/// Returns true is PCON encoder is DSC 1.2 else returns false.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_enc_is_dsc_1_2")]
extern "C" fn drm_dp_pcon_enc_is_dsc_1_2(pcon_dsc_dpcd: *const u8) -> bool {
let buf: u8 = unsafe { *pcon_dsc_dpcd.wrapping_offset(1i32 as isize) };
let major_v: u8 = (((buf as i32) & 15i32) >> 0i32) as u8;
let minor_v: u8 = (((buf as i32) & 240i32) >> 4i32) as u8;
if (major_v as i32) == 1 && (minor_v as i32) == 2 {
return true;
}
false
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_enc_is_dsc_1_2_790: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> bool>, *mut ()>(Some(drm_dp_pcon_enc_is_dsc_1_2)) };

/// drm_dp_pcon_dsc_max_slices - Get max slices supported by PCON DSC Encoder
/// @pcon_dsc_dpcd: DSC capabilities of the PCON DSC Encoder
///
/// Returns maximum no. of slices supported by the PCON DSC Encoder.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_dsc_max_slices")]
extern "C" fn drm_dp_pcon_dsc_max_slices(pcon_dsc_dpcd: *const u8) -> i32 {
let slice_cap1: u8 = unsafe { *pcon_dsc_dpcd.wrapping_offset(4i32 as isize) };
let slice_cap2: u8 = unsafe { *pcon_dsc_dpcd.wrapping_offset(10i32 as isize) };
if (slice_cap2 as i32) & 4i32 != 0 {
return 24;
}
if (slice_cap2 as i32) & 2i32 != 0 {
return 20;
}
if (slice_cap2 as i32) & 1i32 != 0 {
return 16;
}
if (slice_cap1 as i32) & 128i32 != 0 {
return 12;
}
if (slice_cap1 as i32) & 64i32 != 0 {
return 10;
}
if (slice_cap1 as i32) & 32i32 != 0 {
return 8;
}
if (slice_cap1 as i32) & 16i32 != 0 {
return 6;
}
if (slice_cap1 as i32) & 8i32 != 0 {
return 4;
}
if (slice_cap1 as i32) & 2i32 != 0 {
return 2;
}
if (slice_cap1 as i32) & 1i32 != 0 {
return 1;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_dsc_max_slices_791: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> i32>, *mut ()>(Some(drm_dp_pcon_dsc_max_slices)) };

/// drm_dp_pcon_dsc_max_slice_width() - Get max slice width for Pcon DSC encoder
/// @pcon_dsc_dpcd: DSC capabilities of the PCON DSC Encoder
///
/// Returns maximum width of the slices in pixel width i.e. no. of pixels x 320.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_dsc_max_slice_width")]
extern "C" fn drm_dp_pcon_dsc_max_slice_width(pcon_dsc_dpcd: *const u8) -> i32 {
let buf: u8 = unsafe { *pcon_dsc_dpcd.wrapping_offset(9i32 as isize) };
(buf as i32).wrapping_mul(320)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_dsc_max_slice_width_792: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> i32>, *mut ()>(Some(drm_dp_pcon_dsc_max_slice_width)) };

/// drm_dp_pcon_dsc_bpp_incr() - Get bits per pixel increment for PCON DSC encoder
/// @pcon_dsc_dpcd: DSC capabilities of the PCON DSC Encoder
///
/// Returns the bpp precision supported by the PCON encoder.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_dsc_bpp_incr")]
extern "C" fn drm_dp_pcon_dsc_bpp_incr(pcon_dsc_dpcd: *const u8) -> i32 {
let buf: u8 = unsafe { *pcon_dsc_dpcd.wrapping_offset(12i32 as isize) };
match (buf as i32) & 7i32 {
0 => {
return 16;
}
1 => {
return 8;
}
2 => {
return 4;
}
3 => {
return 2;
}
4 => {
return 1;
}
_ => {
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_dsc_bpp_incr_793: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> i32>, *mut ()>(Some(drm_dp_pcon_dsc_bpp_incr)) };

fn drm_dp_pcon_configure_dsc_enc(aux: *mut drm_dp_aux, pps_buf_config: u8) -> i32 {
let mut buf: u8 = 0;
let ret: i32 = drm_dp_dpcd_read_byte(aux, 12370u32, core::ptr::addr_of_mut!(buf));
if ret < 0 {
return ret;
}
buf = ((buf as i32) | 2i32) as u8;
if (pps_buf_config as i32) <= 2 {
let __v3: u8 = ((buf as i32) & -13i32) as u8;
buf = __v3;
buf = ((buf as i32) | (pps_buf_config as i32) << 2i32) as u8;
}
drm_dp_dpcd_write_byte(aux, 12370u32, buf)
}

/// drm_dp_pcon_pps_default() - Let PCON fill the default pps parameters
/// for DSC1.2 between PCON & HDMI2.1 sink
/// @aux: DisplayPort AUX channel
///
/// Returns 0 on success, else returns negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_pps_default")]
extern "C" fn drm_dp_pcon_pps_default(aux: *mut drm_dp_aux) -> i32 {
drm_dp_pcon_configure_dsc_enc(aux, 0u8)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_pps_default_794: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux) -> i32>, *mut ()>(Some(drm_dp_pcon_pps_default)) };

/// drm_dp_pcon_pps_override_buf() - Configure PPS encoder override buffer for
/// HDMI sink
/// @aux: DisplayPort AUX channel
/// @pps_buf: 128 bytes to be written into PPS buffer for HDMI sink by PCON.
///
/// Returns 0 on success, else returns negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_pps_override_buf")]
extern "C" fn drm_dp_pcon_pps_override_buf(aux: *mut drm_dp_aux, mut pps_buf: *mut u8) -> i32 {
let ret: i32 = drm_dp_dpcd_write_data(aux, 12544u32, core::ptr::addr_of_mut!(pps_buf) as *mut (), 128u64);
if ret < 0 {
return ret;
}
drm_dp_pcon_configure_dsc_enc(aux, 2u8)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_pps_override_buf_795: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *mut u8) -> i32>, *mut ()>(Some(drm_dp_pcon_pps_override_buf)) };

/// drm_dp_pcon_pps_override_param() - Write PPS parameters to DSC encoder
/// override registers
/// @aux: DisplayPort AUX channel
/// @pps_param: 3 Parameters (2 Bytes each) : Slice Width, Slice Height,
/// bits_per_pixel.
///
/// Returns 0 on success, else returns negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_pps_override_param")]
extern "C" fn drm_dp_pcon_pps_override_param(aux: *mut drm_dp_aux, pps_param: *mut u8) -> i32 {
let mut ret: i32 = drm_dp_dpcd_write_data(aux, 12672u32, pps_param.wrapping_offset(0 as isize) as *mut (), 2u64);
if ret < 0 {
return ret;
}
ret = drm_dp_dpcd_write_data(aux, 12674u32, pps_param.wrapping_offset(2 as isize) as *mut (), 2u64);
if ret < 0 {
return ret;
}
ret = drm_dp_dpcd_write_data(aux, 12676u32, pps_param.wrapping_offset(4 as isize) as *mut (), 2u64);
if ret < 0 {
return ret;
}
drm_dp_pcon_configure_dsc_enc(aux, 2u8)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_pps_override_param_796: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *mut u8) -> i32>, *mut ()>(Some(drm_dp_pcon_pps_override_param)) };

/// drm_dp_pcon_convert_rgb_to_ycbcr() - Configure the PCon to convert RGB to Ycbcr
/// @aux: displayPort AUX channel
/// @color_spc: Color-space/s for which conversion is to be enabled, 0 for disable.
///
/// Returns 0 on success, else returns negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_pcon_convert_rgb_to_ycbcr")]
extern "C" fn drm_dp_pcon_convert_rgb_to_ycbcr(aux: *mut drm_dp_aux, color_spc: u8) -> i32 {
let mut ret: i32 = 0;
let mut buf: u8 = 0;
ret = drm_dp_dpcd_read_byte(aux, 12370u32, core::ptr::addr_of_mut!(buf));
if ret < 0 {
return ret;
}
if (color_spc as i32) & 112i32 != 0 {
buf = ((buf as i32) | (color_spc as i32) & 112i32) as u8;
} else {
let __v3: u8 = ((buf as i32) & -113i32) as u8;
buf = __v3;
}
drm_dp_dpcd_write_byte(aux, 12370u32, buf)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_pcon_convert_rgb_to_ycbcr_797: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, u8) -> i32>, *mut ()>(Some(drm_dp_pcon_convert_rgb_to_ycbcr)) };

/// drm_edp_backlight_set_level() - Set the backlight level of an eDP panel via AUX
/// @aux: The DP AUX channel to use
/// @bl: Backlight capability info from drm_edp_backlight_init()
/// @level: The brightness level to set
///
/// Sets the brightness level of an eDP panel's backlight. Note that the panel's backlight must
/// already have been enabled by the driver by calling drm_edp_backlight_enable().
///
/// Returns: %0 on success, negative error code on failure
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_edp_backlight_set_level")]
extern "C" fn drm_edp_backlight_set_level(aux: *mut drm_dp_aux, bl: *const drm_edp_backlight_info, mut level: u32) -> i32 {
let mut ret: i32 = 0;
let mut offset: u32 = 1826;
let mut buf: [u8; 3] = {
let mut __t0: [u8; 3] = unsafe { core::mem::zeroed() };
__t0[0] = 0u8;
__t0
};
let mut len: u64 = 2;
// The panel uses the PWM for controlling brightness levels
if !(((unsafe { (*bl).__slate_bits_0.__get_aux_set() }) as i32) != 0 || ((unsafe { (*bl).__slate_bits_0.__get_luminance_set() }) as i32) != 0) {
return 0;
}
if ((unsafe { (*bl).__slate_bits_0.__get_luminance_set() }) as i32) != 0 {
level = level.wrapping_mul(1000);
level &= 16777215;
buf[0] = (level & 255) as u8;
buf[1] = ((level & 65280) >> 8i32) as u8;
buf[2] = ((level & 16711680) >> 16i32) as u8;
offset = 1844;
len = 3;
} else {
if ((unsafe { (*bl).__slate_bits_0.__get_lsb_reg_used() }) as i32) != 0 {
buf[0] = ((level & 65280) >> 8i32) as u8;
buf[1] = (level & 255) as u8;
} else {
buf[0] = level as u8;
}
}
ret = drm_dp_dpcd_write_data(aux, offset, buf.as_mut_ptr() as *mut (), len);
if ret < 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, c"[drm] *ERROR* %s: Failed to write aux backlight level: %d\n".as_ptr() as *const u8, (*aux).name, ret) };
return ret;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edp_backlight_set_level_798: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *const drm_edp_backlight_info, u32) -> i32>, *mut ()>(Some(drm_edp_backlight_set_level)) };

fn drm_edp_backlight_set_enable(aux: *mut drm_dp_aux, bl: *const drm_edp_backlight_info, enable: bool) -> i32 {
let mut ret: i32 = 0;
let mut buf: u8 = 0;
// This panel uses the EDP_BL_PWR GPIO for enablement
if ((unsafe { (*bl).__slate_bits_0.__get_aux_enable() }) as i32) == 0 {
return 0;
}
ret = drm_dp_dpcd_read_byte(aux, 1824u32, core::ptr::addr_of_mut!(buf));
if ret < 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, c"[drm] *ERROR* %s: Failed to read eDP display control register: %d\n".as_ptr() as *const u8, (*aux).name, ret) };
return ret;
}
if enable {
buf = ((buf as i32) | 1i32) as u8;
} else {
let __v4: u8 = ((buf as i32) & -2i32) as u8;
buf = __v4;
}
ret = drm_dp_dpcd_write_byte(aux, 1824u32, buf);
if ret < 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, c"[drm] *ERROR* %s: Failed to write eDP display control register: %d\n".as_ptr() as *const u8, (*aux).name, ret) };
return ret;
}
0
}

/// drm_edp_backlight_enable() - Enable an eDP panel's backlight using DPCD
/// @aux: The DP AUX channel to use
/// @bl: Backlight capability info from drm_edp_backlight_init()
/// @level: The initial backlight level to set via AUX, if there is one
///
/// This function handles enabling DPCD backlight controls on a panel over DPCD, while additionally
/// restoring any important backlight state such as the given backlight level, the brightness byte
/// count, backlight frequency, etc.
///
/// Note that certain panels do not support being enabled or disabled via DPCD, but instead require
/// that the driver handle enabling/disabling the panel through implementation-specific means using
/// the EDP_BL_PWR GPIO. For such panels, &drm_edp_backlight_info.aux_enable will be set to %false,
/// this function becomes a no-op, and the driver is expected to handle powering the panel on using
/// the EDP_BL_PWR GPIO.
///
/// Returns: %0 on success, negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_edp_backlight_enable")]
extern "C" fn drm_edp_backlight_enable(aux: *mut drm_dp_aux, bl: *const drm_edp_backlight_info, level: u32) -> i32 {
let mut ret: i32 = 0;
let mut dpcd_buf: u8 = if ((unsafe { (*bl).__slate_bits_0.__get_aux_set() }) as i32) != 0 {
2u8
} else {
0u8
};
if ((unsafe { (*bl).__slate_bits_0.__get_luminance_set() }) as i32) != 0 {
dpcd_buf = ((dpcd_buf as i32) | 128i32) as u8;
}
if (unsafe { (*bl).pwmgen_bit_count }) != 0 {
ret = drm_dp_dpcd_write_byte(aux, 1828u32, unsafe { (*bl).pwmgen_bit_count });
if ret < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Failed to write aux pwmgen bit count: %d\n".as_ptr() as *const u8, (*aux).name, ret) };
}
}
if (unsafe { (*bl).pwm_freq_pre_divider }) != 0 {
ret = drm_dp_dpcd_write_byte(aux, 1832u32, unsafe { (*bl).pwm_freq_pre_divider });
if ret < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Failed to write aux backlight frequency: %d\n".as_ptr() as *const u8, (*aux).name, ret) };
} else {
dpcd_buf = ((dpcd_buf as i32) | 8i32) as u8;
}
}
ret = drm_dp_dpcd_write_byte(aux, 1825u32, dpcd_buf);
if ret < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Failed to write aux backlight mode: %d\n".as_ptr() as *const u8, (*aux).name, ret) };
return if ret < 0 { ret } else { 5i32.wrapping_neg() };
}
ret = drm_edp_backlight_set_level(aux, bl, level);
if ret < 0 {
return ret;
}
ret = drm_edp_backlight_set_enable(aux, bl, true);
if ret < 0 {
return ret;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edp_backlight_enable_799: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *const drm_edp_backlight_info, u32) -> i32>, *mut ()>(Some(drm_edp_backlight_enable)) };

/// drm_edp_backlight_disable() - Disable an eDP backlight using DPCD, if supported
/// @aux: The DP AUX channel to use
/// @bl: Backlight capability info from drm_edp_backlight_init()
///
/// This function handles disabling DPCD backlight controls on a panel over AUX.
///
/// Note that certain panels do not support being enabled or disabled via DPCD, but instead require
/// that the driver handle enabling/disabling the panel through implementation-specific means using
/// the EDP_BL_PWR GPIO. For such panels, &drm_edp_backlight_info.aux_enable will be set to %false,
/// this function becomes a no-op, and the driver is expected to handle powering the panel off using
/// the EDP_BL_PWR GPIO.
///
/// Returns: %0 on success or no-op, negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_edp_backlight_disable")]
extern "C" fn drm_edp_backlight_disable(aux: *mut drm_dp_aux, bl: *const drm_edp_backlight_info) -> i32 {
let ret: i32 = drm_edp_backlight_set_enable(aux, bl, false);
if ret < 0 {
return ret;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edp_backlight_disable_800: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *const drm_edp_backlight_info) -> i32>, *mut ()>(Some(drm_edp_backlight_disable)) };

fn drm_edp_backlight_probe_max(aux: *mut drm_dp_aux, bl: *mut drm_edp_backlight_info, driver_pwm_freq_hz: u16, edp_dpcd: *const u8) -> i32 {
let mut __UNIQUE_ID_h__808: i32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_l__807: i32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_v__806: i32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_h__803: u8 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_l__802: u8 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_v__801: u8 = unsafe { core::mem::zeroed() };
let mut bit_count: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut u8 = core::ptr::addr_of_mut!(bit_count) as *mut u8;
let mut pn_max: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut u8 = core::ptr::addr_of_mut!(pn_max) as *mut u8;
let mut pn_min: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut u8 = core::ptr::addr_of_mut!(pn_min) as *mut u8;
let mut pn: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut u8 = core::ptr::addr_of_mut!(pn) as *mut u8;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut f: i32 = unsafe { core::mem::zeroed() };
let mut fxp_actual: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut i32 = core::ptr::addr_of_mut!(fxp_actual) as *mut i32;
let mut fxp_max: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut i32 = core::ptr::addr_of_mut!(fxp_max) as *mut i32;
let mut fxp_min: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut i32 = core::ptr::addr_of_mut!(fxp_min) as *mut i32;
let mut fxp: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_37: *mut i32 = core::ptr::addr_of_mut!(fxp) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_37 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
f = 1;
let __slate_auto_bytes_4 = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_6 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_7 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
if ((*bl).__slate_bits_0.__get_aux_set() as i32) == 0 {
return 0;
} else {
*__slate_slot_32 = drm_dp_dpcd_read_byte(aux, 1828u32, core::ptr::addr_of_mut!(*__slate_slot_28));
if *__slate_slot_32 < 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Failed to read pwmgen bit count cap: %d\n".as_ptr() as *const u8, (*aux).name, *__slate_slot_32);
return 19i32.wrapping_neg();
} else {
*__slate_slot_28 = ((*__slate_slot_28 as i32) & 31i32) as u8;
*__slate_slot_32 = drm_dp_dpcd_read_byte(aux, 1829u32, core::ptr::addr_of_mut!(*__slate_slot_30));
if *__slate_slot_32 < 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Failed to read pwmgen bit count cap min: %d\n".as_ptr() as *const u8, (*aux).name, *__slate_slot_32);
return 19i32.wrapping_neg();
} else {
*__slate_slot_30 = ((*__slate_slot_30 as i32) & 31i32) as u8;
*__slate_slot_32 = drm_dp_dpcd_read_byte(aux, 1830u32, core::ptr::addr_of_mut!(*__slate_slot_29));
if *__slate_slot_32 < 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Failed to read pwmgen bit count cap max: %d\n".as_ptr() as *const u8, (*aux).name, *__slate_slot_32);
return 19i32.wrapping_neg();
} else {
*__slate_slot_29 = ((*__slate_slot_29 as i32) & 31i32) as u8;
if unlikely__2!((*__slate_slot_30 as i32) > (*__slate_slot_29 as i32)) != 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Invalid pwmgen bit count cap min/max returned: %d %d\n".as_ptr() as *const u8, (*aux).name, *__slate_slot_30 as i32, *__slate_slot_29 as i32);
return 22i32.wrapping_neg();
} else {
// Per VESA eDP Spec v1.4b, section 3.3.10.2:
// If DP_EDP_PWMGEN_BIT_COUNT is less than DP_EDP_PWMGEN_BIT_COUNT_CAP_MIN,
// the sink must use the MIN value as the effective PWM bit count.
// Clamp the reported value to the [MIN, MAX] capability range to ensure
// correct brightness scaling on compliant eDP panels.
// Only enable this logic if the [MIN, MAX] range is valid in regard to Spec.
*__slate_slot_31 = *__slate_slot_28;
if (*__slate_slot_28 as i32) < (*__slate_slot_30 as i32) {
__UNIQUE_ID_v__801 = *__slate_slot_28;
__UNIQUE_ID_l__802 = *__slate_slot_30;
__UNIQUE_ID_h__803 = *__slate_slot_29;
loop {
if !(!statically_true!((__UNIQUE_ID_l__802 as i32) > (__UNIQUE_ID_h__803 as i32))) {
__compiletime_assert_804();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
while false {
}
*__slate_slot_31 = (if (__UNIQUE_ID_v__801 as i32) >= (__UNIQUE_ID_h__803 as i32) { __UNIQUE_ID_h__803 as i32 } else { if (__UNIQUE_ID_v__801 as i32) <= (__UNIQUE_ID_l__802 as i32) { __UNIQUE_ID_l__802 as i32 } else { __UNIQUE_ID_v__801 as i32 } }) as u8;
}
(*bl).max = (1i32 << (*__slate_slot_31 as i32)).wrapping_sub(1) as u32;
if driver_pwm_freq_hz == 0 {
if (*__slate_slot_31 as i32) == (*__slate_slot_28 as i32) {
return 0;
}
} else {
// Set PWM Frequency divider to match desired frequency provided by the driver.
// The PWM Frequency is calculated as 27Mhz / (F x P).
// - Where F = PWM Frequency Pre-Divider value programmed by field 7:0 of the
//             EDP_BACKLIGHT_FREQ_SET register (DPCD Address 00728h)
// - Where P = 2^Pn, where Pn is the value programmed by field 4:0 of the
//             EDP_PWMGEN_BIT_COUNT register (DPCD Address 00724h)
let __x_3: i32 = 27000000i32;
let __d_3: u16 = driver_pwm_freq_hz;
// Find desired value of (F x P)
// Note that, if F x P is out of supported range, the maximum value or minimum value will
// applied automatically. So no need to check that.
*__slate_slot_37 = if 1i32.wrapping_neg() > 0 || ((1i32.wrapping_neg() as u16) as i32) > 0 || ((__x_3 > 0) as i32) == (((__d_3 as i32) > 0) as i32) { __x_3.wrapping_add((__d_3 as i32) / 2) / (__d_3 as i32) } else { __x_3.wrapping_sub((__d_3 as i32) / 2) / (__d_3 as i32) };
let __x_2: i32 = (*__slate_slot_37).wrapping_mul(3);
let __d_2: i32 = 4;
// Use highest possible value of Pn for more granularity of brightness adjustment while
// satisfying the conditions below.
// - Pn is in the range of Pn_min and Pn_max
// - F is in the range of 1 and 255
// - FxP is within 25% of desired value.
//   Note: 25% is arbitrary value and may need some tweak.
//
// Ensure frequency is within 25% of desired value
*__slate_slot_36 = if 1i32.wrapping_neg() > 0 || 1i32.wrapping_neg() > 0 || ((__x_2 > 0) as i32) == ((__d_2 > 0) as i32) { __x_2.wrapping_add(__d_2 / 2) / __d_2 } else { __x_2.wrapping_sub(__d_2 / 2) / __d_2 };
let __x_1: i32 = (*__slate_slot_37).wrapping_mul(5);
let __d_1: i32 = 4;
*__slate_slot_35 = if 1i32.wrapping_neg() > 0 || 1i32.wrapping_neg() > 0 || ((__x_1 > 0) as i32) == ((__d_1 > 0) as i32) { __x_1.wrapping_add(__d_1 / 2) / __d_1 } else { __x_1.wrapping_sub(__d_1 / 2) / __d_1 };
if *__slate_slot_36 < 1i32 << (*__slate_slot_30 as i32) || 255i32 << (*__slate_slot_29 as i32) < *__slate_slot_35 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Driver defined backlight frequency (%d) out of range\n".as_ptr() as *const u8, (*aux).name, driver_pwm_freq_hz as i32);
return 0;
} else {
*__slate_slot_31 = *__slate_slot_29;
while (*__slate_slot_31 as i32) >= (*__slate_slot_30 as i32) {
let __x: i32 = *__slate_slot_37;
let __d: i32 = 1i32 << (*__slate_slot_31 as i32);
__UNIQUE_ID_v__806 = if 1i32.wrapping_neg() > 0 || 1i32.wrapping_neg() > 0 || ((__x > 0) as i32) == ((__d > 0) as i32) { __x.wrapping_add(__d / 2) / __d } else { __x.wrapping_sub(__d / 2) / __d };
__UNIQUE_ID_l__807 = 1;
__UNIQUE_ID_h__808 = 255;
loop {
if !(!statically_true!(__UNIQUE_ID_l__807 > __UNIQUE_ID_h__808)) {
__compiletime_assert_809();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
loop {
if __types_ok3!(__UNIQUE_ID_v__806, __UNIQUE_ID_l__807, __UNIQUE_ID_h__808) == 0 {
__compiletime_assert_810();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
f = if __UNIQUE_ID_v__806 >= __UNIQUE_ID_h__808 { __UNIQUE_ID_h__808 } else { if __UNIQUE_ID_v__806 <= __UNIQUE_ID_l__807 { __UNIQUE_ID_l__807 } else { __UNIQUE_ID_v__806 } };
*__slate_slot_34 = f << (*__slate_slot_31 as i32);
if *__slate_slot_36 <= *__slate_slot_34 && *__slate_slot_34 <= *__slate_slot_35 {
break;
} else {
*__slate_slot_31 = (*__slate_slot_31 as i32).wrapping_sub(1) as u8;
}
}
}
}
*__slate_slot_32 = drm_dp_dpcd_write_byte(aux, 1828u32, *__slate_slot_31);
if *__slate_slot_32 < 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, (b"%s: Failed to write aux pwmgen bit count: %d\n\0".as_ptr() as *mut u8) as *const u8, (*aux).name, *__slate_slot_32);
return 0;
} else {
if driver_pwm_freq_hz == 0 {
return 0;
} else {
(*bl).pwmgen_bit_count = *__slate_slot_31;
(*bl).max = (1i32 << (*__slate_slot_31 as i32)).wrapping_sub(1) as u32;
if (*edp_dpcd.wrapping_offset(2 as isize) as i32) & 32i32 != 0 {
(*bl).pwm_freq_pre_divider = f as u8;
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Using backlight frequency from driver (%dHz)\n".as_ptr() as *const u8, (*aux).name, driver_pwm_freq_hz as i32);
}
return 0;
}
}
}
}
}
}
}
}
}

fn drm_edp_backlight_probe_state(aux: *mut drm_dp_aux, bl: *mut drm_edp_backlight_info, current_mode: *mut u8) -> i32 {
let mut ret: i32 = 0;
let mut buf: [u8; 3] = [0u8; 3];
let mut mode_reg: u8 = 0;
ret = drm_dp_dpcd_read_byte(aux, 1825u32, core::ptr::addr_of_mut!(mode_reg));
if ret < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Failed to read backlight mode: %d\n".as_ptr() as *const u8, (*aux).name, ret) };
return if ret < 0 { ret } else { 5i32.wrapping_neg() };
}
unsafe {
*current_mode = ((mode_reg as i32) & 3i32) as u8;
}
if ((unsafe { (*bl).__slate_bits_0.__get_aux_set() }) as i32) == 0 {
return 0;
}
if ((unsafe { *current_mode }) as i32) == 2i32 {
let size: i32 = 1i32.wrapping_add((unsafe { (*bl).__slate_bits_0.__get_lsb_reg_used() }) as i32);
if ((unsafe { (*bl).__slate_bits_0.__get_luminance_set() }) as i32) != 0 {
ret = drm_dp_dpcd_read_data(aux, 1844u32, buf.as_mut_ptr() as *mut (), core::mem::size_of::<[u8; 3]>() as u64);
if ret < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Failed to read backlight level: %d\n".as_ptr() as *const u8, (*aux).name, ret) };
return ret;
}
// Incase luminance is set we want to send the value back in nits but
// since DP_EDP_PANEL_TARGET_LUMINANCE stores values in millinits we
// need to divide by 1000.
return ((buf[0] as i32) | (buf[1] as i32) << 8i32 | (buf[2] as i32) << 16i32) / 1000;
} else {
ret = drm_dp_dpcd_read_data(aux, 1826u32, buf.as_mut_ptr() as *mut (), size as u64);
if ret < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Failed to read backlight level: %d\n".as_ptr() as *const u8, (*aux).name, ret) };
return ret;
}
if ((unsafe { (*bl).__slate_bits_0.__get_lsb_reg_used() }) as i32) != 0 {
return (buf[0] as i32) << 8i32 | (buf[1] as i32);
} else {
return buf[0] as i32;
}
}
}
// If we're not in DPCD control mode yet, the programmed brightness value is meaningless and
// the driver should assume max brightness
(unsafe { (*bl).max }) as i32
}

/// drm_edp_backlight_init() - Probe a display panel's TCON using the standard VESA eDP backlight
/// interface.
/// @aux: The DP aux device to use for probing
/// @bl: The &drm_edp_backlight_info struct to fill out with information on the backlight
/// @max_luminance: max luminance when need luminance is set as true
/// @driver_pwm_freq_hz: Optional PWM frequency from the driver in hz
/// @edp_dpcd: A cached copy of the eDP DPCD
/// @current_level: Where to store the probed brightness level, if any
/// @current_mode: Where to store the currently set backlight control mode
/// @need_luminance: Tells us if a we want to manipulate backlight using luminance values
///
/// Initializes a &drm_edp_backlight_info struct by probing @aux for it's backlight capabilities,
/// along with also probing the current and maximum supported brightness levels.
///
/// If @driver_pwm_freq_hz is non-zero, this will be used as the backlight frequency. Otherwise, the
/// default frequency from the panel is used.
///
/// Returns: %0 on success, negative error code on failure.
#[unsafe(no_mangle)]
extern "C" fn drm_edp_backlight_init(aux: *mut drm_dp_aux, bl: *mut drm_edp_backlight_info, max_luminance: u32, driver_pwm_freq_hz: u16, edp_dpcd: *const u8, current_level: *mut u32, current_mode: *mut u8, need_luminance: bool) -> i32 {
let mut ret: i32 = 0;
if ((unsafe { *edp_dpcd.wrapping_offset(1 as isize) }) as i32) & 4i32 != 0 {
unsafe {
let __t0 = true;
(*bl).__slate_bits_0.__set_aux_enable(__t0);
}
}
if ((unsafe { *edp_dpcd.wrapping_offset(2 as isize) }) as i32) & 2i32 != 0 {
unsafe {
let __t1 = true;
(*bl).__slate_bits_0.__set_aux_set(__t1);
}
}
if ((unsafe { *edp_dpcd.wrapping_offset(2 as isize) }) as i32) & 4i32 != 0 {
unsafe {
let __t2 = true;
(*bl).__slate_bits_0.__set_lsb_reg_used(__t2);
}
}
if ((unsafe { *edp_dpcd.wrapping_offset(0 as isize) }) as i32) & 6 != 0 && ((unsafe { *edp_dpcd.wrapping_offset(3 as isize) }) as i32) & 16i32 != 0 && need_luminance {
unsafe {
let __t3 = true;
(*bl).__slate_bits_0.__set_luminance_set(__t3);
}
}
// Sanity check caps
if ((unsafe { (*bl).__slate_bits_0.__get_aux_set() }) as i32) == 0 && ((unsafe { *edp_dpcd.wrapping_offset(2 as isize) }) as i32) & 1i32 == 0 && ((unsafe { (*bl).__slate_bits_0.__get_luminance_set() }) as i32) == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Panel does not support AUX, PWM or luminance-based brightness control. Aborting\n".as_ptr() as *const u8, (*aux).name) };
return 22i32.wrapping_neg();
}
if ((unsafe { (*bl).__slate_bits_0.__get_luminance_set() }) as i32) != 0 {
unsafe {
(*bl).max = max_luminance;
}
} else {
ret = drm_edp_backlight_probe_max(aux, bl, driver_pwm_freq_hz, edp_dpcd);
if ret < 0 {
return ret;
}
}
ret = drm_edp_backlight_probe_state(aux, bl, current_mode);
if ret < 0 {
return ret;
}
unsafe {
*current_level = ret as u32;
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Found backlight: aux_set=%d aux_enable=%d mode=%d\n".as_ptr() as *const u8, (*aux).name, (*bl).__slate_bits_0.__get_aux_set() as i32, (*bl).__slate_bits_0.__get_aux_enable() as i32, *current_mode as i32) };
if ((unsafe { (*bl).__slate_bits_0.__get_aux_set() }) as i32) != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*aux).drm_dev as *const drm_device) as *const device, 2u32, c"%s: Backlight caps: level=%d/%d pwm_freq_pre_divider=%d lsb_reg_used=%d\n".as_ptr() as *const u8, (*aux).name, *current_level, (*bl).max, (*bl).pwm_freq_pre_divider as i32, (*bl).__slate_bits_0.__get_lsb_reg_used() as i32) };
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_edp_backlight_init_811: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_dp_aux, *mut drm_edp_backlight_info, u32, u16, *const u8, *mut u32, *mut u8, bool) -> i32>, *mut ()>(Some(drm_edp_backlight_init)) };

#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.dp_aux_backlight_update_status")]
extern "C" fn dp_aux_backlight_update_status(bd: *mut backlight_device) -> i32 {
let bl: *mut dp_aux_backlight = bl_get_data(bd) as *mut dp_aux_backlight;
let brightness: u16 = backlight_get_brightness(bd as *const backlight_device) as u16;
let mut ret: i32 = 0;
if !backlight_is_blank(bd as *const backlight_device) {
if !unsafe { (*bl).enabled } {
drm_edp_backlight_enable(unsafe { (*bl).aux }, (unsafe { core::ptr::addr_of_mut!((*bl).info) }) as *const drm_edp_backlight_info, brightness as u32);
unsafe {
(*bl).enabled = true;
}
return 0;
}
ret = drm_edp_backlight_set_level(unsafe { (*bl).aux }, (unsafe { core::ptr::addr_of_mut!((*bl).info) }) as *const drm_edp_backlight_info, brightness as u32);
} else {
if unsafe { (*bl).enabled } {
drm_edp_backlight_disable(unsafe { (*bl).aux }, (unsafe { core::ptr::addr_of_mut!((*bl).info) }) as *const drm_edp_backlight_info);
unsafe {
(*bl).enabled = false;
}
}
}
ret
}

static dp_aux_bl_ops: __SlateConst<backlight_ops> = __SlateConst({
let mut __t0: backlight_ops = unsafe { core::mem::zeroed() };
__t0.update_status = Some(dp_aux_backlight_update_status);
__t0
});

/// drm_panel_dp_aux_backlight - create and use DP AUX backlight
/// @panel: DRM panel
/// @aux: The DP AUX channel to use
///
/// Use this function to create and handle backlight if your panel
/// supports backlight control over DP AUX channel using DPCD
/// registers as per VESA's standard backlight control interface.
///
/// When the panel is enabled backlight will be enabled after a
/// successful call to &drm_panel_funcs.enable()
///
/// When the panel is disabled backlight will be disabled before the
/// call to &drm_panel_funcs.disable().
///
/// A typical implementation for a panel driver supporting backlight
/// control over DP AUX will call this function at probe time.
/// Backlight will then be handled transparently without requiring
/// any intervention from the driver.
///
/// Return: 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_panel_dp_aux_backlight")]
extern "C" fn drm_panel_dp_aux_backlight(panel: *mut drm_panel, aux: *mut drm_dp_aux) -> i32 {
let mut bl: *mut dp_aux_backlight = unsafe { core::mem::zeroed() };
let mut props: backlight_properties = {
let mut __t0: backlight_properties = unsafe { core::mem::zeroed() };
__t0.brightness = 0i32;
__t0
};
let mut current_level: u32 = 0;
let mut current_mode: u8 = 0;
let mut edp_dpcd: [u8; 5] = [0u8; 5];
let mut ret: i32 = 0;
if panel.is_null() || unsafe { (*panel).dev }.is_null() || aux.is_null() {
return 22i32.wrapping_neg();
}
ret = drm_dp_dpcd_read_data(aux, 1792u32, edp_dpcd.as_mut_ptr() as *mut (), 5u64);
if ret < 0 {
return ret;
}
if !drm_edp_backlight_supported(edp_dpcd.as_mut_ptr() as *const u8) {
unsafe { drm_dev_printk((*panel).dev as *const device, c"\x016".as_ptr() as *const u8, c"DP AUX backlight is not supported\n".as_ptr() as *const u8) };
return 0;
}
bl = devm_kzalloc(unsafe { (*panel).dev }, core::mem::size_of::<dp_aux_backlight>() as u64, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *mut dp_aux_backlight;
if bl.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*bl).aux = aux;
}
ret = drm_edp_backlight_init(aux, unsafe { core::ptr::addr_of_mut!((*bl).info) }, 0u32, 0u16, edp_dpcd.as_mut_ptr() as *const u8, core::ptr::addr_of_mut!(current_level), core::ptr::addr_of_mut!(current_mode), false);
if ret < 0 {
return ret;
}
props.r#type = 1;
props.brightness = current_level as i32;
props.max_brightness = (unsafe { (*bl).info.max }) as i32;
unsafe {
(*bl).base = devm_backlight_device_register((*panel).dev, c"dp_aux_backlight".as_ptr() as *const u8, (*panel).dev, bl as *mut (), core::ptr::addr_of!(dp_aux_bl_ops.0), core::ptr::addr_of_mut!(props) as *const backlight_properties);
}
if IS_ERR((unsafe { (*bl).base }) as *const ()) {
return PTR_ERR((unsafe { (*bl).base }) as *const ()) as i32;
}
backlight_disable(unsafe { (*bl).base });
unsafe {
(*panel).backlight = (*bl).base;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_panel_dp_aux_backlight_812: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_panel, *mut drm_dp_aux) -> i32>, *mut ()>(Some(drm_panel_dp_aux_backlight)) };

/// See DP Standard v2.1 2.6.4.4.1.1, 2.8.4.4, 2.8.7
fn drm_dp_link_data_symbol_cycles(lane_count: i32, pixels: i32, bpp_x16: i32, symbol_size: i32, is_mst: bool) -> i32 {
let cycles: i32 = pixels.wrapping_mul(bpp_x16).wrapping_add(16i32.wrapping_mul(symbol_size).wrapping_mul(lane_count)).wrapping_sub(1) / 16i32.wrapping_mul(symbol_size).wrapping_mul(lane_count);
let align: i32 = if is_mst { 4 / lane_count } else { 1 };
ALIGN__2!(cycles, align)
}

/// drm_dp_link_symbol_cycles - calculate the link symbol count with/without dsc
/// @lane_count: DP link lane count
/// @pixels: number of pixels in a scanline
/// @dsc_slice_count: number of slices for DSC or '0' for non-DSC
/// @bpp_x16: bits per pixel in .4 binary fixed format
/// @symbol_size: DP symbol size
/// @is_mst: %true for MST and %false for SST
///
/// Calculate the link symbol cycles for both DSC (@dsc_slice_count !=0) and
/// non-DSC case (@dsc_slice_count == 0) and return the count.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_link_symbol_cycles")]
extern "C" fn drm_dp_link_symbol_cycles(lane_count: i32, pixels: i32, dsc_slice_count: i32, bpp_x16: i32, symbol_size: i32, is_mst: bool) -> i32 {
let mut slice_count: i32 = 0;
let __v0: i32 = dsc_slice_count;
slice_count = if __v0 != 0 { __v0 } else { 1 };
let slice_pixels: i32 = pixels.wrapping_add(slice_count).wrapping_sub(1) / slice_count;
let slice_data_cycles: i32 = drm_dp_link_data_symbol_cycles(lane_count, slice_pixels, bpp_x16, symbol_size, is_mst);
let mut slice_eoc_cycles: i32 = 0;
if dsc_slice_count != 0 {
slice_eoc_cycles = if is_mst { 4 / lane_count } else { 1 };
}
slice_count.wrapping_mul(slice_data_cycles.wrapping_add(slice_eoc_cycles))
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_link_symbol_cycles_813: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, i32, i32, i32, i32, bool) -> i32>, *mut ()>(Some(drm_dp_link_symbol_cycles)) };

/// drm_dp_bw_overhead - Calculate the BW overhead of a DP link stream
/// @lane_count: DP link lane count
/// @hactive: pixel count of the active period in one scanline of the stream
/// @dsc_slice_count: number of slices for DSC or '0' for non-DSC
/// @bpp_x16: bits per pixel in .4 binary fixed point
/// @flags: DRM_DP_OVERHEAD_x flags
///
/// Calculate the BW allocation overhead of a DP link stream, depending
/// on the link's
/// - @lane_count
/// - SST/MST mode (@flags / %DRM_DP_OVERHEAD_MST)
/// - symbol size (@flags / %DRM_DP_OVERHEAD_UHBR)
/// - FEC mode (@flags / %DRM_DP_OVERHEAD_FEC)
/// - SSC/REF_CLK mode (@flags / %DRM_DP_OVERHEAD_SSC_REF_CLK)
/// as well as the stream's
/// - @hactive timing
/// - @bpp_x16 color depth
/// - compression mode (@dsc_slice_count != 0)
/// Note that this overhead doesn't account for the 8b/10b, 128b/132b
/// channel coding efficiency, for that see
/// @drm_dp_link_bw_channel_coding_efficiency().
///
/// Returns the overhead as 100% + overhead% in 1ppm units.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_bw_overhead")]
extern "C" fn drm_dp_bw_overhead(lane_count: i32, hactive: i32, dsc_slice_count: i32, bpp_x16: i32, flags: u64) -> i32 {
let symbol_size: i32 = if flags & BIT!(1i32) != 0 { 32 } else { 8 };
let is_mst: bool = flags & BIT!(0i32) != 0;
let mut overhead: u32 = 1000000;
let mut symbol_cycles: i32 = 0;
if lane_count == 0 || hactive == 0 || bpp_x16 == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"Invalid BW overhead params: lane_count %d, hactive %d, bpp_x16 %d.%04d\n".as_ptr() as *const u8, lane_count, hactive, fxp_q4_to_int(bpp_x16), fxp_q4_to_frac(bpp_x16).wrapping_mul(625)) };
return 0;
}
// DP Standard v2.1 2.6.4.1
// SSC downspread and ref clock variation margin:
//   5300ppm + 300ppm ~ 0.6%
if flags & BIT!(2i32) != 0 {
overhead = overhead.wrapping_add(6000);
}
// DP Standard v2.1 2.6.4.1.1, 3.5.1.5.4:
// FEC symbol insertions for 8b/10b channel coding:
// After each 250 data symbols on 2-4 lanes:
//   250 LL + 5 FEC_PARITY_PH + 1 CD_ADJ   (256 byte FEC block)
// After each 2 x 250 data symbols on 1 lane:
//   2 * 250 LL + 11 FEC_PARITY_PH + 1 CD_ADJ (512 byte FEC block)
// After 256 (2-4 lanes) or 128 (1 lane) FEC blocks:
//   256 * 256 bytes + 1 FEC_PM
// or
//   128 * 512 bytes + 1 FEC_PM
// (256 * 6 + 1) / (256 * 250) = 2.4015625 %
if flags & BIT!(3i32) != 0 {
overhead = overhead.wrapping_add(24016);
}
// DP Standard v2.1 2.7.9, 5.9.7
// The FEC overhead for UHBR is accounted for in its 96.71% channel
// coding efficiency.
let mut __v4: i64;
let __ret_warn_on: i32 = (flags & BIT!(1i32) != 0 && flags & BIT!(3i32) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("814: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 814b - ., 3; .popsection\n# {0}", const 814 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_dp_bw_overhead_5, sym __slate_asm_string_drm_dp_bw_overhead_6, const 4844 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("815: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 815b - ., 4; .popsection\n# {0}", const 815 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v4 = (__ret_warn_on != 0) as i64;
symbol_cycles = drm_dp_link_symbol_cycles(lane_count, hactive, dsc_slice_count, bpp_x16, symbol_size, is_mst);
DIV_ROUND_UP_ULL!(mul_u32_u32(symbol_cycles.wrapping_mul(symbol_size).wrapping_mul(lane_count) as u32, overhead.wrapping_mul(16)), hactive.wrapping_mul(bpp_x16)) as i32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_bw_overhead_816: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, i32, i32, i32, u64) -> i32>, *mut ()>(Some(drm_dp_bw_overhead)) };

/// drm_dp_bw_channel_coding_efficiency - Get a DP link's channel coding efficiency
/// @is_uhbr: Whether the link has a 128b/132b channel coding
///
/// Return the channel coding efficiency of the given DP link type, which is
/// either 8b/10b or 128b/132b (aka UHBR). The corresponding overhead includes
/// the 8b -> 10b, 128b -> 132b pixel data to link symbol conversion overhead
/// and for 128b/132b any link or PHY level control symbol insertion overhead
/// (LLCP, FEC, PHY sync, see DP Standard v2.1 3.5.2.18). For 8b/10b the
/// corresponding FEC overhead is BW allocation specific, included in the value
/// returned by drm_dp_bw_overhead().
///
/// Returns the efficiency in the 100%/coding-overhead% ratio in
/// 1ppm units.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_bw_channel_coding_efficiency")]
extern "C" fn drm_dp_bw_channel_coding_efficiency(is_uhbr: bool) -> i32 {
if is_uhbr {
return 967100;
} else {
return 800000;
}
// Note that on 8b/10b MST the efficiency is only
// 78.75% due to the 1 out of 64 MTPH packet overhead,
// not accounted for here.
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_bw_channel_coding_efficiency_817: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(bool) -> i32>, *mut ()>(Some(drm_dp_bw_channel_coding_efficiency)) };

/// drm_dp_max_dprx_data_rate - Get the max data bandwidth of a DPRX sink
/// @max_link_rate: max DPRX link rate in 10kbps units
/// @max_lanes: max DPRX lane count
///
/// Given a link rate and lanes, get the data bandwidth.
///
/// Data bandwidth is the actual payload rate, which depends on the data
/// bandwidth efficiency and the link rate.
///
/// Note that protocol layers above the DPRX link level considered here can
/// further limit the maximum data rate. Such layers are the MST topology (with
/// limits on the link between the source and first branch device as well as on
/// the whole MST path until the DPRX link) and (Thunderbolt) DP tunnels -
/// which in turn can encapsulate an MST link with its own limit - with each
/// SST or MST encapsulated tunnel sharing the BW of a tunnel group.
///
/// Returns the maximum data rate in kBps units.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_dp_helper.drm_dp_max_dprx_data_rate")]
extern "C" fn drm_dp_max_dprx_data_rate(max_link_rate: i32, max_lanes: i32) -> i32 {
let ch_coding_efficiency: i32 = drm_dp_bw_channel_coding_efficiency(drm_dp_is_uhbr_rate(max_link_rate));
DIV_ROUND_DOWN_ULL!(mul_u32_u32(max_link_rate.wrapping_mul(10).wrapping_mul(max_lanes) as u32, ch_coding_efficiency as u32), 8000000i32) as i32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_dp_max_dprx_data_rate_818: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, i32) -> i32>, *mut ()>(Some(drm_dp_max_dprx_data_rate)) };

static __slate_asm_string_drm_dp_aux_crc_work_2: [u8; 1] = *b"\0";

static __slate_asm_string_drm_dp_aux_crc_work_3: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_helper.c\0";

static __slate_asm_string_drm_dp_aux_get_crc_1: [u8; 1] = *b"\0";

static __slate_asm_string_drm_dp_aux_get_crc_2: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_helper.c\0";

static __slate_asm_string_drm_dp_aux_register_1: [u8; 1] = *b"\0";

static __slate_asm_string_drm_dp_aux_register_2: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_helper.c\0";

static __slate_asm_string_drm_dp_bw_overhead_5: [u8; 1] = *b"\0";

static __slate_asm_string_drm_dp_bw_overhead_6: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_helper.c\0";

static __slate_asm_string_drm_dp_dpcd_probe_1: [u8; 1] = *b"\0";

static __slate_asm_string_drm_dp_dpcd_probe_2: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_helper.c\0";

static __slate_asm_string_drm_dp_dump_lttpr_desc_3: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_drm_dp_dump_lttpr_desc_4: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_helper.c\0";

static __slate_asm_string_drm_dp_phy_name_2: [u8; 1] = *b"\0";

static __slate_asm_string_drm_dp_phy_name_3: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_helper.c\0";

static __slate_asm_string_drm_dp_vsc_sdp_pack_13: [u8; 17] = *b"Missing case %d\n\0";

static __slate_asm_string_drm_dp_vsc_sdp_pack_14: [u8; 69] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/display/drm_dp_helper.c\0";
