
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
//! Universal Interface for Intel High Definition Audio Codec
//!
//! Copyright (c) 2004 Takashi Iwai <tiwai@suse.de>
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
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! SCHED_ISO: reserved but not implemented yet
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! @a is a power of 2 value
//! PTR_IF - evaluate to @ptr if @cond is true, or to NULL otherwise.
//! @cond: A conditional, usually in a form of IS_ENABLED(CONFIG_FOO)
//! @ptr: A pointer to assign if @cond is true.
//!
//! PTR_IF(IS_ENABLED(CONFIG_FOO), ptr) evaluates to @ptr if CONFIG_FOO is set
//! to 'y' or 'm', or to NULL otherwise. The @ptr argument must be a pointer.
//!
//! The macro can be very useful to help compiler dropping dead code.
//!
//! For instance, consider the following::
//!
//!     #ifdef CONFIG_FOO_SUSPEND
//!     static int foo_suspend(struct device *dev)
//!     {
//!        ...
//!     }
//!     #endif
//!
//!     static struct pm_ops foo_ops = {
//!     #ifdef CONFIG_FOO_SUSPEND
//!         .suspend = foo_suspend,
//!     #endif
//!     };
//!
//! While this works, the foo_suspend() macro is compiled conditionally,
//! only when CONFIG_FOO_SUSPEND is set. This is problematic, as there could
//! be a build bug in this function, we wouldn't have a way to know unless
//! the configuration option is set.
//!
//! An alternative is to declare foo_suspend() always, but mark it
//! as __maybe_unused. This works, but the __maybe_unused attribute
//! is required to instruct the compiler that the function may not
//! be referenced anywhere, and is safe to remove without making
//! a fuss about it. This makes the programmer responsible for tagging
//! the functions that can be garbage-collected.
//!
//! With the macro it is possible to write the following::
//!
//!     static int foo_suspend(struct device *dev)
//!     {
//!        ...
//!     }
//!
//!     static struct pm_ops foo_ops = {
//!         .suspend = PTR_IF(IS_ENABLED(CONFIG_FOO_SUSPEND), foo_suspend),
//!     };
//!
//! The foo_suspend() function will now be automatically dropped by the
//! compiler, and it does not require any specific attribute.
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
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! Generic info of form tag = "info"
//! Generate SD flag indexes
//! Runtime PM flag argument bits
//! These are specified by iBCS2
//! Epoll event masks
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! See the comment in asm-generic/io.h about ioremap_np().
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! MAX_PORT is the number of port
//! It must be sync with I915_MAX_PORTS defined i915_drv.h
//! fake mute bit set to amp verbs
//! AES/IEC958 channel status bits
//! dB scale
//! We abuse kcontrol_new.subdev field to pass the NID corresponding to
//! the given new control.  If id.subdev has a bit flag HDA_SUBDEV_NID_FLAG,
//! snd_hda_ctl_add() takes the lower-bit subdev value as a valid NID.
//!
//! Note that the subdevice field is cleared again before the real registration
//! in snd_hda_ctl_add(), so that this value won't appear in the outside.
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
// Memory page has hardware error
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
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
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
// Set the IDR flag and the IDR_FREE tag
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
// Generate SD flag bits
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
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
// GET verbs
// SET verbs
// AMP-out present
// AMP-parameter override
// digital I/O
// power control
// channel count ext
// headphone drive capable
// output capable
// input capable
// EAPD capable
// 50%
// ground
// 80%
// 100%
// Amplifier capabilities
// 0dB offset
// number of steps
// step size 0-32dB
// in 0.25dB
// mute capable
// driver-specific amp-caps: using bits 24-30
// min-volume = mute
// Display pin's device list length
// DIGITAL1 bits
// Hi-Z
// 50%
// ground
// 80%
// 100%
// max. codec address
// 0 = consumer, 1 = professional
// 0 = audio, 1 = non-audio
// mask - emphasis
// 50/15us emphasis
// 0 = copyright, 1 = not copyright
// mask - emphasis
// 50/15us emphasis
// this bits depends on the category code
// Accessor offsets for TLV data items
// Accessor offsets for min, mute and step items in dB scale type TLV
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
macro_rules! ARRAY_SIZE__9 {
() => {
((core::mem::size_of::<[u16; 5]>() as u64) / (core::mem::size_of::<u16>() as u64)).wrapping_add(__must_be_array__31!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__24 {
() => {
((core::mem::size_of::<[u16; 5]>() as u64) / (core::mem::size_of::<u16>() as u64)).wrapping_add(__must_be_array__64!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__26 {
() => {
((core::mem::size_of::<[u16; 5]>() as u64) / (core::mem::size_of::<u16>() as u64)).wrapping_add(__must_be_array__30!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__29 {
() => {
((core::mem::size_of::<[[u32; 2]; 4]>() as u64) / (core::mem::size_of::<[u32; 2]>() as u64)).wrapping_add(__must_be_array__37!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__33 {
() => {
((core::mem::size_of::<[[u32; 2]; 4]>() as u64) / (core::mem::size_of::<[u32; 2]>() as u64)).wrapping_add(__must_be_array__35!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__45 {
() => {
((core::mem::size_of::<[u32; 5]>() as u64) / (core::mem::size_of::<u32>() as u64)).wrapping_add(__must_be_array__11!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__49 {
() => {
((core::mem::size_of::<[u16; 32]>() as u64) / (core::mem::size_of::<u16>() as u64)).wrapping_add(__must_be_array__58!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__53 {
() => {
((core::mem::size_of::<[u16; 5]>() as u64) / (core::mem::size_of::<u16>() as u64)).wrapping_add(__must_be_array__61!() as u64)
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

macro_rules! __WRITE_ONCE__6 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u64, $val) };
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

/// __flex_counter() - Get pointer to counter member for the given
///                    flexible array, if it was annotated with __counted_by()
/// @FAM: Pointer to flexible array member of an addressable struct instance
///
/// For example, with:
///
/// 	struct foo {
/// 		int counter;
/// 		short array[] __counted_by(counter);
/// 	} *p;
///
/// __flex_counter(p->array) will resolve to &p->counter.
///
/// Note that Clang may not allow this to be assigned to a separate
/// variable; it must be used directly.
///
/// If p->array is unannotated, this returns (void *)NULL.
macro_rules! __flex_counter__2 {
($__obj_ptr:expr) => {
unsafe { core::ptr::addr_of_mut!((*$__obj_ptr).len) }
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

macro_rules! __must_be_array__11 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_4103>() as i32
};
}

macro_rules! __must_be_array__15 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_126_4>() as i32
};
}

macro_rules! __must_be_array__23 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_3536>() as i32
};
}

macro_rules! __must_be_array__25 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_1644>() as i32
};
}

macro_rules! __must_be_array__30 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_3842>() as i32
};
}

macro_rules! __must_be_array__31 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_3877>() as i32
};
}

macro_rules! __must_be_array__35 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_3957>() as i32
};
}

macro_rules! __must_be_array__37 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_3960>() as i32
};
}

macro_rules! __must_be_array__58 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_153>() as i32
};
}

macro_rules! __must_be_array__61 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_3825>() as i32
};
}

macro_rules! __must_be_array__64 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_3873>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__3 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_1644_3>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__45 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_4034>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__47 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_3536_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__49 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_1644_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__60 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_3536_3>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__76 {
() => {
core::mem::size_of::<__SlateRecord_codec_c_4034_2>() as i32
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

/// __set_flex_counter() - Set the counter associated with the given flexible
///                        array member that has been annoated by __counted_by().
/// @FAM: Instance of flexible array member within a given struct.
/// @COUNT: Value to store to the __counted_by annotated @FAM_PTR's counter.
///
/// This is a no-op if no annotation exists. Count needs to be checked with
/// overflows_flex_counter_type() before using this function.
macro_rules! __set_flex_counter__2 {
($COUNT:expr, $__obj_ptr:expr) => {
let mut __v: i32;
let __v2: i32 = ($COUNT) as i32;
unsafe {
*__flex_counter__2!($__obj_ptr) = __v2;
}
__v = __v2;
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
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u32>() }) as u64) < 4) as i32)) }
};
}

macro_rules! __timer_init {
($_timer:expr, $_fn:expr, $_flags:expr) => {
unsafe { timer_init_key($_timer, $_fn, ($_flags) as u32, core::ptr::null::<u8>(), core::ptr::null_mut::<lock_class_key>()) };
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

macro_rules! codec_has_clkstop {
($codec:expr) => {
(unsafe { (*$codec).core.power_caps }) & 1073741824u32
};
}

macro_rules! codec_has_epss {
($codec:expr) => {
(unsafe { (*$codec).core.power_caps }) & 2147483648u32
};
}

macro_rules! codec_in_pm {
($codec:expr) => {
snd_hdac_is_in_pm(unsafe { core::ptr::addr_of_mut!((*$codec).core) })
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
let mut __v: *mut hda_conn_list;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__6!() as isize)) }) as *mut ()) as *mut hda_conn_list;
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
macro_rules! container_of__11 {
($ptr:expr) => {
{
let mut __v: *mut hda_pcm;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__11!() as isize)) }) as *mut ()) as *mut hda_pcm;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

macro_rules! dev_to_hda_codec {
($_dev:expr) => {
{
let mut __v: *mut hda_codec;
__v = ((unsafe { ((($_dev) as *mut ()) as *const u8).wrapping_offset(-(offsetof__13!(hda_codec) as isize)) }) as *mut ()) as *mut hda_codec;
__v
}
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
size_mul($count, (core::mem::size_of::<u16>() as u64).wrapping_add(__must_be_array__15!() as u64))
};
}

macro_rules! get_amp_channels {
($kc:expr) => {
(unsafe { (*$kc).private_value }) >> 16i32 & 3
};
}

macro_rules! get_amp_direction_ {
($pv:expr) => {
($pv) >> 18i32 & 1
};
}

macro_rules! get_amp_direction {
($kc:expr) => {
get_amp_direction_!(unsafe { (*$kc).private_value })
};
}

macro_rules! get_amp_index_ {
($pv:expr) => {
($pv) >> 19i32 & 15
};
}

macro_rules! get_amp_index {
($kc:expr) => {
get_amp_index_!(unsafe { (*$kc).private_value })
};
}

macro_rules! get_amp_min_mute {
($kc:expr) => {
(unsafe { (*$kc).private_value }) >> 29i32 & 1
};
}

/// AMP control callbacks
///
/// retrieve parameters from private_value
macro_rules! get_amp_nid_ {
($pv:expr) => {
($pv) & 65535
};
}

macro_rules! get_amp_nid {
($kc:expr) => {
get_amp_nid_!(unsafe { (*$kc).private_value })
};
}

macro_rules! get_amp_offset {
($kc:expr) => {
(unsafe { (*$kc).private_value }) >> 23i32 & 63
};
}

macro_rules! hda_codec_dev {
($_dev:expr) => {
unsafe { core::ptr::addr_of_mut!((*$_dev).core.dev) }
};
}

macro_rules! hda_codec_is_power_on {
($codec:expr) => {
snd_hdac_is_power_on(unsafe { core::ptr::addr_of_mut!((*$codec).core) })
};
}

macro_rules! hda_codec_to_driver {
($codec:expr) => {
{
let mut __v: *mut hda_codec_driver;
__v = ((unsafe { (((*$codec).core.dev.driver as *mut ()) as *const u8).wrapping_offset(-(offsetof__14!(hda_codec_driver) as isize)) }) as *mut ()) as *mut hda_codec_driver;
__v
}
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
(1i32.wrapping_neg() as u32) < 1
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
macro_rules! list_entry__11 {
($ptr:expr) => {
container_of__16!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__3 {
($ptr:expr) => {
container_of__11!($ptr)
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
list_entry__3!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__14 {
($ptr:expr) => {
list_entry__11!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__20 {
($pos:expr, $member:ident) => {
list_entry__11!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__4 {
($pos:expr, $member:ident) => {
list_entry__3!(unsafe { (*$pos).$member.next })
};
}

macro_rules! offsetof__34 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, jackpoll_work.work) as u64
};
}

macro_rules! offsetof__11 {
() => {
core::mem::offset_of!(hda_pcm, list) as u64
};
}

macro_rules! offsetof__13 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, core.dev) as u64
};
}

macro_rules! offsetof__14 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, core.driver) as u64
};
}

macro_rules! offsetof__4 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, core) as u64
};
}

macro_rules! offsetof__6 {
() => {
0u64
};
}

macro_rules! offsetof {
() => {
core::mem::offset_of!(hda_codec, core.list) as u64
};
}

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
};
}

macro_rules! raw_spin_unlock {
($lock:expr) => {
unsafe { _raw_spin_unlock($lock) };
};
}

macro_rules! snd_BUG_ON__2 {
($condition:expr) => {
{
let mut __v: i64;
let __v2: i32 = (($condition) != 0) as i32;
__v = (__v2 != 0) as i64;
__v
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

macro_rules! snd_ctl_sync_vmaster_hook {
($kctl:expr) => {
unsafe { snd_ctl_sync_vmaster($kctl, true) };
};
}

/// lowlevel accessor with caching; use carefully
macro_rules! snd_hda_codec_amp_read {
($codec:expr, $nid:expr, $ch:expr, $dir:expr, $idx:expr) => {
snd_hdac_regmap_get_amp(unsafe { core::ptr::addr_of_mut!((*$codec).core) }, $nid, $ch, $dir, $idx)
};
}

/// lowlevel accessor with caching; use carefully
macro_rules! snd_hda_codec_amp_read__2 {
($codec:expr, $nid:expr, $ch:expr, $dir:expr, $idx:expr) => {
snd_hdac_regmap_get_amp(unsafe { core::ptr::addr_of_mut!((*$codec).core) }, $nid, $ch, ($dir) as i32, ($idx) as i32)
};
}

macro_rules! snd_hda_codec_cleanup_stream {
($codec:expr, $nid:expr) => {
__snd_hda_codec_cleanup_stream($codec, $nid, 0i32);
};
}

macro_rules! snd_hda_get_num_raw_conns {
($codec:expr, $nid:expr) => {
unsafe { snd_hdac_get_connections(core::ptr::addr_of_mut!((*$codec).core), $nid, core::ptr::null_mut::<u16>(), 0i32) }
};
}

macro_rules! snd_hda_get_raw_connections {
($codec:expr, $nid:expr, $list:expr, $max_conns:expr) => {
unsafe { snd_hdac_get_connections(core::ptr::addr_of_mut!((*$codec).core), $nid, $list, $max_conns) }
};
}

macro_rules! snd_hda_get_raw_connections__2 {
($codec:expr, $nid:expr, $list:expr, $max_conns:expr) => {
unsafe { snd_hdac_get_connections(core::ptr::addr_of_mut!((*$codec).core), $nid, $list, ($max_conns) as i32) }
};
}

macro_rules! snd_hda_is_supported_format {
($codec:expr, $nid:expr, $fmt:expr) => {
unsafe { snd_hdac_is_supported_format(core::ptr::addr_of_mut!((*$codec).core), $nid, $fmt) }
};
}

macro_rules! snd_hda_param_read {
($codec:expr, $nid:expr, $param:expr) => {
snd_hdac_read_parm(unsafe { core::ptr::addr_of_mut!((*$codec).core) }, $nid, $param)
};
}

macro_rules! snd_hda_power_down {
($codec:expr) => {
unsafe { snd_hdac_power_down(core::ptr::addr_of_mut!((*$codec).core)) };
};
}

macro_rules! snd_hda_power_down_pm {
($codec:expr) => {
unsafe { snd_hdac_power_down_pm(core::ptr::addr_of_mut!((*$codec).core)) };
};
}

/// power saving
macro_rules! snd_hda_power_up {
($codec:expr) => {
unsafe { snd_hdac_power_up(core::ptr::addr_of_mut!((*$codec).core)) }
};
}

macro_rules! snd_hda_power_up_pm {
($codec:expr) => {
unsafe { snd_hdac_power_up_pm(core::ptr::addr_of_mut!((*$codec).core)) }
};
}

macro_rules! snd_hda_power_up_pm__2 {
($codec:expr) => {
unsafe { snd_hdac_power_up_pm(core::ptr::addr_of_mut!((*$codec).core)) };
};
}

macro_rules! snd_hda_query_supported_pcm {
($codec:expr, $nid:expr, $ratesp:expr, $fmtsp:expr, $bpsp:expr) => {
unsafe { snd_hdac_query_supported_pcm(core::ptr::addr_of_mut!((*$codec).core), $nid, $ratesp, $fmtsp, core::ptr::null_mut::<u32>(), $bpsp) };
};
}

macro_rules! snd_hda_query_supported_pcm__2 {
($codec:expr, $nid:expr, $ratesp:expr, $fmtsp:expr, $subfmtp:expr, $bpsp:expr) => {
unsafe { snd_hdac_query_supported_pcm(core::ptr::addr_of_mut!((*$codec).core), $nid, $ratesp, $fmtsp, $subfmtp, $bpsp) }
};
}

macro_rules! snd_hda_regmap_sync {
($codec:expr) => {
unsafe { snd_hdac_regmap_sync(core::ptr::addr_of_mut!((*$codec).core)) };
};
}

/// snd_hdac_regmap_encode_amp - encode the AMP verb to a pseudo register
/// @nid: widget NID
/// @ch: channel (left = 0, right = 1)
/// @dir: direction (#HDA_INPUT, #HDA_OUTPUT)
/// @idx: input index value
///
/// Returns an encoded pseudo register.
macro_rules! snd_hdac_regmap_encode_amp {
($nid:expr, $ch:expr, $dir:expr, $idx:expr) => {
snd_hdac_regmap_encode_verb__2!($nid, 2816i32) | ((if $ch { 0i32 } else { 8192i32 }) as u32) | ((if ($dir) == 1 { 32768i32 } else { 0i32 }) as u32) | (($idx) as u32)
};
}

/// snd_hdac_regmap_encode_verb - encode the verb to a pseudo register
/// @nid: widget NID
/// @verb: codec verb
///
/// Returns an encoded pseudo register.
macro_rules! snd_hdac_regmap_encode_verb {
($nid:expr, $verb:expr) => {
($verb) << 8i32 | 524288 | (($nid) as u32) << 20i32
};
}

/// snd_hdac_regmap_encode_verb - encode the verb to a pseudo register
/// @nid: widget NID
/// @verb: codec verb
///
/// Returns an encoded pseudo register.
macro_rules! snd_hdac_regmap_encode_verb__2 {
($nid:expr, $verb:expr) => {
((($verb) << 8i32 | 524288) as u32) | (($nid) as u32) << 20i32
};
}

macro_rules! snd_kcontrol_chip {
($kcontrol:expr) => {
unsafe { (*$kcontrol).private_data }
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
macro_rules! struct_size__4 {
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
macro_rules! struct_size_t {
($count:expr) => {
struct_size__4!($count)
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

macro_rules! va_end {
() => {
};
}

unsafe extern "C" {
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut kmsg_fops: file_operations;
    static mut cpuinfo_op: seq_operations;
    static mut jiffies: u64;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut snd_pcm_std_chmaps: [snd_pcm_chmap_elem; 0];
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn strlen(arg0: *const u8) -> u64;
    fn strcmp(arg0: *const u8, arg1: *const u8) -> i32;
    fn sized_strscpy(arg0: *mut u8, arg1: *const u8, arg2: u64) -> i64;
    fn strncmp(arg0: *const u8, arg1: *const u8, arg2: u64) -> i32;
    fn kstrdup(arg0: *const u8, arg1: u32) -> *mut u8;
    fn sprintf(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn vsprintf(arg0: *mut u8, arg1: *const u8, arg2: core::ffi::VaList<'static>) -> i32;
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn scnprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn kvasprintf(arg0: u32, arg1: *const u8, arg2: core::ffi::VaList<'static>) -> *mut u8;
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn __compiletime_assert_16();
    fn __compiletime_assert_19();
    fn __compiletime_assert_20();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn delayed_work_timer_fn(arg0: *mut timer_list);
    fn msleep(arg0: u32);
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn ktime_get_mono_fast_ns() -> u64;
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn queue_delayed_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut delayed_work, arg3: u64) -> bool;
    fn cancel_delayed_work_sync(arg0: *mut delayed_work) -> bool;
    fn current_work() -> *mut work_struct;
    fn kfree(arg0: *const ());
    fn __compiletime_assert_208();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __bad_copy_from();
    fn __bad_copy_to();
    fn __copy_overflow(arg0: i32, arg1: u64);
    fn _copy_to_user(arg0: *mut (), arg1: *const (), arg2: u64) -> u64;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_368();
    fn __compiletime_assert_393();
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn device_release_driver(arg0: *mut device);
    fn put_device(arg0: *mut device);
    fn pm_runtime_force_suspend(arg0: *mut device) -> i32;
    fn __pm_runtime_resume(arg0: *mut device, arg1: i32) -> i32;
    fn pm_runtime_enable(arg0: *mut device);
    fn __pm_runtime_disable(arg0: *mut device, arg1: bool);
    fn pm_runtime_allow(arg0: *mut device);
    fn pm_runtime_forbid(arg0: *mut device);
    fn __pm_runtime_use_autosuspend(arg0: *mut device, arg1: bool);
    fn pm_runtime_set_autosuspend_delay(arg0: *mut device, arg1: i32);
    fn __compiletime_assert_459();
    fn pm_runtime_force_resume(arg0: *mut device) -> i32;
    fn snd_refcount_init(arg0: *mut snd_refcount);
    fn snd_refcount_put(arg0: *mut snd_refcount);
    fn snd_component_add(arg0: *mut snd_card, arg1: *const u8) -> i32;
    fn snd_device_new(arg0: *mut snd_card, arg1: u32, arg2: *mut (), arg3: *const snd_device_ops) -> i32;
    fn snd_device_disconnect(arg0: *mut snd_card, arg1: *mut ());
    fn snd_device_free(arg0: *mut snd_card, arg1: *mut ());
    fn snd_ctl_notify_one(arg0: *mut snd_card, arg1: u32, arg2: *mut snd_kcontrol, arg3: u32);
    fn snd_ctl_new1(arg0: *const snd_kcontrol_new, arg1: *mut ()) -> *mut snd_kcontrol;
    fn snd_ctl_add(arg0: *mut snd_card, arg1: *mut snd_kcontrol) -> i32;
    fn snd_ctl_remove(arg0: *mut snd_card, arg1: *mut snd_kcontrol) -> i32;
    fn snd_ctl_rename_id(arg0: *mut snd_card, arg1: *mut snd_ctl_elem_id, arg2: *mut snd_ctl_elem_id) -> i32;
    fn snd_ctl_find_id(arg0: *mut snd_card, arg1: *const snd_ctl_elem_id) -> *mut snd_kcontrol;
    fn snd_ctl_boolean_mono_info(arg0: *mut snd_kcontrol, arg1: *mut snd_ctl_elem_info) -> i32;
    fn snd_ctl_enum_info(arg0: *mut snd_ctl_elem_info, arg1: u32, arg2: u32, arg3: *const *const u8) -> i32;
    fn snd_ctl_make_virtual_master(arg0: *mut u8, arg1: *const u32) -> *mut snd_kcontrol;
    fn _snd_ctl_add_follower(arg0: *mut snd_kcontrol, arg1: *mut snd_kcontrol, arg2: u32) -> i32;
    fn snd_ctl_add_vmaster_hook(arg0: *mut snd_kcontrol, arg1: Option<unsafe extern "C" fn(*mut (), i32)>, arg2: *mut ()) -> i32;
    fn snd_ctl_sync_vmaster(arg0: *mut snd_kcontrol, arg1: bool);
    fn snd_ctl_apply_vmaster_followers(arg0: *mut snd_kcontrol, arg1: Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_kcontrol, *mut ()) -> i32>, arg2: *mut ()) -> i32;
    fn snd_pcm_suspend_all(arg0: *mut snd_pcm) -> i32;
    fn snd_pcm_hw_constraint_step(arg0: *mut snd_pcm_runtime, arg1: u32, arg2: i32, arg3: u64) -> i32;
    fn snd_pcm_add_chmap_ctls(arg0: *mut snd_pcm, arg1: i32, arg2: *const snd_pcm_chmap_elem, arg3: i32, arg4: u64, arg5: *mut *mut snd_pcm_chmap) -> i32;
    fn snd_hdac_device_init(arg0: *mut hdac_device, arg1: *mut hdac_bus, arg2: *const u8, arg3: u32) -> i32;
    fn snd_hdac_device_exit(arg0: *mut hdac_device);
    fn snd_hdac_device_unregister(arg0: *mut hdac_device);
    fn snd_hdac_refresh_widgets(arg0: *mut hdac_device) -> i32;
    fn snd_hdac_read(arg0: *mut hdac_device, arg1: u16, arg2: u32, arg3: u32, arg4: *mut u32) -> i32;
    fn _snd_hdac_read_parm(arg0: *mut hdac_device, arg1: u16, arg2: i32, arg3: *mut u32) -> i32;
    fn snd_hdac_read_parm_uncached(arg0: *mut hdac_device, arg1: u16, arg2: i32) -> i32;
    fn snd_hdac_override_parm(arg0: *mut hdac_device, arg1: u16, arg2: u32, arg3: u32) -> i32;
    fn snd_hdac_get_connections(arg0: *mut hdac_device, arg1: u16, arg2: *mut u16, arg3: i32) -> i32;
    fn snd_hdac_query_supported_pcm(arg0: *mut hdac_device, arg1: u16, arg2: *mut u32, arg3: *mut u64, arg4: *mut u32, arg5: *mut u32) -> i32;
    fn snd_hdac_is_supported_format(arg0: *mut hdac_device, arg1: u16, arg2: u32) -> bool;
    fn snd_hdac_codec_read(arg0: *mut hdac_device, arg1: u16, arg2: i32, arg3: u32, arg4: u32) -> i32;
    fn snd_hdac_codec_write(arg0: *mut hdac_device, arg1: u16, arg2: i32, arg3: u32, arg4: u32) -> i32;
    fn snd_hdac_check_power_state(arg0: *mut hdac_device, arg1: u16, arg2: u32) -> bool;
    fn snd_hdac_sync_power_state(arg0: *mut hdac_device, arg1: u16, arg2: u32) -> u32;
    fn snd_hdac_power_up(arg0: *mut hdac_device) -> i32;
    fn snd_hdac_power_down(arg0: *mut hdac_device) -> i32;
    fn snd_hdac_power_up_pm(arg0: *mut hdac_device) -> i32;
    fn snd_hdac_power_down_pm(arg0: *mut hdac_device) -> i32;
    fn snd_hdac_bus_exec_verb_unlocked(arg0: *mut hdac_bus, arg1: u32, arg2: u32, arg3: *mut u32) -> i32;
    fn snd_hdac_codec_link_up(arg0: *mut hdac_device);
    fn snd_hdac_codec_link_down(arg0: *mut hdac_device);
    fn snd_array_new(arg0: *mut snd_array) -> *mut ();
    fn snd_array_free(arg0: *mut snd_array);
    fn regcache_mark_dirty(arg0: *mut regmap);
    fn snd_hdac_regmap_exit(arg0: *mut hdac_device);
    fn snd_hdac_regmap_read_raw(arg0: *mut hdac_device, arg1: u32, arg2: *mut u32) -> i32;
    fn snd_hdac_regmap_write_raw(arg0: *mut hdac_device, arg1: u32, arg2: u32) -> i32;
    fn snd_hdac_regmap_update_raw(arg0: *mut hdac_device, arg1: u32, arg2: u32, arg3: u32) -> i32;
    fn snd_hdac_regmap_update_raw_once(arg0: *mut hdac_device, arg1: u32, arg2: u32, arg3: u32) -> i32;
    fn snd_hdac_regmap_sync(arg0: *mut hdac_device);
    fn snd_hda_attach_pcm_stream(arg0: *mut hda_bus, arg1: *mut hda_codec, arg2: *mut hda_pcm) -> i32;
    fn snd_hda_bus_reset(arg0: *mut hda_bus);
    fn snd_hda_codec_proc_new(arg0: *mut hda_codec) -> i32;
    fn snd_hda_create_hwdep(arg0: *mut hda_codec) -> i32;
    fn snd_hda_sysfs_init(arg0: *mut hda_codec);
    fn snd_hda_sysfs_clear(arg0: *mut hda_codec);
    fn snd_hda_jack_tbl_clear(arg0: *mut hda_codec);
    fn snd_hda_jack_set_dirty_all(arg0: *mut hda_codec);
    fn snd_hda_jack_report_sync(arg0: *mut hda_codec);
    fn snd_hda_jack_poll_all(arg0: *mut hda_codec);
    fn snd_hdac_display_power(arg0: *mut hdac_bus, arg1: u32, arg2: bool);
    fn __compiletime_assert_531();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.objectsize.i64.p0"]
    fn __slate_intrinsic_2913c0303de876b3(arg0: *const (), arg1: bool, arg2: bool, arg3: bool) -> u64;
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

// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
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

// instrument_get_user() - add instrumentation to get_user()-like macros
// @to: destination variable, may not be address-taken
//
// get_user() and friends are fragile, so it may depend on the implementation
// whether the instrumentation happens before or after the data is copied from
// the userspace.
#[inline(always)]
fn clear_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_clear_bit(nr, addr);
}

#[inline(always)]
fn test_and_set_bit(nr: i64, addr: *mut u64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_set_bit(nr, addr)
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits112U0,
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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
    __slate_bits_0: __slate_bits::__SlateBits117U0,
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
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[inline(always)]
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
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
fn atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_read(v)
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
fn atomic_inc(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc(v);
}

#[inline(always)]
fn atomic_dec(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
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
    __slate_bits_0: __slate_bits::__SlateBits118U0,
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct futex_unlock_cs_range {
    start_ip: u64,
    len: u32,
    pop_size32: u32,
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
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_head {
    rb_root: rb_root_cached,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spin_lock(lock: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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


#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
fn class_spinlock_constructor(l: *mut spinlock) -> __SlateRecord_spinlock_h_610 {
let mut _t: __SlateRecord_spinlock_h_610 = __SlateRecord_spinlock_h_610 { lock: l };
let _T: *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(_t);
spin_lock(unsafe { (*_T).lock });
return _t;
}

#[inline(always)]
fn __class_spinlock_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_610) {
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct load_weight {
    weight: u64,
    inv_weight: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
    __slate_bits_0: __slate_bits::__SlateBits277U0,
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
struct uprobe_xol_ops {
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
    __slate_bits_0: __slate_bits::__SlateBits68U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits68U1,
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
    __slate_bits_2: __slate_bits::__SlateBits68U2,
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
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
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
struct alloc_tag_counters {
    bytes: u64,
    calls: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
}


static class_mutex_is_conditional: bool = 0i32 != 0i32;

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
struct __SlateRecord_mutex_h_253 {
    lock: *mut mutex,
}


#[inline(always)]
fn class_mutex_destructor(_T: *mut __SlateRecord_mutex_h_253) {
unsafe { mutex_unlock((*_T).lock) };
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


#[inline(always)]
fn class_mutex_lock_ptr(_T: &mut __SlateRecord_mutex_h_253) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_mutex_h_253)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
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


#[inline(always)]
fn class_mutex_constructor(l: *mut mutex) -> __SlateRecord_mutex_h_253 {
let mut _t: __SlateRecord_mutex_h_253 = __SlateRecord_mutex_h_253 { lock: l };
let _T: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(_t);
unsafe { mutex_lock((*_T).lock) };
return _t;
}

#[inline(always)]
fn __class_mutex_cleanup_ctx(_T: *mut *mut __SlateRecord_mutex_h_253) {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
}


fn __init_work(work: *mut work_struct, onstack: i32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
}


fn queue_delayed_work(wq: *mut workqueue_struct, dwork: *mut delayed_work, delay: u64) -> bool {
unsafe { queue_delayed_work_on(64i32, wq, dwork, delay) }
}

fn schedule_delayed_work(dwork: *mut delayed_work, delay: u64) -> bool {
queue_delayed_work(unsafe { system_percpu_wq }, dwork, delay)
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
struct kmem_cache {
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
struct user_namespace {
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
struct folio_queue {
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
struct arch_uprobe {
    __slate_anon_0: __SlateRecord_uprobes_h_31,
    ops: *const uprobe_xol_ops,
    __slate_anon_2: __SlateRecord_uprobes_h_38,
    flags: u64,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_288 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits405U0,
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


// Opaque type representing current VMA (vm_area_struct) flag state. Must be
// accessed via vma_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_871 {
    __vma_flags: [u64; 1],
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
    __slate_bits_0: __slate_bits::__SlateBits438U0,
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
struct __SlateRecord_slab_h_532 {
}


// no-op
#[inline(always)]
fn __free_kfree(p: *mut ()) {
let _T: *mut () = unsafe { *(p as *mut *mut ()) };
if !IS_ERR_OR_NULL(_T as *const ()) {
unsafe { kfree(_T as *const ()) };
}
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
// kmalloc_objs - Allocate an array of the given type
// @VAR_OR_TYPE: Variable or type to allocate an array of.
// @COUNT: How many elements in the array.
// @...: optional GFP flags for the allocation (GFP_KERNEL when not specified).
//
// Returns: newly allocated pointer to array of @VAR_OR_TYPE on success,
// or NULL on failure.
// kmalloc_flex - Allocate a single instance of the given flexible structure
// @VAR_OR_TYPE: Variable or type to allocate (with its flex array).
// @FAM: The name of the flexible array member of the structure.
// @COUNT: How many flexible array member elements are desired.
// @...: optional GFP flags for the allocation (GFP_KERNEL when not specified).
//
// Returns: newly allocated pointer to @VAR_OR_TYPE on success, NULL on
// failure. If @FAM has been annotated with __counted_by(), the allocation
// will immediately fail if @COUNT is larger than what the type of the
// struct's counter variable can represent.
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
}


fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
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
core::arch::asm!("210: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 210b - ., 3; .popsection\n# {0}", const 210 as i32, options(nostack, att_syntax));
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
core::arch::asm!("211: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 211b - ., 4; .popsection\n# {0}", const 211 as i32, options(nostack, att_syntax));
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
#[repr(C)]
#[derive(Clone, Copy)]
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
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
fn copy_to_user(to: *mut (), from: *const (), n: u64) -> u64 {
if !check_copy_size(from, n, true) {
return n;
}
unsafe { _copy_to_user(to, from, n) }
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
struct __SlateRecord_capability_h_24 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct group_info {
    usage: refcount_struct,
    ngroups: i32,
    gid: [__SlateRecord_uidgid_types_h_13; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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


fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
struct ns_common {
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
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
    __slate_bits_0: __slate_bits::__SlateBits280U0,
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
unsafe { __compiletime_assert_368() };
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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
struct wake_irq {
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
unsafe { __compiletime_assert_393() };
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


#[repr(C)]
#[derive(Clone, Copy)]
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
union __SlateRecord_info_h_65 {
    text: snd_info_entry_text,
    ops: *const snd_info_entry_ops,
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
    __slate_bits_0: __slate_bits::__SlateBits630U0,
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
union __SlateRecord_asound_h_1164 {
    integer: __SlateRecord_asound_h_1165,
    integer64: __SlateRecord_asound_h_1170,
    enumerated: __SlateRecord_asound_h_1175,
    reserved: [u8; 128],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_asound_h_1165 {
    min: i64,
    max: i64,
    step: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_asound_h_1170 {
    min: i64,
    max: i64,
    step: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_asound_h_1175 {
    items: u32,
    item: u32,
    name: [u8; 64],
    names_ptr: u64,
    names_length: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct attribute {
    name: *const u8,
    mode: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_asound_h_1190 {
    integer: __SlateRecord_asound_h_1191,
    integer64: __SlateRecord_asound_h_1195,
    enumerated: __SlateRecord_asound_h_1199,
    bytes: __SlateRecord_asound_h_1203,
    iec958: snd_aes_iec958,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_asound_h_1191 {
    value: [i64; 128],
    value_ptr: *mut i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_asound_h_1195 {
    value: [i64; 64],
    value_ptr: *mut i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_asound_h_1199 {
    item: [u32; 128],
    item_ptr: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_asound_h_1203 {
    data: [u8; 512],
    data_ptr: *mut u8,
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
union __SlateRecord_control_h_59 {
    c: Option<unsafe extern "C" fn(*mut snd_kcontrol, i32, u32, *mut u32) -> i32>,
    p: *const u32,
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
    __slate_bits_0: __slate_bits::__SlateBits574U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_control_h_78 {
    c: Option<unsafe extern "C" fn(*mut snd_kcontrol, i32, u32, *mut u32) -> i32>,
    p: *const u32,
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
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
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
union __SlateRecord_pm_qos_h_109 {
    pnode: plist_node,
    flr: pm_qos_flags_request,
    freq: freq_qos_request,
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
    __slate_bits_0: __slate_bits::__SlateBits672U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits672U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits672U2,
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
    __slate_bits_3: __slate_bits::__SlateBits672U3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer {
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
struct hotplug_slot {
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
    __slate_bits_0: __slate_bits::__SlateBits669U0,
}


fn device_is_registered(dev: *mut device) -> i32 {
(unsafe { (*dev).kobj.__slate_bits_0.__get_state_in_sysfs() }) as i32
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

fn pm_runtime_mark_last_busy(dev: *mut device) {
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_459() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__6!((*dev).power.last_busy, unsafe { ktime_get_mono_fast_ns() });
if true {
break;
}
}
}

fn pm_request_resume(dev: *mut device) -> i32 {
unsafe { __pm_runtime_resume(dev, 1i32) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hdac_widget_tree {
}


fn pm_runtime_disable(dev: *mut device) {
unsafe { __pm_runtime_disable(dev, true) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct regmap {
}


fn pm_runtime_use_autosuspend(dev: *mut device) {
unsafe { __pm_runtime_use_autosuspend(dev, true) };
}

fn pm_runtime_dont_use_autosuspend(dev: *mut device) {
unsafe { __pm_runtime_use_autosuspend(dev, false) };
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


fn snd_refcount_get(r#ref: *mut snd_refcount) {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*r#ref).count) });
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


#[repr(C)]
#[derive(Clone, Copy)]
struct hda_device_id {
    vendor_id: u32,
    rev_id: u32,
    api_version: u8,
    name: *const u8,
    driver_data: u64,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct input_dev {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_aes_iec958 {
    status: [u8; 24],
    subcode: [u8; 147],
    pad: u8,
    dig_subframe: [u8; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_hwdep_dsp_status {
    version: u32,
    id: [u8; 32],
    num_dsps: u32,
    dsp_loaded: u32,
    chip_ready: u32,
    reserved: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_hwdep_dsp_image {
    index: u32,
    name: [u8; 64],
    image: *mut u8,
    length: u64,
    driver_data: u64,
}


// stream is running
// stream is draining
// Channels
// Approx rate
// Approx frames between
// interrupts
// Approx interrupts per
// buffer
// Size of buffer in frames
// Size of buffer in bytes
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_interval {
    min: u32,
    max: u32,
    __slate_bits_0: __slate_bits::__SlateBits738U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_mask {
    bits: [u32; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_pcm_hw_params {
    flags: u32,
    masks: [snd_mask; 3],
    mres: [snd_mask; 5],
    intervals: [snd_interval; 12],
    ires: [snd_interval; 9],
    rmask: u32,
    cmask: u32,
    info: u32,
    msbits: u32,
    rate_num: u32,
    rate_den: u32,
    fifo_size: u64,
    sync: [u8; 16],
    reserved: [u8; 48],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_hda_local_h_289 {
    pins: *const hda_pintbl,
    verbs: *const hda_verb,
    func: Option<unsafe extern "C" fn(*mut hda_codec, *const hda_fixup, i32)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_pcm_mmap_status {
    state: i32,
    pad1: u32,
    __pad1: [u8; 0],
    hw_ptr: u64,
    __pad2: [u8; 0],
    tstamp: __kernel_timespec,
    suspended_state: i32,
    pad3: u32,
    audio_tstamp: __kernel_timespec,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_pcm_mmap_control {
    __pad1: [u8; 0],
    appl_ptr: u64,
    __pad2: [u8; 0],
    __pad3: [u8; 0],
    avail_min: u64,
    __pad4: [u8; 0],
}


// boolean type
// integer type
// enumerated type
// IEC958 (S/PDIF) setup
// virtual mixer device
// (1 << 3) is unused.
// TLV read is possible
// kernel use a TLV callback
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_elem_id {
    numid: u32,
    iface: i32,
    device: u32,
    subdevice: u32,
    name: [u8; 44],
    index: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_elem_info {
    id: snd_ctl_elem_id,
    r#type: i32,
    access: u32,
    count: u32,
    owner: i32,
    value: __SlateRecord_asound_h_1164,
    reserved: [u8; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_elem_value {
    id: snd_ctl_elem_id,
    __slate_bits_0: __slate_bits::__SlateBits754U0,
    value: __SlateRecord_asound_h_1190,
    reserved: [u8; 128],
}


// element value was changed
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_kcontrol_new {
    iface: i32,
    device: u32,
    subdevice: u32,
    name: *const u8,
    index: u32,
    access: u32,
    count: u32,
    info: Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_info) -> i32>,
    get: Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_value) -> i32>,
    put: Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_value) -> i32>,
    tlv: __SlateRecord_control_h_59,
    private_value: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_126_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_153 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_kcontrol_volatile {
    owner: *mut snd_ctl_file,
    access: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_1644 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_1644_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_1644_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_kcontrol {
    list: list_head,
    id: snd_ctl_elem_id,
    count: u32,
    info: Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_info) -> i32>,
    get: Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_value) -> i32>,
    put: Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_value) -> i32>,
    tlv: __SlateRecord_control_h_78,
    private_value: u64,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_kcontrol)>,
    vd: [snd_kcontrol_volatile; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_3536 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_3536_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_3536_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_3825 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_3842 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_3873 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_3877 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_3957 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_3960 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_4034 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_4034_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_codec_c_4103 {
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
    pub struct __SlateBits277U0 {
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
    pub struct __SlateBits438U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits630U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits574U0 {
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
    pub struct __SlateBits117U0 {
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
    pub struct __SlateBits405U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits68U0 {
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
    pub struct __SlateBits68U1 {
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
    pub struct __SlateBits280U0 {
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
    pub struct __SlateBits68U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits112U0 {
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
    pub struct __SlateBits118U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits672U0 {
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
    pub struct __SlateBits669U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits672U1 {
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
    pub struct __SlateBits672U2 {
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
    pub struct __SlateBits672U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits840U0 {
        #[bits(1)]
        pub chip_init: bool,
        #[bits(1)]
        pub aligned_mmio: bool,
        #[bits(1)]
        pub sync_write: bool,
        #[bits(1)]
        pub use_posbuf: bool,
        #[bits(1)]
        pub snoop: bool,
        #[bits(1)]
        pub align_bdle_4k: bool,
        #[bits(1)]
        pub reverse_assign: bool,
        #[bits(1)]
        pub corbrp_self_clear: bool,
        #[bits(1)]
        pub polling_mode: bool,
        #[bits(1)]
        pub needs_damn_long_delay: bool,
        #[bits(1)]
        pub not_use_interrupts: bool,
        #[bits(1)]
        pub access_sdnctl_in_dword: bool,
        #[bits(1)]
        pub use_pio_for_commands: bool,
        #[bits(3, access = na)]
        pub __slate_pad_13: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits841U0 {
        #[bits(1)]
        pub afg_unsol: u32,
        #[bits(1)]
        pub mfg_unsol: u32,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits841U1 {
        #[bits(1)]
        pub lazy_cache: bool,
        #[bits(1)]
        pub caps_overwriting: bool,
        #[bits(1)]
        pub cache_coef: bool,
        #[bits(1)]
        pub registered: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits435U0 {
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
    pub struct __SlateBits815U0 {
        #[bits(1)]
        pub per_func_slot: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits709U0 {
        #[bits(3)]
        pub pcie_mpss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits709U1 {
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
    pub struct __SlateBits709U2 {
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
    pub struct __SlateBits709U3 {
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
    pub struct __SlateBits709U4 {
        #[bits(1)]
        pub pasid_required: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits851U0 {
        #[bits(1)]
        pub allow_bus_reset: u32,
        #[bits(1)]
        pub shutdown: u32,
        #[bits(1)]
        pub response_reset: u32,
        #[bits(1)]
        pub in_reset: u32,
        #[bits(1)]
        pub no_response_fallback: u32,
        #[bits(1)]
        pub bus_probing: u32,
        #[bits(1)]
        pub keep_power: u32,
        #[bits(1)]
        pub jackpoll_in_suspend: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits852U0 {
        #[bits(1)]
        pub registered: u32,
        #[bits(1)]
        pub enabled: u32,
        #[bits(1)]
        pub linear_tone: u32,
        #[bits(1)]
        pub playing: u32,
        #[bits(1)]
        pub keep_power_at_enable: u32,
        #[bits(3, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits807U0 {
        #[bits(1)]
        pub exclusive: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits853U0 {
        #[bits(1)]
        pub configured: u32,
        #[bits(1)]
        pub in_freeing: u32,
        #[bits(1)]
        pub display_power_control: u32,
        #[bits(1)]
        pub spdif_status_reset: u32,
        #[bits(1)]
        pub pin_amp_workaround: u32,
        #[bits(1)]
        pub single_adc_amp: u32,
        #[bits(1)]
        pub no_sticky_stream: u32,
        #[bits(1)]
        pub pins_shutup: u32,
        #[bits(1)]
        pub no_trigger_sense: u32,
        #[bits(1)]
        pub no_jack_detect: u32,
        #[bits(1)]
        pub inv_eapd: u32,
        #[bits(1)]
        pub inv_jack_detect: u32,
        #[bits(1)]
        pub pcm_format_first: u32,
        #[bits(1)]
        pub cached_write: u32,
        #[bits(1)]
        pub dp_mst: u32,
        #[bits(1)]
        pub dump_coef: u32,
        #[bits(1)]
        pub power_save_node: u32,
        #[bits(1)]
        pub auto_runtime_pm: u32,
        #[bits(1)]
        pub force_pin_prefix: u32,
        #[bits(1)]
        pub link_down_at_suspend: u32,
        #[bits(1)]
        pub relaxed_resume: u32,
        #[bits(1)]
        pub forced_resume: u32,
        #[bits(1)]
        pub acomp_requested_resume: u32,
        #[bits(1)]
        pub no_stream_clean_at_suspend: u32,
        #[bits(1)]
        pub ctl_dev_id: u32,
        #[bits(1)]
        pub eld_jack_detect: u32,
        #[bits(6, access = na)]
        pub __slate_pad_26: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits860U0 {
        #[bits(1)]
        pub chained: bool,
        #[bits(1)]
        pub chained_before: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits754U0 {
        #[bits(1)]
        pub indirect: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits738U0 {
        #[bits(1)]
        pub openmin: u32,
        #[bits(1)]
        pub openmax: u32,
        #[bits(1)]
        pub integer: u32,
        #[bits(1)]
        pub empty: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits794U0 {
        #[bits(4)]
        pub type_requested: u32,
        #[bits(1)]
        pub report_delay: u32,
        #[bits(3, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits795U0 {
        #[bits(1)]
        pub valid: u32,
        #[bits(4)]
        pub actual_type: u32,
        #[bits(1)]
        pub accuracy_report: u32,
        #[bits(2, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits800U0 {
        #[bits(1)]
        pub no_period_wakeup: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits800U1 {
        #[bits(1)]
        pub buffer_changed: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits793U0 {
        #[bits(1)]
        pub timer_running: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits793U1 {
        #[bits(1)]
        pub hw_opened: u32,
        #[bits(1)]
        pub managed_buffer_alloc: u32,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits854U0 {
        #[bits(1)]
        pub disconnected: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits880U0 {
        #[bits(1)]
        pub jack_detect: u32,
        #[bits(1)]
        pub jack_dirty: u32,
        #[bits(1)]
        pub phantom_jack: u32,
        #[bits(1)]
        pub block_report: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_file {
    list: list_head,
    card: *mut snd_card,
    pid: *mut pid,
    preferred_subdevice: [i32; 2],
    change_sleep: wait_queue_head,
    read_lock: spinlock,
    fasync: *mut snd_fasync,
    subscribed: i32,
    events: list_head,
}


// optional flags for follower
fn snd_ctl_add_follower(master: *mut snd_kcontrol, follower: *mut snd_kcontrol) -> i32 {
unsafe { _snd_ctl_add_follower(master, follower, 0u32) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct snd_dma_device {
    r#type: i32,
    dir: u32,
    need_sync: bool,
    dev: *mut device,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_dma_buffer {
    dev: snd_dma_device,
    area: *mut u8,
    addr: u64,
    bytes: u64,
    private_data: *mut (),
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
struct freq_constraints {
    min_freq: pm_qos_constraints,
    min_freq_notifiers: blocking_notifier_head,
    max_freq: pm_qos_constraints,
    max_freq_notifiers: blocking_notifier_head,
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
struct snd_pcm_hardware {
    info: u32,
    formats: u64,
    subformats: u32,
    rates: u32,
    rate_min: u32,
    rate_max: u32,
    channels_min: u32,
    channels_max: u32,
    buffer_bytes_max: u64,
    period_bytes_min: u64,
    period_bytes_max: u64,
    periods_min: u32,
    periods_max: u32,
    fifo_size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_pcm_ops {
    open: Option<unsafe extern "C" fn(*mut snd_pcm_substream) -> i32>,
    close: Option<unsafe extern "C" fn(*mut snd_pcm_substream) -> i32>,
    ioctl: Option<unsafe extern "C" fn(*mut snd_pcm_substream, u32, *mut ()) -> i32>,
    hw_params: Option<unsafe extern "C" fn(*mut snd_pcm_substream, *mut snd_pcm_hw_params) -> i32>,
    hw_free: Option<unsafe extern "C" fn(*mut snd_pcm_substream) -> i32>,
    prepare: Option<unsafe extern "C" fn(*mut snd_pcm_substream) -> i32>,
    trigger: Option<unsafe extern "C" fn(*mut snd_pcm_substream, i32) -> i32>,
    sync_stop: Option<unsafe extern "C" fn(*mut snd_pcm_substream) -> i32>,
    pointer: Option<unsafe extern "C" fn(*mut snd_pcm_substream) -> u64>,
    get_time_info: Option<unsafe extern "C" fn(*mut snd_pcm_substream, *mut timespec64, *mut timespec64, *mut snd_pcm_audio_tstamp_config, *mut snd_pcm_audio_tstamp_report) -> i32>,
    fill_silence: Option<unsafe extern "C" fn(*mut snd_pcm_substream, i32, u64, u64) -> i32>,
    copy: Option<unsafe extern "C" fn(*mut snd_pcm_substream, i32, u64, *mut iov_iter, u64) -> i32>,
    page: Option<unsafe extern "C" fn(*mut snd_pcm_substream, u64) -> *mut page>,
    mmap: Option<unsafe extern "C" fn(*mut snd_pcm_substream, *mut vm_area_struct) -> i32>,
    ack: Option<unsafe extern "C" fn(*mut snd_pcm_substream) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_pcm_hw_rule {
    cond: u32,
    var: i32,
    deps: [i32; 5],
    func: Option<unsafe extern "C" fn(*mut snd_pcm_hw_params, *mut snd_pcm_hw_rule) -> i32>,
    private: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_pcm_hw_constraints {
    masks: [snd_mask; 3],
    intervals: [snd_interval; 12],
    rules_num: u32,
    rules_all: u32,
    rules: *mut snd_pcm_hw_rule,
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct snd_pcm_audio_tstamp_config {
    __slate_bits_0: __slate_bits::__SlateBits794U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_pcm_audio_tstamp_report {
    __slate_bits_0: __slate_bits::__SlateBits795U0,
    accuracy: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_pcm_runtime {
    state: i32,
    suspended_state: i32,
    trigger_master: *mut snd_pcm_substream,
    trigger_tstamp: timespec64,
    trigger_tstamp_latched: bool,
    overrange: i32,
    avail_max: u64,
    hw_ptr_base: u64,
    hw_ptr_interrupt: u64,
    hw_ptr_jiffies: u64,
    hw_ptr_buffer_jiffies: u64,
    delay: i64,
    hw_ptr_wrap: u64,
    access: i32,
    format: i32,
    subformat: i32,
    rate: u32,
    channels: u32,
    period_size: u64,
    periods: u32,
    buffer_size: u64,
    min_align: u64,
    byte_align: u64,
    frame_bits: u32,
    sample_bits: u32,
    info: u32,
    rate_num: u32,
    rate_den: u32,
    __slate_bits_0: __slate_bits::__SlateBits800U0,
    tstamp_mode: i32,
    period_step: u32,
    start_threshold: u64,
    stop_threshold: u64,
    silence_threshold: u64,
    silence_size: u64,
    boundary: u64,
    silence_start: u64,
    silence_filled: u64,
    std_sync_id: bool,
    status: *mut snd_pcm_mmap_status,
    control: *mut snd_pcm_mmap_control,
    twake: u64,
    sleep: wait_queue_head,
    tsleep: wait_queue_head,
    fasync: *mut snd_fasync,
    stop_operating: bool,
    buffer_mutex: mutex,
    buffer_accessing: __SlateRecord_types_h_190,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_pcm_runtime)>,
    hw: snd_pcm_hardware,
    hw_constraints: snd_pcm_hw_constraints,
    timer_resolution: u32,
    tstamp_type: i32,
    dma_area: *mut u8,
    dma_addr: u64,
    dma_bytes: u64,
    dma_buffer_p: *mut snd_dma_buffer,
    __slate_bits_1: __slate_bits::__SlateBits800U1,
    audio_tstamp_config: snd_pcm_audio_tstamp_config,
    audio_tstamp_report: snd_pcm_audio_tstamp_report,
    driver_tstamp: timespec64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_pcm_group {
    lock: spinlock,
    mutex: mutex,
    substreams: list_head,
    refs: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_pcm_substream {
    pcm: *mut snd_pcm,
    pstr: *mut snd_pcm_str,
    private_data: *mut (),
    number: i32,
    name: [u8; 32],
    stream: i32,
    latency_pm_qos_req: pm_qos_request,
    buffer_bytes_max: u64,
    dma_buffer: snd_dma_buffer,
    dma_max: u64,
    ops: *const snd_pcm_ops,
    runtime: *mut snd_pcm_runtime,
    timer: *mut snd_timer,
    __slate_bits_0: __slate_bits::__SlateBits793U0,
    wait_time: i64,
    next: *mut snd_pcm_substream,
    link_list: list_head,
    self_group: snd_pcm_group,
    group: *mut snd_pcm_group,
    ref_count: i32,
    mmap_count: __SlateRecord_types_h_190,
    f_flags: u32,
    pcm_release: Option<unsafe extern "C" fn(*mut snd_pcm_substream)>,
    pid: *mut pid,
    proc_root: *mut snd_info_entry,
    __slate_bits_1: __slate_bits::__SlateBits793U1,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_pcm_str {
    stream: i32,
    pcm: *mut snd_pcm,
    substream_count: u32,
    substream_opened: u32,
    substream: *mut snd_pcm_substream,
    proc_root: *mut snd_info_entry,
    chmap_kctl: *mut snd_kcontrol,
    dev: *mut device,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_pcm {
    card: *mut snd_card,
    list: list_head,
    device: i32,
    info_flags: u32,
    dev_class: u16,
    dev_subclass: u16,
    id: [u8; 64],
    name: [u8; 80],
    streams: [snd_pcm_str; 2],
    open_mutex: mutex,
    open_wait: wait_queue_head,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_pcm)>,
    internal: bool,
    nonatomic: bool,
    no_device_suspend: bool,
}


// snd_pcm_stream_lock_irqsave - Lock the PCM stream
// @substream: PCM substream
// @flags: irq flags
//
// This locks the PCM stream like snd_pcm_stream_lock() but with the local
// IRQ (only when nonatomic is false).  In nonatomic case, this is identical
// as snd_pcm_stream_lock().
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_pcm_chmap_elem {
    channels: u8,
    map: [u8; 15],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_pcm_chmap {
    pcm: *mut snd_pcm,
    stream: i32,
    kctl: *mut snd_kcontrol,
    chmap: *const snd_pcm_chmap_elem,
    max_channels: u32,
    channel_mask: u32,
    private_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_hwdep_ops {
    llseek: Option<unsafe extern "C" fn(*mut snd_hwdep, *mut file, i64, i32) -> i64>,
    read: Option<unsafe extern "C" fn(*mut snd_hwdep, *mut u8, i64, *mut i64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut snd_hwdep, *const u8, i64, *mut i64) -> i64>,
    open: Option<unsafe extern "C" fn(*mut snd_hwdep, *mut file) -> i32>,
    release: Option<unsafe extern "C" fn(*mut snd_hwdep, *mut file) -> i32>,
    poll: Option<unsafe extern "C" fn(*mut snd_hwdep, *mut file, *mut poll_table_struct) -> u32>,
    ioctl: Option<unsafe extern "C" fn(*mut snd_hwdep, *mut file, u32, u64) -> i32>,
    ioctl_compat: Option<unsafe extern "C" fn(*mut snd_hwdep, *mut file, u32, u64) -> i32>,
    mmap: Option<unsafe extern "C" fn(*mut snd_hwdep, *mut file, *mut vm_area_struct) -> i32>,
    dsp_status: Option<unsafe extern "C" fn(*mut snd_hwdep, *mut snd_hwdep_dsp_status) -> i32>,
    dsp_load: Option<unsafe extern "C" fn(*mut snd_hwdep, *mut snd_hwdep_dsp_image) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_hwdep {
    card: *mut snd_card,
    list: list_head,
    device: i32,
    id: [u8; 32],
    name: [u8; 80],
    iface: i32,
    ops: snd_hwdep_ops,
    open_wait: wait_queue_head,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_hwdep)>,
    dev: *mut device,
    open_mutex: mutex,
    used: i32,
    dsp_loaded: u32,
    __slate_bits_0: __slate_bits::__SlateBits807U0,
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
    __slate_bits_0: __slate_bits::__SlateBits815U0,
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
    __slate_bits_0: __slate_bits::__SlateBits709U0,
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
    __slate_bits_1: __slate_bits::__SlateBits709U1,
    d3hot_delay: u32,
    d3cold_delay: u32,
    l1ss: u16,
    link_state: *mut pcie_link_state,
    __slate_bits_2: __slate_bits::__SlateBits709U2,
    error_state: u32,
    dev: device,
    cfg_size: i32,
    irq: u32,
    resource: [resource; 11],
    driver_exclusive_resource: resource,
    __slate_bits_3: __slate_bits::__SlateBits709U3,
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
    __slate_bits_4: __slate_bits::__SlateBits709U4,
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
    __slate_bits_0: __slate_bits::__SlateBits435U0,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_audio_component_ops {
    owner: *mut module,
    get_power: Option<unsafe extern "C" fn(*mut device) -> u64>,
    put_power: Option<unsafe extern "C" fn(*mut device, u64)>,
    codec_wake_override: Option<unsafe extern "C" fn(*mut device, bool)>,
    get_cdclk_freq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync_audio_rate: Option<unsafe extern "C" fn(*mut device, i32, i32, i32) -> i32>,
    get_eld: Option<unsafe extern "C" fn(*mut device, i32, i32, *mut bool, *mut u8, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_audio_component_audio_ops {
    audio_ptr: *mut (),
    pin_eld_notify: Option<unsafe extern "C" fn(*mut (), i32, i32)>,
    pin2port: Option<unsafe extern "C" fn(*mut (), i32) -> i32>,
    master_bind: Option<unsafe extern "C" fn(*mut device, *mut drm_audio_component) -> i32>,
    master_unbind: Option<unsafe extern "C" fn(*mut device, *mut drm_audio_component)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_audio_component {
    dev: *mut device,
    ops: *const drm_audio_component_ops,
    audio_ops: *const drm_audio_component_audio_ops,
    master_bind_complete: completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_array {
    used: u32,
    alloced: u32,
    elem_size: u32,
    alloc_align: u32,
    list: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdac_device {
    dev: device,
    r#type: i32,
    bus: *mut hdac_bus,
    addr: u32,
    list: list_head,
    afg: u16,
    mfg: u16,
    vendor_id: u32,
    subsystem_id: u32,
    revision_id: u32,
    afg_function_id: u32,
    mfg_function_id: u32,
    __slate_bits_0: __slate_bits::__SlateBits841U0,
    power_caps: u32,
    vendor_name: *const u8,
    chip_name: *const u8,
    exec_verb: Option<unsafe extern "C" fn(*mut hdac_device, u32, u32, *mut u32) -> i32>,
    num_nodes: u32,
    start_nid: u16,
    end_nid: u16,
    in_pm: __SlateRecord_types_h_190,
    widget_lock: mutex,
    widgets: *mut hdac_widget_tree,
    regmap: *mut regmap,
    regmap_lock: mutex,
    vendor_verbs: snd_array,
    __slate_bits_1: __slate_bits::__SlateBits841U1,
}


fn snd_hdac_read_parm(codec: *mut hdac_device, nid: u16, parm: i32) -> i32 {
let mut val: u32 = 0;
(if (unsafe { _snd_hdac_read_parm(codec, nid, parm, core::ptr::addr_of_mut!(val)) }) < 0 { 1i32.wrapping_neg() as u32 } else { val }) as i32
}

fn snd_hdac_enter_pm(codec: *mut hdac_device) {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*codec).in_pm) });
}

fn snd_hdac_leave_pm(codec: *mut hdac_device) {
atomic_dec(unsafe { core::ptr::addr_of_mut!((*codec).in_pm) });
}

fn snd_hdac_is_in_pm(codec: *mut hdac_device) -> bool {
atomic_read((unsafe { core::ptr::addr_of_mut!((*codec).in_pm) }) as *const __SlateRecord_types_h_190) != 0
}

fn snd_hdac_is_power_on(codec: *mut hdac_device) -> bool {
!pm_runtime_suspended(unsafe { core::ptr::addr_of_mut!((*codec).dev) })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hdac_driver {
    driver: device_driver,
    r#type: i32,
    id_table: *const hda_device_id,
    r#match: Option<unsafe extern "C" fn(*mut hdac_device, *const hdac_driver) -> i32>,
    unsol_event: Option<unsafe extern "C" fn(*mut hdac_device, u32)>,
    probe: Option<unsafe extern "C" fn(*mut hdac_device) -> i32>,
    remove: Option<unsafe extern "C" fn(*mut hdac_device) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut hdac_device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdac_bus_ops {
    command: Option<unsafe extern "C" fn(*mut hdac_bus, u32) -> i32>,
    get_response: Option<unsafe extern "C" fn(*mut hdac_bus, u32, *mut u32) -> i32>,
    link_power: Option<unsafe extern "C" fn(*mut hdac_device, bool)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdac_ext_bus_ops {
    hdev_attach: Option<unsafe extern "C" fn(*mut hdac_device) -> i32>,
    hdev_detach: Option<unsafe extern "C" fn(*mut hdac_device) -> i32>,
}


// limit by controller side
#[repr(C)]
#[derive(Clone, Copy)]
struct hdac_rb {
    buf: *mut u32,
    addr: u64,
    rp: u16,
    wp: u16,
    cmds: [i32; 8],
    res: [u32; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdac_bus {
    dev: *mut device,
    ops: *const hdac_bus_ops,
    ext_ops: *const hdac_ext_bus_ops,
    addr: u64,
    remap_addr: *mut (),
    irq: i32,
    ppcap: *mut (),
    spbcap: *mut (),
    mlcap: *mut (),
    gtscap: *mut (),
    drsmcap: *mut (),
    codec_list: list_head,
    num_codecs: u32,
    caddr_tbl: [*mut hdac_device; 16],
    unsol_queue: [u32; 128],
    unsol_rp: u32,
    unsol_wp: u32,
    unsol_work: work_struct,
    codec_mask: u64,
    codec_powered: u64,
    corb: hdac_rb,
    rirb: hdac_rb,
    last_cmd: [u32; 8],
    rirb_wq: wait_queue_head,
    rb: snd_dma_buffer,
    posbuf: snd_dma_buffer,
    dma_type: i32,
    stream_list: list_head,
    __slate_bits_0: __slate_bits::__SlateBits840U0,
    poll_count: i32,
    bdl_pos_adj: i32,
    dma_stop_delay: u32,
    reg_lock: spinlock,
    cmd_mutex: mutex,
    lock: mutex,
    audio_component: *mut drm_audio_component,
    display_power_status: i64,
    display_power_active: u64,
    num_streams: i32,
    idx: i32,
    hlink_list: list_head,
    cmd_dma_state: bool,
    sdo_limit: u32,
    addr_offset: u64,
}


fn snd_array_init(array: *mut snd_array, size: u32, align: u32) {
unsafe {
(*array).elem_size = size;
}
unsafe {
(*array).alloc_align = align;
}
}

fn snd_array_elem(array: *mut snd_array, idx: u32) -> *mut () {
(unsafe { ((*array).list as *const u8).wrapping_offset(idx.wrapping_mul((*array).elem_size) as isize) }) as *mut ()
}

// a helper macro to iterate for each snd_array element
// snd_hdac_regmap_encode_amp_stereo - encode a pseudo register for stereo AMPs
// @nid: widget NID
// @dir: direction (#HDA_INPUT, #HDA_OUTPUT)
// @idx: input index value
//
// Returns an encoded pseudo register.
// both bits set!
fn snd_hdac_regmap_write(codec: *mut hdac_device, nid: u16, verb: u32, val: u32) -> i32 {
let cmd: u32 = snd_hdac_regmap_encode_verb!(nid, verb);
unsafe { snd_hdac_regmap_write_raw(codec, cmd, val) }
}

fn snd_hdac_regmap_update(codec: *mut hdac_device, nid: u16, verb: u32, mask: u32, val: u32) -> i32 {
let cmd: u32 = snd_hdac_regmap_encode_verb!(nid, verb);
unsafe { snd_hdac_regmap_update_raw(codec, cmd, mask, val) }
}

fn snd_hdac_regmap_read(codec: *mut hdac_device, nid: u16, verb: u32, val: *mut u32) -> i32 {
let cmd: u32 = snd_hdac_regmap_encode_verb!(nid, verb);
unsafe { snd_hdac_regmap_read_raw(codec, cmd, val) }
}

fn snd_hdac_regmap_get_amp(codec: *mut hdac_device, nid: u16, ch: i32, dir: i32, idx: i32) -> i32 {
let cmd: u32 = snd_hdac_regmap_encode_amp!(nid, ch != 0, dir, idx);
let mut err: i32 = 0;
let mut val: i32 = 0;
err = unsafe { snd_hdac_regmap_read_raw(codec, cmd, core::ptr::addr_of_mut!(val) as *mut u32) };
if err < 0 { err } else { val }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hda_bus {
    core: hdac_bus,
    card: *mut snd_card,
    pci: *mut pci_dev,
    modelname: *const u8,
    prepare_mutex: mutex,
    pcm_dev_bits: [u64; 1],
    __slate_bits_0: __slate_bits::__SlateBits851U0,
    primary_dig_out_type: i32,
    mixer_assigned: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hda_codec_driver {
    core: hdac_driver,
    id: *const hda_device_id,
    ops: *const hda_codec_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hda_codec_ops {
    probe: Option<unsafe extern "C" fn(*mut hda_codec, *const hda_device_id) -> i32>,
    remove: Option<unsafe extern "C" fn(*mut hda_codec)>,
    build_controls: Option<unsafe extern "C" fn(*mut hda_codec) -> i32>,
    build_pcms: Option<unsafe extern "C" fn(*mut hda_codec) -> i32>,
    init: Option<unsafe extern "C" fn(*mut hda_codec) -> i32>,
    unsol_event: Option<unsafe extern "C" fn(*mut hda_codec, u32)>,
    set_power_state: Option<unsafe extern "C" fn(*mut hda_codec, u16, u32)>,
    suspend: Option<unsafe extern "C" fn(*mut hda_codec) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut hda_codec) -> i32>,
    check_power_status: Option<unsafe extern "C" fn(*mut hda_codec, u16) -> i32>,
    stream_pm: Option<unsafe extern "C" fn(*mut hda_codec, u16, bool)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hda_pcm_ops {
    open: Option<unsafe extern "C" fn(*mut hda_pcm_stream, *mut hda_codec, *mut snd_pcm_substream) -> i32>,
    close: Option<unsafe extern "C" fn(*mut hda_pcm_stream, *mut hda_codec, *mut snd_pcm_substream) -> i32>,
    prepare: Option<unsafe extern "C" fn(*mut hda_pcm_stream, *mut hda_codec, u32, u32, *mut snd_pcm_substream) -> i32>,
    cleanup: Option<unsafe extern "C" fn(*mut hda_pcm_stream, *mut hda_codec, *mut snd_pcm_substream) -> i32>,
    get_delay: Option<unsafe extern "C" fn(*mut hda_pcm_stream, *mut hda_codec, *mut snd_pcm_substream) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hda_pcm_stream {
    substreams: u32,
    channels_min: u32,
    channels_max: u32,
    nid: u16,
    rates: u32,
    formats: u64,
    subformats: u32,
    maxbps: u32,
    chmap: *const snd_pcm_chmap_elem,
    ops: hda_pcm_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hda_pcm {
    name: *mut u8,
    stream: [hda_pcm_stream; 2],
    pcm_type: u32,
    device: i32,
    pcm: *mut snd_pcm,
    own_chmap: bool,
    codec: *mut hda_codec,
    list: list_head,
    __slate_bits_0: __slate_bits::__SlateBits854U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hda_codec {
    core: hdac_device,
    bus: *mut hda_bus,
    card: *mut snd_card,
    addr: u32,
    probe_id: u32,
    preset: *const hda_device_id,
    modelname: *const u8,
    pcm_list_head: list_head,
    pcm_ref: snd_refcount,
    spec: *mut (),
    beep: *mut hda_beep,
    beep_mode: u32,
    beep_just_power_on: bool,
    wcaps: *mut u32,
    mixers: snd_array,
    nids: snd_array,
    conn_list: list_head,
    spdif_mutex: mutex,
    control_mutex: mutex,
    spdif_out: snd_array,
    spdif_in_enable: u32,
    follower_dig_outs: *const u16,
    init_pins: snd_array,
    driver_pins: snd_array,
    cvt_setups: snd_array,
    user_mutex: mutex,
    hwdep: *mut snd_hwdep,
    __slate_bits_0: __slate_bits::__SlateBits853U0,
    power_on_acct: u64,
    power_off_acct: u64,
    power_jiffies: u64,
    power_filter: Option<unsafe extern "C" fn(*mut hda_codec, u16, u32) -> u32>,
    proc_widget_hook: Option<unsafe extern "C" fn(*mut snd_info_buffer, *mut hda_codec, u16)>,
    jacktbl: snd_array,
    jackpoll_interval: u64,
    jackpoll_work: delayed_work,
    depop_delay: i32,
    fixup_id: i32,
    fixup_list: *const hda_fixup,
    fixup_name: *const u8,
    verbs: snd_array,
}


// snd_hda_codec_read/write optional flags
fn snd_hda_codec_read(codec: *mut hda_codec, nid: u16, flags: i32, verb: u32, parm: u32) -> u32 {
(unsafe { snd_hdac_codec_read(core::ptr::addr_of_mut!((*codec).core), nid, flags, verb, parm) }) as u32
}

fn snd_hda_codec_write(codec: *mut hda_codec, nid: u16, flags: i32, verb: u32, parm: u32) -> i32 {
unsafe { snd_hdac_codec_write(core::ptr::addr_of_mut!((*codec).core), nid, flags, verb, parm) }
}

fn snd_hda_codec_write_sync(codec: *mut hda_codec, nid: u16, flags: i32, verb: u32, parm: u32) -> i32 {
// use snd_hda_codec_read() for writing;
// the returned value is usually discarded
unsafe { snd_hdac_codec_read(core::ptr::addr_of_mut!((*codec).core), nid, flags, verb, parm) }
}

#[unsafe(no_mangle)]
extern "C" fn snd_hda_get_devices(codec: *mut hda_codec, nid: u16, dev_list: *mut u8, __UNIQUE_ID_y__530: u32) -> u32 {
let mut parm: u32 = snd_hda_get_num_devices(codec, nid);
let mut dev_len: u32 = 0;
let mut devices: u32 = 0;
if parm == 0 {
return 0;
}
// not multi-stream capable
let mut __v0: u32;
let __UNIQUE_ID_x__529: u32 = parm.wrapping_add(1);
loop {
if __types_ok!(__UNIQUE_ID_x__529, __UNIQUE_ID_y__530) == 0 {
unsafe { __compiletime_assert_531() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__529 < __UNIQUE_ID_y__530 { __UNIQUE_ID_x__529 } else { __UNIQUE_ID_y__530 };
dev_len = __v0;
while devices < dev_len {
if (unsafe { snd_hdac_read(core::ptr::addr_of_mut!((*codec).core), nid, 3894u32, devices, core::ptr::addr_of_mut!(parm)) }) != 0 {
break;
}
// error
for i in 0u32..8u32 {
unsafe {
*dev_list.wrapping_offset(devices as isize) = parm as u8;
}
parm >>= 4i32;
devices = devices.wrapping_add(1);
if devices >= dev_len {
break;
}
}
}
devices
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hda_verb {
    nid: u16,
    verb: u32,
    param: u32,
}


fn snd_hda_codec_write_cache(codec: *mut hda_codec, nid: u16, flags: i32, verb: u32, parm: u32) -> i32 {
snd_hdac_regmap_write(unsafe { core::ptr::addr_of_mut!((*codec).core) }, nid, verb, parm)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hda_pincfg {
    nid: u16,
    ctrl: u8,
    target: u8,
    cfg: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hda_spdif_out {
    nid: u16,
    status: u32,
    ctls: u16,
}


fn snd_hda_codec_pcm_get(pcm: *mut hda_pcm) {
snd_refcount_get(unsafe { core::ptr::addr_of_mut!((*(*pcm).codec).pcm_ref) });
}

fn snd_hda_codec_pcm_put(pcm: *mut hda_pcm) {
unsafe { snd_refcount_put(core::ptr::addr_of_mut!((*(*pcm).codec).pcm_ref)) };
}

fn hda_call_check_power_status(codec: *mut hda_codec, nid: u16) -> i32 {
let driver: *mut hda_codec_driver = hda_codec_to_driver!(codec);
if !unsafe { (*driver).ops }.is_null() && (unsafe { (*(*driver).ops).check_power_status }) != None {
return unsafe { unsafe { (*(*driver).ops).check_power_status }.unwrap()(codec, nid) };
}
0
}

fn hda_codec_need_resume(codec: *mut hda_codec) -> bool {
((unsafe { (*codec).__slate_bits_0.__get_relaxed_resume() }) as i32) == 0 && (unsafe { (*codec).jacktbl.used }) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __hda_power_obj {
    codec: *mut hda_codec,
    err: i32,
}


fn __snd_hda_power_up(codec: *mut hda_codec) -> __hda_power_obj {
let mut T: __hda_power_obj = {
let mut __t0: __hda_power_obj = unsafe { core::mem::zeroed() };
__t0.codec = codec;
__t0
};
T.err = snd_hda_power_up!(codec);
return T;
}

fn __snd_hda_power_up_pm(codec: *mut hda_codec) -> __hda_power_obj {
let mut T: __hda_power_obj = {
let mut __t0: __hda_power_obj = unsafe { core::mem::zeroed() };
__t0.codec = codec;
__t0
};
T.err = snd_hda_power_up_pm!(codec);
return T;
}

#[inline(always)]
fn class_snd_hda_power_destructor(p: *mut __hda_power_obj) {
let _T: __hda_power_obj = unsafe { *p };
snd_hda_power_down!(_T.codec);
}

#[inline(always)]
fn class_snd_hda_power_constructor(codec: *mut hda_codec) -> __hda_power_obj {
let t: __hda_power_obj = __snd_hda_power_up(codec);
return t;
}

#[inline(always)]
fn class_snd_hda_power_pm_destructor(p: *mut __hda_power_obj) {
let _T: __hda_power_obj = unsafe { *p };
snd_hda_power_down_pm!(_T.codec);
}

#[inline(always)]
fn class_snd_hda_power_pm_constructor(codec: *mut hda_codec) -> __hda_power_obj {
let t: __hda_power_obj = __snd_hda_power_up_pm(codec);
return t;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct snd_jack {
    kctl_list: list_head,
    card: *mut snd_card,
    id: *const u8,
    input_dev: *mut input_dev,
    input_dev_lock: mutex,
    registered: i32,
    r#type: i32,
    name: [u8; 100],
    key: [u32; 6],
    hw_status_cache: i32,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_jack)>,
}


#[unsafe(no_mangle)]
static mut snd_hda_pcm_type_name: __SlateAlign16<[*const u8; 4]> = __SlateAlign16([(b"Audio\0".as_ptr() as *mut u8) as *const u8, (b"SPDIF\0".as_ptr() as *mut u8) as *const u8, (b"HDMI\0".as_ptr() as *mut u8) as *const u8, (b"Modem\0".as_ptr() as *mut u8) as *const u8]);

#[repr(C)]
#[derive(Clone, Copy)]
struct hda_vmaster_mute_hook {
    sw_kctl: *mut snd_kcontrol,
    hook: Option<unsafe extern "C" fn(*mut (), i32)>,
    codec: *mut hda_codec,
}


// amp value bits
// input MUX helper
#[repr(C)]
#[derive(Clone, Copy)]
struct hda_input_mux_item {
    label: [u8; 32],
    index: u32,
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_sequence_write: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_get_conn_list: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_get_connections: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_override_conn_list: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_get_conn_index: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_get_num_devices: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_get_dev_select: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_set_dev_select: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_set_pincfg: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_get_pincfg: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_set_pin_target: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_get_pin_target: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_shutup_pins: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_pcm_new: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_cleanup_for_unbind: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_register: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_unregister: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_device_init: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_new: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_device_new: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_update_widgets: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_setup_stream: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___snd_hda_codec_cleanup_stream: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_query_amp_caps: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_check_amp_caps: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_override_amp_caps: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_amp_update: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_amp_stereo: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_amp_init: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {28} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_amp_init_stereo: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {29} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_mixer_amp_volume_info: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {30} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_mixer_amp_volume_get: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {31} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_mixer_amp_volume_put: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {32} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_mixer_amp_tlv: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {33} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_set_vmaster_tlv: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {34} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_find_mixer_ctl: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {35} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_ctl_add: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {36} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_lock_devices: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {37} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_unlock_devices: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {38} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___snd_hda_add_vmaster: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {39} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_add_vmaster_hook: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {40} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_sync_vmaster_hook: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {41} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_mixer_amp_switch_info: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {42} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_mixer_amp_switch_get: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {43} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_mixer_amp_switch_put: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {44} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_create_dig_out_ctls: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {45} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_spdif_out_of_nid: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {46} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_spdif_ctls_unassign: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {47} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_spdif_ctls_assign: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {48} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_create_spdif_share_sw: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {49} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_create_spdif_in_ctls: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {50} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_set_power_to_all: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {51} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_eapd_power_filter: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {52} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_pcm_2_1_chmaps: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {53} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_build_controls: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {54} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_prepare: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {55} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_cleanup: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {56} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_parse_pcms: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {57} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_add_new_ctls: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {58} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_set_power_save: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {59} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_set_power_save: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {60} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_check_amp_list_power: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {61} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_input_mux_info: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {62} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_input_mux_put: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {63} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_enum_helper_info: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {64} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_multi_out_dig_open: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {65} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_multi_out_dig_prepare: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {66} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_multi_out_dig_cleanup: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {67} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_multi_out_dig_close: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {68} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_multi_out_analog_open: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {69} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_multi_out_analog_prepare: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {70} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_multi_out_analog_cleanup: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {71} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_get_default_vref: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {72} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_correct_pin_ctl: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {73} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol__snd_hda_set_pin_ctl: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {74} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_add_imux_item: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {75} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_hda_codec_set_gpio: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {76} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_print_pcm_bits: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {77} ; .previous", sym snd_hda_sequence_write, sym snd_hda_get_conn_list, sym snd_hda_get_connections, sym snd_hda_override_conn_list, sym snd_hda_get_conn_index, sym snd_hda_get_num_devices, sym snd_hda_get_dev_select, sym snd_hda_set_dev_select, sym snd_hda_codec_set_pincfg, sym snd_hda_codec_get_pincfg, sym snd_hda_codec_set_pin_target, sym snd_hda_codec_get_pin_target, sym snd_hda_shutup_pins, sym snd_hda_codec_pcm_new, sym snd_hda_codec_cleanup_for_unbind, sym snd_hda_codec_register, sym snd_hda_codec_unregister, sym snd_hda_codec_device_init, sym snd_hda_codec_new, sym snd_hda_codec_device_new, sym snd_hda_codec_update_widgets, sym snd_hda_codec_setup_stream, sym __snd_hda_codec_cleanup_stream, sym query_amp_caps, sym snd_hda_check_amp_caps, sym snd_hda_override_amp_caps, sym snd_hda_codec_amp_update, sym snd_hda_codec_amp_stereo, sym snd_hda_codec_amp_init, sym snd_hda_codec_amp_init_stereo, sym snd_hda_mixer_amp_volume_info, sym snd_hda_mixer_amp_volume_get, sym snd_hda_mixer_amp_volume_put, sym snd_hda_mixer_amp_tlv, sym snd_hda_set_vmaster_tlv, sym snd_hda_find_mixer_ctl, sym snd_hda_ctl_add, sym snd_hda_lock_devices, sym snd_hda_unlock_devices, sym __snd_hda_add_vmaster, sym snd_hda_add_vmaster_hook, sym snd_hda_sync_vmaster_hook, sym snd_hda_mixer_amp_switch_info, sym snd_hda_mixer_amp_switch_get, sym snd_hda_mixer_amp_switch_put, sym snd_hda_create_dig_out_ctls, sym snd_hda_spdif_out_of_nid, sym snd_hda_spdif_ctls_unassign, sym snd_hda_spdif_ctls_assign, sym snd_hda_create_spdif_share_sw, sym snd_hda_create_spdif_in_ctls, sym snd_hda_codec_set_power_to_all, sym snd_hda_codec_eapd_power_filter, sym snd_pcm_2_1_chmaps, sym snd_hda_codec_build_controls, sym snd_hda_codec_prepare, sym snd_hda_codec_cleanup, sym snd_hda_codec_parse_pcms, sym snd_hda_add_new_ctls, sym snd_hda_codec_set_power_save, sym snd_hda_set_power_save, sym snd_hda_check_amp_list_power, sym snd_hda_input_mux_info, sym snd_hda_input_mux_put, sym snd_hda_enum_helper_info, sym snd_hda_multi_out_dig_open, sym snd_hda_multi_out_dig_prepare, sym snd_hda_multi_out_dig_cleanup, sym snd_hda_multi_out_dig_close, sym snd_hda_multi_out_analog_open, sym snd_hda_multi_out_analog_prepare, sym snd_hda_multi_out_analog_cleanup, sym snd_hda_get_default_vref, sym snd_hda_correct_pin_ctl, sym _snd_hda_set_pin_ctl, sym snd_hda_add_imux_item, sym snd_hda_codec_set_gpio, sym snd_print_pcm_bits, options(att_syntax));

#[repr(C)]
#[derive(Clone, Copy)]
struct hda_input_mux {
    num_items: u32,
    items: [hda_input_mux_item; 36],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hda_multi_out {
    num_dacs: i32,
    dac_nids: *const u16,
    hp_nid: u16,
    hp_out_nid: [u16; 5],
    extra_out_nid: [u16; 5],
    dig_out_nid: u16,
    follower_dig_outs: *const u16,
    max_channels: i32,
    dig_out_used: i32,
    no_share_stream: i32,
    share_spdif: i32,
    analog_rates: u32,
    analog_maxbps: u32,
    analog_formats: u64,
    spdif_rates: u32,
    spdif_maxbps: u32,
    spdif_formats: u64,
    share_spdif_kctl: *mut snd_kcontrol,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hda_pintbl {
    nid: u16,
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hda_fixup {
    r#type: i32,
    __slate_bits_0: __slate_bits::__SlateBits860U0,
    chain_id: i32,
    v: __SlateRecord_hda_local_h_289,
}


fn get_wcaps(codec: *mut hda_codec, nid: u16) -> u32 {
if (nid as i32) < ((unsafe { (*codec).core.start_nid }) as i32) || ((nid as i32) as u32) >= (((unsafe { (*codec).core.start_nid }) as i32) as u32).wrapping_add(unsafe { (*codec).core.num_nodes }) {
return 0;
}
unsafe { *unsafe { (*codec).wcaps }.wrapping_offset((nid as i32).wrapping_sub((*codec).core.start_nid as i32) as isize) }
}

fn get_wcaps_type(wcaps: u32) -> i32 {
if wcaps == 0 {
return 1i32.wrapping_neg();
}
// invalid type
((wcaps & 15728640u32) >> 20i32) as i32
}

fn snd_hda_override_wcaps(codec: *mut hda_codec, nid: u16, val: u32) {
if (nid as i32) >= ((unsafe { (*codec).core.start_nid }) as i32) && ((nid as i32) as u32) < (((unsafe { (*codec).core.start_nid }) as i32) as u32).wrapping_add(unsafe { (*codec).core.num_nodes }) {
unsafe {
*unsafe { (*codec).wcaps }.wrapping_offset((nid as i32).wrapping_sub((*codec).core.start_nid as i32) as isize) = val;
}
}
}

fn snd_hda_query_pin_caps(codec: *mut hda_codec, nid: u16) -> u32 {
snd_hda_param_read!(codec, nid, 12i32) as u32
}

// flags for hda_nid_item
#[repr(C)]
#[derive(Clone, Copy)]
struct hda_nid_item {
    kctl: *mut snd_kcontrol,
    index: u32,
    nid: u16,
    flags: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hda_amp_list {
    nid: u16,
    dir: u8,
    idx: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hda_loopback_check {
    amplist: *const hda_amp_list,
    power_on: i32,
}


fn snd_hda_check_power_state(codec: *mut hda_codec, nid: u16, target_state: u32) -> bool {
unsafe { snd_hdac_check_power_state(core::ptr::addr_of_mut!((*codec).core), nid, target_state) }
}

fn snd_hda_sync_power_state(codec: *mut hda_codec, nid: u16, target_state: u32) -> u32 {
unsafe { snd_hdac_sync_power_state(core::ptr::addr_of_mut!((*codec).core), nid, target_state) }
}

fn snd_hda_codec_init(codec: *mut hda_codec) -> i32 {
let driver: *mut hda_codec_driver = hda_codec_to_driver!(codec);
if (unsafe { (*(*driver).ops).init }) != None {
return unsafe { unsafe { (*(*driver).ops).init }.unwrap()(codec) };
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hda_beep {
    dev: *mut input_dev,
    codec: *mut hda_codec,
    phys: [u8; 32],
    tone: i32,
    nid: u16,
    __slate_bits_0: __slate_bits::__SlateBits852U0,
    beep_work: work_struct,
    power_hook: Option<unsafe extern "C" fn(*mut hda_beep, bool)>,
}


fn snd_hda_detach_beep_device(codec: *mut hda_codec) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hda_jack_callback {
    nid: u16,
    dev_id: i32,
    func: Option<unsafe extern "C" fn(*mut hda_codec, *mut hda_jack_callback)>,
    private_data: u32,
    unsol_res: u32,
    jack: *mut hda_jack_tbl,
    next: *mut hda_jack_callback,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hda_jack_tbl {
    nid: u16,
    dev_id: i32,
    tag: u8,
    callback: *mut hda_jack_callback,
    pin_sense: u32,
    __slate_bits_0: __slate_bits::__SlateBits880U0,
    gating_jack: u16,
    gated_jack: u16,
    key_report_jack: u16,
    r#type: i32,
    button_state: i32,
    jack: *mut snd_jack,
}


fn call_exec_verb(bus: *mut hda_bus, codec: *mut hda_codec, cmd: u32, flags: u32, res: *mut u32) -> i32 {
let mut err: i32 = 0;
let mut pm: __hda_power_obj = class_snd_hda_power_pm_constructor(codec);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(pm) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_snd_hda_power_pm_destructor(__slate_cleanup_ptr_0 as *mut _));
if pm.err < 0 && ((unsafe { (*bus).core.__slate_bits_0.__get_chip_init() }) as i32) == 0 {
while false {
}
unsafe { _dev_warn(hda_codec_dev!(codec) as *const device, c"Failed to send cmd 0x%x ret=[%d], hda control stopped\n".as_ptr() as *const u8, cmd, pm.err) };
return pm.err;
}
let mut __UNIQUE_ID_guard_521: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*bus).core.cmd_mutex) });
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_521) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_2 as *mut _));
let mut __UNIQUE_ID_unlock_522: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*bus).core.cmd_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_522) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _));
if flags & 1u32 != 0 {
unsafe {
(*bus).__slate_bits_0.__set_no_response_fallback(1u32);
}
}
err = unsafe { snd_hdac_bus_exec_verb_unlocked(core::ptr::addr_of_mut!((*bus).core), (*codec).core.addr, cmd, res) };
unsafe {
(*bus).__slate_bits_0.__set_no_response_fallback(0u32);
}
err
}

/// Send and receive a verb - passed to exec_verb override for hdac_device
#[unsafe(link_section = ".text.slate_distinct.codec.codec_exec_verb")]
extern "C" fn codec_exec_verb(dev: *mut hdac_device, cmd: u32, flags: u32, res: *mut u32) -> i32 {
// clear reset-flag when the communication gets recovered
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut bus: *mut hda_bus = unsafe { core::mem::zeroed() };
let mut codec: *mut hda_codec = unsafe { core::mem::zeroed() };
unsafe {
codec = container_of__2!(dev, hda_codec);
bus = (*codec).bus;
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if cmd == 4294967295u32 {
return 1i32.wrapping_neg();
} else {
loop {
*__slate_slot_1 = call_exec_verb(bus, codec, cmd, flags, res);
if !codec_in_pm!(codec) && !res.is_null() && *__slate_slot_1 == 11i32.wrapping_neg() {
if ((*bus).__slate_bits_0.__get_response_reset() as i32) != 0 {
snd_hda_bus_reset(bus);
}
} else {
break;
}
}
if *__slate_slot_1 == 0 {
__v0 = true;
} else {
__v0 = codec_in_pm!(codec);
}
if __v0 {
(*bus).__slate_bits_0.__set_response_reset(0u32);
}
return *__slate_slot_1;
}
}
}

/// snd_hda_sequence_write - sequence writes
/// @codec: the HDA codec
/// @seq: VERB array to send
///
/// Send the commands sequentially from the given array.
/// The array must be terminated with NID=0.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_sequence_write")]
extern "C" fn snd_hda_sequence_write(codec: *mut hda_codec, mut seq: *const hda_verb) {
while (unsafe { (*seq).nid }) != 0 {
snd_hda_codec_write(codec, unsafe { (*seq).nid }, 0i32, unsafe { (*seq).verb }, unsafe { (*seq).param });
let __v1: *const hda_verb = seq;
seq = __v1.wrapping_offset(1 as isize);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_sequence_write_523: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *const hda_verb)>, *mut ()>(Some(snd_hda_sequence_write)) };

/// connection list element
#[repr(C)]
#[derive(Clone, Copy)]
struct hda_conn_list {
    list: list_head,
    len: i32,
    nid: u16,
    conns: [u16; 0],
}


/// look up the cached results
fn lookup_conn_list(codec: *mut hda_codec, nid: u16) -> *mut hda_conn_list {
let mut p: *mut hda_conn_list = unsafe { core::mem::zeroed() };
p = list_first_entry__14!(unsafe { core::ptr::addr_of_mut!((*codec).conn_list) });
while list_entry_is_head!(p, unsafe { core::ptr::addr_of_mut!((*codec).conn_list) }, list) == 0 {
if ((unsafe { (*p).nid }) as i32) == (nid as i32) {
return p;
}
p = list_next_entry__20!(p, list);
}
core::ptr::null_mut::<hda_conn_list>()
}

fn add_conn_list(codec: *mut hda_codec, nid: u16, len: i32, list: *const u16) -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut p: *mut hda_conn_list = unsafe { core::mem::zeroed() };
let mut __v0: *mut hda_conn_list;
let __count: u64 = len as u64;
let __obj_size: u64 = struct_size_t!(__count);
let mut __obj_ptr: *mut hda_conn_list = unsafe { core::mem::zeroed() };
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
__obj_ptr = __v1 as *mut hda_conn_list;
if !__obj_ptr.is_null() {
__set_flex_counter__2!(__count, __obj_ptr);
}
__v0 = __obj_ptr;
p = __v0;
if p.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*p).len = len;
}
unsafe {
(*p).nid = nid;
}
unsafe { memcpy((core::ptr::addr_of_mut!((*p).conns) as *mut u16) as *mut (), list as *const (), (len as u64).wrapping_mul(core::mem::size_of::<u16>() as u64)) };
list_add(unsafe { core::ptr::addr_of_mut!((*p).list) }, unsafe { core::ptr::addr_of_mut!((*codec).conn_list) });
0
}

fn remove_conn_list(codec: *mut hda_codec) {
while list_empty((unsafe { core::ptr::addr_of_mut!((*codec).conn_list) }) as *const list_head) == 0 {
let mut p: *mut hda_conn_list = unsafe { core::mem::zeroed() };
p = list_first_entry__14!(unsafe { core::ptr::addr_of_mut!((*codec).conn_list) });
list_del(unsafe { core::ptr::addr_of_mut!((*p).list) });
unsafe { kfree(p as *const ()) };
}
}

/// read the connection and add to the cache
fn read_and_add_raw_conns(codec: *mut hda_codec, nid: u16) -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut list: __SlateAlign16<[u16; 32]> = __SlateAlign16([0u16; 32]);
let mut result: *mut u16 = list.0.as_mut_ptr() as *mut u16;
let mut len: i32 = snd_hda_get_raw_connections__2!(codec, nid, list.0.as_mut_ptr() as *mut u16, ARRAY_SIZE__49!());
if len == 28i32.wrapping_neg() {
len = snd_hda_get_num_raw_conns!(codec, nid);
let mut __v0: *mut u16;
let __obj_size: u64 = size_mul(core::mem::size_of::<u16>() as u64, len as u64);
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
__v0 = __v1 as *mut u16;
result = __v0;
if result.is_null() {
return 12i32.wrapping_neg();
}
len = snd_hda_get_raw_connections!(codec, nid, result, len);
}
if len >= 0 {
len = snd_hda_override_conn_list(codec, nid, len, result as *const u16);
}
if result != (list.0.as_mut_ptr() as *mut u16) {
unsafe { kfree(result as *const ()) };
}
len
}

/// snd_hda_get_conn_list - get connection list
/// @codec: the HDA codec
/// @nid: NID to parse
/// @listp: the pointer to store NID list
///
/// Parses the connection list of the given widget and stores the pointer
/// to the list of NIDs.
///
/// Returns the number of connections, or a negative error code.
///
/// Note that the returned pointer isn't protected against the list
/// modification.  If snd_hda_override_conn_list() might be called
/// concurrently, protect with a mutex appropriately.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_get_conn_list")]
extern "C" fn snd_hda_get_conn_list(codec: *mut hda_codec, nid: u16, listp: *mut *const u16) -> i32 {
let mut added: bool = false;
loop {
let mut err: i32 = 0;
let mut p: *const hda_conn_list = unsafe { core::mem::zeroed() };
// if the connection-list is already cached, read it
p = lookup_conn_list(codec, nid) as *const hda_conn_list;
if !p.is_null() {
if !listp.is_null() {
unsafe {
*listp = core::ptr::addr_of!((*p).conns) as *const u16;
}
}
return unsafe { (*p).len };
}
if snd_BUG_ON!(added) != 0 {
return 22i32.wrapping_neg();
}
err = read_and_add_raw_conns(codec, nid);
if err < 0 {
return err;
}
added = true;
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_get_conn_list_524: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, *mut *const u16) -> i32>, *mut ()>(Some(snd_hda_get_conn_list)) };

/// snd_hda_get_connections - copy connection list
/// @codec: the HDA codec
/// @nid: NID to parse
/// @conn_list: connection list array; when NULL, checks only the size
/// @max_conns: max. number of connections to store
///
/// Parses the connection list of the given widget and stores the list
/// of NIDs.
///
/// Returns the number of connections, or a negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_get_connections")]
extern "C" fn snd_hda_get_connections(codec: *mut hda_codec, nid: u16, conn_list: *mut u16, max_conns: i32) -> i32 {
let mut list: *const u16 = unsafe { core::mem::zeroed() };
let len: i32 = snd_hda_get_conn_list(codec, nid, core::ptr::addr_of_mut!(list));
if len > 0 && !conn_list.is_null() {
if len > max_conns {
while false {
}
unsafe { _dev_err(hda_codec_dev!(codec) as *const device, c"Too many connections %d for NID 0x%x\n".as_ptr() as *const u8, len, nid as i32) };
return 22i32.wrapping_neg();
}
unsafe { memcpy(conn_list as *mut (), list as *const (), (len as u64).wrapping_mul(core::mem::size_of::<u16>() as u64)) };
}
len
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_get_connections_525: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, *mut u16, i32) -> i32>, *mut ()>(Some(snd_hda_get_connections)) };

/// snd_hda_override_conn_list - add/modify the connection-list to cache
/// @codec: the HDA codec
/// @nid: NID to parse
/// @len: number of connection list entries
/// @list: the list of connection entries
///
/// Add or modify the given connection-list to the cache.  If the corresponding
/// cache already exists, invalidate it and append a new one.
///
/// Returns zero or a negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_override_conn_list")]
extern "C" fn snd_hda_override_conn_list(codec: *mut hda_codec, nid: u16, len: i32, list: *const u16) -> i32 {
let mut p: *mut hda_conn_list = unsafe { core::mem::zeroed() };
p = lookup_conn_list(codec, nid);
if !p.is_null() {
list_del(unsafe { core::ptr::addr_of_mut!((*p).list) });
unsafe { kfree(p as *const ()) };
}
add_conn_list(codec, nid, len, list)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_override_conn_list_526: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, i32, *const u16) -> i32>, *mut ()>(Some(snd_hda_override_conn_list)) };

/// snd_hda_get_conn_index - get the connection index of the given NID
/// @codec: the HDA codec
/// @mux: NID containing the list
/// @nid: NID to select
/// @recursive: 1 when searching NID recursively, otherwise 0
///
/// Parses the connection list of the widget @mux and checks whether the
/// widget @nid is present.  If it is, return the connection index.
/// Otherwise it returns -1.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_get_conn_index")]
extern "C" fn snd_hda_get_conn_index(codec: *mut hda_codec, mux: u16, nid: u16, mut recursive: i32) -> i32 {
let mut conn: *const u16 = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let nums: i32 = snd_hda_get_conn_list(codec, mux, core::ptr::addr_of_mut!(conn));
while i < nums {
if ((unsafe { *conn.wrapping_offset(i as isize) }) as i32) == (nid as i32) {
return i;
}
i = i.wrapping_add(1);
}
if recursive == 0 {
return 1i32.wrapping_neg();
}
if recursive > 10 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, hda_codec_dev!(codec) as *const device, c"too deep connection for 0x%x\n".as_ptr() as *const u8, nid as i32) };
}
return 1i32.wrapping_neg();
}
recursive = recursive.wrapping_add(1);
i = 0;
while i < nums {
let r#type: u32 = get_wcaps_type(get_wcaps(codec, unsafe { *conn.wrapping_offset(i as isize) })) as u32;
if !(r#type == 4 || r#type == 0) {
if snd_hda_get_conn_index(codec, unsafe { *conn.wrapping_offset(i as isize) }, nid, recursive) >= 0 {
return i;
}
}
i = i.wrapping_add(1);
}
1i32.wrapping_neg()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_get_conn_index_527: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, u16, i32) -> i32>, *mut ()>(Some(snd_hda_get_conn_index)) };

/// snd_hda_get_num_devices - get DEVLIST_LEN parameter of the given widget
///  @codec: the HDA codec
///  @nid: NID of the pin to parse
///
/// Get the device entry number on the given widget. This is a feature of
/// DP MST audio. Each pin can have several device entries in it.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_get_num_devices")]
extern "C" fn snd_hda_get_num_devices(codec: *mut hda_codec, nid: u16) -> u32 {
let wcaps: u32 = get_wcaps(codec, nid);
let mut parm: i32 = 0;
let __v0: bool = ((unsafe { (*codec).__slate_bits_0.__get_dp_mst() }) as i32) == 0 || wcaps & 512u32 == 0 || get_wcaps_type(wcaps) != 4;
if __v0 {
return 0;
}
parm = unsafe { snd_hdac_read_parm_uncached(core::ptr::addr_of_mut!((*codec).core), nid, 21i32) };
if parm == 1i32.wrapping_neg() {
parm = 0;
}
(parm & 63) as u32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_get_num_devices_528: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16) -> u32>, *mut ()>(Some(snd_hda_get_num_devices)) };

// snd_hda_get_devices - copy device list without cache
// @codec: the HDA codec
// @nid: NID of the pin to parse
// @dev_list: device list array
// @max_devices: max. number of devices to store
//
// Copy the device list. This info is dynamic and so not cached.
// Currently called only from hda_proc.c, so not exported.
/// snd_hda_get_dev_select - get device entry select on the pin
/// @codec: the HDA codec
/// @nid: NID of the pin to get device entry select
///
/// Get the devcie entry select on the pin. Return the device entry
/// id selected on the pin. Return 0 means the first device entry
/// is selected or MST is not supported.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_get_dev_select")]
extern "C" fn snd_hda_get_dev_select(codec: *mut hda_codec, nid: u16) -> i32 {
// not support dp_mst will always return 0, using first dev_entry
if ((unsafe { (*codec).__slate_bits_0.__get_dp_mst() }) as i32) == 0 {
return 0;
}
snd_hda_codec_read(codec, nid, 0i32, 3893u32, 0u32) as i32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_get_dev_select_532: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16) -> i32>, *mut ()>(Some(snd_hda_get_dev_select)) };

/// snd_hda_set_dev_select - set device entry select on the pin
/// @codec: the HDA codec
/// @nid: NID of the pin to set device entry select
/// @dev_id: device entry id to be set
///
/// Set the device entry select on the pin nid.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_set_dev_select")]
extern "C" fn snd_hda_set_dev_select(codec: *mut hda_codec, nid: u16, dev_id: i32) -> i32 {
let mut ret: i32 = 0;
let mut num_devices: i32 = 0;
// not support dp_mst will always return 0, using first dev_entry
if ((unsafe { (*codec).__slate_bits_0.__get_dp_mst() }) as i32) == 0 {
return 0;
}
// AC_PAR_DEVLIST_LEN is 0 based.
num_devices = snd_hda_get_num_devices(codec, nid).wrapping_add(1) as i32;
// If Device List Length is 0 (num_device = 1),
// the pin is not multi stream capable.
// Do nothing in this case.
if num_devices == 1 {
return 0;
}
// Behavior of setting index being equal to or greater than
// Device List Length is not predictable
if num_devices <= dev_id {
return 22i32.wrapping_neg();
}
ret = snd_hda_codec_write(codec, nid, 0i32, 1845u32, dev_id as u32);
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_set_dev_select_533: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, i32) -> i32>, *mut ()>(Some(snd_hda_set_dev_select)) };

/// read widget caps for each widget and store in cache
fn read_widget_caps(codec: *mut hda_codec, fg_node: u16) -> i32 {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut nid: u16 = 0;
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_array_noprof((unsafe { (*codec).core.num_nodes }) as u64, 4u64, 3264u32, unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_array_noprof((unsafe { (*codec).core.num_nodes }) as u64, 4u64, 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
__v0 = __v1;
unsafe {
(*codec).wcaps = __v0 as *mut u32;
}
if unsafe { (*codec).wcaps }.is_null() {
return 12i32.wrapping_neg();
}
nid = unsafe { (*codec).core.start_nid };
while (i as u32) < unsafe { (*codec).core.num_nodes } {
unsafe {
*unsafe { (*codec).wcaps }.wrapping_offset(i as isize) = snd_hdac_read_parm_uncached(core::ptr::addr_of_mut!((*codec).core), nid, 9i32) as u32;
}
i = i.wrapping_add(1);
nid = (nid as i32).wrapping_add(1) as u16;
}
0
}

/// read all pin default configurations and save codec->init_pins
fn read_pin_defaults(codec: *mut hda_codec) -> i32 {
let mut nid: u16 = unsafe { (*codec).core.start_nid };
while (nid as i32) < ((unsafe { (*codec).core.end_nid }) as i32) {
let wcaps: u32 = get_wcaps(codec, nid);
let wid_type: u32 = get_wcaps_type(wcaps) as u32;
if wid_type == 4 {
let pin: *mut hda_pincfg = (unsafe { snd_array_new(core::ptr::addr_of_mut!((*codec).init_pins)) }) as *mut hda_pincfg;
if pin.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*pin).nid = nid;
}
unsafe {
(*pin).cfg = snd_hda_codec_read(codec, nid, 0i32, 3868u32, 0u32);
}
// all device entries are the same widget control so far
// fixme: if any codec is different, need fix here
unsafe {
(*pin).ctrl = snd_hda_codec_read(codec, nid, 0i32, 3847u32, 0u32) as u8;
}
}
nid = (nid as i32).wrapping_add(1) as u16;
}
0
}

/// look up the given pin config list and return the item matching with NID
fn look_up_pincfg(codec: *mut hda_codec, array: *mut snd_array, nid: u16) -> *mut hda_pincfg {
let mut pin: *mut hda_pincfg = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
pin = (unsafe { (*array).list }) as *mut hda_pincfg;
while (i as u32) < unsafe { (*array).used } {
if ((unsafe { (*pin).nid }) as i32) == (nid as i32) {
return pin;
}
let __v3: i32 = i.wrapping_add(1);
i = __v3;
pin = snd_array_elem(array, __v3 as u32) as *mut hda_pincfg;
}
core::ptr::null_mut::<hda_pincfg>()
}

/// set the current pin config value for the given NID.
/// the value is cached, and read via snd_hda_codec_get_pincfg()
#[unsafe(no_mangle)]
extern "C" fn snd_hda_add_pincfg(codec: *mut hda_codec, list: *mut snd_array, nid: u16, cfg: u32) -> i32 {
let mut pin: *mut hda_pincfg = unsafe { core::mem::zeroed() };
pin = look_up_pincfg(codec, list, nid);
if pin.is_null() {
pin = (unsafe { snd_array_new(list) }) as *mut hda_pincfg;
if pin.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*pin).nid = nid;
}
}
unsafe {
(*pin).cfg = cfg;
}
0
}

/// snd_hda_codec_set_pincfg - Override a pin default configuration
/// @codec: the HDA codec
/// @nid: NID to set the pin config
/// @cfg: the pin default config value
///
/// Override a pin default configuration value in the cache.
/// This value can be read by snd_hda_codec_get_pincfg() in a higher
/// priority than the real hardware value.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_set_pincfg")]
extern "C" fn snd_hda_codec_set_pincfg(codec: *mut hda_codec, nid: u16, cfg: u32) -> i32 {
snd_hda_add_pincfg(codec, unsafe { core::ptr::addr_of_mut!((*codec).driver_pins) }, nid, cfg)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_set_pincfg_534: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, u32) -> i32>, *mut ()>(Some(snd_hda_codec_set_pincfg)) };

/// snd_hda_codec_get_pincfg - Obtain a pin-default configuration
/// @codec: the HDA codec
/// @nid: NID to get the pin config
///
/// Get the current pin config value of the given pin NID.
/// If the pincfg value is cached or overridden via sysfs or driver,
/// returns the cached value.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_get_pincfg")]
extern "C" fn snd_hda_codec_get_pincfg(codec: *mut hda_codec, nid: u16) -> u32 {
let mut pin: *mut hda_pincfg = unsafe { core::mem::zeroed() };
pin = look_up_pincfg(codec, unsafe { core::ptr::addr_of_mut!((*codec).driver_pins) }, nid);
if !pin.is_null() {
return unsafe { (*pin).cfg };
}
pin = look_up_pincfg(codec, unsafe { core::ptr::addr_of_mut!((*codec).init_pins) }, nid);
if !pin.is_null() {
return unsafe { (*pin).cfg };
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_get_pincfg_535: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16) -> u32>, *mut ()>(Some(snd_hda_codec_get_pincfg)) };

/// snd_hda_codec_set_pin_target - remember the current pinctl target value
/// @codec: the HDA codec
/// @nid: pin NID
/// @val: assigned pinctl value
///
/// This function stores the given value to a pinctl target value in the
/// pincfg table.  This isn't always as same as the actually written value
/// but can be referred at any time via snd_hda_codec_get_pin_target().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_set_pin_target")]
extern "C" fn snd_hda_codec_set_pin_target(codec: *mut hda_codec, nid: u16, val: u32) -> i32 {
let pin: *mut hda_pincfg = look_up_pincfg(codec, unsafe { core::ptr::addr_of_mut!((*codec).init_pins) }, nid);
if pin.is_null() {
return 22i32.wrapping_neg();
}
unsafe {
(*pin).target = val as u8;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_set_pin_target_536: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, u32) -> i32>, *mut ()>(Some(snd_hda_codec_set_pin_target)) };

/// snd_hda_codec_get_pin_target - return the current pinctl target value
/// @codec: the HDA codec
/// @nid: pin NID
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_get_pin_target")]
extern "C" fn snd_hda_codec_get_pin_target(codec: *mut hda_codec, nid: u16) -> i32 {
let pin: *mut hda_pincfg = look_up_pincfg(codec, unsafe { core::ptr::addr_of_mut!((*codec).init_pins) }, nid);
if pin.is_null() {
return 0;
}
(unsafe { (*pin).target }) as i32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_get_pin_target_537: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16) -> i32>, *mut ()>(Some(snd_hda_codec_get_pin_target)) };

/// snd_hda_shutup_pins - Shut up all pins
/// @codec: the HDA codec
///
/// Clear all pin controls to shup up before suspend for avoiding click noise.
/// The controls aren't cached so that they can be resumed properly.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_shutup_pins")]
extern "C" fn snd_hda_shutup_pins(codec: *mut hda_codec) {
let mut pin: *const hda_pincfg = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
// don't shut up pins when unloading the driver; otherwise it breaks
// the default pin setup at the next load of the driver
if ((unsafe { (*(*codec).bus).__slate_bits_0.__get_shutdown() }) as i32) != 0 {
return;
}
let __v0: *const hda_pincfg = (unsafe { (*core::ptr::addr_of_mut!((*codec).init_pins)).list }) as *const hda_pincfg;
pin = __v0;
while (i as u32) < unsafe { (*core::ptr::addr_of_mut!((*codec).init_pins)).used } {
snd_hda_codec_write_sync(codec, unsafe { (*pin).nid }, 0i32, 1799u32, 0u32);
let __v3: i32 = i.wrapping_add(1);
i = __v3;
pin = snd_array_elem(unsafe { core::ptr::addr_of_mut!((*codec).init_pins) }, __v3 as u32) as *const hda_pincfg;
}
unsafe {
(*codec).__slate_bits_0.__set_pins_shutup(1u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_shutup_pins_538: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec)>, *mut ()>(Some(snd_hda_shutup_pins)) };

/// Restore the pin controls cleared previously via snd_hda_shutup_pins()
fn restore_shutup_pins(codec: *mut hda_codec) {
let mut pin: *const hda_pincfg = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
if ((unsafe { (*codec).__slate_bits_0.__get_pins_shutup() }) as i32) == 0 {
return;
}
if ((unsafe { (*(*codec).bus).__slate_bits_0.__get_shutdown() }) as i32) != 0 {
return;
}
let __v0: *const hda_pincfg = (unsafe { (*core::ptr::addr_of_mut!((*codec).init_pins)).list }) as *const hda_pincfg;
pin = __v0;
while (i as u32) < unsafe { (*core::ptr::addr_of_mut!((*codec).init_pins)).used } {
snd_hda_codec_write(codec, unsafe { (*pin).nid }, 0i32, 1799u32, (unsafe { (*pin).ctrl }) as u32);
let __v3: i32 = i.wrapping_add(1);
i = __v3;
pin = snd_array_elem(unsafe { core::ptr::addr_of_mut!((*codec).init_pins) }, __v3 as u32) as *const hda_pincfg;
}
unsafe {
(*codec).__slate_bits_0.__set_pins_shutup(0u32);
}
}

#[unsafe(link_section = ".text.slate_distinct.codec.hda_jackpoll_work")]
extern "C" fn hda_jackpoll_work(work: *mut work_struct) {
let mut codec: *mut hda_codec = unsafe { core::mem::zeroed() };
let mut __v0: *mut hda_codec;
__v0 = ((unsafe { (work as *const u8).wrapping_offset(-(offsetof__34!(hda_codec) as isize)) }) as *mut ()) as *mut hda_codec;
codec = __v0;
if (unsafe { (*codec).jackpoll_interval }) == 0 {
return;
}
// the power-up/down sequence triggers the runtime resume
let mut pm: __hda_power_obj = class_snd_hda_power_constructor(codec);
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(pm) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_snd_hda_power_destructor(__slate_cleanup_ptr_1 as *mut _));
// update jacks manually if polling is required, too
unsafe { snd_hda_jack_set_dirty_all(codec) };
unsafe { snd_hda_jack_poll_all(codec) };
schedule_delayed_work(unsafe { core::ptr::addr_of_mut!((*codec).jackpoll_work) }, unsafe { (*codec).jackpoll_interval });
}

/// release all pincfg lists
fn free_init_pincfgs(codec: *mut hda_codec) {
unsafe { snd_array_free(core::ptr::addr_of_mut!((*codec).driver_pins)) };
unsafe { snd_array_free(core::ptr::addr_of_mut!((*codec).init_pins)) };
}

/// audio-converter setup caches
#[repr(C)]
#[derive(Clone, Copy)]
struct hda_cvt_setup {
    nid: u16,
    stream_tag: u8,
    channel_id: u8,
    format_id: u16,
    /// cvt is currently used
    active: u8,
    /// setups should be cleared
    dirty: u8,
}


/// get or create a cache entry for the given audio converter NID
fn get_hda_cvt_setup(codec: *mut hda_codec, nid: u16) -> *mut hda_cvt_setup {
let mut p: *mut hda_cvt_setup = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let __v0: *mut hda_cvt_setup = (unsafe { (*core::ptr::addr_of_mut!((*codec).cvt_setups)).list }) as *mut hda_cvt_setup;
p = __v0;
while (i as u32) < unsafe { (*core::ptr::addr_of_mut!((*codec).cvt_setups)).used } {
if ((unsafe { (*p).nid }) as i32) == (nid as i32) {
return p;
}
let __v3: i32 = i.wrapping_add(1);
i = __v3;
p = snd_array_elem(unsafe { core::ptr::addr_of_mut!((*codec).cvt_setups) }, __v3 as u32) as *mut hda_cvt_setup;
}
p = (unsafe { snd_array_new(core::ptr::addr_of_mut!((*codec).cvt_setups)) }) as *mut hda_cvt_setup;
if !p.is_null() {
unsafe {
(*p).nid = nid;
}
}
p
}

/// PCM device
#[unsafe(no_mangle)]
unsafe extern "C" fn snd_hda_codec_pcm_new(codec: *mut hda_codec, fmt: *const u8, mut __va_args: ...) -> *mut hda_pcm {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut pcm: *mut hda_pcm = unsafe { core::mem::zeroed() };
let mut args: core::ffi::VaList<'static> = unsafe { core::mem::zeroed() };
let mut __v0: *mut hda_pcm;
let __obj_size: u64 = size_mul(core::mem::size_of::<hda_pcm>() as u64, 1u64);
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
__v0 = __v1 as *mut hda_pcm;
pcm = __v0;
if pcm.is_null() {
return core::ptr::null_mut::<hda_pcm>();
}
unsafe {
(*pcm).codec = codec;
}
args = unsafe { core::mem::transmute::<core::ffi::VaList<'_>, core::ffi::VaList<'static>>(__va_args.clone()) };
unsafe {
(*pcm).name = kvasprintf(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32), fmt, args.clone());
}
va_end!();
if unsafe { (*pcm).name }.is_null() {
unsafe { kfree(pcm as *const ()) };
return core::ptr::null_mut::<hda_pcm>();
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*pcm).list) }, unsafe { core::ptr::addr_of_mut!((*codec).pcm_list_head) });
snd_hda_codec_pcm_get(pcm);
pcm
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_pcm_new_539: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *const u8, ...) -> *mut hda_pcm>, *mut ()>(Some(snd_hda_codec_pcm_new)) };

/// codec destructor
#[unsafe(no_mangle)]
extern "C" fn snd_hda_codec_disconnect_pcms(codec: *mut hda_codec) {
let mut pcm: *mut hda_pcm = unsafe { core::mem::zeroed() };
pcm = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*codec).pcm_list_head) });
while list_entry_is_head!(pcm, unsafe { core::ptr::addr_of_mut!((*codec).pcm_list_head) }, list) == 0 {
if ((unsafe { (*pcm).__slate_bits_0.__get_disconnected() }) as i32) == 0 {
if !unsafe { (*pcm).pcm }.is_null() {
unsafe { snd_device_disconnect((*codec).card, (*pcm).pcm as *mut ()) };
}
snd_hda_codec_pcm_put(pcm);
unsafe {
(*pcm).__slate_bits_0.__set_disconnected(1u32);
}
}
pcm = list_next_entry__4!(pcm, list);
}
}

fn codec_release_pcms(codec: *mut hda_codec) {
let mut pcm: *mut hda_pcm = unsafe { core::mem::zeroed() };
let mut n: *mut hda_pcm = unsafe { core::mem::zeroed() };
pcm = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*codec).pcm_list_head) });
n = list_next_entry__4!(pcm, list);
while list_entry_is_head!(pcm, unsafe { core::ptr::addr_of_mut!((*codec).pcm_list_head) }, list) == 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*pcm).list) });
if !unsafe { (*pcm).pcm }.is_null() {
unsafe { snd_device_free((*(*pcm).codec).card, (*pcm).pcm as *mut ()) };
}
clear_bit((unsafe { (*pcm).device }) as i64, (unsafe { (*(*(*pcm).codec).bus).pcm_dev_bits.as_mut_ptr() as *mut u64 }) as *mut u64);
unsafe { kfree((*pcm).name as *const ()) };
unsafe { kfree(pcm as *const ()) };
pcm = n;
n = list_next_entry__4!(n, list);
}
}

/// snd_hda_codec_cleanup_for_unbind - Prepare codec for removal
/// @codec: codec device to cleanup
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_cleanup_for_unbind")]
extern "C" fn snd_hda_codec_cleanup_for_unbind(codec: *mut hda_codec) {
if ((unsafe { (*codec).core.__slate_bits_1.__get_registered() }) as i32) != 0 {
// pm_runtime_put() is called in snd_hdac_device_exit()
pm_runtime_get_noresume(hda_codec_dev!(codec));
pm_runtime_disable(hda_codec_dev!(codec));
unsafe {
(*codec).core.__slate_bits_1.__set_registered(0u32);
}
}
snd_hda_codec_disconnect_pcms(codec);
unsafe { cancel_delayed_work_sync(core::ptr::addr_of_mut!((*codec).jackpoll_work)) };
if ((unsafe { (*codec).__slate_bits_0.__get_in_freeing() }) as i32) == 0 {
snd_hda_ctls_clear(codec);
}
codec_release_pcms(codec);
snd_hda_detach_beep_device(codec);
unsafe { snd_hda_jack_tbl_clear(codec) };
unsafe {
(*codec).proc_widget_hook = None;
}
unsafe {
(*codec).spec = core::ptr::null_mut::<()>();
}
// free only driver_pins so that init_pins + user_pins are restored
unsafe { snd_array_free(core::ptr::addr_of_mut!((*codec).driver_pins)) };
unsafe { snd_array_free(core::ptr::addr_of_mut!((*codec).cvt_setups)) };
unsafe { snd_array_free(core::ptr::addr_of_mut!((*codec).spdif_out)) };
unsafe { snd_array_free(core::ptr::addr_of_mut!((*codec).verbs)) };
unsafe {
(*codec).follower_dig_outs = core::ptr::null::<u16>();
}
unsafe {
(*codec).__slate_bits_0.__set_spdif_status_reset(0u32);
}
unsafe { snd_array_free(core::ptr::addr_of_mut!((*codec).mixers)) };
unsafe { snd_array_free(core::ptr::addr_of_mut!((*codec).nids)) };
remove_conn_list(codec);
unsafe { snd_hdac_regmap_exit(core::ptr::addr_of_mut!((*codec).core)) };
unsafe {
(*codec).__slate_bits_0.__set_configured(0u32);
}
unsafe { snd_refcount_init(core::ptr::addr_of_mut!((*codec).pcm_ref)) }; // reset refcount
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_cleanup_for_unbind_540: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec)>, *mut ()>(Some(snd_hda_codec_cleanup_for_unbind)) };

/// enable/disable display power per codec
#[unsafe(no_mangle)]
extern "C" fn snd_hda_codec_display_power(codec: *mut hda_codec, enable: bool) {
if ((unsafe { (*codec).__slate_bits_0.__get_display_power_control() }) as i32) != 0 {
unsafe { snd_hdac_display_power(core::ptr::addr_of_mut!((*(*codec).bus).core), (*codec).addr, enable) };
}
}

/// snd_hda_codec_register - Finalize codec initialization
/// @codec: codec device to register
///
/// Also called from hda_bind.c
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_register")]
extern "C" fn snd_hda_codec_register(codec: *mut hda_codec) {
if ((unsafe { (*codec).core.__slate_bits_1.__get_registered() }) as i32) != 0 {
return;
}
if device_is_registered(hda_codec_dev!(codec)) != 0 {
snd_hda_codec_display_power(codec, true);
unsafe { pm_runtime_enable(hda_codec_dev!(codec)) };
// it was powered up in snd_hda_codec_new(), now all done
snd_hda_power_down!(codec);
unsafe {
(*codec).core.__slate_bits_1.__set_registered(1u32);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_register_541: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec)>, *mut ()>(Some(snd_hda_codec_register)) };

#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_dev_register")]
extern "C" fn snd_hda_codec_dev_register(device: *mut snd_device) -> i32 {
snd_hda_codec_register((unsafe { (*device).device_data }) as *mut hda_codec);
0
}

/// snd_hda_codec_unregister - Unregister specified codec device
/// @codec: codec device to unregister
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_unregister")]
extern "C" fn snd_hda_codec_unregister(codec: *mut hda_codec) {
unsafe {
(*codec).__slate_bits_0.__set_in_freeing(1u32);
}
// snd_hda_codec_device_new() is used by legacy HDA and ASoC driver.
// We can't unregister ASoC device since it will be unregistered in
// snd_hdac_ext_bus_device_remove().
if (unsafe { (*codec).core.r#type }) == 1 {
unsafe { snd_hdac_device_unregister(core::ptr::addr_of_mut!((*codec).core)) };
}
snd_hda_codec_display_power(codec, false);
// In the case of ASoC HD-audio bus, the device refcount is released in
// snd_hdac_ext_bus_device_remove() explicitly.
if (unsafe { (*codec).core.r#type }) == 1 {
unsafe { put_device(hda_codec_dev!(codec)) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_unregister_542: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec)>, *mut ()>(Some(snd_hda_codec_unregister)) };

#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_dev_free")]
extern "C" fn snd_hda_codec_dev_free(device: *mut snd_device) -> i32 {
snd_hda_codec_unregister((unsafe { (*device).device_data }) as *mut hda_codec);
0
}

#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_dev_release")]
extern "C" fn snd_hda_codec_dev_release(dev: *mut device) {
let codec: *mut hda_codec = dev_to_hda_codec!(dev);
free_init_pincfgs(codec);
unsafe { snd_hdac_device_exit(core::ptr::addr_of_mut!((*codec).core)) };
unsafe { snd_hda_sysfs_clear(codec) };
unsafe { kfree((*codec).modelname as *const ()) };
unsafe { kfree((*codec).wcaps as *const ()) };
unsafe { kfree(codec as *const ()) };
}

/// snd_hda_codec_device_init - allocate HDA codec device
/// @bus: codec's parent bus
/// @codec_addr: the codec address on the parent bus
/// @fmt: format string for the device's name
///
/// Returns newly allocated codec device or ERR_PTR() on failure.
#[unsafe(no_mangle)]
unsafe extern "C" fn snd_hda_codec_device_init(bus: *mut hda_bus, codec_addr: u32, fmt: *const u8, mut __va_args: ...) -> *mut hda_codec {
let mut __slate_compound_7: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut vargs: core::ffi::VaList<'static> = unsafe { core::mem::zeroed() };
let mut name: __SlateAlign16<[u8; 31]> = __SlateAlign16([0u8; 31]);
let mut codec: *mut hda_codec = unsafe { core::mem::zeroed() };
let mut err: i32 = 0;
if snd_BUG_ON!(bus.is_null()) != 0 {
return ERR_PTR(22i32.wrapping_neg() as i64) as *mut hda_codec;
}
if snd_BUG_ON!(codec_addr > 15) != 0 {
return ERR_PTR(22i32.wrapping_neg() as i64) as *mut hda_codec;
}
let mut __v4: *mut hda_codec;
let __obj_size: u64 = size_mul(core::mem::size_of::<hda_codec>() as u64, 1u64);
let mut __v5: *mut ();
let mut __v6: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_7 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_7)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
} });
}
__v6 = _res;
__v5 = __v6;
__v4 = __v5 as *mut hda_codec;
codec = __v4;
if codec.is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut hda_codec;
}
vargs = unsafe { core::mem::transmute::<core::ffi::VaList<'_>, core::ffi::VaList<'static>>(__va_args.clone()) };
unsafe { vsprintf(name.0.as_mut_ptr() as *mut u8, fmt, vargs.clone()) };
va_end!();
err = unsafe { snd_hdac_device_init(core::ptr::addr_of_mut!((*codec).core), core::ptr::addr_of_mut!((*bus).core), name.0.as_mut_ptr() as *const u8, codec_addr) };
if err < 0 {
unsafe { kfree(codec as *const ()) };
return ERR_PTR(err as i64) as *mut hda_codec;
}
unsafe {
(*codec).bus = bus;
}
unsafe {
(*codec).depop_delay = 1i32.wrapping_neg();
}
unsafe {
(*codec).fixup_id = 1i32.wrapping_neg();
}
unsafe {
(*codec).core.dev.release = Some(snd_hda_codec_dev_release);
}
unsafe {
(*codec).core.r#type = 1;
}
loop {
__mutex_init(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }, c"&codec->spdif_mutex".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key__916) });
if true {
break;
}
}
loop {
__mutex_init(unsafe { core::ptr::addr_of_mut!((*codec).control_mutex) }, c"&codec->control_mutex".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key__917) });
if true {
break;
}
}
snd_array_init(unsafe { core::ptr::addr_of_mut!((*codec).mixers) }, core::mem::size_of::<hda_nid_item>() as u32, 32u32);
snd_array_init(unsafe { core::ptr::addr_of_mut!((*codec).nids) }, core::mem::size_of::<hda_nid_item>() as u32, 32u32);
snd_array_init(unsafe { core::ptr::addr_of_mut!((*codec).init_pins) }, core::mem::size_of::<hda_pincfg>() as u32, 16u32);
snd_array_init(unsafe { core::ptr::addr_of_mut!((*codec).driver_pins) }, core::mem::size_of::<hda_pincfg>() as u32, 16u32);
snd_array_init(unsafe { core::ptr::addr_of_mut!((*codec).cvt_setups) }, core::mem::size_of::<hda_cvt_setup>() as u32, 8u32);
snd_array_init(unsafe { core::ptr::addr_of_mut!((*codec).spdif_out) }, core::mem::size_of::<hda_spdif_out>() as u32, 16u32);
snd_array_init(unsafe { core::ptr::addr_of_mut!((*codec).jacktbl) }, core::mem::size_of::<hda_jack_tbl>() as u32, 16u32);
snd_array_init(unsafe { core::ptr::addr_of_mut!((*codec).verbs) }, core::mem::size_of::<*mut hda_verb>() as u32, 8u32);
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*codec).conn_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*codec).pcm_list_head) });
'__slate_break_15: loop {
'__slate_break_14: loop {
'__slate_break_13: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*codec).jackpoll_work)).work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*codec).jackpoll_work)).work)).data = *{
__slate_compound_12 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_12)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*codec).jackpoll_work)).work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*codec).jackpoll_work)).work)).func = Some(hda_jackpoll_work);
}
if true {
break;
}
}
if true {
break;
}
}
__timer_init!(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*codec).jackpoll_work)).timer) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) }, 2097152i32);
if true {
break;
}
}
unsafe { snd_refcount_init(core::ptr::addr_of_mut!((*codec).pcm_ref)) };
codec
}

static mut __key__916: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__917: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__928: lock_class_key = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_device_init_543: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_bus, u32, *const u8, ...) -> *mut hda_codec>, *mut ()>(Some(snd_hda_codec_device_init)) };

/// snd_hda_codec_new - create a HDA codec
/// @bus: the bus to assign
/// @card: card for this codec
/// @codec_addr: the codec address
/// @codecp: the pointer to store the generated codec
///
/// Returns 0 if successful, or a negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_new")]
extern "C" fn snd_hda_codec_new(bus: *mut hda_bus, card: *mut snd_card, codec_addr: u32, codecp: *mut *mut hda_codec) -> i32 {
let mut ret: i32 = 0;
let codec: *mut hda_codec = unsafe { snd_hda_codec_device_init(bus, codec_addr, c"hdaudioC%dD%d".as_ptr() as *const u8, (*card).number, codec_addr) };
if IS_ERR(codec as *const ()) {
return PTR_ERR(codec as *const ()) as i32;
}
unsafe {
*codecp = codec;
}
ret = snd_hda_codec_device_new(bus, card, codec_addr, unsafe { *codecp }, true);
if ret != 0 {
unsafe { put_device(hda_codec_dev!(unsafe { *codecp })) };
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_new_544: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_bus, *mut snd_card, u32, *mut *mut hda_codec) -> i32>, *mut ()>(Some(snd_hda_codec_new)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_device_new")]
extern "C" fn snd_hda_codec_device_new(bus: *mut hda_bus, card: *mut snd_card, codec_addr: u32, codec: *mut hda_codec, snddev_managed: bool) -> i32 {
let mut __slate_compound_4: pm_message = unsafe { core::mem::zeroed() };
let mut component: __SlateAlign16<[u8; 31]> = __SlateAlign16([0u8; 31]);
let mut fg: u16 = 0;
let mut err: i32 = 0;
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*card).dev as *const device, c"%s: entry\n".as_ptr() as *const u8, c"snd_hda_codec_device_new".as_ptr() as *mut u8) };
}
if snd_BUG_ON!(bus.is_null()) != 0 {
return 22i32.wrapping_neg();
}
if snd_BUG_ON!(codec_addr > 15) != 0 {
return 22i32.wrapping_neg();
}
unsafe {
(*codec).core.exec_verb = Some(codec_exec_verb);
}
unsafe {
(*codec).card = card;
}
unsafe {
(*codec).addr = codec_addr;
}
unsafe {
(*codec).power_jiffies = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
}
unsafe { snd_hda_sysfs_init(codec) };
if !unsafe { (*(*codec).bus).modelname }.is_null() {
unsafe {
(*codec).modelname = kstrdup((*(*codec).bus).modelname, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *const u8;
}
if unsafe { (*codec).modelname }.is_null() {
return 12i32.wrapping_neg();
}
}
fg = (if (unsafe { (*codec).core.afg }) != 0 { (unsafe { (*codec).core.afg }) as i32 } else { (unsafe { (*codec).core.mfg }) as i32 }) as u16;
err = read_widget_caps(codec, fg);
if err < 0 {
return err;
}
err = read_pin_defaults(codec);
if err < 0 {
return err;
}
// power-up all before initialization
hda_set_power_state(codec, 0u32);
unsafe {
(*codec).core.dev.power.power_state = *{
__slate_compound_4 = pm_message { event: 0 };
core::ptr::addr_of_mut!(__slate_compound_4)
};
}
unsafe { snd_hda_codec_proc_new(codec) };
unsafe { snd_hda_create_hwdep(codec) };
unsafe { sprintf(component.0.as_mut_ptr() as *mut u8, c"HDA:%08x,%08x,%08x".as_ptr() as *const u8, (*codec).core.vendor_id, (*codec).core.subsystem_id, (*codec).core.revision_id) };
unsafe { snd_component_add(card, component.0.as_mut_ptr() as *const u8) };
if snddev_managed {
// ASoC features component management instead
err = unsafe { snd_device_new(card, 3u32, codec as *mut (), core::ptr::addr_of!(dev_ops.0)) };
if err < 0 {
return err;
}
}
// PM runtime needs to be enabled later after binding codec
if ((unsafe { (*codec).core.dev.power.__slate_bits_2.__get_runtime_auto() }) as i32) != 0 {
unsafe { pm_runtime_forbid(core::ptr::addr_of_mut!((*codec).core.dev)) };
} else {
pm_runtime_get_noresume(unsafe { core::ptr::addr_of_mut!((*codec).core.dev) });
}
// Keep the usage_count consistent across subsequent probing
0
}

static dev_ops: __SlateConst<snd_device_ops> = __SlateConst({
let mut __t0: snd_device_ops = unsafe { core::mem::zeroed() };
__t0.dev_free = Some(snd_hda_codec_dev_free);
__t0.dev_register = Some(snd_hda_codec_dev_register);
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_device_new_545: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_bus, *mut snd_card, u32, *mut hda_codec, bool) -> i32>, *mut ()>(Some(snd_hda_codec_device_new)) };

/// snd_hda_codec_update_widgets - Refresh widget caps and pin defaults
/// @codec: the HDA codec
///
/// Forcibly refresh the all widget caps and the init pin configurations of
/// the given codec.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_update_widgets")]
extern "C" fn snd_hda_codec_update_widgets(codec: *mut hda_codec) -> i32 {
let mut fg: u16 = 0;
let mut err: i32 = unsafe { snd_hdac_refresh_widgets(core::ptr::addr_of_mut!((*codec).core)) };
if err < 0 {
return err;
}
// Assume the function group node does not change,
// only the widget nodes may change.
unsafe { kfree((*codec).wcaps as *const ()) };
fg = (if (unsafe { (*codec).core.afg }) != 0 { (unsafe { (*codec).core.afg }) as i32 } else { (unsafe { (*codec).core.mfg }) as i32 }) as u16;
err = read_widget_caps(codec, fg);
if err < 0 {
return err;
}
unsafe { snd_array_free(core::ptr::addr_of_mut!((*codec).init_pins)) };
err = read_pin_defaults(codec);
err
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_update_widgets_546: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec) -> i32>, *mut ()>(Some(snd_hda_codec_update_widgets)) };

/// update the stream-id if changed
fn update_pcm_stream_id(codec: *mut hda_codec, p: *mut hda_cvt_setup, nid: u16, stream_tag: u32, channel_id: i32) {
let mut oldval: u32 = 0;
let mut newval: u32 = 0;
if (((unsafe { (*p).stream_tag }) as i32) as u32) != stream_tag || ((unsafe { (*p).channel_id }) as i32) != channel_id {
oldval = snd_hda_codec_read(codec, nid, 0i32, 3846u32, 0u32);
newval = stream_tag << 4i32 | (channel_id as u32);
if oldval != newval {
snd_hda_codec_write(codec, nid, 0i32, 1798u32, newval);
}
unsafe {
(*p).stream_tag = stream_tag as u8;
}
unsafe {
(*p).channel_id = channel_id as u8;
}
}
}

/// update the format-id if changed
fn update_pcm_format(codec: *mut hda_codec, p: *mut hda_cvt_setup, nid: u16, format: i32) {
let mut oldval: u32 = 0;
if ((unsafe { (*p).format_id }) as i32) != format {
oldval = snd_hda_codec_read(codec, nid, 0i32, 2560u32, 0u32);
if oldval != (format as u32) {
unsafe { msleep(1u32) };
snd_hda_codec_write(codec, nid, 0i32, 512u32, format as u32);
}
unsafe {
(*p).format_id = format as u16;
}
}
}

/// snd_hda_codec_setup_stream - set up the codec for streaming
/// @codec: the CODEC to set up
/// @nid: the NID to set up
/// @stream_tag: stream tag to pass, it's between 0x1 and 0xf.
/// @channel_id: channel id to pass, zero based.
/// @format: stream format.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_setup_stream")]
extern "C" fn snd_hda_codec_setup_stream(codec: *mut hda_codec, nid: u16, stream_tag: u32, channel_id: i32, format: i32) {
let driver: *mut hda_codec_driver = hda_codec_to_driver!(codec);
let mut c: *mut hda_codec = unsafe { core::mem::zeroed() };
let mut p: *mut hda_cvt_setup = unsafe { core::mem::zeroed() };
let mut r#type: i32 = 0;
let mut i: i32 = 0;
if nid == 0 {
return;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, hda_codec_dev!(codec) as *const device, c"hda_codec_setup_stream: NID=0x%x, stream=0x%x, channel=%d, format=0x%x\n".as_ptr() as *const u8, nid as i32, stream_tag, channel_id, format) };
}
p = get_hda_cvt_setup(codec, nid);
if p.is_null() {
return;
}
if (unsafe { (*(*driver).ops).stream_pm }) != None {
unsafe { unsafe { (*(*driver).ops).stream_pm }.unwrap()(codec, nid, true) };
}
if ((unsafe { (*codec).__slate_bits_0.__get_pcm_format_first() }) as i32) != 0 {
update_pcm_format(codec, p, nid, format);
}
update_pcm_stream_id(codec, p, nid, stream_tag, channel_id);
if ((unsafe { (*codec).__slate_bits_0.__get_pcm_format_first() }) as i32) == 0 {
update_pcm_format(codec, p, nid, format);
}
unsafe {
(*p).active = 1;
}
unsafe {
(*p).dirty = 0;
}
// make other inactive cvts with the same stream-tag dirty
r#type = get_wcaps_type(get_wcaps(codec, nid));
let mut __v1: *mut hda_codec;
__v1 = ((unsafe { (((*core::ptr::addr_of_mut!((*(*codec).bus).core.codec_list)).next as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut hda_codec;
c = __v1;
while list_is_head((unsafe { core::ptr::addr_of_mut!((*c).core.list) }) as *const list_head, (unsafe { core::ptr::addr_of_mut!((*(*codec).bus).core.codec_list) }) as *const list_head) == 0 {
i = 0;
let __v2: *mut hda_cvt_setup = (unsafe { (*core::ptr::addr_of_mut!((*c).cvt_setups)).list }) as *mut hda_cvt_setup;
p = __v2;
while (i as u32) < unsafe { (*core::ptr::addr_of_mut!((*c).cvt_setups)).used } {
let __v3: bool = (unsafe { (*p).active }) == 0 && (((unsafe { (*p).stream_tag }) as i32) as u32) == stream_tag && get_wcaps_type(get_wcaps(c, unsafe { (*p).nid })) == r#type;
if __v3 {
unsafe {
(*p).dirty = 1;
}
}
let __v6: i32 = i.wrapping_add(1);
i = __v6;
p = snd_array_elem(unsafe { core::ptr::addr_of_mut!((*c).cvt_setups) }, __v6 as u32) as *mut hda_cvt_setup;
}
let mut __v8: *mut hda_codec;
__v8 = ((unsafe { (((*c).core.list.next as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut hda_codec;
c = __v8;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_setup_stream_547: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, u32, i32, i32)>, *mut ()>(Some(snd_hda_codec_setup_stream)) };

/// __snd_hda_codec_cleanup_stream - clean up the codec for closing
/// @codec: the CODEC to clean up
/// @nid: the NID to clean up
/// @do_now: really clean up the stream instead of clearing the active flag
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.__snd_hda_codec_cleanup_stream")]
extern "C" fn __snd_hda_codec_cleanup_stream(codec: *mut hda_codec, nid: u16, mut do_now: i32) {
if nid == 0 {
return;
}
if ((unsafe { (*codec).__slate_bits_0.__get_no_sticky_stream() }) as i32) != 0 {
do_now = 1;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, hda_codec_dev!(codec) as *const device, c"hda_codec_cleanup_stream: NID=0x%x\n".as_ptr() as *const u8, nid as i32) };
}
let p: *mut hda_cvt_setup = get_hda_cvt_setup(codec, nid);
if !p.is_null() {
// here we just clear the active flag when do_now isn't set;
// actual clean-ups will be done later in
// purify_inactive_streams() called from snd_hda_codec_prpapre()
if do_now != 0 {
really_cleanup_stream(codec, p);
} else {
unsafe {
(*p).active = 0;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___snd_hda_codec_cleanup_stream_548: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, i32)>, *mut ()>(Some(__snd_hda_codec_cleanup_stream)) };

fn really_cleanup_stream(codec: *mut hda_codec, q: *mut hda_cvt_setup) {
let driver: *mut hda_codec_driver = hda_codec_to_driver!(codec);
let nid: u16 = unsafe { (*q).nid };
if (unsafe { (*q).stream_tag }) != 0 || (unsafe { (*q).channel_id }) != 0 {
snd_hda_codec_write(codec, nid, 0i32, 1798u32, 0u32);
}
if (unsafe { (*q).format_id }) != 0 {
snd_hda_codec_write(codec, nid, 0i32, 512u32, 0u32);
}
unsafe { memset(q as *mut (), 0i32, core::mem::size_of::<hda_cvt_setup>() as u64) };
unsafe {
(*q).nid = nid;
}
if (unsafe { (*(*driver).ops).stream_pm }) != None {
unsafe { unsafe { (*(*driver).ops).stream_pm }.unwrap()(codec, nid, false) };
}
}

/// clean up the all conflicting obsolete streams
fn purify_inactive_streams(codec: *mut hda_codec) {
let mut c: *mut hda_codec = unsafe { core::mem::zeroed() };
let mut p: *mut hda_cvt_setup = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut __v0: *mut hda_codec;
__v0 = ((unsafe { (((*core::ptr::addr_of_mut!((*(*codec).bus).core.codec_list)).next as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut hda_codec;
c = __v0;
while list_is_head((unsafe { core::ptr::addr_of_mut!((*c).core.list) }) as *const list_head, (unsafe { core::ptr::addr_of_mut!((*(*codec).bus).core.codec_list) }) as *const list_head) == 0 {
i = 0;
let __v1: *mut hda_cvt_setup = (unsafe { (*core::ptr::addr_of_mut!((*c).cvt_setups)).list }) as *mut hda_cvt_setup;
p = __v1;
while (i as u32) < unsafe { (*core::ptr::addr_of_mut!((*c).cvt_setups)).used } {
if (unsafe { (*p).dirty }) != 0 {
really_cleanup_stream(c, p);
}
let __v4: i32 = i.wrapping_add(1);
i = __v4;
p = snd_array_elem(unsafe { core::ptr::addr_of_mut!((*c).cvt_setups) }, __v4 as u32) as *mut hda_cvt_setup;
}
let mut __v6: *mut hda_codec;
__v6 = ((unsafe { (((*c).core.list.next as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut hda_codec;
c = __v6;
}
}

/// clean up all streams; called from suspend
fn hda_cleanup_all_streams(codec: *mut hda_codec) {
let mut p: *mut hda_cvt_setup = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let __v0: *mut hda_cvt_setup = (unsafe { (*core::ptr::addr_of_mut!((*codec).cvt_setups)).list }) as *mut hda_cvt_setup;
p = __v0;
while (i as u32) < unsafe { (*core::ptr::addr_of_mut!((*codec).cvt_setups)).used } {
if (unsafe { (*p).stream_tag }) != 0 {
really_cleanup_stream(codec, p);
}
let __v3: i32 = i.wrapping_add(1);
i = __v3;
p = snd_array_elem(unsafe { core::ptr::addr_of_mut!((*codec).cvt_setups) }, __v3 as u32) as *mut hda_cvt_setup;
}
}

// amp access functions
/// query_amp_caps - query AMP capabilities
/// @codec: the HD-auio codec
/// @nid: the NID to query
/// @direction: either #HDA_INPUT or #HDA_OUTPUT
///
/// Query AMP capabilities for the given widget and direction.
/// Returns the obtained capability bits.
///
/// When cap bits have been already read, this doesn't read again but
/// returns the cached value.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.query_amp_caps")]
extern "C" fn query_amp_caps(codec: *mut hda_codec, mut nid: u16, direction: i32) -> u32 {
if get_wcaps(codec, nid) & 8u32 == 0 {
nid = unsafe { (*codec).core.afg };
}
snd_hda_param_read!(codec, nid, if direction == 1 { 18i32 } else { 13i32 }) as u32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_query_amp_caps_549: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, i32) -> u32>, *mut ()>(Some(query_amp_caps)) };

/// snd_hda_check_amp_caps - query AMP capabilities
/// @codec: the HD-audio codec
/// @nid: the NID to query
/// @dir: either #HDA_INPUT or #HDA_OUTPUT
/// @bits: bit mask to check the result
///
/// Check whether the widget has the given amp capability for the direction.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_check_amp_caps")]
extern "C" fn snd_hda_check_amp_caps(codec: *mut hda_codec, nid: u16, dir: i32, bits__0: u32) -> bool {
if nid == 0 {
return false;
}
if get_wcaps(codec, nid) & ((1i32 << dir.wrapping_add(1)) as u32) != 0 {
if query_amp_caps(codec, nid, dir) & bits__0 != 0 {
return true;
}
}
false
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_check_amp_caps_550: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, i32, u32) -> bool>, *mut ()>(Some(snd_hda_check_amp_caps)) };

/// snd_hda_override_amp_caps - Override the AMP capabilities
/// @codec: the CODEC to clean up
/// @nid: the NID to clean up
/// @dir: either #HDA_INPUT or #HDA_OUTPUT
/// @caps: the capability bits to set
///
/// Override the cached AMP caps bits value by the given one.
/// This function is useful if the driver needs to adjust the AMP ranges,
/// e.g. limit to 0dB, etc.
///
/// Returns zero if successful or a negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_override_amp_caps")]
extern "C" fn snd_hda_override_amp_caps(codec: *mut hda_codec, nid: u16, dir: i32, caps: u32) -> i32 {
let mut parm: u32 = 0;
snd_hda_override_wcaps(codec, nid, get_wcaps(codec, nid) | 8u32);
parm = (if dir == 1 { 18i32 } else { 13i32 }) as u32;
unsafe { snd_hdac_override_parm(core::ptr::addr_of_mut!((*codec).core), nid, parm, caps) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_override_amp_caps_551: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, i32, u32) -> i32>, *mut ()>(Some(snd_hda_override_amp_caps)) };

fn encode_amp(codec: *mut hda_codec, nid: u16, ch: i32, dir: i32, idx: i32) -> u32 {
let mut cmd: u32 = snd_hdac_regmap_encode_amp!(nid, ch != 0, dir, idx);
// enable fake mute if no h/w mute but min=mute
if query_amp_caps(codec, nid, dir) & 3221225472u32 == 1073741824u32 {
cmd |= 16;
}
cmd
}

/// snd_hda_codec_amp_update - update the AMP mono value
/// @codec: HD-audio codec
/// @nid: NID to read the AMP value
/// @ch: channel to update (0 or 1)
/// @dir: #HDA_INPUT or #HDA_OUTPUT
/// @idx: the index value (only for input direction)
/// @mask: bit mask to set
/// @val: the bits value to set
///
/// Update the AMP values for the given channel, direction and index.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_amp_update")]
extern "C" fn snd_hda_codec_amp_update(codec: *mut hda_codec, nid: u16, ch: i32, dir: i32, idx: i32, mask: i32, val: i32) -> i32 {
let cmd: u32 = encode_amp(codec, nid, ch, dir, idx);
unsafe { snd_hdac_regmap_update_raw(core::ptr::addr_of_mut!((*codec).core), cmd, mask as u32, val as u32) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_amp_update_552: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, i32, i32, i32, i32, i32) -> i32>, *mut ()>(Some(snd_hda_codec_amp_update)) };

/// snd_hda_codec_amp_stereo - update the AMP stereo values
/// @codec: HD-audio codec
/// @nid: NID to read the AMP value
/// @direction: #HDA_INPUT or #HDA_OUTPUT
/// @idx: the index value (only for input direction)
/// @mask: bit mask to set
/// @val: the bits value to set
///
/// Update the AMP values like snd_hda_codec_amp_update(), but for a
/// stereo widget with the same mask and value.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_amp_stereo")]
extern "C" fn snd_hda_codec_amp_stereo(codec: *mut hda_codec, nid: u16, direction: i32, idx: i32, mut mask: i32, val: i32) -> i32 {
let mut ret: i32 = 0;
if snd_BUG_ON__2!(mask & !255) != 0 {
mask &= 255;
}
for ch in 0i32..2i32 {
let __v4: i32 = snd_hda_codec_amp_update(codec, nid, ch, direction, idx, mask, val);
ret |= __v4;
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_amp_stereo_553: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, i32, i32, i32, i32) -> i32>, *mut ()>(Some(snd_hda_codec_amp_stereo)) };

/// snd_hda_codec_amp_init - initialize the AMP value
/// @codec: the HDA codec
/// @nid: NID to read the AMP value
/// @ch: channel (left=0 or right=1)
/// @dir: #HDA_INPUT or #HDA_OUTPUT
/// @idx: the index value (only for input direction)
/// @mask: bit mask to set
/// @val: the bits value to set
///
/// Works like snd_hda_codec_amp_update() but it writes the value only at
/// the first access.  If the amp was already initialized / updated beforehand,
/// this does nothing.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_amp_init")]
extern "C" fn snd_hda_codec_amp_init(codec: *mut hda_codec, nid: u16, ch: i32, dir: i32, idx: i32, mask: i32, val: i32) -> i32 {
let cmd: u32 = encode_amp(codec, nid, ch, dir, idx);
if unsafe { (*codec).core.regmap }.is_null() {
return 22i32.wrapping_neg();
}
unsafe { snd_hdac_regmap_update_raw_once(core::ptr::addr_of_mut!((*codec).core), cmd, mask as u32, val as u32) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_amp_init_554: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, i32, i32, i32, i32, i32) -> i32>, *mut ()>(Some(snd_hda_codec_amp_init)) };

/// snd_hda_codec_amp_init_stereo - initialize the stereo AMP value
/// @codec: the HDA codec
/// @nid: NID to read the AMP value
/// @dir: #HDA_INPUT or #HDA_OUTPUT
/// @idx: the index value (only for input direction)
/// @mask: bit mask to set
/// @val: the bits value to set
///
/// Call snd_hda_codec_amp_init() for both stereo channels.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_amp_init_stereo")]
extern "C" fn snd_hda_codec_amp_init_stereo(codec: *mut hda_codec, nid: u16, dir: i32, idx: i32, mut mask: i32, val: i32) -> i32 {
let mut ret: i32 = 0;
if snd_BUG_ON__2!(mask & !255) != 0 {
mask &= 255;
}
for ch in 0i32..2i32 {
let __v4: i32 = snd_hda_codec_amp_init(codec, nid, ch, dir, idx, mask, val);
ret |= __v4;
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_amp_init_stereo_555: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, i32, i32, i32, i32) -> i32>, *mut ()>(Some(snd_hda_codec_amp_init_stereo)) };

fn get_amp_max_value(codec: *mut hda_codec, nid: u16, dir: i32, ofs: u32) -> u32 {
let mut caps: u32 = query_amp_caps(codec, nid, dir);
// get num steps
caps = (caps & 32512u32) >> 8i32;
if ofs < caps {
caps = caps.wrapping_sub(ofs);
}
caps
}

/// snd_hda_mixer_amp_volume_info - Info callback for a standard AMP mixer
/// @kcontrol: referred ctl element
/// @uinfo: pointer to get/store the data
///
/// The control element is supposed to have the private_value field
/// set up via HDA_COMPOSE_AMP_VAL*() or related macros.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_mixer_amp_volume_info")]
extern "C" fn snd_hda_mixer_amp_volume_info(kcontrol: *mut snd_kcontrol, uinfo: *mut snd_ctl_elem_info) -> i32 {
let codec: *mut hda_codec = snd_kcontrol_chip!(kcontrol) as *mut hda_codec;
let nid: u16 = get_amp_nid!(kcontrol) as u16;
let chs: u8 = get_amp_channels!(kcontrol) as u8;
let dir: i32 = get_amp_direction!(kcontrol) as i32;
let ofs: u32 = get_amp_offset!(kcontrol) as u32;
unsafe {
(*uinfo).r#type = 2;
}
unsafe {
(*uinfo).count = (if (chs as i32) == 3 { 2i32 } else { 1i32 }) as u32;
}
unsafe {
(*uinfo).value.integer.min = 0;
}
unsafe {
(*uinfo).value.integer.max = get_amp_max_value(codec, nid, dir, ofs) as i64;
}
if (unsafe { (*uinfo).value.integer.max }) == 0 {
while false {
}
unsafe { _dev_warn(hda_codec_dev!(codec) as *const device, c"num_steps = 0 for NID=0x%x (ctl = %s)\n".as_ptr() as *const u8, nid as i32, (*kcontrol).id.name.as_mut_ptr() as *mut u8) };
return 22i32.wrapping_neg();
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_mixer_amp_volume_info_556: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_info) -> i32>, *mut ()>(Some(snd_hda_mixer_amp_volume_info)) };

fn read_amp_value(codec: *mut hda_codec, nid: u16, ch: i32, dir: i32, idx: i32, ofs: u32) -> u32 {
let mut val: u32 = snd_hda_codec_amp_read!(codec, nid, ch, dir, idx) as u32;
val &= 127;
if val >= ofs {
val = val.wrapping_sub(ofs);
} else {
val = 0;
}
val
}

fn update_amp_value(codec: *mut hda_codec, nid: u16, ch: i32, dir: i32, idx: i32, ofs: u32, mut val: u32) -> i32 {
let mut maxval: u32 = 0;
if val > 0 {
val = val.wrapping_add(ofs);
}
// ofs = 0: raw max value
maxval = get_amp_max_value(codec, nid, dir, 0u32);
if val > maxval {
return 22i32.wrapping_neg();
}
snd_hda_codec_amp_update(codec, nid, ch, dir, idx, 127i32, val as i32)
}

/// snd_hda_mixer_amp_volume_get - Get callback for a standard AMP mixer volume
/// @kcontrol: ctl element
/// @ucontrol: pointer to get/store the data
///
/// The control element is supposed to have the private_value field
/// set up via HDA_COMPOSE_AMP_VAL*() or related macros.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_mixer_amp_volume_get")]
extern "C" fn snd_hda_mixer_amp_volume_get(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
let codec: *mut hda_codec = snd_kcontrol_chip!(kcontrol) as *mut hda_codec;
let nid: u16 = get_amp_nid!(kcontrol) as u16;
let chs: i32 = get_amp_channels!(kcontrol) as i32;
let dir: i32 = get_amp_direction!(kcontrol) as i32;
let idx: i32 = get_amp_index!(kcontrol) as i32;
let ofs: u32 = get_amp_offset!(kcontrol) as u32;
let mut valp: *mut i64 = unsafe { (*ucontrol).value.integer.value.as_mut_ptr() as *mut i64 };
if chs & 1 != 0 {
let __v0: *mut i64 = valp;
valp = __v0.wrapping_offset(1 as isize);
unsafe {
*__v0 = read_amp_value(codec, nid, 0i32, dir, idx, ofs) as i64;
}
}
if chs & 2 != 0 {
unsafe {
*valp = read_amp_value(codec, nid, 1i32, dir, idx, ofs) as i64;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_mixer_amp_volume_get_557: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_value) -> i32>, *mut ()>(Some(snd_hda_mixer_amp_volume_get)) };

/// snd_hda_mixer_amp_volume_put - Put callback for a standard AMP mixer volume
/// @kcontrol: ctl element
/// @ucontrol: pointer to get/store the data
///
/// The control element is supposed to have the private_value field
/// set up via HDA_COMPOSE_AMP_VAL*() or related macros.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_mixer_amp_volume_put")]
extern "C" fn snd_hda_mixer_amp_volume_put(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
let codec: *mut hda_codec = snd_kcontrol_chip!(kcontrol) as *mut hda_codec;
let nid: u16 = get_amp_nid!(kcontrol) as u16;
let chs: i32 = get_amp_channels!(kcontrol) as i32;
let dir: i32 = get_amp_direction!(kcontrol) as i32;
let idx: i32 = get_amp_index!(kcontrol) as i32;
let ofs: u32 = get_amp_offset!(kcontrol) as u32;
let mut valp: *mut i64 = unsafe { (*ucontrol).value.integer.value.as_mut_ptr() as *mut i64 };
let mut change: i32 = 0;
let mut err: i32 = 0;
if chs & 1 != 0 {
err = update_amp_value(codec, nid, 0i32, dir, idx, ofs, (unsafe { *valp }) as u32);
if err < 0 {
return err;
}
change |= err;
let __v2: *mut i64 = valp;
valp = __v2.wrapping_offset(1 as isize);
}
if chs & 2 != 0 {
err = update_amp_value(codec, nid, 1i32, dir, idx, ofs, (unsafe { *valp }) as u32);
if err < 0 {
return err;
}
change |= err;
}
change
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_mixer_amp_volume_put_558: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_value) -> i32>, *mut ()>(Some(snd_hda_mixer_amp_volume_put)) };

/// inquiry the amp caps and convert to TLV
fn get_ctl_amp_tlv(kcontrol: *mut snd_kcontrol, tlv: *mut u32) {
let codec: *mut hda_codec = snd_kcontrol_chip!(kcontrol) as *mut hda_codec;
let nid: u16 = get_amp_nid!(kcontrol) as u16;
let dir: i32 = get_amp_direction!(kcontrol) as i32;
let ofs: u32 = get_amp_offset!(kcontrol) as u32;
let min_mute: bool = get_amp_min_mute!(kcontrol) != 0;
let caps: u32 = query_amp_caps(codec, nid, dir);
let mut val1: u32 = 0;
let mut val2: u32 = (caps & 8323072u32) >> 16i32;
val2 = val2.wrapping_add(1).wrapping_mul(25);
val1 = ((caps & 127u32) >> 0i32).wrapping_neg();
val1 = val1.wrapping_add(ofs);
val1 = (val1 as i32).wrapping_mul(val2 as i32) as u32;
if min_mute || caps & 1073741824u32 != 0 {
val2 |= 65536;
}
unsafe {
*tlv.wrapping_offset(0 as isize) = 1;
}
unsafe {
*tlv.wrapping_offset(1 as isize) = 2u64.wrapping_mul(core::mem::size_of::<u32>() as u64) as u32;
}
unsafe {
*tlv.wrapping_offset(2 as isize) = val1;
}
unsafe {
*tlv.wrapping_offset(3 as isize) = val2;
}
}

/// snd_hda_mixer_amp_tlv - TLV callback for a standard AMP mixer volume
/// @kcontrol: ctl element
/// @op_flag: operation flag
/// @size: byte size of input TLV
/// @_tlv: TLV data
///
/// The control element is supposed to have the private_value field
/// set up via HDA_COMPOSE_AMP_VAL*() or related macros.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_mixer_amp_tlv")]
extern "C" fn snd_hda_mixer_amp_tlv(kcontrol: *mut snd_kcontrol, op_flag: i32, size: u32, _tlv: *mut u32) -> i32 {
let mut tlv: __SlateAlign16<[u32; 4]> = __SlateAlign16([0u32; 4]);
if (size as u64) < 4u64.wrapping_mul(core::mem::size_of::<u32>() as u64) {
return 12i32.wrapping_neg();
}
get_ctl_amp_tlv(kcontrol, tlv.0.as_mut_ptr() as *mut u32);
if copy_to_user(_tlv as *mut (), tlv.0.as_mut_ptr() as *const (), core::mem::size_of::<[u32; 4]>() as u64) != 0 {
return 14i32.wrapping_neg();
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_mixer_amp_tlv_559: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_kcontrol, i32, u32, *mut u32) -> i32>, *mut ()>(Some(snd_hda_mixer_amp_tlv)) };

/// snd_hda_set_vmaster_tlv - Set TLV for a virtual master control
/// @codec: HD-audio codec
/// @nid: NID of a reference widget
/// @dir: #HDA_INPUT or #HDA_OUTPUT
/// @tlv: TLV data to be stored, at least 4 elements
///
/// Set (static) TLV data for a virtual master volume using the AMP caps
/// obtained from the reference NID.
/// The volume range is recalculated as if the max volume is 0dB.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_set_vmaster_tlv")]
extern "C" fn snd_hda_set_vmaster_tlv(codec: *mut hda_codec, nid: u16, dir: i32, tlv: *mut u32) {
let caps: u32 = query_amp_caps(codec, nid, dir);
let nums: i32 = ((caps & 32512u32) >> 8i32) as i32;
let mut step: i32 = ((caps & 8323072u32) >> 16i32) as i32;
step = step.wrapping_add(1).wrapping_mul(25);
unsafe {
*tlv.wrapping_offset(0 as isize) = 1;
}
unsafe {
*tlv.wrapping_offset(1 as isize) = 2u64.wrapping_mul(core::mem::size_of::<u32>() as u64) as u32;
}
unsafe {
*tlv.wrapping_offset(2 as isize) = nums.wrapping_neg().wrapping_mul(step) as u32;
}
unsafe {
*tlv.wrapping_offset(3 as isize) = step as u32;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_set_vmaster_tlv_560: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, i32, *mut u32)>, *mut ()>(Some(snd_hda_set_vmaster_tlv)) };

/// find a mixer control element with the given name
fn find_mixer_ctl(codec: *mut hda_codec, name: *const u8, dev: i32, idx: i32) -> *mut snd_kcontrol {
let mut id: snd_ctl_elem_id = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(id) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
unsafe { memset(core::ptr::addr_of_mut!(id) as *mut (), 0i32, core::mem::size_of::<snd_ctl_elem_id>() as u64) };
id.iface = 2;
id.device = dev as u32;
id.index = idx as u32;
if snd_BUG_ON!((unsafe { strlen(name) }) >= (core::mem::size_of::<[u8; 44]>() as u64)) != 0 {
return core::ptr::null_mut::<snd_kcontrol>();
}
unsafe { sized_strscpy(id.name.as_mut_ptr() as *mut u8, name, (core::mem::size_of::<[u8; 44]>() as u64).wrapping_add(__must_be_array__25!() as u64).wrapping_add(__must_be_cstr__49!() as u64).wrapping_add(__must_be_cstr__3!() as u64)) };
unsafe { snd_ctl_find_id((*codec).card, core::ptr::addr_of_mut!(id) as *const snd_ctl_elem_id) }
}

/// snd_hda_find_mixer_ctl - Find a mixer control element with the given name
/// @codec: HD-audio codec
/// @name: ctl id name string
///
/// Get the control element with the given id string and IFACE_MIXER.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_find_mixer_ctl")]
extern "C" fn snd_hda_find_mixer_ctl(codec: *mut hda_codec, name: *const u8) -> *mut snd_kcontrol {
find_mixer_ctl(codec, name, 0i32, 0i32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_find_mixer_ctl_561: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *const u8) -> *mut snd_kcontrol>, *mut ()>(Some(snd_hda_find_mixer_ctl)) };

fn find_empty_mixer_ctl_idx(codec: *mut hda_codec, name: *const u8, __v0: i32) -> i32 {
// 16 ctlrs should be large enough
let mut i: i32 = 0;
let mut idx: i32 = 0;
idx = __v0;
while i < 16 {
if find_mixer_ctl(codec, name, 0i32, idx).is_null() {
return idx;
}
i = i.wrapping_add(1);
idx = idx.wrapping_add(1);
}
16i32.wrapping_neg()
}

/// snd_hda_ctl_add - Add a control element and assign to the codec
/// @codec: HD-audio codec
/// @nid: corresponding NID (optional)
/// @kctl: the control element to assign
///
/// Add the given control element to an array inside the codec instance.
/// All control elements belonging to a codec are supposed to be added
/// by this function so that a proper clean-up works at the free or
/// reconfiguration time.
///
/// If non-zero @nid is passed, the NID is assigned to the control element.
/// The assignment is shown in the codec proc file.
///
/// snd_hda_ctl_add() checks the control subdev id field whether
/// #HDA_SUBDEV_NID_FLAG bit is set.  If set (and @nid is zero), the lower
/// bits value is taken as the NID to assign. The #HDA_NID_ITEM_AMP bit
/// specifies if kctl->private_value is a HDA amplifier value.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_ctl_add")]
extern "C" fn snd_hda_ctl_add(codec: *mut hda_codec, mut nid: u16, kctl: *mut snd_kcontrol) -> i32 {
let mut err: i32 = 0;
let mut flags: u16 = 0;
if kctl.is_null() {
return 22i32.wrapping_neg();
}
if (unsafe { (*kctl).id.subdevice }) & 1073741824u32 != 0 {
flags = ((flags as i32) | 1i32) as u16;
if (nid as i32) == 0 {
nid = get_amp_nid_!(unsafe { (*kctl).private_value }) as u16;
}
}
if (unsafe { (*kctl).id.subdevice }) & 2147483648u32 != 0 && (nid as i32) == 0 {
nid = ((unsafe { (*kctl).id.subdevice }) & 65535) as u16;
}
if (unsafe { (*kctl).id.subdevice }) & 3221225472u32 != 0 {
unsafe {
(*kctl).id.subdevice = 0;
}
}
err = unsafe { snd_ctl_add((*codec).card, kctl) };
if err < 0 {
return err;
}
let item: *mut hda_nid_item = (unsafe { snd_array_new(core::ptr::addr_of_mut!((*codec).mixers)) }) as *mut hda_nid_item;
if item.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*item).kctl = kctl;
}
unsafe {
(*item).nid = nid;
}
unsafe {
(*item).flags = flags;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_ctl_add_562: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, *mut snd_kcontrol) -> i32>, *mut ()>(Some(snd_hda_ctl_add)) };

/// snd_hda_ctls_clear - Clear all controls assigned to the given codec
/// @codec: HD-audio codec
#[unsafe(no_mangle)]
extern "C" fn snd_hda_ctls_clear(codec: *mut hda_codec) {
let mut i: i32 = 0;
let items: *mut hda_nid_item = (unsafe { (*codec).mixers.list }) as *mut hda_nid_item;
while (i as u32) < unsafe { (*codec).mixers.used } {
unsafe { snd_ctl_remove((*codec).card, (*items.wrapping_offset(i as isize)).kctl) };
i = i.wrapping_add(1);
}
unsafe { snd_array_free(core::ptr::addr_of_mut!((*codec).mixers)) };
unsafe { snd_array_free(core::ptr::addr_of_mut!((*codec).nids)) };
}

/// snd_hda_lock_devices - pseudo device locking
/// @bus: the BUS
///
/// toggle card->shutdown to allow/disallow the device access (as a hack)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_lock_devices")]
extern "C" fn snd_hda_lock_devices(bus: *mut hda_bus) -> i32 {
let mut cpcm: core::mem::MaybeUninit<*mut hda_pcm> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut hda_pcm = core::ptr::addr_of_mut!(cpcm) as *mut *mut hda_pcm;
let mut __UNIQUE_ID_unlock_564: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_610> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_564) as *mut *mut __SlateRecord_spinlock_h_610;
let mut __UNIQUE_ID_guard_563: core::mem::MaybeUninit<__SlateRecord_spinlock_h_610> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_563) as *mut __SlateRecord_spinlock_h_610;
let mut codec: core::mem::MaybeUninit<*mut hda_codec> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut hda_codec = core::ptr::addr_of_mut!(codec) as *mut *mut hda_codec;
let mut card: *mut snd_card = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_4 = __slate_slot_4;
let __slate_cleanup_armed_4 = core::cell::Cell::new(false);
let __slate_cleanup_4 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_4.get() {
unsafe { class_spinlock_destructor(__slate_cleanup_ptr_4 as *mut _) };
}
});
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _) };
}
});
unsafe {
card = (*bus).card;
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_4, class_spinlock_constructor(core::ptr::addr_of_mut!((*card).files_lock)));
__slate_cleanup_armed_4.set(true);
core::ptr::write(__slate_slot_3, (core::ptr::addr_of_mut!((*card).files_lock) as u64) as *mut __SlateRecord_spinlock_h_610);
__slate_cleanup_armed_3.set(true);
if (*card).shutdown != 0 {
return 22i32.wrapping_neg();
} else {
'__join_0: {
(*card).shutdown = 1;
if list_empty(core::ptr::addr_of_mut!((*card).ctl_files) as *const list_head) != 0 {
let __v2: *mut hda_codec = ((((*core::ptr::addr_of_mut!((*bus).core.codec_list)).next as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) as *mut ()) as *mut hda_codec;
*__slate_slot_5 = __v2;
while list_is_head(core::ptr::addr_of_mut!((*(*__slate_slot_5)).core.list) as *const list_head, core::ptr::addr_of_mut!((*bus).core.codec_list) as *const list_head) == 0 {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_1 = list_first_entry__3!(core::ptr::addr_of_mut!((*(*__slate_slot_5)).pcm_list_head));
while list_entry_is_head!(*__slate_slot_1, core::ptr::addr_of_mut!((*(*__slate_slot_5)).pcm_list_head), list) == 0 {
if !unsafe { (*(*__slate_slot_1)).pcm }.is_null() {
if (*((*(*(*__slate_slot_1)).pcm).streams.as_mut_ptr() as *mut snd_pcm_str).wrapping_offset(0 as isize)).substream_opened != 0 || (*((*(*(*__slate_slot_1)).pcm).streams.as_mut_ptr() as *mut snd_pcm_str).wrapping_offset(1 as isize)).substream_opened != 0 {
break '__join_0;
}
}
*__slate_slot_1 = list_next_entry__4!(*__slate_slot_1, list);
}
let __v0: *mut hda_codec = ((((*(*__slate_slot_5)).core.list.next as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) as *mut ()) as *mut hda_codec;
*__slate_slot_5 = __v0;
}
return 0;
}
}
(*card).shutdown = 0;
return 22i32.wrapping_neg();
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_lock_devices_565: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_bus) -> i32>, *mut ()>(Some(snd_hda_lock_devices)) };

/// snd_hda_unlock_devices - pseudo device unlocking
/// @bus: the BUS
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_unlock_devices")]
extern "C" fn snd_hda_unlock_devices(bus: *mut hda_bus) {
let card: *mut snd_card = unsafe { (*bus).card };
let mut __UNIQUE_ID_guard_566: __SlateRecord_spinlock_h_610 = class_spinlock_constructor(unsafe { core::ptr::addr_of_mut!((*card).files_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_566) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_567: *mut __SlateRecord_spinlock_h_610 = ((unsafe { core::ptr::addr_of_mut!((*card).files_lock) }) as u64) as *mut __SlateRecord_spinlock_h_610;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_567) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
unsafe {
(*card).shutdown = 0;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_unlock_devices_568: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_bus)>, *mut ()>(Some(snd_hda_unlock_devices)) };

/// snd_hda_codec_reset - Clear all objects assigned to the codec
/// @codec: HD-audio codec
///
/// This frees the all PCM and control elements assigned to the codec, and
/// clears the caches and restores the pin default configurations.
///
/// When a device is being used, it returns -EBSY.  If successfully freed,
/// returns zero.
#[unsafe(no_mangle)]
extern "C" fn snd_hda_codec_reset(codec: *mut hda_codec) -> i32 {
let bus: *mut hda_bus = unsafe { (*codec).bus };
if snd_hda_lock_devices(bus) < 0 {
return 16i32.wrapping_neg();
}
// OK, let it free
unsafe { device_release_driver(hda_codec_dev!(codec)) };
// allow device access again
snd_hda_unlock_devices(bus);
0
}

/// apply the function to all matching follower ctls in the mixer list
fn map_followers(codec: *mut hda_codec, followers: *const *const u8, suffix: *const u8, func: Option<unsafe extern "C" fn(*mut hda_codec, *mut (), *mut snd_kcontrol) -> i32>, data: *mut ()) -> i32 {
let mut s: *const *const u8 = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut err: i32 = 0;
let items: *mut hda_nid_item = (unsafe { (*codec).mixers.list }) as *mut hda_nid_item;
while (i as u32) < unsafe { (*codec).mixers.used } {
let sctl: *mut snd_kcontrol = unsafe { (*items.wrapping_offset(i as isize)).kctl };
if !(sctl.is_null() || (unsafe { (*sctl).id.iface }) != 2) {
s = followers;
while !unsafe { *s }.is_null() {
let mut tmpname: __SlateAlign16<[u8; 44]> = __SlateAlign16([0u8; 44]);
let mut name: *const u8 = unsafe { *s };
if !suffix.is_null() {
unsafe { snprintf(tmpname.0.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 44]>() as u64, c"%s %s".as_ptr() as *const u8, name, suffix) };
name = tmpname.0.as_mut_ptr() as *const u8;
}
if (unsafe { strcmp(((*sctl).id.name.as_mut_ptr() as *mut u8) as *const u8, name) }) == 0 {
err = unsafe { func.unwrap()(codec, data, sctl) };
if err != 0 {
return err;
}
break;
}
let __v2: *const *const u8 = s;
s = __v2.wrapping_offset(1 as isize);
}
}
i = i.wrapping_add(1);
}
0
}

#[unsafe(link_section = ".text.slate_distinct.codec.check_follower_present")]
extern "C" fn check_follower_present(codec: *mut hda_codec, data: *mut (), sctl: *mut snd_kcontrol) -> i32 {
1
}

/// call kctl->put with the given value(s)
fn put_kctl_with_value(kctl: *mut snd_kcontrol, val: i32) -> i32 {
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut ucontrol: *mut snd_ctl_elem_value = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(ucontrol) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_kfree(__slate_cleanup_ptr_0 as *mut _));
let mut __v1: *mut snd_ctl_elem_value;
let __obj_size: u64 = size_mul(core::mem::size_of::<snd_ctl_elem_value>() as u64, 1u64);
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
__v1 = __v2 as *mut snd_ctl_elem_value;
ucontrol = __v1;
if ucontrol.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
*((*ucontrol).value.integer.value.as_mut_ptr() as *mut i64).wrapping_offset(0 as isize) = val as i64;
}
unsafe {
*((*ucontrol).value.integer.value.as_mut_ptr() as *mut i64).wrapping_offset(1 as isize) = val as i64;
}
unsafe { unsafe { (*kctl).put }.unwrap()(kctl, ucontrol) };
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct follower_init_arg {
    codec: *mut hda_codec,
    step: i32,
}


/// initialize the follower volume with 0dB via snd_ctl_apply_vmaster_followers()
#[unsafe(link_section = ".text.slate_distinct.codec.init_follower_0dB")]
extern "C" fn init_follower_0dB(follower: *mut snd_kcontrol, kctl: *mut snd_kcontrol, _arg: *mut ()) -> i32 {
let arg: *mut follower_init_arg = _arg as *mut follower_init_arg;
let mut _tlv: __SlateAlign16<[i32; 4]> = __SlateAlign16([0i32; 4]);
let mut tlv: *const i32 = core::ptr::null::<i32>();
let mut step: i32 = 0;
let mut val: i32 = 0;
if (unsafe { (*(core::ptr::addr_of_mut!((*kctl).vd) as *mut snd_kcontrol_volatile).wrapping_offset(0 as isize)).access }) & 268435456u32 != 0 {
if (unsafe { (*kctl).tlv.c }) != Some(snd_hda_mixer_amp_tlv) {
while false {
}
unsafe { _dev_err(hda_codec_dev!(unsafe { (*arg).codec }) as *const device, c"Unexpected TLV callback for follower %s:%d\n".as_ptr() as *const u8, (*kctl).id.name.as_mut_ptr() as *mut u8, (*kctl).id.index) };
return 0; // ignore
}
get_ctl_amp_tlv(kctl, _tlv.0.as_mut_ptr() as *mut u32);
tlv = _tlv.0.as_mut_ptr() as *const i32;
} else {
if (unsafe { (*(core::ptr::addr_of_mut!((*kctl).vd) as *mut snd_kcontrol_volatile).wrapping_offset(0 as isize)).access }) & 16u32 != 0 {
tlv = (unsafe { (*kctl).tlv.p }) as *const i32;
}
}
if tlv.is_null() || (unsafe { *tlv.wrapping_offset(0 as isize) }) != 1 {
return 0;
}
step = unsafe { *tlv.wrapping_offset(3 as isize) };
let __v2: i32 = step & !65536;
step = __v2;
if step == 0 {
return 0;
}
if (unsafe { (*arg).step }) != 0 && (unsafe { (*arg).step }) != step {
while false {
}
unsafe { _dev_err(hda_codec_dev!(unsafe { (*arg).codec }) as *const device, c"Mismatching dB step for vmaster follower (%d!=%d)\n".as_ptr() as *const u8, (*arg).step, step) };
return 0;
}
unsafe {
(*arg).step = step;
}
val = unsafe { *tlv.wrapping_offset(2 as isize) }.wrapping_neg() / step;
if val > 0 {
put_kctl_with_value(follower, val);
return val;
}
0
}

/// unmute the follower via snd_ctl_apply_vmaster_followers()
#[unsafe(link_section = ".text.slate_distinct.codec.init_follower_unmute")]
extern "C" fn init_follower_unmute(follower: *mut snd_kcontrol, kctl: *mut snd_kcontrol, _arg: *mut ()) -> i32 {
put_kctl_with_value(follower, 1i32)
}

#[unsafe(link_section = ".text.slate_distinct.codec.add_follower")]
extern "C" fn add_follower(codec: *mut hda_codec, data: *mut (), follower: *mut snd_kcontrol) -> i32 {
snd_ctl_add_follower(data as *mut snd_kcontrol, follower)
}

/// __snd_hda_add_vmaster - create a virtual master control and add followers
/// @codec: HD-audio codec
/// @name: vmaster control name
/// @tlv: TLV data (optional)
/// @followers: follower control names (optional)
/// @suffix: suffix string to each follower name (optional)
/// @init_follower_vol: initialize followers to unmute/0dB
/// @access: kcontrol access rights
/// @ctl_ret: store the vmaster kcontrol in return
///
/// Create a virtual master control with the given name.  The TLV data
/// must be either NULL or a valid data.
///
/// @followers is a NULL-terminated array of strings, each of which is a
/// follower control name.  All controls with these names are assigned to
/// the new virtual master control.
///
/// This function returns zero if successful or a negative error code.
#[unsafe(no_mangle)]
extern "C" fn __snd_hda_add_vmaster(codec: *mut hda_codec, name: *mut u8, tlv: *mut u32, followers: *const *const u8, suffix: *const u8, init_follower_vol: bool, access: u32, ctl_ret: *mut *mut snd_kcontrol) -> i32 {
let mut kctl: *mut snd_kcontrol = unsafe { core::mem::zeroed() };
let mut err: i32 = 0;
if !ctl_ret.is_null() {
unsafe {
*ctl_ret = core::ptr::null_mut::<snd_kcontrol>();
}
}
err = map_followers(codec, followers, suffix, Some(check_follower_present), core::ptr::null_mut::<()>());
if err != 1 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, hda_codec_dev!(codec) as *const device, c"No follower found for %s\n".as_ptr() as *const u8, name) };
}
return 0;
}
kctl = unsafe { snd_ctl_make_virtual_master(name, tlv as *const u32) };
if kctl.is_null() {
return 12i32.wrapping_neg();
}
let __v0: *mut snd_kcontrol_volatile = unsafe { core::ptr::addr_of_mut!((*kctl).vd) as *mut snd_kcontrol_volatile }.wrapping_offset(0 as isize);
unsafe {
(*__v0).access = (*__v0).access | access;
}
err = snd_hda_ctl_add(codec, 0u16, kctl);
if err < 0 {
return err;
}
err = map_followers(codec, followers, suffix, Some(add_follower), kctl as *mut ());
if err < 0 {
return err;
}
// init with master mute & zero volume
put_kctl_with_value(kctl, 0i32);
if init_follower_vol {
let mut arg: follower_init_arg = {
let mut __t0: follower_init_arg = unsafe { core::mem::zeroed() };
__t0.codec = codec;
__t0.step = 0i32;
__t0
};
unsafe { snd_ctl_apply_vmaster_followers(kctl, {
let __t1: Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_kcontrol, *mut ()) -> i32> = if !tlv.is_null() { Some(init_follower_0dB) } else { Some(init_follower_unmute) };
__t1
}, core::ptr::addr_of_mut!(arg) as *mut ()) };
}
if !ctl_ret.is_null() {
unsafe {
*ctl_ret = kctl;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___snd_hda_add_vmaster_569: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *mut u8, *mut u32, *const *const u8, *const u8, bool, u32, *mut *mut snd_kcontrol) -> i32>, *mut ()>(Some(__snd_hda_add_vmaster)) };

/// meta hook to call each driver's vmaster hook
#[unsafe(link_section = ".text.slate_distinct.codec.vmaster_hook")]
extern "C" fn vmaster_hook(private_data: *mut (), enabled: i32) {
unsafe { unsafe { (*(private_data as *mut hda_vmaster_mute_hook)).hook }.unwrap()((*(private_data as *mut hda_vmaster_mute_hook)).codec as *mut (), enabled) };
}

/// snd_hda_add_vmaster_hook - Add a vmaster hw specific hook
/// @codec: the HDA codec
/// @hook: the vmaster hook object
///
/// Add a hw specific hook (like EAPD) with the given vmaster switch kctl.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_add_vmaster_hook")]
extern "C" fn snd_hda_add_vmaster_hook(codec: *mut hda_codec, hook: *mut hda_vmaster_mute_hook) -> i32 {
if (unsafe { (*hook).hook }) == None || unsafe { (*hook).sw_kctl }.is_null() {
return 0;
}
unsafe {
(*hook).codec = codec;
}
unsafe { snd_ctl_add_vmaster_hook((*hook).sw_kctl, Some(vmaster_hook), hook as *mut ()) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_add_vmaster_hook_570: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *mut hda_vmaster_mute_hook) -> i32>, *mut ()>(Some(snd_hda_add_vmaster_hook)) };

/// snd_hda_sync_vmaster_hook - Sync vmaster hook
/// @hook: the vmaster hook
///
/// Call the hook with the current value for synchronization.
/// Should be called in init callback.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_sync_vmaster_hook")]
extern "C" fn snd_hda_sync_vmaster_hook(hook: *mut hda_vmaster_mute_hook) {
if (unsafe { (*hook).hook }) == None || unsafe { (*hook).codec }.is_null() {
return;
}
// don't call vmaster hook in the destructor since it might have
// been already destroyed
if ((unsafe { (*(*(*hook).codec).bus).__slate_bits_0.__get_shutdown() }) as i32) != 0 {
return;
}
snd_ctl_sync_vmaster_hook!(unsafe { (*hook).sw_kctl });
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_sync_vmaster_hook_571: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_vmaster_mute_hook)>, *mut ()>(Some(snd_hda_sync_vmaster_hook)) };

/// snd_hda_mixer_amp_switch_info - Info callback for a standard AMP mixer switch
/// @kcontrol: referred ctl element
/// @uinfo: pointer to get/store the data
///
/// The control element is supposed to have the private_value field
/// set up via HDA_COMPOSE_AMP_VAL*() or related macros.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_mixer_amp_switch_info")]
extern "C" fn snd_hda_mixer_amp_switch_info(kcontrol: *mut snd_kcontrol, uinfo: *mut snd_ctl_elem_info) -> i32 {
let chs: i32 = get_amp_channels!(kcontrol) as i32;
unsafe {
(*uinfo).r#type = 1;
}
unsafe {
(*uinfo).count = (if chs == 3 { 2i32 } else { 1i32 }) as u32;
}
unsafe {
(*uinfo).value.integer.min = 0;
}
unsafe {
(*uinfo).value.integer.max = 1;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_mixer_amp_switch_info_572: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_info) -> i32>, *mut ()>(Some(snd_hda_mixer_amp_switch_info)) };

/// snd_hda_mixer_amp_switch_get - Get callback for a standard AMP mixer switch
/// @kcontrol: ctl element
/// @ucontrol: pointer to get/store the data
///
/// The control element is supposed to have the private_value field
/// set up via HDA_COMPOSE_AMP_VAL*() or related macros.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_mixer_amp_switch_get")]
extern "C" fn snd_hda_mixer_amp_switch_get(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
let codec: *mut hda_codec = snd_kcontrol_chip!(kcontrol) as *mut hda_codec;
let nid: u16 = get_amp_nid!(kcontrol) as u16;
let chs: i32 = get_amp_channels!(kcontrol) as i32;
let dir: i32 = get_amp_direction!(kcontrol) as i32;
let idx: i32 = get_amp_index!(kcontrol) as i32;
let mut valp: *mut i64 = unsafe { (*ucontrol).value.integer.value.as_mut_ptr() as *mut i64 };
if chs & 1 != 0 {
let __v0: *mut i64 = valp;
valp = __v0.wrapping_offset(1 as isize);
unsafe {
*__v0 = (if snd_hda_codec_amp_read!(codec, nid, 0i32, dir, idx) & 128 != 0 { 0i32 } else { 1i32 }) as i64;
}
}
if chs & 2 != 0 {
unsafe {
*valp = (if snd_hda_codec_amp_read!(codec, nid, 1i32, dir, idx) & 128 != 0 { 0i32 } else { 1i32 }) as i64;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_mixer_amp_switch_get_573: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_value) -> i32>, *mut ()>(Some(snd_hda_mixer_amp_switch_get)) };

/// snd_hda_mixer_amp_switch_put - Put callback for a standard AMP mixer switch
/// @kcontrol: ctl element
/// @ucontrol: pointer to get/store the data
///
/// The control element is supposed to have the private_value field
/// set up via HDA_COMPOSE_AMP_VAL*() or related macros.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_mixer_amp_switch_put")]
extern "C" fn snd_hda_mixer_amp_switch_put(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
let codec: *mut hda_codec = snd_kcontrol_chip!(kcontrol) as *mut hda_codec;
let nid: u16 = get_amp_nid!(kcontrol) as u16;
let chs: i32 = get_amp_channels!(kcontrol) as i32;
let dir: i32 = get_amp_direction!(kcontrol) as i32;
let idx: i32 = get_amp_index!(kcontrol) as i32;
let mut valp: *mut i64 = unsafe { (*ucontrol).value.integer.value.as_mut_ptr() as *mut i64 };
let mut change: i32 = 0;
if chs & 1 != 0 {
if (unsafe { *valp }) < 0 || (unsafe { *valp }) > 1 {
return 22i32.wrapping_neg();
}
change = snd_hda_codec_amp_update(codec, nid, 0i32, dir, idx, 128i32, if (unsafe { *valp }) != 0 { 0i32 } else { 128i32 });
let __v0: *mut i64 = valp;
valp = __v0.wrapping_offset(1 as isize);
}
if chs & 2 != 0 {
if (unsafe { *valp }) < 0 || (unsafe { *valp }) > 1 {
return 22i32.wrapping_neg();
}
let __v2: i32 = snd_hda_codec_amp_update(codec, nid, 1i32, dir, idx, 128i32, if (unsafe { *valp }) != 0 { 0i32 } else { 128i32 });
change |= __v2;
}
hda_call_check_power_status(codec, nid);
change
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_mixer_amp_switch_put_574: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_value) -> i32>, *mut ()>(Some(snd_hda_mixer_amp_switch_put)) };

// SPDIF out controls
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_spdif_mask_info")]
extern "C" fn snd_hda_spdif_mask_info(kcontrol: *mut snd_kcontrol, uinfo: *mut snd_ctl_elem_info) -> i32 {
unsafe {
(*uinfo).r#type = 5;
}
unsafe {
(*uinfo).count = 1;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_spdif_cmask_get")]
extern "C" fn snd_hda_spdif_cmask_get(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
unsafe {
*((*ucontrol).value.iec958.status.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = 15u8;
}
unsafe {
*((*ucontrol).value.iec958.status.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) = 255u8;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_spdif_pmask_get")]
extern "C" fn snd_hda_spdif_pmask_get(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
unsafe {
*((*ucontrol).value.iec958.status.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = 15u8;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_spdif_default_get")]
extern "C" fn snd_hda_spdif_default_get(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
let codec: *mut hda_codec = snd_kcontrol_chip!(kcontrol) as *mut hda_codec;
let idx: i32 = (unsafe { (*kcontrol).private_value }) as i32;
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*codec).spdif_out.used }) <= (idx as u32)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("575: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 575b - ., 3; .popsection\n# {0}", const 575 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_snd_hda_spdif_default_get_1, sym __slate_asm_string_snd_hda_spdif_default_get_2, const 2175 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("576: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 576b - ., 4; .popsection\n# {0}", const 576 as i32, options(nostack, att_syntax));
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
let mut __UNIQUE_ID_guard_577: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) });
let __slate_cleanup_ptr_5 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_577) as *mut _;
let __slate_cleanup_5 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_5 as *mut _));
let mut __UNIQUE_ID_unlock_578: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_6 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_578) as *mut _;
let __slate_cleanup_6 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_6 as *mut _));
let spdif: *mut hda_spdif_out = snd_array_elem(unsafe { core::ptr::addr_of_mut!((*codec).spdif_out) }, idx as u32) as *mut hda_spdif_out;
unsafe {
*((*ucontrol).value.iec958.status.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = ((*spdif).status & 255) as u8;
}
unsafe {
*((*ucontrol).value.iec958.status.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) = ((*spdif).status >> 8i32 & 255) as u8;
}
unsafe {
*((*ucontrol).value.iec958.status.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) = ((*spdif).status >> 16i32 & 255) as u8;
}
unsafe {
*((*ucontrol).value.iec958.status.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) = ((*spdif).status >> 24i32 & 255) as u8;
}
0
}

/// convert from SPDIF status bits to HDA SPDIF bits
/// bit 0 (DigEn) is always set zero (to be filled later)
fn convert_from_spdif_status(sbits: u32) -> u16 {
let mut val: u16 = 0;
if sbits & 1u32 != 0 {
val = ((val as i32) | 64i32) as u16;
}
if sbits & 2u32 != 0 {
val = ((val as i32) | 32i32) as u16;
}
if sbits & 1u32 != 0 {
if sbits & 28u32 == 12u32 {
val = ((val as i32) | 8i32) as u16;
}
} else {
if sbits & 56u32 == 8u32 {
val = ((val as i32) | 8i32) as u16;
}
if sbits & 4u32 == 0 {
val = ((val as i32) | 16i32) as u16;
}
if sbits & 32768u32 != 0 {
val = ((val as i32) | 128i32) as u16;
}
val = (((val as i32) as u32) | sbits & 32512u32) as u16;
}
val
}

/// convert to SPDIF status bits from HDA SPDIF bits
fn convert_to_spdif_status(val: u16) -> u32 {
let mut sbits: u32 = 0;
if (val as i32) & 32i32 != 0 {
sbits |= 2u32;
}
if (val as i32) & 64i32 != 0 {
sbits |= 1u32;
}
if sbits & 1u32 != 0 {
if (val as i32) & 8i32 != 0 {
sbits |= 12u32;
}
} else {
if (val as i32) & 8i32 != 0 {
sbits |= 8u32;
}
if (val as i32) & 16i32 == 0 {
sbits |= 4u32;
}
if (val as i32) & 128i32 != 0 {
sbits |= 32768u32;
}
sbits |= ((val as i32) & 32512i32) as u32;
}
sbits
}

/// set digital convert verbs both for the given NID and its followers
fn set_dig_out(codec: *mut hda_codec, nid: u16, mask: i32, val: i32) {
let mut d: *const u16 = unsafe { core::mem::zeroed() };
snd_hdac_regmap_update(unsafe { core::ptr::addr_of_mut!((*codec).core) }, nid, 1805u32, mask as u32, val as u32);
d = unsafe { (*codec).follower_dig_outs };
if d.is_null() {
return;
}
while (unsafe { *d }) != 0 {
snd_hdac_regmap_update(unsafe { core::ptr::addr_of_mut!((*codec).core) }, unsafe { *d }, 1805u32, mask as u32, val as u32);
let __v1: *const u16 = d;
d = __v1.wrapping_offset(1 as isize);
}
}

fn set_dig_out_convert(codec: *mut hda_codec, nid: u16, dig1: i32, dig2: i32) {
let mut mask: u32 = 0;
let mut val: u32 = 0;
if dig1 != 1i32.wrapping_neg() {
mask |= 255;
val = dig1 as u32;
}
if dig2 != 1i32.wrapping_neg() {
mask |= 65280;
val |= (dig2 << 8i32) as u32;
}
set_dig_out(codec, nid, mask as i32, val as i32);
}

#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_spdif_default_put")]
extern "C" fn snd_hda_spdif_default_put(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
let codec: *mut hda_codec = snd_kcontrol_chip!(kcontrol) as *mut hda_codec;
let idx: i32 = (unsafe { (*kcontrol).private_value }) as i32;
let mut nid: u16 = 0;
let mut old_status: u32 = 0;
let mut val: u16 = 0;
let mut change: i32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*codec).spdif_out.used }) <= (idx as u32)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("579: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 579b - ., 3; .popsection\n# {0}", const 579 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_snd_hda_spdif_default_put_1, sym __slate_asm_string_snd_hda_spdif_default_put_2, const 2284 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("580: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 580b - ., 4; .popsection\n# {0}", const 580 as i32, options(nostack, att_syntax));
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
let mut __UNIQUE_ID_guard_581: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) });
let __slate_cleanup_ptr_5 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_581) as *mut _;
let __slate_cleanup_5 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_5 as *mut _));
let mut __UNIQUE_ID_unlock_582: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_6 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_582) as *mut _;
let __slate_cleanup_6 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_6 as *mut _));
let spdif: *mut hda_spdif_out = snd_array_elem(unsafe { core::ptr::addr_of_mut!((*codec).spdif_out) }, idx as u32) as *mut hda_spdif_out;
nid = unsafe { (*spdif).nid };
old_status = unsafe { (*spdif).status };
unsafe {
(*spdif).status = ((*((*ucontrol).value.iec958.status.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32) as u32) | (*((*ucontrol).value.iec958.status.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) as u32) << 8i32 | (*((*ucontrol).value.iec958.status.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) as u32) << 16i32 | (*((*ucontrol).value.iec958.status.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) as u32) << 24i32;
}
val = convert_from_spdif_status(unsafe { (*spdif).status });
val = ((val as i32) | ((unsafe { (*spdif).ctls }) as i32) & 1) as u16;
change = (((unsafe { (*spdif).ctls }) as i32) != (val as i32)) as i32;
unsafe {
(*spdif).ctls = val;
}
if change != 0 && (nid as i32) != ((1i32.wrapping_neg() as u16) as i32) {
set_dig_out_convert(codec, nid, (val as i32) & 255, (val as i32) >> 8i32 & 255);
}
(change != 0 || (unsafe { (*spdif).status }) != old_status) as i32
}

#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_spdif_out_switch_get")]
extern "C" fn snd_hda_spdif_out_switch_get(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
let codec: *mut hda_codec = snd_kcontrol_chip!(kcontrol) as *mut hda_codec;
let idx: i32 = (unsafe { (*kcontrol).private_value }) as i32;
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*codec).spdif_out.used }) <= (idx as u32)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("583: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 583b - ., 3; .popsection\n# {0}", const 583 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_snd_hda_spdif_out_switch_get_1, sym __slate_asm_string_snd_hda_spdif_out_switch_get_2, const 2312 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("584: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 584b - ., 4; .popsection\n# {0}", const 584 as i32, options(nostack, att_syntax));
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
let mut __UNIQUE_ID_guard_585: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) });
let __slate_cleanup_ptr_5 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_585) as *mut _;
let __slate_cleanup_5 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_5 as *mut _));
let mut __UNIQUE_ID_unlock_586: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_6 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_586) as *mut _;
let __slate_cleanup_6 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_6 as *mut _));
let spdif: *mut hda_spdif_out = snd_array_elem(unsafe { core::ptr::addr_of_mut!((*codec).spdif_out) }, idx as u32) as *mut hda_spdif_out;
unsafe {
*((*ucontrol).value.integer.value.as_mut_ptr() as *mut i64).wrapping_offset(0 as isize) = (((*spdif).ctls as i32) & 1i32) as i64;
}
0
}

fn set_spdif_ctls(codec: *mut hda_codec, nid: u16, dig1: i32, dig2: i32) {
set_dig_out_convert(codec, nid, dig1, dig2);
// unmute amp switch (if any)
if get_wcaps(codec, nid) & 4u32 != 0 && dig1 & 1i32 != 0 {
snd_hda_codec_amp_stereo(codec, nid, 1i32, 0i32, 128i32, 0i32);
}
}

#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_spdif_out_switch_put")]
extern "C" fn snd_hda_spdif_out_switch_put(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
let codec: *mut hda_codec = snd_kcontrol_chip!(kcontrol) as *mut hda_codec;
let idx: i32 = (unsafe { (*kcontrol).private_value }) as i32;
let mut nid: u16 = 0;
let mut val: u16 = 0;
let mut change: i32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*codec).spdif_out.used }) <= (idx as u32)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("587: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 587b - ., 3; .popsection\n# {0}", const 587 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_snd_hda_spdif_out_switch_put_1, sym __slate_asm_string_snd_hda_spdif_out_switch_put_2, const 2341 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("588: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 588b - ., 4; .popsection\n# {0}", const 588 as i32, options(nostack, att_syntax));
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
let mut __UNIQUE_ID_guard_589: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) });
let __slate_cleanup_ptr_5 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_589) as *mut _;
let __slate_cleanup_5 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_5 as *mut _));
let mut __UNIQUE_ID_unlock_590: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_6 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_590) as *mut _;
let __slate_cleanup_6 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_6 as *mut _));
let spdif: *mut hda_spdif_out = snd_array_elem(unsafe { core::ptr::addr_of_mut!((*codec).spdif_out) }, idx as u32) as *mut hda_spdif_out;
nid = unsafe { (*spdif).nid };
val = (((unsafe { (*spdif).ctls }) as i32) & -2i32) as u16;
if (unsafe { *((*ucontrol).value.integer.value.as_mut_ptr() as *mut i64).wrapping_offset(0 as isize) }) != 0 {
val = ((val as i32) | 1i32) as u16;
}
change = (((unsafe { (*spdif).ctls }) as i32) != (val as i32)) as i32;
unsafe {
(*spdif).ctls = val;
}
if change != 0 && (nid as i32) != ((1i32.wrapping_neg() as u16) as i32) {
set_spdif_ctls(codec, nid, (val as i32) & 255, 1i32.wrapping_neg());
}
change
}

/// end
static dig_mixes: __SlateAlign16<[snd_kcontrol_new; 5]> = __SlateAlign16([{
let mut __t0: snd_kcontrol_new = unsafe { core::mem::zeroed() };
__t0.iface = 2i32;
__t0.name = (b"IEC958 Playback Con Mask\0".as_ptr() as *mut u8) as *const u8;
__t0.access = 1u32;
__t0.info = Some(snd_hda_spdif_mask_info);
__t0.get = Some(snd_hda_spdif_cmask_get);
__t0
}, {
let mut __t1: snd_kcontrol_new = unsafe { core::mem::zeroed() };
__t1.iface = 2i32;
__t1.name = (b"IEC958 Playback Pro Mask\0".as_ptr() as *mut u8) as *const u8;
__t1.access = 1u32;
__t1.info = Some(snd_hda_spdif_mask_info);
__t1.get = Some(snd_hda_spdif_pmask_get);
__t1
}, {
let mut __t2: snd_kcontrol_new = unsafe { core::mem::zeroed() };
__t2.iface = 2i32;
__t2.name = (b"IEC958 Playback Default\0".as_ptr() as *mut u8) as *const u8;
__t2.info = Some(snd_hda_spdif_mask_info);
__t2.get = Some(snd_hda_spdif_default_get);
__t2.put = Some(snd_hda_spdif_default_put);
__t2
}, {
let mut __t3: snd_kcontrol_new = unsafe { core::mem::zeroed() };
__t3.iface = 2i32;
__t3.name = (b"IEC958 Playback Switch\0".as_ptr() as *mut u8) as *const u8;
__t3.info = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_info) -> i32>>(snd_ctl_boolean_mono_info as *const ()) };
__t3.get = Some(snd_hda_spdif_out_switch_get);
__t3.put = Some(snd_hda_spdif_out_switch_put);
__t3
}, {
let mut __t4: snd_kcontrol_new = unsafe { core::mem::zeroed() };
__t4
}]);

/// snd_hda_create_dig_out_ctls - create Output SPDIF-related controls
/// @codec: the HDA codec
/// @associated_nid: NID that new ctls associated with
/// @cvt_nid: converter NID
/// @type: HDA_PCM_TYPE_*
/// Creates controls related with the digital output.
/// Called from each codec driver supporting the digital out.
///
/// Returns 0 if successful, or a negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_create_dig_out_ctls")]
extern "C" fn snd_hda_create_dig_out_ctls(codec: *mut hda_codec, associated_nid: u16, cvt_nid: u16, r#type: i32) -> i32 {
let mut err: i32 = 0;
let mut kctl: *mut snd_kcontrol = unsafe { core::mem::zeroed() };
let mut dig_mix: *const snd_kcontrol_new = unsafe { core::mem::zeroed() };
let mut idx: i32 = 0;
let mut val: i32 = 0;
let spdif_index: i32 = 16;
let bus: *mut hda_bus = unsafe { (*codec).bus };
if (unsafe { (*bus).primary_dig_out_type }) == 2 && r#type == 1 {
idx = spdif_index;
} else {
if (unsafe { (*bus).primary_dig_out_type }) == 1 && r#type == 2 {
// suppose a single SPDIF device
dig_mix = unsafe { core::ptr::addr_of!(dig_mixes.0) as *const snd_kcontrol_new };
while !unsafe { (*dig_mix).name }.is_null() {
let mut id: snd_ctl_elem_id = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(id) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
kctl = find_mixer_ctl(codec, unsafe { (*dig_mix).name }, 0i32, 0i32);
if kctl.is_null() {
break;
}
id = unsafe { (*kctl).id };
id.index = spdif_index as u32;
err = unsafe { snd_ctl_rename_id((*codec).card, core::ptr::addr_of_mut!((*kctl).id), core::ptr::addr_of_mut!(id)) };
if err < 0 {
return err;
}
let __v1: *const snd_kcontrol_new = dig_mix;
dig_mix = __v1.wrapping_offset(1 as isize);
}
unsafe {
(*bus).primary_dig_out_type = 2;
}
}
}
if (unsafe { (*bus).primary_dig_out_type }) == 0 {
unsafe {
(*bus).primary_dig_out_type = r#type;
}
}
idx = find_empty_mixer_ctl_idx(codec, c"IEC958 Playback Switch".as_ptr() as *const u8, idx);
if idx < 0 {
while false {
}
unsafe { _dev_err(hda_codec_dev!(codec) as *const device, c"too many IEC958 outputs\n".as_ptr() as *const u8) };
return 16i32.wrapping_neg();
}
let spdif: *mut hda_spdif_out = (unsafe { snd_array_new(core::ptr::addr_of_mut!((*codec).spdif_out)) }) as *mut hda_spdif_out;
if spdif.is_null() {
return 12i32.wrapping_neg();
}
dig_mix = unsafe { core::ptr::addr_of!(dig_mixes.0) as *const snd_kcontrol_new };
while !unsafe { (*dig_mix).name }.is_null() {
kctl = unsafe { snd_ctl_new1(dig_mix, codec as *mut ()) };
if kctl.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*kctl).id.index = idx as u32;
}
unsafe {
(*kctl).private_value = unsafe { (*codec).spdif_out.used }.wrapping_sub(1) as u64;
}
err = snd_hda_ctl_add(codec, associated_nid, kctl);
if err < 0 {
return err;
}
let __v5: *const snd_kcontrol_new = dig_mix;
dig_mix = __v5.wrapping_offset(1 as isize);
}
unsafe {
(*spdif).nid = cvt_nid;
}
snd_hdac_regmap_read(unsafe { core::ptr::addr_of_mut!((*codec).core) }, cvt_nid, 3853u32, core::ptr::addr_of_mut!(val) as *mut u32);
unsafe {
(*spdif).ctls = val as u16;
}
unsafe {
(*spdif).status = convert_to_spdif_status((*spdif).ctls);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_create_dig_out_ctls_591: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, u16, i32) -> i32>, *mut ()>(Some(snd_hda_create_dig_out_ctls)) };

/// snd_hda_spdif_out_of_nid - get the hda_spdif_out entry from the given NID
/// @codec: the HDA codec
/// @nid: widget NID
///
/// call within spdif_mutex lock
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_spdif_out_of_nid")]
extern "C" fn snd_hda_spdif_out_of_nid(codec: *mut hda_codec, nid: u16) -> *mut hda_spdif_out {
let mut spdif: *mut hda_spdif_out = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let __v0: *mut hda_spdif_out = (unsafe { (*core::ptr::addr_of_mut!((*codec).spdif_out)).list }) as *mut hda_spdif_out;
spdif = __v0;
while (i as u32) < unsafe { (*core::ptr::addr_of_mut!((*codec).spdif_out)).used } {
if ((unsafe { (*spdif).nid }) as i32) == (nid as i32) {
return spdif;
}
let __v3: i32 = i.wrapping_add(1);
i = __v3;
spdif = snd_array_elem(unsafe { core::ptr::addr_of_mut!((*codec).spdif_out) }, __v3 as u32) as *mut hda_spdif_out;
}
core::ptr::null_mut::<hda_spdif_out>()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_spdif_out_of_nid_592: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16) -> *mut hda_spdif_out>, *mut ()>(Some(snd_hda_spdif_out_of_nid)) };

/// snd_hda_spdif_ctls_unassign - Unassign the given SPDIF ctl
/// @codec: the HDA codec
/// @idx: the SPDIF ctl index
///
/// Unassign the widget from the given SPDIF control.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_spdif_ctls_unassign")]
extern "C" fn snd_hda_spdif_ctls_unassign(codec: *mut hda_codec, idx: i32) {
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*codec).spdif_out.used }) <= (idx as u32)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("593: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 593b - ., 3; .popsection\n# {0}", const 593 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_snd_hda_spdif_ctls_unassign_1, sym __slate_asm_string_snd_hda_spdif_ctls_unassign_2, const 2495 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("594: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 594b - ., 4; .popsection\n# {0}", const 594 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return;
}
let mut __UNIQUE_ID_guard_595: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) });
let __slate_cleanup_ptr_5 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_595) as *mut _;
let __slate_cleanup_5 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_5 as *mut _));
let mut __UNIQUE_ID_unlock_596: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_6 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_596) as *mut _;
let __slate_cleanup_6 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_6 as *mut _));
let spdif: *mut hda_spdif_out = snd_array_elem(unsafe { core::ptr::addr_of_mut!((*codec).spdif_out) }, idx as u32) as *mut hda_spdif_out;
unsafe {
(*spdif).nid = 1i32.wrapping_neg() as u16;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_spdif_ctls_unassign_597: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, i32)>, *mut ()>(Some(snd_hda_spdif_ctls_unassign)) };

/// snd_hda_spdif_ctls_assign - Assign the SPDIF controls to the given NID
/// @codec: the HDA codec
/// @idx: the SPDIF ctl idx
/// @nid: widget NID
///
/// Assign the widget to the SPDIF control with the given index.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_spdif_ctls_assign")]
extern "C" fn snd_hda_spdif_ctls_assign(codec: *mut hda_codec, idx: i32, nid: u16) {
let mut val: u16 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*codec).spdif_out.used }) <= (idx as u32)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("598: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 598b - ., 3; .popsection\n# {0}", const 598 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_snd_hda_spdif_ctls_assign_1, sym __slate_asm_string_snd_hda_spdif_ctls_assign_2, const 2516 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("599: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 599b - ., 4; .popsection\n# {0}", const 599 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return;
}
let mut __UNIQUE_ID_guard_600: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) });
let __slate_cleanup_ptr_5 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_600) as *mut _;
let __slate_cleanup_5 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_5 as *mut _));
let mut __UNIQUE_ID_unlock_601: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_6 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_601) as *mut _;
let __slate_cleanup_6 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_6 as *mut _));
let spdif: *mut hda_spdif_out = snd_array_elem(unsafe { core::ptr::addr_of_mut!((*codec).spdif_out) }, idx as u32) as *mut hda_spdif_out;
if ((unsafe { (*spdif).nid }) as i32) != (nid as i32) {
unsafe {
(*spdif).nid = nid;
}
val = unsafe { (*spdif).ctls };
set_spdif_ctls(codec, nid, (val as i32) & 255, (val as i32) >> 8i32 & 255);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_spdif_ctls_assign_602: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, i32, u16)>, *mut ()>(Some(snd_hda_spdif_ctls_assign)) };

/// SPDIF sharing with analog output
#[unsafe(link_section = ".text.slate_distinct.codec.spdif_share_sw_get")]
extern "C" fn spdif_share_sw_get(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
let codec: *mut hda_codec = snd_kcontrol_chip!(kcontrol) as *mut hda_codec;
let mout: *mut hda_multi_out = ((unsafe { (*kcontrol).private_value }) as *mut ()) as *mut hda_multi_out;
let mut __UNIQUE_ID_guard_603: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_603) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_604: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_604) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
unsafe {
*((*ucontrol).value.integer.value.as_mut_ptr() as *mut i64).wrapping_offset(0 as isize) = (*mout).share_spdif as i64;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.codec.spdif_share_sw_put")]
extern "C" fn spdif_share_sw_put(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
let codec: *mut hda_codec = snd_kcontrol_chip!(kcontrol) as *mut hda_codec;
let mout: *mut hda_multi_out = ((unsafe { (*kcontrol).private_value }) as *mut ()) as *mut hda_multi_out;
let val: bool = (unsafe { *((*ucontrol).value.integer.value.as_mut_ptr() as *mut i64).wrapping_offset(0 as isize) }) != 0;
let mut change: i32 = 0;
let mut __UNIQUE_ID_guard_605: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_605) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_606: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_606) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
change = ((unsafe { (*mout).share_spdif }) != (val as i32)) as i32;
unsafe {
(*mout).share_spdif = val as i32;
}
change
}

static spdif_share_sw: __SlateConst<snd_kcontrol_new> = __SlateConst({
let mut __t0: snd_kcontrol_new = unsafe { core::mem::zeroed() };
__t0.iface = 2i32;
__t0.name = (b"IEC958 Default PCM Playback Switch\0".as_ptr() as *mut u8) as *const u8;
__t0.info = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_info) -> i32>>(snd_ctl_boolean_mono_info as *const ()) };
__t0.get = Some(spdif_share_sw_get);
__t0.put = Some(spdif_share_sw_put);
__t0
});

fn notify_spdif_share_sw(codec: *mut hda_codec, mout: *mut hda_multi_out) {
if !unsafe { (*mout).share_spdif_kctl }.is_null() {
unsafe { snd_ctl_notify_one((*codec).card, 1u32, (*mout).share_spdif_kctl, 0u32) };
}
}

/// snd_hda_create_spdif_share_sw - create Default PCM switch
/// @codec: the HDA codec
/// @mout: multi-out instance
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_create_spdif_share_sw")]
extern "C" fn snd_hda_create_spdif_share_sw(codec: *mut hda_codec, mout: *mut hda_multi_out) -> i32 {
let mut err: i32 = 0;
if (unsafe { (*mout).dig_out_nid }) == 0 {
return 0;
}
let kctl: *mut snd_kcontrol = unsafe { snd_ctl_new1(core::ptr::addr_of!(spdif_share_sw.0), codec as *mut ()) };
if kctl.is_null() {
return 12i32.wrapping_neg();
}
// snd_ctl_new1() stores @codec in private_data; stash @mout in
// private_value for the share-switch callbacks and cache the
// assigned control for forced-disable notifications.
unsafe {
(*kctl).private_value = mout as u64;
}
err = snd_hda_ctl_add(codec, unsafe { (*mout).dig_out_nid }, kctl);
if err < 0 {
return err;
}
unsafe {
(*mout).share_spdif_kctl = kctl;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_create_spdif_share_sw_607: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *mut hda_multi_out) -> i32>, *mut ()>(Some(snd_hda_create_spdif_share_sw)) };

// SPDIF input
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_spdif_in_switch_get")]
extern "C" fn snd_hda_spdif_in_switch_get(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
let codec: *mut hda_codec = snd_kcontrol_chip!(kcontrol) as *mut hda_codec;
unsafe {
*((*ucontrol).value.integer.value.as_mut_ptr() as *mut i64).wrapping_offset(0 as isize) = (*codec).spdif_in_enable as i64;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_spdif_in_switch_put")]
extern "C" fn snd_hda_spdif_in_switch_put(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
let codec: *mut hda_codec = snd_kcontrol_chip!(kcontrol) as *mut hda_codec;
let nid: u16 = (unsafe { (*kcontrol).private_value }) as u16;
let val: u32 = ((unsafe { *((*ucontrol).value.integer.value.as_mut_ptr() as *mut i64).wrapping_offset(0 as isize) }) != 0) as u32;
let mut change: i32 = 0;
let mut __UNIQUE_ID_guard_608: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_608) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_609: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_609) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
change = ((unsafe { (*codec).spdif_in_enable }) != val) as i32;
if change != 0 {
unsafe {
(*codec).spdif_in_enable = val;
}
snd_hdac_regmap_write(unsafe { core::ptr::addr_of_mut!((*codec).core) }, nid, 1805u32, val);
}
change
}

#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_spdif_in_status_get")]
extern "C" fn snd_hda_spdif_in_status_get(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
let codec: *mut hda_codec = snd_kcontrol_chip!(kcontrol) as *mut hda_codec;
let nid: u16 = (unsafe { (*kcontrol).private_value }) as u16;
let mut val: u32 = 0;
let mut sbits: u32 = 0;
snd_hdac_regmap_read(unsafe { core::ptr::addr_of_mut!((*codec).core) }, nid, 3853u32, core::ptr::addr_of_mut!(val));
sbits = convert_to_spdif_status(val as u16);
unsafe {
*((*ucontrol).value.iec958.status.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = sbits as u8;
}
unsafe {
*((*ucontrol).value.iec958.status.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) = (sbits >> 8i32) as u8;
}
unsafe {
*((*ucontrol).value.iec958.status.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) = (sbits >> 16i32) as u8;
}
unsafe {
*((*ucontrol).value.iec958.status.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) = (sbits >> 24i32) as u8;
}
0
}

/// end
static dig_in_ctls: __SlateAlign16<[snd_kcontrol_new; 3]> = __SlateAlign16([{
let mut __t0: snd_kcontrol_new = unsafe { core::mem::zeroed() };
__t0.iface = 2i32;
__t0.name = (b"IEC958 Capture Switch\0".as_ptr() as *mut u8) as *const u8;
__t0.info = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_info) -> i32>>(snd_ctl_boolean_mono_info as *const ()) };
__t0.get = Some(snd_hda_spdif_in_switch_get);
__t0.put = Some(snd_hda_spdif_in_switch_put);
__t0
}, {
let mut __t1: snd_kcontrol_new = unsafe { core::mem::zeroed() };
__t1.iface = 2i32;
__t1.name = (b"IEC958 Capture Default\0".as_ptr() as *mut u8) as *const u8;
__t1.access = 1u32;
__t1.info = Some(snd_hda_spdif_mask_info);
__t1.get = Some(snd_hda_spdif_in_status_get);
__t1
}, {
let mut __t2: snd_kcontrol_new = unsafe { core::mem::zeroed() };
__t2
}]);

/// snd_hda_create_spdif_in_ctls - create Input SPDIF-related controls
/// @codec: the HDA codec
/// @nid: audio in widget NID
///
/// Creates controls related with the SPDIF input.
/// Called from each codec driver supporting the SPDIF in.
///
/// Returns 0 if successful, or a negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_create_spdif_in_ctls")]
extern "C" fn snd_hda_create_spdif_in_ctls(codec: *mut hda_codec, nid: u16) -> i32 {
let mut err: i32 = 0;
let mut dig_mix: *const snd_kcontrol_new = unsafe { core::mem::zeroed() };
let idx: i32 = find_empty_mixer_ctl_idx(codec, c"IEC958 Capture Switch".as_ptr() as *const u8, 0i32);
if idx < 0 {
while false {
}
unsafe { _dev_err(hda_codec_dev!(codec) as *const device, c"too many IEC958 inputs\n".as_ptr() as *const u8) };
return 16i32.wrapping_neg();
}
dig_mix = unsafe { core::ptr::addr_of!(dig_in_ctls.0) as *const snd_kcontrol_new };
while !unsafe { (*dig_mix).name }.is_null() {
let kctl: *mut snd_kcontrol = unsafe { snd_ctl_new1(dig_mix, codec as *mut ()) };
if kctl.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*kctl).private_value = nid as u64;
}
err = snd_hda_ctl_add(codec, nid, kctl);
if err < 0 {
return err;
}
let __v2: *const snd_kcontrol_new = dig_mix;
dig_mix = __v2.wrapping_offset(1 as isize);
}
unsafe {
(*codec).spdif_in_enable = snd_hda_codec_read(codec, nid, 0i32, 3853u32, 0u32) & 1u32;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_create_spdif_in_ctls_610: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16) -> i32>, *mut ()>(Some(snd_hda_create_spdif_in_ctls)) };

/// snd_hda_codec_set_power_to_all - Set the power state to all widgets
/// @codec: the HDA codec
/// @fg: function group (not used now)
/// @power_state: the power state to set (AC_PWRST_*)
///
/// Set the given power state to all widgets that have the power control.
/// If the codec has power_filter set, it evaluates the power state and
/// filter out if it's unchanged as D3.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_set_power_to_all")]
extern "C" fn snd_hda_codec_set_power_to_all(codec: *mut hda_codec, fg: u16, power_state: u32) {
let mut nid: u16 = unsafe { (*codec).core.start_nid };
while (nid as i32) < ((unsafe { (*codec).core.end_nid }) as i32) {
'__slate_continue_0: {
let wcaps: u32 = get_wcaps(codec, nid);
let mut state: u32 = power_state;
if wcaps & 1024u32 != 0 {
if (unsafe { (*codec).power_filter }) != None {
state = unsafe { unsafe { (*codec).power_filter }.unwrap()(codec, nid, power_state) };
if state != power_state && power_state == 3 {
break '__slate_continue_0;
}
}
snd_hda_codec_write(codec, nid, 0i32, 1797u32, state);
}
}
nid = (nid as i32).wrapping_add(1) as u16;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_set_power_to_all_611: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, u32)>, *mut ()>(Some(snd_hda_codec_set_power_to_all)) };

/// snd_hda_codec_eapd_power_filter - A power filter callback for EAPD
/// @codec: the HDA codec
/// @nid: widget NID
/// @power_state: power state to evalue
///
/// Don't power down the widget if it controls eapd and EAPD_BTLENABLE is set.
/// This can be used a codec power_filter callback.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_eapd_power_filter")]
extern "C" fn snd_hda_codec_eapd_power_filter(codec: *mut hda_codec, nid: u16, power_state: u32) -> u32 {
if (nid as i32) == ((unsafe { (*codec).core.afg }) as i32) || (nid as i32) == ((unsafe { (*codec).core.mfg }) as i32) {
return power_state;
}
let __v0: bool = power_state == 3 && get_wcaps_type(get_wcaps(codec, nid)) == 4;
let __v1: bool = __v0 && snd_hda_query_pin_caps(codec, nid) & 65536u32 != 0;
if __v1 {
let eapd: i32 = snd_hda_codec_read(codec, nid, 0i32, 3852u32, 0u32) as i32;
if eapd & 2 != 0 {
return 0;
}
}
power_state
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_eapd_power_filter_612: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, u32) -> u32>, *mut ()>(Some(snd_hda_codec_eapd_power_filter)) };

/// set power state of the codec, and return the power state
fn hda_set_power_state(codec: *mut hda_codec, power_state: u32) -> u32 {
let driver: *mut hda_codec_driver = hda_codec_to_driver!(codec);
let fg: u16 = (if (unsafe { (*codec).core.afg }) != 0 { (unsafe { (*codec).core.afg }) as i32 } else { (unsafe { (*codec).core.mfg }) as i32 }) as u16;
let mut state: u32 = 0;
let mut flags: i32 = 0;
// this delay seems necessary to avoid click noise at power-down
if power_state == 3 {
if (unsafe { (*codec).depop_delay }) < 0 {
unsafe { msleep((if codec_has_epss!(codec) != 0 { 10i32 } else { 100i32 }) as u32) };
} else {
if (unsafe { (*codec).depop_delay }) > 0 {
unsafe { msleep((*codec).depop_delay as u32) };
}
}
flags = 1i32;
}
// repeat power states setting at most 10 times
for count in 0i32..10i32 {
// might be called before binding to driver, too
if !driver.is_null() && !unsafe { (*driver).ops }.is_null() && (unsafe { (*(*driver).ops).set_power_state }) != None {
unsafe { unsafe { (*(*driver).ops).set_power_state }.unwrap()(codec, fg, power_state) };
} else {
state = power_state;
if (unsafe { (*codec).power_filter }) != None {
state = unsafe { unsafe { (*codec).power_filter }.unwrap()(codec, fg, state) };
}
if state == power_state || power_state != 3 {
snd_hda_codec_write_sync(codec, fg, flags, 1797u32, state);
}
snd_hda_codec_set_power_to_all(codec, fg, power_state);
}
state = snd_hda_sync_power_state(codec, fg, power_state);
if state & 256u32 == 0 {
break;
}
}
state
}

/// sync power states of all widgets;
/// this is called at the end of codec parsing
fn sync_power_up_states(codec: *mut hda_codec) {
let mut nid: u16 = 0;
// don't care if no filter is used
if (unsafe { (*codec).power_filter }) == None {
return;
}
nid = unsafe { (*codec).core.start_nid };
while (nid as i32) < ((unsafe { (*codec).core.end_nid }) as i32) {
let wcaps: u32 = get_wcaps(codec, nid);
let mut target: u32 = 0;
if wcaps & 1024u32 != 0 {
target = unsafe { unsafe { (*codec).power_filter }.unwrap()(codec, nid, 0u32) };
if target != 0 {
if !snd_hda_check_power_state(codec, nid, target) {
snd_hda_codec_write(codec, nid, 0i32, 1797u32, target);
}
}
}
nid = (nid as i32).wrapping_add(1) as u16;
}
}

fn hda_exec_init_verbs(codec: *mut hda_codec) {
}

/// update the power on/off account with the current jiffies
fn update_power_acct(codec: *mut hda_codec, on: bool) {
let delta: u64 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }.wrapping_sub(unsafe { (*codec).power_jiffies });
if on {
let __v1: u64 = unsafe { (*codec).power_on_acct };
unsafe {
(*codec).power_on_acct = __v1.wrapping_add(delta);
}
} else {
let __v4: u64 = unsafe { (*codec).power_off_acct };
unsafe {
(*codec).power_off_acct = __v4.wrapping_add(delta);
}
}
let __v7: u64 = unsafe { (*codec).power_jiffies };
unsafe {
(*codec).power_jiffies = __v7.wrapping_add(delta);
}
}

#[unsafe(no_mangle)]
extern "C" fn snd_hda_update_power_acct(codec: *mut hda_codec) {
update_power_acct(codec, hda_codec_is_power_on!(codec));
}

/// call suspend and power-down; used both from PM and power-save
/// this function returns the power state in the end
fn hda_call_codec_suspend(codec: *mut hda_codec) -> u32 {
let driver: *mut hda_codec_driver = hda_codec_to_driver!(codec);
let mut state: u32 = 0;
snd_hdac_enter_pm(unsafe { core::ptr::addr_of_mut!((*codec).core) });
if (unsafe { (*(*driver).ops).suspend }) != None {
unsafe { unsafe { (*(*driver).ops).suspend }.unwrap()(codec) };
}
if ((unsafe { (*codec).__slate_bits_0.__get_no_stream_clean_at_suspend() }) as i32) == 0 {
hda_cleanup_all_streams(codec);
}
state = hda_set_power_state(codec, 3u32);
update_power_acct(codec, true);
snd_hdac_leave_pm(unsafe { core::ptr::addr_of_mut!((*codec).core) });
state
}

/// kick up codec; used both from PM and power-save
fn hda_call_codec_resume(codec: *mut hda_codec) {
let mut __slate_compound_1: pm_message = unsafe { core::mem::zeroed() };
let driver: *mut hda_codec_driver = hda_codec_to_driver!(codec);
snd_hdac_enter_pm(unsafe { core::ptr::addr_of_mut!((*codec).core) });
if !unsafe { (*codec).core.regmap }.is_null() {
unsafe { regcache_mark_dirty((*codec).core.regmap) };
}
unsafe {
(*codec).power_jiffies = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
}
hda_set_power_state(codec, 0u32);
restore_shutup_pins(codec);
hda_exec_init_verbs(codec);
unsafe { snd_hda_jack_set_dirty_all(codec) };
if (unsafe { (*(*driver).ops).resume }) != None {
unsafe { unsafe { (*(*driver).ops).resume }.unwrap()(codec) };
} else {
snd_hda_codec_init(codec);
snd_hda_regmap_sync!(codec);
}
unsafe { snd_hda_jack_report_sync(codec) };
unsafe {
(*codec).core.dev.power.power_state = *{
__slate_compound_1 = pm_message { event: 0 };
core::ptr::addr_of_mut!(__slate_compound_1)
};
}
snd_hdac_leave_pm(unsafe { core::ptr::addr_of_mut!((*codec).core) });
if (unsafe { (*codec).jackpoll_interval }) != 0 {
schedule_delayed_work(unsafe { core::ptr::addr_of_mut!((*codec).jackpoll_work) }, unsafe { (*codec).jackpoll_interval });
}
}

#[unsafe(link_section = ".text.slate_distinct.codec.hda_codec_runtime_suspend")]
extern "C" fn hda_codec_runtime_suspend(dev: *mut device) -> i32 {
let codec: *mut hda_codec = dev_to_hda_codec!(dev);
let mut state: u32 = 0;
// Nothing to do if card registration fails and the component driver never probes
if unsafe { (*codec).card }.is_null() {
return 0;
}
state = hda_call_codec_suspend(codec);
if ((unsafe { (*codec).__slate_bits_0.__get_link_down_at_suspend() }) as i32) != 0 || codec_has_clkstop!(codec) != 0 && codec_has_epss!(codec) != 0 && state & 512u32 != 0 {
unsafe { snd_hdac_codec_link_down(core::ptr::addr_of_mut!((*codec).core)) };
}
snd_hda_codec_display_power(codec, false);
0
}

#[unsafe(link_section = ".text.slate_distinct.codec.hda_codec_runtime_resume")]
extern "C" fn hda_codec_runtime_resume(dev: *mut device) -> i32 {
let codec: *mut hda_codec = dev_to_hda_codec!(dev);
// Nothing to do if card registration fails and the component driver never probes
if unsafe { (*codec).card }.is_null() {
return 0;
}
snd_hda_codec_display_power(codec, true);
unsafe { snd_hdac_codec_link_up(core::ptr::addr_of_mut!((*codec).core)) };
hda_call_codec_resume(codec);
pm_runtime_mark_last_busy(dev);
0
}

#[unsafe(link_section = ".text.slate_distinct.codec.hda_codec_runtime_idle")]
extern "C" fn hda_codec_runtime_idle(dev: *mut device) -> i32 {
let codec: *mut hda_codec = dev_to_hda_codec!(dev);
if (unsafe { (*codec).jackpoll_interval }) != 0 && ((unsafe { (*(*codec).bus).__slate_bits_0.__get_jackpoll_in_suspend() }) as i32) == 0 {
return 16i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.codec.hda_codec_pm_prepare")]
extern "C" fn hda_codec_pm_prepare(dev: *mut device) -> i32 {
let mut __slate_compound_1: pm_message = unsafe { core::mem::zeroed() };
let codec: *mut hda_codec = dev_to_hda_codec!(dev);
unsafe { cancel_delayed_work_sync(core::ptr::addr_of_mut!((*codec).jackpoll_work)) };
unsafe {
(*dev).power.power_state = *{
__slate_compound_1 = pm_message { event: 2 };
core::ptr::addr_of_mut!(__slate_compound_1)
};
}
pm_runtime_suspended(dev) as i32
}

#[unsafe(link_section = ".text.slate_distinct.codec.hda_codec_pm_complete")]
extern "C" fn hda_codec_pm_complete(dev: *mut device) {
let mut __slate_compound_1: pm_message = unsafe { core::mem::zeroed() };
let codec: *mut hda_codec = dev_to_hda_codec!(dev);
// If no other pm-functions are called between prepare() and complete()
if (unsafe { (*dev).power.power_state.event }) == 2 {
unsafe {
(*dev).power.power_state = *{
__slate_compound_1 = pm_message { event: 16 };
core::ptr::addr_of_mut!(__slate_compound_1)
};
}
}
let __v2: bool = if pm_runtime_suspended(dev) {
let __v3: bool = (unsafe { (*codec).jackpoll_interval }) != 0 || hda_codec_need_resume(codec);
__v3 || ((unsafe { (*codec).__slate_bits_0.__get_forced_resume() }) as i32) != 0 || ((unsafe { (*codec).__slate_bits_0.__get_acomp_requested_resume() }) as i32) != 0
} else {
false
};
if __v2 {
unsafe {
(*codec).__slate_bits_0.__set_acomp_requested_resume(0u32);
}
pm_request_resume(dev);
}
}

#[unsafe(link_section = ".text.slate_distinct.codec.hda_codec_pm_suspend")]
extern "C" fn hda_codec_pm_suspend(dev: *mut device) -> i32 {
let mut __slate_compound_0: pm_message = unsafe { core::mem::zeroed() };
unsafe {
(*dev).power.power_state = *{
__slate_compound_0 = pm_message { event: 2 };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
unsafe { pm_runtime_force_suspend(dev) }
}

#[unsafe(link_section = ".text.slate_distinct.codec.hda_codec_pm_resume")]
extern "C" fn hda_codec_pm_resume(dev: *mut device) -> i32 {
let mut __slate_compound_0: pm_message = unsafe { core::mem::zeroed() };
unsafe {
(*dev).power.power_state = *{
__slate_compound_0 = pm_message { event: 16 };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
unsafe { pm_runtime_force_resume(dev) }
}

#[unsafe(link_section = ".text.slate_distinct.codec.hda_codec_pm_freeze")]
extern "C" fn hda_codec_pm_freeze(dev: *mut device) -> i32 {
let mut __slate_compound_1: pm_message = unsafe { core::mem::zeroed() };
let codec: *mut hda_codec = dev_to_hda_codec!(dev);
unsafe { cancel_delayed_work_sync(core::ptr::addr_of_mut!((*codec).jackpoll_work)) };
unsafe {
(*dev).power.power_state = *{
__slate_compound_1 = pm_message { event: 1 };
core::ptr::addr_of_mut!(__slate_compound_1)
};
}
unsafe { pm_runtime_force_suspend(dev) }
}

#[unsafe(link_section = ".text.slate_distinct.codec.hda_codec_pm_thaw")]
extern "C" fn hda_codec_pm_thaw(dev: *mut device) -> i32 {
let mut __slate_compound_0: pm_message = unsafe { core::mem::zeroed() };
unsafe {
(*dev).power.power_state = *{
__slate_compound_0 = pm_message { event: 32 };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
unsafe { pm_runtime_force_resume(dev) }
}

#[unsafe(link_section = ".text.slate_distinct.codec.hda_codec_pm_restore")]
extern "C" fn hda_codec_pm_restore(dev: *mut device) -> i32 {
let mut __slate_compound_0: pm_message = unsafe { core::mem::zeroed() };
unsafe {
(*dev).power.power_state = *{
__slate_compound_0 = pm_message { event: 64 };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
unsafe { pm_runtime_force_resume(dev) }
}

/// referred in hda_bind.c
#[unsafe(no_mangle)]
static hda_codec_driver_pm: __SlateConst<dev_pm_ops> = __SlateConst({
let mut __t0: dev_pm_ops = unsafe { core::mem::zeroed() };
__t0.prepare = Some(hda_codec_pm_prepare);
__t0.complete = Some(hda_codec_pm_complete);
__t0.suspend = Some(hda_codec_pm_suspend);
__t0.resume = Some(hda_codec_pm_resume);
__t0.freeze = Some(hda_codec_pm_freeze);
__t0.thaw = Some(hda_codec_pm_thaw);
__t0.poweroff = Some(hda_codec_pm_suspend);
__t0.restore = Some(hda_codec_pm_restore);
__t0.runtime_suspend = Some(hda_codec_runtime_suspend);
__t0.runtime_resume = Some(hda_codec_runtime_resume);
__t0.runtime_idle = Some(hda_codec_runtime_idle);
__t0
});

/// suspend the codec at shutdown; called from driver's shutdown callback
#[unsafe(no_mangle)]
extern "C" fn snd_hda_codec_shutdown(codec: *mut hda_codec) {
let mut cpcm: *mut hda_pcm = unsafe { core::mem::zeroed() };
// Skip the shutdown if codec is not registered
if ((unsafe { (*codec).core.__slate_bits_1.__get_registered() }) as i32) == 0 {
return;
}
unsafe {
(*codec).jackpoll_interval = 0;
}
// don't poll any longer
unsafe { cancel_delayed_work_sync(core::ptr::addr_of_mut!((*codec).jackpoll_work)) };
cpcm = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*codec).pcm_list_head) });
while list_entry_is_head!(cpcm, unsafe { core::ptr::addr_of_mut!((*codec).pcm_list_head) }, list) == 0 {
unsafe { snd_pcm_suspend_all((*cpcm).pcm) };
cpcm = list_next_entry__4!(cpcm, list);
}
unsafe { pm_runtime_force_suspend(hda_codec_dev!(codec)) };
pm_runtime_disable(hda_codec_dev!(codec));
}

/// add standard channel maps if not specified
fn add_std_chmaps(codec: *mut hda_codec) -> i32 {
let mut pcm: *mut hda_pcm = unsafe { core::mem::zeroed() };
let mut err: i32 = 0;
pcm = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*codec).pcm_list_head) });
while list_entry_is_head!(pcm, unsafe { core::ptr::addr_of_mut!((*codec).pcm_list_head) }, list) == 0 {
for str in 0i32..2i32 {
let hinfo: *mut hda_pcm_stream = unsafe { (*pcm).stream.as_mut_ptr() as *mut hda_pcm_stream }.wrapping_offset(str as isize);
let mut chmap: *mut snd_pcm_chmap = unsafe { core::mem::zeroed() };
if !(unsafe { (*pcm).pcm }.is_null() || unsafe { (*pcm).own_chmap } || (unsafe { (*hinfo).substreams }) == 0) {
let elem: *const snd_pcm_chmap_elem = if !unsafe { (*hinfo).chmap }.is_null() { unsafe { (*hinfo).chmap } } else { unsafe { core::ptr::addr_of!(snd_pcm_std_chmaps) as *const snd_pcm_chmap_elem } };
err = unsafe { snd_pcm_add_chmap_ctls((*pcm).pcm, str, elem, (*hinfo).channels_max as i32, 0u64, core::ptr::addr_of_mut!(chmap)) };
if err < 0 {
return err;
}
unsafe {
(*chmap).channel_mask = 340u32;
}
}
}
pcm = list_next_entry__4!(pcm, list);
}
0
}

/// default channel maps for 2.1 speakers;
/// since HD-audio supports only stereo, odd number channels are omitted
#[unsafe(no_mangle)]
static snd_pcm_2_1_chmaps: __SlateAlign16<[snd_pcm_chmap_elem; 3]> = __SlateAlign16([snd_pcm_chmap_elem { channels: 2u8, map: {
let mut __t0: [u8; 15] = unsafe { core::mem::zeroed() };
__t0[0] = 3u8;
__t0[1] = 4u8;
__t0
} }, snd_pcm_chmap_elem { channels: 4u8, map: {
let mut __t1: [u8; 15] = unsafe { core::mem::zeroed() };
__t1[0] = 3u8;
__t1[1] = 4u8;
__t1[2] = 8u8;
__t1[3] = 8u8;
__t1
} }, {
let mut __t2: snd_pcm_chmap_elem = unsafe { core::mem::zeroed() };
__t2
}]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_pcm_2_1_chmaps_613: *mut () = (unsafe { core::ptr::addr_of!(snd_pcm_2_1_chmaps.0) }) as *mut ();

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_build_controls")]
extern "C" fn snd_hda_codec_build_controls(codec: *mut hda_codec) -> i32 {
let driver: *mut hda_codec_driver = hda_codec_to_driver!(codec);
let mut err: i32 = 0;
hda_exec_init_verbs(codec);
// continue to initialize...
err = snd_hda_codec_init(codec);
if err < 0 {
return err;
}
if (unsafe { (*(*driver).ops).build_controls }) != None {
err = unsafe { unsafe { (*(*driver).ops).build_controls }.unwrap()(codec) };
if err < 0 {
return err;
}
}
// we create chmaps here instead of build_pcms
err = add_std_chmaps(codec);
if err < 0 {
return err;
}
unsafe { snd_hda_jack_report_sync(codec) }; // call at the last init point
if (unsafe { (*codec).jackpoll_interval }) != 0 {
schedule_delayed_work(unsafe { core::ptr::addr_of_mut!((*codec).jackpoll_work) }, unsafe { (*codec).jackpoll_interval });
}
sync_power_up_states(codec);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_build_controls_614: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec) -> i32>, *mut ()>(Some(snd_hda_codec_build_controls)) };

/// PCM stuff
#[unsafe(link_section = ".text.slate_distinct.codec.hda_pcm_default_open_close")]
extern "C" fn hda_pcm_default_open_close(hinfo: *mut hda_pcm_stream, codec: *mut hda_codec, substream: *mut snd_pcm_substream) -> i32 {
0
}

#[unsafe(link_section = ".text.slate_distinct.codec.hda_pcm_default_prepare")]
extern "C" fn hda_pcm_default_prepare(hinfo: *mut hda_pcm_stream, codec: *mut hda_codec, stream_tag: u32, format: u32, substream: *mut snd_pcm_substream) -> i32 {
snd_hda_codec_setup_stream(codec, unsafe { (*hinfo).nid }, stream_tag, 0i32, format as i32);
0
}

#[unsafe(link_section = ".text.slate_distinct.codec.hda_pcm_default_cleanup")]
extern "C" fn hda_pcm_default_cleanup(hinfo: *mut hda_pcm_stream, codec: *mut hda_codec, substream: *mut snd_pcm_substream) -> i32 {
snd_hda_codec_cleanup_stream!(codec, unsafe { (*hinfo).nid });
0
}

fn set_pcm_default_values(codec: *mut hda_codec, info: *mut hda_pcm_stream) -> i32 {
let mut err: i32 = 0;
// query support PCM information from the given NID
if (unsafe { (*info).nid }) != 0 && ((unsafe { (*info).rates }) == 0 || (unsafe { (*info).formats }) == 0) {
err = snd_hda_query_supported_pcm__2!(codec, unsafe { (*info).nid }, if (unsafe { (*info).rates }) != 0 { core::ptr::null_mut::<u32>() } else { unsafe { core::ptr::addr_of_mut!((*info).rates) } }, if (unsafe { (*info).formats }) != 0 { core::ptr::null_mut::<u64>() } else { unsafe { core::ptr::addr_of_mut!((*info).formats) } }, if (unsafe { (*info).subformats }) != 0 { core::ptr::null_mut::<u32>() } else { unsafe { core::ptr::addr_of_mut!((*info).subformats) } }, if (unsafe { (*info).maxbps }) != 0 { core::ptr::null_mut::<u32>() } else { unsafe { core::ptr::addr_of_mut!((*info).maxbps) } });
if err < 0 {
return err;
}
}
if (unsafe { (*info).ops.open }) == None {
unsafe {
(*info).ops.open = Some(hda_pcm_default_open_close);
}
}
if (unsafe { (*info).ops.close }) == None {
unsafe {
(*info).ops.close = Some(hda_pcm_default_open_close);
}
}
if (unsafe { (*info).ops.prepare }) == None {
if snd_BUG_ON!((unsafe { (*info).nid }) == 0) != 0 {
return 22i32.wrapping_neg();
}
unsafe {
(*info).ops.prepare = Some(hda_pcm_default_prepare);
}
}
if (unsafe { (*info).ops.cleanup }) == None {
if snd_BUG_ON!((unsafe { (*info).nid }) == 0) != 0 {
return 22i32.wrapping_neg();
}
unsafe {
(*info).ops.cleanup = Some(hda_pcm_default_cleanup);
}
}
0
}

/// codec prepare/cleanup entries
///
/// snd_hda_codec_prepare - Prepare a stream
/// @codec: the HDA codec
/// @hinfo: PCM information
/// @stream: stream tag to assign
/// @format: format id to assign
/// @substream: PCM substream to assign
///
/// Calls the prepare callback set by the codec with the given arguments.
/// Clean up the inactive streams when successful.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_prepare")]
extern "C" fn snd_hda_codec_prepare(codec: *mut hda_codec, hinfo: *mut hda_pcm_stream, stream: u32, format: u32, substream: *mut snd_pcm_substream) -> i32 {
let mut ret: i32 = 0;
let mut __UNIQUE_ID_guard_615: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*(*codec).bus).prepare_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_615) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_616: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*(*codec).bus).prepare_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_616) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if (unsafe { (*hinfo).ops.prepare }) != None {
ret = unsafe { unsafe { (*hinfo).ops.prepare }.unwrap()(hinfo, codec, stream, format, substream) };
} else {
ret = 19i32.wrapping_neg();
}
if ret >= 0 {
purify_inactive_streams(codec);
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_prepare_617: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *mut hda_pcm_stream, u32, u32, *mut snd_pcm_substream) -> i32>, *mut ()>(Some(snd_hda_codec_prepare)) };

/// snd_hda_codec_cleanup - Clean up stream resources
/// @codec: the HDA codec
/// @hinfo: PCM information
/// @substream: PCM substream
///
/// Calls the cleanup callback set by the codec with the given arguments.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_cleanup")]
extern "C" fn snd_hda_codec_cleanup(codec: *mut hda_codec, hinfo: *mut hda_pcm_stream, substream: *mut snd_pcm_substream) {
let mut __UNIQUE_ID_guard_618: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*(*codec).bus).prepare_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_618) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_619: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*(*codec).bus).prepare_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_619) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if (unsafe { (*hinfo).ops.cleanup }) != None {
unsafe { unsafe { (*hinfo).ops.cleanup }.unwrap()(hinfo, codec, substream) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_cleanup_620: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *mut hda_pcm_stream, *mut snd_pcm_substream)>, *mut ()>(Some(snd_hda_codec_cleanup)) };

// global
/// get the empty PCM device number to assign
fn get_empty_pcm_device(bus: *mut hda_bus, r#type: u32) -> i32 {
// audio device indices; not linear to keep compatibility
//
// assigned to static slots up to dev#10; if more needed, assign
// the later slot dynamically (when CONFIG_SND_DYNAMIC_MINORS=y)
let mut i: i32 = 0;
if r#type >= 4 {
while false {
}
unsafe { _dev_err((*(*bus).card).dev as *const device, c"Invalid PCM type %d\n".as_ptr() as *const u8, r#type) };
return 22i32.wrapping_neg();
}
while (unsafe { audio_idx.0[r#type as usize][i as usize] }) >= 0 {
if (unsafe { audio_idx.0[r#type as usize][i as usize] }) >= 8 {
break;
}
if !test_and_set_bit((unsafe { audio_idx.0[r#type as usize][i as usize] }) as i64, (unsafe { (*bus).pcm_dev_bits.as_mut_ptr() as *mut u64 }) as *mut u64) {
return unsafe { audio_idx.0[r#type as usize][i as usize] };
}
i = i.wrapping_add(1);
}
while false {
}
unsafe { _dev_warn((*(*bus).card).dev as *const device, c"Too many %s devices\n".as_ptr() as *const u8, snd_hda_pcm_type_name.0[r#type as usize]) };
while false {
}
unsafe { _dev_warn((*(*bus).card).dev as *const device, c"Consider building the kernel with CONFIG_SND_DYNAMIC_MINORS=y\n".as_ptr() as *const u8) };
11i32.wrapping_neg()
}

static audio_idx: __SlateAlign16<[[i32; 5]; 4]> = __SlateAlign16([[0i32, 2i32, 4i32, 5i32, 1i32.wrapping_neg()], {
let mut __t0: [i32; 5] = unsafe { core::mem::zeroed() };
__t0[0] = 1i32;
__t0[1] = 1i32.wrapping_neg();
__t0
}, [3i32, 7i32, 8i32, 9i32, 1i32.wrapping_neg()], {
let mut __t1: [i32; 5] = unsafe { core::mem::zeroed() };
__t1[0] = 6i32;
__t1[1] = 1i32.wrapping_neg();
__t1
}]);

/// call build_pcms ops of the given codec and set up the default parameters
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_parse_pcms")]
extern "C" fn snd_hda_codec_parse_pcms(codec: *mut hda_codec) -> i32 {
let driver: *mut hda_codec_driver = hda_codec_to_driver!(codec);
let mut cpcm: *mut hda_pcm = unsafe { core::mem::zeroed() };
let mut err: i32 = 0;
if list_empty((unsafe { core::ptr::addr_of_mut!((*codec).pcm_list_head) }) as *const list_head) == 0 {
return 0;
}
// already parsed
if (unsafe { (*(*driver).ops).build_pcms }) == None {
return 0;
}
err = unsafe { unsafe { (*(*driver).ops).build_pcms }.unwrap()(codec) };
if err < 0 {
while false {
}
unsafe { _dev_err(hda_codec_dev!(codec) as *const device, c"cannot build PCMs for #%d (error %d)\n".as_ptr() as *const u8, (*codec).core.addr, err) };
return err;
}
cpcm = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*codec).pcm_list_head) });
while list_entry_is_head!(cpcm, unsafe { core::ptr::addr_of_mut!((*codec).pcm_list_head) }, list) == 0 {
let mut stream: i32 = 0;
while stream <= 1 {
let info: *mut hda_pcm_stream = unsafe { (*cpcm).stream.as_mut_ptr() as *mut hda_pcm_stream }.wrapping_offset(stream as isize);
if (unsafe { (*info).substreams }) != 0 {
err = set_pcm_default_values(codec, info);
if err < 0 {
while false {
}
unsafe { _dev_warn(hda_codec_dev!(codec) as *const device, c"fail to setup default for PCM %s\n".as_ptr() as *const u8, (*cpcm).name) };
return err;
}
}
stream = stream.wrapping_add(1);
}
cpcm = list_next_entry__4!(cpcm, list);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_parse_pcms_621: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec) -> i32>, *mut ()>(Some(snd_hda_codec_parse_pcms)) };

/// assign all PCMs of the given codec
#[unsafe(no_mangle)]
extern "C" fn snd_hda_codec_build_pcms(codec: *mut hda_codec) -> i32 {
let bus: *mut hda_bus = unsafe { (*codec).bus };
let mut cpcm: *mut hda_pcm = unsafe { core::mem::zeroed() };
let mut dev: i32 = 0;
let mut err: i32 = snd_hda_codec_parse_pcms(codec);
if err < 0 {
return err;
}
// attach a new PCM streams
cpcm = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*codec).pcm_list_head) });
while list_entry_is_head!(cpcm, unsafe { core::ptr::addr_of_mut!((*codec).pcm_list_head) }, list) == 0 {
'__slate_continue_1: {
if unsafe { (*cpcm).pcm }.is_null() {
// already attached
if !((unsafe { (*((*cpcm).stream.as_mut_ptr() as *mut hda_pcm_stream).wrapping_offset(0 as isize)).substreams }) == 0 && (unsafe { (*((*cpcm).stream.as_mut_ptr() as *mut hda_pcm_stream).wrapping_offset(1 as isize)).substreams }) == 0) {
// no substreams assigned
dev = get_empty_pcm_device(bus, unsafe { (*cpcm).pcm_type });
if dev < 0 {
unsafe {
(*cpcm).device = 1i32.wrapping_neg();
}
break '__slate_continue_1; // no fatal error
}
unsafe {
(*cpcm).device = dev;
}
err = unsafe { snd_hda_attach_pcm_stream(bus, codec, cpcm) };
if err < 0 {
while false {
}
unsafe { _dev_err(hda_codec_dev!(codec) as *const device, c"cannot attach PCM stream %d for codec #%d\n".as_ptr() as *const u8, dev, (*codec).core.addr) };
break '__slate_continue_1; // no fatal error
}
}
}
}
cpcm = list_next_entry__4!(cpcm, list);
}
0
}

/// snd_hda_add_new_ctls - create controls from the array
/// @codec: the HDA codec
/// @knew: the array of struct snd_kcontrol_new
///
/// This helper function creates and add new controls in the given array.
/// The array must be terminated with an empty entry as terminator.
///
/// Returns 0 if successful, or a negative error code.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_add_new_ctls")]
extern "C" fn snd_hda_add_new_ctls(codec: *mut hda_codec, mut knew: *const snd_kcontrol_new) -> i32 {
let mut err: i32 = 0;
while !unsafe { (*knew).name }.is_null() {
let mut addr: i32 = 0;
let mut idx: i32 = 0;
if (unsafe { (*knew).iface }) != 1i32.wrapping_neg() {
// skip this codec private value
loop {
let kctl: *mut snd_kcontrol = unsafe { snd_ctl_new1(knew, codec as *mut ()) };
if kctl.is_null() {
return 12i32.wrapping_neg();
}
// Do not use the id.device field for MIXER elements.
// This field is for real device numbers (like PCM) but codecs
// are hidden components from the user space view (unrelated
// to the mixer element identification).
if addr > 0 && ((unsafe { (*codec).__slate_bits_0.__get_ctl_dev_id() }) as i32) != 0 {
unsafe {
(*kctl).id.device = addr as u32;
}
}
if idx > 0 {
unsafe {
(*kctl).id.index = idx as u32;
}
}
err = snd_hda_ctl_add(codec, 0u16, kctl);
if err == 0 {
break;
}
// try first with another device index corresponding to
// the codec addr; if it still fails (or it's the
// primary codec), then try another control index
if addr == 0 && (unsafe { (*codec).core.addr }) != 0 {
addr = (unsafe { (*codec).core.addr }) as i32;
if ((unsafe { (*codec).__slate_bits_0.__get_ctl_dev_id() }) as i32) == 0 {
idx = idx.wrapping_add(10i32.wrapping_mul(addr));
}
} else {
if idx == 0 && (unsafe { (*knew).index }) == 0 {
idx = find_empty_mixer_ctl_idx(codec, unsafe { (*knew).name }, 0i32);
if idx <= 0 {
return err;
}
} else {
return err;
}
}
}
}
let __v4: *const snd_kcontrol_new = knew;
knew = __v4.wrapping_offset(1 as isize);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_add_new_ctls_622: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *const snd_kcontrol_new) -> i32>, *mut ()>(Some(snd_hda_add_new_ctls)) };

/// snd_hda_codec_set_power_save - Configure codec's runtime PM
/// @codec: codec device to configure
/// @delay: autosuspend delay
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_set_power_save")]
extern "C" fn snd_hda_codec_set_power_save(codec: *mut hda_codec, mut delay: i32) {
let dev: *mut device = hda_codec_dev!(codec);
if delay == 0 && ((unsafe { (*codec).__slate_bits_0.__get_auto_runtime_pm() }) as i32) != 0 {
delay = 3000;
}
if delay > 0 {
unsafe { pm_runtime_set_autosuspend_delay(dev, delay) };
pm_runtime_use_autosuspend(dev);
unsafe { pm_runtime_allow(dev) };
if !pm_runtime_suspended(dev) {
pm_runtime_mark_last_busy(dev);
}
} else {
pm_runtime_dont_use_autosuspend(dev);
unsafe { pm_runtime_forbid(dev) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_set_power_save_623: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, i32)>, *mut ()>(Some(snd_hda_codec_set_power_save)) };

/// snd_hda_set_power_save - reprogram autosuspend for the given delay
/// @bus: HD-audio bus
/// @delay: autosuspend delay in msec, 0 = off
///
/// Synchronize the runtime PM autosuspend state from the power_save option.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_set_power_save")]
extern "C" fn snd_hda_set_power_save(bus: *mut hda_bus, delay: i32) {
let mut c: *mut hda_codec = unsafe { core::mem::zeroed() };
let mut __v0: *mut hda_codec;
__v0 = ((unsafe { (((*core::ptr::addr_of_mut!((*bus).core.codec_list)).next as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut hda_codec;
c = __v0;
while list_is_head((unsafe { core::ptr::addr_of_mut!((*c).core.list) }) as *const list_head, (unsafe { core::ptr::addr_of_mut!((*bus).core.codec_list) }) as *const list_head) == 0 {
snd_hda_codec_set_power_save(c, delay);
let mut __v2: *mut hda_codec;
__v2 = ((unsafe { (((*c).core.list.next as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut hda_codec;
c = __v2;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_set_power_save_624: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_bus, i32)>, *mut ()>(Some(snd_hda_set_power_save)) };

/// snd_hda_check_amp_list_power - Check the amp list and update the power
/// @codec: HD-audio codec
/// @check: the object containing an AMP list and the status
/// @nid: NID to check / update
///
/// Check whether the given NID is in the amp list.  If it's in the list,
/// check the current AMP status, and update the power-status according
/// to the mute status.
///
/// This function is supposed to be set or called from the check_power_status
/// patch ops.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_check_amp_list_power")]
extern "C" fn snd_hda_check_amp_list_power(codec: *mut hda_codec, check: *mut hda_loopback_check, nid: u16) -> i32 {
let mut p: *const hda_amp_list = unsafe { core::mem::zeroed() };
let mut v: i32 = 0;
if unsafe { (*check).amplist }.is_null() {
return 0;
}
p = unsafe { (*check).amplist };
while (unsafe { (*p).nid }) != 0 {
if ((unsafe { (*p).nid }) as i32) == (nid as i32) {
break;
}
let __v1: *const hda_amp_list = p;
p = __v1.wrapping_offset(1 as isize);
}
if (unsafe { (*p).nid }) == 0 {
return 0;
}
// nothing changed
p = unsafe { (*check).amplist };
while (unsafe { (*p).nid }) != 0 {
for ch in 0i32..2i32 {
v = snd_hda_codec_amp_read__2!(codec, unsafe { (*p).nid }, ch, unsafe { (*p).dir }, unsafe { (*p).idx });
if v & 128 == 0 && v > 0 {
if (unsafe { (*check).power_on }) == 0 {
unsafe {
(*check).power_on = 1;
}
snd_hda_power_up_pm__2!(codec);
}
return 1;
}
}
let __v7: *const hda_amp_list = p;
p = __v7.wrapping_offset(1 as isize);
}
if (unsafe { (*check).power_on }) != 0 {
unsafe {
(*check).power_on = 0;
}
snd_hda_power_down_pm!(codec);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_check_amp_list_power_625: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *mut hda_loopback_check, u16) -> i32>, *mut ()>(Some(snd_hda_check_amp_list_power)) };

// input MUX helper
/// snd_hda_input_mux_info - Info callback helper for the input-mux enum
/// @imux: imux helper object
/// @uinfo: pointer to get/store the data
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_input_mux_info")]
extern "C" fn snd_hda_input_mux_info(imux: *const hda_input_mux, uinfo: *mut snd_ctl_elem_info) -> i32 {
let mut index: u32 = 0;
unsafe {
(*uinfo).r#type = 3;
}
unsafe {
(*uinfo).count = 1;
}
unsafe {
(*uinfo).value.enumerated.items = (*imux).num_items;
}
if (unsafe { (*imux).num_items }) == 0 {
return 0;
}
index = unsafe { (*uinfo).value.enumerated.item };
if index >= unsafe { (*imux).num_items } {
index = unsafe { (*imux).num_items }.wrapping_sub(1);
}
unsafe { sized_strscpy((*uinfo).value.enumerated.name.as_mut_ptr() as *mut u8, (*((*imux).items.as_ptr() as *const hda_input_mux_item).wrapping_offset(index as isize)).label.as_ptr() as *const u8, (core::mem::size_of::<[u8; 64]>() as u64).wrapping_add(__must_be_array__23!() as u64).wrapping_add(__must_be_cstr__47!() as u64).wrapping_add(__must_be_cstr__60!() as u64)) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_input_mux_info_626: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const hda_input_mux, *mut snd_ctl_elem_info) -> i32>, *mut ()>(Some(snd_hda_input_mux_info)) };

/// snd_hda_input_mux_put - Put callback helper for the input-mux enum
/// @codec: the HDA codec
/// @imux: imux helper object
/// @ucontrol: pointer to get/store the data
/// @nid: input mux NID
/// @cur_val: pointer to get/store the current imux value
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_input_mux_put")]
extern "C" fn snd_hda_input_mux_put(codec: *mut hda_codec, imux: *const hda_input_mux, ucontrol: *mut snd_ctl_elem_value, nid: u16, cur_val: *mut u32) -> i32 {
let mut idx: u32 = 0;
if (unsafe { (*imux).num_items }) == 0 {
return 0;
}
idx = unsafe { *((*ucontrol).value.enumerated.item.as_mut_ptr() as *mut u32).wrapping_offset(0 as isize) };
if idx >= unsafe { (*imux).num_items } {
idx = unsafe { (*imux).num_items }.wrapping_sub(1);
}
if (unsafe { *cur_val }) == idx {
return 0;
}
snd_hda_codec_write_cache(codec, nid, 0i32, 1793u32, unsafe { (*((*imux).items.as_ptr() as *const hda_input_mux_item).wrapping_offset(idx as isize)).index });
unsafe {
*cur_val = idx;
}
1
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_input_mux_put_627: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *const hda_input_mux, *mut snd_ctl_elem_value, u16, *mut u32) -> i32>, *mut ()>(Some(snd_hda_input_mux_put)) };

/// snd_hda_enum_helper_info - Helper for simple enum ctls
/// @kcontrol: ctl element
/// @uinfo: pointer to get/store the data
/// @num_items: number of enum items
/// @texts: enum item string array
///
/// process kcontrol info callback of a simple string enum array
/// when @num_items is 0 or @texts is NULL, assume a boolean enum array
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_enum_helper_info")]
extern "C" fn snd_hda_enum_helper_info(kcontrol: *mut snd_kcontrol, uinfo: *mut snd_ctl_elem_info, mut num_items: i32, mut texts: *const *const u8) -> i32 {
if texts.is_null() || num_items == 0 {
num_items = 2;
texts = unsafe { core::ptr::addr_of!(texts_default.0) as *const *const u8 };
}
unsafe { snd_ctl_enum_info(uinfo, 1u32, num_items as u32, texts) }
}

static texts_default: __SlateAlign16<[*const u8; 2]> = __SlateAlign16([(b"Disabled\0".as_ptr() as *mut u8) as *const u8, (b"Enabled\0".as_ptr() as *mut u8) as *const u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_enum_helper_info_628: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_info, i32, *const *const u8) -> i32>, *mut ()>(Some(snd_hda_enum_helper_info)) };

// Multi-channel / digital-out PCM helper functions
/// setup SPDIF output stream
fn setup_dig_out_stream(codec: *mut hda_codec, nid: u16, stream_tag: u32, format: u32) {
let mut curr_fmt: u32 = 0;
let mut reset: bool = false;
let spdif: *mut hda_spdif_out = snd_hda_spdif_out_of_nid(codec, nid);
// Add sanity check to pass klockwork check.
// This should never happen.
let mut __v0: i64;
let __ret_warn_on: i32 = spdif.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("629: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 629b - ., 3; .popsection\n# {0}", const 629 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_setup_dig_out_stream_1, sym __slate_asm_string_setup_dig_out_stream_2, const 3615 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
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
return;
}
curr_fmt = snd_hda_codec_read(codec, nid, 0i32, 2560u32, 0u32);
reset = ((unsafe { (*codec).__slate_bits_0.__get_spdif_status_reset() }) as i32) != 0 && ((unsafe { (*spdif).ctls }) as i32) & 1i32 != 0 && curr_fmt != format;
// turn off SPDIF if needed; otherwise the IEC958 bits won't be
// updated
if reset {
set_dig_out_convert(codec, nid, ((unsafe { (*spdif).ctls }) as i32) & -2i32 & 255, 1i32.wrapping_neg());
}
snd_hda_codec_setup_stream(codec, nid, stream_tag, 0i32, format as i32);
if !unsafe { (*codec).follower_dig_outs }.is_null() {
let mut d: *const u16 = unsafe { core::mem::zeroed() };
d = unsafe { (*codec).follower_dig_outs };
while (unsafe { *d }) != 0 {
snd_hda_codec_setup_stream(codec, unsafe { *d }, stream_tag, 0i32, format as i32);
let __v6: *const u16 = d;
d = __v6.wrapping_offset(1 as isize);
}
}
// turn on again (if needed)
if reset {
set_dig_out_convert(codec, nid, ((unsafe { (*spdif).ctls }) as i32) & 255, 1i32.wrapping_neg());
}
}

fn cleanup_dig_out_stream(codec: *mut hda_codec, nid: u16) {
snd_hda_codec_cleanup_stream!(codec, nid);
if !unsafe { (*codec).follower_dig_outs }.is_null() {
let mut d: *const u16 = unsafe { core::mem::zeroed() };
d = unsafe { (*codec).follower_dig_outs };
while (unsafe { *d }) != 0 {
snd_hda_codec_cleanup_stream!(codec, unsafe { *d });
let __v1: *const u16 = d;
d = __v1.wrapping_offset(1 as isize);
}
}
}

/// snd_hda_multi_out_dig_open - open the digital out in the exclusive mode
/// @codec: the HDA codec
/// @mout: hda_multi_out object
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_multi_out_dig_open")]
extern "C" fn snd_hda_multi_out_dig_open(codec: *mut hda_codec, mout: *mut hda_multi_out) -> i32 {
let mut __UNIQUE_ID_guard_631: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_631) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_632: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_632) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if (unsafe { (*mout).dig_out_used }) == 2 {
cleanup_dig_out_stream(codec, unsafe { (*mout).dig_out_nid });
}
// already opened as analog dup; reset it once
unsafe {
(*mout).dig_out_used = 1;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_multi_out_dig_open_633: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *mut hda_multi_out) -> i32>, *mut ()>(Some(snd_hda_multi_out_dig_open)) };

/// snd_hda_multi_out_dig_prepare - prepare the digital out stream
/// @codec: the HDA codec
/// @mout: hda_multi_out object
/// @stream_tag: stream tag to assign
/// @format: format id to assign
/// @substream: PCM substream to assign
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_multi_out_dig_prepare")]
extern "C" fn snd_hda_multi_out_dig_prepare(codec: *mut hda_codec, mout: *mut hda_multi_out, stream_tag: u32, format: u32, substream: *mut snd_pcm_substream) -> i32 {
let mut __UNIQUE_ID_guard_634: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_634) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_635: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_635) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
setup_dig_out_stream(codec, unsafe { (*mout).dig_out_nid }, stream_tag, format);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_multi_out_dig_prepare_636: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *mut hda_multi_out, u32, u32, *mut snd_pcm_substream) -> i32>, *mut ()>(Some(snd_hda_multi_out_dig_prepare)) };

/// snd_hda_multi_out_dig_cleanup - clean-up the digital out stream
/// @codec: the HDA codec
/// @mout: hda_multi_out object
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_multi_out_dig_cleanup")]
extern "C" fn snd_hda_multi_out_dig_cleanup(codec: *mut hda_codec, mout: *mut hda_multi_out) -> i32 {
let mut __UNIQUE_ID_guard_637: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_637) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_638: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_638) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
cleanup_dig_out_stream(codec, unsafe { (*mout).dig_out_nid });
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_multi_out_dig_cleanup_639: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *mut hda_multi_out) -> i32>, *mut ()>(Some(snd_hda_multi_out_dig_cleanup)) };

/// snd_hda_multi_out_dig_close - release the digital out stream
/// @codec: the HDA codec
/// @mout: hda_multi_out object
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_multi_out_dig_close")]
extern "C" fn snd_hda_multi_out_dig_close(codec: *mut hda_codec, mout: *mut hda_multi_out) -> i32 {
let mut __UNIQUE_ID_guard_640: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_640) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_641: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_641) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
unsafe {
(*mout).dig_out_used = 0;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_multi_out_dig_close_642: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *mut hda_multi_out) -> i32>, *mut ()>(Some(snd_hda_multi_out_dig_close)) };

/// snd_hda_multi_out_analog_open - open analog outputs
/// @codec: the HDA codec
/// @mout: hda_multi_out object
/// @substream: PCM substream to assign
/// @hinfo: PCM information to assign
///
/// Open analog outputs and set up the hw-constraints.
/// If the digital outputs can be opened as follower, open the digital
/// outputs, too.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_multi_out_analog_open")]
extern "C" fn snd_hda_multi_out_analog_open(codec: *mut hda_codec, mout: *mut hda_multi_out, substream: *mut snd_pcm_substream, hinfo: *mut hda_pcm_stream) -> i32 {
let runtime: *mut snd_pcm_runtime = unsafe { (*substream).runtime };
let mut notify_share_sw: bool = false;
unsafe {
(*runtime).hw.channels_max = (*mout).max_channels as u32;
}
if (unsafe { (*mout).dig_out_nid }) != 0 {
if (unsafe { (*mout).analog_rates }) == 0 {
unsafe {
(*mout).analog_rates = (*hinfo).rates;
}
unsafe {
(*mout).analog_formats = (*hinfo).formats;
}
unsafe {
(*mout).analog_maxbps = (*hinfo).maxbps;
}
} else {
unsafe {
(*runtime).hw.rates = (*mout).analog_rates;
}
unsafe {
(*runtime).hw.formats = (*mout).analog_formats;
}
unsafe {
(*hinfo).maxbps = (*mout).analog_maxbps;
}
}
if (unsafe { (*mout).spdif_rates }) == 0 {
snd_hda_query_supported_pcm!(codec, unsafe { (*mout).dig_out_nid }, unsafe { core::ptr::addr_of_mut!((*mout).spdif_rates) }, unsafe { core::ptr::addr_of_mut!((*mout).spdif_formats) }, unsafe { core::ptr::addr_of_mut!((*mout).spdif_maxbps) });
}
let mut __UNIQUE_ID_guard_643: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_643) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_644: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_644) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if (unsafe { (*mout).share_spdif }) != 0 {
if (unsafe { (*runtime).hw.rates }) & unsafe { (*mout).spdif_rates } != 0 && (unsafe { (*runtime).hw.formats }) & unsafe { (*mout).spdif_formats } != 0 {
unsafe {
(*runtime).hw.rates = (*runtime).hw.rates & (*mout).spdif_rates;
}
unsafe {
(*runtime).hw.formats = (*runtime).hw.formats & (*mout).spdif_formats;
}
if (unsafe { (*mout).spdif_maxbps }) < unsafe { (*hinfo).maxbps } {
unsafe {
(*hinfo).maxbps = (*mout).spdif_maxbps;
}
}
} else {
unsafe {
(*mout).share_spdif = 0;
}
notify_share_sw = true;
}
}
}
if notify_share_sw {
notify_spdif_share_sw(codec, mout);
}
unsafe { snd_pcm_hw_constraint_step((*substream).runtime, 0u32, 10i32, 2u64) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_multi_out_analog_open_645: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *mut hda_multi_out, *mut snd_pcm_substream, *mut hda_pcm_stream) -> i32>, *mut ()>(Some(snd_hda_multi_out_analog_open)) };

/// snd_hda_multi_out_analog_prepare - Preapre the analog outputs.
/// @codec: the HDA codec
/// @mout: hda_multi_out object
/// @stream_tag: stream tag to assign
/// @format: format id to assign
/// @substream: PCM substream to assign
///
/// Set up the i/o for analog out.
/// When the digital out is available, copy the front out to digital out, too.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_multi_out_analog_prepare")]
extern "C" fn snd_hda_multi_out_analog_prepare(codec: *mut hda_codec, mout: *mut hda_multi_out, stream_tag: u32, format: u32, substream: *mut snd_pcm_substream) -> i32 {
let mut ch: i32 = unsafe { core::mem::zeroed() };
let mut __v7: bool = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_unlock_647: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_647) as *mut *mut __SlateRecord_mutex_h_253;
let mut scope: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_mutex_h_253;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut spdif: core::mem::MaybeUninit<*mut hda_spdif_out> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut hda_spdif_out = core::ptr::addr_of_mut!(spdif) as *mut *mut hda_spdif_out;
let mut chs: i32 = unsafe { core::mem::zeroed() };
let mut nids: *const u16 = unsafe { core::mem::zeroed() };
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
unsafe {
nids = (*mout).dac_nids;
chs = (*(*substream).runtime).channels as i32;
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
core::ptr::write(__slate_slot_9, class_mutex_constructor(core::ptr::addr_of_mut!((*codec).spdif_mutex)));
__slate_cleanup_armed_9.set(true);
core::ptr::write(__slate_slot_8, (core::ptr::addr_of_mut!((*codec).spdif_mutex) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_8.set(true);
if !class_mutex_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_9))).is_null() || !class_mutex_is_conditional {
if true {
*__slate_slot_11 = snd_hda_spdif_out_of_nid(codec, (*mout).dig_out_nid);
if (*mout).dig_out_nid != 0 && (*mout).share_spdif != 0 && (*mout).dig_out_used != 1 {
if chs == 2 && !(*__slate_slot_11).is_null() {
__v7 = snd_hda_is_supported_format!(codec, unsafe { (*mout).dig_out_nid }, format);
} else {
__v7 = false;
}
if __v7 && (*(*__slate_slot_11)).status & 2u32 == 0 {
(*mout).dig_out_used = 2;
setup_dig_out_stream(codec, (*mout).dig_out_nid, stream_tag, format);
} else {
(*mout).dig_out_used = 0;
cleanup_dig_out_stream(codec, (*mout).dig_out_nid);
}
}
}
}
__slate_cleanup_armed_8.set(false);
__class_mutex_cleanup_ctx(__slate_slot_8 as *mut _);
__slate_cleanup_armed_9.set(false);
class_mutex_destructor(__slate_slot_9 as *mut _);
// front
snd_hda_codec_setup_stream(codec, *nids.wrapping_offset(0 as isize), stream_tag, 0i32, format as i32);
if (*mout).no_share_stream == 0 && (*mout).hp_nid != 0 && ((*mout).hp_nid as i32) != (*nids.wrapping_offset(0 as isize) as i32) {
snd_hda_codec_setup_stream(codec, (*mout).hp_nid, stream_tag, 0i32, format as i32);
}
// headphone out will just decode front left/right (stereo)
// extra outputs copied from front
*__slate_slot_10 = 0;
while (*__slate_slot_10 as u64) < ARRAY_SIZE__53!() {
if (*mout).no_share_stream == 0 && *((*mout).hp_out_nid.as_mut_ptr() as *mut u16).wrapping_offset(*__slate_slot_10 as isize) != 0 {
snd_hda_codec_setup_stream(codec, *((*mout).hp_out_nid.as_mut_ptr() as *mut u16).wrapping_offset(*__slate_slot_10 as isize), stream_tag, 0i32, format as i32);
}
let __v6: i32 = *__slate_slot_10;
*__slate_slot_10 = __v6.wrapping_add(1);
}
// surrounds
*__slate_slot_10 = 1;
while *__slate_slot_10 < (*mout).num_dacs {
if chs >= (*__slate_slot_10).wrapping_add(1).wrapping_mul(2) {
snd_hda_codec_setup_stream(codec, *nids.wrapping_offset(*__slate_slot_10 as isize), stream_tag, (*__slate_slot_10).wrapping_mul(2), format as i32);
} else {
if (*mout).no_share_stream == 0 {
snd_hda_codec_setup_stream(codec, *nids.wrapping_offset(*__slate_slot_10 as isize), stream_tag, 0i32, format as i32);
}
}
// independent out
// copy front
let __v4: i32 = *__slate_slot_10;
*__slate_slot_10 = __v4.wrapping_add(1);
}
// extra surrounds
*__slate_slot_10 = 0;
while (*__slate_slot_10 as u64) < ARRAY_SIZE__26!() {
ch = 0;
if *((*mout).extra_out_nid.as_mut_ptr() as *mut u16).wrapping_offset(*__slate_slot_10 as isize) == 0 {
break;
} else {
if chs >= (*__slate_slot_10).wrapping_add(1).wrapping_mul(2) {
ch = (*__slate_slot_10).wrapping_mul(2);
} else {
if (*mout).no_share_stream == 0 {
break;
}
}
snd_hda_codec_setup_stream(codec, *((*mout).extra_out_nid.as_mut_ptr() as *mut u16).wrapping_offset(*__slate_slot_10 as isize), stream_tag, ch, format as i32);
let __v1: i32 = *__slate_slot_10;
*__slate_slot_10 = __v1.wrapping_add(1);
}
}
return 0;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_multi_out_analog_prepare_648: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *mut hda_multi_out, u32, u32, *mut snd_pcm_substream) -> i32>, *mut ()>(Some(snd_hda_multi_out_analog_prepare)) };

/// snd_hda_multi_out_analog_cleanup - clean up the setting for analog out
/// @codec: the HDA codec
/// @mout: hda_multi_out object
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_multi_out_analog_cleanup")]
extern "C" fn snd_hda_multi_out_analog_cleanup(codec: *mut hda_codec, mout: *mut hda_multi_out) -> i32 {
let nids: *const u16 = unsafe { (*mout).dac_nids };
let mut i: i32 = 0;
while i < unsafe { (*mout).num_dacs } {
snd_hda_codec_cleanup_stream!(codec, unsafe { *nids.wrapping_offset(i as isize) });
i = i.wrapping_add(1);
}
if (unsafe { (*mout).hp_nid }) != 0 {
snd_hda_codec_cleanup_stream!(codec, unsafe { (*mout).hp_nid });
}
i = 0;
while (i as u64) < ARRAY_SIZE__24!() {
if (unsafe { *((*mout).hp_out_nid.as_mut_ptr() as *mut u16).wrapping_offset(i as isize) }) != 0 {
snd_hda_codec_cleanup_stream!(codec, unsafe { *((*mout).hp_out_nid.as_mut_ptr() as *mut u16).wrapping_offset(i as isize) });
}
i = i.wrapping_add(1);
}
i = 0;
while (i as u64) < ARRAY_SIZE__9!() {
if (unsafe { *((*mout).extra_out_nid.as_mut_ptr() as *mut u16).wrapping_offset(i as isize) }) != 0 {
snd_hda_codec_cleanup_stream!(codec, unsafe { *((*mout).extra_out_nid.as_mut_ptr() as *mut u16).wrapping_offset(i as isize) });
}
i = i.wrapping_add(1);
}
let mut __UNIQUE_ID_guard_649: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) });
let __slate_cleanup_ptr_9 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_649) as *mut _;
let __slate_cleanup_9 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_9 as *mut _));
let mut __UNIQUE_ID_unlock_650: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*codec).spdif_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_10 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_650) as *mut _;
let __slate_cleanup_10 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_10 as *mut _));
if (unsafe { (*mout).dig_out_nid }) != 0 && (unsafe { (*mout).dig_out_used }) == 2 {
cleanup_dig_out_stream(codec, unsafe { (*mout).dig_out_nid });
unsafe {
(*mout).dig_out_used = 0;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_multi_out_analog_cleanup_651: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *mut hda_multi_out) -> i32>, *mut ()>(Some(snd_hda_multi_out_analog_cleanup)) };

/// snd_hda_get_default_vref - Get the default (mic) VREF pin bits
/// @codec: the HDA codec
/// @pin: referred pin NID
///
/// Guess the suitable VREF pin bits to be set as the pin-control value.
/// Note: the function doesn't set the AC_PINCTL_IN_EN bit.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_get_default_vref")]
extern "C" fn snd_hda_get_default_vref(codec: *mut hda_codec, pin: u16) -> u32 {
let mut pincap: u32 = 0;
let oldval: u32 = snd_hda_codec_read(codec, pin, 0i32, 3847u32, 0u32);
pincap = snd_hda_query_pin_caps(codec, pin);
pincap = (pincap & 14080u32) >> 8i32;
// Exception: if the default pin setup is vref50, we give it priority
if pincap & 16u32 != 0 && oldval != 33u32 {
return 4;
} else {
if pincap & 2u32 != 0 {
return 1;
} else {
if pincap & 32u32 != 0 {
return 5;
} else {
if pincap & 4u32 != 0 {
return 2;
}
}
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_get_default_vref_652: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16) -> u32>, *mut ()>(Some(snd_hda_get_default_vref)) };

/// snd_hda_correct_pin_ctl - correct the pin ctl value for matching with the pin cap
/// @codec: the HDA codec
/// @pin: referred pin NID
/// @val: pin ctl value to audit
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_correct_pin_ctl")]
extern "C" fn snd_hda_correct_pin_ctl(codec: *mut hda_codec, pin: u16, mut val: u32) -> u32 {
let mut cap: u32 = 0;
if val == 0 {
return 0;
}
cap = snd_hda_query_pin_caps(codec, pin);
if cap == 0 {
return val;
}
// don't know what to do...
if val & 64u32 != 0 {
if cap & 16u32 == 0 {
let __v1: u32 = val & 4294967103u32;
val = __v1;
} else {
if val & 128u32 != 0 && cap & 8u32 == 0 {
let __v3: u32 = val & 4294967167u32;
val = __v3;
}
}
}
if val & 32u32 != 0 {
if cap & 32u32 == 0 {
let __v5: u32 = val & 4294967256u32;
val = __v5;
} else {
let vcap: u32 = (cap & 14080u32) >> 8i32;
let mut vref: u32 = val & 7u32;
let mut i: i32 = 0;
while (i as u64) < ARRAY_SIZE__33!() {
if vref == unsafe { cap_lists.0[i as usize][0] } && vcap & unsafe { cap_lists.0[i as usize][1] } == 0 {
if (i as u64) == ARRAY_SIZE__29!().wrapping_sub(1) {
vref = 0;
} else {
vref = unsafe { cap_lists.0[i.wrapping_add(1) as usize][0] };
}
}
i = i.wrapping_add(1);
}
let __v10: u32 = val & 4294967288u32;
val = __v10;
val |= vref;
}
}
val
}

static cap_lists: __SlateAlign16<[[u32; 2]; 4]> = __SlateAlign16([[5u32, 32u32], [4u32, 16u32], [1u32, 2u32], [2u32, 4u32]]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_correct_pin_ctl_653: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, u32) -> u32>, *mut ()>(Some(snd_hda_correct_pin_ctl)) };

/// _snd_hda_set_pin_ctl - Helper to set pin ctl value
/// @codec: the HDA codec
/// @pin: referred pin NID
/// @val: pin control value to set
/// @cached: access over codec pinctl cache or direct write
///
/// This function is a helper to set a pin ctl value more safely.
/// It corrects the pin ctl value via snd_hda_correct_pin_ctl(), stores the
/// value in pin target array via snd_hda_codec_set_pin_target(), then
/// actually writes the value via either snd_hda_codec_write_cache() or
/// snd_hda_codec_write() depending on @cached flag.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec._snd_hda_set_pin_ctl")]
extern "C" fn _snd_hda_set_pin_ctl(codec: *mut hda_codec, pin: u16, mut val: u32, cached: bool) -> i32 {
val = snd_hda_correct_pin_ctl(codec, pin, val);
snd_hda_codec_set_pin_target(codec, pin, val);
if cached {
return snd_hda_codec_write_cache(codec, pin, 0i32, 1799u32, val);
} else {
return snd_hda_codec_write(codec, pin, 0i32, 1799u32, val);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable__snd_hda_set_pin_ctl_654: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u16, u32, bool) -> i32>, *mut ()>(Some(_snd_hda_set_pin_ctl)) };

/// snd_hda_add_imux_item - Add an item to input_mux
/// @codec: the HDA codec
/// @imux: imux helper object
/// @label: the name of imux item to assign
/// @index: index number of imux item to assign
/// @type_idx: pointer to store the resultant label index
///
/// When the same label is used already in the existing items, the number
/// suffix is appended to the label.  This label index number is stored
/// to type_idx when non-NULL pointer is given.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_add_imux_item")]
extern "C" fn snd_hda_add_imux_item(codec: *mut hda_codec, imux: *mut hda_input_mux, label: *const u8, index: i32, type_idx: *mut i32) -> i32 {
let mut i: i32 = 0;
let mut label_idx: i32 = 0;
if (unsafe { (*imux).num_items }) >= 36 {
while false {
}
unsafe { _dev_err(hda_codec_dev!(codec) as *const device, c"hda_codec: Too many imux items!\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg();
}
while (i as u32) < unsafe { (*imux).num_items } {
if (unsafe { strncmp(label, ((*((*imux).items.as_mut_ptr() as *mut hda_input_mux_item).wrapping_offset(i as isize)).label.as_mut_ptr() as *mut u8) as *const u8, strlen(label)) }) == 0 {
label_idx = label_idx.wrapping_add(1);
}
i = i.wrapping_add(1);
}
if !type_idx.is_null() {
unsafe {
*type_idx = label_idx;
}
}
if label_idx > 0 {
unsafe { snprintf((*((*imux).items.as_mut_ptr() as *mut hda_input_mux_item).wrapping_offset((*imux).num_items as isize)).label.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 32]>() as u64, c"%s %d".as_ptr() as *const u8, label, label_idx) };
} else {
unsafe { sized_strscpy((*((*imux).items.as_mut_ptr() as *mut hda_input_mux_item).wrapping_offset((*imux).num_items as isize)).label.as_mut_ptr() as *mut u8, label, (core::mem::size_of::<[u8; 32]>() as u64).wrapping_add(__must_be_cstr__45!() as u64).wrapping_add(__must_be_cstr__76!() as u64)) };
}
unsafe {
(*((*imux).items.as_mut_ptr() as *mut hda_input_mux_item).wrapping_offset((*imux).num_items as isize)).index = index as u32;
}
let __v7: u32 = unsafe { (*imux).num_items };
unsafe {
(*imux).num_items = __v7.wrapping_add(1);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_add_imux_item_655: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, *mut hda_input_mux, *const u8, i32, *mut i32) -> i32>, *mut ()>(Some(snd_hda_add_imux_item)) };

/// snd_hda_bus_reset_codecs - Reset the bus
/// @bus: HD-audio bus
#[unsafe(no_mangle)]
extern "C" fn snd_hda_bus_reset_codecs(bus: *mut hda_bus) {
let mut codec: *mut hda_codec = unsafe { core::mem::zeroed() };
let mut __v0: *mut hda_codec;
__v0 = ((unsafe { (((*core::ptr::addr_of_mut!((*bus).core.codec_list)).next as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut hda_codec;
codec = __v0;
while list_is_head((unsafe { core::ptr::addr_of_mut!((*codec).core.list) }) as *const list_head, (unsafe { core::ptr::addr_of_mut!((*bus).core.codec_list) }) as *const list_head) == 0 {
// FIXME: maybe a better way needed for forced reset
if (unsafe { current_work() }) != unsafe { core::ptr::addr_of_mut!((*codec).jackpoll_work.work) } {
unsafe { cancel_delayed_work_sync(core::ptr::addr_of_mut!((*codec).jackpoll_work)) };
}
if hda_codec_is_power_on!(codec) {
hda_call_codec_suspend(codec);
hda_call_codec_resume(codec);
}
let mut __v2: *mut hda_codec;
__v2 = ((unsafe { (((*codec).core.list.next as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut hda_codec;
codec = __v2;
}
}

/// snd_hda_codec_set_gpio - Set up GPIO bits for AFG
/// @codec: the HDA codec
/// @mask: GPIO bitmask
/// @dir: GPIO direction bits
/// @data: GPIO data bits
/// @delay: the delay in msec before writing GPIO data bits
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_hda_codec_set_gpio")]
extern "C" fn snd_hda_codec_set_gpio(codec: *mut hda_codec, mask: u32, dir: u32, data: u32, delay: u32) {
snd_hda_codec_write(codec, unsafe { (*codec).core.afg }, 0i32, 1814u32, mask);
if delay != 0 {
snd_hda_codec_write_sync(codec, unsafe { (*codec).core.afg }, 0i32, 1815u32, dir);
unsafe { msleep(delay) };
snd_hda_codec_write_sync(codec, unsafe { (*codec).core.afg }, 0i32, 1813u32, data);
} else {
snd_hda_codec_write(codec, unsafe { (*codec).core.afg }, 0i32, 1815u32, dir);
snd_hda_codec_write(codec, unsafe { (*codec).core.afg }, 0i32, 1813u32, data);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_hda_codec_set_gpio_656: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hda_codec, u32, u32, u32, u32)>, *mut ()>(Some(snd_hda_codec_set_gpio)) };

/// snd_print_pcm_bits - Print the supported PCM fmt bits to the string buffer
/// @pcm: PCM caps bits
/// @buf: the string buffer to write
/// @buflen: the max buffer length
///
/// used by hda_proc.c and hda_eld.c
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.codec.snd_print_pcm_bits")]
extern "C" fn snd_print_pcm_bits(pcm: i32, buf: *mut u8, buflen: i32) {
let mut i: i32 = 0;
let mut j: i32 = 0;
while (i as u64) < ARRAY_SIZE__45!() {
if pcm & 65536i32 << i != 0 {
let __v0: i32 = unsafe { scnprintf(buf.wrapping_offset(j as isize), buflen.wrapping_sub(j) as u64, c" %d".as_ptr() as *const u8, bits.0[i as usize]) };
j = j.wrapping_add(__v0);
}
i = i.wrapping_add(1);
}
unsafe {
*buf.wrapping_offset(j as isize) = 0;
}
// necessary when j == 0
}

static bits: __SlateAlign16<[u32; 5]> = __SlateAlign16([8u32, 16u32, 20u32, 24u32, 32u32]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_print_pcm_bits_657: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, *mut u8, i32)>, *mut ()>(Some(snd_print_pcm_bits)) };

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_658: [u8; 41] = [115, 110, 100, 95, 104, 100, 97, 95, 99, 111, 100, 101, 99, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 72, 68, 65, 32, 99, 111, 100, 101, 99, 32, 99, 111, 114, 101, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_659: [u8; 50] = [115, 110, 100, 95, 104, 100, 97, 95, 99, 111, 100, 101, 99, 46, 102, 105, 108, 101, 61, 115, 111, 117, 110, 100, 47, 104, 100, 97, 47, 99, 111, 109, 109, 111, 110, 47, 115, 110, 100, 45, 104, 100, 97, 45, 99, 111, 100, 101, 99, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_660: [u8; 26] = [115, 110, 100, 95, 104, 100, 97, 95, 99, 111, 100, 101, 99, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_check_copy_size_1: [u8; 1] = *b"\0";

static __slate_asm_string_check_copy_size_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/include/linux/ucopysize.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_setup_dig_out_stream_1: [u8; 1] = *b"\0";

static __slate_asm_string_setup_dig_out_stream_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/sound/hda/common/codec.c\0";

static __slate_asm_string_snd_hda_spdif_ctls_assign_1: [u8; 1] = *b"\0";

static __slate_asm_string_snd_hda_spdif_ctls_assign_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/sound/hda/common/codec.c\0";

static __slate_asm_string_snd_hda_spdif_ctls_unassign_1: [u8; 1] = *b"\0";

static __slate_asm_string_snd_hda_spdif_ctls_unassign_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/sound/hda/common/codec.c\0";

static __slate_asm_string_snd_hda_spdif_default_get_1: [u8; 1] = *b"\0";

static __slate_asm_string_snd_hda_spdif_default_get_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/sound/hda/common/codec.c\0";

static __slate_asm_string_snd_hda_spdif_default_put_1: [u8; 1] = *b"\0";

static __slate_asm_string_snd_hda_spdif_default_put_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/sound/hda/common/codec.c\0";

static __slate_asm_string_snd_hda_spdif_out_switch_get_1: [u8; 1] = *b"\0";

static __slate_asm_string_snd_hda_spdif_out_switch_get_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/sound/hda/common/codec.c\0";

static __slate_asm_string_snd_hda_spdif_out_switch_put_1: [u8; 1] = *b"\0";

static __slate_asm_string_snd_hda_spdif_out_switch_put_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/sound/hda/common/codec.c\0";
