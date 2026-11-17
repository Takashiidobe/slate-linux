
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
//!  Base port operations for 8250/16550-type serial ports
//!
//!  Based on drivers/char/serial.c, by Linus Torvalds, Theodore Ts'o.
//!  Split from 8250_core.c, Copyright (C) 2001 Russell King.
//!
//! A note about mapbase / membase
//!
//!  mapbase is the physical address of the IO port.
//!  membase is an 'ioremapped' cookie.
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
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
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
//! SCHED_ISO: reserved but not implemented yet
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
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! Generic info of form tag = "info"
//! Generate SD flag indexes
//! c_iflag bits
//! Transmitter physically empty
//! Pulse per Second
//! DLAB=0
//! These flags used only by the kernel as part of the
//! irq handling routines.
//!
//! IRQF_SHARED - allow sharing the irq among several devices
//! IRQF_PROBE_SHARED - set by callers when they expect sharing mismatches to occur
//! IRQF_TIMER - Flag to mark this interrupt as timer interrupt
//! IRQF_PERCPU - Interrupt is per cpu
//! IRQF_NOBALANCING - Flag to exclude this interrupt from irq balancing
//! IRQF_IRQPOLL - Interrupt is used for polling (only the interrupt that is
//!                registered first in a shared interrupt is considered for
//!                performance reasons)
//! IRQF_ONESHOT - Interrupt is not reenabled after the hardirq handler finished.
//!                Used by threaded interrupts which need to keep the
//!                irq line disabled until the threaded handler has been run.
//! IRQF_NO_SUSPEND - Do not disable this IRQ during suspend.  Does not guarantee
//!                   that this interrupt will wake the system from a suspended
//!                   state.  See Documentation/power/suspend-and-interrupts.rst
//! IRQF_FORCE_RESUME - Force enable it on resume even if IRQF_NO_SUSPEND is set
//! IRQF_NO_THREAD - Interrupt cannot be threaded
//! IRQF_EARLY_RESUME - Resume IRQ early during syscore instead of at device
//!                resume time.
//! IRQF_COND_SUSPEND - If the IRQ is shared with a NO_SUSPEND user, execute this
//!                interrupt handler after suspending interrupts. For system
//!                wakeup devices users need to implement wakeup detection in
//!                their interrupt handlers.
//! IRQF_NO_AUTOEN - Don't enable IRQ or NMI automatically when users request it.
//!                Users will enable it explicitly by enable_irq() or enable_nmi()
//!                later.
//! IRQF_NO_DEBUG - Exclude from runnaway detection for IPI and similar handlers,
//! 		   depends on IRQF_PERCPU.
//! IRQF_COND_ONESHOT - Agree to do IRQF_ONESHOT if already set for a shared
//!                 interrupt.
//! The type definitions.  These are from Ted Ts'o's serial.h
//! By historical reasons the values from 0 to 13 are defined
//! in the include/uapi/linux/serial.h, do not define them here.
//! Values 0 to 19 are used by setserial from busybox and must never
//! be modified.
//! Runtime PM flag argument bits
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
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
// Don't. Just don't. See commit 771c035372a0 ("deprecate the '__deprecated'
// attribute warnings entirely and for good") for more information.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-deprecated-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-deprecated-type-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-deprecated-variable-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Enumerator-Attributes.html#index-deprecated-enumerator-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#deprecated
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
// "tsc" Time Stamp Counter
// "apic" Onboard APIC
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
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// No static inline type checking - see Subtlety (1) above.
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
// See the comment in asm-generic/io.h about ioremap_np().
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
// Generate SD flag bits
// Ignore break condition
// Signal interrupt on break
// Ignore characters with parity errors
// Mark parity and framing errors
// Enable input parity check
// mark or space (stick) parity
// flow control
// In:  Receive buffer
// Out: Transmit buffer
// Out: Interrupt Enable Register
// Enable Modem status interrupt
// Enable receiver line status interrupt
// Enable Transmitter holding register int.
// Enable receiver data interrupt
// Sleep mode for ST16650 and TI16750.  For the ST16650, EFR[4]=1
// Enable sleep mode
// In:  Interrupt ID Register
// No interrupts pending
// Mask for the interrupt ID
// Transmitter holding register empty
// Receiver data interrupt
// Receiver line status interrupt
// OMAP RX Timeout interrupt
// FIFOs enabled / port type identification
// 8250: no FIFO
// 16550: (broken/unusable) FIFO
// 16550A: FIFO enabled
// 16750: 64 bytes FIFO enabled
// Out: FIFO Control Register
// Enable the FIFO
// Clear the RCVR FIFO
// Clear the XMIT FIFO
// For DMA applications
// Note: The FIFO trigger levels are chip specific:
// 	RX:76 = 00  01  10  11	TX:54 = 00  01  10  11
// PC16550D:	 1   4   8  14		xx  xx  xx  xx
// TI16C550A:	 1   4   8  14          xx  xx  xx  xx
// TI16C550C:	 1   4   8  14          xx  xx  xx  xx
// ST16C550:	 1   4   8  14		xx  xx  xx  xx
// ST16C650:	 8  16  24  28		16   8  24  30	PORT_16650V2
// NS16C552:	 1   4   8  14		xx  xx  xx  xx
// ST16C654:	 8  16  56  60		 8  16  32  56	PORT_16654
// TI16C750:	 1  16  32  56		xx  xx  xx  xx	PORT_16750
// TI16C752:	 8  16  56  60		 8  16  32  56
// OX16C950:	16  32 112 120		16  32  64 112	PORT_16C950
// Tegra:	 1   4   8  14		16   8   4   1	PORT_TEGRA
// Mask for the FIFO trigger range
// Mask for trigger set at 1
// Go into 64 byte mode (TI16C750 and
// some Freescale UARTs)
// Out: Line Control Register
// Note: if the word length is 5 bits (UART_LCR_WLEN5), then setting
// UART_LCR_STOP will select 1.5 stop bits, not 2 stop bits.
// Divisor latch access bit
// Set break control
// Stick parity (?)
// Even parity select
// Parity Enable
// Stop bits: 0=1 bit, 1=2 bits
// Wordlength: 8 bits
// Access to some registers depends on register access / configuration
// mode.
// Configutation mode A
// Configutation mode B
// Out: Modem Control Register
// Enable auto-RTS/CTS (TI16C550C/TI16C750)
// Enable loopback test mode
// Out2 complement
// Out1 complement
// RTS complement
// DTR complement
// In:  Line Status Register
// Fifo error
// Transmitter empty
// Transmit-hold-register empty
// Break interrupt indicator
// Frame error indicator
// Parity error indicator
// Overrun error indicator
// Receiver data ready
// In:  Modem Status Register
// Data Carrier Detect
// Ring Indicator
// Data Set Ready
// Clear to Send
// Delta DCD
// Trailing edge ring indicator
// Delta DSR
// Delta CTS
// I/O: Scratch Register
// DLAB=1
// Out: Divisor Latch Low
// Out: Divisor Latch High
// Max divisor value
// LCR=0xBF (or DLAB=1 for 16C660)
// I/O: Extended Features Register
// I/O: Extended Features Register (XR17D15x)
// CTS flow control
// Enhanced control bit
// LCR=0xBF, XR16C85x
// FCTR bit 7 selects Rx or Tx
// In: Fifo count
// Out: Fifo custom trigger levels
// Feature Control Register
// Tx/Rx 850 programmable trigger select
// Programmable trigger mode select
// Programmable trigger mode select
// UART Unit Enable
// Receiver Time Out Interrupt Enable
// Index Control Register
// The 16950 ICR registers
// Additional Control Register
// ID #1
// ID #2
// ID #3
// Revision
// Channel Software Reset
// Transmitter disable
// ICR Read enable
// Extra registers for TI DA8xx/66AK2x
// PWREMU_MGMT register bits
// Free-running mode
// Receiver reset/enable
// Transmitter reset/enable
// OMAP1510 12MHz osc select
// These are definitions for the Altera ALTR_16550_F32/F64/F128
// Normalized from 0x100 to 0x40 because of shift by 2 (32 bit regs).
// Additional Features Register
// Enable the TX FIFO Low Watermark
// Tx FIFO Low Watermark
// Cavium OCTEON internal UART
// ST-Ericsson U6xxx internal UART
// NVIDIA Tegra internal UART
// Exar XR17D15x UART
// NXP LPC32xx SoC "Standard" UART
// CIR infrared port, has its own driver
// Exar XR17V35x UARTs
// Altera 16550 UART with 32 FIFOs
// Altera 16550 UART with 64 FIFOs
// Altera 16550 UART with 128 FIFOs
// Ralink RT2880 internal UART
// Freescale 16550 UART with 64 FIFOs
// Nuvoton UART
// ASPEED AST2x00 virtual UART
// Moxa MUEx50 UART
// Broadcom BCM7271 UART
// TI DA8xx/66AK2x
// MCHP 16550A UART with 256 byte FIFOs
// MediaTek BTIF
// Sunix UART
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
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
((core::mem::size_of::<[serial8250_config; 122]>() as u64) / (core::mem::size_of::<serial8250_config>() as u64)).wrapping_add(__must_be_array__18!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__10 {
() => {
((core::mem::size_of::<[serial8250_config; 122]>() as u64) / (core::mem::size_of::<serial8250_config>() as u64)).wrapping_add(__must_be_array__2!() as u64)
};
}

macro_rules! BIT_MASK {
($nr:expr) => {
UL!() << ($nr) % 64
};
}

macro_rules! BIT_ULL {
($nr:expr) => {
ULL!() << $nr
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

macro_rules! PRESL {
($x:expr) => {
(($x) as i32) & 48
};
}

/// UART_ENABLE_MS - determine if port should enable modem status irqs
macro_rules! UART_ENABLE_MS {
($port:expr, $cflag:expr) => {
(unsafe { (*$port).flags }) & 2048u64 != 0 || ($cflag) & 2147483648 != 0 || ($cflag) & 2048 == 0
};
}

macro_rules! UART_FCR_R_TRIG_BITS {
($x:expr) => {
(($x) & 192) >> 6i32
};
}

macro_rules! UART_FCR_R_TRIG_BITS__2 {
($x:expr) => {
((($x) as i32) & 192) >> 6i32
};
}

/// Helper for dealing with UART_LCR_WLEN* defines
macro_rules! UART_LCR_WLEN {
($x:expr) => {
(($x) as i32).wrapping_sub(5)
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

macro_rules! _BITUL {
($x:expr) => {
_UL!() << $x
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

macro_rules! __WRITE_ONCE__3 {
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

macro_rules! __irqd_to_state {
($d:expr) => {
unsafe { (*(*$d).common).state_use_accessors }
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

/// helper macro to distinguish between real in place fifo where the fifo
/// array is a part of the structure and the fifo type where the array is
/// outside of the fifo structure.
macro_rules! __is_kfifo_ptr__9 {
() => {
(core::mem::size_of::<__SlateRecord_tty_port_h_118>() as u64) == (core::mem::size_of::<__SlateRecord_serial_core_h_885>() as u64)
};
}

macro_rules! __must_be_array__2 {
() => {
core::mem::size_of::<__SlateRecord_8250_port_c_3177>() as i32
};
}

macro_rules! __must_be_array__18 {
() => {
core::mem::size_of::<__SlateRecord_8250_port_c_3167>() as i32
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

macro_rules! __ratelimit__5 {
($state:expr) => {
unsafe { ___ratelimit($state, c"serial8250_do_startup".as_ptr() as *const u8) }
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

macro_rules! alloc_tag_restore {
() => {
while false {
}
};
}

macro_rules! call_io_delay {
() => {
unsafe { pv_info.io_delay }
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
macro_rules! container_of__2 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof!($type) as isize)) }) as *mut ()) as *mut $type;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__3!() as isize)) }) as *mut ()) as *mut $type;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__29!($type) as isize)) }) as *mut ()) as *mut $type;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__31!($type) as isize)) }) as *mut ()) as *mut $type;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__15!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
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

/// kfifo_is_empty - returns true if the fifo is empty
/// @fifo: address of the fifo to be used
macro_rules! kfifo_is_empty {
($fifo:expr) => {
{
let mut __v: bool;
let __v2: *mut __SlateRecord_tty_port_h_118 = $fifo;
__v = (unsafe { (*__v2).__slate_anon_0.kfifo.r#in }) == unsafe { (*__v2).__slate_anon_0.kfifo.out };
__v
}
};
}

/// kfifo_len - returns the number of used elements in the fifo
/// @fifo: address of the fifo to be used
macro_rules! kfifo_len {
($fifo:expr) => {
{
let mut __v: u32;
let __v2: *mut __SlateRecord_tty_port_h_118 = $fifo;
__v = unsafe { (*__v2).__slate_anon_0.kfifo.r#in }.wrapping_sub(unsafe { (*__v2).__slate_anon_0.kfifo.out });
__v
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

macro_rules! likely__2 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! lockdep_assert_held_once {
($l:expr) => {
loop {
$l;
if true {
break;
}
}
};
}

/// mdelay - Inserting a delay based on milliseconds with busy waiting
/// @n:	requested delay in milliseconds
///
/// See udelay() for basic information about mdelay() and it's variants.
///
/// Please double check, whether mdelay() is the right way to go or whether a
/// refactoring of the code is the better variant to be able to use msleep()
/// instead.
macro_rules! mdelay {
($n:expr) => {
if true && ($n) <= 5 {
udelay($n.wrapping_mul(1000) as u64);
} else {
let mut __ms: u64 = ($n) as u64;
loop {
let __v: u64 = __ms;
__ms = __v.wrapping_sub(1);
if __v == 0 {
break;
}
udelay(1000u64);
}
}
};
}

/// mdelay - Inserting a delay based on milliseconds with busy waiting
/// @n:	requested delay in milliseconds
///
/// See udelay() for basic information about mdelay() and it's variants.
///
/// Please double check, whether mdelay() is the right way to go or whether a
/// refactoring of the code is the better variant to be able to use msleep()
/// instead.
macro_rules! mdelay__2 {
($n:expr) => {
if (unsafe { __slate_intrinsic_1c5364fcd9da51fe($n) }) && ($n) <= 5 {
udelay($n.wrapping_mul(1000) as u64);
} else {
let mut __ms: u64 = ($n) as u64;
loop {
let __v: u64 = __ms;
__ms = __v.wrapping_sub(1);
if __v == 0 {
break;
}
udelay(1000u64);
}
}
};
}

macro_rules! offsetof__15 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, console_msr_work) as u64
};
}

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, port) as u64
};
}

macro_rules! offsetof__29 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, stop_tx_timer) as u64
};
}

macro_rules! offsetof__3 {
() => {
0u64
};
}

macro_rules! offsetof__31 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, start_tx_timer) as u64
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

macro_rules! raw_spin_lock_irqsave__2 {
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

macro_rules! release_mem_region {
($start:expr, $n:expr) => {
unsafe { __release_region(core::ptr::addr_of_mut!(iomem_resource), $start, ($n) as u64) };
};
}

/// Compatibility cruft
macro_rules! release_region {
($start:expr, $n:expr) => {
unsafe { __release_region(core::ptr::addr_of_mut!(ioport_resource), $start, ($n) as u64) };
};
}

macro_rules! request_mem_region {
($start:expr, $n:expr, $name:expr) => {
unsafe { __request_region(core::ptr::addr_of_mut!(iomem_resource), $start, ($n) as u64, ($name) as *const u8, 0i32) }
};
}

/// Convenience shorthand with allocation
macro_rules! request_region {
($start:expr, $n:expr, $name:expr) => {
unsafe { __request_region(core::ptr::addr_of_mut!(ioport_resource), $start, ($n) as u64, ($name) as *const u8, 0i32) }
};
}

macro_rules! scoped_tty {
($scope:expr) => {
class_tty_port_tty_lock_ptr(core::ptr::addr_of_mut!($scope)) as *mut tty_struct
};
}

macro_rules! spin_lock_irqsave__2 {
($lock:expr, $flags:expr) => {
loop {
raw_spin_lock_irqsave__2!(spinlock_check($lock), $flags);
__release!();
__acquire!();
if true {
break;
}
}
};
}

macro_rules! test_bit {
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

macro_rules! test_bit__2 {
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
time_after__2!($b, $a)
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

macro_rules! uart_console {
($port:expr) => {
!unsafe { (*$port).cons }.is_null() && ((unsafe { (*(*$port).cons).index }) as u32) == unsafe { (*$port).line }
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

macro_rules! wake_up_interruptible {
($x:expr) => {
unsafe { __wake_up($x, 1u32, 1i32, core::ptr::null_mut::<()>()) };
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut pv_info: pv_info;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut jiffies: u64;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut ioport_resource: resource;
    static mut iomem_resource: resource;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn ___ratelimit(arg0: *mut ratelimit_state, arg1: *const u8) -> i32;
    fn nbcon_device_try_acquire(arg0: *mut console) -> bool;
    fn nbcon_device_release(arg0: *mut console);
    fn __compiletime_assert_13();
    fn __bad_size_call_parameter();
    fn __compiletime_assert_29();
    fn __xchg_wrong_size();
    fn kstrtou8(arg0: *const u8, arg1: u32, arg2: *mut u8) -> i32;
    fn irq_get_nr_irqs() -> u32;
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn __wake_up(arg0: *mut wait_queue_head, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn down_write(arg0: *mut rw_semaphore);
    fn up_write(arg0: *mut rw_semaphore);
    fn ktime_get_mono_fast_ns() -> u64;
    fn hrtimer_setup(arg0: *mut hrtimer, arg1: Option<unsafe extern "C" fn(*mut hrtimer) -> u32>, arg2: i32, arg3: u32);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn hrtimer_cancel(arg0: *mut hrtimer) -> i32;
    fn hrtimer_try_to_cancel(arg0: *mut hrtimer) -> i32;
    fn __compiletime_assert_361();
    fn up(arg0: *mut semaphore);
    fn __compiletime_assert_386();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_392();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn sysfs_emit(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn __request_region(arg0: *mut resource, arg1: u64, arg2: u64, arg3: *const u8, arg4: i32) -> *mut resource;
    fn __release_region(arg0: *mut resource, arg1: u64, arg2: u64);
    fn ioremap(arg0: u64, arg1: u64) -> *mut ();
    fn iounmap(arg0: *mut ());
    fn native_io_delay();
    fn ioread32be(arg0: *const ()) -> u32;
    fn iowrite32be(arg0: u32, arg1: *mut ());
    fn irq_work_queue(arg0: *mut irq_work) -> bool;
    fn irq_work_sync(arg0: *mut irq_work);
    fn irq_get_irq_data(arg0: u32) -> *mut irq_data;
    fn nbcon_enter_unsafe(arg0: *mut nbcon_write_context) -> bool;
    fn nbcon_exit_unsafe(arg0: *mut nbcon_write_context) -> bool;
    fn nbcon_reacquire_nobuf(arg0: *mut nbcon_write_context);
    fn handle_sysrq(arg0: u8);
    fn sysrq_mask() -> i32;
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn _dev_info(arg0: *const device, arg1: *const u8, ...);
    fn pm_wakeup_dev_event(arg0: *mut device, arg1: u32, arg2: bool);
    fn __kfifo_out_r(arg0: *mut __kfifo, arg1: *mut (), arg2: u32, arg3: u64) -> u32;
    fn tty_port_tty_get(arg0: *mut tty_port) -> *mut tty_struct;
    fn tty_kref_put(arg0: *mut tty_struct);
    fn do_SAK(arg0: *mut tty_struct);
    fn tty_termios_baud_rate(arg0: *const ktermios) -> u32;
    fn tty_termios_encode_baud_rate(arg0: *mut ktermios, arg1: u32, arg2: u32);
    fn tty_get_char_size(arg0: u32) -> u8;
    fn tty_flip_buffer_push(arg0: *mut tty_port);
    fn synchronize_irq(arg0: u32);
    fn disable_irq(arg0: u32);
    fn enable_irq(arg0: u32);
    fn probe_irq_on() -> u64;
    fn probe_irq_off(arg0: u64) -> i32;
    fn uart_write_wakeup(arg0: *mut uart_port);
    fn uart_update_timeout(arg0: *mut uart_port, arg1: u32, arg2: u32);
    fn uart_get_baud_rate(arg0: *mut uart_port, arg1: *mut ktermios, arg2: *const ktermios, arg3: u32, arg4: u32) -> u32;
    fn uart_get_divisor(arg0: *mut uart_port, arg1: u32) -> u32;
    fn uart_parse_options(arg0: *const u8, arg1: *mut i32, arg2: *mut i32, arg3: *mut i32, arg4: *mut i32);
    fn uart_set_options(arg0: *mut uart_port, arg1: *mut console, arg2: i32, arg3: i32, arg4: i32, arg5: i32) -> i32;
    fn uart_console_write(arg0: *mut uart_port, arg1: *const u8, arg2: u32, arg3: Option<unsafe extern "C" fn(*mut uart_port, u8)>);
    fn uart_handle_dcd_change(arg0: *mut uart_port, arg1: bool);
    fn uart_handle_cts_change(arg0: *mut uart_port, arg1: bool);
    fn uart_insert_char(arg0: *mut uart_port, arg1: u32, arg2: u32, arg3: u8, arg4: u8);
    fn uart_xchar_out(arg0: *mut uart_port, arg1: i32);
    fn uart_try_toggle_sysrq(arg0: *mut uart_port, arg1: u8) -> bool;
    fn uart_iotype_mmio(arg0: i32) -> bool;
    fn uart_iotype_io(arg0: i32) -> bool;
    fn __pm_runtime_idle(arg0: *mut device, arg1: i32) -> i32;
    fn __pm_runtime_suspend(arg0: *mut device, arg1: i32) -> i32;
    fn __pm_runtime_resume(arg0: *mut device, arg1: i32) -> i32;
    fn __compiletime_assert_543();
    fn rsa_enable(arg0: *mut uart_8250_port);
    fn rsa_disable(arg0: *mut uart_8250_port);
    fn rsa_autoconfig(arg0: *mut uart_8250_port);
    fn rsa_reset(arg0: *mut uart_8250_port);
    fn serial8250_tx_dma(arg0: *mut uart_8250_port) -> i32;
    fn serial8250_tx_dma_flush(arg0: *mut uart_8250_port);
    fn serial8250_rx_dma(arg0: *mut uart_8250_port) -> i32;
    fn serial8250_rx_dma_flush(arg0: *mut uart_8250_port);
    fn serial8250_request_dma(arg0: *mut uart_8250_port) -> i32;
    fn serial8250_release_dma(arg0: *mut uart_8250_port);
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
fn hlist_unhashed_lockless(h: *const hlist_node) -> i32 {
let mut __v0: *mut *mut hlist_node;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut *mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_13() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*h).pprev) as *const *mut *mut hlist_node) };
__v0.is_null() as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timespec64 {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pv_info {
    io_delay: bool,
    name: *const u8,
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


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD__2!(nr) as isize);
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
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
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
    __slate_bits_0: __slate_bits::__SlateBits120U0,
}


#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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
    __slate_bits_0: __slate_bits::__SlateBits125U0,
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[inline(always)]
fn native_pause() {
unsafe {
core::arch::asm!("pause", options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[inline(always)]
fn cpu_relax() {
native_pause();
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
    __slate_bits_0: __slate_bits::__SlateBits126U0,
    pkru: u32,
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag_counters {
    bytes: u64,
    calls: u64,
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
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
}


fn mem_alloc_profiling_enabled() -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
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
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct seccomp {
    mode: i32,
    filter_count: __SlateRecord_types_h_190,
    filter: *mut seccomp_filter,
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
struct workqueue_struct {
}


#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn spin_unlock_irq(lock: *mut spinlock) {
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
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
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
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
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
    __slate_bits_0: __slate_bits::__SlateBits285U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
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
struct user_namespace {
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
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mutex_h_253 {
    lock: *mut mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[inline(always)]
fn class_mutex_destructor(_T: *mut __SlateRecord_mutex_h_253) {
unsafe { mutex_unlock((*_T).lock) };
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
struct uprobe {
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
struct __SlateRecord_rwsem_h_270 {
    lock: *mut rw_semaphore,
}


#[inline(always)]
fn class_rwsem_write_destructor(_T: *mut __SlateRecord_rwsem_h_270) {
unsafe { up_write((*_T).lock) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
}


#[inline(always)]
fn class_rwsem_write_constructor(l: *mut rw_semaphore) -> __SlateRecord_rwsem_h_270 {
let mut _t: __SlateRecord_rwsem_h_270 = __SlateRecord_rwsem_h_270 { lock: l };
let _T: *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(_t);
unsafe { down_write((*_T).lock) };
return _t;
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


#[inline(always)]
fn __class_rwsem_write_cleanup_ctx(_T: *mut *mut __SlateRecord_rwsem_h_270) {
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
struct xol_area {
}


fn ns_to_ktime(ns: u64) -> i64 {
ns as i64
}

fn ms_to_ktime(ms: u64) -> i64 {
ms.wrapping_mul(1000000) as i64
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
struct ldt_struct {
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
struct mem_cgroup {
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
struct uprobes_state {
    xol_area: *mut xol_area,
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
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
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
struct folio_queue {
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
struct vm_userfaultfd_ctx {
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
struct kmem_cache {
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
struct mnt_idmap {
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
struct assoc_array_ptr {
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
struct key_type {
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
struct list_lru {
    node: *mut list_lru_node,
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
struct hlist_bl_node {
    next: *mut hlist_bl_node,
    pprev: *mut *mut hlist_bl_node,
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
struct taskstats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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

fn hrtimer_start(timer: *mut hrtimer, tim: i64, mode: u32) {
unsafe { hrtimer_start_range_ns(timer, tim, 0u64, mode) };
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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
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
    __slate_bits_0: __slate_bits::__SlateBits288U0,
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
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
struct __SlateRecord_projid_h_24 {
    val: u32,
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
struct kqid {
    __slate_anon_0: __SlateRecord_quota_h_69,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct semaphore_waiter {
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
struct fs_context {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct poll_table_struct {
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
struct quota_format_type {
    qf_fmt_id: i32,
    qf_ops: *const quota_format_ops,
    qf_owner: *mut module,
    qf_next: *mut quota_format_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
struct io_uring_cmd {
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
struct semaphore {
    lock: raw_spinlock,
    count: u32,
    first_waiter: *mut semaphore_waiter,
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
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
    __slate_bits_0: __slate_bits::__SlateBits572U0,
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
// kmalloc_obj - Allocate a single instance of the given type
// @VAR_OR_TYPE: Variable or type to allocate.
// @...: optional GFP flags for the allocation (GFP_KERNEL when not specified).
//
// Returns: newly allocated pointer to a @VAR_OR_TYPE on success, or NULL
// on failure.
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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
struct msi_msg {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_desc {
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
struct proc_dir_entry {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct printk_buffers {
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
struct gpio_desc {
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
    __slate_bits_0: __slate_bits::__SlateBits641U0,
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
struct wake_irq {
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
struct dev_pm_qos {
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
    __slate_bits_0: __slate_bits::__SlateBits583U0,
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
// Resource extended types
// No address assigned yet
#[inline(always)]
fn __outb(mut value: u8, mut port: u16) {
unsafe {
core::arch::asm!("outb %al, %dx", in("ax") (value as u8) as u32, in("dx") port as u16, options(nostack, att_syntax));
}
}

#[inline(always)]
fn __inb(mut port: u16) -> u8 {
let mut value: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("inb %dx, %al", lateout("ax") __t0, in("dx") port as u16, options(nostack, att_syntax));
value = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
value
}

fn readb(mut addr: *const ()) -> u8 {
let mut ret: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb ({1:r}),{0:l}", lateout(reg) __t0, in(reg) addr as *mut u8, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
ret
}

fn readw(mut addr: *const ()) -> u16 {
let mut ret: u16 = 0;
unsafe {
let __t0: u16;
core::arch::asm!("movw ({1:r}),{0:x}", lateout(reg) __t0, in(reg) addr as *mut u16, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u16, u16>(__t0 as u16) };
}
ret
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_tty_buffer_h_11 {
    next: *mut tty_buffer,
    free: llist_node,
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

fn writeb(mut val: u8, mut addr: *mut ()) {
unsafe {
core::arch::asm!("movb {0:l},({1:r})", in(reg) (val as u8) as u32, in(reg) addr as *mut u8, options(nostack, att_syntax));
}
}

fn writew(mut val: u16, mut addr: *mut ()) {
unsafe {
core::arch::asm!("movw {0:x},({1:r})", in(reg) val as u16, in(reg) addr as *mut u16, options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_tty_port_h_118 {
    __slate_anon_0: __SlateRecord_tty_port_h_118_2,
    buf: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_tty_port_h_118_2 {
    kfifo: __kfifo,
    r#type: *mut u8,
    const_type: *const u8,
    rectype: *mut [u8; 0],
    ptr: *mut u8,
    ptr_const: *const u8,
}


fn writel(mut val: u32, mut addr: *mut ()) {
unsafe {
core::arch::asm!("movl {0:e},({1:r})", in(reg) val as u32, in(reg) addr as *mut u32, options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_tty_h_212 {
    lock: spinlock,
    stopped: bool,
    tco_stopped: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_tty_h_218 {
    pgrp: *mut pid,
    session: *mut pid,
    lock: spinlock,
    pktstatus: u8,
    packet: bool,
}


fn slow_down_io() {
if !call_io_delay!() {
return;
}
unsafe { native_io_delay() };
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_serial_h_161 {
    padding: [u32; 5],
    __slate_anon_1: __SlateRecord_serial_h_164,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_serial_h_164 {
    addr_recv: u8,
    addr_dest: u8,
    padding0: [u8; 2],
    padding1: [u32; 4],
}


fn outb_p(value: u8, port: u16) {
__outb(value, port);
slow_down_io();
}

fn inb_p(port: u16) -> u8 {
let value: u8 = __inb(port);
slow_down_io();
value
}

#[repr(C)]
#[derive(Clone, Copy)]
struct serial_port_device {
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
struct __SlateRecord_serial_core_h_885 {
    __slate_anon_0: __SlateRecord_serial_core_h_885_2,
    buf: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_serial_core_h_885_2 {
    kfifo: __kfifo,
    r#type: *mut u8,
    const_type: *const u8,
    rectype: *mut [u8; 0],
    ptr: *mut u8,
    ptr_const: *const u8,
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


fn irqd_is_wakeup_set(d: *mut irq_data) -> bool {
__irqd_to_state!(d) & 16384 != 0
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
struct nbcon_context {
    console: *mut console,
    spinwait_max_us: u32,
    prio: u32,
    __slate_bits_0: __slate_bits::__SlateBits691U0,
    pbufs: *mut printk_buffers,
    seq: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mctrl_gpios {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nbcon_write_context {
    ctxt: nbcon_context,
    outbuf: *mut u8,
    len: u32,
    unsafe_takeover: bool,
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
struct console {
    name: [u8; 16],
    write: Option<unsafe extern "C" fn(*mut console, *const u8, u32)>,
    read: Option<unsafe extern "C" fn(*mut console, *mut u8, u32) -> i32>,
    device: Option<unsafe extern "C" fn(*mut console, *mut i32) -> *mut tty_driver>,
    unblank: Option<unsafe extern "C" fn()>,
    setup: Option<unsafe extern "C" fn(*mut console, *mut u8) -> i32>,
    exit: Option<unsafe extern "C" fn(*mut console) -> i32>,
    r#match: Option<unsafe extern "C" fn(*mut console, *mut u8, i32, *mut u8) -> i32>,
    flags: i16,
    index: i16,
    cflag: i32,
    ispeed: u32,
    ospeed: u32,
    seq: u64,
    dropped: u64,
    data: *mut (),
    node: hlist_node,
    write_atomic: Option<unsafe extern "C" fn(*mut console, *mut nbcon_write_context)>,
    write_thread: Option<unsafe extern "C" fn(*mut console, *mut nbcon_write_context)>,
    device_lock: Option<unsafe extern "C" fn(*mut console, *mut u64)>,
    device_unlock: Option<unsafe extern "C" fn(*mut console, u64)>,
    nbcon_state: __SlateRecord_types_h_190,
    nbcon_seq: __SlateRecord_types_h_197,
    nbcon_device_ctxt: nbcon_context,
    nbcon_prev_seq: __SlateRecord_types_h_197,
    pbufs: *mut printk_buffers,
    kthread: *mut task_struct,
    rcuwait: rcuwait,
    irq_work: irq_work,
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
    __slate_bits_0: __slate_bits::__SlateBits706U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits706U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits706U2,
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
    __slate_bits_3: __slate_bits::__SlateBits706U3,
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
struct __SlateRecord_8250_port_c_3110 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_8250_port_c_3110_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_8250_port_c_3110_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_8250_port_c_3110_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_8250_port_c_3110_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_8250_port_c_3110_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_8250_port_c_3167 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_8250_port_c_3177 {
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
    pub struct __SlateBits285U0 {
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
    pub struct __SlateBits288U0 {
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
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits706U0 {
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
    pub struct __SlateBits703U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits706U1 {
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
    pub struct __SlateBits706U2 {
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
    pub struct __SlateBits706U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits756U0 {
        #[bits(1)]
        pub console: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
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
    pub struct __SlateBits120U0 {
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
    pub struct __SlateBits126U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits572U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits641U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits583U0 {
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
    pub struct __SlateBits125U0 {
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
    pub struct __SlateBits691U0 {
        #[bits(1)]
        pub allow_unsafe_takeover: u32,
        #[bits(1)]
        pub backlog: u32,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits797U0 {
        #[bits(1)]
        pub tx_stopped: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
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
    __slate_bits_0: __slate_bits::__SlateBits703U0,
}


fn pm_wakeup_event(dev: *mut device, msec: u32) {
unsafe { pm_wakeup_dev_event(dev, msec, false) };
}

fn dev_get_drvdata(dev: *const device) -> *mut () {
unsafe { (*dev).driver_data }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ktermios {
    c_iflag: u32,
    c_oflag: u32,
    c_cflag: u32,
    c_lflag: u32,
    c_line: u8,
    c_cc: [u8; 19],
    c_ispeed: u32,
    c_ospeed: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct winsize {
    ws_row: u16,
    ws_col: u16,
    ws_xpixel: u16,
    ws_ypixel: u16,
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
struct tty_operations {
    lookup: Option<unsafe extern "C" fn(*mut tty_driver, *mut file, i32) -> *mut tty_struct>,
    install: Option<unsafe extern "C" fn(*mut tty_driver, *mut tty_struct) -> i32>,
    remove: Option<unsafe extern "C" fn(*mut tty_driver, *mut tty_struct)>,
    open: Option<unsafe extern "C" fn(*mut tty_struct, *mut file) -> i32>,
    close: Option<unsafe extern "C" fn(*mut tty_struct, *mut file)>,
    shutdown: Option<unsafe extern "C" fn(*mut tty_struct)>,
    cleanup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    write: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, u64) -> i64>,
    put_char: Option<unsafe extern "C" fn(*mut tty_struct, u8) -> i32>,
    flush_chars: Option<unsafe extern "C" fn(*mut tty_struct)>,
    write_room: Option<unsafe extern "C" fn(*mut tty_struct) -> u32>,
    chars_in_buffer: Option<unsafe extern "C" fn(*mut tty_struct) -> u32>,
    ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i32>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i64>,
    set_termios: Option<unsafe extern "C" fn(*mut tty_struct, *const ktermios)>,
    throttle: Option<unsafe extern "C" fn(*mut tty_struct)>,
    unthrottle: Option<unsafe extern "C" fn(*mut tty_struct)>,
    stop: Option<unsafe extern "C" fn(*mut tty_struct)>,
    start: Option<unsafe extern "C" fn(*mut tty_struct)>,
    hangup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    break_ctl: Option<unsafe extern "C" fn(*mut tty_struct, i32) -> i32>,
    flush_buffer: Option<unsafe extern "C" fn(*mut tty_struct)>,
    ldisc_ok: Option<unsafe extern "C" fn(*mut tty_struct, i32) -> i32>,
    set_ldisc: Option<unsafe extern "C" fn(*mut tty_struct)>,
    wait_until_sent: Option<unsafe extern "C" fn(*mut tty_struct, i32)>,
    send_xchar: Option<unsafe extern "C" fn(*mut tty_struct, u8)>,
    tiocmget: Option<unsafe extern "C" fn(*mut tty_struct) -> i32>,
    tiocmset: Option<unsafe extern "C" fn(*mut tty_struct, u32, u32) -> i32>,
    resize: Option<unsafe extern "C" fn(*mut tty_struct, *mut winsize) -> i32>,
    get_icount: Option<unsafe extern "C" fn(*mut tty_struct, *mut serial_icounter_struct) -> i32>,
    get_serial: Option<unsafe extern "C" fn(*mut tty_struct, *mut serial_struct) -> i32>,
    set_serial: Option<unsafe extern "C" fn(*mut tty_struct, *mut serial_struct) -> i32>,
    show_fdinfo: Option<unsafe extern "C" fn(*mut tty_struct, *mut seq_file)>,
    proc_show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_driver {
    kref: kref,
    cdevs: *mut *mut cdev,
    owner: *mut module,
    driver_name: *const u8,
    name: *const u8,
    name_base: i32,
    major: i32,
    minor_start: i32,
    num: u32,
    r#type: u32,
    subtype: u32,
    init_termios: ktermios,
    flags: u64,
    proc_entry: *mut proc_dir_entry,
    other: *mut tty_driver,
    flip_wq: *mut workqueue_struct,
    ttys: *mut *mut tty_struct,
    ports: *mut *mut tty_port,
    termios: *mut *mut ktermios,
    driver_state: *mut (),
    ops: *const tty_operations,
    tty_drivers: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ld_semaphore {
    count: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    wait_readers: u32,
    read_wait: list_head,
    write_wait: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_ldisc_ops {
    name: *mut u8,
    num: i32,
    open: Option<unsafe extern "C" fn(*mut tty_struct) -> i32>,
    close: Option<unsafe extern "C" fn(*mut tty_struct)>,
    flush_buffer: Option<unsafe extern "C" fn(*mut tty_struct)>,
    read: Option<unsafe extern "C" fn(*mut tty_struct, *mut file, *mut u8, u64, *mut *mut (), u64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut tty_struct, *mut file, *const u8, u64) -> i64>,
    ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i32>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut tty_struct, u32, u64) -> i32>,
    set_termios: Option<unsafe extern "C" fn(*mut tty_struct, *const ktermios)>,
    poll: Option<unsafe extern "C" fn(*mut tty_struct, *mut file, *mut poll_table_struct) -> u32>,
    hangup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    receive_buf: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, *const u8, u64)>,
    write_wakeup: Option<unsafe extern "C" fn(*mut tty_struct)>,
    dcd_change: Option<unsafe extern "C" fn(*mut tty_struct, bool)>,
    receive_buf2: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, *const u8, u64) -> u64>,
    lookahead_buf: Option<unsafe extern "C" fn(*mut tty_struct, *const u8, *const u8, u64)>,
    owner: *mut module,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_ldisc {
    ops: *const tty_ldisc_ops,
    tty: *mut tty_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __kfifo {
    r#in: u32,
    out: u32,
    mask: u32,
    esize: u32,
    data: *mut (),
}


// DECLARE_KFIFO_PTR - macro to declare a fifo pointer object
// @fifo: name of the declared fifo
// @type: type of the fifo elements
fn __kfifo_uint_must_check_helper(val: u32) -> u32 {
val
}

// kfifo_skip_count - skip output data
// @fifo: address of the fifo to be used
// @count: count of data to skip
// kfifo_get - get data from the fifo
// @fifo: address of the fifo to be used
// @val: address where to store the data
//
// This macro reads the data from the fifo.
// It returns 0 if the fifo was empty. Otherwise it returns the number
// processed elements.
//
// Note that with only one concurrent reader and one concurrent
// writer, you don't need extra locking to use these macro.
// kfifo_out - get data from the fifo
// @fifo: address of the fifo to be used
// @buf: pointer to the storage buffer
// @n: max. number of elements to get
//
// This macro gets some data from the fifo and returns the numbers of elements
// copied.
//
// Note that with only one concurrent reader and one concurrent
// writer, you don't need extra locking to use these macro.
#[repr(C)]
#[derive(Clone, Copy)]
struct tty_buffer {
    __slate_anon_0: __SlateRecord_tty_buffer_h_11,
    used: u32,
    size: u32,
    commit: u32,
    lookahead: u32,
    read: u32,
    flags: bool,
    __slate_pad_7: [u8; 3],
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_bufhead {
    head: *mut tty_buffer,
    flip_wq: *mut workqueue_struct,
    work: work_struct,
    lock: mutex,
    priority: __SlateRecord_types_h_190,
    sentinel: tty_buffer,
    free: llist_head,
    mem_used: __SlateRecord_types_h_190,
    mem_limit: i32,
    tail: *mut tty_buffer,
}


// When a break, frame error, or parity error happens, these codes are
// stuffed into the flags buffer.
#[repr(C)]
#[derive(Clone, Copy)]
struct tty_port_operations {
    carrier_raised: Option<unsafe extern "C" fn(*mut tty_port) -> bool>,
    dtr_rts: Option<unsafe extern "C" fn(*mut tty_port, bool)>,
    shutdown: Option<unsafe extern "C" fn(*mut tty_port)>,
    activate: Option<unsafe extern "C" fn(*mut tty_port, *mut tty_struct) -> i32>,
    destruct: Option<unsafe extern "C" fn(*mut tty_port)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_port_client_operations {
    receive_buf: Option<unsafe extern "C" fn(*mut tty_port, *const u8, *const u8, u64) -> u64>,
    lookahead_buf: Option<unsafe extern "C" fn(*mut tty_port, *const u8, *const u8, u64)>,
    write_wakeup: Option<unsafe extern "C" fn(*mut tty_port)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_port {
    buf: tty_bufhead,
    tty: *mut tty_struct,
    itty: *mut tty_struct,
    ops: *const tty_port_operations,
    client_ops: *const tty_port_client_operations,
    lock: spinlock,
    blocked_open: i32,
    count: i32,
    open_wait: wait_queue_head,
    delta_msr_wait: wait_queue_head,
    flags: u64,
    iflags: u64,
    __slate_bits_0: __slate_bits::__SlateBits756U0,
    mutex: mutex,
    buf_mutex: mutex,
    xmit_buf: *mut u8,
    xmit_fifo: __SlateRecord_tty_port_h_118,
    close_delay: u32,
    closing_wait: u32,
    drain_delay: i32,
    kref: kref,
    client_data: *mut (),
}


// tty_port::iflags bits -- use atomic bit ops
// device is initialized
// device is suspended
// device is open
// uart drivers: use the uart_port::status field and the UPSTAT_* defines
// for s/w-based flow control steering and carrier detection status
// h/w flow control enabled
// carrier detect enabled
// device exclusively opened by
// kernel
fn tty_port_initialized(port: *const tty_port) -> bool {
test_bit__2!(0i32, unsafe { core::ptr::addr_of!((*port).iflags) }) != 0
}

static class_tty_port_tty_is_conditional: bool = 1i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_tty_port_h_271 {
    lock: *mut tty_struct,
}


#[inline(always)]
fn class_tty_port_tty_destructor(_T: *mut __SlateRecord_tty_port_h_271) {
unsafe { tty_kref_put((*_T).lock) };
}

#[inline(always)]
fn class_tty_port_tty_lock_ptr(_T: *mut __SlateRecord_tty_port_h_271) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*_T).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

fn class_tty_port_tty_constructor(tport: *mut tty_port) -> __SlateRecord_tty_port_h_271 {
let _t: __SlateRecord_tty_port_h_271 = __SlateRecord_tty_port_h_271 { lock: unsafe { tty_port_tty_get(tport) } };
return _t;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct tty_struct {
    kref: kref,
    index: i32,
    dev: *mut device,
    driver: *mut tty_driver,
    port: *mut tty_port,
    ops: *const tty_operations,
    ldisc: *mut tty_ldisc,
    ldisc_sem: ld_semaphore,
    atomic_write_lock: mutex,
    legacy_mutex: mutex,
    throttle_mutex: mutex,
    termios_rwsem: rw_semaphore,
    winsize_mutex: mutex,
    termios: ktermios,
    termios_locked: ktermios,
    name: [u8; 64],
    flags: u64,
    count: i32,
    receive_room: u32,
    winsize: winsize,
    flow: __SlateRecord_tty_h_212,
    ctrl: __SlateRecord_tty_h_218,
    hw_stopped: bool,
    closing: bool,
    flow_change: i32,
    link: *mut tty_struct,
    fasync: *mut fasync_struct,
    write_wait: wait_queue_head,
    read_wait: wait_queue_head,
    break_wait: wait_queue_head,
    hangup_work: work_struct,
    disc_data: *mut (),
    driver_data: *mut (),
    files_lock: spinlock,
    write_cnt: i32,
    write_buf: *mut u8,
    tty_files: list_head,
    SAK_work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serial_struct {
    r#type: i32,
    line: i32,
    port: u32,
    irq: i32,
    flags: i32,
    xmit_fifo_size: i32,
    custom_divisor: i32,
    baud_base: i32,
    close_delay: u16,
    io_type: u8,
    reserved_char: [u8; 1],
    hub6: i32,
    closing_wait: u16,
    closing_wait2: u16,
    iomem_base: *mut u8,
    iomem_reg_shift: u16,
    port_high: u32,
    iomap_base: u64,
}


// These are the supported serial types.
// Oxford Semiconductor
// RSA-DV II/S card
#[repr(C)]
#[derive(Clone, Copy)]
struct serial_icounter_struct {
    cts: i32,
    dsr: i32,
    rng: i32,
    dcd: i32,
    rx: i32,
    tx: i32,
    frame: i32,
    overrun: i32,
    parity: i32,
    brk: i32,
    buf_overrun: i32,
    reserved: [i32; 9],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serial_rs485 {
    flags: u32,
    delay_rts_before_send: u32,
    delay_rts_after_send: u32,
    __slate_anon_3: __SlateRecord_serial_h_161,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serial_iso7816 {
    flags: u32,
    tg: u32,
    sc_fi: u32,
    sc_di: u32,
    clk: u32,
    reserved: [u32; 5],
}


// FIFO and shifting register empty
fn uart_lsr_tx_empty(lsr: u16) -> bool {
(lsr as i32) & (64 | 32) == 64 | 32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uart_ops {
    tx_empty: Option<unsafe extern "C" fn(*mut uart_port) -> u32>,
    set_mctrl: Option<unsafe extern "C" fn(*mut uart_port, u32)>,
    get_mctrl: Option<unsafe extern "C" fn(*mut uart_port) -> u32>,
    stop_tx: Option<unsafe extern "C" fn(*mut uart_port)>,
    start_tx: Option<unsafe extern "C" fn(*mut uart_port)>,
    throttle: Option<unsafe extern "C" fn(*mut uart_port)>,
    unthrottle: Option<unsafe extern "C" fn(*mut uart_port)>,
    send_xchar: Option<unsafe extern "C" fn(*mut uart_port, u8)>,
    stop_rx: Option<unsafe extern "C" fn(*mut uart_port)>,
    start_rx: Option<unsafe extern "C" fn(*mut uart_port)>,
    enable_ms: Option<unsafe extern "C" fn(*mut uart_port)>,
    break_ctl: Option<unsafe extern "C" fn(*mut uart_port, i32)>,
    startup: Option<unsafe extern "C" fn(*mut uart_port) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut uart_port)>,
    flush_buffer: Option<unsafe extern "C" fn(*mut uart_port)>,
    set_termios: Option<unsafe extern "C" fn(*mut uart_port, *mut ktermios, *const ktermios)>,
    set_ldisc: Option<unsafe extern "C" fn(*mut uart_port, *mut ktermios)>,
    pm: Option<unsafe extern "C" fn(*mut uart_port, u32, u32)>,
    r#type: Option<unsafe extern "C" fn(*mut uart_port) -> *const u8>,
    release_port: Option<unsafe extern "C" fn(*mut uart_port)>,
    request_port: Option<unsafe extern "C" fn(*mut uart_port) -> i32>,
    config_port: Option<unsafe extern "C" fn(*mut uart_port, i32)>,
    verify_port: Option<unsafe extern "C" fn(*mut uart_port, *mut serial_struct) -> i32>,
    ioctl: Option<unsafe extern "C" fn(*mut uart_port, u32, u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uart_icount {
    cts: u32,
    dsr: u32,
    rng: u32,
    dcd: u32,
    rx: u32,
    tx: u32,
    frame: u32,
    overrun: u32,
    parity: u32,
    brk: u32,
    buf_overrun: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uart_port {
    lock: spinlock,
    iobase: u64,
    membase: *mut u8,
    serial_in: Option<unsafe extern "C" fn(*mut uart_port, u32) -> u32>,
    serial_out: Option<unsafe extern "C" fn(*mut uart_port, u32, u32)>,
    set_termios: Option<unsafe extern "C" fn(*mut uart_port, *mut ktermios, *const ktermios)>,
    set_ldisc: Option<unsafe extern "C" fn(*mut uart_port, *mut ktermios)>,
    get_mctrl: Option<unsafe extern "C" fn(*mut uart_port) -> u32>,
    set_mctrl: Option<unsafe extern "C" fn(*mut uart_port, u32)>,
    get_divisor: Option<unsafe extern "C" fn(*mut uart_port, u32, *mut u32) -> u32>,
    set_divisor: Option<unsafe extern "C" fn(*mut uart_port, u32, u32, u32)>,
    get_rxtrig: Option<unsafe extern "C" fn(*mut uart_port) -> i32>,
    set_rxtrig: Option<unsafe extern "C" fn(*mut uart_port, u8) -> i32>,
    startup: Option<unsafe extern "C" fn(*mut uart_port) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut uart_port)>,
    throttle: Option<unsafe extern "C" fn(*mut uart_port)>,
    unthrottle: Option<unsafe extern "C" fn(*mut uart_port)>,
    break_ctl: Option<unsafe extern "C" fn(*mut uart_port, i32)>,
    handle_irq: Option<unsafe extern "C" fn(*mut uart_port) -> i32>,
    pm: Option<unsafe extern "C" fn(*mut uart_port, u32, u32)>,
    handle_break: Option<unsafe extern "C" fn(*mut uart_port)>,
    rs485_config: Option<unsafe extern "C" fn(*mut uart_port, *mut ktermios, *mut serial_rs485) -> i32>,
    iso7816_config: Option<unsafe extern "C" fn(*mut uart_port, *mut serial_iso7816) -> i32>,
    ctrl_id: u32,
    port_id: u32,
    irq: u32,
    irqflags: u64,
    uartclk: u32,
    fifosize: u32,
    x_char: u8,
    regshift: u8,
    quirks: u8,
    iotype: i32,
    read_status_mask: u32,
    ignore_status_mask: u32,
    state: *mut uart_state,
    icount: uart_icount,
    cons: *mut console,
    flags: u64,
    status: u32,
    hw_stopped: bool,
    cons_flow: bool,
    mctrl: u32,
    frame_time: u32,
    r#type: u32,
    ops: *const uart_ops,
    custom_divisor: u32,
    line: u32,
    minor: u32,
    mapbase: u64,
    mapsize: u64,
    dev: *mut device,
    port_dev: *mut serial_port_device,
    sysrq: u64,
    sysrq_ch: u8,
    has_sysrq: u8,
    sysrq_seq: u8,
    hub6: u8,
    suspended: u8,
    console_reinit: u8,
    name: *const u8,
    attr_group: *mut attribute_group,
    tty_groups: *mut *const attribute_group,
    rs485: serial_rs485,
    rs485_supported: serial_rs485,
    rs485_term_gpio: *mut gpio_desc,
    rs485_rx_during_tx_gpio: *mut gpio_desc,
    iso7816: serial_iso7816,
    private_data: *mut (),
}


fn __uart_port_using_nbcon(up: *mut uart_port) -> bool {
lockdep_assert_held_once!(unsafe { core::ptr::addr_of_mut!((*up).lock) });
if likely!(!uart_console!(up)) != 0 {
return false;
}
// @up->cons is only modified under the port lock. Therefore it is
// certain that it cannot disappear here.
//
// @up->cons->node is added/removed from the console list under the
// port lock. Therefore it is certain that the registration status
// cannot change here, thus @up->cons->flags can be read directly.
if hlist_unhashed_lockless((unsafe { core::ptr::addr_of_mut!((*(*up).cons).node) }) as *const hlist_node) != 0 || ((unsafe { (*(*up).cons).flags }) as i32) & 256 == 0 || (unsafe { (*(*up).cons).write_atomic }) == None {
return false;
}
true
}

fn __uart_port_nbcon_acquire(up: *mut uart_port) {
if !__uart_port_using_nbcon(up) {
return;
}
while !unsafe { nbcon_device_try_acquire((*up).cons) } {
cpu_relax();
}
}

fn __uart_port_nbcon_release(up: *mut uart_port) {
if !__uart_port_using_nbcon(up) {
return;
}
unsafe { nbcon_device_release((*up).cons) };
}

fn uart_port_lock(up: *mut uart_port) {
spin_lock(unsafe { core::ptr::addr_of_mut!((*up).lock) });
__uart_port_nbcon_acquire(up);
}

fn uart_port_lock_irq(up: *mut uart_port) {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*up).lock) });
__uart_port_nbcon_acquire(up);
}

fn uart_port_lock_irqsave(up: *mut uart_port, flags: *mut u64) {
spin_lock_irqsave__2!(unsafe { core::ptr::addr_of_mut!((*up).lock) }, *flags);
__uart_port_nbcon_acquire(up);
}

fn uart_port_unlock(up: *mut uart_port) {
__uart_port_nbcon_release(up);
spin_unlock(unsafe { core::ptr::addr_of_mut!((*up).lock) });
}

fn uart_port_unlock_irq(up: *mut uart_port) {
__uart_port_nbcon_release(up);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*up).lock) });
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_serial8250_clear_fifos: ; .asciz \"GPL\" ; .ascii \"SERIAL_8250\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_clear_and_reinit_fifos: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_rpm_get: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_rpm_put: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_em485_destroy: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_em485_supported: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_em485_config: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_em485_stop_tx: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_em485_start_tx: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_read_char: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_rx_chars: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_tx_chars: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_modem_status: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_handle_irq_locked: ; .asciz \"GPL\" ; .ascii \"SERIAL_8250\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_handle_irq: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_do_get_mctrl: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_do_set_mctrl: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_do_break_ctl: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_do_startup: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_do_shutdown: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_do_set_divisor: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_update_uartclk: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_do_set_termios: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_do_set_ldisc: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_do_pm: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_init_port: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_set_defaults: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_serial8250_fifo_wait_for_lsr_thre: ; .asciz \"GPL\" ; .ascii \"SERIAL_8250\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous", sym serial8250_clear_fifos, sym serial8250_clear_and_reinit_fifos, sym serial8250_rpm_get, sym serial8250_rpm_put, sym serial8250_em485_destroy, sym serial8250_em485_supported, sym serial8250_em485_config, sym serial8250_em485_stop_tx, sym serial8250_em485_start_tx, sym serial8250_read_char, sym serial8250_rx_chars, sym serial8250_tx_chars, sym serial8250_modem_status, sym serial8250_handle_irq_locked, sym serial8250_handle_irq, sym serial8250_do_get_mctrl, sym serial8250_do_set_mctrl, sym serial8250_do_break_ctl, sym serial8250_do_startup, sym serial8250_do_shutdown, sym serial8250_do_set_divisor, sym serial8250_update_uartclk, sym serial8250_do_set_termios, sym serial8250_do_set_ldisc, sym serial8250_do_pm, sym serial8250_init_port, sym serial8250_set_defaults, sym serial8250_fifo_wait_for_lsr_thre, options(att_syntax));

fn uart_port_unlock_irqrestore(up: *mut uart_port, flags: u64) {
__uart_port_nbcon_release(up);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*up).lock) }, flags);
}

#[inline(always)]
fn class_uart_port_lock_destructor(p: *mut *mut uart_port) {
let _T: *mut uart_port = unsafe { *p };
uart_port_unlock(_T);
}

#[inline(always)]
fn class_uart_port_lock_constructor(_T: *mut uart_port) -> *mut uart_port {
let mut __v0: *mut uart_port;
uart_port_lock(_T);
__v0 = _T;
let t: *mut uart_port = __v0;
t
}

#[inline(always)]
fn class_uart_port_lock_irq_destructor(p: *mut *mut uart_port) {
let _T: *mut uart_port = unsafe { *p };
uart_port_unlock_irq(_T);
}

#[inline(always)]
fn class_uart_port_lock_irq_constructor(_T: *mut uart_port) -> *mut uart_port {
let mut __v0: *mut uart_port;
uart_port_lock_irq(_T);
__v0 = _T;
let t: *mut uart_port = __v0;
t
}

static class_uart_port_lock_irq_is_conditional: bool = 0i32 != 0i32;

#[inline(always)]
fn class_uart_port_lock_irq_lock_ptr(_T: &*mut uart_port) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *(_T as *const *mut uart_port) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

static class_uart_port_lock_irqsave_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_serial_core_h_803 {
    lock: *mut uart_port,
    flags: u64,
}


#[inline(always)]
fn class_uart_port_lock_irqsave_destructor(_T: *mut __SlateRecord_serial_core_h_803) {
uart_port_unlock_irqrestore(unsafe { (*_T).lock }, unsafe { (*_T).flags });
}

#[inline(always)]
fn class_uart_port_lock_irqsave_lock_ptr(_T: &mut __SlateRecord_serial_core_h_803) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_serial_core_h_803)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[inline(always)]
fn class_uart_port_lock_irqsave_constructor(l: *mut uart_port) -> __SlateRecord_serial_core_h_803 {
let mut _t: __SlateRecord_serial_core_h_803 = {
let mut __t0: __SlateRecord_serial_core_h_803 = unsafe { core::mem::zeroed() };
__t0.lock = l;
__t0
};
let _T: *mut __SlateRecord_serial_core_h_803 = core::ptr::addr_of_mut!(_t);
uart_port_lock_irqsave(unsafe { (*_T).lock }, unsafe { core::ptr::addr_of_mut!((*_T).flags) });
return _t;
}

fn serial_port_in(up: *mut uart_port, offset: i32) -> i32 {
(unsafe { unsafe { (*up).serial_in }.unwrap()(up, offset as u32) }) as i32
}

fn serial_port_out(up: *mut uart_port, offset: i32, value: i32) {
unsafe { unsafe { (*up).serial_out }.unwrap()(up, offset as u32, value as u32) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uart_state {
    port: tty_port,
    pm_state: u32,
    refcount: __SlateRecord_types_h_190,
    remove_wait: wait_queue_head,
    uart_port: *mut uart_port,
}


// number of characters left in xmit buffer before we ask for more
fn uart_fifo_get(up: *mut uart_port, __val: *mut u8) -> u32 {
let tport: *mut tty_port = unsafe { core::ptr::addr_of_mut!((*(*up).state).port) };
let mut chars: u32 = 0;
let mut __v0: u32;
let __tmp: *mut __SlateRecord_tty_port_h_118 = unsafe { core::ptr::addr_of_mut!((*tport).xmit_fifo) };
let mut __ret: u32 = 0;
let __recsize: u64 = core::mem::size_of::<[u8; 0]>() as u64;
let __kfifo: *mut __kfifo = unsafe { core::ptr::addr_of_mut!((*__tmp).__slate_anon_0.kfifo) };
if __recsize != 0 {
__ret = unsafe { __kfifo_out_r(__kfifo, __val as *mut (), core::mem::size_of::<u8>() as u32, __recsize) };
} else {
__ret = !kfifo_is_empty!(__tmp) as u32;
if __ret != 0 {
unsafe {
*__val = *if __is_kfifo_ptr__9!() { (*__kfifo).data as *mut u8 } else { (*__tmp).buf.as_mut_ptr() as *mut u8 }.wrapping_offset(((*__kfifo).out & (*__tmp).__slate_anon_0.kfifo.mask) as isize);
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
let __v6: u32 = unsafe { (*__kfifo).out };
unsafe {
(*__kfifo).out = __v6.wrapping_add(1);
}
}
}
__v0 = __ret;
chars = __kfifo_uint_must_check_helper(__v0);
let __v9: u32 = unsafe { (*up).icount.tx };
unsafe {
(*up).icount.tx = __v9.wrapping_add(chars);
}
chars
}

fn uart_tx_stopped(port: *mut uart_port) -> i32 {
let tty: *mut tty_struct = unsafe { (*(*port).state).port.tty };
if !tty.is_null() && unsafe { (*tty).flow.stopped } || unsafe { (*port).hw_stopped } {
return 1;
}
0
}

fn uart_cts_enabled(uport: *mut uart_port) -> bool {
(unsafe { (*uport).status }) & 1u32 != 0
}

fn uart_set_cons_flow_enabled(uport: *mut uart_port, enabled: bool) {
unsafe {
(*uport).cons_flow = enabled;
}
}

fn uart_cons_flow_enabled(uport: *const uart_port) -> bool {
unsafe { (*uport).cons_flow }
}

fn uart_console_hwflow_active(uport: *mut uart_port) -> bool {
let __v0: bool = uart_console!(uport) && ((unsafe { (*uport).rs485.flags }) as u64) & _BITUL!(0i32) == 0 && uart_cons_flow_enabled(uport as *const uart_port);
__v0 && uart_cts_enabled(uport)
}

fn uart_prepare_sysrq_char(port: *mut uart_port, ch: u8) -> i32 {
if (unsafe { (*port).sysrq }) == 0 {
return 0;
}
let __v0: bool = ch != 0 && time_before!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, unsafe { (*port).sysrq });
if __v0 {
if (unsafe { sysrq_mask() }) != 0 {
unsafe {
(*port).sysrq_ch = ch;
}
unsafe {
(*port).sysrq = 0;
}
return 1;
}
if unsafe { uart_try_toggle_sysrq(port, ch) } {
return 1;
}
}
unsafe {
(*port).sysrq = 0;
}
0
}

fn uart_unlock_and_check_sysrq_irqrestore(port: *mut uart_port, flags: u64) {
let mut sysrq_ch: u8 = 0;
if (unsafe { (*port).has_sysrq }) == 0 {
uart_port_unlock_irqrestore(port, flags);
return;
}
sysrq_ch = unsafe { (*port).sysrq_ch };
unsafe {
(*port).sysrq_ch = 0;
}
uart_port_unlock_irqrestore(port, flags);
if sysrq_ch != 0 {
unsafe { handle_sysrq(sysrq_ch) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_serial_core_h_1308 {
    lock: *mut uart_port,
    flags: u64,
}


#[inline(always)]
fn class_uart_port_lock_check_sysrq_irqsave_destructor(_T: *mut __SlateRecord_serial_core_h_1308) {
uart_unlock_and_check_sysrq_irqrestore(unsafe { (*_T).lock }, unsafe { (*_T).flags });
}

#[inline(always)]
fn class_uart_port_lock_check_sysrq_irqsave_constructor(l: *mut uart_port) -> __SlateRecord_serial_core_h_1308 {
let mut _t: __SlateRecord_serial_core_h_1308 = {
let mut __t0: __SlateRecord_serial_core_h_1308 = unsafe { core::mem::zeroed() };
__t0.lock = l;
__t0
};
let _T: *mut __SlateRecord_serial_core_h_1308 = core::ptr::addr_of_mut!(_t);
uart_port_lock_irqsave(unsafe { (*_T).lock }, unsafe { core::ptr::addr_of_mut!((*_T).flags) });
return _t;
}

fn uart_handle_break(port: *mut uart_port) -> i32 {
let state: *mut uart_state = unsafe { (*port).state };
if (unsafe { (*port).handle_break }) != None {
unsafe { unsafe { (*port).handle_break }.unwrap()(port) };
}
if (unsafe { (*port).has_sysrq }) != 0 && uart_console!(port) {
if (unsafe { (*port).sysrq }) == 0 {
unsafe {
(*port).sysrq = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(5000u64);
}
return 1;
}
unsafe {
(*port).sysrq = 0;
}
}
if (unsafe { (*port).flags }) & 4u64 != 0 {
unsafe { do_SAK((*state).port.tty) };
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uart_8250_ops {
    setup_irq: Option<unsafe extern "C" fn(*mut uart_8250_port) -> i32>,
    release_irq: Option<unsafe extern "C" fn(*mut uart_8250_port)>,
    setup_timer: Option<unsafe extern "C" fn(*mut uart_8250_port)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uart_8250_em485 {
    start_tx_timer: hrtimer,
    stop_tx_timer: hrtimer,
    active_timer: *mut hrtimer,
    port: *mut uart_8250_port,
    __slate_bits_0: __slate_bits::__SlateBits797U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uart_8250_port {
    port: uart_port,
    timer: timer_list,
    list: list_head,
    capabilities: u32,
    bugs: u16,
    tx_loadsz: u32,
    acr: u8,
    fcr: u8,
    ier: u8,
    lcr: u8,
    mcr: u8,
    cur_iotype: u8,
    rpm_tx_active: u32,
    canary: u8,
    probe: u8,
    gpios: *mut mctrl_gpios,
    lsr_saved_flags: u16,
    lsr_save_mask: u16,
    console_line_ended: bool,
    console_msr_work_allow: bool,
    msr_saved_flags: u8,
    console_msr_work: irq_work,
    dma: *mut uart_8250_dma,
    ops: *const uart_8250_ops,
    dl_read: Option<unsafe extern "C" fn(*mut uart_8250_port) -> u32>,
    dl_write: Option<unsafe extern "C" fn(*mut uart_8250_port, u32)>,
    em485: *mut uart_8250_em485,
    rs485_start_tx: Option<unsafe extern "C" fn(*mut uart_8250_port, bool)>,
    rs485_stop_tx: Option<unsafe extern "C" fn(*mut uart_8250_port, bool)>,
    overrun_backoff: delayed_work,
    overrun_backoff_time_ms: u32,
}


fn up_to_u8250p(up: *mut uart_port) -> *mut uart_8250_port {
container_of__2!(up, uart_8250_port)
}

fn touch_softlockup_watchdog() {
}

fn arch_touch_nmi_watchdog() {
}

fn touch_nmi_watchdog() {
// Pass on to the hardlockup detector selected via CONFIG_. Note that
// the hardlockup detector may not be arch-specific nor using NMIs
// and the arch_touch_nmi_watchdog() function will likely be renamed
// in the future.
arch_touch_nmi_watchdog();
touch_softlockup_watchdog();
}

// Request is asynchronous
// Increment/decrement the
// usage_count
// Use autosuspend_delay
// Succeed if runtime PM is disabled
fn pm_runtime_mark_last_busy(dev: *mut device) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_543() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!((*dev).power.last_busy, unsafe { ktime_get_mono_fast_ns() });
if true {
break;
}
}
}

fn pm_runtime_get_sync(dev: *mut device) -> i32 {
unsafe { __pm_runtime_resume(dev, 4i32) }
}

fn __pm_runtime_put_autosuspend(dev: *mut device) -> i32 {
unsafe { __pm_runtime_suspend(dev, 13i32) }
}

fn pm_runtime_put_autosuspend(dev: *mut device) -> i32 {
pm_runtime_mark_last_busy(dev);
__pm_runtime_put_autosuspend(dev)
}

fn pm_runtime_put_sync(dev: *mut device) -> i32 {
unsafe { __pm_runtime_idle(dev, 4i32) }
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
struct scatterlist {
    page_link: u64,
    offset: u32,
    length: u32,
    dma_address: u64,
    dma_length: u32,
    dma_flags: u32,
}


// These macros should be used after a dma_map_sg call has been done
// to get bus addresses of each of the SG entries and their lengths.
// You should only work with the number of sg entries dma_map_sg
// returns, or alternatively stop on the first sg_dma_len(sg) which
// is 0.
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
#[repr(C)]
#[derive(Clone, Copy)]
struct data_chunk {
    size: u64,
    icg: u64,
    dst_icg: u64,
    src_icg: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_interleaved_template {
    src_start: u64,
    dst_start: u64,
    dir: u32,
    src_inc: bool,
    dst_inc: bool,
    src_sgl: bool,
    dst_sgl: bool,
    numf: u64,
    frame_size: u64,
    sgl: [data_chunk; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_vec {
    addr: u64,
    len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dmaengine_h_240 {
    bits: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_chan_percpu {
    memcpy_count: u64,
    bytes_transferred: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_router {
    dev: *mut device,
    route_free: Option<unsafe extern "C" fn(*mut device, *mut ())>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_chan {
    device: *mut dma_device,
    slave: *mut device,
    cookie: i32,
    completed_cookie: i32,
    lock: spinlock,
    chan_id: i32,
    dev: *mut dma_chan_dev,
    name: *const u8,
    dbg_client_name: *mut u8,
    device_node: list_head,
    local: *mut dma_chan_percpu,
    client_count: i32,
    table_count: i32,
    router: *mut dma_router,
    route_data: *mut (),
    private: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_chan_dev {
    chan: *mut dma_chan,
    device: device,
    dev_id: i32,
    chan_dma_dev: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_slave_config {
    direction: u32,
    src_addr: u64,
    dst_addr: u64,
    src_addr_width: u32,
    dst_addr_width: u32,
    src_maxburst: u32,
    dst_maxburst: u32,
    src_port_window_size: u32,
    dst_port_window_size: u32,
    device_fc: bool,
    peripheral_config: *mut (),
    peripheral_size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_slave_caps {
    src_addr_widths: u32,
    dst_addr_widths: u32,
    directions: u32,
    min_burst: u32,
    max_burst: u32,
    max_sg_burst: u32,
    cmd_pause: bool,
    cmd_resume: bool,
    cmd_terminate: bool,
    residue_granularity: u32,
    descriptor_reuse: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dmaengine_result {
    result: u32,
    residue: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dmaengine_unmap_data {
    map_cnt: u8,
    to_cnt: u8,
    from_cnt: u8,
    bidi_cnt: u8,
    dev: *mut device,
    kref: kref,
    len: u64,
    addr: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_descriptor_metadata_ops {
    attach: Option<unsafe extern "C" fn(*mut dma_async_tx_descriptor, *mut (), u64) -> i32>,
    get_ptr: Option<unsafe extern "C" fn(*mut dma_async_tx_descriptor, *mut u64, *mut u64) -> *mut ()>,
    set_len: Option<unsafe extern "C" fn(*mut dma_async_tx_descriptor, u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_async_tx_descriptor {
    cookie: i32,
    flags: u32,
    phys: u64,
    chan: *mut dma_chan,
    tx_submit: Option<unsafe extern "C" fn(*mut dma_async_tx_descriptor) -> i32>,
    desc_free: Option<unsafe extern "C" fn(*mut dma_async_tx_descriptor) -> i32>,
    callback: Option<unsafe extern "C" fn(*mut ())>,
    callback_result: Option<unsafe extern "C" fn(*mut (), *const dmaengine_result)>,
    callback_param: *mut (),
    unmap: *mut dmaengine_unmap_data,
    desc_metadata_mode: u32,
    metadata_ops: *const dma_descriptor_metadata_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_tx_state {
    last: i32,
    used: i32,
    residue: u32,
    in_flight_bytes: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_slave_map {
    devname: *const u8,
    slave: *const u8,
    param: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_filter {
    r#fn: Option<unsafe extern "C" fn(*mut dma_chan, *mut ()) -> bool>,
    mapcnt: i32,
    map: *const dma_slave_map,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_device {
    r#ref: kref,
    chancnt: u32,
    privatecnt: u32,
    channels: list_head,
    global_node: list_head,
    filter: dma_filter,
    cap_mask: __SlateRecord_dmaengine_h_240,
    desc_metadata_modes: u32,
    max_xor: u16,
    max_pq: u16,
    copy_align: u32,
    xor_align: u32,
    pq_align: u32,
    fill_align: u32,
    dev_id: i32,
    dev: *mut device,
    owner: *mut module,
    chan_ida: ida,
    src_addr_widths: u32,
    dst_addr_widths: u32,
    directions: u32,
    min_burst: u32,
    max_burst: u32,
    max_sg_burst: u32,
    descriptor_reuse: bool,
    residue_granularity: u32,
    device_alloc_chan_resources: Option<unsafe extern "C" fn(*mut dma_chan) -> i32>,
    device_router_config: Option<unsafe extern "C" fn(*mut dma_chan) -> i32>,
    device_free_chan_resources: Option<unsafe extern "C" fn(*mut dma_chan)>,
    device_prep_dma_memcpy: Option<unsafe extern "C" fn(*mut dma_chan, u64, u64, u64, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_xor: Option<unsafe extern "C" fn(*mut dma_chan, u64, *mut u64, u32, u64, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_xor_val: Option<unsafe extern "C" fn(*mut dma_chan, *mut u64, u32, u64, *mut u32, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_pq: Option<unsafe extern "C" fn(*mut dma_chan, *mut u64, *mut u64, u32, *const u8, u64, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_pq_val: Option<unsafe extern "C" fn(*mut dma_chan, *mut u64, *mut u64, u32, *const u8, u64, *mut u32, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_memset: Option<unsafe extern "C" fn(*mut dma_chan, u64, i32, u64, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_memset_sg: Option<unsafe extern "C" fn(*mut dma_chan, *mut scatterlist, u32, i32, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_interrupt: Option<unsafe extern "C" fn(*mut dma_chan, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_peripheral_dma_vec: Option<unsafe extern "C" fn(*mut dma_chan, *const dma_vec, u64, u32, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_slave_sg: Option<unsafe extern "C" fn(*mut dma_chan, *mut scatterlist, u32, u32, u64, *mut ()) -> *mut dma_async_tx_descriptor>,
    device_prep_config_sg: Option<unsafe extern "C" fn(*mut dma_chan, *mut scatterlist, u32, u32, u64, *mut dma_slave_config) -> *mut dma_async_tx_descriptor>,
    device_prep_dma_cyclic: Option<unsafe extern "C" fn(*mut dma_chan, u64, u64, u64, u32, u64) -> *mut dma_async_tx_descriptor>,
    device_prep_interleaved_dma: Option<unsafe extern "C" fn(*mut dma_chan, *mut dma_interleaved_template, u64) -> *mut dma_async_tx_descriptor>,
    device_caps: Option<unsafe extern "C" fn(*mut dma_chan, *mut dma_slave_caps)>,
    device_config: Option<unsafe extern "C" fn(*mut dma_chan, *mut dma_slave_config) -> i32>,
    device_pause: Option<unsafe extern "C" fn(*mut dma_chan) -> i32>,
    device_resume: Option<unsafe extern "C" fn(*mut dma_chan) -> i32>,
    device_terminate_all: Option<unsafe extern "C" fn(*mut dma_chan) -> i32>,
    device_synchronize: Option<unsafe extern "C" fn(*mut dma_chan)>,
    device_tx_status: Option<unsafe extern "C" fn(*mut dma_chan, i32, *mut dma_tx_state) -> u32>,
    device_issue_pending: Option<unsafe extern "C" fn(*mut dma_chan)>,
    device_release: Option<unsafe extern "C" fn(*mut dma_device)>,
    dbg_summary_show: Option<unsafe extern "C" fn(*mut seq_file, *mut dma_device)>,
    dbg_dev_root: *mut dentry,
}


fn mctrl_gpio_set(gpios: *mut mctrl_gpios, mctrl: u32) {
}

fn mctrl_gpio_get(gpios: *mut mctrl_gpios, mctrl: &u32) -> u32 {
unsafe { *(mctrl as *const u32) }
}

fn mctrl_gpio_get_outputs(gpios: *mut mctrl_gpios, mctrl: &u32) -> u32 {
unsafe { *(mctrl as *const u32) }
}

fn mctrl_gpio_enable_ms(gpios: *mut mctrl_gpios) {
}

fn mctrl_gpio_disable_ms_no_sync(gpios: *mut mctrl_gpios) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uart_8250_dma {
    tx_dma: Option<unsafe extern "C" fn(*mut uart_8250_port) -> i32>,
    rx_dma: Option<unsafe extern "C" fn(*mut uart_8250_port) -> i32>,
    prepare_tx_dma: Option<unsafe extern "C" fn(*mut uart_8250_port)>,
    prepare_rx_dma: Option<unsafe extern "C" fn(*mut uart_8250_port)>,
    r#fn: Option<unsafe extern "C" fn(*mut dma_chan, *mut ()) -> bool>,
    rx_param: *mut (),
    tx_param: *mut (),
    rxconf: dma_slave_config,
    txconf: dma_slave_config,
    rxchan: *mut dma_chan,
    txchan: *mut dma_chan,
    rx_dma_addr: u64,
    tx_dma_addr: u64,
    rx_addr: u64,
    tx_addr: u64,
    rx_cookie: i32,
    tx_cookie: i32,
    rx_buf: *mut (),
    rx_size: u64,
    tx_size: u64,
    tx_running: u8,
    tx_err: u8,
    rx_running: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serial8250_config {
    name: *const u8,
    fifo_size: u16,
    tx_loadsz: u16,
    fcr: u8,
    rxtrig_bytes: [u8; 4],
    flags: u32,
}


// UART has FIFO
// UART has EFR
// UART has IER sleep
// MCR-based hw flow control
// UART needs IER bit 6 set (Xscale)
// UART needs IER bit 4 set (Xscale, Tegra)
// UART has a "hidden" FIFO
// Runtime PM is active while idle
// Mini UART on BCM283X family lacks:
// STOP PARITY EPAR SPAR WLEN5 WLEN6
// UART without interrupt on TEMT available
// UART has buggy quot LSB
// UART has buggy TX IIR status
// UART has buggy MSR status bits (Au1x00)
// UART has buggy THRE reassertion
// UART Tx fails to set remote DR
fn serial_in(up: *mut uart_8250_port, offset: i32) -> i32 {
(unsafe { unsafe { (*up).port.serial_in }.unwrap()(core::ptr::addr_of_mut!((*up).port), offset as u32) }) as i32
}

fn serial_out(up: *mut uart_8250_port, offset: i32, value: i32) {
unsafe { unsafe { (*up).port.serial_out }.unwrap()(core::ptr::addr_of_mut!((*up).port), offset as u32, value as u32) };
}

fn serial_lsr_in(up: *mut uart_8250_port) -> u16 {
let mut lsr: u16 = unsafe { (*up).lsr_saved_flags };
let __v0: i32 = serial_in(up, 5i32);
lsr = ((lsr as i32) | __v0) as u16;
unsafe {
(*up).lsr_saved_flags = ((lsr as i32) & ((*up).lsr_save_mask as i32)) as u16;
}
lsr
}

fn serial_icr_write(up: *mut uart_8250_port, offset: i32, value: i32) {
serial_out(up, 7i32, offset);
serial_out(up, 5i32, value);
}

fn serial_icr_read(up: *mut uart_8250_port, offset: i32) -> u32 {
let mut value: u32 = 0;
serial_icr_write(up, 0i32, ((unsafe { (*up).acr }) as i32) | 64);
serial_out(up, 7i32, offset);
value = serial_in(up, 5i32) as u32;
serial_icr_write(up, 0i32, (unsafe { (*up).acr }) as i32);
value
}

#[inline(always)]
fn class_serial8250_rpm_destructor(p: *mut *mut uart_8250_port) {
let _T: *mut uart_8250_port = unsafe { *p };
serial8250_rpm_put(_T);
}

#[inline(always)]
fn class_serial8250_rpm_constructor(_T: *mut uart_8250_port) -> *mut uart_8250_port {
let mut __v0: *mut uart_8250_port;
serial8250_rpm_get(_T);
__v0 = _T;
let t: *mut uart_8250_port = __v0;
t
}

static class_serial8250_rpm_is_conditional: bool = 0i32 != 0i32;

#[inline(always)]
fn class_serial8250_rpm_lock_ptr(_T: &*mut uart_8250_port) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *(_T as *const *mut uart_8250_port) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

fn serial_dl_read(up: *mut uart_8250_port) -> u32 {
unsafe { unsafe { (*up).dl_read }.unwrap()(up) }
}

fn serial_dl_write(up: *mut uart_8250_port, value: u32) {
unsafe { unsafe { (*up).dl_write }.unwrap()(up, value) };
}

fn serial8250_set_THRI(up: *mut uart_8250_port) -> bool {
// Port locked to synchronize UART_IER access against the console.
lockdep_assert_held_once!(unsafe { core::ptr::addr_of_mut!((*up).port.lock) });
if ((unsafe { (*up).ier }) as i32) & 2 != 0 {
return false;
}
unsafe {
(*up).ier = (((*up).ier as i32) | 2) as u8;
}
serial_out(up, 1i32, (unsafe { (*up).ier }) as i32);
true
}

fn serial8250_clear_THRI(up: *mut uart_8250_port) -> bool {
// Port locked to synchronize UART_IER access against the console.
lockdep_assert_held_once!(unsafe { core::ptr::addr_of_mut!((*up).port.lock) });
if ((unsafe { (*up).ier }) as i32) & 2 == 0 {
return false;
}
let __v3: u8 = (((unsafe { (*up).ier }) as i32) & !2) as u8;
unsafe {
(*up).ier = __v3;
}
serial_out(up, 1i32, (unsafe { (*up).ier }) as i32);
true
}

fn serial8250_TIOCM_to_MCR(tiocm: i32) -> i32 {
let mut mcr: i32 = 0;
if tiocm & 4 != 0 {
mcr |= 2;
}
if tiocm & 2 != 0 {
mcr |= 1;
}
if tiocm & 8192 != 0 {
mcr |= 4;
}
if tiocm & 16384 != 0 {
mcr |= 8;
}
if tiocm & 32768 != 0 {
mcr |= 16;
}
mcr
}

fn serial8250_MCR_to_TIOCM(mcr: i32) -> i32 {
let mut tiocm: i32 = 0;
if mcr & 2 != 0 {
tiocm |= 4;
}
if mcr & 1 != 0 {
tiocm |= 2;
}
if mcr & 4 != 0 {
tiocm |= 8192;
}
if mcr & 8 != 0 {
tiocm |= 16384;
}
if mcr & 16 != 0 {
tiocm |= 32768;
}
tiocm
}

fn serial8250_MSR_to_TIOCM(msr: i32) -> i32 {
let mut tiocm: i32 = 0;
if msr & 128 != 0 {
tiocm |= 64;
}
if msr & 64 != 0 {
tiocm |= 128;
}
if msr & 32 != 0 {
tiocm |= 256;
}
if msr & 16 != 0 {
tiocm |= 32;
}
tiocm
}

fn serial8250_out_MCR(up: *mut uart_8250_port, value: i32) {
serial_out(up, 4i32, value);
if !unsafe { (*up).gpios }.is_null() {
mctrl_gpio_set(unsafe { (*up).gpios }, serial8250_MCR_to_TIOCM(value) as u32);
}
}

fn serial8250_in_MCR(up: *mut uart_8250_port) -> i32 {
let mut mctrl: i32 = serial_in(up, 4i32);
if !unsafe { (*up).gpios }.is_null() {
let mut mctrl_gpio: u32 = 0;
mctrl_gpio = mctrl_gpio_get_outputs(unsafe { (*up).gpios }, unsafe { &(*core::ptr::addr_of_mut!(mctrl_gpio)) });
let __v0: i32 = serial8250_TIOCM_to_MCR(mctrl_gpio as i32);
mctrl |= __v0;
}
mctrl
}

fn fintek_8250_probe(uart: *mut uart_8250_port) -> i32 {
0
}

fn is_omap1_8250(pt: *mut uart_8250_port) -> i32 {
0
}

fn is_omap1510_8250(pt: *mut uart_8250_port) -> i32 {
0
}

fn serial8250_tx_dma_running(p: *mut uart_8250_port) -> bool {
let dma: *mut uart_8250_dma = unsafe { (*p).dma };
!dma.is_null() && (unsafe { (*dma).tx_running }) != 0
}

fn ns16550a_goto_highspeed(up: *mut uart_8250_port) -> i32 {
// EXCR2
let mut status: u8 = serial_in(up, 4i32) as u8;
if PRESL!(status) == 16 {
// already in high speed mode
return 0;
} else {
let __v1: u8 = ((status as i32) & !176) as u8;
status = __v1; // Disable LOCK, mask out PRESL[01]
status = ((status as i32) | 16) as u8; // 1.625 divisor for baud_base --> 921600
serial_out(up, 4i32, status as i32);
}
1
}

// Here we define the default xmit fifo size used for each type of UART.
/// UART_CAP_EFR breaks billionon CF bluetooth card.
/// tx_loadsz is set to 63-bytes instead of 64-bytes to implement
/// workaround of errata A-008006 which states that tx_loadsz should
/// be configured less than Maximum supported fifo bytes.
static uart_config: __SlateAlign16<[serial8250_config; 122]> = __SlateAlign16({
let mut __t0: [serial8250_config; 122] = unsafe { core::mem::zeroed() };
__t0[0] = {
let mut __t1: serial8250_config = unsafe { core::mem::zeroed() };
__t1.name = (b"unknown\0".as_ptr() as *mut u8) as *const u8;
__t1.fifo_size = 1u16;
__t1.tx_loadsz = 1u16;
__t1
};
__t0[1] = {
let mut __t2: serial8250_config = unsafe { core::mem::zeroed() };
__t2.name = (b"8250\0".as_ptr() as *mut u8) as *const u8;
__t2.fifo_size = 1u16;
__t2.tx_loadsz = 1u16;
__t2
};
__t0[2] = {
let mut __t3: serial8250_config = unsafe { core::mem::zeroed() };
__t3.name = (b"16450\0".as_ptr() as *mut u8) as *const u8;
__t3.fifo_size = 1u16;
__t3.tx_loadsz = 1u16;
__t3
};
__t0[3] = {
let mut __t4: serial8250_config = unsafe { core::mem::zeroed() };
__t4.name = (b"16550\0".as_ptr() as *mut u8) as *const u8;
__t4.fifo_size = 1u16;
__t4.tx_loadsz = 1u16;
__t4
};
__t0[4] = {
let mut __t5: serial8250_config = unsafe { core::mem::zeroed() };
__t5.name = (b"16550A\0".as_ptr() as *mut u8) as *const u8;
__t5.fifo_size = 16u16;
__t5.tx_loadsz = 16u16;
__t5.fcr = 129u8;
__t5.rxtrig_bytes = [1u8, 4u8, 8u8, 14u8];
__t5.flags = 256u32;
__t5
};
__t0[5] = {
let mut __t6: serial8250_config = unsafe { core::mem::zeroed() };
__t6.name = (b"Cirrus\0".as_ptr() as *mut u8) as *const u8;
__t6.fifo_size = 1u16;
__t6.tx_loadsz = 1u16;
__t6
};
__t0[6] = {
let mut __t7: serial8250_config = unsafe { core::mem::zeroed() };
__t7.name = (b"ST16650\0".as_ptr() as *mut u8) as *const u8;
__t7.fifo_size = 1u16;
__t7.tx_loadsz = 1u16;
__t7.flags = 1792u32;
__t7
};
__t0[7] = {
let mut __t8: serial8250_config = unsafe { core::mem::zeroed() };
__t8.name = (b"ST16650V2\0".as_ptr() as *mut u8) as *const u8;
__t8.fifo_size = 32u16;
__t8.tx_loadsz = 16u16;
__t8.fcr = 65u8;
__t8.rxtrig_bytes = [8u8, 16u8, 24u8, 28u8];
__t8.flags = 1792u32;
__t8
};
__t0[8] = {
let mut __t9: serial8250_config = unsafe { core::mem::zeroed() };
__t9.name = (b"TI16750\0".as_ptr() as *mut u8) as *const u8;
__t9.fifo_size = 64u16;
__t9.tx_loadsz = 64u16;
__t9.fcr = 161u8;
__t9.rxtrig_bytes = [1u8, 16u8, 32u8, 56u8];
__t9.flags = 3328u32;
__t9
};
__t0[9] = {
let mut __t10: serial8250_config = unsafe { core::mem::zeroed() };
__t10.name = (b"Startech\0".as_ptr() as *mut u8) as *const u8;
__t10.fifo_size = 1u16;
__t10.tx_loadsz = 1u16;
__t10
};
__t0[10] = {
let mut __t11: serial8250_config = unsafe { core::mem::zeroed() };
__t11.name = (b"16C950/954\0".as_ptr() as *mut u8) as *const u8;
__t11.fifo_size = 128u16;
__t11.tx_loadsz = 128u16;
__t11.fcr = 65u8;
__t11.rxtrig_bytes = [16u8, 32u8, 112u8, 120u8];
__t11.flags = 1280u32;
__t11
};
__t0[11] = {
let mut __t12: serial8250_config = unsafe { core::mem::zeroed() };
__t12.name = (b"ST16654\0".as_ptr() as *mut u8) as *const u8;
__t12.fifo_size = 64u16;
__t12.tx_loadsz = 32u16;
__t12.fcr = 97u8;
__t12.rxtrig_bytes = [8u8, 16u8, 56u8, 60u8];
__t12.flags = 1792u32;
__t12
};
__t0[12] = {
let mut __t13: serial8250_config = unsafe { core::mem::zeroed() };
__t13.name = (b"XR16850\0".as_ptr() as *mut u8) as *const u8;
__t13.fifo_size = 128u16;
__t13.tx_loadsz = 128u16;
__t13.fcr = 129u8;
__t13.flags = 1792u32;
__t13
};
__t0[13] = {
let mut __t14: serial8250_config = unsafe { core::mem::zeroed() };
__t14.name = (b"RSA\0".as_ptr() as *mut u8) as *const u8;
__t14.fifo_size = 2048u16;
__t14.tx_loadsz = 2048u16;
__t14.fcr = 193u8;
__t14.flags = 256u32;
__t14
};
__t0[14] = {
let mut __t15: serial8250_config = unsafe { core::mem::zeroed() };
__t15.name = (b"NS16550A\0".as_ptr() as *mut u8) as *const u8;
__t15.fifo_size = 16u16;
__t15.tx_loadsz = 16u16;
__t15.fcr = 129u8;
__t15.flags = 264u32;
__t15
};
__t0[15] = {
let mut __t16: serial8250_config = unsafe { core::mem::zeroed() };
__t16.name = (b"XScale\0".as_ptr() as *mut u8) as *const u8;
__t16.fifo_size = 32u16;
__t16.tx_loadsz = 32u16;
__t16.fcr = 129u8;
__t16.flags = 12544u32;
__t16
};
__t0[17] = {
let mut __t17: serial8250_config = unsafe { core::mem::zeroed() };
__t17.name = (b"OCTEON\0".as_ptr() as *mut u8) as *const u8;
__t17.fifo_size = 64u16;
__t17.tx_loadsz = 64u16;
__t17.fcr = 129u8;
__t17.flags = 256u32;
__t17
};
__t0[19] = {
let mut __t18: serial8250_config = unsafe { core::mem::zeroed() };
__t18.name = (b"U6_16550A\0".as_ptr() as *mut u8) as *const u8;
__t18.fifo_size = 64u16;
__t18.tx_loadsz = 64u16;
__t18.fcr = 129u8;
__t18.flags = 2304u32;
__t18
};
__t0[20] = {
let mut __t19: serial8250_config = unsafe { core::mem::zeroed() };
__t19.name = (b"Tegra\0".as_ptr() as *mut u8) as *const u8;
__t19.fifo_size = 32u16;
__t19.tx_loadsz = 8u16;
__t19.fcr = 81u8;
__t19.rxtrig_bytes = [1u8, 4u8, 8u8, 14u8];
__t19.flags = 8448u32;
__t19
};
__t0[21] = {
let mut __t20: serial8250_config = unsafe { core::mem::zeroed() };
__t20.name = (b"XR17D15X\0".as_ptr() as *mut u8) as *const u8;
__t20.fifo_size = 64u16;
__t20.tx_loadsz = 64u16;
__t20.fcr = 129u8;
__t20.flags = 3840u32;
__t20
};
__t0[22] = {
let mut __t21: serial8250_config = unsafe { core::mem::zeroed() };
__t21.name = (b"LPC3220\0".as_ptr() as *mut u8) as *const u8;
__t21.fifo_size = 64u16;
__t21.tx_loadsz = 32u16;
__t21.fcr = 9u8;
__t21.flags = 256u32;
__t21
};
__t0[23] = {
let mut __t22: serial8250_config = unsafe { core::mem::zeroed() };
__t22.name = (b"CIR port\0".as_ptr() as *mut u8) as *const u8;
__t22
};
__t0[24] = {
let mut __t23: serial8250_config = unsafe { core::mem::zeroed() };
__t23.name = (b"XR17V35X\0".as_ptr() as *mut u8) as *const u8;
__t23.fifo_size = 256u16;
__t23.tx_loadsz = 256u16;
__t23.fcr = 241u8;
__t23.flags = 3840u32;
__t23
};
__t0[25] = {
let mut __t24: serial8250_config = unsafe { core::mem::zeroed() };
__t24.name = (b"TruManage\0".as_ptr() as *mut u8) as *const u8;
__t24.fifo_size = 1u16;
__t24.tx_loadsz = 1024u16;
__t24.flags = 16384u32;
__t24
};
__t0[26] = {
let mut __t25: serial8250_config = unsafe { core::mem::zeroed() };
__t25.name = (b"Altera 16550 FIFO32\0".as_ptr() as *mut u8) as *const u8;
__t25.fifo_size = 32u16;
__t25.tx_loadsz = 32u16;
__t25.fcr = 129u8;
__t25.rxtrig_bytes = [1u8, 8u8, 16u8, 30u8];
__t25.flags = 2304u32;
__t25
};
__t0[27] = {
let mut __t26: serial8250_config = unsafe { core::mem::zeroed() };
__t26.name = (b"Altera 16550 FIFO64\0".as_ptr() as *mut u8) as *const u8;
__t26.fifo_size = 64u16;
__t26.tx_loadsz = 64u16;
__t26.fcr = 129u8;
__t26.rxtrig_bytes = [1u8, 16u8, 32u8, 62u8];
__t26.flags = 2304u32;
__t26
};
__t0[28] = {
let mut __t27: serial8250_config = unsafe { core::mem::zeroed() };
__t27.name = (b"Altera 16550 FIFO128\0".as_ptr() as *mut u8) as *const u8;
__t27.fifo_size = 128u16;
__t27.tx_loadsz = 128u16;
__t27.fcr = 129u8;
__t27.rxtrig_bytes = [1u8, 32u8, 64u8, 126u8];
__t27.flags = 2304u32;
__t27
};
__t0[29] = {
let mut __t28: serial8250_config = unsafe { core::mem::zeroed() };
__t28.name = (b"Palmchip BK-3103\0".as_ptr() as *mut u8) as *const u8;
__t28.fifo_size = 16u16;
__t28.tx_loadsz = 16u16;
__t28.fcr = 129u8;
__t28.rxtrig_bytes = [1u8, 4u8, 8u8, 14u8];
__t28.flags = 256u32;
__t28
};
__t0[30] = {
let mut __t29: serial8250_config = unsafe { core::mem::zeroed() };
__t29.name = (b"16550A_FSL64\0".as_ptr() as *mut u8) as *const u8;
__t29.fifo_size = 64u16;
__t29.tx_loadsz = 63u16;
__t29.fcr = 161u8;
__t29.flags = 262400u32;
__t29
};
__t0[40] = {
let mut __t30: serial8250_config = unsafe { core::mem::zeroed() };
__t30.name = (b"Nuvoton 16550\0".as_ptr() as *mut u8) as *const u8;
__t30.fifo_size = 16u16;
__t30.tx_loadsz = 16u16;
__t30.fcr = 135u8;
__t30.rxtrig_bytes = [1u8, 4u8, 8u8, 14u8];
__t30.flags = 256u32;
__t30
};
__t0[42] = {
let mut __t31: serial8250_config = unsafe { core::mem::zeroed() };
__t31.name = (b"ASPEED VUART\0".as_ptr() as *mut u8) as *const u8;
__t31.fifo_size = 16u16;
__t31.tx_loadsz = 16u16;
__t31.fcr = 1u8;
__t31.rxtrig_bytes = [1u8, 4u8, 8u8, 14u8];
__t31.flags = 256u32;
__t31
};
__t0[65] = {
let mut __t32: serial8250_config = unsafe { core::mem::zeroed() };
__t32.name = (b"Moxa MUEx50 UART\0".as_ptr() as *mut u8) as *const u8;
__t32.fifo_size = 128u16;
__t32.tx_loadsz = 128u16;
__t32.fcr = 129u8;
__t32.rxtrig_bytes = [15u8, 31u8, 63u8, 111u8];
__t32.flags = 256u32;
__t32
};
__t0[76] = {
let mut __t33: serial8250_config = unsafe { core::mem::zeroed() };
__t33.name = (b"Broadcom BCM7271 UART\0".as_ptr() as *mut u8) as *const u8;
__t33.fifo_size = 32u16;
__t33.tx_loadsz = 32u16;
__t33.fcr = 65u8;
__t33.rxtrig_bytes = [1u8, 8u8, 16u8, 30u8];
__t33.flags = 2304u32;
__t33
};
__t0[95] = {
let mut __t34: serial8250_config = unsafe { core::mem::zeroed() };
__t34.name = (b"TI DA8xx/66AK2x\0".as_ptr() as *mut u8) as *const u8;
__t34.fifo_size = 16u16;
__t34.tx_loadsz = 16u16;
__t34.fcr = 137u8;
__t34.rxtrig_bytes = [1u8, 4u8, 8u8, 14u8];
__t34.flags = 2304u32;
__t34
};
__t0[100] = {
let mut __t35: serial8250_config = unsafe { core::mem::zeroed() };
__t35.name = (b"MCHP16550A\0".as_ptr() as *mut u8) as *const u8;
__t35.fifo_size = 256u16;
__t35.tx_loadsz = 256u16;
__t35.fcr = 65u8;
__t35.rxtrig_bytes = [2u8, 66u8, 130u8, 194u8];
__t35.flags = 256u32;
__t35
};
__t0[117] = {
let mut __t36: serial8250_config = unsafe { core::mem::zeroed() };
__t36.name = (b"MediaTek BTIF\0".as_ptr() as *mut u8) as *const u8;
__t36.fifo_size = 16u16;
__t36.tx_loadsz = 16u16;
__t36.fcr = 7u8;
__t36.flags = 256u32;
__t36
};
__t0[121] = {
let mut __t37: serial8250_config = unsafe { core::mem::zeroed() };
__t37.name = (b"Sunix\0".as_ptr() as *mut u8) as *const u8;
__t37.fifo_size = 128u16;
__t37.tx_loadsz = 128u16;
__t37.fcr = 129u8;
__t37.rxtrig_bytes = [1u8, 32u8, 64u8, 112u8];
__t37.flags = 1280u32;
__t37
};
__t0
});

/// Uart divisor latch read
#[unsafe(link_section = ".text.slate_distinct._8250_port.default_serial_dl_read")]
extern "C" fn default_serial_dl_read(up: *mut uart_8250_port) -> u32 {
// Assign these in pieces to truncate any bits above 7.
let dll: u8 = serial_in(up, 0i32) as u8;
let dlm: u8 = serial_in(up, 1i32) as u8;
((dll as i32) | (dlm as i32) << 8i32) as u32
}

/// Uart divisor latch write
#[unsafe(link_section = ".text.slate_distinct._8250_port.default_serial_dl_write")]
extern "C" fn default_serial_dl_write(up: *mut uart_8250_port, value: u32) {
serial_out(up, 0i32, (value & 255) as i32);
serial_out(up, 1i32, (value >> 8i32 & 255) as i32);
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.hub6_serial_in")]
extern "C" fn hub6_serial_in(p: *mut uart_port, mut offset: u32) -> u32 {
offset <<= (unsafe { (*p).regshift }) as i32;
__outb((((unsafe { (*p).hub6 }) as i32).wrapping_sub(1) as u32).wrapping_add(offset) as u8, (unsafe { (*p).iobase }) as u16);
__inb(unsafe { (*p).iobase }.wrapping_add(1) as u16) as u32
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.hub6_serial_out")]
extern "C" fn hub6_serial_out(p: *mut uart_port, mut offset: u32, value: u32) {
offset <<= (unsafe { (*p).regshift }) as i32;
__outb((((unsafe { (*p).hub6 }) as i32).wrapping_sub(1) as u32).wrapping_add(offset) as u8, (unsafe { (*p).iobase }) as u16);
__outb(value as u8, unsafe { (*p).iobase }.wrapping_add(1) as u16);
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.mem_serial_in")]
extern "C" fn mem_serial_in(p: *mut uart_port, mut offset: u32) -> u32 {
offset <<= (unsafe { (*p).regshift }) as i32;
readb(unsafe { (*p).membase }.wrapping_offset(offset as isize) as *const ()) as u32
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.mem_serial_out")]
extern "C" fn mem_serial_out(p: *mut uart_port, mut offset: u32, value: u32) {
offset <<= (unsafe { (*p).regshift }) as i32;
writeb(value as u8, unsafe { (*p).membase }.wrapping_offset(offset as isize) as *mut ());
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.mem16_serial_out")]
extern "C" fn mem16_serial_out(p: *mut uart_port, mut offset: u32, value: u32) {
offset <<= (unsafe { (*p).regshift }) as i32;
writew(value as u16, unsafe { (*p).membase }.wrapping_offset(offset as isize) as *mut ());
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.mem16_serial_in")]
extern "C" fn mem16_serial_in(p: *mut uart_port, mut offset: u32) -> u32 {
offset <<= (unsafe { (*p).regshift }) as i32;
readw(unsafe { (*p).membase }.wrapping_offset(offset as isize) as *const ()) as u32
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.mem32_serial_out")]
extern "C" fn mem32_serial_out(p: *mut uart_port, mut offset: u32, value: u32) {
offset <<= (unsafe { (*p).regshift }) as i32;
writel(value, unsafe { (*p).membase }.wrapping_offset(offset as isize) as *mut ());
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.mem32_serial_in")]
extern "C" fn mem32_serial_in(p: *mut uart_port, mut offset: u32) -> u32 {
offset <<= (unsafe { (*p).regshift }) as i32;
readl(unsafe { (*p).membase }.wrapping_offset(offset as isize) as *const ())
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.mem32be_serial_out")]
extern "C" fn mem32be_serial_out(p: *mut uart_port, mut offset: u32, value: u32) {
offset <<= (unsafe { (*p).regshift }) as i32;
unsafe { iowrite32be(value, unsafe { (*p).membase }.wrapping_offset(offset as isize) as *mut ()) };
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.mem32be_serial_in")]
extern "C" fn mem32be_serial_in(p: *mut uart_port, mut offset: u32) -> u32 {
offset <<= (unsafe { (*p).regshift }) as i32;
unsafe { ioread32be(unsafe { (*p).membase }.wrapping_offset(offset as isize) as *const ()) }
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.io_serial_in")]
extern "C" fn io_serial_in(p: *mut uart_port, mut offset: u32) -> u32 {
offset <<= (unsafe { (*p).regshift }) as i32;
__inb(unsafe { (*p).iobase }.wrapping_add(offset as u64) as u16) as u32
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.io_serial_out")]
extern "C" fn io_serial_out(p: *mut uart_port, mut offset: u32, value: u32) {
offset <<= (unsafe { (*p).regshift }) as i32;
__outb(value as u8, unsafe { (*p).iobase }.wrapping_add(offset as u64) as u16);
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.no_serial_in")]
extern "C" fn no_serial_in(p: *mut uart_port, offset: u32) -> u32 {
!0
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.no_serial_out")]
extern "C" fn no_serial_out(p: *mut uart_port, offset: u32, value: u32) {
}

fn set_io_from_upio(p: *mut uart_port) {
let up: *mut uart_8250_port = up_to_u8250p(p);
unsafe {
(*up).dl_read = Some(default_serial_dl_read);
}
unsafe {
(*up).dl_write = Some(default_serial_dl_write);
}
match (unsafe { (*p).iotype }) as i32 {
1 => {
unsafe {
(*p).serial_in = Some(hub6_serial_in);
}
unsafe {
(*p).serial_out = Some(hub6_serial_out);
}
}
2 => {
unsafe {
(*p).serial_in = Some(mem_serial_in);
}
unsafe {
(*p).serial_out = Some(mem_serial_out);
}
}
7 => {
unsafe {
(*p).serial_in = Some(mem16_serial_in);
}
unsafe {
(*p).serial_out = Some(mem16_serial_out);
}
}
3 => {
unsafe {
(*p).serial_in = Some(mem32_serial_in);
}
unsafe {
(*p).serial_out = Some(mem32_serial_out);
}
}
6 => {
unsafe {
(*p).serial_in = Some(mem32be_serial_in);
}
unsafe {
(*p).serial_out = Some(mem32be_serial_out);
}
}
0 => {
unsafe {
(*p).serial_in = Some(io_serial_in);
}
unsafe {
(*p).serial_out = Some(io_serial_out);
}
}
4 => {
}
_ => {
let mut __v1: i64;
let __ret_warn_on: i32 = (((unsafe { (*p).iotype }) as i32) != 0 || (unsafe { (*p).iobase }) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Unsupported UART type %x\n".as_ptr() as *const u8, (*p).iotype as i32) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_set_io_from_upio_4, sym __slate_asm_string_set_io_from_upio_5, const 473 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, (*p).iotype as i32) };
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
(*p).serial_in = Some(no_serial_in);
}
unsafe {
(*p).serial_out = Some(no_serial_out);
}
}
}
// Remember loaded iotype
unsafe {
(*up).cur_iotype = ((*p).iotype as i32) as u8;
}
unsafe {
(*p).handle_irq = Some(serial8250_default_handle_irq);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_591: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn serial_port_out_sync(p: *mut uart_port, offset: i32, value: i32) {
if unsafe { uart_iotype_mmio((*p).iotype) } {
unsafe { unsafe { (*p).serial_out }.unwrap()(p, offset as u32, value as u32) };
unsafe { unsafe { (*p).serial_in }.unwrap()(p, 3u32) }; // safe, no side-effects
} else {
unsafe { unsafe { (*p).serial_out }.unwrap()(p, offset as u32, value as u32) };
}
}

/// FIFO support.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_clear_fifos")]
extern "C" fn serial8250_clear_fifos(p: *mut uart_8250_port) {
if ((unsafe { (*p).capabilities }) as u64) & BIT!(8i32) != 0 {
serial_out(p, 2i32, 1i32);
serial_out(p, 2i32, 7i32);
serial_out(p, 2i32, 0i32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_clear_fifos_592: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_8250_port)>, *mut ()>(Some(serial8250_clear_fifos)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_clear_and_reinit_fifos")]
extern "C" fn serial8250_clear_and_reinit_fifos(p: *mut uart_8250_port) {
serial8250_clear_fifos(p);
serial_out(p, 2i32, (unsafe { (*p).fcr }) as i32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_clear_and_reinit_fifos_593: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_8250_port)>, *mut ()>(Some(serial8250_clear_and_reinit_fifos)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_rpm_get")]
extern "C" fn serial8250_rpm_get(p: *mut uart_8250_port) {
if ((unsafe { (*p).capabilities }) as u64) & BIT!(15i32) == 0 {
return;
}
pm_runtime_get_sync(unsafe { (*p).port.dev });
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_rpm_get_594: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_8250_port)>, *mut ()>(Some(serial8250_rpm_get)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_rpm_put")]
extern "C" fn serial8250_rpm_put(p: *mut uart_8250_port) {
if ((unsafe { (*p).capabilities }) as u64) & BIT!(15i32) == 0 {
return;
}
pm_runtime_mark_last_busy(unsafe { (*p).port.dev });
pm_runtime_put_autosuspend(unsafe { (*p).port.dev });
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_rpm_put_595: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_8250_port)>, *mut ()>(Some(serial8250_rpm_put)) };

/// serial8250_em485_init() - put uart_8250_port into rs485 emulating
/// @p:	uart_8250_port port instance
///
/// The function is used to start rs485 software emulating on the
/// &struct uart_8250_port* @p. Namely, RTS is switched before/after
/// transmission. The function is idempotent, so it is safe to call it
/// multiple times.
///
/// The caller MUST enable interrupt on empty shift register before
/// calling serial8250_em485_init(). This interrupt is not a part of
/// 8250 standard, but implementation defined.
///
/// The function is supposed to be called from .rs485_config callback
/// or from any other callback protected with p->port.lock spinlock.
///
/// See also serial8250_em485_destroy()
///
/// Return 0 - success, -errno - otherwise
fn serial8250_em485_init(p: *mut uart_8250_port) -> i32 {
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
unsafe {
// Port locked to synchronize UART_IER access against the console.
lockdep_assert_held_once!(core::ptr::addr_of_mut!((*p).port.lock));
if unsafe { (*p).em485 }.is_null() {
__obj_size = size_mul(core::mem::size_of::<uart_8250_em485>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kmalloc_noprof(__obj_size, 2080u32, *{
__slate_compound_6 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_6)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kmalloc_noprof(__obj_size, 2080u32, *{
__slate_compound_8 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_8)
});
}
(*p).em485 = *__slate_slot_1 as *mut uart_8250_em485;
if unsafe { (*p).em485 }.is_null() {
return 12i32.wrapping_neg();
} else {
hrtimer_setup(core::ptr::addr_of_mut!((*(*p).em485).stop_tx_timer), Some(serial8250_em485_handle_stop_tx), 1i32, 1u32);
hrtimer_setup(core::ptr::addr_of_mut!((*(*p).em485).start_tx_timer), Some(serial8250_em485_handle_start_tx), 1i32, 1u32);
(*(*p).em485).port = p;
(*(*p).em485).active_timer = core::ptr::null_mut::<hrtimer>();
(*(*p).em485).__slate_bits_0.__set_tx_stopped(1u32);
}
}
if ((*(*p).em485).__slate_bits_0.__get_tx_stopped() as i32) != 0 {
unsafe { (*p).rs485_stop_tx }.unwrap()(p, true);
}
return 0;
}
}

/// serial8250_em485_destroy() - put uart_8250_port into normal state
/// @p:	uart_8250_port port instance
///
/// The function is used to stop rs485 software emulating on the
/// &struct uart_8250_port* @p. The function is idempotent, so it is safe to
/// call it multiple times.
///
/// The function is supposed to be called from .rs485_config callback
/// or from any other callback protected with p->port.lock spinlock.
///
/// See also serial8250_em485_init()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_em485_destroy")]
extern "C" fn serial8250_em485_destroy(p: *mut uart_8250_port) {
if unsafe { (*p).em485 }.is_null() {
return;
}
unsafe { hrtimer_cancel(core::ptr::addr_of_mut!((*(*p).em485).start_tx_timer)) };
unsafe { hrtimer_cancel(core::ptr::addr_of_mut!((*(*p).em485).stop_tx_timer)) };
unsafe { kfree((*p).em485 as *const ()) };
unsafe {
(*p).em485 = core::ptr::null_mut::<uart_8250_em485>();
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_em485_destroy_596: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_8250_port)>, *mut ()>(Some(serial8250_em485_destroy)) };

#[unsafe(no_mangle)]
static mut serial8250_em485_supported: serial_rs485 = {
let mut __t0: serial_rs485 = unsafe { core::mem::zeroed() };
__t0.flags = 55u32;
__t0.delay_rts_before_send = 1u32;
__t0.delay_rts_after_send = 1u32;
__t0
};

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_em485_supported_597: *mut () = (unsafe { core::ptr::addr_of_mut!(serial8250_em485_supported) }) as *mut ();

/// serial8250_em485_config() - generic ->rs485_config() callback
/// @port: uart port
/// @termios: termios structure
/// @rs485: rs485 settings
///
/// Generic callback usable by 8250 uart drivers to activate rs485 settings
/// if the uart is incapable of driving RTS as a Transmit Enable signal in
/// hardware, relying on software emulation instead.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_em485_config")]
extern "C" fn serial8250_em485_config(port: *mut uart_port, termios: *mut ktermios, rs485: *mut serial_rs485) -> i32 {
let up: *mut uart_8250_port = up_to_u8250p(port);
// Both serial8250_em485_init() and serial8250_em485_destroy()
// are idempotent.
if ((unsafe { (*rs485).flags }) as u64) & _BITUL!(0i32) != 0 {
return serial8250_em485_init(up);
}
serial8250_em485_destroy(up);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_em485_config_598: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_port, *mut ktermios, *mut serial_rs485) -> i32>, *mut ()>(Some(serial8250_em485_config)) };

/// These two wrappers ensure that enable_runtime_pm_tx() can be called more than
/// once and disable_runtime_pm_tx() will still disable RPM because the fifo is
/// empty and the HW can idle again.
fn serial8250_rpm_get_tx(p: *mut uart_8250_port) {
let mut rpm_active: u8 = 0;
if ((unsafe { (*p).capabilities }) as u64) & BIT!(15i32) == 0 {
return;
}
let mut __v0: u32;
let mut __ai_ptr: *mut u32 = unsafe { core::ptr::addr_of_mut!((*p).rpm_tx_active) };
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<u32>() as u64);
let mut __v2: u32;
let mut __ret: u32 = 1;
match (const { core::mem::size_of::<u32>() }) as u64 {
1 => {
unsafe {
let __t0: u32;
core::arch::asm!("xchgb {0:l}, ({1:r})", inlateout(reg) __ret as u32 => __t0, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
}
2 => {
unsafe {
let __t1: u32;
core::arch::asm!("xchgw {0:x}, ({1:r})", inlateout(reg) __ret as u32 => __t1, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
}
4 => {
unsafe {
let __t2: u32;
core::arch::asm!("xchgl {0:e}, ({1:r})", inlateout(reg) __ret as u32 => __t2, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
}
8 => {
unsafe {
let __t3: u32;
core::arch::asm!("xchgq {0:r}, ({1:r})", inlateout(reg) __ret as u32 => __t3, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t3 as u32) };
}
}
_ => {
unsafe { __xchg_wrong_size() };
}
}
__v2 = __ret;
__v0 = __v2;
rpm_active = __v0 as u8;
if rpm_active != 0 {
return;
}
pm_runtime_get_sync(unsafe { (*p).port.dev });
}

fn serial8250_rpm_put_tx(p: *mut uart_8250_port) {
let mut rpm_active: u8 = 0;
if ((unsafe { (*p).capabilities }) as u64) & BIT!(15i32) == 0 {
return;
}
let mut __v0: u32;
let mut __ai_ptr: *mut u32 = unsafe { core::ptr::addr_of_mut!((*p).rpm_tx_active) };
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<u32>() as u64);
let mut __v2: u32;
let mut __ret: u32 = 0;
match (const { core::mem::size_of::<u32>() }) as u64 {
1 => {
unsafe {
let __t0: u32;
core::arch::asm!("xchgb {0:l}, ({1:r})", inlateout(reg) __ret as u32 => __t0, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
}
2 => {
unsafe {
let __t1: u32;
core::arch::asm!("xchgw {0:x}, ({1:r})", inlateout(reg) __ret as u32 => __t1, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
}
4 => {
unsafe {
let __t2: u32;
core::arch::asm!("xchgl {0:e}, ({1:r})", inlateout(reg) __ret as u32 => __t2, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
}
8 => {
unsafe {
let __t3: u32;
core::arch::asm!("xchgq {0:r}, ({1:r})", inlateout(reg) __ret as u32 => __t3, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t3 as u32) };
}
}
_ => {
unsafe { __xchg_wrong_size() };
}
}
__v2 = __ret;
__v0 = __v2;
rpm_active = __v0 as u8;
if rpm_active == 0 {
return;
}
pm_runtime_mark_last_busy(unsafe { (*p).port.dev });
pm_runtime_put_autosuspend(unsafe { (*p).port.dev });
}

/// IER sleep support.  UARTs which have EFRs need the "extended
/// capability" bit enabled.  Note that on XR16C850s, we need to
/// reset LCR to write to IER.
fn serial8250_set_sleep(p: *mut uart_8250_port, sleep: i32) {
let mut lcr: u8 = 0;
let mut efr: u8 = 0;
let mut __UNIQUE_ID_guard_599: *mut uart_8250_port = class_serial8250_rpm_constructor(p);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_599) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_serial8250_rpm_destructor(__slate_cleanup_ptr_0 as *mut _));
if ((unsafe { (*p).capabilities }) as u64) & BIT!(10i32) == 0 {
return;
}
// Synchronize UART_IER access against the console.
let mut __UNIQUE_ID_guard_600: *mut uart_port = class_uart_port_lock_irq_constructor(unsafe { core::ptr::addr_of_mut!((*p).port) });
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_600) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_uart_port_lock_irq_destructor(__slate_cleanup_ptr_1 as *mut _));
if ((unsafe { (*p).capabilities }) as u64) & BIT!(9i32) != 0 {
lcr = serial_in(p, 3i32) as u8;
efr = serial_in(p, 2i32) as u8;
serial_out(p, 3i32, 191i32);
serial_out(p, 2i32, 16i32);
serial_out(p, 3i32, 0i32);
}
serial_out(p, 1i32, if sleep != 0 { 16i32 } else { 0i32 });
if ((unsafe { (*p).capabilities }) as u64) & BIT!(9i32) != 0 {
serial_out(p, 3i32, 191i32);
serial_out(p, 2i32, efr as i32);
serial_out(p, 3i32, lcr as i32);
}
}

/// Clear the interrupt registers.
fn serial8250_clear_interrupts(port: *mut uart_port) {
serial_port_in(port, 5i32);
serial_port_in(port, 0i32);
serial_port_in(port, 2i32);
serial_port_in(port, 6i32);
}

/// Only to be directly used by serial8250_console_write() and
/// serial8250_put_poll_char(), which do not require the port lock.
/// Use serial8250_clear_IER() instead for all other cases.
fn __serial8250_clear_IER(up: *mut uart_8250_port) {
if ((unsafe { (*up).capabilities }) as u64) & BIT!(12i32) != 0 {
serial_out(up, 1i32, 64i32);
} else {
serial_out(up, 1i32, 0i32);
}
}

fn serial8250_clear_IER(up: *mut uart_8250_port) {
// Port locked to synchronize UART_IER access against the console
lockdep_assert_held_once!(unsafe { core::ptr::addr_of_mut!((*up).port.lock) });
__serial8250_clear_IER(up);
}

/// This is a quickie test to see how big the FIFO is.
/// It doesn't work at all the time, more's the pity.
fn size_fifo(up: *mut uart_8250_port) -> i32 {
let mut old_fcr: u8 = 0;
let mut old_mcr: u8 = 0;
let old_lcr: u8 = serial_in(up, 3i32) as u8;
let mut old_dl: u32 = 0;
let mut count: i32 = 0;
serial_out(up, 3i32, 0i32);
old_fcr = serial_in(up, 2i32) as u8;
old_mcr = serial8250_in_MCR(up) as u8;
serial_out(up, 2i32, 7i32);
serial8250_out_MCR(up, 16i32);
serial_out(up, 3i32, 128i32);
old_dl = serial_dl_read(up);
serial_dl_write(up, 1u32);
serial_out(up, 3i32, 3i32);
while count < 256 {
serial_out(up, 0i32, count);
count = count.wrapping_add(1);
}
mdelay!(20i32); // FIXME - schedule_timeout
count = 0;
while serial_in(up, 5i32) & 1 != 0 && count < 256 {
serial_in(up, 0i32);
count = count.wrapping_add(1);
}
serial_out(up, 2i32, old_fcr as i32);
serial8250_out_MCR(up, old_mcr as i32);
serial_out(up, 3i32, 128i32);
serial_dl_write(up, old_dl);
serial_out(up, 3i32, old_lcr as i32);
count
}

/// Read UART ID using the divisor method - set DLL and DLM to zero
/// and the revision will be in DLL and device type in DLM.  We
/// preserve the device state across this.
fn autoconfig_read_divisor_id(p: *mut uart_8250_port) -> u32 {
let old_lcr: u8 = serial_in(p, 3i32) as u8;
let mut id: u32 = 0;
let mut old_dl: u32 = 0;
serial_out(p, 3i32, 128i32);
old_dl = serial_dl_read(p);
serial_dl_write(p, 0u32);
id = serial_dl_read(p);
serial_dl_write(p, old_dl);
serial_out(p, 3i32, old_lcr as i32);
id
}

/// This is a helper routine to autodetect StarTech/Exar/Oxsemi UART's.
/// When this function is called we know it is at least a StarTech
/// 16650 V2, but it might be one of several StarTech UARTs, or one of
/// its clones.  (We treat the broken original StarTech 16650 V1 as a
/// 16550, and why not?  Startech doesn't seem to even acknowledge its
/// existence.)
///
/// What evil have men's minds wrought...
fn autoconfig_has_efr(up: *mut uart_8250_port) {
let mut id1: u32 = 0;
let mut id2: u32 = 0;
let mut id3: u32 = 0;
let mut rev: u32 = 0;
// Everything with an EFR has SLEEP
let __v1: u32 = unsafe { (*up).capabilities };
let __v2: u32 = ((__v1 as u64) | (BIT!(9i32) | BIT!(10i32))) as u32;
unsafe {
(*up).capabilities = __v2;
}
// First we check to see if it's an Oxford Semiconductor UART.
//
// If we have to do this here because some non-National
// Semiconductor clone chips lock up if you try writing to the
// LSR register (which serial_icr_read does)
// Check for Oxford Semiconductor 16C950.
//
// EFR [4] must be set else this test fails.
//
// This shouldn't be necessary, but Mike Hudson (Exoray@isys.ca)
// claims that it's needed for 952 dual UART's (which are not
// recommended for new designs).
unsafe {
(*up).acr = 0;
}
serial_out(up, 3i32, 191i32);
serial_out(up, 2i32, 16i32);
serial_out(up, 3i32, 0i32);
id1 = serial_icr_read(up, 8i32);
id2 = serial_icr_read(up, 9i32);
id3 = serial_icr_read(up, 10i32);
rev = serial_icr_read(up, 11i32);
if id1 == 22 && id2 == 201 && (id3 == 80 || id3 == 82 || id3 == 84) {
unsafe {
(*up).port.r#type = 10;
}
// Enable work around for the Oxford Semiconductor 952 rev B
// chip which causes it to seriously miscalculate baud rates
// when DLL is 0.
if id3 == 82 && rev == 1 {
let __v4: u16 = unsafe { (*up).bugs };
let __v5: u16 = (((__v4 as i32) as u64) | BIT!(0i32)) as u16;
unsafe {
(*up).bugs = __v5;
}
}
return;
}
// We check for a XR16C850 by setting DLL and DLM to 0, and then
// reading back DLL and DLM.  The chip type depends on the DLM
// value read back:
//  0x10 - XR16C850 and the DLL contains the chip revision.
//  0x12 - XR16C2850.
//  0x14 - XR16C854.
id1 = autoconfig_read_divisor_id(up);
id2 = id1 >> 8i32;
if id2 == 16 || id2 == 18 || id2 == 20 {
unsafe {
(*up).port.r#type = 12;
}
return;
}
// It wasn't an XR16C850.
//
// We distinguish between the '654 and the '650 by counting
// how many bytes are in the FIFO.  I'm using this for now,
// since that's the technique that was sent to me in the
// serial driver update, but I'm not convinced this works.
// I've had problems doing this in the past.  -TYT
if size_fifo(up) == 64 {
unsafe {
(*up).port.r#type = 11;
}
} else {
unsafe {
(*up).port.r#type = 7;
}
}
}

/// We detected a chip without a FIFO.  Only two fall into
/// this category - the original 8250 and the 16450.  The
/// 16450 has a scratch register (accessible with LCR=0)
fn autoconfig_8250(up: *mut uart_8250_port) {
let mut scratch: u8 = 0;
let mut status1: u8 = 0;
let mut status2: u8 = 0;
unsafe {
(*up).port.r#type = 1;
}
scratch = serial_in(up, 7i32) as u8;
serial_out(up, 7i32, 165i32);
status1 = serial_in(up, 7i32) as u8;
serial_out(up, 7i32, 90i32);
status2 = serial_in(up, 7i32) as u8;
serial_out(up, 7i32, scratch as i32);
if (status1 as i32) == 165 && (status2 as i32) == 90 {
unsafe {
(*up).port.r#type = 2;
}
}
}

fn broken_efr(up: *mut uart_8250_port) -> i32 {
// Exar ST16C2550 "A2" devices incorrectly detect as
// having an EFR, and report an ID of 0x0201.  See
// http://linux.derkeiler.com/Mailing-Lists/Kernel/2004-11/4812.html
let __v0: bool = autoconfig_read_divisor_id(up) == 513 && size_fifo(up) == 16;
if __v0 {
return 1;
}
0
}

/// We know that the chip has FIFOs.  Does it have an EFR?  The
/// EFR is located in the same register position as the IIR and
/// we know the top two bits of the IIR are currently set.  The
/// EFR should contain zero.  Try to read the EFR.
fn autoconfig_16550a(up: *mut uart_8250_port) {
let mut status1: u8 = 0;
let mut status2: u8 = 0;
let mut iersave: u32 = 0;
// Port locked to synchronize UART_IER access against the console.
lockdep_assert_held_once!(unsafe { core::ptr::addr_of_mut!((*up).port.lock) });
unsafe {
(*up).port.r#type = 4;
}
let __v2: u32 = unsafe { (*up).capabilities };
let __v3: u32 = ((__v2 as u64) | BIT!(8i32)) as u32;
unsafe {
(*up).capabilities = __v3;
}
if IS_ENABLED!() == 0 && (unsafe { (*up).port.flags }) & BIT_ULL!(32i32) == 0 {
return;
}
// Check for presence of the EFR when DLAB is set.
// Only ST16C650V1 UARTs pass this test.
serial_out(up, 3i32, 128i32);
if serial_in(up, 2i32) == 0 {
serial_out(up, 2i32, 168i32);
if serial_in(up, 2i32) != 0 {
unsafe {
(*up).port.r#type = 6;
}
let __v5: u32 = unsafe { (*up).capabilities };
let __v6: u32 = ((__v5 as u64) | (BIT!(9i32) | BIT!(10i32))) as u32;
unsafe {
(*up).capabilities = __v6;
}
} else {
serial_out(up, 3i32, 0i32);
serial_out(up, 2i32, 33i32);
status1 = (serial_in(up, 2i32) & 224) as u8;
serial_out(up, 2i32, 0i32);
serial_out(up, 3i32, 0i32);
if (status1 as i32) == 224 {
unsafe {
(*up).port.r#type = 30;
}
}
}
serial_out(up, 2i32, 0i32);
return;
}
// Maybe it requires 0xbf to be written to the LCR.
// (other ST16C650V2 UARTs, TI16C752A, etc)
serial_out(up, 3i32, 191i32);
let __v7: bool = serial_in(up, 2i32) == 0 && broken_efr(up) == 0;
if __v7 {
autoconfig_has_efr(up);
return;
}
// Check for a National Semiconductor SuperIO chip.
// Attempt to switch to bank 2, read the value of the LOOP bit
// from EXCR1. Switch back to bank 0, change it in MCR. Then
// switch back to bank 2, read it from EXCR1 again and check
// it's changed. If so, set baud_base in EXCR2 to 921600. -- dwmw2
serial_out(up, 3i32, 0i32);
status1 = serial8250_in_MCR(up) as u8;
serial_out(up, 3i32, 224i32);
status2 = serial_in(up, 2i32) as u8; // EXCR1
if ((status2 as i32) ^ (status1 as i32)) & 16 == 0 {
serial_out(up, 3i32, 0i32);
serial8250_out_MCR(up, (status1 as i32) ^ 16);
serial_out(up, 3i32, 224i32);
status2 = serial_in(up, 2i32) as u8; // EXCR1
serial_out(up, 3i32, 0i32);
serial8250_out_MCR(up, status1 as i32);
if ((status2 as i32) ^ (status1 as i32)) & 16 != 0 {
let mut quot: u16 = 0;
serial_out(up, 3i32, 224i32);
quot = serial_dl_read(up) as u16;
quot = ((quot as i32) << 3i32) as u16;
if ns16550a_goto_highspeed(up) != 0 {
serial_dl_write(up, quot as u32);
}
serial_out(up, 3i32, 0i32);
unsafe {
(*up).port.uartclk = 14745600u32;
}
unsafe {
(*up).port.r#type = 14;
}
unsafe {
(*up).capabilities = (*up).capabilities | 8;
}
return;
}
}
// No EFR.  Try to detect a TI16750, which only sets bit 5 of
// the IIR when 64 byte FIFO mode is enabled when DLAB is set.
// Try setting it with and without DLAB set.  Cheap clones
// set bit 5 without DLAB set.
serial_out(up, 3i32, 0i32);
serial_out(up, 2i32, 33i32);
status1 = (serial_in(up, 2i32) & 224) as u8;
serial_out(up, 2i32, 1i32);
serial_out(up, 3i32, 128i32);
serial_out(up, 2i32, 33i32);
status2 = (serial_in(up, 2i32) & 224) as u8;
serial_out(up, 2i32, 1i32);
serial_out(up, 3i32, 0i32);
if (status1 as i32) == 192 && (status2 as i32) == 224 {
unsafe {
(*up).port.r#type = 8;
}
let __v14: u32 = unsafe { (*up).capabilities };
let __v15: u32 = ((__v14 as u64) | (BIT!(11i32) | BIT!(10i32))) as u32;
unsafe {
(*up).capabilities = __v15;
}
return;
}
// Try writing and reading the UART_IER_UUE bit (b6).
// If it works, this is probably one of the Xscale platform's
// internal UARTs.
// We're going to explicitly set the UUE bit to 0 before
// trying to write and read a 1 just to make sure it's not
// already a 1 and maybe locked there before we even start.
iersave = serial_in(up, 1i32) as u32;
serial_out(up, 1i32, (iersave & 4294967231u32) as i32);
if serial_in(up, 1i32) & 64 == 0 {
// OK it's in a known zero state, try writing and reading
// without disturbing the current state of the other bits.
serial_out(up, 1i32, (iersave | 64) as i32);
if serial_in(up, 1i32) & 64 != 0 {
// It's an Xscale.
// We'll leave the UART_IER_UUE bit set to 1 (enabled).
unsafe {
(*up).port.r#type = 15;
}
let __v17: u32 = unsafe { (*up).capabilities };
let __v18: u32 = ((__v17 as u64) | (BIT!(12i32) | BIT!(13i32))) as u32;
unsafe {
(*up).capabilities = __v18;
}
return;
}
}
serial_out(up, 1i32, iersave as i32);
// We distinguish between 16550A and U6 16550A by counting
// how many bytes are in the FIFO.
let __v19: bool = (unsafe { (*up).port.r#type }) == 4 && size_fifo(up) == 64;
if __v19 {
unsafe {
(*up).port.r#type = 19;
}
let __v21: u32 = unsafe { (*up).capabilities };
let __v22: u32 = ((__v21 as u64) | BIT!(11i32)) as u32;
unsafe {
(*up).capabilities = __v22;
}
}
}

/// This routine is called by rs_init() to initialize a specific serial
/// port.  It determines what type of UART chip this serial port is
/// using: 8250, 16450, 16550, 16550A.  The important question is
/// whether or not this UART is a 16550A or not, since this will
/// determine whether or not we can use its FIFO features or not.
fn autoconfig(up: *mut uart_8250_port) {
let mut status1: u8 = 0;
let mut scratch: u8 = 0;
let mut scratch2: u8 = 0;
let mut scratch3: u8 = 0;
let mut save_lcr: u8 = 0;
let mut save_mcr: u8 = 0;
let port: *mut uart_port = unsafe { core::ptr::addr_of_mut!((*up).port) };
let mut flags: u64 = 0;
let mut old_capabilities: u32 = 0;
if (unsafe { (*port).iobase }) == 0 && (unsafe { (*port).mapbase }) == 0 && unsafe { (*port).membase }.is_null() {
return;
}
// We really do need global IRQs disabled here - we're going to
// be frobbing the chips IRQ enable register to see if it exists.
//
// Synchronize UART_IER access against the console.
uart_port_lock_irqsave(port, core::ptr::addr_of_mut!(flags));
unsafe {
(*up).capabilities = 0;
}
unsafe {
(*up).bugs = 0;
}
if (unsafe { (*port).flags }) & 16384u64 == 0 {
// Do a simple existence test first; if we fail this,
// there's no point trying anything else.
//
// 0x80 is used as a nonsense port to prevent against
// false positives due to ISA bus float.  The
// assumption is that 0x80 is a non-existent port;
// which should be safe since include/asm/io.h also
// makes this assumption.
//
// Note: this is safe as long as MCR bit 4 is clear
// and the device is in "PC" mode.
scratch = serial_in(up, 1i32) as u8;
serial_out(up, 1i32, 0i32);
// Mask out IER[7:4] bits for test as some UARTs (e.g. TL
// 16C754B) allow only to modify them if an EFR bit is set.
scratch2 = (serial_in(up, 1i32) & (8 | 4 | 2 | 1)) as u8;
serial_out(up, 1i32, 15i32);
scratch3 = (serial_in(up, 1i32) & (8 | 4 | 2 | 1)) as u8;
serial_out(up, 1i32, scratch as i32);
if (scratch2 as i32) != 0 || (scratch3 as i32) != 8 | 4 | 2 | 1 {
// We failed; there's nothing here
uart_port_unlock_irqrestore(port, flags);
return;
}
}
save_mcr = serial8250_in_MCR(up) as u8;
save_lcr = serial_in(up, 3i32) as u8;
// Check to see if a UART is really there.  Certain broken
// internal modems based on the Rockwell chipset fail this
// test, because they apparently don't implement the loopback
// test mode.  So this test is skipped on the COM 1 through
// COM 4 ports.  This *should* be safe, since no board
// manufacturer would be stupid enough to design a board
// that conflicts with COM 1-4 --- we hope!
if (unsafe { (*port).flags }) & 64u64 == 0 {
serial8250_out_MCR(up, 26i32);
status1 = (serial_in(up, 6i32) & (128 | 64 | 32 | 16)) as u8;
serial8250_out_MCR(up, save_mcr as i32);
if (status1 as i32) != 128 | 16 {
uart_port_unlock_irqrestore(port, flags);
return;
}
}
// We're pretty sure there's a port here.  Lets find out what
// type of port it is.  The IIR top two bits allows us to find
// out if it's 8250 or 16450, 16550, 16550A or later.  This
// determines what we test for next.
//
// We also initialise the EFR (if any) to zero for later.  The
// EFR occupies the same register location as the FCR and IIR.
serial_out(up, 3i32, 191i32);
serial_out(up, 2i32, 0i32);
serial_out(up, 3i32, 0i32);
serial_out(up, 2i32, 1i32);
match serial_in(up, 2i32) & 192 {
0 => {
autoconfig_8250(up);
}
128 => {
unsafe {
(*port).r#type = 3;
}
}
192 => {
autoconfig_16550a(up);
}
_ => {
unsafe {
(*port).r#type = 0;
}
}
}
unsafe { rsa_autoconfig(up) };
serial_out(up, 3i32, save_lcr as i32);
unsafe {
(*port).fifosize = uart_config.0[(*up).port.r#type as usize].fifo_size as u32;
}
old_capabilities = unsafe { (*up).capabilities };
unsafe {
(*up).capabilities = uart_config.0[(*port).r#type as usize].flags;
}
unsafe {
(*up).tx_loadsz = uart_config.0[(*port).r#type as usize].tx_loadsz as u32;
}
if (unsafe { (*port).r#type }) != 0 {
// Reset the UART.
unsafe { rsa_reset(up) };
serial8250_out_MCR(up, save_mcr as i32);
serial8250_clear_fifos(up);
serial_in(up, 0i32);
serial8250_clear_IER(up);
}
uart_port_unlock_irqrestore(port, flags);
// Check if the device is a Fintek F81216A
if (unsafe { (*port).r#type }) == 4 && ((unsafe { (*port).iotype }) as i32) == 0 {
fintek_8250_probe(up);
}
if (unsafe { (*up).capabilities }) != old_capabilities {
while false {
}
unsafe { _dev_warn((*port).dev as *const device, c"detected caps %08x should be %08x\n".as_ptr() as *const u8, old_capabilities, (*up).capabilities) };
}
}

fn autoconfig_irq(up: *mut uart_8250_port) {
// Synchronize UART_IER access against the console.
let mut scope: core::mem::MaybeUninit<*mut uart_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut uart_port = core::ptr::addr_of_mut!(scope) as *mut *mut uart_port;
// Synchronize UART_IER access against the console.
let mut scope_1: core::mem::MaybeUninit<*mut uart_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut uart_port = core::ptr::addr_of_mut!(scope_1) as *mut *mut uart_port;
// Synchronize UART_IER access against the console.
let mut scope: core::mem::MaybeUninit<*mut uart_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut uart_port = core::ptr::addr_of_mut!(scope) as *mut *mut uart_port;
let mut irq: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(irq) as *mut i32;
let mut irqs: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u64 = core::ptr::addr_of_mut!(irqs) as *mut u64;
let mut ICP: u32 = unsafe { core::mem::zeroed() };
let mut save_ICP: u8 = unsafe { core::mem::zeroed() };
let mut save_ier: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u8 = core::ptr::addr_of_mut!(save_ier) as *mut u8;
let mut save_mcr: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u8 = core::ptr::addr_of_mut!(save_mcr) as *mut u8;
let mut port: *mut uart_port = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_2 = __slate_slot_2;
let __slate_cleanup_armed_2 = core::cell::Cell::new(false);
let __slate_cleanup_2 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_2.get() {
unsafe { class_uart_port_lock_irq_destructor(__slate_cleanup_ptr_2 as *mut _) };
}
});
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_uart_port_lock_irq_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { class_uart_port_lock_irq_destructor(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
port = core::ptr::addr_of_mut!((*up).port);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
save_ICP = 0;
ICP = 0;
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if (*port).flags & 2u64 != 0 {
ICP = ((*port).iobase & 4064 | 31) as u32;
save_ICP = inb_p(ICP as u16);
outb_p(128u8, ICP as u16);
inb_p(ICP as u16);
}
// forget possible initially masked and pending IRQ
probe_irq_off(probe_irq_on());
*__slate_slot_8 = serial8250_in_MCR(up) as u8;
core::ptr::write(__slate_slot_2, class_uart_port_lock_irq_constructor(port));
__slate_cleanup_armed_2.set(true);
if !class_uart_port_lock_irq_lock_ptr(&(*core::ptr::addr_of_mut!(*__slate_slot_2))).is_null() || !class_uart_port_lock_irq_is_conditional {
if true {
*__slate_slot_7 = serial_in(up, 1i32) as u8;
}
}
__slate_cleanup_armed_2.set(false);
class_uart_port_lock_irq_destructor(__slate_slot_2 as *mut _);
serial8250_out_MCR(up, 12i32);
*__slate_slot_4 = probe_irq_on();
serial8250_out_MCR(up, 0i32);
udelay(10u64);
if (*port).flags & 2u64 != 0 {
serial8250_out_MCR(up, 3i32);
} else {
serial8250_out_MCR(up, 11i32);
}
core::ptr::write(__slate_slot_1, class_uart_port_lock_irq_constructor(port));
__slate_cleanup_armed_1.set(true);
if !class_uart_port_lock_irq_lock_ptr(&(*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_uart_port_lock_irq_is_conditional {
if true {
serial_out(up, 1i32, 15i32);
}
}
__slate_cleanup_armed_1.set(false);
class_uart_port_lock_irq_destructor(__slate_slot_1 as *mut _);
serial8250_clear_interrupts(port);
serial_out(up, 0i32, 255i32);
udelay(20u64);
*__slate_slot_3 = probe_irq_off(*__slate_slot_4);
serial8250_out_MCR(up, *__slate_slot_8 as i32);
core::ptr::write(__slate_slot_0, class_uart_port_lock_irq_constructor(port));
__slate_cleanup_armed_0.set(true);
if !class_uart_port_lock_irq_lock_ptr(&(*core::ptr::addr_of_mut!(*__slate_slot_0))).is_null() || !class_uart_port_lock_irq_is_conditional {
if true {
serial_out(up, 1i32, *__slate_slot_7 as i32);
}
}
__slate_cleanup_armed_0.set(false);
class_uart_port_lock_irq_destructor(__slate_slot_0 as *mut _);
if (*port).flags & 2u64 != 0 {
outb_p(save_ICP, ICP as u16);
}
(*port).irq = (if *__slate_slot_3 > 0 { *__slate_slot_3 } else { 0i32 }) as u32;
}
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_stop_rx")]
extern "C" fn serial8250_stop_rx(port: *mut uart_port) {
let up: *mut uart_8250_port = up_to_u8250p(port);
// Port locked to synchronize UART_IER access against the console.
lockdep_assert_held_once!(unsafe { core::ptr::addr_of_mut!((*port).lock) });
let mut __UNIQUE_ID_guard_604: *mut uart_8250_port = class_serial8250_rpm_constructor(up);
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_604) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_serial8250_rpm_destructor(__slate_cleanup_ptr_1 as *mut _));
let __v4: u8 = (((unsafe { (*up).ier }) as i32) & !(4 | 1)) as u8;
unsafe {
(*up).ier = __v4;
}
serial_port_out(port, 1i32, (unsafe { (*up).ier }) as i32);
}

/// serial8250_em485_stop_tx() - generic ->rs485_stop_tx() callback
/// @p: uart 8250 port
/// @toggle_ier: true to allow enabling receive interrupts
///
/// Generic callback usable by 8250 uart drivers to stop rs485 transmission.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_em485_stop_tx")]
extern "C" fn serial8250_em485_stop_tx(p: *mut uart_8250_port, toggle_ier: bool) {
let mut mcr: u8 = serial8250_in_MCR(p) as u8;
if ((unsafe { (*p).port.rs485.flags }) as u64) & _BITUL!(2i32) != 0 {
mcr = ((mcr as i32) | 2) as u8;
} else {
let __v3: u8 = ((mcr as i32) & !2) as u8;
mcr = __v3;
}
serial8250_out_MCR(p, mcr as i32);
// Empty the RX FIFO, we are not interested in anything
// received during the half-duplex transmission.
// Enable previously disabled RX interrupts.
if ((unsafe { (*p).port.rs485.flags }) as u64) & _BITUL!(4i32) == 0 {
serial8250_clear_and_reinit_fifos(p);
if toggle_ier {
// Port locked to synchronize UART_IER access against
// the console. The lockdep_assert must be restricted
// to this condition because only here is it
// guaranteed that the port lock is held. The other
// hardware access in this function is synchronized
// by console ownership.
lockdep_assert_held_once!(unsafe { core::ptr::addr_of_mut!((*p).port.lock) });
unsafe {
(*p).ier = (((*p).ier as i32) | (4 | 1)) as u8;
}
serial_port_out(unsafe { core::ptr::addr_of_mut!((*p).port) }, 1i32, (unsafe { (*p).ier }) as i32);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_em485_stop_tx_605: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_8250_port, bool)>, *mut ()>(Some(serial8250_em485_stop_tx)) };

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_em485_handle_stop_tx")]
extern "C" fn serial8250_em485_handle_stop_tx(t: *mut hrtimer) -> u32 {
let em485: *mut uart_8250_em485 = container_of__15!(t, uart_8250_em485);
let p: *mut uart_8250_port = unsafe { (*em485).port };
let mut __UNIQUE_ID_guard_606: *mut uart_8250_port = class_serial8250_rpm_constructor(p);
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_606) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_serial8250_rpm_destructor(__slate_cleanup_ptr_1 as *mut _));
let mut __UNIQUE_ID_guard_607: __SlateRecord_serial_core_h_803 = class_uart_port_lock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!((*p).port) });
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_607) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || class_uart_port_lock_irqsave_destructor(__slate_cleanup_ptr_2 as *mut _));
if (unsafe { (*em485).active_timer }) == unsafe { core::ptr::addr_of_mut!((*em485).stop_tx_timer) } {
unsafe { unsafe { (*p).rs485_stop_tx }.unwrap()(p, true) };
unsafe {
(*em485).active_timer = core::ptr::null_mut::<hrtimer>();
}
unsafe {
(*em485).__slate_bits_0.__set_tx_stopped(1u32);
}
}
0
}

fn start_hrtimer_ms(hrt: *mut hrtimer, msec: u64) {
hrtimer_start(hrt, ms_to_ktime(msec), 1u32);
}

fn __stop_tx_rs485(p: *mut uart_8250_port, mut stop_delay: u64) {
let em485: *mut uart_8250_em485 = unsafe { (*p).em485 };
// Port locked to synchronize UART_IER access against the console.
lockdep_assert_held_once!(unsafe { core::ptr::addr_of_mut!((*p).port.lock) });
stop_delay = stop_delay.wrapping_add(((unsafe { (*p).port.rs485.delay_rts_after_send }) as u64).wrapping_mul(1000000));
// rs485_stop_tx() is going to set RTS according to config
// AND flush RX FIFO if required.
if stop_delay > 0 {
unsafe {
(*em485).active_timer = core::ptr::addr_of_mut!((*em485).stop_tx_timer);
}
hrtimer_start(unsafe { core::ptr::addr_of_mut!((*em485).stop_tx_timer) }, ns_to_ktime(stop_delay), 1u32);
} else {
unsafe { unsafe { (*p).rs485_stop_tx }.unwrap()(p, true) };
unsafe {
(*em485).active_timer = core::ptr::null_mut::<hrtimer>();
}
unsafe {
(*em485).__slate_bits_0.__set_tx_stopped(1u32);
}
}
}

fn __stop_tx(p: *mut uart_8250_port) {
let em485: *mut uart_8250_em485 = unsafe { (*p).em485 };
if !em485.is_null() {
let lsr: u16 = serial_lsr_in(p);
let mut stop_delay: u64 = 0;
if (lsr as i32) & 32 == 0 {
return;
}
// To provide required timing and allow FIFO transfer,
// __stop_tx_rs485() must be called only when both FIFO and
// shift register are empty. The device driver should either
// enable interrupt on TEMT or set UART_CAP_NOTEMT that will
// enlarge stop_tx_timer by the tx time of one frame to cover
// for emptying of the shift register.
if (lsr as i32) & 64 == 0 {
if ((unsafe { (*p).capabilities }) as u64) & BIT!(18i32) == 0 {
return;
}
// RTS might get deasserted too early with the normal
// frame timing formula. It seems to suggest THRE might
// get asserted already during tx of the stop bit
// rather than after it is fully sent.
// Roughly estimate 1 extra bit here with / 7.
stop_delay = unsafe { (*p).port.frame_time }.wrapping_add(unsafe { (*p).port.frame_time }.wrapping_add(7).wrapping_sub(1) / 7) as u64;
}
__stop_tx_rs485(p, stop_delay);
}
if serial8250_clear_THRI(p) {
serial8250_rpm_put_tx(p);
}
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_stop_tx")]
extern "C" fn serial8250_stop_tx(port: *mut uart_port) {
let up: *mut uart_8250_port = up_to_u8250p(port);
let mut __UNIQUE_ID_guard_608: *mut uart_8250_port = class_serial8250_rpm_constructor(up);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_608) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_serial8250_rpm_destructor(__slate_cleanup_ptr_0 as *mut _));
__stop_tx(up);
// We really want to stop the transmitter from sending.
if (unsafe { (*port).r#type }) == 10 {
unsafe {
(*up).acr = (((*up).acr as i32) | 2) as u8;
}
serial_icr_write(up, 0i32, (unsafe { (*up).acr }) as i32);
}
}

fn __start_tx(port: *mut uart_port) {
let up: *mut uart_8250_port = up_to_u8250p(port);
let __v0: bool = !unsafe { (*up).dma }.is_null() && (unsafe { unsafe { (*(*up).dma).tx_dma }.unwrap()(up) }) == 0;
if __v0 {
return;
}
if serial8250_set_THRI(up) {
if (((unsafe { (*up).bugs }) as i32) as u64) & BIT!(1i32) != 0 {
let lsr: u16 = serial_lsr_in(up);
if (lsr as i32) & 32 != 0 {
serial8250_tx_chars(up);
}
}
}
// Re-enable the transmitter if we disabled it.
if (unsafe { (*port).r#type }) == 10 && ((unsafe { (*up).acr }) as i32) & 2 != 0 {
let __v3: u8 = (((unsafe { (*up).acr }) as i32) & !2) as u8;
unsafe {
(*up).acr = __v3;
}
serial_icr_write(up, 0i32, (unsafe { (*up).acr }) as i32);
}
}

/// serial8250_em485_start_tx() - generic ->rs485_start_tx() callback
/// @up: uart 8250 port
/// @toggle_ier: true to allow disabling receive interrupts
///
/// Generic callback usable by 8250 uart drivers to start rs485 transmission.
/// Assumes that setting the RTS bit in the MCR register means RTS is high.
/// (Some chips use inverse semantics.)  Further assumes that reception is
/// stoppable by disabling the UART_IER_RDI interrupt.  (Some chips set the
/// UART_LSR_DR bit even when UART_IER_RDI is disabled, foiling this approach.)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_em485_start_tx")]
extern "C" fn serial8250_em485_start_tx(up: *mut uart_8250_port, toggle_ier: bool) {
let mut mcr: u8 = serial8250_in_MCR(up) as u8;
if ((unsafe { (*up).port.rs485.flags }) as u64) & _BITUL!(4i32) == 0 && toggle_ier {
serial8250_stop_rx(unsafe { core::ptr::addr_of_mut!((*up).port) });
}
if ((unsafe { (*up).port.rs485.flags }) as u64) & _BITUL!(1i32) != 0 {
mcr = ((mcr as i32) | 2) as u8;
} else {
let __v3: u8 = ((mcr as i32) & !2) as u8;
mcr = __v3;
}
serial8250_out_MCR(up, mcr as i32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_em485_start_tx_609: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_8250_port, bool)>, *mut ()>(Some(serial8250_em485_start_tx)) };

/// Returns false, if start_tx_timer was setup to defer TX start
fn start_tx_rs485(port: *mut uart_port) -> bool {
let up: *mut uart_8250_port = up_to_u8250p(port);
let em485: *mut uart_8250_em485 = unsafe { (*up).em485 };
// While serial8250_em485_handle_stop_tx() is a noop if
// em485->active_timer != &em485->stop_tx_timer, it might happen that
// the timer is still armed and triggers only after the current bunch of
// chars is send and em485->active_timer == &em485->stop_tx_timer again.
// So cancel the timer. There is still a theoretical race condition if
// the timer is already running and only comes around to check for
// em485->active_timer when &em485->stop_tx_timer is armed again.
if (unsafe { (*em485).active_timer }) == unsafe { core::ptr::addr_of_mut!((*em485).stop_tx_timer) } {
unsafe { hrtimer_try_to_cancel(core::ptr::addr_of_mut!((*em485).stop_tx_timer)) };
}
unsafe {
(*em485).active_timer = core::ptr::null_mut::<hrtimer>();
}
if ((unsafe { (*em485).__slate_bits_0.__get_tx_stopped() }) as i32) != 0 {
unsafe {
(*em485).__slate_bits_0.__set_tx_stopped(0u32);
}
unsafe { unsafe { (*up).rs485_start_tx }.unwrap()(up, true) };
if (unsafe { (*up).port.rs485.delay_rts_before_send }) > 0 {
unsafe {
(*em485).active_timer = core::ptr::addr_of_mut!((*em485).start_tx_timer);
}
start_hrtimer_ms(unsafe { core::ptr::addr_of_mut!((*em485).start_tx_timer) }, (unsafe { (*up).port.rs485.delay_rts_before_send }) as u64);
return false;
}
}
true
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_em485_handle_start_tx")]
extern "C" fn serial8250_em485_handle_start_tx(t: *mut hrtimer) -> u32 {
let em485: *mut uart_8250_em485 = container_of__24!(t, uart_8250_em485);
let p: *mut uart_8250_port = unsafe { (*em485).port };
let mut __UNIQUE_ID_guard_610: __SlateRecord_serial_core_h_803 = class_uart_port_lock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!((*p).port) });
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_610) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_uart_port_lock_irqsave_destructor(__slate_cleanup_ptr_1 as *mut _));
if (unsafe { (*em485).active_timer }) == unsafe { core::ptr::addr_of_mut!((*em485).start_tx_timer) } {
__start_tx(unsafe { core::ptr::addr_of_mut!((*p).port) });
unsafe {
(*em485).active_timer = core::ptr::null_mut::<hrtimer>();
}
}
0
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_start_tx")]
extern "C" fn serial8250_start_tx(port: *mut uart_port) {
let up: *mut uart_8250_port = up_to_u8250p(port);
let em485: *mut uart_8250_em485 = unsafe { (*up).em485 };
// Port locked to synchronize UART_IER access against the console.
lockdep_assert_held_once!(unsafe { core::ptr::addr_of_mut!((*port).lock) });
let __v1: bool = (unsafe { (*port).x_char }) == 0 && kfifo_is_empty!(unsafe { core::ptr::addr_of_mut!((*(*port).state).port.xmit_fifo) });
if __v1 {
return;
}
serial8250_rpm_get_tx(up);
if !em485.is_null() {
let __v4: bool = (unsafe { (*em485).active_timer }) == unsafe { core::ptr::addr_of_mut!((*em485).start_tx_timer) } || !start_tx_rs485(port);
if __v4 {
return;
}
}
__start_tx(port);
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_throttle")]
extern "C" fn serial8250_throttle(port: *mut uart_port) {
unsafe { unsafe { (*port).throttle }.unwrap()(port) };
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_unthrottle")]
extern "C" fn serial8250_unthrottle(port: *mut uart_port) {
unsafe { unsafe { (*port).unthrottle }.unwrap()(port) };
}

fn serial8250_disable_ms(port: *mut uart_port) {
let up: *mut uart_8250_port = up_to_u8250p(port);
// Port locked to synchronize UART_IER access against the console.
lockdep_assert_held_once!(unsafe { core::ptr::addr_of_mut!((*port).lock) });
// no MSR capabilities
if (((unsafe { (*up).bugs }) as i32) as u64) & BIT!(2i32) != 0 {
return;
}
mctrl_gpio_disable_ms_no_sync(unsafe { (*up).gpios });
let __v3: u8 = (((unsafe { (*up).ier }) as i32) & !8) as u8;
unsafe {
(*up).ier = __v3;
}
serial_port_out(port, 1i32, (unsafe { (*up).ier }) as i32);
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_enable_ms")]
extern "C" fn serial8250_enable_ms(port: *mut uart_port) {
let up: *mut uart_8250_port = up_to_u8250p(port);
// Port locked to synchronize UART_IER access against the console.
lockdep_assert_held_once!(unsafe { core::ptr::addr_of_mut!((*port).lock) });
// no MSR capabilities
if (((unsafe { (*up).bugs }) as i32) as u64) & BIT!(2i32) != 0 {
return;
}
mctrl_gpio_enable_ms(unsafe { (*up).gpios });
unsafe {
(*up).ier = (((*up).ier as i32) | 8) as u8;
}
let mut __UNIQUE_ID_guard_611: *mut uart_8250_port = class_serial8250_rpm_constructor(up);
let __slate_cleanup_ptr_4 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_611) as *mut _;
let __slate_cleanup_4 = ::slate_support::Guard(&mut || class_serial8250_rpm_destructor(__slate_cleanup_ptr_4 as *mut _));
serial_port_out(port, 1i32, (unsafe { (*up).ier }) as i32);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_read_char")]
extern "C" fn serial8250_read_char(up: *mut uart_8250_port, mut lsr: u16) {
let port: *mut uart_port = unsafe { core::ptr::addr_of_mut!((*up).port) };
let mut ch: u8 = 0;
let mut flag: u8 = 0;
if likely__2!((lsr as i32) & 1) != 0 {
ch = serial_in(up, 0i32) as u8;
} else {
ch = 0;
}
// Intel 82571 has a Serial Over Lan device that will
// set UART_LSR_BI without setting UART_LSR_DR when
// it receives a break. To avoid reading from the
// receive buffer without UART_LSR_DR bit set, we
// just force the read character to be 0
let __v1: u32 = unsafe { (*port).icount.rx };
unsafe {
(*port).icount.rx = __v1.wrapping_add(1);
}
lsr = ((lsr as i32) | ((unsafe { (*up).lsr_saved_flags }) as i32)) as u16;
unsafe {
(*up).lsr_saved_flags = 0;
}
if unlikely__2!((lsr as i32) & (16 | 8 | 4 | 2)) != 0 {
if (lsr as i32) & 16 != 0 {
let __v6: u16 = ((lsr as i32) & !(8 | 4)) as u16;
lsr = __v6;
let __v8: u32 = unsafe { (*port).icount.brk };
unsafe {
(*port).icount.brk = __v8.wrapping_add(1);
}
// We do the SysRQ and SAK checking
// here because otherwise the break
// may get masked by ignore_status_mask
// or read_status_mask.
if uart_handle_break(port) != 0 {
return;
}
} else {
if (lsr as i32) & 4 != 0 {
let __v11: u32 = unsafe { (*port).icount.parity };
unsafe {
(*port).icount.parity = __v11.wrapping_add(1);
}
} else {
if (lsr as i32) & 8 != 0 {
let __v14: u32 = unsafe { (*port).icount.frame };
unsafe {
(*port).icount.frame = __v14.wrapping_add(1);
}
}
}
}
if (lsr as i32) & 2 != 0 {
let __v17: u32 = unsafe { (*port).icount.overrun };
unsafe {
(*port).icount.overrun = __v17.wrapping_add(1);
}
}
// Mask off conditions which should be ignored.
lsr = (((lsr as i32) as u32) & unsafe { (*port).read_status_mask }) as u16;
if (lsr as i32) & 16 != 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*port).dev as *const device, c"handling break\n".as_ptr() as *const u8) };
}
flag = 1;
} else {
if (lsr as i32) & 4 != 0 {
flag = 3;
} else {
if (lsr as i32) & 8 != 0 {
flag = 2;
}
}
}
}
if uart_prepare_sysrq_char(port, ch) != 0 {
return;
}
unsafe { uart_insert_char(port, lsr as u32, 2u32, ch, flag) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_read_char_612: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_8250_port, u16)>, *mut ()>(Some(serial8250_read_char)) };

/// serial8250_rx_chars - Read characters. The first LSR value must be passed in.
///
/// Returns LSR bits. The caller should rely only on non-Rx related LSR bits
/// (such as THRE) because the LSR value might come from an already consumed
/// character.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_rx_chars")]
extern "C" fn serial8250_rx_chars(up: *mut uart_8250_port, mut lsr: u16) -> u16 {
let port: *mut uart_port = unsafe { core::ptr::addr_of_mut!((*up).port) };
let mut max_count: i32 = 256;
loop {
serial8250_read_char(up, lsr);
let __v1: i32 = max_count.wrapping_sub(1);
max_count = __v1;
if __v1 == 0 {
break;
}
lsr = serial_in(up, 5i32) as u16;
if (lsr as i32) & (1 | 16) == 0 {
break;
}
}
unsafe { tty_flip_buffer_push(core::ptr::addr_of_mut!((*(*port).state).port)) };
lsr
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_rx_chars_613: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_8250_port, u16) -> u16>, *mut ()>(Some(serial8250_rx_chars)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_tx_chars")]
extern "C" fn serial8250_tx_chars(up: *mut uart_8250_port) {
let port: *mut uart_port = unsafe { core::ptr::addr_of_mut!((*up).port) };
let tport: *mut tty_port = unsafe { core::ptr::addr_of_mut!((*(*port).state).port) };
let mut count: i32 = 0;
if (unsafe { (*port).x_char }) != 0 {
unsafe { uart_xchar_out(port, 0i32) };
return;
}
if uart_tx_stopped(port) != 0 {
serial8250_stop_tx(port);
return;
}
if kfifo_is_empty!(unsafe { core::ptr::addr_of_mut!((*tport).xmit_fifo) }) {
__stop_tx(up);
return;
}
count = (unsafe { (*up).tx_loadsz }) as i32;
loop {
let mut c: u8 = 0;
if uart_fifo_get(port, core::ptr::addr_of_mut!(c)) == 0 {
break;
}
serial_out(up, 0i32, c as i32);
if (((unsafe { (*up).bugs }) as i32) as u64) & BIT!(5i32) != 0 {
// The Aspeed BMC virtual UARTs have a bug where data
// may get stuck in the BMC's Tx FIFO from bursts of
// writes on the APB interface.
//
// Delay back-to-back writes by a read cycle to avoid
// stalling the VUART. Read a register that won't have
// side-effects and discard the result.
serial_in(up, 7i32);
}
let __v3: bool = ((unsafe { (*up).capabilities }) as u64) & BIT!(14i32) != 0 && !uart_lsr_tx_empty(serial_in(up, 5i32) as u16);
if __v3 {
break;
}
// The BCM2835 MINI UART THRE bit is really a not-full bit.
let __v4: bool = ((unsafe { (*up).capabilities }) as u64) & BIT!(17i32) != 0 && serial_in(up, 5i32) & 32 == 0;
if __v4 {
break;
}
let __v6: i32 = count.wrapping_sub(1);
count = __v6;
if __v6 <= 0 {
break;
}
}
if kfifo_len!(unsafe { core::ptr::addr_of_mut!((*tport).xmit_fifo) }) < 256 {
unsafe { uart_write_wakeup(port) };
}
// With RPM enabled, we have to wait until the FIFO is empty before the
// HW can go idle. So we get here once again with empty FIFO and disable
// the interrupt and RPM in __stop_tx()
if kfifo_is_empty!(unsafe { core::ptr::addr_of_mut!((*tport).xmit_fifo) }) && ((unsafe { (*up).capabilities }) as u64) & BIT!(15i32) == 0 {
__stop_tx(up);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_tx_chars_614: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_8250_port)>, *mut ()>(Some(serial8250_tx_chars)) };

/// Caller holds uart port lock
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_modem_status")]
extern "C" fn serial8250_modem_status(up: *mut uart_8250_port) -> u32 {
let port: *mut uart_port = unsafe { core::ptr::addr_of_mut!((*up).port) };
let mut status: u32 = serial_in(up, 6i32) as u32;
status |= ((unsafe { (*up).msr_saved_flags }) as i32) as u32;
unsafe {
(*up).msr_saved_flags = 0;
}
if status & 15u32 != 0 && ((unsafe { (*up).ier }) as i32) & 8 != 0 && !unsafe { (*port).state }.is_null() {
if status & 4 != 0 {
let __v3: u32 = unsafe { (*port).icount.rng };
unsafe {
(*port).icount.rng = __v3.wrapping_add(1);
}
}
if status & 2 != 0 {
let __v6: u32 = unsafe { (*port).icount.dsr };
unsafe {
(*port).icount.dsr = __v6.wrapping_add(1);
}
}
if status & 8 != 0 {
unsafe { uart_handle_dcd_change(port, status & 128 != 0) };
}
if status & 1 != 0 {
unsafe { uart_handle_cts_change(port, status & 16 != 0) };
}
wake_up_interruptible!(unsafe { core::ptr::addr_of_mut!((*(*port).state).port.delta_msr_wait) });
}
status
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_modem_status_615: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_8250_port) -> u32>, *mut ()>(Some(serial8250_modem_status)) };

fn handle_rx_dma(up: *mut uart_8250_port, iir: u32) -> bool {
// Postpone DMA or not decision to IIR_RDI or IIR_RX_TIMEOUT
// because it's impossible to do an informed decision about
// that with IIR_THRI.
//
// This also fixes one known DMA Rx corruption issue where
// DR is asserted but DMA Rx only gets a corrupted zero byte
// (too early DR?).
match iir & 63 {
2 => {
return false;
// Postpone DMA or not decision to IIR_RDI or IIR_RX_TIMEOUT
// because it's impossible to do an informed decision about
// that with IIR_THRI.
//
// This also fixes one known DMA Rx corruption issue where
// DR is asserted but DMA Rx only gets a corrupted zero byte
// (too early DR?).
}
4 => {
if (unsafe { (*(*up).dma).rx_running }) != 0 {
unsafe { serial8250_rx_dma_flush(up) };
return true;
}
}
6 | 12 => {
unsafe { serial8250_rx_dma_flush(up) };
return true;
}
_ => {
}
}
(unsafe { unsafe { (*(*up).dma).rx_dma }.unwrap()(up) }) != 0
}

/// Context: port's lock must be held by the caller. The caller must
/// release it via guard(uart_port_lock_check_sysrq_irqsave) or
/// uart_unlock_and_check_sysrq_irqrestore(), which captures SysRq
/// character on unlock.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_handle_irq_locked")]
extern "C" fn serial8250_handle_irq_locked(port: *mut uart_port, iir: u32) {
let up: *mut uart_8250_port = up_to_u8250p(port);
let tport: *mut tty_port = unsafe { core::ptr::addr_of_mut!((*(*port).state).port) };
let mut skip_rx: bool = false;
let mut status: u16 = 0;
lockdep_assert_held_once!(unsafe { core::ptr::addr_of_mut!((*port).lock) });
status = serial_lsr_in(up);
// Recover from no-data-ready and FIFO error condition to avoid getting
// stuck in the ISR.
if (status as i32) & 1 == 0 && (status as i32) & 128 != 0 {
serial8250_clear_and_reinit_fifos(up);
}
// If port is stopped and there are no error conditions in the
// FIFO, then don't drain the FIFO, as this may lead to TTY buffer
// overflow. Not servicing, RX FIFO would trigger auto HW flow
// control when FIFO occupancy reaches preset threshold, thus
// halting RX. This only works when auto HW flow control is
// available.
if (status as i32) & (128 | (16 | 8 | 4 | 2)) == 0 && (unsafe { (*port).status }) & 12u32 != 0 && ((unsafe { (*up).ier }) as i32) & (4 | 1) == 0 {
skip_rx = true;
}
if (status as i32) & (1 | 16) != 0 && !skip_rx {
let d: *mut irq_data = unsafe { irq_get_irq_data((*port).irq) };
let __v1: bool = !d.is_null() && irqd_is_wakeup_set(d);
if __v1 {
pm_wakeup_event(unsafe { (*(*tport).tty).dev }, 0u32);
}
let __v2: bool = unsafe { (*up).dma }.is_null() || handle_rx_dma(up, iir);
if __v2 {
status = serial8250_rx_chars(up, status);
}
}
serial8250_modem_status(up);
if (status as i32) & 32 != 0 && ((unsafe { (*up).ier }) as i32) & 2 != 0 {
if unsafe { (*up).dma }.is_null() || (unsafe { (*(*up).dma).tx_err }) != 0 {
serial8250_tx_chars(up);
} else {
if (unsafe { (*(*up).dma).tx_running }) == 0 {
__stop_tx(up);
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_handle_irq_locked_616: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_port, u32)>, *mut ()>(Some(serial8250_handle_irq_locked)) };

/// This handles the interrupt from one port.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_handle_irq")]
extern "C" fn serial8250_handle_irq(port: *mut uart_port, iir: u32) -> i32 {
if iir & 1 != 0 {
return 0;
}
let mut __UNIQUE_ID_guard_617: __SlateRecord_serial_core_h_1308 = class_uart_port_lock_check_sysrq_irqsave_constructor(port);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_617) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_uart_port_lock_check_sysrq_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
serial8250_handle_irq_locked(port, iir);
1
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_handle_irq_618: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_port, u32) -> i32>, *mut ()>(Some(serial8250_handle_irq)) };

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_default_handle_irq")]
extern "C" fn serial8250_default_handle_irq(port: *mut uart_port) -> i32 {
let up: *mut uart_8250_port = up_to_u8250p(port);
let mut iir: u32 = 0;
let mut __UNIQUE_ID_guard_619: *mut uart_8250_port = class_serial8250_rpm_constructor(up);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_619) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_serial8250_rpm_destructor(__slate_cleanup_ptr_0 as *mut _));
iir = serial_port_in(port, 2i32) as u32;
serial8250_handle_irq(port, iir)
}

/// Newer 16550 compatible parts such as the SC16C650 & Altera 16550 Soft IP
/// have a programmable TX threshold that triggers the THRE interrupt in
/// the IIR register. In this case, the THRE interrupt indicates the FIFO
/// has space available. Load it up with tx_loadsz bytes.
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_tx_threshold_handle_irq")]
extern "C" fn serial8250_tx_threshold_handle_irq(port: *mut uart_port) -> i32 {
let mut iir: u32 = serial_port_in(port, 2i32) as u32;
// TX Threshold IRQ triggered so load up FIFO
if iir & 14 == 2 {
let up: *mut uart_8250_port = up_to_u8250p(port);
let mut __UNIQUE_ID_guard_620: __SlateRecord_serial_core_h_803 = class_uart_port_lock_irqsave_constructor(port);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_620) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_uart_port_lock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
serial8250_tx_chars(up);
}
iir = serial_port_in(port, 2i32) as u32;
serial8250_handle_irq(port, iir)
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_tx_empty")]
extern "C" fn serial8250_tx_empty(port: *mut uart_port) -> u32 {
let up: *mut uart_8250_port = up_to_u8250p(port);
let mut __UNIQUE_ID_guard_621: *mut uart_8250_port = class_serial8250_rpm_constructor(up);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_621) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_serial8250_rpm_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_guard_622: __SlateRecord_serial_core_h_803 = class_uart_port_lock_irqsave_constructor(port);
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_622) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_uart_port_lock_irqsave_destructor(__slate_cleanup_ptr_1 as *mut _));
let __v2: bool = !serial8250_tx_dma_running(up) && uart_lsr_tx_empty(serial_lsr_in(up));
if __v2 {
return 1;
}
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_do_get_mctrl")]
extern "C" fn serial8250_do_get_mctrl(port: *mut uart_port) -> u32 {
let mut scope: core::mem::MaybeUninit<*mut uart_8250_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut uart_8250_port = core::ptr::addr_of_mut!(scope) as *mut *mut uart_8250_port;
let mut val: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u32 = core::ptr::addr_of_mut!(val) as *mut u32;
let mut status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u32 = core::ptr::addr_of_mut!(status) as *mut u32;
let mut up: *mut uart_8250_port = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { class_serial8250_rpm_destructor(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
up = up_to_u8250p(port);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
core::ptr::write(__slate_slot_0, class_serial8250_rpm_constructor(up));
__slate_cleanup_armed_0.set(true);
if !class_serial8250_rpm_lock_ptr(&(*core::ptr::addr_of_mut!(*__slate_slot_0))).is_null() || !class_serial8250_rpm_is_conditional {
if true {
*__slate_slot_2 = serial8250_modem_status(up);
}
}
__slate_cleanup_armed_0.set(false);
class_serial8250_rpm_destructor(__slate_slot_0 as *mut _);
*__slate_slot_1 = serial8250_MSR_to_TIOCM(*__slate_slot_2 as i32) as u32;
if !unsafe { (*up).gpios }.is_null() {
return mctrl_gpio_get((*up).gpios, &(*core::ptr::addr_of_mut!(*__slate_slot_1)));
} else {
return *__slate_slot_1;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_do_get_mctrl_624: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_port) -> u32>, *mut ()>(Some(serial8250_do_get_mctrl)) };

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_get_mctrl")]
extern "C" fn serial8250_get_mctrl(port: *mut uart_port) -> u32 {
if (unsafe { (*port).get_mctrl }) != None {
return unsafe { unsafe { (*port).get_mctrl }.unwrap()(port) };
}
serial8250_do_get_mctrl(port)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_do_set_mctrl")]
extern "C" fn serial8250_do_set_mctrl(port: *mut uart_port, mctrl: u32) {
let up: *mut uart_8250_port = up_to_u8250p(port);
let mut mcr: u8 = serial8250_TIOCM_to_MCR(mctrl as i32) as u8;
mcr = ((mcr as i32) | ((unsafe { (*up).mcr }) as i32)) as u8;
serial8250_out_MCR(up, mcr as i32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_do_set_mctrl_625: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_port, u32)>, *mut ()>(Some(serial8250_do_set_mctrl)) };

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_set_mctrl")]
extern "C" fn serial8250_set_mctrl(port: *mut uart_port, mctrl: u32) {
if ((unsafe { (*port).rs485.flags }) as u64) & _BITUL!(0i32) != 0 {
return;
}
if (unsafe { (*port).set_mctrl }) != None {
unsafe { unsafe { (*port).set_mctrl }.unwrap()(port, mctrl) };
} else {
serial8250_do_set_mctrl(port, mctrl);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_do_break_ctl")]
extern "C" fn serial8250_do_break_ctl(port: *mut uart_port, break_state: i32) {
let up: *mut uart_8250_port = up_to_u8250p(port);
let mut __UNIQUE_ID_guard_626: *mut uart_8250_port = class_serial8250_rpm_constructor(up);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_626) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_serial8250_rpm_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_guard_627: __SlateRecord_serial_core_h_803 = class_uart_port_lock_irqsave_constructor(port);
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_627) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_uart_port_lock_irqsave_destructor(__slate_cleanup_ptr_1 as *mut _));
if break_state == 1i32.wrapping_neg() {
unsafe {
(*up).lcr = (((*up).lcr as i32) | 64) as u8;
}
} else {
let __v7: u8 = (((unsafe { (*up).lcr }) as i32) & !64) as u8;
unsafe {
(*up).lcr = __v7;
}
}
serial_port_out(port, 3i32, (unsafe { (*up).lcr }) as i32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_do_break_ctl_628: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_port, i32)>, *mut ()>(Some(serial8250_do_break_ctl)) };

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_break_ctl")]
extern "C" fn serial8250_break_ctl(port: *mut uart_port, break_state: i32) {
if (unsafe { (*port).break_ctl }) != None {
unsafe { unsafe { (*port).break_ctl }.unwrap()(port, break_state) };
} else {
serial8250_do_break_ctl(port, break_state);
}
}

/// Returns true if @bits were set, false on timeout
fn wait_for_lsr(up: *mut uart_8250_port, bits: i32) -> bool {
let mut status: u32 = 0;
let mut tmout: u32 = 0;
// Wait for a character to be sent. Fallback to a safe default
// timeout value if @frame_time is not available.
if (unsafe { (*up).port.frame_time }) != 0 {
tmout = ((unsafe { (*up).port.frame_time }.wrapping_mul(2) as i64) / 1000) as u32;
} else {
tmout = 10000;
}
loop {
status = serial_lsr_in(up) as u32;
if status & (bits as u32) == (bits as u32) {
break;
}
let __v2: u32 = tmout.wrapping_sub(1);
tmout = __v2;
if __v2 == 0 {
break;
}
udelay(1u64);
touch_nmi_watchdog();
}
tmout != 0
}

/// Wait for transmitter and holding register to empty with timeout
fn wait_for_xmitr(up: *mut uart_8250_port, bits: i32) {
let mut tmout: u32 = 0;
let mut tx_ready: bool = false;
tx_ready = wait_for_lsr(up, bits);
// Wait up to 1s for flow control if necessary.
// When 'no_console_suspend' is active (in the window between
// suspend() and resume()), flow control is temporarily ignored
// because the canary workaround is not reliable in all situations,
// leading to flow control timeouts for every character.
if uart_console_hwflow_active(unsafe { core::ptr::addr_of_mut!((*up).port) }) && (unsafe { (*up).canary }) == 0 {
tmout = 1000000;
while tmout != 0 {
let msr: u32 = serial_in(up, 6i32) as u32;
unsafe {
(*up).msr_saved_flags = ((((*up).msr_saved_flags as i32) as u32) | msr & 15u32) as u8;
}
if msr & 16 != 0 {
if !tx_ready {
wait_for_lsr(up, bits);
}
break;
}
udelay(1u64);
touch_nmi_watchdog();
tmout = tmout.wrapping_sub(1);
}
}
}

fn serial8250_startup_special(port: *mut uart_port) {
let mut scope: core::mem::MaybeUninit<__SlateRecord_serial_core_h_803> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut __SlateRecord_serial_core_h_803 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_serial_core_h_803;
// Wake up and initialize UART
//
// Synchronize UART_IER access against the console.
let mut __UNIQUE_ID_guard_629: core::mem::MaybeUninit<__SlateRecord_serial_core_h_803> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_serial_core_h_803 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_629) as *mut __SlateRecord_serial_core_h_803;
let mut up: *mut uart_8250_port = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_uart_port_lock_irqsave_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { class_uart_port_lock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
up = up_to_u8250p(port);
let __t0: u32 = (*port).r#type;
if __t0 == 10 {
core::ptr::write(__slate_slot_1, class_uart_port_lock_irqsave_constructor(port));
__slate_cleanup_armed_1.set(true);
(*up).acr = 0;
serial_port_out(port, 3i32, 191i32);
serial_port_out(port, 2i32, 16i32);
serial_port_out(port, 1i32, 0i32);
serial_port_out(port, 3i32, 0i32);
serial_icr_write(up, 12i32, 0i32); // Reset the UART
serial_port_out(port, 3i32, 191i32);
serial_port_out(port, 2i32, 16i32);
serial_port_out(port, 3i32, 0i32);
__slate_cleanup_armed_1.set(false);
class_uart_port_lock_irqsave_destructor(__slate_slot_1 as *mut _);
} else {
if __t0 == 95 {
core::ptr::write(__slate_slot_0, class_uart_port_lock_irqsave_constructor(port));
__slate_cleanup_armed_0.set(true);
if !class_uart_port_lock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_0))).is_null() || !class_uart_port_lock_irqsave_is_conditional {
if true {
serial_port_out(port, 1i32, 0i32);
serial_port_out(port, 12i32, 0i32);
}
}
__slate_cleanup_armed_0.set(false);
class_uart_port_lock_irqsave_destructor(__slate_slot_0 as *mut _);
// Reset the port
//
// Synchronize UART_IER access against the console.
mdelay!(10i32);
// Enable Tx, Rx and free run mode
serial_port_out(port, 12i32, 24577i32);
} else {
if __t0 == 13 {
rsa_enable(up);
}
}
}
}
}

fn serial8250_set_TRG_levels(port: *mut uart_port) {
let up: *mut uart_8250_port = up_to_u8250p(port);
'__slate_break_2: {
match unsafe { (*port).r#type } {
12 => {
// For a XR16C850, we need to set the trigger levels
let mut fctr: u8 = 0;
serial_out(up, 3i32, 191i32);
fctr = (serial_in(up, 1i32) & !(0 | 128)) as u8;
fctr = ((fctr as i32) | 48) as u8;
serial_port_out(port, 1i32, (fctr as i32) | 0);
serial_port_out(port, 0i32, 96i32);
serial_port_out(port, 1i32, (fctr as i32) | 128);
serial_port_out(port, 0i32, 96i32);
serial_port_out(port, 3i32, 0i32);
break '__slate_break_2;
// For the Altera 16550 variants, set TX threshold trigger level.
}
26..=28 => {
if (unsafe { (*port).fifosize }) <= 1 {
return;
}
// Bounds checking of TX threshold (valid 0 to fifosize-2)
if (unsafe { (*up).tx_loadsz }) < 2 || (unsafe { (*up).tx_loadsz }) > unsafe { (*port).fifosize } {
while false {
}
unsafe { _dev_err((*port).dev as *const device, c"TX FIFO Threshold errors, skipping\n".as_ptr() as *const u8) };
return;
}
serial_port_out(port, 64i32, 1i32);
serial_port_out(port, 65i32, unsafe { (*port).fifosize }.wrapping_sub(unsafe { (*up).tx_loadsz }) as i32);
unsafe {
(*port).handle_irq = Some(serial8250_tx_threshold_handle_irq);
}
}
_ => {
}
}
}
}

fn serial8250_THRE_test(port: *mut uart_port) {
// Test for UARTs that do not reassert THRE when the transmitter is idle and the interrupt
// has already been cleared.  Real 16550s should always reassert this interrupt whenever the
// transmitter is idle and the interrupt is enabled.  Delays are necessary to allow register
// changes to become visible.
//
// Synchronize UART_IER access against the console.
let mut scope: core::mem::MaybeUninit<__SlateRecord_serial_core_h_803> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut __SlateRecord_serial_core_h_803 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_serial_core_h_803;
let mut iir_noint2: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut bool = core::ptr::addr_of_mut!(iir_noint2) as *mut bool;
let mut iir_noint1: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut bool = core::ptr::addr_of_mut!(iir_noint1) as *mut bool;
let mut up: *mut uart_8250_port = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { class_uart_port_lock_irqsave_destructor(__slate_cleanup_ptr_3 as *mut _) };
}
});
unsafe {
up = up_to_u8250p(port);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
if (*port).irq == 0 {
return;
} else {
if (*up).port.flags & BIT_ULL!(19i32) != 0 {
return;
} else {
disable_irq((*port).irq);
core::ptr::write(__slate_slot_3, class_uart_port_lock_irqsave_constructor(port));
__slate_cleanup_armed_3.set(true);
if !class_uart_port_lock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_3))).is_null() || !class_uart_port_lock_irqsave_is_conditional {
if true {
wait_for_xmitr(up, 32i32);
serial_port_out_sync(port, 1i32, 2i32);
udelay(1u64); // allow THRE to set
*__slate_slot_5 = serial_port_in(port, 2i32) & 1 != 0;
serial_port_out(port, 1i32, 0i32);
serial_port_out_sync(port, 1i32, 2i32);
udelay(1u64); // allow a working UART time to re-assert THRE
*__slate_slot_4 = serial_port_in(port, 2i32) & 1 != 0;
serial_port_out(port, 1i32, 0i32);
}
}
__slate_cleanup_armed_3.set(false);
class_uart_port_lock_irqsave_destructor(__slate_slot_3 as *mut _);
enable_irq((*port).irq);
// If the interrupt is not reasserted, or we otherwise don't trust the iir, setup a timer to
// kick the UART on a regular basis.
if !(*__slate_slot_5) && *__slate_slot_4 || (*up).port.flags & BIT_ULL!(26i32) != 0 {
let __v1: u16 = (*up).bugs;
let __v0: u16 = (((__v1 as i32) as u64) | BIT!(3i32)) as u16;
(*up).bugs = __v0;
}
}
}
}
}

fn serial8250_init_mctrl(port: *mut uart_port) {
if (unsafe { (*port).flags }) & 2u64 != 0 {
if (unsafe { (*port).irq }) == 0 {
unsafe {
(*port).mctrl = (*port).mctrl | 8192;
}
}
} else {
// Most PC uarts need OUT2 raised to enable interrupts.
if (unsafe { (*port).irq }) != 0 {
unsafe {
(*port).mctrl = (*port).mctrl | 16384;
}
}
}
serial8250_set_mctrl(port, unsafe { (*port).mctrl });
}

fn serial8250_iir_txen_test(port: *mut uart_port) {
let up: *mut uart_8250_port = up_to_u8250p(port);
let mut lsr_temt: bool = false;
let mut iir_noint: bool = false;
if (((unsafe { (*port).quirks }) as i32) as u64) & BIT!(0i32) != 0 {
return;
}
// Do a quick test to see if we receive an interrupt when we enable the TX irq.
serial_port_out(port, 1i32, 2i32);
lsr_temt = serial_port_in(port, 5i32) & 64 != 0;
iir_noint = serial_port_in(port, 2i32) & 1 != 0;
serial_port_out(port, 1i32, 0i32);
// Serial over Lan (SoL) hack:
// Intel 8257x Gigabit ethernet chips have a 16550 emulation, to be used for Serial Over
// Lan.  Those chips take a longer time than a normal serial device to signalize that a
// transmission data was queued. Due to that, the above test generally fails. One solution
// would be to delay the reading of iir. However, this is not reliable, since the timeout is
// variable. So, in case of UPQ_NO_TXEN_TEST, let's just don't test if we receive TX irq.
// This way, we'll never enable UART_BUG_TXEN.
if lsr_temt && iir_noint {
if (((unsafe { (*up).bugs }) as i32) as u64) & BIT!(1i32) == 0 {
let __v1: u16 = unsafe { (*up).bugs };
let __v2: u16 = (((__v1 as i32) as u64) | BIT!(1i32)) as u16;
unsafe {
(*up).bugs = __v2;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*port).dev as *const device, c"enabling bad tx status workarounds\n".as_ptr() as *const u8) };
}
}
return;
}
// FIXME: why is this needed?
let __v4: u16 = unsafe { (*up).bugs };
let __v5: u16 = (((__v4 as i32) as u64) & !BIT!(1i32)) as u16;
unsafe {
(*up).bugs = __v5;
}
}

fn serial8250_initialize(port: *mut uart_port) {
let mut __UNIQUE_ID_guard_632: __SlateRecord_serial_core_h_803 = class_uart_port_lock_irqsave_constructor(port);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_632) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_uart_port_lock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
serial_port_out(port, 3i32, 3i32);
serial8250_init_mctrl(port);
serial8250_iir_txen_test(port);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_do_startup")]
extern "C" fn serial8250_do_startup(port: *mut uart_port) -> i32 {
let up: *mut uart_8250_port = up_to_u8250p(port);
let mut retval: i32 = 0;
if (unsafe { (*port).fifosize }) == 0 {
unsafe {
(*port).fifosize = uart_config.0[(*port).r#type as usize].fifo_size as u32;
}
}
if (unsafe { (*up).tx_loadsz }) == 0 {
unsafe {
(*up).tx_loadsz = uart_config.0[(*port).r#type as usize].tx_loadsz as u32;
}
}
if (unsafe { (*up).capabilities }) == 0 {
unsafe {
(*up).capabilities = uart_config.0[(*port).r#type as usize].flags;
}
}
unsafe {
(*up).mcr = 0;
}
if ((unsafe { (*port).iotype }) as i32) != ((unsafe { (*up).cur_iotype }) as i32) {
set_io_from_upio(port);
}
let mut __UNIQUE_ID_guard_633: *mut uart_8250_port = class_serial8250_rpm_constructor(up);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_633) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_serial8250_rpm_destructor(__slate_cleanup_ptr_0 as *mut _));
serial8250_startup_special(port);
// Clear the FIFO buffers and disable them.
// (they will be reenabled in set_termios())
serial8250_clear_fifos(up);
serial8250_clear_interrupts(port);
// At this point, there's no way the LSR could still be 0xff;
// if it is, then bail out, because there's likely no UART
// here.
let __v1: bool = (unsafe { (*port).flags }) & 16384u64 == 0 && serial_port_in(port, 5i32) == 255;
if __v1 {
loop {
if __ratelimit__5!(unsafe { core::ptr::addr_of_mut!(_rs__2340) }) != 0 {
while false {
}
unsafe { _dev_info((*port).dev as *const device, c"LSR safety check engaged!\n".as_ptr() as *const u8) };
}
if true {
break;
}
}
return 19i32.wrapping_neg();
}
serial8250_set_TRG_levels(port);
// Check if we need to have shared IRQs
if (unsafe { (*port).irq }) != 0 && (unsafe { (*up).port.flags }) & BIT_ULL!(24i32) != 0 {
unsafe {
(*up).port.irqflags = (*up).port.irqflags | 128;
}
}
retval = unsafe { unsafe { (*(*up).ops).setup_irq }.unwrap()(up) };
if retval != 0 {
return retval;
}
serial8250_THRE_test(port);
unsafe { unsafe { (*(*up).ops).setup_timer }.unwrap()(up) };
serial8250_initialize(port);
// Clear the interrupt registers again for luck, and clear the
// saved flags to avoid getting false values from polling
// routines or the previous session.
serial8250_clear_interrupts(port);
unsafe {
(*up).lsr_saved_flags = 0;
}
unsafe {
(*up).msr_saved_flags = 0;
}
// Request DMA channels for both RX and TX.
if !unsafe { (*up).dma }.is_null() {
let mut msg: *const u8 = core::ptr::null::<u8>();
if uart_console!(port) {
msg = c"forbid DMA for kernel console".as_ptr() as *const u8;
} else {
if (unsafe { serial8250_request_dma(up) }) != 0 {
msg = c"failed to request DMA".as_ptr() as *const u8;
}
}
if !msg.is_null() {
loop {
if __ratelimit__5!(unsafe { core::ptr::addr_of_mut!(_rs__2380) }) != 0 {
while false {
}
unsafe { _dev_warn((*port).dev as *const device, c"%s\n".as_ptr() as *const u8, msg) };
}
if true {
break;
}
}
unsafe {
(*up).dma = core::ptr::null_mut::<uart_8250_dma>();
}
}
}
// Set the IER shadow for rx interrupts but defer actual interrupt
// enable until after the FIFOs are enabled; otherwise, an already-
// active sender can swamp the interrupt handler with "too much work".
unsafe {
(*up).ier = 5u8;
}
if (unsafe { (*port).flags }) & 2u64 != 0 {
let mut icp: u32 = 0;
// Enable interrupts on the AST Fourport board
icp = ((unsafe { (*port).iobase }) & 4064 | 31) as u32;
outb_p(128u8, icp as u16);
inb_p(icp as u16);
}
0
}

static mut _rs__2340: ratelimit_state = {
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

static mut _rs__2380: ratelimit_state = {
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

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_do_startup_634: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_port) -> i32>, *mut ()>(Some(serial8250_do_startup)) };

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_startup")]
extern "C" fn serial8250_startup(port: *mut uart_port) -> i32 {
if (unsafe { (*port).startup }) != None {
return unsafe { unsafe { (*port).startup }.unwrap()(port) };
}
serial8250_do_startup(port)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_do_shutdown")]
extern "C" fn serial8250_do_shutdown(port: *mut uart_port) {
let mut scope: core::mem::MaybeUninit<__SlateRecord_serial_core_h_803> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut __SlateRecord_serial_core_h_803 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_serial_core_h_803;
// Disable interrupts from this port
//
// Synchronize UART_IER access against the console.
let mut scope: core::mem::MaybeUninit<__SlateRecord_serial_core_h_803> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut __SlateRecord_serial_core_h_803 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_serial_core_h_803;
let mut lcr: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u32 = core::ptr::addr_of_mut!(lcr) as *mut u32;
let mut up: *mut uart_8250_port = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_9 = __slate_slot_9;
let __slate_cleanup_armed_9 = core::cell::Cell::new(false);
let __slate_cleanup_9 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_9.get() {
unsafe { class_uart_port_lock_irqsave_destructor(__slate_cleanup_ptr_9 as *mut _) };
}
});
let __slate_cleanup_ptr_8 = __slate_slot_8;
let __slate_cleanup_armed_8 = core::cell::Cell::new(false);
let __slate_cleanup_8 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_8.get() {
unsafe { class_uart_port_lock_irqsave_destructor(__slate_cleanup_ptr_8 as *mut _) };
}
});
unsafe {
up = up_to_u8250p(port);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
serial8250_rpm_get(up);
core::ptr::write(__slate_slot_9, class_uart_port_lock_irqsave_constructor(port));
__slate_cleanup_armed_9.set(true);
if !class_uart_port_lock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_9))).is_null() || !class_uart_port_lock_irqsave_is_conditional {
if true {
(*up).ier = 0;
serial_port_out(port, 1i32, 0i32);
}
}
__slate_cleanup_armed_9.set(false);
class_uart_port_lock_irqsave_destructor(__slate_slot_9 as *mut _);
synchronize_irq((*port).irq);
serial8250_release_dma(up);
(*up).dma = core::ptr::null_mut::<uart_8250_dma>();
core::ptr::write(__slate_slot_8, class_uart_port_lock_irqsave_constructor(port));
__slate_cleanup_armed_8.set(true);
if !class_uart_port_lock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_8))).is_null() || !class_uart_port_lock_irqsave_is_conditional {
if true {
if (*port).flags & 2u64 != 0 {
// reset interrupts on the AST Fourport board
__inb(((*port).iobase & 4064 | 31) as u16);
(*port).mctrl = (*port).mctrl | 8192;
} else {
let __v5: u32 = (*port).mctrl & 4294950911u32;
(*port).mctrl = __v5;
}
serial8250_set_mctrl(port, (*port).mctrl);
// Disable break condition
*__slate_slot_10 = serial_port_in(port, 3i32) as u32;
let __v0: u32 = *__slate_slot_10 & 4294967231u32;
*__slate_slot_10 = __v0;
serial_port_out(port, 3i32, *__slate_slot_10 as i32);
}
}
__slate_cleanup_armed_8.set(false);
class_uart_port_lock_irqsave_destructor(__slate_slot_8 as *mut _);
serial8250_clear_fifos(up);
rsa_disable(up);
// Read data port to reset things, and then unlink from
// the IRQ chain.
serial_port_in(port, 0i32);
// LCR writes on DW UART can trigger late (unmaskable) IRQs.
// Handle them before releasing the handler.
synchronize_irq((*port).irq);
serial8250_rpm_put(up);
unsafe { (*(*up).ops).release_irq }.unwrap()(up);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_do_shutdown_637: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_port)>, *mut ()>(Some(serial8250_do_shutdown)) };

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_shutdown")]
extern "C" fn serial8250_shutdown(port: *mut uart_port) {
if (unsafe { (*port).shutdown }) != None {
unsafe { unsafe { (*port).shutdown }.unwrap()(port) };
} else {
serial8250_do_shutdown(port);
}
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_flush_buffer")]
extern "C" fn serial8250_flush_buffer(port: *mut uart_port) {
let up: *mut uart_8250_port = up_to_u8250p(port);
if !unsafe { (*up).dma }.is_null() {
unsafe { serial8250_tx_dma_flush(up) };
}
}

fn serial8250_do_get_divisor(port: *mut uart_port, baud: u32) -> u32 {
let magic_multiplier: u64 = (unsafe { (*port).flags }) & 65536u64;
let up: *mut uart_8250_port = up_to_u8250p(port);
let mut quot: u32 = 0;
// Handle magic divisors for baud rates above baud_base on SMSC
// Super I/O chips.  We clamp custom rates from clk/6 and clk/12
// up to clk/4 (0x8001) and clk/8 (0x8002) respectively.  These
// magic divisors actually reprogram the baud rate generator's
// reference clock derived from chips's 14.318MHz clock input.
//
// Documentation claims that with these magic divisors the base
// frequencies of 7.3728MHz and 3.6864MHz are used respectively
// for the extra baud rates of 460800bps and 230400bps rather
// than the usual base frequency of 1.8462MHz.  However empirical
// evidence contradicts that.
//
// Instead bit 7 of the DLM register (bit 15 of the divisor) is
// effectively used as a clock prescaler selection bit for the
// base frequency of 7.3728MHz, always used.  If set to 0, then
// the base frequency is divided by 4 for use by the Baud Rate
// Generator, for the usual arrangement where the value of 1 of
// the divisor produces the baud rate of 115200bps.  Conversely,
// if set to 1 and high-speed operation has been enabled with the
// Serial Port Mode Register in the Device Configuration Space,
// then the base frequency is supplied directly to the Baud Rate
// Generator, so for the divisor values of 0x8001, 0x8002, 0x8003,
// 0x8004, etc. the respective baud rates produced are 460800bps,
// 230400bps, 153600bps, 115200bps, etc.
//
// In all cases only low 15 bits of the divisor are used to divide
// the baud base and therefore 32767 is the maximum divisor value
// possible, even though documentation says that the programmable
// Baud Rate Generator is capable of dividing the internal PLL
// clock by any divisor from 1 to 65535.
if magic_multiplier != 0 && baud >= (unsafe { (*port).uartclk }) / 6 {
quot = 32769;
} else {
if magic_multiplier != 0 && baud >= (unsafe { (*port).uartclk }) / 12 {
quot = 32770;
} else {
quot = unsafe { uart_get_divisor(port, baud) };
}
}
// Oxford Semi 952 rev B workaround
if (((unsafe { (*up).bugs }) as i32) as u64) & BIT!(0i32) != 0 && quot & 255 == 0 {
quot = quot.wrapping_add(1);
}
quot
}

fn serial8250_get_divisor(port: *mut uart_port, baud: u32, frac: *mut u32) -> u32 {
if (unsafe { (*port).get_divisor }) != None {
return unsafe { unsafe { (*port).get_divisor }.unwrap()(port, baud, frac) };
}
serial8250_do_get_divisor(port, baud)
}

fn serial8250_compute_lcr(up: *mut uart_8250_port, c_cflag: u32) -> u8 {
let mut lcr: u8 = UART_LCR_WLEN!(unsafe { tty_get_char_size(c_cflag) }) as u8;
if c_cflag & 64 != 0 {
lcr = ((lcr as i32) | 4) as u8;
}
if c_cflag & 256 != 0 {
lcr = ((lcr as i32) | 8) as u8;
}
if c_cflag & 512 == 0 {
lcr = ((lcr as i32) | 16) as u8;
}
if c_cflag & 1073741824 != 0 {
lcr = ((lcr as i32) | 32) as u8;
}
lcr
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_do_set_divisor")]
extern "C" fn serial8250_do_set_divisor(port: *mut uart_port, baud: u32, mut quot: u32) {
let up: *mut uart_8250_port = up_to_u8250p(port);
// Workaround to enable 115200 baud on OMAP1510 internal ports
if is_omap1510_8250(up) != 0 {
if baud == 115200 {
quot = 1;
serial_port_out(port, 19i32, 1i32);
} else {
serial_port_out(port, 19i32, 0i32);
}
}
// For NatSemi, switch to bank 2 not bank 1, to avoid resetting EXCR2,
// otherwise just set DLAB
if (unsafe { (*up).capabilities }) & 8 != 0 {
serial_port_out(port, 3i32, 224i32);
} else {
serial_port_out(port, 3i32, ((unsafe { (*up).lcr }) as i32) | 128);
}
serial_dl_write(up, quot);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_do_set_divisor_638: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_port, u32, u32)>, *mut ()>(Some(serial8250_do_set_divisor)) };

fn serial8250_set_divisor(port: *mut uart_port, baud: u32, quot: u32, quot_frac: u32) {
if (unsafe { (*port).set_divisor }) != None {
unsafe { unsafe { (*port).set_divisor }.unwrap()(port, baud, quot, quot_frac) };
} else {
serial8250_do_set_divisor(port, baud, quot);
}
}

fn serial8250_get_baud_rate(port: *mut uart_port, termios: *mut ktermios, old: *const ktermios) -> u32 {
let tolerance: u32 = (unsafe { (*port).uartclk }) / 100;
let mut min: u32 = 0;
let mut max: u32 = 0;
// Handle magic divisors for baud rates above baud_base on SMSC
// Super I/O chips.  Enable custom rates of clk/4 and clk/8, but
// disable divisor values beyond 32767, which are unavailable.
if (unsafe { (*port).flags }) & 65536u64 != 0 {
min = (unsafe { (*port).uartclk }) / 16 / 65535 >> 1i32;
max = unsafe { (*port).uartclk }.wrapping_add(tolerance) / 4;
} else {
min = (unsafe { (*port).uartclk }) / 16 / 65535;
max = unsafe { (*port).uartclk }.wrapping_add(tolerance) / 16;
}
// Ask the core to calculate the divisor for us.
// Allow 1% tolerance at the upper limit so uart clks marginally
// slower than nominal still match standard baud rates without
// causing transmission errors.
unsafe { uart_get_baud_rate(port, termios, old, min, max) }
}

/// Note in order to avoid the tty port mutex deadlock don't use the next method
/// within the uart port callbacks. Primarily it's supposed to be utilized to
/// handle a sudden reference clock rate change.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_update_uartclk")]
extern "C" fn serial8250_update_uartclk(port: *mut uart_port, uartclk: u32) {
let mut __UNIQUE_ID_unlock_645: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_645) as *mut *mut __SlateRecord_mutex_h_253;
let mut __UNIQUE_ID_guard_644: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_644) as *mut __SlateRecord_mutex_h_253;
let mut __UNIQUE_ID_unlock_643: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_643) as *mut *mut __SlateRecord_mutex_h_253;
let mut __UNIQUE_ID_guard_642: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_642) as *mut __SlateRecord_mutex_h_253;
let mut __UNIQUE_ID_unlock_641: core::mem::MaybeUninit<*mut __SlateRecord_rwsem_h_270> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_641) as *mut *mut __SlateRecord_rwsem_h_270;
let mut __UNIQUE_ID_guard_640: core::mem::MaybeUninit<__SlateRecord_rwsem_h_270> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_640) as *mut __SlateRecord_rwsem_h_270;
let mut tty: *mut tty_struct = unsafe { core::mem::zeroed() };
let mut scope: core::mem::MaybeUninit<__SlateRecord_tty_port_h_271> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut __SlateRecord_tty_port_h_271 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_tty_port_h_271;
let mut tport: *mut tty_port = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_7 = __slate_slot_7;
let __slate_cleanup_armed_7 = core::cell::Cell::new(false);
let __slate_cleanup_7 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_7.get() {
unsafe { class_tty_port_tty_destructor(__slate_cleanup_ptr_7 as *mut _) };
}
});
let __slate_cleanup_ptr_5 = __slate_slot_5;
let __slate_cleanup_armed_5 = core::cell::Cell::new(false);
let __slate_cleanup_5 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_5.get() {
unsafe { class_rwsem_write_destructor(__slate_cleanup_ptr_5 as *mut _) };
}
});
let __slate_cleanup_ptr_4 = __slate_slot_4;
let __slate_cleanup_armed_4 = core::cell::Cell::new(false);
let __slate_cleanup_4 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_4.get() {
unsafe { __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_4 as *mut _) };
}
});
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
tport = core::ptr::addr_of_mut!((*(*port).state).port);
core::ptr::write(__slate_slot_7, class_tty_port_tty_constructor(tport));
__slate_cleanup_armed_7.set(true);
if !class_tty_port_tty_lock_ptr(core::ptr::addr_of_mut!(*__slate_slot_7)).is_null() || !class_tty_port_tty_is_conditional {
if true {
tty = scoped_tty!(*__slate_slot_7);
core::ptr::write(__slate_slot_5, class_rwsem_write_constructor(core::ptr::addr_of_mut!((*tty).termios_rwsem)));
__slate_cleanup_armed_5.set(true);
core::ptr::write(__slate_slot_4, (core::ptr::addr_of_mut!((*tty).termios_rwsem) as u64) as *mut __SlateRecord_rwsem_h_270);
__slate_cleanup_armed_4.set(true);
core::ptr::write(__slate_slot_3, class_mutex_constructor(core::ptr::addr_of_mut!((*tport).mutex)));
__slate_cleanup_armed_3.set(true);
core::ptr::write(__slate_slot_2, (core::ptr::addr_of_mut!((*tport).mutex) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_2.set(true);
if (*port).uartclk == uartclk {
return;
} else {
(*port).uartclk = uartclk;
if !tty_port_initialized(tport as *const tty_port) {
return;
} else {
serial8250_do_set_termios(port, core::ptr::addr_of_mut!((*tty).termios), core::ptr::null::<ktermios>());
return;
}
}
}
}
__slate_cleanup_armed_7.set(false);
class_tty_port_tty_destructor(__slate_slot_7 as *mut _);
core::ptr::write(__slate_slot_1, class_mutex_constructor(core::ptr::addr_of_mut!((*tport).mutex)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!((*tport).mutex) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_0.set(true);
(*port).uartclk = uartclk;
__slate_cleanup_armed_0.set(false);
__class_mutex_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_mutex_destructor(__slate_slot_1 as *mut _);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_update_uartclk_646: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_port, u32)>, *mut ()>(Some(serial8250_update_uartclk)) };

fn serial8250_set_mini(port: *mut uart_port, termios: *mut ktermios) {
let up: *mut uart_8250_port = up_to_u8250p(port);
if ((unsafe { (*up).capabilities }) as u64) & BIT!(17i32) == 0 {
return;
}
let __v2: u32 = (unsafe { (*termios).c_cflag }) & 3221224639u32;
unsafe {
(*termios).c_cflag = __v2;
}
let csize: u32 = (unsafe { (*termios).c_cflag }) & 48;
if csize == 0 || csize == 16 {
let __v5: u32 = (unsafe { (*termios).c_cflag }) & 4294967247u32;
unsafe {
(*termios).c_cflag = __v5;
}
unsafe {
(*termios).c_cflag = (*termios).c_cflag | 32;
}
}
}

fn serial8250_set_trigger_for_slow_speed(port: *mut uart_port, termios: *mut ktermios, baud: u32) {
let up: *mut uart_8250_port = up_to_u8250p(port);
if ((unsafe { (*up).capabilities }) as u64) & BIT!(8i32) == 0 {
return;
}
if (unsafe { (*port).fifosize }) <= 1 {
return;
}
if baud >= 2400 {
return;
}
if !unsafe { (*up).dma }.is_null() {
return;
}
let __v2: u8 = (((unsafe { (*up).fcr }) as i32) & !192) as u8;
unsafe {
(*up).fcr = __v2;
}
unsafe {
(*up).fcr = (((*up).fcr as i32) | 0) as u8;
}
}

/// MCR-based auto flow control. When AFE is enabled, RTS will be deasserted when the receive FIFO
/// contains more characters than the trigger, or the MCR RTS bit is cleared.
fn serial8250_set_afe(port: *mut uart_port, termios: *mut ktermios) {
let up: *mut uart_8250_port = up_to_u8250p(port);
if ((unsafe { (*up).capabilities }) as u64) & BIT!(11i32) == 0 {
return;
}
let __v2: u8 = (((unsafe { (*up).mcr }) as i32) & !32) as u8;
unsafe {
(*up).mcr = __v2;
}
if (unsafe { (*termios).c_cflag }) & 2147483648 != 0 {
unsafe {
(*up).mcr = (((*up).mcr as i32) | 32) as u8;
}
}
}

fn serial8250_set_errors_and_ignores(port: *mut uart_port, termios: *mut ktermios) {
// Specify which conditions may be considered for error handling and the ignoring of
// characters. The actual ignoring of characters only occurs if the bit is set in
// @ignore_status_mask as well.
unsafe {
(*port).read_status_mask = 3u32;
}
if (unsafe { (*termios).c_iflag }) & 16 != 0 {
unsafe {
(*port).read_status_mask = (*port).read_status_mask | 12u32;
}
}
if (unsafe { (*termios).c_iflag }) & 11u32 != 0 {
unsafe {
(*port).read_status_mask = (*port).read_status_mask | 16;
}
}
// Characters to ignore
unsafe {
(*port).ignore_status_mask = 0;
}
if (unsafe { (*termios).c_iflag }) & 4 != 0 {
unsafe {
(*port).ignore_status_mask = (*port).ignore_status_mask | 12u32;
}
}
if (unsafe { (*termios).c_iflag }) & 1 != 0 {
unsafe {
(*port).ignore_status_mask = (*port).ignore_status_mask | 16;
}
// If we're ignoring parity and break indicators, ignore overruns too (for real raw
// support).
if (unsafe { (*termios).c_iflag }) & 4 != 0 {
unsafe {
(*port).ignore_status_mask = (*port).ignore_status_mask | 2;
}
}
}
// ignore all characters if CREAD is not set
if (unsafe { (*termios).c_cflag }) & 128 == 0 {
unsafe {
(*port).ignore_status_mask = (*port).ignore_status_mask | 1;
}
}
}

fn serial8250_set_ier(port: *mut uart_port, termios: *mut ktermios) {
let up: *mut uart_8250_port = up_to_u8250p(port);
// CTS flow control flag and modem status interrupts
let __v2: u8 = (((unsafe { (*up).ier }) as i32) & !8) as u8;
unsafe {
(*up).ier = __v2;
}
if (((unsafe { (*up).bugs }) as i32) as u64) & BIT!(2i32) == 0 && UART_ENABLE_MS!(unsafe { core::ptr::addr_of_mut!((*up).port) }, unsafe { (*termios).c_cflag }) {
unsafe {
(*up).ier = (((*up).ier as i32) | 8) as u8;
}
}
if ((unsafe { (*up).capabilities }) as u64) & BIT!(12i32) != 0 {
unsafe {
(*up).ier = (((*up).ier as i32) | 64) as u8;
}
}
if ((unsafe { (*up).capabilities }) as u64) & BIT!(13i32) != 0 {
unsafe {
(*up).ier = (((*up).ier as i32) | 16) as u8;
}
}
serial_port_out(port, 1i32, (unsafe { (*up).ier }) as i32);
}

fn serial8250_set_efr(port: *mut uart_port, termios: *mut ktermios) {
let up: *mut uart_8250_port = up_to_u8250p(port);
let mut efr_reg: u8 = 2;
let mut efr: u8 = 0;
if ((unsafe { (*up).capabilities }) as u64) & BIT!(9i32) == 0 {
return;
}
// TI16C752/Startech hardware flow control.  FIXME:
// - TI16C752 requires control thresholds to be set.
// - UART_MCR_RTS is ineffective if auto-RTS mode is enabled.
if (unsafe { (*termios).c_cflag }) & 2147483648 != 0 {
efr = ((efr as i32) | 128) as u8;
}
if (unsafe { (*port).flags }) & BIT_ULL!(25i32) != 0 {
efr_reg = 9;
}
serial_port_out(port, 3i32, 191i32);
serial_port_out(port, efr_reg as i32, efr as i32);
}

fn serial8250_set_fcr(port: *mut uart_port, termios: *mut ktermios) {
let up: *mut uart_8250_port = up_to_u8250p(port);
let is_16750: bool = (unsafe { (*port).r#type }) == 8;
if is_16750 {
serial_port_out(port, 2i32, (unsafe { (*up).fcr }) as i32);
}
// LCR DLAB must be reset to enable 64-byte FIFO mode. If the FCR is written without DLAB
// set, this mode will be disabled.
serial_port_out(port, 3i32, (unsafe { (*up).lcr }) as i32);
if is_16750 {
return;
}
// emulated UARTs (Lucent Venus 167x) need two steps
if ((unsafe { (*up).fcr }) as i32) & 1 != 0 {
serial_port_out(port, 2i32, 1i32);
}
serial_port_out(port, 2i32, (unsafe { (*up).fcr }) as i32);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_do_set_termios")]
extern "C" fn serial8250_do_set_termios(port: *mut uart_port, termios: *mut ktermios, old: *const ktermios) {
let mut __UNIQUE_ID_guard_648: core::mem::MaybeUninit<__SlateRecord_serial_core_h_803> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut __SlateRecord_serial_core_h_803 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_648) as *mut __SlateRecord_serial_core_h_803;
// Ok, we're now changing the port state. Do it with interrupts disabled.
//
// Synchronize UART_IER access against the console.
let mut scope: core::mem::MaybeUninit<*mut uart_8250_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut uart_8250_port = core::ptr::addr_of_mut!(scope) as *mut *mut uart_8250_port;
let mut lcr: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u8 = core::ptr::addr_of_mut!(lcr) as *mut u8;
let mut frac: u32 = unsafe { core::mem::zeroed() };
let mut quot: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(quot) as *mut u32;
let mut baud: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(baud) as *mut u32;
let mut up: *mut uart_8250_port = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_4 = __slate_slot_4;
let __slate_cleanup_armed_4 = core::cell::Cell::new(false);
let __slate_cleanup_4 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_4.get() {
unsafe { class_serial8250_rpm_destructor(__slate_cleanup_ptr_4 as *mut _) };
}
});
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { class_uart_port_lock_irqsave_destructor(__slate_cleanup_ptr_3 as *mut _) };
}
});
unsafe {
up = up_to_u8250p(port);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
frac = 0;
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
serial8250_set_mini(port, termios);
*__slate_slot_5 = serial8250_compute_lcr(up, (*termios).c_cflag);
*__slate_slot_8 = serial8250_get_baud_rate(port, termios, old);
*__slate_slot_7 = serial8250_get_divisor(port, *__slate_slot_8, core::ptr::addr_of_mut!(frac));
core::ptr::write(__slate_slot_4, class_serial8250_rpm_constructor(up));
__slate_cleanup_armed_4.set(true);
if !class_serial8250_rpm_lock_ptr(&(*core::ptr::addr_of_mut!(*__slate_slot_4))).is_null() || !class_serial8250_rpm_is_conditional {
if true {
core::ptr::write(__slate_slot_3, class_uart_port_lock_irqsave_constructor(port));
__slate_cleanup_armed_3.set(true);
(*up).lcr = *__slate_slot_5;
serial8250_set_trigger_for_slow_speed(port, termios, *__slate_slot_8);
serial8250_set_afe(port, termios);
uart_update_timeout(port, (*termios).c_cflag, *__slate_slot_8);
serial8250_set_errors_and_ignores(port, termios);
serial8250_set_ier(port, termios);
serial8250_set_efr(port, termios);
serial8250_set_divisor(port, *__slate_slot_8, *__slate_slot_7, frac);
serial8250_set_fcr(port, termios);
// Consoles manually poll CTS for hardware flow control.
if uart_console!(port) && ((*port).rs485.flags as u64) & _BITUL!(0i32) == 0 && (*termios).c_cflag & 2147483648 != 0 {
(*port).mctrl = (*port).mctrl | 4;
}
serial8250_set_mctrl(port, (*port).mctrl);
__slate_cleanup_armed_3.set(false);
class_uart_port_lock_irqsave_destructor(__slate_slot_3 as *mut _);
}
}
__slate_cleanup_armed_4.set(false);
class_serial8250_rpm_destructor(__slate_slot_4 as *mut _);
// Don't rewrite B0
if tty_termios_baud_rate(termios as *const ktermios) != 0 {
tty_termios_encode_baud_rate(termios, *__slate_slot_8, *__slate_slot_8);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_do_set_termios_649: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_port, *mut ktermios, *const ktermios)>, *mut ()>(Some(serial8250_do_set_termios)) };

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_set_termios")]
extern "C" fn serial8250_set_termios(port: *mut uart_port, termios: *mut ktermios, old: *const ktermios) {
if (unsafe { (*port).set_termios }) != None {
unsafe { unsafe { (*port).set_termios }.unwrap()(port, termios, old) };
} else {
serial8250_do_set_termios(port, termios, old);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_do_set_ldisc")]
extern "C" fn serial8250_do_set_ldisc(port: *mut uart_port, termios: *mut ktermios) {
if ((unsafe { (*termios).c_line }) as i32) == 18 {
unsafe {
(*port).flags = (*port).flags | 2048u64;
}
let mut __UNIQUE_ID_guard_650: *mut uart_port = class_uart_port_lock_irq_constructor(port);
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_650) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || class_uart_port_lock_irq_destructor(__slate_cleanup_ptr_3 as *mut _));
serial8250_enable_ms(port);
} else {
let __v6: u64 = (unsafe { (*port).flags }) & 18446744073709549567u64;
unsafe {
(*port).flags = __v6;
}
if !UART_ENABLE_MS!(port, unsafe { (*termios).c_cflag }) {
let mut __UNIQUE_ID_guard_651: *mut uart_port = class_uart_port_lock_irq_constructor(port);
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_651) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_uart_port_lock_irq_destructor(__slate_cleanup_ptr_7 as *mut _));
serial8250_disable_ms(port);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_do_set_ldisc_652: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_port, *mut ktermios)>, *mut ()>(Some(serial8250_do_set_ldisc)) };

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_set_ldisc")]
extern "C" fn serial8250_set_ldisc(port: *mut uart_port, termios: *mut ktermios) {
if (unsafe { (*port).set_ldisc }) != None {
unsafe { unsafe { (*port).set_ldisc }.unwrap()(port, termios) };
} else {
serial8250_do_set_ldisc(port, termios);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_do_pm")]
extern "C" fn serial8250_do_pm(port: *mut uart_port, state: u32, oldstate: u32) {
let p: *mut uart_8250_port = up_to_u8250p(port);
serial8250_set_sleep(p, (state != 0) as i32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_do_pm_653: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_port, u32, u32)>, *mut ()>(Some(serial8250_do_pm)) };

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_pm")]
extern "C" fn serial8250_pm(port: *mut uart_port, state: u32, oldstate: u32) {
if (unsafe { (*port).pm }) != None {
unsafe { unsafe { (*port).pm }.unwrap()(port, state, oldstate) };
} else {
serial8250_do_pm(port, state, oldstate);
}
}

fn serial8250_port_size(pt: *mut uart_8250_port) -> u32 {
if (unsafe { (*pt).port.mapsize }) != 0 {
return (unsafe { (*pt).port.mapsize }) as u32;
}
if is_omap1_8250(pt) != 0 {
return (22i32 << ((unsafe { (*pt).port.regshift }) as i32)) as u32;
}
(8i32 << ((unsafe { (*pt).port.regshift }) as i32)) as u32
}

/// Resource handling.
fn serial8250_request_std_resource(up: *mut uart_8250_port) -> i32 {
let size: u32 = serial8250_port_size(up);
let port: *mut uart_port = unsafe { core::ptr::addr_of_mut!((*up).port) };
if unsafe { uart_iotype_mmio((*port).iotype) } {
if (unsafe { (*port).mapbase }) == 0 {
return 22i32.wrapping_neg();
}
if request_mem_region!(unsafe { (*port).mapbase }, size, c"serial".as_ptr() as *mut u8).is_null() {
return 16i32.wrapping_neg();
}
if (unsafe { (*port).flags }) & BIT_ULL!(31i32) != 0 {
unsafe {
(*port).membase = ioremap((*port).mapbase, size as u64) as *mut u8;
}
if unsafe { (*port).membase }.is_null() {
release_mem_region!(unsafe { (*port).mapbase }, size);
return 12i32.wrapping_neg();
}
}
} else {
if unsafe { uart_iotype_io((*port).iotype) } {
if request_region!(unsafe { (*port).iobase }, size, c"serial".as_ptr() as *mut u8).is_null() {
return 16i32.wrapping_neg();
}
}
}
0
}

fn serial8250_release_std_resource(up: *mut uart_8250_port) {
let size: u32 = serial8250_port_size(up);
let port: *mut uart_port = unsafe { core::ptr::addr_of_mut!((*up).port) };
if unsafe { uart_iotype_mmio((*port).iotype) } {
if (unsafe { (*port).mapbase }) == 0 {
return;
}
if (unsafe { (*port).flags }) & BIT_ULL!(31i32) != 0 {
unsafe { iounmap((*port).membase as *mut ()) };
unsafe {
(*port).membase = core::ptr::null_mut::<u8>();
}
}
release_mem_region!(unsafe { (*port).mapbase }, size);
} else {
if unsafe { uart_iotype_io((*port).iotype) } {
release_region!(unsafe { (*port).iobase }, size);
}
}
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_release_port")]
extern "C" fn serial8250_release_port(port: *mut uart_port) {
let up: *mut uart_8250_port = up_to_u8250p(port);
serial8250_release_std_resource(up);
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_request_port")]
extern "C" fn serial8250_request_port(port: *mut uart_port) -> i32 {
let up: *mut uart_8250_port = up_to_u8250p(port);
serial8250_request_std_resource(up)
}

fn fcr_get_rxtrig_bytes(up: *mut uart_8250_port) -> i32 {
let conf_type: *const serial8250_config = unsafe { core::ptr::addr_of!(uart_config.0) as *const serial8250_config }.wrapping_offset((unsafe { (*up).port.r#type }) as isize);
let bytes: u8 = unsafe { *((*conf_type).rxtrig_bytes.as_ptr() as *const u8).wrapping_offset(UART_FCR_R_TRIG_BITS__2!(unsafe { (*up).fcr }) as isize) };
if bytes != 0 { bytes as i32 } else { 95i32.wrapping_neg() }
}

fn bytes_to_fcr_rxtrig(up: *mut uart_8250_port, bytes: u8) -> i32 {
let conf_type: *const serial8250_config = unsafe { core::ptr::addr_of!(uart_config.0) as *const serial8250_config }.wrapping_offset((unsafe { (*up).port.r#type }) as isize);
let mut i: i32 = 1;
if (unsafe { *((*conf_type).rxtrig_bytes.as_ptr() as *const u8).wrapping_offset(UART_FCR_R_TRIG_BITS!(0i32) as isize) }) == 0 {
return 95i32.wrapping_neg();
}
while i < 4 {
if (bytes as i32) < ((unsafe { *((*conf_type).rxtrig_bytes.as_ptr() as *const u8).wrapping_offset(i as isize) }) as i32) {
let __v1: i32 = i.wrapping_sub(1);
i = __v1;
return __v1 << 6i32;
}
// Use the nearest lower value
i = i.wrapping_add(1);
}
192
}

fn do_get_rxtrig(port: *mut tty_port) -> i32 {
let state: *mut uart_state = container_of!(port, uart_state);
let uport: *mut uart_port = unsafe { (*state).uart_port };
let up: *mut uart_8250_port = up_to_u8250p(uport);
if ((unsafe { (*up).capabilities }) as u64) & BIT!(8i32) == 0 || (unsafe { (*uport).fifosize }) <= 1 {
return 22i32.wrapping_neg();
}
fcr_get_rxtrig_bytes(up)
}

fn do_serial8250_get_rxtrig(port: *mut tty_port) -> i32 {
let mut rxtrig_bytes: i32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*port).mutex)) };
rxtrig_bytes = do_get_rxtrig(port);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*port).mutex)) };
rxtrig_bytes
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.rx_trig_bytes_show")]
extern "C" fn rx_trig_bytes_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let port: *mut tty_port = dev_get_drvdata(dev as *const device) as *mut tty_port;
let state: *mut uart_state = container_of!(port, uart_state);
let uport: *mut uart_port = unsafe { (*state).uart_port };
let rxtrig_bytes: i32 = if (unsafe { (*uport).get_rxtrig }) != None {
unsafe { unsafe { (*uport).get_rxtrig }.unwrap()(uport) }
} else {
do_serial8250_get_rxtrig(port)
};
if rxtrig_bytes < 0 {
return rxtrig_bytes as i64;
}
(unsafe { sysfs_emit(buf, c"%d\n".as_ptr() as *const u8, rxtrig_bytes) }) as i64
}

fn do_set_rxtrig(port: *mut tty_port, bytes: u8) -> i32 {
let state: *mut uart_state = container_of!(port, uart_state);
let uport: *mut uart_port = unsafe { (*state).uart_port };
let up: *mut uart_8250_port = up_to_u8250p(uport);
let mut rxtrig: i32 = 0;
if ((unsafe { (*up).capabilities }) as u64) & BIT!(8i32) == 0 || (unsafe { (*uport).fifosize }) <= 1 {
return 22i32.wrapping_neg();
}
rxtrig = bytes_to_fcr_rxtrig(up, bytes);
if rxtrig < 0 {
return rxtrig;
}
serial8250_clear_fifos(up);
let __v3: u8 = (((unsafe { (*up).fcr }) as i32) & !192) as u8;
unsafe {
(*up).fcr = __v3;
}
unsafe {
(*up).fcr = (((*up).fcr as i32) | ((rxtrig as u8) as i32)) as u8;
}
serial_out(up, 2i32, (unsafe { (*up).fcr }) as i32);
0
}

fn do_serial8250_set_rxtrig(port: *mut tty_port, bytes: u8) -> i32 {
let mut ret: i32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*port).mutex)) };
ret = do_set_rxtrig(port, bytes);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*port).mutex)) };
ret
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.rx_trig_bytes_store")]
extern "C" fn rx_trig_bytes_store(dev: *mut device, attr: *mut device_attribute, buf: *const u8, count: u64) -> i64 {
let port: *mut tty_port = dev_get_drvdata(dev as *const device) as *mut tty_port;
let state: *mut uart_state = container_of!(port, uart_state);
let uport: *mut uart_port = unsafe { (*state).uart_port };
let mut bytes: u8 = 0;
let mut ret: i32 = 0;
if count == 0 {
return 22i32.wrapping_neg() as i64;
}
ret = unsafe { kstrtou8(buf, 10u32, core::ptr::addr_of_mut!(bytes)) };
if ret < 0 {
return ret as i64;
}
if (unsafe { (*uport).set_rxtrig }) != None {
ret = unsafe { unsafe { (*uport).set_rxtrig }.unwrap()(uport, bytes) };
} else {
ret = do_serial8250_set_rxtrig(port, bytes);
}
if ret < 0 {
return ret as i64;
}
count as i64
}

static mut dev_attr_rx_trig_bytes: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"rx_trig_bytes\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_8250_port_c_3110>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_8250_port_c_3110_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_8250_port_c_3110_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_8250_port_c_3110_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_8250_port_c_3110_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_8250_port_c_3110_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(rx_trig_bytes_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(rx_trig_bytes_store) } };

static mut serial8250_dev_attrs: __SlateAlign16<[*mut attribute; 2]> = __SlateAlign16([unsafe { core::ptr::addr_of_mut!(dev_attr_rx_trig_bytes.attr) }, core::ptr::null_mut::<attribute>()]);

static mut serial8250_dev_attr_group: attribute_group = {
let mut __t0: attribute_group = unsafe { core::mem::zeroed() };
__t0.__slate_anon_4 = __SlateRecord_sysfs_h_118 { attrs: unsafe { core::ptr::addr_of_mut!(serial8250_dev_attrs.0) as *mut *mut attribute } };
__t0
};

fn register_dev_spec_attr_grp(up: *mut uart_8250_port) {
let conf_type: *const serial8250_config = unsafe { core::ptr::addr_of!(uart_config.0) as *const serial8250_config }.wrapping_offset((unsafe { (*up).port.r#type }) as isize);
if (unsafe { *((*conf_type).rxtrig_bytes.as_ptr() as *const u8).wrapping_offset(0 as isize) }) != 0 {
unsafe {
(*up).port.attr_group = core::ptr::addr_of_mut!(serial8250_dev_attr_group);
}
}
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_config_port")]
extern "C" fn serial8250_config_port(port: *mut uart_port, flags: i32) {
let up: *mut uart_8250_port = up_to_u8250p(port);
let mut ret: i32 = 0;
// Find the region that we can probe for.  This in turn
// tells us whether we can probe for the type of port.
ret = serial8250_request_std_resource(up);
if ret < 0 {
return;
}
if ((unsafe { (*port).iotype }) as i32) != ((unsafe { (*up).cur_iotype }) as i32) {
set_io_from_upio(port);
}
if flags & 1i32 != 0 {
autoconfig(up);
}
// HW bugs may trigger IRQ while IIR == NO_INT
if (unsafe { (*port).r#type }) == 20 {
let __v1: u16 = unsafe { (*up).bugs };
let __v2: u16 = (((__v1 as i32) as u64) | BIT!(2i32)) as u16;
unsafe {
(*up).bugs = __v2;
}
}
if (unsafe { (*port).r#type }) != 0 && flags & 2i32 != 0 {
autoconfig_irq(up);
}
if (unsafe { (*port).r#type }) == 0 {
serial8250_release_std_resource(up);
}
register_dev_spec_attr_grp(up);
unsafe {
(*up).fcr = uart_config.0[(*up).port.r#type as usize].fcr;
}
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_verify_port")]
extern "C" fn serial8250_verify_port(port: *mut uart_port, ser: *mut serial_struct) -> i32 {
if ((unsafe { (*ser).irq }) as u32) >= unsafe { irq_get_nr_irqs() } || (unsafe { (*ser).irq }) < 0 || (unsafe { (*ser).baud_base }) < 9600 || (unsafe { (*ser).r#type }) < 0 || ((unsafe { (*ser).r#type }) as u64) >= ARRAY_SIZE!() || (unsafe { (*ser).r#type }) == 5 || (unsafe { (*ser).r#type }) == 9 {
return 22i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_type")]
extern "C" fn serial8250_type(port: *mut uart_port) -> *const u8 {
let mut r#type: i32 = (unsafe { (*port).r#type }) as i32;
if (r#type as u64) >= ARRAY_SIZE__10!() {
r#type = 0;
}
unsafe { uart_config.0[r#type as usize].name }
}

static serial8250_pops: __SlateConst<uart_ops> = __SlateConst({
let mut __t0: uart_ops = unsafe { core::mem::zeroed() };
__t0.tx_empty = Some(serial8250_tx_empty);
__t0.set_mctrl = Some(serial8250_set_mctrl);
__t0.get_mctrl = Some(serial8250_get_mctrl);
__t0.stop_tx = Some(serial8250_stop_tx);
__t0.start_tx = Some(serial8250_start_tx);
__t0.throttle = Some(serial8250_throttle);
__t0.unthrottle = Some(serial8250_unthrottle);
__t0.stop_rx = Some(serial8250_stop_rx);
__t0.enable_ms = Some(serial8250_enable_ms);
__t0.break_ctl = Some(serial8250_break_ctl);
__t0.startup = Some(serial8250_startup);
__t0.shutdown = Some(serial8250_shutdown);
__t0.flush_buffer = Some(serial8250_flush_buffer);
__t0.set_termios = Some(serial8250_set_termios);
__t0.set_ldisc = Some(serial8250_set_ldisc);
__t0.pm = Some(serial8250_pm);
__t0.r#type = Some(serial8250_type);
__t0.release_port = Some(serial8250_release_port);
__t0.request_port = Some(serial8250_request_port);
__t0.config_port = Some(serial8250_config_port);
__t0.verify_port = Some(serial8250_verify_port);
__t0
});

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_init_port")]
extern "C" fn serial8250_init_port(up: *mut uart_8250_port) {
let mut __slate_compound_0: spinlock = unsafe { core::mem::zeroed() };
let port: *mut uart_port = unsafe { core::ptr::addr_of_mut!((*up).port) };
'__slate_break_1: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*port).lock) });
unsafe {
*core::ptr::addr_of_mut!((*port).lock) = *{
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
unsafe {
(*port).ctrl_id = 0;
}
unsafe {
(*port).pm = None;
}
unsafe {
(*port).ops = core::ptr::addr_of!(serial8250_pops.0);
}
unsafe {
(*port).has_sysrq = IS_ENABLED__2!() as u8;
}
unsafe {
(*up).cur_iotype = 255u8;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_init_port_654: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_8250_port)>, *mut ()>(Some(serial8250_init_port)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_set_defaults")]
extern "C" fn serial8250_set_defaults(up: *mut uart_8250_port) {
let port: *mut uart_port = unsafe { core::ptr::addr_of_mut!((*up).port) };
if (unsafe { (*up).port.flags }) & BIT_ULL!(27i32) != 0 {
let r#type: u32 = unsafe { (*up).port.r#type };
if (unsafe { (*up).port.fifosize }) == 0 {
unsafe {
(*up).port.fifosize = uart_config.0[r#type as usize].fifo_size as u32;
}
}
if (unsafe { (*up).tx_loadsz }) == 0 {
unsafe {
(*up).tx_loadsz = uart_config.0[r#type as usize].tx_loadsz as u32;
}
}
if (unsafe { (*up).capabilities }) == 0 {
unsafe {
(*up).capabilities = uart_config.0[r#type as usize].flags;
}
}
}
set_io_from_upio(port);
// default dma handlers
if !unsafe { (*up).dma }.is_null() {
if (unsafe { (*(*up).dma).tx_dma }) == None {
unsafe {
(*(*up).dma).tx_dma = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut uart_8250_port) -> i32>>(serial8250_tx_dma as *const ()) };
}
}
if (unsafe { (*(*up).dma).rx_dma }) == None {
unsafe {
(*(*up).dma).rx_dma = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut uart_8250_port) -> i32>>(serial8250_rx_dma as *const ()) };
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_set_defaults_655: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_8250_port)>, *mut ()>(Some(serial8250_set_defaults)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_fifo_wait_for_lsr_thre")]
extern "C" fn serial8250_fifo_wait_for_lsr_thre(up: *mut uart_8250_port, wctxt: *mut nbcon_write_context, count: u32) {
// For console writing, enter/exit an unsafe section for each byte
// in order to pass the ownership as quickly as possible if a higher
// priority context wants ownership. Otherwise, an attempt to take
// over the ownership might timeout. The new owner will wait for
// UART_LSR_THRE before reusing the fifo.
for i in 0..count {
let mut tx_ready: bool = false;
let __v0: bool = !wctxt.is_null() && !unsafe { nbcon_enter_unsafe(wctxt) };
if __v0 {
return;
}
tx_ready = wait_for_lsr(up, 32i32);
if !wctxt.is_null() {
unsafe { nbcon_exit_unsafe(wctxt) };
}
if tx_ready {
break;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_serial8250_fifo_wait_for_lsr_thre_656: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut uart_8250_port, *mut nbcon_write_context, u32)>, *mut ()>(Some(serial8250_fifo_wait_for_lsr_thre)) };

fn serial8250_console_putchar(port: *mut uart_port, ch: u8) {
let up: *mut uart_8250_port = up_to_u8250p(port);
serial_port_out(port, 0i32, ch as i32);
unsafe {
(*up).console_line_ended = (ch as i32) == 10;
}
}

#[unsafe(link_section = ".text.slate_distinct._8250_port.serial8250_console_wait_putchar")]
extern "C" fn serial8250_console_wait_putchar(port: *mut uart_port, ch: u8) {
let up: *mut uart_8250_port = up_to_u8250p(port);
wait_for_xmitr(up, 32i32);
serial8250_console_putchar(port, ch);
}

/// Restore serial console when h/w power-off detected
fn serial8250_console_restore(up: *mut uart_8250_port) {
let port: *mut uart_port = unsafe { core::ptr::addr_of_mut!((*up).port) };
let mut termios: ktermios = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(termios) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 44usize);
}
let mut baud: u32 = 0;
let mut quot: u32 = 0;
let mut frac: u32 = 0;
termios.c_cflag = (unsafe { (*(*port).cons).cflag }) as u32;
termios.c_ispeed = unsafe { (*(*port).cons).ispeed };
termios.c_ospeed = unsafe { (*(*port).cons).ospeed };
if !unsafe { (*(*port).state).port.tty }.is_null() && termios.c_cflag == 0 {
termios.c_cflag = unsafe { (*(*(*port).state).port.tty).termios.c_cflag };
termios.c_ispeed = unsafe { (*(*(*port).state).port.tty).termios.c_ispeed };
termios.c_ospeed = unsafe { (*(*(*port).state).port.tty).termios.c_ospeed };
}
baud = serial8250_get_baud_rate(port, core::ptr::addr_of_mut!(termios), core::ptr::null::<ktermios>());
quot = serial8250_get_divisor(port, baud, core::ptr::addr_of_mut!(frac));
serial8250_set_divisor(port, baud, quot, frac);
serial_port_out(port, 3i32, (unsafe { (*up).lcr }) as i32);
serial8250_out_MCR(up, ((unsafe { (*up).mcr }) as i32) | 1 | 2);
}

/// Print a string to the serial port using the device FIFO
///
/// It sends fifosize bytes and then waits for the fifo
/// to get empty.
fn __serial8250_console_fifo_write(up: *mut uart_8250_port, wctxt: *mut nbcon_write_context, mut s: *const u8, count: u32) {
let end: *const u8 = s.wrapping_offset(count as isize);
let fifosize: u32 = unsafe { (*up).tx_loadsz };
let port: *mut uart_port = unsafe { core::ptr::addr_of_mut!((*up).port) };
let mut tx_count: u32 = 0;
let mut cr_sent: bool = false;
let mut i: u32 = 0;
while s != end {
// Allow timeout for each byte of a possibly full FIFO
serial8250_fifo_wait_for_lsr_thre(up, wctxt, fifosize);
// Fill the FIFO. If a handover or takeover occurs, writing
// must be aborted since the string data is no longer valid.
i = 0;
while i < fifosize && s != end {
if !unsafe { nbcon_enter_unsafe(wctxt) } {
return;
}
if ((unsafe { *s }) as i32) == 10 && !cr_sent {
serial8250_console_putchar(port, 13u8);
cr_sent = true;
} else {
let __v0: *const u8 = s;
s = __v0.wrapping_offset(1 as isize);
serial8250_console_putchar(port, unsafe { *__v0 });
cr_sent = false;
}
unsafe { nbcon_exit_unsafe(wctxt) };
i = i.wrapping_add(1);
}
tx_count = i;
}
// Allow timeout for each byte written since the caller will only wait
// for UART_LSR_BOTH_EMPTY using the timeout of a single character
serial8250_fifo_wait_for_lsr_thre(up, wctxt, tx_count);
}

fn serial8250_console_fifo_write(up: *mut uart_8250_port, wctxt: *mut nbcon_write_context) {
__serial8250_console_fifo_write(up, wctxt, (unsafe { (*wctxt).outbuf }) as *const u8, unsafe { (*wctxt).len });
}

fn __serial8250_console_byte_write(up: *mut uart_8250_port, wctxt: *mut nbcon_write_context, mut s: *const u8, count: u32) {
let port: *mut uart_port = unsafe { core::ptr::addr_of_mut!((*up).port) };
let end: *const u8 = s.wrapping_offset(count as isize);
// Write out the message. If a handover or takeover occurs, writing
// must be aborted since the string data is no longer valid.
while s != end {
if !unsafe { nbcon_enter_unsafe(wctxt) } {
return;
}
let __v0: *const u8 = s;
s = __v0.wrapping_offset(1 as isize);
unsafe { uart_console_write(port, __v0, 1u32, Some(serial8250_console_wait_putchar)) };
unsafe { nbcon_exit_unsafe(wctxt) };
}
}

fn serial8250_console_byte_write(up: *mut uart_8250_port, wctxt: *mut nbcon_write_context) {
__serial8250_console_byte_write(up, wctxt, (unsafe { (*wctxt).outbuf }) as *const u8, unsafe { (*wctxt).len });
}

/// Print the console line using the appropriate variant. If ownership is lost
/// at any time during printing, the printing is aborted.
fn __serial8250_console_write(up: *mut uart_8250_port, wctxt: *mut nbcon_write_context, use_fifo: bool) {
// If the console printer did not fully output the previous line, it
// must have been handed or taken over. Insert a newline in order to
// maintain clean output.
if !unsafe { (*up).console_line_ended } {
if use_fifo {
__serial8250_console_fifo_write(up, wctxt, c"\n".as_ptr() as *const u8, 1u32);
} else {
__serial8250_console_byte_write(up, wctxt, c"\n".as_ptr() as *const u8, 1u32);
}
}
if use_fifo {
serial8250_console_fifo_write(up, wctxt);
} else {
serial8250_console_byte_write(up, wctxt);
}
}

/// Print a string to the serial port trying not to disturb
/// any possible real use of the port...
#[unsafe(no_mangle)]
extern "C" fn serial8250_console_write(up: *mut uart_8250_port, wctxt: *mut nbcon_write_context, is_atomic: bool) {
let em485: *mut uart_8250_em485 = unsafe { (*up).em485 };
let port: *mut uart_port = unsafe { core::ptr::addr_of_mut!((*up).port) };
let mut ier: u32 = 0;
let mut use_fifo: bool = false;
if !unsafe { nbcon_enter_unsafe(wctxt) } {
return;
}
// First, save the IER, then disable the interrupts. The special
// variant to clear the IER is used because emergency and panic
// printing is synchronized only by nbcon ownership without
// holding the port lock.
ier = serial_port_in(port, 1i32) as u32;
__serial8250_clear_IER(up);
// check scratch reg to see if port powered off during system sleep
let __v0: bool = (unsafe { (*up).canary }) != 0 && ((unsafe { (*up).canary }) as i32) != serial_port_in(port, 7i32);
if __v0 {
serial8250_console_restore(up);
unsafe {
(*up).canary = 0;
}
}
if !em485.is_null() {
if ((unsafe { (*em485).__slate_bits_0.__get_tx_stopped() }) as i32) != 0 {
unsafe { unsafe { (*up).rs485_start_tx }.unwrap()(up, false) };
}
mdelay__2!(unsafe { (*port).rs485.delay_rts_before_send });
}
let __v4: bool = ((unsafe { (*up).capabilities }) as u64) & BIT!(8i32) != 0 && ((unsafe { (*up).capabilities }) as u64) & BIT!(17i32) == 0 && (unsafe { (*up).tx_loadsz }) > 1 && ((unsafe { (*up).fcr }) as i32) & 1 != 0 && !unsafe { (*port).state }.is_null() && test_bit!(0i32, unsafe { core::ptr::addr_of_mut!((*(*port).state).port.iflags) }) != 0;
use_fifo = __v4 && !uart_console_hwflow_active(unsafe { core::ptr::addr_of_mut!((*up).port) });
// BCM283x requires to check the fifo
// after each byte.
// tx_loadsz contains the transmit fifo size
// After we put a data in the fifo, the controller will send
// it regardless of the CTS state. Therefore, only use fifo
// if we don't use control flow.
unsafe { nbcon_exit_unsafe(wctxt) };
__serial8250_console_write(up, wctxt, use_fifo);
// Re-enter an unsafe section in order to perform final actions
// (such as re-enabling interrupts). If ownership was lost, this
// context must reacquire ownership.
while !unsafe { nbcon_enter_unsafe(wctxt) } {
unsafe { nbcon_reacquire_nobuf(wctxt) };
}
// Finally, wait for transmitter to become empty
// and restore the IER
wait_for_xmitr(up, 96i32);
if !em485.is_null() {
mdelay__2!(unsafe { (*port).rs485.delay_rts_after_send });
// Toggle unsafe after possibly long delay
unsafe { nbcon_exit_unsafe(wctxt) };
while !unsafe { nbcon_enter_unsafe(wctxt) } {
unsafe { nbcon_reacquire_nobuf(wctxt) };
}
if ((unsafe { (*em485).__slate_bits_0.__get_tx_stopped() }) as i32) != 0 {
unsafe { unsafe { (*up).rs485_stop_tx }.unwrap()(up, false) };
}
}
// Toggle unsafe after possibly long delay
unsafe { nbcon_exit_unsafe(wctxt) };
while !unsafe { nbcon_enter_unsafe(wctxt) } {
unsafe { nbcon_reacquire_nobuf(wctxt) };
}
serial_port_out(port, 1i32, ier as i32);
// The receive handling will happen properly because the
// receive ready bit will still be set; it is not cleared
// on read.  However, modem control will not, we must
// call it if we have saved something in the saved flags
// while processing with interrupts off.
if (unsafe { (*up).msr_saved_flags }) != 0 {
if is_atomic {
// For atomic, MSR handling must be deferred to
// irq_work because this may be a context that does
// not permit waking up tasks.
//
// But no irq_work may be queued when suspending.
// In that case, the MSR handling will occur during
// resume in serial8250_resume_port().
if unsafe { (*up).console_msr_work_allow } {
unsafe { irq_work_queue(core::ptr::addr_of_mut!((*up).console_msr_work)) };
}
} else {
serial8250_modem_status(up);
}
}
unsafe { nbcon_exit_unsafe(wctxt) };
}

fn probe_baud(port: *mut uart_port) -> u32 {
let lcr: u8 = serial_port_in(port, 3i32) as u8;
let mut dll: u8 = 0;
let mut dlm: u8 = 0;
let mut quot: u32 = 0;
serial_port_out(port, 3i32, (lcr as i32) | 128);
dll = serial_port_in(port, 0i32) as u8;
dlm = serial_port_in(port, 1i32) as u8;
serial_port_out(port, 3i32, lcr as i32);
quot = ((dlm as i32) << 8i32 | (dll as i32)) as u32;
(unsafe { (*port).uartclk }) / 16 / quot
}

/// irq_work handler to perform modem control during console output.
/// Only triggered via ->write_atomic() callback because it may be
/// in a scheduler or NMI context, unable to wake tasks.
#[unsafe(link_section = ".text.slate_distinct._8250_port.console_msr_handler")]
extern "C" fn console_msr_handler(iwp: *mut irq_work) {
let up: *mut uart_8250_port = container_of__26!(iwp, uart_8250_port);
let port: *mut uart_port = unsafe { core::ptr::addr_of_mut!((*up).port) };
let mut __UNIQUE_ID_guard_657: *mut uart_port = class_uart_port_lock_constructor(port);
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_657) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_uart_port_lock_destructor(__slate_cleanup_ptr_1 as *mut _));
serial8250_modem_status(up);
}

#[unsafe(no_mangle)]
extern "C" fn serial8250_console_setup(port: *mut uart_port, options: *mut u8, probe: bool) -> i32 {
let mut __slate_compound_0: irq_work = unsafe { core::mem::zeroed() };
let up: *mut uart_8250_port = up_to_u8250p(port);
let mut baud: i32 = 9600;
let mut bits: i32 = 8;
let mut parity: i32 = 110;
let mut flow: i32 = 110;
let mut ret: i32 = 0;
if (unsafe { (*port).iobase }) == 0 && unsafe { (*port).membase }.is_null() {
return 19i32.wrapping_neg();
}
unsafe {
(*up).console_line_ended = true;
}
unsafe {
(*up).console_msr_work_allow = true;
}
unsafe {
(*up).console_msr_work = *{
__slate_compound_0 = irq_work { node: {
let mut __t0: __call_single_node = core::mem::zeroed();
__t0.__slate_anon_1 = __SlateRecord_smp_types_h_60 { u_flags: 4 };
__t0
}, func: Some(console_msr_handler), irqwait: rcuwait { task: core::ptr::null_mut::<task_struct>() } };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
if !options.is_null() {
unsafe { uart_parse_options(options as *const u8, core::ptr::addr_of_mut!(baud), core::ptr::addr_of_mut!(parity), core::ptr::addr_of_mut!(bits), core::ptr::addr_of_mut!(flow)) };
} else {
if probe {
baud = probe_baud(port) as i32;
}
}
ret = unsafe { uart_set_options(port, (*port).cons, baud, parity, bits, flow) };
if ret != 0 {
return ret;
}
// Track user-specified console flow control.
uart_set_cons_flow_enabled(port, flow == 114);
if !unsafe { (*port).dev }.is_null() {
pm_runtime_get_sync(unsafe { (*port).dev });
}
0
}

#[unsafe(no_mangle)]
extern "C" fn serial8250_console_exit(port: *mut uart_port) -> i32 {
let up: *mut uart_8250_port = up_to_u8250p(port);
unsafe { irq_work_sync(core::ptr::addr_of_mut!((*up).console_msr_work)) };
if !unsafe { (*port).dev }.is_null() {
pm_runtime_put_sync(unsafe { (*port).dev });
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_658: [u8; 76] = [56, 50, 53, 48, 95, 98, 97, 115, 101, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 66, 97, 115, 101, 32, 112, 111, 114, 116, 32, 111, 112, 101, 114, 97, 116, 105, 111, 110, 115, 32, 102, 111, 114, 32, 56, 50, 53, 48, 47, 49, 54, 53, 53, 48, 45, 116, 121, 112, 101, 32, 115, 101, 114, 105, 97, 108, 32, 112, 111, 114, 116, 115, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_659: [u8; 49] = [56, 50, 53, 48, 95, 98, 97, 115, 101, 46, 102, 105, 108, 101, 61, 100, 114, 105, 118, 101, 114, 115, 47, 116, 116, 121, 47, 115, 101, 114, 105, 97, 108, 47, 56, 50, 53, 48, 47, 56, 50, 53, 48, 95, 98, 97, 115, 101, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_660: [u8; 22] = [56, 50, 53, 48, 95, 98, 97, 115, 101, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_set_io_from_upio_4: [u8; 26] = *b"Unsupported UART type %x\n\0";

static __slate_asm_string_set_io_from_upio_5: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/tty/serial/8250/8250_port.c\0";
