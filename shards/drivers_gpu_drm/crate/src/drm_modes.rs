
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
//! Copyright © 1997-2003 by The XFree86 Project, Inc.
//! Copyright © 2007 Dave Airlie
//! Copyright © 2007-2008 Intel Corporation
//!   Jesse Barnes <jesse.barnes@intel.com>
//! Copyright 2005-2006 Luc Verhaegen
//! Copyright (c) 2001, Andy Ritger  aritger@nvidia.com
//!
//! Permission is hereby granted, free of charge, to any person obtaining a
//! copy of this software and associated documentation files (the "Software"),
//! to deal in the Software without restriction, including without limitation
//! the rights to use, copy, modify, merge, publish, distribute, sublicense,
//! and/or sell copies of the Software, and to permit persons to whom the
//! Software is furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in
//! all copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
//! THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
//! OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
//! ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
//! OTHER DEALINGS IN THE SOFTWARE.
//!
//! Except as contained in this notice, the name of the copyright holder(s)
//! and author(s) shall not be used in advertising or otherwise to promote
//! the sale, use or other dealings in this Software without prior written
//! authorization from the copyright holder(s) and author(s).
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
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
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
//! Resource type
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
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
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
//! 	Enough for the VT console needs, see its max_font_width/height
//! Number of bits in the handle that stack depot doesn't use. Users may store
//! information in them via stack_depot_set/get_extra_bits.
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
//! Video mode flags
//!
//! bit compatible with the xrandr RR_ definitions (bits 0-13)
//!
//! ABI warning: Existing userspace really expects
//! the mode flags to match the xrandr definitions. Any
//! changes that don't match the xrandr definitions will
//! likely need a new client cap or some other mechanism
//! to avoid breaking existing userspace. This includes
//! allocating new flags in the previously unused bits!
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
// Math result not representable
// Invalid system call number
// Value too large for defined data type
// Operation not supported on transport endpoint
// Memory page has hardware error
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// Resource extended types
// No address assigned yet
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
// Are we running in atomic context?  WARNING: this macro cannot
// always detect atomic context; in particular, it cannot know about
// held spinlocks in non-preemptible kernels.  Thus it should not be
// used in the general case to determine whether sleeping is possible.
// Do not use in_atomic() in driver code.
// Overflow Flag
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
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
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
macro_rules! ARRAY_SIZE__23 {
() => {
((core::mem::size_of::<[drm_named_mode; 4]>() as u64) / (core::mem::size_of::<drm_named_mode>() as u64)).wrapping_add(__must_be_array__80!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__51 {
() => {
((core::mem::size_of::<[drm_named_mode; 4]>() as u64) / (core::mem::size_of::<drm_named_mode>() as u64)).wrapping_add(__must_be_array__15!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__52 {
() => {
((core::mem::size_of::<[*const u8; 40]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__56!() as u64)
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

/// Same as above but for u64 dividends. divisor must be a 32-bit
/// number.
macro_rules! DIV_ROUND_CLOSEST_ULL__2 {
($x:expr, $divisor:expr) => {
{
let mut __v: u64;
let __v2: u32 = $divisor;
let mut __v3: u64 = $x.wrapping_add((__v2 / 2) as u64);
do_div__2!(__v3, __v2);
__v = __v3;
__v
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

macro_rules! KHZ2PICOS {
($a:expr) => {
1000000000 / (($a) as u64)
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
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

macro_rules! __must_be_array__15 {
() => {
core::mem::size_of::<__SlateRecord_drm_modes_c_2534>() as i32
};
}

macro_rules! __must_be_array__56 {
() => {
core::mem::size_of::<__SlateRecord_drm_modes_c_1812>() as i32
};
}

macro_rules! __must_be_array__80 {
() => {
core::mem::size_of::<__SlateRecord_drm_modes_c_2337>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr {
() => {
core::mem::size_of::<__SlateRecord_drm_modes_c_2645_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__2 {
() => {
core::mem::size_of::<__SlateRecord_drm_modes_c_2645>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__3 {
() => {
core::mem::size_of::<__SlateRecord_drm_modes_c_2345>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__5 {
() => {
core::mem::size_of::<__SlateRecord_drm_modes_c_2686>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__16 {
() => {
core::mem::size_of::<__SlateRecord_drm_modes_c_2345_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__18 {
() => {
core::mem::size_of::<__SlateRecord_drm_modes_c_2686_2>() as i32
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
if is_signed_type__6!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<*mut u8>() }) as u64) < 4) as i32)) }
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
macro_rules! __sign_use__5 {
($ux:expr) => {
if is_signed_type__3!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u16>() }) as u64) < 4) as i32)) }
};
}

macro_rules! __types_ok__7 {
($ux:expr, $uy:expr) => {
__sign_use__4!($ux) & __sign_use__4!($uy)
};
}

macro_rules! __types_ok__3 {
($ux:expr, $uy:expr) => {
__sign_use__5!($ux) & __sign_use__5!($uy)
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

/// check_mul_overflow() - Calculate multiplication with overflow checking
/// @a: first factor
/// @b: second factor
/// @d: pointer to store product
///
/// Returns true on wrap-around, false otherwise.
///
/// *@d holds the results of the attempted multiplication, regardless of whether
/// wrap-around occurred.
macro_rules! check_mul_overflow__2 {
($a:expr, $b:expr, $d:expr) => {
__must_check_overflow({
let __t: i128 = ($a) as i128;
let __t2: i128 = ($b) as i128;
let __t3: i128 = __t.wrapping_mul(__t2);
unsafe {
*$d = __t3 as u32;
}
__t.checked_mul(__t2).is_none() || ((__t3 as u32) as i128) != __t3
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
($ptr:expr) => {
{
let mut __v: *mut drm_display_mode;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!() as isize)) }) as *mut ()) as *mut drm_display_mode;
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
macro_rules! container_of__9 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__10!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! do_div__3 {
($n:expr, $base:expr) => {
let mut __v: u32;
let __base: u32 = ($base) as u32;
let __rem: u32 = ((($n) as u64) % (__base as u64)) as u32;
$n = (($n) as u64) / (__base as u64);
__v = __rem;
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
macro_rules! do_div__4 {
($n:expr, $base:expr) => {
let mut __v: u32;
let __base: u32 = $base;
let __rem: u32 = ((($n) as u64) % (__base as u64)) as u32;
$n = (($n) as u64) / (__base as u64);
__v = __rem;
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
macro_rules! do_div__2 {
($n:expr, $base:expr) => {
let mut __v: u32;
let __v2: u32 = $base;
let __v3: u32 = ((($n) as u64) % (__v2 as u64)) as u32;
$n = (($n) as u64) / (__v2 as u64);
__v = __v3;
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__3 {
() => {
((1i32.wrapping_neg() as u16) as i32) < 1
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__6 {
() => {
(1i32.wrapping_neg() as *mut u8) < (1i32 as *mut u8)
};
}

macro_rules! kcsan_check_atomic_read {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 4i32);
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
macro_rules! list_entry__4 {
($ptr:expr) => {
container_of__8!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__6 {
($ptr:expr, $type:ty) => {
container_of__9!($ptr, $type)
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

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry {
($pos:expr, $member:ident) => {
list_entry__4!(unsafe { (*$pos).$member.next })
};
}

macro_rules! offsetof__10 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, head) as u64
};
}

macro_rules! offsetof__4 {
() => {
core::mem::offset_of!(drm_display_mode, head) as u64
};
}

macro_rules! sized_strscpy_pad__2 {
($dest:expr, $src:expr, $count:expr) => {
let mut __v: i64;
let __dst: *mut u8 = $dest;
let __src: *const u8 = $src;
let __count: u64 = $count;
let __wrote: i64 = unsafe { sized_strscpy(__dst, __src, __count) };
if __wrote >= 0 && (__wrote as u64) < __count {
unsafe { memset(__dst.wrapping_offset(__wrote as isize).wrapping_offset(1 as isize) as *mut (), 0i32, __count.wrapping_sub(__wrote as u64).wrapping_sub(1)) };
}
__v = __wrote;
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

macro_rules! test_bit__2 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_70b71f185ebb8c4a($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
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

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut kstat: kernel_stat;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn iomem_get_mapping() -> *mut address_space;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __compiletime_assert_14();
    fn __compiletime_assert_17();
    fn __bad_size_call_parameter();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn strlen(arg0: *const u8) -> u64;
    fn strcmp(arg0: *const u8, arg1: *const u8) -> i32;
    fn sized_strscpy(arg0: *mut u8, arg1: *const u8, arg2: u64) -> i64;
    fn strncmp(arg0: *const u8, arg1: *const u8, arg2: u64) -> i32;
    fn strchr(arg0: *const u8, arg1: i32) -> *mut u8;
    fn strnchr(arg0: *const u8, arg1: u64, arg2: i32) -> *mut u8;
    fn simple_strtol(arg0: *const u8, arg1: *mut *mut u8, arg2: u32) -> i64;
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn __compiletime_assert_69();
    fn mutex_is_locked(arg0: *mut mutex) -> bool;
    fn kfree(arg0: *const ());
    fn __compiletime_assert_220();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_395();
    fn __compiletime_assert_416();
    fn acpi_create_platform_device(arg0: *mut acpi_device, arg1: *const property_entry) -> *mut platform_device;
    fn list_sort(arg0: *mut (), arg1: *mut list_head, arg2: Option<unsafe extern "C" fn(*mut (), *const list_head, *const list_head) -> i32>);
    fn drm_modeset_lock(arg0: *mut drm_modeset_lock, arg1: *mut drm_modeset_acquire_ctx) -> i32;
    fn drm_get_tv_mode_from_name(arg0: *const u8, arg1: u64) -> i32;
    fn drm_match_cea_mode(arg0: *const drm_display_mode) -> u8;
    fn __drm_dev_dbg(arg0: *mut _ddebug, arg1: *const device, arg2: u32, arg3: *const u8, ...);
    fn drm_mode_fixup_1366x768(arg0: *mut drm_display_mode);
    fn __compiletime_assert_671();
    fn __compiletime_assert_674();
    fn __compiletime_assert_677();
    fn __compiletime_assert_680();
    fn __compiletime_assert_701();
}

unsafe extern "llvm-intrinsic" {
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
fn isdigit(c: i32) -> i32 {
(48 <= c && c <= 57) as i32
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


fn __list_add_valid(new: *mut list_head, prev: *mut list_head, next: *mut list_head) -> bool {
true
}

fn __list_del_entry_valid(entry: *mut list_head) -> bool {
true
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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

fn list_move_tail(list: *mut list_head, head: *mut list_head) {
__list_del_entry(list);
list_add_tail(list, head);
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
    __slate_bits_0: __slate_bits::__SlateBits106U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits107U0,
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


// 4th level page in 5-level paging case
// 3rd level page
// PMD_SHIFT determines the size of the area a middle-level
// page table can map
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
struct io_bitmap {
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
#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

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
// strscpy_pad() - Copy a C-string into a sized buffer
// @dst: Where to copy the string to
// @src: Where to copy the string from
// @...: Size of destination buffer
//
// Copy the string, or as much of it as fits, into the dest buffer. The
// behavior is undefined if the string buffers overlap. The destination
// buffer is always %NUL terminated, unless it's zero-sized.
//
// If the source string is shorter than the destination buffer, the
// remaining bytes in the buffer will be filled with %NUL bytes.
//
// For full explanation of why you may want to consider using the
// 'strscpy' functions please see the function docstring for strscpy().
//
// Returns:
// * The number of characters copied (not including the trailing %NULs)
// * -E2BIG if count is 0 or @src was truncated.
#[inline(always)]
fn str_has_prefix(str: *const u8, prefix: *const u8) -> u64 {
let len: u64 = unsafe { strlen(prefix) };
if (unsafe { strncmp(str, prefix, len) }) == 0 { len } else { 0 }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
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
    __slate_bits_0: __slate_bits::__SlateBits126U0,
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
struct mutex_waiter {
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
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
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
struct futex_private_hash {
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
struct sem_undo_list {
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct timerqueue_head {
    rb_root: rb_root_cached,
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
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
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
struct k_sigaction {
    sa: sigaction,
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
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
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
struct mm_cid_pcpu {
    cid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
    __slate_bits_0: __slate_bits::__SlateBits293U0,
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
union __SlateRecord_sched_h_826 {
    ipi_mask_ptr: *mut cpumask,
    ipi_mask_val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_class {
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
struct gendisk {
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
struct vm_struct {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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
struct uprobe_xol_ops {
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
struct hprobe {
    state: u32,
    srcu_scp: *mut srcu_ctr,
    uprobe: *mut uprobe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobes_state {
    xol_area: *mut xol_area,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
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
    __slate_bits_0: __slate_bits::__SlateBits413U0,
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
    __slate_bits_0: __slate_bits::__SlateBits444U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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
    __slate_bits_0: __slate_bits::__SlateBits471U0,
    vm_ops: *const vm_operations_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
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
    __slate_bits_0: __slate_bits::__SlateBits473U0,
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
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits498U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits498U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits498U2,
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
    __slate_bits_3: __slate_bits::__SlateBits498U3,
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
struct wake_irq {
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
struct dev_pm_qos {
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
struct folio_queue {
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
struct __SlateRecord_capability_h_24 {
    val: u64,
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
struct mnt_idmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
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
struct sighand_struct {
    siglock: spinlock,
    count: refcount_struct,
    signalfd_wqh: wait_queue_head,
    action: [k_sigaction; 64],
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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
struct tty_struct {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct lockref {
    __slate_anon_0: __SlateRecord_lockref_h_26,
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
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
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
struct io_context {
    refcount: __SlateRecord_types_h_197,
    active_ref: __SlateRecord_types_h_190,
    ioprio: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct request_queue {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct posix_acl {
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
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
union __SlateRecord_dynamic_debug_h_55 {
    dd_key_true: static_key_true,
    dd_key_false: static_key_false,
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
struct module_param_attrs {
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
struct _ddebug {
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
    format: *const u8,
    __slate_bits_0: __slate_bits::__SlateBits696U0,
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
struct mod_tree_node {
    r#mod: *mut module,
    node: latch_tree_node,
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
struct device_dma_parameters {
    max_segment_size: u32,
    min_align_mask: u32,
    segment_boundary_mask: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits495U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_acpi_bus_h_249 {
    __slate_bits_0: __slate_bits::__SlateBits778U0,
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
    __slate_bits_0: __slate_bits::__SlateBits763U0,
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
struct __SlateRecord_acpi_bus_h_281 {
    __slate_bits_0: __slate_bits::__SlateBits782U0,
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
    __slate_bits_0: __slate_bits::__SlateBits769U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_flags {
    __slate_bits_0: __slate_bits::__SlateBits770U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_dir {
    entry: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_pnp_type {
    __slate_bits_0: __slate_bits::__SlateBits774U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_gpio_mapping {
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
    __slate_bits_0: __slate_bits::__SlateBits776U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct platform_device {
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
    __slate_bits_0: __slate_bits::__SlateBits780U0,
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
struct acpi_device_perf {
    state: i32,
    flags: acpi_device_perf_flags,
    state_count: i32,
    states: *mut acpi_device_perf_state,
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
struct acpi_device_wakeup_flags {
    __slate_bits_0: __slate_bits::__SlateBits784U0,
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
struct cgroup_taskset {
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
struct uid_gid_extent {
    first: u32,
    lower_first: u32,
    count: u32,
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
struct uid_gid_map {
    __slate_anon_0: __SlateRecord_user_namespace_h_26,
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
struct irq_data {
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
struct irq_domain_chip_generic {
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
    __slate_bits_0: __slate_bits::__SlateBits820U0,
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
struct drm_file {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_master {
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
    __slate_bits_0: __slate_bits::__SlateBits611U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_hdmi_h_406 {
    hdmi_type1: hdr_static_metadata,
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
struct __SlateRecord_hdmi_h_419 {
    r#type: u32,
    version: u8,
    length: u8,
    oui: u32,
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
struct fs_pin {
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
struct drm_connector_helper_funcs {
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
struct __SlateRecord_drm_connector_h_1088 {
    avi: drm_connector_hdmi_infoframe,
    hdr_drm: drm_connector_hdmi_infoframe,
    spd: drm_connector_hdmi_infoframe,
    hdmi: drm_connector_hdmi_infoframe,
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
struct ww_mutex {
    base: mutex,
    ctx: *mut ww_acquire_ctx,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_atomic_commit {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_commit {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_writeback_job {
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
struct __SlateRecord_drm_connector_h_2062 {
    lock: mutex,
    audio: drm_connector_hdmi_infoframe,
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
struct drm_format_info {
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
struct drm_framebuffer {
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
struct drm_mode_config_helper_funcs {
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
struct dma_fence {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_colorop {
}


// When adding a new stereo mode don't forget to adjust DRM_MODE_FLAGS_3D_MAX
// (define not exposed to user space).
// Aspect ratio flag bitmask (4 bits 22:19)
// Link Status options
// DRM_MODE_ROTATE_<degrees>
//
// Signals that a drm plane is been rotated <degrees> degrees in counter
// clockwise direction.
//
// This define is provided as a convenience, looking up the property id
// using the name->prop id lookup is the preferred method.
// DRM_MODE_ROTATE_MASK
//
// Bitmask used to look for drm plane rotations.
// DRM_MODE_REFLECT_<axis>
//
// Signals that the contents of a drm plane is reflected along the <axis> axis,
// in the same way as mirroring.
// See kerneldoc chapter "Plane Composition Properties" for more details.
//
// This define is provided as a convenience, looking up the property id
// using the name->prop id lookup is the preferred method.
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_modeinfo {
    clock: u32,
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
    vrefresh: u32,
    flags: u32,
    r#type: u32,
    name: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane_helper_funcs {
}


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
struct drm_self_refresh_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_pending_vblank_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_helper_funcs {
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
struct __SlateRecord_drm_print_h_181 {
    series: u32,
    counter: u32,
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
struct __SlateRecord_drm_modes_c_1812 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_luminance_range_info {
    min_luminance: u32,
    max_luminance: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_modes_c_2337 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_modes_c_2345 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_modes_c_2345_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_modes_c_2534 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_modes_c_2645 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_modes_c_2645_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_modes_c_2686 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_modes_c_2686_2 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits611U0 {
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
    pub struct __SlateBits444U0 {
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
    pub struct __SlateBits471U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits820U0 {
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
    pub struct __SlateBits71U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits107U0 {
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
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits498U0 {
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
    pub struct __SlateBits495U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits498U1 {
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
    pub struct __SlateBits498U2 {
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
    pub struct __SlateBits498U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits473U0 {
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
    pub struct __SlateBits106U0 {
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
    pub struct __SlateBits413U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits998U0 {
        #[bits(1)]
        pub color_mgmt_changed: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1010U0 {
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
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits769U0 {
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
    pub struct __SlateBits770U0 {
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
    pub struct __SlateBits774U0 {
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
    pub struct __SlateBits776U0 {
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
    pub struct __SlateBits778U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub explicit_set: u8,
        #[bits(6)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits784U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(1)]
        pub notifier_present: u8,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits780U0 {
        #[bits(8)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits782U0 {
        #[bits(1)]
        pub valid: u8,
        #[bits(7)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits763U0 {
        #[bits(1)]
        pub enabled: bool,
        #[bits(1)]
        pub demand_offline: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits696U0 {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_tile_group {
    refcount: kref,
    dev: *mut drm_device,
    id: i32,
    group_data: [u8; 9],
}


// halve V values for interlacing
// adjust timings for stereo modes
// don't adjust doublescan
// don't adjust doublescan
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
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_set_crtcinfo")]
extern "C" fn drm_mode_set_crtcinfo(p: *mut drm_display_mode, adjust_flags: i32) {
if p.is_null() {
return;
}
unsafe {
(*p).crtc_clock = (*p).clock;
}
unsafe {
(*p).crtc_hdisplay = (*p).hdisplay;
}
unsafe {
(*p).crtc_hsync_start = (*p).hsync_start;
}
unsafe {
(*p).crtc_hsync_end = (*p).hsync_end;
}
unsafe {
(*p).crtc_htotal = (*p).htotal;
}
unsafe {
(*p).crtc_hskew = (*p).hskew;
}
unsafe {
(*p).crtc_vdisplay = (*p).vdisplay;
}
unsafe {
(*p).crtc_vsync_start = (*p).vsync_start;
}
unsafe {
(*p).crtc_vsync_end = (*p).vsync_end;
}
unsafe {
(*p).crtc_vtotal = (*p).vtotal;
}
if (unsafe { (*p).flags }) & 16u32 != 0 {
if adjust_flags & 1i32 != 0 {
unsafe {
(*p).crtc_vdisplay = (((*p).crtc_vdisplay as i32) / 2) as u16;
}
unsafe {
(*p).crtc_vsync_start = (((*p).crtc_vsync_start as i32) / 2) as u16;
}
unsafe {
(*p).crtc_vsync_end = (((*p).crtc_vsync_end as i32) / 2) as u16;
}
unsafe {
(*p).crtc_vtotal = (((*p).crtc_vtotal as i32) / 2) as u16;
}
}
}
if adjust_flags & 4i32 == 0 {
if (unsafe { (*p).flags }) & 32u32 != 0 {
unsafe {
(*p).crtc_vdisplay = ((*p).crtc_vdisplay as i32).wrapping_mul(2) as u16;
}
unsafe {
(*p).crtc_vsync_start = ((*p).crtc_vsync_start as i32).wrapping_mul(2) as u16;
}
unsafe {
(*p).crtc_vsync_end = ((*p).crtc_vsync_end as i32).wrapping_mul(2) as u16;
}
unsafe {
(*p).crtc_vtotal = ((*p).crtc_vtotal as i32).wrapping_mul(2) as u16;
}
}
}
if adjust_flags & 8i32 == 0 {
if ((unsafe { (*p).vscan }) as i32) > 1 {
unsafe {
(*p).crtc_vdisplay = ((*p).crtc_vdisplay as i32).wrapping_mul((*p).vscan as i32) as u16;
}
unsafe {
(*p).crtc_vsync_start = ((*p).crtc_vsync_start as i32).wrapping_mul((*p).vscan as i32) as u16;
}
unsafe {
(*p).crtc_vsync_end = ((*p).crtc_vsync_end as i32).wrapping_mul((*p).vscan as i32) as u16;
}
unsafe {
(*p).crtc_vtotal = ((*p).crtc_vtotal as i32).wrapping_mul((*p).vscan as i32) as u16;
}
}
}
if adjust_flags & 2i32 != 0 {
let layout: u32 = (unsafe { (*p).flags }) & 507904u32;
match layout {
16384 => {
let __v37: i32 = unsafe { (*p).crtc_clock };
unsafe {
(*p).crtc_clock = __v37.wrapping_mul(2);
}
unsafe {
(*p).crtc_vdisplay = ((*p).crtc_vdisplay as i32).wrapping_add((*p).crtc_vtotal as i32) as u16;
}
unsafe {
(*p).crtc_vsync_start = ((*p).crtc_vsync_start as i32).wrapping_add((*p).crtc_vtotal as i32) as u16;
}
unsafe {
(*p).crtc_vsync_end = ((*p).crtc_vsync_end as i32).wrapping_add((*p).crtc_vtotal as i32) as u16;
}
unsafe {
(*p).crtc_vtotal = ((*p).crtc_vtotal as i32).wrapping_add((*p).crtc_vtotal as i32) as u16;
}
}
_ => {
}
}
}
let mut __v52: i32;
let __UNIQUE_ID_x__669: u16 = unsafe { (*p).crtc_vsync_start };
let __UNIQUE_ID_y__670: u16 = unsafe { (*p).crtc_vdisplay };
loop {
if __types_ok__3!(__UNIQUE_ID_x__669, __UNIQUE_ID_y__670) == 0 {
unsafe { __compiletime_assert_671() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v52 = if (__UNIQUE_ID_x__669 as i32) < (__UNIQUE_ID_y__670 as i32) { __UNIQUE_ID_x__669 as i32 } else { __UNIQUE_ID_y__670 as i32 };
unsafe {
(*p).crtc_vblank_start = __v52 as u16;
}
let mut __v54: i32;
let __UNIQUE_ID_x__672: u16 = unsafe { (*p).crtc_vsync_end };
let __UNIQUE_ID_y__673: u16 = unsafe { (*p).crtc_vtotal };
loop {
if __types_ok__3!(__UNIQUE_ID_x__672, __UNIQUE_ID_y__673) == 0 {
unsafe { __compiletime_assert_674() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v54 = if (__UNIQUE_ID_x__672 as i32) > (__UNIQUE_ID_y__673 as i32) { __UNIQUE_ID_x__672 as i32 } else { __UNIQUE_ID_y__673 as i32 };
unsafe {
(*p).crtc_vblank_end = __v54 as u16;
}
let mut __v56: i32;
let __UNIQUE_ID_x__675: u16 = unsafe { (*p).crtc_hsync_start };
let __UNIQUE_ID_y__676: u16 = unsafe { (*p).crtc_hdisplay };
loop {
if __types_ok__3!(__UNIQUE_ID_x__675, __UNIQUE_ID_y__676) == 0 {
unsafe { __compiletime_assert_677() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v56 = if (__UNIQUE_ID_x__675 as i32) < (__UNIQUE_ID_y__676 as i32) { __UNIQUE_ID_x__675 as i32 } else { __UNIQUE_ID_y__676 as i32 };
unsafe {
(*p).crtc_hblank_start = __v56 as u16;
}
let mut __v58: i32;
let __UNIQUE_ID_x__678: u16 = unsafe { (*p).crtc_hsync_end };
let __UNIQUE_ID_y__679: u16 = unsafe { (*p).crtc_htotal };
loop {
if __types_ok__3!(__UNIQUE_ID_x__678, __UNIQUE_ID_y__679) == 0 {
unsafe { __compiletime_assert_680() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v58 = if (__UNIQUE_ID_x__678 as i32) > (__UNIQUE_ID_y__679 as i32) { __UNIQUE_ID_x__678 as i32 } else { __UNIQUE_ID_y__679 as i32 };
unsafe {
(*p).crtc_hblank_end = __v58 as u16;
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_parse_command_line_for_connector")]
extern "C" fn drm_mode_parse_command_line_for_connector(mode_option: *const u8, connector: *const drm_connector, mode: *mut drm_cmdline_mode) -> bool {
let mut freestanding: bool = false;
let mut parse_extras: bool = false;
let mut bpp_off: u32 = 0;
let mut refresh_off: u32 = 0;
let mut options_off: u32 = 0;
let mut mode_end: u32 = 0;
let mut bpp_ptr: *const u8 = core::ptr::null::<u8>();
let mut refresh_ptr: *const u8 = core::ptr::null::<u8>();
let mut extra_ptr: *const u8 = core::ptr::null::<u8>();
let mut options_ptr: *const u8 = core::ptr::null::<u8>();
let mut bpp_end_ptr: *mut u8 = core::ptr::null_mut::<u8>();
let mut refresh_end_ptr: *mut u8 = core::ptr::null_mut::<u8>();
let mut len: i32 = 0;
let mut ret: i32 = 0;
unsafe { memset(mode as *mut (), 0i32, core::mem::size_of::<drm_cmdline_mode>() as u64) };
unsafe {
(*mode).panel_orientation = -1;
}
if mode_option.is_null() {
return false;
}
let name: *const u8 = mode_option;
// Locate the start of named options
options_ptr = (unsafe { strchr(name, 44i32) }) as *const u8;
if !options_ptr.is_null() {
options_off = ((unsafe { options_ptr.offset_from(name as *const u8) }) as i64) as u32;
} else {
options_off = (unsafe { strlen(name) }) as u32;
}
// Try to locate the bpp and refresh specifiers, if any
bpp_ptr = (unsafe { strnchr(name, options_off as u64, 45i32) }) as *const u8;
loop {
let __v0: bool = !bpp_ptr.is_null() && isdigit((unsafe { *bpp_ptr.wrapping_offset(1 as isize) }) as i32) == 0;
if !__v0 {
break;
}
bpp_ptr = (unsafe { strnchr(bpp_ptr.wrapping_offset(1 as isize), options_off as u64, 45i32) }) as *const u8;
}
if !bpp_ptr.is_null() {
bpp_off = ((unsafe { bpp_ptr.offset_from(name as *const u8) }) as i64) as u32;
}
refresh_ptr = (unsafe { strnchr(name, options_off as u64, 64i32) }) as *const u8;
if !refresh_ptr.is_null() {
refresh_off = ((unsafe { refresh_ptr.offset_from(name as *const u8) }) as i64) as u32;
}
// Locate the end of the name / resolution, and parse it
if !bpp_ptr.is_null() {
mode_end = bpp_off;
} else {
if !refresh_ptr.is_null() {
mode_end = refresh_off;
} else {
if !options_ptr.is_null() {
mode_end = options_off;
parse_extras = true;
} else {
mode_end = (unsafe { strlen(name) }) as u32;
parse_extras = true;
}
}
}
if mode_end == 0 {
return false;
}
ret = drm_mode_parse_cmdline_named_mode(name, mode_end, mode);
if ret < 0 {
return false;
}
// Having a mode that starts by a letter (and thus is named) and
// an at-sign (used to specify a refresh rate) is disallowed.
if ret != 0 && !refresh_ptr.is_null() {
return false;
}
// No named mode? Check for a normal mode argument, e.g. 1024x768
let __v2: bool = !unsafe { (*mode).specified } && isdigit((unsafe { *name.wrapping_offset(0 as isize) }) as i32) != 0;
if __v2 {
ret = drm_mode_parse_cmdline_res_mode(name, mode_end, parse_extras, connector, mode);
if ret != 0 {
return false;
}
unsafe {
(*mode).specified = true;
}
}
// No mode? Check for freestanding extras and/or options
if !unsafe { (*mode).specified } {
let len: u32 = (unsafe { strlen(mode_option) }) as u32;
if !bpp_ptr.is_null() || !refresh_ptr.is_null() {
return false;
}
// syntax error
if len == 1 || len >= 2 && ((unsafe { *mode_option.wrapping_offset(1 as isize) }) as i32) == 44 {
extra_ptr = mode_option;
} else {
options_ptr = mode_option.wrapping_offset(-(1 as isize));
}
freestanding = true;
}
if !bpp_ptr.is_null() {
ret = drm_mode_parse_cmdline_bpp(bpp_ptr, core::ptr::addr_of_mut!(bpp_end_ptr), mode);
if ret != 0 {
return false;
}
unsafe {
(*mode).bpp_specified = true;
}
}
if !refresh_ptr.is_null() {
ret = drm_mode_parse_cmdline_refresh(refresh_ptr, core::ptr::addr_of_mut!(refresh_end_ptr), mode);
if ret != 0 {
return false;
}
unsafe {
(*mode).refresh_specified = true;
}
}
// Locate the end of the bpp / refresh, and parse the extras
// if relevant
if !bpp_ptr.is_null() && !refresh_ptr.is_null() {
let mut __v3: *mut u8;
let __UNIQUE_ID_x__699: *mut u8 = bpp_end_ptr;
let __UNIQUE_ID_y__700: *mut u8 = refresh_end_ptr;
loop {
if __types_ok__7!(__UNIQUE_ID_x__699, __UNIQUE_ID_y__700) == 0 {
unsafe { __compiletime_assert_701() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__699 > __UNIQUE_ID_y__700 { __UNIQUE_ID_x__699 } else { __UNIQUE_ID_y__700 };
extra_ptr = __v3 as *const u8;
} else {
if !bpp_ptr.is_null() {
extra_ptr = bpp_end_ptr as *const u8;
} else {
if !refresh_ptr.is_null() {
extra_ptr = refresh_end_ptr as *const u8;
}
}
}
if !extra_ptr.is_null() {
if !options_ptr.is_null() {
len = ((unsafe { options_ptr.offset_from(extra_ptr as *const u8) }) as i64) as i32;
} else {
len = (unsafe { strlen(extra_ptr) }) as i32;
}
ret = drm_mode_parse_cmdline_extra(extra_ptr, len, freestanding, connector, mode);
if ret != 0 {
return false;
}
}
if !options_ptr.is_null() {
ret = drm_mode_parse_cmdline_options(options_ptr.wrapping_offset(1 as isize), freestanding, connector, mode);
if ret != 0 {
return false;
}
}
true
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
    __slate_bits_0: __slate_bits::__SlateBits998U0,
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
    __slate_bits_0: __slate_bits::__SlateBits1010U0,
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


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_debug_printmodeline: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_create: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_destroy: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_probed_add: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_analog_tv_mode: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_cvt_mode: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_gtf_mode_complex: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_gtf_mode: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_set_name: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_vrefresh: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_get_hv_timing: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_set_crtcinfo: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_copy: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_init: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_duplicate: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_match: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_equal: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_equal_no_clocks: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_equal_no_clocks_no_stereo: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_validate_driver: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_validate_size: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_validate_ycbcr420: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_prune_invalid: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_sort: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_connector_list_update: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_parse_command_line_for_connector: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_create_from_cmdline_mode: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_is_420_only: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_is_420_also: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {28} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_is_420: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {29} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_set_preferred_mode: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {30} ; .previous", sym drm_mode_debug_printmodeline, sym drm_mode_create, sym drm_mode_destroy, sym drm_mode_probed_add, sym drm_analog_tv_mode, sym drm_cvt_mode, sym drm_gtf_mode_complex, sym drm_gtf_mode, sym drm_mode_set_name, sym drm_mode_vrefresh, sym drm_mode_get_hv_timing, sym drm_mode_set_crtcinfo, sym drm_mode_copy, sym drm_mode_init, sym drm_mode_duplicate, sym drm_mode_match, sym drm_mode_equal, sym drm_mode_equal_no_clocks, sym drm_mode_equal_no_clocks_no_stereo, sym drm_mode_validate_driver, sym drm_mode_validate_size, sym drm_mode_validate_ycbcr420, sym drm_mode_prune_invalid, sym drm_mode_sort, sym drm_connector_list_update, sym drm_mode_parse_command_line_for_connector, sym drm_mode_create_from_cmdline_mode, sym drm_mode_is_420_only, sym drm_mode_is_420_also, sym drm_mode_is_420, sym drm_set_preferred_mode, options(att_syntax));

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

// for KHZ2PICOS()
// Helper for struct drm_device based logging.
// NOTE: this is deprecated in favor of drm_dbg_kms(NULL, ...).
/// drm_mode_debug_printmodeline - print a mode to dmesg
/// @mode: mode to print
///
/// Describe @mode using DRM_DEBUG.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_debug_printmodeline")]
extern "C" fn drm_mode_debug_printmodeline(mode: *const drm_display_mode) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 2u32, c"Modeline \"%s\": %d %d %d %d %d %d %d %d %d %d 0x%x 0x%x\n".as_ptr() as *const u8, (*mode).name.as_ptr() as *const u8, drm_mode_vrefresh(mode), (*mode).clock, (*mode).hdisplay as i32, (*mode).hsync_start as i32, (*mode).hsync_end as i32, (*mode).htotal as i32, (*mode).vdisplay as i32, (*mode).vsync_start as i32, (*mode).vsync_end as i32, (*mode).vtotal as i32, (*mode).r#type as i32, (*mode).flags) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_debug_printmodeline_656: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_display_mode)>, *mut ()>(Some(drm_mode_debug_printmodeline)) };

/// drm_mode_create - create a new display mode
/// @dev: DRM device
///
/// Create a new, cleared drm_display_mode with kzalloc, allocate an ID for it
/// and return it.
///
/// Returns:
/// Pointer to new mode on success, NULL on error.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_create")]
extern "C" fn drm_mode_create(dev: *mut drm_device) -> *mut drm_display_mode {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __v0: *mut drm_display_mode;
let __obj_size: u64 = size_mul(core::mem::size_of::<drm_display_mode>() as u64, 1u64);
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
__v0 = __v1 as *mut drm_display_mode;
let nmode: *mut drm_display_mode = __v0;
if nmode.is_null() {
return core::ptr::null_mut::<drm_display_mode>();
}
nmode
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_create_657: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device) -> *mut drm_display_mode>, *mut ()>(Some(drm_mode_create)) };

/// drm_mode_destroy - remove a mode
/// @dev: DRM device
/// @mode: mode to remove
///
/// Release @mode's unique ID, then free it @mode structure itself using kfree.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_destroy")]
extern "C" fn drm_mode_destroy(dev: *mut drm_device, mode: *mut drm_display_mode) {
if mode.is_null() {
return;
}
unsafe { kfree(mode as *const ()) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_destroy_658: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_display_mode)>, *mut ()>(Some(drm_mode_destroy)) };

/// drm_mode_probed_add - add a mode to a connector's probed_mode list
/// @connector: connector the new mode
/// @mode: mode data
///
/// Add @mode to @connector's probed_mode list for later use. This list should
/// then in a second step get filtered and all the modes actually supported by
/// the hardware moved to the @connector's modes list.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_probed_add")]
extern "C" fn drm_mode_probed_add(connector: *mut drm_connector, mode: *mut drm_display_mode) {
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!unsafe { mutex_is_locked(core::ptr::addr_of_mut!((*(*connector).dev).mode_config.mutex)) })) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("659: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 659b - ., 3; .popsection\n# {0}", const 659 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_mode_probed_add_1, sym __slate_asm_string_drm_mode_probed_add_2, const 114 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("660: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 660b - ., 4; .popsection\n# {0}", const 660 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
list_add_tail(unsafe { core::ptr::addr_of_mut!((*mode).head) }, unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) });
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_probed_add_661: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_display_mode)>, *mut ()>(Some(drm_mode_probed_add)) };

// The timings come from:
// - https://web.archive.org/web/20220406232708/http://www.kolumbus.fi/pami1/video/pal_ntsc.html
// - https://web.archive.org/web/20220406124914/http://martin.hinner.info/vga/pal.html
// - https://web.archive.org/web/20220609202433/http://www.batsocks.co.uk/readme/video_timing.htm
// I couldn't find the actual tolerance for the back porch, so let's
// just reuse the sync length ones.
#[repr(C)]
#[derive(Clone, Copy)]
struct analog_param_field {
    even: u32,
    odd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct analog_param_range {
    min: u32,
    typ: u32,
    max: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct analog_parameters {
    num_lines: u32,
    line_duration_ns: u32,
    hact_ns: analog_param_range,
    hfp_ns: analog_param_range,
    hslen_ns: analog_param_range,
    hbp_ns: analog_param_range,
    hblk_ns: analog_param_range,
    bt601_hfp: u32,
    vfp_lines: analog_param_field,
    vslen_lines: analog_param_field,
    vbp_lines: analog_param_field,
}


/// The front porch is actually 6 short sync
/// pulses for the even field, and 5 for the
/// odd field. Each sync takes half a life so
/// the odd field front porch is shorter by
/// half a line.
///
/// In progressive, we're supposed to use 6
/// pulses, so we're fine there
/// The vsync length is 5 long sync pulses,
/// each field taking half a line. We're
/// shorter for both fields by half a line.
///
/// In progressive, we're supposed to use 5
/// pulses, so we're off by half
/// a line.
///
/// In interlace, we're now off by half a line
/// for the even field and one line for the odd
/// field.
/// The back porch starts with post-equalizing
/// pulses, consisting in 5 short sync pulses
/// for the even field, 4 for the odd field. In
/// progressive, it's 5 short syncs.
///
/// In progressive, we thus have 2.5 lines,
/// plus the 0.5 line we were missing
/// previously, so we should use 3 lines.
///
/// In interlace, the even field is in the
/// exact same case than progressive. For the
/// odd field, we should be using 2 lines but
/// we're one line short, so we'll make up for
/// it here by using 3.
///
/// The entire blanking area is supposed to
/// take 25 lines, so we also need to account
/// for the rest of the blanking area that
/// can't be in either the front porch or sync
/// period.
static tv_modes_parameters: __SlateAlign16<[analog_parameters; 2]> = __SlateAlign16([analog_parameters { num_lines: 525u32, line_duration_ns: 63556u32, hact_ns: analog_param_range { min: 52456u32, typ: 52656u32, max: 52856u32 }, hfp_ns: analog_param_range { min: 1270u32, typ: 1500u32, max: 2220u32 }, hslen_ns: analog_param_range { min: 4600u32, typ: 4700u32, max: 4800u32 }, hbp_ns: analog_param_range { min: 4600u32, typ: 4700u32, max: 4800u32 }, hblk_ns: analog_param_range { min: 10700u32, typ: 10900u32, max: 11100u32 }, bt601_hfp: 16u32, vfp_lines: analog_param_field { even: 3u32, odd: 3u32 }, vslen_lines: analog_param_field { even: 3u32, odd: 3u32 }, vbp_lines: analog_param_field { even: 17u32, odd: 16u32 } }, analog_parameters { num_lines: 625u32, line_duration_ns: 64000u32, hact_ns: analog_param_range { min: 51850u32, typ: 51950u32, max: 52350u32 }, hfp_ns: analog_param_range { min: 1550u32, typ: 1650u32, max: 2050u32 }, hslen_ns: analog_param_range { min: 4500u32, typ: 4700u32, max: 4900u32 }, hbp_ns: analog_param_range { min: 5500u32, typ: 5700u32, max: 5900u32 }, hblk_ns: analog_param_range { min: 11650u32, typ: 12050u32, max: 12150u32 }, bt601_hfp: 12u32, vfp_lines: analog_param_field { even: 2u32, odd: 3u32 }, vslen_lines: analog_param_field { even: 3u32, odd: 3u32 }, vbp_lines: analog_param_field { even: 20u32, odd: 19u32 } }]);

fn fill_analog_mode(dev: *mut drm_device, mode: *mut drm_display_mode, params: *const analog_parameters, pixel_clock_hz: u64, hactive: u32, vactive: u32, interlace: bool) -> i32 {
let pixel_duration_ns: u64 = 1000000000 / pixel_clock_hz;
let mut htotal: u32 = 0;
let mut vtotal: u32 = 0;
let mut max_hact: u32 = 0;
let mut hact_duration_ns: u32 = 0;
let mut hblk: u32 = 0;
let mut hblk_duration_ns: u32 = 0;
let mut hfp: u32 = 0;
let mut hfp_duration_ns: u32 = 0;
let mut hslen: u32 = 0;
let mut hslen_duration_ns: u32 = 0;
let mut hbp: u32 = 0;
let mut hbp_duration_ns: u32 = 0;
let mut porches: u32 = 0;
let mut porches_duration_ns: u32 = 0;
let mut vfp: u32 = 0;
let mut vfp_min: u32 = 0;
let mut vbp: u32 = 0;
let mut vbp_min: u32 = 0;
let mut vslen: u32 = 0;
let mut bt601: bool = false;
let mut porches_rem: i32 = 0;
let mut result: u64 = 0;
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Generating a %ux%u%c, %u-line mode with a %lu kHz clock\n".as_ptr() as *const u8, hactive, vactive, if interlace { 105i32 } else { 112i32 }, (*params).num_lines, pixel_clock_hz / 1000) };
max_hact = (((unsafe { (*params).hact_ns.max }) as u64) / pixel_duration_ns) as u32;
if pixel_clock_hz == 13500000 && hactive > max_hact && hactive <= 720 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Trying to generate a BT.601 mode. Disabling checks.\n".as_ptr() as *const u8) };
bt601 = true;
}
// Our pixel duration is going to be round down by the division,
// so rounding up is probably going to introduce even more
// deviation.
result = ((unsafe { (*params).line_duration_ns }) as u64).wrapping_mul(pixel_clock_hz);
do_div__3!(result, 1000000000i64);
htotal = result as u32;
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Total Horizontal Number of Pixels: %u\n".as_ptr() as *const u8, htotal) };
hact_duration_ns = (hactive as u64).wrapping_mul(pixel_duration_ns) as u32;
if !bt601 && (hact_duration_ns < unsafe { (*params).hact_ns.min } || hact_duration_ns > unsafe { (*params).hact_ns.max }) {
while false {
}
unsafe { _dev_err(__drm_to_dev(dev as *const drm_device) as *const device, c"[drm] *ERROR* Invalid horizontal active area duration: %uns (min: %u, max %u)\n".as_ptr() as *const u8, hact_duration_ns, (*params).hact_ns.min, (*params).hact_ns.max) };
return 22i32.wrapping_neg();
}
hblk = htotal.wrapping_sub(hactive);
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Horizontal Blanking Period: %u\n".as_ptr() as *const u8, hblk) };
hblk_duration_ns = (hblk as u64).wrapping_mul(pixel_duration_ns) as u32;
if !bt601 && (hblk_duration_ns < unsafe { (*params).hblk_ns.min } || hblk_duration_ns > unsafe { (*params).hblk_ns.max }) {
while false {
}
unsafe { _dev_err(__drm_to_dev(dev as *const drm_device) as *const device, c"[drm] *ERROR* Invalid horizontal blanking duration: %uns (min: %u, max %u)\n".as_ptr() as *const u8, hblk_duration_ns, (*params).hblk_ns.min, (*params).hblk_ns.max) };
return 22i32.wrapping_neg();
}
hslen = (((unsafe { (*params).hslen_ns.typ }) as u64).wrapping_add(pixel_duration_ns).wrapping_sub(1) / pixel_duration_ns) as u32;
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Horizontal Sync Period: %u\n".as_ptr() as *const u8, hslen) };
hslen_duration_ns = (hslen as u64).wrapping_mul(pixel_duration_ns) as u32;
if !bt601 && (hslen_duration_ns < unsafe { (*params).hslen_ns.min } || hslen_duration_ns > unsafe { (*params).hslen_ns.max }) {
while false {
}
unsafe { _dev_err(__drm_to_dev(dev as *const drm_device) as *const device, c"[drm] *ERROR* Invalid horizontal sync duration: %uns (min: %u, max %u)\n".as_ptr() as *const u8, hslen_duration_ns, (*params).hslen_ns.min, (*params).hslen_ns.max) };
return 22i32.wrapping_neg();
}
porches = hblk.wrapping_sub(hslen);
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Remaining horizontal pixels for both porches: %u\n".as_ptr() as *const u8, porches) };
porches_duration_ns = (porches as u64).wrapping_mul(pixel_duration_ns) as u32;
if !bt601 && (porches_duration_ns > unsafe { (*params).hfp_ns.max }.wrapping_add(unsafe { (*params).hbp_ns.max }) || porches_duration_ns < unsafe { (*params).hfp_ns.min }.wrapping_add(unsafe { (*params).hbp_ns.min })) {
while false {
}
unsafe { _dev_err(__drm_to_dev(dev as *const drm_device) as *const device, c"[drm] *ERROR* Invalid horizontal porches duration: %uns\n".as_ptr() as *const u8, porches_duration_ns) };
return 22i32.wrapping_neg();
}
if bt601 {
hfp = unsafe { (*params).bt601_hfp };
} else {
let hfp_min: u32 = (((unsafe { (*params).hfp_ns.min }) as u64).wrapping_add(pixel_duration_ns).wrapping_sub(1) / pixel_duration_ns) as u32;
let hbp_min: u32 = (((unsafe { (*params).hbp_ns.min }) as u64).wrapping_add(pixel_duration_ns).wrapping_sub(1) / pixel_duration_ns) as u32;
let porches_rem: i32 = porches.wrapping_sub(hfp_min).wrapping_sub(hbp_min) as i32;
hfp = hfp_min.wrapping_add((porches_rem.wrapping_add(2).wrapping_sub(1) / 2) as u32);
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Horizontal Front Porch: %u\n".as_ptr() as *const u8, hfp) };
hfp_duration_ns = (hfp as u64).wrapping_mul(pixel_duration_ns) as u32;
if !bt601 && (hfp_duration_ns < unsafe { (*params).hfp_ns.min } || hfp_duration_ns > unsafe { (*params).hfp_ns.max }) {
while false {
}
unsafe { _dev_err(__drm_to_dev(dev as *const drm_device) as *const device, c"[drm] *ERROR* Invalid horizontal front porch duration: %uns (min: %u, max %u)\n".as_ptr() as *const u8, hfp_duration_ns, (*params).hfp_ns.min, (*params).hfp_ns.max) };
return 22i32.wrapping_neg();
}
hbp = porches.wrapping_sub(hfp);
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Horizontal Back Porch: %u\n".as_ptr() as *const u8, hbp) };
hbp_duration_ns = (hbp as u64).wrapping_mul(pixel_duration_ns) as u32;
if !bt601 && (hbp_duration_ns < unsafe { (*params).hbp_ns.min } || hbp_duration_ns > unsafe { (*params).hbp_ns.max }) {
while false {
}
unsafe { _dev_err(__drm_to_dev(dev as *const drm_device) as *const device, c"[drm] *ERROR* Invalid horizontal back porch duration: %uns (min: %u, max %u)\n".as_ptr() as *const u8, hbp_duration_ns, (*params).hbp_ns.min, (*params).hbp_ns.max) };
return 22i32.wrapping_neg();
}
if htotal != hactive.wrapping_add(hfp).wrapping_add(hslen).wrapping_add(hbp) {
return 22i32.wrapping_neg();
}
unsafe {
(*mode).clock = (pixel_clock_hz / 1000) as i32;
}
unsafe {
(*mode).hdisplay = hactive as u16;
}
unsafe {
(*mode).hsync_start = (((*mode).hdisplay as i32) as u32).wrapping_add(hfp) as u16;
}
unsafe {
(*mode).hsync_end = (((*mode).hsync_start as i32) as u32).wrapping_add(hslen) as u16;
}
unsafe {
(*mode).htotal = (((*mode).hsync_end as i32) as u32).wrapping_add(hbp) as u16;
}
if interlace {
vfp_min = unsafe { (*params).vfp_lines.even }.wrapping_add(unsafe { (*params).vfp_lines.odd });
vbp_min = unsafe { (*params).vbp_lines.even }.wrapping_add(unsafe { (*params).vbp_lines.odd });
vslen = unsafe { (*params).vslen_lines.even }.wrapping_add(unsafe { (*params).vslen_lines.odd });
} else {
// By convention, NTSC (aka 525/60) systems start with
// the even field, but PAL (aka 625/50) systems start
// with the odd one.
//
// PAL systems also have asymmetric timings between the
// even and odd field, while NTSC is symmetric.
//
// Moreover, if we want to create a progressive mode for
// PAL, we need to use the odd field timings.
//
// Since odd == even for NTSC, we can just use the odd
// one all the time to simplify the code a bit.
vfp_min = unsafe { (*params).vfp_lines.odd };
vbp_min = unsafe { (*params).vbp_lines.odd };
vslen = unsafe { (*params).vslen_lines.odd };
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Vertical Sync Period: %u\n".as_ptr() as *const u8, vslen) };
porches = unsafe { (*params).num_lines }.wrapping_sub(vactive).wrapping_sub(vslen);
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Remaining vertical pixels for both porches: %u\n".as_ptr() as *const u8, porches) };
porches_rem = porches.wrapping_sub(vfp_min).wrapping_sub(vbp_min) as i32;
vfp = vfp_min.wrapping_add((porches_rem / 2) as u32);
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Vertical Front Porch: %u\n".as_ptr() as *const u8, vfp) };
vbp = porches.wrapping_sub(vfp);
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Vertical Back Porch: %u\n".as_ptr() as *const u8, vbp) };
vtotal = vactive.wrapping_add(vfp).wrapping_add(vslen).wrapping_add(vbp);
if (unsafe { (*params).num_lines }) != vtotal {
while false {
}
unsafe { _dev_err(__drm_to_dev(dev as *const drm_device) as *const device, c"[drm] *ERROR* Invalid vertical total: %upx (expected %upx)\n".as_ptr() as *const u8, vtotal, (*params).num_lines) };
return 22i32.wrapping_neg();
}
unsafe {
(*mode).vdisplay = vactive as u16;
}
unsafe {
(*mode).vsync_start = (((*mode).vdisplay as i32) as u32).wrapping_add(vfp) as u16;
}
unsafe {
(*mode).vsync_end = (((*mode).vsync_start as i32) as u32).wrapping_add(vslen) as u16;
}
unsafe {
(*mode).vtotal = (((*mode).vsync_end as i32) as u32).wrapping_add(vbp) as u16;
}
if (((unsafe { (*mode).vtotal }) as i32) as u32) != unsafe { (*params).num_lines } {
return 22i32.wrapping_neg();
}
unsafe {
(*mode).r#type = 64u8;
}
unsafe {
(*mode).flags = 10u32;
}
if interlace {
unsafe {
(*mode).flags = (*mode).flags | 16u32;
}
}
drm_mode_set_name(mode);
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Generated mode \"%s\": %d %d %d %d %d %d %d %d %d %d 0x%x 0x%x\n".as_ptr() as *const u8, (*mode).name.as_mut_ptr() as *mut u8, drm_mode_vrefresh(mode as *const drm_display_mode), (*mode).clock, (*mode).hdisplay as i32, (*mode).hsync_start as i32, (*mode).hsync_end as i32, (*mode).htotal as i32, (*mode).vdisplay as i32, (*mode).vsync_start as i32, (*mode).vsync_end as i32, (*mode).vtotal as i32, (*mode).r#type as i32, (*mode).flags) };
0
}

/// drm_analog_tv_mode - create a display mode for an analog TV
/// @dev: drm device
/// @tv_mode: TV Mode standard to create a mode for. See DRM_MODE_TV_MODE_*.
/// @pixel_clock_hz: Pixel Clock Frequency, in Hertz
/// @hdisplay: hdisplay size
/// @vdisplay: vdisplay size
/// @interlace: whether to compute an interlaced mode
///
/// This function creates a struct drm_display_mode instance suited for
/// an analog TV output, for one of the usual analog TV modes. Where
/// this is DRM_MODE_TV_MODE_MONOCHROME, a 625-line mode will be created.
///
/// Note that @hdisplay is larger than the usual constraints for the PAL
/// and NTSC timings, and we'll choose to ignore most timings constraints
/// to reach those resolutions.
///
/// Returns:
/// A pointer to the mode, allocated with drm_mode_create(). Returns NULL
/// on error.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_analog_tv_mode")]
extern "C" fn drm_analog_tv_mode(dev: *mut drm_device, tv_mode: u32, pixel_clock_hz: u64, hdisplay: u32, vdisplay: u32, interlace: bool) -> *mut drm_display_mode {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut analog: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u32 = core::ptr::addr_of_mut!(analog) as *mut u32;
let mut mode: core::mem::MaybeUninit<*mut drm_display_mode> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut drm_display_mode = core::ptr::addr_of_mut!(mode) as *mut *mut drm_display_mode;
unsafe {
'__join_4: {
'__join_10: {
'__join_11: {
'__join_12: {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __t0: u32 = tv_mode as u32;
if __t0 == 0 {
break '__join_12;
} else {
if __t0 == 1 {
break '__join_12;
} else {
if __t0 == 2 {
break '__join_11;
} else {
if __t0 == 4 {
break '__join_10;
} else {
if __t0 != 3 {
if __t0 != 5 {
if __t0 != 6 {
if __t0 != 7 {
return core::ptr::null_mut::<drm_display_mode>();
}
}
}
}
}
}
}
}
*__slate_slot_1 = 1;
break '__join_4;
}
}
}
*__slate_slot_1 = 0;
}
*__slate_slot_2 = drm_mode_create(dev);
if (*__slate_slot_2).is_null() {
return core::ptr::null_mut::<drm_display_mode>();
} else {
*__slate_slot_0 = fill_analog_mode(dev, *__slate_slot_2, (core::ptr::addr_of!(tv_modes_parameters.0) as *const analog_parameters).wrapping_offset((*__slate_slot_1 as u32) as isize), pixel_clock_hz, hdisplay, vdisplay, interlace);
if *__slate_slot_0 != 0 {
drm_mode_destroy(dev, *__slate_slot_2);
return core::ptr::null_mut::<drm_display_mode>();
} else {
return *__slate_slot_2;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_analog_tv_mode_662: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, u32, u64, u32, u32, bool) -> *mut drm_display_mode>, *mut ()>(Some(drm_analog_tv_mode)) };

/// drm_cvt_mode -create a modeline based on the CVT algorithm
/// @dev: drm device
/// @hdisplay: hdisplay size
/// @vdisplay: vdisplay size
/// @vrefresh: vrefresh rate
/// @reduced: whether to use reduced blanking
/// @interlaced: whether to compute an interlaced mode
/// @margins: whether to add margins (borders)
///
/// This function is called to generate the modeline based on CVT algorithm
/// according to the hdisplay, vdisplay, vrefresh.
/// It is based from the VESA(TM) Coordinated Video Timing Generator by
/// Graham Loveridge April 9, 2003 available at
/// http://www.elo.utfsm.cl/~elo212/docs/CVTd6r1.xls
///
/// And it is copied from xf86CVTmode in xserver/hw/xfree86/modes/xf86cvt.c.
/// What I have done is to translate it by using integer calculation.
///
/// Returns:
/// The modeline based on the CVT algorithm stored in a drm_display_mode object.
/// The display mode object is allocated with drm_mode_create(). Returns NULL
/// when no mode could be allocated.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_cvt_mode")]
extern "C" fn drm_cvt_mode(dev: *mut drm_device, hdisplay: i32, vdisplay: i32, mut vrefresh: i32, reduced: bool, interlaced: bool, margins: bool) -> *mut drm_display_mode {
// 1) top/bottom margin size (% of height) - default: 1.8,
// 2) character cell horizontal granularity (pixels) - default 8
// 3) Minimum vertical porch (lines) - default 3
// 4) Minimum number of vertical back porch lines - default 6
// Pixel Clock step (kHz)
let mut vfieldrate: u32 = 0;
let mut hperiod: u32 = 0;
let mut hdisplay_rnd: i32 = 0;
// determine the left&right borders
let mut hmargin: i32 = 0;
let mut vdisplay_rnd: i32 = 0;
// find the top & bottom borders
let mut vmargin: i32 = 0;
let mut vsync: i32 = 0;
let mut interlace: i32 = 0;
let mut tmp: u64 = 0;
if hdisplay == 0 || vdisplay == 0 {
return core::ptr::null_mut::<drm_display_mode>();
}
// allocate the drm_display_mode structure. If failure, we will
// return directly
let drm_mode: *mut drm_display_mode = drm_mode_create(dev);
if drm_mode.is_null() {
return core::ptr::null_mut::<drm_display_mode>();
}
// the CVT default refresh rate is 60Hz
if vrefresh == 0 {
vrefresh = 60;
}
// the required field fresh rate
if interlaced {
vfieldrate = vrefresh.wrapping_mul(2) as u32;
} else {
vfieldrate = vrefresh as u32;
}
// horizontal pixels
hdisplay_rnd = hdisplay.wrapping_sub(hdisplay % 8);
if margins {
hmargin = hdisplay_rnd.wrapping_mul(18) / 1000;
hmargin = hmargin.wrapping_sub(hmargin % 8);
}
// find the total active pixels
unsafe {
(*drm_mode).hdisplay = hdisplay_rnd.wrapping_add(2i32.wrapping_mul(hmargin)) as u16;
}
// find the number of lines per field
if interlaced {
vdisplay_rnd = vdisplay / 2;
} else {
vdisplay_rnd = vdisplay;
}
if margins {
vmargin = vdisplay_rnd.wrapping_mul(18) / 1000;
}
unsafe {
(*drm_mode).vdisplay = vdisplay.wrapping_add(2i32.wrapping_mul(vmargin)) as u16;
}
// Interlaced
if interlaced {
interlace = 1;
} else {
interlace = 0;
}
// Determine VSync Width from aspect ratio
if vdisplay % 3 == 0 && vdisplay.wrapping_mul(4) / 3 == hdisplay {
vsync = 4;
} else {
if vdisplay % 9 == 0 && vdisplay.wrapping_mul(16) / 9 == hdisplay {
vsync = 5;
} else {
if vdisplay % 10 == 0 && vdisplay.wrapping_mul(16) / 10 == hdisplay {
vsync = 6;
} else {
if vdisplay % 4 == 0 && vdisplay.wrapping_mul(5) / 4 == hdisplay {
vsync = 7;
} else {
if vdisplay % 9 == 0 && vdisplay.wrapping_mul(15) / 9 == hdisplay {
vsync = 7;
} else {
vsync = 10;
}
}
}
}
}
// custom
if !reduced {
// simplify the GTF calculation
//
// 4) Minimum time of vertical sync + back porch interval (µs)
// default 550.0
let mut tmp1: i32 = 0;
let mut tmp2: i32 = 0;
// 3) Nominal HSync width (% of line period) - default 8
let mut hblank_percentage: u32 = 0;
let mut vsyncandback_porch: i32 = 0;
let mut vback_porch: i32 = 0;
let mut hblank: i32 = 0;
// estimated the horizontal period
tmp1 = 1000000000u32.wrapping_sub(550000u32.wrapping_mul(vfieldrate)) as i32;
tmp2 = vdisplay_rnd.wrapping_add(2i32.wrapping_mul(vmargin)).wrapping_add(3).wrapping_mul(2).wrapping_add(interlace);
hperiod = (tmp1.wrapping_mul(2) as u32) / (tmp2 as u32).wrapping_mul(vfieldrate);
tmp1 = (550000u32 / hperiod).wrapping_add(1) as i32;
// 9. Find number of lines in sync + backporch
if tmp1 < vsync.wrapping_add(3) {
vsyncandback_porch = vsync.wrapping_add(3);
} else {
vsyncandback_porch = tmp1;
}
// 10. Find number of lines in back porch
vback_porch = vsyncandback_porch.wrapping_sub(vsync);
unsafe {
(*drm_mode).vtotal = vdisplay_rnd.wrapping_add(2i32.wrapping_mul(vmargin)).wrapping_add(vsyncandback_porch).wrapping_add(3) as u16;
}
// 5) Definition of Horizontal blanking time limitation
//
// Gradient (%/kHz) - default 600
// Offset (%) - default 40
// Blanking time scaling factor - default 128
// Scaling factor weighting - default 20
// 12. Find ideal blanking duty cycle from formula
hblank_percentage = 30000u32.wrapping_sub(300u32.wrapping_mul(hperiod) / 1000);
// 13. Blanking time
if hblank_percentage < 20000u32 {
hblank_percentage = 20000u32;
}
hblank = ((((unsafe { (*drm_mode).hdisplay }) as i32) as u32).wrapping_mul(hblank_percentage) / 100000u32.wrapping_sub(hblank_percentage)) as i32;
hblank = hblank.wrapping_sub(hblank % 16i32);
// 14. find the total pixels per line
unsafe {
(*drm_mode).htotal = ((*drm_mode).hdisplay as i32).wrapping_add(hblank) as u16;
}
unsafe {
(*drm_mode).hsync_end = ((*drm_mode).hdisplay as i32).wrapping_add(hblank / 2) as u16;
}
unsafe {
(*drm_mode).hsync_start = ((*drm_mode).hsync_end as i32).wrapping_sub(((*drm_mode).htotal as i32).wrapping_mul(8) / 100) as u16;
}
unsafe {
(*drm_mode).hsync_start = ((*drm_mode).hsync_start as i32).wrapping_add(8i32.wrapping_sub(((*drm_mode).hsync_start as i32) % 8)) as u16;
}
// fill the Vsync values
unsafe {
(*drm_mode).vsync_start = ((*drm_mode).vdisplay as i32).wrapping_add(3) as u16;
}
unsafe {
(*drm_mode).vsync_end = ((*drm_mode).vsync_start as i32).wrapping_add(vsync) as u16;
}
} else {
// Reduced blanking
//
// Minimum vertical blanking interval time (µs)- default 460
// Fixed number of clocks for horizontal sync
// Fixed number of clocks for horizontal blanking
// Fixed number of lines for vertical front porch - default 3
let mut vbilines: i32 = 0;
let mut tmp1: i32 = 0;
let mut tmp2: i32 = 0;
// 8. Estimate Horizontal period.
tmp1 = 1000000000u32.wrapping_sub(460000u32.wrapping_mul(vfieldrate)) as i32;
tmp2 = vdisplay_rnd.wrapping_add(2i32.wrapping_mul(vmargin));
hperiod = (tmp1 as u32) / (tmp2 as u32).wrapping_mul(vfieldrate);
// 9. Find number of lines in vertical blanking
vbilines = (460000u32 / hperiod).wrapping_add(1) as i32;
// 10. Check if vertical blanking is sufficient
if vbilines < 3i32.wrapping_add(vsync).wrapping_add(6) {
vbilines = 3i32.wrapping_add(vsync).wrapping_add(6);
}
// 11. Find total number of lines in vertical field
unsafe {
(*drm_mode).vtotal = vdisplay_rnd.wrapping_add(2i32.wrapping_mul(vmargin)).wrapping_add(vbilines) as u16;
}
// 12. Find total number of pixels in a line
unsafe {
(*drm_mode).htotal = ((*drm_mode).hdisplay as i32).wrapping_add(160) as u16;
}
// Fill in HSync values
unsafe {
(*drm_mode).hsync_end = ((*drm_mode).hdisplay as i32).wrapping_add(160 / 2) as u16;
}
unsafe {
(*drm_mode).hsync_start = ((*drm_mode).hsync_end as i32).wrapping_sub(32) as u16;
}
// Fill in VSync values
unsafe {
(*drm_mode).vsync_start = ((*drm_mode).vdisplay as i32).wrapping_add(3) as u16;
}
unsafe {
(*drm_mode).vsync_end = ((*drm_mode).vsync_start as i32).wrapping_add(vsync) as u16;
}
}
// 15/13. Find pixel clock frequency (kHz for xf86)
tmp = (unsafe { (*drm_mode).htotal }) as u64; // perform intermediate calcs in u64
tmp = tmp.wrapping_mul(1000000u64);
do_div__4!(tmp, hperiod);
tmp = tmp.wrapping_sub(((unsafe { (*drm_mode).clock }) % 250) as u64);
unsafe {
(*drm_mode).clock = tmp as i32;
}
// 18/16. Find actual vertical frame frequency
//
// ignore - just set the mode flag for interlaced
if interlaced {
unsafe {
(*drm_mode).vtotal = ((*drm_mode).vtotal as i32).wrapping_mul(2) as u16;
}
unsafe {
(*drm_mode).flags = (*drm_mode).flags | 16u32;
}
}
// Fill the mode line name
drm_mode_set_name(drm_mode);
if reduced {
unsafe {
(*drm_mode).flags = (*drm_mode).flags | 9u32;
}
} else {
unsafe {
(*drm_mode).flags = (*drm_mode).flags | 6u32;
}
}
drm_mode
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_cvt_mode_663: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, i32, i32, i32, bool, bool, bool) -> *mut drm_display_mode>, *mut ()>(Some(drm_cvt_mode)) };

/// drm_gtf_mode_complex - create the modeline based on the full GTF algorithm
/// @dev: drm device
/// @hdisplay: hdisplay size
/// @vdisplay: vdisplay size
/// @vrefresh: vrefresh rate.
/// @interlaced: whether to compute an interlaced mode
/// @margins: desired margin (borders) size
/// @GTF_M: extended GTF formula parameters
/// @GTF_2C: extended GTF formula parameters
/// @GTF_K: extended GTF formula parameters
/// @GTF_2J: extended GTF formula parameters
///
/// GTF feature blocks specify C and J in multiples of 0.5, so we pass them
/// in here multiplied by two.  For a C of 40, pass in 80.
///
/// Returns:
/// The modeline based on the full GTF algorithm stored in a drm_display_mode object.
/// The display mode object is allocated with drm_mode_create(). Returns NULL
/// when no mode could be allocated.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_gtf_mode_complex")]
extern "C" fn drm_gtf_mode_complex(dev: *mut drm_device, hdisplay: i32, vdisplay: i32, vrefresh: i32, interlaced: bool, margins: i32, GTF_M: i32, GTF_2C: i32, GTF_K: i32, GTF_2J: i32) -> *mut drm_display_mode {
// 1) top/bottom margin size (% of height) - default: 1.8,
// 2) character cell horizontal granularity (pixels) - default 8
// 3) Minimum vertical porch (lines) - default 3
// width of vsync in lines
// width of hsync as % of total line
// min time of vsync + back porch (microsec)
// C' and M' are part of the Blanking Duty Cycle computation
let mut hdisplay_rnd: u32 = 0;
let mut vdisplay_rnd: u32 = 0;
let mut vfieldrate_rqd: u32 = 0;
// 4. Find number of lines in Top margin:
let mut top_margin: i32 = 0;
let mut bottom_margin: i32 = 0;
let mut interlace: i32 = 0;
let mut hfreq_est: u32 = 0;
let mut vsync_plus_bp: i32 = 0;
let mut vback_porch: i32 = 0;
let mut vtotal_lines: u32 = 0;
let mut vfieldrate_est: u32 = 0;
let mut hperiod: u32 = 0;
let mut vfield_rate: u32 = 0;
let mut vframe_rate: u32 = 0;
let mut left_margin: i32 = 0;
let mut right_margin: i32 = 0;
let mut total_active_pixels: u32 = 0;
let mut ideal_duty_cycle: u32 = 0;
let mut hblank: u32 = 0;
let mut total_pixels: u32 = 0;
let mut pixel_freq: u32 = 0;
let mut hsync: i32 = 0;
let mut hfront_porch: i32 = 0;
//  36. Find the number of lines in the odd front porch period:
let vodd_front_porch_lines: i32 = 1;
let mut tmp1: u32 = 0;
let mut tmp2: u32 = 0;
if hdisplay == 0 || vdisplay == 0 {
return core::ptr::null_mut::<drm_display_mode>();
}
let drm_mode: *mut drm_display_mode = drm_mode_create(dev);
if drm_mode.is_null() {
return core::ptr::null_mut::<drm_display_mode>();
}
// 1. In order to give correct results, the number of horizontal
// pixels requested is first processed to ensure that it is divisible
// by the character size, by rounding it to the nearest character
// cell boundary:
hdisplay_rnd = (hdisplay.wrapping_add(8 / 2) / 8) as u32;
hdisplay_rnd = hdisplay_rnd.wrapping_mul(8);
// 2. If interlace is requested, the number of vertical lines assumed
// by the calculation must be halved, as the computation calculates
// the number of vertical lines per field.
if interlaced {
vdisplay_rnd = (vdisplay / 2) as u32;
} else {
vdisplay_rnd = vdisplay as u32;
}
// 3. Find the frame rate required:
if interlaced {
vfieldrate_rqd = vrefresh.wrapping_mul(2) as u32;
} else {
vfieldrate_rqd = vrefresh as u32;
}
if margins != 0 {
top_margin = (vdisplay_rnd.wrapping_mul(18).wrapping_add(500) / 1000) as i32;
}
// 5. Find number of lines in bottom margin:
bottom_margin = top_margin;
// 6. If interlace is required, then set variable interlace:
if interlaced {
interlace = 1;
} else {
interlace = 0;
}
// 7. Estimate the Horizontal frequency
tmp1 = 1000000u32.wrapping_sub(550u32.wrapping_mul(vfieldrate_rqd)) / 500;
tmp2 = vdisplay_rnd.wrapping_add(2i32.wrapping_mul(top_margin) as u32).wrapping_add(1).wrapping_mul(2).wrapping_add(interlace as u32);
hfreq_est = tmp2.wrapping_mul(1000).wrapping_mul(vfieldrate_rqd) / tmp1;
// 8. Find the number of lines in V sync + back porch
//
// [V SYNC+BP] = RINT(([MIN VSYNC+BP] * hfreq_est / 1000000))
vsync_plus_bp = (550u32.wrapping_mul(hfreq_est) / 1000) as i32;
vsync_plus_bp = vsync_plus_bp.wrapping_add(500) / 1000;
//  9. Find the number of lines in V back porch alone:
vback_porch = vsync_plus_bp.wrapping_sub(3);
//  10. Find the total number of lines in Vertical field period:
vtotal_lines = vdisplay_rnd.wrapping_add(top_margin as u32).wrapping_add(bottom_margin as u32).wrapping_add(vsync_plus_bp as u32).wrapping_add(1);
//  11. Estimate the Vertical field frequency:
vfieldrate_est = hfreq_est / vtotal_lines;
//  12. Find the actual horizontal period:
hperiod = 1000000 / vfieldrate_rqd.wrapping_mul(vtotal_lines);
//  13. Find the actual Vertical field frequency:
vfield_rate = hfreq_est / vtotal_lines;
//  14. Find the Vertical frame frequency:
if interlaced {
vframe_rate = vfield_rate / 2;
} else {
vframe_rate = vfield_rate;
}
//  15. Find number of pixels in left margin:
if margins != 0 {
left_margin = (hdisplay_rnd.wrapping_mul(18).wrapping_add(500) / 1000) as i32;
} else {
left_margin = 0;
}
// 16.Find number of pixels in right margin:
right_margin = left_margin;
// 17.Find total number of active pixels in image and left and right
total_active_pixels = hdisplay_rnd.wrapping_add(left_margin as u32).wrapping_add(right_margin as u32);
// 18.Find the ideal blanking duty cycle from blanking duty cycle
ideal_duty_cycle = (((GTF_2C.wrapping_sub(GTF_2J).wrapping_mul(GTF_K) / 256).wrapping_add(GTF_2J) / 2).wrapping_mul(1000) as u32).wrapping_sub(((GTF_K.wrapping_mul(GTF_M) / 256).wrapping_mul(1000000) as u32) / hfreq_est);
// 19.Find the number of pixels in the blanking time to the nearest
// double character cell:
hblank = total_active_pixels.wrapping_mul(ideal_duty_cycle) / 100000u32.wrapping_sub(ideal_duty_cycle);
hblank = hblank.wrapping_add(8) / 16u32;
hblank = hblank.wrapping_mul(2).wrapping_mul(8);
// 20.Find total number of pixels:
total_pixels = total_active_pixels.wrapping_add(hblank);
// 21.Find pixel clock frequency:
pixel_freq = total_pixels.wrapping_mul(hfreq_est) / 1000;
// Stage 1 computations are now complete; I should really pass
// the results to another function and do the Stage 2 computations,
// but I only need a few more values so I'll just append the
// computations here for now
//
// 17. Find the number of pixels in the horizontal sync period:
hsync = (8u32.wrapping_mul(total_pixels) / 100) as i32;
hsync = hsync.wrapping_add(8 / 2) / 8;
hsync = hsync.wrapping_mul(8);
// 18. Find the number of pixels in horizontal front porch period
hfront_porch = (hblank / 2).wrapping_sub(hsync as u32) as i32;
// finally, pack the results in the mode struct
unsafe {
(*drm_mode).hdisplay = hdisplay_rnd as u16;
}
unsafe {
(*drm_mode).hsync_start = hdisplay_rnd.wrapping_add(hfront_porch as u32) as u16;
}
unsafe {
(*drm_mode).hsync_end = ((*drm_mode).hsync_start as i32).wrapping_add(hsync) as u16;
}
unsafe {
(*drm_mode).htotal = total_pixels as u16;
}
unsafe {
(*drm_mode).vdisplay = vdisplay_rnd as u16;
}
unsafe {
(*drm_mode).vsync_start = vdisplay_rnd.wrapping_add(vodd_front_porch_lines as u32) as u16;
}
unsafe {
(*drm_mode).vsync_end = ((*drm_mode).vsync_start as i32).wrapping_add(3) as u16;
}
unsafe {
(*drm_mode).vtotal = vtotal_lines as u16;
}
unsafe {
(*drm_mode).clock = pixel_freq as i32;
}
if interlaced {
unsafe {
(*drm_mode).vtotal = ((*drm_mode).vtotal as i32).wrapping_mul(2) as u16;
}
unsafe {
(*drm_mode).flags = (*drm_mode).flags | 16u32;
}
}
drm_mode_set_name(drm_mode);
if GTF_M == 600 && GTF_2C == 80 && GTF_K == 128 && GTF_2J == 40 {
unsafe {
(*drm_mode).flags = 6u32;
}
} else {
unsafe {
(*drm_mode).flags = 9u32;
}
}
drm_mode
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_gtf_mode_complex_664: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, i32, i32, i32, bool, i32, i32, i32, i32, i32) -> *mut drm_display_mode>, *mut ()>(Some(drm_gtf_mode_complex)) };

/// drm_gtf_mode - create the modeline based on the GTF algorithm
/// @dev: drm device
/// @hdisplay: hdisplay size
/// @vdisplay: vdisplay size
/// @vrefresh: vrefresh rate.
/// @interlaced: whether to compute an interlaced mode
/// @margins: desired margin (borders) size
///
/// return the modeline based on GTF algorithm
///
/// This function is to create the modeline based on the GTF algorithm.
/// Generalized Timing Formula is derived from:
///
/// 	GTF Spreadsheet by Andy Morrish (1/5/97)
/// 	available at https://www.vesa.org
///
/// And it is copied from the file of xserver/hw/xfree86/modes/xf86gtf.c.
/// What I have done is to translate it by using integer calculation.
/// I also refer to the function of fb_get_mode in the file of
/// drivers/video/fbmon.c
///
/// Standard GTF parameters::
///
///     M = 600
///     C = 40
///     K = 128
///     J = 20
///
/// Returns:
/// The modeline based on the GTF algorithm stored in a drm_display_mode object.
/// The display mode object is allocated with drm_mode_create(). Returns NULL
/// when no mode could be allocated.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_gtf_mode")]
extern "C" fn drm_gtf_mode(dev: *mut drm_device, hdisplay: i32, vdisplay: i32, vrefresh: i32, interlaced: bool, margins: i32) -> *mut drm_display_mode {
drm_gtf_mode_complex(dev, hdisplay, vdisplay, vrefresh, interlaced, margins, 600i32, 80i32, 128i32, 40i32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_gtf_mode_665: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, i32, i32, i32, bool, i32) -> *mut drm_display_mode>, *mut ()>(Some(drm_gtf_mode)) };

/// drm_mode_set_name - set the name on a mode
/// @mode: name will be set in this mode
///
/// Set the name of @mode to a standard format which is <hdisplay>x<vdisplay>
/// with an optional 'i' suffix for interlaced modes.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_set_name")]
extern "C" fn drm_mode_set_name(mode: *mut drm_display_mode) {
let interlaced: bool = (unsafe { (*mode).flags }) & 16u32 != 0;
unsafe { snprintf((*mode).name.as_mut_ptr() as *mut u8, 32u64, c"%dx%d%s".as_ptr() as *const u8, (*mode).hdisplay as i32, (*mode).vdisplay as i32, if interlaced { c"i".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_set_name_666: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_display_mode)>, *mut ()>(Some(drm_mode_set_name)) };

/// drm_mode_vrefresh - get the vrefresh of a mode
/// @mode: mode
///
/// Returns:
/// @modes's vrefresh rate in Hz, rounded to the nearest integer.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_vrefresh")]
extern "C" fn drm_mode_vrefresh(mode: *const drm_display_mode) -> i32 {
let mut num: u32 = 1;
let mut den: u32 = 1;
if ((unsafe { (*mode).htotal }) as i32) == 0 || ((unsafe { (*mode).vtotal }) as i32) == 0 {
return 0;
}
if (unsafe { (*mode).flags }) & 16u32 != 0 {
num = num.wrapping_mul(2);
}
if (unsafe { (*mode).flags }) & 32u32 != 0 {
den = den.wrapping_mul(2);
}
if ((unsafe { (*mode).vscan }) as i32) > 1 {
den = den.wrapping_mul(((unsafe { (*mode).vscan }) as i32) as u32);
}
if check_mul_overflow__2!(unsafe { (*mode).clock }, num, core::ptr::addr_of_mut!(num)) {
return 0;
}
if check_mul_overflow__2!(((unsafe { (*mode).htotal }) as i32).wrapping_mul((unsafe { (*mode).vtotal }) as i32), den, core::ptr::addr_of_mut!(den)) {
return 0;
}
DIV_ROUND_CLOSEST_ULL__2!(mul_u32_u32(num, 1000u32), den) as i32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_vrefresh_667: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_display_mode) -> i32>, *mut ()>(Some(drm_mode_vrefresh)) };

/// drm_mode_get_hv_timing - Fetches hdisplay/vdisplay for given mode
/// @mode: mode to query
/// @hdisplay: hdisplay value to fill in
/// @vdisplay: vdisplay value to fill in
///
/// The vdisplay value will be doubled if the specified mode is a stereo mode of
/// the appropriate layout.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_get_hv_timing")]
extern "C" fn drm_mode_get_hv_timing(mode: *const drm_display_mode, hdisplay: *mut i32, vdisplay: *mut i32) {
let mut adjusted: drm_display_mode = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(adjusted) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 120usize);
}
drm_mode_init(core::ptr::addr_of_mut!(adjusted), mode);
drm_mode_set_crtcinfo(core::ptr::addr_of_mut!(adjusted), 14i32);
unsafe {
*hdisplay = adjusted.crtc_hdisplay as i32;
}
unsafe {
*vdisplay = adjusted.crtc_vdisplay as i32;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_get_hv_timing_668: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_display_mode, *mut i32, *mut i32)>, *mut ()>(Some(drm_mode_get_hv_timing)) };

// drm_mode_set_crtcinfo - set CRTC modesetting timing parameters
// @p: mode
// @adjust_flags: a combination of adjustment flags
//
// Setup the CRTC modesetting timing parameters for @p, adjusting if necessary.
//
// - The CRTC_INTERLACE_HALVE_V flag can be used to halve vertical timings of
//   interlaced modes.
// - The CRTC_STEREO_DOUBLE flag can be used to compute the timings for
//   buffers containing two eyes (only adjust the timings when needed, eg. for
//   "frame packing" or "side by side full").
// - The CRTC_NO_DBLSCAN and CRTC_NO_VSCAN flags request that adjustment *not*
//   be performed for doublescan and vscan > 1 modes respectively.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_set_crtcinfo_681: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_display_mode, i32)>, *mut ()>(Some(drm_mode_set_crtcinfo)) };

/// drm_mode_copy - copy the mode
/// @dst: mode to overwrite
/// @src: mode to copy
///
/// Copy an existing mode into another mode, preserving the
/// list head of the destination mode.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_copy")]
extern "C" fn drm_mode_copy(dst: *mut drm_display_mode, src: *const drm_display_mode) {
let head: list_head = unsafe { (*dst).head };
unsafe {
*dst = *src;
}
unsafe {
(*dst).head = head;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_copy_682: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_display_mode, *const drm_display_mode)>, *mut ()>(Some(drm_mode_copy)) };

/// drm_mode_init - initialize the mode from another mode
/// @dst: mode to overwrite
/// @src: mode to copy
///
/// Copy an existing mode into another mode, zeroing the
/// list head of the destination mode. Typically used
/// to guarantee the list head is not left with stack
/// garbage in on-stack modes.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_init")]
extern "C" fn drm_mode_init(dst: *mut drm_display_mode, src: *const drm_display_mode) {
unsafe { memset(dst as *mut (), 0i32, core::mem::size_of::<drm_display_mode>() as u64) };
drm_mode_copy(dst, src);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_init_683: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_display_mode, *const drm_display_mode)>, *mut ()>(Some(drm_mode_init)) };

/// drm_mode_duplicate - allocate and duplicate an existing mode
/// @dev: drm_device to allocate the duplicated mode for
/// @mode: mode to duplicate
///
/// Just allocate a new mode, copy the existing mode into it, and return
/// a pointer to it.  Used to create new instances of established modes.
///
/// Returns:
/// Pointer to duplicated mode on success, NULL on error.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_duplicate")]
extern "C" fn drm_mode_duplicate(dev: *mut drm_device, mode: *const drm_display_mode) -> *mut drm_display_mode {
let nmode: *mut drm_display_mode = drm_mode_create(dev);
if nmode.is_null() {
return core::ptr::null_mut::<drm_display_mode>();
}
drm_mode_copy(nmode, mode);
nmode
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_duplicate_684: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *const drm_display_mode) -> *mut drm_display_mode>, *mut ()>(Some(drm_mode_duplicate)) };

fn drm_mode_match_timings_vrr(mode1: *const drm_display_mode, mode2: *const drm_display_mode) -> bool {
let mode1_vsync_start_offset: i32 = ((unsafe { (*mode1).vtotal }) as i32).wrapping_sub((unsafe { (*mode1).vsync_start }) as i32);
let mode1_vsync_end_offset: i32 = ((unsafe { (*mode1).vtotal }) as i32).wrapping_sub((unsafe { (*mode1).vsync_end }) as i32);
let mode2_vsync_start_offset: i32 = ((unsafe { (*mode2).vtotal }) as i32).wrapping_sub((unsafe { (*mode2).vsync_start }) as i32);
let mode2_vsync_end_offset: i32 = ((unsafe { (*mode2).vtotal }) as i32).wrapping_sub((unsafe { (*mode2).vsync_end }) as i32);
((unsafe { (*mode1).hdisplay }) as i32) == ((unsafe { (*mode2).hdisplay }) as i32) && ((unsafe { (*mode1).hsync_start }) as i32) == ((unsafe { (*mode2).hsync_start }) as i32) && ((unsafe { (*mode1).hsync_end }) as i32) == ((unsafe { (*mode2).hsync_end }) as i32) && ((unsafe { (*mode1).htotal }) as i32) == ((unsafe { (*mode2).htotal }) as i32) && ((unsafe { (*mode1).hskew }) as i32) == ((unsafe { (*mode2).hskew }) as i32) && ((unsafe { (*mode1).vdisplay }) as i32) == ((unsafe { (*mode2).vdisplay }) as i32) && mode1_vsync_start_offset == mode2_vsync_start_offset && mode1_vsync_end_offset == mode2_vsync_end_offset && ((unsafe { (*mode1).vscan }) as i32) == ((unsafe { (*mode2).vscan }) as i32)
}

fn drm_mode_match_timings(mode1: *const drm_display_mode, mode2: *const drm_display_mode) -> bool {
((unsafe { (*mode1).hdisplay }) as i32) == ((unsafe { (*mode2).hdisplay }) as i32) && ((unsafe { (*mode1).hsync_start }) as i32) == ((unsafe { (*mode2).hsync_start }) as i32) && ((unsafe { (*mode1).hsync_end }) as i32) == ((unsafe { (*mode2).hsync_end }) as i32) && ((unsafe { (*mode1).htotal }) as i32) == ((unsafe { (*mode2).htotal }) as i32) && ((unsafe { (*mode1).hskew }) as i32) == ((unsafe { (*mode2).hskew }) as i32) && ((unsafe { (*mode1).vdisplay }) as i32) == ((unsafe { (*mode2).vdisplay }) as i32) && ((unsafe { (*mode1).vsync_start }) as i32) == ((unsafe { (*mode2).vsync_start }) as i32) && ((unsafe { (*mode1).vsync_end }) as i32) == ((unsafe { (*mode2).vsync_end }) as i32) && ((unsafe { (*mode1).vtotal }) as i32) == ((unsafe { (*mode2).vtotal }) as i32) && ((unsafe { (*mode1).vscan }) as i32) == ((unsafe { (*mode2).vscan }) as i32)
}

fn drm_mode_match_clock(mode1: *const drm_display_mode, mode2: *const drm_display_mode) -> bool {
// do clock check convert to PICOS
// so fb modes get matched the same
if (unsafe { (*mode1).clock }) != 0 && (unsafe { (*mode2).clock }) != 0 {
return KHZ2PICOS!(unsafe { (*mode1).clock }) == KHZ2PICOS!(unsafe { (*mode2).clock });
} else {
return (unsafe { (*mode1).clock }) == unsafe { (*mode2).clock };
}
}

fn drm_mode_match_flags(mode1: *const drm_display_mode, mode2: *const drm_display_mode) -> bool {
(unsafe { (*mode1).flags }) & 4294459391u32 == (unsafe { (*mode2).flags }) & 4294459391u32
}

fn drm_mode_match_3d_flags(mode1: *const drm_display_mode, mode2: *const drm_display_mode) -> bool {
(unsafe { (*mode1).flags }) & 507904u32 == (unsafe { (*mode2).flags }) & 507904u32
}

fn drm_mode_match_aspect_ratio(mode1: *const drm_display_mode, mode2: *const drm_display_mode) -> bool {
((unsafe { (*mode1).picture_aspect_ratio }) as u32) == ((unsafe { (*mode2).picture_aspect_ratio }) as u32)
}

/// drm_mode_match - test modes for (partial) equality
/// @mode1: first mode
/// @mode2: second mode
/// @match_flags: which parts need to match (DRM_MODE_MATCH_*)
///
/// Check to see if @mode1 and @mode2 are equivalent.
///
/// Returns:
/// True if the modes are (partially) equal, false otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_match")]
extern "C" fn drm_mode_match(mode1: *const drm_display_mode, mode2: *const drm_display_mode, match_flags: u32) -> bool {
if mode1.is_null() && mode2.is_null() {
return true;
}
if mode1.is_null() || mode2.is_null() {
return false;
}
let __v0: bool = match_flags & 32u32 != 0 && !drm_mode_match_timings_vrr(mode1, mode2);
if __v0 {
return false;
}
let __v1: bool = match_flags & 1u32 != 0 && !drm_mode_match_timings(mode1, mode2);
if __v1 {
return false;
}
let __v2: bool = match_flags & 2u32 != 0 && !drm_mode_match_clock(mode1, mode2);
if __v2 {
return false;
}
let __v3: bool = match_flags & 4u32 != 0 && !drm_mode_match_flags(mode1, mode2);
if __v3 {
return false;
}
let __v4: bool = match_flags & 8u32 != 0 && !drm_mode_match_3d_flags(mode1, mode2);
if __v4 {
return false;
}
let __v5: bool = match_flags & 16u32 != 0 && !drm_mode_match_aspect_ratio(mode1, mode2);
if __v5 {
return false;
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_match_685: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_display_mode, *const drm_display_mode, u32) -> bool>, *mut ()>(Some(drm_mode_match)) };

/// drm_mode_equal - test modes for equality
/// @mode1: first mode
/// @mode2: second mode
///
/// Check to see if @mode1 and @mode2 are equivalent.
///
/// Returns:
/// True if the modes are equal, false otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_equal")]
extern "C" fn drm_mode_equal(mode1: *const drm_display_mode, mode2: *const drm_display_mode) -> bool {
drm_mode_match(mode1, mode2, 31u32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_equal_686: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_display_mode, *const drm_display_mode) -> bool>, *mut ()>(Some(drm_mode_equal)) };

/// drm_mode_equal_no_clocks - test modes for equality
/// @mode1: first mode
/// @mode2: second mode
///
/// Check to see if @mode1 and @mode2 are equivalent, but
/// don't check the pixel clocks.
///
/// Returns:
/// True if the modes are equal, false otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_equal_no_clocks")]
extern "C" fn drm_mode_equal_no_clocks(mode1: *const drm_display_mode, mode2: *const drm_display_mode) -> bool {
drm_mode_match(mode1, mode2, 13u32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_equal_no_clocks_687: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_display_mode, *const drm_display_mode) -> bool>, *mut ()>(Some(drm_mode_equal_no_clocks)) };

/// drm_mode_equal_no_clocks_no_stereo - test modes for equality
/// @mode1: first mode
/// @mode2: second mode
///
/// Check to see if @mode1 and @mode2 are equivalent, but
/// don't check the pixel clocks nor the stereo layout.
///
/// Returns:
/// True if the modes are equal, false otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_equal_no_clocks_no_stereo")]
extern "C" fn drm_mode_equal_no_clocks_no_stereo(mode1: *const drm_display_mode, mode2: *const drm_display_mode) -> bool {
drm_mode_match(mode1, mode2, 5u32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_equal_no_clocks_no_stereo_688: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_display_mode, *const drm_display_mode) -> bool>, *mut ()>(Some(drm_mode_equal_no_clocks_no_stereo)) };

fn drm_mode_validate_basic(mode: *const drm_display_mode) -> i32 {
if ((unsafe { (*mode).r#type }) as i32) & -105i32 != 0 {
return -2;
}
if (unsafe { (*mode).flags }) & 4294446080u32 != 0 {
return -2;
}
if (unsafe { (*mode).flags }) & 507904u32 > 131072u32 {
return -2;
}
if (unsafe { (*mode).clock }) == 0 {
return 16;
}
if ((unsafe { (*mode).hdisplay }) as i32) == 0 || ((unsafe { (*mode).hsync_start }) as i32) < ((unsafe { (*mode).hdisplay }) as i32) || ((unsafe { (*mode).hsync_end }) as i32) < ((unsafe { (*mode).hsync_start }) as i32) || ((unsafe { (*mode).htotal }) as i32) < ((unsafe { (*mode).hsync_end }) as i32) {
return 3;
}
if ((unsafe { (*mode).vdisplay }) as i32) == 0 || ((unsafe { (*mode).vsync_start }) as i32) < ((unsafe { (*mode).vdisplay }) as i32) || ((unsafe { (*mode).vsync_end }) as i32) < ((unsafe { (*mode).vsync_start }) as i32) || ((unsafe { (*mode).vtotal }) as i32) < ((unsafe { (*mode).vsync_end }) as i32) {
return 4;
}
0
}

/// drm_mode_validate_driver - make sure the mode is somewhat sane
/// @dev: drm device
/// @mode: mode to check
///
/// First do basic validation on the mode, and then allow the driver
/// to check for device/driver specific limitations via the optional
/// &drm_mode_config_helper_funcs.mode_valid hook.
///
/// Returns:
/// The mode status
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_validate_driver")]
extern "C" fn drm_mode_validate_driver(dev: *mut drm_device, mode: *const drm_display_mode) -> i32 {
let status: i32 = drm_mode_validate_basic(mode);
if (status as i32) != 0 {
return status;
}
if (unsafe { (*(*dev).mode_config.funcs).mode_valid }) != None {
return unsafe { unsafe { (*(*dev).mode_config.funcs).mode_valid }.unwrap()(dev, mode) };
} else {
return 0;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_validate_driver_689: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *const drm_display_mode) -> i32>, *mut ()>(Some(drm_mode_validate_driver)) };

/// drm_mode_validate_size - make sure modes adhere to size constraints
/// @mode: mode to check
/// @maxX: maximum width
/// @maxY: maximum height
///
/// This function is a helper which can be used to validate modes against size
/// limitations of the DRM device/connector. If a mode is too big its status
/// member is updated with the appropriate validation failure code. The list
/// itself is not changed.
///
/// Returns:
/// The mode status
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_validate_size")]
extern "C" fn drm_mode_validate_size(mode: *const drm_display_mode, maxX: i32, maxY: i32) -> i32 {
if maxX > 0 && ((unsafe { (*mode).hdisplay }) as i32) > maxX {
return 11;
}
if maxY > 0 && ((unsafe { (*mode).vdisplay }) as i32) > maxY {
return 12;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_validate_size_690: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_display_mode, i32, i32) -> i32>, *mut ()>(Some(drm_mode_validate_size)) };

/// drm_mode_validate_ycbcr420 - add 'ycbcr420-only' modes only when allowed
/// @mode: mode to check
/// @connector: drm connector under action
///
/// This function is a helper which can be used to filter out any YCBCR420
/// only mode, when the source doesn't support it.
///
/// Returns:
/// The mode status
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_validate_ycbcr420")]
extern "C" fn drm_mode_validate_ycbcr420(mode: *const drm_display_mode, connector: *mut drm_connector) -> i32 {
let __v0: bool = !unsafe { (*connector).ycbcr_420_allowed } && drm_mode_is_420_only((unsafe { core::ptr::addr_of_mut!((*connector).display_info) }) as *const drm_display_info, mode);
if __v0 {
return 36;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_validate_ycbcr420_691: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_display_mode, *mut drm_connector) -> i32>, *mut ()>(Some(drm_mode_validate_ycbcr420)) };

static drm_mode_status_names: __SlateAlign16<[*const u8; 40]> = __SlateAlign16([(b"STALE\0".as_ptr() as *mut u8) as *const u8, (b"BAD\0".as_ptr() as *mut u8) as *const u8, (b"ERROR\0".as_ptr() as *mut u8) as *const u8, (b"OK\0".as_ptr() as *mut u8) as *const u8, (b"HSYNC\0".as_ptr() as *mut u8) as *const u8, (b"VSYNC\0".as_ptr() as *mut u8) as *const u8, (b"H_ILLEGAL\0".as_ptr() as *mut u8) as *const u8, (b"V_ILLEGAL\0".as_ptr() as *mut u8) as *const u8, (b"BAD_WIDTH\0".as_ptr() as *mut u8) as *const u8, (b"NOMODE\0".as_ptr() as *mut u8) as *const u8, (b"NO_INTERLACE\0".as_ptr() as *mut u8) as *const u8, (b"NO_DBLESCAN\0".as_ptr() as *mut u8) as *const u8, (b"NO_VSCAN\0".as_ptr() as *mut u8) as *const u8, (b"MEM\0".as_ptr() as *mut u8) as *const u8, (b"VIRTUAL_X\0".as_ptr() as *mut u8) as *const u8, (b"VIRTUAL_Y\0".as_ptr() as *mut u8) as *const u8, (b"MEM_VIRT\0".as_ptr() as *mut u8) as *const u8, (b"NOCLOCK\0".as_ptr() as *mut u8) as *const u8, (b"CLOCK_HIGH\0".as_ptr() as *mut u8) as *const u8, (b"CLOCK_LOW\0".as_ptr() as *mut u8) as *const u8, (b"CLOCK_RANGE\0".as_ptr() as *mut u8) as *const u8, (b"BAD_HVALUE\0".as_ptr() as *mut u8) as *const u8, (b"BAD_VVALUE\0".as_ptr() as *mut u8) as *const u8, (b"BAD_VSCAN\0".as_ptr() as *mut u8) as *const u8, (b"HSYNC_NARROW\0".as_ptr() as *mut u8) as *const u8, (b"HSYNC_WIDE\0".as_ptr() as *mut u8) as *const u8, (b"HBLANK_NARROW\0".as_ptr() as *mut u8) as *const u8, (b"HBLANK_WIDE\0".as_ptr() as *mut u8) as *const u8, (b"VSYNC_NARROW\0".as_ptr() as *mut u8) as *const u8, (b"VSYNC_WIDE\0".as_ptr() as *mut u8) as *const u8, (b"VBLANK_NARROW\0".as_ptr() as *mut u8) as *const u8, (b"VBLANK_WIDE\0".as_ptr() as *mut u8) as *const u8, (b"PANEL\0".as_ptr() as *mut u8) as *const u8, (b"INTERLACE_WIDTH\0".as_ptr() as *mut u8) as *const u8, (b"ONE_WIDTH\0".as_ptr() as *mut u8) as *const u8, (b"ONE_HEIGHT\0".as_ptr() as *mut u8) as *const u8, (b"ONE_SIZE\0".as_ptr() as *mut u8) as *const u8, (b"NO_REDUCED\0".as_ptr() as *mut u8) as *const u8, (b"NO_STEREO\0".as_ptr() as *mut u8) as *const u8, (b"NO_420\0".as_ptr() as *mut u8) as *const u8]);

#[unsafe(no_mangle)]
extern "C" fn drm_get_mode_status_name(status: i32) -> *const u8 {
let index: i32 = (status as i32).wrapping_add(3);
let mut __v0: i64;
let __ret_warn_on: i32 = (index < 0 || (index as u64) >= ARRAY_SIZE__52!()) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("692: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 692b - ., 3; .popsection\n# {0}", const 692 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_get_mode_status_name_1, sym __slate_asm_string_drm_get_mode_status_name_2, const 1812 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
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
if __v0 != 0 {
return c"".as_ptr() as *const u8;
}
unsafe { drm_mode_status_names.0[index as usize] }
}

/// drm_mode_prune_invalid - remove invalid modes from mode list
/// @dev: DRM device
/// @mode_list: list of modes to check
/// @verbose: be verbose about it
///
/// This helper function can be used to prune a display mode list after
/// validation has been completed. All modes whose status is not MODE_OK will be
/// removed from the list, and if @verbose the status code and mode name is also
/// printed to dmesg.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_prune_invalid")]
extern "C" fn drm_mode_prune_invalid(dev: *mut drm_device, mode_list: *mut list_head, verbose: bool) {
let mut mode: *mut drm_display_mode = unsafe { core::mem::zeroed() };
let mut t: *mut drm_display_mode = unsafe { core::mem::zeroed() };
mode = list_first_entry__4!(mode_list);
t = list_next_entry!(mode, head);
while list_entry_is_head!(mode, mode_list, head) == 0 {
if ((unsafe { (*mode).status }) as i32) != 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*mode).head) });
if ((unsafe { (*mode).r#type }) as i32) & 32i32 != 0 {
while false {
}
unsafe { _dev_warn(__drm_to_dev(dev as *const drm_device) as *const device, c"[drm] User-defined mode not supported: \"%s\": %d %d %d %d %d %d %d %d %d %d 0x%x 0x%x\n".as_ptr() as *const u8, (*mode).name.as_mut_ptr() as *mut u8, drm_mode_vrefresh(mode as *const drm_display_mode), (*mode).clock, (*mode).hdisplay as i32, (*mode).hsync_start as i32, (*mode).hsync_end as i32, (*mode).htotal as i32, (*mode).vdisplay as i32, (*mode).vsync_start as i32, (*mode).vsync_end as i32, (*mode).vtotal as i32, (*mode).r#type as i32, (*mode).flags) };
}
if verbose {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 2u32, c"Rejected mode: \"%s\": %d %d %d %d %d %d %d %d %d %d 0x%x 0x%x (%s)\n".as_ptr() as *const u8, (*mode).name.as_mut_ptr() as *mut u8, drm_mode_vrefresh(mode as *const drm_display_mode), (*mode).clock, (*mode).hdisplay as i32, (*mode).hsync_start as i32, (*mode).hsync_end as i32, (*mode).htotal as i32, (*mode).vdisplay as i32, (*mode).vsync_start as i32, (*mode).vsync_end as i32, (*mode).vtotal as i32, (*mode).r#type as i32, (*mode).flags, drm_get_mode_status_name((*mode).status)) };
}
drm_mode_destroy(dev, mode);
}
mode = t;
t = list_next_entry!(t, head);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_prune_invalid_694: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut list_head, bool)>, *mut ()>(Some(drm_mode_prune_invalid)) };

/// drm_mode_compare - compare modes for favorability
/// @priv: unused
/// @lh_a: list_head for first mode
/// @lh_b: list_head for second mode
///
/// Compare two modes, given by @lh_a and @lh_b, returning a value indicating
/// which is better.
///
/// Returns:
/// Negative if @lh_a is better than @lh_b, zero if they're equivalent, or
/// positive if @lh_b is better than @lh_a.
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_compare")]
extern "C" fn drm_mode_compare(r#priv: *mut (), lh_a: *const list_head, lh_b: *const list_head) -> i32 {
let a: *mut drm_display_mode = list_entry__6!(lh_a, drm_display_mode);
let b: *mut drm_display_mode = list_entry__6!(lh_b, drm_display_mode);
let mut diff: i32 = ((((unsafe { (*b).r#type }) as i32) & 8i32 != 0) as i32).wrapping_sub((((unsafe { (*a).r#type }) as i32) & 8i32 != 0) as i32);
if diff != 0 {
return diff;
}
diff = ((unsafe { (*b).hdisplay }) as i32).wrapping_mul((unsafe { (*b).vdisplay }) as i32).wrapping_sub(((unsafe { (*a).hdisplay }) as i32).wrapping_mul((unsafe { (*a).vdisplay }) as i32));
if diff != 0 {
return diff;
}
diff = drm_mode_vrefresh(b as *const drm_display_mode).wrapping_sub(drm_mode_vrefresh(a as *const drm_display_mode));
if diff != 0 {
return diff;
}
diff = unsafe { (*b).clock }.wrapping_sub(unsafe { (*a).clock });
diff
}

/// drm_mode_sort - sort mode list
/// @mode_list: list of drm_display_mode structures to sort
///
/// Sort @mode_list by favorability, moving good modes to the head of the list.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_sort")]
extern "C" fn drm_mode_sort(mode_list: *mut list_head) {
unsafe { list_sort(core::ptr::null_mut::<()>(), mode_list, Some(drm_mode_compare)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_sort_695: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut list_head)>, *mut ()>(Some(drm_mode_sort)) };

/// drm_connector_list_update - update the mode list for the connector
/// @connector: the connector to update
///
/// This moves the modes from the @connector probed_modes list
/// to the actual mode list. It compares the probed mode against the current
/// list and only adds different/new modes.
///
/// This is just a helper functions doesn't validate any modes itself and also
/// doesn't prune any invalid modes. Callers need to do that themselves.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_connector_list_update")]
extern "C" fn drm_connector_list_update(connector: *mut drm_connector) {
let mut pmode: *mut drm_display_mode = unsafe { core::mem::zeroed() };
let mut pt: *mut drm_display_mode = unsafe { core::mem::zeroed() };
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!unsafe { mutex_is_locked(core::ptr::addr_of_mut!((*(*connector).dev).mode_config.mutex)) })) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("696: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 696b - ., 3; .popsection\n# {0}", const 696 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_connector_list_update_1, sym __slate_asm_string_drm_connector_list_update_2, const 1914 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("697: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 697b - ., 4; .popsection\n# {0}", const 697 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
pmode = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) });
pt = list_next_entry!(pmode, head);
while list_entry_is_head!(pmode, unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) }, head) == 0 {
let mut mode: *mut drm_display_mode = unsafe { core::mem::zeroed() };
let mut found_it: bool = false;
// go through current modes checking for the new probed mode
mode = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*connector).modes) });
while list_entry_is_head!(mode, unsafe { core::ptr::addr_of_mut!((*connector).modes) }, head) == 0 {
if drm_mode_equal(pmode as *const drm_display_mode, mode as *const drm_display_mode) {
found_it = true;
// If the old matching mode is stale (ie. left over
// from a previous probe) just replace it outright.
// Otherwise just merge the type bits between all
// equal probed modes.
//
// If two probed modes are considered equal, pick the
// actual timings from the one that's marked as
// preferred (in case the match isn't 100%). If
// multiple or zero preferred modes are present, favor
// the mode added to the probed_modes list first.
if ((unsafe { (*mode).status }) as i32) == -3 {
drm_mode_copy(mode, pmode as *const drm_display_mode);
} else {
if ((unsafe { (*mode).r#type }) as i32) & 8i32 == 0 && ((unsafe { (*pmode).r#type }) as i32) & 8i32 != 0 {
unsafe {
(*pmode).r#type = (((*pmode).r#type as i32) | ((*mode).r#type as i32)) as u8;
}
drm_mode_copy(mode, pmode as *const drm_display_mode);
} else {
unsafe {
(*mode).r#type = (((*mode).r#type as i32) | ((*pmode).r#type as i32)) as u8;
}
}
}
list_del(unsafe { core::ptr::addr_of_mut!((*pmode).head) });
drm_mode_destroy(unsafe { (*connector).dev }, pmode);
break;
}
mode = list_next_entry!(mode, head);
}
if !found_it {
list_move_tail(unsafe { core::ptr::addr_of_mut!((*pmode).head) }, unsafe { core::ptr::addr_of_mut!((*connector).modes) });
}
pmode = pt;
pt = list_next_entry!(pt, head);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_connector_list_update_698: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector)>, *mut ()>(Some(drm_connector_list_update)) };

fn drm_mode_parse_cmdline_bpp(mut str: *const u8, end_ptr: *mut *mut u8, mode: *mut drm_cmdline_mode) -> i32 {
let mut bpp: u32 = 0;
if ((unsafe { *str.wrapping_offset(0 as isize) }) as i32) != 45 {
return 22i32.wrapping_neg();
}
let __v0: *const u8 = str;
str = __v0.wrapping_offset(1 as isize);
bpp = (unsafe { simple_strtol(str, end_ptr, 10u32) }) as u32;
if (unsafe { *end_ptr }) == (str as *mut u8) {
return 22i32.wrapping_neg();
}
unsafe {
(*mode).bpp = bpp as i32;
}
unsafe {
(*mode).bpp_specified = true;
}
0
}

fn drm_mode_parse_cmdline_refresh(mut str: *const u8, end_ptr: *mut *mut u8, mode: *mut drm_cmdline_mode) -> i32 {
let mut refresh: u32 = 0;
if ((unsafe { *str.wrapping_offset(0 as isize) }) as i32) != 64 {
return 22i32.wrapping_neg();
}
let __v0: *const u8 = str;
str = __v0.wrapping_offset(1 as isize);
refresh = (unsafe { simple_strtol(str, end_ptr, 10u32) }) as u32;
if (unsafe { *end_ptr }) == (str as *mut u8) {
return 22i32.wrapping_neg();
}
unsafe {
(*mode).refresh = refresh as i32;
}
unsafe {
(*mode).refresh_specified = true;
}
0
}

fn drm_mode_parse_cmdline_extra(str: *const u8, length: i32, freestanding: bool, connector: *const drm_connector, mode: *mut drm_cmdline_mode) -> i32 {
for i in 0..length {
match (unsafe { *str.wrapping_offset(i as isize) }) as i32 {
105 => {
if freestanding {
return 22i32.wrapping_neg();
}
unsafe {
(*mode).interlace = true;
}
}
109 => {
if freestanding {
return 22i32.wrapping_neg();
}
unsafe {
(*mode).margins = true;
}
}
68 => {
if ((unsafe { (*mode).force }) as u32) != 0 {
return 22i32.wrapping_neg();
}
if (unsafe { (*connector).connector_type }) != 2 && (unsafe { (*connector).connector_type }) != 12 {
unsafe {
(*mode).force = 2;
}
} else {
unsafe {
(*mode).force = 3;
}
}
}
100 => {
if ((unsafe { (*mode).force }) as u32) != 0 {
return 22i32.wrapping_neg();
}
unsafe {
(*mode).force = 1;
}
}
101 => {
if ((unsafe { (*mode).force }) as u32) != 0 {
return 22i32.wrapping_neg();
}
unsafe {
(*mode).force = 2;
}
}
_ => {
return 22i32.wrapping_neg();
}
}
}
0
}

fn drm_mode_parse_cmdline_res_mode(mut str: *const u8, length: u32, extras: bool, connector: *const drm_connector, mode: *mut drm_cmdline_mode) -> i32 {
let str_start: *const u8 = str;
let mut rb: bool = false;
let mut cvt: bool = false;
let mut xres: i32 = 0;
let mut yres: i32 = 0;
let mut remaining: i32 = 0;
let mut end_ptr: *mut u8 = unsafe { core::mem::zeroed() };
xres = (unsafe { simple_strtol(str, core::ptr::addr_of_mut!(end_ptr), 10u32) }) as i32;
if end_ptr == (str as *mut u8) {
return 22i32.wrapping_neg();
}
if ((unsafe { *end_ptr.wrapping_offset(0 as isize) }) as i32) != 120 {
return 22i32.wrapping_neg();
}
let __v0: *mut u8 = end_ptr;
end_ptr = __v0.wrapping_offset(1 as isize);
str = end_ptr as *const u8;
yres = (unsafe { simple_strtol(str, core::ptr::addr_of_mut!(end_ptr), 10u32) }) as i32;
if end_ptr == (str as *mut u8) {
return 22i32.wrapping_neg();
}
remaining = (length as i64).wrapping_sub((unsafe { end_ptr.offset_from(str_start as *mut u8) }) as i64) as i32;
if remaining < 0 {
return 22i32.wrapping_neg();
}
for i in 0..remaining {
match (unsafe { *end_ptr.wrapping_offset(i as isize) }) as i32 {
77 => {
cvt = true;
}
82 => {
rb = true;
}
_ => {
if extras {
let ret: i32 = drm_mode_parse_cmdline_extra(end_ptr.wrapping_offset(i as isize) as *const u8, 1i32, false, connector, mode);
if ret != 0 {
return ret;
}
} else {
return 22i32.wrapping_neg();
}
// Try to pass that to our extras parsing
// function to handle the case where the
// extras are directly after the resolution
}
}
}
unsafe {
(*mode).xres = xres;
}
unsafe {
(*mode).yres = yres;
}
unsafe {
(*mode).cvt = cvt;
}
unsafe {
(*mode).rb = rb;
}
0
}

fn drm_mode_parse_cmdline_int(delim: *const u8, int_ret: &mut u32) -> i32 {
let mut endp: *mut u8 = unsafe { core::mem::zeroed() };
// delim must point to the '=', otherwise it is a syntax error and
// if delim points to the terminating zero, then delim + 1 will point
// past the end of the string.
if ((unsafe { *delim }) as i32) != 61 {
return 22i32.wrapping_neg();
}
let value: *const u8 = delim.wrapping_offset(1 as isize);
unsafe {
*(int_ret as *mut u32) = simple_strtol(value, core::ptr::addr_of_mut!(endp), 10u32) as u32;
}
// Make sure we have parsed something
if endp == (value as *mut u8) {
return 22i32.wrapping_neg();
}
0
}

fn drm_mode_parse_panel_orientation(mut delim: *const u8, mode: *mut drm_cmdline_mode) -> i32 {
if ((unsafe { *delim }) as i32) != 61 {
return 22i32.wrapping_neg();
}
let value: *const u8 = delim.wrapping_offset(1 as isize);
delim = (unsafe { strchr(value, 44i32) }) as *const u8;
if delim.is_null() {
delim = value.wrapping_offset((unsafe { strlen(value) }) as isize);
}
if (unsafe { strncmp(value, c"normal".as_ptr() as *const u8, (delim.offset_from(value as *const u8) as i64) as u64) }) == 0 {
unsafe {
(*mode).panel_orientation = 0;
}
} else {
if (unsafe { strncmp(value, c"upside_down".as_ptr() as *const u8, (delim.offset_from(value as *const u8) as i64) as u64) }) == 0 {
unsafe {
(*mode).panel_orientation = 1;
}
} else {
if (unsafe { strncmp(value, c"left_side_up".as_ptr() as *const u8, (delim.offset_from(value as *const u8) as i64) as u64) }) == 0 {
unsafe {
(*mode).panel_orientation = 2;
}
} else {
if (unsafe { strncmp(value, c"right_side_up".as_ptr() as *const u8, (delim.offset_from(value as *const u8) as i64) as u64) }) == 0 {
unsafe {
(*mode).panel_orientation = 3;
}
} else {
return 22i32.wrapping_neg();
}
}
}
}
0
}

fn drm_mode_parse_tv_mode(mut delim: *const u8, mode: *mut drm_cmdline_mode) -> i32 {
let mut ret: i32 = 0;
if ((unsafe { *delim }) as i32) != 61 {
return 22i32.wrapping_neg();
}
let value: *const u8 = delim.wrapping_offset(1 as isize);
delim = (unsafe { strchr(value, 44i32) }) as *const u8;
if delim.is_null() {
delim = value.wrapping_offset((unsafe { strlen(value) }) as isize);
}
ret = unsafe { drm_get_tv_mode_from_name(value, (delim.offset_from(value as *const u8) as i64) as u64) };
if ret < 0 {
return ret;
}
unsafe {
(*mode).tv_mode_specified = true;
}
unsafe {
(*mode).tv_mode = ret as u32;
}
0
}

fn drm_mode_parse_cmdline_options(str: *const u8, freestanding: bool, connector: *const drm_connector, mode: *mut drm_cmdline_mode) -> i32 {
let mut deg: u32 = 0;
let mut margin: u32 = 0;
let mut rotation: u32 = 0;
let mut delim: *const u8 = unsafe { core::mem::zeroed() };
let mut option: *const u8 = unsafe { core::mem::zeroed() };
let mut sep: *const u8 = unsafe { core::mem::zeroed() };
option = str;
loop {
delim = (unsafe { strchr(option, 61i32) }) as *const u8;
if delim.is_null() {
delim = (unsafe { strchr(option, 44i32) }) as *const u8;
if delim.is_null() {
delim = option.wrapping_offset((unsafe { strlen(option) }) as isize);
}
}
if (unsafe { strncmp(option, c"rotate".as_ptr() as *const u8, (delim.offset_from(option as *const u8) as i64) as u64) }) == 0 {
if drm_mode_parse_cmdline_int(delim, unsafe { &mut (*core::ptr::addr_of_mut!(deg)) }) != 0 {
return 22i32.wrapping_neg();
}
match deg {
0 => {
rotation |= 1u32;
}
90 => {
rotation |= 2u32;
}
180 => {
rotation |= 4u32;
}
270 => {
rotation |= 8u32;
}
_ => {
return 22i32.wrapping_neg();
}
}
} else {
if (unsafe { strncmp(option, c"reflect_x".as_ptr() as *const u8, (delim.offset_from(option as *const u8) as i64) as u64) }) == 0 {
rotation |= 16u32;
} else {
if (unsafe { strncmp(option, c"reflect_y".as_ptr() as *const u8, (delim.offset_from(option as *const u8) as i64) as u64) }) == 0 {
rotation |= 32u32;
} else {
if (unsafe { strncmp(option, c"margin_right".as_ptr() as *const u8, (delim.offset_from(option as *const u8) as i64) as u64) }) == 0 {
if drm_mode_parse_cmdline_int(delim, unsafe { &mut (*core::ptr::addr_of_mut!(margin)) }) != 0 {
return 22i32.wrapping_neg();
}
unsafe {
(*mode).tv_margins.right = margin;
}
} else {
if (unsafe { strncmp(option, c"margin_left".as_ptr() as *const u8, (delim.offset_from(option as *const u8) as i64) as u64) }) == 0 {
if drm_mode_parse_cmdline_int(delim, unsafe { &mut (*core::ptr::addr_of_mut!(margin)) }) != 0 {
return 22i32.wrapping_neg();
}
unsafe {
(*mode).tv_margins.left = margin;
}
} else {
if (unsafe { strncmp(option, c"margin_top".as_ptr() as *const u8, (delim.offset_from(option as *const u8) as i64) as u64) }) == 0 {
if drm_mode_parse_cmdline_int(delim, unsafe { &mut (*core::ptr::addr_of_mut!(margin)) }) != 0 {
return 22i32.wrapping_neg();
}
unsafe {
(*mode).tv_margins.top = margin;
}
} else {
if (unsafe { strncmp(option, c"margin_bottom".as_ptr() as *const u8, (delim.offset_from(option as *const u8) as i64) as u64) }) == 0 {
if drm_mode_parse_cmdline_int(delim, unsafe { &mut (*core::ptr::addr_of_mut!(margin)) }) != 0 {
return 22i32.wrapping_neg();
}
unsafe {
(*mode).tv_margins.bottom = margin;
}
} else {
if (unsafe { strncmp(option, c"panel_orientation".as_ptr() as *const u8, (delim.offset_from(option as *const u8) as i64) as u64) }) == 0 {
if drm_mode_parse_panel_orientation(delim, mode) != 0 {
return 22i32.wrapping_neg();
}
} else {
if (unsafe { strncmp(option, c"tv_mode".as_ptr() as *const u8, (delim.offset_from(option as *const u8) as i64) as u64) }) == 0 {
if drm_mode_parse_tv_mode(delim, mode) != 0 {
return 22i32.wrapping_neg();
}
} else {
return 22i32.wrapping_neg();
}
}
}
}
}
}
}
}
}
sep = (unsafe { strchr(delim, 44i32) }) as *const u8;
option = sep.wrapping_offset(1 as isize);
if sep.is_null() {
break;
}
}
if rotation != 0 && freestanding {
return 22i32.wrapping_neg();
}
if rotation & 15u32 == 0 {
rotation |= 1u32;
}
// Make sure there is exactly one rotation defined
if !is_power_of_2((rotation & 15u32) as u64) {
return 22i32.wrapping_neg();
}
unsafe {
(*mode).rotation_reflection = rotation;
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_named_mode {
    name: *const u8,
    pixel_clock_khz: u32,
    xres: u32,
    yres: u32,
    flags: u32,
    tv_mode: u32,
}


static drm_named_modes: __SlateAlign16<[drm_named_mode; 4]> = __SlateAlign16([{
let mut __t0: drm_named_mode = unsafe { core::mem::zeroed() };
__t0.name = (b"NTSC\0".as_ptr() as *mut u8) as *const u8;
__t0.pixel_clock_khz = 13500u32;
__t0.xres = 720u32;
__t0.yres = 480u32;
__t0.flags = 16u32;
__t0.tv_mode = 0u32;
__t0
}, {
let mut __t1: drm_named_mode = unsafe { core::mem::zeroed() };
__t1.name = (b"NTSC-J\0".as_ptr() as *mut u8) as *const u8;
__t1.pixel_clock_khz = 13500u32;
__t1.xres = 720u32;
__t1.yres = 480u32;
__t1.flags = 16u32;
__t1.tv_mode = 2u32;
__t1
}, {
let mut __t2: drm_named_mode = unsafe { core::mem::zeroed() };
__t2.name = (b"PAL\0".as_ptr() as *mut u8) as *const u8;
__t2.pixel_clock_khz = 13500u32;
__t2.xres = 720u32;
__t2.yres = 576u32;
__t2.flags = 16u32;
__t2.tv_mode = 3u32;
__t2
}, {
let mut __t3: drm_named_mode = unsafe { core::mem::zeroed() };
__t3.name = (b"PAL-M\0".as_ptr() as *mut u8) as *const u8;
__t3.pixel_clock_khz = 13500u32;
__t3.xres = 720u32;
__t3.yres = 480u32;
__t3.flags = 16u32;
__t3.tv_mode = 4u32;
__t3
}]);

fn drm_mode_parse_cmdline_named_mode(name: *const u8, name_end: u32, cmdline_mode: *mut drm_cmdline_mode) -> i32 {
// We're sure we're a named mode at this point, iterate over the
// list of modes we're aware of.
let mut i: u32 = 0;
if name_end == 0 {
return 0;
}
// If the name starts with a digit, it's not a named mode
if isdigit((unsafe { *name.wrapping_offset(0 as isize) }) as i32) != 0 {
return 0;
}
// If there's an equal sign in the name, the command-line
// contains only an option and no mode.
if !unsafe { strnchr(name, name_end as u64, 61i32) }.is_null() {
return 0;
}
// The connection status extras can be set without a mode.
if name_end == 1 && (((unsafe { *name.wrapping_offset(0 as isize) }) as i32) == 100 || ((unsafe { *name.wrapping_offset(0 as isize) }) as i32) == 68 || ((unsafe { *name.wrapping_offset(0 as isize) }) as i32) == 101) {
return 0;
}
while (i as u64) < ARRAY_SIZE__23!() {
let mode: *const drm_named_mode = unsafe { core::ptr::addr_of!(drm_named_modes.0) as *const drm_named_mode }.wrapping_offset(i as isize);
let ret: i32 = str_has_prefix(name, unsafe { (*mode).name }) as i32;
if (ret as u32) == name_end {
unsafe { sized_strscpy((*cmdline_mode).name.as_mut_ptr() as *mut u8, (*mode).name, (core::mem::size_of::<[u8; 32]>() as u64).wrapping_add(__must_be_cstr__3!() as u64).wrapping_add(__must_be_cstr__16!() as u64)) };
unsafe {
(*cmdline_mode).pixel_clock = (*mode).pixel_clock_khz;
}
unsafe {
(*cmdline_mode).xres = (*mode).xres as i32;
}
unsafe {
(*cmdline_mode).yres = (*mode).yres as i32;
}
unsafe {
(*cmdline_mode).interlace = (*mode).flags & 16u32 != 0;
}
unsafe {
(*cmdline_mode).tv_mode = (*mode).tv_mode as u32;
}
unsafe {
(*cmdline_mode).tv_mode_specified = true;
}
unsafe {
(*cmdline_mode).specified = true;
}
return 1;
}
i = i.wrapping_add(1);
}
22i32.wrapping_neg()
}

// drm_mode_parse_command_line_for_connector - parse command line modeline for connector
// @mode_option: optional per connector mode option
// @connector: connector to parse modeline for
// @mode: preallocated drm_cmdline_mode structure to fill out
//
// This parses @mode_option command line modeline for modes and options to
// configure the connector.
//
// This uses the same parameters as the fb modedb.c, except for an extra
// force-enable, force-enable-digital and force-disable bit at the end::
//
// 	<xres>x<yres>[M][R][-<bpp>][@<refresh>][i][m][eDd]
//
// Additionals options can be provided following the mode, using a comma to
// separate each option. Valid options can be found in
// Documentation/fb/modedb.rst.
//
// The intermediate drm_cmdline_mode structure is required to store additional
// options from the command line modline like the force-enable/disable flag.
//
// Returns:
// True if a valid modeline has been parsed, false otherwise.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_parse_command_line_for_connector_702: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8, *const drm_connector, *mut drm_cmdline_mode) -> bool>, *mut ()>(Some(drm_mode_parse_command_line_for_connector)) };

fn drm_named_mode(dev: *mut drm_device, cmd: *mut drm_cmdline_mode) -> *mut drm_display_mode {
let mut i: u32 = 0;
while (i as u64) < ARRAY_SIZE__51!() {
let named_mode: *const drm_named_mode = unsafe { core::ptr::addr_of!(drm_named_modes.0) as *const drm_named_mode }.wrapping_offset(i as isize);
if (unsafe { strcmp(((*cmd).name.as_mut_ptr() as *mut u8) as *const u8, (*named_mode).name) }) == 0 {
if unsafe { (*cmd).tv_mode_specified } {
return drm_analog_tv_mode(dev, (unsafe { (*named_mode).tv_mode }) as u32, unsafe { (*named_mode).pixel_clock_khz }.wrapping_mul(1000) as u64, unsafe { (*named_mode).xres }, unsafe { (*named_mode).yres }, (unsafe { (*named_mode).flags }) & 16u32 != 0);
}
}
i = i.wrapping_add(1);
}
core::ptr::null_mut::<drm_display_mode>()
}

/// drm_mode_create_from_cmdline_mode - convert a command line modeline into a DRM display mode
/// @dev: DRM device to create the new mode for
/// @cmd: input command line modeline
///
/// Returns:
/// Pointer to converted mode on success, NULL on error.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_create_from_cmdline_mode")]
extern "C" fn drm_mode_create_from_cmdline_mode(dev: *mut drm_device, cmd: *mut drm_cmdline_mode) -> *mut drm_display_mode {
let mut mode: *mut drm_display_mode = unsafe { core::mem::zeroed() };
if (unsafe { (*cmd).xres }) == 0 || (unsafe { (*cmd).yres }) == 0 {
return core::ptr::null_mut::<drm_display_mode>();
}
if (unsafe { strlen(((*cmd).name.as_mut_ptr() as *mut u8) as *const u8) }) != 0 {
mode = drm_named_mode(dev, cmd);
} else {
if unsafe { (*cmd).cvt } {
mode = drm_cvt_mode(dev, unsafe { (*cmd).xres }, unsafe { (*cmd).yres }, if unsafe { (*cmd).refresh_specified } { unsafe { (*cmd).refresh } } else { 60i32 }, unsafe { (*cmd).rb }, unsafe { (*cmd).interlace }, unsafe { (*cmd).margins });
} else {
mode = drm_gtf_mode(dev, unsafe { (*cmd).xres }, unsafe { (*cmd).yres }, if unsafe { (*cmd).refresh_specified } { unsafe { (*cmd).refresh } } else { 60i32 }, unsafe { (*cmd).interlace }, (unsafe { (*cmd).margins }) as i32);
}
}
if mode.is_null() {
return core::ptr::null_mut::<drm_display_mode>();
}
unsafe {
(*mode).r#type = (((*mode).r#type as i32) | 32i32) as u8;
}
// fix up 1368x768: GFT/CVT can't express 1366 width due to alignment
if (unsafe { (*cmd).xres }) == 1366 {
unsafe { drm_mode_fixup_1366x768(mode) };
}
drm_mode_set_crtcinfo(mode, 1i32);
mode
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_create_from_cmdline_mode_703: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_cmdline_mode) -> *mut drm_display_mode>, *mut ()>(Some(drm_mode_create_from_cmdline_mode)) };

/// drm_mode_convert_to_umode - convert a drm_display_mode into a modeinfo
/// @out: drm_mode_modeinfo struct to return to the user
/// @in: drm_display_mode to use
///
/// Convert a drm_display_mode into a drm_mode_modeinfo structure to return to
/// the user.
#[unsafe(no_mangle)]
extern "C" fn drm_mode_convert_to_umode(out: *mut drm_mode_modeinfo, r#in: *const drm_display_mode) {
unsafe {
(*out).clock = (*r#in).clock as u32;
}
unsafe {
(*out).hdisplay = (*r#in).hdisplay;
}
unsafe {
(*out).hsync_start = (*r#in).hsync_start;
}
unsafe {
(*out).hsync_end = (*r#in).hsync_end;
}
unsafe {
(*out).htotal = (*r#in).htotal;
}
unsafe {
(*out).hskew = (*r#in).hskew;
}
unsafe {
(*out).vdisplay = (*r#in).vdisplay;
}
unsafe {
(*out).vsync_start = (*r#in).vsync_start;
}
unsafe {
(*out).vsync_end = (*r#in).vsync_end;
}
unsafe {
(*out).vtotal = (*r#in).vtotal;
}
unsafe {
(*out).vscan = (*r#in).vscan;
}
unsafe {
(*out).vrefresh = drm_mode_vrefresh(r#in) as u32;
}
unsafe {
(*out).flags = (*r#in).flags;
}
unsafe {
(*out).r#type = (*r#in).r#type as u32;
}
match (unsafe { (*r#in).picture_aspect_ratio }) as u32 {
1 => {
unsafe {
(*out).flags = (*out).flags | 524288u32;
}
}
2 => {
unsafe {
(*out).flags = (*out).flags | 1048576u32;
}
}
3 => {
unsafe {
(*out).flags = (*out).flags | 1572864u32;
}
}
4 => {
unsafe {
(*out).flags = (*out).flags | 2097152u32;
}
}
0 => {
unsafe {
(*out).flags = (*out).flags | 0u32;
}
}
_ => {
let mut __v13: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Invalid aspect ratio (0%x) on mode\n".as_ptr() as *const u8, (*r#in).picture_aspect_ratio as u32) };
let mut __v14: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v14 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v15: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t1: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t1, sym __slate_asm_string_drm_mode_convert_to_umode_16, sym __slate_asm_string_drm_mode_convert_to_umode_17, const 2638 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t1 as u64) };
}
__v15 = bug;
unsafe { __v14.unwrap()(__v15, (*r#in).picture_aspect_ratio as u32) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v13 = (__ret_warn_on != 0) as i64;
let _v23: u32 = unsafe { (*out).flags };
let _v24: u32 = _v23 | 0u32;
unsafe {
(*out).flags = _v24;
}
}
}
sized_strscpy_pad__2!(unsafe { (*out).name.as_mut_ptr() as *mut u8 }, unsafe { (*r#in).name.as_ptr() as *const u8 }, (core::mem::size_of::<[u8; 32]>() as u64).wrapping_add(__must_be_cstr__2!() as u64).wrapping_add(__must_be_cstr!() as u64));
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_704: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

/// drm_mode_convert_umode - convert a modeinfo into a drm_display_mode
/// @dev: drm device
/// @out: drm_display_mode to return to the user
/// @in: drm_mode_modeinfo to use
///
/// Convert a drm_mode_modeinfo into a drm_display_mode structure to return to
/// the caller.
///
/// Returns:
/// Zero on success, negative errno on failure.
#[unsafe(no_mangle)]
extern "C" fn drm_mode_convert_umode(dev: *mut drm_device, out: *mut drm_display_mode, r#in: *const drm_mode_modeinfo) -> i32 {
if (unsafe { (*r#in).clock }) > 2147483647u32 || (unsafe { (*r#in).vrefresh }) > 2147483647u32 {
return 34i32.wrapping_neg();
}
unsafe {
(*out).clock = (*r#in).clock as i32;
}
unsafe {
(*out).hdisplay = (*r#in).hdisplay;
}
unsafe {
(*out).hsync_start = (*r#in).hsync_start;
}
unsafe {
(*out).hsync_end = (*r#in).hsync_end;
}
unsafe {
(*out).htotal = (*r#in).htotal;
}
unsafe {
(*out).hskew = (*r#in).hskew;
}
unsafe {
(*out).vdisplay = (*r#in).vdisplay;
}
unsafe {
(*out).vsync_start = (*r#in).vsync_start;
}
unsafe {
(*out).vsync_end = (*r#in).vsync_end;
}
unsafe {
(*out).vtotal = (*r#in).vtotal;
}
unsafe {
(*out).vscan = (*r#in).vscan;
}
unsafe {
(*out).flags = (*r#in).flags;
}
// Old xf86-video-vmware (possibly others too) used to
// leave 'type' uninitialized. Just ignore any bits we
// don't like. It's a just hint after all, and more
// useful for the kernel->userspace direction anyway.
unsafe {
(*out).r#type = ((*r#in).r#type & 104u32) as u8;
}
sized_strscpy_pad__2!(unsafe { (*out).name.as_mut_ptr() as *mut u8 }, unsafe { (*r#in).name.as_ptr() as *const u8 }, (core::mem::size_of::<[u8; 32]>() as u64).wrapping_add(__must_be_cstr__5!() as u64).wrapping_add(__must_be_cstr__18!() as u64));
// Clearing picture aspect ratio bits from out flags,
// as the aspect-ratio information is not stored in
// flags for kernel-mode, but in picture_aspect_ratio.
let __v3: u32 = (unsafe { (*out).flags }) & 4287102975u32;
unsafe {
(*out).flags = __v3;
}
match (unsafe { (*r#in).flags }) & 7864320u32 {
524288 => {
unsafe {
(*out).picture_aspect_ratio = 1;
}
}
1048576 => {
unsafe {
(*out).picture_aspect_ratio = 2;
}
}
1572864 => {
unsafe {
(*out).picture_aspect_ratio = 3;
}
}
2097152 => {
unsafe {
(*out).picture_aspect_ratio = 4;
}
}
0 => {
unsafe {
(*out).picture_aspect_ratio = 0;
}
}
_ => {
return 22i32.wrapping_neg();
}
}
unsafe {
(*out).status = drm_mode_validate_driver(dev, out as *const drm_display_mode);
}
if ((unsafe { (*out).status }) as i32) != 0 {
return 22i32.wrapping_neg();
}
drm_mode_set_crtcinfo(out, 1i32);
0
}

/// drm_mode_is_420_only - if a given videomode can be only supported in YCBCR420
/// output format
///
/// @display: display under action
/// @mode: video mode to be tested.
///
/// Returns:
/// true if the mode can be supported in YCBCR420 format
/// false if not.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_is_420_only")]
extern "C" fn drm_mode_is_420_only(display: *const drm_display_info, mode: *const drm_display_mode) -> bool {
let vic: u8 = unsafe { drm_match_cea_mode(mode) };
test_bit__2!(vic, unsafe { (*display).hdmi.y420_vdb_modes.as_ptr() as *const u64 }) != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_is_420_only_705: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_display_info, *const drm_display_mode) -> bool>, *mut ()>(Some(drm_mode_is_420_only)) };

/// drm_mode_is_420_also - if a given videomode can be supported in YCBCR420
/// output format also (along with RGB/YCBCR444/422)
///
/// @display: display under action.
/// @mode: video mode to be tested.
///
/// Returns:
/// true if the mode can be support YCBCR420 format
/// false if not.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_is_420_also")]
extern "C" fn drm_mode_is_420_also(display: *const drm_display_info, mode: *const drm_display_mode) -> bool {
let vic: u8 = unsafe { drm_match_cea_mode(mode) };
test_bit__2!(vic, unsafe { (*display).hdmi.y420_cmdb_modes.as_ptr() as *const u64 }) != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_is_420_also_706: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_display_info, *const drm_display_mode) -> bool>, *mut ()>(Some(drm_mode_is_420_also)) };

/// drm_mode_is_420 - if a given videomode can be supported in YCBCR420
/// output format
///
/// @display: display under action.
/// @mode: video mode to be tested.
///
/// Returns:
/// true if the mode can be supported in YCBCR420 format
/// false if not.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_mode_is_420")]
extern "C" fn drm_mode_is_420(display: *const drm_display_info, mode: *const drm_display_mode) -> bool {
drm_mode_is_420_only(display, mode) || drm_mode_is_420_also(display, mode)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_is_420_707: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_display_info, *const drm_display_mode) -> bool>, *mut ()>(Some(drm_mode_is_420)) };

/// drm_set_preferred_mode - Sets the preferred mode of a connector
/// @connector: connector whose mode list should be processed
/// @hpref: horizontal resolution of preferred mode
/// @vpref: vertical resolution of preferred mode
///
/// Marks a mode as preferred if it matches the resolution specified by @hpref
/// and @vpref.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_modes.drm_set_preferred_mode")]
extern "C" fn drm_set_preferred_mode(connector: *mut drm_connector, hpref: i32, vpref: i32) {
let mut mode: *mut drm_display_mode = unsafe { core::mem::zeroed() };
mode = list_first_entry__4!(unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) });
while list_entry_is_head!(mode, unsafe { core::ptr::addr_of_mut!((*connector).probed_modes) }, head) == 0 {
if ((unsafe { (*mode).hdisplay }) as i32) == hpref && ((unsafe { (*mode).vdisplay }) as i32) == vpref {
unsafe {
(*mode).r#type = (((*mode).r#type as i32) | 8i32) as u8;
}
}
mode = list_next_entry!(mode, head);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_set_preferred_mode_708: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, i32, i32)>, *mut ()>(Some(drm_set_preferred_mode)) };

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_drm_connector_list_update_1: [u8; 1] = *b"\0";

static __slate_asm_string_drm_connector_list_update_2: [u8; 57] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_modes.c\0";

static __slate_asm_string_drm_get_mode_status_name_1: [u8; 1] = *b"\0";

static __slate_asm_string_drm_get_mode_status_name_2: [u8; 57] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_modes.c\0";

static __slate_asm_string_drm_mode_convert_to_umode_16: [u8; 36] = *b"Invalid aspect ratio (0%x) on mode\n\0";

static __slate_asm_string_drm_mode_convert_to_umode_17: [u8; 57] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_modes.c\0";

static __slate_asm_string_drm_mode_probed_add_1: [u8; 1] = *b"\0";

static __slate_asm_string_drm_mode_probed_add_2: [u8; 57] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_modes.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";
