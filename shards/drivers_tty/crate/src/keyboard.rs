
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
//! Written for linux by Johan Myreen as a translation from
//! the assembly version by Linus (with diacriticals added)
//!
//! Some additional features added by Christoph Niemann (ChN), March 1993
//!
//! Loadable keymaps by Risto Kankkunen, May 1993
//!
//! Diacriticals redone & other small changes, aeb@cwi.nl, June 1993
//! Added decr/incr_console, dynamic keymaps, Unicode support,
//! dynamic function/string keys, led setting,  Sept 1994
//! `Sticky' modifier keys, 951006.
//!
//! 11-11-96: SAK should now work in the raw mode (Martin Mares)
//!
//! Modified to provide 'generic' keyboard support by Hamish Macdonald
//! Merge with the m68k keyboard driver and split-off of the PC low-level
//! parts by Geert Uytterhoeven, May 1997
//!
//! 27-05-97: Added support for the Magic SysRq Key (Martin Mares)
//! 30-07-98: Dead keys redone, aeb@cwi.nl.
//! 21-08-02: Converted to input API, major cleanup. (Vojtech Pavlik)
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
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
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
//! Internal kernel timer frequency
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
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
//! array_index_mask_nospec() - generate a mask that is ~0UL when the
//! 	bounds check succeeds and 0 otherwise
//! @idx: array element index
//! @sz: number of elements in array
//!
//! Returns:
//!     0 - (@idx < @sz)
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
//! Value too large for defined data type
//! No ioctl command
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
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
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
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
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! Generate SD flag indexes
//! Resource type
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
//! Input
//! return current led state
//! This means 128Kb if all keymaps are allocated. Only the superuser
//! may increase the number of keymaps beyond MAX_NR_OF_USER_KEYMAPS.
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! serial lines start at 64
//! number of edd_info structs starting at EDDBUF
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
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
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
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
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
// Operation not supported on transport endpoint
// Memory page has hardware error
// restart by calling sys_restart_syscall
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
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
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
// Set the IDR flag and the IDR_FREE tag
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
// ptr must be evaluated and assigned to the temporary __ptr_pu before
// the assignment of x to __val_pu, to avoid any function calls
// involved in the ptr expression (possibly implicitly generated due
// to KASAN) from clobbering %ax.
// eval x once
// eval ptr once
// put_user - Write a simple value into user space.
// @x:   Value to copy to user space.
// @ptr: Destination address, in user space.
//
// Context: User context only. This function may sleep if pagefaults are
//          enabled.
//
// This macro copies a single simple value from kernel space to user
// space.  It supports simple types like char and int, but not larger
// data types like structures or arrays.
//
// @ptr must have pointer-to-simple-variable type, and @x must be assignable
// to the result of dereferencing @ptr.
//
// Return: zero on success, or -EFAULT on error.
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
// should be at least 7
// max nr of strings assigned to keys
// we depend on this being zero
// symbol that can be acted upon by CapsLock
// CSI sequences with modifier support
// PGUP is a synonym for PRIOR
// PGDN is a synonym for NEXT
// dynamically allocated keymap
// returned by KDGKBENT
// key-pad enter
// key-pad comma: kludge...
// key-pad dot (period): kludge...
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
macro_rules! ARRAY_SIZE__14 {
() => {
((core::mem::size_of::<[Option<unsafe extern "C" fn(*mut vc_data)>; 20]>() as u64) / (core::mem::size_of::<Option<unsafe extern "C" fn(*mut vc_data)>>() as u64)).wrapping_add(__must_be_array__23!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__15 {
() => {
((core::mem::size_of::<[kbd_led_trigger; 12]>() as u64) / (core::mem::size_of::<kbd_led_trigger>() as u64)).wrapping_add(__must_be_array__38!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__19 {
() => {
((core::mem::size_of::<[kbd_led_trigger; 12]>() as u64) / (core::mem::size_of::<kbd_led_trigger>() as u64)).wrapping_add(__must_be_array__11!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__32 {
() => {
((core::mem::size_of::<[*mut u8; 256]>() as u64) / (core::mem::size_of::<*mut u8>() as u64)).wrapping_add(__must_be_array__9!() as u64)
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

macro_rules! BIT_WORD__2 {
($nr:expr) => {
($nr) / 64
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

macro_rules! KTYP {
($x:expr) => {
(($x) as i32) >> 8i32
};
}

macro_rules! KTYP__2 {
($x:expr) => {
($x) >> 8i32
};
}

macro_rules! KVAL {
($x:expr) => {
($x) & 255
};
}

macro_rules! KVAL__3 {
($x:expr) => {
($x) & 255
};
}

macro_rules! KVAL__2 {
($x:expr) => {
(($x) as i32) & 255
};
}

macro_rules! K {
($t:expr, $v:expr) => {
($t) << 8i32 | $v
};
}

macro_rules! K__2 {
($t:expr, $v:expr) => {
((($t) << 8i32) as u32) | $v
};
}

macro_rules! L_ECHO {
($tty:expr) => {
_L_FLAG!($tty, 8i32)
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

macro_rules! U__3 {
($x:expr) => {
($x) ^ 61440
};
}

macro_rules! U__2 {
($x:expr) => {
($x) ^ 61440
};
}

macro_rules! U {
($x:expr) => {
(($x) as i32) ^ 61440
};
}

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! _L_FLAG {
($tty:expr, $f:expr) => {
(unsafe { (*$tty).termios.c_lflag }) & (($f) as u32)
};
}

macro_rules! _UL {
() => {
_AC!()
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

macro_rules! __chk_user_ptr {
() => {
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

macro_rules! __get_and_null {
($p:expr) => {
{
let mut __v: *mut input_handle;
let __v2: *mut *mut input_handle = core::ptr::addr_of_mut!($p);
let __v3: *mut input_handle = unsafe { *__v2 };
unsafe {
*__v2 = core::ptr::null_mut::<input_handle>();
}
__v = __v3;
__v
}
};
}

macro_rules! __get_and_null__3 {
($p:expr) => {
{
let mut __v: *mut u16;
let __v2: *mut *mut u16 = core::ptr::addr_of_mut!($p);
let __v3: *mut u16 = unsafe { *__v2 };
unsafe {
*__v2 = core::ptr::null_mut::<u16>();
}
__v = __v3;
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

macro_rules! __must_be_array__9 {
() => {
core::mem::size_of::<__SlateRecord_keyboard_c_757>() as i32
};
}

macro_rules! __must_be_array__11 {
() => {
core::mem::size_of::<__SlateRecord_keyboard_c_1145>() as i32
};
}

macro_rules! __must_be_array__23 {
() => {
core::mem::size_of::<__SlateRecord_keyboard_c_652>() as i32
};
}

macro_rules! __must_be_array__38 {
() => {
core::mem::size_of::<__SlateRecord_keyboard_c_1120>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__2 {
() => {
core::mem::size_of::<__SlateRecord_keyboard_c_2096_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__5 {
() => {
core::mem::size_of::<__SlateRecord_keyboard_c_2096>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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

macro_rules! __smp_mb__after_atomic {
() => {
while false {
}
};
}

/// Atomic operations are already serializing on x86
macro_rules! __smp_mb__before_atomic {
() => {
while false {
}
};
}

macro_rules! __test_and_set_bit {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_70b71f185ebb8c4a($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___test_and_set_bit(($nr) as u64, ($addr) as *mut u64) as i32
} else {
___test_and_set_bit(($nr) as u64, ($addr) as *mut u64) as i32
};
__v
}
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
let __vpp_verify: *const () = core::ptr::null_mut::<*mut pt_regs>() as *const ();
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

/// assign_bit - Assign value to a bit in memory
/// @nr: the bit to set
/// @addr: the address to start counting from
/// @value: the value to assign
macro_rules! assign_bit {
($nr:expr, $addr:expr, $value:expr) => {
if $value {
set_bit(($nr) as i64, ($addr) as *mut u64);
} else {
clear_bit(($nr) as i64, ($addr) as *mut u64);
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
macro_rules! container_of__11 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__23!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// instrument_get_user() - add instrumentation to get_user()-like macros
/// @to: destination variable, may not be address-taken
///
/// get_user() and friends are fragile, so it may depend on the implementation
/// whether the instrumentation happens before or after the data is copied from
/// the userspace.
macro_rules! instrument_get_user__2 {
($to:expr) => {
let mut __v: u8;
let mut __tmp: u64 = ($to) as u64;
kmsan_unpoison_memory(core::ptr::addr_of_mut!(__tmp) as *const (), core::mem::size_of::<u64>() as u64);
let __v2: u8 = __tmp as u8;
$to = __v2;
__v = __v2;
};
}

/// instrument_get_user() - add instrumentation to get_user()-like macros
/// @to: destination variable, may not be address-taken
///
/// get_user() and friends are fragile, so it may depend on the implementation
/// whether the instrumentation happens before or after the data is copied from
/// the userspace.
macro_rules! instrument_get_user {
($to:expr) => {
let mut __v: u32;
let mut __tmp: u64 = ($to) as u64;
kmsan_unpoison_memory(core::ptr::addr_of_mut!(__tmp) as *const (), core::mem::size_of::<u64>() as u64);
let __v2: u32 = __tmp as u32;
$to = __v2;
__v = __v2;
};
}

/// instrument_put_user() - add instrumentation to put_user()-like macros
/// @from: source address
/// @ptr: userspace pointer to copy to
/// @size: number of bytes to copy
///
/// put_user() and friends are fragile, so it may depend on the implementation
/// whether the instrumentation happens before or after the data is copied from
/// the userspace.
macro_rules! instrument_put_user__2 {
($from:expr, $ptr:expr) => {
kmsan_copy_to_user(($ptr) as *mut (), core::ptr::addr_of_mut!($from) as *const (), core::mem::size_of::<u32>() as u64, 0u64);
};
}

/// instrument_put_user() - add instrumentation to put_user()-like macros
/// @from: source address
/// @ptr: userspace pointer to copy to
/// @size: number of bytes to copy
///
/// put_user() and friends are fragile, so it may depend on the implementation
/// whether the instrumentation happens before or after the data is copied from
/// the userspace.
macro_rules! instrument_put_user__3 {
($from:expr, $ptr:expr) => {
kmsan_copy_to_user(($ptr) as *mut (), core::ptr::addr_of_mut!($from) as *const (), core::mem::size_of::<u8>() as u64, 0u64);
};
}

/// instrument_put_user() - add instrumentation to put_user()-like macros
/// @from: source address
/// @ptr: userspace pointer to copy to
/// @size: number of bytes to copy
///
/// put_user() and friends are fragile, so it may depend on the implementation
/// whether the instrumentation happens before or after the data is copied from
/// the userspace.
macro_rules! instrument_put_user__4 {
($from:expr, $ptr:expr) => {
kmsan_copy_to_user(($ptr) as *mut (), core::ptr::addr_of_mut!($from) as *const (), core::mem::size_of::<u16>() as u64, 0u64);
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

/// kcsan_check_read - check regular read access for races
///
/// @ptr: address of access
/// @size: size of access
macro_rules! kcsan_check_read {
($ptr:expr, $size:expr) => {
kcsan_check_access(($ptr) as *const (), $size, 0i32);
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

macro_rules! no_free_ptr {
($p:expr) => {
__must_check_fn(__get_and_null__3!($p) as *const ()) as *mut u16
};
}

macro_rules! offsetof__23 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, trigger) as u64
};
}

macro_rules! printk_ratelimit {
() => {
unsafe { __printk_ratelimit(c"kbd_keycode".as_ptr() as *const u8) }
};
}

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
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

/// Only for situations where an allocation is handed in to another function
/// and consumed by that function on success.
///
/// 	struct foo *f __free(kfree) = kzalloc(sizeof(*f), GFP_KERNEL);
///
/// 	setup(f);
/// 	if (some_condition)
/// 		return -EINVAL;
/// 	....
/// 	ret = bar(f);
/// 	if (!ret)
/// 		retain_and_null_ptr(f);
/// 	return ret;
///
/// After retain_and_null_ptr(f) the variable f is NULL and cannot be
/// dereferenced anymore.
macro_rules! retain_and_null_ptr__2 {
($p:expr) => {
__get_and_null!($p);
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

macro_rules! smp_mb__after_atomic {
() => {
loop {
kcsan_mb!();
__smp_mb__after_atomic!();
if true {
break;
}
}
};
}

macro_rules! smp_mb__before_atomic {
() => {
loop {
kcsan_mb!();
__smp_mb__before_atomic!();
if true {
break;
}
}
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
macro_rules! time_after__3 {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck__3!(u64) != 0 && typecheck__2!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
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

unsafe extern "C" {
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut jiffies: u64;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut param_ops_uint: kernel_param_ops;
    static mut accent_table: [kbdiacruc; 0];
    static mut accent_table_size: u32;
    static mut key_maps: [*mut u16; 256];
    static mut plain_map: [u16; 256];
    static mut func_table: [*mut u8; 256];
    static mut do_poke_blanked_console: i32;
    static mut keymap_count: u32;
    static mut vc_cons: [vc; 63];
    static mut fg_console: i32;
    static mut last_console: i32;
    static mut want_console: i32;
    static mut default_utf8: i32;
    static mut __irq_regs: *mut pt_regs;
    static mut ist_info: ist_info;
    static mut boot_params: boot_params;
    fn conv_8bit_to_uni(arg0: u8) -> u32;
    fn conv_uni_to_8bit(arg0: u32) -> i32;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __printk_ratelimit(arg0: *const u8) -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn _find_next_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn strndup_user(arg0: *const u8, arg1: i64) -> *mut u8;
    fn memdup_user(arg0: *const (), arg1: u64) -> *mut ();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn strlen(arg0: *const u8) -> u64;
    fn strcpy(arg0: *mut u8, arg1: *const u8) -> *mut u8;
    fn sized_strscpy(arg0: *mut u8, arg1: *const u8, arg2: u64) -> i64;
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn mod_timer(arg0: *mut timer_list, arg1: u64) -> i32;
    fn timer_delete_sync(arg0: *mut timer_list) -> i32;
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn atomic_notifier_chain_register(arg0: *mut atomic_notifier_head, arg1: *mut notifier_block) -> i32;
    fn atomic_notifier_chain_unregister(arg0: *mut atomic_notifier_head, arg1: *mut notifier_block) -> i32;
    fn atomic_notifier_call_chain(arg0: *mut atomic_notifier_head, arg1: u64, arg2: *mut ()) -> i32;
    fn __bad_copy_from();
    fn __bad_copy_to();
    fn __copy_overflow(arg0: i32, arg1: u64);
    fn _copy_from_user(arg0: *mut (), arg1: *const (), arg2: u64) -> u64;
    fn _copy_to_user(arg0: *mut (), arg1: *const (), arg2: u64) -> u64;
    fn capable(arg0: i32) -> bool;
    fn put_pid(arg0: *mut pid);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn kill_pid(arg0: *mut pid, arg1: i32, arg2: i32) -> i32;
    fn __compiletime_assert_388();
    fn down(arg0: *mut semaphore);
    fn __compiletime_assert_409();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_415();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn input_register_handler(arg0: *mut input_handler) -> i32;
    fn input_handler_for_each_handle(arg0: *mut input_handler, arg1: *mut (), arg2: Option<unsafe extern "C" fn(*mut input_handle, *mut ()) -> i32>) -> i32;
    fn input_register_handle(arg0: *mut input_handle) -> i32;
    fn input_unregister_handle(arg0: *mut input_handle);
    fn input_open_device(arg0: *mut input_handle) -> i32;
    fn input_close_device(arg0: *mut input_handle);
    fn input_inject_event(arg0: *mut input_handle, arg1: u32, arg2: u32, arg3: i32);
    fn input_get_keycode(arg0: *mut input_dev, arg1: *mut input_keymap_entry) -> i32;
    fn input_set_keycode(arg0: *mut input_dev, arg1: *const input_keymap_entry) -> i32;
    fn tty_port_tty_get(arg0: *mut tty_port) -> *mut tty_struct;
    fn tty_kref_put(arg0: *mut tty_struct);
    fn stop_tty(arg0: *mut tty_struct);
    fn start_tty(arg0: *mut tty_struct);
    fn tty_chars_in_buffer(arg0: *mut tty_struct) -> u32;
    fn tasklet_unlock_wait(arg0: *mut tasklet_struct);
    fn __tasklet_schedule(arg0: *mut tasklet_struct);
    fn set_console(arg0: i32) -> i32;
    fn schedule_console_callback();
    fn led_set_brightness(arg0: *mut led_classdev, arg1: u32);
    fn led_trigger_register(arg0: *mut led_trigger) -> i32;
    fn led_trigger_event(arg0: *mut led_trigger, arg1: u32);
    fn __show_mem(arg0: u32, arg1: *const __SlateRecord_nodemask_types_h_17, arg2: i32);
    fn ctrl_alt_del();
    fn show_state_filter(arg0: u32);
    fn show_regs(arg0: *mut pt_regs);
    fn __tty_insert_flip_string_flags(arg0: *mut tty_port, arg1: *const u8, arg2: *const u8, arg3: bool, arg4: u64) -> u64;
    fn tty_flip_buffer_push(arg0: *mut tty_port);
    fn vc_cons_allocated(arg0: u32) -> i32;
    fn scrollback(arg0: *mut vc_data);
    fn scrollfront(arg0: *mut vc_data, arg1: i32);
    fn __compiletime_assert_506();
    fn __compiletime_assert_548();
    fn __compiletime_assert_549();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.objectsize.i64.p0"]
    fn __slate_intrinsic_2913c0303de876b3(arg0: *const (), arg1: bool, arg2: bool, arg3: bool) -> u64;
    #[link_name = "llvm.is.constant.i8"]
    fn __slate_intrinsic_70b71f185ebb8c4a(arg0: u8) -> bool;
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
struct static_call_site {
    addr: i32,
    key: i32,
}


// __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
// the symbol table so that objtool can reference it when it generates the
// .static_call_sites section.
#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
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
struct lockdep_map {
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
struct fred_cs {
    __slate_bits_0: __slate_bits::__SlateBits95U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fred_ss {
    __slate_bits_0: __slate_bits::__SlateBits96U0,
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
#[inline(always)]
fn __this_cpu_preempt_check(op: *const u8) {
}

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


fn kmsan_unpoison_memory(address: *const (), size: u64) {
}

fn kmsan_copy_to_user(to: *mut (), from: *const (), to_copy: u64, left: u64) {
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
fn generic___test_and_set_bit(nr: u64, addr: *mut u64) -> bool {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD__2!(nr) as isize);
let old: u64 = unsafe { *p };
unsafe {
*p = old | mask;
}
old & mask != 0
}

// const_*() definitions provide good compile-time optimizations when
// the passed arguments can be resolved at compile time.
#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD__2!(nr) as isize);
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

#[inline(always)]
fn arch___test_and_set_bit(mut nr: u64, mut addr: *mut u64) -> bool {
let mut oldbit: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!(" btsq  {2:r},({1:r})\nsetc {0}", lateout(reg_byte) __t0, in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
oldbit = (__t0 as u8) != 0;
}
oldbit
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
fn instrument_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_write!(v, size);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_67 {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn set_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_set_bit(nr, addr);
}

#[inline(always)]
fn clear_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_clear_bit(nr, addr);
}

#[inline(always)]
fn test_and_set_bit(nr: i64, addr: *mut u64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_set_bit(nr, addr)
}

#[inline(always)]
fn __instrument_read_write_bitop(nr: i64, addr: *mut u64) {
if IS_ENABLED!() != 0 {
// We treat non-atomic read-write bitops a little more special.
// Given the operations here only modify a single bit, assuming
// non-atomicity of the writer is sufficient may be reasonable
// for certain usage (and follows the permissible nature of the
// assume-plain-writes-atomic rule):
// 1. report read-modify-write races -> check read;
// 2. do not report races with marked readers, but do report
//    races with unmarked readers -> check "atomic" write.
kcsan_check_read!(addr.wrapping_offset(BIT_WORD!(nr) as isize), core::mem::size_of::<i64>() as u64);
// Use generic write instrumentation, in case other sanitizers
// or tools are enabled alongside KCSAN.
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
} else {
instrument_read_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[inline(always)]
fn ___test_and_set_bit(nr: u64, addr: *mut u64) -> bool {
__instrument_read_write_bitop(nr as i64, addr);
arch___test_and_set_bit(nr, addr)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits120U0,
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
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
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[inline(always)]
fn IS_ERR_OR_NULL(ptr: *const ()) -> bool {
unlikely!(ptr.is_null()) != 0 || IS_ERR_VALUE!(ptr as u64) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[inline(always)]
fn __must_check_fn(val: *const ()) -> *const () {
val
}

// Helper macro for scoped_guard().
//
// Note that the "!__is_cond_ptr(_name)" part of the condition ensures that
// compiler would be sure that for the unconditional locks the body of the
// loop (caller-provided code glued to the else clause) could not be skipped.
// It is needed because the other part - "__guard_ptr(_name)(&scope)" - is too
// hard to deduce (even if could be proven true for unconditional locks).
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
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
}

#[inline(always)]
fn raw_atomic_dec(v: *mut __SlateRecord_types_h_190) {
arch_atomic_dec(v);
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

fn memdup_array_user(src: *const (), n: u64, size: u64) -> *mut () {
let mut nbytes: u64 = 0;
if check_mul_overflow!(n, size, core::ptr::addr_of_mut!(nbytes)) {
return ERR_PTR(75i32.wrapping_neg() as i64);
}
unsafe { memdup_user(src, nbytes) }
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
struct hrtimer_clock_base {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


fn might_fault() {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
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
struct restart_block {
    arch_data: u64,
    r#fn: Option<unsafe extern "C" fn(*mut restart_block) -> i64>,
    __slate_anon_2: __SlateRecord_restart_block_h_29,
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
struct thread_info {
    flags: u64,
    syscall_work: u64,
    status: u32,
    cpu: u32,
}


// 32-bit address space on 64 bits
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
struct workqueue_struct {
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
struct rseq {
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
struct rb_node_linked {
    node: rb_node,
    prev: *mut rb_node_linked,
    next: *mut rb_node_linked,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
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
fn spin_lock(lock: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
}

static class_spinlock_is_conditional: bool = 0i32 != 0i32;

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
fn class_spinlock_lock_ptr(_T: &mut __SlateRecord_spinlock_h_610) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_spinlock_h_610)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
}


#[inline(always)]
fn class_spinlock_constructor(l: *mut spinlock) -> __SlateRecord_spinlock_h_610 {
let mut _t: __SlateRecord_spinlock_h_610 = __SlateRecord_spinlock_h_610 { lock: l };
let _T: *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(_t);
spin_lock(unsafe { (*_T).lock });
return _t;
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
fn __class_spinlock_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_610) {
}

static class_spinlock_irqsave_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_spinlock_h_642 {
    lock: *mut spinlock,
    flags: u64,
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
fn class_spinlock_irqsave_destructor(_T: *mut __SlateRecord_spinlock_h_642) {
spin_unlock_irqrestore(unsafe { (*_T).lock }, unsafe { (*_T).flags });
}

#[inline(always)]
fn class_spinlock_irqsave_lock_ptr(_T: &mut __SlateRecord_spinlock_h_642) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_spinlock_h_642)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[inline(always)]
fn class_spinlock_irqsave_constructor(l: *mut spinlock) -> __SlateRecord_spinlock_h_642 {
let mut _t: __SlateRecord_spinlock_h_642 = {
let mut __t0: __SlateRecord_spinlock_h_642 = unsafe { core::mem::zeroed() };
__t0.lock = l;
__t0
};
let _T: *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(_t);
spin_lock_irqsave__2!(unsafe { (*_T).lock }, (*_T).flags);
return _t;
}

#[inline(always)]
fn __class_spinlock_irqsave_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_642) {
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
struct syscall_user_dispatch {
    selector: *mut u8,
    offset: u64,
    len: u64,
    on_dispatch: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_tlbflush_unmap_batch {
    cpumask: cpumask,
    unmapped_pages: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
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
union __SlateRecord_input_h_480 {
    constant: ff_constant_effect,
    ramp: ff_ramp_effect,
    periodic: ff_periodic_effect,
    condition: [ff_condition_effect; 2],
    rumble: ff_rumble_effect,
    haptic: ff_haptic_effect,
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_13 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe_xol_ops {
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
struct kmap_ctrl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
    __slate_bits_0: __slate_bits::__SlateBits66U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits66U1,
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
    __slate_bits_2: __slate_bits::__SlateBits66U2,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_id {
    bustype: u16,
    vendor: u16,
    product: u16,
    version: u16,
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
struct input_absinfo {
    value: i32,
    minimum: i32,
    maximum: i32,
    fuzz: i32,
    flat: i32,
    resolution: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_keymap_entry {
    flags: u8,
    len: u8,
    index: u16,
    keycode: u32,
    scancode: [u8; 32],
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
struct ff_replay {
    length: u16,
    delay: u16,
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
struct ff_trigger {
    button: u16,
    interval: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ff_envelope {
    attack_length: u16,
    attack_level: u16,
    fade_length: u16,
    fade_level: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ff_constant_effect {
    level: i16,
    envelope: ff_envelope,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ff_ramp_effect {
    start_level: i16,
    end_level: i16,
    envelope: ff_envelope,
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
struct ff_condition_effect {
    right_saturation: u16,
    left_saturation: u16,
    right_coeff: i16,
    left_coeff: i16,
    deadband: u16,
    center: i16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ff_periodic_effect {
    waveform: u16,
    period: u16,
    magnitude: i16,
    offset: i16,
    phase: u16,
    envelope: ff_envelope,
    custom_len: u32,
    custom_data: *mut i16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ff_rumble_effect {
    strong_magnitude: u16,
    weak_magnitude: u16,
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
struct ff_haptic_effect {
    hid_usage: u16,
    vendor_id: u16,
    vendor_waveform_page: u8,
    intensity: u16,
    repeat_count: u16,
    retrigger_period: u16,
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
struct ff_effect {
    r#type: u16,
    id: i16,
    direction: u16,
    trigger: ff_trigger,
    replay: ff_replay,
    u: __SlateRecord_input_h_480,
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
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
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
struct ns_common {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
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
struct completion {
    done: u32,
    wait: swait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


fn queue_work(wq: *mut workqueue_struct, work: *mut work_struct) -> bool {
unsafe { queue_work_on(64i32, wq, work) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


fn schedule_work(work: *mut work_struct) -> bool {
queue_work(unsafe { system_percpu_wq }, work)
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
struct atomic_notifier_head {
    lock: spinlock,
    head: *mut notifier_block,
}


// Suits me
// Don't call further
// Bad/Veto action
//
// Clean way to return from the notifier and stop further calls.
// Console keyboard events.
// Note: KBD_KEYCODE is always sent before KBD_UNBOUND_KEYCODE, KBD_UNICODE and
// KBD_KEYSYM.
// Keyboard keycode, called before any other
// Keyboard keycode which is not bound to any other
// Keyboard unicode
// Keyboard keysym
// Called after keyboard keysym interpretation
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
struct return_consumer {
    cookie: u64,
    id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
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
struct dev_pm_qos {
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
struct fwnode_handle {
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
struct folio_queue {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits424U0,
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
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
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
struct mnt_idmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_symlink {
    target_kn: *mut kernfs_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
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
struct key_type {
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
    __slate_bits_0: __slate_bits::__SlateBits471U0,
    vm_ops: *const vm_operations_struct,
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
struct attribute {
    name: *const u8,
    mode: u16,
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
struct taskstats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
core::arch::asm!("242: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 242b - ., 3; .popsection\n# {0}", const 242 as i32, options(nostack, att_syntax));
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
core::arch::asm!("243: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 243b - ., 4; .popsection\n# {0}", const 243 as i32, options(nostack, att_syntax));
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
fn copy_from_user(to: *mut (), from: *const (), n: u64) -> u64 {
if !check_copy_size(to as *const (), n, false) {
return n;
}
unsafe { _copy_from_user(to, from, n) }
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


#[inline(always)]
fn copy_to_user(to: *mut (), from: *const (), n: u64) -> u64 {
if !check_copy_size(from, n, true) {
return n;
}
unsafe { _copy_to_user(to, from, n) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct semaphore_waiter {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
struct input_dev_poller {
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
struct input_mt {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct serial_icounter_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serial_struct {
}


fn mnt_idmap(mnt: &vfsmount) -> *mut mnt_idmap {
// Pairs with smp_store_release() in do_idmap_mount().
let mut __v0: *mut mnt_idmap;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
struct proc_dir_entry {
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
    __slate_bits_0: __slate_bits::__SlateBits627U0,
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
#[inline(always)]
fn __free_kfree(p: *mut ()) {
let _T: *mut () = unsafe { *(p as *mut *mut ()) };
if !IS_ERR_OR_NULL(_T as *const ()) {
unsafe { kfree(_T as *const ()) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_tty_buffer_h_11 {
    next: *mut tty_buffer,
    free: llist_node,
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
unsafe { __compiletime_assert_415() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("416: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 416b - ., 3; .popsection\n# {0}", const 416 as i32, options(nostack, att_syntax));
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


// __alloc_objs - Allocate objects of a given type using
// @KMALLOC: which size-based kmalloc wrapper to allocate with.
// @GFP: GFP flags for the allocation.
// @TYPE: type to allocate space for.
// @COUNT: how many @TYPE objects to allocate.
//
// Returns: Newly allocated pointer to (first) @TYPE of @COUNT-many
// allocated @TYPE objects, or NULL on failure.
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
union __SlateRecord_interrupt_h_697 {
    func: Option<unsafe extern "C" fn(u64)>,
    callback: Option<unsafe extern "C" fn(*mut tasklet_struct)>,
}


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


#[repr(C)]
#[derive(Clone, Copy)]
struct inode_state_flags {
    __state: u32,
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
struct uni_pagedict {
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
struct consw {
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
union __SlateRecord_edd_h_87 {
    isa: __SlateRecord_edd_h_88,
    pci: __SlateRecord_edd_h_93,
    ibnd: __SlateRecord_edd_h_101,
    xprs: __SlateRecord_edd_h_104,
    htpt: __SlateRecord_edd_h_107,
    unknown: __SlateRecord_edd_h_110,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_88 {
    base_address: u16,
    reserved1: u16,
    reserved2: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_93 {
    bus: u8,
    slot: u8,
    function: u8,
    channel: u8,
    reserved: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_101 {
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_104 {
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_107 {
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_110 {
    reserved: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_edd_h_114 {
    ata: __SlateRecord_edd_h_115,
    atapi: __SlateRecord_edd_h_122,
    scsi: __SlateRecord_edd_h_130,
    usb: __SlateRecord_edd_h_136,
    i1394: __SlateRecord_edd_h_140,
    fibre: __SlateRecord_edd_h_144,
    i2o: __SlateRecord_edd_h_148,
    raid: __SlateRecord_edd_h_152,
    sata: __SlateRecord_edd_h_157,
    unknown: __SlateRecord_edd_h_164,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_115 {
    device: u8,
    reserved1: u8,
    reserved2: u16,
    reserved3: u32,
    reserved4: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_122 {
    device: u8,
    lun: u8,
    reserved1: u8,
    reserved2: u8,
    reserved3: u32,
    reserved4: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_130 {
    id: u16,
    lun: u64,
    reserved1: u16,
    reserved2: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_136 {
    serial_number: u64,
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_140 {
    eui: u64,
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_144 {
    wwid: u64,
    lun: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_148 {
    identity_tag: u64,
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_152 {
    array_number: u32,
    reserved1: u32,
    reserved2: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_157 {
    device: u8,
    reserved1: u8,
    reserved2: u16,
    reserved3: u32,
    reserved4: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_164 {
    reserved1: u64,
    reserved2: u64,
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
struct elf64_sym {
    st_name: u32,
    st_info: u8,
    st_other: u8,
    st_shndx: u16,
    st_value: u64,
    st_size: u64,
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
struct __SlateRecord_keyboard_c_116 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_117 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_133 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kparam_string {
    maxlen: u32,
    string: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_652 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_757 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_967 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_967_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_967_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_967_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_967_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_967_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_971 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_971_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_971_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_971_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_971_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_971_6 {
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
struct __SlateRecord_keyboard_c_1120 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_1145 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_2096 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_keyboard_c_2096_2 {
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
    pub struct __SlateBits66U0 {
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
    pub struct __SlateBits66U1 {
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
    pub struct __SlateBits757U0 {
        #[bits(1)]
        pub console: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits66U2 {
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
    pub struct __SlateBits627U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
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
    pub struct __SlateBits424U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits791U0 {
        #[bits(1)]
        pub charset: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits67U0 {
        #[bits(1)]
        pub vc_disp_ctrl: u32,
        #[bits(1)]
        pub vc_toggle_meta: u32,
        #[bits(1)]
        pub vc_decscnm: u32,
        #[bits(1)]
        pub vc_decom: u32,
        #[bits(1)]
        pub vc_decawm: u32,
        #[bits(1)]
        pub vc_deccm: u32,
        #[bits(1)]
        pub vc_decim: u32,
        #[bits(3)]
        pub vc_priv: u32,
        #[bits(1)]
        pub vc_need_wrap: u32,
        #[bits(1)]
        pub vc_can_do_color: u32,
        #[bits(2)]
        pub vc_report_mouse: u32,
        #[bits(1)]
        pub vc_bracketed_paste: u32,
        #[bits(1)]
        pub vc_utf: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits95U0 {
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
    pub struct __SlateBits96U0 {
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
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits775U0 {
        #[bits(1)]
        pub ledmode: u8,
        #[bits(4)]
        pub ledflagstate: u8,
        #[bits(3, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits775U1 {
        #[bits(4)]
        pub default_ledflagstate: u8,
        #[bits(3)]
        pub kbdmode: u8,
        #[bits(1, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits775U2 {
        #[bits(5)]
        pub modeflags: u8,
        #[bits(3, access = na)]
        pub __slate_pad_1: u8,
    }
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


// Format: __mod_device_table__kmod_<modname>__<type>__<name>
// Parts of the string `__kmod_` and `__` are used as delimiters when parsing
// a symbol in file2alias.c
// Creates an alias so file2alias.c can find device table.
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
    __slate_bits_0: __slate_bits::__SlateBits495U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_device_id {
    flags: u64,
    bustype: u16,
    vendor: u16,
    product: u16,
    version: u16,
    evbit: [u64; 1],
    keybit: [u64; 12],
    relbit: [u64; 1],
    absbit: [u64; 1],
    mscbit: [u64; 1],
    ledbit: [u64; 1],
    sndbit: [u64; 1],
    ffbit: [u64; 2],
    swbit: [u64; 1],
    propbit: [u64; 1],
    driver_info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_value {
    r#type: u16,
    code: u16,
    value: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_dev {
    name: *const u8,
    phys: *const u8,
    uniq: *const u8,
    id: input_id,
    propbit: [u64; 1],
    evbit: [u64; 1],
    keybit: [u64; 12],
    relbit: [u64; 1],
    absbit: [u64; 1],
    mscbit: [u64; 1],
    ledbit: [u64; 1],
    sndbit: [u64; 1],
    ffbit: [u64; 2],
    swbit: [u64; 1],
    hint_events_per_packet: u32,
    keycodemax: u32,
    keycodesize: u32,
    keycode: *mut (),
    setkeycode: Option<unsafe extern "C" fn(*mut input_dev, *const input_keymap_entry, *mut u32) -> i32>,
    getkeycode: Option<unsafe extern "C" fn(*mut input_dev, *mut input_keymap_entry) -> i32>,
    ff: *mut ff_device,
    poller: *mut input_dev_poller,
    repeat_key: u32,
    timer: timer_list,
    rep: [i32; 2],
    mt: *mut input_mt,
    absinfo: *mut input_absinfo,
    key: [u64; 12],
    led: [u64; 1],
    snd: [u64; 1],
    sw: [u64; 1],
    open: Option<unsafe extern "C" fn(*mut input_dev) -> i32>,
    close: Option<unsafe extern "C" fn(*mut input_dev)>,
    flush: Option<unsafe extern "C" fn(*mut input_dev, *mut file) -> i32>,
    event: Option<unsafe extern "C" fn(*mut input_dev, u32, u32, i32) -> i32>,
    grab: *mut input_handle,
    event_lock: spinlock,
    mutex: mutex,
    users: u32,
    going_away: bool,
    dev: device,
    h_list: list_head,
    node: list_head,
    num_vals: u32,
    max_vals: u32,
    vals: *mut input_value,
    devres_managed: bool,
    timestamp: [i64; 3],
    inhibited: bool,
    ready: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_handler {
    private: *mut (),
    event: Option<unsafe extern "C" fn(*mut input_handle, u32, u32, i32)>,
    events: Option<unsafe extern "C" fn(*mut input_handle, *mut input_value, u32) -> u32>,
    filter: Option<unsafe extern "C" fn(*mut input_handle, u32, u32, i32) -> bool>,
    r#match: Option<unsafe extern "C" fn(*mut input_handler, *mut input_dev) -> bool>,
    connect: Option<unsafe extern "C" fn(*mut input_handler, *mut input_dev, *const input_device_id) -> i32>,
    disconnect: Option<unsafe extern "C" fn(*mut input_handle)>,
    start: Option<unsafe extern "C" fn(*mut input_handle)>,
    passive_observer: bool,
    legacy_minors: bool,
    minor: i32,
    name: *const u8,
    id_table: *const input_device_id,
    h_list: list_head,
    node: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_handle {
    private: *mut (),
    open: i32,
    name: *const u8,
    dev: *mut input_dev,
    handler: *mut input_handler,
    handle_events: Option<unsafe extern "C" fn(*mut input_handle, *mut input_value, u32) -> u32>,
    d_node: list_head,
    h_node: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ff_device {
    upload: Option<unsafe extern "C" fn(*mut input_dev, *mut ff_effect, *mut ff_effect) -> i32>,
    erase: Option<unsafe extern "C" fn(*mut input_dev, i32) -> i32>,
    playback: Option<unsafe extern "C" fn(*mut input_dev, i32, i32) -> i32>,
    set_gain: Option<unsafe extern "C" fn(*mut input_dev, u16)>,
    set_autocenter: Option<unsafe extern "C" fn(*mut input_dev, u16)>,
    destroy: Option<unsafe extern "C" fn(*mut ff_device)>,
    stop: Option<unsafe extern "C" fn(*mut ff_device)>,
    private: *mut (),
    ffbit: [u64; 2],
    mutex: mutex,
    max_effects: i32,
    effects: *mut ff_effect,
    effect_owners: [*mut file; 0],
}


// set led state [lights, not flags]
// get led flags (not lights)
// set led flags (not lights)
#[repr(C)]
#[derive(Clone, Copy)]
struct kbentry {
    kb_table: u8,
    kb_index: u8,
    kb_value: u16,
}


// gets one entry in translation table
// sets one entry in translation table
#[repr(C)]
#[derive(Clone, Copy)]
struct kbsentry {
    kb_func: u8,
    kb_string: [u8; 512],
}


// gets one function key string entry
// sets one function key string entry
#[repr(C)]
#[derive(Clone, Copy)]
struct kbdiacr {
    diacr: u8,
    base: u8,
    result: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kbdiacrs {
    kb_cnt: u32,
    kbdiacr: [kbdiacr; 256],
}


// read kernel accent table
// write kernel accent table
#[repr(C)]
#[derive(Clone, Copy)]
struct kbdiacruc {
    diacr: u32,
    base: u32,
    result: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kbdiacrsuc {
    kb_cnt: u32,
    kbdiacruc: [kbdiacruc; 256],
}


// read kernel accent table - UCS
// write kernel accent table - UCS
#[repr(C)]
#[derive(Clone, Copy)]
struct kbkeycode {
    scancode: u32,
    keycode: u32,
}


// read kernel keycode table entry
// write kernel keycode table entry
#[repr(C)]
#[derive(Clone, Copy)]
struct kbd_repeat {
    delay: i32,
    period: i32,
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


fn char_buf_ptr(b: *mut tty_buffer, ofs: u32) -> *mut u8 {
unsafe { core::ptr::addr_of_mut!((*b).data) as *mut u8 }.wrapping_offset(ofs as isize)
}

fn flag_buf_ptr(b: *mut tty_buffer, ofs: u32) -> *mut u8 {
char_buf_ptr(b, ofs).wrapping_offset((unsafe { (*b).size }) as isize)
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
    __slate_bits_0: __slate_bits::__SlateBits757U0,
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
struct tasklet_struct {
    next: *mut tasklet_struct,
    state: u64,
    count: __SlateRecord_types_h_190,
    use_callback: bool,
    __slate_anon_4: __SlateRecord_interrupt_h_697,
    data: u64,
}


fn tasklet_schedule(t: *mut tasklet_struct) {
if !test_and_set_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*t).state) }) as *mut u64) {
unsafe { __tasklet_schedule(t) };
}
}

fn tasklet_disable_nosync(t: *mut tasklet_struct) {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*t).count) });
smp_mb__after_atomic!();
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_register_keyboard_notifier: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_unregister_keyboard_notifier: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_kd_mksound: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_vt_get_leds: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous", sym register_keyboard_notifier, sym unregister_keyboard_notifier, sym kd_mksound, sym vt_get_leds, options(att_syntax));

fn tasklet_disable(t: *mut tasklet_struct) {
tasklet_disable_nosync(t);
unsafe { tasklet_unlock_wait(t) };
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
}

fn tasklet_enable(t: *mut tasklet_struct) {
smp_mb__before_atomic!();
atomic_dec(unsafe { core::ptr::addr_of_mut!((*t).count) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct keyboard_notifier_param {
    vc: *mut vc_data,
    down: i32,
    shift: i32,
    ledstate: i32,
    value: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kbd_struct {
    lockstate: u8,
    slockstate: u8,
    __slate_bits_0: __slate_bits::__SlateBits775U0,
    __slate_bits_1: __slate_bits::__SlateBits775U1,
    __slate_bits_2: __slate_bits::__SlateBits775U2,
}


fn vc_kbd_mode(kbd__0: *mut kbd_struct, flag: i32) -> i32 {
((unsafe { (*kbd__0).__slate_bits_2.__get_modeflags() }) as i32) >> flag & 1
}

fn vc_kbd_led(kbd__0: *mut kbd_struct, flag: i32) -> i32 {
((unsafe { (*kbd__0).__slate_bits_0.__get_ledflagstate() }) as i32) >> flag & 1
}

fn set_vc_kbd_mode(__v0: *mut kbd_struct, flag: i32) {
let __v1: u8 = unsafe { (*__v0).__slate_bits_2.__get_modeflags() };
unsafe {
(*__v0).__slate_bits_2.__set_modeflags(((__v1 as i32) | 1i32 << flag) as u8);
}
}

fn set_vc_kbd_led(__v0: *mut kbd_struct, flag: i32) {
let __v1: u8 = unsafe { (*__v0).__slate_bits_0.__get_ledflagstate() };
unsafe {
(*__v0).__slate_bits_0.__set_ledflagstate(((__v1 as i32) | 1i32 << flag) as u8);
}
}

fn clr_vc_kbd_mode(__v0: *mut kbd_struct, flag: i32) {
let __v1: u8 = unsafe { (*__v0).__slate_bits_2.__get_modeflags() };
let __v2: u8 = ((__v1 as i32) & !(1i32 << flag)) as u8;
unsafe {
(*__v0).__slate_bits_2.__set_modeflags(__v2);
}
}

fn clr_vc_kbd_led(__v0: *mut kbd_struct, flag: i32) {
let __v1: u8 = unsafe { (*__v0).__slate_bits_0.__get_ledflagstate() };
let __v2: u8 = ((__v1 as i32) & !(1i32 << flag)) as u8;
unsafe {
(*__v0).__slate_bits_0.__set_ledflagstate(__v2);
}
}

fn chg_vc_kbd_lock(__v0: *mut kbd_struct, flag: i32) {
unsafe {
(*__v0).lockstate = (((*__v0).lockstate as i32) ^ 1i32 << flag) as u8;
}
}

fn chg_vc_kbd_slock(__v0: *mut kbd_struct, flag: i32) {
unsafe {
(*__v0).slockstate = (((*__v0).slockstate as i32) ^ 1i32 << flag) as u8;
}
}

fn chg_vc_kbd_led(__v0: *mut kbd_struct, flag: i32) {
let __v1: u8 = unsafe { (*__v0).__slate_bits_0.__get_ledflagstate() };
unsafe {
(*__v0).__slate_bits_0.__set_ledflagstate(((__v1 as i32) ^ 1i32 << flag) as u8);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct led_hw_trigger_type {
    dummy: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct led_classdev {
    name: *const u8,
    brightness: u32,
    max_brightness: u32,
    color: u32,
    flags: i32,
    work_flags: u64,
    brightness_set: Option<unsafe extern "C" fn(*mut led_classdev, u32)>,
    brightness_set_blocking: Option<unsafe extern "C" fn(*mut led_classdev, u32) -> i32>,
    brightness_get: Option<unsafe extern "C" fn(*mut led_classdev) -> u32>,
    blink_set: Option<unsafe extern "C" fn(*mut led_classdev, *mut u64, *mut u64) -> i32>,
    pattern_set: Option<unsafe extern "C" fn(*mut led_classdev, *mut led_pattern, u32, i32) -> i32>,
    pattern_clear: Option<unsafe extern "C" fn(*mut led_classdev) -> i32>,
    dev: *mut device,
    groups: *mut *const attribute_group,
    node: list_head,
    default_trigger: *const u8,
    blink_delay_on: u64,
    blink_delay_off: u64,
    blink_timer: timer_list,
    blink_brightness: i32,
    new_blink_brightness: i32,
    flash_resume: Option<unsafe extern "C" fn(*mut led_classdev)>,
    wq: *mut workqueue_struct,
    set_brightness_work: work_struct,
    delayed_set_value: i32,
    delayed_delay_on: u64,
    delayed_delay_off: u64,
    trigger_lock: rw_semaphore,
    trigger: *mut led_trigger,
    trig_list: list_head,
    trigger_data: *mut (),
    activated: bool,
    trigger_type: *mut led_hw_trigger_type,
    hw_control_trigger: *const u8,
    hw_control_is_supported: Option<unsafe extern "C" fn(*mut led_classdev, u64) -> i32>,
    hw_control_set: Option<unsafe extern "C" fn(*mut led_classdev, u64) -> i32>,
    hw_control_get: Option<unsafe extern "C" fn(*mut led_classdev, *mut u64) -> i32>,
    hw_control_get_device: Option<unsafe extern "C" fn(*mut led_classdev) -> *mut device>,
    led_access: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct led_trigger {
    name: *const u8,
    activate: Option<unsafe extern "C" fn(*mut led_classdev) -> i32>,
    deactivate: Option<unsafe extern "C" fn(*mut led_classdev)>,
    brightness: u32,
    trigger_type: *mut led_hw_trigger_type,
    leddev_list_lock: spinlock,
    led_cdevs: list_head,
    next_trig: list_head,
    groups: *mut *const attribute_group,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct led_pattern {
    delta_t: u32,
    brightness: i32,
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
fn show_mem() {
unsafe { __show_mem(0u32, core::ptr::null::<__SlateRecord_nodemask_types_h_17>(), 3i32) };
}

// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
fn show_state() {
unsafe { show_state_filter(0u32) };
}

fn tty_insert_flip_string_fixed_flag(port: *mut tty_port, chars: *const u8, mut flag: u8, size: u64) -> u64 {
unsafe { __tty_insert_flip_string_flags(port, chars, core::ptr::addr_of_mut!(flag) as *const u8, false, size) }
}

fn tty_insert_flip_char(port: *mut tty_port, mut ch: u8, mut flag: u8) -> u64 {
let tb: *mut tty_buffer = unsafe { (*port).buf.tail };
let change: i32 = (!unsafe { (*tb).flags } && (flag as i32) != 0) as i32;
if change == 0 && (unsafe { (*tb).used }) < unsafe { (*tb).size } {
if unsafe { (*tb).flags } {
unsafe {
*flag_buf_ptr(tb, (*tb).used) = flag;
}
}
let __v1: u32 = unsafe { (*tb).used };
unsafe {
(*tb).used = __v1.wrapping_add(1);
}
unsafe {
*char_buf_ptr(tb, __v1) = ch;
}
return 1;
}
unsafe { __tty_insert_flip_string_flags(port, core::ptr::addr_of_mut!(ch) as *const u8, core::ptr::addr_of_mut!(flag) as *const u8, false, 1u64) }
}

fn tty_insert_flip_string(port: *mut tty_port, chars: *const u8, size: u64) -> u64 {
tty_insert_flip_string_fixed_flag(port, chars, 0u8, size)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vt_mode {
    mode: u8,
    waitv: u8,
    relsig: i16,
    acqsig: i16,
    frsig: i16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vc_state {
    x: u32,
    y: u32,
    color: u8,
    Gx_charset: [u8; 2],
    __slate_bits_0: __slate_bits::__SlateBits791U0,
    intensity: u32,
    italic: bool,
    underline: bool,
    blink: bool,
    reverse: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vc_font {
    width: u32,
    height: u32,
    charcount: u32,
    data: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vc_data {
    port: tty_port,
    state: vc_state,
    saved_state: vc_state,
    vc_num: u16,
    vc_cols: u32,
    vc_rows: u32,
    vc_size_row: u32,
    vc_scan_lines: u32,
    vc_cell_height: u32,
    vc_origin: u64,
    vc_scr_end: u64,
    vc_visible_origin: u64,
    vc_top: u32,
    vc_bottom: u32,
    vc_sw: *const consw,
    vc_screenbuf: *mut u16,
    vc_screenbuf_size: u32,
    vc_mode: u8,
    vc_attr: u8,
    vc_def_color: u8,
    vc_ulcolor: u8,
    vc_itcolor: u8,
    vc_halfcolor: u8,
    vc_cursor_type: u32,
    vc_complement_mask: u16,
    vc_s_complement_mask: u16,
    vc_pos: u64,
    vc_hi_font_mask: u16,
    vc_font: vc_font,
    vc_video_erase_char: u16,
    vc_state: u32,
    vc_npar: u32,
    vc_par: [u32; 16],
    vt_mode: vt_mode,
    vt_pid: *mut pid,
    vt_newvt: i32,
    paste_wait: wait_queue_head,
    __slate_bits_0: __slate_bits::__SlateBits67U0,
    vc_utf_count: u8,
    vc_utf_char: i32,
    vc_tab_stop: [u64; 4],
    vc_palette: [u8; 48],
    vc_translate: *mut u16,
    vc_bell_pitch: u32,
    vc_bell_duration: u32,
    vc_cur_blink_ms: u16,
    vc_display_fg: *mut *mut vc_data,
    uni_pagedict: *mut uni_pagedict,
    uni_pagedict_loc: *mut *mut uni_pagedict,
    vc_uni_lines: *mut *mut u32,
    vc_saved_screen: *mut u16,
    vc_saved_uni_lines: *mut *mut u32,
    vc_saved_cols: u32,
    vc_saved_rows: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vc {
    d: *mut vc_data,
    SAK_work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vt_spawn_console {
    lock: spinlock,
    pid: *mut pid,
    sig: i32,
}


#[unsafe(no_mangle)]
extern "C" fn vt_do_kdgkb_ioctl(cmd: i32, mut user_kdgkb: *mut kbsentry, perm: i32) -> i32 {
let mut __slate_compound_26: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_28: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_unlock_553: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_553) as *mut *mut __SlateRecord_spinlock_h_642;
let mut __UNIQUE_ID_guard_552: core::mem::MaybeUninit<__SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_552) as *mut __SlateRecord_spinlock_h_642;
let mut kbs: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut u8 = core::ptr::addr_of_mut!(kbs) as *mut *mut u8;
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_unlock_551: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_551) as *mut *mut __SlateRecord_spinlock_h_642;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_642;
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut kbs: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut u8 = core::ptr::addr_of_mut!(kbs) as *mut *mut u8;
// size should have been a struct member
let mut len: i64 = unsafe { core::mem::zeroed() };
let mut __mask: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u64 = core::ptr::addr_of_mut!(__mask) as *mut u64;
let mut __sz: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut i32 = core::ptr::addr_of_mut!(__sz) as *mut i32;
let mut __idx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(__idx) as *mut i32;
let mut _mask: u64 = unsafe { core::mem::zeroed() };
let mut _i: u8 = unsafe { core::mem::zeroed() };
let mut __val_gu: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u8 = core::ptr::addr_of_mut!(__val_gu) as *mut u8;
let mut __ret_gu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut i32 = core::ptr::addr_of_mut!(__ret_gu) as *mut i32;
let mut kb_func: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut u8 = core::ptr::addr_of_mut!(kb_func) as *mut u8;
let __slate_cleanup_ptr_11 = __slate_slot_11;
let __slate_cleanup_armed_11 = core::cell::Cell::new(false);
let __slate_cleanup_11 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_11.get() {
unsafe { __free_kfree(__slate_cleanup_ptr_11 as *mut _) };
}
});
let __slate_cleanup_ptr_6 = __slate_slot_6;
let __slate_cleanup_armed_6 = core::cell::Cell::new(false);
let __slate_cleanup_6 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_6.get() {
unsafe { class_spinlock_irqsave_destructor(__slate_cleanup_ptr_6 as *mut _) };
}
});
let __slate_cleanup_ptr_5 = __slate_slot_5;
let __slate_cleanup_armed_5 = core::cell::Cell::new(false);
let __slate_cleanup_5 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_5.get() {
unsafe { __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_5 as *mut _) };
}
});
let __slate_cleanup_ptr_2 = __slate_slot_2;
let __slate_cleanup_armed_2 = core::cell::Cell::new(false);
let __slate_cleanup_2 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_2.get() {
unsafe { __free_kfree(__slate_cleanup_ptr_2 as *mut _) };
}
});
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_spinlock_irqsave_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes_3 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 1usize);
might_fault();
let __slate_auto_bytes_4 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 1usize);
__chk_user_ptr!();
let __t2: u64;
let __t3: u32;
core::arch::asm!("call __get_user_{0}", const 1 as u64, inlateout("ax") core::ptr::addr_of_mut!((*user_kdgkb).kb_func) as u64 => __t2, lateout("dx") __t3, options(att_syntax));
*__slate_slot_22 = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
*__slate_slot_21 = unsafe { core::mem::transmute::<u8, u8>(__t3 as u8) };
instrument_get_user__2!(*__slate_slot_21);
*__slate_slot_25 = *__slate_slot_21;
let __v24: i64 = *__slate_slot_22 as i64;
if __v24 != 0 {
return 14i32.wrapping_neg();
} else {
_i = *__slate_slot_25;
let _s: i32 = 256;
core::ptr::write(__slate_slot_15, _i as i32);
core::ptr::write(__slate_slot_14, _s);
let __slate_auto_bytes_2 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __t1: u64;
core::arch::asm!("cmp {1:e},{2:e}; sbb {0:r},{0:r}", lateout(reg) __t1, in(reg) *__slate_slot_14 as u32, in(reg) *__slate_slot_15 as u32, options(nostack, att_syntax));
*__slate_slot_13 = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
_mask = *__slate_slot_13;
while false {
}
while false {
}
*__slate_slot_25 = (((_i as i32) as u64) & _mask) as u8;
let __t0: i32 = cmd;
if __t0 == 19272 {
len = core::mem::size_of::<[u8; 512]>() as i64;
__slate_cleanup_armed_11.set(true);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_7 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_8 = _kmalloc_noprof(len as u64, 3264u32, *{
__slate_compound_26 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_26)
});
alloc_tag_restore!();
} else {
*__slate_slot_8 = _kmalloc_noprof(len as u64, 3264u32, *{
__slate_compound_28 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_28)
});
}
*__slate_slot_11 = *__slate_slot_8 as *mut u8;
if (*__slate_slot_11).is_null() {
return 12i32.wrapping_neg();
} else {
core::ptr::write(__slate_slot_6, class_spinlock_irqsave_constructor(core::ptr::addr_of_mut!(func_buf_lock)));
__slate_cleanup_armed_6.set(true);
core::ptr::write(__slate_slot_5, (core::ptr::addr_of_mut!(func_buf_lock) as u64) as *mut __SlateRecord_spinlock_h_642);
__slate_cleanup_armed_5.set(true);
if !class_spinlock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_6))).is_null() || !class_spinlock_irqsave_is_conditional {
if true {
let __v4: *mut u8 = func_table[(*__slate_slot_25 as i32) as usize];
len = sized_strscpy(*__slate_slot_11, (if !__v4.is_null() { __v4 } else { c"".as_ptr() as *mut u8 }) as *const u8, len.wrapping_add(__must_be_cstr__5!() as i64).wrapping_add(__must_be_cstr__2!() as i64) as u64);
}
}
__slate_cleanup_armed_5.set(false);
__class_spinlock_irqsave_cleanup_ctx(__slate_slot_5 as *mut _);
__slate_cleanup_armed_6.set(false);
class_spinlock_irqsave_destructor(__slate_slot_6 as *mut _);
if len < 0 {
return 28i32.wrapping_neg();
} else {
if copy_to_user(((*user_kdgkb).kb_string.as_mut_ptr() as *mut u8) as *mut (), *__slate_slot_11 as *const (), len.wrapping_add(1) as u64) != 0 {
return 14i32.wrapping_neg();
} else {
return 0;
}
}
}
} else {
if __t0 == 19273 {
if perm == 0 {
__v3 = true;
} else {
__v3 = !capable(26i32);
}
if __v3 {
return 1i32.wrapping_neg();
} else {
core::ptr::write(__slate_slot_2, strndup_user(((*user_kdgkb).kb_string.as_mut_ptr() as *mut u8) as *const u8, core::mem::size_of::<[u8; 512]>() as i64));
__slate_cleanup_armed_2.set(true);
if IS_ERR(*__slate_slot_2 as *const ()) {
return PTR_ERR(*__slate_slot_2 as *const ()) as i32;
} else {
core::ptr::write(__slate_slot_1, class_spinlock_irqsave_constructor(core::ptr::addr_of_mut!(func_buf_lock)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!(func_buf_lock) as u64) as *mut __SlateRecord_spinlock_h_642);
__slate_cleanup_armed_0.set(true);
*__slate_slot_2 = vt_kdskbsent(*__slate_slot_2, *__slate_slot_25);
return 0;
}
}
} else {
return 0;
}
}
}
}
}

fn get_irq_regs() -> *mut pt_regs {
let mut __v0: *mut pt_regs;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v1: *mut pt_regs;
let mut pscr_ret__: *mut pt_regs = unsafe { core::mem::zeroed() };
__verify_pcpu_ptr__3!();
match (const { core::mem::size_of::<*mut pt_regs>() }) as u64 {
1 => {
let mut __v3: *mut pt_regs;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym __irq_regs, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v3 = (pfo_val__ as u64) as *mut pt_regs;
pscr_ret__ = __v3;
}
2 => {
let mut __v5: *mut pt_regs;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym __irq_regs, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v5 = (pfo_val__ as u64) as *mut pt_regs;
pscr_ret__ = __v5;
}
4 => {
let mut __v6: *mut pt_regs;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym __irq_regs, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v6 = (pfo_val__ as u64) as *mut pt_regs;
pscr_ret__ = __v6;
}
8 => {
let mut __v7: *mut pt_regs;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym __irq_regs, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v7 = pfo_val__ as *mut pt_regs;
pscr_ret__ = __v7;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v1 = pscr_ret__;
__v0 = __v1;
__v0
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct boot_e820_entry {
    addr: u64,
    size: u64,
    r#type: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct screen_info {
    orig_x: u8,
    orig_y: u8,
    ext_mem_k: u16,
    orig_video_page: u16,
    orig_video_mode: u8,
    orig_video_cols: u8,
    flags: u8,
    unused2: u8,
    orig_video_ega_bx: u16,
    unused3: u16,
    orig_video_lines: u8,
    orig_video_isVGA: u8,
    orig_video_points: u16,
    lfb_width: u16,
    lfb_height: u16,
    lfb_depth: u16,
    lfb_base: u32,
    lfb_size: u32,
    cl_magic: u16,
    cl_offset: u16,
    lfb_linelength: u16,
    red_size: u8,
    red_pos: u8,
    green_size: u8,
    green_pos: u8,
    blue_size: u8,
    blue_pos: u8,
    rsvd_size: u8,
    rsvd_pos: u8,
    vesapm_seg: u16,
    vesapm_off: u16,
    pages: u16,
    vesa_attributes: u16,
    capabilities: u32,
    ext_lfb_base: u32,
    _reserved: [u8; 2],
}


// Monochrome Text Display
// CGA Display
// EGA/VGA in Monochrome Mode
// EGA in Color Mode
// VGA+ in Color Mode
// VESA VGA in graphic mode
// ACER PICA-61 local S3 video
// MIPS Magnum 4000 G364 video
// Various SGI graphics hardware
// DEC TGA
// Sun frame buffer.
// Sun PCI based frame buffer.
// PowerMacintosh frame buffer.
// EFI graphic mode
// Frame buffer base is 64-bit
#[repr(C)]
#[derive(Clone, Copy)]
struct apm_bios_info {
    version: u16,
    cseg: u16,
    offset: u32,
    cseg_16: u16,
    dseg: u16,
    flags: u16,
    cseg_len: u16,
    cseg_16_len: u16,
    dseg_len: u16,
}


// max number of signatures to store
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct edd_device_params {
    length: u16,
    info_flags: u16,
    num_default_cylinders: u32,
    num_default_heads: u32,
    sectors_per_track: u32,
    number_of_sectors: u64,
    bytes_per_sector: u16,
    dpte_ptr: u32,
    key: u16,
    device_path_info_length: u8,
    reserved2: u8,
    reserved3: u16,
    host_bus_type: [u8; 4],
    interface_type: [u8; 8],
    interface_path: __SlateRecord_edd_h_87,
    device_path: __SlateRecord_edd_h_114,
    reserved4: u8,
    checksum: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct edd_info {
    device: u8,
    version: u8,
    interface_support: u16,
    legacy_max_cylinder: u16,
    legacy_max_head: u8,
    legacy_sectors_per_track: u8,
    params: edd_device_params,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ist_info {
    signature: u32,
    command: u32,
    event: u32,
    perf_level: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct edid_info {
    dummy: [u8; 128],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct setup_header {
    setup_sects: u8,
    root_flags: u16,
    syssize: u32,
    ram_size: u16,
    vid_mode: u16,
    root_dev: u16,
    boot_flag: u16,
    jump: u16,
    header: u32,
    version: u16,
    realmode_swtch: u32,
    start_sys_seg: u16,
    kernel_version: u16,
    type_of_loader: u8,
    loadflags: u8,
    setup_move_size: u16,
    code32_start: u32,
    ramdisk_image: u32,
    ramdisk_size: u32,
    bootsect_kludge: u32,
    heap_end_ptr: u16,
    ext_loader_ver: u8,
    ext_loader_type: u8,
    cmd_line_ptr: u32,
    initrd_addr_max: u32,
    kernel_alignment: u32,
    relocatable_kernel: u8,
    min_alignment: u8,
    xloadflags: u16,
    cmdline_size: u32,
    hardware_subarch: u32,
    hardware_subarch_data: u64,
    payload_offset: u32,
    payload_length: u32,
    setup_data: u64,
    pref_address: u64,
    init_size: u32,
    handover_offset: u32,
    kernel_info_offset: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sys_desc_table {
    length: u16,
    table: [u8; 14],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct olpc_ofw_header {
    ofw_magic: u32,
    ofw_version: u32,
    cif_handler: u32,
    irq_desc_table: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct efi_info {
    efi_loader_signature: u32,
    efi_systab: u32,
    efi_memdesc_size: u32,
    efi_memdesc_version: u32,
    efi_memmap: u32,
    efi_memmap_size: u32,
    efi_systab_hi: u32,
    efi_memmap_hi: u32,
}


// This is the maximum number of entries in struct boot_params::e820_table
// (the zeropage), which is part of the x86 boot protocol ABI:
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct boot_params {
    screen_info: screen_info,
    apm_bios_info: apm_bios_info,
    _pad2: [u8; 4],
    tboot_addr: u64,
    ist_info: ist_info,
    acpi_rsdp_addr: u64,
    _pad3: [u8; 8],
    hd0_info: [u8; 16],
    hd1_info: [u8; 16],
    sys_desc_table: sys_desc_table,
    olpc_ofw_header: olpc_ofw_header,
    ext_ramdisk_image: u32,
    ext_ramdisk_size: u32,
    ext_cmd_line_ptr: u32,
    _pad4: [u8; 112],
    cc_blob_address: u32,
    edid_info: edid_info,
    efi_info: efi_info,
    alt_mem_k: u32,
    scratch: u32,
    e820_entries: u8,
    eddbuf_entries: u8,
    edd_mbr_sig_buf_entries: u8,
    kbd_status: u8,
    secure_boot: u8,
    _pad5: [u8; 2],
    sentinel: u8,
    _pad6: [u8; 1],
    hdr: setup_header,
    _pad7: [u8; 36],
    edd_mbr_sig_buffer: [u32; 16],
    e820_table: [boot_e820_entry; 128],
    _pad8: [u8; 48],
    eddbuf: [edd_info; 6],
    _pad9: [u8; 276],
}


fn kbd_defleds() -> i32 {
if ((unsafe { boot_params.kbd_status }) as i32) & 32 != 0 { 2i32 } else { 0 }
}

// Exported functions/variables
// Handler Tables.
static mut k_handler: __SlateAlign16<[Option<unsafe extern "C" fn(*mut vc_data, u8, u8)>; 16]> = __SlateAlign16([Some(k_self), Some(k_fn), Some(k_spec), Some(k_pad), Some(k_dead), Some(k_cons), Some(k_cur), Some(k_shift), Some(k_meta), Some(k_ascii), Some(k_lock), Some(k_lowercase), Some(k_slock), Some(k_dead2), Some(k_brl), Some(k_csi)]);

static mut fn_handler: __SlateAlign16<[Option<unsafe extern "C" fn(*mut vc_data)>; 20]> = __SlateAlign16([Some(fn_null), Some(fn_enter), Some(fn_show_ptregs), Some(fn_show_mem), Some(fn_show_state), Some(fn_send_intr), Some(fn_lastcons), Some(fn_caps_toggle), Some(fn_num), Some(fn_hold), Some(fn_scroll_forw), Some(fn_scroll_back), Some(fn_boot_it), Some(fn_caps_on), Some(fn_compose), Some(fn_SAK), Some(fn_dec_console), Some(fn_inc_console), Some(fn_spawn_con), Some(fn_bare_num)]);

// Variables exported for vt_ioctl.c
#[unsafe(no_mangle)]
static mut vt_spawn_con: vt_spawn_console = {
let mut __t0: vt_spawn_console = unsafe { core::mem::zeroed() };
__t0.lock = spinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t1.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t2: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t2.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t2
} } };
__t1
} };
__t0.pid = core::ptr::null_mut::<pid>();
__t0.sig = 0i32;
__t0
};

// Internal Data.
static mut kbd_table: __SlateAlign16<[kbd_struct; 63]> = __SlateAlign16(unsafe { core::mem::zeroed() });

static mut kbd: *mut kbd_struct = unsafe { core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct };

/// maximum values each key_handler can handle
static max_vals: __SlateAlign16<[u8; 16]> = __SlateAlign16([255u8, ((core::mem::size_of::<[*mut u8; 256]>() as u64) / (core::mem::size_of::<*mut u8>() as u64)).wrapping_add(((core::mem::size_of::<__SlateRecord_keyboard_c_116>() as u64) as i32) as u64).wrapping_sub(1u64) as u8, ((core::mem::size_of::<[Option<unsafe extern "C" fn(*mut vc_data)>; 20]>() as u64) / (core::mem::size_of::<Option<unsafe extern "C" fn(*mut vc_data)>>() as u64)).wrapping_add(((core::mem::size_of::<__SlateRecord_keyboard_c_117>() as u64) as i32) as u64).wrapping_sub(1u64) as u8, 19u8, 26u8, 255u8, 3u8, 8u8, 255u8, 25u8, 8u8, 255u8, 8u8, 255u8, 10u8, 99u8]);

static NR_TYPES: i32 = ((core::mem::size_of::<[u8; 16]>() as u64) / (core::mem::size_of::<u8>() as u64)).wrapping_add(((core::mem::size_of::<__SlateRecord_keyboard_c_133>() as u64) as i32) as u64) as i32;

static mut keyboard_tasklet: tasklet_struct = {
let mut __t0: tasklet_struct = unsafe { core::mem::zeroed() };
__t0.count = __SlateRecord_types_h_190 { counter: 1i32 };
__t0.use_callback = 1i32 != 0i32;
__t0.__slate_anon_4 = __SlateRecord_interrupt_h_697 { callback: Some(kbd_bh) };
__t0
};

static mut kbd_handler: input_handler = {
let mut __t0: input_handler = unsafe { core::mem::zeroed() };
__t0.event = Some(kbd_event);
__t0.r#match = Some(kbd_match);
__t0.connect = Some(kbd_connect);
__t0.disconnect = Some(kbd_disconnect);
__t0.start = Some(kbd_start);
__t0.name = (b"kbd\0".as_ptr() as *mut u8) as *const u8;
__t0.id_table = unsafe { core::ptr::addr_of!(kbd_ids.0) as *const input_device_id };
__t0
};

static mut kbd_event_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

static mut led_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

/// guard 'func_buf'  and friends
static mut func_buf_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

/// keyboard key bitmap
static mut key_down: __SlateAlign16<[u64; 12]> = __SlateAlign16(unsafe { core::mem::zeroed() });

/// shift state counters..
static mut shift_down: [u8; 9] = unsafe { core::mem::zeroed() };

static mut dead_key_next: bool = unsafe { core::mem::zeroed() };

/// Handles a number being assembled on the number pad
static mut npadch_active: bool = unsafe { core::mem::zeroed() };

static mut npadch_value: u32 = unsafe { core::mem::zeroed() };

static mut diacr: u32 = unsafe { core::mem::zeroed() };

/// flag telling character repeat
static mut rep: bool = unsafe { core::mem::zeroed() };

static mut shift_state: i32 = 0;

/// undefined
static mut ledstate: u32 = 1u32.wrapping_neg();

static mut ledioctl: u8 = unsafe { core::mem::zeroed() };

static mut vt_switch: bool = unsafe { core::mem::zeroed() };

/// Notifier list for console keyboard events
static mut keyboard_notifier_list: atomic_notifier_head = {
let mut __t0: atomic_notifier_head = unsafe { core::mem::zeroed() };
__t0.lock = spinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t1.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t2: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t2.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t2
} } };
__t1
} };
__t0.head = core::ptr::null_mut::<notifier_block>();
__t0
};

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.keyboard.register_keyboard_notifier")]
extern "C" fn register_keyboard_notifier(nb: *mut notifier_block) -> i32 {
unsafe { atomic_notifier_chain_register(core::ptr::addr_of_mut!(keyboard_notifier_list), nb) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_register_keyboard_notifier_501: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut notifier_block) -> i32>, *mut ()>(Some(register_keyboard_notifier)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.keyboard.unregister_keyboard_notifier")]
extern "C" fn unregister_keyboard_notifier(nb: *mut notifier_block) -> i32 {
unsafe { atomic_notifier_chain_unregister(core::ptr::addr_of_mut!(keyboard_notifier_list), nb) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_unregister_keyboard_notifier_502: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut notifier_block) -> i32>, *mut ()>(Some(unregister_keyboard_notifier)) };

// Translation of scancodes to keycodes. We set them on only the first
// keyboard in the list that accepts the scancode and keycode.
// Explanation for not choosing the first attached keyboard anymore:
//  USB keyboards for example have two event devices: one for all "normal"
//  keys and one for extra function keys (like "volume up", "make coffee",
//  etc.). So this means that scancodes for the extra function keys won't
//  be valid for the first event device, but will be for the second.
#[repr(C)]
#[derive(Clone, Copy)]
struct getset_keycode_data {
    ke: input_keymap_entry,
    error: i32,
}


#[unsafe(link_section = ".text.slate_distinct.keyboard.getkeycode_helper")]
extern "C" fn getkeycode_helper(handle: *mut input_handle, data: *mut ()) -> i32 {
unsafe {
(*(data as *mut getset_keycode_data)).error = input_get_keycode((*handle).dev, core::ptr::addr_of_mut!((*(data as *mut getset_keycode_data)).ke));
}
return ((unsafe { (*(data as *mut getset_keycode_data)).error }) == 0) as i32; // stop as soon as we successfully get one
}

fn getkeycode(mut scancode: u32) -> i32 {
let mut d: getset_keycode_data = getset_keycode_data { ke: {
let mut __t0: input_keymap_entry = unsafe { core::mem::zeroed() };
__t0.flags = 0u8;
__t0.len = core::mem::size_of::<u32>() as u8;
__t0.keycode = 0u32;
__t0
}, error: 19i32.wrapping_neg() };
unsafe { memcpy(d.ke.scancode.as_mut_ptr() as *mut (), core::ptr::addr_of_mut!(scancode) as *const (), core::mem::size_of::<u32>() as u64) };
unsafe { input_handler_for_each_handle(core::ptr::addr_of_mut!(kbd_handler), core::ptr::addr_of_mut!(d) as *mut (), Some(getkeycode_helper)) };
let __v0: i32 = d.error;
(if __v0 != 0 { __v0 as u32 } else { d.ke.keycode }) as i32
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.setkeycode_helper")]
extern "C" fn setkeycode_helper(handle: *mut input_handle, data: *mut ()) -> i32 {
unsafe {
(*(data as *mut getset_keycode_data)).error = input_set_keycode((*handle).dev, core::ptr::addr_of_mut!((*(data as *mut getset_keycode_data)).ke) as *const input_keymap_entry);
}
return ((unsafe { (*(data as *mut getset_keycode_data)).error }) == 0) as i32; // stop as soon as we successfully set one
}

fn setkeycode(mut scancode: u32, keycode: u32) -> i32 {
let mut d: getset_keycode_data = getset_keycode_data { ke: {
let mut __t0: input_keymap_entry = unsafe { core::mem::zeroed() };
__t0.flags = 0u8;
__t0.len = core::mem::size_of::<u32>() as u8;
__t0.keycode = keycode;
__t0
}, error: 19i32.wrapping_neg() };
unsafe { memcpy(d.ke.scancode.as_mut_ptr() as *mut (), core::ptr::addr_of_mut!(scancode) as *const (), core::mem::size_of::<u32>() as u64) };
unsafe { input_handler_for_each_handle(core::ptr::addr_of_mut!(kbd_handler), core::ptr::addr_of_mut!(d) as *mut (), Some(setkeycode_helper)) };
d.error
}

// Making beeps and bells. Note that we prefer beeps to bells, but when
// shutting the sound off we do both.
#[unsafe(link_section = ".text.slate_distinct.keyboard.kd_sound_helper")]
extern "C" fn kd_sound_helper(handle: *mut input_handle, data: *mut ()) -> i32 {
let hz: *mut u32 = data as *mut u32;
let dev: *mut input_dev = unsafe { (*handle).dev };
if test_bit!(18i32, unsafe { (*dev).evbit.as_mut_ptr() as *mut u64 }) != 0 {
if test_bit!(2i32, unsafe { (*dev).sndbit.as_mut_ptr() as *mut u64 }) != 0 {
unsafe { input_inject_event(handle, 18u32, 2u32, *hz as i32) };
if (unsafe { *hz }) != 0 {
return 0;
}
}
if test_bit!(1i32, unsafe { (*dev).sndbit.as_mut_ptr() as *mut u64 }) != 0 {
unsafe { input_inject_event(handle, 18u32, 1u32, if *hz != 0 { 1i32 } else { 0i32 }) };
}
}
0
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.kd_nosound")]
extern "C" fn kd_nosound(unused: *mut timer_list) {
unsafe { input_handler_for_each_handle(core::ptr::addr_of_mut!(kbd_handler), core::ptr::addr_of_mut!(zero) as *mut (), Some(kd_sound_helper)) };
}

static mut zero: u32 = unsafe { core::mem::zeroed() };

static mut kd_mksound_timer: timer_list = {
let mut __t0: timer_list = unsafe { core::mem::zeroed() };
__t0.entry = {
let mut __t1: hlist_node = unsafe { core::mem::zeroed() };
__t1.next = ((unsafe { ((768i32 as *mut ()) as *const u8).wrapping_offset(16045481047390945280u64 as isize) }) as *mut ()) as *mut hlist_node;
__t1
};
__t0.function = Some(kd_nosound);
__t0.flags = 0u32;
__t0
};

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.keyboard.kd_mksound")]
extern "C" fn kd_mksound(mut hz: u32, ticks: u32) {
unsafe { timer_delete_sync(core::ptr::addr_of_mut!(kd_mksound_timer)) };
unsafe { input_handler_for_each_handle(core::ptr::addr_of_mut!(kbd_handler), core::ptr::addr_of_mut!(hz) as *mut (), Some(kd_sound_helper)) };
if hz != 0 && ticks != 0 {
unsafe { mod_timer(core::ptr::addr_of_mut!(kd_mksound_timer), core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(ticks as u64)) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_kd_mksound_503: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, u32)>, *mut ()>(Some(kd_mksound)) };

// Setting the keyboard rate.
#[unsafe(link_section = ".text.slate_distinct.keyboard.kbd_rate_helper")]
extern "C" fn kbd_rate_helper(handle: *mut input_handle, data: *mut ()) -> i32 {
let dev: *mut input_dev = unsafe { (*handle).dev };
let rpt: *mut kbd_repeat = data as *mut kbd_repeat;
if test_bit!(20i32, unsafe { (*dev).evbit.as_mut_ptr() as *mut u64 }) != 0 {
if (unsafe { (*rpt.wrapping_offset(0 as isize)).delay }) > 0 {
unsafe { input_inject_event(handle, 20u32, 0u32, (*rpt.wrapping_offset(0 as isize)).delay) };
}
if (unsafe { (*rpt.wrapping_offset(0 as isize)).period }) > 0 {
unsafe { input_inject_event(handle, 20u32, 1u32, (*rpt.wrapping_offset(0 as isize)).period) };
}
unsafe {
(*rpt.wrapping_offset(1 as isize)).delay = *((*dev).rep.as_mut_ptr() as *mut i32).wrapping_offset(0 as isize);
}
unsafe {
(*rpt.wrapping_offset(1 as isize)).period = *((*dev).rep.as_mut_ptr() as *mut i32).wrapping_offset(1 as isize);
}
}
0
}

#[unsafe(no_mangle)]
extern "C" fn kbd_rate(rpt: *mut kbd_repeat) -> i32 {
let mut data: __SlateAlign16<[kbd_repeat; 2]> = __SlateAlign16({
let mut __t0: [kbd_repeat; 2] = unsafe { core::mem::zeroed() };
__t0[0] = unsafe { *rpt };
__t0
});
unsafe { input_handler_for_each_handle(core::ptr::addr_of_mut!(kbd_handler), data.0.as_mut_ptr() as *mut (), Some(kbd_rate_helper)) };
unsafe {
*rpt = data.0[1];
}
// Copy currently used settings
0
}

/// Helper Functions.
fn put_queue(vc: *mut vc_data, ch: i32) {
tty_insert_flip_char(unsafe { core::ptr::addr_of_mut!((*vc).port) }, ch as u8, 0u8);
unsafe { tty_flip_buffer_push(core::ptr::addr_of_mut!((*vc).port)) };
}

fn puts_queue(vc: *mut vc_data, cp: *const u8) {
tty_insert_flip_string(unsafe { core::ptr::addr_of_mut!((*vc).port) }, cp as *const u8, unsafe { strlen(cp) });
unsafe { tty_flip_buffer_push(core::ptr::addr_of_mut!((*vc).port)) };
}

fn applkey(vc: *mut vc_data, key: i32, mode: u8) {
unsafe {
buf[1] = (if mode != 0 { 79i32 } else { 91i32 }) as u8;
}
unsafe {
buf[2] = key as u8;
}
puts_queue(vc, (unsafe { core::ptr::addr_of_mut!(buf) as *mut u8 }) as *const u8);
}

static mut buf: [u8; 4] = [27, 79, 0, 0];

/// Many other routines do put_queue, but I think either
/// they produce ASCII, or they produce some user-assigned
/// string, and in both cases we might assume that it is
/// in utf-8 already.
fn to_utf8(vc: *mut vc_data, c: u32) {
if c < 128 {
put_queue(vc, c as i32);
} else {
if c < 2048 {
// 110***** 10******
put_queue(vc, (192 | c >> 6i32) as i32);
put_queue(vc, (128 | c & 63) as i32);
} else {
if c < 65536 {
if c >= 55296 && c < 57344 {
return;
}
if c == 65535 {
return;
}
// 1110**** 10****** 10******
put_queue(vc, (224 | c >> 12i32) as i32);
put_queue(vc, (128 | c >> 6i32 & 63) as i32);
put_queue(vc, (128 | c & 63) as i32);
} else {
if c < 1114112 {
// 11110*** 10****** 10****** 10******
put_queue(vc, (240 | c >> 18i32) as i32);
put_queue(vc, (128 | c >> 12i32 & 63) as i32);
put_queue(vc, (128 | c >> 6i32 & 63) as i32);
put_queue(vc, (128 | c & 63) as i32);
}
}
}
}
//  0*******
}

fn put_queue_utf8(vc: *mut vc_data, value: u32) {
if ((unsafe { (*kbd).__slate_bits_1.__get_kbdmode() }) as i32) == 3 {
to_utf8(vc, value);
} else {
let c: i32 = unsafe { conv_uni_to_8bit(value) };
if c != 1i32.wrapping_neg() {
put_queue(vc, c);
}
}
}

/// FIXME: review locking for vt.c callers
fn set_leds() {
tasklet_schedule(unsafe { core::ptr::addr_of_mut!(keyboard_tasklet) });
}

// Called after returning from RAW mode or when changing consoles - recompute
// shift_down[] and shift_state from key_down[] maybe called when keymap is
// undefined, so that shiftkey release is seen. The caller must hold the
// kbd_event_lock.
fn do_compute_shiftstate() {
let mut k: u32 = 0;
let mut sym: u32 = 0;
let mut val: u32 = 0;
unsafe {
shift_state = 0;
}
unsafe { memset((core::ptr::addr_of_mut!(shift_down) as *mut u8) as *mut (), 0i32, core::mem::size_of::<[u8; 9]>() as u64) };
loop {
let mut __v10: i32;
let __v0: i32 = 256;
let __v1: i32 = 768i32;
loop {
if (if 1i32.wrapping_neg() < 1 { 2i32.wrapping_add(((unsafe { __slate_intrinsic_1cfe0293d0a2af8d((__v0 as i64) >= 0) }) && (__v0 as i64) >= 0) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }) & if 1i32.wrapping_neg() < 1 { 2i32.wrapping_add(((unsafe { __slate_intrinsic_1cfe0293d0a2af8d((__v1 as i64) >= 0) }) && (__v1 as i64) >= 0) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) } == 0 {
unsafe { __compiletime_assert_506() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = if __v0 < __v1 { __v0 } else { __v1 };
k = find_next_bit((unsafe { core::ptr::addr_of_mut!(key_down.0) as *mut u64 }) as *const u64, __v10 as u64, k as u64) as u32;
let mut __v12: i32;
let __v2: i32 = 256;
let __v3: i32 = 768i32;
loop {
if (if 1i32.wrapping_neg() < 1 { 2i32.wrapping_add(((unsafe { __slate_intrinsic_1cfe0293d0a2af8d((__v2 as i64) >= 0) }) && (__v2 as i64) >= 0) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }) & if 1i32.wrapping_neg() < 1 { 2i32.wrapping_add(((unsafe { __slate_intrinsic_1cfe0293d0a2af8d((__v3 as i64) >= 0) }) && (__v3 as i64) >= 0) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) } == 0 {
unsafe { __compiletime_assert_506() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v12 = if __v2 < __v3 { __v2 } else { __v3 };
if k >= (__v12 as u32) {
break;
}
sym = U!(unsafe { *unsafe { key_maps[0] }.wrapping_offset(k as isize) }) as u32;
if !(KTYP__2!(sym) != 7 && KTYP__2!(sym) != 12) {
val = KVAL__3!(sym);
if val == (KVAL!(K!(7i32, 8i32)) as u32) {
val = KVAL!(K!(7i32, 0i32)) as u32;
}
let __v5: *mut u8 = unsafe { core::ptr::addr_of_mut!(shift_down) as *mut u8 }.wrapping_offset(val as isize);
unsafe {
*__v5 = (*__v5 as i32).wrapping_add(1) as u8;
}
let __v8: i32 = unsafe { shift_state };
unsafe {
shift_state = ((__v8 as u64) | BIT!(val)) as i32;
}
}
k = k.wrapping_add(1);
}
}

/// We still have to export this method to vt.c
#[unsafe(no_mangle)]
extern "C" fn vt_set_leds_compute_shiftstate() {
// When VT is switched, the keyboard led needs to be set once.
// Ensure that after the switch is completed, the state of the
// keyboard LED is consistent with the state of the keyboard lock.
unsafe {
vt_switch = true;
}
set_leds();
let mut __UNIQUE_ID_guard_507: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(kbd_event_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_507) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_508: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(kbd_event_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_508) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
do_compute_shiftstate();
}

/// We have a combining character DIACR here, followed by the character CH.
/// If the combination occurs in the table, return the corresponding value.
/// Otherwise, if CH is a space or equals DIACR, return DIACR.
/// Otherwise, conclude that DIACR was not combining after all,
/// queue it and return CH.
fn handle_diacr(vc: *mut vc_data, ch: u32) -> u32 {
let d: u32 = unsafe { diacr };
let mut i: u32 = 0;
unsafe {
diacr = 0;
}
if d & 4294967040u32 == 10240 {
if ch & 4294967040u32 == 10240 {
return d | ch;
}
} else {
i = 0;
while i < unsafe { accent_table_size } {
if (unsafe { (*(core::ptr::addr_of_mut!(accent_table) as *mut kbdiacruc).wrapping_offset(i as isize)).diacr }) == d && (unsafe { (*(core::ptr::addr_of_mut!(accent_table) as *mut kbdiacruc).wrapping_offset(i as isize)).base }) == ch {
return unsafe { (*(core::ptr::addr_of_mut!(accent_table) as *mut kbdiacruc).wrapping_offset(i as isize)).result };
}
i = i.wrapping_add(1);
}
}
if ch == 32 || ch == 10240u32 || ch == d {
return d;
}
put_queue_utf8(vc, d);
ch
}

/// Special function handlers
#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_enter")]
extern "C" fn fn_enter(vc: *mut vc_data) {
if (unsafe { diacr }) != 0 {
put_queue_utf8(vc, unsafe { diacr });
unsafe {
diacr = 0;
}
}
put_queue(vc, 13i32);
if vc_kbd_mode(unsafe { kbd }, 3i32) != 0 {
put_queue(vc, 10i32);
}
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_caps_toggle")]
extern "C" fn fn_caps_toggle(vc: *mut vc_data) {
if unsafe { rep } {
return;
}
chg_vc_kbd_led(unsafe { kbd }, 2i32);
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_caps_on")]
extern "C" fn fn_caps_on(vc: *mut vc_data) {
if unsafe { rep } {
return;
}
set_vc_kbd_led(unsafe { kbd }, 2i32);
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_show_ptregs")]
extern "C" fn fn_show_ptregs(vc: *mut vc_data) {
let regs: *mut pt_regs = get_irq_regs();
if !regs.is_null() {
unsafe { show_regs(regs) };
}
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_hold")]
extern "C" fn fn_hold(vc: *mut vc_data) {
let tty: *mut tty_struct = unsafe { (*vc).port.tty };
if (unsafe { rep }) || tty.is_null() {
return;
}
// Note: SCROLLOCK will be set (cleared) by stop_tty (start_tty);
// these routines are also activated by ^S/^Q.
// (And SCROLLOCK can also be set by the ioctl KDSKBLED.)
if unsafe { (*tty).flow.stopped } {
unsafe { start_tty(tty) };
} else {
unsafe { stop_tty(tty) };
}
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_num")]
extern "C" fn fn_num(vc: *mut vc_data) {
if vc_kbd_mode(unsafe { kbd }, 0i32) != 0 {
applkey(vc, 80i32, 1u8);
} else {
fn_bare_num(vc);
}
}

/// Bind this to Shift-NumLock if you work in application keypad mode
/// but want to be able to change the NumLock flag.
/// Bind this to NumLock if you prefer that the NumLock key always
/// changes the NumLock flag.
#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_bare_num")]
extern "C" fn fn_bare_num(vc: *mut vc_data) {
if !unsafe { rep } {
chg_vc_kbd_led(unsafe { kbd }, 1i32);
}
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_lastcons")]
extern "C" fn fn_lastcons(vc: *mut vc_data) {
// switch to the last used console, ChN
unsafe { set_console(last_console) };
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_dec_console")]
extern "C" fn fn_dec_console(vc: *mut vc_data) {
let mut i: i32 = 0;
let mut cur: i32 = unsafe { fg_console };
// Currently switching?  Queue this next switch relative to that.
if (unsafe { want_console }) != 1i32.wrapping_neg() {
cur = unsafe { want_console };
}
i = cur.wrapping_sub(1);
while i != cur {
if i == 1i32.wrapping_neg() {
i = 62i32;
}
if (unsafe { vc_cons_allocated(i as u32) }) != 0 {
break;
}
i = i.wrapping_sub(1);
}
unsafe { set_console(i) };
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_inc_console")]
extern "C" fn fn_inc_console(vc: *mut vc_data) {
let mut i: i32 = 0;
let mut cur: i32 = unsafe { fg_console };
// Currently switching?  Queue this next switch relative to that.
if (unsafe { want_console }) != 1i32.wrapping_neg() {
cur = unsafe { want_console };
}
i = cur.wrapping_add(1);
while i != cur {
if i == 63 {
i = 0;
}
if (unsafe { vc_cons_allocated(i as u32) }) != 0 {
break;
}
i = i.wrapping_add(1);
}
unsafe { set_console(i) };
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_send_intr")]
extern "C" fn fn_send_intr(vc: *mut vc_data) {
tty_insert_flip_char(unsafe { core::ptr::addr_of_mut!((*vc).port) }, 0u8, 1u8);
unsafe { tty_flip_buffer_push(core::ptr::addr_of_mut!((*vc).port)) };
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_scroll_forw")]
extern "C" fn fn_scroll_forw(vc: *mut vc_data) {
unsafe { scrollfront(vc, 0i32) };
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_scroll_back")]
extern "C" fn fn_scroll_back(vc: *mut vc_data) {
unsafe { scrollback(vc) };
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_show_mem")]
extern "C" fn fn_show_mem(vc: *mut vc_data) {
show_mem();
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_show_state")]
extern "C" fn fn_show_state(vc: *mut vc_data) {
show_state();
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_boot_it")]
extern "C" fn fn_boot_it(vc: *mut vc_data) {
unsafe { ctrl_alt_del() };
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_compose")]
extern "C" fn fn_compose(vc: *mut vc_data) {
unsafe {
dead_key_next = true;
}
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_spawn_con")]
extern "C" fn fn_spawn_con(vc: *mut vc_data) {
let mut __UNIQUE_ID_guard_509: __SlateRecord_spinlock_h_610 = class_spinlock_constructor(unsafe { core::ptr::addr_of_mut!(vt_spawn_con.lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_509) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_510: *mut __SlateRecord_spinlock_h_610 = ((unsafe { core::ptr::addr_of_mut!(vt_spawn_con.lock) }) as u64) as *mut __SlateRecord_spinlock_h_610;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_510) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if !unsafe { vt_spawn_con.pid }.is_null() {
if (unsafe { kill_pid(vt_spawn_con.pid, vt_spawn_con.sig, 1i32) }) != 0 {
unsafe { put_pid(vt_spawn_con.pid) };
unsafe {
vt_spawn_con.pid = core::ptr::null_mut::<pid>();
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_SAK")]
extern "C" fn fn_SAK(vc: *mut vc_data) {
let SAK_work: *mut work_struct = unsafe { core::ptr::addr_of_mut!(vc_cons[fg_console as usize].SAK_work) };
schedule_work(SAK_work);
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.fn_null")]
extern "C" fn fn_null(vc: *mut vc_data) {
do_compute_shiftstate();
}

/// Special key handlers
#[unsafe(link_section = ".text.slate_distinct.keyboard.k_spec")]
extern "C" fn k_spec(vc: *mut vc_data, value: u8, up_flag: u8) {
if up_flag != 0 {
return;
}
if ((value as i32) as u64) >= ARRAY_SIZE__14!() {
return;
}
if (((unsafe { (*kbd).__slate_bits_1.__get_kbdmode() }) as i32) == 2 || ((unsafe { (*kbd).__slate_bits_1.__get_kbdmode() }) as i32) == 1 || ((unsafe { (*kbd).__slate_bits_1.__get_kbdmode() }) as i32) == 4) && (value as i32) != KVAL!(K!(2i32, 15i32)) {
return;
}
// SAK is allowed even in raw mode
unsafe { unsafe { fn_handler.0[(value as i32) as usize] }.unwrap()(vc) };
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.k_lowercase")]
extern "C" fn k_lowercase(vc: *mut vc_data, value: u8, up_flag: u8) {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013keyboard: k_lowercase was called - impossible\n".as_ptr() as *const u8) };
}

fn k_unicode(vc: *mut vc_data, mut value: u32, up_flag: u8) {
if up_flag != 0 {
return;
}
// no action, if this is a key release
if (unsafe { diacr }) != 0 {
value = handle_diacr(vc, value);
}
if unsafe { dead_key_next } {
unsafe {
dead_key_next = false;
}
unsafe {
diacr = value;
}
return;
}
put_queue_utf8(vc, value);
}

/// Handle dead key. Note that we now may have several
/// dead keys modifying the same character. Very useful
/// for Vietnamese.
fn k_deadunicode(vc: *mut vc_data, value: u32, up_flag: u8) {
if up_flag != 0 {
return;
}
let __v0: u32 = if (unsafe { diacr }) != 0 {
handle_diacr(vc, value)
} else {
value
};
unsafe {
diacr = __v0;
}
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.k_self")]
extern "C" fn k_self(vc: *mut vc_data, value: u8, up_flag: u8) {
k_unicode(vc, unsafe { conv_8bit_to_uni(value) }, up_flag);
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.k_dead2")]
extern "C" fn k_dead2(vc: *mut vc_data, value: u8, up_flag: u8) {
k_deadunicode(vc, value as u32, up_flag);
}

/// Obsolete - for backwards compatibility only
#[unsafe(link_section = ".text.slate_distinct.keyboard.k_dead")]
extern "C" fn k_dead(vc: *mut vc_data, value: u8, up_flag: u8) {
// dead_grave
// dead_acute
// dead_circumflex
// dead_tilda
// dead_diaeresis
// dead_cedilla
// dead_macron
// dead_breve
// dead_abovedot
// dead_abovering
// dead_doubleacute
// dead_caron
// dead_ogonek
// dead_iota
// dead_voiced_sound
// dead_semivoiced_sound
// dead_belowdot
// dead_hook
// dead_horn
// dead_stroke
// dead_abovecomma
// dead_abovereversedcomma
// dead_doublegrave
// dead_invertedbreve
// dead_belowcomma
// dead_currency
// dead_greek
k_deadunicode(vc, (unsafe { ret_diacr.0[(value as i32) as usize] }) as u32, up_flag);
}

static ret_diacr: __SlateAlign16<[u8; 27]> = __SlateAlign16([96u8, 39u8, 94u8, 126u8, 34u8, 44u8, 95u8, 85u8, 46u8, 42u8, 61u8, 99u8, 107u8, 105u8, 35u8, 111u8, 33u8, 63u8, 43u8, 45u8, 41u8, 40u8, 58u8, 110u8, 59u8, 36u8, 64u8]);

#[unsafe(link_section = ".text.slate_distinct.keyboard.k_cons")]
extern "C" fn k_cons(vc: *mut vc_data, value: u8, up_flag: u8) {
if up_flag != 0 {
return;
}
unsafe { set_console(value as i32) };
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.k_fn")]
extern "C" fn k_fn(vc: *mut vc_data, value: u8, up_flag: u8) {
if up_flag != 0 {
return;
}
if ((value as u32) as u64) < ARRAY_SIZE__32!() {
let mut __UNIQUE_ID_guard_511: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(func_buf_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_511) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_512: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(func_buf_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_512) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if !unsafe { func_table[(value as i32) as usize] }.is_null() {
puts_queue(vc, (unsafe { func_table[(value as i32) as usize] }) as *const u8);
}
} else {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x013keyboard: k_fn called with value=%d\n".as_ptr() as *const u8, value as i32) };
}
}

/// Compute xterm-style modifier parameter for CSI sequences.
/// Returns 1 + (shift ? 1 : 0) + (alt ? 2 : 0) + (ctrl ? 4 : 0)
///
/// Only the canonical modifier weights are counted. The left/right variants
/// (KG_SHIFTL, KG_SHIFTR, KG_CTRLL, KG_CTRLR) and KG_ALTGR are commonly
/// repurposed as keymap layout-group or level selectors rather than as plain
/// modifiers (for instance XKB-derived keymaps select the layout group with
/// KG_SHIFTL/KG_SHIFTR), so counting them would encode a spurious modifier.
fn csi_modifier_param() -> i32 {
let mut r#mod: i32 = 1;
if ((unsafe { shift_state }) as u64) & BIT!(0i32) != 0 {
r#mod = r#mod.wrapping_add(1);
}
if ((unsafe { shift_state }) as u64) & BIT!(3i32) != 0 {
r#mod = r#mod.wrapping_add(2);
}
if ((unsafe { shift_state }) as u64) & BIT!(2i32) != 0 {
r#mod = r#mod.wrapping_add(4);
}
r#mod
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.k_cur")]
extern "C" fn k_cur(vc: *mut vc_data, value: u8, up_flag: u8) {
let mut r#mod: i32 = 0;
if up_flag != 0 {
return;
}
r#mod = csi_modifier_param();
if r#mod > 1 {
let mut buf__0: [u8; 7] = [27, 91, 49, 59, 48i32.wrapping_add(r#mod) as u8, unsafe { cur_chars[(value as i32) as usize] }, 0];
puts_queue(vc, buf__0.as_mut_ptr() as *const u8);
} else {
applkey(vc, (unsafe { cur_chars[(value as i32) as usize] }) as i32, vc_kbd_mode(unsafe { kbd }, 1i32) as u8);
}
}

static cur_chars: [u8; 5] = [66, 68, 67, 65, 0];

#[unsafe(link_section = ".text.slate_distinct.keyboard.k_pad")]
extern "C" fn k_pad(vc: *mut vc_data, value: u8, up_flag: u8) {
if up_flag != 0 {
return;
}
// no action, if this is a key release
// kludge... shift forces cursor/number keys
if vc_kbd_mode(unsafe { kbd }, 0i32) != 0 && (unsafe { shift_down[0] }) == 0 {
applkey(vc, (unsafe { app_map.0[(value as i32) as usize] }) as i32, 1u8);
return;
}
if vc_kbd_led(unsafe { kbd }, 1i32) == 0 {
match value as i32 {
15 | 16 => {
k_fn(vc, KVAL!(K!(1i32, 22i32)) as u8, 0u8);
return;
}
0 => {
k_fn(vc, KVAL!(K!(1i32, 21i32)) as u8, 0u8);
return;
}
1 => {
k_fn(vc, KVAL!(K!(1i32, 23i32)) as u8, 0u8);
return;
}
2 => {
k_cur(vc, KVAL!(K!(6i32, 0i32)) as u8, 0u8);
return;
}
3 => {
k_fn(vc, KVAL!(K!(1i32, 25i32)) as u8, 0u8);
return;
}
4 => {
k_cur(vc, KVAL!(K!(6i32, 1i32)) as u8, 0u8);
return;
}
6 => {
k_cur(vc, KVAL!(K!(6i32, 2i32)) as u8, 0u8);
return;
}
7 => {
k_fn(vc, KVAL!(K!(1i32, 20i32)) as u8, 0u8);
return;
}
8 => {
k_cur(vc, KVAL!(K!(6i32, 3i32)) as u8, 0u8);
return;
}
9 => {
k_fn(vc, KVAL!(K!(1i32, 24i32)) as u8, 0u8);
return;
}
5 => {
applkey(vc, 71i32, vc_kbd_mode(unsafe { kbd }, 0i32) as u8);
return;
}
_ => {
}
}
}
put_queue(vc, (unsafe { pad_chars.0[(value as i32) as usize] }) as i32);
let __v1: bool = (value as i32) == KVAL!(K!(3i32, 14i32)) && vc_kbd_mode(unsafe { kbd }, 3i32) != 0;
if __v1 {
put_queue(vc, 10i32);
}
}

static pad_chars: __SlateAlign16<[u8; 22]> = __SlateAlign16([48u8, 49u8, 50u8, 51u8, 52u8, 53u8, 54u8, 55u8, 56u8, 57u8, 43u8, 45u8, 42u8, 47u8, 13u8, 44u8, 46u8, 63u8, 40u8, 41u8, 35u8, 0u8]);

static app_map: __SlateAlign16<[u8; 22]> = __SlateAlign16([112u8, 113u8, 114u8, 115u8, 116u8, 117u8, 118u8, 119u8, 120u8, 121u8, 108u8, 83u8, 82u8, 81u8, 77u8, 110u8, 110u8, 109u8, 80u8, 81u8, 83u8, 0u8]);

#[unsafe(link_section = ".text.slate_distinct.keyboard.k_shift")]
extern "C" fn k_shift(vc: *mut vc_data, mut value: u8, up_flag: u8) {
let old_state: i32 = unsafe { shift_state };
if unsafe { rep } {
return;
}
// Mimic typewriter:
// a CapsShift key acts like Shift but undoes CapsLock
if (value as i32) == KVAL!(K!(7i32, 8i32)) {
value = KVAL!(K!(7i32, 0i32)) as u8;
if up_flag == 0 {
clr_vc_kbd_led(unsafe { kbd }, 2i32);
}
}
if up_flag != 0 {
// handle the case that two shift or control
// keys are depressed simultaneously
if (unsafe { shift_down[(value as i32) as usize] }) != 0 {
let __v0: *mut u8 = unsafe { core::ptr::addr_of_mut!(shift_down) as *mut u8 }.wrapping_offset((value as i32) as isize);
unsafe {
*__v0 = (*__v0 as i32).wrapping_sub(1) as u8;
}
}
} else {
let __v3: *mut u8 = unsafe { core::ptr::addr_of_mut!(shift_down) as *mut u8 }.wrapping_offset((value as i32) as isize);
unsafe {
*__v3 = (*__v3 as i32).wrapping_add(1) as u8;
}
}
if (unsafe { shift_down[(value as i32) as usize] }) != 0 {
let __v6: i32 = unsafe { shift_state };
unsafe {
shift_state = ((__v6 as u64) | BIT__2!(value)) as i32;
}
} else {
let __v8: i32 = unsafe { shift_state };
unsafe {
shift_state = ((__v8 as u64) & !BIT__2!(value)) as i32;
}
}
// kludge
if up_flag != 0 && (unsafe { shift_state }) != old_state && unsafe { npadch_active } {
if ((unsafe { (*kbd).__slate_bits_1.__get_kbdmode() }) as i32) == 3 {
to_utf8(vc, unsafe { npadch_value });
} else {
put_queue(vc, ((unsafe { npadch_value }) & 255) as i32);
}
unsafe {
npadch_active = false;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.k_meta")]
extern "C" fn k_meta(vc: *mut vc_data, value: u8, up_flag: u8) {
if up_flag != 0 {
return;
}
if vc_kbd_mode(unsafe { kbd }, 4i32) != 0 {
put_queue(vc, 27i32);
put_queue(vc, value as i32);
} else {
put_queue(vc, (((value as i32) as u64) | BIT!(7i32)) as i32);
}
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.k_ascii")]
extern "C" fn k_ascii(vc: *mut vc_data, mut value: u8, up_flag: u8) {
let mut base: u32 = 0;
if up_flag != 0 {
return;
}
if (value as i32) < 10 {
// decimal input of code, while Alt depressed
base = 10;
} else {
// hexadecimal input of code, while AltGr depressed
value = (value as i32).wrapping_sub(10) as u8;
base = 16;
}
if !unsafe { npadch_active } {
unsafe {
npadch_value = 0;
}
unsafe {
npadch_active = true;
}
}
unsafe {
npadch_value = unsafe { npadch_value }.wrapping_mul(base).wrapping_add((value as i32) as u32);
}
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.k_lock")]
extern "C" fn k_lock(vc: *mut vc_data, value: u8, up_flag: u8) {
if up_flag != 0 || unsafe { rep } {
return;
}
chg_vc_kbd_lock(unsafe { kbd }, value as i32);
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.k_slock")]
extern "C" fn k_slock(vc: *mut vc_data, value: u8, up_flag: u8) {
k_shift(vc, value, up_flag);
if up_flag != 0 || unsafe { rep } {
return;
}
chg_vc_kbd_slock(unsafe { kbd }, value as i32);
// try to make Alt, oops, AltGr and such work
if unsafe { key_maps[(((*kbd).lockstate as i32) ^ ((*kbd).slockstate as i32)) as usize] }.is_null() {
unsafe {
(*kbd).slockstate = 0;
}
chg_vc_kbd_slock(unsafe { kbd }, value as i32);
}
}

/// by default, 300ms interval for combination release
static mut brl_timeout: u32 = 300;

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_513: [u8; 95] = [107, 101, 121, 98, 111, 97, 114, 100, 46, 112, 97, 114, 109, 61, 98, 114, 108, 95, 116, 105, 109, 101, 111, 117, 116, 58, 66, 114, 97, 105, 108, 108, 101, 32, 107, 101, 121, 115, 32, 114, 101, 108, 101, 97, 115, 101, 32, 100, 101, 108, 97, 121, 32, 105, 110, 32, 109, 115, 32, 40, 48, 32, 102, 111, 114, 32, 99, 111, 109, 109, 105, 116, 32, 111, 110, 32, 102, 105, 114, 115, 116, 32, 107, 101, 121, 32, 114, 101, 108, 101, 97, 115, 101, 41, 0];

fn __check_brl_timeout() -> *mut u32 {
unsafe { core::ptr::addr_of_mut!(brl_timeout) }
}

static __param_str_brl_timeout: __SlateAlign16<[u8; 21]> = __SlateAlign16([107u8, 101u8, 121u8, 98u8, 111u8, 97u8, 114u8, 100u8, 46u8, 98u8, 114u8, 108u8, 95u8, 116u8, 105u8, 109u8, 101u8, 111u8, 117u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_brl_timeout: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_brl_timeout.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_uint) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_keyboard_c_967>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_keyboard_c_967_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_keyboard_c_967_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_keyboard_c_967_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_keyboard_c_967_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_keyboard_c_967_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(brl_timeout) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_514: [u8; 35] = [107, 101, 121, 98, 111, 97, 114, 100, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 98, 114, 108, 95, 116, 105, 109, 101, 111, 117, 116, 58, 117, 105, 110, 116, 0];

static mut brl_nbchords: u32 = 1;

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_515: [u8; 95] = [107, 101, 121, 98, 111, 97, 114, 100, 46, 112, 97, 114, 109, 61, 98, 114, 108, 95, 110, 98, 99, 104, 111, 114, 100, 115, 58, 78, 117, 109, 98, 101, 114, 32, 111, 102, 32, 99, 104, 111, 114, 100, 115, 32, 116, 104, 97, 116, 32, 112, 114, 111, 100, 117, 99, 101, 32, 97, 32, 98, 114, 97, 105, 108, 108, 101, 32, 112, 97, 116, 116, 101, 114, 110, 32, 40, 48, 32, 102, 111, 114, 32, 100, 101, 97, 100, 32, 99, 104, 111, 114, 100, 115, 41, 0];

fn __check_brl_nbchords() -> *mut u32 {
unsafe { core::ptr::addr_of_mut!(brl_nbchords) }
}

static __param_str_brl_nbchords: __SlateAlign16<[u8; 22]> = __SlateAlign16([107u8, 101u8, 121u8, 98u8, 111u8, 97u8, 114u8, 100u8, 46u8, 98u8, 114u8, 108u8, 95u8, 110u8, 98u8, 99u8, 104u8, 111u8, 114u8, 100u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_brl_nbchords: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_brl_nbchords.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_uint) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_keyboard_c_971>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_keyboard_c_971_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_keyboard_c_971_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_keyboard_c_971_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_keyboard_c_971_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_keyboard_c_971_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(brl_nbchords) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_516: [u8; 36] = [107, 101, 121, 98, 111, 97, 114, 100, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 98, 114, 108, 95, 110, 98, 99, 104, 111, 114, 100, 115, 58, 117, 105, 110, 116, 0];

fn k_brlcommit(vc: *mut vc_data, pattern: u32, up_flag: u8) {
if (unsafe { brl_nbchords }) == 0 {
k_deadunicode(vc, 10240 | pattern, up_flag);
} else {
unsafe {
committed = committed | pattern;
}
let __v2: u64 = unsafe { chords };
unsafe {
chords = __v2.wrapping_add(1);
}
if (unsafe { chords }) == ((unsafe { brl_nbchords }) as u64) {
k_unicode(vc, 10240 | unsafe { committed }, up_flag);
unsafe {
chords = 0;
}
unsafe {
committed = 0;
}
}
}
}

static mut chords: u64 = unsafe { core::mem::zeroed() };

static mut committed: u32 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.keyboard.k_brl")]
extern "C" fn k_brl(vc: *mut vc_data, value: u8, up_flag: u8) {
if ((unsafe { (*kbd).__slate_bits_1.__get_kbdmode() }) as i32) != 3 {
if up_flag == 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014keyboard: keyboard mode must be unicode for braille patterns\n".as_ptr() as *const u8) };
}
return;
}
if value == 0 {
k_unicode(vc, 10240u32, up_flag);
return;
}
if (value as i32) > 8 {
return;
}
if up_flag == 0 {
let __v2: u32 = unsafe { pressed };
unsafe {
pressed = ((__v2 as u64) | BIT!((value as i32).wrapping_sub(1))) as u32;
}
if (unsafe { brl_timeout }) == 0 {
unsafe {
committing = pressed;
}
}
} else {
if (unsafe { brl_timeout }) != 0 {
let __v4: bool = (unsafe { committing }) == 0 || time_after__3!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, unsafe { releasestart }.wrapping_add(msecs_to_jiffies(unsafe { brl_timeout })));
if __v4 {
unsafe {
committing = pressed;
}
unsafe {
releasestart = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
}
}
let __v13: u32 = unsafe { pressed };
unsafe {
pressed = ((__v13 as u64) & !BIT!((value as i32).wrapping_sub(1))) as u32;
}
if (unsafe { pressed }) == 0 && (unsafe { committing }) != 0 {
k_brlcommit(vc, unsafe { committing }, 0u8);
unsafe {
committing = 0;
}
}
} else {
if (unsafe { committing }) != 0 {
k_brlcommit(vc, unsafe { committing }, 0u8);
unsafe {
committing = 0;
}
}
let __v15: u32 = unsafe { pressed };
unsafe {
pressed = ((__v15 as u64) & !BIT!((value as i32).wrapping_sub(1))) as u32;
}
}
}
}

static mut pressed: u32 = unsafe { core::mem::zeroed() };

static mut committing: u32 = unsafe { core::mem::zeroed() };

static mut releasestart: u64 = unsafe { core::mem::zeroed() };

/// Handle KT_CSI keysym type: generate CSI tilde sequences with modifier
/// support. The value encodes the CSI parameter number, producing sequences
/// like ESC [ <value> ~ or ESC [ <value> ; <mod> ~ when modifiers are held.
#[unsafe(link_section = ".text.slate_distinct.keyboard.k_csi")]
extern "C" fn k_csi(vc: *mut vc_data, value: u8, up_flag: u8) {
let mut buf__0: [u8; 10] = [0u8; 10];
let mut i: i32 = 0;
let mut r#mod: i32 = 0;
if up_flag != 0 {
return;
}
r#mod = csi_modifier_param();
let __v0: i32 = i;
i = __v0.wrapping_add(1);
buf__0[__v0 as usize] = 27;
let __v2: i32 = i;
i = __v2.wrapping_add(1);
buf__0[__v2 as usize] = 91;
if (value as i32) >= 10 {
let __v4: i32 = i;
i = __v4.wrapping_add(1);
buf__0[__v4 as usize] = 48i32.wrapping_add((value as i32) / 10) as u8;
}
let __v6: i32 = i;
i = __v6.wrapping_add(1);
buf__0[__v6 as usize] = 48i32.wrapping_add((value as i32) % 10) as u8;
if r#mod > 1 {
let __v8: i32 = i;
i = __v8.wrapping_add(1);
buf__0[__v8 as usize] = 59;
let __v10: i32 = i;
i = __v10.wrapping_add(1);
buf__0[__v10 as usize] = 48i32.wrapping_add(r#mod) as u8;
}
let __v12: i32 = i;
i = __v12.wrapping_add(1);
buf__0[__v12 as usize] = 126;
buf__0[i as usize] = 0;
puts_queue(vc, buf__0.as_mut_ptr() as *const u8);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kbd_led_trigger {
    trigger: led_trigger,
    mask: u32,
}


#[unsafe(link_section = ".text.slate_distinct.keyboard.kbd_led_trigger_activate")]
extern "C" fn kbd_led_trigger_activate(cdev: *mut led_classdev) -> i32 {
let trigger: *mut kbd_led_trigger = container_of__11!(unsafe { (*cdev).trigger }, kbd_led_trigger);
tasklet_disable(unsafe { core::ptr::addr_of_mut!(keyboard_tasklet) });
if (unsafe { ledstate }) != 1u32.wrapping_neg() {
unsafe { led_set_brightness(cdev, (if ledstate & (*trigger).mask != 0 { 255i32 } else { 0i32 }) as u32) };
}
tasklet_enable(unsafe { core::ptr::addr_of_mut!(keyboard_tasklet) });
0
}

static mut kbd_led_triggers: __SlateAlign16<[kbd_led_trigger; 12]> = __SlateAlign16([{
let mut __t0: kbd_led_trigger = unsafe { core::mem::zeroed() };
__t0.trigger = {
let mut __t1: led_trigger = unsafe { core::mem::zeroed() };
__t1.name = (b"kbd-scrolllock\0".as_ptr() as *mut u8) as *const u8;
__t1.activate = Some(kbd_led_trigger_activate);
__t1
};
__t0.mask = 1u32;
__t0
}, {
let mut __t2: kbd_led_trigger = unsafe { core::mem::zeroed() };
__t2.trigger = {
let mut __t3: led_trigger = unsafe { core::mem::zeroed() };
__t3.name = (b"kbd-numlock\0".as_ptr() as *mut u8) as *const u8;
__t3.activate = Some(kbd_led_trigger_activate);
__t3
};
__t2.mask = 2u32;
__t2
}, {
let mut __t4: kbd_led_trigger = unsafe { core::mem::zeroed() };
__t4.trigger = {
let mut __t5: led_trigger = unsafe { core::mem::zeroed() };
__t5.name = (b"kbd-capslock\0".as_ptr() as *mut u8) as *const u8;
__t5.activate = Some(kbd_led_trigger_activate);
__t5
};
__t4.mask = 4u32;
__t4
}, {
let mut __t6: kbd_led_trigger = unsafe { core::mem::zeroed() };
__t6.trigger = {
let mut __t7: led_trigger = unsafe { core::mem::zeroed() };
__t7.name = (b"kbd-kanalock\0".as_ptr() as *mut u8) as *const u8;
__t7.activate = Some(kbd_led_trigger_activate);
__t7
};
__t6.mask = 8u32;
__t6
}, {
let mut __t8: kbd_led_trigger = unsafe { core::mem::zeroed() };
__t8.trigger = {
let mut __t9: led_trigger = unsafe { core::mem::zeroed() };
__t9.name = (b"kbd-shiftlock\0".as_ptr() as *mut u8) as *const u8;
__t9.activate = Some(kbd_led_trigger_activate);
__t9
};
__t8.mask = 256u32;
__t8
}, {
let mut __t10: kbd_led_trigger = unsafe { core::mem::zeroed() };
__t10.trigger = {
let mut __t11: led_trigger = unsafe { core::mem::zeroed() };
__t11.name = (b"kbd-altgrlock\0".as_ptr() as *mut u8) as *const u8;
__t11.activate = Some(kbd_led_trigger_activate);
__t11
};
__t10.mask = 512u32;
__t10
}, {
let mut __t12: kbd_led_trigger = unsafe { core::mem::zeroed() };
__t12.trigger = {
let mut __t13: led_trigger = unsafe { core::mem::zeroed() };
__t13.name = (b"kbd-ctrllock\0".as_ptr() as *mut u8) as *const u8;
__t13.activate = Some(kbd_led_trigger_activate);
__t13
};
__t12.mask = 1024u32;
__t12
}, {
let mut __t14: kbd_led_trigger = unsafe { core::mem::zeroed() };
__t14.trigger = {
let mut __t15: led_trigger = unsafe { core::mem::zeroed() };
__t15.name = (b"kbd-altlock\0".as_ptr() as *mut u8) as *const u8;
__t15.activate = Some(kbd_led_trigger_activate);
__t15
};
__t14.mask = 2048u32;
__t14
}, {
let mut __t16: kbd_led_trigger = unsafe { core::mem::zeroed() };
__t16.trigger = {
let mut __t17: led_trigger = unsafe { core::mem::zeroed() };
__t17.name = (b"kbd-shiftllock\0".as_ptr() as *mut u8) as *const u8;
__t17.activate = Some(kbd_led_trigger_activate);
__t17
};
__t16.mask = 4096u32;
__t16
}, {
let mut __t18: kbd_led_trigger = unsafe { core::mem::zeroed() };
__t18.trigger = {
let mut __t19: led_trigger = unsafe { core::mem::zeroed() };
__t19.name = (b"kbd-shiftrlock\0".as_ptr() as *mut u8) as *const u8;
__t19.activate = Some(kbd_led_trigger_activate);
__t19
};
__t18.mask = 8192u32;
__t18
}, {
let mut __t20: kbd_led_trigger = unsafe { core::mem::zeroed() };
__t20.trigger = {
let mut __t21: led_trigger = unsafe { core::mem::zeroed() };
__t21.name = (b"kbd-ctrlllock\0".as_ptr() as *mut u8) as *const u8;
__t21.activate = Some(kbd_led_trigger_activate);
__t21
};
__t20.mask = 16384u32;
__t20
}, {
let mut __t22: kbd_led_trigger = unsafe { core::mem::zeroed() };
__t22.trigger = {
let mut __t23: led_trigger = unsafe { core::mem::zeroed() };
__t23.name = (b"kbd-ctrlrlock\0".as_ptr() as *mut u8) as *const u8;
__t23.activate = Some(kbd_led_trigger_activate);
__t23
};
__t22.mask = 32768u32;
__t22
}]);

fn kbd_propagate_led_state(old_state: u32, new_state: u32) {
let mut trigger: *mut kbd_led_trigger = unsafe { core::mem::zeroed() };
let changed: u32 = old_state ^ new_state;
let mut i: i32 = 0;
while (i as u64) < ARRAY_SIZE__15!() {
trigger = unsafe { core::ptr::addr_of_mut!(kbd_led_triggers.0) as *mut kbd_led_trigger }.wrapping_offset(i as isize);
if changed & unsafe { (*trigger).mask } != 0 {
unsafe { led_trigger_event(core::ptr::addr_of_mut!((*trigger).trigger), (if new_state & (*trigger).mask != 0 { 255i32 } else { 0i32 }) as u32) };
}
i = i.wrapping_add(1);
}
}

fn kbd_update_leds_helper(handle: *mut input_handle, data: *mut ()) -> i32 {
let led_state: u32 = unsafe { *(data as *mut u32) };
if test_bit!(17i32, unsafe { (*(*handle).dev).evbit.as_mut_ptr() as *mut u64 }) != 0 {
kbd_propagate_led_state(!led_state, led_state);
}
0
}

fn kbd_init_leds() {
let mut error: i32 = 0;
let mut i: i32 = 0;
while (i as u64) < ARRAY_SIZE__19!() {
error = unsafe { led_trigger_register(core::ptr::addr_of_mut!(kbd_led_triggers.0[i as usize].trigger)) };
if error != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013keyboard: error %d while registering trigger %s\n".as_ptr() as *const u8, error, kbd_led_triggers.0[i as usize].trigger.name) };
}
i = i.wrapping_add(1);
}
}

/// The leds display either (i) the status of NumLock, CapsLock, ScrollLock,
/// or (ii) whatever pattern of lights people want to show using KDSETLED,
/// or (iii) specified bits of specified words in kernel memory.
fn getledstate() -> u8 {
((unsafe { ledstate }) & 255) as u8
}

#[unsafe(no_mangle)]
extern "C" fn setledstate(kb: *mut kbd_struct, led: u32) {
let mut __UNIQUE_ID_guard_517: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(led_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_517) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_518: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(led_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_518) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if led & 4294967288u32 == 0 {
unsafe {
ledioctl = led as u8;
}
unsafe {
(*kb).__slate_bits_0.__set_ledmode(1u8);
}
} else {
unsafe {
(*kb).__slate_bits_0.__set_ledmode(0u8);
}
}
set_leds();
}

fn getleds() -> u8 {
let kb: *mut kbd_struct = unsafe { core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct }.wrapping_offset((unsafe { fg_console }) as isize);
if ((unsafe { (*kb).__slate_bits_0.__get_ledmode() }) as i32) == 1 {
return unsafe { ledioctl };
}
((unsafe { (*kb).__slate_bits_0.__get_ledflagstate() }) as i32) as u8
}

/// vt_get_leds	-	helper for braille console
/// @console: console to read
/// @flag: flag we want to check
///
/// Check the status of a keyboard led flag and report it back
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.keyboard.vt_get_leds")]
extern "C" fn vt_get_leds(console: u32, flag: i32) -> i32 {
let kb: *mut kbd_struct = unsafe { core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct }.wrapping_offset(console as isize);
let mut __UNIQUE_ID_guard_519: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(led_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_519) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_520: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(led_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_520) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
vc_kbd_led(kb, flag)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_vt_get_leds_521: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, i32) -> i32>, *mut ()>(Some(vt_get_leds)) };

/// vt_set_led_state	-	set LED state of a console
/// @console: console to set
/// @leds: LED bits
///
/// Set the LEDs on a console. This is a wrapper for the VT layer
/// so that we can keep kbd knowledge internal
#[unsafe(no_mangle)]
extern "C" fn vt_set_led_state(console: u32, leds: i32) {
let kb: *mut kbd_struct = unsafe { core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct }.wrapping_offset(console as isize);
setledstate(kb, leds as u32);
}

/// vt_kbd_con_start	-	Keyboard side of console start
/// @console: console
///
/// Handle console start. This is a wrapper for the VT layer
/// so that we can keep kbd knowledge internal
///
/// FIXME: We eventually need to hold the kbd lock here to protect
/// the LED updating. We can't do it yet because fn_hold calls stop_tty
/// and start_tty under the kbd_event_lock, while normal tty paths
/// don't hold the lock. We probably need to split out an LED lock
/// but not during an -rc release!
#[unsafe(no_mangle)]
extern "C" fn vt_kbd_con_start(console: u32) {
let kb: *mut kbd_struct = unsafe { core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct }.wrapping_offset(console as isize);
let mut __UNIQUE_ID_guard_522: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(led_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_522) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_523: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(led_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_523) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
clr_vc_kbd_led(kb, 0i32);
set_leds();
}

/// vt_kbd_con_stop		-	Keyboard side of console stop
/// @console: console
///
/// Handle console stop. This is a wrapper for the VT layer
/// so that we can keep kbd knowledge internal
#[unsafe(no_mangle)]
extern "C" fn vt_kbd_con_stop(console: u32) {
let kb: *mut kbd_struct = unsafe { core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct }.wrapping_offset(console as isize);
let mut __UNIQUE_ID_guard_524: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(led_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_524) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_525: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(led_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_525) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
set_vc_kbd_led(kb, 0i32);
set_leds();
}

/// This is the tasklet that updates LED state of LEDs using standard
/// keyboard triggers. The reason we use tasklet is that we need to
/// handle the scenario when keyboard handler is not registered yet
/// but we already getting updates from the VT to update led state.
#[unsafe(link_section = ".text.slate_distinct.keyboard.kbd_bh")]
extern "C" fn kbd_bh(unused: *mut tasklet_struct) {
let mut __UNIQUE_ID_unlock_527: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_527) as *mut *mut __SlateRecord_spinlock_h_642;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_642;
let mut leds: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(leds) as *mut u32;
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { class_spinlock_irqsave_destructor(__slate_cleanup_ptr_3 as *mut _) };
}
});
let __slate_cleanup_ptr_2 = __slate_slot_2;
let __slate_cleanup_armed_2 = core::cell::Cell::new(false);
let __slate_cleanup_2 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_2.get() {
unsafe { __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
core::ptr::write(__slate_slot_3, class_spinlock_irqsave_constructor(core::ptr::addr_of_mut!(led_lock)));
__slate_cleanup_armed_3.set(true);
core::ptr::write(__slate_slot_2, (core::ptr::addr_of_mut!(led_lock) as u64) as *mut __SlateRecord_spinlock_h_642);
__slate_cleanup_armed_2.set(true);
if !class_spinlock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_3))).is_null() || !class_spinlock_irqsave_is_conditional {
if true {
*__slate_slot_4 = getleds() as u32;
*__slate_slot_4 |= ((*kbd).lockstate as u32) << 8i32;
}
}
__slate_cleanup_armed_2.set(false);
__class_spinlock_irqsave_cleanup_ctx(__slate_slot_2 as *mut _);
__slate_cleanup_armed_3.set(false);
class_spinlock_irqsave_destructor(__slate_slot_3 as *mut _);
if vt_switch {
ledstate = !(*__slate_slot_4);
vt_switch = false;
}
if *__slate_slot_4 != ledstate {
kbd_propagate_led_state(ledstate, *__slate_slot_4);
ledstate = *__slate_slot_4;
}
}
}

fn kbd_is_hw_raw(dev: *const input_dev) -> bool {
let __v0: bool = test_bit__2!(4i32, unsafe { (*dev).evbit.as_ptr() as *const u64 }) == 0 || test_bit__2!(3i32, unsafe { (*dev).mscbit.as_ptr() as *const u64 }) == 0;
if __v0 {
return false;
}
((unsafe { (*dev).id.bustype }) as i32) == 17 && ((unsafe { (*dev).id.vendor }) as i32) == 1 && ((unsafe { (*dev).id.product }) as i32) == 1
}

static x86_keycodes: __SlateAlign16<[u16; 256]> = __SlateAlign16({
let mut __t0: [u16; 256] = unsafe { core::mem::zeroed() };
__t0[0] = 0u16;
__t0[1] = 1u16;
__t0[2] = 2u16;
__t0[3] = 3u16;
__t0[4] = 4u16;
__t0[5] = 5u16;
__t0[6] = 6u16;
__t0[7] = 7u16;
__t0[8] = 8u16;
__t0[9] = 9u16;
__t0[10] = 10u16;
__t0[11] = 11u16;
__t0[12] = 12u16;
__t0[13] = 13u16;
__t0[14] = 14u16;
__t0[15] = 15u16;
__t0[16] = 16u16;
__t0[17] = 17u16;
__t0[18] = 18u16;
__t0[19] = 19u16;
__t0[20] = 20u16;
__t0[21] = 21u16;
__t0[22] = 22u16;
__t0[23] = 23u16;
__t0[24] = 24u16;
__t0[25] = 25u16;
__t0[26] = 26u16;
__t0[27] = 27u16;
__t0[28] = 28u16;
__t0[29] = 29u16;
__t0[30] = 30u16;
__t0[31] = 31u16;
__t0[32] = 32u16;
__t0[33] = 33u16;
__t0[34] = 34u16;
__t0[35] = 35u16;
__t0[36] = 36u16;
__t0[37] = 37u16;
__t0[38] = 38u16;
__t0[39] = 39u16;
__t0[40] = 40u16;
__t0[41] = 41u16;
__t0[42] = 42u16;
__t0[43] = 43u16;
__t0[44] = 44u16;
__t0[45] = 45u16;
__t0[46] = 46u16;
__t0[47] = 47u16;
__t0[48] = 48u16;
__t0[49] = 49u16;
__t0[50] = 50u16;
__t0[51] = 51u16;
__t0[52] = 52u16;
__t0[53] = 53u16;
__t0[54] = 54u16;
__t0[55] = 55u16;
__t0[56] = 56u16;
__t0[57] = 57u16;
__t0[58] = 58u16;
__t0[59] = 59u16;
__t0[60] = 60u16;
__t0[61] = 61u16;
__t0[62] = 62u16;
__t0[63] = 63u16;
__t0[64] = 64u16;
__t0[65] = 65u16;
__t0[66] = 66u16;
__t0[67] = 67u16;
__t0[68] = 68u16;
__t0[69] = 69u16;
__t0[70] = 70u16;
__t0[71] = 71u16;
__t0[72] = 72u16;
__t0[73] = 73u16;
__t0[74] = 74u16;
__t0[75] = 75u16;
__t0[76] = 76u16;
__t0[77] = 77u16;
__t0[78] = 78u16;
__t0[79] = 79u16;
__t0[80] = 80u16;
__t0[81] = 81u16;
__t0[82] = 82u16;
__t0[83] = 83u16;
__t0[84] = 84u16;
__t0[85] = 118u16;
__t0[86] = 86u16;
__t0[87] = 87u16;
__t0[88] = 88u16;
__t0[89] = 115u16;
__t0[90] = 120u16;
__t0[91] = 119u16;
__t0[92] = 121u16;
__t0[93] = 112u16;
__t0[94] = 123u16;
__t0[95] = 92u16;
__t0[96] = 284u16;
__t0[97] = 285u16;
__t0[98] = 309u16;
__t0[99] = 0u16;
__t0[100] = 312u16;
__t0[101] = 91u16;
__t0[102] = 327u16;
__t0[103] = 328u16;
__t0[104] = 329u16;
__t0[105] = 331u16;
__t0[106] = 333u16;
__t0[107] = 335u16;
__t0[108] = 336u16;
__t0[109] = 337u16;
__t0[110] = 338u16;
__t0[111] = 339u16;
__t0[112] = 367u16;
__t0[113] = 288u16;
__t0[114] = 302u16;
__t0[115] = 304u16;
__t0[116] = 350u16;
__t0[117] = 89u16;
__t0[118] = 334u16;
__t0[119] = 326u16;
__t0[120] = 267u16;
__t0[121] = 126u16;
__t0[122] = 268u16;
__t0[123] = 269u16;
__t0[124] = 125u16;
__t0[125] = 347u16;
__t0[126] = 348u16;
__t0[127] = 349u16;
__t0[128] = 360u16;
__t0[129] = 261u16;
__t0[130] = 262u16;
__t0[131] = 263u16;
__t0[132] = 268u16;
__t0[133] = 376u16;
__t0[134] = 100u16;
__t0[135] = 101u16;
__t0[136] = 321u16;
__t0[137] = 316u16;
__t0[138] = 373u16;
__t0[139] = 286u16;
__t0[140] = 289u16;
__t0[141] = 102u16;
__t0[142] = 351u16;
__t0[143] = 355u16;
__t0[144] = 103u16;
__t0[145] = 104u16;
__t0[146] = 105u16;
__t0[147] = 275u16;
__t0[148] = 287u16;
__t0[149] = 279u16;
__t0[150] = 258u16;
__t0[151] = 106u16;
__t0[152] = 274u16;
__t0[153] = 107u16;
__t0[154] = 294u16;
__t0[155] = 364u16;
__t0[156] = 358u16;
__t0[157] = 363u16;
__t0[158] = 362u16;
__t0[159] = 361u16;
__t0[160] = 291u16;
__t0[161] = 108u16;
__t0[162] = 381u16;
__t0[163] = 281u16;
__t0[164] = 290u16;
__t0[165] = 272u16;
__t0[166] = 292u16;
__t0[167] = 305u16;
__t0[168] = 280u16;
__t0[169] = 99u16;
__t0[170] = 112u16;
__t0[171] = 257u16;
__t0[172] = 306u16;
__t0[173] = 359u16;
__t0[174] = 113u16;
__t0[175] = 114u16;
__t0[176] = 264u16;
__t0[177] = 117u16;
__t0[178] = 271u16;
__t0[179] = 374u16;
__t0[180] = 379u16;
__t0[181] = 265u16;
__t0[182] = 266u16;
__t0[183] = 93u16;
__t0[184] = 94u16;
__t0[185] = 95u16;
__t0[186] = 85u16;
__t0[187] = 259u16;
__t0[188] = 375u16;
__t0[189] = 260u16;
__t0[190] = 90u16;
__t0[191] = 116u16;
__t0[192] = 377u16;
__t0[193] = 109u16;
__t0[194] = 111u16;
__t0[195] = 277u16;
__t0[196] = 278u16;
__t0[197] = 282u16;
__t0[198] = 283u16;
__t0[199] = 295u16;
__t0[200] = 296u16;
__t0[201] = 297u16;
__t0[202] = 299u16;
__t0[203] = 300u16;
__t0[204] = 301u16;
__t0[205] = 293u16;
__t0[206] = 303u16;
__t0[207] = 307u16;
__t0[208] = 308u16;
__t0[209] = 310u16;
__t0[210] = 313u16;
__t0[211] = 314u16;
__t0[212] = 315u16;
__t0[213] = 317u16;
__t0[214] = 318u16;
__t0[215] = 319u16;
__t0[216] = 320u16;
__t0[217] = 357u16;
__t0[218] = 322u16;
__t0[219] = 323u16;
__t0[220] = 324u16;
__t0[221] = 325u16;
__t0[222] = 276u16;
__t0[223] = 330u16;
__t0[224] = 332u16;
__t0[225] = 340u16;
__t0[226] = 365u16;
__t0[227] = 342u16;
__t0[228] = 343u16;
__t0[229] = 344u16;
__t0[230] = 345u16;
__t0[231] = 346u16;
__t0[232] = 356u16;
__t0[233] = 270u16;
__t0[234] = 341u16;
__t0[235] = 368u16;
__t0[236] = 369u16;
__t0[237] = 370u16;
__t0[238] = 371u16;
__t0[239] = 372u16;
__t0
});

fn emulate_raw(vc: *mut vc_data, keycode: u32, up_flag: u8) -> i32 {
let mut code: i32 = 0;
match keycode {
119 => {
put_queue(vc, 225i32);
put_queue(vc, 29 | (up_flag as i32));
put_queue(vc, 69 | (up_flag as i32));
}
122 => {
if up_flag == 0 {
put_queue(vc, 242i32);
}
}
123 => {
if up_flag == 0 {
put_queue(vc, 241i32);
}
}
99 => {
let __v1: bool = test_bit!(56i32, unsafe { core::ptr::addr_of_mut!(key_down.0) as *mut u64 }) != 0 || test_bit!(100i32, unsafe { core::ptr::addr_of_mut!(key_down.0) as *mut u64 }) != 0;
if __v1 {
put_queue(vc, 84 | (up_flag as i32));
} else {
put_queue(vc, 224i32);
put_queue(vc, 42 | (up_flag as i32));
put_queue(vc, 224i32);
put_queue(vc, 55 | (up_flag as i32));
}
// Real AT keyboards (that's what we're trying
// to emulate here) emit 0xe0 0x2a 0xe0 0x37 when
// pressing PrtSc/SysRq alone, but simply 0x54
// when pressing Alt+PrtSc/SysRq.
}
_ => {
if keycode > 255 {
return 1i32.wrapping_neg();
}
code = (unsafe { x86_keycodes.0[keycode as usize] }) as i32;
if code == 0 {
return 1i32.wrapping_neg();
}
if code & 256 != 0 {
put_queue(vc, 224i32);
}
put_queue(vc, code & 127 | (up_flag as i32));
}
}
0
}

fn kbd_rawcode(data: u8) {
let vc: *mut vc_data = unsafe { vc_cons[fg_console as usize].d };
unsafe {
kbd = (core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct).wrapping_offset(((*vc).vc_num as i32) as isize);
}
if ((unsafe { (*kbd).__slate_bits_1.__get_kbdmode() }) as i32) == 2 {
put_queue(vc, data as i32);
}
}

fn kbd_keycode(keycode: u32, down: i32, hw_raw: bool) {
let vc: *mut vc_data = unsafe { vc_cons[fg_console as usize].d };
let mut keysym: u16 = 0;
let mut key_map: *mut u16 = unsafe { core::mem::zeroed() };
let mut r#type: u8 = 0;
let mut raw_mode: bool = false;
let mut shift_final: i32 = 0;
let mut param: keyboard_notifier_param = {
let mut __t0: keyboard_notifier_param = unsafe { core::mem::zeroed() };
__t0.vc = vc;
__t0.down = down;
__t0.value = keycode;
__t0
};
let mut rc: i32 = 0;
let tty: *mut tty_struct = unsafe { tty_port_tty_get(core::ptr::addr_of_mut!((*vc).port)) };
if !tty.is_null() && unsafe { (*tty).driver_data }.is_null() {
// No driver data? Strange. Okay we fix it then.
unsafe {
(*tty).driver_data = vc as *mut ();
}
}
unsafe {
kbd = (core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct).wrapping_offset(((*vc).vc_num as i32) as isize);
}
unsafe {
rep = down == 2;
}
raw_mode = ((unsafe { (*kbd).__slate_bits_1.__get_kbdmode() }) as i32) == 2;
if raw_mode && !hw_raw {
if emulate_raw(vc, keycode, (((down == 0) as i32) << 7i32) as u8) != 0 {
let __v0: bool = keycode < 256 && printk_ratelimit!() != 0;
if __v0 {
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x014keyboard: can't emulate rawmode for keycode %d\n".as_ptr() as *const u8, keycode) };
}
}
}
if ((unsafe { (*kbd).__slate_bits_1.__get_kbdmode() }) as i32) == 1 {
// This is extended medium raw mode, with keys above 127
// encoded as 0, high 7 bits, low 7 bits, with the 0 bearing
// the 'up' flag if needed. 0 is reserved, so this shouldn't
// interfere with anything else. The two bytes after 0 will
// always have the up flag set not to interfere with older
// applications. This allows for 16384 different keycodes,
// which should be enough.
if keycode < 128 {
put_queue(vc, (keycode | ((((down == 0) as i32) << 7i32) as u32)) as i32);
} else {
put_queue(vc, ((down == 0) as i32) << 7i32);
put_queue(vc, (((keycode >> 7i32) as u64) | BIT!(7i32)) as i32);
put_queue(vc, ((keycode as u64) | BIT!(7i32)) as i32);
}
raw_mode = true;
}
assign_bit!(keycode, unsafe { core::ptr::addr_of_mut!(key_down.0) as *mut u64 }, down != 0);
let __v3: bool = (unsafe { rep }) && (vc_kbd_mode(unsafe { kbd }, 2i32) == 0 || !tty.is_null() && L_ECHO!(tty) == 0 && (unsafe { tty_chars_in_buffer(tty) }) != 0);
if __v3 {
// Don't repeat a key if the input buffers are not empty and the
// characters get aren't echoed locally. This makes key repeat
// usable with slow applications and under heavy loads.
unsafe { tty_kref_put(tty) };
return;
}
unsafe { tty_kref_put(tty) };
let __v6: i32 = ((unsafe { shift_state }) | ((unsafe { (*kbd).slockstate }) as i32)) ^ ((unsafe { (*kbd).lockstate }) as i32);
shift_final = __v6;
param.shift = __v6;
param.ledstate = (unsafe { (*kbd).__slate_bits_0.__get_ledflagstate() }) as i32;
key_map = unsafe { key_maps[shift_final as usize] };
// Fall back to the plain map if modifiers are active, the modifier-
// specific map is missing or has no entry, and the plain map has a
// modifier-aware key type (KT_CUR or KT_CSI). These handlers encode
// the modifier state into the emitted escape sequence.
if shift_final != 0 && keycode < 256 && (key_map.is_null() || ((unsafe { *key_map.wrapping_offset(keycode as isize) }) as i32) == K!(2i32, 0i32)) && !unsafe { key_maps[0] }.is_null() {
let plain: u16 = unsafe { *unsafe { key_maps[0] }.wrapping_offset(keycode as isize) };
let r#type: u8 = KTYP!(plain) as u8;
if (r#type as i32) >= 240 && ((r#type as i32).wrapping_sub(240) == 6 || (r#type as i32).wrapping_sub(240) == 15) {
key_map = unsafe { key_maps[0] };
}
}
rc = unsafe { atomic_notifier_call_chain(core::ptr::addr_of_mut!(keyboard_notifier_list), 1u64, core::ptr::addr_of_mut!(param) as *mut ()) };
if rc == 1 | 32768 || key_map.is_null() {
unsafe { atomic_notifier_call_chain(core::ptr::addr_of_mut!(keyboard_notifier_list), 2u64, core::ptr::addr_of_mut!(param) as *mut ()) };
do_compute_shiftstate();
unsafe {
(*kbd).slockstate = 0;
}
return;
}
if keycode < 256 {
keysym = unsafe { *key_map.wrapping_offset(keycode as isize) };
} else {
if keycode >= 497 && keycode <= 504 {
keysym = U__3!(K__2!(14i32, keycode.wrapping_sub(497).wrapping_add(1))) as u16;
} else {
return;
}
}
r#type = KTYP!(keysym) as u8;
if (r#type as i32) < 240 {
param.value = keysym as u32;
rc = unsafe { atomic_notifier_call_chain(core::ptr::addr_of_mut!(keyboard_notifier_list), 3u64, core::ptr::addr_of_mut!(param) as *mut ()) };
if rc != 1 | 32768 {
if down != 0 && !(raw_mode || ((unsafe { (*kbd).__slate_bits_1.__get_kbdmode() }) as i32) == 4) {
k_unicode(vc, keysym as u32, (down == 0) as u8);
}
}
return;
}
r#type = (r#type as i32).wrapping_sub(240) as u8;
if (r#type as i32) == 11 {
r#type = 0;
if vc_kbd_led(unsafe { kbd }, 2i32) != 0 {
key_map = unsafe { key_maps[((shift_final as u64) ^ BIT!(0i32)) as usize] };
if !key_map.is_null() {
keysym = unsafe { *key_map.wrapping_offset(keycode as isize) };
}
}
}
param.value = keysym as u32;
rc = unsafe { atomic_notifier_call_chain(core::ptr::addr_of_mut!(keyboard_notifier_list), 4u64, core::ptr::addr_of_mut!(param) as *mut ()) };
if rc == 1 | 32768 {
return;
}
if (raw_mode || ((unsafe { (*kbd).__slate_bits_1.__get_kbdmode() }) as i32) == 4) && (r#type as i32) != 2 && (r#type as i32) != 7 {
return;
}
unsafe { unsafe { k_handler.0[(r#type as i32) as usize] }.unwrap()(vc, KVAL__2!(keysym) as u8, (down == 0) as u8) };
param.ledstate = (unsafe { (*kbd).__slate_bits_0.__get_ledflagstate() }) as i32;
unsafe { atomic_notifier_call_chain(core::ptr::addr_of_mut!(keyboard_notifier_list), 5u64, core::ptr::addr_of_mut!(param) as *mut ()) };
if (r#type as i32) != 12 {
unsafe {
(*kbd).slockstate = 0;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.kbd_event")]
extern "C" fn kbd_event(handle: *mut input_handle, event_type: u32, event_code: u32, value: i32) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_unlock_529: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_610> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_529) as *mut *mut __SlateRecord_spinlock_h_610;
// We are called with interrupts disabled, just take the lock
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_610> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_610;
let __slate_cleanup_ptr_2 = __slate_slot_2;
let __slate_cleanup_armed_2 = core::cell::Cell::new(false);
let __slate_cleanup_2 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_2.get() {
unsafe { class_spinlock_destructor(__slate_cleanup_ptr_2 as *mut _) };
}
});
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _) };
}
});
unsafe {
core::ptr::write(__slate_slot_2, class_spinlock_constructor(core::ptr::addr_of_mut!(kbd_event_lock)));
__slate_cleanup_armed_2.set(true);
core::ptr::write(__slate_slot_1, (core::ptr::addr_of_mut!(kbd_event_lock) as u64) as *mut __SlateRecord_spinlock_h_610);
__slate_cleanup_armed_1.set(true);
if !class_spinlock_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_2))).is_null() || !class_spinlock_is_conditional {
if true {
if event_type == 4 && event_code == 3 {
__v0 = kbd_is_hw_raw((*handle).dev as *const input_dev);
} else {
__v0 = false;
}
if __v0 {
kbd_rawcode(value as u8);
}
if event_type == 1 && event_code <= 767 {
kbd_keycode(event_code, value, kbd_is_hw_raw((*handle).dev as *const input_dev));
}
}
}
__slate_cleanup_armed_1.set(false);
__class_spinlock_cleanup_ctx(__slate_slot_1 as *mut _);
__slate_cleanup_armed_2.set(false);
class_spinlock_destructor(__slate_slot_2 as *mut _);
tasklet_schedule(core::ptr::addr_of_mut!(keyboard_tasklet));
do_poke_blanked_console = 1;
schedule_console_callback();
}
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.kbd_match")]
extern "C" fn kbd_match(handler: *mut input_handler, dev: *mut input_dev) -> bool {
if test_bit!(18i32, unsafe { (*dev).evbit.as_mut_ptr() as *mut u64 }) != 0 {
return true;
}
if test_bit!(1i32, unsafe { (*dev).evbit.as_mut_ptr() as *mut u64 }) != 0 {
if find_next_bit((unsafe { (*dev).keybit.as_mut_ptr() as *mut u64 }) as *const u64, 256u64, 0u64) < 256 {
return true;
}
if find_next_bit((unsafe { (*dev).keybit.as_mut_ptr() as *mut u64 }) as *const u64, 507u64, 497u64) <= 506 {
return true;
}
}
false
}

/// When a keyboard (or other input device) is found, the kbd_connect
/// function is called. The function then looks at the device, and if it
/// likes it, it can open it and get events from it. In this (kbd_connect)
/// function, we should decide which VT to bind that keyboard to initially.
#[unsafe(link_section = ".text.slate_distinct.keyboard.kbd_connect")]
extern "C" fn kbd_connect(handler: *mut input_handler, dev: *mut input_dev, id: *const input_device_id) -> i32 {
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut handle: core::mem::MaybeUninit<*mut input_handle> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut input_handle = core::ptr::addr_of_mut!(handle) as *mut *mut input_handle;
let mut error: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(error) as *mut i32;
let __slate_cleanup_ptr_6 = __slate_slot_6;
let __slate_cleanup_armed_6 = core::cell::Cell::new(false);
let __slate_cleanup_6 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_6.get() {
unsafe { __free_kfree(__slate_cleanup_ptr_6 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
__slate_cleanup_armed_6.set(true);
__obj_size = size_mul(core::mem::size_of::<input_handle>() as u64, 1u64);
let __slate_auto_bytes_2 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
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
*__slate_slot_6 = *__slate_slot_1 as *mut input_handle;
if (*__slate_slot_6).is_null() {
return 12i32.wrapping_neg();
} else {
(*(*__slate_slot_6)).dev = dev;
(*(*__slate_slot_6)).handler = handler;
(*(*__slate_slot_6)).name = c"kbd".as_ptr() as *const u8;
*__slate_slot_7 = input_register_handle(*__slate_slot_6);
if *__slate_slot_7 != 0 {
return *__slate_slot_7;
} else {
*__slate_slot_7 = input_open_device(*__slate_slot_6);
if *__slate_slot_7 != 0 {
input_unregister_handle(*__slate_slot_6);
return *__slate_slot_7;
} else {
retain_and_null_ptr__2!(*__slate_slot_6);
return 0;
}
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.keyboard.kbd_disconnect")]
extern "C" fn kbd_disconnect(handle: *mut input_handle) {
unsafe { input_close_device(handle) };
unsafe { input_unregister_handle(handle) };
unsafe { kfree(handle as *const ()) };
}

/// Start keyboard handler on the new keyboard by refreshing LED state to
/// match the rest of the system.
#[unsafe(link_section = ".text.slate_distinct.keyboard.kbd_start")]
extern "C" fn kbd_start(handle: *mut input_handle) {
tasklet_disable(unsafe { core::ptr::addr_of_mut!(keyboard_tasklet) });
if (unsafe { ledstate }) != 1u32.wrapping_neg() {
kbd_update_leds_helper(handle, (unsafe { core::ptr::addr_of_mut!(ledstate) }) as *mut ());
}
tasklet_enable(unsafe { core::ptr::addr_of_mut!(keyboard_tasklet) });
}

/// Terminating entry
static kbd_ids: __SlateAlign16<[input_device_id; 3]> = __SlateAlign16([{
let mut __t0: input_device_id = unsafe { core::mem::zeroed() };
__t0.flags = 16u64;
__t0.evbit = [2u64];
__t0
}, {
let mut __t1: input_device_id = unsafe { core::mem::zeroed() };
__t1.flags = 16u64;
__t1.evbit = [262144u64];
__t1
}, {
let mut __t2: input_device_id = unsafe { core::mem::zeroed() };
__t2
}]);

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn kbd_init() -> i32 {
let mut error: i32 = 0;
for i in 0i32..63i32 {
unsafe {
let __t0 = kbd_defleds() as u8;
(*core::ptr::addr_of_mut!(kbd_table.0[i as usize].__slate_bits_0)).__set_ledflagstate(__t0);
}
unsafe {
let __t1 = kbd_defleds() as u8;
(*core::ptr::addr_of_mut!(kbd_table.0[i as usize].__slate_bits_1)).__set_default_ledflagstate(__t1);
}
unsafe {
(*core::ptr::addr_of_mut!(kbd_table.0[i as usize].__slate_bits_0)).__set_ledmode(0u8);
}
unsafe {
kbd_table.0[i as usize].lockstate = 0;
}
unsafe {
kbd_table.0[i as usize].slockstate = 0;
}
unsafe {
let __t2 = (BIT!(2i32) | BIT!(4i32)) as u8;
(*core::ptr::addr_of_mut!(kbd_table.0[i as usize].__slate_bits_2)).__set_modeflags(__t2);
}
unsafe {
let __t3 = (if default_utf8 != 0 { 3i32 } else { 0i32 }) as u8;
(*core::ptr::addr_of_mut!(kbd_table.0[i as usize].__slate_bits_1)).__set_kbdmode(__t3);
}
}
kbd_init_leds();
error = unsafe { input_register_handler(core::ptr::addr_of_mut!(kbd_handler)) };
if error != 0 {
return error;
}
tasklet_enable(unsafe { core::ptr::addr_of_mut!(keyboard_tasklet) });
tasklet_schedule(unsafe { core::ptr::addr_of_mut!(keyboard_tasklet) });
0
}

// Ioctl support code
fn vt_do_kdgkbdiacr(udp: *mut ()) -> i32 {
let mut __slate_compound_19: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_21: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __x: u32 = unsafe { core::mem::zeroed() };
let mut __val_pu: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u32 = core::ptr::addr_of_mut!(__val_pu) as *mut u32;
let mut __ptr_pu: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu) as *mut *mut ();
let mut __ret_pu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(__ret_pu) as *mut i32;
let mut __UNIQUE_ID_unlock_531: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_531) as *mut *mut __SlateRecord_spinlock_h_642;
// Lock the diacriticals table, make a copy and then
// copy it after we unlock
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_642;
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut dia: core::mem::MaybeUninit<*mut kbdiacr> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut kbdiacr = core::ptr::addr_of_mut!(dia) as *mut *mut kbdiacr;
let mut asize: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(asize) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut a: *mut kbdiacrs = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_15 = __slate_slot_15;
let __slate_cleanup_armed_15 = core::cell::Cell::new(false);
let __slate_cleanup_15 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_15.get() {
unsafe { __free_kfree(__slate_cleanup_ptr_15 as *mut _) };
}
});
let __slate_cleanup_ptr_10 = __slate_slot_10;
let __slate_cleanup_armed_10 = core::cell::Cell::new(false);
let __slate_cleanup_10 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_10.get() {
unsafe { class_spinlock_irqsave_destructor(__slate_cleanup_ptr_10 as *mut _) };
}
});
let __slate_cleanup_ptr_9 = __slate_slot_9;
let __slate_cleanup_armed_9 = core::cell::Cell::new(false);
let __slate_cleanup_9 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_9.get() {
unsafe { __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_9 as *mut _) };
}
});
unsafe {
a = udp as *mut kbdiacrs;
let __slate_auto_bytes_2 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
__slate_cleanup_armed_15.set(true);
let __slate_auto_bytes_4 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_11 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_12 = _kmalloc_array_noprof(256u64, core::mem::size_of::<kbdiacr>() as u64, 3264u32, *{
__slate_compound_19 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_19)
});
alloc_tag_restore!();
} else {
*__slate_slot_12 = _kmalloc_array_noprof(256u64, core::mem::size_of::<kbdiacr>() as u64, 3264u32, *{
__slate_compound_21 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_21)
});
}
*__slate_slot_15 = *__slate_slot_12 as *mut kbdiacr;
if (*__slate_slot_15).is_null() {
return 12i32.wrapping_neg();
} else {
core::ptr::write(__slate_slot_10, class_spinlock_irqsave_constructor(core::ptr::addr_of_mut!(kbd_event_lock)));
__slate_cleanup_armed_10.set(true);
core::ptr::write(__slate_slot_9, (core::ptr::addr_of_mut!(kbd_event_lock) as u64) as *mut __SlateRecord_spinlock_h_642);
__slate_cleanup_armed_9.set(true);
if !class_spinlock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_10))).is_null() || !class_spinlock_irqsave_is_conditional {
if true {
*__slate_slot_16 = accent_table_size as i32;
*__slate_slot_17 = 0;
while *__slate_slot_17 < *__slate_slot_16 {
(*(*__slate_slot_15).wrapping_offset(*__slate_slot_17 as isize)).diacr = conv_uni_to_8bit((*(core::ptr::addr_of_mut!(accent_table) as *mut kbdiacruc).wrapping_offset(*__slate_slot_17 as isize)).diacr) as u8;
(*(*__slate_slot_15).wrapping_offset(*__slate_slot_17 as isize)).base = conv_uni_to_8bit((*(core::ptr::addr_of_mut!(accent_table) as *mut kbdiacruc).wrapping_offset(*__slate_slot_17 as isize)).base) as u8;
(*(*__slate_slot_15).wrapping_offset(*__slate_slot_17 as isize)).result = conv_uni_to_8bit((*(core::ptr::addr_of_mut!(accent_table) as *mut kbdiacruc).wrapping_offset(*__slate_slot_17 as isize)).result) as u8;
let __v8: i32 = *__slate_slot_17;
*__slate_slot_17 = __v8.wrapping_add(1);
}
}
}
__slate_cleanup_armed_9.set(false);
__class_spinlock_irqsave_cleanup_ctx(__slate_slot_9 as *mut _);
__slate_cleanup_armed_10.set(false);
class_spinlock_irqsave_destructor(__slate_slot_10 as *mut _);
might_fault();
let __slate_auto_bytes_5 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x = *__slate_slot_16 as u32;
let __ptr: *mut u32 = core::ptr::addr_of_mut!((*a).kb_cnt);
__chk_user_ptr!();
*__slate_slot_3 = __ptr as *mut ();
*__slate_slot_2 = __x;
let __t0: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_3 as u64 => __t0, in("ax") *__slate_slot_2 as u32, options(att_syntax));
*__slate_slot_4 = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
instrument_put_user__2!(__x, __ptr);
let __v6: i64 = *__slate_slot_4 as i64;
if __v6 != 0 {
return 14i32.wrapping_neg();
} else {
if copy_to_user(((*a).kbdiacr.as_mut_ptr() as *mut kbdiacr) as *mut (), *__slate_slot_15 as *const (), (*__slate_slot_16 as u64).wrapping_mul(core::mem::size_of::<kbdiacr>() as u64)) != 0 {
return 14i32.wrapping_neg();
} else {
return 0;
}
}
}
}
}

fn vt_do_kdgkbdiacruc(udp: *mut ()) -> i32 {
let mut __slate_compound_16: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_18: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __x: u32 = unsafe { core::mem::zeroed() };
let mut __val_pu: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u32 = core::ptr::addr_of_mut!(__val_pu) as *mut u32;
let mut __ptr_pu: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu) as *mut *mut ();
let mut __ret_pu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(__ret_pu) as *mut i32;
let mut __UNIQUE_ID_unlock_533: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_533) as *mut *mut __SlateRecord_spinlock_h_642;
// Lock the diacriticals table, make a copy and then
// copy it after we unlock
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_642;
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut buf__0: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut () = core::ptr::addr_of_mut!(buf__0) as *mut *mut ();
let mut asize: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut i32 = core::ptr::addr_of_mut!(asize) as *mut i32;
let mut a: *mut kbdiacrsuc = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_13 = __slate_slot_13;
let __slate_cleanup_armed_13 = core::cell::Cell::new(false);
let __slate_cleanup_13 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_13.get() {
unsafe { __free_kfree(__slate_cleanup_ptr_13 as *mut _) };
}
});
let __slate_cleanup_ptr_8 = __slate_slot_8;
let __slate_cleanup_armed_8 = core::cell::Cell::new(false);
let __slate_cleanup_8 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_8.get() {
unsafe { class_spinlock_irqsave_destructor(__slate_cleanup_ptr_8 as *mut _) };
}
});
let __slate_cleanup_ptr_7 = __slate_slot_7;
let __slate_cleanup_armed_7 = core::cell::Cell::new(false);
let __slate_cleanup_7 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_7.get() {
unsafe { __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_7 as *mut _) };
}
});
unsafe {
a = udp as *mut kbdiacrsuc;
let __slate_auto_bytes_2 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
__slate_cleanup_armed_13.set(true);
let __slate_auto_bytes_3 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_9 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_10 = _kmalloc_array_noprof(256u64, core::mem::size_of::<kbdiacruc>() as u64, 3264u32, *{
__slate_compound_16 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_16)
});
alloc_tag_restore!();
} else {
*__slate_slot_10 = _kmalloc_array_noprof(256u64, core::mem::size_of::<kbdiacruc>() as u64, 3264u32, *{
__slate_compound_18 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_18)
});
}
*__slate_slot_13 = *__slate_slot_10;
if (*__slate_slot_13).is_null() {
return 12i32.wrapping_neg();
} else {
core::ptr::write(__slate_slot_8, class_spinlock_irqsave_constructor(core::ptr::addr_of_mut!(kbd_event_lock)));
__slate_cleanup_armed_8.set(true);
core::ptr::write(__slate_slot_7, (core::ptr::addr_of_mut!(kbd_event_lock) as u64) as *mut __SlateRecord_spinlock_h_642);
__slate_cleanup_armed_7.set(true);
if !class_spinlock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_8))).is_null() || !class_spinlock_irqsave_is_conditional {
if true {
*__slate_slot_14 = accent_table_size as i32;
memcpy(*__slate_slot_13, (core::ptr::addr_of_mut!(accent_table) as *mut kbdiacruc) as *const (), (*__slate_slot_14 as u64).wrapping_mul(core::mem::size_of::<kbdiacruc>() as u64));
}
}
__slate_cleanup_armed_7.set(false);
__class_spinlock_irqsave_cleanup_ctx(__slate_slot_7 as *mut _);
__slate_cleanup_armed_8.set(false);
class_spinlock_irqsave_destructor(__slate_slot_8 as *mut _);
might_fault();
let __slate_auto_bytes_4 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x = *__slate_slot_14 as u32;
let __ptr: *mut u32 = core::ptr::addr_of_mut!((*a).kb_cnt);
__chk_user_ptr!();
*__slate_slot_3 = __ptr as *mut ();
*__slate_slot_2 = __x;
let __t0: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_3 as u64 => __t0, in("ax") *__slate_slot_2 as u32, options(att_syntax));
*__slate_slot_4 = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
instrument_put_user__2!(__x, __ptr);
let __v6: i64 = *__slate_slot_4 as i64;
if __v6 != 0 {
return 14i32.wrapping_neg();
} else {
if copy_to_user(((*a).kbdiacruc.as_mut_ptr() as *mut kbdiacruc) as *mut (), *__slate_slot_13 as *const (), (*__slate_slot_14 as u64).wrapping_mul(core::mem::size_of::<kbdiacruc>() as u64)) != 0 {
return 14i32.wrapping_neg();
} else {
return 0;
}
}
}
}
}

fn vt_do_kdskbdiacr(udp: *mut (), perm: i32) -> i32 {
let mut a: *mut kbdiacrs = udp as *mut kbdiacrs;
let mut dia: *mut kbdiacr = core::ptr::null_mut::<kbdiacr>();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(dia) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_kfree(__slate_cleanup_ptr_0 as *mut _));
let mut ct: u32 = 0;
let mut i: i32 = 0;
if perm == 0 {
return 1i32.wrapping_neg();
}
let mut __v1: i64;
might_fault();
let mut __v2: i64;
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t0: u64;
let __t1: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*a).kb_cnt) as u64 => __t0, lateout("dx") __t1, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
instrument_get_user!(__val_gu);
ct = __val_gu;
__v2 = __ret_gu as i64;
__v1 = __v2;
if __v1 != 0 {
return 14i32.wrapping_neg();
}
if ct >= 256 {
return 22i32.wrapping_neg();
}
if ct != 0 {
dia = memdup_array_user((unsafe { (*a).kbdiacr.as_mut_ptr() as *mut kbdiacr }) as *const (), ct as u64, core::mem::size_of::<kbdiacr>() as u64) as *mut kbdiacr;
if IS_ERR(dia as *const ()) {
return PTR_ERR(dia as *const ()) as i32;
}
}
let mut __UNIQUE_ID_guard_534: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(kbd_event_lock) });
let __slate_cleanup_ptr_5 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_534) as *mut _;
let __slate_cleanup_5 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_5 as *mut _));
let mut __UNIQUE_ID_unlock_535: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(kbd_event_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_6 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_535) as *mut _;
let __slate_cleanup_6 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_6 as *mut _));
unsafe {
accent_table_size = ct;
}
while (i as u32) < ct {
unsafe {
(*(core::ptr::addr_of_mut!(accent_table) as *mut kbdiacruc).wrapping_offset(i as isize)).diacr = conv_8bit_to_uni((*dia.wrapping_offset(i as isize)).diacr);
}
unsafe {
(*(core::ptr::addr_of_mut!(accent_table) as *mut kbdiacruc).wrapping_offset(i as isize)).base = conv_8bit_to_uni((*dia.wrapping_offset(i as isize)).base);
}
unsafe {
(*(core::ptr::addr_of_mut!(accent_table) as *mut kbdiacruc).wrapping_offset(i as isize)).result = conv_8bit_to_uni((*dia.wrapping_offset(i as isize)).result);
}
i = i.wrapping_add(1);
}
0
}

fn vt_do_kdskbdiacruc(udp: *mut (), perm: i32) -> i32 {
let mut a: *mut kbdiacrsuc = udp as *mut kbdiacrsuc;
let mut ct: u32 = 0;
let mut buf__0: *mut () = core::ptr::null_mut::<()>();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(buf__0) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_kfree(__slate_cleanup_ptr_0 as *mut _));
if perm == 0 {
return 1i32.wrapping_neg();
}
let mut __v1: i64;
might_fault();
let mut __v2: i64;
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t0: u64;
let __t1: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*a).kb_cnt) as u64 => __t0, lateout("dx") __t1, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
instrument_get_user!(__val_gu);
ct = __val_gu;
__v2 = __ret_gu as i64;
__v1 = __v2;
if __v1 != 0 {
return 14i32.wrapping_neg();
}
if ct >= 256 {
return 22i32.wrapping_neg();
}
if ct != 0 {
buf__0 = memdup_array_user((unsafe { (*a).kbdiacruc.as_mut_ptr() as *mut kbdiacruc }) as *const (), ct as u64, core::mem::size_of::<kbdiacruc>() as u64);
if IS_ERR(buf__0 as *const ()) {
return PTR_ERR(buf__0 as *const ()) as i32;
}
}
let mut __UNIQUE_ID_guard_536: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(kbd_event_lock) });
let __slate_cleanup_ptr_5 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_536) as *mut _;
let __slate_cleanup_5 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_5 as *mut _));
let mut __UNIQUE_ID_unlock_537: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(kbd_event_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_6 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_537) as *mut _;
let __slate_cleanup_6 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_6 as *mut _));
if ct != 0 {
unsafe { memcpy((core::ptr::addr_of_mut!(accent_table) as *mut kbdiacruc) as *mut (), buf__0 as *const (), (ct as u64).wrapping_mul(core::mem::size_of::<kbdiacruc>() as u64)) };
}
unsafe {
accent_table_size = ct;
}
0
}

/// vt_do_diacrit		-	diacritical table updates
/// @cmd: ioctl request
/// @udp: pointer to user data for ioctl
/// @perm: permissions check computed by caller
///
/// Update the diacritical tables atomically and safely. Lock them
/// against simultaneous keypresses
#[unsafe(no_mangle)]
extern "C" fn vt_do_diacrit(cmd: u32, udp: *mut (), perm: i32) -> i32 {
match cmd {
19274 => {
return vt_do_kdgkbdiacr(udp);
}
19450 => {
return vt_do_kdgkbdiacruc(udp);
}
19275 => {
return vt_do_kdskbdiacr(udp, perm);
}
19451 => {
return vt_do_kdskbdiacruc(udp, perm);
}
_ => {
}
}
0
}

/// vt_do_kdskbmode		-	set keyboard mode ioctl
/// @console: the console to use
/// @arg: the requested mode
///
/// Update the keyboard mode bits while holding the correct locks.
/// Return 0 for success or an error code.
#[unsafe(no_mangle)]
extern "C" fn vt_do_kdskbmode(console: u32, arg: u32) -> i32 {
let kb: *mut kbd_struct = unsafe { core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct }.wrapping_offset(console as isize);
let mut __UNIQUE_ID_guard_538: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(kbd_event_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_538) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_539: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(kbd_event_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_539) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
match arg {
0 => {
unsafe {
(*kb).__slate_bits_1.__set_kbdmode(2u8);
}
return 0;
}
2 => {
unsafe {
(*kb).__slate_bits_1.__set_kbdmode(1u8);
}
return 0;
}
1 => {
unsafe {
(*kb).__slate_bits_1.__set_kbdmode(0u8);
}
do_compute_shiftstate();
return 0;
}
3 => {
unsafe {
(*kb).__slate_bits_1.__set_kbdmode(3u8);
}
do_compute_shiftstate();
return 0;
}
4 => {
unsafe {
(*kb).__slate_bits_1.__set_kbdmode(4u8);
}
return 0;
}
_ => {
return 22i32.wrapping_neg();
}
}
unsafe { core::mem::zeroed() }
}

/// vt_do_kdskbmeta		-	set keyboard meta state
/// @console: the console to use
/// @arg: the requested meta state
///
/// Update the keyboard meta bits while holding the correct locks.
/// Return 0 for success or an error code.
#[unsafe(no_mangle)]
extern "C" fn vt_do_kdskbmeta(console: u32, arg: u32) -> i32 {
let kb: *mut kbd_struct = unsafe { core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct }.wrapping_offset(console as isize);
let mut __UNIQUE_ID_guard_540: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(kbd_event_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_540) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_541: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(kbd_event_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_541) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
match arg {
3 => {
clr_vc_kbd_mode(kb, 4i32);
return 0;
}
4 => {
set_vc_kbd_mode(kb, 4i32);
return 0;
}
_ => {
return 22i32.wrapping_neg();
}
}
unsafe { core::mem::zeroed() }
}

#[unsafe(no_mangle)]
extern "C" fn vt_do_kbkeycode_ioctl(cmd: i32, user_kbkc: *mut kbkeycode, perm: i32) -> i32 {
let mut tmp: kbkeycode = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tmp) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut kc: i32 = 0;
if copy_from_user(core::ptr::addr_of_mut!(tmp) as *mut (), user_kbkc as *const (), core::mem::size_of::<kbkeycode>() as u64) != 0 {
return 14i32.wrapping_neg();
}
match cmd {
19276 => {
kc = getkeycode(tmp.scancode);
if kc < 0 {
return kc;
}
let mut __v0: i64;
might_fault();
let mut __v1: i64;
let mut __ret_pu: i32 = 0;
let mut __ptr_pu: *mut () = unsafe { core::mem::zeroed() };
let mut __val_pu: u32 = 0;
let mut __x: u32 = kc as u32;
let __ptr: *mut u32 = unsafe { core::ptr::addr_of_mut!((*user_kbkc).keycode) };
__chk_user_ptr!();
__ptr_pu = __ptr as *mut ();
__val_pu = __x;
unsafe {
let __t0: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") __ptr_pu as u64 => __t0, in("ax") __val_pu as u32, options(att_syntax));
__ret_pu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
instrument_put_user__2!(__x, __ptr);
__v1 = __ret_pu as i64;
__v0 = __v1;
return __v0 as i32;
}
19277 => {
if perm == 0 {
return 1i32.wrapping_neg();
}
return setkeycode(tmp.scancode, tmp.keycode);
}
_ => {
}
}
0
}

fn vt_kdgkbent(kbdmode: u8, idx: u8, map: u8) -> u16 {
// Ensure another thread doesn't free it under us
let mut __UNIQUE_ID_guard_542: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(kbd_event_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_542) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_543: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(kbd_event_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_543) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
let key_map: *mut u16 = unsafe { key_maps[(map as i32) as usize] };
if !key_map.is_null() {
let val: u16 = U!(unsafe { *key_map.wrapping_offset((idx as i32) as isize) }) as u16;
if (kbdmode as i32) != 3 && KTYP!(val) >= unsafe { NR_TYPES } {
return K!(2i32, 0i32) as u16;
}
return val;
}
(if idx != 0 { K!(2i32, 0i32) } else { K!(2i32, 127i32) }) as u16
}

fn vt_kdskbent(kbdmode: u8, idx: u8, map: u8, val: u16) -> i32 {
let mut __slate_compound_7: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut key_map: *mut u16 = unsafe { core::mem::zeroed() };
let mut oldval: u16 = 0;
if idx == 0 && (val as i32) == K!(2i32, 127i32) {
let mut __UNIQUE_ID_guard_544: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(kbd_event_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_544) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_545: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(kbd_event_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_545) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
// deallocate map
key_map = unsafe { key_maps[(map as i32) as usize] };
if map != 0 && !key_map.is_null() {
unsafe {
key_maps[(map as i32) as usize] = core::ptr::null_mut::<u16>();
}
if ((unsafe { *key_map.wrapping_offset(0 as isize) }) as i32) == U__2!(K!(2i32, 126i32)) {
unsafe { kfree(key_map as *const ()) };
let __v2: u32 = unsafe { keymap_count };
unsafe {
keymap_count = __v2.wrapping_sub(1);
}
}
}
return 0;
}
if KTYP!(val) < unsafe { NR_TYPES } {
if KVAL__2!(val) > ((unsafe { max_vals.0[KTYP!(val) as usize] }) as i32) {
return 22i32.wrapping_neg();
}
} else {
if (kbdmode as i32) != 3 {
return 22i32.wrapping_neg();
}
}
// ++Geert: non-PC keyboards may generate keycode zero
//
// assignment to entry 0 only tests validity of args
if idx == 0 {
return 0;
}
let mut new_map: *mut u16 = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_4 = core::ptr::addr_of_mut!(new_map) as *mut _;
let __slate_cleanup_4 = ::slate_support::Guard(&mut || __free_kfree(__slate_cleanup_ptr_4 as *mut _));
let mut __v5: *mut ();
let mut __v6: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(core::mem::size_of::<[u16; 256]>() as u64, 3264u32, unsafe { *{
__slate_compound_7 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_7)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(core::mem::size_of::<[u16; 256]>() as u64, 3264u32, unsafe { *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
} });
}
__v6 = _res;
__v5 = __v6;
new_map = __v5 as *mut u16;
if new_map.is_null() {
return 12i32.wrapping_neg();
}
let mut __UNIQUE_ID_guard_546: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(kbd_event_lock) });
let __slate_cleanup_ptr_10 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_546) as *mut _;
let __slate_cleanup_10 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_10 as *mut _));
let mut __UNIQUE_ID_unlock_547: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(kbd_event_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_11 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_547) as *mut _;
let __slate_cleanup_11 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_11 as *mut _));
key_map = unsafe { key_maps[(map as i32) as usize] };
if key_map.is_null() {
let __v12: bool = (unsafe { keymap_count }) >= 256 && !unsafe { capable(24i32) };
if __v12 {
return 1i32.wrapping_neg();
}
let __v13: *mut u16 = no_free_ptr!(new_map);
unsafe {
key_maps[(map as i32) as usize] = __v13;
}
key_map = __v13;
unsafe {
*key_map.wrapping_offset(0 as isize) = U__2!(K!(2i32, 126i32)) as u16;
}
for j in 1i32..256i32 {
unsafe {
*key_map.wrapping_offset(j as isize) = U__2!(K!(2i32, 0i32)) as u16;
}
}
let __v20: u32 = unsafe { keymap_count };
unsafe {
keymap_count = __v20.wrapping_add(1);
}
}
oldval = U!(unsafe { *key_map.wrapping_offset((idx as i32) as isize) }) as u16;
if (val as i32) == (oldval as i32) {
return 0;
}
// Attention Key
let __v22: bool = ((oldval as i32) == K!(2i32, 15i32) || (val as i32) == K!(2i32, 15i32)) && !unsafe { capable(21i32) };
if __v22 {
return 1i32.wrapping_neg();
}
unsafe {
*key_map.wrapping_offset((idx as i32) as isize) = U!(val) as u16;
}
if map == 0 && (KTYP!(oldval) == 7 || KTYP!(val) == 7) {
do_compute_shiftstate();
}
0
}

#[unsafe(no_mangle)]
extern "C" fn vt_do_kdsk_ioctl(cmd: i32, user_kbe: *mut kbentry, perm: i32, console: u32) -> i32 {
let kb: *mut kbd_struct = unsafe { core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct }.wrapping_offset(console as isize);
let mut kbe: kbentry = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(kbe) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
if copy_from_user(core::ptr::addr_of_mut!(kbe) as *mut (), user_kbe as *const (), core::mem::size_of::<kbentry>() as u64) != 0 {
return 14i32.wrapping_neg();
}
match cmd {
19270 => {
let mut __v0: i64;
might_fault();
let mut __v1: i64;
let mut __ret_pu: i32 = 0;
let mut __ptr_pu: *mut () = unsafe { core::mem::zeroed() };
let mut __val_pu: u16 = 0;
let mut __x: u16 = vt_kdgkbent(((unsafe { (*kb).__slate_bits_1.__get_kbdmode() }) as i32) as u8, kbe.kb_index, kbe.kb_table);
let __ptr: *mut u16 = unsafe { core::ptr::addr_of_mut!((*user_kbe).kb_value) };
__chk_user_ptr!();
__ptr_pu = __ptr as *mut ();
__val_pu = __x;
unsafe {
let __t0: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 2 as u64, out(reg) _, inlateout("cx") __ptr_pu as u64 => __t0, in("ax") __val_pu as u16, options(att_syntax));
__ret_pu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
instrument_put_user__4!(__x, __ptr);
__v1 = __ret_pu as i64;
__v0 = __v1;
return __v0 as i32;
}
19271 => {
let __v2: bool = perm == 0 || !unsafe { capable(26i32) };
if __v2 {
return 1i32.wrapping_neg();
}
return vt_kdskbent(((unsafe { (*kb).__slate_bits_1.__get_kbdmode() }) as i32) as u8, kbe.kb_index, kbe.kb_table, kbe.kb_value);
}
_ => {
}
}
0
}

fn vt_kdskbsent(kbs: *mut u8, cur: u8) -> *mut u8 {
let cur_f: *mut u8 = unsafe { func_table[(cur as i32) as usize] };
let __v0: bool = !cur_f.is_null() && (unsafe { strlen(cur_f as *const u8) }) >= unsafe { strlen(kbs as *const u8) };
if __v0 {
unsafe { strcpy(cur_f, kbs as *const u8) };
return kbs;
}
unsafe {
func_table[(cur as i32) as usize] = kbs;
}
if __test_and_set_bit!(cur, unsafe { core::ptr::addr_of_mut!(is_kmalloc.0) as *mut u64 }) != 0 { cur_f } else { core::ptr::null_mut::<u8>() }
}

static mut is_kmalloc: __SlateAlign16<[u64; 4]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[unsafe(no_mangle)]
extern "C" fn vt_do_kdskled(console: u32, cmd: i32, arg: u64, perm: i32) -> i32 {
let mut __x: u8 = unsafe { core::mem::zeroed() };
let mut __val_pu: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u8 = core::ptr::addr_of_mut!(__val_pu) as *mut u8;
let mut __ptr_pu: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu) as *mut *mut ();
let mut __ret_pu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(__ret_pu) as *mut i32;
let mut __UNIQUE_ID_unlock_557: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_557) as *mut *mut __SlateRecord_spinlock_h_642;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_642;
let mut __x_1: u8 = unsafe { core::mem::zeroed() };
let mut __val_pu: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u8 = core::ptr::addr_of_mut!(__val_pu) as *mut u8;
let mut __ptr_pu: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu) as *mut *mut ();
let mut __ret_pu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i32 = core::ptr::addr_of_mut!(__ret_pu) as *mut i32;
let mut __UNIQUE_ID_unlock_555: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_555) as *mut *mut __SlateRecord_spinlock_h_642;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_642> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_642;
let mut ucval: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut u8 = core::ptr::addr_of_mut!(ucval) as *mut u8;
let mut kb: *mut kbd_struct = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_17 = __slate_slot_17;
let __slate_cleanup_armed_17 = core::cell::Cell::new(false);
let __slate_cleanup_17 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_17.get() {
unsafe { class_spinlock_irqsave_destructor(__slate_cleanup_ptr_17 as *mut _) };
}
});
let __slate_cleanup_ptr_16 = __slate_slot_16;
let __slate_cleanup_armed_16 = core::cell::Cell::new(false);
let __slate_cleanup_16 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_16.get() {
unsafe { __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_16 as *mut _) };
}
});
let __slate_cleanup_ptr_8 = __slate_slot_8;
let __slate_cleanup_armed_8 = core::cell::Cell::new(false);
let __slate_cleanup_8 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_8.get() {
unsafe { class_spinlock_irqsave_destructor(__slate_cleanup_ptr_8 as *mut _) };
}
});
let __slate_cleanup_ptr_7 = __slate_slot_7;
let __slate_cleanup_armed_7 = core::cell::Cell::new(false);
let __slate_cleanup_7 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_7.get() {
unsafe { __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_7 as *mut _) };
}
});
unsafe {
kb = (core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct).wrapping_offset(console as isize);
let __slate_auto_bytes_4 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 1usize);
let __t4: i32 = cmd;
if __t4 == 19300 {
core::ptr::write(__slate_slot_17, class_spinlock_irqsave_constructor(core::ptr::addr_of_mut!(kbd_event_lock)));
__slate_cleanup_armed_17.set(true);
core::ptr::write(__slate_slot_16, (core::ptr::addr_of_mut!(kbd_event_lock) as u64) as *mut __SlateRecord_spinlock_h_642);
__slate_cleanup_armed_16.set(true);
if !class_spinlock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_17))).is_null() || !class_spinlock_irqsave_is_conditional {
if true {
*__slate_slot_18 = (((*kb).__slate_bits_0.__get_ledflagstate() as i32) | ((*kb).__slate_bits_1.__get_default_ledflagstate() as i32) << 4i32) as u8;
}
}
__slate_cleanup_armed_16.set(false);
__class_spinlock_irqsave_cleanup_ctx(__slate_slot_16 as *mut _);
__slate_cleanup_armed_17.set(false);
class_spinlock_irqsave_destructor(__slate_slot_17 as *mut _);
might_fault();
let __slate_auto_bytes_2 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
__x_1 = *__slate_slot_18;
let __ptr: *mut u8 = arg as *mut u8;
__chk_user_ptr!();
*__slate_slot_12 = __ptr as *mut ();
*__slate_slot_11 = __x_1;
let __t3: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 1 as u64, out(reg) _, inlateout("cx") *__slate_slot_12 as u64 => __t3, in("ax") (*__slate_slot_11 as u8) as u32, options(att_syntax));
*__slate_slot_13 = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
instrument_put_user__3!(__x_1, __ptr);
return (*__slate_slot_13 as i64) as i32;
} else {
if __t4 == 19301 {
if perm == 0 {
return 1i32.wrapping_neg();
} else {
if arg & 18446744073709551496u64 != 0 {
return 22i32.wrapping_neg();
} else {
core::ptr::write(__slate_slot_8, class_spinlock_irqsave_constructor(core::ptr::addr_of_mut!(led_lock)));
__slate_cleanup_armed_8.set(true);
core::ptr::write(__slate_slot_7, (core::ptr::addr_of_mut!(led_lock) as u64) as *mut __SlateRecord_spinlock_h_642);
__slate_cleanup_armed_7.set(true);
if !class_spinlock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_8))).is_null() || !class_spinlock_irqsave_is_conditional {
if true {
let __t1 = (arg & 7) as u8;
(*kb).__slate_bits_0.__set_ledflagstate(__t1);
let __t2 = (arg >> 4i32 & 7) as u8;
(*kb).__slate_bits_1.__set_default_ledflagstate(__t2);
set_leds();
}
}
__slate_cleanup_armed_7.set(false);
__class_spinlock_irqsave_cleanup_ctx(__slate_slot_7 as *mut _);
__slate_cleanup_armed_8.set(false);
class_spinlock_irqsave_destructor(__slate_slot_8 as *mut _);
return 0;
}
}
} else {
if __t4 == 19249 {
*__slate_slot_18 = getledstate();
might_fault();
let __slate_auto_bytes_5 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
__x = *__slate_slot_18;
let __ptr: *mut u8 = arg as *mut u8;
__chk_user_ptr!();
*__slate_slot_3 = __ptr as *mut ();
*__slate_slot_2 = __x;
let __t0: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 1 as u64, out(reg) _, inlateout("cx") *__slate_slot_3 as u64 => __t0, in("ax") (*__slate_slot_2 as u8) as u32, options(att_syntax));
*__slate_slot_4 = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
instrument_put_user__3!(__x, __ptr);
return (*__slate_slot_4 as i64) as i32;
} else {
if __t4 == 19250 {
if perm == 0 {
return 1i32.wrapping_neg();
} else {
setledstate(kb, arg as u32);
return 0;
}
} else {
return 515i32.wrapping_neg();
}
}
}
}
}
// the ioctls below read/set the flags usually shown in the leds
//
// don't use them - they will go away without warning
// the ioctls below only set the lights, not the functions
//
// for those, see KDGKBLED and KDSKBLED above
}

#[unsafe(no_mangle)]
extern "C" fn vt_do_kdgkbmode(console: u32) -> i32 {
let kb: *mut kbd_struct = unsafe { core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct }.wrapping_offset(console as isize);
// This is a spot read so needs no locking
match (unsafe { (*kb).__slate_bits_1.__get_kbdmode() }) as i32 {
2 => {
return 0;
}
1 => {
return 2;
}
3 => {
return 3;
}
4 => {
return 4;
}
_ => {
return 1;
}
}
unsafe { core::mem::zeroed() }
}

/// vt_do_kdgkbmeta		-	report meta status
/// @console: console to report
///
/// Report the meta flag status of this console
#[unsafe(no_mangle)]
extern "C" fn vt_do_kdgkbmeta(console: u32) -> i32 {
let kb: *mut kbd_struct = unsafe { core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct }.wrapping_offset(console as isize);
// Again a spot read so no locking
if vc_kbd_mode(kb, 4i32) != 0 { 4 } else { 3 }
}

/// vt_reset_unicode	-	reset the unicode status
/// @console: console being reset
///
/// Restore the unicode console state to its default
#[unsafe(no_mangle)]
extern "C" fn vt_reset_unicode(console: u32) {
let mut __UNIQUE_ID_guard_558: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(kbd_event_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_558) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_559: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(kbd_event_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_559) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
unsafe {
let __t0 = (if default_utf8 != 0 { 3i32 } else { 0i32 }) as u8;
(*core::ptr::addr_of_mut!(kbd_table.0[console as usize].__slate_bits_1)).__set_kbdmode(__t0);
}
}

/// vt_get_shift_state	-	shift bit state
///
/// Report the shift bits from the keyboard state. We have to export
/// this to support some oddities in the vt layer.
#[unsafe(no_mangle)]
extern "C" fn vt_get_shift_state() -> i32 {
// Don't lock as this is a transient report
unsafe { shift_state }
}

/// vt_reset_keyboard	-	reset keyboard state
/// @console: console to reset
///
/// Reset the keyboard bits for a console as part of a general console
/// reset event
#[unsafe(no_mangle)]
extern "C" fn vt_reset_keyboard(console: u32) {
let kb: *mut kbd_struct = unsafe { core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct }.wrapping_offset(console as isize);
let mut __UNIQUE_ID_guard_560: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(kbd_event_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_560) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_561: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(kbd_event_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_561) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
set_vc_kbd_mode(kb, 2i32);
clr_vc_kbd_mode(kb, 1i32);
clr_vc_kbd_mode(kb, 0i32);
clr_vc_kbd_mode(kb, 3i32);
unsafe {
(*kb).lockstate = 0;
}
unsafe {
(*kb).slockstate = 0;
}
let mut __UNIQUE_ID_guard_562: __SlateRecord_spinlock_h_610 = class_spinlock_constructor(unsafe { core::ptr::addr_of_mut!(led_lock) });
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_562) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || class_spinlock_destructor(__slate_cleanup_ptr_2 as *mut _));
let mut __UNIQUE_ID_unlock_563: *mut __SlateRecord_spinlock_h_610 = ((unsafe { core::ptr::addr_of_mut!(led_lock) }) as u64) as *mut __SlateRecord_spinlock_h_610;
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_563) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _));
unsafe {
(*kb).__slate_bits_0.__set_ledmode(0u8);
}
unsafe {
let __t0 = ((*kb).__slate_bits_1.__get_default_ledflagstate() as i32) as u8;
(*kb).__slate_bits_0.__set_ledflagstate(__t0);
}
// do not do set_leds here because this causes an endless tasklet loop
// when the keyboard hasn't been initialized yet
}

// vt_get_kbd_mode_bit	-	read keyboard status bits
// @console: console to read from
// @bit: mode bit to read
//
// Report back a vt mode bit. We do this without locking so the
// caller must be sure that there are no synchronization needs
#[unsafe(no_mangle)]
extern "C" fn vt_get_kbd_mode_bit(console: u32, bit: i32) -> i32 {
let kb: *mut kbd_struct = unsafe { core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct }.wrapping_offset(console as isize);
vc_kbd_mode(kb, bit)
}

// vt_set_kbd_mode_bit	-	read keyboard status bits
// @console: console to read from
// @bit: mode bit to read
//
// Set a vt mode bit. We do this without locking so the
// caller must be sure that there are no synchronization needs
#[unsafe(no_mangle)]
extern "C" fn vt_set_kbd_mode_bit(console: u32, bit: i32) {
let kb: *mut kbd_struct = unsafe { core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct }.wrapping_offset(console as isize);
let mut __UNIQUE_ID_guard_564: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(kbd_event_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_564) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_565: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(kbd_event_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_565) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
set_vc_kbd_mode(kb, bit);
}

// vt_clr_kbd_mode_bit	-	read keyboard status bits
// @console: console to read from
// @bit: mode bit to read
//
// Report back a vt mode bit. We do this without locking so the
// caller must be sure that there are no synchronization needs
#[unsafe(no_mangle)]
extern "C" fn vt_clr_kbd_mode_bit(console: u32, bit: i32) {
let kb: *mut kbd_struct = unsafe { core::ptr::addr_of_mut!(kbd_table.0) as *mut kbd_struct }.wrapping_offset(console as isize);
let mut __UNIQUE_ID_guard_566: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(kbd_event_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_566) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_567: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!(kbd_event_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_567) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
clr_vc_kbd_mode(kb, bit);
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_check_copy_size_1: [u8; 1] = *b"\0";

static __slate_asm_string_check_copy_size_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/include/linux/ucopysize.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";
