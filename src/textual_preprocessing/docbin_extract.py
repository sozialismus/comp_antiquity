args = parser.parse_args()

# --- (#11) Input Validation ---
valid_inputs = True
if not validate_input_file(args.mapping_csv, "Mapping CSV"): valid_inputs = False
if not validate_input_file(args.main_index_csv, "Main Index CSV"): valid_inputs = False
if not validate_input_file(args.ner_index_csv, "NER Index CSV"): valid_inputs = False
# Can't fully validate base_dir without knowing expected structure, assume OK if exists
if not os.path.isdir(args.base_dir): logging.warning(f"Base directory '{args.base_dir}' does not exist.") # Warn but don't fail yet
if not validate_output_dir(args.output_dir): valid_inputs = False
# Add conda env validation? Could run `conda env list` but complex. Assume names are correct.
if args.use_locking and not FILELOCK_AVAILABLE:
    logging.warning("--use-locking specified but 'filelock' package not found. Disabling locking.")
    args.use_locking = False

if not valid_inputs:
    script_logger.error("Invalid input paths provided. Please check arguments. Exiting.")
    sys.exit(1)
# --- End Input Validation ---

start_time = time.time()
logger = FileOperationLogger(log_file_path=args.log_file, use_wandb=(not args.no_wandb), wandb_project=args.wandb_project)

script_logger.info(f"Starting corpus reorganization. Overwrite: {args.overwrite}, Workers: {args.num_workers}, Locking: {args.use_locking}")
script_logger.info("-" * 30); [script_logger.info(f"{arg:<20}: {value}") for arg, value in vars(args).items()]; script_logger.info("-" * 30)

if os.path.abspath(args.main_index_csv) == os.path.abspath(args.ner_index_csv):
    logging.warning("="*60); logging.warning("Warn: Main & NER index files same."); logging.warning(" NER may fail/produce 'O' tags if DocBins"); logging.warning(" were not created with NER model."); logging.warning("="*60)

if logger.use_wandb and wandb.run:
     try:
          args_for_config = vars(args).copy(); current_wandb_log_file_actual = wandb.config.get("log_file_actual")
          if "log_file" in args_for_config and current_wandb_log_file_actual is not None: del args_for_config["log_file"]
          wandb.config.update(args_for_config, allow_val_change=True)
     except Exception as e: logging.warning(f"Failed to update wandb config: {e}")

script_logger.info("Loading mappings and indices...")
mappings = parse_csv_mapping(args.mapping_csv)
main_index = load_index(args.main_index_csv)
ner_index = load_index(args.ner_index_csv)
script_logger.info(f"Loaded {len(mappings)} ID mappings."); script_logger.info(f"Loaded {len(main_index)} main index entries."); script_logger.info(f"Loaded {len(ner_index)} NER index entries.")
if not mappings or not main_index: script_logger.error("Mapping/Main Index empty/failed. Exiting."); exit(1)

tasks = []
skipped_count = 0
for old_id, new_id in mappings.items():
    main_docbin_path_check = main_index.get(old_id)
    # (#11) Optionally validate ID format here
    # try: validate_document_id(old_id); validate_document_id(new_id) # If needed
    # except ValueError as e: logging.warning(f"Invalid ID format: {e}. Skipping."); skipped_count+=1; continue

    if not main_docbin_path_check:
         if logger: logger.log_operation(old_id=old_id, new_id=new_id, corpus_prefix=(old_id.split('_')[0] if '_' in old_id else '?'), operation_type="lookup", file_type="main_docbin", status="skipped", details="Not found in main index.")
         skipped_count += 1
    elif not os.path.exists(main_docbin_path_check):
         if logger: logger.log_operation(old_id=old_id, new_id=new_id, corpus_prefix=(old_id.split('_')[0] if '_' in old_id else '?'), operation_type="lookup", file_type="main_docbin", status="skipped", details="Path not found on disk.")
         skipped_count += 1
    else:
        tasks.append({'old_id': old_id, 'new_id': new_id, 'base_dir': args.base_dir, 'output_base_dir': args.output_dir,'main_index': main_index, 'ner_index': ner_index, 'main_env': args.main_env, 'ner_env': args.ner_env,'logger': logger, 'overwrite': args.overwrite, 'use_locking': args.use_locking}) # Pass locking flag

processed_count = len(tasks)
failed_count = 0
script_logger.info(f"Processing {processed_count} documents ({skipped_count} skipped)...")

def process_document_wrapper(task_args):
    try:
        # (#9) Timing decorator already applied to process_document
        return process_document(**task_args)
    except Exception as worker_exc:
        worker_old_id = task_args.get('old_id', 'unknown_worker_task')
        script_logger.error(f"Exception in worker for task {worker_old_id}: {worker_exc}")
        traceback.print_exc()
        return False, f"Failed: Worker Exception {type(worker_exc).__name__}"

executor = None # Define executor in outer scope for shutdown handler
if args.num_workers > 1 and processed_count > 0 :
    script_logger.info(f"Starting parallel processing with {args.num_workers} workers.")
    pbar = tqdm.tqdm(total=processed_count, desc="Processing Docs", unit="doc", dynamic_ncols=True)
    try:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            _executor_ref = executor # (#8) Set global ref for signal handler
            setup_graceful_shutdown() # (#8) Setup handlers AFTER pool starts
            futures = {executor.submit(process_document_wrapper, task): task for task in tasks} # Map future to task
            for future in as_completed(futures):
                task_info = futures[future] # Get corresponding task info
                task_id_str = f"{task_info.get('old_id','?')}->{task_info.get('new_id','?')}"
                if _shutdown_requested: # (#8) Check if shutdown was requested
                     pbar.set_postfix_str(f"Shutdown initiated, waiting...", refresh=True)
                     break # Stop processing new results
                try:
                    success, final_status_msg = future.result()
                    if not success: failed_count += 1
                    pbar.set_postfix_str(f"Last: {task_id_str}, Status: {final_status_msg}", refresh=True)
                except Exception as exc:
                     script_logger.error(f'Task {task_id_str} generated exception: {exc}')
                     traceback.print_exc()
                     failed_count += 1
                     pbar.set_postfix_str(f"Last: {task_id_str}, Status: Exception", refresh=True)
                finally:
                     pbar.update(1)
        _executor_ref = None # Clear ref after pool closes
    except KeyboardInterrupt: # Handle Ctrl+C if it happens before signal handler setup or during wait
        script_logger.warning("KeyboardInterrupt received during parallel execution.")
        if _executor_ref: _executor_ref.shutdown(wait=False, cancel_futures=True) # Try to cancel pending
    finally:
        if pbar: pbar.close() # Ensure pbar is closed

else: # Sequential processing
    script_logger.info("Starting sequential processing.")
    pbar = tqdm.tqdm(tasks, desc="Processing Docs", unit="doc", dynamic_ncols=True)
    setup_graceful_shutdown() # (#8) Setup handlers even for sequential to catch Ctrl+C
    for task in pbar:
        if _shutdown_requested: # (#8) Check flag
             script_logger.warning("Shutdown requested, stopping sequential processing.")
             break
        task_id_str = f"{task.get('old_id','?')}->{task.get('new_id','?')}"
        pbar.set_postfix_str(f"Current: {task_id_str}", refresh=False)
        success, final_status_msg = process_document_wrapper(task)
        if not success: failed_count += 1
        pbar.set_postfix_str(f"Last: {task_id_str}, Status: {final_status_msg}", refresh=True)
    pbar.close()

# Final Summary... (same as before)
end_time = time.time(); duration = end_time - start_time
summary_msg = ("\n" + "-"*30 + "\n--- Reorganization Summary ---" +
               f"\nDocuments in mapping file: {len(mappings)}" + f"\nDocuments skipped (missing main docbin): {skipped_count}" +
               f"\nDocuments attempted processing: {processed_count}" + f"\n-> Successfully processed: {processed_count - failed_count}" +
               f"\n-> Failed processing: {failed_count}" + f"\nTotal time taken: {duration:.2f} seconds ({duration/60:.2f} minutes)" + "\n" + "-"*30)
logging.info(summary_msg); print(summary_msg)
summary_stats = logger.summarize_and_close()
if logger.log_file_path: logging.info(f"Detailed log saved to: {logger.log_file_path}")
if not args.no_wandb:
    if wandb.run and wandb.run.url: logging.info(f"W&B Run URL: {wandb.run.url}")
    else: logging.info("W&B logging was enabled but run may have finished or URL not available.")
logging.info("Script finished."); print("\nScript finished.")
