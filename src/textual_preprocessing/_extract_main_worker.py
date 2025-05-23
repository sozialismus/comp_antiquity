# _extract_main_worker.py
import argparse
import csv
import json
import logging
import os
import re
import sys
import traceback
from difflib import SequenceMatcher
# Removed unused imports like shutil, subprocess, time, tempfile, signal, glob, pandas, tqdm, wandb, datetime, Any etc.
# Keep only necessary imports for this specific worker
from pathlib import Path # Keep Path if used for file operations
from typing import Dict, List, Optional, Tuple # Keep necessary types

import spacy # Essential
from spacy.tokens import DocBin # Essential

# Configure basic logging within the script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] Main-Worker: %(message)s',
    stream=sys.stderr
)

# --- Alignment Function (copied from original script) ---
def attempt_ner_alignment(tokens, ner_tags):
    # This function runs directly now, so standard f-strings are fine.
    logging.info(f"Entering attempt_ner_alignment. len(tokens)={len(tokens)}, len(ner_tags)={len(ner_tags)}")
    token_texts = [str(t.text) for t in tokens]
    logging.debug(f"Token texts (first 10): {token_texts[:10]}...")
    matcher = SequenceMatcher(None, token_texts, ner_tags, autojunk=False)
    aligned_tags = [None] * len(tokens)
    mismatch_details = []
    alignment_stats = {
        'status': 'failed_no_matches', 'aligned_count': 0,
        'total_tokens': len(tokens), 'total_ner_tags': len(ner_tags),
        'success_rate': 0.0, 'details': mismatch_details
    }
    blocks_processed = 0
    for block in matcher.get_matching_blocks():
        if block.size == 0: continue
        blocks_processed += 1
        token_start, ner_start, size = block.a, block.b, block.size
        logging.debug(f"Match block: token_idx={token_start}, ner_idx={ner_start}, size={size}")
        for i in range(size):
            token_idx = token_start + i; ner_idx = ner_start + i
            if token_idx < len(aligned_tags) and ner_idx < len(ner_tags):
                 aligned_tags[token_idx] = ner_tags[ner_idx]
                 alignment_stats['aligned_count'] += 1
                 if len(mismatch_details) < 20:
                     token_text_safe = token_texts[token_idx] if token_idx < len(token_texts) else "TOKEN_OOB"
                     ner_tag_safe = ner_tags[ner_idx]
                     # Standard f-string is now correct here
                     mismatch_details.append(f"Align: tok[{token_idx}]='{token_text_safe}' <-> tag[{ner_idx}]='{ner_tag_safe}'")
            else:
                 logging.warning(f"Alignment index out of bounds: token_idx={token_idx} (max={len(aligned_tags)-1}), ner_idx={ner_idx} (max={len(ner_tags)-1})")

    logging.info(f"Alignment processed {blocks_processed} matching blocks. Total aligned: {alignment_stats['aligned_count']}")
    if alignment_stats['aligned_count'] > 0:
         alignment_stats['success_rate'] = alignment_stats['aligned_count'] / len(tokens) if len(tokens) > 0 else 0
         alignment_stats['status'] = 'success' if alignment_stats['aligned_count'] == len(tokens) else 'partial'
         if alignment_stats['status'] == 'partial':
             unaligned_tokens = len(tokens) - alignment_stats['aligned_count']
             logging.info(f"{unaligned_tokens} tokens could not be aligned.")

    # Define success_threshold locally (this was likely the missing definition causing the NameError)
    success_threshold = 0.5
    if alignment_stats['success_rate'] < success_threshold and alignment_stats['status'] != 'success':
        # Standard f-string is now correct here
        logging.warning(f"Alignment quality low ({alignment_stats['success_rate']:.1%} < {success_threshold*100:.0f}%). Discarding alignment results.")
        alignment_stats['status'] = 'failed_low_quality'
        # Standard f-string is now correct here
        mismatch_details.append(f"Failed: Alignment rate below threshold ({success_threshold:.1%})")
        return None, alignment_stats
    # Standard f-string is now correct here
    logging.info(f"Alignment Result: Status={alignment_stats['status']}, Rate={alignment_stats['success_rate']:.2%} (Aligned={alignment_stats['aligned_count']}/{len(tokens)})")
    return aligned_tags, alignment_stats
# --- End Alignment Function ---


def extract_main_data(
    docbin_path: str,
    output_txt_joined: str,
    output_txt_fullstop: str,
    output_csv_lemma: str,
    output_csv_upos: str,
    output_csv_stop: str,
    output_csv_dot: str,
    output_conllu: str,
    ner_tags_path: Optional[str], # Optional should be imported from typing
    doc_id_str: str,
    main_model_name: str = 'grc_proiel_trf'
):
    """Loads main model, processes docbin, integrates NER, writes outputs."""
    logging.info(f"Starting main extraction for doc ID: {doc_id_str}")
    logging.info(f"Main DocBin Path: {docbin_path}")
    logging.info(f"NER Tags Path (Input): {ner_tags_path}")
    logging.info(f"Main Model: {main_model_name}")

    dirs_to_check=set([os.path.dirname(p) for p in [
        output_txt_joined, output_txt_fullstop, output_csv_lemma, output_csv_upos,
        output_csv_stop, output_csv_dot, output_conllu
    ] if p])
    for d in dirs_to_check:
        if d:
            try: os.makedirs(d, exist_ok=True); logging.info(f"Ensured directory exists: {d}")
            except OSError as e: logging.error(f"Failed to create directory {d}: {e}"); sys.exit(1)

    try:
        logging.info(f"Loading spaCy model '{main_model_name}'...")
        nlp = spacy.load(main_model_name);
        logging.info(f"Loading DocBin from {docbin_path}...")
        doc_bin = DocBin().from_disk(docbin_path); docs = list(doc_bin.get_docs(nlp.vocab));
        if not docs: logging.error(f"DocBin file is empty or failed to load: {docbin_path}"); sys.exit(1)
        doc = docs[0]; num_tokens = len(doc)
        logging.info(f"Successfully loaded doc with {num_tokens} tokens.")

        doc_text = doc.text
        logging.info(f"Writing joined text to {output_txt_joined}...")
        with open(output_txt_joined,'w',encoding='utf-8') as f: f.write(doc_text)

        logging.info(f"Processing and writing fullstop text to {output_txt_fullstop}...")
        try:
            # Using original complex regex patterns - ensure they are correct for the task
            tfs = re.sub(r'\\.(?!\\.)', '.\n', doc_text); # Replace literal \. not followed by \. with .\n
            tfs = re.sub(r'\s+\n','\n', tfs);             # Replace whitespace before newline with newline
            tfs = re.sub(r'\n\s+', '\n', tfs).strip()    # Replace newline followed by whitespace with newline
            with open(output_txt_fullstop,'w',encoding='utf-8') as f: f.write(tfs)
        except Exception as fs_e: logging.warning(f"Failed to generate fullstop format: {fs_e}", exc_info=True)

        logging.info("Writing CSV annotation files...")
        try:
            with open(output_csv_lemma,'w',encoding='utf-8',newline='') as fl, \
                 open(output_csv_upos,'w',encoding='utf-8',newline='') as fu, \
                 open(output_csv_stop,'w',encoding='utf-8',newline='') as fs, \
                 open(output_csv_dot,'w',encoding='utf-8',newline='') as fd:
                wl=csv.writer(fl,quoting=csv.QUOTE_ALL); wl.writerow(['ID','TOKEN','LEMMA']);
                wu=csv.writer(fu,quoting=csv.QUOTE_ALL); wu.writerow(['ID','TOKEN','UPOS']);
                ws=csv.writer(fs,quoting=csv.QUOTE_ALL); ws.writerow(['ID','TOKEN','IS_STOP']);
                wd=csv.writer(fd,quoting=csv.QUOTE_ALL); wd.writerow(['ID','TOKEN','IS_PUNCT'])
                for i,t in enumerate(doc):
                    tid,ttxt=i+1,str(t.text);
                    wl.writerow([tid,ttxt,t.lemma_]); wu.writerow([tid,ttxt,t.pos_]);
                    ws.writerow([tid,ttxt,'TRUE' if t.is_stop else 'FALSE']); wd.writerow([tid,ttxt,'TRUE' if t.is_punct else 'FALSE'])
            logging.info("Finished writing CSV files.")
        except Exception as csv_e: logging.error(f"Failed during CSV writing: {csv_e}", exc_info=True); sys.exit(1)

        original_ner_tags = None; ner_tags_to_use = None; alignment_info = None; mismatch_detected = False
        num_ner_tags = 0; mismatch_data_for_json = {}
        logging.info(f"Checking for NER tags input file: {ner_tags_path}")
        if ner_tags_path and os.path.exists(ner_tags_path):
            try:
                logging.info(f"Reading NER tags from {ner_tags_path}")
                with open(ner_tags_path,'r',encoding='utf-8') as fn: original_ner_tags=[ln.strip() for ln in fn if ln.strip()]
                num_ner_tags = len(original_ner_tags); logging.info(f"Read {num_ner_tags} NER tags.")
                mismatch_data_for_json = {"document_id": doc_id_str,"main_model_tokens": num_tokens,"ner_model_tags": num_ner_tags,"mismatch_detected": False,"alignment_info": None}
                if num_ner_tags != num_tokens:
                    mismatch_detected = True; mismatch_data_for_json["mismatch_detected"] = True
                    logging.warning(f"Token count mismatch! Main model: {num_tokens}, NER tags: {num_ner_tags}. Attempting alignment.")
                    aligned_result, alignment_info = attempt_ner_alignment(doc, original_ner_tags)
                    mismatch_data_for_json["alignment_info"] = alignment_info
                    if aligned_result: ner_tags_to_use = aligned_result; logging.info("Using ALIGNED NER tags for CoNLL-U.")
                    else: ner_tags_to_use = None; logging.warning("Alignment failed or discarded. NER tags will be OMITTED from CoNLL-U.")
                else:
                    ner_tags_to_use = original_ner_tags; logging.info("Token counts match. Using original NER tags."); mismatch_data_for_json = None
            except Exception as e:
                logging.warning(f"Failed to read or align NER tags: {e}.", exc_info=True); ner_tags_to_use=None
                if mismatch_data_for_json: mismatch_data_for_json["error_during_ner_processing"] = str(e)
                else: mismatch_data_for_json = {"document_id": doc_id_str, "error_message": f"Error processing NER tags: {e}"}
        elif ner_tags_path:
            logging.warning(f"NER tags path provided but not found: {ner_tags_path}")
            mismatch_data_for_json = { "document_id": doc_id_str, "error_message": "NER tags path not found", "path_checked": ner_tags_path }
            ner_tags_to_use = None
        else:
            logging.info("No NER tags path provided. Skipping NER integration."); mismatch_data_for_json = None

        logging.info(f"Writing CoNLL-U file to {output_conllu}...")
        try:
            with open(output_conllu,"w",encoding="utf-8") as fo:
                fo.write(f"# newdoc id = {doc_id_str}\n")
                if mismatch_detected and mismatch_data_for_json and mismatch_data_for_json.get("mismatch_detected"):
                    fo.write(f"# ner_token_mismatch = True\n"); fo.write(f"# main_model_tokens = {num_tokens}\n"); fo.write(f"# ner_model_tags = {num_ner_tags}\n")
                    if alignment_info:
                        fo.write(f"# ner_alignment_status = {alignment_info.get('status','?')}\n"); fo.write(f"# ner_alignment_rate = {alignment_info.get('success_rate', 0.0):.4f}\n")
                        if alignment_info.get('status') in ('partial', 'failed_low_quality') and alignment_info.get('details'):
                             details_preview = "; ".join(alignment_info['details'][:3]); fo.write(f"# ner_alignment_details_preview = {details_preview}\n")
                sidc = 1
                for sent_idx, sent in enumerate(doc.sents):
                    stc=str(sent.text).replace('\n',' ').replace('\r','').strip()
                    fo.write(f"\n# sent_id = {doc_id_str}-{sidc}\n"); fo.write(f"# text = {stc}\n")
                    tsic = 1
                    for token_idx_in_doc, t in enumerate(sent):
                        head_idx_in_doc = t.head.i; head_idx_in_sent = head_idx_in_doc - sent.start + 1 if head_idx_in_doc != t.i else 0
                        fts = str(t.morph) if t.morph else "_"
                        mp = []
                        if ner_tags_to_use:
                            abs_token_index = t.i
                            if abs_token_index < len(ner_tags_to_use):
                                nt = ner_tags_to_use[abs_token_index];
                                if nt and nt != 'O': mp.append(f"NER={nt}")
                            else: logging.warning(f"Token index {abs_token_index} out of bounds for ner_tags_to_use (len={len(ner_tags_to_use)}) in sentence {sidc}")
                        if (t.i + 1) < len(doc) and doc[t.i + 1].idx == (t.idx + len(t.text)): mp.append("SpaceAfter=No")
                        mf = "|".join(mp) if mp else "_"
                        dpr = str(t.dep_).strip() if t.dep_ else "dep"
                        if not dpr: dpr = "dep"
                        dpf = "_" # Assign default value
                        cols = [str(tsic),str(t.text),str(t.lemma_),str(t.pos_),str(t.tag_),fts,str(head_idx_in_sent),dpr,dpf,mf]
                        fo.write("\t".join(cols) + "\n"); tsic += 1
                    sidc += 1
            logging.info("Finished writing CoNLL-U file.")
        except Exception as conllu_e: logging.error(f"Failed during CoNLL-U writing: {conllu_e}", exc_info=True); sys.exit(1)

        if mismatch_data_for_json:
            json_filename = f"{doc_id_str}_ner_mismatch_info.json" if mismatch_data_for_json.get('mismatch_detected') else f"{doc_id_str}_ner_error_info.json"
            json_output_path = os.path.join(os.path.dirname(output_conllu), json_filename)
            logging.info(f"Writing NER processing info to {json_output_path}...")
            try:
                with open(json_output_path, 'w', encoding='utf-8') as fm: json.dump(mismatch_data_for_json, fm, indent=2)
                logging.info("Finished writing NER info JSON.")
            except Exception as e: logging.error(f"Failed to write NER info JSON file: {e}", exc_info=True)

    except Exception as e:
        logging.error(f"Unhandled exception in main worker script: {e}", exc_info=True)
        sys.exit(1)

    logging.info(f"Main extraction script finished successfully for doc ID: {doc_id_str}.")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Model Worker Script")
    parser.add_argument("--docbin-path", required=True, help="Path to input DocBin file.")
    parser.add_argument("--output-txt-joined", required=True)
    parser.add_argument("--output-txt-fullstop", required=True)
    parser.add_argument("--output-csv-lemma", required=True)
    parser.add_argument("--output-csv-upos", required=True)
    parser.add_argument("--output-csv-stop", required=True)
    parser.add_argument("--output-csv-dot", required=True)
    parser.add_argument("--output-conllu", required=True)
    parser.add_argument("--ner-tags-path", default=None, help="Optional path to NER tags text file.")
    parser.add_argument("--doc-id", required=True, help="New document ID (Sort ID).")
    parser.add_argument("--main-model-name", default='grc_proiel_trf', help="Name of main spaCy model.")

    args = parser.parse_args()

    extract_main_data(
        docbin_path=args.docbin_path,
        output_txt_joined=args.output_txt_joined,
        output_txt_fullstop=args.output_txt_fullstop,
        output_csv_lemma=args.output_csv_lemma,
        output_csv_upos=args.output_csv_upos,
        output_csv_stop=args.output_csv_stop,
        output_csv_dot=args.output_csv_dot,
        output_conllu=args.output_conllu,
        ner_tags_path=args.ner_tags_path,
        doc_id_str=args.doc_id,
        main_model_name=args.main_model_name
    )
