from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import sqlite3
import sys
import webbrowser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from . import __version__
from .parser import parse_file
from .index import Index, SyncResult, default_db_path, DEFAULT_MODEL


def _is_http_url(url: str) -> bool:
    p = urlparse(url)
    return p.scheme.lower() in {"http", "https"} and bool(p.netloc)


def _check_url_status(url: str, timeout: float) -> tuple[str, int | None, str | None]:
    """Return (url, status_code, error_message)."""
    if not _is_http_url(url):
        return url, None, "skipped (non-http URL)"

    headers = {"User-Agent": "mindmark/0.x (+bookmark-validation)"}
    try:
        req = Request(url, headers=headers, method="HEAD")
        with urlopen(req, timeout=timeout) as resp:
            return url, int(getattr(resp, "status", 0) or 0), None
    except HTTPError as e:
        # HTTP errors still include a useful status code.
        return url, int(e.code), str(e.reason) if e.reason else "HTTP error"
    except Exception:
        pass

    # Fallback to GET for servers that reject HEAD.
    try:
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            return url, int(getattr(resp, "status", 0) or 0), None
    except HTTPError as e:
        return url, int(e.code), str(e.reason) if e.reason else "HTTP error"
    except URLError as e:
        return url, None, str(e.reason) if e.reason else "connection error"
    except Exception as e:  # pragma: no cover - defensive fallback
        return url, None, str(e)


def _cmd_validate(args):
    idx = Index(db_path=args.db)
    try:
        bookmarks = idx.all_bookmarks()
        if not bookmarks:
            print("index is empty — run 'mindmark sync' first.")
            return 1

        total = len(bookmarks)
        print(f"validating {total} indexed bookmarks...")

        url_to_bm = {b["url"]: b for b in bookmarks}
        stale = []
        skipped = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(_check_url_status, b["url"], args.timeout): b["url"]
                for b in bookmarks
            }
            for fut in concurrent.futures.as_completed(futs):
                url, code, error = fut.result()
                if error == "skipped (non-http URL)":
                    skipped += 1
                    continue
                if code is None or code >= 400:
                    stale.append((url_to_bm[url], code, error))

        checked = total - skipped
        healthy = checked - len(stale)

        print(
            f"checked={checked} healthy={healthy} stale={len(stale)} skipped={skipped}"
        )

        if not stale:
            print("all checked bookmarks look valid.")
            return 0

        print("\nstale bookmarks found:")
        for i, (bm, code, error) in enumerate(stale, 1):
            reason = f"HTTP {code}" if code is not None else (error or "unreachable")
            folder = bm["folder_path"] or "(root)"
            print(f"\n{i}. {bm['title']}")
            print(f"   status: {reason}")
            print(f"   url:    {bm['url']}")
            print(f"   path:   {folder}")

        return 0
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return 1
    finally:
        idx.close()


def _cmd_drop_index(args):
    db_path = Path(args.db).expanduser() if args.db else default_db_path()

    if not db_path.exists():
        print(f"index not found: {db_path}")
        return 0

    if not args.yes:
        try:
            ans = input(f"drop local index at '{db_path}'? [y/N] ").strip().lower()
            if ans != "y":
                print("cancelled.")
                return 0
        except (EOFError, OSError):
            print("cancelled.")
            return 0

    try:
        if db_path.is_file():
            db_path.unlink()
        elif db_path.is_dir():
            shutil.rmtree(db_path)
        else:
            print(f"index path is not a file or directory: {db_path}")
            return 1
    except PermissionError as e:
        # Windows can keep SQLite files locked by another process handle.
        # If deletion fails, try clearing index data in-place as a fallback.
        if db_path.is_file() and _clear_index_contents(db_path):
            print(f"index file is in use; cleared index contents instead: {db_path}")
            return 0
        print(f"error: failed to remove index: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: failed to remove index: {e}", file=sys.stderr)
        return 1

    print(f"dropped local index: {db_path}")
    return 0


def _clear_index_contents(db_path: Path) -> bool:
    """Best-effort fallback when index file cannot be deleted due to locks."""
    try:
        con = sqlite3.connect(str(db_path), timeout=1.0)
        cur = con.cursor()
        cur.execute("DELETE FROM bookmark_enrichment")
        cur.execute("DELETE FROM bookmark_sources")
        cur.execute("DELETE FROM bookmarks")
        cur.execute("DELETE FROM meta")
        con.commit()
        con.close()
        return True
    except sqlite3.Error:
        return False


def _cmd_index(args):
    path = Path(args.path).expanduser()
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2
    print(f"[1/3] parsing {path}")
    bookmarks = parse_file(str(path))
    print(f"      parsed {len(bookmarks)} unique bookmarks")
    print(f"[2/3] loading embedding model ({args.model})")
    idx = Index(db_path=args.db, model_name=args.model)
    print(f"[3/3] embedding + writing index to {idx.db_path}")
    info = idx.rebuild(bookmarks, batch_size=args.batch_size)
    print(f"done. indexed={info['indexed']} dim={info.get('dim','?')} model={info['model']}")
    return 0


def _auto_sync_hint(idx: Index) -> None:
    """Print a hint when the index is empty."""
    if not idx.is_empty():
        return
    print("index is empty — run 'mindmark sync' to import bookmarks from your browsers,")
    print("or run 'mindmark index <bookmarks.html>' to import from an exported file.")
    print()


def _cmd_find(args):
    idx = Index(db_path=args.db)
    if not getattr(args, 'json', False):
        _auto_sync_hint(idx)
    include_excerpt = getattr(args, 'excerpt', False)
    results = idx.search(
        query=args.query, k=args.top,
        domain=args.domain, folder=args.folder,
        include_excerpt=include_excerpt,
    )
    if not results:
        print("no results (is the index empty? run: mindmark sync)")
        return 1

    if args.open is not None:
        n = args.open - 1
        if not 0 <= n < len(results):
            print(f"error: --open {args.open} out of range (1..{len(results)})", file=sys.stderr)
            return 2
        webbrowser.open(results[n]["url"])
        print(f"opened: {results[n]['title']}")
        return 0

    import json
    if getattr(args, "json", False):
        print(json.dumps(results, indent=2))
    else:
        for i, r in enumerate(results, 1):
            domain = urlparse(r["url"]).netloc
            folder = r["folder_path"]
            path = f"{folder}/" if folder else ""
            print(f"{i:2d}. {r['title']}")
            print(f"    {path}{domain}")
            if include_excerpt and r.get("relevant_excerpt"):
                excerpt = r["relevant_excerpt"]
                print(f"    ⤵ {excerpt}")

    return 0


def _cmd_stats(args):
    idx = Index(db_path=args.db)
    try:
        stats = idx.stats()
        print(f"bookmarks: {stats['total']}")
        if stats['total'] > 0:
            print(f"model:     {stats['model']}")
            if stats['top_domains']:
                print(f"\ntop domains:")
                for domain, count in stats['top_domains']:
                    print(f"  {domain}: {count}")
            if stats['top_folders']:
                print(f"\ntop folders:")
                for folder, count in stats['top_folders']:
                    print(f"  {folder}: {count}")
        return 0
    finally:
        idx.close()


def _cmd_enrich(args):
    from .enricher import enrich_pending

    idx = Index(db_path=args.db)
    try:
        pending = idx.pending_enrichment_urls(
            limit=None if args.refresh_failed else args.limit
        )
        if args.refresh_failed:
            reset = idx.reset_failed_enrichment()
            if reset:
                print(f"reset {reset} failed enrichment rows to pending")
            # re-query after reset, respecting --limit
            pending = idx.pending_enrichment_urls(limit=args.limit)

        estats = idx.enrichment_stats()
        total_pending = estats.get("pending", 0)

        if not pending:
            print("nothing to enrich — run 'mindmark sync' first, or use --refresh-failed")
            return 0

        to_process = len(pending)
        print(
            f"enriching {to_process} bookmarks "
            f"(pending={total_pending} workers={args.workers} timeout={args.timeout}s)"
        )

        result = enrich_pending(
            idx,
            limit=args.limit,
            workers=args.workers,
            timeout=args.timeout,
            refresh_failed=False,  # already handled above
        )
        print(f"done. {result}")
        return 0
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return 1
    finally:
        idx.close()


def _cmd_sync(args):
    from .browsers import parse_browser_bookmarks, detect_browsers
    
    browsers = detect_browsers()
    if not browsers:
        print("error: no browsers detected", file=sys.stderr)
        return 1
        
    print(f"[1/2] collecting bookmarks from {', '.join(b.browser_name for b in browsers)}")
    bookmarks = []; [bookmarks.extend(parse_browser_bookmarks(b)) for b in browsers]
    if not bookmarks:
        print("no bookmarks found.")
        return 0
    print(f"      found {len(bookmarks)} unique bookmarks")
    
    print(f"[2/2] syncing to {args.db or default_db_path()}")
    idx = Index(db_path=args.db, model_name=args.model)
    res = idx.sync(bookmarks)
    
    print(f"done. added={res.added} updated={res.updated} removed={res.removed}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="mindmark",
        description="mindmark — local semantic search over your browser bookmarks.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument(
        "--db", default=os.environ.get("MINDMARK_DB"),
        help=f"SQLite index path (default: {default_db_path()})",
    )

    sub = p.add_subparsers(dest="cmd")

    pi = sub.add_parser("index", help="build/refresh the index from an exported bookmarks HTML file")
    pi.add_argument("path", help="path to the exported Netscape bookmarks HTML file")
    pi.add_argument("--model", default=DEFAULT_MODEL)
    pi.add_argument("--batch-size", type=int, default=64)
    pi.set_defaults(func=_cmd_index)

    pf = sub.add_parser("find", help="search bookmarks by natural-language query")
    pf.add_argument("query")
    pf.add_argument("-k", "--top", type=int, default=10)
    pf.add_argument("--domain")
    pf.add_argument("--folder")
    pf.add_argument("--json", action="store_true")
    pf.add_argument("--open", type=int, metavar="N")
    pf.add_argument(
        "--excerpt", action="store_true",
        help="include excerpt from enriched page content (requires mindmark enrich)",
    )
    pf.set_defaults(func=_cmd_find)

    ps = sub.add_parser("stats", help="show index stats")
    ps.set_defaults(func=_cmd_stats)

    py = sub.add_parser("sync", help="automatically sync bookmarks from local browsers")
    py.add_argument("--model", default=DEFAULT_MODEL)
    py.set_defaults(func=_cmd_sync)

    pv = sub.add_parser("validate", help="validate indexed bookmark URLs and report stale entries (read-only)")
    pv.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="per-request timeout in seconds (default: 8.0)",
    )
    pv.add_argument(
        "--workers",
        type=int,
        default=16,
        help="parallel request workers (default: 16)",
    )
    pv.set_defaults(func=_cmd_validate)

    pd = sub.add_parser("drop-index", help="drop (delete) the local index database")
    pd.add_argument(
        "--yes",
        action="store_true",
        help="auto-confirm index deletion",
    )
    pd.set_defaults(func=_cmd_drop_index)

    pe = sub.add_parser(
        "enrich",
        help="fetch page content for bookmarks and build summary embeddings (local, no cloud)",
    )
    pe.add_argument(
        "--limit", type=int, default=None,
        help="max bookmarks to process per run (default: all pending)",
    )
    pe.add_argument(
        "--workers", type=int, default=8,
        help="parallel fetch workers (default: 8)",
    )
    pe.add_argument(
        "--timeout", type=float, default=10.0,
        help="per-request fetch timeout in seconds (default: 10.0)",
    )
    pe.add_argument(
        "--refresh-failed", action="store_true",
        help="retry previously failed enrichments",
    )
    pe.set_defaults(func=_cmd_enrich)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "validate":
        if args.timeout <= 0:
            parser.error("--timeout must be > 0")
        if args.workers <= 0:
            parser.error("--workers must be > 0")
        return args.func(args)
    if args.cmd == "enrich":
        if args.workers <= 0:
            parser.error("--workers must be > 0")
        if args.timeout <= 0:
            parser.error("--timeout must be > 0")
        return args.func(args)
    if args.cmd is None:
        parser.print_help()
        return 2
    return args.func(args)
