from __future__ import annotations

import argparse
import concurrent.futures
import os
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
        stale: list[tuple[dict, int | None, str | None]] = []
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

        if not getattr(args, "yes", False):
            try:
                ans = input(f"\nRemove {len(stale)} stale bookmarks from index? [y/N] ").strip().lower()
                if ans != "y":
                    print("Skipping removal.")
                    return 0
            except (EOFError, OSError):
                return 0

        removed = idx.remove_urls([bm["url"] for bm, _, _ in stale])
        print(f"Successfully removed {removed} stale bookmarks from index.")
        return 0
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return 1
    finally:
        idx.close()


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
    results = idx.search(
        query=args.query, k=args.top,
        domain=args.domain, folder=args.folder,
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

    return 0


def _cmd_stats(args):
    idx = Index(db_path=args.db)
    try:
        stats = idx.stats()
        print(f"bookmarks: {stats['count']}")
        if stats['count'] > 0:
            print(f"model:     {stats['model']}")
            print(f"dimension: {stats['dim']}")
        return 0
    finally:
        idx.close()


def _cmd_sync(args):
    from .browsers import collect_all_bookmarks, detect_browsers
    
    browsers = detect_browsers()
    if not browsers:
        print("error: no browsers detected", file=sys.stderr)
        return 1
        
    print(f"[1/2] collecting bookmarks from {', '.join(b.name for b in browsers)}")
    bookmarks = collect_all_bookmarks(browsers)
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
        "--validate",
        action="store_true",
        help="validate indexed bookmark URLs and optionally trim stale entries",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="per-request timeout in seconds for --validate (default: 8.0)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="parallel request workers for --validate (default: 16)",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="auto-confirm trimming stale bookmarks during --validate",
    )
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
    pf.set_defaults(func=_cmd_find)

    ps = sub.add_parser("stats", help="show index stats")
    ps.set_defaults(func=_cmd_stats)

    py = sub.add_parser("sync", help="automatically sync bookmarks from local browsers")
    py.add_argument("--model", default=DEFAULT_MODEL)
    py.set_defaults(func=_cmd_sync)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.validate:
        if args.cmd is not None:
            parser.error("--validate cannot be combined with subcommands")
        if args.timeout <= 0:
            parser.error("--timeout must be > 0")
        if args.workers <= 0:
            parser.error("--workers must be > 0")
        return _cmd_validate(args)
    if args.cmd is None:
        parser.print_help()
        return 2
    return args.func(args)
