"""Linha de comando do Margem de Segurança.

  ms rodar BBAS3 [WEGE3 ...]     rodada completa (um ativo ou lote)
  ms abrir BBAS3                 atualização rápida e abre a tela do ativo (sem rodar o modelo)
  ms ranking-dos-ativos          atualização rápida e abre a página do ranking dos ativos
  ms refazer desatualizados      mostra a lista e roda de novo, como lote, os desatualizados
  ms refazer todos               idem, para todos
  ms tirar BBAS3                 tira o ativo do ranking dos ativos (sem apagar nada)
  ms teste-vazamento BBAS3       teste de vazamento em meses sorteados
  ms retomar                     retoma um lote que parou no meio
Use --sem-abrir para não abrir o navegador.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser

from . import AVISO, caminhos, formatos


def _abrir(caminho, sem_abrir: bool) -> None:
    print(f"  página: {caminho}")
    if sem_abrir or os.environ.get("MS_SEM_ABRIR"):
        return
    try:
        if sys.platform == "win32":
            os.startfile(str(caminho))  # type: ignore[attr-defined]
        else:
            webbrowser.open(caminho.as_uri())
    except Exception:
        pass


def cmd_rodar(args) -> int:
    from . import rodada

    tickers = [t.upper() for t in args.tickers]
    if len(tickers) == 1:
        try:
            r = rodada.rodar_ativo(tickers[0])
        except rodada.SetorForaDoMapa as e:
            print(f"Precisa do seu OK: {e}")
            return 2
        a = r["analise"]
        if a["estimativa"]:
            print(f"{a['ticker']}: preço justo {formatos.moeda(a['preco_justo'])} "
                  f"(faixa {formatos.moeda(a['p10'])} a {formatos.moeda(a['p90'])}), preço {formatos.moeda(a['preco'])}.")
        else:
            print(f"{a['ticker']}: sem estimativa confiável. Motivos: {a['motivo_sem_estimativa']}.")
        print(f"Posição: {r['posicao']}.")
        print(f"  relatório: {r['relatorio']}")
        _abrir(r["tela"], args.sem_abrir)
    else:
        out = rodada.rodar_lote(tickers)
        from . import ranking_dos_ativos
        res = ranking_dos_ativos.atualizar(progresso=None)
        print(rodada.resumo_ranking(res))
        if out["falhas"]:
            print("Ativos que falharam ou precisam do seu OK:")
            for t, m in out["falhas"].items():
                print(f"  {t}: {m}")
        _abrir(caminhos.pagina_ranking_dos_ativos(), args.sem_abrir)
    print(AVISO)
    return 0


def cmd_abrir(args) -> int:
    from . import historico, ranking_dos_ativos

    t = args.ticker.upper()
    if t not in historico.mais_recentes():
        print(f"{t} nunca foi rodado no Margem de Segurança. Rode com: ms rodar {t}")
        return 3
    ranking_dos_ativos.atualizar()
    _abrir(caminhos.pasta_ativos() / f"{t}.html", args.sem_abrir)
    return 0


def cmd_ranking(args) -> int:
    from . import ranking_dos_ativos, rodada

    r = ranking_dos_ativos.atualizar()
    print(rodada.resumo_ranking(r))
    _abrir(caminhos.pagina_ranking_dos_ativos(), args.sem_abrir)
    print(AVISO)
    return 0


def cmd_refazer(args) -> int:
    from . import historico, ranking_dos_ativos, rodada

    r = ranking_dos_ativos.atualizar(progresso=None)
    todos = r["linhas"] + r["sem"]
    if args.quais == "todos":
        lista = sorted(x["ticker"] for x in todos)
    else:
        lista = sorted(x["ticker"] for x in todos if x["desatualizada"])
    if not lista:
        print("Nenhum ativo para refazer.")
        return 0
    print("Ativos a refazer: " + ", ".join(lista))
    if args.so_listar:
        return 0
    out = rodada.rodar_lote(lista)
    res = ranking_dos_ativos.atualizar(progresso=None)
    print(rodada.resumo_ranking(res))
    for t, m in out["falhas"].items():
        print(f"  falhou {t}: {m}")
    _abrir(caminhos.pagina_ranking_dos_ativos(), args.sem_abrir)
    return 0


def cmd_tirar(args) -> int:
    from . import historico, ranking_dos_ativos

    t = args.ticker.upper()
    historico.registrar_fora(t, "saida", "pedido do usuário")
    ranking_dos_ativos.atualizar(progresso=None)
    print(f"{t} saiu do ranking dos ativos (histórico, relatórios e tela mantidos). Rodar o ativo de novo o traz de volta.")
    return 0


def cmd_vazamento(args) -> int:
    from . import vazamento

    res = vazamento.testar(args.ticker, meses=args.meses, semente=args.semente)
    ok = True
    for r in res:
        ok &= r["passou"]
        print(f"  {r['mes']:%m/%Y}: guardado {formatos.moeda(r['guardado'])} · refeito {formatos.moeda(r['refeito'])} · "
              f"diferença {formatos.num(r['diferenca'], 6)} · {'passou' if r['passou'] else 'FALHOU'}")
    print("Teste de vazamento: " + ("passou" if ok else "FALHOU"))
    return 0 if ok else 1


def cmd_retomar(args) -> int:
    arq = caminhos.arquivo_lote()
    if not arq.exists():
        print("Nenhum lote interrompido.")
        return 0
    estado = json.loads(arq.read_text(encoding="utf-8"))
    args.tickers = estado["tickers"]
    return cmd_rodar(args)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ms", description="Margem de Segurança")
    p.add_argument("--sem-abrir", action="store_true", help="não abrir o navegador")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("rodar"); s.add_argument("tickers", nargs="+"); s.set_defaults(f=cmd_rodar)
    s = sub.add_parser("abrir"); s.add_argument("ticker"); s.set_defaults(f=cmd_abrir)
    s = sub.add_parser("ranking-dos-ativos"); s.set_defaults(f=cmd_ranking)
    s = sub.add_parser("refazer"); s.add_argument("quais", choices=["desatualizados", "todos"])
    s.add_argument("--so-listar", action="store_true"); s.set_defaults(f=cmd_refazer)
    s = sub.add_parser("tirar"); s.add_argument("ticker"); s.set_defaults(f=cmd_tirar)
    s = sub.add_parser("teste-vazamento"); s.add_argument("ticker"); s.add_argument("--meses", type=int, default=3)
    s.add_argument("--semente", type=int, default=2026); s.set_defaults(f=cmd_vazamento)
    s = sub.add_parser("retomar"); s.set_defaults(f=cmd_retomar)
    args = p.parse_args(argv)
    return args.f(args)


if __name__ == "__main__":
    sys.exit(main())
