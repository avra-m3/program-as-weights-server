"""Self-hosted PAW compile server.

A FastAPI implementation of the programasweights.com REST protocol,
backed by the local compile pipeline in `paw_server.compile`. Point the
official SDK at it with:

    PAW_API_URL=http://127.0.0.1:8100 python -c "
    import programasweights as paw
    program = paw.compile('...spec...')
    fn = paw.function(program.id)
    print(fn('...input...'))
    "

The wire contract (endpoints, status codes, bundle layout) was read out
of the installed SDK (`programasweights.client` / `.cache`); see
docs/HOW_IT_WORKS.md §4.
"""
