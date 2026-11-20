export default {
  fetch(request, env) {
    const { pathname } = new URL(request.url);
    if (pathname === '/smoke' || pathname.startsWith('/smoke/')) {
      return env.ASSETS.fetch(request);
    }
    return fetch(request);
  },
};
