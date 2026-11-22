# Shopping Sessions: native Rust pipeline, static binary on an empty image.
FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src src
# AVX2 baseline (scorers are Cascade Lake or newer) and a fully static binary
ENV RUSTFLAGS="-C target-cpu=x86-64-v3 -C target-feature=+crt-static"
RUN cargo build --release --locked --target x86_64-unknown-linux-gnu \
 && cp target/x86_64-unknown-linux-gnu/release/shop /shop

FROM scratch
COPY --from=build /shop /shop
USER 1000:1000
CMD ["/shop"]
