# Shopping Sessions: native Rust pipeline, static binary on an empty image.
FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo
COPY src src
RUN cargo build --release --locked --target x86_64-unknown-linux-gnu \
 && cp target/x86_64-unknown-linux-gnu/release/shop /shop

FROM scratch
COPY --from=build /shop /shop
USER 1000:1000
CMD ["/shop"]
