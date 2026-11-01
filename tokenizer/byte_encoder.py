
def bytes_to_unicode():
    """
    Create a reversible mapping:
        byte value (0-255) <-> Unicode character
    """

    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )

    unicode_values = byte_values.copy()
    extra = 0

    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_values.append(256 + extra)
            extra += 1

    unicode_chars = [chr(value) for value in unicode_values]

    encoder = dict(zip(byte_values, unicode_chars))
    decoder = {
        character: byte
        for byte, character in encoder.items()
    }

    return encoder, decoder
