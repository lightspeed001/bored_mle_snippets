library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- SRAM-backed MAC controller (teaching sketch)
-- Toy model only: one MAC lane, serialized channels and output filters.
-- Real accelerators use many parallel lanes, banked SRAM, and richer scheduling.
entity toy_conv1x1_engine is
  generic (
    K_CH : positive := 64;   -- input channels
    N_CH : positive := 32;   -- output channels / filters
    ACC_W : positive := 32
  );
  port (
    clk, rst_n : in std_logic;
    start      : in std_logic;
    done       : out std_logic;
    x_data     : in signed(15 downto 0); -- value supplied by input SRAM
    w_data     : in signed(15 downto 0); -- value supplied by weight SRAM
    y_data     : out signed(ACC_W-1 downto 0);
    x_addr     : out unsigned(15 downto 0);
    w_addr     : out unsigned(15 downto 0)
  );
end entity;

architecture rtl of toy_conv1x1_engine is
  type state_t is (IDLE, MAC, WRITE_OUT);
  signal state : state_t := IDLE;
  signal k_idx : natural range 0 to K_CH-1 := 0;
  signal n_idx : natural range 0 to N_CH-1 := 0;
  signal acc   : signed(ACC_W-1 downto 0) := (others => '0');
begin
  -- Addresses reflect W[n,k] and one spatial site's X[k]. A real design
  -- would pipeline SRAM latency and process multiple sites/channels in parallel.
  x_addr <= to_unsigned(k_idx, x_addr'length);
  w_addr <= to_unsigned(n_idx * K_CH + k_idx, w_addr'length);
  y_data <= acc;
  done <= '1' when state = IDLE else '0';

  process(clk)
    variable product : signed(31 downto 0);
  begin
    if rising_edge(clk) then
      if rst_n = '0' then
        state <= IDLE; k_idx <= 0; n_idx <= 0; acc <= (others => '0');
      else
        case state is
          when IDLE =>
            if start = '1' then
              k_idx <= 0; n_idx <= 0; acc <= (others => '0'); state <= MAC;
            end if;
          when MAC =>
            -- Signed 16x16 multiply, accumulate into a wider output sum.
            product := x_data * w_data;
            acc <= acc + resize(product, ACC_W);
            if k_idx = K_CH-1 then state <= WRITE_OUT;
            else k_idx <= k_idx + 1;
            end if;
          when WRITE_OUT =>
            -- The outer controller would write this result and advance n_idx/site.
            acc <= (others => '0');
            if n_idx = N_CH-1 then state <= IDLE;
            else n_idx <= n_idx + 1; k_idx <= 0; state <= MAC;
            end if;
        end case;
      end if;
    end if;
  end process;
end architecture;
